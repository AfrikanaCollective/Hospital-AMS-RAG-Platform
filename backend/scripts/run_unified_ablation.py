"""CLI: python -m scripts.run_unified_ablation
[--concepts-path PATH] [--k-values 2,4,...,20]
[--bm25-weight-values 0.0,...,1.0] [--fusion minmax|rrf] [--rrf-k N]
[--no-figures] [--run-id ID] [--results-root DIR]
(PRD-112 / ARCH-043; UNIFIED-ABLATION-PROPOSAL.md §3.9, §12; Makefile
`make unified-ablation-report`).

Runs the Level-1 x Level-2 x bm25_weight x K sweep
(`app.eval.unified_ablation.runner`) — Level 3 is a single continuous
BM25/SapBERT weighted-rank-fusion sweep (MedCPT and RRF fusion dropped
entirely, DEVIATIONS.md #201) — against the real Qdrant guideline
collection and the real `eval.eval_question` table, writes the per-query
results and a reproducibility snapshot to `results/ablation/<run_id>/`
(operator-specified layout, DEVIATIONS.md #192), computes the three
Level 1/2/3 statistical comparisons (`app.eval.unified_ablation.summary`,
primary metric recall@k), persisting them to `statistical_summary.json`
in the same run directory. No combined PNG report is rendered (dropped at
operator request, DEVIATIONS.md #212). The three `scripts/plot_*` figures
(Recall@K by weight, Recall vs weight by K, MRR vs weight by arm) are
rendered into the run directory at the end of every run
(`scripts.plot_ablation_figures`, DEVIATIONS.md #229) unless `--no-figures`.

Needs a real Postgres + Qdrant with an already-ingested guideline corpus and
an already-seeded auto-generated question set — not runnable against
`:memory:`/stub backends, which is what `tests/test_unified_ablation_runner.py`
covers instead (CLAUDE.md §5). Requires the `retrieval-tuning` optional
extra (seaborn/pandas/matplotlib) and, for a real (non-stub) run,
`MODEL_ABLATION_BACKEND=local` plus the `local-models` extra
(sentence-transformers/torch, for SapBERT) — same conventions as
`run_model_ablation.py`.

**Level 2's vocabulary enrichment requires `data/clinical_concepts.yaml` to
be attested**, same convention as `run_model_ablation.py`/
`run_orchestration_ablation.py`. If it isn't, every Level-2 "enriched" row
falls back to the Level-1 text unchanged (`runner.sweep_questions`'s own
documented behavior) — this script still runs and prints why, rather than
failing or running Arm-C-equivalent logic against a placeholder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.db.session import session_scope
from app.eval.ablation_config import (
    Level1Condition,
    bm25_weight_values,
    fusion,
    k_values,
    mrr_k,
    rrf_k,
)
from app.eval.bootstrap import DEFAULT_BOOTSTRAP_SEED
from app.eval.orchestration_ablation.ablation import load_attested_vocabulary
from app.eval.question_gen.areas import load_query_areas
from app.eval.unified_ablation.per_query import (
    PerQueryResult,
    file_sha256,
    run_dir_for,
    write_configuration,
    write_per_query_results,
)
from app.eval.unified_ablation.runner import run_unified_ablation
from app.eval.unified_ablation.summary import (
    Level1Summary,
    Level2Summary,
    Level3Curve,
    summarize_best_weight_vs_bm25,
    summarize_level1,
    summarize_level2,
    summarize_level3_by_weight_and_k,
    summarize_level3_curve,
)
from app.retrieval.vectorstore import QdrantVectorStore

_DEFAULT_CONCEPTS_PATH = Path(
    os.environ.get("CLINICAL_CONCEPTS_PATH", "data/clinical_concepts.yaml")
)
# CWD-relative, matching `run_orchestration_ablation.py`'s own
# `_DEFAULT_OUT_DIR` convention (DEVIATIONS.md #150) — the real invocation
# environment is `docker compose exec api ...` with WORKDIR=/app.
_DEFAULT_RESULTS_ROOT = Path(os.environ.get("UNIFIED_ABLATION_RESULTS_ROOT", "results/ablation"))


def _parse_values(raw: str | None, *, env_var: str) -> None:
    """Overrides the settings-driven K/bm25_weight grid for this process
    only, via the same env var `app.config.Settings` already reads — not a
    second, competing source of truth (proposal §3.6: "never hardcoded")."""
    if raw is not None:
        os.environ[env_var] = raw


def _report_statistics(
    rows: list[PerQueryResult],
    *,
    k: int,
    weights: tuple[float, ...],
    ks: tuple[int, ...],
    run_dir: Path,
) -> tuple[Level1Summary, dict[Level1Condition, Level2Summary], list[Level3Curve]]:
    """Computes, prints, and persists the Level 1/2/3 statistical comparisons
    (DEVIATIONS.md #202) -- split out of `main()` to keep it under ruff's
    statement-count limit (PLR0915), not a reusable abstraction elsewhere."""
    l1 = summarize_level1(rows, k=k)
    l2 = summarize_level2(rows, k=k)
    l3 = summarize_level3_curve(rows, k=k, weight_values=weights)
    l3_grid = summarize_level3_by_weight_and_k(rows, k_values=ks, weight_values=weights)
    best_vs_bm25 = summarize_best_weight_vs_bm25(rows, k=k, weight_values=weights)

    print(
        f"[unified-ablation] Level 1 delta (present-only - all-assessed) Recall@{k}: "
        f"{l1.delta.mean_delta:+.3f} [{l1.delta.ci_low:+.3f}, {l1.delta.ci_high:+.3f}], "
        f"p={l1.delta.p_value:.4f}"
    )
    for level1, l2_summary in l2.items():
        d = l2_summary.delta
        print(
            f"[unified-ablation] Level 2 delta ({level1}, enriched - raw) Recall@{k}: "
            f"{d.mean_delta:+.3f} [{d.ci_low:+.3f}, {d.ci_high:+.3f}], p={d.p_value:.4f}"
        )
    for curve in l3:
        d = curve.endpoints_delta
        print(
            f"[unified-ablation] Level 3 delta ({curve.level1}/{curve.level2}, "
            f"bm25_weight=1.0 - bm25_weight=0.0) Recall@{k}: {d.mean_delta:+.3f} "
            f"[{d.ci_low:+.3f}, {d.ci_high:+.3f}], p={d.p_value:.4f}"
        )
    print(
        f"[unified-ablation] Level 3 full grid: Recall@k computed for {len(weights)} "
        f"bm25_weight value(s) x {len(ks)} k value(s) = {len(l3_grid)} points "
        f"(pooled across Level 1 x Level 2) -- see statistical_summary.json"
    )
    print(
        f"[unified-ablation] Does BM25 weighting help at all? best bm25_weight="
        f"{best_vs_bm25.selected_weight} (selected AFTER seeing the data) vs. "
        f"bm25_weight=1.0 (pure BM25), Recall@{k}: "
        f"{best_vs_bm25.delta.mean_delta:+.3f} [{best_vs_bm25.delta.ci_low:+.3f}, "
        f"{best_vs_bm25.delta.ci_high:+.3f}], p={best_vs_bm25.delta.p_value:.4f} "
        f"-- POST-HOC: the weight was picked as the max of "
        f"{len(weights)} empirical means, so this CI/p-value understates the true "
        f"uncertainty of 'does the best of these weights beat BM25' (multiple-comparisons "
        f"/ winner's-curse caveat -- see BestWeightVsBM25's own docstring)."
    )

    statistical_summary = {
        "k": k,
        "level1": asdict(l1),
        "level2": {level1: asdict(s) for level1, s in l2.items()},
        "level3_curve": [asdict(c) for c in l3],
        "level3_by_weight_and_k": [asdict(p) for p in l3_grid],
        "best_weight_vs_bm25": asdict(best_vs_bm25),
    }
    summary_path = run_dir / "statistical_summary.json"
    summary_path.write_text(json.dumps(statistical_summary, indent=2), encoding="utf-8")
    print(f"[unified-ablation] wrote {summary_path}")
    return l1, l2, l3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified hierarchical (Level 1 x 2 x 3) ablation (PRD-112/ARCH-043)"
    )
    parser.add_argument("--concepts-path", type=Path, default=_DEFAULT_CONCEPTS_PATH)
    parser.add_argument("--k-values", type=str, default=None, help="comma-separated, e.g. 2,4,6")
    parser.add_argument(
        "--bm25-weight-values", type=str, default=None, help="comma-separated, e.g. 0.0,0.5,1.0"
    )
    parser.add_argument(
        "--fusion",
        choices=["minmax", "rrf"],
        default=None,
        help="how the BM25/SapBERT sweep combines channels (default: ABLATION_FUSION)",
    )
    parser.add_argument("--rrf-k", type=str, default=None, help="RRF damping constant")
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="skip the three PNG figures (written into the run directory by default)",
    )
    parser.add_argument(
        "--question-set",
        choices=("four_area", "single_topic"),
        default="four_area",
        help="four_area (DEVIATIONS.md #264): the per-area pool, rebuilt by area; "
        "single_topic: the original pool",
    )
    parser.add_argument("--areas-path", type=Path, default=Path("data/query_areas.yaml"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--results-root", type=Path, default=_DEFAULT_RESULTS_ROOT)
    args = parser.parse_args(argv)

    _parse_values(args.k_values, env_var="ABLATION_K_VALUES")
    _parse_values(args.bm25_weight_values, env_var="ABLATION_BM25_WEIGHT_VALUES")
    _parse_values(args.fusion, env_var="ABLATION_FUSION")
    _parse_values(args.rrf_k, env_var="ABLATION_RRF_K")
    if any(
        v is not None for v in (args.k_values, args.bm25_weight_values, args.fusion, args.rrf_k)
    ):
        get_settings.cache_clear()

    settings = get_settings()
    fusion()  # an invalid ABLATION_FUSION fails here, before hours of encoding
    if settings.model_ablation_backend == "stub":
        print(
            "[unified-ablation] MODEL_ABLATION_BACKEND=stub — using deterministic fake "
            "embeddings, not real SapBERT. Set MODEL_ABLATION_BACKEND=local for a real run.",
            file=sys.stderr,
        )

    vocabulary, vocab_error = load_attested_vocabulary(args.concepts_path)
    # Hashed at load time, not at write time: the file can be edited during a
    # long run, and the hash must identify the version actually loaded
    # (DEVIATIONS.md #211).
    concepts_sha256 = file_sha256(args.concepts_path)
    if vocabulary is None:
        print(
            f"[unified-ablation] Level 2 enrichment skipped — {args.concepts_path} is not "
            f"attested: {vocab_error}. Every 'enriched' row will match its 'raw' counterpart.",
            file=sys.stderr,
        )

    # Fails closed if the areas file isn't attested (DEVIATIONS.md #258).
    query_areas = load_query_areas(args.areas_path) if args.question_set == "four_area" else None
    query_areas_sha256 = query_areas.sha256 if query_areas else None

    run_id = (
        args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"
    )
    run_dir = run_dir_for(run_id, results_root=args.results_root)

    store = QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_guideline_collection,
    )

    experiment_id = str(uuid.uuid4())
    with session_scope() as session:
        rows = list(
            run_unified_ablation(
                session,
                store,
                experiment_id=experiment_id,
                vocabulary=vocabulary,
                question_set=args.question_set,
                query_areas=query_areas,
            )
        )
        generated_with = _check_area_hashes(session, query_areas_sha256)

    if not rows:
        print(
            "[unified-ablation] no well_supported auto_generated questions with a "
            f"gold_relevant_chunks set (and a resolvable source record) found for "
            f"--question-set {args.question_set} — generate them first "
            "(python -m scripts.generate_ablation_holdout_questions).",
            file=sys.stderr,
        )
        return 1

    print(
        f"[unified-ablation] {len(rows)} per-query-result row(s) across {len(k_values())} "
        f"k value(s) and {len(bm25_weight_values())} bm25_weight value(s)..."
    )

    write_per_query_results(run_dir, rows)
    write_configuration(
        run_dir,
        {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "seed": DEFAULT_BOOTSTRAP_SEED,
            "k_values": list(k_values()),
            "bm25_weight_values": list(bm25_weight_values()),
            "mrr_k": mrr_k(),
            "fusion": fusion(),
            "rrf_k": rrf_k() if fusion() == "rrf" else None,
            "primary_metric": "recall_at_k",
            "sapbert_model_id": settings.sapbert_model_id,
            "sapbert_model_verified": settings.sapbert_model_verified,
            "retrieval_depth": "full_corpus_brute_force",  # no ANN/candidate-depth truncation
            "question_set": args.question_set,
            "template_version": (
                {
                    "question": "four-area-v2 (one area per record)",
                    "all_assessed": "area builder, include_absent=True",
                    "present_only": "area builder, include_absent=False",
                }
                if args.question_set == "four_area"
                else {
                    "all_assessed": "deterministic-v1",
                    "present_only": "deterministic-present-only-v1",
                }
            ),
            "query_areas_path": str(args.areas_path) if query_areas else None,
            "query_areas_sha256": query_areas_sha256,
            "query_areas_sha256_at_generation": generated_with or None,
            "questions_per_area": _questions_per_area(rows) if query_areas else None,
            "vocabulary_attested": vocabulary is not None,
            "concepts_path": str(args.concepts_path),
            "concepts_sha256": concepts_sha256,
            "n_rows": len(rows),
            "distinct_queries_evaluated": len({r.query_id for r in rows}),
            "distinct_records_used": len({r.patient_id_or_case_id for r in rows}),
        },
    )
    print(f"[unified-ablation] wrote {run_dir / 'configuration.json'}")
    print(f"[unified-ablation] wrote {run_dir / 'per_query_results.jsonl'}")

    k = mrr_k()
    _report_statistics(rows, k=k, weights=bm25_weight_values(), ks=k_values(), run_dir=run_dir)

    if args.no_figures:
        return 0
    # Imported here so `--no-figures` runs without the plotting libraries.
    # The results above are already on disk if rendering fails
    # (DEVIATIONS.md #229).
    from scripts.plot_ablation_figures import render_all  # noqa: PLC0415

    del rows  # the figures re-read per_query_results.jsonl; free the rows first
    for path in render_all(run_dir, weights=bm25_weight_values(), mrr_k=k):
        print(f"[unified-ablation] wrote {path}")
    return 0


def _questions_per_area(rows: list) -> dict[str, int]:
    seen: dict[str, set[str]] = {}
    for r in rows:
        seen.setdefault(r.question_area or "none", set()).add(r.query_id)
    return {area: len(ids) for area, ids in sorted(seen.items())}


def _check_area_hashes(session, current: str | None) -> list[str]:  # noqa: ANN001
    """The query_areas.yaml hashes the four-area questions were generated
    with; warns when the file used for this run's rebuild differs."""
    if current is None:
        return []
    generated_with = _generation_area_hashes(session)
    if generated_with and generated_with != [current]:
        print(
            "[unified-ablation] WARNING: questions were generated with query_areas.yaml "
            f"sha256 {generated_with}, but this run rebuilds them with {current}; "
            "rebuilt question text may differ from the text their gold sets came from.",
            file=sys.stderr,
        )
    return generated_with


def _generation_area_hashes(session) -> list[str]:  # noqa: ANN001
    """Distinct query_areas.yaml hashes the four-area questions were generated with."""
    from sqlalchemy import select  # noqa: PLC0415

    from app.db.models.eval import EvalQuestion  # noqa: PLC0415
    from app.eval.auto_seed import FOUR_AREA_TEMPLATE  # noqa: PLC0415

    stmt = select(EvalQuestion.generator_meta["query_areas_sha256"].astext).where(
        EvalQuestion.generator_meta["template_version"].astext == FOUR_AREA_TEMPLATE
    )
    return sorted({h for h in session.execute(stmt).scalars() if h})


if __name__ == "__main__":
    sys.exit(main())
