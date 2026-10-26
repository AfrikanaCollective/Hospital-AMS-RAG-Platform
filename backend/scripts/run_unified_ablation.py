"""CLI: python -m scripts.run_unified_ablation
[--concepts-path PATH] [--k-values 2,4,...,20]
[--bm25-weight-values 0.0,...,1.0] [--fusion minmax|rrf] [--rrf-k N]
[--lexical-scorer okapi|tfidf] [--area-question-repeats N] [--area-question-arm]
[--enrichment-scope area|record]
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
Every arm is also reported against the question-blind popularity baseline
at every K (`arm_vs_popularity_baseline.csv`, `scripts.report_popularity_baseline`,
DEVIATIONS.md #278).

Since DEVIATIONS.md #311 the run also has an `area_question` arm (the area's
question with no patient details) and an Okapi BM25 lexical scorer
(`okapi_bm25_sapbert`) beside the original TF x IDF one. The Level 1/2/3
statistics and the popularity baseline stay on the original arms
(`summary.core_rows`); the new arms are compared against them in
`added_arms_comparisons.csv` and `statistical_summary.json["added_arms"]`,
and drawn in the figures (the Okapi figures carry an `_okapi_bm25` suffix).

**Since DEVIATIONS.md #312** a run uses one lexical scorer, Okapi BM25 by
default (`--lexical-scorer tfidf` / `ABLATION_LEXICAL_SCORER=tfidf` for the
original TF x IDF one); four-area queries carry their area question
`--area-question-repeats` times (default 3, `ABLATION_AREA_QUESTION_REPEATS`;
1 = the pre-#312 queries); the area-question-only arm is opt-in
(`--area-question-arm`). Fusion stays `ABLATION_FUSION` (default `minmax`).

**Since DEVIATIONS.md #313** Level-2 enrichment of a four-area query adds only
concepts on a field the area's question states (`--enrichment-scope area`,
default, `ABLATION_ENRICHMENT_SCOPE`); `record` restores whole-record
enrichment.

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
import csv
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
    area_question_repeats,
    bm25_weight_values,
    enrichment_scope,
    fusion,
    k_values,
    lexical_scorer,
    mrr_k,
    rrf_k,
)
from app.eval.bootstrap import DEFAULT_BOOTSTRAP_SEED
from app.eval.orchestration_ablation.ablation import load_attested_vocabulary
from app.eval.question_gen.areas import load_query_areas
from app.eval.unified_ablation.blend import OKAPI_B, OKAPI_K1
from app.eval.unified_ablation.per_query import (
    PerQueryResult,
    file_sha256,
    run_dir_for,
    write_configuration,
    write_per_query_results,
)
from app.eval.unified_ablation.runner import run_unified_ablation
from app.eval.unified_ablation.summary import (
    AddedArmComparison,
    Level1Summary,
    Level2Summary,
    Level3Curve,
    PopularityBaseline,
    core_rows,
    summarize_added_arms,
    summarize_best_weight_vs_bm25,
    summarize_level1,
    summarize_level2,
    summarize_level3_by_weight_and_k,
    summarize_level3_curve,
    summarize_popularity_baseline,
)
from app.retrieval.vectorstore import QdrantVectorStore
from scripts.report_popularity_baseline import write_popularity_report

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
    statement-count limit (PLR0915), not a reusable abstraction elsewhere.
    `rows` is the whole run; the Level 1/2/3 statistics and the popularity
    baseline use the original arms only (DEVIATIONS.md #311)."""
    added = summarize_added_arms(rows, k=k, weight_values=weights)
    rows = core_rows(rows, lexical_scorer())
    l1 = summarize_level1(rows, k=k)
    l2 = summarize_level2(rows, k=k)
    l3 = summarize_level3_curve(rows, k=k, weight_values=weights)
    l3_grid = summarize_level3_by_weight_and_k(rows, k_values=ks, weight_values=weights)
    best_vs_bm25 = summarize_best_weight_vs_bm25(rows, k=k, weight_values=weights)
    popularity = summarize_popularity_baseline(rows, k=k, weight_values=weights)

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

    _print_popularity_baseline(popularity)

    statistical_summary = {
        "k": k,
        "level1": asdict(l1),
        "level2": {level1: asdict(s) for level1, s in l2.items()},
        "level3_curve": [asdict(c) for c in l3],
        "level3_by_weight_and_k": [asdict(p) for p in l3_grid],
        "best_weight_vs_bm25": asdict(best_vs_bm25),
        "popularity_baseline": asdict(popularity),
        "added_arms": [asdict(c) for c in added],
    }
    _write_added_arms(added, run_dir)
    summary_path = run_dir / "statistical_summary.json"
    summary_path.write_text(json.dumps(statistical_summary, indent=2), encoding="utf-8")
    print(f"[unified-ablation] wrote {summary_path}")
    return l1, l2, l3


ADDED_ARMS_CSV = "added_arms_comparisons.csv"


def _write_added_arms(added: list[AddedArmComparison], run_dir: Path) -> None:
    """`added_arms_comparisons.csv` (one row per comparison x metric x
    weight) and a line per headline comparison at bm25_weight=1.0
    (DEVIATIONS.md #311)."""
    if not added:
        return
    path = run_dir / ADDED_ARMS_CSV
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "comparison", "metric", "k", "bm25_weight", "arm", "baseline_arm", "n",
                "arm_mean", "baseline_mean", "delta", "ci_low", "ci_high", "p_value",
            ]
        )  # fmt: skip
        for c in added:
            d = c.delta
            writer.writerow(
                [
                    c.comparison, c.metric, c.k, c.bm25_weight,
                    f"{c.level1}/{c.level2}/{c.level3}",
                    f"{c.baseline_level1}/{c.level2}/{c.baseline_level3}",
                    d.n, c.arm.mean, c.baseline.mean, d.mean_delta, d.ci_low, d.ci_high,
                    d.p_value,
                ]
            )  # fmt: skip
    for c in added:
        if c.bm25_weight == 1.0 and c.level2 == "raw":
            d = c.delta
            print(
                f"[unified-ablation] {c.comparison}: {d.label_a} - {d.label_b} "
                f"{c.metric}@{c.k} {d.mean_delta:+.3f} [{d.ci_low:+.3f}, {d.ci_high:+.3f}], "
                f"p={d.p_value:.4f}"
            )
    print(f"[unified-ablation] wrote {path}")


def _print_popularity_baseline(popularity: PopularityBaseline) -> None:
    """One line for the baseline, one per metric for how many arms beat it
    (DEVIATIONS.md #272)."""
    k = popularity.k
    b = popularity.baseline
    print(
        f"[unified-ablation] Popularity baseline (query-blind, leave-one-out) @{k}: "
        f"Recall {b['recall'].mean:.3f}, MRR {b['mrr'].mean:.3f}, Hit {b['hit'].mean:.3f}"
    )
    for metric, deltas in popularity.arm_vs_baseline.items():
        above = sum(d.ci_low > 0 for d in deltas)
        below = sum(d.ci_high < 0 for d in deltas)
        best = max(deltas, key=lambda d: d.mean_delta)
        print(
            f"[unified-ablation]   {metric}@{k}: {above}/{len(deltas)} arm(s) significantly "
            f"above the baseline, {below} below; best {best.label_a} "
            f"{best.mean_delta:+.3f} [{best.ci_low:+.3f}, {best.ci_high:+.3f}]"
        )


def _build_parser() -> argparse.ArgumentParser:
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
        "--lexical-scorer",
        choices=("okapi", "tfidf"),
        default=None,
        help="lexical channel (default: ABLATION_LEXICAL_SCORER, okapi; DEVIATIONS.md #312)",
    )
    parser.add_argument(
        "--area-question-repeats",
        type=str,
        default=None,
        help="times a four-area query carries its area question "
        "(default: ABLATION_AREA_QUESTION_REPEATS, 3; 1 = pre-#312 queries)",
    )
    parser.add_argument(
        "--area-question-arm",
        action="store_true",
        help="also run the area-question-only arm (DEVIATIONS.md #311)",
    )
    parser.add_argument(
        "--enrichment-scope",
        choices=("area", "record"),
        default=None,
        help="concepts Level-2 enrichment may add to a four-area query: those on the "
        "area's own fields, or any the record matches "
        "(default: ABLATION_ENRICHMENT_SCOPE, area; DEVIATIONS.md #313)",
    )
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
    return parser


def _apply_overrides(args: argparse.Namespace) -> None:
    """CLI values override the settings-driven grid via the same env vars
    `app.config.Settings` reads (one source of truth)."""
    _parse_values(args.k_values, env_var="ABLATION_K_VALUES")
    _parse_values(args.bm25_weight_values, env_var="ABLATION_BM25_WEIGHT_VALUES")
    _parse_values(args.fusion, env_var="ABLATION_FUSION")
    _parse_values(args.rrf_k, env_var="ABLATION_RRF_K")
    _parse_values(args.lexical_scorer, env_var="ABLATION_LEXICAL_SCORER")
    _parse_values(args.area_question_repeats, env_var="ABLATION_AREA_QUESTION_REPEATS")
    _parse_values(args.enrichment_scope, env_var="ABLATION_ENRICHMENT_SCOPE")
    overrides = (
        args.k_values,
        args.bm25_weight_values,
        args.fusion,
        args.rrf_k,
        args.lexical_scorer,
        args.area_question_repeats,
        args.enrichment_scope,
    )
    if any(v is not None for v in overrides):
        get_settings.cache_clear()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _apply_overrides(args)

    settings = get_settings()
    fusion()  # an invalid ABLATION_FUSION fails here, before hours of encoding
    lexical_scorer()  # likewise ABLATION_LEXICAL_SCORER / _AREA_QUESTION_REPEATS (#312)
    area_question_repeats()
    enrichment_scope()  # and ABLATION_ENRICHMENT_SCOPE (#313)
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
                area_question_arm=args.area_question_arm,
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
            # DEVIATIONS.md #311: level3_condition names the lexical scorer.
            # DEVIATIONS.md #312: one lexical scorer per run, named by level3_condition.
            "lexical_scorer": lexical_scorer(),
            "lexical_scorer_detail": (
                f"Okapi BM25, k1={OKAPI_K1}, b={OKAPI_B}, statistics over the retrievable corpus"
                if lexical_scorer() == "okapi_bm25_sapbert"
                else "Qdrant sparse search, TF x IDF (no length normalisation)"
            ),
            "area_question_repeats": area_question_repeats(),
            "area_question_weighting": "SapBERT encodes the area question repeated; the "
            "lexical query weights area-question terms by the same factor",
            "area_question_arm": args.area_question_arm,
            "enrichment_scope": enrichment_scope(),
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
    # Every arm vs. the question-blind baseline at every K (DEVIATIONS.md #278).
    report = write_popularity_report(
        run_dir,
        core_rows(rows, lexical_scorer()),
        k_values=k_values(),
        weights=bm25_weight_values(),
    )
    print(f"[unified-ablation] wrote {report}")

    if args.no_figures:
        return 0
    # Imported here so `--no-figures` runs without the plotting libraries.
    # The results above are already on disk if rendering fails
    # (DEVIATIONS.md #229).
    from scripts.plot_ablation_figures import render_all  # noqa: PLC0415

    del rows  # the figures re-read per_query_results.jsonl; free the rows first
    for path in render_all(run_dir, weights=bm25_weight_values()):
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
