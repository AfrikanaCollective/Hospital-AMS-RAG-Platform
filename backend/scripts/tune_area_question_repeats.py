"""CLI: python -m scripts.tune_area_question_repeats
[--repeats-grid 1,2,3,4,5,6,8] [--concepts-path PATH] [--areas-path PATH]
[--run-id ID] [--results-root DIR]
(PRD-112 / ARCH-043; DEVIATIONS.md #313).

Chooses `ABLATION_AREA_QUESTION_REPEATS` on a hash-split `tune` half of the
four-area pool and reports the `heldout` half
(`app.eval.unified_ablation.repeats_tuning`). Runs the four patient arms at
K = `ABLATION_MRR_K` over the configured BM25 weights, with the run's
lexical scorer, fusion and enrichment scope (`ABLATION_*`, as
`run_unified_ablation`). The corpus is encoded once and reused for every N.

Writes `results/ablation/repeats_tuning_<run_id>/`: `cells.csv` (mean MRR and
Recall per split x N x arm x weight), `tuning.json` (selection scores, the
chosen N, held-out comparisons, configuration). Changes no default: the
operator sets `ABLATION_AREA_QUESTION_REPEATS` from the result.

Same environment as `run_unified_ablation` (real Postgres + Qdrant,
`MODEL_ABLATION_BACKEND=local`); run it in the `api` container.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.db.session import session_scope
from app.eval.ablation_config import (
    OKAPI_SCORER,
    area_question_repeats,
    bm25_weight_values,
    enrichment_scope,
    fusion,
    lexical_scorer,
    mrr_k,
    rrf_k,
)
from app.eval.model_ablation.ablation import _l2_normalize_rows, fetch_corpus
from app.eval.model_ablation.encoders import get_sapbert_encoder
from app.eval.orchestration_ablation.ablation import load_attested_vocabulary
from app.eval.question_gen.areas import load_query_areas
from app.eval.retrieval_tuning.sweep import SweepQuestion
from app.eval.unified_ablation.blend import fetch_okapi_bm25
from app.eval.unified_ablation.per_query import PerQueryResult, file_sha256
from app.eval.unified_ablation.repeats_tuning import split_of, tune_repeats
from app.eval.unified_ablation.runner import (
    _load_record_index,
    fetch_four_area_questions,
    sweep_questions,
)
from app.retrieval.vectorstore import QdrantVectorStore
from scripts.run_unified_ablation import _DEFAULT_CONCEPTS_PATH, _DEFAULT_RESULTS_ROOT


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune ABLATION_AREA_QUESTION_REPEATS (#313)")
    parser.add_argument("--repeats-grid", type=str, default="1,2,3,4,5,6,8")
    parser.add_argument("--concepts-path", type=Path, default=_DEFAULT_CONCEPTS_PATH)
    parser.add_argument("--areas-path", type=Path, default=Path("data/query_areas.yaml"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--results-root", type=Path, default=_DEFAULT_RESULTS_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    grid = tuple(int(v) for v in args.repeats_grid.split(","))
    settings = get_settings()
    scorer = lexical_scorer()
    scope = enrichment_scope()
    k = mrr_k()
    if settings.model_ablation_backend == "stub":
        print(
            "[repeats-tuning] MODEL_ABLATION_BACKEND=stub — using deterministic fake "
            "embeddings, not real SapBERT. Set MODEL_ABLATION_BACKEND=local for a real run.",
            file=sys.stderr,
        )
    vocabulary, vocab_error = load_attested_vocabulary(args.concepts_path)
    if vocabulary is None:
        print(f"[repeats-tuning] enrichment skipped: {vocab_error}", file=sys.stderr)
    query_areas = load_query_areas(args.areas_path)
    run_id = (
        args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"
    )
    out_dir = args.results_root / f"repeats_tuning_{run_id}"

    store = QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_guideline_collection,
    )
    with session_scope() as session:
        questions = fetch_four_area_questions(session)
        record_index = _load_record_index(session, None)
    if not questions:
        print("[repeats-tuning] no four-area holdout questions found", file=sys.stderr)
        return 1
    corpus = fetch_corpus(store)
    encoder = get_sapbert_encoder()
    matrix = _l2_normalize_rows(np.asarray(encoder.encode(corpus.texts), dtype=np.float64))
    okapi = fetch_okapi_bm25(store, corpus.chunk_ids) if scorer == OKAPI_SCORER else None

    def sweep(qs: list[SweepQuestion], n: int) -> list[PerQueryResult]:
        print(f"[repeats-tuning] N={n}: {len(qs)} questions", file=sys.stderr, flush=True)
        return list(
            sweep_questions(
                qs,
                store,
                corpus,
                sapbert_encoder=encoder,
                sapbert_chunk_matrix=matrix,
                record_index=record_index,
                vocabulary=vocabulary,
                experiment_id=run_id,
                k_grid=(k,),
                fusion_method=fusion(),
                rrf_damping=rrf_k(),
                query_areas=query_areas,
                okapi=okapi,
                scorers=(scorer,),
                area_question_repeats=n,
                area_scoped_enrichment=scope == "area",
            )
        )

    result = tune_repeats(questions, sweep, repeats_grid=grid)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "cells.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = list(asdict(result.cells[0])) if result.cells else []
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(c) for c in result.cells)
    split_counts = {s: sum(1 for q in questions if split_of(q) == s) for s in ("tune", "heldout")}
    payload = {
        "run_id": run_id,
        "chosen_repeats": result.chosen,
        "selection_metric": f"own-gold MRR@{k}, mean over the four patient arms and all "
        "BM25 weights, then over questions; ties to the smaller N",
        "selection_scores": result.selection_scores,
        "heldout_comparisons": [asdict(c) for c in result.heldout_comparisons],
        "configuration": {
            "repeats_grid": list(grid),
            "current_default_repeats": area_question_repeats(),
            "k": k,
            "bm25_weight_values": list(bm25_weight_values()),
            "lexical_scorer": scorer,
            "fusion": fusion(),
            "enrichment_scope": scope,
            "split": "sha256(source_record_id) mod 2",
            "split_counts": split_counts,
            "clinical_concepts_sha256": file_sha256(args.concepts_path),
            "query_areas_sha256": query_areas.sha256,
            "model_ablation_backend": settings.model_ablation_backend,
            "sapbert_model_id": settings.sapbert_model_id,
        },
    }
    (out_dir / "tuning.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[repeats-tuning] split: {split_counts}")
    for split in ("tune", "heldout"):
        scores = ", ".join(f"N={n}: {v:.3f}" for n, v in result.selection_scores[split].items())
        print(f"[repeats-tuning] {split} MRR@{k}: {scores}")
    print(f"[repeats-tuning] chosen on tune: N={result.chosen}")
    for c in result.heldout_comparisons:
        print(
            f"[repeats-tuning] heldout N={c.repeats} - N={c.baseline_repeats}: "
            f"{c.mean_delta:+.3f} [{c.ci_low:+.3f}, {c.ci_high:+.3f}] p={c.p_value:.3f}"
        )
    print(f"[repeats-tuning] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
