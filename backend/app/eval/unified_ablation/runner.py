"""The Level-1 x Level-2 x bm25_weight x K sweep (UNIFIED-ABLATION-PROPOSAL.md
§3, §6, §12; PRD-112 / ARCH-043).

**Restructured 2026-09-23** (operator request, DEVIATIONS.md #201): Level 3
is now a single continuous BM25/SapBERT weighted-rank-fusion sweep — MedCPT
and RRF fusion (proposal §11, Option B) are dropped entirely, not merely
unused. `_QueryChannelScores` no longer carries a MedCPT channel;
`get_medcpt_encoders`/`combine_dense_scores`/`rrf_combine_scores` are gone.

Reuses, unchanged: `model_ablation.ablation.fetch_corpus`/`Corpus`,
`model_ablation.encoders.get_sapbert_encoder`,
`retrieval_tuning.sweep.SweepQuestion`/`fetch_calibration_questions`,
`question_gen.deterministic.build_deterministic_narrative`/
`build_present_only_narrative` (Level 1), `orchestration_ablation.ablation
.load_attested_vocabulary`, `orchestration_ablation.augment
.build_arm_c_query`/`load_synthetic_record_index`/`resolve_source_record`
(Level 2), `unified_ablation.blend` (Level 3), `app.eval.metrics.mrr`/
`precision_recall_at_k`/`hit_at_k`.

One real BM25 query + one SapBERT encode per (query, Level-1, Level-2)
combination — 4 combinations per query — shared across every
`bm25_weight`/`k` row derived from them, not recomputed per row.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import select

from app.db.models.eval import EvalQuestion
from app.eval.ablation_config import ALL_ARMS as _ALL_ARMS
from app.eval.ablation_config import bm25_weight_values, fusion, k_values, rrf_k
from app.eval.auto_seed import _TOPIC, FOUR_AREA_TEMPLATE
from app.eval.deidentified_source import load_deidentified_records
from app.eval.metrics import hit_at_k, mrr, precision_recall_at_k
from app.eval.model_ablation.ablation import Corpus, _l2_normalize_rows, fetch_corpus
from app.eval.model_ablation.encoders import Encoder, get_sapbert_encoder
from app.eval.orchestration_ablation.augment import (
    build_arm_c_query,
    load_synthetic_record_index,
    resolve_source_record,
)
from app.eval.question_gen.areas import QueryAreas, build_area_question
from app.eval.question_gen.deterministic import (
    build_deterministic_narrative,
    build_present_only_narrative,
    extract_topic,
)
from app.eval.retrieval_tuning.sweep import SweepQuestion, fetch_calibration_questions
from app.eval.unified_ablation.blend import (
    blend_bm25_dense,
    blend_bm25_dense_rrf,
    bm25_raw_scores,
    cosine_raw_scores,
)
from app.eval.unified_ablation.per_query import PerQueryResult
from app.schemas.enums import ExpectedOutcome, Provenance

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.records.concepts import ConceptVocabulary
    from app.retrieval.vectorstore import QdrantVectorStore

_LEVEL1_BUILDERS = {
    "present_only": build_present_only_narrative,
    "all_assessed": build_deterministic_narrative,
}


def fetch_four_area_questions(session: Session) -> list[SweepQuestion]:
    """The four-area calibration pool (DEVIATIONS.md #264): `four-area-v2`
    questions (one area per record), `well_supported`, auto-generated, with a
    non-empty gold set, each carrying its area. The 2026-10-05 trial rows
    (`four-area-v1`, four areas per record) are never selected."""
    rows = (
        session.execute(
            select(EvalQuestion).where(
                EvalQuestion.expected_outcome == ExpectedOutcome.WELL_SUPPORTED,
                EvalQuestion.provenance == Provenance.AUTO_GENERATED,
                EvalQuestion.generator_meta["template_version"].astext == FOUR_AREA_TEMPLATE,
            )
        )
        .scalars()
        .all()
    )
    return [
        SweepQuestion(
            question_id=str(row.id),
            text=row.text,
            gold_chunk_ids=frozenset(row.gold_relevant_chunks),
            source_record_id=row.source_record_id,
            area=(row.generator_meta or {}).get("area"),
        )
        for row in rows
        if row.gold_relevant_chunks
    ]


def _any_four_area_question_ids(session: Session) -> set[str]:
    """Ids of every four-area question (`four-area-v1` trial and later), kept
    out of the single-topic pool."""
    stmt = select(EvalQuestion.id).where(
        EvalQuestion.generator_meta["template_version"].astext.like("four-area-%")
    )
    return {str(i) for i in session.execute(stmt).scalars()}


def _first_relevant_rank(retrieved: list[str], gold: set[str]) -> int | None:
    """The actual 1-indexed rank of the first gold hit in the FULL
    (untruncated) ranking — requirement IX's own `first_relevant_rank`
    field, independent of any one k."""
    for i, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold:
            return i
    return None


def _load_record_index(session: Session, records_dir: str | None) -> dict[uuid.UUID, dict]:
    """De-identified first, synthetic fallback — the same combined
    resolution `app.eval.auto_seed._existing_accepted_records` already
    established (DEVIATIONS.md #188), reused here since a calibration
    question's `source_record_id` can come from either pipeline. Imports
    are module-level here (unlike `app.eval.auto_seed`'s own lazy imports of
    the same functions) — `app.eval.tasks` imports `auto_seed.py` at ITS OWN
    module level, which is what forces `auto_seed.py` to import these
    lazily; nothing imports `unified_ablation` from `app.eval.tasks`, so
    this module isn't part of that cycle (confirmed live, not assumed)."""
    deidentified = dict(load_deidentified_records(session, dataset_id=None))
    synthetic = load_synthetic_record_index(records_dir)
    return {**synthetic, **deidentified}


class _QueryChannelScores:
    """BM25/SapBERT raw scores for one already-built query text — computed
    once, reused across every `bm25_weight` swept for this (query, Level-1,
    Level-2) combination's text."""

    __slots__ = ("bm25", "sapbert")

    def __init__(self, bm25: dict[str, float], sapbert: dict[str, float]):
        self.bm25 = bm25
        self.sapbert = sapbert


def sweep_questions(
    questions: list[SweepQuestion],
    store: QdrantVectorStore,
    corpus: Corpus,
    *,
    sapbert_encoder: Encoder,
    sapbert_chunk_matrix: np.ndarray,
    record_index: dict[uuid.UUID, dict],
    vocabulary: ConceptVocabulary | None,
    experiment_id: str,
    k_grid: tuple[int, ...],
    fusion_method: str = "minmax",
    rrf_damping: int = 60,
    query_areas: QueryAreas | None = None,
) -> Iterator[PerQueryResult]:
    """Pure sweep over already-fetched `questions`/`corpus` — no I/O beyond
    the encoders/`store` callers already hold open, no Postgres session.
    Mirrors `model_ablation.ablation.run_ablation`/`retrieval_tuning.sweep
    .run_sweep`'s own split: fast, offline-testable (`:memory:` Qdrant +
    stub encoders), separate from the thin DB-touching orchestrator
    (`run_unified_ablation`) below.

    `vocabulary` is `None` when `data/clinical_concepts.yaml` isn't
    attested (`orchestration_ablation.ablation.load_attested_vocabulary`,
    reused unchanged) — every Level-2-enriched row then falls back to the
    Level-1 text unchanged, same convention `app.eval.auto_seed`/
    `orchestration_ablation` already use, never a reason to fail the whole
    run.

    A question whose own source record can't be resolved in `record_index`
    (`resolve_source_record` returns `None` — no `source_record_id` at all,
    or one present in neither the de-identified nor the synthetic index) is
    skipped entirely, not evaluated with a guessed/missing record: Level 1's
    whole point is deterministically rebuilding the narrative from the
    record's own fields, so there is nothing to build from without one.

    A four-area question (`question.area` set, DEVIATIONS.md #264) is rebuilt
    with its own area's opening and record facts
    (`question_gen.areas.build_area_question`): `all_assessed` keeps
    assessed-absent findings, `present_only` drops them. An area with no
    findings in its facts (antibiotic course) gives identical text for both.
    A four-area question whose area isn't in `query_areas` is skipped."""
    areas_by_name = {a.name: a for a in query_areas.areas} if query_areas else {}
    for question in questions:
        gold = set(question.gold_chunk_ids)
        record = resolve_source_record(question.source_record_id, record_index)
        if record is None:
            continue

        # The topic the question was actually generated (and its gold set
        # produced) with, read back from the stored text -- not the current
        # `_TOPIC`, which changed after the earliest review-queue rows were
        # seeded (DEVIATIONS.md #208). `_TOPIC` only as a fallback for text
        # not built by the deterministic template.
        topic = extract_topic(question.text) or _TOPIC
        area = areas_by_name.get(question.area) if question.area else None
        if question.area and area is None:
            continue

        for level1, builder in _LEVEL1_BUILDERS.items():
            # The SAME topic for both Level-1 branches (DEVIATIONS.md #190/
            # #191), so the only thing that differs between them is
            # present/absent handling, per the comparability requirement
            # (proposal §III).
            record_dict = record.model_dump(mode="json")
            if area is not None:
                base_text = build_area_question(
                    record_dict, area, include_absent=level1 == "all_assessed"
                )
            else:
                base_text = builder(record_dict, topic=topic)

            enriched_text = base_text
            if vocabulary is not None:
                enriched_text = build_arm_c_query(base_text, vocabulary, record).text

            level2_texts = {"raw": base_text, "enriched": enriched_text}

            for level2, query_text in level2_texts.items():
                bm25 = bm25_raw_scores(store, query_text, corpus.chunk_ids)
                sapbert_qvec = sapbert_encoder.encode([query_text])[0]
                scores = _QueryChannelScores(
                    bm25=bm25,
                    sapbert=cosine_raw_scores(sapbert_qvec, sapbert_chunk_matrix, corpus.chunk_ids),
                )

                arm = next(a for a in _ALL_ARMS if a.level1 == level1 and a.level2 == level2)
                for weight in bm25_weight_values():
                    if fusion_method == "rrf":
                        ranking = blend_bm25_dense_rrf(
                            scores.bm25, scores.sapbert, bm25_weight=weight, rrf_k=rrf_damping
                        )
                    else:
                        ranking = blend_bm25_dense(scores.bm25, scores.sapbert, bm25_weight=weight)
                    first_rank = _first_relevant_rank(ranking, gold)
                    for k in k_grid:
                        top_k = ranking[:k]
                        yield PerQueryResult(
                            query_id=question.question_id,
                            patient_id_or_case_id=str(question.source_record_id or ""),
                            experiment_id=experiment_id,
                            level1_condition=level1,
                            level2_condition=level2,
                            level3_condition=arm.level3,
                            k=k,
                            bm25_weight=weight,
                            query_text=query_text,
                            concept_enriched_query=enriched_text,
                            retrieved_ids=top_k,
                            relevant_ids=sorted(gold),
                            first_relevant_rank=first_rank,
                            recall_at_k=precision_recall_at_k(top_k, gold, k)[1],
                            reciprocal_rank_at_k=mrr(top_k, gold),
                            question_area=question.area,
                            hit_at_k=hit_at_k(top_k, gold, k),
                        )


def run_unified_ablation(
    session: Session,
    store: QdrantVectorStore,
    *,
    experiment_id: str | None = None,
    vocabulary: ConceptVocabulary | None,
    records_dir: str | None = None,
    question_set: str = "four_area",
    query_areas: QueryAreas | None = None,
) -> Iterator[PerQueryResult]:
    """Thin orchestrator: fetch questions/corpus/records from the real
    Postgres session + Qdrant store, then delegate to `sweep_questions` (the
    part that's actually testable offline). Yields one `PerQueryResult` per
    (query, level1, level2, bm25_weight, k) row — a generator, not a list,
    since a real run's row count is large and
    `per_query.write_per_query_results` streams them to disk as they're
    produced rather than holding the whole run in memory."""
    exp_id = experiment_id or str(uuid.uuid4())
    # `four_area` (DEVIATIONS.md #264): the per-area pool, rebuilt by area;
    # `single_topic`: the original pool. Never mixed in one run.
    if question_set == "four_area":
        if query_areas is None:
            raise ValueError("question_set='four_area' needs the attested query_areas")
        questions: list[SweepQuestion] = fetch_four_area_questions(session)
    elif question_set == "single_topic":
        four_area_ids = _any_four_area_question_ids(session)
        questions = [
            q for q in fetch_calibration_questions(session) if q.question_id not in four_area_ids
        ]
    else:
        raise ValueError(f"unknown question_set {question_set!r}")
    corpus: Corpus = fetch_corpus(store)

    sapbert_encoder = get_sapbert_encoder()
    sapbert_chunk_matrix = _l2_normalize_rows(
        np.asarray(sapbert_encoder.encode(corpus.texts), dtype=np.float64)
    )

    # Needed unconditionally for Level 1 (rebuilding the narrative from the
    # record's own fields), not only when `vocabulary` is attested.
    record_index = _load_record_index(session, records_dir)

    yield from sweep_questions(
        questions,
        store,
        corpus,
        sapbert_encoder=sapbert_encoder,
        sapbert_chunk_matrix=sapbert_chunk_matrix,
        record_index=record_index,
        vocabulary=vocabulary,
        experiment_id=exp_id,
        k_grid=k_values(),
        fusion_method=fusion(),
        rrf_damping=rrf_k(),
        query_areas=query_areas,
    )
