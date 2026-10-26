"""The Level-1 x Level-2 x bm25_weight x K sweep (UNIFIED-ABLATION-PROPOSAL.md
§3, §6, §12; PRD-112 / ARCH-043).

**Restructured 2026-09-23** (operator request, DEVIATIONS.md #201): Level 3
is now a single continuous BM25/SapBERT weighted-rank-fusion sweep — MedCPT
and RRF fusion (proposal §11, Option B) are dropped entirely, not merely
unused. `_QueryChannelScores` (removed in DEVIATIONS.md #311) no longer carried a MedCPT channel;
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

**Changed 2026-10-21 (DEVIATIONS.md #312):** one lexical scorer per run,
Okapi BM25 by default (`ABLATION_LEXICAL_SCORER=tfidf` for the original);
four-area queries carry the area question `ABLATION_AREA_QUESTION_REPEATS`
times (default 3); the `area_question` arm is opt-in.

**Changed 2026-10-26 (DEVIATIONS.md #313):** Level-2 enrichment of a
four-area query adds only concepts on a field the area's question states
(`ABLATION_ENRICHMENT_SCOPE=area`, default; `record` for the earlier
whole-record enrichment).

**Extended 2026-10-21 (DEVIATIONS.md #311):** a third Level-1 condition,
`area_question` (the area's question alone, no patient details; four-area
questions only, `raw` only), and a second Level-3 scorer,
`okapi_bm25_sapbert` (Okapi BM25 for the lexical channel instead of the
collection's TF x IDF sparse search), when an `OkapiBM25` index is passed.
Each query text is still encoded once; the Okapi scores are computed in
memory from the same indexed term counts.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import select

from app.db.models.eval import EvalQuestion
from app.eval.ablation_config import ALL_ARMS as _ALL_ARMS
from app.eval.ablation_config import (
    OKAPI_SCORER,
    TFIDF_SCORER,
    Level3Condition,
    area_question_repeats,
    bm25_weight_values,
    enrichment_scope,
    fusion,
    k_values,
    lexical_scorer,
    rrf_k,
)
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
from app.eval.question_gen.areas import (
    QueryArea,
    QueryAreas,
    area_vocabulary,
    build_area_question,
    build_area_question_only,
)
from app.eval.question_gen.deterministic import (
    build_deterministic_narrative,
    build_present_only_narrative,
    extract_topic,
)
from app.eval.retrieval_tuning.sweep import SweepQuestion, fetch_calibration_questions
from app.eval.unified_ablation.blend import (
    OkapiBM25,
    blend_bm25_dense,
    blend_bm25_dense_rrf,
    bm25_raw_scores,
    cosine_raw_scores,
    fetch_okapi_bm25,
    lexical_query_vector,
)
from app.eval.unified_ablation.per_query import PerQueryResult
from app.schemas.enums import ExpectedOutcome, Provenance

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.records.concepts import ConceptVocabulary
    from app.retrieval.vectorstore import QdrantVectorStore
    from app.schemas.record import PatientRecord

_LEVEL1_BUILDERS = {
    "present_only": build_present_only_narrative,
    "all_assessed": build_deterministic_narrative,
}


def fetch_four_area_questions(session: Session) -> list[SweepQuestion]:
    """The four-area calibration pool (DEVIATIONS.md #264): `four-area-v2`
    holdout questions (one area per record), `well_supported`, auto-generated,
    with a non-empty gold set, each carrying its area. Review-queue items
    built from the same template (DEVIATIONS.md #280) are not selected. The 2026-10-05 trial rows
    (`four-area-v1`, four areas per record) are never selected."""
    rows = (
        session.execute(
            select(EvalQuestion).where(
                EvalQuestion.expected_outcome == ExpectedOutcome.WELL_SUPPORTED,
                EvalQuestion.provenance == Provenance.AUTO_GENERATED,
                EvalQuestion.generator_meta["template_version"].astext == FOUR_AREA_TEMPLATE,
                # Four-area review-queue items share the template (#280).
                EvalQuestion.generator_meta["purpose"].astext == "ablation_holdout",
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


def _query_texts(  # noqa: PLR0913
    record: PatientRecord,
    area: QueryArea | None,
    topic: str,
    vocabulary: ConceptVocabulary | None,
    *,
    question_repeats: int = 1,
    area_question_arm: bool = False,
    area_scoped_enrichment: bool = False,
) -> list[tuple[str, str, str, str]]:
    """(level1, level2, query text, enriched text) for every Level-1 x
    Level-2 text of one question; a four-area question carries its area
    question `question_repeats` times (DEVIATIONS.md #312), and with
    `area_question_arm` also gets `area_question/raw` (#311). With
    `area_scoped_enrichment` a four-area question is enriched only from
    concepts on its area's fields (#313)."""
    texts: list[tuple[str, str, str, str]] = []
    record_dict = record.model_dump(mode="json")
    if vocabulary is not None and area is not None and area_scoped_enrichment:
        vocabulary = area_vocabulary(vocabulary, area)
    for level1, builder in _LEVEL1_BUILDERS.items():
        # The SAME topic for both Level-1 branches (DEVIATIONS.md #190/
        # #191), so the only thing that differs between them is
        # present/absent handling, per the comparability requirement
        # (proposal §III).
        if area is not None:
            base_text = build_area_question(
                record_dict,
                area,
                include_absent=level1 == "all_assessed",
                question_repeats=question_repeats,
            )
        else:
            base_text = builder(record_dict, topic=topic)
        enriched_text = base_text
        if vocabulary is not None:
            enriched_text = build_arm_c_query(base_text, vocabulary, record).text
        texts.append((level1, "raw", base_text, enriched_text))
        texts.append((level1, "enriched", enriched_text, enriched_text))
    if area is not None and area_question_arm:
        # No patient details, so nothing to enrich from the record (#311).
        stem = build_area_question_only(area)
        texts.append(("area_question", "raw", stem, stem))
    return texts


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
    okapi: OkapiBM25 | None = None,
    scorers: tuple[Level3Condition, ...] = (TFIDF_SCORER,),
    area_question_repeats: int = 1,
    area_question_arm: bool = False,
    area_scoped_enrichment: bool = False,
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
    A four-area question whose area isn't in `query_areas` is skipped.

    `scorers` names the lexical channel(s) to rank with: `bm25_sapbert`
    (TF x IDF, Qdrant's sparse search) and/or `okapi_bm25_sapbert` (Okapi
    BM25, needs `okapi`). A run uses one since DEVIATIONS.md #312 (Okapi by
    default, `run_unified_ablation`); the defaults here keep the pre-#311
    behaviour for callers that pass nothing.

    `area_question_repeats` (#312): a four-area query carries its area
    question that many times; SapBERT encodes the repeated text and the
    lexical channel weights the area question's terms by the same factor
    (`blend.lexical_query_vector`), since a term-presence query ignores
    repetition. `area_question_arm` (#311, opt-in since #312) adds the
    `area_question/raw` condition: the area's question with no patient
    details, identical for every question in the area.

    `area_scoped_enrichment` (#313): enrich a four-area query only with
    concepts on a field its area's question states
    (`question_gen.areas.area_vocabulary`); off keeps whole-record
    enrichment, the behaviour before #313 and of the single-topic pool."""
    if OKAPI_SCORER in scorers and okapi is None:
        raise ValueError("the okapi_bm25_sapbert scorer needs an OkapiBM25 index")
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

        emphasis = build_area_question_only(area) if area is not None else None
        for level1, level2, query_text, enriched_text in _query_texts(
            record,
            area,
            topic,
            vocabulary,
            question_repeats=area_question_repeats,
            area_question_arm=area_question_arm,
            area_scoped_enrichment=area_scoped_enrichment,
        ):
            sapbert = cosine_raw_scores(
                sapbert_encoder.encode([query_text])[0], sapbert_chunk_matrix, corpus.chunk_ids
            )
            qvec = lexical_query_vector(
                query_text, emphasis=emphasis, weight=float(area_question_repeats)
            )
            lexical: dict[Level3Condition, dict[str, float]] = {}
            if TFIDF_SCORER in scorers:
                lexical[TFIDF_SCORER] = bm25_raw_scores(
                    store, query_text, corpus.chunk_ids, query_vector=qvec
                )
            if okapi is not None and OKAPI_SCORER in scorers:
                lexical[OKAPI_SCORER] = okapi.scores(query_text, query_vector=qvec)
            for level3, bm25 in lexical.items():
                arm = next(
                    a
                    for a in _ALL_ARMS
                    if (a.level1, a.level2, a.level3) == (level1, level2, level3)
                )
                for weight in bm25_weight_values():
                    if fusion_method == "rrf":
                        ranking = blend_bm25_dense_rrf(
                            bm25, sapbert, bm25_weight=weight, rrf_k=rrf_damping
                        )
                    else:
                        ranking = blend_bm25_dense(bm25, sapbert, bm25_weight=weight)
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
    area_question_arm: bool = False,
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
    scorer = lexical_scorer()  # fails on a bad ABLATION_LEXICAL_SCORER before encoding
    scope = enrichment_scope()  # likewise ABLATION_ENRICHMENT_SCOPE (#313)
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
        okapi=fetch_okapi_bm25(store, corpus.chunk_ids) if scorer == OKAPI_SCORER else None,
        scorers=(scorer,),
        area_question_repeats=area_question_repeats(),
        area_question_arm=area_question_arm,
        area_scoped_enrichment=scope == "area",
    )
