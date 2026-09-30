"""Brute-force in-memory embedding ablation: SapBERT+BM25, MedCPT+BM25,
SapBERT+MedCPT+BM25 (PRD-110 / ARCH-041,
PHASE2-EMBEDDING-ABLATION-PROPOSAL.md).

Reuses `app.eval.retrieval_tuning.sweep.fetch_calibration_questions` and
`app.eval.metrics.precision_recall_at_k`/`mrr` unchanged — this module only
supplies alternative *rankings* to score against the eval-question set's
known gold chunks.

**Fully offline, no persistence** (proposal §2): the live guideline
collection has 314 points (confirmed 2026-09-17) — small enough to rank
*exactly* by brute-force cosine similarity over the whole corpus, so no
ANN/candidate-depth truncation is needed and no SapBERT/MedCPT vector is
ever written to Qdrant; they exist only for the duration of one script run.

**RRF combine is reimplemented client-side** (proposal §5): SapBERT/MedCPT
aren't in Qdrant, so there's no collection to run Qdrant's own RRF against.
Uses the standard Cormack et al. (2009) formula with `settings.rrf_k`
(default 60 — the same constant `vectorstore.py`'s docstring documents as
"matching Qdrant's own internal constant", DEVIATIONS.md #49) — this is not
expected to reproduce Qdrant's fused scores number-for-number, only to be a
reasonable, consistently-applied combine across the three ablation arms.

**Single-stage vs. multi-stage (panel B only, DEVIATIONS.md #184)**: per
follow-up request, panel B (MRR@MRR_K) additionally splits each of the four
arms above into a single-stage point (raw question text — identical
computation/values to every prior run) and a multi-stage point (query
augmented with Phase 7's operator-vocabulary expansion, `VOCABULARY`/Arm C
from `app.eval.orchestration_ablation`, reused unchanged). `criteria_reuse`
(Arm B) was deliberately NOT used for "multi-stage" — documented
(DEVIATIONS.md #152/#159) as byte-identical to single-stage on this corpus
(zero `criteria`-type chunks, structurally unfireable), so it would show no
difference at all; `vocabulary` is standalone-computable before any
retrieval (`build_arm_c_query`) and is the only Phase 7 arm that can
actually move this chart. Panel A (recall@k) is untouched — still
single-stage only, per the request's own scope ("for panel B ..."). When
`data/clinical_concepts.yaml` isn't attested, multi-stage is skipped
entirely (same convention as Phase 7's own `vocabulary_attested` flag) and
panel B falls back to its previous single-stage-only appearance.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.config import get_settings
from app.eval.bootstrap import DEFAULT_BOOTSTRAP_SEED
from app.eval.bootstrap import bootstrap_ci as _bootstrap_ci
from app.eval.metrics import mrr, precision_recall_at_k
from app.eval.model_ablation.encoders import Encoder, get_medcpt_encoders, get_sapbert_encoder
from app.eval.orchestration_ablation.ablation import load_attested_vocabulary
from app.eval.orchestration_ablation.augment import (
    build_arm_c_query,
    load_synthetic_record_index,
    resolve_source_record,
)
from app.eval.retrieval_tuning.sweep import (
    SweepQuestion,
    fetch_calibration_questions,
)
from app.ingestion.embed import embed_texts
from app.ingestion.review import NOT_RETRIEVABLE
from app.records.concepts import ConceptVocabulary
from app.retrieval.hybrid import _expand_abbreviations
from app.retrieval.sparse import query_sparse_vector
from app.retrieval.vectorstore import QdrantVectorStore
from app.schemas.record import PatientRecord

K_VALUES: tuple[int, ...] = tuple(range(2, 21, 2))  # panel A's own grid, per follow-up request
MRR_K = 12  # matches retrieval_tuning.sweep.MRR_K, for direct comparability (DEVIATIONS #183)

ARMS: tuple[str, ...] = ("sapbert_bm25", "medcpt_bm25", "sapbert_medcpt_bm25")
REFERENCE_ARM = "rrf_production"
ALL_ARMS: tuple[str, ...] = (*ARMS, REFERENCE_ARM)

# Panel B's stage dimension (DEVIATIONS.md #184) — see module docstring.
SINGLE_STAGE = "single_stage"
MULTI_STAGE = "multi_stage"
_DEFAULT_CONCEPTS_PATH = "data/clinical_concepts.yaml"


@dataclass(frozen=True)
class Corpus:
    chunk_ids: list[str]
    texts: list[str]


# Only what production retrieval could return: active versions, and no chunk
# held for review or rejected (ARCH-044). Every Qdrant query an ablation makes
# must use it, not only `fetch_corpus`: superseded versions stay in the
# collection, and an unfiltered BM25 query ranked them into the top k, where
# they can never be gold (DEVIATIONS.md #226).
CORPUS_FILTER: dict = {"status": "active", "exclude_review_status": list(NOT_RETRIEVABLE)}


def fetch_corpus(store: QdrantVectorStore) -> Corpus:
    """Every guideline chunk, once per run — see module docstring for why
    this is feasible without an ANN index (314 points, 2026-09-17)."""
    points = store.scroll_all(flt=CORPUS_FILTER)
    return Corpus(
        chunk_ids=[p["chunk_id"] for p in points],
        texts=[p.get("text", "") for p in points],
    )


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _cosine_rank(
    query_vec: list[float], chunk_matrix: np.ndarray, chunk_ids: list[str]
) -> list[str]:
    q = np.asarray(query_vec, dtype=np.float64)
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm
    sims = chunk_matrix @ q
    order = np.argsort(-sims, kind="stable")
    return [chunk_ids[i] for i in order]


def _bm25_rank(store: QdrantVectorStore, question_text: str, chunk_ids: list[str]) -> list[str]:
    """Existing `sparse` named vector already in Qdrant (unchanged from
    Phase 6) — no new BM25 computation. A chunk sharing no term with the
    query gets no score from Qdrant's sparse query at all; such chunks are
    appended, in a deterministic (sorted) order, after every chunk that did
    score — they have zero lexical signal, so last is the correct rank, but
    the tie order among them must not depend on incidental iteration order."""
    sparse = query_sparse_vector(question_text)
    hits = store.single_vector_search(
        using="sparse", query=sparse, limit=len(chunk_ids), flt=CORPUS_FILTER
    )
    corpus_ids = set(chunk_ids)
    ranked = [h["chunk_id"] for h in hits if h["chunk_id"] in corpus_ids]
    missing = sorted(set(chunk_ids) - set(ranked))
    return ranked + missing


def _rrf_combine(rankings: list[list[str]], *, rrf_k: int) -> list[str]:
    """Cormack et al. (2009): score(d) = sum_r 1 / (rrf_k + rank_r(d) + 1),
    1-indexed rank. Every ranking here is already complete over the whole
    corpus (no missing chunks), so no separate "unranked" handling is
    needed, unlike a real ANN-truncated candidate list."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(scores, key=lambda cid: (-scores[cid], cid))


@dataclass(frozen=True)
class QuestionArmRankings:
    question_id: str
    gold_chunk_ids: frozenset[str]
    rankings: dict[str, list[str]]  # arm name -> full ranked chunk_ids (single-stage)
    # Panel B's multi-stage rankings (DEVIATIONS.md #184) — same arm keys as
    # `rankings`, computed from the vocabulary-augmented query text. `None`
    # when the whole run has no attested vocabulary (Arm C skipped entirely,
    # same convention as `app.eval.orchestration_ablation`'s
    # `vocabulary_attested`); identical to `rankings` for a question where
    # augmentation didn't fire (no resolvable record, or no concept match).
    multi_stage_rankings: dict[str, list[str]] | None = None
    multi_stage_fired: bool = False


def _rank_text(
    store: QdrantVectorStore,
    text: str,
    corpus: Corpus,
    *,
    sapbert_encoder: Encoder,
    sapbert_chunk_matrix: np.ndarray,
    medcpt_query_encoder: Encoder,
    medcpt_chunk_matrix: np.ndarray,
    rrf_k: int,
) -> dict[str, list[str]]:
    """Every ablation arm's combined ranking plus the fresh production RRF
    reference, for one already-expanded query text — factored out of
    `rank_question` so panel B's multi-stage variant (DEVIATIONS.md #184)
    can reuse it against a second, augmented text without duplicating the
    per-channel wiring."""
    bm25_rank = _bm25_rank(store, text, corpus.chunk_ids)

    sapbert_qvec = sapbert_encoder.encode([text])[0]
    sapbert_rank = _cosine_rank(sapbert_qvec, sapbert_chunk_matrix, corpus.chunk_ids)

    medcpt_qvec = medcpt_query_encoder.encode([text])[0]
    medcpt_rank = _cosine_rank(medcpt_qvec, medcpt_chunk_matrix, corpus.chunk_ids)

    dense = embed_texts([text], is_query=True)[0]
    sparse = query_sparse_vector(text)
    n = len(corpus.chunk_ids)
    prod_hits = store.hybrid_search(
        dense=dense, sparse=sparse, prefetch_limit=n, limit=n, flt=CORPUS_FILTER
    )
    rrf_production = [h["chunk_id"] for h in prod_hits]

    return {
        "sapbert_bm25": _rrf_combine([sapbert_rank, bm25_rank], rrf_k=rrf_k),
        "medcpt_bm25": _rrf_combine([medcpt_rank, bm25_rank], rrf_k=rrf_k),
        "sapbert_medcpt_bm25": _rrf_combine([sapbert_rank, medcpt_rank, bm25_rank], rrf_k=rrf_k),
        REFERENCE_ARM: rrf_production,
    }


def rank_question(
    store: QdrantVectorStore,
    question: SweepQuestion,
    corpus: Corpus,
    *,
    sapbert_encoder: Encoder,
    sapbert_chunk_matrix: np.ndarray,
    medcpt_query_encoder: Encoder,
    medcpt_chunk_matrix: np.ndarray,
    rrf_k: int,
    vocabulary: ConceptVocabulary | None = None,
    record: PatientRecord | None = None,
) -> QuestionArmRankings:
    """Embeds/queries this question once per channel and produces every
    ablation arm's combined ranking plus the fresh production RRF reference
    (recomputed live, not a hardcoded historical number — see
    PHASE2-EMBEDDING-ABLATION-PROPOSAL.md §5's "reference points" note).

    `vocabulary`/`record` are optional (default `None`, meaning "no
    multi-stage computed for this question" — preserves every prior caller's
    behavior unchanged). When `vocabulary` is not `None`, also computes
    panel B's multi-stage rankings (DEVIATIONS.md #184) via
    `app.eval.orchestration_ablation.augment.build_arm_c_query`, reusing the
    exact same Phase 7 augmentation Arm C already uses — falls back to the
    single-stage rankings (no second embed/query round-trip) whenever
    augmentation doesn't fire for this question, matching Phase 7's own
    "Arm C == Arm A" degradation."""
    expanded = _expand_abbreviations(question.text)
    single_stage_rankings = _rank_text(
        store,
        expanded,
        corpus,
        sapbert_encoder=sapbert_encoder,
        sapbert_chunk_matrix=sapbert_chunk_matrix,
        medcpt_query_encoder=medcpt_query_encoder,
        medcpt_chunk_matrix=medcpt_chunk_matrix,
        rrf_k=rrf_k,
    )

    multi_stage_rankings: dict[str, list[str]] | None = None
    multi_stage_fired = False
    if vocabulary is not None:
        augmented = build_arm_c_query(question.text, vocabulary, record)
        multi_stage_fired = augmented.fired
        if augmented.fired:
            multi_expanded = _expand_abbreviations(augmented.text)
            multi_stage_rankings = _rank_text(
                store,
                multi_expanded,
                corpus,
                sapbert_encoder=sapbert_encoder,
                sapbert_chunk_matrix=sapbert_chunk_matrix,
                medcpt_query_encoder=medcpt_query_encoder,
                medcpt_chunk_matrix=medcpt_chunk_matrix,
                rrf_k=rrf_k,
            )
        else:
            multi_stage_rankings = single_stage_rankings

    return QuestionArmRankings(
        question_id=question.question_id,
        gold_chunk_ids=question.gold_chunk_ids,
        rankings=single_stage_rankings,
        multi_stage_rankings=multi_stage_rankings,
        multi_stage_fired=multi_stage_fired,
    )


@dataclass(frozen=True)
class AblationResult:
    # each row: {"k": int, "arm": str, "recall": float} — single-stage only (panel A)
    recall_rows: list[dict]
    # each row: {"arm": str, "stage": str, "mrr": float, "ci_low": float, "ci_high": float}
    # (panel B; DEVIATIONS.md #184) — "stage" is SINGLE_STAGE or MULTI_STAGE
    mrr_rows: list[dict]
    n_questions: int
    # False when no question in this run had a multi-stage ranking computed
    # (vocabulary not attested for the whole run) — mrr_rows then carries
    # SINGLE_STAGE rows only, and panel B should render its previous,
    # single-stage-only appearance.
    multi_stage_available: bool = False
    # Fraction of questions where Arm C's vocabulary augmentation actually
    # changed the query text (0.0 when multi_stage_available is False) —
    # diagnostic only, same convention as
    # orchestration_ablation.SliceReport.fired_rate.
    multi_stage_fired_rate: float = 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# Bootstrap CI for panel B (DEVIATIONS.md #132, follow-up request). Promoted
# to `app.eval.bootstrap` (DEVIATIONS.md #192, PRD-112) so it has one shared
# home instead of a private per-module copy -- imported here under its old
# private name so every existing call site in this module is untouched.
_BOOTSTRAP_SEED = DEFAULT_BOOTSTRAP_SEED


def run_ablation(rankings: list[QuestionArmRankings]) -> AblationResult:
    """Pure aggregation over already-computed rankings — no I/O, mirroring
    `retrieval_tuning.sweep.run_sweep`'s split (fast, offline-testable)."""
    recall_rows: list[dict] = []
    for k in K_VALUES:
        for arm in ALL_ARMS:
            recalls = [
                precision_recall_at_k(r.rankings[arm], set(r.gold_chunk_ids), k)[1]
                for r in rankings
            ]
            recall_rows.append({"k": k, "arm": arm, "recall": _mean(recalls)})

    mrr_rows: list[dict] = []
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    for arm in ALL_ARMS:
        scores = [mrr(r.rankings[arm][:MRR_K], set(r.gold_chunk_ids)) for r in rankings]
        ci_low, ci_high = _bootstrap_ci(scores, rng=rng)
        mrr_rows.append(
            {
                "arm": arm,
                "stage": SINGLE_STAGE,
                "mrr": _mean(scores),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    multi_stage_available = any(r.multi_stage_rankings is not None for r in rankings)
    multi_stage_fired_rate = 0.0
    if multi_stage_available:
        multi_stage_fired_rate = _mean([1.0 if r.multi_stage_fired else 0.0 for r in rankings])
        for arm in ALL_ARMS:
            scores = [
                mrr((r.multi_stage_rankings or r.rankings)[arm][:MRR_K], set(r.gold_chunk_ids))
                for r in rankings
            ]
            ci_low, ci_high = _bootstrap_ci(scores, rng=rng)
            mrr_rows.append(
                {
                    "arm": arm,
                    "stage": MULTI_STAGE,
                    "mrr": _mean(scores),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    return AblationResult(
        recall_rows=recall_rows,
        mrr_rows=mrr_rows,
        n_questions=len(rankings),
        multi_stage_available=multi_stage_available,
        multi_stage_fired_rate=multi_stage_fired_rate,
    )


def run_full_ablation(
    session: Session,
    store: QdrantVectorStore,
    *,
    records_dir: str | None = None,
    concepts_path: str | Path = _DEFAULT_CONCEPTS_PATH,
) -> AblationResult:
    """Wires fetch -> embed -> rank -> aggregate for a real (or stub) run.
    Kept separate from `run_ablation` so the aggregation math stays testable
    without a Qdrant/model dependency (CLAUDE.md §5).

    `records_dir`/`concepts_path` feed panel B's multi-stage computation
    (DEVIATIONS.md #184) — same defaults/attestation convention as
    `app.eval.orchestration_ablation.run_full_ablation`. When
    `concepts_path` isn't attested, `vocabulary` is `None` and every
    question is ranked single-stage only, exactly as before this change."""
    settings = get_settings()
    questions = fetch_calibration_questions(session)
    if not questions:
        return AblationResult(recall_rows=[], mrr_rows=[], n_questions=0)
    corpus = fetch_corpus(store)

    sapbert_encoder = get_sapbert_encoder()
    medcpt_query_encoder, medcpt_article_encoder = get_medcpt_encoders()

    sapbert_chunk_matrix = _l2_normalize_rows(
        np.asarray(sapbert_encoder.encode(corpus.texts), dtype=np.float64)
    )
    medcpt_chunk_matrix = _l2_normalize_rows(
        np.asarray(medcpt_article_encoder.encode(corpus.texts), dtype=np.float64)
    )

    vocabulary, _vocab_error = load_attested_vocabulary(concepts_path)
    record_index: dict[uuid.UUID, dict] = (
        load_synthetic_record_index(records_dir) if vocabulary is not None else {}
    )

    rankings = [
        rank_question(
            store,
            q,
            corpus,
            sapbert_encoder=sapbert_encoder,
            sapbert_chunk_matrix=sapbert_chunk_matrix,
            medcpt_query_encoder=medcpt_query_encoder,
            medcpt_chunk_matrix=medcpt_chunk_matrix,
            rrf_k=settings.rrf_k,
            vocabulary=vocabulary,
            record=resolve_source_record(q.source_record_id, record_index)
            if vocabulary is not None
            else None,
        )
        for q in questions
    ]
    return run_ablation(rankings)
