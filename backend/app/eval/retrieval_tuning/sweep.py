"""Grid sweep over (k, alpha) for the offline weighted-fusion experiment
(Phase 6, PRD-109 / ARCH-040). Reuses `app.eval.metrics.precision_recall_at_k`
and `.mrr` unchanged — this module only supplies alternative *rankings* to
score against the eval-question set's known gold chunks; it does not
reimplement the metrics themselves.

**RRF baseline design note:** the reference line plotted alongside the sweep
runs the same candidate lists through `QdrantVectorStore.hybrid_search`
(server-side RRF) rather than through the full `app.retrieval.hybrid.retrieve()`
pipeline. Production `retrieve()` reranks the fused list with a cross-encoder
after fusion; this experiment isolates the *fusion* algorithm's own effect
(RRF vs. a weighted linear score blend), so both arms are compared
pre-rerank, on the same footing — reranking either arm would measure the
reranker, not the fusion choice. This refines PHASE6-PROPOSAL.md §4's "run
the same set through the real, unmodified retrieve()" language; logged as a
DEVIATIONS.md entry (see README.md Phase 6 entry).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.eval import EvalQuestion
from app.eval.metrics import mrr, precision_recall_at_k
from app.eval.retrieval_tuning.offline_fusion import fetch_candidate_scores, weighted_rank
from app.ingestion.embed import embed_texts
from app.retrieval.hybrid import _expand_abbreviations
from app.retrieval.sparse import query_sparse_vector
from app.retrieval.vectorstore import QdrantVectorStore
from app.schemas.enums import ExpectedOutcome, Provenance

K_VALUES: tuple[int, ...] = tuple(range(2, 21, 2))  # 2..20 step 2 (10 values); chart 1 x-axis,
# narrowed from 2..60 step 2 per follow-up request (DEVIATIONS.md #178). _MAX_K below still
# resolves to CHART2_K_VALUES's own max (48 as of this comment), so candidate depth/MRR_K
# truncation are unaffected by this narrowing.
ALPHA_VALUES: tuple[float, ...] = tuple(round(i / 10, 1) for i in range(11))  # 0.0..1.0 step 0.1
# Chart 2's own focus-depth grid, independent of K_VALUES — changed per follow-up request to
# 8..16 step 2 (5 values, DEVIATIONS.md #179; previously 24..48 step 4, #163; previously 4..36
# step 4, #126; previously 3..36 step 3, #125; previously {20,22,...,40}, #124). Kept as a
# permanently separate constant rather than folded into K_VALUES even now that this particular
# set is a subset of it again: chart 1 plots every K_VALUES point on one x-axis line per alpha,
# and this value has already changed five times on follow-up request — coupling it to K_VALUES
# would mean each change either has to check subset membership or risks silently densifying
# chart 1's curves with points nobody asked to add there.
CHART2_K_VALUES: tuple[int, ...] = tuple(range(8, 17, 2))
# Independent of settings.candidate_k (ARCH-040) — see offline_fusion.py. Must stay comfortably
# above max(K_VALUES, CHART2_K_VALUES): a candidate pool exactly equal to the deepest k would
# silently cap recall@k at whatever recall@candidate_depth already was, rather than reflecting a
# genuine ranking effect at that depth.
CANDIDATE_DEPTH = 100
# Chart 3's MRR depth. History (DEVIATIONS.md): originally fixed at settings.top_k=8
# (production's real depth, matching eval_min_precision_at_8's own k — PHASE6-PROPOSAL.md §5's
# original justification, #121); briefly derived as "the best-recall k from chart 2"
# (mechanically CHART2_K_VALUES's max, since recall@k is monotonically non-decreasing in k — #127);
# fixed at a specific chart-2 member, 24, per direct follow-up request (#128); then 10 (#180),
# 12 (#181), 6 (#182 — the first value that ISN'T a CHART2_K_VALUES member {8,10,12,14,16},
# confirming MRR_K was never actually required to be one); now back to 12 (#183), the operator's
# own choice after weighing the tradeoff directly (values are essentially flat across this whole
# swept range regardless, so the deciding factor was interpretability: 12 sits inside panel B's
# own k=8..16 range, 6 sat below its floor). No longer tied to the production depth or the
# best-recall derivation either way.
MRR_K = 12
# Every ranked list `rank_question` produces is truncated to _MAX_K before
# either recall@k (chart 1/2) or MRR@MRR_K (chart 3) is computed from it, so
# _MAX_K must cover all three, not just the two chart-depth grids -- omitting
# MRR_K here was a latent bug that happened to never bite because every prior
# CHART2_K_VALUES had a max >= 24 (DEVIATIONS.md #179: CHART2_K_VALUES's max
# dropped to 16 on this follow-up request, the first time it's ever been
# below MRR_K, which would have silently truncated "MRR@24" to MRR@16 with no
# error at all had this not included MRR_K explicitly).
_MAX_K = max(*K_VALUES, *CHART2_K_VALUES, MRR_K)


@dataclass(frozen=True)
class SweepQuestion:
    question_id: str
    text: str
    gold_chunk_ids: frozenset[str]
    # Added for model_ablation's single-stage/multi-stage panel B split
    # (DEVIATIONS.md #184) — resolves a question's own synthetic patient
    # record for Phase 7-style vocabulary query augmentation. `None` for a
    # question with no source record, or one sourced from a de-identified
    # record (never resolved outside the synthetic index — see
    # app.eval.orchestration_ablation.augment's module docstring). Additive:
    # this module's own sweep never reads the field.
    source_record_id: uuid.UUID | None = None
    # Four-area questions (DEVIATIONS.md #264): the attested query area the
    # question was generated for; None for single-topic questions.
    area: str | None = None


def fetch_calibration_questions(session: Session) -> list[SweepQuestion]:
    """`well_supported`, auto-generated, with a non-empty gold chunk set —
    the only rows recall@k/MRR@k are meaningful for; the other two
    expected-outcome classes have no gold chunk (PHASE6-PROPOSAL.md §4).
    Uses the full `auto_generated` pool, not just `in_fixed_testset` rows,
    for statistical power — the fixed testset is deliberately small and
    pinned for CI stability, which this offline sweep doesn't need."""
    stmt = select(EvalQuestion).where(
        EvalQuestion.expected_outcome == ExpectedOutcome.WELL_SUPPORTED,
        EvalQuestion.provenance == Provenance.AUTO_GENERATED,
    )
    rows = session.execute(stmt).scalars().all()
    return [
        SweepQuestion(
            question_id=str(row.id),
            text=row.text,
            gold_chunk_ids=frozenset(row.gold_relevant_chunks),
            source_record_id=row.source_record_id,
        )
        for row in rows
        if row.gold_relevant_chunks
    ]


@dataclass(frozen=True)
class QuestionRankings:
    question_id: str
    gold_chunk_ids: frozenset[str]
    weighted_by_alpha: dict[float, list[str]]  # alpha -> ranked chunk_ids, len <= _MAX_K
    rrf_baseline: list[str]  # server-side RRF, un-reranked, len <= _MAX_K


def rank_question(
    store: QdrantVectorStore, question: SweepQuestion, *, flt: dict | None = None
) -> QuestionRankings:
    """Embeds/analyzes the question once, fetches the candidate pool once,
    and produces every swept alpha's ranking plus the RRF baseline ranking —
    so each question costs one embedding call + 3 Qdrant queries
    (dense-only, sparse-only, RRF-fused) regardless of grid size."""
    expanded = _expand_abbreviations(question.text)
    dense = embed_texts([expanded], is_query=True)[0]
    sparse = query_sparse_vector(expanded)

    candidates = fetch_candidate_scores(
        store, dense=dense, sparse=sparse, candidate_depth=CANDIDATE_DEPTH, flt=flt
    )
    weighted_by_alpha = {
        alpha: weighted_rank(candidates, alpha=alpha)[:_MAX_K] for alpha in ALPHA_VALUES
    }

    rrf_hits = store.hybrid_search(
        dense=dense, sparse=sparse, prefetch_limit=CANDIDATE_DEPTH, limit=_MAX_K, flt=flt
    )
    rrf_baseline = [h["chunk_id"] for h in rrf_hits]

    return QuestionRankings(
        question_id=question.question_id,
        gold_chunk_ids=question.gold_chunk_ids,
        weighted_by_alpha=weighted_by_alpha,
        rrf_baseline=rrf_baseline,
    )


@dataclass(frozen=True)
class SweepResult:
    # each row: {"k": int, "alpha": float | "rrf", "recall": float}; k in K_VALUES (chart 1)
    recall_rows: list[dict]
    # same shape as recall_rows, but k in CHART2_K_VALUES (chart 2's own focus-depth grid)
    chart2_recall_rows: list[dict]
    # each row: {"alpha": float | "rrf", "mrr": float}
    mrr_rows: list[dict]
    n_questions: int


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_sweep(rankings: list[QuestionRankings]) -> SweepResult:
    """Pure aggregation over already-computed rankings — no I/O, so this is
    the piece exercised by a small, fast offline unit test independent of
    `rank_question`'s Qdrant/embedding calls."""
    recall_rows: list[dict] = []
    for k in K_VALUES:
        for alpha in ALPHA_VALUES:
            recalls = [
                precision_recall_at_k(r.weighted_by_alpha[alpha], set(r.gold_chunk_ids), k)[1]
                for r in rankings
            ]
            recall_rows.append({"k": k, "alpha": alpha, "recall": _mean(recalls)})
        rrf_recalls = [
            precision_recall_at_k(r.rrf_baseline, set(r.gold_chunk_ids), k)[1] for r in rankings
        ]
        recall_rows.append({"k": k, "alpha": "rrf", "recall": _mean(rrf_recalls)})

    chart2_recall_rows: list[dict] = []
    for k in CHART2_K_VALUES:
        for alpha in ALPHA_VALUES:
            recalls = [
                precision_recall_at_k(r.weighted_by_alpha[alpha], set(r.gold_chunk_ids), k)[1]
                for r in rankings
            ]
            chart2_recall_rows.append({"k": k, "alpha": alpha, "recall": _mean(recalls)})
        rrf_recalls = [
            precision_recall_at_k(r.rrf_baseline, set(r.gold_chunk_ids), k)[1] for r in rankings
        ]
        chart2_recall_rows.append({"k": k, "alpha": "rrf", "recall": _mean(rrf_recalls)})

    mrr_rows: list[dict] = []
    for alpha in ALPHA_VALUES:
        scores = [mrr(r.weighted_by_alpha[alpha][:MRR_K], set(r.gold_chunk_ids)) for r in rankings]
        mrr_rows.append({"alpha": alpha, "mrr": _mean(scores)})
    rrf_mrr = [mrr(r.rrf_baseline[:MRR_K], set(r.gold_chunk_ids)) for r in rankings]
    mrr_rows.append({"alpha": "rrf", "mrr": _mean(rrf_mrr)})

    return SweepResult(
        recall_rows=recall_rows,
        chart2_recall_rows=chart2_recall_rows,
        mrr_rows=mrr_rows,
        n_questions=len(rankings),
    )
