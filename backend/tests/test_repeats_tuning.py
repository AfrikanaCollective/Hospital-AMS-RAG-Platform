"""Choosing the area-question repeat count on a tune split and checking it on
a held-out split (PRD-112 / ARCH-043; DEVIATIONS.md #313). Uses a fake
sweep, so no encoder or Qdrant is needed."""

from __future__ import annotations

import uuid

import pytest

from app.eval.retrieval_tuning.sweep import SweepQuestion
from app.eval.unified_ablation.per_query import PerQueryResult
from app.eval.unified_ablation.repeats_tuning import split_of, tune_repeats


def _question(i: int) -> SweepQuestion:
    return SweepQuestion(
        question_id=f"q{i}",
        text="t",
        gold_chunk_ids=frozenset({"g"}),
        source_record_id=uuid.uuid5(uuid.NAMESPACE_URL, f"rec-{i}"),
        area="assessment",
    )


QUESTIONS = [_question(i) for i in range(40)]


def _row(q: SweepQuestion, rr: float, weight: float) -> PerQueryResult:
    return PerQueryResult(
        query_id=q.question_id,
        patient_id_or_case_id=str(q.source_record_id),
        experiment_id="e",
        level1_condition="present_only",
        level2_condition="raw",
        level3_condition="okapi_bm25_sapbert",
        k=12,
        bm25_weight=weight,
        query_text="t",
        concept_enriched_query="t",
        retrieved_ids=["g"],
        relevant_ids=["g"],
        first_relevant_rank=1,
        recall_at_k=rr,
        reciprocal_rank_at_k=rr,
        question_area=q.area,
    )


def _fake_sweep(best: int):
    """RR peaks at `best` and falls off by 0.1 per step, for every question."""

    def sweep(qs: list[SweepQuestion], n: int) -> list[PerQueryResult]:
        rr = max(0.0, 1.0 - 0.1 * abs(n - best))
        return [_row(q, rr, w) for q in qs for w in (0.0, 1.0)]

    return sweep


def test_split_is_by_source_record_and_stable() -> None:
    splits = [split_of(q) for q in QUESTIONS]
    assert set(splits) == {"tune", "heldout"}
    assert splits == [split_of(q) for q in QUESTIONS]
    twin = SweepQuestion(
        question_id="other",
        text="t",
        gold_chunk_ids=frozenset(),
        source_record_id=QUESTIONS[0].source_record_id,
    )
    assert split_of(twin) == split_of(QUESTIONS[0])


def test_chooses_the_best_n_on_tune_and_compares_it_on_heldout() -> None:
    result = tune_repeats(QUESTIONS, _fake_sweep(best=4), repeats_grid=(1, 2, 3, 4, 6))
    assert result.chosen == 4
    assert result.selection_scores["tune"][4] == pytest.approx(1.0)
    assert {c.baseline_repeats for c in result.heldout_comparisons} == {1, 3}
    vs3 = next(c for c in result.heldout_comparisons if c.baseline_repeats == 3)
    assert vs3.mean_delta == pytest.approx(0.1)
    assert vs3.n_questions == sum(1 for q in QUESTIONS if split_of(q) == "heldout")
    # one cell per split x N x weight for the single fake arm
    assert len(result.cells) == 2 * 5 * 2


def test_ties_go_to_the_smaller_n_and_the_baseline_itself_is_not_compared() -> None:
    flat = tune_repeats(
        QUESTIONS, lambda qs, n: [_row(q, 0.5, 0.0) for q in qs], repeats_grid=(3, 1, 2)
    )
    assert flat.chosen == 1
    assert {c.baseline_repeats for c in flat.heldout_comparisons} == {3}


def test_heldout_questions_never_reach_the_selection() -> None:
    """Held-out questions score best at N=6, tune questions at N=2: the
    choice follows tune."""
    heldout_ids = {q.question_id for q in QUESTIONS if split_of(q) == "heldout"}

    def sweep(qs: list[SweepQuestion], n: int) -> list[PerQueryResult]:
        return [
            _row(q, 1.0 - 0.1 * abs(n - (6 if q.question_id in heldout_ids else 2)), 0.0)
            for q in qs
        ]

    result = tune_repeats(QUESTIONS, sweep, repeats_grid=(1, 2, 6))
    assert result.chosen == 2
    assert result.selection_scores["heldout"][6] > result.selection_scores["heldout"][2]


def test_invalid_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="repeats_grid"):
        tune_repeats(QUESTIONS, _fake_sweep(best=1), repeats_grid=(0, 1))
