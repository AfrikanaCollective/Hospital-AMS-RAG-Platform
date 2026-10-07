"""Eval metrics (ARCH §16.1; PRD-070, PRD-071)."""

from __future__ import annotations

import pytest

from app.eval.metrics import (
    citation_locus_accuracy,
    expected_outcome_pass,
    hit_at_k,
    mrr,
    precision_recall_at_k,
)
from app.schemas.enums import ExpectedOutcome, ObservedOutcome


def test_precision_recall_at_k() -> None:
    p, r = precision_recall_at_k(["a", "b", "c"], {"a", "z"}, k=3)
    assert p == pytest.approx(1 / 3)
    assert r == pytest.approx(1 / 2)


def test_precision_recall_empty_retrieved() -> None:
    assert precision_recall_at_k([], {"a"}, k=5) == (0.0, 0.0)


def test_hit_at_k_is_not_capped_by_gold_set_size() -> None:
    """DEVIATIONS.md #269: one gold chunk in the top 2 of a 7-chunk gold set
    is recall 1/7 but a hit."""
    gold = {"a", "b", "c", "d", "e", "f", "g"}
    assert hit_at_k(["x", "a", "y"], gold, k=2) == 1.0
    assert precision_recall_at_k(["x", "a", "y"], gold, k=2)[1] == pytest.approx(1 / 7)
    assert hit_at_k(["x", "y", "a"], gold, k=2) == 0.0
    assert hit_at_k([], gold, k=5) == 0.0


def test_mrr_first_hit() -> None:
    assert mrr(["x", "a", "b"], {"a"}) == pytest.approx(0.5)


def test_mrr_no_hit() -> None:
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_citation_locus_accuracy_within_one_page_and_section_prefix() -> None:
    assert citation_locus_accuracy(5, 4, "3.2.1", "3.2") is True
    assert citation_locus_accuracy(5, 4, "3.2", "3.2.1") is True
    assert citation_locus_accuracy(5, 2, "3.2", "3.2") is False  # page too far


@pytest.mark.parametrize(
    ("expected", "observed", "had_recommendation", "expected_pass"),
    [
        (ExpectedOutcome.WELL_SUPPORTED, ObservedOutcome.WELL_SUPPORTED, False, True),
        (ExpectedOutcome.WELL_SUPPORTED, ObservedOutcome.ESCALATED, False, False),
        (ExpectedOutcome.MISSING_INFO_EXPECTED, ObservedOutcome.MISSING_INFO, False, True),
        (ExpectedOutcome.NO_GUIDELINE_EXPECTED, ObservedOutcome.NO_GUIDELINE, False, True),
        (ExpectedOutcome.NO_GUIDELINE_EXPECTED, ObservedOutcome.WELL_SUPPORTED, False, False),
        # a recommendation is an automatic fail no matter what was expected
        (ExpectedOutcome.WELL_SUPPORTED, ObservedOutcome.WELL_SUPPORTED, True, False),
    ],
)
def test_expected_outcome_pass(expected, observed, had_recommendation, expected_pass) -> None:  # noqa: ANN001
    assert (
        expected_outcome_pass(expected, observed, had_recommendation=had_recommendation)
        is expected_pass
    )
