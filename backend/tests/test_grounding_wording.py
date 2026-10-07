"""Deterministic wording filter (ARCH §8.3 step 4; SCOPE-1.2, PRD-088)."""

from __future__ import annotations

import pytest

from app.grounding.wording import has_directive_phrasing, scan_segment


@pytest.mark.parametrize(
    "text",
    [
        "You should start antibiotics now.",
        "We recommend that you escalate to HDU.",
        "The next step for this patient is a CT scan.",
        "I recommend apixaban.",
    ],
)
def test_directive_phrasing_flagged(text: str) -> None:
    assert has_directive_phrasing(text)
    assert "directive_phrasing" in scan_segment(text, cited_quotes=[])


@pytest.mark.parametrize(
    "text",
    [
        "Guideline 001 recommends recording respiratory rate at presentation.",
        "Per the retrieved source, supplemental oxygen is recommended to a documented target.",
        "The guideline does not document an alternative for this scenario.",
    ],
)
def test_reported_content_not_flagged(text: str) -> None:
    assert not has_directive_phrasing(text)
    assert scan_segment(text, cited_quotes=[]) == []


def test_dose_figure_matches_across_unicode_spacing() -> None:
    """Guideline PDFs put non-breaking / thin spaces between number and unit
    (DEVIATIONS.md #267): "5 mg/kg" in an answer is the same figure."""
    for quote in ("gentamicin 5 mg/kg every 36 hours", "gentamicin 5 mg/kg", "5mg/kg"):
        assert scan_segment("NICE gives gentamicin 5 mg/kg.", cited_quotes=[quote]) == []


def test_dose_figure_still_must_be_in_the_claims_own_quote() -> None:
    reasons = scan_segment(
        "NICE gives gentamicin 7 mg/kg.", cited_quotes=["gentamicin 5 mg/kg every 36 hours"]
    )
    assert reasons == ["dosing_beyond_source"]
    # spacing is ignored, digits are not: 7.5 is not 75
    assert scan_segment("75 mg/kg", cited_quotes=["7.5 mg/kg"]) == ["dosing_beyond_source"]
