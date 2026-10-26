"""Explicit configuration representation for the unified hierarchical
ablation (PRD-112 / ARCH-043). Restructured 2026-09-23 (DEVIATIONS.md
#201): Level 3 collapses to a single continuous BM25/SapBERT
weighted-rank-fusion sweep -- MedCPT and RRF are gone."""

from __future__ import annotations

import pytest

from app.eval.ablation_config import (
    ALL_ARMS,
    LEVEL1_CONDITIONS,
    LEVEL2_CONDITIONS,
    LEVEL3_CONDITIONS,
    AblationArm,
    area_question_repeats,
    bm25_weight_values,
    enrichment_scope,
    k_values,
    lexical_scorer,
    mrr_k,
)


def test_all_arms_has_exactly_ten_generated_leaf_configurations() -> None:
    """(2 patient-detail L1 x 2 L2 + area_question/raw) x 2 lexical scorers
    (DEVIATIONS.md #311; 4 before)."""
    assert len(ALL_ARMS) == 10
    assert len(ALL_ARMS) == len(set(ALL_ARMS))  # no duplicates


def test_all_arms_covers_every_combination_except_enriched_area_question() -> None:
    """Enrichment adds the record's concepts, which would bring patient
    details back into the area-question arm, so it has no enriched variant."""
    seen = {(a.level1, a.level2, a.level3) for a in ALL_ARMS}
    expected = {
        (l1, l2, l3)
        for l1 in LEVEL1_CONDITIONS
        for l2 in LEVEL2_CONDITIONS
        for l3 in LEVEL3_CONDITIONS
        if (l1, l2) != ("area_question", "enriched")
    }
    assert seen == expected


def test_arm_label_is_stable_and_human_readable() -> None:
    arm = AblationArm(level1="present_only", level2="enriched", level3="bm25_sapbert")
    assert arm.label == "L1-present_only__L2-enriched__L3-bm25_sapbert"


def test_arm_is_frozen_and_hashable() -> None:
    arm = AblationArm(level1="all_assessed", level2="raw", level3="bm25_sapbert")
    with pytest.raises(AttributeError):
        arm.level1 = "present_only"  # type: ignore[misc]
    assert {arm} == {arm}  # must be hashable to dedupe/set-compare, as the tests above rely on


def test_level3_conditions_are_the_two_lexical_scorers() -> None:
    """The original TF x IDF scorer keeps its `bm25_sapbert` name; Okapi BM25
    was added beside it (DEVIATIONS.md #311). `bm25_weight` is the swept
    parameter within each."""
    assert set(LEVEL3_CONDITIONS) == {"bm25_sapbert", "okapi_bm25_sapbert"}


def test_bm25_weight_values_has_six_points_at_0_2_granularity() -> None:
    """w_BM25 in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} (operator request
    2026-09-25, DEVIATIONS.md #207 / proposal §14) -- down from 11 points
    at 0.1 granularity (#201)."""
    values = bm25_weight_values()
    assert values == pytest.approx((0.0, 0.2, 0.4, 0.6, 0.8, 1.0))


def test_k_values_and_bm25_weight_values_and_mrr_k_are_settings_driven_not_hardcoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.config import get_settings

    monkeypatch.setenv("ABLATION_K_VALUES", "3,6,9")
    monkeypatch.setenv("ABLATION_BM25_WEIGHT_VALUES", "0.25,0.75")
    monkeypatch.setenv("ABLATION_MRR_K", "5")
    get_settings.cache_clear()
    try:
        assert k_values() == (3, 6, 9)
        assert bm25_weight_values() == (0.25, 0.75)
        assert mrr_k() == 5
    finally:
        get_settings.cache_clear()


def test_okapi_is_the_default_lexical_scorer_and_tfidf_is_an_option(monkeypatch) -> None:  # noqa: ANN001
    """DEVIATIONS.md #312."""
    from app.config import get_settings  # noqa: PLC0415

    monkeypatch.delenv("ABLATION_LEXICAL_SCORER", raising=False)
    get_settings.cache_clear()
    assert lexical_scorer() == "okapi_bm25_sapbert"
    monkeypatch.setenv("ABLATION_LEXICAL_SCORER", "tfidf")
    get_settings.cache_clear()
    assert lexical_scorer() == "bm25_sapbert"
    monkeypatch.setenv("ABLATION_LEXICAL_SCORER", "bm25")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="ABLATION_LEXICAL_SCORER"):
        lexical_scorer()
    get_settings.cache_clear()


def test_area_question_repeats_defaults_to_3_and_must_be_positive(monkeypatch) -> None:  # noqa: ANN001
    from app.config import get_settings  # noqa: PLC0415

    monkeypatch.delenv("ABLATION_AREA_QUESTION_REPEATS", raising=False)
    get_settings.cache_clear()
    assert area_question_repeats() == 3
    monkeypatch.setenv("ABLATION_AREA_QUESTION_REPEATS", "0")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="ABLATION_AREA_QUESTION_REPEATS"):
        area_question_repeats()
    get_settings.cache_clear()


def test_enrichment_scope_defaults_to_area_and_record_is_an_option(monkeypatch) -> None:  # noqa: ANN001
    """DEVIATIONS.md #313."""
    from app.config import get_settings  # noqa: PLC0415

    monkeypatch.delenv("ABLATION_ENRICHMENT_SCOPE", raising=False)
    get_settings.cache_clear()
    assert enrichment_scope() == "area"
    monkeypatch.setenv("ABLATION_ENRICHMENT_SCOPE", "record")
    get_settings.cache_clear()
    assert enrichment_scope() == "record"
    monkeypatch.setenv("ABLATION_ENRICHMENT_SCOPE", "everything")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="ABLATION_ENRICHMENT_SCOPE"):
        enrichment_scope()
    get_settings.cache_clear()
