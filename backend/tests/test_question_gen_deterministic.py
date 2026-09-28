"""Deterministic ablation-study narrative construction (DEVIATIONS.md #156,
#190, #192; PRD-112 for `build_present_only_narrative`, Level 1A of the
unified hierarchical ablation)."""

from __future__ import annotations

from app.eval.question_gen.deterministic import (
    build_deterministic_narrative,
    build_present_only_narrative,
    extract_topic,
)

RECORD = {
    "record_id": "SYNREC-TEST",
    "mrn": "SYN-TEST",
    "sex": "female",
    "encounter": {
        "gestational_age_weeks": 33.0,
        "birth_weight_g": 1800.0,
        "day_of_life": 2,
        "care_setting": "NBU",
        "presenting_complaint": "poor feeding",
    },
    "examination_findings": [
        {"name": "grunting", "present": True},
        {"name": "difficulty_feeding", "present": True},
        {"name": "convulsions", "present": False},
        {"name": "floppy", "present": False},
    ],
    "maternal_risk_factors": [
        {"name": "maternal_infection", "present": True},
        {"name": "prom", "present": False},
    ],
    "vitals": [{"heart_rate_bpm": 190.0, "resp_rate_bpm": 65.0}],
    "medications": [{"name": "gentamicin", "active": True}],
    "interventions": [{"name": "cpap", "active": True}],
}


def test_includes_topic_line() -> None:
    text = build_deterministic_narrative(RECORD, topic="antibiotics or infection")
    assert text.startswith(
        "What does the guideline recommend about antibiotics or infection "
        "based only on the content provided below:"
    )


def test_includes_demographic_fields_present() -> None:
    text = build_deterministic_narrative(RECORD, topic="x")
    assert "female" in text
    assert "is 2 days old" in text
    assert "born at 33.0 weeks gestation" in text
    assert "birth weight of 1800.0 g" in text
    assert "currently in NBU" in text
    assert "presenting with poor feeding" in text


def test_omits_absent_demographic_fields_never_states_them() -> None:
    record = {"record_id": "r", "mrn": "m", "sex": "male", "encounter": {}}
    text = build_deterministic_narrative(record, topic="x")
    assert "gestation" not in text
    assert "birth weight" not in text
    assert "NBU" not in text


def test_present_and_absent_exam_findings_both_appear_with_correct_phrasing() -> None:
    text = build_deterministic_narrative(RECORD, topic="x")
    assert "the patient had grunting, and difficulty feeding." in text
    assert "the patient did NOT have convulsions, and floppy." in text


def test_never_assessed_finding_is_never_mentioned_either_way() -> None:
    record = {
        **RECORD,
        "examination_findings": [{"name": "grunting", "present": True}],  # apnoea never assessed
        "maternal_risk_factors": [],
    }
    text = build_deterministic_narrative(record, topic="x")
    assert "apnoea" not in text
    assert "had grunting" in text
    assert "did NOT have" not in text  # nothing was assessed-negative here


def test_maternal_risk_factors_tri_state() -> None:
    text = build_deterministic_narrative(RECORD, topic="x")
    assert "the mother had maternal infection" in text
    assert "the mother did NOT have prom" in text


def test_vitals_included() -> None:
    text = build_deterministic_narrative(RECORD, topic="x")
    assert "heart rate (bpm) 190.0" in text
    assert "respiratory rate (bpm) 65.0" in text


def test_never_includes_medications_or_interventions() -> None:
    """DEVIATIONS.md #155's constraint applies here independently too."""
    text = build_deterministic_narrative(RECORD, topic="x")
    assert "gentamicin" not in text
    assert "cpap" not in text
    assert "medication" not in text.lower()
    assert "intervention" not in text.lower()


def test_no_findings_assessed_at_all_omits_the_assessment_section() -> None:
    record = {"record_id": "r", "mrn": "m", "sex": "male", "encounter": {}}
    text = build_deterministic_narrative(record, topic="x")
    assert "assessments at admission" not in text.lower()


def test_no_maternal_risk_factors_assessed_omits_that_clause() -> None:
    record = {"record_id": "r", "mrn": "m", "sex": "male", "encounter": {}}
    text = build_deterministic_narrative(record, topic="x")
    assert "maternal risk factors" not in text.lower()


def test_single_present_finding_has_no_trailing_and() -> None:
    record = {**RECORD, "examination_findings": [{"name": "grunting", "present": True}]}
    text = build_deterministic_narrative(record, topic="x")
    assert "the patient had grunting." in text


def test_empty_record_produces_only_the_topic_line_and_bare_sex_sentence() -> None:
    text = build_deterministic_narrative({"record_id": "r", "mrn": "m"}, topic="x")
    assert "unspecified sex" in text
    assert "assessments at admission" not in text.lower()
    assert "maternal risk factors" not in text.lower()


# ── build_present_only_narrative (Level 1A, PRD-112) ─────────────────────────


def test_present_only_includes_present_findings_but_never_the_absent_clause() -> None:
    text = build_present_only_narrative(RECORD, topic="x")
    assert "the patient had grunting, and difficulty feeding." in text
    assert "did NOT have" not in text
    assert "convulsions" not in text  # assessed-absent -- present-only omits it entirely
    assert "floppy" not in text


def test_present_only_maternal_risk_factors_omit_the_absent_clause_too() -> None:
    text = build_present_only_narrative(RECORD, topic="x")
    assert "the mother had maternal infection" in text
    assert "did NOT have" not in text
    assert "prom" not in text  # assessed-absent -- omitted


def test_present_only_still_never_infers_an_unassessed_sign_as_absent() -> None:
    """No inference either way (UNIFIED-ABLATION-PROPOSAL.md §II): a sign
    never assessed is never mentioned, same as today's all-assessed mode."""
    record = {
        **RECORD,
        "examination_findings": [{"name": "grunting", "present": True}],  # apnoea never assessed
        "maternal_risk_factors": [],
    }
    text = build_present_only_narrative(record, topic="x")
    assert "apnoea" not in text
    assert "had grunting" in text


def test_present_only_omits_the_assessment_section_when_everything_assessed_was_absent() -> None:
    """All signs assessed-absent, none present -- present-only mode has
    nothing positive to report, so the whole section is omitted (distinct
    from all-assessed mode, which would still render the "did NOT have"
    line for the same record)."""
    record = {
        **RECORD,
        "examination_findings": [{"name": "convulsions", "present": False}],
        "maternal_risk_factors": [{"name": "prom", "present": False}],
    }
    present_only_text = build_present_only_narrative(record, topic="x")
    all_assessed_text = build_deterministic_narrative(record, topic="x")
    assert "assessments at admission" not in present_only_text.lower()
    assert "maternal risk factors" not in present_only_text.lower()
    assert "did NOT have" not in present_only_text
    # Contrast: all-assessed mode DOES render it for the identical record.
    assert "did NOT have convulsions" in all_assessed_text
    assert "did NOT have prom" in all_assessed_text


def test_present_only_and_all_assessed_agree_on_everything_but_the_absent_clause() -> None:
    """The two Level-1 conditions must be identical apart from the
    present/absent distinction itself (comparability requirement,
    UNIFIED-ABLATION-PROPOSAL.md §III) -- same topic line, same demographic
    sentence, same vitals, same present findings."""
    present_only = build_present_only_narrative(RECORD, topic="antibiotics")
    all_assessed = build_deterministic_narrative(RECORD, topic="antibiotics")
    assert present_only.splitlines()[0] == all_assessed.splitlines()[0]  # topic line
    assert "female" in present_only and "female" in all_assessed
    assert "heart rate (bpm) 190.0" in present_only
    assert "heart rate (bpm) 190.0" in all_assessed
    assert "the patient had grunting, and difficulty feeding." in present_only
    assert "the patient had grunting, and difficulty feeding." in all_assessed


def test_extract_topic_round_trips_both_builders() -> None:
    """DEVIATIONS.md #208: the unified ablation recovers a stored question's
    topic with `extract_topic`."""
    topic = "antibiotics or infection in hospital settings for this newborn's presentation"
    assert extract_topic(build_deterministic_narrative(RECORD, topic=topic)) == topic
    assert extract_topic(build_present_only_narrative(RECORD, topic=topic)) == topic


def test_extract_topic_returns_none_for_non_template_text() -> None:
    assert extract_topic("What antibiotics are used for neonatal sepsis?") is None
    assert extract_topic("") is None
