"""Operator-authored concept vocabulary: attestation gate + evaluator
(SCOPE-2.6 / ARCH-042; PRD-057; PHASE7-PROPOSAL.md). Fixture vocabularies
here use obviously non-clinical values ("unit-test fixture, not a real
clinical value") -- never invented real-looking thresholds, per
PHASE7-PROPOSAL.md §4."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.records.concepts import (
    Concept,
    ConceptVocabulary,
    ConceptVocabularyNotAttested,
    build_expansion_term,
    evaluate_concepts,
    load_vocabulary,
    render_value,
)

_ATTESTED = {
    "authored_by": "unit-test fixture",
    "authored_date": "2026-09-19",
    "concepts": {
        "fake_concept": {
            "field": "vitals.heart_rate_bpm",
            "operator": ">",
            "value": 999999,
            "source": "unit-test fixture, not a real clinical value",
            "synonyms": ["fake concept", "made-up finding"],
            "notes": "not clinically meaningful",
        },
        "fake_concept_no_synonyms": {
            "field": "vitals.resp_rate_bpm",
            "operator": "<",
            "value": -999999,
            "source": "unit-test fixture, not a real clinical value",
        },
        "fake_range_concept": {
            "field": "encounter.day_of_life",
            "operator": "between",
            "low": 700,
            "high": 990,
            "source": "unit-test fixture, not a real clinical range",
            "synonyms": ["fake age band"],
        },
        "fake_presence_concept": {
            "field": "examination_findings.made_up_sign",
            "operator": "present",
            "source": "unit-test fixture, not a real clinical sign",
            "synonyms": ["fake sign"],
        },
    },
}


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "concepts.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_load_vocabulary_accepts_fully_attested_file(tmp_path: Path) -> None:
    vocab = load_vocabulary(_write(tmp_path, _ATTESTED))
    assert vocab.authored_by == "unit-test fixture"
    names = {c.name for c in vocab.concepts}
    assert names == {
        "fake_concept",
        "fake_concept_no_synonyms",
        "fake_range_concept",
        "fake_presence_concept",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(authored_by="TODO_CONFIRM"),
        lambda d: d.update(authored_date=None),
        lambda d: d["concepts"]["fake_concept"].update(value="TODO_CONFIRM"),
        lambda d: d["concepts"]["fake_concept"].update(source="TODO_CONFIRM"),
        lambda d: d["concepts"]["fake_concept"].update(field=None),
        lambda d: d["concepts"]["fake_concept"].update(operator="TODO_CONFIRM"),
        lambda d: d["concepts"]["fake_concept"].update(synonyms=["TODO_CONFIRM"]),
        lambda d: d["concepts"]["fake_range_concept"].update(low="TODO_CONFIRM"),
        lambda d: d["concepts"]["fake_range_concept"].update(high=None),
    ],
)
def test_load_vocabulary_fails_closed_on_any_placeholder(tmp_path: Path, mutate) -> None:  # noqa: ANN001
    import copy

    data = copy.deepcopy(_ATTESTED)
    mutate(data)
    with pytest.raises(ConceptVocabularyNotAttested):
        load_vocabulary(_write(tmp_path, data))


def test_load_vocabulary_rejects_unrecognized_operator(tmp_path: Path) -> None:
    import copy

    data = copy.deepcopy(_ATTESTED)
    data["concepts"]["fake_concept"]["operator"] = "!="
    with pytest.raises(ConceptVocabularyNotAttested):
        load_vocabulary(_write(tmp_path, data))


def test_load_vocabulary_rejects_range_with_low_greater_than_high(tmp_path: Path) -> None:
    import copy

    data = copy.deepcopy(_ATTESTED)
    data["concepts"]["fake_range_concept"].update(low=990, high=700)
    with pytest.raises(ConceptVocabularyNotAttested):
        load_vocabulary(_write(tmp_path, data))


def test_load_vocabulary_rejects_present_operator_with_a_value_set(tmp_path: Path) -> None:
    import copy

    data = copy.deepcopy(_ATTESTED)
    data["concepts"]["fake_presence_concept"]["value"] = 1
    with pytest.raises(ConceptVocabularyNotAttested):
        load_vocabulary(_write(tmp_path, data))


def test_load_vocabulary_synonyms_are_optional(tmp_path: Path) -> None:
    vocab = load_vocabulary(_write(tmp_path, _ATTESTED))
    no_syn = next(c for c in vocab.concepts if c.name == "fake_concept_no_synonyms")
    assert no_syn.synonyms == ()


def test_evaluate_concepts_fires_on_matching_recorded_value() -> None:
    vocab = load_vocabulary_from_dict(_ATTESTED)
    matched = evaluate_concepts(vocab, {"vitals.heart_rate_bpm": 1000000})
    assert [c.name for c in matched] == ["fake_concept"]


def test_evaluate_concepts_does_not_fire_below_threshold() -> None:
    vocab = load_vocabulary_from_dict(_ATTESTED)
    matched = evaluate_concepts(vocab, {"vitals.heart_rate_bpm": 100})
    assert matched == []


def test_evaluate_concepts_never_fires_on_a_field_never_assessed() -> None:
    """A field absent from `features` (never assessed / not recorded) must
    never fire, regardless of the threshold -- ARCH-039's tri-state."""
    vocab = load_vocabulary_from_dict(_ATTESTED)
    matched = evaluate_concepts(vocab, {"encounter.day_of_life": 2})  # no vitals at all
    assert matched == []


def test_evaluate_concepts_ignores_non_numeric_feature_values() -> None:
    vocab = load_vocabulary_from_dict(_ATTESTED)
    matched = evaluate_concepts(vocab, {"vitals.heart_rate_bpm": "not a number"})
    assert matched == []


def test_build_expansion_term_appends_synonyms_parenthetically() -> None:
    vocab = load_vocabulary_from_dict(_ATTESTED)
    concept = next(c for c in vocab.concepts if c.name == "fake_concept")
    assert build_expansion_term(concept) == "fake_concept (fake concept, made-up finding)"


def test_build_expansion_term_bare_name_when_no_synonyms() -> None:
    vocab = load_vocabulary_from_dict(_ATTESTED)
    concept = next(c for c in vocab.concepts if c.name == "fake_concept_no_synonyms")
    assert build_expansion_term(concept) == "fake_concept_no_synonyms"


def test_evaluate_concepts_between_fires_inside_range_inclusive() -> None:
    vocab = load_vocabulary_from_dict(_ATTESTED)
    for boundary in (700, 990, 850):
        matched = evaluate_concepts(vocab, {"encounter.day_of_life": boundary})
        assert [c.name for c in matched] == ["fake_range_concept"]


def test_evaluate_concepts_between_does_not_fire_outside_range() -> None:
    vocab = load_vocabulary_from_dict(_ATTESTED)
    assert evaluate_concepts(vocab, {"encounter.day_of_life": 699}) == []
    assert evaluate_concepts(vocab, {"encounter.day_of_life": 991}) == []


def test_evaluate_concepts_present_fires_only_on_exact_true() -> None:
    vocab = load_vocabulary_from_dict(_ATTESTED)
    matched = evaluate_concepts(vocab, {"examination_findings.made_up_sign": True})
    assert [c.name for c in matched] == ["fake_presence_concept"]


def test_evaluate_concepts_present_does_not_fire_on_assessed_negative() -> None:
    """Assessed and absent (False) must not fire the presence concept --
    distinct from never-assessed (absent from features), which also doesn't
    fire via the earlier `field not in features` check."""
    vocab = load_vocabulary_from_dict(_ATTESTED)
    assert evaluate_concepts(vocab, {"examination_findings.made_up_sign": False}) == []


_VALUE_SPEC = {
    "field": "encounter.made_up_mass_g",
    "operator": "value",
    "scale": 0.001,
    "unit": "kg",
    "renderings": [
        {"round_to": None, "decimals": None},
        {"round_to": 0.25, "decimals": 2},
        {"round_to": 1, "decimals": 1},
    ],
    "source": "unit-test fixture, not a real clinical value",
}


def _value_vocab(tmp_path: Path, **overrides: object) -> ConceptVocabulary:
    spec = {**_VALUE_SPEC, **overrides}
    data = {**_ATTESTED, "concepts": {"fake mass (kg)": spec}}
    return load_vocabulary(_write(tmp_path, data))


def test_value_operator_loads_with_renderings(tmp_path: Path) -> None:
    (concept,) = _value_vocab(tmp_path).concepts
    assert concept.operator == "value" and concept.unit == "kg" and concept.scale == 0.001
    assert concept.renderings == ((None, None), (0.25, 2), (1.0, 1))


@pytest.mark.parametrize(
    "overrides",
    [
        {"value": 1},
        {"low": 1, "high": 2},
        {"renderings": []},
        {"renderings": [{"round_to": 0, "decimals": 2}]},
        {"renderings": [{"round_to": 0.25, "decimals": -1}]},
        {"renderings": [{"round_to": 0.25, "decimals": 1.5}]},
        {"scale": 0},
        {"scale": "a lot"},
    ],
)
def test_value_operator_rejects_bad_specs(tmp_path: Path, overrides: dict) -> None:
    with pytest.raises(ConceptVocabularyNotAttested):
        _value_vocab(tmp_path, **overrides)


@pytest.mark.parametrize(
    ("grams", "expected"),
    [
        (2350, ("2.35 kg", "2.25 kg", "2.0 kg")),
        (2375, ("2.375 kg", "2.50 kg", "2.0 kg")),  # half-up at the 0.25 midpoint
        (4000, ("4.0 kg", "4.00 kg", "4.0 kg")),  # one per rendering, duplicates kept
        (1500, ("1.5 kg", "1.50 kg", "2.0 kg")),  # half-up to the whole kg
    ],
)
def test_render_value_as_provided_and_rounded(tmp_path: Path, grams: int, expected: tuple) -> None:
    (concept,) = _value_vocab(tmp_path).concepts
    assert render_value(concept, grams) == expected


def test_value_concept_fires_with_rendered_values_as_synonyms(tmp_path: Path) -> None:
    vocab = _value_vocab(tmp_path)
    (matched,) = evaluate_concepts(vocab, {"encounter.made_up_mass_g": 2350})
    assert build_expansion_term(matched) == "fake mass (kg) (2.35 kg, 2.25 kg, 2.0 kg)"


def test_value_concept_never_fires_on_missing_bool_or_text(tmp_path: Path) -> None:
    vocab = _value_vocab(tmp_path)
    assert evaluate_concepts(vocab, {}) == []
    assert evaluate_concepts(vocab, {"encounter.made_up_mass_g": True}) == []
    assert evaluate_concepts(vocab, {"encounter.made_up_mass_g": "heavy"}) == []


def load_vocabulary_from_dict(data: dict) -> ConceptVocabulary:
    concepts = tuple(
        Concept(
            name=name,
            field=spec["field"],
            operator=spec["operator"],
            source=spec["source"],
            value=float(spec["value"]) if "value" in spec else None,
            low=float(spec["low"]) if "low" in spec else None,
            high=float(spec["high"]) if "high" in spec else None,
            synonyms=tuple(spec.get("synonyms") or ()),
            notes=spec.get("notes"),
        )
        for name, spec in data["concepts"].items()
    )
    return ConceptVocabulary(
        authored_by=data["authored_by"], authored_date=data["authored_date"], concepts=concepts
    )
