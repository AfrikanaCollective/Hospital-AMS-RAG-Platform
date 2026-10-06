"""Four-area question generation (DEVIATIONS.md #258, #264): the attestation-gated
`data/query_areas.yaml` loader, the per-area question builder, and
`run_four_area_holdout_generation`. Fixture records use made-up values;
the real attested file is loaded read-only."""

from __future__ import annotations

import uuid
from pathlib import Path

import httpx
import pytest
import yaml

from app.db.models.eval import EvalQuestion
from app.eval import auto_seed
from app.eval.question_gen.areas import (
    AREA_ORDER,
    QueryAreasNotAttested,
    build_area_question,
    load_query_areas,
)
from app.eval.question_gen.deterministic import extract_topic
from app.schemas.enums import ScopeLabel
from app.scope.classifier import classify_scope

REAL_AREAS = Path(__file__).resolve().parents[2] / "data" / "query_areas.yaml"

RECORD = {
    "sex": "female",
    "encounter": {
        "day_of_life": 12,
        "gestational_age_weeks": 36.0,
        "birth_weight_g": 2350.0,
        "care_setting": "Newborn Unit",
        "presenting_complaint": "made-up complaint",
    },
    "vitals": [{"heart_rate_bpm": 150.0, "weight_g": 2845.0}],
    "examination_findings": [
        {"name": "made_up_sign", "present": True},
        {"name": "other_made_up_sign", "present": False},
    ],
    "maternal_risk_factors": [{"name": "made_up_risk", "present": True}],
    "medications": [{"name": "SHOULD_NEVER_APPEAR"}],
}


def _areas_dict() -> dict:
    return yaml.safe_load(REAL_AREAS.read_text(encoding="utf-8"))


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "areas.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


# ── loader ──


def test_the_attested_file_loads() -> None:
    qa = load_query_areas(REAL_AREAS)
    assert [a.name for a in qa.areas] == list(AREA_ORDER)
    assert qa.authored_by and qa.authored_role and len(qa.sha256) == 64
    caps = {a.name: a.per_guideline_cap for a in qa.areas}
    assert caps["antibiotic_course"] >= caps["assessment"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(authored_by="TODO_CONFIRM"),
        lambda d: d.pop("authored_role"),
        lambda d: d["areas"]["assessment"].update(attested=""),
        lambda d: d["areas"].pop("investigations"),
        lambda d: d["areas"]["assessment"].update(record_facts=["made_up_fact"]),
        lambda d: d["areas"]["assessment"].update(per_guideline_cap=0),
        lambda d: d["areas"]["antibiotic_course"].pop("age_bands"),
        lambda d: d["areas"]["antibiotic_course"]["weight"].update(renderings=[]),
        lambda d: d["areas"]["antibiotic_course"]["weight"].update(
            renderings=[{"round_to": -1, "decimals": 2}]
        ),
    ],
)
def test_loader_fails_closed(tmp_path: Path, mutate) -> None:  # noqa: ANN001
    data = _areas_dict()
    mutate(data)
    with pytest.raises(QueryAreasNotAttested):
        load_query_areas(_write(tmp_path, data))


# ── question builder ──


def test_each_question_carries_its_opening_and_routes_as_guideline_lookup() -> None:
    for area in load_query_areas(REAL_AREAS).areas:
        text = build_area_question(RECORD, area)
        assert extract_topic(text) == area.opening
        # never a SCOPE-2.3/2.4 boundary request (CLAUDE.md §3 rule 4)
        assert classify_scope(text, has_patient=False) == ScopeLabel.SCOPE_1
        assert "SHOULD_NEVER_APPEAR" not in text  # medications never (#155)


def test_only_the_listed_record_facts_appear(tmp_path: Path) -> None:
    data = _areas_dict()
    data["areas"]["assessment"]["record_facts"] = ["age_days", "presenting_complaint"]
    area = load_query_areas(_write(tmp_path, data)).areas[0]
    text = build_area_question(RECORD, area)
    assert "is 12 days old" in text and "presenting with made-up complaint" in text
    assert "36.0 weeks" not in text and "Vitals" not in text and "made up sign" not in text


def test_assessed_absent_findings_follow_include_absent() -> None:
    area = next(a for a in load_query_areas(REAL_AREAS).areas if a.name == "assessment")
    full = build_area_question(RECORD, area)
    present_only = build_area_question(RECORD, area, include_absent=False)
    assert "did NOT have other made up sign" in full
    assert "other made up sign" not in present_only
    assert "had made up sign" in present_only


def test_antibiotic_course_has_age_band_and_weight_renderings() -> None:
    area = next(a for a in load_query_areas(REAL_AREAS).areas if a.name == "antibiotic_course")
    text = build_area_question(RECORD, area)
    band = next(b.text for b in area.age_bands if b.from_day <= 12 <= b.to_day)
    assert f"is aged 12 days ({band})" in text
    # current weight preferred (2845 g), birth weight noted
    assert "Weight (kg): 2.845, 2.75, 3.0, 3.00" in text  # as provided, 0.25, whole, ...
    assert "(current weight; birth weight 2.35 kg)" in text


def test_weight_falls_back_to_birth_weight() -> None:
    area = next(a for a in load_query_areas(REAL_AREAS).areas if a.name == "antibiotic_course")
    record = {**RECORD, "vitals": []}
    assert "Weight (kg): 2.35" in build_area_question(record, area)
    assert "(birth weight)" in build_area_question(record, area)


# ── generation ──


class _FakeSession:
    def __init__(self) -> None:
        self.added: list = []
        self.commits = 0

    def add(self, obj: object) -> None:
        self.added.append(obj)

    def flush(self) -> None:
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()

    def commit(self) -> None:
        self.commits += 1

    @property
    def questions(self) -> list[EvalQuestion]:
        return [o for o in self.added if isinstance(o, EvalQuestion)]


def _citing(chunk_id: str) -> dict:
    return {
        "observed_outcome": "well_supported",
        "final_answer": {
            "segments": [
                {"type": "claim", "text": "Guideline X recommends Y.", "citation_ids": ["c1"]}
            ],
            "citations": [
                {
                    "citation_id": "c1",
                    "chunk_id": chunk_id,
                    "document_id": "doc-1",
                    "document_title": "Example Guideline",
                    "document_version_id": "v1",
                    "version_label": "1",
                    "quote": "recommends Y",
                    "page_start": 1,
                    "page_end": 1,
                    "char_start": 0,
                    "char_end": 12,
                    "quote_char_start": 0,
                    "quote_char_end": 12,
                }
            ],
        },
    }


def _records(n: int) -> list[tuple[uuid.UUID, dict]]:
    """Distinct made-up records (different findings, so the near-duplicate
    filter accepts them all)."""
    out = []
    for i in range(n):
        rec = {
            **RECORD,
            "examination_findings": [{"name": f"made_up_sign_{i}", "present": True}],
            "encounter": {**RECORD["encounter"], "day_of_life": i % 50},
        }
        out.append((uuid.uuid4(), rec))
    return out


@pytest.fixture
def seams(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, int | None]] = []

    def pipeline(text: str, *, per_guideline_cap: int | None = None) -> dict:
        calls.append((text, per_guideline_cap))
        return _citing(f"chunk-{len(calls)}")

    monkeypatch.setattr(auto_seed, "_COUNT_USABLE_FOUR_AREA_FN", lambda session: {})  # noqa: ARG005
    monkeypatch.setattr(auto_seed, "_LOAD_RECORDS_FN", lambda *a, **k: _records(40))  # noqa: ARG005
    monkeypatch.setattr(auto_seed, "_USED_PATIENT_IDS_FN", lambda session: set())  # noqa: ARG005
    monkeypatch.setattr(auto_seed, "_EXISTING_RECORDS_FN", lambda session: [])  # noqa: ARG005
    monkeypatch.setattr(auto_seed, "_INVOKE_PIPELINE_FN", pipeline)
    return calls


def _area_of(q: EvalQuestion) -> str:
    return q.generator_meta["area"]


def test_each_record_feeds_one_area_with_one_call_until_every_area_has_its_target(
    seams,  # noqa: ANN001
) -> None:
    session = _FakeSession()
    created = auto_seed.run_four_area_holdout_generation(
        session, per_area_target=2, areas_path=str(REAL_AREAS)
    )
    qs = session.questions
    assert len(created) == len(qs) == len(seams) == 2 * len(AREA_ORDER)  # one call each
    assert len({q.source_record_id for q in qs}) == len(qs)  # a record feeds one area only
    assert {a: sum(_area_of(q) == a for q in qs) for a in AREA_ORDER} == dict.fromkeys(
        AREA_ORDER, 2
    )
    caps = {a.name: a.per_guideline_cap for a in load_query_areas(REAL_AREAS).areas}
    for q, (text, cap) in zip(qs, seams, strict=True):
        assert q.text == text and cap == caps[_area_of(q)]
        assert q.generator_meta["template_version"] == auto_seed.FOUR_AREA_TEMPLATE
        assert q.target_guideline_ref == {"topic": extract_topic(text), "area": _area_of(q)}
        assert q.gold_relevant_chunks


def test_an_answer_without_citations_uses_its_record_but_does_not_count(
    seams, monkeypatch: pytest.MonkeyPatch
) -> None:  # noqa: ANN001
    n = {"calls": 0}

    def pipeline(text: str, *, per_guideline_cap: int | None = None) -> dict:  # noqa: ARG001
        n["calls"] += 1
        if n["calls"] == 1:  # the first (assessment) answer escalates
            return {"observed_outcome": "escalated", "escalation": {"trigger_code": "x"}}
        return _citing(f"chunk-{n['calls']}")

    monkeypatch.setattr(auto_seed, "_INVOKE_PIPELINE_FN", pipeline)
    session = _FakeSession()
    auto_seed.run_four_area_holdout_generation(
        session, per_area_target=1, areas_path=str(REAL_AREAS)
    )
    usable = [q for q in session.questions if q.gold_relevant_chunks]
    assert len(session.questions) == len(AREA_ORDER) + 1
    assert sorted(_area_of(q) for q in usable) == sorted(AREA_ORDER)  # exactly 1 usable each


def test_areas_already_at_target_get_no_more(seams, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        auto_seed,
        "_COUNT_USABLE_FOUR_AREA_FN",
        lambda session: {"assessment": 2, "investigations": 2, "antibiotic_course": 1},  # noqa: ARG005
    )
    session = _FakeSession()
    auto_seed.run_four_area_holdout_generation(
        session, per_area_target=2, areas_path=str(REAL_AREAS)
    )
    assert sorted(_area_of(q) for q in session.questions) == [
        "antibiotic_course",
        "severity_classification",
        "severity_classification",
    ]


def test_a_gateway_error_writes_nothing_for_that_record(
    seams, monkeypatch: pytest.MonkeyPatch
) -> None:  # noqa: ANN001
    def flaky(text: str, *, per_guideline_cap: int | None = None) -> dict:  # noqa: ARG001
        raise httpx.ConnectTimeout("gateway timeout")

    monkeypatch.setattr(auto_seed, "_INVOKE_PIPELINE_FN", flaky)
    session = _FakeSession()
    assert (
        auto_seed.run_four_area_holdout_generation(
            session, per_area_target=1, areas_path=str(REAL_AREAS)
        )
        == []
    )
    assert session.questions == []


def test_generation_refuses_an_unattested_areas_file(seams, tmp_path: Path) -> None:  # noqa: ANN001, ARG001
    data = _areas_dict()
    data["authored_by"] = "TODO_CONFIRM"
    with pytest.raises(QueryAreasNotAttested):
        auto_seed.run_four_area_holdout_generation(
            _FakeSession(), per_area_target=1, areas_path=str(_write(tmp_path, data))
        )
