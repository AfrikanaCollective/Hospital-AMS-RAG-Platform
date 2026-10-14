"""Triage-agreement review question and the triage-stratified review queue
(ARCH-046, PRD-114; DEVIATIONS.md #298-#300)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import app.audit.log as audit_log
import app.rubric.workflow as wf
from app.db.models.eval import EvalQuestion, Result, RubricRating
from app.eval import auto_seed
from app.eval.question_gen.areas import AREA_ORDER, build_area_question, load_query_areas
from app.eval.question_gen.generate import _SECTION_FIELDS
from app.eval.triage import (
    TRIAGE_AGREEMENT_DOMAIN,
    TRIAGE_AGREEMENT_DOMAIN_CODE,
    TRIAGE_CATEGORIES,
    TriageMortalityNotAttested,
    load_triage_mortality,
    record_triage_category,
    triage_narrative,
    triage_review_applies,
)
from app.ingestion.eav import MappingSpec
from app.rubric.domains import RUBRIC_DOMAIN_CODES
from app.schemas.enums import HitlAcceptAction
from app.schemas.rubric import DomainScore, RatingRoundRequest
from tests.test_four_area_questions import (
    REAL_AREAS,
    _citing,
    _FakeAugmented,
    _FakeSession,
    _review_records,
)

REPO = Path(__file__).resolve().parents[2]
REAL_MORTALITY = REPO / "data" / "triage_mortality.yaml"
REAL_MAPPING = REPO / "data/patient_records/deidentified/newborn_nbu_2021/field_mapping.yaml"


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "triage.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


# ── attested mortality table ──


def test_the_attested_table_loads() -> None:
    table = load_triage_mortality(REAL_MORTALITY)
    assert set(table.mortality) == set(TRIAGE_CATEGORIES)
    assert table.mortality == {
        "Critical": ">60%",
        "Substantial": ">45%",
        "Moderate": "~9%",
        "Low": "~3%",
        "Minimal": "<1%",
    }
    assert "10.1371/journal.pone.0359212" in table.url


def test_the_narrative_uses_the_agreed_wording() -> None:
    table = load_triage_mortality(REAL_MORTALITY)
    assert triage_narrative("Moderate", table) == (
        "This newborn was assigned triage category Moderate based on the information "
        "available at the time of admission. In the PLOS ONE study "
        "(doi:10.1371/journal.pone.0359212), newborns in this category had a mortality "
        "rate of ~9%. How strongly do you agree with this triage category, given the "
        "clinical features shown?"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(attested_by="TODO_CONFIRM"),
        lambda d: d.pop("attested_date"),
        lambda d: d["source"].update(url=""),
        lambda d: d["categories"].pop("Low"),
        lambda d: d["categories"].update(Extra={"mortality": "1%"}),
        lambda d: d["categories"]["Critical"].update(mortality="TODO_CONFIRM"),
    ],
)
def test_the_loader_fails_closed(tmp_path: Path, mutate) -> None:  # noqa: ANN001
    data = yaml.safe_load(REAL_MORTALITY.read_text(encoding="utf-8"))
    mutate(data)
    with pytest.raises(TriageMortalityNotAttested):
        load_triage_mortality(_write(tmp_path, data))


def test_a_missing_table_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(TriageMortalityNotAttested):
        load_triage_mortality(tmp_path / "absent.yaml")


def test_only_assessment_items_with_a_known_category_ask_it() -> None:
    assert triage_review_applies("assessment", "Critical")
    for area in AREA_ORDER:
        if area != "assessment":
            assert not triage_review_applies(area, "Critical")
    assert not triage_review_applies("assessment", None)
    assert not triage_review_applies("assessment", "None")


def test_the_triage_domain_is_not_one_of_the_eleven() -> None:
    assert TRIAGE_AGREEMENT_DOMAIN_CODE not in RUBRIC_DOMAIN_CODES
    assert TRIAGE_AGREEMENT_DOMAIN.anchors[0] == "Strongly disagree"
    assert TRIAGE_AGREEMENT_DOMAIN.anchors[4] == "Strongly agree"


# ── never used for guideline retrieval ──


def _record(category: str | None = "Critical") -> dict:
    rec = _review_records(1)[0][1]
    return {**rec, "encounter": {**rec["encounter"], "triage_category": category}}


def test_no_area_question_mentions_the_triage_category() -> None:
    areas = load_query_areas(REAL_AREAS)
    rec = _record("Substantial")
    for area in areas.areas:
        text = build_area_question(rec, area)
        assert "Substantial" not in text
        assert "triage" not in text.lower()


def test_the_single_topic_generator_does_not_render_triage() -> None:
    assert "triage_category" not in {key for key, _ in _SECTION_FIELDS}


# ── mapping (2026-10-14 export) ──


def test_mapping_adds_difficulty_breathing_and_drops_birth_date_time() -> None:
    spec = MappingSpec.from_yaml(REAL_MAPPING)
    raw = yaml.safe_load(REAL_MAPPING.read_text(encoding="utf-8"))
    assert "difficulty_breathing" in raw["list_targets"]["examination_findings"]["members"]
    assert "birth_date_time" not in spec.fields
    assert spec.fields["triage_category"]["target"] == "encounter.triage_category"
    assert raw["derived"]["date_of_birth"] == {"rule": "admitted_minus_days"}


# ── stratified record selection ──


def _stratified_records(per_category: int) -> list[tuple[uuid.UUID, dict]]:
    out = []
    recs = _review_records(per_category * len(TRIAGE_CATEGORIES))
    for i, (pid, rec) in enumerate(recs):
        category = TRIAGE_CATEGORIES[i % len(TRIAGE_CATEGORIES)]
        out.append((pid, {**rec, "encounter": {**rec["encounter"], "triage_category": category}}))
    return out


def test_record_triage_category_reads_known_values_only() -> None:
    assert record_triage_category(_record("Low")) == "Low"
    assert record_triage_category(_record(None)) is None
    assert record_triage_category(_record("low")) is None


def test_plan_picks_per_category_and_skips_used_records() -> None:
    records = _stratified_records(4)
    used = {records[0][0]}  # a Critical record
    picks, short = auto_seed.plan_triage_review_records(
        records,
        already={},
        used_ids=used,
        per_category=3,
        seed=1,
        accepted_records=[],
        dedup_threshold=0.9,
    )
    by_category: dict[str, int] = {}
    for pid, _rec, category in picks:
        assert pid not in used
        by_category[category] = by_category.get(category, 0) + 1
    assert by_category == {c: 3 for c in TRIAGE_CATEGORIES}
    assert short == {c: 0 for c in TRIAGE_CATEGORIES}


def test_plan_counts_records_already_queued_and_reports_shortfall() -> None:
    records = _stratified_records(2)
    already = {uuid.uuid4(): "Critical"}
    picks, short = auto_seed.plan_triage_review_records(
        records,
        already=already,
        used_ids=set(),
        per_category=3,
        seed=1,
        accepted_records=[],
        dedup_threshold=0.9,
    )
    critical = [p for p in picks if p[2] == "Critical"]
    assert len(critical) == 2
    assert short["Critical"] == 0
    assert short["Low"] == 1


# ── queue generation ──


@pytest.fixture
def triage_seams(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, int | None, dict]] = []
    records = _stratified_records(3)

    def pipeline(text: str, *, per_guideline_cap: int | None = None, **kw: object) -> dict:
        calls.append((text, per_guideline_cap, kw))
        return _citing(f"chunk-{len(calls)}")

    monkeypatch.setattr(auto_seed, "_TRIAGE_REVIEW_PROGRESS_FN", lambda session: {})  # noqa: ARG005
    monkeypatch.setattr(auto_seed, "_LOAD_RECORDS_FN", lambda *a, **k: records)  # noqa: ARG005
    monkeypatch.setattr(auto_seed, "_USED_PATIENT_IDS_FN", lambda session: set())  # noqa: ARG005
    monkeypatch.setattr(auto_seed, "_EXISTING_RECORDS_FN", lambda session: [])  # noqa: ARG005
    monkeypatch.setattr(auto_seed, "_INVOKE_PIPELINE_FN", pipeline)
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_ATTESTED_VOCABULARY_FN",
        lambda path: (object(), None),  # noqa: ARG005
    )
    monkeypatch.setattr(
        auto_seed,
        "_BUILD_ARM_C_QUERY_FN",
        lambda text, vocabulary, record: _FakeAugmented(text),  # noqa: ARG005
    )
    return calls, records


def _run(session: _FakeSession, per_category: int = 2, **kw: object) -> list[uuid.UUID]:
    return auto_seed.run_triage_review_queue(
        session,
        per_category=per_category,
        seed=7,
        areas_path=str(REAL_AREAS),
        mortality_path=str(REAL_MORTALITY),
        **kw,
    )


def test_each_record_gets_all_four_areas_per_category(triage_seams) -> None:  # noqa: ANN001
    session = _FakeSession()
    created = _run(session, per_category=2)
    questions = session.questions
    assert len(created) == len(questions) == 2 * len(TRIAGE_CATEGORIES) * len(AREA_ORDER)
    per_record: dict[uuid.UUID, set[str]] = {}
    per_category: dict[str, set[uuid.UUID]] = {}
    for q in questions:
        meta = q.generator_meta
        assert meta["purpose"] == auto_seed.FOUR_AREA_REVIEW_PURPOSE
        assert meta["queue_design"] == auto_seed.TRIAGE_REVIEW_QUEUE_DESIGN
        per_record.setdefault(q.source_record_id, set()).add(meta["area"])
        per_category.setdefault(meta["triage_category"], set()).add(q.source_record_id)
    assert all(areas == set(AREA_ORDER) for areas in per_record.values())
    assert {c: len(ids) for c, ids in per_category.items()} == {c: 2 for c in TRIAGE_CATEGORIES}


def test_only_assessment_items_pin_the_mortality_table(triage_seams) -> None:  # noqa: ANN001
    session = _FakeSession()
    _run(session, per_category=1)
    sha = load_triage_mortality(REAL_MORTALITY).sha256
    for q in session.questions:
        if q.generator_meta["area"] == "assessment":
            assert q.generator_meta["triage_mortality_sha256"] == sha
        else:
            assert "triage_mortality_sha256" not in q.generator_meta


def test_the_triage_category_never_reaches_the_pipeline(triage_seams) -> None:  # noqa: ANN001
    calls, _ = triage_seams
    _run(_FakeSession(), per_category=1)
    assert calls
    for text, _cap, _kw in calls:
        assert "triage" not in text.lower()
        assert not any(c in text for c in TRIAGE_CATEGORIES)


def test_unfinished_records_are_completed_first(
    triage_seams,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, records = triage_seams
    pid, rec = next(r for r in records if r[1]["encounter"]["triage_category"] == "Low")
    monkeypatch.setattr(
        auto_seed,
        "_TRIAGE_REVIEW_PROGRESS_FN",
        lambda session: {pid: ("Low", {"assessment", "investigations"})},  # noqa: ARG005
    )
    session = _FakeSession()
    _run(session, per_category=1)
    done_for_pid = {
        q.generator_meta["area"] for q in session.questions if q.source_record_id == pid
    }
    assert done_for_pid == {"severity_classification", "antibiotic_course"}
    low = {
        q.source_record_id
        for q in session.questions
        if q.generator_meta["triage_category"] == "Low"
    }
    assert low == {pid}  # Low already had its one record


def test_generation_refuses_an_unattested_mortality_table(
    triage_seams,  # noqa: ANN001, ARG001
    tmp_path: Path,
) -> None:
    data = yaml.safe_load(REAL_MORTALITY.read_text(encoding="utf-8"))
    data["attested_by"] = "TODO_CONFIRM"
    with pytest.raises(TriageMortalityNotAttested):
        auto_seed.run_triage_review_queue(
            _FakeSession(),
            per_category=1,
            areas_path=str(REAL_AREAS),
            mortality_path=str(_write(tmp_path, data)),
        )


# ── rating: required on assessment items, refused elsewhere ──


class _QuestionSession:
    def __init__(self, question: EvalQuestion | None) -> None:
        self.question = question

    def get(self, model: object, pk: object) -> object:  # noqa: ARG002
        return self.question


def _question(area: str, category: str | None) -> EvalQuestion:
    meta: dict = {"area": area}
    if category is not None:
        meta["triage_category"] = category
    return EvalQuestion(id=uuid.uuid4(), text="q", generator_meta=meta)


def _result_for(question: EvalQuestion | None) -> Result:
    return Result(
        id=uuid.uuid4(),
        eval_question_id=question.id if question is not None else None,
        provenance="auto_generated",
        queue_state="open",
        created_at=datetime.now(UTC),
    )


@pytest.mark.parametrize(
    ("area", "category", "expected"),
    [
        ("assessment", "Critical", "Critical"),
        ("investigations", "Critical", None),
        ("antibiotic_course", "Minimal", None),
        ("assessment", None, None),
    ],
)
def test_triage_category_for_result(area: str, category: str | None, expected: str | None) -> None:
    q = _question(area, category)
    assert wf.triage_category_for_result(_QuestionSession(q), _result_for(q)) == expected


def test_a_result_without_a_question_never_asks() -> None:
    assert wf.triage_category_for_result(_QuestionSession(None), _result_for(None)) is None


class _RatingSession:
    def __init__(self, result: Result, question: EvalQuestion) -> None:
        self.result, self.question = result, question
        self.added: list = []

    def get(self, model: object, pk: object) -> object:
        if model is EvalQuestion:
            return self.question if pk == self.question.id else None
        return self.result if pk == self.result.id else None

    def add(self, obj: object) -> None:
        self.added.append(obj)

    def flush(self) -> None:
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()

    def execute(self, stmt: object) -> object:  # noqa: ARG002
        class _Res:
            def scalars(self) -> _Res:
                return self

            def all(self) -> list:
                return []

        return _Res()


@pytest.fixture(autouse=True)
def _no_real_audit_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit_log, "_fetch_last_row_hash", lambda session: None)  # noqa: ARG005


def _scores() -> list[DomainScore]:
    return [DomainScore(domain_code=c, score=4) for c in RUBRIC_DOMAIN_CODES]


def _rate(area: str, triage_agreement: int | None) -> _RatingSession:
    q = _question(area, "Moderate")
    session = _RatingSession(_result_for(q), q)
    wf.submit_rating(
        session,  # type: ignore[arg-type]
        result_id=session.result.id,
        rater_id=uuid.uuid4(),
        scores=_scores(),
        accept_action=HitlAcceptAction.FULL_ACCEPT,
        triage_agreement=triage_agreement,
    )
    return session


def test_an_assessment_item_stores_the_triage_score_as_its_own_row() -> None:
    session = _rate("assessment", 2)
    ratings = [o for o in session.added if isinstance(o, RubricRating)]
    triage = [r for r in ratings if r.domain_code == TRIAGE_AGREEMENT_DOMAIN_CODE]
    assert len(ratings) == len(RUBRIC_DOMAIN_CODES) + 1
    assert [r.score for r in triage] == [2]


def test_an_assessment_item_refuses_a_rating_without_the_triage_score() -> None:
    with pytest.raises(wf.TriageAgreementError):
        _rate("assessment", None)


def test_other_areas_refuse_a_triage_score() -> None:
    with pytest.raises(wf.TriageAgreementError):
        _rate("investigations", 3)


def test_other_areas_rate_without_it() -> None:
    session = _rate("severity_classification", None)
    ratings = [o for o in session.added if isinstance(o, RubricRating)]
    assert len(ratings) == len(RUBRIC_DOMAIN_CODES)


@pytest.mark.parametrize("bad", [0, 6])
def test_the_request_schema_bounds_the_score(bad: int) -> None:
    with pytest.raises(ValidationError):
        RatingRoundRequest(scores=_scores(), accept_action="full_accept", triage_agreement=bad)


def test_the_vocabulary_has_a_difficulty_breathing_concept() -> None:
    from app.records.concepts import load_vocabulary  # noqa: PLC0415

    vocab = load_vocabulary(REPO / "data" / "clinical_concepts.yaml")
    concept = next(
        c for c in vocab.concepts if c.field == "examination_findings.difficulty_breathing"
    )
    assert concept.operator == "present"
