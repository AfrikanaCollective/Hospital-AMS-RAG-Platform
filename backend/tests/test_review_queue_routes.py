"""`/review-queue/*` HTTP routes (PRD-042, PRD-047; ARCH §14.2)."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.api.deps import Principal, current_principal, get_db
from app.api.routes.rubric import get_queue_store
from app.db.models.eval import EvalQuestion, RatingRound, Result
from app.main import app as fastapi_app
from tests.review_queue_fakes import FakeStore

RATER_ID = uuid.uuid4()


class _FakeSession:
    def __init__(
        self,
        results: list[Result],
        rounds: list[RatingRound],
        questions: list[EvalQuestion] | None = None,
    ) -> None:
        self._results = results
        self._rounds = rounds
        self._questions = questions or []

    def get(self, model: object, pk: object) -> object:
        rows = self._questions if model is EvalQuestion else self._results
        return next((r for r in rows if r.id == pk), None)

    def execute(self, stmt: object) -> object:
        descriptions = stmt.column_descriptions  # type: ignore[attr-defined]
        entity = descriptions[0]["entity"]

        class _Res:
            def __init__(self, rows: list) -> None:
                self._rows = rows

            def scalars(self) -> _Res:
                return self

            def all(self) -> list:
                return self._rows

        if entity is RatingRound:
            return _Res(self._rounds)
        if len(descriptions) > 1:
            # app.rubric.workflow.list_queue_candidates's tuple-select
            # (Result.id, Result.created_at, distinct rater count) — a
            # different shape from the whole-entity `select(Result)` below,
            # so it needs its own branch here. Replicates that function's
            # filters in plain Python: `queue_state == "open"`, excluding
            # anything the calling rater (RATER_ID — the only principal any
            # test in this module acts as) has already rated.
            already_rated = {rr.result_id for rr in self._rounds if rr.rater_id == RATER_ID}
            rows = [
                (
                    r.id,
                    r.created_at,
                    len({rr.rater_id for rr in self._rounds if rr.result_id == r.id}),
                )
                for r in self._results
                if r.queue_state == "open" and r.id not in already_rated
            ]
            return _Res(rows)
        # The only remaining (whole-entity) Result select in these routes
        # fetches by id, after `list_queue_candidates` has already filtered.
        return _Res([r for r in self._results if r.queue_state == "open"])


@pytest.fixture(autouse=True)
def _principal_override():
    fastapi_app.dependency_overrides[current_principal] = lambda: Principal(
        user_id=str(RATER_ID), roles=frozenset({"reviewer"}), is_clinician=True
    )
    yield
    fastapi_app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(fastapi_app)


def _store_for(results: list[Result]) -> FakeStore:
    """A queue store whose results are exactly `results` (all open, unrated)."""
    store = FakeStore(0)
    for i, r in enumerate(results):
        store.results.append(r.id)
        store.created[r.id] = datetime(2026, 10, 1, 0, i, tzinfo=UTC)
        store.ratings[r.id] = set()
    return store


def test_queue_lists_the_reviewers_batch(client: TestClient) -> None:
    """DEVIATIONS.md #294: the list is `app.rubric.assignment`'s view."""
    a = Result(
        id=uuid.uuid4(),
        provenance="auto_generated",
        expected_outcome="well_supported",
        queue_state="open",
        created_at=datetime.now(UTC),
    )
    b = Result(id=uuid.uuid4(), provenance="auto_generated", queue_state="open")
    store = _store_for([a, b])
    fastapi_app.dependency_overrides[get_db] = lambda: _FakeSession([a, b], [])
    fastapi_app.dependency_overrides[get_queue_store] = lambda: store

    body = client.get("/api/review-queue").json()
    assert body["mode"] == "batch" and body["batch"] == {"done": 0, "total": 2}
    assert [i["result_id"] for i in body["items"]] == [str(a.id), str(b.id)]
    assert body["items"][0]["expected_outcome"] == "well_supported"
    assert body["draft_result_id"] is None and body["draft_dropped"] is False


def test_draft_routes_save_load_and_lock(client: TestClient) -> None:
    a = Result(id=uuid.uuid4(), provenance="auto_generated", queue_state="open")
    b = Result(id=uuid.uuid4(), provenance="auto_generated", queue_state="open")
    store = _store_for([a, b])
    fastapi_app.dependency_overrides[get_db] = lambda: _FakeSession([a, b], [])
    fastapi_app.dependency_overrides[get_queue_store] = lambda: store

    assert client.get("/api/review-queue/draft").status_code == 404
    payload = {"scores": {"medical_consensus_alignment": 4}, "step": 1}
    ok = client.put("/api/review-queue/draft", json={"result_id": str(a.id), "payload": payload})
    assert ok.status_code == 200
    assert client.get("/api/review-queue/draft").json() == {
        "result_id": str(a.id),
        "payload": payload,
    }
    body = client.get("/api/review-queue").json()
    assert body["mode"] == "draft" and body["draft_result_id"] == str(a.id)
    assert [i["result_id"] for i in body["items"]] == [str(a.id)]
    other = client.put("/api/review-queue/draft", json={"result_id": str(b.id), "payload": {}})
    assert other.status_code == 409


def test_get_queue_item_hides_other_raters_scores(client: TestClient) -> None:
    result = Result(
        id=uuid.uuid4(),
        provenance="auto_generated",
        queue_state="open",
        answer_enc=None,
        citations=[{"citation_id": "c1"}],
        grounding_report={"action": "release"},
        created_at=datetime.now(UTC),
    )
    session = _FakeSession([result], [])
    fastapi_app.dependency_overrides[get_db] = lambda: session

    resp = client.get(f"/api/review-queue/{result.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["citations"] == [{"citation_id": "c1"}]
    assert body["question"] is None
    assert body["segments"] is None
    assert "scores" not in body
    assert "rater_id" not in body


def test_get_queue_item_includes_generated_question_and_segments(client: TestClient) -> None:
    """The review-queue detail view surfaces the auto-generated question the
    result answers, and the per-segment citation_ids the answer's inline
    citations need (DEVIATIONS.md #120) — both absent before this change."""
    from app.crypto.provider import get_crypto
    from app.db.models.eval import result_answer_aad, result_segments_aad

    question = EvalQuestion(
        id=uuid.uuid4(),
        text="A 3-day-old neonate has a fever of 38.5C. What does the guideline recommend?",
        provenance="auto_generated",
        expected_outcome="well_supported",
    )
    result_id = uuid.uuid4()
    crypto = get_crypto()
    segments = [
        {
            "type": "claim",
            "text": "Neonates with signs of sepsis should be treated with ampicillin.",
            "citation_ids": ["c1"],
            "grounding_note": None,
        }
    ]
    result = Result(
        id=result_id,
        eval_question_id=question.id,
        provenance="auto_generated",
        queue_state="open",
        answer_enc=crypto.encrypt(
            b"Neonates with signs of sepsis should be treated with ampicillin.",
            aad=result_answer_aad(result_id),
        ),
        answer_segments_enc=crypto.encrypt(
            json.dumps(segments).encode("utf-8"), aad=result_segments_aad(result_id)
        ),
        citations=[{"citation_id": "c1"}],
        grounding_report={"action": "release"},
        created_at=datetime.now(UTC),
    )
    session = _FakeSession([result], [], questions=[question])
    fastapi_app.dependency_overrides[get_db] = lambda: session

    resp = client.get(f"/api/review-queue/{result.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["question"] == question.text
    assert body["segments"] == segments


def test_get_queue_item_exposes_multi_stage_audit_fields_read_only(client: TestClient) -> None:
    """DEVIATIONS.md #186: the multi-stage (vocabulary-augmented) audit
    counterpart is exposed for traceability/comparison, decrypted the same
    way as the single-stage answer -- but only ever as a nested, clearly
    separate `multi_stage` key, never merged into the fields a rater's
    score is actually based on."""
    from app.crypto.provider import get_crypto
    from app.db.models.eval import result_multi_stage_answer_aad

    result_id = uuid.uuid4()
    crypto = get_crypto()
    result = Result(
        id=result_id,
        provenance="auto_generated",
        queue_state="open",
        citations=[{"citation_id": "c1"}],
        grounding_report={"action": "release"},
        multi_stage_query="augmented question text; fake fast breathing",
        multi_stage_fired=True,
        multi_stage_answer_enc=crypto.encrypt(
            b"Multi-stage answer text.", aad=result_multi_stage_answer_aad(result_id)
        ),
        multi_stage_citations=[{"citation_id": "c2"}],
        multi_stage_grounding_report={"action": "release"},
        created_at=datetime.now(UTC),
    )
    session = _FakeSession([result], [])
    fastapi_app.dependency_overrides[get_db] = lambda: session

    resp = client.get(f"/api/review-queue/{result.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["multi_stage"] == {
        "query": "augmented question text; fake fast breathing",
        "fired": True,
        "answer": "Multi-stage answer text.",
        "citations": [{"citation_id": "c2"}],
        "grounding_report": {"action": "release"},
    }
    # The single-stage fields a rater's score is based on are untouched.
    assert body["citations"] == [{"citation_id": "c1"}]


def test_get_queue_item_multi_stage_absent_when_not_computed(client: TestClient) -> None:
    """No vocabulary attested for the run that created this result -- the
    multi_stage sub-object's fields fall back to their plain defaults, not
    an error or a missing key. `multi_stage_fired`/`citations`/
    `grounding_report` are passed explicitly here (rather than left off) to
    match what a REAL fetched row looks like once the ORM/migration column
    defaults have actually resolved (`False`/`[]`/`{}`) -- constructing a
    `Result` directly in Python, as this fake session does, otherwise leaves
    those `None` since a plain `default=` only applies at insert time, an
    artifact of this test's own construction, not real row shape."""
    result = Result(
        id=uuid.uuid4(),
        provenance="auto_generated",
        queue_state="open",
        citations=[{"citation_id": "c1"}],
        multi_stage_fired=False,
        multi_stage_citations=[],
        multi_stage_grounding_report={},
        created_at=datetime.now(UTC),
    )
    session = _FakeSession([result], [])
    fastapi_app.dependency_overrides[get_db] = lambda: session

    resp = client.get(f"/api/review-queue/{result.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["multi_stage"] == {
        "query": None,
        "fired": False,
        "answer": None,
        "citations": [],
        "grounding_report": {},
    }


def test_get_queue_item_not_queued_is_404(client: TestClient) -> None:
    result = Result(
        id=uuid.uuid4(),
        provenance="auto_generated",
        queue_state="not_queued",
        created_at=datetime.now(UTC),
    )
    session = _FakeSession([result], [])
    fastapi_app.dependency_overrides[get_db] = lambda: session

    resp = client.get(f"/api/review-queue/{result.id}")
    assert resp.status_code == 404


def test_queue_and_item_show_the_query_area(client: TestClient) -> None:
    """DEVIATIONS.md #290: the four-area query area from generator_meta."""
    question = EvalQuestion(
        id=uuid.uuid4(),
        text="What does the guideline recommend about ...",
        provenance="auto_generated",
        generator_meta={"area": "antibiotic_course", "purpose": "review_queue"},
    )
    with_area = Result(
        id=uuid.uuid4(),
        provenance="auto_generated",
        queue_state="open",
        eval_question_id=question.id,
        created_at=datetime.now(UTC),
    )
    no_question = Result(
        id=uuid.uuid4(),
        provenance="clinician_submitted",
        queue_state="open",
        created_at=datetime.now(UTC),
    )
    session = _FakeSession([with_area, no_question], [], questions=[question])
    fastapi_app.dependency_overrides[get_db] = lambda: session
    fastapi_app.dependency_overrides[get_queue_store] = lambda: _store_for([with_area, no_question])

    rows = {row["result_id"]: row for row in client.get("/api/review-queue").json()["items"]}
    assert rows[str(with_area.id)]["query_area"] == "antibiotic_course"
    assert rows[str(no_question.id)]["query_area"] is None
    item = client.get(f"/api/review-queue/{with_area.id}").json()
    assert item["query_area"] == "antibiotic_course"


def test_preview_account_lists_every_open_case_and_cannot_save(client: TestClient) -> None:
    """DEVIATIONS.md #323: listed, never assigned, 403 on a draft save."""
    results = [
        Result(id=uuid.uuid4(), provenance="auto_generated", queue_state="open") for _ in range(15)
    ]
    store = _store_for(results)
    store.preview.add(RATER_ID)
    fastapi_app.dependency_overrides[get_db] = lambda: _FakeSession(results, [])
    fastapi_app.dependency_overrides[get_queue_store] = lambda: store

    body = client.get("/api/review-queue").json()
    assert body["mode"] == "preview" and body["batch"] is None
    assert [i["result_id"] for i in body["items"]] == [str(r.id) for r in results]
    put = client.put(
        "/api/review-queue/draft", json={"result_id": str(results[0].id), "payload": {}}
    )
    assert put.status_code == 403
    assert client.get("/api/review-queue/draft").status_code == 404
    assert store.items == [] and store.drafts == {}
