"""`/rubric/*` HTTP routes (PRD-040..PRD-044; ARCH §14)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import app.audit.log as audit_log
from app.api.deps import Principal, current_principal, get_db
from app.api.routes.rubric import get_queue_store
from app.db.models.eval import IRRScore, RatingRound, Result, RubricRating
from app.main import app as fastapi_app
from app.rubric.domains import RUBRIC_DOMAIN_CODES
from tests.review_queue_fakes import FakeStore

RESULT_ID = uuid.uuid4()


class _FakeSession:
    """Generic `add()` (matching `tests/test_hitl_decisions.py`'s pattern):
    `submit_rating` now also creates a `HitlDecision` + `AuditEvent` via
    `apply_rating_accept_action` (ARCH §13.2 "Both axes together"), not just
    `RatingRound`/`RubricRating`/`IRRScore`."""

    def __init__(self) -> None:
        self.result = Result(
            id=RESULT_ID,
            provenance="clinician_submitted",
            queue_state="not_queued",
            created_at=datetime.now(UTC),
        )
        self.added: list = []

    @property
    def rounds(self) -> list[RatingRound]:
        return [o for o in self.added if isinstance(o, RatingRound)]

    @property
    def ratings(self) -> list[RubricRating]:
        return [o for o in self.added if isinstance(o, RubricRating)]

    @property
    def irr(self) -> list[IRRScore]:
        return [o for o in self.added if isinstance(o, IRRScore)]

    def get(self, model: object, pk: object) -> object:  # noqa: ARG002
        return self.result if pk == self.result.id else None

    def add(self, obj: object) -> None:
        self.added.append(obj)

    def flush(self) -> None:
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()

    def execute(self, stmt: object) -> object:
        entity = stmt.column_descriptions[0]["entity"]  # type: ignore[attr-defined]

        class _Res:
            def __init__(self, rows: list) -> None:
                self._rows = rows

            def scalars(self) -> _Res:
                return self

            def all(self) -> list:
                return self._rows

        table = {RatingRound: self.rounds, RubricRating: self.ratings, IRRScore: self.irr}
        return _Res(table.get(entity, []))


@pytest.fixture(autouse=True)
def _overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(audit_log, "_fetch_last_row_hash", lambda session: None)  # noqa: ARG005
    session = _FakeSession()
    fastapi_app.dependency_overrides[get_db] = lambda: session
    # The queue check (DEVIATIONS.md #294) sees no draft and no open items:
    # RESULT_ID is not yet queued, so a first rating may open it.
    fastapi_app.dependency_overrides[get_queue_store] = lambda: FakeStore(0)
    fastapi_app.dependency_overrides[current_principal] = lambda: Principal(
        user_id=str(uuid.uuid4()), roles=frozenset({"reviewer"}), is_clinician=True
    )
    yield session
    fastapi_app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(fastapi_app)


def test_list_domains_returns_eleven() -> None:
    resp = TestClient(fastapi_app).get("/api/rubric/domains")
    assert resp.status_code == 200
    assert len(resp.json()) == len(RUBRIC_DOMAIN_CODES)


def test_submit_rating_requires_all_domains(client: TestClient) -> None:
    resp = client.post(
        f"/api/rubric/results/{RESULT_ID}/ratings",
        json={
            "scores": [{"domain_code": "medical_consensus_alignment", "score": 5}],
            "accept_action": "full_accept",
        },
    )
    assert resp.status_code == 422


def test_submit_rating_requires_accept_action(client: TestClient) -> None:
    """ARCH §13.2 "Both axes together" — omitting it is now a 422, not an
    optional field."""
    scores = [{"domain_code": c, "score": 4} for c in RUBRIC_DOMAIN_CODES]
    resp = client.post(f"/api/rubric/results/{RESULT_ID}/ratings", json={"scores": scores})
    assert resp.status_code == 422


def test_submit_rating_reject_succeeds_without_reason_code(client: TestClient) -> None:
    """DEVIATIONS.md #100: a ranker's task is exactly the rubric + the
    accept-axis pick — no reason/justification text is asked for or
    required, unlike the standalone escalation-resolution workflow."""
    scores = [{"domain_code": c, "score": 4} for c in RUBRIC_DOMAIN_CODES]
    resp = client.post(
        f"/api/rubric/results/{RESULT_ID}/ratings",
        json={"scores": scores, "accept_action": "reject"},
    )
    assert resp.status_code == 200, resp.text


def test_submit_rating_partial_accept_succeeds_without_edited_answer(client: TestClient) -> None:
    """DEVIATIONS.md #101: a ranker is never required to write anything —
    `accept_accepted_context_ids` alone is a complete, valid partial_accept."""
    scores = [{"domain_code": c, "score": 4} for c in RUBRIC_DOMAIN_CODES]
    resp = client.post(
        f"/api/rubric/results/{RESULT_ID}/ratings",
        json={"scores": scores, "accept_action": "partial_accept"},
    )
    assert resp.status_code == 200, resp.text


def test_submit_rating_opens_queue(client: TestClient) -> None:
    scores = [{"domain_code": c, "score": 4} for c in RUBRIC_DOMAIN_CODES]
    resp = client.post(
        f"/api/rubric/results/{RESULT_ID}/ratings",
        json={"scores": scores, "accept_action": "full_accept"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "rating_round_id" in body
    assert "accept_action_id" in body


def test_submit_rating_unknown_result_404(client: TestClient) -> None:
    scores = [{"domain_code": c, "score": 4} for c in RUBRIC_DOMAIN_CODES]
    resp = client.post(
        f"/api/rubric/results/{uuid.uuid4()}/ratings",
        json={"scores": scores, "accept_action": "full_accept"},
    )
    assert resp.status_code == 404


def test_get_result_ratings_shape(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ARG001
    resp = client.get(f"/api/rubric/results/{RESULT_ID}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["result_id"] == str(RESULT_ID)
    assert body["distinct_rater_count"] == 0
    assert body["archived"] is False


def test_submit_rating_refused_while_a_draft_for_another_item_exists(client: TestClient) -> None:
    """DEVIATIONS.md #294: a half-finished rating must be submitted first."""
    store = FakeStore(1)
    rater = uuid.uuid4()
    fastapi_app.dependency_overrides[get_queue_store] = lambda: store
    fastapi_app.dependency_overrides[current_principal] = lambda: Principal(
        user_id=str(rater), roles=frozenset({"reviewer"}), is_clinician=True
    )
    from app.rubric.assignment import save_draft

    save_draft(store, rater, store.results[0], {"scores": {}}, batch_size=12)
    scores = [{"domain_code": c, "score": 4} for c in RUBRIC_DOMAIN_CODES]
    resp = client.post(
        f"/api/rubric/results/{RESULT_ID}/ratings",
        json={"scores": scores, "accept_action": "full_accept"},
    )
    assert resp.status_code == 409 and "saved rating first" in resp.json()["detail"]


def test_submit_rating_refused_for_a_preview_account(client: TestClient) -> None:
    """DEVIATIONS.md #323: a preview account's rating is never stored."""
    store = FakeStore(0)
    rater = uuid.uuid4()
    store.preview.add(rater)
    fastapi_app.dependency_overrides[get_queue_store] = lambda: store
    fastapi_app.dependency_overrides[current_principal] = lambda: Principal(
        user_id=str(rater), roles=frozenset({"reviewer"}), is_clinician=True
    )
    scores = [{"domain_code": c, "score": 4} for c in RUBRIC_DOMAIN_CODES]
    resp = client.post(
        f"/api/rubric/results/{RESULT_ID}/ratings",
        json={"scores": scores, "accept_action": "full_accept"},
    )
    assert resp.status_code == 403 and "preview account" in resp.json()["detail"]
