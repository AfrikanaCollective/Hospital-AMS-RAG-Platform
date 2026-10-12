"""Multi-rater workflow (ARCH §14.2-§14.5; PRD-042, PRD-043, PRD-047)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

import app.audit.log as audit_log
import app.rubric.workflow as wf
from app.config import get_settings
from app.db.models.eval import RatingRound, Result, RubricRating
from app.rubric.domains import RUBRIC_DOMAIN_CODES
from app.schemas.enums import HitlAcceptAction
from app.schemas.rubric import DomainScore

RESULT_ID = uuid.uuid4()
N_DOMAINS = len(RUBRIC_DOMAIN_CODES)


class _FakeSession:
    """Generic: assigns a uuid to anything added with `id is None` on flush
    (matching `tests/test_hitl_decisions.py`'s pattern) — `submit_rating` now
    also creates a `HitlDecision` + `AuditEvent` via
    `app.hitl.decisions.apply_rating_accept_action` (ARCH §13.2 "Both axes
    together"), not just `RatingRound`/`RubricRating`."""

    def __init__(self, result: Result) -> None:
        self._result = result
        self.added: list = []

    @property
    def rounds(self) -> list[RatingRound]:
        return [o for o in self.added if isinstance(o, RatingRound)]

    @property
    def ratings(self) -> list[RubricRating]:
        return [o for o in self.added if isinstance(o, RubricRating)]

    @property
    def archives(self) -> list:
        return [o for o in self.added if type(o).__name__ == "ResultArchive"]

    @property
    def irr_scores(self) -> list:
        return [o for o in self.added if type(o).__name__ == "IRRScore"]

    def get(self, model: object, pk: object) -> object:  # noqa: ARG002
        return self._result if pk == self._result.id else None

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

        rows = self.rounds if entity is RatingRound else self.ratings
        return _Res(rows)


@pytest.fixture(autouse=True)
def _no_real_audit_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit_log, "_fetch_last_row_hash", lambda session: None)  # noqa: ARG005


def _full_scores(base: int = 4) -> list[DomainScore]:
    return [DomainScore(domain_code=c, score=base) for c in RUBRIC_DOMAIN_CODES]


def _result() -> Result:
    return Result(
        id=RESULT_ID,
        provenance="clinician_submitted",
        queue_state="not_queued",
        created_at=datetime.now(UTC),
    )


def _submit(session, **kwargs):  # noqa: ANN001, ANN201
    """`submit_rating` wrapper defaulting the now-required accept axis to a
    reason-free action, so tests focused on the rank-mode mechanics don't
    each have to spell it out."""
    kwargs.setdefault("accept_action", HitlAcceptAction.FULL_ACCEPT)
    return wf.submit_rating(session, **kwargs)


def test_first_rating_opens_the_queue() -> None:
    result = _result()
    session = _FakeSession(result)
    _submit(session, result_id=RESULT_ID, rater_id=uuid.uuid4(), scores=_full_scores())
    assert result.queue_state == "open"
    assert len(session.ratings) == N_DOMAINS


def test_duplicate_rater_rejected() -> None:
    result = _result()
    session = _FakeSession(result)
    rater = uuid.uuid4()
    _submit(session, result_id=RESULT_ID, rater_id=rater, scores=_full_scores())
    with pytest.raises(wf.DuplicateRaterError):
        _submit(session, result_id=RESULT_ID, rater_id=rater, scores=_full_scores())


def test_unknown_result_raises() -> None:
    session = _FakeSession(_result())
    with pytest.raises(wf.ResultNotFoundError):
        _submit(session, result_id=uuid.uuid4(), rater_id=uuid.uuid4(), scores=_full_scores())


def test_third_distinct_rater_triggers_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRR_MIN_RATERS", "3")
    get_settings.cache_clear()
    result = _result()
    session = _FakeSession(result)
    _submit(session, result_id=RESULT_ID, rater_id=uuid.uuid4(), scores=_full_scores(4))
    assert result.queue_state == "open"
    _submit(session, result_id=RESULT_ID, rater_id=uuid.uuid4(), scores=_full_scores(4))
    assert result.queue_state == "open"
    _submit(session, result_id=RESULT_ID, rater_id=uuid.uuid4(), scores=_full_scores(5))
    assert result.queue_state == "archived"
    assert len(session.irr_scores) == N_DOMAINS
    assert len(session.archives) == 1
    get_settings.cache_clear()


def test_an_archived_result_refuses_a_fourth_rating(monkeypatch: pytest.MonkeyPatch) -> None:
    """DEVIATIONS.md #294: e.g. a draft submitted after others finished it."""
    monkeypatch.setenv("IRR_MIN_RATERS", "3")
    get_settings.cache_clear()
    result = _result()
    session = _FakeSession(result)
    for _ in range(3):
        _submit(session, result_id=RESULT_ID, rater_id=uuid.uuid4(), scores=_full_scores(4))
    assert result.queue_state == "archived"
    with pytest.raises(wf.ResultArchivedError):
        _submit(session, result_id=RESULT_ID, rater_id=uuid.uuid4(), scores=_full_scores(4))
    get_settings.cache_clear()


def test_accept_action_is_required() -> None:
    """ARCH §13.2 "Both axes together": every rating submission records an
    accept-axis decision too — `submit_rating` has no default for it."""
    session = _FakeSession(_result())
    with pytest.raises(TypeError):
        wf.submit_rating(  # type: ignore[call-arg]
            session, result_id=RESULT_ID, rater_id=uuid.uuid4(), scores=_full_scores()
        )


def test_accept_action_links_rating_round_to_hitl_decision() -> None:
    """A ranker's task is exactly two things — the rubric and the accept-axis
    pick, no reason code (DEVIATIONS.md #100) — so `reject` succeeds here
    with no `accept_reason_code` at all."""
    result = _result()
    session = _FakeSession(result)
    round_row = _submit(
        session,
        result_id=RESULT_ID,
        rater_id=uuid.uuid4(),
        scores=_full_scores(),
        accept_action=HitlAcceptAction.REJECT,
    )
    assert round_row.accept_action_id is not None
    decisions = [o for o in session.added if type(o).__name__ == "HitlDecision"]
    assert len(decisions) == 1
    assert decisions[0].id == round_row.accept_action_id
    assert decisions[0].action == "reject"
    assert decisions[0].reason_code is None
    assert decisions[0].escalation_id is None  # not tied to any escalation (DEVIATIONS #99)


def test_submit_rating_never_creates_an_escalation() -> None:
    """DEVIATIONS.md #100: "no further escalation" — a rank-mode submission,
    whatever the accept-axis pick, must never create a `hitl.escalation` row.
    Structurally true (`apply_rating_accept_action` always passes
    `escalation_id=None`); asserted here so a future change that broke it
    would fail a test, not just a code review. Also exercises every action,
    including `partial_accept`, with no `accept_edited_answer` at all
    (DEVIATIONS.md #101: a ranker is never required to write anything)."""
    # A fresh result per action: a result is archived at its third rating
    # and then refuses more (DEVIATIONS.md #294).
    for action in HitlAcceptAction:
        session = _FakeSession(_result())
        _submit(
            session,
            result_id=RESULT_ID,
            rater_id=uuid.uuid4(),
            scores=_full_scores(),
            accept_action=action,
        )
        assert not any(type(o).__name__ == "Escalation" for o in session.added)


def test_submit_rating_trusts_already_validated_accept_fields() -> None:
    """`submit_rating` doesn't itself re-validate the accept-axis fields — it
    trusts an already-validated request (the HTTP layer's `RatingRoundRequest`
    is the single source of truth for what's required) and just passes them
    through to `apply_rating_accept_action`. Exercised here with an edited
    answer volunteered anyway, to confirm that still-optional path works."""
    result = _result()
    session = _FakeSession(result)
    round_row = _submit(
        session,
        result_id=RESULT_ID,
        rater_id=uuid.uuid4(),
        scores=_full_scores(),
        accept_action=HitlAcceptAction.PARTIAL_ACCEPT,
        accept_edited_answer="edited text",
    )
    decision = next(o for o in session.added if type(o).__name__ == "HitlDecision")
    assert round_row.accept_action_id == decision.id
    assert decision.edited_answer_enc is not None


def test_submit_rating_partial_accept_succeeds_without_edited_answer() -> None:
    """DEVIATIONS.md #101: `accept_accepted_context_ids` alone is a complete,
    valid `partial_accept` — no edited-answer text required."""
    result = _result()
    session = _FakeSession(result)
    round_row = _submit(
        session,
        result_id=RESULT_ID,
        rater_id=uuid.uuid4(),
        scores=_full_scores(),
        accept_action=HitlAcceptAction.PARTIAL_ACCEPT,
    )
    decision = next(o for o in session.added if type(o).__name__ == "HitlDecision")
    assert round_row.accept_action_id == decision.id
    assert decision.edited_answer_enc is None
