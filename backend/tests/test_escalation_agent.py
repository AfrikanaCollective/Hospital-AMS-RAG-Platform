"""Escalation agent (ARCH §10.2, §12)."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass

import app.agents.escalation_agent as ea
from app.schemas.enums import EscalationTrigger


@contextmanager
def _fake_session_scope():
    yield None


@dataclass
class _FakeEscalation:
    id: uuid.UUID


def test_run_persists_escalation_and_sets_escalation_id(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(ea, "_SESSION_SCOPE", _fake_session_scope)
    captured = {}

    def _fake_create(session, **kwargs):  # noqa: ANN001, ARG001
        captured.update(kwargs)
        return _FakeEscalation(id=uuid.uuid4())

    monkeypatch.setattr(ea, "_CREATE_ESCALATION_FN", _fake_create)
    monkeypatch.setattr(ea, "_notify_reviewers", lambda eid, code: None)  # noqa: ARG005

    state = {
        "escalation": {"trigger_code": EscalationTrigger.LOW_CONFIDENCE, "message": "too low"},
        "candidate_segments": [{"type": "framing", "text": "draft"}],
    }
    out = ea.run(state)  # type: ignore[arg-type]
    assert captured["trigger_code"] == EscalationTrigger.LOW_CONFIDENCE
    assert captured["candidate_answer"] == "draft"
    assert "escalation_id" in out["escalation"]


def test_notify_reviewers_is_a_noop_without_webhook_configured() -> None:
    # No REVIEW_WEBHOOK_URL in the test env -> must not raise or try a real HTTP call.
    ea._notify_reviewers(uuid.uuid4(), "low_confidence")


def test_offline_run_does_not_record_an_escalation(monkeypatch) -> None:  # noqa: ANN001
    """Offline question generation (DEVIATIONS.md #266): the escalation stays
    in the output, but no HITL row is written and no reviewer is notified."""
    calls: list = []
    monkeypatch.setattr(ea, "_CREATE_ESCALATION_FN", lambda *a, **k: calls.append("create"))
    monkeypatch.setattr(ea, "_notify_reviewers", lambda *a, **k: calls.append("notify"))
    state = {
        "conversation_id": str(uuid.uuid4()),
        "purpose": "ablation_holdout",
        "record_escalations": False,
        "escalation": {"trigger_code": EscalationTrigger.GROUNDING_FAILURE, "message": "m"},
    }
    out = ea.run(state)  # type: ignore[arg-type]
    assert calls == []
    assert out["escalation"]["trigger_code"] == EscalationTrigger.GROUNDING_FAILURE
    assert out["escalation"]["escalation_id"] is None and out["escalation"]["recorded"] is False
