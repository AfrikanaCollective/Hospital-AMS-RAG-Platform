"""Escalation agent (ARCH §10.2, §12).

Does: package an escalation — reason code, trigger detail, candidate answer,
retrieved evidence, patient-context handle; create the hitl.escalation row;
enqueue for review; notify reviewers.
Does NOT: answer the clinical question; alter the candidate.
Access: hitl schema (write), memory.conversation (read).
Tools: create_escalation, enqueue_review, notify_reviewers.

`state["escalation"]` (set by an earlier node — orchestrator/retrieval/
guideline-synthesis/citation-verifier/stage-classifier/missing-info — always
carries at least `trigger_code` + `message`) is the input; this node's job is
purely to persist it and produce the escalation id the orchestrator surfaces
to the caller (`EscalationInfo.escalation_id` in `QueryResponse`). It never
inspects or changes *why* the escalation happened.

`notify_reviewers` (ARCH §12.1's "notify reviewers" tool): `REVIEW_WEBHOOK_URL`
is unset by default (no dev/CI webhook target), so this call is a documented
no-op when unset rather than a hard dependency (DEVIATIONS.md #73) — reviewers
still see the escalation via the open queue (`GET /review-queue`) regardless.

Offline question generation (the ablation holdout, DEVIATIONS.md #266) runs
the real pipeline with `record_escalations=False`: the escalation stays in the
graph output (so that question gets no gold set), but no `hitl.escalation`
row is created and no reviewer is notified, since no clinician is waiting on
an answer. Every live path leaves the flag unset (recorded).
"""

from __future__ import annotations

import uuid

import httpx

from app.agents.state import GraphState
from app.config import get_settings
from app.db.session import session_scope
from app.hitl.escalation import create_escalation
from app.logging import get_logger

logger = get_logger(__name__)

_SESSION_SCOPE = session_scope
_CREATE_ESCALATION_FN = create_escalation


def _notify_reviewers(escalation_id: uuid.UUID, trigger_code: str) -> None:
    webhook = get_settings().review_webhook_url
    if not webhook:
        logger.info(
            "hitl_escalation_created_no_webhook",
            escalation_id=str(escalation_id),
            trigger_code=trigger_code,
        )
        return
    payload = {"escalation_id": str(escalation_id), "trigger_code": trigger_code}
    try:
        httpx.post(webhook, json=payload, timeout=5.0)
    except httpx.HTTPError as exc:
        logger.warning("hitl_reviewer_notification_failed", error=str(exc))


def run(state: GraphState) -> GraphState:
    escalation_info = state.get("escalation") or {}
    trigger_code = escalation_info.get("trigger_code") or "unknown"
    if state.get("record_escalations") is False:
        logger.info(
            "hitl_escalation_not_recorded_offline_run",
            trigger_code=trigger_code,
            purpose=state.get("purpose"),
        )
        state["escalation"] = {**escalation_info, "escalation_id": None, "recorded": False}
        return state
    candidate_answer = None
    segments = state.get("candidate_segments")
    if segments:
        candidate_answer = "\n".join(seg.get("text", "") for seg in segments)

    roles = state.get("roles")
    actor_role = roles[0] if roles else None
    with _SESSION_SCOPE() as session:
        escalation = _CREATE_ESCALATION_FN(
            session,
            trigger_code=trigger_code,
            trigger_detail={
                "message": escalation_info.get("message"),
                "grounding_report": state.get("grounding_report"),
            },
            conversation_id=uuid.UUID(state["conversation_id"])
            if state.get("conversation_id")
            else None,
            candidate_answer=candidate_answer,
            actor_id=uuid.UUID(state["user_id"]) if state.get("user_id") else None,
            actor_role=actor_role,
            purpose=state.get("purpose"),
        )
        escalation_id = escalation.id

    _notify_reviewers(escalation_id, trigger_code)
    state["escalation"] = {**escalation_info, "escalation_id": str(escalation_id)}
    return state
