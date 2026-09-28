"""Orchestrator / supervisor agent (ARCH §10.2).

Does: scope-classify the query; route to sub-agents; assemble the final
response; inject the non-removable disclaimer (ARCH-037); own the
escalate-vs-release decision; enforce the segment structure (ARCH §8.2).
Does NOT: retrieve; read PHI values; generate guideline claims; resolve
source conflicts.
Access: memory.conversation, hitl.escalation (write), patient_id handle only.
Tools: classify_scope, dispatch, assemble_response, apply_disclaimer, open_escalation.

This node is visited TWICE in the compiled graph (`app.agents.graph`): once
on entry (classify + dispatch) and once after `citation_verifier` succeeds
(assemble + disclaimer). It tells the two visits apart by what's already in
state — `"scope_label" not in state`, i.e. first visit, has not classified
yet — rather than needing a separate node name, since LangGraph's routing
after this node (`app.agents.graph._route_from_orchestrator`) already has to
inspect the very same state to decide where to go next either way.
"""

from __future__ import annotations

from app.agents.state import GraphState
from app.grounding.wording import scan_segment
from app.schemas.citation import correction_notices
from app.schemas.enums import EscalationTrigger, ObservedOutcome, ScopeLabel, SegmentType
from app.schemas.query import DISCLAIMER_TEXT
from app.scope.classifier import classify_scope

OUT_OF_SCOPE_TEXT = (
    "This question is outside what this system answers: it reports and cites retrieved "
    "clinical guideline content and structured patient-record inference only."
)


def _classify_and_dispatch(state: GraphState) -> GraphState:
    has_patient = bool(state.get("patient_id"))
    label = classify_scope(state["query"], has_patient=has_patient)
    state["scope_label"] = label

    if label == ScopeLabel.SCOPE_2_EXCLUDED:
        state["escalation"] = {
            "trigger_code": EscalationTrigger.SCOPE_BOUNDARY,
            "message": (
                "This request would require synthesising patient data with guideline "
                "content into a directive recommendation, or adjusting guideline content "
                "for a local constraint — this system does not perform either. "
                "Routing to clinician review."
            ),
        }
    elif label == ScopeLabel.OUT_OF_SCOPE:
        state["candidate_segments"] = [{"type": SegmentType.FRAMING, "text": OUT_OF_SCOPE_TEXT}]
        state["candidate_citations"] = []
        state["observed_outcome"] = ObservedOutcome.NO_GUIDELINE
        _finalize(state)
    return state


def _finalize(state: GraphState) -> GraphState:
    segments = state.get("candidate_segments") or []
    citations = state.get("candidate_citations") or []

    # Second, deterministic wording pass over the assembled framing text
    # (ARCH-037 second line of defense) — claim segments were already
    # scanned by the grounding gate; this catches anything in framing text
    # this orchestrator itself composed (e.g. OUT_OF_SCOPE_TEXT edits later).
    for seg in segments:
        is_dirty_framing = seg.get("type") == SegmentType.FRAMING and scan_segment(
            seg.get("text", ""), cited_quotes=[]
        )
        if is_dirty_framing:
            state["escalation"] = {
                "trigger_code": EscalationTrigger.SAFETY_FILTER,
                "message": "Assembled response failed the wording safety check.",
            }
            return state

    observed = state.get("observed_outcome")
    if observed is None:
        if state.get("missing_info"):
            # SCOPE-2.2: the released answer's substance is a missing-info
            # request (DEVIATIONS.md #76) — distinct from an ordinary
            # well-supported claim answer for eval-harness scoring purposes
            # (ARCH §16.1 scores missing_info_expected against this).
            observed = ObservedOutcome.MISSING_INFO
        else:
            has_claim = any(s.get("type") == SegmentType.CLAIM for s in segments)
            observed = ObservedOutcome.WELL_SUPPORTED if has_claim else ObservedOutcome.NO_GUIDELINE
    state["observed_outcome"] = observed

    state["final_answer"] = {
        "segments": [dict(s) for s in segments],
        "citations": list(citations),
        "disclaimer": DISCLAIMER_TEXT,
        # ARCH-044 §5.11: attested source corrections are always shown.
        "correction_notices": correction_notices(list(citations)),
    }
    return state


def run(state: GraphState) -> GraphState:
    if "scope_label" not in state:
        return _classify_and_dispatch(state)
    if state.get("escalation"):
        # Escalated (whether or not it has been persisted yet) — the graph
        # routes this to the escalation node, not here; never release an
        # answer once an escalation has been raised for this turn.
        return state
    return _finalize(state)
