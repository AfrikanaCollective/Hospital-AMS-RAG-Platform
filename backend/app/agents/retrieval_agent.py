"""Retrieval agent (ARCH §10.2, §7).

Does: build queries; hybrid search (dense + sparse) + RRF fusion + cross-encoder
rerank; assess confidence; detect conflicts; expand context.
Does NOT: read PHI; synthesize prose; decide the final answer.
Access: Qdrant guideline collections + corpus schema (read). NO PHI.
Tools: hybrid_search, rerank, expand_context, list_corpus_topics, get_chunk,
get_version_status.
"""

from __future__ import annotations

import uuid

from app.agents.state import GraphState
from app.db.session import session_scope
from app.retrieval.hybrid import retrieve

# Indirection points for tests: monkeypatch to avoid a real Qdrant/embedding
# call, and/or a real DB connection.
_RETRIEVE_FN = retrieve
_SESSION_SCOPE = session_scope


def run(state: GraphState) -> GraphState:
    patient_id = state.get("patient_id")
    conversation_id = state.get("conversation_id")
    user_id = state.get("user_id")
    roles = state.get("roles")
    actor_role = roles[0] if roles else None
    with _SESSION_SCOPE() as session:
        items, snapshot = _RETRIEVE_FN(
            state["query"],
            session=session,
            conversation_id=uuid.UUID(conversation_id) if conversation_id else None,
            patient_id=uuid.UUID(patient_id) if patient_id else None,
            actor_id=uuid.UUID(user_id) if user_id else None,
            actor_role=actor_role,
            purpose=state.get("purpose"),
            per_guideline_cap=state.get("per_guideline_cap"),
        )
    state["retrieval"] = items
    state["retrieval_confidence"] = {
        **snapshot["confidence"],
        "conflicts": snapshot.get("conflicts", []),
    }
    return state
