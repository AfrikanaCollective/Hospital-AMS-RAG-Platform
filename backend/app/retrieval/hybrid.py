"""Hybrid retrieval pipeline (ARCH §7).

1. query construction (curated abbreviation expansion only — never invent
   clinical content),
2. dense + sparse search in Qdrant with access filter (CANDIDATE_K each),
3. RRF fusion (server-side, Qdrant's Query API — FUSED_K),
4. cross-encoder rerank -> TOP_K,
5. confidence assessment (RETRIEVAL_MIN_SCORE / MIN_SUPPORTING_CHUNKS),
6. conflict detection (multi-version same-topic / lexical contradiction),
7. context expansion: NOT done here — `expand_context` (parent_chunk_id ->
   surrounding text) is a Phase-3 agent *tool*, invoked on demand by the
   synthesis agent, not a step this pipeline runs unconditionally,
8. return ranked items + confidence/conflict verdict; writes a minimal
   `retrieval` audit event when a DB session is supplied (DEVIATIONS.md #50 —
   the caller decides whether a session is available; offline unit tests pass
   none and skip the write).
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import TYPE_CHECKING

from app.agents.state import RetrievalItem
from app.audit.log import write_event
from app.config import get_settings
from app.ingestion.embed import embed_texts
from app.ingestion.review import NOT_RETRIEVABLE
from app.retrieval.confidence import assess
from app.retrieval.conflict import detect_conflicts
from app.retrieval.rerank import rerank
from app.retrieval.sparse import query_sparse_vector
from app.retrieval.vectorstore import QdrantVectorStore, VectorStore

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# Curated only — never invent clinical content (ARCH §7 step 1). Chosen for
# the bundled neonatal dev corpus (WHO newborn health / SBI in young infants,
# Kenya MOH newborn care protocols); extend deliberately, one confirmed term
# at a time (DEVIATIONS.md #55).
_ABBREVIATIONS: dict[str, str] = {
    "sbi": "serious bacterial infection",
    "lbw": "low birth weight",
    "kmc": "kangaroo mother care",
    "nbu": "newborn unit",
    "bp": "blood pressure",
    "hr": "heart rate",
    "rr": "respiratory rate",
    "spo2": "oxygen saturation",
    "iv": "intravenous",
    "im": "intramuscular",
}
_ABBREVIATION_PATTERNS = {
    abbr: re.compile(rf"\b{re.escape(abbr)}\b", re.IGNORECASE) for abbr in _ABBREVIATIONS
}


def _expand_abbreviations(query: str) -> str:
    """Append `(expansion)` after a recognised abbreviation; the original
    term is preserved (exact-term/BM25 matches still work) and the expansion
    adds vocabulary useful to the dense retriever. Never invents content
    beyond the curated map."""
    expanded = query
    for abbr, full in _ABBREVIATIONS.items():
        pattern = _ABBREVIATION_PATTERNS[abbr]

        def _add_expansion(m: re.Match[str], full: str = full) -> str:
            return f"{m.group(0)} ({full})"

        if pattern.search(expanded):
            expanded = pattern.sub(_add_expansion, expanded)
    return expanded


def _default_vectorstore() -> VectorStore:
    s = get_settings()
    return QdrantVectorStore(
        url=s.qdrant_url, api_key=s.qdrant_api_key, collection=s.qdrant_guideline_collection
    )


def _to_retrieval_item(candidate: dict, score: float) -> RetrievalItem:
    return RetrievalItem(
        chunk_id=candidate["chunk_id"],
        score=score,
        section_path=candidate.get("section_path"),
        section_number=candidate.get("section_number"),
        page_start=candidate["page_start"],
        page_end=candidate["page_end"],
        char_start=candidate["char_start"],
        char_end=candidate["char_end"],
        document_id=candidate["document_id"],
        document_title=candidate["document_title"],
        document_version_id=candidate["document_version_id"],
        version_label=candidate["version_label"],
        effective_date=candidate.get("effective_date"),
        version_status=candidate.get("status", "active"),
        chunk_type=candidate.get("chunk_type", "prose"),
        text=candidate["text"],
        heading=candidate.get("heading"),
        meta=candidate.get("meta") or {},
    )


def retrieve(
    query: str,
    *,
    access_filter: dict | None = None,
    vectorstore: VectorStore | None = None,
    session: Session | None = None,
    conversation_id: uuid.UUID | None = None,
    patient_id: uuid.UUID | None = None,
    actor_id: uuid.UUID | None = None,
    actor_role: str | None = None,
    purpose: str | None = None,
) -> tuple[list[RetrievalItem], dict]:
    """Run the full hybrid retrieval pipeline for one query.

    `vectorstore` defaults to a real `QdrantVectorStore` built from config;
    pass one explicitly for tests (e.g. `QdrantVectorStore(url=":memory:", ...)`
    ) or to reuse a warm client. `session` is optional — when given, one
    `retrieval` audit event is written (DEVIATIONS.md #50); offline unit
    tests pass none.
    """
    settings = get_settings()
    store = vectorstore or _default_vectorstore()

    expanded_query = _expand_abbreviations(query)
    dense = embed_texts([expanded_query], is_query=True)[0]
    sparse = query_sparse_vector(expanded_query)

    flt = dict(access_filter or {})
    flt.setdefault("status", "active")
    flt.setdefault("exclude_review_status", list(NOT_RETRIEVABLE))

    candidates = store.hybrid_search(
        dense=dense,
        sparse=sparse,
        prefetch_limit=settings.candidate_k,
        limit=settings.fused_k,
        flt=flt,
    )

    passages = [c["text"] for c in candidates]
    rerank_scores = rerank(expanded_query, passages)
    ranked = sorted(zip(candidates, rerank_scores, strict=True), key=lambda cs: cs[1], reverse=True)
    top = ranked[: settings.top_k]

    items = [_to_retrieval_item(c, s) for c, s in top]
    confidence = assess([s for _, s in top])
    conflicts = detect_conflicts(items)

    snapshot = {
        "query": query,
        "expanded_query": expanded_query,
        "items": [dict(item) for item in items],
        "confidence": {
            "top_score": confidence.top_score,
            "supporting_count": confidence.supporting_count,
            "low_confidence": confidence.low_confidence,
            "essentially_empty": confidence.essentially_empty,
        },
        "conflicts": conflicts,
    }

    if session is not None:
        write_event(
            session,
            action="retrieval",
            actor_id=actor_id,
            actor_role=actor_role,
            purpose=purpose,
            conversation_id=conversation_id,
            patient_id=patient_id,
            query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            retrieved=[
                {
                    "chunk_id": item["chunk_id"],
                    "score": item["score"],
                    "fusion": "rrf",
                    "rerank": settings.reranker_backend,
                }
                for item in items
            ],
        )

    return items, snapshot
