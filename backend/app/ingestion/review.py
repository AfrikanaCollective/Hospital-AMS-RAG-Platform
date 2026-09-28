"""Chunk review gate (ARCH-044, ARCH §5.1 step 2; LAYOUT-INGESTION-PROPOSAL.md
§8).

A chunk is **held** (`meta.review_status = "pending"`) and excluded from
retrieval when:

- it contains OCR text with digits (set at chunking time — OCR'd doses are
  the highest-risk extraction), or
- it contains a vision-LLM table transcription (D12; always, even when it
  agreed with OCR), or
- its document's `parse_quality` is below `INGEST_MIN_PARSE_QUALITY`
  (ARCH §5.1 always specified this hold; nothing enforced it before —
  DEVIATIONS.md #215).

An admin reviews the held chunk against its page crop and **confirms** it
(it becomes retrievable) or **rejects** it (it stays out). A reviewer can't
edit source text: a wrong OCR reading is fixed with a manifest
`text_corrections` entry (`kind = ocr_override`) and re-ingestion. Every
decision is an append-only audit event.

Qdrant is the retrieval filter's source of truth (`review_status` payload),
so every status change is written to both Postgres and Qdrant — the same
applies to document-version status (`sync_version_status`), which
supersession and withdrawal previously changed in Postgres only.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import select

from app.audit.log import write_event
from app.db.models.corpus import Chunk, DocumentVersion

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.retrieval.vectorstore import VectorStore

PENDING = "pending"
CONFIRMED = "confirmed"
REJECTED = "rejected"
# Excluded from retrieval (hybrid search and the offline ablation corpus).
NOT_RETRIEVABLE = (PENDING, REJECTED)


class ChunkReviewError(ValueError):
    pass


def hold_low_quality_chunks(chunk_dicts: list[dict], parse_quality: float, threshold: float) -> int:
    """Hold every chunk of a document parsed below `threshold`. Returns how
    many were newly held."""
    if parse_quality >= threshold:
        return 0
    held = 0
    for c in chunk_dicts:
        meta = c.setdefault("meta", {})
        if meta.get("review_status") != PENDING:
            held += 1
        meta["review_status"] = PENDING
        reasons = meta.setdefault("review_reasons", [])
        if "low_parse_quality" not in reasons:
            reasons.append("low_parse_quality")
    return held


def review_reasons(meta: dict) -> list[str]:
    reasons = list(dict.fromkeys(meta.get("review_reasons", [])))
    if meta.get("ocr", {}).get("has_digits") and "ocr_numeric" not in reasons:
        reasons.append("ocr_numeric")
    return reasons


def list_pending(session: Session, *, limit: int = 200) -> list[Chunk]:
    return list(
        session.execute(
            select(Chunk)
            .where(Chunk.meta["review_status"].astext == PENDING)
            .order_by(Chunk.document_version_id, Chunk.ordinal)
            .limit(limit)
        )
        .scalars()
        .all()
    )


def review_chunk(
    session: Session,
    store: VectorStore,
    chunk_id: uuid.UUID,
    *,
    decision: str,
    note: str | None,
    actor_id: uuid.UUID | None,
    actor_role: str | None,
) -> Chunk:
    if decision not in (CONFIRMED, REJECTED):
        raise ChunkReviewError(f"decision must be {CONFIRMED!r} or {REJECTED!r}")
    chunk = session.get(Chunk, chunk_id)
    if chunk is None:
        raise ChunkReviewError(f"no chunk {chunk_id}")
    previous = (chunk.meta or {}).get("review_status")
    if previous not in (PENDING, CONFIRMED, REJECTED):
        raise ChunkReviewError(f"chunk {chunk_id} is not under review")
    meta = dict(chunk.meta or {})
    meta["review_status"] = decision
    meta["review"] = {
        "decision": decision,
        "note": note,
        "actor_id": str(actor_id) if actor_id else None,
        "at": datetime.now(UTC).isoformat(),
        "reasons": review_reasons(meta),
    }
    chunk.meta = meta
    session.flush()
    store.set_payload([str(chunk.id)], {"review_status": decision})
    write_event(
        session,
        action="config_change",
        actor_id=actor_id,
        actor_role=actor_role,
        outcome=f"chunk_review_{decision}",
        detail={
            "chunk_id": str(chunk.id),
            "document_version_id": str(chunk.document_version_id),
            "previous_status": previous,
            "reasons": meta["review"]["reasons"],
            "note": note,
        },
    )
    return chunk


def sync_version_status(session: Session, store: VectorStore, document_id: uuid.UUID) -> int:
    """Push every version's Postgres `status` (active | superseded | withdrawn)
    into its Qdrant points' `status` payload, which retrieval filters on.
    Returns the number of versions synced."""
    versions = (
        session.execute(select(DocumentVersion).where(DocumentVersion.document_id == document_id))
        .scalars()
        .all()
    )
    for v in versions:
        store.set_payload_by_version(str(v.id), {"status": v.status})
    return len(versions)
