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

A chunk is **excluded** (`meta.review_status = "excluded"`) when its section
is on its document's manifest `exclude_sections` list -- navigation, front
matter, committee rationale, research recommendations: text that is not a
recommendation and must never be cited as one (DEVIATIONS.md #251). It is
kept for provenance but never retrieved, and it is not a review decision:
it does not enter the review queue and a reviewer cannot confirm it.

Qdrant is the retrieval filter's source of truth (`review_status` payload),
so every status change is written to both Postgres and Qdrant — the same
applies to document-version status (`sync_version_status`), which
supersession and withdrawal previously changed in Postgres only.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Select, select

from app.audit.log import write_event
from app.db.models.corpus import Chunk, DocumentVersion

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.retrieval.vectorstore import VectorStore

PENDING = "pending"
CONFIRMED = "confirmed"
REJECTED = "rejected"
EXCLUDED = "excluded"
# Excluded from retrieval (hybrid search and the offline ablation corpus).
NOT_RETRIEVABLE = (PENDING, REJECTED, EXCLUDED)

# `exclude_unheaded` in a manifest's `exclude_sections` selects chunks with no
# heading path at all (title pages, logos).
UNHEADED = ""


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


def load_exclude_sections(entry: dict | None) -> list[str]:
    """A manifest entry's `exclude_sections`: heading-path prefixes whose
    chunks are excluded from retrieval. A prefix matches its own section and
    every subsection ("Rationale and impact" matches "Rationale and impact ›
    Lumbar puncture"), never a longer heading that merely starts with the
    same words. An empty string selects chunks with no heading path."""
    raw = (entry or {}).get("exclude_sections") or []
    if not isinstance(raw, list) or not all(isinstance(p, str) for p in raw):
        raise ValueError("exclude_sections must be a list of heading-path strings")
    return [p.strip() for p in raw]


def section_excluded(section_path: str | None, prefixes: list[str]) -> str | None:
    """The `exclude_sections` prefix that excludes `section_path`, or None."""
    path = (section_path or "").strip()
    for prefix in prefixes:
        if prefix == UNHEADED:
            if not path:
                return prefix
        elif path == prefix or path.startswith(f"{prefix} ›"):
            return prefix
    return None


def exclude_sections(chunk_dicts: list[dict], prefixes: list[str]) -> int:
    """Mark every chunk in an excluded section (at ingest, before persisting).
    Overrides a pending hold: an excluded chunk is never reviewed. Returns how
    many were excluded."""
    excluded = 0
    for c in chunk_dicts:
        rule = section_excluded(c.get("section_path"), prefixes)
        if rule is None:
            continue
        meta = c.setdefault("meta", {})
        meta["exclusion"] = {"rule": rule, "previous_status": meta.get("review_status")}
        meta["review_status"] = EXCLUDED
        excluded += 1
    return excluded


def identical_confirmed_source(
    text: str, table_source: str | None, confirmed: list[tuple[str, str, str | None]]
) -> str | None:
    """Id of a previously confirmed chunk that already contains every block of
    `text` verbatim (blocks split on blank lines -- e.g. a table part's
    repeated title and its row), with the same table source pin, or None.
    `confirmed` is `[(chunk_id, text, table_source)]`. Used only to carry a
    reviewer's confirmation across a re-chunking of the same, unchanged text
    (DEVIATIONS.md #255); any difference at all leaves the chunk held."""
    blocks = _blocks(text)
    if not blocks:
        return None
    for chunk_id, old_text, old_source in confirmed:
        # whole blocks only: a block cut short ("Agent: 1" of "Agent: 12")
        # is a substring of the confirmed text but not one of its blocks
        if old_source == table_source and set(blocks) <= set(_blocks(old_text)):
            return chunk_id
    return None


def _blocks(text: str) -> list[str]:
    return [b.strip() for b in text.split("\n\n") if b.strip()]


def review_reasons(meta: dict) -> list[str]:
    reasons = list(dict.fromkeys(meta.get("review_reasons", [])))
    if meta.get("ocr", {}).get("has_digits") and "ocr_numeric" not in reasons:
        reasons.append("ocr_numeric")
    return reasons


def pending_query(*, limit: int = 200) -> Select[tuple[Chunk]]:
    """Held chunks of **active** document versions only. A re-ingest
    supersedes the prior version, whose chunks leave retrieval but keep their
    `pending` status, so without the version filter every re-ingest left a
    duplicate set of held chunks in the queue, with nothing on the page to
    tell them apart (DEVIATIONS.md #225)."""
    return (
        select(Chunk)
        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
        .where(Chunk.meta["review_status"].astext == PENDING)
        .where(DocumentVersion.status == "active")
        .order_by(Chunk.document_version_id, Chunk.ordinal)
        .limit(limit)
    )


def list_pending(session: Session, *, limit: int = 200) -> list[Chunk]:
    return list(session.execute(pending_query(limit=limit)).scalars().all())


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
