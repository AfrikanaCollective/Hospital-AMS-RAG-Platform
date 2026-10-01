"""Guideline document + document_version persistence (ARCH §5.1 steps 1, 6, 7;
ARCH-038).

Metadata (title, publisher, external_ref, licence, version_label,
effective_date, topic_tags, format_profile) is operator-supplied via the
ingest manifest — never inferred from the PDF (ARCH-038). This module only
creates/supersedes `document`/`document_version` rows; parsing, chunking, and
embedding happen in `app.ingestion.tasks.process_document` once a version row
exists (ARCH §5.1 step 1: rows are created synchronously at submission,
before the async parse/chunk/embed task runs).

Ingestion is idempotent on `content_sha256` (ARCH §5.1): re-submitting an
identical file is a no-op.

Supersession: a newer `version_label`/`effective_date` for the same
`external_ref` marks the prior `active` version `superseded`, keeping its
chunks/vectors retrievable so existing citations still resolve (ARCH §5.1
step 6). See DEVIATIONS.md #59 for how "newer" is decided here — PRD-Q1
(open question) covers the harder / undated cases this simple rule doesn't
resolve.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import select

from app.db.models.corpus import Document, DocumentVersion

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


@dataclass(frozen=True)
class DocumentMetadata:
    title: str
    publisher: str | None
    external_ref: str | None
    source_uri: str | None
    licence: str | None
    version_label: str
    effective_date: date | None
    topic_tags: list[str]
    format_profile: str | None


def _find_version_by_sha256(session: Session, content_sha256: str) -> DocumentVersion | None:
    """A version of this exact file, preferring the active one, else the most
    recent. Several versions can share a hash: `create_reparse_version`
    re-parses the same file as a new version (ARCH-044), which made the old
    `scalar_one_or_none()` raise instead of answering "already ingested"
    (DEVIATIONS.md #240)."""
    versions = list(
        session.execute(
            select(DocumentVersion).where(DocumentVersion.content_sha256 == content_sha256)
        )
        .scalars()
        .all()
    )
    if not versions:
        return None
    active = [v for v in versions if v.status == "active"]
    pool = active or versions
    return max(pool, key=lambda v: v.ingested_at or datetime.min.replace(tzinfo=UTC))


def _find_document_by_external_ref(session: Session, external_ref: str) -> Document | None:
    return session.execute(
        select(Document).where(Document.external_ref == external_ref)
    ).scalar_one_or_none()


def _find_active_versions(session: Session, document_id: uuid.UUID) -> list[DocumentVersion]:
    return list(
        session.execute(
            select(DocumentVersion).where(
                DocumentVersion.document_id == document_id, DocumentVersion.status == "active"
            )
        )
        .scalars()
        .all()
    )


def _is_newer(
    new_date: date | None, new_label: str, prior_date: date | None, prior_label: str
) -> bool:
    """ARCH §5.1 step 6's "newer" check (DEVIATIONS.md #59). `effective_date`
    is authoritative when both sides have one; a side with a date beats a
    side without one; with neither dated, falls back to a string comparison
    of the version label — a deliberately simple tie-break, not a full
    version-authority policy (PRD-Q1 remains open for the harder cases)."""
    if new_date is not None and prior_date is not None:
        return new_date > prior_date
    if new_date is not None:
        return True
    if prior_date is not None:
        return False
    return new_label > prior_label


def create_or_supersede_document_version(
    session: Session,
    meta: DocumentMetadata,
    *,
    content_sha256: str,
    page_count: int | None = None,
) -> tuple[DocumentVersion, bool]:
    """Returns `(version, created)`. `created=False` means this call was an
    idempotent no-op — a file with this exact `content_sha256` was already
    ingested (ARCH §5.1)."""
    existing = _find_version_by_sha256(session, content_sha256)
    if existing is not None:
        return existing, False

    document = (
        _find_document_by_external_ref(session, meta.external_ref) if meta.external_ref else None
    )
    if document is None:
        document = Document(
            external_ref=meta.external_ref,
            title=meta.title,
            publisher=meta.publisher,
            source_uri=meta.source_uri,
            licence=meta.licence,
        )
        session.add(document)
        session.flush()

    supersedes_id = None
    for prior in _find_active_versions(session, document.id):
        if _is_newer(
            meta.effective_date, meta.version_label, prior.effective_date, prior.version_label
        ):
            prior.status = "superseded"
            supersedes_id = prior.id

    version = DocumentVersion(
        document_id=document.id,
        version_label=meta.version_label,
        effective_date=meta.effective_date,
        ingested_at=datetime.now(UTC),
        supersedes_id=supersedes_id,
        status="active",
        content_sha256=content_sha256,
        page_count=page_count,
        format_profile=meta.format_profile,
    )
    session.add(version)
    session.flush()
    return version, True


def create_reparse_version(
    session: Session, prior: DocumentVersion, *, parser_version: str
) -> DocumentVersion:
    """A new version of the **same file** built with a different parser
    (ARCH-044 re-ingestion, LAYOUT-INGESTION-PROPOSAL.md §10). Same
    `content_sha256`, `version_label`, `effective_date` and `format_profile`;
    the prior version becomes `superseded` (never deleted, so every citation
    already issued against it still resolves — ARCH §5.1 step 6). This is the
    one path that deliberately bypasses `content_sha256` idempotency: the
    bytes are unchanged, the parse is not."""
    prior.status = "superseded"
    version = DocumentVersion(
        document_id=prior.document_id,
        version_label=prior.version_label,
        effective_date=prior.effective_date,
        ingested_at=datetime.now(UTC),
        supersedes_id=prior.id,
        status="active",
        content_sha256=prior.content_sha256,
        page_count=prior.page_count,
        format_profile=prior.format_profile,
        parser_version=parser_version,
    )
    session.add(version)
    session.flush()
    return version
