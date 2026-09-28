"""Corpus read access + version withdrawal (ARCH §5.1; PRD-004, PRD-005).

Backs `GET /corpus/documents`, `GET /corpus/documents/{id}/versions`,
`GET /corpus/chunks/{chunk_id}` (citation re-verification, PRD-016), and
`POST /corpus/versions/{id}/withdraw`. Sibling to `app.ingestion.documents`
(the write-side document/version persistence) — corpus rows are non-PHI, not
RLS-protected, so no `patient_scope` concern here.

`get_chunk` resolves by id regardless of the owning version's `status` —
"withdrawn guideline after citations were issued": citations still resolve
with a withdrawn badge; only active *retrieval* excludes withdrawn versions
(ARCH §5.1, §21c "Withdrawn guideline..." edge case). Callers wanting the
badge use the returned chunk's `document_version_id` against
`list_document_versions`/a direct lookup.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import select

from app.audit.log import write_event
from app.db.models.corpus import Chunk, Document, DocumentVersion

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.retrieval.vectorstore import VectorStore


class DocumentNotFoundError(LookupError):
    pass


class DocumentVersionNotFoundError(LookupError):
    pass


class ChunkNotFoundError(LookupError):
    pass


def list_documents(session: Session) -> list[Document]:
    return list(session.execute(select(Document).order_by(Document.title)).scalars().all())


def list_document_versions(session: Session, document_id: uuid.UUID) -> list[DocumentVersion]:
    if session.get(Document, document_id) is None:
        raise DocumentNotFoundError(f"no document {document_id}")
    return list(
        session.execute(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.ingested_at.desc())
        )
        .scalars()
        .all()
    )


def get_chunk(session: Session, chunk_id: uuid.UUID) -> Chunk:
    chunk = session.get(Chunk, chunk_id)
    if chunk is None:
        raise ChunkNotFoundError(f"no chunk {chunk_id}")
    return chunk


def withdraw_version(
    session: Session,
    version_id: uuid.UUID,
    *,
    actor_id: uuid.UUID | None = None,
    actor_role: str | None = None,
    store: VectorStore | None = None,
) -> DocumentVersion:
    """Mark a version withdrawn (idempotent). Never deletes anything — chunks/
    vectors/citations for a withdrawn version remain resolvable (PRD-005's
    append-only supersession model); only the `status` badge changes."""
    version = session.get(DocumentVersion, version_id)
    if version is None:
        raise DocumentVersionNotFoundError(f"no document_version {version_id}")
    version.status = "withdrawn"
    session.flush()
    if store is not None:
        # Retrieval filters on the Qdrant payload; without this a withdrawn
        # version stayed retrievable (ARCH-044 review, DEVIATIONS.md #215).
        store.set_payload_by_version(str(version_id), {"status": "withdrawn"})
    write_event(
        session,
        action="config_change",
        actor_id=actor_id,
        actor_role=actor_role,
        outcome="withdrawn",
        detail={
            "document_version_id": str(version_id),
            "document_id": str(version.document_id),
        },
    )
    return version
