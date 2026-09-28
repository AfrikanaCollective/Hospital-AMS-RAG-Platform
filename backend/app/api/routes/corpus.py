"""Corpus / guideline version management (PRD-004, PRD-005, PRD-016; ARCH §5.1).

GET  /corpus/documents                         -> every document + its versions
GET  /corpus/documents/{id}/versions           -> one document's versions
GET  /corpus/chunks/{chunk_id}                 -> chunk text + offsets (citation
                                                   re-verification, PRD-016)
POST /corpus/versions/{id}/withdraw   (admin)  -> status=withdrawn; old citations
                                                   still resolve (ARCH §5.1)
GET  /corpus/review-queue             (admin)  -> chunks held for review (OCR'd
                                                   numbers, low parse quality; ARCH-044)
POST /corpus/chunks/{id}/review       (admin)  -> confirm | reject a held chunk
GET  /corpus/crops/{sha256}.png                -> figure/table crop for the
                                                   reviewer and citation views

Reads (`corpus:read`, `app.auth.rbac.ROUTE_PERMISSIONS`) are open to any
authenticated role that can reach a citation or corpus listing (clinician,
reviewer, admin) — corpus content is non-PHI reference material, not gated
per-patient. Withdrawal (`corpus:manage`) is admin-only, per ARCH §17.3
("Admins manage corpus").
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import Principal, get_db, principal_uuid, require_role
from app.config import get_settings
from app.db.models.corpus import Chunk, Document, DocumentVersion
from app.ingestion.corpus_access import (
    ChunkNotFoundError,
    DocumentNotFoundError,
    DocumentVersionNotFoundError,
    list_document_versions,
    list_documents,
    withdraw_version,
)
from app.ingestion.corpus_access import (
    get_chunk as get_chunk_row,
)
from app.ingestion.review import ChunkReviewError, list_pending, review_chunk, review_reasons
from app.retrieval.vectorstore import QdrantVectorStore, VectorStore

router = APIRouter()


def _serialize_version(v: DocumentVersion) -> dict:
    return {
        "id": str(v.id),
        "document_id": str(v.document_id),
        "version_label": v.version_label,
        "effective_date": v.effective_date.isoformat() if v.effective_date else None,
        "ingested_at": v.ingested_at.isoformat(),
        "supersedes_id": str(v.supersedes_id) if v.supersedes_id else None,
        "status": v.status,
        "content_sha256": v.content_sha256,
        "page_count": v.page_count,
        "format_profile": v.format_profile,
        "parse_quality": v.parse_quality,
        "parser_version": getattr(v, "parser_version", None),
        "parse_report": getattr(v, "parse_report", None),
    }


def _serialize_document(d: Document, versions: list[DocumentVersion]) -> dict:
    return {
        "id": str(d.id),
        "external_ref": d.external_ref,
        "title": d.title,
        "publisher": d.publisher,
        "source_uri": d.source_uri,
        "classification": d.classification,
        "licence": d.licence,
        "versions": [_serialize_version(v) for v in versions],
    }


def _serialize_chunk(c: Chunk) -> dict:
    return {
        "id": str(c.id),
        "document_version_id": str(c.document_version_id),
        "section_path": c.section_path,
        "section_number": c.section_number,
        "heading": c.heading,
        "page_start": c.page_start,
        "page_end": c.page_end,
        "char_start": c.char_start,
        "char_end": c.char_end,
        "ordinal": c.ordinal,
        "chunk_type": c.chunk_type,
        "text": c.text,
        "figure_ref": getattr(c, "figure_ref", None),
        "meta": {k: v for k, v in (c.meta or {}).items() if k != "embedding_text"},
    }


@router.get("/documents")
async def list_documents_route(
    _principal: Principal = Depends(require_role("clinician", "reviewer", "admin")),
    session: Session = Depends(get_db),
) -> list[dict]:
    documents = list_documents(session)
    return [_serialize_document(d, list_document_versions(session, d.id)) for d in documents]


@router.get("/documents/{document_id}/versions")
async def list_versions_route(
    document_id: str,
    _principal: Principal = Depends(require_role("clinician", "reviewer", "admin")),
    session: Session = Depends(get_db),
) -> list[dict]:
    try:
        versions = list_document_versions(session, uuid.UUID(document_id))
    except DocumentNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return [_serialize_version(v) for v in versions]


@router.get("/chunks/{chunk_id}")
async def get_chunk_route(
    chunk_id: str,
    _principal: Principal = Depends(require_role("clinician", "reviewer", "admin")),
    session: Session = Depends(get_db),
) -> dict:
    try:
        chunk = get_chunk_row(session, uuid.UUID(chunk_id))
    except ChunkNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return _serialize_chunk(chunk)


@router.post("/versions/{version_id}/withdraw")
async def withdraw_version_route(
    version_id: str,
    principal: Principal = Depends(require_role("admin")),
    session: Session = Depends(get_db),
) -> dict:
    try:
        version = withdraw_version(
            session,
            uuid.UUID(version_id),
            actor_id=principal_uuid(principal),
            actor_role="admin",
            store=get_vectorstore(),
        )
    except DocumentVersionNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return _serialize_version(version)


def get_vectorstore() -> VectorStore:
    s = get_settings()
    return QdrantVectorStore(
        url=s.qdrant_url, api_key=s.qdrant_api_key, collection=s.qdrant_guideline_collection
    )


class ChunkReviewRequest(BaseModel):
    decision: str  # confirmed | rejected
    note: str | None = None


@router.get("/review-queue")
async def review_queue_route(
    _principal: Principal = Depends(require_role("admin")),
    session: Session = Depends(get_db),
) -> list[dict]:
    """Chunks held for review (ARCH-044): OCR'd numeric content and chunks of
    low-parse-quality documents. Each item carries its crop reference so the
    reviewer can compare the extracted text with the page."""
    out = []
    for c in list_pending(session):
        version = session.get(DocumentVersion, c.document_version_id)
        document = session.get(Document, version.document_id) if version else None
        item = _serialize_chunk(c)
        item["document_title"] = document.title if document else None
        item["version_label"] = version.version_label if version else None
        item["review_reasons"] = review_reasons(c.meta or {})
        out.append(item)
    return out


@router.post("/chunks/{chunk_id}/review")
async def review_chunk_route(
    chunk_id: str,
    body: ChunkReviewRequest,
    principal: Principal = Depends(require_role("admin")),
    session: Session = Depends(get_db),
) -> dict:
    try:
        chunk = review_chunk(
            session,
            get_vectorstore(),
            uuid.UUID(chunk_id),
            decision=body.decision,
            note=body.note,
            actor_id=principal_uuid(principal),
            actor_role="admin",
        )
    except ChunkReviewError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return _serialize_chunk(chunk)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@router.get("/crops/{sha256}.png")
async def crop_route(
    sha256: str,
    _principal: Principal = Depends(require_role("clinician", "reviewer", "admin")),
) -> FileResponse:
    """A stored figure/table crop (content-addressed; non-PHI guideline
    content, same access as corpus reads)."""
    if not _SHA256_RE.match(sha256):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid crop id")
    path = Path(get_settings().ingest_crop_dir) / f"{sha256}.png"
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "crop not found")
    return FileResponse(path, media_type="image/png")
