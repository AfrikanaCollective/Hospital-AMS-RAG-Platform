"""Celery ingestion tasks (ARCH §5; PRD-105).

`process_document` is implemented (Phase 2): parse -> chunk -> embed ->
persist chunk rows -> upsert Qdrant -> update `document_version.page_count`/
`parse_quality` (ARCH §5.1 steps 2-4). The `document`/`document_version` rows
already exist by the time this runs — created synchronously at submission
(ARCH §5.1 step 1) — and `document.source_uri` names the stored file.
`topic_tags` (the manifest's document-level list, ARCH-038) is passed as a
task argument rather than a DB column: it is only ever needed here, once, to
derive each chunk's own tags (`app.ingestion.topics.assign_topic_tags`), so
there is nothing else that would read it back from a stored column
(DEVIATIONS.md #58).

**Partial-extract page provenance (DEVIATIONS.md #166, ARCH-038 extension):**
if the manifest declares `source_pages` for this file (an operator attesting
"this file is really pages X-Y of a larger publication"), `parse_document`'s
own page numbers are remapped to the true source pages before chunking, so
`Chunk.page_start`/`page_end` — and therefore every citation — point at the
real document's pagination, not the extract's own 1..N. See
`app.ingestion.page_provenance`.

**Parser selection (ARCH-044, PRD-113):** `INGEST_PARSER=layout` uses the
layout-aware parser (`app.ingestion.layout.pipeline.parse_with_layout`:
Docling + pdfplumber + local OCR, with the manifest's `text_corrections`,
`boilerplate_patterns` and `flowchart_attestations`); the default `pypdf`
keeps the original text-only path. Either way, a document whose
`parse_quality` is below `INGEST_MIN_PARSE_QUALITY` has every chunk held for
admin review (`app.ingestion.review`), which ARCH §5.1 always specified but
nothing enforced before (DEVIATIONS.md #215).

`process_record_batch` (async ingestion of a large record batch) and
`reembed_corpus` (full corpus re-embed on an `EMBEDDING_MODEL_ID` change)
remain unimplemented — DEVIATIONS.md #60: both need infrastructure ARCH
doesn't yet specify (a batch-reference mechanism for the former; a plan for
running two Qdrant collections + a cutover step for the latter) and
implementing either now would mean guessing that design rather than building
to a spec.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from app.config import get_settings
from app.db.models.corpus import Chunk, Document, DocumentVersion
from app.db.session import session_scope
from app.ingestion.chunk_persistence import persist_chunks
from app.ingestion.chunking import chunk_document
from app.ingestion.embed import embed_texts
from app.ingestion.layout.pipeline import parse_with_layout
from app.ingestion.page_provenance import (
    apply_source_pages,
    load_manifest_entry,
    load_source_pages,
)
from app.ingestion.pdf_parse import parse_document
from app.ingestion.review import exclude_sections, hold_low_quality_chunks, load_exclude_sections
from app.retrieval.vectorstore import QdrantVectorStore, VectorStore
from app.worker import celery_app

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


def _default_vectorstore() -> VectorStore:
    s = get_settings()
    return QdrantVectorStore(
        url=s.qdrant_url, api_key=s.qdrant_api_key, collection=s.qdrant_guideline_collection
    )


def _run_process_document(
    session: Session,
    document_version_id: str,
    *,
    topic_tags: list[str] | None = None,
    vectorstore: VectorStore | None = None,
) -> list[Chunk]:
    version = session.get(DocumentVersion, uuid.UUID(document_version_id))
    if version is None:
        raise ValueError(f"document_version {document_version_id!r} not found")
    document = session.get(Document, version.document_id)
    if document is None or not document.source_uri:
        raise ValueError(f"document {version.document_id} has no source_uri to parse")

    settings = get_settings()
    filename = Path(document.source_uri).name
    manifest_entry = load_manifest_entry(settings.sample_guidelines_dir, filename)
    if settings.ingest_parser == "layout":
        parsed = parse_with_layout(document.source_uri, manifest_entry)
    else:
        parsed = parse_document(document.source_uri)
    source_pages = load_source_pages(settings.sample_guidelines_dir, filename)
    if source_pages is not None:
        apply_source_pages(parsed, source_pages)
    version.page_count = parsed.page_count
    version.parse_quality = parsed.parse_quality
    version.parser_version = parsed.parser_version or "pypdf"
    version.parse_report = parsed.parse_report or None

    format_profile = version.format_profile or "narrative"
    chunk_dicts = chunk_document(parsed, format_profile=format_profile)
    hold_low_quality_chunks(chunk_dicts, parsed.parse_quality, settings.ingest_min_parse_quality)
    # Non-recommendation sections never reach retrieval (DEVIATIONS.md #251).
    exclude_sections(chunk_dicts, load_exclude_sections(manifest_entry))

    store = vectorstore or _default_vectorstore()
    dense_dim = len(embed_texts(["dimension probe"])[0])
    store.ensure_collection(dense_dim=dense_dim)

    rows = persist_chunks(
        session,
        store,
        document_version_id=version.id,
        document_id=document.id,
        document_title=document.title,
        version_label=version.version_label,
        effective_date=version.effective_date,
        chunk_dicts=chunk_dicts,
        document_topic_tags=topic_tags or [],
    )
    # A new version may have superseded an older one (ARCH §5.1 step 6).
    # Retrieval filters on the Qdrant payload, which supersession never
    # updated before (DEVIATIONS.md #215), so push the status there too.
    superseded_id = getattr(version, "supersedes_id", None)
    if superseded_id is not None:
        store.set_payload_by_version(str(superseded_id), {"status": "superseded"})
    return rows


@celery_app.task(name="ingestion.process_document")
def process_document(document_version_id: str, topic_tags: list[str] | None = None) -> None:
    """parse -> chunk -> embed -> upsert Qdrant -> persist chunk rows (ARCH §5.1)."""
    with session_scope() as session:
        _run_process_document(session, document_version_id, topic_tags=topic_tags)


@celery_app.task(name="ingestion.process_record_batch")
def process_record_batch(batch_id: str) -> None:
    raise NotImplementedError(
        "Phase 2+ (DEVIATIONS #60): needs a batch-reference mechanism not yet designed"
    )


@celery_app.task(name="ingestion.reembed_corpus")
def reembed_corpus(new_collection: str) -> None:
    """Full re-embed into a new Qdrant collection on EMBEDDING_MODEL_ID change (ARCH §6)."""
    raise NotImplementedError("Phase 2+ (DEVIATIONS #60): full corpus re-embed, deferred")
