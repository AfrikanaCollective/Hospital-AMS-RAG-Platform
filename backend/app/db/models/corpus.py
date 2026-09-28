"""`corpus` schema — guideline documents, versions, chunks (ARCH §4.1).

PRD-004: each chunk retains document id + version + section path + page + char
offset span. PRD-005: new versions supersede without deleting.
ARCH-038: `document.licence` + `document_version.format_profile` come from the
operator-supplied ingest manifest, never from PDF metadata.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPk

SCHEMA = "corpus"


class Document(UUIDPk, TimestampMixin, Base):
    __tablename__ = "document"
    __table_args__ = {"schema": SCHEMA}

    external_ref: Mapped[str | None] = mapped_column(String(128))
    title: Mapped[str] = mapped_column(Text)
    publisher: Mapped[str | None] = mapped_column(String(256))
    source_uri: Mapped[str | None] = mapped_column(Text)
    classification: Mapped[str] = mapped_column(String(16), default="public")  # public | internal
    licence: Mapped[str | None] = mapped_column(
        Text
    )  # usage terms, from the ingest manifest (ARCH-038)


class DocumentVersion(UUIDPk, Base):
    __tablename__ = "document_version"
    __table_args__ = {"schema": SCHEMA}

    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(f"{SCHEMA}.document.id"))
    version_label: Mapped[str] = mapped_column(String(256))
    effective_date: Mapped[date | None] = mapped_column()
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    supersedes_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(f"{SCHEMA}.document_version.id")
    )
    status: Mapped[str] = mapped_column(String(16), default="active")  # active|superseded|withdrawn
    content_sha256: Mapped[str] = mapped_column(String(64))
    page_count: Mapped[int | None] = mapped_column(Integer)
    # grade_recommendations | clinical_protocol | narrative (ARCH-038 / ARCH §6 rule 0)
    format_profile: Mapped[str | None] = mapped_column(String(24))
    parse_quality: Mapped[float | None] = mapped_column(
        Float
    )  # 0-1; below INGEST_MIN_PARSE_QUALITY -> admin hold
    # ARCH-044: parser stack that built this version's chunks, and its parse
    # audit (boilerplate dropped, OCR, tables, flowcharts, corrections).
    parser_version: Mapped[str | None] = mapped_column(Text)
    parse_report: Mapped[dict | None] = mapped_column(JSONB)


class Chunk(UUIDPk, Base):
    """id == chunk_id used in citations (ARCH-014)."""

    __tablename__ = "chunk"
    __table_args__ = {"schema": SCHEMA}

    document_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(f"{SCHEMA}.document_version.id")
    )
    section_path: Mapped[str | None] = mapped_column(Text)
    section_number: Mapped[str | None] = mapped_column(String(32))
    heading: Mapped[str | None] = mapped_column(Text)
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    ordinal: Mapped[int] = mapped_column(Integer)
    parent_chunk_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey(f"{SCHEMA}.chunk.id"))
    # prose|recommendation|protocol_step|table|figure|flowchart|list|criteria
    # (ARCH §6; `flowchart` ARCH-044)
    chunk_type: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text)
    figure_ref: Mapped[dict | None] = mapped_column(
        JSONB
    )  # chunk_type=figure: {page, bbox, image_sha256}
    token_count: Mapped[int | None] = mapped_column(Integer)
    vector_id: Mapped[str | None] = mapped_column(String(64))  # Qdrant point id
    # evidence grade, recommendation strength, criteria[], has_embedded_text
    # (figures), split_group_id, topic_tags; ARCH-044: text_origins, ocr,
    # review_status, corrections, flowchart, table_part
    meta: Mapped[dict] = mapped_column(JSONB, default=dict)


class CorpusSnapshot(UUIDPk, TimestampMixin, Base):
    """Pinned corpus state for reproducible eval runs (PRD-072)."""

    __tablename__ = "corpus_snapshot"
    __table_args__ = {"schema": SCHEMA}

    label: Mapped[str] = mapped_column(String(64))
    embedding_collection: Mapped[str] = mapped_column(String(128))
    document_version_ids: Mapped[list] = mapped_column(JSONB, default=list)
    notes: Mapped[str | None] = mapped_column(Text)


class ChunkLineage(UUIDPk, Base):
    """Old chunk -> new chunk mapping recorded on re-ingestion (ARCH-044,
    LAYOUT-INGESTION-PROPOSAL.md §10), used to remap eval gold sets."""

    __tablename__ = "chunk_lineage"
    __table_args__ = {"schema": SCHEMA}

    old_chunk_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(f"{SCHEMA}.chunk.id"))
    new_chunk_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(f"{SCHEMA}.chunk.id"))
    score: Mapped[float] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
