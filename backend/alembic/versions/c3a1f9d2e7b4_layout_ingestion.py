"""layout-aware ingestion: parser_version, parse_report, chunk_lineage

Revision ID: c3a1f9d2e7b4
Revises: 6f8b6444aa4a
Create Date: 2026-09-28 09:00:00.000000

ARCH-044 / PRD-113 (LAYOUT-INGESTION-PROPOSAL.md §9-§10):

- `corpus.document_version.parser_version`: which parser stack built this
  version's chunks (`pypdf`, `docling-…+pdfplumber-…`, `pypdf-fallback`).
  Re-ingesting the same file with a different parser creates a new version,
  so this is also what separates two versions sharing one `content_sha256`.
- `corpus.document_version.parse_report`: the per-document parse audit
  (boilerplate dropped, OCR pages/confidence, tables, flowcharts by
  verification state, corrections applied, fallback reason).
- `corpus.chunk_lineage`: old chunk -> new chunk mapping recorded when a
  document is re-ingested, with the overlap score, so eval gold sets can be
  remapped deterministically and the remap is auditable.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "c3a1f9d2e7b4"
down_revision = "6f8b6444aa4a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "document_version",
        sa.Column("parser_version", sa.Text(), nullable=True),
        schema="corpus",
    )
    op.add_column(
        "document_version",
        sa.Column("parse_report", postgresql.JSONB(), nullable=True),
        schema="corpus",
    )
    op.create_table(
        "chunk_lineage",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("old_chunk_id", sa.Uuid(), sa.ForeignKey("corpus.chunk.id"), nullable=False),
        sa.Column("new_chunk_id", sa.Uuid(), sa.ForeignKey("corpus.chunk.id"), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("method", sa.String(32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="corpus",
    )
    op.create_index(
        "ix_chunk_lineage_old", "chunk_lineage", ["old_chunk_id"], schema="corpus"
    )


def downgrade() -> None:
    op.drop_index("ix_chunk_lineage_old", table_name="chunk_lineage", schema="corpus")
    op.drop_table("chunk_lineage", schema="corpus")
    op.drop_column("document_version", "parse_report", schema="corpus")
    op.drop_column("document_version", "parser_version", schema="corpus")
