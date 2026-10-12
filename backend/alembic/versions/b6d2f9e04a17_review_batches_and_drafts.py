"""review batches and rating drafts

Revision ID: b6d2f9e04a17
Revises: a4c8e1f27d93
Create Date: 2026-10-12 15:00:00.000000

DEVIATIONS.md #294: `eval.review_batch_item` (each reviewer's batch of fresh
results) and `eval.rating_draft` (one half-finished rating per reviewer).
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "b6d2f9e04a17"
down_revision = "a4c8e1f27d93"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "review_batch_item",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("rater_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("batch_no", sa.Integer(), nullable=False),
        sa.Column(
            "result_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("eval.result.id"),
            nullable=False,
        ),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("dropped_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("rater_id", "result_id", name="review_batch_item_once_per_rater"),
        schema="eval",
    )
    op.create_index(
        "ix_review_batch_item_rater_id", "review_batch_item", ["rater_id"], schema="eval"
    )
    op.create_index(
        "ix_review_batch_item_result_id", "review_batch_item", ["result_id"], schema="eval"
    )
    op.create_table(
        "rating_draft",
        sa.Column("rater_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "result_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("eval.result.id"),
            nullable=False,
        ),
        sa.Column("payload_enc", sa.LargeBinary(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        schema="eval",
    )


def downgrade() -> None:
    op.drop_table("rating_draft", schema="eval")
    op.drop_index("ix_review_batch_item_result_id", "review_batch_item", schema="eval")
    op.drop_index("ix_review_batch_item_rater_id", "review_batch_item", schema="eval")
    op.drop_table("review_batch_item", schema="eval")
