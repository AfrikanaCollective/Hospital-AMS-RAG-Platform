"""reviewer registration resubmission: submission_count, last_rejected_at

Revision ID: e3f5b8a2c614
Revises: d7e2a4c91f30
Create Date: 2026-10-07 15:00:00.000000

DEVIATIONS.md #284: a rejected reviewer may resubmit their registration.
`submission_count` counts submissions (existing submitted rows start at 1);
`last_rejected_at` keeps the previous rejection time when the decision is
cleared for the new review.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "e3f5b8a2c614"
down_revision = "d7e2a4c91f30"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "reviewer_profile",
        sa.Column("submission_count", sa.SmallInteger(), nullable=False, server_default="0"),
        schema="iam",
    )
    op.add_column(
        "reviewer_profile",
        sa.Column("last_rejected_at", sa.DateTime(timezone=True), nullable=True),
        schema="iam",
    )
    op.execute(
        "UPDATE iam.reviewer_profile SET submission_count = 1 WHERE submitted_at IS NOT NULL"
    )


def downgrade() -> None:
    op.drop_column("reviewer_profile", "last_rejected_at", schema="iam")
    op.drop_column("reviewer_profile", "submission_count", schema="iam")
