"""reviewer registration rejection reason

Revision ID: f1a9c3d75b20
Revises: e3f5b8a2c614
Create Date: 2026-10-07 16:00:00.000000

DEVIATIONS.md #285: `iam.reviewer_profile.rejection_reason`, the admin's
reason for the most recent rejection, shown to the reviewer.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "f1a9c3d75b20"
down_revision = "e3f5b8a2c614"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "reviewer_profile", sa.Column("rejection_reason", sa.Text(), nullable=True), schema="iam"
    )


def downgrade() -> None:
    op.drop_column("reviewer_profile", "rejection_reason", schema="iam")
