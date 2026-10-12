"""reviewer registration: cadre "Other" specified

Revision ID: a4c8e1f27d93
Revises: f1a9c3d75b20
Create Date: 2026-10-12 12:00:00.000000

DEVIATIONS.md #289: `iam.reviewer_profile.cadre_other`, the reviewer's own
cadre when they choose "Other"; NULL for every other cadre.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "a4c8e1f27d93"
down_revision = "f1a9c3d75b20"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "reviewer_profile", sa.Column("cadre_other", sa.String(30), nullable=True), schema="iam"
    )


def downgrade() -> None:
    op.drop_column("reviewer_profile", "cadre_other", schema="iam")
