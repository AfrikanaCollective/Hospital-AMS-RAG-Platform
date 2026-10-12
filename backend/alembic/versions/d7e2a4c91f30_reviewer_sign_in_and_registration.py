"""reviewer sign-in and registration: iam.user login columns, iam.reviewer_profile

Revision ID: d7e2a4c91f30
Revises: c3a1f9d2e7b4
Create Date: 2026-10-07 12:00:00.000000

DEVIATIONS.md #282:

- `iam.user.auth_method` / `google_sub` / `password_hash`: how a user signs
  in (Google, email + password, or the .env admin). Seeded users keep NULLs.
- `iam.reviewer_profile`: consent (version + sha256 of the text agreed to),
  the registration details, and the admin's decision. The Safaricom number
  is stored encrypted (`phone_enc`).

Grants come from the schema's default privileges
(deploy/postgres/init/01_schemas_roles.sql).
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "d7e2a4c91f30"
down_revision = "c3a1f9d2e7b4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("user", sa.Column("auth_method", sa.String(16), nullable=True), schema="iam")
    op.add_column("user", sa.Column("google_sub", sa.String(64), nullable=True), schema="iam")
    op.add_column("user", sa.Column("password_hash", sa.String(256), nullable=True), schema="iam")
    op.create_unique_constraint("uq_user_google_sub", "user", ["google_sub"], schema="iam")

    op.create_table(
        "reviewer_profile",
        sa.Column(
            "user_id", sa.Uuid(), sa.ForeignKey("iam.user.id"), primary_key=True, nullable=False
        ),
        sa.Column("status", sa.String(16), nullable=False, server_default="consented"),
        sa.Column("consent_version", sa.String(64), nullable=False),
        sa.Column("consent_sha256", sa.String(64), nullable=False),
        sa.Column("consented_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sex", sa.String(32), nullable=True),
        sa.Column("cadre", sa.String(32), nullable=True),
        sa.Column("expertise_level", sa.String(32), nullable=True),
        sa.Column("years_post_internship", sa.SmallInteger(), nullable=True),
        sa.Column("organisation", sa.String(30), nullable=True),
        sa.Column("role_title", sa.String(30), nullable=True),
        sa.Column("phone_enc", sa.LargeBinary(), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decided_by", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "status IN ('consented', 'pending', 'approved', 'rejected')",
            name="ck_reviewer_profile_status",
        ),
        schema="iam",
    )
    op.create_index("ix_reviewer_profile_status", "reviewer_profile", ["status"], schema="iam")


def downgrade() -> None:
    op.drop_index("ix_reviewer_profile_status", table_name="reviewer_profile", schema="iam")
    op.drop_table("reviewer_profile", schema="iam")
    op.drop_constraint("uq_user_google_sub", "user", schema="iam", type_="unique")
    op.drop_column("user", "password_hash", schema="iam")
    op.drop_column("user", "google_sub", schema="iam")
    op.drop_column("user", "auth_method", schema="iam")
