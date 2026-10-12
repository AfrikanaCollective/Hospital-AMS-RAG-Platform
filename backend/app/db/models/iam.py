"""`iam` schema — users, roles, sessions (ARCH §4.7, ARCH-011, ARCH-034).

Reviewers (rubric raters) must be clinicians: role set includes `reviewer` AND
`is_clinician` is true.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, LargeBinary, SmallInteger, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPk

SCHEMA = "iam"

ROLES = ("clinician", "reviewer", "admin", "service")


class User(UUIDPk, TimestampMixin, Base):
    __tablename__ = "user"
    __table_args__ = {"schema": SCHEMA}

    email: Mapped[str] = mapped_column(String(256), unique=True)
    display_name: Mapped[str] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(16), default="active")
    is_clinician: Mapped[bool] = mapped_column(Boolean, default=False)
    # duplicate-account detection support (PRD-043) — hashed identity attributes
    identity_fingerprint: Mapped[str | None] = mapped_column(String(64))
    # How this user signs in (DEVIATIONS.md #282): "google", "password",
    # "admin", or None for seeded users (dev-login only).
    auth_method: Mapped[str | None] = mapped_column(String(16))
    google_sub: Mapped[str | None] = mapped_column(String(64), unique=True)
    # scrypt hash (app.auth.passwords); None unless the user set a password.
    password_hash: Mapped[str | None] = mapped_column(String(256))


class Role(Base):
    __tablename__ = "role"
    __table_args__ = {"schema": SCHEMA}

    code: Mapped[str] = mapped_column(String(16), primary_key=True)  # one of ROLES


class UserRole(Base):
    __tablename__ = "user_role"
    __table_args__ = {"schema": SCHEMA}

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(f"{SCHEMA}.user.id"), primary_key=True)
    role_code: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.role.code"), primary_key=True)


class ServiceAccount(UUIDPk, TimestampMixin, Base):
    __tablename__ = "service_account"
    __table_args__ = {"schema": SCHEMA}

    name: Mapped[str] = mapped_column(String(128), unique=True)
    token_hash: Mapped[str] = mapped_column(String(64))


class Session(UUIDPk, TimestampMixin, Base):
    __tablename__ = "session"
    __table_args__ = {"schema": SCHEMA}

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(f"{SCHEMA}.user.id"))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


# Reviewer registration states (DEVIATIONS.md #282): consent recorded ->
# details submitted -> admin decision.
PROFILE_STATUSES = ("consented", "pending", "approved", "rejected")


class ReviewerProfile(TimestampMixin, Base):
    """A clinician reviewer's consent and registration details. Created when
    the user agrees to the consent text; a decline creates no row."""

    __tablename__ = "reviewer_profile"
    __table_args__ = {"schema": SCHEMA}

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(f"{SCHEMA}.user.id"), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default="consented")
    consent_version: Mapped[str] = mapped_column(String(64))
    consent_sha256: Mapped[str] = mapped_column(String(64))
    consented_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    sex: Mapped[str | None] = mapped_column(String(32))
    cadre: Mapped[str | None] = mapped_column(String(32))
    expertise_level: Mapped[str | None] = mapped_column(String(32))
    years_post_internship: Mapped[int | None] = mapped_column(SmallInteger)
    organisation: Mapped[str | None] = mapped_column(String(30))
    role_title: Mapped[str | None] = mapped_column(String(30))
    # Encrypted with reviewer_phone_aad(user_id); admin-only on read.
    phone_enc: Mapped[bytes | None] = mapped_column(LargeBinary)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decided_by: Mapped[uuid.UUID | None] = mapped_column()
    # Times the form has been submitted; above 1 after a rejected reviewer
    # resubmits (DEVIATIONS.md #284).
    submission_count: Mapped[int] = mapped_column(SmallInteger, default=0)
    last_rejected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # The admin's reason for the most recent rejection, shown to the reviewer
    # (DEVIATIONS.md #285). Kept after a resubmission so the admin sees it.
    rejection_reason: Mapped[str | None] = mapped_column(Text)


def reviewer_phone_aad(user_id: uuid.UUID) -> bytes:
    """AAD for `ReviewerProfile.phone_enc`, shared by writer and reader."""
    return f"iam.reviewer_profile.phone:{user_id}".encode()
