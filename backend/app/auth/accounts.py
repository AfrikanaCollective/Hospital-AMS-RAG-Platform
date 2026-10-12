"""Reviewer accounts: sign-in, consent, registration, admin approval and
progress (DEVIATIONS.md #282).

Three ways in, all ending in the same signed session token
(`app.auth.devjwt.mint_token`):

- Google: a verified Google ID token (`app.auth.google`);
- email + password, for reviewers without Google (scrypt hash);
- the single admin, whose username and password come from `.env`.

A new Google or password user has no roles. They must agree to the consent
text, then submit the registration form; an admin then approves (role
`reviewer`, `is_clinician`) or rejects them. Seeded users (dev-login) and the
admin skip registration.

All storage goes through `AccountStore`, so this module is plain logic that
tests drive with an in-memory store; `SqlAccountStore` is the real one.
"""

from __future__ import annotations

import hmac
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.audit.log import write_event
from app.auth.consent import ConsentText
from app.auth.google import GoogleIdentity
from app.auth.passwords import hash_password, password_problem, verify_password
from app.crypto.provider import get_crypto
from app.db.models.eval import RatingRound
from app.db.models.iam import ReviewerProfile, User, UserRole, reviewer_phone_aad
from app.schemas.auth import RegistrationIn

# Registration states reported to the frontend.
NEEDS_CONSENT = "needs_consent"
NEEDS_REGISTRATION = "needs_registration"
PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"

# Hashed once so a sign-in for an unknown email costs the same as a wrong
# password (no timing hint about which emails exist).
_DUMMY_HASH = hash_password("not-a-real-password-0000")


class AccountError(Exception):
    """A refused account action, with the HTTP status the route returns."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class AccountStore(Protocol):
    def get_user(self, user_id: uuid.UUID) -> User | None: ...
    def find_by_email(self, email: str) -> User | None: ...
    def find_by_google_sub(self, sub: str) -> User | None: ...
    def add_user(self, user: User) -> User: ...
    def roles(self, user_id: uuid.UUID) -> frozenset[str]: ...
    def add_role(self, user_id: uuid.UUID, role: str) -> None: ...
    def get_profile(self, user_id: uuid.UUID) -> ReviewerProfile | None: ...
    def add_profile(self, profile: ReviewerProfile) -> None: ...
    def profiles(self, status: str | None) -> list[tuple[User, ReviewerProfile]]: ...
    def completed_cases(self) -> dict[uuid.UUID, int]: ...
    def audit(
        self,
        *,
        action: str,
        actor_id: uuid.UUID | None,
        actor_role: str | None,
        outcome: str,
        detail: dict,
    ) -> None: ...


def _now() -> datetime:
    return datetime.now(UTC)


def _normalise_email(email: str) -> str:
    return email.strip().lower()


def registration_status(store: AccountStore, user: User) -> str:
    if user.auth_method in (None, "admin"):
        return APPROVED  # seeded dev users and the admin don't register
    profile = store.get_profile(user.id)
    if profile is None:
        return NEEDS_CONSENT
    if profile.status == "consented":
        return NEEDS_REGISTRATION
    return profile.status


def _new_user(email: str, method: str) -> User:
    return User(
        id=uuid.uuid4(),
        email=email,
        display_name=email,
        status="active",
        is_clinician=False,
        auth_method=method,
    )


# ── sign-in ──


def sign_in_google(store: AccountStore, identity: GoogleIdentity) -> User:
    user = store.find_by_google_sub(identity.sub)
    if user is None:
        user = store.find_by_email(identity.email)
        if user is not None and user.auth_method != "password":
            raise AccountError(409, "this email belongs to an account that cannot use Google")
        if user is not None:
            # Google has verified this address, so it may join the existing
            # password account; either method then reaches the same profile.
            user.google_sub = identity.sub
        else:
            user = _new_user(identity.email, "google")
            user.google_sub = identity.sub
            store.add_user(user)
    store.audit(
        action="login",
        actor_id=user.id,
        actor_role=None,
        outcome="issued",
        detail={"provider": "google"},
    )
    return user


def register_password(store: AccountStore, email: str, password: str) -> User:
    email = _normalise_email(email)
    problem = password_problem(password)
    if problem:
        raise AccountError(422, problem)
    if store.find_by_email(email) is not None:
        raise AccountError(409, "an account with this email already exists; sign in instead")
    user = _new_user(email, "password")
    user.password_hash = hash_password(password)
    store.add_user(user)
    store.audit(
        action="account",
        actor_id=user.id,
        actor_role=None,
        outcome="created",
        detail={"provider": "password"},
    )
    return user


def sign_in_password(store: AccountStore, email: str, password: str) -> User:
    user = store.find_by_email(_normalise_email(email))
    ok = verify_password(password, user.password_hash if user else _DUMMY_HASH)
    if user is None or not ok or user.auth_method == "admin":
        store.audit(
            action="login",
            actor_id=user.id if user else None,
            actor_role=None,
            outcome="denied",
            detail={"provider": "password"},
        )
        raise AccountError(401, "email or password is incorrect")
    store.audit(
        action="login",
        actor_id=user.id,
        actor_role=None,
        outcome="issued",
        detail={"provider": "password"},
    )
    return user


def sign_in_admin(
    store: AccountStore, username: str, password: str, *, admin_username: str, admin_password: str
) -> User:
    if not admin_username or not admin_password:
        raise AccountError(503, "admin sign-in is not configured (ADMIN_USERNAME / ADMIN_PASSWORD)")
    # Compare both fields in full every time (constant-time).
    user_ok = hmac.compare_digest(username.encode(), admin_username.encode())
    pass_ok = hmac.compare_digest(password.encode(), admin_password.encode())
    if not (user_ok and pass_ok):
        store.audit(
            action="login",
            actor_id=None,
            actor_role="admin",
            outcome="denied",
            detail={"provider": "admin"},
        )
        raise AccountError(401, "username or password is incorrect")
    user = store.find_by_email(admin_username)
    if user is None:
        user = _new_user(admin_username, "admin")
        user.display_name = "Administrator"
        store.add_user(user)
    elif user.auth_method != "admin":
        raise AccountError(409, "ADMIN_USERNAME matches a non-admin account")
    if "admin" not in store.roles(user.id):
        store.add_role(user.id, "admin")
    store.audit(
        action="login",
        actor_id=user.id,
        actor_role="admin",
        outcome="issued",
        detail={"provider": "admin"},
    )
    return user


# ── consent and registration ──


def record_consent(
    store: AccountStore, user: User, consent: ConsentText, *, agree: bool, version: str, sha256: str
) -> str:
    """Records an agreement as a new profile; a decline creates nothing.
    Returns the user's registration status afterwards."""
    if registration_status(store, user) != NEEDS_CONSENT:
        raise AccountError(409, "consent has already been recorded")
    if version != consent.version or sha256 != consent.sha256:
        raise AccountError(409, "the consent text has changed; please read it again")
    store.audit(
        action="account",
        actor_id=user.id,
        actor_role=None,
        outcome="consent_given" if agree else "consent_declined",
        detail={"consent_version": consent.version, "consent_sha256": consent.sha256},
    )
    if not agree:
        return NEEDS_CONSENT
    store.add_profile(
        ReviewerProfile(
            user_id=user.id,
            status="consented",
            consent_version=consent.version,
            consent_sha256=consent.sha256,
            consented_at=_now(),
        )
    )
    return NEEDS_REGISTRATION


def resubmissions_left(profile: ReviewerProfile, max_resubmissions: int) -> int:
    """Resubmissions still allowed (DEVIATIONS.md #286). The first
    submission is not a resubmission."""
    used = max(0, (profile.submission_count or 0) - 1)
    return max(0, max_resubmissions - used)


def submit_registration(
    store: AccountStore, user: User, form: RegistrationIn, *, max_resubmissions: int
) -> str:
    """First submission, or a resubmission after a rejection (DEVIATIONS.md
    #284): the earlier consent still stands, the previous decision is kept
    as `last_rejected_at`, and the registration goes back to pending. At most
    `max_resubmissions` resubmissions (#286)."""
    status = registration_status(store, user)
    if status == NEEDS_CONSENT:
        raise AccountError(409, "consent must be given before registration")
    if status not in (NEEDS_REGISTRATION, REJECTED):
        raise AccountError(409, "registration has already been submitted")
    profile = store.get_profile(user.id)
    assert profile is not None
    resubmission = status == REJECTED
    if resubmission and resubmissions_left(profile, max_resubmissions) == 0:
        raise AccountError(409, "no resubmissions left; please contact the study team")
    if resubmission:
        profile.last_rejected_at = profile.decided_at
        profile.decided_at = None
        profile.decided_by = None
    profile.submission_count = (profile.submission_count or 0) + 1
    profile.sex = form.sex
    profile.cadre = form.cadre
    profile.expertise_level = form.expertise_level
    profile.years_post_internship = form.years_post_internship
    profile.organisation = form.organisation
    profile.role_title = form.role_title
    profile.phone_enc = (
        None
        if form.safaricom_phone is None
        else get_crypto().encrypt(
            form.safaricom_phone.encode("utf-8"), aad=reviewer_phone_aad(user.id)
        )
    )
    profile.status = PENDING
    profile.submitted_at = _now()
    store.audit(
        action="account",
        actor_id=user.id,
        actor_role=None,
        outcome="registration_resubmitted" if resubmission else "registration_submitted",
        detail={"submission_count": profile.submission_count},
    )
    return PENDING


def rejection_reason(store: AccountStore, user: User) -> str | None:
    """The admin's reason, only while the registration is rejected."""
    profile = store.get_profile(user.id)
    if profile is None or profile.status != REJECTED:
        return None
    return profile.rejection_reason


def own_registration(store: AccountStore, user: User) -> dict[str, object] | None:
    """The user's own submitted answers (phone decrypted), to prefill a
    resubmission; None before the form was ever submitted."""
    profile = store.get_profile(user.id)
    if profile is None or profile.submitted_at is None:
        return None
    return {
        "sex": profile.sex,
        "cadre": profile.cadre,
        "expertise_level": profile.expertise_level,
        "years_post_internship": profile.years_post_internship,
        "organisation": profile.organisation,
        "role_title": profile.role_title,
        "safaricom_phone": decrypt_phone(profile),
    }


# ── admin ──


REJECTION_REASON_MIN, REJECTION_REASON_MAX = 5, 500


def decide(
    store: AccountStore,
    admin_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    approve: bool,
    reason: str | None = None,
) -> str:
    """Approve, or reject with a reason the reviewer will see (DEVIATIONS.md
    #285)."""
    profile = store.get_profile(user_id)
    if profile is None or profile.status != PENDING:
        raise AccountError(409, "no pending registration for this user")
    if not approve:
        reason = (reason or "").strip()
        if not REJECTION_REASON_MIN <= len(reason) <= REJECTION_REASON_MAX:
            raise AccountError(
                422,
                f"a rejection needs a reason of {REJECTION_REASON_MIN} to "
                f"{REJECTION_REASON_MAX} characters",
            )
        profile.rejection_reason = reason
    user = store.get_user(user_id)
    assert user is not None
    profile.status = APPROVED if approve else REJECTED
    profile.decided_at = _now()
    profile.decided_by = admin_id
    if approve:
        # Reviewers must be clinicians (ARCH-011, PRD-086).
        user.is_clinician = True
        store.add_role(user_id, "reviewer")
    store.audit(
        action="account",
        actor_id=admin_id,
        actor_role="admin",
        outcome="registration_approved" if approve else "registration_rejected",
        # The reason text itself stays on the profile, not in the audit log.
        detail={"user_id": str(user_id), "reason_given": not approve},
    )
    return profile.status


def reset_password(
    store: AccountStore, admin_id: uuid.UUID, user_id: uuid.UUID, new_password: str
) -> None:
    """Admin sets a new password for an email + password reviewer (there is
    no email service for self-service resets)."""
    user = store.get_user(user_id)
    if user is None or user.auth_method != "password":
        raise AccountError(409, "this user does not sign in with a password")
    problem = password_problem(new_password)
    if problem:
        raise AccountError(422, problem)
    user.password_hash = hash_password(new_password)
    store.audit(
        action="account",
        actor_id=admin_id,
        actor_role="admin",
        outcome="password_reset",
        detail={"user_id": str(user_id)},
    )


def decrypt_phone(profile: ReviewerProfile) -> str | None:
    if profile.phone_enc is None:
        return None
    return get_crypto().decrypt(profile.phone_enc, aad=reviewer_phone_aad(profile.user_id)).decode()


@dataclass(frozen=True)
class Progress:
    user: User
    profile: ReviewerProfile
    completed_cases: int
    threshold_met: bool


def reviewer_progress(store: AccountStore, *, threshold: int) -> list[Progress]:
    """Approved reviewers with the number of cases they have completed (one
    submitted rating round per review item), most first."""
    counts = store.completed_cases()
    rows = [
        Progress(
            user=user,
            profile=profile,
            completed_cases=counts.get(user.id, 0),
            threshold_met=counts.get(user.id, 0) >= threshold,
        )
        for user, profile in store.profiles(APPROVED)
    ]
    return sorted(rows, key=lambda r: (-r.completed_cases, r.user.email))


# ── SQL store ──


class SqlAccountStore:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_user(self, user_id: uuid.UUID) -> User | None:
        return self.session.get(User, user_id)

    def find_by_email(self, email: str) -> User | None:
        return self.session.execute(
            select(User).where(func.lower(User.email) == email.lower())
        ).scalar_one_or_none()

    def find_by_google_sub(self, sub: str) -> User | None:
        return self.session.execute(select(User).where(User.google_sub == sub)).scalar_one_or_none()

    def add_user(self, user: User) -> User:
        self.session.add(user)
        self.session.flush()
        return user

    def roles(self, user_id: uuid.UUID) -> frozenset[str]:
        stmt = select(UserRole.role_code).where(UserRole.user_id == user_id)
        return frozenset(self.session.execute(stmt).scalars().all())

    def add_role(self, user_id: uuid.UUID, role: str) -> None:
        if role not in self.roles(user_id):
            self.session.add(UserRole(user_id=user_id, role_code=role))
            self.session.flush()

    def get_profile(self, user_id: uuid.UUID) -> ReviewerProfile | None:
        return self.session.get(ReviewerProfile, user_id)

    def add_profile(self, profile: ReviewerProfile) -> None:
        self.session.add(profile)
        self.session.flush()

    def profiles(self, status: str | None) -> list[tuple[User, ReviewerProfile]]:
        stmt = select(User, ReviewerProfile).join(
            ReviewerProfile, ReviewerProfile.user_id == User.id
        )
        if status is not None:
            stmt = stmt.where(ReviewerProfile.status == status)
        return [(u, p) for u, p in self.session.execute(stmt.order_by(ReviewerProfile.created_at))]

    def completed_cases(self) -> dict[uuid.UUID, int]:
        stmt = select(
            RatingRound.rater_id, func.count(func.distinct(RatingRound.result_id))
        ).group_by(RatingRound.rater_id)
        return {rater: n for rater, n in self.session.execute(stmt)}

    def audit(
        self,
        *,
        action: str,
        actor_id: uuid.UUID | None,
        actor_role: str | None,
        outcome: str,
        detail: dict,
    ) -> None:
        write_event(
            self.session,
            action=action,
            actor_id=actor_id,
            actor_role=actor_role,
            outcome=outcome,
            detail=detail,
        )
