"""Auth request/response models: dev-login (ARCH-011) and reviewer sign-in,
consent, registration and admin review (DEVIATIONS.md #282)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from app.auth.passwords import MAX_PASSWORD_LENGTH
from app.auth.registration_options import (
    CADRE_OPTIONS,
    EXPERTISE_OPTIONS,
    PHONE_PATTERN,
    SEX_OPTIONS,
    TEXT_MAX,
    TEXT_MIN,
    YEARS_MAX,
    YEARS_MIN,
)


class DevLoginRequest(BaseModel):
    email: str


class DevLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    roles: list[str]


# ── Reviewer sign-in and registration (DEVIATIONS.md #282) ──


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    roles: list[str]
    registration_status: str


class GoogleLoginRequest(BaseModel):
    id_token: str = Field(min_length=1)


class PasswordCredentials(BaseModel):
    email: str = Field(min_length=3, max_length=256, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)


class AdminLoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=256)


class MeResponse(BaseModel):
    user_id: str
    email: str
    roles: list[str]
    auth_method: str | None
    registration_status: str
    # Only while rejected: the admin's reason (DEVIATIONS.md #285) and how
    # many resubmissions remain (#286).
    rejection_reason: str | None = None
    resubmissions_left: int | None = None


class ConsentSectionOut(BaseModel):
    heading: str
    body: str


class ConsentOut(BaseModel):
    version: str
    sha256: str
    title: str
    sections: list[ConsentSectionOut]
    agreement: str


class ConsentDecision(BaseModel):
    agree: bool
    version: str
    sha256: str


class RegistrationOption(BaseModel):
    value: str
    label: str


class RegistrationOptions(BaseModel):
    sex: list[RegistrationOption]
    cadre: list[RegistrationOption]
    expertise_level: list[RegistrationOption]
    years_min: int
    years_max: int
    text_min: int
    text_max: int


class RegistrationIn(BaseModel):
    """Every field mandatory (operator spec) except the phone number, which
    only participants in Kenya give, for reimbursement (DEVIATIONS.md #287).
    Strict types: years must be a JSON integer, not 3.5 or "3"."""

    model_config = ConfigDict(str_strip_whitespace=True)

    sex: str
    cadre: str
    expertise_level: str
    years_post_internship: StrictInt = Field(ge=YEARS_MIN, le=YEARS_MAX)
    organisation: str = Field(min_length=TEXT_MIN, max_length=TEXT_MAX)
    role_title: str = Field(min_length=TEXT_MIN, max_length=TEXT_MAX)
    # Field name kept for API/client stability; Safaricom or Airtel (#287).
    safaricom_phone: str | None = Field(default=None, pattern=PHONE_PATTERN)

    @field_validator("safaricom_phone", mode="before")
    @classmethod
    def _blank_phone(cls, v: object) -> object:
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("sex")
    @classmethod
    def _sex(cls, v: str) -> str:
        return _one_of(v, SEX_OPTIONS, "sex")

    @field_validator("cadre")
    @classmethod
    def _cadre(cls, v: str) -> str:
        return _one_of(v, CADRE_OPTIONS, "cadre")

    @field_validator("expertise_level")
    @classmethod
    def _expertise(cls, v: str) -> str:
        return _one_of(v, EXPERTISE_OPTIONS, "expertise_level")


def _one_of(value: str, options: dict[str, str], name: str) -> str:
    if value not in options:
        raise ValueError(f"{name} must be one of {sorted(options)}")
    return value


class ReviewerRow(BaseModel):
    """One reviewer as the admin sees them."""

    user_id: str
    email: str
    auth_method: str | None
    status: str
    sex: str | None
    cadre: str | None
    expertise_level: str | None
    years_post_internship: int | None
    organisation: str | None
    role_title: str | None
    safaricom_phone: str | None
    consent_version: str
    consented_at: str
    submitted_at: str | None
    decided_at: str | None
    submission_count: int
    last_rejected_at: str | None
    rejection_reason: str | None
    # After a rejection of this submission, how many more are allowed (#286).
    resubmissions_left: int


class RejectRequest(BaseModel):
    """Shown to the reviewer (DEVIATIONS.md #285)."""

    reason: str = Field(min_length=5, max_length=500)


class OwnRegistration(BaseModel):
    """A reviewer's own last submitted answers (DEVIATIONS.md #284)."""

    sex: str | None
    cadre: str | None
    expertise_level: str | None
    years_post_internship: int | None
    organisation: str | None
    role_title: str | None
    safaricom_phone: str | None


class ReviewerProgressRow(BaseModel):
    user_id: str
    email: str
    cadre: str | None
    organisation: str | None
    completed_cases: int
    threshold: int
    threshold_met: bool


class ReviewerProgress(BaseModel):
    threshold: int
    reviewers: list[ReviewerProgressRow]


class PasswordReset(BaseModel):
    new_password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)
