"""Reviewer sign-in, consent, registration, admin approval and progress
(DEVIATIONS.md #282). An in-memory AccountStore stands in for Postgres; the
Google key set is a locally generated RSA key (no network)."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import jwt
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.deps import Principal, current_principal
from app.api.routes.auth import get_account_store
from app.auth import accounts, google
from app.auth.accounts import AccountError
from app.auth.consent import ConsentNotAttested, load_consent
from app.auth.google import GoogleTokenError, verify_google_id_token
from app.auth.passwords import hash_password, verify_password
from app.auth.rate_limit import (
    MemoryCounter,
    RedisCounter,
    SignInLimiter,
    SignInLimits,
    account_key,
    get_signin_limiter,
)
from app.config import get_settings
from app.db.models.iam import ReviewerProfile, User
from app.main import app as fastapi_app
from app.schemas.auth import RegistrationIn

REAL_CONSENT = Path(__file__).resolve().parents[2] / "data" / "reviewer_consent.yaml"
CLIENT_ID = "test-client.apps.googleusercontent.com"
ADMIN = {"admin_username": "study-admin", "admin_password": "a-long-admin-password"}
REASON = "Please give your current organisation in full."

FORM = {
    "sex": "female",
    "cadre": "medical_officer",
    "expertise_level": "general_practitioner",
    "years_post_internship": 3,
    "organisation": "Kenyatta Hospital",
    "role_title": "Medical officer",
    "safaricom_phone": "0712345678",
}


class FakeStore:
    def __init__(self) -> None:
        self.users: dict[uuid.UUID, User] = {}
        self.user_roles: dict[uuid.UUID, set[str]] = {}
        self.profile_rows: dict[uuid.UUID, ReviewerProfile] = {}
        self.completed: dict[uuid.UUID, int] = {}
        self.events: list[dict] = []

    def get_user(self, user_id: uuid.UUID) -> User | None:
        return self.users.get(user_id)

    def find_by_email(self, email: str) -> User | None:
        return next((u for u in self.users.values() if u.email.lower() == email.lower()), None)

    def find_by_google_sub(self, sub: str) -> User | None:
        return next((u for u in self.users.values() if u.google_sub == sub), None)

    def add_user(self, user: User) -> User:
        self.users[user.id] = user
        return user

    def roles(self, user_id: uuid.UUID) -> frozenset[str]:
        return frozenset(self.user_roles.get(user_id, set()))

    def add_role(self, user_id: uuid.UUID, role: str) -> None:
        self.user_roles.setdefault(user_id, set()).add(role)

    def get_profile(self, user_id: uuid.UUID) -> ReviewerProfile | None:
        return self.profile_rows.get(user_id)

    def add_profile(self, profile: ReviewerProfile) -> None:
        self.profile_rows[profile.user_id] = profile

    def profiles(self, status: str | None) -> list[tuple[User, ReviewerProfile]]:
        return [
            (self.users[uid], p)
            for uid, p in self.profile_rows.items()
            if status is None or p.status == status
        ]

    def completed_cases(self) -> dict[uuid.UUID, int]:
        return dict(self.completed)

    def audit(self, **event: object) -> None:
        self.events.append(event)


def _attested_consent(tmp_path: Path, **changes: object) -> Path:
    data = yaml.safe_load(REAL_CONSENT.read_text(encoding="utf-8"))
    data.update(
        authored_by="Test Operator",
        authored_role="Tester",
        authored_date="2026-10-07",
        ethics_approved_version="test",
    )
    for section in data["sections"]:
        section["body"] = section["body"].split("[CONFIRM")[0].strip() or "Filled in."
    data.update(changes)
    p = tmp_path / "consent.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


# ── passwords ──


def test_password_hash_round_trip() -> None:
    stored = hash_password("correct horse battery")
    assert stored.startswith("scrypt$") and "correct" not in stored
    assert verify_password("correct horse battery", stored)
    assert not verify_password("wrong horse battery", stored)
    assert not verify_password("anything", None)
    assert not verify_password("anything", "not$a$valid$hash")


# ── consent text ──


def test_the_shipped_consent_is_attested_and_finished() -> None:
    """The operator finalised data/reviewer_consent.yaml on 2026-10-12."""
    consent = load_consent(REAL_CONSENT)
    assert consent.ethics_approved_version and len(consent.sha256) == 64


@pytest.mark.parametrize(
    "field", ["authored_by", "authored_role", "authored_date", "ethics_approved_version"]
)
def test_unattested_consent_is_refused(tmp_path: Path, field: str) -> None:
    with pytest.raises(ConsentNotAttested, match=field):
        load_consent(_attested_consent(tmp_path, **{field: "TODO_CONFIRM"}))


def test_attested_consent_loads_with_its_hash(tmp_path: Path) -> None:
    consent = load_consent(_attested_consent(tmp_path))
    assert consent.version and len(consent.sha256) == 64 and consent.sections


def test_unresolved_confirm_items_are_refused(tmp_path: Path) -> None:
    path = _attested_consent(tmp_path, agreement="I agree [CONFIRM: wording]")
    with pytest.raises(ConsentNotAttested, match="CONFIRM"):
        load_consent(path)


# ── Google ID tokens ──


@pytest.fixture
def google_key(monkeypatch: pytest.MonkeyPatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    class _Jwks:
        def get_signing_key_from_jwt(self, token: str):  # noqa: ANN202, ARG002
            return type("K", (), {"key": key.public_key()})()

    monkeypatch.setattr(google, "_JWKS_CLIENT_FN", lambda: _Jwks())
    return key


def _id_token(key, **claims: object) -> str:  # noqa: ANN001
    now = int(time.time())
    payload = {
        "iss": "https://accounts.google.com",
        "aud": CLIENT_ID,
        "sub": "google-sub-1",
        "email": "Reviewer@Gmail.com",
        "email_verified": True,
        "iat": now,
        "exp": now + 600,
        **claims,
    }
    return jwt.encode(payload, key, algorithm="RS256")


def test_a_valid_google_token_gives_the_lowercased_identity(google_key) -> None:  # noqa: ANN001
    identity = verify_google_id_token(_id_token(google_key), client_id=CLIENT_ID)
    assert identity.sub == "google-sub-1" and identity.email == "reviewer@gmail.com"


@pytest.mark.parametrize(
    "claims",
    [
        {"aud": "another-client"},
        {"iss": "https://evil.example"},
        {"email_verified": False},
        {"exp": int(time.time()) - 10},
    ],
)
def test_bad_google_tokens_are_refused(google_key, claims: dict) -> None:  # noqa: ANN001
    with pytest.raises(GoogleTokenError):
        verify_google_id_token(_id_token(google_key, **claims), client_id=CLIENT_ID)


def test_google_sign_in_needs_a_client_id(google_key) -> None:  # noqa: ANN001
    with pytest.raises(GoogleTokenError, match="not configured"):
        verify_google_id_token(_id_token(google_key), client_id="")


# ── registration form validation ──


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("safaricom_phone", "071234567"),  # 9 digits
        ("safaricom_phone", "07123456789"),  # 11 digits
        ("safaricom_phone", "07123a5678"),
        ("safaricom_phone", "+254712345"),
        ("safaricom_phone", "0812345678"),  # not a Kenyan mobile prefix (#287)
        ("years_post_internship", 3.5),
        ("years_post_internship", "3"),
        ("years_post_internship", -1),
        ("organisation", "KNH"),  # 3 characters
        ("role_title", "x" * 31),
        ("cadre", "surgeon"),
        ("cadre", "nurse"),  # dropped from the list (#288)
        ("sex", "other"),
        ("expertise_level", ""),
    ],
)
def test_registration_form_rejects(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        RegistrationIn(**{**FORM, field: value})


def test_registration_form_accepts_a_leading_zero_and_trims_text() -> None:
    form = RegistrationIn(**{**FORM, "organisation": "  Kenyatta Hospital  "})
    assert form.safaricom_phone == "0712345678" and form.organisation == "Kenyatta Hospital"


def test_every_registration_field_but_the_phone_is_mandatory() -> None:
    for field in FORM:
        if field == "safaricom_phone":
            continue
        with pytest.raises(ValidationError):
            RegistrationIn(**{k: v for k, v in FORM.items() if k != field})


@pytest.mark.parametrize(
    ("cadre", "ok"),
    [
        ("medical_officer", True),
        ("other", True),
        ("neonatologist", False),
        ("paediatrician", False),
        ("clinical_pharmacist", False),
    ],
)
def test_general_practitioner_only_for_medical_officer_or_other(cadre: str, ok: bool) -> None:
    """DEVIATIONS.md #288."""
    specify = {"cadre_other": "Clinical officer"} if cadre == "other" else {}
    form = {**FORM, **specify, "cadre": cadre, "expertise_level": "general_practitioner"}
    if ok:
        assert RegistrationIn(**form).cadre == cadre
    else:
        with pytest.raises(ValidationError, match="not open to this cadre"):
            RegistrationIn(**form)
    for level in ("registrar", "specialist"):
        assert RegistrationIn(**{**FORM, **specify, "cadre": cadre, "expertise_level": level})


def test_other_cadre_must_be_specified() -> None:
    """DEVIATIONS.md #289."""
    other = {**FORM, "cadre": "other"}
    for missing in ({}, {"cadre_other": None}, {"cadre_other": "   "}, {"cadre_other": "CO"}):
        with pytest.raises(ValidationError, match="specify your cadre"):
            RegistrationIn(**other, **missing)
    with pytest.raises(ValidationError):
        RegistrationIn(**other, cadre_other="x" * 31)
    assert RegistrationIn(**other, cadre_other="  Nurse  ").cadre_other == "Nurse"


def test_cadre_other_is_dropped_for_a_listed_cadre() -> None:
    form = RegistrationIn(**{**FORM, "cadre_other": "left over from Other"})
    assert form.cadre == "medical_officer" and form.cadre_other is None


def test_cadre_other_is_stored_and_returned(tmp_path: Path) -> None:
    store = FakeStore()
    consent = load_consent(_attested_consent(tmp_path))
    user = accounts.register_password(store, "other@example.org", "long-enough-pw")
    accounts.record_consent(
        store, user, consent, agree=True, version=consent.version, sha256=consent.sha256
    )
    form = RegistrationIn(**{**FORM, "cadre": "other", "cadre_other": "Clinical officer"})
    accounts.submit_registration(store, user, form, max_resubmissions=2)
    assert store.get_profile(user.id).cadre_other == "Clinical officer"
    own = accounts.own_registration(store, user)
    assert own is not None and own["cadre_other"] == "Clinical officer"


def test_phone_is_optional_for_participants_outside_kenya() -> None:
    """DEVIATIONS.md #287: only participants in Kenya are reimbursed."""
    no_phone = {k: v for k, v in FORM.items() if k != "safaricom_phone"}
    assert RegistrationIn(**no_phone).safaricom_phone is None
    assert RegistrationIn(**{**FORM, "safaricom_phone": "  "}).safaricom_phone is None
    assert RegistrationIn(**{**FORM, "safaricom_phone": None}).safaricom_phone is None
    assert RegistrationIn(**{**FORM, "safaricom_phone": "0112345678"}).safaricom_phone == (
        "0112345678"
    )


def test_registration_without_a_phone_stores_none(tmp_path: Path) -> None:
    store = FakeStore()
    consent = load_consent(_attested_consent(tmp_path))
    user = accounts.register_password(store, "abroad@example.org", "long-enough-pw")
    accounts.record_consent(
        store, user, consent, agree=True, version=consent.version, sha256=consent.sha256
    )
    form = RegistrationIn(**{**FORM, "safaricom_phone": ""})
    accounts.submit_registration(store, user, form, max_resubmissions=2)
    profile = store.get_profile(user.id)
    assert profile.status == accounts.PENDING
    assert profile.phone_enc is None and accounts.decrypt_phone(profile) is None


# ── account flow ──


def test_password_reviewer_flow_from_sign_up_to_approval(tmp_path: Path) -> None:
    store = FakeStore()
    consent = load_consent(_attested_consent(tmp_path))
    user = accounts.register_password(store, "New.Reviewer@example.org", "long-enough-pw")
    assert user.email == "new.reviewer@example.org"
    assert accounts.registration_status(store, user) == accounts.NEEDS_CONSENT
    assert store.roles(user.id) == frozenset()

    accounts.record_consent(
        store, user, consent, agree=True, version=consent.version, sha256=consent.sha256
    )
    assert accounts.registration_status(store, user) == accounts.NEEDS_REGISTRATION

    accounts.submit_registration(store, user, RegistrationIn(**FORM), max_resubmissions=2)
    profile = store.get_profile(user.id)
    assert profile.status == accounts.PENDING
    assert b"0712345678" not in profile.phone_enc  # stored encrypted
    assert accounts.decrypt_phone(profile) == "0712345678"

    admin_id = uuid.uuid4()
    accounts.decide(store, admin_id, user.id, approve=True)
    assert accounts.registration_status(store, user) == accounts.APPROVED
    assert store.roles(user.id) == {"reviewer"} and user.is_clinician
    assert profile.decided_by == admin_id


def test_declining_consent_records_nothing_but_the_decline(tmp_path: Path) -> None:
    store = FakeStore()
    consent = load_consent(_attested_consent(tmp_path))
    user = accounts.register_password(store, "a@example.org", "long-enough-pw")
    status = accounts.record_consent(
        store, user, consent, agree=False, version=consent.version, sha256=consent.sha256
    )
    assert status == accounts.NEEDS_CONSENT and store.get_profile(user.id) is None
    assert store.events[-1]["outcome"] == "consent_declined"


def test_consent_to_an_outdated_text_is_refused(tmp_path: Path) -> None:
    store = FakeStore()
    consent = load_consent(_attested_consent(tmp_path))
    user = accounts.register_password(store, "a@example.org", "long-enough-pw")
    with pytest.raises(AccountError) as exc:
        accounts.record_consent(store, user, consent, agree=True, version="old", sha256="x" * 64)
    assert exc.value.status == 409


def test_registration_before_consent_or_twice_is_refused(tmp_path: Path) -> None:
    store = FakeStore()
    consent = load_consent(_attested_consent(tmp_path))
    user = accounts.register_password(store, "a@example.org", "long-enough-pw")
    with pytest.raises(AccountError):
        accounts.submit_registration(store, user, RegistrationIn(**FORM), max_resubmissions=2)
    accounts.record_consent(
        store, user, consent, agree=True, version=consent.version, sha256=consent.sha256
    )
    accounts.submit_registration(store, user, RegistrationIn(**FORM), max_resubmissions=2)
    with pytest.raises(AccountError):
        accounts.submit_registration(store, user, RegistrationIn(**FORM), max_resubmissions=2)


def test_rejection_grants_no_role(tmp_path: Path) -> None:
    store = FakeStore()
    consent = load_consent(_attested_consent(tmp_path))
    user = accounts.register_password(store, "a@example.org", "long-enough-pw")
    accounts.record_consent(
        store, user, consent, agree=True, version=consent.version, sha256=consent.sha256
    )
    accounts.submit_registration(store, user, RegistrationIn(**FORM), max_resubmissions=2)
    accounts.decide(store, uuid.uuid4(), user.id, approve=False, reason=REASON)
    assert accounts.registration_status(store, user) == accounts.REJECTED
    assert store.roles(user.id) == frozenset() and not user.is_clinician


def test_password_sign_in() -> None:
    store = FakeStore()
    accounts.register_password(store, "a@example.org", "long-enough-pw")
    assert (
        accounts.sign_in_password(store, "A@Example.org", "long-enough-pw").email == "a@example.org"
    )
    for email, password in (("a@example.org", "wrong-password"), ("nobody@example.org", "x" * 12)):
        with pytest.raises(AccountError) as exc:
            accounts.sign_in_password(store, email, password)
        assert exc.value.status == 401


def test_short_passwords_and_duplicate_emails_are_refused() -> None:
    store = FakeStore()
    with pytest.raises(AccountError) as exc:
        accounts.register_password(store, "a@example.org", "short")
    assert exc.value.status == 422
    accounts.register_password(store, "a@example.org", "long-enough-pw")
    with pytest.raises(AccountError) as exc:
        accounts.register_password(store, "A@example.org", "long-enough-pw")
    assert exc.value.status == 409


def test_google_sign_in_creates_once_and_joins_a_password_account() -> None:
    store = FakeStore()
    g = google.GoogleIdentity(sub="s1", email="g@gmail.com")
    first = accounts.sign_in_google(store, g)
    assert accounts.sign_in_google(store, g).id == first.id and first.auth_method == "google"

    pw = accounts.register_password(store, "both@example.org", "long-enough-pw")
    joined = accounts.sign_in_google(
        store, google.GoogleIdentity(sub="s2", email="both@example.org")
    )
    assert joined.id == pw.id and pw.google_sub == "s2"


def test_admin_sign_in() -> None:
    store = FakeStore()
    with pytest.raises(AccountError) as exc:
        accounts.sign_in_admin(store, "x", "y", admin_username="", admin_password="")
    assert exc.value.status == 503
    with pytest.raises(AccountError) as exc:
        accounts.sign_in_admin(store, "study-admin", "wrong", **ADMIN)
    assert exc.value.status == 401
    admin = accounts.sign_in_admin(store, "study-admin", "a-long-admin-password", **ADMIN)
    assert store.roles(admin.id) == {"admin"}
    assert accounts.registration_status(store, admin) == accounts.APPROVED
    again = accounts.sign_in_admin(store, "study-admin", "a-long-admin-password", **ADMIN)
    assert again.id == admin.id
    with pytest.raises(AccountError):  # the admin cannot use the password form
        accounts.sign_in_password(store, "study-admin", "a-long-admin-password")


def test_progress_counts_cases_and_marks_the_threshold(tmp_path: Path) -> None:
    store = FakeStore()
    consent = load_consent(_attested_consent(tmp_path))
    ids = []
    for i in range(3):
        u = accounts.register_password(store, f"r{i}@example.org", "long-enough-pw")
        accounts.record_consent(
            store, u, consent, agree=True, version=consent.version, sha256=consent.sha256
        )
        accounts.submit_registration(store, u, RegistrationIn(**FORM), max_resubmissions=2)
        accounts.decide(store, uuid.uuid4(), u.id, approve=True)
        ids.append(u.id)
    store.completed = {ids[0]: 5, ids[1]: 4}
    rows = accounts.reviewer_progress(store, threshold=5)
    assert [(r.completed_cases, r.threshold_met) for r in rows] == [
        (5, True),
        (4, False),
        (0, False),
    ]


def test_admin_password_reset_only_for_password_users() -> None:
    store = FakeStore()
    user = accounts.register_password(store, "a@example.org", "long-enough-pw")
    accounts.reset_password(store, uuid.uuid4(), user.id, "a-new-long-password")
    assert accounts.sign_in_password(store, "a@example.org", "a-new-long-password")
    g = accounts.sign_in_google(store, google.GoogleIdentity(sub="s", email="g@gmail.com"))
    with pytest.raises(AccountError):
        accounts.reset_password(store, uuid.uuid4(), g.id, "a-new-long-password")


# ── routes ──


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = FakeStore()
    monkeypatch.setenv("REVIEWER_CONSENT_PATH", str(_attested_consent(tmp_path)))
    monkeypatch.setenv("ADMIN_USERNAME", ADMIN["admin_username"])
    monkeypatch.setenv("ADMIN_PASSWORD", ADMIN["admin_password"])
    get_settings.cache_clear()
    fastapi_app.dependency_overrides[get_account_store] = lambda: store
    limiter = SignInLimiter(MemoryCounter(), SignInLimits(5, 20, 900))
    fastapi_app.dependency_overrides[get_signin_limiter] = lambda: limiter
    yield TestClient(fastapi_app), store
    fastapi_app.dependency_overrides.clear()
    get_settings.cache_clear()


def _as(user_id: str, *roles: str) -> None:
    fastapi_app.dependency_overrides[current_principal] = lambda: Principal(
        user_id=user_id, roles=frozenset(roles)
    )


def test_route_flow_sign_up_consent_register_approve(api) -> None:  # noqa: ANN001
    client, store = api
    r = client.post(
        "/api/auth/password/register", json={"email": "r@example.org", "password": "long-enough-pw"}
    )
    assert r.status_code == 200 and r.json()["registration_status"] == "needs_consent"
    assert r.json()["roles"] == []
    uid = r.json()["user_id"]
    _as(uid)

    consent = client.get("/api/auth/consent").json()
    r = client.post(
        "/api/auth/consent",
        json={"agree": True, "version": consent["version"], "sha256": consent["sha256"]},
    )
    assert r.json()["registration_status"] == "needs_registration"
    assert (
        client.post("/api/auth/registration", json={**FORM, "safaricom_phone": "12"}).status_code
        == 422
    )
    assert (
        client.post("/api/auth/registration", json=FORM).json()["registration_status"] == "pending"
    )

    # a reviewer cannot use the admin queues
    assert client.get("/api/admin/reviewers").status_code == 403

    _as(str(uuid.uuid4()), "admin")
    pending = client.get("/api/admin/reviewers").json()
    assert [p["user_id"] for p in pending] == [uid]
    assert pending[0]["safaricom_phone"] == "0712345678"
    assert client.post(f"/api/admin/reviewers/{uid}/approve").json()["status"] == "approved"
    assert client.get("/api/admin/reviewers").json() == []

    store.completed = {uuid.UUID(uid): 5}
    progress = client.get("/api/admin/reviewers/progress").json()
    assert progress["threshold"] == 5
    assert progress["reviewers"][0]["completed_cases"] == 5
    assert progress["reviewers"][0]["threshold_met"] is True

    _as(uid)
    r = client.post("/api/auth/refresh")
    assert r.json()["roles"] == ["reviewer"] and r.json()["registration_status"] == "approved"


def test_route_admin_login_and_config(api) -> None:  # noqa: ANN001
    client, _ = api
    cfg = client.get("/api/auth/config").json()
    assert cfg["admin_sign_in"] is True and cfg["password_sign_in"] is True
    bad = client.post("/api/auth/admin/login", json={"username": "study-admin", "password": "no"})
    assert bad.status_code == 401
    ok = client.post(
        "/api/auth/admin/login",
        json={"username": "study-admin", "password": "a-long-admin-password"},
    )
    assert ok.status_code == 200 and ok.json()["roles"] == ["admin"]


def test_route_consent_is_closed_while_unattested(
    api,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,  # noqa: ANN001
) -> None:
    client, _ = api
    unattested = _attested_consent(tmp_path, authored_by="TODO_CONFIRM")
    monkeypatch.setenv("REVIEWER_CONSENT_PATH", str(unattested))
    get_settings.cache_clear()
    _as(str(uuid.uuid4()))
    r = client.get("/api/auth/consent")
    assert r.status_code == 503 and "consent text is not ready" in r.json()["detail"]


def test_route_registration_options(api) -> None:  # noqa: ANN001
    client, _ = api
    opts = client.get("/api/auth/registration/options").json()
    assert [o["value"] for o in opts["sex"]] == ["female", "male"]
    assert [o["label"] for o in opts["cadre"]] == [
        "Neonatologist",
        "Paediatrician",
        "Clinical Pharmacist",
        "Medical Officer",
        "Other",
    ]
    cadres = {o["value"]: o["cadres"] for o in opts["expertise_level"]}
    assert cadres == {
        "general_practitioner": ["medical_officer", "other"],
        "registrar": None,
        "specialist": None,
    }
    assert (opts["text_min"], opts["text_max"]) == (4, 30)


# ── failed sign-in limits (DEVIATIONS.md #283) ──


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_limiter_locks_an_account_after_max_failures_until_the_window_ends() -> None:
    clock = _Clock()
    limiter = SignInLimiter(MemoryCounter(clock), SignInLimits(3, 100, 900))
    acct = account_key("password", "a@example.org")
    for _ in range(2):
        limiter.failed(acct, "10.0.0.1")
    assert limiter.retry_after(acct, "10.0.0.1") is None
    limiter.failed(acct, "10.0.0.1")
    wait = limiter.retry_after(acct, "10.0.0.1")
    assert wait is not None and 0 < wait <= 901
    clock.t += 901
    assert limiter.retry_after(acct, "10.0.0.1") is None  # window over


def test_limiter_success_clears_the_account_but_not_the_ip() -> None:
    limiter = SignInLimiter(MemoryCounter(), SignInLimits(2, 3, 900))
    a, b = account_key("password", "a@x.org"), account_key("password", "b@x.org")
    limiter.failed(a, "10.0.0.1")
    limiter.succeeded(a)
    limiter.failed(a, "10.0.0.1")
    assert limiter.retry_after(a, None) is None  # 1 failure since the success
    limiter.failed(b, "10.0.0.1")  # third failure from this IP
    assert limiter.retry_after(account_key("password", "c@x.org"), "10.0.0.1") is not None
    assert limiter.retry_after(account_key("password", "c@x.org"), "10.0.0.2") is None


def test_account_keys_hide_the_email_and_ignore_case() -> None:
    key = account_key("password", "A@Example.org ")
    assert key == account_key("password", "a@example.org") and "example" not in key


def test_redis_outage_falls_back_to_memory() -> None:
    # Nothing listens on port 1: every call fails at once and degrades.
    fallback = MemoryCounter()
    limiter = SignInLimiter(
        RedisCounter("redis://127.0.0.1:1/0", fallback=fallback), SignInLimits(2, 100, 900)
    )
    acct = account_key("password", "a@example.org")
    limiter.failed(acct, None)
    limiter.failed(acct, None)
    assert limiter.retry_after(acct, None) is not None
    assert fallback.get(acct)[0] == 2


def test_route_password_sign_in_is_locked_after_five_failures(api) -> None:  # noqa: ANN001
    client, store = api
    creds = {"email": "r@example.org", "password": "long-enough-pw"}
    client.post("/api/auth/password/register", json=creds)
    wrong = {**creds, "password": "not-the-password"}
    for _ in range(5):
        assert client.post("/api/auth/password/login", json=wrong).status_code == 401
    locked = client.post("/api/auth/password/login", json=creds)  # even the right password
    assert locked.status_code == 429
    assert int(locked.headers["Retry-After"]) > 0
    assert "too many failed sign-in attempts" in locked.json()["detail"]
    assert store.events[-1]["outcome"] == "rate_limited"
    # another account from the same address is not locked yet
    other = {"email": "s@example.org", "password": "long-enough-pw"}
    client.post("/api/auth/password/register", json=other)
    assert client.post("/api/auth/password/login", json=other).status_code == 200


def test_route_success_resets_the_account_count(api) -> None:  # noqa: ANN001
    client, _ = api
    creds = {"email": "r@example.org", "password": "long-enough-pw"}
    client.post("/api/auth/password/register", json=creds)
    wrong = {**creds, "password": "not-the-password"}
    for _ in range(4):
        client.post("/api/auth/password/login", json=wrong)
    assert client.post("/api/auth/password/login", json=creds).status_code == 200
    for _ in range(4):
        assert client.post("/api/auth/password/login", json=wrong).status_code == 401


def test_route_admin_sign_in_is_limited_whatever_username_is_tried(api) -> None:  # noqa: ANN001
    client, _ = api
    for i in range(5):
        r = client.post("/api/auth/admin/login", json={"username": f"guess{i}", "password": "x"})
        assert r.status_code == 401
    r = client.post(
        "/api/auth/admin/login",
        json={"username": "study-admin", "password": "a-long-admin-password"},
    )
    assert r.status_code == 429


def test_client_ip_trusts_the_address_nginx_appended() -> None:
    from starlette.requests import Request  # noqa: PLC0415

    from app.api.routes.auth import client_ip  # noqa: PLC0415

    def req(headers: list[tuple[bytes, bytes]]) -> Request:
        return Request({"type": "http", "headers": headers, "client": ("172.18.0.5", 1234)})

    spoofed = req([(b"x-forwarded-for", b"1.2.3.4, 41.90.1.2")])
    assert client_ip(spoofed) == "41.90.1.2"
    assert client_ip(req([])) == "172.18.0.5"


# ── resubmission after rejection (DEVIATIONS.md #284) ──


def _rejected_reviewer(store: FakeStore, tmp_path: Path) -> User:
    consent = load_consent(_attested_consent(tmp_path))
    user = accounts.register_password(store, "r@example.org", "long-enough-pw")
    accounts.record_consent(
        store, user, consent, agree=True, version=consent.version, sha256=consent.sha256
    )
    accounts.submit_registration(store, user, RegistrationIn(**FORM), max_resubmissions=2)
    accounts.decide(store, uuid.uuid4(), user.id, approve=False, reason=REASON)
    return user


def test_a_rejected_reviewer_can_resubmit_and_be_approved(tmp_path: Path) -> None:
    store = FakeStore()
    user = _rejected_reviewer(store, tmp_path)
    profile = store.get_profile(user.id)
    rejected_at = profile.decided_at
    assert profile.submission_count == 1

    changed = RegistrationIn(**{**FORM, "organisation": "Kilifi County Hosp"})
    assert (
        accounts.submit_registration(store, user, changed, max_resubmissions=2) == accounts.PENDING
    )
    assert profile.organisation == "Kilifi County Hosp"
    assert profile.submission_count == 2
    assert profile.last_rejected_at == rejected_at
    assert profile.decided_at is None and profile.decided_by is None
    assert store.events[-1]["outcome"] == "registration_resubmitted"
    assert profile.consent_version  # the earlier consent still stands

    accounts.decide(store, uuid.uuid4(), user.id, approve=True)
    assert store.roles(user.id) == {"reviewer"}


def test_only_rejected_or_unsubmitted_registrations_can_be_submitted(tmp_path: Path) -> None:
    store = FakeStore()
    user = _rejected_reviewer(store, tmp_path)
    accounts.submit_registration(
        store, user, RegistrationIn(**FORM), max_resubmissions=2
    )  # resubmitted: pending
    with pytest.raises(AccountError):  # not again while pending
        accounts.submit_registration(store, user, RegistrationIn(**FORM), max_resubmissions=2)
    accounts.decide(store, uuid.uuid4(), user.id, approve=True)
    with pytest.raises(AccountError):  # nor once approved
        accounts.submit_registration(store, user, RegistrationIn(**FORM), max_resubmissions=2)


def test_own_registration_returns_the_last_answers(tmp_path: Path) -> None:
    store = FakeStore()
    fresh = accounts.register_password(store, "new@example.org", "long-enough-pw")
    assert accounts.own_registration(store, fresh) is None
    user = _rejected_reviewer(store, tmp_path)
    assert accounts.own_registration(store, user) == {**FORM, "cadre_other": None}


def test_route_resubmission(api, tmp_path: Path) -> None:  # noqa: ANN001
    client, store = api
    user = _rejected_reviewer(store, tmp_path)
    # the api fixture's consent file has the same content, so the hash matches
    _as(str(user.id))
    assert client.get("/api/auth/me").json()["registration_status"] == "rejected"
    assert client.get("/api/auth/registration").json() == {**FORM, "cadre_other": None}
    r = client.post("/api/auth/registration", json={**FORM, "role_title": "Senior MO"})
    assert r.json()["registration_status"] == "pending"

    _as(str(uuid.uuid4()), "admin")
    [row] = client.get("/api/admin/reviewers").json()
    assert row["submission_count"] == 2 and row["last_rejected_at"] is not None
    assert row["role_title"] == "Senior MO"


def test_route_own_registration_is_404_before_submitting(api) -> None:  # noqa: ANN001
    client, _ = api
    r = client.post(
        "/api/auth/password/register", json={"email": "n@example.org", "password": "long-enough-pw"}
    )
    _as(r.json()["user_id"])
    assert client.get("/api/auth/registration").status_code == 404


# ── rejection reason (DEVIATIONS.md #285) ──


def _pending_reviewer(store: FakeStore, tmp_path: Path) -> User:
    consent = load_consent(_attested_consent(tmp_path))
    user = accounts.register_password(store, "p@example.org", "long-enough-pw")
    accounts.record_consent(
        store, user, consent, agree=True, version=consent.version, sha256=consent.sha256
    )
    accounts.submit_registration(store, user, RegistrationIn(**FORM), max_resubmissions=2)
    return user


@pytest.mark.parametrize("reason", [None, "", "   ", "no", "x" * 501])
def test_a_rejection_needs_a_reason(tmp_path: Path, reason: str | None) -> None:
    store = FakeStore()
    user = _pending_reviewer(store, tmp_path)
    with pytest.raises(AccountError) as exc:
        accounts.decide(store, uuid.uuid4(), user.id, approve=False, reason=reason)
    assert exc.value.status == 422
    assert accounts.registration_status(store, user) == accounts.PENDING


def test_the_reason_is_shown_only_while_rejected_and_kept_for_the_admin(tmp_path: Path) -> None:
    store = FakeStore()
    user = _pending_reviewer(store, tmp_path)
    assert accounts.rejection_reason(store, user) is None
    accounts.decide(store, uuid.uuid4(), user.id, approve=False, reason=f"  {REASON} ")
    assert accounts.rejection_reason(store, user) == REASON  # trimmed
    assert store.events[-1]["detail"] == {"user_id": str(user.id), "reason_given": True}

    accounts.submit_registration(
        store, user, RegistrationIn(**FORM), max_resubmissions=2
    )  # resubmitted
    assert accounts.rejection_reason(store, user) is None  # pending again
    assert store.get_profile(user.id).rejection_reason == REASON  # admin still sees it


def test_route_rejection_reason_reaches_the_reviewer(api, tmp_path: Path) -> None:  # noqa: ANN001
    client, store = api
    user = _pending_reviewer(store, tmp_path)
    _as(str(uuid.uuid4()), "admin")
    assert client.post(f"/api/admin/reviewers/{user.id}/reject").status_code == 422
    short = client.post(f"/api/admin/reviewers/{user.id}/reject", json={"reason": "no"})
    assert short.status_code == 422
    ok = client.post(f"/api/admin/reviewers/{user.id}/reject", json={"reason": REASON})
    assert ok.json()["status"] == "rejected"
    [row] = client.get("/api/admin/reviewers?status=rejected").json()
    assert row["rejection_reason"] == REASON

    _as(str(user.id))
    me = client.get("/api/auth/me").json()
    assert me["registration_status"] == "rejected" and me["rejection_reason"] == REASON


# ── resubmission limit (DEVIATIONS.md #286) ──


def test_resubmissions_stop_at_the_limit(tmp_path: Path) -> None:
    store = FakeStore()
    user = _rejected_reviewer(store, tmp_path)  # submission 1, rejected
    profile = store.get_profile(user.id)
    assert accounts.resubmissions_left(profile, 2) == 2
    for left_after in (1, 0):
        accounts.submit_registration(store, user, RegistrationIn(**FORM), max_resubmissions=2)
        accounts.decide(store, uuid.uuid4(), user.id, approve=False, reason=REASON)
        assert accounts.resubmissions_left(profile, 2) == left_after
    with pytest.raises(AccountError) as exc:
        accounts.submit_registration(store, user, RegistrationIn(**FORM), max_resubmissions=2)
    assert exc.value.status == 409 and "no resubmissions left" in exc.value.message
    assert profile.submission_count == 3 and profile.status == accounts.REJECTED


def test_zero_resubmissions_makes_the_first_rejection_final(tmp_path: Path) -> None:
    store = FakeStore()
    user = _rejected_reviewer(store, tmp_path)
    with pytest.raises(AccountError):
        accounts.submit_registration(store, user, RegistrationIn(**FORM), max_resubmissions=0)


def test_the_first_submission_is_not_limited(tmp_path: Path) -> None:
    """Even with no resubmissions allowed, the first submission goes through."""
    store = FakeStore()
    consent = load_consent(_attested_consent(tmp_path))
    user = accounts.register_password(store, "first@example.org", "long-enough-pw")
    accounts.record_consent(
        store, user, consent, agree=True, version=consent.version, sha256=consent.sha256
    )
    status = accounts.submit_registration(store, user, RegistrationIn(**FORM), max_resubmissions=0)
    assert status == accounts.PENDING


def test_route_reports_resubmissions_left(api, tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    client, store = api
    monkeypatch.setenv("REVIEWER_MAX_RESUBMISSIONS", "1")
    get_settings.cache_clear()
    user = _rejected_reviewer(store, tmp_path)
    _as(str(user.id))
    assert client.get("/api/auth/me").json()["resubmissions_left"] == 1
    assert client.post("/api/auth/registration", json=FORM).status_code == 200
    me = client.get("/api/auth/me").json()
    assert me["registration_status"] == "pending" and me["resubmissions_left"] is None

    _as(str(uuid.uuid4()), "admin")
    [row] = client.get("/api/admin/reviewers").json()
    assert row["resubmissions_left"] == 0  # a rejection now would be final
    client.post(f"/api/admin/reviewers/{user.id}/reject", json={"reason": REASON})

    _as(str(user.id))
    assert client.get("/api/auth/me").json()["resubmissions_left"] == 0
    blocked = client.post("/api/auth/registration", json=FORM)
    assert blocked.status_code == 409 and "no resubmissions left" in blocked.json()["detail"]
