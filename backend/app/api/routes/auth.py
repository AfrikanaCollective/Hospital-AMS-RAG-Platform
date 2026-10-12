"""Sign-in routes (ARCH-011; PRD-086): dev-login, and reviewer sign-in,
consent and registration (DEVIATIONS.md #282, further down).

Dev-JWT login:

POST /auth/dev-login   body: {"email": "..."} -> a signed dev JWT for a
seeded demo user (`scripts/seed_db.py` / `make seed`).

**Not a real authentication flow — dev/CI only.** There is no password: this
mints a token for whoever's email matches a seeded `iam.user` row, embedding
their real seeded roles. It is unconditionally unavailable
(`404 Not Found`, the route "doesn't exist") whenever `AUTH_PROVIDER` is not
`devjwt` — a real deployment sets `AUTH_PROVIDER=oidc` and this endpoint
disappears rather than needing to be separately disabled/firewalled.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.api.deps import Principal, current_principal, get_db, principal_uuid
from app.audit.log import write_event
from app.auth.accounts import (
    AccountError,
    AccountStore,
    SqlAccountStore,
    own_registration,
    record_consent,
    register_password,
    registration_status,
    rejection_reason,
    resubmissions_left,
    sign_in_admin,
    sign_in_google,
    sign_in_password,
    submit_registration,
)
from app.auth.consent import ConsentNotAttested, ConsentText, load_consent
from app.auth.devjwt import mint_token
from app.auth.google import GoogleTokenError, verify_google_id_token
from app.auth.rate_limit import SignInLimiter, account_key, get_signin_limiter
from app.auth.registration_options import (
    CADRE_OPTIONS,
    EXPERTISE_CADRES,
    EXPERTISE_OPTIONS,
    SEX_OPTIONS,
    TEXT_MAX,
    TEXT_MIN,
    YEARS_MAX,
    YEARS_MIN,
)
from app.auth.repository import find_user_with_roles
from app.config import get_settings
from app.db.models.iam import User
from app.schemas.auth import (
    AdminLoginRequest,
    ConsentDecision,
    ConsentOut,
    ConsentSectionOut,
    DevLoginRequest,
    DevLoginResponse,
    GoogleLoginRequest,
    MeResponse,
    OwnRegistration,
    PasswordCredentials,
    RegistrationIn,
    RegistrationOption,
    RegistrationOptions,
    TokenResponse,
)

router = APIRouter()

P = ParamSpec("P")
T = TypeVar("T")


@router.post("/dev-login", response_model=DevLoginResponse)
async def dev_login(body: DevLoginRequest, session: Session = Depends(get_db)) -> DevLoginResponse:
    settings = get_settings()
    if settings.auth_provider != "devjwt":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")

    found = find_user_with_roles(session, body.email)
    if found is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"no seeded user {body.email!r} (run `make seed`)"
        )
    user, roles = found

    token = mint_token(user_id=str(user.id), roles=roles, is_clinician=user.is_clinician)
    write_event(
        session,
        action="login",
        actor_id=user.id,
        actor_role=sorted(roles)[0] if roles else None,
        outcome="issued",
        detail={"provider": "devjwt", "roles": sorted(roles)},
    )
    return DevLoginResponse(access_token=token, user_id=str(user.id), roles=sorted(roles))


# ── Reviewer sign-in, consent and registration (DEVIATIONS.md #282) ──
#
# Routes stay thin; the logic is in app.auth.accounts.


def get_account_store(session: Session = Depends(get_db)) -> AccountStore:
    """Overridden in tests with an in-memory store."""
    return SqlAccountStore(session)


def _token_response(store: AccountStore, user: User) -> TokenResponse:
    roles = store.roles(user.id)
    token = mint_token(user_id=str(user.id), roles=roles, is_clinician=user.is_clinician)
    return TokenResponse(
        access_token=token,
        user_id=str(user.id),
        roles=sorted(roles),
        registration_status=registration_status(store, user),
    )


def _call(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Runs an accounts function, turning AccountError into its HTTP error."""
    try:
        return fn(*args, **kwargs)
    except AccountError as exc:
        raise HTTPException(exc.status, exc.message) from exc


def _current_user(store: AccountStore, principal: Principal) -> User:
    user = store.get_user(principal_uuid(principal))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "unknown user")
    return user


def _consent() -> ConsentText:
    try:
        return load_consent(get_settings().reviewer_consent_path)
    except ConsentNotAttested as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"registration is closed: the consent text is not ready ({exc})",
        ) from exc


@router.get("/config")
async def auth_config() -> dict[str, object]:
    """Which sign-in methods the login page should offer. Public."""
    s = get_settings()
    return {
        "google_client_id": s.google_client_id or None,
        "password_sign_in": True,
        "admin_sign_in": bool(s.admin_username and s.admin_password),
        "dev_login": s.auth_provider == "devjwt",
    }


@router.post("/google", response_model=TokenResponse)
async def google_login(
    body: GoogleLoginRequest, store: AccountStore = Depends(get_account_store)
) -> TokenResponse:
    try:
        identity = verify_google_id_token(body.id_token, client_id=get_settings().google_client_id)
    except GoogleTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc
    user = _call(sign_in_google, store, identity)
    return _token_response(store, user)


@router.post("/password/register", response_model=TokenResponse)
async def password_register(
    body: PasswordCredentials, store: AccountStore = Depends(get_account_store)
) -> TokenResponse:
    user = _call(register_password, store, body.email, body.password)
    return _token_response(store, user)


def client_ip(request: Request) -> str | None:
    """The caller's address. nginx is the only published entry point and
    appends the address it saw to X-Forwarded-For, so the last entry is
    the one to trust; direct calls (tests, in-container) use the socket."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip() or None
    return request.client.host if request.client else None


def _limited(
    limiter: SignInLimiter,
    store: AccountStore,
    *,
    account: str,
    ip: str | None,
    provider: str,
    sign_in: Callable[[], User],
) -> User:
    """Runs a password sign-in under the failed-attempt limits
    (DEVIATIONS.md #283): refused with 429 while locked, without checking the
    password; a wrong password counts against the account and the IP."""
    wait = limiter.retry_after(account, ip)
    if wait is not None:
        store.audit(
            action="login",
            actor_id=None,
            actor_role=None,
            outcome="rate_limited",
            detail={"provider": provider},
        )
        minutes = max(1, -(-wait // 60))
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"too many failed sign-in attempts; try again in {minutes} minute"
            f"{'' if minutes == 1 else 's'}",
            headers={"Retry-After": str(wait)},
        )
    try:
        user = sign_in()
    except AccountError as exc:
        if exc.status == status.HTTP_401_UNAUTHORIZED:
            limiter.failed(account, ip)
        raise HTTPException(exc.status, exc.message) from exc
    limiter.succeeded(account)
    return user


@router.post("/password/login", response_model=TokenResponse)
async def password_login(
    body: PasswordCredentials,
    request: Request,
    store: AccountStore = Depends(get_account_store),
    limiter: SignInLimiter = Depends(get_signin_limiter),
) -> TokenResponse:
    user = _limited(
        limiter,
        store,
        account=account_key("password", body.email),
        ip=client_ip(request),
        provider="password",
        sign_in=lambda: sign_in_password(store, body.email, body.password),
    )
    return _token_response(store, user)


@router.post("/admin/login", response_model=TokenResponse)
async def admin_login(
    body: AdminLoginRequest,
    request: Request,
    store: AccountStore = Depends(get_account_store),
    limiter: SignInLimiter = Depends(get_signin_limiter),
) -> TokenResponse:
    s = get_settings()
    user = _limited(
        limiter,
        store,
        # One counter for the admin login whatever username is typed, so
        # guessing usernames doesn't reset it.
        account=account_key("admin", "admin"),
        ip=client_ip(request),
        provider="admin",
        sign_in=lambda: sign_in_admin(
            store,
            body.username,
            body.password,
            admin_username=s.admin_username,
            admin_password=s.admin_password,
        ),
    )
    return _token_response(store, user)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    principal: Principal = Depends(current_principal),
    store: AccountStore = Depends(get_account_store),
) -> TokenResponse:
    """A new token with the user's current roles, e.g. after approval."""
    return _token_response(store, _current_user(store, principal))


@router.get("/me", response_model=MeResponse)
async def me(
    principal: Principal = Depends(current_principal),
    store: AccountStore = Depends(get_account_store),
) -> MeResponse:
    user = _current_user(store, principal)
    return MeResponse(
        user_id=str(user.id),
        email=user.email,
        roles=sorted(store.roles(user.id)),
        auth_method=user.auth_method,
        registration_status=registration_status(store, user),
        rejection_reason=rejection_reason(store, user),
        resubmissions_left=_resubmissions_left(store, user),
    )


def _resubmissions_left(store: AccountStore, user: User) -> int | None:
    """Only while rejected (DEVIATIONS.md #286)."""
    profile = store.get_profile(user.id)
    if profile is None or profile.status != "rejected":
        return None
    return resubmissions_left(profile, get_settings().reviewer_max_resubmissions)


@router.get("/consent", response_model=ConsentOut)
async def get_consent(principal: Principal = Depends(current_principal)) -> ConsentOut:  # noqa: ARG001
    c = _consent()
    return ConsentOut(
        version=c.version,
        sha256=c.sha256,
        title=c.title,
        sections=[ConsentSectionOut(heading=s.heading, body=s.body) for s in c.sections],
        agreement=c.agreement,
    )


@router.post("/consent", response_model=MeResponse)
async def post_consent(
    body: ConsentDecision,
    principal: Principal = Depends(current_principal),
    store: AccountStore = Depends(get_account_store),
) -> MeResponse:
    user = _current_user(store, principal)
    _call(
        record_consent,
        store,
        user,
        _consent(),
        agree=body.agree,
        version=body.version,
        sha256=body.sha256,
    )
    return await me(principal, store)


@router.get("/registration/options", response_model=RegistrationOptions)
async def registration_options() -> RegistrationOptions:
    def opts(d: dict[str, str]) -> list[RegistrationOption]:
        return [RegistrationOption(value=k, label=v) for k, v in d.items()]

    return RegistrationOptions(
        sex=opts(SEX_OPTIONS),
        cadre=opts(CADRE_OPTIONS),
        expertise_level=[
            RegistrationOption(
                value=k,
                label=v,
                cadres=sorted(EXPERTISE_CADRES[k]) if k in EXPERTISE_CADRES else None,
            )
            for k, v in EXPERTISE_OPTIONS.items()
        ],
        years_min=YEARS_MIN,
        years_max=YEARS_MAX,
        text_min=TEXT_MIN,
        text_max=TEXT_MAX,
    )


@router.get("/registration", response_model=OwnRegistration)
async def get_own_registration(
    principal: Principal = Depends(current_principal),
    store: AccountStore = Depends(get_account_store),
) -> OwnRegistration:
    """The caller's own last submitted answers, to prefill a resubmission
    after a rejection (DEVIATIONS.md #284). 404 before any submission."""
    answers = own_registration(store, _current_user(store, principal))
    if answers is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no registration submitted yet")
    return OwnRegistration.model_validate(answers)


@router.post("/registration", response_model=MeResponse)
async def post_registration(
    body: RegistrationIn,
    principal: Principal = Depends(current_principal),
    store: AccountStore = Depends(get_account_store),
) -> MeResponse:
    user = _current_user(store, principal)
    _call(
        submit_registration,
        store,
        user,
        body,
        max_resubmissions=get_settings().reviewer_max_resubmissions,
    )
    return await me(principal, store)
