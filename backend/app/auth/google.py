"""Google sign-in: verify a Google ID token (DEVIATIONS.md #282).

The frontend gets an ID token from Google Identity Services and posts it to
`POST /auth/google`. This module checks its RS256 signature against Google's
published keys, the audience (`GOOGLE_CLIENT_ID`), the issuer, expiry, and
that Google has verified the email address. Only the user's own identity
claims are exchanged with Google; no patient data is involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import jwt

GOOGLE_CERTS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = ("accounts.google.com", "https://accounts.google.com")


class GoogleTokenError(ValueError):
    """The ID token is missing, invalid, expired, or for another client."""


@dataclass(frozen=True)
class GoogleIdentity:
    sub: str
    email: str


_jwks_client: jwt.PyJWKClient | None = None


def _get_jwks_client() -> jwt.PyJWKClient:
    # One client per process: it caches Google's keys between sign-ins.
    global _jwks_client  # noqa: PLW0603
    if _jwks_client is None:
        _jwks_client = jwt.PyJWKClient(GOOGLE_CERTS_URL, cache_keys=True)
    return _jwks_client


# Seam for tests (no network).
_JWKS_CLIENT_FN = _get_jwks_client


def verify_google_id_token(id_token: str, *, client_id: str) -> GoogleIdentity:
    if not client_id:
        raise GoogleTokenError("Google sign-in is not configured (GOOGLE_CLIENT_ID)")
    try:
        key = _JWKS_CLIENT_FN().get_signing_key_from_jwt(id_token).key
        claims = jwt.decode(
            id_token,
            key,
            algorithms=["RS256"],
            audience=client_id,
            options={"require": ["sub", "email", "iss", "exp", "aud"]},
        )
    except (jwt.InvalidTokenError, jwt.PyJWKClientError) as exc:
        raise GoogleTokenError(f"invalid Google ID token: {exc}") from exc
    if claims["iss"] not in GOOGLE_ISSUERS:
        raise GoogleTokenError("ID token was not issued by Google")
    if claims.get("email_verified") is not True:
        raise GoogleTokenError("Google has not verified this email address")
    return GoogleIdentity(sub=str(claims["sub"]), email=str(claims["email"]).strip().lower())
