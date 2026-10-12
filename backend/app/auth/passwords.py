"""Password hashing for email + password sign-in (DEVIATIONS.md #282).

scrypt from the standard library (no new dependency), a random 16-byte salt
per password, stored as `scrypt$n$r$p$<salt b64>$<hash b64>`. Verification
is constant-time.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 128
_N, _R, _P, _DKLEN = 2**14, 8, 1, 32


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=_DKLEN)
    return "$".join(
        (
            "scrypt",
            str(_N),
            str(_R),
            str(_P),
            base64.b64encode(salt).decode(),
            base64.b64encode(digest).decode(),
        )
    )


def verify_password(password: str, stored: str | None) -> bool:
    """False for a missing or malformed hash, never an exception."""
    if not stored:
        return False
    try:
        scheme, n, r, p, salt_b64, digest_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        digest = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, expected)


def password_problem(password: str) -> str | None:
    """Why `password` is not acceptable, or None."""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"password must be at least {MIN_PASSWORD_LENGTH} characters"
    if len(password) > MAX_PASSWORD_LENGTH:
        return f"password must be at most {MAX_PASSWORD_LENGTH} characters"
    return None
