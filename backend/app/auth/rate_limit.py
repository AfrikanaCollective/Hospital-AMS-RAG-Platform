"""Failed sign-in limits for password sign-ins (DEVIATIONS.md #283).

Counts failed email + password and admin sign-ins in fixed windows:

- per account: `SIGNIN_MAX_FAILURES_PER_ACCOUNT` (5) failures within
  `SIGNIN_WINDOW_SECONDS` (900) lock that account until the window ends;
- per client IP: `SIGNIN_MAX_FAILURES_PER_IP` (20) in the same window, so one
  address cannot try many accounts.

While locked, sign-in is refused without checking the password. A successful
sign-in clears that account's count (not the IP's).

Counters live in Redis so they survive API restarts and are shared by every
API process. Account keys are sha256 hashes, so no email address is stored
there. If Redis cannot be reached the limiter falls back to an in-process
counter and logs a warning, so a Redis outage weakens the limit instead of
blocking every sign-in.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from app.logging import get_logger

logger = get_logger(__name__)

_PREFIX = "signin_fail"


@dataclass(frozen=True)
class SignInLimits:
    max_per_account: int
    max_per_ip: int
    window_seconds: int


def account_key(kind: str, identifier: str) -> str:
    digest = hashlib.sha256(identifier.strip().lower().encode()).hexdigest()
    return f"{_PREFIX}:{kind}:{digest}"


def ip_key(ip: str) -> str:
    return f"{_PREFIX}:ip:{ip}"


class FailureCounter(Protocol):
    def get(self, key: str) -> tuple[int, int]:
        """(failures in the current window, seconds until it ends)."""
        ...

    def incr(self, key: str, window_seconds: int) -> None: ...
    def clear(self, key: str) -> None: ...


class MemoryCounter:
    """In-process counter: tests, and the fallback when Redis is down."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._data: dict[str, tuple[int, float]] = {}  # key -> (count, window end)

    def get(self, key: str) -> tuple[int, int]:
        with self._lock:
            count, end = self._data.get(key, (0, 0.0))
            remaining = end - self._clock()
            if remaining <= 0:
                self._data.pop(key, None)
                return 0, 0
            return count, int(remaining) + 1

    def incr(self, key: str, window_seconds: int) -> None:
        with self._lock:
            now = self._clock()
            count, end = self._data.get(key, (0, 0.0))
            if end <= now:
                count, end = 0, now + window_seconds
            self._data[key] = (count + 1, end)

    def clear(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)


class RedisCounter:
    """Fixed window: the first failure sets the key's expiry."""

    def __init__(self, url: str, fallback: MemoryCounter) -> None:
        import redis  # noqa: PLC0415

        self._redis = redis.Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2)
        self._fallback = fallback
        self._errors = (redis.RedisError, OSError)

    def _degraded(self, exc: Exception) -> None:
        logger.warning("signin_rate_limit_redis_unavailable_using_memory", error=str(exc))

    def get(self, key: str) -> tuple[int, int]:
        try:
            pipe = self._redis.pipeline()
            pipe.get(key)
            pipe.ttl(key)
            raw, ttl = pipe.execute()
        except self._errors as exc:
            self._degraded(exc)
            return self._fallback.get(key)
        return (int(raw), max(int(ttl), 1)) if raw is not None else (0, 0)

    def incr(self, key: str, window_seconds: int) -> None:
        try:
            pipe = self._redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, window_seconds, nx=True)
            pipe.execute()
        except self._errors as exc:
            self._degraded(exc)
            self._fallback.incr(key, window_seconds)

    def clear(self, key: str) -> None:
        try:
            self._redis.delete(key)
        except self._errors as exc:
            self._degraded(exc)
        self._fallback.clear(key)


class SignInLimiter:
    def __init__(self, counter: FailureCounter, limits: SignInLimits) -> None:
        self.counter = counter
        self.limits = limits

    def retry_after(self, account: str, ip: str | None) -> int | None:
        """Seconds until sign-in is allowed again, or None if it is allowed."""
        waits = []
        count, ttl = self.counter.get(account)
        if count >= self.limits.max_per_account:
            waits.append(ttl)
        if ip:
            count, ttl = self.counter.get(ip_key(ip))
            if count >= self.limits.max_per_ip:
                waits.append(ttl)
        return max(waits) if waits else None

    def failed(self, account: str, ip: str | None) -> None:
        self.counter.incr(account, self.limits.window_seconds)
        if ip:
            self.counter.incr(ip_key(ip), self.limits.window_seconds)

    def succeeded(self, account: str) -> None:
        self.counter.clear(account)


_limiter: SignInLimiter | None = None


def get_signin_limiter() -> SignInLimiter:
    """One limiter per process (FastAPI dependency; overridden in tests)."""
    global _limiter  # noqa: PLW0603
    if _limiter is None:
        from app.config import get_settings  # noqa: PLC0415

        s = get_settings()
        _limiter = SignInLimiter(
            RedisCounter(s.redis_url, fallback=MemoryCounter()),
            SignInLimits(
                max_per_account=s.signin_max_failures_per_account,
                max_per_ip=s.signin_max_failures_per_ip,
                window_seconds=s.signin_window_seconds,
            ),
        )
    return _limiter
