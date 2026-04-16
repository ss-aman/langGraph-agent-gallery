"""
src/security/rate_limiter.py — Async token-bucket rate limiter.

One bucket per user_id. Tokens refill continuously at `rpm / 60`
tokens per second up to `burst` capacity.

Thread-safety: uses asyncio.Lock (single event loop).
For horizontal scaling: replace the in-process dict with Redis INCR +
EXPIRE, or use a sliding-window algorithm with Redis sorted sets.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from src.config import settings


@dataclass
class _Bucket:
    """State for one user's token bucket."""
    tokens: float
    last_refill: float = field(default_factory=time.monotonic)


class AsyncRateLimiter:
    """
    In-process async token-bucket rate limiter.

    Usage:
        limiter = AsyncRateLimiter()
        allowed, remaining = await limiter.check("user-id")
    """

    def __init__(
        self,
        rpm: int | None = None,
        burst: int | None = None,
    ) -> None:
        self._rpm = rpm or settings.rate_limit_rpm
        self._burst = burst or settings.rate_limit_burst
        self._refill_rate = self._rpm / 60.0  # tokens per second
        self._buckets: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> tuple[bool, int]:
        """
        Check whether `key` (user_id or IP) is within rate limits.

        Returns:
            (allowed: bool, remaining_tokens: int)
        """
        async with self._lock:
            now = time.monotonic()

            if key not in self._buckets:
                self._buckets[key] = _Bucket(tokens=float(self._burst))

            bucket = self._buckets[key]

            # Refill tokens based on elapsed time
            elapsed = now - bucket.last_refill
            bucket.tokens = min(
                self._burst,
                bucket.tokens + elapsed * self._refill_rate,
            )
            bucket.last_refill = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, int(bucket.tokens)
            else:
                return False, 0

    async def reset(self, key: str) -> None:
        """Reset a bucket (useful in tests)."""
        async with self._lock:
            self._buckets.pop(key, None)


# Module-level singleton — shared across all graph invocations in this process
_rate_limiter: AsyncRateLimiter | None = None


def get_rate_limiter() -> AsyncRateLimiter:
    """Return the process-wide rate limiter singleton."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = AsyncRateLimiter()
    return _rate_limiter
