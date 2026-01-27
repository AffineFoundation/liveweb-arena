"""Base API client with common rate limiting infrastructure."""

import asyncio
import time
from abc import ABC
from typing import ClassVar


class RateLimiter:
    """Simple rate limiter for API clients."""

    def __init__(self, min_interval: float = 0.5):
        self.min_interval = min_interval
        self._last_request: float = 0
        self._lock = asyncio.Lock()

    async def wait(self):
        """Wait if needed to respect rate limit."""
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_request
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_request = time.time()


class BaseAPIClient(ABC):
    """
    Base class for API clients with rate limiting.

    Subclasses should:
    1. Set _rate_limiter class variable with appropriate interval
    2. Call await self._rate_limiter.wait() before making requests
    """

    _rate_limiter: ClassVar[RateLimiter]

    @classmethod
    async def _rate_limit(cls):
        """Apply rate limiting. Subclasses can override for custom behavior."""
        await cls._rate_limiter.wait()
