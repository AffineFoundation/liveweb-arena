"""Base API client with common rate limiting infrastructure."""

import asyncio
import time
from abc import ABC
from typing import Any, ClassVar, Dict, Optional

import aiohttp


class APIFetchError(Exception):
    """
    Raised when API data fetch fails.

    This is a GT-critical error - evaluation cannot proceed without valid data.
    All API clients should raise this instead of returning None/empty dict.
    """

    def __init__(
        self,
        message: str,
        source: str = None,
        status_code: int = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.source = source
        self.status_code = status_code
        self.metadata = dict(metadata or {})


def validate_api_response(data: Any, expected_type: type, context: str) -> None:
    """
    Validate API response type. Raises APIFetchError if invalid.

    Args:
        data: The response data to validate
        expected_type: Expected Python type (dict, list, etc.)
        context: Description for error message (e.g., "coin_id=bitcoin")
    """
    if not isinstance(data, expected_type):
        raise APIFetchError(
            f"API returned {type(data).__name__}, expected {expected_type.__name__} for {context}",
            source="validation",
        )


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


def create_http_session(
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> aiohttp.ClientSession:
    """
    Create a shared-config aiohttp session for plugin API access.

    Important: trust_env=True allows the evaluation environment's proxy settings
    to apply consistently, which is required on this machine for some sites/APIs.
    """
    timeout_cfg = aiohttp.ClientTimeout(total=timeout) if timeout is not None else None
    connector = aiohttp.TCPConnector(ttl_dns_cache=300, limit=32, ssl=False)
    return aiohttp.ClientSession(
        headers=headers,
        timeout=timeout_cfg,
        trust_env=True,
        connector=connector,
    )
