"""Stooq API client with caching support"""

import asyncio
import contextvars
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from liveweb_arena.plugins.base_client import RateLimiter, create_http_session

logger = logging.getLogger(__name__)

CACHE_SOURCE = "stooq"

# Global rate limiter: ALL Stooq CSV requests must go through this.
# Shared across fetch_cache_api_data (homepage bulk) and fetch_single_asset_data (detail).
# 0.5s interval: homepage bulk (28 symbols) completes in ~14s, under 25s prefetch timeout.
_global_csv_limiter = RateLimiter(min_interval=0.5)

# Rate limit tracking - once hit, don't retry until reset.
# Per-context: each evaluation gets its own rate limit state via contextvars,
# so concurrent evaluations don't interfere with each other.
_rate_limited: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_stooq_rate_limited", default=False
)

# Negative cache: symbols that returned no data in this evaluation.
# Prevents repeated API calls for symbols that are temporarily unavailable.
_negative_cache: contextvars.ContextVar[Optional[set]] = contextvars.ContextVar(
    "_stooq_negative_cache", default=None
)


def _get_negative_cache() -> set:
    cache = _negative_cache.get()
    if cache is None:
        cache = set()
        _negative_cache.set(cache)
    return cache


class StooqRateLimitError(Exception):
    """Raised when Stooq API rate limit is exceeded."""
    pass


def _parse_stooq_csv(csv_text: str, symbol: str = "") -> Optional[Dict[str, Any]]:
    """
    Parse Stooq CSV response into price data dict.

    Args:
        csv_text: Raw CSV text from Stooq API
        symbol: Optional symbol to include in result

    Returns:
        Dict with price data or None if parsing fails.
        Includes 'history' field with recent daily data for historical queries.
    """
    # Normalize line endings
    csv_text = csv_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = csv_text.strip().split("\n")

    if len(lines) < 2:
        return None

    headers = lines[0].lower().split(",")
    today_values = lines[-1].split(",")
    today_data = dict(zip(headers, today_values))

    def parse_float(val):
        try:
            return float(val) if val else None
        except (ValueError, TypeError):
            return None

    close = parse_float(today_data.get("close"))
    if close is None:
        return None

    # Calculate daily change from previous day
    daily_change = None
    daily_change_pct = None
    if len(lines) >= 3:
        prev_values = lines[-2].split(",")
        prev_data = dict(zip(headers, prev_values))
        prev_close = parse_float(prev_data.get("close"))
        if prev_close and prev_close > 0:
            daily_change = close - prev_close
            daily_change_pct = (daily_change / prev_close) * 100

    result = {
        "date": today_data.get("date", ""),
        "open": parse_float(today_data.get("open")),
        "high": parse_float(today_data.get("high")),
        "low": parse_float(today_data.get("low")),
        "close": close,
        "volume": parse_float(today_data.get("volume")),
        "daily_change": daily_change,
        "daily_change_pct": daily_change_pct,
    }
    if symbol:
        result["symbol"] = symbol

    # Parse historical data (last 30 days for historical queries)
    history = []
    data_lines = lines[1:]  # Skip header
    for line in data_lines[-30:]:  # Last 30 days
        values = line.split(",")
        if len(values) >= len(headers):
            row_data = dict(zip(headers, values))
            row_close = parse_float(row_data.get("close"))
            if row_close is not None:
                history.append({
                    "date": row_data.get("date", ""),
                    "open": parse_float(row_data.get("open")),
                    "high": parse_float(row_data.get("high")),
                    "low": parse_float(row_data.get("low")),
                    "close": row_close,
                    "volume": parse_float(row_data.get("volume")),
                })
    result["history"] = history

    return result


class StooqClient:
    """Stooq CSV API client with rate limiting."""

    CSV_URL = "https://stooq.com/q/d/l/"

    @classmethod
    async def get_price_data(
        cls,
        symbol: str,
        timeout: float = 15.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Get price data for a symbol.

        Args:
            symbol: Stooq symbol (e.g., "gc.c", "^spx", "aapl.us")
            timeout: Request timeout in seconds

        Returns:
            Dict with price data or None on error:
            {
                "symbol": str,
                "date": str,
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float or None,
                "daily_change": float or None,
                "daily_change_pct": float or None,
            }

        Raises:
            StooqRateLimitError: If API rate limit is exceeded
        """
        # If already rate limited, raise immediately
        if _rate_limited.get():
            raise StooqRateLimitError(
                "Stooq API daily limit exceeded. Cache is empty. "
                "Wait for daily reset or manually populate cache."
            )

        # Global rate limiter shared with all Stooq CSV requests
        await _global_csv_limiter.wait()

        try:
            async with create_http_session(timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}) as session:
                params = {"s": symbol, "i": "d"}
                async with session.get(
                    cls.CSV_URL,
                    params=params,
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Stooq error for {symbol}: {response.status}")
                        return None
                    csv_text = await response.text()

            # Check for rate limit error
            if "Exceeded the daily hits limit" in csv_text:
                _rate_limited.set(True)
                logger.error("Stooq API daily limit exceeded!")
                raise StooqRateLimitError(
                    "Stooq API daily limit exceeded. Wait for reset or use cached data."
                )

            return _parse_stooq_csv(csv_text, symbol)

        except asyncio.TimeoutError:
            logger.warning(f"Stooq timeout for {symbol}")
            return None
        except StooqRateLimitError:
            raise
        except Exception as e:
            logger.warning(f"Stooq error for {symbol}: {e}")
            return None


# ============================================================
# Cache Data Fetcher (used by snapshot_integration)
# ============================================================

def _get_all_symbols() -> List[str]:
    """Homepage-visible symbols only (no US stocks — not shown on homepage)."""
    from .templates.variables import INDICES, CURRENCIES, COMMODITIES

    symbols = []
    symbols.extend(s.symbol for s in INDICES)
    symbols.extend(s.symbol for s in CURRENCIES)
    symbols.extend(s.symbol for s in COMMODITIES)
    return symbols


async def fetch_cache_api_data() -> Optional[Dict[str, Any]]:
    """
    Fetch Stooq price data for all assets defined in variables.

    Returns data structure:
    {
        "_meta": {"source": "stooq", "asset_count": N},
        "assets": {
            "aapl.us": {"date": ..., "open": ..., "close": ..., "daily_change_pct": ...},
            ...
        }
    }
    """
    assets = _get_all_symbols()
    logger.info(f"Fetching Stooq data for {len(assets)} assets...")

    result = {
        "_meta": {
            "source": CACHE_SOURCE,
            "asset_count": 0,
        },
        "assets": {},
    }
    failed = 0

    # Sequential fetch with global rate limiter — avoid IP bans
    async with create_http_session(timeout=15, headers={"User-Agent": "Mozilla/5.0"}) as session:
        for symbol in assets:
            await _global_csv_limiter.wait()
            try:
                url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
                async with session.get(
                    url,
                ) as response:
                    if response.status != 200:
                        failed += 1
                        continue

                    text = await response.text()
                    if "Exceeded the daily hits limit" in text:
                        _rate_limited.set(True)
                        logger.error("Stooq API daily limit exceeded during bulk fetch")
                        break

                    parsed = _parse_stooq_csv(text, symbol)
                    if parsed:
                        result["assets"][symbol] = parsed

            except Exception:
                failed += 1

    result["_meta"]["asset_count"] = len(result["assets"])
    logger.info(f"Fetched {len(result['assets'])} assets from Stooq ({failed} failed)")
    return result


def _get_file_cache_path() -> Path:
    """Get path for stooq homepage file cache."""
    cache_dir = os.environ.get("LIVEWEB_CACHE_DIR", "/var/lib/liveweb-arena/cache")
    return Path(cache_dir) / "_plugin_init" / "stooq_homepage.json"


def _get_cache_ttl() -> int:
    """Get cache TTL from environment."""
    from liveweb_arena.core.cache import DEFAULT_TTL
    return int(os.environ.get("LIVEWEB_CACHE_TTL", str(DEFAULT_TTL)))


def _is_file_cache_valid() -> bool:
    """Check if homepage file cache exists and is within TTL."""
    cache_file = _get_file_cache_path()
    if not cache_file.exists():
        return False
    try:
        cached = json.loads(cache_file.read_text())
        if time.time() - cached.get("_fetched_at", 0) < _get_cache_ttl():
            return bool(cached.get("assets"))
    except Exception:
        pass
    return False


def initialize_cache():
    """
    Pre-warm homepage file cache synchronously.

    Called by plugin.initialize() before evaluation starts (no timeout pressure).
    Uses file lock to prevent multiple instances from fetching simultaneously.
    If file cache is valid, this is a no-op.
    """
    import fcntl

    # Quick check without lock — avoids lock contention when cache is warm
    if _is_file_cache_valid():
        logger.info("Stooq init: homepage cache valid (quick check)")
        return

    # Acquire file lock — only one process fetches, others wait
    lock_path = _get_file_cache_path().with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)  # Blocking wait

        # Re-check after acquiring lock — another process may have filled cache
        if _is_file_cache_valid():
            logger.info("Stooq init: homepage cache filled by another process")
            return

        # Fetch and cache
        logger.info("Stooq init: pre-warming homepage cache...")
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(lambda: asyncio.run(fetch_homepage_api_data())).result()
        else:
            asyncio.run(fetch_homepage_api_data())
    finally:
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        fd.close()


async def fetch_homepage_api_data() -> Dict[str, Any]:
    """
    Fetch API data for Stooq homepage (all assets).

    Uses file cache to avoid repeated CSV requests within TTL.

    Returns homepage format:
    {
        "assets": {
            "gc.c": {<price_data>},
            ...
        }
    }
    """
    # 1. Check file cache
    cache_file = _get_file_cache_path()
    ttl = _get_cache_ttl()
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if time.time() - cached.get("_fetched_at", 0) < ttl:
                assets = cached.get("assets", {})
                if assets:
                    logger.info(f"Stooq homepage: {len(assets)} assets from file cache")
                    return {"assets": assets}
        except Exception:
            pass

    # 2. Fetch from API
    data = await fetch_cache_api_data()
    assets = data.get("assets", {}) if data else {}

    # 3. Write file cache
    if assets:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"assets": assets, "_fetched_at": time.time()}))
        except Exception:
            pass

    return {"assets": assets}


async def fetch_single_asset_data(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch price data for a single asset.

    Tries the symbol as-is first, then with common suffixes (.us)
    since Stooq's CSV API requires suffixed symbols for some markets.
    Uses negative cache to avoid repeated requests for symbols with no data.
    """
    if _rate_limited.get():
        raise StooqRateLimitError("Stooq API rate limited (persistent for this session)")

    neg = _get_negative_cache()
    if symbol in neg:
        return {}

    # Try .us suffix first for bare symbols (canonical form for US stocks)
    variants = [symbol]
    if "." not in symbol and not symbol.startswith("^"):
        variants = [f"{symbol}.us", symbol]

    for sym in variants:
        await _global_csv_limiter.wait()
        try:
            async with create_http_session(timeout=15, headers={"User-Agent": "Mozilla/5.0"}) as session:
                url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
                async with session.get(
                    url,
                ) as response:
                    if response.status != 200:
                        continue

                    text = await response.text()
                    if "Exceeded the daily hits limit" in text:
                        _rate_limited.set(True)
                        raise StooqRateLimitError("Stooq API daily limit exceeded")

                    if "No data" in text:
                        continue

                    result = _parse_stooq_csv(text, sym)
                    if result:
                        return result

        except StooqRateLimitError:
            raise
        except Exception:
            continue

    # All variants failed — add to negative cache
    neg.add(symbol)
    return {}
