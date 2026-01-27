"""Shared utilities for hybrid plugin templates."""

import asyncio
import logging
import time
from typing import Any, Callable, Optional, TypeVar

from liveweb_arena.plugins.coingecko.api_client import CoinGeckoClient
from liveweb_arena.plugins.stooq.api_client import StooqClient, StooqRateLimitError
from liveweb_arena.utils.logger import progress, progress_done, is_verbose

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Page extraction state (for HYBRID GT that prefers page data)
_extraction_state: Optional[Any] = None


def set_extraction_state(state: Optional[Any]):
    """Set the extraction state for GT lookups."""
    global _extraction_state
    _extraction_state = state


def get_extraction_state() -> Optional[Any]:
    """Get the current extraction state."""
    return _extraction_state


async def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 10,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    operation_name: str = "operation",
) -> T:
    """
    Retry an async operation with exponential backoff.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        operation_name: Name for logging purposes

    Returns:
        Result of the function

    Raises:
        RuntimeError: If all retries fail
        StooqRateLimitError: If Stooq rate limit is hit (no retry)
    """
    last_exception = None
    start_time = time.time()

    for attempt in range(max_retries):
        try:
            if is_verbose():
                elapsed = time.time() - start_time
                progress("GT", elapsed, 120, f"[{attempt+1}/{max_retries}] {operation_name}")
            result = await func()
            if result is not None:
                if is_verbose():
                    progress_done("GT", f"{operation_name} done in {time.time()-start_time:.1f}s")
                return result
            raise ValueError(f"{operation_name} returned None")
        except StooqRateLimitError:
            logger.error(f"{operation_name}: Stooq rate limit exceeded - stopping retries")
            raise
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                # Show progress during wait
                wait_start = time.time()
                while time.time() - wait_start < delay:
                    if is_verbose():
                        elapsed = time.time() - start_time
                        progress("GT", elapsed, 120, f"[{attempt+1}/{max_retries}] retry wait {operation_name}")
                    await asyncio.sleep(min(1.0, delay - (time.time() - wait_start)))
            else:
                logger.error(f"{operation_name} failed after {max_retries} attempts: {e}")

    raise RuntimeError(f"{operation_name} failed after {max_retries} retries: {last_exception}")


async def get_crypto_24h_change(coin_id: str) -> float:
    """
    Get 24h percentage change from CoinGecko.

    Priority order:
    1. Page extraction state (if agent visited the page)
    2. Collected API data from GTCollector (page-bound data)
    3. Live API call (fallback in live mode)

    This ensures GT matches what the agent sees on visited pages.

    Args:
        coin_id: CoinGecko coin identifier

    Returns:
        24h percentage change

    Raises:
        RuntimeError: If data not found (agent must visit the page)
    """
    # First, try page extraction (most accurate - matches what agent sees)
    ext_state = get_extraction_state()
    if ext_state is not None:
        merged = ext_state.get_merged_data()
        if coin_id in merged:
            asset_data = merged[coin_id]
            change = asset_data.get("change_24h")
            if change is not None:
                logger.debug(f"Page extraction hit: CoinGecko {coin_id} change={change}")
                return change

    # Second, try collected API data from GTCollector (page-bound)
    from liveweb_arena.core.gt_collector import get_current_gt_collector
    gt_collector = get_current_gt_collector()
    if gt_collector is not None:
        api_data = gt_collector.get_collected_api_data()
        if coin_id in api_data:
            coin_data = api_data[coin_id]
            change = coin_data.get("price_change_percentage_24h")
            if change is not None:
                logger.debug(f"Collected API hit: CoinGecko {coin_id} change={change}")
                return change

        # If we have collected data but not for this asset, it's an error in cache mode
        # But if collected data is empty, we're likely in live mode - fall through to API
        if api_data:
            collected = list(api_data.keys())
            raise RuntimeError(
                f"CoinGecko data for '{coin_id}' not found. "
                f"Agent must visit the page. Collected: {collected[:10]}..."
            )

    # Live mode fallback - fetch from API directly
    async def fetch():
        data = await CoinGeckoClient.get_coin_market_data(coin_id)
        if data and len(data) > 0:
            change = data[0].get("price_change_percentage_24h")
            if change is not None:
                return change
        return None

    return await retry_with_backoff(
        fetch,
        max_retries=10,
        base_delay=1.0,
        operation_name=f"CoinGecko fetch {coin_id}",
    )


async def get_stooq_price(symbol: str) -> float:
    """
    Get current price from Stooq.

    Priority order:
    1. Collected API data from GTCollector (page-bound data)
    2. Live API call (fallback in live mode)

    Args:
        symbol: Stooq symbol

    Returns:
        Current price

    Raises:
        RuntimeError: If data not found (agent must visit the page)
    """
    # Try collected API data from GTCollector (page-bound)
    from liveweb_arena.core.gt_collector import get_current_gt_collector
    gt_collector = get_current_gt_collector()
    if gt_collector is not None:
        api_data = gt_collector.get_collected_api_data()
        if symbol in api_data:
            asset_data = api_data[symbol]
            price = asset_data.get("close")
            if price is not None:
                logger.debug(f"Collected API hit: Stooq {symbol} price={price}")
                return price

        # If we have collected data but not for this asset, it's an error in cache mode
        # But if collected data is empty, we're likely in live mode - fall through to API
        if api_data:
            collected = list(api_data.keys())
            raise RuntimeError(
                f"Stooq data for '{symbol}' not found. "
                f"Agent must visit the page. Collected: {collected[:10]}..."
            )

    # Live mode fallback - fetch from API directly
    async def fetch():
        data = await StooqClient.get_price_data(symbol)
        if data:
            price = data.get("close")
            if price is not None:
                return price
        return None

    return await retry_with_backoff(
        fetch,
        max_retries=10,
        base_delay=1.0,
        operation_name=f"Stooq price {symbol}",
    )


async def get_stooq_24h_change(symbol: str) -> float:
    """
    Get daily percentage change from Stooq.

    Priority order:
    1. Page extraction state (if agent visited the page)
    2. Collected API data from GTCollector (page-bound data)
    3. Live API call (fallback in live mode)

    This ensures GT matches what the agent sees on visited pages.

    Args:
        symbol: Stooq symbol

    Returns:
        Daily percentage change

    Raises:
        RuntimeError: If data not found (agent must visit the page)
    """
    # First, try page extraction (most accurate - matches what agent sees)
    ext_state = get_extraction_state()
    if ext_state is not None:
        merged = ext_state.get_merged_data()
        # Try symbol directly, also try lowercase
        symbol_lower = symbol.lower()
        for sym in [symbol, symbol_lower]:
            if sym in merged:
                asset_data = merged[sym]
                change = asset_data.get("daily_change_pct")
                if change is not None:
                    logger.debug(f"Page extraction hit: Stooq {symbol} change={change}")
                    return change

    # Second, try collected API data from GTCollector (page-bound)
    from liveweb_arena.core.gt_collector import get_current_gt_collector
    gt_collector = get_current_gt_collector()
    if gt_collector is not None:
        api_data = gt_collector.get_collected_api_data()
        symbol_lower = symbol.lower()
        for sym in [symbol, symbol_lower]:
            if sym in api_data:
                asset_data = api_data[sym]
                change = asset_data.get("daily_change_pct")
                if change is not None:
                    logger.debug(f"Collected API hit: Stooq {symbol} change={change}")
                    return change

        # If we have collected data but not for this asset, it's an error in cache mode
        # But if collected data is empty, we're likely in live mode - fall through to API
        if api_data:
            collected = list(api_data.keys())
            raise RuntimeError(
                f"Stooq data for '{symbol}' not found. "
                f"Agent must visit the page. Collected: {collected[:10]}..."
            )

    # Live mode fallback - fetch from API directly
    async def fetch():
        data = await StooqClient.get_price_data(symbol)
        if data:
            change = data.get("daily_change_pct")
            if change is not None:
                return change
        return None

    return await retry_with_backoff(
        fetch,
        max_retries=10,
        base_delay=1.0,
        operation_name=f"Stooq change {symbol}",
    )
