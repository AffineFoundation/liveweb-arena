"""
Stooq Plugin.

Plugin for financial market data from stooq.com.
"""

import re
from typing import Any, Dict, List
from urllib.parse import urlparse, parse_qs

from liveweb_arena.plugins.base import BasePlugin
from .api_client import fetch_single_asset_data, fetch_homepage_api_data
from liveweb_arena.utils.logger import log


class StooqPlugin(BasePlugin):
    """
    Stooq plugin for financial market data.

    Handles pages like:
    - https://stooq.com/ (homepage - all assets)
    - https://stooq.com/q/?s=aapl.us (stocks)
    - https://stooq.com/q/?s=^spx (indices)
    - https://stooq.com/q/?s=gc.f (commodities)
    - https://stooq.com/q/?s=eurusd (forex)

    API data includes: open, high, low, close, volume, daily_change_pct, etc.
    """

    name = "stooq"

    allowed_domains = [
        "stooq.com",
        "www.stooq.com",
    ]

    def get_blocked_patterns(self) -> List[str]:
        """Block direct CSV download to force agents to use the website."""
        return [
            "*/q/d/l/*",  # CSV download endpoint
        ]

    async def fetch_api_data(self, url: str) -> Dict[str, Any]:
        """
        Fetch API data for a Stooq page.

        - Homepage: Returns all assets in {"assets": {...}} format
        - Detail page: Returns single asset data with "symbol" field

        Args:
            url: Page URL

        Returns:
            API data appropriate for the page type
        """
        # Check for detail page first
        symbol = self._extract_symbol(url)
        if symbol:
            data = await fetch_single_asset_data(symbol)
            return data if data else {}

        # Homepage - return all assets
        if self._is_homepage(url):
            return await fetch_homepage_api_data()

        return {}

    def _is_homepage(self, url: str) -> bool:
        """Check if URL is the Stooq homepage."""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        # Homepage has no path or just "/"
        return path == '' and not parsed.query

    def _extract_symbol(self, url: str) -> str:
        """
        Extract symbol from Stooq URL.

        Examples:
            https://stooq.com/q/?s=aapl.us -> aapl.us
            https://stooq.com/q/?s=^spx -> ^spx
            https://stooq.com/q/d/?s=gc.f -> gc.f
        """
        parsed = urlparse(url)
        query = parse_qs(parsed.query)

        if "s" in query:
            return query["s"][0].lower()

        return ""
