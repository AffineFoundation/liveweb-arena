"""
CoinGecko Plugin.

Plugin for cryptocurrency market data from CoinGecko.
"""

import re
from typing import Any, Dict, List
from urllib.parse import urlparse

from liveweb_arena.plugins.base import BasePlugin
from .api_client import fetch_single_coin_data, fetch_homepage_api_data
from liveweb_arena.utils.logger import log


class CoinGeckoPlugin(BasePlugin):
    """
    CoinGecko plugin for cryptocurrency data.

    Handles pages like:
    - https://www.coingecko.com/ (homepage - all coins)
    - https://www.coingecko.com/en/coins/bitcoin (detail page)
    - https://www.coingecko.com/en/coins/ethereum

    API data includes: current_price, market_cap, volume, 24h change, etc.
    """

    name = "coingecko"

    allowed_domains = [
        "coingecko.com",
        "www.coingecko.com",
    ]

    def get_blocked_patterns(self) -> List[str]:
        """Block direct API access to force agents to use the website."""
        return [
            "*api.coingecko.com*",
            "*geckoterminal*",
            "*/tagmetrics/*",
            "*/accounts/*",
            "*/onboarding/*",
            "*/sentiment_votes/*",
            "*/portfolios/*",
            "*/portfolio_summary*",
            "*/price_charts/*",
            "*-emoji-*",
        ]

    async def fetch_api_data(self, url: str) -> Dict[str, Any]:
        """
        Fetch API data for a CoinGecko page.

        - Homepage: Returns all coins in {"coins": {...}} format
        - Detail page: Returns single coin data with "id" field

        Args:
            url: Page URL

        Returns:
            API data appropriate for the page type
        """
        # Check for detail page first
        coin_id = self._extract_coin_id(url)
        if coin_id:
            data = await fetch_single_coin_data(coin_id)
            return data if data else {}

        # Homepage - return all coins
        if self._is_homepage(url):
            return await fetch_homepage_api_data()

        return {}

    def _is_homepage(self, url: str) -> bool:
        """Check if URL is the CoinGecko homepage."""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        # Homepage patterns: "", "en", "en/"
        return path in ('', 'en')

    def _extract_coin_id(self, url: str) -> str:
        """
        Extract coin ID from CoinGecko URL.

        Examples:
            https://www.coingecko.com/en/coins/bitcoin -> bitcoin
            https://www.coingecko.com/en/coins/ethereum -> ethereum
        """
        parsed = urlparse(url)
        path = parsed.path

        # Pattern: /en/coins/{coin_id} or /coins/{coin_id}
        match = re.search(r'/coins/([^/?#]+)', path)
        if match:
            return match.group(1).lower()

        return ""
