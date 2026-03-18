"""
CoinGecko Plugin.

Plugin for cryptocurrency market data from CoinGecko.
"""

import re
from typing import Any, Dict, List
from urllib.parse import urlparse

from liveweb_arena.plugins.base import BasePlugin
from .api_client import fetch_single_coin_data, fetch_homepage_api_data


# URL slug to API coin ID mapping for coins where they differ
# CoinGecko sometimes uses different identifiers in URLs vs API
URL_SLUG_TO_COIN_ID = {
    "polygon": "polygon-ecosystem-token",  # Polygon rebranded MATIC to POL
    "matic-network": "polygon-ecosystem-token",
    "avalanche": "avalanche-2",
    "hedera": "hedera-hashgraph",
    "lido-staked-ether": "staked-ether",
    "render": "render-token",
    "fetch": "fetch-ai",
    "graph": "the-graph",
    "injective": "injective-protocol",
}


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

    def get_stable_url_patterns(self) -> List[str]:
        return [
            "/en",
            "/en/coins/",
            "/en/coins/*/historical_data",
            "/en/highlights/",
        ]

    def classify_url(self, url: str) -> str | None:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path = parsed.path.lower()
        if "coingecko.com" not in host:
            return None
        coin_id = self._extract_coin_id(url)
        if coin_id and not self.is_plausible_asset_id(url):
            return "model_invalid_asset_id"
        if "/coins/" in path and coin_id:
            return None
        if self._is_homepage(url) or "/en/highlights/" in path:
            return None
        return None

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
            if not data:
                raise ValueError(f"CoinGecko API returned no data for coin_id={coin_id}")
            return data

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

    def needs_api_data(self, url: str) -> bool:
        """
        Determine if this URL needs API data for ground truth.

        Only homepage and coin detail pages can provide API data.
        Other pages (charts, portfolios, etc.) are navigation-only.

        Args:
            url: Page URL

        Returns:
            True if API data is needed and available, False otherwise
        """
        # Coin detail page needs API data
        if self._extract_coin_id(url):
            return True
        # Homepage needs API data
        if self._is_homepage(url):
            return True
        # Other pages (charts, global-charts, categories, etc.) don't need API data
        return False

    def _extract_coin_id(self, url: str) -> str:
        """
        Extract coin ID from CoinGecko URL.

        Handles URL slug to API coin ID translation for coins where they differ.

        Examples:
            https://www.coingecko.com/en/coins/bitcoin -> bitcoin
            https://www.coingecko.com/en/coins/polygon -> polygon-ecosystem-token
        """
        parsed = urlparse(url)
        path = parsed.path

        # Pattern: /en/coins/{coin_id} or /coins/{coin_id}
        match = re.search(r'/coins/([^/?#]+)', path)
        if match:
            url_slug = match.group(1).lower()
            # Translate URL slug to API coin ID if mapping exists
            return URL_SLUG_TO_COIN_ID.get(url_slug, url_slug)

        return ""

    def is_plausible_asset_id(self, url: str) -> bool:
        coin_id = self._extract_coin_id(url)
        if not coin_id:
            return True
        # Known false positives from hybrid/company-name hallucinations.
        invalid = {
            "microsoft",
            "google",
            "exxon-mobil",
            "jpmorgan-chase",
            "tesla",
            "walmart",
            "apple",
            "amazon",
        }
        return coin_id not in invalid
