"""
Hybrid Plugin.

Plugin for cross-site queries requiring multiple data sources.
"""

from typing import Any, Dict, List

from liveweb_arena.plugins.base import BasePlugin


class HybridPlugin(BasePlugin):
    """
    Hybrid plugin for cross-site queries.

    This plugin doesn't have its own data source - it combines data
    from other plugins (coingecko, stooq) for cross-site tasks.

    API data is bound to pages and collected by GTCollector during
    agent navigation.
    """

    name = "hybrid"

    allowed_domains = [
        "coingecko.com",
        "www.coingecko.com",
        "stooq.com",
        "www.stooq.com",
    ]

    def get_blocked_patterns(self) -> List[str]:
        """Block API access (same as individual plugins)."""
        return [
            "*api.coingecko.com*",
            "*/q/d/l/*",
        ]

    async def fetch_api_data(self, url: str) -> Dict[str, Any]:
        """
        Fetch API data for a hybrid page.

        Delegates to the appropriate plugin based on URL domain.

        Args:
            url: Page URL

        Returns:
            API data from the appropriate plugin
        """
        url_lower = url.lower()

        if "coingecko.com" in url_lower:
            from liveweb_arena.plugins.coingecko.coingecko import CoinGeckoPlugin
            plugin = CoinGeckoPlugin()
            return await plugin.fetch_api_data(url)

        elif "stooq.com" in url_lower:
            from liveweb_arena.plugins.stooq.stooq import StooqPlugin
            plugin = StooqPlugin()
            return await plugin.fetch_api_data(url)

        return {}
