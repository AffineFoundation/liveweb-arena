"""
Taostats Plugin.

Plugin for Bittensor network data from taostats.io.
Uses official taostats.io API for ground truth.
"""

import re
from typing import Any, Dict, List
from urllib.parse import urlparse

from liveweb_arena.plugins.base import BasePlugin
from .api_client import fetch_single_subnet_data, fetch_homepage_api_data, initialize_cache


class TaostatsPlugin(BasePlugin):
    """
    Taostats plugin for Bittensor network data.

    Handles pages like:
    - https://taostats.io/ (homepage - all subnets)
    - https://taostats.io/subnets (subnet list)
    - https://taostats.io/subnets/27 (subnet detail)

    API data comes from taostats.io API (same source as website).
    """

    name = "taostats"

    allowed_domains = [
        "taostats.io",
        "www.taostats.io",
    ]

    def initialize(self):
        """Initialize plugin - fetch API data for question generation."""
        initialize_cache()

    def get_blocked_patterns(self) -> List[str]:
        """Block direct API access to force agents to use the website."""
        return [
            "*api.taostats.io*",
        ]

    def needs_api_data(self, url: str) -> bool:
        """
        Determine if this URL needs API data for ground truth.

        - Homepage/subnet list: needs API data (bulk subnets)
        - Subnet detail page: needs API data (single subnet)
        - Other pages: no API data needed
        """
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        # Homepage or subnets list
        if path == "" or path == "subnets":
            return True

        # Subnet detail page: /subnets/{id}
        if self._extract_subnet_id(url):
            return True

        return False

    async def fetch_api_data(self, url: str) -> Dict[str, Any]:
        """
        Fetch API data for a Taostats page.

        - Homepage/subnets list: Returns all subnets in {"subnets": {...}} format
        - Subnet detail page: Returns single subnet data

        Args:
            url: Page URL

        Returns:
            API data appropriate for the page type
        """
        # Check for detail page first
        subnet_id = self._extract_subnet_id(url)
        if subnet_id:
            data = await fetch_single_subnet_data(subnet_id)
            return data if data else {}

        # Homepage or subnets list - return all subnets
        if self._is_list_page(url):
            return await fetch_homepage_api_data()

        return {}

    def _is_list_page(self, url: str) -> bool:
        """Check if URL is homepage or subnets list."""
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        return path == "" or path == "subnets"

    def _extract_subnet_id(self, url: str) -> str:
        """
        Extract subnet ID from Taostats URL.

        Examples:
            https://taostats.io/subnets/27 -> 27
            https://taostats.io/subnets/1 -> 1
        """
        parsed = urlparse(url)
        path = parsed.path

        # Pattern: /subnets/{subnet_id}
        match = re.search(r'/subnets/(\d+)', path)
        if match:
            return match.group(1)

        return ""

    async def setup_page_for_cache(self, page, url: str) -> None:
        """
        Setup page before caching - click "ALL" to show all subnets.

        On taostats.io/subnets, the default view shows only 10-25 rows.
        Click "ALL" to show all ~128 subnets for complete visibility.
        """
        if not self._is_list_page(url):
            return

        try:
            # Click the "ALL" option in the rows selector
            # The selector shows: 10, 25, 50, 100, ALL
            all_button = page.locator('text="ALL"').first
            if await all_button.is_visible(timeout=3000):
                await all_button.click()
                # Wait for table to update with all rows
                await page.wait_for_timeout(2000)
                # Wait for network to settle after loading all rows
                try:
                    await page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    pass
        except Exception:
            # If "ALL" button not found or click fails, continue with default view
            pass
