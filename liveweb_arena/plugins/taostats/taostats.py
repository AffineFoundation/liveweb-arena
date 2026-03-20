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


class TaostatsPrefetchSetupError(RuntimeError):
    def __init__(self, message: str, *, prefetch_phase: str, wait_target: str | None = None):
        super().__init__(message)
        self.prefetch_phase = prefetch_phase
        self.wait_target = wait_target


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

    def get_stable_url_patterns(self) -> List[str]:
        return [
            "/",
            "/subnets",
            "/subnets/<id>",
            "/subnets/<id>/chart",
            "/validators",
        ]

    def classify_url(self, url: str) -> str | None:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path = parsed.path.lower()
        if "taostats.io" not in host:
            return None
        if path in {"", "/", "/subnets", "/validators"}:
            return None
        if path.startswith("/subnets/"):
            return None
        return "model_invalid_url_shape"

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
            if not data:
                raise ValueError(f"Taostats API returned no data for subnet_id={subnet_id}")
            return data

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
            https://taostats.io/subnets/netuid-27/ -> 27
            https://taostats.io/subnets/1 -> 1
        """
        parsed = urlparse(url)
        path = parsed.path

        # Pattern: /subnets/{subnet_id} or /subnets/netuid-{subnet_id}
        match = re.search(r'/subnets/(?:netuid-)?(\d+)', path)
        if match:
            return match.group(1)

        return ""

    async def setup_page_for_cache(self, page, url: str) -> None:
        """
        Setup page before caching - click "ALL" to show all subnets.

        On taostats.io/subnets, the default view shows only 10-25 rows.
        Click "ALL" to show all ~128 subnets for complete visibility.
        """
        if self._is_list_page(url):
            try:
                all_button = page.get_by_text("ALL", exact=True).first
                if await all_button.is_visible(timeout=5000):
                    await all_button.click()
                    await page.wait_for_timeout(2500)
                    try:
                        await page.wait_for_load_state("networkidle", timeout=10000)
                    except Exception:
                        pass
                await page.locator("text=Rows:").first.wait_for(timeout=5000)
            except Exception:
                pass
            return

        if self._extract_subnet_id(url):
            last_error = None
            for selector in (
                "text=Statistics",
                "text=Transactions",
                "text=Holders",
                "text=Price Impact",
            ):
                try:
                    await page.locator(selector).first.wait_for(timeout=4000)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = TaostatsPrefetchSetupError(
                        f"Taostats detail wait failed for {selector}: {exc}",
                        prefetch_phase="setup_page_for_cache",
                        wait_target=selector,
                    )
                    continue
            if last_error is not None:
                raise last_error
            await page.wait_for_timeout(2000)
