"""
Taostats Plugin.

Plugin for Bittensor network data from taostats.io.
"""

import re
from typing import Any, Dict, List
from urllib.parse import urlparse

from liveweb_arena.plugins.base import BasePlugin
from .api_client import fetch_single_subnet_data


class TaostatsPlugin(BasePlugin):
    """
    Taostats plugin for Bittensor network data.

    Handles pages like:
    - https://taostats.io/subnets/27
    - https://taostats.io/subnets

    API data includes: subnet name, owner, price, etc.
    """

    name = "taostats"

    allowed_domains = [
        "taostats.io",
        "www.taostats.io",
    ]

    def get_blocked_patterns(self) -> List[str]:
        """Block API access to force agents to use the website."""
        return [
            "*api.taostats.io*",
        ]

    async def fetch_api_data(self, url: str) -> Dict[str, Any]:
        """
        Fetch API data for a Taostats page.

        Extracts subnet_id from URL and fetches subnet data.

        Args:
            url: Page URL (e.g., https://taostats.io/subnets/27)

        Returns:
            Subnet data from Bittensor network
        """
        subnet_id = self._extract_subnet_id(url)
        if not subnet_id:
            return {}

        data = await fetch_single_subnet_data(subnet_id)
        return data if data else {}

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
