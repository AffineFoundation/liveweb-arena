"""
Weather Plugin.

Plugin for weather data from wttr.in.
"""

import re
from typing import Any, Dict, List
from urllib.parse import urlparse, unquote

from liveweb_arena.plugins.base import BasePlugin
from .api_client import fetch_single_location_data


class WeatherPlugin(BasePlugin):
    """
    Weather plugin for wttr.in data.

    Handles pages like:
    - https://wttr.in/London
    - https://wttr.in/New+York
    - https://v2.wttr.in/Tokyo

    API data includes: current_condition, 3-day forecast, astronomy, etc.
    """

    name = "weather"

    allowed_domains = [
        "wttr.in",
        "v2.wttr.in",
    ]

    def get_blocked_patterns(self) -> List[str]:
        """Block JSON API access to force agents to use the HTML website."""
        return [
            "*?format=j1*",
            "*?format=json*",
        ]

    async def fetch_api_data(self, url: str) -> Dict[str, Any]:
        """
        Fetch API data for a wttr.in weather page.

        Extracts location from URL and fetches weather data.

        Args:
            url: Page URL (e.g., https://wttr.in/London)

        Returns:
            Complete weather JSON data from wttr.in
        """
        location = self._extract_location(url)
        if not location:
            return {}

        data = await fetch_single_location_data(location)
        return data if data else {}

    def _extract_location(self, url: str) -> str:
        """
        Extract location from wttr.in URL.

        Examples:
            https://wttr.in/London -> London
            https://wttr.in/New+York -> New+York
            https://v2.wttr.in/Tokyo,Japan -> Tokyo,Japan
        """
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        if not path:
            return ""

        # Decode URL encoding
        location = unquote(path)

        # Remove format suffix if present
        location = re.sub(r'\?.*$', '', location)

        return location
