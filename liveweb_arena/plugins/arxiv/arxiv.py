"""
ArXiv Plugin.

Plugin for browsing and querying ArXiv academic paper listings.
Supports category listing pages (/list/cs.AI/new).
"""

import re
from typing import Any, Dict, List
from urllib.parse import urlparse

from liveweb_arena.plugins.base import BasePlugin
from .api_client import fetch_listing_api_data


class ArxivPlugin(BasePlugin):
    """
    ArXiv plugin for academic paper queries.

    Handles pages like:
    - https://arxiv.org/list/cs.AI/new (new submissions listing)

    Data source: HTML listing page — parsed directly so GT matches
    the page content the agent sees.
    """

    name = "arxiv"

    allowed_domains = [
        "arxiv.org",
    ]

    def get_blocked_patterns(self) -> List[str]:
        return [
            "*export.arxiv.org/api/*",
            "*rss.arxiv.org/*",
        ]

    async def fetch_api_data(self, url: str) -> Dict[str, Any]:
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        # Listing page: /list/<category>/new
        category = self._extract_category(path)
        if category:
            return await fetch_listing_api_data(category)

        return {}

    def needs_api_data(self, url: str) -> bool:
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        return bool(self._extract_category(path))

    @staticmethod
    def _extract_category(path: str) -> str:
        """Extract category from listing path like 'list/cs.AI/new'.

        Matches /new and /recent listings — both trigger GT collection.
        GT data is always fetched from /new regardless of which path the
        agent visited, so the ground truth is consistent.  /pastweek and
        month-archive paths are excluded (different paper sets).

        Handles all ArXiv category formats:
        - cs.AI, math.CO (group.SUBCAT)
        - hep-th, quant-ph (hyphenated, no dot)
        - cond-mat.str-el (hyphenated group with lowercase subcat)
        - astro-ph.CO (hyphenated group with uppercase subcat)
        """
        match = re.match(r"list/([a-z-]+(?:\.[A-Za-z-]+)?)/(new|recent)", path)
        if match:
            return match.group(1)
        return ""

