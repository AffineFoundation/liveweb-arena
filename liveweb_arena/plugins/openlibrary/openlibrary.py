"""
Open Library Plugin.

Plugin for browsing and querying Open Library book data.
Supports subject pages, search results, and individual work pages.
"""

import re
from typing import Any, Dict, List
from urllib.parse import urlparse, parse_qs

from liveweb_arena.plugins.base import BasePlugin
from .api_client import (
    fetch_subject_api_data,
    fetch_search_api_data,
    fetch_work_api_data,
)


class OpenLibraryPlugin(BasePlugin):
    """
    Open Library plugin for book data queries.

    Handles pages like:
    - https://openlibrary.org/subjects/science_fiction (subject listing)
    - https://openlibrary.org/search?q=... (search results)
    - https://openlibrary.org/works/OL103123W/... (work detail)
    - https://openlibrary.org/authors/OL34184A/... (author page)

    API data includes: edition counts, ratings, read counts, publish years.
    """

    name = "openlibrary"

    allowed_domains = [
        "openlibrary.org",
    ]

    def get_blocked_patterns(self) -> List[str]:
        return [
            "*openlibrary.org/search.json*",
            "*openlibrary.org/subjects/*.json*",
            "*openlibrary.org/works/*.json*",
            "*openlibrary.org/authors/*.json*",
            "*openlibrary.org/trending/*.json*",
        ]

    async def fetch_api_data(self, url: str) -> Dict[str, Any]:
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        # Subject page: /subjects/<subject>
        subject = self._extract_subject(path)
        if subject:
            return await fetch_subject_api_data(subject, limit=20)

        # Search page: /search?q=...
        if path == "search":
            query = parse_qs(parsed.query).get("q", [""])[0]
            sort = parse_qs(parsed.query).get("sort", [None])[0]
            mode = parse_qs(parsed.query).get("mode", [None])[0]
            if query:
                return await fetch_search_api_data(query, limit=20, sort=sort, mode=mode)
            return {}

        # Work detail page: /works/OL...W or /works/OL...W/Title
        work_key = self._extract_work_key(path)
        if work_key:
            return await fetch_work_api_data(work_key)

        return {}

    def needs_api_data(self, url: str) -> bool:
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        if self._extract_subject(path):
            return True

        if path == "search" and parse_qs(parsed.query).get("q"):
            return True

        if self._extract_work_key(path):
            return True

        return False

    @staticmethod
    def _extract_subject(path: str) -> str:
        """Extract subject slug from URL path like 'subjects/science_fiction'."""
        match = re.match(r"subjects/([a-z0-9_]+)", path)
        if match:
            return match.group(1)
        return ""

    @staticmethod
    def _extract_work_key(path: str) -> str:
        """Extract work key from URL path like 'works/OL103123W' or 'works/OL103123W/Title'."""
        match = re.match(r"works/(OL\d+W)", path)
        if match:
            return f"/works/{match.group(1)}"
        return ""
