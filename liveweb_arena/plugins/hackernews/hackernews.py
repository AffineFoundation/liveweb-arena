"""
Hacker News Plugin.

Plugin for browsing and querying Hacker News content.
Supports external navigation to story link destinations.
"""

import contextvars
import re
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse, parse_qs

from liveweb_arena.plugins.base import BasePlugin
from .api_client import (
    fetch_homepage_api_data,
    fetch_category_api_data,
    fetch_item_api_data,
    fetch_user_api_data,
)

# Per-evaluation state via contextvars.
# Each evaluation context gets its own external URL tracking,
# so concurrent evaluations don't pollute each other's URL whitelists.
_external_urls_var: contextvars.ContextVar[Optional[Dict[str, Dict[str, Any]]]] = (
    contextvars.ContextVar("_hn_external_urls", default=None)
)
_external_domains_var: contextvars.ContextVar[Optional[Set[str]]] = (
    contextvars.ContextVar("_hn_external_domains", default=None)
)


class HackerNewsPlugin(BasePlugin):
    """
    Hacker News plugin for content queries.

    Handles pages like:
    - https://news.ycombinator.com/ (homepage - top stories)
    - https://news.ycombinator.com/item?id=12345 (story/comment detail)
    - https://news.ycombinator.com/ask (Ask HN)
    - https://news.ycombinator.com/show (Show HN)
    - https://news.ycombinator.com/jobs (Jobs)
    - https://news.ycombinator.com/user?id=username (user profile)
    - External story URLs (allowed dynamically based on API data)

    API data includes: title, score, author, comment count, etc.

    Anti-cheat: External URLs are only allowed if they appear in actual
    HN story data from the API. This prevents agents from navigating to
    arbitrary URLs not linked from HN.
    """

    name = "hackernews"

    allowed_domains = [
        "news.ycombinator.com",
    ]

    @classmethod
    def _get_urls(cls) -> Dict[str, Dict[str, Any]]:
        """Get per-context external URLs dict, initializing if needed."""
        urls = _external_urls_var.get()
        if urls is None:
            urls = {}
            _external_urls_var.set(urls)
        return urls

    @classmethod
    def _get_domains(cls) -> Set[str]:
        """Get per-context external domains set, initializing if needed."""
        domains = _external_domains_var.get()
        if domains is None:
            domains = set()
            _external_domains_var.set(domains)
        return domains

    def get_blocked_patterns(self) -> List[str]:
        """Block direct API access to force agents to use the website."""
        return [
            "*hacker-news.firebaseio.com*",  # Block Firebase API
            "*hn.algolia.com*",               # Block Algolia search API
        ]

    @classmethod
    def _extract_external_urls(cls, api_data: Dict[str, Any]) -> None:
        """
        Extract external URLs from HN API data and store them for validation.

        This enables anti-cheat: only URLs that appear in actual HN stories
        are allowed for external navigation.
        """
        urls = cls._get_urls()
        domains = cls._get_domains()
        stories = api_data.get("stories", {})
        for story_id, story in stories.items():
            url = story.get("url")
            if url and isinstance(url, str):
                # Skip HN internal URLs
                if "ycombinator.com" in url.lower():
                    continue
                try:
                    parsed = urlparse(url)
                    domain = parsed.netloc.lower()
                    if ":" in domain:
                        domain = domain.split(":")[0]
                    if domain.startswith("www."):
                        domain = domain[4:]
                    if domain:
                        urls[url] = story
                        domains.add(domain)
                except Exception:
                    pass

    @classmethod
    def get_external_domains(cls) -> Set[str]:
        """
        Get domains from legitimate external URLs found in HN stories.

        Returns:
            Set of domain names that can be navigated to
        """
        return cls._get_domains().copy()

    @classmethod
    def get_external_urls(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get all legitimate external URLs and their story data.

        Returns:
            Dict mapping URL to story data
        """
        return cls._get_urls().copy()

    @classmethod
    def _normalize_url_for_matching(cls, url: str) -> str:
        """Normalize URL for comparison (remove scheme variations, www, trailing slash)."""
        try:
            parsed = urlparse(url.lower())
            domain = parsed.netloc
            if domain.startswith("www."):
                domain = domain[4:]
            path = parsed.path.rstrip('/')
            # Return domain + path (ignore scheme, query, fragment)
            return f"{domain}{path}"
        except Exception:
            return url.lower().rstrip('/')

    @classmethod
    def is_legitimate_external_url(cls, url: str) -> bool:
        """
        Check if a URL is a legitimate external link from HN stories.

        Handles URL variations including:
        - Trailing slashes
        - www prefix
        - http vs https
        - Minor path differences from redirects (fallback to domain check)

        Args:
            url: URL to validate

        Returns:
            True if URL appears in HN story data or domain was linked from HN
        """
        urls = cls._get_urls()
        domains = cls._get_domains()

        # Exact match
        if url in urls:
            return True

        # Check without trailing slash variations
        url_clean = url.rstrip('/')
        if url_clean in urls:
            return True
        if url_clean + '/' in urls:
            return True

        # Normalized URL matching (handles www, scheme differences)
        normalized = cls._normalize_url_for_matching(url)
        for stored_url in urls:
            if cls._normalize_url_for_matching(stored_url) == normalized:
                return True

        # Domain-based fallback for redirect handling
        # If we're on a domain that was linked from HN, allow it
        try:
            parsed = urlparse(url.lower())
            domain = parsed.netloc
            if ":" in domain:
                domain = domain.split(":")[0]
            if domain.startswith("www."):
                domain = domain[4:]
            if domain in domains:
                return True
        except Exception:
            pass

        return False

    @classmethod
    def clear_external_urls(cls) -> None:
        """Clear tracked external URLs (call between evaluations)."""
        _external_urls_var.set(None)
        _external_domains_var.set(None)

    def _get_external_url_data(self, url: str) -> Dict[str, Any]:
        """
        Get data for an external URL linked from HN.

        Returns minimal data; title will be extracted from accessibility tree
        by gt_collector during page visit.

        Args:
            url: External URL

        Returns:
            Dict with is_external=True and HN story context
        """
        data = {
            "url": url,
            "is_external": True,
        }

        urls = self._get_urls()

        # Try to find the HN story that links to this URL
        # First try exact match
        url_clean = url.rstrip('/')
        story_data = (
            urls.get(url) or
            urls.get(url_clean) or
            urls.get(url_clean + '/')
        )

        # If no exact match, try normalized matching
        if not story_data:
            normalized = self._normalize_url_for_matching(url)
            for stored_url, stored_story in urls.items():
                if self._normalize_url_for_matching(stored_url) == normalized:
                    story_data = stored_story
                    break

        # If still no match, try domain-based matching (for redirects)
        if not story_data:
            try:
                parsed = urlparse(url.lower())
                domain = parsed.netloc
                if ":" in domain:
                    domain = domain.split(":")[0]
                if domain.startswith("www."):
                    domain = domain[4:]
                # Find any story linking to this domain
                for stored_url, stored_story in urls.items():
                    stored_parsed = urlparse(stored_url.lower())
                    stored_domain = stored_parsed.netloc
                    if stored_domain.startswith("www."):
                        stored_domain = stored_domain[4:]
                    if stored_domain == domain:
                        story_data = stored_story
                        break
            except Exception:
                pass

        if story_data:
            data["hn_story_id"] = story_data.get("id")
            data["hn_story_title"] = story_data.get("title")
            data["hn_story_rank"] = story_data.get("rank")

        return data

    async def fetch_api_data(self, url: str) -> Dict[str, Any]:
        """
        Fetch API data for a Hacker News page or external link.

        - Homepage: Returns top stories in {"stories": {...}} format
        - Category pages (ask/show/jobs): Returns category stories
        - Item page: Returns item details with "id" field
        - User page: Returns user info
        - External URL: Returns {is_external: True, ...} with HN story context

        Args:
            url: Page URL

        Returns:
            API data appropriate for the page type
        """
        parsed = urlparse(url)
        host = parsed.netloc.lower()

        # Check if this is an external URL (not HN domain)
        if "ycombinator.com" not in host:
            return self._get_external_url_data(url)

        path = parsed.path.strip('/')
        query = parse_qs(parsed.query)

        # Item detail page: /item?id=12345
        if path == "item" and "id" in query:
            item_id = int(query["id"][0])
            return await fetch_item_api_data(item_id)

        # User page: /user?id=username
        if path == "user" and "id" in query:
            username = query["id"][0]
            return await fetch_user_api_data(username)

        # Category pages
        if path == "ask":
            data = await fetch_category_api_data("ask")
            self._extract_external_urls(data)
            return data
        if path == "show":
            data = await fetch_category_api_data("show")
            self._extract_external_urls(data)
            return data
        if path == "jobs":
            data = await fetch_category_api_data("jobs")
            self._extract_external_urls(data)
            return data

        # Homepage (including news, newest, front, etc. - all show top stories)
        if path in ("", "news", "newest", "front") or not path:
            data = await fetch_homepage_api_data()
            self._extract_external_urls(data)
            return data

        # Unknown HN page type - return empty
        return {}

    def needs_api_data(self, url: str) -> bool:
        """
        Determine if this URL needs API data for ground truth.

        Args:
            url: Page URL

        Returns:
            True if API data is needed and available, False otherwise
        """
        parsed = urlparse(url)
        host = parsed.netloc.lower()

        # External URLs (non-HN) need data extraction for GT
        if "ycombinator.com" not in host:
            return self.is_legitimate_external_url(url)

        path = parsed.path.strip('/')
        query = parse_qs(parsed.query)

        # Item detail page needs API data
        if path == "item" and "id" in query:
            return True

        # User page needs API data
        if path == "user" and "id" in query:
            return True

        # Category pages need API data
        if path in ("ask", "show", "jobs"):
            return True

        # Homepage needs API data
        if path in ("", "news", "newest", "front") or not path:
            return True

        # Other pages (submit, login, etc.) don't need API data
        return False

    def is_url_allowed(self, url: str) -> bool:
        """
        Check if a URL is allowed for this plugin.

        Extends base domain check to include legitimate external URLs.

        Args:
            url: URL to check

        Returns:
            True if URL is allowed, False otherwise
        """
        parsed = urlparse(url)
        host = parsed.netloc.lower()

        # Always allow HN domain
        if "ycombinator.com" in host:
            return True

        # Allow legitimate external URLs from HN stories
        return self.is_legitimate_external_url(url)
