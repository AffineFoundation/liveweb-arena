"""
Cache Module - On-demand page caching with file locking.

Design:
- Each URL gets its own directory based on URL structure
- HTML and API data are fetched together and stored atomically
- File locks ensure multi-process safety
- TTL-based expiration with automatic refresh

Directory structure:
    cache/
    └── www.coingecko.com/
        └── en/
            └── coins/
                └── bitcoin/
                    ├── page.json   # {url, html, api_data, fetched_at}
                    └── .lock
"""

import fcntl
import json
import logging
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from liveweb_arena.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

# Default TTL: 24 hours
DEFAULT_TTL = 24 * 3600


def log(tag: str, message: str):
    """Simple logging helper."""
    print(f"[{tag}] {message}")


@dataclass
class CachedPage:
    """Cached page data."""
    url: str
    html: str
    api_data: Optional[Dict[str, Any]]
    fetched_at: float

    def is_expired(self, ttl: int) -> bool:
        return time.time() > self.fetched_at + ttl

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "html": self.html,
            "api_data": self.api_data,
            "fetched_at": self.fetched_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CachedPage":
        return cls(
            url=data["url"],
            html=data["html"],
            api_data=data.get("api_data"),
            fetched_at=data["fetched_at"],
        )


@dataclass
class PageRequirement:
    """Page caching requirement."""
    url: str
    need_api: bool = False

    @staticmethod
    def nav(url: str) -> "PageRequirement":
        """Create navigation page requirement (HTML only)."""
        return PageRequirement(url, need_api=False)

    @staticmethod
    def data(url: str) -> "PageRequirement":
        """Create data page requirement (HTML + API)."""
        return PageRequirement(url, need_api=True)


@contextmanager
def file_lock(lock_path: Path):
    """Cross-process file lock."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, 'w') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def safe_path_component(s: str) -> str:
    """Convert string to safe path component."""
    # Replace dangerous characters
    s = re.sub(r'[<>:"/\\|?*]', '_', s)
    s = s.replace(' ', '_')
    s = s.replace(',', '_')
    s = s.replace('&', '_')
    # Limit length
    if len(s) > 200:
        s = s[:200]
    return s


def normalize_url(url: str) -> str:
    """
    Normalize URL for cache lookup.

    Rules:
    1. Lowercase domain
    2. Remove default ports
    3. Remove tracking parameters
    4. Sort remaining query parameters
    5. Lowercase query parameter values (for case-insensitive matching)
    """
    parsed = urlparse(url)

    # Lowercase domain
    domain = parsed.netloc.lower()

    # Remove default ports
    if domain.endswith(':80') or domain.endswith(':443'):
        domain = domain.rsplit(':', 1)[0]

    # Path (preserve case for path components)
    path = parsed.path or '/'

    # Filter, sort, and lowercase query parameters
    if parsed.query:
        params = []
        tracking = {'utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term', 'ref', 'source'}
        for part in parsed.query.split('&'):
            if '=' in part:
                key = part.split('=')[0].lower()
                if key not in tracking:
                    # Lowercase the entire parameter (key=value)
                    params.append(part.lower())
            else:
                params.append(part.lower())
        query = '&'.join(sorted(params))
    else:
        query = ''

    result = f"{parsed.scheme}://{domain}{path}"
    if query:
        result += f"?{query}"
    return result


def url_to_cache_dir(cache_dir: Path, url: str) -> Path:
    """
    Convert URL to cache directory path.

    Examples:
    https://www.coingecko.com/en/coins/bitcoin
    → cache/www.coingecko.com/en/coins/bitcoin/

    https://stooq.com/q/?s=aapl.us
    → cache/stooq.com/q/__s=aapl.us/
    """
    parsed = urlparse(url)

    # Domain (lowercase)
    domain = parsed.netloc.lower()
    if domain.endswith(':80') or domain.endswith(':443'):
        domain = domain.rsplit(':', 1)[0]

    # Path parts
    path = parsed.path.strip('/')
    if path:
        path_parts = [safe_path_component(p) for p in path.split('/')]
    else:
        path_parts = ['_root_']

    # Query parameters (lowercase for case-insensitive matching)
    if parsed.query:
        query_part = '__' + safe_path_component(parsed.query.lower())
        path_parts[-1] = path_parts[-1] + query_part

    return cache_dir / domain / '/'.join(path_parts)


def url_display(url: str) -> str:
    """Get short display string for URL."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path
    if len(path) > 40:
        path = path[:37] + '...'
    return f"{domain}{path}"


class CacheManager:
    """
    Unified cache manager.

    Features:
    - On-demand caching
    - File lock protection for multi-process safety
    - TTL-based expiration
    - API data caching for ground truth validation
    """

    def __init__(self, cache_dir: Path, ttl: int = DEFAULT_TTL):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self._browser = None

    async def ensure_cached(
        self,
        pages: List[PageRequirement],
        plugin: "BasePlugin",
    ) -> Dict[str, CachedPage]:
        """
        Ensure specified pages are cached.

        Args:
            pages: List of page requirements
            plugin: Plugin for fetching API data

        Returns:
            {normalized_url: CachedPage} mapping
        """
        result = {}

        for page_req in pages:
            normalized = normalize_url(page_req.url)
            cached = await self._ensure_single(page_req.url, plugin, page_req.need_api)
            result[normalized] = cached

        return result

    async def _ensure_single(
        self,
        url: str,
        plugin: "BasePlugin",
        need_api: bool,
    ) -> CachedPage:
        """Ensure single URL is cached."""
        normalized = normalize_url(url)
        cache_dir = url_to_cache_dir(self.cache_dir, normalized)
        cache_file = cache_dir / "page.json"
        lock_file = cache_dir / ".lock"

        page_type = "data" if need_api else "nav"

        # 1. Quick check (no lock)
        cached = self._load_if_valid(cache_file, need_api)
        if cached:
            log("Cache", f"HIT {page_type} - {url_display(normalized)}")
            return cached

        # 2. Need update, acquire lock
        with file_lock(lock_file):
            # 3. Double check (another process may have updated)
            cached = self._load_if_valid(cache_file, need_api)
            if cached:
                log("Cache", f"HIT {page_type} (after lock) - {url_display(normalized)}")
                return cached

            # 4. Actually fetch
            log("Cache", f"MISS {page_type} - fetching {url_display(normalized)}")
            start = time.time()

            # Fetch page HTML
            html = await self._fetch_page(url)

            # Only fetch API data for data pages
            api_data = None
            if need_api:
                try:
                    api_data = await plugin.fetch_api_data(url)
                except Exception as e:
                    logger.warning(f"Failed to fetch API data for {url}: {e}")

            cached = CachedPage(
                url=url,
                html=html,
                api_data=api_data,
                fetched_at=time.time(),
            )

            self._save(cache_file, cached)
            elapsed = time.time() - start
            log("Cache", f"SAVED {page_type} - {url_display(normalized)} ({elapsed:.1f}s)")
            return cached

    def _load_if_valid(self, cache_file: Path, need_api: bool) -> Optional[CachedPage]:
        """Load cache if valid."""
        if not cache_file.exists():
            return None

        try:
            cached = self._load(cache_file)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_file}: {e}")
            return None

        if cached.is_expired(self.ttl):
            return None

        # If need API data but cache doesn't have it, treat as invalid
        # Empty dict {} is also invalid - must have actual data
        if need_api and not cached.api_data:
            return None

        return cached

    def _load(self, cache_file: Path) -> CachedPage:
        """Load cache from file."""
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return CachedPage.from_dict(data)

    def _save(self, cache_file: Path, cached: CachedPage):
        """Save cache to file."""
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cached.to_dict(), f, ensure_ascii=False)

    async def _fetch_page(self, url: str) -> str:
        """Fetch page HTML using Playwright."""
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
            )
            try:
                context = await browser.new_context(
                    viewport={"width": 1280, "height": 720},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                )
                page = await context.new_page()

                await page.goto(url, timeout=60000, wait_until="domcontentloaded")

                # Wait for network idle
                try:
                    await page.wait_for_load_state("networkidle", timeout=15000)
                except Exception:
                    pass

                # Scroll to trigger lazy loading
                for pos in [0, 500, 1000, 2000]:
                    await page.evaluate(f"window.scrollTo(0, {pos})")
                    await page.wait_for_timeout(300)

                await page.evaluate("window.scrollTo(0, 0)")
                await page.wait_for_timeout(500)

                html = await page.content()

                await context.close()
                return html

            finally:
                await browser.close()

    def get_cached(self, url: str) -> Optional[CachedPage]:
        """Get cached page without triggering update."""
        normalized = normalize_url(url)
        cache_dir = url_to_cache_dir(self.cache_dir, normalized)
        cache_file = cache_dir / "page.json"

        if not cache_file.exists():
            return None

        try:
            return self._load(cache_file)
        except Exception:
            return None


# Global instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        cache_dir = Path(__file__).parent.parent.parent / "cache"
        _cache_manager = CacheManager(cache_dir)
    return _cache_manager


def set_cache_manager(manager: CacheManager):
    """Set global cache manager instance."""
    global _cache_manager
    _cache_manager = manager
