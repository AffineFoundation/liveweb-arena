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

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import unquote, urlparse

if TYPE_CHECKING:
    from liveweb_arena.plugins.base import BasePlugin

logger = logging.getLogger(__name__)

# Default TTL: 48 hours
DEFAULT_TTL = 48 * 3600


class CacheFatalError(Exception):
    """
    Raised when page caching fails due to network issues.

    This indicates the browser cannot load the page, making evaluation invalid.
    Evaluation should be terminated immediately.
    """

    def __init__(
        self,
        message: str,
        url: str = None,
        kind: str = "fatal",
        fatal: bool = True,
        status_code: int | None = None,
        evidence: dict | None = None,
        soft_fail_applied: bool = False,
        stale_fallback_used: bool = False,
        plugin_name: str | None = None,
    ):
        super().__init__(message)
        self.url = url
        self.kind = kind
        self.fatal = fatal
        self.status_code = status_code
        self.evidence = evidence or {}
        self.soft_fail_applied = soft_fail_applied
        self.stale_fallback_used = stale_fallback_used
        self.plugin_name = plugin_name


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
    accessibility_tree: Optional[str] = None  # Cached for deterministic evaluation
    need_api: bool = True  # Whether this page requires API data (default True for safety)

    def is_expired(self, ttl: int) -> bool:
        return time.time() > self.fetched_at + ttl

    def is_complete(self) -> bool:
        """Check if cache is complete (has API data if needed)."""
        if self.need_api:
            return self.api_data is not None and len(self.api_data) > 0
        return True

    def to_dict(self) -> dict:
        result = {
            "url": self.url,
            "html": self.html,
            "api_data": self.api_data,
            "fetched_at": self.fetched_at,
            "need_api": self.need_api,
        }
        if self.accessibility_tree:
            result["accessibility_tree"] = self.accessibility_tree
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "CachedPage":
        return cls(
            url=data["url"],
            html=data["html"],
            api_data=data.get("api_data"),
            fetched_at=data["fetched_at"],
            accessibility_tree=data.get("accessibility_tree"),
            need_api=data.get("need_api", True),  # Default True for old caches
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


@dataclass
class CacheFetchResult:
    """Result of ensuring one cached page."""

    page: CachedPage
    source: str  # hit|hit_after_lock|stale|fresh
    normalized_url: str
    domain_key: str
    latency_s: float = 0.0
    stale: bool = False


async def async_file_lock_acquire(lock_path: Path, timeout: float = 60.0) -> int:
    """
    Acquire file lock asynchronously (non-blocking with retry).

    Returns file descriptor that must be released with async_file_lock_release().

    This avoids blocking the event loop while waiting for the lock.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    while True:
        fd = open(lock_path, 'w')
        try:
            # Try non-blocking lock
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd  # Lock acquired, return file object
        except BlockingIOError:
            fd.close()
            # Lock held by another process, wait and retry
            if time.time() - start > timeout:
                raise TimeoutError(f"Could not acquire lock {lock_path} within {timeout}s")
            await asyncio.sleep(0.1)  # Yield to event loop
        except Exception:
            fd.close()
            raise


def async_file_lock_release(fd):
    """Release file lock acquired by async_file_lock_acquire()."""
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
    finally:
        fd.close()


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
    4. Decode percent-encoding in path and query values
    5. Lowercase query keys and values
    6. Sort remaining query parameters
    """
    parsed = urlparse(url)

    # Lowercase domain
    domain = parsed.netloc.lower()

    # Remove default ports
    if domain.endswith(':80') or domain.endswith(':443'):
        domain = domain.rsplit(':', 1)[0]

    # Path: decode percent-encoding, normalize spaces to +
    path = unquote(parsed.path or '/').replace(' ', '+')

    # Filter, sort, and normalize query parameters
    if parsed.query:
        params = []
        tracking = {'utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term', 'ref', 'source'}
        for part in parsed.query.split('&'):
            if '=' in part:
                key = unquote(part.split('=')[0]).lower()
                if key not in tracking:
                    # Decode percent-encoding and lowercase for consistent matching
                    value = unquote(part.split('=', 1)[1]).lower()
                    params.append(f"{key}={value}")
            else:
                params.append(unquote(part).lower())
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

    # Path parts - decode percent-encoding, normalize spaces to +
    path = unquote(parsed.path).replace(' ', '+').strip('/')
    if path:
        path_parts = [safe_path_component(p) for p in path.split('/')]
    else:
        path_parts = ['_root_']

    # Query parameters (decode percent-encoding + lowercase for consistent matching)
    if parsed.query:
        query_part = '__' + safe_path_component(unquote(parsed.query).lower())
        path_parts[-1] = path_parts[-1] + query_part

    return cache_dir / domain / '/'.join(path_parts)


def url_display(url: str) -> str:
    """Get short display string for URL."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path
    query = f"?{parsed.query}" if parsed.query else ""
    display = f"{domain}{path}{query}"
    if len(display) > 80:
        display = display[:77] + '...'
    return display


class CacheManager:
    """
    Unified cache manager.

    Features:
    - On-demand caching
    - File lock protection for multi-process safety
    - TTL-based expiration
    - API data caching for ground truth validation
    """

    # Minimum interval between consecutive cache-miss fetches (seconds)
    _PREFETCH_INTERVAL = 1.0

    def __init__(self, cache_dir: Path, ttl: int = None):
        self.cache_dir = Path(cache_dir)
        if ttl is None:
            ttl = int(os.environ.get("LIVEWEB_CACHE_TTL", str(DEFAULT_TTL)))
        self.ttl = ttl
        self._prefetch_interval = float(
            os.environ.get("LIVEWEB_CACHE_PREFETCH_INTERVAL", str(self._PREFETCH_INTERVAL))
        )
        self.allow_stale = os.environ.get("LIVEWEB_CACHE_ALLOW_STALE", "1") == "1"
        self.stale_max_age = int(os.environ.get("LIVEWEB_CACHE_STALE_MAX_AGE", str(24 * 3600)))
        self.max_cache_fetches = int(os.environ.get("LIVEWEB_MAX_CACHE_FETCHES", "6"))
        self._playwright = None
        self._browser = None
        self._browser_lock = asyncio.Lock()
        self._last_fetch_time: float = 0
        self._global_fetch_semaphore = asyncio.Semaphore(max(1, self.max_cache_fetches))
        self._domain_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._refresh_tasks: Dict[str, asyncio.Task] = {}

    async def _ensure_browser(self):
        """Ensure shared Playwright browser is running (lazy singleton)."""
        if self._browser is not None:
            return
        async with self._browser_lock:
            if self._browser is not None:
                return
            from playwright.async_api import async_playwright
            from liveweb_arena.core.block_patterns import STEALTH_BROWSER_ARGS
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True, args=STEALTH_BROWSER_ARGS,
            )

    async def shutdown(self):
        """Shutdown shared browser and Playwright."""
        async with self._browser_lock:
            if self._browser:
                try:
                    await asyncio.wait_for(self._browser.close(), timeout=3)
                except Exception:
                    pass
                self._browser = None
            if self._playwright:
                try:
                    await asyncio.wait_for(self._playwright.stop(), timeout=3)
                except Exception:
                    pass
                self._playwright = None

    async def ensure_cached(
        self,
        pages: List[PageRequirement],
        plugin: "BasePlugin",
    ) -> Dict[str, CacheFetchResult]:
        """
        Ensure specified pages are cached.

        Args:
            pages: List of page requirements
            plugin: Plugin for fetching API data

        Returns:
            {normalized_url: CachedPage} mapping
        """
        result: Dict[str, CacheFetchResult] = {}

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
        allow_stale_lookup: bool = True,
    ) -> CacheFetchResult:
        """Ensure single URL is cached."""
        normalized = normalize_url(url)
        cache_dir = url_to_cache_dir(self.cache_dir, normalized)
        cache_file = cache_dir / "page.json"
        lock_file = cache_dir / ".lock"
        domain_key = self._domain_key_for_url(url)
        page_type = "data" if need_api else "nav"

        # 1. Quick check (no lock)
        status, cached = self._load_with_status(cache_file, need_api)
        if status == "valid" and cached:
            log("Cache", f"HIT {page_type} - {url_display(normalized)}")
            return CacheFetchResult(
                page=cached,
                source="hit",
                normalized_url=normalized,
                domain_key=domain_key,
            )
        if allow_stale_lookup and status == "stale" and cached:
            log("Cache", f"STALE {page_type} - serving {url_display(normalized)}")
            self._schedule_refresh(url, plugin, need_api)
            return CacheFetchResult(
                page=cached,
                source="stale",
                normalized_url=normalized,
                domain_key=domain_key,
                stale=True,
            )

        # 2. Need update, acquire async lock (non-blocking to avoid deadlock)
        lock_fd = await async_file_lock_acquire(lock_file)
        try:
            # 3. Double check (another process may have updated)
            status, cached = self._load_with_status(cache_file, need_api)
            if status == "valid" and cached:
                log("Cache", f"HIT {page_type} (after lock) - {url_display(normalized)}")
                return CacheFetchResult(
                    page=cached,
                    source="hit_after_lock",
                    normalized_url=normalized,
                    domain_key=domain_key,
                )
            if allow_stale_lookup and status == "stale" and cached:
                log("Cache", f"STALE {page_type} (after lock) - serving {url_display(normalized)}")
                self._schedule_refresh(url, plugin, need_api)
                return CacheFetchResult(
                    page=cached,
                    source="stale",
                    normalized_url=normalized,
                    domain_key=domain_key,
                    stale=True,
                )

            # 4. Actually fetch - page and API in parallel when possible
            log("Cache", f"MISS {page_type} - fetching {url_display(normalized)}")

            # Rate limit consecutive fetches to avoid triggering anti-bot
            now = time.time()
            wait = self._prefetch_interval - (now - self._last_fetch_time)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_fetch_time = time.time()

            start = time.time()
            async with self._global_fetch_semaphore:
                async with self._domain_semaphore(domain_key):
                    html, accessibility_tree, api_data = await self._fetch_and_build_cache(
                        url=url,
                        plugin=plugin,
                        need_api=need_api,
                    )

            cached = CachedPage(
                url=url,
                html=html,
                api_data=api_data,
                fetched_at=time.time(),
                accessibility_tree=accessibility_tree,
                need_api=need_api,
            )

            self._save(cache_file, cached)
            elapsed = time.time() - start
            log("Cache", f"SAVED {page_type} - {url_display(normalized)} ({elapsed:.1f}s)")
            return CacheFetchResult(
                page=cached,
                source="fresh",
                normalized_url=normalized,
                domain_key=domain_key,
                latency_s=elapsed,
            )
        finally:
            async_file_lock_release(lock_fd)

    def _load_with_status(self, cache_file: Path, need_api: bool) -> tuple[str, Optional[CachedPage]]:
        """Load cache and classify it as valid, stale, invalid, or missing."""
        if not cache_file.exists():
            return "missing", None

        try:
            cached = self._load(cache_file)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_file}: {e}")
            self._delete_cache(cache_file)
            return "invalid", None

        if not cached.is_complete() or (need_api and not cached.api_data):
            log("Cache", f"Incomplete (missing API) - deleting {url_display(cached.url)}")
            self._delete_cache(cache_file)
            return "invalid", None

        age = time.time() - cached.fetched_at
        if age <= self.ttl:
            return "valid", cached

        if self.allow_stale and age <= self.ttl + self.stale_max_age:
            return "stale", cached

        self._delete_cache(cache_file)
        return "expired", None

    def _load_if_valid(self, cache_file: Path, need_api: bool) -> Optional[CachedPage]:
        """Backward-compatible wrapper for tests and older callers.

        Returns the cached page only when it is currently valid under the
        configured TTL and completeness rules. Stale, invalid, expired, and
        missing entries all map to ``None``.
        """
        status, cached = self._load_with_status(cache_file, need_api)
        return cached if status == "valid" else None

    async def _fetch_and_build_cache(
        self,
        url: str,
        plugin: "BasePlugin",
        need_api: bool,
    ) -> tuple[str, str, Optional[Dict[str, Any]]]:
        if need_api:
            page_task = asyncio.create_task(self._fetch_page(url, plugin))
            api_task = asyncio.create_task(plugin.fetch_api_data(url))

            page_result = None
            page_error = None
            api_data = None
            api_error = None

            try:
                page_result = await page_task
            except Exception as e:
                page_error = e
                api_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await api_task

            if page_error is None:
                try:
                    api_data = await api_task
                except Exception as e:
                    api_error = e

            if page_error is not None:
                if isinstance(page_error, CacheFatalError):
                    raise page_error
                raise CacheFatalError(
                    f"Page fetch failed (browser cannot load): {page_error}",
                    url=url,
                )

            html, accessibility_tree = page_result

            if api_error is not None:
                raise CacheFatalError(
                    f"API data fetch failed (GT will be invalid): {api_error}",
                    url=url,
                    kind="api_error",
                    fatal=False,
                )
            if not api_data:
                raise CacheFatalError(
                    "API data is empty (GT will be invalid)",
                    url=url,
                    kind="api_empty",
                    fatal=False,
                )
            return html, accessibility_tree, api_data

        try:
            html, accessibility_tree = await self._fetch_page(url, plugin)
        except Exception as e:
            if isinstance(e, CacheFatalError):
                raise e
            raise CacheFatalError(
                f"Page fetch failed (browser cannot load): {e}",
                url=url,
            )
        return html, accessibility_tree, None

    def _schedule_refresh(self, url: str, plugin: "BasePlugin", need_api: bool):
        normalized = normalize_url(url)
        existing = self._refresh_tasks.get(normalized)
        if existing and not existing.done():
            return
        task = asyncio.create_task(self._background_refresh(url, plugin, need_api, normalized))
        self._refresh_tasks[normalized] = task

    async def _background_refresh(self, url: str, plugin: "BasePlugin", need_api: bool, normalized: str):
        try:
            await self._ensure_single(url, plugin, need_api, allow_stale_lookup=False)
        except Exception as e:
            log("Cache", f"Background refresh skipped for {url_display(normalized)}: {e}")
        finally:
            self._refresh_tasks.pop(normalized, None)

    def _domain_key_for_url(self, url: str) -> str:
        hostname = urlparse(url).netloc.lower()
        if "coingecko" in hostname:
            return "coingecko"
        if "stooq" in hostname:
            return "stooq"
        if "news.ycombinator" in hostname:
            return "news_ycombinator"
        if "taostats" in hostname:
            return "taostats"
        return "default"

    def _domain_limit_for_key(self, domain_key: str) -> int:
        env_name = f"LIVEWEB_CACHE_DOMAIN_LIMIT_{domain_key.upper()}"
        default_limit = int(os.environ.get("LIVEWEB_CACHE_DOMAIN_LIMIT_DEFAULT", "2"))
        return max(1, int(os.environ.get(env_name, str(default_limit))))

    def _domain_semaphore(self, domain_key: str) -> asyncio.Semaphore:
        sem = self._domain_semaphores.get(domain_key)
        if sem is None:
            sem = asyncio.Semaphore(self._domain_limit_for_key(domain_key))
            self._domain_semaphores[domain_key] = sem
        return sem

    def _delete_cache(self, cache_file: Path):
        """Delete cache file."""
        try:
            if cache_file.exists():
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete cache {cache_file}: {e}")

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

    async def _fetch_page(self, url: str, plugin=None) -> tuple:
        """
        Fetch page HTML and accessibility tree using shared Playwright browser.

        Args:
            url: Page URL to fetch
            plugin: Optional plugin for page setup (e.g., click "Show All")

        Returns:
            (html, accessibility_tree) tuple
        """
        from liveweb_arena.core.block_patterns import (
            STEALTH_INIT_SCRIPT, STEALTH_USER_AGENT,
            is_captcha_page, should_block_url,
        )

        await self._ensure_browser()
        try:
            context = await self._browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent=STEALTH_USER_AGENT,
            )
        except Exception:
            # Browser process may have crashed — clean up and retry once
            await self.shutdown()
            await self._ensure_browser()
            context = await self._browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent=STEALTH_USER_AGENT,
            )
        try:
            page = await context.new_page()
            await page.add_init_script(STEALTH_INIT_SCRIPT)

            # Block tracking/ads to avoid networkidle delays
            # Merge global + plugin-specific block patterns
            plugin_block_res = []
            if plugin and hasattr(plugin, 'get_blocked_patterns'):
                for pat in plugin.get_blocked_patterns():
                    regex_pat = re.escape(pat).replace(r"\*", ".*")
                    plugin_block_res.append(re.compile(regex_pat, re.IGNORECASE))

            async def _block_tracking(route):
                req_url = route.request.url
                if should_block_url(req_url):
                    await route.abort("blockedbyclient")
                    return
                for pat_re in plugin_block_res:
                    if pat_re.search(req_url):
                        await route.abort("blockedbyclient")
                        return
                await route.continue_()

            await page.route("**/*", _block_tracking)

            response = await page.goto(url, timeout=60000, wait_until="domcontentloaded")

            # Layer 1: HTTP status check
            if response and response.status >= 400:
                fatal = response.status not in (404, 410)
                raise CacheFatalError(
                    f"HTTP {response.status} for {url}",
                    url=url,
                    kind=f"http_{response.status}",
                    fatal=fatal,
                    status_code=response.status,
                    evidence={"page_url": url},
                )

            # Wait for network idle (short timeout: ads are blocked, so
            # legitimate content loads in ~3-4s; streaming endpoints like
            # aq*.stooq.com keep connections open indefinitely)
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass

            # Plugin-specific page setup (e.g., click "ALL" to show all rows)
            if plugin and hasattr(plugin, 'setup_page_for_cache'):
                try:
                    await plugin.setup_page_for_cache(page, url)
                except Exception as e:
                    log("Cache", f"Page setup failed (continuing): {e}")

            # Scroll to trigger lazy loading
            for pos in [0, 500, 1000, 2000]:
                await page.evaluate(f"window.scrollTo(0, {pos})")
                await page.wait_for_timeout(300)

            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(500)

            html = await page.content()

            # Layer 2: CAPTCHA/challenge detection
            page_title = await page.title()
            if is_captcha_page(html, page_title):
                raise CacheFatalError(
                    f"CAPTCHA/challenge page detected (title: {page_title!r})",
                    url=url,
                    evidence={"page_title": page_title},
                )

            # Layer 3: Minimum content length (real pages are >5KB)
            if len(html) < 1000:
                raise CacheFatalError(
                    f"Page too short ({len(html)} bytes, title: {page_title!r})",
                    url=url,
                    evidence={"page_title": page_title, "html_length": len(html)},
                )

            # Extract accessibility tree for deterministic caching
            a11y_tree = ""
            try:
                a11y_snapshot = await page.accessibility.snapshot()
                if a11y_snapshot:
                    a11y_tree = self._format_accessibility_tree(a11y_snapshot)
            except Exception:
                pass

            # If accessibility tree is empty, get page text content
            if len(a11y_tree.strip()) < 100:
                try:
                    page_text = await page.evaluate("""
                        () => {
                            const preElements = document.querySelectorAll('pre');
                            if (preElements.length > 0) {
                                return Array.from(preElements).map(el => el.innerText).join('\\n');
                            }
                            return document.body.innerText || '';
                        }
                    """)
                    if page_text.strip():
                        if a11y_tree.strip():
                            a11y_tree += "\n\n--- Page Text Content ---\n" + page_text
                        else:
                            a11y_tree = page_text
                except Exception:
                    pass

            return html, a11y_tree
        finally:
            await context.close()

    def _format_accessibility_tree(self, node: dict, indent: int = 0) -> str:
        """Format accessibility tree node recursively."""
        if not node:
            return ""

        lines = []
        prefix = "\t" * indent

        role = node.get("role", "")
        name = node.get("name", "")
        value = node.get("value", "")

        parts = [role]
        if name:
            parts.append(f'"{name}"')
        if value:
            parts.append(f'value="{value}"')

        lines.append(f"{prefix}{' '.join(parts)}")

        children = node.get("children", [])
        for child in children:
            lines.append(self._format_accessibility_tree(child, indent + 1))

        return "\n".join(lines)

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
