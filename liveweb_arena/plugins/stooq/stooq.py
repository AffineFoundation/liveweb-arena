"""
Stooq Plugin.

Plugin for financial market data from stooq.com.
"""

import asyncio
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

from liveweb_arena.plugins.base import BasePlugin
from liveweb_arena.plugins.base_client import APIFetchError
from liveweb_arena.utils.logger import log
from .api_client import fetch_single_asset_data, fetch_homepage_api_data, get_last_failure_metadata, initialize_cache


class StooqPlugin(BasePlugin):
    """
    Stooq plugin for financial market data.

    Handles pages like:
    - https://stooq.com/ (homepage - all assets)
    - https://stooq.com/q/?s=aapl.us (stocks)
    - https://stooq.com/q/?s=^spx (indices)
    - https://stooq.com/q/?s=gc.c (commodities)
    - https://stooq.com/q/?s=eurusd (forex)

    API data includes: open, high, low, close, volume, daily_change_pct, etc.
    """

    name = "stooq"
    _known_symbols_cache = None
    _quote_warmup_started = False
    _quote_warmup_lock = threading.Lock()

    allowed_domains = [
        "stooq.com",
        "www.stooq.com",
    ]

    def initialize(self):
        """Pre-warm homepage file cache and a few common quote pages."""
        initialize_cache()
        self._initialize_quote_warmup()

    def _initialize_quote_warmup(self) -> None:
        if os.environ.get("LIVEWEB_DISABLE_STOOQ_WARMUP", "").lower() in {"1", "true"}:
            return

        with StooqPlugin._quote_warmup_lock:
            if StooqPlugin._quote_warmup_started:
                return
            StooqPlugin._quote_warmup_started = True

        cache_dir = Path(
            os.environ.get(
                "LIVEWEB_CACHE_DIR",
                str(Path(__file__).resolve().parents[3] / ".cache" / "liveweb"),
            )
        )
        symbols_raw = os.environ.get("LIVEWEB_STOOQ_WARMUP_SYMBOLS", "jnj.us,aapl.us,^spx")
        urls = [
            f"https://stooq.com/q/?s={symbol.strip()}"
            for symbol in symbols_raw.split(",")
            if symbol.strip()
        ]
        if not urls:
            return

        async def _warm() -> None:
            from liveweb_arena.core.cache import CacheManager, PageRequirement

            mgr = CacheManager(cache_dir=cache_dir)
            pending = []
            try:
                for url in urls:
                    cached = mgr.get_cached(url)
                    if cached and not cached.is_expired(mgr.ttl) and cached.is_complete():
                        continue
                    pending.append(PageRequirement.data(url))
                if pending:
                    try:
                        await mgr.ensure_cached(pending, self)
                    except Exception as exc:
                        log("Stooq", f"Quote warmup skipped after fetch error: {exc}", force=True)
            finally:
                await mgr.shutdown()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(_warm())
            except Exception as exc:
                log("Stooq", f"Quote warmup skipped after init error: {exc}", force=True)
            return

        def _run_background() -> None:
            try:
                asyncio.run(_warm())
            except Exception as exc:
                log("Stooq", f"Quote warmup background task failed: {exc}", force=True)

        threading.Thread(
            target=_run_background,
            name="stooq-quote-warmup",
            daemon=True,
        ).start()

    def get_blocked_patterns(self) -> List[str]:
        """Block direct CSV download and ads."""
        return [
            "*/q/d/l/*",  # CSV download endpoint
            "*stooq.com/ads/*",  # Ad frames
        ]

    def get_stable_url_patterns(self) -> List[str]:
        return [
            "/",
            "/q/",
            "/q/?s=",
            "/q/i/?s=",
            "/q/d/?s=",
        ]

    def classify_url(self, url: str) -> Optional[str]:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path = parsed.path.lower()
        query = parsed.query.lower()
        if "stooq.com" not in host:
            return None
        if host.startswith("www."):
            return "env_tls_error"
        if (
            "/q/conv/" in path
            or "/s/mst/" in path
            or "quote.php" in path
            or "/q/plus/" in path
            or path.startswith("/q/l/")
            or path.startswith("/q/d/l/")
            or path.startswith("/q/nl/")
            or path.startswith("/t/")
        ):
            return "model_invalid_url_shape"
        if "q=" in query and "s=" not in query and "e=" not in query:
            return "model_invalid_url_shape"
        if path.startswith("/q/s/"):
            return "model_invalid_url_shape"
        return None

    def get_synthetic_page(self, url: str) -> Optional[str]:
        """Return synthetic error page for unknown symbols (zero network requests)."""
        symbol = self._extract_symbol(url)
        if symbol and symbol not in self._get_known_symbols():
            return (
                "<html><body>"
                "<h1>The page you requested does not exist</h1>"
                "<p>or has been moved</p>"
                f"<p>Symbol: {symbol}</p>"
                "</body></html>"
            )
        return None

    def _get_known_symbols(self) -> set:
        """All symbols defined in templates (cached at class level).

        Includes bare forms (e.g. 'xom') alongside suffixed forms ('xom.us')
        because agents commonly navigate to URLs like ?s=XOM without suffix.
        """
        if StooqPlugin._known_symbols_cache is None:
            from .templates.variables import US_STOCKS, INDICES, CURRENCIES, COMMODITIES
            from .templates.sector_analysis import ALL_STOCKS, ALL_INDICES
            symbols = set()
            for src in (
                [s.symbol for s in US_STOCKS],
                [s.symbol for s in INDICES],
                [s.symbol for s in CURRENCIES],
                [s.symbol for s in COMMODITIES],
                [sym for sym, _ in ALL_STOCKS],
                [sym for sym, _ in ALL_INDICES],
            ):
                for sym in src:
                    symbols.add(sym)
                    # Add bare form: 'xom.us' → also add 'xom'
                    bare = sym.split(".")[0] if "." in sym else None
                    if bare:
                        symbols.add(bare)
            StooqPlugin._known_symbols_cache = symbols
        return StooqPlugin._known_symbols_cache

    async def fetch_api_data(self, url: str) -> Dict[str, Any]:
        """
        Fetch API data for a Stooq page.

        - Homepage: Returns all assets in {"assets": {...}} format
        - Detail page (known symbol): Returns single asset data
        - Detail page (unknown symbol): Returns {} — pure navigation

        Args:
            url: Page URL

        Returns:
            API data appropriate for the page type
        """
        symbol = self._extract_symbol(url)
        if symbol:
            if symbol not in self._get_known_symbols():
                return {}  # Unknown symbol — skip API, zero requests
            try:
                return await fetch_single_asset_data(symbol)
            except APIFetchError as exc:
                metadata = dict(exc.metadata or {})
                metadata.setdefault("plugin", self.name)
                metadata.setdefault("symbol", symbol)
                metadata.setdefault("page_url", url)
                raise APIFetchError(
                    str(exc),
                    source=exc.source or self.name,
                    status_code=exc.status_code,
                    metadata=metadata or get_last_failure_metadata(),
                ) from exc

        # Homepage - return all assets
        if self._is_homepage(url):
            return await fetch_homepage_api_data()

        return {}

    def _is_homepage(self, url: str) -> bool:
        """Check if URL is the Stooq homepage."""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        # Homepage has no path or just "/"
        return path == '' and not parsed.query

    def needs_api_data(self, url: str) -> bool:
        """
        Determine if this URL needs API data for ground truth.

        Only homepage and known-symbol detail pages provide API data.
        Unknown symbols are treated as pure navigation (no API call).

        Args:
            url: Page URL

        Returns:
            True if API data is needed and available, False otherwise
        """
        symbol = self._extract_symbol(url)
        if symbol:
            return symbol in self._get_known_symbols()
        if self._is_homepage(url):
            return True
        return False

    def _extract_symbol(self, url: str) -> str:
        """
        Extract symbol from Stooq URL.

        Examples:
            https://stooq.com/q/?s=aapl.us -> aapl.us
            https://stooq.com/q/?s=^spx -> ^spx
            https://stooq.com/q/d/?s=gc.c -> gc.c
            http://stooq.com/q/s/?e=abbv&t= -> abbv (redirected URL format)
        """
        parsed = urlparse(url)
        query = parse_qs(parsed.query)

        # Check for 's' parameter (original format)
        if "s" in query:
            return query["s"][0].lower()

        # Check for 'e' parameter (redirected URL format: /q/s/?e=symbol&t=)
        if "e" in query:
            return query["e"][0].lower()

        return ""

    def is_plausible_asset_id(self, url: str) -> bool:
        symbol = self._extract_symbol(url)
        if not symbol:
            return True
        return symbol in self._get_known_symbols()
