"""
Taostats Plugin.

Plugin for Bittensor network data from taostats.io.
Uses official taostats.io API for ground truth.
"""

import time
import re
from typing import Any, Dict, List
from urllib.parse import urlparse

from liveweb_arena.core.cache import normalize_url
from liveweb_arena.plugins.base import BasePlugin
from .api_client import fetch_single_subnet_data, fetch_homepage_api_data, initialize_cache


class TaostatsPrefetchSetupError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        prefetch_phase: str,
        wait_target: str | None = None,
        evidence: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.prefetch_phase = prefetch_phase
        self.wait_target = wait_target
        self.evidence = dict(evidence or {})


_LIST_SHOW_ALL_COOLDOWN_S = 180
_list_show_all_cooldown_until: dict[str, float] = {}


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

    async def _wait_for_minimum_body_text(self, page, *, min_chars: int, timeout_ms: int) -> None:
        await page.wait_for_function(
            """minChars => {
                const text = (document.body?.innerText || '').trim();
                return text.length >= minChars;
            }""",
            min_chars,
            timeout=timeout_ms,
        )

    async def _capture_body_snapshot(self, page) -> Dict[str, Any]:
        snapshot = await page.evaluate(
            """() => {
                const text = (document.body?.innerText || '').trim();
                const title = document.title || '';
                const hasMain = !!document.querySelector('main');
                const hasTable = !!document.querySelector('table, [role="table"], .table, .rt-table');
                const textSample = text.slice(0, 2000);
                return {
                    title,
                    textLength: text.length,
                    textSample,
                    hasMain,
                    hasTable,
                };
            }"""
        )
        return dict(snapshot or {})

    def _is_list_body_ready(self, snapshot: Dict[str, Any]) -> bool:
        sample = str(snapshot.get("textSample") or "").lower()
        text_length = int(snapshot.get("textLength") or 0)
        return bool(
            text_length >= 120
            and (
                snapshot.get("hasTable")
                or snapshot.get("hasMain")
                or "subnet" in sample
                or "subnets" in sample
                or "netuid" in sample
            )
        )

    def _is_detail_body_ready(self, snapshot: Dict[str, Any]) -> bool:
        sample = str(snapshot.get("textSample") or "").lower()
        text_length = int(snapshot.get("textLength") or 0)
        return bool(
            text_length >= 120
            and (
                snapshot.get("hasMain")
                or "statistics" in sample
                or "subnet" in sample
                or "netuid" in sample
                or "emission" in sample
                or "tempo" in sample
            )
        )

    def _list_show_all_cooldown_active(self, url: str) -> bool:
        key = normalize_url(url)
        until = _list_show_all_cooldown_until.get(key)
        if until is None:
            return False
        if until > time.monotonic():
            return True
        _list_show_all_cooldown_until.pop(key, None)
        return False

    def _activate_list_show_all_cooldown(self, url: str) -> None:
        _list_show_all_cooldown_until[normalize_url(url)] = time.monotonic() + _LIST_SHOW_ALL_COOLDOWN_S

    async def _best_effort_expand_list(self, page, url: str) -> Dict[str, Any]:
        if self._list_show_all_cooldown_active(url):
            return {
                "interaction_kind": "show_all",
                "target_locator": None,
                "cooldown_active": True,
                "expanded": False,
            }

        attempts: list[tuple[str, Any]] = [
            ("text=ALL", page.get_by_text("ALL", exact=True).first),
            ("role=button name=ALL", page.get_by_role("button", name="ALL").first),
        ]

        last_error: Exception | None = None
        last_locator: str | None = None
        for locator_name, handle in attempts:
            last_locator = locator_name
            try:
                if await handle.is_visible(timeout=2500):
                    await handle.click(timeout=2500)
                    await page.wait_for_timeout(1200)
                    try:
                        await page.wait_for_load_state("networkidle", timeout=5000)
                    except Exception:
                        pass
                    return {
                        "interaction_kind": "show_all",
                        "target_locator": locator_name,
                        "cooldown_active": False,
                        "expanded": True,
                    }
            except Exception as exc:
                last_error = exc

        select_locator = "select"
        try:
            await page.locator(select_locator).first.select_option(label="ALL", timeout=2500)
            await page.wait_for_timeout(1200)
            return {
                "interaction_kind": "show_all",
                "target_locator": select_locator,
                "cooldown_active": False,
                "expanded": True,
            }
        except Exception as exc:
            last_error = exc

        try:
            await page.locator(select_locator).first.select_option(value="-1", timeout=2500)
            await page.wait_for_timeout(1200)
            return {
                "interaction_kind": "show_all",
                "target_locator": select_locator,
                "cooldown_active": False,
                "expanded": True,
            }
        except Exception as exc:
            last_error = exc

        self._activate_list_show_all_cooldown(url)
        return {
            "interaction_kind": "show_all",
            "target_locator": last_locator or select_locator,
            "cooldown_active": False,
            "expanded": False,
            "cooldown_applied": True,
            "raw_exception_type": type(last_error).__name__ if last_error is not None else None,
            "raw_exception_message": str(last_error) if last_error is not None else None,
        }

    async def setup_page_for_cache(self, page, url: str) -> Dict[str, Any] | None:
        """
        Setup page before caching - click "ALL" to show all subnets.

        On taostats.io/subnets, the default view shows only 10-25 rows.
        Click "ALL" to show all ~128 subnets for complete visibility.
        """
        if self._is_list_page(url):
            metadata: Dict[str, Any] = {
                "page_kind": "taostats_list",
                "interaction_kind": "show_all",
                "selector_syntax_invalid": False,
            }
            try:
                await self._wait_for_minimum_body_text(page, min_chars=120, timeout_ms=8000)
            except Exception:
                metadata["body_wait_timed_out"] = True

            snapshot = await self._capture_body_snapshot(page)
            metadata["page_body_ready"] = self._is_list_body_ready(snapshot)
            if metadata["page_body_ready"]:
                expand_result = await self._best_effort_expand_list(page, url)
                metadata.update(expand_result)
                if not expand_result.get("expanded"):
                    metadata["list_setup_soft_failed"] = True
            else:
                metadata["list_setup_soft_failed"] = True
            return metadata

        if self._extract_subnet_id(url):
            try:
                await self._wait_for_minimum_body_text(page, min_chars=140, timeout_ms=10000)
            except Exception:
                pass

            snapshot = await self._capture_body_snapshot(page)
            page_body_ready = self._is_detail_body_ready(snapshot)
            if not page_body_ready:
                raise TaostatsPrefetchSetupError(
                    "Taostats detail body did not become ready",
                    prefetch_phase="setup_page_for_cache",
                    wait_target="detail_body_ready",
                    evidence={
                        "page_kind": "taostats_detail",
                        "page_body_ready": False,
                        "detail_setup_soft_failed": False,
                    },
                )

            soft_failure: dict[str, Any] | None = None
            for selector in (
                "text=Subnet",
                "text=Netuid",
                "text=Statistics",
                "text=Transactions",
                "text=Holders",
                "text=Price Impact",
            ):
                try:
                    await page.locator(selector).first.wait_for(timeout=8000)
                    soft_failure = None
                    break
                except Exception as exc:
                    soft_failure = {
                        "page_kind": "taostats_detail",
                        "page_body_ready": True,
                        "detail_setup_soft_failed": True,
                        "prefetch_phase": "setup_page_for_cache",
                        "wait_target": selector,
                        "raw_exception_type": type(exc).__name__,
                        "raw_exception_message": str(exc),
                    }
                    continue

            await page.wait_for_timeout(1200)
            return soft_failure or {
                "page_kind": "taostats_detail",
                "page_body_ready": True,
                "detail_setup_soft_failed": False,
                "prefetch_phase": "setup_page_for_cache",
            }
