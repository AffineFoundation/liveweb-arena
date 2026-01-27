"""
Unified Ground Truth Collection System

This module provides a unified GT collection system that:
1. Allows templates to declare their GT source type (PAGE_ONLY, API_ONLY, HYBRID)
2. Collects GT data in real-time during page visits
3. Supports API-based GT fetching for templates that require it
4. Provides a unified interface for GT retrieval

Design principles:
- GT source type is a design-time declaration, not a runtime fallback
- PAGE_ONLY is the default (ensures data consistency)
- HYBRID allows page extraction with API supplementation
- API_ONLY is for complex aggregations or SDK calls
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from liveweb_arena.core.gt_extraction import (
    GTExtractionState,
    extract_from_page,
    _load_extractors,
)
from liveweb_arena.core.ground_truth_trigger import UrlPatternTrigger, TriggerConfig
from liveweb_arena.utils.logger import log

if TYPE_CHECKING:
    from liveweb_arena.core.task_manager import SubTask

logger = logging.getLogger(__name__)

# Global reference for hybrid utils to access collected API data
_current_gt_collector: Optional["GTCollector"] = None


def get_current_gt_collector() -> Optional["GTCollector"]:
    """Get the current GTCollector instance."""
    return _current_gt_collector


def set_current_gt_collector(collector: Optional["GTCollector"]):
    """Set the current GTCollector instance."""
    global _current_gt_collector
    _current_gt_collector = collector


class GTSourceType(Enum):
    """
    Ground truth source type declaration.

    Templates declare their GT source type to specify where GT data should come from:
    - PAGE_ONLY: GT is extracted from page content (accessibility tree)
    - API_ONLY: GT is fetched from API (for complex aggregations, SDK calls)
    - HYBRID: GT primarily from page, with API supplementation for specific fields
    """
    PAGE_ONLY = "page_only"
    API_ONLY = "api_only"
    HYBRID = "hybrid"


@dataclass
class GTResult:
    """Result of GT collection for a single subtask."""
    tag: str
    source_type: GTSourceType
    value: Optional[str] = None
    page_data: Optional[Dict[str, Any]] = None
    api_data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    @property
    def success(self) -> bool:
        return self.value is not None

    @property
    def formatted_value(self) -> Optional[str]:
        """Get the formatted GT value ready for validation."""
        return self.value


class GTCollector:
    """
    Unified GT collector that manages GT collection for all subtasks.

    This replaces the old GroundTruthManager and provides:
    1. Real-time page extraction during navigation
    2. API-based GT fetching for API_ONLY/HYBRID templates
    3. Unified GT state management
    4. Clear failure reporting
    """

    def __init__(self, subtasks: List["SubTask"], task_manager=None):
        """
        Initialize GT collector.

        Args:
            subtasks: List of subtasks to collect GT for
            task_manager: TaskManager for accessing plugins
        """
        self.subtasks = subtasks
        self._task_manager = task_manager
        self._page_extractors = _load_extractors()

        # GT extraction state for page-based collection
        self._extraction_state = GTExtractionState()

        # API fetch results per subtask
        self._api_results: Dict[str, Any] = {}

        # Track which subtasks have been processed
        self._processed: Dict[str, bool] = {}

        # Track visited URLs for each subtask
        self._visited_urls: Dict[str, List[str]] = {st.answer_tag: [] for st in subtasks}

        # Collected API data from page visits {asset_id: {field: value}}
        # This replaces the global API cache for HYBRID templates
        self._collected_api_data: Dict[str, Dict[str, Any]] = {}

    def _get_source_type(self, subtask: "SubTask") -> GTSourceType:
        """Get GT source type for a subtask."""
        if self._task_manager is None:
            return GTSourceType.PAGE_ONLY

        plugin = self._task_manager.get_plugin(subtask.plugin_name)
        if plugin is None:
            return GTSourceType.PAGE_ONLY

        # Check if plugin has get_gt_source method
        if hasattr(plugin, 'get_gt_source'):
            return plugin.get_gt_source(subtask.validation_info)

        return GTSourceType.PAGE_ONLY

    def _get_trigger_config(self, subtask: "SubTask") -> Optional["TriggerConfig"]:
        """Get trigger configuration for a subtask."""
        if self._task_manager is None:
            return None

        plugin = self._task_manager.get_plugin(subtask.plugin_name)
        if plugin is None:
            return None

        return plugin.get_ground_truth_trigger(subtask.validation_info)

    def _get_trigger(self, subtask: "SubTask") -> Optional[UrlPatternTrigger]:
        """Get URL trigger for a subtask."""
        config = self._get_trigger_config(subtask)
        if config is None:
            return None
        return config.trigger

    def _should_trigger_api(self, url: str, subtask: "SubTask") -> bool:
        """Check if URL should trigger API fetch for subtask."""
        trigger = self._get_trigger(subtask)
        if trigger is None:
            return False
        return trigger.matches(url)

    async def on_page_visit(
        self,
        url: str,
        content: str,
        api_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Handle page visit event - extract GT from page content and merge api_data.

        This is called during agent navigation to collect GT in real-time.

        Args:
            url: The URL being visited
            content: Accessibility tree content of the page
            api_data: Page-bound API data from cache (for HYBRID GT)
        """
        if not url or url == "about:blank":
            return

        log("GT", f"Page visit: {url[:60]}...")

        # Extract data from page
        extraction = extract_from_page(url, content)
        if extraction.data:
            self._extraction_state.add_extraction(extraction)
            # Log extracted fields for debugging
            for asset_id, fields in extraction.data.items():
                field_names = list(fields.keys())[:3]
                log("GT", f"  Extracted [{asset_id}]: {field_names}")
        else:
            log("GT", f"  No data extracted from {extraction.page_type} page")

        # Merge API data from page cache (for HYBRID templates)
        if api_data:
            self._merge_api_data(url, api_data)

        # Track visited URLs for each subtask
        # Note: API fetching is always deferred to end of trajectory
        # This ensures GT matches what agent sees on pages
        for subtask in self.subtasks:
            tag = subtask.answer_tag
            self._visited_urls[tag].append(url)

            source_type = self._get_source_type(subtask)

            # Log trigger matches for debugging, but don't fetch during navigation
            if source_type in (GTSourceType.HYBRID, GTSourceType.API_ONLY):
                if self._should_trigger_api(url, subtask):
                    log("GT", f"  Trigger matched for [{tag}] ({source_type.value} - deferred to end)")

    def _merge_api_data(self, url: str, api_data: Dict[str, Any]):
        """
        Merge API data from page cache into collected data.

        Later visits override earlier visits (detail page overrides homepage).

        Args:
            url: The URL that was visited
            api_data: API data bound to the page
        """
        url_lower = url.lower()

        if "coingecko.com" in url_lower:
            if "coins" in api_data:
                # Homepage format: {"coins": {"bitcoin": {...}, "ethereum": {...}}}
                for coin_id, data in api_data["coins"].items():
                    self._collected_api_data[coin_id] = data
                    log("GT", f"  Merged API [{coin_id}] from homepage")
            elif "id" in api_data:
                # Detail page format: {"id": "bitcoin", "current_price": ...}
                coin_id = api_data["id"]
                self._collected_api_data[coin_id] = api_data
                log("GT", f"  Merged API [{coin_id}] from detail page (override)")
            else:
                log("GT", f"  Unknown CoinGecko api_data format: {list(api_data.keys())[:5]}")

        elif "stooq.com" in url_lower:
            if "assets" in api_data:
                # Homepage format: {"assets": {"aapl.us": {...}, "gc.f": {...}}}
                for symbol, data in api_data["assets"].items():
                    self._collected_api_data[symbol] = data
                    log("GT", f"  Merged API [{symbol}] from homepage")
            elif "symbol" in api_data:
                # Detail page format: {"symbol": "aapl.us", "close": ...}
                symbol = api_data["symbol"]
                self._collected_api_data[symbol] = api_data
                log("GT", f"  Merged API [{symbol}] from detail page (override)")
            else:
                log("GT", f"  Unknown Stooq api_data format: {list(api_data.keys())[:5]}")

    def get_collected_api_data(self) -> Dict[str, Dict[str, Any]]:
        """Get all collected API data from page visits."""
        return self._collected_api_data

    async def _fetch_api_gt(self, subtask: "SubTask"):
        """Fetch GT from API for a subtask."""
        tag = subtask.answer_tag

        if self._task_manager is None:
            return

        plugin = self._task_manager.get_plugin(subtask.plugin_name)
        if plugin is None:
            return

        try:
            log("GT", f"Fetching API GT for [{tag}]...")
            result = await plugin.get_ground_truth(subtask.validation_info)

            # Handle GroundTruthResult or raw value
            from liveweb_arena.core.ground_truth_trigger import GroundTruthResult
            if isinstance(result, GroundTruthResult):
                if result.success:
                    self._api_results[tag] = result.value
                    log("GT", f"API GT [{tag}]: {str(result.value)[:50]}...")
                else:
                    log("GT", f"API GT [{tag}] failed: {result.error}")
            else:
                self._api_results[tag] = result
                log("GT", f"API GT [{tag}]: {str(result)[:50]}...")

        except Exception as e:
            logger.warning(f"API GT fetch failed for {tag}: {e}")

    async def fetch_remaining_api_gt(self):
        """
        Fetch API GT for subtasks that weren't triggered during navigation.

        Called at the end of agent trajectory for API_ONLY/HYBRID templates.
        """
        for subtask in self.subtasks:
            tag = subtask.answer_tag
            source_type = self._get_source_type(subtask)

            if source_type in (GTSourceType.API_ONLY, GTSourceType.HYBRID):
                if tag not in self._api_results:
                    await self._fetch_api_gt(subtask)

    def get_gt_for_subtask(self, subtask: "SubTask") -> Optional[str]:
        """
        Get formatted GT value for a subtask.

        This is the main entry point for GT retrieval. It returns the GT value
        based on the template's declared source type.

        Args:
            subtask: The subtask to get GT for

        Returns:
            Formatted GT string or None if unavailable
        """
        tag = subtask.answer_tag
        source_type = self._get_source_type(subtask)
        vi = subtask.validation_info

        if source_type == GTSourceType.PAGE_ONLY:
            # Only use page extraction
            return self._extraction_state.get_gt_for_template(vi)

        elif source_type == GTSourceType.API_ONLY:
            # Only use API result
            return self._api_results.get(tag)

        elif source_type == GTSourceType.HYBRID:
            # Try page extraction first, fall back to API
            page_gt = self._extraction_state.get_gt_for_template(vi)
            if page_gt is not None:
                return page_gt

            # Use API result as supplement
            return self._api_results.get(tag)

        return None

    def get_failure_reason(self, subtask: "SubTask") -> str:
        """
        Get detailed reason why GT collection failed for a subtask.

        Args:
            subtask: The subtask that failed

        Returns:
            Human-readable failure reason
        """
        tag = subtask.answer_tag
        source_type = self._get_source_type(subtask)
        vi = subtask.validation_info
        visited = self._visited_urls.get(tag, [])

        if source_type == GTSourceType.PAGE_ONLY:
            reason = self._extraction_state.get_extraction_failure_reason(vi)
            if visited:
                return f"{reason}. Visited: {visited[:3]}"
            return f"{reason}. No relevant pages visited."

        elif source_type == GTSourceType.API_ONLY:
            if tag in self._api_results:
                return "API returned invalid data"
            return "API was never triggered (agent didn't visit required pages)"

        elif source_type == GTSourceType.HYBRID:
            page_reason = self._extraction_state.get_extraction_failure_reason(vi)
            api_status = "fetched" if tag in self._api_results else "not fetched"
            return f"Page: {page_reason}. API: {api_status}"

        return "Unknown failure"

    def get_extraction_state(self) -> GTExtractionState:
        """Get the underlying extraction state for direct access."""
        return self._extraction_state

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        page_count = len(self._extraction_state.extractions)
        api_count = len(self._api_results)

        stats = {
            "total_subtasks": len(self.subtasks),
            "page_extractions": page_count,
            "api_fetches": api_count,
        }

        # Count by source type
        by_type = {t: 0 for t in GTSourceType}
        for subtask in self.subtasks:
            source_type = self._get_source_type(subtask)
            by_type[source_type] += 1

        stats["by_source_type"] = {t.value: c for t, c in by_type.items()}

        return stats
