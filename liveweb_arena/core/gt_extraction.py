"""
Ground Truth Extraction from Page Content

Extracts GT data from accessibility tree content that agent actually sees.
This ensures GT matches what the agent could observe, eliminating data source mismatches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from liveweb_arena.core.value_formatter import ValueFormatter


@dataclass
class ExtractedData:
    """Data extracted from a single page visit"""
    url: str
    page_type: str  # "homepage", "detail", "search"
    data: Dict[str, Any]  # asset_id -> {field: value}
    timestamp: float = 0.0


class PageExtractor(ABC):
    """Base class for page content extractors"""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Source identifier (e.g., 'coingecko', 'stooq')"""
        pass

    @abstractmethod
    def matches_url(self, url: str) -> bool:
        """Check if this extractor handles the URL"""
        pass

    @abstractmethod
    def classify_page(self, url: str) -> str:
        """Classify page type: 'homepage', 'detail', 'search', 'other'"""
        pass

    @abstractmethod
    def extract(self, url: str, content: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract data from page content (accessibility tree).

        Returns:
            Dict mapping asset_id -> {field: value}
            e.g., {"bitcoin": {"price": 100000, "change_24h": 2.5}}
        """
        pass


# Lazy-loaded registry of extractors from plugins
_EXTRACTORS: Optional[List[PageExtractor]] = None


def _load_extractors() -> List[PageExtractor]:
    """Load extractors from plugin directories"""
    extractors = []

    # Import extractors from plugins
    try:
        from liveweb_arena.plugins.coingecko.extractor import CoinGeckoExtractor
        extractors.append(CoinGeckoExtractor())
    except ImportError:
        pass

    try:
        from liveweb_arena.plugins.stooq.extractor import StooqExtractor
        extractors.append(StooqExtractor())
    except ImportError:
        pass

    try:
        from liveweb_arena.plugins.weather.extractor import WeatherExtractor
        extractors.append(WeatherExtractor())
    except ImportError:
        pass

    try:
        from liveweb_arena.plugins.taostats.extractor import TaostatsExtractor
        extractors.append(TaostatsExtractor())
    except ImportError:
        pass

    return extractors


def get_extractor(url: str) -> Optional[PageExtractor]:
    """Get appropriate extractor for URL"""
    global _EXTRACTORS
    if _EXTRACTORS is None:
        _EXTRACTORS = _load_extractors()

    for extractor in _EXTRACTORS:
        if extractor.matches_url(url):
            return extractor
    return None


def extract_from_page(url: str, content: str) -> ExtractedData:
    """
    Extract GT data from a page.

    Args:
        url: Page URL
        content: Accessibility tree content

    Returns:
        ExtractedData with extracted values
    """
    import time

    extractor = get_extractor(url)
    if not extractor:
        return ExtractedData(url=url, page_type="other", data={})

    page_type = extractor.classify_page(url)
    data = extractor.extract(url, content)

    return ExtractedData(
        url=url,
        page_type=page_type,
        data=data,
        timestamp=time.time(),
    )


@dataclass
class GTExtractionState:
    """Tracks extracted GT data across page visits"""
    extractions: List[ExtractedData] = field(default_factory=list)

    def add_extraction(self, extraction: ExtractedData):
        """Add a new extraction result"""
        if extraction.data:
            self.extractions.append(extraction)

    def get_merged_data(self, required_fields: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Merge all extractions with priority rules:
        1. Detail pages override homepage data
        2. Later extractions override earlier ones (same page type)

        Args:
            required_fields: Optional list of required fields (e.g., ['change_24h'])

        Returns:
            Merged data: asset_id -> {field: value}
        """
        PAGE_PRIORITY = {
            "detail": 3,
            "search": 2,
            "homepage": 1,
            "other": 0,
        }

        merged = {}

        # Sort by (priority, timestamp) - lower priority/earlier first, so later ones override
        sorted_extractions = sorted(
            self.extractions,
            key=lambda e: (PAGE_PRIORITY.get(e.page_type, 0), e.timestamp)
        )

        for extraction in sorted_extractions:
            for asset_id, asset_data in extraction.data.items():
                if asset_id not in merged:
                    merged[asset_id] = {}

                # Check if this is a detail page overriding unsigned homepage data
                if extraction.page_type == "detail":
                    # Remove unsigned flag if detail page provides signed data
                    if 'change_unsigned' in merged[asset_id]:
                        del merged[asset_id]['change_unsigned']

                merged[asset_id].update(asset_data)

        return merged

    def format_gt_value(self, value: Any, validation_info: Dict[str, Any]) -> str:
        """Format extracted value according to template expectations."""
        return ValueFormatter.format_from_validation_info(value, validation_info)

    def _normalize_location(self, location: str) -> str:
        """Normalize location string for matching."""
        # Convert to lowercase
        loc = location.lower()
        # Replace + and , with space
        loc = loc.replace('+', ' ').replace(',', ' ')
        # Remove extra spaces
        loc = ' '.join(loc.split())
        return loc

    def _find_matching_asset(self, merged: Dict, target_id: str) -> Optional[str]:
        """Find matching asset_id with flexible matching for locations."""
        # Direct match
        if target_id in merged:
            return target_id

        # Normalize target for comparison
        target_normalized = self._normalize_location(target_id)
        target_parts = target_normalized.split()

        # Try to find a partial match (for locations like "Cape Town" vs "Cape+Town,South+Africa")
        for asset_id in merged.keys():
            asset_normalized = self._normalize_location(asset_id)

            # Check if asset contains target or vice versa
            if target_normalized in asset_normalized or asset_normalized in target_normalized:
                return asset_id

            # Check if first part (city name) matches
            if target_parts and target_parts[0] in asset_normalized:
                return asset_id

            # Check if asset's first part matches target
            asset_parts = asset_normalized.split()
            if asset_parts and asset_parts[0] in target_normalized:
                return asset_id

        return None

    def get_gt_for_template(
        self,
        validation_info: Dict[str, Any],
    ) -> Optional[str]:
        """
        Get formatted GT value for a template based on validation_info.

        This is the main entry point for getting GT from page extractions.

        Args:
            validation_info: Template's validation_info dict containing:
                - coin_id/symbol/asset_id: Asset identifier
                - metric_type/metric: What metric to retrieve
                - Other formatting hints

        Returns:
            Formatted GT string or None if not available
        """
        merged = self.get_merged_data()

        # Determine asset_id from various possible fields
        asset_id = (
            validation_info.get("coin_id") or
            validation_info.get("symbol") or
            validation_info.get("asset_id") or
            validation_info.get("location")  # For weather
        )

        # Special handling for taostats network metrics
        metric = validation_info.get("metric", "")
        if metric in ("subnet_count", "current_block"):
            # Look in "taostats" key for network-level data
            asset_id = "taostats"

        if not asset_id:
            return None

        # Try to find matching asset with flexible matching (for locations)
        matched_id = self._find_matching_asset(merged, asset_id)
        if not matched_id:
            return None

        asset_data = merged.get(matched_id, {})
        if not asset_data:
            return None

        # Determine which field to retrieve
        metric_type = validation_info.get("metric_type", "")
        api_field = validation_info.get("api_field", "")

        # Map metric_type/metric/api_field to internal field names
        field_map = {
            # CoinGecko
            "change_24h": "change_24h",
            "current_price": "current_price",
            "market_cap": "market_cap",
            # Stooq
            "last_price": "last_price",
            "change_percent": "daily_change_pct",
            "change_absolute": "daily_change",
            "open": "open",
            "high": "high",
            "low": "low",
            # Taostats
            "subnet_count": "subnet_count",
            "current_block": "current_block",
            # Weather - api_field names to internal field names
            "tempC": "temperature",
            "temperature": "temperature",
            "humidity": "humidity",
            "windspeedKmph": "wind_speed",
            "wind_speed": "wind_speed",
            "FeelsLikeC": "feels_like",
            "feels_like": "feels_like",
            "chanceofrain": "precipitation_chance",
            "maxtempC": "temperature_high",
            "mintempC": "temperature_low",
            "uvIndex": "uv_index",
            "cloudcover": "cloud_cover",
        }

        # Try metric_type first, then metric, then api_field
        field = field_map.get(metric_type) or field_map.get(metric) or api_field or metric_type or metric

        # Also try direct field names
        value = asset_data.get(field)

        # Fallback to common aliases
        if value is None:
            if field in ("last_price", "current_price"):
                value = asset_data.get("last_price") or asset_data.get("current_price")
            elif field in ("daily_change_pct", "change_24h"):
                value = asset_data.get("daily_change_pct") or asset_data.get("change_24h")

        if value is None:
            return None

        # Check for unsigned data that can't be used
        if field in ("change_24h", "daily_change_pct") and asset_data.get('change_unsigned'):
            return None

        # Format the value
        return self.format_gt_value(value, validation_info)

    def get_gt_value(
        self,
        asset_id: str,
        field: str,
        metric_type: str = None,
    ) -> Optional[Any]:
        """
        Get GT value for a specific asset and field.

        Args:
            asset_id: Asset identifier (e.g., 'bitcoin', 'aapl.us')
            field: Field name (e.g., 'change_24h', 'current_price')
            metric_type: Optional metric type for formatting

        Returns:
            Formatted GT value or None if not available
        """
        merged = self.get_merged_data()
        asset_data = merged.get(asset_id, {})

        value = asset_data.get(field)
        if value is None:
            return None

        # Check if data is unsigned (from homepage without explicit sign)
        if asset_data.get('change_unsigned') and field == 'change_24h':
            return None

        # Use unified formatter
        return ValueFormatter.format_metric(value, metric_type=metric_type or field)

    def is_complete(self, asset_id: str, required_fields: List[str]) -> bool:
        """Check if all required fields are available for an asset"""
        merged = self.get_merged_data()
        asset_data = merged.get(asset_id, {})

        for field in required_fields:
            if field not in asset_data:
                return False
            # Check for unsigned data that can't be used
            if field == 'change_24h' and asset_data.get('change_unsigned'):
                return False

        return True

    def get_missing_reason(self, asset_id: str, required_fields: List[str]) -> str:
        """Get reason why data is incomplete"""
        merged = self.get_merged_data()
        asset_data = merged.get(asset_id, {})

        if not asset_data:
            return f"No page visited for {asset_id}"

        missing = []
        for field in required_fields:
            if field not in asset_data:
                missing.append(field)
            elif field == 'change_24h' and asset_data.get('change_unsigned'):
                missing.append(f"{field} (sign unknown - only homepage visited)")

        if missing:
            return f"Missing data: {', '.join(missing)}"

        return "Data complete"

    def get_extraction_failure_reason(self, validation_info: Dict[str, Any]) -> str:
        """
        Get detailed reason why GT extraction failed for a template.

        Args:
            validation_info: Template's validation_info dict

        Returns:
            Human-readable failure reason
        """
        merged = self.get_merged_data()

        # Determine asset_id
        asset_id = (
            validation_info.get("coin_id") or
            validation_info.get("symbol") or
            validation_info.get("asset_id") or
            validation_info.get("location")
        )

        metric = validation_info.get("metric", "")
        if metric in ("subnet_count", "current_block"):
            asset_id = "taostats"

        if not asset_id:
            return "No asset identifier in validation_info"

        # Try flexible matching
        matched_id = self._find_matching_asset(merged, asset_id)
        if not matched_id:
            available = list(merged.keys())[:5]  # Show first 5
            return f"No page visited for {asset_id}. Available: {available}"

        asset_data = merged.get(matched_id, {})

        # Determine expected field
        metric_type = validation_info.get("metric_type", "")
        field_map = {
            "change_24h": "change_24h",
            "current_price": "current_price",
            "market_cap": "market_cap",
            "last_price": "last_price",
            "change_percent": "daily_change_pct",
            "subnet_count": "subnet_count",
            "current_block": "current_block",
        }
        field = field_map.get(metric_type) or field_map.get(metric) or metric_type or metric

        if field not in asset_data:
            available = list(asset_data.keys())
            return f"Field '{field}' not found. Available: {available}"

        if field in ("change_24h", "daily_change_pct") and asset_data.get('change_unsigned'):
            return f"Change sign unknown (only homepage visited, need detail page)"

        return "Unknown extraction failure"
