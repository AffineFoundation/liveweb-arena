"""Unified value formatting for GT and validation."""

from typing import Any, Optional


class ValueFormatter:
    """Centralized value formatting for prices, percentages, and other metrics."""

    @staticmethod
    def format_price(value: float, currency: str = "$") -> str:
        """Format price with appropriate decimal places."""
        if value >= 1:
            return f"{currency}{value:,.2f}"
        elif value >= 0.01:
            return f"{currency}{value:.4f}"
        elif value >= 0.0001:
            return f"{currency}{value:.6f}"
        else:
            return f"{currency}{value:.10f}"

    @staticmethod
    def format_percentage(value: float, include_sign: bool = True) -> str:
        """Format percentage with sign."""
        if include_sign:
            sign = "+" if value >= 0 else ""
            return f"{sign}{value:.2f}%"
        return f"{value:.2f}%"

    @staticmethod
    def format_large_number(value: float, currency: str = "$") -> str:
        """Format large numbers with magnitude (billion/trillion)."""
        if value >= 1e12:
            return f"{currency}{value/1e12:.2f} trillion"
        elif value >= 1e9:
            return f"{currency}{value/1e9:.2f} billion"
        elif value >= 1e6:
            return f"{currency}{value/1e6:.2f} million"
        else:
            return f"{currency}{value:,.0f}"

    @staticmethod
    def format_with_unit(value: Any, unit: str) -> str:
        """Format value with unit suffix."""
        if isinstance(value, float):
            return f"{int(value)}{unit}"
        return f"{value}{unit}"

    @classmethod
    def format_metric(
        cls,
        value: Any,
        metric_type: str = "",
        metric: str = "",
        api_field: str = "",
        unit: str = "",
        is_percentage: bool = False,
    ) -> str:
        """
        Format value based on metric type hints.

        Args:
            value: Raw value to format
            metric_type: High-level metric type (e.g., 'current_price', 'change_24h')
            metric: Specific metric name (e.g., 'last_price', 'open')
            api_field: API field name (e.g., 'tempC', 'windspeedKmph')
            unit: Explicit unit to append
            is_percentage: Force percentage formatting

        Returns:
            Formatted string
        """
        if value is None:
            return ""

        # Percentage values
        if is_percentage or metric_type in ("change_24h", "change_percent", "daily_change_pct"):
            if isinstance(value, (int, float)):
                return cls.format_percentage(value)

        # Price values
        if metric_type == "current_price" or metric == "last_price":
            if isinstance(value, (int, float)):
                return cls.format_price(value)

        # Market cap (large numbers)
        if metric_type == "market_cap":
            if isinstance(value, (int, float)):
                return cls.format_large_number(value)

        # Stooq metrics
        if metric in ("last_price", "open", "high", "low"):
            if isinstance(value, (int, float)):
                return f"{value:.2f}"

        if metric == "change_absolute":
            if isinstance(value, (int, float)):
                return f"{value:+.2f}"

        # Explicit unit
        if unit:
            return cls.format_with_unit(value, unit)

        # Weather-specific fields
        weather_units = {
            "tempC": "°C",
            "FeelsLikeC": "°C",
            "maxtempC": "°C",
            "mintempC": "°C",
            "humidity": "%",
            "windspeedKmph": " km/h",
            "chanceofrain": "%",
            "cloudcover": "%",
        }
        if api_field in weather_units:
            if isinstance(value, (int, float)):
                return f"{int(value)}{weather_units[api_field]}"

        # Integer values
        if isinstance(value, int):
            return str(value)

        # Default
        return str(value) if value is not None else ""

    @classmethod
    def format_from_validation_info(cls, value: Any, validation_info: dict) -> str:
        """Format value using validation_info dict."""
        return cls.format_metric(
            value,
            metric_type=validation_info.get("metric_type", ""),
            metric=validation_info.get("metric", ""),
            api_field=validation_info.get("api_field", ""),
            unit=validation_info.get("unit", ""),
            is_percentage=validation_info.get("is_percentage", False),
        )
