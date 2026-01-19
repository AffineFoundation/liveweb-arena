"""Variable definitions for dynamic question generation"""

import random
from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from ....core.validators.base import Variable, VariableType


class LocationType(Enum):
    """Types of location specifications supported by wttr.in"""
    CITY_NAME = "city_name"
    AIRPORT_CODE = "airport_code"
    AREA_CODE = "area_code"
    GPS_COORDS = "gps_coords"
    LANDMARK = "landmark"


@dataclass
class LocationSpec:
    """Specification of a location with multiple representations"""
    location_type: LocationType
    value: Any  # The actual value (city name, coords tuple, etc.)
    display_name: str  # Human-readable name
    api_query: str  # wttr.in API query format


class LocationVariable(Variable):
    """
    Variable for location specifications.

    Supports multiple location types:
    - City names (from major world cities)
    - Airport codes (IATA 3-letter codes)
    - Area codes (US ZIP, postal codes)
    - GPS coordinates
    - Landmarks

    Uses rule-based generation rather than exhaustive enumeration.
    """

    # Major world cities grouped by region (not exhaustive, but diverse)
    # These serve as seeds for generating more locations dynamically
    CITY_SEEDS = {
        "asia": [
            ("Tokyo", "Japan"), ("Beijing", "China"), ("Seoul", "South Korea"),
            ("Mumbai", "India"), ("Singapore", "Singapore"), ("Bangkok", "Thailand"),
            ("Hong Kong", "China"), ("Shanghai", "China"), ("Delhi", "India"),
            ("Jakarta", "Indonesia"), ("Manila", "Philippines"), ("Osaka", "Japan"),
        ],
        "europe": [
            ("London", "UK"), ("Paris", "France"), ("Berlin", "Germany"),
            ("Madrid", "Spain"), ("Rome", "Italy"), ("Amsterdam", "Netherlands"),
            ("Vienna", "Austria"), ("Prague", "Czech Republic"), ("Stockholm", "Sweden"),
            ("Warsaw", "Poland"), ("Brussels", "Belgium"), ("Zurich", "Switzerland"),
        ],
        "americas": [
            ("New York", "USA"), ("Los Angeles", "USA"), ("Chicago", "USA"),
            ("Toronto", "Canada"), ("Mexico City", "Mexico"), ("Sao Paulo", "Brazil"),
            ("Buenos Aires", "Argentina"), ("Miami", "USA"), ("Seattle", "USA"),
            ("Vancouver", "Canada"), ("Houston", "USA"), ("Boston", "USA"),
        ],
        "oceania": [
            ("Sydney", "Australia"), ("Melbourne", "Australia"), ("Auckland", "New Zealand"),
            ("Brisbane", "Australia"), ("Perth", "Australia"), ("Wellington", "New Zealand"),
        ],
        "africa_middle_east": [
            ("Dubai", "UAE"), ("Cairo", "Egypt"), ("Johannesburg", "South Africa"),
            ("Tel Aviv", "Israel"), ("Istanbul", "Turkey"), ("Lagos", "Nigeria"),
            ("Casablanca", "Morocco"), ("Nairobi", "Kenya"),
        ],
    }

    # Major airport codes
    AIRPORT_CODES = [
        "JFK", "LAX", "LHR", "CDG", "FRA", "AMS", "DXB", "SIN",
        "HKG", "NRT", "ICN", "PEK", "SYD", "MEL", "YYZ", "ORD",
        "SFO", "MIA", "ATL", "DFW", "SEA", "BOS", "DEN", "PHX",
    ]

    # US major area codes (sample, not exhaustive)
    AREA_CODES_US = [
        "10001", "90210", "60601", "77001", "85001", "98101",
        "02101", "30301", "48201", "55401", "80201", "97201",
    ]

    # Famous landmarks
    LANDMARKS = [
        ("Eiffel Tower", "Paris"), ("Statue of Liberty", "New York"),
        ("Big Ben", "London"), ("Sydney Opera House", "Sydney"),
        ("Taj Mahal", "India"), ("Great Wall", "Beijing"),
        ("Colosseum", "Rome"), ("Golden Gate Bridge", "San Francisco"),
    ]

    def __init__(
        self,
        allowed_types: List[LocationType] = None,
        regions: List[str] = None,
    ):
        """
        Initialize location variable.

        Args:
            allowed_types: Which location types to use (default: all)
            regions: Which regions to sample from (default: all)
        """
        super().__init__("location", VariableType.LOCATION)
        self.allowed_types = allowed_types or list(LocationType)
        self.regions = regions or list(self.CITY_SEEDS.keys())

    def sample(self, rng: random.Random) -> LocationSpec:
        """Sample a location specification"""
        loc_type = rng.choice(self.allowed_types)

        if loc_type == LocationType.CITY_NAME:
            return self._sample_city(rng)
        elif loc_type == LocationType.AIRPORT_CODE:
            return self._sample_airport(rng)
        elif loc_type == LocationType.AREA_CODE:
            return self._sample_area_code(rng)
        elif loc_type == LocationType.GPS_COORDS:
            return self._sample_gps(rng)
        elif loc_type == LocationType.LANDMARK:
            return self._sample_landmark(rng)

    def _sample_city(self, rng: random.Random) -> LocationSpec:
        """Sample a city name"""
        region = rng.choice(self.regions)
        cities = self.CITY_SEEDS.get(region, self.CITY_SEEDS["americas"])
        city, country = rng.choice(cities)

        return LocationSpec(
            location_type=LocationType.CITY_NAME,
            value={"city": city, "country": country},
            display_name=f"{city}, {country}",
            api_query=f"{city},{country}".replace(" ", "+"),
        )

    def _sample_airport(self, rng: random.Random) -> LocationSpec:
        """Sample an airport code"""
        code = rng.choice(self.AIRPORT_CODES)
        return LocationSpec(
            location_type=LocationType.AIRPORT_CODE,
            value=code,
            display_name=f"{code} airport",
            api_query=code.lower(),
        )

    def _sample_area_code(self, rng: random.Random) -> LocationSpec:
        """Sample a US area code"""
        code = rng.choice(self.AREA_CODES_US)
        return LocationSpec(
            location_type=LocationType.AREA_CODE,
            value=code,
            display_name=f"area code {code}",
            api_query=code,
        )

    def _sample_gps(self, rng: random.Random) -> LocationSpec:
        """Sample GPS coordinates (within reasonable bounds)"""
        # Generate coordinates for populated areas
        lat = rng.uniform(-60, 70)  # Avoid extreme poles
        lon = rng.uniform(-180, 180)

        lat_str = f"{lat:.2f}"
        lon_str = f"{lon:.2f}"

        return LocationSpec(
            location_type=LocationType.GPS_COORDS,
            value={"lat": lat, "lon": lon},
            display_name=f"coordinates ({lat_str}, {lon_str})",
            api_query=f"{lat_str},{lon_str}",
        )

    def _sample_landmark(self, rng: random.Random) -> LocationSpec:
        """Sample a landmark"""
        landmark, location = rng.choice(self.LANDMARKS)
        return LocationSpec(
            location_type=LocationType.LANDMARK,
            value={"landmark": landmark, "location": location},
            display_name=landmark,
            api_query="~" + landmark.replace(" ", "+"),
        )

    def get_display_value(self, value: LocationSpec) -> str:
        """Get human-readable location name"""
        return value.display_name

    def get_api_value(self, value: LocationSpec) -> str:
        """Get API query string"""
        return value.api_query


class DateType(Enum):
    """Types of date specifications"""
    TODAY = "today"
    TOMORROW = "tomorrow"
    SPECIFIC_DATE = "specific_date"
    RELATIVE_WEEKDAY = "relative_weekday"
    DAYS_FROM_NOW = "days_from_now"


@dataclass
class DateSpec:
    """Specification of a date with multiple representations"""
    date_type: DateType
    value: Any  # The actual value
    display_text: str  # Human-readable text
    api_date: str  # API format (YYYY-MM-DD or offset)
    forecast_day: int  # 0=today, 1=tomorrow, etc.


class DateVariable(Variable):
    """
    Variable for date specifications.

    Supports:
    - Today/Tomorrow
    - Specific dates
    - Relative weekdays (this Monday, next Friday)
    - Days from now (in 3 days)
    """

    WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    WEEKDAYS_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

    def __init__(
        self,
        max_forecast_days: int = 7,
        allowed_types: List[DateType] = None,
        use_chinese: bool = False,
    ):
        """
        Initialize date variable.

        Args:
            max_forecast_days: Maximum days into the future (wttr.in supports up to 7)
            allowed_types: Which date types to use (default: all)
            use_chinese: Use Chinese weekday names
        """
        super().__init__("date", VariableType.DATE)
        self.max_forecast_days = min(max_forecast_days, 7)  # wttr.in limit
        self.allowed_types = allowed_types or list(DateType)
        self.use_chinese = use_chinese

    def sample(self, rng: random.Random) -> DateSpec:
        """Sample a date specification"""
        date_type = rng.choice(self.allowed_types)
        today = datetime.now().date()

        if date_type == DateType.TODAY:
            return DateSpec(
                date_type=DateType.TODAY,
                value=today,
                display_text="today" if not self.use_chinese else "今天",
                api_date=today.strftime("%Y-%m-%d"),
                forecast_day=0,
            )

        elif date_type == DateType.TOMORROW:
            tomorrow = today + timedelta(days=1)
            return DateSpec(
                date_type=DateType.TOMORROW,
                value=tomorrow,
                display_text="tomorrow" if not self.use_chinese else "明天",
                api_date=tomorrow.strftime("%Y-%m-%d"),
                forecast_day=1,
            )

        elif date_type == DateType.SPECIFIC_DATE:
            days_ahead = rng.randint(0, self.max_forecast_days)
            target_date = today + timedelta(days=days_ahead)
            return DateSpec(
                date_type=DateType.SPECIFIC_DATE,
                value=target_date,
                display_text=target_date.strftime("%B %d, %Y"),
                api_date=target_date.strftime("%Y-%m-%d"),
                forecast_day=days_ahead,
            )

        elif date_type == DateType.RELATIVE_WEEKDAY:
            return self._sample_relative_weekday(rng, today)

        elif date_type == DateType.DAYS_FROM_NOW:
            days = rng.randint(2, self.max_forecast_days)
            target_date = today + timedelta(days=days)
            return DateSpec(
                date_type=DateType.DAYS_FROM_NOW,
                value=target_date,
                display_text=f"in {days} days" if not self.use_chinese else f"{days}天后",
                api_date=target_date.strftime("%Y-%m-%d"),
                forecast_day=days,
            )

    def _sample_relative_weekday(self, rng: random.Random, today) -> DateSpec:
        """Sample a relative weekday (this/next Monday, etc.)"""
        current_weekday = today.weekday()
        target_weekday = rng.randint(0, 6)
        weekday_names = self.WEEKDAYS_ZH if self.use_chinese else self.WEEKDAYS

        # Decide if "this" or "next" week
        is_next_week = rng.choice([True, False])

        if is_next_week:
            days_ahead = (target_weekday - current_weekday + 7) % 7 + 7
            prefix = "next" if not self.use_chinese else "下"
        else:
            days_ahead = (target_weekday - current_weekday + 7) % 7
            if days_ahead == 0:
                days_ahead = 7  # Same weekday means next week
            prefix = "this" if not self.use_chinese else "这"

        # Clamp to forecast limit
        if days_ahead > self.max_forecast_days:
            days_ahead = self.max_forecast_days

        target_date = today + timedelta(days=days_ahead)

        return DateSpec(
            date_type=DateType.RELATIVE_WEEKDAY,
            value=target_date,
            display_text=f"{prefix} {weekday_names[target_weekday]}",
            api_date=target_date.strftime("%Y-%m-%d"),
            forecast_day=days_ahead,
        )

    def get_display_value(self, value: DateSpec) -> str:
        """Get human-readable date text"""
        return value.display_text

    def get_api_value(self, value: DateSpec) -> str:
        """Get API date format"""
        return value.api_date


class TimeOfDay(Enum):
    """Time periods within a day (maps to wttr.in hourly indices)"""
    MORNING = "morning"      # 6:00-12:00 (indices 2-4)
    AFTERNOON = "afternoon"  # 12:00-18:00 (indices 4-6)
    EVENING = "evening"      # 18:00-21:00 (indices 6-7)
    NIGHT = "night"          # 21:00-6:00 (indices 7, 0-2)


@dataclass
class TimeOfDaySpec:
    """Specification of a time period"""
    time_of_day: TimeOfDay
    display_name: str
    hourly_indices: List[int]  # wttr.in hourly array indices (0-7)


class TimeOfDayVariable(Variable):
    """Variable for time-of-day specifications."""

    TIMES = {
        TimeOfDay.MORNING: TimeOfDaySpec(TimeOfDay.MORNING, "morning", [2, 3, 4]),
        TimeOfDay.AFTERNOON: TimeOfDaySpec(TimeOfDay.AFTERNOON, "afternoon", [4, 5, 6]),
        TimeOfDay.EVENING: TimeOfDaySpec(TimeOfDay.EVENING, "evening", [6, 7]),
        TimeOfDay.NIGHT: TimeOfDaySpec(TimeOfDay.NIGHT, "night", [0, 1, 7]),
    }

    def __init__(self, allowed_times: List[TimeOfDay] = None):
        super().__init__("time_of_day", VariableType.DATE)
        self.allowed_times = allowed_times or list(TimeOfDay)

    def sample(self, rng: random.Random) -> TimeOfDaySpec:
        time = rng.choice(self.allowed_times)
        return self.TIMES[time]

    def get_display_value(self, value: TimeOfDaySpec) -> str:
        return value.display_name

    def get_api_value(self, value: TimeOfDaySpec) -> str:
        return value.time_of_day.value


class MetricType(Enum):
    """Types of weather metrics"""
    TEMPERATURE = "temperature"
    TEMPERATURE_HIGH = "temperature_high"
    TEMPERATURE_LOW = "temperature_low"
    FEELS_LIKE = "feels_like"
    HUMIDITY = "humidity"
    WIND_SPEED = "wind_speed"
    PRECIPITATION = "precipitation"
    PRECIPITATION_CHANCE = "precipitation_chance"
    UV_INDEX = "uv_index"
    VISIBILITY = "visibility"
    PRESSURE = "pressure"
    CLOUD_COVER = "cloud_cover"
    CONDITION = "condition"  # Sunny, Cloudy, Rainy, etc.
    HAS_RAIN = "has_rain"  # Boolean: will it rain?


@dataclass
class MetricSpec:
    """Specification of a weather metric"""
    metric_type: MetricType
    display_name: str  # Human-readable name
    api_field: str  # Field name in wttr.in API
    unit: str  # Unit for display
    is_boolean: bool = False  # Whether this is a yes/no question
    full_tolerance: float = 0  # For numeric validation
    partial_tolerance: float = 0


class WeatherMetricVariable(Variable):
    """
    Variable for weather metrics/measurements.

    Supports various weather data points available from wttr.in.
    """

    METRICS: Dict[MetricType, MetricSpec] = {
        MetricType.TEMPERATURE: MetricSpec(
            MetricType.TEMPERATURE, "temperature", "tempC", "°C",
            full_tolerance=0, partial_tolerance=0
        ),
        MetricType.TEMPERATURE_HIGH: MetricSpec(
            MetricType.TEMPERATURE_HIGH, "high temperature", "maxtempC", "°C",
            full_tolerance=0, partial_tolerance=0
        ),
        MetricType.TEMPERATURE_LOW: MetricSpec(
            MetricType.TEMPERATURE_LOW, "low temperature", "mintempC", "°C",
            full_tolerance=0, partial_tolerance=0
        ),
        MetricType.FEELS_LIKE: MetricSpec(
            MetricType.FEELS_LIKE, "feels like temperature", "FeelsLikeC", "°C",
            full_tolerance=0, partial_tolerance=0
        ),
        MetricType.HUMIDITY: MetricSpec(
            MetricType.HUMIDITY, "humidity", "humidity", "%",
            full_tolerance=5, partial_tolerance=15
        ),
        MetricType.WIND_SPEED: MetricSpec(
            MetricType.WIND_SPEED, "wind speed", "windspeedKmph", "km/h",
            full_tolerance=5, partial_tolerance=10
        ),
        MetricType.PRECIPITATION: MetricSpec(
            MetricType.PRECIPITATION, "precipitation", "precipMM", "mm",
            full_tolerance=1, partial_tolerance=3
        ),
        MetricType.PRECIPITATION_CHANCE: MetricSpec(
            MetricType.PRECIPITATION_CHANCE, "peak chance of rain", "chanceofrain", "%",
            full_tolerance=5, partial_tolerance=10
        ),
        MetricType.UV_INDEX: MetricSpec(
            MetricType.UV_INDEX, "UV index", "uvIndex", "",
            full_tolerance=1, partial_tolerance=2
        ),
        MetricType.VISIBILITY: MetricSpec(
            MetricType.VISIBILITY, "visibility", "visibility", "km",
            full_tolerance=2, partial_tolerance=5
        ),
        MetricType.PRESSURE: MetricSpec(
            MetricType.PRESSURE, "atmospheric pressure", "pressure", "hPa",
            full_tolerance=5, partial_tolerance=10
        ),
        MetricType.CLOUD_COVER: MetricSpec(
            MetricType.CLOUD_COVER, "cloud cover", "cloudcover", "%",
            full_tolerance=10, partial_tolerance=20
        ),
        MetricType.CONDITION: MetricSpec(
            MetricType.CONDITION, "weather condition", "weatherDesc", "",
        ),
        MetricType.HAS_RAIN: MetricSpec(
            MetricType.HAS_RAIN, "rain", "chanceofrain", "",
            is_boolean=True
        ),
    }

    def __init__(self, allowed_metrics: List[MetricType] = None):
        """
        Initialize weather metric variable.

        Args:
            allowed_metrics: Which metrics to use (default: common ones)
        """
        super().__init__("metric", VariableType.METRIC)
        self.allowed_metrics = allowed_metrics or [
            MetricType.TEMPERATURE,
            MetricType.TEMPERATURE_HIGH,
            MetricType.TEMPERATURE_LOW,
            MetricType.HUMIDITY,
            MetricType.WIND_SPEED,
            MetricType.PRECIPITATION_CHANCE,
        ]

    def sample(self, rng: random.Random) -> MetricSpec:
        """Sample a weather metric"""
        metric_type = rng.choice(self.allowed_metrics)
        return self.METRICS[metric_type]

    def get_display_value(self, value: MetricSpec) -> str:
        """Get human-readable metric name"""
        return value.display_name

    def get_api_value(self, value: MetricSpec) -> str:
        """Get API field name"""
        return value.api_field
