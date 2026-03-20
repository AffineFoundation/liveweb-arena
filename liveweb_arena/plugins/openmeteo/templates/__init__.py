"""Open Meteo question templates."""

from .current_weather import OpenMeteoCurrentWeatherTemplate
from .comparison import OpenMeteoComparisonTemplate
from .hourly_extrema import OpenMeteoHourlyExtremaTemplate
from .forecast_trend import OpenMeteoForecastTrendTemplate
from .daily_range import OpenMeteoDailyRangeTemplate
from .precip_window_count import OpenMeteoPrecipWindowCountTemplate
from .humidity_band_hours import OpenMeteoHumidityBandHoursTemplate
from .wind_shift import OpenMeteoWindShiftTemplate
from .city_pair_forecast_gap import OpenMeteoCityPairForecastGapTemplate
from .comfort_index import OpenMeteoComfortIndexTemplate

__all__ = [
    "OpenMeteoCurrentWeatherTemplate",
    "OpenMeteoComparisonTemplate",
    "OpenMeteoHourlyExtremaTemplate",
    "OpenMeteoForecastTrendTemplate",
    "OpenMeteoDailyRangeTemplate",
    "OpenMeteoPrecipWindowCountTemplate",
    "OpenMeteoHumidityBandHoursTemplate",
    "OpenMeteoWindShiftTemplate",
    "OpenMeteoCityPairForecastGapTemplate",
    "OpenMeteoComfortIndexTemplate",
]
