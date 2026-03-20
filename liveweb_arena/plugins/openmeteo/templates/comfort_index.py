"""Comfort index template for Open Meteo - HARD difficulty."""

import random
from typing import Any, Dict, Optional

from liveweb_arena.core.ground_truth_trigger import (
    GroundTruthResult,
    TriggerConfig,
    UrlPatternTrigger,
)
from liveweb_arena.core.gt_collector import GTSourceType
from liveweb_arena.core.validators.base import (
    GeneratedQuestion,
    QuestionTemplate,
    ValidationResult,
    register_template,
)

from .common import DOCS_HOME_URL, get_collected_location_data
from .variables import CITIES

PATTERNS = [
    "For {city} on Open-Meteo, compute the comfort index: temperature_2m - 0.2 * wind_speed_10m - 0.05 * relative_humidity_2m. What is the value?",
    "Using Open-Meteo current weather for {city}, calculate CI = T - 0.2W - 0.05H, where T is temperature (°C), W is wind speed (km/h), H is humidity (%).",
    "According to Open-Meteo, what is {city}'s comfort index defined as T - 0.2W - 0.05H from current weather values?",
]


@register_template("openmeteo_comfort_index")
class OpenMeteoComfortIndexTemplate(QuestionTemplate):
    """Compute a deterministic index from three current-weather fields."""

    GT_SOURCE = GTSourceType.PAGE_ONLY

    def __init__(self):
        super().__init__("openmeteo_comfort_index")

    def generate(self, seed: int, variant: Optional[int] = None) -> GeneratedQuestion:
        rng = random.Random(seed)
        city = rng.choice(CITIES)
        pattern = rng.choice(PATTERNS)

        return GeneratedQuestion(
            question_text=pattern.format(city=city.display_name),
            start_url=DOCS_HOME_URL,
            variables={"city": city.name},
            validation_info={
                "city_name": city.name,
                "coord_key": city.coord_key,
                "formula": "T - 0.2W - 0.05H",
                "unit": "index-points",
            },
            template_name=self.name,
            expected_steps=8,
        )

    def get_validation_rules(self, validation_info: Dict[str, Any]) -> str:
        return (
            "Task-Specific Rules (Open Meteo Comfort Index):\n"
            f"- City: {validation_info.get('city_name', '')}\n"
            "- Formula: CI = temperature - 0.2*wind_speed - 0.05*humidity\n"
            "- Use current_weather values from Open-Meteo\n"
            "- Score 1.0: within ±0.8 index-points\n"
            "- Score 0.5: within ±2.0 index-points\n"
            "- Score 0.0: otherwise"
        )

    async def get_ground_truth(self, validation_info: Dict[str, Any]) -> GroundTruthResult:
        data, failure = get_collected_location_data(
            validation_info.get("coord_key", ""),
            validation_info.get("city_name", ""),
        )
        if failure is not None:
            return failure

        current = data.get("current_weather")
        hourly = data.get("hourly")
        if not isinstance(current, dict):
            return GroundTruthResult.fail("No current_weather in API response")
        if not isinstance(hourly, dict):
            return GroundTruthResult.fail("No hourly data in API response")

        temp_raw = current.get("temperature")
        wind_raw = current.get("windspeed")
        if temp_raw is None or wind_raw is None:
            return GroundTruthResult.fail("Missing temperature/windspeed in current_weather")
        try:
            temp = float(temp_raw)
            wind = float(wind_raw)
        except (TypeError, ValueError):
            return GroundTruthResult.fail("Non-numeric temperature/windspeed")

        # Humidity may not be in current_weather; use hourly value nearest current time.
        times = hourly.get("time")
        humidity = hourly.get("relative_humidity_2m")
        if not isinstance(times, list) or not isinstance(humidity, list) or len(times) != len(humidity):
            return GroundTruthResult.fail("Invalid hourly humidity arrays")

        current_time = current.get("time")
        humidity_value = None
        if isinstance(current_time, str) and current_time in times:
            idx = times.index(current_time)
            if idx < len(humidity) and humidity[idx] is not None:
                humidity_value = humidity[idx]
        if humidity_value is None and humidity:
            humidity_value = humidity[0]

        try:
            hum = float(humidity_value)
        except (TypeError, ValueError):
            return GroundTruthResult.fail("Non-numeric humidity value")

        ci = temp - 0.2 * wind - 0.05 * hum
        return GroundTruthResult.ok(f"{ci:.2f}")

    async def validate_answer(self, answer: str, validation_info: Dict[str, Any]) -> ValidationResult:
        return ValidationResult(
            score=0.0,
            is_correct=False,
            expected=None,
            actual=answer,
            details="Use LLM validation",
        )

    def get_ground_truth_trigger(self, validation_info: dict) -> TriggerConfig:
        return TriggerConfig(trigger=UrlPatternTrigger(domains=["open-meteo.com"]))

    @classmethod
    def get_cache_source(cls) -> str:
        return "openmeteo"

    def get_gt_source(self) -> GTSourceType:
        return self.GT_SOURCE
