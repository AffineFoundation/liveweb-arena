"""Humidity band counting template for Open Meteo - MEDIUM difficulty."""

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

from .common import DOCS_HOME_URL, get_collected_location_data, get_today_hourly_time_value_pairs
from .variables import CITIES

HUMIDITY_BANDS = [
    (30, 50),
    (40, 60),
    (50, 70),
    (60, 80),
]

PATTERNS = [
    "In {city} today, how many hourly points on Open-Meteo have relative humidity between {low}% and {high}% (inclusive)?",
    "Using Open-Meteo hourly humidity for {city}, count hours where humidity is within [{low}%, {high}%].",
    "According to Open-Meteo, what is the number of today's hours in {city} with relative humidity from {low}% to {high}%?",
]


@register_template("openmeteo_humidity_band_hours")
class OpenMeteoHumidityBandHoursTemplate(QuestionTemplate):
    """Count today's hourly humidity values within an inclusive numeric band."""

    GT_SOURCE = GTSourceType.PAGE_ONLY

    def __init__(self):
        super().__init__("openmeteo_humidity_band_hours")

    def generate(self, seed: int, variant: Optional[int] = None) -> GeneratedQuestion:
        rng = random.Random(seed)
        city = rng.choice(CITIES)
        low, high = rng.choice(HUMIDITY_BANDS)
        pattern = rng.choice(PATTERNS)

        return GeneratedQuestion(
            question_text=pattern.format(city=city.display_name, low=low, high=high),
            start_url=DOCS_HOME_URL,
            variables={"city": city.name, "band_low": low, "band_high": high},
            validation_info={
                "city_name": city.name,
                "coord_key": city.coord_key,
                "band_low": low,
                "band_high": high,
            },
            template_name=self.name,
            expected_steps=7,
        )

    def get_validation_rules(self, validation_info: Dict[str, Any]) -> str:
        return (
            "Task-Specific Rules (Open Meteo Humidity Band):\n"
            f"- City: {validation_info.get('city_name', '')}\n"
            f"- Inclusive humidity band: {validation_info.get('band_low', '')}% to {validation_info.get('band_high', '')}%\n"
            "- Score 1.0: exact count\n"
            "- Score 0.5: off by 1\n"
            "- Score 0.0: otherwise"
        )

    async def get_ground_truth(self, validation_info: Dict[str, Any]) -> GroundTruthResult:
        data, failure = get_collected_location_data(
            validation_info.get("coord_key", ""),
            validation_info.get("city_name", ""),
        )
        if failure is not None:
            return failure

        pairs, failure = get_today_hourly_time_value_pairs(data, "relative_humidity_2m")
        if failure is not None:
            return failure

        low = float(validation_info.get("band_low", 0))
        high = float(validation_info.get("band_high", 100))
        count = sum(1 for _, value in pairs if low <= value <= high)
        return GroundTruthResult.ok(str(count))

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
