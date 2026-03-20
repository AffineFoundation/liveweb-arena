"""Daily temperature range template for Open Meteo - MEDIUM difficulty."""

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

from .common import DOCS_HOME_URL, get_collected_location_data, get_daily_value
from .variables import CITIES

DAY_CHOICES = [
    (0, "today"),
    (1, "tomorrow"),
    (2, "the day after tomorrow"),
]

PATTERNS = [
    "According to Open-Meteo, what is the temperature range in {city} for {day_label}?",
    "On Open-Meteo, calculate {day_label}'s max-minus-min temperature in {city}.",
    "Using Open-Meteo forecast data, what is {city}'s {day_label} daily temperature range in °C?",
]


@register_template("openmeteo_daily_range")
class OpenMeteoDailyRangeTemplate(QuestionTemplate):
    """Compute daily max - min for a target city/day."""

    GT_SOURCE = GTSourceType.PAGE_ONLY

    def __init__(self):
        super().__init__("openmeteo_daily_range")

    def generate(self, seed: int, variant: Optional[int] = None) -> GeneratedQuestion:
        rng = random.Random(seed)
        city = rng.choice(CITIES)
        day_idx, day_label = rng.choice(DAY_CHOICES)
        pattern = rng.choice(PATTERNS)

        return GeneratedQuestion(
            question_text=pattern.format(city=city.display_name, day_label=day_label),
            start_url=DOCS_HOME_URL,
            variables={"city": city.name, "day_idx": day_idx},
            validation_info={
                "city_name": city.name,
                "coord_key": city.coord_key,
                "day_idx": day_idx,
                "day_label": day_label,
                "unit": "°C",
            },
            template_name=self.name,
            expected_steps=7,
        )

    def get_validation_rules(self, validation_info: Dict[str, Any]) -> str:
        return (
            "Task-Specific Rules (Open Meteo Daily Range):\n"
            f"- City: {validation_info.get('city_name', '')}\n"
            f"- Day: {validation_info.get('day_label', 'target day')}\n"
            "- Expected answer is (max temperature - min temperature) in °C\n"
            "- Score 1.0: within ±1.0°C\n"
            "- Score 0.5: within ±2.5°C\n"
            "- Score 0.0: otherwise"
        )

    async def get_ground_truth(self, validation_info: Dict[str, Any]) -> GroundTruthResult:
        data, failure = get_collected_location_data(
            validation_info.get("coord_key", ""),
            validation_info.get("city_name", ""),
        )
        if failure is not None:
            return failure

        day_idx = int(validation_info.get("day_idx", 0))
        max_temp, failure = get_daily_value(data, "temperature_2m_max", day_idx)
        if failure is not None:
            return failure
        min_temp, failure = get_daily_value(data, "temperature_2m_min", day_idx)
        if failure is not None:
            return failure

        value = max_temp - min_temp
        return GroundTruthResult.ok(f"{value:.1f}°C")

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
