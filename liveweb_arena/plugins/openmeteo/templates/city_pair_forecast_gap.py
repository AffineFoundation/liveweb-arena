"""Two-city daily forecast comparison template - HARD difficulty."""

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
from .variables import CITIES, DailyMetric

DAY_CHOICES = [
    (0, "today"),
    (1, "tomorrow"),
    (2, "the day after tomorrow"),
]

PATTERNS = [
    "Using Open-Meteo, what is the signed difference in {metric_label} for {day_label} between {city1} and {city2} (answer as {city1} minus {city2})?",
    "On Open-Meteo, compare {day_label}'s {metric_label} in {city1} vs {city2}. Report {city1} - {city2} with unit.",
    "According to Open-Meteo forecast data, by how much is {city1}'s {day_label} {metric_label} above or below {city2}'s? (signed {city1} - {city2})",
]


@register_template("openmeteo_city_pair_forecast_gap")
class OpenMeteoCityPairForecastGapTemplate(QuestionTemplate):
    """Compare same-day metric across two cities (signed city1 - city2)."""

    GT_SOURCE = GTSourceType.PAGE_ONLY

    def __init__(self):
        super().__init__("openmeteo_city_pair_forecast_gap")

    def generate(self, seed: int, variant: Optional[int] = None) -> GeneratedQuestion:
        rng = random.Random(seed)
        city1, city2 = rng.sample(CITIES, 2)
        metric = rng.choice(list(DailyMetric))
        day_idx, day_label = rng.choice(DAY_CHOICES)
        pattern = rng.choice(PATTERNS)

        return GeneratedQuestion(
            question_text=pattern.format(
                metric_label=metric.display_name,
                day_label=day_label,
                city1=city1.display_name,
                city2=city2.display_name,
            ),
            start_url=DOCS_HOME_URL,
            variables={
                "city1": city1.name,
                "city2": city2.name,
                "metric": metric.name,
                "day_idx": day_idx,
            },
            validation_info={
                "city1_name": city1.name,
                "city1_coord_key": city1.coord_key,
                "city2_name": city2.name,
                "city2_coord_key": city2.coord_key,
                "metric_field": metric.api_field,
                "metric_label": metric.display_name,
                "unit": metric.unit,
                "day_idx": day_idx,
                "day_label": day_label,
            },
            template_name=self.name,
            expected_steps=10,
        )

    def get_validation_rules(self, validation_info: Dict[str, Any]) -> str:
        return (
            "Task-Specific Rules (Open Meteo City Pair Forecast Gap):\n"
            f"- Day: {validation_info.get('day_label', '')}\n"
            f"- Metric: {validation_info.get('metric_label', '')}\n"
            f"- Signed difference must be {validation_info.get('city1_name', 'city1')} - {validation_info.get('city2_name', 'city2')}\n"
            "- Score 1.0: signed value within ±1.0 unit\n"
            "- Score 0.5: absolute magnitude close but sign wrong OR error <=3.0 units\n"
            "- Score 0.0: otherwise"
        )

    async def get_ground_truth(self, validation_info: Dict[str, Any]) -> GroundTruthResult:
        day_idx = int(validation_info.get("day_idx", 0))
        metric_field = validation_info.get("metric_field", "temperature_2m_max")
        unit = validation_info.get("unit", "")

        city1_data, failure = get_collected_location_data(
            validation_info.get("city1_coord_key", ""),
            validation_info.get("city1_name", ""),
        )
        if failure is not None:
            return failure
        city2_data, failure = get_collected_location_data(
            validation_info.get("city2_coord_key", ""),
            validation_info.get("city2_name", ""),
        )
        if failure is not None:
            return failure

        value1, failure = get_daily_value(city1_data, metric_field, day_idx)
        if failure is not None:
            return failure
        value2, failure = get_daily_value(city2_data, metric_field, day_idx)
        if failure is not None:
            return failure

        diff = value1 - value2
        return GroundTruthResult.ok(f"{diff:.1f}{unit}")

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
