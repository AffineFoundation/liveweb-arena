"""Hourly precipitation-threshold counting template - MEDIUM difficulty."""

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

from .common import (
    DOCS_HOME_URL,
    extract_hour_of_day,
    get_collected_location_data,
    get_today_hourly_time_value_pairs,
)
from .variables import CITIES

WINDOWS = [
    ("morning", 6, 12),
    ("afternoon", 12, 18),
    ("evening", 18, 24),
    ("overnight", 0, 6),
]
THRESHOLDS = [20, 40, 60, 80]

PATTERNS = [
    "For {city}, how many {window} hours today have precipitation probability at least {threshold}% on Open-Meteo?",
    "Using Open-Meteo hourly data for {city}, count {window} time slots where precipitation probability is >= {threshold}%.",
    "According to Open-Meteo, in {city} today, how many hours in the {window} have precipitation probability of {threshold}% or higher?",
]


@register_template("openmeteo_precip_window_count")
class OpenMeteoPrecipWindowCountTemplate(QuestionTemplate):
    """Count hourly precipitation probability exceedances inside a time window."""

    GT_SOURCE = GTSourceType.PAGE_ONLY

    def __init__(self):
        super().__init__("openmeteo_precip_window_count")

    def generate(self, seed: int, variant: Optional[int] = None) -> GeneratedQuestion:
        rng = random.Random(seed)
        city = rng.choice(CITIES)
        window_name, start_hour, end_hour = rng.choice(WINDOWS)
        threshold = rng.choice(THRESHOLDS)
        pattern = rng.choice(PATTERNS)

        return GeneratedQuestion(
            question_text=pattern.format(
                city=city.display_name,
                window=window_name,
                threshold=threshold,
            ),
            start_url=DOCS_HOME_URL,
            variables={"city": city.name, "window": window_name, "threshold": threshold},
            validation_info={
                "city_name": city.name,
                "coord_key": city.coord_key,
                "window_name": window_name,
                "start_hour": start_hour,
                "end_hour": end_hour,
                "threshold": threshold,
            },
            template_name=self.name,
            expected_steps=7,
        )

    def get_validation_rules(self, validation_info: Dict[str, Any]) -> str:
        return (
            "Task-Specific Rules (Open Meteo Precip Window Count):\n"
            f"- City: {validation_info.get('city_name', '')}\n"
            f"- Window: {validation_info.get('window_name', '')}\n"
            f"- Threshold: >= {validation_info.get('threshold', 0)}%\n"
            "- Score 1.0: exact count\n"
            "- Score 0.5: off by 1\n"
            "- Score 0.0: off by >=2 or non-numeric"
        )

    async def get_ground_truth(self, validation_info: Dict[str, Any]) -> GroundTruthResult:
        data, failure = get_collected_location_data(
            validation_info.get("coord_key", ""),
            validation_info.get("city_name", ""),
        )
        if failure is not None:
            return failure

        pairs, failure = get_today_hourly_time_value_pairs(data, "precipitation_probability")
        if failure is not None:
            return failure

        start_hour = int(validation_info.get("start_hour", 0))
        end_hour = int(validation_info.get("end_hour", 24))
        threshold = float(validation_info.get("threshold", 0))

        count = 0
        for ts, value in pairs:
            hour = extract_hour_of_day(ts)
            if hour is None:
                return GroundTruthResult.fail(f"Invalid hourly timestamp format: {ts}")
            if start_hour <= hour < end_hour and value >= threshold:
                count += 1
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
