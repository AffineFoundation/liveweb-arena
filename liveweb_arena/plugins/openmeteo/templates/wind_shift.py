"""Wind shift template for Open Meteo - MEDIUM/HARD difficulty."""

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

WINDOW_PAIRS = [
    ("morning", 6, 12, "evening", 18, 24),
    ("overnight", 0, 6, "afternoon", 12, 18),
    ("morning", 6, 12, "afternoon", 12, 18),
]

PATTERNS = [
    "For {city} on Open-Meteo, compare average wind speed in the {window_a} vs {window_b}. What is the signed difference ({window_b} - {window_a}) in km/h?",
    "Using Open-Meteo hourly wind speeds for {city}, compute average {window_b} wind speed minus average {window_a} wind speed.",
    "According to Open-Meteo, by how many km/h does average wind speed change from {window_a} to {window_b} in {city} today?",
]


@register_template("openmeteo_wind_shift")
class OpenMeteoWindShiftTemplate(QuestionTemplate):
    """Compute signed difference of average wind speed between two time windows."""

    GT_SOURCE = GTSourceType.PAGE_ONLY

    def __init__(self):
        super().__init__("openmeteo_wind_shift")

    def generate(self, seed: int, variant: Optional[int] = None) -> GeneratedQuestion:
        rng = random.Random(seed)
        city = rng.choice(CITIES)
        window_a, a_start, a_end, window_b, b_start, b_end = rng.choice(WINDOW_PAIRS)
        pattern = rng.choice(PATTERNS)

        return GeneratedQuestion(
            question_text=pattern.format(
                city=city.display_name,
                window_a=window_a,
                window_b=window_b,
            ),
            start_url=DOCS_HOME_URL,
            variables={"city": city.name, "window_a": window_a, "window_b": window_b},
            validation_info={
                "city_name": city.name,
                "coord_key": city.coord_key,
                "window_a": window_a,
                "window_a_start": a_start,
                "window_a_end": a_end,
                "window_b": window_b,
                "window_b_start": b_start,
                "window_b_end": b_end,
                "unit": "km/h",
            },
            template_name=self.name,
            expected_steps=8,
        )

    def get_validation_rules(self, validation_info: Dict[str, Any]) -> str:
        return (
            "Task-Specific Rules (Open Meteo Wind Shift):\n"
            f"- City: {validation_info.get('city_name', '')}\n"
            f"- Signed difference = avg({validation_info.get('window_b', 'B')}) - avg({validation_info.get('window_a', 'A')})\n"
            "- Positive means second window is windier\n"
            "- Score 1.0: signed value within ±1.0 km/h\n"
            "- Score 0.5: sign correct but absolute error >1.0 and <=3.0 km/h\n"
            "- Score 0.0: wrong sign or large error"
        )

    async def get_ground_truth(self, validation_info: Dict[str, Any]) -> GroundTruthResult:
        data, failure = get_collected_location_data(
            validation_info.get("coord_key", ""),
            validation_info.get("city_name", ""),
        )
        if failure is not None:
            return failure

        pairs, failure = get_today_hourly_time_value_pairs(data, "wind_speed_10m")
        if failure is not None:
            return failure

        a_start = int(validation_info.get("window_a_start", 0))
        a_end = int(validation_info.get("window_a_end", 24))
        b_start = int(validation_info.get("window_b_start", 0))
        b_end = int(validation_info.get("window_b_end", 24))

        a_vals = []
        b_vals = []
        for ts, value in pairs:
            hour = extract_hour_of_day(ts)
            if hour is None:
                return GroundTruthResult.fail(f"Invalid hourly timestamp format: {ts}")
            if a_start <= hour < a_end:
                a_vals.append(value)
            if b_start <= hour < b_end:
                b_vals.append(value)

        if not a_vals:
            return GroundTruthResult.fail("No data points in first wind window")
        if not b_vals:
            return GroundTruthResult.fail("No data points in second wind window")

        avg_a = sum(a_vals) / len(a_vals)
        avg_b = sum(b_vals) / len(b_vals)
        diff = avg_b - avg_a
        return GroundTruthResult.ok(f"{diff:.1f}km/h")

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
