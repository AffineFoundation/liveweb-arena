"""Diversity and stability tests for expanded OpenMeteo template set."""

from collections import Counter

import pytest

from liveweb_arena.core.validators.base import get_template
from liveweb_arena.plugins.openmeteo.templates.city_pair_forecast_gap import (
    OpenMeteoCityPairForecastGapTemplate,
)
from liveweb_arena.plugins.openmeteo.templates.comfort_index import OpenMeteoComfortIndexTemplate
from liveweb_arena.plugins.openmeteo.templates.daily_range import OpenMeteoDailyRangeTemplate
from liveweb_arena.plugins.openmeteo.templates.humidity_band_hours import (
    OpenMeteoHumidityBandHoursTemplate,
)
from liveweb_arena.plugins.openmeteo.templates.precip_window_count import (
    OpenMeteoPrecipWindowCountTemplate,
)
from liveweb_arena.plugins.openmeteo.templates.wind_shift import OpenMeteoWindShiftTemplate


TEMPLATE_MATRIX = [
    ("openmeteo_daily_range", OpenMeteoDailyRangeTemplate),
    ("openmeteo_precip_window_count", OpenMeteoPrecipWindowCountTemplate),
    ("openmeteo_humidity_band_hours", OpenMeteoHumidityBandHoursTemplate),
    ("openmeteo_wind_shift", OpenMeteoWindShiftTemplate),
    ("openmeteo_city_pair_forecast_gap", OpenMeteoCityPairForecastGapTemplate),
    ("openmeteo_comfort_index", OpenMeteoComfortIndexTemplate),
]


@pytest.mark.parametrize("name,template_cls", TEMPLATE_MATRIX)
def test_registry_lookup_matches_class(name, template_cls):
    assert get_template(name) is template_cls


@pytest.mark.parametrize("name,template_cls", TEMPLATE_MATRIX)
def test_question_fields_present_for_many_seeds(name, template_cls):
    template = template_cls()
    for seed in range(1, 150):
        q = template.generate(seed)
        assert q.question_text
        assert q.start_url.startswith("https://open-meteo.com/")
        assert isinstance(q.variables, dict) and q.variables
        assert isinstance(q.validation_info, dict) and q.validation_info
        assert q.template_name == name
        assert q.expected_steps >= 7


@pytest.mark.parametrize("template_cls", [OpenMeteoDailyRangeTemplate, OpenMeteoComfortIndexTemplate])
def test_city_distribution_is_not_collapsed(template_cls):
    template = template_cls()
    cities = []
    for seed in range(1, 500):
        q = template.generate(seed)
        key = q.validation_info.get("city_name")
        assert key is not None
        cities.append(key)
    unique = set(cities)
    assert len(unique) >= 80


def test_precip_window_threshold_distribution():
    template = OpenMeteoPrecipWindowCountTemplate()
    thresholds = []
    windows = []
    for seed in range(1, 500):
        q = template.generate(seed)
        thresholds.append(q.validation_info["threshold"])
        windows.append(q.validation_info["window_name"])
    threshold_counter = Counter(thresholds)
    window_counter = Counter(windows)
    assert set(threshold_counter.keys()) == {20, 40, 60, 80}
    assert set(window_counter.keys()) == {"morning", "afternoon", "evening", "overnight"}
    assert all(v > 50 for v in threshold_counter.values())
    assert all(v > 50 for v in window_counter.values())


def test_humidity_band_distribution():
    template = OpenMeteoHumidityBandHoursTemplate()
    bands = []
    for seed in range(1, 500):
        q = template.generate(seed)
        bands.append((q.validation_info["band_low"], q.validation_info["band_high"]))
    counts = Counter(bands)
    assert set(counts.keys()) == {(30, 50), (40, 60), (50, 70), (60, 80)}
    assert all(v > 50 for v in counts.values())


def test_wind_shift_window_pairs_distribution():
    template = OpenMeteoWindShiftTemplate()
    pairs = []
    for seed in range(1, 500):
        q = template.generate(seed)
        pairs.append((q.validation_info["window_a"], q.validation_info["window_b"]))
    counts = Counter(pairs)
    assert set(counts.keys()) == {
        ("morning", "evening"),
        ("overnight", "afternoon"),
        ("morning", "afternoon"),
    }
    assert all(v > 80 for v in counts.values())


def test_city_pair_template_uses_two_different_cities():
    template = OpenMeteoCityPairForecastGapTemplate()
    for seed in range(1, 300):
        q = template.generate(seed)
        assert q.validation_info["city1_name"] != q.validation_info["city2_name"]
        assert q.validation_info["city1_coord_key"] != q.validation_info["city2_coord_key"]


def test_city_pair_metric_day_distribution():
    template = OpenMeteoCityPairForecastGapTemplate()
    metric_counter = Counter()
    day_counter = Counter()
    for seed in range(1, 500):
        q = template.generate(seed)
        metric_counter[q.validation_info["metric_field"]] += 1
        day_counter[q.validation_info["day_idx"]] += 1
    assert set(metric_counter.keys()) == {
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_probability_max",
    }
    assert set(day_counter.keys()) == {0, 1, 2}
    assert all(v > 90 for v in metric_counter.values())
    assert all(v > 120 for v in day_counter.values())


def test_comfort_index_prompt_contains_formula_signal():
    template = OpenMeteoComfortIndexTemplate()
    found_formula_prompts = 0
    for seed in range(1, 100):
        q = template.generate(seed)
        txt = q.question_text.lower()
        if "0.2" in txt and "0.05" in txt:
            found_formula_prompts += 1
    assert found_formula_prompts >= 90


def test_new_templates_have_nonempty_validation_rules():
    for _, template_cls in TEMPLATE_MATRIX:
        template = template_cls()
        q = template.generate(123)
        rules = template.get_validation_rules(q.validation_info)
        assert isinstance(rules, str)
        assert len(rules) > 30


@pytest.mark.parametrize("seed", [1, 3, 7, 9, 11, 13, 21, 31, 41, 51])
def test_daily_range_template_invariants(seed):
    q = OpenMeteoDailyRangeTemplate().generate(seed)
    assert q.validation_info["day_idx"] in {0, 1, 2}
    assert q.validation_info["unit"] == "°C"


@pytest.mark.parametrize("seed", [2, 4, 6, 8, 10, 12, 14, 18, 24, 30])
def test_precip_template_invariants(seed):
    q = OpenMeteoPrecipWindowCountTemplate().generate(seed)
    assert q.validation_info["threshold"] in {20, 40, 60, 80}
    assert 0 <= q.validation_info["start_hour"] < q.validation_info["end_hour"] <= 24


@pytest.mark.parametrize("seed", [5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
def test_humidity_template_invariants(seed):
    q = OpenMeteoHumidityBandHoursTemplate().generate(seed)
    low = q.validation_info["band_low"]
    high = q.validation_info["band_high"]
    assert low < high
    assert low % 10 == 0
    assert high % 10 == 0


@pytest.mark.parametrize("seed", [16, 26, 36, 46, 56, 66, 76, 86, 96, 106])
def test_wind_shift_template_invariants(seed):
    q = OpenMeteoWindShiftTemplate().generate(seed)
    assert q.validation_info["window_a"] != q.validation_info["window_b"]
    assert q.validation_info["window_a_start"] < q.validation_info["window_a_end"]
    assert q.validation_info["window_b_start"] < q.validation_info["window_b_end"]


@pytest.mark.parametrize("seed", [17, 27, 37, 47, 57, 67, 77, 87, 97, 107])
def test_city_pair_template_invariants(seed):
    q = OpenMeteoCityPairForecastGapTemplate().generate(seed)
    assert q.validation_info["metric_field"] in {
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_probability_max",
    }
    assert q.validation_info["day_idx"] in {0, 1, 2}
    assert q.validation_info["unit"] in {"°C", "%"}


@pytest.mark.parametrize("seed", [19, 29, 39, 49, 59, 69, 79, 89, 99, 109])
def test_comfort_template_invariants(seed):
    q = OpenMeteoComfortIndexTemplate().generate(seed)
    assert "formula" in q.validation_info
    assert "coord_key" in q.validation_info


def test_diversity_floor_daily_range():
    template = OpenMeteoDailyRangeTemplate()
    questions = {template.generate(seed).question_text for seed in range(1, 400)}
    assert len(questions) >= 200


def test_diversity_floor_precip():
    template = OpenMeteoPrecipWindowCountTemplate()
    questions = {template.generate(seed).question_text for seed in range(1, 400)}
    assert len(questions) >= 200


def test_diversity_floor_humidity():
    template = OpenMeteoHumidityBandHoursTemplate()
    questions = {template.generate(seed).question_text for seed in range(1, 400)}
    assert len(questions) >= 200


def test_diversity_floor_wind_shift():
    template = OpenMeteoWindShiftTemplate()
    questions = {template.generate(seed).question_text for seed in range(1, 400)}
    assert len(questions) >= 200


def test_diversity_floor_city_pair():
    template = OpenMeteoCityPairForecastGapTemplate()
    questions = {template.generate(seed).question_text for seed in range(1, 400)}
    assert len(questions) >= 200


def test_diversity_floor_comfort():
    template = OpenMeteoComfortIndexTemplate()
    questions = {template.generate(seed).question_text for seed in range(1, 400)}
    assert len(questions) >= 140
