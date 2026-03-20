"""Expanded integration tests for newly added OpenMeteo templates."""

import asyncio
from copy import deepcopy

import pytest

from liveweb_arena.core.gt_collector import GTCollector, GTSourceType, set_current_gt_collector
from liveweb_arena.core.task_registry import TaskRegistry
from liveweb_arena.core.validators.base import get_registered_templates
from liveweb_arena.plugins.base import SubTask
from liveweb_arena.plugins.openmeteo.templates.common import (
    extract_hour_of_day,
    get_daily_series,
    get_daily_value,
    get_today_hourly_time_value_pairs,
)
from liveweb_arena.plugins.openmeteo.templates.daily_range import OpenMeteoDailyRangeTemplate
from liveweb_arena.plugins.openmeteo.templates.precip_window_count import (
    OpenMeteoPrecipWindowCountTemplate,
)
from liveweb_arena.plugins.openmeteo.templates.humidity_band_hours import (
    OpenMeteoHumidityBandHoursTemplate,
)
from liveweb_arena.plugins.openmeteo.templates.wind_shift import OpenMeteoWindShiftTemplate
from liveweb_arena.plugins.openmeteo.templates.city_pair_forecast_gap import (
    OpenMeteoCityPairForecastGapTemplate,
)
from liveweb_arena.plugins.openmeteo.templates.comfort_index import OpenMeteoComfortIndexTemplate


def run_async(coro):
    return asyncio.run(coro)


@pytest.fixture
def collector():
    gt_collector = GTCollector(
        subtasks=[SubTask(plugin_name="openmeteo", intent="test", validation_info={}, answer_tag="answer1")]
    )
    set_current_gt_collector(gt_collector)
    try:
        yield gt_collector
    finally:
        set_current_gt_collector(None)


def _seed_weather(loc_key: str):
    return {
        "_location_key": loc_key,
        "current_weather": {
            "temperature": 21.4,
            "windspeed": 12.0,
            "time": "2026-03-20T10:00",
        },
        "hourly": {
            "time": [
                "2026-03-20T00:00",
                "2026-03-20T03:00",
                "2026-03-20T06:00",
                "2026-03-20T09:00",
                "2026-03-20T12:00",
                "2026-03-20T15:00",
                "2026-03-20T18:00",
                "2026-03-20T21:00",
                "2026-03-21T00:00",
            ],
            "temperature_2m": [9.0, 8.0, 11.0, 15.0, 20.0, 23.0, 19.0, 14.0, 10.0],
            "relative_humidity_2m": [82, 80, 72, 63, 54, 49, 58, 68, 75],
            "wind_speed_10m": [5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 14.0, 9.0, 7.0],
            "precipitation_probability": [10, 20, 35, 40, 70, 65, 55, 25, 15],
        },
        "daily": {
            "time": ["2026-03-20", "2026-03-21", "2026-03-22"],
            "temperature_2m_max": [24.0, 22.0, 19.0],
            "temperature_2m_min": [8.0, 10.0, 7.0],
            "precipitation_probability_max": [70, 60, 50],
        },
    }


def _merge_city(collector_obj, key: str):
    collector_obj._merge_api_data(
        f"https://open-meteo.com/en/docs?latitude={key.split(',')[0]}&longitude={key.split(',')[1]}",
        _seed_weather(key),
    )


@pytest.mark.parametrize(
    ("template_cls", "expected_steps_min"),
    [
        (OpenMeteoDailyRangeTemplate, 7),
        (OpenMeteoPrecipWindowCountTemplate, 7),
        (OpenMeteoHumidityBandHoursTemplate, 7),
        (OpenMeteoWindShiftTemplate, 8),
        (OpenMeteoCityPairForecastGapTemplate, 10),
        (OpenMeteoComfortIndexTemplate, 8),
    ],
)
def test_new_template_generation_shape(template_cls, expected_steps_min):
    q = template_cls().generate(42)
    assert q.question_text
    assert q.start_url == "https://open-meteo.com/en/docs"
    assert q.expected_steps >= expected_steps_min
    assert "coord_key" in q.validation_info or "city1_coord_key" in q.validation_info


def test_template_registration_contains_new_openmeteo_templates():
    templates = get_registered_templates()
    assert "openmeteo_daily_range" in templates
    assert "openmeteo_precip_window_count" in templates
    assert "openmeteo_humidity_band_hours" in templates
    assert "openmeteo_wind_shift" in templates
    assert "openmeteo_city_pair_forecast_gap" in templates
    assert "openmeteo_comfort_index" in templates


def test_task_registry_contains_new_openmeteo_ids():
    expected = {
        96: ("openmeteo", "openmeteo_daily_range"),
        97: ("openmeteo", "openmeteo_precip_window_count"),
        98: ("openmeteo", "openmeteo_humidity_band_hours"),
        99: ("openmeteo", "openmeteo_wind_shift"),
        100: ("openmeteo", "openmeteo_city_pair_forecast_gap"),
        101: ("openmeteo", "openmeteo_comfort_index"),
    }
    for tid, item in expected.items():
        assert TaskRegistry.TEMPLATES[tid] == item


@pytest.mark.parametrize(
    ("timestamp", "expected_hour"),
    [
        ("2026-03-20T00:00", 0),
        ("2026-03-20T09:45", 9),
        ("2026-03-20T23:59", 23),
        ("bad", None),
        ("2026-03-20", None),
    ],
)
def test_extract_hour_of_day(timestamp, expected_hour):
    assert extract_hour_of_day(timestamp) == expected_hour


def test_get_daily_series_and_value_success():
    data = _seed_weather("1.00,2.00")
    series, failure = get_daily_series(data, "temperature_2m_max")
    assert failure is None
    assert series == [24.0, 22.0, 19.0]

    val, failure = get_daily_value(data, "temperature_2m_min", 2)
    assert failure is None
    assert val == 7.0


def test_get_daily_series_failure_missing_field():
    data = _seed_weather("1.00,2.00")
    del data["daily"]["temperature_2m_max"]
    series, failure = get_daily_series(data, "temperature_2m_max")
    assert series is None
    assert failure is not None
    assert "missing" in str(failure.error).lower()


def test_get_daily_value_failure_out_of_bounds():
    data = _seed_weather("1.00,2.00")
    value, failure = get_daily_value(data, "temperature_2m_min", 30)
    assert value is None
    assert failure is not None


def test_get_today_hourly_time_value_pairs_success():
    data = _seed_weather("1.00,2.00")
    pairs, failure = get_today_hourly_time_value_pairs(data, "wind_speed_10m")
    assert failure is None
    assert len(pairs) == 8
    assert pairs[0][0].startswith("2026-03-20")
    assert pairs[-1][0].startswith("2026-03-20")


def test_get_today_hourly_time_value_pairs_failure_non_numeric():
    data = _seed_weather("1.00,2.00")
    data["hourly"]["wind_speed_10m"][1] = "not-a-number"
    pairs, failure = get_today_hourly_time_value_pairs(data, "wind_speed_10m")
    assert pairs is None
    assert failure is not None


def test_daily_range_ground_truth(collector):
    loc = "35.68,139.65"
    _merge_city(collector, loc)

    result = run_async(
        OpenMeteoDailyRangeTemplate().get_ground_truth(
            {
                "city_name": "Tokyo",
                "coord_key": loc,
                "day_idx": 0,
                "day_label": "today",
            }
        )
    )
    assert result.success is True
    assert result.value == "16.0°C"  # 24 - 8


def test_daily_range_missing_data(collector):
    loc = "35.68,139.65"
    payload = _seed_weather(loc)
    del payload["daily"]["temperature_2m_min"]
    collector._merge_api_data("https://open-meteo.com/en/docs?latitude=35.68&longitude=139.65", payload)

    result = run_async(
        OpenMeteoDailyRangeTemplate().get_ground_truth(
            {
                "city_name": "Tokyo",
                "coord_key": loc,
                "day_idx": 0,
                "day_label": "today",
            }
        )
    )
    assert result.success is False


@pytest.mark.parametrize(
    ("window_name", "start", "end", "threshold", "expected"),
    [
        ("morning", 6, 12, 30, 2),  # 06:00=35, 09:00=40
        ("afternoon", 12, 18, 60, 2),  # 12:00=70, 15:00=65
        ("evening", 18, 24, 50, 1),  # 18:00=55
        ("overnight", 0, 6, 15, 1),  # 03:00=20
    ],
)
def test_precip_window_count_ground_truth(collector, window_name, start, end, threshold, expected):
    loc = "35.68,139.65"
    _merge_city(collector, loc)
    result = run_async(
        OpenMeteoPrecipWindowCountTemplate().get_ground_truth(
            {
                "city_name": "Tokyo",
                "coord_key": loc,
                "window_name": window_name,
                "start_hour": start,
                "end_hour": end,
                "threshold": threshold,
            }
        )
    )
    assert result.success is True
    assert result.value == str(expected)


def test_precip_window_count_invalid_timestamp(collector):
    loc = "35.68,139.65"
    payload = _seed_weather(loc)
    payload["hourly"]["time"][0] = "2026-03-20TXX:00"
    collector._merge_api_data("https://open-meteo.com/en/docs?latitude=35.68&longitude=139.65", payload)

    result = run_async(
        OpenMeteoPrecipWindowCountTemplate().get_ground_truth(
            {
                "city_name": "Tokyo",
                "coord_key": loc,
                "window_name": "overnight",
                "start_hour": 0,
                "end_hour": 6,
                "threshold": 20,
            }
        )
    )
    assert result.success is False


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (40, 60, 3),  # 54,49,58
        (50, 70, 4),  # 63,54,58,68
        (70, 90, 3),  # 82,80,72
        (0, 30, 0),
    ],
)
def test_humidity_band_hours_ground_truth(collector, low, high, expected):
    loc = "35.68,139.65"
    _merge_city(collector, loc)
    result = run_async(
        OpenMeteoHumidityBandHoursTemplate().get_ground_truth(
            {
                "city_name": "Tokyo",
                "coord_key": loc,
                "band_low": low,
                "band_high": high,
            }
        )
    )
    assert result.success is True
    assert result.value == str(expected)


@pytest.mark.parametrize(
    ("a_start", "a_end", "b_start", "b_end", "expected"),
    [
        (6, 12, 18, 24, "2.5km/h"),  # morning avg=(8+10)/2=9.0, evening avg=(14+9)/2=11.5
        (0, 6, 12, 18, "8.0km/h"),   # overnight avg=(5+6)/2=5.5, afternoon avg=(12+15)/2=13.5
    ],
)
def test_wind_shift_ground_truth(collector, a_start, a_end, b_start, b_end, expected):
    loc = "35.68,139.65"
    _merge_city(collector, loc)
    result = run_async(
        OpenMeteoWindShiftTemplate().get_ground_truth(
            {
                "city_name": "Tokyo",
                "coord_key": loc,
                "window_a": "A",
                "window_a_start": a_start,
                "window_a_end": a_end,
                "window_b": "B",
                "window_b_start": b_start,
                "window_b_end": b_end,
            }
        )
    )
    assert result.success is True
    assert result.value == expected


def test_wind_shift_no_data_in_window(collector):
    loc = "35.68,139.65"
    _merge_city(collector, loc)
    result = run_async(
        OpenMeteoWindShiftTemplate().get_ground_truth(
            {
                "city_name": "Tokyo",
                "coord_key": loc,
                "window_a": "A",
                "window_a_start": 1,
                "window_a_end": 2,
                "window_b": "B",
                "window_b_start": 2,
                "window_b_end": 3,
            }
        )
    )
    assert result.success is False


def test_city_pair_forecast_gap_ground_truth(collector):
    city1_key = "35.68,139.65"
    city2_key = "51.51,-0.13"
    _merge_city(collector, city1_key)
    payload2 = _seed_weather(city2_key)
    payload2["daily"]["temperature_2m_max"] = [20.0, 18.0, 16.0]
    collector._merge_api_data(
        "https://open-meteo.com/en/docs?latitude=51.51&longitude=-0.13",
        payload2,
    )

    result = run_async(
        OpenMeteoCityPairForecastGapTemplate().get_ground_truth(
            {
                "city1_name": "Tokyo",
                "city1_coord_key": city1_key,
                "city2_name": "London",
                "city2_coord_key": city2_key,
                "metric_field": "temperature_2m_max",
                "day_idx": 1,
                "unit": "°C",
            }
        )
    )
    assert result.success is True
    assert result.value == "4.0°C"  # 22 - 18


def test_city_pair_forecast_gap_missing_second_city(collector):
    city1_key = "35.68,139.65"
    _merge_city(collector, city1_key)
    result = run_async(
        OpenMeteoCityPairForecastGapTemplate().get_ground_truth(
            {
                "city1_name": "Tokyo",
                "city1_coord_key": city1_key,
                "city2_name": "London",
                "city2_coord_key": "51.51,-0.13",
                "metric_field": "temperature_2m_max",
                "day_idx": 0,
                "unit": "°C",
            }
        )
    )
    assert result.success is False
    assert result.is_data_not_collected()


def test_comfort_index_ground_truth(collector):
    loc = "35.68,139.65"
    _merge_city(collector, loc)
    result = run_async(
        OpenMeteoComfortIndexTemplate().get_ground_truth(
            {
                "city_name": "Tokyo",
                "coord_key": loc,
            }
        )
    )
    assert result.success is True
    # temp=21.4, wind=12, humidity(at 10:00 not exact -> fallback first=82): 21.4-2.4-4.1=14.9
    assert result.value == "14.90"


def test_comfort_index_uses_matched_hourly_humidity(collector):
    loc = "35.68,139.65"
    payload = _seed_weather(loc)
    payload["current_weather"]["time"] = "2026-03-20T12:00"
    collector._merge_api_data("https://open-meteo.com/en/docs?latitude=35.68&longitude=139.65", payload)
    result = run_async(
        OpenMeteoComfortIndexTemplate().get_ground_truth(
            {
                "city_name": "Tokyo",
                "coord_key": loc,
            }
        )
    )
    assert result.success is True
    # humidity at 12:00 is 54 -> 21.4 - 2.4 - 2.7 = 16.3
    assert result.value == "16.30"


def test_comfort_index_failure_invalid_arrays(collector):
    loc = "35.68,139.65"
    payload = _seed_weather(loc)
    payload["hourly"]["relative_humidity_2m"] = ["x", "y"]
    collector._merge_api_data("https://open-meteo.com/en/docs?latitude=35.68&longitude=139.65", payload)
    result = run_async(
        OpenMeteoComfortIndexTemplate().get_ground_truth(
            {"city_name": "Tokyo", "coord_key": loc}
        )
    )
    assert result.success is False


@pytest.mark.parametrize(
    "template_cls",
    [
        OpenMeteoDailyRangeTemplate,
        OpenMeteoPrecipWindowCountTemplate,
        OpenMeteoHumidityBandHoursTemplate,
        OpenMeteoWindShiftTemplate,
        OpenMeteoCityPairForecastGapTemplate,
        OpenMeteoComfortIndexTemplate,
    ],
)
def test_new_templates_gt_source(template_cls):
    assert template_cls().get_gt_source() == GTSourceType.PAGE_ONLY


@pytest.mark.parametrize(
    "template_cls",
    [
        OpenMeteoDailyRangeTemplate,
        OpenMeteoPrecipWindowCountTemplate,
        OpenMeteoHumidityBandHoursTemplate,
        OpenMeteoWindShiftTemplate,
        OpenMeteoCityPairForecastGapTemplate,
        OpenMeteoComfortIndexTemplate,
    ],
)
def test_new_templates_cache_source(template_cls):
    assert template_cls.get_cache_source() == "openmeteo"


def test_seed_stability_daily_range():
    t = OpenMeteoDailyRangeTemplate()
    q1 = t.generate(2026, variant=3)
    q2 = t.generate(2026, variant=3)
    assert q1.question_text == q2.question_text
    assert q1.validation_info == q2.validation_info


def test_seed_stability_precip():
    t = OpenMeteoPrecipWindowCountTemplate()
    q1 = t.generate(2026, variant=7)
    q2 = t.generate(2026, variant=7)
    assert q1.question_text == q2.question_text
    assert q1.validation_info == q2.validation_info


def test_seed_stability_humidity():
    t = OpenMeteoHumidityBandHoursTemplate()
    q1 = t.generate(2026, variant=7)
    q2 = t.generate(2026, variant=7)
    assert q1.question_text == q2.question_text
    assert q1.validation_info == q2.validation_info


def test_seed_stability_wind_shift():
    t = OpenMeteoWindShiftTemplate()
    q1 = t.generate(2026, variant=7)
    q2 = t.generate(2026, variant=7)
    assert q1.question_text == q2.question_text
    assert q1.validation_info == q2.validation_info


def test_seed_stability_city_pair_gap():
    t = OpenMeteoCityPairForecastGapTemplate()
    q1 = t.generate(2026, variant=7)
    q2 = t.generate(2026, variant=7)
    assert q1.question_text == q2.question_text
    assert q1.validation_info == q2.validation_info


def test_seed_stability_comfort_index():
    t = OpenMeteoComfortIndexTemplate()
    q1 = t.generate(2026, variant=7)
    q2 = t.generate(2026, variant=7)
    assert q1.question_text == q2.question_text
    assert q1.validation_info == q2.validation_info


def test_daily_range_day_indices_are_supported(collector):
    loc = "35.68,139.65"
    _merge_city(collector, loc)
    template = OpenMeteoDailyRangeTemplate()
    for idx, expected in [(0, "16.0°C"), (1, "12.0°C"), (2, "12.0°C")]:
        result = run_async(
            template.get_ground_truth(
                {
                    "city_name": "Tokyo",
                    "coord_key": loc,
                    "day_idx": idx,
                    "day_label": f"day-{idx}",
                }
            )
        )
        assert result.success is True
        assert result.value == expected


def test_city_pair_metric_variants(collector):
    city1_key = "35.68,139.65"
    city2_key = "51.51,-0.13"
    _merge_city(collector, city1_key)
    city2 = _seed_weather(city2_key)
    city2["daily"]["temperature_2m_min"] = [3.0, 5.0, 4.0]
    city2["daily"]["precipitation_probability_max"] = [40, 40, 40]
    collector._merge_api_data(
        "https://open-meteo.com/en/docs?latitude=51.51&longitude=-0.13",
        city2,
    )
    template = OpenMeteoCityPairForecastGapTemplate()

    result_min = run_async(
        template.get_ground_truth(
            {
                "city1_name": "Tokyo",
                "city1_coord_key": city1_key,
                "city2_name": "London",
                "city2_coord_key": city2_key,
                "metric_field": "temperature_2m_min",
                "day_idx": 0,
                "unit": "°C",
            }
        )
    )
    assert result_min.success is True
    assert result_min.value == "5.0°C"  # 8 - 3

    result_precip = run_async(
        template.get_ground_truth(
            {
                "city1_name": "Tokyo",
                "city1_coord_key": city1_key,
                "city2_name": "London",
                "city2_coord_key": city2_key,
                "metric_field": "precipitation_probability_max",
                "day_idx": 1,
                "unit": "%",
            }
        )
    )
    assert result_precip.success is True
    assert result_precip.value == "20.0%"


def test_precip_window_count_counts_boundaries(collector):
    loc = "35.68,139.65"
    payload = _seed_weather(loc)
    payload["hourly"]["precipitation_probability"][2] = 40
    payload["hourly"]["precipitation_probability"][3] = 40
    collector._merge_api_data("https://open-meteo.com/en/docs?latitude=35.68&longitude=139.65", payload)

    result = run_async(
        OpenMeteoPrecipWindowCountTemplate().get_ground_truth(
            {
                "city_name": "Tokyo",
                "coord_key": loc,
                "window_name": "morning",
                "start_hour": 6,
                "end_hour": 12,
                "threshold": 40,
            }
        )
    )
    # 06:00 and 09:00 both count due to >= threshold
    assert result.success is True
    assert result.value == "2"


def test_humidity_band_inclusive_logic(collector):
    loc = "35.68,139.65"
    payload = _seed_weather(loc)
    payload["hourly"]["relative_humidity_2m"][4] = 60
    collector._merge_api_data("https://open-meteo.com/en/docs?latitude=35.68&longitude=139.65", payload)
    result = run_async(
        OpenMeteoHumidityBandHoursTemplate().get_ground_truth(
            {
                "city_name": "Tokyo",
                "coord_key": loc,
                "band_low": 60,
                "band_high": 60,
            }
        )
    )
    assert result.success is True
    assert result.value == "1"


def test_wind_shift_signed_result_can_be_negative(collector):
    loc = "35.68,139.65"
    payload = _seed_weather(loc)
    # Make evening calmer than morning.
    payload["hourly"]["wind_speed_10m"][6] = 5.0
    payload["hourly"]["wind_speed_10m"][7] = 4.0
    collector._merge_api_data("https://open-meteo.com/en/docs?latitude=35.68&longitude=139.65", payload)
    result = run_async(
        OpenMeteoWindShiftTemplate().get_ground_truth(
            {
                "city_name": "Tokyo",
                "coord_key": loc,
                "window_a": "morning",
                "window_a_start": 6,
                "window_a_end": 12,
                "window_b": "evening",
                "window_b_start": 18,
                "window_b_end": 24,
            }
        )
    )
    assert result.success is True
    assert result.value.startswith("-")


def test_helper_functions_do_not_mutate_source_payload():
    source = _seed_weather("10.00,20.00")
    baseline = deepcopy(source)
    _ = get_daily_series(source, "temperature_2m_max")
    _ = get_daily_value(source, "temperature_2m_min", 0)
    _ = get_today_hourly_time_value_pairs(source, "wind_speed_10m")
    assert source == baseline
