"""
tests/test_aqi_utils.py
-----------------------
Unit tests for CPCB AQI sub-index calculations, category mapping,
and multi-pollutant validity rules.
"""

from __future__ import annotations

import pytest

from src.features.aqi_utils import (
    aqi_to_category,
    calc_overall_aqi,
    calc_sub_index,
    load_cpcb_breakpoints,
)


def test_load_cpcb_breakpoints():
    bp = load_cpcb_breakpoints()
    assert "pm25" in bp
    assert "pm10" in bp
    assert "no2" in bp
    assert "so2" in bp
    assert "co" in bp
    assert "o3" in bp


@pytest.mark.parametrize(
    "pollutant,concentration,expected_sub_idx",
    [
        ("pm25", 0.0, 0.0),
        ("pm25", 30.0, 50.0),
        ("pm25", 60.0, 100.0),
        ("pm25", 90.0, 200.0),
        ("pm25", 120.0, 300.0),
        ("pm10", 50.0, 50.0),
        ("pm10", 100.0, 100.0),
        ("co", 1.0, 50.0),
        ("co", 2.0, 100.0),
        ("no2", 40.0, 50.0),
        ("no2", 80.0, 100.0),
        ("so2", 40.0, 50.0),
        ("so2", 80.0, 100.0),
    ],
)
def test_calc_sub_index_breakpoints(pollutant, concentration, expected_sub_idx):
    sub = calc_sub_index(pollutant, concentration)
    assert sub is not None
    assert abs(sub - expected_sub_idx) <= 1.0


def test_calc_sub_index_interpolation():
    # Intermediate value for PM2.5 between 0 and 30: 15 µg/m³ -> 25 subindex
    sub_15 = calc_sub_index("pm25", 15.0)
    assert sub_15 is not None
    assert 24.0 <= sub_15 <= 26.0


def test_invalid_concentration():
    assert calc_sub_index("pm25", -10.0) is None
    assert calc_sub_index("pm25", None) is None
    assert calc_sub_index("unknown_pollutant", 50.0) is None


@pytest.mark.parametrize(
    "aqi_val,expected_category",
    [
        (0.0, "Good"),
        (45.0, "Good"),
        (50.0, "Good"),
        (51.0, "Satisfactory"),
        (100.0, "Satisfactory"),
        (101.0, "Moderate"),
        (200.0, "Moderate"),
        (250.0, "Poor"),
        (300.0, "Poor"),
        (350.0, "Very Poor"),
        (400.0, "Very Poor"),
        (450.0, "Severe"),
        (600.0, "Severe"),
    ],
)
def test_aqi_to_category(aqi_val, expected_category):
    assert aqi_to_category(aqi_val) == expected_category


def test_calc_overall_aqi_valid():
    # 3 pollutants including PM2.5
    sub_indices = {
        "pm25": 150.0,
        "no2": 60.0,
        "co": 40.0,
    }
    aqi, cat, dom = calc_overall_aqi(sub_indices, min_subindices=3, require_pm=True)
    assert aqi == 150.0
    assert cat == "Moderate"
    assert dom == "pm25"


def test_calc_overall_aqi_insufficient_pollutants():
    # Only 2 pollutants
    sub_indices = {
        "pm25": 150.0,
        "no2": 60.0,
    }
    aqi, cat, dom = calc_overall_aqi(sub_indices, min_subindices=3, require_pm=True)
    assert aqi is None
    assert cat is None
    assert dom is None


def test_calc_overall_aqi_missing_pm():
    # 3 pollutants but NO PM
    sub_indices = {
        "no2": 80.0,
        "so2": 70.0,
        "co": 60.0,
    }
    aqi, cat, dom = calc_overall_aqi(sub_indices, min_subindices=3, require_pm=True)
    assert aqi is None
    assert cat is None
    assert dom is None
