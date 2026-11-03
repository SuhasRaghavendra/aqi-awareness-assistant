"""
tests/test_fires.py
--------------------
Tests for NASA FIRMS fire data downloader and city aggregation logic.
"""
import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.download_fires import (
    haversine_km,
    normalise_firms,
    aggregate_fires_to_city_day,
    CITY_COORDS,
)


def test_haversine_known_distance():
    """Delhi–Mumbai distance should be approximately 1150 km."""
    delhi_lat, delhi_lon = CITY_COORDS["Delhi"]
    mumbai_lat, mumbai_lon = CITY_COORDS["Mumbai"]
    dist = haversine_km(delhi_lat, delhi_lon, mumbai_lat, mumbai_lon)
    assert 1100 <= dist <= 1200, f"Delhi-Mumbai distance {dist:.1f} km outside expected range"


def test_haversine_zero_distance():
    """Same point should give 0 km."""
    assert haversine_km(28.6, 77.2, 28.6, 77.2) == pytest.approx(0.0, abs=0.01)


def test_normalise_firms_schema():
    """Normalised FIRMS DataFrame must have required columns."""
    raw = pd.DataFrame({
        "latitude":   [30.1, 31.2, 29.5],
        "longitude":  [75.3, 76.1, 74.8],
        "acq_date":   ["2023-10-15", "2023-10-16", "2023-10-15"],
        "frp":        [25.4, 110.2, 55.0],
        "confidence": ["h", "n", "l"],
    })
    df = normalise_firms(raw)
    assert "date" in df.columns
    assert "latitude" in df.columns
    assert "longitude" in df.columns
    assert "frp" in df.columns
    assert len(df) == 3


def test_normalise_firms_drops_invalid():
    """Rows with missing lat/lon should be dropped."""
    raw = pd.DataFrame({
        "latitude":   [30.1, float("nan"), 29.5],
        "longitude":  [75.3, 76.1, float("nan")],
        "acq_date":   ["2023-10-15", "2023-10-16", "2023-10-15"],
        "frp":        [25.4, 110.2, 55.0],
        "confidence": ["h", "n", "l"],
    })
    df = normalise_firms(raw)
    assert len(df) == 1  # only first row is valid


def test_aggregate_fires_within_radius():
    """Fires close to Delhi should appear in Delhi aggregation."""
    # Place 5 fires ~200 km NW of Delhi (Punjab region — stubble burning)
    fires = pd.DataFrame({
        "date": pd.to_datetime(["2023-10-20"] * 5),
        "latitude":  [30.5, 30.7, 31.0, 30.3, 30.8],
        "longitude": [75.5, 75.8, 76.2, 75.2, 75.9],
        "frp": [50.0, 120.0, 80.0, 45.0, 200.0],
    })
    result = aggregate_fires_to_city_day(fires, ["Delhi"], radius_km=500)
    assert not result.empty
    assert result["fire_count_500km"].iloc[0] == 5
    assert result["fire_frp_sum_500km"].iloc[0] == pytest.approx(495.0, abs=0.1)


def test_aggregate_fires_outside_radius():
    """Fires far from city (>500 km) should not appear in aggregation."""
    # Mumbai lat/lon ~1150 km from Delhi
    fires = pd.DataFrame({
        "date": pd.to_datetime(["2023-10-20"] * 3),
        "latitude":  [19.0, 19.1, 18.9],
        "longitude": [72.8, 72.9, 72.7],
        "frp": [50.0, 80.0, 40.0],
    })
    result = aggregate_fires_to_city_day(fires, ["Delhi"], radius_km=500)
    # Should be empty — Mumbai fires are ~1150 km from Delhi
    assert result.empty or result["fire_count_500km"].sum() == 0


def test_aggregate_fires_multiple_cities():
    """Fires should be correctly attributed to their nearest cities."""
    fires = pd.DataFrame({
        "date": pd.to_datetime(["2023-10-20"] * 4),
        "latitude":  [30.5, 30.7, 18.8, 19.2],     # 2 near Delhi, 2 near Mumbai
        "longitude": [75.5, 75.8, 72.5, 73.0],
        "frp": [50.0, 80.0, 30.0, 60.0],
    })
    result = aggregate_fires_to_city_day(fires, ["Delhi", "Mumbai"], radius_km=500)
    if not result.empty:
        delhi_fires = result[result["city"] == "Delhi"]["fire_count_500km"].sum()
        mumbai_fires = result[result["city"] == "Mumbai"]["fire_count_500km"].sum()
        # Delhi should have 2, Mumbai should have 2
        assert delhi_fires == 2
        assert mumbai_fires == 2

