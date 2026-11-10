"""
tests/test_ingest_merge.py
---------------------------
Tests for the root-cause data merging pipeline.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingest_and_merge import (
    CITY_COORDS,
    CITY_STATE,
    add_calendar_features,
    add_meteorological_derived,
    add_policy_features,
    add_static_city_features,
    build_date_spine,
)

CITIES = ["Delhi", "Mumbai", "Bengaluru"]


def test_date_spine_completeness():
    """Spine must cover every city × day without gaps."""
    spine = build_date_spine(CITIES, "2022-01-01", "2022-01-31")
    assert len(spine) == len(CITIES) * 31
    assert set(spine["city"].unique()) == set(CITIES)
    assert spine["date"].nunique() == 31


def test_date_spine_no_duplicates():
    """No duplicate city × date combinations."""
    spine = build_date_spine(CITIES, "2023-06-01", "2023-06-30")
    dupes = spine.duplicated(subset=["city", "date"]).sum()
    assert dupes == 0, f"{dupes} duplicate city-date rows found"


def test_add_calendar_features():
    """Calendar features must be correct for known dates."""
    df = build_date_spine(["Delhi"], "2023-01-01", "2023-12-31")
    df = add_calendar_features(df)
    assert "month" in df.columns
    assert "season" in df.columns
    assert "weekend_flag" in df.columns
    assert "burning_season_flag" in df.columns
    # January should be winter
    jan_rows = df[df["month"] == 1]
    assert (jan_rows["season"] == "winter").all()
    # June–September should be monsoon
    monsoon = df[df["month"].isin([6, 7, 8, 9])]
    assert (monsoon["season"] == "monsoon").all()


def test_burning_season_flag():
    """Burning season flag must be 1 for Oct 15–Nov 30, 0 otherwise."""
    df = build_date_spine(["Delhi"], "2023-10-01", "2023-12-31")
    df = add_calendar_features(df)
    # Oct 1–14: not burning season
    early_oct = df[df["date"] < "2023-10-15"]
    assert (early_oct["burning_season_flag"] == 0).all()
    # Oct 15 onward: burning season
    late_oct = df[(df["date"] >= "2023-10-15") & (df["date"] <= "2023-11-30")]
    assert (late_oct["burning_season_flag"] == 1).all()


def test_static_city_features():
    """Each city must get valid elevation and population values."""
    df = build_date_spine(CITIES, "2023-01-01", "2023-01-07")
    df = add_static_city_features(df)
    assert "city_elevation_m" in df.columns
    assert "city_population_lakh" in df.columns
    assert df["city_elevation_m"].notna().all()
    assert df["city_population_lakh"].notna().all()
    assert (df["city_population_lakh"] > 0).all()


def test_grap_only_delhi():
    """GRAP and odd-even features must be 0 for non-Delhi cities."""
    df = build_date_spine(["Delhi", "Mumbai", "Bengaluru"], "2022-10-05", "2022-11-15")
    # Create minimal GRAP calendar
    grap_df = pd.DataFrame(
        {
            "date": ["2022-10-05", "2022-11-03"],
            "grap_stage": [1, 4],
            "odd_even_active": [0, 0],
        }
    )
    df = add_policy_features(df, grap_df, festival_days=set())
    non_delhi = df[df["city"] != "Delhi"]
    assert (non_delhi["grap_stage"] == 0).all(), "GRAP stage must be 0 for non-Delhi cities"
    assert (non_delhi["odd_even_active"] == 0).all()


def test_bsvi_flag():
    """BS-VI flag must be 1 on/after April 1, 2020."""
    df = build_date_spine(["Delhi"], "2020-03-25", "2020-04-10")
    df = add_policy_features(
        df, pd.DataFrame(columns=["date", "grap_stage", "odd_even_active"]), set()
    )
    before = df[df["date"] < "2020-04-01"]
    after = df[df["date"] >= "2020-04-01"]
    assert (before["bsvi_active"] == 0).all()
    assert (after["bsvi_active"] == 1).all()


def test_meteorological_derived():
    """Derived met flags must be binary."""
    df = pd.DataFrame(
        {
            "wind_speed_10m_mean": [2.0, 15.0, 1.5, 20.0],
            "relative_humidity_2m_mean": [85.0, 50.0, 90.0, 30.0],
            "precipitation_sum": [0.0, 0.0, 10.0, 0.0],
        }
    )
    df = add_meteorological_derived(df)
    for col in ["inversion_risk_flag", "fog_risk_flag", "rain_washout_flag"]:
        assert col in df.columns
        assert df[col].isin([0, 1]).all(), f"{col} must be binary"


def test_city_state_mapping_complete():
    """All 9 cities must have state mappings."""
    expected = [
        "Delhi",
        "Mumbai",
        "Bengaluru",
        "Chennai",
        "Kolkata",
        "Hyderabad",
        "Pune",
        "Ahmedabad",
        "Lucknow",
    ]
    for city in expected:
        assert city in CITY_STATE, f"No state mapping for {city}"
        assert city in CITY_COORDS, f"No coordinates for {city}"
