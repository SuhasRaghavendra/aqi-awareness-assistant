"""
tests/test_build_features.py
------------------------------
Tests for the root-cause feature engineering pipeline.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingest_and_merge import build_date_spine, add_calendar_features, add_static_city_features
from src.features.build_features import (
    make_target,
    add_fire_features,
    add_aqi_history_features,
    add_interaction_features,
    add_city_onehot,
    add_season_onehot,
    split_chronological,
    NON_FEATURE_COLS,
)
from src.features.aqi_utils import aqi_to_category


CITIES = ["Delhi", "Mumbai"]


def _make_sample_df(n_days: int = 100) -> pd.DataFrame:
    """Build a minimal sample merged dataframe for feature tests."""
    df = build_date_spine(CITIES, "2022-01-01",
                          pd.Timestamp("2022-01-01") + pd.Timedelta(days=n_days - 1))
    df = add_calendar_features(df)
    df = add_static_city_features(df)
    df["aqi"] = np.random.default_rng(42).uniform(50, 300, len(df))
    df["fire_count_500km"] = np.random.default_rng(42).integers(0, 200, len(df)).astype(float)
    df["fire_frp_sum_500km"] = np.random.default_rng(43).uniform(0, 5000, len(df))
    df["wind_speed_10m_mean"] = np.random.default_rng(44).uniform(1, 30, len(df))
    df["relative_humidity_2m_mean"] = np.random.default_rng(45).uniform(20, 90, len(df))
    df["temperature_2m_mean"] = np.random.default_rng(46).uniform(10, 45, len(df))
    df["precipitation_sum"] = np.random.default_rng(47).uniform(0, 20, len(df))
    df["inversion_risk_flag"] = np.random.default_rng(48).integers(0, 2, len(df))
    df["burning_season_flag"] = df["burning_season_flag"] if "burning_season_flag" in df.columns else 0
    df["festival_flag"] = 0
    df["grap_stage"] = 0
    df["diesel_frac"] = 0.35
    df["cng_frac"] = 0.08
    df["ev_frac"] = 0.01
    df["vehicles_per_lakh"] = 2500.0
    df["iip_manufacturing_index"] = 130.0
    return df


def test_make_target_no_leakage():
    """Target AQI must be tomorrow's AQI — no leakage from same-day."""
    df = _make_sample_df()
    df_with_target = make_target(df)
    for city in CITIES:
        city_df = df_with_target[df_with_target["city"] == city].sort_values("date").reset_index(drop=True)
        # For any row i, target_aqi[i] should equal aqi[i+1]
        for i in range(len(city_df) - 1):
            if pd.notna(city_df.loc[i, "target_aqi"]) and pd.notna(city_df.loc[i+1, "aqi"]):
                assert abs(city_df.loc[i, "target_aqi"] - city_df.loc[i+1, "aqi"]) < 0.01, \
                    f"Target leakage at row {i}"


def test_make_target_category():
    """Target category must match CPCB breakpoints for target_aqi."""
    df = _make_sample_df()
    df_with_target = make_target(df)
    valid = df_with_target.dropna(subset=["target_aqi", "target_category"])
    for _, row in valid.head(50).iterrows():
        expected = aqi_to_category(row["target_aqi"])
        assert row["target_category"] == expected, \
            f"Category mismatch: AQI={row['target_aqi']}, expected={expected}, got={row['target_category']}"


def test_fire_features_no_future_leakage():
    """Fire lag features must only use past values (shift >= 1)."""
    df = _make_sample_df()
    df = make_target(df)
    df = add_fire_features(df, lags=[1, 3, 7], rolling_windows=[7])
    # fire_count_500km_lag_1 should be the previous day's count (no same-day info)
    for city in CITIES:
        city_df = df[df["city"] == city].sort_values("date").reset_index(drop=True)
        for i in range(1, min(10, len(city_df))):
            expected_lag1 = city_df.loc[i-1, "fire_count_500km"]
            actual_lag1 = city_df.loc[i, "fire_count_500km_lag_1"]
            if pd.notna(expected_lag1) and pd.notna(actual_lag1):
                assert abs(expected_lag1 - actual_lag1) < 0.01, \
                    f"Fire lag_1 mismatch at row {i}"


def test_city_onehot():
    """Each city row must have exactly 1 one-hot flag set."""
    df = _make_sample_df()
    df = add_city_onehot(df, CITIES)
    onehot_cols = [f"city_{c.lower()}" for c in CITIES]
    assert set(onehot_cols).issubset(df.columns)
    row_sums = df[onehot_cols].sum(axis=1)
    assert (row_sums == 1).all(), "Each row must have exactly one city flag set"


def test_season_onehot():
    """Season one-hot columns must sum to exactly 1 per row."""
    df = _make_sample_df()
    df = add_season_onehot(df)
    season_cols = [c for c in df.columns if c.startswith("season_")]
    assert len(season_cols) == 4
    assert (df[season_cols].sum(axis=1) == 1).all()


def test_chronological_split_no_overlap():
    """Train/val/test splits must be strictly chronological with no overlap."""
    df = _make_sample_df(n_days=400)
    df = make_target(df)
    df_train, df_val, df_test = split_chronological(df, "2022-06-30", "2022-10-31")
    train_dates = set(df_train["date"])
    val_dates = set(df_val["date"])
    test_dates = set(df_test["date"])
    assert len(train_dates & val_dates) == 0, "Train/val date overlap"
    assert len(val_dates & test_dates) == 0, "Val/test date overlap"
    assert len(train_dates & test_dates) == 0, "Train/test date overlap"
    # Verify ordering
    assert max(df_train["date"]) <= "2022-06-30"
    assert min(df_val["date"]) > "2022-06-30"
    assert max(df_val["date"]) <= "2022-10-31"
    assert min(df_test["date"]) > "2022-10-31"


def test_interaction_features_exist():
    """Interaction features must be created without errors."""
    df = _make_sample_df()
    df = add_interaction_features(df)
    assert "fire_x_low_wind" in df.columns
    assert "winter_stagnation_index" in df.columns
    assert "diesel_vehicle_intensity" in df.columns
    assert (df["diesel_vehicle_intensity"] >= 0).all()

