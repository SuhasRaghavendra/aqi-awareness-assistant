"""
tests/test_baselines.py
-----------------------
Unit tests for Phase 6 baseline models:
- PersistenceBaseline
- WeeklySeasonalNaiveBaseline
- StationMonthMeanBaseline
- MajorityClassBaseline
- RegressThenBucketClassifier
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import (
    MajorityClassBaseline,
    PersistenceBaseline,
    RegressThenBucketClassifier,
    StationMonthMeanBaseline,
    WeeklySeasonalNaiveBaseline,
)


def test_persistence_baseline() -> None:
    X = pd.DataFrame({"aqi": [50.0, 150.0, 250.0, np.nan]})
    y = np.array([55.0, 140.0, 260.0, 100.0])

    model = PersistenceBaseline(aqi_col="aqi")
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == 4
    assert preds[0] == 50.0
    assert preds[1] == 150.0
    assert preds[2] == 250.0
    # Fallback to median (150.0) for NaN
    assert preds[3] == 150.0


def test_weekly_seasonal_naive_baseline() -> None:
    X = pd.DataFrame({
        "aqi_lag_7": [80.0, np.nan, 210.0],
        "aqi": [90.0, 100.0, 200.0],
    })
    y = np.array([85.0, 105.0, 215.0])

    model = WeeklySeasonalNaiveBaseline(lag7_col="aqi_lag_7", fallback_col="aqi")
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == 3
    assert preds[0] == 80.0
    # Row 1 has NaN in lag-7, so falls back to same-day aqi
    assert preds[1] == 100.0
    assert preds[2] == 210.0


def test_station_month_mean_baseline() -> None:
    train_df = pd.DataFrame({
        "location_id": [101, 101, 102, 102],
        "month": [1, 1, 1, 2],
    })
    y_train = np.array([100.0, 120.0, 200.0, 50.0])

    model = StationMonthMeanBaseline(station_col="location_id", month_col="month")
    model.fit(train_df, y_train)

    test_df = pd.DataFrame({
        "location_id": [101, 102, 101, 999],  # 999 is unseen station
        "month": [1, 2, 2, 1],  # (101, 2) is unseen pair
    })
    preds = model.predict(test_df)

    assert len(preds) == 4
    # (101, 1) -> mean of [100, 120] = 110.0
    assert preds[0] == 110.0
    # (102, 2) -> 50.0
    assert preds[1] == 50.0
    # (101, 2) unseen pair -> fallback to station 101 mean (110.0)
    assert preds[2] == 110.0
    # 999 unseen station -> fallback to global mean: (100 + 120 + 200 + 50) / 4 = 117.5
    assert preds[3] == 117.5


def test_majority_class_baseline() -> None:
    X = pd.DataFrame({"feat": [1, 2, 3, 4, 5]})
    y = pd.Series(["Moderate", "Moderate", "Moderate", "Poor", "Satisfactory"])

    model = MajorityClassBaseline()
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == 5
    assert all(p == "Moderate" for p in preds)

    proba = model.predict_proba(X)
    assert proba.shape == (5, 3)  # 3 unique classes: Moderate, Poor, Satisfactory
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_regress_then_bucket_classifier() -> None:
    class MockRegressor:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.array([25.0, 75.0, 150.0, 250.0, 350.0, 450.0])

    mock_reg = MockRegressor()
    clf = RegressThenBucketClassifier(regressor=mock_reg)
    clf.fit(pd.DataFrame(), np.array([]))

    preds = clf.predict(pd.DataFrame({"dummy": range(6)}))
    expected = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    assert list(preds) == expected
