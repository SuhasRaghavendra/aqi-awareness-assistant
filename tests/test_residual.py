"""Tests for the persistence-anchored residual regressor (Phase 0.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.models.residual import ResidualRegressor


def test_residual_regressor_learns_delta_and_adds_aqi():
    X = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    aqi = np.array([100.0, 110.0, 120.0, 130.0])
    y = aqi + 5.0

    model = ResidualRegressor(DummyRegressor(strategy="constant", constant=5.0))
    model.fit(X, y, aqi)

    np.testing.assert_allclose(model.predict(X, aqi), y)


def test_residual_regressor_reads_aqi_column_when_not_passed():
    X = pd.DataFrame({"feature": [0.0, 1.0], "aqi": [50.0, 60.0]})
    y = np.array([52.0, 63.0])

    model = ResidualRegressor(DummyRegressor(strategy="constant", constant=0.0))
    model.fit(X, y)

    np.testing.assert_allclose(model.predict(X), X["aqi"].to_numpy())
