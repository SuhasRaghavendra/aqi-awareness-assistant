"""
src/models/residual.py
----------------------
Persistence-anchored (residual) regression for next-day AQI.

Instead of predicting the level of tomorrow's AQI directly, the residual
variant predicts the *change* from today's observation:

    target   = target_aqi - aqi_t
    y_hat_t+1 = aqi_t + model.predict(X)

This gives the learner a persistence anchor and lets it model only the
day-over-day deviation.  It is used by Phase 0.4 for the Ridge and LightGBM
candidate models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class ResidualRegressor(BaseEstimator, RegressorMixin):
    """Wrap a regressor so it learns ``target_aqi - aqi`` and adds ``aqi`` back."""

    def __init__(self, base_model: BaseEstimator, aqi_col: str = "aqi") -> None:
        self.base_model = base_model
        self.aqi_col = aqi_col

    @staticmethod
    def _resolve_aqi(
        X: pd.DataFrame | np.ndarray,
        aqi: np.ndarray | pd.Series | None,
        aqi_col: str,
    ) -> np.ndarray:
        if aqi is not None:
            return np.asarray(aqi, dtype=float)
        if isinstance(X, pd.DataFrame) and aqi_col in X.columns:
            return X[aqi_col].to_numpy(dtype=float)
        raise ValueError(
            f"aqi must be supplied explicitly unless X is a DataFrame containing '{aqi_col}'"
        )

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series,
        aqi: np.ndarray | pd.Series | None = None,
    ) -> ResidualRegressor:
        aqi_arr = self._resolve_aqi(X, aqi, self.aqi_col)
        residual_target = np.asarray(y, dtype=float) - aqi_arr
        self.base_model.fit(X, residual_target)
        return self

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        aqi: np.ndarray | pd.Series | None = None,
    ) -> np.ndarray:
        aqi_arr = self._resolve_aqi(X, aqi, self.aqi_col)
        return aqi_arr + np.asarray(self.base_model.predict(X), dtype=float)
