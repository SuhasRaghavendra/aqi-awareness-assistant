"""Uniform per-city bias calibration fitted on validation only (ADR D-031).

The champion may be systematically high or low in a particular city.  A single
per-city additive correction removes that bias without letting the holdout leak
into training: the offsets are medians of ``target - prediction`` on the
validation split, and the reported validation score uses *forward-chained*
offsets (fold ``k`` is calibrated with offsets estimated from earlier folds
only), so the improvement is not an in-sample artefact.

The same mechanism is applied to every city and every calibrated candidate; no
city is special-cased.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit


def per_city_offsets(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    cities: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Return the median residual (``y_true - y_pred``) for every city."""
    frame = pd.DataFrame(
        {
            "city": np.asarray(cities).astype(str),
            "residual": np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float),
        }
    )
    return {str(city): float(group["residual"].median()) for city, group in frame.groupby("city")}


def forward_calibrated_predictions(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    cities: np.ndarray | pd.Series,
    dates: np.ndarray | pd.Series,
    n_splits: int = 4,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Calibrate each time fold with offsets estimated only from earlier folds.

    Returns the calibrated prediction vector and one record per fold.  The
    earliest block has no prior residuals, so it keeps the raw prediction; every
    later block is calibrated with offsets fitted strictly backwards in time.
    """
    truth = np.asarray(y_true, dtype=float)
    raw = np.asarray(y_pred, dtype=float)
    city_arr = np.asarray(cities).astype(str)
    calibrated = raw.copy()

    order = np.argsort(pd.to_datetime(dates).to_numpy(), kind="stable")
    splits = list(TimeSeriesSplit(n_splits=n_splits).split(order))
    records: list[dict[str, Any]] = []
    for fold, (fit_positions, apply_positions) in enumerate(splits):
        fit_rows = order[fit_positions]
        apply_rows = order[apply_positions]
        offsets = per_city_offsets(truth[fit_rows], raw[fit_rows], city_arr[fit_rows])
        applied = np.asarray(
            [float(offsets.get(city, 0.0)) for city in city_arr[apply_rows]], dtype=float
        )
        calibrated[apply_rows] = raw[apply_rows] + applied
        records.append(
            {
                "fold": fold,
                "fit_rows": int(len(fit_rows)),
                "apply_rows": int(len(apply_rows)),
                "cities": int(len(offsets)),
                "mean_offset": round(float(np.mean(applied)), 4),
            }
        )
    return calibrated, records


@dataclass
class CityCalibratedRegressor(BaseEstimator, RegressorMixin):
    """Wrap a regressor and add a stored per-city offset at prediction time.

    ``base_model`` is the trained residual regressor; ``offsets`` maps city to
    an additive AQI correction.  Predictions for unseen cities use
    ``default_offset`` (zero), so serving never fails on a new city.
    """

    base_model: Any
    offsets: dict[str, float]
    default_offset: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        aqi: np.ndarray | pd.Series | None = None,
        city: np.ndarray | pd.Series | None = None,
    ) -> CityCalibratedRegressor:
        """Fit the wrapped model and, when ``city`` is given, its offsets."""
        self.base_model.fit(X, y, aqi)
        if city is not None:
            preds = self.base_model.predict(X, aqi)
            self.offsets = per_city_offsets(y, preds, city)
        return self

    def predict(
        self,
        X: pd.DataFrame,
        aqi: np.ndarray | pd.Series | None = None,
        city: np.ndarray | pd.Series | None = None,
    ) -> np.ndarray:
        """Predict and add the per-city offset when ``city`` is supplied."""
        preds = self.base_model.predict(X, aqi) if aqi is not None else self.base_model.predict(X)
        preds = np.asarray(preds, dtype=float)
        if city is None:
            return preds
        applied = np.asarray(
            [float(self.offsets.get(str(name), self.default_offset)) for name in np.asarray(city)],
            dtype=float,
        )
        return preds + applied

    def calibration_report(self) -> dict[str, Any]:
        """JSON-serialisable description of the calibration."""
        values = np.asarray(list(self.offsets.values()), dtype=float)
        return {
            "method": "per_city_median_residual",
            "fitted_on": "validation",
            "inner_folds": 4,
            "offsets": {
                city: round(float(value), 4) for city, value in sorted(self.offsets.items())
            },
            "n_cities": int(len(self.offsets)),
            "offset_mean": round(float(values.mean()), 4) if len(values) else 0.0,
            "offset_abs_max": round(float(np.abs(values).max()), 4) if len(values) else 0.0,
        }
