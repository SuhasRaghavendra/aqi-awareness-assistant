"""
src/models/baselines.py
-----------------------
Phase 6: Naive and heuristic baselines for Next-Day AQI (Task A)
and Next-Day CPCB AQI Category (Task B).

Baselines implemented:
1. PersistenceBaseline: predicts tomorrow's AQI = today's AQI (y_{t+1} = aqi_t).
2. WeeklySeasonalNaiveBaseline: predicts tomorrow's AQI using weekly recurrence
   (lag-7 AQI from features, representing the same day-of-week).
3. StationMonthMeanBaseline: predicts historical training mean for each (station, month).
4. MajorityClassBaseline: predicts the most frequent target category from training set.
5. RegressThenBucketClassifier: wraps any regressor and maps predicted AQI to CPCB category.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from src.features.aqi_utils import aqi_to_category


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """
    Persistence (Random Walk) Baseline.
    Predicts tomorrow's AQI as today's AQI:
        y_hat_{t+1} = aqi_t
    """

    def __init__(self, aqi_col: str = "aqi") -> None:
        self.aqi_col = aqi_col
        self.default_value_: float = 100.0

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series | None = None) -> PersistenceBaseline:
        if isinstance(X, pd.DataFrame) and self.aqi_col in X.columns:
            self.default_value_ = float(X[self.aqi_col].median())
        elif y is not None:
            self.default_value_ = float(np.nanmedian(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and self.aqi_col in X.columns:
            preds = X[self.aqi_col].fillna(self.default_value_).to_numpy(dtype=float)
        else:
            preds = np.full(len(X), self.default_value_, dtype=float)
        return preds


class WeeklySeasonalNaiveBaseline(BaseEstimator, RegressorMixin):
    """
    Weekly Seasonal Naive Baseline.
    Predicts tomorrow's AQI using the same day-of-week observation from the prior week.
    Uses 'aqi_lag_7' feature column (7 days prior), with fallback to today's 'aqi'.
    """

    def __init__(self, lag7_col: str = "aqi_lag_7", fallback_col: str = "aqi") -> None:
        self.lag7_col = lag7_col
        self.fallback_col = fallback_col
        self.default_value_: float = 100.0

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series | None = None) -> WeeklySeasonalNaiveBaseline:
        if isinstance(X, pd.DataFrame):
            if self.fallback_col in X.columns:
                self.default_value_ = float(X[self.fallback_col].median())
        elif y is not None:
            self.default_value_ = float(np.nanmedian(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.lag7_col in X.columns:
                preds = X[self.lag7_col].copy()
                if self.fallback_col in X.columns:
                    preds = preds.fillna(X[self.fallback_col])
                return preds.fillna(self.default_value_).to_numpy(dtype=float)
            elif self.fallback_col in X.columns:
                return X[self.fallback_col].fillna(self.default_value_).to_numpy(dtype=float)
        return np.full(len(X), self.default_value_, dtype=float)


class StationMonthMeanBaseline(BaseEstimator, RegressorMixin):
    """
    Station-Month Mean Baseline.
    Fits the mean AQI strictly on training data grouped by (location_id, month).
    Falls back to station mean, then global mean if unseen.
    """

    def __init__(
        self,
        station_col: str = "location_id",
        month_col: str = "month",
        date_col: str = "date",
    ) -> None:
        self.station_col = station_col
        self.month_col = month_col
        self.date_col = date_col
        self.station_month_means_: dict[tuple[int | str, int], float] = {}
        self.station_means_: dict[int | str, float] = {}
        self.global_mean_: float = 100.0

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> StationMonthMeanBaseline:
        df = X.copy()
        df["_target"] = np.asarray(y, dtype=float)

        if self.month_col not in df.columns and self.date_col in df.columns:
            df[self.month_col] = pd.to_datetime(df[self.date_col]).dt.month

        self.global_mean_ = float(df["_target"].mean())

        if self.station_col in df.columns and self.month_col in df.columns:
            grouped = df.groupby([self.station_col, self.month_col])["_target"].mean()
            self.station_month_means_ = {(st, m): float(val) for (st, m), val in grouped.items()}
            st_grouped = df.groupby(self.station_col)["_target"].mean()
            self.station_means_ = {st: float(val) for st, val in st_grouped.items()}

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        df = X.copy()
        if self.month_col not in df.columns and self.date_col in df.columns:
            df[self.month_col] = pd.to_datetime(df[self.date_col]).dt.month

        preds = []
        for _, row in df.iterrows():
            st = row.get(self.station_col, None)
            m = row.get(self.month_col, None)
            val = self.station_month_means_.get((st, m))
            if val is None:
                val = self.station_means_.get(st, self.global_mean_)
            preds.append(val)
        return np.array(preds, dtype=float)


class MajorityClassBaseline(BaseEstimator, ClassifierMixin):
    """
    Majority Class Classifier.
    Predicts the single most frequent CPCB category observed in training set.
    """

    def __init__(self) -> None:
        self.majority_class_: str = "Moderate"
        self.classes_: np.ndarray = np.array([])

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> MajorityClassBaseline:
        s = pd.Series(y).dropna()
        self.classes_ = np.unique(s)
        self.majority_class_ = str(s.mode().iloc[0]) if not s.empty else "Moderate"
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.majority_class_, dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        k = len(self.classes_)
        proba = np.zeros((n, k), dtype=float)
        if self.majority_class_ in self.classes_:
            idx = int(np.where(self.classes_ == self.majority_class_)[0][0])
            proba[:, idx] = 1.0
        return proba


class RegressThenBucketClassifier(BaseEstimator, ClassifierMixin):
    """
    Converts any regression model's continuous AQI predictions
    into CPCB category classifications using official breakpoints.
    """

    def __init__(self, regressor: BaseEstimator) -> None:
        self.regressor = regressor
        self.classes_: np.ndarray = np.array([
            "Good",
            "Satisfactory",
            "Moderate",
            "Poor",
            "Very Poor",
            "Severe",
        ])

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> RegressThenBucketClassifier:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        continuous_preds = self.regressor.predict(X)
        return np.array([aqi_to_category(float(p)) for p in continuous_preds], dtype=object)
