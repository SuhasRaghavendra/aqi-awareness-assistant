"""Rolling per-city observation history for the inference service.

The champion model needs lagged and rolling AQI/fire/weather features.  The
service keeps the most recent raw observations for every city, seeded from the
bundle's 45-day history seed and extended by every accepted request.  A
per-request ``history`` array can be supplied for stateless deployments.

Only the columns in ``HISTORY_COLUMNS`` are retained: the full 111-column
training row is not needed to rebuild lags, and trimming it keeps appends (and
therefore request latency) cheap.
"""

from __future__ import annotations

from threading import RLock

import numpy as np
import pandas as pd

from src.features.serving_features import HISTORY_COLUMNS

MAX_HISTORY_DAYS = 90


def _trim(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns needed for feature reconstruction."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    keep = [column for column in HISTORY_COLUMNS if column in frame.columns]
    return frame.loc[:, keep].copy()


class HistoryStore:
    """Thread-safe rolling window of raw observations, keyed by city."""

    def __init__(self, seed: pd.DataFrame | None = None, max_days: int = MAX_HISTORY_DAYS):
        self.max_days = max_days
        self._frames: dict[str, pd.DataFrame] = {}
        self._lock = RLock()
        self.reset(seed)

    def reset(self, seed: pd.DataFrame | None) -> None:
        """Replace the store with the bundle seed (or clear it when ``None``)."""
        with self._lock:
            self._frames = {}
            if seed is None or seed.empty:
                return
            frame = _trim(seed)
            frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
            for city, group in frame.groupby("city"):
                self._frames[str(city)] = (
                    group.sort_values("date").tail(self.max_days).reset_index(drop=True)
                )

    def history_before(self, city: str, date: str) -> pd.DataFrame:
        """Return all stored rows for *city* strictly before *date*."""
        with self._lock:
            frame = self._frames.get(str(city))
            if frame is None:
                return pd.DataFrame()
            return frame[frame["date"] < str(date)].copy()

    def history_days(self, city: str, date: str) -> int:
        """Number of distinct observed days before *date*."""
        frame = self.history_before(city, date)
        return int(frame["date"].nunique()) if not frame.empty else 0

    def last_date(self, city: str, date: str | None = None) -> str | None:
        """Most recent stored date for *city* (optionally before *date*)."""
        with self._lock:
            frame = self._frames.get(str(city))
            if frame is None or frame.empty:
                return None
            if date is not None:
                frame = frame[frame["date"] < str(date)]
            if frame.empty:
                return None
            return str(frame["date"].iloc[-1])

    def append(self, row: dict) -> None:
        """Insert/replace one observation and trim the rolling window."""
        city = str(row["city"])
        date = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")

        with self._lock:
            frame = self._frames.get(city)
            if frame is None or frame.empty:
                frame = pd.DataFrame(columns=HISTORY_COLUMNS)
            elif "date" in frame.columns:
                frame = frame[frame["date"] != date]

            record = {
                column: row.get(column, np.nan)
                for column in frame.columns
                if column not in ("city", "date")
            }
            record["city"] = city
            record["date"] = date
            if frame.empty:
                updated = pd.DataFrame([record])
            else:
                updated = pd.concat([frame, pd.DataFrame([record])], ignore_index=True)
            self._frames[city] = (
                updated.sort_values("date").tail(self.max_days).reset_index(drop=True)
            )

    def snapshot(self) -> dict[str, int]:
        """Return the number of stored rows per city (for /ready and metadata)."""
        with self._lock:
            return {
                city: int(len(frame))
                for city, frame in self._frames.items()
                if frame is not None and not frame.empty
            }


def frame_from_records(records: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from a stateless ``history`` request payload."""
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(record) for record in records])
    if "date" not in frame.columns:
        raise ValueError("Each history row must include a date")
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame.sort_values(["city", "date"]).reset_index(drop=True)


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))
