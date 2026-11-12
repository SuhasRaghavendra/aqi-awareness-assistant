"""Serving-side feature construction shared with the training pipeline.

The functions in this module are the single source of truth for the engineered
feature set.  ``build_features.py`` imports them for offline training and
``src.serving.bundle`` / ``src.service`` import them for online inference, so a
model never sees a different feature definition at serve time than it saw
during training.

Only lightweight dependencies live here (pandas, numpy and the config loaders);
no plotting or MLflow imports, so the FastAPI image does not need the training
stack.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import date as date_type
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.ingest_and_merge import (
    CITY_COORDS,
    CITY_ELEVATION_M,
    CITY_POPULATION_LAKH,
    add_calendar_features,
    add_meteorological_derived,
    add_policy_features,
    add_static_city_features,
    load_festival_dates,
    load_grap_calendar,
)

CATEGORY_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
AQI_BANDS = [50.0, 100.0, 200.0, 300.0, 400.0]

# Raw observation fields accepted from a client (today's city-day observations).
OBSERVATION_COLUMNS = [
    "aqi",
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "wind_direction_10m_dominant",
    "precipitation_sum",
    "surface_pressure_mean",
    "fire_count_500km",
    "fire_frp_sum_500km",
    "fire_dist_min_km",
    "grap_stage",
    "festival_flag",
]

# Slow-moving fields carried forward from the most recent observation.
STATIC_COLUMNS = [
    "diesel_frac",
    "cng_frac",
    "ev_frac",
    "two_wheeler_frac",
    "vehicles_per_lakh",
    "iip_manufacturing_index",
]

# Columns the rolling history must retain to rebuild lag/rolling features.
HISTORY_COLUMNS = [
    "date",
    "city",
    "aqi",
    "fire_count_500km",
    "fire_frp_sum_500km",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "precipitation_sum",
    "inversion_risk_flag",
    "grap_stage",
    *STATIC_COLUMNS,
]


# ---------------------------------------------------------------------------
# Feature engineering primitives (moved from build_features.py)
# ---------------------------------------------------------------------------


def add_fire_features(
    df: pd.DataFrame,
    lags: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """Lag and rolling features for fire events (per city)."""
    fire_cols = ["fire_count_500km", "fire_frp_sum_500km"]

    for _city, grp_idx in df.groupby("city").groups.items():
        sub = df.loc[grp_idx].sort_values("date")
        sorted_idx = sub.index

        grp = sub.set_index("date")
        full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq="D")
        grp = grp.reindex(full_idx)

        for col in fire_cols:
            if col not in grp.columns:
                continue
            for lag in lags:
                vals = grp[col].shift(lag).reindex(sub["date"]).values
                df.loc[sorted_idx, f"{col}_lag_{lag}"] = vals
            for w in rolling_windows:
                mean_vals = (
                    grp[col].shift(1).rolling(w, min_periods=1).mean().reindex(sub["date"]).values
                )
                max_vals = (
                    grp[col].shift(1).rolling(w, min_periods=1).max().reindex(sub["date"]).values
                )
                df.loc[sorted_idx, f"{col}_roll_mean_{w}d"] = mean_vals
                df.loc[sorted_idx, f"{col}_roll_max_{w}d"] = max_vals

    if "burning_season_flag" in df.columns and "fire_count_500km" in df.columns:
        df["fire_burning_season"] = df["fire_count_500km"] * df["burning_season_flag"]

    return df


def add_aqi_history_features(
    df: pd.DataFrame,
    lags: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """AQI lag/rolling features using the same causal logic as training (ADR D-019)."""
    for _city, grp_idx in df.groupby("city").groups.items():
        sub = df.loc[grp_idx].sort_values("date")
        sorted_idx = sub.index

        grp = sub.set_index("date")
        full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq="D")
        grp = grp.reindex(full_idx)

        if "aqi" not in grp.columns:
            continue

        for lag in lags:
            df.loc[sorted_idx, f"aqi_lag_{lag}"] = grp["aqi"].shift(lag).reindex(sub["date"]).values
        for w in rolling_windows:
            df.loc[sorted_idx, f"aqi_roll_mean_{w}d"] = (
                grp["aqi"].shift(1).rolling(w, min_periods=1).mean().reindex(sub["date"]).values
            )
            df.loc[sorted_idx, f"aqi_roll_std_{w}d"] = (
                grp["aqi"].shift(1).rolling(w, min_periods=1).std().reindex(sub["date"]).values
            )
        df.loc[sorted_idx, "aqi_diff_1d"] = grp["aqi"].diff(1).reindex(sub["date"]).values
        df.loc[sorted_idx, "aqi_diff_7d"] = grp["aqi"].diff(7).reindex(sub["date"]).values

    return df


def add_weather_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Lag yesterday's weather (most impactful meteorological features)."""
    weather_lag_cols = [
        "temperature_2m_mean",
        "wind_speed_10m_mean",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "inversion_risk_flag",
    ]
    for col in weather_lag_cols:
        if col not in df.columns:
            continue
        for lag in lags[:2]:
            df[f"{col}_lag_{lag}"] = df.groupby("city")[col].shift(lag)

    if "wind_speed_10m_mean" in df.columns:
        df["wind_roll_mean_3d"] = df.groupby("city")["wind_speed_10m_mean"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        df["wind_roll_mean_7d"] = df.groupby("city")["wind_speed_10m_mean"].transform(
            lambda s: s.shift(1).rolling(7, min_periods=1).mean()
        )

    if "precipitation_sum" in df.columns:
        df["precip_roll_sum_7d"] = df.groupby("city")["precipitation_sum"].transform(
            lambda s: s.shift(1).rolling(7, min_periods=1).sum()
        )

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal interaction features that model how root causes compound."""
    if "fire_count_500km" in df.columns and "wind_speed_10m_mean" in df.columns:
        df["fire_x_low_wind"] = df["fire_count_500km"] / (df["wind_speed_10m_mean"].clip(lower=0.5))

    if "inversion_risk_flag" in df.columns and "fire_count_500km" in df.columns:
        df["inversion_x_fire"] = df["inversion_risk_flag"] * df["fire_count_500km"]

    if "festival_flag" in df.columns and "burning_season_flag" in df.columns:
        df["festival_burning_overlap"] = df["festival_flag"] * df["burning_season_flag"]

    if "vehicles_per_lakh" in df.columns and "diesel_frac" in df.columns:
        df["diesel_vehicle_intensity"] = df["vehicles_per_lakh"] * df["diesel_frac"]

    if all(
        c in df.columns
        for c in ["temperature_2m_mean", "relative_humidity_2m_mean", "wind_speed_10m_mean"]
    ):
        df["winter_stagnation_index"] = (
            (df["relative_humidity_2m_mean"].clip(0, 100) / 100)
            * (1 / df["wind_speed_10m_mean"].clip(lower=0.5))
            * (df["temperature_2m_mean"] < 15).astype(float)
        )

    return df


def add_city_onehot(df: pd.DataFrame, cities: list[str]) -> pd.DataFrame:
    """One-hot encode city."""
    for city in cities:
        df[f"city_{city.lower().replace(' ', '_')}"] = (df["city"] == city).astype(int)
    return df


def add_season_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode season."""
    if "season" in df.columns:
        for season in ["winter", "summer", "monsoon", "post_monsoon"]:
            df[f"season_{season}"] = (df["season"] == season).astype(int)
    return df


def add_grap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flags from GRAP stage + lag (yesterday's restriction stage)."""
    if "grap_stage" in df.columns:
        df["grap_active"] = (df["grap_stage"] > 0).astype(int)
        df["grap_severe"] = (df["grap_stage"] >= 3).astype(int)
        df["grap_lag_1"] = df.groupby("city")["grap_stage"].shift(1)
    return df


# ---------------------------------------------------------------------------
# Shared feature-frame builder
# ---------------------------------------------------------------------------


def build_feature_frame(
    panel: pd.DataFrame,
    feature_names: list[str],
    params: Mapping[str, Any],
    train_medians: Mapping[str, float],
) -> pd.DataFrame:
    """Rebuild the full Phase 0 feature set from a raw city-day panel slice.

    Both the offline audit (``model_risk.rebuild_features``) and the online
    service call this function.  ``feature_names`` and ``train_medians`` come
    from the bundle, never from a hard-coded count.
    """
    if panel.empty:
        raise ValueError("Cannot build features from an empty panel")

    df = panel.copy()
    # Keep dates as datetime64: the lag/rolling helpers reindex onto a
    # DatetimeIndex, so string dates would silently produce all-NaN lags.
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    feat_cfg = params.get("features", {})
    aqi_lags = list(feat_cfg.get("aqi_lags", [1, 3, 7, 14]))
    fire_lags = list(feat_cfg.get("fire_lags", [1, 3, 7]))
    rolling_windows = list(feat_cfg.get("rolling_windows", [7, 14, 30]))
    cities = list(params.get("data", {}).get("cities", sorted(df["city"].unique())))

    df = add_fire_features(df, fire_lags, rolling_windows)
    df = add_aqi_history_features(df, aqi_lags, rolling_windows)
    df = add_weather_lag_features(df, aqi_lags[:2])
    df = add_interaction_features(df)
    df = add_city_onehot(df, cities)
    df = add_season_onehot(df)
    df = add_grap_features(df)

    for name in feature_names:
        if name not in df.columns:
            df[name] = np.nan
    df[feature_names] = df[feature_names].fillna(pd.Series(dict(train_medians)))
    return df


def _scalar(value: Any) -> float:
    """Coerce a scalar to float, mapping missing values to NaN."""
    return float(value) if pd.notna(value) else float("nan")


def _lag_value(array: np.ndarray, lag: int) -> float:
    index = len(array) - 1 - lag
    return float(array[index]) if index >= 0 else float("nan")


def _rolling_value(array: np.ndarray, window: int, kind: str) -> float:
    """Trailing-window aggregate excluding the target day (shift(1) semantics)."""
    stop = len(array) - 1
    start = max(0, stop - window)
    valid = array[start:stop]
    valid = valid[~np.isnan(valid)]
    if valid.size == 0:
        return float("nan")
    if kind == "mean":
        return float(valid.mean())
    if kind == "max":
        return float(valid.max())
    if kind == "sum":
        return float(valid.sum())
    return float(valid.std(ddof=1)) if valid.size > 1 else float("nan")


def build_feature_record(
    history: pd.DataFrame | None,
    observation: Mapping[str, Any],
    feature_names: list[str],
    params: Mapping[str, Any],
    train_medians: Mapping[str, float],
) -> dict[str, Any]:
    """Assemble one feature row as a dict (no DataFrame construction).

    ``history`` holds the previous days (oldest first); ``observation`` is the
    already-enriched target day.  This is the serving hot path: it mirrors
    :func:`build_feature_frame` but avoids hundreds of per-column pandas
    operations, keeping p95 single-row latency under 50 ms.
    """
    obs = dict(observation)
    target = pd.Timestamp(obs["date"])
    city = str(obs["city"])

    hist = history if history is not None else pd.DataFrame()
    if not hist.empty:
        hist = hist.copy()
        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist.sort_values("date").reset_index(drop=True)

    feat_cfg = params.get("features", {})
    aqi_lags = list(feat_cfg.get("aqi_lags", [1, 3, 7, 14]))
    fire_lags = list(feat_cfg.get("fire_lags", [1, 3, 7]))
    rolling_windows = list(feat_cfg.get("rolling_windows", [7, 14, 30]))
    cities = list(params.get("data", {}).get("cities", sorted({city})))

    start = hist["date"].min() if not hist.empty else target
    full_idx = pd.date_range(start, target, freq="D")

    def aligned_array(column: str) -> np.ndarray:
        """Values on a gap-free daily calendar (NaN on missing days)."""
        array = np.full(len(full_idx), np.nan)
        if not hist.empty and column in hist.columns:
            positions = full_idx.get_indexer(hist["date"])
            values = hist[column].to_numpy(dtype=float)
            valid = positions >= 0
            array[positions[valid]] = values[valid]
        array[-1] = _scalar(obs.get(column))
        return array

    def row_array(column: str) -> np.ndarray:
        """Values on the observed rows only (matches groupby(...).shift)."""
        parts = []
        if not hist.empty and column in hist.columns:
            parts.append(hist[column].to_numpy(dtype=float))
        parts.append(np.array([_scalar(obs.get(column))]))
        return np.concatenate(parts)

    values: dict[str, Any] = {}

    for column in ("fire_count_500km", "fire_frp_sum_500km"):
        if not hist.empty and column not in hist.columns:
            continue
        array = aligned_array(column)
        for lag in fire_lags:
            values[f"{column}_lag_{lag}"] = _lag_value(array, lag)
        for window in rolling_windows:
            values[f"{column}_roll_mean_{window}d"] = _rolling_value(array, window, "mean")
            values[f"{column}_roll_max_{window}d"] = _rolling_value(array, window, "max")

    aqi_array = aligned_array("aqi")
    for lag in aqi_lags:
        values[f"aqi_lag_{lag}"] = _lag_value(aqi_array, lag)
    for window in rolling_windows:
        values[f"aqi_roll_mean_{window}d"] = _rolling_value(aqi_array, window, "mean")
        values[f"aqi_roll_std_{window}d"] = _rolling_value(aqi_array, window, "std")
    values["aqi_diff_1d"] = aqi_array[-1] - _lag_value(aqi_array, 1)
    values["aqi_diff_7d"] = aqi_array[-1] - _lag_value(aqi_array, 7)

    for column in [
        "temperature_2m_mean",
        "wind_speed_10m_mean",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "inversion_risk_flag",
    ]:
        if not hist.empty and column not in hist.columns:
            continue
        array = row_array(column)
        for lag in aqi_lags[:2]:
            values[f"{column}_lag_{lag}"] = _lag_value(array, lag)

    wind_array = row_array("wind_speed_10m_mean")
    values["wind_roll_mean_3d"] = _rolling_value(wind_array, 3, "mean")
    values["wind_roll_mean_7d"] = _rolling_value(wind_array, 7, "mean")
    values["precip_roll_sum_7d"] = _rolling_value(row_array("precipitation_sum"), 7, "sum")

    # Non-lag features depend only on the target day itself.
    row = obs
    row.update(values)

    fire = _scalar(row.get("fire_count_500km"))
    wind = _scalar(row.get("wind_speed_10m_mean"))
    inversion = _scalar(row.get("inversion_risk_flag"))
    burning = _scalar(row.get("burning_season_flag"))
    festival = _scalar(row.get("festival_flag"))
    vehicles = _scalar(row.get("vehicles_per_lakh"))
    diesel = _scalar(row.get("diesel_frac"))
    temperature = _scalar(row.get("temperature_2m_mean"))
    humidity = _scalar(row.get("relative_humidity_2m_mean"))

    row["fire_burning_season"] = fire * burning
    row["fire_x_low_wind"] = fire / max(wind, 0.5) if pd.notna(wind) else float("nan")
    row["inversion_x_fire"] = inversion * fire
    row["festival_burning_overlap"] = festival * burning
    row["diesel_vehicle_intensity"] = vehicles * diesel
    row["winter_stagnation_index"] = (
        (min(max(humidity, 0.0), 100.0) / 100.0)
        * (1.0 / max(wind, 0.5))
        * (1.0 if pd.notna(temperature) and temperature < 15 else 0.0)
        if pd.notna(humidity) and pd.notna(wind)
        else float("nan")
    )

    for name in cities:
        row[f"city_{name.lower().replace(' ', '_')}"] = int(city == name)
    for season in ("winter", "summer", "monsoon", "post_monsoon"):
        row[f"season_{season}"] = int(row.get("season") == season)

    grap = _scalar(row.get("grap_stage"))
    row["grap_active"] = int(grap > 0) if pd.notna(grap) else 0
    row["grap_severe"] = int(grap >= 3) if pd.notna(grap) else 0
    row["grap_lag_1"] = _lag_value(row_array("grap_stage"), 1)

    record: dict[str, Any] = {name: _scalar(row.get(name, float("nan"))) for name in feature_names}
    for name, value in record.items():
        if pd.isna(value):
            record[name] = float(train_medians.get(name, 0.0))

    out = {"city": city, "date": target, **record}
    out["aqi"] = _scalar(row.get("aqi"))
    return out


def build_feature_row(
    panel: pd.DataFrame,
    feature_names: list[str],
    params: Mapping[str, Any],
    train_medians: Mapping[str, float],
) -> pd.DataFrame:
    """DataFrame wrapper around :func:`build_feature_record` for a raw panel."""
    if panel.empty:
        raise ValueError("Cannot build features from an empty panel")
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    observation = df.iloc[-1].to_dict()
    record = build_feature_record(df.iloc[:-1], observation, feature_names, params, train_medians)
    return pd.DataFrame([record])


# ---------------------------------------------------------------------------
# Observation -> engineered row(s)
# ---------------------------------------------------------------------------


def build_grap_lookup(grap_calendar: pd.DataFrame | None) -> dict[str, tuple[int, int]]:
    """Index the GRAP calendar by ISO date for O(1) serving lookups."""
    if grap_calendar is None or grap_calendar.empty:
        return {}
    frame = grap_calendar.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return {
        str(row.date): (int(row.grap_stage), int(row.odd_even_active))
        for row in frame.itertuples(index=False)
    }


def _season_for_month(month: int) -> str:
    if month in (11, 12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "summer"
    if month in (6, 7, 8, 9):
        return "monsoon"
    return "post_monsoon"


def build_observation_record(
    record: Mapping[str, Any],
    *,
    grap_lookup: Mapping[str, tuple[int, int]] | None = None,
    festival_days: Iterable[str] = (),
    static_reference: Mapping[str, Any] | None = None,
    train_medians: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Derive one engineered city-day row without pandas overhead.

    This is the scalar twin of :func:`build_observation_frame`; a test asserts
    both produce identical columns and values so the fast serving path cannot
    drift from the offline pipeline.
    """
    data = dict(record)
    city = str(data["city"])
    raw_date = data["date"]
    if isinstance(raw_date, datetime):
        day_value = raw_date.date()
    elif isinstance(raw_date, date_type):
        day_value = raw_date
    elif isinstance(raw_date, str):
        day_value = date_type.fromisoformat(raw_date[:10])
    else:
        day_value = pd.Timestamp(raw_date).date()
    date_str = day_value.isoformat()
    data["date"] = date_str

    grap_override = data.get("grap_stage")
    festival_override = data.get("festival_flag")

    for col in OBSERVATION_COLUMNS:
        data.setdefault(col, np.nan)
        if data[col] is None:
            data[col] = np.nan

    if pd.isna(data.get("fire_dist_min_km")):
        data["fire_dist_min_km"] = 999.0

    for col in STATIC_COLUMNS:
        value = data.get(col)
        if value is None or pd.isna(value):
            reference = (static_reference or {}).get(col)
            if reference is not None and pd.notna(reference):
                value = float(reference)
            elif train_medians is not None:
                value = float(train_medians.get(col, 0.0))
        data[col] = value

    # Meteorological derived flags (mirrors add_meteorological_derived).
    wind = data.get("wind_speed_10m_mean")
    wind = 5.0 if wind is None or pd.isna(wind) else float(wind)
    humidity = data.get("relative_humidity_2m_mean")
    humidity = 50.0 if humidity is None or pd.isna(humidity) else float(humidity)
    rain = data.get("precipitation_sum")
    rain = 0.0 if rain is None or pd.isna(rain) else float(rain)
    data["inversion_risk_flag"] = int(wind < 5.0 and humidity > 70.0)
    data["fog_risk_flag"] = int(humidity > 85.0 and wind < 3.0)
    data["rain_washout_flag"] = int(rain > 5.0)

    # Calendar features (mirrors add_calendar_features).
    month = day_value.month
    day = day_value.day
    day_of_week = day_value.weekday()
    season = _season_for_month(month)
    data.update(
        {
            "day_of_week": day_of_week,
            "month": month,
            "day_of_year": day_value.timetuple().tm_yday,
            "year": day_value.year,
            "weekend_flag": int(day_of_week >= 5),
            "burning_season_flag": int((month == 10 and day >= 15) or month == 11),
            "season": season,
            "month_sin": math.sin(2 * math.pi * month / 12),
            "month_cos": math.cos(2 * math.pi * month / 12),
            "dow_sin": math.sin(2 * math.pi * day_of_week / 7),
            "dow_cos": math.cos(2 * math.pi * day_of_week / 7),
        }
    )

    # Static city attributes (mirrors add_static_city_features).
    coords = CITY_COORDS.get(city)
    data["city_lat"] = coords[0] if coords else np.nan
    data["city_lon"] = coords[1] if coords else np.nan
    data["city_elevation_m"] = CITY_ELEVATION_M.get(city, np.nan)
    data["city_population_lakh"] = CITY_POPULATION_LAKH.get(city, np.nan)

    # Policy features (mirrors add_policy_features).
    stage, odd_even = (grap_lookup or {}).get(date_str, (0, 0))
    if city != "Delhi":
        stage, odd_even = 0, 0
    if grap_override is not None and not pd.isna(grap_override):
        stage = int(grap_override) if city == "Delhi" else 0
    data["grap_stage"] = int(stage)
    data["odd_even_active"] = int(odd_even)
    data["bsvi_active"] = int(day_value >= date_type(2020, 4, 1))
    if festival_override is not None and not pd.isna(festival_override):
        data["festival_flag"] = int(festival_override)
    else:
        data["festival_flag"] = int(date_str in set(festival_days))

    return data


def _last_static_reference(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the most recent record carrying the slow-moving reference fields."""
    reference: dict[str, Any] = {}
    for record in records:
        if any(col in record and pd.notna(record[col]) for col in STATIC_COLUMNS):
            reference = {col: record[col] for col in STATIC_COLUMNS if col in record}
    return reference


def build_observation_frame(
    records: list[Mapping[str, Any]],
    *,
    params: Mapping[str, Any],
    grap_calendar: pd.DataFrame,
    festival_days: set[str],
    static_reference: Mapping[str, Any] | None = None,
    train_medians: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Turn raw API observations into the derived non-lag columns.

    The calendar, policy, city-static and meteorological-flag logic is imported
    directly from ingest_and_merge.py, so online rows match the offline panel
    byte-for-byte before the lag/rolling stage runs.
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame([dict(record) for record in records])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Preserve per-row GRAP/festival overrides before policy enrichment.  These
    # two columns are produced by add_policy_features(); pre-creating them here
    # would make its calendar merge clash (grap_stage_x / grap_stage_y).
    empty = pd.Series(np.nan, index=df.index, dtype=float)
    grap_override = df["grap_stage"].copy() if "grap_stage" in df.columns else empty
    festival_override = df["festival_flag"].copy() if "festival_flag" in df.columns else empty
    df = df.drop(columns=["grap_stage", "festival_flag"], errors="ignore")

    for col in OBSERVATION_COLUMNS:
        if col in ("grap_stage", "festival_flag"):
            continue
        if col not in df.columns:
            df[col] = np.nan

    if df["fire_dist_min_km"].isna().all():
        df["fire_dist_min_km"] = 999.0
    else:
        df["fire_dist_min_km"] = df["fire_dist_min_km"].fillna(999.0)

    reference = dict(static_reference or _last_static_reference(records))
    for col in STATIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        if col in reference and pd.notna(reference[col]):
            df[col] = df[col].fillna(float(reference[col]))
        if train_medians is not None:
            df[col] = df[col].fillna(float(train_medians.get(col, 0.0)))

    df = add_meteorological_derived(df)
    df = add_calendar_features(df)
    df = add_static_city_features(df)
    df = add_policy_features(df, grap_calendar, festival_days)

    def _restore(column: str, overrides: pd.Series) -> None:
        mask = overrides.notna()
        if not mask.any():
            return
        df.loc[mask, column] = overrides[mask].to_numpy()
        if column == "grap_stage":
            df.loc[df["city"] != "Delhi", column] = 0
        df[column] = df[column].astype(int)

    _restore("grap_stage", grap_override)
    _restore("festival_flag", festival_override)

    return df


# ---------------------------------------------------------------------------
# Category probabilities
# ---------------------------------------------------------------------------


def category_probabilities(prediction: float, sigma: float) -> dict[str, float]:
    """Approximate CPCB category probabilities from a normal residual model.

    The bundle stores residual quantiles (q10/q50/q90), not a classifier, so the
    interval mass over the CPCB cut-offs is converted into class probabilities.
    This is documented in the risk card / model card as an approximation.
    """
    sigma = max(float(sigma), 1e-6)

    def _cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf((value - prediction) / (sigma * math.sqrt(2.0))))

    cumulative = [_cdf(band) for band in AQI_BANDS]
    probabilities = [
        cumulative[0],
        cumulative[1] - cumulative[0],
        cumulative[2] - cumulative[1],
        cumulative[3] - cumulative[2],
        cumulative[4] - cumulative[3],
        1.0 - cumulative[4],
    ]
    total = sum(probabilities) or 1.0
    return {
        label: round(max(value, 0.0) / total, 6)
        for label, value in zip(CATEGORY_ORDER, probabilities, strict=False)
    }


def residual_sigma(quantiles: Mapping[str, float]) -> float:
    """Recover a normal-equivalent sigma from the 10th/90th residual quantiles."""
    return max((float(quantiles["q90"]) - float(quantiles["q10"])) / 2.5631031310892, 1e-6)


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------


def load_serving_configs(config_dir: str | Path) -> tuple[pd.DataFrame, set[str]]:
    """Load the GRAP calendar and festival dates used by the offline pipeline."""
    base = Path(config_dir)
    return (
        load_grap_calendar(base / "grap_calendar.csv"),
        load_festival_dates(base / "festival_dates.yaml"),
    )
