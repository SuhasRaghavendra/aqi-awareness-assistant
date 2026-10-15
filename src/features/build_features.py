"""
src/features/build_features.py
------------------------------
Phase 5, Step 008: Feature Engineering and Chronological Train/Val/Test Split.

Generates leakage-free features using information available at day t to predict day t+1:
  1. Next-day targets: target_aqi = AQI(t+1), target_category = CPCB bucket(target_aqi)
     (Computed strictly per station with date continuity; no shifting across date gaps).
  2. Lags: [1, 2, 3, 7] days for AQI, PM2.5, PM10, NO2, O3, CO, SO2.
  3. Rolling windows: [3, 7, 14] days mean, std, max for AQI and PM2.5 (past values only).
  4. Differences: aqi_diff1, pm25_diff1.
  5. Weather features: same-day weather, previous-day weather, 3-day rolling averages.
  6. Calendar: day of week, month, day of year, weekend flag, sin/cos cyclical encodings, season, festival flag.
  7. Ratios: PM2.5/PM10 ratio.
  8. Spatial context: City one-hot encodings and leave-one-out city mean AQI at day t.
  9. Chronological Split: train (<= 2023), val (2024), test (2025+).
  10. Scaling: StandardScaler fit exclusively on train split -> models/scaler.joblib.

Outputs:
  - data/processed/train.parquet
  - data/processed/val.parquet
  - data/processed/test.parquet
  - data/processed/feature_list.json
  - reports/metrics/features_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from src.config import load_params
from src.features.aqi_utils import aqi_to_category
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.stage_stats import track_stage

logger = get_logger("build_features")


def load_festival_dates(filepath: Path | str = "configs/festival_dates.yaml") -> set[str]:
    """Load festival dates into a set of YYYY-MM-DD date strings."""
    fp = Path(filepath)
    if not fp.exists():
        return set()
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return set()

    festival_days: set[str] = set()
    for period in data.get("diwali_periods", []):
        start = period.get("start")
        end = period.get("end")
        if start and end:
            dr = pd.date_range(start, end)
            for d in dr:
                festival_days.add(d.strftime("%Y-%m-%d"))

    for d_str in data.get("dussehra_dates", []):
        festival_days.add(str(d_str))
    for d_str in data.get("holi_dates", []):
        festival_days.add(str(d_str))

    return festival_days


def get_season_name(month: int) -> str:
    """Classify month into standard Indian meteorological season."""
    if month in [11, 12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "summer"
    elif month in [6, 7, 8, 9]:
        return "monsoon"
    else:
        return "post_monsoon"


def create_features_for_station(
    df_st: pd.DataFrame,
    lags: list[int],
    rolling_windows: list[int],
    festival_days: set[str],
) -> pd.DataFrame:
    """
    Construct temporal and meteorological features for a single station's time series.
    Guarantees no cross-station leakage and no shifting across date gaps.
    """
    df = df_st.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])

    # Build continuous calendar date index to prevent shifting across date gaps
    min_date = df["date"].min()
    max_date = df["date"].max()
    full_idx = pd.date_range(min_date, max_date, freq="D")
    df_full = df.set_index("date").reindex(full_idx)
    df_full.index.name = "date"

    # Forward fill static metadata for reindexed rows
    for col in ["location_id", "station_name", "city", "latitude", "longitude"]:
        if col in df_full.columns:
            df_full[col] = df_full[col].ffill().bfill()

    # 1. Target creation: next-day AQI and category (shift -1 on continuous calendar)
    df_full["target_aqi"] = df_full["aqi"].shift(-1)
    df_full["target_category"] = df_full["target_aqi"].apply(aqi_to_category)

    # 2. Lag features for pollutants
    lag_pollutants = ["aqi", "pm25", "pm10", "no2", "so2", "co", "o3"]
    for p in lag_pollutants:
        if p in df_full.columns:
            for lag in lags:
                df_full[f"{p}_lag_{lag}"] = df_full[p].shift(lag)

    # 3. Rolling window features (past values only, current day t and previous days)
    for w in rolling_windows:
        if "aqi" in df_full.columns:
            df_full[f"aqi_rolling_mean_{w}"] = df_full["aqi"].rolling(window=w, min_periods=1).mean()
            df_full[f"aqi_rolling_std_{w}"] = df_full["aqi"].rolling(window=w, min_periods=1).std().fillna(0.0)
            df_full[f"aqi_rolling_max_{w}"] = df_full["aqi"].rolling(window=w, min_periods=1).max()
        if "pm25" in df_full.columns:
            df_full[f"pm25_rolling_mean_{w}"] = df_full["pm25"].rolling(window=w, min_periods=1).mean()
            df_full[f"pm25_rolling_std_{w}"] = df_full["pm25"].rolling(window=w, min_periods=1).std().fillna(0.0)
            df_full[f"pm25_rolling_max_{w}"] = df_full["pm25"].rolling(window=w, min_periods=1).max()

    # 4. Differencing features
    if "aqi_lag_1" in df_full.columns and "aqi" in df_full.columns:
        df_full["aqi_diff1"] = df_full["aqi"] - df_full["aqi_lag_1"]
    if "pm25_lag_1" in df_full.columns and "pm25" in df_full.columns:
        df_full["pm25_diff1"] = df_full["pm25"] - df_full["pm25_lag_1"]

    # 5. Weather lag and rolling features
    weather_base = [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "surface_pressure_mean",
        "precipitation_sum",
        "wind_speed_10m_mean",
    ]
    for w_col in weather_base:
        if w_col in df_full.columns:
            df_full[f"{w_col}_lag1"] = df_full[w_col].shift(1)
            df_full[f"{w_col}_roll3_mean"] = df_full[w_col].rolling(window=3, min_periods=1).mean()

    # 6. Ratios
    if "pm25" in df_full.columns and "pm10" in df_full.columns:
        df_full["pm25_pm10_ratio"] = (df_full["pm25"] / (df_full["pm10"] + 1e-5)).clip(0.0, 1.0)

    # 7. Calendar and Temporal features
    dt_series = df_full.index
    df_full["day_of_week"] = dt_series.dayofweek
    df_full["is_weekend"] = (dt_series.dayofweek >= 5).astype(int)
    df_full["month"] = dt_series.month
    df_full["day_of_year"] = dt_series.dayofyear

    # Cyclical encodings
    df_full["sin_day_of_year"] = np.sin(2 * np.pi * dt_series.dayofyear / 365.25)
    df_full["cos_day_of_year"] = np.cos(2 * np.pi * dt_series.dayofyear / 365.25)
    df_full["sin_month"] = np.sin(2 * np.pi * dt_series.month / 12.0)
    df_full["cos_month"] = np.cos(2 * np.pi * dt_series.month / 12.0)

    # Season encoding
    season_series = dt_series.month.map(get_season_name)
    for s_name in ["winter", "summer", "monsoon", "post_monsoon"]:
        df_full[f"season_{s_name}"] = (season_series == s_name).astype(int)

    # Festival flag
    date_str_series = dt_series.strftime("%Y-%m-%d")
    df_full["is_festival"] = date_str_series.isin(festival_days).astype(int)

    # Re-filter back to only dates that were present in the original valid data
    df_full = df_full.reset_index()
    valid_dates_set = set(df["date"].dt.strftime("%Y-%m-%d"))
    df_out = df_full[df_full["date"].dt.strftime("%Y-%m-%d").isin(valid_dates_set)].copy()

    # Drop the final day where target_aqi is NaN
    df_out = df_out[df_out["target_aqi"].notna()].reset_index(drop=True)
    return df_out


def build_full_feature_panel(
    df: pd.DataFrame,
    lags: list[int],
    rolling_windows: list[int],
    festival_days: set[str],
) -> pd.DataFrame:
    """Process all stations and append cross-station spatial context."""
    station_dfs = []
    for loc_id, group in df.groupby("location_id"):
        st_df = create_features_for_station(group, lags, rolling_windows, festival_days)
        if not st_df.empty:
            station_dfs.append(st_df)

    if not station_dfs:
        return pd.DataFrame()

    combined = pd.concat(station_dfs, ignore_index=True)

    # Spatial context: leave-one-out city mean AQI at day t
    city_day_sum = combined.groupby(["city", "date"])["aqi"].transform("sum")
    city_day_count = combined.groupby(["city", "date"])["aqi"].transform("count")
    # Leave-one-out mean: (sum - self_val) / (count - 1)
    other_count = city_day_count - 1
    leave_one_out = np.where(
        other_count > 0,
        (city_day_sum - combined["aqi"]) / other_count,
        combined["aqi"],  # fallback to own AQI if only 1 station in city
    )
    combined["city_mean_aqi_other_stations"] = np.round(leave_one_out, 2)

    # One-hot encode city
    city_dummies = pd.get_dummies(combined["city"], prefix="city", dtype=int)
    combined = pd.concat([combined, city_dummies], axis=1)

    combined = combined.sort_values(["date", "city", "location_id"]).reset_index(drop=True)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: Build Features and Chronological Splits")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/interim/station_day_clean.parquet"),
        help="Input clean station-day parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for train/val/test splits",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save fitted scaler and artifacts",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("reports/metrics/features_stats.json"),
        help="Output features stage statistics JSON",
    )
    args = parser.parse_args()

    setup_logging(stage="features")
    logger.info("Starting Phase 5: Feature Engineering & Chronological Dataset Splitting...")

    if not args.input_file.exists():
        logger.error(f"Input file not found at {args.input_file}. Run clean stage first.")
        raise FileNotFoundError(f"{args.input_file} does not exist.")

    params = load_params()
    feat_cfg = params.get("features", {})
    lags = feat_cfg.get("lags", [1, 2, 3, 7])
    rolling_windows = feat_cfg.get("rolling_windows", [3, 7, 14])

    split_cfg = params.get("split", {})
    train_end = split_cfg.get("train_end", "2023-12-31")
    val_end = split_cfg.get("val_end", "2024-12-31")

    festival_days = load_festival_dates()

    df_clean = pd.read_parquet(args.input_file)
    logger.info(f"Loaded {len(df_clean):,} clean records across {df_clean['location_id'].nunique()} stations.")

    # Build features
    df_features = build_full_feature_panel(df_clean, lags, rolling_windows, festival_days)
    logger.info(f"Engineered feature panel: {len(df_features):,} rows x {len(df_features.columns)} columns.")

    # Split Chronologically: train (<= train_end), val (train_end < date <= val_end), test (> val_end)
    train_cutoff = pd.to_datetime(train_end)
    val_cutoff = pd.to_datetime(val_end)

    df_train = df_features[df_features["date"] <= train_cutoff].copy().reset_index(drop=True)
    df_val = df_features[(df_features["date"] > train_cutoff) & (df_features["date"] <= val_cutoff)].copy().reset_index(drop=True)
    df_test = df_features[df_features["date"] > val_cutoff].copy().reset_index(drop=True)

    # Boundary verification assert
    if not df_train.empty and not df_val.empty:
        assert df_train["date"].max() < df_val["date"].min(), "Leakage detected: train overlaps with validation!"
    if not df_val.empty and not df_test.empty:
        assert df_val["date"].max() < df_test["date"].min(), "Leakage detected: validation overlaps with test!"

    train_range = f"{df_train['date'].min().strftime('%Y-%m-%d')} to {df_train['date'].max().strftime('%Y-%m-%d')}" if not df_train.empty else "Empty"
    val_range = f"{df_val['date'].min().strftime('%Y-%m-%d')} to {df_val['date'].max().strftime('%Y-%m-%d')}" if not df_val.empty else "Empty"
    test_range = f"{df_test['date'].min().strftime('%Y-%m-%d')} to {df_test['date'].max().strftime('%Y-%m-%d')}" if not df_test.empty else "Empty"

    logger.info(
        f"Chronological Splits:\n"
        f"  - Train: {len(df_train):,} rows ({train_range})\n"
        f"  - Val:   {len(df_val):,} rows ({val_range})\n"
        f"  - Test:  {len(df_test):,} rows ({test_range})"
    )

    # Exclude metadata and non-predictive columns from feature matrix
    non_feature_cols = [
        "date",
        "location_id",
        "station_name",
        "city",
        "target_aqi",
        "target_category",
        "aqi_category",
        "dominant_pollutant",
        "is_aqi_valid",
    ]
    feature_cols = [c for c in df_features.columns if c not in non_feature_cols and np.issubdtype(df_features[c].dtype, np.number)]

    # Handle any remaining NaNs in lag features (e.g. initial lag window) by filling with train median
    train_medians = df_train[feature_cols].median().fillna(0.0).to_dict()
    df_train[feature_cols] = df_train[feature_cols].fillna(train_medians)
    df_val[feature_cols] = df_val[feature_cols].fillna(train_medians)
    df_test[feature_cols] = df_test[feature_cols].fillna(train_medians)

    # Fit StandardScaler exclusively on train split
    args.models_dir.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])
    scaler_path = args.models_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Fitted StandardScaler on training split ({len(df_train):,} rows) -> {scaler_path}")

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.parquet"
    val_path = args.output_dir / "val.parquet"
    test_path = args.output_dir / "test.parquet"
    feat_list_path = args.output_dir / "feature_list.json"

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)

    with open(feat_list_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_features": len(feature_cols),
                "feature_names": feature_cols,
                "target_regression": "target_aqi",
                "target_classification": "target_category",
                "generated_at": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    logger.info(f"Saved splits to {args.output_dir} ({len(feature_cols)} engineered features)")

    # Produce feature ranking / correlation figure
    corr_series = df_train[feature_cols].apply(lambda s: s.corr(df_train["target_aqi"])).dropna()
    top_corrs = corr_series.abs().sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_corrs.sort_values().plot(kind="barh", ax=ax, color="#1f77b4")
    ax.set_title("Top 15 Features by Absolute Correlation with Next-Day Target AQI (Train Split)", fontsize=12, pad=10)
    ax.set_xlabel("Absolute Pearson Correlation |r|")
    plt.tight_layout()
    fig_dir = Path("reports/figures/features")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "feature_correlations.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved feature correlation figure -> {fig_path}")

    # Track stage stats
    inputs = [str(args.input_file)]
    outputs = [str(train_path), str(val_path), str(test_path), str(feat_list_path)]
    with track_stage("features", inputs=inputs, outputs=outputs) as st:
        st.record_input_frame(df_clean)
        st.record_output_frame(df_train)

    print("\n" + "=" * 65)
    print("FEATURE ENGINEERING & CHRONOLOGICAL SPLIT SUMMARY")
    print("=" * 65)
    print(f"Total Features Generated:      {len(feature_cols)}")
    print(f"Train Rows:                    {len(df_train):,} ({len(df_train)/len(df_features):.1%})")
    print(f"Validation Rows:               {len(df_val):,} ({len(df_val)/len(df_features):.1%})")
    print(f"Test Rows:                     {len(df_test):,} ({len(df_test)/len(df_features):.1%})")
    print(f"Target Regression Range:       {df_train['target_aqi'].min():.1f} to {df_train['target_aqi'].max():.1f}")
    print(f"Target Categories:             {sorted(df_train['target_category'].dropna().unique().tolist())}")
    print(f"Scaler Saved:                  {scaler_path}")
    print(f"Feature List:                  {feat_list_path}")
    print(f"Outputs:                       {train_path}, {val_path}, {test_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
