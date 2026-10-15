"""
src/data/clean.py
-----------------
Phase 5, Step 007: Clean and Impute Station-Day Panel without Data Leakage.

1. Sort by (location_id, date) and drop duplicates.
2. Filter to valid target days: drop rows with missing AQI.
3. Implausible value handling: replace negative/physically impossible values with NaN.
4. Strictly Leakage-Free Winsorization:
   - Quantiles (0.5% lower, 99.5% upper) fit exclusively on the training split (date <= train_end).
   - Applied to clip outliers across full dataset.
5. Column dropping: remove columns exceeding drop_cols_over_missing (0.70).
6. Strictly Leakage-Free Imputation:
   - Forward-fill per station up to ffill_limit_days (2 days).
   - Remaining NaNs filled with each station's training-split median (or global training median).
   - Add binary indicators (<col>_was_missing) for imputed columns.
7. Emit stage statistics and append change log.

Outputs:
  - data/interim/station_day_clean.parquet
  - reports/metrics/clean_stats.json
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_params
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.stage_stats import track_stage

logger = get_logger("clean")

NUMERIC_POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3", "aqi"]
WEATHER_COLS = [
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "precipitation_sum",
    "wind_speed_10m_mean",
    "wind_direction_10m_mean",
]


def clean_station_day_panel(
    df: pd.DataFrame,
    train_end_date: str = "2023-12-31",
    winsor_lower_q: float = 0.005,
    winsor_upper_q: float = 0.995,
    max_missing_frac_col: float = 0.70,
    ffill_limit: int = 2,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Clean, winsorize, and impute station-day panel with strictly zero future leakage.
    """
    drops: dict[str, int] = {
        "missing_target_aqi": 0,
        "duplicate_rows": 0,
        "implausible_values": 0,
    }

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 1. Deduplication
    initial_len = len(df)
    df = df.sort_values(["location_id", "date"]).drop_duplicates(
        subset=["location_id", "date"]
    ).reset_index(drop=True)
    drops["duplicate_rows"] = initial_len - len(df)

    # 2. Filter missing target AQI
    valid_aqi_mask = df["aqi"].notna() & (df["aqi"] > 0)
    drops["missing_target_aqi"] = int((~valid_aqi_mask).sum())
    df = df[valid_aqi_mask].reset_index(drop=True)

    if df.empty:
        return df, drops

    # 3. Handle impossible / negative concentrations
    cols_to_check = [c for c in NUMERIC_POLLUTANTS + WEATHER_COLS if c in df.columns]
    for col in cols_to_check:
        if col in ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3", "aqi"]:
            neg_mask = df[col] < 0
            if neg_mask.any():
                drops["implausible_values"] += int(neg_mask.sum())
                df.loc[neg_mask, col] = np.nan
        if col == "relative_humidity_2m_mean":
            invalid_rh = (df[col] < 0) | (df[col] > 100)
            if invalid_rh.any():
                drops["implausible_values"] += int(invalid_rh.sum())
                df.loc[invalid_rh, col] = np.nan

    # 4. Zero-Leakage Winsorization
    # Identify training split mask
    train_mask = df["date"] <= pd.to_datetime(train_end_date)
    if train_mask.sum() == 0:
        logger.warning(
            f"No rows found with date <= {train_end_date}. Using first 60% of sorted records as reference."
        )
        train_mask = pd.Series(False, index=df.index)
        train_mask.iloc[: int(len(df) * 0.6)] = True

    df_train = df[train_mask]
    logger.info(
        f"Fitting winsorization quantiles on {len(df_train):,} training records (dates <= {train_end_date})..."
    )

    numeric_cols = [
        c for c in cols_to_check if c != "aqi" and np.issubdtype(df[c].dtype, np.number)
    ]

    for col in numeric_cols:
        train_col_valid = df_train[col].dropna()
        if len(train_col_valid) > 20:
            lower_val = float(train_col_valid.quantile(winsor_lower_q))
            upper_val = float(train_col_valid.quantile(winsor_upper_q))
            # Apply training bounds to full dataset
            df[col] = df[col].clip(lower=lower_val, upper=upper_val)

    # 5. Drop columns exceeding max missing fraction
    cols_to_drop = []
    for col in df.columns:
        if col in ["location_id", "date", "station_name", "city", "aqi", "aqi_category"]:
            continue
        missing_frac = df[col].isna().mean()
        if missing_frac > max_missing_frac_col:
            cols_to_drop.append(col)

    if cols_to_drop:
        logger.info(f"Dropping columns with >{max_missing_frac_col:.0%} missing values: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # 6. Zero-Leakage Imputation & Missingness Indicators
    active_num_cols = [c for c in numeric_cols if c in df.columns]

    # Add missingness flags
    for col in active_num_cols:
        if df[col].isna().any():
            df[f"{col}_was_missing"] = df[col].isna().astype(int)

    # Step A: Forward-fill within station (mimics operational persistence up to ffill_limit days)
    df_imputed_list = []
    for loc_id, group in df.groupby("location_id", sort=False):
        grp = group.sort_values("date").copy()
        for col in active_num_cols:
            grp[col] = grp[col].ffill(limit=ffill_limit)
        df_imputed_list.append(grp)

    df = pd.concat(df_imputed_list, ignore_index=True)

    # Step B: Fill remaining NaNs with station's training-set median (or global training median)
    df_train_updated = df[df["date"] <= pd.to_datetime(train_end_date)]
    if df_train_updated.empty:
        df_train_updated = df.iloc[: int(len(df) * 0.6)]

    for col in active_num_cols:
        if df[col].isna().any():
            station_medians = df_train_updated.groupby("location_id")[col].median().to_dict()
            global_train_median = df_train_updated[col].median()
            if pd.isna(global_train_median):
                global_train_median = 0.0

            def fill_func(row: pd.Series) -> float:
                val = row[col]
                if pd.notna(val):
                    return float(val)
                loc = row["location_id"]
                st_med = station_medians.get(loc, np.nan)
                if pd.notna(st_med):
                    return float(st_med)
                return float(global_train_median)

            df[col] = df.apply(fill_func, axis=1)

    df = df.sort_values(["city", "location_id", "date"]).reset_index(drop=True)
    return df, drops


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: Clean and Impute Station-Day Panel")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/interim/station_day.parquet"),
        help="Input merged station-day panel parquet file",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/interim/station_day_clean.parquet"),
        help="Output clean station-day panel parquet file",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("reports/metrics/clean_stats.json"),
        help="Output clean stage statistics JSON",
    )
    args = parser.parse_args()

    setup_logging(stage="clean")
    logger.info("Starting Phase 5: Cleaning and Imputing Station-Day Panel...")

    if not args.input_file.exists():
        logger.error(f"Input file not found at {args.input_file}. Run merge stage first.")
        raise FileNotFoundError(f"{args.input_file} does not exist.")

    params = load_params()
    train_end = params.get("split", {}).get("train_end", "2023-12-31")
    clean_cfg = params.get("clean", {})
    winsor_lower = clean_cfg.get("winsor_lower_q", 0.005)
    winsor_upper = clean_cfg.get("winsor_upper_q", 0.995)
    drop_col_missing = clean_cfg.get("drop_cols_over_missing", 0.70)
    ffill_days = clean_cfg.get("ffill_limit_days", 2)

    df_in = pd.read_parquet(args.input_file)
    logger.info(f"Loaded {len(df_in):,} raw station-day records from {args.input_file}")

    df_clean, drops = clean_station_day_panel(
        df_in,
        train_end_date=train_end,
        winsor_lower_q=winsor_lower,
        winsor_upper_q=winsor_upper,
        max_missing_frac_col=drop_col_missing,
        ffill_limit=ffill_days,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(args.output_file, index=False)
    logger.info(f"Saved {len(df_clean):,} clean station-day records to {args.output_file}")

    # Track stage stats
    inputs = [str(args.input_file)]
    with track_stage("clean", inputs=inputs, outputs=[str(args.output_file)]) as st:
        st.record_input_frame(df_in)
        st.record_output_frame(df_clean)
        for reason, n in drops.items():
            if n > 0:
                st.drop(reason, n)

    print("\n" + "=" * 65)
    print("STATION-DAY CLEANING & IMPUTATION SUMMARY")
    print("=" * 65)
    print(f"Rows In:                       {len(df_in):,}")
    print(f"Rows Out:                      {len(df_clean):,}")
    print(f"Rows Dropped (Missing AQI):    {drops['missing_target_aqi']:,}")
    print(f"Duplicates Removed:            {drops['duplicate_rows']:,}")
    print(f"Implausible Values Replaced:   {drops['implausible_values']:,}")
    print(f"Total Columns Out:             {len(df_clean.columns)}")
    print(f"Output File:                   {args.output_file}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
