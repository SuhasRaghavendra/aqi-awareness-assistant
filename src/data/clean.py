"""
src/data/clean.py
-----------------
Phase 3 – Cleaning: Apply data quality transformations to the merged panel.

Steps (train-split safe — all statistics computed on train only):
  1. Drop rows with excessive missing values (> max_missing_frac_row)
  2. Winsorize numeric features (clip outliers to [q0.5%, q99.5%]) — train only
  3. Forward-fill slowly-changing features (vehicle, IIP) within city × year
  4. Fill fire features with 0 on days without fires
  5. Interpolate sparse industrial features (IIP, power) linearly within city
  6. Drop cities/date ranges with AQI coverage below minimum threshold
  7. Compute AQI category label from continuous AQI (using CPCB breakpoints)

Outputs:
    data/interim/city_day_clean.parquet
    reports/metrics/clean_stats.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_params
from src.features.aqi_utils import aqi_to_category
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.stage_stats import track_stage

logger = get_logger("clean")

# Columns that should stay 0 (not NaN) when there are no fires
FIRE_COLS_ZERO_FILL = [
    "fire_count_500km", "fire_frp_sum_500km",
]
FIRE_COLS_999_FILL = ["fire_dist_min_km"]

# Slowly-changing features interpolated within city
SLOW_INTERP_COLS = [
    "diesel_frac", "cng_frac", "ev_frac",
    "two_wheeler_frac", "vehicles_per_lakh",
    "iip_manufacturing_index",
    "energy_requirement_mu", "energy_availability_mu",
]

# Columns never to winsorize (flags, identifiers, targets, ordinal scales)
NO_WINSOR_COLS = {
    "city", "date", "aqi", "aqi_category", "month", "year",
    "day_of_week", "day_of_year", "season", "weekend_flag",
    "burning_season_flag", "festival_flag", "grap_stage",
    "bsvi_active", "odd_even_active", "rain_washout_flag",
    "inversion_risk_flag", "fog_risk_flag",
    "city_lat", "city_lon", "city_elevation_m", "city_population_lakh",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
}


def drop_high_missing_rows(df: pd.DataFrame, max_frac: float) -> pd.DataFrame:
    """Drop rows where more than max_frac of numeric columns are NaN."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # Exclude target and flag columns from the count
    numeric_cols = [c for c in numeric_cols if c not in NO_WINSOR_COLS]
    before = len(df)
    missing_frac = df[numeric_cols].isna().mean(axis=1)
    df = df[missing_frac <= max_frac].copy()
    logger.info(f"Dropped {before - len(df):,} rows with >{max_frac:.0%} missing features")
    return df


def winsorise(
    df_train: pd.DataFrame,
    df_full: pd.DataFrame,
    lower_q: float,
    upper_q: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """
    Compute winsorization bounds on train split only.
    Apply bounds to full dataset. Returns clipped DataFrame + bounds dict.
    """
    numeric_cols = [
        c for c in df_full.select_dtypes(include="number").columns
        if c not in NO_WINSOR_COLS
    ]
    bounds: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        lo = float(df_train[col].quantile(lower_q))
        hi = float(df_train[col].quantile(upper_q))
        bounds[col] = {"lo": round(lo, 4), "hi": round(hi, 4)}
        df_full[col] = df_full[col].clip(lower=lo, upper=hi)
    logger.info(f"Winsorized {len(bounds)} numeric columns using train-split quantiles")
    return df_full, bounds


def interpolate_slow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill and interpolate slowly-changing features within each city."""
    for col in SLOW_INTERP_COLS:
        if col not in df.columns:
            continue
        df[col] = (
            df.groupby("city")[col]
            .transform(lambda s: s.interpolate(method="linear").ffill().bfill())
        )
    logger.info(f"Interpolated {len(SLOW_INTERP_COLS)} slow-changing feature columns")
    return df


def fill_fire_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Days with no fire data → 0 fires (not NaN)."""
    for col in FIRE_COLS_ZERO_FILL:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    for col in FIRE_COLS_999_FILL:
        if col in df.columns:
            df[col] = df[col].fillna(999.0)
    return df


def forward_fill_weather(df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Short forward-fill for weather gaps (API outages etc.) within city."""
    weather_cols = [
        c for c in df.columns
        if any(c.startswith(p) for p in [
            "temperature_2m", "relative_humidity", "wind_speed",
            "wind_direction", "precipitation", "surface_pressure",
            "vapour_pressure", "inversion_risk", "fog_risk", "rain_washout",
        ])
    ]
    for col in weather_cols:
        if col in df.columns:
            df[col] = df.groupby("city")[col].transform(
                lambda s: s.ffill(limit=limit).bfill(limit=1)
            )
    return df


def add_aqi_category(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CPCB AQI category from continuous AQI value."""
    if "aqi" in df.columns:
        df["aqi_category"] = df["aqi"].apply(
            lambda v: aqi_to_category(v) if pd.notna(v) else None
        )
    return df


def drop_low_aqi_coverage(
    df: pd.DataFrame,
    min_valid_days: int,
    split_end: str,
) -> pd.DataFrame:
    """Drop city × year groups where AQI coverage is too sparse."""
    df["_year"] = pd.to_datetime(df["date"]).dt.year
    train_df = df[df["date"] <= split_end]

    city_year_counts = (
        train_df[train_df["aqi"].notna()]
        .groupby(["city", "_year"])
        .size()
        .reset_index(name="aqi_days")
    )
    good = city_year_counts[city_year_counts["aqi_days"] >= min_valid_days][["city", "_year"]]
    before = len(df)
    # Only drop training years; keep all test/val rows regardless
    val_test = df[df["date"] > split_end]
    train_filtered = train_df.merge(good, on=["city", "_year"], how="inner")
    df_clean = pd.concat([train_filtered, val_test], ignore_index=True)
    df_clean.drop(columns=["_year"], inplace=True)
    logger.info(
        f"AQI coverage filter: dropped {before - len(df_clean):,} train rows below "
        f"{min_valid_days} valid AQI days per city-year"
    )
    return df_clean


def main() -> None:
    setup_logging()
    params = load_params()
    clean_cfg = params.get("clean", {})
    split_cfg = params.get("split", {})

    max_missing_frac = clean_cfg.get("max_missing_frac_row", 0.5)
    ffill_limit = clean_cfg.get("ffill_limit_days", 3)
    winsor_lo = clean_cfg.get("winsor_lower_q", 0.005)
    winsor_hi = clean_cfg.get("winsor_upper_q", 0.995)
    train_end = split_cfg.get("train_end", "2022-12-31")
    min_aqi_days = params.get("aqi", {}).get("min_valid_days_per_city_year", 180)

    input_path = Path("data/interim/city_day_raw.parquet")
    output_dir = Path("data/interim")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    logger.info(f"Loaded raw panel: {len(df):,} rows, {len(df.columns)} cols")
    rows_before = len(df)

    # 1. Fill fire defaults (0 not NaN)
    df = fill_fire_defaults(df)

    # 2. Interpolate slow features
    df = interpolate_slow_features(df)

    # 3. Forward-fill weather gaps
    df = forward_fill_weather(df, limit=ffill_limit)

    # 4. AQI category label
    df = add_aqi_category(df)

    # 5. Drop low-AQI-coverage city-years (train only)
    df = drop_low_aqi_coverage(df, min_aqi_days, train_end)

    # 6. Drop rows with excessive missing
    df = drop_high_missing_rows(df, max_missing_frac)

    # 7. Winsorize using train-split statistics only
    df_train = df[df["date"] <= train_end].copy()
    df, winsor_bounds = winsorise(df_train, df, winsor_lo, winsor_hi)

    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    rows_after = len(df)

    output_path = output_dir / "city_day_clean.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved clean panel → {output_path} ({rows_after:,} rows)")

    stats: dict[str, Any] = {
        "cleaned_at": datetime.now().isoformat(),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_dropped": rows_before - rows_after,
        "columns": len(df.columns),
        "train_end": train_end,
        "winsor_bounds": winsor_bounds,
        "aqi_valid": int(df["aqi"].notna().sum()),
        "aqi_coverage_pct": round(df["aqi"].notna().mean() * 100, 1),
        "cities": sorted(df["city"].unique().tolist()),
    }

    metrics_path = Path("reports/metrics/clean_stats.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(stats, f, indent=2)

    with track_stage("clean", inputs=[str(input_path)], outputs=[str(output_path)]) as st:
        st.record_input_frame(pd.read_parquet(input_path))
        st.record_output_frame(df)

    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"Rows before: {rows_before:,}  →  after: {rows_after:,}")
    print(f"AQI valid:   {stats['aqi_valid']:,} ({stats['aqi_coverage_pct']}%)")
    print(f"Columns:     {stats['columns']}")
    print(f"Output:      {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

