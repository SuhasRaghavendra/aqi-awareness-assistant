"""
src/data/ingest_and_merge.py
-----------------------------
Phase 2: Merge all root-cause data sources into a single city × date panel.

Inputs:
    data/raw/aqi/city_aqi_daily.parquet          (XKDR — AQI target labels)
    data/raw/fires/city_fires_daily.parquet       (NASA FIRMS)
    data/raw/weather/city_weather_daily.parquet   (Open-Meteo ERA5)
    data/raw/vehicles/city_vehicles_annual.parquet (data.gov.in / MoRTH)
    data/raw/industrial/iip_monthly.parquet        (MOSPI)
    data/raw/industrial/state_power_monthly.parquet (CEA)
    configs/grap_calendar.csv                      (GRAP policy calendar)
    configs/festival_dates.yaml                    (festival calendar)

Merge strategy:
    1. AQI panel (city × date) is the left spine.
    2. Fire features → daily exact join.
    3. Weather features → daily exact join.
    4. Vehicle data (annual) → forward-fill after yearly step-up.
    5. IIP data (monthly) → forward-fill within month.
    6. Power data (monthly) → forward-fill + state→city mapping.
    7. Policy/calendar features → date-based flag joins.

Outputs:
    data/interim/city_day_raw.parquet    (full merged, before cleaning)
    reports/metrics/merge_stats.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.config import load_params
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.stage_stats import track_stage

logger = get_logger("ingest_and_merge")

CITY_STATE: dict[str, str] = {
    "Delhi":     "Delhi",
    "Mumbai":    "Maharashtra",
    "Bengaluru": "Karnataka",
    "Chennai":   "Tamil Nadu",
    "Kolkata":   "West Bengal",
    "Hyderabad": "Telangana",
    "Pune":      "Maharashtra",
    "Ahmedabad": "Gujarat",
    "Lucknow":   "Uttar Pradesh",
}

CITY_COORDS: dict[str, tuple[float, float]] = {
    "Delhi":     (28.6139, 77.2090),
    "Mumbai":    (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai":   (13.0827, 80.2707),
    "Kolkata":   (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Pune":      (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Lucknow":   (26.8467, 80.9462),
}

CITY_ELEVATION_M: dict[str, float] = {
    "Delhi":     216.0,
    "Mumbai":     14.0,
    "Bengaluru": 920.0,
    "Chennai":     6.0,
    "Kolkata":     9.0,
    "Hyderabad": 536.0,
    "Pune":      560.0,
    "Ahmedabad":  53.0,
    "Lucknow":   120.0,
}

CITY_POPULATION_LAKH: dict[str, float] = {
    "Delhi":     190.8,
    "Mumbai":    184.1,
    "Bengaluru": 121.6,
    "Chennai":    87.5,
    "Kolkata":    45.9,
    "Hyderabad":  97.0,
    "Pune":       50.5,
    "Ahmedabad":  72.5,
    "Lucknow":    29.0,
}


def load_festival_dates(path: Path = Path("configs/festival_dates.yaml")) -> set[str]:
    if not path.exists():
        return set()
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    days: set[str] = set()
    for period in data.get("diwali_periods", []):
        dr = pd.date_range(period["start"], period["end"])
        for d in dr:
            days.add(d.strftime("%Y-%m-%d"))
    for d in data.get("dussehra_dates", []):
        days.add(str(d))
    for d in data.get("holi_dates", []):
        days.add(str(d))
    return days


def load_grap_calendar(path: Path = Path("configs/grap_calendar.csv")) -> pd.DataFrame:
    """
    Load GRAP action level calendar for Delhi NCR.
    Expected columns: date, grap_stage (0=none,1=poor,2=very_poor,3=severe,4=severe+)
    Also: odd_even_active (0/1)
    """
    if not path.exists():
        logger.warning(
            f"{path} not found. GRAP features will be 0. "
            "Create configs/grap_calendar.csv with columns: date, grap_stage, odd_even_active"
        )
        return pd.DataFrame(columns=["date", "grap_stage", "odd_even_active"])
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def build_date_spine(cities: list[str], start: str, end: str) -> pd.DataFrame:
    """Create a complete city × date spine (no gaps)."""
    dates = pd.date_range(start, end, freq="D")
    frames = []
    for city in cities:
        df = pd.DataFrame({"date": dates, "city": city})
        frames.append(df)
    spine = pd.concat(frames, ignore_index=True)
    spine["date"] = spine["date"].dt.strftime("%Y-%m-%d")
    return spine


def merge_aqi(spine: pd.DataFrame, aqi_path: Path) -> pd.DataFrame:
    """Left join AQI labels onto spine."""
    if not aqi_path.exists():
        logger.error(f"AQI file not found: {aqi_path}")
        spine["aqi"] = float("nan")
        return spine

    df_aqi = pd.read_parquet(aqi_path)
    df_aqi["date"] = pd.to_datetime(df_aqi["date"]).dt.strftime("%Y-%m-%d")
    aqi_cols = ["city", "date", "aqi"]
    if "aqi_median" in df_aqi.columns:
        aqi_cols.append("aqi_median")
    if "station_count" in df_aqi.columns:
        aqi_cols.append("station_count")
    df_aqi = df_aqi[[c for c in aqi_cols if c in df_aqi.columns]]

    merged = spine.merge(df_aqi, on=["city", "date"], how="left")
    valid = merged["aqi"].notna().sum()
    logger.info(f"AQI joined: {valid:,} / {len(merged):,} city-days have AQI labels")
    return merged


def merge_fires(df: pd.DataFrame, fires_path: Path) -> pd.DataFrame:
    """Join fire features (daily)."""
    if not fires_path.exists():
        logger.warning(f"Fire data not found: {fires_path} — fire features will be 0")
        for col in ["fire_count_500km", "fire_frp_sum_500km", "fire_dist_min_km"]:
            df[col] = 0.0
        return df

    df_fires = pd.read_parquet(fires_path)
    df_fires["date"] = pd.to_datetime(df_fires["date"]).dt.strftime("%Y-%m-%d")
    df_fires = df_fires.rename(columns={
        "fire_count": "fire_count_500km",
        "fire_frp_sum": "fire_frp_sum_500km",
        "fire_dist_min_km": "fire_dist_min_km",
    })
    fire_cols = ["city", "date", "fire_count_500km", "fire_frp_sum_500km", "fire_dist_min_km"]
    df_fires = df_fires[[c for c in fire_cols if c in df_fires.columns]]

    merged = df.merge(df_fires, on=["city", "date"], how="left")
    # Days with no fires → fill 0
    for col in ["fire_count_500km", "fire_frp_sum_500km"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
    if "fire_dist_min_km" in merged.columns:
        merged["fire_dist_min_km"] = merged["fire_dist_min_km"].fillna(999.0)
    logger.info(f"Fires joined: {(merged['fire_count_500km'] > 0).sum():,} city-days with fires")
    return merged


def merge_weather(df: pd.DataFrame, weather_path: Path) -> pd.DataFrame:
    """Join weather features (daily)."""
    if not weather_path.exists():
        logger.error(f"Weather data not found: {weather_path}")
        return df

    df_w = pd.read_parquet(weather_path)
    df_w["date"] = pd.to_datetime(df_w["date"]).dt.strftime("%Y-%m-%d")
    # Drop city-id-type cols that might conflict
    merge_cols = [c for c in df_w.columns if c not in ["city_id", "station_id"]]
    merged = df.merge(df_w[merge_cols], on=["city", "date"], how="left")
    logger.info(f"Weather joined: {merged['temperature_2m_mean'].notna().sum():,} rows with temperature")
    return merged


def merge_vehicles(df: pd.DataFrame, vehicles_path: Path) -> pd.DataFrame:
    """Join vehicle features (annual → forward-fill)."""
    if not vehicles_path.exists():
        logger.warning(f"Vehicle data not found: {vehicles_path}")
        return df

    df_v = pd.read_parquet(vehicles_path)
    df_v["year"] = df_v["year"].astype(int)

    # Extract year from date for join
    df["_year"] = pd.to_datetime(df["date"]).dt.year.astype(int)

    vehicle_feat_cols = [
        "city", "year", "diesel_frac", "cng_frac", "ev_frac",
        "two_wheeler_frac", "vehicles_per_lakh",
    ]
    df_v_slim = df_v[[c for c in vehicle_feat_cols if c in df_v.columns]].drop_duplicates(
        subset=["city", "year"]
    )

    merged = df.merge(df_v_slim, left_on=["city", "_year"], right_on=["city", "year"], how="left")
    merged.drop(columns=["_year", "year"], errors="ignore", inplace=True)

    # Forward-fill within city for missing years (interpolate between known years)
    vehicle_cols = ["diesel_frac", "cng_frac", "ev_frac", "two_wheeler_frac", "vehicles_per_lakh"]
    for col in vehicle_cols:
        if col in merged.columns:
            merged[col] = merged.groupby("city")[col].transform(
                lambda s: s.interpolate(method="linear").ffill().bfill()
            )

    logger.info(f"Vehicles joined: {merged['diesel_frac'].notna().sum():,} rows with vehicle features")
    return merged


def merge_industrial(df: pd.DataFrame, iip_path: Path, power_path: Path) -> pd.DataFrame:
    """Join IIP + power features (monthly → forward-fill)."""
    df["_year"] = pd.to_datetime(df["date"]).dt.year.astype(int)
    df["_month"] = pd.to_datetime(df["date"]).dt.month.astype(int)

    # IIP (national, same for all cities)
    if iip_path.exists():
        df_iip = pd.read_parquet(iip_path)
        df_iip["year"] = df_iip["year"].astype(int)
        df_iip["month"] = df_iip["month"].astype(int)
        df_iip = df_iip[["year", "month", "iip_manufacturing_index"]].drop_duplicates()
        df = df.merge(df_iip, left_on=["_year", "_month"], right_on=["year", "month"], how="left")
        df.drop(columns=["year", "month"], errors="ignore", inplace=True)
        df["iip_manufacturing_index"] = df["iip_manufacturing_index"].interpolate(method="linear").ffill().bfill()
        logger.info(f"IIP joined: {df['iip_manufacturing_index'].notna().sum():,} rows")

    # State power (per city via state mapping)
    if power_path.exists():
        df_power = pd.read_parquet(power_path)
        df_power["year"] = df_power["year"].astype(int)
        df_power["month"] = df_power["month"].astype(int)

        power_records = []
        for city in df["city"].unique():
            state = CITY_STATE.get(city, "")
            if "state" in df_power.columns:
                df_state = df_power[df_power["state"].str.contains(state, case=False, na=False)].copy()
            else:
                df_state = pd.DataFrame()
            if df_state.empty:
                continue
            df_state["city"] = city
            power_records.append(df_state)

        if power_records:
            df_power_city = pd.concat(power_records, ignore_index=True)
            power_cols = ["city", "year", "month"] + [
                c for c in ["energy_requirement_mu", "energy_availability_mu"]
                if c in df_power_city.columns
            ]
            df = df.merge(
                df_power_city[power_cols],
                left_on=["city", "_year", "_month"],
                right_on=["city", "year", "month"],
                how="left",
            )
            df.drop(columns=["year", "month"], errors="ignore", inplace=True)

    df.drop(columns=["_year", "_month"], errors="ignore", inplace=True)
    return df


def add_policy_features(
    df: pd.DataFrame,
    grap_df: pd.DataFrame,
    festival_days: set[str],
) -> pd.DataFrame:
    """Add GRAP stage, festival, and policy flags."""
    # GRAP calendar (Delhi NCR)
    if not grap_df.empty:
        df = df.merge(grap_df[["date", "grap_stage", "odd_even_active"]], on="date", how="left")
    else:
        df["grap_stage"] = 0
        df["odd_even_active"] = 0

    # GRAP only applies to Delhi
    if "grap_stage" in df.columns:
        df.loc[df["city"] != "Delhi", "grap_stage"] = 0
    if "odd_even_active" in df.columns:
        df.loc[df["city"] != "Delhi", "odd_even_active"] = 0

    df["grap_stage"] = df["grap_stage"].fillna(0).astype(int)
    df["odd_even_active"] = df["odd_even_active"].fillna(0).astype(int)

    # BS-VI vehicle emission standard (national, from April 1, 2020)
    df["bsvi_active"] = (pd.to_datetime(df["date"]) >= pd.Timestamp("2020-04-01")).astype(int)

    # Festival flag
    df["festival_flag"] = df["date"].isin(festival_days).astype(int)

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date-derived calendar features."""
    dt = pd.to_datetime(df["date"])
    df["day_of_week"] = dt.dt.dayofweek          # 0=Monday, 6=Sunday
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df["year"] = dt.dt.year
    df["weekend_flag"] = (dt.dt.dayofweek >= 5).astype(int)
    df["burning_season_flag"] = (
        (dt.dt.month == 10) & (dt.dt.day >= 15) |
        (dt.dt.month == 11)
    ).astype(int)

    # Indian meteorological seasons
    def get_season(m: int) -> str:
        if m in [11, 12, 1, 2]:
            return "winter"
        elif m in [3, 4, 5]:
            return "summer"
        elif m in [6, 7, 8, 9]:
            return "monsoon"
        else:
            return "post_monsoon"

    df["season"] = dt.dt.month.map(get_season)

    # Cyclical encodings
    import math
    df["month_sin"] = df["month"].map(lambda m: math.sin(2 * math.pi * m / 12))
    df["month_cos"] = df["month"].map(lambda m: math.cos(2 * math.pi * m / 12))
    df["dow_sin"] = df["day_of_week"].map(lambda d: math.sin(2 * math.pi * d / 7))
    df["dow_cos"] = df["day_of_week"].map(lambda d: math.cos(2 * math.pi * d / 7))
    return df


def add_static_city_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add city-level static geographic features."""
    df["city_lat"] = df["city"].map({c: v[0] for c, v in CITY_COORDS.items()})
    df["city_lon"] = df["city"].map({c: v[1] for c, v in CITY_COORDS.items()})
    df["city_elevation_m"] = df["city"].map(CITY_ELEVATION_M)
    df["city_population_lakh"] = df["city"].map(CITY_POPULATION_LAKH)
    return df


def add_meteorological_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Derive secondary meteorological features useful for pollution trapping."""
    # Temperature inversion proxy: low wind + low pressure + high humidity
    if all(c in df.columns for c in ["wind_speed_10m_mean", "relative_humidity_2m_mean"]):
        wind = df["wind_speed_10m_mean"].fillna(5.0)
        humidity = df["relative_humidity_2m_mean"].fillna(50.0)
        # Inversion risk: high when wind < 5 km/h AND humidity > 70%
        df["inversion_risk_flag"] = ((wind < 5.0) & (humidity > 70.0)).astype(int)
        # Fog risk: humidity > 85% AND wind < 3 km/h (mainly winter)
        df["fog_risk_flag"] = ((humidity > 85.0) & (wind < 3.0)).astype(int)

    # Precipitation washout flag
    if "precipitation_sum" in df.columns:
        df["rain_washout_flag"] = (df["precipitation_sum"].fillna(0) > 5.0).astype(int)

    return df


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Merge all root-cause data into city × day panel")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end",   default="2026-09-30")
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim"))
    args = parser.parse_args()

    params = load_params()
    cities: list[str] = params["data"]["cities"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build complete spine
    logger.info("Building city × date spine …")
    df = build_date_spine(cities, args.start, args.end)
    logger.info(f"Spine: {len(df):,} city-days ({len(cities)} cities × {args.end}→{args.start})")

    # Merge all sources
    logger.info("Merging AQI labels …")
    df = merge_aqi(df, Path("data/raw/aqi/city_aqi_daily.parquet"))

    logger.info("Merging fire events …")
    df = merge_fires(df, Path("data/raw/fires/city_fires_daily.parquet"))

    logger.info("Merging weather …")
    df = merge_weather(df, Path("data/raw/weather/city_weather_daily.parquet"))

    logger.info("Merging vehicle data …")
    df = merge_vehicles(df, Path("data/raw/vehicles/city_vehicles_annual.parquet"))

    logger.info("Merging industrial data …")
    df = merge_industrial(
        df,
        Path("data/raw/industrial/iip_monthly.parquet"),
        Path("data/raw/industrial/state_power_monthly.parquet"),
    )

    logger.info("Adding policy & calendar features …")
    festival_days = load_festival_dates()
    grap_df = load_grap_calendar()
    df = add_policy_features(df, grap_df, festival_days)
    df = add_calendar_features(df)
    df = add_static_city_features(df)
    df = add_meteorological_derived(df)

    # Sort and save
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    output_path = args.output_dir / "city_day_raw.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved merged panel → {output_path}")

    # Stats
    total = len(df)
    aqi_valid = int(df["aqi"].notna().sum()) if "aqi" in df.columns else 0
    stats: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "total_rows": total,
        "aqi_valid_rows": aqi_valid,
        "aqi_completeness_pct": round(aqi_valid / total * 100, 1) if total else 0,
        "columns": list(df.columns),
        "cities": cities,
        "missing_pct_by_col": {
            col: round(df[col].isna().mean() * 100, 1)
            for col in df.columns
            if df[col].isna().any()
        },
    }
    metrics_path = Path("reports/metrics/merge_stats.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    print(f"Total city-day rows:   {total:,}")
    print(f"AQI valid rows:        {aqi_valid:,} ({stats['aqi_completeness_pct']}%)")
    print(f"Total features:        {len(df.columns)}")
    print(f"Output:                {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

