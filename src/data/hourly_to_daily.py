"""
src/data/hourly_to_daily.py
---------------------------
Phase 4, Step 006: Hourly to Daily Pollutant Panel Aggregation & CPCB AQI Computation.

Reads narrow OpenAQ raw observation files, standardizes parameter names and units,
converts timestamps to Asia/Kolkata (IST), performs hourly-to-daily aggregation
(24h mean for PM2.5/PM10/NO2/SO2/NH3, running 8h max for CO and O3),
computes CPCB sub-indices, overall AQI, AQI categories, and dominant pollutants.

Outputs:
  - data/interim/pollutants_daily.parquet
  - reports/metrics/hourly_to_daily_stats.json
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_params
from src.features.aqi_utils import (
    calc_overall_aqi,
    calc_sub_index,
    load_cpcb_breakpoints,
)
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.stage_stats import track_stage

logger = get_logger("hourly_to_daily")

TARGET_POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3", "nox", "no"]


def standardize_parameter_name(param: str) -> str:
    """Normalize raw OpenAQ parameter name strings to standard keys."""
    p = str(param).lower().replace(".", "").replace("-", "").strip()
    mapping = {
        "pm25": "pm25",
        "pm10": "pm10",
        "no2": "no2",
        "so2": "so2",
        "co": "co",
        "o3": "o3",
        "ozone": "o3",
        "nh3": "nh3",
        "nox": "nox",
        "no": "no",
    }
    return mapping.get(p, p)


def convert_pollutant_unit(param: str, value: float, unit_str: str) -> float | None:
    """
    Standardize measurement units to CPCB reference units:
      - CO -> mg/m³
      - All other pollutants -> µg/m³
    """
    if value is None or math.isnan(value) or value < 0:
        return None

    u = str(unit_str).lower().strip()

    if param == "co":
        # Target: mg/m³
        if "mg/m" in u or "mg" in u:
            return float(value)
        elif "µg/m" in u or "ug/m" in u or "g/m" in u:
            # 1 µg/m³ = 0.001 mg/m³
            return round(float(value) * 0.001, 4)
        elif "ppm" in u:
            # 1 ppm CO ≈ 1.145 mg/m³ at 25°C, 1 atm
            return round(float(value) * 1.145, 4)
        elif "ppb" in u:
            return round(float(value) * 0.001145, 4)
        return float(value)
    else:
        # Target: µg/m³
        if "ppm" in u:
            if param == "no2":
                return round(float(value) * 1880.0, 2)
            elif param == "so2":
                return round(float(value) * 2620.0, 2)
            elif param == "o3":
                return round(float(value) * 1960.0, 2)
            elif param == "nh3":
                return round(float(value) * 696.0, 2)
        elif "ppb" in u:
            if param == "no2":
                return round(float(value) * 1.88, 2)
            elif param == "so2":
                return round(float(value) * 2.62, 2)
            elif param == "o3":
                return round(float(value) * 1.96, 2)
            elif param == "nh3":
                return round(float(value) * 0.70, 2)

        return float(value)


def read_single_csv_file(fp: str | Path) -> pd.DataFrame | None:
    """Read a single OpenAQ compressed CSV partition."""
    try:
        df = pd.read_csv(fp, compression="gzip")
        if df.empty or "parameter" not in df.columns or "value" not in df.columns:
            return None
        return df
    except Exception:
        return None


def load_station_hourly_data(
    station_files: list[str], max_workers: int = 8
) -> pd.DataFrame:
    """Load and combine raw CSV files for a given station in parallel."""
    frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for df in executor.map(read_single_csv_file, station_files):
            if df is not None and not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate_station_hourly_to_daily(
    df_raw: pd.DataFrame,
    loc_id: int,
    station_meta: dict[str, Any],
    breakpoints: dict[str, Any],
    min_hours_per_day: int = 16,
    min_subindices: int = 3,
    require_pm: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Process raw hourly observations for a station into daily panel with CPCB AQI.
    """
    drops = {"invalid_value": 0, "unsupported_param": 0, "duplicate": 0}

    # 1. Standardize parameter names
    df = df_raw.copy()
    df["param_std"] = df["parameter"].apply(standardize_parameter_name)
    valid_param_mask = df["param_std"].isin(TARGET_POLLUTANTS)
    drops["unsupported_param"] += int((~valid_param_mask).sum())
    df = df[valid_param_mask].copy()

    # 2. Parse timestamps to Asia/Kolkata
    try:
        df["dt_utc"] = pd.to_datetime(df["datetime"], utc=True)
        df["dt_ist"] = df["dt_utc"].dt.tz_convert("Asia/Kolkata")
        df["date"] = df["dt_ist"].dt.date
        df["hour"] = df["dt_ist"].dt.hour
    except Exception as exc:
        logger.warning(f"Failed to parse datetimes for station {loc_id}: {exc}")
        return pd.DataFrame(), drops

    # 3. Unit conversion & value cleaning
    converted_vals = []
    for _, row in df.iterrows():
        val = row["value"]
        p = row["param_std"]
        u = row.get("units", "µg/m³")
        c_val = convert_pollutant_unit(p, val, u)
        converted_vals.append(c_val)

    df["clean_value"] = converted_vals
    invalid_mask = df["clean_value"].isna() | (df["clean_value"] < 0) | (df["clean_value"] > 5000)
    drops["invalid_value"] += int(invalid_mask.sum())
    df = df[~invalid_mask].copy()

    if df.empty:
        return pd.DataFrame(), drops

    # 4. Hourly deduplication
    before_dedup = len(df)
    df_hourly = (
        df.groupby(["date", "hour", "param_std"])["clean_value"]
        .mean()
        .reset_index()
    )
    drops["duplicate"] += before_dedup - len(df_hourly)

    # 5. Daily Aggregation
    daily_records = []
    unique_dates = sorted(df_hourly["date"].unique())

    for d in unique_dates:
        day_slice = df_hourly[df_hourly["date"] == d]
        row_dict: dict[str, Any] = {
            "location_id": loc_id,
            "station_name": station_meta.get("name", f"Station_{loc_id}"),
            "city": station_meta.get("city", "Unknown"),
            "latitude": station_meta.get("latitude", np.nan),
            "longitude": station_meta.get("longitude", np.nan),
            "date": pd.to_datetime(d),
        }

        sub_indices: dict[str, float | None] = {}

        for p in TARGET_POLLUTANTS:
            p_slice = day_slice[day_slice["param_std"] == p]
            n_hours = len(p_slice)
            row_dict[f"n_hours_{p}"] = n_hours

            if n_hours < min_hours_per_day:
                row_dict[p] = np.nan
                sub_indices[p] = None
                continue

            if p in ["co", "o3"]:
                # Running 8-hour mean max for CO and O3
                hours_series = pd.Series(np.nan, index=range(24))
                for _, hr_row in p_slice.iterrows():
                    hours_series[int(hr_row["hour"])] = hr_row["clean_value"]
                # 8h rolling mean (min 6 valid hours)
                rolling_8h = hours_series.rolling(window=8, min_periods=6).mean()
                daily_val = rolling_8h.max()
                if math.isnan(daily_val):
                    daily_val = p_slice["clean_value"].mean()
            else:
                # 24h mean for particulates and other gases
                daily_val = p_slice["clean_value"].mean()

            row_dict[p] = round(daily_val, 2)
            sub_indices[p] = calc_sub_index(p, daily_val, breakpoints)

        # Record sub-indices in output
        for p, s_val in sub_indices.items():
            row_dict[f"sub_{p}"] = s_val

        # Compute overall AQI
        overall_aqi, aqi_cat, dom_pollutant = calc_overall_aqi(
            sub_indices, min_subindices=min_subindices, require_pm=require_pm
        )
        row_dict["aqi"] = overall_aqi
        row_dict["aqi_category"] = aqi_cat
        row_dict["dominant_pollutant"] = dom_pollutant
        row_dict["is_aqi_valid"] = overall_aqi is not None

        daily_records.append(row_dict)

    if not daily_records:
        return pd.DataFrame(), drops

    return pd.DataFrame(daily_records), drops


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Hourly to Daily Panel & AQI Computation")
    parser.add_argument(
        "--raw-openaq-dir",
        type=Path,
        default=Path("data/raw/openaq"),
        help="Directory containing raw OpenAQ CSV files",
    )
    parser.add_argument(
        "--catalog-file",
        type=Path,
        default=Path("data/raw/station_catalog.csv"),
        help="Station catalog CSV",
    )
    parser.add_argument(
        "--stations-file",
        type=Path,
        default=Path("configs/stations_selected.csv"),
        help="Stations selected configuration CSV",
    )
    parser.add_argument(
        "--breakpoints-file",
        type=Path,
        default=Path("configs/cpcb_breakpoints.yaml"),
        help="CPCB breakpoints YAML file",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/interim/pollutants_daily.parquet"),
        help="Output daily pollutants parquet file",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("reports/metrics/hourly_to_daily_stats.json"),
        help="Output stage statistics JSON",
    )
    parser.add_argument(
        "--max-stations",
        type=int,
        default=None,
        help="Optional max stations to process (for fast verification)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Thread pool workers for reading CSV files",
    )
    args = parser.parse_args()

    setup_logging(stage="hourly_to_daily")
    logger.info("Starting Phase 4 Step 006: Hourly to Daily Aggregation & CPCB AQI Calculation...")

    params = load_params()
    min_hours = params.get("aggregation", {}).get("min_hours_per_day", 16)
    min_sub = params.get("aqi", {}).get("min_subindices", 3)
    require_pm = params.get("aqi", {}).get("require_pm", True)
    breakpoints = load_cpcb_breakpoints(args.breakpoints_file)

    # Load station catalog for names, coordinates, and cities
    station_meta_map: dict[int, dict[str, Any]] = {}
    if args.catalog_file.exists():
        df_cat = pd.read_csv(args.catalog_file)
        for _, r in df_cat.iterrows():
            loc_id = int(r["location_id"])
            station_meta_map[loc_id] = {
                "name": str(r.get("name", f"Station_{loc_id}")),
                "city": str(r.get("city", "Unknown")),
                "latitude": float(r.get("latitude", np.nan)),
                "longitude": float(r.get("longitude", np.nan)),
            }

    # Find station folders in raw-openaq-dir
    raw_dirs = list(args.raw_openaq_dir.glob("**/locationid=*"))
    logger.info(f"Discovered {len(raw_dirs)} station directories under {args.raw_openaq_dir}")

    # Prioritize stations covering all target cities per params.yaml
    target_cities = params.get("data", {}).get(
        "cities",
        ["Bengaluru", "Delhi", "Mumbai", "Chennai", "Kolkata", "Hyderabad", "Pune", "Lucknow", "Ahmedabad"],
    )

    # Group location dirs by city
    loc_by_id = {}
    for d in raw_dirs:
        m = re.search(r"locationid=(\d+)", d.name)
        if m:
            loc_by_id[int(m.group(1))] = d

    prioritized_dirs = []
    # Pick up to 3 for Bengaluru, up to 2 for Delhi/Mumbai, up to 1 for others
    for city in target_cities:
        city_stations = [
            (loc_id, station_meta_map[loc_id])
            for loc_id in loc_by_id
            if loc_id in station_meta_map and station_meta_map[loc_id].get("city") == city
        ]
        take_n = 3 if city == "Bengaluru" else (2 if city in ["Delhi", "Mumbai"] else 1)
        for loc_id, _ in city_stations[:take_n]:
            if loc_by_id[loc_id] not in prioritized_dirs:
                prioritized_dirs.append(loc_by_id[loc_id])

    # Add any remaining directories
    for d in raw_dirs:
        if d not in prioritized_dirs:
            prioritized_dirs.append(d)

    location_dirs = prioritized_dirs
    if args.max_stations:
        location_dirs = location_dirs[: args.max_stations]
        logger.info(f"Limiting execution to {args.max_stations} prioritized stations across target cities.")

    all_station_panels: list[pd.DataFrame] = []
    total_raw_rows = 0
    total_drops = {"invalid_value": 0, "unsupported_param": 0, "duplicate": 0}

    for idx, loc_dir in enumerate(location_dirs, start=1):
        match = re.search(r"locationid=(\d+)", loc_dir.name)
        if not match:
            continue
        loc_id = int(match.group(1))

        csv_files = [str(p) for p in loc_dir.glob("**/*.csv.gz")]
        if not csv_files:
            continue

        meta = station_meta_map.get(
            loc_id, {"name": f"Station_{loc_id}", "city": "Unknown", "latitude": np.nan, "longitude": np.nan}
        )
        logger.info(
            f"[{idx}/{len(location_dirs)}] Processing Station {loc_id} ({meta['name']}, {meta['city']}) - {len(csv_files)} files..."
        )

        df_station_raw = load_station_hourly_data(csv_files, max_workers=args.max_workers)
        if df_station_raw.empty:
            continue

        total_raw_rows += len(df_station_raw)
        df_daily, drops = aggregate_station_hourly_to_daily(
            df_station_raw,
            loc_id=loc_id,
            station_meta=meta,
            breakpoints=breakpoints,
            min_hours_per_day=min_hours,
            min_subindices=min_sub,
            require_pm=require_pm,
        )

        for k, v in drops.items():
            total_drops[k] += v

        if not df_daily.empty:
            all_station_panels.append(df_daily)

    if not all_station_panels:
        logger.error("No daily panel data produced across all stations.")
        sys.exit(1)

    combined_daily_panel = pd.concat(all_station_panels, ignore_index=True)
    combined_daily_panel = combined_daily_panel.sort_values(["city", "location_id", "date"]).reset_index(drop=True)

    # Save output
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_daily_panel.to_parquet(args.output_file, index=False)
    logger.info(f"Saved {len(combined_daily_panel):,} daily panel records to {args.output_file}")

    # Track stage stats
    input_paths = [str(args.raw_openaq_dir), str(args.breakpoints_file)]
    with track_stage("hourly_to_daily", inputs=input_paths, outputs=[str(args.output_file)]) as st:
        st.record_output_frame(combined_daily_panel)
        for reason, count in total_drops.items():
            if count > 0:
                st.drop(reason, count)

    n_valid_aqi = int(combined_daily_panel["is_aqi_valid"].sum())
    aqi_valid_frac = round(n_valid_aqi / max(len(combined_daily_panel), 1) * 100, 1)

    print("\n" + "=" * 65)
    print("HOURLY TO DAILY AGGREGATION & AQI SUMMARY")
    print("=" * 65)
    print(f"Stations Processed:            {len(location_dirs)}")
    print(f"Raw Hourly Rows Read:          {total_raw_rows:,}")
    print(f"Station-Days Generated:        {len(combined_daily_panel):,}")
    print(f"Valid AQI Station-Days:        {n_valid_aqi:,} ({aqi_valid_frac}%)")
    print(f"Unique Cities:                 {combined_daily_panel['city'].nunique()}")
    print(f"Date Range:                    {combined_daily_panel['date'].min()} to {combined_daily_panel['date'].max()}")
    print(f"Output File:                   {args.output_file}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
