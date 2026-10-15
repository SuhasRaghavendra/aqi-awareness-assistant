"""
src/data/merge.py
-----------------
Phase 4, Step 006: Merge Daily Pollutants Panel with Daily Weather Enrichment.

Reads daily pollutants panel from data/interim/pollutants_daily.parquet,
extracts hourly Open-Meteo ERA5 weather variables from data/raw/weather,
aggregates weather to daily metrics (mean/min/max temperature, mean humidity,
mean surface pressure, total daily precipitation, mean wind speed and direction),
and merges with the pollutants panel on (location_id, date).

Outputs:
  - data/interim/station_day.parquet
  - reports/metrics/merge_stats.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_params
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.stage_stats import track_stage

logger = get_logger("merge")

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
]


def load_station_daily_weather(weather_dir: Path, loc_id: int) -> pd.DataFrame:
    """Load and aggregate hourly weather records for a single station location ID."""
    # Check for json directory data/raw/weather/<loc_id>/*.json
    json_files = glob.glob(str(weather_dir / str(loc_id) / "*.json"))
    parquet_files = glob.glob(str(weather_dir / f"locationid={loc_id}" / "*.parquet"))

    frames = []
    if json_files:
        for fp in json_files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                hourly = data.get("hourly", {})
                if hourly and "time" in hourly:
                    df = pd.DataFrame(hourly)
                    frames.append(df)
            except Exception as exc:
                logger.warning(f"Failed to read weather file {fp}: {exc}")

    if parquet_files:
        for fp in parquet_files:
            try:
                df = pd.read_parquet(fp)
                frames.append(df)
            except Exception as exc:
                logger.warning(f"Failed to read weather parquet {fp}: {exc}")

    if not frames:
        return pd.DataFrame()

    df_weather = pd.concat(frames, ignore_index=True)
    if df_weather.empty or "time" not in df_weather.columns:
        return pd.DataFrame()

    # Parse time to date (Asia/Kolkata)
    df_weather["dt"] = pd.to_datetime(df_weather["time"])
    df_weather["date"] = pd.to_datetime(df_weather["dt"].dt.date)

    # Compute daily weather aggregations
    agg_funcs: dict[str, Any] = {}
    if "temperature_2m" in df_weather.columns:
        agg_funcs["temperature_2m"] = ["mean", "min", "max"]
    if "relative_humidity_2m" in df_weather.columns:
        agg_funcs["relative_humidity_2m"] = ["mean"]
    if "surface_pressure" in df_weather.columns:
        agg_funcs["surface_pressure"] = ["mean"]
    if "precipitation" in df_weather.columns:
        agg_funcs["precipitation"] = ["sum"]
    if "wind_speed_10m" in df_weather.columns:
        agg_funcs["wind_speed_10m"] = ["mean"]
    if "wind_direction_10m" in df_weather.columns:
        agg_funcs["wind_direction_10m"] = ["mean"]

    daily_w = df_weather.groupby("date").agg(agg_funcs)

    # Flatten multi-level columns
    flat_cols = []
    for col, stat in daily_w.columns:
        if stat in ["min", "max", "sum"]:
            flat_cols.append(f"{col}_{stat}")
        else:
            flat_cols.append(f"{col}_mean")
    daily_w.columns = flat_cols
    daily_w = daily_w.reset_index()
    daily_w["location_id"] = loc_id

    return daily_w


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Merge Pollutants Panel with Daily Weather")
    parser.add_argument(
        "--pollutants-file",
        type=Path,
        default=Path("data/interim/pollutants_daily.parquet"),
        help="Input daily pollutants parquet file",
    )
    parser.add_argument(
        "--weather-dir",
        type=Path,
        default=Path("data/raw/weather"),
        help="Directory containing raw weather records",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/interim/station_day.parquet"),
        help="Output merged station-day panel parquet file",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("reports/metrics/merge_stats.json"),
        help="Output stage statistics JSON",
    )
    args = parser.parse_args()

    setup_logging(stage="merge")
    logger.info("Starting Phase 4: Merging Pollutants Panel with Daily Weather...")

    if not args.pollutants_file.exists():
        logger.error(f"Pollutants file not found at {args.pollutants_file}. Run hourly_to_daily first.")
        raise FileNotFoundError(f"{args.pollutants_file} does not exist.")

    df_pollutants = pd.read_parquet(args.pollutants_file)
    logger.info(f"Loaded {len(df_pollutants):,} pollutant records from {args.pollutants_file}")
    df_pollutants["date"] = pd.to_datetime(df_pollutants["date"])

    unique_stations = df_pollutants["location_id"].unique()
    logger.info(f"Merging weather for {len(unique_stations)} distinct stations...")

    weather_frames = []
    for loc_id in unique_stations:
        w_df = load_station_daily_weather(args.weather_dir, int(loc_id))
        if not w_df.empty:
            weather_frames.append(w_df)

    if weather_frames:
        combined_weather = pd.concat(weather_frames, ignore_index=True)
        logger.info(f"Generated {len(combined_weather):,} total daily weather records.")
        # Left merge pollutants with weather
        merged_panel = df_pollutants.merge(
            combined_weather, on=["location_id", "date"], how="left"
        )
    else:
        logger.warning("No matching weather files found for stations. Adding empty weather features.")
        merged_panel = df_pollutants.copy()
        for v in WEATHER_VARIABLES:
            merged_panel[f"{v}_mean"] = np.nan

    merged_panel = merged_panel.sort_values(["city", "location_id", "date"]).reset_index(drop=True)

    # Save output
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_panel.to_parquet(args.output_file, index=False)
    logger.info(f"Saved {len(merged_panel):,} merged records to {args.output_file}")

    # Track stage stats
    inputs = [str(args.pollutants_file), str(args.weather_dir)]
    with track_stage("merge", inputs=inputs, outputs=[str(args.output_file)]) as st:
        st.record_input_frame(df_pollutants)
        st.record_output_frame(merged_panel)

    print("\n" + "=" * 65)
    print("STATION-DAY WEATHER MERGE SUMMARY")
    print("=" * 65)
    print(f"Total Rows In (Pollutants):    {len(df_pollutants):,}")
    print(f"Total Rows Out (Merged):       {len(merged_panel):,}")
    print(f"Columns In:                    {len(df_pollutants.columns)}")
    print(f"Columns Out:                   {len(merged_panel.columns)}")
    print(f"Weather Columns Added:         {[c for c in merged_panel.columns if 'temperature' in c or 'wind' in c or 'humidity' in c or 'pressure' in c or 'precipitation' in c]}")
    print(f"Output File:                   {args.output_file}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
