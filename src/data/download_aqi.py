"""
src/data/download_aqi.py
------------------------
Phase 1 - Step A: Download city-level daily AQI labels from XKDR India Air
Quality Database (https://airquality.xkdr.org).

The XKDR database harmonises CPCB's CAAQM network (550+ stations, 2009-present)
into clean CSV files queryable via a free-signup REST API.

Strategy:
  - Query parameter=PM2.5 and parameter=PM10 (XKDR does NOT support parameter=AQI directly)
  - Convert PM2.5 / PM10 concentrations to AQI sub-indices using CPCB breakpoints
  - Aggregate station readings to city x day mean AQI
  - Query year-by-year in parallel to stay within API row limits and finish quickly

Usage:
    python -m src.data.download_aqi [--start 2019-01-01] [--end 2026-09-30]

Outputs:
    data/raw/aqi/city_aqi_daily.parquet   (city x date AQI, aggregated)
    data/raw/aqi/station_aqi_raw.parquet  (station-level PM backup)
    reports/metrics/download_aqi_stats.json

Environment:
    XKDR_API_KEY   - free key from https://airquality.xkdr.org  (sign up)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import load_env, load_params
from src.features.aqi_utils import calc_sub_index
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("download_aqi")

# City name -> approximate lat/lon centroid
CITY_COORDS: dict[str, tuple[float, float]] = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Lucknow": (26.8467, 80.9462),
}

# XKDR city name mapping
XKDR_CITY_MAP: dict[str, str] = {
    "Delhi": "Delhi",
    "Mumbai": "Mumbai",
    "Bengaluru": "Bengaluru",
    "Chennai": "Chennai",
    "Kolkata": "Kolkata",
    "Hyderabad": "Hyderabad",
    "Pune": "Pune",
    "Ahmedabad": "Ahmedabad",
    "Lucknow": "Lucknow",
}

XKDR_BASE = "https://airquality.xkdr.org"

# Parameters to query
POLLUTANT_PARAMS = ["PM2.5", "PM10"]


def get_xkdr_api_key() -> str:
    """Return XKDR API key from env."""
    load_env()
    key = os.environ.get("XKDR_API_KEY", "").strip()
    if not key:
        key = "aqi_demo_wbf92Qx21zX-Wa_Tg8Dx1nXe"
        logger.warning(
            "XKDR_API_KEY not set. Using public demo key -- limited to 2024 data "
            "and 10 000 rows per query. Set XKDR_API_KEY in .env for full access."
        )
    return key


def fetch_xkdr_year(
    city: str,
    year: int,
    parameter: str,
    api_key: str,
    retries: int = 3,
    delay: float = 1.5,
) -> pd.DataFrame:
    """
    Fetch one year of daily station-level data for a city + pollutant from XKDR.
    """
    xkdr_city = XKDR_CITY_MAP.get(city, city)
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    params: dict[str, Any] = {
        "city": xkdr_city,
        "parameter": parameter,
        "start": start,
        "end": end,
        "agg": "daily",
        "format": "csv",
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{XKDR_BASE}/v1/measurements"

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=90)
            if resp.status_code == 200:
                text = resp.text.strip()
                if not text:
                    return pd.DataFrame()
                df = pd.read_csv(StringIO(text))
                if df.empty:
                    return pd.DataFrame()
                df["city"] = city
                df["parameter"] = parameter
                return df
            else:
                logger.warning(
                    f"[{city}/{year}/{parameter}] attempt {attempt}/{retries}: "
                    f"HTTP {resp.status_code} -- {resp.text[:150]}"
                )
        except Exception as exc:
            logger.warning(
                f"[{city}/{year}/{parameter}] attempt {attempt}/{retries} exception: {exc}"
            )
        time.sleep(delay * attempt)

    return pd.DataFrame()


def pm25_to_aqi(concentration: float | None) -> float | None:
    """Convert PM2.5 concentration (ug/m3) to CPCB AQI sub-index."""
    if concentration is None:
        return None
    return calc_sub_index("pm25", concentration)


def pm10_to_aqi(concentration: float | None) -> float | None:
    """Convert PM10 concentration (ug/m3) to CPCB AQI sub-index."""
    if concentration is None:
        return None
    return calc_sub_index("pm10", concentration)


def download_city_aqi(
    city: str,
    start_year: int,
    end_year: int,
    api_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download and convert PM2.5 (and PM10) data to AQI for a city using a thread pool.
    """
    tasks = [
        (year, param) for year in range(start_year, end_year + 1) for param in POLLUTANT_PARAMS
    ]

    all_frames: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {
            executor.submit(fetch_xkdr_year, city, yr, p, api_key): (yr, p) for yr, p in tasks
        }
        for future in as_completed(future_map):
            yr, p = future_map[future]
            try:
                df = future.result()
                if not df.empty:
                    all_frames.append(df)
            except Exception as exc:
                logger.warning(f"[{city}/{yr}/{p}] failed: {exc}")

    if not all_frames:
        logger.warning(f"[{city}] No data retrieved for any parameter")
        return pd.DataFrame(), pd.DataFrame()

    df_all = pd.concat(all_frames, ignore_index=True)

    # Normalize columns: XKDR returns period_start, mean, station_id
    if "period_start" not in df_all.columns:
        for col in ["date", "datetime", "time"]:
            if col in df_all.columns:
                df_all = df_all.rename(columns={col: "period_start"})
                break

    if "mean" not in df_all.columns:
        for col in ["value", "concentration"]:
            if col in df_all.columns:
                df_all = df_all.rename(columns={col: "mean"})
                break

    # Parse date
    df_all["date"] = pd.to_datetime(df_all["period_start"], errors="coerce").dt.date.astype(str)
    df_all["concentration"] = pd.to_numeric(df_all["mean"], errors="coerce")

    # Compute AQI sub-index per station row based on parameter
    df_all["aqi_sub"] = float("nan")
    pm25_mask = df_all["parameter"].astype(str).str.contains("2.5|pm25", case=False, na=False)
    pm10_mask = df_all["parameter"].astype(str).str.contains("10|pm10", case=False, na=False)
    if pm25_mask.any():
        df_all.loc[pm25_mask, "aqi_sub"] = df_all.loc[pm25_mask, "concentration"].apply(pm25_to_aqi)
    if pm10_mask.any():
        df_all.loc[pm10_mask, "aqi_sub"] = df_all.loc[pm10_mask, "concentration"].apply(pm10_to_aqi)

    # Aggregate to city x day: take mean AQI sub-index per day
    df_valid = df_all.dropna(subset=["date", "aqi_sub"]).copy()
    df_valid["date"] = pd.to_datetime(df_valid["date"])

    if df_valid.empty:
        logger.warning(f"[{city}] All PM conversions resulted in NaN")
        return df_all, pd.DataFrame()

    city_day = (
        df_valid.groupby("date")["aqi_sub"].agg(aqi="mean", station_count="count").reset_index()
    )
    city_day["aqi"] = city_day["aqi"].round(1)
    city_day["city"] = city

    logger.info(
        f"[{city}] {len(city_day):,} city-days, "
        f"AQI range {city_day['aqi'].min():.0f}-{city_day['aqi'].max():.0f}"
    )

    return df_all, city_day


def main() -> None:
    setup_logging(stage="download_aqi")
    parser = argparse.ArgumentParser(description="Download daily city AQI labels from XKDR")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-09-30")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/aqi"))
    args = parser.parse_args()

    params = load_params()
    cities: list[str] = params["data"]["cities"]
    api_key = get_xkdr_api_key()

    start_year = int(args.start[:4])
    end_year = int(args.end[:4])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_station_frames: list[pd.DataFrame] = []
    all_city_day_frames: list[pd.DataFrame] = []

    for city in cities:
        logger.info(f"Downloading PM2.5/PM10 -> AQI for {city} ({start_year}-{end_year}) ...")
        station_df, city_day_df = download_city_aqi(city, start_year, end_year, api_key)
        if not station_df.empty:
            station_df["city"] = city
            all_station_frames.append(station_df)
        if not city_day_df.empty:
            all_city_day_frames.append(city_day_df)

    if not all_city_day_frames:
        raise RuntimeError("No AQI data was computed. Check XKDR_API_KEY and network access.")

    # Save station-level backup
    if all_station_frames:
        df_station_all = pd.concat(all_station_frames, ignore_index=True)
        station_path = args.output_dir / "station_aqi_raw.parquet"
        df_station_all.to_parquet(station_path, index=False)
        logger.info(f"Saved station-level data -> {station_path} ({len(df_station_all):,} rows)")
    else:
        pd.DataFrame(columns=["city", "date", "concentration"]).to_parquet(
            args.output_dir / "station_aqi_raw.parquet", index=False
        )

    # Aggregate city x day
    city_day = pd.concat(all_city_day_frames, ignore_index=True)
    city_day = city_day.sort_values(["city", "date"]).reset_index(drop=True)
    city_day_path = args.output_dir / "city_aqi_daily.parquet"
    city_day.to_parquet(city_day_path, index=False)
    logger.info(f"Saved city-day AQI -> {city_day_path} ({len(city_day):,} rows)")

    # Summary stats
    coverage: dict[str, int] = (
        city_day.groupby("city")["date"].count().rename("days").astype(int).to_dict()
    )
    stats: dict[str, Any] = {
        "downloaded_at": datetime.now().isoformat(),
        "start": args.start,
        "end": args.end,
        "cities": cities,
        "total_city_day_rows": int(len(city_day)),
        "coverage_by_city": coverage,
        "aqi_range": {
            "min": float(city_day["aqi"].min()),
            "max": float(city_day["aqi"].max()),
            "mean": float(city_day["aqi"].mean()),
        },
    }
    metrics_path = Path("reports/metrics/download_aqi_stats.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("AQI DOWNLOAD SUMMARY")
    print("=" * 60)
    for city, days in coverage.items():
        print(f"  {city:<12} {days:>5} city-days")
    print(f"\nTotal city-day rows: {len(city_day):,}")
    print(f"AQI range: {stats['aqi_range']['min']:.0f} - {stats['aqi_range']['max']:.0f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
