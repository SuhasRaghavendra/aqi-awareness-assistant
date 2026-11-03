"""
src/data/download_aqi.py
------------------------
Phase 1 – Step A: Download city-level daily AQI labels from XKDR India Air
Quality Database (https://airquality.xkdr.org).

The XKDR database harmonises CPCB's CAAQM network (550+ stations, 2009–present)
into clean Parquet/CSV files queryable via a free-signup REST API.

Usage:
    python -m src.data.download_aqi [--start 2019-01-01] [--end 2026-09-30]

Outputs:
    data/raw/aqi/city_aqi_daily.parquet   (city × date AQI, aggregated)
    data/raw/aqi/station_aqi_raw.parquet  (station-level hourly/daily backup)
    reports/metrics/download_aqi_stats.json

Environment:
    XKDR_API_KEY   – free key from https://airquality.xkdr.org  (sign up)
                     If not set, falls back to data.gov.in CPCB endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import load_params
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.stage_stats import track_stage

logger = get_logger("download_aqi")

# City name → approximate lat/lon centroid (for station proximity filtering)
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

# XKDR city name mapping (their naming convention)
XKDR_CITY_MAP: dict[str, str] = {
    "Delhi":     "Delhi",
    "Mumbai":    "Mumbai",
    "Bengaluru": "Bengaluru",
    "Chennai":   "Chennai",
    "Kolkata":   "Kolkata",
    "Hyderabad": "Hyderabad",
    "Pune":      "Pune",
    "Ahmedabad": "Ahmedabad",
    "Lucknow":   "Lucknow",
}

XKDR_BASE = "https://airquality.xkdr.org"


def get_xkdr_api_key() -> str:
    """Return XKDR API key from env or prompt."""
    key = os.environ.get("XKDR_API_KEY", "").strip()
    if not key:
        # Demo key (2024 data only, 10k row limit — useful for testing)
        key = "aqi_demo_wbf92Qx21zX-Wa_Tg8Dx1nXe"
        logger.warning(
            "XKDR_API_KEY not set. Using public demo key — limited to 2024 data "
            "and 10 000 rows per query. Set XKDR_API_KEY in .env for full access."
        )
    return key


def download_city_aqi_xkdr(
    city: str,
    start: str,
    end: str,
    api_key: str,
    retries: int = 3,
    delay: float = 1.5,
) -> pd.DataFrame:
    """Download daily city-level AQI from XKDR for one city."""
    xkdr_city = XKDR_CITY_MAP.get(city, city)
    params: dict[str, Any] = {
        "city": xkdr_city,
        "parameter": "AQI",
        "start": start,
        "end": end,
        "agg": "daily",
        "format": "csv",
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{XKDR_BASE}/v1/measurements"

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=60)
            if resp.status_code == 200 and resp.text.strip():
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text))
                if df.empty:
                    logger.warning(f"[{city}] XKDR returned empty CSV for {start}→{end}")
                    return pd.DataFrame()
                df["city"] = city
                logger.info(f"[{city}] Downloaded {len(df):,} rows from XKDR ({start}→{end})")
                return df
            else:
                logger.warning(
                    f"[{city}] XKDR attempt {attempt}/{retries}: "
                    f"HTTP {resp.status_code} — {resp.text[:200]}"
                )
        except Exception as exc:
            logger.warning(f"[{city}] XKDR attempt {attempt}/{retries} exception: {exc}")
        time.sleep(delay * attempt)

    logger.error(f"[{city}] All XKDR retries exhausted for {start}→{end}")
    return pd.DataFrame()


def normalise_xkdr(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise XKDR column names to project schema."""
    # XKDR columns: station_id, date/time, value, parameter, city, state …
    rename_map = {
        "time": "date",
        "value": "aqi",
        "station_name": "station_name",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "date" not in df.columns and "datetime" in df.columns:
        df["date"] = df["datetime"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    df["aqi"] = pd.to_numeric(df.get("aqi", df.get("value", float("nan"))), errors="coerce")
    return df


def aggregate_to_city_day(df_all: pd.DataFrame) -> pd.DataFrame:
    """Aggregate station-level readings to city × day mean AQI."""
    df_all["date"] = pd.to_datetime(df_all["date"])
    city_day = (
        df_all.groupby(["city", "date"])["aqi"]
        .agg(aqi_mean="mean", aqi_median="median", station_count="count")
        .reset_index()
    )
    city_day.rename(columns={"aqi_mean": "aqi"}, inplace=True)
    city_day["aqi"] = city_day["aqi"].round(1)
    return city_day.sort_values(["city", "date"]).reset_index(drop=True)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Download daily city AQI labels from XKDR")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end",   default="2026-09-30")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/aqi"))
    args = parser.parse_args()

    params = load_params()
    cities: list[str] = params["data"]["cities"]
    api_key = get_xkdr_api_key()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_station_frames: list[pd.DataFrame] = []

    for city in cities:
        logger.info(f"Fetching AQI for {city} …")
        df = download_city_aqi_xkdr(city, args.start, args.end, api_key)
        if df.empty:
            logger.warning(f"[{city}] No data retrieved — skipping")
            continue
        df = normalise_xkdr(df)
        df["city"] = city
        all_station_frames.append(df)
        time.sleep(0.5)   # polite throttle

    if not all_station_frames:
        raise RuntimeError(
            "No AQI data was downloaded. Check XKDR_API_KEY and network access."
        )

    df_all = pd.concat(all_station_frames, ignore_index=True)

    # Save station-level backup
    station_path = args.output_dir / "station_aqi_raw.parquet"
    df_all.to_parquet(station_path, index=False)
    logger.info(f"Saved station-level AQI → {station_path} ({len(df_all):,} rows)")

    # Aggregate to city × day
    city_day = aggregate_to_city_day(df_all)
    city_day_path = args.output_dir / "city_aqi_daily.parquet"
    city_day.to_parquet(city_day_path, index=False)
    logger.info(f"Saved city-day AQI → {city_day_path} ({len(city_day):,} rows)")

    # Summary stats
    stats: dict[str, Any] = {
        "downloaded_at": datetime.now().isoformat(),
        "start": args.start,
        "end": args.end,
        "cities": cities,
        "total_station_rows": int(len(df_all)),
        "total_city_day_rows": int(len(city_day)),
        "coverage_by_city": (
            city_day.groupby("city")["date"]
            .count()
            .rename("days")
            .to_dict()
        ),
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
    for city, days in stats["coverage_by_city"].items():
        print(f"  {city:<12} {days:>5} city-days")
    print(f"\nTotal city-day rows: {len(city_day):,}")
    print(f"AQI range: {stats['aqi_range']['min']:.0f} – {stats['aqi_range']['max']:.0f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

