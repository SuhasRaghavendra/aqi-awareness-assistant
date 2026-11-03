"""
src/data/download_fires.py
--------------------------
Phase 1 – Step B: Download daily fire/burning event data from NASA FIRMS
(Fire Information for Resource Management System) using the VIIRS 375m
Standard Processing archive.

Data covers:
  - Crop stubble burning (Punjab/Haryana, Oct–Nov)
  - Industrial/waste burning
  - Forest fires
  - All open burning within 500 km radius of each city

API Reference: https://firms.modaps.eosdis.nasa.gov/api/
Sensor: VIIRS_SNPP_SP (Suomi-NPP VIIRS Standard Processing, 375m)

Usage:
    python -m src.data.download_fires [--start 2019-01-01] [--end 2026-09-30]

Environment:
    FIRMS_MAP_KEY   – free key from https://firms.modaps.eosdis.nasa.gov/api/map_key/
                      (email signup, instant)

Outputs:
    data/raw/fires/india_fires_raw.parquet      (all fire detections, India)
    data/raw/fires/city_fires_daily.parquet     (city × date aggregated)
    reports/metrics/download_fires_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import load_params
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.stage_stats import track_stage

logger = get_logger("download_fires")

# City centroids for distance computation
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

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/country/csv"
SENSOR = "VIIRS_SNPP_SP"
COUNTRY = "IND"

# Batch size in days to avoid FIRMS API timeouts (max ~10 days per call)
BATCH_DAYS = 10


def get_firms_key() -> str:
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "FIRMS_MAP_KEY not set. Get a free key at:\n"
            "  https://firms.modaps.eosdis.nasa.gov/api/map_key/\n"
            "Then add it to your .env file: FIRMS_MAP_KEY=your_key_here"
        )
    return key


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_firms_batch(
    api_key: str,
    start_date: date,
    n_days: int,
    retries: int = 3,
) -> pd.DataFrame:
    """Fetch FIRMS data for a batch of n_days starting from start_date."""
    date_str = start_date.strftime("%Y-%m-%d")
    url = f"{FIRMS_BASE}/{api_key}/{SENSOR}/{COUNTRY}/{n_days}/{date_str}"

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code == 200:
                text = resp.text.strip()
                if not text or "latitude" not in text:
                    return pd.DataFrame()
                df = pd.read_csv(StringIO(text))
                return df
            logger.warning(
                f"FIRMS batch {date_str}+{n_days}d attempt {attempt}/{retries}: "
                f"HTTP {resp.status_code}"
            )
        except Exception as exc:
            logger.warning(f"FIRMS batch {date_str}+{n_days}d attempt {attempt}: {exc}")
        time.sleep(2.0 * attempt)

    return pd.DataFrame()


def download_all_fires(
    api_key: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download all FIRMS VIIRS fire detections for India in batches."""
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()

    frames: list[pd.DataFrame] = []
    current = start_date
    total_days = (end_date - start_date).days + 1
    fetched_days = 0

    while current <= end_date:
        remaining = (end_date - current).days + 1
        batch_n = min(BATCH_DAYS, remaining)
        logger.info(
            f"Downloading fires: {current} + {batch_n} days "
            f"({fetched_days}/{total_days} days done)"
        )
        batch_df = fetch_firms_batch(api_key, current, batch_n)
        if not batch_df.empty:
            frames.append(batch_df)
            logger.debug(f"  → {len(batch_df):,} fire detections")
        current += timedelta(days=batch_n)
        fetched_days += batch_n
        time.sleep(0.3)   # polite throttle

    if not frames:
        raise RuntimeError("No fire data was downloaded from FIRMS. Check FIRMS_MAP_KEY.")

    df_all = pd.concat(frames, ignore_index=True)
    logger.info(f"Total raw FIRMS records: {len(df_all):,}")
    return df_all


def normalise_firms(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise FIRMS column names and types."""
    df = df.copy()
    # Standard VIIRS columns: latitude, longitude, bright_ti4, scan, track,
    # acq_date, acq_time, satellite, instrument, confidence, version, bright_ti5,
    # frp (fire radiative power in MW), daynight, type
    df["date"] = pd.to_datetime(df.get("acq_date", df.get("date", "")), errors="coerce")
    df["latitude"] = pd.to_numeric(df.get("latitude", float("nan")), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude", float("nan")), errors="coerce")
    df["frp"] = pd.to_numeric(df.get("frp", 0.0), errors="coerce").fillna(0.0)
    # Confidence: 'l','n','h' or 0-100
    if "confidence" in df.columns:
        conf = df["confidence"].astype(str).str.lower()
        conf_map = {"l": 33, "n": 67, "h": 95}
        df["confidence_pct"] = conf.map(conf_map).fillna(
            pd.to_numeric(df["confidence"], errors="coerce")
        )
    else:
        df["confidence_pct"] = 67.0
    return df.dropna(subset=["date", "latitude", "longitude"])


def aggregate_fires_to_city_day(
    df_fires: pd.DataFrame,
    cities: list[str],
    radius_km: float = 500.0,
) -> pd.DataFrame:
    """
    For each city × date, count fires and sum FRP within radius_km.
    Also compute minimum distance to any fire that day.
    """
    records: list[dict[str, Any]] = []

    # Pre-compute arrays for speed
    fire_lats = df_fires["latitude"].values
    fire_lons = df_fires["longitude"].values
    fire_frp = df_fires["frp"].values
    fire_dates = df_fires["date"].dt.date.values

    for city in cities:
        clat, clon = CITY_COORDS[city]
        logger.info(f"Computing fire proximity for {city} …")

        # Distance from city to every fire detection
        distances = [
            haversine_km(clat, clon, flat, flon)
            for flat, flon in zip(fire_lats, fire_lons)
        ]
        dist_arr = pd.Series(distances)
        within_mask = dist_arr <= radius_km

        df_city = df_fires.copy()
        df_city["distance_km"] = distances
        df_city_within = df_city[within_mask].copy()

        if df_city_within.empty:
            logger.warning(f"[{city}] No fires within {radius_km} km in dataset")
            continue

        df_city_within["city"] = city
        city_day = (
            df_city_within.groupby(["city", "date"])
            .agg(
                fire_count=("frp", "count"),
                fire_frp_sum=("frp", "sum"),
                fire_frp_mean=("frp", "mean"),
                fire_dist_min_km=("distance_km", "min"),
            )
            .reset_index()
        )
        records.append(city_day)

    if not records:
        return pd.DataFrame()

    result = pd.concat(records, ignore_index=True)
    result["fire_count_500km"] = result["fire_count"]
    result["fire_frp_sum_500km"] = result["fire_frp_sum"].round(2)
    result["fire_frp_sum"] = result["fire_frp_sum"].round(2)
    result["fire_frp_mean"] = result["fire_frp_mean"].round(2)
    result["fire_dist_min_km"] = result["fire_dist_min_km"].round(1)
    return result.sort_values(["city", "date"]).reset_index(drop=True)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Download NASA FIRMS fire data for India")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end",   default="2026-09-30")
    parser.add_argument("--radius-km", type=float, default=500.0)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/fires"))
    args = parser.parse_args()

    params = load_params()
    cities: list[str] = params["data"]["cities"]

    api_key = get_firms_key()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download raw fire detections
    df_raw = download_all_fires(api_key, args.start, args.end)
    df_raw = normalise_firms(df_raw)

    raw_path = args.output_dir / "india_fires_raw.parquet"
    df_raw.to_parquet(raw_path, index=False)
    logger.info(f"Saved raw fire data → {raw_path} ({len(df_raw):,} records)")

    # Aggregate to city × day
    df_city = aggregate_fires_to_city_day(df_raw, cities, radius_km=args.radius_km)
    city_path = args.output_dir / "city_fires_daily.parquet"
    df_city.to_parquet(city_path, index=False)
    logger.info(f"Saved city-day fire features → {city_path} ({len(df_city):,} rows)")

    # Stats
    stats: dict[str, Any] = {
        "downloaded_at": datetime.now().isoformat(),
        "start": args.start,
        "end": args.end,
        "radius_km": args.radius_km,
        "total_raw_fire_detections": int(len(df_raw)),
        "total_city_day_rows": int(len(df_city)),
        "fire_coverage_by_city": (
            df_city.groupby("city")["fire_count"].sum().astype(int).to_dict()
            if not df_city.empty else {}
        ),
    }
    metrics_path = Path("reports/metrics/download_fires_stats.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("FIRE DATA DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Raw fire detections (India):  {len(df_raw):,}")
    print(f"City-day aggregated rows:     {len(df_city):,}")
    for city, cnt in stats["fire_coverage_by_city"].items():
        print(f"  {city:<12}: {cnt:>8,} total fires within {args.radius_km:.0f} km")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
