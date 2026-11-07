"""
src/data/download_fires.py
--------------------------
Phase 1 - Step B: Download daily fire/burning event data from NASA FIRMS
(Fire Information for Resource Management System) using the VIIRS 375m
Standard Processing archive.

Data covers:
  - Crop stubble burning (Punjab/Haryana, Oct-Nov)
  - Industrial/waste burning
  - Forest fires
  - All open burning within 500 km radius of each city

API Reference: https://firms.modaps.eosdis.nasa.gov/api/
Sensor: VIIRS_SNPP_SP (Suomi-NPP VIIRS Standard Processing, 375m)
        VIIRS_SNPP_NRT (near-real-time for recent dates)

Usage:
    python -m src.data.download_fires [--start 2019-01-01] [--end 2026-09-30]

Environment:
    FIRMS_MAP_KEY   - free key from https://firms.modaps.eosdis.nasa.gov/api/map_key/
                      (email signup, instant)

Outputs:
    data/raw/fires/india_fires_raw.parquet      (all fire detections, India)
    data/raw/fires/city_fires_daily.parquet     (city x date aggregated)
    reports/metrics/download_fires_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.config import load_env, load_params
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

FIRMS_AREA_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
INDIA_BBOX = "68,6,98,37"

SENSOR_SP  = "VIIRS_SNPP_SP"
SENSOR_NRT = "VIIRS_SNPP_NRT"
SP_END_DATE = "2026-07-31"

BATCH_DAYS = 5
MAX_WORKERS = 8


def get_firms_key() -> str:
    load_env()
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "FIRMS_MAP_KEY not set. Get a free key at:\n"
            "  https://firms.modaps.eosdis.nasa.gov/api/map_key/\n"
            "Then add it to your .env file: FIRMS_MAP_KEY=your_key_here"
        )
    return key


def get_sensor_for_date(date_str: str) -> str:
    if date_str > SP_END_DATE:
        return SENSOR_NRT
    return SENSOR_SP


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points (scalar)."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_km_vec(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Fast vectorized great-circle distance in km using NumPy."""
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lats)
    dphi = np.radians(lats - lat1)
    dlam = np.radians(lons - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def fetch_firms_batch(
    api_key: str,
    start_date: date,
    n_days: int,
    cache_dir: Path | None = None,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch FIRMS VIIRS data for a batch of n_days starting from start_date.
    Checks cache_dir first; caches downloaded batch as parquet.
    """
    date_str = start_date.strftime("%Y-%m-%d")

    if cache_dir is not None:
        cache_file = cache_dir / f"{date_str}.parquet"
        if cache_file.exists():
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass

    sensor = get_sensor_for_date(date_str)
    url = f"{FIRMS_AREA_BASE}/{api_key}/{sensor}/{INDIA_BBOX}/{n_days}/{date_str}"

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code == 200:
                text = resp.text.strip()
                if not text or "latitude" not in text:
                    df = pd.DataFrame()
                else:
                    df = pd.read_csv(StringIO(text), dtype={"version": str})
                if cache_dir is not None and not df.empty:
                    df.to_parquet(cache_dir / f"{date_str}.parquet", index=False)
                return df
            logger.warning(
                f"FIRMS batch {date_str}+{n_days}d attempt {attempt}/{retries}: "
                f"HTTP {resp.status_code} -- {resp.text[:200]}"
            )
        except Exception as exc:
            logger.warning(f"FIRMS batch {date_str}+{n_days}d attempt {attempt}: {exc}")
        time.sleep(1.5 * attempt)

    return pd.DataFrame()


def download_all_fires(
    api_key: str,
    start: str,
    end: str,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Download all FIRMS VIIRS fire detections for India in batches using concurrency."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()

    batches: list[tuple[date, int]] = []
    current = start_date
    while current <= end_date:
        remaining = (end_date - current).days + 1
        batch_n = min(BATCH_DAYS, remaining)
        batches.append((current, batch_n))
        current += timedelta(days=batch_n)

    total_batches = len(batches)
    logger.info(f"Downloading fires in {total_batches} batches ({start} -> {end}) with {MAX_WORKERS} workers...")

    frames: list[pd.DataFrame] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(fetch_firms_batch, api_key, b_date, b_n, cache_dir): b_date
            for b_date, b_n in batches
        }
        for future in as_completed(future_map):
            completed += 1
            if completed % 50 == 0 or completed == total_batches:
                logger.info(f"  Fires progress: {completed}/{total_batches} batches completed")
            try:
                b_df = future.result()
                if not b_df.empty:
                    frames.append(b_df)
            except Exception as exc:
                logger.warning(f"Batch failed with error: {exc}")

    if not frames:
        raise RuntimeError("No fire data was downloaded from FIRMS. Check FIRMS_MAP_KEY.")

    df_all = pd.concat(frames, ignore_index=True)
    logger.info(f"Total raw FIRMS records: {len(df_all):,}")
    return df_all


def normalise_firms(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise FIRMS column names and types."""
    df = df.copy()
    df["date"] = pd.to_datetime(df.get("acq_date", df.get("date", "")), errors="coerce")
    df["latitude"] = pd.to_numeric(df.get("latitude", float("nan")), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude", float("nan")), errors="coerce")
    df["frp"] = pd.to_numeric(df.get("frp", 0.0), errors="coerce").fillna(0.0)

    # Cast version to string to avoid mixed int/str type errors in pyarrow
    if "version" in df.columns:
        df["version"] = df["version"].astype(str)

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
    For each city x date, count fires and sum FRP within radius_km.
    Uses vectorized NumPy distance calculation for high performance.
    """
    records: list[pd.DataFrame] = []

    fire_lats = df_fires["latitude"].to_numpy(dtype=np.float64)
    fire_lons = df_fires["longitude"].to_numpy(dtype=np.float64)

    for city in cities:
        clat, clon = CITY_COORDS[city]
        logger.info(f"Computing fire proximity for {city} (vectorized) ...")

        # Vectorized great-circle distance
        distances = haversine_km_vec(clat, clon, fire_lats, fire_lons)
        within_mask = distances <= radius_km

        if not np.any(within_mask):
            logger.warning(f"[{city}] No fires within {radius_km} km in dataset")
            continue

        df_city_within = df_fires.iloc[within_mask].copy()
        df_city_within["distance_km"] = distances[within_mask]
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
    setup_logging(stage="download_fires")
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
    cache_dir = args.output_dir / "cache"

    # Download raw fire detections with caching and 8 workers
    df_raw = download_all_fires(api_key, args.start, args.end, cache_dir=cache_dir)
    df_raw = normalise_firms(df_raw)

    raw_path = args.output_dir / "india_fires_raw.parquet"
    df_raw.to_parquet(raw_path, index=False)
    logger.info(f"Saved raw fire data -> {raw_path} ({len(df_raw):,} records)")

    # Fast vectorized aggregation to city x day
    df_city = aggregate_fires_to_city_day(df_raw, cities, radius_km=args.radius_km)
    city_path = args.output_dir / "city_fires_daily.parquet"
    df_city.to_parquet(city_path, index=False)
    logger.info(f"Saved city-day fire features -> {city_path} ({len(df_city):,} rows)")

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
