"""
src/data/download_weather.py
-----------------------------
Phase 1 – Step C: Download daily meteorological data from Open-Meteo ERA5
archive for all 9 target cities.

No API key required. Open-Meteo is completely free for non-commercial use
(10,000 requests/day). Uses ECMWF ERA5 reanalysis: 1940–present, ~0.25° grid.

Variables downloaded (daily aggregates):
  temperature_2m_mean / max / min    (°C)
  relative_humidity_2m_mean          (%)
  wind_speed_10m_mean                (km/h)
  wind_direction_10m_dominant        (degrees)
  precipitation_sum                  (mm)
  surface_pressure_mean              (hPa)
  vapour_pressure_deficit_mean       (kPa)

Usage:
    python -m src.data.download_weather [--start 2019-01-01] [--end 2026-09-30]
    python -m src.data.download_weather --mock    # fast offline test

Outputs:
    data/raw/weather/city_weather_daily.parquet
    reports/metrics/download_weather_stats.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.config import load_params
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.stage_stats import track_stage

logger = get_logger("download_weather")

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_TIMEZONE = "Asia/Kolkata"

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

DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "wind_direction_10m_dominant",
    "precipitation_sum",
    "surface_pressure_mean",
]

OPEN_METEO_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "wind_direction_10m_dominant",
    "precipitation_sum",
    "pressure_msl_mean",
]


def fetch_city_weather(
    city: str,
    lat: float,
    lon: float,
    start: str,
    end: str,
    retries: int = 4,
    delay: float = 1.5,
) -> pd.DataFrame:
    """Fetch daily ERA5 weather for one city from Open-Meteo."""
    max_archive_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    eff_end = min(end, max_archive_date)
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": eff_end,
        "daily": OPEN_METEO_VARS,
        "timezone": DEFAULT_TIMEZONE,
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
            if resp.status_code == 429:
                wait = 30 * attempt
                logger.warning(f"[{city}] Rate-limited (429). Waiting {wait}s …")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            daily = data.get("daily", {})
            if not daily or "time" not in daily:
                raise ValueError("Open-Meteo response missing daily.time")
            df = pd.DataFrame(daily)
            df.rename(
                columns={"time": "date", "pressure_msl_mean": "surface_pressure_mean"}, inplace=True
            )
            df["city"] = city
            logger.info(f"[{city}] Downloaded {len(df):,} daily weather rows ({start}->{end})")
            return df
        except Exception as exc:
            logger.warning(f"[{city}] Weather attempt {attempt}/{retries}: {exc}")
            time.sleep(delay * attempt)

    logger.error(f"[{city}] All weather download retries exhausted")
    return pd.DataFrame()


def generate_mock_weather(
    city: str,
    lat: float,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Generate realistic mock daily weather for testing without network."""
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    rng = np.random.default_rng(abs(hash(city)) % (2**31))

    # Seasonal temperature variation
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    temp_base = 25.0 + (10.0 if lat > 20 else 3.0) * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    humidity_base = 60.0 + 25.0 * np.sin(2 * np.pi * (day_of_year - 180) / 365)

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "city": city,
            "temperature_2m_mean": (temp_base + rng.normal(0, 2, n)).round(1),
            "temperature_2m_max": (temp_base + 5 + rng.normal(0, 2, n)).round(1),
            "temperature_2m_min": (temp_base - 5 + rng.normal(0, 2, n)).round(1),
            "relative_humidity_2m_mean": np.clip(
                humidity_base + rng.normal(0, 10, n), 15, 97
            ).round(1),
            "wind_speed_10m_mean": np.clip(rng.exponential(8, n), 0.5, 60).round(1),
            "wind_direction_10m_dominant": rng.uniform(0, 360, n).round(0),
            "precipitation_sum": np.where(rng.random(n) > 0.85, rng.exponential(5, n), 0).round(1),
            "surface_pressure_mean": (1010.0 + rng.normal(0, 5, n)).round(1),
            "vapour_pressure_deficit_mean": np.clip(rng.exponential(0.8, n), 0, 4).round(2),
        }
    )
    return df


def add_derived_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Add secondary weather-derived features for pollution trapping."""
    if "wind_speed_10m_mean" in df.columns and "relative_humidity_2m_mean" in df.columns:
        wind = df["wind_speed_10m_mean"].fillna(5.0)
        hum = df["relative_humidity_2m_mean"].fillna(50.0)
        df["inversion_risk_flag"] = ((wind < 5.0) & (hum > 70.0)).astype(int)
        df["fog_risk_flag"] = ((hum > 85.0) & (wind < 3.0)).astype(int)

    if "precipitation_sum" in df.columns:
        df["rain_washout_flag"] = (df["precipitation_sum"].fillna(0) > 5.0).astype(int)

    return df


def main() -> None:
    setup_logging(stage="download_weather")
    parser = argparse.ArgumentParser(
        description="Download Open-Meteo ERA5 daily weather for 9 cities"
    )
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-09-30")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/weather"))
    parser.add_argument("--mock", action="store_true", help="Generate mock data (no network)")
    args = parser.parse_args()

    params = load_params()
    cities: list[str] = params["data"]["cities"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for city in cities:
        lat, lon = CITY_COORDS[city]
        logger.info(f"Downloading weather for {city} ({lat}, {lon}) …")
        if args.mock:
            df = generate_mock_weather(city, lat, args.start, args.end)
        else:
            df = fetch_city_weather(city, lat, lon, args.start, args.end)
            time.sleep(0.5)  # polite throttle

        if not df.empty:
            df = add_derived_weather(df)
            frames.append(df)

    if not frames:
        raise RuntimeError("No weather data downloaded. Check network access.")

    df_all = pd.concat(frames, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.strftime("%Y-%m-%d")
    df_all = df_all.sort_values(["city", "date"]).reset_index(drop=True)

    out_path = args.output_dir / "city_weather_daily.parquet"
    df_all.to_parquet(out_path, index=False)
    logger.info(f"Saved city weather -> {out_path} ({len(df_all):,} rows)")

    stats: dict[str, Any] = {
        "downloaded_at": datetime.now().isoformat(),
        "source": "mock" if args.mock else "Open-Meteo ERA5",
        "start": args.start,
        "end": args.end,
        "cities": cities,
        "total_rows": len(df_all),
        "columns": list(df_all.columns),
        "coverage_by_city": df_all.groupby("city")["date"].count().astype(int).to_dict(),
    }
    metrics_path = Path("reports/metrics/download_weather_stats.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(stats, f, indent=2)

    with track_stage("download_weather", inputs=[], outputs=[str(out_path)]) as st:
        st.record_output_frame(df_all)

    print("\n" + "=" * 60)
    print("WEATHER DOWNLOAD SUMMARY")
    print("=" * 60)
    for city, cnt in stats["coverage_by_city"].items():
        print(f"  {city:<12}: {cnt:>5} city-days")
    print(f"\nTotal rows: {len(df_all):,}  |  Columns: {len(df_all.columns)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
