"""
src/data/download_weather.py
----------------------------
Phase 1, Step 003: Download Open-Meteo ERA5 historical weather data.

For each selected air quality monitoring station coordinate, fetches hourly
weather reanalysis data from Open-Meteo in yearly chunks:
  Variables:
    - temperature_2m (°C)
    - relative_humidity_2m (%)
    - dew_point_2m (°C)
    - surface_pressure (hPa)
    - precipitation (mm)
    - wind_speed_10m (km/h or m/s)
    - wind_direction_10m (degrees)
    - boundary_layer_height (m)

Key Features:
  - Local disk caching per (station, year) in Parquet format
  - Throttled requests with exponential backoff on errors
  - Timezone standardized to Asia/Kolkata
  - Manifest generation: data/raw/weather_manifest.csv
  - Offline --mock mode for testing and sandbox execution

Usage:
  python -m src.data.download_weather --help
  python -m src.data.download_weather --mock
  python -m src.data.download_weather --limit-stations 2 --limit-years 1
"""

from __future__ import annotations

import argparse
import hashlib
import time
from datetime import datetime, timezone
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
DEFAULT_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "boundary_layer_height",
]


def compute_sha256(filepath: Path) -> str:
    """Calculate the SHA256 hash of a file."""
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_yearly_intervals(window_start: str, snapshot_end: str) -> list[tuple[int, str, str]]:
    """Split date range into (year, chunk_start, chunk_end) tuples."""
    start_dt = datetime.fromisoformat(window_start)
    end_dt = datetime.fromisoformat(snapshot_end)

    intervals = []
    for year in range(start_dt.year, end_dt.year + 1):
        chunk_start = f"{year}-01-01" if year > start_dt.year else window_start
        chunk_end = f"{year}-12-31" if year < end_dt.year else snapshot_end
        intervals.append((year, chunk_start, chunk_end))

    return intervals


def fetch_weather_chunk(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    variables: list[str],
    timezone_name: str = DEFAULT_TIMEZONE,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo for a specific coordinate and date range."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": timezone_name,
    }

    retries = max_retries
    while retries > 0:
        try:
            resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
            if resp.status_code == 429:
                logger.warning("Rate limit hit (429) from Open-Meteo. Throttling for 10s...")
                time.sleep(10)
                retries -= 1
                continue
            resp.raise_for_status()
            data = resp.json()
            hourly = data.get("hourly", {})
            if not hourly or "time" not in hourly:
                raise ValueError("Open-Meteo response missing 'hourly.time' data.")
            df = pd.DataFrame(hourly)
            return df
        except Exception as exc:
            retries -= 1
            logger.warning(
                f"Weather request failed ({start_date} to {end_date}, retries left={retries}): {exc}"
            )
            if retries == 0:
                raise
            time.sleep(2.0 * (max_retries - retries + 1))

    raise RuntimeError(
        f"Failed to fetch weather data for ({lat}, {lon}) {start_date} to {end_date}"
    )


def generate_mock_weather_chunk(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    variables: list[str],
) -> pd.DataFrame:
    """Generate synthetic hourly weather dataframe matching Open-Meteo ERA5 schema."""
    date_range = pd.date_range(start=f"{start_date} 00:00:00", end=f"{end_date} 23:00:00", freq="h")
    n_hours = len(date_range)

    # Base weather patterns
    hours = date_range.hour.values
    t_base = 25.0 + 5.0 * np.sin((hours - 9) * 2 * np.pi / 24) + np.random.normal(0, 1.0, n_hours)
    rh_base = np.clip(
        65.0 - 15.0 * np.sin((hours - 9) * 2 * np.pi / 24) + np.random.normal(0, 3.0, n_hours),
        20.0,
        98.0,
    )
    wind_spd = np.clip(
        10.0 + 3.0 * np.sin(hours * 2 * np.pi / 24) + np.random.normal(0, 1.5, n_hours), 0.5, 35.0
    )

    data: dict[str, Any] = {"time": date_range.strftime("%Y-%m-%dT%H:%M")}

    for v in variables:
        if "temperature" in v:
            data[v] = np.round(t_base, 2)
        elif "relative_humidity" in v:
            data[v] = np.round(rh_base, 1)
        elif "dew_point" in v:
            data[v] = np.round(t_base - ((100 - rh_base) / 5.0), 2)
        elif "surface_pressure" in v:
            data[v] = np.round(920.0 + np.random.normal(0, 1.5, n_hours), 1)
        elif "precipitation" in v:
            data[v] = np.round(
                np.where(
                    np.random.random(n_hours) > 0.92, np.random.exponential(2.0, n_hours), 0.0
                ),
                2,
            )
        elif "wind_speed" in v:
            data[v] = np.round(wind_spd, 2)
        elif "wind_direction" in v:
            data[v] = np.round((np.random.uniform(0, 360, n_hours)), 1)
        elif "boundary_layer_height" in v:
            data[v] = np.round(
                np.clip(800.0 + 600.0 * np.sin((hours - 6) * np.pi / 12), 150.0, 2200.0), 1
            )
        else:
            data[v] = np.round(np.random.normal(10, 2, n_hours), 2)

    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Download Open-Meteo ERA5 weather data")
    parser.add_argument(
        "--stations-file",
        type=Path,
        default=Path("configs/stations_selected.csv"),
        help="Selected stations CSV",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw/weather"), help="Weather output directory"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw/weather_manifest.csv"),
        help="Weather manifest CSV",
    )
    parser.add_argument("--limit-stations", type=int, default=None, help="Limit number of stations")
    parser.add_argument(
        "--limit-years", type=int, default=None, help="Limit number of years per station"
    )
    parser.add_argument(
        "--throttle-delay", type=float, default=0.5, help="Delay between API requests in seconds"
    )
    parser.add_argument(
        "--mock", action="store_true", help="Generate realistic mock weather data for testing"
    )
    args = parser.parse_args()

    setup_logging(stage="download_weather")
    logger.info("Starting Phase 1 Step 003: Open-Meteo Weather Download...")

    params = load_params()
    data_cfg = params.get("data", {})
    window_start = data_cfg.get("window_start", "2019-01-01")
    snapshot_end = data_cfg.get("snapshot_end", "2026-09-30")

    weather_cfg = params.get("weather", {})
    variables = weather_cfg.get("variables", DEFAULT_VARIABLES)
    # Ensure dew_point_2m and boundary_layer_height are included
    for var in ["dew_point_2m", "boundary_layer_height"]:
        if var not in variables:
            variables.append(var)

    if not args.stations_file.exists():
        logger.error(
            f"Stations file not found at {args.stations_file}. Run discover_stations first."
        )
        raise FileNotFoundError(f"{args.stations_file} does not exist.")

    stations_df = pd.read_csv(args.stations_file)
    if "selected" in stations_df.columns:
        stations_df = stations_df[stations_df["selected"]].copy()

    if args.limit_stations:
        stations_df = stations_df.head(args.limit_stations)
        logger.info(f"Limiting to first {args.limit_stations} stations.")

    yearly_chunks = generate_yearly_intervals(window_start, snapshot_end)
    if args.limit_years:
        yearly_chunks = yearly_chunks[: args.limit_years]
        logger.info(f"Limiting to first {args.limit_years} years.")

    logger.info(
        f"Loaded {len(stations_df)} stations. Processing {len(yearly_chunks)} year intervals per station."
    )
    logger.info(f"Variables to fetch: {variables}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, Any]] = []
    total_chunks = len(stations_df) * len(yearly_chunks)
    chunk_counter = 0

    for _, station in stations_df.iterrows():
        loc_id = int(station["location_id"])
        lat = float(station["latitude"])
        lon = float(station["longitude"])
        station_name = station["name"]

        station_dir = args.output_dir / f"locationid={loc_id}"
        station_dir.mkdir(parents=True, exist_ok=True)

        for year, chunk_start, chunk_end in yearly_chunks:
            chunk_counter += 1
            out_file = station_dir / f"weather_{loc_id}_{year}.parquet"
            now_iso = datetime.now(timezone.utc).isoformat()

            # Resumable cache check
            if out_file.exists() and out_file.stat().st_size > 1000:
                logger.info(
                    f"[{chunk_counter}/{total_chunks}] Station {loc_id} ({year}) already cached at {out_file.name}"
                )
                manifest_records.append(
                    {
                        "location_id": loc_id,
                        "year": year,
                        "start_date": chunk_start,
                        "end_date": chunk_end,
                        "local_path": str(out_file),
                        "rows_count": len(pd.read_parquet(out_file)),
                        "size_bytes": out_file.stat().st_size,
                        "sha256": compute_sha256(out_file),
                        "model_source": "Open-Meteo ERA5 Reanalysis",
                        "download_time": now_iso,
                        "status": "cached",
                    }
                )
                continue

            logger.info(
                f"[{chunk_counter}/{total_chunks}] Fetching weather for station {loc_id} ({station_name}) [{chunk_start} -> {chunk_end}]..."
            )

            if args.mock:
                weather_df = generate_mock_weather_chunk(
                    lat, lon, chunk_start, chunk_end, variables
                )
            else:
                weather_df = fetch_weather_chunk(lat, lon, chunk_start, chunk_end, variables)
                time.sleep(args.throttle_delay)

            weather_df["location_id"] = loc_id
            weather_df.to_parquet(out_file, index=False)
            file_size = out_file.stat().st_size
            sha256_hash = compute_sha256(out_file)

            manifest_records.append(
                {
                    "location_id": loc_id,
                    "year": year,
                    "start_date": chunk_start,
                    "end_date": chunk_end,
                    "local_path": str(out_file),
                    "rows_count": len(weather_df),
                    "size_bytes": file_size,
                    "sha256": sha256_hash,
                    "model_source": "Open-Meteo ERA5 Reanalysis",
                    "download_time": now_iso,
                    "status": "downloaded" if not args.mock else "mock_generated",
                }
            )

    manifest_df = pd.DataFrame(manifest_records)
    if manifest_df.empty:
        manifest_df = pd.DataFrame(
            columns=[
                "location_id",
                "year",
                "start_date",
                "end_date",
                "local_path",
                "rows_count",
                "size_bytes",
                "sha256",
                "model_source",
                "download_time",
                "status",
            ]
        )

    # Track stage stats
    with track_stage(
        "download_weather", inputs=[str(args.stations_file)], outputs=[str(args.manifest)]
    ) as st:
        st.record_input_frame(stations_df)
        st.record_output_frame(manifest_df)
        manifest_df.to_csv(args.manifest, index=False)
        logger.info(f"Saved weather manifest with {len(manifest_df)} entries to {args.manifest}")

    total_size_mb = (
        manifest_df["size_bytes"].sum() / (1024 * 1024) if not manifest_df.empty else 0.0
    )
    total_hours = manifest_df["rows_count"].sum() if not manifest_df.empty else 0
    status_counts = manifest_df["status"].value_counts().to_dict() if not manifest_df.empty else {}

    print("\n" + "=" * 60)
    print("WEATHER DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Stations Processed:       {len(stations_df)}")
    print(f"Weather Chunks Processed: {len(manifest_df)}")
    print(f"Total Hourly Rows:        {total_hours:,}")
    print(f"Total Weather Volume:     {total_size_mb:.2f} MB")
    print(f"Status Breakdown:         {status_counts}")
    print(f"Manifest Path:            {args.manifest}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
