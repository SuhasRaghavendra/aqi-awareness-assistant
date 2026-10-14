"""
src/data/download_openaq.py
---------------------------
Phase 1, Step 003: Download OpenAQ hourly records from AWS S3.

Connects to the OpenAQ archive on AWS S3 anonymously without credentials,
syncs hourly narrow CSV.GZ partitions for selected stations across the study window:
  s3://openaq-data-archive/records/csv.gz/locationid=<id>/year=<yyyy>/month=<mm>/

Key Features:
  - Fully anonymous (no AWS account required, uses unsigned config)
  - Resumable: checks local file existence and size, skipping re-downloads
  - Parallel downloads using bounded ThreadPoolExecutor
  - Checksum validation: writes SHA256 checksums to download manifest
  - Logs counts of missing partitions
  - Offline --mock mode for testing and sandbox execution
  - Manifest generation: data/raw/openaq_manifest.csv

Usage:
  python -m src.data.download_openaq --help
  python -m src.data.download_openaq --mock
  python -m src.data.download_openaq --limit-stations 2 --limit-months 1
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.client import Config

from src.config import load_params
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.stage_stats import track_stage

logger = get_logger("download_openaq")

OPENAQ_BUCKET = "openaq-data-archive"
S3_REGION = "us-east-1"
BASE_PREFIX = "records/csv.gz"


def get_anonymous_s3_client() -> Any:
    """Return an anonymous S3 client configured for openaq-data-archive."""
    return boto3.client(
        "s3",
        region_name=S3_REGION,
        config=Config(signature_version=UNSIGNED),
    )


def compute_sha256(filepath: Path) -> str:
    """Calculate the SHA256 hash of a file."""
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_year_month_tuples(start_date_str: str, end_date_str: str) -> list[tuple[int, int]]:
    """Generate (year, month) integer tuples from start_date to end_date inclusive."""
    start_dt = datetime.fromisoformat(start_date_str)
    end_dt = datetime.fromisoformat(end_date_str)

    months = []
    curr_year = start_dt.year
    curr_month = start_dt.month

    while (curr_year < end_dt.year) or (curr_year == end_dt.year and curr_month <= end_dt.month):
        months.append((curr_year, curr_month))
        curr_month += 1
        if curr_month > 12:
            curr_month = 1
            curr_year += 1

    return months


def download_single_file(
    s3_client: Any,
    bucket: str,
    key: str,
    s3_size: int,
    local_path: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Download an S3 object to a local path if not already present with matching size."""
    now_str = datetime.now(timezone.utc).isoformat()
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if local file exists and matches size (resumable cache check)
    if local_path.exists() and local_path.stat().st_size == s3_size:
        sha256_hash = compute_sha256(local_path)
        return {
            "s3_key": key,
            "local_path": str(local_path),
            "size_bytes": s3_size,
            "sha256": sha256_hash,
            "download_time": now_str,
            "status": "cached",
        }

    if dry_run:
        return {
            "s3_key": key,
            "local_path": str(local_path),
            "size_bytes": s3_size,
            "sha256": "(dry-run)",
            "download_time": now_str,
            "status": "dry-run",
        }

    # Download from S3
    try:
        s3_client.download_file(bucket, key, str(local_path))
        sha256_hash = compute_sha256(local_path)
        return {
            "s3_key": key,
            "local_path": str(local_path),
            "size_bytes": s3_size,
            "sha256": sha256_hash,
            "download_time": now_str,
            "status": "downloaded",
        }
    except Exception as exc:
        logger.error(f"Failed to download {key}: {exc}")
        return {
            "s3_key": key,
            "local_path": str(local_path),
            "size_bytes": s3_size,
            "sha256": "",
            "download_time": now_str,
            "status": f"failed: {exc}",
        }


def process_station_month(
    s3_client: Any,
    location_id: int,
    year: int,
    month: int,
    output_dir: Path,
    dry_run: bool = False,
) -> tuple[int, int, list[dict[str, Any]], bool]:
    """List and download all CSV.GZ files for a station month partition."""
    prefix = f"{BASE_PREFIX}/locationid={location_id}/year={year}/month={month:02d}/"
    paginator = s3_client.get_paginator("list_objects_v2")
    records: list[dict[str, Any]] = []

    has_files = False
    try:
        for page in paginator.paginate(Bucket=OPENAQ_BUCKET, Prefix=prefix):
            contents = page.get("Contents", [])
            for obj in contents:
                key = obj["Key"]
                size = obj["Size"]
                if not key.endswith(".csv.gz"):
                    continue
                has_files = True
                filename = Path(key).name
                local_file = (
                    output_dir
                    / f"locationid={location_id}"
                    / f"year={year}"
                    / f"month={month:02d}"
                    / filename
                )
                rec = download_single_file(
                    s3_client, OPENAQ_BUCKET, key, size, local_file, dry_run=dry_run
                )
                rec["location_id"] = location_id
                rec["year"] = year
                rec["month"] = month
                records.append(rec)
    except Exception as exc:
        logger.warning(f"Error listing prefix {prefix}: {exc}")
        return (location_id, year, [], True)

    is_missing = not has_files
    return (location_id, year, records, is_missing)


def generate_mock_openaq_data(
    stations_df: pd.DataFrame,
    output_dir: Path,
    window_start: str,
    snapshot_end: str,
    limit_months: int = 2,
) -> list[dict[str, Any]]:
    """Generate sample mock openaq narrow gzipped CSV files for testing."""
    logger.info("Generating realistic mock OpenAQ narrow data for testing...")
    records: list[dict[str, Any]] = []
    ym_tuples = generate_year_month_tuples(window_start, snapshot_end)[:limit_months]

    for _, station in stations_df.iterrows():
        loc_id = int(station["location_id"])
        loc_name = station["name"]
        lat = station["latitude"]
        lon = station["longitude"]
        params_str = str(station.get("parameters", "pm25,pm10,no2,so2,co,o3"))
        params_list = [p.strip() for p in params_str.split(",") if p.strip()]

        for year, month in ym_tuples:
            target_dir = output_dir / f"locationid={loc_id}" / f"year={year}" / f"month={month:02d}"
            target_dir.mkdir(parents=True, exist_ok=True)

            # Generate sample daily files for the 1st and 15th of the month
            for day in [1, 15]:
                date_str = f"{year}{month:02d}{day:02d}"
                filename = f"location-{loc_id}-{date_str}.csv.gz"
                local_file = target_dir / filename

                # Generate hourly narrow CSV data
                # Header: location_id,sensors_id,location,datetime,lat,lon,parameter,units,value
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow(
                    [
                        "location_id",
                        "sensors_id",
                        "location",
                        "datetime",
                        "lat",
                        "lon",
                        "parameter",
                        "units",
                        "value",
                    ]
                )

                for hour in range(24):
                    dt_iso = f"{year}-{month:02d}-{day:02d}T{hour:02d}:00:00+00:00"
                    for s_idx, p in enumerate(params_list, start=1):
                        unit = "mg/m3" if p == "co" else "ug/m3"
                        # Realistic baseline synthetic concentrations
                        base_val = 45.0 if p == "pm25" else (85.0 if p == "pm10" else 25.0)
                        val = round(base_val + (hour % 5) * 2.5, 2)
                        writer.writerow([loc_id, s_idx, loc_name, dt_iso, lat, lon, p, unit, val])

                # Compress and write
                compressed_bytes = gzip.compress(buf.getvalue().encode("utf-8"))
                local_file.write_bytes(compressed_bytes)

                sha = hashlib.sha256(compressed_bytes).hexdigest()
                now_str = datetime.now(timezone.utc).isoformat()
                records.append(
                    {
                        "location_id": loc_id,
                        "year": year,
                        "month": month,
                        "s3_key": f"{BASE_PREFIX}/locationid={loc_id}/year={year}/month={month:02d}/{filename}",
                        "local_path": str(local_file),
                        "size_bytes": len(compressed_bytes),
                        "sha256": sha,
                        "download_time": now_str,
                        "status": "mock_generated",
                    }
                )

    logger.info(f"Generated {len(records)} mock OpenAQ files across {len(stations_df)} stations.")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: Download OpenAQ hourly records from AWS S3"
    )
    parser.add_argument(
        "--stations-file",
        type=Path,
        default=Path("configs/stations_selected.csv"),
        help="Path to selected stations CSV",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw/openaq"), help="Target download directory"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw/openaq_manifest.csv"),
        help="Path to download manifest CSV",
    )
    parser.add_argument("--max-workers", type=int, default=8, help="Number of worker threads")
    parser.add_argument(
        "--limit-stations", type=int, default=None, help="Limit number of stations to process"
    )
    parser.add_argument(
        "--limit-months", type=int, default=None, help="Limit number of months per station"
    )
    parser.add_argument("--dry-run", action="store_true", help="List files without downloading")
    parser.add_argument(
        "--mock", action="store_true", help="Generate mock OpenAQ narrow data for testing"
    )
    args = parser.parse_args()

    setup_logging(stage="download_openaq")
    logger.info("Starting Phase 1 Step 003: OpenAQ S3 Data Download...")

    params = load_params()
    data_cfg = params.get("data", {})
    window_start = data_cfg.get("window_start", "2019-01-01")
    snapshot_end = data_cfg.get("snapshot_end", "2026-09-30")

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

    logger.info(f"Loaded {len(stations_df)} selected stations from {args.stations_file}")

    ym_tuples = generate_year_month_tuples(window_start, snapshot_end)
    if args.limit_months:
        ym_tuples = ym_tuples[: args.limit_months]
        logger.info(f"Limiting to first {args.limit_months} months.")

    logger.info(
        f"Window: {window_start} -> {snapshot_end} ({len(ym_tuples)} months to check per station)"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    all_manifest_records: list[dict[str, Any]] = []
    missing_partitions_count = 0

    if args.mock:
        all_manifest_records = generate_mock_openaq_data(
            stations_df=stations_df,
            output_dir=args.output_dir,
            window_start=window_start,
            snapshot_end=snapshot_end,
            limit_months=args.limit_months or 2,
        )
    else:
        s3_client = get_anonymous_s3_client()
        tasks = []
        for loc_id in stations_df["location_id"]:
            for year, month in ym_tuples:
                tasks.append((int(loc_id), year, month))

        logger.info(
            f"Checking {len(tasks)} station-month partitions with {args.max_workers} worker threads..."
        )

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    process_station_month,
                    s3_client,
                    loc_id,
                    year,
                    month,
                    args.output_dir,
                    args.dry_run,
                ): (loc_id, year, month)
                for loc_id, year, month in tasks
            }

            for future in as_completed(future_to_task):
                loc_id, year, month = future_to_task[future]
                try:
                    _, _, records, is_missing = future.result()
                    if is_missing:
                        missing_partitions_count += 1
                    else:
                        all_manifest_records.extend(records)
                except Exception as exc:
                    logger.warning(
                        f"Failed processing partition for station {loc_id} ({year}-{month:02d}): {exc}"
                    )

    manifest_df = pd.DataFrame(all_manifest_records)
    if manifest_df.empty:
        manifest_df = pd.DataFrame(
            columns=[
                "location_id",
                "year",
                "month",
                "s3_key",
                "local_path",
                "size_bytes",
                "sha256",
                "download_time",
                "status",
            ]
        )

    # Track stage stats
    with track_stage(
        "download_openaq", inputs=[str(args.stations_file)], outputs=[str(args.manifest)]
    ) as st:
        st.record_input_frame(stations_df)
        st.record_output_frame(manifest_df)
        if missing_partitions_count > 0:
            st.drop("Missing S3 Partitions (no files reported for month)", missing_partitions_count)

        manifest_df.to_csv(args.manifest, index=False)
        logger.info(f"Saved download manifest with {len(manifest_df)} entries to {args.manifest}")

    total_size_mb = (
        manifest_df["size_bytes"].sum() / (1024 * 1024) if not manifest_df.empty else 0.0
    )
    status_counts = manifest_df["status"].value_counts().to_dict() if not manifest_df.empty else {}

    print("\n" + "=" * 60)
    print("OPENAQ DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Stations Processed:       {len(stations_df)}")
    print(f"Total Partitions Checked: {len(stations_df) * len(ym_tuples)}")
    print(f"Files in Manifest:        {len(manifest_df)}")
    print(f"Total Data Volume:        {total_size_mb:.2f} MB")
    print(f"Missing Partitions:       {missing_partitions_count}")
    print(f"Status Breakdown:         {status_counts}")
    print(f"Manifest Path:            {args.manifest}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
