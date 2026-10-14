"""
src/data/ingest.py
------------------
Phase 1, Step 004: Ingest and validate raw OpenAQ and weather observations.

Reads raw downloaded partitions from data/raw/openaq and data/raw/weather,
applies schema validation checks via src.data.validate,
produces reports/metrics/data_validation.json, and hard-fails on schema breakage.

Pipeline stage command for dvc.yaml:
  python -m src.data.ingest
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import load_params
from src.data.validate import validate_openaq_df, validate_weather_df
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.stage_stats import track_stage

logger = get_logger("ingest")


def load_openaq_partitions(raw_dir: Path, max_files: int | None = None) -> pd.DataFrame:
    """Read and concatenate OpenAQ raw narrow CSV.GZ partition files."""
    csv_files = list(raw_dir.glob("**/*.csv.gz"))
    if not csv_files:
        logger.warning(f"No .csv.gz files found under {raw_dir}")
        return pd.DataFrame()

    if max_files:
        csv_files = csv_files[:max_files]

    logger.info(f"Loading {len(csv_files)} OpenAQ partition files from {raw_dir}...")
    frames: list[pd.DataFrame] = []
    for fp in csv_files:
        try:
            df = pd.read_csv(fp, compression="gzip")
            frames.append(df)
        except Exception as exc:
            logger.warning(f"Failed to read {fp}: {exc}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(combined):,} total OpenAQ pollutant observations.")
    return combined


def load_weather_partitions(raw_dir: Path, max_files: int | None = None) -> pd.DataFrame:
    """Read and concatenate Open-Meteo ERA5 raw Parquet partition files."""
    parquet_files = list(raw_dir.glob("**/*.parquet"))
    if not parquet_files:
        logger.warning(f"No .parquet files found under {raw_dir}")
        return pd.DataFrame()

    if max_files:
        parquet_files = parquet_files[:max_files]

    logger.info(f"Loading {len(parquet_files)} Weather partition files from {raw_dir}...")
    frames: list[pd.DataFrame] = []
    for fp in parquet_files:
        try:
            df = pd.read_parquet(fp)
            frames.append(df)
        except Exception as exc:
            logger.warning(f"Failed to read {fp}: {exc}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(combined):,} total hourly weather records.")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Ingest and Validate Raw Data")
    parser.add_argument(
        "--openaq-dir", type=Path, default=Path("data/raw/openaq"), help="Raw OpenAQ directory"
    )
    parser.add_argument(
        "--weather-dir", type=Path, default=Path("data/raw/weather"), help="Raw weather directory"
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path("reports/metrics/data_validation.json"),
        help="Output validation JSON",
    )
    parser.add_argument(
        "--stations-file",
        type=Path,
        default=Path("configs/stations_selected.csv"),
        help="Stations selected CSV",
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Max files to sample for fast verification"
    )
    parser.add_argument(
        "--allow-empty", action="store_true", help="Do not hard-fail if data directories are empty"
    )
    args = parser.parse_args()

    setup_logging(stage="validate")
    logger.info("Starting Phase 1 Step 004: Raw Data Ingestion & Validation...")

    params = load_params()
    weather_vars = params.get("weather", {}).get("variables", [])

    openaq_df = load_openaq_partitions(args.openaq_dir, max_files=args.max_files)
    weather_df = load_weather_partitions(args.weather_dir, max_files=args.max_files)

    # Execute validations
    openaq_report = validate_openaq_df(openaq_df, allow_empty=args.allow_empty)
    weather_report = validate_weather_df(
        weather_df, expected_variables=weather_vars, allow_empty=args.allow_empty
    )

    overall_passed = openaq_report.passed and weather_report.passed

    validation_summary = {
        "validation_passed": overall_passed,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "openaq": openaq_report.to_dict(),
        "weather": weather_report.to_dict(),
    }

    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_metrics, "w", encoding="utf-8") as f:
        json.dump(validation_summary, f, indent=2)

    # Track stage stats
    input_paths = [
        str(p) for p in [args.openaq_dir, args.weather_dir, args.stations_file] if p.exists()
    ]
    with track_stage("validate", inputs=input_paths, outputs=[str(args.output_metrics)]) as st:
        if not openaq_df.empty:
            st.record_input_frame(openaq_df)
        if not weather_df.empty:
            st.record_output_frame(weather_df)

        # Record drops / issues detected
        neg_count = openaq_report.metrics.get("negative_values_count", 0)
        sentinel_count = openaq_report.metrics.get("sentinel_values_count", 0)
        dup_count = openaq_report.metrics.get("duplicate_observations", 0)
        if neg_count > 0:
            st.drop("Flagged negative pollutant values", neg_count)
        if sentinel_count > 0:
            st.drop("Flagged sentinel missing values (-999)", sentinel_count)
        if dup_count > 0:
            st.drop("Flagged duplicate pollutant observations", dup_count)

    print("\n" + "=" * 60)
    print("DATA VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Overall Status:            {'PASSED [OK]' if overall_passed else 'FAILED [ERR]'}")
    print(f"OpenAQ Rows Validated:     {openaq_report.total_rows:,}")
    print(f"  - Passed:                {openaq_report.passed}")
    print(f"  - Errors:                {len(openaq_report.errors)}")
    print(f"  - Warnings:              {len(openaq_report.warnings)}")
    print(f"  - Observed Pollutants:   {openaq_report.metrics.get('observed_parameters', [])}")
    print(f"Weather Rows Validated:    {weather_report.total_rows:,}")
    print(f"  - Passed:                {weather_report.passed}")
    print(f"  - Errors:                {len(weather_report.errors)}")
    print(f"  - Warnings:              {len(weather_report.warnings)}")
    print(f"Validation Report Saved:   {args.output_metrics}")
    print("=" * 60 + "\n")

    if not overall_passed and not args.allow_empty:
        logger.error("Data validation failed hard-fail threshold. See report for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
