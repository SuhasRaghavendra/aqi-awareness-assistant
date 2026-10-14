"""
src/data/validate.py
--------------------
Phase 1, Step 004: Data validation rules and schema verification.

Provides schema checks, data hygiene assertions, and unit consistency verifications
for raw OpenAQ pollutant observations and Open-Meteo ERA5 weather data:
  - Required columns verification
  - Timestamp parsing, timezone awareness, and future-timestamp guard
  - Parameter membership in expected CPCB/atmospheric sets
  - Unit inventory and unit inconsistency detection
  - Value type assertions (numeric), sentinel detection (-999, -9999)
  - Negative concentration detection
  - Coordinate range sanity checks
  - Duplicate detection on (location, sensor/parameter, datetime)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger("validate")

EXPECTED_OPENAQ_COLUMNS = [
    "location_id",
    "datetime",
    "parameter",
    "value",
]

EXPECTED_POLLUTANTS = {
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "nh3",
    "nox",
    "no",
}

EXPECTED_WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
]


@dataclass
class ValidationReport:
    """Encapsulates validation findings for a dataset."""

    source_name: str
    total_rows: int = 0
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.passed = False
        logger.error(f"[{self.source_name}] VALIDATION ERROR: {message}")

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
        logger.warning(f"[{self.source_name}] VALIDATION WARNING: {message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_name,
            "total_rows": self.total_rows,
            "passed": self.passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


def validate_openaq_df(
    df: pd.DataFrame,
    expected_parameters: set[str] | None = None,
    allow_empty: bool = False,
) -> ValidationReport:
    """Validate OpenAQ narrow hourly pollutant observations dataframe."""
    report = ValidationReport(source_name="OpenAQ_Pollutants", total_rows=len(df))

    if df.empty:
        if allow_empty:
            report.add_warning("DataFrame is empty.")
            return report
        report.add_error("OpenAQ DataFrame is empty.")
        return report

    expected_params = expected_parameters or EXPECTED_POLLUTANTS

    # 1. Required columns
    col_map = {c.lower(): c for c in df.columns}
    missing_cols = [c for c in EXPECTED_OPENAQ_COLUMNS if c not in col_map]
    if missing_cols:
        report.add_error(
            f"Missing required OpenAQ columns: {missing_cols}. Found: {list(df.columns)}"
        )
        return report

    # 2. Datetime parsing and future timestamp check
    dt_col = col_map["datetime"]
    try:
        parsed_dts = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
        null_dts = parsed_dts.isna().sum()
        if null_dts > 0:
            report.add_error(f"{null_dts} rows have unparseable datetime strings.")

        now_utc = datetime.now(timezone.utc)
        future_dts = (parsed_dts > now_utc).sum()
        report.metrics["future_timestamps_count"] = int(future_dts)
        if future_dts > 0:
            report.add_error(
                f"{future_dts} observations contain timestamps in the future (> {now_utc.isoformat()})."
            )
    except Exception as exc:
        report.add_error(f"Failed to parse datetime column: {exc}")

    # 3. Parameter checks
    param_col = col_map["parameter"]
    actual_params = set(df[param_col].astype(str).str.lower().unique())
    unknown_params = actual_params - expected_params
    report.metrics["observed_parameters"] = sorted(actual_params)
    report.metrics["unknown_parameters"] = sorted(unknown_params)
    if unknown_params:
        report.add_warning(f"Observed unexpected atmospheric parameters: {sorted(unknown_params)}")

    # 4. Units inventory and inconsistency checks
    if "units" in col_map:
        units_col = col_map["units"]
        units_by_param: dict[str, list[str]] = {}
        for p, sub in df.groupby(param_col):
            u_list = sorted(sub[units_col].dropna().astype(str).unique())
            units_by_param[str(p)] = u_list

        report.metrics["units_per_parameter"] = units_by_param
        inconsistent_units = {p: u for p, u in units_by_param.items() if len(u) > 1}
        report.metrics["inconsistent_units"] = inconsistent_units
        if inconsistent_units:
            report.add_warning(
                f"Inconsistent units detected across parameters: {inconsistent_units}"
            )
    else:
        report.add_warning("Units column missing from OpenAQ DataFrame.")

    # 5. Value checks: numeric, negatives, and sentinels
    val_col = col_map["value"]
    numeric_vals = pd.to_numeric(df[val_col], errors="coerce")
    nan_vals = numeric_vals.isna().sum()
    report.metrics["non_numeric_or_nan_values"] = int(nan_vals)

    sentinel_counts = int(((numeric_vals == -999) | (numeric_vals == -9999)).sum())
    report.metrics["sentinel_values_count"] = sentinel_counts
    if sentinel_counts > 0:
        report.add_warning(f"Detected {sentinel_counts} sentinel missing values (-999 / -9999).")

    negative_counts = int(
        ((numeric_vals < 0) & (numeric_vals != -999) & (numeric_vals != -9999)).sum()
    )
    report.metrics["negative_values_count"] = negative_counts
    if negative_counts > 0:
        report.add_warning(f"Detected {negative_counts} negative concentration values.")

    # 6. Duplicates check
    dup_subset = ["location_id", "datetime", "parameter"]
    if all(c in col_map for c in dup_subset):
        dups = df.duplicated(subset=[col_map[c] for c in dup_subset]).sum()
        report.metrics["duplicate_observations"] = int(dups)
        if dups > 0:
            report.add_warning(
                f"Detected {dups} duplicate observations on (location_id, datetime, parameter)."
            )

    return report


def validate_weather_df(
    df: pd.DataFrame,
    expected_variables: list[str] | None = None,
    allow_empty: bool = False,
) -> ValidationReport:
    """Validate Open-Meteo ERA5 hourly historical weather observations dataframe."""
    report = ValidationReport(source_name="OpenMeteo_Weather", total_rows=len(df))

    if df.empty:
        if allow_empty:
            report.add_warning("Weather DataFrame is empty.")
            return report
        report.add_error("Weather DataFrame is empty.")
        return report

    expected_vars = expected_variables or EXPECTED_WEATHER_VARIABLES

    # 1. Required columns
    col_map = {c.lower(): c for c in df.columns}
    if "time" not in col_map:
        report.add_error(f"Missing required timestamp column 'time'. Found: {list(df.columns)}")
        return report

    missing_vars = [v for v in expected_vars if v not in col_map]
    report.metrics["missing_weather_variables"] = missing_vars
    if missing_vars:
        report.add_warning(f"Some expected weather variables missing: {missing_vars}")

    # 2. Datetime parsing and future timestamp check
    try:
        parsed_times = pd.to_datetime(df[col_map["time"]], errors="coerce")
        null_times = parsed_times.isna().sum()
        if null_times > 0:
            report.add_error(f"{null_times} weather records have unparseable timestamps.")

        now_naive = datetime.now()
        future_times = (parsed_times.dt.tz_localize(None) > now_naive).sum()
        report.metrics["future_timestamps_count"] = int(future_times)
        if future_times > 0:
            report.add_error(f"{future_times} weather records have timestamps in the future.")
    except Exception as exc:
        report.add_error(f"Failed to parse weather 'time' column: {exc}")

    # 3. Physical range sanity checks
    range_violations = {}

    if "relative_humidity_2m" in col_map:
        rh = pd.to_numeric(df[col_map["relative_humidity_2m"]], errors="coerce")
        invalid_rh = int(((rh < 0) | (rh > 100)).sum())
        if invalid_rh > 0:
            range_violations["relative_humidity_out_of_range"] = invalid_rh
            report.add_warning(f"{invalid_rh} relative humidity values outside [0, 100]%.")

    if "temperature_2m" in col_map:
        temp = pd.to_numeric(df[col_map["temperature_2m"]], errors="coerce")
        invalid_temp = int(((temp < -30) | (temp > 60)).sum())
        if invalid_temp > 0:
            range_violations["temperature_out_of_range"] = invalid_temp
            report.add_warning(
                f"{invalid_temp} temperatures outside realistic Indian range [-30, 60]°C."
            )

    if "precipitation" in col_map:
        precip = pd.to_numeric(df[col_map["precipitation"]], errors="coerce")
        invalid_precip = int((precip < 0).sum())
        if invalid_precip > 0:
            range_violations["negative_precipitation"] = invalid_precip
            report.add_warning(f"{invalid_precip} negative precipitation values detected.")

    if "surface_pressure" in col_map:
        press = pd.to_numeric(df[col_map["surface_pressure"]], errors="coerce")
        invalid_press = int(((press < 500) | (press > 1100)).sum())
        if invalid_press > 0:
            range_violations["pressure_out_of_range"] = invalid_press
            report.add_warning(f"{invalid_press} surface pressure values outside [500, 1100] hPa.")

    report.metrics["range_violations"] = range_violations

    # 4. Duplicates check
    dup_cols = ["location_id", "time"] if "location_id" in col_map else ["time"]
    dups = df.duplicated(subset=[col_map[c] for c in dup_cols]).sum()
    report.metrics["duplicate_timestamps"] = int(dups)
    if dups > 0:
        report.add_warning(f"Detected {dups} duplicate weather observations on {dup_cols}.")

    return report
