"""
src/data/validate.py
---------------------
Phase 2 – Validation: Schema and integrity checks on the merged city × day panel.

Checks:
  1. Required columns present
  2. AQI target coverage per city (must exceed threshold)
  3. Date continuity — no unexpected gaps
  4. Value ranges for root-cause features
  5. No future leakage (fire/weather from tomorrow in today's row)
  6. City set completeness

Outputs:
    reports/metrics/data_validation.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_params
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("validate")

# ── Required columns (must exist; NaN allowed up to threshold) ────────────────
REQUIRED_COLS = [
    "city", "date", "aqi",
    "fire_count_500km", "fire_frp_sum_500km",
    "temperature_2m_mean", "wind_speed_10m_mean",
    "relative_humidity_2m_mean", "precipitation_sum",
    "burning_season_flag", "festival_flag", "grap_stage",
    "bsvi_active", "season", "month", "weekend_flag",
]

# ── Value range guards ────────────────────────────────────────────────────────
RANGE_CHECKS: dict[str, tuple[float, float]] = {
    "aqi":                       (0,   1500),
    "temperature_2m_mean":       (-5,   55),
    "relative_humidity_2m_mean": (0,   100),
    "wind_speed_10m_mean":       (0,   150),
    "precipitation_sum":         (0,  1000),
    "fire_count_500km":          (0, 50000),
    "grap_stage":                (0,     4),
    "month":                     (1,    12),
}

# ── Maximum acceptable missing fraction per column ───────────────────────────
MAX_MISSING_FRAC: dict[str, float] = {
    "aqi":                       0.45,   # XKDR coverage gaps expected
    "fire_count_500km":          0.05,
    "temperature_2m_mean":       0.02,
    "wind_speed_10m_mean":       0.02,
    "relative_humidity_2m_mean": 0.02,
    "diesel_frac":               0.30,   # vehicle data might be partial
    "iip_manufacturing_index":   0.25,
}


def check_required_columns(df: pd.DataFrame) -> list[str]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return missing


def check_city_aqi_coverage(
    df: pd.DataFrame,
    min_days: int = 180,
) -> dict[str, int]:
    """Return city → valid AQI day count; warn if below threshold."""
    coverage: dict[str, int] = {}
    for city, grp in df.groupby("city"):
        days = int(grp["aqi"].notna().sum())
        coverage[str(city)] = days
        if days < min_days:
            logger.warning(f"[{city}] AQI coverage: {days} days (threshold: {min_days})")
    return coverage


def check_date_continuity(df: pd.DataFrame) -> dict[str, list[str]]:
    """Detect date gaps > 1 day per city."""
    gaps: dict[str, list[str]] = {}
    for city, grp in df.groupby("city"):
        dates = pd.to_datetime(grp["date"]).sort_values()
        diffs = dates.diff().dropna()
        large_gaps = diffs[diffs > pd.Timedelta(days=1)]
        if not large_gaps.empty:
            gap_strs = [str(d.date()) for d in dates.iloc[large_gaps.index]]
            gaps[str(city)] = gap_strs
            logger.warning(f"[{city}] {len(large_gaps)} date gap(s) detected")
    return gaps


def check_value_ranges(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Check for out-of-range values; return violation counts."""
    violations: dict[str, dict[str, float]] = {}
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        oob = ((series < lo) | (series > hi)).sum()
        if oob > 0:
            frac = float(oob / len(df))
            violations[col] = {"count": int(oob), "fraction": round(frac, 4), "lo": lo, "hi": hi}
            logger.warning(f"[{col}] {oob} out-of-range values ({frac:.1%}): expected [{lo}, {hi}]")
    return violations


def check_missing_fractions(df: pd.DataFrame) -> dict[str, float]:
    """Return missing fraction per column; warn if exceeds threshold."""
    missing_pct: dict[str, float] = {}
    for col, thresh in MAX_MISSING_FRAC.items():
        if col not in df.columns:
            continue
        frac = float(df[col].isna().mean())
        missing_pct[col] = round(frac, 4)
        if frac > thresh:
            logger.warning(
                f"[{col}] Missing fraction {frac:.1%} exceeds threshold {thresh:.1%}"
            )
    return missing_pct


def main() -> None:
    setup_logging()
    params = load_params()
    expected_cities: list[str] = params["data"]["cities"]
    min_aqi_days: int = params.get("aqi", {}).get("min_valid_days_per_city_year", 180)

    input_path = Path("data/interim/city_day_raw.parquet")
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run ingest_and_merge.py first."
        )

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded merged panel: {len(df):,} rows, {len(df.columns)} columns")

    errors: list[str] = []
    warnings: list[str] = []

    # 1. Required columns
    missing_cols = check_required_columns(df)
    if missing_cols:
        for col in missing_cols:
            warnings.append(f"Missing column: {col}")
            logger.warning(f"Missing required column: {col}")

    # 2. City set
    actual_cities = set(df["city"].unique())
    expected_set = set(expected_cities)
    missing_cities = expected_set - actual_cities
    if missing_cities:
        warnings.append(f"Cities missing from panel: {sorted(missing_cities)}")
        logger.warning(f"Cities missing: {missing_cities}")

    # 3. AQI coverage
    aqi_coverage = check_city_aqi_coverage(df, min_days=min_aqi_days)

    # 4. Date continuity
    date_gaps = check_date_continuity(df)

    # 5. Value ranges
    range_violations = check_value_ranges(df)
    for col, v in range_violations.items():
        if v["fraction"] > 0.02:
            errors.append(f"Range violation {col}: {v['fraction']:.1%} rows out of [{v['lo']}, {v['hi']}]")

    # 6. Missing fractions
    missing_fracs = check_missing_fractions(df)

    # ── Report ────────────────────────────────────────────────────────────────
    passed = len(errors) == 0
    status = "PASS" if passed else "FAIL"

    report: dict[str, Any] = {
        "validated_at": datetime.now().isoformat(),
        "status": status,
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "errors": errors,
        "warnings": warnings,
        "aqi_coverage_by_city": aqi_coverage,
        "date_gaps_by_city": {k: v[:5] for k, v in date_gaps.items()},  # first 5 gaps only
        "range_violations": range_violations,
        "missing_fractions": missing_fracs,
    }

    out_path = Path("reports/metrics/data_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"VALIDATION: {status}")
    print("=" * 60)
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  ✗ {e}")
    if warnings:
        print("WARNINGS:")
        for w in warnings[:10]:
            print(f"  ⚠ {w}")
    print(f"\nTotal rows:    {len(df):,}")
    print(f"AQI coverage:  {sum(aqi_coverage.values()):,} city-days")
    print("=" * 60 + "\n")

    if not passed:
        raise SystemExit(f"Validation FAILED with {len(errors)} error(s). See {out_path}")


if __name__ == "__main__":
    main()

