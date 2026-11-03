"""
src/data/download_vehicles.py
------------------------------
Phase 1 – Step C: Download state-wise vehicle registration data from the
Open Government Data (OGD) platform India (data.gov.in).

Data: Annual state-level vehicle counts by fuel type (Petrol, Diesel, CNG,
Electric, Others) from MoRTH / Vahan data.

This gives us proxy features for city-level vehicle emission pressure:
  - total vehicles on road (per lakh population)
  - diesel vehicle fraction (high NOx, PM)
  - CNG vehicle fraction (lower emissions)
  - EV count (zero tailpipe emissions)
  - two-wheeler density (high HC, CO)

These are annual figures (finest available), later interpolated to monthly/daily.

Usage:
    python -m src.data.download_vehicles

Environment:
    DATAGOV_API_KEY  – free key from https://data.gov.in/ (user registration)

Fallback:
    If API key unavailable, script loads a bundled static reference CSV from
    configs/vehicle_state_reference.csv (manually compiled from MoRTH reports).

Outputs:
    data/raw/vehicles/state_vehicles_annual.parquet
    data/raw/vehicles/city_vehicles_annual.parquet   (state → city mapped)
    reports/metrics/download_vehicles_stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import load_params
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("download_vehicles")

# State → cities mapping (city is in this state)
CITY_STATE: dict[str, str] = {
    "Delhi":     "Delhi",
    "Mumbai":    "Maharashtra",
    "Bengaluru": "Karnataka",
    "Chennai":   "Tamil Nadu",
    "Kolkata":   "West Bengal",
    "Hyderabad": "Telangana",
    "Pune":      "Maharashtra",
    "Ahmedabad": "Gujarat",
    "Lucknow":   "Uttar Pradesh",
}

# City population (lakh) from Census 2011 / 2021 provisional
# Used to normalize vehicle counts per-capita
CITY_POPULATION_LAKH: dict[str, float] = {
    "Delhi":     190.8,
    "Mumbai":    184.1,
    "Bengaluru": 121.6,
    "Chennai":    87.5,
    "Kolkata":    45.9,
    "Hyderabad":  97.0,
    "Pune":       50.5,
    "Ahmedabad":  72.5,
    "Lucknow":    29.0,
}

# data.gov.in resource ID for state-wise vehicle registration by fuel type
# Resource: "State and Fuel Type-wise Number of Vehicles Registered"
# Note: resource IDs can change; check data.gov.in if this fails
DATAGOV_RESOURCE_IDS = [
    "9ef84268-d588-465a-a308-a864a43d0070",   # Fuel type state-wise (primary)
    "b4f6f1e4-5fd3-4c71-a5d3-e05f12e4b30b",   # Vehicle category state-wise (backup)
]

DATAGOV_BASE = "https://api.data.gov.in/resource"


def get_datagov_key() -> str | None:
    key = os.environ.get("DATAGOV_API_KEY", "").strip()
    if not key:
        logger.warning(
            "DATAGOV_API_KEY not set. Will use static reference CSV from "
            "configs/vehicle_state_reference.csv. "
            "Get a free key at: https://data.gov.in/"
        )
        return None
    return key


def fetch_datagov_vehicles(api_key: str, resource_id: str) -> pd.DataFrame:
    """Fetch paginated data from data.gov.in for a resource."""
    all_records: list[dict] = []
    offset = 0
    limit = 500

    while True:
        params = {
            "api-key": api_key,
            "format": "json",
            "offset": offset,
            "limit": limit,
        }
        try:
            resp = requests.get(
                f"{DATAGOV_BASE}/{resource_id}",
                params=params,
                timeout=60,
            )
            if resp.status_code != 200:
                logger.warning(f"data.gov.in HTTP {resp.status_code} for resource {resource_id}")
                break
            data = resp.json()
            records = data.get("records", data.get("data", []))
            if not records:
                break
            all_records.extend(records)
            total = int(data.get("total", len(records)))
            logger.debug(f"Fetched {len(all_records)}/{total} records (resource {resource_id})")
            if len(all_records) >= total:
                break
            offset += limit
            time.sleep(0.5)
        except Exception as exc:
            logger.error(f"data.gov.in fetch error: {exc}")
            break

    return pd.DataFrame(all_records)


def load_static_reference() -> pd.DataFrame:
    """
    Load manually compiled MoRTH vehicle reference CSV.
    Format: state | year | total_vehicles | diesel_count | cng_count | ev_count
            petrol_count | two_wheeler_count
    """
    ref_path = Path("configs/vehicle_state_reference.csv")
    if ref_path.exists():
        df = pd.read_csv(ref_path)
        logger.info(f"Loaded static vehicle reference: {len(df)} rows from {ref_path}")
        return df
    else:
        logger.warning(
            f"{ref_path} not found. Creating empty reference — vehicle features will be NaN.\n"
            "Please download MoRTH Road Transport Year Book tables and save as "
            "configs/vehicle_state_reference.csv with columns: "
            "state, year, total_vehicles, diesel_count, cng_count, ev_count, "
            "petrol_count, two_wheeler_count"
        )
        return pd.DataFrame()


def normalise_vehicle_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise vehicle data to common schema."""
    df = df.copy()
    # Try to extract year column
    for col_candidate in ["year", "Year", "FY", "financial_year"]:
        if col_candidate in df.columns:
            df["year"] = pd.to_numeric(df[col_candidate], errors="coerce")
            break

    # State column
    for col_candidate in ["state", "State", "state_name", "StateName"]:
        if col_candidate in df.columns:
            df["state"] = df[col_candidate].astype(str).str.strip()
            break

    # Fuel type columns
    col_map = {
        "Diesel": "diesel_count",
        "diesel": "diesel_count",
        "CNG": "cng_count",
        "cng": "cng_count",
        "Electric": "ev_count",
        "electric": "ev_count",
        "EV": "ev_count",
        "Petrol": "petrol_count",
        "petrol": "petrol_count",
        "total": "total_vehicles",
        "Total": "total_vehicles",
        "two_wheeler": "two_wheeler_count",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    numeric_cols = [
        "total_vehicles", "diesel_count", "cng_count",
        "ev_count", "petrol_count", "two_wheeler_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.strip(),
                errors="coerce",
            )

    return df.dropna(subset=["year", "state"])


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fractions and derived metrics."""
    df = df.copy()
    total = df.get("total_vehicles", pd.Series(dtype=float))
    df["diesel_frac"] = (df.get("diesel_count", 0) / total).clip(0, 1)
    df["cng_frac"] = (df.get("cng_count", 0) / total).clip(0, 1)
    df["ev_frac"] = (df.get("ev_count", 0) / total).clip(0, 1)
    df["petrol_frac"] = (df.get("petrol_count", 0) / total).clip(0, 1)
    df["two_wheeler_frac"] = (df.get("two_wheeler_count", 0) / total).clip(0, 1)
    return df


def map_to_cities(df_state: pd.DataFrame, cities: list[str]) -> pd.DataFrame:
    """Map state-level data to cities."""
    records: list[dict] = []
    for city in cities:
        state = CITY_STATE[city]
        df_city = df_state[df_state["state"].str.contains(state, case=False, na=False)].copy()
        if df_city.empty:
            logger.warning(f"[{city}] No vehicle data found for state: {state}")
            continue
        df_city["city"] = city
        df_city["population_lakh"] = CITY_POPULATION_LAKH[city]
        if "total_vehicles" in df_city.columns:
            df_city["vehicles_per_lakh"] = (
                df_city["total_vehicles"] / df_city["population_lakh"]
            ).round(0)
        records.append(df_city)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True).sort_values(["city", "year"])


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Download vehicle registration data from data.gov.in")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/vehicles"))
    args = parser.parse_args()

    params = load_params()
    cities: list[str] = params["data"]["cities"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    api_key = get_datagov_key()
    df_vehicles = pd.DataFrame()

    if api_key:
        for resource_id in DATAGOV_RESOURCE_IDS:
            logger.info(f"Trying data.gov.in resource: {resource_id}")
            df_try = fetch_datagov_vehicles(api_key, resource_id)
            if not df_try.empty:
                df_vehicles = df_try
                logger.info(f"Got {len(df_try)} records from resource {resource_id}")
                break
            time.sleep(1.0)

    # Fall back to static reference
    if df_vehicles.empty:
        logger.info("Using static reference CSV …")
        df_vehicles = load_static_reference()

    if df_vehicles.empty:
        logger.error(
            "No vehicle data available. Pipeline will proceed without vehicle features.\n"
            "To add real data:\n"
            "  1. Sign up at data.gov.in and set DATAGOV_API_KEY\n"
            "  OR\n"
            "  2. Download MoRTH Road Transport Year Book tables and save as\n"
            "     configs/vehicle_state_reference.csv"
        )
        # Create empty placeholder
        df_city = pd.DataFrame(columns=[
            "city", "year", "state", "total_vehicles", "diesel_frac",
            "cng_frac", "ev_frac", "two_wheeler_frac", "vehicles_per_lakh",
        ])
        city_path = args.output_dir / "city_vehicles_annual.parquet"
        df_city.to_parquet(city_path, index=False)
        return

    df_vehicles = normalise_vehicle_df(df_vehicles)
    df_vehicles = compute_derived_features(df_vehicles)

    # Save state-level
    state_path = args.output_dir / "state_vehicles_annual.parquet"
    df_vehicles.to_parquet(state_path, index=False)
    logger.info(f"Saved state vehicle data → {state_path}")

    # Map to cities
    df_city = map_to_cities(df_vehicles, cities)
    city_path = args.output_dir / "city_vehicles_annual.parquet"
    df_city.to_parquet(city_path, index=False)
    logger.info(f"Saved city vehicle data → {city_path} ({len(df_city)} rows)")

    stats: dict[str, Any] = {
        "downloaded_at": datetime.now().isoformat(),
        "source": "data.gov.in" if api_key else "static_reference",
        "cities_covered": list(df_city["city"].unique()) if not df_city.empty else [],
        "years_covered": sorted(df_city["year"].dropna().astype(int).unique().tolist())
        if not df_city.empty else [],
    }
    metrics_path = Path("reports/metrics/download_vehicles_stats.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("VEHICLE DATA SUMMARY")
    print("=" * 60)
    if not df_city.empty:
        print(df_city[["city", "year", "total_vehicles", "diesel_frac", "cng_frac"]].to_string(index=False))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

