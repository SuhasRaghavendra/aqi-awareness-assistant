"""
src/data/discover_stations.py
-----------------------------
Phase 1, Step 002: Pre-approval station discovery.

Queries OpenAQ API v3 for air quality monitoring locations in India,
extracts station metadata, sensors, parameters, and active dates,
applies the parameterised selection rules from params.yaml,
and generates:
  1. Full Indian station catalog -> data/raw/station_catalog.csv
  2. Selected stations list       -> configs/stations_selected.csv
  3. Coverage metrics JSON        -> reports/metrics/station_coverage.json
  4. Coverage visualization       -> reports/figures/eda/station_coverage.png

Usage:
  python -m src.data.discover_stations
  python -m src.data.discover_stations --mock  # offline mock mode for testing without API key
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests

from src.config import get_openaq_api_key, load_params
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.stage_stats import track_stage

logger = get_logger("discover_stations")

OPENAQ_API_BASE = "https://api.openaq.org/v3"
INDIA_ISO = "IN"


def parse_iso_datetime(dt_str: str | None) -> datetime | None:
    """Parse ISO datetime string safely, ensuring UTC timezone."""
    if not dt_str:
        return None
    try:
        clean_str = str(dt_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def calculate_active_years(first_dt: datetime | None, last_dt: datetime | None) -> float:
    """Calculate active span in years between first and last recording."""
    if not first_dt or not last_dt:
        return 0.0
    days = (last_dt - first_dt).total_seconds() / 86400.0
    return max(0.0, round(days / 365.25, 2))


def fetch_openaq_locations(
    api_key: str,
    limit: int = 100,
    max_pages: int | None = None,
    rate_limit_delay: float = 0.5,
) -> list[dict[str, Any]]:
    """Fetch monitoring locations in India from OpenAQ API v3 with pagination and backoff."""
    headers = {
        "X-API-Key": api_key,
        "Accept": "application/json",
        "User-Agent": "AQI-Awareness-Assistant/1.0",
    }
    url = f"{OPENAQ_API_BASE}/locations"
    all_locations: list[dict[str, Any]] = []
    page = 1

    logger.info("Fetching Indian monitoring locations from OpenAQ API v3...")

    while True:
        params: dict[str, Any] = {
            "iso": INDIA_ISO,
            "limit": limit,
            "page": page,
        }

        retries = 3
        resp = None
        while retries > 0:
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=30)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    logger.warning(f"Rate limited (429). Backing off for {retry_after}s...")
                    time.sleep(retry_after)
                    retries -= 1
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                retries -= 1
                logger.warning(f"Request failed (page={page}, retries left={retries}): {exc}")
                if retries == 0:
                    raise
                time.sleep(2.0 * (4 - retries))

        if resp is None:
            break

        data = resp.json()
        results = data.get("results", [])
        if not results:
            logger.info(f"No more results found at page {page}.")
            break

        all_locations.extend(results)
        logger.info(f"Page {page}: fetched {len(results)} locations (total: {len(all_locations)})")

        # Check pagination meta if available
        meta = data.get("meta", {})
        found = meta.get("found", None)
        if isinstance(found, int) and len(all_locations) >= found:
            break

        if len(results) < limit:
            break

        if max_pages and page >= max_pages:
            logger.info(f"Reached user-specified max_pages limit ({max_pages}).")
            break

        page += 1
        time.sleep(rate_limit_delay)

    logger.info(f"Successfully retrieved {len(all_locations)} total locations from OpenAQ.")
    return all_locations


def generate_mock_locations() -> list[dict[str, Any]]:
    """Generate realistic mock station metadata for testing and offline development."""
    logger.info("Generating realistic mock locations for offline development...")
    cities_data = [
        # Bengaluru stations
        (
            "BTM Layout, Bengaluru",
            "Bengaluru",
            12.9165,
            77.6101,
            ["pm25", "pm10", "no2", "so2", "co", "o3"],
            "2019-01-01T00:00:00Z",
            "2026-09-01T00:00:00Z",
        ),
        (
            "Hebbal, Bengaluru",
            "Bengaluru",
            13.0358,
            77.5970,
            ["pm25", "pm10", "no2", "so2", "co", "o3"],
            "2019-02-15T00:00:00Z",
            "2026-08-30T00:00:00Z",
        ),
        (
            "Peenya, Bengaluru",
            "Bengaluru",
            13.0329,
            77.5273,
            ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3"],
            "2018-11-01T00:00:00Z",
            "2026-09-15T00:00:00Z",
        ),
        (
            "Silk Board, Bengaluru",
            "Bengaluru",
            12.9172,
            77.6227,
            ["pm25", "pm10", "no2", "co"],
            "2020-03-01T00:00:00Z",
            "2026-09-10T00:00:00Z",
        ),
        (
            "BWSSB Kadabesanahalli, Bengaluru",
            "Bengaluru",
            12.9351,
            77.6961,
            ["pm25", "pm10", "no2", "so2", "co", "o3"],
            "2019-06-01T00:00:00Z",
            "2026-09-20T00:00:00Z",
        ),
        (
            "City Railway Station, Bengaluru",
            "Bengaluru",
            12.9779,
            77.5667,
            ["pm25", "pm10", "no2", "so2", "o3"],
            "2019-01-01T00:00:00Z",
            "2026-09-01T00:00:00Z",
        ),
        (
            "Saneguruvanahalli, Bengaluru",
            "Bengaluru",
            12.9917,
            77.5447,
            ["pm25", "pm10", "no2"],
            "2021-01-01T00:00:00Z",
            "2026-07-01T00:00:00Z",
        ),
        (
            "Jayanagar, Bengaluru",
            "Bengaluru",
            12.9308,
            77.5838,
            ["pm25", "pm10", "no2", "so2", "co", "o3"],
            "2019-01-01T00:00:00Z",
            "2026-09-25T00:00:00Z",
        ),
        # Delhi stations
        (
            "Anand Vihar, Delhi",
            "Delhi",
            28.6476,
            77.3158,
            ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3"],
            "2018-01-01T00:00:00Z",
            "2026-09-30T00:00:00Z",
        ),
        (
            "R K Puram, Delhi",
            "Delhi",
            28.5632,
            77.1869,
            ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3"],
            "2018-05-01T00:00:00Z",
            "2026-09-30T00:00:00Z",
        ),
        (
            "Punjabi Bagh, Delhi",
            "Delhi",
            28.6740,
            77.1310,
            ["pm25", "pm10", "no2", "so2", "co", "o3"],
            "2019-01-01T00:00:00Z",
            "2026-09-30T00:00:00Z",
        ),
        (
            "Mandir Marg, Delhi",
            "Delhi",
            28.6364,
            77.1994,
            ["pm25", "pm10", "no2", "so2", "co", "o3"],
            "2019-01-01T00:00:00Z",
            "2026-09-30T00:00:00Z",
        ),
        (
            "IHBAS Dilshad Garden, Delhi",
            "Delhi",
            28.6811,
            77.3152,
            ["pm25", "pm10", "no2", "so2"],
            "2019-03-01T00:00:00Z",
            "2026-09-30T00:00:00Z",
        ),
        # Mumbai stations
        (
            "Bandra Kurla Complex, Mumbai",
            "Mumbai",
            19.0657,
            72.8687,
            ["pm25", "pm10", "no2", "so2", "co", "o3"],
            "2019-01-01T00:00:00Z",
            "2026-09-25T00:00:00Z",
        ),
        (
            "Colaba, Mumbai",
            "Mumbai",
            18.9067,
            72.8147,
            ["pm25", "pm10", "no2", "so2", "co"],
            "2019-04-01T00:00:00Z",
            "2026-09-20T00:00:00Z",
        ),
        (
            "Worli, Mumbai",
            "Mumbai",
            19.0178,
            72.8178,
            ["pm25", "pm10", "no2", "o3"],
            "2019-02-01T00:00:00Z",
            "2026-09-25T00:00:00Z",
        ),
        # Chennai stations
        (
            "Alandur Bus Depot, Chennai",
            "Chennai",
            13.0040,
            80.2016,
            ["pm25", "pm10", "no2", "so2", "co", "o3"],
            "2019-01-01T00:00:00Z",
            "2026-09-15T00:00:00Z",
        ),
        (
            "Velachery Res. Area, Chennai",
            "Chennai",
            12.9759,
            80.2212,
            ["pm25", "pm10", "no2", "so2", "co", "o3"],
            "2019-03-01T00:00:00Z",
            "2026-09-25T00:00:00Z",
        ),
        # Kolkata stations
        (
            "Victoria Memorial, Kolkata",
            "Kolkata",
            22.5448,
            88.3426,
            ["pm25", "pm10", "no2", "so2", "co", "o3"],
            "2019-01-01T00:00:00Z",
            "2026-09-20T00:00:00Z",
        ),
        (
            "Jadavpur, Kolkata",
            "Kolkata",
            22.4988,
            88.3718,
            ["pm25", "pm10", "no2", "so2", "co", "o3"],
            "2019-05-01T00:00:00Z",
            "2026-09-25T00:00:00Z",
        ),
        # Hyderabad stations
        (
            "Sanathnagar, Hyderabad",
            "Hyderabad",
            17.4589,
            78.4419,
            ["pm25", "pm10", "no2", "so2", "co", "o3"],
            "2019-01-01T00:00:00Z",
            "2026-09-20T00:00:00Z",
        ),
        (
            "Zoo Park, Hyderabad",
            "Hyderabad",
            17.3489,
            78.4519,
            ["pm25", "pm10", "no2", "so2", "co", "o3"],
            "2019-02-01T00:00:00Z",
            "2026-09-20T00:00:00Z",
        ),
        # Pune stations
        (
            "Shivajinagar, Pune",
            "Pune",
            18.5314,
            73.8446,
            ["pm25", "pm10", "no2", "so2", "co", "o3"],
            "2019-01-01T00:00:00Z",
            "2026-09-25T00:00:00Z",
        ),
        (
            "Karve Road, Pune",
            "Pune",
            18.5018,
            73.8286,
            ["pm25", "pm10", "no2", "so2"],
            "2019-04-01T00:00:00Z",
            "2026-09-20T00:00:00Z",
        ),
        # Short / insufficient stations (for exclusion test)
        (
            "Short Term Station, Bengaluru",
            "Bengaluru",
            12.9500,
            77.6000,
            ["pm25"],
            "2024-01-01T00:00:00Z",
            "2024-06-01T00:00:00Z",
        ),
        (
            "No PM Station, Delhi",
            "Delhi",
            28.6000,
            77.2000,
            ["no2", "so2", "co"],
            "2019-01-01T00:00:00Z",
            "2026-09-01T00:00:00Z",
        ),
        (
            "Rural Remote Station, Karnataka",
            "Mandya",
            12.5200,
            76.9000,
            ["pm25", "pm10"],
            "2020-01-01T00:00:00Z",
            "2025-01-01T00:00:00Z",
        ),
    ]

    mock_results = []
    for idx, (name, city, lat, lon, params, dt_first, dt_last) in enumerate(cities_data, start=101):
        sensors = []
        for s_idx, p in enumerate(params, start=1):
            unit = "mg/m³" if p == "co" else "µg/m³"
            sensors.append(
                {
                    "id": idx * 100 + s_idx,
                    "name": f"{p}-sensor",
                    "parameter": {"id": s_idx, "name": p, "units": unit, "displayName": p.upper()},
                    "datetimeFirst": {"utc": dt_first, "local": dt_first},
                    "datetimeLast": {"utc": dt_last, "local": dt_last},
                }
            )

        mock_results.append(
            {
                "id": idx,
                "name": name,
                "locality": city,
                "timezone": "Asia/Kolkata",
                "country": {"id": 9, "code": "IN", "name": "India"},
                "provider": {"id": 1, "name": "CPCB / State Pollution Control Board"},
                "coordinates": {"latitude": lat, "longitude": lon},
                "sensors": sensors,
                "datetimeFirst": {"utc": dt_first, "local": dt_first},
                "datetimeLast": {"utc": dt_last, "local": dt_last},
            }
        )

    return mock_results


def parse_location_to_record(loc: dict[str, Any]) -> dict[str, Any]:
    """Extract standard station metadata dictionary from OpenAQ location object."""
    loc_id = loc.get("id")
    name = loc.get("name") or f"Station_{loc_id}"
    locality = loc.get("locality") or ""

    # Provider extraction
    provider_obj = loc.get("provider") or loc.get("owner") or {}
    provider_name = (
        provider_obj.get("name") if isinstance(provider_obj, dict) else str(provider_obj)
    )
    if not provider_name or provider_name == "None":
        provider_name = "CPCB / SPCB"

    # Coordinates
    coords = loc.get("coordinates") or {}
    lat = coords.get("latitude")
    lon = coords.get("longitude")

    # Sensors and parameters
    sensors = loc.get("sensors") or []
    param_list: list[str] = []
    param_units: dict[str, str] = {}

    for s in sensors:
        if not isinstance(s, dict):
            continue
        p_obj = s.get("parameter") or {}
        if isinstance(p_obj, dict):
            p_name = str(p_obj.get("name", "")).lower().strip()
            p_unit = str(p_obj.get("units", "µg/m³")).strip()
            if p_name and p_name not in param_list:
                param_list.append(p_name)
                param_units[p_name] = p_unit

    # Datetimes
    dt_first_obj = loc.get("datetimeFirst") or {}
    dt_first_str = dt_first_obj.get("utc") if isinstance(dt_first_obj, dict) else str(dt_first_obj)

    dt_last_obj = loc.get("datetimeLast") or {}
    dt_last_str = dt_last_obj.get("utc") if isinstance(dt_last_obj, dict) else str(dt_last_obj)

    dt_first = parse_iso_datetime(dt_first_str)
    dt_last = parse_iso_datetime(dt_last_str)
    active_years = calculate_active_years(dt_first, dt_last)

    has_pm25 = "pm25" in param_list
    has_pm10 = "pm10" in param_list

    return {
        "location_id": loc_id,
        "name": name,
        "locality": locality,
        "provider": provider_name,
        "latitude": lat,
        "longitude": lon,
        "sensor_count": len(sensors),
        "parameters": ",".join(param_list),
        "parameter_units": json.dumps(param_units),
        "has_pm25": has_pm25,
        "has_pm10": has_pm10,
        "has_pm": (has_pm25 or has_pm10),
        "datetime_first_utc": dt_first.isoformat() if dt_first else "",
        "datetime_last_utc": dt_last.isoformat() if dt_last else "",
        "active_years": active_years,
    }


def map_to_target_city(name: str, locality: str, target_cities: list[str]) -> str | None:
    """Match station name/locality to target cities list (case-insensitive)."""
    text = f"{name} {locality}".lower()

    # Bengaluru aliases
    if "bengaluru" in text or "bangalore" in text:
        return "Bengaluru"

    for city in target_cities:
        if city.lower() in text:
            return city

    return None


def evaluate_selection_rules(
    df: pd.DataFrame,
    target_cities: list[str],
    min_years: float,
    window_start: str,
    snapshot_end: str,
) -> pd.DataFrame:
    """Apply selection criteria: city match, PM2.5/PM10 present, active years >= threshold."""
    df = df.copy()

    win_start_dt = parse_iso_datetime(window_start)
    snap_end_dt = parse_iso_datetime(snapshot_end)

    matched_cities = []
    selected_flags = []
    exclusion_reasons = []

    for _, row in df.iterrows():
        matched_city = map_to_target_city(str(row["name"]), str(row["locality"]), target_cities)
        matched_cities.append(matched_city or "")

        reasons = []
        if not matched_city:
            reasons.append("City not in target geographic scope")
        if not row["has_pm"]:
            reasons.append("Lacks both PM2.5 and PM10 sensors")
        if row["active_years"] < min_years:
            reasons.append(f"Active span {row['active_years']:.1f}y < {min_years}y threshold")

        # Window check
        dt_first = parse_iso_datetime(row["datetime_first_utc"])
        dt_last = parse_iso_datetime(row["datetime_last_utc"])

        if snap_end_dt and dt_first and dt_first > snap_end_dt:
            reasons.append(f"First reading after snapshot end ({snapshot_end})")
        if win_start_dt and dt_last and dt_last < win_start_dt:
            reasons.append(f"Last reading before study window ({window_start})")

        is_selected = len(reasons) == 0
        selected_flags.append(is_selected)
        exclusion_reasons.append("; ".join(reasons) if reasons else "Selected")

    df["target_city"] = matched_cities
    df["selected"] = selected_flags
    df["selection_status"] = exclusion_reasons
    return df


def estimate_download_volume(
    selected_df: pd.DataFrame, window_start: str, snapshot_end: str
) -> dict[str, Any]:
    """Estimate AWS S3 monthly partition files count and expected data size."""
    start_dt = parse_iso_datetime(window_start) or datetime(2019, 1, 1)
    end_dt = parse_iso_datetime(snapshot_end) or datetime(2026, 9, 30)

    # Number of months in the window
    months_span = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
    station_count = len(selected_df)

    # In OpenAQ S3, partition is locationid=<id>/year=<yyyy>/month=<mm>/
    # Average daily gzip CSV size ~10-25 KB, ~30 files per month per station
    # Total monthly gzipped archive per station is ~300 KB - 1 MB
    est_files = station_count * months_span * 30
    est_compressed_mb = round(station_count * months_span * 0.5, 2)
    est_uncompressed_mb = round(est_compressed_mb * 7.5, 2)

    return {
        "selected_stations_count": station_count,
        "window_months": months_span,
        "estimated_files_count": est_files,
        "estimated_compressed_size_mb": est_compressed_mb,
        "estimated_uncompressed_size_mb": est_uncompressed_mb,
    }


def plot_coverage(
    catalog_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate professional coverage figures showing stations by city and selection status."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Plot 1: Stations per target city (Selected vs Excluded)
    city_counts = (
        catalog_df[catalog_df["target_city"] != ""]
        .groupby(["target_city", "selected"])
        .size()
        .unstack(fill_value=0)
    )

    if not city_counts.empty:
        city_counts.plot(
            kind="bar",
            stacked=True,
            ax=axes[0],
            color=["#d9534f", "#2b8a3e"],
            edgecolor="black",
            alpha=0.85,
        )
        axes[0].set_title(
            "Stations per Target City (Selected vs Excluded)", fontsize=13, fontweight="bold"
        )
        axes[0].set_xlabel("City", fontsize=11)
        axes[0].set_ylabel("Station Count", fontsize=11)
        axes[0].legend(["Excluded", "Selected"], frameon=True)
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)
        axes[0].tick_params(axis="x", rotation=45)
    else:
        axes[0].text(0.5, 0.5, "No target cities matched", ha="center", va="center")

    # Plot 2: Active years distribution for selected stations
    if not selected_df.empty:
        axes[1].hist(
            selected_df["active_years"],
            bins=10,
            color="#1c7ed6",
            edgecolor="black",
            alpha=0.85,
        )
        axes[1].axvline(
            selected_df["active_years"].median(),
            color="red",
            linestyle="dashed",
            linewidth=1.5,
            label=f"Median: {selected_df['active_years'].median():.1f} yrs",
        )
        axes[1].set_title(
            "Active Monitoring Span (Selected Stations)", fontsize=13, fontweight="bold"
        )
        axes[1].set_xlabel("Active Span (Years)", fontsize=11)
        axes[1].set_ylabel("Number of Stations", fontsize=11)
        axes[1].legend(frameon=True)
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    else:
        axes[1].text(0.5, 0.5, "No selected stations", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved coverage figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Discover OpenAQ Indian Stations")
    parser.add_argument(
        "--mock", action="store_true", help="Use realistic offline mock data without API key"
    )
    parser.add_argument("--limit", type=int, default=100, help="Results limit per API page")
    parser.add_argument("--max-pages", type=int, default=None, help="Max API pages to fetch")
    parser.add_argument("--output-catalog", type=Path, default=Path("data/raw/station_catalog.csv"))
    parser.add_argument(
        "--output-selected", type=Path, default=Path("configs/stations_selected.csv")
    )
    parser.add_argument(
        "--output-metrics", type=Path, default=Path("reports/metrics/station_coverage.json")
    )
    parser.add_argument(
        "--output-figure", type=Path, default=Path("reports/figures/eda/station_coverage.png")
    )
    args = parser.parse_args()

    setup_logging(stage="discover_stations")
    logger.info("Starting Phase 1 Step 002: Station Discovery...")

    params = load_params()
    data_cfg = params.get("data", {})
    target_cities = data_cfg.get("cities", ["Bengaluru", "Delhi", "Mumbai"])
    min_years = float(data_cfg.get("min_years_per_station", 3.0))
    window_start = data_cfg.get("window_start", "2019-01-01")
    snapshot_end = data_cfg.get("snapshot_end", "2026-09-30")

    logger.info(f"Target cities: {target_cities}")
    logger.info(
        f"Selection criteria: min_years={min_years}, window={window_start} -> {snapshot_end}"
    )

    # Fetch or Mock locations
    if args.mock:
        raw_locations = generate_mock_locations()
    else:
        try:
            api_key = get_openaq_api_key()
            raw_locations = fetch_openaq_locations(
                api_key=api_key, limit=args.limit, max_pages=args.max_pages
            )
        except OSError as e:
            logger.error(f"{e}")
            logger.info("TIP: To test immediately without an API key, run with --mock")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error fetching OpenAQ locations: {e}")
            sys.exit(1)

    # Parse into catalog records
    records = [parse_location_to_record(loc) for loc in raw_locations]
    catalog_df = pd.DataFrame(records)

    # Evaluate selection rules
    catalog_df = evaluate_selection_rules(
        catalog_df,
        target_cities=target_cities,
        min_years=min_years,
        window_start=window_start,
        snapshot_end=snapshot_end,
    )

    # Filter selected stations
    selected_df = catalog_df[catalog_df["selected"]].copy()

    # Track stage stats
    args.output_catalog.parent.mkdir(parents=True, exist_ok=True)
    args.output_selected.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)

    with track_stage(
        "discover_stations",
        inputs=[],
        outputs=[str(args.output_catalog), str(args.output_selected)],
    ) as st:
        st.record_input_frame(catalog_df)
        st.record_output_frame(selected_df)
        excluded_count = len(catalog_df) - len(selected_df)
        if excluded_count > 0:
            st.drop("Excluded (not meeting geographic scope or minimum criteria)", excluded_count)

        catalog_df.to_csv(args.output_catalog, index=False)
        selected_df.to_csv(args.output_selected, index=False)
        logger.info(f"Wrote full station catalog ({len(catalog_df)} rows) to {args.output_catalog}")
        logger.info(
            f"Wrote selected station list ({len(selected_df)} rows) to {args.output_selected}"
        )

        # Metrics calculation
        download_estimates = estimate_download_volume(selected_df, window_start, snapshot_end)

        stations_per_city = (
            catalog_df[catalog_df["target_city"] != ""]
            .groupby("target_city")["location_id"]
            .count()
            .to_dict()
        )
        selected_per_city = selected_df.groupby("target_city")["location_id"].count().to_dict()
        providers = catalog_df["provider"].value_counts().to_dict()

        metrics = {
            "total_locations_discovered": len(catalog_df),
            "target_cities": target_cities,
            "stations_per_city_discovered": stations_per_city,
            "stations_per_city_selected": selected_per_city,
            "total_selected_stations": len(selected_df),
            "providers": providers,
            "download_estimates": download_estimates,
            "generated_at": datetime.now().isoformat(),
        }

        with args.output_metrics.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Wrote coverage metrics to {args.output_metrics}")

        # Plot figure
        plot_coverage(catalog_df, selected_df, args.output_figure)

    print("\n" + "=" * 60)
    print("STATION DISCOVERY SUMMARY")
    print("=" * 60)
    print(f"Total Indian Stations Discovered: {len(catalog_df)}")
    print(f"Total Stations Selected:          {len(selected_df)}")
    print("\nSelected Stations Per City:")
    for c, cnt in selected_per_city.items():
        print(f"  - {c:15s}: {cnt} stations")
    print("\nEstimated Download Volume:")
    print(f"  - Files count:       ~{download_estimates['estimated_files_count']:,} files")
    print(f"  - Compressed size:   ~{download_estimates['estimated_compressed_size_mb']:.1f} MB")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
