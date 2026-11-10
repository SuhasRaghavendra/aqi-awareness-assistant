"""
src/data/download_industrial.py
---------------------------------
Phase 1 - Step D: Ingest industrial activity proxy data.

Sources:
  1. Index of Industrial Production (IIP) - Manufacturing sector monthly index
     from Ministry of Statistics and Programme Implementation (MOSPI).
     URL: https://esankhyiki.mospi.gov.in/

  2. Central Electricity Authority (CEA) monthly state-level power generation
     and requirement data (thermal/industrial power demand proxy).
     URL: https://npp.gov.in/

Both sources provide monthly data; interpolated to daily in Phase 2.

Usage:
    python -m src.data.download_industrial

Inputs (manual download required if not cached):
    configs/iip_manufacturing.csv
    configs/cea_state_power.csv

Outputs:
    data/raw/industrial/iip_monthly.parquet
    data/raw/industrial/state_power_monthly.parquet
    reports/metrics/download_industrial_stats.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_params
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("download_industrial")

# City -> State mapping for CEA power data
CITY_STATE: dict[str, str] = {
    "Delhi": "Delhi",
    "Mumbai": "Maharashtra",
    "Bengaluru": "Karnataka",
    "Chennai": "Tamil Nadu",
    "Kolkata": "West Bengal",
    "Hyderabad": "Telangana",
    "Pune": "Maharashtra",
    "Ahmedabad": "Gujarat",
    "Lucknow": "Uttar Pradesh",
}

IIP_CSV_PATH = Path("configs/iip_manufacturing.csv")
POWER_CSV_PATH = Path("configs/cea_state_power.csv")


def load_iip(path: Path) -> pd.DataFrame:
    """Load and parse MOSPI IIP Manufacturing index."""
    if not path.exists():
        logger.warning(
            f"IIP file not found: {path}\n"
            "Please download from: https://esankhyiki.mospi.gov.in/\n"
            "  -> 'Index of Industrial Production' -> Manufacturing -> Monthly CSV\n"
            f"  -> Save as {path}"
        )
        return pd.DataFrame()

    df = pd.read_csv(path)
    # Standardise column names: year, month, iip_manufacturing_index
    col_map = {
        "Year": "year",
        "Month": "month",
        "IIP_Manufacturing": "iip_manufacturing_index",
        "Manufacturing": "iip_manufacturing_index",
        "IIP": "iip_manufacturing_index",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Normalise month: string name -> 1-12
    if "month" in df.columns and df["month"].dtype == object:
        month_map = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }
        df["month"] = df["month"].astype(str).str.strip().str.lower().map(month_map)

    # Convert numeric
    for col in ["year", "month", "iip_manufacturing_index"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["year", "month", "iip_manufacturing_index"]).reset_index(drop=True)


def load_state_power(path: Path, cities: list[str]) -> pd.DataFrame:
    """Load CEA state-level monthly power demand data."""
    if not path.exists():
        logger.warning(
            f"CEA power file not found: {path}\n"
            "Please download from: https://npp.gov.in\n"
            "  -> Reports -> State Wise Generation/Requirement Report (monthly Excel)\n"
            f"  -> Save as {path}\n"
            "  Expected columns: state, year, month, energy_requirement_mu, energy_availability_mu"
        )
        return pd.DataFrame()

    df = pd.read_csv(path)
    # Standardise columns
    col_map = {
        "State": "state",
        "Year": "year",
        "Month": "month",
        "Energy_Requirement_MU": "energy_requirement_mu",
        "Energy_Availability_MU": "energy_availability_mu",
        "Requirement": "energy_requirement_mu",
        "Availability": "energy_availability_mu",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Normalise month
    if "month" in df.columns and df["month"].dtype == object:
        month_map = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        df["month"] = df["month"].astype(str).str.strip().str.lower().str[:3].map(month_map)

    for col in ["year", "month", "energy_requirement_mu", "energy_availability_mu"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

    result = df.dropna(subset=["year", "month"]).reset_index(drop=True)
    return result


def main() -> None:
    setup_logging(stage="download_industrial")
    params = load_params()
    cities: list[str] = params["data"]["cities"]

    output_dir = Path("data/raw/industrial")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {"downloaded_at": datetime.now().isoformat()}

    # --- IIP ---
    df_iip = load_iip(IIP_CSV_PATH)
    iip_path = output_dir / "iip_monthly.parquet"
    if not df_iip.empty:
        df_iip.to_parquet(iip_path, index=False)
        logger.info(f"Saved IIP -> {iip_path}")
        stats["iip_rows"] = len(df_iip)
        stats["iip_years"] = sorted(df_iip["year"].dropna().astype(int).unique().tolist())
    else:
        pd.DataFrame(columns=["year", "month", "iip_manufacturing_index"]).to_parquet(
            iip_path, index=False
        )
        stats["iip_rows"] = 0
        logger.warning("IIP data missing -- feature will be NaN until CSV is provided")

    # --- State Power ---
    df_power = load_state_power(POWER_CSV_PATH, cities)
    power_path = output_dir / "state_power_monthly.parquet"
    if not df_power.empty:
        df_power.to_parquet(power_path, index=False)
        logger.info(f"Saved CEA power -> {power_path}")
        stats["power_rows"] = len(df_power)
        stats["power_states"] = (
            list(df_power["state"].unique()) if "state" in df_power.columns else []
        )
    else:
        pd.DataFrame(
            columns=["state", "year", "month", "energy_requirement_mu", "energy_availability_mu"]
        ).to_parquet(power_path, index=False)
        stats["power_rows"] = 0
        logger.warning("CEA power data missing -- feature will be NaN until CSV is provided")

    metrics_path = Path("reports/metrics/download_industrial_stats.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("INDUSTRIAL DATA SUMMARY")
    print("=" * 60)
    print(f"IIP Manufacturing rows:   {stats['iip_rows']}")
    print(f"CEA State Power rows:     {stats['power_rows']}")
    if stats["iip_rows"] == 0 or stats["power_rows"] == 0:
        print("\n[!]  Some industrial data files are missing.")
        print("   See instructions in src/data/download_industrial.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
