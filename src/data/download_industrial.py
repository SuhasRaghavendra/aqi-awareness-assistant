"""
src/data/download_industrial.py
--------------------------------
Phase 1 – Step D: Download industrial activity data from:
  1. MOSPI IIP (Index of Industrial Production) — monthly manufacturing index
  2. CEA / NPP (National Power Portal) — monthly state-wise power consumption

These are the best open proxies for industrial emission intensity.

For IIP: Download from esankhyiki.mospi.gov.in (national manufacturing index)
For Power: Download from npp.gov.in state-wise energy requirement/availability tables

Since both sources require manual browser download (no REST API with bulk access),
this script:
  a) Checks if pre-downloaded CSVs exist in configs/
  b) If not, provides precise download instructions and URLs
  c) Processes and normalises the data to Parquet

Manual Download Instructions:
  IIP:
    URL: https://esankhyiki.mospi.gov.in/
    → Search "Index of Industrial Production"
    → Filter: Monthly, Manufacturing sector
    → Download CSV (2012–present)
    → Save as: configs/iip_manufacturing.csv

  Power (CEA/NPP):
    URL: https://npp.gov.in → Reports → State Wise Generation Report
    → Download monthly state-wise energy requirement/availability (Excel)
    → Save as: configs/cea_state_power.csv
    Columns expected: state, year, month, energy_requirement_mu, energy_availability_mu

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

# Expected paths for manually downloaded files
IIP_CSV_PATH = Path("configs/iip_manufacturing.csv")
POWER_CSV_PATH = Path("configs/cea_state_power.csv")

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


def load_iip(path: Path) -> pd.DataFrame:
    """
    Load and normalise IIP Manufacturing Index CSV.
    Expected columns: year, month, iip_general, iip_manufacturing, iip_mining, iip_electricity
    OR: date (YYYY-MM), iip_manufacturing_index
    """
    if not path.exists():
        logger.warning(
            f"IIP file not found: {path}\n"
            f"Please download from: https://esankhyiki.mospi.gov.in/\n"
            f"  → 'Index of Industrial Production' → Manufacturing → Monthly CSV\n"
            f"  → Save as {path}"
        )
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info(f"Loaded IIP: {path} ({len(df)} rows, columns: {list(df.columns)})")

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Find date/year/month columns
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
    elif "year" in df.columns and "month" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["month"] = pd.to_numeric(df["month"], errors="coerce")

    # Find IIP manufacturing column
    mfg_candidates = [
        "iip_manufacturing", "manufacturing", "iip_mfg",
        "general_index", "iip_general", "index",
    ]
    iip_col = None
    for c in mfg_candidates:
        if c in df.columns:
            iip_col = c
            break
    if iip_col is None:
        # Use first numeric column
        numeric_cols = df.select_dtypes("number").columns.tolist()
        if numeric_cols:
            iip_col = numeric_cols[0]
            logger.warning(f"IIP column ambiguous, using '{iip_col}'")

    if iip_col:
        df["iip_manufacturing_index"] = pd.to_numeric(df[iip_col], errors="coerce")

    result = df[["year", "month", "iip_manufacturing_index"]].dropna(subset=["year", "month"])
    result = result.sort_values(["year", "month"]).reset_index(drop=True)
    logger.info(f"IIP: {len(result)} monthly records from {result['year'].min():.0f} to {result['year'].max():.0f}")
    return result


def load_state_power(path: Path, cities: list[str]) -> pd.DataFrame:
    """
    Load and normalise CEA state-wise power consumption CSV.
    Expected columns: state, year, month, energy_requirement_mu, energy_availability_mu
    """
    if not path.exists():
        logger.warning(
            f"CEA power file not found: {path}\n"
            f"Please download from: https://npp.gov.in\n"
            f"  → Reports → State Wise Generation/Requirement Report (monthly Excel)\n"
            f"  → Save as {path}\n"
            f"  Expected columns: state, year, month, energy_requirement_mu, energy_availability_mu"
        )
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info(f"Loaded CEA power: {path} ({len(df)} rows)")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Normalise
    for old, new in [
        ("state_name", "state"), ("statename", "state"),
        ("energy_req", "energy_requirement_mu"),
        ("requirement_mu", "energy_requirement_mu"),
        ("energy_avail", "energy_availability_mu"),
        ("availability_mu", "energy_availability_mu"),
    ]:
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    df["year"] = pd.to_numeric(df.get("year", float("nan")), errors="coerce")
    df["month"] = pd.to_numeric(df.get("month", float("nan")), errors="coerce")

    # Filter to states relevant to our 9 cities
    target_states = set(CITY_STATE.values())
    if "state" in df.columns:
        df = df[df["state"].str.contains(
            "|".join(target_states), case=False, na=False
        )]

    numeric_cols = ["energy_requirement_mu", "energy_availability_mu"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""), errors="coerce"
            )

    result = df.dropna(subset=["year", "month"]).reset_index(drop=True)
    return result


def main() -> None:
    setup_logging()
    params = load_params()
    cities: list[str] = params["data"]["cities"]

    output_dir = Path("data/raw/industrial")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {"downloaded_at": datetime.now().isoformat()}

    # --- IIP ---
    df_iip = load_iip(IIP_CSV_PATH)
    if not df_iip.empty:
        iip_path = output_dir / "iip_monthly.parquet"
        df_iip.to_parquet(iip_path, index=False)
        logger.info(f"Saved IIP → {iip_path}")
        stats["iip_rows"] = len(df_iip)
        stats["iip_years"] = sorted(df_iip["year"].dropna().astype(int).unique().tolist())
    else:
        stats["iip_rows"] = 0
        logger.warning("IIP data missing — feature will be NaN until CSV is provided")

    # --- State Power ---
    df_power = load_state_power(POWER_CSV_PATH, cities)
    if not df_power.empty:
        power_path = output_dir / "state_power_monthly.parquet"
        df_power.to_parquet(power_path, index=False)
        logger.info(f"Saved CEA power → {power_path}")
        stats["power_rows"] = len(df_power)
        stats["power_states"] = list(df_power["state"].unique()) if "state" in df_power.columns else []
    else:
        stats["power_rows"] = 0
        logger.warning("CEA power data missing — feature will be NaN until CSV is provided")

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
        print("\n⚠  Some industrial data files are missing.")
        print("   See instructions in src/data/download_industrial.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

