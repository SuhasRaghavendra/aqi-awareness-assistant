"""
scripts/simulate_data_v2.py
---------------------------
Phase 3, Step 005: Synthetic/Simulated Versioning Demonstration (data-v1 -> data-v2).

Demonstrates genuine data versioning per Implementation Plan Section 7.2:
- Adds an additional monitoring station / refreshed partition to data/raw/openaq
- Updates configs/stations_selected.csv
- Updates data/raw/openaq_manifest.csv with real SHA-256 and size
- Backs up data-v1 state so --revert can toggle between data versions

Usage:
  python scripts/simulate_data_v2.py --apply   # Create data-v2 update
  python scripts/simulate_data_v2.py --revert  # Revert back to data-v1
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BACKUP_DIR_NAME = ".dvc_v1_backup"


def compute_sha256(filepath: Path) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def apply_data_v2(project_root: Path) -> None:
    stations_csv = project_root / "configs" / "stations_selected.csv"
    manifest_csv = project_root / "data" / "raw" / "openaq_manifest.csv"
    openaq_dir = project_root / "data" / "raw" / "openaq"
    backup_dir = project_root / BACKUP_DIR_NAME

    if not stations_csv.exists():
        print(f"[ERR] {stations_csv} not found. Please ensure Phase 1 data discovery was run.")
        sys.exit(1)

    # 1. Create backup of data-v1 state
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stations_csv, backup_dir / "stations_selected.csv")
    if manifest_csv.exists():
        shutil.copy2(manifest_csv, backup_dir / "openaq_manifest.csv")
    print(f"[OK] Backed up data-v1 configuration to {backup_dir}")

    # 2. Select an additional station or expand coverage in configs/stations_selected.csv
    df_stations = pd.read_csv(stations_csv)
    unselected_idx = df_stations.index[~df_stations["selected"].fillna(False)].tolist()

    added_station_name = "Synthetic Station Refresh (Bengaluru East - KSPCB)"
    target_loc_id = 9999
    if unselected_idx:
        # Pick the first unselected station to activate
        target_idx = unselected_idx[0]
        df_stations.loc[target_idx, "selected"] = True
        df_stations.loc[target_idx, "exclusion_reason"] = "Restored in data-v2 network refresh"
        added_station_name = df_stations.loc[target_idx, "name"]
        target_loc_id = int(df_stations.loc[target_idx, "location_id"])
        print(f"[OK] Activated additional station in catalog: ID {target_loc_id} ({added_station_name})")
    else:
        print("[INFO] All stations were already selected; updating date span and records.")

    df_stations.to_csv(stations_csv, index=False)

    # 3. Create simulated new partition for this station in data/raw/openaq
    partition_rel = Path(f"records/csv.gz/locationid={target_loc_id}/year=2026/month=10")
    partition_dir = openaq_dir / partition_rel
    partition_dir.mkdir(parents=True, exist_ok=True)
    simulated_file = partition_dir / f"location-{target_loc_id}-202610-simulated.csv.gz"

    # Generate synthetic observations for 7 days
    dates = pd.date_range("2026-10-01", "2026-10-07 23:00:00", freq="h", tz="Asia/Kolkata")
    records = []
    for dt in dates:
        records.append({
            "locationId": target_loc_id,
            "location": added_station_name,
            "parameter": "pm25",
            "value": 48.5,
            "unit": "\u00b5g/m\u00b3",
            "datetime": dt.isoformat(),
            "latitude": 12.97,
            "longitude": 77.59,
        })
        records.append({
            "locationId": target_loc_id,
            "location": added_station_name,
            "parameter": "pm10",
            "value": 92.0,
            "unit": "\u00b5g/m\u00b3",
            "datetime": dt.isoformat(),
            "latitude": 12.97,
            "longitude": 77.59,
        })

    df_records = pd.DataFrame(records)
    with gzip.open(simulated_file, "wt", encoding="utf-8") as f:
        df_records.to_csv(f, index=False)

    file_size = simulated_file.stat().st_size
    file_sha = compute_sha256(simulated_file)
    print(f"[OK] Generated data-v2 partition at {simulated_file.relative_to(project_root)} ({file_size} bytes)")

    # 4. Append to manifest if present
    if manifest_csv.exists():
        manifest_df = pd.read_csv(manifest_csv)
        new_manifest_row = pd.DataFrame([{
            "path": str(simulated_file.relative_to(project_root)).replace("\\", "/"),
            "size_bytes": file_size,
            "sha256": file_sha,
            "download_time": datetime.now(timezone.utc).isoformat(),
            "location_id": target_loc_id,
        }])
        manifest_df = pd.concat([manifest_df, new_manifest_row], ignore_index=True)
        manifest_df.to_csv(manifest_csv, index=False)
        print(f"[OK] Appended partition checksum to {manifest_csv.relative_to(project_root)}")

    print("\n" + "=" * 60)
    print("DATA-V2 UPDATE APPLIED SUCCESSFULLY")
    print("=" * 60)
    print(f"Station Activated: ID {target_loc_id} ({added_station_name})")
    print(f"Added File:        {simulated_file.relative_to(project_root)}")
    print(f"File SHA-256:      {file_sha}")
    print("\nNext manual steps to demonstrate genuine DVC versioning:")
    print("  1. dvc add data/raw/openaq configs/stations_selected.csv")
    print("  2. git add data/raw/openaq.dvc configs/stations_selected.csv")
    print('  3. git commit -m "feat(step-005): update raw dataset to data-v2 (added station partition)"')
    print('  4. git tag -a data-v2 -m "data-v2: station network refresh with expanded coverage"')
    print("  5. dvc push")
    print("  6. dvc diff data-v1 data-v2")
    print("=" * 60 + "\n")


def revert_data_v2(project_root: Path) -> None:
    stations_csv = project_root / "configs" / "stations_selected.csv"
    manifest_csv = project_root / "data" / "raw" / "openaq_manifest.csv"
    backup_dir = project_root / BACKUP_DIR_NAME

    if not backup_dir.exists():
        print(f"[ERR] Backup directory {backup_dir} not found. Cannot revert automatically.")
        sys.exit(1)

    backup_stations = backup_dir / "stations_selected.csv"
    if backup_stations.exists():
        shutil.copy2(backup_stations, stations_csv)
        print(f"[OK] Restored {stations_csv} from backup.")

    backup_manifest = backup_dir / "openaq_manifest.csv"
    if backup_manifest.exists():
        shutil.copy2(backup_manifest, manifest_csv)
        print(f"[OK] Restored {manifest_csv} from backup.")

    # Remove backup directory
    shutil.rmtree(backup_dir)
    print("[OK] Reverted configuration to data-v1 state.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate dataset update from data-v1 to data-v2")
    parser.add_argument("--apply", action="store_true", help="Apply simulated data-v2 update")
    parser.add_argument("--revert", action="store_true", help="Revert back to data-v1 state")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.apply:
        apply_data_v2(project_root)
    elif args.revert:
        revert_data_v2(project_root)
    else:
        print("Please specify either --apply or --revert.")
        parser.print_help()


if __name__ == "__main__":
    main()
