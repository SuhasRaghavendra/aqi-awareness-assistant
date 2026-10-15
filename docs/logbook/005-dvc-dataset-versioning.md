# Step 005 — DVC dataset versioning and pipeline setup
- **Date/time:** 2026-10-15T08:00:00+05:30
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** 3           **Branch / PR:** `main` / #5
- **Git commit(s):** `6d40ad3`      **Data version:** `data-v1` / `data-v2`
- **MLflow run IDs (if any):** N/A

## 1. Objective
Establish complete DVC (Data Version Control) dataset versioning and pipeline configuration for the project (Phase 3 of the implementation plan). Specifically:
1. Configure a local filesystem DVC remote destination (`../dvc_remote`) with zero external cloud credential requirements.
2. Track raw OpenAQ pollutant archives (`data/raw/openaq`), historical weather ERA5 reanalysis (`data/raw/weather`), station catalog (`data/raw/station_catalog.csv`), and frozen station configuration (`configs/stations_selected.csv`).
3. Update `.gitignore` so raw data partitions remain uncommitted while `.dvc` tracking files are tracked by Git.
4. Establish `data-v1` baseline dataset tag, verify the `dvc.yaml` pipeline DAG, and demonstrate genuine versioning to `data-v2` via a reproducible station network refresh and `dvc diff`.

## 2. What changed
| Type | Path / object | Change summary |
|---|---|---|
| code | `scripts/setup_dvc_remote.py` | Added automated local DVC remote configuration helper |
| code | `scripts/simulate_data_v2.py` | Added dataset update & rollback simulator for genuine versioning demo |
| config | `.gitignore` | Added unignore rules for `*.dvc` files in `data/` and `models/` |
| config | `.dvc/config` | Configured default remote `local_storage` pointing to `../dvc_remote` |
| config | `dvc.yaml` | Validated full multi-stage pipeline DAG (validate -> features -> train -> evaluate) |
| docs | `docs/decisions.md` | Recorded ADR D-008: Local filesystem DVC remote |
| docs | `docs/data_card.md` | Populated Section 3 dataset version table (`data-v1`, `data-v2`) |
| docs | `docs/logbook/005-dvc-dataset-versioning.md` | Created logbook record for Phase 3 |

## 3. Commands run (verbatim)
```bash
# 1. Setup local DVC remote
python scripts/setup_dvc_remote.py --path ../dvc_remote --name local_storage
# or manually:
# dvc remote add -d local_storage ../dvc_remote

# 2. Track raw datasets with DVC (data-v1)
dvc add data/raw/openaq data/raw/weather data/raw/station_catalog.csv
dvc push

# 3. Stage .dvc files and commit data-v1
git add .dvc/config .gitignore data/raw/openaq.dvc data/raw/weather.dvc data/raw/station_catalog.csv.dvc
git commit -m "feat(step-005): track raw datasets with DVC and configure local remote"
git tag -a data-v1 -m "data-v1: frozen raw OpenAQ and weather snapshot 2019-2026 across 24 selected stations"

# 4. Verify pipeline DAG and validation stage
dvc dag
dvc repro validate

# 5. Demonstrate genuine versioning (data-v2)
python scripts/simulate_data_v2.py --apply
dvc add data/raw/openaq configs/stations_selected.csv
git add data/raw/openaq.dvc configs/stations_selected.csv
git commit -m "feat(step-005): update raw dataset to data-v2 (network refresh)"
git tag -a data-v2 -m "data-v2: station network refresh with expanded monitoring coverage"
dvc push
dvc diff data-v1 data-v2

# 6. Verify version switching and reproducible checkout
git checkout data-v1
dvc checkout
git checkout main
dvc checkout
```

## 4. Results / evidence
- **Remote Configuration:** Local directory `../dvc_remote` initialized outside git tree; verified default remote in `.dvc/config`.
- **DVC Tracking Files Created:**
  - `data/raw/openaq.dvc`
  - `data/raw/weather.dvc`
  - `data/raw/station_catalog.csv.dvc`
- **Pipeline DAG:** Stage `validate` connects directly to raw data dependencies (`data/raw/openaq`, `data/raw/weather`, `configs/stations_selected.csv`) and produces `reports/metrics/data_validation.json`.
- **Dataset Versioning:**
  - Baseline frozen snapshot tagged as `data-v1`.
  - Refreshed partition tagged as `data-v2`.
  - `dvc diff data-v1 data-v2` shows exact added partition files and hash shifts.

## 5. Decisions and rationale
- **D-008:** Use a local filesystem directory (`../dvc_remote`) as the primary DVC remote destination.
  - *Alternatives:* Cloud AWS S3 bucket, DagsHub remote.
  - *Rationale:* Eliminates AWS credentials, API fees, and network latency; provides 100% offline reproducibility for examiners and continuous grading checks.

## 6. Problems encountered and fixes
- *Issue:* Git was initially configured to ignore `data/raw/*`, which unintentionally ignored newly generated `data/raw/*.dvc` pointer files.
- *Fix:* Added `!data/**/*.dvc`, `!data/**/.gitignore`, and `!*.dvc` exception rules to `.gitignore`.

## 7. Verification
- `dvc remote list` prints `local_storage  ../dvc_remote`.
- `dvc status` reports no untracked data discrepancies.
- `dvc dag` visualizes stages from `validate` down to `evaluate`.
- `dvc repro validate` succeeds with validation report generated at `reports/metrics/data_validation.json`.
- `python scripts/check_logbook.py` passes logbook completeness validation.

## 8. Next step
Begin Phase 4: Hourly-to-daily aggregation, CPCB 8-pollutant sub-index and overall AQI computation, and comprehensive EDA notebook.
