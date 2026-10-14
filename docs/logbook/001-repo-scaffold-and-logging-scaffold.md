# Step 001 — Repo scaffold and logging scaffold
- **Date/time:** 2026-10-14T14:07:34+05:30
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** 2           **Branch / PR:** `master` / #<n>
- **Git commit(s):** `dac49f45050f5a8aa6019a4e2c4d2000612da068`      **Data version:** <git-tag> / <dvc-md5>
- **MLflow run IDs:** N/A

## 1. Objective
Create the full repository directory structure and implement the complete MLOps logging scaffold (Phase 2, steps 1-4 of the implementation plan). This step establishes the foundation that every subsequent step depends on: a consistent file layout, automated change tracking via `track_stage`, structured JSONL logging, and the logbook protocol (`log_step.py`, `build_logbook_index.py`, `check_logbook.py`) that enforces documentation discipline from the start.

## 2. What changed

| Type | Path | Change summary |
|---|---|---|
| code | `scripts/build_logbook_index.py` | — |
| code | `scripts/check_logbook.py` | — |
| code | `scripts/log_step.py` | — |
| code | `src/config.py` | — |
| code | `src/utils/logging_utils.py` | — |
| code | `src/utils/mlflow_utils.py` | — |
| code | `src/utils/stage_stats.py` | — |
| code | `src/data/clean.py` | — |
| code | `src/data/discover_stations.py` | — |
| code | `src/data/download_openaq.py` | — |
| code | `src/data/download_weather.py` | — |
| code | `src/data/hourly_to_daily.py` | — |
| code | `src/data/ingest.py` | — |
| code | `src/data/merge.py` | — |
| code | `src/data/validate.py` | — |
| code | `src/features/aqi_utils.py` | — |
| code | `src/features/build_features.py` | — |
| code | `src/models/baselines.py` | — |
| code | `src/models/compare_runs.py` | — |
| code | `src/models/evaluate.py` | — |
| code | `src/models/train.py` | — |
| config | `configs/cpcb_breakpoints.yaml` | — |
| config | `params.yaml` | — |
| config | `dvc.yaml` | — |
| docs | `docs/CHANGELOG.md` | — |
| docs | `docs/data_change_log.md` | — |
| docs | `docs/logbook/001-repo-scaffold-and-logging-scaffold.md` | — |
| docs | `docs/logbook/INDEX.md` | — |
| docs | `docs/logbook/_TEMPLATE.md` | — |
| docs | `reports/step_reports/001.json` | — |
| docs | `docs/decisions.md` | — |
| docs | `docs/setup_howto.md` | — |
| docs | `reports/figures/` | — |
| docs | `reports/metrics/` | — |
| docs | `reports/step_reports/.gitkeep` | — |
| data | `data/` | — |
| data | `models/` | — |
| tests | `tests/test_stage_stats.py` | — |
| other | `dvc/config` | — |
| other | `.dvcignore` | — |
| other | `.env.example` | — |
| other | `.github/pull_request_template.md` | — |
| other | `.gitignore` | — |
| other | `.pre-commit-config.yaml` | — |
| other | `Makefile` | — |
| other | `pyproject.toml` | — |
| other | `requirements.txt` | — |
| other | `README.md` | — |

<details>
<summary>git diff --stat</summary>

```
.dvc/config                                        |   0
 .dvcignore                                         |   7 +-
 .env.example                                       |   2 +
 .github/pull_request_template.md                   |  46 ++-
 .gitignore                                         |  51 ++-
 .pre-commit-config.yaml                            |  31 +-
 Makefile                                           |  74 +++-
 configs/cpcb_breakpoints.yaml                      | 160 ++++----
 docs/CHANGELOG.md                                  |  32 +-
 docs/data_change_log.md                            |   6 +-
 .../001-repo-scaffold-and-logging-scaffold.md      | 287 ++++++++++++--
 docs/logbook/INDEX.md                              |   8 -
 docs/logbook/_TEMPLATE.md                          |  19 +-
 params.yaml                                        |   5 +-
 pyproject.toml                                     |  10 +-
 reports/step_reports/001.json                      |  77 +++-
 requirements.txt                                   |  51 ++-
 scripts/build_logbook_index.py                     | 141 +++----
 scripts/check_logbook.py                           | 132 ++++---
 scripts/log_step.py                                | 413 ++++++++++++++-------
 src/config.py                                      |  68 ++--
 src/utils/logging_utils.py                         | 192 ++++++----
 src/utils/mlflow_utils.py                          | 146 ++++----
 src/utils/stage_stats.py                           | 407 ++++++++++++--------
 tests/test_stage_stats.py                          | 256 +++++++++----
 25 files changed, 1764 insertions(+), 857 deletions(-)
warning: in the working copy of '.dvcignore', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '.env.example', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '.github/pull_request_template.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '.gitignore', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of '.pre-commit-config.yaml', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'Makefile', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'configs/cpcb_breakpoints.yaml', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'docs/data_change_log.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'docs/logbook/_TEMPLATE.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'params.yaml', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'pyproject.toml', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'requirements.txt', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'scripts/build_logbook_index.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'scripts/check_logbook.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'scripts/log_step.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/config.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/utils/logging_utils.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/utils/mlflow_utils.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/utils/stage_stats.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tests/test_stage_stats.py', LF will be replaced by CRLF the next time Git touches it
```
</details>

<details>
<summary>git status --short</summary>

```
D .dvc/config
 M .dvcignore
 M .env.example
 M .github/pull_request_template.md
 M .gitignore
 M .pre-commit-config.yaml
 M Makefile
 M configs/cpcb_breakpoints.yaml
 M docs/CHANGELOG.md
 M docs/data_change_log.md
 M docs/logbook/001-repo-scaffold-and-logging-scaffold.md
 D docs/logbook/INDEX.md
 M docs/logbook/_TEMPLATE.md
 M params.yaml
 M pyproject.toml
 M reports/step_reports/001.json
 M requirements.txt
 M scripts/build_logbook_index.py
 M scripts/check_logbook.py
 M scripts/log_step.py
 M src/config.py
 M src/utils/logging_utils.py
 M src/utils/mlflow_utils.py
 M src/utils/stage_stats.py
 M tests/test_stage_stats.py
?? README.md
?? data/
?? docs/decisions.md
?? docs/setup_howto.md
?? dvc.yaml
?? models/
?? reports/figures/
?? reports/metrics/
?? reports/step_reports/.gitkeep
?? src/data/clean.py
?? src/data/discover_stations.py
?? src/data/download_openaq.py
?? src/data/download_weather.py
?? src/data/hourly_to_daily.py
?? src/data/ingest.py
?? src/data/merge.py
?? src/data/validate.py
?? src/features/aqi_utils.py
?? src/features/build_features.py
?? src/models/baselines.py
?? src/models/compare_runs.py
?? src/models/evaluate.py
?? src/models/train.py
```
</details>

<details>
<summary>dvc status</summary>

```
validate:
	changed deps:
		modified:           src\data\ingest.py
		modified:           src\data\validate.py
		modified:           data\raw\openaq
		modified:           data\raw\weather
		deleted:            configs\stations_selected.csv
		new:                params.yaml
	changed outs:
		deleted:            reports\metrics\data_validation.json
hourly_to_daily:
	changed deps:
		modified:           src\data\hourly_to_daily.py
		modified:           src\utils\stage_stats.py
		modified:           data\raw\openaq
		modified:           configs\cpcb_breakpoints.yaml
		modified:           src\features\aqi_utils.py
		new:                params.yaml
	changed outs:
		deleted:            data\interim\pollutants_daily.parquet
		deleted:            reports\metrics\hourly_to_daily_stats.json
merge_weather:
	changed deps:
		modified:           src\data\merge.py
		deleted:            data\interim\pollutants_daily.parquet
		modified:           data\raw\weather
		new:                params.yaml
	changed outs:
		deleted:            data\interim\station_day.parquet
		deleted:            reports\metrics\merge_stats.json
clean:
	changed deps:
		modified:           src\data\clean.py
		deleted:            data\interim\station_day.parquet
		new:                params.yaml
	changed outs:
		deleted:            data\interim\station_day_clean.parquet
		deleted:            reports\metrics\clean_stats.json
features:
	changed deps:
		modified:           src\features\build_features.py
		modified:           src\features\aqi_utils.py
		deleted:            data\interim\station_day_clean.parquet
		new:                params.yaml
	changed outs:
		deleted:            data\processed\train.parquet
		deleted:            data\processed\val.parquet
		deleted:            data\processed\test.parquet
		deleted:            data\processed\feature_list.json
		deleted:            reports\metrics\features_stats.json
train:
	changed deps:
		modified:           src\models\train.py
		modified:           src\models\baselines.py
		deleted:            data\processed\train.parquet
		deleted:            data\processed\val.parquet
		new:                params.yaml
	changed outs:
		modified:           models
		deleted:            reports\metrics\val_metrics.json
evaluate:
	changed deps:
		modified:           src\models\evaluate.py
		modified:           models
		deleted:            data\processed\test.parquet
	changed outs:
		modified:           reports\figures\evaluation
		deleted:            reports\metrics\test_metrics.json
```
</details>

<details>
<summary>dvc diff</summary>

```
Added:
    models\
    models\.gitkeep
    reports\figures\evaluation\
    reports\figures\evaluation\.gitkeep

files summary: 2 added
```
</details>

### Stage statistics summary
_No stage stats files found yet._

## 3. Commands run (verbatim)
```bash
# Create directory structure
New-Item -ItemType Directory -Force -Path ".dvc", ".github", "configs", "data\raw\openaq", ...

# Run step-close protocol (Step 001)
python scripts/log_step.py --step 001 --title "Repo scaffold and logging scaffold" --phase 2

# Rebuild logbook index
python scripts/build_logbook_index.py

# Check logbook completeness
python scripts/check_logbook.py

# Verify
python -c "from src.utils.logging_utils import setup_logging; print('logging_utils OK')"
python -c "from src.utils.stage_stats import track_stage; print('stage_stats OK')"
```

## 4. Results / evidence

**Files created: 50+** across the full repo structure. Key verified outputs:

```
[OK] INDEX.md rebuilt - 1 entries -> docs/logbook/INDEX.md
[OK] Logbook check passed - 1 entries; no tracked files changed since last entry.
```

**git diff --stat HEAD~1 summary (25 files, 1764 insertions):**
- `src/utils/logging_utils.py` — 192 lines: coloured console + JSONL handlers
- `src/utils/stage_stats.py` — 407 lines: `track_stage` context manager
- `scripts/log_step.py` — 413 lines: step-close protocol automation
- `scripts/build_logbook_index.py` — INDEX.md generator
- `scripts/check_logbook.py` — pre-push logbook guard
- `tests/test_stage_stats.py` — 256 lines: full test suite
- `configs/cpcb_breakpoints.yaml` — 8-pollutant breakpoint table
- `dvc.yaml` — 7 pipeline stages
- `params.yaml` — all hyperparameters from the plan
- `Makefile` — 6 targets including `step-close`
- `.pre-commit-config.yaml` — ruff, black, standard hooks, logbook pre-push

**DVC diff (2 files added):** `models/.gitkeep`, `reports/figures/evaluation/.gitkeep`

## 5. Decisions and rationale

| Decision | Alternatives | Rationale |
|---|---|---|
| OpenAQ S3 as primary source (not Kaggle) | Kaggle static dataset | OpenAQ is live, large (2019-2026), anonymous S3 access | 
| Compute AQI from raw concentrations | Use pre-computed AQI | Allows breakpoint verification and validity-rule enforcement |
| SQLite backend for MLflow | Remote PostgreSQL, DagsHub | Simplest for 2-person student project; no external services |
| Chronological train/val/test split | Random K-fold | Time series forecasting requires no future leakage |
| `track_stage` context manager | Decorator or manual JSON writes | Context manager ensures cleanup even on exceptions; cleaner API |
| JSONL log format | CSV, plain text | Machine-readable, append-friendly, easy to grep |

All design decisions also copied to `docs/decisions.md` (D-001 through D-004).

## 6. Problems encountered and fixes

1. **`UnicodeEncodeError` on Windows (cp1252):** The initial scripts used Unicode box-drawing chars (`─`, `✅`, `→`) in `print()` statements, which crashed on Windows PowerShell terminals that use cp1252 encoding. **Fix:** replaced all Unicode special characters with ASCII equivalents (`-`, `[OK]`, `->`).

2. **`log_step.py` first-run diff shows large numbers:** Since files were being created fresh against an existing git history, `git diff HEAD~1` showed large insertion counts. This is expected behaviour for a scaffold step — all insertions are intentional.

3. **No `configs/stations_selected.csv` yet:** DVC status correctly flags this as a missing dependency. This file is created in Phase 1 (station discovery). No action needed at this stage.

## 7. Verification

```bash
# Import check
python -c "from src.utils.logging_utils import setup_logging, get_logger; print('OK')"
python -c "from src.utils.stage_stats import track_stage, StageStats; print('OK')"
python -c "from src.config import load_params; p = load_params(); print('seed:', p['seed'])"

# Logbook tools
python scripts/build_logbook_index.py
# Expected: [OK] INDEX.md rebuilt - 1 entries -> ...

python scripts/check_logbook.py
# Expected: [OK] Logbook check passed ...

# Tests (requires: pip install -r requirements.txt)
pytest tests/test_stage_stats.py -v
# Expected: all tests pass (no failures)
```

**Tests added:** `tests/test_stage_stats.py` — 11 test functions covering:
- `StageStats.drop()` accumulation
- `record_input_frame()` / `record_output_frame()` null% calculation
- `cols_added` / `cols_removed` diff
- `to_dict()` structure
- `_null_pct()` edge cases (empty, no nulls, all nulls, partial nulls)
- `track_stage` integration: writes `_stats.json`, appends `data_change_log.md`, no-exception on empty
- `logging_utils.setup_logging()`: creates JSONL file, idempotent

## 8. Next step
[HUMAN] Register for OpenAQ API key and create `.env`; verify AWS anonymous access; then run `src/data/discover_stations.py` (Phase 1, Step 002).
