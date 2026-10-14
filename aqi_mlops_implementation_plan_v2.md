# AI-Based Air Quality Awareness Assistant — MLOps Milestone 1: Implementation Plan v2 (AI-Executable Prompt)

> **What changed from v1:**
> 1. **No Kaggle.** The dataset is now built from primary, public sources: the **OpenAQ archive on AWS S3** (hourly station data, India, recent years) enriched with **Open-Meteo historical weather (ERA5)**. The AQI target is computed by our own code from the official CPCB formula.
> 2. **Large and recent:** hourly station data 2019 → 2026 across many Indian cities, modelled as a **station-day panel** (hundreds of thousands of rows).
> 3. **Mandatory change-logging protocol:** every step must show what changed, log it automatically, and be documented in a logbook (Section 4). No phase counts as done until its logbook entry exists.
>
> **How to use:** paste this file into an AI coding assistant. Tell it to run **one phase at a time** and to finish each phase with the *Step-Close Protocol* (Section 4.4). Items marked **[HUMAN]** must be done by the students.

---

## 0. ROLE AND GROUND RULES FOR THE AI

You are a senior MLOps engineer helping two undergraduate students (course AI252TA, Machine Learning Operations) build Milestone 1 of **"AI-Based Air Quality Awareness Assistant"**.

1. One reproducible Git repo with the structure in Section 3. `dvc repro` must rebuild everything end to end.
2. Every script is a CLI module that reads settings from `params.yaml`; no hard-coded paths or magic numbers.
3. **No notebook-only logic.** Notebooks only for EDA; reusable logic lives in `src/`.
4. **No data leakage.** Chronological splits; imputers, scalers and clipping limits are fit on the training period only.
5. Global seed `42` in `params.yaml`.
6. Git tracks code, configs, docs, `.dvc` files and `dvc.lock`. DVC tracks data and models. Never commit raw data, `mlruns/`, `mlartifacts/`, or secrets.
7. Python 3.10+, type hints, docstrings, `logging` (not `print`), `pathlib`.
8. **Never invent facts about the data.** Inspect actual column names, units and counts, and write what you observe. If a source's schema differs from what this plan assumes, adapt the code, record it in `docs/decisions.md`, and continue.
9. **Show your work:** after every step print (a) what changed, (b) the exact commands to verify it, (c) the logbook entry path. See Section 4.
10. Figures: labelled axes with units, titles, 300 dpi PNG in `reports/figures/`.
11. Respect licenses and rate limits; put attribution text in `README.md` and `docs/data_card.md`.

---

## 1. PROJECT CONTEXT AND SCOPE

**Long-term vision:** a system that predicts air quality and turns it into plain-language health guidance.
**This milestone:** ML foundation + MLOps tooling only.

| # | Deliverable |
|---|---|
| 1 | Dataset acquisition and approval |
| 2 | Data exploration (EDA) |
| 3 | Feature engineering |
| 4 | Baseline models |
| 5 | Git repository |
| 6 | DVC dataset versioning |
| 7 | MLflow experiment tracking |
| 8 | Initial model evaluation |
| + | **Change log and documentation for every step (new)** |

**Out of scope now** (design so they can be added later): FastAPI service, Docker, CI/CD, drift monitoring, LLM advice layer, deployment.

**ML tasks (station-day panel):**
- **Task A, regression:** predict **next-day AQI** (0–500) for a monitoring station.
- **Task B, classification:** predict **next-day CPCB AQI category** (Good, Satisfactory, Moderate, Poor, Very Poor, Severe), derived from the Task A target.

**Team:** 2 students (declared exception to the course's standard group of 4).

---

## 2. DATA STRATEGY (REPLACES KAGGLE)

### 2.1 Sources

| Role | Source | Access | Notes |
|---|---|---|---|
| **Primary: pollutants** | **OpenAQ archive** on the AWS Open Data program, bucket `openaq-data-archive` (region `us-east-1`). Aggregates government and research-grade monitors, including Indian CPCB-sourced stations. | Anonymous S3 access with AWS CLI `--no-sign-request` (no AWS account needed). Narrow-format gzipped CSVs partitioned by location and date, e.g. `records/csv.gz/locationid=<id>/year=<yyyy>/month=<mm>/`. | Update frequency hourly. **License varies by data provider**, so record the license of each provider used in the data card. |
| **Primary helper: station discovery** | **OpenAQ API v3** (`/v3/locations`, `/v3/sensors`, `/v3/parameters`) | Free API key via OpenAQ Explore registration **[HUMAN]**; keep in `.env` as `OPENAQ_API_KEY`, never committed. | Used only to find Indian location IDs, coordinates, providers, and first/last-reading dates. |
| **Enrichment: weather** | **Open-Meteo Historical Weather API** (`https://archive-api.open-meteo.com/v1/archive`), ERA5-based reanalysis | No key. Free for non-commercial use within fair limits (about 10,000 calls/day). Data licensed CC BY 4.0, so attribute Open-Meteo. | One call per station per year-chunk; cache responses to disk. |
| **Fallback (if OpenAQ India coverage is too thin)** | **CPCB CAAQMS portal** (official; hourly/daily data downloadable per station) | Manual download with the portal's filters **[HUMAN]**; slow, many filters per query. | Place CSVs in `data/raw/cpcb_manual/` and write a loader with the same output schema. |

> **Why this is "large and recent":** hourly readings for several pollutants across tens to hundreds of Indian stations over ~7 years gives tens of millions of raw narrow rows, and a modelling table of several hundred thousand station-day rows.

> **Verify before trusting (the AI must do this and record results):** OpenAQ coverage differs by city and over time. Some stations stop reporting, some lack PM2.5, and Bengaluru has only about a dozen monitoring stations. The plan therefore **discovers coverage first**, then freezes the station list.

### 2.2 Study window and snapshot

- **Study window:** `2019-01-01` to a fixed `snapshot_end` date (set it to the last full month before you download, e.g., `2026-09-30`). Put both in `params.yaml`.
- **Freeze rule:** once downloaded, the raw snapshot is **never re-downloaded in place**. A refresh becomes a new DVC version (`data-v3`...). Record the snapshot date in the data card, because OpenAQ files can be patched retroactively.

### 2.3 Geographic scope (parameterised)
- Mandatory focus: **Bengaluru** (local relevance).
- Additional cities (default list, adjust to coverage found): Delhi, Mumbai, Chennai, Kolkata, Hyderabad, Pune, Ahmedabad, Lucknow.
- Selection rule: a station is kept only if it has ≥ 3 years of data in the window and ≥ 60% hourly completeness for **PM2.5 or PM10** over its active period. Parameterise thresholds.

### 2.4 Target definition: AQI computed by us (CPCB method)
OpenAQ provides raw concentrations, not AQI, so implement `src/features/aqi_utils.py` using the CPCB National AQI method:
- Sub-index per pollutant by linear interpolation within CPCB breakpoint bands; **AQI = max sub-index**.
- Averaging periods: PM2.5, PM10, NO2, SO2, NH3 → 24-hour mean; CO and O3 → 8-hour running mean (take the daily maximum).
- Validity rule: compute AQI for a station-day only if **≥ 3 pollutant sub-indices are available and at least one is PM2.5 or PM10**. Also require ≥ 16 valid hourly readings (≈ 75%) for a pollutant's daily value to count. Parameterise these.
- Category bands: 0–50 Good, 51–100 Satisfactory, 101–200 Moderate, 201–300 Poor, 301–400 Very Poor, 401–500 Severe.
- Store breakpoints in `configs/cpcb_breakpoints.yaml`. **[HUMAN + AI]** cross-check every breakpoint against the official CPCB AQI document and cite it in `docs/data_card.md`; the AI must not rely on memory alone. Unit tests must cover each band boundary.
- **Units:** inspect the `units` column. Convert everything to µg/m³ (CO to mg/m³ for the CPCB table) and document each conversion rule used (ppm/ppb conversions need temperature and pressure assumptions, so state them).

---

## 3. TECH STACK AND REPOSITORY STRUCTURE

| Concern | Tool |
|---|---|
| Language/env | Python 3.10+, `venv`, pinned `requirements.txt` |
| Data | pandas, numpy, pyarrow (parquet), `requests`, `awscli` or `boto3` (anonymous config), tqdm |
| Viz | matplotlib, seaborn |
| ML | scikit-learn, xgboost, lightgbm |
| Versioning | Git + GitHub, DVC |
| Tracking | MLflow (SQLite backend) |
| Quality | pytest, ruff, black, pre-commit |
| Docs/logging | Markdown logbook + auto-generated step reports (Section 4) |

```
aqi-awareness-assistant/
├── .dvc/
├── .github/
│   └── pull_request_template.md
├── configs/
│   └── cpcb_breakpoints.yaml
├── data/
│   ├── raw/
│   │   ├── openaq/                 # narrow CSV.GZ files as downloaded (DVC)
│   │   ├── weather/                # cached Open-Meteo JSON/CSV (DVC)
│   │   └── station_catalog.csv     # discovered stations + metadata (DVC)
│   ├── interim/                    # hourly -> daily, cleaned, merged (DVC outputs)
│   └── processed/                  # model-ready train/val/test (DVC outputs)
├── docs/
│   ├── CHANGELOG.md                # human-readable, one entry per step (Keep a Changelog style)
│   ├── logbook/
│   │   ├── INDEX.md                # auto-generated table of all entries
│   │   ├── _TEMPLATE.md
│   │   └── 001-<slug>.md ...       # one file per step
│   ├── data_change_log.md          # auto-appended before/after data statistics per transform
│   ├── dataset_approval.md
│   ├── data_card.md
│   ├── decisions.md                # design decision records (ADR-style)
│   ├── eda_summary.md
│   └── setup_howto.md
├── logs/                           # structured run logs (JSONL), git-ignored; summaries committed via reports/
├── models/                         # DVC outputs
├── notebooks/01_eda.ipynb
├── reports/
│   ├── figures/{eda,features,evaluation}/
│   ├── metrics/                    # JSON metrics + per-stage stats JSON (DVC metrics)
│   ├── step_reports/               # auto-generated snapshot per step (git diff stat, dvc status, etc.)
│   └── milestone1_report.md
├── scripts/
│   ├── log_step.py                 # creates a logbook entry + step report (Section 4)
│   ├── build_logbook_index.py
│   └── check_logbook.py            # fails if code changed without a logbook entry
├── src/
│   ├── config.py
│   ├── data/{discover_stations,download_openaq,download_weather,ingest,validate,hourly_to_daily,clean,merge}.py
│   ├── features/{aqi_utils,build_features}.py
│   ├── models/{baselines,train,evaluate,compare_runs}.py
│   └── utils/{logging_utils,stage_stats,mlflow_utils}.py
├── tests/{test_aqi_utils,test_units,test_features,test_no_leakage,test_stage_stats}.py
├── .env.example                    # OPENAQ_API_KEY=  (real .env is git-ignored)
├── .gitignore  .dvcignore  .pre-commit-config.yaml  pyproject.toml
├── Makefile                        # make step-close STEP=..., make repro, make test
├── dvc.yaml  params.yaml  requirements.txt  README.md
```

---

## 4. CHANGE-LOGGING AND DOCUMENTATION PROTOCOL (MANDATORY FOR EVERY STEP)

**Principle:** every change to code, data, parameters or models must be *visible* (shown in the terminal), *logged* (machine-readable record), and *documented* (human-readable explanation). The logbook is also the raw material for the final report.

### 4.1 Definition of a "step"
A step is one coherent unit of work that ends in a Git commit (or small group of commits), e.g., "discover stations", "download OpenAQ for Bengaluru", "implement AQI computation", "add rolling features", "train XGBoost baseline". Number steps sequentially (`001`, `002`, ...).

### 4.2 Logbook entry template (`docs/logbook/_TEMPLATE.md`)

```markdown
# Step <NNN> — <short title>
- **Date/time:** <ISO timestamp, IST>
- **Author:** <student>      **Reviewer:** <other student>
- **Phase:** <1–9>           **Branch / PR:** <branch> / #<n>
- **Git commit(s):** <hash(es)>      **Data version:** <git tag + dvc md5 of tracked inputs>
- **MLflow run IDs (if any):** <ids>

## 1. Objective
What this step is meant to achieve and why (1–3 sentences).

## 2. What changed
| Type | Path / object | Change summary |
|---|---|---|
| code | src/... | added / modified: ... |
| params | params.yaml | key: old → new |
| data | data/... | rows: A → B; columns added/removed |
| docs/config | ... | ... |

(Paste the `git diff --stat` and, for data, the before/after stats table that the pipeline stage emitted.)

## 3. Commands run (verbatim)
```bash
<exact commands>
```

## 4. Results / evidence
Key numbers, small tables, figure paths, screenshot paths. Quote real output, never invented numbers.

## 5. Decisions and rationale
Each decision, alternatives considered, why chosen (also copy to docs/decisions.md if it is a design decision).

## 6. Problems encountered and fixes
Errors, surprises, data issues, and how they were resolved.

## 7. Verification
How to reproduce/check this step (commands + expected output). Tests added/passing.

## 8. Next step
One line.
```

### 4.3 Automatic logging the AI must implement (Phase 2)

1. **`src/utils/logging_utils.py`:** configure `logging` to write human-readable output to the console and **JSON Lines** to `logs/<stage>_<timestamp>.jsonl` (fields: timestamp, stage, level, message, extra). Every `src/` script calls it first.
2. **`src/utils/stage_stats.py`:** a context manager `with track_stage("clean", inputs=[...], outputs=[...]) as st:` that records and writes `reports/metrics/<stage>_stats.json` containing: start/end time, duration, git commit, params hash, input/output file paths + sizes + row/column counts, **columns added/removed/changed**, null % before vs after per column, and any dropped-row counts **with reasons** (`st.drop(reason="negative PM2.5", n=...)`). Reuse it in *every* data-transforming stage.
3. **`docs/data_change_log.md` auto-append:** after each stage, append a dated section: stage, rows in → out, rows dropped by reason, columns added/removed, imputation counts, a one-line summary. This is the "show the changes made to the data" trail.
4. **`scripts/log_step.py --step <NNN> --title "<title>" --phase <n> [--mlflow-run <id>]`** which:
   - creates `docs/logbook/NNN-<slug>.md` from the template, pre-filled with timestamp, git branch, last commit hash, `git diff --stat HEAD~1` (or since last logged step), `git status --short`, `dvc status`, `dvc diff` (if data tags exist), the latest `reports/metrics/*_stats.json` summaries, and the list of changed files grouped by type;
   - writes a snapshot `reports/step_reports/NNN.json` (machine-readable);
   - prints the filled sections to the terminal so the students can **see** the changes, then leaves the narrative sections (objective, decisions, problems) for the human/AI to complete;
   - appends a line to `docs/CHANGELOG.md` under `## [Unreleased]` using Keep a Changelog headings (`Added`, `Changed`, `Fixed`, `Data`).
5. **`scripts/build_logbook_index.py`:** regenerates `docs/logbook/INDEX.md` (table: step, date, title, phase, commit, author, link).
6. **`scripts/check_logbook.py`** (run in pre-push hook and `make test`): fails if commits since the last logbook entry changed anything under `src/`, `params.yaml`, `dvc.yaml` or `configs/` without a new entry.
7. **Git integration:** `.github/pull_request_template.md` requires: logbook entry link, "what changed" summary, tests run, data version, MLflow run IDs. Commit messages use Conventional Commits and include the step number: `feat(step-014): compute CPCB sub-indices`.
8. **DVC integration:** every data-changing step saves `dvc diff` output into the logbook; data versions are Git-tagged (`data-v1`, `data-v2`...) with the tag message describing the change.
9. **MLflow integration:** each run is tagged `step_id`, `logbook_entry`, `git_commit`, `dvc_data_version`; the logbook entry file and `params.yaml` are logged as artifacts.

### 4.4 Step-Close Protocol (the AI runs this at the end of every step, then waits for confirmation)
1. Run tests: `pytest -q` and lint (`ruff`, `black --check`).
2. Show changes: `git status`, `git diff --stat`, `dvc status` and, if data changed, `dvc diff`.
3. Run `make step-close STEP=<NNN> TITLE="<title>"` (wraps `scripts/log_step.py`, index rebuild, logbook check).
4. Complete the narrative sections of the logbook entry from real outputs; copy design decisions to `docs/decisions.md`.
5. Commit (`feat(step-NNN): ...`, plus `docs(step-NNN): logbook`), push, `dvc push` if data changed.
6. Print a **step-close summary**: files changed, data rows/columns before→after, tests passing, entry path, commit hash, and the next step.

### 4.5 Evidence standards
- Quote real terminal output and real numbers only; if something fails, log the failure and the fix.
- Save screenshots to `reports/figures/screenshots/` with names like `step-014-dvc-diff.png`. **[HUMAN]** takes screenshots of the GitHub PR page, DVC version switching, and MLflow UI.
- Keep a running **issue/decision register** in `docs/decisions.md` (ID, date, decision, alternatives, rationale, step link).

---

## 5. PHASE 1 — DATASET DISCOVERY, ACQUISITION AND APPROVAL

### 5.1 Pre-approval discovery (small, before asking for approval)
1. **[HUMAN]** Register for a free OpenAQ API key; create `.env` from `.env.example`. Confirm the AWS CLI works anonymously: `aws s3 ls --no-sign-request s3://openaq-data-archive/` (region `us-east-1`).
2. `src/data/discover_stations.py`: call OpenAQ API v3 to list locations in India (paginate, respect rate limits, retry with backoff). For each location capture: location id, name, city/locality, provider, coordinates, sensors (parameter + units), first and last datetime. Save `data/raw/station_catalog.csv`.
3. Print coverage tables: stations per city, per provider, per pollutant; stations active in 2019–snapshot; stations with PM2.5/PM10. Save as `reports/metrics/station_coverage.json` and a figure.
4. Apply selection rules (Section 2.3) and write the **frozen station list** to `configs/stations_selected.csv` with the reason for any exclusion. Estimate download volume (files and approximate size) before downloading; record it.
5. Write `docs/dataset_approval.md` (one page): problem statement; sources, access method and **license per provider**; selected cities/stations; study window; expected raw size and modelling table size; target definition (CPCB AQI computed from raw pollutants); weather enrichment; why it suits the task; risks (missing data, station outages, unit inconsistencies, COVID-era behaviour is outside the 2019+ test period but 2020–21 appears in training); split strategy; ethical note (public environmental data, no personal data).
6. **[HUMAN]** Send for faculty approval; record date, approver, decision and comments in the same file, with a screenshot/email copy. If faculty rejects, switch to the CPCB-portal fallback and log the decision.

### 5.2 Download (after approval)
1. `src/data/download_openaq.py`: for each selected `location_id` and month in the window, copy from S3 anonymously (e.g., `aws s3 sync --no-sign-request` on `records/csv.gz/locationid=<id>/year=<yyyy>/month=<mm>/` or boto3 with unsigned config) into `data/raw/openaq/`. Requirements: resumable, parallel (bounded workers), skips files already present, writes a **download manifest** (`data/raw/openaq_manifest.csv`: path, size, sha256, download time) and logs counts of missing partitions.
2. `src/data/download_weather.py`: for each station coordinate call the Open-Meteo archive API in yearly chunks for **hourly** variables: `temperature_2m, relative_humidity_2m, dew_point_2m, surface_pressure, precipitation, wind_speed_10m, wind_direction_10m, boundary_layer_height` (confirm each variable name against the API docs; drop any unsupported). Use `timezone=Asia/Kolkata`, cache each response, throttle requests, and record the model/source in the manifest. Save to `data/raw/weather/`.
3. `src/data/ingest.py` + `src/data/validate.py`: schema checks on both sources:
   - Required columns present (record the actual OpenAQ column names found); datetime parses with timezone; no future timestamps; `parameter` ∈ expected set; `units` recorded per parameter and **unit inconsistencies reported**; values numeric; negative or sentinel values (e.g., -999) counted; duplicates on (location, sensor, datetime) counted.
   - Output `reports/metrics/data_validation.json` and hard-fail on schema breakage.
4. Write `docs/data_card.md`: sources + URLs, **download/snapshot date**, license per provider, window, station count, raw file count/size, row counts, parameters and units, known limitations (outages, sensor drift, provider differences, no emissions or traffic data, ERA5 is gridded reanalysis not station weather), ethical note, intended and non-intended use (not for medical decisions), attribution text (OpenAQ and data providers; Open-Meteo CC BY 4.0).

### 5.3 Step-close
Run the Step-Close Protocol for each sub-step (discovery, approval doc, OpenAQ download, weather download, validation).

**Acceptance:** station catalog + frozen station list exist; approval documented; raw data downloaded with manifests; validation passes; data card complete; logbook entries 001–00N exist.

---

## 6. PHASE 2 — GIT REPOSITORY (AND LOGGING SCAFFOLD)

1. **[HUMAN]** Create the GitHub repo `aqi-awareness-assistant` and add the teammate. `git init`, branch `main`, add remote.
2. Create the full structure (Section 3), `.gitignore` (`__pycache__/`, `.venv/`, `.ipynb_checkpoints/`, `mlruns/`, `mlartifacts/`, `mlflow.db`, `.env`, `logs/`, `data/raw/*`, `data/interim/*`, `data/processed/*`, `models/*`, keeping `.dvc` files), `.env.example`, `requirements.txt` (pinned), `pyproject.toml`, `Makefile`, `.pre-commit-config.yaml` (ruff, black, trailing whitespace, end-of-file, check-added-large-files at 500 KB, local hook running `scripts/check_logbook.py` on pre-push).
3. **Implement the logging scaffold first** (Section 4.3: `logging_utils`, `stage_stats`, `log_step.py`, `build_logbook_index.py`, `check_logbook.py`, templates, PR template, `Makefile` targets) with tests (`tests/test_stage_stats.py`). Log this as **Step 001**.
4. README skeleton, `docs/setup_howto.md` (how to create the env, set the API key, run each stage).
5. **Branching:** `main` always runnable; `feature/<name>` per phase; PRs reviewed by the other student; squash or merge with Conventional Commit messages containing step numbers. Aim for 25+ meaningful commits across both authors.
6. **Note on ordering:** Phase 2's scaffold should be built **before** the Phase 1 downloads so those steps are logged properly. Do Phase 2 steps 1–4 first, then Phase 1.

**Acceptance:** repo pushed with both contributors; pre-commit and logbook check work; Step 001 logbook entry exists; at least 4 merged PRs by milestone end; tag `v0.1-milestone1` at the end.

---

## 7. PHASE 3 — DVC DATASET VERSIONING

### 7.1 Setup
1. `dvc init`; commit.
2. Remote: (a) local/shared folder remote (always works), or (b) DagsHub / Google Drive remote. Credentials via `dvc remote modify --local`; document the choice in `decisions.md`.
3. Track raw data directories: `dvc add data/raw/openaq data/raw/weather data/raw/station_catalog.csv`; commit `.dvc` files; `dvc push`. Tag `data-v1` with a message describing the snapshot (window, stations, date).

### 7.2 Demonstrate genuine versioning (graded evidence)
- **data-v2** must be a *real, documented change*, for example: add more cities/stations or extend the window after a refresh, or apply a documented fix to the station list after EDA reveals bad stations. Run `dvc add`, commit, tag, `dvc push`.
- Show `dvc diff data-v1 data-v2`, then `git checkout data-v1 && dvc checkout` and back. Save outputs into the logbook and screenshots.
- Record in the data card a **dataset version table** (tag, date, what changed, size, file count, md5 of the `.dvc`).

### 7.3 Pipeline (`dvc.yaml`) stages

```yaml
stages:
  validate:
    cmd: python -m src.data.ingest
    deps: [src/data/ingest.py, src/data/validate.py, data/raw/openaq, data/raw/weather, configs/stations_selected.csv]
    params: [data]
    metrics: [reports/metrics/data_validation.json: {cache: false}]
  hourly_to_daily:
    cmd: python -m src.data.hourly_to_daily
    deps: [src/data/hourly_to_daily.py, src/utils/stage_stats.py, data/raw/openaq, configs/cpcb_breakpoints.yaml, src/features/aqi_utils.py]
    params: [units, aggregation, aqi]
    outs: [data/interim/pollutants_daily.parquet]
    metrics: [reports/metrics/hourly_to_daily_stats.json: {cache: false}]
  merge_weather:
    cmd: python -m src.data.merge
    deps: [src/data/merge.py, data/interim/pollutants_daily.parquet, data/raw/weather]
    params: [weather]
    outs: [data/interim/station_day.parquet]
    metrics: [reports/metrics/merge_stats.json: {cache: false}]
  clean:
    cmd: python -m src.data.clean
    deps: [src/data/clean.py, data/interim/station_day.parquet]
    params: [clean, split, seed]
    outs: [data/interim/station_day_clean.parquet]
    metrics: [reports/metrics/clean_stats.json: {cache: false}]
  features:
    cmd: python -m src.features.build_features
    deps: [src/features/build_features.py, src/features/aqi_utils.py, data/interim/station_day_clean.parquet]
    params: [features, split, seed]
    outs: [data/processed/train.parquet, data/processed/val.parquet, data/processed/test.parquet, data/processed/feature_list.json]
    metrics: [reports/metrics/features_stats.json: {cache: false}]
  train:
    cmd: python -m src.models.train
    deps: [src/models/train.py, src/models/baselines.py, data/processed/train.parquet, data/processed/val.parquet]
    params: [train, seed]
    outs: [models/]
    metrics: [reports/metrics/val_metrics.json: {cache: false}]
  evaluate:
    cmd: python -m src.models.evaluate
    deps: [src/models/evaluate.py, models/, data/processed/test.parquet]
    outs: [reports/figures/evaluation/]
    metrics: [reports/metrics/test_metrics.json: {cache: false}]
```

Run `dvc repro`, `dvc dag`, commit `dvc.yaml` + `dvc.lock`. Demonstrate that changing one parameter reruns only affected stages and capture the output in the logbook.

### 7.4 `params.yaml`

```yaml
seed: 42
data:
  window_start: "2019-01-01"
  snapshot_end: "2026-09-30"        # set to the real freeze date
  cities: [Bengaluru, Delhi, Mumbai, Chennai, Kolkata, Hyderabad, Pune, Ahmedabad, Lucknow]
  min_years_per_station: 3
  min_hourly_completeness: 0.60
units:
  target_unit_ugm3: true
  co_unit: mg/m3
aggregation:
  min_hours_per_day: 16             # ~75% of hours required for a valid daily value
  rolling_8h_pollutants: [co, o3]
aqi:
  min_subindices: 3
  require_pm: true
weather:
  variables: [temperature_2m, relative_humidity_2m, surface_pressure, precipitation, wind_speed_10m, wind_direction_10m]
clean:
  max_missing_frac_row: 0.5
  ffill_limit_days: 2
  winsor_lower_q: 0.005
  winsor_upper_q: 0.995
  drop_cols_over_missing: 0.7
features:
  horizon_days: 1
  lags: [1, 2, 3, 7]
  rolling_windows: [3, 7, 14]
  use_calendar: true
  use_station_embedding: city_onehot  # options: city_onehot | station_code
split:
  train_end: "2023-12-31"
  val_end: "2024-12-31"             # test = 2025-01-01 .. snapshot_end
train:
  models: [persistence, seasonal_naive, city_month_mean, ridge, random_forest, xgboost, lightgbm]
  cv_folds: 4
```

(Adjust split dates to the data actually obtained so that train ≥ ~60% of rows, val and test each ≥ ~12 months. Record any change as a decision.)

**Acceptance:** remote configured; `data-v1` and `data-v2` tagged and demonstrated; `dvc repro` runs clean from a fresh clone; `dvc dag` captured.

---

## 8. PHASE 4 — DATA PROCESSING AND EXPLORATION (EDA)

### 8.1 Hourly → daily panel (`hourly_to_daily.py`)
1. Read narrow OpenAQ files; pivot to wide per (station, hour); standardise parameter names (`pm25, pm10, no2, so2, co, o3, nh3, nox, no`) and units (apply tested conversions).
2. Convert timestamps to Asia/Kolkata; define the day as local calendar day.
3. Daily aggregation: mean for 24h pollutants; running 8h means → daily max for CO and O3; require `min_hours_per_day`; also store `n_hours_<pollutant>`.
4. Compute CPCB sub-indices and **AQI + AQI_category** per station-day.
5. Emit stage stats: raw hourly rows, rows dropped by reason (invalid values, unit failures, duplicates), station-days produced, AQI-valid station-days.

### 8.2 EDA notebook (`notebooks/01_eda.ipynb`, ≥ 14 figures to `reports/figures/eda/`)
Each section ends with a **Finding → Decision** cell, copied into `docs/eda_summary.md` and `docs/decisions.md`.

1. **Coverage:** stations per city; date span per station (Gantt-style plot); rows per city/year; station-days with valid AQI.
2. **Missingness:** per pollutant; station × month heatmap of missing fraction; stations that stop reporting; which pollutants drive AQI validity.
3. **Distributions:** histograms (log scale where skewed) per pollutant; boxplots per city; implausible values list; AQI category class distribution overall and per city (imbalance).
4. **Temporal patterns:** AQI time series for Bengaluru and 4–5 other cities; month × city seasonality heatmap; day-of-week effect; annual trend; note festival periods (e.g., Diwali) and monsoon effects if visible.
5. **Weather relationships:** PM2.5 vs wind speed, humidity, precipitation (scatter + binned means); boundary layer height effect if included.
6. **Correlation:** Pearson and Spearman among pollutants, weather and AQI; ACF/PACF of AQI for several stations (justifies lag features).
7. **Dominant pollutant analysis:** share of days each pollutant sets the AQI, per city.
8. **Consistency checks:** AQI vs category mapping; city-mean vs station values; stations that are outliers relative to neighbours (candidate removal, which would motivate `data-v2`).
9. **Next-day difficulty:** distribution of day-over-day AQI change; persistence-baseline preview.
10. **Bengaluru spotlight:** station-level seasonal profile and data availability.

**Acceptance:** notebook runs top-to-bottom; figures saved; `docs/eda_summary.md` lists 8–10 findings with decisions; logbook entries for processing and for EDA.

---

## 9. PHASE 5 — FEATURE ENGINEERING

All features use only information available at day *t* to predict day *t+1*, computed **per station**.

### 9.1 Cleaning (`clean.py`)
1. Sort by (station, date); drop duplicate station-days.
2. Drop station-days with missing AQI (target undefined). Log the count.
3. Replace impossible values with NaN; **winsorise using training-period quantiles only**.
4. Drop columns above `drop_cols_over_missing`.
5. Imputation: forward-fill within station up to `ffill_limit_days`; then fill with the **station's training-period median**; add `<col>_was_missing` flags for main pollutants and weather.
6. Emit full stage stats and `data_change_log.md` entry (counts per reason).

### 9.2 Feature set (`build_features.py`)
1. **Target:** `target_aqi = AQI.shift(-1)` per station (only if the next calendar day exists; do not shift across date gaps); `target_category = aqi_to_bucket(target_aqi)`.
2. **Lags** of AQI, PM2.5, PM10, NO2, O3, CO, SO2 for `[1,2,3,7]` days.
3. **Rolling mean/std/max** over `[3,7,14]` days for AQI and PM2.5 (shifted so no future leakage).
4. **Differences:** `AQI_diff1`, `PM25_diff1`.
5. **Weather:** same-day weather, previous-day weather, 3-day means; **optionally the next-day weather** as a feature **only if documented as using a weather forecast proxy**; by default use only same-day and past weather to avoid leaking future information.
6. **Calendar:** day of week, month, day of year, weekend flag, sin/cos encodings, season (winter Nov–Feb, summer Mar–May, monsoon Jun–Sep, post-monsoon Oct), and a festival-period flag (list dates in `configs/festival_dates.yaml` with a cited source).
7. **Ratios:** PM2.5/PM10 (guard zero).
8. **Spatial context:** city one-hot (linear) and station code (trees); optional **city-mean AQI of other stations** at day *t* (excluding the station itself).
9. **Chronological split** per `params.yaml`; assert max(train) < min(val) < min(test); print sizes and class balance.
10. **Scaling:** `StandardScaler` fit on train only → `models/scaler.joblib`.
11. Save parquet splits and `feature_list.json`; produce a train-only feature ranking figure.

### 9.3 Tests
Lags and rolling windows do not cross stations or dates gaps; no future info in features; split ordering; no NaN in matrices; `target_category` consistency; unit-conversion and AQI band-boundary tests.

**Acceptance:** `dvc repro features` produces three splits; `pytest` passes; each transformation logged in `data_change_log.md`; logbook entries exist.

---

## 10. PHASE 6 — BASELINE MODELS

**Naive baselines:** persistence (tomorrow = today); weekly seasonal naive (state the exact lag used in the docstring); station-month mean from training; majority class.

**Regression (Task A):** Ridge (alpha grid via `TimeSeriesSplit`), RandomForestRegressor, XGBRegressor (early stopping on validation), LGBMRegressor.

**Classification (Task B):** LogisticRegression (balanced), RandomForestClassifier (`balanced_subsample`), XGBClassifier or LGBMClassifier with class/sample weights; also "regress-then-bucket" from the best regressor.

**`train.py`:** seeds; load splits; for each model fit, predict on validation, compute metrics (Section 12), log to MLflow (Section 11), save `models/<name>.joblib`; write `val_metrics.json`; print a leaderboard. Use `TimeSeriesSplit` (never shuffled K-fold). Keep tuning minimal; heavy tuning is a later milestone. Wrap the run in `track_stage("train")` so the logbook records models trained, time per model, and validation metrics.

**Acceptance:** all baselines trained; models saved; if ML models do not beat persistence, report honestly and investigate (autocorrelation, missing weather, data issues).

---

## 11. PHASE 7 — MLFLOW EXPERIMENT TRACKING

1. Tracking URI `sqlite:///mlflow.db`; artifacts `./mlartifacts`; experiments `aqi-regression-baselines` and `aqi-classification-baselines`. Launch UI with `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`.
2. **Per run:**
   - Tags: `task`, `model_family`, `git_commit`, `dvc_data_version` (git tag + md5 from the `.dvc` files), `author`, `stage=baseline`, **`step_id`, `logbook_entry`**.
   - Params: all hyperparameters, seed, window and split dates, horizon, lags, rolling windows, number of features, train/val sizes, number of stations and cities.
   - Metrics: validation metrics, `cv_rmse_mean/std`, train time.
   - Artifacts: model with signature + input example, `params.yaml`, `feature_list.json`, the logbook entry file, importance plot, predicted-vs-actual, residuals, confusion matrix, classification report.
3. **Comparison:** `src/models/compare_runs.py` → `reports/metrics/run_comparison.csv`; MLflow UI compare screenshots **[HUMAN]**.
4. **Required experiments:** ≥ 12 runs; at least two parameter variations for one model; **ablations**: (a) with vs without lag features, (b) **with vs without weather features** (this shows the value of the enrichment), (c) with vs without spatial context features. Each ablation is a logged step with a logbook entry.
5. Optional: register the best regressor as `aqi-regressor` with alias `candidate`.

**Acceptance:** ≥ 12 runs with full params/metrics/artifacts/tags; comparison CSV; screenshots; each run traceable to a logbook entry and commit.

---

## 12. PHASE 8 — INITIAL MODEL EVALUATION

Touch the **test set once**, at the end, for the chosen models and all naive baselines.

**Regression metrics:** MAE, RMSE, R², median AE, within ±20 AQI points rate, skill score vs persistence (`1 − RMSE/RMSE_persistence`).
**Classification metrics:** accuracy, macro-F1, weighted-F1, balanced accuracy, Cohen's kappa, per-class precision/recall/F1, **within-one-category accuracy**, and **recall for Poor+Very Poor+Severe** (missing a dangerous day is the costly error).

**Analyses (figures to `reports/figures/evaluation/`):**
1. Full leaderboard (validation and test).
2. Predicted vs actual, residual histogram, residuals over time, normalised confusion matrix, per-class F1.
3. **Error breakdown** by city, station, season/month, AQI level, weekday vs weekend, and by data completeness of the station.
4. Weather-ablation result and feature importance (permutation importance on validation).
5. **Bengaluru spotlight:** predicted vs actual time series and metrics.
6. Validation-vs-test shift: discuss any distribution change over time (e.g., station network changes, post-2023 trends).
7. **Limitations and next steps:** station outages, no emissions/traffic/fire data, ERA5 resolution, 1-day horizon only, class imbalance; future: tuning, multi-day horizon, forecast weather inputs, API, Docker, CI/CD, monitoring, LLM advice layer.

Wrap in `track_stage("evaluate")`, write `test_metrics.json`, and log the step.

**Acceptance:** `test_metrics.json` and all plots exist; results reproducible with `dvc repro evaluate`; honest discussion of limits.

---

## 13. PHASE 9 — DOCUMENTATION, REPRODUCIBILITY, SUBMISSION

1. **README:** summary; problem; data sources with licenses and attribution; Mermaid pipeline diagram (OpenAQ + Open-Meteo → validate → daily panel → merge → clean → features → train → evaluate, annotated with Git/DVC/MLflow/logbook); setup (`venv`, `pip install`, `.env`, `dvc pull`, `dvc repro`, MLflow UI, `make test`); results table; repo structure; how the logbook works; team contributions; limitations.
2. **Fresh-clone test (mandatory):** clone → venv → install → `dvc pull` → `dvc repro` → `pytest` → `python scripts/check_logbook.py`. Log fixes.
3. **`reports/milestone1_report.md`** assembled from the logbook and step reports: 1 Introduction; 2 Dataset and approval; 3 Data processing and EDA; 4 Feature engineering; 5 Baselines; 6 Git workflow (commit graph, PRs); 7 DVC (dag, diff, version switch); 8 MLflow (experiments, comparison); 9 Evaluation; 10 **Change log summary** (table from `INDEX.md`); 11 Limitations/future work; 12 Team contributions; 13 References and attributions.
4. 5-minute demo script: repo → logbook index → `dvc dag` → switch data version → `dvc repro` → MLflow comparison → test results → one error-analysis insight.
5. Finalise `CHANGELOG.md` (`[0.1.0]`), tag `v0.1-milestone1`, push tags, `dvc push`.

---

## 14. TWO-PERSON SPLIT

| Area | Student A | Student B |
|---|---|---|
| Logging scaffold + repo | logging utils, `stage_stats`, Makefile | `log_step.py`, PR template, pre-commit hooks |
| Phase 1 data | station discovery, OpenAQ download | weather download, validation, data card, approval doc |
| Phase 3 DVC | remote, data-v1 | data-v2, `dvc.yaml`, version demo |
| Phase 4 | hourly→daily + AQI computation + tests | EDA notebook |
| Phase 5 | cleaning + imputation | feature builder + split + leakage tests |
| Phase 6–7 | naive, linear, RF + MLflow utils | boosting, classifiers, ablations, comparison |
| Phase 8–9 | metrics, error breakdown | Bengaluru spotlight, report, fresh-clone test |

Each student reviews the other's PRs and logbook entries.

---

## 15. TIMELINE (~3–4 weeks; compress as needed)

| Days | Work |
|---|---|
| 1–2 | Phase 2: repo + logging scaffold (Step 001) |
| 3–5 | Phase 1: station discovery, approval request sent, coverage analysis |
| 6–8 | Phase 1/3: downloads, validation, data card, DVC data-v1 |
| 9–11 | Phase 4: daily panel, AQI computation, EDA |
| 12–14 | Phase 3: data-v2 (informed by EDA); Phase 5 features and tests |
| 15–18 | Phase 6–7: baselines, MLflow, ablations |
| 19–21 | Phase 8: evaluation |
| 22–25 | Phase 9: README, report, fresh-clone test, demo rehearsal |

Downloads can take long; start them early and run in the background while building later steps on a small subset (parameterise `cities`).

---

## 16. DEFINITION OF DONE

- [ ] Dataset approved; OpenAQ + Open-Meteo sources documented with licenses and attribution; data card complete
- [ ] Raw snapshot frozen, manifests with checksums, `data-v1` and `data-v2` tagged and demonstrated
- [ ] AQI computed with CPCB method, breakpoints verified against the official document, boundary tests pass
- [ ] `dvc repro` works from a fresh clone
- [ ] EDA with ≥ 14 figures and `eda_summary.md`
- [ ] Leakage-free features; chronological split; tests pass
- [ ] ≥ 4 regression and ≥ 3 classification baselines, ≥ 12 MLflow runs, ablations including weather
- [ ] Test-set evaluation with error breakdown, Bengaluru spotlight, honest limitations
- [ ] **Logbook entry for every step; `INDEX.md`, `CHANGELOG.md`, `data_change_log.md` complete; `check_logbook.py` passes**
- [ ] README, report, demo script done; tag `v0.1-milestone1`

---

## 17. PITFALLS THE AI MUST AVOID

1. Random train/test splits on time series, or computing lags/rolling features across stations or date gaps.
2. Fitting imputers, scalers or clipping limits on the full data.
3. Assuming the OpenAQ schema, units or station coverage without inspecting them.
4. Mixing units (ppb vs µg/m³ vs mg/m³) or time zones (UTC vs IST) when aggregating.
5. Computing AQI without the CPCB validity rule, or using non-verified breakpoints.
6. Re-downloading raw data in place (breaks the frozen snapshot) or committing data, keys or `.env`.
7. Using next-day weather as a feature without documenting that it implies forecast inputs.
8. Reporting accuracy only on imbalanced classes; tuning on the test set.
9. **Making a change without logging it** (no logbook entry, no stage stats, no data change log).
10. Overstating results; if models barely beat persistence, say so and explain why.

---

## 18. FIRST INSTRUCTION TO THE AI

Start with **Phase 2, steps 1–4**: create the repository structure from Section 3, config files, `.env.example`, `Makefile`, pre-commit config, and the **complete logging scaffold** (`logging_utils.py`, `stage_stats.py`, `log_step.py`, `build_logbook_index.py`, `check_logbook.py`, logbook template, PR template, CHANGELOG, tests). Run the Step-Close Protocol to produce **logbook entry 001**, show me the generated entry and the terminal output of what changed, and then stop and wait for my confirmation before starting Phase 1.
