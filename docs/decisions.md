# Design Decision Register

> ADR-style record of all design decisions made during the project.
> Each entry links to the logbook step that generated it.
> Append new entries; never delete or overwrite existing ones.

---

## Decision Register

| ID | Date | Decision | Alternatives considered | Rationale | Step link |
|---|---|---|---|---|---|
| D-001 | 2026-10-14 | Use OpenAQ archive (S3) as primary pollutant source instead of Kaggle | Kaggle CPCB dataset (static, older); direct CPCB portal download | OpenAQ is large, recent (2019–2026), and provides anonymous S3 access; no manual download needed | [Step 001](logbook/001-repo-scaffold-and-logging-scaffold.md) |
| D-002 | 2026-10-14 | Compute AQI from raw concentrations using CPCB method | Use pre-computed AQI if available | Raw concentrations allow us to verify breakpoints, units, and validity rules; pre-computed AQI sources are unverifiable | [Step 001](logbook/001-repo-scaffold-and-logging-scaffold.md) |
| D-003 | 2026-10-14 | Use SQLite backend for MLflow (`sqlite:///mlflow.db`) | Remote PostgreSQL; DagsHub MLflow | Simplest setup for a two-person student project; no external services needed; easy to reset | [Step 001](logbook/001-repo-scaffold-and-logging-scaffold.md) |
| D-004 | 2026-10-14 | Chronological train/val/test split (train≤2023, val 2024, test 2025+) | Random split; walk-forward CV only | Time-series forecasting requires no future leakage; chronological split mimics real deployment | [Step 001](logbook/001-repo-scaffold-and-logging-scaffold.md) |
| D-005 | 2026-10-15 | Select 24 stations with PM2.5/PM10 and ≥ 3 years active span across target cities | Keep all 27 discovered stations unconditionally | Minimizes missing data and ensures robust station-day panel forecasting with particulate target validity | [Step 002](logbook/002-station-discovery-and-selection.md) |
| D-006 | 2026-10-15 | Anonymous unsigned S3 sync with SHA256 manifests and yearly Parquet weather caching | Single monolithic download | Provides resumable transfers, skip-on-cached semantics, checksum audit trails, and zero credential dependencies | [Step 003](logbook/003-data-acquisition-openaq-and-weather.md) |
| D-007 | 2026-10-15 | Enforce strict schema validation gates and sentinel missing-value flagging | Silent dropping | Halts pipeline on unparseable datetimes or missing columns, while logging warnings on physical range limits and sentinels (-999) | [Step 004](logbook/004-data-validation-and-data-card.md) |
| D-008 | 2026-10-15 | Use local filesystem remote (`../dvc_remote`) for DVC data storage | Cloud S3 / DagsHub remote | Zero external credentials or billing required; fast local caching for offline reproducibility and graded submission | [Step 005](logbook/005-dvc-dataset-versioning.md) |
| D-009 | 2026-10-15 | Drop NH3, NOx, NO columns from cleaned panel (>70% missing) | Impute with zeros; retain as high-missing indicators | All three exceed the `params.yaml:clean.drop_cols_over_missing=0.7` threshold; retaining them would introduce mostly-imputed noise. Flagged in EDA Fig 03. | [Step 006](logbook/006-hourly-to-daily-aqi-and-eda.md) |
| D-010 | 2026-10-15 | Use `class_weight='balanced'` for classifiers; report macro-F1 alongside accuracy | Accuracy-only; majority-class baseline | EDA Fig 06 shows severe class imbalance (Moderate dominant, Good <5%). Accuracy alone is misleading; macro-F1 penalises neglect of minority classes. | [Step 006](logbook/006-hourly-to-daily-aqi-and-eda.md) |
| D-011 | 2026-10-15 | Winsorise at 0.5–99.5th percentile quantiles fit exclusively on training data | Global winsorisation; no winsorisation | Extreme outliers in SO2, CO distort tree splits and scaler ranges. Fitting on train only prevents test-set leakage of distributional info. | [Step 007](logbook/007-feature-engineering-and-split.md) |
| D-012 | 2026-10-15 | Include lags [1, 2, 3, 7] for AQI and PM2.5 based on ACF/PACF analysis | Lags [1, 3, 7] or longer lags | ACF/PACF (Fig 12) confirms significant autocorrelation specifically at lags 1, 2, 3, and 7 days. Lag 7 captures weekly patterns. | [Step 007](logbook/007-feature-engineering-and-split.md) |
| D-013 | 2026-10-15 | Weather merge restricts modelling panel to stations with matched weather files (Delhi) | Discard weather; use all stations without weather | Temporal scope was limited by available weather file coverage; this is flagged as a limitation. Future work will expand weather to all 8 cities. | [Step 007](logbook/007-feature-engineering-and-split.md) |
| D-014 | 2026-10-15 | Dual evaluation strategy for Task B (Direct Multi-class Classifiers and Regress-Then-Bucket) | Only train direct classification models | Regress-Then-Bucket leverages continuous AQI predictions and achieves 97.97% within-one-category accuracy, while direct Random Forest Classifier achieves 0.6077 Macro-F1. | [Step 008](logbook/008-baseline-models-evaluation-and-mlflow-tracking.md) |
| D-015 | 2026-10-15 | Prioritize `import lightgbm` before pandas/pyarrow on Windows environments | Disable LightGBM; reinstall system C++ runtimes | Resolves an OpenMP C-runtime access violation caused by PyArrow's threading DLL colliding with LightGBM's C-API on Windows. | [Step 008](logbook/008-baseline-models-evaluation-and-mlflow-tracking.md) |
| D-016 | 2026-10-15 | Use project-local `.venv` directory for Python virtual environment | Windows Store Python global site-packages | Avoids Windows 260-character `MAX_PATH` file path exhaustion during package extraction (e.g. MLflow DB migration scripts). | [Step 008](logbook/008-baseline-models-evaluation-and-mlflow-tracking.md) |
| D-017 | 2026-10-15 | Retain full multimodal feature space (Lags + Weather + Spatial + Calendar) based on ablation evidence | Lag-only model; discard weather due to API latency | Dropping lags degrades RMSE by +10.50% (worse than persistence), and dropping weather degrades RMSE by +3.30%. Both are vital for production accuracy. | [Step 009](logbook/009-feature-ablation-studies-and-mlflow-experiment-comparison.md) |

---
_Add new rows above this line._

