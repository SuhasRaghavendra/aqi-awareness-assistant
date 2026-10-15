# Changelog

All notable changes to this project will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
adhering to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
and include the step number, e.g. `feat(step-008): baseline models and MLflow tracking`.

## [Unreleased]

<!-- Future steps for Milestone 2 will be appended here automatically -->

---

## [0.1.0] - 2026-10-15 — Milestone 1 Release

### Added
- **Step 010** (Phase 9): Milestone 1 synthesis report and release tagging — [010-milestone-1-synthesis-report-and-release-tagging.md](logbook/010-milestone-1-synthesis-report-and-release-tagging.md) — 2026-10-15
- **Step 009** (Phase 7): Feature ablation studies and MLflow experiment comparison — [009-feature-ablation-studies-and-mlflow-experiment-comparison.md](logbook/009-feature-ablation-studies-and-mlflow-experiment-comparison.md) — 2026-10-15
- **Step 008** (Phase 6 & 8): Baseline models evaluation, out-of-sample test evaluation, and MLflow tracking — [008-baseline-models-evaluation-and-mlflow-tracking.md](logbook/008-baseline-models-evaluation-and-mlflow-tracking.md) — 2026-10-15
- **Step 007** (Phase 5): Feature engineering, winsorisation, imputation, and chronological split — [007-feature-engineering-and-split.md](logbook/007-feature-engineering-and-split.md) — 2026-10-15
- **Step 006** (Phase 4): Hourly to daily panel, CPCB AQI sub-index calculation, and 15-figure EDA notebook — [006-hourly-to-daily-aqi-and-eda.md](logbook/006-hourly-to-daily-aqi-and-eda.md) — 2026-10-15
- **Step 005** (Phase 3): DVC dataset versioning (`data-v1`, `data-v2`), local remote setup, and pipeline DAG — [005-dvc-dataset-versioning.md](logbook/005-dvc-dataset-versioning.md) — 2026-10-15
- **Step 004** (Phase 1): Data validation rules, sentinel handling, and formal data card — [004-data-validation-and-data-card.md](logbook/004-data-validation-and-data-card.md) — 2026-10-15
- **Step 003** (Phase 1): Data acquisition modules for OpenAQ S3 public archive and Open-Meteo ERA5 weather — [003-data-acquisition-openaq-and-weather.md](logbook/003-data-acquisition-openaq-and-weather.md) — 2026-10-15
- **Step 002** (Phase 1): Indian station discovery, selection rules, and coverage metrics — [002-station-discovery-and-selection.md](logbook/002-station-discovery-and-selection.md) — 2026-10-15
- **Step 001** (Phase 2): Repository scaffold, logging utilities, context-managed stage statistics, and Makefile — [001-repo-scaffold-and-logging-scaffold.md](logbook/001-repo-scaffold-and-logging-scaffold.md) — 2026-10-14

### Data
- Frozen 24-station regulatory monitoring dataset across 8 major Indian cities spanning 2019 to 2026.
- 13,763 station-days aggregated with 9,565 valid CPCB AQI records.
- 129 multimodal engineered features across chronological train (3,023 rows), val (2,998 rows), and test (2,331 rows) splits.

### Models & Results
- **Task A (Regression):** Ridge Regression achieves Test RMSE **44.05** (R²: 0.6607), outperforming Persistence (**47.22**, R²: 0.6101) with **+6.72% skill score**.
- **Task B (Classification):** Random Forest Classifier achieves **71.30% Accuracy** (Macro-F1: 0.5581), with Regress-Then-Bucket achieving **97.21% within-one-category accuracy**.
- **MLflow Tracking:** 22 tracked runs in `sqlite:///mlflow.db` across `aqi-regression-baselines`, `aqi-classification-baselines`, and `aqi-ablations`.
