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

---
_Add new rows above this line._
