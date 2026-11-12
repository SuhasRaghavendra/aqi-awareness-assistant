# AI-Based Air Quality Awareness Assistant — MLOps Milestone 1

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![DVC Tracked](https://img.shields.io/badge/DVC-tracked-white.svg?logo=dvc)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.13.0-0194E2.svg?logo=mlflow)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A production-grade, leak-free, reproducible MLOps pipeline for forecasting Next-Day Air Quality Index (AQI) and CPCB Health Risk Categories across Indian metropolitan areas, enriched with ECMWF ERA5 meteorological reanalysis.

---

## 1. Problem Statement & Motivation

Air pollution in Indian cities is a leading environmental risk factor for cardiopulmonary disease and mortality. Regulatory monitoring stations report ambient pollutant concentrations retrospectively, leaving healthcare facilities, municipal administrators, and citizens reactive.

This project delivers an end-to-end Machine Learning engineering pipeline to forecast:
1. **Task A (Next-Day AQI Regression):** Continuous CPCB AQI value at day $t+1$ given multimodal observations up to day $t$.
2. **Task B (Next-Day CPCB Category Classification):** Statutory risk category (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) at day $t+1$.

---

## 2. Pipeline Architecture

The entire data, feature, training, and evaluation lifecycle is codified as a deterministic, reproducible Directed Acyclic Graph (DAG) managed via **DVC** and tracked via **MLflow**:

```mermaid
flowchart TD
    subgraph Ingestion ["1. Data Ingestion & Validation"]
        A[OpenAQ S3 Archive<br/>Hourly Pollutants] --> V[src.data.validate<br/>Schema & Sentinels]
        B[Open-Meteo API<br/>ERA5 Hourly Weather] --> V
    end

    subgraph Transformation ["2. Aggregation & Panel Construction"]
        V --> H[src.data.hourly_to_daily<br/>16h/day rule & CPCB Sub-indices]
        H --> M[src.data.merge<br/>Spatial-Temporal Join]
        M --> C[src.data.clean<br/>Train-only Winsor & Impute]
    end

    subgraph FeatureEngineering ["3. Feature Engineering & Split"]
        C --> F[src.features.build_features<br/>129 Multimodal Features]
        F --> S[Chronological Split<br/>Train: 2019-23 | Val: 2024 | Test: 2025-26]
    end

    subgraph Modeling ["4. Training, Ablations & Tracking"]
        S --> T[src.models.train<br/>Ridge, Trees, Regress-then-Bucket]
        S --> ABL[src.models.ablations<br/>Lags, Weather, Spatial Ablation]
        T --> MLF[(MLflow Tracking Store<br/>sqlite:///mlflow.db - 22 Runs)]
        ABL --> MLF
    end

    subgraph Evaluation ["5. Holdout Test Evaluation"]
        T --> E[src.models.evaluate<br/>2026 Test Evaluation]
        E --> REP[reports/figures/evaluation/<br/>Leaderboard & Diagnostics]
    end

    classDef stage fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef storage fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    class V,H,M,C,F,T,ABL,E stage;
    class A,B,S,MLF,REP storage;
```

---

## 3. Results & Leaderboard

All models were evaluated on the **unseen 2026 Test Holdout split** (`data/processed/test.parquet`, 2,331 station-days) representing genuine out-of-sample forward deployment.

### Task A: Next-Day Continuous AQI (Regression)

| Model | Architecture | Test RMSE (2026) | Test MAE | Test R² | Test Within $\pm 20$ Rate | Skill Score vs Persistence | Status |
|---|---|---|---|---|---|---|---|
| **Ridge Regression** | Regularized Linear (L2) | **44.05** | 27.32 | **0.6607** | 56.28% | **+6.72%** | **Best Regressor** |
| **Persistence Baseline** | Naive Heuristic ($\hat{y}_{t+1}=y_t$) | **47.22** | **23.54** | 0.6101 | **71.00%** | 0.00% | Benchmark |
| **Seasonal Naive** | Weekly Lag ($\hat{y}_{t+1}=y_{t-6}$) | 71.10 | 42.34 | 0.1161 | 46.80% | -50.57% | Underperforms |
| **Station-Month Mean** | Historical Station Average | 82.53 | 58.56 | -0.1912 | 29.00% | -74.79% | Underperforms |
| **XGBoost Regressor** | Gradient Boosted Trees | 108.98 | 100.80 | -1.0767 | 4.63% | -130.79% | Overfits Inversions |
| **LightGBM Regressor** | Leaf-wise GBDT | 171.80 | 160.87 | -4.1612 | 2.36% | -263.84% | Overfits Inversions |
| **Random Forest Regressor** | Bagged Ensemble | 211.11 | 198.93 | -6.7935 | 4.55% | -347.10% | Overfits Inversions |

> **Key Insight:** In time-series panel forecasting across shifting distribution regimes, unconstrained tree regressors fail to extrapolate beyond historical training peaks. Ridge Regression achieves the lowest test error (**RMSE: 44.05**), beating the strong Persistence baseline by **+6.72%**.

### Task B: Next-Day CPCB Category (Classification)

| Model | Architecture | Test Accuracy (2026) | Test Macro-F1 | Test Weighted-F1 | Within-One-Category Rate | Safety Evaluation |
|---|---|---|---|---|---|---|
| **Random Forest Classifier** | Balanced Subsample Forest | **71.30%** | **0.5581** | **0.7079** | **96.35%** | **Best Overall Classifier** |
| **Regress-Then-Bucket** | Hybrid (Ridge $\to$ CPCB cutoffs) | 63.36% | 0.3927 | 0.6178 | **97.21%** | **Highest Safety (Lowest Extreme Errors)** |
| **LightGBM Classifier** | Multi-class GBDT | 62.81% | 0.4532 | 0.6283 | 96.44% | Robust Categorization |
| **Majority Class Baseline** | Heuristic ('Moderate') | 41.36% | 0.0975 | 0.2420 | 88.42% | Zero Minority Recall |
| **Logistic Regression** | Multinomial (Balanced) | 33.76% | 0.2764 | 0.3244 | 90.39% | Sub-optimal Linear Boundary |

---

## 4. Feature Ablation Studies (Phase 7)

Ablation experiments evaluated on the validation split empirically quantify the predictive value of each data stream:

| Ablation Subset | Features Removed | Remaining Features | Val RMSE | Val R² | % Degradation vs Full | Conclusion |
|---|---|---|---|---|---|---|
| **Full Multimodal Feature Set** | *None (All Features)* | **129** | **42.79** | **0.7473** | **0.00%** | Baseline |
| **No Spatial Context** | City one-hots & city-mean AQI | 118 | 43.33 | 0.7409 | **+1.27%** | Geographic baselines calibrate multi-city panels |
| **No Weather Features** | Open-Meteo ERA5 variables | 103 | 44.20 | 0.7304 | **+3.30%** | **Meteorological enrichment adds real physical signal** |
| **No Autoregressive Lags** | Pollutant and AQI lags [1,2,3,7] | 76 | 47.28 | 0.6915 | **+10.50%** | **Lags are the single most critical predictor** |

---

## 5. Inference Service (Phase 3)

The Phase 2 champion (`residual_lightgbm_monotone`, 103 features, bundle `1.1`) is served by a FastAPI app in `src/service/`. The client sends **today's raw observations only**; the server rebuilds every calendar, city-static, vehicle/IIP, GRAP, festival and lag/rolling feature from the bundle's 45-day per-city history seed, using the same feature code as offline training (`src/features/serving_features.py`). The model is a persistence-anchored LightGBM with monotone fire/wind constraints and a uniform per-city additive calibration fitted on validation (ADR D-030/D-031/D-038).

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe (always 200 while the process is alive) |
| `GET` | `/ready` | Readiness: 503 until the bundle, scaler and history seed are loaded |
| `GET` | `/metadata` | Model version, val/test metrics, gate status, feature counts |
| `POST` | `/predict` | Single city-day prediction |
| `POST` | `/predict/batch` | Vectorised batch (up to 63 rows = 9 cities × 7 days) |
| `GET` | `/metrics` | Prometheus exposition (requests, latency, predictions, out-of-range inputs) |
| `GET` | `/docs` | OpenAPI / Swagger UI |

```bash
uvicorn src.service.app:app --host 0.0.0.0 --port 8000
```

Request and response:

```jsonc
// POST /predict
{
  "city": "Delhi",
  "date": "2024-12-31",
  "aqi": 180.0,
  "temperature_2m_mean": 14.0,
  "wind_speed_10m_mean": 4.5,
  "relative_humidity_2m_mean": 72.0,
  "fire_count_500km": 12.0
}
```

```jsonc
// 200 OK (real response from the champion bundle)
{
  "city": "Delhi",
  "date": "2024-12-31",
  "target_date": "2025-01-01",
  "aqi_pred": 212.29964672681731,
  "category": "Poor",
  "category_probs": {
    "Good": 1e-06, "Satisfactory": 0.000518, "Moderate": 0.359182,
    "Poor": 0.63509, "Very Poor": 0.005209, "Severe": 0.0
  },
  "aqi_interval_80": {
    "lower": 168.3883467268173,
    "median": 212.29964672681731,
    "upper": 256.1392467268173
  },
  "model_name": "residual_lightgbm_monotone",
  "model_version": "1.1",
  "alias": "champion",
  "model_used": "residual_lightgbm_monotone",
  "fallback_used": false,
  "warnings": [],
  "latency_ms": 26.463,
  "degraded": false,
  "history_days": 45
}
```

- **Stateless history:** pass an optional `history` array of previous city-days (max 90) to rebuild lags without the server store. Supplying it does not mutate the store.
- **Errors:** `422` schema/validation (unknown city, bad date, extra feature fields), `400` physically impossible values (e.g. negative AQI, humidity > 100), `503` model not ready.
- **Observability:** every response echoes or generates `X-Request-ID`; each request emits one JSON log line; `degraded=true` flags short history, a calendar gap, or a persistence fallback.
- **Parity:** the service matches the offline pipeline within `1e-6` on 50 validation rows (predictions and all 103 features). **Latency:** p95 ≈ 11 ms single row and ≈ 63 ms for the 9-city batch, inside the Plan's 50 ms / 100 ms budgets. Both are enforced by `tests/test_service.py`.

---

## 6. Quick Start & Setup

### 6.1 Environment Configuration
```bash
# 1. Clone repository
git clone https://github.com/SuhasRaghavendra/aqi-awareness-assistant.git
cd aqi-awareness-assistant

# 2. Create project-local virtual environment (avoids Windows MAX_PATH limits - ADR D-016)
python -m venv .venv
.\.venv\Scripts\activate       # Windows
# source .venv/bin/activate    # Linux / macOS

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Set OpenAQ API credentials
cp .env.example .env
# Edit .env and set OPENAQ_API_KEY=<your-key> (optional for unsigned S3 archive access)
```

### 6.2 Verification & Execution
```bash
# Run the full test suite (162 tests: data, features, models, integrity, registry, Phase 3 service)
pytest -q

# Recompute every corruption/leakage invariant (64 checks; exits non-zero on drift)
python scripts/audit_integrity.py

# Run automated logbook governance check
python scripts/check_logbook.py

# Inspect modelling dataset interactively
python scripts/view_dataset.py --split test --rows 5

# Reproduce pipeline via DVC
dvc repro

# Launch local MLflow Experiment Tracking UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Every hashed artifact is written atomically and verified before use. The
champion bundle is loaded through `AqiBundle.load(verify=True)`, which checks
`MANIFEST.sha256` → `MANIFEST.json` → every payload **before** unpickling, so a
tampered or half-written `bundle.joblib` can never be served or registered. The
train/val/test md5 digests are frozen in `params.yaml` (`integrity:`), enforced
by the audit and by `tests/test_data_integrity.py`; changing a split now
requires a deliberate hash update and an ADR.

---

## 7. Repository Layout

```
aqi-awareness-assistant/
├── artifacts/
│   └── champion/             # Byte-stable champion bundle (model, scaler, manifest, 45-day history seed)
├── configs/                  # Frozen station lists, CPCB breakpoints, festival dates, GRAP calendar
├── data/                     # DVC-versioned datasets
│   ├── raw/                  # OpenAQ S3 archives, ERA5 weather CSVs, station catalog
│   ├── interim/              # Daily aggregated panels, cleaned panels
│   └── processed/            # Chronological splits (train, val, test) & feature list
├── docs/
│   ├── logbook/              # Audit trail: 18 step-close entries (001 to 018) & INDEX.md
│   ├── CHANGELOG.md          # Keep-a-Changelog release notes
│   ├── decisions.md          # Architectural Decision Records (ADRs D-001 to D-038)
│   ├── data_card.md          # Formal dataset schema, sourcing, and validation card
│   └── dataset_approval.md   # Faculty project approval document
├── models/                   # Serialized model artifacts (.joblib, scalers)
├── notebooks/                # 01_eda.ipynb (read-only exploratory analysis)
├── reports/
│   ├── figures/evaluation/   # Diagnostics (predicted_vs_actual, residuals, confusion_matrix)
│   ├── figures/eda/          # 15 exploratory analysis figures
│   ├── metrics/              # JSON metrics per stage, run_comparison.csv, ablations
│   ├── step_reports/         # Machine-readable JSON snapshots per step
│   ├── demo_script.md        # 5-minute presentation & evaluation walkthrough
│   └── milestone1_report.md  # Comprehensive final synthesis report
├── scripts/                  # log_step.py, build_logbook_index.py, view_dataset.py, audit_integrity.py
├── src/
│   ├── data/                 # discover, download, ingest, validate, hourly_to_daily, merge, clean
│   ├── features/             # aqi_utils (CPCB breakpoints), build_features, serving_features (shared online/offline)
│   ├── models/               # baselines, train, ablations, evaluate, compare_runs, package_model, register_model
│   ├── service/              # FastAPI app, schemas, predictor, history store, middleware
│   ├── serving/              # Champion bundle loader and MLflow pyfunc adapter
│   └── utils/                # logging_utils, stage_stats, mlflow_utils, reproducibility
└── tests/                    # 162 automated pytest tests (data, features, models, integrity, registry, service parity/latency)
```

---

## 8. Logbook & Governance (Step-Close Protocol)

To maintain an unforgeable development audit trail, every advancement strictly executed the **Step-Close Protocol**:
1. Run automated test suite (`pytest tests/ -q`).
2. Run data transformation with context-managed audit emission (`reports/metrics/<stage>_stats.json`).
3. Optionally append the data change log (`AQI_STAGE_CHANGELOG=1`).
4. Generate step-close entry (`scripts/log_step.py --step <NNN> --title "<title>"`).
5. Document ADRs in `docs/decisions.md` and rebuild `docs/logbook/INDEX.md`.
6. Enforce pre-push verification via `scripts/check_logbook.py`.

### Complete Logbook Index (`docs/logbook/INDEX.md`)

| Step | Date | Title | Phase | Author | Commit |
|---|---|---|---|---|---|
| **001** | 2026-10-14 | Repo scaffold and logging scaffold | Phase 2 | Suhas Raghavendra | `6d00fbe` |
| **002** | 2026-10-14 | Station discovery and selection | Phase 1 | Suhas Raghavendra | `9ac854f` |
| **003** | 2026-10-14 | Data acquisition OpenAQ and weather | Phase 1 | Suhas Raghavendra | `9879063` |
| **004** | 2026-10-14 | Data validation and data card | Phase 1 | Suhas Raghavendra | `92251db` |
| **005** | 2026-10-15 | DVC dataset versioning and pipeline setup | Phase 3 | Suhas Raghavendra | `6d40ad3` |
| **006** | 2026-10-15 | Hourly to daily AQI computation and EDA | Phase 4 | Suhas Raghavendra | `41ecc50` |
| **007** | 2026-10-15 | Feature engineering and chronological split | Phase 5 | Suhas Raghavendra | `9cc4b9a` |
| **008** | 2026-10-15 | Baseline models evaluation and MLflow tracking | Phase 6 & 8 | Srijan Kumar G S | `04fcfbd` |
| **009** | 2026-10-15 | Feature ablation studies and MLflow experiment comparison | Phase 7 | Srijan Kumar G S | `04fcfbd` |
| **010** | 2026-10-15 | Milestone 1 synthesis report and release tagging | Phase 9 | Srijan Kumar G S | `2a5d1dd` |
| **011** | 2026-11-10 | Phase 0 baseline fixes | Phase 0 | Suhas Raghavendra | `d988c89` |
| **012** | 2026-11-10 | Registry and bundling | Phase 2 | Suhas Raghavendra | `4b9146c` |
| **013** | 2026-11-10 | Byte-stable champion bundle | Phase 2 | Suhas Raghavendra | `a587632` |
| **014** | 2026-11-10 | Validation-selected G4 and deterministic manifest | Phase 1 / 2 | Suhas Raghavendra | `a587632` |
| **015** | 2026-11-12 | FastAPI inference service | Phase 3 | Suhas Raghavendra | `5700eac` |
| **016** | 2026-11-12 | Integrity, atomic artifacts and serving safety | Phases 0-3 | Suhas Raghavendra | `f9fa35922404` |
| **017** | 2026-11-12 | CI repair and hermetic quality gates | Phases 0-3 | Suhas Raghavendra | `a7598a55df8a` |
| **018** | 2026-11-12 | Gate 3 monotone constraints and per-city calibration | Phases 0-3 | Suhas Raghavendra | `5d6ff1538100` |

---

## 9. Team Contributions

| Team Member | Core Responsibilities |
|---|---|
| **Suhas Raghavendra (Student A)** | Logging scaffold, OpenAQ station discovery & download, DVC remote configuration, daily aggregation & CPCB sub-index logic, data cleaning. |
| **Srijan Kumar G S (Student B)** | Weather acquisition, ingestion validation gates, feature engineering, baseline & tree model training, MLflow tracking (22 runs), ablation studies, holdout evaluation, synthesis report. |

---

## 10. Data Sources & Licensing

- **OpenAQ Archive:** OpenAQ Community, AWS Open Data Archive (`s3://openaq-data-archive/`). Aggregated under provider licenses.
- **Open-Meteo ERA5:** European Centre for Medium-Range Weather Forecasts (ECMWF) / Open-Meteo, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **CPCB NAQI Standard:** Central Pollution Control Board, Ministry of Environment, Forest and Climate Change, Govt. of India (2014).
- **Codebase License:** Distributed under the [MIT License](LICENSE).
