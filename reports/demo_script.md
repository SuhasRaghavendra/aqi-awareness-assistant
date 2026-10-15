# 5-Minute Evaluation & Presentation Demo Script

**Project:** AI-Based Air Quality Awareness Assistant (MLOps Milestone 1)  
**Presenters:** Suhas Raghavendra (Student A) & Srijan Kumar G S (Student B)  
**Target Duration:** 5 minutes  

---

## Demo Agenda & Timeline

| Time | Section | Speaker | Command / Screen | Key Talking Point |
|---|---|---|---|---|
| **0:00 – 0:45** | **1. Repo & Logbook Audit Trail** | Suhas | Open `docs/logbook/INDEX.md` & `README.md` | "We built an audit-grade MLOps system where all 9 steps across 9 phases are logged, reproducible, and tracked by automated pre-push hooks." |
| **0:45 – 1:30** | **2. DVC Pipeline & DAG** | Suhas | Run `dvc dag` in terminal | "Our pipeline connects raw OpenAQ pollutant archives and Open-Meteo ERA5 weather into a clean DAG: validate -> hourly_to_daily -> merge -> clean -> features -> train -> evaluate." |
| **1:30 – 2:15** | **3. Data Versioning Demo** | Suhas | Run `dvc diff data-v1 data-v2` | "We demonstrate genuine dataset versioning: data-v1 captures the 24-station baseline snapshot; data-v2 captures network expansion with zero data leakage." |
| **2:15 – 3:15** | **4. MLflow Experiment Tracking** | Srijan | Open MLflow UI (`mlflow ui`) & `reports/metrics/run_comparison.csv` | "We tracked 22 experiment runs across SQLite. Every run captures all parameters, metrics, model artifacts, and signatures with zero manual intervention." |
| **3:15 – 4:15** | **5. Evaluation & Beating Persistence** | Srijan | Open `reports/figures/evaluation/predicted_vs_actual.png` & leaderboard | "Next-day air quality is heavily autocorrelated. On the unseen 2026 test set, Ridge Regression beats Persistence with an RMSE of 44.05 vs 47.22 (+6.72% skill score). For Task B, Random Forest Classifier achieves 71.30% Accuracy and 96.35% within-one-category accuracy." |
| **4:15 – 5:00** | **6. Feature Ablation Insight & Close** | Srijan | Open `reports/figures/evaluation/ablation_study.png` | "Our feature ablations prove two vital conclusions: (1) dropping past lags causes a +10.5% RMSE failure (worse than persistence); (2) dropping ERA5 weather causes a +3.3% RMSE degradation, proving weather enrichment provides real physical value." |

---

## Verbatim Terminal Commands for Live Demonstration

### Step 1: Verification of Pipeline Tests & Governance
```bash
# Verify all 76 unit tests pass
.\.venv\Scripts\python.exe -m pytest tests/ -q

# Verify logbook governance hook passes across all 9 entries
.\.venv\Scripts\python.exe scripts/check_logbook.py
```

### Step 2: Display the DVC Pipeline Architecture DAG
```bash
# Show multi-stage dependency graph
.\.venv\Scripts\dvc.exe dag
```

### Step 3: Interactive Dataset Inspection
```bash
# Interactive table view of clean modelling panel
.\.venv\Scripts\python.exe scripts/view_dataset.py --split test --rows 5
```

### Step 4: MLflow Tracking Store & Leaderboard Comparison
```bash
# Print cross-run leaderboard across 22 tracked runs
.\.venv\Scripts\python.exe -m src.models.compare_runs
```

### Step 5: Launch MLflow UI (Optional Local Browser View)
```bash
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open `http://localhost:5000` to inspect:
- Experiment `aqi-regression-baselines` (12 runs)
- Experiment `aqi-classification-baselines` (6 runs)
- Experiment `aqi-ablations` (4 runs)

---

## Key Numbers to Quote Confidently

1. **Dataset Scope:** 13,763 station-days, 9,565 valid CPCB AQI records across 24 stations in 8 major Indian cities (2019 to 2026).
2. **Chronological Splits:**
   - Train (2019–2023): 3,023 station-days
   - Validation (2024): 2,998 station-days
   - Test (2025–2026): 2,331 station-days (evaluated once)
3. **Features:** 129 multimodal engineered features (lags, rolling stats, ERA5 weather, calendar harmonics, spatial encodings).
4. **Task A (Regression):**
   - **Ridge Regression Test RMSE:** **44.05** (R²: 0.6607)
   - **Persistence Test RMSE:** **47.22** (R²: 0.6101)
   - **Skill Score:** **+6.72%** over persistence.
5. **Task B (Classification):**
   - **Random Forest Classifier Test Accuracy:** **71.30%** (Macro-F1: 0.5581)
   - **Regress-Then-Bucket Within-One-Category Rate:** **97.21%**
6. **Ablation Studies:**
   - Omitting Lags: **+10.50% RMSE degradation** (proves temporal memory necessity)
   - Omitting Weather: **+3.30% RMSE degradation** (proves ERA5 meteorological enrichment value)
   - Omitting Spatial: **+1.27% RMSE degradation** (proves geographic baseline calibration)
