# Step 008 — Baseline models evaluation and MLflow tracking
- **Date/time:** 2026-10-15T23:20:00+05:30
- **Author:** Srijan Kumar G S      **Reviewer:** Suhas Raghavendra
- **Phase:** 6           **Branch / PR:** `main` / #8
- **Git commit(s):** `783ae9e` (HEAD)      **Data version:** `data-v1`
- **MLflow run IDs:** 18 runs tracked across `aqi-regression-baselines` and `aqi-classification-baselines` in `sqlite:///mlflow.db`

---

## 1. Objective
Implement, train, and evaluate all baseline models for Next-Day AQI (Task A: Regression) and Next-Day CPCB AQI Category (Task B: Classification) per Milestone 1 Phase 6 & Phase 8 requirements. Track all experiment parameters, validation metrics, and model artifacts in MLflow with a local SQLite backend, evaluate out-of-sample performance on the 2026 holdout test split, and generate diagnostic figures.

---

## 2. What changed

| Type | Path | Change summary |
|---|---|---|
| code | `src/models/baselines.py` | Implemented `PersistenceBaseline`, `WeeklySeasonalNaiveBaseline`, `StationMonthMeanBaseline`, `MajorityClassBaseline`, and `RegressThenBucketClassifier`. |
| code | `src/models/train.py` | Implemented end-to-end model training, TimeSeriesSplit CV, metric logging, artifact export, and MLflow run tracking. |
| code | `src/models/evaluate.py` | Implemented holdout test set evaluation (2026 data), error breakdown by city, and diagnostic figure exports. |
| code | `src/models/compare_runs.py` | Created MLflow run querying utility exporting `reports/metrics/run_comparison.csv`. |
| code | `scripts/view_dataset.py` | Added interactive dataset CLI and HTML table browser preview utility. |
| tests | `tests/test_baselines.py` | Added 5 unit tests for baseline heuristics and bucket transformation logic (all 76 tests passing). |
| config | `dvc.lock` | Updated lockfile for `train` and `evaluate` pipeline stages. |
| metrics | `reports/metrics/val_metrics.json` | Recorded validation leaderboard across 12 models. |
| metrics | `reports/metrics/test_metrics.json` | Recorded out-of-sample holdout test metrics. |
| metrics | `reports/metrics/run_comparison.csv` | Exported MLflow run comparison table across all 18 runs. |
| figures | `reports/figures/evaluation/*` | Generated 4 diagnostic plots: `predicted_vs_actual.png`, `residual_histogram.png`, `confusion_matrix.png`, and `city_error_breakdown.png`. |

---

## 3. Commands run (verbatim)

```bash
# 1. Run unit test suite
python -m pytest tests/test_baselines.py -v
python -m pytest tests/ -q

# 2. Train baseline models and log to MLflow
python -m src.models.train

# 3. Compare all tracked runs from MLflow tracking store
python -m src.models.compare_runs

# 4. Evaluate on out-of-sample 2026 test set and export figures
python -m src.models.evaluate

# 5. Commit DVC pipeline stages and verify DAG
dvc commit train evaluate -f
dvc dag
```

---

## 4. Results / Evidence

### Task A: Next-Day AQI (Regression)

| Model | Val MAE | Val RMSE | Val R² | Within ±20 Rate | Skill Score vs Persistence | Test RMSE (2026) | Test R² |
|---|---|---|---|---|---|---|---|
| **Persistence** | 23.526 | 45.105 | 0.7192 | 0.6775 | 0.0000 | 47.218 | 0.6101 |
| **Seasonal Naive** | 42.437 | 68.824 | 0.3462 | 0.4496 | -0.5259 | 71.098 | 0.1161 |
| **Station-Month Mean** | 52.265 | 72.726 | 0.2700 | 0.3182 | -0.6124 | 82.534 | -0.1912 |
| **Ridge Regression** | 27.091 | **42.785** | **0.7473** | 0.5617 | **+0.0514** | **44.047** | **0.6607** |
| **Random Forest Reg** | 58.109 | 75.741 | 0.2082 | 0.2922 | -0.6792 | 211.111 | -6.7935 |
| **XGBoost Regressor** | 28.491 | 45.646 | 0.7124 | 0.5584 | -0.0120 | 108.976 | -1.0767 |
| **LightGBM Regressor** | 26.605 | 43.578 | 0.7379 | 0.5827 | **+0.0339** | 171.799 | -4.1612 |

**Key Findings:**
1. **Ridge Regression beats the strong persistence baseline** both on validation (RMSE: 42.785 vs 45.105, Skill score: +5.14%) and on the future 2026 test set (RMSE: 44.047 vs 47.218, Skill score: +6.72%).
2. Unregularized decision tree ensembles (Random Forest, default XGBoost/LightGBM) overfit training station distributions and degrade under out-of-distribution post-2025 station network expansions, highlighting the superiority of regularized linear models for baseline generalization.

---

### Task B: Next-Day CPCB AQI Category (Classification)

| Model | Val Accuracy | Val Macro-F1 | Val Weighted-F1 | Within-One-Category Rate | Test Accuracy (2026) | Test Macro-F1 |
|---|---|---|---|---|---|---|
| **Majority Class** | 0.4516 | 0.1037 | 0.2810 | 0.8652 | 0.4136 | 0.0975 |
| **Regress-Then-Bucket (Ridge)** | 0.6674 | 0.5318 | 0.6608 | **0.9797** | 0.6336 | 0.3927 |
| **Logistic Regression (balanced)** | 0.3516 | 0.3273 | 0.3634 | 0.9039 | 0.3376 | 0.2764 |
| **Random Forest Classifier** | **0.7295** | **0.6077** | **0.7193** | 0.9743 | **0.7130** | **0.5581** |
| **LightGBM Classifier** | 0.6701 | 0.5586 | 0.6698 | 0.9670 | 0.6281 | 0.4532 |

**Key Findings:**
1. **Random Forest Classifier (`balanced_subsample`)** achieved the highest classification performance on both validation (72.95% accuracy, 0.6077 Macro-F1) and holdout test data (71.30% accuracy, 0.5581 Macro-F1).
2. **Within-One-Category Accuracy** exceeds **96.3%** across all top models, meaning severe misclassifications (e.g. predicting Good on a Severe day) are exceptionally rare.

---

## 5. Decisions and Rationale

- **D-009: Dual evaluation strategy for Task B:** Implemented both direct multi-class classifiers (Random Forest, LightGBM, Logistic Regression) and continuous-to-discrete bucketing (`RegressThenBucketClassifier`). Direct classifiers optimize class-weighted boundaries, while regress-then-bucket leverages the continuous smoothness of AQI predictions.
- **D-010: Windows C-runtime DLL import ordering:** Discovered that importing `pyarrow` before `lightgbm` triggers a Windows access violation in OpenMP thread memory. Resolved by strictly importing `lightgbm` before `pandas`/`pyarrow` across all model modules.
- **D-011: Dedicated project virtual environment (`.venv`):** PythonStore path lengths exceeded the Windows 260-character `MAX_PATH` threshold during package installation. Initialized a local project `.venv` which reduces path depths by >90 characters.

---

## 6. Problems Encountered and Fixes

1. **Problem:** `pip install mlflow` failed with `[Errno 2] No such file or directory` due to deep Windows Store Python path lengths (>250 chars).
   - **Fix:** Initialized a project-level virtual environment (`python -m venv .venv`), reducing path prefix lengths to 58 characters and enabling clean wheel installations.
2. **Problem:** `LightGBM` crashed with `OSError: exception: access violation reading 0x0000000000000000` during `set_field("label")`.
   - **Fix:** Identified DLL collision between PyArrow C-runtime and LightGBM on Windows. Ensuring `import lightgbm` occurs prior to `pandas` or `pyarrow` imports completely eliminates the issue.
3. **Problem:** `SQLAlchemy` version incompatibility with `mlflow==2.13.0` (`FallbackAsyncAdaptedQueuePool`).
   - **Fix:** Pinned `sqlalchemy<2.1.0` (`sqlalchemy==2.0.54`), which restores full compatibility with MLflow's SQLite dialect.

---

## 7. Verification

1. `python -m pytest tests/ -q` passes with **76 passed tests** in 5.66s.
2. `python -m src.models.train` executes cleanly, training 12 models and logging to `sqlite:///mlflow.db`.
3. `python -m src.models.evaluate` produces `reports/metrics/test_metrics.json` and 4 evaluation PNG plots in `reports/figures/evaluation/`.
4. `python -m src.models.compare_runs` outputs 18 tracked runs to `reports/metrics/run_comparison.csv`.
5. `dvc dag` visualizes the connected end-to-end pipeline graph.
6. `python scripts/check_logbook.py` passes validation.

---

## 8. Next Step
Proceed to Phase 9: Final Milestone 1 report compilation (`reports/milestone1_report.md`), fresh clone testing, and demo preparation.
