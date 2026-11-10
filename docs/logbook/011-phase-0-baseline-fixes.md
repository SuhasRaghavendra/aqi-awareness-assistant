# Step 011 — Phase 0 baseline fixes
- **Date/time:** 2026-11-10T13:14:30+05:30
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** 2           **Branch / PR:** `main` / #<n>
- **Git commit(s):** `d988c890945117e83c366b72bfe7d041b796d4ec`      **Data version:** <git-tag> / <dvc-md5>
- **MLflow run IDs:** N/A

## 1. Objective
<!-- TODO: 1–3 sentences describing what this step achieves and why. -->

## 2. What changed

| Type | Path | Change summary |
|---|---|---|
| code | `scripts/build_eda_notebook.py` | — |
| code | `scripts/check_logbook.py` | — |
| code | `scripts/feature_ranking_plot.py` | — |
| code | `scripts/generate_eda_figures.py` | — |
| code | `scripts/log_step.py` | — |
| code | `scripts/run_eda.py` | — |
| code | `scripts/setup_dvc_remote.py` | — |
| code | `scripts/simulate_data_v2.py` | — |
| code | `scripts/view_dataset.py` | — |
| code | `src/config.py` | — |
| code | `src/data/clean.py` | — |
| code | `src/data/download_aqi.py` | — |
| code | `src/data/download_fires.py` | — |
| code | `src/data/download_industrial.py` | — |
| code | `src/data/download_vehicles.py` | — |
| code | `src/data/download_weather.py` | — |
| code | `src/data/ingest_and_merge.py` | — |
| code | `src/data/validate.py` | — |
| code | `src/features/aqi_utils.py` | — |
| code | `src/features/build_features.py` | — |
| code | `src/models/ablations.py` | — |
| code | `src/models/baselines.py` | — |
| code | `src/models/compare_runs.py` | — |
| code | `src/models/evaluate.py` | — |
| code | `src/models/train.py` | — |
| code | `src/utils/logging_utils.py` | — |
| code | `src/utils/mlflow_utils.py` | — |
| code | `src/utils/stage_stats.py` | — |
| config | `dvc.lock` | — |
| config | `dvc.yaml` | — |
| config | `params.yaml` | — |
| docs | `docs/data_change_log.md` | — |
| docs | `docs/decisions.md` | — |
| docs | `reports/figures/features/feature_correlations.png` | — |
| docs | `reports/metrics/clean_stats.json` | — |
| docs | `reports/metrics/evaluate_stats.json` | — |
| docs | `reports/metrics/features_stats.json` | — |
| docs | `reports/metrics/test_metrics.json` | — |
| docs | `reports/metrics/train_stats.json` | — |
| docs | `reports/metrics/val_metrics.json` | — |
| docs | `docs/Plan.md` | — |
| docs | `reports/metrics/test_per_city_metrics.json` | — |
| tests | `tests/test_build_features.py` | — |
| tests | `tests/test_download_weather.py` | — |
| tests | `tests/test_fires.py` | — |
| tests | `tests/test_ingest_merge.py` | — |
| other | `qi_mlops_implementation_plan_v2.md` | — |
| other | `pyproject.toml` | — |
| other | `.gitattributes` | — |
| other | `requirements-dev.txt` | — |
| other | `requirements-serve.txt` | — |

<details>
<summary>git diff --stat</summary>

```
aqi_mlops_implementation_plan_v2.md               |  572 ------------
 docs/data_change_log.md                           |  220 +++++
 docs/decisions.md                                 |    9 +-
 dvc.lock                                          |  485 ++++++++++
 dvc.yaml                                          |   10 +-
 params.yaml                                       |    5 +-
 pyproject.toml                                    |    4 +-
 reports/figures/.gitignore                        |    1 +
 reports/figures/features/feature_correlations.png |  Bin 0 -> 62214 bytes
 reports/metrics/clean_stats.json                  |  224 +++++
 reports/metrics/data_validation.json              |   30 +
 reports/metrics/download_aqi_stats.json           |   33 +
 reports/metrics/download_fires_stats.json         |   19 +
 reports/metrics/download_industrial_stats.json    |    5 +
 reports/metrics/download_vehicles_stats.json      |   22 +
 reports/metrics/download_weather_stats.json       |   73 ++
 reports/metrics/evaluate_stats.json               |  398 ++++++++
 reports/metrics/features_stats.json               |  665 ++++++++++++++
 reports/metrics/merge_stats.json                  |   67 ++
 reports/metrics/test_metrics.json                 |  126 +++
 reports/metrics/train_stats.json                  |  567 ++++++++++++
 reports/metrics/val_metrics.json                  |  139 +++
 scripts/build_eda_notebook.py                     | 1003 +++++++++++----------
 scripts/check_logbook.py                          |   20 +-
 scripts/feature_ranking_plot.py                   |   49 +-
 scripts/generate_eda_figures.py                   |  115 ++-
 scripts/log_step.py                               |   21 +-
 scripts/run_eda.py                                |  174 +++-
 scripts/setup_dvc_remote.py                       |    4 +-
 scripts/simulate_data_v2.py                       |   75 +-
 scripts/view_dataset.py                           |   25 +-
 src/config.py                                     |    4 +-
 src/data/clean.py                                 |   91 +-
 src/data/download_aqi.py                          |  276 ++++--
 src/data/download_fires.py                        |  198 ++--
 src/data/download_industrial.py                   |  265 +++---
 src/data/download_vehicles.py                     |   90 +-
 src/data/download_weather.py                      |   87 +-
 src/data/ingest_and_merge.py                      |  141 +--
 src/data/validate.py                              |   68 +-
 src/features/aqi_utils.py                         |   15 +-
 src/features/build_features.py                    |  238 +++--
 src/models/ablations.py                           |   70 +-
 src/models/baselines.py                           |   34 +-
 src/models/compare_runs.py                        |   19 +-
 src/models/evaluate.py                            |  197 +++-
 src/models/train.py                               |  379 ++++++--
 src/utils/logging_utils.py                        |   13 +-
 src/utils/mlflow_utils.py                         |    5 +-
 src/utils/stage_stats.py                          |   15 +-
 tests/test_build_features.py                      |   92 +-
 tests/test_download_weather.py                    |   19 +-
 tests/test_fires.py                               |   83 +-
 tests/test_ingest_merge.py                        |   57 +-
 54 files changed, 5634 insertions(+), 1982 deletions(-)
warning: in the working copy of 'params.yaml', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'pyproject.toml', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'scripts/build_eda_notebook.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'scripts/check_logbook.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'scripts/feature_ranking_plot.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'scripts/generate_eda_figures.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'scripts/run_eda.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/data/download_aqi.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/data/download_fires.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/data/download_industrial.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/features/aqi_utils.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/utils/mlflow_utils.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tests/test_build_features.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tests/test_download_weather.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tests/test_fires.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tests/test_ingest_merge.py', LF will be replaced by CRLF the next time Git touches it
```
</details>

<details>
<summary>git status --short</summary>

```
D aqi_mlops_implementation_plan_v2.md
 M docs/data_change_log.md
 M docs/decisions.md
 M dvc.lock
 M dvc.yaml
 M params.yaml
 M pyproject.toml
 M reports/figures/features/feature_correlations.png
 M reports/metrics/clean_stats.json
 M reports/metrics/evaluate_stats.json
 M reports/metrics/features_stats.json
 M reports/metrics/test_metrics.json
 M reports/metrics/train_stats.json
 M reports/metrics/val_metrics.json
 M scripts/build_eda_notebook.py
 M scripts/check_logbook.py
 M scripts/feature_ranking_plot.py
 M scripts/generate_eda_figures.py
 M scripts/log_step.py
 M scripts/run_eda.py
 M scripts/setup_dvc_remote.py
 M scripts/simulate_data_v2.py
 M scripts/view_dataset.py
 M src/config.py
 M src/data/clean.py
 M src/data/download_aqi.py
 M src/data/download_fires.py
 M src/data/download_industrial.py
 M src/data/download_vehicles.py
 M src/data/download_weather.py
 M src/data/ingest_and_merge.py
 M src/data/validate.py
 M src/features/aqi_utils.py
 M src/features/build_features.py
 M src/models/ablations.py
 M src/models/baselines.py
 M src/models/compare_runs.py
 M src/models/evaluate.py
 M src/models/train.py
 M src/utils/logging_utils.py
 M src/utils/mlflow_utils.py
 M src/utils/stage_stats.py
 M tests/test_build_features.py
 M tests/test_download_weather.py
 M tests/test_fires.py
 M tests/test_ingest_merge.py
?? .gitattributes
?? docs/Plan.md
?? reports/metrics/test_per_city_metrics.json
?? requirements-dev.txt
?? requirements-serve.txt
```
</details>

<details>
<summary>dvc status</summary>

```
download_aqi:
	changed deps:
		modified:           src\data\download_aqi.py
		modified:           src\config.py
download_fires:
	changed deps:
		modified:           src\data\download_fires.py
		modified:           src\config.py
download_weather:
	changed deps:
		modified:           src\data\download_weather.py
		modified:           src\config.py
download_vehicles:
	changed deps:
		modified:           src\data\download_vehicles.py
		modified:           src\config.py
download_industrial:
	changed deps:
		modified:           src\data\download_industrial.py
		modified:           src\config.py
ingest_and_merge:
	changed deps:
		modified:           src\data\ingest_and_merge.py
		modified:           src\config.py
validate:
	changed deps:
		modified:           src\data\validate.py
		modified:           src\config.py
clean:
	changed deps:
		modified:           src\data\clean.py
		modified:           src\features\aqi_utils.py
		modified:           src\config.py
features:
	changed deps:
		modified:           src\features\build_features.py
		modified:           src\features\aqi_utils.py
		modified:           src\config.py
train:
	changed deps:
		modified:           src\models\train.py
		modified:           src\models\baselines.py
		modified:           src\config.py
evaluate:
	changed deps:
		modified:           src\models\evaluate.py
		modified:           src\config.py
```
</details>

<details>
<summary>dvc diff</summary>

```
Added:
    models\xgboost_clf.joblib

Modified:
    data\interim\city_day_clean.parquet
    data\processed\feature_list.json
    data\processed\scaler.joblib
    data\processed\test.parquet
    data\processed\train.parquet
    data\processed\val.parquet
    models\
    models\lightgbm_clf.joblib
    models\lightgbm_reg.joblib
    models\logistic_regression.joblib
    models\persistence.joblib
    models\random_forest_clf.joblib
    models\random_forest_reg.joblib
    models\regress_then_bucket.joblib
    models\ridge.joblib
    models\seasonal_naive.joblib
    models\station_month_mean.joblib
    models\xgboost_reg.joblib
    reports\figures\evaluation\
    reports\figures\evaluation\city_error_breakdown.png
    reports\figures\evaluation\confusion_matrix.png
    reports\figures\evaluation\predicted_vs_actual.png
    reports\figures\evaluation\residual_histogram.png

files summary: 1 added, 21 modified
```
</details>

### Stage statistics summary
- **clean_stats**: rows 25470 -> 24739  |  dropped 0  |  0.021s
- **download_aqi_stats**: rows ? -> ?  |  dropped ?  |  ?s
- **download_fires_stats**: rows ? -> ?  |  dropped ?  |  ?s
- **download_industrial_stats**: rows ? -> ?  |  dropped ?  |  ?s
- **download_vehicles_stats**: rows ? -> ?  |  dropped ?  |  ?s
- **download_weather_stats**: rows ? -> 25470  |  dropped 0  |  0.009s
- **evaluate_stats**: rows 1852 -> ?  |  dropped 0  |  0.019s
- **features_stats**: rows 20643 -> 15502  |  dropped 0  |  0.05s
- **merge_stats**: rows ? -> ?  |  dropped ?  |  ?s
- **train_stats**: rows 15502 -> 3289  |  dropped 0  |  0.016s

## 3. Commands run (verbatim)
```bash
# TODO: paste the exact commands run in this step
```

## 4. Results / evidence
<!-- TODO: key numbers, table excerpts, figure paths. Quote real output only. -->

## 5. Decisions and rationale
<!-- TODO: design decisions, alternatives considered, why chosen. Copy ADR-worthy ones to docs/decisions.md -->

## 6. Problems encountered and fixes
<!-- TODO: errors, surprises, data issues and how they were resolved. -->

## 7. Verification
<!-- TODO: commands + expected output to reproduce this step; tests added/passing. -->

## 8. Next step
<!-- TODO: one-line description of the next step. -->
