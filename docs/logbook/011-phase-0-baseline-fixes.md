# Step 011 — Phase 0 baseline fixes

- **Date/time:** 2026-11-10T13:14:30+05:30
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** 0           **Branch / PR:** `main` / #<n>
- **Git commit(s):** `d988c890945117e83c366b72bfe7d041b796d4ec`      **Data version:** `train.parquet` md5 `49bd4c88748327afa12057c2fe71b0e7`
- **MLflow run IDs:** N/A (Phase 0 model runs are tracked under the existing train experiments)

## 1. Objective

Make the baseline credible before adding serving work: restore the documented
chronological split, fix the broken AQI-history feature, use one fitted scaler,
add persistence-anchored candidates, and put the evaluation protocol and
tooling on a reproducible footing. The previous split left only 83 Hyderabad
rows in test, and every learned model lost to persistence.

## 2. What changed

| Type | Path | Change summary |
|---|---|---|
| params | `params.yaml` | `split.train_end` 2022-12-31 → 2023-12-31; `split.val_end` 2025-12-31 → 2024-12-31 (ADR D-018) |
| code | `src/features/build_features.py` | Added day-t `aqi`; fixed the `aqi` history reindex bug; saved the single train-fitted scaler (ADR D-019, D-020) |
| code | `src/models/train.py` | Loads the saved scaler; added residual Ridge and residual LightGBM with TimeSeriesSplit CV; added `xgboost_clf` |
| code | `src/models/residual.py` | `ResidualRegressor`: learns `target_aqi − aqi` and adds `aqi` back at prediction time (ADR D-021) |
| code | `src/models/evaluate.py` | Loads the saved scaler; reports per-city MAE/RMSE, skill vs persistence and weekly bootstrap CIs |
| config | `dvc.yaml` / `dvc.lock` | Scaler and feature manifest are declared dependencies/outputs; metrics remain Git-tracked |
| tests | `tests/test_build_features.py` | Scaler-vs-train-split test now fails if artifacts are missing instead of skipping |
| tests | `tests/test_requirements.py` | Enforces installed training pins and training/serving pin parity (ADR D-024) |
| docs | `docs/decisions.md` | ADR D-018 to D-022 and D-024 |
| docs | `docs/data_card.md` | Split, feature count, Pune limitation and DVC graph refreshed |
| tooling | `requirements*.txt` | Training/serving pins aligned to the environment that produced the artifacts (ADR D-024) |

## 3. Commands run (verbatim)

```bash
dvc status
python -m pytest -q
ruff check src/ scripts/ tests/
black --check src/ scripts/ tests/
python scripts/check_logbook.py
```

The full pipeline reproduction command remains:

```bash
dvc repro
```

## 4. Results / evidence

- Chronological split: train `2019-01-01` → `2023-12-31` (15,502 rows),
  validation `2024-01-01` → `2024-12-31` (3,289 rows), test `2025-01-01` →
  `2026-03-26` (1,852 rows). All splits contain all nine cities.
- Feature manifest: **103 features**, including explicit day-t `aqi`.
- Validation RMSE (learned vs persistence):
  - persistence: 22.584
  - residual LightGBM: **20.099** (skill +0.1100)
  - LightGBM: 20.103
  - XGBoost: 20.168
  - residual Ridge: 20.973
  - Ridge: 21.139
- Holdout test for the selected residual LightGBM: RMSE **23.821**, MAE
  **14.114**, R² **0.8779** (persistence test RMSE 25.518).
- `ruff check` and `black --check` pass.
- Pune remains the known data-thin city (900 train rows; ADR D-022).

## 5. Decisions and rationale

- **D-018:** restore the D-004 split. The old configuration produced an
  83-row, Hyderabad-only test set; the restored split gives 1,852 rows across
  nine cities.
- **D-019:** add `aqi` and fix the reindex bug. `aqi_lag_1` was effectively 0
  for every row because a DatetimeIndex Series was reindexed with integer
  labels. Adding `aqi` and fixing the reindex lifted validation skill from
  roughly zero to +0.11.
- **D-020:** load one scaler fitted on the train split in both training and
  evaluation; never refit it downstream.
- **D-021:** model the day-over-day change for Ridge and LightGBM. Hyperparameters
  are selected by TimeSeriesSplit CV on train; validation is a single check and
  test is not used for selection.
- **D-022:** keep Pune and report its sample-size caveat rather than dropping it.
- **D-024:** pin requirements to the runtime that produced the artifacts. The
  previous plan-era pins no longer matched the environment or the saved bundle.

## 6. Problems encountered and fixes

- The original test split was 83 rows from Hyderabad only, so per-city gates
  were impossible. Fixed by the D-018 split.
- Every direct learned model lost to persistence. Fixed by adding day-t `aqi`,
  repairing the AQI history features, and adding persistence-anchored residual
  candidates.
- Three different scalers were fitted in `build_features.py`, `train.py` and
  `evaluate.py`. Fixed by saving one train-fitted scaler and loading it in both
  downstream stages.
- The `xgboost_clf` model was listed in the train docstring but had no artifact.
  It is now trained and evaluated.
- Requirements pinned older package versions than the environment that produced
  the pickled models. Pins were aligned with ADR D-024 and are enforced by
  `tests/test_requirements.py`.

## 7. Verification

```bash
dvc repro
python -m pytest -q
ruff check src/ scripts/ tests/
black --check src/ scripts/ tests/
```

Expected: DVC is up to date; all tests pass; ruff and black are clean; the
validation leaderboard shows at least one learned candidate with lower RMSE
than persistence.

Current verification: **70+ tests pass** (including the new requirements checks),
`ruff` clean, `black` clean, and `dvc status` reports the pipeline up to date.

## 8. Next step

Phase 1: add the model-risk and quality-gate audit on top of the corrected
baseline.
