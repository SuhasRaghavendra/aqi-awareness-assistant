# Step 018 — Gate 3: monotone constraints and uniform per-city calibration

- **Date/time:** 2026-11-12T12:30:00+05:30
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** 0–3           **Branch / PR:** `main` / #<n>
- **Git commit(s):** `<hash>`      **Data version:** frozen splits unchanged (`train` `49bd4c88748327afa12057c2fe71b0e7`, `val` `462b7712b4cea342fe36ca80868a29f4`, `test` `bd1de9ea97083402bd9ccaceb1072e3c`)
- **MLflow run IDs (if any):** candidates logged under step `018`; `Indian-AQI-Predictor` v23 `@champion`

## 1. Objective

Close Gate 3 without leaking the holdout into any decision:

1. **A2** — make the fire/wind direction structural with LightGBM monotone
   constraints and restore the Plan's 5% metamorphic tolerance (D-030).
2. **A3** — attempt G3 (Kolkata) with a uniform, validation-only mechanism
   applied to all nine cities, freeze the candidate, then evaluate the test
   split exactly once (D-031).
3. **A4** — put train/validation/CV overfitting evidence in the risk card.

No gate threshold was changed. The test split was not consulted while choosing
the model, its configuration, or the calibration.

## 2. What changed

| Type | Path / object | Change summary |
|---|---|---|
| model | `src/models/monotone.py` (new) | Generic constraint rule: fire count/FRP and their lags/rolling aggregates `+1`, `fire_burning_season`, `inversion_x_fire`, `fire_x_low_wind` and `winter_stagnation_index` `+1`; `wind_speed_10m_mean`, its lags and rolling means `-1`. `fire_dist_min_km` stays unconstrained. |
| model | `src/models/calibration.py` (new) | `per_city_offsets` (median residual), forward-chained calibration for honest validation scoring, and `CityCalibratedRegressor`. |
| model | `src/models/train.py` | Constrained candidates for `l2` and `huber`; each reuses the three pre-specified LightGBM configurations and keeps the best validation OOF RMSE. Records `selected`, `val_max_city_ratio`, `cv_rmse_*`, calibration folds and pre-registered acceptance. |
| model | `src/models/final_eval.py`, `scripts/final_test_eval.py` (new) | Single-shot holdout report: split-hash guard, candidate identity, metrics, weekly block-bootstrap CI and per-city rows. `--check` re-derives and diffs; `--allow-new-candidate` is used by the DVC stage. |
| model | `src/models/model_risk.py` | Champion selection consumes the `selected` flag; holdout numbers come from `final_test_eval.json` (with test-md5 and artifact-md5 checks); adds the A4 diagnostics block. |
| serving | `src/serving/bundle.py` | Recursive estimator walk, calibrated prediction path, calibration-coverage check against the history seed. |
| packaging | `src/models/package_model.py` | Records `calibrated`, calibration offsets and monotone constraints; `gate_waivers` is now derived from the gates that actually fail. |
| evaluation | `src/models/evaluate.py` | Per-city analysis follows the validation-selected champion; never re-ranks candidates on test. |
| pipeline | `dvc.yaml`, `params.yaml` | `final_test_eval` stage between `evaluate` and `model_risk`; `train.monotone` block; explicit `val_metrics.json` dependencies. |
| tests | `tests/test_calibration.py`, `test_final_eval.py`, `test_monotone.py`, `test_model_risk.py`, `test_service.py`, `test_bundle.py` | 13 new tests; tolerance restored to 0.05; diagnostics and hash-guard coverage; model-name assertions read the bundle instead of a hard-coded name. |

## 3. Commands run (verbatim)

```bash
python -m pytest tests -q
python -m ruff check src tests scripts
python -m black --check .
dvc repro
python scripts/audit_integrity.py
python scripts/verify_registry.py
python -m src.models.register_model --promote
python -m src.models.register_model --verify
```

## 4. Results / evidence

### Validation (selection data)

| Candidate | Val RMSE | vs champion | Val max city ratio | Metamorphic |
|---|---|---|---|---|
| `residual_lightgbm` (incumbent) | 20.099 | — | 0.9722 | 13.2% non-monotone |
| `residual_lightgbm_monotone` | 20.610 (raw 20.340) | +1.20% raw, +2.54% calibrated | 0.9673 | **0.0%** fire, **0.0%** wind |
| `residual_lightgbm_monotone_huber` | 22.366 | +11.3% | 0.9903 | 0.0% |

The constrained `l2` candidate passes the pre-registered A2 acceptance:
uncalibrated validation RMSE is +1.20% (limit 2%), non-monotone rate 0% (limit
5%), mean ΔAQI +2.387 (fire ×5) and +0.605 (wind ÷3), both ≥ 0. It is promoted.
The calibration is a separate, uniformly applied A3 step; its forward-chained
OOF cost (+1.34 pp) is reported rather than hidden.

### Holdout (observed once, after freezing)

```
rows 1852 / 9 cities
MAE 14.028   RMSE 23.886   R2 0.8772
skill vs persistence 0.0640  [0.0364, 0.0882]
test md5 bd1de9ea97083402bd9ccaceb1072e3c
```

| Gate | Value | Status |
|---|---|---|
| G1 | val skill 0.1026, test skill 0.0640 | pass |
| G2 | test R² 0.8772, MAE 14.028 | pass |
| G3 | max ratio 1.0597 (Kolkata 14.593 vs 13.771) | **fail — waived (D-023/D-038)** |
| G4 | within-one 0.9951 | pass |
| G5 | p95 single-row latency < 50 ms | pass |

G3 improved from 1.0922 to 1.0597 under the uniform calibration but remains
above the frozen 1.05 threshold. Per D-031 the result stands: no post-test
tuning, no threshold change. The waiver is recorded in `MANIFEST.json` and the
MLflow version, and the failing city remains visible in the risk card.

### Overfitting evidence (A4)

```
train RMSE 21.921   val RMSE 20.610   gap -1.311   ratio 0.9402
CV RMSE 24.879 ± 2.949 over 4 TimeSeriesSplit folds
interpretation: validation RMSE is within 10% of training RMSE
```

The negative gap reflects the residual model fitting a wider training-period
target distribution; the fold spread is reported so the point estimate is not
read as certainty.

### Integrity and serving

```
162 passed                              # python -m pytest tests -q
Checks: 64  failed: 0                   # scripts/audit_integrity.py
max |diff| 0.000e+00, 423 rows          # scripts/verify_registry.py
max diff 0.000e+00, 423 rows            # register_model --verify
bundle_sha256 3bdc05c8…f77f8d, 878,496 bytes
```

## 5. Decisions and rationale

- **D-030 executed.** All wind-derived features (current, lags, rolling means)
  are constrained negative and the stagnation index positive. This extension to
  the original list is required because the metamorphic fixture shifts the whole
  raw wind column: leaving lags unconstrained let them cancel the current-day
  constraint (initial run: 13.2% non-monotone). The rule is generic — no city,
  month or fixture row is special-cased.
- **D-031 executed.** Every candidate gets the same per-city median-residual
  calibration fitted on validation; selection never reads test. The A2
  acceptance judges the constrained model against the incumbent; the separately
  reported OOF calibration cost does not retroactively change A2.
- **D-038.** Records the val-frozen champion, the single test run and the
  retained G3 waiver.

## 6. Problems encountered and fixes

1. **First constrained candidate still failed the wind metamorphic check
   (13.2%).** The rebuilt feature frame also changes `wind_*_lag_*` and
   `wind_roll_*`, which were unconstrained. Extending the generic wind rule to
   all wind-derived features removed the violation entirely.
2. **First constrained run exceeded the 2% A2 limit (+2.69%).** The candidate
   only used the unconstrained family's winning config. Reusing all three
   pre-specified configs and selecting by validation OOF RMSE brought the raw
   cost to +1.20%.
3. **`dvc repro` skipped `final_test_eval` after only the acceptance bookkeeping
   changed.** The stage did not depend on `val_metrics.json`; added it to both
   `final_test_eval` and `model_risk` so a changed selection can never evaluate
   the wrong candidate.
4. **Service tests hard-coded the old champion name.** They now read
   `bundle.manifest["model"]["name"]`, so a legitimate promotion does not look
   like a contract regression.

## 7. Verification

```bash
python -m pytest tests -q                 # 162 passed
python -m ruff check src tests scripts    # clean
python -m black --check .                 # clean
python scripts/audit_integrity.py         # 64/64
python scripts/verify_registry.py         # 423 rows, max diff 0.0
python -m src.models.register_model --verify  # 423 rows, max diff 0.0
dvc status                                # clean
```

## 8. Next step

Step 019 — Gate 4: model card, final ADRs, README/CHANGELOG, live latency probe,
container smoke, tag `phase3-final`.
