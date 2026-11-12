# Model Card — Next-Day City AQI Forecast (Phase 3 champion)

| Field | Value |
|---|---|
| **Model name** | `residual_lightgbm_monotone` |
| **Artifact** | `models/residual_lightgbm_monotone.joblib` (md5 `728d68b88f7108cc0aa957a1c137a3cd`) |
| **Bundle** | `artifacts/champion` — bundle format `1.1`, sha256 `3bdc05c84ce0c2213aabd4894f17cfb25adf3d3aeba7986950fa927d4af77f8d`, 878,496 bytes |
| **Registry** | `models:/Indian-AQI-Predictor@champion` (v23 at the time of writing) |
| **Task** | Regression: next-day city-mean CPCB AQI, then bucket to a CPCB category |
| **Features** | 103 (manifest-recorded, never hard-coded) |
| **Training window** | 2019-01-01 → 2023-12-31 (15,502 rows) |
| **Validation window** | 2024-01-01 → 2024-12-31 (3,289 rows) |
| **Holdout window** | 2025-01-01 → 2026-03-26 (1,852 rows, 9 cities, 447 dates) |
| **Owner / authors** | Suhas Raghavendra, Srijan Kumar G S — course AI252TA (MLOps) |
| **Decisions** | ADRs D-018, D-019, D-023, D-030, D-031, D-038 in [`decisions.md`](decisions.md) |

---

## 1. Intended use and out-of-scope use

**Intended use**

- Next-day (day *t+1*) city-level AQI and CPCB category for the nine modelled
  cities: Ahmedabad, Bengaluru, Chennai, Delhi, Hyderabad, Kolkata, Lucknow,
  Mumbai and Pune.
- Public-awareness dashboards, classroom demonstrations of an end-to-end MLOps
  pipeline, and research on forecast skill relative to persistence.
- Inputs are today's city-mean observations plus lagged/rolling history; the
  service rebuilds calendar, policy, static, fire, weather and pollutant
  features from the bundle's 45-day per-city seed.

**Out of scope**

- Sub-city, station-level or hourly nowcasting. The model predicts a daily
  city mean and cannot resolve intra-city hotspots.
- Clinical, epidemiological or individual exposure advice; emergency response;
  regulatory enforcement or legal air-quality claims.
- Cities, countries or date ranges outside the nine-city, 2019–2026 panel.
- Long-horizon forecasts (2+ days) and counterfactual "what if we cut
  emissions" analysis: this is a statistical association model, not a
  source-apportionment or dispersion model.
- Predicting unseen pollution regimes (for example a new fuel standard or a
  lockdown-scale shock) with calibrated confidence.

---

## 2. Model family and version

The served model is a **persistence-anchored LightGBM with monotone
constraints and per-city additive calibration**:

```
Prediction = clip( AQI_today + f(features) + offset[city], 0, aqi_max )
```

- `f` is `LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=6)`
  fitted on scaled residual targets `y - AQI_today`.
- Monotone constraints (LightGBM `monotone_constraints_method="advanced"`):
  fire count/FRP and their lags and rolling aggregates `+1`;
  `fire_burning_season`, `inversion_x_fire`, `fire_x_low_wind` and
  `winter_stagnation_index` `+1`; `wind_speed_10m_mean` and all wind lags and
  rolling means `-1` (ADR D-030).
- `offset[city]` is the **median validation residual per city**, fitted for all
  nine cities with the same rule (ADR D-031). Offsets range from −3.17
  (Lucknow) to −0.99 (Hyderabad) AQI points.
- The category is produced by bucketing the point prediction onto the CPCB
  bands. A separate classifier was evaluated; on validation it was not better
  enough to justify a second artifact, so the risk card records
  `regress_then_bucket` and the classifier-selection evidence (ADR D-029/D-037).
- Random Forest was excluded from champion eligibility on artifact size; it is
  still evaluated and reported on the leaderboard.

Hyper-parameters were chosen by `TimeSeriesSplit` cross-validation on the
training split only, and the constrained candidate reused the same three
pre-specified LightGBM configurations as the unconstrained family. The
champion was frozen on validation before the holdout was touched.

---

## 3. Data sources and licensing

| Stream | Source | Rights / notes |
|---|---|---|
| AQI labels | XKDR India Air Quality Database (CPCB CAAQM harmonised) | CPCB-derived PM2.5/PM10; AQI computed locally |
| Fire detections | NASA FIRMS VIIRS SNPP | NASA open data |
| Weather | Open-Meteo Historical Weather API (ERA5 reanalysis) | CC BY 4.0 (Copernicus/ECMWF) |
| Vehicles | data.gov.in / MoRTH Vahan state registrations | GODL; annual, state level |
| Industry | MOSPI IIP, CEA state power | GODL; **empty in the current snapshot** |
| Policy / calendars | Project-curated GRAP and festival calendars | Project-maintained |
| City attributes | Public reference values | Public |

The full schema, provenance and licences are in [`data_card.md`](data_card.md).
No personally identifiable information is used.

---

## 4. Preprocessing and leakage controls

- **Chronological split** (ADR D-004/D-018): train ≤ 2023-12-31, validation
  2024, holdout ≥ 2025-01-01. Splits are materialised by the `features` stage
  and frozen by md5 in `params.yaml` (`integrity.train_md5`, `val_md5`,
  `test_md5`), enforced by `scripts/audit_integrity.py` and
  `tests/test_data_integrity.py`.
- **Target construction**: `target_aqi` is the *next calendar day's* AQI within
  each city's own continuous calendar, so no target is created by shifting
  across a missing date.
- **Scaler**: one `StandardScaler` fitted on the training split only and reused
  unchanged by training, evaluation and serving (ADR D-020). The bundle stores
  the fitted scaler.
- **Lags and rolling windows** use `shift(1)` semantics, so no feature at day
  *t* can see day *t+1*.
- **Feature list** is read from `data/processed/feature_list.json`; counts and
  order are never hard-coded in training or serving.
- **Selection**: every model, configuration and calibration decision is made on
  the validation split. The holdout is evaluated exactly once by
  `scripts/final_test_eval.py`, which refuses to run if a split hash or the
  champion artifact hash has changed since the report was written (ADR A1 plan
  item, `src/models/final_eval.py`).
- **Metamorphic testing**: the validation fixture is perturbed (`fire ×5`,
  `wind ÷3`) and the prediction change is checked for direction and magnitude.
  Under the monotone champion the non-monotone rate is **0.0%** for both
  checks, with mean ΔAQI `+2.387` (fire) and `+0.605` (wind).
- **Integrity**: the serving bundle is hash-verified *before* unpickling
  (`AqiBundle.load(verify=True)`), and `MANIFEST.json` records payload hashes,
  the DVC data md5, the params hash, the risk-card hash and the git commit
  (ADR D-032).

---

## 5. Performance

### 5.1 Headline metrics

| Split | MAE | RMSE | R² | Skill vs persistence | Notes |
|---|---|---|---|---|---|
| Train | — | 21.921 | — | — | Diagnostics only, not a selection metric |
| Validation (calibrated OOF) | 13.233 | 20.610 | 0.8977 | 0.0874 | Selection split; forward-chained calibration |
| Validation (raw, uncalibrated) | — | 20.340 | — | — | Basis for the monotone acceptance rule |
| **Holdout (single run)** | **14.028** | **23.886** | **0.8772** | **0.0640** | 95% weekly block-bootstrap CI `[0.0364, 0.0882]` |

Persistence baseline on the holdout: MAE 14.720, RMSE 25.518, R² 0.8598.
The champion beats persistence on every headline metric, and the whole
skill confidence interval is above zero.

### 5.2 Per-city holdout error

| City | Model MAE | Persistence MAE | Ratio | Status |
|---|---|---|---|---|
| Ahmedabad | 11.794 | 12.411 | 0.9503 | pass |
| Bengaluru | 4.844 | 5.240 | 0.9243 | pass |
| Chennai | 11.158 | 11.802 | 0.9454 | pass |
| Delhi | 21.578 | 24.540 | 0.8793 | pass |
| Hyderabad | 19.077 | 19.694 | 0.9687 | pass |
| Kolkata | 14.593 | 13.771 | **1.0597** | **fail (G3, waived)** |
| Lucknow | 13.368 | 14.196 | 0.9417 | pass |
| Mumbai | 15.003 | 16.193 | 0.9265 | pass |
| Pune | 7.238 | 7.495 | 0.9657 | pass (thin training history) |

### 5.3 Overfitting and stability evidence

```
train RMSE 21.921   val RMSE 20.610   gap -1.311   ratio 0.9402
TimeSeriesSplit CV RMSE 24.879 ± 2.949 over 4 folds
interpretation: no material overfitting; validation RMSE is within 10% of train RMSE
```

The negative train/validation gap reflects the residual model fitting a wider
target distribution in the training period; the fold spread is reported so the
point estimate is not read as certainty. These numbers are computed from
train/validation only — the holdout is not used to judge fit.

### 5.4 Latency

| Path | p50 | p95 | p99 | Budget |
|---|---|---|---|---|
| In-process single row (`risk card`, 200 iterations) | 7.65 ms | **7.90 ms** | 8.28 ms | < 50 ms |
| Live `uvicorn` single row (server-reported, 100 iterations) | 12.42 ms | **13.20 ms** | 13.85 ms | < 50 ms |
| Live `uvicorn` 9-city batch (wall clock, 30 iterations) | 68.68 ms | **76.36 ms** | 79.17 ms | < 100 ms |

Reports: [`reports/metrics/latency.json`](../reports/metrics/latency.json) and
[`reports/metrics/latency_probe.json`](../reports/metrics/latency_probe.json).
Latency is hardware-dependent; the budget is a coarse regression guard.

---

## 6. Prediction interval and `category_probs` method

**Prediction interval.** The bundle stores per-city residual quantiles
`q10`, `q50`, `q90` (with a `__global__` fallback), estimated on the validation
split. The 80% interval is

```
lower  = clip(pred + q10[city], 0, aqi_max)
median = clip(pred + q50[city], 0, aqi_max)
upper  = clip(pred + q90[city], 0, aqi_max)
```

The interval is **not** a calibrated posterior and the quantiles are not
re-estimated online. It reflects the validation-period error distribution, so
it can be too narrow in new regimes (for example an unprecedented stagnation
episode) and too wide in calm periods.

**`category_probs`.** There is no classifier artifact in the champion. The
service converts the interval mass over the CPCB band edges into a category
distribution, assuming a normal residual with

```
sigma = (q90 - q10) / 2.5631031310892     # normal-equivalent 10-90 spread
P(band) = Phi((upper_edge - pred) / sigma) - Phi((lower_edge - pred) / sigma)
```

where `Phi` is the standard normal CDF, the edges are the CPCB AQI cut-offs
(50, 100, 200, 300, 400), the first and last bands are open-ended, and the
vector is renormalised to sum to 1. This is an **approximation**: it inherits
the residual quantiles' normality assumption and cannot express asymmetric or
heavy-tailed risk. It is documented here because Plan Phase 3 requires the
method to be stated explicitly, and it is flagged in the API response as
`category_probs` rather than presented as a calibrated classifier.

The separate `regress_then_bucket` category path uses the same CPCB cut-offs
applied to `aqi_pred`; on validation it reached a within-one-category rate of
0.9991 and a macro F1 of 0.649. On the holdout the within-one-category rate is
0.9951.

---

## 7. Gate status

| Gate | Rule | Value | Status |
|---|---|---|---|
| G1 | Skill vs persistence > 0 on val and test | val 0.1026; test 0.0640 `[0.0364, 0.0882]` | pass |
| G2 | Test R² ≥ 0.65 and test MAE ≤ 30 | 0.8772; 14.028 | pass |
| G3 | Model MAE ≤ persistence MAE × 1.05 in **every** city | max ratio 1.0597 (Kolkata) | **fail — waived** |
| G4 | Within-one-category ≥ 0.95; Poor+ recall reported | 0.9951 | pass |
| G5 | p95 single-row latency < 50 ms | 7.90 ms in-process; 13.20 ms live | pass |
| A2 | Monotone candidate within 2% of incumbent val RMSE; non-monotone ≤ 5% | +1.20% raw; 0.0% fire and wind | pass |
| A4 | Overfitting diagnostics on train/validation only | gap −1.311, ratio 0.9402, CV 24.879 ± 2.949 | pass |

**G3 waiver (D-023, reaffirmed by D-038).** Kolkata is 14.593 MAE versus
persistence 13.771 (ratio 1.0597), above the frozen 1.05 threshold. The
threshold was not changed. The champion was frozen on validation, the holdout
was run once, and no post-test tuning or city-specific patch was applied. The
failure is recorded in the bundle `MANIFEST.json`, the risk card and the MLflow
version. Kolkata is the only failing city; uniform per-city calibration had
already improved the ratio from 1.0922 to 1.0597.

---

## 8. Known limitations

1. **Pune has a thin history** (900 training rows versus ~1,820 for other
   cities, ADR D-022). Its low holdout MAE (7.238) is measured on an easier
   AQI regime, not proof of better generalisation.
2. **Weather is reanalysis, not forecast.** Training uses ERA5 reanalysis, while
   production will use current or forecast weather. This train/serving skew can
   shift errors in fast-changing synoptic conditions.
3. **Vehicle features are annual state totals interpolated to daily**, not
   city-registered counts; `vehicles_per_lakh` is an emission-pressure proxy.
4. **Industrial features are placeholders.** `iip_manufacturing_index` is empty
   in the current snapshot and median-filled; it carries no signal.
5. **The fire proxy is coarse.** A 500 km radius aggregate mixes local and
   transported smoke and cannot identify sources; monotonicity is a structural
   prior, not evidence of causation.
6. **Category probabilities are approximate** (Section 6) and assume normal
   residuals; they should not be used as risk probabilities.
7. **Interval calibration is static.** Residual quantiles come from the 2024
   validation period and are not updated online; drift is not detected
   automatically.
8. **Missingness is non-random.** Station outages and sensor drift are not
   modelled; city means hide intra-city heterogeneity.
9. **Policy calendars are curated, not live.** GRAP stages and festival flags
   come from project files and may lag official changes.
10. **G3 is red for Kolkata.** Reported, waived, and not hidden.
11. **Regime shift.** COVID-19 lockdowns are inside the training window, and no
    mechanism guarantees accuracy under unprecedented emission changes.

---

## 9. Ethical and safety notes

- **No PII.** Inputs are public environmental, meteorological, registration and
  policy data.
- **No causal claims.** Feature importance and monotone directions describe
  associations in this panel; they are not source-apportionment evidence.
- **Health communication.** AQI categories are informational. The model must not
  be the sole basis for medical advice, school closures, or enforcement.
- **Failure modes to monitor:** missing or stale weather/fire feeds, an
  out-of-range `aqi` input, a city outside the nine supported values, a
  corrupted or unverified bundle (the service refuses to start), and drift in
  the residual distribution. Out-of-range inputs are flagged in the response
  `warnings` and the `degraded` flag; the service returns 400 for impossible
  values and 503 when the bundle is not ready.
- **Fallback.** If the champion cannot be loaded, the bundle ships a
  persistence fallback; responses report `fallback_used=true` and
  `degraded=true` rather than silently substituting a different model.

---

## 10. Reproduction

```bash
# Full pipeline (data -> features -> train -> evaluate -> final_test_eval
#                 -> model_risk -> package -> register)
dvc repro
dvc status                      # expect: Data and pipelines are up to date.

# Verification
python -m pytest tests -q                                  # 162 tests
python -m ruff check src tests scripts
python -m black --check .
python scripts/audit_integrity.py                          # 64/64
python scripts/verify_registry.py                          # 423 rows, max |diff| 0.0
python -m src.models.register_model --verify               # 423 rows, max |diff| 0.0
python scripts/latency_probe.py                            # live uvicorn p50/p95
```

**Pinned hashes**

| Artifact | Hash |
|---|---|
| `data/processed/train.parquet` | md5 `49bd4c88748327afa12057c2fe71b0e7` |
| `data/processed/val.parquet` | md5 `462b7712b4cea342fe36ca80868a29f4` |
| `data/processed/test.parquet` | md5 `bd1de9ea97083402bd9ccaceb1072e3c` |
| `models/residual_lightgbm_monotone.joblib` | md5 `728d68b88f7108cc0aa957a1c137a3cd` |
| `artifacts/champion/bundle.joblib` | sha256 `3bdc05c84ce0c2213aabd4894f17cfb25adf3d3aeba7986950fa927d4af77f8d` |
| `artifacts/champion/MANIFEST.json` | sha256 `148a5b4fd5c997d2aca670557571363643b41943ef81f2440c3984e3f7a6e9fb` |
| Step-018 commit | `5d6ff1538100863868ddeca3701d4d6138e2b3ce` |

The bundle was built by the pipeline at step-017 HEAD `5c71a34` and committed
as step 018. `MANIFEST.json` records the build-time commit, the DVC data md5,
the params hash and the risk-card hash so a rebuilt bundle can be compared
byte-for-byte.

Runtime: Python 3.12.7, pandas 2.3.3, numpy 1.26.4, scikit-learn 1.5.1,
lightgbm 4.6.0, joblib 1.4.2, pyarrow 22.0.0, mlflow 3.17.1.

---

## 11. Change history

| Date | Change | Reference |
|---|---|---|
| 2026-11-12 | Gate 3: monotone fire/wind constraints, uniform per-city calibration, single-shot holdout report; G3 improves to 1.0597 and stays waived | [Step 018](logbook/018-gate3-monotone-and-calibration.md), D-030/D-031/D-038 |
| 2026-11-10 | Phase 2 champion bundle, registry promotion and G3 waiver | [Step 012](logbook/012-registry-and-bundling.md), D-023 |
