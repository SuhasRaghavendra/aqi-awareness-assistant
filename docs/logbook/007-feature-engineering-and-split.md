# Step 007 — Feature engineering and chronological split
- **Date/time:** 2026-10-15T18:15:00+05:30
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** 5           **Branch / PR:** `main` / #7
- **Git commit(s):** `9cc4b9a`      **Data version:** `data-v1`
- **MLflow run IDs:** N/A

---

## What changed

### 1. Cleaning (`src/data/clean.py`)
- **Input:** `data/interim/station_day.parquet` — 4,064 rows (after weather merge restricted to Delhi stations with weather data).
- Sorted by `(station, date)`. Dropped duplicate station-days.
- Dropped station-days with missing AQI (target undefined) — 1,193 rows dropped.
- Replaced impossible values with NaN.
- **Winsorisation:** fit lower/upper quantiles (0.005, 0.995) **exclusively on training period** (≤2022-12-31); applied clip to full dataset.
- Dropped columns with >70% missing fraction (NH3, NOx, NO, sub-indices for those).
- Imputation: forward-fill per station up to 2 days; then fill with station training-period median; added `<col>_was_missing` flags for main pollutants and weather.
- **Output:** `data/interim/station_day_clean.parquet`.

### 2. Feature Engineering (`src/features/build_features.py`)
All features use only information available at day *t* to predict day *t+1*. Computed per station on a continuous calendar index (prevents lag shifting across date gaps).

| Feature Group | Details |
|---|---|
| **Target** | `target_aqi = AQI.shift(-1)` per station; `target_category = aqi_to_bucket(target_aqi)` |
| **Lags** | AQI, PM2.5, PM10, NO2, O3, CO, SO2 × lags [1, 2, 3, 7] days |
| **Rolling** | Mean/std/max over [3, 7, 14] days for AQI and PM2.5 (past values only) |
| **Differences** | `aqi_diff1`, `pm25_diff1` |
| **Weather** | Same-day + previous-day + 3-day rolling means; temperature, humidity, pressure, precipitation, wind speed/direction |
| **Calendar** | Day-of-week, month, day-of-year, weekend flag, sin/cos cyclic encodings, season (winter/summer/monsoon/post-monsoon), festival flag |
| **Ratios** | PM2.5/PM10 ratio (guard zero-division) |
| **Spatial** | City one-hot; leave-one-out city-mean AQI at day t (excluding station itself) |
| **Station code** | Numeric station ID for tree models |

**Total features: 114 numeric columns.**

### 3. Chronological Split
| Split | Date Range | Rows |
|---|---|---|
| **Train** | 2019-01-01 → 2022-12-31 | **1,052** |
| **Val** | 2023-01-01 → 2025-12-31 | **832** |
| **Test** | 2026-01-01 → 2026-09-30 | **685** |

**Assertion:** `max(train_date) < min(val_date) < min(test_date)` ✓

**Train class balance:**
| Category | Count |
|---|---|
| Moderate | 316 |
| Very Poor | 274 |
| Satisfactory | 167 |
| Poor | 139 |
| Severe | 134 |
| Good | 22 |

### 4. Scaling
`StandardScaler` fit on train only → `models/scaler.joblib`. Applied to numeric feature columns.

### 5. Feature ranking figure
`reports/figures/features/feature_importance_train.png` — top 30 features by coefficient of variation on training set (proxy for variability-based importance), colour-coded by feature group.

---

## Stage Stats (`reports/metrics/features_stats.json`)

| Metric | Value |
|---|---|
| Input rows (station_day_clean) | 4,064 |
| Train rows | 1,052 |
| Val rows | 832 |
| Test rows | 685 |
| Total feature columns | 114 |
| Target (regression) | `target_aqi` |
| Target (classification) | `target_category` |

---

## Decisions recorded

1. **Split dates match params.yaml:** `train_end = 2022-12-31`, `val_end = 2025-12-31`. Test = 2026+.
2. **Weather merge restricted station scope:** only stations with matching weather files are in `station_day.parquet`. With the available weather data covering Delhi stations, the split reflects Delhi-only rows. This is documented in `docs/decisions.md`. Future work: expand weather coverage to all 8 cities.
3. **No data leakage:** winsorisation clipping limits, scaler parameters, and station-training-period medians are all fit exclusively on training rows.
4. **Festival dates loaded from `configs/festival_dates.yaml`** (Diwali, Dussehra, Holi). Source cited in config file.
5. **City mean AQI leave-one-out** calculated per station per day using training statistics to avoid station self-leakage.

---

## Commands to verify

```bash
python -m src.data.clean                     # re-run clean stage
python -m src.features.build_features       # re-run features stage
pytest tests/ -q                            # all 71 tests pass
python -c "
import pandas as pd
tr = pd.read_parquet('data/processed/train.parquet')
va = pd.read_parquet('data/processed/val.parquet')
te = pd.read_parquet('data/processed/test.parquet')
assert tr['date'].max() < va['date'].min()
assert va['date'].max() < te['date'].min()
print('Splits OK:', tr.shape, va.shape, te.shape)
"
```

---

## Logbook protocol checklist
- [x] Stage stats written (`reports/metrics/features_stats.json`)
- [x] `docs/data_change_log.md` updated
- [x] Feature ranking figure saved (`reports/figures/features/feature_importance_train.png`)
- [x] 71 pytest tests pass (includes leakage tests, split ordering, AQI band boundary tests)
- [x] Scaler saved to `models/scaler.joblib`
- [x] Feature list saved to `data/processed/feature_list.json`
- [x] Logbook entry created
- [x] `docs/logbook/INDEX.md` updated
