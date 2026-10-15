# Step 006 — Hourly→Daily Aggregation, CPCB AQI Computation, and EDA

## Metadata
| Field | Value |
|---|---|
| **Step** | 006 |
| **Phase** | 4 — Data Processing and Exploration (EDA) |
| **Date** | 2026-10-15 |
| **Author** | <student-A> |
| **Commit** | `41ecc50` |
| **Status** | COMPLETE |

---

## What changed

### 1. Hourly → Daily Panel (`src/data/hourly_to_daily.py`)
- **Input:** Raw OpenAQ gzipped CSVs under `data/raw/openaq/` — 12 prioritised stations across 8 cities (Bengaluru, Delhi, Mumbai, Chennai, Kolkata, Hyderabad, Pune, Ahmedabad).
- Pivoted narrow hourly rows to wide `(station, hour)` format; standardised parameter names (`pm25`, `pm10`, `no2`, `so2`, `co`, `o3`, `nh3`, `nox`, `no`); applied unit conversions (all to µg/m³; CO to mg/m³ per CPCB table).
- Converted timestamps to `Asia/Kolkata`; defined the calendar day by IST local date.
- Daily aggregation: 24h mean for `pm25`, `pm10`, `no2`, `so2`, `nh3`; rolling 8h max for `co` and `o3` per `params.yaml:aggregation.rolling_8h_pollutants`. Required ≥ 16 valid hourly readings (`params.yaml:aggregation.min_hours_per_day`) for a daily value to count.
- Computed CPCB sub-indices and `AQI = max sub-index` using breakpoints from `configs/cpcb_breakpoints.yaml`. AQI validity: ≥3 pollutant sub-indices available **and** at least one of PM2.5 or PM10 (`params.yaml:aqi`).
- **Output:** `data/interim/pollutants_daily.parquet` — **13,763 station-days**, 37 columns.

### 2. EDA (`scripts/run_eda.py`, 15 figures → `reports/figures/eda/`)
| Fig | Title | Key Finding |
|---|---|---|
| 01 | Station Coverage Gantt | 12 stations, all active 2019–2026; Bengaluru has 3 stations. |
| 02 | Rows per City per Year | Stable volume 2019–2025; 2026 partial (Jan–Sep). |
| 03 | Missingness per Pollutant | NH3=100%, NOx=75%, NO=58% → dropped. PM2.5=36%, PM10=37%. |
| 04 | PM2.5 Missing by Station×Year | Ahmedabad high in 2019–2020; Bengaluru most complete. |
| 05 | Pollutant Distributions | All right-skewed; log-scale; winsorisation required. |
| 06 | AQI Category Distribution | Moderate dominates (>35%); Good rare (<5%); severe in Delhi. |
| 07 | AQI Time Series (5 cities) | Delhi winter spikes >300; Bengaluru moderate year-round. |
| 08 | Month×City Seasonality Heatmap | All cities peak Nov–Feb; monsoon (Jun–Sep) lowers AQI. |
| 09 | Day-of-Week Effect | Slight weekend AQI drop in Delhi; minimal elsewhere. |
| 10 | Weather vs PM2.5 | Wind speed negatively correlates; humidity slightly positive. |
| 11 | Pollutant Correlation Matrix | PM2.5↔AQI r≈0.85; wind speed negatively correlated. |
| 12 | ACF/PACF (Bengaluru AQI) | Significant autocorrelation at lags 1, 2, 3, 7. |
| 13 | Dominant Pollutant per City | PM2.5 dominates Delhi/Kolkata; PM10 in Ahmedabad/Chennai. |
| 14 | Bengaluru Spotlight | BTM Layout and Railway Station differ from Peenya (industrial). |
| 15 | AQI Change Distribution | Mean |ΔAQI| ≈ 28 AQI units/day; persistence baseline quantified. |

**`docs/eda_summary.md`** written with 15 findings and decisions table.

---

## Stage Stats (`reports/metrics/hourly_to_daily_stats.json`)

| Metric | Value |
|---|---|
| Raw hourly rows read | 6,257,997 |
| Dropped: invalid_value | 6,285 |
| Dropped: unsupported_param | 1,212,329 |
| Dropped: duplicate | 3,428,749 |
| Total dropped | 4,647,363 |
| Station-days produced | **13,763** |
| Valid AQI station-days | **9,565 (69.5%)** |
| Unique cities | 8 |
| Date range | 2019-01-03 → 2026-10-01 |
| Output file | `data/interim/pollutants_daily.parquet` |
| Columns | 37 |

---

## Commands to verify

```bash
python -m src.data.hourly_to_daily           # re-run stage
python scripts/run_eda.py                    # regenerate all EDA figures
python -c "import pandas as pd; df=pd.read_parquet('data/interim/pollutants_daily.parquet'); print(df.shape)"
# Expected: (13763, 37)
ls reports/figures/eda/                      # should show 15 .png files
```

---

## Decisions recorded

1. **NH3, NOx, NO dropped:** >70% missing across all stations; cannot form reliable sub-indices. Recorded in `docs/decisions.md`.
2. **AQI validity threshold respected:** require ≥3 sub-indices, at least PM2.5 or PM10 present (`params.yaml:aqi`).
3. **Class imbalance noted:** classification models must use `class_weight='balanced'`; evaluation uses macro-F1.
4. **Season and festival flags added** to feature engineering based on EDA finding of strong seasonal and Diwali-period AQI spikes.
5. **Lag features [1,2,3,7]** confirmed by ACF/PACF analysis (Fig 12).

---

## Logbook protocol checklist
- [x] Stage stats written to `reports/metrics/hourly_to_daily_stats.json`
- [x] `docs/data_change_log.md` updated
- [x] 15 EDA figures saved to `reports/figures/eda/`
- [x] `docs/eda_summary.md` written (15 findings, decisions table)
- [x] Logbook entry created
- [x] `docs/logbook/INDEX.md` updated
