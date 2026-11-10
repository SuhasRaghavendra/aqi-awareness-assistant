# Dataset Card: Urban India Air Quality, Fires & Weather Panel

**Project:** AI-Based Air Quality Awareness Assistant
**Milestone:** 2 - MLOps Baseline (Phase 0)
**Snapshot / Freeze Date:** November 10, 2026
**Temporal Window:** `2019-01-01` to `2026-09-30`
**Authors:** Student A & Student B (Course AI252TA: Machine Learning Operations)

---

## 1. Dataset Summary & Problem Formulation

This dataset is a **city x day panel** for nine Indian cities. It combines AQI
labels derived from the XKDR India Air Quality Database (a harmonised view of
CPCB CAAQM data) with root-cause predictors: NASA FIRMS fire detections,
Open-Meteo ERA5 meteorology, state-level vehicle registrations, policy/festival
calendars, and static city attributes.

Unlike the earlier Milestone 1 station panel, the modelling unit is the **city
day**. AQI is computed from city PM2.5/PM10 sub-indices using the CPCB method and
averaged across reporting stations for each city-day.

### Target Formulations
- **Task A (Regression):** Predict next-day continuous city AQI (0-500).
- **Task B (Classification):** Predict next-day CPCB AQI category (*Good 0-50,
  Satisfactory 51-100, Moderate 101-200, Poor 201-300, Very Poor 301-400,
  Severe 401-500*).

Targets are built with a per-city continuous calendar (`make_target`), so a
target is never produced by shifting across a missing date. Day-t `aqi` is an
explicit feature and is also the anchor for the residual candidate models
(ADR D-019, D-021).

---

## 2. Data Sources & Provenance

| Stream | Provider | Source URL / Path | Access Protocol | Rights / Notes |
|---|---|---|---|---|
| **AQI labels** | XKDR India Air Quality Database (CPCB CAAQM harmonised) | `https://airquality.xkdr.org` | REST API; `XKDR_API_KEY` | CPCB-derived; query PM2.5 and PM10 and compute AQI locally |
| **Fire detections** | NASA FIRMS VIIRS SNPP (Standard Processing + NRT) | `https://firms.modaps.eosdis.nasa.gov/api/` | REST API; `FIRMS_MAP_KEY` | NASA open data |
| **Weather** | Open-Meteo Historical Weather API (ERA5 reanalysis) | `https://archive-api.open-meteo.com/v1/archive` | REST, no key | CC BY 4.0 (Copernicus/ECMWF) |
| **Vehicles** | data.gov.in / MoRTH Vahan state registrations | `https://data.gov.in/`; fallback `configs/vehicle_state_reference.csv` | REST API; `DATAGOV_API_KEY` or static CSV | GODL; annual, state-level |
| **Industry** | MOSPI IIP and CEA state power | Manual CSVs under `configs/` | Manual download | GODL; **empty in the current snapshot** |
| **Policy / calendars** | GRAP and festival calendars | `configs/grap_calendar.csv`, `configs/festival_dates.yaml` | Curated files | Project-maintained |
| **City attributes** | Latitude, longitude, elevation, population | `src/data/ingest_and_merge.py` | Static constants | Public reference values |

Raw artifacts are downloaded by `src/data/download_*.py` and pinned by DVC
(`dvc.lock`); the raw sources are never edited in place.

---

## 3. Scope & City Coverage

Nine cities are modelled:

| City | State | Train rows (2019-2023) | Val rows (2024) | Test rows (2025+) | Valid AQI city-days |
|---|---|---|---|---|---|
| Delhi | Delhi | 1,826 | 365 | 154 | 2,346 |
| Mumbai | Maharashtra | 1,826 | 366 | 215 | 2,408 |
| Bengaluru | Karnataka | 1,826 | 365 | 154 | 2,346 |
| Chennai | Tamil Nadu | 1,826 | 366 | 243 | 2,436 |
| Kolkata | West Bengal | 1,826 | 366 | 205 | 2,398 |
| Hyderabad | Telangana | 1,826 | 366 | 419 | 2,612 |
| Pune | Maharashtra | **900** | 365 | 154 | 1,469 |
| Ahmedabad | Gujarat | 1,820 | 365 | 154 | 2,340 |
| Lucknow | Uttar Pradesh | 1,826 | 365 | 154 | 2,346 |
| **Total** | | **15,502** | **3,289** | **1,852** | **20,701** |

- The chronological split follows ADR D-004/D-018: train <= 2023-12-31,
  validation 2024, test >= 2025-01-01.
- The test window is `2025-01-01` to `2026-03-26` and contains all nine cities.
- **Pune is a known weak city (ADR D-022):** its history starts in 2021, so it
  has 900 train rows versus roughly 1,820-1,826 for other cities. Per-city
  results for Pune should be read with that sample-size caveat.

---

## 4. Schema & Parameters

### 4.1 Raw AQI labels (`data/raw/aqi/city_aqi_daily.parquet`, 20,701 rows)

| Column | Type | Unit | Description |
|---|---|---|---|
| `date` | Date | - | Calendar day |
| `city` | String | - | One of the nine modelled cities |
| `aqi` | Float | AQI points (0-500) | City-day mean AQI from PM2.5/PM10 CPCB sub-indices |
| `station_count` | Integer | - | Number of reporting stations contributing to the city-day |

Station-level PM observations are retained in
`data/raw/aqi/station_aqi_raw.parquet` but are not the modelling spine.

### 4.2 Fire features (`data/raw/fires/city_fires_daily.parquet`)

| Column | Description |
|---|---|
| `fire_count`, `fire_frp_sum`, `fire_frp_mean` | Detections within the state/region aggregate |
| `fire_dist_min_km` | Distance from city centroid to the nearest detection |
| `fire_count_500km`, `fire_frp_sum_500km` | Detections and fire radiative power within 500 km of the city |

About 6.34 million raw VIIRS detections are aggregated to 19,466 city-days.
The 500 km radius captures crop-residue burning and regional transport; it is a
proxy, not source apportionment.

### 4.3 Weather features (`data/raw/weather/city_weather_daily.parquet`)

Daily aggregates for each city in `Asia/Kolkata`:

`temperature_2m_mean`, `temperature_2m_max`, `temperature_2m_min`,
`relative_humidity_2m_mean`, `wind_speed_10m_mean`,
`wind_direction_10m_dominant`, `precipitation_sum`, `surface_pressure_mean`,
plus derived `inversion_risk_flag`, `fog_risk_flag`, `rain_washout_flag`.

### 4.4 Vehicle registrations (`data/raw/vehicles/city_vehicles_annual.parquet`)

45 state-year rows with `total_vehicles`, fuel-type counts and fractions
(`diesel_frac`, `cng_frac`, `ev_frac`, `petrol_frac`, `two_wheeler_frac`),
`population_lakh`, and `vehicles_per_lakh`. Values are state-level annual totals
mapped to cities and interpolated to daily; they are emission-pressure proxies,
not city-registered vehicle counts.

### 4.5 Industrial proxies

`data/raw/industrial/iip_monthly.parquet` and
`data/raw/industrial/state_power_monthly.parquet` have the expected schemas but
contain **0 rows in the current snapshot** because the MOSPI/CEA manual CSVs are
not present. Consequently `iip_manufacturing_index` is 100% missing in the
merged panel and is median-filled (0.0) during feature construction. Industrial
features should therefore be treated as placeholders until the source files are
added.

### 4.6 Merged and cleaned panel

`data/interim/city_day_raw.parquet` has 25,470 city-day rows and 43 columns;
`data/interim/city_day_clean.parquet` has 24,739 rows and 44 columns after
adding `aqi_category`. The cleaned panel contains:

- identifiers and target: `date`, `city`, `aqi`, `aqi_category`, `station_count`;
- fire: `fire_count_500km`, `fire_frp_sum_500km`, `fire_dist_min_km`;
- weather: the nine ERA5 variables plus three derived flags;
- vehicles: four fuel fractions, `vehicles_per_lakh`;
- industry: `iip_manufacturing_index` (currently all missing);
- policy: `grap_stage`, `odd_even_active`, `bsvi_active`;
- calendar: `festival_flag`, `day_of_week`, `month`, `day_of_year`, `year`,
  `weekend_flag`, `burning_season_flag`, `season`, sin/cos encodings;
- static: `city_lat`, `city_lon`, `city_elevation_m`, `city_population_lakh`.

AQI is missing on 16.5% of cleaned city-days; after constructing the next-day
target the modelling panel contains 20,643 rows.

### 4.7 Engineered features (`data/processed/feature_list.json`, 103 features)

`n_features` is read from the manifest; it is never hard-coded. Groups:

| Group | Count | Examples |
|---|---|---|
| Fire / burning | 26 | fire lags and 7/14/30-day rolling means/maxima, fire x season |
| Meteorological | 26 | weather lags, wind rolling means, inversion/fog/rain flags |
| Vehicle emissions | 8 | fuel fractions, vehicles per lakh, diesel intensity |
| Industrial | 2 | IIP placeholder, stagnation index components |
| Policy / governance | 6 | GRAP active/severe/lag, odd-even, BS-VI |
| AQI history | 12 | day-t `aqi`, lags 1/3/7/14, rolling mean/std, 1/7-day diffs |
| Calendar / seasonal | 15 | month, day-of-year, weekend, festival, season one-hots |
| Spatial | 16 | city one-hots, lat/lon, elevation, population |
| Interactions | 5 | fire x low wind, inversion x fire, festival overlap, etc. |

The single `StandardScaler` in `data/processed/scaler.joblib` is fitted on the
train split only and is reused by train and evaluate (ADR D-020).

---

## 5. Dataset Versions & Manifest Tracking

| Snapshot | Date | Description | Pinning |
|---|---|---|---|
| Current | 2026-11-10 | 9-city panel, 2019-2026, 103 features, ADR D-018 split | `dvc.lock` + DagsHub DVC remote |

Verified hashes in `dvc.lock`:

| Artifact | MD5 |
|---|---|
| `data/processed/feature_list.json` | `a994715d33a1a3e9cd929d34916dfe99` |
| `data/processed/train.parquet` | `49bd4c88748327afa12057c2fe71b0e7` |
| `data/processed/test.parquet` | `bd1de9ea97083402bd9ccaceb1072e3c` |

The DVC graph is `download_* -> ingest_and_merge -> validate -> clean ->
features -> train -> evaluate -> model_risk -> register`; data and the full
model set are stored on the DagsHub remote and are not committed to Git. The
small champion bundle in `artifacts/champion/` is tracked with DVC
`cache: false` and is committed to Git for CI and serving.

---

## 6. Known Limitations & Potential Biases

1. **Pune data thinness:** 900 train rows versus ~1,820-1,826 for other cities;
   its per-city error estimates have wider uncertainty (ADR D-022).
2. **Industrial features unavailable:** MOSPI IIP and CEA power files are empty
   in this snapshot, so `iip_manufacturing_index` is median-filled and carries no
   signal. Industrial source data must be added before using those features.
3. **Annual, state-level vehicles:** registrations are interpolated to daily and
   are not urban-only. `vehicles_per_lakh` is a relative emission-pressure proxy.
4. **Reanalysis weather vs forecast weather:** training uses ERA5 reanalysis,
   while production predictions will use current/forecast weather. This
   train/serving skew should be stated alongside any deployed model.
5. **Coarse fire proxy:** the 500 km radius aggregate mixes local and transported
   smoke and cannot identify individual sources.
6. **Missingness and sensor drift:** station outages, calibration downtime, and
   sensor degradation create non-random missingness; city means can hide
   intra-city heterogeneity.
7. **AQI gaps:** 16.5% of cleaned city-days lack AQI, and next-day targets
   require consecutive dates, reducing the modelling panel to 20,643 rows.
8. **COVID-19 lockdown dynamics (2020-2021):** atypical emission reductions are
   inside the training distribution and should not be over-generalised.
9. **Curated policy calendars:** GRAP stages and festival flags come from project
   calendars rather than a live feed.
10. **No point-source emissions:** traffic counts, stack emissions, and explicit
    source apportionment are not included.

---

## 7. Ethical Considerations & Appropriate Use

- **Privacy:** no personally identifiable information is processed; all inputs
  are public environmental, meteorological, registration, or policy data.
- **Intended use:** academic demonstration of MLOps pipelines, time-series
  forecasting, and public air-quality awareness.
- **Not intended for:** acute medical or clinical decisions, industrial
  compliance enforcement, or legal air-quality litigation.

---

## 8. Attribution & Citations

- **XKDR India Air Quality Database / CPCB CAAQM:** city AQI labels derived from
  CPCB PM2.5 and PM10 observations.
- **NASA FIRMS:** Fire Information for Resource Management System, VIIRS SNPP
  375 m active-fire detections.
- **Open-Meteo:** Historical Weather API, ERA5 reanalysis from the Copernicus
  Climate Change Service / ECMWF, CC BY 4.0.
- **data.gov.in / MoRTH Vahan:** state-level vehicle registration statistics.
- **MOSPI:** Index of Industrial Production; **CEA:** state power requirement and
  availability statistics (source files pending).
- **CPCB:** National Air Quality Index breakpoints and calculation guidelines.
