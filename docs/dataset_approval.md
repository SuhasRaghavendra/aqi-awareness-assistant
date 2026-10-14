# Dataset Acquisition & Approval Request

**Project:** AI-Based Air Quality Awareness Assistant  
**Course:** AI252TA — Machine Learning Operations (Milestone 1)  
**Authors:** Student A & Student B (2-person team)  
**Date Submitted:** October 15, 2026  
**Status:** Pending Faculty Approval  

---

## 1. Problem Statement & Scope

Air pollution poses a severe public health hazard across urban India. While raw pollutant concentrations are monitored, the public relies on the simplified National Air Quality Index (AQI) to gauge health impact. This project establishes an end-to-end MLOps foundation to:
- **Task A (Regression):** Predict next-day continuous AQI ($0 \le \text{AQI} \le 500$) for monitoring stations in Indian metropolitan centers.
- **Task B (Classification):** Forecast the next-day official CPCB AQI Category (*Good, Satisfactory, Moderate, Poor, Very Poor, Severe*).

Milestone 1 delivers data acquisition, validation, CPCB AQI calculation, DVC versioning, feature engineering, baseline models, and MLflow experiment tracking.

---

## 2. Primary Data Sources & Access

| Data Stream | Primary Provider / Source | Access Protocol | License & Attribution |
|---|---|---|---|
| **Pollutants (Hourly)** | **OpenAQ Archive** (AWS Open Data Program, bucket `openaq-data-archive`, region `us-east-1`) | Anonymous S3 access (`--no-sign-request`) via `boto3` / AWS CLI. Narrow-format gzipped CSVs partitioned by `locationid=<id>/year=<yyyy>/month=<mm>/`. | OpenAQ Data License. Original measurements sourced from Central Pollution Control Board (CPCB) and State Pollution Control Boards (SPCBs) under India Open Government Data (OGD) license. |
| **Station Discovery** | **OpenAQ API v3** (`/v3/locations`, `/v3/sensors`) | REST API with free registration key (`OPENAQ_API_KEY`). | Open Data Commons / CC BY 4.0. |
| **Weather Enrichment** | **Open-Meteo Historical Weather API** (ERA5 Reanalysis) | REST API, free tier (non-commercial, ~10,000 calls/day), cached locally. Hourly surface metrics: temperature, humidity, wind speed/direction, surface pressure, precipitation. | Creative Commons Attribution 4.0 International (CC BY 4.0). Open-Meteo attribution included in repo README and documentation. |

---

## 3. Geographic Scope & Selected Monitoring Network

* **Primary Anchor City:** **Bengaluru** (Mandatory local focus).
* **Additional Cities:** Delhi, Mumbai, Chennai, Kolkata, Hyderabad, Pune, Ahmedabad, Lucknow.
* **Selection Criteria:**
  1. Geographic relevance: Station situated in identified urban agglomeration.
  2. Critical pollutant tracking: Station measures at least one particulate metric (**PM2.5** or **PM10**).
  3. Historical continuity: Minimum **3.0 years** of active data recording.
  4. Temporal window: Active within the study period (`2019-01-01` to `2026-09-30`).

The automated discovery process generated the frozen station list in `configs/stations_selected.csv`:
* **Total Discovered Indian Stations:** 27
* **Stations Meeting Quality Criteria:** 24 stations (Bengaluru: 8, Delhi: 5, Mumbai: 3, Chennai: 2, Kolkata: 2, Hyderabad: 2, Pune: 2).

---

## 4. Dataset Dimensions & Volume Estimates

* **Study Period:** January 1, 2019 → September 30, 2026 (~93 calendar months).
* **Estimated Raw File Count:** ~66,960 gzipped CSV partitions across selected stations and months.
* **Estimated Raw Data Size:** ~1.1 GB compressed (~8.4 GB uncompressed hourly data).
* **Target Modelling Table:** Station-day panel of ~65,000 to ~70,000 rows across 24 stations, with 40+ lag, rolling, weather, and calendar features.

---

## 5. Target Definition: CPCB National AQI

Rather than trusting pre-computed third-party indices, target AQIs are calculated directly from raw pollutant hourly readings using the official Central Pollution Control Board (CPCB) formula (`src/features/aqi_utils.py` and `configs/cpcb_breakpoints.yaml`):
1. **Aggregations:** 24-hour mean for PM2.5, PM10, NO2, SO2, NH3; 8-hour running mean daily maximum for CO and O3.
2. **Validity Rules:** Daily average requires $\ge 16$ valid hourly measurements ($\ge 75\%$). Station-day AQI requires sub-indices for $\ge 3$ pollutants, at least one of which must be PM2.5 or PM10.
3. **Composite Index:** $\text{AQI} = \max(\text{Sub-indices})$.

---

## 6. Train/Val/Test Split Strategy

To strictly avoid lookahead bias and temporal leakage in time-series forecasting:
* **Train Set:** `2019-01-01` to `2023-12-31` (~65% of rows). All scalers, imputers, and clipping quantiles are fit exclusively on this window.
* **Validation Set:** `2024-01-01` to `2024-12-31` (~15% of rows) for hyperparameter tuning and model selection.
* **Test Set:** `2025-01-01` to `2026-09-30` (~20% of rows), evaluated strictly once at milestone completion.

---

## 7. Risk Analysis & Mitigation

1. **Station Outages / Missing Data:** Mitigated via station-level forward-filling (limit 2 days) and station training medians, with explicit `<pollutant>_was_missing` binary indicators.
2. **Unit Discrepancies:** All pollutants standardized to $\mu\text{g/m}^3$ (and CO to $\text{mg/m}^3$ per CPCB standard) with schema validation assertions in `src/data/validate.py`.
3. **COVID-19 Anomaly (2020–2021):** Lockdowns induced atypical air quality. Included in training data to reflect variance, while the test set (2025–2026) evaluates performance on normalized post-pandemic conditions.
4. **Ethics & Privacy:** All data comprises public ambient atmospheric readings; zero personally identifiable information (PII) is accessed or stored.

---

## 8. Faculty Review & Approval Record

| Field | Detail |
|---|---|
| **Reviewer / Faculty Member** | *[Pending Review]* |
| **Review Date** | *[Pending]* |
| **Decision** | **[ APPROVED / REJECTED / REVISIONS REQUESTED ]** |
| **Faculty Comments** | *Pending submission* |

*(Attach screenshot/PDF copy of approval email upon receipt.)*
