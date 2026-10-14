# Dataset Card: Urban India Air Quality & Weather Reanalysis Panel

**Project:** AI-Based Air Quality Awareness Assistant  
**Milestone:** 1 — MLOps Data Foundation & Baselines  
**Snapshot / Freeze Date:** October 15, 2026  
**Temporal Window:** `2019-01-01` to `2026-09-30` (~93 months)  
**Authors:** Student A & Student B (Course AI252TA: Machine Learning Operations)  

---

## 1. Dataset Summary & Problem Formulation

This dataset comprises an hourly station-day panel built by pairing surface atmospheric pollutant observations from the **OpenAQ AWS S3 Open Data archive** with surface atmospheric weather variables from the **Open-Meteo Historical Weather API (ERA5 Reanalysis)**.

### Target Formulations
- **Task A (Regression):** Predict next-day continuous National Air Quality Index (AQI, 0–500) for a given monitoring station.
- **Task B (Classification):** Predict next-day CPCB AQI Category (*Good: 0–50, Satisfactory: 51–100, Moderate: 101–200, Poor: 201–300, Very Poor: 301–400, Severe: 401–500*).

Target AQIs are derived programmatically using the official Central Pollution Control Board (CPCB) National AQI algorithm implemented in `src/features/aqi_utils.py` and calibrated with breakpoints in `configs/cpcb_breakpoints.yaml`.

---

## 2. Data Sources & Provenance

| Stream | Provider | Source URL / Bucket | Access Protocol | License & Rights |
|---|---|---|---|---|
| **Pollutants** | **OpenAQ Archive** (aggregating Central Pollution Control Board & State Pollution Control Boards) | `s3://openaq-data-archive/records/csv.gz/` | AWS S3 anonymous (`--no-sign-request`) | Open Data Commons / Government Open Data License - India (GODL) |
| **Station Discovery** | **OpenAQ API v3** | `https://api.openaq.org/v3` | REST API via registered key (`OPENAQ_API_KEY`) | CC BY 4.0 |
| **Weather** | **Open-Meteo Historical API** | `https://archive-api.open-meteo.com/v1/archive` | REST API (yearly cached chunks) | Creative Commons Attribution 4.0 International (CC BY 4.0) |

---

## 3. Scope & Selected Monitoring Stations

A station is incorporated into the modeling network only if it meets all of the following quality gates:
1. Located within targeted metropolitan urban agglomerations (mandatory anchor: **Bengaluru**; additional: Delhi, Mumbai, Chennai, Kolkata, Hyderabad, Pune).
2. Contains active monitors for particulate matter (**PM2.5** or **PM10**).
3. Active recording span of $\ge 3.0$ years across the study window.

### Network Composition
- **Total Selected Stations:** 24 monitoring stations
- **Bengaluru Stations (8):** BTM Layout, Hebbal, Peenya, Silk Board, BWSSB Kadabesanahalli, City Railway Station, Saneguruvanahalli, Jayanagar.
- **Delhi Stations (5):** Anand Vihar, R K Puram, Punjabi Bagh, Mandir Marg, IHBAS Dilshad Garden.
- **Mumbai Stations (3):** Bandra Kurla Complex (BKC), Colaba, Worli.
- **Chennai Stations (2):** Alandur Bus Depot, Velachery Residential Area.
- **Kolkata Stations (2):** Victoria Memorial, Jadavpur.
- **Hyderabad Stations (2):** Sanathnagar, Nehru Zoo Park.
- **Pune Stations (2):** Shivajinagar, Karve Road.

Full metadata catalog available in `configs/stations_selected.csv`.

---

## 4. Schema & Parameters

### Raw OpenAQ Pollutant Observations (`data/raw/openaq/`)
Narrow-format gzipped CSV files organized as:
`records/csv.gz/locationid=<id>/year=<yyyy>/month=<mm>/location-<id>-<yyyymmdd>.csv.gz`

| Column | Type | Unit | Description |
|---|---|---|---|
| `location_id` | Integer | — | Unique OpenAQ monitoring station identifier |
| `sensors_id` | Integer | — | Identifier for the specific physical sensor component |
| `location` | String | — | Station descriptive name |
| `datetime` | String (ISO 8601) | UTC | Observation timestamp (e.g., `2023-01-01T00:00:00+00:00`) |
| `lat` | Float | Degrees | Latitude coordinate |
| `lon` | Float | Degrees | Longitude coordinate |
| `parameter` | String | — | Pollutant metric (`pm25`, `pm10`, `no2`, `so2`, `co`, `o3`, `nh3`) |
| `units` | String | — | Measurement unit (`µg/m³`, `mg/m³`) |
| `value` | Float | — | Measured atmospheric concentration |

### Raw Weather Reanalysis (`data/raw/weather/`)
Columnar Parquet files per station and year: `weather_<location_id>_<year>.parquet`

| Variable | Unit | Source Model | Description |
|---|---|---|---|
| `time` | String (ISO) | `Asia/Kolkata` | Local hourly timestamp |
| `temperature_2m` | °C | ERA5 Reanalysis | 2-meter air temperature |
| `relative_humidity_2m` | % | ERA5 Reanalysis | 2-meter relative humidity |
| `dew_point_2m` | °C | ERA5 Reanalysis | 2-meter atmospheric dew point |
| `surface_pressure` | hPa | ERA5 Reanalysis | Atmospheric surface pressure |
| `precipitation` | mm | ERA5 Reanalysis | Hourly liquid precipitation amount |
| `wind_speed_10m` | km/h | ERA5 Reanalysis | Horizontal wind speed at 10 meters |
| `wind_direction_10m` | ° | ERA5 Reanalysis | Wind direction in meteorological degrees (0–360°) |
| `boundary_layer_height` | m | ERA5 Reanalysis | Planetary boundary layer height |

---

## 5. Dataset Versions & Manifest Tracking

| Version | Date | Description | Files Tracked | Manifest Path |
|---|---|---|---|---|
| `data-v1` | 2026-10-15 | Initial snapshot: 24 Indian stations across 7 cities, 2019–2026 window | ~66,960 partitions | `data/raw/openaq_manifest.csv`, `data/raw/weather_manifest.csv` |

All raw and interim artifacts are cryptographically hashed (SHA256) in their respective manifests, ensuring reproducible pipeline states under DVC.

---

## 6. Known Limitations & Potential Biases

1. **Station Outages & Irregular Sampling:** Monitors undergo maintenance, power disruptions, or calibration downtime, creating non-random temporal missingness.
2. **Sensor Drift:** Optical particle counters and electrochemical gas sensors degrade over time without regular recalibration by municipal bodies.
3. **Reanalysis Weather vs. Microclimate:** ERA5 is gridded atmospheric reanalysis (~31 km spatial resolution) rather than in-situ station thermometers; localized urban heat island effects and street-canyon wind deflections are smoothed out.
4. **COVID-19 Lockdown Dynamics (2020–2021):** Industrial and vehicular restrictions in early 2020 created an atypical dip in urban emissions. While included in the training distribution, models must not over-rely on lockdown anomalies.
5. **Absence of Point-Source Emissions:** Traffic density feeds, industrial stack emissions, and farm residue burning satellite detections (e.g., VIIRS fire spots) are not included directly in Milestone 1.

---

## 7. Ethical Considerations & Appropriate Use

* **Privacy & Surveillance:** Zero personally identifiable information (PII) is gathered or processed. Data consists entirely of public outdoor atmospheric readings.
* **Intended Use:** Academic and research demonstration of MLOps pipelines, time-series forecasting, and public health awareness tools.
* **Non-Intended / Prohibited Use:** **Not designed for acute medical or clinical decision-making**, industrial compliance enforcement, or legal air quality litigation.

---

## 8. Attribution & Citations

* **OpenAQ:** OpenAQ Open Data Program. AWS Open Data Archive `s3://openaq-data-archive`. Data provided courtesy of Central Pollution Control Board (CPCB) and State Pollution Control Boards of India.
* **Open-Meteo:** Open-Meteo Weather API (ERA5 Reanalysis data provided by the Copernicus Climate Change Service / ECMWF). Licensed under CC BY 4.0.
* **CPCB Methodology:** Central Pollution Control Board, Ministry of Environment, Forest and Climate Change, Government of India: *National Air Quality Index (NAQI) Breakpoints and Calculation Guidelines*.
