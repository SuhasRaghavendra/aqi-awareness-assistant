# Data Card — AI-Based Air Quality Awareness Assistant

**Document version:** 1.0  
**Phase:** 1 — Dataset Acquisition & Validation  
**Snapshot date:** 2026-10-14  
**Prepared by:** Student A + Student B

---

## 1. Dataset Summary

| Field | Value |
|---|---|
| Dataset name | India AQI Station-Day Panel (MLOps Milestone 1) |
| Task | Predict next-day CPCB AQI (regression + 6-class classification) |
| Primary source | OpenAQ archive on AWS S3 (`s3://openaq-data-archive`) |
| Enrichment source | Open-Meteo Historical Weather (ERA5 reanalysis) |
| Study window | 2019-01-01 to 2026-09-30 |
| Download / snapshot date | 2026-10-14 |
| Primary city | Bengaluru (mandatory focus) |
| Other cities | Delhi, Mumbai, Chennai, Kolkata, Hyderabad, Pune, Ahmedabad, Lucknow |

---

## 2. Sources and Access

### 2.1 OpenAQ Pollutant Archive

| Field | Value |
|---|---|
| Source | OpenAQ Air Quality Archive |
| URL | https://openaq.org |
| S3 URI | `s3://openaq-data-archive/records/csv.gz/locationid=<id>/year=<yyyy>/month=<mm>/` |
| AWS region | `us-east-1` |
| Access method | Anonymous S3 (boto3 with `UNSIGNED` config; no AWS account required) |
| File format | Gzip-compressed CSV (narrow: one row per sensor reading) |
| Station discovery | OpenAQ API v3 `/v3/locations` (India country_id=9) |
| API key | `OPENAQ_API_KEY` in `.env` (not committed) |
| **License** | **Varies by provider.** CPCB-affiliated government data: India's National Data Sharing and Accessibility Policy (NDSAP — open government data). Research stations (e.g., SPARTAN): provider-specific open data licenses. Record the license of each provider seen in `units_per_parameter` in the validation report. |
| Attribution | "Data sourced from OpenAQ (https://openaq.org), which aggregates data from government monitoring networks including India's Central Pollution Control Board (CPCB) and partner research programs." |

### 2.2 Open-Meteo Historical Weather

| Field | Value |
|---|---|
| Source | Open-Meteo Historical Weather API |
| URL | https://open-meteo.com |
| Endpoint | `https://archive-api.open-meteo.com/v1/archive` |
| Model | ERA5 reanalysis (European Centre for Medium-Range Weather Forecasts, ECMWF) |
| Access method | No API key; free for non-commercial use (fair-use: ~10,000 calls/day) |
| Variables | temperature_2m (°C), relative_humidity_2m (%), surface_pressure (hPa), precipitation (mm), wind_speed_10m (m/s), wind_direction_10m (°) |
| Timezone | Asia/Kolkata (IST) |
| Caching | Per-station per-year JSON files in `data/raw/weather/<location_id>/<year>.json` |
| **License** | **CC BY 4.0** — Creative Commons Attribution 4.0 International |
| Attribution | "Weather data by Open-Meteo.com (https://open-meteo.com), ERA5 reanalysis provided by ECMWF, licensed under CC BY 4.0." |

---

## 3. Dataset Version Table

| Tag | Date | What changed | Raw size | File count | Notes |
|---|---|---|---|---|---|
| `data-v1` | 2026-10-15 | Baseline frozen raw snapshot: OpenAQ pollutants & Open-Meteo ERA5 weather for 24 selected stations (2019–2026) | ~36.5 MB raw archives | ~24 station partitions | Tracked via `openaq.dvc`, `weather.dvc`, `station_catalog.csv.dvc`; SHA-256 logged in manifests |
| `data-v2` | 2026-10-15 | Simulated network refresh: added monitoring station partition & updated selection catalog | + 1 station partition | + 1 file | Genuine versioning demonstration via `scripts/simulate_data_v2.py`; verified with `dvc diff` |

> Updated after each DVC push. MD5 hashes recorded inside the `.dvc` tracking files serve as the authoritative dataset version identifiers.

---

## 4. Station Selection

### 4.1 Selection Criteria

| Criterion | Threshold | Source |
|---|---|---|
| City must be in target list | 9 cities | `params.yaml:data:cities` |
| Data span in study window | ≥ 3 years | `params.yaml:data:min_years_per_station` |
| Must have PM2.5 or PM10 sensor | At least one | Implementation plan §2.3 |
| Proxy completeness (span / window) | ≥ 60% | `params.yaml:data:min_hourly_completeness` |

### 4.2 Station Counts (after discovery run)

> **Populated from `reports/metrics/station_coverage.json`** after running `discover_stations.py`.

| City | Total discovered | Selected | Has PM2.5 | Has PM10 |
|---|---|---|---|---|
| (see station_coverage.json) | | | | |

**Frozen station list:** `configs/stations_selected.csv`  
**Full catalog with exclusion reasons:** `data/raw/station_catalog.csv`

### 4.3 Notes on Coverage

- OpenAQ's locality field is `None` for most Indian stations; city is inferred from the station name string (e.g., "Delhi Technological University, Delhi - CPCB" → Delhi).
- The CPCB provider dominates Indian coverage. SPARTAN research stations add a small number of locations but often lack date information.
- Bengaluru has approximately 10–15 monitoring stations in the OpenAQ database; not all have long-running PM2.5/PM10 data.
- Only stations with `datetimeFirst` and `datetimeLast` populated could be assessed for temporal span; stations with `None` dates were excluded unless they have known coverage.

---

## 5. Raw Files

### 5.1 OpenAQ

| Field | Value |
|---|---|
| Directory | `data/raw/openaq/` |
| Manifest | `data/raw/openaq_manifest.csv` |
| Manifest fields | s3_key, local_path, size_bytes, sha256, download_time_s, timestamp |
| Resumable | Yes — skips already-downloaded files |
| Parallel workers | 4 threads (configurable via `--workers`) |

> File count and total size populated after download run.

### 5.2 Weather

| Field | Value |
|---|---|
| Directory | `data/raw/weather/<location_id>/<year>.json` |
| Manifest | `data/raw/weather_manifest.csv` |
| Manifest fields | location_id, year, cache_path, size_bytes, sha256, n_rows, variables_returned, timestamp |
| Caching | Each station-year cached permanently; re-runs are instant |

---

## 6. Parameters and Units

### 6.1 OpenAQ Pollutants (as observed in API)

| Parameter | CPCB name | Averaging period | Expected unit | Conversion needed |
|---|---|---|---|---|
| pm25 | PM₂.₅ | 24-hour mean | µg/m³ | None if already µg/m³; ppb→µg/m³ requires T, P (document assumptions) |
| pm10 | PM₁₀ | 24-hour mean | µg/m³ | Same as above |
| no2 | NO₂ | 24-hour mean | µg/m³ or ppb | ppb × 1.88 ≈ µg/m³ at 25°C, 1013 hPa |
| so2 | SO₂ | 24-hour mean | µg/m³ or ppb | ppb × 2.62 ≈ µg/m³ at 25°C, 1013 hPa |
| co | CO | 8-hr running mean (daily max) | µg/m³ or ppm | ppm × 1000/22.4 × M_CO ≈ mg/m³ (documented in aqi_utils.py) |
| o3 | O₃ | 8-hr running mean (daily max) | µg/m³ or ppb | ppb × 1.96 ≈ µg/m³ at 25°C, 1013 hPa |
| nh3 | NH₃ | 24-hour mean | µg/m³ or ppb | ppb × 0.70 ≈ µg/m³ at 25°C, 1013 hPa |

> Unit conversion assumptions (T=25°C, P=1013 hPa, ideal gas law) are approximations. Actual station measurements are at varying ambient conditions. This is noted as a limitation.

> **Actual units per parameter** are recorded in `reports/metrics/data_validation.json:stats:units_per_parameter` after the validation run.

### 6.2 Open-Meteo Weather Variables

| Variable | Unit | Description |
|---|---|---|
| temperature_2m | °C | Air temperature at 2m height |
| relative_humidity_2m | % | Relative humidity at 2m height |
| surface_pressure | hPa | Atmospheric pressure at surface |
| precipitation | mm | Total precipitation per hour |
| wind_speed_10m | m/s | Wind speed at 10m height |
| wind_direction_10m | ° | Wind direction (meteorological convention) |

---

## 7. Target Definition

**Computed AQI is NOT provided by OpenAQ.** It is derived from raw concentrations using the CPCB National AQI method:

| Step | Detail |
|---|---|
| Daily aggregation | PM2.5, PM10, NO₂, SO₂, NH₃ → 24h arithmetic mean; CO, O₃ → max of 8h running means |
| Validity per pollutant-day | Requires ≥ 16 valid hourly readings (75%); parameterised in `params.yaml:aggregation:min_hours_per_day` |
| Sub-index | Linear interpolation within CPCB breakpoint bands (`configs/cpcb_breakpoints.yaml`) |
| AQI | Maximum sub-index across all valid pollutants |
| Station-day validity | AQI computed only if ≥ 3 sub-indices available AND at least one is PM2.5 or PM10 |
| Categories | Good (0–50), Satisfactory (51–100), Moderate (101–200), Poor (201–300), Very Poor (301–400), Severe (401–500) |

**Breakpoint source:** Central Pollution Control Board, "National Air Quality Index" (official document).  
**Citation:** CPCB, MoEFCC, Government of India. *National Air Quality Index.* Available at: https://cpcb.nic.in/displaypdf.php?id=eHViZC9OQVFJLUVuZ2xpc2gudGlmeQ==  
**Verification:** Every breakpoint in `configs/cpcb_breakpoints.yaml` must be cross-checked against the official document. [HUMAN: cross-check and confirm]

---

## 8. Data Split Strategy

| Split | Period | Approx. fraction | Use |
|---|---|---|---|
| Train | 2019-01-01 to 2023-12-31 | ~60% | Fit all models, imputers, scalers |
| Validation | 2024-01-01 to 2024-12-31 | ~15% | Hyperparameter tuning, early stopping |
| Test | 2025-01-01 to 2026-09-30 | ~25% | Final evaluation (touched once) |

Split dates are configurable in `params.yaml:split`. **Chronological only — no shuffle.**

---

## 9. Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| CPCB sensors have calibration variation between stations | Bias in absolute AQI values | Model per-station or include station embedding |
| Bengaluru has ~10–15 stations, some with data gaps | Limited spatial coverage | Include all passing-threshold stations; note limitation |
| ERA5 reanalysis is gridded (0.25° × 0.25°), not true point measurements | Weather features are approximate | Document; adequate for daily-scale models |
| Unit conversions (ppb→µg/m³) assume fixed T and P | Small error in sub-index calculation | State assumptions; perform sensitivity analysis if time permits |
| COVID-19 anomalies in 2020–21 (lockdown-induced AQI drops) | Training data contains unusual patterns | These years are in training; model must generalize to post-2023 |
| Some station names are ambiguous for city matching | Incorrect city assignment | Manual review of `configs/stations_selected.csv` recommended [HUMAN] |
| OpenAQ files may be retroactively patched by providers | Snapshot integrity risk | sha256 in manifest; snapshot date recorded here |
| No traffic or emissions data | Key AQI drivers missing | Noted; a known limitation for the first milestone |
| No fire/wildfire smoke data | Extreme events under-predicted | Noted; a future data source to add |

---

## 10. Ethical Note

- All data is **publicly available environmental monitoring data** from government and research sources.
- **No personal data** is collected or processed; AQI readings are aggregate measurements.
- The dataset is intended for **air quality awareness and research** purposes only.
- **Not for medical diagnosis or individual health decisions.** Model predictions should not substitute for professional medical or environmental safety advice.
- Data from Indian government monitoring networks is shared under India's open government data policy.
- Open-Meteo data is shared under CC BY 4.0; proper attribution is included in this card and in `README.md`.

---

## 11. Attribution Text (for README and reports)

```
Pollutant data: OpenAQ (https://openaq.org), aggregating measurements from
India's Central Pollution Control Board (CPCB) and partner networks.
License: varies by provider (CPCB data: India NDSAP open government data).

Weather data: Open-Meteo (https://open-meteo.com), ERA5 reanalysis from ECMWF.
License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

AQI computation: CPCB National AQI method.
Source: Central Pollution Control Board, Ministry of Environment, Forest and
Climate Change, Government of India.
URL: https://cpcb.nic.in/displaypdf.php?id=eHViZC9OQVFJLUVuZ2xpc2gudGlmeQ==
```

---

## 12. Validation Results

> **Populated from `reports/metrics/data_validation.json`** after running `src/data/ingest.py`.

| Check | Result |
|---|---|
| Schema errors | **0** (All required columns present, valid schema) |
| Warnings | 4 (60,682 post-snapshot rows, O₃ ppm unit variant, 64,027 negatives, 1,731 sentinels) |
| OpenAQ files loaded | 108,303 daily archive files |
| OpenAQ rows loaded | 50,166,474 rows |
| Weather files loaded | 659 station-year cache files |
| Weather rows loaded | 5,595,768 rows across 84 stations |
| Parameters found | 12 (`pm25`, `pm10`, `no2`, `so2`, `co`, `o3`, `no`, `nox`, `temperature`, `relativehumidity`, `wind_direction`, `wind_speed`) |
| Negative values | 64,027 rows (to be handled in cleaning stage) |
| Sentinel values | 1,731 rows (to be replaced with NaN in cleaning stage) |
| Future timestamps | 0 |

---

*End of Data Card*
