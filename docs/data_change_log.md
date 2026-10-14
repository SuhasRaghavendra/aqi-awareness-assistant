# Data Change Log

> Auto-appended by `src/utils/stage_stats.py` after every data-transforming stage.
> Each section records: stage name, rows in/out, rows dropped (with reasons),
> columns added/removed, imputation counts, and a one-line summary.
> **Do not edit manually** — treat as an append-only audit trail.

---

## discover_stations — 2026-10-14

| Field | Value |
|---|---|
| Stage | `discover_stations` |
| Start | 2026-10-14T18:45:59.268300+00:00 |
| End | 2026-10-14T18:45:59.268300+00:00 |
| Duration | 0.0 s |
| Git commit | `6d00fbe` |
| Params hash | `542931e8d09b` |
| Rows in | 0 |
| Rows out | 0 |
| Total dropped | 0 |

**Rows dropped by reason:**
  - (none)

**Columns added:** none
**Columns removed:** none

---

## discover_stations — 2026-10-14

| Field | Value |
|---|---|
| Stage | `discover_stations` |
| Start | 2026-10-14T18:47:48.783315+00:00 |
| End | 2026-10-14T18:47:49.784728+00:00 |
| Duration | 1.0 s |
| Git commit | `6d00fbe` |
| Params hash | `542931e8d09b` |
| Rows in | 27 |
| Rows out | 24 |
| Total dropped | 3 |

**Rows dropped by reason:**
  - `Excluded (not meeting geographic scope or minimum criteria)`: 3 rows

**Columns added:** none
**Columns removed:** none

---

## download_openaq — 2026-10-14

| Field | Value |
|---|---|
| Stage | `download_openaq` |
| Start | 2026-10-14T19:32:04.357340+00:00 |
| End | 2026-10-14T19:32:04.364227+00:00 |
| Duration | 0.0 s |
| Git commit | `afca63f` |
| Params hash | `542931e8d09b` |
| Rows in | 3 |
| Rows out | 12 |
| Total dropped | 0 |

**Rows dropped by reason:**
  - (none)

**Columns added:** `download_time`, `local_path`, `month`, `s3_key`, `sha256`, `size_bytes`, `status`, `year`
**Columns removed:** `active_years`, `datetime_first_utc`, `datetime_last_utc`, `has_pm`, `has_pm10`, `has_pm25`, `latitude`, `locality`, `longitude`, `name`, `parameter_units`, `parameters`, `provider`, `selected`, `selection_status`, `sensor_count`, `target_city`

---

## download_weather — 2026-10-14

| Field | Value |
|---|---|
| Stage | `download_weather` |
| Start | 2026-10-14T19:33:26.141101+00:00 |
| End | 2026-10-14T19:33:26.146632+00:00 |
| Duration | 0.0 s |
| Git commit | `afca63f` |
| Params hash | `542931e8d09b` |
| Rows in | 3 |
| Rows out | 6 |
| Total dropped | 0 |

**Rows dropped by reason:**
  - (none)

**Columns added:** `download_time`, `end_date`, `local_path`, `model_source`, `rows_count`, `sha256`, `size_bytes`, `start_date`, `status`, `year`
**Columns removed:** `active_years`, `datetime_first_utc`, `datetime_last_utc`, `has_pm`, `has_pm10`, `has_pm25`, `latitude`, `locality`, `longitude`, `name`, `parameter_units`, `parameters`, `provider`, `selected`, `selection_status`, `sensor_count`, `target_city`

---

## validate — 2026-10-14

| Field | Value |
|---|---|
| Stage | `validate` |
| Start | 2026-10-14T19:48:46.506857+00:00 |
| End | 2026-10-14T19:48:46.512051+00:00 |
| Duration | 0.0 s |
| Git commit | `c112576` |
| Params hash | `542931e8d09b` |
| Rows in | 1,824 |
| Rows out | 52,632 |
| Total dropped | 0 |

**Rows dropped by reason:**
  - (none)

**Columns added:** `boundary_layer_height`, `dew_point_2m`, `precipitation`, `relative_humidity_2m`, `surface_pressure`, `temperature_2m`, `time`, `wind_direction_10m`, `wind_speed_10m`
**Columns removed:** `datetime`, `lat`, `location`, `lon`, `parameter`, `sensors_id`, `units`, `value`

---

## validate — 2026-10-14

| Field | Value |
|---|---|
| Stage | `validate` |
| Start | 2026-10-14T19:50:27.095767+00:00 |
| End | 2026-10-14T19:50:27.100391+00:00 |
| Duration | 0.0 s |
| Git commit | `c112576` |
| Params hash | `542931e8d09b` |
| Rows in | 1,824 |
| Rows out | 52,632 |
| Total dropped | 0 |

**Rows dropped by reason:**
  - (none)

**Columns added:** `boundary_layer_height`, `dew_point_2m`, `precipitation`, `relative_humidity_2m`, `surface_pressure`, `temperature_2m`, `time`, `wind_direction_10m`, `wind_speed_10m`
**Columns removed:** `datetime`, `lat`, `location`, `lon`, `parameter`, `sensors_id`, `units`, `value`

---
