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

## hourly_to_daily — 2026-10-15

| Field | Value |
|---|---|
| Stage | `hourly_to_daily` |
| Start | 2026-10-15T12:10:59.829599+00:00 |
| End | 2026-10-15T12:10:59.837081+00:00 |
| Duration | 0.0 s |
| Git commit | `6d40ad3` |
| Params hash | `197d7666db10` |
| Rows in | 0 |
| Rows out | 4,064 |
| Total dropped | 1,501,639 |

**Rows dropped by reason:**
  - `unsupported_param`: 458,128 rows
  - `duplicate`: 1,043,511 rows

**Columns added:** `aqi`, `aqi_category`, `city`, `co`, `date`, `dominant_pollutant`, `is_aqi_valid`, `latitude`, `location_id`, `longitude`, `n_hours_co`, `n_hours_nh3`, `n_hours_no`, `n_hours_no2`, `n_hours_nox`, `n_hours_o3`, `n_hours_pm10`, `n_hours_pm25`, `n_hours_so2`, `nh3`, `no`, `no2`, `nox`, `o3`, `pm10`, `pm25`, `so2`, `station_name`, `sub_co`, `sub_nh3`, `sub_no`, `sub_no2`, `sub_nox`, `sub_o3`, `sub_pm10`, `sub_pm25`, `sub_so2`
**Columns removed:** none

---

## merge — 2026-10-15

| Field | Value |
|---|---|
| Stage | `merge` |
| Start | 2026-10-15T12:12:18.161925+00:00 |
| End | 2026-10-15T12:12:18.172220+00:00 |
| Duration | 0.0 s |
| Git commit | `6d40ad3` |
| Params hash | `197d7666db10` |
| Rows in | 4,064 |
| Rows out | 4,064 |
| Total dropped | 0 |

**Rows dropped by reason:**
  - (none)

**Columns added:** `precipitation_sum`, `relative_humidity_2m_mean`, `surface_pressure_mean`, `temperature_2m_max`, `temperature_2m_mean`, `temperature_2m_min`, `wind_direction_10m_mean`, `wind_speed_10m_mean`
**Columns removed:** none

---

## clean — 2026-10-15

| Field | Value |
|---|---|
| Stage | `clean` |
| Start | 2026-10-15T12:12:59.173938+00:00 |
| End | 2026-10-15T12:12:59.189172+00:00 |
| Duration | 0.0 s |
| Git commit | `6d40ad3` |
| Params hash | `197d7666db10` |
| Rows in | 4,064 |
| Rows out | 2,871 |
| Total dropped | 1,193 |

**Rows dropped by reason:**
  - `missing_target_aqi`: 1,193 rows

**Columns added:** `co_was_missing`, `no2_was_missing`, `o3_was_missing`, `pm10_was_missing`, `pm25_was_missing`, `so2_was_missing`
**Columns removed:** `nh3`, `sub_nh3`, `sub_no`, `sub_nox`

---

## features — 2026-10-15

| Field | Value |
|---|---|
| Stage | `features` |
| Start | 2026-10-15T12:15:50.411643+00:00 |
| End | 2026-10-15T12:15:50.440322+00:00 |
| Duration | 0.0 s |
| Git commit | `6d40ad3` |
| Params hash | `49d91b162fbf` |
| Rows in | 2,871 |
| Rows out | 1,052 |
| Total dropped | 0 |

**Rows dropped by reason:**
  - (none)

**Columns added:** `aqi_diff1`, `aqi_lag_1`, `aqi_lag_2`, `aqi_lag_3`, `aqi_lag_7`, `aqi_rolling_max_14`, `aqi_rolling_max_3`, `aqi_rolling_max_7`, `aqi_rolling_mean_14`, `aqi_rolling_mean_3`, `aqi_rolling_mean_7`, `aqi_rolling_std_14`, `aqi_rolling_std_3`, `aqi_rolling_std_7`, `city_Delhi`, `city_mean_aqi_other_stations`, `co_lag_1`, `co_lag_2`, `co_lag_3`, `co_lag_7`, `cos_day_of_year`, `cos_month`, `day_of_week`, `day_of_year`, `is_festival`, `is_weekend`, `month`, `no2_lag_1`, `no2_lag_2`, `no2_lag_3`, `no2_lag_7`, `o3_lag_1`, `o3_lag_2`, `o3_lag_3`, `o3_lag_7`, `pm10_lag_1`, `pm10_lag_2`, `pm10_lag_3`, `pm10_lag_7`, `pm25_diff1`, `pm25_lag_1`, `pm25_lag_2`, `pm25_lag_3`, `pm25_lag_7`, `pm25_pm10_ratio`, `pm25_rolling_max_14`, `pm25_rolling_max_3`, `pm25_rolling_max_7`, `pm25_rolling_mean_14`, `pm25_rolling_mean_3`, `pm25_rolling_mean_7`, `pm25_rolling_std_14`, `pm25_rolling_std_3`, `pm25_rolling_std_7`, `precipitation_sum_lag1`, `precipitation_sum_roll3_mean`, `relative_humidity_2m_mean_lag1`, `relative_humidity_2m_mean_roll3_mean`, `season_monsoon`, `season_post_monsoon`, `season_summer`, `season_winter`, `sin_day_of_year`, `sin_month`, `so2_lag_1`, `so2_lag_2`, `so2_lag_3`, `so2_lag_7`, `surface_pressure_mean_lag1`, `surface_pressure_mean_roll3_mean`, `target_aqi`, `target_category`, `temperature_2m_mean_lag1`, `temperature_2m_mean_roll3_mean`, `wind_speed_10m_mean_lag1`, `wind_speed_10m_mean_roll3_mean`
**Columns removed:** none

---

## hourly_to_daily — 2026-10-15

| Field | Value |
|---|---|
| Stage | `hourly_to_daily` |
| Start | 2026-10-15T12:23:24.079083+00:00 |
| End | 2026-10-15T12:23:24.096544+00:00 |
| Duration | 0.0 s |
| Git commit | `6d40ad3` |
| Params hash | `49d91b162fbf` |
| Rows in | 0 |
| Rows out | 13,763 |
| Total dropped | 4,647,363 |

**Rows dropped by reason:**
  - `invalid_value`: 6,285 rows
  - `unsupported_param`: 1,212,329 rows
  - `duplicate`: 3,428,749 rows

**Columns added:** `aqi`, `aqi_category`, `city`, `co`, `date`, `dominant_pollutant`, `is_aqi_valid`, `latitude`, `location_id`, `longitude`, `n_hours_co`, `n_hours_nh3`, `n_hours_no`, `n_hours_no2`, `n_hours_nox`, `n_hours_o3`, `n_hours_pm10`, `n_hours_pm25`, `n_hours_so2`, `nh3`, `no`, `no2`, `nox`, `o3`, `pm10`, `pm25`, `so2`, `station_name`, `sub_co`, `sub_nh3`, `sub_no`, `sub_no2`, `sub_nox`, `sub_o3`, `sub_pm10`, `sub_pm25`, `sub_so2`
**Columns removed:** none

---
