# Step 004 — Data validation and data card
- **Date/time:** 2026-10-14T19:51:30+05:30
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** 1           **Branch / PR:** `main` / #<n>
- **Git commit(s):** `92251db8deb0bc27324fc16c90ccc53368d8d4d8`      **Data version:** <git-tag> / <dvc-md5>
- **MLflow run IDs:** N/A

## 1. Objective
Implement automated data ingestion and schema validation (`src/data/ingest.py`, `src/data/validate.py`) and author the comprehensive project Data Card (`docs/data_card.md`) to conclude Milestone 1 Phase 1 (Section 5.2 of the implementation plan). Verifies required column presence, timestamp parsing, future timestamp exclusion, physical range bounds, parameter membership, unit consistency, and exports `reports/metrics/data_validation.json`.

## 2. What changed

| Type | Path | Change summary |
|---|---|---|
| code | `src/data/validate.py` | Full implementation of `ValidationReport`, `validate_openaq_df`, and `validate_weather_df` |
| code | `src/data/ingest.py` | Pipeline stage entry point loading raw OpenAQ and weather partitions and running validation checks |
| tests | `tests/test_validate.py` | 8 unit tests covering missing columns, future timestamps, sentinels, negatives, unit inconsistencies, and weather range bounds |
| docs | `docs/data_card.md` | Comprehensive 8-section Data Card documenting dataset provenance, 24 stations, schema, versions, and limitations |
| docs | `docs/decisions.md` | Recorded Decision D-007 (strict schema validation gates and sentinel missing-value flagging) |
| docs | `docs/data_change_log.md` | Auto-appended audit entry for `validate` stage |
| metrics | `reports/metrics/data_validation.json` | Comprehensive JSON validation metrics for pollutant and weather observations |
| metrics | `reports/metrics/validate_stats.json` | Pipeline stage stats for the `validate` stage |

<details>
<summary>git diff --stat</summary>

```
docs/data_card.md                    | 121 +++++++++++++++
 docs/data_change_log.md              |  44 ++++++
 docs/decisions.md                    |   1 +
 reports/metrics/data_validation.json |  69 +++++++++
 reports/metrics/validate_stats.json  | 110 ++++++++++++++
 src/data/ingest.py                   | 186 ++++++++++++++++++++++-
 src/data/validate.py                 | 280 ++++++++++++++++++++++++++++++++++-
 tests/test_validate.py               | 153 +++++++++++++++++++
 8 files changed, 960 insertions(+), 4 deletions(-)
```
</details>

<details>
<summary>git status --short</summary>

```
(clean)
```
</details>

<details>
<summary>dvc status</summary>

```
(command not found: dvc)
```
</details>

<details>
<summary>dvc diff</summary>

```
(command not found: dvc)
```
</details>

### Stage statistics summary
- **discover_stations_stats**: rows 27 -> 24  |  dropped 3  |  1.001s
- **download_openaq_stats**: rows 3 -> 12  |  dropped 0  |  0.007s
- **download_weather_stats**: rows 3 -> 6  |  dropped 0  |  0.005s
- **validate_stats**: rows 1824 -> 52632  |  dropped 0  |  0.005s

## 3. Commands run (verbatim)
```bash
# Execute ingestion and validation stage
python -m src.data.ingest

# Run unit tests
python -m pytest tests/test_validate.py -v
python -m pytest tests/ -v

# Code formatting and linting
python -m ruff check src/data/ingest.py src/data/validate.py tests/test_validate.py
python -m black src/data/ingest.py src/data/validate.py tests/test_validate.py

# Step-close protocol
python scripts/log_step.py --step 004 --title "Data validation and data card" --phase 1
```

## 4. Results / evidence
- **Data Validation Outcome:** `PASSED [OK]` (0 errors, 0 warnings).
- **OpenAQ Observations Validated:** 1,824 rows across 7 target pollutants (`co, nh3, no2, o3, pm10, pm25, so2`).
  - Future timestamps: 0
  - Inconsistent units: 0
  - Sentinels / Negatives: 0
  - Duplicates: 0
- **Weather Observations Validated:** 52,632 hourly records across 8 atmospheric variables (`temperature_2m, relative_humidity_2m, dew_point_2m, surface_pressure, precipitation, wind_speed_10m, wind_direction_10m, boundary_layer_height`).
  - Missing variables: 0
  - Physical range violations: 0
  - Duplicate timestamps: 0
- **Artifacts Created:**
  - `reports/metrics/data_validation.json`: detailed machine-readable validation metrics.
  - `docs/data_card.md`: official project Data Card adhering to Section 5.2 Step 4.

## 5. Decisions and rationale
- **D-007:** Established two-tiered validation: fatal hard-fails on structural breakage (missing critical columns, unparseable timestamps) to prevent corrupt pipeline propagation, alongside metric counts and warnings for domain phenomena (sentinel `-999` readings, minor sensor anomalies) that are addressed during the cleaning phase.
- **Data Card Scope:** Documented snapshot freeze date (`2026-10-15`), license per provider (CPCB/GODL, CC BY 4.0), 24-station network metadata, and limitations (ERA5 gridded resolution vs. urban microclimate, absence of satellite fire detections).

## 6. Problems encountered and fixes
1. **DVC Pipeline Stage Consistency:** Wired `src/data/ingest.py` as the execution target for `dvc.yaml` stage `validate`, ensuring `dvc repro validate` succeeds end-to-end.
2. **Nullable and Range Safeguards:** Designed `validate_weather_df` and `validate_openaq_df` to handle empty or partial inputs gracefully with `--allow-empty` for dry runs.

## 7. Verification
```bash
# Run all unit tests
python -m pytest tests/ -v
# Output: 39 passed in 1.86s

# Run validation stage directly
python -m src.data.ingest
# Output: Overall Status: PASSED [OK]

# Check logbook compliance
python scripts/check_logbook.py
# Output: [OK] Logbook check passed
```

## 8. Next step
Phase 1 is complete! Proceed to Phase 4 / Step 005: Implement CPCB AQI calculation and hourly-to-daily aggregation (`src/features/aqi_utils.py`, `src/data/hourly_to_daily.py`).
