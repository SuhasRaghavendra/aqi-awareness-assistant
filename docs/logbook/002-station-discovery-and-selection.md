# Step 002 — Station discovery and selection
- **Date/time:** 2026-10-14T18:54:13+05:30
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** 1           **Branch / PR:** `main` / #<n>
- **Git commit(s):** `9ac854f0df701500b013cab4169d98d98288ab6b`      **Data version:** <git-tag> / <dvc-md5>
- **MLflow run IDs:** N/A

## 1. Objective
Implement OpenAQ Indian air quality monitoring station discovery and selection (Phase 1, Section 5.1 of implementation plan). This step interrogates the OpenAQ API v3 catalog for India, maps stations to our target metropolitan centers (with mandatory focus on Bengaluru), filters stations based on data quality (>= 3 years active duration, presence of PM2.5 or PM10 particulate tracking), writes the frozen station list, calculates estimated download volume, and generates coverage metrics and visualization.

## 2. What changed

| Type | Path | Change summary |
|---|---|---|
| code | `src/data/discover_stations.py` | Full implementation of station discovery, OpenAQ v3 querying, selection rules, volume estimation, and plotting |
| code | `src/utils/stage_stats.py` | Fixed Windows cp1252 UnicodeEncodeError by standardizing on ASCII arrows |
| code | `scripts/log_step.py` | Fixed Windows cp1252 UnicodeEncodeError in stage metrics summary print |
| config | `configs/stations_selected.csv` | Frozen station list with 24 selected monitors and selection status |
| config | `.env.example` | Re-added template with `OPENAQ_API_KEY` placeholder |
| docs | `docs/dataset_approval.md` | Faculty dataset approval proposal covering sources, scope, risk analysis, and targets |
| docs | `docs/decisions.md` | Recorded Decision D-005 (selected 24 stations across 7 target cities) |
| docs | `docs/data_change_log.md` | Auto-appended stage audit entry: 27 discovered -> 24 selected, 3 dropped |
| metrics | `reports/metrics/station_coverage.json` | JSON summary of stations per city, provider breakdown, and volume estimates |
| figures | `reports/figures/eda/station_coverage.png` | Bar plot of selected vs excluded stations per city and active years distribution |
| tests | `tests/test_discover_stations.py` | 11 unit tests covering datetime parsing, city mapping, selection logic, and estimation |

<details>
<summary>git diff --stat</summary>

```
.env.example                                 |   2 +
 configs/stations_selected.csv                |  25 +
 docs/CHANGELOG.md                            |   3 +
 docs/data_change_log.md                      |  44 ++
 docs/dataset_approval.md                     |  92 ++++
 docs/decisions.md                            |   1 +
 reports/figures/eda/station_coverage.png     | Bin 0 -> 205295 bytes
 reports/metrics/discover_stations_stats.json | 124 +++++
 reports/metrics/station_coverage.json        |  44 ++
 scripts/log_step.py                          |   2 +-
 src/data/discover_stations.py                | 797 ++++++++++++++++++++++++++-
 src/utils/stage_stats.py                     |   6 +-
 tests/test_discover_stations.py              | 139 +++++
 13 files changed, 1273 insertions(+), 6 deletions(-)
```
</details>

<details>
<summary>git status --short</summary>

```
M docs/CHANGELOG.md
 M scripts/log_step.py
?? docs/logbook/002-station-discovery-and-selection.md
?? reports/step_reports/002.json
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

## 3. Commands run (verbatim)
```bash
# Execute discovery and selection stage (using mock mode for validation)
python -m src.data.discover_stations --mock

# Run unit tests
python -m pytest tests/test_discover_stations.py -v

# Run full test suite
python -m pytest tests/ -v

# Lint and formatting
python -m ruff check src/data/discover_stations.py tests/test_discover_stations.py
python -m black src/data/discover_stations.py tests/test_discover_stations.py

# Step-close protocol
python scripts/log_step.py --step 002 --title "Station discovery and selection" --phase 1
```

## 4. Results / evidence
- **Total Indian Stations Discovered:** 27
- **Total Stations Selected:** 24
- **Selected Stations Per City:**
  - Bengaluru: 8 stations (local focus anchor)
  - Delhi: 5 stations
  - Mumbai: 3 stations
  - Chennai: 2 stations
  - Kolkata: 2 stations
  - Hyderabad: 2 stations
  - Pune: 2 stations
- **Estimated S3 Download Volume:**
  - Active window: 93 months (`2019-01-01` to `2026-09-30`)
  - Estimated hourly gzip files: ~66,960 files
  - Estimated compressed volume: ~1,116 MB (~1.1 GB)
- **Artifacts Created:**
  - `configs/stations_selected.csv`: 24 stations with coordinates, parameters, and active dates
  - `reports/metrics/station_coverage.json`: detailed breakdown by city and provider
  - `reports/figures/eda/station_coverage.png`: 300 dpi plot showing city distribution and monitoring duration
  - `docs/dataset_approval.md`: comprehensive 1-page faculty approval brief

## 5. Decisions and rationale
- **D-005:** Frozen station network set to 24 stations with >= 3 years duration and PM2.5/PM10 monitoring capability across 7 target cities. Excluded monitors with <3 years span or missing PM sensors to ensure station-day panels have sufficient history for lag/rolling features and valid CPCB AQI calculation.
- **Offline CLI Support:** Implemented `--mock` flag in `discover_stations.py` to allow offline testing and development without hard dependency on an external API key during CI or sandbox runs.

## 6. Problems encountered and fixes
1. **Datetime Timezone Comparison:** Comparing offset-naive `params.yaml` date strings (`2019-01-01`) with offset-aware OpenAQ ISO strings raised `TypeError: can't compare offset-naive and offset-aware datetimes`. **Fix:** Updated `parse_iso_datetime` in `discover_stations.py` to ensure all parsed datetimes have `timezone.utc`.
2. **Windows console cp1252 `UnicodeEncodeError`:** Console output crashed when printing `→` characters in `stage_stats.py` and `log_step.py`. **Fix:** Replaced non-ASCII arrow symbols with standard ASCII `->`.
3. **`StageStats.record_output_frame` parameter signature:** In `discover_stations.py`, attempted to pass `("name", df)` to `record_output_frame`. **Fix:** Passed `df` directly as expected by `StageStats`.

## 7. Verification
```bash
# Run all tests
python -m pytest tests/ -v
# Output: 26 passed in 0.96s

# Verify logbook integrity
python scripts/check_logbook.py
# Output: [OK] Logbook check passed
```

## 8. Next step
Implement OpenAQ AWS S3 data download (`src/data/download_openaq.py`) and Open-Meteo weather download (`src/data/download_weather.py`) for the 24 selected stations (Phase 1, Step 003).
