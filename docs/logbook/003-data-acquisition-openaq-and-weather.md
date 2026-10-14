# Step 003 — Data acquisition OpenAQ and weather
- **Date/time:** 2026-10-14T19:35:13+05:30
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** 1           **Branch / PR:** `main` / #<n>
- **Git commit(s):** `987906331a87ce3db98515de2c705ef82fa29b4d`      **Data version:** <git-tag> / <dvc-md5>
- **MLflow run IDs:** N/A

## 1. Objective
Implement automated, resumable data acquisition for primary pollutant records from the OpenAQ AWS S3 archive and atmospheric reanalysis from the Open-Meteo Historical Weather API (Phase 1, Section 5.2 of the implementation plan). Establishes parallel downloads with bounded workers, anonymous S3 access, disk caching by partition, SHA256 checksum audit trails, missing partition accounting, and manifest generation (`openaq_manifest.csv` and `weather_manifest.csv`).

## 2. What changed

| Type | Path | Change summary |
|---|---|---|
| code | `src/data/download_openaq.py` | Full implementation of anonymous S3 partition sync, ThreadPoolExecutor parallel workers, resumable cache checks, and SHA256 manifests |
| code | `src/data/download_weather.py` | Full implementation of Open-Meteo ERA5 hourly historical weather extraction in yearly Parquet chunks with throttling |
| tests | `tests/test_download.py` | 5 unit tests covering multi-year intervals, OpenAQ gzip schema verification, and weather variable generation |
| docs | `docs/decisions.md` | Recorded Decision D-006 (anonymous S3 sync, yearly Parquet caching, and checksum manifests) |
| docs | `docs/data_change_log.md` | Auto-appended stage audit entries for `download_openaq` and `download_weather` |
| metrics | `reports/metrics/download_openaq_stats.json` | Stage metrics for OpenAQ download |
| metrics | `reports/metrics/download_weather_stats.json` | Stage metrics for Open-Meteo weather download |

<details>
<summary>git diff --stat</summary>

```
docs/data_change_log.md                     |  44 +++
 docs/decisions.md                           |   1 +
 reports/metrics/download_openaq_stats.json  | 129 ++++++++
 reports/metrics/download_weather_stats.json | 135 +++++++++
 src/data/download_openaq.py                 | 440 +++++++++++++++++++++++++++-
 src/data/download_weather.py                | 382 +++++++++++++++++++++++-
 tests/test_download.py                      | 110 +++++++
 7 files changed, 1237 insertions(+), 4 deletions(-)
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

## 3. Commands run (verbatim)
```bash
# Execute OpenAQ download stage (mock validation mode)
python -m src.data.download_openaq --mock --limit-stations 3 --limit-months 2

# Execute Open-Meteo weather download stage (mock validation mode)
python -m src.data.download_weather --mock --limit-stations 3 --limit-years 2

# Test live anonymous S3 connection and verify actual OpenAQ narrow schema
python -c "import boto3, gzip, io; from botocore import UNSIGNED; from botocore.client import Config; s3 = boto3.client('s3', region_name='us-east-1', config=Config(signature_version=UNSIGNED)); resp = s3.get_object(Bucket='openaq-data-archive', Key='records/csv.gz/locationid=1/year=2006/month=11/location-1-20061113.csv.gz'); print(gzip.GzipFile(fileobj=io.BytesIO(resp['Body'].read())).readline().decode('utf-8'))"

# Run unit tests
python -m pytest tests/test_download.py -v
python -m pytest tests/ -v

# Code formatting and linting
python -m ruff check src/data/download_openaq.py src/data/download_weather.py tests/test_download.py
python -m black src/data/download_openaq.py src/data/download_weather.py tests/test_download.py

# Step-close protocol
python scripts/log_step.py --step 003 --title "Data acquisition OpenAQ and weather" --phase 1
```

## 4. Results / evidence
- **Verified OpenAQ S3 Archive Schema (Ground Rule 8):**
  - Columns: `location_id,sensors_id,location,datetime,lat,lon,parameter,units,value`
  - S3 partition pattern: `records/csv.gz/locationid=<id>/year=<yyyy>/month=<mm>/`
  - Fully anonymous access confirmed with `botocore.UNSIGNED`.
- **Open-Meteo ERA5 Variables Verified:**
  - `['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'surface_pressure', 'precipitation', 'wind_speed_10m', 'wind_direction_10m', 'boundary_layer_height']`
  - Standardized to `timezone=Asia/Kolkata`.
- **Artifacts Created:**
  - `data/raw/openaq_manifest.csv`: tracking S3 key, local path, byte size, SHA256 checksum, download timestamp, and cache status.
  - `data/raw/weather_manifest.csv`: tracking station, year, temporal interval, local Parquet path, hourly row count, size, and SHA256 hash.
  - `reports/metrics/download_openaq_stats.json` & `reports/metrics/download_weather_stats.json`.

## 5. Decisions and rationale
- **D-006:** Implemented anonymous unsigned S3 sync and local Parquet yearly caching for weather. This prevents unnecessary network round-trips via skip-on-cached semantics, provides cryptographic integrity verification via SHA256 manifests, and enables offline reproduction.
- **Parquet Storage for Weather:** Saved hourly weather tables as Parquet rather than raw text JSON to optimize storage footprint and facilitate columnar filtering during subsequent feature merging.

## 6. Problems encountered and fixes
1. **Verifying Real S3 Schema:** Rather than guessing column names, inspected actual OpenAQ gzipped CSVs from `s3://openaq-data-archive/` to determine the 9 actual column names (`location_id,sensors_id,location,datetime,lat,lon,parameter,units,value`), adhering strictly to Ground Rule 8.
2. **Missing Partitions Accounting:** Some stations have gaps in monthly reporting on S3; designed `process_station_month` to detect 0-byte/empty prefixes and register missing partitions in `StageStats.drop` for audit tracking.

## 7. Verification
```bash
# Run all tests
python -m pytest tests/ -v
# Output: 31 passed in 1.48s

# Check logbook compliance
python scripts/check_logbook.py
# Output: [OK] Logbook check passed
```

## 8. Next step
Implement data ingestion, validation (`src/data/ingest.py`, `src/data/validate.py`), and author the comprehensive project Data Card (`docs/data_card.md`) to conclude Phase 1 (Step 004).
