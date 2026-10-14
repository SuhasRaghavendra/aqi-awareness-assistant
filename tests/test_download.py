"""
tests/test_download.py
----------------------
Unit tests for data download modules:
  - src/data/download_openaq.py
  - src/data/download_weather.py
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from src.data.download_openaq import (
    compute_sha256,
    generate_mock_openaq_data,
    generate_year_month_tuples,
)
from src.data.download_weather import (
    generate_mock_weather_chunk,
    generate_yearly_intervals,
)


class TestOpenAQDownloadUtils:
    def test_generate_year_month_tuples_multi_year(self):
        tuples = generate_year_month_tuples("2021-10-01", "2022-03-01")
        expected = [
            (2021, 10),
            (2021, 11),
            (2021, 12),
            (2022, 1),
            (2022, 2),
            (2022, 3),
        ]
        assert tuples == expected

    def test_compute_sha256(self, tmp_path: Path):
        f = tmp_path / "sample.txt"
        f.write_text("hello openaq", encoding="utf-8")
        sha = compute_sha256(f)
        assert len(sha) == 64

    def test_generate_mock_openaq_data(self, tmp_path: Path):
        station_df = pd.DataFrame(
            [
                {
                    "location_id": 101,
                    "name": "BTM Layout, Bengaluru",
                    "latitude": 12.9165,
                    "longitude": 77.6101,
                    "parameters": "pm25,pm10,no2",
                }
            ]
        )
        records = generate_mock_openaq_data(
            stations_df=station_df,
            output_dir=tmp_path,
            window_start="2022-01-01",
            snapshot_end="2022-02-01",
            limit_months=1,
        )
        assert len(records) > 0
        local_file = Path(records[0]["local_path"])
        assert local_file.exists()

        # Check gzip content and schema
        with gzip.open(local_file, "rt", encoding="utf-8") as gz:
            lines = gz.readlines()
        assert len(lines) > 1
        header = lines[0].strip().split(",")
        assert "location_id" in header
        assert "parameter" in header
        assert "value" in header
        assert "datetime" in header


class TestWeatherDownloadUtils:
    def test_generate_yearly_intervals(self):
        intervals = generate_yearly_intervals("2021-06-15", "2023-04-10")
        assert len(intervals) == 3
        assert intervals[0] == (2021, "2021-06-15", "2021-12-31")
        assert intervals[1] == (2022, "2022-01-01", "2022-12-31")
        assert intervals[2] == (2023, "2023-01-01", "2023-04-10")

    def test_generate_mock_weather_chunk(self):
        vars_to_test = [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "surface_pressure",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "boundary_layer_height",
        ]
        df = generate_mock_weather_chunk(
            lat=12.9716,
            lon=77.5946,
            start_date="2023-01-01",
            end_date="2023-01-03",
            variables=vars_to_test,
        )
        assert len(df) == 72  # 3 days * 24 hours
        assert "time" in df.columns
        for v in vars_to_test:
            assert v in df.columns
            assert not df[v].isna().any()
