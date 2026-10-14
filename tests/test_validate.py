"""
tests/test_validate.py
----------------------
Unit tests for src/data/validate.py.
"""

from __future__ import annotations

import pandas as pd

from src.data.validate import validate_openaq_df, validate_weather_df


class TestValidateOpenAQ:
    def test_valid_openaq_df(self):
        df = pd.DataFrame(
            [
                {
                    "location_id": 101,
                    "datetime": "2023-01-01T00:00:00+00:00",
                    "parameter": "pm25",
                    "units": "ug/m3",
                    "value": 45.2,
                },
                {
                    "location_id": 101,
                    "datetime": "2023-01-01T01:00:00+00:00",
                    "parameter": "pm10",
                    "units": "ug/m3",
                    "value": 85.0,
                },
            ]
        )
        report = validate_openaq_df(df)
        assert report.passed is True
        assert report.total_rows == 2
        assert len(report.errors) == 0

    def test_missing_required_column(self):
        df = pd.DataFrame([{"location_id": 101, "datetime": "2023-01-01T00:00:00Z"}])
        report = validate_openaq_df(df)
        assert report.passed is False
        assert any("Missing required OpenAQ columns" in e for e in report.errors)

    def test_future_timestamps(self):
        df = pd.DataFrame(
            [
                {
                    "location_id": 101,
                    "datetime": "2099-01-01T00:00:00+00:00",
                    "parameter": "pm25",
                    "units": "ug/m3",
                    "value": 25.0,
                }
            ]
        )
        report = validate_openaq_df(df)
        assert report.passed is False
        assert report.metrics["future_timestamps_count"] == 1

    def test_sentinels_and_negatives(self):
        df = pd.DataFrame(
            [
                {
                    "location_id": 1,
                    "datetime": "2023-01-01T00:00:00Z",
                    "parameter": "pm25",
                    "units": "ug/m3",
                    "value": -999,
                },
                {
                    "location_id": 1,
                    "datetime": "2023-01-01T01:00:00Z",
                    "parameter": "pm25",
                    "units": "ug/m3",
                    "value": -5.0,
                },
                {
                    "location_id": 1,
                    "datetime": "2023-01-01T02:00:00Z",
                    "parameter": "pm25",
                    "units": "ug/m3",
                    "value": 30.0,
                },
            ]
        )
        report = validate_openaq_df(df)
        assert report.metrics["sentinel_values_count"] == 1
        assert report.metrics["negative_values_count"] == 1

    def test_inconsistent_units(self):
        df = pd.DataFrame(
            [
                {
                    "location_id": 1,
                    "datetime": "2023-01-01T00:00:00Z",
                    "parameter": "pm25",
                    "units": "ug/m3",
                    "value": 30.0,
                },
                {
                    "location_id": 1,
                    "datetime": "2023-01-01T01:00:00Z",
                    "parameter": "pm25",
                    "units": "ppm",
                    "value": 0.03,
                },
            ]
        )
        report = validate_openaq_df(df)
        assert "pm25" in report.metrics["inconsistent_units"]


class TestValidateWeather:
    def test_valid_weather_df(self):
        df = pd.DataFrame(
            [
                {
                    "location_id": 101,
                    "time": "2023-01-01T00:00",
                    "temperature_2m": 24.5,
                    "relative_humidity_2m": 65.0,
                    "surface_pressure": 925.0,
                    "precipitation": 0.0,
                    "wind_speed_10m": 12.0,
                    "wind_direction_10m": 180.0,
                }
            ]
        )
        report = validate_weather_df(df)
        assert report.passed is True
        assert len(report.errors) == 0

    def test_missing_time_column(self):
        df = pd.DataFrame([{"temperature_2m": 25.0}])
        report = validate_weather_df(df)
        assert report.passed is False
        assert any("Missing required timestamp column 'time'" in e for e in report.errors)

    def test_range_violations(self):
        df = pd.DataFrame(
            [
                {
                    "time": "2023-01-01T00:00",
                    "relative_humidity_2m": 120.0,  # Invalid (>100)
                    "precipitation": -2.0,  # Invalid (<0)
                }
            ]
        )
        report = validate_weather_df(df)
        rv = report.metrics["range_violations"]
        assert rv.get("relative_humidity_out_of_range") == 1
        assert rv.get("negative_precipitation") == 1
