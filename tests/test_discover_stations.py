"""
tests/test_discover_stations.py
--------------------------------
Unit tests for src/data/discover_stations.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.data.discover_stations import (
    calculate_active_years,
    estimate_download_volume,
    evaluate_selection_rules,
    generate_mock_locations,
    map_to_target_city,
    parse_iso_datetime,
    parse_location_to_record,
)


class TestDatetimeUtils:
    def test_parse_iso_datetime_utc_z(self):
        dt = parse_iso_datetime("2021-01-01T12:00:00Z")
        assert dt is not None
        assert dt.year == 2021
        assert dt.tzinfo == timezone.utc

    def test_parse_iso_datetime_naive(self):
        dt = parse_iso_datetime("2022-05-10T08:30:00")
        assert dt is not None
        assert dt.year == 2022
        assert dt.tzinfo == timezone.utc

    def test_parse_iso_datetime_empty_and_invalid(self):
        assert parse_iso_datetime(None) is None
        assert parse_iso_datetime("") is None
        assert parse_iso_datetime("invalid-date") is None

    def test_calculate_active_years(self):
        dt1 = datetime(2020, 1, 1, tzinfo=timezone.utc)
        dt2 = datetime(2023, 1, 1, tzinfo=timezone.utc)
        years = calculate_active_years(dt1, dt2)
        assert 2.95 < years < 3.05

    def test_calculate_active_years_none(self):
        dt1 = datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert calculate_active_years(dt1, None) == 0.0
        assert calculate_active_years(None, dt1) == 0.0


class TestCityMapping:
    def test_bengaluru_aliases(self):
        target_cities = ["Bengaluru", "Delhi", "Mumbai"]
        assert map_to_target_city("Silk Board", "Bengaluru", target_cities) == "Bengaluru"
        assert map_to_target_city("City Station, Bangalore", "", target_cities) == "Bengaluru"
        assert map_to_target_city("BTM Layout", "Bangalore South", target_cities) == "Bengaluru"

    def test_other_cities(self):
        target_cities = ["Bengaluru", "Delhi", "Mumbai"]
        assert map_to_target_city("R K Puram", "New Delhi", target_cities) == "Delhi"
        assert map_to_target_city("Bandra Kurla", "Mumbai", target_cities) == "Mumbai"

    def test_unmatched_city(self):
        target_cities = ["Bengaluru", "Delhi"]
        assert map_to_target_city("Rural Monitor", "Mandya", target_cities) is None


class TestSelectionRules:
    def test_evaluate_selection_rules(self):
        mock_data = [
            {
                "location_id": 1,
                "name": "BTM Layout, Bengaluru",
                "locality": "Bengaluru",
                "has_pm": True,
                "has_pm25": True,
                "has_pm10": True,
                "active_years": 5.0,
                "datetime_first_utc": "2019-01-01T00:00:00+00:00",
                "datetime_last_utc": "2024-01-01T00:00:00+00:00",
            },
            {
                "location_id": 2,
                "name": "Short Monitor, Bengaluru",
                "locality": "Bengaluru",
                "has_pm": True,
                "has_pm25": True,
                "has_pm10": False,
                "active_years": 1.2,
                "datetime_first_utc": "2023-01-01T00:00:00+00:00",
                "datetime_last_utc": "2024-03-01T00:00:00+00:00",
            },
            {
                "location_id": 3,
                "name": "No PM Station, Delhi",
                "locality": "Delhi",
                "has_pm": False,
                "has_pm25": False,
                "has_pm10": False,
                "active_years": 4.5,
                "datetime_first_utc": "2019-01-01T00:00:00+00:00",
                "datetime_last_utc": "2023-06-01T00:00:00+00:00",
            },
        ]
        df = pd.DataFrame(mock_data)
        res = evaluate_selection_rules(
            df,
            target_cities=["Bengaluru", "Delhi"],
            min_years=3.0,
            window_start="2019-01-01",
            snapshot_end="2026-09-30",
        )
        assert bool(res.loc[0, "selected"]) is True
        assert bool(res.loc[1, "selected"]) is False
        assert "Active span" in res.loc[1, "selection_status"]
        assert bool(res.loc[2, "selected"]) is False
        assert "Lacks both PM2.5 and PM10" in res.loc[2, "selection_status"]


class TestParsingAndEstimates:
    def test_parse_location_to_record(self):
        loc = generate_mock_locations()[0]
        rec = parse_location_to_record(loc)
        assert rec["location_id"] == 101
        assert rec["has_pm25"] is True
        assert rec["has_pm10"] is True
        assert "pm25" in rec["parameters"]
        assert rec["active_years"] > 7.0

    def test_estimate_download_volume(self):
        df = pd.DataFrame([{"id": 1}, {"id": 2}])
        est = estimate_download_volume(df, "2019-01-01", "2023-12-31")
        assert est["selected_stations_count"] == 2
        assert est["window_months"] == 60
        assert est["estimated_files_count"] == 2 * 60 * 30
        assert est["estimated_compressed_size_mb"] > 0
