"""
tests/test_download_weather.py
-------------------------------
Tests for the new city-based Open-Meteo weather downloader.
"""
import json
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.download_weather import (
    CITY_COORDS,
    DAILY_VARIABLES,
    add_derived_weather,
    generate_mock_weather,
)


def test_city_coords_complete():
    """All 9 target cities must have coordinates."""
    from src.config import load_params
    params = load_params()
    for city in params["data"]["cities"]:
        assert city in CITY_COORDS, f"No coordinates for city: {city}"
        lat, lon = CITY_COORDS[city]
        assert 6 <= lat <= 38, f"[{city}] Latitude {lat} out of India range"
        assert 68 <= lon <= 98, f"[{city}] Longitude {lon} out of India range"


def test_mock_weather_schema():
    """Mock weather output must have all required columns."""
    df = generate_mock_weather("Delhi", 28.6139, "2023-01-01", "2023-01-31")
    assert not df.empty
    assert "date" in df.columns
    assert "city" in df.columns
    for col in DAILY_VARIABLES:
        assert col in df.columns, f"Missing column: {col}"


def test_mock_weather_row_count():
    """Mock weather must produce exactly 31 rows for January."""
    df = generate_mock_weather("Mumbai", 19.0760, "2023-01-01", "2023-01-31")
    assert len(df) == 31


def test_mock_weather_value_ranges():
    """Mock weather values must be within plausible Indian meteorological ranges."""
    df = generate_mock_weather("Chennai", 13.0827, "2022-06-01", "2022-08-31")
    assert df["temperature_2m_mean"].between(-5, 55).all(), "Temperature out of range"
    assert df["relative_humidity_2m_mean"].between(0, 100).all(), "Humidity out of range"
    assert df["wind_speed_10m_mean"].between(0, 150).all(), "Wind speed out of range"
    assert (df["precipitation_sum"] >= 0).all(), "Precipitation cannot be negative"


def test_add_derived_weather():
    """Derived flags must be binary integers."""
    df = generate_mock_weather("Delhi", 28.6139, "2022-11-01", "2022-11-30")
    df = add_derived_weather(df)
    assert "inversion_risk_flag" in df.columns
    assert "fog_risk_flag" in df.columns
    assert "rain_washout_flag" in df.columns
    assert df["inversion_risk_flag"].isin([0, 1]).all()
    assert df["fog_risk_flag"].isin([0, 1]).all()
    assert df["rain_washout_flag"].isin([0, 1]).all()


def test_derived_inversion_logic():
    """Inversion risk should be 1 when wind < 5 and humidity > 70."""
    import pandas as pd
    df = pd.DataFrame({
        "wind_speed_10m_mean": [2.0, 10.0, 3.0, 8.0],
        "relative_humidity_2m_mean": [80.0, 80.0, 60.0, 60.0],
        "precipitation_sum": [0.0, 0.0, 0.0, 0.0],
    })
    df = add_derived_weather(df)
    # Row 0: wind=2 < 5, hum=80 > 70 → 1
    # Row 1: wind=10 >= 5 → 0
    # Row 2: hum=60 <= 70 → 0
    # Row 3: both fail → 0
    assert df["inversion_risk_flag"].tolist() == [1, 0, 0, 0]

