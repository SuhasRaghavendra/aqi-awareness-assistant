"""The live latency probe's percentile and budget arithmetic."""

from __future__ import annotations

import pytest

from scripts.latency_probe import (
    BATCH_P95_BUDGET_MS,
    CITIES,
    SINGLE_P95_BUDGET_MS,
    _observation,
    _percentiles,
)


def test_percentiles_use_nearest_rank() -> None:
    values = [float(value) for value in range(1, 101)]
    stats = _percentiles(values)

    assert stats["p50"] == 50.0
    assert stats["p95"] == 95.0
    assert stats["p99"] == 99.0
    assert stats["min"] == 1.0
    assert stats["max"] == 100.0
    assert stats["mean"] == 50.5


def test_percentiles_handle_a_single_sample() -> None:
    stats = _percentiles([7.5])

    assert stats == {
        "p50": 7.5,
        "p95": 7.5,
        "p99": 7.5,
        "min": 7.5,
        "max": 7.5,
        "mean": 7.5,
    }


def test_budgets_match_the_plan() -> None:
    assert SINGLE_P95_BUDGET_MS == 50.0
    assert BATCH_P95_BUDGET_MS == 100.0


@pytest.mark.parametrize("city", CITIES)
def test_probe_observation_is_in_range(city: str) -> None:
    row = _observation(city, "2024-12-30")

    assert row["city"] == city
    assert 0.0 <= row["aqi"] <= 500.0
    assert row["wind_speed_10m_mean"] > 0.0
    assert row["temperature_2m_mean"] is not None
