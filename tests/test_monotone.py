"""Gate 3: monotone fire/wind constraints (ADR D-030)."""

from __future__ import annotations

from src.models.monotone import (
    build_monotone_constraints,
    constrained_features,
    monotone_direction,
)


def test_direction_covers_fire_wind_and_stagnation() -> None:
    assert monotone_direction("fire_count_500km") == 1
    assert monotone_direction("fire_frp_sum_500km_roll_max_30d") == 1
    assert monotone_direction("fire_x_low_wind") == 1
    assert monotone_direction("wind_speed_10m_mean") == -1
    assert monotone_direction("wind_speed_10m_mean_lag_1") == -1
    assert monotone_direction("wind_roll_mean_7d") == -1
    assert monotone_direction("winter_stagnation_index") == 1
    # Nearest-fire distance is not a clean monotone signal inside a 500 km sum.
    assert monotone_direction("fire_dist_min_km") == 0
    assert monotone_direction("temperature_2m_mean") == 0


def test_constraint_vector_is_aligned_with_the_feature_order() -> None:
    features = [
        "temperature_2m_mean",
        "fire_count_500km",
        "wind_speed_10m_mean_lag_1",
        "fire_x_low_wind",
    ]
    assert build_monotone_constraints(features) == [0, 1, -1, 1]
    assert constrained_features(features) == {
        "fire_count_500km": 1,
        "wind_speed_10m_mean_lag_1": -1,
        "fire_x_low_wind": 1,
    }
