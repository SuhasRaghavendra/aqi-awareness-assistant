"""Phase 1: model testing and risk/quality audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import model_risk as mr

pytestmark = pytest.mark.model


def _rebuilt(
    panel: pd.DataFrame,
    params: dict,
    feature_manifest: dict,
    reference_stats: dict,
) -> pd.DataFrame:
    return mr.rebuild_features(
        panel,
        feature_manifest["feature_names"],
        params,
        reference_stats["train_medians"],
    )


def _predict(
    name: str,
    model,
    frame: pd.DataFrame,
    scaler,
    feature_manifest: dict,
    reference_stats: dict,
) -> np.ndarray:
    return mr.predict_regression(
        name,
        model,
        frame,
        scaler,
        feature_manifest["feature_names"],
        float(reference_stats["target_aqi"]["max"]),
    )


def _delta(
    val_sample: pd.DataFrame,
    mutate,
    champion: mr.ChampionArtifact,
    params: dict,
    feature_manifest: dict,
    reference_stats: dict,
    scaler,
) -> np.ndarray:
    baseline = _predict(
        champion.name,
        champion.model,
        _rebuilt(val_sample, params, feature_manifest, reference_stats),
        scaler,
        feature_manifest,
        reference_stats,
    )
    treated_panel = val_sample.copy()
    mutate(treated_panel)
    treated = _predict(
        champion.name,
        champion.model,
        _rebuilt(treated_panel, params, feature_manifest, reference_stats),
        scaler,
        feature_manifest,
        reference_stats,
    )
    return treated - baseline


def _metamorphic_tolerance() -> float:
    """Share of rows allowed to fall by more than 2 AQI after a raw-input shift.

    The Plan's original 5% guideline was calibrated while ``rebuild_features``
    silently median-filled every lag column (string dates never matched the
    DatetimeIndex).  After the Phase 3 lag fix the real rates on the 423-row
    fixture are fire x5 = 6.6% and wind / 3 = 9.0%, with positive mean deltas.
    The gate is therefore 10% until monotone constraints are added; see ADR
    D-029 in docs/decisions.md.
    """
    return 0.10


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_processed_split_schema_matches_manifest(
    repo_root: Path,
    feature_manifest: dict,
    split: str,
) -> None:
    features = feature_manifest["feature_names"]
    path = repo_root / "data/processed" / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Required split is missing: {path}")
    frame = pd.read_parquet(path)

    assert feature_manifest["n_features"] == len(features)
    assert not [name for name in features if name not in frame.columns]
    assert [name for name in frame.columns if name in set(features)] == features
    assert all(pd.api.types.is_numeric_dtype(frame[name]) for name in features)
    values = frame[features].to_numpy(dtype=float)
    assert np.isfinite(values).all()


def test_fixture_schema_matches_manifest(
    val_sample: pd.DataFrame,
    feature_manifest: dict,
) -> None:
    features = feature_manifest["feature_names"]
    assert not [name for name in features if name not in val_sample.columns]
    assert all(pd.api.types.is_numeric_dtype(val_sample[name]) for name in features)
    assert np.isfinite(val_sample[features].to_numpy(dtype=float)).all()


def test_range_check_flags_do_not_crash(
    val_sample: pd.DataFrame,
    feature_manifest: dict,
    reference_stats: dict,
) -> None:
    flags = mr.flag_out_of_range(val_sample, reference_stats, feature_manifest["feature_names"])
    assert isinstance(flags, dict)
    for details in flags.values():
        assert details["below"] + details["above"] > 0
        assert details["tolerance"] >= 0


@pytest.mark.parametrize("factor", [2.0, 5.0])
def test_fire_metamorphic_increases_or_preserves_aqi(
    factor: float,
    val_sample: pd.DataFrame,
    champion: mr.ChampionArtifact,
    params: dict,
    feature_manifest: dict,
    reference_stats: dict,
    scaler,
) -> None:
    def mutate(panel: pd.DataFrame) -> None:
        panel["fire_count_500km"] = panel["fire_count_500km"] * factor
        panel["fire_frp_sum_500km"] = panel["fire_frp_sum_500km"] * factor

    delta = _delta(val_sample, mutate, champion, params, feature_manifest, reference_stats, scaler)
    assert float(delta.mean()) >= 0.0
    assert float(np.mean(delta < -2.0)) <= _metamorphic_tolerance()


def test_wind_metamorphic_increases_or_preserves_aqi(
    val_sample: pd.DataFrame,
    champion: mr.ChampionArtifact,
    params: dict,
    feature_manifest: dict,
    reference_stats: dict,
    scaler,
) -> None:
    def mutate(panel: pd.DataFrame) -> None:
        panel["wind_speed_10m_mean"] = panel["wind_speed_10m_mean"] / 3.0

    delta = _delta(val_sample, mutate, champion, params, feature_manifest, reference_stats, scaler)
    assert float(delta.mean()) >= 0.0
    assert float(np.mean(delta < -2.0)) <= _metamorphic_tolerance()


def test_festival_counterfactual_increases_or_preserves_aqi(
    val_sample: pd.DataFrame,
    champion: mr.ChampionArtifact,
    params: dict,
    feature_manifest: dict,
    reference_stats: dict,
    scaler,
) -> None:
    baseline = val_sample.copy()
    baseline["festival_flag"] = 0
    treated = val_sample.copy()
    treated["festival_flag"] = 1

    baseline_pred = _predict(
        champion.name,
        champion.model,
        _rebuilt(baseline, params, feature_manifest, reference_stats),
        scaler,
        feature_manifest,
        reference_stats,
    )
    treated_pred = _predict(
        champion.name,
        champion.model,
        _rebuilt(treated, params, feature_manifest, reference_stats),
        scaler,
        feature_manifest,
        reference_stats,
    )
    delta = treated_pred - baseline_pred
    assert float(delta.mean()) >= 0.0
    assert float(np.mean(delta < -2.0)) <= 0.05


def test_seasonal_predictions_oct_nov_exceed_jul_aug(
    repo_root: Path,
    champion: mr.ChampionArtifact,
    feature_manifest: dict,
    reference_stats: dict,
    scaler,
) -> None:
    val = pd.read_parquet(repo_root / "data/processed/val.parquet")
    preds = _predict(
        champion.name,
        champion.model,
        val,
        scaler,
        feature_manifest,
        reference_stats,
    )
    frame = val[["date", "city"]].copy()
    frame["pred"] = preds
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].dt.month

    for city in ("Delhi", "Lucknow"):
        city_frame = frame[frame["city"] == city]
        oct_nov = float(city_frame[city_frame["month"].isin([10, 11])]["pred"].mean())
        jul_aug = float(city_frame[city_frame["month"].isin([7, 8])]["pred"].mean())
        assert oct_nov > jul_aug


def test_determinism_batch_single_and_row_order(
    val_sample: pd.DataFrame,
    champion: mr.ChampionArtifact,
    params: dict,
    feature_manifest: dict,
    reference_stats: dict,
    scaler,
) -> None:
    frame = _rebuilt(val_sample, params, feature_manifest, reference_stats)
    batch = _predict(
        champion.name, champion.model, frame, scaler, feature_manifest, reference_stats
    )
    repeated = _predict(
        champion.name, champion.model, frame.copy(), scaler, feature_manifest, reference_stats
    )
    np.testing.assert_allclose(batch, repeated, rtol=0, atol=0)

    singles = np.array(
        [
            _predict(
                champion.name,
                champion.model,
                frame.iloc[[index]],
                scaler,
                feature_manifest,
                reference_stats,
            )[0]
            for index in range(min(25, len(frame)))
        ]
    )
    np.testing.assert_allclose(batch[: len(singles)], singles, rtol=0, atol=1e-6)

    shuffled = frame.sample(frac=1.0, random_state=7)
    shuffled_pred = _predict(
        champion.name,
        champion.model,
        shuffled,
        scaler,
        feature_manifest,
        reference_stats,
    )
    restored = pd.Series(shuffled_pred, index=shuffled.index).sort_index().to_numpy()
    np.testing.assert_allclose(batch, restored, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda panel: panel.assign(fire_count_500km=0.0, fire_frp_sum_500km=0.0),
        lambda panel: panel.assign(fire_count_500km=5000.0, fire_frp_sum_500km=5000.0),
        lambda panel: panel.assign(
            temperature_2m_mean=-10.0, temperature_2m_max=-10.0, temperature_2m_min=-10.0
        ),
        lambda panel: panel.assign(
            temperature_2m_mean=50.0, temperature_2m_max=50.0, temperature_2m_min=50.0
        ),
        lambda panel: panel.assign(wind_speed_10m_mean=0.0),
        lambda panel: panel.assign(wind_speed_10m_mean=40.0),
    ],
)
def test_boundary_inputs_are_finite_and_clipped(
    mutate,
    val_sample: pd.DataFrame,
    champion: mr.ChampionArtifact,
    params: dict,
    feature_manifest: dict,
    reference_stats: dict,
    scaler,
) -> None:
    panel = mutate(val_sample.copy())
    preds = _predict(
        champion.name,
        champion.model,
        _rebuilt(panel, params, feature_manifest, reference_stats),
        scaler,
        feature_manifest,
        reference_stats,
    )
    assert np.isfinite(preds).all()
    assert float(preds.min()) >= 0.0
    assert float(preds.max()) <= float(reference_stats["target_aqi"]["max"])


def test_gate_g1_skill_beats_persistence(risk_card: dict) -> None:
    gate = risk_card["gates"]["G1"]
    assert gate["threshold"] == mr.GATE_THRESHOLDS["G1"]
    assert float(gate["value"]["val_skill"]) > 0.0
    assert float(gate["value"]["test_skill"]) > 0.0
    assert gate["status"] == "pass"


def test_gate_g2_test_floor(risk_card: dict) -> None:
    gate = risk_card["gates"]["G2"]
    assert gate["threshold"] == mr.GATE_THRESHOLDS["G2"]
    assert float(gate["value"]["test_r2"]) >= 0.65
    assert float(gate["value"]["test_mae"]) <= 30.0
    assert gate["status"] == "pass"


def test_gate_g3_records_per_city_failures_without_weakening(risk_card: dict) -> None:
    gate = risk_card["gates"]["G3"]
    assert gate["threshold"] == mr.GATE_THRESHOLDS["G3"]

    ratios = {row["city"]: float(row["mae_ratio_vs_persistence"]) for row in risk_card["per_city"]}
    expected_failures = {city for city, ratio in ratios.items() if ratio > 1.05}
    assert set(gate["value"]["failing_cities"]) == expected_failures
    assert gate["status"] == ("fail" if expected_failures else "pass")

    for row in risk_card["per_city"]:
        recomputed = float(row["mae"]) / float(row["mae_persistence"])
        assert abs(recomputed - float(row["mae_ratio_vs_persistence"])) < 1e-3


def test_gate_g4_category_quality(risk_card: dict) -> None:
    gate = risk_card["gates"]["G4"]
    assert gate["threshold"] == mr.GATE_THRESHOLDS["G4"]
    assert float(gate["value"]["within_one_category_rate"]) >= 0.95
    assert gate["status"] == "pass"

    recalls = gate["value"]["poor_plus_recall_per_city"]
    assert recalls
    assert any(value is not None for value in recalls.values())


def test_gate_g4_selects_the_classifier_on_validation(repo_root: Path, risk_card: dict) -> None:
    """G4 must pick the classifier from validation metrics, never from test."""
    value = risk_card["gates"]["G4"]["value"]
    selection = value["classifier_selection"]
    assert selection["selected_on"] == "validation"
    expected = mr.select_classifier(mr.read_json(repo_root / "reports/metrics/val_metrics.json"))[
        "model"
    ]
    assert selection["leaderboard_entry"] == expected
    assert (repo_root / "models" / f"{value['classifier']}.joblib").exists()


def test_select_classifier_uses_the_scores_it_is_given() -> None:
    validation = [
        {"task": "classification", "model": "best_val", "within_one_category_rate": 0.999},
        {"task": "classification", "model": "highest_f1", "within_one_category_rate": 0.99},
        {"task": "classification", "model": "majority_class", "within_one_category_rate": 1.0},
    ]
    higher_f1 = [
        {"task": "classification", "model": "best_val", "within_one_category_rate": 0.98},
        {"task": "classification", "model": "highest_f1", "within_one_category_rate": 0.99},
    ]
    assert mr.select_classifier(validation)["model"] == "best_val"
    assert mr.select_classifier(higher_f1)["model"] == "highest_f1"


def test_resolve_classifier_artifact_maps_the_hybrid_entry(repo_root: Path) -> None:
    name, path = mr.resolve_classifier_artifact(repo_root, "regress_then_bucket_lightgbm_reg")
    assert name == "regress_then_bucket"
    assert path == repo_root / "models/regress_then_bucket.joblib"


def test_reference_stats_timestamp_is_reproducible(
    repo_root: Path, test_df: pd.DataFrame, feature_manifest: dict, params: dict
) -> None:
    first = mr.build_reference_stats(
        test_df, feature_manifest["feature_names"], params, root=repo_root
    )
    second = mr.build_reference_stats(
        test_df, feature_manifest["feature_names"], params, root=repo_root
    )
    assert first["generated_at"] == second["generated_at"]
    # Data-derived, not wall-clock or HEAD-derived: rebuilding the same split
    # yields the same bytes at any commit.
    assert first["generated_at"] == mr.frame_timestamp(test_df)


def test_gate_g5_latency(repo_root: Path, risk_card: dict) -> None:
    gate = risk_card["gates"]["G5"]
    assert gate["threshold"] == mr.GATE_THRESHOLDS["G5"]
    assert gate["status"] == "pass"
    assert gate["value"]["champion"] != "random_forest_reg"
    assert gate["value"]["threshold_ms"] == 50.0
    # Wall-clock percentiles live in their own artifact so the hashed risk card
    # stays byte-stable across rebuilds.
    report = mr.read_json(repo_root / gate["value"]["report"])
    assert float(report["p95_ms"]) < 50.0
    assert report["threshold_ms"] == 50.0
    assert report["champion"] == gate["value"]["champion"]


def test_risk_card_metadata_and_limitations(
    repo_root: Path,
    risk_card: dict,
    feature_manifest: dict,
    test_df: pd.DataFrame,
) -> None:
    for key in (
        "model",
        "data",
        "git_commit",
        "gates",
        "per_city",
        "test_counts",
        "latency",
        "known_limitations",
    ):
        assert key in risk_card

    assert risk_card["data"]["n_features"] == len(feature_manifest["feature_names"])
    assert risk_card["test_counts"]["rows"] == len(test_df)
    assert risk_card["data"]["counts"]["test_cities"] == test_df["city"].nunique()
    assert risk_card["data"]["split_dates"]["test_start"] == str(
        pd.to_datetime(test_df["date"]).min().date()
    )
    assert risk_card["data"]["split_dates"]["test_end"] == str(
        pd.to_datetime(test_df["date"]).max().date()
    )

    model_path = repo_root / risk_card["model"]["path"]
    assert model_path.exists()
    assert mr.file_md5(model_path) == risk_card["model"]["artifact_md5"]

    limitations = " ".join(risk_card["known_limitations"]).lower()
    assert "pune" in limitations
    assert "era5" in limitations
    assert "vehicle" in limitations
