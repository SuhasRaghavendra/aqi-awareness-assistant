"""Phase 2: champion bundle contract tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.models import model_risk as mr
from src.models.package_model import HISTORY_SEED_DAYS, read_dvc_data_md5
from src.serving.bundle import AqiBundle, sha256_file

pytestmark = pytest.mark.model


@pytest.fixture(scope="session")
def bundle_dir(repo_root: Path) -> Path:
    path = repo_root / "artifacts/champion"
    if not path.exists():
        raise FileNotFoundError(f"Required champion bundle is missing: {path}")
    return path


@pytest.fixture(scope="session")
def bundle(bundle_dir: Path) -> AqiBundle:
    return AqiBundle.load(bundle_dir)


@pytest.fixture(scope="session")
def manifest(bundle_dir: Path) -> dict:
    return mr.read_json(bundle_dir / "MANIFEST.json")


def test_manifest_sha_matches_file(bundle_dir: Path) -> None:
    expected = (bundle_dir / "MANIFEST.sha256").read_text(encoding="utf-8").strip()
    assert expected == sha256_file(bundle_dir / "MANIFEST.json")


def test_payload_hashes_match_manifest(bundle_dir: Path, manifest: dict) -> None:
    assert manifest["reference_stats_sha"] == sha256_file(bundle_dir / "reference_stats.json")
    assert manifest["history_seed_sha"] == sha256_file(bundle_dir / "history_seed.parquet")


def test_text_artifacts_use_lf_bytes(bundle_dir: Path) -> None:
    """Manifest digests are byte-exact, so no text artifact may contain CRLF."""
    for name in ("MANIFEST.json", "MANIFEST.sha256", "reference_stats.json"):
        data = (bundle_dir / name).read_bytes()
        assert b"\r\n" not in data, f"{name} contains CRLF; digests break on a fresh checkout"


def test_git_never_rewrites_bundle_line_endings(repo_root: Path) -> None:
    rules = [
        line.split("#", 1)[0].strip()
        for line in (repo_root / ".gitattributes").read_text(encoding="utf-8").splitlines()
    ]
    assert "artifacts/champion/** -text" in rules


def test_bundle_load_accepts_a_string_path(
    bundle_dir: Path, val_sample: pd.DataFrame, bundle: AqiBundle
) -> None:
    loaded = AqiBundle.load(str(bundle_dir))
    assert loaded.feature_names == bundle.feature_names
    np.testing.assert_allclose(loaded.predict(val_sample), bundle.predict(val_sample))


def test_feature_count_is_not_hard_coded(
    bundle: AqiBundle, manifest: dict, feature_manifest: dict
) -> None:
    expected = len(feature_manifest["feature_names"])
    assert manifest["features_count"] == expected
    assert bundle.feature_count == expected
    assert len(bundle.feature_names) == expected
    digest = hashlib.sha256("\n".join(feature_manifest["feature_names"]).encode("utf-8"))
    assert manifest["feature_names_sha256"] == digest.hexdigest()


def test_manifest_points_at_the_risk_card_champion(
    repo_root: Path, bundle: AqiBundle, manifest: dict, risk_card: dict
) -> None:
    assert manifest["model"]["name"] == risk_card["model"]["name"]
    assert bundle.model is not None
    assert manifest["model"]["artifact_md5"] == mr.file_md5(repo_root / manifest["model"]["path"])
    assert manifest["risk_card_sha"] == sha256_file(
        repo_root / "reports/metrics/model_risk_card.json"
    )
    assert manifest["fallback"]["artifact_md5"] == mr.file_md5(
        repo_root / manifest["fallback"]["path"]
    )


def test_manifest_records_waived_gates_only(manifest: dict, risk_card: dict) -> None:
    status = {gate: item["status"] for gate, item in risk_card["gates"].items()}
    assert manifest["gate_status"] == status
    failing = {gate for gate, value in status.items() if value != "pass"}
    assert set(manifest["gate_waivers"]) <= failing
    assert set(manifest["gate_waivers"]) == {"G3"}


def test_manifest_uses_the_dvc_data_md5(repo_root: Path, manifest: dict) -> None:
    expected = read_dvc_data_md5(repo_root, manifest["dvc_data_path"])
    assert manifest["dvc_data_path"] == "data/processed/train.parquet"
    assert manifest["dvc_data_md5"] == expected
    assert manifest["dvc_data_version"] == expected


def test_manifest_records_the_params_hash(repo_root: Path, manifest: dict) -> None:
    assert manifest["params_sha256"] == sha256_file(repo_root / "params.yaml")


def test_register_stage_keeps_the_bundle_out_of_the_dvc_cache(repo_root: Path) -> None:
    with (repo_root / "dvc.yaml").open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    outputs = config["stages"]["register"]["outs"]
    champion_out = next(item for item in outputs if "artifacts/champion/" in item)
    assert champion_out["artifacts/champion/"]["cache"] is False


def test_bundle_reproduces_champion_predictions(
    val_sample: pd.DataFrame,
    bundle: AqiBundle,
    scaler,
    feature_manifest: dict,
    reference_stats: dict,
) -> None:
    direct = mr.predict_regression(
        bundle.manifest["model"]["name"],
        bundle.model,
        val_sample,
        scaler,
        feature_manifest["feature_names"],
        float(reference_stats["target_aqi"]["max"]),
    )
    np.testing.assert_allclose(bundle.predict(val_sample), direct, rtol=0, atol=1e-10)


def test_predict_raw_rebuilds_features_like_the_pipeline(
    val_sample: pd.DataFrame,
    bundle: AqiBundle,
    params: dict,
    scaler,
    feature_manifest: dict,
    reference_stats: dict,
) -> None:
    rebuilt = mr.rebuild_features(
        val_sample,
        feature_manifest["feature_names"],
        params,
        reference_stats["train_medians"],
    )
    direct = mr.predict_regression(
        bundle.manifest["model"]["name"],
        bundle.model,
        rebuilt,
        scaler,
        feature_manifest["feature_names"],
        float(reference_stats["target_aqi"]["max"]),
    )
    np.testing.assert_allclose(bundle.predict_raw(val_sample), direct, rtol=0, atol=1e-8)


def test_residual_quantiles_cover_every_city(val_sample: pd.DataFrame, bundle: AqiBundle) -> None:
    assert "__global__" in bundle.residual_quantiles
    assert set(val_sample["city"].unique()) <= set(bundle.residual_quantiles)
    for quantiles in bundle.residual_quantiles.values():
        assert quantiles["q10"] <= quantiles["q50"] <= quantiles["q90"]


def test_prediction_interval_brackets_the_point_forecast(
    val_sample: pd.DataFrame, bundle: AqiBundle
) -> None:
    preds = bundle.predict(val_sample)
    lower, median, upper = bundle.predict_interval(val_sample)
    assert preds.shape == lower.shape == median.shape == upper.shape
    assert np.all(lower <= median + 1e-9)
    assert np.all(median <= upper + 1e-9)
    assert np.all(lower >= 0.0)
    assert np.all(upper <= bundle.aqi_max)


def test_history_seed_has_the_configured_lookback(
    repo_root: Path, val_sample: pd.DataFrame, bundle: AqiBundle
) -> None:
    seed = bundle.history_seed
    expected_cities = set(val_sample["city"].unique())
    assert set(seed["city"].unique()) == expected_cities
    counts = seed.groupby("city")["date"].count()
    assert bundle.manifest["history_seed_days"] == HISTORY_SEED_DAYS
    assert bundle.manifest["history_seed_days"] >= 45
    assert (counts == bundle.manifest["history_seed_days"]).all()
    assert int(counts.sum()) == bundle.manifest["history_seed_rows"]
    val_full = pd.read_parquet(repo_root / "data/processed/val.parquet")
    val_max = (
        pd.to_datetime(val_full["date"]).groupby(val_full["city"]).max().dt.strftime("%Y-%m-%d")
    )
    seed_max = pd.to_datetime(seed["date"]).groupby(seed["city"]).max().dt.strftime("%Y-%m-%d")
    pd.testing.assert_series_equal(seed_max, val_max, check_names=False)


def test_persistence_fallback_is_bundled(val_sample: pd.DataFrame, bundle: AqiBundle) -> None:
    expected = np.clip(
        np.asarray(bundle.fallback_model.predict(val_sample), dtype=float),
        0.0,
        bundle.aqi_max,
    )
    np.testing.assert_allclose(bundle.predict_fallback(val_sample), expected, rtol=0, atol=1e-12)


def test_bundle_round_trips_through_disk(
    tmp_path: Path, val_sample: pd.DataFrame, bundle: AqiBundle
) -> None:
    bundle.save(tmp_path, include_payload=True)
    loaded = AqiBundle.load(tmp_path)
    np.testing.assert_allclose(
        loaded.predict(val_sample), bundle.predict(val_sample), rtol=0, atol=0
    )
    assert loaded.feature_names == bundle.feature_names
    assert loaded.input_kind == bundle.input_kind
    assert (tmp_path / "MANIFEST.sha256").exists()
