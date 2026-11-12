"""Shared fixtures for the Phase 1 model-risk test-suite.

Artifacts that are required by the audit are loaded here without pytest.skip:
a missing model, split, manifest, scaler, reference-stat file, or risk card is a
test failure, not a silently skipped test.

Several fixtures fall back to the committed champion bundle
(``artifacts/champion``) when the DVC-managed ``data/processed`` outputs are not
present. That is what lets CI run the model gate on a clean checkout with only
the bundle, the committed metrics and the hash-pinned validation snapshot.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from src.config import load_params
from src.models import model_risk as mr
from src.serving.bundle import AqiBundle

REPO_ROOT = Path(__file__).resolve().parents[1]
VAL_SNAPSHOT = REPO_ROOT / "tests" / "fixtures" / "val_snapshot.parquet"
VAL_SNAPSHOT_MD5 = "462b7712b4cea342fe36ca80868a29f4"


def resolve_val_path(root: Path | None = None) -> Path:
    """Return the validation split, preferring a caller override and the snapshot.

    Order: ``AQI_VAL_PATH`` env var -> committed ``tests/fixtures/val_snapshot.parquet``
    -> DVC-managed ``data/processed/val.parquet``. CI therefore never needs the
    DVC remote just to run the service and model gates.
    """
    base = root or REPO_ROOT
    candidates = []
    override = os.environ.get("AQI_VAL_PATH")
    if override:
        candidates.append(Path(override))
    candidates.extend(
        [base / "tests/fixtures/val_snapshot.parquet", base / "data/processed/val.parquet"]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No validation split found; set AQI_VAL_PATH or commit tests/fixtures/val_snapshot.parquet"
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def params(repo_root: Path) -> dict:
    return load_params(repo_root / "params.yaml")


@pytest.fixture(scope="session")
def champion_bundle(repo_root: Path) -> AqiBundle:
    return AqiBundle.load(repo_root / "artifacts/champion")


@pytest.fixture(scope="session")
def feature_manifest(repo_root: Path, champion_bundle: AqiBundle) -> dict:
    """Feature list from ``data/processed`` or, on a clean checkout, the bundle."""
    path = repo_root / "data/processed/feature_list.json"
    if path.exists():
        return mr.load_manifest(path)
    return {
        "n_features": len(champion_bundle.feature_names),
        "feature_names": list(champion_bundle.feature_names),
        "generated_at": champion_bundle.manifest.get("created_at"),
    }


@pytest.fixture(scope="session")
def scaler(repo_root: Path, champion_bundle: AqiBundle):
    path = repo_root / "data/processed/scaler.joblib"
    return mr.load_scaler(path) if path.exists() else champion_bundle.scaler


@pytest.fixture(scope="session")
def reference_stats(repo_root: Path) -> dict:
    return mr.read_json(repo_root / "reports/metrics/reference_stats.json")


@pytest.fixture(scope="session")
def risk_card(repo_root: Path) -> dict:
    return mr.read_json(repo_root / "reports/metrics/model_risk_card.json")


@pytest.fixture(scope="session")
def val_frame(repo_root: Path) -> pd.DataFrame:
    """The hash-pinned validation slice (snapshot unless overridden)."""
    frame = pd.read_parquet(resolve_val_path(repo_root))
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame.sort_values(["city", "date"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def val_sample(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "tests/fixtures/val_sample.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Required test fixture is missing: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def champion(repo_root: Path, champion_bundle: AqiBundle) -> mr.ChampionArtifact:
    """Champion from ``models/`` or, on a clean checkout, from the bundle."""
    name = str(champion_bundle.manifest["model"]["name"])
    path = repo_root / "models" / f"{name}.joblib"
    if path.exists():
        return mr.load_champion_artifact(repo_root)
    return mr.ChampionArtifact(
        name=name,
        model=champion_bundle.model,
        path=path,
        val_metrics=dict(champion_bundle.manifest["model"].get("val_metrics", {})),
    )


@pytest.fixture(scope="session")
def test_df(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "data/processed/test.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Required holdout split is missing: {path}")
    return pd.read_parquet(path)
