"""Shared fixtures for the Phase 1 model-risk test-suite.

Artifacts that are required by the audit are loaded here without pytest.skip:
a missing model, split, manifest, scaler, reference-stat file, or risk card is a
test failure, not a silently skipped test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import load_params
from src.models import model_risk as mr

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def params(repo_root: Path) -> dict:
    return load_params(repo_root / "params.yaml")


@pytest.fixture(scope="session")
def feature_manifest(repo_root: Path) -> dict:
    return mr.load_manifest(repo_root / "data/processed/feature_list.json")


@pytest.fixture(scope="session")
def scaler(repo_root: Path):
    return mr.load_scaler(repo_root / "data/processed/scaler.joblib")


@pytest.fixture(scope="session")
def reference_stats(repo_root: Path) -> dict:
    return mr.read_json(repo_root / "reports/metrics/reference_stats.json")


@pytest.fixture(scope="session")
def risk_card(repo_root: Path) -> dict:
    return mr.read_json(repo_root / "reports/metrics/model_risk_card.json")


@pytest.fixture(scope="session")
def val_sample(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "tests/fixtures/val_sample.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Required test fixture is missing: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def champion(repo_root: Path) -> mr.ChampionArtifact:
    return mr.load_champion_artifact(repo_root)


@pytest.fixture(scope="session")
def test_df(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "data/processed/test.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Required holdout split is missing: {path}")
    return pd.read_parquet(path)
