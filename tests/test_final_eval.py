"""Gate 3: single-shot holdout evaluation (Plan A1)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.models import model_risk as mr
from src.models.final_eval import (
    FINAL_EVAL_RELPATH,
    compare_reports,
    verify_split_hashes,
)


def _write_splits(root: Path, rows: int = 6) -> dict[str, str]:
    target = root / "data" / "processed"
    target.mkdir(parents=True)
    hashes: dict[str, str] = {}
    for split in ("train", "val", "test"):
        frame = pd.DataFrame(
            {
                "city": ["Delhi", "Mumbai"] * (rows // 2),
                "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
                "target_aqi": [100.0 + index for index in range(rows)],
            }
        )
        path = target / f"{split}.parquet"
        frame.to_parquet(path, index=False)
        hashes[split] = mr.file_md5(path)
    return hashes


def test_split_hash_guard_accepts_frozen_hashes(tmp_path: Path) -> None:
    hashes = _write_splits(tmp_path)
    params = {"integrity": {f"{name}_md5": value for name, value in hashes.items()}}

    assert verify_split_hashes(tmp_path, params) == hashes


def test_split_hash_guard_aborts_when_one_byte_changes(tmp_path: Path) -> None:
    hashes = _write_splits(tmp_path)
    params = {"integrity": {f"{name}_md5": value for name, value in hashes.items()}}
    path = tmp_path / "data" / "processed" / "test.parquet"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0xFF
    path.write_bytes(bytes(payload))

    with pytest.raises(ValueError, match="refusing to evaluate a changed holdout"):
        verify_split_hashes(tmp_path, params)


def test_split_hash_guard_requires_all_three_splits(tmp_path: Path) -> None:
    hashes = _write_splits(tmp_path)
    (tmp_path / "data" / "processed" / "val.parquet").unlink()
    params = {"integrity": {f"{name}_md5": value for name, value in hashes.items()}}

    with pytest.raises(FileNotFoundError):
        verify_split_hashes(tmp_path, params)


def test_compare_reports_flags_candidate_data_and_metric_drift() -> None:
    base = {
        "candidate": {"name": "champion", "artifact_md5": "abc"},
        "splits": {"test": "deadbeef"},
        "metrics": {"mae": 20.0, "rmse": 30.0, "r2": 0.8},
        "skill_vs_persistence": 0.1,
        "per_city": [{"city": "Delhi"}],
    }
    assert compare_reports(base, json.loads(json.dumps(base))) == []

    changed = json.loads(json.dumps(base))
    changed["metrics"]["mae"] = 20.1
    changed["candidate"]["name"] = "other"
    problems = compare_reports(base, changed)

    assert any(problem.startswith("candidate.name") for problem in problems)
    assert any(problem.startswith("metrics.mae") for problem in problems)


@pytest.mark.model
def test_committed_holdout_report_matches_the_frozen_champion(
    repo_root: Path, params: dict, champion_bundle
) -> None:
    report = mr.read_json(repo_root / FINAL_EVAL_RELPATH)

    assert report["splits"]["test"] == params["integrity"]["test_md5"]
    assert report["splits"]["train"] == params["integrity"]["train_md5"]
    assert report["splits"]["val"] == params["integrity"]["val_md5"]
    assert report["candidate"]["name"] == champion_bundle.manifest["model"]["name"]
    assert report["counts"]["rows"] > 0
    assert len(report["per_city"]) == 9
    assert report["skill_ci_lo_95"] <= report["skill_vs_persistence"] <= report["skill_ci_hi_95"]


@pytest.mark.model
def test_committed_holdout_report_agrees_with_the_risk_card(
    repo_root: Path, risk_card: dict
) -> None:
    report = mr.read_json(repo_root / FINAL_EVAL_RELPATH)

    for key in ("mae", "rmse", "r2"):
        assert float(report["metrics"][key]) == pytest.approx(
            float(risk_card["model"]["test_metrics"][key]), abs=1e-6
        )
