"""Single-shot holdout evaluation for a frozen champion (Plan A1).

The test split is observed exactly once per frozen candidate.  This module is
the only place that turns that observation into numbers:

* it refuses to run unless the train/val/test md5s match ``params.yaml``;
* it records the candidate identity (name + artifact md5) and the split hashes
  in the report, so the numbers cannot be detached from what produced them;
* it computes regression metrics, the weekly block-bootstrap skill interval and
  the per-city rows that the risk card consumes.

``scripts/final_test_eval.py`` is the CLI; ``--check`` re-derives the report and
fails if the committed numbers drift by more than ``1e-9``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import load_params
from src.models import model_risk as mr
from src.utils.io import atomic_write_json

FINAL_EVAL_RELPATH = "reports/metrics/final_test_eval.json"


def verify_split_hashes(
    root: Path | None = None, params: dict[str, Any] | None = None
) -> dict[str, str]:
    """Return the split md5s, aborting when any differs from the frozen value."""
    base = Path(root) if root is not None else mr.repo_root()
    params = params or load_params(base / "params.yaml")
    expected = params.get("integrity", {})
    actual: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = base / "data" / "processed" / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Required split is missing: {path}")
        digest = mr.file_md5(path)
        frozen = expected.get(f"{split}_md5")
        if frozen and digest != frozen:
            raise ValueError(
                f"{split} split md5 {digest} does not match the frozen "
                f"{frozen} in params.yaml integrity.{split}_md5; refusing to "
                "evaluate a changed holdout"
            )
        actual[split] = digest
    return actual


def compute_final_test_eval(
    root: Path | None = None,
    champion: mr.ChampionArtifact | None = None,
) -> dict[str, Any]:
    """Compute the authoritative holdout report for the frozen champion."""
    base = Path(root) if root is not None else mr.repo_root()
    params = load_params(base / "params.yaml")
    hashes = verify_split_hashes(base, params)

    feature_manifest = mr.load_manifest(base / "data/processed/feature_list.json")
    feature_names = list(feature_manifest["feature_names"])
    scaler = mr.load_scaler(base / "data/processed/scaler.joblib")
    champ = champion or mr.load_champion_artifact(base)

    train_df = pd.read_parquet(base / "data/processed/train.parquet")
    test_df = pd.read_parquet(base / "data/processed/test.parquet")
    aqi_max = float(train_df["target_aqi"].max())

    persistence_path = base / "models/persistence.joblib"
    if not persistence_path.exists():
        raise FileNotFoundError(f"Persistence benchmark is missing: {persistence_path}")
    persistence = joblib.load(persistence_path)

    y_true = test_df["target_aqi"].to_numpy(dtype=float)
    persist_pred = mr.predict_regression(
        "persistence", persistence, test_df, scaler, feature_names, aqi_max
    )
    pred = mr.predict_regression(champ.name, champ.model, test_df, scaler, feature_names, aqi_max)

    metrics = mr.regression_metrics(y_true, pred)
    persist_metrics = mr.regression_metrics(y_true, persist_pred)
    skill = mr.skill_vs_persistence(y_true, pred, persist_pred)
    ci_lo, ci_hi = mr.bootstrap_skill_ci(y_true, pred, persist_pred, test_df["date"].to_numpy())
    per_city = mr.per_city_records(test_df, pred, persist_pred)

    split = params.get("split", {})
    return {
        "schema_version": "1.0",
        "generated_at": mr.frame_timestamp(test_df),
        "git_commit": mr.git_commit(base),
        "candidate": {
            "name": champ.name,
            "path": str(champ.path.relative_to(base)).replace("\\", "/"),
            "artifact_md5": mr.file_md5(champ.path),
            "val_metrics": {
                key: round(float(value), 4)
                for key, value in champ.val_metrics.items()
                if isinstance(value, (int, float))
            },
        },
        "splits": {**hashes, "frozen_in": "params.yaml:integrity"},
        "counts": {
            "rows": int(len(test_df)),
            "cities": int(test_df["city"].nunique()),
            "dates": int(test_df["date"].nunique()),
        },
        "split_dates": {
            "train_end": split.get("train_end"),
            "val_end": split.get("val_end"),
            "test_start": str(pd.to_datetime(test_df["date"]).min().date()),
            "test_end": str(pd.to_datetime(test_df["date"]).max().date()),
        },
        "metrics": {
            "mae": round(metrics["mae"], 3),
            "rmse": round(metrics["rmse"], 3),
            "r2": round(metrics["r2"], 4),
        },
        "persistence_metrics": {
            "mae": round(persist_metrics["mae"], 3),
            "rmse": round(persist_metrics["rmse"], 3),
            "r2": round(persist_metrics["r2"], 4),
        },
        "skill_vs_persistence": round(skill, 4),
        "skill_ci_lo_95": round(ci_lo, 4),
        "skill_ci_hi_95": round(ci_hi, 4),
        "ci_method": "weekly block bootstrap, 2000 resamples",
        "per_city": per_city,
    }


def write_final_test_eval(
    root: Path | None = None,
    output: Path | None = None,
    champion: mr.ChampionArtifact | None = None,
) -> tuple[dict[str, Any], Path]:
    """Compute and atomically write ``final_test_eval.json``."""
    base = Path(root) if root is not None else mr.repo_root()
    report = compute_final_test_eval(base, champion=champion)
    path = Path(output) if output is not None else base / FINAL_EVAL_RELPATH
    if not path.is_absolute():
        path = base / path
    atomic_write_json(path, report)
    return report, path


def compare_reports(existing: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    """Return human-readable differences between two holdout reports."""
    problems: list[str] = []
    for key in ("artifact_md5", "name"):
        if existing.get("candidate", {}).get(key) != fresh.get("candidate", {}).get(key):
            problems.append(f"candidate.{key} changed")
    if existing.get("splits", {}).get("test") != fresh.get("splits", {}).get("test"):
        problems.append("splits.test changed")
    for key in ("mae", "rmse", "r2"):
        old = existing.get("metrics", {}).get(key)
        new = fresh.get("metrics", {}).get(key)
        if old is None or new is None or abs(float(old) - float(new)) > 1e-9:
            problems.append(f"metrics.{key}: {old!r} -> {new!r}")
    if existing.get("skill_vs_persistence") != fresh.get("skill_vs_persistence"):
        problems.append("skill_vs_persistence changed")
    if len(existing.get("per_city", [])) != len(fresh.get("per_city", [])):
        problems.append("per_city row count changed")
    return problems
