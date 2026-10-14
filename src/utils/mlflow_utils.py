"""
src/utils/mlflow_utils.py
-------------------------
MLflow helper utilities: experiment setup, run tagging, and artifact logging.

All runs are tagged with step_id, logbook_entry, git_commit, and
dvc_data_version as required by Section 4.3 item 9 of the implementation plan.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import mlflow

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_commit_full() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=_REPO_ROOT,
        )
        return r.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def setup_experiment(name: str, tracking_uri: str = "sqlite:///mlflow.db") -> str:
    """Create or retrieve an MLflow experiment and return its ID."""
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment(name)
    logger.info("MLflow experiment: '%s' (id=%s)", name, exp.experiment_id)
    return exp.experiment_id


def start_run(
    *,
    step_id: str,
    logbook_entry: str,
    dvc_data_version: str = "unknown",
    tags: dict[str, str] | None = None,
    **kwargs: Any,
) -> mlflow.ActiveRun:
    """Start an MLflow run with mandatory step/logbook tags pre-applied."""
    base_tags = {
        "step_id": step_id,
        "logbook_entry": logbook_entry,
        "git_commit": _git_commit_full(),
        "dvc_data_version": dvc_data_version,
    }
    if tags:
        base_tags.update(tags)
    run = mlflow.start_run(tags=base_tags, **kwargs)
    logger.info("MLflow run started: %s", run.info.run_id)
    return run


def log_params_yaml(run: mlflow.ActiveRun | None = None) -> None:
    """Log params.yaml as an MLflow artifact and flatten top-level keys as params."""
    import yaml  # lazy import

    params_path = _REPO_ROOT / "params.yaml"
    if not params_path.exists():
        logger.warning("params.yaml not found; skipping MLflow param logging")
        return
    with params_path.open() as fh:
        params = yaml.safe_load(fh)
    # Flatten one level for MLflow params panel
    flat: dict[str, Any] = {}
    for key, val in params.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                flat[f"{key}.{subkey}"] = subval
        else:
            flat[key] = val
    mlflow.log_params({k: str(v) for k, v in flat.items()})
    mlflow.log_artifact(str(params_path), artifact_path="config")
    logger.info("params.yaml logged to MLflow")
