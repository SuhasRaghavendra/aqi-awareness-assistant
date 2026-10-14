"""
src/utils/mlflow_utils.py
Helper utilities for MLflow experiment tracking.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import mlflow


def get_git_commit() -> str:
    """Return the current HEAD commit hash."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def setup_mlflow(
    experiment_name: str,
    tracking_uri: str = "sqlite:///mlflow.db",
    artifact_location: str = "./mlartifacts",
) -> str:
    """
    Configure MLflow tracking URI, create/get experiment.

    Args:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: Backend store URI.
        artifact_location: Root artifact directory.

    Returns:
        experiment_id as a string.
    """
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(
            experiment_name, artifact_location=artifact_location
        )
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_name)
    return exp_id


def log_standard_tags(
    step_id: str = "",
    logbook_entry: str = "",
    dvc_data_version: str = "",
    author: str = "",
    stage: str = "baseline",
    task: str = "",
    model_family: str = "",
) -> None:
    """
    Log standard MLflow tags required by the plan.

    Args:
        step_id: Logbook step number, e.g. '007'.
        logbook_entry: Path to the logbook markdown file.
        dvc_data_version: Git tag + md5 from .dvc files.
        author: Student name/ID.
        stage: Pipeline stage name.
        task: 'regression' or 'classification'.
        model_family: Model family name.
    """
    mlflow.set_tags(
        {
            "git_commit": get_git_commit(),
            "step_id": step_id,
            "logbook_entry": logbook_entry,
            "dvc_data_version": dvc_data_version,
            "author": author,
            "stage": stage,
            "task": task,
            "model_family": model_family,
        }
    )


def log_params_yaml(params_path: Path | None = None) -> None:
    """Log params.yaml as an MLflow artifact."""
    if params_path is None:
        params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    if params_path.exists():
        mlflow.log_artifact(str(params_path))
