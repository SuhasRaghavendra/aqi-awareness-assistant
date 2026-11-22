"""Publish DVC data versions as an MLflow experiment for easy browsing.

The model registry remains the source of model versions. This command creates
one lightweight MLflow run per distinct DVC hash already attached to registered
model versions, so data lineage is visible in the MLflow UI without copying
datasets into MLflow or deploying anything.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.register_model import DEFAULT_MODEL_NAME, DEFAULT_TRACKING_URI

EXPERIMENT_NAME = "aqi-data-versions"
DATA_VERSION_TAG = "dvc_data_version"


def publish(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    model_name: str = DEFAULT_MODEL_NAME,
) -> int:
    """Create missing MLflow runs for distinct DVC versions in the registry."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = (
        experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT_NAME)
    )

    versions = client.search_model_versions(f"name='{model_name}'")
    grouped: dict[str, list] = defaultdict(list)
    for version in versions:
        data_version = version.tags.get(DATA_VERSION_TAG)
        if data_version:
            grouped[data_version].append(version)

    existing = client.search_runs(
        [experiment_id],
        f"tags.{DATA_VERSION_TAG} != ''",
    )
    existing_versions = {
        run.data.tags.get(DATA_VERSION_TAG)
        for run in existing
        if run.data.tags.get(DATA_VERSION_TAG)
    }

    created = 0
    for data_version, model_versions in sorted(grouped.items()):
        if data_version in existing_versions:
            continue
        model_version_numbers = sorted(str(version.version) for version in model_versions)
        git_commits = sorted(
            {
                version.tags["git_commit"]
                for version in model_versions
                if version.tags.get("git_commit")
            }
        )
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"DVC {data_version[:12]}",
            tags={
                DATA_VERSION_TAG: data_version,
                "source": "MLflow model registry",
                "model_name": model_name,
            },
        ):
            mlflow.log_params(
                {
                    "dvc_data_version": data_version,
                    "model_versions": ",".join(model_version_numbers),
                    "git_commits": ",".join(git_commits),
                    "model_count": len(model_version_numbers),
                }
            )
            mlflow.log_dict(
                {
                    "model_name": model_name,
                    "dvc_data_version": data_version,
                    "model_versions": model_version_numbers,
                    "git_commits": git_commits,
                },
                "data_version.json",
            )
        created += 1

    print(
        f"Published {created} new DVC data version(s) to " f"MLflow experiment '{EXPERIMENT_NAME}'."
    )
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()
    publish(args.tracking_uri, args.model_name)


if __name__ == "__main__":
    main()
