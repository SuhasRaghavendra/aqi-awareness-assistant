"""
scripts/sync_mlflow_remote.py
------------------------------
Copies experiments, runs, metrics, tags, and registered model versions
from the local SQLite database (sqlite:///mlflow.db) to a remote MLflow tracking
server (such as DagsHub MLflow or any hosted MLflow server).

Usage:
    python scripts/sync_mlflow_remote.py --remote https://dagshub.com/SuhasRaghavendra/aqi-awareness-assistant.mlflow
"""

from __future__ import annotations

import argparse
import contextlib
import os

from mlflow.tracking import MlflowClient


def sync_experiments_and_models(
    src_uri: str = "sqlite:///mlflow.db",
    dest_uri: str | None = None,
    username: str | None = None,
    token: str | None = None,
) -> None:
    if not dest_uri:
        dest_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            "https://dagshub.com/SuhasRaghavendra/aqi-awareness-assistant.mlflow",
        )

    if username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
    if token:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    print("=" * 70)
    print(f"Source MLflow:      {src_uri}")
    print(f"Destination MLflow: {dest_uri}")
    print("=" * 70)

    src_client = MlflowClient(tracking_uri=src_uri)
    dest_client = MlflowClient(tracking_uri=dest_uri)

    src_experiments = src_client.search_experiments()
    print(f"Found {len(src_experiments)} experiments locally.\n")

    run_id_map: dict[str, str] = {}

    for exp in src_experiments:
        if exp.name == "Default":
            continue

        print(f"Syncing experiment: '{exp.name}'...")
        dest_exp = dest_client.get_experiment_by_name(exp.name)
        if dest_exp is None:
            dest_exp_id = dest_client.create_experiment(exp.name)
        else:
            dest_exp_id = dest_exp.experiment_id

        src_runs = src_client.search_runs(experiment_ids=[exp.experiment_id], max_results=1000)
        print(f"  Found {len(src_runs)} runs in '{exp.name}'. Uploading...")

        for r in src_runs:
            # Create corresponding run on destination
            try:
                dest_run = dest_client.create_run(
                    experiment_id=dest_exp_id,
                    run_name=r.info.run_name,
                    start_time=r.info.start_time,
                    tags={k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")},
                )
                run_id_map[r.info.run_id] = dest_run.info.run_id

                # Log params
                for k, v in r.data.params.items():
                    dest_client.log_param(dest_run.info.run_id, k, v)

                # Log metrics
                for k, v in r.data.metrics.items():
                    dest_client.log_metric(dest_run.info.run_id, k, v)

                dest_client.set_terminated(dest_run.info.run_id, status=r.info.status)
                print(f"    Synced run: {r.info.run_name} -> id {dest_run.info.run_id[:8]}")
            except Exception as e:
                print(f"    [WARN] Could not sync run {r.info.run_name}: {e}")

    # Sync Registered Models
    print("\nSyncing Registered Models...")
    src_models = src_client.search_registered_models()
    for m in src_models:
        print(f"Syncing model: '{m.name}'...")
        with contextlib.suppress(Exception):
            dest_client.create_registered_model(m.name, description=m.description or "")

        versions = src_client.search_model_versions(f"name='{m.name}'")
        print(f"  Found {len(versions)} versions for '{m.name}'.")
        for v in sorted(versions, key=lambda x: int(x.version)):
            dest_run_id = run_id_map.get(v.run_id, "")
            try:
                dest_v = dest_client.create_model_version(
                    name=m.name,
                    source=f"runs:/{dest_run_id}/model" if dest_run_id else v.source,
                    run_id=dest_run_id,
                    description=v.description
                    or f"Model version {v.version} ({v.tags.get('champion', '')})",
                    tags=v.tags,
                )
                print(f"    Synced Model Version {v.version} -> Remote Version {dest_v.version}")
                for alias in v.aliases:
                    dest_client.set_registered_model_alias(m.name, alias, dest_v.version)
                    print(f"      Set alias @{alias} on version {dest_v.version}")
            except Exception as e:
                print(f"    [WARN] Version {v.version}: {e}")

    print("\n" + "=" * 70)
    print("ALL RUNS AND MODEL VERSIONS SYNCED TO REMOTE MLFLOW!")
    print(f"Open: {dest_uri.replace('.mlflow', '/experiments')}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="sqlite:///mlflow.db")
    parser.add_argument("--remote", default=None)
    parser.add_argument("--username", default=None)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    sync_experiments_and_models(
        src_uri=args.src,
        dest_uri=args.remote,
        username=args.username,
        token=args.token,
    )
