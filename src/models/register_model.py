"""Phase 2: register the packaged champion in the MLflow model registry."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.models.model_risk import file_md5
from src.models.package_model import (
    build_bundle,
    check_gate_policy,
    read_dvc_data_md5,
    repo_root,
)
from src.serving.bundle import (
    MANIFEST_FILENAME,
    MANIFEST_SHA_FILENAME,
    AqiBundle,
    AqiBundlePyfunc,
    sha256_file,
)
from src.utils.mlflow_utils import setup_experiment, start_run

DEFAULT_MODEL_NAME = "Indian-AQI-Predictor"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
LOGBOOK_ENTRY = "012-registry-and-bundling.md"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_bundle_and_manifest(bundle_dir: Path) -> tuple[AqiBundle, dict[str, Any], str]:
    """Load a bundle and verify that MANIFEST.json matches MANIFEST.sha256."""
    manifest_path = bundle_dir / MANIFEST_FILENAME
    sha_path = bundle_dir / MANIFEST_SHA_FILENAME
    if not manifest_path.exists() or not sha_path.exists():
        raise FileNotFoundError(f"Bundle manifest is incomplete: {bundle_dir}")
    expected = sha_path.read_text(encoding="utf-8").strip()
    actual = sha256_file(manifest_path)
    if expected != actual:
        raise RuntimeError(
            f"MANIFEST sha mismatch for {manifest_path}: expected {expected}, got {actual}"
        )
    bundle = AqiBundle.load(bundle_dir)
    return bundle, read_json(manifest_path), actual


def verify_bundle_consistency(manifest: dict[str, Any], base: Path) -> None:
    """Tie the bundle manifest to the committed risk card, dvc.lock and manifest."""
    risk_path = base / "reports/metrics/model_risk_card.json"
    if sha256_file(risk_path) != manifest["risk_card_sha"]:
        raise RuntimeError("Bundle risk_card_sha does not match the committed risk card")
    risk_card = read_json(risk_path)
    status = {gate: item["status"] for gate, item in risk_card["gates"].items()}
    if status != manifest["gate_status"]:
        raise RuntimeError("Bundle gate_status does not match the committed risk card")
    if risk_card["model"]["name"] != manifest["model"]["name"]:
        raise RuntimeError("Bundle champion does not match the committed risk card")
    data_path = manifest.get("dvc_data_path", "data/processed/train.parquet")
    if read_dvc_data_md5(base, data_path) != manifest["dvc_data_md5"]:
        raise RuntimeError("Bundle dvc_data_md5 does not match dvc.lock")
    if manifest.get("params_sha256") != sha256_file(base / "params.yaml"):
        raise RuntimeError("Bundle params_sha256 does not match params.yaml")

    feature_manifest = read_json(base / "data/processed/feature_list.json")
    feature_names = feature_manifest["feature_names"]
    if manifest["features_count"] != len(feature_names):
        raise RuntimeError("Bundle features_count does not match the feature manifest")
    digest = hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest()
    if manifest["feature_names_sha256"] != digest:
        raise RuntimeError("Bundle feature_names_sha256 does not match the feature manifest")
    model_path = base / manifest["model"]["path"]
    if file_md5(model_path) != manifest["model"]["artifact_md5"]:
        raise RuntimeError("Bundle champion artifact hash does not match the manifest")


def verify_model_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    manifest: dict[str, Any],
    manifest_sha: str,
) -> None:
    """Verify registry tags still describe the bundle being promoted."""
    model_version = client.get_model_version(model_name, version)
    tags = model_version.tags or {}
    if tags.get("manifest_sha256") != manifest_sha:
        raise RuntimeError(
            f"Model version {version} manifest sha {tags.get('manifest_sha256')!r} "
            f"does not match bundle sha {manifest_sha!r}"
        )
    if tags.get("risk_card_sha") != manifest["risk_card_sha"]:
        raise RuntimeError(
            f"Model version {version} risk-card sha does not match the bundle manifest"
        )


def promote_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    manifest: dict[str, Any],
    manifest_sha: str,
) -> None:
    """Validate gates and manifest identity, then set the champion alias."""
    check_gate_policy(manifest)
    verify_model_version(client, model_name, version, manifest, manifest_sha)
    client.set_registered_model_alias(model_name, "champion", version)
    print(f"Promoted {model_name} v{version} to @champion")


def build_pip_requirements(manifest: dict[str, Any]) -> list[str]:
    versions = manifest.get("package_versions", {})
    return [
        f"{name}=={version}"
        for name, version in versions.items()
        if version and version != "unknown"
    ]


def log_pyfunc_bundle(bundle_path: Path, manifest: dict[str, Any]) -> Any:
    """Log the bundle as a pyfunc model, supporting MLflow 2.x and 3.x APIs."""
    parameters = inspect.signature(mlflow.pyfunc.log_model).parameters
    kwargs: dict[str, Any] = {
        "python_model": AqiBundlePyfunc(),
        "artifacts": {"bundle_dir": str(bundle_path)},
        "pip_requirements": build_pip_requirements(manifest),
    }
    if "name" in parameters:
        kwargs["name"] = "model"
    else:
        kwargs["artifact_path"] = "model"
    return mlflow.pyfunc.log_model(**kwargs)


def register(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    model_name: str = DEFAULT_MODEL_NAME,
    bundle_dir: Path | None = None,
    promote: bool = False,
) -> str:
    """Register the packaged bundle and set the challenger alias."""
    base = repo_root()
    bundle_path = bundle_dir or base / "artifacts/champion"
    bundle, manifest, manifest_sha = load_bundle_and_manifest(bundle_path)
    verify_bundle_consistency(manifest, base)
    check_gate_policy(manifest)

    setup_experiment("aqi-serving", tracking_uri)
    tags = {
        "features_count": str(manifest["features_count"]),
        "target": "next_day_aqi",
        "risk_card_sha": manifest["risk_card_sha"],
        "manifest_sha256": manifest_sha,
        "dvc_data_md5": manifest["dvc_data_md5"],
        "params_sha256": manifest["params_sha256"],
        "champion": str(manifest["model"]["name"]),
        "gate_status": json.dumps(manifest["gate_status"], sort_keys=True),
        "gate_waivers": ",".join(sorted(manifest.get("gate_waivers", {}))),
    }

    with start_run(
        step_id="012",
        logbook_entry=LOGBOOK_ENTRY,
        dvc_data_version=manifest["dvc_data_md5"],
        tags=tags,
    ) as run:
        mlflow.log_params(
            {
                "features_count": manifest["features_count"],
                "champion": manifest["model"]["name"],
                "target": "next_day_aqi",
                "dvc_data_path": manifest["dvc_data_path"],
                "dvc_data_md5": manifest["dvc_data_md5"],
                "params_sha256": manifest["params_sha256"],
                "git_commit": manifest["git_commit"],
            }
        )
        for filename in (MANIFEST_FILENAME, MANIFEST_SHA_FILENAME):
            mlflow.log_artifact(str(bundle_path / filename), artifact_path="bundle")
        model_info = log_pyfunc_bundle(bundle_path, manifest)
        model_version = mlflow.register_model(
            model_info.model_uri,
            model_name,
            tags={
                "features_count": str(manifest["features_count"]),
                "target": "next_day_aqi",
                "risk_card_sha": manifest["risk_card_sha"],
                "manifest_sha256": manifest_sha,
                "dvc_data_md5": manifest["dvc_data_md5"],
                "dvc_data_version": manifest["dvc_data_md5"],
                "params_sha256": manifest["params_sha256"],
                "git_commit": manifest["git_commit"],
                "champion": str(manifest["model"]["name"]),
            },
        )
        run_id = run.info.run_id

    client = MlflowClient()
    client.set_registered_model_alias(model_name, "challenger", str(model_version.version))
    print(f"Run: {run_id}")
    print(f"Registered {model_name} v{model_version.version} as @challenger")
    if promote:
        promote_version(client, model_name, str(model_version.version), manifest, manifest_sha)
    return str(model_version.version)


def promote_challenger(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    model_name: str = DEFAULT_MODEL_NAME,
    bundle_dir: Path | None = None,
) -> str:
    """Promote the current challenger after validating the committed bundle."""
    base = repo_root()
    bundle_path = bundle_dir or base / "artifacts/champion"
    _, manifest, manifest_sha = load_bundle_and_manifest(bundle_path)
    verify_bundle_consistency(manifest, base)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, "challenger")
    promote_version(client, model_name, str(model_version.version), manifest, manifest_sha)
    return str(model_version.version)


def verify_champion(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    model_name: str = DEFAULT_MODEL_NAME,
    bundle_dir: Path | None = None,
    fixture_path: Path | None = None,
) -> float:
    """Load @champion through MLflow and reproduce bundle predictions."""
    base = repo_root()
    bundle_path = bundle_dir or base / "artifacts/champion"
    fixture = fixture_path or base / "tests/fixtures/val_sample.parquet"
    bundle = AqiBundle.load(bundle_path)
    verify_bundle_consistency(bundle.manifest, base)
    frame = pd.read_parquet(fixture)
    expected = bundle.predict(frame)

    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
    actual = np.asarray(model.predict(frame), dtype=float)
    max_diff = float(np.max(np.abs(actual - expected)))
    if max_diff > 1e-6:
        raise AssertionError(
            f"Registry predictions differ from bundle by {max_diff:.3e} (limit 1e-6)"
        )
    print(f"Loaded models:/{model_name}@champion on {len(frame)} rows; max diff {max_diff:.3e}")
    return max_diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--promote-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.promote and args.promote_only:
        parser.error("--promote and --promote-only are mutually exclusive")
    if args.build and args.bundle_dir:
        parser.error("--build writes artifacts/champion and cannot be combined with --bundle-dir")
    if args.build and (args.promote_only or args.verify):
        parser.error("--build is only valid for registration")

    if args.promote_only:
        promote_challenger(args.tracking_uri, args.model_name, args.bundle_dir)
    elif args.verify:
        verify_champion(args.tracking_uri, args.model_name, args.bundle_dir)
    else:
        bundle_dir = args.bundle_dir
        if args.build:
            _, _, bundle_dir = build_bundle()
        register(args.tracking_uri, args.model_name, bundle_dir, args.promote)


if __name__ == "__main__":
    main()
