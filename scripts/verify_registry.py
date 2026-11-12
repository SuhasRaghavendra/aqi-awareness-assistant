"""Reproduce the champion bundle in a temporary MLflow registry and verify parity.

Registration is normally local-only because ``mlflow.db`` is gitignored
(D-003). That makes it impossible to prove in CI that ``models:/...@champion``
still reproduces the committed bundle. This script closes the gap without a
shared registry:

1. create a throwaway SQLite tracking store and artifact root under a temp dir;
2. load the committed bundle with full hash verification;
3. log it as a pyfunc, register it, and set the ``champion`` alias;
4. load the alias back through MLflow and compare predictions on the committed
   fixture with the bundle's own predictions;
5. write ``reports/metrics/registry_parity.json`` and exit non-zero on any
   difference above ``1e-6``.

Nothing under ``mlruns/`` or ``mlflow.db`` is touched: every path is inside the
temporary directory, which is deleted on exit.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import tempfile
from pathlib import Path

# Allow `python scripts/verify_registry.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.models.package_model import check_gate_policy  # noqa: E402
from src.models.register_model import (  # noqa: E402
    MANIFEST_FILENAME,
    build_pip_requirements,
    log_pyfunc_bundle,
)
from src.serving.bundle import AqiBundle, sha256_file  # noqa: E402
from src.utils.io import atomic_write_json  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO_ROOT / "tests/fixtures/val_sample.parquet"
DEFAULT_REPORT = REPO_ROOT / "reports/metrics/registry_parity.json"
TOLERANCE = 1e-6
EXPERIMENT = "aqi-registry-parity"


def verify_registry(
    bundle_dir: Path | None = None,
    fixture_path: Path | None = None,
    model_name: str = "Indian-AQI-Predictor-Parity",
) -> dict:
    """Register the bundle in a temp registry and return the parity report."""
    bundle_dir = bundle_dir or REPO_ROOT / "artifacts/champion"
    fixture_path = fixture_path or DEFAULT_FIXTURE

    bundle = AqiBundle.load(bundle_dir)  # verifies every hash before unpickling
    manifest = bundle.manifest
    check_gate_policy(manifest)

    fixture = pd.read_parquet(fixture_path)
    expected = bundle.predict(fixture)

    with tempfile.TemporaryDirectory(prefix="aqi-registry-", ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.create_experiment(EXPERIMENT, artifact_location=(tmp_path / "artifacts").as_uri())
        mlflow.set_experiment(EXPERIMENT)

        client = MlflowClient()
        with mlflow.start_run(run_name="registry-parity"):
            model_info = log_pyfunc_bundle(bundle_dir, manifest)
            version = mlflow.register_model(
                model_info.model_uri,
                model_name,
                tags={
                    "manifest_sha256": sha256_file(bundle_dir / MANIFEST_FILENAME),
                    "bundle_sha256": manifest["bundle_sha256"],
                    "risk_card_sha": manifest["risk_card_sha"],
                    "champion": str(manifest["model"]["name"]),
                },
            )
        client.set_registered_model_alias(model_name, "champion", str(version.version))

        loaded = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
        actual = np.asarray(loaded.predict(fixture), dtype=float).reshape(-1)

        # Windows keeps the SQLite file locked while MLflow's engine pool is
        # open, which makes TemporaryDirectory cleanup fail. Dispose it before
        # leaving the context (private API, but the store exposes no public
        # close and nothing else in this process uses the throwaway URI).
        with contextlib.suppress(Exception):
            MlflowClient()._tracking_client.store._dispose_engine()

    if actual.shape != expected.shape:
        raise RuntimeError(f"Registry returned {actual.shape}, bundle returned {expected.shape}")
    max_diff = float(np.max(np.abs(actual - expected))) if len(actual) else 0.0
    report = {
        "schema_version": "1.0",
        "tracking_uri": "temporary sqlite (deleted)",
        "model_name": model_name,
        "model_version": str(version.version),
        "alias": "champion",
        "rows": int(len(fixture)),
        "fixture": str(fixture_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "fixture_sha256": sha256_file(fixture_path),
        "manifest_sha256": sha256_file(bundle_dir / MANIFEST_FILENAME),
        "bundle_sha256": manifest["bundle_sha256"],
        "max_abs_diff": max_diff,
        "tolerance": TOLERANCE,
        "passed": max_diff <= TOLERANCE,
        "pip_requirements": build_pip_requirements(manifest),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=REPO_ROOT / "artifacts/champion")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    report = verify_registry(args.bundle_dir, args.fixture)
    report_path = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    atomic_write_json(report_path, report)

    print("\n" + "=" * 78)
    print("REGISTRY PARITY (temporary sqlite)")
    print("=" * 78)
    print(f"  model:      {report['model_name']} v{report['model_version']} @champion")
    print(f"  rows:       {report['rows']}")
    print(f"  max |diff|: {report['max_abs_diff']:.3e} (tolerance {TOLERANCE:.0e})")
    print(f"  report:     {report_path}")
    print("=" * 78 + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
