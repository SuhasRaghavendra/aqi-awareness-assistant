"""Phase 0-3 integrity audit tests.

``scripts/audit_integrity.py`` is the executable specification for the
project's no-leakage / no-corruption / reproducible-build claims. These tests
run it against the real artifacts and prove that it actually fails when a
frozen input drifts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import audit_integrity as ai

pytestmark = [pytest.mark.data, pytest.mark.model]


def test_full_integrity_audit_passes(repo_root: Path) -> None:
    report = ai.run_audit(repo_root)
    assert report["passed"], report["failures"]
    assert report["checks_failed"] == 0
    assert report["checks_total"] >= 50


def test_bundle_only_audit_passes(repo_root: Path) -> None:
    report = ai.run_audit(repo_root, bundle_only=True)
    assert report["passed"], report["failures"]


def test_frozen_split_hashes_match_params(repo_root: Path, params: dict) -> None:
    for name in ("train", "val", "test"):
        path = repo_root / "data/processed" / f"{name}.parquet"
        digest = hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - DVC identity
        assert (
            digest == params["integrity"][f"{name}_md5"]
        ), f"{name}.parquet drifted from the frozen hash; any change needs an ADR"


def test_audit_detects_a_split_hash_mismatch(
    repo_root: Path, params: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    tampered = {**params, "integrity": {**params["integrity"], "train_md5": "0" * 32}}
    monkeypatch.setattr(ai, "load_params", lambda path: tampered)  # noqa: ARG005

    audit = ai.Audit()
    ai.audit_splits(audit, repo_root)

    failures = {item["check"] for item in audit.failures()}
    assert "split.train_md5_matches_params" in failures
