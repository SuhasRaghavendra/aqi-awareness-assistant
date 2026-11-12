"""Registry parity (Plan B6 / V4).

The real registry is local-only (`mlflow.db` is gitignored, D-003), so CI proves
parity by registering the committed bundle into a throwaway SQLite store and
loading it back through `models:/...@champion`.
"""

from __future__ import annotations

import pytest

from scripts.verify_registry import TOLERANCE, verify_registry

pytestmark = pytest.mark.registry


@pytest.fixture(scope="module")
def report() -> dict:
    return verify_registry()


def test_registry_reproduces_the_bundle(report: dict) -> None:
    assert report["passed"], report
    assert report["max_abs_diff"] <= TOLERANCE
    assert report["rows"] >= 50
    assert report["alias"] == "champion"
    assert report["model_version"]


def test_registry_records_the_bundle_identity(report: dict) -> None:
    assert report["bundle_sha256"]
    assert report["manifest_sha256"]
    assert report["fixture_sha256"]
    assert report["tolerance"] == TOLERANCE
    assert report["tracking_uri"].startswith("temporary")
