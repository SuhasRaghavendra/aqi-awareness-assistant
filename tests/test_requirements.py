"""Phase 0.6: requirements pin consistency checks."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_EXACT_PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)$")

CORE_SERVE_PACKAGES = [
    "pandas",
    "numpy",
    "pyarrow",
    "scikit-learn",
    "lightgbm",
    "joblib",
    "pyyaml",
    "python-dotenv",
]


def _exact_pins(filename: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in (REPO_ROOT / filename).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _EXACT_PIN.match(line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def test_training_requirements_match_installed_versions() -> None:
    missing: list[str] = []
    mismatched: list[tuple[str, str, str]] = []
    for name, expected in _exact_pins("requirements.txt").items():
        try:
            actual = version(name)
        except PackageNotFoundError:
            missing.append(name)
            continue
        if actual != expected:
            mismatched.append((name, expected, actual))

    assert not missing, f"Training requirements are not installed: {missing}"
    assert not mismatched, f"Training requirement pin drift: {mismatched}"


def test_serve_requirements_use_the_training_pins() -> None:
    training = _exact_pins("requirements.txt")
    serving = _exact_pins("requirements-serve.txt")
    for name in CORE_SERVE_PACKAGES:
        assert serving.get(name) == training.get(name), (
            f"{name} differs between training and serving requirements: "
            f"training={training.get(name)!r}, serving={serving.get(name)!r}"
        )
