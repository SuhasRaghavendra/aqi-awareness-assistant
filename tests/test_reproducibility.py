"""Reproducible build timestamps (manifest determinism)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.utils.reproducibility import build_timestamp, git_commit_timestamp


def test_source_date_epoch_overrides_the_commit_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    assert build_timestamp(tmp_path) == "2023-11-14T22:13:20+00:00"


def test_build_timestamp_is_stable_for_the_same_commit(repo_root: Path) -> None:
    first = build_timestamp(repo_root)
    second = build_timestamp(repo_root)
    assert first == second
    commit_timestamp = git_commit_timestamp(repo_root)
    if commit_timestamp is not None:
        assert first == commit_timestamp


def test_build_timestamp_falls_back_without_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    timestamp = build_timestamp(tmp_path)
    assert timestamp
    assert isinstance(timestamp, str)


def test_git_commit_timestamp_matches_git(repo_root: Path) -> None:
    expected = subprocess.run(
        ["git", "log", "-1", "--format=%cI"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert git_commit_timestamp(repo_root) == expected
