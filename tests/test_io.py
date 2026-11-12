"""Atomic-write contract (Phase 0-3 integrity).

Every hashed artifact is written through ``src.utils.io``; these tests prove
that an interrupted write cannot destroy the previous good file and that text
artifacts are byte-identical across platforms.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.utils import io as io_module
from src.utils.io import (
    atomic_write_bytes,
    atomic_write_joblib,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    replace_directory,
)


def _temp_files(directory) -> list:
    return sorted(p.name for p in directory.iterdir() if p.name.endswith(".tmp"))


def test_atomic_write_bytes_replaces_content(tmp_path) -> None:
    target = tmp_path / "artifact.bin"
    atomic_write_bytes(target, b"first")
    atomic_write_bytes(target, b"second")
    assert target.read_bytes() == b"second"


def test_failed_replace_preserves_the_previous_file_and_cleans_up(tmp_path, monkeypatch) -> None:
    target = tmp_path / "artifact.json"
    atomic_write_json(target, {"version": 1})
    before = target.read_bytes()

    def boom(src, dst):  # noqa: ARG001 - signature must match os.replace
        raise OSError("disk full")

    monkeypatch.setattr(io_module.os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_json(target, {"version": 2})

    assert target.read_bytes() == before
    assert _temp_files(tmp_path) == []


def test_failed_parquet_write_preserves_the_previous_file(tmp_path, monkeypatch) -> None:
    target = tmp_path / "frame.parquet"
    atomic_write_parquet(pd.DataFrame({"a": [1, 2, 3]}), target)
    before = target.read_bytes()

    def boom(frame, path, index=False):  # noqa: ARG001
        raise RuntimeError("writer crashed")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(RuntimeError):
        atomic_write_parquet(pd.DataFrame({"a": [9]}), target)

    assert target.read_bytes() == before
    assert _temp_files(tmp_path) == []


def test_failed_joblib_write_preserves_the_previous_file(tmp_path, monkeypatch) -> None:
    target = tmp_path / "model.joblib"
    atomic_write_joblib({"version": 1}, target)
    before = target.read_bytes()

    def boom(payload, path):  # noqa: ARG001
        raise RuntimeError("pickler crashed")

    monkeypatch.setattr("joblib.dump", boom)
    with pytest.raises(RuntimeError):
        atomic_write_joblib({"version": 2}, target)

    assert target.read_bytes() == before
    assert _temp_files(tmp_path) == []


def test_text_writes_are_lf_on_every_platform(tmp_path) -> None:
    target = tmp_path / "artifact.txt"
    atomic_write_text(target, "one\ntwo\n")
    assert target.read_bytes() == b"one\ntwo\n"
    assert b"\r\n" not in target.read_bytes()


def test_text_writer_can_request_crlf_explicitly(tmp_path) -> None:
    target = tmp_path / "artifact.txt"
    atomic_write_text(target, "one\ntwo\n", newline="\r\n")
    assert target.read_bytes() == b"one\r\ntwo\r\n"


def test_json_writer_matches_json_dumps_bytes(tmp_path) -> None:
    payload = {"b": 1, "a": [1, 2]}
    target = tmp_path / "artifact.json"
    atomic_write_json(target, payload)
    assert target.read_bytes() == json.dumps(payload, indent=2).encode("utf-8")


def test_replace_directory_keeps_the_old_copy_until_the_swap_succeeds(tmp_path) -> None:
    target = tmp_path / "champion"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".champion.staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")

    replace_directory(staging, target)

    assert not staging.exists()
    assert not (tmp_path / "champion.bak").exists()
    assert sorted(p.name for p in target.iterdir()) == ["new.txt"]


def test_replace_directory_restores_the_old_copy_on_failure(tmp_path, monkeypatch) -> None:
    target = tmp_path / "champion"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".champion.staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")

    real_replace = io_module.os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:  # first rename (old -> .bak) succeeds, second fails
            raise OSError("cannot publish")
        return real_replace(src, dst)

    monkeypatch.setattr(io_module.os, "replace", flaky)
    with pytest.raises(OSError):
        replace_directory(staging, target)

    assert (target / "old.txt").read_text(encoding="utf-8") == "old"
