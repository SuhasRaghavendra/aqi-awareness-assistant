"""Atomic, LF-stable artifact writes.

Every artifact that is hashed, published or consumed at serving time is written
through these helpers. Two properties matter:

1. **Atomicity.** The bytes are written to a temporary file in the destination
   directory, flushed to disk, and moved into place with ``os.replace``. A
   crash or a full disk therefore leaves either the previous good artifact or
   the complete new one, never a truncated file.
2. **Byte stability.** Text artifacts are encoded as UTF-8 with LF line endings
   on every platform, so the SHA-256 digests recorded in ``MANIFEST.json`` and
   ``MANIFEST.sha256`` describe the same bytes on Windows, Linux and CI.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

__all__ = [
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_joblib",
    "atomic_write_parquet",
    "atomic_write_text",
    "replace_directory",
]


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync of a directory so a rename survives power loss."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _new_temp_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    return Path(name)


def _commit(temp: Path, path: Path) -> None:
    os.replace(temp, path)
    _fsync_directory(path.parent)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write *data* to *path* atomically."""
    path = Path(path)
    temp = _new_temp_file(path)
    try:
        with temp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        _commit(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | Path, text: str, *, newline: str = "\n") -> None:
    """Write *text* atomically as UTF-8 with explicit line endings."""
    if newline != "\n":
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(
    path: str | Path, payload: Any, *, indent: int = 2, default: Any = None
) -> None:
    """Serialise *payload* exactly as ``json.dumps(indent=...)`` and write it atomically."""
    atomic_write_bytes(path, json.dumps(payload, indent=indent, default=default).encode("utf-8"))


def atomic_write_parquet(frame: Any, path: str | Path, *, index: bool = False) -> None:
    """Write a pandas frame to parquet atomically."""
    path = Path(path)
    temp = _new_temp_file(path)
    try:
        frame.to_parquet(temp, index=index)
        # "rb+" (not "rb"): Windows cannot fsync a read-only handle.
        with temp.open("rb+") as fh:
            os.fsync(fh.fileno())
        _commit(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def atomic_write_joblib(payload: Any, path: str | Path) -> None:
    """Pickle *payload* with joblib and move it into place atomically."""
    import joblib

    path = Path(path)
    temp = _new_temp_file(path)
    try:
        joblib.dump(payload, temp)
        with temp.open("rb+") as fh:
            os.fsync(fh.fileno())
        _commit(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def replace_directory(staging: str | Path, target: str | Path) -> None:
    """Swap *staging* into *target*, keeping the old copy until the swap succeeds.

    The previous directory is renamed to ``<target>.bak`` first and only deleted
    once the new directory is in place, so an interrupted publish cannot leave
    the project without a champion bundle.
    """
    staging = Path(staging)
    target = Path(target)
    if not staging.is_dir():
        raise FileNotFoundError(f"Staging directory is missing: {staging}")
    backup = target.with_name(f"{target.name}.bak")
    if backup.exists():
        shutil.rmtree(backup)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        os.replace(target, backup)
    try:
        os.replace(staging, target)
    except BaseException:
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    _fsync_directory(target.parent)
