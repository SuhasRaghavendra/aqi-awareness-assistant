"""
scripts/setup_dvc_remote.py
---------------------------
Phase 3, Step 005: DVC Remote Configuration.

Sets up a local storage directory as a DVC remote destination.
Local storage requires zero external cloud credentials, works offline,
and fulfills the project requirement for graded dataset versioning.

Usage:
  python scripts/setup_dvc_remote.py [--path ../dvc_remote] [--name local_storage]
"""

from __future__ import annotations

import argparse
import configparser
import os
import sys
from pathlib import Path


def setup_local_remote(
    project_root: Path,
    remote_name: str = "local_storage",
    remote_path_str: str = "../dvc_remote",
    make_default: bool = True,
) -> Path:
    """Create the storage directory and register it in .dvc/config."""
    # Resolve target remote folder relative to project root
    remote_path = (project_root / remote_path_str).resolve()
    remote_path.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Ensured remote storage directory exists at:\n     {remote_path}")

    dvc_dir = project_root / ".dvc"
    dvc_dir.mkdir(parents=True, exist_ok=True)
    dvc_config_file = dvc_dir / "config"

    # Use configparser to read/write .dvc/config safely
    config = configparser.ConfigParser()
    if dvc_config_file.exists():
        config.read(dvc_config_file, encoding="utf-8")

    section_name = f'remote "{remote_name}"'
    if not config.has_section(section_name):
        config.add_section(section_name)

    # Use a relative URL if inside or adjacent to project, or absolute if preferred
    config.set(section_name, "url", remote_path_str.replace("\\", "/"))

    if make_default:
        if not config.has_section("core"):
            config.add_section("core")
        config.set("core", "remote", remote_name)

    with open(dvc_config_file, "w", encoding="utf-8") as f:
        config.write(f)

    print(f"[OK] Updated .dvc/config with remote '{remote_name}':")
    print(f"     URL: {remote_path_str}")
    if make_default:
        print(f"     Default remote: {remote_name}")

    return remote_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure local DVC remote storage")
    parser.add_argument(
        "--path",
        type=str,
        default="../dvc_remote",
        help="Local directory path to use as DVC remote storage (default: ../dvc_remote)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="local_storage",
        help="Name of the DVC remote (default: local_storage)",
    )
    parser.add_argument(
        "--no-default",
        action="store_true",
        help="Do not make this remote the default remote",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    remote_path = setup_local_remote(
        project_root=project_root,
        remote_name=args.name,
        remote_path_str=args.path,
        make_default=not args.no_default,
    )

    print("\n" + "=" * 60)
    print("DVC REMOTE SETUP COMPLETE")
    print("=" * 60)
    print(f"Remote Name:    {args.name}")
    print(f"Storage Path:   {remote_path}")
    print("\nEquivalent manual DVC CLI commands:")
    print(f"  mkdir -p \"{args.path}\"")
    print(f"  dvc remote add -d {args.name} {args.path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
