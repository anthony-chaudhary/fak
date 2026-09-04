#!/usr/bin/env python3
"""Standalone installer: venv + editable install + symlinks into ~/.local/bin."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"
ENTRY_POINTS = ("l3-bench", "cama-bench")


def main():
    parser = argparse.ArgumentParser(description="Install l3-client into a local venv.")
    parser.add_argument("--fresh", action="store_true", help="Remove existing .venv and recreate")
    parser.add_argument("--no-symlink", action="store_true", help="Skip symlink creation")
    parser.add_argument("--bin-dir", type=Path, default=Path.home() / ".local" / "bin",
                        help="Directory for symlinks (default: ~/.local/bin)")
    parser.add_argument("--extras", default="rdma,dev", help="Extras spec for pip install (default: rdma,dev)")
    args = parser.parse_args()

    # --- Check Python version ---
    if sys.version_info < (3, 10):
        print(f"Error: Python >= 3.10 required, got {sys.version}", file=sys.stderr)
        sys.exit(1)

    if sys.platform != "linux":
        print(f"Warning: this installer targets Linux; current platform is {sys.platform}")

    # --- Create / recreate venv ---
    if args.fresh and VENV_DIR.exists():
        print(f"Removing existing venv: {VENV_DIR}")
        shutil.rmtree(VENV_DIR)

    if VENV_DIR.exists():
        print(f"Venv already exists: {VENV_DIR}")
    else:
        print(f"Creating venv: {VENV_DIR}")
        _run([sys.executable, "-m", "venv", str(VENV_DIR)])

    venv_python = VENV_DIR / "bin" / "python"
    if not venv_python.exists():
        print(f"Error: venv python not found at {venv_python}", file=sys.stderr)
        sys.exit(1)

    # --- Upgrade pip + ensure build deps ---
    # pip 26+ no longer bundles setuptools; install it explicitly so that
    # --no-build-isolation (needed for the RDMA C++ extension) can find the
    # build backend declared in pyproject.toml.
    print("Upgrading pip and installing build dependencies...")
    _run([str(venv_python), "-m", "pip", "install", "--upgrade",
          "pip", "setuptools>=68.0", "wheel"])

    # --- Install package ---
    extras = args.extras
    print(f"Installing package (extras: {extras})...")
    # --no-build-isolation ensures the RDMA C++ extension rebuilds correctly
    _run([str(venv_python), "-m", "pip", "install", "--no-build-isolation",
          "-e", f".[{extras}]"], cwd=PROJECT_DIR)

    # --- Symlinks ---
    if args.no_symlink:
        print("Skipping symlink creation (--no-symlink)")
    else:
        _create_symlinks(args.bin_dir)

    # --- Summary ---
    print("\n--- Installation complete ---")
    print(f"  venv:    {VENV_DIR}")
    if not args.no_symlink:
        for name in ENTRY_POINTS:
            print(f"  {name:12s} -> {args.bin_dir / name}")
    print(f"\nTo activate the venv directly:  source {VENV_DIR}/bin/activate")


def _create_symlinks(bin_dir: Path):
    bin_dir.mkdir(parents=True, exist_ok=True)

    for name in ENTRY_POINTS:
        source = VENV_DIR / "bin" / name
        target = bin_dir / name

        if not source.exists():
            print(f"Warning: entry point {source} not found, skipping symlink for '{name}'")
            continue

        if target.is_symlink():
            existing = target.resolve()
            if existing == source.resolve():
                print(f"Symlink already correct: {target} -> {source}")
                continue
            print(f"Updating symlink: {target} (was -> {existing})")
            target.unlink()
        elif target.exists():
            print(f"Warning: {target} exists and is not a symlink — skipping (remove it manually to fix)")
            continue

        os.symlink(source.resolve(), target)
        print(f"Created symlink: {target} -> {source}")

    # PATH check — auto-add to shell profile if missing
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    bin_str = str(bin_dir.resolve())
    on_path = str(bin_dir) in path_dirs or bin_str in path_dirs
    if not on_path:
        _ensure_path_in_profile(bin_dir)


def _ensure_path_in_profile(bin_dir: Path):
    """Append PATH export to all relevant shell profiles."""
    bin_resolved = str(bin_dir.resolve())
    export_line = f'export PATH="{bin_resolved}:$PATH"'
    marker = "# Added by l3-client installer"
    block = f"\n{marker}\n{export_line}\n"

    home = Path.home()
    candidates = [home / ".profile", home / ".bash_profile", home / ".bashrc"]

    updated = []
    for rc in candidates:
        if rc.exists():
            contents = rc.read_text()
            if bin_resolved in contents:
                print(f"  PATH already in {rc}")
                continue
            with open(rc, "a") as f:
                f.write(block)
            updated.append(rc)

    # If no profile files existed at all, create ~/.profile
    if not updated and not any(rc.exists() for rc in candidates):
        rc = home / ".profile"
        with open(rc, "a") as f:
            f.write(block)
        updated.append(rc)

    if updated:
        names = ", ".join(str(p) for p in updated)
        print(f"Added {bin_resolved} to PATH in {names}")
        print("  Changes take effect on next login / new shell.")
    else:
        print(f"  PATH already configured in all profile files.")


def _run(cmd, **kwargs):
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"Error: command failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
