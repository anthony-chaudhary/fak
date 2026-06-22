#!/usr/bin/env python3
"""bench_migrate_comprehensive.py - migrate all experiment artifacts.

This script comprehensively migrates benchmark and experiment artifacts
from across the experiments/ directory into the catalog structure.

Usage:
  python tools/bench_migrate_comprehensive.py --dry-run
  python tools/bench_migrate_comprehensive.py --apply
"""
import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "fak" / "experiments" / "benchmark"
RUNS_DIR = BENCHMARK_DIR / "runs" / "by-machine"
CATALOG_PATH = BENCHMARK_DIR / "catalog.json"
EXPERIMENTS_DIR = ROOT / "fak" / "experiments"

# Default machine
DEFAULT_MACHINE = "anthony"

# File count for progress reporting
TOTAL_MIGRATED = 0


def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file, return None on error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_json(path: Path, data: Any) -> bool:
    """Save JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        tmp.replace(path)
        return True
    except OSError as e:
        print(f"[ERROR] Failed to save {path}: {e}", file=sys.stderr)
        return False


def get_timestamp_from_path(path: Path) -> str:
    """Extract or generate timestamp from path/file."""
    ts_match = re.search(r"(\d{8})[T-]?(\d{6})", path.name)
    if ts_match:
        date, time = ts_match.groups()
        return f"{date}T{time}Z"

    try:
        ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return ts.strftime("%Y%m%dT%H%M%SZ")
    except OSError:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def infer_machine_from_path(path: Path) -> str:
    """Infer machine ID from path or filename."""
    name_lower = path.name.lower()
    parent_lower = path.parent.name.lower()

    if "mac" in name_lower or "mbp" in name_lower or "m3" in name_lower or "node-macos-a" in parent_lower:
        return "node-macos-a"
    if "anthony" in name_lower or "antho" in name_lower:
        return "anthony"
    if "4070" in name_lower or "rtx" in name_lower:
        return "anthony"
    if "vulkan" in name_lower or "amd" in name_lower:
        return "anthony"

    # Check parent directory
    if "mac" in parent_lower or "mbp" in parent_lower:
        return "node-macos-a"

    return DEFAULT_MACHINE


def migrate_qwen36(dry_run: bool) -> int:
    """Migrate all Qwen36 experiment files."""
    global TOTAL_MIGRATED
    migrated = 0
    source_dir = EXPERIMENTS_DIR / "qwen36"

    if not source_dir.exists():
        return 0

    print("[migrate] Processing qwen36/...", file=sys.stderr)

    for json_file in source_dir.glob("*.json"):
        if json_file.name.startswith("README") or json_file.name.endswith(".md"):
            continue

        machine_id = infer_machine_from_path(json_file)
        ts = get_timestamp_from_path(json_file)

        # Create a simple run entry
        run_id = f"{machine_id}-qwen36-{json_file.stem[:20]}-{ts[:8]}"
        run_dir = RUNS_DIR / machine_id / f"{ts}-qwen36"

        manifest = {
            "$schema": "benchmark/run-manifest.v1",
            "run_id": run_id,
            "machine_id": machine_id,
            "timestamp": ts,
            "git": {"rev": "unknown", "branch": "master", "dirty": False},
            "harness": {"name": "qwen36-experiment", "version": "unknown"},
            "model": {"name": "qwen3.6-27b", "precision": "unknown"},
            "config": {},
            "tags": ["qwen36", "experiment"],
            "artifacts": {"qwen36": json_file.name}
        }

        if not dry_run:
            manifest_path = run_dir / "manifest.json"
            save_json(manifest_path, manifest)
            dest_artifact = run_dir / json_file.name
            shutil.copy2(json_file, dest_artifact)
            migrated += 1
        else:
            migrated += 1

    print(f"  -> {migrated} files", file=sys.stderr)
    TOTAL_MIGRATED += migrated
    return migrated


def migrate_fleet(dry_run: bool) -> int:
    """Migrate fleet experiment files."""
    global TOTAL_MIGRATED
    migrated = 0
    source_dir = EXPERIMENTS_DIR / "fleet"

    if not source_dir.exists():
        return 0

    print("[migrate] Processing fleet/...", file=sys.stderr)

    for file_path in list(source_dir.glob("*.json"))[:50]:  # Limit to 50 for now
        machine_id = DEFAULT_MACHINE
        ts = get_timestamp_from_path(file_path)

        run_id = f"{machine_id}-fleet-{file_path.stem[:30]}-{ts[:8]}"
        run_dir = RUNS_DIR / machine_id / f"{ts}-fleet"

        manifest = {
            "$schema": "benchmark/run-manifest.v1",
            "run_id": run_id,
            "machine_id": machine_id,
            "timestamp": ts,
            "git": {"rev": "unknown", "branch": "master", "dirty": False},
            "harness": {"name": "fleet-experiment", "version": "unknown"},
            "model": {"name": "synthetic", "precision": "n/a"},
            "config": {},
            "tags": ["fleet", "experiment"],
            "artifacts": {"fleet": file_path.name}
        }

        if not dry_run:
            manifest_path = run_dir / "manifest.json"
            save_json(manifest_path, manifest)
            dest_artifact = run_dir / file_path.name
            shutil.copy2(file_path, dest_artifact)
            migrated += 1
        else:
            migrated += 1

    # Also migrate CSV files
    for file_path in list(source_dir.glob("*.csv"))[:50]:
        machine_id = DEFAULT_MACHINE
        ts = get_timestamp_from_path(file_path)

        run_id = f"{machine_id}-fleet-csv-{file_path.stem[:20]}-{ts[:8]}"
        run_dir = RUNS_DIR / machine_id / f"{ts}-fleet"

        manifest_path = run_dir / "manifest.json"
        if not dry_run and not manifest_path.exists():  # Only create if not already created
            manifest = {
                "$schema": "benchmark/run-manifest.v1",
                "run_id": run_id,
                "machine_id": machine_id,
                "timestamp": ts,
                "git": {"rev": "unknown", "branch": "master", "dirty": False},
                "harness": {"name": "fleet-csv", "version": "unknown"},
                "model": {"name": "synthetic", "precision": "n/a"},
                "config": {},
                "tags": ["fleet", "csv"],
                "artifacts": {"csv": file_path.name}
            }
            save_json(manifest_path, manifest)

        dest_artifact = run_dir / file_path.name
        if not dry_run:
            shutil.copy2(file_path, dest_artifact)
            migrated += 1
        else:
            migrated += 1

    print(f"  -> {migrated} files", file=sys.stderr)
    TOTAL_MIGRATED += migrated
    return migrated


def migrate_agent_live(dry_run: bool) -> int:
    """Migrate agent-live experiment files."""
    global TOTAL_MIGRATED
    migrated = 0
    source_dir = EXPERIMENTS_DIR / "agent-live"

    if not source_dir.exists():
        return 0

    print("[migrate] Processing agent-live/...", file=sys.stderr)

    for file_path in source_dir.glob("*.json"):
        if "README" in file_path.name:
            continue

        machine_id = infer_machine_from_path(file_path)
        ts = get_timestamp_from_path(file_path)

        run_id = f"{machine_id}-agent-live-{file_path.stem[:20]}-{ts[:8]}"
        run_dir = RUNS_DIR / machine_id / f"{ts}-agent-live"

        manifest = {
            "$schema": "benchmark/run-manifest.v1",
            "run_id": run_id,
            "machine_id": machine_id,
            "timestamp": ts,
            "git": {"rev": "unknown", "branch": "master", "dirty": False},
            "harness": {"name": "agent-live", "version": "unknown"},
            "model": {"name": "agent-workload", "precision": "n/a"},
            "config": {},
            "tags": ["agent-live", "experiment"],
            "artifacts": {"agent": file_path.name}
        }

        if not dry_run:
            manifest_path = run_dir / "manifest.json"
            save_json(manifest_path, manifest)
            dest_artifact = run_dir / file_path.name
            shutil.copy2(file_path, dest_artifact)
            migrated += 1
        else:
            migrated += 1

    print(f"  -> {migrated} files", file=sys.stderr)
    TOTAL_MIGRATED += migrated
    return migrated


def migrate_other_experiments(dry_run: bool) -> int:
    """Migrate files from other experiment directories."""
    global TOTAL_MIGRATED
    migrated = 0

    other_dirs = [
        "parity", "api-host-bridge", "subsystem-checks",
        "turn-tax", "recall", "contextq", "value-sweep",
        "safetensors-load-rss", "permission-systems", "cdb"
    ]

    for dir_name in other_dirs:
        source_dir = EXPERIMENTS_DIR / dir_name
        if not source_dir.exists():
            continue

        print(f"[migrate] Processing {dir_name}/...", file=sys.stderr)

        for file_path in source_dir.glob("*.json"):
            machine_id = infer_machine_from_path(file_path)
            ts = get_timestamp_from_path(file_path)

            run_id = f"{machine_id}-{dir_name}-{file_path.stem[:15]}-{ts[:8]}"
            run_dir = RUNS_DIR / machine_id / f"{ts}-{dir_name}"

            manifest = {
                "$schema": "benchmark/run-manifest.v1",
                "run_id": run_id,
                "machine_id": machine_id,
                "timestamp": ts,
                "git": {"rev": "unknown", "branch": "master", "dirty": False},
                "harness": {"name": dir_name, "version": "unknown"},
                "model": {"name": "experiment", "precision": "n/a"},
                "config": {},
                "tags": [dir_name, "experiment"],
                "artifacts": {dir_name: file_path.name}
            }

            if not dry_run:
                manifest_path = run_dir / "manifest.json"
                save_json(manifest_path, manifest)
                dest_artifact = run_dir / file_path.name
                shutil.copy2(file_path, dest_artifact)
                migrated += 1
            else:
                migrated += 1

        print(f"  -> {migrated} files from {dir_name}", file=sys.stderr)

    TOTAL_MIGRATED += migrated
    return migrated


def migrate_csv_files(dry_run: bool) -> int:
    """Migrate CSV files from benchmark directories."""
    global TOTAL_MIGRATED
    migrated = 0

    # CSV files in fanout and other directories
    csv_dirs = ["fanout", "fleet-nodes", "value-sweep"]

    for dir_name in csv_dirs:
        source_dir = EXPERIMENTS_DIR / dir_name
        if not source_dir.exists():
            continue

        print(f"[migrate] Processing CSV files from {dir_name}/...", file=sys.stderr)

        for file_path in source_dir.glob("*.csv"):
            machine_id = infer_machine_from_path(file_path)
            ts = get_timestamp_from_path(file_path)

            run_id = f"{machine_id}-{dir_name}-csv-{file_path.stem[:15]}-{ts[:8]}"
            run_dir = RUNS_DIR / machine_id / f"{ts}-{dir_name}-csv"

            manifest_path = run_dir / "manifest.json"
            if not dry_run and not manifest_path.exists():
                manifest = {
                    "$schema": "benchmark/run-manifest.v1",
                    "run_id": run_id,
                    "machine_id": machine_id,
                    "timestamp": ts,
                    "git": {"rev": "unknown", "branch": "master", "dirty": False},
                    "harness": {"name": f"{dir_name}-csv", "version": "unknown"},
                    "model": {"name": "csv-data", "precision": "n/a"},
                    "config": {},
                    "tags": [dir_name, "csv"],
                    "artifacts": {"csv": file_path.name}
                }
                save_json(manifest_path, manifest)

            dest_artifact = run_dir / file_path.name
            if not dry_run:
                shutil.copy2(file_path, dest_artifact)
                migrated += 1
            else:
                migrated += 1

        print(f"  -> {migrated} CSV files from {dir_name}", file=sys.stderr)

    TOTAL_MIGRATED += migrated
    return migrated


def rebuild_catalog() -> bool:
    """Rebuild the catalog from scratch."""
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from bench_catalog import build_catalog, save_json, CATALOG_PATH
        catalog = build_catalog()
        return save_json(CATALOG_PATH, catalog)
    except ImportError:
        return False


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Preview changes")
    ap.add_argument("--apply", action="store_true", help="Apply migration")
    args = ap.parse_args(argv)

    if not args.apply and not args.dry_run:
        print("[ERROR] Specify --apply or --dry-run", file=sys.stderr)
        return 1

    print("[migrate] Comprehensive migration of experiment artifacts...", file=sys.stderr)
    print()

    migrated = 0
    migrated += migrate_qwen36(args.dry_run)
    migrated += migrate_fleet(args.dry_run)
    migrated += migrate_agent_live(args.dry_run)
    migrated += migrate_other_experiments(args.dry_run)
    migrated += migrate_csv_files(args.dry_run)

    print()
    print(f"[migrate] Total files migrated: {TOTAL_MIGRATED}", file=sys.stderr)
    print(f"[migrate] Total runs created: {migrated}", file=sys.stderr)

    if not args.dry_run:
        print("[migrate] Rebuilding catalog...", file=sys.stderr)
        if rebuild_catalog():
            # Count final runs
            catalog = load_json(CATALOG_PATH)
            if catalog:
                run_count = len(catalog.get("runs", []))
                print(f"[migrate] Final catalog has {run_count} runs", file=sys.stderr)
        else:
            print("[migrate] Failed to rebuild catalog", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
