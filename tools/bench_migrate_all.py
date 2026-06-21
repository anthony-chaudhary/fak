#!/usr/bin/env python3
"""bench_migrate_all.py - migrate all benchmark artifacts to catalog structure.

This script scans and migrates benchmark results from various experiment directories
into the unified benchmark catalog structure.

Usage:
  python tools/bench_migrate_all.py --dry-run
  python tools/bench_migrate_all.py --apply
"""
import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "fak" / "experiments" / "benchmark"
RUNS_DIR = BENCHMARK_DIR / "runs" / "by-machine"
MACHINES_DIR = BENCHMARK_DIR / "machines"
CATALOG_PATH = BENCHMARK_DIR / "catalog.json"

# Default machine for benchmarks without explicit machine ID
DEFAULT_MACHINE = "anthony"

# Benchmark type mappings
BENCHMARK_TYPES = {
    "model-baseline": "model-benchmark",
    "fanout": "fan-benchmark",
    "session": "session-benchmark",
    "radixattention": "radix-benchmark",
    "gpu": "gpu-benchmark",
    "qwen36": "qwen-benchmark",
}


def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file, return None on error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
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
    # Try to extract from filename
    ts_match = re.search(r"(\d{8})[T-]?(\d{6})", path.name)
    if ts_match:
        date, time = ts_match.groups()
        return f"{date}T{time}Z"

    # Use file modification time
    try:
        ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return ts.strftime("%Y%m%dT%H%M%SZ")
    except OSError:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def infer_machine_from_path(path: Path) -> str:
    """Infer machine ID from path or filename."""
    # Check filename for machine hints
    name_lower = path.name.lower()
    if "mac" in name_lower or "mbp" in name_lower or "m3" in name_lower:
        return "node-macos-a"
    if "anthony" in name_lower or "antho" in name_lower:
        return "anthony"
    if "4070" in name_lower or "rtx" in name_lower:
        return "anthony"  # RTX 4070 is on anthony
    if "vulkan" in name_lower or "amd" in name_lower:
        return "anthony"  # AMD Vulkan tests on anthony
    return DEFAULT_MACHINE


def migrate_model_baseline(dry_run: bool) -> List[Dict]:
    """Migrate model-baseline benchmarks."""
    migrated = []
    source_dir = ROOT / "fak" / "experiments" / "model-baseline"

    if not source_dir.exists():
        return migrated

    for json_file in source_dir.glob("*.json"):
        if json_file.name == "comparison.json":
            continue  # Skip comparison files for now

        data = load_json(json_file)
        if not data:
            continue

        machine_id = infer_machine_from_path(json_file)
        ts = get_timestamp_from_path(json_file)
        model = data.get("model", "unknown").lower().replace(" ", "-")
        precision = data.get("precision", "unknown").lower()

        run_id = f"{machine_id}-{model}-{precision}-{ts[:8]}"
        run_dir = RUNS_DIR / machine_id / f"{ts}-model-baseline"

        # Create manifest
        manifest = {
            "$schema": "benchmark/run-manifest.v1",
            "run_id": run_id,
            "machine_id": machine_id,
            "timestamp": ts,
            "git": {"rev": "unknown", "branch": "master", "dirty": False},
            "harness": {"name": "fak", "version": "unknown"},
            "model": {
                "name": data.get("model", "unknown"),
                "precision": precision,
                "quantization": precision
            },
            "config": {
                "workers": data.get("workers", 1),
                "go_threads": data.get("go_threads", "")
            },
            "tags": ["model-benchmark"],
            "artifacts": {"kernel": json_file.name}
        }

        if not dry_run:
            manifest_path = run_dir / "manifest.json"
            save_json(manifest_path, manifest)
            # Copy the artifact
            dest_artifact = run_dir / json_file.name
            shutil.copy2(json_file, dest_artifact)
            print(f"  -> {json_file.name}", file=sys.stderr)
        else:
            print(f"  [DRY RUN] Would create {run_dir}", file=sys.stderr)

        migrated.append({
            "run_id": run_id,
            "machine_id": machine_id,
            "timestamp": ts,
            "path": str(run_dir.relative_to(ROOT))
        })

    return migrated


def migrate_fanout(dry_run: bool) -> List[Dict]:
    """Migrate fanout/fanbench benchmarks."""
    migrated = []
    source_dir = ROOT / "fak" / "experiments" / "fanout"

    if not source_dir.exists():
        return migrated

    for json_file in source_dir.glob("fanbench*.json"):
        data = load_json(json_file)
        if not data:
            continue

        machine_id = infer_machine_from_path(json_file)
        ts = get_timestamp_from_path(json_file)
        profile_name = data.get("profile", {}).get("name", "unknown")

        run_id = f"{machine_id}-fanbench-{profile_name}-{ts[:8]}"
        run_dir = RUNS_DIR / machine_id / f"{ts}-fanbench"

        manifest = {
            "$schema": "benchmark/run-manifest.v1",
            "run_id": run_id,
            "machine_id": machine_id,
            "timestamp": ts,
            "git": {"rev": "unknown", "branch": "master", "dirty": False},
            "harness": {"name": "fanbench", "version": data.get("app_version", "unknown")},
            "model": {"name": "synthetic", "precision": "n/a"},
            "config": {
                "goal_pool": data.get("profile", {}).get("goal_pool", 0),
                "trials": data.get("trials", 0)
            },
            "tags": ["fan-benchmark", "multi-agent"],
            "artifacts": {"fanbench": json_file.name}
        }

        if not dry_run:
            manifest_path = run_dir / "manifest.json"
            save_json(manifest_path, manifest)
            dest_artifact = run_dir / json_file.name
            shutil.copy2(json_file, dest_artifact)
            print(f"  -> {json_file.name}", file=sys.stderr)
        else:
            print(f"  [DRY RUN] Would create {run_dir}", file=sys.stderr)

        migrated.append({
            "run_id": run_id,
            "machine_id": machine_id,
            "timestamp": ts,
            "path": str(run_dir.relative_to(ROOT))
        })

    return migrated


def migrate_session(dry_run: bool) -> List[Dict]:
    """Migrate session benchmarks."""
    migrated = []
    source_dir = ROOT / "fak" / "experiments" / "session"

    if not source_dir.exists():
        return migrated

    for json_file in source_dir.glob("sessionbench*.json"):
        data = load_json(json_file)
        if not data:
            continue

        machine_id = infer_machine_from_path(json_file)
        ts = get_timestamp_from_path(json_file)

        run_id = f"{machine_id}-sessionbench-{ts[:8]}"
        run_dir = RUNS_DIR / machine_id / f"{ts}-session"

        manifest = {
            "$schema": "benchmark/run-manifest.v1",
            "run_id": run_id,
            "machine_id": machine_id,
            "timestamp": ts,
            "git": {"rev": "unknown", "branch": "master", "dirty": False},
            "harness": {"name": "sessionbench", "version": data.get("app_version", "unknown")},
            "model": {"name": "multi-agent-session", "precision": "n/a"},
            "config": {},
            "tags": ["session-benchmark", "multi-agent", "value-stack"],
            "artifacts": {"session": json_file.name}
        }

        if not dry_run:
            manifest_path = run_dir / "manifest.json"
            save_json(manifest_path, manifest)
            dest_artifact = run_dir / json_file.name
            shutil.copy2(json_file, dest_artifact)
            print(f"  -> {json_file.name}", file=sys.stderr)
        else:
            print(f"  [DRY RUN] Would create {run_dir}", file=sys.stderr)

        migrated.append({
            "run_id": run_id,
            "machine_id": machine_id,
            "timestamp": ts,
            "path": str(run_dir.relative_to(ROOT))
        })

    return migrated


def migrate_radix(dry_run: bool) -> List[Dict]:
    """Migrate radixattention benchmarks."""
    migrated = []
    source_dir = ROOT / "fak" / "experiments" / "radixattention"

    if not source_dir.exists():
        return migrated

    for json_file in source_dir.glob("radixbench*.json"):
        data = load_json(json_file)
        if not data:
            continue

        machine_id = DEFAULT_MACHINE
        ts = get_timestamp_from_path(json_file)
        model = data.get("model", "unknown").lower().replace(" ", "-")

        run_id = f"{machine_id}-radixbench-{model}-{ts[:8]}"
        run_dir = RUNS_DIR / machine_id / f"{ts}-radix"

        manifest = {
            "$schema": "benchmark/run-manifest.v1",
            "run_id": run_id,
            "machine_id": machine_id,
            "timestamp": ts,
            "git": {"rev": "unknown", "branch": "master", "dirty": False},
            "harness": {"name": "radixbench", "version": "unknown"},
            "model": {"name": data.get("model", "unknown"), "precision": "q8"},
            "config": {"quant": data.get("quant", False)},
            "tags": ["radix-benchmark", "cache-hit-rate"],
            "artifacts": {"radix": json_file.name}
        }

        if not dry_run:
            manifest_path = run_dir / "manifest.json"
            save_json(manifest_path, manifest)
            dest_artifact = run_dir / json_file.name
            shutil.copy2(json_file, dest_artifact)
            print(f"  -> {json_file.name}", file=sys.stderr)
        else:
            print(f"  [DRY RUN] Would create {run_dir}", file=sys.stderr)

        migrated.append({
            "run_id": run_id,
            "machine_id": machine_id,
            "timestamp": ts,
            "path": str(run_dir.relative_to(ROOT))
        })

    return migrated


def migrate_gpu(dry_run: bool) -> List[Dict]:
    """Migrate GPU benchmarks."""
    migrated = []
    source_dir = ROOT / "fak" / "experiments" / "gpu"

    if not source_dir.exists():
        return migrated

    for json_file in source_dir.glob("*.json"):
        if json_file.suffix != ".json" or json_file.name in ["comparison.json", "README.md"]:
            continue

        data = load_json(json_file)
        if not data:
            continue

        machine_id = "anthony"  # GPU tests are on anthony
        ts = get_timestamp_from_path(json_file)
        model = data.get("model", "unknown").lower().replace(" ", "-")

        run_id = f"{machine_id}-gpu-{model}-{ts[:8]}"
        run_dir = RUNS_DIR / machine_id / f"{ts}-gpu"

        manifest = {
            "$schema": "benchmark/run-manifest.v1",
            "run_id": run_id,
            "machine_id": machine_id,
            "timestamp": ts,
            "git": {"rev": "unknown", "branch": "master", "dirty": False},
            "harness": {"name": "fak-gpu-bench", "version": "unknown"},
            "model": {"name": data.get("model", "unknown"), "precision": data.get("precision", "unknown")},
            "config": {"backend": data.get("backend", {}).get("selected", "unknown")},
            "tags": ["gpu-benchmark", "cuda"],
            "artifacts": {"gpu": json_file.name}
        }

        if not dry_run:
            manifest_path = run_dir / "manifest.json"
            save_json(manifest_path, manifest)
            dest_artifact = run_dir / json_file.name
            shutil.copy2(json_file, dest_artifact)
            print(f"  -> {json_file.name}", file=sys.stderr)
        else:
            print(f"  [DRY RUN] Would create {run_dir}", file=sys.stderr)

        migrated.append({
            "run_id": run_id,
            "machine_id": machine_id,
            "timestamp": ts,
            "path": str(run_dir.relative_to(ROOT))
        })

    return migrated


def migrate_qwen(dry_run: bool) -> List[Dict]:
    """Migrate Qwen benchmarks."""
    migrated = []
    source_dir = ROOT / "fak" / "experiments" / "qwen36"

    if not source_dir.exists():
        return migrated

    # Migrate benchmark-style JSON files
    patterns = ["*.json"]
    exclude = ["README.md", ".md"]

    for json_file in source_dir.glob("*bench*.json"):
        data = load_json(json_file)
        if not data:
            continue

        machine_id = infer_machine_from_path(json_file)
        ts = get_timestamp_from_path(json_file)

        # Infer benchmark type from filename
        benchmark_type = "qwen-benchmark"
        if "gateway" in json_file.name.lower():
            benchmark_type = "gateway-benchmark"
        elif "llamacpp" in json_file.name.lower():
            benchmark_type = "llamacpp-benchmark"

        run_id = f"{machine_id}-qwen-{benchmark_type}-{ts[:8]}"
        run_dir = RUNS_DIR / machine_id / f"{ts}-qwen"

        manifest = {
            "$schema": "benchmark/run-manifest.v1",
            "run_id": run_id,
            "machine_id": machine_id,
            "timestamp": ts,
            "git": {"rev": "unknown", "branch": "master", "dirty": False},
            "harness": {"name": benchmark_type, "version": "unknown"},
            "model": {"name": "qwen3.6-27b", "precision": "q4_k_m"},
            "config": {},
            "tags": ["qwen-benchmark", benchmark_type],
            "artifacts": {"qwen": json_file.name}
        }

        if not dry_run:
            manifest_path = run_dir / "manifest.json"
            save_json(manifest_path, manifest)
            dest_artifact = run_dir / json_file.name
            shutil.copy2(json_file, dest_artifact)
            print(f"  -> {json_file.name}", file=sys.stderr)
        else:
            print(f"  [DRY RUN] Would create {run_dir}", file=sys.stderr)

        migrated.append({
            "run_id": run_id,
            "machine_id": machine_id,
            "timestamp": ts,
            "path": str(run_dir.relative_to(ROOT))
        })

    return migrated


def rebuild_catalog() -> bool:
    """Rebuild the catalog from scratch."""
    # Import catalog builder
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from bench_catalog import build_catalog, save_json, CATALOG_PATH
        catalog = build_catalog()
        return save_json(CATALOG_PATH, catalog)
    except ImportError:
        print("[ERROR] Failed to import bench_catalog", file=sys.stderr)
        return False


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Preview changes")
    ap.add_argument("--apply", action="store_true", help="Apply migration")
    args = ap.parse_args(argv)

    if not args.apply and not args.dry_run:
        print("[ERROR] Specify --apply or --dry-run", file=sys.stderr)
        return 1

    all_runs = []

    print("[migrate_all] Migrating all benchmark artifacts...", file=sys.stderr)

    # Migrate each benchmark type
    print("[migrate_all] model-baseline...", file=sys.stderr)
    all_runs.extend(migrate_model_baseline(args.dry_run))

    print("[migrate_all] fanout...", file=sys.stderr)
    all_runs.extend(migrate_fanout(args.dry_run))

    print("[migrate_all] session...", file=sys.stderr)
    all_runs.extend(migrate_session(args.dry_run))

    print("[migrate_all] radixattention...", file=sys.stderr)
    all_runs.extend(migrate_radix(args.dry_run))

    print("[migrate_all] gpu...", file=sys.stderr)
    all_runs.extend(migrate_gpu(args.dry_run))

    print("[migrate_all] qwen36...", file=sys.stderr)
    all_runs.extend(migrate_qwen(args.dry_run))

    print(f"[migrate_all] Total runs migrated: {len(all_runs)}", file=sys.stderr)

    if not args.dry_run:
        print("[migrate_all] Rebuilding catalog...", file=sys.stderr)
        if rebuild_catalog():
            print(f"[migrate_all] Catalog updated", file=sys.stderr)
        else:
            print("[migrate_all] Failed to rebuild catalog", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
