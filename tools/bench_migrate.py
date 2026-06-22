#!/usr/bin/env python3
"""bench_migrate.py -- migrate existing benchmark data to new structure.

This script converts data from:
- fak/experiments/fleet-nodes/ → experiments/benchmark/runs/by-machine/
<!-- fak/experiments/dgx/ indexing excluded from the public copy (operator-private lab infra). -->

Usage:
  python tools/bench_migrate.py --dry-run                    # Preview changes
  python tools/bench_migrate.py --apply                     # Do the migration
  python tools/bench_migrate.py --apply --catalog-only       # Only update catalog
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "fak" / "experiments" / "benchmark"
MACHINES_DIR = BENCHMARK_DIR / "machines"
RUNS_DIR = BENCHMARK_DIR / "runs" / "by-machine"

# Legacy paths
FLEET_NODES_DIR = ROOT / "fak" / "experiments" / "fleet-nodes"
# NOTE: fak/experiments/dgx/ is operator-private lab infra, excluded from the
# public copy (see PUBLIC-SCRUB-POLICY.md). Its index step is omitted here.


def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file, return None on error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def config_hash(config: Dict) -> str:
    """Generate 8-char hash from config dict."""
    s = json.dumps(config, sort_keys=True)
    return sha256(s.encode()).hexdigest()[:8]


def timestamp_from_path(path: Path) -> Optional[str]:
    """Extract timestamp from a path if possible."""
    # Try to get file modification time
    try:
        ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return ts.strftime("%Y%m%dT%H%M%SZ")
    except OSError:
        return None


def migrate_fleet_nodes(dry_run: bool) -> List[Dict]:
    """Migrate fleet-nodes to new structure."""
    migrated = []

    if not FLEET_NODES_DIR.exists():
        print("[migrate] No fleet-nodes directory found", file=sys.stderr)
        return migrated

    for node_dir in FLEET_NODES_DIR.iterdir():
        if not node_dir.is_dir():
            continue

        machine_id = node_dir.name
        print(f"[migrate] Processing node: {machine_id}", file=sys.stderr)

        # Migrate specs
        specs_path = node_dir / "node-info.json"
        if specs_path.exists():
            specs = load_json(specs_path)
            if specs:
                new_specs = {
                    "$schema": "benchmark/machine-specs.v1",
                    "machine_id": machine_id,
                    "hostname": specs.get("hostname") or specs.get("host", machine_id),
                    "registered_at": datetime.now(timezone.utc).isoformat(),
                    "hardware": {
                        "cpu": {
                            "model": specs.get("cpu", "?"),
                            "cores_physical": specs.get("cores", specs.get("cpu_cores", 1)),
                            "cores_logical": specs.get("cores", specs.get("cpu_cores", 1)),
                            "architecture": specs.get("arch", "?")
                        },
                        "gpu": [],
                        "ram_gb": 0
                    },
                    "os": {
                        "name": specs.get("os", "?"),
                        "version": "",
                        "kernel": ""
                    },
                    "runtime": {
                        "go_version": specs.get("go", "?"),
                        "python_version": ""
                    },
                    "tags": [specs.get("arch", ""), specs.get("os", "")]
                }

                dest_specs = MACHINES_DIR / machine_id / "specs.json"
                if not dry_run:
                    dest_specs.parent.mkdir(parents=True, exist_ok=True)
                    with open(dest_specs, "w", encoding="utf-8") as f:
                        json.dump(new_specs, f, indent=2)
                    print(f"  -> Wrote {dest_specs}", file=sys.stderr)
                else:
                    print(f"  [DRY RUN] Would write {dest_specs}", file=sys.stderr)

        # Create manifest for the run
        manifest_path = node_dir / "production-readiness-manifest.json"
        if manifest_path.exists():
            manifest = load_json(manifest_path)
            if not manifest:
                manifest = {}

            ts = timestamp_from_path(node_dir) or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"{machine_id}-smollm2-135m-q8-batch-{ts}"

            # Infer model from path
            model = "SmolLM2-135M-Instruct"

            new_manifest = {
                "$schema": "benchmark/run-manifest.v1",
                "run_id": run_id,
                "machine_id": machine_id,
                "timestamp": ts,
                "git": {
                    "rev": manifest.get("git", specs.get("git", "unknown")),
                    "branch": "master",
                    "dirty": False
                },
                "harness": {
                    "name": "fak",
                    "version": "0.19.0"
                },
                "model": {
                    "name": model,
                    "family": "smollm2",
                    "parameters": "135M",
                    "source": "HuggingFaceTB/SmolLM2-135M-Instruct",
                    "precision": "q8",
                    "quantization": "Q8_0"
                },
                "config": {
                    "batch_sizes": [1, 4, 8, 16, 32, 64, 128, 256, 512],
                    "workers": 16,
                    "decode_steps": int(manifest.get("decode_steps", 32)),
                    "prefill_sizes": [16, 64, 256],
                    "workload_file": manifest.get("workload", ""),
                    "workload_prefill_cap": int(manifest.get("workload_prefill_cap", 0)),
                    "workload_prompt_cap": int(manifest.get("workload_prompt_cap", 0))
                },
                "tags": ["kernel", "batch", "phase0"],
                "artifacts": {
                    "kernel": "kernel.json",
                    "batch": "batch.json",
                    "fleetbench": "fleetbench.json"
                }
            }

            run_dir = RUNS_DIR / machine_id / f"{ts}-phase0"
            dest_manifest = run_dir / "manifest.json"

            if not dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                with open(dest_manifest, "w", encoding="utf-8") as f:
                    json.dump(new_manifest, f, indent=2)
                print(f"  -> Wrote {dest_manifest}", file=sys.stderr)

                # Copy existing artifacts
                for name in ["batchbench-q8.json", "fleetbench.json", "modelprof.json"]:
                    src = node_dir / name
                    if src.exists():
                        dest = run_dir / name.replace("batchbench-q8", "batch").replace("modelprof", "kernel")
                        shutil.copy2(src, dest)
                        print(f"  -> Copied {name}", file=sys.stderr)

                # Copy kernel data from q8kernel.txt
                kernel_txt = node_dir / "q8kernel.txt"
                kernel_json = node_dir / "modelbench-q8.json"
                if kernel_txt.exists() and kernel_json.exists():
                    # Convert to kernel.json format
                    kernel_data = load_json(kernel_json)
                    if kernel_data:
                        dest_kernel = run_dir / "kernel.json"
                        with open(dest_kernel, "w", encoding="utf-8") as f:
                            json.dump(kernel_data, f, indent=2)
                        print("  -> Converted q8kernel.txt + modelbench-q8.json to kernel.json", file=sys.stderr)
            else:
                print(f"  [DRY RUN] Would create {dest_manifest}", file=sys.stderr)

            migrated.append({
                "run_id": run_id,
                "machine_id": machine_id,
                "timestamp": ts,
                "path": str(run_dir.relative_to(ROOT))
            })

    return migrated


# index_dgx_runs() omitted from the public copy -- it indexed the operator's
# private GPU server dir (fak/experiments/dgx/, excluded from export). The public
# copy has no such dir; see PUBLIC-SCRUB-POLICY.md.


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                   help="Preview changes without applying")
    ap.add_argument("--apply", action="store_true",
                   help="Apply the migration")
    ap.add_argument("--catalog-only", action="store_true",
                   help="Only update catalog, don't move files")
    args = ap.parse_args(argv)

    if not args.apply and not args.dry_run:
        print("[ERROR] Specify --apply or --dry-run", file=sys.stderr)
        return 1

    print(f"[migrate] Starting migration (dry_run={args.dry_run})", file=sys.stderr)
    print()

    all_runs = []

    # Migrate fleet nodes
    if not args.catalog_only:
        fleet_runs = migrate_fleet_nodes(args.dry_run)
        all_runs.extend(fleet_runs)
        print(f"[migrate] Migrated {len(fleet_runs)} fleet-node runs", file=sys.stderr)
        print()

    # DGX run indexing omitted (operator-private lab infra; see module note).

    # Build catalog
    print("[migrate] Building catalog...", file=sys.stderr)
    # Import catalog builder (simplified here)
    if not args.dry_run:
        # Trigger catalog rebuild
        from bench_catalog import build_catalog, save_json, CATALOG_PATH
        catalog = build_catalog()
        save_json(CATALOG_PATH, catalog)
        print(f"[migrate] Catalog saved to {CATALOG_PATH.relative_to(ROOT)}", file=sys.stderr)

    print()
    print(f"[migrate] Total runs processed: {len(all_runs)}", file=sys.stderr)
    print("[migrate] Migration complete. Run 'python tools/bench_cli.py list' to verify.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
