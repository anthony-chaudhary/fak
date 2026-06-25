#!/usr/bin/env python3
"""bench_catalog.py -- maintain the benchmark catalog and index.

The catalog is the single source of truth for:
- All registered machines
- All benchmark runs
- Cross-reference indexes (by model, precision, date)

Usage:
  python tools/bench_catalog.py build                    # Rebuild catalog from scratch
  python tools/bench_catalog.py update                    # Incremental update
  python tools/bench_catalog.py validate                  # Validate catalog integrity
  python tools/bench_catalog.py add-machine <specs.json>  # Register a new machine
  python tools/bench_catalog.py add-run <manifest.json>   # Register a new run
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# The shared 4-way provenance classifier. Stamped onto each run at scan time (on a
# bench node, where the per-run artifacts are still local) so the measured/modeled/
# functional/unknown verdict survives even after those artifacts are gitignored on a
# driver clone. Imported defensively: a clone missing the module still builds, just
# without the stamp (the consumer re-derives from tags/run_id).
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bench_provenance  # noqa: E402
except ImportError:  # pragma: no cover - defensive
    bench_provenance = None  # type: ignore[assignment]

# Paths (relative to repo root). The Go module — and the benchmark tree — is the
# repository root: experiments/benchmark/, NOT fak/experiments/benchmark/. The
# stale "fak/" prefix is a fossil from when the module sat under a fak/ subdir; it
# pointed every command at a non-existent directory, so build scanned nothing (and
# would have WIPED the committed catalog), validate flagged every run as missing,
# and add-machine wrote specs to a dead path.
ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "experiments" / "benchmark"
MACHINES_DIR = BENCHMARK_DIR / "machines"
RUNS_DIR = BENCHMARK_DIR / "runs" / "by-machine"
CATALOG_PATH = BENCHMARK_DIR / "catalog.json"
SCHEMAS_DIR = ROOT / "tools" / "schemas"


def normalize_run_path(path_str: str) -> str:
    """Strip a stale leading ``fak/`` (or ``fak\\``) segment from a run path.

    The benchmark tree is rooted at ``<repo>/experiments/benchmark``; older
    catalog data embedded a non-existent ``fak/experiments/...`` prefix from when
    the module sat under a ``fak/`` subdir. Separator style is preserved so the
    value still matches what ``scan_runs`` emits on each OS.
    """
    if not path_str:
        return path_str
    for prefix in ("fak/", "fak\\"):
        if path_str.startswith(prefix):
            return path_str[len(prefix):]
    return path_str


def load_json(path: Path) -> Any:
    """Load JSON file, return None on error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[WARN] Failed to load {path}: {e}", file=sys.stderr)
        return None


def save_json(path: Path, data: Any) -> bool:
    """Save JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        # newline="\n": the catalog is a committed, cross-machine-regenerated
        # artifact — pin LF so a Windows rebuild does not churn the whole file to
        # CRLF vs an LF original (and vice-versa).
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        tmp.replace(path)
        return True
    except OSError as e:
        print(f"[ERROR] Failed to save {path}: {e}", file=sys.stderr)
        return False


def load_catalog() -> Dict:
    """Load existing catalog or return empty structure."""
    if CATALOG_PATH.exists():
        data = load_json(CATALOG_PATH)
        if data:
            return data
    return {
        "$schema": "benchmark/catalog.v1",
        "version": "1.0",
        "last_updated": None,
        "machines": {},
        "runs": [],
        "index": {"by_model": {}, "by_precision": {}, "by_date": {}}
    }


def scan_machines() -> Dict[str, Dict]:
    """Scan machines directory for all machine specs."""
    machines = {}
    if not MACHINES_DIR.exists():
        return machines

    for specs_path in MACHINES_DIR.glob("*/specs.json"):
        machine_id = specs_path.parent.name
        specs = load_json(specs_path)
        if not specs:
            continue
        machines[machine_id] = {
            "id": machine_id,
            "hostname": specs.get("hostname", "?"),
            "os": specs.get("os", {}).get("name", "?"),
            "arch": specs.get("hardware", {}).get("cpu", {}).get("architecture", "?"),
            "cpu_cores": specs.get("hardware", {}).get("cpu", {}).get("cores_logical", 0),
            "gpu": extract_gpu_name(specs),
            # Surface placement in the rollup: an "agent-host" node is kept for
            # agent use and is NOT a bench target (run-on-bench-nodes-by-default);
            # bench runs route to the "bench-node" machines.
            "role": specs.get("role", "bench-node"),
            "runs": 0,
            "last_run": None
        }
    return machines


def extract_gpu_name(specs: Dict) -> str:
    """Extract GPU model name from specs."""
    gpus = specs.get("hardware", {}).get("gpu", [])
    if gpus and isinstance(gpus, list) and len(gpus) > 0:
        return gpus[0].get("model", "unknown")
    return "none"


def scan_engine_fields(run_dir: Path) -> List[str]:
    """Collect the ``engine`` (or ``generated_by``) strings from a run's artifacts.

    The per-bench result JSONs (e.g. ``01-qwen25-1.5b-q8-decode.json``) name the
    engine that produced them — "fak-in-kernel ... decode", "fak radixbench", "fak
    model load". That is the highest-trust provenance signal (Rule 1), but it lives
    only in the bench-node-local artifact, so we read it here at scan time and let
    the caller stamp the derived verdict onto the catalog run.
    """
    engines: List[str] = []
    if not run_dir.is_dir():
        return engines
    for jf in sorted(run_dir.glob("*.json")):
        data = load_json(jf)
        if isinstance(data, dict):
            eng = data.get("engine") or data.get("generated_by")
            if isinstance(eng, str) and eng:
                engines.append(eng)
    return engines


def scan_runs() -> List[Dict]:
    """Scan all run directories for manifest.json files."""
    runs = []
    if not RUNS_DIR.exists():
        return runs

    for manifest_path in RUNS_DIR.glob("*/*/manifest.json"):
        manifest = load_json(manifest_path)
        if not manifest:
            continue

        run_dir = manifest_path.parent
        machine_id = run_dir.parent.name

        # Extract summary metrics from associated files
        batch_path = run_dir / "batch.json"

        peak_tok = None
        baseline_tok = None
        speedup = None

        if batch_path.exists():
            batch = load_json(batch_path)
            if batch:
                peak = batch.get("peak", {})
                peak_tok = peak.get("agg_tok_per_sec")
                baseline = batch.get("baseline", {})
                baseline_tok = baseline.get("tok_per_sec")
                if peak_tok and baseline_tok:
                    speedup = peak_tok / baseline_tok

        run = {
            "run_id": manifest.get("run_id", run_dir.name),
            "machine_id": machine_id,
            "timestamp": manifest.get("timestamp"),
            "model": manifest.get("model", {}).get("name"),
            "precision": manifest.get("model", {}).get("precision"),
            "tags": manifest.get("tags", []),
            "peak_tok_per_sec": peak_tok,
            "baseline_tok_per_sec": baseline_tok,
            "speedup": speedup,
            "path": str(run_dir.relative_to(ROOT))
        }
        # Stamp the provenance verdict from the (now-local) artifact engine fields,
        # so it travels with the catalog after the artifacts are gitignored away.
        if bench_provenance is not None:
            engines = scan_engine_fields(run_dir)
            run["provenance"] = bench_provenance.classify(run, artifact_engines=engines)
        runs.append(run)
    return runs


def build_indexes(runs: List[Dict]) -> Dict[str, Dict[str, List[str]]]:
    """Build cross-reference indexes."""
    indexes: Dict[str, Dict[str, List[str]]] = {
        "by_model": {},
        "by_precision": {},
        "by_date": {}
    }

    for run in runs:
        run_id = run.get("run_id")
        if not run_id:
            continue
        model = run.get("model", "unknown")
        precision = run.get("precision", "unknown")
        timestamp = run.get("timestamp", "")

        # Parse date from timestamp
        date_match = re.match(r"(\d{4}-\d{2}-\d{2})", timestamp)
        date = date_match.group(1) if date_match else "unknown"

        indexes["by_model"].setdefault(model, []).append(run_id)
        indexes["by_precision"].setdefault(precision, []).append(run_id)
        indexes["by_date"].setdefault(date, []).append(run_id)

    return indexes


def update_machine_stats(catalog: Dict) -> None:
    """Update run counts and last run timestamps per machine."""
    runs_by_machine: Dict[str, int] = {}
    last_run_by_machine: Dict[str, str] = {}

    for run in catalog["runs"]:
        mid = run.get("machine_id")
        if not mid:
            continue
        runs_by_machine[mid] = runs_by_machine.get(mid, 0) + 1
        ts = run.get("timestamp", "")
        if ts:
            current = last_run_by_machine.get(mid, "")
            if ts > current:
                last_run_by_machine[mid] = ts

    for machine_id, stats in catalog["machines"].items():
        stats["runs"] = runs_by_machine.get(machine_id, 0)
        stats["last_run"] = last_run_by_machine.get(machine_id)


def build_catalog() -> Dict:
    """Rebuild the catalog by scanning the filesystem, merging with the committed copy.

    Merge — not scan-and-replace — because a run's artifacts (manifest.json /
    batch.json) live on the bench node that produced them and are gitignored in a
    driver/agent-host clone. A naive scan there finds zero runs and would erase the
    committed run history. So we union: locally-scanned runs refresh their entry by
    normalized run-directory path; committed runs with no local artifact are
    preserved (with their stale ``fak/`` path prefix normalized). Machine specs.json
    overlay any existing catalog entry. On a full-artifact bench node the local scan
    simply overlays everything, so the result is identical to a from-scratch build.
    """
    existing = load_catalog()

    print("[bench_catalog] Scanning machines...", file=sys.stderr)
    scanned_machines = scan_machines()
    machines = dict(existing.get("machines", {}))
    for mid, rec in scanned_machines.items():
        machines[mid] = {**machines.get(mid, {}), **rec}
    print(f"[bench_catalog] {len(scanned_machines)} machine spec(s) on disk, "
          f"{len(machines)} in catalog", file=sys.stderr)

    print("[bench_catalog] Scanning runs...", file=sys.stderr)
    scanned_runs = scan_runs()
    # Key the merge by run DIRECTORY, not run_id: a run *is* its directory, the
    # path is globally unique (it embeds machine_id + timestamp), and the run_id
    # generator can collide across distinct dirs. Normalize separators for the key
    # so a Windows-captured (backslash) and a Mac-captured (forward-slash) path for
    # the same run collapse instead of duplicating.
    def _path_key(p: str) -> str:
        return normalize_run_path(p or "").replace("\\", "/")
    runs_by_path: Dict[str, Dict] = {}
    for i, run in enumerate(existing.get("runs", [])):
        run = dict(run)
        run["path"] = normalize_run_path(run.get("path", ""))
        # A path-less committed run (legacy/hand-edited) must not collapse onto the
        # empty key and overwrite its siblings — give each a unique fallback.
        runs_by_path[_path_key(run["path"]) or f"\x00existing#{i}"] = run
    n_preserved = len(runs_by_path)
    for run in scanned_runs:
        runs_by_path[_path_key(run.get("path", "")) or f"\x00scanned#{run.get('run_id')}"] = run
    runs = list(runs_by_path.values())
    print(f"[bench_catalog] {len(scanned_runs)} run(s) scanned locally, "
          f"{n_preserved} preserved from catalog, {len(runs)} total", file=sys.stderr)

    print("[bench_catalog] Building indexes...", file=sys.stderr)
    indexes = build_indexes(runs)

    catalog = {
        "$schema": "benchmark/catalog.v1",
        "version": existing.get("version", "1.0"),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "machines": machines,
        "runs": runs,
        "index": indexes
    }
    update_machine_stats(catalog)

    return catalog


def validate_catalog(catalog: Dict) -> List[str]:
    """Validate structural catalog integrity, return list of hard errors.

    Hard errors are inconsistencies the catalog itself owns: missing top-level
    keys, runs pointing at an unregistered machine, and index entries pointing at
    a run that is not in ``runs``. A run whose local directory is absent is NOT a
    hard error — those artifacts legitimately live on the remote bench node and
    are gitignored in a driver clone; see ``missing_run_paths`` for that warning.
    """
    errors = []

    # Check structure
    for key in ["version", "machines", "runs", "index"]:
        if key not in catalog:
            errors.append(f"Missing required key: {key}")

    # Check machine references
    machine_ids = set(catalog.get("machines", {}).keys())
    for run in catalog.get("runs", []):
        mid = run.get("machine_id")
        if mid and mid not in machine_ids:
            errors.append(f"Run {run.get('run_id')} references unknown machine: {mid}")

    # Check index consistency
    index = catalog.get("index", {})
    all_run_ids = {r.get("run_id") for r in catalog.get("runs", [])}

    for idx_type, idx in index.items():
        for key, run_ids in idx.items():
            for rid in run_ids:
                if rid not in all_run_ids:
                    errors.append(f"Index {idx_type}/{key} references unknown run: {rid}")

    return errors


def missing_run_paths(catalog: Dict) -> List[str]:
    """Return non-fatal warnings for runs whose local directory is absent.

    Run artifacts are produced on (and stay on) the bench node; a driver/agent-host
    clone holds the committed catalog rollup but not the gitignored per-run dirs. A
    missing path is therefore expected here — surfaced as a warning, never an error.
    """
    warnings = []
    for run in catalog.get("runs", []):
        path_str = run.get("path")
        if path_str and not (ROOT / path_str).exists():
            warnings.append(f"Run {run.get('run_id')} has no local dir: {path_str}")
    return warnings


def add_machine(specs_path: Path) -> bool:
    """Register a new machine from specs file."""
    specs = load_json(specs_path)
    if not specs:
        print(f"[ERROR] Failed to load specs from {specs_path}", file=sys.stderr)
        return False

    machine_id = specs.get("machine_id") or specs_path.parent.name
    dest_dir = MACHINES_DIR / machine_id
    dest_path = dest_dir / "specs.json"

    if dest_path.exists():
        print(f"[WARN] Machine {machine_id} already registered, overwriting", file=sys.stderr)

    dest_dir.mkdir(parents=True, exist_ok=True)
    return save_json(dest_path, specs)


def add_run(manifest_path: Path) -> bool:
    """Register a new run from manifest file."""
    manifest = load_json(manifest_path)
    if not manifest:
        print(f"[ERROR] Failed to load manifest from {manifest_path}", file=sys.stderr)
        return False

    # Verify manifest is in correct location
    run_dir = manifest_path.parent
    expected_name = manifest.get("run_id")
    if expected_name and run_dir.name != expected_name:
        print(f"[WARN] Manifest run_id '{expected_name}' doesn't match directory '{run_dir.name}'",
              file=sys.stderr)

    # Rebuild catalog to include new run
    catalog = build_catalog()
    return save_json(CATALOG_PATH, catalog)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", help="Command to run")

    sub.add_parser("build", help="Rebuild catalog from scratch")
    sub.add_parser("update", help="Incremental update (same as build)")
    sub.add_parser("validate", help="Validate catalog integrity")

    add_machine_p = sub.add_parser("add-machine", help="Register a new machine")
    add_machine_p.add_argument("specs", type=Path, help="Path to specs.json file")

    add_run_p = sub.add_parser("add-run", help="Register a new run")
    add_run_p.add_argument("manifest", type=Path, help="Path to manifest.json file")

    sub.add_parser("show", help="Show current catalog (JSON to stdout)")

    args = ap.parse_args(argv)

    if not args.command:
        ap.print_help()
        return 1

    if args.command == "build" or args.command == "update":
        catalog = build_catalog()
        if save_json(CATALOG_PATH, catalog):
            print(f"[bench_catalog] Catalog saved to {CATALOG_PATH.relative_to(ROOT)}", file=sys.stderr)
            return 0
        return 1

    if args.command == "validate":
        catalog = load_json(CATALOG_PATH)
        if not catalog:
            print(f"[ERROR] Failed to load catalog from {CATALOG_PATH}", file=sys.stderr)
            return 1
        errors = validate_catalog(catalog)
        warnings = missing_run_paths(catalog)
        if warnings:
            print(f"[bench_catalog] {len(warnings)} run(s) have no local dir "
                  f"(artifacts live on the bench node — expected in a driver clone)",
                  file=sys.stderr)
        if errors:
            print(f"[ERROR] Catalog validation failed with {len(errors)} error(s):", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 1
        print("[bench_catalog] Catalog is structurally valid", file=sys.stderr)
        return 0

    if args.command == "add-machine":
        if add_machine(args.specs):
            return 0
        return 1

    if args.command == "add-run":
        if add_run(args.manifest):
            return 0
        return 1

    if args.command == "show":
        catalog = load_json(CATALOG_PATH)
        if catalog:
            print(json.dumps(catalog, indent=2))
            return 0
        return 1

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
