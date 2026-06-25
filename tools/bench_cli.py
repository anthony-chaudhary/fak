#!/usr/bin/env python3
"""bench_cli.py -- query and compare benchmark results.

Usage:
  python tools/bench_cli.py list [--machine M] [--model M] [--precision P]
  python tools/bench_cli.py show <run-id>
  python tools/bench_cli.py compare <run-id-1> <run-id-2>
  python tools/bench_cli.py best --model M --metric peak_tok_per_sec
  python tools/bench_cli.py table --model M --format markdown
  python tools/bench_cli.py summary --group-by machine
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
# The Go module is the repository root, so committed benchmark results live at
# experiments/benchmark  -  NOT fak/experiments/benchmark (a leftover doubled-root
# prefix from when the module lived in a fak/ subdir). The doubled prefix made
# `bench_cli.py list` fail to find catalog.json from the repo root.
BENCHMARK_DIR = ROOT / "experiments" / "benchmark"
CATALOG_PATH = BENCHMARK_DIR / "catalog.json"


def load_catalog() -> Optional[Dict]:
    """Load catalog, return None on error."""
    if not CATALOG_PATH.exists():
        print(f"[ERROR] Catalog not found at {CATALOG_PATH}", file=sys.stderr)
        print("[ERROR] Run 'python tools/bench_catalog.py build' first", file=sys.stderr)
        return None
    try:
        with open(CATALOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[ERROR] Failed to load catalog: {e}", file=sys.stderr)
        return None


def load_run(run_id: str, catalog: Dict) -> Optional[Dict]:
    """Load full run data including manifest and results."""
    # Find run in catalog
    run_entry = None
    for r in catalog.get("runs", []):
        if r["run_id"] == run_id:
            run_entry = r
            break

    if not run_entry:
        print(f"[ERROR] Run '{run_id}' not found in catalog", file=sys.stderr)
        return None

    run_dir = ROOT / run_entry["path"]

    # Load manifest
    manifest_path = run_dir / "manifest.json"
    manifest = None
    if manifest_path.exists():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    # Load results
    results = {}
    for name in ["kernel", "batch", "modelbench", "fleetbench"]:
        path = run_dir / f"{name}.json"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    results[name] = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

    return {
        "entry": run_entry,
        "manifest": manifest,
        "results": results
    }


def cmd_list(args: argparse.Namespace, catalog: Dict) -> int:
    """List runs with optional filtering."""
    runs = catalog.get("runs", [])

    # Apply filters
    if args.machine:
        runs = [r for r in runs if r.get("machine_id") == args.machine]
    if args.model:
        runs = [r for r in runs if args.model.lower() in r.get("model", "").lower()]
    if args.precision:
        runs = [r for r in runs if r.get("precision") == args.precision]
    if args.since:
        runs = [r for r in runs if r.get("timestamp", "") >= args.since]
    if args.until:
        runs = [r for r in runs if r.get("timestamp", "") <= args.until]

    if args.format == "json":
        print(json.dumps(runs, indent=2))
        return 0

    # Table output
    if not runs:
        print("No runs found matching filters", file=sys.stderr)
        return 0

    # Header
    print(f"{'RUN ID':<50} {'MACHINE':<20} {'MODEL':<20} {'PREC':<8} {'PEAK T/S':>12}")
    print("-" * 110)

    for r in sorted(runs, key=lambda x: x.get("timestamp", ""), reverse=True):
        peak = r.get("peak_tok_per_sec")
        peak_str = f"{peak:.1f}" if peak else "-"
        print(f"{r['run_id']:<50} {r['machine_id']:<20} {r.get('model', '-'):<20} "
              f"{r.get('precision', '-'):<8} {peak_str:>12}")

    print(f"\n{len(runs)} run(s)")

    return 0


def cmd_show(args: argparse.Namespace, catalog: Dict) -> int:
    """Show detailed run information."""
    run = load_run(args.run_id, catalog)
    if not run:
        return 1

    manifest = run["manifest"] or {}
    entry = run["entry"]
    results = run["results"]

    print(f"Run: {entry['run_id']}")
    print(f"Machine: {entry['machine_id']}")
    print(f"Timestamp: {entry.get('timestamp', '?')}")
    print()

    if manifest:
        print("Config:")
        config = manifest.get("config", {})
        print(f"  Batch sizes: {config.get('batch_sizes', [])}")
        print(f"  Workers: {config.get('workers', '?')}")
        print(f"  Decode steps: {config.get('decode_steps', '?')}")
        print()

    print("Results:")
    if "batch" in results:
        batch = results["batch"]
        peak = batch.get("peak", {})
        baseline = batch.get("baseline", {})
        print(f"  Baseline (B=1): {baseline.get('tok_per_sec', '?'):.1f} tok/s")
        print(f"  Peak (B={peak.get('batch', '?')}): {peak.get('agg_tok_per_sec', '?'):.1f} tok/s")
        print(f"  Speedup: {peak.get('speedup_vs_baseline', '?'):.2f}x")
    else:
        print("  No batch results")
    print()

    print(f"Path: {entry['path']}")

    return 0


def cmd_compare(args: argparse.Namespace, catalog: Dict) -> int:
    """Compare two runs."""
    run1 = load_run(args.run_id_1, catalog)
    run2 = load_run(args.run_id_2, catalog)

    if not run1 or not run2:
        return 1

    print("=== Benchmark Run Comparison ===")
    print()
    print(f"Run 1: {run1['entry']['run_id']}")
    print(f"  Machine: {run1['entry']['machine_id']}")
    print(f"  Timestamp: {run1['entry'].get('timestamp', '?')}")
    print()
    print(f"Run 2: {run2['entry']['run_id']}")
    print(f"  Machine: {run2['entry']['machine_id']}")
    print(f"  Timestamp: {run2['entry'].get('timestamp', '?')}")
    print()

    # Compare batch results
    batch1 = run1["results"].get("batch")
    batch2 = run2["results"].get("batch")

    if batch1 and batch2:
        print("Batch Decode Comparison:")
        print(f"  {'Metric':<25} {'Run 1':>15} {'Run 2':>15} {'Ratio':>10}")
        print("-" * 65)

        b1 = batch1.get("baseline", {})
        b2 = batch2.get("baseline", {})
        print(f"  {'Baseline (B=1) tok/s':<25} {b1.get('tok_per_sec', 0):>15.1f} "
              f"{b2.get('tok_per_sec', 0):>15.1f} {b2.get('tok_per_sec', 0)/b1.get('tok_per_sec', 1):>10.2f}x")

        p1 = batch1.get("peak", {})
        p2 = batch2.get("peak", {})
        print(f"  {'Peak tok/s':<25} {p1.get('agg_tok_per_sec', 0):>15.1f} "
              f"{p2.get('agg_tok_per_sec', 0):>15.1f} {p2.get('agg_tok_per_sec', 0)/p1.get('agg_tok_per_sec', 1):>10.2f}x")

        print(f"  {'Peak batch size':<25} {p1.get('batch', 0):>15} {p2.get('batch', 0):>15} -")

        s1 = p1.get('speedup_vs_baseline', 0)
        s2 = p2.get('speedup_vs_baseline', 0)
        print(f"  {'Speedup vs baseline':<25} {s1:>15.2f}x {s2:>15.2f}x -")

    return 0


def cmd_best(args: argparse.Namespace, catalog: Dict) -> int:
    """Find best run for a model by metric."""
    runs = catalog.get("runs", [])

    if args.model:
        runs = [r for r in runs if r.get("model") == args.model]

    metric = args.metric or "peak_tok_per_sec"

    if metric == "peak_tok_per_sec":
        best = max(runs, key=lambda r: r.get("peak_tok_per_sec") or 0)
    elif metric == "speedup":
        best = max(runs, key=lambda r: r.get("speedup") or 0)
    else:
        print(f"[ERROR] Unknown metric: {metric}", file=sys.stderr)
        return 1

    print(f"Best run by {metric}:")
    print(f"  Run ID: {best['run_id']}")
    print(f"  Machine: {best['machine_id']}")
    print(f"  Model: {best.get('model', '?')}")
    print(f"  Value: {best.get(metric, '?')}")
    print(f"  Timestamp: {best.get('timestamp', '?')}")

    return 0


def cmd_table(args: argparse.Namespace, catalog: Dict) -> int:
    """Generate comparison table."""
    runs = catalog.get("runs", [])

    if args.model:
        runs = [r for r in runs if r.get("model") == args.model]
    if args.precision:
        runs = [r for r in runs if r.get("precision") == args.precision]

    if not runs:
        print("No runs found", file=sys.stderr)
        return 0

    if args.format == "json":
        print(json.dumps(runs, indent=2))
        return 0

    # Markdown table
    print(f"| {'Machine':<20} | {'Peak T/S':>12} | {'Baseline T/S':>14} | {'Speedup':>8} |")
    print("|" + "-" * 21 + "|" + "-" * 14 + "|" + "-" * 16 + "|" + "-" * 10 + "|")

    for r in runs:
        peak = r.get("peak_tok_per_sec") or 0
        baseline = r.get("baseline_tok_per_sec") or 0
        speedup = r.get("speedup") or 0
        print(f"| {r['machine_id']:<20} | {peak:>12.1f} | {baseline:>14.1f} | {speedup:>8.2f}x |")

    return 0


def cmd_summary(args: argparse.Namespace, catalog: Dict) -> int:
    """Show summary statistics grouped by field."""
    runs = catalog.get("runs", [])

    if args.group_by == "machine":
        groups: Dict[str, List[Dict]] = {}
        for r in runs:
            mid = r.get("machine_id", "unknown")
            groups.setdefault(mid, []).append(r)

        print("=== Summary by Machine ===")
        print()
        for machine_id, machine_runs in sorted(groups.items()):
            count = len(machine_runs)
            best = max(machine_runs, key=lambda r: r.get("peak_tok_per_sec") or 0)
            avg_peak = sum(r.get("peak_tok_per_sec") or 0 for r in machine_runs) / count
            print(f"{machine_id}:")
            print(f"  Runs: {count}")
            print(f"  Best peak: {best.get('peak_tok_per_sec', 0):.1f} tok/s")
            print(f"  Avg peak: {avg_peak:.1f} tok/s")
            print()

    elif args.group_by == "model":
        groups: Dict[str, List[Dict]] = {}
        for r in runs:
            model = r.get("model", "unknown")
            groups.setdefault(model, []).append(r)

        print("=== Summary by Model ===")
        print()
        for model, model_runs in sorted(groups.items()):
            count = len(model_runs)
            print(f"{model}: {count} run(s)")

    return 0


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", help="Command to run")

    # list
    list_p = sub.add_parser("list", help="List benchmark runs")
    list_p.add_argument("--machine", help="Filter by machine ID")
    list_p.add_argument("--model", help="Filter by model name (substring)")
    list_p.add_argument("--precision", help="Filter by precision")
    list_p.add_argument("--since", help="Filter by start date (ISO)")
    list_p.add_argument("--until", help="Filter by end date (ISO)")
    list_p.add_argument("--format", choices=["table", "json"], default="table",
                       help="Output format")

    # show
    show_p = sub.add_parser("show", help="Show detailed run info")
    show_p.add_argument("run_id", help="Run ID to show")

    # compare
    compare_p = sub.add_parser("compare", help="Compare two runs")
    compare_p.add_argument("run_id_1", help="First run ID")
    compare_p.add_argument("run_id_2", help="Second run ID")

    # best
    best_p = sub.add_parser("best", help="Find best run")
    best_p.add_argument("--model", help="Filter by model")
    best_p.add_argument("--metric", choices=["peak_tok_per_sec", "speedup"],
                       default="peak_tok_per_sec", help="Metric to optimize")

    # table
    table_p = sub.add_parser("table", help="Generate comparison table")
    table_p.add_argument("--model", help="Filter by model")
    table_p.add_argument("--precision", help="Filter by precision")
    table_p.add_argument("--format", choices=["markdown", "json"], default="markdown",
                       help="Output format")

    # summary
    summary_p = sub.add_parser("summary", help="Summary statistics")
    summary_p.add_argument("--group-by", choices=["machine", "model"],
                          default="machine", help="Group by field")

    args = ap.parse_args(argv)

    if not args.command:
        ap.print_help()
        return 1

    catalog = load_catalog()
    if not catalog:
        return 1

    if args.command == "list":
        return cmd_list(args, catalog)
    if args.command == "show":
        return cmd_show(args, catalog)
    if args.command == "compare":
        return cmd_compare(args, catalog)
    if args.command == "best":
        return cmd_best(args, catalog)
    if args.command == "table":
        return cmd_table(args, catalog)
    if args.command == "summary":
        return cmd_summary(args, catalog)

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
