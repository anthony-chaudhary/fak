#!/usr/bin/env python3
"""bench_slack.py -- Slack integration for benchmark catalog.

Provides Slack-friendly commands and formatting for the benchmark infrastructure,
designed to work with the slack-control bridge used for DGX and remote machines.

Usage (through slack-control bridge):
  !bench list
  !bench list --machine anthony
  !bench show <run-id>
  !bench summary --group-by machine
  !bench best --model SmolLM2-135M-Instruct

Or directly:
  python tools/bench_slack.py list --machine anthony --format slack
  python tools/bench_slack.py transfer --help
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def _slack_helpers_candidates() -> List[Path]:
    """Where to look for the optional slack-helpers package, most-specific first.

    Honors ``$SLACK_HELPERS_PATH`` (the override the not-found error advertises),
    then falls back to generic, machine-independent locations: a sibling checkout
    next to this repo and one under the user's home. No operator-specific machine
    path is baked in (that would be a redact-needle on a public mirror).
    """
    out: List[Path] = []
    env = os.environ.get("SLACK_HELPERS_PATH")
    if env:
        out.append(Path(env))
    out += [ROOT.parent / "slack-helpers", Path.home() / "slack-helpers"]
    return out
BENCHMARK_DIR = ROOT / "fak" / "experiments" / "benchmark"
CATALOG_PATH = BENCHMARK_DIR / "catalog.json"


def load_catalog() -> Optional[Dict]:
    """Load catalog, return None on error."""
    if not CATALOG_PATH.exists():
        return None
    try:
        with open(CATALOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def format_slack_table(runs: List[Dict], show_machine: bool = True) -> str:
    """Format runs as a Slack table."""
    if not runs:
        return "_No runs found_"

    # Slack code table format
    lines = ["```"]

    # Header
    if show_machine:
        lines.append(f"{'RUN ID':<32} {'MACHINE':<12} {'MODEL':<15} {'PEAK T/S':>10}")
        lines.append("-" * 70)
    else:
        lines.append(f"{'RUN ID':<45} {'MODEL':<15} {'PEAK T/S':>10}")
        lines.append("-" * 70)

    # Sort by timestamp (most recent first)
    sorted_runs = sorted(runs, key=lambda r: r.get("timestamp", ""), reverse=True)

    for r in sorted_runs[:10]:  # Limit to 10 most recent for Slack
        run_id = r['run_id'][:32]  # Truncate for display
        peak = r.get("peak_tok_per_sec")
        peak_str = f"{peak:.1f}" if peak else "-"

        if show_machine:
            lines.append(f"{run_id:<32} {r['machine_id']:<12} {r.get('model', '-')[:15]:<15} {peak_str:>10}")
        else:
            lines.append(f"{run_id:<45} {r.get('model', '-')[:15]:<15} {peak_str:>10}")

    if len(sorted_runs) > 10:
        lines.append(f"... and {len(sorted_runs) - 10} more")

    lines.append("```")
    lines.append(f"_Total: {len(runs)} run(s)_")

    return "\n".join(lines)


def format_slack_summary(catalog: Dict) -> str:
    """Format catalog summary for Slack."""
    machines = catalog.get("machines", {})
    runs = catalog.get("runs", [])

    lines = ["*Benchmark Catalog Summary*", ""]

    # Machine stats
    lines.append("*Machines:*")
    for machine_id, machine_info in sorted(machines.items()):
        run_count = machine_info.get("runs", 0)
        gpu = machine_info.get("gpu", "unknown")
        lines.append(f"  • `{machine_id}`: {run_count} run(s) | GPU: {gpu}")

    lines.append("")

    # Recent runs
    lines.append("*Recent Runs (last 5):*")
    recent = sorted(runs, key=lambda r: r.get("timestamp", ""), reverse=True)[:5]
    for r in recent:
        peak = r.get("peak_tok_per_sec")
        peak_str = f"{peak:.1f} tok/s" if peak else "N/A"
        lines.append(f"  • `{r['run_id'][:30]}...` | {r['machine_id']} | {peak_str}")

    return "\n".join(lines)


def format_slack_show(run: Dict) -> str:
    """Format detailed run info for Slack."""
    lines = [f"*Run: {run.get('run_id', 'unknown')}*"]
    lines.append(f"Machine: `{run.get('machine_id', 'unknown')}`")
    lines.append(f"Timestamp: {run.get('timestamp', 'unknown')}")
    lines.append("")

    # Config
    config = run.get("config", {})
    if config:
        lines.append("*Configuration:*")
        if config.get("batch_sizes"):
            lines.append(f"  Batch sizes: {config.get('batch_sizes')}")
        if config.get("workers"):
            lines.append(f"  Workers: {config.get('workers')}")
        lines.append("")

    # Results
    peak = run.get("peak_tok_per_sec")
    baseline = run.get("baseline_tok_per_sec")
    speedup = run.get("speedup")

    lines.append("*Results:*")
    if baseline:
        lines.append(f"  Baseline: {baseline:.1f} tok/s")
    if peak:
        lines.append(f"  Peak: {peak:.1f} tok/s")
    if speedup:
        lines.append(f"  Speedup: {speedup:.2f}x")

    tags = run.get("tags", [])
    if tags:
        lines.append(f"  Tags: {', '.join(tags)}")

    return "\n".join(lines)


def cmd_list_slack(args: argparse.Namespace, catalog: Dict) -> str:
    """List runs with Slack formatting."""
    runs = catalog.get("runs", [])

    # Apply filters
    if args.machine:
        runs = [r for r in runs if r.get("machine_id") == args.machine]
    if args.model:
        runs = [r for r in runs if args.model.lower() in r.get("model", "").lower()]
    if args.precision:
        runs = [r for r in runs if r.get("precision") == args.precision]

    if args.format == "summary":
        return format_slack_summary(catalog)

    return format_slack_table(runs, show_machine=True)


def cmd_show_slack(args: argparse.Namespace, catalog: Dict) -> str:
    """Show run with Slack formatting."""
    # Find run
    run = None
    for r in catalog.get("runs", []):
        if r["run_id"] == args.run_id or r["run_id"].startswith(args.run_id):
            run = r
            break

    if not run:
        return f"_Run '{args.run_id}' not found_"

    return format_slack_show(run)


def cmd_best_slack(args: argparse.Namespace, catalog: Dict) -> str:
    """Find best run with Slack formatting."""
    runs = catalog.get("runs", [])

    if args.model:
        runs = [r for r in runs if args.model.lower() in r.get("model", "").lower()]

    if not runs:
        return "_No matching runs_"

    best = max(runs, key=lambda r: r.get("peak_tok_per_sec") or 0)

    lines = ["*Best Run by Peak Throughput:*"]
    lines.append(format_slack_show(best))

    return "\n".join(lines)


def cmd_summary_slack(args: argparse.Namespace, catalog: Dict) -> str:
    """Summary with Slack formatting."""
    return format_slack_summary(catalog)


def cmd_status(args: argparse.Namespace, catalog: Dict) -> str:
    """Quick status for Slack !status command."""
    machines = catalog.get("machines", {})
    runs = catalog.get("runs", [])

    total_runs = len(runs)
    total_machines = len(machines)

    # Find most recent run
    recent = sorted(runs, key=lambda r: r.get("timestamp", ""), reverse=True)[:1]

    lines = ["*Benchmark Catalog Status*"]
    lines.append(f"[STATUS] {total_runs} runs across {total_machines} machines")

    if recent:
        r = recent[0]
        lines.append(f"[Latest] `{r['run_id'][:40]}...` on {r['machine_id']}")

    lines.append("")
    lines.append("*Commands:* `!bench list`, `!bench summary`, `!bench best --model <model>`")

    return "\n".join(lines)


def register_dgx_run(run_dir: Path, machine_id: str = "dgx") -> Optional[str]:
    """Register a DGX run into the catalog.

    Called from DGX test scripts after a benchmark completes.
    Creates a manifest and updates the catalog.

    Args:
        run_dir: Directory containing benchmark results
        machine_id: Machine identifier (default: dgx)

    Returns:
        Run ID if successful, None otherwise
    """
    # Import catalog builder
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from bench_catalog import build_catalog, save_json, CATALOG_PATH
    except ImportError:
        return None

    # Check if run_dir exists
    if not run_dir.exists():
        return None

    # Look for existing results
    results = {}
    for name in ["kernel.json", "batch.json", "fleetbench.json", "dgx-summary.json"]:
        path = run_dir / name
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    results[name.replace(".json", "")] = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

    if not results:
        return None

    # Generate timestamp
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model = "unknown"
    precision = "unknown"

    # Try to extract model/precision from results
    if "dgx-summary" in results:
        summary = results["dgx-summary"]
        model = summary.get("model", model)
        precision = summary.get("precision", precision)
    elif "batch" in results:
        batch = results["batch"]
        model = batch.get("model", model)
        precision = batch.get("precision", precision)

    run_id = f"{machine_id}-{model.replace(' ', '-').lower()}-{precision}-{ts}"

    # Create manifest
    manifest = {
        "$schema": "benchmark/run-manifest.v1",
        "run_id": run_id,
        "machine_id": machine_id,
        "timestamp": ts,
        "git": {"rev": "unknown", "branch": "master", "dirty": False},
        "harness": {"name": "dgx-benchmark", "version": "unknown"},
        "model": {
            "name": model,
            "precision": precision,
            "quantization": precision
        },
        "config": {},
        "tags": ["dgx", "remote", "slack-control"],
        "artifacts": {k: f"{k}.json" for k in results.keys()}
    }

    # Write manifest
    catalog_runs_dir = BENCHMARK_DIR / "runs" / "by-machine" / machine_id / f"{ts}-dgx"
    catalog_runs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = catalog_runs_dir / "manifest.json"
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except OSError:
        return None

    # Copy result files
    for name in results.keys():
        src = run_dir / f"{name}.json"
        if src.exists():
            import shutil
            shutil.copy2(src, catalog_runs_dir / f"{name}.json")

    # Rebuild catalog
    catalog = build_catalog()
    save_json(CATALOG_PATH, catalog)

    return run_id


def run_benchmark_and_transfer(
    command: str,
    slack_helpers_path: Optional[Path] = None,
    control_channel: str = "dgx-control",
    state_file: Optional[str] = None,
    poll_interval: int = 30,
    timeout: int = 3600,
    machine_id: str = "dgx",
) -> Optional[str]:
    """End-to-end: run benchmark via Slack control and transfer results.

    Args:
        command: Benchmark command to run on DGX
        slack_helpers_path: Path to slack-helpers package (auto-detected if None)
        control_channel: Slack control channel name
        state_file: State file path for control bridge
        poll_interval: Seconds between polls for completion
        timeout: Maximum seconds to wait for completion
        machine_id: Machine ID for catalog registration

    Returns:
        Run ID if successful, None otherwise
    """
    # Auto-detect slack-helpers path
    if slack_helpers_path is None:
        for cand in _slack_helpers_candidates():
            if (cand / "slack_helpers").exists():
                slack_helpers_path = cand
                break
        if slack_helpers_path is None:
            print("ERROR: Could not find slack-helpers package", file=sys.stderr)
            print("Set SLACK_HELPERS_PATH or pass --slack-helpers-path", file=sys.stderr)
            return None

    # Default state file paths
    if state_file is None:
        state_file = str(ROOT / "fak" / ".slack_control_transfer_state.json")

    lock_file = state_file + ".lock"
    transcript_file = state_file + ".transcript.jsonl"

    print(f"Slack helpers: {slack_helpers_path}")
    print(f"Control channel: {control_channel}")
    print(f"Command: {command}")
    print()

    # Step 1: Send command via Slack control
    print("[1/4] Sending command via Slack control...")
    control_args = [
        sys.executable, "-m", "slack_helpers.cli", "control",
        "--channel", control_channel,
        "--state-file", state_file,
        "--lock-file", lock_file,
        "--transcript-file", transcript_file,
        "--command", command,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(slack_helpers_path)

    try:
        result = subprocess.run(
            control_args,
            cwd=str(slack_helpers_path),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min for initial send
        )
        if result.returncode != 0:
            print(f"ERROR: Failed to send command: {result.stderr}", file=sys.stderr)
            return None
        print("[OK] Command sent")
    except subprocess.TimeoutExpired:
        print("ERROR: Timeout sending command", file=sys.stderr)
        return None

    # Step 2: Poll for completion via Slack download
    print(f"\n[2/4] Polling for completion (every {poll_interval}s, timeout {timeout}s)...")

    start_time = time.time()
    downloaded_file = None
    dest_dir = ROOT / "fak" / "experiments" / "dgx" / "incoming"
    dest_dir.mkdir(parents=True, exist_ok=True)

    while time.time() - start_time < timeout:
        time.sleep(poll_interval)

        # Try to download dgx-summary.json
        download_args = [
            sys.executable, "-m", "slack_helpers.cli", "download",
            "dgx-summary.*\\.json",
            "--dest", str(dest_dir),
            "--force",
        ]

        try:
            result = subprocess.run(
                download_args,
                cwd=str(slack_helpers_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                # Check if file was downloaded
                for f in dest_dir.glob("dgx-summary*.json"):
                    # Check modification time to see if it's new
                    mtime = f.stat().st_mtime
                    if mtime > start_time:
                        downloaded_file = f
                        break
                if downloaded_file:
                    print(f"[OK] Results downloaded: {downloaded_file.name}")
                    break
            else:
                print("  Poll... (no results yet)")

        except subprocess.TimeoutExpired:
            print("  Poll... (download timeout, retrying)")
            continue
    else:
        print(f"ERROR: Timeout after {timeout}s waiting for results", file=sys.stderr)
        return None

    # Step 3: Register the run in the catalog
    print("\n[3/4] Registering run in catalog...")

    run_id = register_dgx_run(downloaded_file.parent, machine_id=machine_id)
    if run_id:
        print(f"[OK] Registered run: {run_id}")
    else:
        print("WARNING: Failed to register run in catalog", file=sys.stderr)

    # Step 4: Cleanup (optional)
    print("\n[4/4] Transfer complete!")
    print(f"  Results: {downloaded_file.parent}")
    print(f"  Catalog: {CATALOG_PATH}")
    print(f"  Run ID: {run_id or 'unknown'}")

    return run_id


def download_and_register(
    pattern: str = "dgx-summary.*\\.json",
    slack_helpers_path: Optional[Path] = None,
    dest_dir: Optional[Path] = None,
    machine_id: str = "dgx",
) -> Optional[str]:
    """Download latest results from Slack and register in catalog.

    Args:
        pattern: Regex pattern to match files for download
        slack_helpers_path: Path to slack-helpers package (auto-detected if None)
        dest_dir: Destination directory for downloads
        machine_id: Machine ID for catalog registration

    Returns:
        Run ID if successful, None otherwise
    """
    # Auto-detect slack-helpers path
    if slack_helpers_path is None:
        for cand in _slack_helpers_candidates():
            if (cand / "slack_helpers").exists():
                slack_helpers_path = cand
                break
        if slack_helpers_path is None:
            print("ERROR: Could not find slack-helpers package", file=sys.stderr)
            print("Set SLACK_HELPERS_PATH or pass --slack-helpers-path", file=sys.stderr)
            return None

    # Default dest directory
    if dest_dir is None:
        dest_dir = ROOT / "fak" / "experiments" / "dgx" / "incoming"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Slack helpers: {slack_helpers_path}")
    print(f"Pattern: {pattern}")
    print(f"Destination: {dest_dir}")
    print()

    # Download file
    print("[1/2] Downloading from Slack...")
    download_args = [
        sys.executable, "-m", "slack_helpers.cli", "download",
        pattern,
        "--dest", str(dest_dir),
        "--force",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(slack_helpers_path)

    try:
        result = subprocess.run(
            download_args,
            cwd=str(slack_helpers_path),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            print(f"ERROR: Download failed: {result.stderr}", file=sys.stderr)
            return None

        print(result.stdout)

    except subprocess.TimeoutExpired:
        print("ERROR: Timeout downloading file", file=sys.stderr)
        return None

    # Find downloaded file
    downloaded_file = None
    for f in dest_dir.glob("dgx-summary*.json"):
        downloaded_file = f
        break

    if not downloaded_file:
        print("ERROR: No file downloaded", file=sys.stderr)
        return None

    print(f"[OK] Downloaded: {downloaded_file.name}")

    # Register in catalog
    print("\n[2/2] Registering in catalog...")
    run_id = register_dgx_run(downloaded_file.parent, machine_id=machine_id)

    if run_id:
        print(f"[OK] Registered run: {run_id}")
    else:
        print("WARNING: Failed to register run in catalog", file=sys.stderr)

    print(f"\nDone! Results: {downloaded_file.parent}")
    return run_id


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", help="Command to run")

    # list
    list_p = sub.add_parser("list", help="List benchmark runs")
    list_p.add_argument("--machine", help="Filter by machine ID")
    list_p.add_argument("--model", help="Filter by model name")
    list_p.add_argument("--precision", help="Filter by precision")
    list_p.add_argument("--format", choices=["table", "summary"], default="table",
                       help="Output format")

    # show
    show_p = sub.add_parser("show", help="Show detailed run info")
    show_p.add_argument("run_id", help="Run ID to show")

    # best
    best_p = sub.add_parser("best", help="Find best run")
    best_p.add_argument("--model", help="Filter by model")

    # summary
    sub.add_parser("summary", help="Summary statistics")

    # status (for !status)
    sub.add_parser("status", help="Quick status for Slack")

    # register-dgx (internal, for DGX scripts)
    register_p = sub.add_parser("register-dgx", help="Register a DGX run (internal)")
    register_p.add_argument("run_dir", type=Path, help="Directory containing DGX run results")
    register_p.add_argument("--machine-id", default="dgx", help="Machine ID")

    # transfer (end-to-end benchmark and results transfer)
    transfer_p = sub.add_parser("transfer", help="End-to-end: run benchmark via Slack and transfer results")
    transfer_p.add_argument("--benchmark-command", help="Benchmark command to run (default: DGX fak test)")
    transfer_p.add_argument("--slack-helpers-path", type=Path, help="Path to slack-helpers package")
    transfer_p.add_argument("--control-channel", default="dgx-control", help="Slack control channel")
    transfer_p.add_argument("--state-file", help="State file path for control bridge")
    transfer_p.add_argument("--poll-interval", type=int, default=30, help="Seconds between polls")
    transfer_p.add_argument("--timeout", type=int, default=3600, help="Maximum seconds to wait")
    transfer_p.add_argument("--machine-id", default="dgx", help="Machine ID for catalog")

    # download-and-register (just the retrieval part)
    dl_p = sub.add_parser("download-and-register", help="Download latest results from Slack and register in catalog")
    dl_p.add_argument("--pattern", default="dgx-summary.*\\.json", help="Regex pattern to match files")
    dl_p.add_argument("--slack-helpers-path", type=Path, help="Path to slack-helpers package")
    dl_p.add_argument("--dest-dir", type=Path, help="Destination directory")
    dl_p.add_argument("--machine-id", default="dgx", help="Machine ID for catalog")

    args = ap.parse_args(argv)

    if not args.command:
        ap.print_help()
        return 1

    # Special handling for register-dgx (doesn't need catalog loaded)
    if args.command == "register-dgx":
        run_id = register_dgx_run(args.run_dir, args.machine_id)
        if run_id:
            print(f"Registered run: {run_id}")
            return 0
        else:
            print("Failed to register run", file=sys.stderr)
            return 1

    # Special handling for transfer (end-to-end process)
    if args.command == "transfer":
        default_command = "bash -lc 'cd /srv/fleet && bash tools/dgx_fak_test.sh'"
        command = args.benchmark_command or default_command

        run_id = run_benchmark_and_transfer(
            command=command,
            slack_helpers_path=args.slack_helpers_path,
            control_channel=args.control_channel,
            state_file=args.state_file,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            machine_id=args.machine_id,
        )

        return 0 if run_id else 1

    # Special handling for download-and-register (retrieval only)
    if args.command == "download-and-register":
        run_id = download_and_register(
            pattern=args.pattern,
            slack_helpers_path=args.slack_helpers_path,
            dest_dir=args.dest_dir,
            machine_id=args.machine_id,
        )

        return 0 if run_id else 1

    catalog = load_catalog()
    if not catalog:
        print("_Catalog not found. Run `python tools/bench_catalog.py build` first_", file=sys.stderr)
        return 1

    if args.command == "list":
        output = cmd_list_slack(args, catalog)
    elif args.command == "show":
        output = cmd_show_slack(args, catalog)
    elif args.command == "best":
        output = cmd_best_slack(args, catalog)
    elif args.command == "summary":
        output = cmd_summary_slack(args, catalog)
    elif args.command == "status":
        output = cmd_status(args, catalog)
    else:
        ap.print_help()
        return 1

    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
