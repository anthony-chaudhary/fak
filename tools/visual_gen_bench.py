#!/usr/bin/env python3
"""visual_gen_bench.py -- the seam that wires the visual-generation quality grade
to the two existing fleet substrates: the benchmark catalog DB and the RSI
keep-or-revert keep-bit.

One pass does three things, mapping exactly onto the goal's three clauses:

  1. GRADE        run tools/visual_gen_grade.py over the visuals deck -> a 0-1
                  mean score + a per-figure report.

  2. DB (durable) write a benchmark-catalog run dir
                    fak/experiments/benchmark/runs/by-machine/<machine>/<ts>-visualgen/
                  with manifest.json (schema benchmark/run-manifest.v1) + the full
                  report, then fold it into catalog.json via bench_catalog.py build.
                  This is how the grade is DURABLY TRACKED -- every grade is a row in
                  the same catalog the inference benchmarks land in.

  3. RSI (feeds)  read the PREVIOUS visual-gen run's mean score back from the catalog
                  (the durable baseline), then run the candidate's measured before/
                  after through fak's own non-forgeable keep-bit (cmd/rsicycle ->
                  internal/shipgate.Evaluate). The Python side CANNOT fabricate a KEEP:
                  the keep-bit is set only inside the Go shipgate from the measured
                  witness. This is how the BENCHMARK FEEDS RSI.

By operator decision the loop SURFACES, it does not auto-revert: a regression or a
below-floor figure is reported as ACTION for a worker/operator to fix; the keep-bit
is recorded as DB evidence, no file is mutated. That matches every other read-only
gardening loop (issue-closure, memory-recall) and is safe in the shared tree.

Usage:
  python tools/visual_gen_bench.py                 # grade -> DB -> RSI, human summary
  python tools/visual_gen_bench.py --json          # machine-readable (control-pane loop)
  python tools/visual_gen_bench.py --no-db         # grade + RSI only, don't touch the catalog
  python tools/visual_gen_bench.py --no-rsi        # grade + DB only, skip the keep-bit
  python tools/visual_gen_bench.py --floor 0.85    # stricter below-floor threshold
  python tools/visual_gen_bench.py --machine anthony --now 20260620T120000Z

Exit status: 0 on a successful run regardless of score (a regression is ACTION data,
not a crash -- the loop-audit contract); 2 only when the grader itself cannot run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
install_no_window_subprocess_defaults(subprocess)
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import visual_gen_grade as grade  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAK = ROOT / "fak"
BENCHMARK_DIR = FAK / "experiments" / "benchmark"
RUNS_DIR = BENCHMARK_DIR / "runs" / "by-machine"
BENCH_CATALOG = ROOT / "tools" / "bench_catalog.py"
RSICYCLE_DIR = FAK  # `go run ./cmd/rsicycle` is invoked with cwd=fak/
SCHEMA = "fleet-visual-gen-bench/1"
RUN_MANIFEST_SCHEMA = "benchmark/run-manifest.v1"
METRIC = "visual_gen_mean_score"
REPORT_NAME = "visualgen-report.json"


# --------------------------------------------------------------------------------
# git context (recorded in the manifest, like the inference run manifests)
# --------------------------------------------------------------------------------

def _git(args: list[str]) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                             text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def git_context() -> dict[str, Any]:
    rev = _git(["rev-parse", "HEAD"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    dirty = bool(_git(["status", "--porcelain"]).strip())
    return {"rev": rev, "branch": branch, "dirty": dirty}


# --------------------------------------------------------------------------------
# clause 2: the benchmark DB
# --------------------------------------------------------------------------------

def previous_mean_score(machine: str) -> float | None:
    """The most recent prior visual-gen run's mean score for this machine, read back
    from the durable benchmark runs on disk. This is the RSI baseline -- the `before`
    the candidate must strictly beat. None when no prior run exists (first grade)."""
    machine_dir = RUNS_DIR / machine
    if not machine_dir.is_dir():
        return None
    candidates: list[tuple[str, float]] = []
    for run_dir in machine_dir.glob("*-visualgen"):
        report = run_dir / REPORT_NAME
        manifest = run_dir / "manifest.json"
        if not report.exists() or not manifest.exists():
            continue
        rep = grade.load_json(report)
        man = grade.load_json(manifest)
        if not isinstance(rep, dict) or not isinstance(man, dict):
            continue
        ts = str(man.get("timestamp") or run_dir.name)
        mean = rep.get("aggregate", {}).get("mean_score")
        if isinstance(mean, (int, float)):
            candidates.append((ts, float(mean)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def write_run(machine: str, now: str, report: dict, git: dict, rsi: dict | None,
              floor: float) -> Path:
    """Write the run dir + manifest + report; return the run dir path."""
    run_id = f"{machine}-visualgen-{now}"
    run_dir = RUNS_DIR / machine / f"{now}-visualgen"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    agg = report["aggregate"]
    manifest = {
        "$schema": RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "machine_id": machine,
        "timestamp": now,
        "harness": {"name": "visual-gen-grade", "version": "1"},
        "model": {"name": "visual-gen-deck", "precision": "n/a"},
        "config": {"floor": floor, "metric": METRIC},
        "git": git,
        "tags": ["visual-gen", "experiment", "rsi"],
        "artifacts": {"report": REPORT_NAME},
        "summary": {
            "mean_score": agg["mean_score"],
            "n": agg["n"],
            "n_below_floor": agg["n_below_floor"],
        },
    }
    if rsi is not None:
        manifest["rsi"] = rsi  # the keep-bit evidence, recorded but not acted on
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return run_dir


def fold_into_catalog() -> str | None:
    """Rebuild catalog.json so the new run is indexed. Returns an error string on
    failure (the run dir is still written -- the catalog fold is best-effort)."""
    if not BENCH_CATALOG.exists():
        return "bench_catalog.py not found"
    try:
        proc = subprocess.run(
            [sys.executable, str(BENCH_CATALOG), "build"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return f"catalog build failed to start: {e}"
    if proc.returncode != 0:
        return f"catalog build exited {proc.returncode}: {proc.stderr.strip()[:200]}"
    return None


# --------------------------------------------------------------------------------
# clause 1: feed the RSI keep-bit (cmd/rsicycle -> internal/shipgate)
# --------------------------------------------------------------------------------

def truth_clean() -> bool:
    """Best-effort truth syscall: is HEAD's claim diff-witnessed? Uses the dos
    commit-audit CLI if present. Missing/unrunnable dos -> conservatively False (we
    do not assert a clean truth we could not check)."""
    try:
        proc = subprocess.run(
            ["dos", "commit-audit", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    if proc.returncode != 0:
        return False
    try:
        rows = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return False
    if isinstance(rows, list) and rows:
        return all(r.get("verdict") == "OK" for r in rows if isinstance(r, dict))
    return False


def run_keep_bit(before: float, after: float, suite_green: bool,
                 truth_ok: bool) -> dict[str, Any]:
    """Run the candidate's measured witness through fak's non-forgeable keep-bit via
    `go run ./cmd/rsicycle`. The decision is the Go shipgate's, never the Python
    caller's. Returns the parsed decision + the witness it was computed from. If go
    is unavailable, the decision is reported as UNAVAILABLE (the keep-bit is never
    FABRICATED in Python)."""
    witness = {
        "metric": METRIC, "before": before, "after": after,
        "lower_better": False, "suite_green": suite_green, "truth_clean": truth_ok,
    }
    cmd = [
        "go", "run", "./cmd/rsicycle",
        "-metric", METRIC,
        "-before", f"{before}", "-after", f"{after}",
        "-lower-better=false",
        f"-suite-green={str(suite_green).lower()}",
        f"-truth-clean={str(truth_ok).lower()}",
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(RSICYCLE_DIR), capture_output=True,
                              text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        return {"decision": "UNAVAILABLE", "kept": False, "witness": witness,
                "note": f"rsicycle could not run: {e}"}
    # rsicycle prints "DECISION=KEEP kept=true"; exit 0 KEEP, 3 REVERT.
    decision = "UNAVAILABLE"
    kept = False
    for line in proc.stdout.splitlines():
        if line.startswith("DECISION="):
            parts = line.split()
            decision = parts[0].split("=", 1)[1]
            kept = parts[1].split("=", 1)[1] == "true" if len(parts) > 1 else False
    if decision == "UNAVAILABLE" and proc.returncode in (0, 3):
        decision = "KEEP" if proc.returncode == 0 else "REVERT"
        kept = proc.returncode == 0
    return {"decision": decision, "kept": kept, "witness": witness,
            "exit_code": proc.returncode}


def suite_green() -> bool:
    """The grader's own test suite is the SuiteGreen witness. Run it headlessly."""
    test = ROOT / "tools" / "visual_gen_grade_test.py"
    if not test.exists():
        return False
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(test), "-q"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


# --------------------------------------------------------------------------------
# the one pass
# --------------------------------------------------------------------------------

def run_bench(*, machine: str, now: str, floor: float, do_db: bool, do_rsi: bool,
              do_render: bool, do_suite: bool) -> dict[str, Any]:
    visuals_dir = grade.DEFAULT_DIR
    report = grade.grade_deck(visuals_dir, floor, render=do_render)
    agg = report["aggregate"]
    after = float(agg["mean_score"])

    baseline = previous_mean_score(machine)
    before = baseline if baseline is not None else after  # first run: no strict gain

    rsi: dict[str, Any] | None = None
    if do_rsi:
        green = suite_green() if do_suite else False
        truth = truth_clean()
        rsi = run_keep_bit(before, after, green, truth)
        rsi["baseline"] = baseline
        rsi["first_run"] = baseline is None

    run_dir = None
    catalog_note = None
    if do_db:
        run_dir = write_run(machine, now, report, git_context(), rsi, floor)
        catalog_note = fold_into_catalog()

    # ACTION when a figure is below floor OR the deck score regressed vs the durable
    # baseline. A healthy, non-regressing grade is OK (the loop-audit contract).
    regressed = baseline is not None and after < baseline - 1e-9
    below = agg["n_below_floor"] > 0
    ok = not (regressed or below)
    if regressed:
        reason = (f"visual-gen score regressed {baseline:.3f} -> {after:.3f} "
                  f"(ACTION: a figure got worse since the last DB run)")
    elif below:
        reason = (f"{agg['n_below_floor']}/{agg['n']} figure(s) below the "
                  f"{floor:.2f} render-quality floor (ACTION: re-render or fix source); "
                  f"mean={after:.3f}")
    else:
        base_txt = f"baseline {baseline:.3f}" if baseline is not None else "first run"
        reason = f"visual-gen deck healthy: mean={after:.3f} ({base_txt}), 0 below floor"

    out: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": ok,
        "reason": reason,
        "mean_score": after,
        "baseline": baseline,
        "regressed": regressed,
        "n": agg["n"],
        "n_below_floor": agg["n_below_floor"],
        "below_floor": agg["below_floor"],
        "floor": floor,
        "machine": machine,
        "timestamp": now,
    }
    if rsi is not None:
        out["rsi"] = {"decision": rsi["decision"], "kept": rsi["kept"],
                      "before": before, "after": after}
    if run_dir is not None:
        try:
            out["run_dir"] = str(run_dir.relative_to(ROOT))
        except ValueError:
            out["run_dir"] = str(run_dir)
    if catalog_note is not None:
        out["catalog_note"] = catalog_note
    return out


def detect_machine() -> str:
    """Default machine id: the first machine dir under runs/by-machine, else the
    local hostname's short form. Keeps grades landing under one stable id."""
    if RUNS_DIR.is_dir():
        existing = sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir())
        if existing:
            return existing[0]
    import socket
    return socket.gethostname().split(".")[0].lower() or "local"


def utc_stamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grade the visuals deck -> DB -> RSI keep-bit.")
    ap.add_argument("--machine", default=None, help="machine id (default: auto-detect)")
    ap.add_argument("--now", default=None, help="UTC timestamp YYYYmmddTHHMMSSZ (default: now)")
    ap.add_argument("--floor", type=float, default=grade.DEFAULT_FLOOR)
    ap.add_argument("--no-db", action="store_true", help="do not write/fold the catalog")
    ap.add_argument("--no-rsi", action="store_true", help="skip the keep-bit")
    ap.add_argument("--no-suite", action="store_true",
                    help="do not run the grader self-test (SuiteGreen witness -> False)")
    ap.add_argument("--render", action="store_true", help="re-render .mmd->.svg first")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not grade.DEFAULT_DIR.is_dir():
        print(f"[ERROR] visuals dir not found: {grade.DEFAULT_DIR}", file=sys.stderr)
        return 2

    machine = args.machine or detect_machine()
    now = args.now or utc_stamp()
    out = run_bench(
        machine=machine, now=now, floor=args.floor,
        do_db=not args.no_db, do_rsi=not args.no_rsi,
        do_render=args.render, do_suite=not args.no_suite,
    )

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        status = "OK" if out["ok"] else "ACTION"
        print(f"[{status}] {out['reason']}")
        if "rsi" in out:
            r = out["rsi"]
            print(f"  RSI keep-bit: {r['decision']} (kept={r['kept']}) "
                  f"before={r['before']:.3f} after={r['after']:.3f}")
        if "run_dir" in out:
            print(f"  DB run: {out['run_dir']}")
        if out.get("catalog_note"):
            print(f"  catalog: {out['catalog_note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
