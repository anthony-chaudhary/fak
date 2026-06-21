#!/usr/bin/env python3
r"""Benchmark-as-keep-bit: the dispatcher's witness is a BENCHMARK number, not the
unit-test suite.

The RSI loop substrate (.claude/rsi-loop-dod.md) gates KEEP on a fact the loop did
not author — historically ``suite_passed`` (the test runner's exit code). The
operator's refinement for the issue dispatcher: *run benchmarks to test*. So for a
code lane, the keep-bit is "did the change keep the kernel's measured throughput
within tolerance of a recorded baseline", read straight from ``go test -bench`` —
an env-authored number (the Go benchmark runner measures it; the agent cannot
narrate it). Deterministic kernel benches reproduce to the ns on a fixed box, so a
regression is real signal, not noise.

Verdict (mirrors ``dos improve`` exit codes so it drops into the same branching):
  KEEP         (exit 0)  measured ns/op <= baseline * (1 + tolerance)   [or NO_BASELINE just recorded]
  REVERT       (exit 3)  measured ns/op regressed past tolerance
  NO_BENCH     (exit 4)  this lane has no benchmark (a doc/tools lane) — caller
                         falls back to the git-ancestry witness (dos verify /
                         issue_closure_audit), which IS the right witness there
  ERROR        (exit 5)  the benchmark could not build/run (fail-safe: never a KEEP)

A lane maps to its Go package by convention (dos.toml [lanes.trees]:
``adjudicator -> fak/internal/adjudicator``), so ``--lane adjudicator`` benches
``./internal/adjudicator``. ``--smoke`` runs the cheapest load-bearing one
(adjudicator BenchmarkDecide, ~270 ns/op, seconds). ``--package`` overrides.

    python tools/bench_witness.py --smoke --record               # set baseline
    python tools/bench_witness.py --lane adjudicator --baseline-file fak/.bench-baseline.json
    python tools/bench_witness.py --lane docs                    # -> NO_BENCH (use dos verify)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fleet-bench-witness/1"

KEEP, REVERT, NO_BENCH, ERROR = "KEEP", "REVERT", "NO_BENCH", "ERROR"
EXIT = {KEEP: 0, REVERT: 3, NO_BENCH: 4, ERROR: 5}

# The cheapest reproducible smoke witness: the adjudicator decide-path bench.
SMOKE_PACKAGE = "./internal/adjudicator"
SMOKE_FILTER = "BenchmarkDecide"
DEFAULT_TOLERANCE_PCT = 25.0  # ns/op carries some run-to-run variance; 25% is a real regression
DEFAULT_BASELINE_FILE = "fak/.bench-baseline.json"

# Lines like: BenchmarkDecide-32   4617806   267.5 ns/op   256 B/op   5 allocs/op
_BENCH_RE = re.compile(
    r"^(?P<name>Benchmark\S+?)(?:-\d+)?\s+\d+\s+(?P<ns>[\d.]+)\s+ns/op")


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def lane_package(lane: str, fak_dir: Path) -> str | None:
    """Map a lane to its Go package dir, by the dos.toml [lanes.trees] convention
    (lane name == internal/ package). Returns a ``./...`` package path or None if
    no Go benchmark could live there (a doc/tools/non-code lane)."""
    cand = fak_dir / "internal" / lane
    if (cand).is_dir() and any(cand.glob("*_test.go")):
        return f"./internal/{lane}"
    return None


def run_bench(package: str, fak_dir: Path, *, count: int, bench_filter: str,
              timeout: int) -> dict[str, Any]:
    """Run ``go test -bench`` for one package; parse ns/op per benchmark (mean)."""
    cmd = ["go", "test", "-run=^$", f"-bench={bench_filter}", "-benchmem",
           f"-count={count}", package]
    try:
        proc = subprocess.run(cmd, cwd=fak_dir, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"benchmark timed out after {timeout}s", "cmd": cmd}
    except OSError as exc:
        return {"error": str(exc), "cmd": cmd}
    out = proc.stdout + "\n" + proc.stderr
    samples: dict[str, list[float]] = {}
    for line in out.splitlines():
        m = _BENCH_RE.match(line.strip())
        if m:
            samples.setdefault(m.group("name"), []).append(float(m.group("ns")))
    if proc.returncode != 0 and not samples:
        return {"error": (proc.stderr or proc.stdout or "go test failed").strip()[-600:],
                "cmd": cmd, "returncode": proc.returncode}
    metrics = {name: round(sum(v) / len(v), 2) for name, v in samples.items()}
    return {"cmd": cmd, "returncode": proc.returncode, "metrics": metrics,
            "primary": _primary_metric(metrics, bench_filter)}


def _primary_metric(metrics: dict[str, float], bench_filter: str) -> dict[str, Any] | None:
    if not metrics:
        return None
    # Prefer a benchmark whose name matches the filter exactly-ish; else the fastest.
    for name, ns in metrics.items():
        if bench_filter and bench_filter.lstrip("^").rstrip("$") in name:
            return {"name": name, "ns_per_op": ns}
    name = min(metrics, key=lambda k: metrics[k])
    return {"name": name, "ns_per_op": metrics[name]}


def load_baseline(path: Path, key: str) -> float | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    entry = (doc.get("baselines") or {}).get(key)
    if isinstance(entry, dict):
        try:
            return float(entry.get("ns_per_op"))
        except (TypeError, ValueError):
            return None
    return None


def save_baseline(path: Path, key: str, ns: float, name: str) -> None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            doc = {}
    except (OSError, ValueError):
        doc = {}
    doc.setdefault("schema", "fleet-bench-baseline/1")
    doc.setdefault("baselines", {})
    doc["baselines"][key] = {"ns_per_op": ns, "bench": name}
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def evaluate(root: Path, *, lane: str | None, package: str | None, smoke: bool,
             count: int, tolerance_pct: float, baseline_ns: float | None,
             baseline_file: str, record: bool, timeout: int) -> dict[str, Any]:
    fak = root / "fak"
    bench_filter = "."
    if smoke:
        package, bench_filter = SMOKE_PACKAGE, SMOKE_FILTER
    elif lane and not package:
        package = lane_package(lane, fak)
        if package is None:
            return {"schema": SCHEMA, "ok": True, "verdict": NO_BENCH,
                    "lane": lane,
                    "reason": (f"lane '{lane}' has no Go benchmark package "
                               f"(fak/internal/{lane}); witness this lane with "
                               "dos verify / issue_closure_audit (git ancestry), "
                               "which is the correct witness for doc/tooling work")}
    if not package:
        return {"schema": SCHEMA, "ok": False, "verdict": ERROR,
                "reason": "no package resolved; pass --lane <code-lane>, --package, or --smoke"}

    key = f"{package}::{bench_filter}"
    bres = run_bench(package, fak, count=count, bench_filter=bench_filter, timeout=timeout)
    if bres.get("error") or not bres.get("primary"):
        return {"schema": SCHEMA, "ok": False, "verdict": ERROR, "lane": lane,
                "package": package, "reason": bres.get("error") or "no benchmark samples parsed",
                "cmd": bres.get("cmd")}

    primary = bres["primary"]
    measured = float(primary["ns_per_op"])
    bfile = (root / baseline_file)
    base = baseline_ns if baseline_ns is not None else load_baseline(bfile, key)

    if base is None:
        if record:
            save_baseline(bfile, key, measured, primary["name"])
        return {"schema": SCHEMA, "ok": True, "verdict": KEEP,
                "lane": lane, "package": package, "bench": primary["name"],
                "measured_ns_per_op": measured, "baseline_ns_per_op": None,
                "tolerance_pct": tolerance_pct, "regression_pct": None,
                "baseline_recorded": bool(record), "baseline_file": str(bfile),
                "reason": (f"no baseline for {key}; measured {measured} ns/op"
                           + (" and recorded it as the baseline" if record
                              else " (pass --record to set it as the keep floor)")),
                "metrics": bres.get("metrics")}

    regression_pct = round((measured - base) / base * 100.0, 2)
    ceiling = base * (1.0 + tolerance_pct / 100.0)
    keep = measured <= ceiling
    verdict = KEEP if keep else REVERT
    if record and keep:
        save_baseline(bfile, key, measured, primary["name"])
    return {"schema": SCHEMA, "ok": keep, "verdict": verdict,
            "lane": lane, "package": package, "bench": primary["name"],
            "measured_ns_per_op": measured, "baseline_ns_per_op": base,
            "tolerance_pct": tolerance_pct, "regression_pct": regression_pct,
            "ceiling_ns_per_op": round(ceiling, 2), "baseline_file": str(bfile),
            "reason": (f"{primary['name']}: {measured} ns/op vs baseline {base} ns/op "
                       f"({regression_pct:+.1f}%, tolerance {tolerance_pct:.0f}%) -> {verdict}"),
            "metrics": bres.get("metrics")}


def render(p: dict[str, Any]) -> str:
    head = f"bench-witness: {p.get('verdict')} ({'keep' if p.get('ok') else 'no-keep'})"
    return head + "\n  " + str(p.get("reason"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark-as-keep-bit witness for the dispatch loop.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--lane", default=None, help="lane name (maps to fak/internal/<lane>)")
    ap.add_argument("--package", default=None, help="explicit go package (e.g. ./internal/ctxmmu)")
    ap.add_argument("--smoke", action="store_true",
                    help="cheapest witness: adjudicator BenchmarkDecide (~seconds)")
    ap.add_argument("--count", type=int, default=6, help="go test -count (default: 6)")
    ap.add_argument("--tolerance-pct", type=float, default=DEFAULT_TOLERANCE_PCT,
                    help=f"regression tolerance %% (default: {DEFAULT_TOLERANCE_PCT})")
    ap.add_argument("--baseline-ns", type=float, default=None,
                    help="explicit baseline ns/op (overrides the baseline file)")
    ap.add_argument("--baseline-file", default=DEFAULT_BASELINE_FILE,
                    help=f"baseline JSON (default: {DEFAULT_BASELINE_FILE})")
    ap.add_argument("--record", action="store_true",
                    help="record the measurement as the baseline (on KEEP / no-baseline)")
    ap.add_argument("--timeout-s", type=int, default=300, help="go test timeout (default: 300)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = evaluate(root, lane=args.lane, package=args.package, smoke=args.smoke,
                       count=args.count, tolerance_pct=args.tolerance_pct,
                       baseline_ns=args.baseline_ns, baseline_file=args.baseline_file,
                       record=args.record, timeout=args.timeout_s)
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return EXIT.get(str(payload.get("verdict")), 5)


if __name__ == "__main__":
    raise SystemExit(main())
