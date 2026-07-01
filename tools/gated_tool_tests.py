#!/usr/bin/env python3
"""Discover + gate EVERY tools/*_test.py so a new tool test can never silently go un-run.

THE BLACKHOLE THIS CLOSES
-------------------------
ci.yml hand-enumerates ~57 of 165 tools/*_test.py across a dozen steps, so a newly
added tools/<x>_test.py ran in NO gate at all. tool_coverage_audit.py is the *static*
complement (it flags a module with no sibling test) but it cannot see that a test
EXISTS yet RUNS nowhere. This is the EXECUTION-gating complement: it discovers every
tools/*_test.py and accounts for each in exactly one bucket --

  * GATED-ELSEWHERE: already invoked by a `python tools/<x>_test.py` line in ci.yml
                     (read live from .github/workflows/ci.yml, so this never drifts).
  * QUARANTINE     : cannot run on the hermetic CI box; an explicit {name: reason} entry
                     whose reason is from a CLOSED vocabulary, so the exclusion is
                     VISIBLE and reviewable -- never a silent hole.
  * HERMETIC       : the default remainder -- pure-stdlib, no network/GPU/external binary.
                     RUN here on every push, aggregating ALL failures (anti-cascade).

Because the default is HERMETIC, a newly added test RUNS automatically. If it can't
(needs the network, a GPU, gh/dos/go, or it's red) the author must either fix it or add
it to QUARANTINE with a reason -- otherwise `--run` fails. That is the no-future-blackhole
forcing function. The classification is Linux-authoritative (CI is ubuntu-latest).

    python tools/gated_tool_tests.py --check   # manifest sanity + accounting (pure fs read)
    python tools/gated_tool_tests.py --run      # run every HERMETIC test, report ALL failures
    python tools/gated_tool_tests.py --all       # also run the gated-elsewhere set (local full run)
    python tools/gated_tool_tests.py --list      # show the partition
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
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

SCHEMA = "fak.gated-tool-tests.v1"

# Closed vocabulary of reasons a test is held OUT of the hermetic gate. A new reason
# string must be added here on purpose -- so the exclusion stays a reviewed decision.
QUARANTINE_REASONS = {
    "needs-network": "reaches the network / a live API (slack, a remote host)",
    "needs-binary": "shells out to a binary/host absent on the hermetic box (ssh, a laptop runner, an agent-setup probe)",
    "needs-gpu": "requires a CUDA/GPU serving node or its run artifacts",
    "pytest": "needs pytest or a third-party import the hermetic box lacks (green only where pytest is installed)",
    "red": "currently failing on the hermetic Linux box; tracked for repair, quarantined so it does not mask other reds",
    "slow": "too heavy for the fast push gate (minutes); runs in a dedicated job",
}

# {test filename: reason}. Everything NOT listed here (and not already gated in ci.yml)
# is HERMETIC and RUNS. Reasons are from QUARANTINE_REASONS. Built from a Linux-
# authoritative classification (run under WSL/ubuntu, the CI environment). Keep sorted.
QUARANTINE: dict[str, str] = {
    # --- needs pytest / a third-party import the hermetic CI box lacks ---
    "claude_account_backup_test.py": "pytest",
    "fleet_compare_test.py": "pytest",  # imports numpy/matplotlib; green only on plotting-equipped hosts
    "fleet_resume_watchdog_test.py": "pytest",
    "gcp_accel_test.py": "pytest",
    "glm52_serve_preflight_test.py": "pytest",
    "leak_scan_test.py": "pytest",
    "resume_watch_test.py": "pytest",
    "session_checkpoint_test.py": "pytest",
    "visual_gen_bench_test.py": "pytest",
    "visual_gen_grade_test.py": "pytest",
    # --- shells out to a binary/host absent on the hermetic box ---
    "extend_preflight_test.py": "needs-binary",
    "fak_laptop_test.py": "needs-binary",
    "fak_laptop_test_test.py": "needs-binary",
    # --- requires a GPU serving node / its run artifacts ---
    "qwen36_perf_gate_test.py": "needs-gpu",
    "qwen36_standalone_readiness_test.py": "needs-gpu",
    # --- currently RED on the hermetic Linux box (tracked for repair) ---
    "api_host_bridge_gate_test.py": "red",
    "api_host_bridge_matrix_test.py": "red",
    "concept_disambiguation_scorecard_test.py": "red",  # live tree has disambiguation coverage debt
    "fleet_bottleneck_test.py": "red",  # reads fleet infra fixtures (grafana dashboards) absent here
    "gen_llms_full_test.py": "red",  # reads the live working tree; non-deterministic in the shared trunk
    "intent_literal_scorecard_test.py": "red",  # live docs scorecard has literal/intent disclosure debt
    "product_scorecard_test.py": "red",  # enforces 100% concept positioning; tree currently at ~90%
    "repo_guard_test.py": "red",  # platform-divergent: green on win32, OUT_OF_TREE_WRITE deny missed on linux
    "steerability_scorecard_test.py": "red",  # live tree has hard steerability scorecard debt
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover(tools_dir: Path) -> list[str]:
    return sorted(p.name for p in tools_dir.glob("*_test.py"))


_CI_INVOKE = re.compile(r"python3?\s+tools/(\w+_test\.py)")


def gated_in_ci(ci_text: str) -> set[str]:
    """Tests already invoked by a `python tools/<x>_test.py` line in ci.yml.

    Read live from the workflow so the runner can never drift from the enumerated
    steps: a test named in a real invocation is GATED-ELSEWHERE (the runner leaves it
    to its own step, no double-run). A bare mention in a comment does NOT match, so a
    commented-out reference correctly falls back into the hermetic set.
    """
    return set(_CI_INVOKE.findall(ci_text))


def partition(all_tests: list[str], gated: set[str]) -> tuple[list[str], dict[str, str], list[str]]:
    gated_here = [t for t in all_tests if t in gated and t not in QUARANTINE]
    quarantined = {t: QUARANTINE[t] for t in all_tests if t in QUARANTINE}
    hermetic = [t for t in all_tests if t not in gated and t not in QUARANTINE]
    return gated_here, quarantined, hermetic


def _read_ci(root: Path) -> str:
    p = root / ".github" / "workflows" / "ci.yml"
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def check(root: Path) -> tuple[int, dict[str, Any]]:
    """Pure filesystem read: assert the manifest is internally consistent vs the tree.

    Fails on (a) a quarantine entry whose file no longer exists (stale), (b) a reason
    outside the closed vocab, or (c) a contradiction: a test both quarantined AND
    invoked by a ci.yml step. Every discovered test is, by construction, in exactly one
    bucket (gated-elsewhere / quarantine / hermetic), so accounting is total -- there is
    no "unclassified" hole. The red-debt count is surfaced as a watched number.
    """
    tools_dir = root / "tools"
    all_tests = discover(tools_dir)
    gated = gated_in_ci(_read_ci(root))
    gated_here, quarantined, hermetic = partition(all_tests, gated)

    problems: list[str] = []
    for name, reason in QUARANTINE.items():
        if not (tools_dir / name).exists():
            problems.append(f"stale quarantine entry (file gone): {name}")
        if reason not in QUARANTINE_REASONS:
            problems.append(f"quarantine reason not in closed vocab: {name} -> {reason!r}")
        if name in gated:
            problems.append(f"contradiction: {name} is quarantined AND invoked by a ci.yml step")
    red_debt = sorted(t for t, r in quarantined.items() if r == "red")
    ok = not problems
    payload = {
        "schema": SCHEMA, "ok": ok, "mode": "check",
        "total": len(all_tests),
        "gated_elsewhere": len(gated_here),
        "quarantined": len(quarantined),
        "hermetic": len(hermetic),
        "red_debt": len(red_debt), "red_debt_tests": red_debt,
        "problems": problems,
    }
    return (0 if ok else 1), payload


def _run_one(root: Path, name: str) -> tuple[str, int, str]:
    path = root / "tools" / name
    try:
        r = subprocess.run([sys.executable, str(path)], cwd=str(root),
                           capture_output=True, text=True, timeout=180,
                           encoding="utf-8", errors="replace")
        rc = r.returncode
    except subprocess.TimeoutExpired:
        return name, 124, "TIMEOUT (>180s) -- quarantine (slow/needs-network) or fix the hang"
    except Exception as e:  # noqa: BLE001
        return name, 1, f"runner error: {e}"
    tail = ((r.stderr or "") + (r.stdout or "")).strip().splitlines()
    return name, rc, (tail[-1][:200] if tail else "")


def run(root: Path, jobs: int, include_gated: bool) -> tuple[int, dict[str, Any]]:
    """Run the HERMETIC set (and, with --all, the gated-elsewhere set too), AGGREGATE,
    and report ALL failures at once -- a failing test never masks the rest."""
    all_tests = discover(root / "tools")
    gated = gated_in_ci(_read_ci(root))
    gated_here, _, hermetic = partition(all_tests, gated)
    targets = sorted(set(hermetic) | (set(gated_here) if include_gated else set()))
    failures = []
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        for name, rc, msg in ex.map(lambda t: _run_one(root, t), targets):
            if rc != 0:
                failures.append({"test": name, "rc": rc, "tail": msg})
    ok = not failures
    payload = {
        "schema": SCHEMA, "ok": ok, "mode": "run",
        "ran": len(targets), "included_gated_elsewhere": include_gated,
        "failed": len(failures),
        "failures": sorted(failures, key=lambda f: f["test"]),
    }
    return (0 if ok else 1), payload


def render(p: dict[str, Any]) -> str:
    if p["mode"] == "check":
        head = (f"gated-tool-tests CHECK: {'OK' if p['ok'] else 'ACTION'} -- {p['total']} tests = "
                f"{p['gated_elsewhere']} gated-elsewhere + {p['quarantined']} quarantined "
                f"+ {p['hermetic']} hermetic  (red-debt {p['red_debt']})")
        lines = [head]
        if p["red_debt_tests"]:
            lines.append("  red-debt (quarantined, awaiting repair): " + ", ".join(p["red_debt_tests"]))
        for prob in p["problems"]:
            lines.append("  PROBLEM: " + prob)
        return "\n".join(lines)
    head = (f"gated-tool-tests RUN: {'OK' if p['ok'] else 'FAIL'} -- ran {p['ran']} "
            f"{'(hermetic+gated)' if p['included_gated_elsewhere'] else 'hermetic'}, {p['failed']} failed")
    if p["failures"]:
        return head + "\n" + "\n".join(
            f"  FAIL {f['test']} [rc={f['rc']}] {f['tail']}" for f in p["failures"])
    return head


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Discover + gate every tools/*_test.py (no-blackhole runner).")
    ap.add_argument("--check", action="store_true", help="manifest sanity + accounting (pure fs read)")
    ap.add_argument("--run", action="store_true", help="run every hermetic test, report all failures")
    ap.add_argument("--all", action="store_true", help="with --run, also run the gated-elsewhere set")
    ap.add_argument("--list", action="store_true", help="show the partition")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(argv)
    root = repo_root()

    if args.list:
        all_tests = discover(root / "tools")
        gated = gated_in_ci(_read_ci(root))
        gated_here, quarantined, hermetic = partition(all_tests, gated)
        print(json.dumps({"gated_elsewhere": gated_here, "quarantine": quarantined,
                          "hermetic": hermetic}, indent=2))
        return 0
    if args.run:
        rc, payload = run(root, args.jobs, include_gated=args.all)
    else:  # default + --check
        rc, payload = check(root)
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
