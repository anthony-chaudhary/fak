#!/usr/bin/env python3
"""guard_hop_rsi.py — the RSI loop that drives guard-hop overhead toward 0 (issue #733).

This is the self-improvement loop that sits ON TOP of the #734 measurement harness
(`tools/guard_hop_bench.py`): read the current guard-hop overhead from live dogfood-fleet
telemetry, propose ONE overhead-reducing change, re-measure, and KEEP it only if the
overhead strictly drops AND a witness the loop did not author confirms the suite is still
green. Otherwise REVERT. It is the guard-hop analogue of the DOS enforcement-tuning loop:
the keep/revert decision is grounded in a re-measured number + an external witness, never
in the loop's own say-so.

DEPENDENCY. The keep/revert rung needs a MEASURED baseline (`guard_hop_bench measure`
against a live gateway). Until that measurement is live (hardware-gated, see #734), this
loop runs in **plan mode**: it loads the PROJECTED baseline, enumerates the candidate
levers, and emits the iteration PLAN with every candidate `status: PENDING_MEASUREMENT`.
No candidate can be marked `kept` without a witness — that is the honesty gate (`--check`),
so the scaffold can't fabricate an improvement it never measured.

Usage:
  python tools/guard_hop_rsi.py plan                 # the iteration plan (PROJECTED baseline)
  python tools/guard_hop_rsi.py plan --json
  python tools/guard_hop_rsi.py plan --measured row.json   # plan against a MEASURED baseline
  python tools/guard_hop_rsi.py --check plan.json    # honesty gate over an emitted plan
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard_hop_bench as ghb  # noqa: E402  (the #734 measurement source)

SCHEMA = "guard-hop-rsi/1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_telemetry(root: Path | None = None) -> dict[str, Any]:
    """Resolve the loop's telemetry source to the REAL journals on disk, not a dead
    string. The latency keep/revert rung stays hardware-gated (#734) - a verdict journal
    holds verdicts, not wall-clock - but the loop can still WITNESS that guarded sessions
    ran by reading the same journals dogfood_coverage discovers. This closes the gap where
    `telemetry_source` named `.dispatch-runs/guard-audit/*.jsonl` but no code ever read it;
    the hardware-free verdict-quality RSI loop that DOES close on these rows is
    tools/guard_verdict_rsi.py."""
    root = root or _repo_root()
    pattern = ".dispatch-runs/guard-audit/*.jsonl"
    try:
        spec = importlib.util.spec_from_file_location(
            "dogfood_coverage", root / "tools" / "dogfood_coverage.py")
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str((root / "tools")))
            spec.loader.exec_module(mod)
            rows, journals = mod.count_audit_rows(root)
            return {"source": pattern, "rows": rows, "journals": journals,
                    "verdict_loop": "tools/guard_verdict_rsi.py",
                    "note": "the latency keep/revert rung is hardware-gated (#734); the "
                            "hardware-free verdict-quality loop closes on these rows"}
    except Exception:  # noqa: BLE001 - a missing sibling must not crash plan mode
        pass
    return {"source": pattern, "rows": 0, "journals": 0,
            "verdict_loop": "tools/guard_verdict_rsi.py"}

# The candidate levers the loop sweeps, worst-overhead-first. Each is a HYPOTHESIS about
# where the guard-hop cost lives + the mechanism that would shrink it. The loop applies
# ONE per iteration, re-measures, and keeps only on a strictly-lower overhead + green
# witness. These are the search space, not claims — none is "done" until measured.
CANDIDATES: list[dict[str, str]] = [
    {
        "id": "verdict-memoize",
        "lever": "memoize the Decide verdict for an identical (tool, args, worldVer)",
        "hypothesis": "agent loops re-propose the same tool call; a per-hop verdict cache "
                      "turns the 2nd..Nth Decide into an O(1) lookup.",
        "mechanism": "an LRU keyed on the adjudication inputs, invalidated on policy/world "
                     "epoch change (reuses the cachemeta epoch already in the gateway).",
    },
    {
        "id": "journal-batch-fsync",
        "lever": "batch the decision-journal fsync instead of one fsync per verdict",
        "hypothesis": "the hash-chained journal's per-row durability fsync dominates the "
                      "hop on a busy worker.",
        "mechanism": "group-commit the journal on a short timer / N-row threshold; the "
                     "sha256 chain is unaffected (ordering preserved).",
    },
    {
        "id": "argpredicate-precompile",
        "lever": "precompile ArgPredicates once per policy load, not per Decide",
        "hypothesis": "the 362 ns -> 605 ns Decide growth under 2000 ArgPredicates "
                      "(committed bench) is re-derivation the loop can hoist.",
        "mechanism": "compile the predicate set into a matcher at policy-load time; Decide "
                     "becomes a lookup over the prebuilt matcher.",
    },
    {
        "id": "inproc-colocate",
        "lever": "keep adjudication in-process (already the win vs spawned fak hook)",
        "hypothesis": "the ~2,849x in-process-vs-spawned tax (committed) is the single "
                      "biggest lever; regressing off it is the thing to guard against.",
        "mechanism": "a sentinel candidate: the loop asserts the in-process path is held, "
                     "so an optimization elsewhere can't silently reintroduce a spawn.",
    },
]


def load_baseline(measured_row: dict[str, Any] | None = None) -> dict[str, Any]:
    """The overhead baseline the loop tries to beat. A MEASURED row (from
    `guard_hop_bench measure`) when supplied; otherwise the PROJECTED row."""
    if measured_row is not None:
        return measured_row["guard_hop_overhead"] if "guard_hop_overhead" in measured_row else measured_row
    return ghb.build_row()["guard_hop_overhead"]


def plan_iteration(measured_row: dict[str, Any] | None = None) -> dict[str, Any]:
    """Emit the RSI iteration plan: the baseline + each candidate with its keep/revert
    rule and the witness required. In plan mode (no MEASURED baseline) every candidate is
    PENDING_MEASUREMENT — the loop cannot keep anything without a re-measured number."""
    baseline = load_baseline(measured_row)
    have_measured = baseline.get("status") == "MEASURED"
    candidates = []
    for c in CANDIDATES:
        candidates.append({
            **c,
            "status": "READY_TO_MEASURE" if have_measured else "PENDING_MEASUREMENT",
            "kept": False,            # never true without a witness (enforced by --check)
            "measured_delta_ms": None,
            "witness": None,          # must be filled with a suite-green witness to keep
        })
    return {
        "schema": SCHEMA,
        "goal": "drive guard-hop overhead toward 0",
        "baseline": baseline,
        "baseline_is_measured": have_measured,
        "keep_revert_rule": "KEEP a candidate iff re-measured overhead is strictly lower "
                            "AND the witness (go test ./... green) confirms no regression; "
                            "else REVERT. One candidate per iteration, worst-first.",
        "measurement_source": "tools/guard_hop_bench.py measure (live gateway, #734)",
        "telemetry_source": ".dispatch-runs/guard-audit/*.jsonl (live dogfood-fleet, #729)",
        "telemetry": resolve_telemetry(),  # the resolved real journals (was a dead string)
        "candidates": candidates,
        "deferred": None if have_measured else
                    "the keep/revert rung needs a MEASURED baseline (hardware-gated, #734); "
                    "this is the scaffold + search space until then.",
    }


def check_plan(plan: dict[str, Any]) -> list[str]:
    """Honesty gate: a candidate may be `kept` ONLY with a witness AND a measured delta;
    a kept candidate's delta must be strictly negative (an actual improvement). This stops
    the loop fabricating an unmeasured/unwitnessed win."""
    violations: list[str] = []
    if plan.get("schema") != SCHEMA:
        violations.append(f"schema must be {SCHEMA!r}, got {plan.get('schema')!r}")
    for c in plan.get("candidates", []):
        cid = c.get("id", "?")
        if c.get("kept"):
            if not c.get("witness"):
                violations.append(f"{cid}: kept=true with no witness")
            delta = c.get("measured_delta_ms")
            if delta is None:
                violations.append(f"{cid}: kept=true with no measured_delta_ms")
            elif delta >= 0:
                violations.append(f"{cid}: kept=true but measured_delta_ms={delta} is not an improvement")
        if not plan.get("baseline_is_measured") and c.get("status") != "PENDING_MEASUREMENT":
            violations.append(f"{cid}: status must be PENDING_MEASUREMENT until the baseline is measured")
    return violations


def render(plan: dict[str, Any]) -> str:
    b = plan["baseline"]
    lines = [f"guard-hop-rsi: {plan['goal']}  (baseline {b.get('status')})"]
    if plan.get("deferred"):
        lines.append(f"  deferred: {plan['deferred']}")
    for c in plan["candidates"]:
        lines.append(f"  [{c['status']:<20}] {c['id']:<22} {c['lever']}")
    lines.append(f"  rule: {plan['keep_revert_rule']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RSI loop scaffold to drive guard-hop overhead toward 0.")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("plan", help="emit the iteration plan")
    p.add_argument("--measured", default="", help="a MEASURED guard_hop_bench row JSON to use as baseline")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default="")
    ap.add_argument("--check", metavar="PLAN.json", default="",
                    help="honesty-gate an emitted plan (exit 1 on any violation)")
    args = ap.parse_args(argv)

    if args.check:
        plan = json.loads(Path(args.check).read_text(encoding="utf-8"))
        violations = check_plan(plan)
        if violations:
            print("guard-hop-rsi --check: FAIL")
            for v in violations:
                print(f"  - {v}")
            return 1
        print("guard-hop-rsi --check: OK (plan is honest)")
        return 0

    measured_row = None
    if getattr(args, "measured", ""):
        measured_row = json.loads(Path(args.measured).read_text(encoding="utf-8"))
    plan = plan_iteration(measured_row)
    out = json.dumps(plan, indent=2)
    if getattr(args, "out", ""):
        Path(args.out).write_text(out + "\n", encoding="utf-8")
    if getattr(args, "json", False):
        print(out)
    else:
        print(render(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
