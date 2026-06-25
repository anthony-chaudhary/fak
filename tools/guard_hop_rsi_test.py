#!/usr/bin/env python3
"""Tests for tools/guard_hop_rsi.py (issue #733) — hermetic, no network.

The RSI scaffold's load-bearing property is that it cannot KEEP a candidate without a
re-measured improvement + a witness. These tests pin that: plan mode marks every
candidate PENDING_MEASUREMENT, a MEASURED baseline flips them to READY_TO_MEASURE, and
the honesty gate FAILS any plan that marks a candidate kept without a witness / a
negative measured delta.

Run: `python tools/guard_hop_rsi_test.py`  (exit 0 = all pass),
or `python -m pytest tools/guard_hop_rsi_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard_hop_rsi as rsi  # noqa: E402


def test_plan_mode_is_all_pending_and_honest() -> None:
    plan = rsi.plan_iteration()
    assert plan["schema"] == rsi.SCHEMA
    assert plan["baseline_is_measured"] is False
    assert plan["baseline"]["status"] == "PROJECTED"
    assert plan["deferred"], "plan mode must disclose the deferred measurement"
    assert plan["candidates"], "the search space must be non-empty"
    for c in plan["candidates"]:
        assert c["status"] == "PENDING_MEASUREMENT", c
        assert c["kept"] is False, c
    assert rsi.check_plan(plan) == [], rsi.check_plan(plan)


def test_measured_baseline_flips_candidates_ready() -> None:
    measured = {
        "guard_hop_overhead": {
            "status": "MEASURED", "reps": 20, "direct_p50_ms": 5.0,
            "gateway_p50_ms": 6.2, "guard_hop_overhead_ms": 1.2,
            "disclosure": "single-box",
        }
    }
    plan = rsi.plan_iteration(measured)
    assert plan["baseline_is_measured"] is True
    for c in plan["candidates"]:
        assert c["status"] == "READY_TO_MEASURE", c
    assert rsi.check_plan(plan) == [], rsi.check_plan(plan)


def test_gate_fails_kept_without_witness() -> None:
    plan = rsi.plan_iteration()
    plan["candidates"][0]["kept"] = True
    plan["candidates"][0]["measured_delta_ms"] = -0.3  # an improvement, but no witness
    violations = rsi.check_plan(plan)
    assert any("witness" in v for v in violations), violations


def test_gate_fails_kept_without_improvement() -> None:
    # Simulate a measured baseline so status is valid, then claim a non-improving keep.
    measured = {"guard_hop_overhead": {"status": "MEASURED", "guard_hop_overhead_ms": 1.0,
                                       "disclosure": "x"}}
    plan = rsi.plan_iteration(measured)
    plan["candidates"][0]["kept"] = True
    plan["candidates"][0]["witness"] = "go test ./... green"
    plan["candidates"][0]["measured_delta_ms"] = 0.4  # overhead went UP — must be refused
    violations = rsi.check_plan(plan)
    assert any("not an improvement" in v for v in violations), violations


def test_gate_accepts_a_real_kept_candidate() -> None:
    measured = {"guard_hop_overhead": {"status": "MEASURED", "guard_hop_overhead_ms": 1.0,
                                       "disclosure": "x"}}
    plan = rsi.plan_iteration(measured)
    c = plan["candidates"][0]
    c["kept"] = True
    c["witness"] = "go test ./... green @ <sha>"
    c["measured_delta_ms"] = -0.25
    assert rsi.check_plan(plan) == [], rsi.check_plan(plan)


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    if failures:
        print(f"\n{failures} test(s) failed")
        return 1
    print("\nall guard-hop-rsi tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
