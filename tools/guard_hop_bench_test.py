#!/usr/bin/env python3
"""Tests for tools/guard_hop_bench.py (issue #734) — hermetic, no network.

Covers the pure projection geometry, the emitted row STRUCTURE, and — the load-bearing
part — the honesty gate `check_row`: it must PASS an honest PROJECTED/PENDING row and
FAIL every dishonest shape (a PENDING arm smuggling a number, a MEASURED arm with no
single-box disclosure, a PROJECTED arm with no committed basis). That gate is what
stops the harness fabricating a "measured" guard-hop number it never measured.

Run: `python tools/guard_hop_bench_test.py`  (exit 0 = all pass),
or `python -m pytest tools/guard_hop_bench_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard_hop_bench as ghb  # noqa: E402


def test_projection_scales_with_calls_and_turns() -> None:
    p = ghb.project_overhead(calls_per_turn=8, turns=50)
    assert p["status"] == "PROJECTED"
    # floor uses 362 ns/call: 362*8/1000 = 2.896 µs/turn.
    assert abs(p["per_turn_overhead_us"]["floor"] - 2.896) < 1e-6, p
    # ceil uses 2427 ns/call: 2427*8/1000 = 19.416 µs/turn.
    assert abs(p["per_turn_overhead_us"]["ceil"] - 19.416) < 1e-6, p
    # per-session scales by turns; ceil floor ordering holds.
    assert p["per_session_overhead_ms"]["ceil"] > p["per_session_overhead_ms"]["floor"]
    assert p["basis"]["commit"] and p["basis"]["artifact"]


def test_projection_rejects_negative() -> None:
    try:
        ghb.project_overhead(-1, 10)
    except ValueError:
        return
    raise AssertionError("expected ValueError on negative calls_per_turn")


def test_describe_row_is_honest() -> None:
    row = ghb.build_row(calls_per_turn=8, turns=50)
    assert row["schema"] == ghb.SCHEMA
    assert row["guard_hop_overhead"]["status"] == "PROJECTED"
    assert row["prompt_cache_preservation"]["status"] == "PENDING"
    assert ghb.check_row(row) == [], ghb.check_row(row)


def test_measured_row_is_honest() -> None:
    measured = {
        "status": "MEASURED", "reps": 20,
        "direct_p50_ms": 5.0, "gateway_p50_ms": 6.2,
        "guard_hop_overhead_ms": 1.2,
        "disclosure": "single-box wall-clock on the measuring host; not a fleet SLA.",
    }
    row = ghb.build_row(measured=measured)
    assert row["guard_hop_overhead"]["status"] == "MEASURED"
    assert ghb.check_row(row) == [], ghb.check_row(row)


def test_gate_fails_pending_arm_with_a_number() -> None:
    row = ghb.build_row()
    # Smuggle a number into the PENDING cache arm — the dishonesty the gate exists to catch.
    row["prompt_cache_preservation"]["guard_hop_overhead_ms"] = 0.0
    violations = ghb.check_row(row)
    assert any("PENDING" in v for v in violations), violations


def test_gate_fails_measured_without_disclosure() -> None:
    measured = {"status": "MEASURED", "reps": 20, "direct_p50_ms": 5.0,
                "gateway_p50_ms": 6.2, "guard_hop_overhead_ms": 1.2}  # no disclosure
    row = ghb.build_row(measured=measured)
    violations = ghb.check_row(row)
    assert any("disclosure" in v for v in violations), violations


def test_gate_fails_projected_without_basis() -> None:
    row = ghb.build_row()
    row["guard_hop_overhead"].pop("basis", None)
    violations = ghb.check_row(row)
    assert any("basis" in v.lower() for v in violations), violations


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
    print("\nall guard-hop-bench tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
