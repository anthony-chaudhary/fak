#!/usr/bin/env python3
"""Unit + smoke tests for tools/garden_bundle.py — the garden bundle fold.

Pure-fixture tests over the interpret/fold/gate core (no subprocess), plus one
fast live smoke that proves the runner emits a well-formed envelope. The smoke
uses FAK_GARDEN=off so it does NOT run the slow members (the scorecard family +
loop-audit) — it asserts the skip path + envelope shape, which is enough for a
CI-fast regression sentinel; the heavy members have their own tests.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import garden_bundle as gb  # noqa: E402


def _member(key, gates, kind):
    return {"key": key, "label": key, "gates": gates, "kind": kind}


def test_interpret_envelope_ok():
    m = _member("scorecard", True, "envelope")
    r = gb.interpret(m, {"ok": True, "verdict": "OK", "reason": "clean"}, 0)
    assert r["state"] == "ok" and r["ok"] is True


def test_interpret_envelope_gating_red():
    m = _member("scorecard", True, "envelope")
    r = gb.interpret(m, {"ok": False, "verdict": "ACTION", "reason": "debt rose"}, 1)
    assert r["state"] == "red", r


def test_interpret_envelope_nongating_action():
    # A non-gating member reporting not-ok is advisory ACTION, not red.
    m = _member("fresh_status", False, "envelope")
    r = gb.interpret(m, {"ok": False, "verdict": "ACTION", "reason": "stale bench"}, 0)
    assert r["state"] == "action", r


def test_interpret_errored_on_no_payload():
    m = _member("scorecard", True, "envelope")
    r = gb.interpret(m, None, -1, error="timed out")
    assert r["state"] == "errored" and r["ok"] is False


def test_interpret_loop_audit_buckets():
    m = _member("loop_audit", False, "loop_audit")
    healthy = gb.interpret(m, {"counts": {"healthy": 3, "action": 0, "broken": 0}}, 0)
    assert healthy["state"] == "ok"
    action = gb.interpret(m, {"counts": {"healthy": 2, "action": 1, "broken": 0}}, 0)
    assert action["state"] == "action"
    broken = gb.interpret(m, {"counts": {"healthy": 1, "action": 0, "broken": 1}}, 1)
    # loop-audit is non-gating advisory: a broken sub-loop surfaces as `action`, not
    # `errored` — a peripheral check can't gate the whole garden red.
    assert broken["state"] == "action", broken


def _row(key, state, gates=False):
    return {"key": key, "label": key, "gates": gates, "state": state,
            "ok": state in ("ok", "action"), "verdict": "OK", "detail": "d",
            "exit_code": 0}


def test_fold_all_clear():
    p = gb.fold([_row("a", "ok"), _row("b", "ok")], workspace="w", commit="c")
    assert p["ok"] is True and p["finding"] == "garden_clear"
    assert gb.check_gate(p)[0] == 0


def test_fold_advisory_is_green():
    # An advisory (non-gating) ACTION keeps the bundle green.
    p = gb.fold([_row("a", "ok"), _row("b", "action")], workspace="w", commit="c")
    assert p["ok"] is True and p["finding"] == "garden_advisory"
    assert gb.check_gate(p)[0] == 0


def test_fold_red_gates():
    p = gb.fold([_row("a", "ok"), _row("sc", "red", gates=True)], workspace="w", commit="c")
    assert p["ok"] is False and p["finding"] == "garden_gate_red"
    assert gb.check_gate(p)[0] == 1


def test_fold_errored_gates_over_red():
    # An unmeasured member trips the gate even alongside a red — and is reported
    # first, since "could not run" is the more fundamental failure.
    p = gb.fold([_row("sc", "red", gates=True), _row("b", "errored")],
                workspace="w", commit="c")
    assert p["finding"] == "garden_member_unmeasured"
    assert gb.check_gate(p)[0] == 1


def test_garden_off_recognizes_vocabulary():
    for v in ("0", "off", "OFF", "false", "no", "disable", "disabled"):
        os.environ["FAK_GARDEN"] = v
        assert gb.garden_off() is True, v
    for v in ("1", "on", "true", ""):
        os.environ["FAK_GARDEN"] = v
        assert gb.garden_off() is False, v
    os.environ.pop("FAK_GARDEN", None)


def test_skipped_payload_is_wellformed_and_green():
    p = gb.skipped_payload(workspace="w", commit="c")
    for k in ("schema", "ok", "verdict", "finding", "reason", "next_action"):
        assert k in p, k
    assert p["ok"] is True and p["finding"] == "garden_skipped"
    assert gb.check_gate(p)[0] == 0


def test_live_smoke_skip_path_emits_valid_envelope():
    """Live runner smoke (fast): FAK_GARDEN=off → valid JSON envelope, exit 0."""
    root = gb.repo_root()
    env = {**os.environ, "FAK_GARDEN": "off"}
    proc = subprocess.run(
        [sys.executable, str(root / "tools" / "garden_bundle.py"), "--check", "--json"],
        cwd=str(root), capture_output=True, text=True, env=env, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["schema"] == gb.SCHEMA
    for k in ("ok", "verdict", "finding", "reason", "next_action"):
        assert k in payload, k
    assert payload.get("skipped") is True


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok  - {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL- {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
