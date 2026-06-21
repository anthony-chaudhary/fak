#!/usr/bin/env python3
"""Hermetic tests for tools/dos_supervisor_canary_audit.py."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "dos_supervisor_canary_audit.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("dos_supervisor_canary_audit", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def readiness(
    verdict: str = "READY_TO_CANARY",
    *,
    ok: bool = True,
    alive: int = 0,
    done_units: int = 0,
    spawn: list[str] | None = None,
    reap: list[str] | None = None,
    flag: list[str] | None = None,
    drift_count: int = 0,
) -> dict:
    return {
        "schema": "fleet-dos-supervisor-status/1",
        "ok": ok,
        "verdict": verdict,
        "why": "test readiness",
        "next_action": "test next",
        "supervise": {
            "target": 3,
            "alive": alive,
            "spawn": spawn if spawn is not None else ["adjudicator"],
            "reap": reap or [],
            "flag": flag or [],
        },
        "plans": {
            "total_plans": 1,
            "total_units": 5,
            "done_units": done_units,
            "drift_count": drift_count,
        },
    }


def dry_run(action: str = "would_enact", *, ok: bool = True) -> dict:
    return {
        "schema": "fleet-dos-supervisor-watchdog/1",
        "ok": ok,
        "action": action,
        "reason": "test watchdog",
        "safety": {"ok": True, "blockers": []},
        "live": False,
        "target": 1,
        "max_ticks": 1,
        "command": ["dos", "loop", "--enact"],
    }


class DosSupervisorCanaryAuditTest(unittest.TestCase):
    def test_pre_canary_ready_names_operator_gate(self) -> None:
        mod = load()
        got = mod.build_payload(
            workspace=Path("C:/work/fleet"),
            readiness=readiness(),
            dry_run=dry_run(),
        )

        self.assertTrue(got["ok"])
        self.assertEqual(got["verdict"], "PRE_CANARY_READY")
        self.assertEqual(got["finding"], "operator_gate")
        self.assertIn("operator-gated", got["reason"])
        self.assertIn("--live", got["next_action"])

    def test_landed_plan_units_report_productive(self) -> None:
        mod = load()
        got = mod.build_payload(
            workspace=Path("C:/work/fleet"),
            readiness=readiness("AT_TARGET", alive=1, done_units=2, spawn=[]),
            dry_run=dry_run("noop"),
        )

        self.assertTrue(got["ok"])
        self.assertEqual(got["verdict"], "PRODUCTIVE")
        self.assertEqual(got["finding"], "landed_work")

    def test_live_worker_without_done_units_is_observed(self) -> None:
        mod = load()
        got = mod.build_payload(
            workspace=Path("C:/work/fleet"),
            readiness=readiness("AT_TARGET", alive=1, spawn=[]),
            dry_run=dry_run("noop"),
        )

        self.assertTrue(got["ok"])
        self.assertEqual(got["verdict"], "CANARY_OBSERVED")
        self.assertEqual(got["finding"], "live_worker_observed")

    def test_readiness_failure_is_typed_blocker(self) -> None:
        mod = load()
        got = mod.build_payload(
            workspace=Path("C:/work/fleet"),
            readiness=readiness("PLAN_DRIFT", ok=False, drift_count=1),
            dry_run=dry_run("refuse", ok=False),
        )

        self.assertFalse(got["ok"])
        self.assertEqual(got["verdict"], "BLOCKED")
        self.assertEqual(got["finding"], "typed_blocker")
        self.assertGreaterEqual(len(got["blockers"]), 1)

    def test_reap_and_flag_lanes_are_blockers(self) -> None:
        mod = load()
        got = mod.build_payload(
            workspace=Path("C:/work/fleet"),
            readiness=readiness("READY", spawn=[], reap=["agent"], flag=["docs"]),
            dry_run=dry_run("noop"),
        )

        self.assertFalse(got["ok"])
        kinds = {blocker["kind"] for blocker in got["blockers"]}
        self.assertIn("reap", kinds)
        self.assertIn("flag", kinds)

    def test_workspace_safety_failure_is_typed_blocker(self) -> None:
        mod = load()
        unsafe = dry_run()
        unsafe["safety"] = {
            "ok": False,
            "blockers": [{"kind": "dirty", "detail": "worktree has 1 dirty path(s)"}],
        }

        got = mod.build_payload(
            workspace=Path("C:/work/fleet"),
            readiness=readiness(),
            dry_run=unsafe,
        )

        self.assertFalse(got["ok"])
        self.assertEqual(got["verdict"], "BLOCKED")
        self.assertIn("workspace_safety:dirty", {blocker["kind"] for blocker in got["blockers"]})


if __name__ == "__main__":
    unittest.main()
