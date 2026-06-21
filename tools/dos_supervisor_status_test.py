#!/usr/bin/env python3
"""Hermetic tests for tools/dos_supervisor_status.py."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "dos_supervisor_status.py"


def load():
    spec = importlib.util.spec_from_file_location("dos_supervisor_status", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def doctor(template: str = "claude /dos-kernel:dos-dispatch-loop --lane {lane}") -> dict:
    return {
        "supervise": {"target": 3, "worker_launch_template": template},
        "paths": {"plans_glob": "PLAN-*.md"},
    }


def loop(verdict: str = "FILLING", *, alive: int = 0) -> dict:
    return {
        "verdict": verdict,
        "target": 3,
        "alive": alive,
        "admissible": 40,
        "spawn": [{"lane": "adjudicator"}, {"lane": "agent"}],
        "reap": [],
        "flag": [],
    }


def plan_audit(total_units: int = 5, drift: list | None = None) -> dict:
    return {
        "counts": {"total_plans": 1},
        "plans": [{"id": "PLAN-x", "total_units": total_units}],
        "drift": drift or [],
        "work_units": {
            "task_weighted": {
                "total_units": total_units,
                "done_units": 0,
                "coverage_plans": 1,
            }
        },
    }


class DosSupervisorStatusTest(unittest.TestCase):
    def test_reads_pretty_multiline_json(self) -> None:
        dss = load()
        self.assertEqual(dss.read_json_from_text('{\n  "ok": true,\n  "n": 2\n}\n'), {"ok": True, "n": 2})

    def test_ready_to_canary_names_bounded_operator_command(self) -> None:
        dss = load()
        payload = dss.build_payload(
            workspace="C:/work/fleet",
            doctor=doctor(),
            loop=loop(),
            plan_audit=plan_audit(),
        )
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["verdict"], "READY_TO_CANARY")
        self.assertEqual(payload["supervise"]["spawn"], ["adjudicator", "agent"])
        self.assertIn("--enact --target 1 --max-ticks 1", payload["next_action"])

    def test_ready_to_canary_targets_one_above_current_alive_count(self) -> None:
        dss = load()
        payload = dss.build_payload(
            workspace="C:/work/fleet",
            doctor=doctor(),
            loop=loop(alive=1),
            plan_audit=plan_audit(),
        )
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["verdict"], "READY_TO_CANARY")
        self.assertIn("alive=1/3", payload["why"])
        self.assertIn("--enact --target 2 --max-ticks 1", payload["next_action"])

    def test_missing_lane_placeholder_blocks_config(self) -> None:
        dss = load()
        payload = dss.build_payload(
            workspace="C:/work/fleet",
            doctor=doctor("claude /dos-kernel:dos-dispatch-loop"),
            loop=loop(),
            plan_audit=plan_audit(),
        )
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["verdict"], "CONFIG_BLOCKED")
        self.assertIn("{lane}", payload["why"])

    def test_empty_plan_surface_blocks_launch(self) -> None:
        dss = load()
        payload = dss.build_payload(
            workspace="C:/work/fleet",
            doctor=doctor(),
            loop=loop(),
            plan_audit=plan_audit(total_units=0),
        )
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["verdict"], "PLAN_SURFACE_EMPTY")

    def test_target_unreachable_blocks_launch(self) -> None:
        dss = load()
        payload = dss.build_payload(
            workspace="C:/work/fleet",
            doctor=doctor(),
            loop={**loop("TARGET_UNREACHABLE"), "reason": "target too high"},
            plan_audit=plan_audit(),
        )
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["verdict"], "TARGET_UNREACHABLE")
        self.assertEqual(payload["why"], "target too high")


if __name__ == "__main__":
    unittest.main()
