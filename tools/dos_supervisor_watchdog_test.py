#!/usr/bin/env python3
"""Hermetic tests for tools/dos_supervisor_watchdog.py."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "dos_supervisor_watchdog.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("dos_supervisor_watchdog", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def readiness(
    verdict: str = "READY_TO_CANARY",
    ok: bool = True,
    spawn: list[str] | None = None,
    *,
    alive: int = 0,
    target: int = 3,
) -> dict:
    return {
        "schema": "fleet-dos-supervisor-status/1",
        "ok": ok,
        "verdict": verdict,
        "why": "test",
        "next_action": "test action",
        "supervise": {
            "spawn": spawn if spawn is not None else ["adjudicator"],
            "alive": alive,
            "target": target,
        },
    }


def clean_safety() -> dict:
    return {"ok": True, "blockers": [], "git": {"dirty": False, "relation": "in_sync"}}


class DosSupervisorWatchdogTest(unittest.TestCase):
    def test_dry_run_ready_to_canary_reports_bounded_command_without_runner(self) -> None:
        mod = load()

        def fail_runner(_cmd, _cwd, _timeout_s):
            raise AssertionError("dry run must not call runner")

        got = mod.run_watchdog(
            workspace=ROOT,
            target=1,
            max_ticks=1,
            live=False,
            timeout_s=120,
            readiness=readiness(),
            runner=fail_runner,
            safety=clean_safety(),
        )

        self.assertTrue(got["ok"])
        self.assertEqual(got["action"], "would_enact")
        self.assertTrue(got["safety"]["ok"])
        # dos v0.28.0: the canary launches dispatch_worker.py directly (no
        # `dos loop --enact`), via the watchdog's OWN interpreter (sys.executable)
        # so it is portable to nodes that only ship python3 (e.g. macOS).
        self.assertEqual(got["command"][0], sys.executable)
        self.assertTrue(got["command"][1].endswith("dispatch_worker.py"))
        self.assertIn("--lane", got["command"])
        self.assertIn("adjudicator", got["command"])
        self.assertNotIn("result", got)

    def test_default_target_is_one_above_current_alive_count(self) -> None:
        mod = load()
        got = mod.run_watchdog(
            workspace=ROOT,
            target=None,
            max_ticks=1,
            live=False,
            timeout_s=120,
            readiness=readiness(alive=1, target=3),
            safety=clean_safety(),
        )

        self.assertTrue(got["ok"])
        self.assertEqual(got["action"], "would_enact")
        # target = one above the current alive count (resolve_target). dos v0.28.0:
        # it lives in the plan's `target` field, not the per-lane canary command.
        self.assertEqual(got["target"], 2)

    def test_blocked_readiness_refuses_without_runner(self) -> None:
        mod = load()

        def fail_runner(_cmd, _cwd, _timeout_s):
            raise AssertionError("blocked state must not call runner")

        got = mod.run_watchdog(
            workspace=Path("C:/work/fleet"),
            target=1,
            max_ticks=1,
            live=True,
            timeout_s=120,
            readiness=readiness("PLAN_DRIFT", ok=False),
            runner=fail_runner,
            safety=clean_safety(),
        )

        self.assertFalse(got["ok"])
        self.assertEqual(got["action"], "refuse")

    def test_at_target_noops_without_runner(self) -> None:
        mod = load()

        def fail_runner(_cmd, _cwd, _timeout_s):
            raise AssertionError("noop state must not call runner")

        got = mod.run_watchdog(
            workspace=Path("C:/work/fleet"),
            target=1,
            max_ticks=1,
            live=True,
            timeout_s=120,
            readiness=readiness("AT_TARGET", ok=True, spawn=[]),
            runner=fail_runner,
            safety=clean_safety(),
        )

        self.assertTrue(got["ok"])
        self.assertEqual(got["action"], "noop")

    def test_live_ready_to_canary_calls_runner_once(self) -> None:
        mod = load()
        calls = []

        def runner(cmd, cwd, timeout_s):
            calls.append((cmd, cwd, timeout_s))
            return {"returncode": 0, "stdout": "{}", "stderr": ""}

        got = mod.run_watchdog(
            workspace=Path("C:/work/fleet"),
            target=1,
            max_ticks=1,
            live=True,
            timeout_s=17,
            readiness=readiness(),
            runner=runner,
            safety=clean_safety(),
        )

        self.assertTrue(got["ok"])
        self.assertEqual(got["action"], "enacted")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], got["command"])
        self.assertEqual(calls[0][2], 17)

    def test_live_runner_failure_is_reported(self) -> None:
        mod = load()

        got = mod.run_watchdog(
            workspace=Path("C:/work/fleet"),
            target=1,
            max_ticks=1,
            live=True,
            timeout_s=120,
            readiness=readiness(),
            runner=lambda _cmd, _cwd, _timeout_s: {"returncode": 9, "stdout": "", "stderr": "boom"},
            safety=clean_safety(),
        )

        self.assertFalse(got["ok"])
        self.assertEqual(got["action"], "enact_failed")
        self.assertEqual(got["result"]["returncode"], 9)

    def test_live_refuses_dirty_workspace_without_runner(self) -> None:
        mod = load()
        dirty = {
            "ok": False,
            "blockers": [{"kind": "dirty", "detail": "worktree has 2 dirty path(s)"}],
            "git": {"dirty": True, "dirty_count": 2},
        }

        def fail_runner(_cmd, _cwd, _timeout_s):
            raise AssertionError("unsafe live run must not call runner")

        got = mod.run_watchdog(
            workspace=Path("C:/work/fleet"),
            target=1,
            max_ticks=1,
            live=True,
            timeout_s=120,
            readiness=readiness(),
            runner=fail_runner,
            safety=dirty,
        )

        self.assertFalse(got["ok"])
        self.assertEqual(got["action"], "refuse")
        self.assertIn("dirty", got["reason"])

    def test_live_allows_dirty_workspace_only_with_operator_override(self) -> None:
        mod = load()
        dirty = {
            "ok": False,
            "blockers": [{"kind": "dirty", "detail": "worktree has 2 dirty path(s)"}],
            "git": {"dirty": True, "dirty_count": 2},
        }
        calls = []

        def runner(cmd, cwd, timeout_s):
            calls.append((cmd, cwd, timeout_s))
            return {"returncode": 0, "stdout": "{}", "stderr": ""}

        got = mod.run_watchdog(
            workspace=Path("C:/work/fleet"),
            target=1,
            max_ticks=1,
            live=True,
            timeout_s=120,
            readiness=readiness(),
            runner=runner,
            safety=dirty,
            allow_dirty=True,
        )

        self.assertTrue(got["ok"])
        self.assertEqual(got["action"], "enacted")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
