#!/usr/bin/env python3
"""Hermetic tests for tools/dos_supervisor_watchdog.py."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
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
        # The canary launches the dispatch worker on the spawn lane -- the compiled
        # Go binary tools/.bin/dispatchworker[.exe] when built, else dispatch_worker.py
        # via this interpreter (both interpreter-portable). Branch-tolerant so the test
        # is deterministic whether or not the binary happens to be built locally.
        cmd0 = got["command"][0]
        launches_worker = (
            (cmd0 == sys.executable and got["command"][1].endswith("dispatch_worker.py"))
            or cmd0.endswith("dispatchworker")
            or cmd0.endswith("dispatchworker.exe")
        )
        self.assertTrue(launches_worker, got["command"])
        self.assertIn("--lane", got["command"])
        self.assertIn("adjudicator", got["command"])
        self.assertNotIn("result", got)

    def test_enact_command_prefers_go_binary_else_python(self) -> None:
        import tempfile

        mod = load()
        with tempfile.TemporaryDirectory() as d:
            ws = Path(d)
            # Binary absent -> fall back to dispatch_worker.py via sys.executable.
            cmd = mod.enact_command(ws, 1, 1, lane="docs")
            self.assertEqual(cmd[0], sys.executable)
            self.assertTrue(cmd[1].endswith("dispatch_worker.py"))
            self.assertIn("docs", cmd)
            # Binary present -> prefer the Go launcher (no interpreter spawn).
            bin_dir = ws / "tools" / ".bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "dispatchworker").write_text("#!/bin/sh\n", encoding="utf-8")
            cmd2 = mod.enact_command(ws, 1, 1, lane="docs")
            self.assertTrue(cmd2[0].endswith("dispatchworker"))
            self.assertEqual(cmd2[1:3], ["--workspace", str(ws)])
            self.assertIn("--lane", cmd2)
            self.assertIn("docs", cmd2)

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


class RouteDefectStopTest(unittest.TestCase):
    """The #381 closure rung: an enact_failed defect-STOP self-routes a pickable row."""

    def test_enact_failed_self_routes_a_findings_queue_row(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tools").mkdir()  # mark this tmp dir a real checkout (sink lives under tools/)
            plan = {
                "action": "enact_failed",
                "reason": "worker dispatch returned non-zero",
                "result": {"returncode": 9, "stderr": "boom"},
                "readiness": {"spawn": ["adjudicator"]},
            }
            verdict = mod.route_defect_stop(root, plan)
            self.assertIsNotNone(verdict)
            self.assertTrue(verdict["routed"])
            queue = root / "tools" / "_registry" / "findings_route" / "findings-followup-queue.md"
            self.assertTrue(queue.exists())
            text = queue.read_text(encoding="utf-8")
            self.assertIn("supervisor-enact-failed", text)
            self.assertIn("adjudicator", text)

    def test_route_defect_stop_is_fail_open_on_bad_plan(self) -> None:
        mod = load()
        # A plan with no result/readiness must not raise — worst case routing is a no-op.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tools").mkdir()
            verdict = mod.route_defect_stop(root, {})
            self.assertIsNotNone(verdict)
            self.assertTrue(verdict.get("routed"))  # routes a 'default'-lane row


if __name__ == "__main__":
    unittest.main()
