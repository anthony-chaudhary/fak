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

sys.path.insert(0, str(SCRIPT.parent))
import dos_fleet_lease as fl  # noqa: E402 - the cross-node lease primitives under test

T0 = 1_750_000_000.0  # fixed injected "now" (epoch seconds), mirrors dos_fleet_lease_test


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

    def test_plan_surface_empty_falls_back_to_issue_resolver(self) -> None:
        mod = load()

        def fail_runner(_cmd, _cwd, _timeout_s):
            raise AssertionError("dry run must not call runner")

        got = mod.run_watchdog(
            workspace=ROOT,
            target=2,
            max_ticks=1,
            live=False,
            timeout_s=120,
            readiness=readiness("PLAN_SURFACE_EMPTY", ok=False, spawn=[], alive=0, target=0),
            issue_fallback={
                "schema": "fleet-issue-lane-router/1",
                "counts": {"open": 5, "routed": 3, "unrouted": 2},
                "lanes": {"tools": {"issues": [1, 2, 3]}},
            },
            runner=fail_runner,
            safety=clean_safety(),
        )

        self.assertTrue(got["ok"])
        self.assertEqual(got["action"], "would_enact")
        self.assertIn("issue_resolve_dispatch.py", " ".join(got["command"]))
        self.assertNotIn("--live", got["command"])
        self.assertEqual(got["issue_fallback"]["routed"], 3)
        self.assertIn("falling back to issue surface", got["reason"])

    def test_plan_surface_empty_without_routed_issues_still_refuses(self) -> None:
        mod = load()

        got = mod.run_watchdog(
            workspace=ROOT,
            target=1,
            max_ticks=1,
            live=False,
            timeout_s=120,
            readiness=readiness("PLAN_SURFACE_EMPTY", ok=False, spawn=[], alive=0, target=0),
            issue_fallback={
                "schema": "fleet-issue-lane-router/1",
                "counts": {"open": 2, "routed": 0, "unrouted": 2},
                "lanes": {},
            },
            safety=clean_safety(),
        )

        self.assertFalse(got["ok"])
        self.assertEqual(got["action"], "refuse")

    def test_live_plan_empty_fallback_runs_issue_resolver_with_live_flag(self) -> None:
        mod = load()
        calls = []

        got = mod.run_watchdog(
            workspace=ROOT,
            target=1,
            max_ticks=1,
            live=True,
            timeout_s=120,
            readiness=readiness("PLAN_SURFACE_EMPTY", ok=False, spawn=[], alive=0, target=0),
            issue_fallback={
                "schema": "fleet-issue-lane-router/1",
                "counts": {"open": 1, "routed": 1, "unrouted": 0},
                "lanes": {"tools": {"issues": [1]}},
            },
            runner=lambda cmd, cwd, timeout_s: (
                calls.append((cmd, cwd, timeout_s)) or
                {"returncode": 0, "stdout": "{}", "stderr": ""}
            ),
            safety=clean_safety(),
        )

        self.assertTrue(got["ok"])
        self.assertEqual(got["action"], "enacted")
        self.assertEqual(len(calls), 1)
        self.assertIn("--live", calls[0][0])
        self.assertIn("issue_resolve_dispatch.py", " ".join(calls[0][0]))

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


class FleetLeaseGateTest(unittest.TestCase):
    """Issue #21: the cross-node global-lane gate the live watchdog runs before launch.

    Exercises the REAL dos_fleet_lease publish/materialize/collision primitives through a
    tempfile LocalDirStore + injected kernel `live` seam — no git, no `dos` subprocess.
    """

    def _store(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return fl.LocalDirStore(Path(tmp.name))

    def test_non_global_lane_proceeds(self):
        mod = load()
        self.assertIsNone(mod.fleet_lease_gate("adjudicator", workspace=ROOT, env={}))

    def test_global_lane_default_store_is_gitref(self):
        # Issue #21 activation: unconfigured global lanes default to the git-ref floor.
        mod = load()
        self.assertIsInstance(mod.fleet_lease_store(ROOT, {}, lane="release"), fl.GitRefStore)

    def test_shard_lane_default_store_remains_local(self):
        mod = load()
        self.assertIsInstance(mod.fleet_lease_store(ROOT, {}, lane="docs"), fl.LocalDirStore)

    def test_global_lane_default_gate_consults_store(self):
        mod = load()
        store = self._store()
        seen = {}
        original = mod.fleet_lease_store

        def fake_store(workspace, env, *, lane=""):
            seen["lane"] = lane
            return store

        mod.fleet_lease_store = fake_store
        try:
            got = mod.fleet_lease_gate(
                "release", workspace=ROOT, env={},
                live=lambda _ws: [], now=T0,
            )
        finally:
            mod.fleet_lease_store = original

        self.assertIsNone(got)
        self.assertEqual(seen["lane"], "release")

    def test_global_lane_proceeds_when_no_foreign_holder(self):
        mod = load()
        store = self._store()
        got = mod.fleet_lease_gate(
            "release", workspace=ROOT, env={}, store=store,
            live=lambda _ws: [], now=T0,
        )
        self.assertIsNone(got)

    def test_global_lane_refuses_when_foreign_peer_holds(self):
        mod = load()
        store = self._store()
        # Node B has published a LIVE `release` lease (heartbeat at T0).
        peer = fl.build_snapshot(
            host="B",
            leases=[{"lane": "release", "host_id": "B", "mode": "exclusive", "holder": "b-loop"}],
            now=T0, prev_epoch=None,
        )
        store.compare_and_swap("B", None, peer)
        got = mod.fleet_lease_gate(
            "release", workspace=ROOT, env={}, store=store,
            live=lambda _ws: [], now=T0,
        )
        self.assertIsNotNone(got)
        self.assertEqual(got["reason"], fl.REASON_HELD_REMOTE)
        self.assertEqual(got["host_id"], "B")
        self.assertIn("release", got["detail"])

    def test_global_publish_failure_fails_closed(self):
        mod = load()

        class FailPublish:
            def read_fleet(self):
                return {}

            def compare_and_swap(self, host, expected, snapshot):
                return False, None  # publish lost / store unreachable

        got = mod.fleet_lease_gate(
            "release", workspace=ROOT, env={},
            store=FailPublish(), live=lambda _ws: [], now=T0,
        )
        self.assertIsNotNone(got)  # the global floor refuses rather than silently double-grant
        self.assertIn("release", got["detail"])

    def test_global_store_exception_fails_closed(self):
        mod = load()

        class BoomStore:
            def read_fleet(self):
                raise RuntimeError("remote down")

            def compare_and_swap(self, *a, **k):
                raise RuntimeError("remote down")

        got = mod.fleet_lease_gate(
            "release", workspace=ROOT, env={},
            store=BoomStore(), live=lambda _ws: [], now=T0,
        )
        self.assertIsNotNone(got)
        self.assertIn("unavailable", got["detail"])


class BuildPlanAllowlistTest(unittest.TestCase):
    """Issue #21: FLEET_LANE_ALLOWLIST is advisory steering at build_plan, not a floor."""

    def test_no_allowlist_keeps_first_proposed_lane(self):
        mod = load()
        plan = mod.build_plan(
            readiness(spawn=["docs", "release"]),
            workspace=ROOT, target=1, max_ticks=1, live=False, safety=clean_safety(), env={},
        )
        self.assertEqual(mod._command_lane(plan["command"]), "docs")

    def test_allowlist_narrows_to_allowed_lane(self):
        mod = load()
        plan = mod.build_plan(
            readiness(spawn=["docs", "release"]),
            workspace=ROOT, target=1, max_ticks=1, live=False, safety=clean_safety(),
            env={"FLEET_LANE_ALLOWLIST": "release"},
        )
        self.assertEqual(plan["action"], "would_enact")
        self.assertEqual(mod._command_lane(plan["command"]), "release")

    def test_allowlist_refuses_when_all_off_list(self):
        mod = load()
        plan = mod.build_plan(
            readiness(spawn=["docs"]),
            workspace=ROOT, target=1, max_ticks=1, live=False, safety=clean_safety(),
            env={"FLEET_LANE_ALLOWLIST": "release, abi"},
        )
        self.assertEqual(plan["action"], "refuse")
        self.assertFalse(plan["ok"])
        self.assertIn("allowlist", plan["reason"].lower())


class RunWatchdogLeaseGateWiringTest(unittest.TestCase):
    """The gate is consulted on a live enact, sees the steered lane, and can block it."""

    def test_gate_refusal_blocks_live_launch(self):
        mod = load()

        def fail_runner(_cmd, _cwd, _timeout_s):
            raise AssertionError("a gate-refused launch must not call the runner")

        got = mod.run_watchdog(
            workspace=ROOT, target=1, max_ticks=1, live=True, timeout_s=120,
            readiness=readiness(spawn=["release"]), runner=fail_runner, safety=clean_safety(),
            lease_gate=lambda lane, **kw: {"detail": "global lane 'release' held by peer B", "host_id": "B"},
        )
        self.assertFalse(got["ok"])
        self.assertEqual(got["action"], "refuse")
        self.assertIn("release", got["reason"])
        self.assertEqual(got["fleet_lease"]["host_id"], "B")

    def test_gate_proceed_runs_worker_once(self):
        mod = load()
        calls = []

        got = mod.run_watchdog(
            workspace=ROOT, target=1, max_ticks=1, live=True, timeout_s=120,
            readiness=readiness(spawn=["release"]), safety=clean_safety(),
            runner=lambda cmd, cwd, t: calls.append(cmd) or {"returncode": 0, "stdout": "{}", "stderr": ""},
            lease_gate=lambda lane, **kw: None,
        )
        self.assertEqual(got["action"], "enacted")
        self.assertEqual(len(calls), 1)

    def test_gate_receives_allowlist_steered_lane(self):
        mod = load()
        seen = {}

        def recording_gate(lane, **kw):
            seen["lane"] = lane
            return {"detail": "blocked for assertion"}  # block so the runner is never reached

        got = mod.run_watchdog(
            workspace=ROOT, target=1, max_ticks=1, live=True, timeout_s=120,
            readiness=readiness(spawn=["docs", "release"]), safety=clean_safety(),
            runner=lambda *a: {"returncode": 0},
            env={"FLEET_LANE_ALLOWLIST": "release"},
            lease_gate=recording_gate,
        )
        # build_plan steered docs->release; the gate must see the lane that will launch.
        self.assertEqual(seen["lane"], "release")
        self.assertEqual(got["action"], "refuse")


class SlackEventTest(unittest.TestCase):
    """slack_event is pure (post only on an actionable tick); maybe_post_slack gates on
    enabled and posts via the injected transport — no network, no real token/channel."""

    def test_slack_event_only_on_actionable_action(self):
        mod = load()
        self.assertIsNone(mod.slack_event({"action": "noop", "reason": "at target"}))
        self.assertIsNone(mod.slack_event({"action": "would_enact", "reason": "dry"}))
        enacted = mod.slack_event({"action": "enacted", "reason": "canary on adjudicator"})
        self.assertEqual(enacted["level"], "info")
        self.assertIn("launched", enacted["title"])
        failed = mod.slack_event({"action": "enact_failed", "reason": "nonzero",
                                  "result": {"returncode": 9}})
        self.assertEqual(failed["level"], "crit")
        self.assertIn("rc=9", failed["detail"])
        refused = mod.slack_event({"action": "refuse", "reason": "peer holds release"})
        self.assertEqual(refused["level"], "warn")

    def test_maybe_post_slack_disabled_is_noop(self):
        mod = load()
        called = []
        verdict = mod.maybe_post_slack({"action": "enacted", "reason": "x"}, enabled=False,
                                       transport=lambda *a: called.append(1))
        self.assertIsNone(verdict)
        self.assertEqual(called, [])

    def test_maybe_post_slack_posts_on_enacted(self):
        import json as _json
        import os
        mod = load()
        saved = {k: os.environ.pop(k, None) for k in
                 ("FAK_DISPATCH_TOKEN", "FAK_DISPATCH_CHANNEL", "FAK_SCOREBOARD_TOKEN")}
        os.environ["FAK_SCOREBOARD_TOKEN"] = "xoxb-test"
        os.environ["FAK_DISPATCH_CHANNEL"] = "C0SUP"
        calls = []

        def transport(url, body, headers, timeout):
            calls.append(_json.loads(body.decode("utf-8")))
            return 200, _json.dumps({"ok": True, "ts": "1.1", "channel": "C0SUP"})

        try:
            verdict = mod.maybe_post_slack({"action": "enacted", "reason": "canary"},
                                           enabled=True, transport=transport)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertTrue(verdict["posted"])
        self.assertIn("launched", calls[0]["text"])


if __name__ == "__main__":
    unittest.main()
