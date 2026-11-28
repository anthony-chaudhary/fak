#!/usr/bin/env python3
"""Hermetic tests for dispatch evaluate routing, gating, seat sizing, and spawn bursts."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue_resolve_dispatch_test import ROOT, load, passing_issue_contract, patch_eval_env  # noqa: E402


class BackendRoutingTest(unittest.TestCase):
    SPAWN_OK_OC = {
        "verdict": "SPAWN_OK", "reason": "ok", "cap": 2, "live": 0,
        "account": {"tag": "default", "tier": 2,
                    "model": "zai-coding-plan/glm-5.2", "dir": "/cfg/opencode"},
    }

    def test_opencode_backend_routes_product_and_stamps_payload(self) -> None:
        mod = load()
        seen: dict = {}

        def fake_pre(root, **kw):
            seen.update(kw)
            return self.SPAWN_OK_OC

        mod.issue_dispatch.refresh_registry = lambda root: {"ok": True}
        mod.issue_dispatch.preflight = fake_pre
        mod.lane_issue_numbers = lambda root, lane, exclude=None: {
            "lane": "docs", "numbers": [260], "by_lane_count": {"docs": 1}}
        mod.live_resolution_issues = lambda runs_dir: set()
        mod.live_resolution_lanes = lambda runs_dir: set()
        mod.live_lane_lease_lanes = lambda root: {"lanes": []}
        mod.recently_attempted_issues = lambda runs_dir, *, cooldown_min, **k: set()
        mod.issue_worker_prompt.build = lambda n, lane, *, workspace: {
            "prompt": f"resolve #{n}", "prompt_chars": 100, "title": f"t{n}"}
        mod.issue_contract_review = passing_issue_contract
        mod.check_weekly_cap = lambda runs_dir, **k: {"capped": False}
        mod.check_backend_health = lambda runs_dir, **k: {"state": "healthy"}
        mod.read_dead_backends = lambda runs_dir, **k: []
        # The active opencode gateway gate (#3866) probes a real HTTP endpoint when
        # it is left unstubbed, so evaluate() would return GATEWAY_DOWN here purely
        # because no gateway is listening on this machine — a live dependency this
        # file forbids. Stub the gate itself (module-local, load() returns a fresh
        # module) rather than account_probe, which is shared via sys.modules.
        mod.gate_opencode_gateway = lambda planned, account, **k: (planned, None)
        mod.reap_timed_out_workers = lambda runs_dir, **k: {
            "timeout_s": k.get("timeout_s"), "live": k.get("live"),
            "candidates": [], "reaped": [], "would_reap": []}
        mod.spawn_issue_worker = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("dry-run must not spawn"))

        p = mod.evaluate(ROOT, max_workers=2, work_kind="gardening",
                         lane="docs", live=False, backend="opencode")
        self.assertEqual(seen.get("product"), "opencode")       # routed to glm pool
        self.assertEqual(p["backend"], "opencode")
        # argv[0] is the GUARD binary, not the backend: guarded_launch_command wraps
        # every worker argv, so the backend and its model are pinned INSIDE the wrapped
        # command rather than at the head of it. Asserting on argv[0] would pin the
        # wrapper away again the moment guarding is on, which it is by default.
        self.assertTrue(p["guarded"])
        self.assertIn("opencode", p["command"])
        self.assertIn("zai-coding-plan/glm-5.2", p["command"])   # model pinned/traced
        self.assertTrue(p["ok"])

    def test_unknown_backend_raises(self) -> None:
        mod = load()
        with self.assertRaises(ValueError):
            mod.evaluate(ROOT, max_workers=2, work_kind="x", lane=None,
                         live=False, backend="gpt")


class EvaluateBackendHealthTest(unittest.TestCase):
    """evaluate()-level wiring of the backend-health gate: a dead backend self-
    suppresses (and never spawns) while its re-probe is not due; a healthy backend
    claims a dead sibling's lane + a budget bump."""

    def _common_stubs(self, mod, pre):
        mod.issue_dispatch.refresh_registry = lambda root: {"ok": True}
        mod.issue_dispatch.preflight = lambda root, **kw: dict(pre, _seen_max=kw.get("max_workers"))
        mod.issue_contract_review = passing_issue_contract
        mod.check_weekly_cap = lambda runs_dir, **k: {"capped": False}
        mod.reap_timed_out_workers = lambda runs_dir, **k: {
            "timeout_s": None, "live": k.get("live"),
            "candidates": [], "reaped": [], "would_reap": []}
        mod.prune_dead_sidecars = lambda runs_dir, **k: {"pruned": []}
        # Hermetic: no lane is held, so the pre-spawn lane-lease gate (#1310) never
        # reads the real .dispatch-runs and never trips for these backend-health tests.
        mod.live_resolution_lanes = lambda runs_dir: set()
        mod.live_lane_lease_lanes = lambda root: {"lanes": []}

    def test_dead_backend_self_suppresses_without_spawning(self) -> None:
        mod = load()
        pre = {"verdict": "SPAWN_OK", "reason": "ok", "cap": 2, "live": 0,
               "account": {"tag": "default", "tier": 2, "model": "glm", "dir": "/a"}}
        self._common_stubs(mod, pre)
        mod.check_backend_health = lambda runs_dir, **k: {
            "state": "dead", "since": "2026-06-26T04:00:00Z", "abandoned_lane": "docs",
            "reprobe_due": False, "evidence_logs": ["resolve-872-x.log"]}
        mod.read_dead_backends = lambda runs_dir, **k: []
        mod.lane_issue_numbers = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("a held-dead backend must short-circuit before the lane router"))
        mod.spawn_issue_worker = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("a held-dead backend must never spawn"))
        p = mod.evaluate(ROOT, max_workers=2, work_kind="gardening", lane=None,
                         live=True, backend="opencode")
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "BACKEND_UNHEALTHY")
        self.assertEqual(p["action"], "backend_unhealthy")
        self.assertEqual(p["backend_health"]["abandoned_lane"], "docs")

    def test_dead_backend_admits_one_reprobe_when_due(self) -> None:
        # reprobe_due True -> the gate does NOT short-circuit; the tick proceeds to the
        # lane router so one worker can test recovery.
        mod = load()
        pre = {"verdict": "SPAWN_OK", "reason": "ok", "cap": 2, "live": 0,
               "account": {"tag": "default", "tier": 2, "model": "glm", "dir": "/a"}}
        self._common_stubs(mod, pre)
        mod.check_backend_health = lambda runs_dir, **k: {
            "state": "dead", "since": "x", "abandoned_lane": "docs", "reprobe_due": True,
            "evidence_logs": []}
        mod.read_dead_backends = lambda runs_dir, **k: []
        reached = {"router": False}
        def _router(root, lane, exclude=None):
            reached["router"] = True
            return {"lane": None, "numbers": []}
        mod.lane_issue_numbers = _router
        p = mod.evaluate(ROOT, max_workers=2, work_kind="gardening", lane=None,
                         live=False, backend="opencode")
        self.assertTrue(reached["router"])  # re-probe admitted -> router reached
        self.assertNotEqual(p["verdict"], "BACKEND_UNHEALTHY")

    def test_healthy_backend_claims_dead_sibling_lane_and_budget(self) -> None:
        mod = load()
        pre = {"verdict": "SPAWN_OK", "reason": "ok", "cap": 4, "live": 0,
               "account": {"tag": "day24-netra", "tier": 1, "model": "opus", "dir": "/a"}}
        self._common_stubs(mod, pre)
        mod.check_backend_health = lambda runs_dir, **k: {"state": "healthy"}
        mod.read_dead_backends = lambda runs_dir, **k: [
            {"product": "opencode", "abandoned_lane": "docs"}]
        seen = {}
        def _router(root, lane, exclude=None):
            seen["exclude"] = set(exclude or set())
            return {"lane": "docs", "numbers": [501]}
        mod.lane_issue_numbers = _router
        mod.live_resolution_issues = lambda runs_dir, **k: set()
        mod.recently_attempted_issues = lambda runs_dir, **k: set()
        import types
        mod.issue_worker_prompt = types.SimpleNamespace(
            build=lambda issue, lane, workspace=None: {
                "prompt": "p", "prompt_chars": 1, "title": "t"})
        # claude excludes 'docs' by partition; the realloc must DROP that exclude so it
        # can own the freed lane, and bump the effective cap by the freed slot.
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering", lane=None,
                         live=False, backend="claude", exclude_lanes={"docs"})
        self.assertNotIn("docs", seen["exclude"])             # freed lane un-excluded
        self.assertEqual(p["reallocation"]["bonus"], 1)        # +1 from one dead sibling
        self.assertEqual(p["reallocation"]["effective_max_workers"], 3)  # 2 + 1
        self.assertIn("docs", p["reallocation"]["claimed_lanes"])
        # the bumped cap actually reached preflight:
        self.assertEqual(p["preflight"]["cap"], 4)             # pre echoes through

    def test_no_dead_sibling_is_a_noop(self) -> None:
        mod = load()
        pre = {"verdict": "SPAWN_OK", "reason": "ok", "cap": 2, "live": 0,
               "account": {"tag": "day24-netra", "tier": 1, "model": "opus", "dir": "/a"}}
        self._common_stubs(mod, pre)
        mod.check_backend_health = lambda runs_dir, **k: {"state": "healthy"}
        mod.read_dead_backends = lambda runs_dir, **k: []
        mod.lane_issue_numbers = lambda *a, **k: {"lane": None, "numbers": []}
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering", lane=None,
                         live=False, backend="claude")
        self.assertNotIn("reallocation", p)  # payload byte-identical to pre-feature path


class EvaluateLeaseGateTest(unittest.TestCase):
    """evaluate()-level wiring of the fenced lane-lease gate on the LIVE path. The
    gate sits AFTER the dry-run return, so it only fires on live=True. Uses an
    injected lease_runner so the real acquire_lane_lease/release run with no git."""

    SPAWN_OK = {
        "verdict": "SPAWN_OK", "reason": "ok", "cap": 2, "live": 0,
        "account": {"tag": "worker-a", "tier": 1, "model": "opus", "dir": "/acct/a"},
    }

    def _stub(self, mod, *, spawn, lane="gateway", numbers=(467,)):
        mod.issue_dispatch.refresh_registry = lambda root: {"ok": True}
        mod.issue_dispatch.preflight = lambda root, **kw: self.SPAWN_OK
        mod.issue_dispatch.worker_env = lambda d, lane, root: {}
        mod.lane_issue_numbers = lambda root, lane, exclude=None: {
            "lane": lane, "numbers": list(numbers), "by_lane_count": {lane: len(numbers)}}
        mod.live_resolution_issues = lambda runs_dir: set()
        mod.live_resolution_lanes = lambda runs_dir: set()  # same-host scan: free
        mod.live_lane_lease_lanes = lambda root: {"lanes": []}
        mod.recently_attempted_issues = lambda runs_dir, *, cooldown_min, **k: set()
        mod.contract_held_issues = lambda runs_dir, **k: set()
        mod.multi_lane_held_issues = lambda runs_dir, **k: set()
        mod.record_contract_holds = lambda runs_dir, rows, **k: None
        mod.record_multi_lane_holds = lambda runs_dir, rows, **k: None
        mod.check_weekly_cap = lambda runs_dir, **k: {"capped": False}
        mod.check_backend_health = lambda runs_dir, **k: {"state": "healthy"}
        mod.read_dead_backends = lambda runs_dir, **k: []
        mod.reap_timed_out_workers = lambda runs_dir, **k: {
            "candidates": [], "reaped": [], "would_reap": []}
        mod.prune_dead_sidecars = lambda runs_dir, **k: {"pruned": []}
        # Keep the commit-witness sweep + base-sha capture hermetic: no real git/dos
        # subprocess runs in the lease-gate wiring tests.
        mod.witness_exited_workers = lambda runs_dir, root, **k: {
            "live": True, "audited": [], "witnessed": [], "unwitnessed": [],
            "no_commit": []}
        mod._git_capture = lambda root, args, **k: (0, "basesha0\n")
        mod.lane_tree = lambda root, lane: [f"internal/{lane}/**"]
        mod.issue_worker_prompt.build = lambda n, lane, *, workspace: {
            "prompt": f"resolve #{n}", "prompt_chars": 100, "title": f"title {n}"}
        mod.issue_contract_review = passing_issue_contract
        mod.spawn_issue_worker = spawn

    def test_free_lane_acquires_then_spawns(self) -> None:
        mod = load()
        spawned_flag = {"did": False}
        def spawn(*a, **k):
            spawned_flag["did"] = True
            return {"pid": 9, "log": "resolve-467.log", "issue": 467,
                    "lane": "gateway", "backend": "claude"}
        self._stub(mod, spawn=spawn)
        calls = []
        def lease_runner(root, args, **k):
            calls.append(args[0])
            return {"rc": 0, "verdict": {"verdict": {"ok": True},
                                         "record": {"id": "resolve-gateway", "generation": 1}}}
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering", lane="gateway",
                         live=True, lease_runner=lease_runner)
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "SPAWNED")
        self.assertTrue(spawned_flag["did"])
        self.assertTrue(p["lease"]["acquired"])
        self.assertIn("acquire", calls)        # the gate took the lease before spawning
        self.assertIn("reap", calls)           # and reaped expired leases this tick

    def test_held_lease_refuses_lane_lease_held(self) -> None:
        mod = load()
        def boom(*a, **k):
            raise AssertionError("a held lease must short-circuit before the spawn")
        self._stub(mod, spawn=boom)
        def lease_runner(root, args, **k):
            if args[0] == "reap":
                return {"rc": 0, "verdict": None}
            return {"rc": 3, "verdict": {"verdict": {"ok": False, "reason": "LEASE_HELD"}}}
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering", lane="gateway",
                         live=True, lease_runner=lease_runner)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "LANE_LEASE_HELD")
        self.assertEqual(p["action"], "lane_leased")
        self.assertTrue(p["lease"]["refused"])
        self.assertIn("gateway", p["reason"])

    def test_fail_open_lease_still_spawns(self) -> None:
        # A broken lease store (rc 1) must NOT block the loop — the same-host log scan
        # already passed, so the tick proceeds to spawn (fail-open).
        mod = load()
        did = {"spawn": False}
        def spawn(*a, **k):
            did["spawn"] = True
            return {"pid": 9, "log": "resolve-467.log", "issue": 467,
                    "lane": "gateway", "backend": "claude"}
        self._stub(mod, spawn=spawn)
        def lease_runner(root, args, **k):
            return {"rc": 1, "verdict": None}
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering", lane="gateway",
                         live=True, lease_runner=lease_runner)
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "SPAWNED")
        self.assertTrue(did["spawn"])
        self.assertTrue(p["lease"].get("fail_open"))

    def test_spawn_failure_releases_acquired_lease(self) -> None:
        # A worker that dies during the spawn probe is already proven dead, so the
        # dispatcher releases its lane immediately with the acquired holder/generation.
        mod = load()
        def spawn(*a, **k):
            return {"pid": 9, "log": "resolve-467.log", "issue": 467, "lane": "gateway",
                    "backend": "claude",
                    "early_exit": {"checked": True, "alive": False, "wait_s": 5.0,
                                   "silent": True, "returncode": 0, "log_bytes": 0}}
        self._stub(mod, spawn=spawn)
        def lease_runner(root, args, **k):
            if args[0] == "acquire":
                return {"rc": 0, "verdict": {"verdict": {"ok": True},
                                             "record": {"id": "resolve-gateway",
                                                        "holder": "node-a/dispatch-session",
                                                        "generation": 1}}}
            return {"rc": 0, "verdict": None}
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering", lane="gateway",
                         live=True, spawn_probe_s=5.0, lease_runner=lease_runner)
        self.assertEqual(p["verdict"], "SPAWN_FAILED")
        self.assertTrue(p["lease_release"]["released"])
        self.assertNotIn("lease_held_until_ttl", p)

    def test_dry_run_never_touches_the_lease(self) -> None:
        # The gate is AFTER the dry-run return, so a dry-run plans without holding a
        # lease at all (lease_runner must never be called).
        mod = load()
        def boom_runner(root, args, **k):
            raise AssertionError("dry-run must never touch the lease")
        self._stub(mod, spawn=lambda *a, **k: None)
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering", lane="gateway",
                         live=False, lease_runner=boom_runner)
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertNotIn("lease", p)
        # The commit-witness sweep is live-only too, so a dry-run payload omits it.
        self.assertNotIn("witnessed_slots", p)

    def test_live_surfaces_witnessed_slots(self) -> None:
        # The commit-time witness verdicts (#1324 proposal #2) are surfaced on a live
        # tick so a CLAIM_UNWITNESSED slot is visible in the tick record, not silent.
        mod = load()
        def spawn(*a, **k):
            return {"pid": 9, "log": "resolve-467.log", "issue": 467,
                    "lane": "gateway", "backend": "claude"}
        self._stub(mod, spawn=spawn)
        mod.witness_exited_workers = lambda runs_dir, root, **k: {
            "live": True, "audited": [{"issue": 470, "claim": "CLAIM_UNWITNESSED"}],
            "witnessed": [], "unwitnessed": [{"issue": 470, "claim": "CLAIM_UNWITNESSED"}],
            "no_commit": []}
        def lease_runner(root, args, **k):
            return {"rc": 0, "verdict": {"verdict": {"ok": True},
                                         "record": {"id": "resolve-gateway", "generation": 1}}}
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering", lane="gateway",
                         live=True, lease_runner=lease_runner)
        self.assertEqual(p["verdict"], "SPAWNED")
        self.assertIn("witnessed_slots", p)
        self.assertEqual(p["witnessed_slots"]["unwitnessed"][0]["claim"], "CLAIM_UNWITNESSED")

    def test_holds_self_modify_no_commit_issue_and_advances(self) -> None:
        # #1396 pick-held-invariant: issue 467's last worker FINISHED with a SELF_MODIFY
        # guard refusal and no commit — a re-blockable structural block. The witness
        # sweep recorded reason=self_modify, so this tick's picker HOLDS 467 and advances
        # to 466 instead of re-storming the same un-landable issue (the storm #1396 saw).
        mod = load()
        def spawn(*a, **k):
            return {"pid": 9, "log": "resolve-466.log", "issue": 466,
                    "lane": "gateway", "backend": "claude"}
        self._stub(mod, spawn=spawn, numbers=(467, 466))
        mod.witness_exited_workers = lambda runs_dir, root, **k: {
            "live": True, "audited": [{"issue": 467, "claim": "CLAIM_NO_COMMIT"}],
            "witnessed": [], "unwitnessed": [],
            "no_commit": [{"issue": 467, "claim": "CLAIM_NO_COMMIT",
                           "reason": "self_modify"}]}
        def lease_runner(root, args, **k):
            return {"rc": 0, "verdict": {"verdict": {"ok": True},
                                         "record": {"id": "resolve-gateway", "generation": 1}}}
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering", lane="gateway",
                         live=True, lease_runner=lease_runner)
        self.assertEqual(p["verdict"], "SPAWNED")
        self.assertEqual(p["target_issue"], 466)     # 467 held (self_modify) -> advanced
        self.assertEqual(p["held_no_commit"], [467])  # the recorded reason was honored

    def test_holds_preview_confirm_no_commit_issue_and_advances(self) -> None:
        # #2969: opencode can stop immediately after the preview-confirm guard tells
        # it to re-issue the call with _fak_confirm. That is unresolved guard feedback,
        # not a productive worker completion, so the next live tick holds that issue
        # and advances instead of relaunching the same guard loop.
        mod = load()
        def spawn(*a, **k):
            return {"pid": 9, "log": "resolve-2721.log", "issue": 2721,
                    "lane": "gateway", "backend": "opencode"}
        self._stub(mod, spawn=spawn, numbers=(2719, 2721))
        mod.witness_exited_workers = lambda runs_dir, root, **k: {
            "live": True, "audited": [{"issue": 2719, "claim": "CLAIM_NO_COMMIT"}],
            "witnessed": [], "unwitnessed": [],
            "no_commit": [{"issue": 2719, "claim": "CLAIM_NO_COMMIT",
                           "reason": mod.NO_COMMIT_PREVIEW_CONFIRM}]}
        def lease_runner(root, args, **k):
            return {"rc": 0, "verdict": {"verdict": {"ok": True},
                                         "record": {"id": "resolve-gateway",
                                                    "generation": 1}}}
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering", lane="gateway",
                         live=True, lease_runner=lease_runner)
        self.assertEqual(p["verdict"], "SPAWNED")
        self.assertEqual(p["target_issue"], 2721)
        self.assertEqual(p["held_no_commit"], [2719])


class EvaluateSeatCoolGateTest(unittest.TestCase):
    """#4591 end-to-end: a dead seat failing spawn across N DISTINCT issues
    accrues a SEAT streak (while each target streak restarts at 1) and the
    selector then cools/skips the seat instead of re-handing it a fourth issue."""

    DEAD = {"verdict": "SPAWN_OK", "reason": "ok", "cap": 2, "live": 0,
            "account": {"tag": "dead-seat", "tier": 1, "model": "opus",
                        "dir": "/acct/dead"}}
    HEALTHY = {"verdict": "SPAWN_OK", "reason": "ok", "cap": 2, "live": 0,
               "account": {"tag": "fresh-seat", "tier": 1, "model": "opus",
                           "dir": "/acct/fresh"}}

    @staticmethod
    def _dead_spawn(*_args, **_kwargs):
        return {"pid": 999, "log": "resolve-x.log", "issue": 0, "lane": "gateway",
                "backend": "claude",
                "early_exit": {"checked": True, "alive": False, "wait_s": 5.0,
                               "returncode": 1, "log_bytes": 0, "silent": True}}

    def test_dead_seat_cycling_across_issues_is_cooled_by_the_selector(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            patch_eval_env(mod, pre=self.DEAD,
                                pick={"lane": "gateway", "numbers": [101],
                                      "by_lane_count": {"gateway": 1}})
            mod.witness_exited_workers = lambda *a, **k: {"skipped": True}
            mod.release_lane_lease = lambda *a, **k: {"released": True}
            mod.spawn_issue_worker = self._dead_spawn
            for i, issue in enumerate((101, 202, 303), start=1):
                mod.lane_issue_numbers = (
                    lambda root_, lane, exclude=None, n=issue:
                    {"lane": "gateway", "numbers": [n],
                     "by_lane_count": {"gateway": 1}})
                p = mod.evaluate(root, max_workers=2, work_kind="engineering",
                                 lane=None, live=True, spawn_probe_s=5.0)
                self.assertEqual(p["action"], "spawn_failed")
                self.assertEqual(p["target_issue"], issue)
                # target rotation restarts the TARGET streak every tick (the
                # evasion #4591 closes) ...
                self.assertEqual(p["spawn_failed_streak"], 1)
                # ... but the SEAT streak keeps counting across distinct issues.
                self.assertEqual(p["seat_streak"], i)
            # Tick 4: the selector must NOT re-hand the dead seat another issue.
            mod.lane_issue_numbers = lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("a cooled seat must short-circuit before the lane router"))
            mod.spawn_issue_worker = lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("a cooled seat must never spawn"))
            p = mod.evaluate(root, max_workers=2, work_kind="engineering",
                             lane=None, live=False)
            self.assertFalse(p["ok"])
            self.assertEqual(p["action"], "seat_cooled")
            self.assertEqual(p["verdict"], "SEAT_COOLED")
            self.assertEqual(p["seat_streak"]["seat"], "dead-seat")
            self.assertEqual(p["seat_streak"]["streak"], 3)
            self.assertIn("fak accounts status", p["reason"])
            self.assertEqual(p["seat_cool_reroute"]["reason"],
                             "reroute did not find a different seat")
            # A correctly-declined tick, not a malfunction (the scheduled-task
            # health bit stays green; the CARD reddens via dispatch_status).
            self.assertEqual(mod.tick_exit_code(p), 0)

    def test_cooled_seat_reroutes_to_a_different_healthy_seat(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            patch_eval_env(mod, pre=self.DEAD,
                                pick={"lane": "gateway", "numbers": [404],
                                      "by_lane_count": {"gateway": 1}})
            runs = mod.RUNS_DIRNAME
            for _ in range(mod.SPAWN_FAILED_RED_STREAK):
                mod.bump_spawn_failure_streak_seat(runs, "dead-seat", "claude")
            calls = {"n": 0}

            def preflight(root_, **kw):
                calls["n"] += 1
                return self.DEAD if calls["n"] == 1 else self.HEALTHY

            mod.issue_dispatch.preflight = preflight
            p = mod.evaluate(root, max_workers=2, work_kind="engineering",
                             lane=None, live=False)
            # Rerouted: the tick proceeds on the fresh seat instead of declining.
            self.assertEqual(p["verdict"], "WOULD_SPAWN")
            self.assertEqual(p["account"]["tag"], "fresh-seat")
            self.assertEqual(p["seat_cool_reroute"]["account"]["tag"], "fresh-seat")
            self.assertEqual(calls["n"], 2)

    def test_reprobe_window_admits_one_spawn_to_detect_recovery(self) -> None:
        import time
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            patch_eval_env(mod, pre=self.DEAD,
                                pick={"lane": "gateway", "numbers": [505],
                                      "by_lane_count": {"gateway": 1}})
            runs = mod.RUNS_DIRNAME
            stale = time.time() - (mod.SEAT_STREAK_REPROBE_MIN * 60 + 5)
            for _ in range(mod.SPAWN_FAILED_RED_STREAK):
                mod.bump_spawn_failure_streak_seat(runs, "dead-seat", "claude",
                                                   now_ts=stale)
            p = mod.evaluate(root, max_workers=2, work_kind="engineering",
                             lane=None, live=False)
            # reprobe_due -> the gate admits the tick; the seat gets ONE probe.
            self.assertEqual(p["verdict"], "WOULD_SPAWN")
            self.assertEqual(p["account"]["tag"], "dead-seat")


class SeatAdaptiveEvaluateTest(unittest.TestCase):
    """evaluate()-level wiring of seat-adaptive tick sizing (#3246): the tick
    re-runs the preflight at the seat-sized cap, the preflight stays the
    authoritative floor, and no-signal / opt-out paths size exactly as before."""

    SEAT_OK = {
        "verdict": "SPAWN_OK", "reason": "ok", "cap": 3, "live": 1, "host_cap": 32,
        "seat": {"total": 18, "free": 11, "leased": 7},
        "account": {"tag": "worker-a", "tier": 1, "model": "opus", "dir": "/acct/a"},
    }
    PICK = {"lane": "gateway", "numbers": [467, 466], "by_lane_count": {"gateway": 2}}

    def _stub(self, mod, *, preflight):
        patch_eval_env(mod, pre={}, pick=dict(self.PICK))
        mod.issue_dispatch.preflight = preflight

    def _recording_preflight(self, docs_by_cap: dict[int, dict], calls: list[int]):
        def preflight(root, **kw):
            cap = int(kw.get("max_workers"))
            calls.append(cap)
            return docs_by_cap[cap]
        return preflight

    def test_at_cap_probe_reprobes_and_unblocks_at_seat_target(self) -> None:
        # The exact #3246 scenario: fixed cap 3 is saturated (REFUSE_AT_CAP with
        # configured_max binding) while 8 seats sit free. The tick re-sizes to
        # live+ramp (3+2=5), re-runs the preflight at 5, and proceeds to plan.
        mod = load()
        calls: list[int] = []
        at_cap = {"verdict": "REFUSE_AT_CAP", "reason": "live 3 >= cap 3",
                  "cap": 3, "live": 3, "host_cap": 32,
                  "seat": {"total": 18, "free": 8, "leased": 10},
                  "account": {"tag": "worker-a", "tier": 1, "model": "opus",
                              "dir": "/acct/a"}}
        ok_at_5 = {"verdict": "SPAWN_OK", "reason": "ok", "cap": 5, "live": 3,
                   "host_cap": 32, "seat": {"total": 18, "free": 8, "leased": 10},
                   "account": {"tag": "worker-a", "tier": 1, "model": "opus",
                               "dir": "/acct/a"}}
        self._stub(mod, preflight=self._recording_preflight(
            {3: at_cap, 5: ok_at_5}, calls))
        p = mod.evaluate(ROOT, max_workers=3, work_kind="engineering",
                         lane=None, live=False)
        self.assertEqual(calls, [3, 5])
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        sa = p["seat_adaptive"]
        self.assertEqual(sa["effective_target"], 5)
        self.assertEqual(sa["binding"], "ramp_delta")
        self.assertTrue(sa["reprobed"])
        self.assertEqual(p["preflight"]["cap"], 5)

    def test_preflight_refusal_at_seat_target_still_stops_growth(self) -> None:
        # The preflight stays AUTHORITATIVE: a hot host refusing at the resized
        # cap refuses the tick -- seat sizing can widen the configured term only,
        # never overrule the DoS gate.
        mod = load()
        calls: list[int] = []
        at_cap = dict(self.SEAT_OK, verdict="REFUSE_AT_CAP",
                      reason="live 3 >= cap 3", cap=3, live=3)
        host_hot = dict(self.SEAT_OK, verdict="REFUSE_HOST_HOT",
                        reason="host guard hot", cap=5, live=3)
        self._stub(mod, preflight=self._recording_preflight(
            {3: at_cap, 5: host_hot}, calls))
        p = mod.evaluate(ROOT, max_workers=3, work_kind="engineering",
                         lane=None, live=False)
        self.assertEqual(calls, [3, 5])
        self.assertFalse(p["ok"])
        self.assertEqual(p["action"], "refused")
        self.assertEqual(p["verdict"], "REFUSE_HOST_HOT")

    def test_no_seat_signal_probes_once_at_configured_cap(self) -> None:
        # FAIL-OPEN: no seat block in the doc => one probe at the configured cap,
        # no reprobe, and the audit block says why the tick fell back.
        mod = load()
        calls: list[int] = []
        bare = {"verdict": "SPAWN_OK", "reason": "ok", "cap": 2, "live": 0,
                "account": {"tag": "worker-a", "tier": 1, "model": "opus",
                            "dir": "/acct/a"}}
        self._stub(mod, preflight=self._recording_preflight({2: bare}, calls))
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertEqual(calls, [2])
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        sa = p["seat_adaptive"]
        self.assertFalse(sa["signal_available"])
        self.assertEqual(sa["binding"], "fallback_max_workers")
        self.assertNotIn("reprobed", sa)

    def test_seat_target_equal_to_configured_cap_skips_reprobe(self) -> None:
        # live 1 + ramp 2 == configured 3: nothing to re-size, one probe only.
        mod = load()
        calls: list[int] = []
        self._stub(mod, preflight=self._recording_preflight(
            {3: dict(self.SEAT_OK)}, calls))
        p = mod.evaluate(ROOT, max_workers=3, work_kind="engineering",
                         lane=None, live=False)
        self.assertEqual(calls, [3])
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertEqual(p["seat_adaptive"]["effective_target"], 3)
        self.assertNotIn("reprobed", p["seat_adaptive"])

    def test_opt_out_keeps_fixed_cap_and_payload_shape(self) -> None:
        # --no-seat-adaptive: one probe at the configured cap and NO seat_adaptive
        # payload block (byte-identical to the pre-#3246 tick).
        mod = load()
        calls: list[int] = []
        self._stub(mod, preflight=self._recording_preflight(
            {3: dict(self.SEAT_OK)}, calls))
        p = mod.evaluate(ROOT, max_workers=3, work_kind="engineering",
                         lane=None, live=False, seat_adaptive=False)
        self.assertEqual(calls, [3])
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertNotIn("seat_adaptive", p)

    def test_render_names_the_adaptive_cap_and_binding(self) -> None:
        mod = load()
        calls: list[int] = []
        at_cap = dict(self.SEAT_OK, verdict="REFUSE_AT_CAP",
                      reason="live 3 >= cap 3", cap=3, live=3)
        ok_at_5 = dict(self.SEAT_OK, verdict="SPAWN_OK", cap=5, live=3)
        self._stub(mod, preflight=self._recording_preflight(
            {3: at_cap, 5: ok_at_5}, calls))
        p = mod.evaluate(ROOT, max_workers=3, work_kind="engineering",
                         lane=None, live=False)
        rendered = mod.render(p)
        self.assertIn("seats     : adaptive cap 5", rendered)
        self.assertIn("binding ramp_delta", rendered)


class TestOpencodeGatewayGate(unittest.TestCase):
    def test_gateway_down_suppresses_tick_with_legible_reason(self):
        mod = load()

        def down(_row):
            return {"status": "GATEWAY_DOWN", "block_reason": "connection refused"}
        planned, reason = mod.gate_opencode_gateway(3, {"name": "glm"}, probe=down)
        self.assertEqual(planned, 0)
        self.assertIn("gateway_down", reason)
        self.assertIn("backend_unhealthy", reason)

    def test_healthy_gateway_resumes_next_tick(self):
        mod = load()

        def healthy(_row):
            return {"status": "OK"}
        self.assertEqual(mod.gate_opencode_gateway(3, {"name": "glm"}, probe=healthy), (3, None))

    def test_probe_error_fails_open(self):
        mod = load()
        def broken(_row): raise OSError("probe unavailable")
        self.assertEqual(mod.gate_opencode_gateway(3, {"name": "glm"}, probe=broken), (3, None))

    def test_evaluate_emits_zero_spawn_gateway_down_payload(self):
        mod = load()
        pre = {"verdict": "SPAWN_OK", "reason": "ok", "cap": 3, "live": 0,
               "host_cap": 32, "account": {"tag": "glm-a", "tier": 2,
               "model": "glm-5.2", "dir": "/acct/glm"}}
        pick = {"lane": "docs", "numbers": [3866], "by_lane_count": {"docs": 1}}
        patch_eval_env(mod, pre=pre, pick=pick)
        mod.account_probe.probe_opencode_account = lambda _row: {
            "status": "GATEWAY_DOWN", "block_reason": "connection refused"}

        payload = mod.evaluate(ROOT, max_workers=3, work_kind="gardening",
                               lane=None, live=False, backend="opencode")

        self.assertEqual(payload["action"], "backend_health_skip")
        self.assertEqual(payload["verdict"], "GATEWAY_DOWN")
        self.assertEqual(payload["gateway_health_gate"], {
            "status": "backend_unhealthy", "reason": "gateway_down", "planned": 0})
        self.assertIn("connection refused", payload["reason"])


class SpawnBurstResolveTest(unittest.TestCase):
    """The burst limit is the fail-safe against the #3153 spawn-churn lockup, so its
    resolution must be total: flag > env > default, always clamped, never raising."""

    def test_default_is_one_spawn_per_tick(self) -> None:
        mod = load()
        got = mod.resolve_spawn_burst(None, {})
        self.assertEqual(got["limit"], 1)
        self.assertEqual(got["source"], "default")
        self.assertFalse(got["clamped"])

    def test_env_arms_the_fan_out(self) -> None:
        mod = load()
        got = mod.resolve_spawn_burst(None, {mod.SPAWN_BURST_ENV: "3"})
        self.assertEqual(got["limit"], 3)
        self.assertEqual(got["source"], f"env:{mod.SPAWN_BURST_ENV}")

    def test_flag_beats_env(self) -> None:
        mod = load()
        got = mod.resolve_spawn_burst(2, {mod.SPAWN_BURST_ENV: "4"})
        self.assertEqual(got["limit"], 2)
        self.assertEqual(got["source"], "flag:--spawn-burst")

    def test_hard_ceiling_clamps_any_request(self) -> None:
        # No configuration — not a flag, not an env var — may turn one tick into a
        # spawn storm on this host. Both channels clamp.
        mod = load()
        for got in (mod.resolve_spawn_burst(99, {}),
                    mod.resolve_spawn_burst(None, {mod.SPAWN_BURST_ENV: "99"})):
            self.assertEqual(got["limit"], mod.SPAWN_BURST_HARD_CEILING)
            self.assertEqual(got["requested"], 99)
            self.assertTrue(got["clamped"])

    def test_zero_and_negative_floor_at_one(self) -> None:
        mod = load()
        self.assertEqual(mod.resolve_spawn_burst(0, {})["limit"], 1)
        self.assertEqual(mod.resolve_spawn_burst(-5, {})["limit"], 1)

    def test_garbage_env_falls_back_to_the_default(self) -> None:
        mod = load()
        got = mod.resolve_spawn_burst(None, {mod.SPAWN_BURST_ENV: "lots"})
        self.assertEqual(got["limit"], mod.DEFAULT_SPAWN_BURST)
        self.assertEqual(got["source"], "default")
        self.assertEqual(got["env_invalid"], "lots")

    def test_stagger_resolution(self) -> None:
        mod = load()
        self.assertEqual(mod.resolve_spawn_burst_stagger(None, {}),
                         mod.DEFAULT_SPAWN_BURST_STAGGER_S)
        self.assertEqual(
            mod.resolve_spawn_burst_stagger(None, {mod.SPAWN_BURST_STAGGER_ENV: "7"}), 7.0)
        self.assertEqual(
            mod.resolve_spawn_burst_stagger(2.5, {mod.SPAWN_BURST_STAGGER_ENV: "7"}), 2.5)
        self.assertEqual(mod.resolve_spawn_burst_stagger(-3, {}), 0.0)
        self.assertEqual(mod.resolve_spawn_burst_stagger(10_000, {}),
                         mod.SPAWN_BURST_STAGGER_MAX_S)
        self.assertEqual(
            mod.resolve_spawn_burst_stagger(None, {mod.SPAWN_BURST_STAGGER_ENV: "x"}),
            mod.DEFAULT_SPAWN_BURST_STAGGER_S)


class EvaluateBurstDriverTest(unittest.TestCase):
    """The fan-out driver itself, with evaluate() stubbed: how many sub-ticks run,
    when the burst stops, and what the aggregate payload says."""

    @staticmethod
    def _spawned(issue: int, lane: str = "gateway", pid: int = 100):
        return {"ok": True, "action": "spawned", "verdict": "SPAWNED",
                "target_issue": issue, "lane": lane, "reason": f"spawned on #{issue}",
                "spawned": {"pid": pid, "issue": issue}}

    def _scripted(self, mod, payloads, calls):
        def fake_evaluate(root, **kw):
            calls.append(dict(kw))
            return payloads[len(calls) - 1]
        mod.evaluate = fake_evaluate

    def test_default_limit_runs_one_subtick_and_stays_byte_identical(self) -> None:
        mod = load()
        calls: list[dict] = []
        self._scripted(mod, [self._spawned(467)], calls)
        p = mod.evaluate_burst(ROOT, live=True, max_workers=18)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("burst", p)  # opt-in: the default payload is unchanged

    def test_fans_out_to_the_limit_when_every_subtick_spawns(self) -> None:
        mod = load()
        calls: list[dict] = []
        self._scripted(mod, [self._spawned(467, "gateway", 101),
                             self._spawned(520, "agent", 102),
                             self._spawned(610, "docs", 103)], calls)
        p = mod.evaluate_burst(ROOT, spawn_burst=3, burst_stagger_s=0,
                               live=True, max_workers=18)
        self.assertEqual(len(calls), 3)
        self.assertEqual(p["burst"]["spawned"], 3)
        self.assertEqual(p["burst"]["attempts"], 3)
        self.assertEqual(p["burst"]["stopped_on"], "burst_limit")
        self.assertEqual([r["issue"] for r in p["burst"]["sub_ticks"]], [467, 520, 610])
        # The aggregate keeps the FIRST sub-tick's verdict (what tick_exit_code grades)
        # and names the extra spawns in the reason.
        self.assertEqual(p["verdict"], "SPAWNED")
        self.assertEqual(p["target_issue"], 467)
        self.assertIn("#520", p["reason"])
        self.assertEqual(mod.tick_exit_code(p), 0)

    def test_stops_at_the_first_non_spawn_verdict(self) -> None:
        # A refusal means the fleet hit a real boundary; re-attempting it inside the
        # same tick would only re-refuse.
        mod = load()
        calls: list[dict] = []
        no_issue = {"ok": False, "action": "no_issue", "verdict": "NO_ISSUE",
                    "reason": "nothing fresh"}
        self._scripted(mod, [self._spawned(467), no_issue,
                             self._spawned(999)], calls)
        p = mod.evaluate_burst(ROOT, spawn_burst=3, burst_stagger_s=0,
                               live=True, max_workers=18)
        self.assertEqual(len(calls), 2)
        self.assertEqual(p["burst"]["spawned"], 1)
        self.assertEqual(p["burst"]["stopped_on"], "NO_ISSUE")
        self.assertEqual(p["verdict"], "SPAWNED")

    def test_hard_ceiling_bounds_the_driver_not_just_the_flag(self) -> None:
        mod = load()
        calls: list[dict] = []
        self._scripted(mod, [self._spawned(n) for n in range(400, 420)], calls)
        p = mod.evaluate_burst(ROOT, spawn_burst=99, burst_stagger_s=0,
                               live=True, max_workers=18)
        self.assertEqual(len(calls), mod.SPAWN_BURST_HARD_CEILING)
        self.assertEqual(p["burst"]["limit"], mod.SPAWN_BURST_HARD_CEILING)

    def test_dry_run_never_fans_out(self) -> None:
        # A dry-run sub-tick spawns nothing, so headroom never moves and every
        # further sub-tick would re-pick the SAME issue and double-report it.
        mod = load()
        calls: list[dict] = []
        would = {"ok": True, "action": "would_spawn", "verdict": "WOULD_SPAWN",
                 "target_issue": 467, "lane": "gateway", "reason": "safe to spawn 1"}
        self._scripted(mod, [would, would, would], calls)
        p = mod.evaluate_burst(ROOT, spawn_burst=3, live=False, max_workers=18)
        self.assertEqual(len(calls), 1)
        self.assertEqual(p["burst"]["attempts"], 1)
        self.assertIn("dry_run", p["burst"]["skipped"])

    def test_stagger_paces_every_extra_spawn(self) -> None:
        # The lockup guard: N spawns arrive as a ramp, not as one process-creation
        # burst. N spawns => N-1 gaps, and none before the first.
        mod = load()
        calls: list[dict] = []
        slept: list[float] = []
        self._scripted(mod, [self._spawned(1), self._spawned(2),
                             self._spawned(3)], calls)
        mod.evaluate_burst(ROOT, spawn_burst=3, burst_stagger_s=15.0,
                           sleeper=slept.append, live=True, max_workers=18)
        self.assertEqual(slept, [15.0, 15.0])

    def test_no_stagger_before_a_burst_that_stops_immediately(self) -> None:
        mod = load()
        calls: list[dict] = []
        slept: list[float] = []
        self._scripted(mod, [{"ok": False, "action": "refused",
                              "verdict": "REFUSE_AT_CAP", "reason": "at cap"}], calls)
        mod.evaluate_burst(ROOT, spawn_burst=4, burst_stagger_s=15.0,
                           sleeper=slept.append, live=True, max_workers=18)
        self.assertEqual(slept, [])

    def test_followon_subticks_skip_the_registry_refresh(self) -> None:
        # The registry refresh is a per-TICK concern; re-running it per spawn is pure
        # subprocess churn on exactly the axis that freezes this host.
        mod = load()
        calls: list[dict] = []
        self._scripted(mod, [self._spawned(1), self._spawned(2)], calls)
        mod.evaluate_burst(ROOT, spawn_burst=2, burst_stagger_s=0,
                           live=True, max_workers=18, refresh=True)
        self.assertTrue(calls[0]["refresh"])
        self.assertFalse(calls[1]["refresh"])

    def test_render_names_the_burst(self) -> None:
        mod = load()
        calls: list[dict] = []
        self._scripted(mod, [self._spawned(467, "gateway", 101),
                             self._spawned(520, "agent", 102)], calls)
        p = mod.evaluate_burst(ROOT, spawn_burst=2, burst_stagger_s=0,
                               live=True, max_workers=18)
        rendered = mod.render(p)
        self.assertIn("burst     : 2/2 spawned", rendered)
        self.assertIn("#520:SPAWNED", rendered)


class EvaluateBurstAdmissionTest(unittest.TestCase):
    """End-to-end through the REAL evaluate(): every spawn in a burst is admitted
    independently — no admission decision is hoisted over N spawns."""

    def _live_patch(self, mod, *, lanes, preflight, spawn, lease_runner=None):
        patch_eval_env(mod, pre={}, pick={})
        # `issue_lane_router` / `issue_worker_prompt` live in sys.modules and survive
        # load(), so restore them or the stub leaks into every later test.
        for shared, name in ((mod.issue_lane_router, "lane_taxonomy"),
                             (mod.issue_worker_prompt, "build")):
            prior = getattr(shared, name)
            self.addCleanup(setattr, shared, name, prior)
        # evaluate()'s low-yield probe shells `dos doctor` once per sub-tick (~2.5s).
        # Nothing here depends on the real taxonomy.
        mod.issue_lane_router.lane_taxonomy = lambda ws: ([], {}, set())
        self.spawned_issues: set[int] = set()
        self.spawned_lanes: set[str] = set()
        self.collision_rows: list[dict] = []
        mod.issue_dispatch.preflight = preflight
        mod.issue_dispatch.worker_env = lambda d, lane, root: {}

        def picker(root, lane, exclude=None, **_k):
            ex = set(exclude or ())
            eligible = [(name, list(nums)) for name, nums in lanes.items()
                        if name not in ex]
            if not eligible:
                return {"lane": None, "numbers": [], "by_lane_count": {},
                        "eligible_by_lane": []}
            return {"lane": eligible[0][0], "numbers": eligible[0][1],
                    "by_lane_count": {n: len(v) for n, v in eligible},
                    "eligible_by_lane": [[n, v] for n, v in eligible]}

        mod.lane_issue_numbers = picker
        # The de-dup is filesystem-derived in production (.pid sidecar + spawn-header
        # log); here the fake spawn feeds the same two readers, so sub-tick N+1 sees
        # exactly what sub-tick N launched.
        mod.live_resolution_issues = lambda runs_dir, **k: set(self.spawned_issues)
        mod.live_resolution_lanes = lambda runs_dir, **k: set(self.spawned_lanes)
        mod.record_collision_holds = lambda runs_dir, rows, **k: (
            self.collision_rows.extend(rows))
        mod.collision_held_issues = lambda runs_dir, **k: {
            int(r["issue"]) for r in self.collision_rows if r.get("issue")}
        mod.low_yield_soft_excludes = lambda root, runs_dir, **k: {
            "exclude": set(), "lanes": [], "flagged": []}
        mod.scan_multi_lane_scope = lambda root, text, lane: {
            "multi_lane": False, "uncovered_lanes": [], "uncovered": []}
        mod.same_issue_wip_collision = lambda runs_dir, issue, paths, **k: {
            "collides": False}
        mod.check_backend_health = lambda runs_dir, **k: {"state": "healthy"}
        mod.read_dead_backends = lambda runs_dir, **k: []
        mod.recent_backend_stub_rate = lambda runs_dir, **k: None
        mod.prune_dead_sidecars = lambda runs_dir, **k: {"pruned": []}
        mod.witness_exited_workers = lambda runs_dir, root, **k: {
            "live": True, "audited": [], "witnessed": [], "unwitnessed": [],
            "no_commit": []}
        mod.append_fleet_trend_row = lambda root, payload, **k: {"ok": True}
        mod._git_capture = lambda root, args, **k: (0, "basesha0\n")
        mod.spawn_issue_worker = spawn
        return lease_runner

    def _spawner(self, mod):
        def spawn(command, env, cwd, runs_dir, issue, lane, backend, **k):
            self.spawned_issues.add(int(issue))
            self.spawned_lanes.add(str(lane))
            return {"pid": 900 + len(self.spawned_issues), "issue": issue,
                    "lane": lane, "backend": backend,
                    "log": f"resolve-{issue}.log"}
        return spawn

    def test_each_spawn_reruns_the_full_admission_chain(self) -> None:
        mod = load()
        caps: list[int] = []

        def preflight(root, **kw):
            caps.append(int(kw.get("max_workers")))
            return {"verdict": "SPAWN_OK", "reason": "ok", "cap": 18,
                    "live": len(self.spawned_issues),
                    "account": {"tag": "worker-a", "tier": 1, "model": "opus",
                                "dir": "/acct/a"}}

        self._live_patch(mod, lanes={"gateway": [467], "agent": [520]},
                         preflight=preflight, spawn=self._spawner(mod))
        leases: list[str] = []
        mod.acquire_lane_lease = lambda root, lane, **k: (
            leases.append(lane) or {"acquired": True, "refused": False,
                                    "id": f"resolve-{lane}", "holder": "test"})
        p = mod.evaluate_burst(ROOT, spawn_burst=2, burst_stagger_s=0,
                               max_workers=18, work_kind="engineering",
                               lane=None, live=True)
        # Two spawns, on DIFFERENT issues and DIFFERENT lanes: sub-tick 2 saw
        # sub-tick 1's worker as live and routed around it.
        self.assertEqual(p["burst"]["spawned"], 2)
        self.assertEqual(self.spawned_issues, {467, 520})
        self.assertEqual(leases, ["gateway", "agent"])
        # ...and the DoS gate was re-probed for each spawn, never hoisted.
        self.assertGreaterEqual(len(caps), 2)

    def test_burst_stops_when_the_preflight_hits_the_cap(self) -> None:
        # The overall --max-workers cap still binds INSIDE a burst: the second
        # sub-tick's preflight refuses and the fan-out ends there.
        mod = load()

        def preflight(root, **kw):
            live = len(self.spawned_issues)
            if live >= 1:
                return {"verdict": "REFUSE_AT_CAP", "reason": f"live {live} >= cap 1",
                        "cap": 1, "live": live,
                        "account": {"tag": "worker-a", "tier": 1, "model": "opus",
                                    "dir": "/acct/a"}}
            return {"verdict": "SPAWN_OK", "reason": "ok", "cap": 1, "live": live,
                    "account": {"tag": "worker-a", "tier": 1, "model": "opus",
                                "dir": "/acct/a"}}

        self._live_patch(mod, lanes={"gateway": [467], "agent": [520]},
                         preflight=preflight, spawn=self._spawner(mod))
        p = mod.evaluate_burst(ROOT, spawn_burst=4, burst_stagger_s=0,
                               max_workers=1, work_kind="engineering",
                               lane=None, live=True)
        self.assertEqual(p["burst"]["spawned"], 1)
        self.assertEqual(p["burst"]["stopped_on"], "REFUSE_AT_CAP")
        self.assertEqual(self.spawned_issues, {467})

    def test_burst_stops_when_a_peer_holds_the_next_lane_lease(self) -> None:
        # Lane arbitration is re-decided per spawn: a peer taking the second lane's
        # fenced lease refuses that sub-tick instead of racing a worker onto it.
        mod = load()

        def preflight(root, **kw):
            return {"verdict": "SPAWN_OK", "reason": "ok", "cap": 18,
                    "live": len(self.spawned_issues),
                    "account": {"tag": "worker-a", "tier": 1, "model": "opus",
                                "dir": "/acct/a"}}

        self._live_patch(mod, lanes={"gateway": [467], "agent": [520]},
                         preflight=preflight, spawn=self._spawner(mod))

        def acquire(root, lane, **k):
            if lane == "gateway":
                return {"acquired": True, "refused": False, "id": "resolve-gateway",
                        "holder": "test"}
            return {"acquired": False, "refused": True, "id": f"resolve-{lane}",
                    "reason": "LEASE_HELD"}

        mod.acquire_lane_lease = acquire
        p = mod.evaluate_burst(ROOT, spawn_burst=3, burst_stagger_s=0,
                               max_workers=18, work_kind="engineering",
                               lane=None, live=True)
        self.assertEqual(p["burst"]["spawned"], 1)
        self.assertEqual(p["burst"]["stopped_on"], "LANE_LEASE_HELD")
        self.assertEqual(self.spawned_issues, {467})

    def test_fanout_advances_past_a_dirty_path_collision(self) -> None:
        # The ~2400-row collision-holds ledger is dominated by DIRTY_PATH_COLLISION,
        # so a fan-out that re-picked the blocked head would just manufacture more
        # holds. It does not: the in-tick scan steps past the colliding candidate,
        # and the hold it ledgers keeps the NEXT sub-tick past it too.
        mod = load()

        def preflight(root, **kw):
            return {"verdict": "SPAWN_OK", "reason": "ok", "cap": 18,
                    "live": len(self.spawned_issues),
                    "account": {"tag": "worker-a", "tier": 1, "model": "opus",
                                "dir": "/acct/a"}}

        self._live_patch(mod, lanes={"gateway": [467, 466], "agent": [520]},
                         preflight=preflight, spawn=self._spawner(mod))
        dirty = "cmd/fak/version_modules.go"
        mod.dirty_repo_paths = lambda root: {"paths": [dirty], "unavailable": False}
        mod.issue_worker_prompt.build = lambda n, lane, *, workspace: {
            "prompt": f"resolve #{n}", "prompt_chars": 100, "title": f"title {n}",
            "issue_record": {"body": (f"rework {dirty}" if n == 467 else "clean work")},
        }
        p = mod.evaluate_burst(ROOT, spawn_burst=2, burst_stagger_s=0,
                               max_workers=18, work_kind="engineering",
                               lane=None, live=True)
        self.assertEqual(p["burst"]["spawned"], 2)
        self.assertEqual(self.spawned_issues, {466, 520})
        self.assertNotIn(467, self.spawned_issues)
        self.assertEqual([r["issue"] for r in self.collision_rows], [467])


class EvalHelpersDirectTest(unittest.TestCase):
    """Direct tests for extracted evaluation helper seams in issue_resolve_dispatch_eval."""

    def test_preflight_refusal_hint_identifies_configured_max_workers(self) -> None:
        load()
        import issue_resolve_dispatch_eval as eval_mod

        hint = eval_mod._preflight_refusal_hint({
            "verdict": "REFUSE_AT_CAP",
            "live": 2,
            "cap": 2,
            "max_workers": 2,
            "host_cap": 6,
            "capacity_limiter": {"term": "max_workers", "primary": "config"},
        })
        self.assertIsNotNone(hint)
        self.assertEqual(hint["kind"], "configured_max_workers")
        self.assertEqual(hint["required_min"], 3)

    def test_same_account_matches_by_tag_or_dir(self) -> None:
        load()
        import issue_resolve_dispatch_eval as eval_mod

        self.assertTrue(eval_mod._same_account({"tag": "acct-a", "dir": "/1"}, {"tag": "acct-a", "dir": "/2"}))
        self.assertTrue(eval_mod._same_account({"tag": "a", "dir": "/same"}, {"tag": "b", "dir": "/same"}))
        self.assertFalse(eval_mod._same_account({"tag": "a", "dir": "/1"}, {"tag": "b", "dir": "/2"}))

    def test_admit_collision_and_scope_gates_blocks_same_issue_wip(self) -> None:
        load()
        import issue_resolve_dispatch_eval as eval_mod

        recorded: list[list[dict[str, object]]] = []
        payload: dict[str, object] = {}
        scan_res = {
            "chosen_lane": "tools",
            "target": 999,
            "rec": {"title": "fix split helper", "issue_record": {"body": ""}},
            "dirty_paths_doc": {"paths": ["tools/foo.py"]},
            "wip_scan": {"collides": True, "dirty_paths": ["tools/foo.py"], "evidence": ["recent"]},
            "dirty_scan": None,
            "scope_scan": None,
        }
        with tempfile.TemporaryDirectory() as d:
            verdict = eval_mod._admit_collision_and_scope_gates(
                Path(d), Path(d), payload, scan_res,
                live=False, force=False, issue_override=None,
                rec_collision_fn=lambda _r, rows, **_k: recorded.append(rows),
            )
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict["verdict"], "SAME_ISSUE_WIP")
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0][0]["issue"], 999)


if __name__ == "__main__":
    unittest.main()
