#!/usr/bin/env python3
"""Hermetic tests for tools/issue_resolve_dispatch.py.

Every shell-out (registry refresh, preflight, lane router, prompt build, spawn)
is stubbed on the module; NOTHING live (gh/dos/claude) runs and no worker is
spawned in dry-run. The pure pickers (pick_target_issue, lane fold) are tested
directly.
"""
from __future__ import annotations


import atexit
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "issue_resolve_dispatch.py"


def _no_window_creationflags() -> int:
    """``creationflags`` that stop a console child (the git shell-outs in the
    hermetic fixtures below) from popping a visible window when this test runs
    windowless on Windows; ``0`` on POSIX. Mirrors
    dispatch_worker.no_window_creationflags, kept local so this test imports only
    stdlib."""
    return 0x08000000 if os.name == "nt" else 0


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("issue_resolve_dispatch", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    runs_tmp = Path(tempfile.mkdtemp(prefix="issue-resolve-test-runs-"))
    atexit.register(shutil.rmtree, runs_tmp, ignore_errors=True)
    mod.RUNS_DIRNAME = runs_tmp
    return mod


def passing_issue_contract(*_args, **_kwargs):
    return {
        "ok": True,
        "unavailable": False,
        "score": 100,
        "spine_priority": 100,
        "review": {
            "ok": True,
            "reasons": [],
            "score": {"total": 100},
            "spine_priority": {"total": 100},
        },
    }



def patch_eval_env(mod, *, pre, pick, live_issues=None, cooled=None,
                   held_lanes=None, prompt_chars=900) -> None:
    mod.issue_dispatch.refresh_registry = lambda root: {"ok": True}
    mod.issue_dispatch.preflight = lambda root, **kw: pre
    mod.lane_issue_numbers = lambda root, lane, exclude=None: pick
    mod.live_resolution_issues = lambda runs_dir: set(live_issues or [])
    mod.live_resolution_lanes = lambda runs_dir: set(held_lanes or [])
    mod.live_lane_lease_lanes = lambda root: {"lanes": []}
    mod.recently_attempted_issues = lambda runs_dir, *, cooldown_min, **k: set(cooled or [])
    mod.locally_witnessed_issues = lambda root, **k: set()
    mod.commit_audit_abstain_holds = lambda root, candidates, **k: []
    mod.open_witnessed_dispositions = lambda root, candidates, **k: []
    mod.issue_worker_prompt.build = lambda n, lane, *, workspace: {
        "prompt": f"resolve #{n}", "prompt_chars": prompt_chars, "title": f"title {n}"}
    mod.issue_contract_review = passing_issue_contract
    mod.dirty_repo_paths = lambda root: {"paths": [], "unavailable": False}
    # Hermetic contract-hold ledger: no prior holds, and never write the real
    # .dispatch-runs ledger from a test tick (live hold ticks append to it).
    mod.contract_held_issues = lambda runs_dir, **k: set()
    mod.contract_held_records = lambda runs_dir, **k: []
    mod.record_contract_holds = lambda runs_dir, rows, **k: None
    mod.multi_lane_held_issues = lambda runs_dir, **k: set()
    mod.multi_lane_held_records = lambda runs_dir, **k: []
    mod.record_multi_lane_holds = lambda runs_dir, rows, **k: None
    mod.collision_held_issues = lambda runs_dir, **k: set()
    mod.collision_held_records = lambda runs_dir, **k: []
    mod.record_collision_holds = lambda runs_dir, rows, **k: None
    # Hermetic next()-queue seams: no real gh updatedAt fetch, no live repair
    # worker, no repair cooldown, and a pure repair-prompt fold.
    mod.open_issue_updated_map = lambda root, **k: {}
    mod.live_repair_workers = lambda runs_dir, **k: []
    mod.recently_repaired_issues = lambda runs_dir, *, cooldown_min, **k: set()
    mod.issue_worker_prompt.build_repair = (
        lambda rows, *, workspace, min_score=100: {
            "kind": "contract-repair",
            "issues": [int(r.get("number") or 0) for r in rows],
            "prompt": "repair " + ",".join(str(r.get("number")) for r in rows),
            "prompt_chars": 10})
    mod.check_weekly_cap = lambda runs_dir, **k: {"capped": False}  # hermetic: not capped
    mod.active_guard_livelock_hold = lambda root, runs_dir, **k: {"active": False}
    mod.active_compact_runaway_hold = lambda root, runs_dir, **k: {"active": False}
    mod.reap_timed_out_workers = lambda runs_dir, **k: {
        "timeout_s": k.get("timeout_s"), "live": k.get("live"),
        "candidates": [], "reaped": [], "would_reap": []}
    # Hermetic lease layer (#1310 residual): neutralize the fenced-lease helpers
    # so a live tick never shells a real `fak leaseref` / `dos` (which would touch
    # real git refs and leak a refs/fak/locks lease into the repo). The default is
    # ACQUIRE (the lane is free) so the live path proceeds to spawn exactly as
    # before; a test that wants the held-lane path overrides acquire_lane_lease.
    mod.acquire_lane_lease = lambda root, lane, **k: {
        "acquired": True, "refused": False, "id": f"resolve-{lane}",
        "holder": "test", "tree": k.get("tree") or []}
    mod.reap_expired_leases = lambda root, **k: {"ok": True, "rc": 0}
    mod.lane_tree = lambda root, lane: [f"internal/{lane}/**"]

    def boom(*a, **k):
        raise AssertionError("dry-run must never spawn")
    mod.spawn_issue_worker = boom


class EvaluateTest(unittest.TestCase):
    SPAWN_OK = {
        "verdict": "SPAWN_OK", "reason": "ok", "cap": 2, "live": 0,
        "account": {"tag": "worker-a", "tier": 1, "model": "opus", "dir": "/acct/a"},
    }


    def _patch(self, mod, *, pre, pick, live_issues=None, cooled=None,
               held_lanes=None, prompt_chars=900) -> None:
        patch_eval_env(mod, pre=pre, pick=pick, live_issues=live_issues,
                       cooled=cooled, held_lanes=held_lanes,
                       prompt_chars=prompt_chars)

    def test_would_spawn_picks_top_unblocked_issue(self) -> None:
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467, 466, 452],
                          "by_lane_count": {"gateway": 3}})
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertEqual(p["lane"], "gateway")
        self.assertEqual(p["target_issue"], 467)
        self.assertIn("467", p["reason"])

    def test_live_tick_appends_fleet_trend_row(self) -> None:
        import json
        mod = load()
        refuse = {
            "verdict": "REFUSE_AT_CAP", "reason": "at cap", "cap": 2, "live": 2,
            "account": {"tag": "worker-a", "tier": 1, "model": "opus", "dir": "/acct/a"},
        }
        self._patch(mod, pre=refuse,
                    pick={"lane": "gateway", "numbers": [467],
                          "by_lane_count": {"gateway": 1}})
        mod.witness_exited_workers = lambda *a, **k: {"skipped": True}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = mod.evaluate(root, max_workers=2, work_kind="engineering",
                             lane=None, live=True)
            self.assertEqual(p["action"], "refused")
            ft = p.get("fleet_trend") or {}
            self.assertTrue(ft.get("ok"), msg=str(ft))
            ledger = root / ".fak" / "nightrun" / "fleet-status-history.jsonl"
            self.assertTrue(ledger.exists())
            row = json.loads(
                ledger.read_text(encoding="utf-8").strip().splitlines()[-1])
            self.assertEqual(row["live"], 2.0)
            self.assertIn("ts", row)

    def test_dry_run_tick_never_writes_fleet_trend(self) -> None:
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467],
                          "by_lane_count": {"gateway": 1}})
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = mod.evaluate(root, max_workers=2, work_kind="engineering",
                             lane=None, live=False)
            self.assertNotIn("fleet_trend", p)
            self.assertFalse(
                (root / ".fak" / "nightrun" / "fleet-status-history.jsonl").exists())

    def test_would_spawn_unguarded_is_not_launch_ready(self) -> None:
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467],
                          "by_lane_count": {"gateway": 1}})
        mod.dispatch_worker.guarded_launch_command = (
            lambda command, lane, backend, root, env: (list(command), False))

        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)

        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertFalse(p["guarded"])
        self.assertFalse(p["launch_gate"]["ready"])
        self.assertEqual(p["launch_gate"]["blockers"][0]["code"], "UNGUARDED_WORKER")
        rendered = mod.render(p)
        self.assertIn("NOT launch-ready", rendered)
        self.assertNotIn("re-run with --live", rendered)

    def test_guarded_would_spawn_is_launch_ready(self) -> None:
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467],
                          "by_lane_count": {"gateway": 1}})
        mod.dispatch_worker.guarded_launch_command = (
            lambda command, lane, backend, root, env: (["fak", "guard", "--", *command], True))

        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)

        self.assertTrue(p["launch_gate"]["ready"])
        self.assertEqual(p["launch_gate"]["blockers"], [])
        self.assertIn("re-run with --live", mod.render(p))

    def test_issue_contract_hold_blocks_spawn(self) -> None:
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467],
                          "by_lane_count": {"gateway": 1}})
        mod.issue_contract_review = lambda *a, **k: {
            "ok": False,
            "unavailable": False,
            "score": 40,
            "spine_priority": 0,
            "review": {
                "ok": False,
                "reasons": ["ISSUE_SCOPE_INCOMPLETE"],
                "missing_fields": ["working_spine"],
                "score": {"total": 40},
            },
        }
        # repair_batch=0: this test pins the plain HOLD path (with repair enabled
        # an all-thin tick dispatches a contract-repair worker instead — covered
        # by the ContractRepair tests below).
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=True, repair_batch=0)
        self.assertFalse(p["ok"])
        self.assertEqual(p["action"], "issue_contract_hold")
        self.assertEqual(p["verdict"], "ISSUE_CONTRACT_HOLD")
        self.assertIn("ISSUE_SCOPE_INCOMPLETE", p["reason"])
        self.assertEqual(p["issue_contract_gate"]["score"], 40)

    def test_contract_hold_scans_to_next_ready_issue(self) -> None:
        """A THIN head issue no longer wedges the tick: the pick scans forward
        (oldest-first order preserved) to the oldest contract-READY issue and
        spawns that, recording the skipped heads transparently."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467, 466, 452],
                          "by_lane_count": {"gateway": 3}})

        def per_issue_review(root, issue, number, **_kw):
            if number == 467:  # the thin head that used to hold every tick
                return {"ok": False, "unavailable": False, "score": 8,
                        "spine_priority": 0,
                        "review": {"ok": False, "reasons": ["ISSUE_SCOPE_INCOMPLETE"],
                                   "missing_fields": ["working_spine"],
                                   "score": {"total": 8}}}
            return passing_issue_contract()

        mod.issue_contract_review = per_issue_review
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertEqual(p["target_issue"], 466)
        self.assertEqual([r["issue"] for r in p["contract_skipped"]], [467])
        self.assertIn("ISSUE_SCOPE_INCOMPLETE", p["contract_skipped"][0]["reason"])
        # The recorded gate is the CHOSEN issue's (passing), not the skipped head's.
        self.assertTrue(p["issue_contract_gate"]["ok"])

    def test_multi_lane_scope_scans_to_next_safe_issue(self) -> None:
        """A broad head issue should not wedge a lane when a later candidate fits
        the lane lease. The scan is bounded and transparent like contract holds."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "docs", "numbers": [467, 466],
                          "by_lane_count": {"docs": 2},
                          "eligible_by_lane": [["docs", [467, 466]]]})

        def scope(root, text, lane):
            if "title 467" in text:
                return {
                    "multi_lane": True,
                    "chosen_lane": lane,
                    "chosen_tree": ["docs/**"],
                    "uncovered_lanes": ["tools"],
                    "uncovered": [{"path": "tools/status.py", "lanes": ["tools"]}],
                }
            return {"multi_lane": False, "chosen_lane": lane,
                    "chosen_tree": ["docs/**"], "uncovered_lanes": [],
                    "uncovered": []}

        mod.scan_multi_lane_scope = scope
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)

        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertEqual(p["target_issue"], 466)
        self.assertEqual([r["issue"] for r in p["multi_lane_skipped"]], [467])
        self.assertIn("multi-lane head", mod.render(p))

    def test_prior_multi_lane_hold_skips_candidate(self) -> None:
        """A live MULTI_LANE_SCOPE hold from a previous tick advances later
        dry-runs instead of repeatedly selecting the same parent issue."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "docs", "numbers": [2319, 2320],
                          "by_lane_count": {"docs": 2},
                          "eligible_by_lane": [["docs", [2319, 2320]]]})
        mod.multi_lane_held_issues = lambda runs_dir, **k: {2319}
        reviewed: list[int] = []

        def counting_review(root, issue, number, **_kw):
            reviewed.append(number)
            return passing_issue_contract()

        mod.issue_contract_review = counting_review
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)

        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertEqual(p["target_issue"], 2320)
        self.assertEqual(reviewed, [2320])
        self.assertEqual(p["multi_lane_held_prior"], 1)

    def test_multi_lane_scope_live_records_hold_when_no_safe_candidate(self) -> None:
        """A live final MULTI_LANE_SCOPE refusal records the exact split action so
        subsequent ticks can skip the parent until it changes."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "docs", "numbers": [2319],
                          "by_lane_count": {"docs": 1},
                          "eligible_by_lane": [["docs", [2319]]]})

        def scope(root, text, lane):
            return {
                "multi_lane": True,
                "chosen_lane": lane,
                "chosen_tree": ["docs/**"],
                "uncovered_lanes": ["tools"],
                "uncovered": [{"path": "tools/seo_aeo_scorecard.py",
                               "lanes": ["tools"]}],
            }

        seen: dict = {}
        mod.scan_multi_lane_scope = scope
        mod.record_multi_lane_holds = (
            lambda runs_dir, rows, **k: seen.update({"rows": rows, "live": k.get("live")}))

        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=True)

        self.assertFalse(p["ok"])
        self.assertEqual(p["action"], "multi_lane_scope")
        self.assertEqual(p["verdict"], "MULTI_LANE_SCOPE")
        self.assertTrue(seen["live"])
        self.assertEqual(seen["rows"][0]["issue"], 2319)
        self.assertEqual(seen["rows"][0]["uncovered_lanes"], ["tools"])
        self.assertIn("tools/seo_aeo_scorecard.py", seen["rows"][0]["reason"])

    def test_dirty_path_collision_scans_to_next_clean_issue(self) -> None:
        """A candidate naming dirty local WIP is skipped before launch pricing,
        so an auto-pick can still admit a later disjoint issue."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "tools", "numbers": [2719, 2721],
                          "by_lane_count": {"tools": 2},
                          "eligible_by_lane": [["tools", [2719, 2721]]]})
        mod.dirty_repo_paths = lambda root: {
            "paths": ["cmd/fak/knownbad.go"], "unavailable": False}

        def build(n, lane, *, workspace):
            body = ("Likely files: `cmd/fak/knownbad.go`"
                    if n == 2719 else "Likely files: `tools/safe.py`")
            return {"prompt": f"resolve #{n}", "prompt_chars": 900,
                    "title": f"title {n}",
                    "issue_record": {"body": body}}

        mod.issue_worker_prompt.build = build
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)

        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertEqual(p["target_issue"], 2721)
        self.assertEqual([r["issue"] for r in p["dirty_path_skipped"]], [2719])
        self.assertIn("dirty-path head", mod.render(p))

    def test_same_issue_wip_scans_to_next_clean_issue(self) -> None:
        """A prior same-issue resolver log with still-dirty files holds that
        candidate only; the picker can still admit a later disjoint issue."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "tools", "numbers": [2718, 2721],
                          "by_lane_count": {"tools": 2},
                          "eligible_by_lane": [["tools", [2718, 2721]]]})
        mod.dirty_repo_paths = lambda root: {
            "paths": ["cmd/fak/knownbad.go"], "unavailable": False}

        def same_issue_wip(runs_dir, issue, dirty_paths, **_kw):
            if issue == 2718:
                return {
                    "collides": True,
                    "issue": 2718,
                    "dirty_paths": ["cmd/fak/knownbad.go"],
                    "evidence": [{"log": "resolve-2718-20260705-191407.log"}],
                }
            return {"collides": False, "dirty_paths": []}

        mod.same_issue_wip_collision = same_issue_wip
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)

        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertEqual(p["target_issue"], 2721)
        self.assertEqual([r["issue"] for r in p["same_issue_wip_skipped"]], [2718])
        self.assertIn("same-issue WIP head", mod.render(p))

    def test_same_issue_wip_holds_when_no_clean_candidate(self) -> None:
        """A final/pinned same-issue WIP candidate is a typed safety hold, not a
        launch-ready dry-run."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "tools", "numbers": [2718],
                          "by_lane_count": {"tools": 1},
                          "eligible_by_lane": [["tools", [2718]]]})
        mod.dirty_repo_paths = lambda root: {
            "paths": ["cmd/fak/knownbad.go"], "unavailable": False}
        mod.same_issue_wip_collision = lambda runs_dir, issue, dirty_paths, **_kw: {
            "collides": True,
            "issue": issue,
            "dirty_paths": ["cmd/fak/knownbad.go"],
            "evidence": [{"log": "resolve-2718-20260705-191407.log"}],
        }

        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)

        self.assertFalse(p["ok"])
        self.assertEqual(p["action"], "same_issue_wip")
        self.assertEqual(p["verdict"], "SAME_ISSUE_WIP")
        self.assertEqual(p["launch_gate"]["blockers"][0]["code"],
                         "SAME_ISSUE_WIP")
        self.assertIn("cmd/fak/knownbad.go", p["reason"])
        self.assertIn("resolve-2718-20260705-191407.log", p["reason"])

    def test_dirty_path_collision_holds_when_no_clean_candidate(self) -> None:
        """A final/pinned dirty-path candidate is a safety hold, not a launch-ready
        dry-run."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "tools", "numbers": [2719],
                          "by_lane_count": {"tools": 1},
                          "eligible_by_lane": [["tools", [2719]]]})
        mod.dirty_repo_paths = lambda root: {
            "paths": ["cmd/fak/knownbad.go"], "unavailable": False}
        mod.issue_worker_prompt.build = lambda n, lane, *, workspace: {
            "prompt": f"resolve #{n}", "prompt_chars": 900,
            "title": f"title {n}",
            "issue_record": {"body": "Likely files: `cmd/fak/knownbad.go`"}}

        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)

        self.assertFalse(p["ok"])
        self.assertEqual(p["action"], "dirty_path_collision")
        self.assertEqual(p["verdict"], "DIRTY_PATH_COLLISION")
        self.assertEqual(p["launch_gate"]["blockers"][0]["code"],
                         "DIRTY_PATH_COLLISION")
        self.assertIn("cmd/fak/knownbad.go", p["reason"])

    def test_active_guard_livelock_blocks_spawn(self) -> None:
        """A live result-side guard livelock pauses new issue spawns before the
        launcher compounds the failing worker class."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "ci", "numbers": [2867],
                          "by_lane_count": {"ci": 1},
                          "eligible_by_lane": [["ci", [2867]]]})
        mod.active_guard_livelock_hold = lambda root, runs_dir, **k: {
            "active": True,
            "reason": "live worker #2720 is already in a guard result livelock",
            "candidates": [{"issue": 2720, "count": 50}],
        }

        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)

        self.assertFalse(p["ok"])
        self.assertEqual(p["action"], "active_guard_livelock")
        self.assertEqual(p["verdict"], "ACTIVE_GUARD_LIVELOCK")
        self.assertEqual(p["launch_gate"]["blockers"][0]["code"],
                         "ACTIVE_GUARD_LIVELOCK")
        self.assertIn("guard result livelock", p["reason"])
        self.assertIn("launch gate: BLOCKED", mod.render(p))

    def test_guard_livelock_on_other_lane_does_not_block_disjoint_lane(self) -> None:
        """A guard livelock on lane A must NOT idle dispatch to lane B (#5861).

        Same shape #5858 fixed for ACTIVE_COMPACT_RUNAWAY: one worker stuck on
        the same quarantined tool_result is a local tool-loop condition, and
        scoping the hold to the colliding lane is what makes it a proportionate
        response instead of a fleet-wide stop.
        """
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "tools", "numbers": [4334],
                          "by_lane_count": {"tools": 1},
                          "eligible_by_lane": [["tools", [4334]]]})
        mod.active_guard_livelock_hold = lambda root, runs_dir, **k: {
            "active": True,
            "lanes": ["quality"],
            "lane_unknown": False,
            "reason": ("live worker #2720 on lane 'quality' is already in a "
                       "guard result livelock"),
            "candidates": [{"issue": 2720, "lane": "quality", "count": 50}],
        }

        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)

        self.assertNotEqual(p.get("verdict"), "ACTIVE_GUARD_LIVELOCK")
        self.assertNotEqual(p.get("action"), "active_guard_livelock")
        # Never a silent drop: the disjoint livelock is still REPORTED, with the
        # lane it was scoped against and the explicit no-collision finding.
        self.assertEqual(p["active_guard_livelock"]["scoped_lane"], "tools")
        self.assertFalse(p["active_guard_livelock"]["collides"])

    def test_guard_livelock_still_blocks_its_own_lane(self) -> None:
        """Lane-scoping must not disarm the hold on the lane actually spinning."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "quality", "numbers": [2867],
                          "by_lane_count": {"quality": 1},
                          "eligible_by_lane": [["quality", [2867]]]})
        mod.active_guard_livelock_hold = lambda root, runs_dir, **k: {
            "active": True,
            "lanes": ["quality"],
            "lane_unknown": False,
            "reason": ("live worker #2720 on lane 'quality' is already in a "
                       "guard result livelock"),
            "candidates": [{"issue": 2720, "lane": "quality", "count": 50}],
        }

        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)

        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "ACTIVE_GUARD_LIVELOCK")
        self.assertTrue(p["active_guard_livelock"]["collides"])

    def test_guard_livelock_with_unknown_lane_fails_closed(self) -> None:
        """An unreadable spawn header cannot prove disjointness, so the hold keeps
        its old fleet-wide behaviour rather than guessing."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "tools", "numbers": [4334],
                          "by_lane_count": {"tools": 1},
                          "eligible_by_lane": [["tools", [4334]]]})
        mod.active_guard_livelock_hold = lambda root, runs_dir, **k: {
            "active": True,
            "lanes": [],
            "lane_unknown": True,
            "reason": "live worker #2720 is already in a guard result livelock",
            "candidates": [{"issue": 2720, "lane": None, "count": 50}],
        }

        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)

        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "ACTIVE_GUARD_LIVELOCK")
        self.assertTrue(p["active_guard_livelock"]["collides"])

    def test_active_compact_runaway_blocks_spawn(self) -> None:
        """A live compact-control runaway pauses new issue spawns before the
        launcher adds more workers to the same failing regime."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "docs", "numbers": [2311],
                          "by_lane_count": {"docs": 1},
                          "eligible_by_lane": [["docs", [2311]]]})
        mod.active_compact_runaway_hold = lambda root, runs_dir, **k: {
            "active": True,
            "reason": "live worker #2311 is already past compact by 45.5k tokens",
            "candidates": [{"issue": 2311, "count": 3, "max_past_k": 45.5}],
        }

        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)

        self.assertFalse(p["ok"])
        self.assertEqual(p["action"], "active_compact_runaway")
        self.assertEqual(p["verdict"], "ACTIVE_COMPACT_RUNAWAY")
        self.assertEqual(p["launch_gate"]["blockers"][0]["code"],
                         "ACTIVE_COMPACT_RUNAWAY")
        self.assertIn("past compact", p["reason"])
        self.assertIn("launch gate: BLOCKED", mod.render(p))

    def test_compact_runaway_on_other_lane_does_not_block_disjoint_lane(self) -> None:
        """A runaway on lane A must NOT idle the fleet's dispatch to lane B (#5858).

        The observed stall: one worker on lane `quality` was past compact, and the
        tick refused to launch #4334 on the DISJOINT lane `tools`, holding 20 free
        slots fleet-wide. A compact runaway is one worker's local context
        condition; scoping the hold to the colliding lane is what makes it a
        proportionate response instead of a fleet-wide stop.
        """
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "tools", "numbers": [4334],
                          "by_lane_count": {"tools": 1},
                          "eligible_by_lane": [["tools", [4334]]]})
        mod.active_compact_runaway_hold = lambda root, runs_dir, **k: {
            "active": True,
            "lanes": ["quality"],
            "lane_unknown": False,
            "reason": "live worker #4568 on lane 'quality' is already past compact",
            "candidates": [{"issue": 4568, "lane": "quality", "count": 9,
                            "max_past_k": 38.1}],
        }

        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)

        self.assertNotEqual(p.get("verdict"), "ACTIVE_COMPACT_RUNAWAY")
        self.assertNotEqual(p.get("action"), "active_compact_runaway")
        # Never a silent drop: the disjoint runaway is still REPORTED, with the
        # lane it was scoped against and the explicit no-collision finding.
        self.assertEqual(p["active_compact_runaway"]["scoped_lane"], "tools")
        self.assertFalse(p["active_compact_runaway"]["collides"])

    def test_compact_runaway_still_blocks_its_own_lane(self) -> None:
        """Lane-scoping must not disarm the hold on the lane actually burning."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "quality", "numbers": [4568],
                          "by_lane_count": {"quality": 1},
                          "eligible_by_lane": [["quality", [4568]]]})
        mod.active_compact_runaway_hold = lambda root, runs_dir, **k: {
            "active": True,
            "lanes": ["quality"],
            "lane_unknown": False,
            "reason": "live worker #4568 on lane 'quality' is already past compact",
            "candidates": [{"issue": 4568, "lane": "quality", "count": 9,
                            "max_past_k": 38.1}],
        }

        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)

        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "ACTIVE_COMPACT_RUNAWAY")
        self.assertTrue(p["active_compact_runaway"]["collides"])

    def test_compact_runaway_with_unknown_lane_fails_closed(self) -> None:
        """An unreadable spawn header cannot prove disjointness, so the hold keeps
        its old fleet-wide behaviour rather than guessing."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "tools", "numbers": [4334],
                          "by_lane_count": {"tools": 1},
                          "eligible_by_lane": [["tools", [4334]]]})
        mod.active_compact_runaway_hold = lambda root, runs_dir, **k: {
            "active": True,
            "lanes": [],
            "lane_unknown": True,
            "reason": "live worker #4568 is already past compact",
            "candidates": [{"issue": 4568, "lane": None, "count": 9}],
        }

        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)

        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "ACTIVE_COMPACT_RUNAWAY")

    def test_contract_scan_bounded_holds_when_none_ready(self) -> None:
        """When every scanned issue is thin, the tick still HOLDs (the floor is
        never relaxed) and the reason names the scan so the hold is legible."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467, 466, 452],
                          "by_lane_count": {"gateway": 3}})
        mod.issue_contract_review = lambda *a, **k: {
            "ok": False, "unavailable": False, "score": 8, "spine_priority": 0,
            "review": {"ok": False, "reasons": ["ISSUE_SCOPE_INCOMPLETE"],
                       "missing_fields": ["working_spine"], "score": {"total": 8}}}
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False, repair_batch=0)  # pin the HOLD path
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "ISSUE_CONTRACT_HOLD")
        self.assertIn("scanned 3 lane issue(s)", p["reason"])
        self.assertEqual([r["issue"] for r in p["contract_skipped"]], [467, 466])
        self.assertEqual(p["target_issue"], 452)

    def test_contract_scan_respects_budget(self) -> None:
        """contract_scan caps how many issues one tick may review: with a budget
        of 2 and three thin issues, only the first two are reviewed."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467, 466, 452],
                          "by_lane_count": {"gateway": 3}})
        reviewed: list[int] = []

        def counting_review(root, issue, number, **_kw):
            reviewed.append(number)
            return {"ok": False, "unavailable": False, "score": 8,
                    "spine_priority": 0,
                    "review": {"ok": False, "reasons": ["ISSUE_SCOPE_INCOMPLETE"],
                               "missing_fields": ["working_spine"],
                               "score": {"total": 8}}}

        mod.issue_contract_review = counting_review
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False, contract_scan=2, repair_batch=0)
        self.assertEqual(p["verdict"], "ISSUE_CONTRACT_HOLD")
        self.assertEqual(reviewed, [467, 466])

    def test_contract_hold_ledger_roundtrip_and_ttl(self) -> None:
        """record_contract_holds appends only on LIVE ticks; contract_held_issues
        honors the TTL horizon and the 0-disables contract."""
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td)
            rows = [{"issue": 1207, "score": 8, "reason": "thin"},
                    {"issue": 1271, "score": 0, "reason": "thin"}]
            mod.record_contract_holds(runs, rows, live=False, now_ts=1_000_000.0)
            self.assertEqual(mod.contract_held_issues(runs, ttl_h=24,
                                                      now_ts=1_000_100.0), set())
            mod.record_contract_holds(runs, rows, live=True, now_ts=1_000_000.0)
            self.assertEqual(mod.contract_held_issues(runs, ttl_h=24,
                                                      now_ts=1_000_100.0),
                             {1207, 1271})
            records = mod.contract_held_records(runs, ttl_h=24,
                                                now_ts=1_000_100.0,
                                                only={1207})
            self.assertEqual([r["issue"] for r in records], [1207])
            self.assertEqual(records[0]["reason"], "thin")
            # Past the TTL horizon the holds expire (a backfill gets re-reviewed)...
            self.assertEqual(mod.contract_held_issues(
                runs, ttl_h=24, now_ts=1_000_000.0 + 25 * 3600), set())
            # ...and 0 disables the gate outright.
            self.assertEqual(mod.contract_held_issues(runs, ttl_h=0,
                                                      now_ts=1_000_100.0), set())

    def test_multi_lane_hold_ledger_roundtrip_ttl_and_update_escape(self) -> None:
        import tempfile
        mod = load()
        rows = [{
            "issue": 2319,
            "lane": "docs",
            "title": "multi lane",
            "reason": "Split into per-lane child issues",
            "uncovered_lanes": ["tools"],
        }]
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td)
            mod.record_multi_lane_holds(runs, rows, live=False, now_ts=1_000_000.0)
            self.assertEqual(mod.multi_lane_held_issues(runs, ttl_h=24,
                                                        now_ts=1_000_100.0), set())
            mod.record_multi_lane_holds(runs, rows, live=True, now_ts=1_000_000.0)
            self.assertEqual(mod.multi_lane_held_issues(runs, ttl_h=24,
                                                        now_ts=1_000_100.0), {2319})
            records = mod.multi_lane_held_records(runs, ttl_h=24,
                                                  now_ts=1_000_100.0)
            self.assertEqual(records[0]["uncovered_lanes"], ["tools"])
            self.assertEqual(mod.multi_lane_held_issues(
                runs, ttl_h=24, now_ts=1_000_000.0 + 25 * 3600), set())
            self.assertEqual(mod.multi_lane_held_issues(
                runs, ttl_h=24, now_ts=1_000_100.0,
                updated_ts={2319: 1_000_010.0}), set())

    def test_collision_hold_ledger_roundtrip_ttl_and_update_escape(self) -> None:
        """A dirty-path / same-issue-WIP collision earns the same durable, TTL-bounded
        skip the contract and multi-lane gates have, so the picker stops re-refusing
        the same colliding head every tick. Dry runs never write; the TTL expires the
        hold; a fresh gh updatedAt does NOT void a local-tree hold (it clears on a
        local commit/revert), but still re-admits a content-keyed hold early."""
        import tempfile
        mod = load()
        rows = [{
            "issue": 2779,
            "kind": "dirty_path",
            "lane": "claude",
            "title": "guard collision",
            "reason": "names dirty local path(s): cmd/fak/guard.go",
            "dirty_paths": ["cmd/fak/guard.go"],
        }]
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td)
            # Dry-run tick never pollutes the durable ledger.
            mod.record_collision_holds(runs, rows, live=False, now_ts=1_000_000.0)
            self.assertEqual(mod.collision_held_issues(runs, ttl_h=3,
                                                       now_ts=1_000_100.0), set())
            # A live collision hold is durable and skips the issue next tick.
            mod.record_collision_holds(runs, rows, live=True, now_ts=1_000_000.0)
            self.assertEqual(mod.collision_held_issues(runs, ttl_h=3,
                                                       now_ts=1_000_100.0), {2779})
            records = mod.collision_held_records(runs, ttl_h=3, now_ts=1_000_100.0)
            self.assertEqual(records[0]["kind"], "dirty_path")
            self.assertEqual(records[0]["dirty_paths"], ["cmd/fak/guard.go"])
            # Past the TTL the hold lapses so the live tree is re-checked.
            self.assertEqual(mod.collision_held_issues(
                runs, ttl_h=3, now_ts=1_000_000.0 + 4 * 3600), set())
            # A gh updatedAt bump does NOT void a LOCAL-tree collision hold: the
            # dirty path clears on a local commit/revert, not on a remote body edit,
            # so re-admitting on updatedAt re-entered the same colliding head every
            # tick (the #3045-style retry loop). The hold now survives the bump; only
            # the local TTL/commit clears it.
            self.assertEqual(mod.collision_held_issues(
                runs, ttl_h=3, now_ts=1_000_100.0,
                updated_ts={2779: 1_000_010.0}), {2779})
            # A CONTENT-keyed (non-local) collision kind still re-admits early on a
            # fresh updatedAt (a body edit can genuinely change that verdict) — the
            # fix is scoped to local-tree kinds, not a blanket disable.
            content_rows = [{**rows[0], "issue": 4242, "kind": "content_probe"}]
            mod.record_collision_holds(runs, content_rows, live=True, now_ts=1_000_000.0)
            self.assertEqual(mod.collision_held_issues(
                runs, ttl_h=3, now_ts=1_000_100.0,
                updated_ts={4242: 1_000_010.0}), {2779})  # 4242 re-admitted, 2779 stays held
            # TTL<=0 disables the ledger entirely (kill switch).
            self.assertEqual(mod.collision_held_issues(runs, ttl_h=0,
                                                       now_ts=1_000_100.0), set())

    def test_prior_contract_holds_advance_the_scan(self) -> None:
        """An issue held on a PRIOR tick is skipped without a re-review, so the
        bounded scan window advances across ticks instead of pinning to the head."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467, 466, 452],
                          "by_lane_count": {"gateway": 3}})
        mod.contract_held_issues = lambda runs_dir, **k: {467}
        reviewed: list[int] = []

        def counting_review(root, issue, number, **_kw):
            reviewed.append(number)
            return passing_issue_contract()

        mod.issue_contract_review = counting_review
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertEqual(p["target_issue"], 466)
        self.assertEqual(reviewed, [466])  # 467 skipped from the ledger, unreviewed
        self.assertEqual(p["contract_held_prior"], 1)

    def test_local_witnessed_issue_skipped_until_push_close(self) -> None:
        """A local diff-witnessed commit can be ahead of origin while a push guard
        is blocked; do not spawn a duplicate worker for that still-open issue."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "tools", "numbers": [1444, 2293],
                          "by_lane_count": {"tools": 2}})
        mod.locally_witnessed_issues = lambda root, **k: {1444}
        reviewed: list[int] = []

        def counting_review(root, issue, number, **_kw):
            reviewed.append(number)
            return passing_issue_contract()

        mod.issue_contract_review = counting_review
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertEqual(p["target_issue"], 2293)
        self.assertEqual(p["locally_witnessed"], [1444])
        self.assertEqual(reviewed, [2293])

    def test_open_witnessed_candidate_closed_not_redispatched(self) -> None:
        """#5071 regression (the #2850 redispatch): an open candidate whose
        resolving commit is already witnessed in trunk ancestry is disposed
        OPEN_WITNESSED — excluded from selection before lease/spawn, reported
        with its witnessed SHA — while a non-witnessed control stays eligible."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "tools", "numbers": [2850, 2293],
                          "by_lane_count": {"tools": 2},
                          "eligible_by_lane": [["tools", [2850, 2293]]]})
        seen: dict[str, set[int]] = {}

        def fake_dispositions(root, candidates, **_kw):
            seen["candidates"] = set(candidates)
            return [{"issue": 2850, "sha": "f8aff29dfd", "code": "OPEN_WITNESSED",
                     "subject": "fix(dispatch): guard tick (#2850) (fak tools)",
                     "verdict": "OK", "witness": "diff-witnessed",
                     "claim_kind": "code_effect",
                     "close_via": "tools/issue_resolve_witnessed.py"}]

        mod.open_witnessed_dispositions = fake_dispositions
        reviewed: list[int] = []

        def counting_review(root, issue, number, **_kw):
            reviewed.append(number)
            return passing_issue_contract()

        mod.issue_contract_review = counting_review
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertEqual(p["target_issue"], 2293)  # the control stays eligible
        self.assertEqual(reviewed, [2293])  # 2850 never reviewed/spawned
        self.assertEqual(seen["candidates"], {2850, 2293})
        self.assertEqual(p["open_witnessed"][0]["issue"], 2850)
        self.assertEqual(p["open_witnessed"][0]["code"], "OPEN_WITNESSED")
        self.assertEqual(p["open_witnessed"][0]["sha"], "f8aff29dfd")
        # dry-run never invokes the close arm — dispositions only
        self.assertNotIn("open_witnessed_close", p)

    def test_commit_audit_abstain_test_commit_holds_candidate(self) -> None:
        """A matching test-scope commit whose witness ABSTAINS is a visible hold,
        not normal resolver fuel for a duplicate worker."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "ci", "numbers": [2867, 2293],
                          "by_lane_count": {"ci": 2},
                          "eligible_by_lane": [["ci", [2867, 2293]]]})
        mod.commit_audit_abstain_holds = lambda root, candidates, **k: [{
            "issue": 2867,
            "sha": "41f23ea7",
            "code": "COMMIT_AUDIT_ABSTAIN",
            "verdict": "ABSTAIN",
            "witness": "abstain",
            "claim_kind": "none",
            "reason": "subject makes no checkable code/test claim",
            "test_files": ["internal/boundarylint/changedetector_test.go"],
        }]
        reviewed: list[int] = []

        def counting_review(root, issue, number, **_kw):
            reviewed.append(number)
            return passing_issue_contract()

        mod.issue_contract_review = counting_review
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertEqual(p["target_issue"], 2293)
        self.assertEqual(reviewed, [2293])
        self.assertEqual(p["commit_audit_abstain_held"][0]["issue"], 2867)
        self.assertEqual(p["commit_audit_abstain_held"][0]["code"],
                         "COMMIT_AUDIT_ABSTAIN")

    def test_all_prior_contract_holds_plan_repair_worker(self) -> None:
        """A tick whose picker is empty only because the hold ledger skipped every
        open candidate is not idle: it plans the existing repair worker from the
        held rows."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467, 466],
                          "eligible_by_lane": [["gateway", [467, 466]]],
                          "by_lane_count": {"gateway": 2}})
        mod.contract_held_issues = lambda runs_dir, **k: {467, 466}
        mod.contract_held_records = lambda runs_dir, **k: [
            {"issue": 467, "number": 467, "score": 8,
             "reason": "ISSUE_SCOPE_INCOMPLETE, missing:working_spine",
             "missing_fields": ["working_spine"]},
            {"issue": 466, "number": 466, "score": 8,
             "reason": "ISSUE_AGENT_CONTEXT_INCOMPLETE, missing:parent_ref",
             "missing_fields": ["parent_ref"]},
        ]

        def boom_review(*_args, **_kwargs):
            raise AssertionError("prior holds should not be re-reviewed")

        mod.issue_contract_review = boom_review
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "WOULD_REPAIR")
        self.assertEqual(p["action"], "would_repair")
        self.assertEqual(p["contract_repair"]["batch"], [467, 466])
        self.assertIn("launch_gate", p)
        self.assertIn("guarded", p)
        self.assertEqual(p["contract_held_prior"], 2)

    def test_hold_tick_records_skipped_and_final_target(self) -> None:
        """A live all-thin tick ledgers every reviewed-and-held issue (skipped heads
        plus the final held target) so the next tick starts past them."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467, 466, 452],
                          "by_lane_count": {"gateway": 3}})
        mod.issue_contract_review = lambda *a, **k: {
            "ok": False, "unavailable": False, "score": 8, "spine_priority": 0,
            "review": {"ok": False, "reasons": ["ISSUE_SCOPE_INCOMPLETE"],
                       "missing_fields": ["working_spine"], "score": {"total": 8}}}
        recorded: list[dict] = []
        mod.record_contract_holds = (
            lambda runs_dir, rows, **k: recorded.extend(rows))
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=True, repair_batch=0)  # pin the HOLD path
        self.assertEqual(p["verdict"], "ISSUE_CONTRACT_HOLD")
        self.assertEqual([r["issue"] for r in recorded], [467, 466, 452])

    def test_scan_crosses_lanes_to_a_ready_issue(self) -> None:
        """Lane-level head-of-line fix: when the busiest lane's head is thin, the
        scan's round-robin stream reaches ANOTHER eligible lane's ready issue in
        the same tick instead of starving it behind the fat lane."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "docs", "numbers": [100, 101],
                          "by_lane_count": {"docs": 2, "tools": 1},
                          "eligible_by_lane": [["docs", [100, 101]],
                                               ["tools", [200]]]})

        def per_issue_review(root, issue, number, **_kw):
            if number == 200:
                return passing_issue_contract()
            return {"ok": False, "unavailable": False, "score": 8,
                    "spine_priority": 0,
                    "review": {"ok": False, "reasons": ["ISSUE_SCOPE_INCOMPLETE"],
                               "missing_fields": ["working_spine"],
                               "score": {"total": 8}}}

        mod.issue_contract_review = per_issue_review
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertEqual(p["target_issue"], 200)
        self.assertEqual(p["lane"], "tools")  # crossed from docs to tools
        self.assertEqual([(r["lane"], r["issue"]) for r in p["contract_skipped"]],
                         [("docs", 100)])  # round-robin: docs head first

    def test_all_thin_dry_run_plans_a_repair_worker(self) -> None:
        """The self-serve arm: a window where EVERYTHING fails the gate plans a
        contract-repair worker on the held issues instead of a bare hold."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467, 466],
                          "by_lane_count": {"gateway": 2}})
        mod.issue_contract_review = lambda *a, **k: {
            "ok": False, "unavailable": False, "score": 8, "spine_priority": 0,
            "review": {"ok": False, "reasons": ["ISSUE_SCOPE_INCOMPLETE"],
                       "missing_fields": ["working_spine"], "score": {"total": 8}}}
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "WOULD_REPAIR")
        self.assertEqual(p["action"], "would_repair")
        self.assertEqual(p["contract_repair"]["batch"], [467, 466])
        self.assertIn("launch_gate", p)
        self.assertIn("guarded", p)
        self.assertEqual(mod.tick_exit_code(p), 0)  # benign: work was dispatched

    def test_all_thin_live_spawns_repair_worker_with_repair_prefix(self) -> None:
        """A live all-thin tick SPAWNS the repair worker: repair log prefix (so
        resolve-only scans never see it), the contract-repair pseudo-lane (no lane
        lease), and the .issues batch sidecar naming every groomed issue."""
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            self._patch(mod, pre=self.SPAWN_OK,
                        pick={"lane": "gateway", "numbers": [467, 466],
                              "by_lane_count": {"gateway": 2}})
            mod.issue_contract_review = lambda *a, **k: {
                "ok": False, "unavailable": False, "score": 8, "spine_priority": 0,
                "review": {"ok": False, "reasons": ["ISSUE_SCOPE_INCOMPLETE"],
                           "missing_fields": ["working_spine"],
                           "score": {"total": 8}}}
            seen: dict = {}

            def fake_spawn(command, env, cwd, log_dir, issue, lane, backend,
                           **kwargs):
                seen.update({"issue": issue, "lane": lane,
                             "log_prefix": kwargs.get("log_prefix"),
                             "env_issues": env.get("FLEET_REPAIR_ISSUES")})
                log = Path(td) / f"repair-{issue}-20260702-190000.log"
                log.write_text("# fak-spawn", encoding="utf-8")
                return {"pid": 41, "log": str(log), "issue": issue,
                        "lane": lane, "backend": backend}

            mod.spawn_issue_worker = fake_spawn
            p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                             lane=None, live=True)
            self.assertTrue(p["ok"])
            self.assertEqual(p["verdict"], "REPAIR_SPAWNED")
            self.assertEqual(p["action"], "repair_spawned")
            self.assertEqual(seen["log_prefix"], "repair")
            self.assertEqual(seen["lane"], mod.REPAIR_LANE)
            self.assertEqual(seen["issue"], 467)  # batch head names the sidecars
            self.assertEqual(seen["env_issues"], "467,466")
            sidecar = Path(td) / ("repair-467-20260702-190000"
                                  + mod.REPAIR_ISSUES_SIDECAR_SUFFIX)
            self.assertEqual(sidecar.read_text(encoding="utf-8"), "467,466")
            # ...and the DURABLE half: the sidecar above is swept with the corpse,
            # so the whole batch is also ledgered where the cooldown can still read
            # it after the groomer dies.
            ledger = (Path(mod.RUNS_DIRNAME)
                      / mod._REPAIR_ATTEMPT_LEDGER)
            rows = [json.loads(x) for x in
                    ledger.read_text(encoding="utf-8").splitlines() if x.strip()]
            self.assertEqual(rows[-1]["issues"], [466, 467])

    def test_live_repair_worker_blocks_a_second_groomer(self) -> None:
        """Repair admission is serialized: with a groomer already alive, the tick
        reports REPAIR_IN_FLIGHT (benign) instead of stacking a second one."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467],
                          "by_lane_count": {"gateway": 1}})
        mod.issue_contract_review = lambda *a, **k: {
            "ok": False, "unavailable": False, "score": 8, "spine_priority": 0,
            "review": {"ok": False, "reasons": ["ISSUE_SCOPE_INCOMPLETE"],
                       "missing_fields": ["working_spine"], "score": {"total": 8}}}
        mod.live_repair_workers = lambda runs_dir, **k: [
            {"pid": 999, "pid_file": "repair-1207-20260702-180000.pid",
             "issue": 1207}]
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=True)  # boom-spawn proves nothing spawns
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "REPAIR_IN_FLIGHT")
        self.assertEqual(mod.tick_exit_code(p), 0)

    def test_repair_cooldown_falls_back_to_plain_hold(self) -> None:
        """Issues a groomer already attempted are not re-groomed inside the
        cooldown: with every held candidate cooled, the tick HOLDs and says why."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467, 466],
                          "by_lane_count": {"gateway": 2}})
        mod.issue_contract_review = lambda *a, **k: {
            "ok": False, "unavailable": False, "score": 8, "spine_priority": 0,
            "review": {"ok": False, "reasons": ["ISSUE_SCOPE_INCOMPLETE"],
                       "missing_fields": ["working_spine"], "score": {"total": 8}}}
        mod.recently_repaired_issues = (
            lambda runs_dir, *, cooldown_min, **k: {467, 466})
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "ISSUE_CONTRACT_HOLD")
        self.assertIn("repair cooldown", p["reason"])
        self.assertIn("cooldown", p["contract_repair"]["skipped"])

    def test_hold_ledger_updatedat_readmits_a_backfilled_issue(self) -> None:
        """The repair-pipeline turnaround: an issue whose GitHub updatedAt is
        NEWER than its held verdict re-enters the pick immediately; one whose
        body has not changed stays held until the TTL."""
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td)
            rows = [{"issue": 1207, "score": 8, "reason": "thin"},
                    {"issue": 1271, "score": 0, "reason": "thin"}]
            mod.record_contract_holds(runs, rows, live=True, now_ts=1_000_000.0)
            held = mod.contract_held_issues(
                runs, ttl_h=24, now_ts=1_000_100.0,
                updated_ts={1207: 1_000_050.0})  # edited AFTER the held verdict
            self.assertEqual(held, {1271})  # 1207 re-admitted, 1271 still held
            held = mod.contract_held_issues(
                runs, ttl_h=24, now_ts=1_000_100.0,
                updated_ts={1207: 999_000.0})  # stale edit: predates the verdict
            self.assertEqual(held, {1207, 1271})

    def test_pinned_issue_is_never_substituted_by_scan(self) -> None:
        """--issue pins an operator-vetted target: a thin pinned issue HOLDs the
        tick exactly as before -- the scan must not swap in a different issue."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467, 401, 388],
                          "by_lane_count": {"gateway": 3}})
        mod.issue_contract_review = lambda *a, **k: {
            "ok": False, "unavailable": False, "score": 8, "spine_priority": 0,
            "review": {"ok": False, "reasons": ["ISSUE_SCOPE_INCOMPLETE"],
                       "missing_fields": ["working_spine"], "score": {"total": 8}}}
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane="gateway", live=False, issue_override=401)
        self.assertEqual(p["verdict"], "ISSUE_CONTRACT_HOLD")
        self.assertEqual(p["target_issue"], 401)
        self.assertNotIn("contract_skipped", p)

    @staticmethod
    def _thin_contract_review(*_a, **_k):
        return {
            "ok": False,
            "unavailable": False,
            "score": 40,
            "spine_priority": 0,
            "review": {
                "ok": False,
                "reasons": ["ISSUE_SCOPE_INCOMPLETE"],
                "missing_fields": ["working_spine"],
                "score": {"total": 40},
            },
        }

    def test_force_downgrades_contract_hold_to_advisory(self) -> None:
        """--force + --force-reason (operator best-effort): a contract that WOULD hold
        is downgraded to advisory and the tick proceeds to WOULD_SPAWN, recording the
        structured reason and the running bypass count (#2637)."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467],
                          "by_lane_count": {"gateway": 1}})
        mod.issue_contract_review = self._thin_contract_review
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False, force=True,
                         force_reason="operator: top real issue, worker prompt carries context")
        # Readiness is relaxed, but the tick is now spawnable (dry-run: WOULD_SPAWN).
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertTrue(p["issue_contract_forced"]["bypassed"])
        self.assertIn("ISSUE_SCOPE_INCOMPLETE",
                      p["issue_contract_forced"]["gate_reason"])
        # The operator's structured reason is carried in the run artifact (#2637).
        self.assertEqual(p["issue_contract_forced"]["reason"],
                         "operator: top real issue, worker prompt carries context")
        self.assertIn("bypass_count", p["issue_contract_forced"])
        # The gate result is still recorded transparently (not hidden by the force).
        self.assertFalse(p["issue_contract_gate"]["ok"])

    def test_default_contract_hold_does_not_spawn(self) -> None:
        """DONE-CONDITION #1 (#2637): a score<floor contract with NO --force does not
        spawn a worker — it refuses ISSUE_CONTRACT_HOLD before any launch."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467],
                          "by_lane_count": {"gateway": 1}})
        mod.issue_contract_review = self._thin_contract_review
        # spawn_issue_worker is `boom`; reaching it would fail the test.
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=True, repair_batch=0)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "ISSUE_CONTRACT_HOLD")
        self.assertNotIn("issue_contract_forced", p)

    def test_force_without_reason_refuses_spawn(self) -> None:
        """DONE-CONDITION #2 (#2637): a bare --force on a FAILED contract gate is not
        enough — it refuses FORCE_REASON_REQUIRED and spawns nothing."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467],
                          "by_lane_count": {"gateway": 1}})
        mod.issue_contract_review = self._thin_contract_review
        # live=True so spawn_issue_worker (boom) would fire if enforcement leaked.
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=True, force=True, force_reason="   ")
        self.assertFalse(p["ok"])
        self.assertEqual(p["action"], "force_reason_required")
        self.assertEqual(p["verdict"], "FORCE_REASON_REQUIRED")
        self.assertEqual(p["launch_gate"]["blockers"][0]["code"],
                         "FORCE_REASON_REQUIRED")
        self.assertNotIn("issue_contract_forced", p)

    def test_force_with_reason_records_bypass_in_ledger(self) -> None:
        """DONE-CONDITION #2+#3 (#2637): a LIVE forced spawn appends the structured
        reason to the audit ledger and the exposed bypass_count reflects it."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467],
                          "by_lane_count": {"gateway": 1}})
        mod.issue_contract_review = self._thin_contract_review
        # A live tick reaches spawn; patch it so we can assert the pre-spawn ledger
        # write without launching a real worker.
        mod.spawn_issue_worker = lambda *a, **k: {
            "pid": 4321, "issue": 467, "log": "resolve-467.log",
            "early_exit": {"checked": True, "alive": True}}
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=True, force=True,
                         force_reason="operator: dispatch top backlog issue")
        self.assertEqual(p["verdict"], "SPAWNED")
        self.assertEqual(p["issue_contract_forced"]["reason"],
                         "operator: dispatch top backlog issue")
        self.assertGreaterEqual(p["issue_contract_forced"]["bypass_count"], 1)
        runs_dir = ROOT / mod.RUNS_DIRNAME
        self.assertEqual(mod.contract_forced_bypass_count(runs_dir),
                         p["issue_contract_forced"]["bypass_count"])

    def test_contract_forced_bypass_ledger_is_live_only(self) -> None:
        """The forced-bypass ledger records LIVE ticks only; a dry-run stays
        side-effect-free (#2637)."""
        mod = load()
        runs_dir = ROOT / mod.RUNS_DIRNAME
        row = {"issue": 2633, "lane": "gateway", "score": 41,
               "reason": "operator: because", "gate_reason": "score:41<floor:100"}
        mod.record_contract_forced_bypass(runs_dir, row, live=False)
        self.assertEqual(mod.contract_forced_bypass_count(runs_dir), 0)
        mod.record_contract_forced_bypass(runs_dir, row, live=True)
        mod.record_contract_forced_bypass(runs_dir, row, live=True)
        self.assertEqual(mod.contract_forced_bypass_count(runs_dir), 2)

    def test_render_surfaces_forced_bypass_count(self) -> None:
        """DONE-CONDITION #3 (#2637): the operator-facing render surfaces the forced
        bypass — its structured reason, the gate reason, and the running count — so a
        bypassed guard is visible, not silent telemetry."""
        mod = load()
        payload = {
            "verdict": "SPAWNED", "ok": True, "backend": "claude", "live": True,
            "target_issue": 2633, "issue_title": "thin issue",
            "issue_contract_forced": {
                "bypassed": True,
                "gate_reason": "score:41<floor:100",
                "reason": "operator: dispatch top backlog issue",
                "bypass_count": 7,
            },
        }
        rendered = mod.render(payload)
        self.assertIn("contract gate bypassed", rendered)
        self.assertIn("operator: dispatch top backlog issue", rendered)
        self.assertIn("total bypasses 7", rendered)

    def test_issue_override_pins_target(self) -> None:
        """--issue pins an explicit target, overriding the freshest-first auto-pick."""
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467, 401, 388],
                          "by_lane_count": {"gateway": 3}})
        mod.issue_contract_review = passing_issue_contract
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane="gateway", live=False, issue_override=401)
        self.assertEqual(p["target_issue"], 401)

    def test_reap_runs_before_preflight_and_is_recorded(self) -> None:
        mod = load()
        state = {"reaped": False}

        def fake_pre(root, **kw):
            self.assertTrue(state["reaped"])
            return self.SPAWN_OK

        mod.issue_dispatch.refresh_registry = lambda root: {"ok": True}
        mod.issue_dispatch.preflight = fake_pre
        mod.reap_timed_out_workers = lambda runs_dir, **k: (
            state.__setitem__("reaped", True) or
            {"timeout_s": k.get("timeout_s"), "live": k.get("live"),
             "candidates": [{"pid": 101}], "reaped": [{"pid": 101}], "would_reap": []})
        mod.check_weekly_cap = lambda runs_dir, **k: {"capped": False}
        mod.contract_held_issues = lambda runs_dir, **k: set()
        mod.multi_lane_held_issues = lambda runs_dir, **k: set()
        mod.open_issue_updated_map = lambda root, **k: {}
        mod.record_contract_holds = lambda runs_dir, rows, **k: None
        mod.record_multi_lane_holds = lambda runs_dir, rows, **k: None
        mod.lane_issue_numbers = lambda root, lane, exclude=None: {
            "lane": "gateway", "numbers": [467], "by_lane_count": {"gateway": 1}}
        mod.live_resolution_issues = lambda runs_dir: set()
        mod.live_resolution_lanes = lambda runs_dir: set()
        mod.live_lane_lease_lanes = lambda root: {"lanes": []}
        mod.recently_attempted_issues = lambda runs_dir, *, cooldown_min, **k: set()
        mod.issue_worker_prompt.build = lambda n, lane, *, workspace: {
            "prompt": f"resolve #{n}", "prompt_chars": 100, "title": f"title {n}"}
        mod.issue_contract_review = passing_issue_contract
        mod.acquire_lane_lease = lambda root, lane, **k: {
            "acquired": True, "refused": False, "id": f"resolve-{lane}",
            "holder": "test", "tree": k.get("tree") or []}
        mod.reap_expired_leases = lambda root, **k: {"ok": True, "rc": 0}
        mod.lane_tree = lambda root, lane: [f"internal/{lane}/**"]
        mod.spawn_issue_worker = lambda *a, **k: {
            "pid": 202, "log": "resolve-467.log", "issue": 467,
            "lane": "gateway", "backend": "claude"}

        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=True)
        self.assertTrue(p["ok"])
        self.assertEqual(p["timed_out_workers"]["reaped"], [{"pid": 101}])

    def test_skips_already_live_issue(self) -> None:
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467, 466], "by_lane_count": {}},
                    live_issues=[467])
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertEqual(p["target_issue"], 466)   # 467 skipped (live)

    def test_refuses_lane_busy_when_explicit_lane_already_held(self) -> None:
        # The concurrency witness (#1310): a second tick targeting a lane that
        # already holds a live worker is REFUSED (COLLISION_RISK), not raced onto
        # the same leaf tree. spawn_issue_worker is the `boom` stub, so reaching it
        # would fail the test — the gate must short-circuit before the live spawn.
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "docs", "numbers": [310, 309],
                          "by_lane_count": {"docs": 2}},
                    held_lanes=["docs"])
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane="docs", live=True)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "LANE_BUSY")
        self.assertEqual(p["action"], "lane_busy")
        self.assertEqual(p["held_lanes"], ["docs"])
        self.assertIn("docs", p["reason"])

    def test_autopick_reroutes_around_a_held_lane(self) -> None:
        # The "queued elsewhere" half (#1310): the busiest-pick must EXCLUDE a held
        # lane so the dispatcher reroutes to a free lane instead of refusing.
        mod = load()
        seen: dict = {}
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467], "by_lane_count": {}},
                    held_lanes=["docs"])

        def _router(root, lane, exclude=None):
            seen["exclude"] = set(exclude or set())
            return {"lane": "gateway", "numbers": [467], "by_lane_count": {"gateway": 1}}
        mod.lane_issue_numbers = _router
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertIn("docs", seen["exclude"])     # held lane folded into the exclude
        self.assertEqual(p["lane"], "gateway")       # rerouted to a free lane
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertEqual(p["held_lanes"], ["docs"])

    def test_skips_cooled_issue_and_advances(self) -> None:
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "docs", "numbers": [465, 464, 455],
                          "by_lane_count": {}},
                    cooled=[465])   # 465 attempted recently -> skip, advance to 464
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertEqual(p["target_issue"], 464)
        self.assertEqual(p["cooled_recently"], [465])

    def test_no_issue_when_all_live(self) -> None:
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467], "by_lane_count": {}},
                    live_issues=[467])
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "NO_ISSUE")

    def test_refused_when_preflight_refuses(self) -> None:
        mod = load()
        self._patch(
            mod,
            pre={"verdict": "REFUSE_AT_CAP", "reason": "2/2 live", "cap": 2,
                 "live": 2, "max_workers": 2, "host_cap": 16,
                 "capacity_limiter": {
                     "primary": "configured_max",
                     "term": "max_workers",
                     "raw": {"max_workers": 2, "host_cap": 16},
                 },
                 "account": {}},
            pick={"lane": "gateway", "numbers": [467], "by_lane_count": {}})
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "REFUSE_AT_CAP")
        self.assertIn("preflight refused", p["reason"])
        self.assertEqual(p["preflight"]["capacity_limiter"]["term"], "max_workers")
        self.assertEqual(p["preflight_hint"]["kind"], "configured_max_workers")
        self.assertEqual(p["preflight_hint"]["required_min"], 3)
        self.assertIn("--max-workers=2", mod.render(p))

    def test_refused_at_host_cap_surfaces_live_limiter(self) -> None:
        mod = load()
        self._patch(
            mod,
            pre={"verdict": "REFUSE_AT_CAP", "reason": "16/16 live", "cap": 1,
                 "live": 16, "max_workers": 1, "host_cap": 16,
                 "capacity_limiter": {
                     "primary": "leases",
                     "term": "live",
                     "raw": {"max_workers": 1, "host_cap": 16},
                 },
                 "account": {}},
            pick={"lane": "gateway", "numbers": [467], "by_lane_count": {}})
        p = mod.evaluate(ROOT, max_workers=1, work_kind="engineering",
                         lane=None, live=False)
        self.assertFalse(p["ok"])
        self.assertEqual(p["preflight_hint"]["kind"], "live")
        self.assertIn("capacity limiter leases/live", p["preflight_hint"]["message"])
        self.assertNotIn("rerun with --max-workers", p["preflight_hint"]["message"])

    def test_no_lane_when_router_empty(self) -> None:
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": None, "numbers": [], "by_lane_count": {}})
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                          lane=None, live=False)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "NO_LANE")

    def test_no_lane_but_self_modify_held_surfaces_distinct_verdict(self) -> None:
        # Every lane with open issues is fak's own source tree -- distinct from the
        # plain "router empty" NO_LANE case, and named in the payload so an operator
        # (or the fleet dashboard) can see WHY, not just that nothing was picked.
        import json
        import tempfile
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": None, "numbers": [], "by_lane_count": {},
                          "self_modify_held": ["cmd", "gateway"]})
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = mod.evaluate(root, max_workers=2, work_kind="engineering",
                             lane=None, live=False)
            last_tick = json.loads(
                (root / mod.RUNS_DIRNAME / "last-resolve-tick-claude.json")
                .read_text(encoding="utf-8"))
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "SELF_MODIFY_HOLD")
        self.assertEqual(p["self_modify_held"], ["cmd", "gateway"])
        self.assertFalse(p["launch_gate"]["ready"])
        self.assertEqual(p["launch_gate"]["blockers"][0]["code"], "SELF_MODIFY_HOLD")
        self.assertIn("route-non-self-source", p["launch_gate"]["blockers"][0]["next_action"])
        self.assertIn("launch gate: BLOCKED", mod.render(p))
        self.assertEqual(last_tick["verdict"], "SELF_MODIFY_HOLD")
        self.assertEqual(last_tick["launch_gate"]["blockers"][0]["code"],
                         "SELF_MODIFY_HOLD")

    def test_no_lane_waits_when_safe_lanes_are_busy_before_self_source(self) -> None:
        import json
        import tempfile
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": None, "numbers": [], "by_lane_count": {},
                          "safe_lanes_busy": ["docs", "tools"],
                          "self_modify_held": ["cmd", "gateway"]},
                    held_lanes=["docs", "tools"])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = mod.evaluate(root, max_workers=2, work_kind="engineering",
                             lane=None, live=False)
            last_tick = json.loads(
                (root / mod.RUNS_DIRNAME / "last-resolve-tick-claude.json")
                .read_text(encoding="utf-8"))
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "SELF_MODIFY_HOLD")
        self.assertEqual(p["safe_lanes_busy"], ["docs", "tools"])
        self.assertEqual(p["launch_gate"]["blockers"][0]["code"], "SAFE_LANES_BUSY")
        self.assertEqual(p["launch_gate"]["blockers"][0]["next_action"],
                         "wait-for-safe-lane-lease")
        self.assertEqual(p["launch_gate"]["blockers"][1]["code"], "SELF_MODIFY_HOLD")
        self.assertEqual(last_tick["launch_gate"]["blockers"][0]["code"],
                         "SAFE_LANES_BUSY")

    def test_live_lane_lease_blocks_dry_run_candidate(self) -> None:
        import tempfile
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "docs", "numbers": [2042], "by_lane_count": {"docs": 1},
                          "eligible_by_lane": [["docs", [2042]]]})
        mod.live_lane_lease_lanes = lambda root: {
            "lanes": ["docs"],
            "records": [{"lane": "resolve-docs", "tree": ["docs/**"]}],
        }
        seen: dict = {}

        def picker(root, lane, exclude=None):
            seen["exclude"] = set(exclude or set())
            return {
                "lane": None,
                "numbers": [],
                "by_lane_count": {"docs": 1, "gateway": 1},
                "eligible_by_lane": [],
                "safe_lanes_busy": ["docs"],
                "self_modify_held": ["gateway"],
            }

        mod.lane_issue_numbers = picker
        with tempfile.TemporaryDirectory() as d:
            p = mod.evaluate(Path(d), max_workers=2, work_kind="engineering",
                             lane=None, live=False)
        self.assertIn("docs", seen["exclude"])
        self.assertEqual(p["lease_held_lanes"], ["docs"])
        self.assertEqual(p["held_lanes"], ["docs"])
        self.assertFalse(p["launch_gate"]["ready"])
        self.assertEqual(p["launch_gate"]["blockers"][0]["code"], "SAFE_LANES_BUSY")

    def test_live_reports_spawn_failed_when_worker_exits_silent_immediately(self) -> None:
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": "gateway", "numbers": [467], "by_lane_count": {}})
        seen: dict = {}

        def fake_spawn(*args, **kwargs):
            seen.update(kwargs)
            return {
                "pid": 202,
                "log": "resolve-467.log",
                "issue": 467,
                "lane": "gateway",
                "backend": "claude",
                "early_exit": {
                    "checked": True,
                    "alive": False,
                    "wait_s": 5.0,
                    "returncode": 0,
                    "log_bytes": 0,
                    "silent": True,
                },
            }

        mod.spawn_issue_worker = fake_spawn
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=True, spawn_probe_s=5.0)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "SPAWN_FAILED")
        self.assertEqual(p["action"], "spawn_failed")
        self.assertEqual(seen["spawn_probe_s"], 5.0)
        self.assertEqual(seen["account"]["tag"], "worker-a")

    def test_live_reports_spawn_failed_when_worker_exits_noisily_immediately(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = root / mod.RUNS_DIRNAME
            self._patch(mod, pre=self.SPAWN_OK,
                        pick={"lane": "tools", "numbers": [1402], "by_lane_count": {}})

            def fake_spawn(*_args, **_kwargs):
                return {
                    "pid": 303,
                    "log": str(runs / "resolve-1402.log"),
                    "issue": 1402,
                    "lane": "tools",
                    "backend": "claude",
                    "early_exit": {
                        "checked": True,
                        "alive": False,
                        "wait_s": 5.0,
                        "returncode": 1,
                        "log_bytes": 120,
                        "silent": False,
                        "tail": "API Error: Request rejected (429) - upstream rate-limited",
                    },
                }

            mod.spawn_issue_worker = fake_spawn
            p = mod.evaluate(root, max_workers=2, work_kind="engineering",
                             lane=None, live=True, spawn_probe_s=5.0)
            self.assertFalse(p["ok"])
            self.assertEqual(p["verdict"], "SPAWN_FAILED")
            self.assertIn("with code 1", p["reason"])
            self.assertEqual(p["quota_cap"]["source"], "early_exit")
            self.assertTrue((runs / "account-cap-claude.json").exists())


if __name__ == "__main__":
    unittest.main()
