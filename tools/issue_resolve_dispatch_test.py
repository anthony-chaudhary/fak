#!/usr/bin/env python3
"""Hermetic tests for tools/issue_resolve_dispatch.py.

Every shell-out (registry refresh, preflight, lane router, prompt build, spawn)
is stubbed on the module; NOTHING live (gh/dos/claude) runs and no worker is
spawned in dry-run. The pure pickers (pick_target_issue, lane fold) are tested
directly.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "issue_resolve_dispatch.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("issue_resolve_dispatch", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PickTargetTest(unittest.TestCase):
    def test_first_not_in_skip(self) -> None:
        mod = load()
        self.assertEqual(mod.pick_target_issue([497, 496, 495], set()), 497)
        self.assertEqual(mod.pick_target_issue([497, 496, 495], {497}), 496)
        self.assertEqual(mod.pick_target_issue([497, 496], {497, 496}), None)
        self.assertEqual(mod.pick_target_issue([], set()), None)


class CooldownTest(unittest.TestCase):
    def test_recent_log_is_in_cooldown_old_one_is_not(self) -> None:
        import os
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            now = 1_000_000.0
            fresh = runs / "resolve-465-20260621-225432.log"
            stale = runs / "resolve-450-20260621-100000.log"
            fresh.write_text("", encoding="utf-8")
            stale.write_text("", encoding="utf-8")
            os.utime(fresh, (now - 10 * 60, now - 10 * 60))    # 10 min ago
            os.utime(stale, (now - 300 * 60, now - 300 * 60))  # 5 h ago
            cooled = mod.recently_attempted_issues(runs, cooldown_min=120, now_ts=now)
        self.assertEqual(cooled, {465})        # 465 fresh (10m < 120m), 450 stale

    def test_cooldown_zero_disables(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            (runs / "resolve-465-x.log").write_text("", encoding="utf-8")
            self.assertEqual(mod.recently_attempted_issues(runs, cooldown_min=0), set())


class LiveResolutionIssuesTest(unittest.TestCase):
    def _mk(self, runs: Path, issue: int, stamp: str, *, pid: int,
            sidecar_mtime: float) -> None:
        import os
        log = runs / f"resolve-{issue}-{stamp}.log"
        log.write_text("", encoding="utf-8")
        pid_file = log.with_suffix(".pid")
        pid_file.write_text(str(pid), encoding="utf-8")
        os.utime(pid_file, (sidecar_mtime, sidecar_mtime))

    def test_counts_issue_when_sidecar_process_matches_spawn_time(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 717, "20260625-062210", pid=101, sidecar_mtime=now)
            # OS hid the cmdline, but the image is a real worker backend → counts.
            def probe(pid):
                return {"alive": True, "create_time": now - 1,
                                             "name": "claude.exe", "cmdline": ""}
            self.assertEqual(mod.live_resolution_issues(runs, probe=probe), {717})

    def test_rejects_issue_when_sidecar_pid_was_reused(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 718, "20260625-060712", pid=102, sidecar_mtime=now)
            def probe(pid):
                return {
                            "alive": True,
                            "create_time": now + 60 * 60,
                            "name": "chrome.exe",
                            "cmdline": "chrome.exe --type=renderer",
                        }
            self.assertEqual(mod.live_resolution_issues(runs, probe=probe), set())


class LiveResolutionLanesTest(unittest.TestCase):
    """The pre-spawn lane-lease set (#1310): a lane is HELD when it already has a
    live worker. Read from the ``# fak-spawn ... lane=<L>`` header
    ``spawn_issue_worker`` flushes, gated by the SAME sidecar-pid liveness check
    as the live-issue set, so a recycled PID never falsely holds a lane."""

    def _mk(self, runs: Path, issue: int, stamp: str, lane: str, *, pid: int,
            sidecar_mtime: float, header: bool = True) -> None:
        import os
        log = runs / f"resolve-{issue}-{stamp}.log"
        body = (f"# fak-spawn {stamp} issue={issue} lane={lane} backend=claude "
                f"argv0=claude.exe\n") if header else ""
        log.write_text(body, encoding="utf-8")
        pid_file = log.with_suffix(".pid")
        pid_file.write_text(str(pid), encoding="utf-8")
        os.utime(pid_file, (sidecar_mtime, sidecar_mtime))

    def test_held_lane_when_live_worker_header_names_it(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 717, "20260625-062210", "docs", pid=101, sidecar_mtime=now)
            def probe(pid):
                return {"alive": True, "create_time": now - 1,
                        "name": "claude.exe", "cmdline": ""}
            self.assertEqual(mod.live_resolution_lanes(runs, probe=probe), {"docs"})

    def test_recycled_pid_does_not_hold_its_lane(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 718, "20260625-060712", "gateway", pid=102, sidecar_mtime=now)
            def probe(pid):  # a different process owns the PID now → worker is gone
                return {"alive": True, "create_time": now + 60 * 60,
                        "name": "chrome.exe", "cmdline": "chrome.exe --type=renderer"}
            self.assertEqual(mod.live_resolution_lanes(runs, probe=probe), set())

    def test_headerless_log_contributes_no_lane(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # live pid but a 0-byte log (died before exec wrote the header) → its
            # lane is unknowable, so it holds nothing (best effort, fail-open).
            self._mk(runs, 719, "20260625-070000", "docs", pid=103,
                     sidecar_mtime=now, header=False)
            def probe(pid):
                return {"alive": True, "create_time": now - 1,
                        "name": "claude.exe", "cmdline": ""}
            self.assertEqual(mod.live_resolution_lanes(runs, probe=probe), set())

    def test_two_live_workers_hold_two_lanes(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 717, "20260625-062210", "docs", pid=101, sidecar_mtime=now)
            self._mk(runs, 720, "20260625-063000", "gateway", pid=104, sidecar_mtime=now)
            def probe(pid):
                return {"alive": True, "create_time": now - 1,
                        "name": "claude.exe", "cmdline": ""}
            self.assertEqual(mod.live_resolution_lanes(runs, probe=probe),
                             {"docs", "gateway"})


class TimedOutWorkerReapTest(unittest.TestCase):
    def _mk(self, runs: Path, issue: int, stamp: str, *, pid: int,
            sidecar_mtime: float) -> Path:
        import os
        log = runs / f"resolve-{issue}-{stamp}.log"
        log.write_text("", encoding="utf-8")
        pid_file = log.with_suffix(".pid")
        pid_file.write_text(str(pid), encoding="utf-8")
        os.utime(pid_file, (sidecar_mtime, sidecar_mtime))
        return pid_file

    def test_dry_run_reports_would_reap_without_killing(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        killed: list[int] = []
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 717, "20260625-062210", pid=101, sidecar_mtime=now - 4000)
            def probe(pid):
                return {"alive": True, "create_time": now - 4001,
                                             "name": "claude.exe", "cmdline": ""}
            out = mod.reap_timed_out_workers(
                runs, timeout_s=1800, live=False, now_ts=now, probe=probe,
                killer=lambda pid: killed.append(pid))
        self.assertEqual([r["pid"] for r in out["would_reap"]], [101])
        self.assertEqual(out["reaped"], [])
        self.assertEqual(killed, [])

    def test_live_reaps_matching_timed_out_worker(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        killed: list[int] = []
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 718, "20260625-060712", pid=102, sidecar_mtime=now - 4000)
            def probe(pid):
                return {"alive": True, "create_time": now - 4001,
                                             "name": "claude.exe", "cmdline": ""}
            out = mod.reap_timed_out_workers(
                runs, timeout_s=1800, live=True, now_ts=now, probe=probe,
                killer=lambda pid: (killed.append(pid) or {"ok": True, "returncode": 0}))
        self.assertEqual([r["pid"] for r in out["reaped"]], [102])
        self.assertEqual(killed, [102])
        self.assertTrue(out["reaped"][0]["kill"]["ok"])

    def test_fresh_worker_is_not_reaped(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 719, "20260625-055209", pid=103, sidecar_mtime=now - 60)
            def probe(pid):
                return {"alive": True, "create_time": now - 61, "cmdline": ""}
            out = mod.reap_timed_out_workers(
                runs, timeout_s=1800, live=True, now_ts=now, probe=probe,
                killer=lambda pid: {"ok": True})
        self.assertEqual(out["candidates"], [])
        self.assertEqual(out["reaped"], [])

    def test_reused_pid_is_not_reaped(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 720, "20260625-055210", pid=104, sidecar_mtime=now - 4000)
            def probe(pid):
                return {
                            "alive": True,
                            "create_time": now + 60,
                            "cmdline": "chrome.exe --type=renderer",
                        }
            out = mod.reap_timed_out_workers(
                runs, timeout_s=1800, live=True, now_ts=now, probe=probe,
                killer=lambda pid: {"ok": True})
        self.assertEqual(out["candidates"], [])
        self.assertEqual(out["reaped"], [])


class PruneDeadSidecarsTest(unittest.TestCase):
    def _mk(self, runs: Path, issue: int, stamp: str, *, pid: int,
            mtime: float, siblings: tuple[str, ...] = ()) -> Path:
        import os
        pid_file = runs / f"resolve-{issue}-{stamp}.pid"
        pid_file.write_text(str(pid), encoding="utf-8")
        stem = pid_file.with_suffix("")
        for suf in siblings:
            stem.with_suffix(suf).write_text("", encoding="utf-8")
        os.utime(pid_file, (mtime, mtime))
        return pid_file

    def test_live_prunes_dead_sidecar_and_siblings(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 825, "20260625-213720", pid=58752, mtime=now - 4000,
                     siblings=(".log", ".backend", ".wave", ".account"))
            def probe(pid):
                return {"alive": False}
            out = mod.prune_dead_sidecars(runs, live=True, now_ts=now, probe=probe)
            self.assertEqual(out["pruned"], ["resolve-825-20260625-213720.pid"])
            self.assertEqual(sorted(p.name for p in runs.glob("resolve-825-*")), [])

    def test_recycled_shell_in_window_is_pruned(self) -> None:
        # The exact ghost: a recycled cmd.exe whose create time lands inside the
        # stale sidecar's spawn window. resolve_sidecar_pid_is_live rejects a bare
        # shell image, so the sidecar is correctly seen as dead and swept.
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 825, "20260625-213720", pid=58752, mtime=now - 4000)
            def probe(pid):
                return {"alive": True, "create_time": now - 4030,
                                             "name": "cmd.exe", "cmdline": ""}
            out = mod.prune_dead_sidecars(runs, live=True, now_ts=now, probe=probe)
            self.assertEqual(out["pruned"], ["resolve-825-20260625-213720.pid"])

    def test_live_worker_sidecar_is_kept(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 822, "20260625-220538", pid=8416, mtime=now - 4000)
            def probe(pid):
                return {"alive": True, "create_time": now - 4001,
                                             "name": "claude.exe",
                                             "cmdline": "claude -p resolve GitHub issue #822"}
            out = mod.prune_dead_sidecars(runs, live=True, now_ts=now, probe=probe)
            self.assertEqual(out["pruned"], [])
            self.assertTrue((runs / "resolve-822-20260625-220538.pid").exists())

    def test_too_fresh_dead_sidecar_survives_min_age(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 830, "20260625-210713", pid=999, mtime=now - 5)
            def probe(pid):
                return {"alive": False}
            out = mod.prune_dead_sidecars(runs, live=True, min_age_s=60.0,
                                          now_ts=now, probe=probe)
            self.assertEqual(out["pruned"], [])
            self.assertTrue((runs / "resolve-830-20260625-210713.pid").exists())

    def test_dry_run_reports_would_prune_without_unlinking(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            pid_file = self._mk(runs, 831, "20260625-205217", pid=111, mtime=now - 4000)
            def probe(pid):
                return {"alive": False}
            out = mod.prune_dead_sidecars(runs, live=False, now_ts=now, probe=probe)
            self.assertEqual(out["would_prune"], ["resolve-831-20260625-205217.pid"])
            self.assertEqual(out["pruned"], [])
            self.assertTrue(pid_file.exists())

    def test_timeout_zero_disables_reap(self) -> None:
        mod = load()
        out = mod.reap_timed_out_workers(Path("does-not-matter"), timeout_s=None, live=True)
        self.assertTrue(out["disabled"])


class LoopLedgerTest(unittest.TestCase):
    def test_record_loop_tick_writes_fire_admit_end_for_dry_run(self) -> None:
        mod = load()
        rows: list[dict[str, object]] = []

        def append(root, ledger, ev):
            rows.append(dict(ev))
            return {"ok": True, "kind": ev["kind"]}

        payload = {
            "schema": mod.SCHEMA,
            "workspace": str(ROOT),
            "live": False,
            "backend": "claude",
            "max_workers": 2,
            "preflight": {"verdict": "SPAWN_OK", "cap": 2, "live": 0},
            "account": {"tag": "worker-a", "tier": 1, "model": "opus"},
            "lane": "gateway",
            "lane_issue_count": 3,
            "target_issue": 717,
            "prompt_chars": 1200,
            "ok": True,
            "action": "would_spawn",
            "verdict": "WOULD_SPAWN",
            "reason": "safe to spawn one worker",
        }
        rec = mod.record_loop_tick(
            ROOT, payload, ledger=Path("loops.jsonl"), append=append,
            mint=lambda root, process: "RID-DISPATCH1")

        self.assertTrue(rec["ok"])
        self.assertEqual(rec["loop_id"], "issue-resolve-dispatch/claude")
        self.assertEqual([r["kind"] for r in rows], ["fire", "admit", "end"])
        self.assertEqual(rows[1]["status"], "admitted")
        self.assertEqual(rows[1]["reason"], "WOULD_SPAWN")
        self.assertEqual(rows[1]["metrics"]["target_issue"], 717)
        self.assertIn(("issue", "717"), rows[1]["evidence"])
        self.assertEqual(payload["run_id"], "RID-DISPATCH1")

    def test_record_loop_tick_refusal_has_fire_and_refused_admit(self) -> None:
        mod = load()
        rows: list[dict[str, object]] = []
        payload = {
            "schema": mod.SCHEMA,
            "workspace": str(ROOT),
            "live": True,
            "backend": "opencode",
            "max_workers": 2,
            "preflight": {"verdict": "REFUSE_NO_ACCOUNT", "cap": 2, "live": 2},
            "account": {},
            "lane": "docs",
            "lane_issue_count": 8,
            "ok": False,
            "action": "refused",
            "verdict": "REFUSE_NO_ACCOUNT",
            "reason": "preflight refused: no account",
        }

        mod.record_loop_tick(
            ROOT,
            payload,
            ledger=Path("loops.jsonl"),
            append=lambda root, ledger, ev: (rows.append(dict(ev)) or {"ok": True, "kind": ev["kind"]}),
            mint=lambda root, process: "RID-DISPATCH2",
        )

        self.assertEqual([r["kind"] for r in rows], ["fire", "admit"])
        self.assertEqual(rows[1]["status"], "refused")
        self.assertEqual(rows[1]["reason"], "REFUSE_NO_ACCOUNT")
        self.assertEqual(rows[1]["metrics"]["preflight_live"], 2)


class EvaluateTest(unittest.TestCase):
    SPAWN_OK = {
        "verdict": "SPAWN_OK", "reason": "ok", "cap": 2, "live": 0,
        "account": {"tag": "worker-a", "tier": 1, "model": "opus", "dir": "/acct/a"},
    }

    def _patch(self, mod, *, pre, pick, live_issues=None, cooled=None,
               held_lanes=None, prompt_chars=900) -> None:
        mod.issue_dispatch.refresh_registry = lambda root: {"ok": True}
        mod.issue_dispatch.preflight = lambda root, **kw: pre
        mod.lane_issue_numbers = lambda root, lane, exclude=None: pick
        mod.live_resolution_issues = lambda runs_dir: set(live_issues or [])
        mod.live_resolution_lanes = lambda runs_dir: set(held_lanes or [])
        mod.recently_attempted_issues = lambda runs_dir, *, cooldown_min, **k: set(cooled or [])
        mod.issue_worker_prompt.build = lambda n, lane, *, workspace: {
            "prompt": f"resolve #{n}", "prompt_chars": prompt_chars, "title": f"title {n}"}
        mod.check_weekly_cap = lambda runs_dir, **k: {"capped": False}  # hermetic: not capped
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
        mod.lane_issue_numbers = lambda root, lane, exclude=None: {
            "lane": "gateway", "numbers": [467], "by_lane_count": {"gateway": 1}}
        mod.live_resolution_issues = lambda runs_dir: set()
        mod.live_resolution_lanes = lambda runs_dir: set()
        mod.recently_attempted_issues = lambda runs_dir, *, cooldown_min, **k: set()
        mod.issue_worker_prompt.build = lambda n, lane, *, workspace: {
            "prompt": f"resolve #{n}", "prompt_chars": 100, "title": f"title {n}"}
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
                 "live": 2, "account": {}},
            pick={"lane": "gateway", "numbers": [467], "by_lane_count": {}})
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "REFUSE_AT_CAP")
        self.assertIn("preflight refused", p["reason"])

    def test_no_lane_when_router_empty(self) -> None:
        mod = load()
        self._patch(mod, pre=self.SPAWN_OK,
                    pick={"lane": None, "numbers": [], "by_lane_count": {}})
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                          lane=None, live=False)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "NO_LANE")

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


class BuildWorkerCommandTest(unittest.TestCase):
    def test_claude_command_shape(self) -> None:
        mod = load()
        self.assertEqual(
            mod.build_worker_command("claude", "PROMPT", None),
            ["claude", "-p", "--permission-mode", "bypassPermissions", "PROMPT"])

    def test_opencode_command_pins_model_and_skips_permissions(self) -> None:
        mod = load()
        self.assertEqual(
            mod.build_worker_command("opencode", "PROMPT", "zai-coding-plan/glm-5.2"),
            ["opencode", "run", "--dangerously-skip-permissions",
             "-m", "zai-coding-plan/glm-5.2", "PROMPT"])

    def test_opencode_command_without_model(self) -> None:
        mod = load()
        self.assertEqual(
            mod.build_worker_command("opencode", "PROMPT", None),
            ["opencode", "run", "--dangerously-skip-permissions", "PROMPT"])

    def test_codex_command_shape(self) -> None:
        mod = load()
        self.assertEqual(
            mod.build_worker_command("codex", "PROMPT", None),
            ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox",
             "--skip-git-repo-check", "PROMPT"])

    def test_codex_command_pins_model(self) -> None:
        mod = load()
        self.assertEqual(
            mod.build_worker_command("codex", "PROMPT", "gpt-5.5"),
            ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox",
             "--skip-git-repo-check", "-m", "gpt-5.5", "PROMPT"])

    def test_codex_env_uses_codex_home_not_claude_vars(self) -> None:
        mod = load()
        env = mod.codex_worker_env("/some/.codex", "tools", Path("/ws"))
        self.assertEqual(env.get("CODEX_HOME"), "/some/.codex")
        self.assertNotIn("CLAUDE_CONFIG_DIR", env)
        self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", env)

    def test_unknown_backend_rejected(self) -> None:
        mod = load()
        with self.assertRaises(ValueError):
            mod.build_worker_command("gpt", "x", None)


class WinCreationFlagsTest(unittest.TestCase):
    def test_batch_launcher_gets_hidden_console_not_detached(self) -> None:
        # Regression: opencode.CMD spawned DETACHED_PROCESS dies at the batch
        # prompt producing 0 bytes — every glm worker was DOA. A .cmd/.bat needs a
        # (hidden) console.
        mod = load()
        self.assertEqual(mod.win_creationflags(r"C:\npm\opencode.CMD"), mod._CREATE_NO_WINDOW)
        self.assertEqual(mod.win_creationflags("opencode.cmd"), mod._CREATE_NO_WINDOW)
        self.assertEqual(mod.win_creationflags("wrap.bat"), mod._CREATE_NO_WINDOW)

    def test_native_launcher_also_gets_hidden_console(self) -> None:
        # Regression: a native worker (claude.exe) spawned DETACHED_PROCESS has NO
        # console, so the git/gh/fak/shell tools it spawns each pop a visible window
        # — the "random popups" during a dispatched run. CREATE_NO_WINDOW gives it one
        # hidden console the whole subtree inherits, so nothing flashes.
        mod = load()
        self.assertEqual(mod.win_creationflags(r"C:\bin\claude.exe"), mod._CREATE_NO_WINDOW)
        self.assertEqual(mod.win_creationflags("/usr/bin/claude"), mod._CREATE_NO_WINDOW)
        self.assertNotEqual(mod.win_creationflags(r"C:\bin\claude.exe"), mod._DETACHED_PROCESS)


class SpawnProbeTest(unittest.TestCase):
    def test_probe_reports_alive_when_timeout_expires(self) -> None:
        import tempfile
        mod = load()

        class _Alive:
            def wait(self, timeout):
                raise mod.subprocess.TimeoutExpired("worker", timeout)

        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "resolve-1.log"
            log.write_text("", encoding="utf-8")
            out = mod.probe_spawned_worker(_Alive(), log, wait_s=0.1)
        self.assertEqual(out, {"checked": True, "alive": True, "wait_s": 0.1})

    def test_probe_reports_silent_exit(self) -> None:
        import tempfile
        mod = load()

        class _Exited:
            def wait(self, timeout):
                return 0

        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "resolve-1.log"
            log.write_text("", encoding="utf-8")
            out = mod.probe_spawned_worker(_Exited(), log, wait_s=0.1)
        self.assertFalse(out["alive"])
        self.assertTrue(out["silent"])
        self.assertEqual(out["log_bytes"], 0)

    def test_spawn_writes_preexec_header_under_stub_floor(self) -> None:
        """A spawned worker's log opens with a flushed spawn header, so a later
        0-byte log means 'died before exec' (not 'exec'd then silent'). The header
        must stay under the stub floor and be banner-free so it never trips the
        stub/cap-banner classifiers."""
        import tempfile
        import sys
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # A trivially-exiting real process (no output of its own).
            res = mod.spawn_issue_worker([sys.executable, "-c", "pass"], env={},
                                         cwd=Path(d), log_dir=runs, issue=42,
                                         lane="docs", backend="claude",
                                         spawn_probe_s=2.0)
            log = Path(res["log"])
            body = log.read_text(encoding="utf-8")
        self.assertTrue(body.startswith("# fak-spawn "))
        self.assertIn("issue=42", body)
        self.assertIn("lane=docs", body)
        self.assertLess(len(body), mod._STUB_LOG_MAX_BYTES)   # header alone stays a stub
        self.assertIsNone(mod._CAP_BANNER_RE.search(body))    # never a false cap banner


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
        mod.recently_attempted_issues = lambda runs_dir, *, cooldown_min, **k: set()
        mod.issue_worker_prompt.build = lambda n, lane, *, workspace: {
            "prompt": f"resolve #{n}", "prompt_chars": 100, "title": f"t{n}"}
        mod.check_weekly_cap = lambda runs_dir, **k: {"capped": False}
        mod.reap_timed_out_workers = lambda runs_dir, **k: {
            "timeout_s": k.get("timeout_s"), "live": k.get("live"),
            "candidates": [], "reaped": [], "would_reap": []}
        mod.spawn_issue_worker = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("dry-run must not spawn"))

        p = mod.evaluate(ROOT, max_workers=2, work_kind="gardening",
                         lane="docs", live=False, backend="opencode")
        self.assertEqual(seen.get("product"), "opencode")       # routed to glm pool
        self.assertEqual(p["backend"], "opencode")
        self.assertEqual(p["command"][0], "opencode")
        self.assertIn("zai-coding-plan/glm-5.2", p["command"])   # model pinned/traced
        self.assertTrue(p["ok"])

    def test_unknown_backend_raises(self) -> None:
        mod = load()
        with self.assertRaises(ValueError):
            mod.evaluate(ROOT, max_workers=2, work_kind="x", lane=None,
                         live=False, backend="gpt")


class ExcludeLaneTest(unittest.TestCase):
    ROUTER = {"lanes": {
        "docs": {"issues": [{"number": 9}, {"number": 8}, {"number": 7}]},
        "gateway": {"issues": [{"number": 6}, {"number": 5}]},
    }}

    def test_busiest_pick_skips_excluded_lane(self) -> None:
        mod = load()
        mod.issue_dispatch.run_json = lambda cmd, root, timeout: self.ROUTER
        # without exclude: docs (3) is busiest
        self.assertEqual(mod.lane_issue_numbers(ROOT, None)["lane"], "docs")
        # excluding docs: gateway (2) wins
        picked = mod.lane_issue_numbers(ROOT, None, exclude={"docs"})
        self.assertEqual(picked["lane"], "gateway")
        self.assertEqual(picked["excluded_lanes"], ["docs"])

    def test_explicit_lane_ignores_exclude(self) -> None:
        mod = load()
        mod.issue_dispatch.run_json = lambda cmd, root, timeout: self.ROUTER
        picked = mod.lane_issue_numbers(ROOT, "docs", exclude={"docs"})
        self.assertEqual(picked["lane"], "docs")   # explicit wins over exclude


class OpencodeConfigHomeTest(unittest.TestCase):
    def test_canonical_dir_uses_parent_as_xdg_home(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            acct = Path(d) / "opencode"
            acct.mkdir()
            home = mod._opencode_config_home(str(acct), Path(d) / "runs")
            self.assertEqual(Path(home), Path(d))   # parent is the XDG config home


class WeeklyCapGateTest(unittest.TestCase):
    """The weekly-cap gate: detect the limit banner from recent worker logs, parse
    the reset, persist a hold, and make evaluate() refuse with WEEKLY_CAPPED."""
    BANNER = "You've hit your weekly limit · resets Jun 25, 1pm (America/Los_Angeles)"

    def _write_worker(self, runs: Path, name: str, body: str, *, mtime: float,
                      backend: str = "claude") -> None:
        import os
        log = runs / name
        log.write_text(body, encoding="utf-8")
        log.with_suffix(".backend").write_text(backend, encoding="utf-8")
        os.utime(log, (mtime, mtime))

    def test_scan_detects_recent_tiny_banner_only(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write_worker(runs, "resolve-517-a.log", self.BANNER, mtime=now - 60)      # fresh banner
            self._write_worker(runs, "resolve-410-b.log", self.BANNER, mtime=now - 99999)   # stale (outside lookback)
            self._write_worker(runs, "resolve-411-c.log",                                    # big prose mentioning "limit"
                               "I fixed the rate limit handler; resets the cache.\n" * 50, mtime=now - 30)
            hit = mod._scan_recent_cap_banner(runs, product="claude", lookback_min=45, now_ts=now)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["reset_text"], "Jun 25, 1pm")
        self.assertEqual(hit["evidence_log"], "resolve-517-a.log")

    def test_scan_scoped_to_backend(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write_worker(runs, "resolve-1-x.log", self.BANNER, mtime=now - 60, backend="opencode")
            # claude scan must NOT pick up an opencode worker's log
            self.assertIsNone(mod._scan_recent_cap_banner(runs, product="claude", lookback_min=45, now_ts=now))
            self.assertIsNotNone(mod._scan_recent_cap_banner(runs, product="opencode", lookback_min=45, now_ts=now))

    def test_parse_reset_date_and_time_to_utc(self) -> None:
        import datetime as dt
        mod = load()
        now = dt.datetime(2026, 6, 23, 8, 0, 0)
        got = mod._parse_reset_to_utc("Jun 25, 1pm", now)
        self.assertEqual(got, dt.datetime(2026, 6, 25, 20, 0, 0))   # 1pm PDT(-7) -> 20:00 UTC

    def test_parse_reset_time_only_next_occurrence(self) -> None:
        import datetime as dt
        mod = load()
        now = dt.datetime(2026, 6, 23, 8, 0, 0)        # 01:00 PT
        got = mod._parse_reset_to_utc("4am", now)       # next 04:00 PT == 11:00 UTC same day
        self.assertEqual(got, dt.datetime(2026, 6, 23, 11, 0, 0))

    def test_parse_reset_unparseable_is_none(self) -> None:
        import datetime as dt
        mod = load()
        self.assertIsNone(mod._parse_reset_to_utc("soon", dt.datetime(2026, 6, 23, 8, 0)))

    def test_check_writes_and_then_honors_persisted_hold(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write_worker(runs, "resolve-517-a.log", self.BANNER, mtime=now - 60)
            first = mod.check_weekly_cap(runs, product="claude", account_tag="smith", now_ts=now)
            self.assertTrue(first["capped"])
            self.assertEqual(first["source"], "banner")
            self.assertTrue((runs / "account-cap-claude.json").exists())
            # Later tick, banner now stale (no fresh evidence) but hold persists:
            later = mod.check_weekly_cap(runs, product="claude", account_tag="smith",
                                         now_ts=now + 3 * 3600)
            self.assertTrue(later["capped"])
            self.assertEqual(later["source"], "state")

    def test_check_clears_expired_hold(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write_worker(runs, "resolve-517-a.log", self.BANNER, mtime=now - 60)
            mod.check_weekly_cap(runs, product="claude", account_tag="smith", now_ts=now)
            # Far past the Jun-25 reset -> hold expires, state cleared, spawning resumes.
            future = 1_000_000.0 + 30 * 86400
            out = mod.check_weekly_cap(runs, product="claude", account_tag="smith", now_ts=future)
            self.assertFalse(out["capped"])
            self.assertFalse((runs / "account-cap-claude.json").exists())

    def test_check_failopen_on_missing_dir(self) -> None:
        mod = load()
        out = mod.check_weekly_cap(Path("/no/such/dir/xyz"), product="claude",
                                   account_tag="smith", now_ts=1_000_000.0)
        self.assertFalse(out["capped"])

    # --- codex credit-wall: banner sits behind a ~700B startup preamble, so the log
    # clears the byte floor; the gate must still detect it (tail scan, no size cap)
    # and parse codex's "try again at <date>" reset (not the Claude reset clause). ---
    CODEX_WALL = (
        "workdir: C:\\work\\fak\nmodel: gpt-5.5\nsession id: 0c1f...\n"
        + ("hook: SessionStart\nhook: UserPromptSubmit\n" * 30)   # ~700B preamble > 512 floor
        + "ERROR: You've hit your usage limit. Visit https://chatgpt.com/codex/settings/usage "
          "to purchase more credits or try again at Jul 1st, 2026 8:41 PM.\n")

    def test_scan_detects_codex_wall_over_size_floor(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # >512 bytes (the old size gate would have skipped it), backend=opencode/codex.
            self.assertGreater(len(self.CODEX_WALL), 512)
            self._write_worker(runs, "resolve-70-z.log", self.CODEX_WALL,
                               mtime=now - 60, backend="opencode")
            hit = mod._scan_recent_cap_banner(runs, product="opencode",
                                              lookback_min=45, now_ts=now)
        self.assertIsNotNone(hit)                         # banner found despite size
        self.assertEqual(hit["reset_text"], "Jul 1st, 2026 8:41 PM")   # codex reset parsed

    def test_codex_wall_parses_to_real_reset_not_short_cooldown(self) -> None:
        import datetime as dt
        mod = load()
        now = dt.datetime(2026, 6, 28, 20, 0, 0)          # Jun 28, well before the Jul-1 wall
        got = mod._parse_reset_to_utc("Jul 1st, 2026 8:41 PM", now)
        # Jul 1 8:41 PM PDT(-7) -> Jul 2 03:41 UTC; days out >> the 60-min fallback.
        self.assertEqual(got, dt.datetime(2026, 7, 2, 3, 41, 0))

    def test_backend_health_classifies_codex_wall_as_stub(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # Three dead-pid codex wall logs (each >512B) must read as a DEAD streak,
            # not three "productive" logs that falsely break it.
            for i in range(mod._BACKEND_DEAD_STREAK):
                name = f"resolve-7{i}-w.log"
                self._write_worker(runs, name, self.CODEX_WALL,
                                   mtime=now - 60 * (i + 1), backend="opencode")
                (runs / name).with_suffix(".pid").write_text("0", encoding="utf-8")
            # alive=empty set -> no pid is live -> the dead-pid guard treats each as a stub.
            out = mod.check_backend_health(runs, product="opencode", now_ts=now, alive=set())
        self.assertEqual(out["state"], "dead")

    # --- session vs weekly: a transient SESSION limit must not become a ~24h wall ---
    SESSION_BANNER = "You've hit your session limit · resets 6:10am (America/Los_Angeles)"

    def test_scan_classifies_session_vs_weekly(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write_worker(runs, "resolve-1-a.log", self.SESSION_BANNER, mtime=now - 60)
            hit = mod._scan_recent_cap_banner(runs, product="claude", lookback_min=45, now_ts=now)
            self.assertEqual(hit["kind"], "session")
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write_worker(runs, "resolve-2-b.log", self.BANNER, mtime=now - 60)
            hit = mod._scan_recent_cap_banner(runs, product="claude", lookback_min=45, now_ts=now)
            self.assertEqual(hit["kind"], "weekly")

    def test_session_limit_hold_is_bounded_not_a_day(self) -> None:
        # The gem8 false-cap: a "session limit · resets 6:10am" whose 6:10am already
        # passed today must NOT push the hold to 6:10am TOMORROW. A session hold is
        # bounded to _SESSION_HOLD_MAX_MIN, so the account is free again within ~90m,
        # not ~24h.
        import tempfile
        import datetime as dt
        mod = load()
        # now = a time where 6:10am PT has already passed today (e.g. 13:00 UTC = 06:00 PT
        # is before, so use 15:00 UTC = 08:00 PT, well past 06:10 PT).
        now_utc = dt.datetime(2026, 6, 25, 15, 0, 0)
        now_ts = (now_utc - dt.datetime(1970, 1, 1)).total_seconds()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write_worker(runs, "resolve-438-x.log", self.SESSION_BANNER, mtime=now_ts - 60)
            out = mod.check_weekly_cap(runs, product="claude", account_tag="gem8-netra",
                                       now_ts=now_ts)
            self.assertTrue(out["capped"])
            self.assertEqual(out["kind"], "session")
            until = dt.datetime.fromisoformat(out["until"].replace("Z", ""))
            held_min = (until - now_utc).total_seconds() / 60.0
            self.assertLessEqual(held_min, mod._SESSION_HOLD_MAX_MIN + 1)
            self.assertGreater(held_min, 0)

    def test_weekly_limit_keeps_full_hold(self) -> None:
        # A genuine weekly limit is NOT bounded by the session cap — it holds to its
        # parsed multi-day reset.
        import tempfile
        import datetime as dt
        mod = load()
        now_utc = dt.datetime(2026, 6, 23, 8, 0, 0)
        now_ts = (now_utc - dt.datetime(1970, 1, 1)).total_seconds()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write_worker(runs, "resolve-9-w.log", self.BANNER, mtime=now_ts - 60)
            out = mod.check_weekly_cap(runs, product="claude", account_tag="smith",
                                       now_ts=now_ts)
            self.assertTrue(out["capped"])
            self.assertEqual(out["kind"], "weekly")
            until = dt.datetime.fromisoformat(out["until"].replace("Z", ""))
            # Jun 25 1pm PDT == Jun 25 20:00 UTC, ~2 days out — far beyond the session cap.
            self.assertEqual(until, dt.datetime(2026, 6, 25, 20, 0, 0))

    def test_evaluate_refuses_when_capped(self) -> None:
        mod = load()
        pre = {"verdict": "SPAWN_OK", "reason": "ok", "cap": 2, "live": 0,
               "account": {"tag": "smith-netra", "tier": 1, "model": "opus", "dir": "/a"}}
        mod.issue_dispatch.refresh_registry = lambda root: {"ok": True}
        mod.issue_dispatch.preflight = lambda root, **kw: pre
        mod.check_weekly_cap = lambda runs_dir, **k: {
            "capped": True, "until": "2026-06-25T20:00:00Z", "reset_text": "Jun 25, 1pm",
            "source": "banner"}
        mod.reap_timed_out_workers = lambda runs_dir, **k: {
            "timeout_s": k.get("timeout_s"), "live": k.get("live"),
            "candidates": [], "reaped": [], "would_reap": []}
        # if the gate works, the lane router / spawn must never be reached:
        mod.lane_issue_numbers = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("capped tick must short-circuit before the lane router"))
        mod.spawn_issue_worker = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("capped tick must never spawn"))
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering", lane=None, live=True)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "WEEKLY_CAPPED")
        self.assertEqual(p["action"], "weekly_capped")
        self.assertIn("Jun 25, 1pm", p["reason"])
        self.assertEqual(p["weekly_cap"]["until"], "2026-06-25T20:00:00Z")


class EvaluateBackendHealthTest(unittest.TestCase):
    """evaluate()-level wiring of the backend-health gate: a dead backend self-
    suppresses (and never spawns) while its re-probe is not due; a healthy backend
    claims a dead sibling's lane + a budget bump."""

    def _common_stubs(self, mod, pre):
        mod.issue_dispatch.refresh_registry = lambda root: {"ok": True}
        mod.issue_dispatch.preflight = lambda root, **kw: dict(pre, _seen_max=kw.get("max_workers"))
        mod.check_weekly_cap = lambda runs_dir, **k: {"capped": False}
        mod.reap_timed_out_workers = lambda runs_dir, **k: {
            "timeout_s": None, "live": k.get("live"),
            "candidates": [], "reaped": [], "would_reap": []}
        mod.prune_dead_sidecars = lambda runs_dir, **k: {"pruned": []}
        # Hermetic: no lane is held, so the pre-spawn lane-lease gate (#1310) never
        # reads the real .dispatch-runs and never trips for these backend-health tests.
        mod.live_resolution_lanes = lambda runs_dir: set()

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


class WaveMembershipTest(unittest.TestCase):
    """A wave is a typed group, not N anonymous lanes: each worker is stamped with
    its rank/wave/size into child_env + a .wave sidecar, so an auditor enumerates
    the whole group from the filesystem and reads the honest under-fill."""

    def test_wave_membership_env_shape(self) -> None:
        mod = load()
        env = mod.wave_membership_env(rank=2, wave_id="wave-abc", size=4, shortfall=1)
        self.assertEqual(env, {
            "FLEET_WAVE_ID": "wave-abc", "FLEET_WAVE_RANK": "2",
            "FLEET_WAVE_SIZE": "4", "FLEET_WAVE_SHORTFALL": "1"})

    def test_enumerate_wave_reads_ranks_and_shortfall_from_sidecars(self) -> None:
        # Simulate a granted-3 wave (requested 5 -> shortfall 2) by writing the same
        # .wave sidecars spawn_issue_worker writes, then enumerate from disk alone.
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            for rank in range(3):
                out_log = runs / f"resolve-{600 + rank}-2026{rank}.log"
                mod.write_wave_sidecar(out_log, rank=rank, wave_id="wave-xyz",
                                       size=3, shortfall=2)
            # A worker from a DIFFERENT wave must not bleed into this enumeration.
            mod.write_wave_sidecar(runs / "resolve-999-other.log", rank=0,
                                   wave_id="wave-other", size=1, shortfall=0)
            grp = mod.enumerate_wave(runs, "wave-xyz")
            self.assertEqual(grp["wave_id"], "wave-xyz")
            self.assertEqual(grp["granted"], 3)
            self.assertEqual(grp["size"], 3)
            self.assertEqual(grp["shortfall"], 2)
            self.assertEqual(grp["ranks"], [0, 1, 2])   # ranks 0..granted-1

    def test_spawn_stamps_membership_env_and_sidecar(self) -> None:
        # End-to-end: spawn_issue_worker stamps FLEET_WAVE_* into the child env AND
        # writes the .wave sidecar — proven without launching a real process.
        import tempfile
        from unittest import mock
        mod = load()
        captured: dict[str, object] = {}

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                captured["env"] = kwargs.get("env")
                kwargs.get("stdout").close()  # the parent's log fh (no real child)
                self.pid = 31337

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            membership = {"rank": 1, "wave_id": "wave-zzz", "size": 2, "shortfall": 0}
            with mock.patch.object(mod.subprocess, "Popen", _FakePopen):
                res = mod.spawn_issue_worker(
                    ["true"], {"DISPATCH_LANE": "tools"}, runs, runs,
                    issue=42, lane="tools", backend="claude", membership=membership)
            env = captured["env"]
            self.assertEqual(env["FLEET_WAVE_RANK"], "1")
            self.assertEqual(env["FLEET_WAVE_ID"], "wave-zzz")
            self.assertEqual(env["FLEET_WAVE_SIZE"], "2")
            self.assertEqual(env["FLEET_WAVE_SHORTFALL"], "0")
            self.assertEqual(env["DISPATCH_LANE"], "tools")  # base env preserved
            self.assertEqual(res["membership"]["rank"], 1)
            # The wave is enumerable from the sidecar that spawn wrote.
            grp = mod.enumerate_wave(runs, "wave-zzz")
            self.assertEqual(grp["ranks"], [1])
            self.assertEqual(grp["shortfall"], 0)

    def test_spawn_stamps_account_sidecar(self) -> None:
        # A silent 0-byte worker log is only useful if it can be attributed back to
        # the selected switcher account. The account sidecar is written at spawn time,
        # before the child has to produce any output.
        import json
        import tempfile
        from unittest import mock
        mod = load()

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                self.pid = 9

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            account = {"tag": "gem8-netra", "tier": 1, "model": "opus", "dir": "/acct/gem8"}
            with mock.patch.object(mod.subprocess, "Popen", _FakePopen):
                res = mod.spawn_issue_worker(["true"], {}, runs, runs,
                                             issue=2, lane="tools", backend="claude",
                                             account=account)
            self.assertEqual(res["account"]["tag"], "gem8-netra")
            sidecars = list(runs.glob("*.account"))
            self.assertEqual(len(sidecars), 1)
            self.assertEqual(json.loads(sidecars[0].read_text(encoding="utf-8")), account)

    def test_spawn_without_membership_writes_no_wave_sidecar(self) -> None:
        # The default single-worker path is unchanged: no membership -> no .wave file.
        import tempfile
        from unittest import mock
        mod = load()

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                kwargs.get("stdout").close()  # the parent's log fh (no real child)
                self.pid = 7

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            with mock.patch.object(mod.subprocess, "Popen", _FakePopen):
                mod.spawn_issue_worker(["true"], {}, runs, runs,
                                       issue=1, lane="tools", backend="claude")
            self.assertEqual(list(runs.glob("*.wave")), [])


class BackendHealthTest(unittest.TestCase):
    """The backend-health reallocation gate: a STREAK of stub (banner-only/0-byte,
    dead-pid) logs for a backend declares it dead; a productive log breaks the streak
    and clears the hold; a still-running stub is never counted."""

    def _mk(self, runs: Path, issue: int, stamp: str, *, backend: str, size: int,
            pid: int | None, mtime: float) -> None:
        import os
        log = runs / f"resolve-{issue}-{stamp}.log"
        log.write_text("x" * size, encoding="utf-8")
        os.utime(log, (mtime, mtime))
        log.with_suffix(".backend").write_text(backend, encoding="utf-8")
        if pid is not None:
            pidf = log.with_suffix(".pid")
            pidf.write_text(str(pid), encoding="utf-8")
            os.utime(pidf, (mtime, mtime))

    def _dead_probe(self):
        # Every pid is dead — the realistic post-mortem state for a stub log.
        return lambda pid: {"alive": False}

    def test_streak_of_stub_logs_declares_dead(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            for i, iss in enumerate((872, 873, 879)):  # 3 == _BACKEND_DEAD_STREAK
                self._mk(runs, iss, f"20260626-04370{i}", backend="opencode",
                         size=32, pid=200 + i, mtime=now - i * 60)
            out = mod.check_backend_health(runs, product="opencode", lane="docs",
                                           now_ts=now, alive=set(), probe=self._dead_probe())
            self.assertEqual(out["state"], "dead")
            self.assertEqual(out["abandoned_lane"], "docs")
            self.assertTrue(out["reprobe_due"])  # first detection -> a re-probe is due
            # The hold is persisted for the ticks after spawns stop.
            self.assertTrue((runs / "backend-health-opencode.json").exists())

    def test_one_productive_log_breaks_streak_and_clears_hold(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # Seed a prior dead hold, then a real turn (large log) lands.
            (runs / "backend-health-opencode.json").write_text(
                '{"product":"opencode","state":"dead","since":"x","abandoned_lane":"docs"}',
                encoding="utf-8")
            self._mk(runs, 900, "20260626-050000", backend="opencode",
                     size=5000, pid=300, mtime=now)  # >512B == productive
            out = mod.check_backend_health(runs, product="opencode",
                                           now_ts=now, alive=set(), probe=self._dead_probe())
            self.assertEqual(out["state"], "healthy")
            self.assertFalse((runs / "backend-health-opencode.json").exists())  # hold cleared

    def test_live_stub_pid_is_not_counted_dead(self) -> None:
        # A claude -p worker streams nothing until its final message: a 0-byte log with
        # a LIVE pid is still-running, not a stub. Must not trip the gate.
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            for i, iss in enumerate((10, 11, 12)):
                self._mk(runs, iss, f"20260626-04000{i}", backend="claude",
                         size=0, pid=400 + i, mtime=now - i * 60)
            def live_probe(pid):
                return {"alive": True, "create_time": now - 1,
                                                  "name": "claude.exe", "cmdline": ""}
            out = mod.check_backend_health(runs, product="claude",
                                           now_ts=now, alive={400, 401, 402},
                                           probe=live_probe)
            self.assertEqual(out["state"], "healthy")  # all still running -> not dead

    def test_fewer_than_streak_is_not_dead(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            for i, iss in enumerate((872, 873)):  # only 2 < streak of 3
                self._mk(runs, iss, f"20260626-04370{i}", backend="opencode",
                         size=32, pid=200 + i, mtime=now - i * 60)
            out = mod.check_backend_health(runs, product="opencode",
                                           now_ts=now, alive=set(), probe=self._dead_probe())
            self.assertEqual(out["state"], "healthy")

    def test_reprobe_gated_after_recent_probe(self) -> None:
        # A second tick within the re-probe window holds (reprobe_due False); a tick
        # past the window admits one re-probe worker again.
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            for i, iss in enumerate((872, 873, 879)):
                self._mk(runs, iss, f"20260626-04370{i}", backend="opencode",
                         size=32, pid=200 + i, mtime=now - i * 60)
            first = mod.check_backend_health(runs, product="opencode", lane="docs",
                                             now_ts=now, alive=set(), probe=self._dead_probe())
            self.assertTrue(first["reprobe_due"])
            soon = mod.check_backend_health(runs, product="opencode", lane="docs",
                                            now_ts=now + 60, alive=set(),
                                            probe=self._dead_probe())  # +1 min < 30
            self.assertFalse(soon["reprobe_due"])
            later = mod.check_backend_health(
                runs, product="opencode", lane="docs",
                now_ts=now + (mod._BACKEND_REPROBE_MIN + 1) * 60, alive=set(),
                probe=self._dead_probe())
            self.assertTrue(later["reprobe_due"])

    def test_read_dead_backends_excludes_self_and_healthy(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            (runs / "backend-health-opencode.json").write_text(
                '{"product":"opencode","state":"dead","abandoned_lane":"docs"}',
                encoding="utf-8")
            (runs / "backend-health-codex.json").write_text(
                '{"product":"codex","state":"dead","abandoned_lane":"model"}',
                encoding="utf-8")
            (runs / "backend-health-claude.json").write_text(
                '{"product":"claude","state":"healthy"}', encoding="utf-8")
            dead = mod.read_dead_backends(runs, exclude="claude")
            prods = sorted(b["product"] for b in dead)
            self.assertEqual(prods, ["codex", "opencode"])  # healthy + self excluded
            # And from the claude tick's view (exclude self), both dead lanes are free.
            lanes = sorted(b["abandoned_lane"] for b in dead)
            self.assertEqual(lanes, ["docs", "model"])

    def test_fail_open_on_missing_runs_dir(self) -> None:
        mod = load()
        out = mod.check_backend_health(Path("/no/such/dir/xyz"), product="opencode")
        self.assertEqual(out["state"], "healthy")  # never wedges the loop


class LaneLeaseHelperTest(unittest.TestCase):
    """The fenced-lease helpers (residual of #1310): acquire maps the three
    `fak leaseref` exit codes (0 acquired / 3 refused / else fail-open) and never
    raises, so the gate can add protection but never wedge the loop."""

    def test_acquire_exit0_is_acquired(self) -> None:
        mod = load()
        def runner(root, args, **k):
            self.assertEqual(args[0], "acquire")
            self.assertIn("--id", args)
            self.assertIn("resolve-gateway", args)
            return {"rc": 0, "verdict": {"verdict": {"ok": True},
                                         "record": {"id": "resolve-gateway", "generation": 2}}}
        out = mod.acquire_lane_lease(ROOT, "gateway", tree=["internal/gateway/**"],
                                     ttl_s=900, runner=runner)
        self.assertTrue(out["acquired"])
        self.assertFalse(out["refused"])
        self.assertEqual(out["id"], "resolve-gateway")
        self.assertEqual(out["generation"], 2)

    def test_acquire_exit3_is_refused(self) -> None:
        mod = load()
        def runner(root, args, **k):
            return {"rc": 3, "verdict": {"verdict": {"ok": False, "reason": "LEASE_HELD"}}}
        out = mod.acquire_lane_lease(ROOT, "docs", tree=["docs/**"], ttl_s=900, runner=runner)
        self.assertTrue(out["refused"])
        self.assertFalse(out["acquired"])
        self.assertEqual(out["reason"], "LEASE_HELD")

    def test_acquire_argv_carries_id_ttl_and_every_tree(self) -> None:
        # Regression guard for the FENCE parameters: the acquire argv MUST carry the
        # TTL (drop it and the lease never expires -> a crashed holder wedges the lane
        # forever, defeating the reap backstop) and ONE --tree per lane glob (drop them
        # and the arbiter cannot reason cross-lane disjointness). Pin the exact argv so
        # a refactor cannot silently weaken the lease the way the issue's gate depends on.
        mod = load()
        seen: dict = {}
        def runner(root, args, **k):
            seen["args"] = list(args)
            return {"rc": 0, "verdict": {"verdict": {"ok": True},
                                         "record": {"id": "resolve-docs", "generation": 1}}}
        out = mod.acquire_lane_lease(ROOT, "docs", tree=["docs/**", "internal/spec/**"],
                                     ttl_s=1234, holder="sess-7", runner=runner)
        self.assertTrue(out["acquired"])
        argv = seen["args"]
        self.assertEqual(argv[0], "acquire")
        # --id resolve-<lane>, --holder, and the exact --ttl value are all present.
        self.assertEqual(argv[argv.index("--id") + 1], "resolve-docs")
        self.assertEqual(argv[argv.index("--holder") + 1], "sess-7")
        self.assertEqual(argv[argv.index("--ttl") + 1], "1234")
        # one --tree per glob, each glob carried through verbatim.
        self.assertEqual(argv.count("--tree"), 2)
        self.assertIn("docs/**", argv)
        self.assertIn("internal/spec/**", argv)

    def test_acquire_other_exit_fails_open(self) -> None:
        mod = load()
        for rc in (1, 2, 127):
            def runner(root, args, _rc=rc, **k):
                return {"rc": _rc, "verdict": None}
            out = mod.acquire_lane_lease(ROOT, "gateway", tree=["internal/gateway/**"],
                                         ttl_s=900, runner=runner)
            self.assertFalse(out["acquired"], rc)
            self.assertFalse(out["refused"], rc)
            self.assertTrue(out["fail_open"], rc)

    def test_no_fak_binary_fails_open(self) -> None:
        # With no FAK_BIN and no `fak` on PATH, the default runner reports rc 127 and
        # the gate fails open. Exercise the real _run_lease via a patched _fak_bin.
        mod = load()
        mod._fak_bin = lambda root: None
        out = mod.acquire_lane_lease(ROOT, "gateway", tree=["internal/gateway/**"], ttl_s=900)
        self.assertTrue(out["fail_open"])
        self.assertEqual(out["rc"], 127)

    def test_lane_tree_falls_back_to_convention(self) -> None:
        # A dos doctor failure (no taxonomy) -> the internal/<lane>/** convention.
        mod = load()
        mod._LANE_TREE_CACHE.clear()
        import types
        mod_router = types.SimpleNamespace(
            lane_taxonomy=lambda root: (_ for _ in ()).throw(RuntimeError("no dos")))
        sys.modules["issue_lane_router"] = mod_router
        try:
            self.assertEqual(mod.lane_tree(ROOT, "gateway"), ["internal/gateway/**"])
        finally:
            sys.modules.pop("issue_lane_router", None)
            mod._LANE_TREE_CACHE.clear()


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
        mod.recently_attempted_issues = lambda runs_dir, *, cooldown_min, **k: set()
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

    def test_spawn_failure_surfaces_lease_held_until_ttl(self) -> None:
        # `fak leaseref` has no delete-one-live verb, so a worker that died at exec
        # leaves its lane lease held until the TTL lapses and a later tick's `reap`
        # sweeps it. The SPAWN_FAILED record must be honest about that held lease.
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
                                             "record": {"id": "resolve-gateway", "generation": 1}}}
            return {"rc": 0, "verdict": None}
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering", lane="gateway",
                         live=True, spawn_probe_s=5.0, lease_runner=lease_runner)
        self.assertEqual(p["verdict"], "SPAWN_FAILED")
        self.assertEqual(p["lease_held_until_ttl"]["id"], "resolve-gateway")

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


class SubjectCitesIssueTest(unittest.TestCase):
    """The per-worker commit binding key (#1324 proposal #2): a subject is THIS
    worker's commit only when it names the worker's #issue at a word boundary —
    never a numeric prefix/suffix or a glued token."""

    def test_word_boundary_binding(self) -> None:
        mod = load()
        self.assertTrue(mod._subject_cites_issue("fix(tools): bind slot (#1324) (fak tools)", 1324))
        self.assertTrue(mod._subject_cites_issue("resolve #1324 now", 1324))
        # a longer number that merely CONTAINS the digits is not a match
        self.assertFalse(mod._subject_cites_issue("fix #13240 thing", 1324))
        self.assertFalse(mod._subject_cites_issue("fix #132 thing", 1324))
        # a glued token (leading word char) is not a binding reference
        self.assertFalse(mod._subject_cites_issue("xoxb-secret#1324", 1324))
        self.assertFalse(mod._subject_cites_issue("no reference here", 1324))


class WorkerResolvingShaTest(unittest.TestCase):
    """worker_resolving_sha picks the NEWEST commit citing the worker's issue, scoped
    to the per-worker base..HEAD window when known, and yields None (claims nothing)
    when no resolving commit exists — the wrong-issue / no-commit slot."""

    def test_picks_newest_matching_subject(self) -> None:
        mod = load()
        # git log is newest-first; only the second line cites #1324.
        log = "deadbeef\x1fchore: bump (#99)\ncafef00d\x1ffix: bind (#1324) (fak tools)\n"
        out = mod.worker_resolving_sha(ROOT, 1324, git=lambda root, args: (0, log))
        self.assertEqual(out, "cafef00d")

    def test_wrong_issue_only_yields_none(self) -> None:
        mod = load()
        log = "deadbeef\x1fchore: bump (#99)\ncafef00d\x1ffix: other (#42)\n"
        self.assertIsNone(mod.worker_resolving_sha(ROOT, 1324, git=lambda root, args: (0, log)))

    def test_base_sha_scopes_the_range(self) -> None:
        mod = load()
        seen: dict = {}
        def git(root, args):
            seen["args"] = list(args)
            return (0, "cafef00d\x1ffix: bind (#1324)\n")
        mod.worker_resolving_sha(ROOT, 1324, base_sha="base000", git=git)
        self.assertIn("base000..HEAD", seen["args"])

    def test_no_base_falls_back_to_recent_window(self) -> None:
        mod = load()
        seen: dict = {}
        def git(root, args):
            seen["args"] = list(args)
            return (0, "")
        mod.worker_resolving_sha(ROOT, 1324, git=git, scan_limit=42)
        self.assertIn("-n", seen["args"])
        self.assertIn("42", seen["args"])

    def test_git_error_fails_open_to_none(self) -> None:
        mod = load()
        self.assertIsNone(mod.worker_resolving_sha(ROOT, 1324, git=lambda root, args: (127, "")))


class AuditCommitWitnessTest(unittest.TestCase):
    """audit_commit_witness grades a sha through the DOS witness rung: witnessed only
    on verdict OK AND a diff-witness; everything else (ABSTAIN, subject-only, an empty
    audit) is the conservative not-witnessed verdict."""

    def test_ok_diff_witnessed_is_witnessed(self) -> None:
        mod = load()
        out = mod.audit_commit_witness(
            ROOT, "abc", runner=lambda root, sha: {"verdict": "OK", "witness": "diff-witnessed"})
        self.assertTrue(out["witnessed"])
        self.assertEqual(out["verdict"], "OK")

    def test_abstain_is_not_witnessed(self) -> None:
        mod = load()
        out = mod.audit_commit_witness(
            ROOT, "abc", runner=lambda root, sha: {"verdict": "ABSTAIN", "witness": "abstain"})
        self.assertFalse(out["witnessed"])

    def test_subject_only_is_not_witnessed(self) -> None:
        mod = load()
        out = mod.audit_commit_witness(
            ROOT, "abc", runner=lambda root, sha: {"verdict": "OK", "witness": "subject-only"})
        self.assertFalse(out["witnessed"])

    def test_empty_audit_fails_open_to_not_witnessed(self) -> None:
        mod = load()
        out = mod.audit_commit_witness(ROOT, "abc", runner=lambda root, sha: {})
        self.assertFalse(out["witnessed"])


class WitnessExitedWorkersTest(unittest.TestCase):
    """The commit-time witness sweep (#1324 proposal #2): each FINISHED (dead-pid)
    worker's slot is graded via dos commit-audit and recorded CLAIM_WITNESSED /
    CLAIM_UNWITNESSED / CLAIM_NO_COMMIT — never a silent productive claim. Dead-pid
    gated and write-on-live, exactly like prune_dead_sidecars."""

    def _mk(self, runs: Path, issue: int, stamp: str, *, pid: int,
            base: str | None = "base000") -> Path:
        log = runs / f"resolve-{issue}-{stamp}.log"
        log.write_text("worker output here", encoding="utf-8")
        log.with_suffix(".pid").write_text(str(pid), encoding="utf-8")
        if base is not None:
            log.with_suffix(".basesha").write_text(base, encoding="utf-8")
        return log

    @staticmethod
    def _dead(pid):
        return {"alive": False}

    def test_unwitnessed_commit_is_recorded_not_silently_productive(self) -> None:
        # The issue's acceptance witness #2: a worker that exits with an unwitnessed
        # commit is recorded CLAIM_UNWITNESSED, and a .witness sidecar persists it.
        import json
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = self._mk(runs, 1324, "20260629-120000", pid=4242)
            def git(root, args):
                return (0, "cafef00d\x1ffix: bind (#1324)\n")
            def audit(root, sha):
                return {"verdict": "ABSTAIN", "witness": "abstain"}
            out = mod.witness_exited_workers(runs, ROOT, live=True, probe=self._dead,
                                             git=git, audit_runner=audit)
            self.assertEqual(len(out["unwitnessed"]), 1)
            self.assertEqual(out["unwitnessed"][0]["claim"], "CLAIM_UNWITNESSED")
            self.assertEqual(out["unwitnessed"][0]["sha"], "cafef00d")
            side = log.with_suffix(".witness")
            self.assertTrue(side.exists())
            self.assertEqual(json.loads(side.read_text(encoding="utf-8"))["claim"],
                             "CLAIM_UNWITNESSED")

    def test_diff_witnessed_commit_claims_the_slot(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 1324, "20260629-120100", pid=4243)
            def git(root, args):
                return (0, "cafef00d\x1ffix: bind (#1324)\n")
            def audit(root, sha):
                return {"verdict": "OK", "witness": "diff-witnessed"}
            out = mod.witness_exited_workers(runs, ROOT, live=True, probe=self._dead,
                                             git=git, audit_runner=audit)
            self.assertEqual(len(out["witnessed"]), 1)
            self.assertEqual(out["witnessed"][0]["claim"], "CLAIM_WITNESSED")

    def test_no_resolving_commit_is_claim_no_commit(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 1324, "20260629-120200", pid=4244)
            # git finds nothing citing #1324 -> the worker landed no resolving commit.
            def git(root, args):
                return (0, "deadbeef\x1fchore: unrelated (#5)\n")
            def boom(root, sha):
                raise AssertionError("no audit without a sha")
            out = mod.witness_exited_workers(runs, ROOT, live=True, probe=self._dead,
                                             git=git, audit_runner=boom)
            self.assertEqual(len(out["no_commit"]), 1)
            self.assertEqual(out["no_commit"][0]["claim"], "CLAIM_NO_COMMIT")
            self.assertIsNone(out["no_commit"][0]["sha"])

    def test_live_worker_is_not_audited(self) -> None:
        # A still-running worker may not have committed yet — never mis-blame it.
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 1324, "20260629-120300", pid=4245)
            def alive_probe(pid):
                return {"alive": True, "create_time": 0.0,
                        "name": "claude.exe",
                        "cmdline": "claude -p resolve GitHub issue #1324"}
            def boom_git(root, args):
                raise AssertionError("a live worker must not be audited")
            out = mod.witness_exited_workers(runs, ROOT, live=True, probe=alive_probe,
                                             git=boom_git)
            self.assertEqual(out["audited"], [])

    def test_already_witnessed_is_skipped(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = self._mk(runs, 1324, "20260629-120400", pid=4246)
            log.with_suffix(".witness").write_text('{"claim": "CLAIM_WITNESSED"}',
                                                   encoding="utf-8")
            def boom_git(root, args):
                raise AssertionError("an already-witnessed worker must not be re-audited")
            out = mod.witness_exited_workers(runs, ROOT, live=True, probe=self._dead,
                                             git=boom_git)
            self.assertEqual(out["audited"], [])

    def test_dry_run_audits_but_writes_no_sidecar(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = self._mk(runs, 1324, "20260629-120500", pid=4247)
            def git(root, args):
                return (0, "cafef00d\x1ffix: bind (#1324)\n")
            def audit(root, sha):
                return {"verdict": "ABSTAIN", "witness": "abstain"}
            out = mod.witness_exited_workers(runs, ROOT, live=False, probe=self._dead,
                                             git=git, audit_runner=audit)
            self.assertEqual(out["unwitnessed"][0]["claim"], "CLAIM_UNWITNESSED")
            self.assertFalse(log.with_suffix(".witness").exists())

    def test_no_pid_sidecar_is_not_auditable(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # a log with no .pid sidecar: we cannot prove it finished.
            (runs / "resolve-1324-20260629-120600.log").write_text("x", encoding="utf-8")
            def boom_git(root, args):
                raise AssertionError("a worker we cannot prove finished must not be audited")
            out = mod.witness_exited_workers(runs, ROOT, live=True, probe=self._dead,
                                             git=boom_git)
            self.assertEqual(out["audited"], [])


if __name__ == "__main__":
    unittest.main()
