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
            probe = lambda pid: {"alive": True, "create_time": now - 1, "cmdline": ""}
            self.assertEqual(mod.live_resolution_issues(runs, probe=probe), {717})

    def test_rejects_issue_when_sidecar_pid_was_reused(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 718, "20260625-060712", pid=102, sidecar_mtime=now)
            probe = lambda pid: {
                "alive": True,
                "create_time": now + 60 * 60,
                "name": "chrome.exe",
                "cmdline": "chrome.exe --type=renderer",
            }
            self.assertEqual(mod.live_resolution_issues(runs, probe=probe), set())


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
            probe = lambda pid: {"alive": True, "create_time": now - 4001, "cmdline": ""}
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
            probe = lambda pid: {"alive": True, "create_time": now - 4001, "cmdline": ""}
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
            probe = lambda pid: {"alive": True, "create_time": now - 61, "cmdline": ""}
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
            probe = lambda pid: {
                "alive": True,
                "create_time": now + 60,
                "cmdline": "chrome.exe --type=renderer",
            }
            out = mod.reap_timed_out_workers(
                runs, timeout_s=1800, live=True, now_ts=now, probe=probe,
                killer=lambda pid: {"ok": True})
        self.assertEqual(out["candidates"], [])
        self.assertEqual(out["reaped"], [])

    def test_timeout_zero_disables_reap(self) -> None:
        mod = load()
        out = mod.reap_timed_out_workers(Path("does-not-matter"), timeout_s=None, live=True)
        self.assertTrue(out["disabled"])


class EvaluateTest(unittest.TestCase):
    SPAWN_OK = {
        "verdict": "SPAWN_OK", "reason": "ok", "cap": 2, "live": 0,
        "account": {"tag": "worker-a", "tier": 1, "model": "opus", "dir": "/acct/a"},
    }

    def _patch(self, mod, *, pre, pick, live_issues=None, cooled=None,
               prompt_chars=900) -> None:
        mod.issue_dispatch.refresh_registry = lambda root: {"ok": True}
        mod.issue_dispatch.preflight = lambda root, **kw: pre
        mod.lane_issue_numbers = lambda root, lane, exclude=None: pick
        mod.live_resolution_issues = lambda runs_dir: set(live_issues or [])
        mod.recently_attempted_issues = lambda runs_dir, *, cooldown_min, **k: set(cooled or [])
        mod.issue_worker_prompt.build = lambda n, lane, *, workspace: {
            "prompt": f"resolve #{n}", "prompt_chars": prompt_chars, "title": f"title {n}"}
        mod.check_weekly_cap = lambda runs_dir, **k: {"capped": False}  # hermetic: not capped
        mod.reap_timed_out_workers = lambda runs_dir, **k: {
            "timeout_s": k.get("timeout_s"), "live": k.get("live"),
            "candidates": [], "reaped": [], "would_reap": []}

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


if __name__ == "__main__":
    unittest.main()
