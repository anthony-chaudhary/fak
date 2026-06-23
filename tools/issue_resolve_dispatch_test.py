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


if __name__ == "__main__":
    unittest.main()
