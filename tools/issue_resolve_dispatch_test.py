#!/usr/bin/env python3
"""Hermetic tests for tools/issue_resolve_dispatch.py.

Every shell-out (registry refresh, preflight, lane router, prompt build, spawn)
is stubbed on the module; NOTHING live (gh/dos/claude) runs and no worker is
spawned in dry-run. The pure pickers (pick_target_issue, lane fold) are tested
directly.
"""
from __future__ import annotations

import importlib.util
import ast
import atexit
import datetime as dt
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock
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


class PickTargetTest(unittest.TestCase):
    def test_first_not_in_skip(self) -> None:
        mod = load()
        self.assertEqual(mod.pick_target_issue([497, 496, 495], set()), 497)
        self.assertEqual(mod.pick_target_issue([497, 496, 495], {497}), 496)
        self.assertEqual(mod.pick_target_issue([497, 496], {497, 496}), None)
        self.assertEqual(mod.pick_target_issue([], set()), None)


class ContractOverlayTest(unittest.TestCase):
    def test_record_merges_overlay_into_body(self) -> None:
        mod = load()
        rec = mod._issue_record_for_contract(
            {"number": 7, "title": "t", "body": "base body"}, 7,
            "Done condition: the gate passes")
        self.assertIn("base body", rec["body"])
        self.assertIn("Done condition: the gate passes", rec["body"])
        self.assertIn("local contract overlay", rec["body"])  # legible merge marker
        # No overlay -> the record is byte-identical to before.
        self.assertEqual(
            mod._issue_record_for_contract({"number": 7, "body": "base body"}, 7)["body"],
            "base body")

    def test_contract_review_invokes_fak_dev_after_command_split(self) -> None:
        mod = load()
        seen = []

        def runner(command, **_kwargs):
            seen.append(command)
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout=json.dumps({
                    "ok": True,
                    "reviews": [{"ok": True, "score": {"total": 100},
                                 "spine_priority": {"total": 100}}],
                }), stderr="")

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
                mod, "_fak_dev_command_prefix", return_value=["fak-dev"]):
            review = mod.issue_contract_review(
                Path(td), {"number": 7, "title": "t", "body": "body"}, 7,
                runner=runner)
        self.assertTrue(review["ok"])
        self.assertEqual(seen[0][:3], ["fak-dev", "issue", "contract"])

    def test_read_and_times_roundtrip(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td)
            self.assertEqual(mod.read_contract_overlay(runs, 1207), "")
            self.assertEqual(mod.contract_overlay_times(runs), {})
            path = mod.contract_overlay_path(runs, 1207)
            path.parent.mkdir(parents=True)
            path.write_text("Scope: one file\n", encoding="utf-8")
            (path.parent / "not-an-overlay.md").write_text("x", encoding="utf-8")
            self.assertEqual(mod.read_contract_overlay(runs, 1207), "Scope: one file")
            times = mod.contract_overlay_times(runs)
            self.assertEqual(sorted(times), [1207])
            self.assertGreater(times[1207], 0)


class ContractScanStreamTest(unittest.TestCase):
    def test_round_robin_across_lanes_oldest_first_within(self) -> None:
        mod = load()
        eligible = [["docs", [1, 2, 3]], ["tools", [10]], ["policy", [20, 21]]]
        self.assertEqual(
            mod.contract_scan_stream(eligible, skip=set()),
            [("docs", 1), ("tools", 10), ("policy", 20),
             ("docs", 2), ("policy", 21), ("docs", 3)])

    def test_skip_and_empty_lanes_drop_out(self) -> None:
        mod = load()
        eligible = [["docs", [1, 2]], ["tools", [10]], ["bench", []]]
        self.assertEqual(mod.contract_scan_stream(eligible, skip={1, 10}),
                         [("docs", 2)])
        self.assertEqual(mod.contract_scan_stream([], skip=set()), [])
        self.assertEqual(mod.contract_scan_stream(None, skip=set()), [])


class OpencodeWorkerEnvTest(unittest.TestCase):
    def test_pinning_xdg_preserves_windows_gh_config(self) -> None:
        import os
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            appdata = root / "AppData" / "Roaming"
            gh_config = appdata / "GitHub CLI"
            gh_config.mkdir(parents=True)
            account = root / "opencode"
            account.mkdir()
            with mock.patch.dict(os.environ, {
                "APPDATA": str(appdata),
                "USERPROFILE": str(root / "home"),
            }, clear=True):
                env = mod.opencode_worker_env(
                    str(account), "contract-repair", root, root / ".dispatch-runs")
        self.assertEqual(env["XDG_CONFIG_HOME"], str(root))
        self.assertEqual(env["GH_CONFIG_DIR"], str(gh_config))

    def test_explicit_gh_config_dir_wins(self) -> None:
        import os
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            explicit = root / "custom-gh"
            account = root / "opencode"
            account.mkdir()
            with mock.patch.dict(os.environ, {
                "GH_CONFIG_DIR": str(explicit),
                "APPDATA": str(root / "AppData" / "Roaming"),
            }, clear=True):
                env = mod.opencode_worker_env(
                    str(account), "contract-repair", root, root / ".dispatch-runs")
        self.assertEqual(env["GH_CONFIG_DIR"], str(explicit))

    def test_temp_vars_are_inside_dispatch_runs(self) -> None:
        import os
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            account = root / "opencode"
            account.mkdir()
            runs = root / ".dispatch-runs"
            with mock.patch.dict(os.environ, {
                "TEMP": str(root / "external-temp"),
                "TMP": str(root / "external-tmp"),
                "TMPDIR": str(root / "external-tmpdir"),
            }, clear=True):
                env = mod.opencode_worker_env(str(account), "docs/tools", root, runs)

            expected = root / ".dispatch-runs" / ".opencode-tmp" / "docs-tools"
            self.assertTrue(expected.is_dir())
            self.assertEqual(env["TEMP"], str(expected))
            self.assertEqual(env["TMP"], str(expected))
            self.assertEqual(env["TMPDIR"], str(expected))


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

    def test_recent_witness_only_attempt_is_in_cooldown(self) -> None:
        import os
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            now = 1_000_000.0
            fresh = runs / "resolve-2898-20260705-214923.witness"
            fresh.write_text('{"claim": "CLAIM_NO_COMMIT", "issue": 2898}',
                             encoding="utf-8")
            os.utime(fresh, (now - 5 * 60, now - 5 * 60))
            cooled = mod.recently_attempted_issues(runs, cooldown_min=120, now_ts=now)
        self.assertEqual(cooled, {2898})

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


class ActiveGuardLivelockTest(unittest.TestCase):
    def test_live_result_quarantine_journal_becomes_spawn_hold(self) -> None:
        import json
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = root / ".dispatch-runs"
            audit_dir = runs / "guard-audit"
            audit_dir.mkdir(parents=True)
            audit = audit_dir / "tools-claude-1234.jsonl"
            rows = [{
                "verdict": "QUARANTINE",
                "tool": "tool_result",
                "reason": "SECRET_EXFIL",
                "args_digest": "abc123",
            } for _ in range(10)]
            audit.write_text("\n".join(json.dumps(r) for r in rows),
                             encoding="utf-8")
            log = runs / "resolve-2720-20260705-202908.log"
            log.write_text(f"audit log  : {audit}\n", encoding="utf-8")
            log.with_suffix(".pid").write_text("1234\n", encoding="utf-8")

            with mock.patch.object(mod.dispatch_preflight,
                                   "resolve_sidecar_pid_is_live",
                                   return_value=True):
                hold = mod.active_guard_livelock_hold(root, runs)

        self.assertTrue(hold["active"])
        self.assertIn("#2720", hold["reason"])
        self.assertEqual(hold["candidates"][0]["issue"], 2720)
        self.assertEqual(hold["candidates"][0]["count"], 10)
        self.assertEqual(hold["candidates"][0]["reason"], "SECRET_EXFIL")
        self.assertEqual(hold["candidates"][0]["digest"], "abc123")

    def test_below_threshold_result_quarantine_fails_open(self) -> None:
        import json
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = root / ".dispatch-runs"
            audit_dir = runs / "guard-audit"
            audit_dir.mkdir(parents=True)
            audit = audit_dir / "tools-claude-1234.jsonl"
            rows = [{
                "verdict": "QUARANTINE",
                "tool": "tool_result",
                "reason": "SECRET_EXFIL",
                "args_digest": "abc123",
            } for _ in range(9)]
            audit.write_text("\n".join(json.dumps(r) for r in rows),
                             encoding="utf-8")
            log = runs / "resolve-2720-20260705-202908.log"
            log.write_text(f"audit log  : {audit}\n", encoding="utf-8")
            log.with_suffix(".pid").write_text("1234\n", encoding="utf-8")

            with mock.patch.object(mod.dispatch_preflight,
                                   "resolve_sidecar_pid_is_live",
                                   return_value=True):
                hold = mod.active_guard_livelock_hold(root, runs)

        self.assertFalse(hold["active"])

    @staticmethod
    def _livelock_fixture(root: Path, rows: list[dict], *,
                          lane: str | None = "quality") -> tuple[Path, Path]:
        """Write a live worker's resolve log + guard journal under ``root``."""
        import json
        runs = root / ".dispatch-runs"
        audit_dir = runs / "guard-audit"
        audit_dir.mkdir(parents=True)
        audit = audit_dir / "tools-claude-1234.jsonl"
        audit.write_text("\n".join(json.dumps(r) for r in rows),
                         encoding="utf-8")
        log = runs / "resolve-2720-20260705-202908.log"
        header = (f"# fak-spawn issue=2720 lane={lane} backend=claude\n"
                  if lane else "")
        log.write_text(f"{header}audit log  : {audit}\n", encoding="utf-8")
        log.with_suffix(".pid").write_text("1234\n", encoding="utf-8")
        return runs, audit

    @staticmethod
    def _quarantine_row() -> dict:
        return {"kind": "QUARANTINE", "verdict": "QUARANTINE",
                "tool": "tool_result", "reason": "SECRET_EXFIL",
                "args_digest": "abc123"}

    @staticmethod
    def _clean_row(i: int) -> dict:
        return {"kind": "DECIDE", "verdict": "ALLOW", "tool": "Bash",
                "args_digest": f"clean{i}"}

    def test_guard_livelock_clears_once_the_worker_escapes(self) -> None:
        """A worker that already broke OUT of the livelock is not livelocked.

        Regression for #5861: the scan counted all-time quarantines in an
        append-only journal, so the tally could never fall back under the
        threshold. Once a worker crossed it the hold LATCHED for the life of the
        journal -- here, through 400 consecutive clean decisions after the
        livelock ended. The hold must describe a livelock happening NOW.
        """
        import tempfile
        mod = load()
        rows = ([self._quarantine_row() for _ in range(10)]
                + [self._clean_row(i) for i in range(400)])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs, audit = self._livelock_fixture(root, rows)
            with mock.patch.object(mod.dispatch_preflight,
                                   "resolve_sidecar_pid_is_live",
                                   return_value=True):
                hold = mod.active_guard_livelock_hold(root, runs)
                hit = mod._guard_result_livelock(audit)

        self.assertFalse(hold["active"])
        self.assertFalse(hit["livelock"])
        # The escape was SEEN and the pre-escape burst forgotten, not merely
        # outweighed -- so the hold clears on recovery instead of latching.
        self.assertEqual(hit["escapes"], 1)
        self.assertEqual(hit["count"], 0)
        self.assertEqual(hit["rows"], 410)

    def test_scattered_quarantines_are_not_a_livelock(self) -> None:
        """Ten quarantines spread singly across thousands of productive rows --
        never two adjacent -- is a lifetime tally, not a livelock (#5861)."""
        import tempfile
        mod = load()
        rows: list[dict] = []
        for burst in range(10):
            rows.extend(self._clean_row(burst * 200 + i) for i in range(200))
            rows.append(self._quarantine_row())
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs, audit = self._livelock_fixture(root, rows)
            with mock.patch.object(mod.dispatch_preflight,
                                   "resolve_sidecar_pid_is_live",
                                   return_value=True):
                hold = mod.active_guard_livelock_hold(root, runs)
                hit = mod._guard_result_livelock(audit)

        self.assertFalse(hold["active"])
        self.assertFalse(hit["livelock"])
        # Each repeat lands a full escape window after the previous one, so each
        # starts a fresh burst instead of accumulating into a false fuse.
        self.assertEqual(hit["count"], 1)

    def test_tight_livelock_still_fires_after_the_escape_window(self) -> None:
        """The recency window must not disarm a worker that IS spinning now."""
        import tempfile
        mod = load()
        # Long productive prefix, then a tight burst that never lets up.
        rows = ([self._clean_row(i) for i in range(300)]
                + [self._quarantine_row() for _ in range(12)])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs, audit = self._livelock_fixture(root, rows)
            with mock.patch.object(mod.dispatch_preflight,
                                   "resolve_sidecar_pid_is_live",
                                   return_value=True):
                hold = mod.active_guard_livelock_hold(root, runs)
                hit = mod._guard_result_livelock(audit)

        self.assertTrue(hold["active"])
        self.assertEqual(hit["count"], 12)
        self.assertEqual(hit["rows_since"], 0)

    def test_guard_livelock_stamps_the_lane_it_burns(self) -> None:
        """Each livelock candidate names its lane, so the hold can be lane-scoped
        instead of idling every disjoint lane (#5861)."""
        import tempfile
        mod = load()
        rows = [self._quarantine_row() for _ in range(10)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs, _ = self._livelock_fixture(root, rows, lane="quality")
            with mock.patch.object(mod.dispatch_preflight,
                                   "resolve_sidecar_pid_is_live",
                                   return_value=True):
                hold = mod.active_guard_livelock_hold(root, runs)

        self.assertTrue(hold["active"])
        self.assertEqual(hold["candidates"][0]["lane"], "quality")
        self.assertEqual(hold["lanes"], ["quality"])
        self.assertFalse(hold["lane_unknown"])
        self.assertIn("lane 'quality'", hold["reason"])

    def test_guard_livelock_without_spawn_header_reports_unknown_lane(self) -> None:
        """An unreadable spawn header yields no lane, and the hold SAYS so, so
        the caller can fail closed rather than guess disjointness (#5861)."""
        import tempfile
        mod = load()
        rows = [self._quarantine_row() for _ in range(10)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs, _ = self._livelock_fixture(root, rows, lane=None)
            with mock.patch.object(mod.dispatch_preflight,
                                   "resolve_sidecar_pid_is_live",
                                   return_value=True):
                hold = mod.active_guard_livelock_hold(root, runs)

        self.assertTrue(hold["active"])
        self.assertIsNone(hold["candidates"][0]["lane"])
        self.assertEqual(hold["lanes"], [])
        self.assertTrue(hold["lane_unknown"])


class ActiveCompactRunawayTest(unittest.TestCase):
    def test_live_worker_past_compact_becomes_spawn_hold(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = root / ".dispatch-runs"
            runs.mkdir(parents=True)
            log = runs / "resolve-2311-20260705-205323.log"
            lines = [
                "fak-turn trace=guard ok compact=fired finish=tool_use "
                "budget=spent:81.5k ctx:81.1k/48.0k dist:33.1k-past-compact",
                "fak-turn trace=guard ok compact=fired finish=tool_use "
                "budget=spent:83.7k ctx:83.3k/48.0k dist:35.3k-past-compact",
                "fak-turn trace=guard ok compact=fired finish=tool_use "
                "budget=spent:93.9k ctx:93.5k/48.0k dist:45.5k-past-compact",
            ]
            log.write_text("\n".join(lines), encoding="utf-8")
            log.with_suffix(".pid").write_text("1234\n", encoding="utf-8")

            with mock.patch.object(mod.dispatch_preflight,
                                   "resolve_sidecar_pid_is_live",
                                   return_value=True):
                hold = mod.active_compact_runaway_hold(root, runs)

        self.assertTrue(hold["active"])
        self.assertIn("#2311", hold["reason"])
        self.assertEqual(hold["candidates"][0]["issue"], 2311)
        self.assertEqual(hold["candidates"][0]["count"], 3)
        self.assertEqual(hold["candidates"][0]["max_past_k"], 45.5)

    def test_compact_runaway_requires_repeated_past_compact_tool_turns(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = root / ".dispatch-runs"
            runs.mkdir(parents=True)
            log = runs / "resolve-2311-20260705-205323.log"
            lines = [
                "fak-turn trace=guard ok compact=fired finish=tool_use "
                "budget=spent:51.5k ctx:51.1k/48.0k dist:3.1k-past-compact",
                "fak-turn trace=guard ok compact=fired finish=tool_use "
                "budget=spent:83.7k ctx:83.3k/48.0k dist:35.3k-past-compact",
                "fak-turn trace=guard ok compact=fired finish=stop "
                "budget=spent:93.9k ctx:93.5k/48.0k dist:45.5k-past-compact",
            ]
            log.write_text("\n".join(lines), encoding="utf-8")
            log.with_suffix(".pid").write_text("1234\n", encoding="utf-8")

            with mock.patch.object(mod.dispatch_preflight,
                                   "resolve_sidecar_pid_is_live",
                                   return_value=True):
                hold = mod.active_compact_runaway_hold(root, runs)

        self.assertFalse(hold["active"])

    def test_compact_runaway_clears_after_compaction_shed(self) -> None:
        """A worker the harness ALREADY pulled back under budget is not a runaway.

        Regression for #5858: the scan used to count all-time past-compact turns,
        so once a worker overshot, the hold LATCHED for the rest of its life even
        though compaction fired, shed the context, and the worker kept making
        progress. The hold must describe compact control failing NOW.
        """
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = root / ".dispatch-runs"
            runs.mkdir(parents=True)
            log = runs / "resolve-4568-20260807-041054.log"
            lines = [
                "# fak-spawn issue=4568 lane=quality backend=claude",
                # Overshoot: 3 turns >=20k past compact (the old latch trigger).
                "fak-turn trace=guard ok compact=none-past-budget finish=tool_use "
                "budget=spent:126.3k ctx:126.3k/96.0k dist:30.3k-past-compact",
                "fak-turn trace=guard ok compact=none-past-budget finish=tool_use "
                "budget=spent:130.1k ctx:130.1k/96.0k dist:34.1k-past-compact",
                "fak-turn trace=guard ok compact=none-past-budget finish=tool_use "
                "budget=spent:134.1k ctx:134.1k/96.0k dist:38.1k-past-compact",
                # Compaction FIRES and sheds 66k: proof compact control works.
                "fak-turn trace=guard ok compact=fired finish=tool_use "
                "budget=spent:68.1k ctx:68.1k/96.0k",
                "fak-turn trace=guard ok compact=fired finish=tool_use "
                "budget=spent:90.3k ctx:90.3k/96.0k",
                "fak-turn trace=guard ok compact=fired finish=tool_use "
                "budget=spent:93.3k ctx:93.3k/96.0k",
            ]
            log.write_text("\n".join(lines), encoding="utf-8")
            log.with_suffix(".pid").write_text("1234\n", encoding="utf-8")

            with mock.patch.object(mod.dispatch_preflight,
                                   "resolve_sidecar_pid_is_live",
                                   return_value=True):
                hold = mod.active_compact_runaway_hold(root, runs)
                hit = mod._compact_runaway_from_log(log)

        self.assertFalse(hold["active"])
        self.assertFalse(hit["runaway"])
        # The shed was seen and the pre-shed overshoot forgotten, not merely
        # outweighed -- so the hold clears immediately on recovery.
        self.assertEqual(hit["sheds"], 1)
        self.assertEqual(hit["count"], 0)

    def test_compact_runaway_stamps_the_lane_it_burns(self) -> None:
        """Each runaway candidate names its lane, so the hold can be lane-scoped
        instead of idling every disjoint lane (#5858)."""
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            runs = root / ".dispatch-runs"
            runs.mkdir(parents=True)
            log = runs / "resolve-4568-20260807-041054.log"
            lines = [
                "# fak-spawn issue=4568 lane=quality backend=claude",
                "fak-turn trace=guard ok compact=none-past-budget finish=tool_use "
                "budget=spent:126.3k ctx:126.3k/96.0k dist:30.3k-past-compact",
                "fak-turn trace=guard ok compact=none-past-budget finish=tool_use "
                "budget=spent:130.1k ctx:130.1k/96.0k dist:34.1k-past-compact",
                "fak-turn trace=guard ok compact=none-past-budget finish=tool_use "
                "budget=spent:134.1k ctx:134.1k/96.0k dist:38.1k-past-compact",
            ]
            log.write_text("\n".join(lines), encoding="utf-8")
            log.with_suffix(".pid").write_text("1234\n", encoding="utf-8")

            with mock.patch.object(mod.dispatch_preflight,
                                   "resolve_sidecar_pid_is_live",
                                   return_value=True):
                hold = mod.active_compact_runaway_hold(root, runs)

        self.assertTrue(hold["active"])
        self.assertEqual(hold["candidates"][0]["lane"], "quality")
        self.assertEqual(hold["lanes"], ["quality"])
        self.assertFalse(hold["lane_unknown"])
        self.assertIn("lane 'quality'", hold["reason"])


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

    def test_banner_noop_log_does_not_hold_its_lane(self) -> None:
        """A worker whose log is a terminal banner no-op (#1275) holds no lane even
        when a recycled pid passes the weak liveness gate (#1398). An exited opencode
        worker runs as a ``node`` image, so a recycled ``node`` pid landing in the
        sidecar's spawn window passes the create-time fallback and would otherwise pin
        ``docs`` at LANE_BUSY forever behind a dead 122-byte no-op."""
        import os
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = runs / "resolve-1398-20260629-101010.log"
            # header + the documented opencode/glm banner-only no-op, well under the
            # 512-byte stub floor (the real #1398 holders were 122 bytes).
            log.write_text(
                "# fak-spawn 20260629-101010 issue=1398 lane=docs backend=opencode "
                "argv0=node\n> build · glm-4.5-air\n",
                encoding="utf-8")
            pid_file = log.with_suffix(".pid")
            pid_file.write_text("777", encoding="utf-8")
            os.utime(pid_file, (now, now))
            def probe(pid):  # a recycled `node` pid in the spawn window → weak-live
                return {"alive": True, "create_time": now - 1,
                        "name": "node.exe", "cmdline": ""}
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

    def test_timed_out_repair_worker_is_reaped_too(self) -> None:
        # A runaway contract-repair worker is subject to the SAME wall-clock cap
        # as a resolution worker — the reaper sweeps both log prefixes.
        import os
        import tempfile
        mod = load()
        now = 1_000_000.0
        killed: list[int] = []
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = runs / "repair-1207-20260702-180000.log"
            log.write_text("", encoding="utf-8")
            pid_file = log.with_suffix(".pid")
            pid_file.write_text("105", encoding="utf-8")
            os.utime(pid_file, (now - 4000, now - 4000))
            def probe(pid):
                return {"alive": True, "create_time": now - 4001,
                        "name": "claude.exe", "cmdline": ""}
            out = mod.reap_timed_out_workers(
                runs, timeout_s=1800, live=True, now_ts=now, probe=probe,
                killer=lambda pid: (killed.append(pid) or {"ok": True, "returncode": 0}))
        self.assertEqual([r["pid"] for r in out["reaped"]], [105])
        self.assertEqual(killed, [105])


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
            log = runs / "resolve-825-20260625-213720.log"
            log.write_text("worker transcript\n", encoding="utf-8")
            def probe(pid):
                return {"alive": False}
            out = mod.prune_dead_sidecars(runs, live=True, now_ts=now, probe=probe)
            self.assertEqual(out["pruned"], ["resolve-825-20260625-213720.pid"])
            self.assertTrue(log.exists())
            self.assertEqual(log.read_text(encoding="utf-8"), "worker transcript\n")
            self.assertEqual(
                sorted(p.name for p in runs.glob("resolve-825-*")),
                ["resolve-825-20260625-213720.log"],
            )

    def test_live_prune_keeps_witness_sidecar_for_cooldown(self) -> None:
        import os
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            pid_file = self._mk(
                runs, 826, "20260625-213721", pid=58752, mtime=now - 4000,
                siblings=(".log", mod.WITNESS_SIDECAR_SUFFIX))
            witness = pid_file.with_suffix(mod.WITNESS_SIDECAR_SUFFIX)
            witness.write_text('{"claim": "CLAIM_NO_COMMIT", "issue": 826}',
                               encoding="utf-8")
            os.utime(witness, (now - 10 * 60, now - 10 * 60))
            def probe(pid):
                return {"alive": False}
            out = mod.prune_dead_sidecars(runs, live=True, now_ts=now, probe=probe)
            cooled = mod.recently_attempted_issues(
                runs, cooldown_min=120, now_ts=now)
            self.assertEqual(out["pruned"], ["resolve-826-20260625-213721.pid"])
            self.assertFalse(pid_file.exists())
            self.assertTrue(witness.exists())
            self.assertEqual(cooled, {826})

    def test_dead_repair_sidecar_and_issues_sibling_are_pruned(self) -> None:
        # Repair corpses (pid + the .issues batch sidecar) are swept like resolve
        # corpses, so stale repair sidecars never pin the seat count or the
        # repair-admission scan's directory.
        import os
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            pid_file = runs / "repair-1207-20260702-180000.pid"
            pid_file.write_text("58753", encoding="utf-8")
            stem = pid_file.with_suffix("")
            for suf in (".log", ".backend", mod.REPAIR_ISSUES_SIDECAR_SUFFIX):
                stem.with_suffix(suf).write_text("", encoding="utf-8")
            os.utime(pid_file, (now - 4000, now - 4000))
            def probe(pid):
                return {"alive": False}
            out = mod.prune_dead_sidecars(runs, live=True, now_ts=now, probe=probe)
            self.assertEqual(out["pruned"], ["repair-1207-20260702-180000.pid"])
            self.assertEqual(sorted(p.name for p in runs.glob("repair-1207-*")), [])

    def test_repair_cooldown_outlives_the_sidecar_sweep(self) -> None:
        """The repair cooldown must survive the corpse it was read off.

        Regression: `recently_repaired_issues` read the `repair-*.log` mtime and
        its `.issues` batch sidecar -- both of which `prune_dead_sidecars` unlinks
        the moment the groomer exits. So the 360-min anti-churn window silently
        collapsed to the groomer's OWN LIFETIME, which the live-repair scan already
        covers, and the same un-repairable heads were re-groomed every ~40 min
        (measured on this host: 62/186 grooming events inside the window, median
        gap 43 min). The durable ledger must still cool the whole batch after the
        sweep has taken every file the old reader depended on.
        """
        import os
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            mod.record_repair_attempt(runs, [3238, 5255, 4318],
                                      live=True, now_ts=now - 40 * 60)
            pid_file = runs / "repair-3238-20260702-180000.pid"
            pid_file.write_text("58753", encoding="utf-8")
            stem = pid_file.with_suffix("")
            stem.with_suffix(".log").write_text("", encoding="utf-8")
            stem.with_suffix(mod.REPAIR_ISSUES_SIDECAR_SUFFIX).write_text(
                "3238,5255,4318", encoding="utf-8")
            os.utime(pid_file, (now - 40 * 60, now - 40 * 60))

            def probe(pid):
                return {"alive": False}

            mod.prune_dead_sidecars(runs, live=True, now_ts=now, probe=probe)
            # Every file the OLD reader used is gone -- that is the bug's premise.
            self.assertEqual(sorted(runs.glob("repair-3238-*")), [])
            cooled = mod.recently_repaired_issues(
                runs, cooldown_min=360, now_ts=now)

        self.assertEqual(cooled, {3238, 5255, 4318})

    def test_repair_cooldown_expires_with_its_window(self) -> None:
        """...and clears itself: past the window the batch is groomable again with
        no reset verb, so an issue is never pinned out permanently."""
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            mod.record_repair_attempt(runs, [3238, 5255],
                                      live=True, now_ts=now - 361 * 60)
            expired = mod.recently_repaired_issues(
                runs, cooldown_min=360, now_ts=now)
            # A dry run stays side-effect-free, so it never cools anything.
            mod.record_repair_attempt(runs, [999], live=False, now_ts=now)
            dry = mod.recently_repaired_issues(runs, cooldown_min=360, now_ts=now)
            disabled = mod.recently_repaired_issues(
                runs, cooldown_min=0, now_ts=now)

        self.assertEqual(expired, set())
        self.assertEqual(dry, set())
        self.assertEqual(disabled, set())

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
        mod.lane_tree = lambda root, lane: [f"internal/{lane}/**"]
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
        # #4322: a refused tick's evidence carries the structured lane fields
        # (not scraped from summary prose) so per-lane collision rate is exact.
        self.assertIn(("lane", "docs"), rows[1]["evidence"])
        self.assertIn(("lane_kind", "cluster"), rows[1]["evidence"])
        self.assertIn(("mode", "shared"), rows[1]["evidence"])
        self.assertIn(("tree", "internal/docs/**"), rows[1]["evidence"])

    def test_record_loop_tick_admitted_tick_has_no_collision_evidence(self) -> None:
        # The structured lane/tree fields are additive on REFUSED ticks only —
        # an admitted (would_spawn/spawned) tick's evidence is unchanged.
        mod = load()
        rows: list[dict[str, object]] = []
        payload = {
            "schema": mod.SCHEMA,
            "workspace": str(ROOT),
            "live": False,
            "backend": "claude",
            "max_workers": 2,
            "preflight": {"verdict": "SPAWN_OK", "cap": 2, "live": 0},
            "account": {"tag": "worker-a"},
            "lane": "gateway",
            "lane_issue_count": 3,
            "target_issue": 717,
            "ok": True,
            "action": "would_spawn",
            "verdict": "WOULD_SPAWN",
            "reason": "safe to spawn one worker",
        }
        mod.record_loop_tick(
            ROOT, payload, ledger=Path("loops.jsonl"),
            append=lambda root, ledger, ev: (rows.append(dict(ev)) or {"ok": True, "kind": ev["kind"]}),
            mint=lambda root, process: "RID-DISPATCH3")

        for r in rows:
            self.assertFalse(any(k == "lane_kind" for k, _ in r["evidence"]))

    def test_record_loop_tick_lane_lease_held_prefers_lease_tree(self) -> None:
        # The fenced-CAS lease refusal carries its OWN requested tree (the
        # dispatch_tick.go convention: lane/lane_kind/mode/tree stamped on the
        # lease map) -- record_loop_tick reads it straight off payload["lease"]
        # rather than re-deriving it from the lane taxonomy.
        mod = load()
        mod.lane_tree = lambda root, lane: [f"SHOULD-NOT-BE-USED/{lane}/**"]
        rows: list[dict[str, object]] = []
        payload = {
            "schema": mod.SCHEMA,
            "workspace": str(ROOT),
            "live": True,
            "backend": "claude",
            "max_workers": 2,
            "preflight": {"verdict": "SPAWN_OK", "cap": 2, "live": 0},
            "account": {"tag": "worker-a"},
            "lane": "cmd",
            "lane_issue_count": 3,
            "target_issue": 900,
            "ok": False,
            "action": "lane_leased",
            "verdict": "LANE_LEASE_HELD",
            "reason": "lane \"cmd\" lease is held by a live peer",
            "lease": {"refused": True, "acquired": False,
                      "tree": ["cmd/fak/**", "internal/dispatchtick/**"]},
        }
        mod.record_loop_tick(
            ROOT, payload, ledger=Path("loops.jsonl"),
            append=lambda root, ledger, ev: (rows.append(dict(ev)) or {"ok": True, "kind": ev["kind"]}),
            mint=lambda root, process: "RID-DISPATCH4")

        self.assertEqual(rows[1]["status"], "refused")
        self.assertIn(("lane", "cmd"), rows[1]["evidence"])
        self.assertIn(("tree", "cmd/fak/**,internal/dispatchtick/**"), rows[1]["evidence"])

    def test_record_loop_tick_dirty_path_collision_carries_paths(self) -> None:
        mod = load()
        mod.lane_tree = lambda root, lane: [f"internal/{lane}/**"]
        rows: list[dict[str, object]] = []
        payload = {
            "schema": mod.SCHEMA,
            "workspace": str(ROOT),
            "live": True,
            "backend": "claude",
            "max_workers": 2,
            "preflight": {"verdict": "SPAWN_OK", "cap": 2, "live": 0},
            "account": {"tag": "worker-a"},
            "lane": "gateway",
            "lane_issue_count": 3,
            "target_issue": 901,
            "ok": False,
            "action": "dirty_path_collision",
            "verdict": "DIRTY_PATH_COLLISION",
            "reason": "dirty path collision",
            "dirty_path_collision": {"collides": True,
                                      "dirty_paths": ["internal/gateway/foo.go"]},
        }
        mod.record_loop_tick(
            ROOT, payload, ledger=Path("loops.jsonl"),
            append=lambda root, ledger, ev: (rows.append(dict(ev)) or {"ok": True, "kind": ev["kind"]}),
            mint=lambda root, process: "RID-DISPATCH5")

        self.assertIn(("paths", "internal/gateway/foo.go"), rows[1]["evidence"])
        self.assertIn(("tree", "internal/gateway/**"), rows[1]["evidence"])


class FleetTrendTickTest(unittest.TestCase):
    """#4594: each live dispatcher tick feeds the fleet-status trend ledger."""

    def test_appends_partial_live_row_from_preflight(self) -> None:
        import json
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            payload = {"backend": "codex", "preflight": {"verdict": "SPAWN_OK", "cap": 4, "live": 3}}
            out = mod.append_fleet_trend_row(root, payload, now="2026-07-14T12:00:00Z")
            self.assertTrue(out["ok"])
            ledger = root / ".fak" / "nightrun" / "fleet-status-history.jsonl"
            self.assertTrue(ledger.exists())
            rows = [json.loads(ln) for ln in ledger.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ts"], "2026-07-14T12:00:00Z")
            self.assertEqual(rows[0]["scope"], "backend")
            self.assertEqual(rows[0]["backend"], "codex")
            self.assertEqual(rows[0]["backend_live"], 3.0)
            # Backend-local probes never masquerade as an aggregate fleet census.
            for absent in ("live", "usable", "sessions", "escalate"):
                self.assertNotIn(absent, rows[0])
            mod.append_fleet_trend_row(root, payload, now="2026-07-14T12:05:00Z")
            rows = [json.loads(ln) for ln in ledger.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)

    def test_missing_preflight_records_scoped_zero(self) -> None:
        import json
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = mod.append_fleet_trend_row(root, {}, now="2026-07-14T12:00:00Z")
            self.assertTrue(out["ok"])
            row = json.loads(Path(out["ledger"]).read_text(encoding="utf-8").strip())
            self.assertEqual(row["scope"], "backend")
            self.assertEqual(row["backend"], "unknown")
            self.assertEqual(row["backend_live"], 0.0)
            self.assertNotIn("live", row)

    def test_append_failure_never_raises(self) -> None:
        mod = load()

        def boom(path, metrics, now, **kw):
            raise OSError("disk full")

        out = mod.append_fleet_trend_row(Path("z:/nope"), {}, append=boom)
        self.assertFalse(out["ok"])
        self.assertIn("disk full", out["error"])


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


class BuildWorkerCommandTest(unittest.TestCase):
    def test_claude_command_shape(self) -> None:
        mod = load()
        self.assertEqual(
            mod.build_worker_command("claude", "PROMPT", None),
            ["claude", "-p", "--permission-mode", "bypassPermissions"])

    def test_claude_command_pins_model_and_effort(self) -> None:
        mod = load()
        self.assertEqual(
            mod.build_worker_command("claude", "PROMPT", "claude-opus-4-8", "xhigh"),
            ["claude", "-p", "--permission-mode", "bypassPermissions",
             "--model", "claude-opus-4-8", "--effort", "xhigh"])

    def test_claude_command_effort_only_when_model_absent(self) -> None:
        # A model-less claude worker still carries its reasoning effort flag.
        mod = load()
        self.assertEqual(
            mod.build_worker_command("claude", "PROMPT", None, "xhigh"),
            ["claude", "-p", "--permission-mode", "bypassPermissions",
             "--effort", "xhigh"])

    def test_worker_model_effort_pins_claude_to_opus_xhigh(self) -> None:
        # A cloud claude account is pinned to Opus at xhigh regardless of the
        # (live-rotated) per-account settings.json model.
        mod = load()
        self.assertEqual(
            mod.worker_model_effort("claude", {"model": "claude-fable-5"}),
            (mod.CLAUDE_WORKER_MODEL, mod.CLAUDE_WORKER_EFFORT))
        self.assertEqual(mod.CLAUDE_WORKER_MODEL, "claude-opus-5")
        self.assertEqual(mod.CLAUDE_WORKER_EFFORT, "xhigh")

    def test_worker_model_effort_keeps_local_claude_and_opencode(self) -> None:
        mod = load()
        # A 'local' claude account is not forced onto cloud Opus, and takes no effort.
        self.assertEqual(mod.worker_model_effort("claude", {"model": "local"}),
                         ("local", None))
        # opencode keeps its own model and takes no effort knob.
        self.assertEqual(mod.worker_model_effort("opencode", {"model": "glm-5.2"}),
                         ("glm-5.2", None))
        # codex is unpinned.
        self.assertEqual(mod.worker_model_effort("codex", {"model": "x"}), (None, None))

    def test_opencode_command_pins_model_and_skips_permissions(self) -> None:
        mod = load()
        self.assertEqual(
            mod.build_worker_command("opencode", "PROMPT", "zai-coding-plan/glm-5.2"),
            ["opencode", "run", "--pure", "--print-logs",
             "--dangerously-skip-permissions",
             "-m", "zai-coding-plan/glm-5.2", mod.OPENCODE_PROMPT_NOTICE])

    def test_opencode_command_without_model(self) -> None:
        mod = load()
        self.assertEqual(
            mod.build_worker_command("opencode", "PROMPT", None),
            ["opencode", "run", "--pure", "--print-logs",
             "--dangerously-skip-permissions",
             mod.OPENCODE_PROMPT_NOTICE])

    def test_opencode_command_keeps_full_prompt_out_of_argv(self) -> None:
        mod = load()
        full_prompt = "your goal: resolve GitHub issue #2588 with lots of detail"
        got = mod.build_worker_command("opencode", full_prompt, "glm")
        self.assertNotIn(full_prompt, got)
        self.assertIn("resolve github issue #", mod.OPENCODE_PROMPT_NOTICE.lower())
        self.assertLessEqual(len(mod.OPENCODE_PROMPT_NOTICE), 96)

    def test_opencode_prompt_file_is_attached_before_notice(self) -> None:
        mod = load()
        got = mod.attach_opencode_prompt_file(
            ["opencode", "run", "--pure", "--print-logs", "-m", "glm",
             mod.OPENCODE_PROMPT_NOTICE],
            r"C:\work\fak\.dispatch-runs\resolve-2588.prompt.txt")
        self.assertEqual(got, [
            "opencode", "run", "--pure", "--print-logs", "-m", "glm",
            "--file", r"C:\work\fak\.dispatch-runs\resolve-2588.prompt.txt",
            "--", mod.OPENCODE_PROMPT_NOTICE])

    def test_opencode_prompt_file_survives_guard_wrapper(self) -> None:
        mod = load()
        got = mod.attach_opencode_prompt_file(
            ["fak", "guard", "--provider", "openai", "--", "opencode", "run",
             mod.OPENCODE_PROMPT_NOTICE],
            "resolve-2588.prompt.txt")
        self.assertEqual(got[-4:], ["--file", "resolve-2588.prompt.txt",
                                    "--", mod.OPENCODE_PROMPT_NOTICE])

    def test_unwrap_opencode_npm_shim_targets_real_executable(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            npm = Path(d)
            real = npm / "node_modules" / "opencode-ai" / "bin" / "opencode.exe"
            real.parent.mkdir(parents=True)
            real.write_text("fake exe", encoding="utf-8")
            self.assertEqual(
                mod.unwrap_opencode_npm_shim(str(npm / "opencode.cmd")), str(real))
            self.assertEqual(mod.unwrap_opencode_npm_shim(str(npm / "claude.cmd")), "")

    def test_codex_command_shape(self) -> None:
        mod = load()
        self.assertEqual(
            mod.build_worker_command("codex", "PROMPT", None),
            ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox",
             "--skip-git-repo-check"])

    def test_codex_command_pins_model(self) -> None:
        mod = load()
        self.assertEqual(
            mod.build_worker_command("codex", "PROMPT", "gpt-5.5"),
            ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox",
             "--skip-git-repo-check", "-m", "gpt-5.5"])

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

    def test_windows_fak_bin_env_keeps_backslashes(self) -> None:
        mod = load()
        with mock.patch.object(mod.os, "name", "nt"):
            self.assertEqual(
                mod.split_command_env(r"C:\work\fak\fak.exe"),
                [r"C:\work\fak\fak.exe"])


class FakCommandPrefixTest(unittest.TestCase):
    """#5856: nothing refreshes the repo-root fak.exe, while FakSelfUpdate rebuilds the
    PATH copy every 20 minutes from a pristine detached origin/main worktree. This prefix
    used to take the repo-root artifact unconditionally and never consult PATH at all --
    so a single dispatcher tick ran two different fak builds, the guarded one of them
    witnessed 34 commits behind HEAD and stamped +dirty."""

    def _layout(self, td: Path):
        exe = "fak.exe" if os.name == "nt" else "fak"
        root, pathdir = td / "root", td / "bin"
        root.mkdir(parents=True)
        pathdir.mkdir(parents=True)
        rootbin, onpath = root / exe, pathdir / exe
        for p in (rootbin, onpath):
            p.write_text("stub", encoding="utf-8")
            p.chmod(0o755)
        return root, rootbin, onpath, pathdir

    def test_explicit_fak_bin_still_wins(self) -> None:
        mod = load()
        with mock.patch.dict(os.environ, {"FAK_BIN": "C:/pinned/fak.exe"}):
            self.assertEqual(mod._fak_command_prefix(Path("/nope")), ["C:/pinned/fak.exe"])

    def test_freshest_build_wins_in_both_directions(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root, rootbin, onpath, pathdir = self._layout(Path(td))
            with mock.patch.dict(os.environ, {"PATH": str(pathdir)}, clear=True):
                # The refreshed PATH copy is newer -> it wins. The fleet case, and the
                # assertion the old repo-root-first prefix fails.
                os.utime(rootbin, (1_000_000, 1_000_000))
                os.utime(onpath, (2_000_000, 2_000_000))
                self.assertEqual(mod._fak_command_prefix(root), [str(onpath)])
                # A just-rebuilt repo-root binary is newer -> it wins. The dev case holds.
                os.utime(rootbin, (3_000_000, 3_000_000))
                self.assertEqual(mod._fak_command_prefix(root), [str(rootbin)])

    def test_no_binary_anywhere_falls_back_to_go_run(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            empty = Path(td) / "empty"
            empty.mkdir()
            with mock.patch.dict(os.environ, {"PATH": str(empty)}, clear=True):
                self.assertEqual(
                    mod._fak_command_prefix(empty), ["go", "run", "./cmd/fak"])


class WinCreationFlagsTest(unittest.TestCase):
    def test_worker_spawn_allocates_no_console_for_native_and_batch_launchers(self) -> None:
        mod = load()
        want = mod._DETACHED_PROCESS if os.name == "nt" else 0
        for exe in (r"C:\npm\opencode.CMD", "opencode.cmd", "wrap.bat",
                    r"C:\bin\claude.exe", "/usr/bin/claude"):
            with self.subTest(exe=exe):
                self.assertEqual(mod.win_creationflags(exe), want)
                self.assertEqual(mod.win_creationflags(exe) & mod._CREATE_NO_WINDOW, 0)


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

    def test_numbers_are_oldest_first_despite_router_newest_first_input(self) -> None:
        # gh issue list (and the router's per-lane "issues") is newest-first; this
        # fold must explicitly reverse it so pick_target_issue's "first not in skip"
        # lands on the OLDEST open issue, per the fleet's stated dispatch priority.
        mod = load()
        mod.issue_dispatch.run_json = lambda cmd, root, timeout: self.ROUTER
        picked = mod.lane_issue_numbers(ROOT, "docs")
        self.assertEqual(picked["numbers"], [7, 8, 9])   # ascending, not [9, 8, 7]


class LaneIssueNumbersSelfModifyHoldTest(unittest.TestCase):
    """Proactive pre-route hold on the ACTUAL production picker: this module
    (not issue_dispatch.py) is what the live Scheduled Tasks invoke. Mirrors
    issue_dispatch.pick_lane's fix -- a lane whose tree is fak's trust-critical referee source is excluded from
    the busiest-pick under guard; ordinary cmd/** and internal/** lanes remain
    dispatchable under the narrowed self-modify policy."""

    ROUTER = {"lanes": {
        "docs": {"issues": [{"number": 2}, {"number": 1}], "tree": ["docs/**"]},
        "policy": {"issues": [{"number": 4}, {"number": 3}, {"number": 2}, {"number": 1}],
                   "tree": ["internal/policy/**"]},
        "tools": {"issues": [{"number": 9}], "tree": ["tools/**", "scripts/**"]},
    }}

    def _router(self, mod) -> None:
        mod.issue_dispatch.run_json = lambda cmd, root, timeout: self.ROUTER

    def test_guarded_skips_self_source_lane_for_richest_safe(self) -> None:
        mod = load()
        self._router(mod)
        picked = mod.lane_issue_numbers(ROOT, None, guarded=True)
        self.assertEqual(picked["lane"], "docs")   # policy (4, trust-critical) excluded
        self.assertEqual(picked["self_modify_held"], ["policy"])

    def test_unguarded_does_not_hold_self_source_lane(self) -> None:
        mod = load()
        self._router(mod)
        picked = mod.lane_issue_numbers(ROOT, None, guarded=False)
        self.assertEqual(picked["lane"], "policy")
        self.assertEqual(picked["self_modify_held"], [])

    def test_all_lanes_self_source_yields_no_lane(self) -> None:
        mod = load()
        mod.issue_dispatch.run_json = lambda cmd, root, timeout: {"lanes": {
            "policy": {"issues": [{"number": 2}, {"number": 1}], "tree": ["internal/policy/**"]},
            "adjudicator": {"issues": [{"number": 9}], "tree": ["internal/adjudicator/**"]},
        }}
        picked = mod.lane_issue_numbers(ROOT, None, guarded=True)
        self.assertIsNone(picked["lane"])
        self.assertEqual(picked["numbers"], [])
        self.assertEqual(picked["self_modify_held"], ["adjudicator", "policy"])

    def test_explicit_lane_honored_despite_self_source(self) -> None:
        mod = load()
        self._router(mod)
        picked = mod.lane_issue_numbers(ROOT, "policy", guarded=True)
        self.assertEqual(picked["lane"], "policy")
        self.assertEqual(picked["self_modify_held"], [])   # explicit path skips the fold

    def test_exclude_and_self_source_hold_compose(self) -> None:
        # docs excluded (busy) + gateway held (self-source) -> only tools left.
        mod = load()
        self._router(mod)
        picked = mod.lane_issue_numbers(ROOT, None, exclude={"docs"}, guarded=True)
        self.assertEqual(picked["lane"], "tools")
        self.assertEqual(picked["safe_lanes_busy"], ["docs"])
        self.assertEqual(picked["self_modify_held"], ["policy"])

    def test_safe_lanes_busy_surfaces_when_only_self_source_remains(self) -> None:
        mod = load()
        self._router(mod)
        picked = mod.lane_issue_numbers(ROOT, None, exclude={"docs", "tools"},
                                        guarded=True)
        self.assertIsNone(picked["lane"])
        self.assertEqual(picked["safe_lanes_busy"], ["docs", "tools"])
        self.assertEqual(picked["self_modify_held"], ["policy"])

    def test_default_guarded_reads_dispatch_worker_guard_enabled(self) -> None:
        mod = load()
        self._router(mod)
        mod.dispatch_worker.guard_enabled = lambda *a, **k: True
        picked = mod.lane_issue_numbers(ROOT, None)
        self.assertEqual(picked["self_modify_held"], ["policy"])


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
    GLM_WALL = "Weekly/Monthly Limit Exhausted. Your limit will reset at 2026-07-04 00:56:38"
    GUARDED_429 = (
        "fak-turn trace=guard FAILED reason=rate_limited wire=anthropic_messages "
        "announced_wait=1h12m16s\n"
        "API Error: Request rejected (429) · upstream rate-limited the request (HTTP 429)\n"
    )
    # #2610: the WEEKLY-cap variant of the guard 429 — a `kind=weekly_limit` token
    # plus the relative `announced_wait` window (6h50m here, deliberately past both the
    # 60-min blind fallback and the 90-min session clamp so honoring it is unambiguous).
    GUARDED_WEEKLY_429 = (
        "fak-turn trace=guard FAILED reason=rate_limited wire=anthropic_messages "
        "kind=weekly_limit announced_wait=6h50m0s\n"
        "API Error: Request rejected (429) · upstream rate-limited the request (HTTP 429)\n"
    )
    # #2610 LITERAL witness: the EXACT banner the July-4 overnight tick logged — a
    # `≈` (approximately, U+2248) separator and a MINUTE-ONLY window (`1h7m`, no
    # trailing seconds), NOT the synthesized `=`/`6h50m0s` form above. Both the
    # `_ANNOUNCED_WAIT_RE` separator class `[=:≈~]?` and its optional-seconds
    # branch have to hold or this real log silently fails to parse and the seat is
    # reoffered straight back into the cap — the precise #2610 regression.
    GUARDED_WEEKLY_429_LIVE = (
        "fak-turn trace=guard FAILED reason=rate_limited wire=anthropic_messages "
        "kind=weekly_limit announced_wait≈1h7m\n"
        "API Error: Request rejected (429)\n"
    )

    def _write_worker(self, runs: Path, name: str, body: str, *, mtime: float,
                      backend: str = "claude", account_tag: str | None = None) -> None:
        import json
        import os
        log = runs / name
        log.write_text(body, encoding="utf-8")
        log.with_suffix(".backend").write_text(backend, encoding="utf-8")
        if account_tag:
            log.with_suffix(".account").write_text(json.dumps({"tag": account_tag}), encoding="utf-8")
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

    def test_scan_detects_guarded_provider_429_as_session_hold(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write_worker(runs, "resolve-1475-guard.log", self.GUARDED_429,
                               mtime=now - 60)
            hit = mod._scan_recent_cap_banner(runs, product="claude",
                                              lookback_min=45, now_ts=now)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["kind"], "session")
        # #2610: the guard names its wall as a relative `announced_wait=<dur>`; the
        # parser now captures that window instead of discarding it (was "").
        self.assertEqual(hit["reset_text"], "1h12m16s")
        self.assertEqual(hit["evidence_log"], "resolve-1475-guard.log")

    def test_weekly_limit_guard_form_cools_to_announced_window(self) -> None:
        """#2610: a guard cap crash names the wall as ``kind=weekly_limit`` + a
        relative ``announced_wait`` (not Claude's absolute 'resets <when>' banner).
        The hold must cool to that ANNOUNCED window — here 6h50m, past both the
        60-min blind fallback and the 90-min session clamp — proving the window is
        honored AND classified weekly (not session). This is the weekly-limit case
        the acceptance asks for, distinct from the stale-credential/auth path
        (#2059/#2075), which routes through the AUTH classifier, not the cap hold."""
        import datetime as dt
        import tempfile
        mod = load()
        now = 1_000_000.0
        now_utc = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=now)
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write_worker(runs, "resolve-2610-weekly.log", self.GUARDED_WEEKLY_429,
                               mtime=now - 60, account_tag="july2-netra")
            hit = mod._scan_recent_cap_banner(runs, product="claude",
                                              lookback_min=45, now_ts=now)
            self.assertIsNotNone(hit)
            self.assertEqual(hit["kind"], "weekly")
            self.assertEqual(hit["reset_text"], "6h50m0s")
            # The account-cap sidecar (the shared availability contract both
            # dispatch_preflight/issue_dispatch honor via runtime_status) now cools
            # this seat to the announced window and names the reason.
            out = mod.check_weekly_cap(runs, product="claude",
                                       account_tag="july2-netra", now_ts=now)
            self.assertTrue(out["capped"])
            self.assertEqual(out["kind"], "weekly")
            self.assertEqual(out["reset_text"], "6h50m0s")
            until = mod._iso_to_utc(out["until"])
            self.assertEqual(until, now_utc + dt.timedelta(hours=6, minutes=50))
            # Not the 60-min blind fallback, and not clamped to the 90-min session cap.
            self.assertGreater(until, now_utc + dt.timedelta(minutes=90))

    def test_weekly_limit_literal_witness_approx_form_cools_seat(self) -> None:
        """#2610 acceptance #4, LITERAL form: the exact guard banner the July-4
        overnight tick logged — ``kind=weekly_limit announced_wait≈1h7m``, a ``≈``
        separator and a minute-only window — must be detected as a WEEKLY cap and
        cool the seat to that announced ~1h7m window. Every other weekly fixture
        uses a synthesized ``=``/``6h50m0s`` form, so this is the ONLY test that
        pins the ``≈``-separator + optional-seconds branch of ``_ANNOUNCED_WAIT_RE``;
        a regex 'cleanup' to ``=``-only would leave the whole suite green while the
        real overnight log silently reoffered the capped seat, the exact #2610 bug."""
        import datetime as dt
        import tempfile
        mod = load()
        now = 1_000_000.0
        now_utc = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=now)
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write_worker(runs, "resolve-2610-live.log",
                               self.GUARDED_WEEKLY_429_LIVE, mtime=now - 60,
                               account_tag="july2-netra")
            hit = mod._scan_recent_cap_banner(runs, product="claude",
                                              lookback_min=45, now_ts=now)
            self.assertIsNotNone(hit)
            self.assertEqual(hit["kind"], "weekly")
            self.assertEqual(hit["reset_text"], "1h7m")   # minute-only, ≈-separated
            out = mod.check_weekly_cap(runs, product="claude",
                                       account_tag="july2-netra", now_ts=now)
            self.assertTrue(out["capped"])
            self.assertEqual(out["kind"], "weekly")
            until = mod._iso_to_utc(out["until"])
            self.assertEqual(until, now_utc + dt.timedelta(hours=1, minutes=7))
            # Not the 60-min blind fallback — the announced window is honored.
            self.assertGreater(until, now_utc + dt.timedelta(minutes=60))

    def test_weekly_cap_sidecar_writer_reader_contract(self) -> None:
        """#2610 writer<->reader contract. ``check_weekly_cap`` WRITES the
        account-cap sidecar; ``fleet_accounts.active_account_cap_throttles`` /
        ``annotate_accounts`` READ it to drop the seat. Each side is unit-tested
        in isolation — the writer above, the reader from a HAND-WRITTEN dict in
        fleet_accounts_test — so a field rename in ``_write_cap_hold``
        (until/kind/account/product) would leave BOTH suites green while the live
        cooldown silently reoffers the weekly-capped seat, the exact regression
        #2610 fixes. This chains the REAL writer into the REAL reader and asserts
        the seat goes unavailable as a weekly usage block — the availability
        contract dispatch_preflight/issue_dispatch honor through the switcher.

        Uses real wall-clock time on purpose: ``annotate_accounts`` reads the
        cooldown with the live clock, so the 6h50m announced window must land in
        the real future for the hold to be active during the run."""
        import datetime as dt
        import tempfile
        import time
        mod = load()
        sys.path.insert(0, str(SCRIPT.parent))
        import fleet_accounts
        now = time.time()
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / "home"
            (home / ".claude-july2-netra-acct" / "projects").mkdir(parents=True)
            cfg = Path(d) / "cfg"   # empty XDG config home -> no opencode accounts
            runs = Path(d) / "runs"
            runs.mkdir()
            # WRITER: a guard-form weekly-limit 429 log -> the real account-cap sidecar.
            self._write_worker(runs, "resolve-2610-weekly.log", self.GUARDED_WEEKLY_429,
                               mtime=now - 60, account_tag="july2-netra")
            wrote = mod.check_weekly_cap(runs, product="claude",
                                         account_tag="july2-netra", now_ts=now)
            self.assertTrue(wrote["capped"])
            self.assertEqual(wrote["kind"], "weekly")
            self.assertTrue((runs / "account-cap-claude-july2-netra.json").exists())

            rows = fleet_accounts.discover_accounts(str(home), config_home=str(cfg))
            # READER 1 (pure fold): the sidecar the writer emitted resolves to a
            # WEEKLY throttle keyed by the discovered account basename.
            thr = fleet_accounts.active_account_cap_throttles(
                rows, runs_dir=str(runs), now=dt.datetime.now(dt.timezone.utc))
            self.assertIn(".claude-july2-netra-acct", thr)
            self.assertEqual(thr[".claude-july2-netra-acct"]["cap_kind"], "weekly")
            self.assertTrue(thr[".claude-july2-netra-acct"].get("weekly"))
            # READER 2 (end-to-end availability): the seat preflight/dispatch would
            # be offered is instead dropped as a usage block until the window.
            annotated = fleet_accounts.annotate_accounts(
                rows, registry={}, cap_runs_dir=str(runs))
            seat = {r["tag"]: r for r in annotated}["july2-netra"]
            self.assertFalse(seat["available"])
            self.assertEqual(seat["block_kind"], "usage")

    def test_parse_relative_wait_ignores_absolute_clause(self) -> None:
        """The relative-wait parser fires ONLY on a bare Go-duration; an absolute
        banner reset clause falls through to _parse_reset_to_utc unchanged (#2610)."""
        import datetime as dt
        mod = load()
        now = dt.datetime(2026, 7, 4, 12, 0, 0)
        self.assertEqual(mod._parse_relative_wait("1h7m0s", now),
                         now + dt.timedelta(hours=1, minutes=7))
        self.assertEqual(mod._parse_relative_wait("90m", now),
                         now + dt.timedelta(minutes=90))
        for absolute in ("Jun 25, 1pm", "6:10am", "2026-07-04 00:56:38", "", "soon"):
            self.assertIsNone(mod._parse_relative_wait(absolute, now))

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

    def test_check_scoped_to_matching_account_sidecar(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write_worker(runs, "resolve-1-generic.log", self.SESSION_BANNER,
                               mtime=now - 60)
            self._write_worker(runs, "resolve-2-july1.log", self.SESSION_BANNER,
                               mtime=now - 30, account_tag="july1-netra")
            out = mod.check_weekly_cap(runs, product="claude",
                                       account_tag="gem8NEW-netra", now_ts=now)
            self.assertFalse(out["capped"])
            self.assertFalse((runs / "account-cap-claude.json").exists())

            self._write_worker(runs, "resolve-3-gem8.log", self.SESSION_BANNER,
                               mtime=now - 10, account_tag="gem8NEW-netra")
            out = mod.check_weekly_cap(runs, product="claude",
                                       account_tag="gem8NEW-netra", now_ts=now)
            self.assertTrue(out["capped"])
            self.assertEqual(out["evidence_log"], "resolve-3-gem8.log")

    def test_persisted_hold_with_unmatched_evidence_is_cleared(self) -> None:
        import datetime as dt
        import json
        import tempfile
        mod = load()
        now_utc = dt.datetime(2026, 7, 3, 20, 0, 0)
        now_ts = (now_utc - dt.datetime(1970, 1, 1)).total_seconds()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write_worker(runs, "resolve-generic.log", self.SESSION_BANNER,
                               mtime=now_ts - 60)
            (runs / "account-cap-claude.json").write_text(json.dumps({
                "product": "claude",
                "account": "gem8NEW-netra",
                "kind": "session",
                "reset_text": "2pm",
                "evidence_log": "resolve-generic.log",
                "detected": "2026-07-03T20:00:00Z",
                "until": "2026-07-03T21:00:00Z",
            }), encoding="utf-8")
            out = mod.check_weekly_cap(runs, product="claude",
                                       account_tag="gem8NEW-netra", now_ts=now_ts)
            self.assertFalse(out["capped"])
            self.assertFalse((runs / "account-cap-claude.json").exists())

    def test_scan_detects_glm_weekly_monthly_wall(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write_worker(runs, "resolve-1451-glm.log", self.GLM_WALL,
                               mtime=now - 60, backend="opencode")
            hit = mod._scan_recent_cap_banner(runs, product="opencode",
                                              lookback_min=45, now_ts=now)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["kind"], "weekly")
        self.assertEqual(hit["reset_text"], "2026-07-04 00:56:38")
        self.assertEqual(hit["evidence_log"], "resolve-1451-glm.log")

    def test_parse_reset_date_and_time_to_utc(self) -> None:
        import datetime as dt
        mod = load()
        now = dt.datetime(2026, 6, 23, 8, 0, 0)
        got = mod._parse_reset_to_utc("Jun 25, 1pm", now)
        self.assertEqual(got, dt.datetime(2026, 6, 25, 20, 0, 0))   # 1pm PDT(-7) -> 20:00 UTC

    def test_parse_reset_bare_iso_timestamp(self) -> None:
        import datetime as dt
        mod = load()
        now = dt.datetime(2026, 6, 30, 8, 0, 0)
        got = mod._parse_reset_to_utc("2026-07-04 00:56:38", now)
        self.assertEqual(got, dt.datetime(2026, 7, 4, 0, 56, 38))

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
            self._write_worker(runs, "resolve-517-a.log", self.BANNER, mtime=now - 60,
                               account_tag="smith")
            first = mod.check_weekly_cap(runs, product="claude", account_tag="smith", now_ts=now)
            self.assertTrue(first["capped"])
            self.assertEqual(first["source"], "banner")
            self.assertTrue((runs / "account-cap-claude.json").exists())
            # Later tick, banner now stale (no fresh evidence) but hold persists:
            later = mod.check_weekly_cap(runs, product="claude", account_tag="smith",
                                         now_ts=now + 3 * 3600)
            self.assertTrue(later["capped"])
            self.assertEqual(later["source"], "state")

    def test_check_keeps_account_holds_when_next_account_caps(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write_worker(runs, "resolve-day26.log", self.BANNER, mtime=now - 60,
                               account_tag="day26NEW-netra")
            first = mod.check_weekly_cap(runs, product="claude",
                                         account_tag="day26NEW-netra", now_ts=now)
            self.assertTrue(first["capped"])
            self.assertTrue((runs / "account-cap-claude-day26NEW-netra.json").exists())

            self._write_worker(runs, "resolve-july1.log", self.SESSION_BANNER,
                               mtime=now + 60, account_tag="july1-netra")
            second = mod.check_weekly_cap(runs, product="claude",
                                          account_tag="july1-netra", now_ts=now + 120)
            self.assertTrue(second["capped"])
            self.assertTrue((runs / "account-cap-claude-july1-netra.json").exists())

            later = mod.check_weekly_cap(runs, product="claude",
                                         account_tag="day26NEW-netra",
                                         now_ts=now + 3 * 3600)
            self.assertTrue(later["capped"])
            self.assertEqual(later["source"], "state")

    def test_check_clears_expired_hold(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write_worker(runs, "resolve-517-a.log", self.BANNER, mtime=now - 60,
                               account_tag="smith")
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
            self._write_worker(runs, "resolve-438-x.log", self.SESSION_BANNER,
                               mtime=now_ts - 60, account_tag="gem8-netra")
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
            self._write_worker(runs, "resolve-9-w.log", self.BANNER, mtime=now_ts - 60,
                               account_tag="smith")
            out = mod.check_weekly_cap(runs, product="claude", account_tag="smith",
                                       now_ts=now_ts)
            self.assertTrue(out["capped"])
            self.assertEqual(out["kind"], "weekly")
            until = dt.datetime.fromisoformat(out["until"].replace("Z", ""))
            # Jun 25 1pm PDT == Jun 25 20:00 UTC, ~2 days out — far beyond the session cap.
            self.assertEqual(until, dt.datetime(2026, 6, 25, 20, 0, 0))

    def test_glm_weekly_monthly_limit_keeps_full_hold(self) -> None:
        import tempfile
        import datetime as dt
        mod = load()
        now_utc = dt.datetime(2026, 6, 30, 8, 0, 0)
        now_ts = (now_utc - dt.datetime(1970, 1, 1)).total_seconds()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._write_worker(runs, "resolve-1451-glm.log", self.GLM_WALL,
                               mtime=now_ts - 60, backend="opencode", account_tag="glm")
            out = mod.check_weekly_cap(runs, product="opencode", account_tag="glm",
                                       now_ts=now_ts)
            self.assertTrue(out["capped"])
            self.assertEqual(out["kind"], "weekly")
            self.assertEqual(out["reset_text"], "2026-07-04 00:56:38")
            until = dt.datetime.fromisoformat(out["until"].replace("Z", ""))
            self.assertEqual(until, dt.datetime(2026, 7, 4, 0, 56, 38))

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

    def test_evaluate_reroutes_once_when_selected_account_is_capped(self) -> None:
        mod = load()
        capped = {"verdict": "SPAWN_OK", "reason": "ok", "cap": 2, "live": 0,
                  "account": {"tag": "day26NEW-netra", "tier": 1,
                              "model": "opus", "dir": "/acct/day26"}}
        spare = {"verdict": "SPAWN_OK", "reason": "ok", "cap": 2, "live": 0,
                 "account": {"tag": "gem8NEW-netra", "tier": 1,
                             "model": "opus", "dir": "/acct/gem8"}}
        preflights: list[str] = []

        def preflight(root, **kw):
            preflights.append(str(kw.get("product")))
            return capped if len(preflights) == 1 else spare

        caps: list[str] = []

        def cap_check(runs_dir, **kw):
            tag = str(kw.get("account_tag") or "")
            caps.append(tag)
            if tag == "day26NEW-netra":
                return {"capped": True, "until": "2026-07-04T00:00:00Z",
                        "reset_text": "", "source": "state"}
            return {"capped": False}

        mod.issue_dispatch.refresh_registry = lambda root: {"ok": True}
        mod.issue_dispatch.preflight = preflight
        mod.check_weekly_cap = cap_check
        mod.check_backend_health = lambda *a, **k: {"state": "healthy"}
        mod.read_dead_backends = lambda *a, **k: []
        mod.reap_timed_out_workers = lambda runs_dir, **k: {
            "timeout_s": k.get("timeout_s"), "live": k.get("live"),
            "candidates": [], "reaped": [], "would_reap": []}
        mod.prune_dead_sidecars = lambda runs_dir, **k: {"pruned": []}
        mod.lane_issue_numbers = lambda root, lane, exclude=None: {
            "lane": "gateway", "numbers": [467],
            "by_lane_count": {"gateway": 1},
            "eligible_by_lane": [["gateway", [467]]],
        }
        mod.live_resolution_issues = lambda runs_dir: set()
        mod.live_resolution_lanes = lambda runs_dir: set()
        mod.recently_attempted_issues = lambda runs_dir, *, cooldown_min, **k: set()
        mod.contract_held_issues = lambda runs_dir, **k: set()
        mod.multi_lane_held_issues = lambda runs_dir, **k: set()
        mod.issue_worker_prompt.build = lambda n, lane, *, workspace: {
            "prompt": f"resolve #{n}", "prompt_chars": 900, "title": f"title {n}"}
        mod.issue_contract_review = passing_issue_contract
        mod.live_lane_lease_lanes = lambda root: {"lanes": []}
        mod.spawn_issue_worker = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("dry-run must never spawn"))

        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False, contract_hold_ttl_h=0)

        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertEqual(p["account"]["tag"], "gem8NEW-netra")
        self.assertEqual(preflights, ["claude", "claude"])
        self.assertEqual(caps, ["day26NEW-netra", "gem8NEW-netra"])
        self.assertEqual(p["account_cap_reroute"]["from"]["tag"], "day26NEW-netra")
        self.assertEqual(p["account_cap_reroute"]["account"]["tag"], "gem8NEW-netra")

    def test_evaluate_keeps_weekly_capped_when_reroute_returns_same_account(self) -> None:
        mod = load()
        pre = {"verdict": "SPAWN_OK", "reason": "ok", "cap": 2, "live": 0,
               "account": {"tag": "day26NEW-netra", "tier": 1,
                           "model": "opus", "dir": "/acct/day26"}}
        preflights = 0

        def preflight(root, **kw):
            nonlocal preflights
            preflights += 1
            return pre

        mod.issue_dispatch.refresh_registry = lambda root: {"ok": True}
        mod.issue_dispatch.preflight = preflight
        mod.check_weekly_cap = lambda runs_dir, **k: {
            "capped": True, "until": "2026-07-04T00:00:00Z",
            "reset_text": "", "source": "state"}
        mod.reap_timed_out_workers = lambda runs_dir, **k: {
            "timeout_s": k.get("timeout_s"), "live": k.get("live"),
            "candidates": [], "reaped": [], "would_reap": []}
        mod.prune_dead_sidecars = lambda runs_dir, **k: {"pruned": []}
        mod.check_backend_health = lambda *a, **k: {"state": "healthy"}
        mod.read_dead_backends = lambda *a, **k: []
        mod.lane_issue_numbers = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("same capped account must short-circuit before lane router"))

        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)

        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "WEEKLY_CAPPED")
        self.assertEqual(preflights, 2)
        self.assertEqual(p["account_cap_reroute"]["reason"],
                         "reroute did not find a different account")


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

    def test_spawn_stamps_lease_sidecar(self) -> None:
        import json
        import tempfile
        from unittest import mock
        mod = load()

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                self.pid = 10

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            lease = {
                "acquired": True,
                "id": "resolve-tools",
                "holder": "session-1",
                "generation": 4,
                "session_id": "sess-1",
                "tree": ["tools/**"],
            }
            with mock.patch.object(mod.subprocess, "Popen", _FakePopen):
                res = mod.spawn_issue_worker(["true"], {}, runs, runs,
                                             issue=2, lane="tools", backend="claude",
                                             lease=lease)
            self.assertEqual(res["lease"]["id"], "resolve-tools")
            sidecars = list(runs.glob("*.lease"))
            self.assertEqual(len(sidecars), 1)
            self.assertEqual(json.loads(sidecars[0].read_text(encoding="utf-8")),
                             {"id": "resolve-tools", "holder": "session-1",
                              "generation": 4, "session_id": "sess-1",
                              "tree": ["tools/**"]})

    def test_opencode_spawn_stages_prompt_and_unwraps_npm_shim(self) -> None:
        import tempfile
        from unittest import mock
        mod = load()
        captured: dict[str, object] = {}

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                captured["argv"] = argv
                captured["env"] = kwargs.get("env")
                self.pid = 2588

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            npm = runs / "npm"
            real = npm / "node_modules" / "opencode-ai" / "bin" / "opencode.exe"
            real.parent.mkdir(parents=True)
            real.write_text("fake exe", encoding="utf-8")
            full_prompt = "your goal: resolve GitHub issue #2588 with private details"
            command = mod.build_worker_command("opencode", full_prompt, "glm")

            def _which(name):
                return str(npm / "opencode.cmd") if name == "opencode" else None

            with mock.patch.object(mod.shutil, "which", _which), \
                    mock.patch.object(mod.os, "name", "nt"), \
                    mock.patch.object(mod.subprocess, "Popen", _FakePopen):
                res = mod.spawn_issue_worker(
                    command, {"DISPATCH_LANE": "dispatch"}, runs, runs,
                    issue=2588, lane="dispatch", backend="opencode",
                    prompt_payload=full_prompt)

            argv = captured["argv"]
            self.assertEqual(argv[0], str(real))
            self.assertIn("--file", argv)
            self.assertIn("--", argv)
            self.assertIn(mod.OPENCODE_PROMPT_NOTICE, argv)
            self.assertNotIn(full_prompt, argv)
            prompt_file = Path(res["prompt_file"])
            self.assertEqual(prompt_file.read_text(encoding="utf-8"), full_prompt)

    def test_codex_long_prompt_is_streamed_via_stdin_file(self) -> None:
        mod = load()
        captured = {}
        full_prompt = "resolve #4779 " + ("x" * 40000)

        class _FakePopen:
            pid = 1234
            def __init__(self, argv, **kwargs):
                captured["argv"] = argv
                captured["stdin_text"] = kwargs["stdin"].read()
            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(mod.subprocess, "Popen", _FakePopen):
            runs = Path(d)
            command = mod.build_worker_command("codex", full_prompt, None)
            res = mod.spawn_issue_worker(
                command, {}, runs, runs, issue=4779, lane="tools",
                backend="codex", prompt_payload=full_prompt)
            prompt_copy = Path(res["prompt_file"]).read_text(encoding="utf-8")

        self.assertNotIn(full_prompt, captured["argv"])
        self.assertEqual(captured["stdin_text"], full_prompt)
        self.assertEqual(prompt_copy, full_prompt)

    def test_claude_long_prompt_is_streamed_via_stdin_file(self) -> None:
        mod = load()
        captured = {}
        full_prompt = "resolve #4779 " + ("x" * 40000)

        class _FakePopen:
            pid = 1234
            def __init__(self, argv, **kwargs):
                captured["argv"] = argv
                captured["stdin_text"] = kwargs["stdin"].read()
            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(mod.subprocess, "Popen", _FakePopen):
            runs = Path(d)
            command = mod.build_worker_command("claude", full_prompt, None)
            res = mod.spawn_issue_worker(
                command, {}, runs, runs, issue=4779, lane="tools",
                backend="claude", prompt_payload=full_prompt)
            prompt_copy = Path(res["prompt_file"]).read_text(encoding="utf-8")

        self.assertNotIn(full_prompt, captured["argv"])
        self.assertEqual(captured["stdin_text"], full_prompt)
        self.assertEqual(prompt_copy, full_prompt)

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


class DispatchAccountStampTest(unittest.TestCase):
    """#5870: the long-``Retry-After`` goal park is account-scoped
    (``goalpark.Record.Blocks``), and a record whose ``account`` is blank blocks
    NOBODY. Only the Go spine ever stamped ``DISPATCH_ACCOUNT``; this dispatcher —
    the live producer of the fleet's ``resolve-*.log`` units — stamped nothing, so
    every park its workers wrote was unattributed and the park was inert. These
    assert the stamp AND that its format matches the Go spine's
    ``dispatchAccountID`` exactly, because ``SameAccount`` is a plain string
    compare that fails silently on a mismatch."""

    def test_identity_prefers_the_tag_then_the_dir_basename(self) -> None:
        mod = load()
        self.assertEqual(
            mod.dispatch_account_id({"tag": "aug5-netra",
                                     "dir": r"C:\Users\u\.claude-aug5-netra"}),
            "aug5-netra")
        # No tag: the config dir's BASE NAME, exactly as Go's
        # filepath.Base(filepath.Clean(dir)) resolves it. A trailing separator must
        # not shift the answer to "".
        self.assertEqual(
            mod.dispatch_account_id({"dir": "/home/u/.claude-july16-netra"}),
            ".claude-july16-netra")
        self.assertEqual(
            mod.dispatch_account_id({"dir": "/home/u/.claude-july16-netra/"}),
            ".claude-july16-netra")
        # Genuinely anonymous, and the blank/absent forms of both fields.
        self.assertEqual(mod.dispatch_account_id({}), "")
        self.assertEqual(mod.dispatch_account_id(None), "")
        self.assertEqual(mod.dispatch_account_id({"tag": "  ", "dir": " "}), "")

    def test_stamp_never_mutates_the_caller_env(self) -> None:
        mod = load()
        base = {"DISPATCH_LANE": "tools"}
        out = mod.stamp_dispatch_account(base, {"tag": "seat-a"})
        self.assertEqual(out["DISPATCH_ACCOUNT"], "seat-a")
        self.assertEqual(out["DISPATCH_LANE"], "tools")
        self.assertNotIn("DISPATCH_ACCOUNT", base)

    def test_anonymous_seat_drops_an_inherited_identity(self) -> None:
        # An unattributable spawn must NOT inherit this dispatcher's own ambient
        # DISPATCH_ACCOUNT: that would file someone else's wall under our seat.
        mod = load()
        out = mod.stamp_dispatch_account(
            {"DISPATCH_ACCOUNT": "stale-dispatcher-identity"}, None)
        self.assertNotIn("DISPATCH_ACCOUNT", out)

    def test_spawn_stamps_the_serving_account_into_the_child_env(self) -> None:
        # End-to-end over the real spawn seam, no live launch: the child env the
        # dispatcher hands `fak guard` carries a NON-EMPTY account, which is the
        # exact string guard writes into the park record and reads back through
        # Blocks. Proven by capturing Popen's env.
        import tempfile
        from unittest import mock
        mod = load()
        captured: dict[str, object] = {}

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                captured["env"] = kwargs.get("env")
                kwargs.get("stdout").close()  # the parent's log fh (no real child)
                self.pid = 4805

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            acct = {"tag": "aug5-netra", "tier": 1, "model": "opus",
                    "dir": r"C:\Users\u\.claude-aug5-netra"}
            with mock.patch.object(mod.subprocess, "Popen", _FakePopen):
                res = mod.spawn_issue_worker(
                    ["true"], {"DISPATCH_LANE": "gateway"}, runs, runs,
                    issue=5870, lane="gateway", backend="claude", account=acct)
            env = captured["env"]
            self.assertEqual(env["DISPATCH_ACCOUNT"], "aug5-netra")
            self.assertEqual(env["DISPATCH_LANE"], "gateway")  # base env preserved
            # The env stamp and the on-disk account sidecar must name ONE seat.
            self.assertEqual(res["account"]["tag"], env["DISPATCH_ACCOUNT"])

    def test_spawn_without_an_account_leaves_no_stale_identity(self) -> None:
        import tempfile
        from unittest import mock
        mod = load()
        captured: dict[str, object] = {}

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                captured["env"] = kwargs.get("env")
                kwargs.get("stdout").close()
                self.pid = 11

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            with mock.patch.object(mod.subprocess, "Popen", _FakePopen):
                mod.spawn_issue_worker(
                    ["true"], {"DISPATCH_ACCOUNT": "stale-dispatcher-identity"},
                    runs, runs, issue=2, lane="tools", backend="claude")
            self.assertNotIn("DISPATCH_ACCOUNT", captured["env"])


class BackendHealthTest(unittest.TestCase):
    """The backend-health reallocation gate: a MAJORITY of stub (banner-only/0-byte,
    dead-pid) logs over the lookback window declares a backend dead — the same signal
    the status card shows (#3247); a productive log in a window it is not mostly
    stubbing clears the hold; a still-running stub is never counted."""

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

    def test_majority_stub_window_declares_dead_despite_one_productive_log(self) -> None:
        # #3247: the gate must agree with the status card's majority-stub signal. A
        # single productive log in the 90-min window must NOT resurrect a backend that
        # is otherwise stubbing 4-out-of-5 spawns -- that short-circuit is what let the
        # codex cron keep feeding seats to a majority-stub backend every tick.
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            for i in range(4):  # 4 dead-pid stubs, newest first
                self._mk(runs, 900 + i, f"20260708-05000{i}", backend="codex",
                         size=32, pid=500 + i, mtime=now - (i + 1) * 60)
            # ...and ONE real turn, still inside the 90-min lookback.
            self._mk(runs, 910, "20260708-043000", backend="codex",
                     size=5000, pid=510, mtime=now - 80 * 60)
            out = mod.check_backend_health(runs, product="codex", lane="tools",
                                           now_ts=now, alive=set(), probe=self._dead_probe())
            self.assertEqual(out["state"], "dead")  # was "healthy" (source="productive")
            self.assertEqual(out["source"], "majority-stub")
            self.assertEqual((out["stub"], out["productive"]), (4, 1))
            self.assertTrue((runs / "backend-health-codex.json").exists())

    def test_gate_and_status_card_agree_on_majority_stub(self) -> None:
        # The two health computations are independent (the card reads logs, the gate
        # reads logs + a sidecar). #3247 was exactly their disagreement: the card said
        # "majority-stub [codex=4/4 stub]" while the gate said healthy and kept spawning.
        import tempfile
        mod = load()
        try:
            import dispatch_status  # noqa: F401
        except ImportError:  # pragma: no cover - status card is optional here
            self.skipTest("dispatch_status not importable")
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            for i in range(4):
                self._mk(runs, 920 + i, f"20260708-06000{i}", backend="codex",
                         size=32, pid=600 + i, mtime=now - (i + 1) * 60)
            self._mk(runs, 930, "20260708-050000", backend="codex",
                     size=5000, pid=610, mtime=now - 80 * 60)
            card = dispatch_status.backend_stub_rates(runs, now_ts=now, alive=set())
            row = next(r for r in card if r["product"] == "codex")
            gate = mod.check_backend_health(runs, product="codex", now_ts=now,
                                            alive=set(), probe=self._dead_probe())
            self.assertTrue(row["majority_stub"])
            self.assertEqual(gate["state"], "dead")  # the card and the gate now agree

    def test_stub_rate_recovery_restores_spawns_automatically(self) -> None:
        # The other half of #3247's acceptance: once the stub logs age out of the
        # lookback window and a real turn lands, the hold clears with no operator.
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            (runs / "backend-health-codex.json").write_text(
                '{"product":"codex","state":"dead","since":"x","abandoned_lane":"tools"}',
                encoding="utf-8")
            for i in range(4):  # stale stubs, OUTSIDE the 90-min lookback
                self._mk(runs, 940 + i, f"20260708-01000{i}", backend="codex",
                         size=32, pid=700 + i, mtime=now - (120 + i) * 60)
            self._mk(runs, 950, "20260708-055000", backend="codex",
                     size=5000, pid=710, mtime=now - 60)  # a fresh real turn
            out = mod.check_backend_health(runs, product="codex", now_ts=now,
                                           alive=set(), probe=self._dead_probe())
            self.assertEqual(out["state"], "healthy")
            self.assertFalse((runs / "backend-health-codex.json").exists())  # hold cleared

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

    # --- permanent auth gap: a missing API key / no login is not a credit wall ---
    # The real codex banner: a doomed spawn dies on this every time, unchanged by any
    # nightly reset. It clears no byte floor games — it is a small stub — but it must be
    # classified as a PERMANENT gap so the hold stops the 30-min re-probe churn.
    AUTHGAP = (
        "# fak-spawn backend=codex issue=1350\n"
        "fak guard: Codex provider env $OPENAI_API_KEY is empty and no ChatGPT "
        "subscription auth.json was resolved. Run `codex login`, export "
        "OPENAI_API_KEY, or pass --api-key-env VAR.\n"
    )

    def _mk_body(self, runs: Path, issue: int, stamp: str, body: str, *,
                 backend: str, pid: int | None, mtime: float) -> None:
        import os
        log = runs / f"resolve-{issue}-{stamp}.log"
        log.write_text(body, encoding="utf-8")
        os.utime(log, (mtime, mtime))
        log.with_suffix(".backend").write_text(backend, encoding="utf-8")
        if pid is not None:
            pidf = log.with_suffix(".pid")
            pidf.write_text(str(pid), encoding="utf-8")
            os.utime(pidf, (mtime, mtime))

    def test_authgap_stubs_declare_dead_with_backed_off_reprobe(self) -> None:
        # A streak of auth-gap deaths is DEAD like any stub streak, but flagged auth_gap
        # so the re-probe backs off from 30 min to _BACKEND_AUTHGAP_REPROBE_MIN — a doomed
        # spawn every 30 min forever was the churn this fixes.
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            for i, iss in enumerate((1350, 1351, 1352)):
                self._mk_body(runs, iss, f"20260708-05000{i}", self.AUTHGAP,
                              backend="codex", pid=800 + i, mtime=now - (i + 1) * 60)
            out = mod.check_backend_health(runs, product="codex", lane="gateway",
                                           now_ts=now, alive=set(), probe=self._dead_probe())
            self.assertEqual(out["state"], "dead")
            self.assertTrue(out["auth_gap"])
            self.assertEqual(out["source"], "auth-gap")
            self.assertEqual(out["reprobe_min"], mod._BACKEND_AUTHGAP_REPROBE_MIN)

    def test_authgap_reprobe_backed_off_vs_credit_wall_cadence(self) -> None:
        # The 30-min credit-wall re-probe would let a doomed worker through 12x/6h; the
        # auth-gap hold admits at most one until _BACKEND_AUTHGAP_REPROBE_MIN elapses.
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            for i, iss in enumerate((1350, 1351, 1352)):
                self._mk_body(runs, iss, f"20260708-05000{i}", self.AUTHGAP,
                              backend="codex", pid=800 + i, mtime=now - (i + 1) * 60)
            first = mod.check_backend_health(runs, product="codex", lane="gateway",
                                             now_ts=now, alive=set(), probe=self._dead_probe())
            self.assertTrue(first["reprobe_due"])
            # +31 min: a credit wall would re-probe here; an auth gap still holds.
            soon = mod.check_backend_health(
                runs, product="codex", lane="gateway",
                now_ts=now + (mod._BACKEND_REPROBE_MIN + 1) * 60, alive=set(),
                probe=self._dead_probe())
            self.assertFalse(soon["reprobe_due"])
            later = mod.check_backend_health(
                runs, product="codex", lane="gateway",
                now_ts=now + (mod._BACKEND_AUTHGAP_REPROBE_MIN + 1) * 60, alive=set(),
                probe=self._dead_probe())
            self.assertTrue(later["reprobe_due"])

    def test_authgap_hold_is_sticky_after_stubs_age_out(self) -> None:
        # The crux: an unauthenticated backend we STOPPED spawning leaves an empty window.
        # A transient credit wall reads that as recovery (#3247) — but a missing key has
        # NOT recovered, so an auth_gap hold must persist with no productive log to clear it.
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            (runs / "backend-health-codex.json").write_text(
                '{"product":"codex","state":"dead","since":"x","abandoned_lane":"gateway",'
                '"auth_gap":true,"evidence_logs":["resolve-1350-x.log"]}',
                encoding="utf-8")
            # No logs at all in the window (spawns stopped) and NO productive turn.
            out = mod.check_backend_health(runs, product="codex", now_ts=now,
                                           alive=set(), probe=self._dead_probe())
            self.assertEqual(out["state"], "dead")   # sticky — NOT healthy no-streak
            self.assertTrue(out["auth_gap"])
            self.assertEqual(out["evidence_logs"], ["resolve-1350-x.log"])  # cause retained

    def test_authgap_hold_cleared_by_productive_log(self) -> None:
        # Recovery path: an operator sets the key and a real turn lands -> the sticky hold
        # clears with no operator touching the sidecar, same as any witnessed restore.
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            (runs / "backend-health-codex.json").write_text(
                '{"product":"codex","state":"dead","since":"x","abandoned_lane":"gateway",'
                '"auth_gap":true}', encoding="utf-8")
            self._mk_body(runs, 1360, "20260708-060000", "x" * 5000,  # a real turn
                          backend="codex", pid=900, mtime=now)
            out = mod.check_backend_health(runs, product="codex", now_ts=now,
                                           alive=set(), probe=self._dead_probe())
            self.assertEqual(out["state"], "healthy")
            self.assertFalse((runs / "backend-health-codex.json").exists())  # hold cleared

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
                                     ttl_s=900, holder="owner-1", runner=runner)
        self.assertTrue(out["acquired"])
        self.assertFalse(out["refused"])
        self.assertEqual(out["id"], "resolve-gateway")
        self.assertEqual(out["generation"], 2)

    def test_live_lane_lease_lanes_strips_resolve_prefix(self) -> None:
        mod = load()

        def runner(root, args, **k):
            self.assertEqual(args, ["live"])
            return {"rc": 0, "verdict": [
                {"lane": "resolve-docs", "tree": ["docs/**"]},
                {"lane": "resolve-tools", "tree": ["tools/**"]},
            ]}

        out = mod.live_lane_lease_lanes(ROOT, runner=runner)
        self.assertEqual(out["lanes"], ["docs", "tools"])
        self.assertEqual(len(out["records"]), 2)

    def test_live_lane_lease_lanes_fails_open_on_bad_output(self) -> None:
        mod = load()
        out = mod.live_lane_lease_lanes(
            ROOT, runner=lambda root, args, **k: {"rc": 0, "verdict": {"bad": True}})
        self.assertEqual(out["lanes"], [])
        self.assertTrue(out["fail_open"])

    def test_acquire_exit3_is_refused(self) -> None:
        mod = load()
        def runner(root, args, **k):
            return {"rc": 3, "verdict": {"verdict": {"ok": False, "reason": "LEASE_HELD"}}}
        out = mod.acquire_lane_lease(ROOT, "docs", tree=["docs/**"], ttl_s=900,
                                     holder="owner-1", runner=runner)
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

    def test_acquire_binds_current_session_when_valid(self) -> None:
        # New leases should not become another legacy host:pid record. When the
        # harness exposes a valid session id, pass it through to `fak leaseref
        # acquire --session` so the read-side liveness witness can classify the
        # lane by the owning session descriptor heartbeat. With no explicit holder,
        # omit --holder so the CLI mints the structured <node-id>/<session-id>
        # holder and then read that holder back from the returned record.
        mod = load()
        seen: dict = {}
        def runner(root, args, **k):
            self.assertEqual(args[0], "acquire")
            seen["args"] = list(args)
            return {"rc": 0, "verdict": {"verdict": {"ok": True},
                                         "record": {"id": "resolve-docs",
                                                    "holder": "node-a/sess_7-abc",
                                                    "generation": 1}}}
        with mock.patch.dict(os.environ, {"FAK_LEASE_SESSION_ID": "sess_7-abc",
                                          "CLAUDE_CODE_SESSION_ID": "",
                                          "FAK_LEASE_OWNER": ""}, clear=False):
            out = mod.acquire_lane_lease(ROOT, "docs", tree=["docs/**"],
                                         ttl_s=900, runner=runner)
        self.assertTrue(out["acquired"])
        self.assertEqual(out["session_id"], "sess_7-abc")
        self.assertEqual(out["holder"], "node-a/sess_7-abc")
        argv = seen["args"]
        self.assertNotIn("--holder", argv)
        self.assertEqual(argv[argv.index("--session") + 1], "sess_7-abc")
        self.assertNotIn("session_publish", out)

    def test_acquire_without_session_env_mints_and_publishes_session(self) -> None:
        mod = load()
        calls: list[list[str]] = []
        def runner(root, args, **k):
            calls.append(list(args))
            if args[0] == "session-publish":
                self.assertEqual(args[args.index("--session") + 1],
                                 "dispatch-HOST-ONE-1234")
                self.assertEqual(args[args.index("--ttl") + 1], "900")
                return {"rc": 0, "verdict": {"id": "dispatch-HOST-ONE-1234"}}
            self.assertEqual(args[0], "acquire")
            return {"rc": 0, "verdict": {"verdict": {"ok": True},
                                         "record": {"id": "resolve-docs",
                                                    "holder": "node-a/dispatch-HOST-ONE-1234",
                                                    "generation": 1}}}
        env = {"FAK_LEASE_SESSION_ID": "", "CLAUDE_CODE_SESSION_ID": "",
               "FAK_LEASE_OWNER": "", "COMPUTERNAME": "HOST ONE"}
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(mod.os, "getpid", return_value=1234):
                out = mod.acquire_lane_lease(ROOT, "docs", tree=["docs/**"],
                                             ttl_s=900, runner=runner)
        self.assertTrue(out["acquired"])
        self.assertEqual(out["session_id"], "dispatch-HOST-ONE-1234")
        self.assertEqual(out["holder"], "node-a/dispatch-HOST-ONE-1234")
        self.assertEqual(out["session_publish"],
                         {"published": True, "session_id": "dispatch-HOST-ONE-1234"})
        self.assertEqual([c[0] for c in calls], ["session-publish", "acquire"])
        argv = calls[1]
        self.assertNotIn("--holder", argv)
        self.assertEqual(argv[argv.index("--session") + 1],
                         "dispatch-HOST-ONE-1234")

    def test_acquire_honors_explicit_holder_with_session(self) -> None:
        mod = load()
        seen: dict = {}
        def runner(root, args, **k):
            self.assertEqual(args[0], "acquire")
            seen["args"] = list(args)
            return {"rc": 0, "verdict": {"verdict": {"ok": True},
                                         "record": {"id": "resolve-docs",
                                                    "holder": "owner-7",
                                                    "generation": 1}}}
        with mock.patch.dict(os.environ, {"FAK_LEASE_SESSION_ID": "sess_7-abc",
                                          "CLAUDE_CODE_SESSION_ID": ""}, clear=False):
            out = mod.acquire_lane_lease(ROOT, "docs", tree=["docs/**"],
                                         ttl_s=900, holder="owner-7", runner=runner)
        self.assertTrue(out["acquired"])
        argv = seen["args"]
        self.assertEqual(argv[argv.index("--holder") + 1], "owner-7")
        self.assertEqual(argv[argv.index("--session") + 1], "sess_7-abc")

    def test_acquire_skips_invalid_session_binding(self) -> None:
        # Fail open to the holder-only lease rather than breaking admission when
        # an external harness exports something that cannot be a refs/fak segment.
        mod = load()
        for sid in ("bad/session", "session-already-prefixed"):
            seen: dict = {}
            def runner(root, args, **k):
                self.assertEqual(args[0], "acquire")
                seen["args"] = list(args)
                return {"rc": 0, "verdict": {"verdict": {"ok": True},
                                             "record": {"id": "resolve-docs",
                                                        "holder": "owner-7",
                                                        "generation": 1}}}
            with mock.patch.dict(os.environ, {"FAK_LEASE_SESSION_ID": sid,
                                              "CLAUDE_CODE_SESSION_ID": ""}, clear=False):
                out = mod.acquire_lane_lease(ROOT, "docs", tree=["docs/**"],
                                             ttl_s=900, holder="owner-7", runner=runner)
            self.assertTrue(out["acquired"], sid)
            self.assertIsNone(out["session_id"], sid)
            self.assertNotIn("--session", seen["args"], sid)

    def test_acquire_other_exit_fails_open(self) -> None:
        mod = load()
        for rc in (1, 2, 127):
            def runner(root, args, _rc=rc, **k):
                return {"rc": _rc, "verdict": None}
            out = mod.acquire_lane_lease(ROOT, "gateway", tree=["internal/gateway/**"],
                                         ttl_s=900, holder="owner-1", runner=runner)
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

    def test_release_argv_carries_holder_and_generation(self) -> None:
        mod = load()
        seen: dict = {}
        def runner(root, args, **k):
            seen["args"] = list(args)
            return {"rc": 0, "verdict": {"ok": True}}
        out = mod.release_lane_lease(
            ROOT,
            {"id": "resolve-docs", "holder": "sess-7", "generation": 3},
            runner=runner)
        self.assertTrue(out["released"])
        argv = seen["args"]
        self.assertEqual(argv[0], "release")
        self.assertEqual(argv[argv.index("--id") + 1], "resolve-docs")
        self.assertEqual(argv[argv.index("--holder") + 1], "sess-7")
        self.assertEqual(argv[argv.index("--generation") + 1], "3")
        self.assertNotIn("--force", argv)

    def test_release_refusal_and_failure_leave_backstop(self) -> None:
        mod = load()
        def refused(root, args, **k):
            return {"rc": 3, "verdict": {"verdict": {"ok": False, "reason": "STALE_LEASE"}}}
        out = mod.release_lane_lease(
            ROOT, {"id": "resolve-docs", "holder": "sess-7"}, runner=refused)
        self.assertFalse(out["released"])
        self.assertTrue(out["refused"])
        self.assertEqual(out["reason"], "STALE_LEASE")

        def failed(root, args, **k):
            return {"rc": 1, "error": "git store busy"}
        out = mod.release_lane_lease(
            ROOT, {"id": "resolve-docs", "holder": "sess-7"}, runner=failed)
        self.assertFalse(out["released"])
        self.assertTrue(out["fail_open"])

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


# A stand-in for `fak leaseref live` shelling git: the grandchild INHERITS this
# process's stdout/stderr pipe (exactly what git.exe gets when fak shells it), so
# terminating only THIS pid leaves the pipe write handle open and any pipe-reader join
# blocks until the grandchild itself exits. The #4368 hang in one file.
_HUNG_GIT_STUB = """\
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
sys.stderr.write("grandchild=%d\\n" % child.pid)
sys.stderr.flush()
time.sleep(60)
"""


class LeaseProbeBoundedTest(unittest.TestCase):
    """#4368: the pre-spawn `fak leaseref` probe terminates on a bounded clock even
    when a descendant git holds the inherited pipe.

    Witnessed ancestry: `issue_resolve_dispatch.py -> fak.exe leaseref live -> git.exe
    -> git.exe`, dispatcher alive, no stdout/stderr, no worker — the tick reached
    neither WOULD_SPAWN/SPAWNED nor a refusal. `subprocess.run(timeout=N)` does NOT
    bound that on Windows: it kills the DIRECT child, then re-enters communicate() with
    no timeout to drain reader threads the grandchild is still holding open."""

    HARD_BOUND_S = 30  # probe 2s + drain 5s + reap slack; the un-reaped hang is 60s

    def _stub_path(self) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="issue-resolve-test-hunggit-"))
        atexit.register(shutil.rmtree, tmp, ignore_errors=True)
        stub = tmp / "hung_git_stub.py"
        stub.write_text(_HUNG_GIT_STUB, encoding="utf-8")
        return stub

    def test_hung_git_grandchild_is_reaped_on_a_bounded_clock(self) -> None:
        mod = load()
        stub = self._stub_path()
        mod._fak_bin = lambda root: [sys.executable, str(stub)]
        started = time.monotonic()
        res = mod._run_lease(ROOT, ["live"], timeout=2)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, self.HARD_BOUND_S,
                        f"lease probe was NOT bounded: {elapsed:.1f}s (the hung git "
                        "grandchild, not the timeout, decided when it returned)")
        self.assertEqual(res["rc"], mod.LEASE_TIMEOUT_RC)
        self.assertTrue(res["timeout"])
        self.assertEqual(res["timeout_s"], 2)
        self.assertIsNone(res["verdict"])
        # Actionable stderr text, not a bare rc: names the verb, the bound, the reap.
        self.assertIn("LEASE_PROBE_TIMEOUT", res["error"])
        self.assertIn("leaseref live", res["error"])
        # No orphan left behind: the tree reap (not Popen.kill) is what closes the
        # grandchild's INHERITED pipe handle, so a completed drain IS the witness that
        # the descendant died rather than outliving the dispatcher's probe.
        self.assertTrue(res["killed"]["ok"], res["killed"])
        self.assertTrue(res["drained"],
                        "the post-reap drain lapsed: a descendant still holds the pipe")

    def test_live_read_propagates_the_typed_timeout(self) -> None:
        # The pre-spawn lane-lease read fails OPEN (advisory visibility must never wedge
        # a tick) but carries the TYPED row, so the payload distinguishes "probe bounded
        # and reaped" from a dispatcher that simply never came back.
        mod = load()

        def timed_out(root, args, **k):
            return {"rc": mod.LEASE_TIMEOUT_RC, "stdout": "", "verdict": None,
                    "timeout": True, "timeout_s": 30, "drained": True,
                    "killed": {"ok": True, "pid": 4321},
                    "error": "LEASE_PROBE_TIMEOUT: `fak leaseref live` exceeded 30s"}

        out = mod.live_lane_lease_lanes(ROOT, runner=timed_out)
        self.assertEqual(out["lanes"], [])
        self.assertTrue(out["fail_open"])
        self.assertTrue(out["timeout"])
        self.assertEqual(out["timeout_s"], 30)
        self.assertTrue(out["drained"])
        self.assertEqual(out["killed"]["pid"], 4321)
        self.assertIn("LEASE_PROBE_TIMEOUT", out["detail"])

    def test_probe_never_reverts_to_unbounded_subprocess_run(self) -> None:
        # The ratchet: `subprocess.run(..., timeout=N)` reads as bounded and is not, so a
        # future edit that "simplifies" the probe back to it re-opens the hang.
        mod = load()
        # Read the CODE, not the prose: _run_lease's own docstring names the trap
        # ("deliberately does NOT use subprocess.run(timeout=...)"), so asserting over
        # the raw source would fail on the sentence that documents the fix. Drop the
        # docstring through the AST rather than by string subtraction — since 3.13 the
        # compiler dedents __doc__, so it is no longer a literal substring of the source.
        tree = ast.parse(textwrap.dedent(inspect.getsource(mod._run_lease)))
        fn = tree.body[0]
        if (fn.body and isinstance(fn.body[0], ast.Expr)
                and isinstance(fn.body[0].value, ast.Constant)
                and isinstance(fn.body[0].value.value, str)):
            fn.body = fn.body[1:]
        src = ast.unparse(fn)
        self.assertNotIn("subprocess.run(", src,
                         "_run_lease must not use subprocess.run: its Windows timeout "
                         "path re-enters communicate() unbounded (#4368)")
        self.assertIn("subprocess.Popen(", src)
        self.assertIn("_terminate_lease_probe(", src)


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


class CommitAuditAbstainHoldsTest(unittest.TestCase):
    """Recent candidate-bound test commits whose audit ABSTAINS are surfaced as
    closure-witness mismatch holds instead of being selected again."""

    def test_test_scope_abstain_commit_holds_matching_candidate(self) -> None:
        mod = load()
        log = (
            "41f23ea7\x1f"
            "test(boundarylint): cover cmp-diff change-detector lint (#2867) (fak boundarylint)\n"
            "\x1e"
        )

        def git(root, args):
            return (0, log)

        def audit(root, sha):
            return {
                "sha": sha,
                "verdict": "ABSTAIN",
                "witness": "abstain",
                "claim_kind": "none",
                "reason": "subject makes no checkable code/test claim",
                "test_files": ["internal/boundarylint/changedetector_test.go"],
            }

        rows = mod.commit_audit_abstain_holds(
            ROOT, {2867, 9999}, git=git, audit_runner=audit)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["issue"], 2867)
        self.assertEqual(rows[0]["code"], "COMMIT_AUDIT_ABSTAIN")
        self.assertEqual(rows[0]["test_files"],
                         ["internal/boundarylint/changedetector_test.go"])


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

    def test_witness_carries_exact_lease_session_identity(self) -> None:
        import json
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = self._mk(runs, 3330, "20260629-120125", pid=4244)
            log.with_suffix(".lease").write_text(json.dumps({
                "id": "resolve-sessionmine", "holder": "worker",
                "session_id": "sess-3330", "tree": ["internal/sessionmine/**"],
            }), encoding="utf-8")
            def git(root, args):
                return (0, "abc1234\x1ffeat: bind (#3330)\n")
            def audit(root, sha):
                return {"verdict": "OK", "witness": "diff-witnessed"}
            out = mod.witness_exited_workers(runs, ROOT, live=True, probe=self._dead,
                                             git=git, audit_runner=audit)
            self.assertEqual(out["witnessed"][0]["session_id"], "sess-3330")
            self.assertEqual(json.loads(log.with_suffix(".witness").read_text(encoding="utf-8"))["session_id"], "sess-3330")

    def test_dead_worker_releases_recorded_lane_lease(self) -> None:
        import json
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = self._mk(runs, 1324, "20260629-120150", pid=4248)
            log.with_suffix(".lease").write_text(json.dumps({
                "id": "resolve-docs",
                "holder": "session-1",
                "generation": 5,
                "tree": ["docs/**"],
            }), encoding="utf-8")
            def git(root, args):
                return (0, "cafef00d\x1ffix: bind (#1324)\n")
            def audit(root, sha):
                return {"verdict": "OK", "witness": "diff-witnessed"}
            seen: dict = {}
            def lease_runner(root, args, **k):
                seen["args"] = list(args)
                return {"rc": 0, "verdict": {"ok": True}}
            out = mod.witness_exited_workers(runs, ROOT, live=True, probe=self._dead,
                                             git=git, audit_runner=audit,
                                             lease_runner=lease_runner)
            self.assertEqual(len(out["lease_released"]), 1)
            self.assertEqual(out["lease_released"][0]["id"], "resolve-docs")
            argv = seen["args"]
            self.assertEqual(argv[0], "release")
            self.assertEqual(argv[argv.index("--holder") + 1], "session-1")
            self.assertEqual(argv[argv.index("--generation") + 1], "5")
            self.assertNotIn("--force", argv)
            side = json.loads(log.with_suffix(".witness").read_text(encoding="utf-8"))
            self.assertTrue(side["lease_release"]["released"])

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

    def test_already_witnessed_dead_worker_releases_recorded_lane_lease(self) -> None:
        import json
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = self._mk(runs, 1324, "20260629-120450", pid=4249)
            log.with_suffix(".witness").write_text(
                json.dumps({"claim": "CLAIM_WITNESSED"}), encoding="utf-8")
            log.with_suffix(".lease").write_text(json.dumps({
                "id": "resolve-tools",
                "holder": "session-2",
                "generation": 8,
                "tree": ["tools/**"],
            }), encoding="utf-8")
            def boom_git(root, args):
                raise AssertionError("an already-witnessed worker must not be re-audited")
            seen: dict = {}
            def lease_runner(root, args, **k):
                seen["args"] = list(args)
                return {"rc": 0, "verdict": {"ok": True}}
            out = mod.witness_exited_workers(runs, ROOT, live=True, probe=self._dead,
                                             git=boom_git, lease_runner=lease_runner)
            self.assertEqual(out["audited"], [])
            self.assertEqual(out["lease_release_retried"], [
                {
                    "log": log.name,
                    "id": "resolve-tools",
                    "reason": "already_witnessed_dead_worker",
                    "released": True,
                }
            ])
            self.assertEqual(len(out["lease_released"]), 1)
            self.assertEqual(out["lease_release_failed"], [])
            self.assertEqual(out["lease_released"][0]["id"], "resolve-tools")
            argv = seen["args"]
            self.assertEqual(argv[0], "release")
            self.assertEqual(argv[argv.index("--holder") + 1], "session-2")
            self.assertEqual(argv[argv.index("--generation") + 1], "8")
            side = json.loads(log.with_suffix(".witness").read_text(encoding="utf-8"))
            self.assertTrue(side["lease_release"]["released"])

    def test_dry_run_audits_but_writes_no_sidecar(self) -> None:
        import json
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = self._mk(runs, 1324, "20260629-120500", pid=4247)
            log.with_suffix(".lease").write_text(json.dumps({
                "id": "resolve-docs", "holder": "session-1", "generation": 5,
            }), encoding="utf-8")
            def git(root, args):
                return (0, "cafef00d\x1ffix: bind (#1324)\n")
            def audit(root, sha):
                return {"verdict": "ABSTAIN", "witness": "abstain"}
            def boom_release(root, args, **k):
                raise AssertionError("dry-run witness must not release a lease")
            out = mod.witness_exited_workers(runs, ROOT, live=False, probe=self._dead,
                                             git=git, audit_runner=audit,
                                             lease_runner=boom_release)
            self.assertEqual(out["unwitnessed"][0]["claim"], "CLAIM_UNWITNESSED")
            self.assertFalse(log.with_suffix(".witness").exists())
            self.assertEqual(out["lease_released"], [])

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


class ClassifyNoCommitReasonTest(unittest.TestCase):
    """classify_no_commit_reason tags WHY a finished worker shipped no commit, from
    the log tail. Each documented terminal-block signature maps to its reason; an
    unrecognized log stays UNKNOWN (no false positive)."""

    def _write(self, text: str):
        import tempfile
        d = tempfile.mkdtemp()
        p = Path(d) / "resolve-1338-20260629-221102.log"
        p.write_text(text, encoding="utf-8")
        return p

    def test_self_modify(self) -> None:
        mod = load()
        p = self._write("...\nfinish=end_turn safety=blocked:1 reason=SELF_MODIFY\n"
                        "[fak] refused 1 tool call(s): Bash (SELF_MODIFY/ESCALATE).\n")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_SELF_MODIFY)

    def test_policy_block(self) -> None:
        mod = load()
        p = self._write("...\nfinish=end_turn safety=blocked:1 reason=POLICY_BLOCK\n"
                        "[fak] refused 1 tool call(s): Bash (POLICY_BLOCK/TERMINAL).\n")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_POLICY_BLOCK)

    def test_auth_wall_glm(self) -> None:
        mod = load()
        p = self._write("> build · glm-5.2\nWeekly Limit Exhausted. Your limit will "
                        "reset at 2026-07-01.\n")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_AUTH_WALL)

    def test_auth_wall_claude_cap_banner(self) -> None:
        mod = load()
        # _CAP_BANNER_RE: "hit your … limit" (Claude/codex wording).
        p = self._write("You've hit your weekly usage limit for Claude.\n")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_AUTH_WALL)

    def test_auth_wall_guarded_provider_429(self) -> None:
        mod = load()
        p = self._write("fak-turn trace=guard FAILED reason=rate_limited\n"
                        "API Error: Request rejected (429) · upstream rate-limited "
                        "the request (HTTP 429)\n")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_AUTH_WALL)

    def test_auth_wall_missing_openai_api_key(self) -> None:
        mod = load()
        p = self._write("ERROR: Missing environment variable: `OPENAI_API_KEY`.\n"
                        "ERROR: Missing environment variable: `OPENAI_API_KEY`.\n"
                        "fak guard: codex exited abnormally (code 1).\n")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_AUTH_WALL)

    def test_off_trunk(self) -> None:
        mod = load()
        p = self._write("git commit refused: OFF_TRUNK — work on main only.\n")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_OFF_TRUNK)

    def test_preview_confirm_feedback(self) -> None:
        mod = load()
        p = self._write(
            "tool call refused: REQUIRE_WITNESS/ESCALATE\n"
            "{\"_fak_confirm\":\"fak-3bff999a1f01335d\"}\n"
            "step=20\n"
            "exiting loop\n")
        self.assertEqual(mod.classify_no_commit_reason(p),
                         mod.NO_COMMIT_PREVIEW_CONFIRM)

    def test_banner_noop_small_log(self) -> None:
        mod = load()
        # The 122-byte opencode no-op: spawn banner + "> build · glm-…" and nothing.
        p = self._write("# fak-spawn 20260629-221102 issue=1338 lane=docs backend=opencode\n"
                        "> build · glm-4.5-air\n")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_BANNER_NOOP)

    def test_banner_in_large_real_log_is_not_noop(self) -> None:
        mod = load()
        # A real worker also prints "> build · …" at startup, but its log is large and
        # carries real turns — it must NOT be misread as a banner no-op.
        big = "> build · glm-5.2\n" + ("fak-turn trace=guard ok saved=80k tok\n" * 200)
        p = self._write(big)
        # No banner no-op, and (#5870) no bare `unknown` either: the tail carries no
        # guard epilogue, so the run reads as killed mid-turn.
        self.assertEqual(mod.classify_no_commit_reason(p),
                         mod.NO_COMMIT_DIED_BEFORE_EPILOGUE)

    def test_restart_exhausted_is_typed_with_count_and_cause(self) -> None:
        mod = load()
        p = self._write("")
        p.write_text("fak guard: managed-context status restart_exhausted count=3 "
                     "dominant_cause=BUDGET_CONTEXT_EXHAUSTED from_trace=t\n", encoding="utf-8")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_RESTART_EXHAUSTED)
        self.assertEqual(mod.classify_restart_exhaustion(p), {
            "reason": mod.NO_COMMIT_RESTART_EXHAUSTED, "restart_count": 3,
            "dominant_cause": "BUDGET_CONTEXT_EXHAUSTED"})

    def test_live_reset_limit_corpus_is_typed(self) -> None:
        mod = load()
        p = self._write("")
        p.write_text("fak guard: managed-context status reset_limit limit=16 "
                     "reason=BUDGET_CONTEXT_EXHAUSTED continuity=degraded\n", encoding="utf-8")
        self.assertEqual(mod.classify_restart_exhaustion(p)["restart_count"], 16)
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_RESTART_EXHAUSTED)

    def test_live_reset_limit_survives_long_guard_epilogue(self) -> None:
        mod = load()
        p = self._write(
            "fak guard: managed-context status reset_limit limit=16 "
            "reason=BUDGET_CONTEXT_EXHAUSTED continuity=degraded\n"
            + ("resource and audit summary padding\n" * 190)
            + "fak guard: claude exited abnormally (code 1).\n")
        # The live #2788 guard epilogue put the typed terminal more than 4 KiB
        # before EOF. The precise restart scan is larger but remains bounded.
        self.assertGreater(p.stat().st_size, mod._CAP_TAIL_BYTES)
        self.assertEqual(mod.classify_restart_exhaustion(p)["restart_count"], 16)
        self.assertEqual(mod.classify_no_commit_reason(p),
                         mod.NO_COMMIT_RESTART_EXHAUSTED)

    def test_residual_with_no_signature_is_typed_not_unknown(self) -> None:
        mod = load()
        # #5870: the residual is no longer a bare `unknown`. This log matches no
        # failure signature AND carries no guard epilogue -> died_before_epilogue.
        p = self._write("the worker ran a few turns and then exited cleanly\n")
        # Asserted with the PRE-EXISTING constant first, so this fails as a
        # wrong-behavior assertion on the old sweep, not merely as a missing symbol.
        self.assertNotEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_UNKNOWN)
        self.assertEqual(mod.classify_no_commit_reason(p),
                         mod.NO_COMMIT_DIED_BEFORE_EPILOGUE)

    def test_missing_log_artifact(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "resolve-1338-20260629-221102.log"
            self.assertEqual(mod.classify_no_commit_reason(p),
                             mod.NO_COMMIT_MISSING_LOG)

    def test_self_modify_wins_over_banner(self) -> None:
        mod = load()
        # Priority: a structural guard block is reported even if a startup banner is
        # also present (the guard refusal is the actionable cause).
        p = self._write("> build · glm-5.2\n...\nreason=SELF_MODIFY\n")
        self.assertEqual(mod.classify_no_commit_reason(p), mod.NO_COMMIT_SELF_MODIFY)


class RefineUnknownNoCommitTest(unittest.TestCase):
    """The residual no-commit bucket is TYPED AT WRITE TIME (#5870), by the same two
    markers internal/dispatchconservation applies at read time (#5867), so the .witness
    sidecar the dispatcher routes off is self-describing instead of an opaque blob."""

    # `guardSection` (cmd/fak/guard_format_layout.go) pads the rule to 60 columns.
    def _section(self, name: str) -> str:
        head = "── guard · " + name + " "
        return head + "─" * max(0, 60 - len(head)) + "\n"

    def test_guard_epilogue_means_clean_exit(self) -> None:
        mod = load()
        tail = ("fak-turn trace=guard ok\n" + self._section("audit")
                + "  refused                    0\n")
        self.assertEqual(mod.refine_unknown_no_commit(tail),
                         mod.NO_COMMIT_CLEAN_EXIT)

    def test_epilogue_without_a_cache_window_section_still_counts(self) -> None:
        mod = load()
        # THE #5867 LESSON, pinned. `guard · cache window` is emitted only when the
        # session recorded cache turns, so keying on it books a quiet clean exit as a
        # death. Keying on the SECTION RULE has no such blind spot: an epilogue whose
        # only section is `audit` is still an epilogue.
        tail = self._section("audit") + "  refused                    0\n"
        self.assertNotIn("cache window", tail)
        self.assertEqual(mod.refine_unknown_no_commit(tail),
                         mod.NO_COMMIT_CLEAN_EXIT)

    def test_no_epilogue_means_died_before_epilogue(self) -> None:
        mod = load()
        tail = "fak-turn trace=e3f9 in-flight saved=12k tok\n"
        self.assertEqual(mod.refine_unknown_no_commit(tail),
                         mod.NO_COMMIT_DIED_BEFORE_EPILOGUE)

    def test_spawn_failure_wins_over_a_partial_epilogue(self) -> None:
        mod = load()
        # A guard that could not exec the agent ALSO prints a partial epilogue, and
        # "the child never started" is the more specific — and more fixable — claim.
        tail = (self._section("audit")
                + 'fak guard: could not run "claude": exec: not found\n')
        self.assertEqual(mod.refine_unknown_no_commit(tail),
                         mod.NO_COMMIT_GUARD_SPAWN_FAILED)

    def test_empty_tail_fails_open_to_died_before_epilogue(self) -> None:
        mod = load()
        self.assertEqual(mod.refine_unknown_no_commit(""),
                         mod.NO_COMMIT_DIED_BEFORE_EPILOGUE)

    def test_classifier_never_returns_a_bare_unknown(self) -> None:
        mod = load()
        # The writer's vocabulary is now total over the residual: `unknown` survives
        # only in HISTORIC sidecars, which is what keeps it meaningful as a
        # vocabulary-drift alarm on the reading side.
        self.assertNotIn(mod.NO_COMMIT_UNKNOWN, {
            mod.refine_unknown_no_commit(t) for t in
            ("", "nothing here\n", self._section("audit"),
             "fak guard: could not run \"x\": e\n")})

    def test_refined_classes_are_not_added_to_the_reblock_hold_set(self) -> None:
        mod = load()
        # MEASURED DECISION, pinned so it is not "fixed" by assumption. #5869 predicted
        # that typing this bucket would lift its re-block streak hold from 6 runs/1.5h
        # to 24 runs/7.3h "with no change here". Replaying the same window (2026-08-04
        # .. 08-07, 293 finished .witness records / 95.3h over 121 issues) against the
        # typed sidecars: with the hold set UNCHANGED the yield is bit-identical — 6
        # runs / 1.5h, the same six runs — because a `clean_exit_no_commit` streak is
        # not a streak of a re-blockable reason. Adding clean_exit_no_commit here DOES
        # reach 33 runs / 10.9h, but it suppresses SIX runs that landed a witnessed
        # commit (62b998c75, 9f1558700, 0573fbe1d, b6a80c576, b4d656107, 0d0255ad8),
        # destroying the zero-witnessed-loss property that is #5869's whole safety
        # argument. A clean exit that lands nothing is not a refusal to re-block on.
        for reason in (mod.NO_COMMIT_CLEAN_EXIT, mod.NO_COMMIT_DIED_BEFORE_EPILOGUE,
                       mod.NO_COMMIT_GUARD_SPAWN_FAILED):
            self.assertNotIn(reason, mod._HOLD_NO_COMMIT_REASONS)
        self.assertEqual(mod.held_no_commit_issues({"no_commit": [
            {"issue": 11, "reason": mod.NO_COMMIT_CLEAN_EXIT},
            {"issue": 22, "reason": mod.NO_COMMIT_DIED_BEFORE_EPILOGUE},
            {"issue": 33, "reason": mod.NO_COMMIT_GUARD_SPAWN_FAILED}]}), set())


class HeldNoCommitIssuesTest(unittest.TestCase):
    """held_no_commit_issues reads the recorded no-commit reason and HOLDS only the
    re-blockable guard refusals; a transient auth_wall or an UNKNOWN is left to the
    time cooldown, not structurally held (#1396)."""

    def test_holds_reblockable_guard_feedback(self) -> None:
        mod = load()
        w = {"no_commit": [
            {"issue": 11, "reason": mod.NO_COMMIT_SELF_MODIFY},
            {"issue": 22, "reason": mod.NO_COMMIT_POLICY_BLOCK},
            {"issue": 33, "reason": mod.NO_COMMIT_PREVIEW_CONFIRM}]}
        self.assertEqual(mod.held_no_commit_issues(w), {11, 22, 33})

    def test_does_not_hold_auth_wall_banner_offtrunk_or_unknown(self) -> None:
        mod = load()
        w = {"no_commit": [
            {"issue": 33, "reason": mod.NO_COMMIT_AUTH_WALL},
            {"issue": 44, "reason": mod.NO_COMMIT_BANNER_NOOP},
            {"issue": 55, "reason": mod.NO_COMMIT_OFF_TRUNK},
            {"issue": 66, "reason": mod.NO_COMMIT_UNKNOWN},
            {"issue": 77, "reason": mod.NO_COMMIT_MISSING_LOG}]}
        self.assertEqual(mod.held_no_commit_issues(w), set())

    def test_mixed_holds_only_reblockable(self) -> None:
        mod = load()
        w = {"no_commit": [
            {"issue": 11, "reason": mod.NO_COMMIT_SELF_MODIFY},
            {"issue": 33, "reason": mod.NO_COMMIT_AUTH_WALL}]}
        self.assertEqual(mod.held_no_commit_issues(w), {11})

    def test_failopen_on_skipped_empty_or_odd_record(self) -> None:
        mod = load()
        self.assertEqual(mod.held_no_commit_issues(None), set())
        self.assertEqual(mod.held_no_commit_issues({"skipped": True}), set())
        self.assertEqual(mod.held_no_commit_issues({"no_commit": []}), set())
        # an odd record (no issue / wrong type) is skipped, never raised
        self.assertEqual(mod.held_no_commit_issues(
            {"no_commit": [{"reason": mod.NO_COMMIT_SELF_MODIFY}, "garbage",
                           {"issue": "x", "reason": mod.NO_COMMIT_SELF_MODIFY}]}), set())


class TickExitCodeTest(unittest.TestCase):
    """A scheduled-task LastResult must flag only a genuine malfunction, never a
    benign no-work / backpressure tick (the chronic-0x1 durability bug)."""

    def test_benign_no_work_and_backpressure_exit_zero(self) -> None:
        mod = load()
        for action in ("no_issue", "no_lane", "lane_busy", "lane_leased",
                       "same_issue_wip", "dirty_path_collision",
                       "refused", "weekly_capped", "backend_unhealthy"):
            with self.subTest(action=action):
                # ok=False on every benign verdict, yet the tick ran correctly.
                self.assertEqual(mod.tick_exit_code({"action": action, "ok": False}), 0)

    def test_dispatched_work_exits_zero(self) -> None:
        mod = load()
        self.assertEqual(mod.tick_exit_code({"action": "spawned", "ok": True}), 0)
        self.assertEqual(mod.tick_exit_code({"action": "would_spawn", "ok": True}), 0)

    def test_single_transient_spawn_failed_is_not_red(self) -> None:
        # #2636: a lone SPAWN_FAILED (child died <5s on an otherwise-correct tick) is the
        # ~1-in-25 self-healing baseline — failover retries it, so it must NOT redden
        # LastTaskResult. An un-stamped payload (or streak 1) reads green.
        mod = load()
        self.assertEqual(mod.tick_exit_code({"action": "spawn_failed", "ok": False}), 0)
        self.assertEqual(mod.tick_exit_code(
            {"action": "spawn_failed", "ok": False, "spawn_failed_streak": 1}), 0)
        # just below the threshold stays green (still plausibly transient noise)
        self.assertEqual(mod.tick_exit_code(
            {"action": "spawn_failed", "ok": False,
             "spawn_failed_streak": mod.SPAWN_FAILED_RED_STREAK - 1}), 0)

    def test_repeated_same_target_spawn_failed_is_red(self) -> None:
        # #2636: a SPAWN_FAILED that keeps recurring on the SAME target (failover not
        # healing it) reaches SPAWN_FAILED_RED_STREAK and DOES go red.
        mod = load()
        self.assertEqual(mod.tick_exit_code(
            {"action": "spawn_failed", "ok": False,
             "spawn_failed_streak": mod.SPAWN_FAILED_RED_STREAK}), 1)
        self.assertEqual(mod.tick_exit_code(
            {"action": "spawn_failed", "ok": False,
             "spawn_failed_streak": mod.SPAWN_FAILED_RED_STREAK + 1}), 1)

    def test_spawn_failure_streak_bumps_per_target_and_clears_on_success(self) -> None:
        # The production wiring behind the done condition: the same-target run-length is
        # per-target and per-backend, and a successful spawn resets it (#2636).
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self.assertEqual(mod.bump_spawn_failure_streak(runs, 1515, "claude"), 1)
            self.assertEqual(mod.bump_spawn_failure_streak(runs, 1515, "claude"), 2)
            # a different target keeps its own independent run-length
            self.assertEqual(mod.bump_spawn_failure_streak(runs, 42, "claude"), 1)
            # a different backend never shares the counter
            self.assertEqual(mod.bump_spawn_failure_streak(runs, 1515, "opencode"), 1)
            # a successful spawn of the same target resets it to zero
            mod.clear_spawn_failure_streak(runs, 1515, "claude")
            self.assertEqual(mod.bump_spawn_failure_streak(runs, 1515, "claude"), 1)

    def test_unknown_or_malformed_action_exits_nonzero(self) -> None:
        mod = load()
        # An unrecognised/missing action fails loud rather than masking a new mode.
        self.assertEqual(mod.tick_exit_code({"action": "kaboom"}), 1)
        self.assertEqual(mod.tick_exit_code({}), 1)
        self.assertEqual(mod.tick_exit_code(None), 1)  # type: ignore[arg-type]

    def test_benign_actions_cover_every_non_failure_verdict(self) -> None:
        # Guard against a new evaluate() action silently defaulting to "failure":
        # spawn_failed is the ONLY emitted action outside the benign set.
        mod = load()
        emitted = {"spawned", "would_spawn", "no_issue", "no_lane", "lane_busy",
                   "lane_leased", "same_issue_wip", "dirty_path_collision",
                   "multi_lane_scope", "refused", "weekly_capped",
                   "backend_unhealthy", "backend_health_skip", "seat_cooled",
                   "spawn_failed"}
        self.assertEqual(emitted - mod.BENIGN_ACTIONS, {"spawn_failed"})


class SeatKeyedSpawnFailureStreakTest(unittest.TestCase):
    """#4591 keying fix: the target-keyed pair lets a dead needs_login seat
    cycling across DIFFERENT issues evade cooldown (every target rotation
    restarts that counter at 1); the seat-keyed pair accrues ONE run-length per
    seat regardless of which issue burned the spawn."""

    def test_seat_cycling_across_distinct_issues_accrues_one_streak(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # The EVASION this test pins: 3 failures on 3 DISTINCT targets never
            # lift any TARGET-keyed run-length above 1 ...
            for target in (101, 202, 303):
                self.assertEqual(
                    mod.bump_spawn_failure_streak(runs, target, "claude"), 1)
            # ... while the SEAT-keyed run-length sees one seat fail 3x in a row.
            self.assertEqual(
                mod.bump_spawn_failure_streak_seat(runs, "july17", "claude"), 1)
            self.assertEqual(
                mod.bump_spawn_failure_streak_seat(runs, "july17", "claude"), 2)
            self.assertEqual(
                mod.bump_spawn_failure_streak_seat(runs, "july17", "claude"), 3)
            state = mod.seat_spawn_failure_state(runs, "july17", "claude")
            self.assertEqual(state["streak"], 3)
            self.assertTrue(state["cooled"])
            self.assertFalse(state["reprobe_due"])  # failure just recorded

    def test_streaks_are_per_seat_and_per_backend(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self.assertEqual(mod.bump_spawn_failure_streak_seat(runs, "a", "claude"), 1)
            self.assertEqual(mod.bump_spawn_failure_streak_seat(runs, "a", "claude"), 2)
            # a different seat keeps its own independent run-length
            self.assertEqual(mod.bump_spawn_failure_streak_seat(runs, "b", "claude"), 1)
            # a different backend never shares the counter
            self.assertEqual(mod.bump_spawn_failure_streak_seat(runs, "a", "opencode"), 1)

    def test_clean_launch_on_the_seat_clears_it_whatever_the_target(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            for _ in range(3):
                mod.bump_spawn_failure_streak_seat(runs, "a", "claude")
            # The clear is SEAT-scoped, not same-target-scoped: any clean launch
            # on the seat breaks the streak.
            mod.clear_spawn_failure_streak_seat(runs, "a", "claude")
            self.assertEqual(mod.bump_spawn_failure_streak_seat(runs, "a", "claude"), 1)

    def test_blank_tag_records_nothing_and_never_cools(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self.assertEqual(mod.bump_spawn_failure_streak_seat(runs, "", "claude"), 0)
            self.assertEqual(mod.bump_spawn_failure_streak_seat(runs, None, "claude"), 0)
            state = mod.seat_spawn_failure_state(runs, "", "claude")
            self.assertEqual(state["streak"], 0)
            self.assertFalse(state["cooled"])

    def test_cooled_seat_earns_a_reprobe_after_the_window(self) -> None:
        # A permanently-skipped seat could never clear its own streak, so the
        # cool admits ONE probe spawn every SEAT_STREAK_REPROBE_MIN minutes.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            t0 = 1_000_000.0
            for _ in range(mod.SPAWN_FAILED_RED_STREAK):
                mod.bump_spawn_failure_streak_seat(runs, "s", "claude", now_ts=t0)
            fresh = mod.seat_spawn_failure_state(runs, "s", "claude", now_ts=t0 + 60)
            self.assertTrue(fresh["cooled"])
            self.assertFalse(fresh["reprobe_due"])
            later = mod.seat_spawn_failure_state(
                runs, "s", "claude",
                now_ts=t0 + mod.SEAT_STREAK_REPROBE_MIN * 60 + 1)
            self.assertTrue(later["cooled"])
            self.assertTrue(later["reprobe_due"])


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
            EvaluateTest._patch(self, mod, pre=self.DEAD,
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
            EvaluateTest._patch(self, mod, pre=self.DEAD,
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
            EvaluateTest._patch(self, mod, pre=self.DEAD,
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


class AppendFleetTrendRowTest(unittest.TestCase):
    """Backend dispatcher ticks preserve scoped diagnostics without claiming an
    authoritative all-backend fleet census. The producer remains best-effort and bounded."""

    def test_live_tick_appends_backend_scoped_row(self) -> None:
        mod = load()
        import fleet_trend
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ledger = str(root / fleet_trend.DEFAULT_LEDGER)
            res = mod.append_fleet_trend_row(
                root, {"backend": "codex", "preflight": {"live": 3}}, now="2026-07-14T17:00:00Z")
            self.assertTrue(res["ok"])
            self.assertEqual(res["ledger"], ledger)
            rows = fleet_trend.tail(ledger, 24)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ts"], "2026-07-14T17:00:00Z")
            self.assertEqual(rows[0]["backend_live"], 3.0)
            self.assertEqual(rows[0]["scope"], "backend")
            self.assertEqual(rows[0]["backend"], "codex")
            self.assertNotIn("live", rows[0])
            # a second live tick APPENDS (a trend needs history), never overwrites.
            mod.append_fleet_trend_row(
                root, {"backend": "codex", "preflight": {"live": 1}}, now="2026-07-14T17:05:00Z")
            self.assertEqual([r["backend_live"] for r in fleet_trend.tail(ledger, 24)],
                             [3.0, 1.0])  # both ticks recorded, oldest→newest

    def test_missing_preflight_live_records_scoped_zero(self) -> None:
        # Missing backend-local evidence remains an explicitly scoped zero; it never
        # fabricates an aggregate fleet-wide zero.
        mod = load()
        import fleet_trend
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ledger = str(root / fleet_trend.DEFAULT_LEDGER)
            res = mod.append_fleet_trend_row(root, {}, now="2026-07-14T17:00:00Z")
            self.assertTrue(res["ok"])
            row = fleet_trend.tail(ledger, 24)[0]
            self.assertEqual(row["backend_live"], 0.0)
            self.assertNotIn("live", row)

    def test_append_error_is_swallowed_never_fails_the_tick(self) -> None:
        # Best-effort: an append that raises must be reported, not propagated — a
        # trend-ledger hiccup can never take down the dispatcher tick that feeds it.
        mod = load()

        def boom(*_a, **_k):
            raise OSError("disk full")

        with tempfile.TemporaryDirectory() as d:
            res = mod.append_fleet_trend_row(
                Path(d), {"preflight": {"live": 2}}, append=boom, now="t")
            self.assertFalse(res["ok"])
            self.assertIn("disk full", res["error"])


class MultiLaneScopeTest(unittest.TestCase):
    """The #2615 guard: refuse an issue whose named file families fall outside the
    chosen lane's LEASE TREE (the collision authority), not just its label. Pure —
    canned trees, no dos/gh."""

    TREES = {
        "ci": [".github/**"],
        "docs": ["docs/**"],
        "claude": [".claude/**"],
        "tools": ["tools/**", "scripts/**"],
        "gateway": ["internal/gateway/**"],
    }
    LANES = set(TREES)

    def test_broad_issue_under_narrow_lane_is_multi_lane(self) -> None:
        # #2233 shape: a ci-leased worker (tree .github/**) told to edit .claude/**
        # and docs/** — the families the narrow lease cannot protect.
        mod = load()
        body = ("Callers live in `.claude/skills/x/SKILL.md`, `.github/workflows/y.yml`, "
                "and `docs/archive/z.md`.")
        scan = mod.multi_lane_scope(body, "ci", [".github/**"], self.TREES, self.LANES)
        self.assertTrue(scan["multi_lane"])
        self.assertEqual(scan["uncovered_lanes"], ["claude", "docs"])
        # The path the lease DOES cover is not flagged.
        self.assertIn(".github/workflows/y.yml", scan["covered_paths"])

    def test_single_lane_issue_is_not_refused(self) -> None:
        # An issue naming only its own lane's tree clears — no false positive.
        mod = load()
        body = "Edit `tools/issue_resolve_dispatch.py` and `tools/issue_lane_router.py`."
        scan = mod.multi_lane_scope(body, "tools", ["tools/**", "scripts/**"],
                                    self.TREES, self.LANES)
        self.assertFalse(scan["multi_lane"])
        self.assertEqual(scan["uncovered"], [])

    def test_lease_covering_every_family_clears(self) -> None:
        # The done-condition escape hatch: when the lease tree covers every named
        # family, the same broad body is admissible (no refusal).
        mod = load()
        body = "Touches `.github/workflows/a.yml` and `docs/b.md`."
        wide = {"wide": [".github/**", "docs/**"]}
        trees = {**self.TREES, **wide}
        scan = mod.multi_lane_scope(body, "wide", [".github/**", "docs/**"],
                                    trees, set(trees))
        self.assertFalse(scan["multi_lane"])

    def test_glob_and_bare_paths_do_not_trip_the_guard(self) -> None:
        # A `tools/*.py` glob or a bare `Makefile`/`dos.toml` is not a rooted path,
        # so it never manufactures a spurious cross-lane family.
        mod = load()
        body = "Rewrite `tools/*.py`, `Makefile`, and `dos.toml` command strings."
        scan = mod.multi_lane_scope(body, "ci", [".github/**"], self.TREES, self.LANES)
        self.assertFalse(scan["multi_lane"])

    def test_reason_names_families_and_split_path(self) -> None:
        mod = load()
        scan = {"chosen_lane": "ci", "chosen_tree": [".github/**"],
                "uncovered": [{"path": ".claude/skills/", "lanes": ["claude"]}],
                "uncovered_lanes": ["claude"]}
        reason = mod.multi_lane_scope_reason(2233, scan)
        self.assertIn("#2233", reason)
        self.assertIn(".claude/skills/", reason)
        self.assertIn("Split into", reason)


class DirtyPathCollisionTest(unittest.TestCase):
    def test_exact_dirty_paths_named_in_issue_collide(self) -> None:
        mod = load()
        scan = mod.dirty_path_collision(
            "Likely files: `cmd/fak/knownbad.go`, `fak/internal/foo/bar.go`.",
            ["cmd/fak/knownbad.go", "internal/foo/bar.go", "docs/clean.md"])
        self.assertTrue(scan["collides"])
        self.assertEqual(scan["dirty_paths"],
                         ["cmd/fak/knownbad.go", "internal/foo/bar.go"])

    def test_bare_filename_does_not_collide(self) -> None:
        mod = load()
        scan = mod.dirty_path_collision(
            "Knownbad work mentions knownbad.go but no repo-relative path.",
            ["cmd/fak/knownbad.go"])
        self.assertFalse(scan["collides"])

    def test_git_status_parser_handles_renames_and_untracked(self) -> None:
        mod = load()
        paths = mod.parse_git_status_paths(
            " M cmd/fak/knownbad.go\n"
            "?? internal/knownbad/knownbad_test.go\n"
            "R  old/path.go -> new/path.go\n")
        self.assertEqual(paths, [
            "cmd/fak/knownbad.go",
            "internal/knownbad/knownbad_test.go",
            "old/path.go",
            "new/path.go",
        ])


class SameIssueWipCollisionTest(unittest.TestCase):
    def test_recent_same_issue_log_with_dirty_path_collides(self) -> None:
        import os
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = runs / "resolve-2718-20260705-191407.log"
            log.write_text(
                "Final report: W6 implementation left as working-tree changes.\n"
                "Uncommitted files:\n"
                "- cmd/fak/knownbad.go\n",
                encoding="utf-8",
            )
            os.utime(log, (now - 60, now - 60))

            scan = mod.same_issue_wip_collision(
                runs, 2718, ["cmd/fak/knownbad.go", "docs/other.md"],
                now_ts=now)

        self.assertTrue(scan["collides"])
        self.assertEqual(scan["dirty_paths"], ["cmd/fak/knownbad.go"])
        self.assertEqual(scan["evidence"][0]["log"],
                         "resolve-2718-20260705-191407.log")

    def test_worktree_wording_collides(self) -> None:
        import os
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = runs / "resolve-2718-20260705-191407.log"
            log.write_text(
                "Final report: implementation left as local worktree changes.\n"
                "- cmd/fak/knownbad.go\n",
                encoding="utf-8",
            )
            os.utime(log, (now - 60, now - 60))

            scan = mod.same_issue_wip_collision(
                runs, 2718, ["cmd/fak/knownbad.go"], now_ts=now)

        self.assertTrue(scan["collides"])
        self.assertEqual(scan["dirty_paths"], ["cmd/fak/knownbad.go"])

    def test_ignores_other_issue_and_stale_log(self) -> None:
        import os
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            other = runs / "resolve-9999-20260705-191407.log"
            other.write_text("Uncommitted: cmd/fak/knownbad.go\n", encoding="utf-8")
            stale = runs / "resolve-2718-20260701-191407.log"
            stale.write_text("Uncommitted: cmd/fak/knownbad.go\n", encoding="utf-8")
            os.utime(other, (now - 60, now - 60))
            os.utime(stale, (now - 3 * 24 * 3600, now - 3 * 24 * 3600))

            scan = mod.same_issue_wip_collision(
                runs, 2718, ["cmd/fak/knownbad.go"], now_ts=now)

        self.assertFalse(scan["collides"])

    def test_no_commit_witness_sidecar_is_wip_evidence(self) -> None:
        import json
        import os
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = runs / "resolve-2718-20260705-191407.log"
            log.write_text("Changed cmd/fak/knownbad.go\n", encoding="utf-8")
            log.with_suffix(mod.WITNESS_SIDECAR_SUFFIX).write_text(
                json.dumps({"claim": "CLAIM_NO_COMMIT"}), encoding="utf-8")
            os.utime(log, (now - 60, now - 60))

            scan = mod.same_issue_wip_collision(
                runs, 2718, ["cmd/fak/knownbad.go"], now_ts=now)
            reason = mod.same_issue_wip_reason(2718, scan)

        self.assertTrue(scan["collides"])
        self.assertEqual(scan["evidence"][0]["witness_claim"], "CLAIM_NO_COMMIT")
        self.assertIn("CLAIM_NO_COMMIT", reason)


class SameIssueWipOrientationDocTest(unittest.TestCase):
    """#4321 -- an orientation doc named in a resolve log is not same-issue WIP.

    The SAME_ISSUE_WIP scan infers "the prior resolver left this path dirty" from the
    path being MENTIONED in that resolver's log. Every worker prompt orders the worker
    to read the repo's orientation docs by name, so those names land in every log
    regardless of what the worker touched -- and those same files are chronically
    dirty in the one shared trunk checkout. The ledger investigation behind #4321
    measured ~175/180 SAME_ISSUE_WIP hits on AGENTS.md alone; this repo's own
    .dispatch-runs/collision-holds.jsonl shows the identical defect under different
    names (README.md 78 / llms.txt 63 / INDEX.md 42 = 183 of 208 same_issue_wip
    path-hits, vs 25 on a real code path). The concentration is therefore NOT a
    property of AGENTS.md, and the issue's proposed fix of splitting AGENTS.md into
    fragments would have relocated the magnet rather than removed it.
    """

    def _log(self, runs, now, body):
        import os
        log = runs / "resolve-2718-20260705-191407.log"
        log.write_text(body, encoding="utf-8")
        os.utime(log, (now - 60, now - 60))
        return log

    def test_orientation_doc_alone_does_not_collide(self) -> None:
        """A log that names ONLY orientation docs no longer refuses the issue."""
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # The backticked form the real worker prompt uses ("orient with
            # `AGENTS.md`", "the doc map `llms.txt`"). Note text_mentions_repo_path's
            # `(?![\\w./-])` lookahead means a name ending a sentence ("AGENTS.md.")
            # does NOT match -- the fixture mirrors the phrasing that does.
            self._log(runs, now,
                      "Read `AGENTS.md` first, then the doc map `llms.txt`, "
                      "plus `README.md`\n"
                      "Final report: left as uncommitted working-tree changes.\n")
            scan = mod.same_issue_wip_collision(
                runs, 2718, ["AGENTS.md", "llms.txt", "README.md"], now_ts=now)

        self.assertFalse(scan["collides"])
        self.assertEqual(scan["dirty_paths"], [])
        # The suppression is auditable, not silent.
        self.assertEqual(sorted(scan["orientation_paths"]),
                         ["AGENTS.md", "README.md", "llms.txt"])

    def test_real_wip_path_still_collides_alongside_orientation_docs(self) -> None:
        """The guard is narrowed, not disabled: a genuine dirty code path still
        refuses, and the orientation doc is dropped from the refusal's path list."""
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._log(runs, now,
                      "Oriented with `AGENTS.md`\n"
                      "Final report: cmd/fak/knownbad.go left uncommitted\n")
            scan = mod.same_issue_wip_collision(
                runs, 2718, ["AGENTS.md", "cmd/fak/knownbad.go"], now_ts=now)

        self.assertTrue(scan["collides"])
        self.assertEqual(scan["dirty_paths"], ["cmd/fak/knownbad.go"])
        self.assertEqual(scan["orientation_paths"], ["AGENTS.md"])

    def test_dirty_path_guard_still_protects_a_real_agents_md_issue(self) -> None:
        """The narrowing is scoped to the log-mention INFERENCE. An issue whose own
        BODY names AGENTS.md as its work site is still refused by the dirty-path
        guard, so a real AGENTS.md edit cannot land on a peer's uncommitted one."""
        mod = load()
        scan = mod.dirty_path_collision(
            "fix(docs): tighten the commit rule in `AGENTS.md`", ["AGENTS.md"])
        self.assertTrue(scan["collides"])
        self.assertEqual(scan["dirty_paths"], ["AGENTS.md"])


class RefusalClassTest(unittest.TestCase):
    """#4321 -- lane-lease and working-tree co-tenancy are separate refusal classes.

    Both mechanisms wear the word "collision" in this launcher (the lane-lease
    refusal's own reason text says "refusing COLLISION_RISK"), which is how ~369
    working-tree refusals got folded in with lease contention during ledger analysis.
    They share no fix: a lane-lease refusal clears itself when the holding peer
    finishes, while a working-tree refusal clears only on a human/peer commit or
    revert. Carrying the class as a FIELD is what stops the two being conflated by
    verdict-name pattern matching.
    """

    def test_classes_are_distinct_and_table_driven(self) -> None:
        mod = load()
        self.assertEqual(mod.refusal_class("LANE_LEASE_HELD"),
                         mod.REFUSAL_CLASS_LANE_LEASE)
        self.assertEqual(mod.refusal_class("SAME_ISSUE_WIP"),
                         mod.REFUSAL_CLASS_WORKTREE_COTENANCY)
        self.assertEqual(mod.refusal_class("DIRTY_PATH_COLLISION"),
                         mod.REFUSAL_CLASS_WORKTREE_COTENANCY)
        self.assertNotEqual(mod.REFUSAL_CLASS_LANE_LEASE,
                            mod.REFUSAL_CLASS_WORKTREE_COTENANCY)
        # Unclassified verdicts stay "" -- a new refusal code must be classified on
        # purpose rather than inherit a class from a substring of its name.
        self.assertEqual(mod.refusal_class("MULTI_LANE_SCOPE"), "")
        self.assertEqual(mod.refusal_class("SOME_NEW_COLLISION"), "")
        self.assertEqual(mod.refusal_class(""), "")

    def test_launch_gate_blocker_carries_the_class(self) -> None:
        mod = load()
        gate = mod.launch_gate_blocked("SAME_ISSUE_WIP", "why", "next")
        self.assertEqual(gate["blockers"][0]["refusal_class"],
                         mod.REFUSAL_CLASS_WORKTREE_COTENANCY)
        # An unclassified hold is unchanged -- no empty field is invented for it.
        plain = mod.launch_gate_blocked("FORCE_REASON_REQUIRED", "why", "next")
        self.assertNotIn("refusal_class", plain["blockers"][0])

    def test_collision_ledger_rows_carry_the_class(self) -> None:
        """Rows on the collision ledger are co-tenancy by construction -- both when
        freshly written and when read back from a pre-#4321 row with no class."""
        import json
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td)
            mod.record_collision_holds(runs, [{
                "issue": 4321, "kind": "same_issue_wip", "lane": "tools",
                "title": "t", "reason": "r", "dirty_paths": ["AGENTS.md"],
            }], live=True, now_ts=1_000_000.0)
            ledger = runs / "collision-holds.jsonl"
            written = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(written["refusal_class"],
                             mod.REFUSAL_CLASS_WORKTREE_COTENANCY)
            # A legacy row with no class reads back as co-tenancy, not as "".
            with ledger.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": 1_000_000.0, "issue": 4322,
                                    "kind": "dirty_path"}) + "\n")
            recs = {r["issue"]: r for r in mod.collision_held_records(
                runs, ttl_h=3, now_ts=1_000_100.0)}
            self.assertEqual(recs[4322]["refusal_class"],
                             mod.REFUSAL_CLASS_WORKTREE_COTENANCY)

    def test_loop_ledger_evidence_carries_the_split(self) -> None:
        """#4321 -- the loops.jsonl evidence pairs make the WIP-vs-lease split
        computable as a FIELD. Lane/tree alone cannot: both refusals emit the SAME
        lane and tree, which is why the two were conflated in the first place."""
        mod = load()

        def ev(payload):
            return dict(mod._dispatch_collision_evidence(ROOT, payload))

        wip = ev({"verdict": "SAME_ISSUE_WIP", "lane": "tools"})
        dirty = ev({"verdict": "DIRTY_PATH_COLLISION", "lane": "tools"})
        lease = ev({"verdict": "LANE_LEASE_HELD", "lane": "tools"})

        self.assertEqual(wip["refusal_class"], mod.REFUSAL_CLASS_WORKTREE_COTENANCY)
        self.assertEqual(dirty["refusal_class"], mod.REFUSAL_CLASS_WORKTREE_COTENANCY)
        self.assertEqual(lease["refusal_class"], mod.REFUSAL_CLASS_LANE_LEASE)
        # Same lane on both sides -- the class is the ONLY separator here.
        self.assertEqual(wip["lane"], lease["lane"])
        self.assertNotEqual(wip["refusal_class"], lease["refusal_class"])
        # A non-contention tick is unchanged (additive, no empty field invented).
        self.assertNotIn("refusal_class",
                         ev({"verdict": "WOULD_SPAWN", "lane": "tools"}))
        # The class survives a refusal that never resolved a lane.
        self.assertEqual(ev({"verdict": "SAME_ISSUE_WIP"})["refusal_class"],
                         mod.REFUSAL_CLASS_WORKTREE_COTENANCY)


class CandidatePriorityTest(unittest.TestCase):
    """candidate_priority maps priority/P* labels to the dispatchorder Candidate.priority
    integer (P0>P1>P2, 0 when unlabeled) -- #3222 DoD item 5."""

    def test_label_to_weight(self) -> None:
        mod = load()
        # gh's {"name": ...} dict shape, heaviest label wins.
        self.assertEqual(mod.candidate_priority([{"name": "priority/P0"}]), 1000)
        self.assertEqual(mod.candidate_priority([{"name": "priority/P1"}]), 400)
        self.assertEqual(mod.candidate_priority([{"name": "priority/P2"}]), 150)
        self.assertEqual(
            mod.candidate_priority([{"name": "priority/P2"}, {"name": "priority/P0"}]),
            1000)
        # Plain-string labels are accepted too.
        self.assertEqual(mod.candidate_priority(["priority/P1"]), 400)

    def test_unlabeled_is_zero(self) -> None:
        mod = load()
        # No priority label, an unrelated label, and empty/None all map to 0 (the
        # additive-no-regression floor: an unlabeled candidate orders by recency).
        self.assertEqual(mod.candidate_priority([{"name": "enhancement"}]), 0)
        self.assertEqual(mod.candidate_priority([]), 0)
        self.assertEqual(mod.candidate_priority(None), 0)


class CandidateBlockedByTest(unittest.TestCase):
    """candidate_blocked_by parses "depends-on:/blocked-by: #N" markers from an issue body into the
    dispatchorder Candidate.blocked_by list (empty when the body names no prerequisite) -- #3224."""

    def test_marker_forms(self) -> None:
        mod = load()
        # Both marker verbs, hyphenated or spaced, colon optional.
        self.assertEqual(mod.candidate_blocked_by("depends-on: #120"), ["120"])
        self.assertEqual(mod.candidate_blocked_by("Depends on #120"), ["120"])
        self.assertEqual(mod.candidate_blocked_by("blocked-by: #120"), ["120"])
        self.assertEqual(mod.candidate_blocked_by("Blocked by #120"), ["120"])

    def test_multiple_refs_and_dedup(self) -> None:
        mod = load()
        # Comma/and/&-separated refs on one marker, and across markers, deduped in first-seen order.
        self.assertEqual(
            mod.candidate_blocked_by("blocked-by: #120, #121 and #122"),
            ["120", "121", "122"])
        self.assertEqual(
            mod.candidate_blocked_by("Depends on #7.\nAlso blocked by #7 & #9."),
            ["7", "9"])

    def test_no_marker_is_empty(self) -> None:
        mod = load()
        # Prose that merely contains the words never matches (the marker must be followed by #N);
        # a body-free / marker-free issue carries no prerequisite edge (additive-no-regression).
        self.assertEqual(mod.candidate_blocked_by("it depends on the weather"), [])
        self.assertEqual(mod.candidate_blocked_by("no markers here #notanumber"), [])
        self.assertEqual(mod.candidate_blocked_by(""), [])
        self.assertEqual(mod.candidate_blocked_by(None), [])


class SeatAdaptiveTargetTest(unittest.TestCase):
    """seat_adaptive_target (#3246): the pure fold that sizes one tick's effective
    cap from the preflight's seat/host signal. All terms are TOTAL populations."""

    def _pre(self, *, live=1, seat_free=11, host_cap=32, seat_total=18):
        return {"verdict": "SPAWN_OK", "live": live, "host_cap": host_cap,
                "seat": {"total": seat_total, "free": seat_free,
                         "leased": seat_total - seat_free}}

    def test_free_seats_bind_when_ramp_disabled(self) -> None:
        mod = load()
        target, info = mod.seat_adaptive_target(
            self._pre(live=1, seat_free=11, host_cap=32),
            fallback=3, ceiling=20, ramp_delta=0)
        self.assertEqual(target, 12)  # live 1 + 11 free seats < host 32 < ceiling 20? no: 12 < 20
        self.assertEqual(info["binding"], "seat_free")
        self.assertTrue(info["signal_available"])

    def test_ramp_delta_bounds_per_tick_growth(self) -> None:
        # The issue's canary-safety knob: 11 free seats never jump the cap in one
        # tick; it lifts at most ramp_delta above the live count.
        mod = load()
        target, info = mod.seat_adaptive_target(
            self._pre(live=1, seat_free=11, host_cap=32),
            fallback=3, ceiling=20, ramp_delta=2)
        self.assertEqual(target, 3)
        self.assertEqual(info["binding"], "ramp_delta")

    def test_host_cap_binds_below_seats(self) -> None:
        mod = load()
        target, info = mod.seat_adaptive_target(
            self._pre(live=10, seat_free=30, host_cap=12),
            fallback=3, ceiling=20, ramp_delta=0)
        self.assertEqual(target, 12)
        self.assertEqual(info["binding"], "host_cap")

    def test_hard_ceiling_binds_last(self) -> None:
        mod = load()
        target, info = mod.seat_adaptive_target(
            self._pre(live=10, seat_free=30, host_cap=64),
            fallback=3, ceiling=20, ramp_delta=0)
        self.assertEqual(target, 20)
        self.assertEqual(info["binding"], "hard_ceiling")

    def test_explicit_max_workers_above_ceiling_raises_it(self) -> None:
        # Seat sizing never SHRINKS an operator's explicit cap: --max-workers 50
        # lifts the effective hard ceiling to 50.
        mod = load()
        target, info = mod.seat_adaptive_target(
            self._pre(live=10, seat_free=30, host_cap=64),
            fallback=50, ceiling=20, ramp_delta=0)
        self.assertEqual(info["hard_ceiling"], 50)
        self.assertEqual(target, 40)  # live 10 + 30 free seats
        self.assertEqual(info["binding"], "seat_free")

    def test_depleted_seats_pin_cap_at_live(self) -> None:
        # 0 free seats => target == live => the reprobe refuses AT_CAP: honest
        # backpressure even when the configured fallback would have admitted more.
        mod = load()
        target, info = mod.seat_adaptive_target(
            self._pre(live=3, seat_free=0, host_cap=32),
            fallback=5, ceiling=20, ramp_delta=2)
        self.assertEqual(target, 3)
        self.assertEqual(info["binding"], "seat_free")

    def test_no_seat_signal_falls_back_to_configured_cap(self) -> None:
        # FAIL-OPEN: a preflight doc predating the seat pool (or a hermetic stub)
        # sizes exactly as before -- the fixed --max-workers stays the cap.
        mod = load()
        target, info = mod.seat_adaptive_target(
            {"verdict": "SPAWN_OK", "cap": 2, "live": 0},  # no seat block
            fallback=3, ceiling=20, ramp_delta=2)
        self.assertEqual(target, 3)
        self.assertFalse(info["signal_available"])
        self.assertEqual(info["binding"], "fallback_max_workers")

    def test_signal_read_from_capacity_limiter_raw(self) -> None:
        # The signal also folds out of capacity_limiter.raw when the top-level
        # seat/live keys are absent (older doc shapes).
        mod = load()
        pre = {"verdict": "SPAWN_OK",
               "capacity_limiter": {"raw": {"live": 2, "seat_free": 4,
                                            "host_cap": 16}}}
        target, info = mod.seat_adaptive_target(pre, fallback=3, ceiling=20,
                                                ramp_delta=0)
        self.assertEqual(target, 6)
        self.assertEqual(info["binding"], "seat_free")


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
        EvaluateTest._patch(self, mod, pre={}, pick=dict(self.PICK))
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


class BackendHealthSpawnGateTest(unittest.TestCase):
    """gate_spawn_on_health / recent_backend_stub_rate (#3247): the backend-health spawn
    gate pins a majority-stub backend to 0 spawns before seat sizing, fails open on an
    unknown rate, and auto-restores because the live rate is re-read every tick."""

    def test_majority_stub_gates_to_zero(self) -> None:
        mod = load()
        planned, reason = mod.gate_spawn_on_health(4, 0.75)
        self.assertEqual(planned, 0)
        self.assertIsNotNone(reason)
        self.assertIn("majority-stub", reason)

    def test_threshold_is_inclusive(self) -> None:
        mod = load()
        planned, reason = mod.gate_spawn_on_health(4, mod._HEALTH_SKIP_STUB_RATE)
        self.assertEqual(planned, 0)
        self.assertIsNotNone(reason)

    def test_healthy_backend_passes_through_unchanged(self) -> None:
        mod = load()
        planned, reason = mod.gate_spawn_on_health(4, 0.25)
        self.assertEqual(planned, 4)
        self.assertIsNone(reason)

    def test_none_rate_fails_open(self) -> None:
        mod = load()
        self.assertEqual(mod.gate_spawn_on_health(4, None), (4, None))

    def test_non_numeric_rate_fails_open(self) -> None:
        mod = load()
        self.assertEqual(mod.gate_spawn_on_health(4, "bogus"), (4, None))

    def test_recent_rate_reads_matching_product(self) -> None:
        mod = load()
        import dispatch_status
        orig = dispatch_status.backend_stub_rates
        try:
            dispatch_status.backend_stub_rates = lambda *a, **k: [
                {"product": "gpt-5-codex", "stub_rate": 0.8},
                {"product": "other", "stub_rate": 0.1}]
            self.assertEqual(
                mod.recent_backend_stub_rate(Path("."), product="gpt-5-codex"), 0.8)
            self.assertIsNone(
                mod.recent_backend_stub_rate(Path("."), product="absent"))
        finally:
            dispatch_status.backend_stub_rates = orig

    def test_recent_rate_fails_open_on_error(self) -> None:
        mod = load()
        import dispatch_status
        orig = dispatch_status.backend_stub_rates

        def boom(*a, **k):
            raise RuntimeError("logs unreadable")

        try:
            dispatch_status.backend_stub_rates = boom
            self.assertIsNone(
                mod.recent_backend_stub_rate(Path("."), product="gpt-5-codex"))
        finally:
            dispatch_status.backend_stub_rates = orig



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
        EvaluateTest._patch(self, mod, pre=pre, pick=pick)
        mod.account_probe.probe_opencode_account = lambda _row: {
            "status": "GATEWAY_DOWN", "block_reason": "connection refused"}

        payload = mod.evaluate(ROOT, max_workers=3, work_kind="gardening",
                               lane=None, live=False, backend="opencode")

        self.assertEqual(payload["action"], "backend_health_skip")
        self.assertEqual(payload["verdict"], "GATEWAY_DOWN")
        self.assertEqual(payload["gateway_health_gate"], {
            "status": "backend_unhealthy", "reason": "gateway_down", "planned": 0})
        self.assertIn("connection refused", payload["reason"])

class SpawnFailedCauseTest(unittest.TestCase):
    """classify_spawn_failed_cause pins each cause bucket on a sample worker-log
    tail — the ~4% SPAWN_FAILED noise made attributable, never a false blame (#2635)."""

    HEADER = "# fak-spawn 20260710-010203 issue=1515 lane=cmd backend=claude argv0=claude.exe\n"

    def test_weekly_limit_banner(self) -> None:
        mod = load()
        tail = self.HEADER + "You've hit your weekly limit · resets 4am (America/Los_Angeles)\n"
        self.assertEqual(mod.classify_spawn_failed_cause({"tail": tail}),
                         mod.SPAWN_CAUSE_WEEKLY_LIMIT)

    def test_glm_wall_is_weekly_limit(self) -> None:
        mod = load()
        tail = self.HEADER + "Limit Exhausted — usage limit reached\n"
        self.assertEqual(mod.classify_spawn_failed_cause({"tail": tail}),
                         mod.SPAWN_CAUSE_WEEKLY_LIMIT)

    def test_stale_cred_auth_gap(self) -> None:
        mod = load()
        tail = self.HEADER + "Missing environment variable: `ANTHROPIC_API_KEY`\n"
        self.assertEqual(mod.classify_spawn_failed_cause({"tail": tail}),
                         mod.SPAWN_CAUSE_STALE_CRED)
        login = self.HEADER + "no ChatGPT subscription auth.json was resolved. Run `codex login`\n"
        self.assertEqual(mod.classify_spawn_failed_cause({"tail": login}),
                         mod.SPAWN_CAUSE_STALE_CRED)

    def test_child_crash_traceback(self) -> None:
        mod = load()
        tail = self.HEADER + "Traceback (most recent call last):\n  File ...\nRuntimeError: boom\n"
        self.assertEqual(mod.classify_spawn_failed_cause({"tail": tail}),
                         mod.SPAWN_CAUSE_CHILD_CRASH)
        oom = self.HEADER + "fatal error: runtime: out of memory\n"
        self.assertEqual(mod.classify_spawn_failed_cause({"tail": oom}),
                         mod.SPAWN_CAUSE_CHILD_CRASH)

    def test_exec_race_header_only_and_empty(self) -> None:
        mod = load()
        self.assertEqual(mod.classify_spawn_failed_cause({"tail": self.HEADER}),
                         mod.SPAWN_CAUSE_EXEC_RACE)
        self.assertEqual(
            mod.classify_spawn_failed_cause({"tail": "", "silent": True, "log_bytes": 0}),
            mod.SPAWN_CAUSE_EXEC_RACE)

    def test_unknown_unrecognized_child_output(self) -> None:
        mod = load()
        tail = self.HEADER + "some novel shutdown line we do not recognize yet\n"
        self.assertEqual(mod.classify_spawn_failed_cause({"tail": tail}),
                         mod.SPAWN_CAUSE_UNKNOWN)

    def test_precedence_quota_beats_crash(self) -> None:
        # A cap banner that also mentions a fatal error is still a weekly_limit,
        # not a child_crash — most-specific (quota) wins.
        mod = load()
        tail = self.HEADER + "HTTP 429 rate-limited\nfatal error: aborted\n"
        self.assertEqual(mod.classify_spawn_failed_cause({"tail": tail}),
                         mod.SPAWN_CAUSE_WEEKLY_LIMIT)


class OpenWitnessedDispositionTest(unittest.TestCase):
    """#5071: the pre-dispatch OPEN_WITNESSED guard against a REAL temporary git
    history — the #2850 fixture (open issue + resolving commit already in trunk
    ancestry) plus a concrete non-witnessed control that remains eligible."""

    @staticmethod
    def _git(repo: Path, *args: str) -> None:
        base = ["git", "-C", str(repo), "-c", "user.email=t@t",
                "-c", "user.name=t", "-c", "core.hooksPath=",
                "-c", "commit.gpgsign=false"]
        subprocess.run(base + list(args), check=True, capture_output=True, text=True,
                       creationflags=_no_window_creationflags())

    @staticmethod
    def _head_sha(repo: Path) -> str:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True,
                             creationflags=_no_window_creationflags())
        return out.stdout.strip()

    def _repo_with_resolving_commit(self, repo: Path) -> str:
        """One trunk commit whose subject cites #2850 (the resolving grammar)."""
        self._git(repo, "init", "-q", "-b", "main")
        (repo / "a.go").write_text("package a\n", encoding="utf-8")
        self._git(repo, "add", "a.go")
        self._git(repo, "commit", "-q", "--no-gpg-sign", "-m",
                  "fix(dispatch): close pre-witnessed guard (#2850) (fak tools)")
        return self._head_sha(repo)

    def test_witnessed_trunk_commit_disposes_candidate_control_stays(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            sha = self._repo_with_resolving_commit(repo)
            audits = {sha: {"sha": sha, "verdict": "OK",
                            "witness": "diff-witnessed",
                            "claim_kind": "code_effect"}}
            rows = mod.open_witnessed_dispositions(
                repo, {2850, 2293},
                audit_runner=lambda root, s, **k: audits.get(s, {}))
            # #2850 is disposed with its witnessing SHA...
            self.assertEqual([r["issue"] for r in rows], [2850])
            self.assertEqual(rows[0]["sha"], sha)
            self.assertEqual(rows[0]["code"], "OPEN_WITNESSED")
            self.assertEqual(rows[0]["witness"], "diff-witnessed")
            # ...and the control (#2293, no resolving commit anywhere) is not.

    def test_unwitnessed_resolving_cite_is_not_shipped(self) -> None:
        """A mere subject mention without the diff-witness keep-bit must not
        dispose the candidate (preserve the existing witness rules)."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            sha = self._repo_with_resolving_commit(repo)
            audits = {sha: {"sha": sha, "verdict": "ABSTAIN", "witness": "abstain"}}
            rows = mod.open_witnessed_dispositions(
                repo, {2850},
                audit_runner=lambda root, s, **k: audits.get(s, {}))
            self.assertEqual(rows, [])

    def test_doc_claim_does_not_dispose_issue(self) -> None:
        """A diff-witnessed doc/triage claim witnesses a note, never the
        candidate's feature — the issue stays eligible (#2998 rule shared)."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            sha = self._repo_with_resolving_commit(repo)
            audits = {sha: {"sha": sha, "verdict": "OK",
                            "witness": "diff-witnessed", "claim_kind": "doc"}}
            rows = mod.open_witnessed_dispositions(
                repo, {2850},
                audit_runner=lambda root, s, **k: audits.get(s, {}))
            self.assertEqual(rows, [])

    def test_git_error_fails_open(self) -> None:
        mod = load()
        rows = mod.open_witnessed_dispositions(
            Path("z:/nope"), {2850},
            git=lambda root, args, **k: (1, ""),
            audit_runner=lambda root, s, **k: {})
        self.assertEqual(rows, [])


class CloseOpenWitnessedTest(unittest.TestCase):
    """#5071: dispositions transition through the EXISTING trusted close path
    (issue_resolve_witnessed.py via a synthetic OPEN_WITNESSED --audit-json)."""

    ROW = {"issue": 2850, "sha": "f8aff29dfd", "code": "OPEN_WITNESSED",
           "subject": "fix(dispatch): guard tick (#2850) (fak tools)",
           "verdict": "OK", "witness": "diff-witnessed",
           "claim_kind": "code_effect"}

    def test_routes_through_close_arm_with_synthetic_audit(self) -> None:
        mod = load()
        calls: list[list[str]] = []

        def fake_run(root, cmd, **_kw):
            calls.append(list(cmd))
            return 0, json.dumps({"ok": True, "verdict": "CLOSED",
                                  "counts": {"closed": 1},
                                  "closed_numbers": [2850]})

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d) / "runs"
            out = mod.close_open_witnessed(Path(d), runs, [dict(self.ROW)],
                                           live=True, runner=fake_run)
            self.assertTrue(out["invoked"])
            self.assertTrue(out["ok"])
            self.assertEqual(out["verdict"], "CLOSED")
            self.assertEqual(out["closed_numbers"], [2850])
            cmd = calls[0]
            self.assertIn("--audit-json", cmd)
            self.assertIn("--live", cmd)
            self.assertTrue(str(cmd[1]).endswith("issue_resolve_witnessed.py"))
            # The synthetic audit carries the OPEN_WITNESSED bucket + witness.
            audit_path = Path(cmd[cmd.index("--audit-json") + 1])
            doc = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(doc["issues"][0]["number"], 2850)
            self.assertEqual(doc["issues"][0]["bucket"], "OPEN_WITNESSED")
            self.assertEqual(doc["issues"][0]["witnessed_commits"][0]["sha"],
                             "f8aff29dfd")

    def test_dry_run_never_passes_live(self) -> None:
        mod = load()
        calls: list[list[str]] = []

        def fake_run(root, cmd, **_kw):
            calls.append(list(cmd))
            return 0, json.dumps({"ok": True, "verdict": "PLANNED"})

        with tempfile.TemporaryDirectory() as d:
            out = mod.close_open_witnessed(Path(d), Path(d) / "runs",
                                           [dict(self.ROW)],
                                           live=False, runner=fake_run)
            self.assertTrue(out["invoked"])
            self.assertNotIn("--live", calls[0])

    def test_close_arm_failure_fails_open(self) -> None:
        mod = load()

        def fake_run(root, cmd, **_kw):
            return 127, "boom: not json"

        with tempfile.TemporaryDirectory() as d:
            out = mod.close_open_witnessed(Path(d), Path(d) / "runs",
                                           [dict(self.ROW)],
                                           live=True, runner=fake_run)
            self.assertTrue(out["invoked"])
            self.assertFalse(out["ok"])
            self.assertIn("boom", out["error"])

    def test_no_rows_is_a_noop(self) -> None:
        mod = load()
        self.assertEqual(
            mod.close_open_witnessed(Path("."), Path("."), [], live=True,
                                     runner=lambda *a, **k: (0, "{}")),
            {})


class SpawnIssueWorkerWorktreeTest(unittest.TestCase):
    """#3181: spawn_issue_worker edits in a per-worker worktree pinned at base_sha
    when the flag is on, fails open to the shared-trunk cwd when off."""

    def _fake_git(self):
        def git(root, args):
            if args and args[0] == "rev-parse":
                return 0, "deadbeef\n"
            return 0, ""
        return git

    def _spawn(self, env_overrides, td):
        m = load()
        seen: dict[str, object] = {}

        class FakeProc:
            pid = 5150

        def fake_popen(argv, cwd=None, env=None, **kw):
            seen["cwd"] = cwd
            return FakeProc()

        runs = Path(td) / "runs"
        with mock.patch.object(m.subprocess, "Popen", fake_popen), \
             mock.patch.dict(os.environ, env_overrides, clear=False):
            res = m.spawn_issue_worker(
                ["python", "-c", "pass"], {"PATH": os.environ.get("PATH", "")},
                Path(td), runs, 3181, "tools", "claude",
                lease={"acquired": True, "id": "L1", "holder": "h",
                       "tree": ["tools/**"]},
                base_sha="basecafe", worktree_git=self._fake_git())
        return m, seen, res, runs

    def test_worktree_spawn_off_uses_root_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _, seen, res, runs = self._spawn({"FLEET_WORKER_WORKTREE": "0"}, td)
            self.assertEqual(seen["cwd"], str(Path(td)))
            self.assertNotIn("worktree", res)
            self.assertEqual(list(runs.glob("*.worktree")), [])  # off -> no sidecar

    def test_worktree_spawn_on_uses_worktree_cwd_pinned_at_base_sha(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m, seen, res, runs = self._spawn(
                {"FLEET_WORKER_WORKTREE": "1",
                 "FLEET_WORKER_WORKTREE_ROOT": str(Path(td) / "wt")}, td)
            self.assertNotEqual(seen["cwd"], str(Path(td)))
            self.assertTrue(m.worker_worktree.is_worker_worktree(str(seen["cwd"])))
            self.assertEqual(res.get("worktree"), seen["cwd"])
            # Sidecar contract the shared witness sweep consumes: a PLAIN worktree path,
            # the pinned base, and the lease tree for a scoped land.
            wt_side = list(runs.glob("*.worktree"))
            self.assertEqual(len(wt_side), 1)
            self.assertEqual(wt_side[0].read_text(encoding="utf-8"), seen["cwd"])
            base_side = list(runs.glob("*.basesha"))
            self.assertEqual(base_side[0].read_text(encoding="utf-8"), "basecafe")
            tree_side = list(runs.glob("*.lease-tree.json"))
            self.assertEqual(json.loads(tree_side[0].read_text(encoding="utf-8")),
                             ["tools/**"])


class UnresolvedAcceptanceGateTest(unittest.TestCase):
    """#5070: a contract whose ``## Acceptance gate`` explicitly needs operator
    input is held BEFORE worker spawn, even when its aggregate score clears the
    floor. Drives the real admission seam (issue_contract_review ->
    issue_contract_hold_reason) with only the `fak issue contract` shell-out
    stubbed, so the gate is the thing under test, not a mock of it."""

    #: The gate #2806 carried when the live resolver wrongly admitted it.
    UNRESOLVED_GATE = "unknown -- needs operator input"
    #: #5070's own gate: concrete, and it happens to contain the word
    #: "unresolved" -- the confusion risk a naive substring match would trip on.
    CONCRETE_GATE = ("`tools/issue_resolve_dispatch_test.py` regression covering "
                     "unresolved and concrete acceptance-gate controls.")

    @staticmethod
    def _body(gate: str, *, trailer: str = "") -> str:
        return (f"## Working spine\nHold unexecutable contracts.\n\n"
                f"## Acceptance gate\n{gate}\n\n"
                f"## Lane\ntools\n{trailer}")

    @staticmethod
    def _passing_runner(*_a, **_k):
        """A `fak issue contract` run that PASSES the numeric floor, so the only
        thing that can hold the contract is the acceptance-gate check."""
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps({
                "ok": True,
                "reviews": [{"ok": True, "reasons": [], "missing_fields": [],
                             "score": {"total": 100},
                             "spine_priority": {"total": 100}}],
            }), stderr="")

    def _review(self, body: str) -> dict:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            return mod.issue_contract_review(
                Path(td), {"number": 2806, "title": "t", "body": body}, 2806,
                runner=self._passing_runner)

    @staticmethod
    def _holds(mod, contract: dict) -> bool:
        """The module's own admission predicate, mirrored: this is exactly what
        the resolver's `_contract_holds` reads before it spawns or leases."""
        return bool(contract.get("unavailable") or not contract.get("ok") or
                    int(contract.get("score") or 0)
                    < mod.DEFAULT_ISSUE_CONTRACT_MIN_SCORE)

    def test_unresolved_gate_holds_a_score_passing_contract(self) -> None:
        mod = load()
        contract = self._review(self._body(self.UNRESOLVED_GATE))
        # The score cleared the floor -- the hold is the gate's doing alone.
        self.assertEqual(contract["score"], mod.DEFAULT_ISSUE_CONTRACT_MIN_SCORE)
        self.assertFalse(contract["ok"])
        self.assertTrue(self._holds(mod, contract))
        reason = mod.issue_contract_hold_reason(contract)
        self.assertTrue(reason.startswith(mod.ACCEPTANCE_GATE_HOLD_REASON), reason)
        self.assertIn("needs operator input", reason)

    def test_concrete_gate_stays_dispatchable(self) -> None:
        mod = load()
        contract = self._review(self._body(self.CONCRETE_GATE))
        self.assertTrue(contract["ok"])
        self.assertFalse(self._holds(mod, contract))
        self.assertEqual(contract["acceptance_gate_hold"], "")
        self.assertEqual(mod.issue_contract_hold_reason(contract),
                         "issue contract passed")

    def test_operator_prose_outside_the_gate_never_holds(self) -> None:
        """The declared confusion risk: bind the Acceptance gate section only."""
        mod = load()
        body = self._body(
            self.CONCRETE_GATE,
            trailer="\n## Coordination\nPing the operator; needs operator input "
                    "before the milestone is set.\n")
        contract = self._review(body)
        self.assertTrue(contract["ok"])
        self.assertFalse(self._holds(mod, contract))

    def test_unresolved_vocabulary_forms(self) -> None:
        mod = load()
        for gate in ("unknown -- needs operator input", "unknown", "TBD",
                     "To be determined", "unresolved: awaiting operator decision",
                     "- unknown, pending operator sign-off", "???"):
            with self.subTest(gate=gate):
                self.assertTrue(mod.unresolved_acceptance_gate(self._body(gate)),
                                f"expected a hold for {gate!r}")

    def test_concrete_vocabulary_forms_stay_dispatchable(self) -> None:
        mod = load()
        for gate in (self.CONCRETE_GATE,
                     "`go test ./internal/dispatchtick` is green.",
                     "make ci green; the unknown-flake list is unchanged.",
                     "A render witness the operator reviewed already."):
            with self.subTest(gate=gate):
                self.assertEqual(mod.unresolved_acceptance_gate(self._body(gate)),
                                 "", f"unexpected hold for {gate!r}")

    def test_crlf_body_still_binds_the_gate(self) -> None:
        """A GitHub-authored body arrives CRLF; the gate must bind there too, or
        the check silently no-ops on exactly the live issues it exists for."""
        mod = load()
        crlf = self._body(self.UNRESOLVED_GATE).replace("\n", "\r\n")
        self.assertIn("\r\n", crlf)
        self.assertTrue(mod.unresolved_acceptance_gate(crlf))
        self.assertEqual(mod.unresolved_acceptance_gate(
            self._body(self.CONCRETE_GATE).replace("\n", "\r\n")), "")

    def test_no_acceptance_gate_section_is_not_a_gate_hold(self) -> None:
        """A body with no gate section is the score gate's business, not this
        check's -- it must stay additive-no-regression."""
        mod = load()
        self.assertEqual(mod.unresolved_acceptance_gate("## Lane\ntools\n"), "")
        self.assertEqual(mod.unresolved_acceptance_gate(None), "")


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
        EvaluateTest._patch(self, mod, pre={}, pick={})
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


class AppendLoopEventArgvTest(unittest.TestCase):
    """The argv `append_loop_event` hands to `fak loop append`.

    This wrapper had no coverage at all — every caller's test stubs `append=`, so the
    only place the argv is actually built was never exercised. It shipped a
    `--verified-state` flag the Go verb does not define, which made the flag parser exit
    2 before appending. Only the WITNESS event carries verified_state, so fire/admit/end
    landed and every witness row was silently dropped (the wrapper is fail-open), and
    `fak loop health` read the loop 0-of-N witnessed with witness_collapse=true.
    """

    def test_fak_loop_cmd_prefers_path_and_never_go_run(self) -> None:
        mod = load()
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(mod.shutil, "which", return_value=r"C:\bin\fak.exe"):
            self.assertEqual(mod.fak_loop_cmd(ROOT), [r"C:\bin\fak.exe"])
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(mod.shutil, "which", return_value=None):
            self.assertEqual(mod.fak_loop_cmd(Path(td)), [])

    def _argv(self, event: dict[str, object]) -> list[str]:
        mod = load()
        seen: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            seen.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with mock.patch.object(mod.subprocess, "run", fake_run):
            out = mod.append_loop_event(ROOT, Path("loops.jsonl"), event)
        self.assertTrue(out["ok"])
        self.assertEqual(len(seen), 1)
        return seen[0]

    def _loop_append_flags(self) -> set[str]:
        """The flag names `fak loop append` really defines, read from its flag set.

        Parsed from the Go source rather than shelled out so the test stays hermetic
        (this file runs nothing live). This is what pins the Python argv to Go reality:
        a flag Python invents but Go never declares is a silent exit-2, not an error a
        fail-open caller can see.
        """
        src = (ROOT / "cmd" / "fak" / "loop.go").read_text(encoding="utf-8")
        start = src.index("func runLoopAppend(")
        end = src.index("\nfunc ", start + 1)
        body = src[start:end]
        return set(re.findall(r'fs\.(?:String|Bool|Int|Int64|Var)\(\s*(?:&\w+,\s*)?"([a-z-]+)"', body))

    def test_verified_state_rides_evidence_not_an_undefined_flag(self) -> None:
        argv = self._argv({
            "loop_id": "issue-resolve-progress", "run_id": "RID-1", "kind": "witness",
            "status": "witnessed_done", "verified_state": "verified_done",
            "reason": "OK", "summary": "open_count=479",
            "evidence": [("progress_log", "runs/progress.jsonl")],
        })
        self.assertNotIn("--verified-state", argv)
        self.assertIn("--evidence", argv)
        self.assertIn("verified_state=verified_done", argv)
        # The pre-existing evidence is preserved, not replaced.
        self.assertIn("progress_log=runs/progress.jsonl", argv)
        # And the row is still a witness carrying its status.
        self.assertEqual(argv[argv.index("--kind") + 1], "witness")
        self.assertEqual(argv[argv.index("--status") + 1], "witnessed_done")

    def test_every_flag_is_one_fak_loop_append_defines(self) -> None:
        defined = self._loop_append_flags()
        self.assertIn("evidence", defined)      # the parse found a real flag set
        self.assertNotIn("verified-state", defined)
        for event in (
            {"loop_id": "l", "run_id": "R", "kind": "witness", "status": "witnessed_done",
             "verified_state": "verified_unavailable", "reason": "AUDIT_UNAVAILABLE",
             "summary": "s", "metrics": {"open_now": 479}, "evidence": [("progress_log", "p")]},
            {"loop_id": "l", "run_id": "R", "kind": "end", "status": "claimed_done",
             "reason": "OK", "summary": "s"},
        ):
            argv = self._argv(event)
            for tok in argv:
                if tok.startswith("--"):
                    self.assertIn(tok[2:], defined,
                                  f"{tok} is not a flag `fak loop append` defines; "
                                  f"it would exit 2 and drop the row (argv={argv})")

    def test_no_verified_state_appends_no_evidence_ref(self) -> None:
        argv = self._argv({"loop_id": "l", "run_id": "R", "kind": "end",
                           "status": "claimed_done", "reason": "OK", "summary": "s"})
        self.assertNotIn("--evidence", argv)


class ReblockStreakHoldTest(unittest.TestCase):
    """#5869: hold on REPETITION, not on the reason CLASS.

    Measured over the clean current-fleet window (2026-08-04 .. 08-07, 286 finished
    -worker .witness records / 94.0h over 117 issues), 24 issues hit a re-blockable
    terminal and drew 54 further worker-units / 16.6h — but 8 of those 54 (15%) went
    on to land a WITNESSED commit. So the #1396 docstring premise ("re-dispatching it
    re-blocks identically") is only ~30% true, and a blanket reason-class cooldown
    would refuse a retry that works one time in seven. These tests pin BOTH
    directions: the hold fires on a genuine consecutive-identical repeat, and it stays
    SILENT on the first retry that the measurement says still lands commits.
    """

    # 2026-08-06T12:00:00Z — the fixture epoch, so a stamped sidecar name and the
    # ``now_ts`` a tick would pass in are derived from the same instant.
    EPOCH = 1786363200.0

    @staticmethod
    def _stamp(ts: float) -> str:
        import datetime as dt
        return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime(
            "%Y%m%d-%H%M%S")

    def _witness(self, runs: Path, issue: int, ts: float, *,
                 claim: str = "CLAIM_NO_COMMIT", reason: str | None = None) -> Path:
        """Write one durable ``resolve-<issue>-<UTC>.witness`` record. Note there is
        no ``os.utime`` here and no ``.pid``/``.log`` sibling: the hold keys off the
        LAUNCH STAMP IN THE FILENAME plus the record body, which is exactly what
        survives ``prune_dead_sidecars``."""
        p = runs / f"resolve-{issue}-{self._stamp(ts)}.witness"
        body: dict = {"claim": claim, "issue": issue}
        if reason is not None:
            body["reason"] = reason
        p.write_text(json.dumps(body), encoding="utf-8")
        return p

    def test_first_retry_after_policy_block_is_admitted_second_repeat_is_held(self):
        """BOTH directions, the load-bearing test.

        One ``policy_block`` is NOT enough: that is the 15%-still-lands case, and
        holding it is precisely the blanket cooldown #5869 rejects. A SECOND
        consecutive identical ``policy_block`` is the repetition that proves futility,
        and only then does the issue leave the candidate stream."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # One re-blockable terminal, 1h ago. The first retry MUST stay admitted.
            self._witness(runs, 4568, self.EPOCH - 3600, reason="policy_block")
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=self.EPOCH), set(),
                "a FIRST retry after a policy_block must still be admitted — 15% of "
                "them land a witnessed commit")
            # The retry ran and re-blocked IDENTICALLY. Now hold it.
            self._witness(runs, 4568, self.EPOCH - 1800, reason="policy_block")
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=self.EPOCH), {4568})
            records = mod.reblock_streak_held_records(runs, now_ts=self.EPOCH)
            self.assertEqual(records[0]["reason"], "policy_block")
            self.assertEqual(records[0]["streak"], 2)

    def test_two_different_terminals_are_not_a_streak(self):
        """``policy_block`` then ``self_modify`` is two blocks but NOT the same block,
        so nothing is proven repeatable and the issue stays free. Same for a
        re-blockable terminal separated by an untyped ``unknown`` — over the window
        ``policy_block -> unknown`` (13) actually OUTNUMBERS ``policy_block ->
        policy_block`` (8), and folding those together is the reason-class hold in
        disguise."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._witness(runs, 3695, self.EPOCH - 3600, reason="policy_block")
            self._witness(runs, 3695, self.EPOCH - 1800, reason="self_modify")
            # Different reason interleaved: not identical, not held.
            self._witness(runs, 3267, self.EPOCH - 5400, reason="policy_block")
            self._witness(runs, 3267, self.EPOCH - 3600, reason="unknown")
            self._witness(runs, 3267, self.EPOCH - 1800, reason="policy_block")
            # A transient wall is not a structural block at all.
            self._witness(runs, 5103, self.EPOCH - 3600, reason="auth_wall")
            self._witness(runs, 5103, self.EPOCH - 1800, reason="auth_wall")
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=self.EPOCH), set())

    def test_landed_commit_in_the_tail_clears_the_hold(self):
        """The structural clear. An admitted run that lands a witnessed commit breaks
        the tail, so the hold evaporates with no reset verb — the property the
        lane-livelock / repair-cooldown / goal-park defects all lacked."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._witness(runs, 5586, self.EPOCH - 7200, reason="policy_block")
            self._witness(runs, 5586, self.EPOCH - 5400, reason="policy_block")
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=self.EPOCH), {5586})
            self._witness(runs, 5586, self.EPOCH - 1800, claim="CLAIM_WITNESSED")
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=self.EPOCH), set(),
                "a landed commit must clear the hold")

    def test_ttl_admits_one_probe_and_the_probe_outcome_decides(self):
        """The TTL does not declare the issue retryable — it admits exactly ONE probe
        so the streak CAN break. Without it a held issue could never produce the
        record that frees it (a self-sealing hold). A probe that re-blocks identically
        re-arms the hold for another window."""
        mod = load()
        ttl = mod.DEFAULT_REBLOCK_STREAK_HOLD_TTL_H
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._witness(runs, 4518, self.EPOCH - 7200, reason="policy_block")
            self._witness(runs, 4518, self.EPOCH - 5400, reason="policy_block")
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=self.EPOCH), {4518})
            # Past the TTL (measured from the NEWEST run in the streak) the hold
            # lapses and one probe is admitted.
            lapsed = self.EPOCH - 5400 + ttl * 3600 + 60
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=lapsed), set())
            # That probe re-blocked identically -> the hold RE-ARMS for another window.
            self._witness(runs, 4518, lapsed, reason="policy_block")
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=lapsed + 60), {4518})
            # A probe with a DIFFERENT outcome would have cleared it instead.
            self._witness(runs, 4518, lapsed + 120, reason="restart_exhausted")
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=lapsed + 180), set())

    # ---- #5869 follow-up: the TTL margin, re-derived over the whole corpus --------
    #
    # The real 10-run trace of #4568 (UTC launch stamps), the ONLY trace in the 2396
    # -record corpus that sets the zero-witnessed-loss boundary. Its 06:52 win is
    # f34f02f84; the load-bearing gap is 17:32 -> 00:30 = 6.97h, between a block and
    # an UNRELATED untyped run.
    _T4568 = [
        ("08-06 02:30", "unknown"), ("08-06 04:56", "policy_block"),
        ("08-06 07:44", "unknown"), ("08-06 11:01", "self_modify"),
        ("08-06 14:00", "policy_block"), ("08-06 17:32", "policy_block"),
        ("08-06 20:49", "policy_block"), ("08-07 00:30", "unknown"),
        ("08-07 04:10", "policy_block"), ("08-07 06:52", None),
    ]

    @staticmethod
    def _utc(label: str) -> float:
        import datetime as dt
        return dt.datetime.strptime("2026-" + label, "%Y-%m-%d %H:%M").replace(
            tzinfo=dt.timezone.utc).timestamp()

    def _replay(self, mod, trace, issue: int, ttl_h: float) -> list[str]:
        """Counterfactual replay of ``trace`` under a ``ttl_h`` hold, driven by the
        REAL gate rather than a re-implementation of it: at each launch instant the
        picker sees only the sidecars of the runs that were not suppressed upstream,
        which is what makes the cascade visible. Returns the SUPPRESSED labels."""
        suppressed: list[str] = []
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            for label, reason in trace:
                ts = self._utc(label)
                held = mod.reblock_streak_held_issues(runs, ttl_h=ttl_h, now_ts=ts)
                if issue in held:
                    suppressed.append(label)
                    continue
                if reason is None:
                    self._witness(runs, issue, ts, claim="CLAIM_WITNESSED")
                else:
                    self._witness(runs, issue, ts, reason=reason)
        return suppressed

    def test_default_ttl_keeps_a_margin_under_the_measured_first_loss(self):
        """The shipped TTL must sit a stated distance BELOW the replayed boundary.

        This is the invariant the first cut of the constant lacked: it was justified
        by a quoted "band", and a band cannot be re-checked when the corpus moves.
        Naming the boundary and the margin makes the justification executable."""
        mod = load()
        self.assertLessEqual(
            mod.DEFAULT_REBLOCK_STREAK_HOLD_TTL_H
            + mod.REBLOCK_STREAK_MIN_TTL_MARGIN_H,
            mod.REBLOCK_STREAK_FIRST_LOSS_TTL_H,
            "the default TTL must stay at least "
            f"{mod.REBLOCK_STREAK_MIN_TTL_MARGIN_H}h under the measured "
            f"{mod.REBLOCK_STREAK_FIRST_LOSS_TTL_H}h first-loss boundary")

    def test_boundary_is_a_suppression_cascade_not_a_late_lander(self):
        """Pins the mechanism, because the mechanism is what was mis-read.

        At the measured boundary the hold does NOT eat the win by out-waiting it: the
        win launches 2.7h after that issue's last block, well inside ANY TTL here.
        It eats an untyped `unknown` 6.97h downstream of an EARLIER block, and only
        the DELETION of that run re-forms a [policy_block, policy_block] tail that
        never existed in the real trace. That is why reading the block-to-win
        distribution alone (nothing between 2.69h and 9.33h) cannot find this."""
        mod = load()
        at_boundary = self._replay(
            mod, self._T4568, 4568, mod.REBLOCK_STREAK_FIRST_LOSS_TTL_H)
        self.assertIn("08-07 00:30", at_boundary,
                      "the cascade TRIGGER: the untyped run 6.97h after the 17:32 "
                      "block is what a 7.0h TTL newly swallows")
        self.assertIn("08-07 06:52", at_boundary,
                      "and deleting it costs the f34f02f84 win — the boundary")
        # The win is NOT a late lander: it launches 2.7h after the last block, so a
        # hold that ate it by duration alone would have eaten it at every TTL >= 3h.
        self.assertLess(
            (self._utc("08-07 06:52") - self._utc("08-07 04:10")) / 3600.0, 3.0)

    def test_shipped_default_loses_nothing_on_the_boundary_trace(self):
        """Fail-safe direction: at the SHIPPED TTL the same real trace keeps its win,
        and it keeps the cascade trigger too — the margin is margin in the mechanism,
        not just in the number."""
        mod = load()
        at_default = self._replay(
            mod, self._T4568, 4568, mod.DEFAULT_REBLOCK_STREAK_HOLD_TTL_H)
        self.assertNotIn("08-07 06:52", at_default,
                         "the shipped default must not destroy f34f02f84")
        self.assertNotIn("08-07 00:30", at_default,
                         "nor arm the cascade that destroys it")
        # It is still a HOLD, not a no-op: the repeat 3.3h after the 14:00/17:32
        # streak is exactly the futile re-dispatch the gate exists to refuse.
        self.assertIn("08-06 20:49", at_default)

    def test_updated_at_bump_and_kill_switches(self):
        """A guard refusal is a verdict on what the ISSUE ASKED FOR, so a re-scope
        (fresh gh ``updatedAt``) genuinely changes it and re-admits early — the same
        content-keyed escape the contract/multi-lane ledgers take, and why this hold
        does NOT take ``collision_held_records``' local-tree opt-out. Either knob at 0
        disables the gate, matching the other holds' kill switch."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._witness(runs, 4568, self.EPOCH - 3600, reason="policy_block")
            self._witness(runs, 4568, self.EPOCH - 1800, reason="policy_block")
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=self.EPOCH), {4568})
            # An updatedAt OLDER than the streak does not void it.
            self.assertEqual(mod.reblock_streak_held_issues(
                runs, now_ts=self.EPOCH, updated_ts={4568: self.EPOCH - 2400}),
                {4568})
            # A re-scope AFTER the last block re-admits the issue.
            self.assertEqual(mod.reblock_streak_held_issues(
                runs, now_ts=self.EPOCH, updated_ts={4568: self.EPOCH - 600}), set())
            self.assertEqual(mod.reblock_streak_held_issues(
                runs, streak_n=0, now_ts=self.EPOCH), set())
            self.assertEqual(mod.reblock_streak_held_issues(
                runs, ttl_h=0, now_ts=self.EPOCH), set())
            # A stricter N than the observed streak does not fire.
            self.assertEqual(mod.reblock_streak_held_issues(
                runs, streak_n=3, now_ts=self.EPOCH), set())

    def test_durable_hold_survives_the_process_the_tick_scoped_one_does_not(self):
        """The DEFECT this closes, stated as a contrast.

        ``held_no_commit_issues`` reads the in-memory result of the witness sweep the
        CURRENT tick just ran, so a fresh dispatcher process — the next tick — has no
        memory of the refusal and re-picks the issue. That is the "in-memory tally
        dies with the process and silently never fires" shape. The durable reader
        keys off the retained ``.witness`` sidecars instead, so the same evidence
        still holds after the process that produced it is gone."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._witness(runs, 4568, self.EPOCH - 3600, reason="policy_block")
            self._witness(runs, 4568, self.EPOCH - 1800, reason="policy_block")
            # A NEW dispatcher process has no in-memory witness result to read.
            self.assertEqual(mod.held_no_commit_issues(None), set())
            self.assertEqual(mod.held_no_commit_issues({"no_commit": []}), set())
            # The durable reader still holds it off the very same evidence.
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=self.EPOCH), {4568})

    def test_history_is_launch_ordered_and_fail_open(self):
        """Ordering comes from the FILENAME stamp, not the mtime the witness sweep
        rewrites at an arbitrary later tick. Unreadable/odd sidecars are skipped, never
        raised, so this can only ever ADD a hold — it can never wedge the picker."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # Written newest-first on disk; history must still come back oldest-first.
            newer = self._witness(runs, 77, self.EPOCH - 1800, reason="policy_block")
            older = self._witness(runs, 77, self.EPOCH - 3600, reason="self_modify")
            os.utime(newer, (self.EPOCH - 9999, self.EPOCH - 9999))
            os.utime(older, (self.EPOCH, self.EPOCH))
            hist = mod.issue_witness_history(runs)
            self.assertEqual([r["reason"] for r in hist[77]],
                             ["self_modify", "policy_block"])
            # Garbage in the runs dir is ignored, not raised.
            (runs / "resolve-nope.witness").write_text("{", encoding="utf-8")
            (runs / "resolve-88-20260806-120000.witness").write_text(
                "not json", encoding="utf-8")
            self.assertNotIn(88, mod.issue_witness_history(runs))
            self.assertEqual(
                mod.reblock_streak_held_issues(runs, now_ts=self.EPOCH), set())
            # A missing runs dir is empty, never an exception.
            self.assertEqual(mod.issue_witness_history(runs / "gone"), {})


class GuardArgvProbeTest(unittest.TestCase):
    """#5868 done-condition 3: stop a DOA wave BEFORE it spawns.

    A ``--compact-solvency-floor`` flag-parse regression killed every worker at spawn
    for six days (~350 worker-units) and read as an idle fleet throughout, because a
    death before the gateway binds classifies as ``unknown``. ``internal/dispatchdoa``
    detects that shape after the fact; this gate refuses to dispatch into it at all.

    The first test is the load-bearing one: it pins the MEASURED finding that the
    obvious probe (``fak guard --help``) does not discriminate this regression, so the
    gate must compare the flag INVENTORY instead of an exit code.
    """

    # The real `fak guard --help` output, exit 0, 1452 bytes — abridged to the shape
    # that matters: it is a curated COMMON-flags block using the `--name` form, and it
    # never mentions --compact-solvency-floor. Measured against the shipped binary.
    SHORT_HELP = (
        "usage: fak guard [flags] -- <agent command...>\n"
        "\ncommon flags:\n"
        "  --policy         capability-floor manifest to enforce\n"
        "  --provider       upstream wire: anthropic|openai|gemini|xai\n"
        "  --audit          change where the decision journal is written\n"
        "\n76 flags in this build. 'fak guard -h -all' lists every one grouped.\n")

    # The flags dispatch_worker.claude_guard_budget_args + guard_wrap actually pass.
    PLANNED = ["provider", "precompact-hook", "session-id", "context-budget-tokens",
               "compact-history-budget", "compact-solvency-floor", "restart-on-budget",
               "restart-limit", "max-duration", "audit"]

    @staticmethod
    def _inventory(names) -> str:
        """`fak guard -h -all` output shape: Go flag.PrintDefaults emits `  -name type`
        at exactly two spaces, with the description wrapped beneath at four+tab."""
        return "usage: fak guard [flags] -- <agent command...>\n" + "".join(
            f"  -{n} string\n    \tdescription of {n}\n" for n in names)

    class _Res:
        def __init__(self, rc, out="", err=""):
            self.returncode, self.stdout, self.stderr = rc, out, err

    def _runner(self, res, calls: list):
        def run(argv):
            calls.append(list(argv))
            if isinstance(res, Exception):
                raise res
            return res
        return run

    def _bin(self, d: Path, body: str = "x" * 64) -> Path:
        p = d / "fak.exe"
        p.write_text(body, encoding="utf-8")
        return p

    def _argv(self, exe: Path, flags=None) -> list[str]:
        out = [str(exe), "guard"]
        for f in (flags if flags is not None else self.PLANNED):
            out += [f"--{f}", "v"]
        # The AGENT's own command, after the separator. Its flags are parsed by
        # claude, not by the guard, so they must never be probed.
        return out + ["--", "claude", "-p", "--dangerously-skip-permissions"]

    def test_guard_help_alone_does_not_discriminate_the_regression(self):
        """THE finding. `fak guard --help` exits 0 on a binary that is missing the
        flag — it proves the binary RUNS, which was never in doubt; the binary ran
        fine, it just did not know one flag. An exit-code probe on --help would have
        reported HEALTHY through all six days of the outage. So the gate must not
        refuse on it, and must not be fooled into thinking it has a verdict."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            exe = self._bin(runs)
            self.assertNotIn("compact-solvency-floor", self.SHORT_HELP,
                             "the real short usage does not even name the flag")
            calls: list = []
            probe = mod.probe_guard_flags(
                exe, runner=self._runner(self._Res(0, err=self.SHORT_HELP), calls))
            self.assertEqual(probe["rc"], 0, "--help succeeds on the broken binary")
            self.assertLess(len(probe["flags"]), mod._GUARD_PROBE_MIN_FLAGS)
            # ...and because that reading is not a plausible inventory, the gate
            # SPAWNS rather than condemning a binary it cannot actually read.
            self.assertIsNone(mod.gate_spawn_on_guard_argv(
                self._argv(exe), runs,
                runner=self._runner(self._Res(0, err=self.SHORT_HELP), calls)))
            self.assertFalse((runs / mod._GUARD_PROBE_LEDGER).exists(),
                             "an inconclusive reading is never cached")

    def test_healthy_build_id_is_not_held(self):
        """A binary that registers every planned flag dispatches, and the probe it
        cost is not re-paid — the whole point of keying on the build id."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            exe = self._bin(runs)
            healthy = self._Res(0, err=self._inventory(
                self.PLANNED + [f"filler-{i}" for i in range(60)]))
            calls: list = []
            for _ in range(3):
                self.assertIsNone(mod.gate_spawn_on_guard_argv(
                    self._argv(exe), runs, runner=self._runner(healthy, calls)))
            self.assertEqual(len(calls), 1, "one probe per build-id, not per spawn")
            self.assertEqual(calls[0][1:], ["guard", "-h", "-all"])

    def test_missing_flag_refuses_the_wave_before_any_spawn(self):
        """The outage shape itself: the producer passes a flag the binary does not
        register. Every spawn against it would die at flag-parse before the gateway
        binds, so the tick plans ZERO spawns instead of ~350 dead worker-units."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            exe = self._bin(runs)
            old = [f for f in self.PLANNED if f != "compact-solvency-floor"]
            stale = self._Res(0, err=self._inventory(
                old + [f"filler-{i}" for i in range(60)]))
            verdict = mod.gate_spawn_on_guard_argv(
                self._argv(exe), runs, runner=self._runner(stale, []))
            self.assertIsNotNone(verdict)
            self.assertEqual(verdict["verdict"], "GUARD_ARGV_UNSUPPORTED")
            self.assertEqual(verdict["missing_flags"], ["compact-solvency-floor"])
            self.assertIn("rebuild or reinstall", verdict["reason"],
                          "a refusal that does not name its recovery path is an "
                          "outage of its own")

    def test_rebuild_clears_the_refusal_with_no_reset_verb(self):
        """Recovery path #1, and the reason the cache is keyed on the BUILD ID: the
        operator action that fixes the outage (rebuild/reinstall) is the same one that
        invalidates the cache. Nothing has to be cleared by hand."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            exe = self._bin(runs)
            old = [f for f in self.PLANNED if f != "compact-solvency-floor"]
            calls: list = []
            self.assertIsNotNone(mod.gate_spawn_on_guard_argv(
                self._argv(exe), runs, runner=self._runner(
                    self._Res(0, err=self._inventory(
                        old + [f"f{i}" for i in range(60)])), calls)))
            before = mod.guard_build_id(exe)
            # Rebuild: a different binary at the same path.
            self._bin(runs, body="y" * 4096)
            self.assertNotEqual(mod.guard_build_id(exe), before)
            self.assertIsNone(mod.gate_spawn_on_guard_argv(
                self._argv(exe), runs, runner=self._runner(
                    self._Res(0, err=self._inventory(
                        self.PLANNED + [f"f{i}" for i in range(60)])), calls)))
            self.assertEqual(len(calls), 2, "the new build id forces exactly one "
                                            "fresh probe, not a probe per spawn")

    def test_producer_side_fix_clears_against_the_same_cached_inventory(self):
        """Recovery path #2, and why the cache holds the INVENTORY rather than a
        verdict: dropping the flag from the dispatcher clears the refusal with no
        re-exec at all, because the verdict is recomputed every tick."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            exe = self._bin(runs)
            old = [f for f in self.PLANNED if f != "compact-solvency-floor"]
            stale = self._Res(0, err=self._inventory(
                old + [f"f{i}" for i in range(60)]))
            calls: list = []
            self.assertIsNotNone(mod.gate_spawn_on_guard_argv(
                self._argv(exe), runs, runner=self._runner(stale, calls)))
            self.assertIsNone(mod.gate_spawn_on_guard_argv(
                self._argv(exe, flags=old), runs, runner=self._runner(stale, calls)))
            self.assertEqual(len(calls), 1, "no re-probe needed to clear")

    def test_every_ambiguous_reading_fails_open_and_is_never_cached(self):
        """A probe failure must not wedge the fleet — failing closed forever is its
        own outage, and this gate refuses to SPAWN, so its fail direction is inverted
        from every other hold in this file. Only a positive observation refuses."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            exe = self._bin(runs)
            argv = self._argv(exe)
            short = self._inventory([f"f{i}" for i in range(5)])
            for label, res in (
                    ("exec error", OSError("binary vanished")),
                    ("probe timeout", subprocess.TimeoutExpired("fak", 20)),
                    ("non-zero exit (an older binary rejects -all)",
                     self._Res(2, err="flag provided but not defined: -all\n")),
                    ("implausibly short inventory", self._Res(0, err=short))):
                with self.subTest(label):
                    self.assertIsNone(
                        mod.gate_spawn_on_guard_argv(
                            argv, runs, runner=self._runner(res, [])),
                        f"{label} must SPAWN, not wedge the fleet")
                    self.assertFalse((runs / mod._GUARD_PROBE_LEDGER).exists())
            # An unstattable binary is ambiguous too, not a condemnation.
            self.assertIsNone(mod.gate_spawn_on_guard_argv(
                self._argv(runs / "gone.exe"), runs,
                runner=self._runner(self._Res(0, err=short), [])))

    def test_agent_argv_and_peeled_flags_are_not_probed(self):
        """Two false-positive sources that would refuse a perfectly good binary: the
        agent's own flags after the ``--`` separator (parsed by claude, not the
        guard), and the postures cmd/fak/guard.go PEELS before fs.Parse — which are
        therefore absent from the FlagSet's own inventory by design."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            exe = self._bin(runs)
            planned = mod.planned_guard_flags(
                [str(exe), "guard", "--provider", "anthropic", "--core-lock-all",
                 "--", "claude", "-p", "--dangerously-skip-permissions"])
            self.assertEqual(planned, {"provider"})
            # Not a guard wrap at all -> nothing to skew against.
            self.assertEqual(mod.planned_guard_flags(
                [str(exe), "--verbose", "--", "claude"]), set())
            self.assertIsNone(mod.gate_spawn_on_guard_argv(
                [str(exe), "claude", "-p"], runs,
                runner=self._runner(self._Res(0, err=""), [])))

    def test_ttl_expiry_and_kill_switch(self):
        """Recovery paths #3 and #4: a cached inventory cannot outlive its evidence
        even when neither side moved, and the gate has the same 0-disables kill switch
        as every other hold in this file."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            exe = self._bin(runs)
            full = self._Res(0, err=self._inventory(
                self.PLANNED + [f"f{i}" for i in range(60)]))
            calls: list = []
            now = 1786363200.0
            self.assertIsNone(mod.gate_spawn_on_guard_argv(
                self._argv(exe), runs, runner=self._runner(full, calls), now_ts=now))
            self.assertIsNone(mod.gate_spawn_on_guard_argv(
                self._argv(exe), runs, runner=self._runner(full, calls),
                now_ts=now + mod.DEFAULT_GUARD_PROBE_TTL_H * 3600 - 60))
            self.assertEqual(len(calls), 1, "inside the TTL the cache answers")
            self.assertIsNone(mod.gate_spawn_on_guard_argv(
                self._argv(exe), runs, runner=self._runner(full, calls),
                now_ts=now + mod.DEFAULT_GUARD_PROBE_TTL_H * 3600 + 60))
            self.assertEqual(len(calls), 2, "past the TTL it re-probes")
            # Kill switch: never probes, never refuses.
            off: list = []
            old = [f for f in self.PLANNED if f != "compact-solvency-floor"]
            self.assertIsNone(mod.gate_spawn_on_guard_argv(
                self._argv(exe), runs, ttl_h=0, runner=self._runner(
                    self._Res(0, err=self._inventory(
                        old + [f"f{i}" for i in range(60)])), off)))
            self.assertEqual(off, [])

    def test_probe_argv_is_the_help_form_and_never_launches_an_agent(self):
        """The probe must stay a probe: `-h` short-circuits inside flag.Parse, so it
        binds no gateway and hands off no task. Asserted on the captured argv, because
        a probe that could launch is not one probe per build-id, it is an outage."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            exe = self._bin(Path(d))
            calls: list = []
            mod.probe_guard_flags(exe, runner=self._runner(self._Res(0), calls))
            self.assertEqual(calls, [[str(exe), "guard", "-h", "-all"]])
            self.assertNotIn("--", calls[0], "no agent-command separator")
            self.assertNotIn("--probe", calls[0],
                             "`fak guard --probe` is a SMOKE mode that still brings "
                             "up the gateway — not a cheap argv probe")

    def test_default_runner_suppresses_the_console_window(self):
        """The DEFAULT runner (the one that actually ships — every other test injects
        a fake) must spawn with CREATE_NO_WINDOW. This is not cosmetic: the probe runs
        once per dispatcher process under headless automation, and the pre-push
        DESKTOP_POPUP_REGRESSION gate refuses a console-tool spawn that would flash a
        window. It refused this function's first cut, which omitted the flag."""
        mod = load()
        seen: dict = {}

        def fake_run(argv, **kwargs):
            seen["argv"], seen["kwargs"] = list(argv), kwargs
            return self._Res(0, err=self._inventory(["provider"]))

        with tempfile.TemporaryDirectory() as d:
            exe = self._bin(Path(d))
            with mock.patch.object(mod.subprocess, "run", fake_run):
                mod.probe_guard_flags(exe)
        self.assertEqual(seen["argv"][1:], ["guard", "-h", "-all"])
        self.assertEqual(seen["kwargs"].get("creationflags"),
                         mod.no_window_creationflags())
        self.assertTrue(seen["kwargs"].get("timeout"),
                        "an unbounded probe could hang the dispatcher")


# --- Cap-KIND corpus (#5890) ----------------------------------------------------
# One row per cap banner Claude actually emits, asserted at the two seams that decide
# fleet behavior: ``_cap_hit_from_text`` (is it seen, and what kind is it?) and
# ``_write_cap_hold`` (how long is the seat held?). The regression pinned: the detector
# matched "hit your WEEKLY limit" but not "hit your 5-HOUR limit" (a ``[\w\s]``-only
# class stops at the hyphen), saw neither of Claude's "…limit reached." phrasings at
# all, read only "resets <when>" and not the modern "will reset at/on <when>", and filed
# every unqualified cap under a `session` kind whose 90-minute ceiling truncates a real
# 5-hour window. Every assertion is pure over an injected clock.
#
# Lives here rather than in its own tools/*_capkind_test.py because internal/pythongate
# refuses any NEW tracked tools/*.py: a fixture corpus for a grandfathered tool is not
# one of the narrow widening cases, so it folds into the grandfathered test module.
# A fixed instant with room on both sides: 2026-05-28 20:26:40 UTC == 13:26 PT, so a "5pm"
# PT reset is ~3.5h out (inside a real 5-hour window, well past the 90-minute session clamp)
# and a "6:10am" PT reset already passed today (the gem8 stale-time-of-day case).
NOW_TS = 1_780_000_000.0
NOW_UTC = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=NOW_TS)


class CapKindClassificationTest(unittest.TestCase):
    """Every banner is SEEN, and is filed under the cap kind it actually names."""

    # (label, banner, expected kind). The first four were undetected outright before #5890.
    CASES = [
        ("5h modern",
         "Claude usage limit reached. Your limit will reset at 5pm (America/Los_Angeles).",
         "rolling_5h"),
        ("5h hit-your-hyphen",
         "You've hit your 5-hour limit. Your limit will reset at 5pm (America/Los_Angeles).",
         "rolling_5h"),
        ("weekly modern",
         "Claude usage limit reached. Your weekly limit will reset on Jun 2 at 9am "
         "(America/Los_Angeles).",
         "weekly"),
        ("weekly for-the-week",
         "You've hit your limit for the week. Try again Jun 2.",
         "weekly"),
        # These three were already detected; they must not regress.
        ("5h resets-form",
         "You've hit your usage limit; resets 5pm (America/Los_Angeles)",
         "rolling_5h"),
        ("weekly hit-your",
         "You've hit your weekly limit. Your limit will reset at Jun 2, 9am "
         "(America/Los_Angeles).",
         "weekly"),
        ("session legacy",
         "You've hit your session limit · resets 6:10am (America/Los_Angeles)",
         "session"),
    ]

    def test_every_banner_is_detected_and_correctly_kinded(self) -> None:
        mod = load()
        for label, banner, want_kind in self.CASES:
            with self.subTest(label):
                hit = mod._cap_hit_from_text(banner)
                self.assertIsNotNone(hit, f"{label}: banner not detected at all")
                self.assertEqual(hit["kind"], want_kind, label)

    def test_hyphen_does_not_hide_the_five_hour_banner(self) -> None:
        """The exact asymmetry that made the fleet see weekly caps and miss 5-hour caps:
        `hit your[\\w\\s]*limit` stops at the hyphen in "5-hour"."""
        mod = load()
        for phrase in ("hit your 5-hour limit", "hit your 5 hour limit",
                       "hit your weekly limit", "hit your usage limit"):
            with self.subTest(phrase):
                self.assertIsNotNone(mod._CAP_BANNER_RE.search(phrase), phrase)

    def test_spawn_header_is_still_not_a_cap_banner(self) -> None:
        """The widened detector must stay phrase-anchored: a worker's spawn header and
        ordinary agent prose that merely mention a limit are NOT cap banners."""
        mod = load()
        for benign in (
            "# fak-spawn issue=42 lane=docs backend=claude",
            "considering the rate limit design for the gateway",
            "the limit of this approach is that it needs a fixture",
        ):
            with self.subTest(benign):
                self.assertIsNone(mod._CAP_BANNER_RE.search(benign), benign)


class CapResetClauseTest(unittest.TestCase):
    """Claude's modern "will reset at/on <when>" clause is read, not dropped."""

    def test_will_reset_clause_is_captured(self) -> None:
        mod = load()
        cases = [
            ("Your limit will reset at 5pm (America/Los_Angeles).", "5pm"),
            ("Your weekly limit will reset on Jun 2 at 9am (America/Los_Angeles).",
             "Jun 2 at 9am"),
            ("Your limit will reset on Jun 2, 9am (America/Los_Angeles).", "Jun 2, 9am"),
        ]
        for text, want in cases:
            with self.subTest(text):
                hit = mod._cap_hit_from_text("You've hit your usage limit. " + text)
                self.assertIsNotNone(hit)
                self.assertEqual(hit["reset_text"], want)

    def test_legacy_resets_clause_still_wins(self) -> None:
        """The pre-existing "resets <when> (America/Los_Angeles)" form is unchanged."""
        mod = load()
        hit = mod._cap_hit_from_text(
            "You've hit your session limit · resets 6:10am (America/Los_Angeles)")
        self.assertEqual(hit["reset_text"], "6:10am")

    def test_announced_wait_still_wins_over_the_reset_clause(self) -> None:
        """#2610's gateway path is untouched: a relative announced_wait is still honored."""
        mod = load()
        hit = mod._cap_hit_from_text(
            "fak-turn trace=guard FAILED reason=rate_limited wire=anthropic_messages "
            "kind=weekly_limit announced_wait=1h7m0s")
        self.assertEqual(hit["kind"], "weekly")
        self.assertEqual(hit["reset_text"], "1h7m0s")


class CapWeekdayResetTest(unittest.TestCase):
    """A weekly cap names its reset by WEEKDAY far more often than by date, and the weekday
    word used to be ignored outright — the bare time-of-day branch resolved it to today or
    tomorrow. NOW_UTC is Thu 2026-05-28 20:26 UTC == Thu 13:26 PT, so the arithmetic below is
    checkable by hand."""

    def test_weekday_resolves_to_the_next_such_day(self) -> None:
        mod = load()
        # Thu 13:26 PT -> next Monday 09:00 PT is 4 days out (Jun 1), not tomorrow.
        got = mod._parse_reset_to_utc("Monday at 9am", NOW_UTC)
        self.assertEqual(got, dt.datetime(2026, 6, 1, 16, 0))

    def test_same_weekday_earlier_in_the_day_rolls_a_full_week(self) -> None:
        """Thursday 9am seen at Thursday 13:26 PT is NEXT Thursday, not four minutes ago
        and not "today" — a past instant would be dropped and fall to the blind fallback."""
        mod = load()
        got = mod._parse_reset_to_utc("Thursday at 9am", NOW_UTC)
        self.assertEqual(got, dt.datetime(2026, 6, 4, 16, 0))

    def test_weekday_under_hold_is_the_one_that_bit(self) -> None:
        """The regression in one number: "Wednesday at 3pm" read on a Thursday used to be
        parsed as TODAY 3pm — a 1.6-hour hold on a cap that has six days left to run."""
        mod = load()
        hit = mod._cap_hit_from_text(
            "You've hit your weekly limit. Your limit will reset on Wednesday at 3pm "
            "(America/Los_Angeles).")
        self.assertEqual(hit["kind"], "weekly")
        until = mod._parse_reset_to_utc(hit["reset_text"], NOW_UTC)
        self.assertGreater((until - NOW_UTC).total_seconds() / 3600.0, 24.0)

    def test_an_explicit_date_still_outranks_a_weekday(self) -> None:
        mod = load()
        # Jun 3 2026 is a WEDNESDAY, so a leading "Mon" token must lose to the date.
        self.assertEqual(mod._parse_reset_to_utc("Mon Jun 3 at 9am", NOW_UTC),
                         dt.datetime(2026, 6, 3, 16, 0))

    def test_a_day_lookalike_word_cannot_manufacture_a_reset(self) -> None:
        """The weekday branch only runs once a time-of-day is present, and the pattern is
        anchored to whole day words — "monitoring" and "summary" are not Monday/Sunday."""
        mod = load()
        self.assertIsNone(mod._parse_reset_to_utc("monitoring the summary", NOW_UTC))
        # A bare time-of-day beside a lookalike word keeps the today/tomorrow branch.
        self.assertEqual(mod._parse_reset_to_utc("monitoring resumes 5pm", NOW_UTC),
                         dt.datetime(2026, 5, 29, 0, 0))


class CapHoldLengthTest(unittest.TestCase):
    """Each cap kind is held for ITS OWN window, not one bucket's ceiling."""

    def _hold_hours(self, mod, hit) -> float:
        with tempfile.TemporaryDirectory() as d:
            out = mod._write_cap_hold(Path(d), product="claude", account_tag="a",
                                      hit=hit, now_ts=NOW_TS, fallback_min=60,
                                      source="test")
        until = dt.datetime.fromisoformat(out["until"].replace("Z", ""))
        return (until - NOW_UTC).total_seconds() / 3600.0

    def test_five_hour_cap_is_not_truncated_to_the_session_clamp(self) -> None:
        """A 5-hour cap naming a reset ~3.5h out is held ~3.5h. Before #5890 the
        `session` bucket's 90-minute ceiling cut it to 1.5h and the dispatcher respawned
        into the same wall two hours early."""
        mod = load()
        hit = mod._cap_hit_from_text(
            "You've hit your 5-hour limit. Your limit will reset at 5pm (America/Los_Angeles).")
        held = self._hold_hours(mod, hit)
        self.assertGreater(held, mod._SESSION_HOLD_MAX_MIN / 60.0,
                           "a real 5-hour reset must outlive the session clamp")
        self.assertLessEqual(held, mod._ROLLING_HOLD_MAX_MIN / 60.0)

    def test_rolling_cap_ceiling_still_bounds_a_stale_time_of_day(self) -> None:
        """The clamp's original motive survives: a bare time-of-day that already passed
        today must not become a ~24h wall, only a ~5h one."""
        mod = load()
        hit = {"kind": "rolling_5h", "reset_text": "6:10am"}
        self.assertLessEqual(self._hold_hours(mod, hit),
                             mod._ROLLING_HOLD_MAX_MIN / 60.0 + 0.01)

    def test_session_limit_hold_is_still_bounded_to_ninety_minutes(self) -> None:
        """The gem8 false-cap guard is untouched for the kind it was written for."""
        mod = load()
        hit = mod._cap_hit_from_text(
            "You've hit your session limit · resets 6:10am (America/Los_Angeles)")
        self.assertEqual(hit["kind"], "session")
        self.assertLessEqual(self._hold_hours(mod, hit),
                             mod._SESSION_HOLD_MAX_MIN / 60.0 + 0.02)

    def test_unannounced_weekly_cap_outlasts_the_rolling_fallback(self) -> None:
        """A weekly cap whose banner names no parseable reset must not take the short
        fallback: its real window is days, so a 1-hour hold re-admits the seat ~168 times
        before it can serve."""
        mod = load()
        hit = mod._cap_hit_from_text("You've hit your limit for the week.")
        self.assertEqual(hit["kind"], "weekly")
        self.assertAlmostEqual(self._hold_hours(mod, hit),
                               mod._WEEKLY_FALLBACK_MIN / 60.0, places=2)

    def test_dated_weekly_cap_keeps_its_full_reset(self) -> None:
        """A weekly cap naming a real dated reset is held to that reset — no kind ceiling
        applies to weekly."""
        mod = load()
        hit = mod._cap_hit_from_text(
            "You've hit your weekly limit. Your limit will reset at Jun 2, 9am "
            "(America/Los_Angeles).")
        self.assertEqual(hit["kind"], "weekly")
        # Jun 2 09:00 PDT == Jun 2 16:00 UTC, ~4.8 days past NOW_UTC (May 28 20:26 UTC).
        self.assertGreater(self._hold_hours(mod, hit), 24.0)


if __name__ == "__main__":
    unittest.main()
