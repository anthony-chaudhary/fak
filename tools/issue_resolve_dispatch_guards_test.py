#!/usr/bin/env python3
"""Hermetic tests for dispatch guards, caps, health, leases, and argv probes."""
from __future__ import annotations

import ast
import atexit
import datetime as dt
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue_resolve_dispatch_test import ROOT, SCRIPT, load, passing_issue_contract  # noqa: E402


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
