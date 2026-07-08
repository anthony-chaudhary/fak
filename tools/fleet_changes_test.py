#!/usr/bin/env python3
"""Hermetic tests for fleet_changes' per-account transition fold.

The diff + render are pure given two ``{tag: state}`` dicts, so most of these touch
neither clock nor disk; the ledger round-trip uses a tempdir. What is pinned: an
account that just lost/regained capacity is announced, a chronically-down or archived
account is NOT restated, and the first/quiet tick renders nothing."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_changes  # noqa: E402


def _snap(available=(), throttled=(), blocked=()):
    """A minimal fleet_top-shaped snapshot carrying just the account block the
    fingerprint reads. ``blocked`` items are (tag, kind) pairs."""
    return {
        "accounts": {
            "available": list(available),
            "throttled": [{"tag": t} for t in throttled],
            "blocked": [{"tag": t, "kind": k} for t, k in blocked],
        }
    }


class FingerprintTest(unittest.TestCase):
    def test_maps_every_account_to_one_state(self):
        fp = fleet_changes.fingerprint(_snap(
            available=["alpha", "foxtrot"],
            throttled=["bravo"],
            blocked=[("charlie", "auth"), ("delta", "access"), ("echo", "other")],
        ))
        self.assertEqual(fp, {
            "alpha": "usable", "foxtrot": "usable",
            "bravo": "throttled",
            "charlie": "login", "delta": "access", "echo": "blocked",
        })

    def test_available_wins_over_a_stale_throttle_row(self):
        # An expired-but-cached throttle already reads available; it must not also appear
        # as a phantom down-row.
        fp = fleet_changes.fingerprint(_snap(available=["alpha"], throttled=["alpha"]))
        self.assertEqual(fp, {"alpha": "usable"})

    def test_partial_snapshot_never_raises(self):
        self.assertEqual(fleet_changes.fingerprint({}), {})
        self.assertEqual(fleet_changes.fingerprint({"accounts": {}}), {})


class DiffTest(unittest.TestCase):
    def test_usable_to_down_is_newly_down(self):
        d = fleet_changes.diff({"bravo": "usable"}, {"bravo": "throttled"})
        self.assertEqual(d["newly_down"], [{"tag": "bravo", "state": "throttled"}])
        self.assertEqual(d["still_down"], [])

    def test_down_to_usable_is_recovered(self):
        d = fleet_changes.diff({"charlie": "login"}, {"charlie": "usable"})
        self.assertEqual(d["recovered"], ["charlie"])

    def test_same_down_state_is_chronic_not_newly_down(self):
        d = fleet_changes.diff({"charlie": "login"}, {"charlie": "login"})
        self.assertEqual(d["newly_down"], [])
        self.assertEqual(d["still_down"], [{"tag": "charlie", "state": "login"}])

    def test_changed_block_kind_is_worsened(self):
        d = fleet_changes.diff({"bravo": "throttled"}, {"bravo": "login"})
        self.assertEqual(d["worsened"], [{"tag": "bravo", "state": "login", "was": "throttled"}])

    def test_first_seen_already_down_is_silent(self):
        # An account we never saw usable (an archived account entering the ledger) is not
        # announced as newly-down.
        d = fleet_changes.diff({}, {"zulu": "access"})
        self.assertEqual(d["newly_down"], [])
        self.assertEqual(d["still_down"], [])

    def test_vanished_account_is_dropped(self):
        # gone from cur -> not repeated. The whole point.
        d = fleet_changes.diff({"gone": "login"}, {"alpha": "usable"})
        self.assertNotIn("gone", str(d))


class ChangeLineTest(unittest.TestCase):
    def test_quiet_tick_renders_nothing(self):
        d = fleet_changes.diff({"alpha": "usable"}, {"alpha": "usable"})
        self.assertEqual(fleet_changes.change_line(d), "")

    def test_a_purely_chronic_tick_renders_nothing(self):
        # Nothing transitioned; a long-blocked account must NOT be restated on its own.
        d = fleet_changes.diff({"charlie": "login"}, {"charlie": "login"})
        self.assertEqual(fleet_changes.change_line(d), "")

    def test_newly_down_is_announced_in_human_terms(self):
        d = fleet_changes.diff({"bravo": "usable"}, {"bravo": "throttled"})
        line = fleet_changes.change_line(d)
        self.assertIn("changes vs last post:", line)
        self.assertIn("bravo hit its rate limit", line)

    def test_recovery_and_chronic_count_on_a_change_tick(self):
        prev = {"alpha": "usable", "bravo": "throttled", "charlie": "login"}
        cur = {"alpha": "throttled", "bravo": "usable", "charlie": "login"}
        line = fleet_changes.change_line(fleet_changes.diff(prev, cur))
        self.assertIn("alpha hit its rate limit", line)   # newly down
        self.assertIn("bravo is back", line)              # recovered
        self.assertIn("1 still down", line)               # chronic folded to a bare count
        self.assertNotIn("charlie", line)                 # ...never re-listed by name


class LedgerTest(unittest.TestCase):
    def test_first_tick_is_silent_then_a_transition_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "state.jsonl")
            # Tick 1: alpha usable, bravo usable. No prior -> nothing to report.
            first = fleet_changes.line_from_ledger(
                path, {"alpha": "usable", "bravo": "usable"}, "2026-06-23T17:00:00Z")
            self.assertEqual(first, "")
            # Tick 2: bravo throttles -> announced exactly once.
            second = fleet_changes.line_from_ledger(
                path, {"alpha": "usable", "bravo": "throttled"}, "2026-06-23T18:00:00Z")
            self.assertIn("bravo hit its rate limit", second)
            # Tick 3: unchanged -> silent (the chronic bravo is NOT repeated).
            third = fleet_changes.line_from_ledger(
                path, {"alpha": "usable", "bravo": "throttled"}, "2026-06-23T19:00:00Z")
            self.assertEqual(third, "")
            self.assertEqual(len(fleet_changes.tail(path, 10)), 3)  # every tick recorded

    def test_dry_run_does_not_write_the_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "state.jsonl")
            fleet_changes.append(path, {"alpha": "usable"}, "2026-06-23T17:00:00Z")
            fleet_changes.line_from_ledger(
                path, {"alpha": "throttled"}, "2026-06-23T18:00:00Z", record=False)
            self.assertEqual(len(fleet_changes.tail(path, 10)), 1)  # dry-run appended nothing

    def test_ledger_is_ring_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "state.jsonl")
            for i in range(5):
                fleet_changes.append(path, {"alpha": "usable"}, f"t{i}", cap=3)
            rows = fleet_changes.tail(path, 100)
            self.assertEqual(len(rows), 3)
            self.assertEqual([r["ts"] for r in rows], ["t2", "t3", "t4"])


if __name__ == "__main__":
    raise SystemExit(unittest.main())
