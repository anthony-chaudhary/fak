#!/usr/bin/env python3
r"""Tests for launch_admission -- the single launch-admission gate (#617).

The headline test reproduces the 2026-06-24 storm: 15 ``claude --resume`` launches
fired onto ONE account inside ~60s. The gate must admit AT MOST the ceiling and
hand every excess launch a STRUCTURED ``DEFER`` reason bound to a retry time --
the throttled account's reset, or the window roll-off.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import launch_admission as LA  # noqa: E402

# The incident window: 22:04:03Z -> 22:04:59Z, ~15 launches ~4s apart.
START = datetime(2026, 6, 24, 22, 4, 3, tzinfo=timezone.utc)


def _counts(verdicts):
    admitted = [v for v in verdicts if v["verdict"] == LA.VERDICT_ADMIT]
    deferred = [v for v in verdicts if v["verdict"] == LA.VERDICT_DEFER]
    return admitted, deferred


class ThrottledBurstTest(unittest.TestCase):
    """The actual incident: q-netra was THROTTLED when the 15-burst hit."""

    def test_15_onto_one_throttled_account_all_defer_bound_to_reset(self):
        reset = "Jun 25, 1pm (America/Los_Angeles)"
        verdicts = LA.simulate_burst(
            ".claude-q-netra", 15, start=START,
            throttled=True, throttle_reset=reset,
            max_per_account=3, window_min=5, global_cap=10,
        )
        admitted, deferred = _counts(verdicts)
        # The throttle-gate is the most-specific refusal: ZERO admitted (<= cap),
        # every launch deferred and BOUND TO THE ACCOUNT'S RESET.
        self.assertEqual(len(admitted), 0)
        self.assertEqual(len(deferred), 15)
        for v in deferred:
            self.assertEqual(v["reason"], LA.REASON_THROTTLED)
            self.assertEqual(v["retry_after"], reset)


class RateCeilingBurstTest(unittest.TestCase):
    """No throttle verdict yet -> the per-account RATE ceiling must still cap it."""

    def test_15_onto_one_account_caps_at_ceiling(self):
        cap = 3
        verdicts = LA.simulate_burst(
            ".claude-q-netra", 15, start=START,
            max_per_account=cap, window_min=5, global_cap=50,
        )
        admitted, deferred = _counts(verdicts)
        # Exactly `cap` admitted; the remaining 12 deferred with a window-bound retry.
        self.assertEqual(len(admitted), cap)
        self.assertEqual(len(deferred), 15 - cap)
        for v in deferred:
            self.assertEqual(v["reason"], LA.REASON_RATE)
            self.assertIsNotNone(v["retry_after"])
            # retry_after is the oldest-in-window launch + the window: in the future
            # of the first launch, parseable, and never None.
            self.assertIsNotNone(LA.parse_ts(v["retry_after"]))
            self.assertGreater(LA.parse_ts(v["retry_after"]), START)

    def test_admitted_are_the_first_n(self):
        verdicts = LA.simulate_burst(
            ".claude-q-netra", 6, start=START,
            max_per_account=2, window_min=5, global_cap=50,
        )
        kinds = [v["verdict"] for v in verdicts]
        self.assertEqual(kinds, [LA.VERDICT_ADMIT, LA.VERDICT_ADMIT]
                         + [LA.VERDICT_DEFER] * 4)


class GlobalCapTest(unittest.TestCase):
    """Distinct accounts each under their own ceiling can still storm the fleet."""

    def test_global_cap_defers_even_a_fresh_account(self):
        # 4 prior launches across 4 DIFFERENT accounts, all inside the window.
        prior = [START + timedelta(seconds=10 * i) for i in range(4)]
        v = LA.admit(
            ".claude-fresh-netra",
            now=START + timedelta(seconds=50),
            account_launches=[],          # this account has launched 0 times
            global_launches=prior,        # but the fleet is at the cap
            max_per_account=3, window_min=5, global_cap=4,
        )
        self.assertEqual(v["verdict"], LA.VERDICT_DEFER)
        self.assertEqual(v["reason"], LA.REASON_GLOBAL_CAP)
        self.assertEqual(v["retry_after"], LA._fmt(prior[0] + timedelta(minutes=5)))


class AdmitWhenClearTest(unittest.TestCase):
    def test_empty_ledger_admits(self):
        v = LA.admit(".claude-day24-netra", now=START,
                     account_launches=[], global_launches=[],
                     max_per_account=3, window_min=5, global_cap=10)
        self.assertEqual(v["verdict"], LA.VERDICT_ADMIT)
        self.assertIsNone(v["reason"])

    def test_window_rolls_off_old_launches(self):
        # 5 launches, but all OLDER than the 5-minute window -> no pressure now.
        old = [START - timedelta(minutes=10) - timedelta(seconds=i) for i in range(5)]
        v = LA.admit(".claude-q-netra", now=START,
                     account_launches=old, global_launches=old,
                     max_per_account=3, window_min=5, global_cap=10)
        self.assertEqual(v["verdict"], LA.VERDICT_ADMIT)


class AccountTagTest(unittest.TestCase):
    def test_dir_tag_and_bare_tag_match(self):
        self.assertEqual(LA._account_tag(".claude-q-netra"), "q-netra")
        self.assertEqual(LA._account_tag("q-netra"), "q-netra")
        self.assertEqual(LA._account_tag(".claude"), "default")
        self.assertEqual(LA._account_tag("opencode-glm"), "glm")
        self.assertEqual(LA._account_tag("C:/Users/u/.claude-q-netra"), "q-netra")


class LedgerRoundTripTest(unittest.TestCase):
    """The live CLI path: count launches from the durable ledger, append on ADMIT."""

    def test_load_counts_only_this_account_and_skips_deferrals(self):
        with tempfile.TemporaryDirectory() as d:
            led = os.path.join(d, "resume_ledger.jsonl")
            rows = [
                {"ts": "2026-06-24T22:04:03Z", "resume_account": ".claude-q-netra"},
                {"ts": "2026-06-24T22:04:07Z", "resume_account": ".claude-q-netra"},
                {"ts": "2026-06-24T22:04:11Z", "account": ".claude-day24-netra"},
                # a deferral is NOT launch pressure -> must be skipped.
                {"ts": "2026-06-24T22:04:15Z", "resume_account": ".claude-q-netra",
                 "phase": "deferred"},
                "{ this is not json }",  # malformed -> skipped, never crashes
            ]
            with open(led, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
            acct, glob = LA.load_launches(led, ".claude-q-netra")
            self.assertEqual(len(acct), 2)   # two real q-netra launches
            self.assertEqual(len(glob), 3)   # + the day24 launch; deferral skipped

    def test_missing_ledger_is_empty_not_error(self):
        acct, glob = LA.load_launches("/no/such/ledger.jsonl", ".claude-q-netra")
        self.assertEqual((acct, glob), ([], []))

    def test_record_then_reload_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            led = os.path.join(d, "resume_ledger.jsonl")
            LA.record_launch(led, ".claude-q-netra", START, cause="test")
            acct, glob = LA.load_launches(led, ".claude-q-netra")
            self.assertEqual(len(acct), 1)
            self.assertEqual(acct[0], START)


class VocabularyTest(unittest.TestCase):
    def test_emittable_reasons_are_the_three_documented_tokens(self):
        self.assertEqual(
            set(LA.EMITTABLE_REASONS),
            {LA.REASON_THROTTLED, LA.REASON_GLOBAL_CAP, LA.REASON_RATE},
        )


if __name__ == "__main__":
    unittest.main()
