#!/usr/bin/env python3
"""Hermetic tests for account_relogin (no real claude spawn -- injected runner)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import account_relogin  # noqa: E402


def status_runner(email, logged_in=True, sub="max"):
    """A fake `claude auth status --json` runner returning a fixed identity."""
    def _run(argv, *, config_dir, timeout):
        if argv[1:3] == ["auth", "status"]:
            return 0, json.dumps({"loggedIn": logged_in, "email": email,
                                  "orgId": "org", "subscriptionType": sub}), ""
        if argv[1:3] == ["auth", "logout"]:
            return 0, "logged out", ""
        return 0, "", ""
    return _run


ROWS = [
    {"product": "claude", "kind": "worker", "tag": "gem5", "dir": "C:/x/.claude-gem5-netra"},
    {"product": "claude", "kind": "worker", "tag": "gem8", "dir": "C:/x/.claude-gem8-netra"},
]


class ParseMapTest(unittest.TestCase):
    def test_csv_spec(self) -> None:
        self.assertEqual(account_relogin.parse_map("gem5=a@x.ai,gem7=b@x.ai"),
                         {"gem5": "a@x.ai", "gem7": "b@x.ai"})

    def test_ignores_blank_and_malformed(self) -> None:
        self.assertEqual(account_relogin.parse_map(" , gem5=a@x.ai , junk "),
                         {"gem5": "a@x.ai"})


class AssessTest(unittest.TestCase):
    def test_match_when_current_equals_intended(self) -> None:
        out = account_relogin.assess(
            {"gem5": "gem5@x.ai"}, ROWS, runner=status_runner("gem5@x.ai"))
        self.assertTrue(out[0]["match"])

    def test_mismatch_when_wrong_account(self) -> None:
        out = account_relogin.assess(
            {"gem5": "gem5@x.ai"}, ROWS, runner=status_runner("agent@x.ai"))
        self.assertFalse(out[0]["match"])
        self.assertIn("wrong account", out[0]["note"])

    def test_case_insensitive_match(self) -> None:
        out = account_relogin.assess(
            {"gem5": "GEM5@X.AI"}, ROWS, runner=status_runner("gem5@x.ai"))
        self.assertTrue(out[0]["match"])

    def test_missing_dir_flagged(self) -> None:
        out = account_relogin.assess(
            {"nope": "x@x.ai"}, ROWS, runner=status_runner("x@x.ai"))
        self.assertFalse(out[0]["match"])
        self.assertIsNone(out[0]["dir"])


class FixTest(unittest.TestCase):
    def test_fix_dry_run_does_not_logout(self) -> None:
        res = account_relogin.fix(
            {"gem5": "gem5@x.ai"}, ROWS, apply=False, runner=status_runner("agent@x.ai"))
        self.assertEqual(res["mismatched"], 1)
        self.assertIsNone(res["steps"][0]["logout"])  # dry-run never logs out

    def test_fix_apply_logs_out_mismatched(self) -> None:
        res = account_relogin.fix(
            {"gem5": "gem5@x.ai"}, ROWS, apply=True, runner=status_runner("agent@x.ai"))
        self.assertTrue(res["steps"][0]["logout"]["ok"])
        self.assertIn("auth login", res["steps"][0]["login_command"])

    def test_fix_skips_already_correct(self) -> None:
        res = account_relogin.fix(
            {"gem5": "gem5@x.ai"}, ROWS, apply=True, runner=status_runner("gem5@x.ai"))
        self.assertEqual(res["mismatched"], 0)
        self.assertEqual(res["steps"], [])


if __name__ == "__main__":
    unittest.main()
