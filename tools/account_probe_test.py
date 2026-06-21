#!/usr/bin/env python3
"""Hermetic tests for active account probes.

No real ``claude`` is ever spawned: a fake runner returns canned
(exit_code, stdout, stderr, timed_out, spawn_error) tuples. The round-trip tests
drive the REAL fleet_sessions mergers so we prove a probe verdict actually clears /
sets the carry-forward blocker latch.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import account_probe  # noqa: E402
import fleet_accounts  # noqa: E402
import fleet_sessions  # noqa: E402


# Real banner strings observed live (2026-06-20), so the classifier is tested against
# exactly what the CLI emits, not a paraphrase.
BANNER_ACCESS = ("Your organization has disabled Claude subscription access for "
                 "Claude Code · Use an Anthropic API key instead, or ask your "
                 "admin to enable access")
BANNER_LOGIN = "Not logged in · Please run /login"
BANNER_401 = ("Please run /login · API Error: 401 Invalid authentication "
              "credentials")
BANNER_LIMIT = "You've hit your session limit · resets 7:50am (America/Los_Angeles)"
BANNER_CREDIT = "Your credit balance is too low to access the Anthropic API"


def runner_returning(exit_code=0, stdout="", stderr="", timed_out=False, spawn_error=""):
    """Build a fake runner that ignores argv and returns a fixed result."""
    def _run(argv, *, config_dir, timeout):
        return (exit_code, stdout, stderr, timed_out, spawn_error)
    return _run


def worker_row(account=".claude-gem8-acct", tag="gem8"):
    return {"dir": f"C:/Users/USER/{account}", "product": "claude",
            "account": account, "tag": tag, "kind": "worker"}


class ClassifyProbeOutputTest(unittest.TestCase):
    def test_clean_answer_is_ok(self) -> None:
        v = account_probe.classify_probe_output(0, "pong", "")
        self.assertEqual(v["status"], "OK")
        self.assertIsNone(v["block_kind"])

    def test_org_access_disabled_is_access_not_auth(self) -> None:
        # The exact stale-roster correction: a /login-shaped banner that is really an
        # org access wall must classify as ACCESS, not AUTH.
        v = account_probe.classify_probe_output(1, "", BANNER_ACCESS)
        self.assertEqual(v["status"], "ACCESS")
        self.assertEqual(v["block_kind"], "access")

    def test_not_logged_in_is_auth(self) -> None:
        v = account_probe.classify_probe_output(1, "", BANNER_LOGIN)
        self.assertEqual(v["status"], "AUTH")
        self.assertEqual(v["block_kind"], "auth")

    def test_401_is_auth(self) -> None:
        v = account_probe.classify_probe_output(1, "", BANNER_401)
        self.assertEqual(v["status"], "AUTH")

    def test_credit_balance_is_credit(self) -> None:
        v = account_probe.classify_probe_output(1, "", BANNER_CREDIT)
        self.assertEqual(v["status"], "CREDIT")
        self.assertEqual(v["block_kind"], "credit")

    def test_session_limit_carries_reset(self) -> None:
        v = account_probe.classify_probe_output(1, "", BANNER_LIMIT)
        self.assertEqual(v["status"], "LIMIT")
        self.assertEqual(v["block_kind"], "usage")
        self.assertIn("7:50am", v["reset"])

    def test_timeout_is_transport(self) -> None:
        v = account_probe.classify_probe_output(124, "", "", timed_out=True)
        self.assertEqual(v["status"], "TRANSPORT")

    def test_spawn_error_is_transport(self) -> None:
        v = account_probe.classify_probe_output(127, "", "", spawn_error="no such file")
        self.assertEqual(v["status"], "TRANSPORT")

    def test_banner_on_stdout_not_mistaken_for_success(self) -> None:
        # A zero exit with a block banner on stdout must NOT read as OK.
        v = account_probe.classify_probe_output(0, BANNER_LIMIT, "")
        self.assertEqual(v["status"], "LIMIT")


class ProbeAccountTest(unittest.TestCase):
    def test_probe_account_shapes_verdict(self) -> None:
        v = account_probe.probe_account(
            worker_row(), runner=runner_returning(0, "pong", ""))
        self.assertEqual(v["status"], "OK")
        self.assertEqual(v["account"], ".claude-gem8-acct")
        self.assertEqual(v["tag"], "gem8")
        self.assertIn("probed_utc", v)
        self.assertGreaterEqual(v["latency_ms"], 0)

    def test_probe_account_access_wall(self) -> None:
        v = account_probe.probe_account(
            worker_row(".claude-gem5-acct", "gem5"),
            runner=runner_returning(1, "", BANNER_ACCESS))
        self.assertEqual(v["status"], "ACCESS")
        self.assertIn("disabled", v["block_reason"].lower())


class VerdictToRowTest(unittest.TestCase):
    def test_ok_becomes_live_row(self) -> None:
        v = account_probe.probe_account(
            worker_row(), runner=runner_returning(0, "pong", ""))
        row = account_probe.verdict_to_row(v)
        self.assertEqual(row["disp"], "LIVE")
        self.assertEqual(row["age_min"], 0.0)
        self.assertFalse(row["throttle_current"])

    def test_limit_becomes_throttle_row(self) -> None:
        v = account_probe.probe_account(
            worker_row(), runner=runner_returning(1, "", BANNER_LIMIT))
        row = account_probe.verdict_to_row(v)
        self.assertEqual(row["disp"], "STOPPED_LIMIT")
        self.assertTrue(row["throttle_current"])
        self.assertIn("7:50am", row["throttle_reset"])

    def test_auth_becomes_infra_auth_row(self) -> None:
        v = account_probe.probe_account(
            worker_row(".claude-faklocal", "faklocal"),
            runner=runner_returning(1, "", BANNER_LOGIN))
        row = account_probe.verdict_to_row(v)
        self.assertEqual(row["disp"], "INFRA_AUTH")


class MergeRoundTripTest(unittest.TestCase):
    """Prove probe rows actually move the carry-forward latch in the real mergers."""

    def _ok_probe_row(self, account, tag):
        v = account_probe.probe_account(
            worker_row(account, tag), runner=runner_returning(0, "pong", ""))
        return account_probe.verdict_to_row(v)

    def test_fresh_ok_probe_clears_stale_auth(self) -> None:
        account = ".claude-gem5-acct"
        # prior registry: a 2-day-old auth blocker on this account
        prev_reg = {
            "generated_utc": "2026-06-18T07:39:17+00:00",
            "auth": {account: {"block_kind": "auth", "block_reason": "auth/login required",
                               "seen_utc": "2026-06-18T07:39:17+00:00"}},
            "throttle": {},
        }
        rows = [self._ok_probe_row(account, "gem5")]
        with mock.patch.object(fleet_accounts, "load_registry", return_value=prev_reg):
            merged = fleet_sessions.merge_known_auth(rows)
        self.assertNotIn(account, merged, "fresh OK probe should clear the stale auth latch")

    def test_fresh_ok_probe_clears_stale_throttle(self) -> None:
        account = ".claude-gem8-acct"
        prev_reg = {
            "generated_utc": "2026-06-18T07:39:17+00:00",
            "auth": {},
            "throttle": {account: {"reset": "Jun 24, 8pm (America/Los_Angeles)"}},
        }
        rows = [self._ok_probe_row(account, "gem8")]
        with mock.patch.object(fleet_accounts, "load_registry", return_value=prev_reg):
            merged = fleet_sessions.merge_known_throttle({}, rows)
        self.assertNotIn(account, merged, "fresh OK probe should clear the stale throttle")

    def test_fresh_auth_probe_sets_blocker(self) -> None:
        account = ".claude-faklocal"
        prev_reg = {"generated_utc": "2026-06-18T00:00:00+00:00", "auth": {}, "throttle": {}}
        v = account_probe.probe_account(
            worker_row(account, "faklocal"), runner=runner_returning(1, "", BANNER_LOGIN))
        rows = [account_probe.verdict_to_row(v)]
        with mock.patch.object(fleet_accounts, "load_registry", return_value=prev_reg):
            merged = fleet_sessions.merge_known_auth(rows)
        self.assertIn(account, merged)
        self.assertEqual(merged[account]["block_kind"], "auth")

    def test_fresh_limit_probe_sets_throttle(self) -> None:
        account = ".claude-gem8-acct"
        prev_reg = {"generated_utc": "2026-06-18T00:00:00+00:00", "auth": {}, "throttle": {}}
        v = account_probe.probe_account(
            worker_row(account, "gem8"), runner=runner_returning(1, "", BANNER_LIMIT))
        row = account_probe.verdict_to_row(v)
        throttle = {account: {"reset": row["throttle_reset"]}}
        with mock.patch.object(fleet_accounts, "load_registry", return_value=prev_reg):
            merged = fleet_sessions.merge_known_throttle(throttle, rows=[row])
        self.assertIn(account, merged)


class SelectTargetsTest(unittest.TestCase):
    def _annotated(self):
        return [
            {"kind": "worker", "account": ".claude", "tag": "default", "available": True,
             "active_sessions": 5, "live_sessions": 2, "block_kind": None},
            {"kind": "worker", "account": ".claude-gem5-acct", "tag": "gem5",
             "available": False, "active_sessions": 0, "live_sessions": 0,
             "block_kind": "auth", "throttled": False},
            {"kind": "worker", "account": ".claude-gem8-acct", "tag": "gem8",
             "available": False, "active_sessions": 0, "live_sessions": 0,
             "block_kind": "usage", "throttled": True,
             "reset": "Jun 24, 8pm (America/Los_Angeles)"},
            {"kind": "excluded", "account": ".claude-adminbackup-acct", "tag": "adminbackup",
             "available": False},
        ]

    def test_blocked_selector_skips_available_and_excluded(self) -> None:
        targets = account_probe.select_targets(self._annotated(), selector="blocked")
        tags = {t["tag"] for t in targets}
        self.assertEqual(tags, {"gem5", "gem8"})
        self.assertNotIn("adminbackup", tags)
        self.assertNotIn("default", tags)

    def test_never_probes_tombstoned_account(self) -> None:
        targets = account_probe.select_targets(self._annotated(), selector="all")
        self.assertNotIn("adminbackup", {t["tag"] for t in targets})

    def test_skip_active_throttle_drops_future_reset_but_keeps_auth(self) -> None:
        targets = account_probe.select_targets(
            self._annotated(), selector="blocked", skip_active_throttle=True)
        tags = {t["tag"] for t in targets}
        self.assertIn("gem5", tags, "auth blockers are always probed")
        self.assertNotIn("gem8", tags, "a still-future throttle is skipped")

    def test_account_filter_overrides_selector(self) -> None:
        targets = account_probe.select_targets(
            self._annotated(), selector="blocked", account="default")
        self.assertEqual([t["tag"] for t in targets], ["default"])


class ProbeAccountsBatchTest(unittest.TestCase):
    def test_batch_probes_all_targets(self) -> None:
        targets = [worker_row(".claude-gem5-acct", "gem5"),
                   worker_row(".claude-gem8-acct", "gem8")]
        verdicts = account_probe.probe_accounts(
            targets, runner=runner_returning(0, "pong", ""), max_workers=2)
        self.assertEqual(len(verdicts), 2)
        self.assertTrue(all(v["status"] == "OK" for v in verdicts))

    def test_batch_isolates_a_raising_probe(self) -> None:
        def boom(argv, *, config_dir, timeout):
            raise RuntimeError("kaboom")
        verdicts = account_probe.probe_accounts(
            [worker_row()], runner=boom, max_workers=1)
        self.assertEqual(len(verdicts), 1)
        self.assertEqual(verdicts[0]["status"], "TRANSPORT")


class ProbeLedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.rd = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def _verdict(self, account, tag, status):
        runner = {
            "OK": runner_returning(0, "pong", ""),
            "AUTH": runner_returning(1, "", BANNER_LOGIN),
            "ACCESS": runner_returning(1, "", BANNER_ACCESS),
            "LIMIT": runner_returning(1, "", BANNER_LIMIT),
        }[status]
        return account_probe.probe_account(worker_row(account, tag), runner=runner)

    def test_ledger_appends_and_reads_back(self) -> None:
        v = self._verdict(".claude-gem5-acct", "gem5", "ACCESS")
        recs = account_probe.append_probe_ledger([v], self.rd)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["status"], "ACCESS")
        self.assertIsNone(recs[0]["prev_status"])
        self.assertFalse(recs[0]["flip"])
        # second read sees it as the latest
        latest = account_probe.last_probe_by_account(self.rd)
        self.assertEqual(latest[".claude-gem5-acct"]["status"], "ACCESS")

    def test_flip_detected_across_two_probes(self) -> None:
        account_probe.append_probe_ledger(
            [self._verdict(".claude-agent", "agent", "ACCESS")], self.rd)
        recs = account_probe.append_probe_ledger(
            [self._verdict(".claude-agent", "agent", "OK")], self.rd)
        self.assertEqual(recs[0]["prev_status"], "ACCESS")
        self.assertEqual(recs[0]["status"], "OK")
        self.assertTrue(recs[0]["flip"])

    def test_no_flip_when_status_unchanged(self) -> None:
        account_probe.append_probe_ledger(
            [self._verdict(".claude-gem5-acct", "gem5", "ACCESS")], self.rd)
        recs = account_probe.append_probe_ledger(
            [self._verdict(".claude-gem5-acct", "gem5", "ACCESS")], self.rd)
        self.assertEqual(recs[0]["prev_status"], "ACCESS")
        self.assertFalse(recs[0]["flip"])

    def test_min_interval_skips_recently_probed(self) -> None:
        account = ".claude-gem8-acct"
        account_probe.append_probe_ledger(
            [self._verdict(account, "gem8", "LIMIT")], self.rd)
        annotated = [{"kind": "worker", "account": account, "tag": "gem8",
                      "available": False, "block_kind": "usage"}]
        # just probed -> a 999-min floor must skip it
        targets = account_probe.select_targets(
            annotated, selector="blocked", min_interval_min=999, reg_dir_path=self.rd)
        self.assertEqual(targets, [])
        # a 0 floor probes it
        targets = account_probe.select_targets(
            annotated, selector="blocked", min_interval_min=0, reg_dir_path=self.rd)
        self.assertEqual(len(targets), 1)

    def test_account_filter_ignores_min_interval(self) -> None:
        # an explicit single-account probe is always honored, even if just probed
        account = ".claude-gem8-acct"
        account_probe.append_probe_ledger(
            [self._verdict(account, "gem8", "LIMIT")], self.rd)
        annotated = [{"kind": "worker", "account": account, "tag": "gem8",
                      "available": False}]
        targets = account_probe.select_targets(
            annotated, account="gem8", min_interval_min=999, reg_dir_path=self.rd)
        self.assertEqual(len(targets), 1)


class StdinNoiseTest(unittest.TestCase):
    def test_stdin_warning_only_is_not_success(self) -> None:
        warn = "Warning: no stdin data received in 3s, proceeding without it."
        v = account_probe.classify_probe_output(0, warn, "")
        self.assertEqual(v["status"], "TRANSPORT")

    def test_stdin_warning_plus_real_answer_is_ok(self) -> None:
        out = "Warning: no stdin data received in 3s, proceeding without it.\npong"
        v = account_probe.classify_probe_output(0, out, "")
        self.assertEqual(v["status"], "OK")


class SummaryTest(unittest.TestCase):
    def test_summary_counts_and_all_blocked(self) -> None:
        verdicts = [{"status": "AUTH"}, {"status": "ACCESS"}, {"status": "LIMIT"}]
        s = account_probe.summarize(verdicts)
        self.assertEqual(s["probed"], 3)
        self.assertEqual(s["ok"], 0)
        self.assertTrue(s["all_blocked"])
        self.assertIn("auth=1", s["line"])

    def test_summary_ok_clears_all_blocked(self) -> None:
        s = account_probe.summarize([{"status": "OK"}, {"status": "AUTH"}])
        self.assertFalse(s["all_blocked"])
        self.assertEqual(s["ok"], 1)

    def test_summary_counts_flips(self) -> None:
        recs = [{"flip": True}, {"flip": False}]
        s = account_probe.summarize([{"status": "OK"}, {"status": "OK"}], recs)
        self.assertEqual(s["flips"], 1)
        self.assertIn("flips=1", s["line"])


if __name__ == "__main__":
    unittest.main()
