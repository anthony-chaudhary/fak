#!/usr/bin/env python3
"""Hermetic tests for active account probes.

No real ``claude`` is ever spawned: a fake runner returns canned
(exit_code, stdout, stderr, timed_out, spawn_error) tuples. The round-trip tests
drive the REAL fleet_sessions mergers so we prove a probe verdict actually clears /
sets the carry-forward blocker latch.
"""
from __future__ import annotations

import os
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

    def test_access_wall_text_with_a_reset_is_a_recovering_cap_not_a_wall(self) -> None:
        # The load-bearing overage case: a banner that carries BOTH the org-disable wording
        # (which the ACCESS wall regex matches) AND a reset window is a self-recovering usage
        # cap, not a permanent wall. LIMIT must win over ACCESS so the seat is not marked
        # permanently STOPPED and a human is not paged for a cap that comes back at its reset.
        banner = ("Your organization has disabled Claude subscription access · "
                  "You've hit your session limit · resets 7:50am (America/Los_Angeles)")
        v = account_probe.classify_probe_output(1, "", banner)
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


class DefaultRunnerEnvTest(unittest.TestCase):
    def test_default_runner_pins_config_dir_and_clears_ambient_token(self) -> None:
        class Proc:
            returncode = 0
            stdout = "pong"
            stderr = ""

        with mock.patch.dict(os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": "ambient"},
                             clear=False):
            with mock.patch.object(account_probe.subprocess, "run", return_value=Proc()
                                   ) as run:
                code, stdout, stderr, timed_out, spawn_error = account_probe._default_runner(
                    ["claude", "-p", "say pong"], config_dir="C:/acct/day24",
                    timeout=3.0)

        self.assertEqual((code, stdout, stderr, timed_out, spawn_error),
                         (0, "pong", "", False, ""))
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["CLAUDE_CONFIG_DIR"], "C:/acct/day24")
        self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", env)


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
        # The probe banner's bare reset ("7:50am") is clock-relative -- expired in the
        # afternoon -- and merge_known_throttle drops EXPIRED throttles. This test is
        # about carry-forward of a LIVE limit, so pin the reset to a clearly-future
        # dated window; the bare-banner parse itself is covered by the classify tests.
        live_reset = "Dec 31, 11pm (America/Los_Angeles)"
        row["throttle_reset"] = live_reset
        throttle = {account: {"reset": live_reset}}
        with mock.patch.object(fleet_accounts, "load_registry", return_value=prev_reg):
            merged = fleet_sessions.merge_known_throttle(throttle, rows=[row])
        self.assertIn(account, merged)


class SelectTargetsTest(unittest.TestCase):
    def _annotated(self):
        # gem8's reset must be UNAMBIGUOUSLY in the future for skip_active_throttle to drop
        # it -- a hardcoded calendar date ("Jun 24, 8pm") silently expires and flips this
        # test red once that wall-clock moment passes. Derive it ~3 days ahead of now in the
        # dated "%b %d, %I%p" shape _reset_is_future parses, so the fixture stays future-valid.
        future_reset = self._future_reset_str(days=3)
        return [
            {"kind": "worker", "account": ".claude", "tag": "default", "available": True,
             "active_sessions": 5, "live_sessions": 2, "block_kind": None},
            {"kind": "worker", "account": ".claude-gem5-acct", "tag": "gem5",
             "available": False, "active_sessions": 0, "live_sessions": 0,
             "block_kind": "auth", "throttled": False},
            {"kind": "worker", "account": ".claude-gem8-acct", "tag": "gem8",
             "available": False, "active_sessions": 0, "live_sessions": 0,
             "block_kind": "usage", "throttled": True,
             "reset": future_reset},
            {"kind": "excluded", "account": ".claude-adminbackup-acct", "tag": "adminbackup",
             "available": False},
        ]

    @staticmethod
    def _future_reset_str(days: int = 3) -> str:
        import datetime as _dt
        # Dated "%b %d, %I%p" -- _reset_is_future parses the zero-padded form ("Jun 27, 08PM")
        # directly, so no platform-specific %-d/%-I munging is needed.
        when = _dt.datetime.now() + _dt.timedelta(days=days)
        return when.strftime("%b %d, %I%p") + " (America/Los_Angeles)"

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

    def test_selector_probes_opencode_but_not_other_products(self) -> None:
        # opencode rows ARE now selector-probed: probe_account routes them to the
        # gateway-aware probe_opencode_account (which pings the guard base URL its
        # worker uses), not the claude `-p pong` surface, so there is no bogus AUTH
        # block to fear. A product with NO probe implementation (codex) stays skipped,
        # and the explicit single-`account` override remains operator-honored.
        # available=False (a stale carried block, the realistic Defect-3 case) so all
        # three selectors have grounds to pick a supported row.
        rows = self._annotated() + [
            {"kind": "worker", "account": "opencode-nim-x", "tag": "nim-x",
             "product": "opencode", "available": False,
             "active_sessions": 0, "live_sessions": 0, "block_kind": "auth"},
            {"kind": "worker", "account": "codex-y", "tag": "codex-y",
             "product": "codex", "available": False,
             "active_sessions": 0, "live_sessions": 0, "block_kind": "auth"},
        ]
        for selector in ("blocked", "stale", "all"):
            tags = {t["tag"] for t in account_probe.select_targets(rows, selector=selector)}
            self.assertIn("nim-x", tags, f"selector={selector} must probe opencode rows")
            self.assertNotIn("codex-y", tags,
                             f"selector={selector} must skip unsupported codex rows")


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


class ProbeLedgerCooldownTest(unittest.TestCase):
    """#1801: a throttle cooldown ledger record carries cooldown start, reason, and the
    next-eligible windows, so a roster fold renders the throttled seat unavailable using
    the RECORDED reason -- not a reconstructed guess."""

    # A live banner carrying BOTH a session (daily) window and a weekly one, so the
    # ledger record exercises reset AND weekly. fleet_session_signals.limit_resets keys
    # "weekly" off the ~24 chars before each "limit ... resets", so the second occurrence
    # ("Weekly limit resets ...") lands in the weekly slot.
    BANNER = ("You've hit your session limit · resets 7:50am (America/Los_Angeles). "
              "Weekly limit resets Jul 10, 9am (America/Los_Angeles).")

    def _limit_verdict(self, account=".claude-gem8-acct", tag="gem8"):
        return account_probe.probe_account(
            worker_row(account, tag), runner=runner_returning(1, "", self.BANNER))

    def test_limit_ledger_records_cooldown_start_reason_and_windows(self) -> None:
        with tempfile.TemporaryDirectory() as rd:
            recs = account_probe.append_probe_ledger([self._limit_verdict()], rd)
            self.assertEqual(len(recs), 1)
            rec = recs[0]
            self.assertEqual(rec["status"], "LIMIT")
            self.assertEqual(rec["block_kind"], "usage")
            self.assertTrue(rec["ts"], "cooldown start (probe-observation ts) must be stamped")
            self.assertIn("usage limit", rec["block_reason"], "the human reason, not just a kind")
            self.assertIn("7:50am", rec["reset"], "daily next-eligible window recorded")
            self.assertIn("Jul 10", rec["weekly"], "weekly next-eligible window recorded")
            # Re-read from disk: the fields persist for the roster fold, not just in-memory.
            latest = account_probe.last_probe_by_account(rd)[".claude-gem8-acct"]
            self.assertIn("usage limit", latest["block_reason"])
            self.assertIn("Jul 10", latest["weekly"])

    def test_throttled_seat_excluded_from_capacity_without_guessing(self) -> None:
        # The witness: the recorded cooldown renders the seat NOT available, and the
        # roster fold surfaces the RECORDED reason/windows instead of reconstructing them.
        import fleet_accounts  # noqa: PLC0415
        account = ".claude-gem8-acct"
        with tempfile.TemporaryDirectory() as rd:
            with mock.patch.dict(os.environ, {"FLEET_REG_DIR": rd}, clear=False):
                account_probe.append_probe_ledger([self._limit_verdict(account, "gem8")])
                fleet_accounts._PROBE_LEDGER_CACHE["key"] = None  # bust the mtime+size memo
                verdict = fleet_accounts._fresh_probe_from_ledger(account)
                recorded_start = account_probe.last_probe_by_account(rd)[account]["ts"]
        self.assertIsNotNone(verdict, "a fresh LIMIT probe must be visible to the roster fold")
        self.assertFalse(verdict["available"], "a throttled seat is not available capacity")
        self.assertEqual(verdict["block_kind"], "usage")
        self.assertIn("usage limit", verdict["block_reason"])
        self.assertIn("7:50am", verdict["reset"])
        self.assertIn("Jul 10", verdict["weekly"], "the recorded weekly window, not None")
        # #1801: the recorded cooldown START rides through to the roster verdict too, so
        # dispatch status can say WHY / WHEN-eligible AND since-when -- not just the first two.
        self.assertTrue(recorded_start, "append_probe_ledger stamps a cooldown start (ts)")
        self.assertEqual(verdict["since"], recorded_start,
                         "the roster verdict carries the recorded cooldown start, not None")


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


def opencode_row(account="opencode", tag="zai-coding-plan",
                 config_dir="C:/Users/USER/.config/opencode"):
    return {"dir": config_dir, "product": "opencode", "account": account,
            "tag": tag, "kind": "worker", "model": "zai-coding-plan/glm-4.5-air"}


def _resolver(base="http://127.0.0.1:18080/v1", model="zai-coding-plan/glm-4.5-air"):
    def _r(row, *, workspace=None, runs_dir=None):
        return model, base
    return _r


def _connector(result):
    def _c(base_url, *, timeout):
        return dict(result, _base=base_url)
    return _c


class ClassifyOpencodeProbeTest(unittest.TestCase):
    def test_unreachable_is_gateway_down(self) -> None:
        v = account_probe.classify_opencode_probe(
            {"reachable": False, "status": None, "body": "", "error": "connection refused"},
            base_url="http://127.0.0.1:18080/v1")
        self.assertEqual(v["status"], "GATEWAY_DOWN")
        self.assertEqual(v["block_kind"], "gateway")
        self.assertIn("connection refused", v["block_reason"])

    def test_2xx_is_ok(self) -> None:
        v = account_probe.classify_opencode_probe(
            {"reachable": True, "status": 200, "body": '{"data":[]}', "error": ""})
        self.assertEqual(v["status"], "OK")

    def test_429_is_limit(self) -> None:
        v = account_probe.classify_opencode_probe(
            {"reachable": True, "status": 429, "body": "Weekly Limit Exhausted; "
             "reset at 2026-07-15", "error": ""})
        self.assertEqual(v["status"], "LIMIT")
        self.assertEqual(v["reset"], "2026-07-15")

    def test_401_is_auth(self) -> None:
        v = account_probe.classify_opencode_probe(
            {"reachable": True, "status": 401, "body": "", "error": ""})
        self.assertEqual(v["status"], "AUTH")

    def test_relay_borrowed_401_is_retryable_not_auth(self) -> None:
        for body in (
                "blocked by upstream provider",
                "REQUEST BLOCKED BY UPSTREAM",
        ):
            with self.subTest(body=body):
                v = account_probe.classify_opencode_probe(
                    {"reachable": True, "status": 401, "body": body, "error": ""})
                self.assertEqual(v["status"], "APIERR")
                self.assertEqual(v["block_kind"], "apierr")

    def test_expired_token_401_remains_auth(self) -> None:
        v = account_probe.classify_opencode_probe(
            {"reachable": True, "status": 401,
             "body": "upstream relay rejected expired token", "error": ""})
        self.assertEqual(v["status"], "AUTH")

    def test_5xx_is_apierr(self) -> None:
        v = account_probe.classify_opencode_probe(
            {"reachable": True, "status": 503, "body": "", "error": ""})
        self.assertEqual(v["status"], "APIERR")

    def test_reachable_but_inconclusive_is_apierr_not_block(self) -> None:
        # A flaky models route must NEVER sideline the seat as a hard block.
        v = account_probe.classify_opencode_probe(
            {"reachable": True, "status": None, "body": "", "error": "read timeout"})
        self.assertEqual(v["status"], "APIERR")


class ProbeOpencodeAccountTest(unittest.TestCase):
    def test_gateway_down_verdict_shape(self) -> None:
        v = account_probe.probe_opencode_account(
            opencode_row(),
            connector=_connector({"reachable": False, "status": None, "body": "",
                                  "error": "refused"}),
            target_resolver=_resolver())
        self.assertEqual(v["status"], "GATEWAY_DOWN")
        self.assertEqual(v["product"], "opencode")
        self.assertEqual(v["base_url"], "http://127.0.0.1:18080/v1")
        self.assertEqual(v["tag"], "zai-coding-plan")

    def test_ok_verdict(self) -> None:
        v = account_probe.probe_opencode_account(
            opencode_row(),
            connector=_connector({"reachable": True, "status": 200, "body": "{}",
                                  "error": ""}),
            target_resolver=_resolver())
        self.assertEqual(v["status"], "OK")

    def test_no_base_url_is_transport_not_gateway_down(self) -> None:
        # An unconfigured base URL is a local config gap, not a down gateway.
        v = account_probe.probe_opencode_account(
            opencode_row(), connector=_connector({"reachable": True, "status": 200}),
            target_resolver=_resolver(base=""))
        self.assertEqual(v["status"], "TRANSPORT")

    def test_probe_account_routes_opencode(self) -> None:
        # The single choke point: probe_account must dispatch an opencode row to the
        # gateway probe, never the claude runner (which would stamp a bogus AUTH).
        v = account_probe.probe_account(
            opencode_row(),
            runner=runner_returning(1, "", BANNER_LOGIN),  # would AUTH if wrongly used
            connector=_connector({"reachable": False, "status": None, "error": "refused"}),
            target_resolver=_resolver())
        self.assertEqual(v["status"], "GATEWAY_DOWN")


class OpencodeVerdictRowTest(unittest.TestCase):
    def test_gateway_down_row_does_not_latch_auth(self) -> None:
        # A transient down gateway maps to a QUIET stopped disp, NOT an auth block, so
        # merge_known_auth never sidelines the seat on a local infra blip.
        v = account_probe.probe_opencode_account(
            opencode_row(),
            connector=_connector({"reachable": False, "status": None, "error": "refused"}),
            target_resolver=_resolver())
        row = account_probe.verdict_to_row(v)
        self.assertEqual(row["disp"], "STOPPED_QUIET")
        self.assertEqual(row["probe_status"], "GATEWAY_DOWN")
        self.assertNotIn(row["disp"], ("INFRA_AUTH",))


if __name__ == "__main__":
    unittest.main()
