#!/usr/bin/env python3
"""Tests for shared fleet transcript signal parsing."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_session_signals  # noqa: E402


class FleetSessionSignalsTest(unittest.TestCase):
    def test_login_401_is_auth_not_transient_api_error(self) -> None:
        text = "Please run /login \u00b7 API Error: 401 Invalid authentication credentials"

        self.assertTrue(fleet_session_signals.is_auth_error(text))
        self.assertFalse(fleet_session_signals.is_api_error(text))
        self.assertEqual(fleet_session_signals.auth_block_kind(text), "auth")

    def test_http_401_authentication_required_is_auth(self) -> None:
        text = "HTTP 401 Authentication required. Please provide an API key."

        self.assertTrue(fleet_session_signals.is_auth_error(text))

    def test_credit_block_is_classified_separately(self) -> None:
        text = "Credit balance is too low"

        self.assertEqual(fleet_session_signals.auth_block_kind(text), "credit")
        self.assertEqual(fleet_session_signals.auth_block_reason(text), "credit balance too low")

    def test_subscription_access_wall_is_not_login_prompt(self) -> None:
        text = (
            "Your organization has disabled Claude subscription access for Claude Code "
            "\u00b7 Use an Anthropic API key instead, or ask your admin to enable access"
        )

        self.assertTrue(fleet_session_signals.is_auth_error(text))
        self.assertEqual(fleet_session_signals.auth_block_kind(text), "access")
        self.assertEqual(
            fleet_session_signals.auth_block_reason(text),
            "Claude subscription access disabled",
        )
        self.assertFalse(fleet_session_signals.needs_login_prompt(text))

    def test_limit_reset_keeps_timezone_suffix(self) -> None:
        text = "You've hit your limit . resets 12:10am (America/Los_Angeles)\n<failures>"

        self.assertEqual(
            fleet_session_signals.limit_reset(text),
            "12:10am (America/Los_Angeles)",
        )

    def test_limit_resets_captures_daily_and_weekly_windows(self) -> None:
        text = (
            "You've hit your limit . resets 6pm (America/Los_Angeles)\n"
            "Your weekly limit . resets Mon Jun 23 at 9am (America/Los_Angeles)"
        )

        windows = fleet_session_signals.limit_resets(text)
        self.assertEqual(windows["daily"], "6pm (America/Los_Angeles)")
        self.assertEqual(windows["weekly"], "Mon Jun 23 at 9am (America/Los_Angeles)")

    def test_limit_reset_falls_back_to_weekly_when_only_weekly_present(self) -> None:
        # A weekly-only banner must still yield a primary reset so throttle
        # detection fires (a weekly cap blocks the account too).
        text = "Your weekly limit . resets Mon Jun 23"

        self.assertEqual(fleet_session_signals.limit_reset(text), "Mon Jun 23")
        windows = fleet_session_signals.limit_resets(text)
        self.assertEqual(windows["weekly"], "Mon Jun 23")
        self.assertNotIn("daily", windows)

    def test_limit_resets_daily_only_when_no_weekly(self) -> None:
        text = "You've hit your limit . resets 12:10am (America/Los_Angeles)"

        self.assertEqual(
            fleet_session_signals.limit_resets(text),
            {"daily": "12:10am (America/Los_Angeles)"},
        )


class ResetPassedTest(unittest.TestCase):
    """reset_passed -- the past/future verdict on a usage-limit window. Anchored to
    the banner's own time so 'resets 6am' written at 06:30 means TOMORROW's 6am."""

    def _utc(self, h, m=0):
        from datetime import datetime, timezone
        return datetime(2026, 6, 23, h, m, tzinfo=timezone.utc)

    def test_future_reset_not_passed(self) -> None:
        # banner written 09:00 PDT (16:00 UTC), resets 11am; now 10:18 PDT (17:18 UTC)
        anchor = self._utc(16, 0)
        now = self._utc(17, 18)
        self.assertFalse(fleet_session_signals.reset_passed(
            "11am (America/Los_Angeles)", now_utc=now, anchor_utc=anchor))

    def test_past_reset_passed(self) -> None:
        # same banner, now 11:05 PDT (18:05 UTC) -> elapsed
        anchor = self._utc(16, 0)
        now = self._utc(18, 5)
        self.assertTrue(fleet_session_signals.reset_passed(
            "11am (America/Los_Angeles)", now_utc=now, anchor_utc=anchor))

    def test_reset_with_minutes(self) -> None:
        # banner 06:00 PDT (13:00 UTC) resets 7:10am; now 08:20 PDT (15:20 UTC) -> passed
        anchor = self._utc(13, 0)
        self.assertTrue(fleet_session_signals.reset_passed(
            "7:10am (America/Los_Angeles)", now_utc=self._utc(15, 20), anchor_utc=anchor))
        # now 07:00 PDT (14:00 UTC) -> not yet
        self.assertFalse(fleet_session_signals.reset_passed(
            "7:10am (America/Los_Angeles)", now_utc=self._utc(14, 0), anchor_utc=anchor))

    def test_banner_after_reset_time_rolls_to_next_day(self) -> None:
        # banner written 06:30 PDT (13:30 UTC) saying 'resets 6am' => tomorrow 6am;
        # now same-day 08:20 PDT is BEFORE tomorrow's 6am -> not passed.
        anchor = self._utc(13, 30)
        self.assertFalse(fleet_session_signals.reset_passed(
            "6am (America/Los_Angeles)", now_utc=self._utc(15, 20), anchor_utc=anchor))

    def test_pm_window(self) -> None:
        anchor = self._utc(19, 0)  # 12:00 PDT
        self.assertFalse(fleet_session_signals.reset_passed(
            "3pm (America/Los_Angeles)", now_utc=self._utc(21, 0), anchor_utc=anchor))  # 14:00 PDT
        self.assertTrue(fleet_session_signals.reset_passed(
            "3pm (America/Los_Angeles)", now_utc=self._utc(22, 30), anchor_utc=anchor))  # 15:30 PDT

    def test_unparseable_returns_none(self) -> None:
        self.assertIsNone(fleet_session_signals.reset_passed("sometime soon"))
        self.assertIsNone(fleet_session_signals.reset_passed(""))

    def test_real_banner_string_parses(self) -> None:
        when = fleet_session_signals.limit_reset(
            "You've hit your session limit · resets 11am (America/Los_Angeles)")
        self.assertEqual(when, "11am (America/Los_Angeles)")
        self.assertIsNotNone(
            fleet_session_signals.reset_passed(when, now_utc=self._utc(20, 0),
                                               anchor_utc=self._utc(16, 0)))


class TerminalFailureTest(unittest.TestCase):
    """terminal_failure -- the shared failure taxonomy, keyed off the ERROR record only.
    The discipline that stops a session being bucketed by what it *narrated*."""

    def test_auth_error_record(self) -> None:
        kind, detail = fleet_session_signals.terminal_failure(
            "Not logged in · Please run /login")
        self.assertEqual(kind, "AUTH")
        self.assertEqual(detail, "auth/login required")

    def test_limit_error_record_carries_reset_window(self) -> None:
        kind, detail = fleet_session_signals.terminal_failure(
            "You've hit your session limit · resets 11am (America/Los_Angeles)")
        self.assertEqual(kind, "LIMIT")
        self.assertEqual(detail, "11am (America/Los_Angeles)")

    def test_transient_529_is_api_err(self) -> None:
        kind, detail = fleet_session_signals.terminal_failure(
            "API Error: Server is temporarily limiting requests "
            "(not your usage limit) · Rate limited")
        self.assertEqual(kind, "API_ERR")
        self.assertEqual(detail, "")

    def test_auth_outranks_limit_and_api(self) -> None:
        # precedence is by remediation cost: a record carrying both an auth wall and a
        # transient blip is the auth wall (the expensive one), never masked by the blip.
        kind, _ = fleet_session_signals.terminal_failure(
            "API Error: 401 Invalid authentication credentials · overloaded")
        self.assertEqual(kind, "AUTH")

    def test_empty_error_text_is_no_failure(self) -> None:
        # No error record => no failure bucket. Never an inference from prose.
        self.assertEqual(fleet_session_signals.terminal_failure(""), ("", ""))
        self.assertEqual(fleet_session_signals.terminal_failure("   "), ("", ""))

    def test_prose_about_auth_is_not_a_failure_when_passed_as_error(self) -> None:
        # The 732edb34 shape: a worker narrating auth work. If this STRING were ever the
        # error record it'd classify, but it is plain status prose with no error token,
        # so the taxonomy returns no failure -- the caller must only feed it the error
        # record, and even then mere discussion ("logged back in") is not a wall.
        kind, _ = fleet_session_signals.terminal_failure(
            "Closed the gem7 auth wall; gem7-netra is logged back in and the "
            "auth-failure blind spot is now a tested monitor signal.")
        self.assertEqual(kind, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
