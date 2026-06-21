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


if __name__ == "__main__":
    unittest.main(verbosity=2)
