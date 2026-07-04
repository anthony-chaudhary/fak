#!/usr/bin/env python3
"""FAK_SESSIONS_PER_ACCOUNT one-knob-one-way override for _session_cap.

Mirrors internal/fleetaccounts/sessioncap_env_test.go: the same env var retunes the
Claude per-account session budget in both languages so Go and Python never drift.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_accounts  # noqa: E402


class SessionCapEnvKnobTest(unittest.TestCase):
    def test_default_when_unset(self):
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop(fleet_accounts.SESSIONS_PER_ACCOUNT_ENV, None)
            self.assertEqual(
                fleet_accounts._session_cap({"product": "claude"}),
                fleet_accounts.DEFAULT_CLAUDE_ACCOUNT_SESSION_CAP,
            )

    def test_override_widens_claude_only(self):
        with mock.patch.dict(os.environ,
                             {fleet_accounts.SESSIONS_PER_ACCOUNT_ENV: "7"}):
            self.assertEqual(fleet_accounts._session_cap({"product": "claude"}), 7)
            # non-Claude products ignore the knob.
            self.assertEqual(
                fleet_accounts._session_cap({"product": "opencode"}),
                fleet_accounts.DEFAULT_ACCOUNT_SESSION_CAP,
            )

    def test_bad_value_keeps_default(self):
        for bad in ("0", "-3", "notanint", ""):
            with mock.patch.dict(os.environ,
                                 {fleet_accounts.SESSIONS_PER_ACCOUNT_ENV: bad}):
                self.assertEqual(
                    fleet_accounts._session_cap({"product": "claude"}),
                    fleet_accounts.DEFAULT_CLAUDE_ACCOUNT_SESSION_CAP,
                    f"bad value {bad!r} should keep the default",
                )


if __name__ == "__main__":
    unittest.main()
