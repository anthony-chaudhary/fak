#!/usr/bin/env python3
"""Hermetic tests for the cross-account re-home decision in fleet_sessions.

These cover the exact gap that left throttled sessions "pinned" to a rate-limited
account: a resumable autonomous session whose owner is throttled must be re-homed
onto a healthy account (AUTO_RESUME + rehomed) when one exists, and must fall back
to DEFER_THROTTLED only when no healthy Claude worker account is available."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_sessions  # noqa: E402


def _row(account, disp, autonomous=True, cwd=None, project="C--work-fleet",
         supervised=False, session="11111111-2222-3333-4444-555555555555"):
    """A minimal session row shaped like classify() output for decide()."""
    return {
        "account": account, "disp": disp, "autonomous": autonomous,
        "supervised": supervised, "cwd": cwd if cwd is not None else os.getcwd(),
        "project": project, "session": session, "git": "master",
        "age_min": 5.0, "last": "", "throttle_reset": None,
    }


def _avail(account, available=True, live=0, active=0):
    tag = account.replace(".claude-", "").replace(".claude", "default")
    if tag.endswith("-netra"):
        tag = tag[: -len("-netra")]
    return {"account": account, "tag": tag or "default",
            "config_dir": os.path.join(fleet_sessions.USER, account),
            "available": available, "live_sessions": live, "active_sessions": active}


class RehomeDecisionTest(unittest.TestCase):
    def test_throttled_autonomous_session_rehomes_to_healthy_account(self) -> None:
        rows = [_row(".claude-gem8-netra", "STOPPED_LIMIT")]
        throttle = {".claude-gem8-netra": {"reset": "Jun 24, 8pm"}}
        availability = [
            _avail(".claude-gem8-netra", available=False),
            _avail(".claude-jack-barker-claude-netra", available=True, live=0),
        ]
        fleet_sessions.decide(rows, throttle, availability)
        r = rows[0]
        self.assertEqual(r["action"], "AUTO_RESUME")
        self.assertTrue(r["rehomed"])
        self.assertEqual(r["resume_account"], ".claude-jack-barker-claude-netra")
        self.assertIn("jack-barker-claude", r["resume_config_dir"])
        # the operator command copies the transcript before resuming
        self.assertIn("Copy-Item", r["resume_cmd"])
        self.assertIn("CLAUDE_CONFIG_DIR", r["resume_cmd"])

    def test_no_healthy_account_falls_back_to_defer(self) -> None:
        rows = [_row(".claude-gem8-netra", "STOPPED_LIMIT")]
        throttle = {".claude-gem8-netra": {"reset": "Jun 24, 8pm"}}
        availability = [_avail(".claude-gem8-netra", available=False)]
        fleet_sessions.decide(rows, throttle, availability)
        r = rows[0]
        self.assertEqual(r["action"], "DEFER_THROTTLED")
        self.assertFalse(r["rehomed"])
        self.assertEqual(r["resume_account"], r["account"])

    def test_opencode_account_is_not_a_rehome_target(self) -> None:
        # a Claude transcript cannot resume under an opencode config dir
        rows = [_row(".claude-gem8-netra", "STOPPED_LIMIT")]
        throttle = {".claude-gem8-netra": {"reset": "Jun 24, 8pm"}}
        availability = [_avail("opencode-glm", available=True)]
        fleet_sessions.decide(rows, throttle, availability)
        self.assertEqual(rows[0]["action"], "DEFER_THROTTLED")

    def test_interactive_throttled_session_does_not_rehome(self) -> None:
        # non-autonomous sessions are never auto-resumed, re-home included
        rows = [_row(".claude-gem8-netra", "STOPPED_LIMIT", autonomous=False)]
        throttle = {".claude-gem8-netra": {"reset": "Jun 24, 8pm"}}
        availability = [_avail(".claude-jack-barker-claude-netra", available=True)]
        fleet_sessions.decide(rows, throttle, availability)
        self.assertEqual(rows[0]["action"], "DEFER_THROTTLED")
        self.assertFalse(rows[0]["rehomed"])

    def test_dead_session_on_throttled_account_rehomes(self) -> None:
        # account-wide throttle (not this row's own limit banner) still re-homes
        rows = [_row(".claude-gem8-netra", "DEAD_MIDTOOL")]
        throttle = {".claude-gem8-netra": {"reset": "Jun 24, 8pm"}}
        availability = [_avail(".claude-jack-barker-claude-netra", available=True)]
        fleet_sessions.decide(rows, throttle, availability)
        self.assertEqual(rows[0]["action"], "AUTO_RESUME")
        self.assertTrue(rows[0]["rehomed"])

    def test_healthy_account_resumes_in_place(self) -> None:
        rows = [_row(".claude-jack-barker-claude-netra", "DEAD_MIDTOOL")]
        availability = [_avail(".claude-jack-barker-claude-netra", available=True)]
        fleet_sessions.decide(rows, {}, availability)
        r = rows[0]
        self.assertEqual(r["action"], "AUTO_RESUME")
        self.assertFalse(r["rehomed"])
        self.assertEqual(r["resume_account"], r["account"])
        self.assertNotIn("Copy-Item", r["resume_cmd"] or "")

    def test_least_loaded_healthy_account_wins(self) -> None:
        rows = [_row(".claude-gem8-netra", "STOPPED_LIMIT")]
        throttle = {".claude-gem8-netra": {"reset": "Jun 24, 8pm"}}
        availability = [
            _avail(".claude-aaa-netra", available=True, live=3, active=5),
            _avail(".claude-bbb-netra", available=True, live=0, active=1),
        ]
        fleet_sessions.decide(rows, throttle, availability)
        self.assertEqual(rows[0]["resume_account"], ".claude-bbb-netra")

    def test_plan_entry_carries_rehome_fields(self) -> None:
        rows = [_row(".claude-gem8-netra", "STOPPED_LIMIT")]
        throttle = {".claude-gem8-netra": {"reset": "Jun 24, 8pm"}}
        availability = [_avail(".claude-jack-barker-claude-netra", available=True)]
        fleet_sessions.decide(rows, throttle, availability)
        entry = fleet_sessions.plan_entry(rows[0])
        for key in ("rehomed", "resume_account", "resume_config_dir",
                    "source_config_dir", "config_dir", "project", "session"):
            self.assertIn(key, entry)
        self.assertTrue(entry["rehomed"])
        self.assertNotEqual(entry["resume_config_dir"], entry["source_config_dir"])


if __name__ == "__main__":
    unittest.main()
