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
    if tag.endswith("-acct"):
        tag = tag[: -len("-acct")]
    return {"account": account, "tag": tag or "default",
            "config_dir": os.path.join(fleet_sessions.USER, account),
            "available": available, "live_sessions": live, "active_sessions": active}


class RehomeDecisionTest(unittest.TestCase):
    def test_throttled_autonomous_session_rehomes_to_healthy_account(self) -> None:
        rows = [_row(".claude-gem8-acct", "STOPPED_LIMIT")]
        throttle = {".claude-gem8-acct": {"reset": "Jun 24, 8pm"}}
        availability = [
            _avail(".claude-gem8-acct", available=False),
            _avail(".claude-jack-barker-claude-acct", available=True, live=0),
        ]
        fleet_sessions.decide(rows, throttle, availability)
        r = rows[0]
        self.assertEqual(r["action"], "AUTO_RESUME")
        self.assertTrue(r["rehomed"])
        self.assertEqual(r["resume_account"], ".claude-jack-barker-claude-acct")
        self.assertIn("jack-barker-claude", r["resume_config_dir"])
        # the operator command copies the transcript before resuming
        self.assertIn("Copy-Item", r["resume_cmd"])
        self.assertIn("CLAUDE_CONFIG_DIR", r["resume_cmd"])

    def test_no_healthy_account_falls_back_to_defer(self) -> None:
        rows = [_row(".claude-gem8-acct", "STOPPED_LIMIT")]
        throttle = {".claude-gem8-acct": {"reset": "Jun 24, 8pm"}}
        availability = [_avail(".claude-gem8-acct", available=False)]
        fleet_sessions.decide(rows, throttle, availability)
        r = rows[0]
        self.assertEqual(r["action"], "DEFER_THROTTLED")
        self.assertFalse(r["rehomed"])
        self.assertEqual(r["resume_account"], r["account"])

    def test_opencode_account_is_not_a_rehome_target(self) -> None:
        # a Claude transcript cannot resume under an opencode config dir
        rows = [_row(".claude-gem8-acct", "STOPPED_LIMIT")]
        throttle = {".claude-gem8-acct": {"reset": "Jun 24, 8pm"}}
        availability = [_avail("opencode-glm", available=True)]
        fleet_sessions.decide(rows, throttle, availability)
        self.assertEqual(rows[0]["action"], "DEFER_THROTTLED")

    def test_interactive_throttled_session_does_not_rehome(self) -> None:
        # non-autonomous sessions are never auto-resumed, re-home included
        rows = [_row(".claude-gem8-acct", "STOPPED_LIMIT", autonomous=False)]
        throttle = {".claude-gem8-acct": {"reset": "Jun 24, 8pm"}}
        availability = [_avail(".claude-jack-barker-claude-acct", available=True)]
        fleet_sessions.decide(rows, throttle, availability)
        self.assertEqual(rows[0]["action"], "DEFER_THROTTLED")
        self.assertFalse(rows[0]["rehomed"])

    def test_dead_session_on_throttled_account_rehomes(self) -> None:
        # account-wide throttle (not this row's own limit banner) still re-homes
        rows = [_row(".claude-gem8-acct", "DEAD_MIDTOOL")]
        throttle = {".claude-gem8-acct": {"reset": "Jun 24, 8pm"}}
        availability = [_avail(".claude-jack-barker-claude-acct", available=True)]
        fleet_sessions.decide(rows, throttle, availability)
        self.assertEqual(rows[0]["action"], "AUTO_RESUME")
        self.assertTrue(rows[0]["rehomed"])

    def test_healthy_account_resumes_in_place(self) -> None:
        rows = [_row(".claude-jack-barker-claude-acct", "DEAD_MIDTOOL")]
        availability = [_avail(".claude-jack-barker-claude-acct", available=True)]
        fleet_sessions.decide(rows, {}, availability)
        r = rows[0]
        self.assertEqual(r["action"], "AUTO_RESUME")
        self.assertFalse(r["rehomed"])
        self.assertEqual(r["resume_account"], r["account"])
        self.assertNotIn("Copy-Item", r["resume_cmd"] or "")

    def test_least_loaded_healthy_account_wins(self) -> None:
        rows = [_row(".claude-gem8-acct", "STOPPED_LIMIT")]
        throttle = {".claude-gem8-acct": {"reset": "Jun 24, 8pm"}}
        availability = [
            _avail(".claude-aaa-acct", available=True, live=3, active=5),
            _avail(".claude-bbb-acct", available=True, live=0, active=1),
        ]
        fleet_sessions.decide(rows, throttle, availability)
        self.assertEqual(rows[0]["resume_account"], ".claude-bbb-acct")

    def test_plan_entry_carries_rehome_fields(self) -> None:
        rows = [_row(".claude-gem8-acct", "STOPPED_LIMIT")]
        throttle = {".claude-gem8-acct": {"reset": "Jun 24, 8pm"}}
        availability = [_avail(".claude-jack-barker-claude-acct", available=True)]
        fleet_sessions.decide(rows, throttle, availability)
        entry = fleet_sessions.plan_entry(rows[0])
        for key in ("rehomed", "resume_account", "resume_config_dir",
                    "source_config_dir", "config_dir", "project", "session"):
            self.assertIn(key, entry)
        self.assertTrue(entry["rehomed"])
        self.assertNotEqual(entry["resume_config_dir"], entry["source_config_dir"])


class RehomeSpreadTest(unittest.TestCase):
    """A burst of throttled sessions must SPREAD across healthy accounts rather than
    all stampede onto the one momentary least-loaded target (the 32->1 concentration
    that wedged every resume onto smith-netra and made it limit-wall)."""

    def _sids(self, n):
        return [f"{i:08d}-2222-3333-4444-555555555555" for i in range(n)]

    def test_burst_spreads_across_healthy_accounts(self) -> None:
        sids = self._sids(4)
        rows = [_row(".claude-owner-acct", "STOPPED_LIMIT", session=s) for s in sids]
        throttle = {".claude-owner-acct": {"reset": "Jun 24, 8pm"}}
        availability = [
            _avail(".claude-aaa-acct", available=True, live=0),
            _avail(".claude-bbb-acct", available=True, live=0),
        ]
        fleet_sessions.decide(rows, throttle, availability)
        targets = [r["resume_account"] for r in rows]
        # 4 sessions, 2 empty healthy accounts -> 2 each, never 4 onto one.
        self.assertEqual(targets.count(".claude-aaa-acct"), 2)
        self.assertEqual(targets.count(".claude-bbb-acct"), 2)

    def test_burst_respects_per_account_cap(self) -> None:
        # With one healthy account and REHOME_CAP=4, the 5th session past the cap
        # must DEFER rather than pile onto an account that will itself limit-wall.
        sids = self._sids(5)
        rows = [_row(".claude-owner-acct", "STOPPED_LIMIT", session=s) for s in sids]
        throttle = {".claude-owner-acct": {"reset": "Jun 24, 8pm"}}
        availability = [_avail(".claude-solo-acct", available=True, live=0)]
        fleet_sessions.decide(rows, throttle, availability)
        actions = [r["action"] for r in rows]
        self.assertEqual(actions.count("AUTO_RESUME"), fleet_sessions.REHOME_CAP)
        self.assertEqual(actions.count("DEFER_THROTTLED"), 5 - fleet_sessions.REHOME_CAP)

    def test_single_session_still_picks_least_loaded(self) -> None:
        # the load-aware change must not regress the single-session least-loaded pick
        rows = [_row(".claude-owner-acct", "STOPPED_LIMIT")]
        throttle = {".claude-owner-acct": {"reset": "Jun 24, 8pm"}}
        availability = [
            _avail(".claude-busy-acct", available=True, live=3),
            _avail(".claude-idle-acct", available=True, live=0),
        ]
        fleet_sessions.decide(rows, throttle, availability)
        self.assertEqual(rows[0]["resume_account"], ".claude-idle-acct")

    def test_proven_healthy_account_ranks_above_unproven(self) -> None:
        # equal load: an account with a fresh positive verdict (probe) beats one whose
        # `available` is merely the absence-of-evidence default ("none").
        proven = _avail(".claude-proven-acct", available=True, live=0)
        proven["verdict_source"] = "probe"
        unproven = _avail(".claude-unproven-acct", available=True, live=0)
        unproven["verdict_source"] = "none"
        # list the unproven FIRST to prove ranking, not list order, decides.
        cands = fleet_sessions._rehome_targets([unproven, proven], ".claude-owner-acct")
        self.assertEqual(cands[0]["account"], ".claude-proven-acct")

    def test_passive_verdict_ranks_above_carried(self) -> None:
        # 'passive' (a real session row inside the window proves the account alive) is
        # genuine positive evidence and must rank above a stale 'carried' verdict --
        # even when the carried account's tag sorts first. (The earlier draft whitelisted
        # a phantom 'registry' and dropped 'passive', so passive tied with carried.)
        carried = _avail(".claude-aaa-acct", available=True, live=0)   # tag sorts first
        carried["verdict_source"] = "carried"
        passive = _avail(".claude-zzz-acct", available=True, live=0)   # tag sorts last
        passive["verdict_source"] = "passive"
        cands = fleet_sessions._rehome_targets([carried, passive], ".claude-owner-acct")
        self.assertEqual(cands[0]["account"], ".claude-zzz-acct",
                         "a proven-alive 'passive' account must beat a stale 'carried' one")


if __name__ == "__main__":
    unittest.main()
