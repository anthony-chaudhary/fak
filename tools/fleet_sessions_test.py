#!/usr/bin/env python3
"""Hermetic tests for the cross-account re-home decision in fleet_sessions.

These cover the exact gap that left throttled sessions "pinned" to a rate-limited
account: a resumable autonomous session whose owner is throttled must be re-homed
onto a healthy account (AUTO_RESUME + rehomed) when one exists, and must fall back
to DEFER_THROTTLED only when no healthy Claude worker account is available."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_sessions  # noqa: E402


def _row(account, disp, autonomous=True, cwd=None, project="C--work-fleet",
         supervised=False, session="11111111-2222-3333-4444-555555555555",
         task_sig="", records=1, seen_utc="", last_ts=""):
    """A minimal session row shaped like classify() output for decide().

    task_sig/records/seen_utc default to the no-dedup case (empty sig) so the
    existing re-home tests are untouched; dedup tests set them explicitly.
    last_ts (the terminal turn's OWN timestamp) defaults empty for the same
    reason -- it is the #3459 newest-copy key, set only by the copy-gate tests."""
    return {
        "account": account, "disp": disp, "autonomous": autonomous,
        "supervised": supervised, "cwd": cwd if cwd is not None else os.getcwd(),
        "project": project, "session": session, "git": "master",
        "age_min": 5.0, "last": "", "throttle_reset": None,
        "task_sig": task_sig, "records": records, "seen_utc": seen_utc,
        "last_ts": last_ts,
    }


def _avail(account, available=True, live=0, active=0, verdict_source="passive"):
    """An availability row shaped like account_availability() output.

    verdict_source defaults to 'passive' (a real session row inside the window proves
    the account alive) -- the production-faithful default, since account_availability
    always stamps a verdict and an account that reads `available` does so on positive
    evidence (a probe OK or a live/done row). Tests exercising the #619 launch-boundary
    rule pass verdict_source='carried' (a stale verdict with no fresh evidence)."""
    tag = account.replace(".claude-", "").replace(".claude", "default")
    if tag.endswith("-acct"):
        tag = tag[: -len("-acct")]
    return {"account": account, "tag": tag or "default",
            "config_dir": os.path.join(fleet_sessions.USER, account),
            "available": available, "live_sessions": live, "active_sessions": active,
            "verdict_source": verdict_source}


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

    def test_merge_known_throttle_prunes_archived_accounts(self) -> None:
        prev = {
            "throttle": {
                ".claude-live": {"reset": "tomorrow"},
                ".claude-gem7-netra": {"reset": "tomorrow"},
            }
        }
        with mock.patch.object(fleet_sessions.fleet_accounts, "load_registry",
                               return_value=prev), \
             mock.patch.object(fleet_sessions, "_account_still_worker",
                               side_effect=lambda acct: acct == ".claude-live"):
            merged = fleet_sessions.merge_known_throttle({}, [])

        self.assertIn(".claude-live", merged)
        self.assertNotIn(".claude-gem7-netra", merged)

    def test_merge_known_auth_prunes_archived_accounts(self) -> None:
        prev = {
            "generated_utc": "2026-07-05T16:00:00+00:00",
            "auth": {
                ".claude-live": {"block_reason": "auth/login required"},
                ".claude-gem7-netra": {"block_reason": "auth/login required"},
            },
        }
        with mock.patch.object(fleet_sessions.fleet_accounts, "load_registry",
                               return_value=prev), \
             mock.patch.object(fleet_sessions, "_account_still_worker",
                               side_effect=lambda acct: acct == ".claude-live"):
            merged = fleet_sessions.merge_known_auth([])

        self.assertIn(".claude-live", merged)
        self.assertNotIn(".claude-gem7-netra", merged)

    def test_interactive_throttled_session_rehomes(self) -> None:
        # #1353: a rate-limit (STOPPED_LIMIT) is server-interrupted, not abandoned, so an
        # INTERACTIVE (autonomous=False) session re-homes onto a healthy account too.
        rows = [_row(".claude-gem8-acct", "STOPPED_LIMIT", autonomous=False)]
        throttle = {".claude-gem8-acct": {"reset": "Jun 24, 8pm"}}
        availability = [
            _avail(".claude-gem8-acct", available=False),
            _avail(".claude-jack-barker-claude-acct", available=True),
        ]
        fleet_sessions.decide(rows, throttle, availability)
        self.assertEqual(rows[0]["action"], "AUTO_RESUME")
        self.assertTrue(rows[0]["rehomed"])
        self.assertEqual(rows[0]["resume_account"], ".claude-jack-barker-claude-acct")

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

    def test_stale_limit_banner_on_healthy_owner_resumes_in_place(self) -> None:
        # #621: a STOPPED_LIMIT disp carried in a re-homed transcript whose CURRENT
        # owner is NOT throttled and reads available must resume IN PLACE -- not re-home
        # off the healthy owner (the bug that stranded 5/15 in the 2026-06-24 incident).
        rows = [_row(".claude-jack-barker-claude-acct", "STOPPED_LIMIT")]
        availability = [
            _avail(".claude-jack-barker-claude-acct", available=True),
            _avail(".claude-other-acct", available=True),
        ]
        fleet_sessions.decide(rows, {}, availability)  # owner NOT in the throttle map
        r = rows[0]
        self.assertEqual(r["action"], "AUTO_RESUME")
        self.assertFalse(r["rehomed"])
        self.assertEqual(r["resume_account"], r["account"])
        self.assertNotIn("Copy-Item", r["resume_cmd"] or "")

    def test_stale_limit_banner_unavailable_owner_still_rehomes(self) -> None:
        # The stale-banner guard only fires for a CURRENTLY-available owner. A
        # STOPPED_LIMIT owner that is not in the throttle map but reads unavailable
        # in the snapshot is not a cleared limit -- re-home onto a healthy account.
        rows = [_row(".claude-owner-acct", "STOPPED_LIMIT")]
        availability = [
            _avail(".claude-owner-acct", available=False),
            _avail(".claude-healthy-acct", available=True),
        ]
        fleet_sessions.decide(rows, {}, availability)
        r = rows[0]
        self.assertEqual(r["action"], "AUTO_RESUME")
        self.assertTrue(r["rehomed"])
        self.assertEqual(r["resume_account"], ".claude-healthy-acct")

    def test_genuinely_throttled_owner_rehomes_even_if_snapshot_shows_available(self) -> None:
        # The throttle map stays authoritative: an owner IN the throttle map re-homes
        # even when a stale snapshot still lists it available, so the guard cannot be
        # tricked into pinning a session onto a genuinely rate-limited account.
        rows = [_row(".claude-gem8-acct", "STOPPED_LIMIT")]
        throttle = {".claude-gem8-acct": {"reset": "Jun 24, 8pm"}}
        availability = [
            _avail(".claude-gem8-acct", available=True),
            _avail(".claude-jack-barker-claude-acct", available=True),
        ]
        fleet_sessions.decide(rows, throttle, availability)
        r = rows[0]
        self.assertEqual(r["action"], "AUTO_RESUME")
        self.assertTrue(r["rehomed"])
        self.assertEqual(r["resume_account"], ".claude-jack-barker-claude-acct")

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
        # _rehome_targets is a pure RANKER: an account with a fresh positive verdict
        # (probe) sorts ahead of one whose `available` is merely the absence-of-evidence
        # default ("none"). (The hard exclusion of carried/none is the launch-boundary
        # rule tested in LaunchBoundaryAdmissionTest via decide()/_admissible_targets.)
        proven = _avail(".claude-proven-acct", available=True, live=0, verdict_source="probe")
        unproven = _avail(".claude-unproven-acct", available=True, live=0, verdict_source="none")
        # list the unproven FIRST to prove ranking, not list order, decides.
        cands = fleet_sessions._rehome_targets([unproven, proven], ".claude-owner-acct")
        self.assertEqual(cands[0]["account"], ".claude-proven-acct")

    def test_passive_verdict_ranks_above_carried(self) -> None:
        # 'passive' (a real session row inside the window proves the account alive) is
        # genuine positive evidence and must rank above a stale 'carried' verdict --
        # even when the carried account's tag sorts first.
        carried = _avail(".claude-aaa-acct", available=True, live=0, verdict_source="carried")
        passive = _avail(".claude-zzz-acct", available=True, live=0, verdict_source="passive")
        cands = fleet_sessions._rehome_targets([carried, passive], ".claude-owner-acct")
        self.assertEqual(cands[0]["account"], ".claude-zzz-acct",
                         "a proven-alive 'passive' account must beat a stale 'carried' one")


class LaunchBoundaryAdmissionTest(unittest.TestCase):
    """#619: ONE authoritative, freshness-stamped verdict gates every launch. Load is
    never admitted onto a CARRIED / absence-of-evidence verdict -- a carried 'available'
    that flip-flops with whether the pass probed cannot route a workload. The decision
    is identical across repeated passes over the SAME evidence (the day24 incident:
    available@22:17, throttled@22:19, available@22:20 -- the carried verdict latched
    routing non-deterministically)."""

    def _carried_only(self):
        """A re-home decision whose ONLY candidate is a carried-verdict 'available'
        account, owner genuinely throttled. Returns the decided row."""
        rows = [_row(".claude-owner-acct", "STOPPED_LIMIT")]
        throttle = {".claude-owner-acct": {"reset": "Jun 24, 8pm"}}
        carried = _avail(".claude-carried-acct", available=True, live=0,
                         verdict_source="carried")
        fleet_sessions.decide(rows, throttle, [carried])
        return rows[0]

    def test_carried_only_verdict_refuses_load(self) -> None:
        # carried 'available' is NOT positive evidence -> not a re-home target -> DEFER.
        r = self._carried_only()
        self.assertEqual(r["action"], "DEFER_THROTTLED")
        self.assertFalse(r["rehomed"])

    def test_carried_only_decision_is_deterministic(self) -> None:
        # the acceptance: identical evidence -> identical decision on every pass.
        a = self._carried_only()
        b = self._carried_only()
        self.assertEqual((a["action"], a["resume_account"]),
                         (b["action"], b["resume_account"]))
        self.assertEqual(a["action"], "DEFER_THROTTLED")

    def test_fresh_probe_admits_load(self) -> None:
        # the same shape but with a fresh PROBE verdict -> positive evidence -> admitted.
        rows = [_row(".claude-owner-acct", "STOPPED_LIMIT")]
        throttle = {".claude-owner-acct": {"reset": "Jun 24, 8pm"}}
        probed = _avail(".claude-probed-acct", available=True, live=0,
                        verdict_source="probe")
        fleet_sessions.decide(rows, throttle, [probed])
        r = rows[0]
        self.assertEqual(r["action"], "AUTO_RESUME")
        self.assertTrue(r["rehomed"])
        self.assertEqual(r["resume_account"], ".claude-probed-acct")

    def test_carried_owner_does_not_resume_in_place(self) -> None:
        # The in-place resume IS a launch: a STOPPED_LIMIT session whose owner is NOT in
        # the throttle map but carries only a stale 'carried' verdict must not resume in
        # place on that unproven owner. With no proven target it DEFERs (re-probe first).
        rows = [_row(".claude-owner-acct", "STOPPED_LIMIT")]
        carried = _avail(".claude-owner-acct", available=True, verdict_source="carried")
        fleet_sessions.decide(rows, {}, [carried])   # owner NOT in the throttle map
        r = rows[0]
        self.assertEqual(r["action"], "DEFER_THROTTLED")
        self.assertFalse(r["rehomed"])

    def test_carried_owner_rehomes_onto_proven_target(self) -> None:
        # carried owner + a fresh-probed alternative: don't resume in place on the
        # unproven owner -- re-home onto the proven-healthy account instead.
        rows = [_row(".claude-owner-acct", "STOPPED_LIMIT")]
        carried_owner = _avail(".claude-owner-acct", available=True, verdict_source="carried")
        probed = _avail(".claude-probed-acct", available=True, live=0, verdict_source="probe")
        fleet_sessions.decide(rows, {}, [carried_owner, probed])
        r = rows[0]
        self.assertEqual(r["action"], "AUTO_RESUME")
        self.assertTrue(r["rehomed"])
        self.assertEqual(r["resume_account"], ".claude-probed-acct")


class OrgDisabledRehomeTest(unittest.TestCase):
    """An org/subscription-disabled account (auth_block_kind == 'access') can't be
    fixed by /login on the owner -- but the transcript re-homes onto a healthy,
    non-org-disabled account WITH usage, exactly like the rate-limit path."""

    def test_org_disabled_session_rehomes_to_healthy_account(self) -> None:
        rows = [_row(".claude-orgdead-acct", "INFRA_ORG_DISABLED")]
        availability = [
            _avail(".claude-orgdead-acct", available=False),
            _avail(".claude-good-acct", available=True, live=0),
        ]
        fleet_sessions.decide(rows, {}, availability)
        r = rows[0]
        self.assertEqual(r["action"], "AUTO_RESUME")
        self.assertTrue(r["rehomed"])
        self.assertEqual(r["resume_account"], ".claude-good-acct")
        # re-homed org-disabled session gets a transcript-copy resume command
        self.assertIn("Copy-Item", r["resume_cmd"])

    def test_org_disabled_no_healthy_account_defers_no_usage(self) -> None:
        # no usable seat -> DEFER_NO_USAGE, NOT BLOCKED_AUTH (re-login won't help).
        rows = [_row(".claude-orgdead-acct", "INFRA_ORG_DISABLED")]
        availability = [_avail(".claude-orgdead-acct", available=False)]
        fleet_sessions.decide(rows, {}, availability)
        r = rows[0]
        self.assertEqual(r["action"], "DEFER_NO_USAGE")
        self.assertFalse(r["rehomed"])

    def test_org_disabled_target_pool_excludes_org_disabled_accounts(self) -> None:
        # the only "available" account is itself blocked -> no target -> DEFER_NO_USAGE
        rows = [_row(".claude-orgdead-acct", "INFRA_ORG_DISABLED")]
        availability = [_avail(".claude-also-dead-acct", available=False)]
        fleet_sessions.decide(rows, {}, availability)
        self.assertEqual(rows[0]["action"], "DEFER_NO_USAGE")

    def test_plain_auth_still_blocks(self) -> None:
        # token-expiry / 401 auth keeps INFRA_AUTH -> BLOCKED_AUTH (genuinely needs /login)
        rows = [_row(".claude-auth-acct", "INFRA_AUTH")]
        availability = [_avail(".claude-good-acct", available=True)]
        fleet_sessions.decide(rows, {}, availability)
        self.assertEqual(rows[0]["action"], "BLOCKED_AUTH")
        self.assertFalse(rows[0]["rehomed"])

    def test_interactive_org_disabled_rehomes(self) -> None:
        # #1353: an org-disabled wall is server-side, not a human stop -> an INTERACTIVE
        # session re-homes onto a healthy seat too, exactly like the autonomous path.
        rows = [_row(".claude-orgdead-acct", "INFRA_ORG_DISABLED", autonomous=False)]
        availability = [
            _avail(".claude-orgdead-acct", available=False),
            _avail(".claude-good-acct", available=True),
        ]
        fleet_sessions.decide(rows, {}, availability)
        self.assertEqual(rows[0]["action"], "AUTO_RESUME")
        self.assertTrue(rows[0]["rehomed"])
        self.assertEqual(rows[0]["resume_account"], ".claude-good-acct")


class InfraStrandRegardlessOfAutonomyTest(unittest.TestCase):
    """#1353: a session the SERVER interrupted (transient 529 / API error / rate-limit /
    org-disabled) is auto-resumable regardless of autonomy -- an interactive chat walled
    mid-conversation was interrupted, not abandoned. Intentional human stops (USER_CLOSED)
    and clean finishes (DONE) stay excluded; an agent crash (DEAD) keeps the autonomy gate."""

    def test_interactive_apierr_resumes_in_place(self) -> None:
        # the exact symptom: init=user, disp=STOPPED_APIERR on a healthy owner -> AUTO_RESUME
        # (resume in place, not surfaced to a human and not re-homed).
        rows = [_row(".claude-good-acct", "STOPPED_APIERR", autonomous=False)]
        availability = [_avail(".claude-good-acct", available=True)]
        fleet_sessions.decide(rows, {}, availability)
        r = rows[0]
        self.assertEqual(r["action"], "AUTO_RESUME")
        self.assertFalse(r["rehomed"])
        self.assertEqual(r["resume_account"], r["account"])

    def test_autonomous_apierr_still_resumes(self) -> None:
        # the agent path (the row that DID enter the plan before) is unchanged.
        rows = [_row(".claude-good-acct", "STOPPED_APIERR", autonomous=True)]
        fleet_sessions.decide(rows, {}, [_avail(".claude-good-acct", available=True)])
        self.assertEqual(rows[0]["action"], "AUTO_RESUME")

    def test_interactive_user_closed_still_skips(self) -> None:
        # Esc/Ctrl-C/`/quit` is an intentional human stop -> never resume.
        rows = [_row(".claude-good-acct", "USER_CLOSED", autonomous=False)]
        fleet_sessions.decide(rows, {}, [_avail(".claude-good-acct", available=True)])
        self.assertEqual(rows[0]["action"], "SKIP_USER_CLOSED")

    def test_interactive_done_still_skips(self) -> None:
        # a clean finish is not an interruption -> never resume.
        rows = [_row(".claude-good-acct", "DONE", autonomous=False)]
        fleet_sessions.decide(rows, {}, [_avail(".claude-good-acct", available=True)])
        self.assertEqual(rows[0]["action"], "SKIP_DONE")

    def test_interactive_dead_keeps_autonomy_gate(self) -> None:
        # an AGENT crash (DEAD) the human walked away from is NOT auto-relaunched; it is
        # surfaced for a human. Only the infra dispositions drop the autonomy gate.
        rows = [_row(".claude-good-acct", "DEAD_MIDTOOL", autonomous=False)]
        fleet_sessions.decide(rows, {}, [_avail(".claude-good-acct", available=True)])
        self.assertEqual(rows[0]["action"], "SURFACE")

    def test_resumable_disp_predicate(self) -> None:
        # the dedup helper agrees: infra dispositions resumable regardless of autonomy;
        # DEAD only when autonomous; USER_CLOSED/DONE never.
        for disp in fleet_sessions.INFRA_RESUMABLE:
            self.assertTrue(fleet_sessions._resumable_disp(_row("a", disp, autonomous=False)))
        self.assertTrue(fleet_sessions._resumable_disp(_row("a", "DEAD_MIDTOOL", autonomous=True)))
        self.assertFalse(fleet_sessions._resumable_disp(_row("a", "DEAD_MIDTOOL", autonomous=False)))
        self.assertFalse(fleet_sessions._resumable_disp(_row("a", "USER_CLOSED", autonomous=False)))


class OperatorStopContextExhaustedTest(unittest.TestCase):
    """#3458: an operator-stop / BUDGET_CONTEXT_EXHAUSTED tail is a TERMINAL wall for
    the TRANSCRIPT -- a raw `claude --resume` reloads the exhausted context and is
    refused with the same 409 forever (the amnesia loop). classify() must give it a
    distinct disposition (never the transient STOPPED_APIERR) and decide() must route
    it away from AUTO_RESUME-in-place with no resume_cmd."""

    OPERATOR_STOP_TAIL = (
        "API Error: 409 session f8d84269 is stopped (operator control); "
        "request refused: BUDGET_CONTEXT_EXHAUSTED")

    def setUp(self):
        # isolate the ledger reads (_ledger_inplace_attempts / _ledger_blocked_sids)
        self._tmp = __import__("tempfile").mkdtemp()
        self._orig_reg = fleet_sessions.REG_DIR
        fleet_sessions.REG_DIR = self._tmp

    def tearDown(self):
        fleet_sessions.REG_DIR = self._orig_reg
        __import__("shutil").rmtree(self._tmp, ignore_errors=True)

    def _transcript(self, tail_text, sid="ab345800-2222-3333-4444-555555555555"):
        """A minimal transcript whose final record is a synthetic assistant banner."""
        proj = os.path.join(self._tmp, "projects", "C--work-fleet")
        os.makedirs(proj, exist_ok=True)
        path = os.path.join(proj, sid + ".jsonl")
        recs = [
            {"type": "user", "sessionId": sid, "cwd": os.getcwd(),
             "message": {"role": "user", "content": "continue the migration"}},
            {"type": "assistant", "sessionId": sid, "cwd": os.getcwd(),
             "message": {"role": "assistant", "model": "<synthetic>",
                         "content": [{"type": "text", "text": tail_text}],
                         "stop_reason": "stop_sequence"}},
        ]
        with open(path, "w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
        return path

    def test_classify_operator_stop_tail_gets_distinct_disposition(self) -> None:
        # the exact loop signature: 409 + operator control + BUDGET_CONTEXT_EXHAUSTED
        r = fleet_sessions.classify(self._transcript(self.OPERATOR_STOP_TAIL))
        self.assertEqual(r["disp"], "STOPPED_CONTEXT_EXHAUSTED")
        self.assertEqual(r["category"], "INFRA")
        self.assertEqual(r["cause"], "context_exhausted")

    def test_classify_plain_500_still_apierr(self) -> None:
        # regression: a genuinely transient transport error keeps the retry path.
        r = fleet_sessions.classify(self._transcript(
            "API Error: 500 Internal Server Error",
            sid="cd345801-2222-3333-4444-555555555555"))
        self.assertEqual(r["disp"], "STOPPED_APIERR")

    def test_decide_never_resumes_in_place(self) -> None:
        # healthy owner (the case _resume_inplace_or_escalate would have grabbed):
        # no AUTO_RESUME, no resume_cmd, no re-home -- escalate to a fresh continuation.
        rows = [_row(".claude-good-acct", "STOPPED_CONTEXT_EXHAUSTED")]
        fleet_sessions.decide(rows, {}, [_avail(".claude-good-acct", available=True)])
        r = rows[0]
        self.assertEqual(r["action"], "ESCALATE_FRESH_CONTINUATION")
        self.assertIsNone(r["resume_cmd"])
        self.assertFalse(r["rehomed"])

    def test_decide_throttled_owner_still_escalates_fresh(self) -> None:
        # the wall is bound to the transcript, not the seat: a throttled owner must not
        # divert it into the DEFER_THROTTLED/re-home ladder (a re-home re-hits the 409).
        rows = [_row(".claude-gem8-acct", "STOPPED_CONTEXT_EXHAUSTED")]
        throttle = {".claude-gem8-acct": {"reset": "Jun 24, 8pm"}}
        availability = [_avail(".claude-gem8-acct", available=False),
                        _avail(".claude-good-acct", available=True)]
        fleet_sessions.decide(rows, throttle, availability)
        self.assertEqual(rows[0]["action"], "ESCALATE_FRESH_CONTINUATION")
        self.assertIsNone(rows[0]["resume_cmd"])
        self.assertFalse(rows[0]["rehomed"])

    def test_decide_apierr_regression_keeps_resume(self) -> None:
        # the transient path is untouched: STOPPED_APIERR still AUTO_RESUMEs with a cmd.
        rows = [_row(".claude-good-acct", "STOPPED_APIERR")]
        fleet_sessions.decide(rows, {}, [_avail(".claude-good-acct", available=True)])
        self.assertEqual(rows[0]["action"], "AUTO_RESUME")
        self.assertIn("claude --resume", rows[0]["resume_cmd"])


class NeverStartedTest(unittest.TestCase):
    """A session that was launched (a goal/prompt was written) but whose model never
    emitted a single real assistant turn is a launch NON-START, not a mid-work hang.
    classify() must split it out of the HANGING bucket (98/114 rows on 2026-07-09), and
    decide() must re-launch an autonomous one (no partial work to lose) rather than
    dumping it on a human as an ambiguous quiet stop."""

    def setUp(self):
        self._tmp = __import__("tempfile").mkdtemp()
        self._orig_reg = fleet_sessions.REG_DIR
        fleet_sessions.REG_DIR = self._tmp

    def tearDown(self):
        fleet_sessions.REG_DIR = self._orig_reg
        __import__("shutil").rmtree(self._tmp, ignore_errors=True)

    def _aged(self, recs, sid="ab345800-2222-3333-4444-999999999999"):
        """Write recs to a transcript and age it past LIVE_MIN (else it reads LIVE)."""
        proj = os.path.join(self._tmp, "projects", "C--work-fleet")
        os.makedirs(proj, exist_ok=True)
        path = os.path.join(proj, sid + ".jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
        old = __import__("time").time() - 3600     # 60 min old -> well past LIVE_MIN
        os.utime(path, (old, old))
        return path

    def _user(self, text, sid):
        return {"type": "user", "sessionId": sid, "cwd": os.getcwd(),
                "message": {"role": "user", "content": text}}

    def test_classify_goal_only_is_never_started(self) -> None:
        sid = "ab345800-2222-3333-4444-000000000001"
        r = fleet_sessions.classify(self._aged([self._user("work on issue #123", sid)], sid))
        self.assertEqual(r["disp"], "NEVER_STARTED")
        self.assertEqual(r["category"], "AGENT")
        self.assertEqual(r["cause"], "never_started")

    def test_classify_all_user_records_never_started(self) -> None:
        # the observed cluster: three user records (goal + wrappers + stop-hook notice),
        # zero assistant turns -- one dispatch wave that wedged at launch.
        sid = "ab345800-2222-3333-4444-000000000002"
        recs = [self._user("goal directive", sid), self._user("<system-reminder>", sid),
                self._user("A session-scoped Stop hook is now active", sid)]
        r = fleet_sessions.classify(self._aged(recs, sid))
        self.assertEqual(r["disp"], "NEVER_STARTED")

    def test_classify_synthetic_assistant_only_still_never_started(self) -> None:
        # a harness banner (model=<synthetic>) is NOT the model running -> still never-started.
        sid = "ab345800-2222-3333-4444-000000000003"
        recs = [self._user("goal", sid),
                {"type": "assistant", "sessionId": sid, "cwd": os.getcwd(),
                 "message": {"role": "assistant", "model": "<synthetic>",
                             "content": [{"type": "text", "text": "No response requested."}],
                             "stop_reason": "stop_sequence"}}]
        r = fleet_sessions.classify(self._aged(recs, sid))
        self.assertEqual(r["disp"], "NEVER_STARTED")

    def test_classify_real_assistant_then_quiet_is_not_never_started(self) -> None:
        # produced a genuine assistant turn, then went quiet: that IS an ambiguous quiet
        # stop (STOPPED_QUIET), distinct from a never-start. Guards the saw_assistant gate.
        sid = "ab345800-2222-3333-4444-000000000004"
        recs = [self._user("goal", sid),
                {"type": "assistant", "sessionId": sid, "cwd": os.getcwd(),
                 "message": {"role": "assistant", "model": "claude-opus-4-8",
                             "content": [{"type": "text", "text": "starting the work"}],
                             "stop_reason": "tool_use"}},
                self._user("some follow-up context", sid)]
        r = fleet_sessions.classify(self._aged(recs, sid))
        self.assertNotEqual(r["disp"], "NEVER_STARTED")
        self.assertEqual(r["disp"], "STOPPED_QUIET")

    def test_decide_autonomous_never_started_relaunches(self) -> None:
        rows = [_row(".claude-good-acct", "NEVER_STARTED", autonomous=True)]
        fleet_sessions.decide(rows, {}, [_avail(".claude-good-acct", available=True)])
        self.assertEqual(rows[0]["action"], "AUTO_RESUME")
        self.assertIn("claude --resume", rows[0]["resume_cmd"])

    def test_decide_nonautonomous_never_started_surfaces(self) -> None:
        rows = [_row(".claude-good-acct", "NEVER_STARTED", autonomous=False)]
        fleet_sessions.decide(rows, {}, [_avail(".claude-good-acct", available=True)])
        self.assertEqual(rows[0]["action"], "SURFACE")

    def test_resumable_disp_gates_on_autonomy(self) -> None:
        self.assertTrue(fleet_sessions._resumable_disp(
            _row("a", "NEVER_STARTED", autonomous=True)))
        self.assertFalse(fleet_sessions._resumable_disp(
            _row("a", "NEVER_STARTED", autonomous=False)))


class NeverStartBurstReportTest(unittest.TestCase):
    """#3553: the operator summary must make a LAUNCH-STORM (many seats launched, none
    produced an assistant turn) legible on its own recency-bucketed line, distinct from the
    apierr `storm:` line; and the per-session table must badge a repeat-crasher with its
    prior in-place relaunch count. Both surfaces are witnessed here through the pure seams
    the summary block renders from (session_storm_summary / _ledger_inplace_attempts +
    _relaunch_badge), so no live scan is needed."""

    def setUp(self):
        # isolate the ledger read so _ledger_inplace_attempts sees ONLY what we write
        self._tmp = __import__("tempfile").mkdtemp()
        self._orig_reg = fleet_sessions.REG_DIR
        fleet_sessions.REG_DIR = self._tmp

    def tearDown(self):
        fleet_sessions.REG_DIR = self._orig_reg
        __import__("shutil").rmtree(self._tmp, ignore_errors=True)

    def test_burst_counts_never_started_by_recency_window(self) -> None:
        # three NEVER_STARTED seats all fresh (age 5m -> inside every window) plus one
        # apierr and one live: the burst count is 3 in each window, the apierr line is 1.
        rows = ([_row(".claude-a", "NEVER_STARTED", session=f"{i:08d}-x") for i in range(3)]
                + [_row(".claude-a", "STOPPED_APIERR", session="apierr-1"),
                   _row(".claude-a", "LIVE", session="live-1")])
        st = fleet_sessions.session_storm_summary(rows, {})["storm"]
        self.assertEqual((st["neverstart_15m"], st["neverstart_30m"], st["neverstart_60m"]),
                         (3, 3, 3))
        # the never-start count does NOT leak into the apierr storm count and vice-versa
        self.assertEqual(st["apierr_15m"], 1)
        self.assertGreater(st["neverstart_per_min_30m"], 0.0)

    def test_burst_respects_the_recency_window(self) -> None:
        # a NEVER_STARTED seat older than 60m falls out of every bucket
        rows = [dict(_row(".claude-a", "NEVER_STARTED", session="old-1"), age_min=90.0)]
        st = fleet_sessions.session_storm_summary(rows, {})["storm"]
        self.assertEqual((st["neverstart_15m"], st["neverstart_30m"], st["neverstart_60m"]),
                         (0, 0, 0))

    def test_relaunch_badge_renders_only_when_attempts_positive(self) -> None:
        self.assertEqual(fleet_sessions._relaunch_badge(0), "")
        self.assertEqual(fleet_sessions._relaunch_badge(3), "  relaunch×3")

    def test_per_session_relaunch_count_comes_from_the_ledger(self) -> None:
        # the exact path the per-session render uses: ledger in-place count -> badge string
        sid = "abcd1111-2222-3333-4444-555555555555"
        with open(os.path.join(self._tmp, "resume_ledger.jsonl"), "w", encoding="utf-8") as fh:
            for _ in range(2):
                fh.write(json.dumps({"session": sid, "phase": "launched", "rehomed": False}) + "\n")
        counts = fleet_sessions._ledger_inplace_attempts()
        self.assertEqual(counts.get(sid), 2)
        self.assertEqual(fleet_sessions._relaunch_badge(counts.get(sid, 0)), "  relaunch×2")


class DedupTaskTest(unittest.TestCase):
    """Identical repeating autonomous tasks (same task_sig across sids) resume ONE
    primary; the rest defer so they never stampede a healthy seat."""

    def setUp(self):
        # isolate the ledger read so _ledger_blocked_sids finds an EMPTY ledger
        self._tmp = __import__("tempfile").mkdtemp()
        self._orig_reg = fleet_sessions.REG_DIR
        fleet_sessions.REG_DIR = self._tmp

    def tearDown(self):
        fleet_sessions.REG_DIR = self._orig_reg
        __import__("shutil").rmtree(self._tmp, ignore_errors=True)

    def _sids(self, n):
        return [f"{i:08d}-2222-3333-4444-555555555555" for i in range(n)]

    def test_identical_autonomous_tasks_dedup_to_one_primary(self) -> None:
        sids = self._sids(6)
        rows = [_row(".claude-good-acct", "DEAD_MIDTOOL", session=s,
                     task_sig="SAMESIG", records=100 + i) for i, s in enumerate(sids)]
        availability = [_avail(".claude-good-acct", available=True, live=0)]
        fleet_sessions.decide(rows, {}, availability)
        actions = [r["action"] for r in rows]
        self.assertEqual(actions.count("AUTO_RESUME"), 1)
        self.assertEqual(actions.count("DEFER_DUPLICATE_TASK"), 5)
        # the most-progressed copy (records=105, the last) is the primary
        primary = next(r for r in rows if r["action"] == "AUTO_RESUME")
        self.assertEqual(primary["records"], 105)

    def test_dedup_primary_is_deterministic_across_reorder(self) -> None:
        sids = self._sids(3)
        def mk(order):
            return [
                    _row(".claude-good-acct", "DEAD_MIDTOOL", session=sids[i],
                         task_sig="SIG", records=r) for i, r in order]
        avail = [_avail(".claude-good-acct", available=True, live=0)]
        rows_a = mk([(0, 10), (1, 30), (2, 20)])
        rows_b = mk([(2, 20), (0, 10), (1, 30)])   # different list order
        fleet_sessions.decide(rows_a, {}, avail)
        fleet_sessions.decide(rows_b, {}, avail)
        pa = next(r["session"] for r in rows_a if r["action"] == "AUTO_RESUME")
        pb = next(r["session"] for r in rows_b if r["action"] == "AUTO_RESUME")
        self.assertEqual(pa, pb)                    # sort, not list order, decides
        self.assertEqual(pa, sids[1])              # records=30 wins

    def test_live_sibling_covers_task_all_duplicates_defer(self) -> None:
        sids = self._sids(4)
        rows = [_row(".claude-good-acct", "LIVE", session=sids[0], task_sig="SIG")]
        rows += [_row(".claude-good-acct", "DEAD_MIDTOOL", session=s, task_sig="SIG")
                 for s in sids[1:]]
        avail = [_avail(".claude-good-acct", available=True, live=0)]
        fleet_sessions.decide(rows, {}, avail)
        # the LIVE one covers the task; ZERO resumable members auto-resume
        self.assertEqual(sum(1 for r in rows if r["action"] == "AUTO_RESUME"), 0)
        self.assertEqual(sum(1 for r in rows if r["action"] == "DEFER_DUPLICATE_TASK"), 3)
        self.assertEqual(rows[0]["action"], "SKIP_LIVE")

    def test_done_sibling_covers_task(self) -> None:
        sids = self._sids(3)
        rows = [_row(".claude-good-acct", "DONE", session=sids[0], task_sig="SIG")]
        rows += [_row(".claude-good-acct", "DEAD_MIDTOOL", session=s, task_sig="SIG")
                 for s in sids[1:]]
        avail = [_avail(".claude-good-acct", available=True, live=0)]
        fleet_sessions.decide(rows, {}, avail)
        self.assertEqual(sum(1 for r in rows if r["action"] == "AUTO_RESUME"), 0)
        self.assertEqual(sum(1 for r in rows if r["action"] == "DEFER_DUPLICATE_TASK"), 2)

    def test_deferred_duplicate_does_not_consume_rehome_cap(self) -> None:
        # 6 identical THROTTLED autonomous sessions + 1 healthy seat: only the primary
        # re-homes; the 5 duplicates defer as DUP (not THROTTLED) and don't eat cap slots.
        sids = self._sids(6)
        rows = [_row(".claude-owner-acct", "STOPPED_LIMIT", session=s,
                     task_sig="SIG", records=10 + i) for i, s in enumerate(sids)]
        throttle = {".claude-owner-acct": {"reset": "x"}}
        avail = [_avail(".claude-good-acct", available=True, live=0)]
        fleet_sessions.decide(rows, throttle, avail)
        actions = [r["action"] for r in rows]
        self.assertEqual(actions.count("AUTO_RESUME"), 1)
        self.assertEqual(actions.count("DEFER_DUPLICATE_TASK"), 5)
        self.assertEqual(actions.count("DEFER_THROTTLED"), 0)

    def test_org_disabled_duplicate_dedups_then_rehomes_primary(self) -> None:
        sids = self._sids(4)
        rows = [_row(".claude-orgdead-acct", "INFRA_ORG_DISABLED", session=s,
                     task_sig="SIG", records=10 + i) for i, s in enumerate(sids)]
        avail = [_avail(".claude-orgdead-acct", available=False),
                 _avail(".claude-good-acct", available=True, live=0)]
        fleet_sessions.decide(rows, {}, avail)
        actions = [r["action"] for r in rows]
        self.assertEqual(actions.count("AUTO_RESUME"), 1)       # dedup-then-rehome
        self.assertEqual(actions.count("DEFER_DUPLICATE_TASK"), 3)
        primary = next(r for r in rows if r["action"] == "AUTO_RESUME")
        self.assertTrue(primary["rehomed"])
        self.assertEqual(primary["resume_account"], ".claude-good-acct")

    def test_distinct_tasks_same_cwd_not_deduped(self) -> None:
        # same project+cwd, DIFFERENT task_sig -> both resume independently
        sids = self._sids(2)
        rows = [_row(".claude-good-acct", "DEAD_MIDTOOL", session=sids[0], task_sig="SIG_A"),
                _row(".claude-good-acct", "DEAD_MIDTOOL", session=sids[1], task_sig="SIG_B")]
        avail = [_avail(".claude-good-acct", available=True, live=0)]
        fleet_sessions.decide(rows, {}, avail)
        self.assertEqual(sum(1 for r in rows if r["action"] == "AUTO_RESUME"), 2)

    def test_non_autonomous_identical_tasks_not_deduped(self) -> None:
        # empty task_sig (the classify() output for non-autonomous rows) never dedups
        sids = self._sids(3)
        rows = [_row(".claude-good-acct", "DEAD_MIDTOOL", autonomous=False,
                     session=s, task_sig="") for s in sids]
        avail = [_avail(".claude-good-acct", available=True, live=0)]
        fleet_sessions.decide(rows, {}, avail)
        # none auto-resume (interactive) and none are mislabeled DEFER_DUPLICATE_TASK
        self.assertEqual(sum(1 for r in rows if r["action"] == "DEFER_DUPLICATE_TASK"), 0)

    def test_ledger_blocked_primary_hands_off_to_next(self) -> None:
        # the most-progressed copy is ledger-blocked (hit the attempt cap); the next-best
        # resumable copy must become primary instead of the task wedging.
        sids = self._sids(3)
        ledger = os.path.join(self._tmp, "resume_ledger.jsonl")
        import json as _json
        cap = int(os.environ.get("FAK_MAX_ATTEMPTS", "8"))  # track the code's default
        with open(ledger, "w", encoding="utf-8") as fh:
            for _ in range(cap):   # MAX_ATTEMPTS launch rows for sids[2] -> ledger-blocked
                fh.write(_json.dumps({"session": sids[2], "phase": "launched"}) + "\n")
        rows = [_row(".claude-good-acct", "DEAD_MIDTOOL", session=sids[0], task_sig="SIG", records=10),
                _row(".claude-good-acct", "DEAD_MIDTOOL", session=sids[1], task_sig="SIG", records=20),
                _row(".claude-good-acct", "DEAD_MIDTOOL", session=sids[2], task_sig="SIG", records=99)]
        avail = [_avail(".claude-good-acct", available=True, live=0)]
        fleet_sessions.decide(rows, {}, avail)
        primary = next(r for r in rows if r["action"] == "AUTO_RESUME")
        self.assertEqual(primary["session"], sids[1])   # records=20, not the blocked 99


class LiveCopyGateTest(unittest.TestCase):
    """#3459: one session id exists as many on-disk COPIES (a re-home writes the same
    uuid under another config dir), and scan() classifies every copy independently. The
    plan must (1) never queue a sid a live `claude --resume` driver is already advancing
    and (2) let only the NEWEST copy speak for the session, so a stale copy's synthetic
    "API Error" tail cannot become the whole session's verdict.

    The witnessed harm: resume_plan.json queued 94aea02a as STOPPED_APIERR/crashed while
    a live process was advancing a newer copy of it under .claude-day26NEW-netra; a
    `--live` watchdog tick would have fired a SECOND driver onto a live transcript."""

    SID = "94aea02a-0000-4000-8000-000000000000"

    @staticmethod
    def _plan(rows):
        """The rows that would reach resume_plan.json -- decide()'s own plan filter
        (fleet_sessions.py: `[plan_entry(r) for r in rows if r["action"] == "AUTO_RESUME"]`)."""
        return [r for r in rows if r["action"] == "AUTO_RESUME"]

    def test_apierr_copy_not_queued_when_a_live_copy_shares_the_uuid(self) -> None:
        # The acceptance fixture: same UUID, one alive copy + one APIERR copy.
        rows = [_row(".claude-stale-acct", "STOPPED_APIERR", session=self.SID,
                     last_ts="2026-08-04T10:00:00Z"),
                _row(".claude-day26NEW-netra", "LIVE", session=self.SID,
                     last_ts="2026-08-04T12:00:00Z")]
        avail = [_avail(".claude-stale-acct", available=True),
                 _avail(".claude-day26NEW-netra", available=True)]
        fleet_sessions.decide(rows, {}, avail)
        self.assertEqual(rows[0]["action"], "SKIP_LIVE_DRIVER")
        self.assertEqual(rows[1]["action"], "SKIP_LIVE")   # the live copy names itself
        self.assertEqual(self._plan(rows), [])             # never reaches resume_plan.json
        self.assertIsNone(rows[0]["resume_cmd"])           # and carries no launch command

    def test_apierr_copy_alone_is_still_queued(self) -> None:
        # Negative control: the SAME row with no live sibling and no census must still be
        # resumed -- the gate must not strand genuinely-crashed sessions.
        rows = [_row(".claude-stale-acct", "STOPPED_APIERR", session=self.SID,
                     last_ts="2026-08-04T10:00:00Z")]
        avail = [_avail(".claude-stale-acct", available=True)]
        fleet_sessions.decide(rows, {}, avail)
        self.assertEqual(rows[0]["action"], "AUTO_RESUME")
        self.assertEqual(len(self._plan(rows)), 1)

    def test_process_census_alone_suppresses_the_sid(self) -> None:
        # The 94aea02a case: NO on-disk copy classified LIVE (the live copy sat in an
        # account dir this scan never read) -- only the process table proves it alive.
        rows = [_row(".claude-stale-acct", "STOPPED_APIERR", session=self.SID,
                     last_ts="2026-08-04T10:00:00Z")]
        avail = [_avail(".claude-stale-acct", available=True)]
        fleet_sessions.decide(rows, {}, avail, live_sids={self.SID})
        self.assertEqual(rows[0]["action"], "SKIP_LIVE_DRIVER")
        self.assertEqual(self._plan(rows), [])

    def test_newest_copy_speaks_for_the_session(self) -> None:
        # Two crashed copies of one sid: the NEWER terminal turn decides, the older is
        # stamped SKIP_STALE_COPY so it can neither re-home nor consume a cap slot.
        old = _row(".claude-stale-acct", "STOPPED_APIERR", session=self.SID,
                   last_ts="2026-08-04T10:00:00Z")
        new = _row(".claude-day26NEW-netra", "DEAD_MIDTOOL", session=self.SID,
                   last_ts="2026-08-04T12:00:00Z")
        rows = [old, new]
        avail = [_avail(".claude-stale-acct", available=True),
                 _avail(".claude-day26NEW-netra", available=True)]
        fleet_sessions.decide(rows, {}, avail)
        self.assertEqual(old["action"], "SKIP_STALE_COPY")
        self.assertEqual(new["action"], "AUTO_RESUME")
        self.assertEqual(self._plan(rows), [new])

    def test_newest_copy_ranks_by_last_turn_not_file_mtime(self) -> None:
        # The mtime trap: a dead driver appends a synthetic "API Error" banner to a stale
        # PREFIX, so that copy's mtime (seen_utc) is the freshest on disk while its last
        # REAL turn is older. The turn timestamp must win over the file stamp.
        stale = _row(".claude-stale-acct", "STOPPED_APIERR", session=self.SID,
                     last_ts="2026-08-04T10:00:00Z", seen_utc="2026-08-04T23:59:00Z")
        real = _row(".claude-day26NEW-netra", "DEAD_MIDTOOL", session=self.SID,
                    last_ts="2026-08-04T12:00:00Z", seen_utc="2026-08-04T12:00:30Z")
        rows = [stale, real]
        avail = [_avail(".claude-stale-acct", available=True),
                 _avail(".claude-day26NEW-netra", available=True)]
        fleet_sessions.decide(rows, {}, avail)
        self.assertEqual(stale["action"], "SKIP_STALE_COPY")
        self.assertEqual(self._plan(rows), [real])

    def test_distinct_sids_are_untouched_by_the_copy_gate(self) -> None:
        # The newest-copy gate is keyed on the session id: two DIFFERENT crashed sessions
        # are not copies of each other and must both stay queued.
        a = _row(".claude-good-acct", "DEAD_MIDTOOL", session="aaaaaaaa-0000-4000-8000-000000000001")
        b = _row(".claude-good-acct", "DEAD_MIDTOOL", session="bbbbbbbb-0000-4000-8000-000000000002")
        rows = [a, b]
        fleet_sessions.decide(rows, {}, [_avail(".claude-good-acct", available=True)])
        self.assertEqual(len(self._plan(rows)), 2)

    def test_live_census_failure_leaves_the_gate_inert(self) -> None:
        # FAIL-OPEN: an unreadable process table (the census is Windows-only today) must
        # never strand a crashed session -- _live_resume_sids swallows it and returns
        # empty, so the on-disk evidence alone decides.
        with mock.patch.dict(sys.modules, {"resume_sweep": None}):
            self.assertEqual(fleet_sessions._live_resume_sids(), set())


class ResumeEscalationTest(unittest.TestCase):
    """A session that keeps dying IN PLACE on its own (healthy) account is re-homed onto a
    fresh seat after RESUME_ESCALATE_AFTER in-place attempts, instead of being re-pinned to
    the account it keeps dying on (#1342/#1345/#1859). The owner-throttled paths already
    re-home; this is the healthy-owner-but-repeatedly-crashing path they missed."""

    def setUp(self):
        # isolate the ledger read so _ledger_inplace_attempts sees ONLY what we write
        self._tmp = __import__("tempfile").mkdtemp()
        self._orig_reg = fleet_sessions.REG_DIR
        fleet_sessions.REG_DIR = self._tmp

    def tearDown(self):
        fleet_sessions.REG_DIR = self._orig_reg
        __import__("shutil").rmtree(self._tmp, ignore_errors=True)

    def _seed_ledger(self, sid, *, inplace=0, rehomed=0):
        import json as _json
        with open(os.path.join(self._tmp, "resume_ledger.jsonl"), "w", encoding="utf-8") as fh:
            for _ in range(inplace):
                fh.write(_json.dumps({"session": sid, "phase": "launched", "rehomed": False}) + "\n")
            for _ in range(rehomed):
                fh.write(_json.dumps({"session": sid, "phase": "launched", "rehomed": True}) + "\n")

    def _two_seat_avail(self):
        return [_avail(".claude-owner-acct", available=True, live=2),
                _avail(".claude-other-acct", available=True, live=0)]

    def test_first_crashes_resume_in_place(self) -> None:
        # under the threshold (1 prior in-place attempt < RESUME_ESCALATE_AFTER=2) -> owner
        sid = "aaaa1111-2222-3333-4444-555555555555"
        self._seed_ledger(sid, inplace=1)
        rows = [_row(".claude-owner-acct", "DEAD_MIDTOOL", session=sid)]
        fleet_sessions.decide(rows, {}, self._two_seat_avail())
        r = rows[0]
        self.assertEqual(r["action"], "AUTO_RESUME")
        self.assertFalse(r["rehomed"])
        self.assertEqual(r["resume_account"], r["account"])

    def test_repeat_crasher_escalates_to_another_seat(self) -> None:
        # at/over the threshold -> re-home onto the OTHER healthy seat, not the failing owner
        sid = "bbbb1111-2222-3333-4444-555555555555"
        self._seed_ledger(sid, inplace=2)
        rows = [_row(".claude-owner-acct", "DEAD_MIDTOOL", session=sid)]
        fleet_sessions.decide(rows, {}, self._two_seat_avail())
        r = rows[0]
        self.assertEqual(r["action"], "AUTO_RESUME")
        self.assertTrue(r["rehomed"])
        self.assertEqual(r["resume_account"], ".claude-other-acct")
        self.assertNotEqual(r["resume_account"], r["account"])
        # the plan/watchdog copies the transcript across before resuming
        self.assertIn("Copy-Item", r["resume_cmd"])

    def test_apierr_repeat_crasher_escalates(self) -> None:
        sid = "cccc1111-2222-3333-4444-555555555555"
        self._seed_ledger(sid, inplace=3)
        rows = [_row(".claude-owner-acct", "STOPPED_APIERR", session=sid)]
        fleet_sessions.decide(rows, {}, self._two_seat_avail())
        self.assertTrue(rows[0]["rehomed"])
        self.assertEqual(rows[0]["resume_account"], ".claude-other-acct")

    def test_escalation_falls_back_in_place_when_no_other_seat(self) -> None:
        # repeatedly crashing but the owner is the ONLY healthy seat -> stay in place
        sid = "dddd1111-2222-3333-4444-555555555555"
        self._seed_ledger(sid, inplace=5)
        rows = [_row(".claude-owner-acct", "DEAD_MIDTOOL", session=sid)]
        fleet_sessions.decide(rows, {}, [_avail(".claude-owner-acct", available=True)])
        r = rows[0]
        self.assertEqual(r["action"], "AUTO_RESUME")
        self.assertFalse(r["rehomed"])
        self.assertEqual(r["resume_account"], r["account"])

    def test_rehomed_attempts_do_not_count_as_in_place(self) -> None:
        # a session that was only ever RE-HOMED has an in-place streak of 0 -> in place,
        # so the escalation keys on failures on the OWN account, not total launches
        sid = "eeee1111-2222-3333-4444-555555555555"
        self._seed_ledger(sid, rehomed=4)
        rows = [_row(".claude-owner-acct", "DEAD_MIDTOOL", session=sid)]
        fleet_sessions.decide(rows, {}, self._two_seat_avail())
        self.assertFalse(rows[0]["rehomed"])

    def test_no_ledger_resumes_in_place(self) -> None:
        # first-ever crash (empty ledger) -> plain in-place resume, no escalation
        sid = "ffff1111-2222-3333-4444-555555555555"
        rows = [_row(".claude-owner-acct", "DEAD_MIDTOOL", session=sid)]
        fleet_sessions.decide(rows, {}, self._two_seat_avail())
        self.assertFalse(rows[0]["rehomed"])


class LoadSpreadDecisionTest(unittest.TestCase):
    """Mirrors internal/resume/rehome's load-spread contract: an AVAILABLE owner
    at/over REHOME_CAP live sessions spreads an in-place resume onto a strictly
    less-loaded admissible seat (the july7 429 pile-up shape); anything short of
    full proof keeps the in-place resume, and FAK_LOAD_REHOME=0 disables."""

    def setUp(self):
        # isolate the ledger read so _ledger_inplace_attempts sees an EMPTY ledger
        self._tmp = __import__("tempfile").mkdtemp()
        self._orig_reg = fleet_sessions.REG_DIR
        fleet_sessions.REG_DIR = self._tmp

    def tearDown(self):
        fleet_sessions.REG_DIR = self._orig_reg
        __import__("shutil").rmtree(self._tmp, ignore_errors=True)

    def test_apierr_on_loaded_owner_spreads_to_freer_seat(self) -> None:
        rows = [_row(".claude-loaded-acct", "STOPPED_APIERR")]
        availability = [
            _avail(".claude-loaded-acct", available=True, live=fleet_sessions.REHOME_CAP),
            _avail(".claude-idle-acct", available=True, live=1),
        ]
        fleet_sessions.decide(rows, {}, availability)
        r = rows[0]
        self.assertEqual(r["action"], "AUTO_RESUME")
        self.assertTrue(r["rehomed"])
        self.assertEqual(r["resume_account"], ".claude-idle-acct")
        self.assertIn("Copy-Item", r["resume_cmd"])

    def test_loaded_owner_without_freer_seat_resumes_in_place(self) -> None:
        rows = [_row(".claude-loaded-acct", "STOPPED_APIERR")]
        availability = [
            _avail(".claude-loaded-acct", available=True, live=fleet_sessions.REHOME_CAP + 2),
            _avail(".claude-alsofull-acct", available=True, live=fleet_sessions.REHOME_CAP),
        ]
        fleet_sessions.decide(rows, {}, availability)
        r = rows[0]
        self.assertEqual(r["action"], "AUTO_RESUME")
        self.assertFalse(r["rehomed"])
        self.assertEqual(r["resume_account"], r["account"])

    def test_under_cap_owner_resumes_in_place(self) -> None:
        rows = [_row(".claude-owner-acct", "DEAD_MIDTOOL")]
        availability = [
            _avail(".claude-owner-acct", available=True, live=fleet_sessions.REHOME_CAP - 1),
            _avail(".claude-idle-acct", available=True, live=0),
        ]
        fleet_sessions.decide(rows, {}, availability)
        self.assertFalse(rows[0]["rehomed"])
        self.assertEqual(rows[0]["resume_account"], rows[0]["account"])

    def test_kill_switch_restores_in_place(self) -> None:
        rows = [_row(".claude-loaded-acct", "STOPPED_APIERR")]
        availability = [
            _avail(".claude-loaded-acct", available=True, live=fleet_sessions.REHOME_CAP + 4),
            _avail(".claude-idle-acct", available=True, live=0),
        ]
        with mock.patch.object(fleet_sessions, "LOAD_REHOME", False):
            fleet_sessions.decide(rows, {}, availability)
        self.assertFalse(rows[0]["rehomed"])
        self.assertEqual(rows[0]["resume_account"], rows[0]["account"])


class AuthAnomalyEscalationTest(unittest.TestCase):
    """An INFRA_AUTH tail whose OWNER seat reads admissible (#619 positive evidence)
    is session-local, not a seat logout: a frozen banner from before a re-login, or a
    guard-gateway child whose recorded auth wiring died with its parent (cbdc1e5d,
    2026-07-02 -- every in-place resume answered the banner while the owner probed
    pong; the same transcript re-homed onto another seat resumed cleanly). Those rows
    take the in-place ladder and escalate to another seat; a seat WITHOUT positive
    evidence keeps the human-re-login BLOCKED_AUTH."""

    def setUp(self):
        self._tmp = __import__("tempfile").mkdtemp()
        self._orig_reg = fleet_sessions.REG_DIR
        fleet_sessions.REG_DIR = self._tmp

    def tearDown(self):
        fleet_sessions.REG_DIR = self._orig_reg
        __import__("shutil").rmtree(self._tmp, ignore_errors=True)

    def _seed_ledger(self, sid, *, inplace=0):
        import json as _json
        with open(os.path.join(self._tmp, "resume_ledger.jsonl"), "w", encoding="utf-8") as fh:
            for _ in range(inplace):
                fh.write(_json.dumps({"session": sid, "phase": "launched", "rehomed": False}) + "\n")

    def _two_seat_avail(self):
        return [_avail(".claude-owner-acct", available=True, live=2),
                _avail(".claude-other-acct", available=True, live=0)]

    def test_auth_banner_on_proven_owner_resumes_in_place_first(self) -> None:
        # fresh anomaly (no prior in-place attempts) -> retry on the proven owner:
        # covers the frozen-banner-after-relogin case, where in place works.
        rows = [_row(".claude-owner-acct", "INFRA_AUTH")]
        fleet_sessions.decide(rows, {}, self._two_seat_avail())
        r = rows[0]
        self.assertEqual(r["action"], "AUTO_RESUME")
        self.assertFalse(r["rehomed"])
        self.assertEqual(r["resume_account"], r["account"])

    def test_auth_banner_repeat_failures_escalate_to_another_seat(self) -> None:
        # the banner survived RESUME_ESCALATE_AFTER in-place attempts -> the session's
        # auth wiring is dead on this seat; re-home onto the other proven seat.
        sid = "abba1111-2222-3333-4444-555555555555"
        self._seed_ledger(sid, inplace=2)
        rows = [_row(".claude-owner-acct", "INFRA_AUTH", session=sid)]
        fleet_sessions.decide(rows, {}, self._two_seat_avail())
        r = rows[0]
        self.assertEqual(r["action"], "AUTO_RESUME")
        self.assertTrue(r["rehomed"])
        self.assertEqual(r["resume_account"], ".claude-other-acct")
        self.assertIn("Copy-Item", r["resume_cmd"])

    def test_auth_banner_on_unproven_owner_still_blocks(self) -> None:
        # owner seat absent from the snapshot -> the banner stands: human /login.
        rows = [_row(".claude-auth-acct", "INFRA_AUTH")]
        fleet_sessions.decide(rows, {}, [_avail(".claude-good-acct", available=True)])
        self.assertEqual(rows[0]["action"], "BLOCKED_AUTH")
        self.assertFalse(rows[0]["rehomed"])

    def test_auth_banner_on_carried_owner_verdict_still_blocks(self) -> None:
        # #619 launch boundary: a carried 'available' is not positive evidence the
        # seat can serve -- the auth block stands until a fresh probe proves it.
        rows = [_row(".claude-owner-acct", "INFRA_AUTH")]
        carried = _avail(".claude-owner-acct", available=True, verdict_source="carried")
        fleet_sessions.decide(rows, {}, [carried])
        self.assertEqual(rows[0]["action"], "BLOCKED_AUTH")
        self.assertFalse(rows[0]["rehomed"])


class TaskSigClassifyTest(unittest.TestCase):
    """task_sig must come from the real first instruction, ignoring harness wrappers
    and the fixed resume prompt a re-home injects."""

    def test_first_instruction_skips_wrappers_and_resume_prompt(self) -> None:
        head = [
            {"type": "user", "message": {"content": "Caveat: local command output below"}},
            {"type": "system", "message": {"content": "<system-reminder>be good</system-reminder>"}},
            {"type": "user", "message": {"content": fleet_sessions.RESUME_PROMPT}},
            {"type": "user", "message": {"content": "Resolve ONE diverged git repository safely, then STOP."}},
        ]
        instr = fleet_sessions._first_instruction(head)
        self.assertEqual(instr, "Resolve ONE diverged git repository safely, then STOP.")

    def test_resume_prompt_alone_yields_no_signature(self) -> None:
        # a re-homed transcript whose head is ONLY the resume prompt must not collapse
        # to a shared signature -- it yields an empty instruction.
        head = [{"type": "user", "message": {"content": fleet_sessions.RESUME_PROMPT}}]
        self.assertEqual(fleet_sessions._first_instruction(head), "")

    def test_same_instruction_same_sig_diff_sid(self) -> None:
        a = fleet_sessions._task_sig("proj", "/cwd", "do the thing")
        b = fleet_sessions._task_sig("proj", "/cwd", "do the thing")
        c = fleet_sessions._task_sig("proj", "/cwd", "do a DIFFERENT thing")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 16)

    def _slash_goal_head(self, goal: str) -> list:
        """The real head a `/goal` session opens with: a <local-command-caveat> block
        and an /effort preamble that are BYTE-IDENTICAL across every such session,
        then the distinguishing /goal directive."""
        return [
            {"type": "user", "message": {"content":
                "<local-command-caveat>Caveat: The messages below were generated by the "
                "user while running local commands. DO NOT respond to these messages."
                "</local-command-caveat>"}},
            {"type": "user", "message": {"content":
                "<command-name>/effort</command-name> <command-message>effort</command-message> "
                "<command-args>ultracode</command-args>"}},
            {"type": "user", "message": {"content":
                "<local-command-stdout>Set effort level to ultracode (this session only): "
                "xhigh + dynamic workflow orchestration</local-command-stdout>"}},
            {"type": "user", "message": {"content":
                f"<command-name>/goal</command-name> <command-args>{goal}</command-args>"}},
            {"type": "user", "message": {"content":
                f'A session-scoped Stop hook is now active with condition: "{goal}".'}},
        ]

    def test_caveat_and_effort_preamble_do_not_collapse_distinct_goals(self) -> None:
        # Regression for the caveat-wrapper false-dedup collapse: 15 distinct /goal
        # workers were stranded as "duplicates" because _first_instruction returned the
        # identical <local-command-caveat> boilerplate (then the identical /effort arg).
        a = fleet_sessions._first_instruction(self._slash_goal_head("model routing first class"))
        b = fleet_sessions._first_instruction(self._slash_goal_head("progress epic 595"))
        self.assertIn("model routing first class", a)
        self.assertIn("progress epic 595", b)
        # neither the caveat nor the /effort "ultracode" arg leaks into the identity
        self.assertNotIn("Caveat", a)
        self.assertNotIn("ultracode", a)
        self.assertNotEqual(
            fleet_sessions._task_sig("proj", "/cwd", a),
            fleet_sessions._task_sig("proj", "/cwd", b))

    def test_config_command_args_alone_are_not_the_task(self) -> None:
        # an /effort-only head (config command, no task command) must NOT adopt
        # "ultracode" as its task identity.
        head = [
            {"type": "user", "message": {"content":
                "<command-name>/effort</command-name> <command-args>ultracode</command-args>"}},
            {"type": "user", "message": {"content": "actually do the real work here"}},
        ]
        self.assertEqual(fleet_sessions._first_instruction(head), "actually do the real work here")


class RegistrySummaryContractTest(unittest.TestCase):
    """The `registry` mode must emit a JSON OBJECT as its last stdout line, the
    cross-language contract the Go dispatch tick's lastJSONObject() parser depends
    on (dispatch_tick.go dispatchRunJSON / dispatchRefreshRegistry). Left untested,
    the helper silently drifted to human-text-only output, so a SUCCESSFUL refresh
    was reported registry_refresh.ok=false on every tick -- a false-failure that
    masked the real state. These lock the shape without a full fleet scan."""

    def test_registry_summary_is_json_object_with_required_keys(self) -> None:
        summary = fleet_sessions.registry_summary(
            "a/sessions.json", "b/resume_plan.json", nsess=5, n=2, probe_verdicts=[])
        # must round-trip through json as an OBJECT (dict), not a bare scalar/list --
        # lastJSONObject() only accepts a trailing {...}.
        obj = json.loads(json.dumps(summary))
        self.assertIsInstance(obj, dict)
        for key in ("ok", "mode", "sessions", "auto_resume", "probed", "wrote"):
            self.assertIn(key, obj)
        self.assertEqual(obj["mode"], "registry")
        self.assertTrue(obj["ok"])
        self.assertEqual(obj["sessions"], 5)
        self.assertEqual(obj["auto_resume"], 2)
        self.assertEqual(obj["wrote"], ["a/sessions.json", "b/resume_plan.json"])

    def test_probed_count_reflects_verdicts(self) -> None:
        summary = fleet_sessions.registry_summary(
            "s.json", "p.json", nsess=9, n=0, probe_verdicts=[{"a": 1}, {"b": 2}])
        self.assertEqual(summary["probed"], 2)

    def test_path_values_are_json_safe_strings(self) -> None:
        # sp/pp arrive as pathlib.Path in production; str() keeps json.dumps happy.
        summary = fleet_sessions.registry_summary(
            Path("x") / "sessions.json", Path("y") / "resume_plan.json",
            nsess=1, n=1, probe_verdicts=[])
        obj = json.loads(json.dumps(summary))  # must not raise on Path inputs
        self.assertTrue(all(isinstance(w, str) for w in obj["wrote"]))


if __name__ == "__main__":
    unittest.main()



class RearmWindowOverrideTest(unittest.TestCase):
    def test_active_rearm_overrides_window_until_consumed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "resume_ledger.jsonl"
            ledger.write_text("\n".join([
                '{bad json',
                '{"session":"old","phase":"rearm"}',
                '{"session":"consumed","phase":"rearm"}',
                '{"session":"consumed","phase":"launched"}',
                '{"session":"settled","phase":"rearm"}',
                '{"session":"settled","action":"consolidate_operator"}',
            ]) + "\n", encoding="utf-8")
            self.assertEqual(fleet_sessions.active_rearmed_session_ids(td), {"old"})

    def test_scan_window_gate_names_rearmed_exception(self):
        source = (Path(__file__).resolve().parent / "fleet_sessions.py").read_text(encoding="utf-8")
        self.assertIn("and base not in rearmed", source)
        self.assertIn("rearmed = active_rearmed_session_ids(REG_DIR)", source)
