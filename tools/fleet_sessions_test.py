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
         supervised=False, session="11111111-2222-3333-4444-555555555555",
         task_sig="", records=1, seen_utc=""):
    """A minimal session row shaped like classify() output for decide().

    task_sig/records/seen_utc default to the no-dedup case (empty sig) so the
    existing re-home tests are untouched; dedup tests set them explicitly."""
    return {
        "account": account, "disp": disp, "autonomous": autonomous,
        "supervised": supervised, "cwd": cwd if cwd is not None else os.getcwd(),
        "project": project, "session": session, "git": "master",
        "age_min": 5.0, "last": "", "throttle_reset": None,
        "task_sig": task_sig, "records": records, "seen_utc": seen_utc,
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


if __name__ == "__main__":
    unittest.main()
