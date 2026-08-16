#!/usr/bin/env python3
r"""Tests for launch_admission -- the single launch-admission gate (#617).

The headline test reproduces the 2026-06-24 storm: 15 ``claude --resume`` launches
fired onto ONE account inside ~60s. The gate must admit AT MOST the ceiling and
hand every excess launch a STRUCTURED ``DEFER`` reason bound to a retry time --
the throttled account's reset, or the window roll-off.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import launch_admission as LA  # noqa: E402

# The incident window: 22:04:03Z -> 22:04:59Z, ~15 launches ~4s apart.
START = datetime(2026, 6, 24, 22, 4, 3, tzinfo=timezone.utc)


def _counts(verdicts):
    admitted = [v for v in verdicts if v["verdict"] == LA.VERDICT_ADMIT]
    deferred = [v for v in verdicts if v["verdict"] == LA.VERDICT_DEFER]
    return admitted, deferred


class ThrottledBurstTest(unittest.TestCase):
    """The actual incident: q-netra was THROTTLED when the 15-burst hit."""

    def test_15_onto_one_throttled_account_all_defer_bound_to_reset(self):
        reset = "Jun 25, 1pm (America/Los_Angeles)"
        verdicts = LA.simulate_burst(
            ".claude-q-netra", 15, start=START,
            throttled=True, throttle_reset=reset,
            max_per_account=3, window_min=5, global_cap=10,
        )
        admitted, deferred = _counts(verdicts)
        # The throttle-gate is the most-specific refusal: ZERO admitted (<= cap),
        # every launch deferred and BOUND TO THE ACCOUNT'S RESET.
        self.assertEqual(len(admitted), 0)
        self.assertEqual(len(deferred), 15)
        for v in deferred:
            self.assertEqual(v["reason"], LA.REASON_THROTTLED)
            self.assertEqual(v["retry_after"], reset)


class RateCeilingBurstTest(unittest.TestCase):
    """No throttle verdict yet -> the per-account RATE ceiling must still cap it."""

    def test_15_onto_one_account_caps_at_ceiling(self):
        cap = 3
        verdicts = LA.simulate_burst(
            ".claude-q-netra", 15, start=START,
            max_per_account=cap, window_min=5, global_cap=50,
        )
        admitted, deferred = _counts(verdicts)
        # Exactly `cap` admitted; the remaining 12 deferred with a window-bound retry.
        self.assertEqual(len(admitted), cap)
        self.assertEqual(len(deferred), 15 - cap)
        for v in deferred:
            self.assertEqual(v["reason"], LA.REASON_RATE)
            self.assertIsNotNone(v["retry_after"])
            # retry_after is the oldest-in-window launch + the window: in the future
            # of the first launch, parseable, and never None.
            self.assertIsNotNone(LA.parse_ts(v["retry_after"]))
            self.assertGreater(LA.parse_ts(v["retry_after"]), START)

    def test_admitted_are_the_first_n(self):
        verdicts = LA.simulate_burst(
            ".claude-q-netra", 6, start=START,
            max_per_account=2, window_min=5, global_cap=50,
        )
        kinds = [v["verdict"] for v in verdicts]
        self.assertEqual(kinds, [LA.VERDICT_ADMIT, LA.VERDICT_ADMIT]
                         + [LA.VERDICT_DEFER] * 4)


class GlobalCapTest(unittest.TestCase):
    """Distinct accounts each under their own ceiling can still storm the fleet."""

    def test_global_cap_defers_even_a_fresh_account(self):
        # 4 prior launches across 4 DIFFERENT accounts, all inside the window.
        prior = [START + timedelta(seconds=10 * i) for i in range(4)]
        v = LA.admit(
            ".claude-fresh-netra",
            now=START + timedelta(seconds=50),
            account_launches=[],          # this account has launched 0 times
            global_launches=prior,        # but the fleet is at the cap
            max_per_account=3, window_min=5, global_cap=4,
        )
        self.assertEqual(v["verdict"], LA.VERDICT_DEFER)
        self.assertEqual(v["reason"], LA.REASON_GLOBAL_CAP)
        self.assertEqual(v["retry_after"], LA._fmt(prior[0] + timedelta(minutes=5)))


class AdmitWhenClearTest(unittest.TestCase):
    def test_empty_ledger_admits(self):
        v = LA.admit(".claude-day24-netra", now=START,
                     account_launches=[], global_launches=[],
                     max_per_account=3, window_min=5, global_cap=10)
        self.assertEqual(v["verdict"], LA.VERDICT_ADMIT)
        self.assertIsNone(v["reason"])

    def test_window_rolls_off_old_launches(self):
        # 5 launches, but all OLDER than the 5-minute window -> no pressure now.
        old = [START - timedelta(minutes=10) - timedelta(seconds=i) for i in range(5)]
        v = LA.admit(".claude-q-netra", now=START,
                     account_launches=old, global_launches=old,
                     max_per_account=3, window_min=5, global_cap=10)
        self.assertEqual(v["verdict"], LA.VERDICT_ADMIT)


class AccountTagTest(unittest.TestCase):
    def test_dir_tag_and_bare_tag_match(self):
        self.assertEqual(LA._account_tag(".claude-q-netra"), "q-netra")
        self.assertEqual(LA._account_tag("q-netra"), "q-netra")
        self.assertEqual(LA._account_tag(".claude"), "default")
        self.assertEqual(LA._account_tag("opencode-glm"), "glm")
        self.assertEqual(LA._account_tag("C:/Users/u/.claude-q-netra"), "q-netra")


class LedgerRoundTripTest(unittest.TestCase):
    """The live CLI path: count launches from the durable ledger, append on ADMIT."""

    def test_load_counts_only_this_account_and_skips_deferrals(self):
        with tempfile.TemporaryDirectory() as d:
            led = os.path.join(d, "resume_ledger.jsonl")
            rows = [
                {"ts": "2026-06-24T22:04:03Z", "resume_account": ".claude-q-netra"},
                {"ts": "2026-06-24T22:04:07Z", "resume_account": ".claude-q-netra"},
                {"ts": "2026-06-24T22:04:11Z", "account": ".claude-day24-netra"},
                # a deferral is NOT launch pressure -> must be skipped.
                {"ts": "2026-06-24T22:04:15Z", "resume_account": ".claude-q-netra",
                 "phase": "deferred"},
                "{ this is not json }",  # malformed -> skipped, never crashes
            ]
            with open(led, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
            acct, glob = LA.load_launches(led, ".claude-q-netra")
            self.assertEqual(len(acct), 2)   # two real q-netra launches
            self.assertEqual(len(glob), 3)   # + the day24 launch; deferral skipped

    def test_missing_ledger_is_empty_not_error(self):
        acct, glob = LA.load_launches("/no/such/ledger.jsonl", ".claude-q-netra")
        self.assertEqual((acct, glob), ([], []))

    def test_record_then_reload_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            led = os.path.join(d, "resume_ledger.jsonl")
            LA.record_launch(led, ".claude-q-netra", START, cause="test")
            acct, glob = LA.load_launches(led, ".claude-q-netra")
            self.assertEqual(len(acct), 1)
            self.assertEqual(acct[0], START)


class VocabularyTest(unittest.TestCase):
    def test_emittable_reasons_are_the_three_documented_tokens(self):
        self.assertEqual(
            set(LA.EMITTABLE_REASONS),
            {LA.REASON_THROTTLED, LA.REASON_GLOBAL_CAP, LA.REASON_RATE},
        )


class AdmitOrDeferSeamTest(unittest.TestCase):
    """The one-call launcher seam (#3552): load the ledger, decide, optionally record.

    This is what a launcher's per-spawn gate call exercises -- so a burst that hits
    the per-account ceiling gets a structured DEFER instead of firing.
    """

    def test_defers_when_over_per_account_ceiling(self):
        with tempfile.TemporaryDirectory() as d:
            led = os.path.join(d, "resume_ledger.jsonl")
            # Seed the ledger AT the per-account ceiling, all inside the window.
            for i in range(3):
                LA.record_launch(led, ".claude-q-netra", START + timedelta(seconds=i))
            v = LA.admit_or_defer(
                ".claude-q-netra", ledger_path=led,
                now=START + timedelta(seconds=10),
                max_per_account=3, window_min=5, global_cap=50,
            )
            self.assertEqual(v["verdict"], LA.VERDICT_DEFER)
            self.assertEqual(v["reason"], LA.REASON_RATE)
            self.assertIsNotNone(v["retry_after"])

    def test_admits_and_records_when_clear(self):
        with tempfile.TemporaryDirectory() as d:
            led = os.path.join(d, "resume_ledger.jsonl")
            v = LA.admit_or_defer(
                ".claude-day24-netra", ledger_path=led, now=START, record=True,
                max_per_account=3, window_min=5, global_cap=10,
            )
            self.assertEqual(v["verdict"], LA.VERDICT_ADMIT)
            # the ADMIT was appended -> a reload sees exactly one launch onto it.
            acct, _ = LA.load_launches(led, ".claude-day24-netra")
            self.assertEqual(len(acct), 1)

    def test_admit_does_not_record_without_the_flag(self):
        with tempfile.TemporaryDirectory() as d:
            led = os.path.join(d, "resume_ledger.jsonl")
            LA.admit_or_defer(".claude-day24-netra", ledger_path=led, now=START,
                              max_per_account=3, window_min=5, global_cap=10)
            acct, _ = LA.load_launches(led, ".claude-day24-netra")
            self.assertEqual(len(acct), 0)   # decide-only: no side effect on the ledger

    def test_throttle_reset_implies_throttled_defer(self):
        with tempfile.TemporaryDirectory() as d:
            led = os.path.join(d, "resume_ledger.jsonl")
            v = LA.admit_or_defer(".claude-q-netra", ledger_path=led, now=START,
                                  throttle_reset="Jun 25, 1pm")
            self.assertEqual(v["verdict"], LA.VERDICT_DEFER)
            self.assertEqual(v["reason"], LA.REASON_THROTTLED)


class WavePlanAdmissionTest(unittest.TestCase):
    """#6492: the plan must ask the SAME gate the launch self-gates on.

    Witnessed 2026-08-11: a dry run advertised ``granted=4, shortfall=0``; the matching
    ``-Launch`` dispatched ``0/4`` with every lane DEFERRED ``LAUNCH_RATE_EXCEEDED``
    (a second report hit ``GLOBAL_LAUNCH_CAP``). The seats were real -- the plan simply
    never consulted admission, so it advertised a grant that was not launchable. These
    cases pin the planner to the launcher's own answer."""

    def test_cooldown_mismatch_plan_grants_zero_not_four(self):
        # The reported shape: one account already AT its ceiling inside the window,
        # four lanes proposed onto it. The old planner said 4/4.
        hist = {".claude-q-netra": [START + timedelta(seconds=i) for i in range(3)]}
        plan = LA.plan_lanes([".claude-q-netra"] * 4,
                             now=START + timedelta(seconds=10),
                             account_launches=hist, global_launches=hist[".claude-q-netra"],
                             max_per_account=3, window_min=5, global_cap=50)
        self.assertEqual(plan["requested"], 4)
        self.assertEqual(plan["granted"], 0)          # was: 4
        self.assertEqual(plan["shortfall"], 4)        # was: 0
        self.assertEqual(plan["reasons"], {LA.REASON_RATE: 4})
        self.assertIsNotNone(plan["retry_after"])     # an honest "come back at"
        self.assertEqual([l["verdict"] for l in plan["lanes"]], [LA.VERDICT_DEFER] * 4)

    def test_global_cap_mismatch_is_also_planned(self):
        # The second reported report: fresh accounts, but the FLEET is at its cap.
        glob = [START + timedelta(seconds=i) for i in range(10)]
        plan = LA.plan_lanes([f".claude-fresh-{i}" for i in range(3)],
                             now=START + timedelta(seconds=20),
                             global_launches=glob,
                             max_per_account=3, window_min=5, global_cap=10)
        self.assertEqual(plan["granted"], 0)
        self.assertEqual(plan["shortfall"], 3)
        self.assertEqual(plan["reasons"], {LA.REASON_GLOBAL_CAP: 3})

    def test_plan_is_stateful_so_n_lanes_on_one_account_stop_at_the_ceiling(self):
        # The subtle half of the bug: a stateless per-lane check would ADMIT all 5
        # (each lane sees an empty ledger). Lane i must see lanes 0..i-1.
        plan = LA.plan_lanes([".claude-q-netra"] * 5, now=START,
                             max_per_account=3, window_min=5, global_cap=50)
        self.assertEqual(plan["granted"], 3)          # exactly the ceiling
        self.assertEqual(plan["shortfall"], 2)
        self.assertEqual([l["verdict"] for l in plan["lanes"]],
                         [LA.VERDICT_ADMIT] * 3 + [LA.VERDICT_DEFER] * 2)

    def test_clear_fleet_still_plans_the_full_grant(self):
        # The gate must not become a brake: distinct fresh accounts all plan ADMIT.
        plan = LA.plan_lanes([f".claude-fresh-{i}" for i in range(4)], now=START,
                             max_per_account=3, window_min=5, global_cap=10)
        self.assertEqual((plan["granted"], plan["shortfall"]), (4, 0))
        self.assertEqual(plan["reasons"], {})
        self.assertIsNone(plan["retry_after"])

    def test_throttled_lane_is_deferred_with_its_reset_as_retry_after(self):
        reset = "Jun 25, 1pm (America/Los_Angeles)"
        plan = LA.plan_lanes([".claude-q-netra", ".claude-day24-netra"], now=START,
                             throttle_resets={".claude-q-netra": reset},
                             max_per_account=3, window_min=5, global_cap=10)
        self.assertEqual((plan["granted"], plan["shortfall"]), (1, 1))
        self.assertEqual(plan["reasons"], {LA.REASON_THROTTLED: 1})
        self.assertEqual(plan["retry_after"], reset)

    def test_earliest_retry_wins_over_a_later_one(self):
        self.assertEqual(
            LA._earliest_retry([None, "2026-06-24T22:09:03Z", "2026-06-24T22:06:03Z"]),
            "2026-06-24T22:06:03Z")

    def test_plan_reads_the_ledger_and_records_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            led = os.path.join(d, "resume_ledger.jsonl")
            for i in range(3):
                LA.record_launch(led, ".claude-q-netra", START + timedelta(seconds=i))
            plan = LA.plan_wave([".claude-q-netra", ".claude-day24-netra"],
                                ledger_path=led, now=START + timedelta(seconds=10),
                                max_per_account=3, window_min=5, global_cap=50)
            self.assertEqual(plan["granted"], 1)      # only the clear account
            self.assertEqual(plan["lanes"][0]["reason"], LA.REASON_RATE)
            # planning must never consume launch budget
            acct, glob = LA.load_launches(led, ".claude-day24-netra")
            self.assertEqual((len(acct), len(glob)), (0, 3))

    def test_plan_agrees_with_the_launchers_own_per_spawn_gate(self):
        # The anti-drift property: whatever the plan says about lane 0, the launcher's
        # own admit_or_defer on the same ledger+clock must say too.
        with tempfile.TemporaryDirectory() as d:
            led = os.path.join(d, "resume_ledger.jsonl")
            for i in range(3):
                LA.record_launch(led, ".claude-q-netra", START + timedelta(seconds=i))
            now = START + timedelta(seconds=10)
            plan = LA.plan_wave([".claude-q-netra"], ledger_path=led, now=now,
                                max_per_account=3, window_min=5, global_cap=50)
            live = LA.admit_or_defer(".claude-q-netra", ledger_path=led, now=now,
                                     max_per_account=3, window_min=5, global_cap=50)
            self.assertEqual(plan["lanes"][0]["verdict"], live["verdict"])
            self.assertEqual(plan["lanes"][0]["reason"], live["reason"])

    def test_plan_cli_exits_3_on_shortfall_and_0_when_clear(self):
        import io
        import contextlib
        with tempfile.TemporaryDirectory() as d:
            led = os.path.join(d, "resume_ledger.jsonl")
            for i in range(3):
                LA.record_launch(led, ".claude-q-netra", START + timedelta(seconds=i))
            argv = ["plan", "--ledger", led, "--now", "2026-06-24T22:04:13Z",
                    "--max-per-account", "3", "--window-min", "5", "--global-cap", "50"]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = LA.main(argv + ["--account", ".claude-q-netra"])
            self.assertEqual(rc, 3)                   # not launchable -> non-zero
            out = json.loads(buf.getvalue())
            self.assertEqual(out["schema"], LA.PLAN_SCHEMA)
            self.assertEqual(out["shortfall"], 1)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = LA.main(argv + ["--account", ".claude-day24-netra"])
            self.assertEqual(rc, 0)                   # clear -> the plan is honest
            self.assertEqual(json.loads(buf.getvalue())["granted"], 1)


class LauncherWiringTest(unittest.TestCase):
    """The gate is only useful once a launcher routes THROUGH it (#3552, the #617
    unwired-gate class). Content checks that the wave launcher invokes the gate and
    paces its spawns -- the LIVE-wave ablation is a separate manual follow-up."""

    WAVE = Path(__file__).resolve().parent / "launch_wave_detached.ps1"

    def test_wave_launcher_invokes_the_admission_gate(self):
        text = self.WAVE.read_text(encoding="utf-8")
        self.assertIn("launch_admission.py", text)

    def test_wave_plan_consults_admission_before_advertising_a_grant(self):
        # #6492: the DRY RUN, not just the launch, must route through the gate --
        # otherwise the plan advertises a grant the launch cannot deliver.
        text = self.WAVE.read_text(encoding="utf-8")
        self.assertIn("Invoke-AdmissionPlan", text)
        self.assertIn("'plan'", text)              # the decide-only verb
        self.assertIn("admissionShortfall", text)  # folded into the plan's shortfall

    def test_wave_launcher_refills_until_requested_total_or_deadline(self):
        text = self.WAVE.read_text(encoding="utf-8")
        self.assertIn("[int]$RefillCadenceSeconds = 60", text)
        self.assertIn("[int]$RefillForMinutes = 240", text)
        self.assertIn("while ($remaining -gt 0 -and (Get-Date) -lt $deadline)", text)
        self.assertIn("Start-Sleep -Seconds $RefillCadenceSeconds", text)
        self.assertIn("'-NoRefill', '-Launch'", text)
        self.assertIn("WAVE REFILL DONE", text)
        self.assertIn("WAVE REFILL STOP", text)
        self.assertIn("WAVE WAIT         initial allocation empty", text)
        self.assertIn("WAVE CENSUS", text)
        self.assertIn("os_worker_procs=", text)
        self.assertIn("seat_free=", text)

    def test_wave_launcher_paces_spawns_with_a_jittered_delay(self):
        text = self.WAVE.read_text(encoding="utf-8")
        self.assertIn("FAK_LAUNCH_SPAWN_PACING_MS", text)   # env-overridable pacing knob
        self.assertIn("Start-Sleep", text)                  # the paced delay
        self.assertIn("Get-Random", text)                   # + jitter to de-sync spawns

    def test_wave_can_explicitly_extend_the_scheduled_issue_gardener(self):
        text = self.WAVE.read_text(encoding="utf-8")
        self.assertIn("[switch]$ExtendStanding", text)
        self.assertIn("Get-StandingGardenerContract", text)
        self.assertIn("issue_resolve_dispatch\\.py", text)
        self.assertIn("STANDING_GARDENER_CONTRACT_MISMATCH", text)
        self.assertIn("extension_mode", text)

    def test_standing_extension_is_forwarded_to_the_live_launch(self):
        text = self.WAVE.read_text(encoding="utf-8")
        self.assertIn("$fwd.ExtendStanding = $true", text)
        self.assertIn("$fwd.StandingTaskName = $StandingTaskName", text)


if __name__ == "__main__":
    unittest.main()
