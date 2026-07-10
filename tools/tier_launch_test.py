#!/usr/bin/env python3
"""Golden parity tests for tools/tier_launch.py against the Go source of truth.

tier_launch.py is the Python MIRROR of internal/dispatchtick/launchprofile.go (+
the tier label grammar in tiertag.go and the tier ordering in
modelroute/tierpolicy.go). These tests keep the two launch surfaces in LOCKSTEP two
ways:

  1. A value-table golden: every canonical profile, the default bucket->profile
     table, and the label grammar are asserted against the same cases the Go unit
     tests use (mirror of TestDefaultTierLaunchTable / TestLaunchProfileForIssue /
     TestIssueTierFromLabels).

  2. A drift-guard that READS internal/dispatchtick/launchprofile.go and asserts the
     mirrored constant literals (model ids, effort/ultracode/ultra tokens) are still
     present there verbatim — so a change to a Go constant fails this test instead of
     silently diverging the Python launcher.

Run: ``python3 tools/tier_launch_test.py``.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import tier_launch as tl  # noqa: E402
import issue_resolve_dispatch as ird  # noqa: E402

REPO = TOOLS.parent
GO_LAUNCHPROFILE = REPO / "internal" / "dispatchtick" / "launchprofile.go"


class TestCanonicalProfiles(unittest.TestCase):
    """The four canonical profiles mirror launchprofile.go's Profile* vars."""

    def test_profile_values(self):
        self.assertEqual(tl.PROFILE_OPUS_XHIGH, tl.LaunchProfile("claude-opus-4-8", "xhigh", False))
        self.assertEqual(tl.PROFILE_OPUS_ULTRACODE, tl.LaunchProfile("claude-opus-4-8", "", True))
        self.assertEqual(tl.PROFILE_FABLE_XHIGH, tl.LaunchProfile("claude-fable-5", "xhigh", False))
        self.assertEqual(tl.PROFILE_FABLE_ULTRACODE, tl.LaunchProfile("claude-fable-5", "", True))

    def test_effort_and_ultracode_are_mutually_exclusive_per_profile(self):
        # Ultracode already implies xhigh, so a canonical profile never sets both.
        for p in (tl.PROFILE_OPUS_XHIGH, tl.PROFILE_OPUS_ULTRACODE,
                  tl.PROFILE_FABLE_XHIGH, tl.PROFILE_FABLE_ULTRACODE):
            self.assertFalse(p.ultracode and bool(p.effort), p)

    def test_constant_literals(self):
        self.assertEqual(tl.WORKER_MODEL_OPUS, "claude-opus-4-8")
        self.assertEqual(tl.WORKER_MODEL_FABLE, "claude-fable-5")
        self.assertEqual(tl.EFFORT_XHIGH, "xhigh")
        self.assertEqual(tl.ULTRACODE_SETTINGS_ARG, '{"ultracode":true}')
        self.assertEqual(tl.ULTRA_LABEL, "tier/ultra")


class TestDefaultTable(unittest.TestCase):
    """Mirror of TestDefaultTierLaunchTable — the routine/normal/hard/ultra rows."""

    def test_table(self):
        table = tl.default_tier_launch_table()
        self.assertEqual(table[tl.BUCKET_ROUTINE], tl.PROFILE_FABLE_XHIGH)
        self.assertEqual(table[tl.BUCKET_NORMAL], tl.PROFILE_OPUS_XHIGH)
        self.assertEqual(table[tl.BUCKET_HARD], tl.PROFILE_OPUS_ULTRACODE)
        self.assertEqual(table[tl.BUCKET_ULTRA], tl.PROFILE_FABLE_ULTRACODE)


class TestTierGrammar(unittest.TestCase):
    """Mirror of TestIssueTierFromLabels — the safe-degrade label grammar."""

    def test_valid_same_tier_both_roles(self):
        it, flags = tl.issue_tier_from_labels(["tier/T1-required", "tier/T1-optimal"])
        self.assertTrue(it.has_tier)
        self.assertEqual((it.required, it.optimal), (tl.TIER_T1, tl.TIER_T1))
        self.assertEqual(flags, [])

    def test_optimal_more_demanding_than_required_is_allowed(self):
        # required T2 (floor), optimal T0 (ideal) -> meets_requirement(0,2) True.
        it, flags = tl.issue_tier_from_labels(["tier/T2-required", "tier/T0-optimal"])
        self.assertTrue(it.has_tier)
        self.assertEqual((it.required, it.optimal), (tl.TIER_T2, tl.TIER_T0))
        self.assertEqual(flags, [])

    def test_contradiction_optimal_weaker_than_required(self):
        # required T0 (frontier floor) but optimal T2 (routine) contradicts it.
        it, flags = tl.issue_tier_from_labels(["tier/T0-required", "tier/T2-optimal"])
        self.assertFalse(it.has_tier)
        self.assertIn(tl.TAG_FLAG_CONTRADICTION, flags)

    def test_missing_roles_are_conservative(self):
        it, flags = tl.issue_tier_from_labels([])
        self.assertFalse(it.has_tier)
        self.assertIn(tl.TAG_FLAG_REQUIRED_MISSING, flags)
        self.assertIn(tl.TAG_FLAG_OPTIMAL_MISSING, flags)

    def test_conflict_two_distinct_required(self):
        it, flags = tl.issue_tier_from_labels(
            ["tier/T0-required", "tier/T1-required", "tier/T0-optimal"])
        self.assertFalse(it.has_tier)
        self.assertIn(tl.TAG_FLAG_REQUIRED_CONFLICT, flags)

    def test_repeated_identical_tier_is_not_conflict(self):
        it, flags = tl.issue_tier_from_labels(
            ["tier/T1-required", "tier/T1-required", "tier/T1-optimal"])
        self.assertTrue(it.has_tier)
        self.assertEqual(flags, [])

    def test_invalid_out_of_range_tier(self):
        it, flags = tl.issue_tier_from_labels(["tier/T3-required", "tier/T3-optimal"])
        self.assertFalse(it.has_tier)
        self.assertIn(tl.TAG_FLAG_REQUIRED_INVALID, flags)

    def test_priority_label_is_not_a_tier(self):
        # "Priority/P1 is not model tier T1": a priority label must never parse.
        it, flags = tl.issue_tier_from_labels(["priority/P1", "priority/P0"])
        self.assertFalse(it.has_tier)
        self.assertIn(tl.TAG_FLAG_REQUIRED_MISSING, flags)


class TestBucketAndProfileForIssue(unittest.TestCase):
    """Mirror of TestLaunchBucketForIssue / TestLaunchProfileForIssue."""

    def test_routine_normal_hard_buckets(self):
        cases = {
            tl.TIER_T2: (tl.BUCKET_ROUTINE, tl.PROFILE_FABLE_XHIGH),
            tl.TIER_T1: (tl.BUCKET_NORMAL, tl.PROFILE_OPUS_XHIGH),
            tl.TIER_T0: (tl.BUCKET_HARD, tl.PROFILE_OPUS_ULTRACODE),
        }
        for tier, (bucket, profile) in cases.items():
            labels = [f"tier/T{tier}-required", f"tier/T{tier}-optimal"]
            prof, buck, ok = tl.launch_profile_for_issue(labels)
            self.assertTrue(ok, labels)
            self.assertEqual(buck, bucket, labels)
            self.assertEqual(prof, profile, labels)

    def test_ultra_label_is_self_sufficient(self):
        # The ultra label uplifts even with NO tier labels co-tagged.
        prof, bucket, ok = tl.launch_profile_for_issue(["tier/ultra"])
        self.assertTrue(ok)
        self.assertEqual(bucket, tl.BUCKET_ULTRA)
        self.assertEqual(prof, tl.PROFILE_FABLE_ULTRACODE)

    def test_ultra_label_beats_tier_labels(self):
        prof, bucket, ok = tl.launch_profile_for_issue(
            ["tier/ultra", "tier/T2-required", "tier/T2-optimal"])
        self.assertTrue(ok)
        self.assertEqual(bucket, tl.BUCKET_ULTRA)

    def test_untagged_issue_has_no_profile(self):
        prof, bucket, ok = tl.launch_profile_for_issue([])
        self.assertFalse(ok)
        self.assertIsNone(prof)

    def test_partial_override_table_fills_gap_from_default(self):
        # A table defining only routine still resolves a hard issue via the built-in.
        prof, bucket, ok = tl.launch_profile_for_issue(
            ["tier/T0-required", "tier/T0-optimal"],
            table={tl.BUCKET_ROUTINE: tl.PROFILE_FABLE_XHIGH})
        self.assertTrue(ok)
        self.assertEqual(prof, tl.PROFILE_OPUS_ULTRACODE)


class TestBuildWorkerCommandUplift(unittest.TestCase):
    """build_worker_command emits ultracode-xor-effort in the Go emit order."""

    def test_ultracode_emits_settings_and_suppresses_effort(self):
        cmd = ird.build_worker_command(
            "claude", "PROMPT", "claude-fable-5", effort="xhigh", ultracode=True)
        self.assertIn("--settings", cmd)
        self.assertEqual(cmd[cmd.index("--settings") + 1], '{"ultracode":true}')
        self.assertNotIn("--effort", cmd)  # ultracode wins; effort suppressed

    def test_effort_emitted_when_not_ultracode(self):
        cmd = ird.build_worker_command("claude", "PROMPT", "claude-opus-4-8", effort="xhigh")
        self.assertIn("--effort", cmd)
        self.assertEqual(cmd[cmd.index("--effort") + 1], "xhigh")
        self.assertNotIn("--settings", cmd)

    def test_emit_order_model_then_knob_then_prompt(self):
        cmd = ird.build_worker_command(
            "claude", "PROMPT", "claude-opus-4-8", ultracode=True)
        self.assertLess(cmd.index("--model"), cmd.index("--settings"))
        self.assertEqual(cmd[-1], "PROMPT")

    def test_default_no_uplift_is_unchanged(self):
        # No effort, no ultracode -> the pre-seam bare claude command.
        cmd = ird.build_worker_command("claude", "PROMPT", None)
        self.assertNotIn("--settings", cmd)
        self.assertNotIn("--effort", cmd)
        self.assertEqual(cmd, ["claude", "-p", "--permission-mode", "bypassPermissions", "PROMPT"])


class TestOptInGate(unittest.TestCase):
    """Mirror of dispatchTierLaunchEnabled / dispatchTierLaunchProfile: default OFF,
    fail-closed on non-claude and untagged."""

    def test_enabled_grammar(self):
        self.assertFalse(tl.tier_launch_enabled({}))  # unset
        for off in ("", "0", "off", "false", "no", "disable", "disabled", "OFF", " false "):
            self.assertFalse(tl.tier_launch_enabled({"FLEET_TIER_LAUNCH": off}), off)
        for on in ("1", "on", "true", "yes", "enable"):
            self.assertTrue(tl.tier_launch_enabled({"FLEET_TIER_LAUNCH": on}), on)

    def test_profile_off_by_default(self):
        prof, bucket = tl.tier_launch_profile(
            "claude", ["tier/T2-required", "tier/T2-optimal"], env={})
        self.assertIsNone(prof)

    def test_profile_non_claude_backend_never_uplifts(self):
        prof, bucket = tl.tier_launch_profile(
            "opencode", ["tier/T2-required", "tier/T2-optimal"],
            env={"FLEET_TIER_LAUNCH": "1"})
        self.assertIsNone(prof)

    def test_profile_untagged_stays_seat_default(self):
        prof, bucket = tl.tier_launch_profile("claude", [], env={"FLEET_TIER_LAUNCH": "1"})
        self.assertIsNone(prof)

    def test_profile_on_and_tagged_resolves(self):
        prof, bucket = tl.tier_launch_profile(
            "claude", ["tier/T0-required", "tier/T0-optimal"],
            env={"FLEET_TIER_LAUNCH": "1"})
        self.assertEqual(prof, tl.PROFILE_OPUS_ULTRACODE)
        self.assertEqual(bucket, tl.BUCKET_HARD)


class TestGoSourceDriftGuard(unittest.TestCase):
    """Read launchprofile.go and assert the mirrored literals are still present, so a
    Go constant change fails HERE instead of silently diverging the Python launcher."""

    def test_go_constants_present(self):
        if not GO_LAUNCHPROFILE.exists():
            self.skipTest(f"Go source not present: {GO_LAUNCHPROFILE}")
        go = GO_LAUNCHPROFILE.read_text(encoding="utf-8")
        for literal in (
            f'"{tl.WORKER_MODEL_OPUS}"',
            f'"{tl.WORKER_MODEL_FABLE}"',
            f'"{tl.EFFORT_XHIGH}"',
            f'"{tl.ULTRA_LABEL}"',
            f'BucketRoutine LaunchBucket = "{tl.BUCKET_ROUTINE}"',
            f'BucketNormal  LaunchBucket = "{tl.BUCKET_NORMAL}"',
            f'BucketHard    LaunchBucket = "{tl.BUCKET_HARD}"',
            f'BucketUltra   LaunchBucket = "{tl.BUCKET_ULTRA}"',
        ):
            self.assertIn(literal, go, f"drift: {literal!r} not found in launchprofile.go")

    def test_ultracode_settings_arg_matches_go(self):
        if not GO_LAUNCHPROFILE.exists():
            self.skipTest(f"Go source not present: {GO_LAUNCHPROFILE}")
        go = GO_LAUNCHPROFILE.read_text(encoding="utf-8")
        # The Go literal is a raw string: UltracodeSettingsArg = `{"ultracode":true}`
        self.assertIn("UltracodeSettingsArg = `" + tl.ULTRACODE_SETTINGS_ARG + "`", go)


if __name__ == "__main__":
    unittest.main()
