#!/usr/bin/env python3
"""Hermetic tests for tools/release_decide.py."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECIDE = ROOT / "tools" / "release_decide.py"


def load():
    spec = importlib.util.spec_from_file_location("release_decide", DECIDE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def payload(**overrides):
    base = {
        "last_tag": "v0.21.0",
        "latest_any_tag": "v0.21.0",
        "unreachable_newer_tags": [],
        "commits_since_tag": [{"subject": "feat(gateway): add a thing"}],
        "version_files": {"version": "0.21.0", "drift": False},
        "ci_on_head": {"status": "green"},
        "tag_drift": {
            "files_ahead_of_tag": False,
            "source_behind_reachable_tag": False,
            "reason": "no cut due",
        },
        "workflows_parse_ok": {"ok": True, "files": {}},
    }
    base.update(overrides)
    return base


class ReleaseDecideTest(unittest.TestCase):
    def test_subject_classification(self) -> None:
        rd = load()
        self.assertEqual(rd.classify_subject("feat(model): add x"), "minor")
        self.assertEqual(rd.classify_subject("fix(model): repair x"), "patch")
        self.assertEqual(rd.classify_subject("perf(vulkan): speed x"), "patch")
        self.assertEqual(rd.classify_subject("feat!: break x"), "major")
        self.assertEqual(rd.classify_subject("plain subject"), "patch")

    def test_highest_signal_and_themes(self) -> None:
        rd = load()
        level, themes = rd.decide_level([
            {"subject": "fix(model): repair"},
            {"subject": "feat(gateway): add"},
            {"subject": "docs(gateway): explain"},
            {"subject": "v0.21.0: prior release"},
        ])
        self.assertEqual(level, "minor")
        self.assertEqual(themes, ["model", "gateway"])

    def test_decide_releases_from_green_substantive_range(self) -> None:
        rd = load()
        verdict = rd.decide(payload())
        self.assertEqual(verdict["decision"], "release")
        self.assertEqual(verdict["level"], "minor")
        self.assertEqual(verdict["next_version"], "0.22.0")
        self.assertEqual(verdict["blockers"], [])

    def test_churn_only_range_holds_unless_forced(self) -> None:
        rd = load()
        churn = payload(commits_since_tag=[{"subject": "docs: tidy"}])
        self.assertEqual(rd.decide(churn)["decision"], "hold")
        forced = rd.decide(churn, force=True)
        self.assertEqual(forced["decision"], "release")
        self.assertEqual(forced["level"], "patch")

    def test_is_significant_failsafe(self) -> None:
        rd = load()
        # Trivial types never justify a release.
        for subj in ("docs: tidy", "chore(deps): bump", "test: add case",
                     "style: gofmt", "ci: pin runner", "build: retag"):
            self.assertFalse(rd.is_significant(subj), subj)
        # Real, shippable change types are significant.
        for subj in ("feat(x): add", "fix(x): repair", "perf(x): speed",
                     "refactor(x): reshape", "revert: undo"):
            self.assertTrue(rd.is_significant(subj), subj)
        # Fail-safe: an unrecognized type, a bare subject, and a breaking docs
        # commit all count as significant (in-doubt -> significant).
        self.assertTrue(rd.is_significant("wip: scratch"))
        self.assertTrue(rd.is_significant("just some words"))
        self.assertTrue(rd.is_significant("docs!: drop a public guide"))
        self.assertTrue(rd.is_significant("chore: x", "BREAKING CHANGE: gone"))

    def test_docs_chore_only_window_held_below_floor(self) -> None:
        rd = load()
        # min_substantive=0 disables the older substantive gate, so ONLY the
        # significance floor can hold this all-trivial window (issue #1389).
        churn = payload(commits_since_tag=[
            {"subject": "docs: tidy readme"},
            {"subject": "chore(deps): bump x"},
            {"subject": "test: add a case"},
        ])
        verdict = rd.decide(churn, min_substantive=0)
        self.assertEqual(verdict["decision"], "hold")
        self.assertIn("BELOW_FLOOR", verdict["blockers"])
        self.assertEqual(verdict["significant"], 0)
        self.assertIn("trivial", verdict["reason"])

    def test_one_fix_in_trivial_window_clears_floor(self) -> None:
        rd = load()
        mixed = payload(commits_since_tag=[
            {"subject": "docs: tidy"},
            {"subject": "chore: bump"},
            {"subject": "fix(gateway): repair a real bug"},
        ])
        verdict = rd.decide(mixed, min_substantive=0)
        self.assertEqual(verdict["decision"], "release")
        self.assertNotIn("BELOW_FLOOR", verdict["blockers"])
        self.assertEqual(verdict["significant"], 1)

    def test_floor_override_forces_a_cut(self) -> None:
        rd = load()
        churn = payload(commits_since_tag=[{"subject": "docs: tidy"}])
        forced = rd.decide(churn, min_substantive=0, force=True)
        self.assertEqual(forced["decision"], "release")
        self.assertNotIn("BELOW_FLOOR", forced["blockers"])
        disabled = rd.decide(churn, min_substantive=0, significance_floor=False)
        self.assertEqual(disabled["decision"], "release")
        self.assertNotIn("BELOW_FLOOR", disabled["blockers"])

    def test_unknown_type_window_failsafe_cuts(self) -> None:
        rd = load()
        # A window of only unrecognized-type commits is NOT provably trivial, so
        # the floor must NOT suppress it (fail-safe).
        unknown = payload(commits_since_tag=[
            {"subject": "wip: scratch work"},
            {"subject": "frobnicate the thing"},
        ])
        verdict = rd.decide(unknown, min_substantive=0)
        self.assertEqual(verdict["decision"], "release")
        self.assertNotIn("BELOW_FLOOR", verdict["blockers"])
        self.assertEqual(verdict["significant"], 2)

    def test_env_knobs_drive_floor_defaults(self) -> None:
        rd = load()
        # FAK_RELEASE_SIGNIFICANCE_FLOOR=0 turns the floor off via main()'s parser.
        import os
        keep = {k: os.environ.get(k) for k in
                ("FAK_RELEASE_SIGNIFICANCE_FLOOR", "FAK_RELEASE_MIN_SUBSTANTIVE")}
        try:
            os.environ["FAK_RELEASE_SIGNIFICANCE_FLOOR"] = "0"
            self.assertFalse(rd._env_flag("FAK_RELEASE_SIGNIFICANCE_FLOOR", True))
            os.environ["FAK_RELEASE_SIGNIFICANCE_FLOOR"] = "1"
            self.assertTrue(rd._env_flag("FAK_RELEASE_SIGNIFICANCE_FLOOR", False))
            os.environ["FAK_RELEASE_MIN_SUBSTANTIVE"] = "3"
            self.assertEqual(rd._env_int("FAK_RELEASE_MIN_SUBSTANTIVE", 1), 3)
            os.environ["FAK_RELEASE_MIN_SUBSTANTIVE"] = "notanint"
            self.assertEqual(rd._env_int("FAK_RELEASE_MIN_SUBSTANTIVE", 1), 1)
        finally:
            for k, v in keep.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_debounce_holds_recent_tag(self) -> None:
        rd = load()
        # A substantive, green window that would otherwise release is HELD when the
        # last tag is younger than the min interval (issue #1389 AC2). 2h < 6h.
        recent = payload(last_tag_age_seconds=2 * 3600)
        verdict = rd.decide(recent, min_interval_hours=6)
        self.assertEqual(verdict["decision"], "hold")
        self.assertIn("TOO_SOON", verdict["blockers"])
        self.assertIn("debounce", verdict["reason"])
        self.assertEqual(verdict["min_interval_hours"], 6)
        self.assertEqual(verdict["last_tag_age_seconds"], 2 * 3600)

    def test_debounce_clears_old_tag(self) -> None:
        rd = load()
        # The same window cuts once the last tag is older than the interval. 8h > 6h.
        old = payload(last_tag_age_seconds=8 * 3600)
        verdict = rd.decide(old, min_interval_hours=6)
        self.assertEqual(verdict["decision"], "release")
        self.assertNotIn("TOO_SOON", verdict["blockers"])

    def test_debounce_off_by_default(self) -> None:
        rd = load()
        # min_interval_hours defaults to 0 (off): a freshly-tagged window still cuts,
        # so manual dispatch (which never arms the env knob) is never debounced.
        recent = payload(last_tag_age_seconds=60)
        verdict = rd.decide(recent)
        self.assertEqual(verdict["decision"], "release")
        self.assertNotIn("TOO_SOON", verdict["blockers"])

    def test_debounce_force_bypasses(self) -> None:
        rd = load()
        recent = payload(last_tag_age_seconds=60)
        verdict = rd.decide(recent, min_interval_hours=6, force=True)
        self.assertEqual(verdict["decision"], "release")
        self.assertNotIn("TOO_SOON", verdict["blockers"])

    def test_debounce_unknown_age_fails_open(self) -> None:
        rd = load()
        # No tag-age evidence (None / missing) must NOT block — fail-open, like the
        # significance floor's in-doubt -> significant rule.
        for age in (None, "garbage", True):
            window = payload(last_tag_age_seconds=age)
            verdict = rd.decide(window, min_interval_hours=6)
            self.assertEqual(verdict["decision"], "release", age)
            self.assertNotIn("TOO_SOON", verdict["blockers"], age)

    def test_debounce_does_not_override_a_real_blocker(self) -> None:
        rd = load()
        # CI red on a recent tag: the real blocker wins the reason; the debounce is
        # only ever the SOLE reason a would-release window holds.
        window = payload(last_tag_age_seconds=60, ci_on_head={"status": "red"})
        verdict = rd.decide(window, min_interval_hours=6)
        self.assertEqual(verdict["decision"], "hold")
        self.assertIn("CI_BASE_RED", verdict["blockers"])
        self.assertNotIn("TOO_SOON", verdict["blockers"])

    def test_debounce_not_added_when_nothing_to_ship(self) -> None:
        rd = load()
        # An empty window holds on NOTHING_TO_SHIP, not the debounce.
        window = payload(commits_since_tag=[], last_tag_age_seconds=60)
        verdict = rd.decide(window, min_interval_hours=6)
        self.assertIn("NOTHING_TO_SHIP", verdict["blockers"])
        self.assertNotIn("TOO_SOON", verdict["blockers"])

    def test_env_min_interval_hours_default(self) -> None:
        rd = load()
        import os
        keep = os.environ.get("FAK_RELEASE_MIN_INTERVAL_HOURS")
        try:
            os.environ["FAK_RELEASE_MIN_INTERVAL_HOURS"] = "6"
            self.assertEqual(rd._env_float("FAK_RELEASE_MIN_INTERVAL_HOURS", 0.0), 6.0)
            os.environ["FAK_RELEASE_MIN_INTERVAL_HOURS"] = "notafloat"
            self.assertEqual(rd._env_float("FAK_RELEASE_MIN_INTERVAL_HOURS", 0.0), 0.0)
            os.environ.pop("FAK_RELEASE_MIN_INTERVAL_HOURS", None)
            self.assertEqual(rd._env_float("FAK_RELEASE_MIN_INTERVAL_HOURS", 0.0), 0.0)
        finally:
            if keep is None:
                os.environ.pop("FAK_RELEASE_MIN_INTERVAL_HOURS", None)
            else:
                os.environ["FAK_RELEASE_MIN_INTERVAL_HOURS"] = keep

    def test_unreachable_newer_tag_is_warning_and_bump_base(self) -> None:
        rd = load()
        verdict = rd.decide(payload(
            latest_any_tag="v0.21.1",
            unreachable_newer_tags=["v0.21.1"],
        ))
        self.assertEqual(verdict["decision"], "release")
        self.assertEqual(verdict["next_version"], "0.22.0")
        self.assertIn("newer semver tag v0.21.1", " ".join(verdict["warnings"]))

    def test_recovery_when_source_is_ahead_of_tags(self) -> None:
        rd = load()
        verdict = rd.decide(payload(
            commits_since_tag=[],
            version_files={"version": "0.22.0", "drift": False},
            tag_drift={
                "files_ahead_of_tag": True,
                "source_behind_reachable_tag": False,
                "reason": "VERSION is ahead",
            },
        ))
        self.assertEqual(verdict["decision"], "release")
        self.assertTrue(verdict["recover"])
        self.assertEqual(verdict["next_version"], "0.22.0")

    def test_workflow_parse_failure_blocks(self) -> None:
        rd = load()
        verdict = rd.decide(payload(workflows_parse_ok={"ok": False, "files": {"x.yml": "bad"}}))
        self.assertEqual(verdict["decision"], "hold")
        self.assertIn("WORKFLOW_UNPARSEABLE", verdict["blockers"])

    def test_ci_base_red_or_none_blocks_and_unknown_can_be_strict(self) -> None:
        rd = load()
        red = rd.decide(payload(ci_on_head={"status": "red"}))
        self.assertEqual(red["decision"], "hold")
        self.assertIn("CI_BASE_RED", red["blockers"])

        none = rd.decide(payload(ci_on_head={"status": "none"}))
        self.assertEqual(none["decision"], "hold")
        self.assertIn("CI_BASE_NONE", none["blockers"])

        soft = rd.decide(payload(ci_on_head={"status": "unknown"}))
        self.assertEqual(soft["decision"], "release")
        strict = rd.decide(payload(ci_on_head={"status": "unknown"}), require_ci_green=True)
        self.assertEqual(strict["decision"], "hold")
        self.assertIn("CI_STATE_UNKNOWN", strict["blockers"])

    def test_retry_to_green_ci_blocks_auto_cut(self) -> None:
        rd = load()
        verdict = rd.decide(payload(ci_on_head={
            "status": "green",
            "latest_trunk_ci": {"conclusion": "success", "attempt": 2},
        }))
        self.assertEqual(verdict["decision"], "hold")
        self.assertIn("CI_RETRY_TO_GREEN", verdict["blockers"])
        self.assertIn("FAK_AUTO_RELEASE=0", verdict["reason"])

    def test_fast_subset_green_clears_ci_while_race_in_progress(self) -> None:
        rd = load()
        # The whole ci.yml run is still in progress (status "none": no decisive
        # completed run yet, because the slow -race job hasn't concluded), but the
        # fast subset (ci-fast.yml) has already concluded green. The cut must clear
        # CI_BASE_RED/NONE on the fast signal and not wait on the -race tail. #1374.
        verdict = rd.decide(payload(
            ci_on_head={"status": "none"},
            ci_fast={"status": "green"},
        ))
        self.assertEqual(verdict["decision"], "release")
        self.assertEqual(verdict["blockers"], [])
        self.assertEqual(verdict["ci_source"], "fast")

    def test_fast_subset_red_still_holds(self) -> None:
        rd = load()
        # A red fast subset holds the cut even if the whole ci.yml hasn't
        # concluded — the fast subset caught a real regression. #1374.
        verdict = rd.decide(payload(
            ci_on_head={"status": "none"},
            ci_fast={"status": "red"},
        ))
        self.assertEqual(verdict["decision"], "hold")
        self.assertIn("CI_BASE_RED", verdict["blockers"])
        self.assertEqual(verdict["ci_source"], "fast")
        self.assertIn("ci-fast.yml", verdict["reason"])

    def test_fail_safe_falls_back_to_whole_ci_when_fast_missing(self) -> None:
        rd = load()
        # No ci_fast field at all: fall back to the whole ci.yml signal. A red
        # whole-ci base still holds — a missing fast signal never cuts blind.
        missing = rd.decide(payload(ci_on_head={"status": "red"}))
        self.assertEqual(missing["decision"], "hold")
        self.assertIn("CI_BASE_RED", missing["blockers"])
        self.assertEqual(missing["ci_source"], "whole")

        # An indecisive (unknown/none) fast signal is NOT trusted: fall back to
        # the whole ci.yml signal rather than treat the fast run as a verdict.
        for fast in ({"status": "unknown"}, {"status": "none"}, {}, None, "garbage"):
            v = rd.decide(payload(ci_on_head={"status": "red"}, ci_fast=fast))
            self.assertEqual(v["ci_source"], "whole", fast)
            self.assertIn("CI_BASE_RED", v["blockers"], fast)

        # And when the fall-back whole-ci base is green, the cut clears.
        green = rd.decide(payload(ci_on_head={"status": "green"}))
        self.assertEqual(green["decision"], "release")
        self.assertEqual(green["ci_source"], "whole")

    def test_retry_to_green_does_not_apply_to_fast_subset(self) -> None:
        rd = load()
        # CI_RETRY_TO_GREEN is a whole-ci concern (flaky -race / heavy suite). A
        # decisive green fast subset clears without that guard, regardless of any
        # attempt count carried on the whole-ci payload.
        verdict = rd.decide(payload(
            ci_fast={"status": "green"},
            ci_on_head={
                "status": "green",
                "latest_trunk_ci": {"conclusion": "success", "attempt": 3},
            },
        ))
        self.assertEqual(verdict["decision"], "release")
        self.assertNotIn("CI_RETRY_TO_GREEN", verdict["blockers"])
        self.assertEqual(verdict["ci_source"], "fast")

    def test_cli_contract_on_live_repo(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(DECIDE), "--json"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        self.assertIn(proc.returncode, (0, 2), proc.stderr)
        verdict = json.loads(proc.stdout)
        self.assertIn(verdict["decision"], ("release", "hold"))
        self.assertEqual(proc.returncode == 0, verdict["decision"] == "release")


if __name__ == "__main__":
    unittest.main(verbosity=2)
