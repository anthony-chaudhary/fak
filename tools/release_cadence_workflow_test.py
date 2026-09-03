#!/usr/bin/env python3
"""Contract tests for the release cadence workflow."""
from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "release-cadence.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
DRY_RUN = ROOT / "tools" / "release_dry_run.py"


class ReleaseCadenceWorkflowTest(unittest.TestCase):
    def test_workflow_uses_checked_release_helpers(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("schedule:", text)
        self.assertIn("workflow_dispatch:", text)
        self.assertIn("dry_run:", text)
        self.assertIn("group: release-cadence", text)
        self.assertIn("cancel-in-progress: false", text)
        # Trunk is `main`; the cadence gate accepts main (master kept as legacy alias)
        # and checks out whichever trunk ref triggered the run.
        self.assertIn("github.ref_name == 'main'", text)
        self.assertIn("ref: ${{ github.ref_name }}", text)
        self.assertIn("fetch-depth: 0", text)

        self.assertIn("actions/setup-go@v5", text)
        self.assertIn("GOTOOLCHAIN: auto", text)
        self.assertIn("go run ./cmd/fak release status", text)
        self.assertIn("--skip-cut-plan", text)
        self.assertIn("python tools/release_decide.py", text)
        self.assertIn("--require-ci-green", text)
        self.assertIn("python tools/release_cut.py", text)
        self.assertIn("--execute", text)
        self.assertIn("python tools/release_lock.py acquire", text)
        self.assertIn("steps.release_lock.outputs.acquired == 'true'", text)
        self.assertIn("--lock-already-held", text)
        self.assertIn("python tools/release_lock.py release", text)
        self.assertIn("this cadence tick defers to the next run", text)
        self.assertIn("python tools/release_tag.py", text)
        self.assertIn("--require-ci", text)
        self.assertIn("--wait-ci", text)
        self.assertIn("--push", text)
        self.assertIn("python tools/fleet_control_pane.py publish --apply --json", text)
        self.assertIn("python tools/release_publish.py", text)
        self.assertIn("--execute", text)

    def test_cadence_cut_shares_the_release_lock(self) -> None:
        # #1391: a SCHEDULED auto-cut and a human `/release` must share the SAME
        # single-writer lock so the unattended tick cannot race VERSION / the tag
        # against a hand-cut. Pin the mutual-exclusion contract beyond mere
        # string-presence — the ORDER, the GATE, the parent-lock handoff, the
        # release-on-every-exit guard, and the defer.
        text = WORKFLOW.read_text(encoding="utf-8")

        acquire = text.index("python tools/release_lock.py acquire")
        cut = text.index("python tools/release_cut.py --execute")
        release = text.index("python tools/release_lock.py release")
        # The lock is taken BEFORE the cut mutates VERSION / the tag, and dropped last.
        self.assertLess(acquire, cut)
        self.assertLess(cut, release)

        # The cut only runs when the lock was actually acquired, and runs under the
        # parent-held lock (--lock-already-held verifies ownership, does not re-acquire).
        self.assertIn("steps.release_lock.outputs.acquired == 'true'", text)
        self.assertIn("--lock-already-held", text)

        # The auto-cut owns the lock under a DISTINCT cadence-bot owner id (per-run),
        # so its ownership is cross-checkable against — and exclusive with — a human
        # owner; that distinct owner is what makes the shared single-writer lock keep
        # the unattended tick out of a hand-cut's way (#1391's "bot session id owner").
        self.assertIn("FAK_RELEASE_OWNER: release-cadence-", text)

        # Release on EVERY exit (success or failure) but only if we acquired — a
        # deferred tick that never took the lock must not drop a peer's lock.
        self.assertIn("always() && steps.release_lock.outputs.acquired == 'true'", text)

        # A lock held by another owner DEFERS this tick (a note + exit 0); it does
        # not error the cadence run.
        defer = text.index("this cadence tick defers to the next run")
        self.assertLess(acquire, defer)
        self.assertIn("exit 0", text[defer:])

    def test_publish_staleness_escalates_under_killed_cadence(self) -> None:
        # #4025: a very_stale @latest under a killed auto-cut (FAK_AUTO_RELEASE=0) must
        # escalate BEYOND the informational ::warning:: — by default via the SHARED,
        # deduped gate-signal feeder, or a hard fail when the repo opts in. Pin the whole
        # wiring so the escalation cannot be silently dropped back to warning-only.
        text = WORKFLOW.read_text(encoding="utf-8")
        # The decide/shape helper is invoked against the staleness envelope + kill switch.
        self.assertIn("python tools/release_stale_escalate.py", text)
        self.assertIn("--from staleness.json", text)
        self.assertIn("--auto-cut-disabled", text)
        self.assertIn("--emit-envelope", text)
        # The kill switch and the fail opt-in are both read from repo variables.
        self.assertIn("FAK_AUTO_RELEASE", text)
        self.assertIn("FAK_STALE_ESCALATION_FAIL", text)
        # Escalation REUSES the deduped gate-signal feeder (not a second notification feed).
        self.assertIn("python tools/gate_signal.py --from escalate-envelope.json --live", text)
        # Filing an issue requires the job to hold issues: write.
        self.assertIn("issues: write", text)
        # Scoped to SCHEDULED ticks — a manual dispatch is operator-driven, never auto-files.
        escalate = text.index("release_stale_escalate.py")
        guard = text.rindex("github.event_name == 'schedule'", 0, escalate)
        self.assertLess(guard, escalate, "the escalate step must be gated to schedule events")

    def test_workflow_does_not_bypass_tag_helper(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("git push --tags", text)
        self.assertNotIn("git tag -a", text)
        self.assertNotIn("git push origin HEAD:master", text)
        self.assertNotIn("gh release create", text)

    def test_release_substrate_runs_this_contract(self) -> None:
        ci_text = CI_WORKFLOW.read_text(encoding="utf-8")
        dry_run_text = DRY_RUN.read_text(encoding="utf-8")
        self.assertIn("python tools/release_cadence_workflow_test.py", ci_text)
        self.assertIn("python tools/release_publish_test.py", ci_text)
        self.assertIn('"tools/release_cadence_workflow_test.py"', dry_run_text)
        self.assertIn('"tools/release_publish_test.py"', dry_run_text)


    def test_release_persists_commit_bound_lease_receipt(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn('--receipt-commit "$release_commit"', text)
        self.assertIn('--receipt-path release-lease-receipt.json', text)
        self.assertIn('actions/upload-artifact@v4', text)
        self.assertIn('name: release-lease-receipt-${{ github.run_id }}-${{ github.run_attempt }}', text)


    def test_release_owner_is_declared_once_for_the_whole_job(self) -> None:
        self.assertEqual(WORKFLOW.read_text(encoding="utf-8").count("FAK_RELEASE_OWNER:"), 1)
        self.assertIn("FAK_RELEASE_OWNER: release-cadence-${{ github.run_id }}-${{ github.run_attempt }}", WORKFLOW.read_text(encoding="utf-8"))
        self.assertNotIn("steps.release_lock.outputs.owner", WORKFLOW.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
