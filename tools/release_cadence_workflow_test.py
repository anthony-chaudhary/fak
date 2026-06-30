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
        self.assertIn("python tools/release_tag.py", text)
        self.assertIn("--require-ci", text)
        self.assertIn("--wait-ci", text)
        self.assertIn("--push", text)
        self.assertIn("python tools/fleet_control_pane.py publish --apply --json", text)
        self.assertIn("python tools/release_publish.py", text)
        self.assertIn("--execute", text)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
