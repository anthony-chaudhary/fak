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
