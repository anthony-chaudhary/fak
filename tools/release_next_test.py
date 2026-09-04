#!/usr/bin/env python3
"""Tests for tools/release_next.py."""
from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "release_next.py"


def load():
    spec = importlib.util.spec_from_file_location("release_next", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ReleaseNextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="release_next_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.mod = load()

    def test_semver_tuple_and_next_version(self) -> None:
        self.assertEqual(self.mod.semver_tuple("v0.48.0"), (0, 48, 0))
        self.assertEqual(self.mod.semver_tuple("0.48.1"), (0, 48, 1))
        self.assertIsNone(self.mod.semver_tuple("invalid"))

        base = (0, 48, 0)
        self.assertEqual(self.mod.next_version_from_base(base, "major"), "1.0.0")
        self.assertEqual(self.mod.next_version_from_base(base, "minor"), "0.49.0")
        self.assertEqual(self.mod.next_version_from_base(base, "patch"), "0.48.1")

    def test_clean_public_subject(self) -> None:
        raw = "feat(release): restore living release draft tracking #11226 (fak release)"
        cleaned = self.mod.clean_public_subject(raw)
        self.assertEqual(cleaned, "Restore living release draft tracking.")

        raw2 = "fix: handle edge case in parser."
        self.assertEqual(self.mod.clean_public_subject(raw2), "Handle edge case in parser.")

        raw3 = ""
        self.assertEqual(self.mod.clean_public_subject(raw3), "")

    def test_classify_commit(self) -> None:
        self.assertEqual(self.mod.classify_commit("feat(core): new capability"), "feat")
        self.assertEqual(self.mod.classify_commit("fix(core): bug fix"), "fix")
        self.assertEqual(self.mod.classify_commit("chore(deps): bump version"), "other")
        self.assertEqual(self.mod.classify_commit("refactor!: change api"), "feat")
        self.assertEqual(self.mod.classify_commit("docs: note", body="BREAKING CHANGE: api removed"), "feat")

    def test_compute_projected_level(self) -> None:
        commits_patch = [{"subject": "fix(core): fix crash (fak core)", "body": ""}]
        self.assertEqual(self.mod.compute_projected_level(commits_patch), "patch")

        commits_minor = [
            {"subject": "fix(core): fix crash (fak core)", "body": ""},
            {"subject": "feat(core): add feature (fak core)", "body": ""},
        ]
        self.assertEqual(self.mod.compute_projected_level(commits_minor), "minor")

        commits_major = [
            {"subject": "fix(core): fix crash (fak core)", "body": ""},
            {"subject": "feat(core)!: breaking change (fak core)", "body": ""},
        ]
        self.assertEqual(self.mod.compute_projected_level(commits_major), "major")

    def test_parse_and_render_draft(self) -> None:
        sample_md = (
            "# fak vNext (targeting v0.49.0): Work in Progress\n\n"
            "## What changed\n\n"
            "- First user feature.\n\n"
            "## Reliability and correctness\n\n"
            "- Crucial bug fix.\n\n"
            "## Engineering quality and evidence\n\n"
            "- Internal refactoring.\n\n"
            "## Upgrade and breaking changes\n\n"
            "- Migration required: run db upgrade.\n"
        )
        sections = self.mod.parse_existing_draft(sample_md)
        self.assertIn("First user feature.", sections["What changed"])
        self.assertIn("Crucial bug fix.", sections["Reliability and correctness"])
        self.assertIn("Migration required: run db upgrade.", sections["Upgrade and breaking changes"])

        state = {
            "projected_version": "0.49.0",
            "projected_level": "minor",
            "base_tag": "v0.48.0",
            "commits": [
                {"subject": "feat(api): second user feature (fak api)", "body": ""},
            ],
        }
        rendered = self.mod.render_next_draft(state, existing_sections=sections)
        self.assertIn("Second user feature.", rendered)
        self.assertIn("First user feature.", rendered)
        self.assertIn("Crucial bug fix.", rendered)
        self.assertIn("Migration required: run db upgrade.", rendered)

    def test_sync_next_and_inspect_state(self) -> None:
        draft_rel = "docs/releases/NEXT.md"
        draft_file = self.tmp / draft_rel
        draft_file.parent.mkdir(parents=True, exist_ok=True)
        draft_file.write_text("# initial\n", encoding="utf-8")

        state = self.mod.inspect_next_state(self.tmp, draft_rel=draft_rel)
        self.assertEqual(state["schema"], "fak-release-next/1")
        self.assertTrue(state["draft_exists"])

        new_state, changed = self.mod.sync_next(self.tmp, draft_rel=draft_rel)
        self.assertTrue(new_state["written"])
        self.assertTrue(draft_file.is_file())
        content = draft_file.read_text(encoding="utf-8")
        self.assertIn("Work in Progress", content)


if __name__ == "__main__":
    unittest.main()
