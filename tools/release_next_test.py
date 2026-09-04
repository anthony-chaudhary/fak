#!/usr/bin/env python3
"""Unit tests for tools/release_next.py."""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import release_next


def git(cwd: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    return res.stdout.strip()


class ReleaseNextTest(unittest.TestCase):
    def test_semver_tuple_and_next_version(self) -> None:
        self.assertEqual(release_next.semver_tuple("0.46.0"), (0, 46, 0))
        self.assertEqual(release_next.semver_tuple("v1.2.3"), (1, 2, 3))
        self.assertIsNone(release_next.semver_tuple("not-a-version"))

        base = (0, 46, 0)
        self.assertEqual(release_next.next_version_from_base(base, "patch"), "0.46.1")
        self.assertEqual(release_next.next_version_from_base(base, "minor"), "0.47.0")
        self.assertEqual(release_next.next_version_from_base(base, "major"), "1.0.0")

    def test_classify_commit(self) -> None:
        self.assertEqual(release_next.classify_commit("feat(cmd): add something"), "feat")
        self.assertEqual(release_next.classify_commit("fix(engine): resolve crash"), "fix")
        self.assertEqual(release_next.classify_commit("docs: update readme"), "other")
        self.assertEqual(release_next.classify_commit("chore(build): update deps"), "other")
        self.assertEqual(release_next.classify_commit("feat!: breaking change"), "feat")
        self.assertEqual(
            release_next.classify_commit("fix: change api", body="BREAKING CHANGE: something changed"),
            "feat",
        )

    def test_clean_public_subject(self) -> None:
        subj1 = "feat(cmd): wire workspace tools into fak chat #11126 (fak cmd)"
        self.assertEqual(
            release_next.clean_public_subject(subj1),
            "Wire workspace tools into fak chat.",
        )
        subj2 = "fix(compute): convert CUDA panics into errors (fak compute)"
        self.assertEqual(
            release_next.clean_public_subject(subj2),
            "Convert CUDA panics into errors.",
        )

    def test_parse_existing_draft(self) -> None:
        sample = """# fak vNext (targeting v0.47.0): Work in Progress

## What changed

- Existing feature 1.
- Existing feature 2.

## Reliability and correctness

- Existing bugfix 1.

## Upgrade and breaking changes

- Note about breaking config.
"""
        parsed = release_next.parse_existing_draft(sample)
        self.assertIn("Existing feature 1.", parsed["What changed"])
        self.assertIn("Existing feature 2.", parsed["What changed"])
        self.assertIn("Existing bugfix 1.", parsed["Reliability and correctness"])
        self.assertIn("Note about breaking config.", parsed["Upgrade and breaking changes"])

    def test_render_next_draft(self) -> None:
        state = {
            "projected_version": "0.47.0",
            "projected_level": "minor",
            "base_tag": "v0.46.0",
            "commits": [
                {"subject": "feat(cmd): add chat feature", "body": ""},
                {"subject": "fix(core): fix deadlock", "body": ""},
            ],
        }
        rendered = release_next.render_next_draft(state)
        self.assertIn("# fak vNext (targeting v0.47.0): Work in Progress", rendered)
        self.assertIn("- Add chat feature.", rendered)
        self.assertIn("- Fix deadlock.", rendered)
        self.assertIn("## Upgrade and breaking changes", rendered)

    def test_inspect_and_sync_flow(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            git(repo, "init", "-b", "main")
            git(repo, "config", "user.name", "Test Agent")
            git(repo, "config", "user.email", "test@example.com")

            # Commit 1
            f1 = repo / "f1.txt"
            f1.write_text("v0.1.0", encoding="utf-8")
            git(repo, "add", "f1.txt")
            git(repo, "commit", "-m", "v0.1.0: initial release")
            git(repo, "tag", "v0.1.0")

            # Commit 2: feat
            f2 = repo / "f2.txt"
            f2.write_text("feature", encoding="utf-8")
            git(repo, "add", "f2.txt")
            git(repo, "commit", "-m", "feat(gateway): add proxy support (fak gateway)")

            # Commit 3: fix
            f3 = repo / "f3.txt"
            f3.write_text("fix", encoding="utf-8")
            git(repo, "add", "f3.txt")
            git(repo, "commit", "-m", "fix(vdso): fix race condition (fak vdso)")

            draft_rel = "docs/releases/NEXT.md"

            # Inspect: draft does not exist yet
            st1 = release_next.inspect_next_state(repo, draft_rel=draft_rel)
            self.assertEqual(st1["base_tag"], "v0.1.0")
            self.assertEqual(st1["projected_level"], "minor")
            self.assertEqual(st1["projected_version"], "0.2.0")
            self.assertEqual(st1["commits_count"], 2)
            self.assertFalse(st1["draft_exists"])
            self.assertFalse(st1["in_sync"])
            self.assertEqual(st1["untracked_count"], 2)

            # Sync draft
            st2, changed = release_next.sync_next(repo, draft_rel=draft_rel)
            self.assertTrue(changed)
            self.assertTrue(st2["draft_exists"])
            self.assertTrue(st2["in_sync"])
            self.assertEqual(st2["untracked_count"], 0)
            self.assertEqual(st2["tracked_count"], 2)

            # Verify file contents
            content = (repo / draft_rel).read_text(encoding="utf-8")
            self.assertIn("# fak vNext (targeting v0.2.0): Work in Progress", content)
            self.assertIn("- Add proxy support.", content)
            self.assertIn("- Fix race condition.", content)

            # Re-sync without changes: should report changed=False
            st3, changed3 = release_next.sync_next(repo, draft_rel=draft_rel)
            self.assertFalse(changed3)
            self.assertTrue(st3["in_sync"])

            # Add manual note in draft and add a new commit
            modified_content = content + "\n- Manual operational note.\n"
            (repo / draft_rel).write_text(modified_content, encoding="utf-8")

            f4 = repo / "f4.txt"
            f4.write_text("more fix", encoding="utf-8")
            git(repo, "add", "f4.txt")
            git(repo, "commit", "-m", "fix(engine): handle timeout (fak engine)")

            # Inspect: now 1 commit untracked
            st4 = release_next.inspect_next_state(repo, draft_rel=draft_rel)
            self.assertEqual(st4["commits_count"], 3)
            self.assertEqual(st4["untracked_count"], 1)
            self.assertFalse(st4["in_sync"])

            # Sync again: preserves manual note and adds new commit
            st5, changed5 = release_next.sync_next(repo, draft_rel=draft_rel)
            self.assertTrue(changed5)
            self.assertTrue(st5["in_sync"])
            new_content = (repo / draft_rel).read_text(encoding="utf-8")
            self.assertIn("- Manual operational note.", new_content)
            self.assertIn("- Handle timeout.", new_content)


if __name__ == "__main__":
    unittest.main()
