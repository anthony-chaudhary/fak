#!/usr/bin/env python3
"""Hermetic tests for tools/release_context.py."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "release_context.py"


def load():
    spec = importlib.util.spec_from_file_location("release_context", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class ReleaseContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="release_ctx_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.old_cwd = Path.cwd()
        self.old_env = {k: os.environ.get(k) for k in (
            "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_NOSYSTEM",
            "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL",
        )}
        os.environ.update({
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
        })
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        os.chdir(self.old_cwd)
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _repo(self) -> Path:
        root = self.tmp / "repo"
        root.mkdir()
        git(root, "init", "-b", "master")
        write(root / "VERSION", "0.1.0\n")
        git(root, "add", "VERSION")
        git(root, "commit", "-m", "seed")
        git(root, "tag", "-a", "v0.1.0", "-m", "v0.1.0")
        os.chdir(root)
        return root

    def test_latest_tag_defaults_to_reachable_semver_tag(self) -> None:
        rc = load()
        root = self._repo()
        git(root, "checkout", "-b", "side")
        write(root / "VERSION", "0.2.0\n")
        git(root, "commit", "-am", "side")
        git(root, "tag", "-a", "v0.2.0", "-m", "v0.2.0")
        git(root, "checkout", "master")

        self.assertEqual(rc.latest_tag(), "v0.1.0")
        self.assertEqual(rc.latest_tag(reachable_only=False), "v0.2.0")
        self.assertEqual(rc.unreachable_newer_tags("v0.1.0"), ["v0.2.0"])

    def test_tag_drift_marks_side_branch_tag_as_non_blocking_warning_state(self) -> None:
        rc = load()
        drift = rc.tag_drift(
            "v0.1.0",
            "v0.2.0",
            {"version": "0.1.0", "drift": False},
            [],
        )
        self.assertFalse(drift["cut_due"])
        self.assertTrue(drift["source_behind_latest_tag"])
        self.assertFalse(drift["source_behind_reachable_tag"])
        self.assertIn("outside HEAD", drift["reason"])

    def test_commits_since_preserves_generation_sidecar(self) -> None:
        rc = load()
        root = self._repo()
        write(root / "x.txt", "x\n")
        git(root, "add", "x.txt")
        git(root, "commit", "-m", "feat(tools): add x #1634 (fak tools)", "-m", "Generation: next")

        commits = rc.commits_since("v0.1.0", 10)

        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0]["generation"], "gen/next")

    def test_fold_latest_trunk_ci_skips_indecisive_runs(self) -> None:
        rc = load()
        status, latest, note = rc.fold_latest_trunk_ci([
            {"conclusion": "cancelled", "headSha": "1111111"},
            {"conclusion": "success", "headSha": "abcdef123456", "updatedAt": "2026-06-19T00:00:00Z"},
        ])
        self.assertEqual(status, "green")
        self.assertIsNone(note)
        self.assertEqual(latest["head_sha"], "abcdef1")
        self.assertIsNone(latest["attempt"])
        self.assertEqual(latest["indecisive_runs_since"], 1)

    def test_fold_latest_trunk_ci_carries_attempt_metadata(self) -> None:
        rc = load()
        status, latest, note = rc.fold_latest_trunk_ci([
            {
                "conclusion": "success",
                "headSha": "abcdef123456",
                "updatedAt": "2026-06-19T00:00:00Z",
                "attempt": 2,
                "databaseId": 123,
                "url": "https://example.test/runs/123",
            },
        ])
        self.assertEqual(status, "green")
        self.assertIsNone(note)
        self.assertEqual(latest["attempt"], 2)
        self.assertEqual(latest["database_id"], 123)
        self.assertEqual(latest["url"], "https://example.test/runs/123")

    def test_fold_latest_trunk_ci_flags_red_and_unknown(self) -> None:
        rc = load()
        status, latest, note = rc.fold_latest_trunk_ci([
            {"conclusion": "failure", "headSha": "abcdef123456"},
        ])
        self.assertEqual(status, "red")
        self.assertEqual(latest["conclusion"], "failure")
        self.assertIn("not green", note)

        status, latest, note = rc.fold_latest_trunk_ci(None)
        self.assertEqual(status, "unknown")
        self.assertIsNone(latest)
        self.assertIn("gh unavailable", note)


if __name__ == "__main__":
    unittest.main(verbosity=2)
