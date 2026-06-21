#!/usr/bin/env python3
"""Hermetic tests for tools/stable_release_context.py."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "stable_release_context.py"


def load():
    spec = importlib.util.spec_from_file_location("stable_release_context", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class StableReleaseContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="stable_ctx_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
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
        self.addCleanup(self._restore_env)

    def _restore_env(self) -> None:
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
        return root

    def test_latest_semver_tag_ignores_unreachable_tags(self) -> None:
        sr = load()
        root = self._repo()
        git(root, "checkout", "-b", "side")
        write(root / "VERSION", "0.2.0\n")
        git(root, "commit", "-am", "side")
        git(root, "tag", "-a", "v0.2.0", "-m", "v0.2.0")
        git(root, "checkout", "master")

        self.assertEqual(sr.latest_semver_tag(root, reachable_only=True), "v0.1.0")
        self.assertEqual(sr.latest_semver_tag(root, reachable_only=False), "v0.2.0")

    def test_idempotency_distinguishes_same_and_different_codename(self) -> None:
        sr = load()
        root = self._repo()
        sha = sr.tag_sha(root, "v0.1.0")
        git(root, "tag", "-a", "stable/2026-06-bedrock", "-m", "stable", sha)
        write(root / "docs" / "stable-releases" / "2026-06-bedrock.md", "evidence\n")

        idem = sr.idempotency(root, "2026-06-bedrock", sha)
        self.assertTrue(idem["tag_exists"])
        self.assertTrue(idem["tag_matches_candidate"])
        self.assertTrue(idem["evidence_file_exists"])

        other = sr.idempotency(root, "2026-06-aster", sha)
        self.assertFalse(other["tag_exists"])
        self.assertEqual(sr.stable_collision(root, "2026-06-aster", sha), "stable/2026-06-bedrock")

    def test_tag_age_gate(self) -> None:
        sr = load()
        root = self._repo()
        created = sr.tag_epoch(root, "v0.1.0")
        self.assertIsInstance(created, int)
        self.assertTrue(sr.gate_tag_age(root, "v0.1.0", 1, now=created + 2 * 86400)["pass"])
        self.assertFalse(sr.gate_tag_age(root, "v0.1.0", 3, now=created + 2 * 86400)["pass"])

    def test_ci_gate_classifies_billing_pre_run_failure(self) -> None:
        sr = load()

        def fake_run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 600) -> tuple[int, str]:
            if cmd[:3] == ["gh", "run", "list"]:
                return 0, json.dumps([{
                    "databaseId": "run-1",
                    "status": "completed",
                    "conclusion": "failure",
                    "url": "https://example.test/run",
                }])
            if cmd == ["gh", "run", "view", "run-1", "--json", "jobs"]:
                return 0, json.dumps({
                    "jobs": [{
                        "databaseId": "job-1",
                        "name": "build",
                        "status": "completed",
                        "conclusion": "failure",
                        "steps": [],
                    }],
                })
            if cmd == ["gh", "api", "repos/:owner/:repo/check-runs/job-1/annotations"]:
                return 0, json.dumps([{
                    "annotation_level": "failure",
                    "path": ".github",
                    "message": (
                        "The job was not started because recent account payments have failed "
                        "or your spending limit needs to be increased."
                    ),
                }])
            raise AssertionError(cmd)

        sr.run = fake_run

        row = sr.gate_ci_green("abc123", skip=False)

        self.assertFalse(row["pass"])
        self.assertEqual(row["verdict"], "CI_BILLING")
        self.assertEqual(row["action"], "fix_ci_billing")
        self.assertEqual(row["diagnosis"]["jobs"][0]["steps"], 0)

    def test_live_cli_smoke_no_mutation(self) -> None:
        codename = "2026-06-context-smoke"
        evidence = ROOT / "docs" / "stable-releases" / f"{codename}.md"
        evidence_before = evidence.exists()
        tag_before = subprocess.run(
            ["git", "tag", "-l", f"stable/{codename}"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        ).stdout
        proc = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--codename", codename,
                "--window-days", "0",
                "--skip-tests",
                "--skip-dos",
                "--skip-ci",
                "--json",
            ],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        self.assertIn(proc.returncode, (0, 1), proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["codename"], codename)
        self.assertIn("gate", payload)
        tag_after = subprocess.run(
            ["git", "tag", "-l", f"stable/{codename}"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        ).stdout
        self.assertEqual(evidence.exists(), evidence_before)
        self.assertEqual(tag_after, tag_before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
