#!/usr/bin/env python3
"""Hermetic tests for tools/release_publish.py."""
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
SCRIPT = ROOT / "tools" / "release_publish.py"


def load():
    spec = importlib.util.spec_from_file_location("release_publish", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git(root: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class ReleasePublishTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="release_publish_"))
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
        self.repo_index = 0
        self.addCleanup(self._restore_env)

    def _restore_env(self) -> None:
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _repo(self, *, tag: bool = True, note: bool = True) -> Path:
        self.repo_index += 1
        root = self.tmp / f"repo-{self.repo_index}"
        root.mkdir()
        git(root, "init", "-b", "master")
        write(root / "VERSION", "0.1.0\n")
        if note:
            write(root / "docs" / "releases" / "v0.1.0.md", "# v0.1.0\n\nRelease notes.\n")
            git(root, "add", "VERSION", "docs/releases/v0.1.0.md")
        else:
            git(root, "add", "VERSION")
        git(root, "commit", "-m", "v0.1.0: seed")
        if tag:
            git(root, "tag", "-a", "v0.1.0", "-m", "v0.1.0")
        return root

    def test_context_requires_local_tag_and_release_note_at_tag(self) -> None:
        rp = load()
        no_tag = self._repo(tag=False)
        missing_tag = rp.build_context(no_tag, version="0.1.0", skip_gh=True)
        self.assertFalse(missing_tag["ok"])
        self.assertIn("missing_local_tag", missing_tag["errors"])

        no_note = self._repo(tag=True, note=False)
        missing_note = rp.build_context(no_note, version="0.1.0", skip_gh=True)
        self.assertFalse(missing_note["ok"])
        self.assertIn("missing_release_note_at_tag:docs/releases/v0.1.0.md", missing_note["errors"])

    def test_dry_run_plans_create_when_release_is_missing(self) -> None:
        rp = load()
        root = self._repo()

        def runner(cmd: list[str]) -> tuple[int, str]:
            self.assertEqual(cmd[:3], ["gh", "release", "view"])
            return 1, "release not found"

        context = rp.build_context(root, version="0.1.0", runner=runner)

        self.assertTrue(context["ok"], context["errors"])
        self.assertTrue(context["planned_create"])
        self.assertEqual(context["github_release"]["status"], "missing")

    def test_execute_creates_release_from_committed_note(self) -> None:
        rp = load()
        root = self._repo()
        calls: list[list[str]] = []

        def runner(cmd: list[str]) -> tuple[int, str]:
            calls.append(cmd)
            if cmd[:3] == ["gh", "release", "view"]:
                return 1, "release not found"
            if cmd[:3] == ["gh", "release", "create"]:
                self.assertIn("--notes-file", cmd)
                notes_path = Path(cmd[cmd.index("--notes-file") + 1])
                self.assertTrue(notes_path.exists())
                self.assertIn("Release notes.", notes_path.read_text(encoding="utf-8"))
                return 0, "https://example.test/release/v0.1.0"
            raise AssertionError(cmd)

        context = rp.build_context(root, version="0.1.0", runner=runner)
        result = rp.publish_release(root, context, execute=True, runner=runner)

        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(result["release_created"])
        create = [cmd for cmd in calls if cmd[:3] == ["gh", "release", "create"]][0]
        self.assertIn("--verify-tag", create)
        self.assertIn("--notes-file", create)
        self.assertNotIn("--notes", create)

    def test_existing_release_is_idempotent(self) -> None:
        rp = load()
        root = self._repo()
        calls: list[list[str]] = []

        def runner(cmd: list[str]) -> tuple[int, str]:
            calls.append(cmd)
            if cmd[:3] == ["gh", "release", "view"]:
                return 0, json.dumps({"tagName": "v0.1.0", "url": "https://example.test/release/v0.1.0"})
            raise AssertionError(cmd)

        context = rp.build_context(root, version="0.1.0", runner=runner)
        result = rp.publish_release(root, context, execute=True, runner=runner)

        self.assertTrue(result["ok"], result["errors"])
        self.assertFalse(result["release_created"])
        self.assertIn("release_create_skipped_existing", result["idempotent_skips"])
        self.assertFalse(any(cmd[:3] == ["gh", "release", "create"] for cmd in calls))

    def test_live_cli_dry_run_no_mutation(self) -> None:
        before = subprocess.run(
            ["git", "tag", "-l"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        ).stdout
        tags = subprocess.run(
            ["git", "tag", "-l", "v[0-9]*", "--sort=-v:refname"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.splitlines()
        self.assertTrue(tags, "release_publish live dry-run needs at least one existing version tag")
        version = tags[0].removeprefix("v")
        proc = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--version", version,
                "--skip-gh",
                "--json",
            ],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["ok"], payload["errors"])
        after = subprocess.run(
            ["git", "tag", "-l"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        ).stdout
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
