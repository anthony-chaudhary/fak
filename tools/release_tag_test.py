#!/usr/bin/env python3
"""Hermetic tests for tools/release_tag.py."""
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
SCRIPT = ROOT / "tools" / "release_tag.py"


def load():
    spec = importlib.util.spec_from_file_location("release_tag", SCRIPT)
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


class ReleaseTagTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="release_tag_"))
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

    def _repo(self, version: str = "0.1.0") -> tuple[Path, str]:
        root = self.tmp / "repo"
        root.mkdir()
        git(root, "init", "-b", "master")
        write(root / "VERSION", version + "\n")
        write(root / "docs" / "releases" / f"v{version}.md", f"# v{version}\n")
        git(root, "add", "VERSION", f"docs/releases/v{version}.md")
        git(root, "commit", "-m", f"v{version}: seed release")
        sha = git(root, "rev-parse", "HEAD")
        return root, sha

    def test_fold_ci_runs_classifies_green_red_pending_and_no_signal(self) -> None:
        rt = load()
        self.assertEqual(rt.fold_ci_runs([], require_ci=False)["verdict"], "NO_SIGNAL")
        self.assertFalse(rt.fold_ci_runs([], require_ci=True)["pass"])
        self.assertTrue(rt.fold_ci_runs([{"status": "completed", "conclusion": "success"}], require_ci=True)["pass"])
        red = rt.fold_ci_runs([{"status": "completed", "conclusion": "failure"}], require_ci=False)
        self.assertFalse(red["pass"])
        self.assertEqual(red["verdict"], "RED")
        pending = rt.fold_ci_runs([{"status": "in_progress"}], require_ci=False)
        self.assertTrue(pending["pass"])
        self.assertEqual(pending["verdict"], "PENDING")

    def test_ci_signal_diagnoses_billing_annotation_on_red_run(self) -> None:
        rt = load()

        def runner(cmd: list[str]) -> tuple[int, str]:
            if cmd[:5] == ["gh", "run", "list", "--workflow", "ci.yml"]:
                return 0, json.dumps([{
                    "databaseId": "run-1",
                    "status": "completed",
                    "conclusion": "failure",
                    "url": "https://example.invalid/run",
                }])
            if cmd == ["gh", "run", "view", "run-1", "--json", "jobs"]:
                return 0, json.dumps({
                    "jobs": [{
                        "databaseId": "job-1",
                        "name": "build",
                        "status": "completed",
                        "conclusion": "failure",
                        "steps": [],
                    }]
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

        folded = rt.ci_signal(Path("repo"), "abc123", workflow="ci.yml", require_ci=True, runner=runner)

        self.assertFalse(folded["pass"])
        self.assertEqual(folded["verdict"], "CI_BILLING")
        self.assertEqual(folded["diagnosis"]["action"], "fix_ci_billing")
        self.assertEqual(folded["diagnosis"]["jobs"][0]["steps"], 0)

    def test_context_checks_committed_version_and_release_note(self) -> None:
        rt = load()
        root, sha = self._repo()
        context = rt.build_context(
            root,
            version="0.1.0",
            ref=sha,
            skip_dry_run=True,
            skip_ci=True,
            skip_remote_tag=True,
        )
        self.assertTrue(context["ok"], context["errors"])
        self.assertTrue(context["checks"]["version_at_ref"]["pass"])
        self.assertTrue(context["checks"]["release_note_at_ref"]["pass"])
        self.assertIn("release_dry_run_skipped", context["idempotent_skips"])

    def test_context_blocks_version_mismatch_and_missing_release_note(self) -> None:
        rt = load()
        root, sha = self._repo()
        context = rt.build_context(
            root,
            version="0.2.0",
            ref=sha,
            skip_dry_run=True,
            skip_ci=True,
            skip_remote_tag=True,
        )
        self.assertFalse(context["ok"])
        self.assertIn("version_mismatch_at_ref:0.1.0", context["errors"])
        self.assertIn("missing_release_note:docs/releases/v0.2.0.md", context["errors"])

    def test_execute_creates_annotated_tag_and_is_idempotent(self) -> None:
        rt = load()
        root, sha = self._repo()
        context = rt.build_context(
            root,
            version="0.1.0",
            ref=sha,
            skip_dry_run=True,
            skip_ci=True,
            skip_remote_tag=True,
        )
        first = rt.execute_tag(root, context)
        self.assertTrue(first["ok"], first["errors"])
        self.assertTrue(first["tag_created"])
        self.assertEqual(git(root, "rev-list", "-n1", "v0.1.0"), sha)
        tag_body = git(root, "tag", "-l", "v0.1.0", "--format=%(contents)")
        self.assertIn("release-dry-run: skipped", tag_body)

        second_context = rt.build_context(
            root,
            version="0.1.0",
            ref=sha,
            skip_dry_run=True,
            skip_ci=True,
            skip_remote_tag=True,
        )
        second = rt.execute_tag(root, second_context)
        self.assertTrue(second["ok"], second["errors"])
        self.assertFalse(second["tag_created"])
        self.assertIn("tag_create_skipped_existing_same_sha", second["idempotent_skips"])

    def test_execute_uses_release_lock_when_present(self) -> None:
        rt = load()
        root, sha = self._repo()
        self._write_lock_helper(root)
        context = rt.build_context(
            root,
            version="0.1.0",
            ref=sha,
            skip_dry_run=True,
            skip_ci=True,
            skip_remote_tag=True,
        )
        result = rt.execute_tag(root, context)
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(result["release_lock"]["held"])
        self.assertTrue(result["release_lock_release"]["ok"])
        self.assertEqual(
            (root / "lock.log").read_text(encoding="utf-8").splitlines(),
            ["acquire:fallback-owner", "release:fallback-owner"],
        )

    def _write_lock_helper(self, root: Path) -> None:
        write(root / "tools" / "release_lock.py", (
            "import json, pathlib, sys\n"
            "cmd = sys.argv[1]\n"
            "owner = sys.argv[sys.argv.index('--owner') + 1] if '--owner' in sys.argv else 'fallback-owner'\n"
            "p = pathlib.Path('lock.log')\n"
            "old = p.read_text(encoding='utf-8') if p.exists() else ''\n"
            "p.write_text(old + cmd + ':' + owner + '\\n', encoding='utf-8')\n"
            "payload = {'ok': True, 'cmd': cmd}\n"
            "if cmd == 'acquire':\n"
            "    payload['lock'] = {'owner': owner}\n"
            "print(json.dumps(payload))\n"
        ))

    def test_tag_collision_blocks_different_commit(self) -> None:
        rt = load()
        root, first_sha = self._repo()
        git(root, "tag", "-a", "v0.1.0", first_sha, "-m", "v0.1.0")
        write(root / "docs" / "note.md", "change\n")
        git(root, "add", "docs/note.md")
        git(root, "commit", "-m", "docs: later")
        context = rt.build_context(
            root,
            version="0.1.0",
            ref="HEAD",
            skip_dry_run=True,
            skip_ci=True,
            skip_remote_tag=True,
        )
        self.assertFalse(context["ok"])
        self.assertIn("local_tag_exists_on_different_commit", context["errors"])

    def test_live_cli_dry_run_no_mutation(self) -> None:
        before = subprocess.run(
            ["git", "tag", "-l"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        ).stdout
        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        proc = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--version", version,
                "--ref", "HEAD",
                "--skip-dry-run",
                "--skip-ci",
                "--skip-remote-tag",
                "--json",
            ],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        self.assertIn(proc.returncode, (0, 1), proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertIn("ok", payload)
        after = subprocess.run(
            ["git", "tag", "-l"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        ).stdout
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
