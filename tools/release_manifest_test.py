#!/usr/bin/env python3
"""Tests for tools/release_manifest.py."""
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
SCRIPT = ROOT / "tools" / "release_manifest.py"


def load():
    spec = importlib.util.spec_from_file_location("release_manifest", SCRIPT)
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


class ReleaseManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="release_manifest_"))
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

    def _repo(self) -> tuple[Path, str]:
        root = self.tmp / "repo"
        root.mkdir()
        git(root, "init", "-b", "master")
        write(root / "feature.txt", "old\n")
        write(root / "screenshots" / "shot.png", "old\n")
        git(root, "add", "feature.txt", "screenshots/shot.png")
        git(root, "commit", "-m", "feat: seed")
        sha = git(root, "rev-parse", "HEAD")
        return root, sha

    def _manifest(self, root: Path, sha: str, *, status: str = "shipped") -> Path:
        path = root / "release-manifest.json"
        path.write_text(json.dumps({
            "schema": "release-manifest-v1",
            "producer": {"skill": "dispatch", "run_id": "RID-1", "packet_path": "packet.md"},
            "picks": [{
                "plan_id": "P",
                "phase_id": "P1",
                "status": status,
                "commit": sha,
                "files": ["feature.txt", "screenshots/shot.png", "not-dirty.txt"],
            }],
        }), encoding="utf-8")
        return path

    def test_consume_intersects_dirty_paths_and_auto_defers_artifacts(self) -> None:
        rm = load()
        root, sha = self._repo()
        path = self._manifest(root, sha)
        write(root / "feature.txt", "new\n")
        write(root / "screenshots" / "shot.png", "new\n")

        envelope = rm.consume_manifest(root, path)

        self.assertTrue(envelope["ok"])
        self.assertEqual(envelope["staged_paths"], ["feature.txt"])
        self.assertEqual(envelope["auto_deferred_paths"], ["screenshots/shot.png"])
        self.assertEqual(envelope["unreachable_commits"], [])
        self.assertEqual(envelope["awaiting_commit_picks"], [])
        self.assertIn("feat: seed", envelope["commits_section_lines"][0])

    def test_non_shipped_pick_blocks_release(self) -> None:
        rm = load()
        root, sha = self._repo()
        path = self._manifest(root, sha, status="awaiting_commit")
        write(root / "feature.txt", "new\n")

        envelope = rm.consume_manifest(root, path)

        self.assertFalse(envelope["ok"])
        self.assertEqual(rm.exit_code(envelope), 4)
        self.assertEqual(envelope["awaiting_commit_picks"][0]["status"], "awaiting_commit")

    def test_rejects_paths_that_escape_repo(self) -> None:
        rm = load()
        for bad_path in ("../secret.txt", "nested/..", "nested/../secret.txt"):
            with self.subTest(path=bad_path):
                with self.assertRaises(rm.ManifestError):
                    rm.normalize_manifest({
                        "schema": "fleet-release-manifest/1",
                        "picks": [{
                            "status": "shipped",
                            "commit": "abcdef0",
                            "files": [bad_path],
                        }],
                    })

    def test_rejects_non_string_paths_and_dedupes_duplicates(self) -> None:
        rm = load()
        with self.assertRaises(rm.ManifestError):
            rm.normalize_manifest({
                "schema": "fleet-release-manifest/1",
                "picks": [{
                    "status": "shipped",
                    "commit": "abcdef0",
                    "files": ["feature.txt", 123],
                }],
            })

        manifest = rm.normalize_manifest({
            "schema": "fleet-release-manifest/1",
            "picks": [{
                "status": "shipped",
                "commit": "abcdef0",
                "files": ["./feature.txt", "feature.txt"],
            }],
        })
        self.assertEqual(manifest["picks"][0]["files"], ["feature.txt"])

    def test_normalize_path_preserves_dot_directories(self) -> None:
        rm = load()
        self.assertEqual(rm.rel_path("./tools/release.py"), "tools/release.py")
        self.assertEqual(rm.rel_path(".claude/skills/release/SKILL.md"), ".claude/skills/release/SKILL.md")

    def test_cli_consume_has_no_side_effects(self) -> None:
        root, sha = self._repo()
        path = self._manifest(root, sha)
        write(root / "feature.txt", "new\n")
        before = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z"],
            cwd=root,
            text=True,
            capture_output=True,
        ).stdout
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "consume", str(path), "--json"],
            cwd=root,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["staged_paths"], ["feature.txt"])
        after = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z"],
            cwd=root,
            text=True,
            capture_output=True,
        ).stdout
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
