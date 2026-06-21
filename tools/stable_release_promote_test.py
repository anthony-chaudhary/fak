#!/usr/bin/env python3
"""Hermetic tests for tools/stable_release_promote.py."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "stable_release_promote.py"


def load():
    spec = importlib.util.spec_from_file_location("stable_release_promote", SCRIPT)
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


class StableReleasePromoteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="stable_promote_"))
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
        write(root / "VERSION", "0.1.0\n")
        git(root, "add", "VERSION")
        git(root, "commit", "-m", "seed")
        git(root, "tag", "-a", "v0.1.0", "-m", "v0.1.0")
        sha = git(root, "rev-parse", "HEAD")
        return root, sha

    def _payload(self, sha: str, **overrides) -> dict:
        payload = {
            "candidate_tag": "v0.1.0",
            "candidate_sha": sha,
            "codename": "2026-06-bedrock",
            "window_days": 0,
            "tag_collision": None,
            "idempotency": {
                "tag_exists": False,
                "tag_sha": None,
                "tag_matches_candidate": None,
                "evidence_file_exists": False,
                "evidence_path": "docs/stable-releases/2026-06-bedrock.md",
            },
            "gate": {
                "release_topology": {"name": "release_topology", "pass": True},
                "local_release_tests": {"name": "local_release_tests", "pass": True},
                "dos_doctor": {"name": "dos_doctor", "pass": True},
                "ci_green": {"name": "ci_green", "pass": True, "advisory": True, "verdict": "NO_SIGNAL"},
                "tag_age": {"name": "tag_age", "pass": True, "age_days": 3, "window_days": 0},
            },
            "summary": {"all_green": True, "blockers": [], "forced": False},
        }
        payload.update(overrides)
        return payload

    def test_render_evidence_contains_frontmatter_and_gate_json(self) -> None:
        srp = load()
        root, sha = self._repo()
        text = srp.render_evidence_file(
            self._payload(sha),
            generated_at_utc="2026-06-18T00:00:00+00:00",
        )
        self.assertTrue(text.startswith("---\n"))
        self.assertIn("underlying_version: \"v0.1.0\"", text)
        self.assertIn("candidate_sha: " + repr(sha).replace("'", '"'), text)
        self.assertIn("```json", text)
        self.assertIn('"release_topology"', text)
        self.assertFalse((root / "docs").exists())

    def test_promote_writes_evidence_and_annotated_tag(self) -> None:
        srp = load()
        root, sha = self._repo()
        result = srp.promote_from_context(root, self._payload(sha))

        self.assertTrue(result["ok"], result["errors"])
        evidence = root / "docs" / "stable-releases" / "2026-06-bedrock.md"
        self.assertTrue(evidence.is_file())
        self.assertTrue(srp.evidence_matches(evidence, self._payload(sha)))
        self.assertEqual(git(root, "rev-list", "-n1", "stable/2026-06-bedrock"), sha)

    def test_push_refuses_until_evidence_is_committed_at_head(self) -> None:
        srp = load()
        root, sha = self._repo()

        result = srp.promote_from_context(root, self._payload(sha), push=True)

        self.assertFalse(result["ok"])
        self.assertTrue(result["evidence_written"])
        self.assertTrue(any("evidence_not_committed_before_tag_push" in e for e in result["errors"]))
        self.assertEqual(git(root, "tag", "-l", "stable/2026-06-bedrock"), "")

    def test_push_succeeds_after_evidence_commit(self) -> None:
        srp = load()
        root, sha = self._repo()
        payload = self._payload(sha)
        evidence = root / "docs" / "stable-releases" / "2026-06-bedrock.md"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(srp.render_evidence_file(payload), encoding="utf-8", newline="")
        git(root, "add", "docs/stable-releases/2026-06-bedrock.md")
        git(root, "commit", "-m", "docs(stable): evidence")
        ok, reason = srp.evidence_committed_at_head(root, Path("docs/stable-releases/2026-06-bedrock.md"), payload)
        self.assertTrue(ok, reason)

        real_git = srp.git

        def fake_git(repo: Path, args: list[str], *, timeout: int = 600) -> tuple[int, str]:
            if args == ["push", "origin", "stable/2026-06-bedrock"]:
                return 0, ""
            return real_git(repo, args, timeout=timeout)

        srp.git = fake_git
        try:
            result = srp.promote_from_context(root, payload, push=True)
        finally:
            srp.git = real_git

        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(result["tag_created"])
        self.assertTrue(result["tag_pushed"])
        self.assertEqual(git(root, "rev-list", "-n1", "stable/2026-06-bedrock"), sha)

    def test_promote_is_idempotent_on_reinvoke(self) -> None:
        srp = load()
        root, sha = self._repo()
        payload = self._payload(sha)
        first = srp.promote_from_context(root, payload)
        self.assertTrue(first["ok"], first["errors"])
        evidence = root / "docs" / "stable-releases" / "2026-06-bedrock.md"
        before = evidence.read_bytes()

        payload["idempotency"] = {
            "tag_exists": True,
            "tag_sha": sha,
            "tag_matches_candidate": True,
            "evidence_file_exists": True,
            "evidence_path": "docs/stable-releases/2026-06-bedrock.md",
        }
        second = srp.promote_from_context(root, payload)

        self.assertTrue(second["ok"], second["errors"])
        self.assertIn("evidence_file_already_exists", second["idempotent_skips"])
        self.assertIn("tag_already_exists_same_sha", second["idempotent_skips"])
        self.assertEqual(evidence.read_bytes(), before)
        self.assertEqual(len(git(root, "tag", "-l", "stable/2026-06-bedrock").splitlines()), 1)

    def test_red_gate_refuses_without_rationale_and_force_records_rationale(self) -> None:
        srp = load()
        root, sha = self._repo()
        red = self._payload(
            sha,
            gate={"tag_age": {"name": "tag_age", "pass": False}},
            summary={"all_green": False, "blockers": ["gate row tag_age did not pass"], "forced": False},
        )

        refused = srp.promote_from_context(root, red)
        self.assertFalse(refused["ok"])
        self.assertFalse((root / "docs" / "stable-releases" / "2026-06-bedrock.md").exists())
        self.assertEqual(git(root, "tag", "-l", "stable/2026-06-bedrock"), "")

        forced = srp.promote_from_context(root, red, rationale="operator accepted short soak")
        self.assertTrue(forced["ok"], forced["errors"])
        text = (root / "docs" / "stable-releases" / "2026-06-bedrock.md").read_text(encoding="utf-8")
        self.assertIn("## Force-promote rationale", text)
        self.assertIn("operator accepted short soak", text)

    def test_dry_run_has_no_side_effects(self) -> None:
        srp = load()
        root, sha = self._repo()
        result = srp.promote_from_context(root, self._payload(sha), dry_run=True)
        self.assertTrue(result["ok"], result["errors"])
        self.assertFalse((root / "docs").exists())
        self.assertEqual(git(root, "tag", "-l", "stable/2026-06-bedrock"), "")

    def test_live_cli_dry_run_no_mutation(self) -> None:
        codename = "2026-06-promote-smoke"
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
                "--dry-run",
                "--json",
            ],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        self.assertIn(proc.returncode, (0, 1), proc.stderr)
        payload = __import__("json").loads(proc.stdout)
        self.assertIn("ok", payload)
        tag_after = subprocess.run(
            ["git", "tag", "-l", f"stable/{codename}"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        ).stdout
        self.assertEqual(evidence.exists(), evidence_before)
        self.assertEqual(tag_after, tag_before)

    def test_live_stable_tags_have_matching_evidence_when_present(self) -> None:
        srp = load()
        code, out = srp.git(ROOT, [
            "for-each-ref",
            "--format=%(refname:short)|%(*objectname)|%(objectname)",
            "refs/tags/stable/",
        ])
        self.assertEqual(code, 0, out)
        failures: list[str] = []
        for line in out.splitlines():
            tag, peeled, raw = line.split("|", 2)
            sha = peeled or raw
            codename = tag.split("/", 1)[1]
            evidence = ROOT / "docs" / "stable-releases" / f"{codename}.md"
            if not evidence.exists():
                failures.append(f"{tag}: missing evidence file")
                continue
            text = evidence.read_text(encoding="utf-8")
            if f"candidate_sha: \"{sha}\"" not in text and sha[:12] not in text:
                failures.append(f"{tag}: evidence does not mention tag commit")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
