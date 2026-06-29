#!/usr/bin/env python3
"""Tests for tools/release_status.py."""
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
SCRIPT = ROOT / "tools" / "release_status.py"
LIVE_SMOKE_PATHS = [
    "VERSION",
    ".github/workflows/release-cadence.yml",
    "docs/releases",
    "docs/stable-releases",
    "tools/release_status.py",
    "tools/release_context.py",
    "tools/release_decide.py",
    "tools/release_cut.py",
    "tools/release_dry_run.py",
    "tools/release_tag.py",
    "tools/release_lock.py",
]


def load():
    spec = importlib.util.spec_from_file_location("release_status", SCRIPT)
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


class ReleaseStatusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="release_status_"))
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

    def test_stable_summary_reports_lag_from_rolling_tags(self) -> None:
        rs = load()
        root = self.tmp / "repo"
        root.mkdir()
        git(root, "init", "-b", "master")
        write(root / "VERSION", "0.1.0\n")
        git(root, "add", "VERSION")
        git(root, "commit", "-m", "v0.1.0 seed")
        git(root, "tag", "-a", "v0.1.0", "-m", "v0.1.0")
        git(root, "tag", "-a", "stable/2026-06-bedrock", "-m", "stable")
        stable_sha = git(root, "rev-list", "-n1", "stable/2026-06-bedrock")
        write(
            root / "docs" / "stable-releases" / "2026-06-bedrock.md",
            (
                "---\n"
                "codename: 2026-06-bedrock\n"
                "underlying_version: v0.1.0\n"
                f"candidate_sha: {stable_sha}\n"
                "---\n"
                "body\n"
            ),
        )
        git(root, "add", "docs/stable-releases/2026-06-bedrock.md")
        git(root, "commit", "-m", "docs(stable): promote bedrock")
        write(root / "VERSION", "0.2.0\n")
        git(root, "add", "VERSION")
        git(root, "commit", "-m", "feat: next")
        git(root, "tag", "-a", "v0.2.0", "-m", "v0.2.0")

        summary = rs.stable_summary(root, rs.semver_tags(root))

        self.assertEqual(summary["latest_stable"]["tag"], "stable/2026-06-bedrock")
        self.assertEqual(summary["latest_stable"]["version"], "0.1.0")
        self.assertEqual(summary["stable_lag"], 1)
        self.assertTrue(summary["evidence"]["ok"])
        self.assertEqual(summary["evidence"]["checked"], 1)
        self.assertIn("stable lags by 1", summary["recommendation"])

    def test_stable_evidence_status_catches_missing_and_mismatched_files(self) -> None:
        rs = load()
        root = self.tmp / "stable_repo"
        root.mkdir()
        git(root, "init", "-b", "master")
        write(root / "VERSION", "0.1.0\n")
        git(root, "add", "VERSION")
        git(root, "commit", "-m", "v0.1.0 seed")
        git(root, "tag", "-a", "v0.1.0", "-m", "v0.1.0")
        git(root, "tag", "-a", "stable/2026-06-bedrock", "-m", "stable")

        missing = rs.stable_evidence_status(root, rs.stable_tags(root))
        self.assertFalse(missing["ok"])
        self.assertIn("missing committed evidence file", missing["failures"][0]["detail"])

        write(
            root / "docs" / "stable-releases" / "2026-06-bedrock.md",
            "---\nunderlying_version: v0.1.0\ncandidate_sha: deadbee\n---\nbody\n",
        )
        git(root, "add", "docs/stable-releases/2026-06-bedrock.md")
        git(root, "commit", "-m", "docs(stable): add bad evidence")
        mismatch = rs.stable_evidence_status(root, rs.stable_tags(root))
        self.assertFalse(mismatch["ok"])
        self.assertTrue(any("does not match" in f["detail"] for f in mismatch["failures"]))

    def test_stable_evidence_status_accepts_matching_frontmatter(self) -> None:
        rs = load()
        root = self.tmp / "stable_ok_repo"
        root.mkdir()
        git(root, "init", "-b", "master")
        write(root / "VERSION", "0.1.0\n")
        git(root, "add", "VERSION")
        git(root, "commit", "-m", "v0.1.0 seed")
        git(root, "tag", "-a", "v0.1.0", "-m", "v0.1.0")
        git(root, "tag", "-a", "stable/2026-06-bedrock", "-m", "stable")
        sha = git(root, "rev-list", "-n1", "stable/2026-06-bedrock")
        write(
            root / "docs" / "stable-releases" / "2026-06-bedrock.md",
            (
                "---\n"
                "codename: 2026-06-bedrock\n"
                "underlying_version: v0.1.0\n"
                f"candidate_sha: {sha}\n"
                "---\n"
                "body\n"
            ),
        )
        git(root, "add", "docs/stable-releases/2026-06-bedrock.md")
        git(root, "commit", "-m", "docs(stable): add evidence")

        status = rs.stable_evidence_status(root, rs.stable_tags(root))

        self.assertTrue(status["ok"])
        self.assertEqual(status["failures"], [])

    def test_stable_candidate_status_reports_soak_ready_and_existing_promotion(self) -> None:
        rs = load()
        root = self.tmp / "stable_candidate_repo"
        root.mkdir()
        git(root, "init", "-b", "master")
        write(root / "VERSION", "0.1.0\n")
        git(root, "add", "VERSION")
        git(root, "commit", "-m", "v0.1.0 seed")
        git(root, "tag", "-a", "v0.1.0", "-m", "v0.1.0")
        created = rs.tag_epoch(root, "v0.1.0")
        self.assertIsInstance(created, int)

        soaking = rs.stable_candidate_status(
            root,
            rs.semver_tags(root),
            [],
            window_days=3,
            now=created + 86400,
        )
        self.assertFalse(soaking["ready"])
        self.assertEqual(soaking["state"], "soaking")
        self.assertEqual(soaking["remaining_days"], 2.0)

        ready = rs.stable_candidate_status(
            root,
            rs.semver_tags(root),
            [],
            window_days=3,
            now=created + 4 * 86400,
        )
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["state"], "ready")
        self.assertRegex(ready["suggested_codename"], r"^\d{4}-\d{2}-stable$")
        self.assertIn("--from v0.1.0", ready["dry_run_command"])

        git(root, "tag", "-a", "stable/2026-06-bedrock", "-m", "stable")
        promoted = rs.stable_candidate_status(
            root,
            rs.semver_tags(root),
            rs.stable_tags(root),
            window_days=3,
            now=created + 4 * 86400,
        )
        self.assertFalse(promoted["ready"])
        self.assertEqual(promoted["state"], "already_promoted")
        self.assertEqual(promoted["stable_tag"], "stable/2026-06-bedrock")

    def test_next_action_prioritizes_release_blockers(self) -> None:
        rs = load()
        self.assertEqual(
            rs.first_ci_run({"value": [{"databaseId": 1, "conclusion": "failure"}]})["databaseId"],
            1,
        )
        self.assertEqual(rs.first_ci_run({"value": []}), {})
        self.assertEqual(
            rs.next_action(
                {"decision": "release", "next_version": "0.3.0"},
                {},
                {"clean": False, "modified_count": 2, "untracked_count": 1},
            )["kind"],
            "clean_worktree",
        )
        billing = rs.classify_ci_annotations([
            "The job was not started because recent account payments have failed or your spending limit needs to be increased.",
        ])
        self.assertEqual(billing["action"], "fix_ci_billing")
        self.assertTrue(rs.is_release_relevant_dirty_path("tools/release_status.py"))
        self.assertTrue(rs.is_release_relevant_dirty_path(".github/workflows/ci.yml"))
        self.assertFalse(rs.is_release_relevant_dirty_path("tools/control_pane.loops.json"))
        unrelated_dirty = {
            "clean": False,
            "modified_count": 1,
            "untracked_count": 0,
            "release_relevant_count": 0,
        }
        self.assertEqual(
            rs.next_action(
                {"decision": "hold", "blockers": ["CI_BASE_RED"]},
                {},
                {"clean": True},
                billing,
            )["kind"],
            "fix_ci_billing",
        )
        self.assertEqual(
            rs.next_action(
                {"decision": "hold", "blockers": ["CI_BASE_RED"]},
                {},
                unrelated_dirty,
                billing,
            )["kind"],
            "fix_ci_billing",
        )
        self.assertEqual(
            rs.next_action(
                {"decision": "hold", "blockers": ["CI_BASE_RED"]},
                {},
                {
                    "clean": False,
                    "modified_count": 1,
                    "untracked_count": 0,
                    "release_relevant_count": 1,
                },
                billing,
            )["kind"],
            "clean_worktree",
        )
        self.assertEqual(
            rs.next_action({"decision": "hold", "blockers": ["CI_BASE_RED"]}, {})["kind"],
            "fix_ci",
        )
        self.assertEqual(
            rs.next_action({"decision": "hold", "blockers": ["CI_RETRY_TO_GREEN"]}, {})["kind"],
            "pause_auto_release",
        )
        self.assertEqual(
            rs.next_action({"decision": "release", "next_version": "0.3.0"}, {}, unrelated_dirty)["kind"],
            "clean_worktree",
        )
        self.assertEqual(
            rs.next_action({"decision": "release", "next_version": "0.3.0"}, {})["kind"],
            "cut_release",
        )
        self.assertEqual(
            rs.next_action({"decision": "hold", "blockers": []}, {"stable_lag": 2, "recommendation": "lag"})["kind"],
            "consider_stable",
        )
        self.assertEqual(
            rs.next_action(
                {"decision": "hold", "blockers": []},
                {
                    "evidence": {"ok": True},
                    "candidate": {
                        "ready": True,
                        "candidate_tag": "v0.2.0",
                        "suggested_codename": "2026-06-stable",
                    },
                },
            )["kind"],
            "promote_stable",
        )
        self.assertEqual(
            rs.next_action(
                {"decision": "hold", "blockers": []},
                {"evidence": {"ok": False, "failures": [{"tag": "stable/x", "detail": "missing"}]}},
            )["kind"],
            "repair_stable_evidence",
        )

    def test_loop_status_fields_make_release_status_pane_friendly(self) -> None:
        rs = load()

        self.assertEqual(
            rs.loop_status_fields({"kind": "wait", "detail": "nothing pending"}),
            {"ok": True, "verdict": "OK", "detail": "nothing pending"},
        )
        self.assertEqual(
            rs.loop_status_fields({"kind": "fix_ci_billing", "detail": "billing needs attention"}),
            {"ok": False, "verdict": "ACTION", "detail": "billing needs attention"},
        )
        self.assertEqual(
            rs.loop_status_fields({"kind": "promote_stable", "detail": "promote v0.2.0"}),
            {"ok": False, "verdict": "ACTION", "detail": "promote v0.2.0"},
        )

    def test_ci_failure_diagnosis_falls_back_to_latest_trunk_annotation(self) -> None:
        rs = load()

        def fake_git(root: Path, args: list[str], *, timeout: int = 120) -> tuple[int, str]:
            self.assertEqual(args, ["rev-parse", "HEAD"])
            return 0, "localhead\n"

        def fake_load_json(cmd: list[str], root: Path, *, timeout: int = 300) -> tuple[dict | None, int, str]:
            if cmd[:3] == ["gh", "run", "list"] and "--commit" in cmd:
                return {"value": []}, 0, "[]"
            if cmd[:3] == ["gh", "run", "list"] and "--branch" in cmd:
                return {
                    "value": [{
                        "databaseId": "run-1",
                        "conclusion": "failure",
                        "status": "completed",
                        "headSha": "abc123",
                    }],
                }, 0, "[]"
            if cmd == ["gh", "run", "view", "run-1", "--json", "jobs"]:
                return {
                    "jobs": [{
                        "databaseId": "job-1",
                        "name": "build",
                        "conclusion": "failure",
                        "status": "completed",
                        "steps": [],
                    }],
                }, 0, "{}"
            if cmd == ["gh", "api", "repos/owner/repo/check-runs/job-1/annotations"]:
                return {
                    "value": [{
                        "annotation_level": "failure",
                        "path": ".github",
                        "message": (
                            "The job was not started because recent account payments have failed "
                            "or your spending limit needs to be increased."
                        ),
                    }],
                }, 0, "[]"
            raise AssertionError(cmd)

        rs.git = fake_git
        rs.load_json = fake_load_json
        rs.github_repo_slug = lambda root: "owner/repo"

        diagnosis = rs.ci_failure_diagnosis(Path("repo"))

        self.assertEqual(diagnosis["status"], "diagnosed")
        self.assertEqual(diagnosis["scope"], "latest_trunk")
        self.assertEqual(diagnosis["action"], "fix_ci_billing")
        self.assertEqual(diagnosis["jobs"][0]["steps"], 0)

    def test_cadence_status_detects_workflow_contract(self) -> None:
        rs = load()
        status = rs.cadence_status(ROOT)
        self.assertTrue(status["present"])
        self.assertTrue(status["manual_dispatch"])
        self.assertTrue(status["dry_run_first"])
        self.assertTrue(status["tag_after_green"])
        self.assertTrue(status["checked_github_release"])

    def test_live_cli_smoke_no_mutation(self) -> None:
        before = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--", *LIVE_SMOKE_PATHS],
            cwd=ROOT,
            text=True,
            capture_output=True,
        ).stdout
        proc = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--json",
                "--skip-gh",
                "--skip-cut-plan",
                "--limit-commits", "20",
            ],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["schema"], "fleet-release-status/1")
        self.assertIn("next_action", payload)
        self.assertIn(payload["verdict"], {"OK", "ACTION"})
        self.assertIsInstance(payload["ok"], bool)
        self.assertEqual(payload["detail"], payload["next_action"]["detail"])
        after = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--", *LIVE_SMOKE_PATHS],
            cwd=ROOT,
            text=True,
            capture_output=True,
        ).stdout
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
