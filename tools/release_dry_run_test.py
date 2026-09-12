#!/usr/bin/env python3
"""Hermetic tests for tools/release_dry_run.py."""
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
SCRIPT = ROOT / "tools" / "release_dry_run.py"


def load():
    spec = importlib.util.spec_from_file_location("release_dry_run", SCRIPT)
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


class ReleaseDryRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="release_dry_run_"))
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
        write(root / "README.md", "seed\n")
        git(root, "add", "README.md")
        git(root, "commit", "-m", "seed")
        return root

    def test_workflow_parse_catches_invalid_yaml_at_ref(self) -> None:
        rd = load()
        root = self._repo()
        write(root / ".github" / "workflows" / "ci.yml", "name: [\n")
        git(root, "add", ".github/workflows/ci.yml")
        git(root, "commit", "-m", "bad workflow")
        verdict = rd.parse_workflows_at_ref(root, "HEAD")
        if verdict["ok"] is None:
            self.skipTest("PyYAML unavailable")
        self.assertFalse(verdict["ok"])
        self.assertIn(".github/workflows/ci.yml", verdict["files"])

    def test_dry_run_runs_present_tests_in_isolated_clone(self) -> None:
        rd = load()
        root = self._repo()
        write(root / ".github" / "workflows" / "ci.yml", "name: ci\non: push\njobs: {}\n")
        write(root / "tools" / "release_context_test.py", (
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_ok(self):\n"
            "        self.assertTrue(True)\n"
            "if __name__ == '__main__':\n"
            "    unittest.main(verbosity=2)\n"
        ))
        git(root, "add", ".github/workflows/ci.yml", "tools/release_context_test.py")
        git(root, "commit", "-m", "add release test")
        verdict = rd.dry_run(root, "HEAD")
        self.assertTrue(verdict["ok"], verdict)
        self.assertTrue(verdict["suite"]["ran"])
        self.assertEqual(verdict["suite"]["tests"][0]["test"], "tools/release_context_test.py")

    def test_fast_mode_runs_only_release_perturbation_subset(self) -> None:
        rd = load()
        root = self._repo()
        write(root / ".github" / "workflows" / "ci.yml", "name: ci\non: push\njobs: {}\n")
        write(root / "tools" / "release_context_test.py", (
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_ok(self):\n"
            "        self.assertTrue(True)\n"
            "if __name__ == '__main__':\n"
            "    unittest.main(verbosity=2)\n"
        ))
        write(root / "tools" / "stable_release_promote_test.py", (
            "raise SystemExit('stable test should not run in fast mode')\n"
        ))
        git(root, "add", ".github/workflows/ci.yml", "tools/release_context_test.py", "tools/stable_release_promote_test.py")
        git(root, "commit", "-m", "add release tests")

        verdict = rd.dry_run(root, "HEAD", fast=True)

        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(verdict["mode"], "fast")
        self.assertEqual([row["test"] for row in verdict["suite"]["tests"]], ["tools/release_context_test.py"])
        self.assertIn("mode=fast", verdict["trailer"])

    def test_live_cli_skip_tests_no_mutation(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "HEAD", "--skip-tests", "--json"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        self.assertIn(proc.returncode, (0, 1), proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertIn("workflows", payload)
        self.assertTrue(payload["suite"]["skipped"])
        self.assertFalse(payload["suite"]["ran"])

    def test_live_cli_fast_skip_tests_reports_mode_without_mutation(self) -> None:
        before = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        ).stdout
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "HEAD", "--fast", "--skip-tests", "--json"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        self.assertIn(proc.returncode, (0, 1), proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["mode"], "fast")
        self.assertEqual(payload["suite"]["mode"], "fast")
        after = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        ).stdout
        self.assertEqual(after, before)

    def test_ci_runs_fast_dry_run_before_full_witness(self) -> None:
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        fast = "python tools/release_dry_run.py --fast --json HEAD"
        full = "python tools/release_dry_run.py --json HEAD"

        self.assertIn(fast, ci)
        self.assertIn(full, ci)
        self.assertLess(ci.index(fast), ci.index(full))

    def _repo_with_tag_contracts(self, failing: str | None = None) -> Path:
        root = self._repo()
        fixture_paths = (
            ".github/workflows/release-artifacts.yml",
            ".github/workflows/release-macos.yml",
            ".github/workflows/release-container.yml",
            ".github/workflows/release-cuda-container.yml",
            ".github/workflows/ci.yml",
            "install.sh",
            "Dockerfile",
            "Dockerfile.cuda",
            "Makefile",
            "scripts/build.sh",
            "internal/compute/cuda_arch.txt",
            "internal/compute/build_cuda.sh",
            "internal/compute/setup_cuda_wsl.sh",
            "tools/build_cuda_windows.ps1",
            "tools/release_artifacts_workflow_test.py",
            "tools/cuda_arch_targets_test.py",
        )
        for path in fixture_paths:
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, target)

        workflow_path = root / ".github/workflows/release-artifacts.yml"
        workflow = workflow_path.read_text(encoding="utf-8")
        if failing == "artifact":
            workflow = workflow.replace(
                "          - goos: darwin\n            goarch: arm64\n", "", 1)
        write(workflow_path, workflow)
        if failing == "cuda":
            docker_path = root / "Dockerfile.cuda"
            write(
                docker_path,
                docker_path.read_text(encoding="utf-8").replace("12.8.1", "12.4.1"),
            )
        git(root, "add", *fixture_paths)
        git(root, "commit", "-m", "add tag contracts")
        return root

    def test_fast_dry_run_reports_artifact_contract_and_command(self) -> None:
        root = self._repo_with_tag_contracts(failing="artifact")
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "HEAD", "--fast", "--json"],
            cwd=root, text=True, encoding="utf-8", capture_output=True,
        )

        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        failure = json.loads(proc.stdout)["suite"]["failures"][0]
        self.assertEqual(failure["contract"], "artifact")
        self.assertEqual(
            failure["command"],
            [sys.executable, "tools/release_artifacts_workflow_test.py"],
        )

    def test_fast_dry_run_labels_cuda_contract_failure(self) -> None:
        root = self._repo_with_tag_contracts(failing="cuda")
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "HEAD", "--fast", "--json"],
            cwd=root, text=True, encoding="utf-8", capture_output=True,
        )

        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        failure = json.loads(proc.stdout)["suite"]["failures"][0]
        self.assertEqual(failure["contract"], "cuda")
        self.assertEqual(
            failure["command"],
            [sys.executable, "tools/cuda_arch_targets_test.py"],
        )
        human = subprocess.run(
            [sys.executable, str(SCRIPT), "HEAD", "--fast"],
            cwd=root, text=True, encoding="utf-8", capture_output=True,
        )
        self.assertEqual(human.returncode, 1, human.stdout + human.stderr)
        self.assertIn(
            "tools/cuda_arch_targets_test.py [cuda contract]: exit 1",
            human.stdout,
        )

    def test_full_dry_run_runs_both_valid_tag_contracts(self) -> None:
        rd = load()
        verdict = rd.dry_run(self._repo_with_tag_contracts(), "HEAD")

        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(verdict["mode"], "full")
        ran = {row["test"] for row in verdict["suite"]["tests"]}
        self.assertIn("tools/release_artifacts_workflow_test.py", ran)
        self.assertIn("tools/cuda_arch_targets_test.py", ran)


if __name__ == "__main__":
    unittest.main(verbosity=2)
