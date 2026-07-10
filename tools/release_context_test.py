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
DECIDE_SCRIPT = ROOT / "tools" / "release_decide.py"


def load():
    spec = importlib.util.spec_from_file_location("release_context", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_decide():
    """Load the CONSUMER so a test can pin the producer/consumer CI-signal contract."""
    import sys
    tools = str(ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location("release_decide_under_test", DECIDE_SCRIPT)
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

    def test_run_age_seconds_parses_and_fails_soft(self) -> None:
        rc = load()
        self.assertIsNone(rc._run_age_seconds(None))
        self.assertIsNone(rc._run_age_seconds(""))
        self.assertIsNone(rc._run_age_seconds("not-a-timestamp"))
        # A long-past timestamp yields a large positive age; a future one clamps to 0.
        self.assertGreater(rc._run_age_seconds("2020-01-01T00:00:00Z"), 0)
        self.assertEqual(rc._run_age_seconds("2099-01-01T00:00:00Z"), 0)

    def test_decisive_runs_with_ancestry_maps_history(self) -> None:
        rc = load()
        root = self._repo()  # seed commit + v0.1.0, branch master, cwd == root
        shas = []
        for i in range(3):
            write(root / "x.txt", f"{i}\n")
            git(root, "add", "x.txt")
            git(root, "commit", "-m", f"c{i}")
            shas.append(subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
        head = shas[-1]
        # A trunk run list, newest-first: an indecisive run on HEAD (skipped), a
        # green run 2 commits back (ancestor), and a red run on a fabricated,
        # unknown sha (non-ancestor).
        trunk = [
            {"conclusion": "cancelled", "headSha": head, "updatedAt": None},
            {"conclusion": "success", "headSha": shas[0], "updatedAt": None},
            {"conclusion": "failure", "headSha": "0" * 40, "updatedAt": None},
        ]
        rows = rc.decisive_runs_with_ancestry(trunk, head)
        # The cancelled run is not decisive and is dropped.
        self.assertEqual(len(rows), 2)
        green, red = rows
        self.assertEqual(green["result"], "green")
        self.assertTrue(green["ancestor_of_head"])
        self.assertEqual(green["commits_behind_head"], 2)
        self.assertEqual(red["result"], "red")
        self.assertFalse(red["ancestor_of_head"])
        self.assertEqual(red["commits_behind_head"], -1)

    def test_decisive_runs_with_ancestry_is_empty_without_inputs(self) -> None:
        rc = load()
        self.assertEqual(rc.decisive_runs_with_ancestry(None, "deadbeef"), [])
        self.assertEqual(rc.decisive_runs_with_ancestry([{"conclusion": "success"}], ""), [])

    def test_fold_latest_trunk_ci_names_the_deciding_workflow(self) -> None:
        rc = load()
        _, _, red_note = rc.fold_latest_trunk_ci(
            [{"conclusion": "failure", "headSha": "abcdef123456"}], "ci-fast.yml")
        self.assertIn("ci-fast.yml", red_note)
        _, _, none_note = rc.fold_latest_trunk_ci([], "ci-fast.yml")
        self.assertIn("ci-fast.yml", none_note)
        # Default stays ci.yml so the whole-CI signal's wording is unchanged.
        _, _, default_note = rc.fold_latest_trunk_ci([])
        self.assertIn("ci.yml", default_note)

    def _commits(self, root: Path, n: int) -> list[str]:
        shas = []
        for i in range(n):
            write(root / "x.txt", f"{i}\n")
            git(root, "add", "x.txt")
            git(root, "commit", "-m", f"c{i}")
            shas.append(subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
        return shas

    def test_ci_signal_for_workflow_queries_that_workflow_and_skips_runs_on_head(self) -> None:
        rc = load()
        root = self._repo()
        head = self._commits(root, 1)[0]
        calls: list[list[str]] = []

        def fake(args: list[str]) -> object:
            calls.append(args)
            # The trunk query is the one filtering on completed runs.
            if "--status" in args:
                return [{"conclusion": "success", "headSha": head, "updatedAt": None}]
            return []

        rc._run_gh_json = fake
        sig = rc.ci_signal_for_workflow("ci-fast.yml", "main", include_runs_on_head=False)
        self.assertEqual(sig["workflow"], "ci-fast.yml")
        self.assertEqual(sig["status"], "green")
        # Exactly one gh call: the runs_on_head probe is skipped off the critical path.
        self.assertEqual(len(calls), 1)
        self.assertIn("ci-fast.yml", calls[0])
        self.assertEqual(sig["recent_decisive"][0]["commits_behind_head"], 0)

    def test_ci_fast_workflow_name_is_env_configurable(self) -> None:
        os.environ["FAK_RELEASE_FAST_CI_WORKFLOW"] = "custom-fast.yml"
        self.addCleanup(os.environ.pop, "FAK_RELEASE_FAST_CI_WORKFLOW", None)
        rc = load()
        self.assertEqual(rc.FAST_CI_WORKFLOW, "custom-fast.yml")

    def test_decisive_ci_fast_wins_the_gate_over_a_red_whole_ci(self) -> None:
        """#1374's contract: the producer's decisive ci_fast is what release_decide gates on.

        Before this producer existed the payload carried no `ci_fast`, so
        `effective_ci` always fell back to the whole `-race`-inclusive ci.yml.
        """
        rc, rd = load(), load_decide()
        root = self._repo()
        head = self._commits(root, 1)[0]
        rc._run_gh_json = lambda args: (
            [{"conclusion": "success", "headSha": head, "updatedAt": None}]
            if "--status" in args else [])
        fast = rc.ci_signal_for_workflow("ci-fast.yml", "main", include_runs_on_head=False)

        payload = {"ci_on_head": {"status": "red", "recent_decisive": []}, "ci_fast": fast}
        signal, source = rd.effective_ci(payload)
        self.assertEqual(source, "fast")
        self.assertEqual(signal["status"], "green")

    def test_churned_fast_signal_relaxes_on_a_green_ancestor(self) -> None:
        """#1374 + #2655 together: the exact always-on-dev starvation case.

        The newest decisive fast run is a RED on a superseded, non-ancestor commit,
        so the head signal is red; a green run 2 commits back IS an ancestor with
        nothing red between it and HEAD. The cut proceeds, relaxed on the ancestor.
        """
        rc, rd = load(), load_decide()
        root = self._repo()
        shas = self._commits(root, 3)
        head = shas[-1]
        trunk = [
            {"conclusion": "failure", "headSha": "0" * 40, "updatedAt": None},
            {"conclusion": "success", "headSha": shas[0], "updatedAt": None},
        ]
        rc._run_gh_json = lambda args: trunk if "--status" in args else []
        fast = rc.ci_signal_for_workflow("ci-fast.yml", "main", include_runs_on_head=False)
        self.assertEqual(fast["status"], "red")  # newest decisive run is the red one

        verdict = rd.decide({
            "commits_since_tag": [{"subject": "feat(release): real work", "body": ""}],
            "ci_on_head": {"status": "red", "recent_decisive": []},
            "ci_fast": fast,
            "version_files": {},
            "tag_drift": {},
            "workflows_parse_ok": {"ok": True},
        }, require_ci_green=True)
        self.assertEqual(verdict["decision"], "release")
        self.assertEqual(verdict["ci_source"], "fast+ancestor")
        self.assertTrue(verdict["ci_ancestor_relaxed"])

    def test_an_ancestor_red_since_the_green_still_holds_the_cut(self) -> None:
        """The safety property survives the new producer: a red BETWEEN green and HEAD holds."""
        rc, rd = load(), load_decide()
        root = self._repo()
        shas = self._commits(root, 3)
        head = shas[-1]
        trunk = [
            {"conclusion": "failure", "headSha": shas[1], "updatedAt": None},
            {"conclusion": "success", "headSha": shas[0], "updatedAt": None},
        ]
        rc._run_gh_json = lambda args: trunk if "--status" in args else []
        fast = rc.ci_signal_for_workflow("ci-fast.yml", "main", include_runs_on_head=False)

        verdict = rd.decide({
            "commits_since_tag": [{"subject": "feat(release): real work", "body": ""}],
            "ci_on_head": {"status": "red", "recent_decisive": []},
            "ci_fast": fast,
            "version_files": {},
            "tag_drift": {},
            "workflows_parse_ok": {"ok": True},
        }, require_ci_green=True)
        self.assertEqual(verdict["decision"], "hold")
        self.assertIn("CI_BASE_RED", verdict["blockers"])
        self.assertFalse(verdict["ci_ancestor_relaxed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
