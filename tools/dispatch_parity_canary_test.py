#!/usr/bin/env python3
"""Hermetic tests for tools/dispatch_parity_canary.py.

The gate composes six rungs of the parity bar. Rungs 1 (shipped) and 2
(witnessed) come from git + `dos commit-audit`; rungs 3-6 are pure functions of
the commit subject and the worker's git-command stream. Here we drive the pure
predicates and the `grade()` folding directly — every PARITY_PROVEN /
PARITY_UNPROVEN_* / PARITY_UNOBSERVED verdict — plus the run-log git extractor.
No git, no subprocess, no network.

The fixtures encode the REAL glm-5.2 canary (#420's evidence): opencode shipped
7528df3 for #545 by explicit pathspec, signed off, #545 in the subject, lane
tree clean — so the proven-path fixture is the one the operator actually relies
on, not a synthetic happy path.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "dispatch_parity_canary.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("dispatch_parity_canary", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pc = load()


# The real glm-5.2 canary's git stream, the way it appears in
# .dispatch-runs/resolve-545-20260623-162209.log (prose stripped to git cmds).
REAL_GLM_GIT_CMDS = "\n".join([
    "git status --porcelain",
    "git log --oneline -5",
    "git config user.name",
    "git commit -s -- internal/recall/ctxplan_store.go internal/recall/ctxplan_store_test.go",
    "git fetch origin main",
    "git push origin main",
    "git rev-parse --short HEAD",
])
REAL_GLM_SUBJECT = (
    "feat(ctxplan): add recall.Session -> ctxplan.Store adapter, "
    "the first real-store backing (#545) (fak recall)"
)


class IssueBound(unittest.TestCase):
    def test_exact_issue_cited(self):
        self.assertTrue(pc.issue_bound(REAL_GLM_SUBJECT, 545))

    def test_wrong_issue_not_bound(self):
        self.assertFalse(pc.issue_bound(REAL_GLM_SUBJECT, 999))

    def test_any_hash_when_issue_unset(self):
        self.assertTrue(pc.issue_bound("fix(x): a thing (#12)", None))
        self.assertFalse(pc.issue_bound("fix(x): a thing", None))

    def test_substring_issue_not_falsely_matched(self):
        # #54 must not match a subject that only carries #545
        self.assertFalse(pc.issue_bound("feat: thing (#545)", 54))


class ByPathspec(unittest.TestCase):
    def test_explicit_path_commit(self):
        self.assertTrue(pc.by_pathspec(REAL_GLM_GIT_CMDS))

    def test_blanket_add_dash_A_refutes(self):
        cmds = "git add -A\ngit commit -s -- foo.go"
        self.assertFalse(pc.by_pathspec(cmds))

    def test_blanket_add_all_refutes(self):
        self.assertFalse(pc.by_pathspec("git add --all\ngit commit -s -- foo"))

    def test_blanket_add_dot_refutes(self):
        self.assertFalse(pc.by_pathspec("git add .\ngit commit -s -- foo"))

    def test_add_named_path_is_not_blanket(self):
        cmds = "git add tools/x.py\ngit commit -s -- tools/x.py"
        self.assertTrue(pc.by_pathspec(cmds))

    def test_no_pathspec_commit_fails(self):
        # committed but with no `-- <path>` after it
        self.assertFalse(pc.by_pathspec("git commit -s -m 'thing'"))


class SignedOff(unittest.TestCase):
    def test_dash_s(self):
        self.assertTrue(pc.signed_off(REAL_GLM_GIT_CMDS))

    def test_long_signoff(self):
        self.assertTrue(pc.signed_off("git commit --signoff -- foo"))

    def test_no_signoff(self):
        self.assertFalse(pc.signed_off("git commit -- foo bar"))


class GitCmdsFromLog(unittest.TestCase):
    def test_extracts_git_from_ansi_log(self):
        # the opencode shell glyph: "\x1b[0m$ \x1b[0m<command>"
        log = (
            "\x1b[0mI'll commit now.\n"
            "\x1b[0m$ \x1b[0mgit commit -s -- a.go b.go\n"
            "\x1b[0mPush landed.\n"
            "\x1b[0m$ \x1b[0mgit push origin main\n"
        )
        cmds = pc.git_cmds_from_log(log)
        self.assertIn("git commit -s -- a.go b.go", cmds)
        self.assertIn("git push origin main", cmds)
        # prose lines without git are dropped
        self.assertNotIn("Push landed", cmds)


class GradeFold(unittest.TestCase):
    def _proven_inputs(self):
        return dict(
            shipped_a_unit=True,
            witnessed=True,
            subject=REAL_GLM_SUBJECT,
            issue=545,
            git_cmds=REAL_GLM_GIT_CMDS,
            lane_tree_clean=True,
        )

    def test_real_glm_canary_is_proven(self):
        g = pc.grade(**self._proven_inputs())
        self.assertTrue(g["proven"])
        self.assertEqual(g["verdict"], "PARITY_PROVEN")
        self.assertIsNone(g["failed_rung"])

    def test_no_unit_fails_first(self):
        inp = self._proven_inputs()
        inp["shipped_a_unit"] = False
        g = pc.grade(**inp)
        self.assertEqual(g["verdict"], "PARITY_UNPROVEN_NO_UNIT")
        self.assertEqual(g["failed_rung"], "shipped_a_unit")

    def test_unwitnessed_fails(self):
        inp = self._proven_inputs()
        inp["witnessed"] = False
        g = pc.grade(**inp)
        self.assertEqual(g["verdict"], "PARITY_UNPROVEN_UNWITNESSED")

    def test_unbound_fails(self):
        inp = self._proven_inputs()
        inp["issue"] = 999  # subject cites #545, not #999
        g = pc.grade(**inp)
        self.assertEqual(g["verdict"], "PARITY_UNPROVEN_UNBOUND")

    def test_blanket_add_fails(self):
        inp = self._proven_inputs()
        inp["git_cmds"] = "git add -A\ngit commit -s -- foo.go"
        g = pc.grade(**inp)
        self.assertEqual(g["verdict"], "PARITY_UNPROVEN_BLANKET_ADD")

    def test_no_signoff_fails(self):
        inp = self._proven_inputs()
        inp["git_cmds"] = "git commit -- foo.go bar.go"
        g = pc.grade(**inp)
        self.assertEqual(g["verdict"], "PARITY_UNPROVEN_NO_SIGNOFF")

    def test_dirty_tree_fails(self):
        inp = self._proven_inputs()
        inp["lane_tree_clean"] = False
        g = pc.grade(**inp)
        self.assertEqual(g["verdict"], "PARITY_UNPROVEN_TREE_DIRTY")

    def test_no_log_is_unobserved_not_refuted(self):
        # git-only rungs pass, but behavioral rungs have no evidence
        inp = self._proven_inputs()
        inp["git_cmds"] = None
        g = pc.grade(**inp)
        self.assertFalse(g["proven"])
        self.assertEqual(g["verdict"], "PARITY_UNOBSERVED")
        self.assertEqual(g["failed_rung"], "by_pathspec")
        self.assertIsNone(g["rungs"]["by_pathspec"])

    def test_unknown_lane_tree_is_unobserved(self):
        inp = self._proven_inputs()
        inp["lane_tree_clean"] = None
        g = pc.grade(**inp)
        self.assertFalse(g["proven"])
        self.assertEqual(g["verdict"], "PARITY_UNOBSERVED")
        self.assertEqual(g["failed_rung"], "lane_tree_clean")

    def test_first_failure_wins_ordering(self):
        # both unwitnessed AND unbound — the earlier rung (witnessed) reports
        inp = self._proven_inputs()
        inp["witnessed"] = False
        inp["issue"] = 999
        g = pc.grade(**inp)
        self.assertEqual(g["verdict"], "PARITY_UNPROVEN_UNWITNESSED")


class BarShape(unittest.TestCase):
    def test_bar_rungs_unique_and_typed(self):
        rungs = [r for r, _ in pc.PARITY_BAR]
        tokens = [t for _, t in pc.PARITY_BAR]
        self.assertEqual(len(rungs), len(set(rungs)), "rung names must be unique")
        self.assertEqual(len(tokens), len(set(tokens)), "verdict tokens must be unique")
        for t in tokens:
            self.assertTrue(t.startswith("PARITY_UNPROVEN_"))


if __name__ == "__main__":
    unittest.main()
