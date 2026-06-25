#!/usr/bin/env python3
"""Hermetic tests for tools/gated_tool_tests.py (the no-blackhole tool-test runner).

Covers the pure logic (ci.yml invocation parsing, partition totality, the check()
manifest-sanity rules, the run() aggregation) plus one self-gate: the COMMITTED
manifest over the REAL tree must be consistent (check() ok) -- so a stale quarantine
entry, a bad reason, or a quarantined-but-gated contradiction reds CI on the commit
that introduces it. No network / no gh / no dos."""
from __future__ import annotations

import sys
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gated_tool_tests as M  # noqa: E402


class GatedInCI(unittest.TestCase):
    def test_matches_invocations_not_comments(self):
        ci = textwrap.dedent("""
            steps:
              - run: |
                  python tools/foo_test.py
                  python3 tools/bar_test.py
              # a comment mentioning tools/ghost_test.py should NOT count as gated
              - run: python tools/release_cut.py --json   # not a *_test.py
        """)
        got = M.gated_in_ci(ci)
        self.assertIn("foo_test.py", got)
        self.assertIn("bar_test.py", got)  # python3 form
        self.assertNotIn("ghost_test.py", got)  # bare comment mention
        self.assertNotIn("release_cut.py", got)


class Partition(unittest.TestCase):
    def test_total_and_disjoint(self):
        all_tests = ["a_test.py", "b_test.py", "c_test.py", "d_test.py"]
        gated = {"a_test.py"}
        # patch QUARANTINE to {b: red}
        old = dict(M.QUARANTINE)
        try:
            M.QUARANTINE.clear()
            M.QUARANTINE["b_test.py"] = "red"
            gated_here, quar, herm = M.partition(all_tests, gated)
            self.assertEqual(gated_here, ["a_test.py"])
            self.assertEqual(quar, {"b_test.py": "red"})
            self.assertEqual(herm, ["c_test.py", "d_test.py"])  # the default remainder
            # every test in exactly one bucket
            buckets = set(gated_here) | set(quar) | set(herm)
            self.assertEqual(buckets, set(all_tests))
            self.assertEqual(len(gated_here) + len(quar) + len(herm), len(all_tests))
        finally:
            M.QUARANTINE.clear()
            M.QUARANTINE.update(old)


class CheckRules(unittest.TestCase):
    def _check_with(self, quarantine, root):
        old = dict(M.QUARANTINE)
        try:
            M.QUARANTINE.clear()
            M.QUARANTINE.update(quarantine)
            return M.check(root)
        finally:
            M.QUARANTINE.clear()
            M.QUARANTINE.update(old)

    def _tmp_tree(self, tmp, tests, ci_text=""):
        (tmp / "tools").mkdir(parents=True, exist_ok=True)
        for name in tests:
            (tmp / "tools" / name).write_text("raise SystemExit(0)\n")
        wf = tmp / ".github" / "workflows"
        wf.mkdir(parents=True, exist_ok=True)
        (wf / "ci.yml").write_text(ci_text)

    def test_flags_stale_entry(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._tmp_tree(tmp, ["a_test.py"])
            rc, p = self._check_with({"gone_test.py": "red"}, tmp)
            self.assertEqual(rc, 1)
            self.assertFalse(p["ok"])
            self.assertTrue(any("stale" in x for x in p["problems"]))

    def test_flags_bad_reason(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._tmp_tree(tmp, ["a_test.py"])
            rc, p = self._check_with({"a_test.py": "not-a-real-reason"}, tmp)
            self.assertEqual(rc, 1)
            self.assertTrue(any("closed vocab" in x for x in p["problems"]))

    def test_flags_contradiction_quarantined_and_gated(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._tmp_tree(tmp, ["a_test.py"], ci_text="run: python tools/a_test.py\n")
            rc, p = self._check_with({"a_test.py": "red"}, tmp)
            self.assertEqual(rc, 1)
            self.assertTrue(any("contradiction" in x for x in p["problems"]))

    def test_clean_tree_ok_and_red_debt_counted(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._tmp_tree(tmp, ["a_test.py", "b_test.py"])
            rc, p = self._check_with({"b_test.py": "red"}, tmp)
            self.assertEqual(rc, 0)
            self.assertTrue(p["ok"])
            self.assertEqual(p["red_debt"], 1)
            self.assertEqual(p["red_debt_tests"], ["b_test.py"])
            self.assertEqual(p["hermetic"], 1)  # a_test.py


class RunAggregation(unittest.TestCase):
    def test_runs_all_and_reports_every_failure(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / "tools").mkdir(parents=True)
            (tmp / "tools" / "pass_test.py").write_text("raise SystemExit(0)\n")
            (tmp / "tools" / "fail1_test.py").write_text("raise SystemExit(1)\n")
            (tmp / "tools" / "fail2_test.py").write_text("raise SystemExit(2)\n")
            rc, p = M.run(tmp, jobs=3, include_gated=False)
            self.assertEqual(rc, 1)
            self.assertEqual(p["ran"], 3)
            # anti-cascade: BOTH failures reported, not just the first
            failed = {f["test"] for f in p["failures"]}
            self.assertEqual(failed, {"fail1_test.py", "fail2_test.py"})


class CommittedManifestSelfGate(unittest.TestCase):
    """The real, committed manifest over the real tree must be self-consistent."""

    def test_real_tree_check_is_ok(self):
        rc, p = M.check(M.repo_root())
        self.assertTrue(p["ok"], f"committed manifest inconsistent: {p['problems']}")
        self.assertEqual(rc, 0)

    def test_every_quarantine_reason_in_closed_vocab(self):
        for name, reason in M.QUARANTINE.items():
            self.assertIn(reason, M.QUARANTINE_REASONS, f"{name} has off-vocab reason {reason!r}")


if __name__ == "__main__":
    unittest.main()
