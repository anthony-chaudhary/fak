#!/usr/bin/env python3
"""Hermetic tests for tools/commit_subject_coverage.py.

The git-log read is seamed; the fold reuses check_commit_msg.verdict (the real
verb set) so the coverage number tracks the same rule the commit-msg hook
enforces. No git / gh / dos is touched.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "commit_subject_coverage.py"

# A representative window: 2 gradeable, 1 abstain (noun-led), 2 exempt.
SUBJECTS = [
    "fix(tools): add a noun-form rung to the closure auditor (fak tools)",  # gradeable
    "feat(model): MiniMax-M3 witness oracle (fak model)",                   # ABSTAIN (noun-led)
    "Merge branch 'main' of origin",                                        # exempt (merge)
    "v0.32.0: cut the release",                                             # exempt (version bump)
    "ci: add the dispatch tool-test cluster to the gate (fak ci)",          # gradeable
]


def load():
    spec = importlib.util.spec_from_file_location("commit_subject_coverage", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()


class ExemptTest(unittest.TestCase):
    def test_merge_revert_version_exempt(self):
        self.assertTrue(m.is_exempt("Merge branch 'main'"))
        self.assertTrue(m.is_exempt("Revert \"feat: x\""))
        self.assertTrue(m.is_exempt("v1.2.3: ship it"))

    def test_real_ship_not_exempt(self):
        self.assertFalse(m.is_exempt("fix(tools): add a rung"))


class CoverageFoldTest(unittest.TestCase):
    def test_counts_and_fraction(self):
        cov = m.coverage(SUBJECTS)
        self.assertEqual(cov["total"], 3)        # 2 exempt dropped from denominator
        self.assertEqual(cov["gradeable"], 2)
        self.assertEqual(cov["abstain"], 1)
        self.assertAlmostEqual(cov["coverage"], 2 / 3, places=3)

    def test_abstain_subject_is_the_noun_led_one(self):
        cov = m.coverage(SUBJECTS)
        subs = [a["subject"] for a in cov["abstain_subjects"]]
        self.assertEqual(len(subs), 1)
        self.assertIn("MiniMax", subs[0])

    def test_empty_window(self):
        cov = m.coverage(["Merge x", "v1.0.0: bump"])
        self.assertEqual(cov["total"], 0)
        self.assertIsNone(cov["coverage"])

    def test_reuses_check_commit_msg_verb_set(self):
        # 'bind' is deliberately NOT in the recognized verb set -> ABSTAIN.
        cov = m.coverage(["fix(tools): bind the convention"])
        self.assertEqual(cov["abstain"], 1)


class PayloadTest(unittest.TestCase):
    def test_below_floor_not_ok(self):
        p = m.build_payload(root="r", cov=m.coverage(SUBJECTS), min_coverage=80.0)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "BELOW_FLOOR")

    def test_above_floor_ok(self):
        p = m.build_payload(root="r", cov=m.coverage(SUBJECTS), min_coverage=50.0)
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "OK")

    def test_no_floor_ok(self):
        p = m.build_payload(root="r", cov=m.coverage(SUBJECTS), min_coverage=None)
        self.assertTrue(p["ok"])
        self.assertEqual(p["coverage_pct"], round(2 / 3 * 100, 1))

    def test_no_gradeable_commits(self):
        p = m.build_payload(root="r", cov=m.coverage(["Merge x"]), min_coverage=80.0)
        self.assertTrue(p["ok"])  # nothing to grade is not a failure
        self.assertEqual(p["verdict"], "NO_GRADEABLE_COMMITS")


class CollectWiringTest(unittest.TestCase):
    def test_collect_uses_injected_fetcher(self):
        p = m.collect(Path("r"), last=5, min_coverage=80.0,
                      fetcher=lambda root, n: SUBJECTS)
        self.assertEqual(p["verdict"], "BELOW_FLOOR")
        self.assertEqual(p["total"], 3)

    def test_main_exit_codes(self):
        orig = m.recent_subjects
        m.recent_subjects = lambda root, n: SUBJECTS
        try:
            self.assertEqual(m.main(["--min-coverage", "80", "--json"]), 1)  # below floor
            self.assertEqual(m.main(["--min-coverage", "50", "--json"]), 0)  # above floor
            self.assertEqual(m.main(["--json"]), 0)                          # no floor
        finally:
            m.recent_subjects = orig


if __name__ == "__main__":
    unittest.main()
