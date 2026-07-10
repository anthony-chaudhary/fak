#!/usr/bin/env python3
"""Tests for commit_quality_scorecard — the folded commit-output card.

Each KPI gets a defect fixture AND a clean fixture; the fold to commit_debt/grade is
pinned; and a live smoke asserts the real repo yields a well-formed control-pane
envelope (the regression sentinel). Fixtures inject the (sha, subject) rows so the
pure KPIs are exercised without touching git.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import commit_quality_scorecard as cq


def payload(rows, *, root=".", last=50):
    return cq.build_payload(root=root, rows=rows, last=last, sha="deadbee")


class SubjectGradeableKPI(unittest.TestCase):
    def test_verb_led_subject_is_gradeable_clean(self):
        rows = [("a1", "feat(model): add MLA forward seam (fak model)"),
                ("a2", "fix(gateway): correct ready race (fak gateway)")]
        k = cq.kpi_subject_gradeable(rows)
        self.assertEqual(k["defects"], [])
        self.assertEqual(k["score"], 100.0)

    def test_noun_led_subject_abstains_is_defect(self):
        rows = [("b1", "feat(model): MiniMax witness parity (fak model)")]
        k = cq.kpi_subject_gradeable(rows)
        self.assertEqual(len(k["defects"]), 1)
        self.assertIn("b1", k["defects"][0])
        self.assertLess(k["score"], 100.0)

    def test_merge_and_bump_exempt_from_denominator(self):
        rows = [("c1", "Merge branch 'x'"), ("c2", "v0.37.0: cut release")]
        k = cq.kpi_subject_gradeable(rows)
        self.assertEqual(k["defects"], [])
        self.assertEqual(k["score"], 100.0)  # empty denominator -> clean


class ShipStampedKPI(unittest.TestCase):
    def test_trailer_stamp_is_clean(self):
        rows = [("d1", "feat(gateway): add ready gate (fak gateway)")]
        k = cq.kpi_ship_stamped(rows)
        self.assertEqual(k["defects"], [])
        self.assertEqual(k["score"], 100.0)

    def test_unstamped_ship_is_defect(self):
        rows = [("e1", "feat(gateway): add ready gate")]  # no (fak <leaf>)
        k = cq.kpi_ship_stamped(rows)
        self.assertEqual(len(k["defects"]), 1)
        self.assertIn("e1", k["defects"][0])
        self.assertEqual(k["score"], 0.0)

    def test_bookkeeping_and_release_exempt(self):
        rows = [("f1", "Merge branch 'x'"),
                ("f2", "v1.2.3: release"),
                ("f3", "docs/_plans: snapshot")]
        k = cq.kpi_ship_stamped(rows)
        self.assertEqual(k["defects"], [])
        self.assertEqual(k["score"], 100.0)


class StampOnLaneKPI(unittest.TestCase):
    def test_unreadable_dos_toml_skips_check(self):
        # A dir with no dos.toml -> declared_lanes() empty -> check SKIPPED, score 100.
        rows = [("g1", "feat(x): add thing (fak totally-made-up-leaf)")]
        k = cq.kpi_stamp_on_lane(rows, Path("/no/such/dir"))
        self.assertEqual(k["defects"], [])
        self.assertEqual(k["score"], 100.0)

    def test_off_lane_stamp_is_defect_on_real_repo(self):
        # Real repo dos.toml is readable, so a phantom leaf is flagged; a real lane
        # ('docs' is a declared lane) is not.
        root = cq.repo_root()
        bad = cq.kpi_stamp_on_lane([("h1", "docs(x): update (fak zzz-phantom-leaf)")], root)
        good = cq.kpi_stamp_on_lane([("h2", "docs(x): update (fak docs)")], root)
        self.assertEqual(len(bad["defects"]), 1)
        self.assertEqual(good["defects"], [])

    def test_internal_pkg_stamp_is_recognized_on_real_repo(self):
        # Drift sentinel for the internal_leaves rung: a stamp naming a REAL
        # internal/<leaf> package binds through the lane whose tree covers
        # internal/<leaf>/** and must NOT read as a phantom off-lane defect — the
        # dominant valid stamp shape. Without internal_leaves() in the recognized
        # set nearly every package-scoped ship would be a false positive, so this
        # locks the rung against a silent drop. Leaf is picked from the live tree,
        # not hardcoded, so a package rename can't rot the test into a false pass.
        root = cq.repo_root()
        internal = cq._stampdoc.internal_leaves(str(root))
        if not internal:
            self.skipTest("no internal/<leaf> packages in this checkout")
        leaf = sorted(internal)[0]
        k = cq.kpi_stamp_on_lane([("j1", f"feat(x): touch it (fak {leaf})")], root)
        self.assertEqual(k["defects"], [], f"real internal leaf '{leaf}' misread as off-lane")


class Fold(unittest.TestCase):
    def test_clean_window_zero_debt_grade_a(self):
        rows = [("i1", "feat(gateway): add ready gate (fak gateway)"),
                ("i2", "fix(model): correct rope (fak model)"),
                ("i3", "Merge branch 'x'")]
        p = payload(rows)
        self.assertEqual(p["corpus"]["commit_debt"], 0)
        self.assertEqual(p["corpus"]["grade"], "A")
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "OK")

    def test_defects_sum_into_commit_debt(self):
        rows = [("j1", "feat(model): MiniMax witness parity (fak model)"),  # abstain subj
                ("j2", "feat(gateway): add ready gate")]                      # unstamped
        p = payload(rows)
        # j1: 1 abstain. j2: unstamped (1). j1 IS stamped+gradeable-verb? 'MiniMax' noun
        # -> abstain, and it carries (fak model) so stamped. j2 verb-led+unstamped.
        self.assertGreaterEqual(p["corpus"]["commit_debt"], 2)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "ACTION")
        self.assertIn("commit_debt", p["reason"])

    def test_envelope_has_control_pane_keys(self):
        p = payload([("k1", "feat(x): add y (fak docs)")])
        for key in ("schema", "ok", "verdict", "corpus", "kpis"):
            self.assertIn(key, p)
        self.assertIn("commit_debt", p["corpus"])
        self.assertIn("grade", p["corpus"])


class CompareDelta(unittest.TestCase):
    def test_compare_reports_2x_drop(self):
        cur = {"corpus": {"commit_debt": 2}}
        base = {"corpus": {"commit_debt": 4}}
        bp = Path(cq.__file__).parent / "_cq_test_baseline.json"
        bp.write_text(json.dumps(base), encoding="utf-8")
        try:
            out = cq.compare(cur, str(bp))
            self.assertIn("4 -> 2", out)
            self.assertIn("2x", out)
        finally:
            bp.unlink(missing_ok=True)

    def test_compare_flags_regression(self):
        cur = {"corpus": {"commit_debt": 5}}
        base = {"corpus": {"commit_debt": 3}}
        bp = Path(cq.__file__).parent / "_cq_test_baseline2.json"
        bp.write_text(json.dumps(base), encoding="utf-8")
        try:
            self.assertIn("REGRESSED", cq.compare(cur, str(bp)))
        finally:
            bp.unlink(missing_ok=True)


class LiveSmoke(unittest.TestCase):
    """The real repo must yield a well-formed envelope — the regression sentinel.
    Does NOT pin an exact debt (history slides as HEAD advances); pins structure and
    invariants only."""

    def test_real_repo_envelope_well_formed(self):
        p = cq.collect(cq.repo_root(), last=30)
        self.assertEqual(p["schema"], cq.SCHEMA)
        self.assertIn(p["verdict"], ("OK", "ACTION"))
        c = p["corpus"]
        self.assertGreaterEqual(c["commit_debt"], 0)
        self.assertIn(c["grade"], ("A", "B", "C", "D", "F"))
        self.assertEqual(len(p["kpis"]), 3)
        # debt is exactly the sum of per-KPI defects (fold integrity)
        self.assertEqual(c["commit_debt"], sum(len(k["defects"]) for k in p["kpis"]))
        # every KPI score is a percentage
        for k in p["kpis"]:
            self.assertGreaterEqual(k["score"], 0.0)
            self.assertLessEqual(k["score"], 100.0)


if __name__ == "__main__":
    unittest.main()
