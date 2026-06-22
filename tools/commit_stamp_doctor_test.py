#!/usr/bin/env python3
"""Tests for commit_stamp_doctor — the per-commit DOS ship-stamp coverage report.

Exercise the PURE surface (classify / stamp_leaf / declared_lanes) with synthetic
subjects, so the recognition rules — what counts as a bindable trailer vs a
referee-blind subject, and which leaf tokens are off-lane — are proven without
depending on live git history."""
import tempfile
import unittest
from pathlib import Path

import commit_stamp_doctor as csd


class Classify(unittest.TestCase):
    def test_trailer_is_stamped(self):
        self.assertEqual(csd.classify("fix(gateway): harden (fak gateway)"), "stamped-trailer")
        self.assertEqual(csd.classify("feat(x): y (refs fak tools)"), "stamped-trailer")
        self.assertEqual(csd.classify("feat(x): y (fak: cachemeta)"), "stamped-trailer")

    def test_direct_prefix_is_stamped(self):
        self.assertEqual(csd.classify("fak/architest: registration gate"), "stamped-direct")

    def test_release_and_bookkeeping(self):
        self.assertEqual(csd.classify("v0.23.0: DGX fak-test packet"), "release")
        self.assertEqual(csd.classify("Merge branch 'origin/master'"), "bookkeeping")
        self.assertEqual(csd.classify("working-dir snapshot: bulk sweep"), "bookkeeping")

    def test_bare_conventional_is_unstamped(self):
        # The whole point: a substantive CC commit with no trailer is referee-blind.
        self.assertEqual(csd.classify("fix(gateway): treat same-tick ready"), "unstamped")
        self.assertEqual(csd.classify("feat(fak): GLM-5.2 DSA plane"), "unstamped")

    def test_bookkeeping_beats_a_trailing_paren(self):
        # A snapshot that happens to end in a paren is still bookkeeping, never a ship.
        self.assertEqual(csd.classify("working-dir snapshot: x (fak tools)"), "bookkeeping")


class StampLeaf(unittest.TestCase):
    def test_extracts_trailer_and_direct(self):
        self.assertEqual(csd.stamp_leaf("fix(x): y (fak gateway)"), "gateway")
        self.assertEqual(csd.stamp_leaf("feat(x): y (refs fak tools)"), "tools")
        self.assertEqual(csd.stamp_leaf("fak/architest: z"), "architest")

    def test_none_when_unstamped(self):
        self.assertIsNone(csd.stamp_leaf("fix(gateway): no trailer here"))
        self.assertIsNone(csd.stamp_leaf("v0.23.0: release"))


class DeclaredLanes(unittest.TestCase):
    def test_reads_lanes_from_toml(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "dos.toml").write_text(
                '[lanes]\nconcurrent = ["gateway", "tools"]\nexclusive = ["abi"]\n'
                '[lanes.trees]\ngateway = ["x/**"]\ncachemeta = ["y/**"]\n',
                encoding="utf-8")
            lanes = csd.declared_lanes(d)
        # concurrent ∪ exclusive ∪ the [lanes.trees] keys, lowercased.
        self.assertEqual(lanes, {"gateway", "tools", "abi", "cachemeta"})

    def test_unreadable_toml_is_empty_not_error(self):
        with tempfile.TemporaryDirectory() as d:
            # No dos.toml in the dir → empty set → the off-lane check is SKIPPED, not failed.
            self.assertEqual(csd.declared_lanes(d), set())

    def test_off_lane_leaf_detected(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "dos.toml").write_text(
                '[lanes]\nconcurrent = ["gateway"]\n', encoding="utf-8")
            lanes = csd.declared_lanes(d)
        # A typo'd leaf is exactly what off-lane detection flags.
        self.assertIn("gateway", lanes)
        self.assertNotIn(csd.stamp_leaf("fix(x): y (fak gatway)"), lanes)


class CmdDemoLeaves(unittest.TestCase):
    """#518: a stamp leaf naming a real `cmd/<name>/` dir is a recognized cmd-lane
    demo, NOT an off-lane typo — so the residual off-lane list means real typos."""

    def test_reads_cmd_subdir_names(self):
        with tempfile.TemporaryDirectory() as d:
            cmd = Path(d) / "cmd"
            (cmd / "turntaxdemo").mkdir(parents=True)
            (cmd / "guarddemo").mkdir()
            (cmd / "README.md").write_text("not a dir", encoding="utf-8")  # files ignored
            demos = csd.cmd_demo_leaves(d)
        self.assertEqual(demos, {"turntaxdemo", "guarddemo"})

    def test_missing_cmd_dir_is_empty_not_error(self):
        with tempfile.TemporaryDirectory() as d:
            # No cmd/ dir → empty set → recognition just falls back to declared lanes.
            self.assertEqual(csd.cmd_demo_leaves(d), set())

    def test_cmd_demo_recognized_real_typo_is_not(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "cmd" / "turntaxdemo").mkdir(parents=True)
            (Path(d) / "dos.toml").write_text(
                '[lanes]\nconcurrent = ["gateway"]\n', encoding="utf-8")
            recognized = csd.declared_lanes(d) | csd.cmd_demo_leaves(d)
        # The cmd/ demo binds (recognized); a genuine typo still does not.
        self.assertIn(csd.stamp_leaf("test(turntaxdemo): add (fak turntaxdemo)"), recognized)
        self.assertNotIn(csd.stamp_leaf("fix(x): y (fak gatway)"), recognized)


if __name__ == "__main__":
    unittest.main()
