#!/usr/bin/env python3
"""Hermetic tests for tools/new_leaf.py — the conforming-leaf scaffolder.

new_leaf.py is load-bearing DOS governance: it writes the new leaf's dos.toml
concurrency lane (add_leaf_lane), its architest tier, and its registrations entry.
A regression in the lane emitter silently reintroduces the stale `fak/` path prefix
that the 2026-06-22 dos-effective-usage audit fixed (a glob matching zero files ->
the arbiter can't detect a same-tree collision). These tests pin the pure text
transforms; no filesystem leaf is created.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "new_leaf.py"


def load():
    spec = importlib.util.spec_from_file_location("new_leaf", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()

_DOS_TOML = (
    "[lanes]\n"
    "concurrent = [\n"
    '  "foo",\n'
    "  # new-leaf:lane\n"
    "]\n"
    "autopick = [\n"
    '  "foo",\n'
    "  # new-leaf:lane\n"
    "]\n"
    "[lanes.trees]\n"
    'foo = ["internal/foo/**"]\n'
    "# new-leaf:tree\n"
    'cmd = ["cmd/**"]\n'
)


class GeneratorTest(unittest.TestCase):
    def test_doc_go_carries_tier_and_package(self):
        out = m.doc_go("fedtrust", "composer", 3, "a federated trust gate")
        self.assertIn("package fedtrust", out)
        self.assertIn("Tier: composer (3)", out)
        self.assertIn("import only", out)

    def test_impl_go_register_adds_abi_and_init(self):
        reg = m.impl_go("fedtrust", True)
        self.assertIn('import "github.com/anthony-chaudhary/fak/internal/abi"', reg)
        self.assertIn("func init()", reg)
        plain = m.impl_go("fedtrust", False)
        self.assertNotIn("abi", plain)
        self.assertIn("func Ready() bool", plain)

    def test_test_go_has_ready_test(self):
        self.assertIn("func TestReady", m.test_go("fedtrust"))


class MarkerInsertTest(unittest.TestCase):
    def test_insert_before_marker(self):
        text = "a\n# MARK\nb\n"
        out = m.insert_before_marker(text, "# MARK", "X\n")
        self.assertEqual(out, "a\nX\n# MARK\nb\n")

    def test_insert_before_marker_missing_raises(self):
        with self.assertRaises(ValueError):
            m.insert_before_marker("a\nb\n", "# NOPE", "X\n")

    def test_insert_before_all_markers_hits_every(self):
        text = "# M\nx\n# M\n"
        out = m.insert_before_all_markers(text, "# M", "Y\n")
        self.assertEqual(out.count("Y\n"), 2)

    def test_add_registration_idempotent(self):
        reg = 'import (\n\t_ "x/foo"\n)\n'
        once = m.add_registration(reg, "bar")
        self.assertIn('_ "github.com/anthony-chaudhary/fak/internal/bar"', once)
        self.assertEqual(m.add_registration(once, "bar"), once)  # idempotent


class LeafLaneTest(unittest.TestCase):
    def test_add_leaf_lane_real_layout_not_fak_prefix(self):
        out = m.add_leaf_lane(_DOS_TOML, "bar")
        # tree uses the REAL repo-root layout, never the stale fak/ prefix.
        self.assertIn('bar = ["internal/bar/**"]', out)
        self.assertNotIn("fak/internal/bar", out)
        # name added to BOTH the concurrent and autopick arrays (two lane markers).
        self.assertEqual(out.count('  "bar",\n'), 2)

    def test_add_leaf_lane_idempotent(self):
        once = m.add_leaf_lane(_DOS_TOML, "bar")
        self.assertEqual(m.add_leaf_lane(once, "bar"), once)

    def test_add_leaf_lane_legacy_prefix_treated_as_present(self):
        legacy = _DOS_TOML + 'baz = ["fak/internal/baz/**"]\n'
        # an existing legacy entry must be seen as already-present (no double add).
        self.assertEqual(m.add_leaf_lane(legacy, "baz"), legacy)


class ConstantsTest(unittest.TestCase):
    def test_tiers_and_name_re(self):
        self.assertEqual(m.TIERS["foundation"], 1)
        self.assertTrue(m.NAME_RE.match("fedtrust"))
        self.assertFalse(m.NAME_RE.match("Fed_Trust"))  # uppercase/underscore rejected


if __name__ == "__main__":
    unittest.main()
