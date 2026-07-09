#!/usr/bin/env python3
"""Hermetic tests for tools/refactor_verify.py — the pure `verify` core, driven by strings
(no git, no repo), exactly as godsplit_plan_test.py drives `plan`."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "refactor_verify.py"


def load():
    spec = importlib.util.spec_from_file_location("refactor_verify", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# A god-file before the split: two decls in one file.
GODFILE = (
    "package demo\n"
    "\n"
    "// Alpha does a thing.\n"
    "func Alpha() int { return 1 }\n"
    "\n"
    "// Beta does another.\n"
    "func Beta() int { return 2 }\n"
)


class VerifyTest(unittest.TestCase):
    def setUp(self):
        self.rv = load()

    def test_clean_motion_preserves_decls(self):
        """Beta moved to a sibling file in the SAME package — a pure split, so the diff is
        empty: nothing dropped, nothing relocated, no over-split (Beta's new file is fine as
        long as it is not a *single*-decl fragment... it IS one here, so it is over-split)."""
        before = {"demo": {"demo.go": GODFILE}}
        after = {"demo": {
            "demo.go": "package demo\n\n// Alpha does a thing.\nfunc Alpha() int { return 1 }\n",
            "demo_beta.go": "package demo\n\n// Beta does another.\nfunc Beta() int { return 2 }\n",
        }}
        rep = self.rv.verify(before, after)
        self.assertEqual(rep["dropped"], [])
        self.assertEqual(rep["relocated"], [])
        # demo_beta.go is new and carries exactly one decl → over-split signal fires.
        self.assertEqual([o["file"] for o in rep["oversplit"]], ["demo/demo_beta.go"])

    def test_split_into_cohesive_files_no_oversplit(self):
        """A three-decl god-file split into a 1-decl + 2-decl file: only the 1-decl file is
        flagged. Cohesive multi-decl concern files are NOT over-split."""
        god = GODFILE + "\n// Gamma too.\nfunc Gamma() int { return 3 }\n"
        before = {"demo": {"demo.go": god}}
        after = {"demo": {
            "demo.go": "package demo\n\nfunc Alpha() int { return 1 }\n",  # stays (not new)
            "demo_pair.go": "package demo\n\nfunc Beta() int { return 2 }\n\nfunc Gamma() int { return 3 }\n",
        }}
        rep = self.rv.verify(before, after)
        self.assertEqual(rep["dropped"], [])
        self.assertEqual(rep["oversplit"], [])  # the one new file has TWO decls — cohesive

    def test_dropped_decl_is_the_missing_definition(self):
        """Beta vanished entirely — present before, absent everywhere after. THIS is the
        'god module then missing a definition' failure the tool exists to catch."""
        before = {"demo": {"demo.go": GODFILE}}
        after = {"demo": {"demo.go": "package demo\n\nfunc Alpha() int { return 1 }\n"}}
        rep = self.rv.verify(before, after)
        self.assertEqual(rep["relocated"], [])
        self.assertEqual(len(rep["dropped"]), 1)
        self.assertEqual((rep["dropped"][0]["kind"], rep["dropped"][0]["name"]), ("func", "Beta"))

    def test_relocated_decl_is_not_dropped(self):
        """Beta left package demo and reappeared in package other — a legit cross-package
        consolidation. It is RELOCATED (informational), never DROPPED."""
        before = {"demo": {"demo.go": GODFILE}, "other": {"other.go": "package other\n"}}
        after = {
            "demo": {"demo.go": "package demo\n\nfunc Alpha() int { return 1 }\n"},
            "other": {"other.go": "package other\n\nfunc Beta() int { return 2 }\n"},
        }
        rep = self.rv.verify(before, after)
        self.assertEqual(rep["dropped"], [])
        self.assertEqual(len(rep["relocated"]), 1)
        self.assertEqual(rep["relocated"][0]["name"], "Beta")
        self.assertEqual(rep["relocated"][0]["to"], ["other"])

    def test_type_alias_consolidation_keeps_local_name_quiet(self):
        """The real refactor wave in the tree: a struct is replaced by `type X = pkg.Y`. The
        local NAME X survives as an alias decl, so decl-level verify stays quiet (no false
        DROP) — the honest v1 boundary (field-drop inside Y is the documented follow-on)."""
        before = {"main": {"a.go": "package main\n\ntype guardInfoSession struct {\n\tRun string\n}\n"}}
        after = {"main": {"a.go": "package main\n\ntype guardInfoSession = guardvars.SessionVars\n"}}
        rep = self.rv.verify(before, after)
        self.assertEqual(rep["dropped"], [])
        self.assertEqual(rep["relocated"], [])

    def test_decls_of_reuses_godsplit_fold(self):
        self.assertEqual(self.rv.decls_of(GODFILE), [("func", "Alpha"), ("func", "Beta")])


if __name__ == "__main__":
    unittest.main()
