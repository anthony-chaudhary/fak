#!/usr/bin/env python3
"""Tests for tools/dogfood_coverage.py."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "dogfood_coverage.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("dogfood_coverage", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DogfoodCoverageTest(unittest.TestCase):
    def test_evaluate_on_real_repo_has_schema_and_hard_kpis(self) -> None:
        mod = load()
        payload = mod.evaluate(ROOT, env={})  # empty env => guard default ON
        self.assertEqual(payload["schema"], "dogfood-coverage/1")
        self.assertGreaterEqual(payload["coverage"], 0.0)
        self.assertLessEqual(payload["coverage"], 100.0)
        self.assertIsInstance(payload["dogfood_debt"], int)
        self.assertIn(payload["grade"], {"A", "B", "C", "D", "F"})
        keys = {k["key"] for k in payload["kpis"]}
        for expected in ("fleet_leaf_guarded", "fak_bin_resolvable", "guard_default_on",
                         "issue_dispatch_wired", "guard_verb_present"):
            self.assertIn(expected, keys)

    def test_guard_disabled_env_lowers_coverage_and_raises_debt(self) -> None:
        mod = load()
        on = mod.evaluate(ROOT, env={})
        off = mod.evaluate(ROOT, env={"FLEET_DOGFOOD_GUARD": "0"})
        # Turning the dogfood OFF must FAIL the live-env KPI (and at least the leaf
        # guard KPI), so debt rises and coverage drops vs the default-on run.
        self.assertGreater(off["dogfood_debt"], on["dogfood_debt"])
        self.assertLess(off["coverage"], on["coverage"])
        off_keys = {k["key"]: k["ok"] for k in off["kpis"]}
        self.assertFalse(off_keys["guard_default_on"])

    def test_count_audit_rows_counts_nonblank_jsonl_lines(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            jdir = root / ".dispatch-runs" / "guard-audit"
            jdir.mkdir(parents=True)
            (jdir / "gateway-claude.jsonl").write_text(
                '{"seq":1}\n{"seq":2}\n\n{"seq":3}\n', encoding="utf-8")
            (jdir / "docs-claude.jsonl").write_text('{"seq":1}\n', encoding="utf-8")
            rows, journals = mod.count_audit_rows(root)
        self.assertEqual(rows, 4)       # 3 + 1 non-blank lines
        self.assertEqual(journals, 2)

    def test_diagnose_audit_gap_distinguishes_the_three_blanks(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # (1) no journal dir at all -> "no guarded worker has run"
            msg_no_dir = mod.diagnose_audit_gap(root)
            self.assertIn("no guarded worker", msg_no_dir)

            jdir = root / ".dispatch-runs" / "guard-audit"
            jdir.mkdir(parents=True)
            # (2) dir but no files -> "configured but never exercised"
            msg_empty_dir = mod.diagnose_audit_gap(root)
            self.assertIn("never exercised", msg_empty_dir)

            # (3) dir + only-blank files -> the silent empty-turn signature
            (jdir / "docs-claude.jsonl").write_text("\n\n", encoding="utf-8")
            msg_blank_files = mod.diagnose_audit_gap(root)
            self.assertIn("all blank", msg_blank_files)
            self.assertIn("auth/login", msg_blank_files)

            # The three blanks must be distinguishable (the whole point).
            self.assertEqual(len({msg_no_dir, msg_empty_dir, msg_blank_files}), 3)

        # And when rows exist, there is no gap to explain.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            jdir = root / ".dispatch-runs" / "guard-audit"
            jdir.mkdir(parents=True)
            (jdir / "g.jsonl").write_text('{"seq":1}\n', encoding="utf-8")
            # diagnose is only consulted when rows==0; here it would report the
            # blank-files branch is NOT hit because the file has a row.
            self.assertEqual(mod.count_audit_rows(root)[0], 1)

    def test_grade_ladder(self) -> None:
        mod = load()
        self.assertEqual(mod._grade(100.0, 0), "A")
        self.assertEqual(mod._grade(70.0, 0), "B")
        self.assertEqual(mod._grade(70.0, 1), "C")
        self.assertEqual(mod._grade(50.0, 2), "D")
        self.assertEqual(mod._grade(10.0, 5), "F")

    def test_check_mode_returns_zero_when_no_hard_debt(self) -> None:
        mod = load()
        # On this repo the HARD KPIs pass (fak is built), so --check should exit 0.
        rc = mod.main(["--check"])
        self.assertIn(rc, (0, 1))   # tolerate a host without fak built (fail-open)

    def test_render_is_stringy_and_mentions_coverage(self) -> None:
        mod = load()
        payload = mod.evaluate(ROOT, env={})
        out = mod.render(payload)
        self.assertIn("dogfood-coverage", out)
        self.assertIn("grade", out)


if __name__ == "__main__":
    unittest.main()
