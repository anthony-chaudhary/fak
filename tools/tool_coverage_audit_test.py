#!/usr/bin/env python3
"""Hermetic tests for tools/tool_coverage_audit.py.

The pure auditor takes literal module/test/refs inputs; the discovery seams are
exercised against a TemporaryDirectory. No real tree, no test execution.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPT = Path(__file__).resolve().parent / "tool_coverage_audit.py"


def load():
    spec = importlib.util.spec_from_file_location("tool_coverage_audit", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()


class AuditPureTest(unittest.TestCase):
    def test_load_bearing_coverage_and_debt(self):
        # foo: tested + referenced; bar: untested + referenced (DEBT); baz: untested + not referenced.
        a = m.audit(["foo", "bar", "baz"], {"foo"}, "the skill calls foo.py and bar.py here")
        self.assertEqual(a["total_modules"], 3)
        self.assertEqual(a["tested"], 1)
        self.assertEqual(a["load_bearing"], 2)
        self.assertEqual(a["load_bearing_tested"], 1)
        self.assertEqual(a["load_bearing_untested"], ["bar"])
        self.assertEqual(a["debt"], 1)
        self.assertEqual(a["load_bearing_coverage_pct"], 50.0)

    def test_unreferenced_untested_is_not_debt(self):
        a = m.audit(["solo"], set(), "")  # no refs -> not load-bearing -> no debt
        self.assertEqual(a["debt"], 0)
        self.assertEqual(a["load_bearing"], 0)
        self.assertIsNone(a["load_bearing_coverage_pct"])

    def test_substring_does_not_false_match(self):
        # 'bench.py' referenced must NOT mark 'bench_node' as referenced.
        a = m.audit(["bench_node"], set(), "we run bench.py in CI")
        self.assertFalse(a["load_bearing"] > 0)


class PayloadTest(unittest.TestCase):
    def test_below_floor(self):
        a = m.audit(["foo", "bar"], set(), "foo.py bar.py")  # 0% lb coverage
        p = m.build_payload(root="r", a=a, min_coverage=90.0)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "BELOW_FLOOR")
        self.assertIn("bar", p["reason"])

    def test_above_floor(self):
        a = m.audit(["foo"], {"foo"}, "foo.py")  # 100%
        p = m.build_payload(root="r", a=a, min_coverage=90.0)
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "OK")

    def test_no_load_bearing(self):
        a = m.audit(["solo"], set(), "")
        p = m.build_payload(root="r", a=a, min_coverage=90.0)
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "NO_LOAD_BEARING_MODULES")


class DiscoveryTest(unittest.TestCase):
    def test_find_stems_and_collect(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            tools = root / "tools"
            tools.mkdir()
            (tools / "foo.py").write_text("x=1", encoding="utf-8")
            (tools / "foo_test.py").write_text("x=1", encoding="utf-8")
            (tools / "bar.py").write_text("x=1", encoding="utf-8")          # load-bearing + untested
            (tools / "__init__.py").write_text("", encoding="utf-8")        # excluded
            skills = root / ".claude" / "skills" / "s"
            skills.mkdir(parents=True)
            (skills / "SKILL.md").write_text("runs tools/bar.py", encoding="utf-8")
            wf = root / ".github" / "workflows"
            wf.mkdir(parents=True)
            (wf / "ci.yml").write_text("python tools/foo.py", encoding="utf-8")

            self.assertEqual(set(m.find_module_stems(tools)), {"foo", "bar"})  # __init__ excluded
            self.assertEqual(m.find_test_stems(tools), {"foo"})

            p = m.collect(root, min_coverage=90.0)
            # foo referenced by ci + tested; bar referenced by skill + untested -> debt 1, lb 50%.
            self.assertEqual(p["debt"], 1)
            self.assertEqual(p["load_bearing_untested"], ["bar"])
            self.assertEqual(p["load_bearing_coverage_pct"], 50.0)
            self.assertEqual(p["verdict"], "BELOW_FLOOR")

    def test_collect_main_exit_code(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            (root / "tools").mkdir()
            (root / "tools" / "foo.py").write_text("x=1", encoding="utf-8")
            self.assertEqual(m.main(["--workspace", str(root), "--json"]), 0)  # no lb -> ok


if __name__ == "__main__":
    unittest.main()
