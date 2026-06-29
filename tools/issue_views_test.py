#!/usr/bin/env python3
"""Hermetic tests for tools/issue_views.py.

No real gh/network: every assertion is over the pure config-loading + query-
assembly functions, plus a structural check of the shipped
``.github/issue-views.json``. The load() importlib pattern mirrors the sibling
tools (issue_lane_router_test.py).
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "issue_views.py"
SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / ".github" / "issue-views.json"


def load():
    spec = importlib.util.spec_from_file_location("issue_views", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()

GOOD = {
    "repo": "owner/repo",
    "default": "ready",
    "limit": 250,
    "views": [
        {"slug": "ready", "title": "Ready", "query": "is:open -label:epic no:assignee"},
        {"slug": "p1", "title": "P1", "query": "is:open label:priority/P1"},
    ],
}


def _write(tmp: Path, cfg: dict) -> Path:
    p = tmp / "issue-views.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


class ShippedConfig(unittest.TestCase):
    """The config that actually ships must load and be internally consistent."""

    def test_loads_and_is_valid(self):
        cfg = m.load_config(SHIPPED_CONFIG)
        self.assertTrue(cfg["views"])
        vm = m.view_map(cfg)
        # default resolves to a real view
        self.assertIn(cfg["default"], vm)
        # every query is real GitHub search scoped to open issues
        for slug, v in vm.items():
            self.assertIn("is:open", v["query"], f"{slug} query must scope is:open")
            self.assertTrue(v.get("title"), f"{slug} needs a title")

    def test_default_is_ready_leaves(self):
        cfg = m.load_config(SHIPPED_CONFIG)
        self.assertEqual(cfg["default"], "ready-leaves")


class LoadValidation(unittest.TestCase):
    def test_good_config_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = m.load_config(_write(Path(d), GOOD))
            self.assertEqual(cfg["default"], "ready")

    def test_rejects_missing_file(self):
        with self.assertRaises(ValueError):
            m.load_config(Path("/no/such/issue-views.json"))

    def test_rejects_empty_views(self):
        bad = {**GOOD, "views": []}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                m.load_config(_write(Path(d), bad))

    def test_rejects_duplicate_slug(self):
        bad = {**GOOD, "views": [GOOD["views"][0], GOOD["views"][0]]}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                m.load_config(_write(Path(d), bad))

    def test_rejects_default_not_in_views(self):
        bad = {**GOOD, "default": "ghost"}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                m.load_config(_write(Path(d), bad))

    def test_rejects_query_less_view(self):
        bad = {**GOOD, "views": [{"slug": "x", "title": "X"}]}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                m.load_config(_write(Path(d), bad))


class ResolveView(unittest.TestCase):
    def test_none_resolves_default(self):
        self.assertEqual(m.resolve_view(GOOD, None)["slug"], "ready")

    def test_named(self):
        self.assertEqual(m.resolve_view(GOOD, "p1")["slug"], "p1")

    def test_unknown_raises(self):
        with self.assertRaises(KeyError):
            m.resolve_view(GOOD, "nope")


class BuildGhArgs(unittest.TestCase):
    def test_includes_search_limit_repo(self):
        view = m.resolve_view(GOOD, "p1")
        args = m.build_gh_args(GOOD, view)
        self.assertEqual(args[:3], ["gh", "issue", "list"])
        self.assertIn("--repo", args)
        self.assertIn("owner/repo", args)
        self.assertIn("--search", args)
        self.assertIn("is:open label:priority/P1", args)
        # default limit comes from the config
        self.assertIn("250", args)
        # no --json unless asked
        self.assertNotIn("--json", args)

    def test_limit_override_and_json_fields(self):
        view = m.resolve_view(GOOD, "ready")
        args = m.build_gh_args(GOOD, view, limit=10, json_fields="number,title")
        self.assertIn("10", args)
        self.assertIn("--json", args)
        self.assertIn("number,title", args)


if __name__ == "__main__":
    unittest.main()
