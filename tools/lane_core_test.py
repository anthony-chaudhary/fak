#!/usr/bin/env python3
"""Hermetic tests for tools/lane_core.py and the value-weighted lane pick in
tools/issue_resolve_dispatch.py::lane_issue_numbers.

Two layers:
  1. The core-source / trust-critical predicates -- the Python mirror of
     internal/dispatchtick/selfmodify.go. The predicate cases MIRROR the Go
     TestIsCoreSourceLaneTree table so a drift between the two dispatch paths
     trips here.
  2. The pick: with the router fold and issue_triage priority INJECTED (no
     subprocess, no gh), assert the guarded pick (a) narrows the hold to the
     trust-critical set only so a CORE lane is dispatchable, (b) ranks core >
     docs/tools, (c) ranks by issue_triage priority within a class, and (d)
     fails open to busiest-by-count when triage yields nothing.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lane_core = _load("lane_core")
ird = _load("issue_resolve_dispatch")


class CorePredicateTest(unittest.TestCase):
    """Mirror of selfmodify.go TestIsCoreSourceLaneTree -- keep in lockstep."""

    def test_is_core_source_lane_tree(self):
        cases = [
            ("gateway leaf is core", ["internal/gateway/**"], True),
            ("agent leaf is core", ["internal/agent/**"], True),
            ("cmd shim is core", ["cmd/fak/**"], True),
            ("module-prefixed core", ["fak/internal/compute/**"], True),
            ("windows-authored core glob", ["internal\\gateway\\**"], True),
            ("docs bucket is not core", ["docs/**"], False),
            ("tools bucket is not core", ["tools/**"], False),
            ("trust-critical kernel is not core", ["internal/kernel/**"], False),
            ("trust-critical adjudicator is not core", ["internal/adjudicator/**"], False),
            ("trust-critical file glob is not core", ["dos.toml"], False),
            ("mixed core+docs is not core", ["internal/gateway/**", "docs/**"], False),
            ("mixed core+trust is not core", ["internal/gateway/**", "internal/abi/**"], False),
            ("empty tree is not core", None, False),
            ("blank glob is not core", ["   "], False),
        ]
        for name, tree, want in cases:
            with self.subTest(name):
                self.assertEqual(lane_core.is_core_source_lane_tree(tree), want)

    def test_is_trust_critical_tree(self):
        for glob in ("internal/kernel/**", "internal/adjudicator/foo", "dos.toml",
                     ".dos/markers", "policy.json", "VERSION", "fak/internal/policy/x",
                     "internal\\shipgate\\**"):
            self.assertTrue(lane_core.is_trust_critical_tree(glob), glob)
        for glob in ("internal/gateway/**", "cmd/fak/**", "docs/**", "tools/**", ""):
            self.assertFalse(lane_core.is_trust_critical_tree(glob), glob)

    def test_lane_dispatchable_under_guard(self):
        # Guarded: core + docs dispatchable, trust-critical held.
        self.assertTrue(lane_core.lane_dispatchable_under_guard(True, ["internal/gateway/**"]))
        self.assertTrue(lane_core.lane_dispatchable_under_guard(True, ["docs/**"]))
        self.assertFalse(lane_core.lane_dispatchable_under_guard(True, ["internal/kernel/**"]))
        self.assertFalse(lane_core.lane_dispatchable_under_guard(True, ["dos.toml"]))
        # Unguarded: everything dispatchable (operator/worktree escape).
        self.assertTrue(lane_core.lane_dispatchable_under_guard(False, ["internal/kernel/**"]))
        # No tree: fail-open dispatchable.
        self.assertTrue(lane_core.lane_dispatchable_under_guard(True, None))
        self.assertTrue(lane_core.lane_dispatchable_under_guard(True, []))


def _router(lanes: dict) -> dict:
    """The router-fold shape lane_issue_numbers consumes."""
    return {"lanes": lanes, "issues": [], "_error": None}


class ValueWeightedPickTest(unittest.TestCase):
    """Inject the router + triage so lane_issue_numbers runs with no subprocess."""

    def setUp(self):
        self._run_json = ird.issue_dispatch.run_json
        self._prio = ird.issue_dispatch.lane_priority_scores

    def tearDown(self):
        ird.issue_dispatch.run_json = self._run_json
        ird.issue_dispatch.lane_priority_scores = self._prio

    def _inject(self, router: dict, priority: dict):
        ird.issue_dispatch.run_json = lambda *a, **k: router
        ird.issue_dispatch.lane_priority_scores = lambda root, nums: dict(priority)

    def test_core_beats_buckets_and_hold_is_narrowed(self):
        self._inject(_router({
            "docs":    {"issues": [{"number": 10}, {"number": 11}, {"number": 12}], "tree": ["docs/**"]},
            "tools":   {"issues": [{"number": 20}, {"number": 21}], "tree": ["tools/**"]},
            "gateway": {"issues": [{"number": 30}], "tree": ["internal/gateway/**"]},
            "kernel":  {"issues": [{"number": 40}], "tree": ["internal/kernel/**"]},
        }), priority={})
        r = ird.lane_issue_numbers(Path("."), None, guarded=True)
        # Core gateway (1 issue) beats docs (3) and tools (2) on the value ladder.
        self.assertEqual(r["lane"], "gateway")
        self.assertEqual(r["core_lanes"], ["gateway"])
        # Narrowed hold: ONLY the trust-critical kernel lane is held -- gateway,
        # which the OLD broad self-source hold would have excluded, is dispatchable.
        self.assertEqual(r["self_modify_held"], ["kernel"])
        self.assertIn("gateway", [row[0] for row in r["eligible_by_lane"]])

    def test_priority_beats_volume_within_noncore(self):
        self._inject(_router({
            "docs":  {"issues": [{"number": 10}, {"number": 11}, {"number": 12}], "tree": ["docs/**"]},
            "tools": {"issues": [{"number": 20}, {"number": 21}], "tree": ["tools/**"]},
        }), priority={"tools": 400})  # tools carries a P1; docs is unlabeled ceremony
        r = ird.lane_issue_numbers(Path("."), None, guarded=True)
        # tools (P1=400, 2 issues) beats docs (default 0, 3 issues) -- value > volume.
        self.assertEqual(r["lane"], "tools")
        self.assertEqual(r["lane_priority"].get("tools"), 400)

    def test_fail_open_to_busiest_when_no_core_no_priority(self):
        self._inject(_router({
            "docs":  {"issues": [{"number": 10}, {"number": 11}, {"number": 12}], "tree": ["docs/**"]},
            "tools": {"issues": [{"number": 20}, {"number": 21}], "tree": ["tools/**"]},
        }), priority={})  # triage hiccup -> {}
        r = ird.lane_issue_numbers(Path("."), None, guarded=True)
        # No core, no priority: the ladder degrades to busiest-by-count (docs) --
        # never worse than the historical pick.
        self.assertEqual(r["lane"], "docs")

    def test_unguarded_holds_nothing(self):
        self._inject(_router({
            "kernel":  {"issues": [{"number": 40}], "tree": ["internal/kernel/**"]},
            "gateway": {"issues": [{"number": 30}], "tree": ["internal/gateway/**"]},
        }), priority={})
        r = ird.lane_issue_numbers(Path("."), None, guarded=False)
        # Unguarded: no hold at all (operator/worktree escape #1334).
        self.assertEqual(r["self_modify_held"], [])


if __name__ == "__main__":
    unittest.main()
