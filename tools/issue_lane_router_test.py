#!/usr/bin/env python3
"""Hermetic tests for tools/issue_lane_router.py.

No real gh/dos: the router is a pure function over a fixture lane taxonomy +
injected issue dicts. The load() importlib pattern mirrors the sibling tools.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "issue_lane_router.py"


def load():
    spec = importlib.util.spec_from_file_location("issue_lane_router", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()

# A 6-lane subset mirroring dos doctor output (+ the exclusive abi lane).
LANES = ["gateway", "compute", "docs", "tools", "experiments", "model", "abi"]
TREES = {
    "gateway": ["fak/internal/gateway/**"],
    "compute": ["fak/internal/compute/**"],
    "docs": ["docs/**"],
    "tools": ["tools/**"],
    "experiments": ["fak/experiments/**"],
    "model": ["fak/internal/model/**"],
    "abi": ["fak/internal/abi/**"],
}


def issue(number: int, title: str, *, labels=None, body: str = "") -> dict:
    return {"number": number, "title": title, "body": body,
            "labels": [{"name": n} for n in (labels or [])]}


def route(iss: dict) -> dict:
    return m.route_issue(iss, LANES, TREES)


class GlobTest(unittest.TestCase):
    def test_doublestar_matches_nested(self):
        rx = m._glob_to_re("fak/internal/gateway/**")
        self.assertTrue(rx.match("fak/internal/gateway/x.go"))
        self.assertTrue(rx.match("fak/internal/gateway/sub/deep/x.go"))

    def test_no_partial_segment_match(self):
        rx = m._glob_to_re("fak/internal/gateway/**")
        self.assertFalse(rx.match("fak/internal/gatewayx/x.go"))

    def test_single_star_within_segment(self):
        rx = m._glob_to_re("VERSION")
        self.assertTrue(rx.match("VERSION"))
        self.assertFalse(rx.match("VERSION/x"))


class RoutingRungTest(unittest.TestCase):
    def test_exact_scope(self):
        r = route(issue(1, "fix(gateway): silent MockPlanner fallback"))
        self.assertEqual(r["lane"], "gateway")
        self.assertEqual(r["confidence"], "exact-scope")

    def test_alias_scope(self):
        r = route(issue(2, "gpu(cuda): residency budget missing"))
        self.assertEqual(r["lane"], "compute")
        self.assertEqual(r["confidence"], "alias")
        self.assertIn("cuda->compute", r["signal"])

    def test_label_only_fallback(self):
        # No conventional scope; routed by label.
        r = route(issue(3, "GPU server Benchmark needs network access", labels=["gpu"]))
        self.assertEqual(r["lane"], "compute")
        self.assertEqual(r["confidence"], "label")

    def test_path_confirmed_overrides_wrong_scope(self):
        # Scope says docs, but the body names a tools/ path -> route to tools.
        r = route(issue(4, "docs(readme): wrong issue linked",
                         body="the bug is really in tools/issue_triage.py near line 90"))
        self.assertEqual(r["lane"], "tools")
        self.assertEqual(r["confidence"], "path-confirmed")
        self.assertTrue(r["signal_conflict"])

    def test_path_confirms_alias_no_conflict(self):
        # Alias swebench->experiments AND a path in fak/experiments -> agree, no conflict.
        r = route(issue(5, "fix(swebench): hardcoded path",
                         body="see fak/experiments/swebench/run.py default"))
        self.assertEqual(r["lane"], "experiments")
        self.assertEqual(r["confidence"], "path-confirmed")
        self.assertFalse(r["signal_conflict"])

    def test_no_scope_no_path_unrouted(self):
        r = route(issue(6, "Merge remaining branches after integration lands"))
        self.assertIsNone(r["lane"])
        self.assertEqual(r["confidence"], "none")
        self.assertTrue(r["unrouted_reason"])

    def test_exclusive_lane_scope_refused(self):
        # abi is exclusive — never auto-route, even though it IS a lane name.
        r = route(issue(7, "abi: hoist the public ABI surface"))
        self.assertIsNone(r["lane"])
        self.assertIn("exclusive", r["unrouted_reason"])

    def test_multi_lane_path_ambiguity_is_deterministic(self):
        body = "touches fak/internal/gateway/a.go and fak/internal/compute/b.go"
        r1 = route(issue(8, "refactor: shared change", body=body))
        r2 = route(issue(8, "refactor: shared change", body=body))
        self.assertEqual(r1["lane"], r2["lane"])  # deterministic
        self.assertTrue(r1["signal_conflict"])
        self.assertIn(r1["lane"], ("gateway", "compute"))

    def test_ambiguity_prefers_scope_matching_lane(self):
        # Two path lanes, scope picks one of them -> prefer it.
        body = "fak/internal/gateway/a.go and fak/internal/compute/b.go"
        r = route(issue(9, "fix(compute): shared", body=body))
        self.assertEqual(r["lane"], "compute")


class PayloadTest(unittest.TestCase):
    def _payload(self, routes, **kw):
        return m.build_payload(workspace="C:/work/fleet", routes=routes, trees=TREES, **kw)

    def test_counts_and_lane_grouping(self):
        routes = [route(issue(1, "fix(gateway): a")),
                  route(issue(2, "fix(gateway): b")),
                  route(issue(3, "gpu(cuda): c"))]
        p = self._payload(routes)
        self.assertEqual(p["counts"]["routed"], 3)
        self.assertEqual(p["lanes"]["gateway"]["count"], 2)
        self.assertEqual(p["lanes"]["compute"]["count"], 1)

    def test_unrouted_first_sort(self):
        routes = [route(issue(1, "fix(gateway): a")),
                  route(issue(2, "Merge branches"))]
        p = self._payload(routes)
        self.assertIsNone(p["issues"][0]["lane"])  # unrouted surfaced first

    def test_verdict_action_when_high_unrouted(self):
        routes = [route(issue(1, "Merge branches")),
                  route(issue(2, "Another epic")),
                  route(issue(3, "fix(gateway): a"))]
        p = self._payload(routes, max_unrouted_frac=0.25)  # 2/3 unrouted > 0.25
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "ACTION")
        self.assertEqual(p["finding"], "high_unrouted")

    def test_verdict_ok_when_mostly_routed(self):
        routes = [route(issue(i, "fix(gateway): a")) for i in range(1, 5)]
        p = self._payload(routes)
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "OK")

    def test_fetch_error_not_ok(self):
        p = self._payload([], fetch_error="dos doctor returned no lanes")
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "FETCH_ERROR")


class ConfigOverrideTest(unittest.TestCase):
    def test_scope_alias_override(self):
        # Custom alias: 'frobnicate' -> gateway.
        r = m.route_issue(
            issue(1, "feat(frobnicate): new"), LANES, TREES,
            scope_alias={"frobnicate": "gateway"},
        )
        self.assertEqual(r["lane"], "gateway")
        self.assertEqual(r["confidence"], "alias")


class CoverageTest(unittest.TestCase):
    def _payload(self, routes, **kw):
        return m.build_payload(workspace="C:/work/fleet", routes=routes, trees=TREES, **kw)

    def test_complete_when_under_cap(self):
        cov = m.compute_coverage(issues_fetched=426, issue_limit=1000)
        self.assertTrue(cov["complete"])
        self.assertEqual(cov["notes"], [])

    def test_truncated_when_fetch_hits_cap(self):
        cov = m.compute_coverage(issues_fetched=400, issue_limit=400)
        self.assertFalse(cov["complete"])
        self.assertTrue(cov["truncated"])
        self.assertTrue(any("issue-limit" in n for n in cov["notes"]))

    def test_incomplete_coverage_forces_action(self):
        # Everything we fetched routed cleanly, but the fetch was truncated -> ACTION.
        routes = [route(issue(i, "fix(gateway): a")) for i in range(1, 5)]
        p = self._payload(routes, coverage={"complete": False, "notes": ["gh fetch hit the cap"]})
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "ACTION")
        self.assertEqual(p["finding"], "incomplete_coverage")

    def test_fetch_error_wins_over_incomplete_coverage(self):
        p = self._payload([], fetch_error="no lanes",
                          coverage={"complete": False, "notes": ["cap"]})
        self.assertEqual(p["verdict"], "FETCH_ERROR")

    def test_complete_coverage_lets_ok_through(self):
        routes = [route(issue(i, "fix(gateway): a")) for i in range(1, 5)]
        p = self._payload(routes, coverage={"complete": True, "notes": []})
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "OK")

    def test_payload_carries_coverage(self):
        routes = [route(issue(1, "fix(gateway): a"))]
        p = self._payload(routes, coverage={"complete": True, "notes": [], "issues_fetched": 1})
        self.assertEqual(p["coverage"]["issues_fetched"], 1)


class CollectWiringTest(unittest.TestCase):
    def test_collect_uses_injected_fetcher_and_taxonomy(self):
        # Patch lane_taxonomy to avoid a real dos doctor call.
        orig = m.lane_taxonomy
        m.lane_taxonomy = lambda ws: (LANES, TREES)
        try:
            p = m.collect(
                Path("C:/work/fleet"),
                fetcher=lambda _ws: [issue(1, "fix(gateway): a"), issue(2, "Merge branches")],
            )
        finally:
            m.lane_taxonomy = orig
        self.assertEqual(p["counts"]["open"], 2)
        self.assertEqual(p["counts"]["routed"], 1)
        self.assertEqual(p["counts"]["unrouted"], 1)

    def test_collect_flags_fetch_error_on_no_lanes(self):
        orig = m.lane_taxonomy
        m.lane_taxonomy = lambda ws: ([], {})
        try:
            p = m.collect(Path("C:/work/fleet"), fetcher=lambda _ws: [issue(1, "fix(gateway): a")])
        finally:
            m.lane_taxonomy = orig
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "FETCH_ERROR")


if __name__ == "__main__":
    unittest.main()
