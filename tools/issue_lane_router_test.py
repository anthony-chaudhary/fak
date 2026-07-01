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
LANES = [
    "gateway", "compute", "docs", "tools", "experiments", "model", "abi",
    "bench", "ci", "sessionimage", "promptmmu", "devindex", "metrics", "examples",
]
# Real-layout trees (the Go module is the repo ROOT): `internal/...`, NOT
# `fak/internal/...`. Mirrors the corrected `dos doctor --json` output after the
# 2026-06-22 dos.toml prefix reconciliation. Issue bodies below still name files in
# the `fak/internal/...` doc-link convention on purpose — path_matches_lane strips
# the `fak/` so the doc-link still routes against these real-layout trees.
TREES = {
    "gateway": ["internal/gateway/**"],
    "compute": ["internal/compute/**"],
    "docs": ["docs/**"],
    "tools": ["tools/**"],
    "experiments": ["experiments/**"],
    "model": ["internal/model/**"],
    "abi": ["internal/abi/**"],
    "bench": ["internal/bench/**"],
    "ci": [".github/**"],
    "sessionimage": ["internal/sessionimage/**"],
    "promptmmu": ["internal/promptmmu/**"],
    "devindex": ["internal/devindex/**"],
    "metrics": ["internal/metrics/**"],
    "examples": ["examples/**"],
}


def issue(number: int, title: str, *, labels=None, body: str = "") -> dict:
    return {"number": number, "title": title, "body": body,
            "labels": [{"name": n} for n in (labels or [])]}


def route(iss: dict) -> dict:
    return m.route_issue(iss, LANES, TREES)


class GlobTest(unittest.TestCase):
    def test_doublestar_matches_nested(self):
        rx = m._glob_to_re("internal/gateway/**")
        self.assertTrue(rx.match("internal/gateway/x.go"))
        self.assertTrue(rx.match("internal/gateway/sub/deep/x.go"))

    def test_no_partial_segment_match(self):
        rx = m._glob_to_re("internal/gateway/**")
        self.assertFalse(rx.match("internal/gatewayx/x.go"))

    def test_single_star_within_segment(self):
        rx = m._glob_to_re("VERSION")
        self.assertTrue(rx.match("VERSION"))
        self.assertFalse(rx.match("VERSION/x"))


class PathNormalizationTest(unittest.TestCase):
    """A doc-link `fak/internal/...` path and the real-layout `internal/...` path
    must BOTH route to the same lane against the real-layout trees — the lockstep
    half of the 2026-06-22 dos.toml prefix reconciliation."""

    def test_doclink_prefix_routes_against_real_layout_trees(self):
        self.assertEqual(m.path_matches_lane("fak/internal/gateway/x.go", TREES), ["gateway"])

    def test_real_layout_path_routes(self):
        self.assertEqual(m.path_matches_lane("internal/gateway/x.go", TREES), ["gateway"])

    def test_non_fak_top_level_path_unaffected(self):
        self.assertEqual(m.path_matches_lane("tools/issue_triage.py", TREES), ["tools"])


class DottedRootPathTest(unittest.TestCase):
    """A leading-dot repo root (`.github/...`, `.claude/...`) must path-confirm — a
    `\\b` cannot anchor before a leading dot, so a workflow-only finding (e.g. a
    scheduled `.github/workflows/security-audit.yml` gate, the #978 ci-lane case)
    used to path-confirm NO lane. Covers the gate-signal feeder's routability."""

    CI_TREES = {"tools": ["tools/**"], "ci": [".github/**"], "claude": [".claude/**"]}
    CI_LANES = ["tools", "ci", "claude"]

    def test_github_path_is_extracted(self):
        # The natural backtick-wrapped position the old `\\b\\.github` never matched.
        found = m._PATH_RE.findall("scheduled by `.github/workflows/security-audit.yml`")
        self.assertEqual(found, [".github/workflows/security-audit.yml"])

    def test_github_path_confirms_ci_lane(self):
        iss = issue(101, "gate-signal: scheduled gate is RED",
                    body="the failing gate lives in `.github/workflows/security-audit.yml`")
        r = m.route_issue(iss, self.CI_LANES, self.CI_TREES)
        self.assertEqual(r["lane"], "ci")
        self.assertEqual(r["confidence"], "path-confirmed")

    def test_claude_path_confirms_its_lane(self):
        self.assertEqual(
            m._PATH_RE.findall("edit `.claude/skills/foo/SKILL.md` please"),
            [".claude/skills/foo/SKILL.md"])

    def test_embedded_dot_root_not_falsely_matched(self):
        # `x.github/y` is preceded by a word char -> the lookbehind rejects it, so a
        # mid-token dotted root never becomes a false path signal.
        self.assertEqual(m._PATH_RE.findall("see x.github/y here"), [])

    def test_word_root_partial_prefix_still_rejected(self):
        # The `\\b` half is untouched: `mytools/` must not match the `tools` root.
        self.assertEqual(m._PATH_RE.findall("the mytools/x dir"), [])


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

    def test_new_scope_aliases_route_rot_prone_issue_families(self):
        cases = [
            ("feat(terminal-bench): official contract", "bench"),
            ("feat(testing): deterministic time-travel harness", "ci"),
            ("feat(rehydrate): cold prompt-cache handling on wake", "sessionimage"),
            ("devex(scorecard): agentic coding loop-index", "devindex"),
            ("feat(dashboard): performance dashboards", "metrics"),
            ("feat(mobile): Android NDK example", "examples"),
            ("feat(support-maturity): scorecard router", "tools"),
        ]
        for title, lane in cases:
            with self.subTest(title=title):
                r = route(issue(20, title))
                self.assertEqual(r["lane"], lane)
                self.assertEqual(r["confidence"], "alias")

    def test_label_only_fallback(self):
        # No conventional scope; routed by label.
        r = route(issue(3, "GPU server Benchmark needs network access", labels=["gpu"]))
        self.assertEqual(r["lane"], "compute")
        self.assertEqual(r["confidence"], "label")

    def test_keyword_fallback_routes_unscoped_lane_terms(self):
        r = route(issue(21, "promptmmu rung 6: generalize skill demotion"))
        self.assertEqual(r["lane"], "promptmmu")
        self.assertEqual(r["confidence"], "keyword")
        self.assertIn("promptmmu->promptmmu", r["signal"])

    def test_keyword_fallback_does_not_match_inside_token(self):
        r = route(issue(22, "gpucheck/modelbench frontier"))
        self.assertIsNone(r["lane"])
        self.assertEqual(r["confidence"], "none")

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
        self.assertEqual(r["blocked_lane"], "abi")
        self.assertEqual(r["blocked_policy"], "exclusive")
        self.assertEqual(
            r["unrouted_reason"],
            "lane-policy:exclusive lane 'abi' is human-owned/operator-gated; held before spawn")
        self.assertIn("do not spawn", r["unblock_action"])

    def test_exclusive_lane_path_refused_even_when_path_confirmed(self):
        r = route(issue(10, "fix: ABI table",
                        body="touches fak/internal/abi/types.go"))
        self.assertIsNone(r["lane"])
        self.assertEqual(r["blocked_lane"], "abi")
        self.assertEqual(r["blocked_policy"], "exclusive")
        self.assertEqual(r["signal"], "path:abi")
        self.assertIn("held before spawn", r["unrouted_reason"])

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

    def test_ambiguity_with_exclusive_lane_is_held(self):
        body = "fak/internal/gateway/a.go and fak/internal/abi/types.go"
        r = route(issue(11, "fix(gateway): shared with ABI", body=body))
        self.assertIsNone(r["lane"])
        self.assertEqual(r["blocked_lane"], "abi")


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

    def test_lane_issues_ordered_by_confidence_then_number_desc(self):
        # A lane's issue list is folded by a dos-dispatch worker, so the best-routed
        # ticket must come first: path-confirmed (rung 1) outranks exact-scope (rung 2),
        # and within one confidence the newer issue number wins. gh fetch order (here
        # 1,2,3) must NOT leak through.
        routes = [
            route(issue(1, "fix(gateway): low a")),                      # exact-scope
            route(issue(2, "perf work", body="see fak/internal/gateway/serve.go")),  # path
            route(issue(3, "fix(gateway): low b")),                      # exact-scope
        ]
        p = self._payload(routes)
        grp = p["lanes"]["gateway"]
        # sanity: the fixtures produced the two confidence rungs we intend to order.
        self.assertEqual({r["number"]: r["confidence"] for r in
                          (route(issue(1, "fix(gateway): low a")),
                           route(issue(2, "perf work", body="see fak/internal/gateway/serve.go")),
                           route(issue(3, "fix(gateway): low b")))},
                         {1: "exact-scope", 2: "path-confirmed", 3: "exact-scope"})
        self.assertEqual(grp["issues"], [2, 3, 1])   # path first, then number desc

    def test_lane_carries_its_own_by_confidence(self):
        routes = [
            route(issue(1, "fix(gateway): a")),                          # exact-scope
            route(issue(2, "perf", body="see fak/internal/gateway/serve.go")),  # path
            route(issue(3, "fix(gateway): b")),                          # exact-scope
        ]
        p = self._payload(routes)
        self.assertEqual(p["lanes"]["gateway"]["by_confidence"],
                         {"path-confirmed": 1, "exact-scope": 2})

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

    def test_keyword_alias_override(self):
        r = m.route_issue(
            issue(2, "plain title with frobnicate inside"), LANES, TREES,
            keyword_alias={"frobnicate": "gateway"},
        )
        self.assertEqual(r["lane"], "gateway")
        self.assertEqual(r["confidence"], "keyword")


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


class DispatchabilityTest(unittest.TestCase):
    """Epic parents and human-blocked issues are kept out of the candidate set."""

    def test_epic_label_is_not_dispatchable(self):
        self.assertTrue(m.is_epic(issue(1, "anything", labels=["epic"])))
        self.assertFalse(m.is_dispatchable(issue(1, "anything", labels=["epic"])))

    def test_epic_title_convention_is_not_dispatchable(self):
        self.assertTrue(m.is_epic(issue(2, "epic(serving): warm the next turn")))
        self.assertTrue(m.is_epic(issue(3, "epic: ECC-style memory integrity")))
        self.assertFalse(m.is_dispatchable(issue(2, "epic(serving): warm the next turn")))

    def test_plain_fix_is_dispatchable(self):
        # A normal issue (no epic label, title doesn't start with epic) is routable.
        self.assertFalse(m.is_epic(issue(4, "fix(gateway): drop bad tool call")))
        self.assertTrue(m.is_dispatchable(issue(4, "fix(gateway): drop bad tool call")))
        # 'epic' merely mentioned mid-title is NOT an epic (anchored at start only).
        self.assertFalse(m.is_epic(issue(5, "docs: describe the epic rollout plan")))

    def test_human_blocked_still_not_dispatchable(self):
        blocked = issue(6, "needs a trademark filing", labels=[m.BLOCKED_BY_HUMAN_LABEL])
        self.assertFalse(m.is_dispatchable(blocked))

    def test_collect_skips_epics_routes_the_rest(self):
        orig = m.lane_taxonomy
        m.lane_taxonomy = lambda ws: (LANES, TREES)
        try:
            p = m.collect(
                Path("C:/work/fleet"),
                fetcher=lambda _ws: [
                    issue(1, "fix(gateway): a"),
                    issue(2, "epic(gateway): umbrella", labels=["epic"]),
                ],
            )
        finally:
            m.lane_taxonomy = orig
        # The epic is skipped (surfaced, not routed); only the plain issue routes.
        self.assertEqual(p["counts"]["routed"], 1)
        self.assertEqual(p["counts"]["skipped_human_blocked"], 1)


class InjectedIssuesTest(unittest.TestCase):
    """`--issues PATH|-` lets a named view (issue_views.py show --json) drive
    routing instead of the built-in gh fetch — the documented composition."""

    def _stdin(self, text: str):
        import io
        from unittest import mock
        return mock.patch.object(m.sys, "stdin", io.StringIO(text))

    def test_load_from_file(self):
        import json
        import tempfile
        rows = [{"number": 7, "title": "fix(tools): x", "labels": [], "body": ""}]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "view.json"
            p.write_text(json.dumps(rows), encoding="utf-8")
            self.assertEqual(m.load_injected_issues(str(p)), rows)

    def test_load_from_stdin(self):
        import json
        rows = [{"number": 8, "title": "t", "labels": [], "body": ""}]
        with self._stdin(json.dumps(rows)):
            self.assertEqual(m.load_injected_issues("-"), rows)

    def test_empty_input_is_empty_list(self):
        with self._stdin("   \n"):
            self.assertEqual(m.load_injected_issues("-"), [])

    def test_non_array_rejected(self):
        with self._stdin('{"number": 1}'):
            with self.assertRaises(ValueError):
                m.load_injected_issues("-")

    def test_invalid_json_rejected(self):
        with self._stdin("not json at all"):
            with self.assertRaises(ValueError):
                m.load_injected_issues("-")

    def test_injected_collect_routes_and_marks_coverage(self):
        # A view slice flows straight into routing; coverage is "complete" (the slice
        # IS the intended backlog), never the silent-truncation ACTION verdict.
        orig = m.lane_taxonomy
        m.lane_taxonomy = lambda root: (LANES, TREES)
        try:
            rows = [issue(30, "fix(gateway): x"), issue(31, "fix(tools): y")]
            p = m.collect(Path("."), fetcher=lambda ws: rows, injected=True)
        finally:
            m.lane_taxonomy = orig
        self.assertTrue(p["coverage"]["injected"])
        self.assertTrue(p["coverage"]["complete"])
        self.assertIn("gateway", p["lanes"])
        self.assertIn("tools", p["lanes"])

    def test_injected_empty_is_not_a_fetch_error(self):
        orig = m.lane_taxonomy
        m.lane_taxonomy = lambda root: (LANES, TREES)
        try:
            p = m.collect(Path("."), fetcher=lambda ws: [], injected=True)
        finally:
            m.lane_taxonomy = orig
        # An empty view slice is a legit "0 issues match", not a gh auth/network error.
        self.assertNotEqual(p.get("finding"), "fetch_error")


class ViewFetchFailSoft(unittest.TestCase):
    """`fetch_view_issues` resolves a named issue-view to a gh search and parses the
    rows; it RAISES on a bad view / gh failure so the --view caller can fail-soft to
    the full open backlog (the unattended tick must never starve on a mis-pointed view)."""

    def test_view_resolves_to_milestone_search_and_parses_rows(self):
        seen = {}

        def fake_run_text(cmd, cwd, *, timeout=60):
            seen["cmd"] = cmd
            return {"stdout": '[{"number":1,"title":"x","labels":[],"body":""}]',
                    "returncode": 0}

        orig = m.run_text
        m.run_text = fake_run_text
        try:
            rows = m.fetch_view_issues(Path("."), "m2-kv-cache")
        finally:
            m.run_text = orig
        self.assertEqual(len(rows), 1)
        # the view's query is milestone-scoped, so the resolved gh argv carries it
        self.assertTrue(any("milestone:" in str(a) for a in seen["cmd"]),
                        f"resolved gh argv not milestone-scoped: {seen.get('cmd')}")

    def test_gh_error_raises_for_failsoft(self):
        def fake_run_text(cmd, cwd, *, timeout=60):
            return {"stdout": "", "stderr": "boom", "returncode": 1}

        orig = m.run_text
        m.run_text = fake_run_text
        try:
            with self.assertRaises(RuntimeError):
                m.fetch_view_issues(Path("."), "m2-kv-cache")
        finally:
            m.run_text = orig

    def test_unknown_view_raises_keyerror(self):
        with self.assertRaises(KeyError):
            m.fetch_view_issues(Path("."), "no-such-view-zzz")


class LaneConfTagTest(unittest.TestCase):
    def test_strongest_rung_first_zeros_omitted(self):
        tag = m._lane_conf_tag({"exact-scope": 4, "path-confirmed": 1,
                                "keyword": 2, "label": 0})
        self.assertEqual(tag, "path 1·scope 4·kw 2")   # rank order; label(0) omitted

    def test_empty_or_none_is_blank(self):
        self.assertEqual(m._lane_conf_tag(None), "")
        self.assertEqual(m._lane_conf_tag({}), "")

    def test_render_shows_lane_confidence_tag(self):
        routes = [route(issue(1, "fix(gateway): a")),
                  route(issue(2, "perf", body="see fak/internal/gateway/serve.go"))]
        text = m.render(m.build_payload(workspace="C:/work/fleet", routes=routes,
                                        trees=TREES))
        self.assertIn("[path 1·scope 1]", text)

    def test_render_shows_exclusive_lane_policy_and_unblock_action(self):
        routes = [route(issue(7, "abi: hoist the public ABI surface"))]
        payload = m.build_payload(workspace="C:/work/fleet", routes=routes, trees=TREES)
        text = m.render(payload)
        self.assertIn("[exclusive:abi]", text)
        self.assertIn("do not spawn an issue worker", text)
        md = m.render_md(payload, date="2026-07-01")
        self.assertIn("| #7 | lane-policy:exclusive lane 'abi' is human-owned/operator-gated; held before spawn | exclusive:abi |", md)
        self.assertIn("do not spawn an issue worker", md)


if __name__ == "__main__":
    unittest.main()
