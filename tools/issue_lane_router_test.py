#!/usr/bin/env python3
"""Hermetic tests for tools/issue_lane_router.py.

No real gh/dos: the router is a pure function over a fixture lane taxonomy +
injected issue dicts. The load() importlib pattern mirrors the sibling tools.
"""
from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

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
    # work-class fixtures: two hard-classed infra lanes, a frontdoor lane, and the
    # operator-gated release lane (exclusive).
    "slackoutbox", "appversion", "release",
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
    "slackoutbox": ["internal/slackoutbox/**"],
    "appversion": ["internal/appversion/**"],
    "release": ["internal/release/**"],
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

    # -- trust-critical pre-route rung (#3122) ------------------------------
    # The HELD direction and the NOT-held direction are both asserted here on
    # purpose: over-holding (routing-time holds of merely-self-source work the
    # guard ships happily) is the specific regression this rung risks, so a run
    # that only exercises the hold does not witness the rung.

    def test_trust_critical_text_misrouted_to_safe_lane_is_held(self):
        # `dispatch` aliases to the shippable `tools` lane, but the real work is
        # in the adjudicator — the referee a guarded worker may never ship. The
        # lane tree (tools/**) never reveals that, so the router must.
        r = route(issue(30, "fix(dispatch): tighten the pre-route verdict",
                        body="the decision lives in internal/adjudicator/decide.go"))
        self.assertIsNone(r["lane"])
        self.assertEqual(r["blocked_lane"], "tools")
        self.assertEqual(r["blocked_policy"], "trust-critical")
        self.assertIn("internal/adjudicator/decide.go", r["signal"])
        self.assertIn("internal/adjudicator/decide.go", r["unrouted_reason"])
        self.assertIn("held before spawn", r["unrouted_reason"])
        self.assertIn("can never ship it", r["unblock_action"])

    def test_merely_self_source_text_still_routes(self):
        # internal/gateway is self-source but NOT trust-critical: the guard ships
        # it, so holding it here would starve the dispatch surface.
        r = route(issue(31, "fix(dispatch): tighten the tick",
                        body="the decision lives in internal/gateway/router.go"))
        self.assertEqual(r["lane"], "tools")
        self.assertEqual(r["confidence"], "alias")
        self.assertIsNone(r["unrouted_reason"])
        self.assertNotIn("blocked_policy", r)

    def test_trust_critical_prefix_does_not_match_inside_a_longer_token(self):
        # `myinternal/policy` / `x/internal/policy` are not fak's own policy tree.
        for body in ("see myinternal/policyfile.go", "vendored at x/internal/policy/x.go"):
            with self.subTest(body=body):
                r = route(issue(32, "fix(dispatch): unrelated", body=body))
                self.assertEqual(r["lane"], "tools")

    def test_trust_critical_hold_does_not_override_an_exclusive_hold(self):
        # An already-held row keeps the stronger exclusive verdict + its witness.
        r = route(issue(33, "abi: hoist the surface",
                        body="also touches internal/kernel/kernel.go"))
        self.assertIsNone(r["lane"])
        self.assertEqual(r["blocked_lane"], "abi")
        self.assertEqual(r["blocked_policy"], "exclusive")


class TrustCriticalPreRouteTest(unittest.TestCase):
    """#3122: the routing-time arm of the self-modify hold.

    Its own taxonomy, because the module-level fixture declares no lane whose tree
    IS trust-critical (other than the exclusive `abi`), and the correctly-routed
    non-hold is the case that keeps this rung from stealing lane attribution from
    the pick-time lane-tree arm (`dispatchtick.SelfModifyHold`).
    """

    LANES = ["tools", "docs", "gateway", "adjudicator", "policy"]
    TREES = {
        "tools": ["tools/**"],
        "docs": ["docs/**"],
        "gateway": ["internal/gateway/**"],
        "adjudicator": ["internal/adjudicator/**"],
        "policy": ["internal/policy/**"],
    }

    def route(self, iss: dict) -> dict:
        return m.route_issue(iss, self.LANES, self.TREES)

    def test_correctly_routed_trust_critical_lane_is_not_held_here(self):
        # Routed to the lane that OWNS the tree: the lane tree already reveals the
        # hazard, so the pick-time arm holds it with the better witness and the
        # router leaves the lane attribution intact.
        r = self.route(issue(40, "fix(adjudicator): reversibility verdict",
                             body="internal/adjudicator/reversibility.go"))
        self.assertEqual(r["lane"], "adjudicator")
        self.assertIsNone(r["unrouted_reason"])

    def test_predicate_matches_every_declared_trust_critical_tree(self):
        for prefix in m.TRUST_CRITICAL_TREE_PREFIXES:
            with self.subTest(prefix=prefix):
                self.assertTrue(m.is_trust_critical_tree(prefix + "x.go"))
                self.assertEqual(
                    m.issue_text_targets_trust_critical(f"work in {prefix}x.go"),
                    prefix + "x.go")

    def test_predicate_normalizes_doclink_and_windows_globs(self):
        for glob in ("fak/internal/kernel/**", "./internal/kernel/**",
                     "internal\\kernel\\kernel.go"):
            with self.subTest(glob=glob):
                self.assertTrue(m.is_trust_critical_tree(glob))

    def test_predicate_rejects_merely_self_source_trees(self):
        for glob in ("internal/gateway/**", "cmd/fak/**", "tools/**", "", "  "):
            with self.subTest(glob=glob):
                self.assertFalse(m.is_trust_critical_tree(glob))

    def test_trust_critical_file_globs_are_held(self):
        self.assertTrue(m.is_trust_critical_tree("dos.toml"))
        self.assertTrue(m.is_trust_critical_tree(".dos/registry.json"))
        self.assertTrue(m.is_trust_critical_tree("VERSION"))

    def test_lane_is_trust_critical_reads_the_lane_tree(self):
        self.assertTrue(m.lane_is_trust_critical("policy", self.TREES))
        self.assertFalse(m.lane_is_trust_critical("gateway", self.TREES))
        self.assertFalse(m.lane_is_trust_critical(None, self.TREES))
        self.assertFalse(m.lane_is_trust_critical("undeclared", self.TREES))

    def test_held_issue_leaves_the_shippable_lane_priority_order(self):
        # #3122's acceptance, asserted on the payload an operator actually reads:
        # the mis-routed issue is gone from lanes['tools'].issues, and the lane the
        # hold names survives as blocked_lane for triage.
        misrouted = issue(41, "fix(dispatch): pre-route verdict",
                          body="internal/architest/architest_test.go")
        ordinary = issue(42, "fix(dispatch): unrelated tick fix")
        routes = [self.route(misrouted), self.route(ordinary)]
        payload = m.build_payload(workspace="C:/work/fleet", routes=routes,
                                  trees=self.TREES)
        self.assertEqual(payload["lanes"]["tools"]["issues"], [42])
        held = [r for r in payload["issues"] if r["number"] == 41][0]
        self.assertIsNone(held["lane"])
        self.assertEqual(held["blocked_lane"], "tools")


class ExclusiveLaneDerivationTest(unittest.TestCase):
    """#4027: the exclusive-lane set is derived from `dos doctor` lanes.exclusive
    (a single source of truth), not a hand-maintained literal, with a drift guard
    pinning it to the repo dos.toml so a lane newly marked exclusive/renamed there
    fails HERE instead of silently desyncing from the router's refusal gate."""

    REPO_ROOT = SCRIPT.resolve().parent.parent

    def _taxonomy_with(self, exclusive):
        """Run lane_taxonomy against a stubbed `dos doctor` payload (hermetic —
        no real dos binary), so the derivation wiring is exercised in isolation."""
        import json as _json
        payload = {"lanes": {
            "concurrent": ["gateway", "tools"],
            "trees": {"gateway": ["internal/gateway/**"], "tools": ["tools/**"]},
            "exclusive": exclusive,
        }}
        orig = m.run_text
        m.run_text = lambda cmd, cwd, **kw: {"stdout": _json.dumps(payload), "returncode": 0}
        try:
            return m.lane_taxonomy(self.REPO_ROOT)
        finally:
            m.run_text = orig

    def test_lane_taxonomy_returns_exclusive_from_doctor(self):
        _concurrent, _trees, exclusive = self._taxonomy_with(["abi", "release", "dos", "global"])
        self.assertEqual(exclusive, {"abi", "release", "dos", "global"})

    def test_lane_taxonomy_falls_back_when_doctor_omits_exclusive(self):
        # An older `dos` (no exclusive field) or a bad workspace → the module
        # literal is the documented fallback, never an empty (block-nothing) set.
        _concurrent, _trees, exclusive = self._taxonomy_with([])
        self.assertEqual(exclusive, set(m.EXCLUSIVE_LANES))
        self.assertTrue(exclusive)

    def test_dos_lane_refused_by_derivation_not_literal(self):
        # The #4027 regression made concrete: `dos` is exclusive in dos.toml but
        # absent from the module literal. A DERIVED exclusive set refuses a
        # dos-scoped issue; the stale literal silently did not.
        iss = issue(4027, "dos: retune the exclusive lane roster", body="edit dos.toml")
        derived = m.route_issue(iss, LANES + ["dos"], TREES,
                                exclusive={"abi", "release", "dos", "global"})
        self.assertIsNone(derived["lane"])
        self.assertEqual(derived["blocked_lane"], "dos")
        self.assertEqual(derived["blocked_policy"], "exclusive")
        # With the pre-#4027 literal (no `dos`), the SAME issue is not held.
        literal = m.route_issue(iss, LANES + ["dos"], TREES,
                                exclusive={"abi", "release", "global"})
        self.assertNotEqual(literal.get("blocked_lane"), "dos")

    def test_repo_dos_toml_marks_dos_exclusive(self):
        # Drift guard against the authoritative source. `dos doctor` echoes
        # dos.toml `[lanes].exclusive`, so pinning the router's derivation to this
        # file catches a lane newly marked exclusive/renamed in dos.toml.
        import tomllib
        with (self.REPO_ROOT / "dos.toml").open("rb") as fh:
            data = tomllib.load(fh)
        declared = {str(x) for x in (data.get("lanes", {}).get("exclusive") or [])}
        self.assertIn("dos", declared,
                      "dos.toml [lanes].exclusive must carry `dos` (#4027 drift)")
        self.assertTrue({"abi", "release", "global"} <= declared)


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


class UnroutedScopeClusterTest(unittest.TestCase):
    """A flat UNROUTED count is not actionable; the operator next-action needs to
    know WHICH scope families are rotting and how big each is (a real
    `internal/<scope>/` leaf with no declared lane, or an ambiguous scope). The
    clusterer buckets the truly-unrouted rows by the same scope key routing uses."""

    def _payload(self, routes, **kw):
        return m.build_payload(workspace="C:/work/fleet", routes=routes, trees=TREES, **kw)

    def test_clusters_unrouted_by_scope_count_desc(self):
        routes = [
            route(issue(1, "feat(negframe): a")),   # no negframe lane/alias -> unrouted
            route(issue(2, "feat(negframe): b")),
            route(issue(3, "feat(quality): c")),
            route(issue(4, "fix(gateway): routed")),  # routed, must be excluded
        ]
        clusters = m.unrouted_scope_clusters(routes)
        self.assertEqual(clusters[0], {"scope": "negframe", "count": 2, "issues": [2, 1]})
        self.assertEqual(clusters[1], {"scope": "quality", "count": 1, "issues": [3]})
        self.assertNotIn("gateway", [c["scope"] for c in clusters])

    def test_bare_prefix_and_no_scope_bucket(self):
        routes = [
            route(issue(5, "harness-res: wire the ledger")),  # bare prefix scope
            route(issue(6, "Merge remaining branches after integration")),  # no scope
        ]
        by = {c["scope"]: c for c in m.unrouted_scope_clusters(routes)}
        self.assertIn("harness-res", by)
        self.assertIn("(no-scope)", by)

    def test_held_exclusive_rows_excluded(self):
        # An exclusive/human-blocked hold carries a blocked_lane and is a DIFFERENT
        # triage surface (the exclusive-lane render) — it must not pollute the
        # scope-cluster worklist of "add a lane/alias to drain these".
        routes = [route(issue(7, "abi: hoist the public ABI surface"))]  # held on abi
        self.assertEqual(m.unrouted_scope_clusters(routes), [])

    def test_payload_carries_unrouted_scopes(self):
        routes = [route(issue(1, "feat(negframe): a")),
                  route(issue(2, "fix(gateway): routed"))]
        p = self._payload(routes)
        self.assertEqual(p["unrouted_scopes"],
                         [{"scope": "negframe", "count": 1, "issues": [1]}])

    def test_render_shows_scope_clusters(self):
        routes = [route(issue(11, "feat(negframe): a")),
                  route(issue(12, "feat(negframe): b"))]
        text = m.render(self._payload(routes))
        self.assertIn("UNROUTED by scope", text)
        self.assertIn("negframe", text)

    def test_render_md_shows_scope_cluster_table(self):
        routes = [route(issue(11, "feat(quality): a"))]
        md = m.render_md(self._payload(routes), date="2026-07-14")
        self.assertIn("## UNROUTED by scope", md)
        self.assertIn("| `quality` | 1 | #11 |", md)


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


class SeparatorFoldedScopeTest(unittest.TestCase):
    """Rung 3b: a hyphenated scope naming an unpunctuated lane must still route.

    Lane names in dos.toml carry no separators (`sessionimage`), but issue authors
    write the same leaf hyphenated in a Conventional-Commits scope
    (`feat(session-image): ...`). Before the fold those titles matched no rung and
    the issue went UNROUTED — and an unrouted issue never enters `lanes[...]`, so
    the dispatch picker can never select it. Measured on the live backlog
    2026-08-06: 26 open issues (cachevalue 12, sessionjournal 6, harnessres 5,
    sessionaudit 1, operatorbrief 1, dogfoodissues 1) were stranded this way.
    """

    def test_hyphenated_scope_routes_to_unpunctuated_lane(self):
        for title, lane in [
            ("feat(session-image): cold prompt-cache handling", "sessionimage"),
            ("fix(slack-outbox): drop the duplicate send", "slackoutbox"),
            ("feat(app-version): stamp the build", "appversion"),
        ]:
            with self.subTest(title=title):
                r = route(issue(30, title))
                self.assertEqual(r["lane"], lane)
                # Graded `alias`, never `exact-scope`: a folded hit is weaker
                # evidence than a literal lane name.
                self.assertEqual(r["confidence"], "alias")

    def test_underscore_and_space_separators_fold_too(self):
        for title in ("feat(session_image): x", "feat(session image): x"):
            with self.subTest(title=title):
                self.assertEqual(route(issue(31, title))["lane"], "sessionimage")

    def test_explicit_alias_still_wins_over_the_fold(self):
        # THE ordering guard. `terminal-bench` is an explicit SCOPE_ALIAS -> bench,
        # and ALSO matches a real `terminalbench` lane once the hyphen is folded.
        # The hand-written alias is the operator's stated intent, so rung 3b must sit
        # BELOW it — otherwise this issue silently re-routes off `bench`.
        r = m.route_issue(issue(32, "feat(terminal-bench): official contract"),
                          LANES + ["terminalbench"], TREES)
        self.assertEqual(r["lane"], "bench")
        self.assertEqual(r["signal"], "scope:terminal-bench->bench")

    def test_path_still_overrides_a_folded_scope(self):
        # The fold is scope-rung evidence; a real path still outranks it. (Uses the
        # `fak/`-prefixed doc-link form: the path rung deliberately skips a bare
        # `internal/...`, so that form would not exercise the override at all.)
        r = route(issue(33, "feat(session-image): x",
                        body="the fix is in fak/internal/gateway/serve.go"))
        self.assertEqual(r["lane"], "gateway")
        self.assertEqual(r["confidence"], "path-confirmed")

    def test_folded_signal_names_the_issues_own_scope_token(self):
        # The signal is the operator's audit trail for WHY a lane was chosen. A
        # folded hit must report `session-image`, not the Conventional-Commits
        # TYPE (`feat`), which the pre-existing token fallback would otherwise emit.
        r = route(issue(35, "feat(session-image): x"))
        self.assertEqual(r["signal"], "scope:session-image->sessionimage")

    def test_fold_does_not_invent_a_lane(self):
        # A hyphenated scope matching NO lane stays unrouted — the fold widens the
        # exact-scope rung, it does not add a catch-all.
        r = route(issue(34, "feat(no-such-leaf): x"))
        self.assertIsNone(r["lane"])
        self.assertEqual(r["confidence"], "none")

    def test_ambiguous_normalized_lane_is_dropped_not_guessed(self):
        # If two lanes ever collapse to the same folded form, resolve NEITHER.
        idx = m._norm_lane_index({"foo-bar", "foobar", "baz"})
        self.assertNotIn("foobar", idx)
        self.assertEqual(idx["baz"], "baz")

    def test_live_lane_roster_has_no_folded_collisions(self):
        # The fold is only unambiguous while the real roster stays collision-free.
        try:
            concurrent, _trees, _excl = m.lane_taxonomy(Path(__file__).resolve().parents[1])
        except Exception:  # pragma: no cover - no dos binary in this environment
            self.skipTest("dos doctor unavailable")
        if not concurrent:
            self.skipTest("empty lane roster")
        folded: dict[str, list[str]] = {}
        for lane in concurrent:
            folded.setdefault(m._norm_scope(lane), []).append(lane)
        self.assertEqual([v for v in folded.values() if len(v) > 1], [])


class CoverageTest(unittest.TestCase):
    def _payload(self, routes, **kw):
        return m.build_payload(workspace="C:/work/fleet", routes=routes, trees=TREES, **kw)

    def test_default_issue_limit_is_the_shared_constant(self):
        # collect() and the CLI must agree, so raising the cap in ONE place lifts it
        # for every dispatch caller (none of them pass --issue-limit explicitly).
        import inspect
        self.assertEqual(inspect.signature(m.collect).parameters["issue_limit"].default,
                         m.DEFAULT_ISSUE_LIMIT)

    def test_default_issue_limit_clears_the_real_backlog(self):
        # `gh issue list` is NEWEST-first, so a fetch that hits the cap drops the
        # OLDEST open issues — they are never routed and never dispatch candidates.
        # The 1000 default truncated a measured 1383-issue backlog (2026-08-06),
        # hiding 383 open issues from every picker. Keep headroom above it.
        self.assertGreaterEqual(m.DEFAULT_ISSUE_LIMIT, 3000)
        self.assertTrue(m.compute_coverage(
            issues_fetched=1383, issue_limit=m.DEFAULT_ISSUE_LIMIT)["complete"])

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
        m.lane_taxonomy = lambda ws: (LANES, TREES, {"abi", "release", "global"})
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
        m.lane_taxonomy = lambda ws: ([], {}, {"abi", "release", "global"})
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
        m.lane_taxonomy = lambda ws: (LANES, TREES, {"abi", "release", "global"})
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
        m.lane_taxonomy = lambda root: (LANES, TREES, {"abi", "release", "global"})
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
        m.lane_taxonomy = lambda root: (LANES, TREES, {"abi", "release", "global"})
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


class WorkClassTest(unittest.TestCase):
    """The work-CLASS axis (frontdoor / infra / dev) derived on top of the lane.

    Precedence is frontdoor > infra > dev: a cross-cutting front-door signal wins
    regardless of lane; else a hard-classed lane decides; else a mixed lane with a
    fleet-plumbing cue is infra; else the `dev` residual."""

    def cls(self, iss: dict) -> str:
        return route(iss)["class"]

    def test_product_leaf_is_dev(self):
        self.assertEqual(self.cls(issue(1, "feat(model): add rope scaling")), m.CLASS_DEV)
        self.assertEqual(self.cls(issue(2, "perf(gateway): batch decode")), m.CLASS_DEV)

    def test_hard_infra_lane_is_infra(self):
        self.assertEqual(self.cls(issue(3, "fix(ci): pin the runner image")), m.CLASS_INFRA)
        self.assertEqual(self.cls(issue(4, "feat(metrics): add a histogram")), m.CLASS_INFRA)
        self.assertEqual(self.cls(issue(5, "feat(slackoutbox): thread card")), m.CLASS_INFRA)

    def test_hard_frontdoor_lane_is_frontdoor(self):
        self.assertEqual(self.cls(issue(6, "feat(appversion): derive versions")),
                         m.CLASS_FRONTDOOR)

    def test_front_door_path_signal_beats_dev_residual(self):
        # A docs-lane issue that names a public front-door surface classes frontdoor,
        # not dev — the fenced release path must not leak into the default dev stream.
        iss = issue(7, "docs(readme): refresh the install section",
                    body="updates README.md and the install.sh on-ramp")
        r = route(iss)
        self.assertEqual(r["lane"], "docs")
        self.assertEqual(r["class"], m.CLASS_FRONTDOOR)

    def test_front_door_scope_signal(self):
        # A release-path scope classes frontdoor even when the lane is exclusive/blocked
        # (release is operator-gated, so the dispatch lane is None) — the class survives.
        r = route(issue(8, "feat(release): promote dev to main"))
        self.assertEqual(r["class"], m.CLASS_FRONTDOOR)

    def test_front_door_label_overrides_infra_lane(self):
        # version-everything is a front-door label: it wins over the ci lane's infra seed.
        r = route(issue(9, "fix(ci): version badge on the README",
                        labels=["version-everything"]))
        self.assertEqual(r["lane"], "ci")
        self.assertEqual(r["class"], m.CLASS_FRONTDOOR)

    def test_mixed_tools_lane_dispatch_cue_is_infra(self):
        r = route(issue(10, "feat(dispatch): supervisor cadence knob",
                        labels=["dispatch"]))
        self.assertEqual(r["lane"], "tools")
        self.assertEqual(r["class"], m.CLASS_INFRA)

    def test_mixed_tools_lane_no_cue_is_dev(self):
        # A generic tools helper with no plumbing cue falls to the dev residual.
        r = route(issue(11, "feat(tools): add a kernel-parse helper"))
        self.assertEqual(r["lane"], "tools")
        self.assertEqual(r["class"], m.CLASS_DEV)

    def test_lane_class_default_is_dev(self):
        self.assertEqual(m.lane_class("engine"), m.CLASS_DEV)
        self.assertEqual(m.lane_class(None), m.CLASS_DEV)
        self.assertEqual(m.lane_class("ci"), m.CLASS_INFRA)
        self.assertEqual(m.lane_class("appversion"), m.CLASS_FRONTDOOR)

    def test_payload_rolls_up_by_class(self):
        routes = [route(issue(1, "feat(model): x")),
                  route(issue(2, "fix(ci): y")),
                  route(issue(3, "feat(appversion): z"))]
        payload = m.build_payload(workspace="C:/work/fleet", routes=routes, trees=TREES)
        self.assertEqual(payload["counts"]["by_class"],
                         {m.CLASS_FRONTDOOR: 1, m.CLASS_INFRA: 1, m.CLASS_DEV: 1})
        self.assertEqual(payload["classes"][m.CLASS_INFRA]["issues"], [2])

    def test_render_shows_class_rollup(self):
        routes = [route(issue(1, "feat(model): x")), route(issue(2, "fix(ci): y"))]
        text = m.render(m.build_payload(workspace="C:/work/fleet", routes=routes,
                                        trees=TREES))
        self.assertIn("by class:", text)
        self.assertIn("infra=1", text)
        self.assertIn("dev=1", text)


class BodyDeclaredLaneTest(unittest.TestCase):
    """`body_declared_lane` extracts the issue body's OWN lane declaration: the
    contract-overlay `## Lane` section and the inline `Lane: `x`` field row."""

    def test_overlay_section_form(self):
        self.assertEqual(m.body_declared_lane("## In scope\nstuff\n\n## Lane\nmodver\n\n## Likely files\n- x"),
                         "modver")

    def test_section_form_tolerates_blank_line_and_backticks(self):
        self.assertEqual(m.body_declared_lane("## Lane\n\n`cmd`\n"), "cmd")

    def test_inline_field_form(self):
        self.assertEqual(
            m.body_declared_lane("Lane: `modver` · Paths: `internal/modver/` · Priority: P3"),
            "modver")

    def test_prose_mention_is_not_a_declaration(self):
        # Mid-sentence "lane:" prose and a bare heading with no value never match.
        self.assertIsNone(m.body_declared_lane("route it to the right lane: whichever fits"))
        self.assertIsNone(m.body_declared_lane("## Lane\n"))
        self.assertIsNone(m.body_declared_lane(""))
        self.assertIsNone(m.body_declared_lane(None))


class WitnessOnlyGithubTest(unittest.TestCase):
    """#2609: `.github/**` named as a WITNESS/mention surface must not path-confirm
    the ci lane when the body carries a stronger binding elsewhere (an explicit
    `## Lane`/`Lane:` declaration, an exact-scope lane token, or a concrete
    non-.github path). The July-4 CI canary drained shipless on exactly this:
    #2464 (body-bound to modver) and #2233 (a multi-surface caller migration)
    were offered as standalone `.github/**` work. A workflow-only issue with NO
    stronger binding still routes ci (the #978 gate-signal case)."""

    LANES = ["gateway", "docs", "tools", "cmd", "ci", "claude", "modver", "abi"]
    TREES = {
        "gateway": ["internal/gateway/**"],
        "docs": ["docs/**"],
        "tools": ["tools/**"],
        "cmd": ["cmd/**"],
        "ci": [".github/**"],
        "claude": [".claude/**"],
        "modver": ["internal/modver/**"],
        "abi": ["internal/abi/**"],
    }

    # Shaped from the REAL #2464 body: lane/path-bound to modver (internal/modver/),
    # with `.github/workflows/*` named only as the key space being modeled.
    BODY_2464 = (
        "<!-- fak-issue-key: modver-workflows-keyspace -->\n\n"
        "Part of #2458 (version-everything).\n\n"
        "## Working spine\n"
        "internal/modver/ (Snapshot/DeltaRows/JoinScores, Runner seam) and the fak\n"
        "version modules shell (cmd/fak/version_modules.go) are the working base to\n"
        "extend; docs/notes/VERSION-EVERYTHING-SPINE-2026-07-03.md is the doctrine.\n\n"
        "## In scope\n"
        ".github/workflows/* key space; optionally tools/register_*.ps1 "
        "scheduled-task installers.\n\n"
        "Lane: `modver` · Paths: `internal/modver/` · Priority: P3\n\n"
        "## Lane\nmodver\n\n"
        "## Likely files\n"
        "- `internal/modver/` — the Runner/Snapshot seam this unit extends\n"
        "- `.github/workflows/` — the workflow keyspace being versioned\n"
    )

    # Shaped from the REAL #2233 body: a broad caller-migration whose surfaces are
    # skills/tools/Makefile/docs/hooks; `.github/workflows/*` is one mentioned
    # surface among many, not an independently dispatchable slice.
    BODY_2233 = (
        "Parent: #2228 (C4 — blocked by C2; C5 is blocked by this)\n\n"
        "# What\n\n"
        "Migrate every in-repo caller of a bare dev-tier spelling to the `fak dev "
        "<verb>` form. Callers live in: `.claude/skills/*/SKILL.md`, `tools/*.py`, "
        "`tools/*.ps1`, `Makefile`, `.github/workflows/*`, scheduled-task "
        "registration scripts, `dos.toml` command strings, docs code blocks, and "
        "the shell hooks under `hooks/`.\n\n"
        "# Method\n\n"
        "1. Build the audit sweep. The sweep must prune `.fak/` + `.claude/` "
        "checkout copies and `docs/archive/`.\n"
        "2. Rewrite callers in reviewable batches (by surface: skills, tools, CI, "
        "Makefile, docs).\n"
    )

    def route(self, iss: dict) -> dict:
        return m.route_issue(iss, self.LANES, self.TREES)

    def test_2464_shape_routes_modver_not_ci(self):
        iss = issue(2464, "modver: version CI/cron/workflow units "
                          "(.github/workflows, scheduled tasks)",
                    labels=["version-everything"], body=self.BODY_2464)
        r = self.route(iss)
        self.assertEqual(r["lane"], "modver")
        self.assertNotEqual(r["lane"], "ci")
        self.assertIn("witness-only .github demoted", r["signal"])

    def test_2233_shape_not_offered_as_standalone_github_worker(self):
        iss = issue(2233, "cli(C4): migrate in-repo callers to 'fak dev' "
                          "spellings + bare-spelling ratchet",
                    body=self.BODY_2233)
        r = self.route(iss)
        self.assertNotEqual(r["lane"], "ci",
                            "witness-only .github mention must not feed the ci lane")
        # Routed to a safe owning surface of the migration (or held) — never ci.
        self.assertIn(r["lane"], ("claude", "docs", "tools", None))
        if r["lane"] is not None:
            self.assertIn("witness-only .github demoted", r["signal"])

    def test_github_only_with_no_stronger_binding_still_routes_ci(self):
        # The #978 gate-signal boundary: a workflow-only issue with no stronger
        # binding keeps its authoritative .github path -> ci.
        iss = issue(978, "gate-signal: scheduled security gate is RED",
                    body="the failing gate lives in "
                         "`.github/workflows/security-audit.yml`")
        r = self.route(iss)
        self.assertEqual(r["lane"], "ci")
        self.assertEqual(r["confidence"], "path-confirmed")
        self.assertNotIn("demoted", r["signal"])

    def test_body_lane_declaration_beats_witness_only_github(self):
        # Generic form (no scope, no non-.github path): the body's own `## Lane`
        # declaration is the stronger binding.
        iss = issue(4001, "harden the nightly gate keyspace",
                    body="## Lane\ngateway\n\nthe key space being modeled is "
                         "`.github/workflows/nightly.yml` (witness surface only)")
        r = self.route(iss)
        self.assertEqual(r["lane"], "gateway")
        self.assertIn("body-lane:gateway", r["signal"])
        self.assertIn("witness-only .github demoted", r["signal"])

    def test_bare_internal_path_binding_beats_witness_only_github(self):
        # A concrete bare-rooted `internal/...` path is a stronger binding than a
        # witness-only workflow mention.
        iss = issue(4002, "version the workflow keyspace",
                    body="extends internal/modver/ (the Runner seam); "
                         "`.github/workflows/*` is only the key space being modeled")
        r = self.route(iss)
        self.assertEqual(r["lane"], "modver")
        self.assertEqual(r["confidence"], "path-confirmed")
        self.assertIn("witness-only .github demoted", r["signal"])

    def test_bare_internal_path_alone_gains_no_routing_power(self):
        # The bare-rooted binding probe is scoped to the witness demotion: without
        # a `.github/**` witness hit it must not become a new routing rung.
        iss = issue(4003, "extend the runner seam",
                    body="extends internal/modver/ (the Runner seam)")
        r = self.route(iss)
        self.assertIsNone(r["lane"])
        self.assertEqual(r["confidence"], "none")

    def test_exclusive_path_still_held_despite_github_witness(self):
        # Exclusive holds are decided BEFORE the witness demotion and never weaken.
        iss = issue(4004, "fix: ABI gate",
                    body="touches fak/internal/abi/types.go; "
                         "gate `.github/workflows/abi.yml`")
        r = self.route(iss)
        self.assertIsNone(r["lane"])
        self.assertEqual(r["blocked_lane"], "abi")
        self.assertEqual(r["blocked_policy"], "exclusive")


class ClassLabelBackfillTest(unittest.TestCase):
    """The gated `--apply-labels` backfill diff is pure + idempotent."""

    def test_adds_missing_class_label(self):
        changes = m.plan_class_label_changes(
            [{"number": 5, "class": "infra"}], {5: set()})
        self.assertEqual(changes, [{"number": 5, "add": ["class:infra"], "remove": []}])

    def test_already_correct_is_noop(self):
        changes = m.plan_class_label_changes(
            [{"number": 5, "class": "dev"}], {5: {"class:dev", "bug"}})
        self.assertEqual(changes, [])

    def test_swaps_stale_sibling(self):
        changes = m.plan_class_label_changes(
            [{"number": 5, "class": "frontdoor"}], {5: {"class:infra"}})
        self.assertEqual(changes,
                         [{"number": 5, "add": ["class:frontdoor"], "remove": ["class:infra"]}])

    def test_apply_false_never_writes(self):
        # apply=False must be a pure preview: no gh invoked, applied flag False.
        result = m.apply_class_label_changes(
            Path("."), [{"number": 5, "add": ["class:infra"], "remove": []}], apply=False)
        self.assertEqual(result, {"applied": False, "changed": 1, "errors": []})

    def test_render_label_plan_marks_dry_run(self):
        text = m.render_label_plan(
            [{"number": 5, "add": ["class:infra"], "remove": []}], applied=False)
        self.assertIn("DRY-RUN", text)
        self.assertIn("#5", text)
        self.assertIn("+class:infra", text)


class ScriptsPathRootTest(unittest.TestCase):
    """#2062 Part A: `scripts/` is now a recognized path root, so a `scripts/…`
    deliverable path-confirms the tools lane (dos.toml gives the tools lane ownership
    of `scripts/**`) instead of falling through to an incidental scope/label signal.
    #1477's deliverable IS `scripts/gcp-glm-demo.sh`."""

    # tools owns scripts/** per dos.toml — mirror that ownership in the fixture.
    ST_TREES = {"tools": ["tools/**", "scripts/**"],
                "gateway": ["internal/gateway/**"], "compute": ["internal/compute/**"]}
    ST_LANES = ["tools", "gateway", "compute"]

    def test_scripts_path_extracted(self):
        self.assertEqual(
            m._PATH_RE.findall("ship `scripts/gcp-glm-demo.sh` (plan-by-default)"),
            ["scripts/gcp-glm-demo.sh"])

    def test_scripts_path_confirms_tools_lane(self):
        iss = issue(1477, "feat(serving): one-command GLM-5.2 demo",
                    body="Ship scripts/gcp-glm-demo.sh (plan-by-default, --apply to run).")
        r = m.route_issue(iss, self.ST_LANES, self.ST_TREES)
        self.assertEqual(r["lane"], "tools")
        self.assertEqual(r["confidence"], "path-confirmed")

    def test_embedded_scripts_token_not_falsely_matched(self):
        # `myscripts/` is preceded by a word char -> the `\\b` half rejects it, so a
        # mid-token `scripts` never becomes a false path signal.
        self.assertEqual(m._PATH_RE.findall("the myscripts/x dir"), [])


class GpuServingRoutingTest(unittest.TestCase):
    """#2062 Part A: multi-gpu/moe/model-support aliases route GPU/serving work to a
    real serving lane (compute/model) instead of rotting unrouted or landing in tools
    incidentally — the #1476 GPU-epic cohort's routing bug."""

    def test_multi_gpu_label_routes_compute(self):
        r = route(issue(1478, "serving throughput regression", labels=["multi-gpu"]))
        self.assertEqual(r["lane"], "compute")
        self.assertEqual(r["confidence"], "label")

    def test_moe_scope_routes_compute(self):
        r = route(issue(1479, "feat(moe): expert-parallel router"))
        self.assertEqual(r["lane"], "compute")
        self.assertEqual(r["confidence"], "alias")
        self.assertIn("moe->compute", r["signal"])

    def test_model_support_label_routes_model(self):
        r = route(issue(1482, "add GLM-5.2 architecture", labels=["model-support"]))
        self.assertEqual(r["lane"], "model")
        self.assertEqual(r["confidence"], "label")


class RequiredCapsTest(unittest.TestCase):
    """#2062 Part A: issue_required_caps annotates the hardware a host must declare
    (FLEET_NODE_CAPS) to run an issue — the signal the dispatcher's Part-B capability
    gate consumes to skip-but-not-stop GPU work on a GPU-less host."""

    def test_gpu_labels_require_gpu(self):
        for lab in ("gpu", "cuda", "multi-gpu"):
            with self.subTest(label=lab):
                self.assertEqual(
                    m.issue_required_caps(issue(1, "x", labels=[lab])), ["gpu"])

    def test_gpu_scope_requires_gpu(self):
        self.assertEqual(
            m.issue_required_caps(issue(1, "feat(multi-gpu): shard experts")), ["gpu"])

    def test_accelerator_keyword_requires_gpu(self):
        self.assertEqual(
            m.issue_required_caps(
                issue(1, "provision an h100 serving node", body="needs 8x h100")),
            ["gpu"])
        self.assertEqual(
            m.issue_required_caps(issue(1, "stand up an a100 pool")), ["gpu"])

    def test_plain_cpu_work_requires_nothing(self):
        self.assertEqual(
            m.issue_required_caps(
                issue(1, "fix(compute): tighten a residency fold",
                      body="edit internal/compute/admit.go")), [])

    def test_moe_and_serving_route_but_are_not_hardware_gated(self):
        # moe/agentic-serving ROUTE (to compute/gateway) but are deliberately NOT
        # accelerator-gated — that code is often unit-testable on a GPU-less host, so
        # gating it would falsely strand legitimate local work.
        self.assertEqual(m.issue_required_caps(issue(1, "feat(moe): router")), [])
        self.assertEqual(
            m.issue_required_caps(issue(1, "serve loop", labels=["agentic-serving"])), [])

    def test_route_record_carries_required_caps(self):
        # The annotation rides every routed record (the flat --json issues list the
        # dispatcher's capability gate reads), keyed off the same labels routing used.
        r = route(issue(1478, "serving regression", labels=["multi-gpu"]))
        self.assertEqual(r["required_caps"], ["gpu"])
        r2 = route(issue(1, "fix(gateway): admit", body="see fak/internal/gateway/x.go"))
        self.assertEqual(r2["required_caps"], [])

    def test_hardware_label_requires_hardware_cap(self):
        # #4835: the maintainer-applied `gated/hardware` label is its OWN capability,
        # not an alias for gpu — it also gates non-accelerator physical work (a systemd
        # host, a real crash/reboot box), so a GPU-only node must not claim it.
        self.assertEqual(
            m.issue_required_caps(
                issue(4750, "feat(service): project desired state into systemd",
                      labels=[m.HARDWARE_CAP_LABEL])),
            ["hardware"])

    def test_hardware_and_gpu_caps_are_anded_not_replaced(self):
        # A datacenter box declares `gpu,hardware`; a single-GPU dev node declaring only
        # `gpu` is correctly skipped for a sanctioned multi-node campaign.
        self.assertEqual(
            m.issue_required_caps(
                issue(4784, "perf(glm52): execute routed experts across all GPUs",
                      body="on the lab a100 box", labels=[m.HARDWARE_CAP_LABEL])),
            ["gpu", "hardware"])

    def test_hardware_label_absent_leaves_ordinary_work_ungated(self):
        # Precision guard: the label is the ONLY hardware signal added. Prose that merely
        # CITES a lab node (#5595/#5594 quote one while being locally-runnable work) stays
        # ungated — a false skip silently starves the backlog.
        self.assertEqual(
            m.issue_required_caps(
                issue(5595, "feat(operator): subtract parked emissions before the verdict",
                      body="one emitted ref waits on a live 8-GPU lab witness")),
            [])

    def test_requires_labels_annotate_caps(self):
        cases = [
            ("requires:gpu", ["gpu"]),
            ("requires:gpu:single", ["gpu"]),
            ("requires:gpu:multi", ["gpu"]),
            ("requires:cuda", ["gpu"]),
            ("requires:hardware", ["hardware"]),
            ("requires:lab-hw", ["hardware"]),
            ("requires:lab", ["hardware"]),
            ("requires:metal", ["metal"]),
            ("requires:quota", ["quota"]),
        ]
        for lab, want in cases:
            with self.subTest(label=lab):
                self.assertEqual(
                    m.issue_required_caps(issue(1, "task", labels=[lab])),
                    want)

    def test_requires_none_is_unconstrained(self):
        # requires:none is the explicit unconstrained baseline and suppresses inference
        # even if body prose mentions an accelerator keyword in passing.
        self.assertEqual(
            m.issue_required_caps(
                issue(1, "task mentioning a100 in title",
                      body="prose mentioning an h100 or dgx keyword in passing",
                      labels=["requires:none"])),
            [])

    def test_body_execution_boundary_declaration(self):
        self.assertEqual(
            m.issue_required_caps(
                issue(1, "task",
                      body="Execution boundary: Single GPU (CUDA / Metal) [requires:gpu]")),
            ["gpu"])
        # Multi-GPU and lab hardware combinations
        self.assertEqual(
            m.issue_required_caps(
                issue(2, "task",
                      body="Execution boundary: Multi-GPU (Tensor Parallel / NCCL / DGX) [requires:gpu, requires:hardware]")),
            ["gpu", "hardware"])
        self.assertEqual(
            m.issue_required_caps(
                issue(3, "task",
                      body="Execution boundary: Sanctioned lab hardware (bare metal / reboot host) [requires:hardware]")),
            ["hardware"])
        self.assertEqual(
            m.issue_required_caps(
                issue(4, "task",
                      body="Execution boundary: Cloud quota / burst network [requires:quota]")),
            ["quota"])
        self.assertEqual(
            m.issue_required_caps(
                issue(5, "task",
                      body="Requires: requires:metal")),
            ["metal"])

    def test_body_standard_runner_is_unconstrained(self):
        self.assertEqual(
            m.issue_required_caps(
                issue(1, "task with a100 log parsing",
                      body="Execution boundary: Standard runner (CPU / local / portable) [default]\ninspect a100 error logs")),
            [])
        self.assertEqual(
            m.issue_required_caps(
                issue(2, "task",
                      body="Execution Target: Standard runner (CPU / local / portable) [default]")),
            [])
        self.assertEqual(
            m.issue_required_caps(
                issue(3, "task",
                      body="Requires: requires:none\ninspect dgx cluster logs")),
            [])


class PhantomTreeRegionFidelityTest(unittest.TestCase):
    """#4320 — lease-region-fidelity defect for cmd/fak scope lanes.

    A cmd/fak subsystem issue (`fix(<scope>): ...` whose only concrete work site is
    `cmd/fak/<scope>_*.go`) routes by scope token to a lane that is NOT declared in
    dos.toml [lanes.trees]. The dispatcher's `lane_tree()` fallback then fabricates
    `internal/<scope>/**` (tools/issue_resolve_dispatch.py:3247) as the acquired
    lease region — a phantom tree that does not cover the real cmd/fak work site, so
    same-subsystem workers cannot serialize on a faithful region (LANE_LEASE_HELD
    fidelity) and the arbiter (dispatchorder.TreesOverlap) sees disjoint phantoms.

    Scope: this witnesses the LEASE-GEOMETRY defect only. It is NOT a reproduction of
    the DIRTY_PATH_COLLISION refusals — that guard is pre-lease and text-only
    (dirty_path_collision(text, dirty_paths), issue_resolve_dispatch.py:3385, runs at
    :5559 before the lease acquire at :5643) and is independent of lease geometry.
    See docs/dispatch/cmd-lane-split-plan.md. Two converging in-repo defects:
      (1) _PATH_RE / named_repo_paths do not extract a BARE `cmd/fak/...` path (only
          the `fak/cmd/...` doc-link form), so the router cannot path-confirm `cmd`
          and falls through to scope routing;
      (2) lane_tree()'s fallback fabricates internal/<scope>/** for the undeclared
          scope lane, a region disjoint from the real cmd/fak file.
    """

    # dispatch is intentionally ABSENT from TREES — mirrors reality: no `dispatch`
    # lane is declared in dos.toml, so lane_tree() hits its internal/<lane>/** fallback.
    LANES = ["cmd", "dispatch", "docs", "gateway"]
    TREES = {"cmd": ["cmd/**"], "docs": ["docs/**"], "gateway": ["internal/gateway/**"]}
    # Faithful inline of tools/issue_resolve_dispatch.py:3227-3247 lane_tree():
    #   declared tree if any, else the phantom internal/<lane>/** fallback.
    ISSUE = issue(4347, "fix(dispatch): retune preflight cap",
                  body="The regression is in cmd/fak/dispatch_tick_preflight.go — "
                       "retune the preflight cap.")
    NAMED = "cmd/fak/dispatch_tick_preflight.go"

    def _lane_tree(self, lane):
        return self.TREES.get(lane) or ["internal/%s/**" % lane]

    def test_current_behavior_routes_to_phantom_region(self):
        # Characterization (PASSES today): documents the buggy state so the red test
        # below is provably reproducing the real gap, not a fixture artifact.
        r = m.route_issue(self.ISSUE, self.LANES, self.TREES)
        self.assertEqual(r["lane"], "dispatch")            # scope token wins
        self.assertEqual(r["confidence"], "exact-scope")
        self.assertEqual(m.named_repo_paths(self.ISSUE["body"]), [])  # defect (1)
        self.assertEqual(self._lane_tree("dispatch"), ["internal/dispatch/**"])  # defect (2)
        # The phantom region does NOT cover the named cmd/fak work site; `cmd` would.
        self.assertEqual(m.path_matches_lane(self.NAMED, {"dispatch": ["internal/dispatch/**"]}), [])
        self.assertEqual(m.path_matches_lane(self.NAMED, {"cmd": ["cmd/**"]}), ["cmd"])

    @unittest.expectedFailure  # remove when #4320 (Option C: lane_tree fallback) lands
    def test_lease_region_covers_named_cmdfak_file(self):
        # THE #4320 invariant: an issue whose only concrete work site is a
        # cmd/fak/<scope> file must acquire a lease REGION that covers that file, so
        # the arbiter can serialize the real collision instead of leaning on the
        # dirty-path backstop. RED today (region is the phantom internal/dispatch/**);
        # GREEN once lane_tree()'s fallback emits cmd/fak/<scope>_*.go for cmd-scoped
        # scope lanes. When this stops being an expected failure, delete the marker.
        r = m.route_issue(self.ISSUE, self.LANES, self.TREES)
        region = self._lane_tree(r["lane"])
        self.assertEqual(
            m.path_matches_lane(self.NAMED, {r["lane"]: region}), [r["lane"]],
            "lane %r region %r does not cover the named cmd/fak file %r"
            % (r["lane"], region, self.NAMED))


class BoundedLabelReconciliationTest(unittest.TestCase):
    def test_over_cap_progress_is_deterministic_and_bounded(self):
        routes = [
            {"number": number, "class": m.CLASS_DEV}
            for number in range(668, 0, -1)
        ]
        labels = {number: set() for number in range(1, 669)}

        first_plan = m.plan_class_label_changes(routes, labels)
        first, remaining = m.bound_class_label_changes(first_plan, 200)

        self.assertEqual([change["number"] for change in first], list(range(1, 201)))
        self.assertEqual(len(first), 200)
        self.assertEqual(remaining, 468)

        for change in first:
            labels[change["number"]].update(change["add"])
            labels[change["number"]].difference_update(change["remove"])
        second_plan = m.plan_class_label_changes(routes, labels)
        second, remaining = m.bound_class_label_changes(second_plan, 200)

        self.assertEqual([change["number"] for change in second], list(range(201, 401)))
        self.assertEqual(len(second), 200)
        self.assertEqual(remaining, 268)

    def test_zero_limit_preserves_unbounded_manual_reconciliation(self):
        changes = [{"number": number, "add": [], "remove": []}
                   for number in range(1, 669)]
        selected, remaining = m.bound_class_label_changes(changes, 0)
        self.assertEqual(selected, changes)
        self.assertEqual(remaining, 0)


class PairedFlagTest(unittest.TestCase):
    def test_apply_labels_write_requires_apply_labels(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            m.main(["--apply-labels-write"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--apply-labels-write requires --apply-labels", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
