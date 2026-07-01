#!/usr/bin/env python3
"""Hermetic tests for tools/issue_contract_repair.py.

Every shell-out (gh fetch, `fak issue contract`, lane router) is stubbed on
the module; nothing live runs. Covers: reason->kind classification (matches
cmd/fak/issue_contract.go's issueContractRepairKinds), per-kind row building
(template/route/scope), the manifest/actions schema, and a static assertion
that this module never references a GitHub-mutating `gh` subcommand.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "issue_contract_repair.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("issue_contract_repair", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _issue(number, title="an issue", body="", labels=None):
    return {"number": number, "title": title, "body": body,
            "labels": [{"name": l} for l in (labels or [])]}


def _contract(*, ok=False, score=0, reasons=None, missing_fields=None, lane=None):
    review = {"ok": ok, "reasons": reasons or [], "missing_fields": missing_fields or []}
    if lane:
        review["lane"] = lane
    return {"ok": ok, "unavailable": False, "score": score,
            "spine_priority": 0, "review": review}


class RepairKindsTest(unittest.TestCase):
    def test_split_reasons(self) -> None:
        mod = load()
        self.assertEqual(mod.repair_kinds(["ISSUE_NOT_DISPATCH_LEAF"]), ["split"])
        self.assertEqual(mod.repair_kinds(["ISSUE_OVERSIZED_EXPECTED_STEPS"]), ["split"])

    def test_multi_reason_order_preserving_dedup(self) -> None:
        mod = load()
        reasons = ["ISSUE_AGENT_CONTEXT_INCOMPLETE", "ISSUE_NOISE_CONTROL_INCOMPLETE",
                   "ISSUE_SCOPE_INCOMPLETE", "ISSUE_UNROUTED"]
        # both AGENT_CONTEXT_INCOMPLETE and NOISE_CONTROL_INCOMPLETE fold to "noise",
        # so it appears once, first -- matches issueContractRepairKinds' add() dedup.
        self.assertEqual(mod.repair_kinds(reasons), ["noise", "scope", "route"])

    def test_unmapped_reason_falls_back_to_other(self) -> None:
        mod = load()
        self.assertEqual(mod.repair_kinds(["SOMETHING_NEW"]), ["other"])

    def test_empty_reasons_falls_back_to_other(self) -> None:
        mod = load()
        self.assertEqual(mod.repair_kinds([]), ["other"])

    def test_template_reason(self) -> None:
        mod = load()
        self.assertEqual(mod.repair_kinds(["ISSUE_UNEXPANDED_TEMPLATE"]), ["template"])

    def test_primary_kind_prefers_split_over_route(self) -> None:
        mod = load()
        self.assertEqual(mod.primary_kind(["route", "split"]), "split")
        self.assertEqual(mod.primary_kind(["other", "scope"]), "scope")


class FieldScaffoldTest(unittest.TestCase):
    def test_known_fields_get_their_question_not_invented_content(self) -> None:
        mod = load()
        out = mod.field_scaffold(["done_condition", "witness"])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["field"], "done_condition")
        self.assertIn("observable state", out[0]["question"])
        self.assertEqual(out[1]["field"], "witness")
        self.assertIn("evidence", out[1]["question"])

    def test_unknown_field_gets_generic_question(self) -> None:
        mod = load()
        out = mod.field_scaffold(["mystery_field"])
        self.assertEqual(out[0]["field"], "mystery_field")
        self.assertIn("mystery_field", out[0]["question"])


class BuildRepairRowTest(unittest.TestCase):
    def test_passing_contract_returns_none(self) -> None:
        mod = load()
        mod.ird.issue_contract_review = lambda *a, **k: _contract(ok=True, score=100)
        row = mod.build_repair_row(ROOT, _issue(1), [], {})
        self.assertIsNone(row)

    def test_scope_kind_lists_missing_fields_only(self) -> None:
        mod = load()
        mod.ird.issue_contract_review = lambda *a, **k: _contract(
            score=8, reasons=["ISSUE_SCOPE_INCOMPLETE"],
            missing_fields=["done_condition", "witness"])
        row = mod.build_repair_row(ROOT, _issue(1207, "fix thing"), [], {})
        self.assertIsNotNone(row)
        self.assertEqual(row["kind"], "scope")
        self.assertFalse(row["ready"])
        self.assertEqual([f["field"] for f in row["missing_fields"]],
                         ["done_condition", "witness"])
        self.assertIsNone(row["proposed_lane"])
        self.assertIsNone(row["proposed_header"])

    def test_route_kind_with_confident_lane_proposes_it(self) -> None:
        mod = load()
        mod.ird.issue_contract_review = lambda *a, **k: _contract(
            score=0, reasons=["ISSUE_UNROUTED"])
        mod.ilr.route_issue = lambda *a, **k: {"lane": "docs", "confidence": "exact-scope"}
        row = mod.build_repair_row(ROOT, _issue(1496, "docs: fix typo"), [], {})
        self.assertEqual(row["kind"], "route")
        self.assertEqual(row["proposed_lane"], "docs")
        self.assertEqual(row["route_confidence"], "exact-scope")

    def test_route_kind_with_no_confident_lane_stays_unset(self) -> None:
        mod = load()
        mod.ird.issue_contract_review = lambda *a, **k: _contract(
            score=0, reasons=["ISSUE_UNROUTED"])
        mod.ilr.route_issue = lambda *a, **k: {"lane": None, "confidence": "none"}
        row = mod.build_repair_row(ROOT, _issue(1612), [], {})
        self.assertEqual(row["kind"], "route")
        self.assertIsNone(row["proposed_lane"])

    def test_template_kind_ready_when_header_computed(self) -> None:
        mod = load()
        mod.ird.issue_contract_review = lambda *a, **k: _contract(
            score=25, reasons=["ISSUE_UNEXPANDED_TEMPLATE"])
        mod.template_repair_plan = lambda *a, **k: {
            "issue_number": 1545, "proposed_normalized_header": "N=27 Lane=api/provider"}
        row = mod.build_repair_row(ROOT, _issue(1545), [], {})
        self.assertEqual(row["kind"], "template")
        self.assertTrue(row["ready"])
        self.assertEqual(row["proposed_header"], "N=27 Lane=api/provider")

    def test_template_kind_not_ready_when_no_plan_found(self) -> None:
        mod = load()
        mod.ird.issue_contract_review = lambda *a, **k: _contract(
            score=25, reasons=["ISSUE_UNEXPANDED_TEMPLATE"])
        mod.template_repair_plan = lambda *a, **k: None
        row = mod.build_repair_row(ROOT, _issue(1545), [], {})
        self.assertEqual(row["kind"], "template")
        self.assertFalse(row["ready"])
        self.assertIsNone(row["proposed_header"])


class ManifestTest(unittest.TestCase):
    def test_manifest_schema_and_counts(self) -> None:
        mod = load()
        mod.ilr.lane_taxonomy = lambda *a, **k: (["docs", "tools"], {})
        mod.fetch_open_issues = lambda *a, **k: [
            _issue(1207, "a"), _issue(1852, "b"), _issue(2000, "c")]

        def fake_review(root, issue, number):
            if number == 2000:
                return _contract(ok=True, score=100)
            return _contract(score=8, reasons=["ISSUE_SCOPE_INCOMPLETE"],
                             missing_fields=["done_condition"])
        mod.ird.issue_contract_review = fake_review

        manifest = mod.build_manifest(ROOT, lane=None, limit=50, as_of="2026-07-01")
        self.assertEqual(manifest["schema"], "fak.issue-contract-repair.v1")
        self.assertEqual(manifest["counts"]["candidates_examined"], 3)
        self.assertEqual(manifest["counts"]["needs_repair"], 2)
        self.assertEqual([r["number"] for r in manifest["issues"]], [1207, 1852])
        self.assertEqual(manifest["counts"]["by_kind"], {"scope": 2})

    def test_lane_filter_keeps_only_matching_or_blocked_lane(self) -> None:
        mod = load()
        mod.ilr.lane_taxonomy = lambda *a, **k: (["docs", "tools"], {})
        mod.fetch_open_issues = lambda *a, **k: [
            _issue(1, "a"), _issue(2, "b"), _issue(3, "c")]

        def fake_route(issue, *a, **k):
            return {1: {"lane": "docs"}, 2: {"lane": "tools"},
                    3: {"lane": None, "blocked_lane": "docs"}}[issue["number"]]
        mod.ilr.route_issue = fake_route
        mod.ird.issue_contract_review = lambda *a, **k: _contract(
            score=0, reasons=["ISSUE_SCOPE_INCOMPLETE"], missing_fields=["done_condition"])

        manifest = mod.build_manifest(ROOT, lane="docs", limit=50, as_of="2026-07-01")
        self.assertEqual([r["number"] for r in manifest["issues"]], [1, 3])


class ActionsAndRenderTest(unittest.TestCase):
    def test_actions_cmd_always_none(self) -> None:
        mod = load()
        manifest = {"issues": [
            {"number": 1, "kind": "scope", "ready": False,
             "reasons": ["ISSUE_SCOPE_INCOMPLETE"], "next_action": "do x"},
            {"number": 2, "kind": "template", "ready": True,
             "reasons": ["ISSUE_UNEXPANDED_TEMPLATE"], "next_action": "do y"},
        ]}
        actions = mod.build_actions(manifest)
        self.assertTrue(all(a["cmd"] is None for a in actions))
        self.assertEqual(actions[0]["reason"], "ISSUE_SCOPE_INCOMPLETE")

    def test_render_markdown_includes_counts_and_rows(self) -> None:
        mod = load()
        manifest = {
            "as_of": "2026-07-01", "lane": "docs",
            "counts": {"candidates_examined": 2, "needs_repair": 1, "ready": 0,
                      "by_kind": {"scope": 1}},
            "issues": [{"number": 1207, "kind": "scope", "ready": False,
                       "score": 8, "title": "fix thing"}],
        }
        rendered = mod.render_markdown(manifest)
        self.assertIn("issue-contract repairs", rendered.lower())
        self.assertIn("#1207", rendered)
        self.assertIn("scope", rendered)


class NoGitHubMutationTest(unittest.TestCase):
    def test_module_never_references_a_mutating_gh_subcommand(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIsNone(
            re.search(r"gh[\"']?\s*,\s*[\"']issue[\"']\s*,\s*[\"'](edit|comment|close|assign)[\"']",
                     source),
            "issue_contract_repair.py must never construct a GitHub-mutating gh command")
        for verb in ("edit", "comment", "close", "assign"):
            self.assertNotIn(f'"issue", "{verb}"', source)
            self.assertNotIn(f"'issue', '{verb}'", source)


if __name__ == "__main__":
    unittest.main()
