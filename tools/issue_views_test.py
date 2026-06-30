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
REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = REPO_ROOT / ".github" / "issue-views.json"
SLACK_WATCHDOG = REPO_ROOT / ".github" / "workflows" / "slack-watchdog.yml"

TRIAGE_EXCLUDES = ["idea-scout", "needs-triage", "triage-only", "triage_only", "guard-complaint"]

DISPATCH_VIEW_SLUGS = {
    "ready-leaves",
    "p0-p1",
    "help-wanted",
    "good-first-issue",
    "agentic-serving",
    "substrate",
    "trust-floor",
    "gpu",
    "current",
    "generation-now",
    "generation-next",
    "generation-second-next",
    "generation-future",
    "m1-durable-sessions",
    "m2-kv-cache",
    "m3-serving",
    "m4-decode",
    "m5-benchmarks",
    "m6-agentic-loop",
    "m7-release",
    "m8-observability",
    "m9-dispatch-fleet",
    "m10-model-support",
    "m11-substrate",
}

GENERATION_VIEW_BINDINGS = {
    "generation-now": ("gen/now", 'milestone:"Generation G0 - Now / Immediate"'),
    "generation-next": ("gen/next", 'milestone:"Generation G1 - Next Gen"'),
    "generation-second-next": ("gen/second-next", 'milestone:"Generation G2 - Second Next Gen"'),
    "generation-future": ("gen/future", 'milestone:"Generation G3 - Future"'),
}

ISSUE_CREATE_PRODUCERS = {
    ".github/workflows/slack-watchdog.yml": [
        "--label needs-triage --label triage-only",
        "dispatchability: \\`triage_only\\`",
    ],
    "cmd/fak/dispatchaudit.go": [
        "dispatchability: `triage_only`",
        '"needs-triage", "triage-only"',
    ],
    "cmd/fak/taskmgr.go": [
        "ReviewHandoffWithOptions",
        "StrictScope:   true",
    ],
    "internal/dogfoodissues/dogfoodissues.go": [
        "issuecontract.ReviewCandidate",
        '"Priority context"',
    ],
    "internal/ideascout/ideascout.go": [
        "TriageOnlyLabel = \"triage-only\"",
        "dispatchability: `triage_only`",
    ],
    "internal/issuecatalog/issuecatalog.go": [
        "issuecontract.ReviewCandidate",
        "IssueBody",
    ],
    "internal/learningdebt/learningdebt.go": [
        "defaultTriageLabels = []string{\"needs-triage\", \"triage-only\"}",
        "dispatchability: `triage_only`",
    ],
    "internal/maturity/issues.go": [
        "maturityTriageLabels = []string{\"needs-triage\", \"triage-only\"}",
        "dispatchability: `triage_only`",
    ],
    "tools/bench_signal.py": [
        "check_issue_contract(",
        "issue_contract_draft(",
    ],
    "tools/dispatch_log_audit.py": [
        "TRIAGE_LABELS = [\"needs-triage\", \"triage-only\"]",
        "Dispatchability:** `triage_only`",
    ],
    "tools/dogfood_issue_sync.py": [
        "TRIAGE_LABELS = [\"needs-triage\", \"triage-only\"]",
        "dispatchability: `triage_only`",
    ],
    "tools/gate_signal.py": [
        "check_issue_contract(",
        "issue_contract_draft(",
    ],
    "tools/idea_scout.py": [
        "TRIAGE_ONLY_LABEL = \"triage-only\"",
        "dispatchability: `triage_only`",
    ],
    "tools/learning_debt_dispatch.py": [
        "TRIAGE_LABELS = [\"needs-triage\", \"triage-only\"]",
        "Dispatchability:** `triage_only`",
    ],
    "tools/score_signal.py": [
        "check_issue_contract(",
        "issue_contract_draft(",
    ],
}


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

    def test_dispatch_views_exclude_triage_only_labels(self):
        cfg = m.load_config(SHIPPED_CONFIG)
        vm = m.view_map(cfg)
        self.assertLessEqual(DISPATCH_VIEW_SLUGS, set(vm))
        for slug in sorted(DISPATCH_VIEW_SLUGS):
            query = vm[slug]["query"]
            for label in TRIAGE_EXCLUDES:
                self.assertIn(f"-label:{label}", query, f"{slug} must exclude {label}")

    def test_generation_views_bind_stream_label_and_milestone(self):
        cfg = m.load_config(SHIPPED_CONFIG)
        vm = m.view_map(cfg)
        for slug, (stream_label, milestone) in GENERATION_VIEW_BINDINGS.items():
            self.assertIn(slug, vm)
            query = vm[slug]["query"]
            self.assertIn("label:generation", query)
            self.assertIn(f"label:{stream_label}", query)
            self.assertIn(milestone, query)
            self.assertIn("-label:epic", query)
            self.assertIn("no:assignee", query)

    def test_slack_watchdog_files_triage_only_issues(self):
        workflow = SLACK_WATCHDOG.read_text(encoding="utf-8")
        self.assertIn("gh label create needs-triage", workflow)
        self.assertIn("gh label create triage-only", workflow)
        self.assertIn("--label needs-triage --label triage-only", workflow)
        self.assertIn("--add-label needs-triage --add-label triage-only", workflow)
        self.assertIn("--body-file issue-body.md --add-label", workflow)
        self.assertIn("dispatchability: \\`triage_only\\`", workflow)

    def test_issue_create_producers_are_contract_or_triage_gated(self):
        found = set()
        for root in [
            REPO_ROOT / "tools",
            REPO_ROOT / "cmd" / "fak",
            REPO_ROOT / "internal",
            REPO_ROOT / ".github" / "workflows",
        ]:
            for path in root.rglob("*"):
                if path.suffix not in {".py", ".go", ".yml", ".yaml"}:
                    continue
                if path.name.endswith("_test.py") or path.name.endswith("_test.go"):
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                creates = False
                if path.suffix in {".py", ".go"}:
                    creates = '"issue", "create"' in text or "'issue', 'create'" in text
                else:
                    creates = "gh issue create" in text
                if creates:
                    found.add(path.relative_to(REPO_ROOT).as_posix())

        self.assertEqual(found, set(ISSUE_CREATE_PRODUCERS))
        for rel, markers in sorted(ISSUE_CREATE_PRODUCERS.items()):
            text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
            missing = [marker for marker in markers if marker not in text]
            self.assertEqual(missing, [], f"{rel} missing issue-create gate markers")


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
