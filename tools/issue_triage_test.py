#!/usr/bin/env python3
"""Hermetic tests for tools/issue_triage.py — the /issue-triage ranking helper.

issue_triage.py drives the fleet's "do next" issue order (the backlog DOS drains),
but shipped with a SKILL.md wrapper and NO sibling test, so its pure scoring /
clustering / action contract was unguarded. These tests pin that contract with a
fixed clock and synthetic issue dicts; the only subprocess boundary (fetch_issues)
is monkeypatched, so no gh / network / git is touched (runs on the Windows host
where native go test is blocked). The importlib load() mirrors the cluster pattern.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "issue_triage.py"
NOW = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)


def load():
    spec = importlib.util.spec_from_file_location("issue_triage", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()


def _iso(days_ago: int) -> str:
    return (NOW - dt.timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")


def _issue(number: int, *, title: str = "t", labels=None, assignees=None,
           created_days_ago: int = 10, idle_days_ago: int = 0) -> dict:
    return {
        "number": number,
        "title": title,
        "url": f"https://x/{number}",
        "state": "OPEN",
        "labels": [{"name": n} for n in (labels or [])],
        "assignees": [{"login": a} for a in (assignees or [])],
        "author": {"login": "u"},
        "milestone": None,
        "comments": 0,
        "createdAt": _iso(created_days_ago),
        "updatedAt": _iso(idle_days_ago),
    }


class ClassifyScoreTest(unittest.TestCase):
    def test_orphan_p0_bug_score_and_tags(self):
        # P0 bug, no in-progress, no assignee, fresh: 1000 + 300(orphan) + 40(bug) + 0(idle).
        g = m.classify(_issue(1, labels=["priority/P0", "bug"], idle_days_ago=0), NOW, {})
        self.assertEqual(g["score"], 1340)
        self.assertIn("orphan", g["tags"])
        self.assertIn("needs-area", g["tags"])        # no area label
        self.assertNotIn("needs-priority", g["tags"])  # has P0
        self.assertNotIn("needs-kind", g["tags"])      # has bug

    def test_fresh_question_penalty(self):
        # No priority (base 60), question, idle 0 -> -200 fresh-question penalty.
        g = m.classify(_issue(2, labels=["question"], idle_days_ago=0), NOW, {})
        self.assertEqual(g["score"], 60 - 200)
        self.assertNotIn("dormant-question", g["tags"])

    def test_dormant_question_tag_and_no_penalty(self):
        # question idle >= Q_IDLE_DAYS (30) -> dormant-question tag, no -200.
        g = m.classify(_issue(3, labels=["question"], idle_days_ago=40), NOW, {})
        self.assertIn("dormant-question", g["tags"])
        self.assertEqual(g["score"], 60 + 40)  # base + min(idle,90); no fresh penalty

    def test_bare_issue_gets_all_missing_tags(self):
        g = m.classify(_issue(4, labels=[], idle_days_ago=0), NOW, {})
        for tag in ("needs-priority", "needs-kind", "needs-area", "bare"):
            self.assertIn(tag, g["tags"])

    def test_stale_tag_for_idle_non_inprogress(self):
        g = m.classify(_issue(5, labels=["priority/P2"], idle_days_ago=70), NOW, {})
        self.assertIn("stale", g["tags"])

    def test_in_progress_suppresses_orphan_and_stale(self):
        g = m.classify(_issue(6, labels=["priority/P0", "in-progress"], idle_days_ago=70), NOW, {})
        self.assertNotIn("orphan", g["tags"])
        self.assertNotIn("stale", g["tags"])


class DupClusterTest(unittest.TestCase):
    def test_similar_titles_cluster_unrelated_excluded(self):
        issues = [
            _issue(1, title="tokenizer cache invalidation resize regression"),
            _issue(2, title="tokenizer cache invalidation resize crash"),
            _issue(3, title="documentation gpu benchmark gallery layout"),
        ]
        clusters = m.dup_clusters(issues)
        self.assertIn(1, clusters)
        self.assertIn(2, clusters)
        self.assertEqual(clusters[1], clusters[2])  # same cluster id
        self.assertNotIn(3, clusters)               # unrelated -> not clustered

    def test_jaccard_and_tokens(self):
        self.assertEqual(m._jaccard(set(), {"a"}), 0.0)
        toks = m._title_tokens("fix(gateway): tool call timeout")
        self.assertIn("gateway", toks)        # scope captured
        self.assertNotIn("fix", toks)         # stopword stripped


class ActionsTest(unittest.TestCase):
    def test_dormant_question_yields_close_cmd(self):
        rows = [m.classify(_issue(3, labels=["question"], idle_days_ago=40), NOW, {})]
        acts = m.build_actions(rows)
        self.assertEqual(acts[0]["kind"], "close-dormant-question")
        self.assertIn("gh issue close 3", acts[0]["cmd"])

    def test_stale_p2_yields_mark_stale(self):
        rows = [m.classify(_issue(5, labels=["priority/P2"], idle_days_ago=70), NOW, {})]
        acts = m.build_actions(rows)
        self.assertEqual(acts[0]["kind"], "mark-stale")
        self.assertIsNotNone(acts[0]["cmd"])

    def test_p0_with_tags_is_review_only(self):
        # An orphan P0 has tags but no mechanical cmd -> review, cmd None.
        rows = [m.classify(_issue(1, labels=["priority/P0", "bug"], idle_days_ago=0), NOW, {})]
        acts = m.build_actions(rows)
        self.assertEqual(acts[0]["kind"], "review")
        self.assertIsNone(acts[0]["cmd"])


class ReportTest(unittest.TestCase):
    def test_rows_sorted_descending_by_score(self):
        issues = [
            _issue(10, labels=["question"], idle_days_ago=0),        # low (penalty)
            _issue(11, labels=["priority/P0", "bug"], idle_days_ago=0),  # high
        ]
        rep = m.build_report(issues, NOW)
        scores = [r["score"] for r in rep["rows"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(rep["rows"][0]["number"], 11)
        self.assertEqual(rep["counts"]["open"], 2)

    def test_main_json_exit0_with_monkeypatched_fetch(self):
        orig = m.fetch_issues
        m.fetch_issues = lambda: [_issue(1, labels=["priority/P0", "bug"])]
        try:
            rc = m.main(["--json", "--as-of", "2026-06-01"])
        finally:
            m.fetch_issues = orig
        self.assertEqual(rc, 0)

    def test_main_infra_error_exit2(self):
        orig = m.fetch_issues

        def boom():
            raise RuntimeError("gh not authed")

        m.fetch_issues = boom
        try:
            rc = m.main(["--json"])
        finally:
            m.fetch_issues = orig
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
