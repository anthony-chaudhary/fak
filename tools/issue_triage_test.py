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
        g = m.classify(_issue(1, labels=["priority/P0", "bug"], idle_days_ago=0), NOW)
        self.assertEqual(g["score"], 1340)
        self.assertIn("orphan", g["tags"])
        self.assertIn("needs-area", g["tags"])        # no area label
        self.assertNotIn("needs-priority", g["tags"])  # has P0
        self.assertNotIn("needs-kind", g["tags"])      # has bug

    def test_fresh_question_penalty(self):
        # No priority (base 60), question, idle 0 -> -200 fresh-question penalty.
        g = m.classify(_issue(2, labels=["question"], idle_days_ago=0), NOW)
        self.assertEqual(g["score"], 60 - 200)
        self.assertNotIn("dormant-question", g["tags"])

    def test_dormant_question_tag_and_no_penalty(self):
        # question idle >= Q_IDLE_DAYS (30) -> dormant-question tag, no -200.
        g = m.classify(_issue(3, labels=["question"], idle_days_ago=40), NOW)
        self.assertIn("dormant-question", g["tags"])
        self.assertEqual(g["score"], 60 + 40)  # base + min(idle,90); no fresh penalty

    def test_bare_issue_gets_all_missing_tags(self):
        g = m.classify(_issue(4, labels=[], idle_days_ago=0), NOW)
        for tag in ("needs-priority", "needs-kind", "needs-area", "needs-class",
                    "needs-milestone", "bare"):
            self.assertIn(tag, g["tags"])

    def test_needs_class_when_no_class_label(self):
        # A fully-area-labeled issue with no class:* label still needs the work-class
        # axis (infra vs dev vs front-door) — a SEPARATE gap from needs-area.
        g = m.classify(_issue(9, labels=["priority/P2", "bug", "gpu"], idle_days_ago=0),
                       NOW)
        self.assertIn("needs-class", g["tags"])
        self.assertNotIn("needs-area", g["tags"])  # gpu is an area

    def test_no_needs_class_when_class_label_present(self):
        g = m.classify(_issue(10, labels=["priority/P2", "bug", "gpu", "class:infra"],
                              idle_days_ago=0), NOW)
        self.assertNotIn("needs-class", g["tags"])

    def test_needs_milestone_when_absent(self):
        # A fully-labeled issue with no milestone is still a triage gap: it never
        # rolls up onto the roadmap. needs-milestone fires; score is unchanged
        # (the tag surfaces attention, it does not reweight the ranking).
        i = _issue(7, labels=["priority/P2", "bug", "gpu"], idle_days_ago=0)
        g = m.classify(i, NOW)
        self.assertIn("needs-milestone", g["tags"])
        self.assertEqual(g["score"], 150 + 40)  # P2 base + bug; no milestone penalty

    def test_no_needs_milestone_when_present(self):
        i = _issue(8, labels=["priority/P1", "bug", "gpu"], idle_days_ago=0)
        i["milestone"] = {"title": "Generation G0 - Now / Immediate"}
        g = m.classify(i, NOW)
        self.assertNotIn("needs-milestone", g["tags"])
        self.assertEqual(g["milestone"], "Generation G0 - Now / Immediate")

    def test_stale_tag_for_idle_non_inprogress(self):
        g = m.classify(_issue(5, labels=["priority/P2"], idle_days_ago=70), NOW)
        self.assertIn("stale", g["tags"])

    def test_in_progress_suppresses_orphan_and_stale(self):
        g = m.classify(_issue(6, labels=["priority/P0", "in-progress"], idle_days_ago=70), NOW)
        self.assertNotIn("orphan", g["tags"])
        self.assertNotIn("stale", g["tags"])


class DependencyTaxonomyTest(unittest.TestCase):
    """The directional blocker pair (#3563): `blocked-by` / `blocks`. The names are
    lowercase-kebab like the rest of the fleet taxonomy, and each is a label FAMILY
    whose scoped `:#N` / `/N` value names the counterpart issue. These tests cover the
    DECLARATION and recognition only: no reader consumes the edges yet, so there is no
    end-to-end graph witness to assert against."""

    def test_pair_is_declared_lowercase_kebab(self):
        self.assertEqual(m.DEPENDENCY, {"blocked-by", "blocks"})
        for name in m.DEPENDENCY:
            self.assertEqual(name, name.lower())
            self.assertRegex(name, r"^[a-z]+(?:-[a-z]+)*$")
        # No bare `blocked` state label — direction is the whole point.
        self.assertNotIn("blocked", m.DEPENDENCY)

    def test_scoped_and_bare_family_labels_are_recognised(self):
        got = m.dependency_labels({"blocked-by:#3224", "blocks/3600", "blocked-by",
                                   "priority/P2", "bug"})
        self.assertEqual(got, ["blocked-by", "blocked-by:#3224", "blocks/3600"])

    def test_blocked_by_human_is_a_different_label(self):
        # The existing dispatch human-block hold must not be read as a graph edge.
        self.assertEqual(m.dependency_labels({"blocked-by-human"}), [])

    def test_classify_surfaces_dependency_without_reweighting(self):
        plain = m.classify(_issue(40, labels=["priority/P2"], idle_days_ago=0), NOW)
        dep = m.classify(_issue(41, labels=["priority/P2", "blocked-by:#3224"],
                                idle_days_ago=0), NOW)
        self.assertEqual(plain["dependency"], [])
        self.assertEqual(dep["dependency"], ["blocked-by:#3224"])
        self.assertEqual(dep["score"], plain["score"])   # surfacing, not ranking
        self.assertEqual(dep["tags"], plain["tags"])     # and not a triage gap

    def test_config_can_override_the_pair(self):
        import json
        import tempfile
        orig = m.DEPENDENCY
        try:
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "cfg.json"
                p.write_text(json.dumps({"dependency": ["needs"]}), encoding="utf-8")
                m._load_config(str(p))
                self.assertEqual(m.DEPENDENCY, {"needs"})
        finally:
            m.DEPENDENCY = orig

    def test_requires_is_declared(self):
        self.assertEqual(m.REQUIRES, {"requires"})

    def test_config_can_override_requires(self):
        import json
        import tempfile
        orig = m.REQUIRES
        try:
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "cfg.json"
                p.write_text(json.dumps({"requires": ["demands"]}), encoding="utf-8")
                m._load_config(str(p))
                self.assertEqual(m.REQUIRES, {"demands"})
        finally:
            m.REQUIRES = orig


class ActionsTest(unittest.TestCase):
    def test_dormant_question_yields_close_cmd(self):
        rows = [m.classify(_issue(3, labels=["question"], idle_days_ago=40), NOW)]
        acts = m.build_actions(rows)
        self.assertEqual(acts[0]["kind"], "close-dormant-question")
        self.assertIn("gh issue close 3", acts[0]["cmd"])

    def test_stale_p2_yields_mark_stale(self):
        rows = [m.classify(_issue(5, labels=["priority/P2"], idle_days_ago=70), NOW)]
        acts = m.build_actions(rows)
        self.assertEqual(acts[0]["kind"], "mark-stale")
        self.assertIsNotNone(acts[0]["cmd"])

    def test_p0_with_tags_is_review_only(self):
        # An orphan P0 has tags but no mechanical cmd -> review, cmd None.
        rows = [m.classify(_issue(1, labels=["priority/P0", "bug"], idle_days_ago=0), NOW)]
        acts = m.build_actions(rows)
        self.assertEqual(acts[0]["kind"], "review")
        self.assertIsNone(acts[0]["cmd"])

    def test_needs_class_is_mechanical_backfill_not_review(self):
        # An issue whose ONLY gap is class:* is fully mechanical: no REVIEW row, and
        # the cohort gets one backfill-class action pointing at the lane router.
        rows = [m.classify(
            _issue(11, labels=["priority/P2", "bug", "gpu"], idle_days_ago=0), NOW)]
        # (also missing a milestone, so review still fires for that judgment gap;
        #  build one with a milestone to isolate the class-only case)
        rows[0]["tags"] = ["needs-class"]
        acts = m.build_actions(rows)
        kinds = [a["kind"] for a in acts]
        self.assertIn("backfill-class", kinds)
        self.assertNotIn("review", kinds)  # a class-only gap is not a judgment call
        backfill = next(a for a in acts if a["kind"] == "backfill-class")
        self.assertIsNotNone(backfill["cmd"])
        self.assertIn("issue_lane_router.py", backfill["cmd"])
        self.assertIn("--apply-labels-write", backfill["cmd"])
        self.assertEqual(backfill["issues"], [11])
        self.assertIsNone(backfill["number"])

    def test_review_reason_excludes_needs_class(self):
        # An issue with a real judgment gap AND a missing class keeps its REVIEW row,
        # but needs-class is stripped from the reason (the batch action handles it).
        rows = [m.classify(_issue(1, labels=["priority/P0", "bug"], idle_days_ago=0), NOW)]
        acts = m.build_actions(rows)
        review = next(a for a in acts if a["kind"] == "review")
        self.assertIn("needs-area", review["reason"])
        self.assertIn("orphan", review["reason"])
        self.assertNotIn("needs-class", review["reason"])
        self.assertTrue(any(a["kind"] == "backfill-class" for a in acts))


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

    def test_needs_milestone_counted_and_rendered(self):
        # One milestoned, one not -> count is 1, and the bucket renders in markdown.
        milestoned = _issue(20, labels=["priority/P2", "bug", "gpu"])
        milestoned["milestone"] = {"title": "M"}
        bare_ms = _issue(21, labels=["priority/P2", "bug", "gpu"])  # milestone None
        rep = m.build_report([milestoned, bare_ms], NOW)
        self.assertEqual(rep["counts"]["needs_milestone"], 1)
        md = m.render_md(rep, "2026-06-01")
        self.assertIn("needs-milestone 1", md)
        self.assertIn("Needs a milestone", md)

    def test_backfill_class_action_rendered_in_markdown(self):
        # The mechanical class backfill surfaces in the proposed-actions block with
        # its runnable command, not as a review-only row.
        rep = m.build_report([_issue(30, labels=["priority/P2", "bug", "gpu"])], NOW)
        md = m.render_md(rep, "2026-06-01")
        self.assertIn("backfill-class", md)
        self.assertIn("issue_lane_router.py --apply-labels", md)

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

    def test_truncated_prefix_is_typed_and_refused(self):
        census = m.reconcile_census(
            scope="repository_issues", state="open", fetched_count=500,
            total_count=2222, snapshot_age_seconds=0, includes_pull_requests=False,
        )
        self.assertFalse(census["page_complete"])
        self.assertEqual(census["reconciliation"], "pagination_truncated")
        with self.assertRaisesRegex(m.IncompleteRankingError, "pagination_truncated"):
            m.build_report([_issue(1)], NOW, census=census)

    def test_complete_fixture_above_500_keeps_all_rows_and_bounds_repairs(self):
        issues = [_issue(n, title=f"unclassified backlog item {n:04d}")
                  for n in range(1, 526)]
        census = m.reconcile_census(
            scope="repository_issues", state="open", fetched_count=len(issues),
            total_count=len(issues), snapshot_age_seconds=0,
            includes_pull_requests=False,
        )
        report = m.build_report(issues, NOW, census=census, repair_batch_size=40)
        self.assertEqual(len(report["rows"]), 525)
        self.assertEqual(report["census"]["fetched_count"], 525)
        self.assertTrue(report["census"]["page_complete"])
        self.assertTrue(report["repair_batches"])
        for batch in report["repair_batches"]:
            self.assertGreater(len(batch["issues"]), 0)
            self.assertLessEqual(len(batch["issues"]), 40)
            self.assertTrue(batch["review_only"])
            self.assertNotIn("cmd", batch)

    def test_reconciliation_causes_are_closed_and_typed(self):
        cases = [
            ("repository_issues", 9, 9, 0, False, "complete"),
            ("repository_issues", 8, 9, 0, False, "pagination_truncated"),
            ("repository_items", 9, 9, 0, True, "pull_requests_included"),
            ("dispatch_cache", 9, 9, 0, False, "scope_mismatch"),
            ("repository_issues", 9, 9, m.SNAPSHOT_MAX_AGE_SECONDS + 1,
             False, "snapshot_stale"),
        ]
        for scope, fetched, total, age, pulls, want in cases:
            with self.subTest(want=want):
                got = m.reconcile_census(
                    scope=scope, state="open", fetched_count=fetched,
                    total_count=total, snapshot_age_seconds=age,
                    includes_pull_requests=pulls,
                )
                self.assertEqual(got["reconciliation"], want)

    def test_live_fetch_requests_the_issue_only_total_above_500(self):
        import json
        from unittest import mock
        issues = [_issue(n) for n in range(1, 526)]
        commands = []

        def fake_run(args):
            commands.append(args)
            self.assertEqual(args[:2], ["issue", "list"])
            return json.dumps(issues)

        with mock.patch.object(m, "_resolve_repo", return_value="owner/repo"), \
             mock.patch.object(m, "_fetch_issue_total", side_effect=[525, 525]), \
             mock.patch.object(m, "_run_gh", side_effect=fake_run):
            got_issues, census = m.fetch_issues()
        self.assertEqual(len(got_issues), 525)
        self.assertEqual(census["reconciliation"], "complete")
        self.assertEqual(census["attempt_count"], 1)
        self.assertTrue(census["snapshot_id"].startswith("sha256:"))
        self.assertEqual(census["attempts"][0]["total_before"], 525)
        self.assertIn("525", commands[0])

    def test_live_fetch_retries_moving_count_until_snapshot_is_consistent(self):
        import json
        from unittest import mock
        first = [_issue(n) for n in range(1, 4)]
        second = [_issue(n) for n in range(1, 5)]
        payloads = iter((json.dumps(first), json.dumps(second)))

        with mock.patch.object(m, "_resolve_repo", return_value="owner/repo"), \
             mock.patch.object(m, "_fetch_issue_total", side_effect=[3, 4, 4, 4]), \
             mock.patch.object(m, "_run_gh", side_effect=lambda _args: next(payloads)):
            issues, census = m.fetch_issues(max_attempts=3)

        self.assertEqual(len(issues), 4)
        self.assertEqual(census["reconciliation"], "complete")
        self.assertEqual(census["attempt_count"], 2)
        self.assertEqual(
            [attempt["reconciliation"] for attempt in census["attempts"]],
            ["count_mismatch", "complete"],
        )
        report = m.build_report(issues, NOW, census=census)
        self.assertEqual(report["census"]["snapshot_id"], m._snapshot_id(issues))
        md = m.render_md(report, "2026-06-01")
        self.assertIn("attempts 2", md)
        self.assertIn(census["snapshot_id"], md)

    def test_live_fetch_retries_transient_truncation_then_succeeds(self):
        import json
        from unittest import mock
        truncated = [_issue(n) for n in range(1, 4)]
        complete = [_issue(n) for n in range(1, 5)]
        payloads = iter((json.dumps(truncated), json.dumps(complete)))

        with mock.patch.object(m, "_resolve_repo", return_value="owner/repo"), \
             mock.patch.object(m, "_fetch_issue_total", side_effect=[4, 4, 4, 4]), \
             mock.patch.object(m, "_run_gh", side_effect=lambda _args: next(payloads)):
            issues, census = m.fetch_issues(max_attempts=2)

        self.assertEqual(len(issues), 4)
        self.assertEqual(census["reconciliation"], "complete")
        self.assertEqual(
            [attempt["reconciliation"] for attempt in census["attempts"]],
            ["pagination_truncated", "complete"],
        )

    def test_live_fetch_stable_truncation_refuses_after_bound(self):
        import json
        from unittest import mock
        truncated = [_issue(n) for n in range(1, 4)]

        with mock.patch.object(m, "_resolve_repo", return_value="owner/repo"), \
             mock.patch.object(m, "_fetch_issue_total", side_effect=[4, 4, 4, 4]), \
             mock.patch.object(m, "_run_gh", return_value=json.dumps(truncated)):
            issues, census = m.fetch_issues(max_attempts=2)

        self.assertEqual(census["reconciliation"], "pagination_truncated")
        self.assertEqual(census["attempt_count"], 2)
        self.assertEqual(len(census["attempts"]), 2)
        with self.assertRaisesRegex(m.IncompleteRankingError, "pagination_truncated"):
            m.build_report(issues, NOW, census=census)


class InjectedIssuesTest(unittest.TestCase):
    """`--issues PATH|-` lets a named view (issue_views.py show --json) drive triage
    instead of always fetching the full backlog via gh."""

    def _stdin(self, text: str):
        import io
        from unittest import mock
        return mock.patch.object(m.sys, "stdin", io.StringIO(text))

    def test_load_from_file(self):
        import json
        import tempfile
        rows = [{"number": 7, "title": "t", "labels": [], "url": "u"}]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "view.json"
            p.write_text(json.dumps(rows), encoding="utf-8")
            self.assertEqual(m.load_injected_issues(str(p)), rows)

    def test_load_from_stdin(self):
        import json
        rows = [{"number": 8, "title": "t", "labels": [], "url": "u"}]
        with self._stdin(json.dumps(rows)):
            self.assertEqual(m.load_injected_issues("-"), rows)

    def test_empty_input_is_empty_list(self):
        with self._stdin("  \n"):
            self.assertEqual(m.load_injected_issues("-"), [])

    def test_non_array_rejected(self):
        with self._stdin('{"number": 1}'):
            with self.assertRaises(ValueError):
                m.load_injected_issues("-")

    def test_invalid_json_rejected(self):
        with self._stdin("not json"):
            with self.assertRaises(ValueError):
                m.load_injected_issues("-")


if __name__ == "__main__":
    unittest.main()
