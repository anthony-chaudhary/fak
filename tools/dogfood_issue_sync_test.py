#!/usr/bin/env python3
"""Tests for tools/dogfood_issue_sync.py (#800).

Exercise the pure plan_issues/render_body fold on fixture report.json payloads —
the code-slop (verdict=ACTION) and dogfood-coverage (grade/`*_debt`) shapes the
recent-feature dogfood packet actually emits. No network: the gh upsert path is
never reached, so the whole suite is offline and deterministic.
"""
from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "dogfood_issue_sync.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("dogfood_issue_sync", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _report(probes):
    return {"schema": "recent-feature-dogfood/1", "out_dir": "/evi/dir", "probes": probes}


def _probe(key, payload):
    return {"key": key, "command": ["py", "tools/x.py", "--json"], "payload": payload}


def _action_report():
    return _report([_probe("code-slop-scorecard", {
        "schema": "code-slop/1", "verdict": "ACTION", "finding": "extract the dup clone",
        "score": 54.1, "grade": "F", "slop_debt": 12})])


def _write_report(path: Path, report=None, *, mtime: datetime | None = None) -> Path:
    path.write_text(json.dumps(report or _action_report()), encoding="utf-8")
    if mtime is not None:
        ts = mtime.timestamp()
        os.utime(path, (ts, ts))
    return path


class PlanIssuesTest(unittest.TestCase):
    def test_action_scorecard_makes_one_stable_issue(self) -> None:
        mod = load()
        rep = _report([_probe("code-slop-scorecard", {
            "schema": "code-slop/1", "verdict": "ACTION", "finding": "extract the dup clone",
            "score": 54.1, "grade": "F", "slop_debt": 12})])
        issues = mod.plan_issues(rep)
        self.assertEqual(len(issues), 1)
        iss = issues[0]
        self.assertIn("code-slop-scorecard", iss["title"])
        # idempotent-upsert marker present and keyed by the scorecard
        self.assertEqual(iss["marker"], "<!-- dogfood-issue-sync:code-slop-scorecard -->")
        self.assertIn(iss["marker"], iss["body"])
        # the body carries the current grade, debt count, next action, and evidence
        self.assertIn("F", iss["body"])
        self.assertIn("12", iss["body"])
        self.assertIn("extract the dup clone", iss["body"])
        self.assertIn("/evi/dir", iss["body"])
        self.assertIn("dispatchability: `triage_only`", iss["body"])
        self.assertEqual(iss["labels"], ["needs-triage", "triage-only"])

    def test_healthy_grade_a_zero_debt_makes_no_issue(self) -> None:
        mod = load()
        rep = _report([_probe("dogfood-coverage-scorecard", {
            "schema": "dogfood-coverage/1", "grade": "A", "dogfood_debt": 0, "coverage": 95.0})])
        self.assertEqual(mod.plan_issues(rep), [])

    def test_positive_debt_without_action_verdict_makes_issue(self) -> None:
        mod = load()
        rep = _report([_probe("dogfood-coverage-scorecard", {
            "schema": "dogfood-coverage/1", "grade": "B", "dogfood_debt": 3, "coverage": 80.0})])
        issues = mod.plan_issues(rep)
        self.assertEqual(len(issues), 1)
        self.assertIn("3", issues[0]["body"])

    def test_non_scorecard_probe_is_ignored(self) -> None:
        mod = load()
        # a vcache-shaped payload: no top-level grade, no *_debt key -> not a scorecard
        rep = _report([_probe("vcache-score", {"index": {"coverage": 0.86}, "ok": True})])
        self.assertEqual(mod.plan_issues(rep), [])

    def test_multiple_scorecards_one_issue_each(self) -> None:
        mod = load()
        rep = _report([
            _probe("code-slop-scorecard", {"verdict": "ACTION", "grade": "F", "slop_debt": 12}),
            _probe("dogfood-coverage-scorecard", {"grade": "A", "dogfood_debt": 0}),  # healthy -> skip
            _probe("other-scorecard", {"grade": "D", "some_debt": 4}),
        ])
        keys = sorted(i["key"] for i in mod.plan_issues(rep))
        self.assertEqual(keys, ["code-slop-scorecard", "other-scorecard"])

    def test_json_reports_fresh_report_timestamp_and_age(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            report = _write_report(Path(td) / "report.json", mtime=datetime.now(timezone.utc) - timedelta(minutes=5))
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = mod.main(["--report", str(report), "--json"])
            self.assertEqual(code, 0, err.getvalue())
            doc = json.loads(out.getvalue())
            fresh = doc["report_freshness"]
            self.assertEqual(fresh["source"], "mtime")
            self.assertRegex(fresh["timestamp"], r"^\d{4}-\d{2}-\d{2}T")
            self.assertLess(fresh["age_seconds"], fresh["max_age_seconds"])
            self.assertFalse(fresh["stale"])

    def test_text_flags_stale_dry_run_without_refusing(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            report = _write_report(Path(td) / "report.json", mtime=datetime.now(timezone.utc) - timedelta(hours=2))
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = mod.main(["--report", str(report), "--max-report-age", "1h"])
            self.assertEqual(code, 0, err.getvalue())
            body = out.getvalue()
            self.assertIn("report timestamp:", body)
            self.assertIn("stale=yes", body)
            self.assertIn("STALE report:", body)

    def test_sync_refuses_stale_report_before_github(self) -> None:
        mod = load()
        calls = []

        def fake_sync(issue, label):
            calls.append((issue, label))
            return {"key": issue["key"], "action": "created"}

        old_sync = mod._sync_issue
        mod._sync_issue = fake_sync
        try:
            with tempfile.TemporaryDirectory() as td:
                report = _write_report(Path(td) / "report.json", mtime=datetime.now(timezone.utc) - timedelta(hours=2))
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = mod.main(["--report", str(report), "--sync", "--json", "--max-report-age", "1h"])
                self.assertEqual(code, 2)
                doc = json.loads(out.getvalue())
                self.assertEqual(doc["error"], "stale_report")
                self.assertTrue(doc["refused"])
                self.assertTrue(doc["report_freshness"]["stale"])
                self.assertIn("--allow-stale-report", err.getvalue())
                self.assertEqual(calls, [])
        finally:
            mod._sync_issue = old_sync

    def test_sync_override_allows_stale_report_without_real_github(self) -> None:
        mod = load()
        calls = []

        def fake_sync(issue, label):
            calls.append((issue["key"], label))
            return {"key": issue["key"], "action": "created", "url": "https://example.invalid/1"}

        old_sync = mod._sync_issue
        mod._sync_issue = fake_sync
        try:
            with tempfile.TemporaryDirectory() as td:
                report = _write_report(Path(td) / "report.json", mtime=datetime.now(timezone.utc) - timedelta(hours=2))
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = mod.main([
                        "--report", str(report), "--sync", "--allow-stale-report",
                        "--json", "--max-report-age", "1h",
                    ])
                self.assertEqual(code, 0, err.getvalue())
                doc = json.loads(out.getvalue())
                self.assertTrue(doc["report_freshness"]["stale"])
                self.assertTrue(doc["report_freshness"]["stale_allowed"])
                self.assertEqual(calls, [("code-slop-scorecard", "")])
        finally:
            mod._sync_issue = old_sync


if __name__ == "__main__":
    unittest.main()
