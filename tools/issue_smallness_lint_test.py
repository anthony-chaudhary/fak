#!/usr/bin/env python3
"""Hermetic tests for the issue-smallness lint.

The load-bearing case (the issue's own witness): a fixture body bundling three
unrelated tasks must FAIL. A clean single-deliverable body must PASS, and a
two-deliverable body is a WARN (not blocking) so a fix+its-test pairing is not
punished. No network — `gh` calls are monkeypatched.
"""
from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue_smallness_lint as lint  # noqa: E402


SINGLE_DELIVERABLE_BODY = """\
## Goal
Add a filing-time lint for unresolved PowerShell template artifacts.

## Scope for one GPT-5.5 session
- Keep this to one focused edit, report row, fixture, or doc update.
- Do not launch live issue workers unless an existing command already supports a dry-run first.

## Done condition
The issue filer rejects bodies containing the known corruption pattern.

## Witness
A regression test fails on a malformed-body fixture and passes on a normal body.
"""

TWO_DELIVERABLE_BODY = """\
## Goal
- Add a retry counter to the worker heartbeat.
- Add a unit test asserting the counter increments on retry.

## Done condition
Both the counter and its test exist and pass.
"""

# The required witness: three CLEARLY unrelated tasks bundled into one issue.
THREE_UNRELATED_TASKS_BODY = """\
## Goal
- Fix the login redirect bug on the accounts page.
- Add a new weekly throughput dashboard for the operator.
- Rewrite the onboarding documentation from scratch.

## Scope for one GPT-5.5 session
- Keep this to one focused edit, report row, fixture, or doc update.

## Done condition
All three items above are complete and merged.

## Witness
Manual review of the three deliverables.
"""

PROSE_THREE_TASKS_BODY = """\
## Goal
Fix the login redirect bug on the accounts page; add a new weekly throughput
dashboard for the operator; and then rewrite the onboarding documentation.

## Done condition
All work above lands.
"""

EMPTY_BODY = ""


class ExtractSection(unittest.TestCase):
    def test_finds_goal_section(self):
        section = lint.extract_section(SINGLE_DELIVERABLE_BODY, ("goal",))
        self.assertIsNotNone(section)
        self.assertIn("Add a filing-time lint", section)

    def test_missing_heading_returns_none(self):
        self.assertIsNone(lint.extract_section("no headings here", ("goal",)))

    def test_stops_at_next_heading(self):
        section = lint.extract_section(SINGLE_DELIVERABLE_BODY, ("goal",))
        self.assertNotIn("Scope for one GPT-5.5", section)


class LintBody(unittest.TestCase):
    def test_single_deliverable_passes(self):
        result = lint.lint_body(SINGLE_DELIVERABLE_BODY)
        self.assertEqual(result["verdict"], lint.PASS)
        self.assertEqual(result["count"], 1)

    def test_two_deliverables_warns(self):
        result = lint.lint_body(TWO_DELIVERABLE_BODY)
        self.assertEqual(result["verdict"], lint.WARN)
        self.assertEqual(result["count"], 2)

    def test_three_unrelated_tasks_fails(self):
        # This is the issue's own required witness fixture.
        result = lint.lint_body(THREE_UNRELATED_TASKS_BODY)
        self.assertEqual(result["verdict"], lint.FAIL)
        self.assertEqual(result["count"], 3)
        self.assertIn("login redirect", " ".join(result["items"]))
        self.assertIn("throughput dashboard", " ".join(result["items"]))
        self.assertIn("onboarding documentation", " ".join(result["items"]))

    def test_three_unrelated_tasks_in_prose_fails(self):
        result = lint.lint_body(PROSE_THREE_TASKS_BODY)
        self.assertEqual(result["verdict"], lint.FAIL)
        self.assertEqual(result["count"], 3)

    def test_empty_body_passes_trivially(self):
        result = lint.lint_body(EMPTY_BODY)
        self.assertEqual(result["verdict"], lint.PASS)
        self.assertEqual(result["count"], 0)

    def test_single_sentence_with_subordinate_clause_is_not_split(self):
        body = ("## Goal\nAdd a retry counter so the dashboard still works "
                "correctly under load.\n")
        result = lint.lint_body(body)
        self.assertEqual(result["verdict"], lint.PASS)
        self.assertEqual(result["count"], 1)

    def test_done_condition_is_not_folded_into_goal_count(self):
        # A Done condition restating the same single Goal must NOT double the
        # count — it is the acceptance check on the same deliverable, not a
        # second ask.
        body = (
            "## Goal\nAdd a filing-time lint for template artifacts.\n"
            "## Done condition\nThe lint rejects bodies with the known pattern.\n"
        )
        result = lint.lint_body(body)
        self.assertEqual(result["verdict"], lint.PASS)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["section_source"], "goal")

    def test_falls_back_to_done_condition_when_no_goal_heading(self):
        body = (
            "## Done condition\n"
            "- Fix the login redirect bug.\n"
            "- Add a weekly throughput dashboard.\n"
            "- Rewrite the onboarding docs.\n"
        )
        result = lint.lint_body(body)
        self.assertEqual(result["verdict"], lint.FAIL)
        self.assertEqual(result["section_source"], "done")


class FindDeliverables(unittest.TestCase):
    def test_dedupes_identical_bullets(self):
        section = "- Add X\n- add x\n- Add Y\n"
        items = lint.find_deliverables(section)
        self.assertEqual(len(items), 2)


class MainCLI(unittest.TestCase):
    def test_body_file_stdin_pass_exit_zero(self):
        with mock.patch("sys.stdin", io.StringIO(SINGLE_DELIVERABLE_BODY)):
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = lint.main(["--body-file", "-", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["verdict"], lint.PASS)

    def test_body_file_three_tasks_exit_one(self):
        with mock.patch("sys.stdin", io.StringIO(THREE_UNRELATED_TASKS_BODY)):
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = lint.main(["--body-file", "-", "--json"])
        self.assertEqual(code, 1)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["verdict"], lint.FAIL)

    def test_open_mode_wires_into_dry_run_report(self):
        fake_issues = [
            {"number": 1, "title": "clean one", "body": SINGLE_DELIVERABLE_BODY},
            {"number": 2, "title": "bundled one", "body": THREE_UNRELATED_TASKS_BODY},
        ]
        with mock.patch.object(lint, "fetch_open_issues", return_value=fake_issues):
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = lint.main(["--open", "--json"])
        self.assertEqual(code, 1)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["mode"], "open")
        self.assertEqual(payload["scanned"], 2)
        self.assertEqual(payload["counts"][lint.FAIL], 1)
        self.assertEqual(len(payload["flagged"]), 1)
        self.assertEqual(payload["flagged"][0]["number"], 2)

    def test_issue_mode_fetches_via_gh(self):
        with mock.patch.object(lint, "fetch_issue_body",
                                return_value=SINGLE_DELIVERABLE_BODY) as m:
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = lint.main(["--issue", "42", "--json"])
        m.assert_called_once_with(42)
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["issue"], 42)

    def test_bad_body_file_path_is_infra_error(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = lint.main(["--body-file", "/no/such/file/anywhere.md"])
        self.assertEqual(code, 2)

    def test_human_report_open_mode_lists_flagged(self):
        fake_issues = [
            {"number": 9, "title": "bad one", "body": THREE_UNRELATED_TASKS_BODY},
        ]
        with mock.patch.object(lint, "fetch_open_issues", return_value=fake_issues):
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = lint.main(["--open"])
        self.assertEqual(code, 1)
        out = buf.getvalue()
        self.assertIn("#9", out)
        self.assertIn("FAIL", out)


if __name__ == "__main__":
    unittest.main()
