#!/usr/bin/env python3
"""Hermetic tests for tools/dispatch_log_audit.py.

NOTHING live runs: `gh` is never called and no real `.dispatch-runs/` is touched.
The pure logic — the detector matchers (incl. the quoted-grep false-positive
guard), message normalization + signature keys, finding aggregation, the two
dedup rungs, issue rendering, and the min-total→dedup→sort→CAP planner — is
exercised directly, plus a tmp-dir round-trip of the seen-ledger and an end-to-end
main() with every gh boundary stubbed.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "dispatch_log_audit.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("dispatch_log_audit", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = load()


class NormalizeTest(unittest.TestCase):
    def test_digits_and_hex_collapse(self) -> None:
        # two panics that differ only in an index map to one signature
        a = M.normalize_message("index out of range [7]")
        b = M.normalize_message("index out of range [12]")
        self.assertEqual(a, b)
        self.assertIn("#", a)

    def test_signature_key_shape(self) -> None:
        k = M.signature_key("panic-traceback", "claude", "panic: boom 0x4a")
        self.assertEqual(k, "panic-traceback::claude::panic: boom #")


class HookDetectorTest(unittest.TestCase):
    def test_counts_hook_failed_lines(self) -> None:
        text = ("hook: SessionStart Failed\nhook: UserPromptSubmit Failed\n"
                "hook: PreToolUse Failed\nsome other line\n")
        found = M._match_hook_failures(text)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["count"], 3)
        self.assertEqual(found[0]["message"], "hook handler failures")

    def test_no_hook_lines_returns_empty(self) -> None:
        self.assertEqual(M._match_hook_failures("nothing here\n"), [])

    def test_storm_floor_applies_per_log(self) -> None:
        # 2 hook failures with --hook-min 3 → below the per-session storm floor
        text = "hook: A Failed\nhook: B Failed\n"
        self.assertEqual(M.scan_text("resolve-1-x.log", text, "claude", hook_min=3), [])
        out = M.scan_text("resolve-1-x.log", text, "claude", hook_min=2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["detector"], "hook-failure-storm")


class PanicDetectorTest(unittest.TestCase):
    def test_go_panic_at_line_start(self) -> None:
        text = "doing work\npanic: runtime error: index out of range [5]\n  goroutine 1\n"
        found = M._match_panic(text)
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0]["message"].startswith("panic:"))

    def test_python_traceback_headline(self) -> None:
        text = ("Traceback (most recent call last):\n"
                "  File \"x.py\", line 3, in <module>\n"
                "KeyError: 'missing'\n")
        found = M._match_panic(text)
        self.assertEqual(len(found), 1)
        self.assertIn("KeyError", found[0]["message"])

    def test_quoted_traceback_in_json_is_not_a_panic(self) -> None:
        # a worker echoing grep/JSON output that mentions Traceback mid-line must
        # NOT be classified as a real worker panic.
        text = '      "detail": "Traceback (most recent call last):\\n  File ..."\n'
        self.assertEqual(M._match_panic(text), [])

    def test_repeated_identical_panic_collapses(self) -> None:
        text = "panic: boom [1]\npanic: boom [2]\npanic: boom [3]\n"
        found = M._match_panic(text)
        self.assertEqual(len(found), 1)  # collapsed by normalized message
        self.assertEqual(found[0]["count"], 3)


class OffTrunkDetectorTest(unittest.TestCase):
    def test_real_refusal_matches(self) -> None:
        text = "guard: OFF_TRUNK refused push to feature/x (not on main)\n"
        found = M._match_off_trunk(text)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["message"], "OFF_TRUNK guard refusal")

    def test_quoted_repo_line_is_skipped(self) -> None:
        # a ripgrep echo of a doc/source file mentioning OFF_TRUNK is not a refusal
        text = ".\\tools\\githooks\\reference-transaction:10:# the OFF_TRUNK reason\n"
        self.assertEqual(M._match_off_trunk(text), [])

    def test_bare_mention_without_refuse_hint_is_skipped(self) -> None:
        self.assertEqual(M._match_off_trunk("we discussed OFF_TRUNK in passing\n"), [])


class AuthWallDetectorTest(unittest.TestCase):
    def test_not_logged_in(self) -> None:
        found = M._match_auth_wall("Error: Not logged in. Please authenticate.\n")
        self.assertEqual(len(found), 1)
        self.assertIn("not logged in", found[0]["message"])

    def test_credit_balance(self) -> None:
        found = M._match_auth_wall("Your credit balance is too low to run.\n")
        self.assertEqual(found[0]["message"], "auth wall: credit balance too low")


class AggregateTest(unittest.TestCase):
    def test_collapses_across_logs(self) -> None:
        findings = [
            {"detector": "hook-failure-storm", "severity": 80, "min_total": 3,
             "backend": "codex", "log": "resolve-1-a.log",
             "message": "hook handler failures", "count": 3, "sample": ["hook: A Failed"]},
            {"detector": "hook-failure-storm", "severity": 80, "min_total": 3,
             "backend": "codex", "log": "resolve-2-b.log",
             "message": "hook handler failures", "count": 5, "sample": ["hook: B Failed"]},
        ]
        cands = M.aggregate_findings(findings)
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0]["count"], 8)
        self.assertEqual(sorted(cands[0]["logs"]), ["resolve-1-a.log", "resolve-2-b.log"])

    def test_distinct_backends_stay_separate(self) -> None:
        findings = [
            {"detector": "auth-wall", "severity": 50, "min_total": 3, "backend": b,
             "log": f"resolve-{b}.log", "message": "auth wall: not logged in",
             "count": 1, "sample": []}
            for b in ("claude", "opencode")]
        self.assertEqual(len(M.aggregate_findings(findings)), 2)


class DedupTest(unittest.TestCase):
    def _cand(self):
        return {"signature_key": "panic-traceback::claude::panic: boom",
                "detector": "panic-traceback", "severity": 100, "min_total": 1,
                "backend": "claude", "message": "panic: boom", "count": 1,
                "logs": ["resolve-1-x.log"], "sample": ["panic: boom"]}

    def test_seen_ledger_rung(self) -> None:
        seen = {"panic-traceback::claude::panic: boom": {"filed_at": "2026-06-29"}}
        self.assertEqual(M.is_duplicate(self._cand(), seen, set(), ""), "seen-ledger")

    def test_open_issue_stamp_rung(self) -> None:
        issues = [{"number": 9, "title": "old",
                   "body": "blah\n<!-- dispatch-log-audit-sig: "
                           "panic-traceback::claude::panic: boom -->"}]
        stamped, blob = M.existing_issue_index(issues)
        self.assertEqual(M.is_duplicate(self._cand(), {}, stamped, blob), "open-issue")

    def test_open_issue_body_substring_rung(self) -> None:
        issues = [{"number": 9, "title": "x",
                   "body": "see panic-traceback::claude::panic: boom for context"}]
        stamped, blob = M.existing_issue_index(issues)
        self.assertEqual(M.is_duplicate(self._cand(), {}, stamped, blob), "open-issue")

    def test_genuinely_new_passes(self) -> None:
        self.assertIsNone(M.is_duplicate(self._cand(), {}, set(), ""))


class RenderTest(unittest.TestCase):
    def test_stamps_signature_and_label(self) -> None:
        cand = {"signature_key": "hook-failure-storm::codex::hook handler failures",
                "detector": "hook-failure-storm", "severity": 80, "min_total": 3,
                "backend": "codex", "message": "hook handler failures", "count": 8,
                "logs": ["resolve-1.log", "resolve-2.log"], "sample": ["hook: A Failed"]}
        issue = M.render_issue(cand, "2026-06-29")
        self.assertTrue(issue["title"].startswith("dispatch-log-audit: "))
        self.assertIn("codex", issue["title"])
        self.assertIn("<!-- dispatch-log-audit-sig: "
                      "hook-failure-storm::codex::hook handler failures -->",
                      issue["body"])
        self.assertEqual(issue["labels"], ["dispatch"])
        self.assertIn("resolve-1.log", issue["body"])


class PlanTest(unittest.TestCase):
    def _cand(self, det, sev, count, key, min_total=1):
        return {"signature_key": key, "detector": det, "severity": sev,
                "min_total": min_total, "backend": "claude", "message": key,
                "count": count, "logs": ["l.log"], "sample": []}

    def test_below_min_dropped(self) -> None:
        cands = [self._cand("auth-wall", 50, 1, "auth-wall::claude::x", min_total=3)]
        to_file, stats = M.plan_issues(cands, {}, set(), "", dict(M.DEFAULTS), "2026-06-29")
        self.assertEqual(to_file, [])
        self.assertEqual(stats["below-min"], 1)

    def test_cap_and_severity_sort(self) -> None:
        cfg = dict(M.DEFAULTS, max_issues=2)
        cands = [
            self._cand("auth-wall", 50, 9, "auth-wall::claude::a"),
            self._cand("panic-traceback", 100, 1, "panic::claude::b"),
            self._cand("hook-failure-storm", 80, 4, "hook::claude::c"),
        ]
        to_file, _ = M.plan_issues(cands, {}, set(), "", cfg, "2026-06-29")
        self.assertEqual(len(to_file), 2)  # capped
        # severity dominates count: panic(100) first, then hook(80) over auth(50)
        self.assertEqual(to_file[0]["detector"], "panic-traceback")
        self.assertEqual(to_file[1]["detector"], "hook-failure-storm")

    def test_seen_ledger_skips_in_plan(self) -> None:
        cands = [self._cand("panic-traceback", 100, 1, "panic::claude::b")]
        seen = {"panic::claude::b": {"filed_at": "2026-06-29"}}
        to_file, stats = M.plan_issues(cands, seen, set(), "", dict(M.DEFAULTS), "2026-06-29")
        self.assertEqual(to_file, [])
        self.assertEqual(stats["seen-ledger"], 1)


class LedgerTest(unittest.TestCase):
    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self.assertEqual(M.load_seen(runs), {})
            M.save_seen(runs, {"k": {"filed_at": "2026-06-29"}})
            self.assertEqual(M.load_seen(runs), {"k": {"filed_at": "2026-06-29"}})
            self.assertTrue(M.ledger_path(runs).exists())


class ScanRunsTest(unittest.TestCase):
    def test_scans_real_log_files(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d) / ".dispatch-runs"
            runs.mkdir()
            (runs / "resolve-100-20260629-010101.log").write_text(
                "hook: SessionStart Failed\nhook: UserPromptSubmit Failed\n"
                "hook: PreToolUse Failed\n", encoding="utf-8")
            (runs / "resolve-200-20260629-020202.log").write_text(
                "working\npanic: runtime error: nil deref\n", encoding="utf-8")
            findings = M.scan_runs(runs, dict(M.DEFAULTS), now_ts=None)
            dets = {f["detector"] for f in findings}
            self.assertIn("hook-failure-storm", dets)
            self.assertIn("panic-traceback", dets)


class MainHermeticTest(unittest.TestCase):
    """Drive main() end to end with every gh boundary stubbed, to lock the
    load-bearing safety contracts: dry-run mutates nothing; --enact files + records;
    the cap holds; gh-down with no ledger refuses."""

    def setUp(self) -> None:
        self._orig = (M.fetch_open_issues, M.create_issue, M.ensure_label)

    def tearDown(self) -> None:
        (M.fetch_open_issues, M.create_issue, M.ensure_label) = self._orig

    def _runs(self, d: str) -> Path:
        runs = Path(d) / ".dispatch-runs"
        runs.mkdir(parents=True, exist_ok=True)
        (runs / "resolve-100-20260629-010101.log").write_text(
            "hook: SessionStart Failed\nhook: UserPromptSubmit Failed\n"
            "hook: PreToolUse Failed\nhook: PostToolUse Failed\n", encoding="utf-8")
        (runs / "resolve-200-20260629-020202.log").write_text(
            "working\npanic: runtime error: nil deref\n", encoding="utf-8")
        return runs

    def _stub(self, *, existing=None, created_url="https://github.com/x/y/issues/1"):
        M.fetch_open_issues = lambda *a, **k: list(existing or [])
        M.ensure_label = lambda: None
        calls: list = []

        def _create(issue):
            calls.append(issue)
            return created_url
        M.create_issue = _create
        return calls

    def _run(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = M.main(argv)
        return rc, buf.getvalue()

    def test_dry_run_files_nothing_and_writes_no_ledger(self) -> None:
        calls = self._stub()
        with tempfile.TemporaryDirectory() as d:
            runs = self._runs(d)
            rc, out = self._run(["--workspace", d])
            self.assertEqual(rc, 0)
            self.assertEqual(calls, [])
            self.assertFalse(M.ledger_path(runs).exists())
            self.assertIn("dry-run", out)

    def test_enact_files_and_records(self) -> None:
        calls = self._stub()
        with tempfile.TemporaryDirectory() as d:
            runs = self._runs(d)
            rc, _ = self._run(["--workspace", d, "--enact"])
            self.assertEqual(rc, 0)
            self.assertGreaterEqual(len(calls), 1)
            seen = M.load_seen(runs)
            self.assertTrue(seen)  # signatures recorded
            # a second enact run is fully deduped by the ledger → no new files
            calls2 = self._stub()
            rc2, _ = self._run(["--workspace", d, "--enact"])
            self.assertEqual(rc2, 0)
            self.assertEqual(calls2, [])

    def test_enact_respects_cap(self) -> None:
        self._stub()
        with tempfile.TemporaryDirectory() as d:
            self._runs(d)  # two distinct signatures present
            calls = self._stub()
            rc, _ = self._run(["--workspace", d, "--enact", "--max-issues", "1"])
            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 1)  # cap holds despite ≥2 candidates

    def test_refuse_when_gh_fails_and_no_ledger(self) -> None:
        self._stub()

        def _boom(*a, **k):
            raise RuntimeError("gh not authed")
        M.fetch_open_issues = _boom
        with tempfile.TemporaryDirectory() as d:
            self._runs(d)
            rc, _ = self._run(["--workspace", d])
            self.assertEqual(rc, 2)  # refuse rather than risk a blind run

    def test_refuse_when_no_runs_dir(self) -> None:
        self._stub()
        with tempfile.TemporaryDirectory() as d:
            rc, _ = self._run(["--workspace", d])  # no .dispatch-runs created
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
