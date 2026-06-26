#!/usr/bin/env python3
"""Hermetic tests for tools/issue_closure_audit.py.

No real gh/dos/git calls: the binding parser is exercised on captured git-log
text, and the grader is exercised on injected issue + audit fixtures. The
load() importlib pattern mirrors the other tools/*_test.py files.
"""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "issue_closure_audit.py"


def load():
    spec = importlib.util.spec_from_file_location("issue_closure_audit", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()


def _log(*records: tuple[str, str, str]) -> str:
    """Build a git-log string in the tool's separator format."""
    out = []
    for sha, subject, body in records:
        out.append(f"{sha}{m._FIELD}{subject}{m._FIELD}{body}{m._RECORD}")
    return "".join(out)


def _issue(number: int, *, state: str = "OPEN", reason: str = "", title: str = "t") -> dict:
    return {"number": number, "state": state, "stateReason": reason, "title": title, "labels": []}


def _audit(verdict: str = "OK", witness: str = "diff-witnessed") -> dict:
    return {"verdict": verdict, "witness": witness, "claim_kind": "fix"}


class BindingClassifierTest(unittest.TestCase):
    def test_resolving_keyword_in_body_binds(self):
        refs = m.parse_git_log(_log(("abc1234", "fix(gateway): drop bad tool call", "Fixes #142.")))
        self.assertEqual(refs[142][0]["kind"], m.RESOLVING)

    def test_ref_in_subject_is_resolving(self):
        # A bare #N in the SUBJECT counts as resolving (fleet's loose convention).
        refs = m.parse_git_log(_log(("abc1234", "fix(tools): NameError (#178)", "body")))
        self.assertEqual(refs[178][0]["kind"], m.RESOLVING)

    def test_relates_to_in_body_is_mention_only(self):
        refs = m.parse_git_log(_log(("abc1234", "docs(memory): note", "Relates to #118 #130; context.")))
        self.assertEqual(refs[118][0]["kind"], m.MENTION)
        self.assertEqual(refs[130][0]["kind"], m.MENTION)

    def test_slack_token_glued_ref_does_not_bind(self):
        # The real false-binding case: a token run followed by a glued #N.
        refs = m.parse_git_log(_log(("abc1234", "chore: bump", "xoxb-secret-#999 leaked nothing")))
        self.assertNotIn(999, refs)

    def test_bare_body_mention_is_mention(self):
        refs = m.parse_git_log(_log(("abc1234", "docs(visuals): front door", "mark #107 plotting lag resolved")))
        # 'resolved' precedes nothing-#107 in close form; #107 is body-only -> mention.
        self.assertEqual(refs[107][0]["kind"], m.MENTION)

    def test_resolving_wins_over_mention_for_same_issue(self):
        refs = m.parse_git_log(
            _log(
                ("aaa", "docs: mention", "see #50 for context"),
                ("bbb", "fix: real", "Closes #50"),
            )
        )
        kinds = {r["kind"] for r in refs[50]}
        self.assertIn(m.RESOLVING, kinds)

    def test_issue_noun_form_binds_resolving(self):
        # The house close convention 'issue #N' (noun form, not a close-verb).
        refs = m.parse_git_log(_log(("efc0e78", "feat(recall): persist index", "Close the false-exact gap (issue #501).")))
        self.assertEqual(refs[501][0]["kind"], m.RESOLVING)

    def test_issues_list_form_binds_all(self):
        # 'issues #N, #M' must bind every number in the comma list.
        refs = m.parse_git_log(_log(("e60b92e", "feat(agent): planner", "Lands the planner (issues #558, #565).")))
        self.assertEqual(refs[558][0]["kind"], m.RESOLVING)
        self.assertEqual(refs[565][0]["kind"], m.RESOLVING)

    def test_bare_prose_hash_is_still_mention(self):
        # The false-binding hazard: a bare '#499' with NO 'issue' anchor stays MENTION.
        refs = m.parse_git_log(_log(("a1fd21a", "feat(journal): digests", "the #499 gap where journal stores digests")))
        self.assertEqual(refs[499][0]["kind"], m.MENTION)

    def test_issue_word_required_for_noun_binding(self):
        # 'see #500 for context' has no 'issue' token and is body-only -> MENTION.
        refs = m.parse_git_log(_log(("c0ffee1", "docs: note", "see #500 for context")))
        self.assertEqual(refs[500][0]["kind"], m.MENTION)

    def test_open_witnessed_discovered_via_issue_noun_form(self):
        # The rung's real payoff: 'issue #N' on a witnessed commit surfaces a
        # shipped-but-still-OPEN fix as OPEN_WITNESSED (closable now).
        refs = m.parse_git_log(_log(("ship777", "feat(recall): candidate index", "Close the gap (issue #74).")))
        self.assertEqual(refs[74][0]["kind"], m.RESOLVING)
        g = m.grade_issue(_issue(74, state="OPEN"), refs[74], {"ship777": _audit()})
        self.assertEqual(g["bucket"], m.OPEN_WITNESSED)


class AuditParseTest(unittest.TestCase):
    def test_first_audit_record_from_array(self):
        # dos commit-audit --json emits an ARRAY; take the first record.
        rec = m._first_audit_record('[{"sha":"abc","verdict":"OK","witness":"diff-witnessed"}]')
        self.assertEqual(rec["verdict"], "OK")
        self.assertEqual(rec["witness"], "diff-witnessed")

    def test_first_audit_record_from_bare_object(self):
        rec = m._first_audit_record('{"sha":"abc","verdict":"ABSTAIN","witness":"abstain"}')
        self.assertEqual(rec["verdict"], "ABSTAIN")

    def test_first_audit_record_empty(self):
        self.assertEqual(m._first_audit_record(""), {})


class GraderTest(unittest.TestCase):
    def test_true_resolved_requires_diff_witness(self):
        # Closed + resolving commit, but the commit only ABSTAINs -> CLAIMED, not TRUE.
        g = m.grade_issue(
            _issue(62, state="CLOSED", reason="COMPLETED"),
            [{"sha": "abstain1", "subject": "docs", "kind": m.RESOLVING}],
            {"abstain1": _audit(verdict="ABSTAIN", witness="abstain")},
        )
        self.assertEqual(g["bucket"], m.CLAIMED_CLOSED)

    def test_true_resolved_when_witnessed(self):
        g = m.grade_issue(
            _issue(178, state="CLOSED", reason="COMPLETED"),
            [{"sha": "good123", "subject": "fix", "kind": m.RESOLVING}],
            {"good123": _audit()},
        )
        self.assertEqual(g["bucket"], m.TRUE_RESOLVED)
        self.assertEqual(g["witnessed_commits"], ["good123"])

    def test_closed_with_no_commit_is_claimed(self):
        g = m.grade_issue(_issue(5, state="CLOSED", reason="COMPLETED"), [], {})
        self.assertEqual(g["bucket"], m.CLAIMED_CLOSED)

    def test_closed_not_planned_excluded(self):
        g = m.grade_issue(_issue(9, state="CLOSED", reason="NOT_PLANNED"), [], {})
        self.assertEqual(g["bucket"], m.CLOSED_NOT_PLANNED)

    def test_open_with_witnessed_commit_is_open_witnessed(self):
        g = m.grade_issue(
            _issue(200, state="OPEN"),
            [{"sha": "ship99a", "subject": "fix", "kind": m.RESOLVING}],
            {"ship99a": _audit()},
        )
        self.assertEqual(g["bucket"], m.OPEN_WITNESSED)

    def test_open_with_only_mention_is_open(self):
        g = m.grade_issue(
            _issue(201, state="OPEN"),
            [{"sha": "m1", "subject": "docs", "kind": m.MENTION}],
            {},
        )
        self.assertEqual(g["bucket"], m.OPEN)

    def test_mention_commit_is_never_witnessed(self):
        # Even if a MENTION commit would audit OK, a mention can't make TRUE_RESOLVED.
        g = m.grade_issue(
            _issue(62, state="CLOSED", reason="COMPLETED"),
            [{"sha": "mentiononly", "subject": "docs", "kind": m.MENTION}],
            {"mentiononly": _audit()},  # OK audit, but it's only a mention
        )
        self.assertEqual(g["bucket"], m.CLAIMED_CLOSED)


class PayloadTest(unittest.TestCase):
    def _payload(self, issues, refs, audits, **kw):
        return m.build_payload(workspace="C:/work/fleet", issues=issues, refs=refs, audits=audits, **kw)

    def test_closure_rate_math(self):
        issues = [
            _issue(1, state="CLOSED", reason="COMPLETED"),  # true
            _issue(2, state="CLOSED", reason="COMPLETED"),  # claimed
            _issue(3, state="CLOSED", reason="COMPLETED"),  # claimed
        ]
        refs = {1: [{"sha": "w1", "subject": "fix", "kind": m.RESOLVING}]}
        audits = {"w1": _audit()}
        p = self._payload(issues, refs, audits)
        self.assertEqual(p["counts"][m.TRUE_RESOLVED], 1)
        self.assertEqual(p["counts"][m.CLAIMED_CLOSED], 2)
        self.assertAlmostEqual(p["closure_rate"], 1 / 3, places=3)

    def test_verdict_action_and_not_ok_when_claimed_present(self):
        issues = [_issue(2, state="CLOSED", reason="COMPLETED")]
        p = self._payload(issues, {}, {})
        self.assertFalse(p["ok"])  # short-circuits the control-pane loop to ACTION
        self.assertEqual(p["verdict"], "ACTION")
        self.assertEqual(p["finding"], "claimed_closed")

    def test_verdict_ok_when_all_witnessed(self):
        issues = [_issue(1, state="CLOSED", reason="COMPLETED")]
        refs = {1: [{"sha": "w1", "subject": "fix", "kind": m.RESOLVING}]}
        p = self._payload(issues, refs, {"w1": _audit()})
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "OK")

    def test_open_witnessed_is_ok_but_flagged(self):
        issues = [_issue(200, state="OPEN")]
        refs = {200: [{"sha": "w1", "subject": "fix", "kind": m.RESOLVING}]}
        p = self._payload(issues, refs, {"w1": _audit()})
        self.assertTrue(p["ok"])
        self.assertEqual(p["finding"], "shipped_but_open")

    def test_claimed_reason_surfaces_closable_open_witnessed(self):
        # When there is BOTH a claimed gap and shipped-but-open work, the claimed
        # headline must still surface the OPEN_WITNESSED issues as closable now.
        issues = [
            _issue(2, state="CLOSED", reason="COMPLETED"),  # claimed (no commit)
            _issue(74, state="OPEN"),                        # open_witnessed
        ]
        refs = {74: [{"sha": "w1", "subject": "fix", "kind": m.RESOLVING}]}
        p = self._payload(issues, refs, {"w1": _audit()}, coverage={"complete": True, "notes": []})
        self.assertEqual(p["finding"], "claimed_closed")
        self.assertEqual(p["counts"][m.OPEN_WITNESSED], 1)
        self.assertIn("closable now", p["reason"])

    def test_audit_error_is_not_ok(self):
        p = self._payload([], {}, {}, audit_error="gh failed")
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "AUDIT_ERROR")

    def test_regression_rate_is_honest_null(self):
        p = self._payload([_issue(1, state="OPEN")], {}, {})
        self.assertIsNone(p["regression_rate"])


class CoverageTest(unittest.TestCase):
    def test_complete_when_under_both_caps(self):
        cov = m.compute_coverage(
            issues_fetched=65, issue_limit=1000,
            commits_scanned=2000, max_commits=2000, total_commits=1092,
        )
        self.assertTrue(cov["complete"])
        self.assertFalse(cov["issues_truncated"])
        self.assertFalse(cov["commits_truncated"])
        self.assertEqual(cov["notes"], [])

    def test_issues_truncated_when_fetch_hits_cap(self):
        # The real bug: gh returned exactly the limit, so older issues were dropped.
        cov = m.compute_coverage(
            issues_fetched=400, issue_limit=400,
            commits_scanned=2000, max_commits=2000, total_commits=1092,
        )
        self.assertFalse(cov["complete"])
        self.assertTrue(cov["issues_truncated"])
        self.assertTrue(any("issue-limit" in n for n in cov["notes"]))

    def test_commits_truncated_when_window_narrower_than_history(self):
        cov = m.compute_coverage(
            issues_fetched=65, issue_limit=1000,
            commits_scanned=800, max_commits=800, total_commits=1092,
        )
        self.assertFalse(cov["complete"])
        self.assertTrue(cov["commits_truncated"])
        self.assertTrue(any("max-commits" in n for n in cov["notes"]))

    def test_commits_truncated_falls_back_to_cap_when_total_unknown(self):
        # git rev-list failed (total None): treat a full window as possibly truncated.
        cov = m.compute_coverage(
            issues_fetched=65, issue_limit=1000,
            commits_scanned=2000, max_commits=2000, total_commits=None,
        )
        self.assertTrue(cov["commits_truncated"])

    def test_incomplete_coverage_blocks_ok_even_with_no_claimed(self):
        # All closed issues we SAW are witnessed, but coverage is partial -> not OK.
        issues = [_issue(1, state="CLOSED", reason="COMPLETED")]
        refs = {1: [{"sha": "w1", "subject": "fix", "kind": m.RESOLVING}]}
        p = m.build_payload(
            workspace="C:/work/fleet", issues=issues, refs=refs, audits={"w1": _audit()},
            coverage={"complete": False, "notes": ["gh fetch hit the cap"]},
        )
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "ACTION")
        self.assertEqual(p["finding"], "incomplete_coverage")

    def test_claimed_gap_still_wins_over_incomplete_coverage(self):
        # A real CLAIMED gap is the loudest signal; partial coverage is noted, not hidden.
        issues = [_issue(2, state="CLOSED", reason="COMPLETED")]
        p = m.build_payload(
            workspace="C:/work/fleet", issues=issues, refs={}, audits={},
            coverage={"complete": False, "notes": ["gh fetch hit the cap"]},
        )
        self.assertFalse(p["ok"])
        self.assertEqual(p["finding"], "claimed_closed")
        self.assertIn("partial coverage", p["reason"])

    def test_complete_coverage_lets_ok_through(self):
        issues = [_issue(1, state="CLOSED", reason="COMPLETED")]
        refs = {1: [{"sha": "w1", "subject": "fix", "kind": m.RESOLVING}]}
        p = m.build_payload(
            workspace="C:/work/fleet", issues=issues, refs=refs, audits={"w1": _audit()},
            coverage={"complete": True, "notes": []},
        )
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "OK")

    def test_payload_carries_coverage_block(self):
        p = m.build_payload(
            workspace="C:/work/fleet", issues=[_issue(1, state="OPEN")], refs={}, audits={},
            coverage={"complete": True, "notes": [], "issues_fetched": 5},
        )
        self.assertEqual(p["coverage"]["issues_fetched"], 5)


class CollectWiringTest(unittest.TestCase):
    def test_collect_only_audits_resolving_commits_for_fetched_issues(self):
        seen: list[str] = []

        def fetcher(_ws):
            return [_issue(142, state="CLOSED", reason="COMPLETED")]

        def auditor(sha, _ws):
            seen.append(sha)
            return _audit()

        # Patch the two git-touching seams to avoid real git calls.
        orig_refs, orig_total = m.git_issue_refs, m.git_total_commits
        m.git_issue_refs = lambda ws, mc: {
            142: [
                {"sha": "resolve1", "subject": "fix", "kind": m.RESOLVING},
                {"sha": "mention1", "subject": "docs", "kind": m.MENTION},
            ],
            999: [{"sha": "other", "subject": "x", "kind": m.RESOLVING}],  # not a fetched issue
        }
        m.git_total_commits = lambda ws: 10  # well under the window -> complete coverage
        try:
            p = m.collect(Path("C:/work/fleet"), fetcher=fetcher, auditor=auditor,
                          use_cache=False)
        finally:
            m.git_issue_refs, m.git_total_commits = orig_refs, orig_total

        # Only the resolving commit for the FETCHED issue is audited.
        self.assertEqual(seen, ["resolve1"])
        self.assertEqual(p["counts"][m.TRUE_RESOLVED], 1)
        self.assertTrue(p["coverage"]["complete"])

    def test_collect_flags_audit_error_on_empty_issues(self):
        orig_refs, orig_total = m.git_issue_refs, m.git_total_commits
        m.git_issue_refs = lambda ws, mc: {}
        m.git_total_commits = lambda ws: 10
        try:
            p = m.collect(Path("C:/work/fleet"), fetcher=lambda _ws: [],
                          auditor=lambda s, w: _audit(), use_cache=False)
        finally:
            m.git_issue_refs, m.git_total_commits = orig_refs, orig_total
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "AUDIT_ERROR")


class AuditCacheTest(unittest.TestCase):
    """The per-SHA cache: a cold miss audits, a warm hit does NOT re-audit, and a
    corrupt cache degrades to a cold audit instead of crashing."""

    def _run(self, ws: Path, seen: list[str]) -> dict:
        def fetcher(_ws):
            return [_issue(142, state="CLOSED", reason="COMPLETED")]

        def auditor(sha, _ws):
            seen.append(sha)
            return _audit()

        orig_refs, orig_total = m.git_issue_refs, m.git_total_commits
        m.git_issue_refs = lambda _ws, _mc: {
            142: [{"sha": "resolve1", "subject": "fix", "kind": m.RESOLVING}],
        }
        m.git_total_commits = lambda _ws: 10
        try:
            return m.collect(ws, fetcher=fetcher, auditor=auditor, use_cache=True)
        finally:
            m.git_issue_refs, m.git_total_commits = orig_refs, orig_total

    def test_cold_miss_audits_then_warm_hit_does_not(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            seen: list[str] = []
            p1 = self._run(ws, seen)
            self.assertEqual(seen, ["resolve1"])           # cold: audited
            self.assertEqual(p1["counts"][m.TRUE_RESOLVED], 1)
            self.assertTrue(m.cache_path(ws).exists())     # cache persisted

            seen2: list[str] = []
            p2 = self._run(ws, seen2)
            self.assertEqual(seen2, [])                     # warm: NOT re-audited
            self.assertEqual(p2["counts"][m.TRUE_RESOLVED], 1)  # same verdict from cache

    def test_corrupt_cache_degrades_to_cold(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            cp = m.cache_path(ws)
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text("{ this is not json", encoding="utf-8")  # garbage
            seen: list[str] = []
            p = self._run(ws, seen)
            self.assertEqual(seen, ["resolve1"])            # cold audit ran anyway
            self.assertEqual(p["counts"][m.TRUE_RESOLVED], 1)

    def test_load_cache_missing_file_is_empty(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(m.load_audit_cache(Path(td) / "nope.json"), {})

    def test_save_then_load_roundtrips(self):
        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / m.AUDIT_CACHE_FILE
            rec = {"sha": "abc", "verdict": "OK", "witness": "diff-witnessed", "claim_kind": "fix"}
            m.save_audit_cache(cp, {"abc": rec})
            self.assertEqual(m.load_audit_cache(cp)["abc"]["verdict"], "OK")


if __name__ == "__main__":
    unittest.main()
