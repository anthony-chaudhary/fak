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


def _audit(verdict: str = "OK", witness: str = "diff-witnessed",
           claim_kind: str = "code_effect") -> dict:
    # claim_kind defaults to a REAL resolving token dos emits (code_effect); the
    # binding gate (#2998) demotes a doc/triage claim on a non-docs issue.
    return {"verdict": verdict, "witness": witness, "claim_kind": claim_kind}


class BindingClassifierTest(unittest.TestCase):
    def test_resolving_keyword_in_body_binds(self):
        refs = m.parse_git_log(_log(("abc1234", "fix(gateway): drop bad tool call", "Fixes #142.")))
        self.assertEqual(refs[142][0]["kind"], m.RESOLVING)

    def test_ref_in_subject_binds_worker_contract(self):
        # Dispatch workers are required to put #N in the subject; that is a
        # resolving position even without a separate GitHub close verb.
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

    def test_issue_noun_form_is_resolving(self):
        # The commit hook accepts the house noun form; the audit must match it.
        refs = m.parse_git_log(_log(("efc0e78", "feat(recall): persist index", "Close the false-exact gap (issue #501).")))
        self.assertEqual(refs[501][0]["kind"], m.RESOLVING)

    def test_issues_list_form_is_resolving(self):
        # The repo's house form binds every listed issue.
        refs = m.parse_git_log(_log(("e60b92e", "feat(agent): planner", "Lands the planner (issues #558, #565).")))
        self.assertEqual(refs[558][0]["kind"], m.RESOLVING)
        self.assertEqual(refs[565][0]["kind"], m.RESOLVING)

    def test_issue_noun_dependency_regression_is_mention(self):
        refs = m.parse_git_log(_log((
            "1d00c878",
            "docs(cli-reference): document the garden -> dispatch boundary",
            "Fixes #1792.\n\n"
            "This links the two dispatch issues #1791 builds on: #1404 and #1462.\n"
            "The #1791 feature does not exist yet.",
        )))
        self.assertEqual(refs[1792][0]["kind"], m.RESOLVING)
        self.assertEqual(refs[1791][0]["kind"], m.MENTION)
        self.assertEqual(refs[1404][0]["kind"], m.MENTION)
        self.assertEqual(refs[1462][0]["kind"], m.MENTION)

    def test_bare_prose_hash_is_still_mention(self):
        # The false-binding hazard: a bare '#499' with NO 'issue' anchor stays MENTION.
        refs = m.parse_git_log(_log(("a1fd21a", "feat(journal): digests", "the #499 gap where journal stores digests")))
        self.assertEqual(refs[499][0]["kind"], m.MENTION)

    def test_issue_word_required_for_noun_binding(self):
        # 'see #500 for context' has no 'issue' token and is body-only -> MENTION.
        refs = m.parse_git_log(_log(("c0ffee1", "docs: note", "see #500 for context")))
        self.assertEqual(refs[500][0]["kind"], m.MENTION)

    def test_open_witnessed_discovered_via_close_verb(self):
        refs = m.parse_git_log(_log(("ship777", "feat(recall): candidate index", "Fixes #74.")))
        self.assertEqual(refs[74][0]["kind"], m.RESOLVING)
        g = m.grade_issue(_issue(74, state="OPEN"), refs[74], {"ship777": _audit()})
        self.assertEqual(g["bucket"], m.OPEN_WITNESSED)

    def test_open_witnessed_discovered_via_subject_ref(self):
        refs = m.parse_git_log(_log(("ship778", "feat(recall): candidate index (#75) (fak recall)", "")))
        self.assertEqual(refs[75][0]["kind"], m.RESOLVING)
        g = m.grade_issue(_issue(75, state="OPEN"), refs[75], {"ship778": _audit()})
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

    def test_data_witnessed_close_is_data_resolved_not_claimed(self):
        # Closed + a resolving commit DOS graded OK on the DATA rung (a data-driven
        # feature: a rows.json / dos.toml change). Honest, but not the diff-witnessed
        # gold standard -> its own bucket, NOT CLAIMED_CLOSED (which would slander it)
        # and NOT TRUE_RESOLVED (which would blur the rung).
        g = m.grade_issue(
            _issue(945, state="CLOSED", reason="COMPLETED"),
            [{"sha": "data123", "subject": "feat", "kind": m.RESOLVING}],
            {"data123": _audit(witness="data-witnessed")},
        )
        self.assertEqual(g["bucket"], m.DATA_RESOLVED)
        self.assertEqual(g["data_witnessed_commits"], ["data123"])
        self.assertEqual(g["witnessed_commits"], [])  # not the diff rung

    def test_diff_witness_beats_data_witness_for_true_resolved(self):
        # An issue with BOTH a diff-witnessed and a data-witnessed resolving commit
        # is TRUE_RESOLVED -- the stronger rung wins.
        g = m.grade_issue(
            _issue(99, state="CLOSED", reason="COMPLETED"),
            [{"sha": "diff1", "subject": "fix", "kind": m.RESOLVING},
             {"sha": "data1", "subject": "feat", "kind": m.RESOLVING}],
            {"diff1": _audit(), "data1": _audit(witness="data-witnessed")},
        )
        self.assertEqual(g["bucket"], m.TRUE_RESOLVED)

    def test_closed_with_no_commit_is_claimed(self):
        g = m.grade_issue(_issue(5, state="CLOSED", reason="COMPLETED"), [], {})
        self.assertEqual(g["bucket"], m.CLAIMED_CLOSED)

    def test_closed_not_planned_excluded(self):
        g = m.grade_issue(_issue(9, state="CLOSED", reason="NOT_PLANNED"), [], {})
        self.assertEqual(g["bucket"], m.CLOSED_NOT_PLANNED)


class ClaimBindingGateTest(unittest.TestCase):
    """#2998 checking-layer parity: a diff-witnessed commit whose claim KIND cannot
    resolve the issue (a doc/triage note on a feature) is NOT TRUE_RESOLVED -- the
    same rung the close arm enforces. Without it the grader over-credits doc-claim
    closes of feature issues, inflating closure_rate."""

    def test_commit_binds_resolution_vocab(self):
        # the real dos resolving tokens bind; doc/triage does not (unless docs rung);
        # an unknown/None kind fails OPEN (never slanders a witnessed close).
        self.assertTrue(m.commit_binds_resolution({"claim_kind": "code_effect"}, "feat(x): y"))
        self.assertTrue(m.commit_binds_resolution({"claim_kind": "test"}, "feat(x): y"))
        self.assertFalse(m.commit_binds_resolution({"claim_kind": "doc"}, "feat(x): y"))
        self.assertTrue(m.commit_binds_resolution({"claim_kind": "doc"}, "docs(guard): note"))
        self.assertTrue(m.commit_binds_resolution({"claim_kind": None}, "feat(x): y"))

    def test_doc_claim_on_feature_issue_is_claimed_not_true_resolved(self):
        # closed feature issue, resolving commit dos graded a DOC claim -> the note
        # witnesses itself, not the feature -> CLAIMED_CLOSED, not TRUE_RESOLVED.
        g = m.grade_issue(
            _issue(2205, state="CLOSED", reason="COMPLETED",
                   title="feat(autoctx): relay-default admission"),
            [{"sha": "note123", "subject": "docs(autoctx): triage note (#2205)",
              "kind": m.RESOLVING}],
            {"note123": _audit(claim_kind="doc")},
        )
        self.assertEqual(g["bucket"], m.CLAIMED_CLOSED)
        self.assertEqual(g["witnessed_commits"], [])

    def test_doc_claim_on_docs_rung_is_true_resolved(self):
        # a docs-shaped issue MAY be resolved by a doc claim (#2998 carve-out).
        g = m.grade_issue(
            _issue(9, state="CLOSED", reason="COMPLETED",
                   title="docs(guard): restart continuity contract"),
            [{"sha": "note123", "subject": "docs(guard): write it (#9)", "kind": m.RESOLVING}],
            {"note123": _audit(claim_kind="doc")},
        )
        self.assertEqual(g["bucket"], m.TRUE_RESOLVED)

    def test_doc_claim_on_open_feature_is_open_not_witnessed(self):
        # the OPEN side: a non-binding witnessed commit must not surface a phantom
        # OPEN_WITNESSED the closer would then refuse (skip_nonresolving).
        g = m.grade_issue(
            _issue(2205, state="OPEN", title="feat(autoctx): relay-default admission"),
            [{"sha": "note123", "subject": "docs(autoctx): note (#2205)", "kind": m.RESOLVING}],
            {"note123": _audit(claim_kind="doc")},
        )
        self.assertEqual(g["bucket"], m.OPEN)

    def test_unknown_claim_kind_still_true_resolved(self):
        # a legacy/uncached audit row (no claim_kind) fails OPEN -> stays witnessed.
        g = m.grade_issue(
            _issue(178, state="CLOSED", reason="COMPLETED", title="feat(x): y"),
            [{"sha": "good123", "subject": "fix", "kind": m.RESOLVING}],
            {"good123": _audit(claim_kind=None)},
        )
        self.assertEqual(g["bucket"], m.TRUE_RESOLVED)

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

    def test_data_resolved_leaves_closure_denom_enters_honest_rate(self):
        # strict closure_rate counts only the diff rung and a DATA_RESOLVED close is
        # NOT a claimed gap, so it leaves the strict denominator; honest_close_rate
        # credits it. 1 true + 1 data + 1 claimed -> strict 1/2, honest 2/3.
        issues = [
            _issue(1, state="CLOSED", reason="COMPLETED"),  # true
            _issue(2, state="CLOSED", reason="COMPLETED"),  # data
            _issue(3, state="CLOSED", reason="COMPLETED"),  # claimed
        ]
        refs = {
            1: [{"sha": "w1", "subject": "fix", "kind": m.RESOLVING}],
            2: [{"sha": "d1", "subject": "feat", "kind": m.RESOLVING}],
        }
        audits = {"w1": _audit(), "d1": _audit(witness="data-witnessed")}
        p = self._payload(issues, refs, audits)
        self.assertEqual(p["counts"][m.TRUE_RESOLVED], 1)
        self.assertEqual(p["counts"][m.DATA_RESOLVED], 1)
        self.assertEqual(p["counts"][m.CLAIMED_CLOSED], 1)
        self.assertAlmostEqual(p["closure_rate"], 1 / 2, places=3)
        self.assertAlmostEqual(p["honest_close_rate"], 2 / 3, places=3)

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


class CoverageVerdictTest(unittest.TestCase):
    """The issue-#2640 witness: a truncated audit must expose a TYPED coverage
    verdict and MACHINE-actionable recommended caps, and mark its summaries partial."""

    def test_complete_coverage_is_typed_complete_no_partial(self):
        cov = m.compute_coverage(
            issues_fetched=65, issue_limit=1000,
            commits_scanned=2000, max_commits=2000, total_commits=1092,
        )
        self.assertEqual(cov["verdict"], m.COVERAGE_COMPLETE)
        # Recommended caps equal the current caps when nothing is truncated.
        self.assertEqual(cov["recommended"]["issue_limit"], 1000)
        self.assertEqual(cov["recommended"]["max_commits"], 2000)

    def test_truncated_both_caps_yields_typed_incomplete_and_recommended(self):
        # The exact witness fixture: issues_fetched == issue_limit AND
        # commits_window (max_commits) < commits_total.
        cov = m.compute_coverage(
            issues_fetched=1000, issue_limit=1000,
            commits_scanned=2000, max_commits=2000, total_commits=5134,
        )
        self.assertFalse(cov["complete"])
        self.assertTrue(cov["issues_truncated"])
        self.assertTrue(cov["commits_truncated"])
        # Typed verdict token, not just complete=false.
        self.assertEqual(cov["verdict"], m.COVERAGE_INCOMPLETE)
        # Machine-actionable caps that would clear the truncation.
        rec = cov["recommended"]
        self.assertGreater(rec["issue_limit"], 1000)            # raised past the cap
        self.assertGreater(rec["max_commits"], 5134)            # jumps above history
        # A followable next command with the raised flags.
        self.assertIn("--issue-limit", rec["command"])
        self.assertIn("--max-commits", rec["command"])
        self.assertIn(str(rec["issue_limit"]), rec["command"])
        self.assertIn(str(rec["max_commits"]), rec["command"])

    def test_commits_truncated_total_unknown_doubles_window(self):
        cov = m.compute_coverage(
            issues_fetched=65, issue_limit=1000,
            commits_scanned=2000, max_commits=2000, total_commits=None,
        )
        self.assertEqual(cov["verdict"], m.COVERAGE_INCOMPLETE)
        self.assertEqual(cov["recommended"]["max_commits"], 4000)  # 2x when total unknown

    def test_payload_marks_summaries_partial_on_incomplete_coverage(self):
        # A truncated audit feeding build_payload: closure-rate summaries are marked
        # partial (machine field) and the coverage block keeps its typed verdict.
        cov = m.compute_coverage(
            issues_fetched=1000, issue_limit=1000,
            commits_scanned=2000, max_commits=2000, total_commits=5134,
        )
        issues = [_issue(1, state="CLOSED", reason="COMPLETED")]
        refs = {1: [{"sha": "w1", "subject": "fix", "kind": m.RESOLVING}]}
        p = m.build_payload(
            workspace="C:/work/fleet", issues=issues, refs=refs,
            audits={"w1": _audit()}, coverage=cov,
        )
        self.assertTrue(p["partial"])                              # summaries flagged partial
        self.assertFalse(p["ok"])                                  # not a final health witness
        self.assertEqual(p["coverage"]["verdict"], m.COVERAGE_INCOMPLETE)
        self.assertIn("command", p["coverage"]["recommended"])

    def test_payload_backfills_typed_verdict_on_bare_coverage(self):
        # A caller that passes a bare {complete, notes} still gets a typed verdict.
        p = m.build_payload(
            workspace="C:/work/fleet", issues=[_issue(2, state="CLOSED", reason="COMPLETED")],
            refs={}, audits={}, coverage={"complete": False, "notes": ["hit the cap"]},
        )
        self.assertTrue(p["partial"])
        self.assertEqual(p["coverage"]["verdict"], m.COVERAGE_INCOMPLETE)

    def test_render_marks_closure_rate_partial(self):
        cov = m.compute_coverage(
            issues_fetched=1000, issue_limit=1000,
            commits_scanned=2000, max_commits=2000, total_commits=5134,
        )
        p = m.build_payload(
            workspace="C:/work/fleet",
            issues=[_issue(1, state="CLOSED", reason="COMPLETED")],
            refs={1: [{"sha": "w1", "subject": "fix", "kind": m.RESOLVING}]},
            audits={"w1": _audit()}, coverage=cov,
        )
        text = m.render(p)
        self.assertIn("PARTIAL", text)
        self.assertIn(m.COVERAGE_INCOMPLETE, text)
        self.assertIn("--issue-limit", text)  # the recommended re-run command is shown


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
