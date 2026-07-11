#!/usr/bin/env python3
"""Hermetic tests for tools/issue_resolve_witnessed.py.

The close-resolved arm shells out three ways — the closure audit, the per-SHA
`dos commit-audit` re-verification, and the `gh issue close` — all of which are
replaced here with synthetic results on the module. NOTHING live (gh/dos/python)
is ever invoked. The re-verification path is exercised hardest because the
`dos commit-audit --json` ARRAY-parsing was a real bug: the oracle emits a JSON
array (one row per audited sha), not a bare object.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "issue_resolve_witnessed.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("issue_resolve_witnessed", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gh_close_then_state(state="CLOSED", reason="COMPLETED", *, record=None):
    """A ``run_capture`` replacement modelling the #2641 close→readback pair: a
    ``gh issue close`` always succeeds (rc 0), and a ``gh issue view --json
    state,stateReason`` reports one fixed authoritative state. ``record`` (a list)
    captures every command so a test can assert which closes/readbacks ran."""
    def run(cmd, cwd, timeout):
        if record is not None:
            record.append(cmd)
        if cmd[:3] == ["gh", "issue", "view"]:
            return 0, json.dumps({"state": state, "stateReason": reason}), ""
        return 0, "", ""
    return run


class OpenWitnessedTest(unittest.TestCase):
    def test_filters_bucket_and_extracts_dict_commit_fields(self) -> None:
        mod = load()
        audit = {"issues": [
            {"number": 10, "title": "ten", "bucket": "OPEN_WITNESSED",
             "witnessed_commits": [{"sha": "aaa111", "subject": "fix ten"}]},
            {"number": 20, "title": "twenty", "bucket": "OPEN_UNWITNESSED",
             "witnessed_commits": [{"sha": "bbb222", "subject": "nope"}]},
            {"number": 30, "title": "thirty", "bucket": "CLOSED"},
        ]}
        rows = mod.open_witnessed(audit)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], {"number": 10, "title": "ten",
                                   "sha": "aaa111", "subject": "fix ten"})

    def test_falls_back_to_resolving_commits_and_sorts_desc_by_number(self) -> None:
        mod = load()
        audit = {"issues": [
            {"number": 5, "title": "five", "bucket": "OPEN_WITNESSED",
             "resolving_commits": [{"sha": "c5", "subject": "five-fix"}]},
            {"number": 99, "title": "nn", "bucket": "OPEN_WITNESSED",
             "witnessed_commits": [{"sha": "c99", "subject": "nn-fix"}]},
            {"number": 42, "title": "ff", "bucket": "OPEN_WITNESSED",
             "resolving_commits": [{"sha": "c42", "subject": "ff-fix"}]},
        ]}
        rows = mod.open_witnessed(audit)
        self.assertEqual([r["number"] for r in rows], [99, 42, 5])
        # the row at #5 came from resolving_commits (witnessed_commits absent).
        self.assertEqual(rows[-1]["sha"], "c5")

    def test_handles_string_commit_and_missing_commits(self) -> None:
        mod = load()
        audit = {"issues": [
            {"number": 7, "title": "str-commit", "bucket": "OPEN_WITNESSED",
             "witnessed_commits": ["deadbeef"]},
            {"number": 8, "title": "no-commit", "bucket": "OPEN_WITNESSED",
             "witnessed_commits": []},
        ]}
        rows = mod.open_witnessed(audit)
        by_num = {r["number"]: r for r in rows}
        self.assertEqual(by_num[7]["sha"], "deadbeef")
        self.assertEqual(by_num[7]["subject"], "")  # string commit has no subject
        self.assertEqual(by_num[8]["sha"], "")       # empty list -> no sha

    def test_empty_audit_yields_no_rows(self) -> None:
        mod = load()
        self.assertEqual(mod.open_witnessed({}), [])
        self.assertEqual(mod.open_witnessed({"issues": []}), [])


class ReverifyArrayParsingTest(unittest.TestCase):
    """`dos commit-audit --json` emits a JSON ARRAY — this is the bug path."""

    def _patch_capture(self, mod, out: str, rc: int = 0, err: str = "") -> None:
        mod.run_capture = lambda cmd, cwd, timeout: (rc, out, err)

    def test_array_with_ok_and_diff_witnessed_is_witness_ok(self) -> None:
        mod = load()
        self._patch_capture(
            mod, '[{"sha":"abc","verdict":"OK","witness":"diff-witnessed"}]')
        rv = mod.reverify(ROOT, "abc")
        self.assertTrue(rv["witness_ok"])
        self.assertEqual(rv["verdict"], "OK")
        self.assertEqual(rv["witness"], "diff-witnessed")
        self.assertIsNone(rv["reason"])

    def test_array_picks_row_matching_the_requested_sha(self) -> None:
        mod = load()
        # the full sha "abc123" startswith the short row sha "abc" -> that row wins
        self._patch_capture(mod, (
            '[{"sha":"zzz","verdict":"FAIL","witness":"none"},'
            '{"sha":"abc","verdict":"OK","witness":"diff-witnessed"}]'))
        rv = mod.reverify(ROOT, "abc123")
        self.assertTrue(rv["witness_ok"])
        self.assertEqual(rv["verdict"], "OK")

    def test_array_non_ok_verdict_is_not_witnessed(self) -> None:
        mod = load()
        self._patch_capture(
            mod, '[{"sha":"abc","verdict":"FAIL","witness":"diff-witnessed"}]')
        rv = mod.reverify(ROOT, "abc")
        self.assertFalse(rv["witness_ok"])
        self.assertEqual(rv["verdict"], "FAIL")
        self.assertIn("verdict=FAIL", rv["reason"])

    def test_array_wrong_witness_is_not_witnessed(self) -> None:
        mod = load()
        self._patch_capture(
            mod, '[{"sha":"abc","verdict":"OK","witness":"test-witnessed"}]')
        rv = mod.reverify(ROOT, "abc")
        self.assertFalse(rv["witness_ok"])
        self.assertEqual(rv["witness"], "test-witnessed")
        self.assertIn("witness=test-witnessed", rv["reason"])

    def test_empty_output_is_not_witnessed(self) -> None:
        mod = load()
        self._patch_capture(mod, "")
        rv = mod.reverify(ROOT, "abc")
        self.assertFalse(rv["witness_ok"])

    def test_empty_array_is_not_witnessed(self) -> None:
        mod = load()
        self._patch_capture(mod, "[]")
        rv = mod.reverify(ROOT, "abc")
        self.assertFalse(rv["witness_ok"])

    def test_garbage_output_is_not_witnessed(self) -> None:
        mod = load()
        self._patch_capture(mod, "not json at all")
        rv = mod.reverify(ROOT, "abc")
        self.assertFalse(rv["witness_ok"])

    def test_bare_object_form_still_parsed(self) -> None:
        mod = load()
        # tolerate the legacy/single-object shape too (dict branch).
        self._patch_capture(
            mod, '{"sha":"abc","verdict":"OK","witness":"diff-witnessed"}')
        rv = mod.reverify(ROOT, "abc")
        self.assertTrue(rv["witness_ok"])

    def test_no_sha_short_circuits(self) -> None:
        mod = load()

        def boom(cmd, cwd, timeout):
            raise AssertionError("must not shell out when sha is empty")

        mod.run_capture = boom
        rv = mod.reverify(ROOT, "")
        self.assertFalse(rv["witness_ok"])
        self.assertEqual(rv["reason"], "no witnessing sha")


class ClaimKindGateTest(unittest.TestCase):
    """The #2998 binding gate: audit-OK + subject-references-(#N) is NOT
    resolves-#N. The witnessed instance: 687cf4d, a docs-only triage note whose
    subject referenced (#2205), closed feature rung #2205. A doc claim binds
    only a docs rung; a code/test claim binds only over non-doc paths."""

    def test_doc_claim_on_feature_issue_is_held(self) -> None:
        mod = load()
        rv = {"witness_ok": True, "claim_kind": "doc", "touches_code": False}
        row = {"number": 2205, "title": "feat(autoctx): relay-default admission"}
        binds, reason = mod.claim_binds_resolution(rv, row)
        self.assertFalse(binds)
        self.assertIn("CLAIM_KIND_NONRESOLVING", reason)

    def test_doc_claim_on_docs_rung_binds(self) -> None:
        mod = load()
        rv = {"witness_ok": True, "claim_kind": "doc", "touches_code": False}
        row = {"number": 9, "title": "docs(guard): restart continuity contract"}
        self.assertEqual(mod.claim_binds_resolution(rv, row), (True, None))

    def test_code_effect_over_source_paths_binds(self) -> None:
        mod = load()
        rv = {"witness_ok": True, "claim_kind": "code_effect", "touches_code": True}
        row = {"number": 5, "title": "fix(guard): budget restart"}
        self.assertEqual(mod.claim_binds_resolution(rv, row), (True, None))

    def test_test_claim_over_source_paths_binds(self) -> None:
        # dos commit-audit emits the bare token `test` (not `test_cover`) for a
        # test-covering commit; a test-witnessed close over non-doc paths must bind
        # (regression: the token drift silently held the whole test class -> 0 closes).
        mod = load()
        self.assertIn("test", mod.RESOLVING_CLAIM_KINDS)
        rv = {"witness_ok": True, "claim_kind": "test", "touches_code": True}
        row = {"number": 3364, "title": "feat(dispatchtick): witness a test-integrity rung"}
        self.assertEqual(mod.claim_binds_resolution(rv, row), (True, None))

    def test_test_claim_with_docs_only_diff_is_held(self) -> None:
        # a `test` claim whose diff touched no code/test path still cannot bind.
        mod = load()
        rv = {"witness_ok": True, "claim_kind": "test", "touches_code": False}
        row = {"number": 5, "title": "feat(x): thing"}
        binds, reason = mod.claim_binds_resolution(rv, row)
        self.assertFalse(binds)
        self.assertIn("CLAIM_KIND_NONRESOLVING", reason)

    def test_code_effect_with_docs_only_diff_is_held(self) -> None:
        mod = load()
        rv = {"witness_ok": True, "claim_kind": "code_effect", "touches_code": False}
        row = {"number": 5, "title": "fix(guard): budget restart"}
        binds, reason = mod.claim_binds_resolution(rv, row)
        self.assertFalse(binds)
        self.assertIn("CLAIM_KIND_NONRESOLVING", reason)

    def test_missing_claim_kind_fails_open(self) -> None:
        mod = load()
        # legacy audit shape (no claim_kind): nothing to bind on — do not wedge.
        rv = {"witness_ok": True}
        row = {"number": 5, "title": "fix(guard): budget restart"}
        self.assertEqual(mod.claim_binds_resolution(rv, row), (True, None))

    def test_evaluate_holds_doc_claim_and_never_calls_gh(self) -> None:
        mod = load()
        mod.load_audit = lambda root, audit_json, max_commits: {
            "closure_rate": 0.5,
            "issues": [
                {"number": 2205, "title": "feat(autoctx): relay rung",
                 "bucket": "OPEN_WITNESSED",
                 "witnessed_commits": [{"sha": "687cf4d", "subject":
                                        "docs(autoctx): triage note (#2205)"}]},
            ]}
        mod.origin_main_resolvable = lambda root: False
        mod.reverify = lambda root, sha: {
            "witness_ok": True, "verdict": "OK", "witness": "diff-witnessed",
            "claim_kind": "doc", "touches_code": False, "reason": None}

        def boom(cmd, cwd, timeout):
            raise AssertionError("a held issue must never reach gh issue close")

        mod.run_capture = boom
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        self.assertEqual(p["results"][0]["action"], "skip_nonresolving")
        self.assertIn("CLAIM_KIND_NONRESOLVING", p["results"][0]["reason"])
        self.assertEqual(p["counts"]["skipped_nonresolving"], 1)
        self.assertEqual(p["counts"]["closed"], 0)
        # the hold renders as a "hold" decision, not a close/failure.
        self.assertEqual(mod.close_decision("skip_nonresolving"), "hold")
        self.assertIn("nonresolving=1", mod.render(p))


class ReverifyClaimKindExtractionTest(unittest.TestCase):
    def test_extracts_claim_kind_and_code_paths_from_array(self) -> None:
        mod = load()
        mod.run_capture = lambda cmd, cwd, timeout: (0, (
            '[{"sha":"abc","verdict":"OK","witness":"diff-witnessed",'
            '"claim_kind":"code_effect","source_files":["a.go"],"test_files":[]}]'), "")
        rv = mod.reverify(ROOT, "abc")
        self.assertTrue(rv["witness_ok"])
        self.assertEqual(rv["claim_kind"], "code_effect")
        self.assertTrue(rv["touches_code"])

    def test_doc_claim_with_no_code_paths(self) -> None:
        mod = load()
        mod.run_capture = lambda cmd, cwd, timeout: (0, (
            '[{"sha":"687cf4d","verdict":"OK","witness":"diff-witnessed",'
            '"claim_kind":"doc","source_files":[],"test_files":[]}]'), "")
        rv = mod.reverify(ROOT, "687cf4d")
        self.assertTrue(rv["witness_ok"])   # the audit gate still passes...
        self.assertEqual(rv["claim_kind"], "doc")
        self.assertFalse(rv["touches_code"])  # ...but the claim cannot bind

    def test_legacy_row_without_kind_or_files_is_tristate_none(self) -> None:
        mod = load()
        mod.run_capture = lambda cmd, cwd, timeout: (0,
            '[{"sha":"abc","verdict":"OK","witness":"diff-witnessed"}]', "")
        rv = mod.reverify(ROOT, "abc")
        self.assertIsNone(rv["claim_kind"])
        self.assertIsNone(rv["touches_code"])


class CloseShapeTest(unittest.TestCase):
    def test_close_comment_cites_sha_and_subject(self) -> None:
        mod = load()
        row = {"number": 12, "sha": "abcdef0123456789", "subject": "fix the thing"}
        comment = mod.close_comment(row)
        self.assertIn("abcdef0123", comment)        # 10-char sha prefix
        self.assertIn("fix the thing", comment)
        self.assertIn("diff-witnessed", comment)

    def test_close_comment_defaults_subject(self) -> None:
        mod = load()
        comment = mod.close_comment({"number": 1, "sha": "abc", "subject": ""})
        self.assertIn("resolving commit", comment)

    def test_close_cmd_shape(self) -> None:
        mod = load()
        row = {"number": 77, "sha": "abc", "subject": "s"}
        cmd = mod.close_cmd(row)
        self.assertEqual(cmd[:4], ["gh", "issue", "close", "77"])
        self.assertEqual(cmd[4], "--comment")
        self.assertEqual(cmd[5], mod.close_comment(row))


class EvaluateTest(unittest.TestCase):
    AUDIT = {
        "closure_rate": 0.5,
        "issues": [
            {"number": 100, "title": "witnessed", "bucket": "OPEN_WITNESSED",
             "witnessed_commits": [{"sha": "wok", "subject": "shipped"}]},
            {"number": 90, "title": "unwitnessed", "bucket": "OPEN_WITNESSED",
             "witnessed_commits": [{"sha": "bad", "subject": "questionable"}]},
        ],
    }

    def _patch(self, mod, reverify_map) -> None:
        mod.load_audit = lambda root, audit_json, max_commits: self.AUDIT
        # Neutralize the durability gate for the base cases (its own behavior is
        # covered in PushedGateTest); with no resolvable origin the gate is inert,
        # so these assertions mirror the pre-gate close logic exactly.
        mod.origin_main_resolvable = lambda root: False
        # Neutralize the #3870 coverage gate too (own behavior in CoverageGateTest):
        # bind every issue so these cases exercise the pre-coverage close logic and
        # never make the read-only body/label gh probe.
        mod.coverage_binds_closure = lambda root, row: (True, None)

        def fake_reverify(root, sha):
            return reverify_map[sha]

        mod.reverify = fake_reverify

    def test_dry_run_would_close_and_skip_and_no_gh(self) -> None:
        mod = load()
        self._patch(mod, {
            "wok": {"witness_ok": True, "verdict": "OK",
                    "witness": "diff-witnessed", "reason": None},
            "bad": {"witness_ok": False, "verdict": "FAIL",
                    "witness": "none", "reason": "commit-audit verdict=FAIL"},
        })

        def boom(cmd, cwd, timeout):
            raise AssertionError("dry-run must not call gh / run_capture")

        mod.run_capture = boom

        p = mod.evaluate(ROOT, limit=10, live=False, audit_json=None, max_commits=600)
        actions = {r["number"]: r["action"] for r in p["results"]}
        self.assertEqual(actions[100], "would_close")
        self.assertEqual(actions[90], "skip_unwitnessed")
        self.assertEqual(p["counts"]["would_close"], 1)
        self.assertEqual(p["counts"]["skipped_unwitnessed"], 1)
        self.assertEqual(p["counts"]["closed"], 0)
        self.assertEqual(p["counts"]["failed"], 0)
        self.assertEqual(p["candidates_total"], 2)
        self.assertEqual(p["planned_count"], 2)
        self.assertEqual(p["verdict"], "PLANNED")
        self.assertTrue(p["ok"])
        self.assertEqual(p["closure_rate_before"], 0.5)

    def test_dry_run_all_unwitnessed(self) -> None:
        mod = load()
        self._patch(mod, {
            "wok": {"witness_ok": False, "reason": "x"},
            "bad": {"witness_ok": False, "reason": "y"},
        })
        mod.run_capture = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not run"))
        p = mod.evaluate(ROOT, limit=10, live=False, audit_json=None, max_commits=600)
        self.assertEqual(p["counts"]["skipped_unwitnessed"], 2)
        self.assertEqual(p["counts"]["would_close"], 0)
        self.assertTrue(p["ok"])  # ok=True in dry-run as long as candidates exist

    def test_limit_bounds_the_batch(self) -> None:
        mod = load()
        self._patch(mod, {
            "wok": {"witness_ok": True, "verdict": "OK",
                    "witness": "diff-witnessed", "reason": None},
            "bad": {"witness_ok": False, "reason": "x"},
        })
        mod.run_capture = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not run"))
        p = mod.evaluate(ROOT, limit=1, live=False, audit_json=None, max_commits=600)
        self.assertEqual(p["planned_count"], 1)        # only the top (#100)
        self.assertEqual(p["candidates_total"], 2)     # but total still counts both
        self.assertEqual(p["results"][0]["number"], 100)

    def test_audit_error_short_circuits(self) -> None:
        mod = load()
        mod.load_audit = lambda root, audit_json, max_commits: {"_error": "boom"}
        p = mod.evaluate(ROOT, limit=10, live=False, audit_json=None, max_commits=600)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "ERROR")
        self.assertEqual(p["reason"], "boom")
        self.assertEqual(p["results"], [])


class PushedGateTest(unittest.TestCase):
    """The durability gate: only close an issue whose resolving commit is reachable
    from origin/main. Guards against the #350 failure mode -- a locally-witnessed
    commit that a shared-tree peer reset orphaned *after* the issue was closed."""

    AUDIT = {
        "closure_rate": 0.5,
        "issues": [
            {"number": 100, "title": "pushed", "bucket": "OPEN_WITNESSED",
             "witnessed_commits": [{"sha": "onmain", "subject": "shipped+pushed"}]},
            {"number": 90, "title": "local-only", "bucket": "OPEN_WITNESSED",
             "witnessed_commits": [{"sha": "localonly", "subject": "shipped-not-pushed"}]},
        ],
    }

    def _patch(self, mod, *, gate_active: bool) -> None:
        mod.load_audit = lambda root, audit_json, max_commits: self.AUDIT
        mod.reverify = lambda root, sha: {
            "witness_ok": True, "verdict": "OK",
            "witness": "diff-witnessed", "reason": None}
        mod.origin_main_resolvable = lambda root: gate_active
        mod.coverage_binds_closure = lambda root, row: (True, None)  # #3870 inert here
        # only "onmain" is an ancestor of origin/main; "localonly" is not.
        mod.reachable_from_origin = lambda root, sha: sha == "onmain"

    def test_unpushed_commit_is_skipped_not_closed(self) -> None:
        mod = load()
        self._patch(mod, gate_active=True)
        closed: list = []
        # readback confirms CLOSED so the pushed issue counts; the held (#90) issue
        # never reaches the close/readback at all.
        mod.run_capture = gh_close_then_state(record=closed)
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        actions = {r["number"]: r["action"] for r in p["results"]}
        self.assertEqual(actions[100], "closed")          # pushed -> closed
        self.assertEqual(actions[90], "skip_unpushed")    # local-only -> held
        self.assertEqual(p["counts"]["closed"], 1)
        self.assertEqual(p["counts"]["skipped_unpushed"], 1)
        self.assertEqual(p["pushed_gate"], "active")
        # the held issue's `gh issue close` must never have run; the pushed one did.
        self.assertFalse(any(c[:4] == ["gh", "issue", "close", "90"] for c in closed))
        self.assertTrue(any(c[:4] == ["gh", "issue", "close", "100"] for c in closed))

    def test_gate_inactive_when_no_origin_closes_local_witness(self) -> None:
        mod = load()
        self._patch(mod, gate_active=False)  # origin/main unresolvable -> degrade
        mod.run_capture = gh_close_then_state()  # readback confirms CLOSED
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        actions = {r["number"]: r["action"] for r in p["results"]}
        self.assertEqual(actions[100], "closed")
        self.assertEqual(actions[90], "closed")           # not gated -> both close
        self.assertEqual(p["counts"]["skipped_unpushed"], 0)
        self.assertEqual(p["pushed_gate"], "no-origin-ref")

    def test_require_pushed_false_disables_gate(self) -> None:
        mod = load()
        self._patch(mod, gate_active=True)  # origin resolvable...
        mod.run_capture = gh_close_then_state()  # readback confirms CLOSED
        # ...but the caller opted out: unpushed commits close anyway.
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None,
                         max_commits=600, require_pushed=False)
        actions = {r["number"]: r["action"] for r in p["results"]}
        self.assertEqual(actions[90], "closed")
        self.assertEqual(p["pushed_gate"], "disabled")


class StateReadbackTest(unittest.TestCase):
    """#2641: a close is counted only after a state readback confirms CLOSED. An
    issue that reads back OPEN/REOPENED is a distinct ``close_not_persistent`` event
    and is never tallied; a repeated close of an already-counted issue in one run is
    counted once. This is the closure-loop fixture with a fake GitHub client that
    returns CLOSED, then OPEN/REOPENED for the same issue, proving the progress
    ledger cannot double-count a reopened issue as closed."""

    RESOLVING_RV = {"witness_ok": True, "verdict": "OK", "witness": "diff-witnessed",
                    "claim_kind": "code_effect", "touches_code": True, "reason": None}

    def _audit(self, *rows):
        # rows: (number, sha) — one OPEN_WITNESSED entry each (numbers may repeat).
        return {"closure_rate": 0.5, "issues": [
            {"number": n, "title": f"fix(x): thing {n}", "bucket": "OPEN_WITNESSED",
             "witnessed_commits": [{"sha": sha, "subject": f"fix {sha}"}]}
            for n, sha in rows]}

    def _fake_gh(self, view_states):
        """`issue close` always succeeds; `issue view <n>` pops the next programmed
        authoritative state for that issue (a queue per number, so the SAME issue
        can read back CLOSED then OPEN/REOPENED across attempts)."""
        calls: list[list[str]] = []
        pending = {n: list(seq) for n, seq in view_states.items()}

        def run(cmd, cwd, timeout):
            calls.append(cmd)
            if cmd[:3] == ["gh", "issue", "view"]:
                n = int(cmd[3])
                seq = pending.get(n) or []
                st = seq.pop(0) if seq else {"state": "CLOSED", "stateReason": "COMPLETED"}
                return 0, json.dumps(st), ""
            return 0, "", ""

        return run, calls

    def _patch(self, mod, audit, view_states):
        mod.load_audit = lambda root, audit_json, max_commits: audit
        mod.origin_main_resolvable = lambda root: False  # inert durability gate
        mod.coverage_binds_closure = lambda root, row: (True, None)  # inert #3870 gate
        mod.reverify = lambda root, sha: dict(self.RESOLVING_RV)
        run, calls = self._fake_gh(view_states)
        mod.run_capture = run
        return calls

    def test_closed_readback_is_counted_once(self) -> None:
        mod = load()
        self._patch(mod, self._audit((2605, "sha1")),
                    {2605: [{"state": "CLOSED", "stateReason": "COMPLETED"}]})
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        r = p["results"][0]
        self.assertEqual(r["action"], "closed")
        self.assertEqual(r["state_after"], "CLOSED")
        self.assertEqual(p["counts"]["closed"], 1)
        self.assertEqual(p["counts"]["close_not_persistent"], 0)
        self.assertEqual(p["closed_numbers"], [2605])

    def test_reopened_readback_is_not_counted(self) -> None:
        # gh issue close rc 0, but the authoritative state reads back OPEN/REOPENED:
        # the close did not persist, so it is NOT tallied as a durable closure.
        mod = load()
        self._patch(mod, self._audit((2605, "sha1")),
                    {2605: [{"state": "OPEN", "stateReason": "REOPENED"}]})
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        r = p["results"][0]
        self.assertEqual(r["action"], mod.CLOSE_NOT_PERSISTENT)
        self.assertEqual(r["state_after"], "OPEN")
        self.assertEqual(r["state_reason"], "REOPENED")
        self.assertEqual(p["counts"]["closed"], 0)
        self.assertEqual(p["counts"]["close_not_persistent"], 1)
        self.assertEqual(p["closed_numbers"], [])
        self.assertEqual(mod.close_decision(mod.CLOSE_NOT_PERSISTENT), "reopened")

    def test_closed_then_reopened_same_issue_no_double_count(self) -> None:
        # The literal Witness: the SAME issue returns CLOSED, then OPEN/REOPENED.
        # Net durable closes = 1 (no double-count); the reopen is a distinct event.
        mod = load()
        self._patch(mod, self._audit((2605, "sha1"), (2605, "sha2")),
                    {2605: [{"state": "CLOSED", "stateReason": "COMPLETED"},
                            {"state": "OPEN", "stateReason": "REOPENED"}]})
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        actions = [r["action"] for r in p["results"]]
        self.assertEqual(actions, ["closed", mod.CLOSE_NOT_PERSISTENT])
        self.assertEqual(p["counts"]["closed"], 1)
        self.assertEqual(p["counts"]["close_not_persistent"], 1)
        self.assertEqual(p["closed_numbers"], [2605])

    def test_repeated_close_same_issue_counted_once(self) -> None:
        # Both attempts read back CLOSED (no intervening reopen): the repeat is
        # 'already_counted', so closed / closed_by_loop_total cannot inflate.
        mod = load()
        self._patch(mod, self._audit((2605, "sha1"), (2605, "sha2")),
                    {2605: [{"state": "CLOSED", "stateReason": "COMPLETED"},
                            {"state": "CLOSED", "stateReason": "COMPLETED"}]})
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        actions = [r["action"] for r in p["results"]]
        self.assertEqual(actions, ["closed", mod.CLOSE_ALREADY_COUNTED])
        self.assertEqual(p["counts"]["closed"], 1)
        self.assertEqual(p["counts"]["already_counted"], 1)
        self.assertEqual(p["closed_numbers"], [2605])

    def test_unreadable_state_fails_open_to_not_persistent(self) -> None:
        # A gh readback error (rc!=0 / no JSON) is UNCONFIRMED: fail-open in the
        # conservative direction -> not counted, surfaced as close_not_persistent.
        mod = load()
        mod.load_audit = lambda root, audit_json, max_commits: self._audit((2605, "sha1"))
        mod.origin_main_resolvable = lambda root: False
        mod.coverage_binds_closure = lambda root, row: (True, None)  # inert #3870 gate
        mod.reverify = lambda root, sha: dict(self.RESOLVING_RV)

        def run(cmd, cwd, timeout):
            if cmd[:3] == ["gh", "issue", "view"]:
                return 1, "", "gh: could not resolve issue"
            return 0, "", ""

        mod.run_capture = run
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        r = p["results"][0]
        self.assertEqual(r["action"], mod.CLOSE_NOT_PERSISTENT)
        self.assertIsNone(r["state_after"])
        self.assertEqual(p["counts"]["closed"], 0)
        self.assertEqual(p["counts"]["close_not_persistent"], 1)

    def test_readback_state_parses_gh_json(self) -> None:
        mod = load()
        mod.run_capture = lambda cmd, cwd, timeout: (
            0, json.dumps({"state": "OPEN", "stateReason": "REOPENED"}), "")
        st = mod.readback_state(ROOT, 2605)
        self.assertEqual(st, {"state": "OPEN", "state_reason": "REOPENED"})

    def test_readback_state_none_number_short_circuits(self) -> None:
        mod = load()

        def boom(cmd, cwd, timeout):
            raise AssertionError("must not shell out for a None issue number")

        mod.run_capture = boom
        self.assertEqual(mod.readback_state(ROOT, None), {})


class CoverageGateTest(unittest.TestCase):
    """The #3870 coverage gate: a single diff-witnessed commit closes an issue only
    when the issue is not EXPLICITLY multi-part. The classifier is pure (body/labels
    in, decision out); the evaluate-level wiring is exercised with a stubbed
    ``fetch_issue_meta`` so no live gh runs."""

    def test_plain_single_scope_body_binds(self) -> None:
        mod = load()
        body = "## In scope\n- add the flag\n- wire it into the picker\n\n## Done when\nit works"
        self.assertEqual(mod.classify_coverage(5, body, set()), (True, None))

    def test_epic_label_holds(self) -> None:
        mod = load()
        binds, reason = mod.classify_coverage(4277, "epic body", {"epic", "class:dev"})
        self.assertFalse(binds)
        self.assertIn("RESOLVED_PARTIAL", reason)
        self.assertIn("epic", reason)

    def test_unchecked_task_box_holds(self) -> None:
        mod = load()
        # an epic's child-issue checklist / an unfinished acceptance criterion.
        body = "## Children\n- [ ] #4278 — L1 structure\n- [x] #4279 done\n"
        binds, reason = mod.classify_coverage(4277, body, set())
        self.assertFalse(binds)
        self.assertIn("unchecked task-list box", reason)

    def test_all_boxes_checked_binds(self) -> None:
        mod = load()
        body = "## Acceptance\n- [x] tests green\n- [X] pushed\n"
        self.assertEqual(mod.classify_coverage(9, body, set()), (True, None))

    def test_spine_first_work_unit_holds(self) -> None:
        mod = load()
        # the literal #3830 marker shape.
        body = ("## Work unit\nSpine-first bugfix leaf. The code lane is occupied, so "
                "this issue is the required first-spine artifact; the missing witness "
                "is the two-denial fixture.")
        binds, reason = mod.classify_coverage(3830, body, set())
        self.assertFalse(binds)
        self.assertIn("spine-first", reason)

    def test_fetch_meta_none_is_coverage_unknown_hold(self) -> None:
        mod = load()
        mod.fetch_issue_meta = lambda root, number: None  # gh unreadable
        binds, reason = mod.coverage_binds_closure(ROOT, {"number": 42})
        self.assertFalse(binds)
        self.assertTrue(reason.startswith("COVERAGE_UNKNOWN"))

    def test_fetch_issue_meta_parses_gh_json(self) -> None:
        mod = load()
        mod.run_capture = lambda cmd, cwd, timeout: (
            0, json.dumps({"body": "hi", "labels": [{"name": "Epic"}, {"name": "bug"}]}), "")
        meta = mod.fetch_issue_meta(ROOT, 4277)
        self.assertEqual(meta["body"], "hi")
        self.assertEqual(meta["labels"], {"epic", "bug"})  # lower-cased

    def test_fetch_issue_meta_gh_error_is_none(self) -> None:
        mod = load()
        mod.run_capture = lambda cmd, cwd, timeout: (1, "", "gh boom")
        self.assertIsNone(mod.fetch_issue_meta(ROOT, 4277))

    def test_fetch_issue_meta_rc0_none_stdout_is_none_not_crash(self) -> None:
        # a rc-0 gh call that still yields None/empty stdout (rare hiccup) must
        # fail-safe to None (COVERAGE_UNKNOWN hold), never AttributeError on .strip().
        mod = load()
        mod.run_capture = lambda cmd, cwd, timeout: (0, None, "")
        self.assertIsNone(mod.fetch_issue_meta(ROOT, 4277))
        mod.run_capture = lambda cmd, cwd, timeout: (0, "", "")
        self.assertIsNone(mod.fetch_issue_meta(ROOT, 4277))

    def _audit(self, number, title, sha="wok", subject="shipped"):
        return {"closure_rate": 0.5, "issues": [
            {"number": number, "title": title, "bucket": "OPEN_WITNESSED",
             "witnessed_commits": [{"sha": sha, "subject": subject}]}]}

    def _patch_through_to_coverage(self, mod, audit):
        """Witness + claim-bind + durability all PASS, so the only gate left is
        coverage -- an issue's disposition is then decided purely by #3870."""
        mod.load_audit = lambda root, audit_json, max_commits: audit
        mod.origin_main_resolvable = lambda root: False
        mod.reverify = lambda root, sha: {
            "witness_ok": True, "verdict": "OK", "witness": "diff-witnessed",
            "claim_kind": "code_effect", "touches_code": True, "reason": None}

    def test_evaluate_holds_epic_and_never_calls_close(self) -> None:
        mod = load()
        self._patch_through_to_coverage(
            mod, self._audit(4277, "epic(deepwiki): witness-verified repo"))
        mod.fetch_issue_meta = lambda root, number: {"body": "grouping issue",
                                                     "labels": {"epic"}}

        def boom(cmd, cwd, timeout):
            raise AssertionError("a coverage-held issue must never reach gh issue close")

        mod.run_capture = boom
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        r = p["results"][0]
        self.assertEqual(r["action"], "skip_partial")
        self.assertIn("RESOLVED_PARTIAL", r["reason"])
        self.assertEqual(p["counts"]["skipped_partial"], 1)
        self.assertEqual(p["counts"]["closed"], 0)
        self.assertEqual(mod.close_decision("skip_partial"), "hold")
        self.assertIn("partial=1", mod.render(p))

    def test_evaluate_dry_run_reflects_partial_hold(self) -> None:
        # the acceptance shape: a spine-first issue is HELD in dry-run (not would_close).
        mod = load()
        self._patch_through_to_coverage(
            mod, self._audit(3830, "fix(x): two-denial binding"))
        mod.fetch_issue_meta = lambda root, number: {
            "body": "## Work unit\nSpine-first bugfix leaf; required first-spine artifact.",
            "labels": {"bug"}}
        p = mod.evaluate(ROOT, limit=10, live=False, audit_json=None, max_commits=600)
        self.assertEqual(p["results"][0]["action"], "skip_partial")
        self.assertEqual(p["counts"]["would_close"], 0)
        self.assertEqual(p["counts"]["skipped_partial"], 1)

    def test_evaluate_covers_plain_issue_closes(self) -> None:
        mod = load()
        self._patch_through_to_coverage(mod, self._audit(9, "fix(x): single scope"))
        mod.fetch_issue_meta = lambda root, number: {
            "body": "## In scope\n- one focused change\n", "labels": {"bug"}}
        mod.run_capture = gh_close_then_state()  # readback confirms CLOSED
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        self.assertEqual(p["results"][0]["action"], "closed")
        self.assertEqual(p["counts"]["closed"], 1)
        self.assertEqual(p["counts"]["skipped_partial"], 0)

    def test_evaluate_coverage_unknown_holds(self) -> None:
        mod = load()
        self._patch_through_to_coverage(mod, self._audit(50, "fix(x): unreadable"))
        mod.fetch_issue_meta = lambda root, number: None  # gh body unreadable

        def boom(cmd, cwd, timeout):
            raise AssertionError("an UNKNOWN-coverage issue must not be closed")

        mod.run_capture = boom
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        r = p["results"][0]
        self.assertEqual(r["action"], "skip_coverage_unknown")
        self.assertEqual(p["counts"]["skipped_coverage_unknown"], 1)
        self.assertEqual(p["counts"]["closed"], 0)
        self.assertEqual(mod.close_decision("skip_coverage_unknown"), "hold")


class RunCaptureEncodingTest(unittest.TestCase):
    """Regression: gh issue bodies carry em-dashes / smart-quotes; on Windows
    text=True defaults to cp1252 and crashed run_capture mid-read, so the coverage
    gate wrongly held every such issue COVERAGE_UNKNOWN. run_capture must decode
    utf-8 total (errors='replace')."""

    def test_utf8_subprocess_output_decodes_without_crash(self) -> None:
        mod = load()
        rc, out, err = mod.run_capture(
            [sys.executable, "-c",
             "import sys; sys.stdout.buffer.write("
             "'em\\u2014dash smart\\u2018q\\u2019 \\u4f60\\u597d\\n'.encode('utf-8'))"],
            ROOT, timeout=30)
        self.assertEqual(rc, 0)
        self.assertIn("em—dash", out)   # U+2014 em-dash survives
        self.assertIn("你好", out)     # non-latin survives


if __name__ == "__main__":
    unittest.main()
