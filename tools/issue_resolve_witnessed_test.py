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
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "issue_resolve_witnessed.py"


def no_window_creationflags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


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
        # Neutralize the #4374 reopen gate as well (own behavior in ReopenGateTest):
        # allow every issue so these cases never make the read-only timeline probe.
        mod.reopen_blocks_close = lambda root, row: (True, None)
        # Neutralize the #4747 observed-effect gate (own behavior in
        # ObservedEffectGateTest) so these cases never make its body/label probe.
        mod.observed_effect_binds_closure = lambda root, row: (True, None)
        # Neutralize the #5865 author-disclaimer gate (own behavior in
        # DisclaimedResolutionGateTest). It is the one gate that reads the COMMIT
        # rather than the issue, so it shells out to `git show` through the same
        # run_capture these cases replace with a raising stub -- leaving it live
        # would trip that stub on a local read, not on the `gh issue close` the
        # stub exists to catch.
        mod.disclaimer_binds_closure = lambda root, row: (True, None)
        # Neutralize the incomplete-evidence gate too (own behavior in
        # IncompleteEvidenceGateTest) -- it is the other COMMIT-reading gate, so it
        # would likewise reach the raising run_capture stub on a local git read.
        mod.evidence_binds_closure = lambda root, row: (True, None)

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
        mod.reopen_blocks_close = lambda root, row: (True, None)     # #4374 inert here
        mod.evidence_binds_closure = lambda root, row: (True, None)  # inert evidence gate
        # only "onmain" is an ancestor of origin/main; "localonly" is not.
        mod.reachable_from_origin = lambda root, sha: sha == "onmain"
        mod.closure_tip = lambda root: "tip"
        mod.effect_survives_at_tip = lambda root, sha, tip: (True, None)

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
        mod.reopen_blocks_close = lambda root, row: (True, None)     # inert #4374 gate
        mod.observed_effect_binds_closure = lambda root, row: (True, None)  # inert #4747
        mod.evidence_binds_closure = lambda root, row: (True, None)  # inert evidence gate
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
        mod.observed_effect_binds_closure = lambda root, row: (True, None)  # inert #4747
        mod.evidence_binds_closure = lambda root, row: (True, None)  # inert evidence gate
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
        mod.reopen_blocks_close = lambda root, row: (True, None)  # inert #4374 gate
        mod.disclaimer_binds_closure = lambda root, row: (True, None)  # inert #5865
        mod.evidence_binds_closure = lambda root, row: (True, None)  # inert evidence gate
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


class ReopenGateTest(unittest.TestCase):
    """The #4374 reopen-supersedes-witness gate: an auto-reclose may not override a
    `reopened` event unless a commit landed AFTER it. The witnessed harm: #4350 was
    reopened with a broken-main regression, then re-closed citing the SAME pre-reopen
    commit `c39ffeebc` with no new work. A reopen with no newer commit stays open; an
    unreadable timeline fails CLOSED (never a false close on a guess)."""

    def test_parse_iso_handles_z_and_offset(self) -> None:
        mod = load()
        z = mod._parse_iso("2026-07-11T22:52:00Z")
        off = mod._parse_iso("2026-07-11T23:05:12+00:00")
        self.assertIsNotNone(z)
        self.assertIsNotNone(off)
        self.assertLess(z, off)  # 22:52Z is before 23:05+00:00
        self.assertIsNone(mod._parse_iso(""))
        self.assertIsNone(mod._parse_iso("not-a-date"))

    def test_parse_iso_naive_is_assumed_utc_and_comparable(self) -> None:
        mod = load()
        # a stamp with no tz must still compare against an aware stamp (not raise).
        naive = mod._parse_iso("2026-07-11T22:52:00")
        aware = mod._parse_iso("2026-07-11T23:00:00Z")
        self.assertIsNotNone(naive)
        self.assertLess(naive, aware)

    def test_latest_reopen_ts_read_error_is_not_ok(self) -> None:
        mod = load()
        mod.run_capture = lambda cmd, cwd, timeout: (1, "", "gh boom")
        ok, ts = mod.latest_reopen_ts(ROOT, 4350)
        self.assertFalse(ok)   # unreadable -> caller fails CLOSED
        self.assertIsNone(ts)

    def test_latest_reopen_ts_no_reopen_is_ok_none(self) -> None:
        mod = load()
        mod.run_capture = lambda cmd, cwd, timeout: (0, "", "")
        ok, ts = mod.latest_reopen_ts(ROOT, 4350)
        self.assertTrue(ok)    # read succeeded, just never reopened
        self.assertIsNone(ts)

    def test_latest_reopen_ts_takes_max_across_pages(self) -> None:
        mod = load()
        # --paginate emits one created_at line per reopened event; take the latest.
        mod.run_capture = lambda cmd, cwd, timeout: (
            0, "2026-07-01T00:00:00Z\n2026-07-11T22:52:00Z\n2026-07-05T00:00:00Z\n", "")
        ok, ts = mod.latest_reopen_ts(ROOT, 4350)
        self.assertTrue(ok)
        self.assertEqual(ts, mod._parse_iso("2026-07-11T22:52:00Z"))

    def test_latest_reopen_ts_none_number_short_circuits(self) -> None:
        mod = load()

        def boom(cmd, cwd, timeout):
            raise AssertionError("must not shell out for a None issue number")

        mod.run_capture = boom
        self.assertEqual(mod.latest_reopen_ts(ROOT, None), (True, None))

    def test_commit_committer_ts_parses_git(self) -> None:
        mod = load()
        mod.run_capture = lambda cmd, cwd, timeout: (0, "2026-07-11T23:05:12+00:00\n", "")
        ts = mod.commit_committer_ts(ROOT, "c39ffeeb")
        self.assertEqual(ts, mod._parse_iso("2026-07-11T23:05:12+00:00"))

    def test_commit_committer_ts_error_is_none(self) -> None:
        mod = load()
        mod.run_capture = lambda cmd, cwd, timeout: (128, "", "bad object")
        self.assertIsNone(mod.commit_committer_ts(ROOT, "deadbeef"))

    def _reopen_at(self, mod, reopen_iso, commit_iso, *, read_ok=True) -> None:
        mod.latest_reopen_ts = lambda root, number: (
            read_ok, mod._parse_iso(reopen_iso) if reopen_iso else None)
        mod.commit_committer_ts = lambda root, sha: (
            mod._parse_iso(commit_iso) if commit_iso else None)

    def test_never_reopened_allows_close(self) -> None:
        mod = load()
        self._reopen_at(mod, None, "2026-07-11T23:05:12Z")
        self.assertEqual(
            mod.reopen_blocks_close(ROOT, {"number": 5, "sha": "abc"}), (True, None))

    def test_commit_after_reopen_allows_close(self) -> None:
        mod = load()
        self._reopen_at(mod, "2026-07-11T22:52:00Z", "2026-07-11T23:30:00Z")
        allowed, reason = mod.reopen_blocks_close(ROOT, {"number": 5, "sha": "abc"})
        self.assertTrue(allowed)  # new work landed after the reopen -> may close
        self.assertIsNone(reason)

    def test_commit_before_reopen_holds(self) -> None:
        # the literal #4350 shape: witness commit predates the reopen -> stays open.
        mod = load()
        self._reopen_at(mod, "2026-07-11T22:52:00Z", "2026-07-11T20:00:00Z")
        allowed, reason = mod.reopen_blocks_close(ROOT, {"number": 4350, "sha": "c39ffeeb"})
        self.assertFalse(allowed)
        self.assertTrue(reason.startswith(mod.REOPEN_NO_NEW_COMMIT_HOLD))
        self.assertIn("#4350", reason)

    def test_commit_equal_reopen_holds(self) -> None:
        # committer date == reopen instant: still "no new work since the reopen".
        mod = load()
        self._reopen_at(mod, "2026-07-11T22:52:00Z", "2026-07-11T22:52:00Z")
        allowed, reason = mod.reopen_blocks_close(ROOT, {"number": 4350, "sha": "c39ffeeb"})
        self.assertFalse(allowed)
        self.assertTrue(reason.startswith(mod.REOPEN_NO_NEW_COMMIT_HOLD))

    def test_timeline_unreadable_fails_closed(self) -> None:
        mod = load()
        self._reopen_at(mod, "x", "x", read_ok=False)  # read failed -> unknown
        allowed, reason = mod.reopen_blocks_close(ROOT, {"number": 5, "sha": "abc"})
        self.assertFalse(allowed)
        self.assertTrue(reason.startswith(mod.REOPEN_UNKNOWN_HOLD))

    def test_reopened_but_commit_date_unreadable_fails_closed(self) -> None:
        mod = load()
        self._reopen_at(mod, "2026-07-11T22:52:00Z", None)  # commit ts None
        allowed, reason = mod.reopen_blocks_close(ROOT, {"number": 5, "sha": "abc"})
        self.assertFalse(allowed)
        self.assertTrue(reason.startswith(mod.REOPEN_UNKNOWN_HOLD))

    def _audit_one(self):
        return {"closure_rate": 0.5, "issues": [
            {"number": 4350, "title": "fix(x): codex loop", "bucket": "OPEN_WITNESSED",
             "witnessed_commits": [{"sha": "c39ffeeb", "subject": "codex loop fix"}]}]}

    def _patch_through_to_reopen(self, mod) -> None:
        # witness + claim-bind + durability all PASS, so the reopen gate decides.
        mod.load_audit = lambda root, audit_json, max_commits: self._audit_one()
        mod.origin_main_resolvable = lambda root: False
        mod.coverage_binds_closure = lambda root, row: (True, None)
        mod.disclaimer_binds_closure = lambda root, row: (True, None)  # inert #5865
        mod.evidence_binds_closure = lambda root, row: (True, None)  # inert evidence gate
        mod.reverify = lambda root, sha: {
            "witness_ok": True, "verdict": "OK", "witness": "diff-witnessed",
            "claim_kind": "code_effect", "touches_code": True, "reason": None}

    def test_evaluate_holds_reopened_no_new_commit_and_never_closes(self) -> None:
        mod = load()
        self._patch_through_to_reopen(mod)
        mod.latest_reopen_ts = lambda root, number: (True, mod._parse_iso("2026-07-11T22:52:00Z"))
        mod.commit_committer_ts = lambda root, sha: mod._parse_iso("2026-07-11T20:00:00Z")

        def boom(cmd, cwd, timeout):
            raise AssertionError("a reopened-no-new-commit issue must never be re-closed")

        mod.run_capture = boom
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        r = p["results"][0]
        self.assertEqual(r["action"], "skip_reopened")
        self.assertTrue(str(r["reason"]).startswith(mod.REOPEN_NO_NEW_COMMIT_HOLD))
        self.assertEqual(p["counts"]["skipped_reopened"], 1)
        self.assertEqual(p["counts"]["closed"], 0)
        self.assertEqual(mod.close_decision("skip_reopened"), "hold")
        self.assertIn("reopened=1", mod.render(p))

    def test_evaluate_reopen_unknown_holds(self) -> None:
        mod = load()
        self._patch_through_to_reopen(mod)
        mod.latest_reopen_ts = lambda root, number: (False, None)  # timeline unreadable

        def boom(cmd, cwd, timeout):
            raise AssertionError("an unknown-reopen issue must never be re-closed")

        mod.run_capture = boom
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        r = p["results"][0]
        self.assertEqual(r["action"], "skip_reopen_unknown")
        self.assertEqual(p["counts"]["skipped_reopen_unknown"], 1)
        self.assertEqual(p["counts"]["closed"], 0)
        self.assertEqual(mod.close_decision("skip_reopen_unknown"), "hold")

    def test_evaluate_commit_after_reopen_still_closes(self) -> None:
        # a genuine post-reopen fix (commit lands AFTER the reopen) closes normally.
        mod = load()
        self._patch_through_to_reopen(mod)
        mod.latest_reopen_ts = lambda root, number: (True, mod._parse_iso("2026-07-11T22:52:00Z"))
        mod.commit_committer_ts = lambda root, sha: mod._parse_iso("2026-07-12T09:00:00Z")
        mod.run_capture = gh_close_then_state()  # readback confirms CLOSED
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        self.assertEqual(p["results"][0]["action"], "closed")
        self.assertEqual(p["counts"]["closed"], 1)
        self.assertEqual(p["counts"]["skipped_reopened"], 0)


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


class ObservedEffectGateTest(unittest.TestCase):
    """The #4747 observed-effect gate: a MODEL-CORRECTNESS defect (real-weight,
    architecture, coherence) may only auto-close on OBSERVED-EFFECT evidence — an
    independent real artifact demonstrating the original symptom is gone. The
    witnessed harm: #4273 and #4627 were closed after instrumentation/gate code
    landed while their required real-27B observed artifacts were still missing;
    the defect stayed operationally present. Instrumentation is enabling work,
    not resolution evidence — closure without the typed evidence block is
    REFUSED (typed hold), with it, allowed."""

    # ---- pure classifier -------------------------------------------------

    def test_model_defect_label_without_evidence_holds(self) -> None:
        mod = load()
        binds, reason = mod.classify_observed_effect(
            77, "## Symptom\nreal-weight decode drifts incoherent\n",
            {"model-defect", "bug"})
        self.assertFalse(binds)
        self.assertIn("MODEL_DEFECT_EFFECT_UNOBSERVED", reason)

    def test_model_defect_with_effect_observed_line_binds(self) -> None:
        mod = load()
        body = ("## Symptom\nreal-weight decode drifts incoherent\n\n"
                "Effect-Observed: 1.3k-token real-27B transcript coherent, "
                "run 2026-07-18, artifact experiments/qwen36/coherence-gate\n")
        self.assertEqual(
            mod.classify_observed_effect(77, body, {"model-defect"}), (True, None))

    def test_model_defect_with_checked_effect_box_binds(self) -> None:
        mod = load()
        body = ("## Acceptance\n- [x] gate shipped\n"
                "- [x] effect observed: real-weight symptom gone on 27B artifact\n")
        self.assertEqual(
            mod.classify_observed_effect(78, body, {"model-defect"}), (True, None))

    def test_unchecked_effect_box_is_not_evidence(self) -> None:
        mod = load()
        body = "## Acceptance\n- [ ] effect observed: pending real 27B run\n"
        binds, _ = mod.classify_observed_effect(78, body, {"model-defect"})
        self.assertFalse(binds)

    def test_plain_issue_without_markers_binds(self) -> None:
        # the gate is high-precision: an ordinary bug never requires the block.
        mod = load()
        self.assertEqual(
            mod.classify_observed_effect(9, "## In scope\n- one change\n", {"bug"}),
            (True, None))

    def test_terminal_class_declaration_requires_but_is_not_evidence(self) -> None:
        # `Resolution-Class: effect-observed` DECLARES the required terminal
        # class (template marker); the declaration itself must never satisfy it.
        mod = load()
        body = ("## Required evidence\nResolution-Class: effect-observed\n\n"
                "## Fix\nkernel corrected\n")
        binds, reason = mod.classify_observed_effect(80, body, {"bug"})
        self.assertFalse(binds)
        self.assertIn("MODEL_DEFECT_EFFECT_UNOBSERVED", reason)
        # ...and with the evidence line added, it binds.
        binds2, _ = mod.classify_observed_effect(
            80, body + "\nEffect-Observed: real artifact attached\n", {"bug"})
        self.assertTrue(binds2)

    # ---- regression fixtures: the #4273 / #4627 root incidents -----------

    def test_fixture_4273_instrumentation_only_must_not_close(self) -> None:
        # #4273's live label set; hidden-state taps + comparators landed, but no
        # independent real-27B artifact showed the repetition gone.
        mod = load()
        binds, reason = mod.classify_observed_effect(
            4273, "## Symptom\nGGUF degenerates into repetition on ~1.3k-token "
                  "prompts\n## Diagnostic\nhiddentap comparator shipped\n",
            {"bug", "gguf", "priority/p1", "qwen", "generation", "gen/now",
             "class:dev"})
        self.assertFalse(binds)
        self.assertIn("MODEL_DEFECT_EFFECT_UNOBSERVED", reason)
        self.assertIn("4273", reason)

    def test_fixture_4627_gate_code_alone_must_not_close(self) -> None:
        # #4627's live label set; the coherence GATE existing is class 1
        # (diagnostic shipped), not class 4 (effect observed).
        mod = load()
        body = "## C4\ncoherence gate over the int8 Q4_K decode path\n"
        binds, reason = mod.classify_observed_effect(
            4627, body,
            {"gguf", "qwen", "testing", "generation", "gen/now", "class:infra"})
        self.assertFalse(binds)
        self.assertIn("MODEL_DEFECT_EFFECT_UNOBSERVED", reason)
        # with the observed real artifact recorded, the same issue closes.
        binds2, _ = mod.classify_observed_effect(
            4627, body + "\nEffect-Observed: real-weights long-prompt run stayed "
                         "coherent (witness artifact committed)\n",
            {"gguf", "qwen", "testing", "generation", "gen/now", "class:infra"})
        self.assertTrue(binds2)

    # ---- evaluate-level wiring ------------------------------------------

    def _audit(self, number, title, sha="wok", subject="shipped"):
        return {"closure_rate": 0.5, "issues": [
            {"number": number, "title": title, "bucket": "OPEN_WITNESSED",
             "witnessed_commits": [{"sha": sha, "subject": subject}]}]}

    def _patch_through_to_effect(self, mod, audit):
        """Witness + claim-bind + durability + reopen all PASS; coverage sees no
        multi-part marker — the disposition is decided purely by #4747."""
        mod.load_audit = lambda root, audit_json, max_commits: audit
        mod.origin_main_resolvable = lambda root: False
        mod.reopen_blocks_close = lambda root, row: (True, None)
        mod.disclaimer_binds_closure = lambda root, row: (True, None)  # inert #5865
        mod.evidence_binds_closure = lambda root, row: (True, None)  # inert evidence gate
        mod.reverify = lambda root, sha: {
            "witness_ok": True, "verdict": "OK", "witness": "diff-witnessed",
            "claim_kind": "code_effect", "touches_code": True, "reason": None}

    def test_evaluate_holds_model_defect_and_never_calls_close(self) -> None:
        mod = load()
        self._patch_through_to_effect(
            mod, self._audit(4273, "fix(inkernel): repetition on long prompts"))
        mod.fetch_issue_meta = lambda root, number: {
            "body": "## Symptom\nrepetition on real weights\n",
            "labels": {"bug", "gguf", "qwen", "generation"}}

        def boom(cmd, cwd, timeout):
            raise AssertionError(
                "a model-defect issue without observed-effect evidence must "
                "never reach gh issue close")

        mod.run_capture = boom
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        r = p["results"][0]
        self.assertEqual(r["action"], "skip_effect_unobserved")
        self.assertIn("MODEL_DEFECT_EFFECT_UNOBSERVED", r["reason"])
        self.assertEqual(p["counts"]["skipped_effect_unobserved"], 1)
        self.assertEqual(p["counts"]["closed"], 0)
        self.assertEqual(mod.close_decision("skip_effect_unobserved"), "hold")
        self.assertIn("effect_unobserved=1", mod.render(p))

    def test_evaluate_model_defect_with_evidence_closes(self) -> None:
        mod = load()
        self._patch_through_to_effect(
            mod, self._audit(4627, "fix(model): coherence holds on int8 Q4_K"))
        mod.fetch_issue_meta = lambda root, number: {
            "body": ("## Symptom\nlong-prompt incoherence\n\n"
                     "Effect-Observed: real-27B long-prompt run coherent, "
                     "artifact committed\n"),
            "labels": {"gguf", "qwen", "generation", "testing"}}
        mod.run_capture = gh_close_then_state()  # readback confirms CLOSED
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        self.assertEqual(p["results"][0]["action"], "closed")
        self.assertEqual(p["counts"]["closed"], 1)
        self.assertEqual(p["counts"]["skipped_effect_unobserved"], 0)

    def test_evaluate_dry_run_reflects_effect_hold(self) -> None:
        mod = load()
        self._patch_through_to_effect(
            mod, self._audit(4273, "fix(inkernel): repetition on long prompts"))
        mod.fetch_issue_meta = lambda root, number: {
            "body": "## Symptom\nrepetition\n",
            "labels": {"gguf", "generation"}}
        p = mod.evaluate(ROOT, limit=10, live=False, audit_json=None, max_commits=600)
        self.assertEqual(p["results"][0]["action"], "skip_effect_unobserved")
        self.assertEqual(p["counts"]["would_close"], 0)
        self.assertEqual(p["counts"]["skipped_effect_unobserved"], 1)


class DisclaimedResolutionGateTest(unittest.TestCase):
    """The #5865 author-disclaimer gate: a commit whose OWN message says it does not
    resolve the issue cannot witness that it does. Every other gate here reasons over
    the issue (labels, body, timeline) or the diff shape (claim kind, touched paths);
    none read the commit MESSAGE, which is exactly where an author states scope. The
    witnessed harm: #2694 was planned `would_close` on 2726d19, whose body says "No
    artifact is claimed under `visuals/` [...] The issue stays open."

    The fixture bodies below are verbatim excerpts of the real commits."""

    # verbatim from 2726d19 -- the #2694 harm.
    BODY_2694 = (
        "docs(mac): add a reachability re-check gate to the blocked showcase-capture "
        "note (#2694) (fak docs)\n\n"
        "The capture itself is still blocked and this commit does not claim it. #2694\n"
        "wants a real terminal capture under `visuals/` showing the preflight panel.\n\n"
        "No artifact is claimed under `visuals/` and no README link is added, because\n"
        "neither would be true yet. The issue stays open.\n\nRefs #2694\n")
    # verbatim from 3dbe6bf2 -- the #3258 case, a `code_effect` claim over real source.
    BODY_3258 = (
        "feat(agent): add the agent-runtime spine endpoint as a standalone tested unit "
        "(#3258) (fak cmd)\n\n"
        "Scope fence, stated rather than implied: this does NOT wire the route onto the "
        "served\ngateway mux. Nothing in the shipped serving path changes, and the "
        "endpoint is reachable\nonly from its own tests. The gateway-side route composes "
        "with this contract when it\nlands; claiming it as a served endpoint now would be "
        "claiming an integration that has no\nwitness.\n")
    # verbatim from fec8da6f76 -- a LEGITIMATE #2299 witness that a naive marker eats.
    BODY_2299 = (
        "feat(gateway): add the single-arbiter fenced lease write plane over the "
        "coordinator store (fak gateway)\n\n"
        "Adds handleLeaseWrite: a single-arbiter fenced acquire/renew/release over\n"
        "internal/leaseref's AcquireFenced/Renew/ReleaseFenced, serialized through a\n"
        "package-level mutex so the coordinator clone is the one arbiter.\n"
        "The cmd/fak host wiring (installing the leaseref-backed func + origin publish)\n"
        "is a deferred follow-on.\n\nIssue: #2299\n")

    # ---- pure classifier -------------------------------------------------

    def test_fixture_2694_disclaimer_is_refused(self) -> None:
        mod = load()
        binds, reason = mod.commit_disclaims_resolution(2694, self.BODY_2694)
        self.assertFalse(binds)
        self.assertIn(mod.DISCLAIMED_HOLD, reason)
        # the hold quotes the author's own sentence, so a human reading the plan can
        # check it without going back to git.
        self.assertIn("The issue stays open", reason)
        self.assertIn("#2694", reason)

    def test_fixture_3258_is_refused_at_the_code_effect_rung(self) -> None:
        # the gate is deliberately RUNG-BLIND: 3dbe6bf2 is a `code_effect` claim over
        # real source paths, so every diff-shape gate passes it -- only the author's
        # own "would be claiming an integration that has no witness" catches it.
        mod = load()
        rv = {"witness_ok": True, "claim_kind": "code_effect", "touches_code": True}
        row = {"number": 3258, "title": "feat(agent): agent-runtime spine endpoint"}
        self.assertEqual(mod.claim_binds_resolution(rv, row), (True, None))  # passes
        binds, reason = mod.commit_disclaims_resolution(3258, self.BODY_3258)
        self.assertFalse(binds)                                              # held here
        self.assertIn(mod.DISCLAIMED_HOLD, reason)

    def test_ordinary_witness_commit_binds(self) -> None:
        mod = load()
        body = ("fix(gateway): treat same-tick ready as positive (fak gateway)\n\n"
                "The readiness probe compared strictly greater-than, so a lease that "
                "became ready on the same tick read as not-ready. Tests cover both "
                "edges.\n\nIssue: #77\n")
        self.assertEqual(mod.commit_disclaims_resolution(77, body), (True, None))

    def test_deferred_follow_on_near_miss_binds(self) -> None:
        # fec8da6f76 is one of #2299's OWN legitimate witnesses. A bare `follow-on`
        # marker was rejected for this set on measurement -- it matches ~4% of recent
        # commits and would falsely refuse exactly this row. Naming a deferred piece of
        # adjacent work is not disclaiming THIS commit's resolution.
        mod = load()
        self.assertEqual(mod.commit_disclaims_resolution(2299, self.BODY_2299),
                         (True, None))

    def test_smart_quote_negation_is_caught(self) -> None:
        # commit prose uses the smart apostrophe U+2019 freely; a marker that only
        # matched the ASCII form would be trivially evadable.
        mod = load()
        for glyph in ("'", "’", ""):
            with self.subTest(glyph=glyph):
                body = f"feat(x): thing\n\nThis commit does n{glyph}t close the issue.\n"
                binds, reason = mod.commit_disclaims_resolution(9, body)
                self.assertFalse(binds)
                self.assertIn(mod.DISCLAIMED_HOLD, reason)

    def test_numbered_disclaimer_is_scoped_to_the_issue_it_names(self) -> None:
        # "#5847 stays OPEN" disclaims THAT issue. Holding every other issue's close on
        # it would be an over-refusal, so the numbered marker compiles per candidate.
        mod = load()
        body = "feat(x): partial rung\n\nRefs #5847. #5847 stays OPEN for the device rung.\n"
        binds, reason = mod.commit_disclaims_resolution(5847, body)
        self.assertFalse(binds)
        self.assertIn(mod.DISCLAIMED_HOLD, reason)
        # a DIFFERENT issue witnessed by the same commit is untouched.
        self.assertEqual(mod.commit_disclaims_resolution(5848, body), (True, None))

    def test_empty_body_binds(self) -> None:
        # silence is not a disclaimer: this gate refuses on POSITIVE evidence only.
        mod = load()
        self.assertEqual(mod.commit_disclaims_resolution(5, ""), (True, None))

    # ---- the git-reading seam --------------------------------------------

    def test_reads_the_witness_sha_and_holds(self) -> None:
        mod = load()
        seen = []

        def fake(cmd, cwd, timeout):
            seen.append(cmd)
            return 0, self.BODY_2694, ""

        mod.run_capture = fake
        binds, reason = mod.disclaimer_binds_closure(
            ROOT, {"number": 2694, "sha": "2726d19806"})
        self.assertFalse(binds)
        self.assertIn(mod.DISCLAIMED_HOLD, reason)
        self.assertEqual(seen[0][:4], ["git", "show", "-s", "--format=%B"])
        self.assertEqual(seen[0][4], "2726d19806")

    def test_unreadable_commit_body_allows_with_a_note(self) -> None:
        # ASYMMETRY ON PURPOSE: unlike the issue-side gates, an unreadable input here
        # ALLOWS with an audit note instead of failing closed. This gate only ever adds
        # a refusal on positive evidence -- a disclaimer the author wrote -- and an
        # absent commit message is not evidence of one. Failing closed would convert
        # every workspace where `git show` cannot resolve the witness into a blanket
        # hold. The row remains subject to every other gate, exactly as before #5865.
        mod = load()
        mod.run_capture = lambda cmd, cwd, timeout: (128, "", "fatal: bad object")
        binds, note = mod.disclaimer_binds_closure(ROOT, {"number": 1, "sha": "deadbeef"})
        self.assertTrue(binds)
        self.assertIn(mod.DISCLAIM_UNREADABLE_NOTE, note)
        self.assertNotIn(mod.DISCLAIMED_HOLD, note)

    def test_missing_sha_allows_with_a_note_and_never_shells_out(self) -> None:
        mod = load()

        def boom(cmd, cwd, timeout):
            raise AssertionError("an empty sha must not reach git")

        mod.run_capture = boom
        binds, note = mod.disclaimer_binds_closure(ROOT, {"number": 1, "sha": ""})
        self.assertTrue(binds)
        self.assertIn(mod.DISCLAIM_UNREADABLE_NOTE, note)

    def test_clean_body_binds_with_no_note(self) -> None:
        mod = load()
        mod.run_capture = lambda cmd, cwd, timeout: (0, "fix(x): a real fix\n", "")
        self.assertEqual(
            mod.disclaimer_binds_closure(ROOT, {"number": 1, "sha": "abc"}), (True, None))

    # ---- evaluate-level wiring -------------------------------------------

    def _patch_through_to_disclaimer(self, mod, number, sha, subject) -> None:
        """Witness + claim-bind all PASS, so the disclaimer gate decides. It runs FIRST
        (ahead of every gh probe), because it is a local git read and a self-disclaimed
        witness needs no tracker round-trip to refuse."""
        mod.load_audit = lambda root, audit_json, max_commits: {
            "closure_rate": 0.5, "issues": [
                {"number": number, "title": "docs(mac): showcase capture",
                 "bucket": "OPEN_WITNESSED",
                 "witnessed_commits": [{"sha": sha, "subject": subject}]}]}
        mod.origin_main_resolvable = lambda root: False
        mod.coverage_binds_closure = lambda root, row: (True, None)
        mod.reopen_blocks_close = lambda root, row: (True, None)
        mod.observed_effect_binds_closure = lambda root, row: (True, None)
        mod.evidence_binds_closure = lambda root, row: (True, None)  # inert evidence gate
        mod.reverify = lambda root, sha_: {
            "witness_ok": True, "verdict": "OK", "witness": "diff-witnessed",
            "claim_kind": "code_effect", "touches_code": True, "reason": None}

    def test_evaluate_holds_disclaimed_and_never_calls_gh(self) -> None:
        mod = load()
        self._patch_through_to_disclaimer(
            mod, 2694, "2726d19806", "docs(mac): reachability re-check gate (#2694)")

        def git_only(cmd, cwd, timeout):
            if cmd[:2] != ["git", "show"]:
                raise AssertionError(
                    "a self-disclaimed witness must never reach gh issue close")
            return 0, self.BODY_2694, ""

        mod.run_capture = git_only
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        r = p["results"][0]
        self.assertEqual(r["action"], "skip_disclaimed")
        self.assertIn(mod.DISCLAIMED_HOLD, r["reason"])
        self.assertEqual(p["counts"]["skipped_disclaimed"], 1)
        self.assertEqual(p["counts"]["closed"], 0)
        self.assertEqual(p["counts"]["would_close"], 0)
        # the hold renders as a "hold" decision, not a close/failure.
        self.assertEqual(mod.close_decision("skip_disclaimed"), "hold")
        self.assertIn("disclaimed=1", mod.render(p))

    def test_evaluate_dry_run_reflects_disclaimed_hold(self) -> None:
        # the acceptance shape: #2694 was planned `would_close` before #5865.
        mod = load()
        self._patch_through_to_disclaimer(
            mod, 2694, "2726d19806", "docs(mac): reachability re-check gate (#2694)")
        mod.run_capture = lambda cmd, cwd, timeout: (0, self.BODY_2694, "")
        p = mod.evaluate(ROOT, limit=10, live=False, audit_json=None, max_commits=600)
        self.assertEqual(p["results"][0]["action"], "skip_disclaimed")
        self.assertEqual(p["counts"]["would_close"], 0)
        self.assertEqual(p["counts"]["skipped_disclaimed"], 1)

    def test_evaluate_undisclaimed_witness_still_closes(self) -> None:
        # the regression side: fec8da6f76 (a real #2299 witness) closes normally.
        mod = load()
        self._patch_through_to_disclaimer(
            mod, 2299, "fec8da6f76", "feat(gateway): fenced lease write plane")
        base = gh_close_then_state()

        def run(cmd, cwd, timeout):
            if cmd[:2] == ["git", "show"]:
                return 0, self.BODY_2299, ""
            return base(cmd, cwd, timeout)

        mod.run_capture = run
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        self.assertEqual(p["results"][0]["action"], "closed")
        self.assertEqual(p["counts"]["closed"], 1)
        self.assertEqual(p["counts"]["skipped_disclaimed"], 0)

    def test_evaluate_unreadable_body_notes_the_gate_and_still_closes(self) -> None:
        # the abstention is VISIBLE on the row: a gate that declined to apply must not
        # be invisible in the plan.
        mod = load()
        self._patch_through_to_disclaimer(
            mod, 2299, "fec8da6f76", "feat(gateway): fenced lease write plane")
        base = gh_close_then_state()

        def run(cmd, cwd, timeout):
            if cmd[:2] == ["git", "show"]:
                return 128, "", "fatal: bad object"
            return base(cmd, cwd, timeout)

        mod.run_capture = run
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        r = p["results"][0]
        self.assertEqual(r["action"], "closed")
        self.assertEqual(p["counts"]["skipped_disclaimed"], 0)
        self.assertTrue(any(mod.DISCLAIM_UNREADABLE_NOTE in n
                            for n in r.get("gate_notes", [])))


class IncompleteEvidenceGateTest(unittest.TestCase):
    """The incomplete-evidence gate: a commit whose own ARTIFACT declares itself
    unfinished cannot witness a resolution. #5865 reads the commit MESSAGE; a
    benchmark-shaped resolution states its scope in the packet it adds instead.

    The witnessed harm: on 2026-08-10 the arm closed 31 issues (#6122 .. #6205) on
    commits that each added a `docs/benchmarks/` comparison packet headed
    `Status: **INCOMPLETE**`, with only a native arm and a tuned baseline executing
    and every external arm a zero-measurement placeholder. Every upstream gate passed
    honestly, so nothing refused.

    The packet fixtures below are verbatim excerpts of the real committed files."""

    # verbatim line 3 of docs/benchmarks/GO-CALL-GRAPH-ALTERNATIVES-2026-08-10.md,
    # added by 8595070f42 -- the commit that closed #6205.
    PACKET_6205 = (
        "# Go call-graph alternatives — 2026-08-10\n\n"
        "Status: **INCOMPLETE**. Issue "
        "[#6205](https://github.com/anthony-chaudhary/fak/issues/6205) tracks real "
        "type-aware and code-intelligence tool runs with independent resource and "
        "cost witnesses.\n")
    # verbatim line 5 of docs/benchmarks/BOUNDED-CACHE-SNAPSHOT-ALTERNATIVES-2026-08-10.md
    # (#6165): the SAME status, written in the other packet dialect this corpus uses.
    PACKET_6165 = (
        "# Bounded cache-snapshot alternatives — 2026-08-10\n\n## Verdict\n\n"
        "**INCOMPLETE.** The local packet executes fak's bounded fsynced JSONL "
        "snapshot/replay and an unbounded append-only JSONL baseline. Prometheus, "
        "OpenTelemetry, SQLite, Prometheus TSDB, and ClickHouse retain zero "
        "measurements until their real stores execute the common retention workload; "
        "[#6165](https://github.com/anthony-chaudhary/fak/issues/6165) tracks those "
        "witnesses.\n")
    # verbatim tail of docs/benchmarks/CACHE-OBSERVABILITY-ALTERNATIVES-2026-08-10.md
    # (#6122): a third dialect that never uppercases the token and states the hold as
    # an author disclaimer instead.
    PACKET_6122 = (
        "# Cache-observability alternatives — 2026-08-10\n\n"
        "Issue: [#6122](https://github.com/anthony-chaudhary/fak/issues/6122)\n\n"
        "## Honest status\n\nThe contract and local fixture are present, but the "
        "comparison is incomplete. Issue #6122 remains open until all five same-trace "
        "arms have independent correctness, latency, resource, and total-cost "
        "witnesses.\n")
    # The shared registry every packet commit also touches. It carries its own
    # INCOMPLETE spine header and an INCOMPLETE row for dozens of OTHER capabilities --
    # the exact shape a file-global "INCOMPLETE in text" check would false-refuse on.
    REGISTRY = (
        "# Native implementation benchmark contracts\n\n"
        "Status: **INCOMPLETE**. This is the machine-readable starting spine for "
        "requiring every fak-native implementation to be compared with the strongest "
        "practical alternatives.\n\n"
        "| Capability | Package | Issue | Status |\n|---|---|---|---:|\n"
        "| Deadline-aware EDF admission | `internal/deadlineadmit` | "
        "[#6135](https://github.com/anthony-chaudhary/fak/issues/6135) | INCOMPLETE |\n"
        "| Syntactic multi-file Go call graph | `internal/codegraph` | "
        "[#6205](https://github.com/anthony-chaudhary/fak/issues/6205) | COMPLETE |\n")

    # ---- the pure classifier ---------------------------------------------

    def test_status_marker_on_the_issues_own_line_holds(self) -> None:
        mod = load()
        binds, reason = mod.evidence_declares_incomplete(
            6205, "docs/benchmarks/GO-CALL-GRAPH-ALTERNATIVES-2026-08-10.md",
            self.PACKET_6205)
        self.assertFalse(binds)
        self.assertIn(mod.EVIDENCE_INCOMPLETE_HOLD, reason)
        self.assertIn("GO-CALL-GRAPH", reason)
        self.assertIn("INCOMPLETE", reason)

    def test_verdict_dialect_holds_too(self) -> None:
        mod = load()
        binds, reason = mod.evidence_declares_incomplete(
            6165, "docs/benchmarks/BOUNDED-CACHE-SNAPSHOT-ALTERNATIVES-2026-08-10.md",
            self.PACKET_6165)
        self.assertFalse(binds)
        self.assertIn(mod.EVIDENCE_INCOMPLETE_HOLD, reason)

    def test_stays_open_disclaimer_dialect_holds(self) -> None:
        mod = load()
        binds, reason = mod.evidence_declares_incomplete(
            6122, "docs/benchmarks/CACHE-OBSERVABILITY-ALTERNATIVES-2026-08-10.md",
            self.PACKET_6122)
        self.assertFalse(binds)
        self.assertIn(mod.EVIDENCE_INCOMPLETE_HOLD, reason)
        self.assertIn("stays open", reason)

    def test_marker_for_another_issue_does_not_bind_this_one(self) -> None:
        # OVER-REFUSAL GUARD: #6205's registry row says COMPLETE. The file is full of
        # INCOMPLETE -- its own spine header and #6135's row -- and none of it is
        # evidence about #6205. A file-global check would wrongly hold here.
        mod = load()
        self.assertEqual(
            mod.evidence_declares_incomplete(
                6205, "docs/benchmarks/NATIVE-IMPLEMENTATION-COMPARISONS.md",
                self.REGISTRY),
            (True, None))
        # ...and the same file DOES bind #6135, whose row still says INCOMPLETE.
        binds, reason = mod.evidence_declares_incomplete(
            6135, "docs/benchmarks/NATIVE-IMPLEMENTATION-COMPARISONS.md", self.REGISTRY)
        self.assertFalse(binds)
        self.assertIn(mod.EVIDENCE_INCOMPLETE_HOLD, reason)

    def test_complete_packet_binds(self) -> None:
        mod = load()
        self.assertEqual(
            mod.evidence_declares_incomplete(
                6205, "docs/benchmarks/GO-CALL-GRAPH-ALTERNATIVES-2026-08-10.md",
                self.PACKET_6205.replace("**INCOMPLETE**", "**COMPLETE**")),
            (True, None))

    # ---- the git-reading seam --------------------------------------------

    def _packet_reader(self, touched, blobs):
        """A ``run_capture`` replacement serving one commit's file list and blobs."""
        seen = []

        def run(cmd, cwd, timeout):
            seen.append(cmd)
            if cmd[:2] == ["git", "diff-tree"]:
                return 0, "\n".join(touched) + "\n", ""
            if cmd[:2] == ["git", "show"]:
                path = cmd[2].split(":", 1)[1]
                if path not in blobs:
                    return 128, "", "fatal: path does not exist"
                return 0, blobs[path], ""
            raise AssertionError(f"unexpected command {cmd}")

        return run, seen

    def test_reads_the_witness_shas_touched_packets_and_holds(self) -> None:
        mod = load()
        packet = "docs/benchmarks/GO-CALL-GRAPH-ALTERNATIVES-2026-08-10.md"
        run, seen = self._packet_reader(
            [packet, "docs/benchmarks/NATIVE-IMPLEMENTATION-COMPARISONS.md",
             "internal/codegraph/compare.go"],
            {packet: self.PACKET_6205,
             "docs/benchmarks/NATIVE-IMPLEMENTATION-COMPARISONS.md": self.REGISTRY})
        mod.run_capture = run
        binds, reason = mod.evidence_binds_closure(
            ROOT, {"number": 6205, "sha": "8595070f42"})
        self.assertFalse(binds)
        self.assertIn(mod.EVIDENCE_INCOMPLETE_HOLD, reason)
        self.assertEqual(seen[0], ["git", "diff-tree", "--no-commit-id",
                                   "--name-only", "-r", "8595070f42"])
        # only the docs/benchmarks/ markdown is read back; the .go file is never opened.
        self.assertEqual(seen[1], ["git", "show", f"8595070f42:{packet}"])
        self.assertNotIn(["git", "show", "8595070f42:internal/codegraph/compare.go"],
                         seen)

    def test_commit_touching_no_packet_binds_without_reading_blobs(self) -> None:
        mod = load()
        run, seen = self._packet_reader(["internal/gateway/http.go"], {})
        mod.run_capture = run
        self.assertEqual(
            mod.evidence_binds_closure(ROOT, {"number": 42, "sha": "abc1234567"}),
            (True, None))
        self.assertEqual(len(seen), 1)  # the file list only

    def test_unreadable_file_list_allows_with_a_note(self) -> None:
        # ASYMMETRY ON PURPOSE, matching disclaimer_binds_closure: this gate refuses on
        # POSITIVE evidence only, so an unreadable commit ALLOWS with an audit note
        # rather than failing closed. Absent evidence is not evidence of incompleteness.
        mod = load()
        mod.run_capture = lambda cmd, cwd, timeout: (128, "", "fatal: bad object")
        binds, note = mod.evidence_binds_closure(ROOT, {"number": 1, "sha": "deadbeef"})
        self.assertTrue(binds)
        self.assertIn(mod.EVIDENCE_UNREADABLE_NOTE, note)
        self.assertNotIn(mod.EVIDENCE_INCOMPLETE_HOLD, note)

    def test_missing_sha_allows_with_a_note_and_never_shells_out(self) -> None:
        mod = load()

        def boom(cmd, cwd, timeout):
            raise AssertionError("an empty sha must not reach git")

        mod.run_capture = boom
        binds, note = mod.evidence_binds_closure(ROOT, {"number": 1, "sha": ""})
        self.assertTrue(binds)
        self.assertIn(mod.EVIDENCE_UNREADABLE_NOTE, note)

    def test_unreadable_packet_does_not_mask_a_readable_incomplete_one(self) -> None:
        mod = load()
        gone = "docs/benchmarks/DELETED.md"
        packet = "docs/benchmarks/GO-CALL-GRAPH-ALTERNATIVES-2026-08-10.md"
        run, _ = self._packet_reader([gone, packet], {packet: self.PACKET_6205})
        mod.run_capture = run
        binds, reason = mod.evidence_binds_closure(
            ROOT, {"number": 6205, "sha": "8595070f42"})
        self.assertFalse(binds)
        self.assertIn(mod.EVIDENCE_INCOMPLETE_HOLD, reason)

    # ---- evaluate-level wiring -------------------------------------------

    def _patch_through_to_evidence(self, mod, number, sha, subject) -> None:
        """Witness + claim-bind all PASS, so the evidence gate decides. It runs beside
        the #5865 message gate, ahead of every gh probe -- both are local git reads."""
        mod.load_audit = lambda root, audit_json, max_commits: {
            "closure_rate": 0.5, "issues": [
                {"number": number, "title": "Benchmark against alternatives",
                 "bucket": "OPEN_WITNESSED",
                 "witnessed_commits": [{"sha": sha, "subject": subject}]}]}
        mod.origin_main_resolvable = lambda root: False
        mod.coverage_binds_closure = lambda root, row: (True, None)
        mod.reopen_blocks_close = lambda root, row: (True, None)
        mod.observed_effect_binds_closure = lambda root, row: (True, None)
        mod.disclaimer_binds_closure = lambda root, row: (True, None)  # inert #5865
        mod.reverify = lambda root, sha_: {
            "witness_ok": True, "verdict": "OK", "witness": "diff-witnessed",
            "claim_kind": "code_effect", "touches_code": True, "reason": None}

    def test_evaluate_holds_incomplete_packet_and_never_calls_gh(self) -> None:
        mod = load()
        self._patch_through_to_evidence(
            mod, 6205, "8595070f42",
            "feat(codegraph): add call graph alternatives comparison #6205")
        packet = "docs/benchmarks/GO-CALL-GRAPH-ALTERNATIVES-2026-08-10.md"
        run, seen = self._packet_reader([packet], {packet: self.PACKET_6205})

        def guarded(cmd, cwd, timeout):
            if cmd[:1] == ["gh"]:
                raise AssertionError(
                    "an issue whose own packet says INCOMPLETE must never reach "
                    "gh issue close")
            return run(cmd, cwd, timeout)

        mod.run_capture = guarded
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        r = p["results"][0]
        self.assertEqual(r["action"], "skip_incomplete_evidence")
        self.assertIn(mod.EVIDENCE_INCOMPLETE_HOLD, r["reason"])
        self.assertEqual(p["counts"]["skipped_incomplete_evidence"], 1)
        self.assertEqual(p["counts"]["closed"], 0)
        self.assertEqual(mod.close_decision("skip_incomplete_evidence"), "hold")
        self.assertIn("incomplete_evidence=1", mod.render(p))

    def test_evaluate_complete_evidence_still_closes(self) -> None:
        mod = load()
        self._patch_through_to_evidence(
            mod, 6205, "8595070f42",
            "feat(codegraph): add call graph alternatives comparison #6205")
        packet = "docs/benchmarks/GO-CALL-GRAPH-ALTERNATIVES-2026-08-10.md"
        complete = self.PACKET_6205.replace("**INCOMPLETE**", "**COMPLETE**")
        run, _ = self._packet_reader([packet], {packet: complete})
        base = gh_close_then_state()  # readback confirms CLOSED

        def either(cmd, cwd, timeout):
            return base(cmd, cwd, timeout) if cmd[:1] == ["gh"] else run(cmd, cwd, timeout)

        mod.run_capture = either
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        self.assertEqual(p["results"][0]["action"], "closed")
        self.assertEqual(p["counts"]["closed"], 1)
        self.assertEqual(p["counts"]["skipped_incomplete_evidence"], 0)

    def test_evaluate_dry_run_reflects_the_evidence_hold(self) -> None:
        mod = load()
        self._patch_through_to_evidence(
            mod, 6122, "1ba26ad4f7",
            "feat(cacheobs): add telemetry alternatives comparison #6122")
        packet = "docs/benchmarks/CACHE-OBSERVABILITY-ALTERNATIVES-2026-08-10.md"
        run, _ = self._packet_reader([packet], {packet: self.PACKET_6122})
        mod.run_capture = run
        p = mod.evaluate(ROOT, limit=10, live=False, audit_json=None, max_commits=600)
        self.assertEqual(p["results"][0]["action"], "skip_incomplete_evidence")
        self.assertEqual(p["counts"]["would_close"], 0)
        self.assertEqual(p["counts"]["skipped_incomplete_evidence"], 1)

    def test_evaluate_unreadable_evidence_notes_the_gate_and_still_closes(self) -> None:
        # the abstention is VISIBLE on the row, matching every other abstaining gate.
        mod = load()
        self._patch_through_to_evidence(
            mod, 6205, "8595070f42", "feat(codegraph): add call graph comparison #6205")
        base = gh_close_then_state()

        def run(cmd, cwd, timeout):
            if cmd[:2] == ["git", "diff-tree"]:
                return 128, "", "fatal: bad object"
            return base(cmd, cwd, timeout)

        mod.run_capture = run
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        r = p["results"][0]
        self.assertEqual(r["action"], "closed")
        self.assertEqual(p["counts"]["skipped_incomplete_evidence"], 0)
        self.assertTrue(any(mod.EVIDENCE_UNREADABLE_NOTE in n
                            for n in r.get("gate_notes", [])))


class EffectSurvivalRepositoryFixtureTest(unittest.TestCase):
    def test_base_candidate_exact_revert_and_surviving_control(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            def git(*args):
                return subprocess.check_output(
                    ["git", *args], cwd=repo, text=True,
                    creationflags=no_window_creationflags()).strip()
            git("init", "-q")
            git("config", "user.name", "fixture")
            git("config", "user.email", "fixture@example.invalid")
            path = repo / "effect.txt"
            path.write_text("base\n")
            git("add", "effect.txt")
            git("commit", "-q", "-m", "base")
            path.write_text("candidate\n")
            git("commit", "-q", "-am", "candidate")
            candidate = git("rev-parse", "HEAD")
            path.write_text("base\n")
            git("commit", "-q", "-am", "exact revert")
            revert_tip = git("rev-parse", "HEAD")
            ok, reason = mod.effect_survives_at_tip(repo, candidate, revert_tip)
            self.assertFalse(ok)
            self.assertIn(mod.EFFECT_REVERTED_HOLD, reason)
            path.write_text("survives\n")
            git("commit", "-q", "-am", "new surviving effect")
            surviving_tip = git("rev-parse", "HEAD")
            self.assertEqual(mod.effect_survives_at_tip(
                repo, candidate, surviving_tip), (True, None))


class EffectSurvivalGateTest(unittest.TestCase):
    def test_exact_revert_at_tip_is_held_with_bound_ids(self) -> None:
        mod = load()
        calls = []

        def run(cmd, cwd, timeout):
            calls.append(cmd)
            if cmd[:3] == ["git", "rev-parse", "--verify"]:
                return 0, "base000\n", ""
            if cmd[:2] == ["git", "diff-tree"]:
                return 0, "internal/model/hot.go\n", ""
            if cmd[:3] == ["git", "diff", "--quiet"]:
                return 0, "", ""  # tip restored candidate parent on touched path
            raise AssertionError(cmd)

        mod.run_capture = run
        ok, reason = mod.effect_survives_at_tip(ROOT, "candidate123", "revert456")
        self.assertFalse(ok)
        self.assertIn(mod.EFFECT_REVERTED_HOLD, reason)
        self.assertIn("candidate=candidate123", reason)
        self.assertIn("tip=revert456", reason)
        self.assertEqual(calls[-1], ["git", "diff", "--quiet", "base000",
                                    "revert456", "--", "internal/model/hot.go"])

    def test_surviving_candidate_at_tip_passes(self) -> None:
        mod = load()

        def run(cmd, cwd, timeout):
            if cmd[:3] == ["git", "rev-parse", "--verify"]:
                return 0, "base000\n", ""
            if cmd[:2] == ["git", "diff-tree"]:
                return 0, "internal/model/hot.go\n", ""
            if cmd[:3] == ["git", "diff", "--quiet"]:
                return 1, "", ""  # candidate effect differs from parent at tip
            raise AssertionError(cmd)

        mod.run_capture = run
        self.assertEqual(mod.effect_survives_at_tip(
            ROOT, "candidate123", "tip789"), (True, None))

    def test_unreadable_comparison_fails_closed(self) -> None:
        mod = load()

        def run(cmd, cwd, timeout):
            if cmd[:3] == ["git", "rev-parse", "--verify"]:
                return 0, "base000\n", ""
            if cmd[:2] == ["git", "diff-tree"]:
                return 0, "internal/model/hot.go\n", ""
            if cmd[:3] == ["git", "diff", "--quiet"]:
                return 128, "", "fatal: bad object"
            raise AssertionError(cmd)

        mod.run_capture = run
        ok, reason = mod.effect_survives_at_tip(ROOT, "candidate123", "tip789")
        self.assertFalse(ok)
        self.assertIn(mod.EFFECT_SURVIVAL_UNKNOWN_HOLD, reason)
        self.assertIn("candidate=candidate123", reason)
        self.assertIn("tip=tip789", reason)

    def test_close_decision_holds_survival_failures(self) -> None:
        mod = load()
        self.assertEqual(mod.close_decision("skip_effect_reverted"), "hold")
        self.assertEqual(mod.close_decision("skip_effect_survival_unknown"), "hold")


if __name__ == "__main__":
    unittest.main()
