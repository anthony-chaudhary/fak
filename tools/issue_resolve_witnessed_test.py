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
        # only "onmain" is an ancestor of origin/main; "localonly" is not.
        mod.reachable_from_origin = lambda root, sha: sha == "onmain"

    def test_unpushed_commit_is_skipped_not_closed(self) -> None:
        mod = load()
        self._patch(mod, gate_active=True)
        closed: list = []
        mod.run_capture = lambda cmd, cwd, timeout: (closed.append(cmd) or (0, "", ""))
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
        mod.run_capture = lambda cmd, cwd, timeout: (0, "", "")
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None, max_commits=600)
        actions = {r["number"]: r["action"] for r in p["results"]}
        self.assertEqual(actions[100], "closed")
        self.assertEqual(actions[90], "closed")           # not gated -> both close
        self.assertEqual(p["counts"]["skipped_unpushed"], 0)
        self.assertEqual(p["pushed_gate"], "no-origin-ref")

    def test_require_pushed_false_disables_gate(self) -> None:
        mod = load()
        self._patch(mod, gate_active=True)  # origin resolvable...
        mod.run_capture = lambda cmd, cwd, timeout: (0, "", "")
        # ...but the caller opted out: unpushed commits close anyway.
        p = mod.evaluate(ROOT, limit=10, live=True, audit_json=None,
                         max_commits=600, require_pushed=False)
        actions = {r["number"]: r["action"] for r in p["results"]}
        self.assertEqual(actions[90], "closed")
        self.assertEqual(p["pushed_gate"], "disabled")


if __name__ == "__main__":
    unittest.main()
