#!/usr/bin/env python3
"""Hermetic tests for tools/cross_agent_ablate.py — the cross-agent (Regime B) controller.

NO network, NO `claude`, NO `fak` binary: every test drives the PURE core (the session_audit
adapter, the journal counter, the CI95 stats, the success-gate, the model-named refusal, the
two-number decompose) over fixtures built in temp files. This is the gate that locks the
validity contract from epic #607 rung 3 (#623): a regression that summed the token vector,
reported a 'saved' number off a failed arm, or collapsed the two decomposed numbers into one
would turn a test red here.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "cross_agent_ablate.py"


def load():
    spec = importlib.util.spec_from_file_location("cross_agent_ablate", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = load()


def _assistant(msg_id, *, inp, out, cread, ccreate, tool=None, tool_id=None, model="claude-opus-4-8"):
    if tool:
        content = [{"type": "tool_use", "name": tool, "id": tool_id or f"toolu_{tool}", "input": {}}]
    else:
        content = [{"type": "text", "text": "ok"}]
    return {
        "type": "assistant",
        "timestamp": "2026-06-24T00:00:00.000Z",
        "uuid": f"uuid-{msg_id}-{out}-{cread}",
        "message": {"id": msg_id, "model": model,
                    "usage": {"input_tokens": inp, "output_tokens": out,
                              "cache_read_input_tokens": cread, "cache_creation_input_tokens": ccreate},
                    "content": content}}


def _write_jsonl(records):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in records:
        tmp.write(json.dumps(r) + "\n")
    tmp.close()
    return tmp.name


def _rep(arm, *, success, model="claude-opus-4-8", out=100, inp=10, cr=50, cc=200, turns=2, adj=None):
    rep = {"arm": arm, "model": model, "completed": True, "success": success, "turns": turns,
           "tokens": {"input": inp, "output": out, "provider_cache_read": cr, "cache_create": cc},
           "tools": {"Write": 1}, "wall_seconds": 1.0}
    if adj is not None:
        rep["adjudication"] = adj
    return rep


class TokenDecomposition(unittest.TestCase):
    def test_session_audit_adapter_decomposes_and_dedups(self):
        # Two billed turns + a DUPLICATE line of the first (same message.id) — session_audit
        # folds the duplicate once, so tokens must not double.
        path = _write_jsonl([
            _assistant("m1", inp=10, out=100, cread=50, ccreate=200, tool="Write"),
            _assistant("m1", inp=10, out=100, cread=50, ccreate=200, tool="Write"),  # dup
            _assistant("m2", inp=5, out=20, cread=0, ccreate=0),
        ])
        a = M.audit_transcript(path)
        self.assertEqual(a["tokens"], {"input": 15, "output": 120,
                                       "provider_cache_read": 50, "cache_create": 200})
        self.assertEqual(a["turns"], 2)
        self.assertEqual(a["tools"], {"Write": 1})   # the dup line shares the tool_use id -> once
        self.assertEqual(a["model"], "claude-opus-4-8")

    def test_tools_counted_across_split_lines(self):
        # The real headless pattern (the bug this locks): Claude Code splits ONE billed turn
        # across lines sharing a message.id -- streamed text on one, the tool_use on the next.
        # session_audit de-dups by id and skips the tool line; the adapter must still count it.
        path = _write_jsonl([
            _assistant("mA", inp=10, out=50, cread=0, ccreate=0),                              # text line
            _assistant("mA", inp=10, out=50, cread=0, ccreate=0, tool="Write", tool_id="toolu_1"),  # tool line, SAME id
        ])
        a = M.audit_transcript(path)
        self.assertEqual(a["tools"], {"Write": 1})   # recovered despite the de-dup
        self.assertEqual(a["turns"], 1)              # one billed turn (folded once)
        self.assertEqual(a["tokens"]["output"], 50)  # usage folded once, NOT doubled

    def test_token_vector_is_never_summed(self):
        tv = M._token_vector(10, 100, 50, 200)
        self.assertEqual(set(tv), {"input", "output", "provider_cache_read", "cache_create"})
        self.assertNotIn("total", tv)
        self.assertNotIn("tokens", tv)

    def test_total_input_excludes_output(self):
        tv = {"input": 10, "output": 9999, "provider_cache_read": 5, "cache_create": 20}
        self.assertEqual(M.total_input_tokens(tv), 35)  # output is NOT folded in

    def test_rep_falls_back_to_result_usage_without_transcript(self):
        result = {"usage": {"input_tokens": 7, "output_tokens": 3,
                            "cache_read_input_tokens": 19, "cache_creation_input_tokens": 5},
                  "num_turns": 1,
                  "modelUsage": {"claude-opus-4-8": {"inputTokens": 7, "cacheCreationInputTokens": 5}}}
        rep = M.rep_from_result_json(result, arm=M.ARM_CLAUDE, success=True, completed=True, wall_seconds=2.0)
        self.assertEqual(rep["tokens"], {"input": 7, "output": 3, "provider_cache_read": 19, "cache_create": 5})
        self.assertEqual(rep["model"], "claude-opus-4-8")
        self.assertEqual(rep["turns"], 1)


class JournalCounting(unittest.TestCase):
    def test_counts_verdicts_and_separates_vdso(self):
        path = _write_jsonl([
            {"seq": 1, "kind": "DECIDE", "tool": "Write", "verdict": "ALLOW"},
            {"seq": 2, "kind": "DENY", "tool": "Bash", "verdict": "DENY"},
            {"seq": 3, "kind": "DECIDE", "tool": "Edit", "verdict": "TRANSFORM"},
            {"seq": 4, "kind": "QUARANTINE", "tool": "Read", "verdict": "QUARANTINE"},
            {"seq": 5, "kind": "DECIDE", "tool": "Glob", "verdict": "DEFER"},
            {"seq": 6, "kind": "VDSO_HIT", "tool": "Read", "verdict": "ALLOW"},  # a cache hit, NOT a decision
        ])
        c = M.count_adjudications(path)
        self.assertEqual(c["allowed"], 1)
        self.assertEqual(c["denied"], 1)
        self.assertEqual(c["repaired"], 1)      # TRANSFORM
        self.assertEqual(c["quarantined"], 1)
        self.assertEqual(c["deferred"], 1)
        self.assertEqual(c["vdso_hits"], 1)     # tallied separately
        self.assertEqual(c["journal_rows"], 5)  # VDSO_HIT excluded from decisions

    def test_missing_journal_is_zeros_not_error(self):
        c = M.count_adjudications("/no/such/journal.jsonl")
        self.assertEqual(c["journal_rows"], 0)
        self.assertEqual(c["denied"], 0)

    def test_tolerates_torn_final_line(self):
        path = _write_jsonl([{"seq": 1, "kind": "DECIDE", "verdict": "ALLOW"}])
        with open(path, "a", encoding="utf-8") as fh:
            fh.write('{"seq":2,"kind":"DECI')  # crash mid-write
        c = M.count_adjudications(path)
        self.assertEqual(c["allowed"], 1)
        self.assertEqual(c["journal_rows"], 1)


class CI95(unittest.TestCase):
    def test_known_interval(self):
        r = M.mean_ci95([2, 4, 6])              # mean 4, stdev 2, n 3, t_.975(2)=4.303
        self.assertEqual(r["mean"], 4.0)
        self.assertEqual(r["n"], 3)
        self.assertAlmostEqual(r["ci95"], 4.303 * 2.0 / (3 ** 0.5), places=3)

    def test_single_rep_has_no_interval(self):
        r = M.mean_ci95([42])
        self.assertEqual(r["mean"], 42.0)
        self.assertIsNone(r["ci95"])            # one rep cannot bound variance — honest None
        self.assertEqual(r["n"], 1)

    def test_empty_is_none(self):
        self.assertEqual(M.mean_ci95([]), {"mean": None, "ci95": None, "n": 0})


class SuccessGate(unittest.TestCase):
    def _arms(self, base_success, treat_success, *, treat_model="claude-opus-4-8"):
        base = M.aggregate_arm(M.ARM_CLAUDE, [_rep(M.ARM_CLAUDE, success=s, out=100) for s in base_success])
        treat = M.aggregate_arm(M.ARM_CLAUDE_FAK,
                                [_rep(M.ARM_CLAUDE_FAK, success=s, out=80, model=treat_model,
                                      adj={"denied": 1, "repaired": 0, "quarantined": 0,
                                           "allowed": 2, "deferred": 1, "vdso_hits": 0, "journal_rows": 3})
                                 for s in treat_success])
        return base, treat

    def test_gate_open_when_both_succeed(self):
        base, treat = self._arms([True] * 5, [True] * 5)
        rep = M.build_report({"id": "pong"}, [base, treat], k=5, generated_by="t", command="t")
        comp = rep["comparison"]
        self.assertTrue(comp["gated"])
        self.assertTrue(comp["variance_ok"])    # >=5 successful reps each
        self.assertEqual(comp["kernel_efficiency"]["output_tokens_ratio"], 0.8)
        self.assertEqual(comp["kernel_efficiency"]["saved_output_tokens"], 20.0)
        self.assertEqual(rep["headline_model"], "claude-opus-4-8")

    def test_gate_closed_when_treatment_all_fail(self):
        base, treat = self._arms([True] * 5, [False] * 5)
        rep = M.build_report({"id": "pong"}, [base, treat], k=5, generated_by="t", command="t")
        comp = rep["comparison"]
        self.assertFalse(comp["gated"])
        self.assertNotIn("kernel_efficiency", comp)   # NO saved number off a failed arm
        self.assertIn("success-gate CLOSED", comp["reason"])

    def test_variance_flag_false_below_five(self):
        base, treat = self._arms([True] * 3, [True] * 3)
        rep = M.build_report({"id": "pong"}, [base, treat], k=3, generated_by="t", command="t")
        self.assertTrue(rep["comparison"]["gated"])
        self.assertFalse(rep["comparison"]["variance_ok"])  # K<5

    def test_gate_requires_completed_not_just_success(self):
        # The contract is both.completed && both.success. A rep flagged success but NOT
        # completed is incoherent data (the live runner makes success imply completed; only
        # a hand-supplied offline reps file can desync them) — it must NOT count as a success
        # nor open the gate, or a 'saved' number would rest on a rep that never finished.
        bad = [{"arm": M.ARM_CLAUDE_FAK, "model": "claude-opus-4-8", "completed": False,
                "success": True, "turns": 2,
                "tokens": {"input": 10, "output": 80, "provider_cache_read": 50, "cache_create": 200},
                "tools": {}, "wall_seconds": 1.0} for _ in range(5)]
        treat = M.aggregate_arm(M.ARM_CLAUDE_FAK, bad)
        self.assertEqual(treat["success_reps"], 0)      # success-without-completed is not a success
        self.assertEqual(treat["completed_reps"], 0)
        base = M.aggregate_arm(M.ARM_CLAUDE, [_rep(M.ARM_CLAUDE, success=True)] * 5)
        comp = M.build_report({"id": "pong"}, [base, treat], k=5, generated_by="t", command="t")["comparison"]
        self.assertFalse(comp["gated"])                 # gate stays CLOSED
        self.assertNotIn("kernel_efficiency", comp)     # no saved number off an unfinished arm


class ModelNamedAndDecompose(unittest.TestCase):
    def test_kernel_efficiency_refused_when_model_varies(self):
        base = M.aggregate_arm(M.ARM_CLAUDE, [_rep(M.ARM_CLAUDE, success=True, model="claude-opus-4-8")] * 5)
        treat = M.aggregate_arm(M.ARM_CLAUDE_FAK, [_rep(M.ARM_CLAUDE_FAK, success=True, model="claude-sonnet-4-6")] * 5)
        rep = M.build_report({"id": "pong"}, [base, treat], k=5, generated_by="t", command="t")
        comp = rep["comparison"]
        self.assertTrue(comp["gated"])                       # both succeeded -> success-gate open
        self.assertTrue(comp["kernel_efficiency"]["refused"])  # but model NOT held constant
        self.assertIn("not held constant", comp["kernel_efficiency"]["reason"].lower())

    def test_kernel_efficiency_refused_on_within_arm_model_drift(self):
        # A single arm that DRIFTED models (3 opus + 2 sonnet) has modal=opus, which would
        # FALSELY match a pure-opus baseline on the modal check alone — but its token means
        # pool across two models, so the kernel delta is unattributable. The refusal must
        # fire on models_seen (within-arm drift), not just the cross-arm modal match.
        base = M.aggregate_arm(M.ARM_CLAUDE, [_rep(M.ARM_CLAUDE, success=True, model="claude-opus-4-8")] * 5)
        treat = M.aggregate_arm(
            M.ARM_CLAUDE_FAK,
            [_rep(M.ARM_CLAUDE_FAK, success=True, model="claude-opus-4-8")] * 3
            + [_rep(M.ARM_CLAUDE_FAK, success=True, model="claude-sonnet-4-6")] * 2)
        self.assertEqual(treat["model"], "claude-opus-4-8")      # modal is still opus
        self.assertEqual(len(treat["models_seen"]), 2)          # but two models were seen
        comp = M.build_report({"id": "pong"}, [base, treat], k=5, generated_by="t", command="t")["comparison"]
        self.assertTrue(comp["gated"])                          # both succeeded
        self.assertTrue(comp["kernel_efficiency"]["refused"])   # yet kernel number REFUSED
        self.assertIn("within an arm", comp["kernel_efficiency"]["reason"].lower())

    def test_two_numbers_never_one(self):
        base = M.aggregate_arm(M.ARM_CLAUDE, [_rep(M.ARM_CLAUDE, success=True)] * 5)
        treat = M.aggregate_arm(M.ARM_CLAUDE_FAK, [_rep(M.ARM_CLAUDE_FAK, success=True, out=80)] * 5)
        comp = M.build_report({"id": "pong"}, [base, treat], k=5, generated_by="t", command="t")["comparison"]
        # The decompose is TWO distinct keys: a resource ratio (model constant) AND a
        # capability rate (model varies) — never a single conflated "improvement".
        self.assertIn("kernel_efficiency", comp)
        self.assertIn("agent_capability", comp)
        self.assertIn("success_rate", comp["agent_capability"]["baseline"])
        self.assertIn("success_rate", comp["agent_capability"]["treatment"])

    def test_arm_names_its_model_and_adjudication_totals(self):
        treat = M.aggregate_arm(M.ARM_CLAUDE_FAK,
                                [_rep(M.ARM_CLAUDE_FAK, success=True,
                                      adj={"denied": 1, "repaired": 1, "quarantined": 0, "allowed": 2,
                                           "deferred": 0, "vdso_hits": 0, "journal_rows": 4})] * 5)
        self.assertEqual(treat["model"], "claude-opus-4-8")
        self.assertEqual(treat["adjudication_totals"]["denied"], 5)
        self.assertEqual(treat["adjudication_totals"]["repaired"], 5)


class OfflineRegen(unittest.TestCase):
    def test_report_rebuilds_from_saved_reps(self):
        # The artifact embeds raw reps so the report regenerates offline (reproducibility).
        reps = {"claude_code": [_rep(M.ARM_CLAUDE, success=True) for _ in range(5)],
                "claude_code+fak": [_rep(M.ARM_CLAUDE_FAK, success=True, out=80,
                                         adj={"denied": 0, "repaired": 0, "quarantined": 0, "allowed": 1,
                                              "deferred": 0, "vdso_hits": 0, "journal_rows": 1}) for _ in range(5)]}
        arms = [M.aggregate_arm(arm, rs) for arm, rs in reps.items()]
        rep = M.build_report({"id": "pong"}, arms, k=5, generated_by="t", command="t")
        self.assertEqual(rep["schema"], M.SCHEMA)
        self.assertTrue(rep["regime"].startswith("B"))
        self.assertEqual(len(rep["arms"]), 2)
        # raw reps survive into the artifact so `report --reps` can re-derive every number.
        self.assertEqual(len(rep["arms"][0]["raw_reps"]), 5)

    def test_report_rebuilds_from_committed_artifact(self):
        # A committed report ARTIFACT (arms[].raw_reps embedded, no top-level "reps" key)
        # regenerates every aggregate from itself via reps_from_doc — so the COMMITTED file
        # is the reproducible source, not only the {"reps":{...}} reps-file shape.
        reps = {"claude_code": [_rep(M.ARM_CLAUDE, success=True) for _ in range(5)],
                "claude_code+fak": [_rep(M.ARM_CLAUDE_FAK, success=True, out=80,
                                         adj={"denied": 0, "repaired": 0, "quarantined": 0, "allowed": 1,
                                              "deferred": 0, "vdso_hits": 0, "journal_rows": 1}) for _ in range(5)]}
        arms = [M.aggregate_arm(arm, rs) for arm, rs in reps.items()]
        artifact = M.build_report({"id": "pong"}, arms, k=5, generated_by="t", command="t",
                                  app_version="9.9.9", wall_clock_utc="2026-06-25T00:00:00Z")
        self.assertNotIn("reps", artifact)              # it is a report artifact, not a reps file
        task, k, reps_by_arm, wall, ver = M.reps_from_doc(artifact)
        self.assertEqual((k, ver, wall), (5, "9.9.9", "2026-06-25T00:00:00Z"))
        self.assertEqual(set(reps_by_arm), {"claude_code", "claude_code+fak"})
        rebuilt = M.build_report(task, [M.aggregate_arm(a, r) for a, r in reps_by_arm.items()],
                                 k=k, generated_by="t", command="t")
        # the data (arms + comparison) round-trips identically off the embedded raw_reps
        self.assertEqual(rebuilt["arms"], artifact["arms"])
        self.assertEqual(rebuilt["comparison"], artifact["comparison"])


if __name__ == "__main__":
    unittest.main()
