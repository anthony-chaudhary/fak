#!/usr/bin/env python3
"""Hermetic tests for tools/codex_turn_health.py."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "codex_turn_health.py"


def load():
    spec = importlib.util.spec_from_file_location("codex_turn_health", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = load()


# --- synthetic rollout-row builders (mirror the codex jsonl schema) ---------
def ctx(model="gpt-5.6-sol"):
    return {"type": "turn_context", "payload": {"model": model}}


def started():
    return {"type": "event_msg", "payload": {"type": "task_started"}}


def complete():
    return {"type": "event_msg", "payload": {"type": "task_complete"}}


def msg(text):
    return {"type": "event_msg", "payload": {"type": "agent_message", "message": text}}


def call():
    return {"type": "response_item", "payload": {"type": "function_call"}}


def tokens(n):
    return {"type": "event_msg", "payload": {"type": "token_count",
            "info": {"last_token_usage": {"input_tokens": n}}}}


def compacted():
    return {"type": "compacted", "payload": {}}


class ClassifyTest(unittest.TestCase):
    def test_guard_refusal_marker(self):
        self.assertEqual(M.classify_zero_tool(
            "All proposed tool calls were refused by the fak kernel: shell_command REQUIRE_WITNESS"),
            "guard_refused")

    def test_preamble_marker(self):
        self.assertEqual(M.classify_zero_tool(
            "I'll use the `super-loop` skill because the goal is to audit workers"),
            "preamble_noop")

    def test_plain_prose_is_talk_only(self):
        self.assertEqual(M.classify_zero_tool("Here is a status update on the work."), "talk_only")

    def test_empty_is_silent(self):
        self.assertEqual(M.classify_zero_tool(""), "silent")


class FoldRowsTest(unittest.TestCase):
    def test_full_turn_counts_tool_not_zero(self):
        rows = [ctx(), started(), call(), call(), msg("done"), complete()]
        s = M.fold_rows(rows)
        self.assertEqual(s["turns"], 1)
        self.assertEqual(s["tool_calls"], 2)
        self.assertEqual(s["zero_tool_total"], 0)
        self.assertEqual(s["model"], "gpt-5.6-sol")

    def test_zero_tool_turn_classified(self):
        rows = [started(), msg("All proposed tool calls were refused by the fak kernel"), complete()]
        s = M.fold_rows(rows)
        self.assertEqual(s["zero_tool"], {"guard_refused": 1})

    def test_compaction_records_last_input_occupancy(self):
        rows = [started(), tokens(92000), call(), complete(), compacted()]
        s = M.fold_rows(rows)
        self.assertEqual(s["compactions"], [92000])

    def test_no_double_count_of_compaction(self):
        # A real compaction emits BOTH `compacted` and event_msg/context_compacted;
        # only the top-level `compacted` must count.
        rows = [started(), tokens(90000), complete(), compacted(),
                {"type": "event_msg", "payload": {"type": "context_compacted"}}]
        self.assertEqual(len(M.fold_rows(rows)["compactions"]), 1)


class RollUpTest(unittest.TestCase):
    def test_flags_and_buckets(self):
        # One healthy session, one guard-refusal-loop session, one premature compaction.
        healthy = {"session": "a", "model": "m", "turns": 10, "tool_calls": 200,
                   "zero_tool": {}, "zero_tool_total": 0, "compactions": [93000]}
        refusal = {"session": "b", "model": "gpt-5.6-sol", "turns": 8, "tool_calls": 5,
                   "zero_tool": {"guard_refused": 6}, "zero_tool_total": 6, "compactions": []}
        stuck = {"session": "c", "model": "m", "turns": 6, "tool_calls": 0,
                 "zero_tool": {"preamble_noop": 6}, "zero_tool_total": 6, "compactions": [7000]}
        rep = M.roll_up([healthy, refusal, stuck])
        self.assertEqual(rep["sessions_with_turns"], 3)
        self.assertEqual(rep["compaction"]["near_budget_96k"], 1)
        self.assertEqual(rep["compaction"]["premature_lt40k"], 1)
        # b is a refusal loop; b and c are turn-inflation offenders.
        self.assertEqual([r["session"] for r in rep["guard_refusal_loops"]], ["b"])
        self.assertEqual({r["session"] for r in rep["turn_inflation"]}, {"b", "c"})
        joined = " ".join(rep["flags"])
        self.assertIn("GUARD_REFUSAL_LOOPS", joined)
        self.assertIn("PREMATURE_COMPACTION", joined)
        self.assertIn("HIGH_ZERO_TOOL_RATE", joined)

    def test_empty_corpus_is_clean(self):
        rep = M.roll_up([])
        self.assertEqual(rep["totals"]["turns"], 0)
        self.assertEqual(rep["flags"], [])


if __name__ == "__main__":
    unittest.main()
