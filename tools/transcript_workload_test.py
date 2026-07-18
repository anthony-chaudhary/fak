#!/usr/bin/env python3
"""Hermetic tests for tools/transcript_workload.py.

The workload profiler's core is a set of pure stat helpers plus ``analyze_turns``,
a fold that collapses Claude Code's per-content-block split records into one
logical turn keyed by ``message.id`` (usage counted ONCE) and attributes trailing
tool_result bytes as R. These tests pin ``_txt_len`` (nested str/list/dict),
``_pct`` (nearest-rank + None-drop), and ``_stats`` (shape + int/float rounding),
then drive ``analyze_turns`` over a SYNTHETIC transcript asserting the doc's
load-bearing guarantee: split records merge (out_tokens not triple-counted) and
the tool-call fraction reflects logical turns, not raw records.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "transcript_workload.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("transcript_workload", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load()


class TestTxtLen(unittest.TestCase):
    def test_plain_string(self):
        self.assertEqual(MOD._txt_len("hello"), 5)

    def test_list_of_blocks_sums_content_and_text(self):
        self.assertEqual(MOD._txt_len([{"content": "abc"}, {"text": "de"}]), 5)

    def test_nested_dict(self):
        self.assertEqual(MOD._txt_len({"content": "abcd"}), 4)

    def test_non_text_is_zero(self):
        self.assertEqual(MOD._txt_len(42), 0)


class TestPct(unittest.TestCase):
    def test_nearest_rank(self):
        self.assertEqual(MOD._pct([1, 2, 3, 4, 5], 90), 5)
        self.assertEqual(MOD._pct([1, 2, 3, 4, 5], 10), 1)
        self.assertEqual(MOD._pct([1, 2, 3, 4, 5], 50), 3)

    def test_drops_none_and_empty(self):
        self.assertEqual(MOD._pct([None, 2, None, 4], 50), 2)
        self.assertIsNone(MOD._pct([], 50))
        self.assertIsNone(MOD._pct([None], 50))


class TestStats(unittest.TestCase):
    def test_empty_reports_zero_n(self):
        self.assertEqual(MOD._stats([]), {"n": 0})

    def test_shape_and_int_rounding(self):
        s = MOD._stats([1, 2, 3, 4])
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["min"], 1)
        self.assertEqual(s["max"], 4)
        self.assertEqual(s["median"], 2)  # statistics.median -> 2.5 -> round -> 2
        self.assertIsInstance(s["mean"], int)

    def test_float_mode_rounds_to_three(self):
        s = MOD._stats([0.1, 0.2, 0.3], ints=False)
        self.assertIsInstance(s["mean"], float)
        self.assertEqual(s["mean"], 0.2)


def _line(**kw):
    return json.dumps(kw)


class TestAnalyzeTurns(unittest.TestCase):
    def _write(self, lines):
        f = tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8")
        f.write("\n".join(lines) + "\n")
        f.close()
        return f.name

    def test_split_records_merge_into_one_turn(self):
        # ONE API response written as TWO records sharing message.id "m1" with the
        # SAME usage stamped on both. Must collapse to ONE turn, out counted once.
        usage = {"output_tokens": 100, "cache_read_input_tokens": 5000,
                 "cache_creation_input_tokens": 0, "input_tokens": 10}
        lines = [
            _line(type="assistant", message={
                "id": "m1", "usage": usage,
                "content": [{"type": "text", "text": "thinking"}]}),
            _line(type="assistant", message={
                "id": "m1", "usage": usage,
                "content": [{"type": "tool_use", "name": "Read"}]}),
            _line(type="user", message={"content": [
                {"type": "tool_result", "content": "x" * 40}]}),
            # two more standalone turns so profile-ish invariants hold
            _line(type="assistant", message={
                "id": "m2", "usage": usage, "content": [{"type": "text", "text": "a"}]}),
            _line(type="assistant", message={
                "id": "m3", "usage": usage, "content": [{"type": "text", "text": "b"}]}),
        ]
        prof = MOD.analyze_turns(self._write(lines))
        self.assertIsNotNone(prof)
        # 3 logical turns (m1 merged), NOT 4 records
        self.assertEqual(prof["n_turns"], 3)
        # only m1 issued a tool_use
        self.assertEqual(prof["n_tool_turns"], 1)
        self.assertEqual(prof["n_tool_calls"], 1)
        self.assertAlmostEqual(prof["tool_call_fraction"], 1 / 3)
        # usage counted once: decode for the merged turn is 100 (not 200)
        self.assertEqual(prof["decode_per_turn"][0], 100)
        # 40 chars of tool_result / 4 chars-per-tok = 10 result tokens on the tool turn
        self.assertEqual(prof["result_per_tool_turn"], [math.ceil(40 / MOD.CHARS_PER_TOK)])
        # Read is read-only -> read_only_fraction is 1.0
        self.assertEqual(prof["read_only_fraction"], 1.0)

    def test_no_assistant_turns_returns_none(self):
        lines = [_line(type="user", message={"content": "just a prompt"})]
        self.assertIsNone(MOD.analyze_turns(self._write(lines)))


if __name__ == "__main__":
    unittest.main()
