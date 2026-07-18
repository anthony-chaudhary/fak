#!/usr/bin/env python3
"""Hermetic tests for tools/agent_walltime.py.

The wall-clock attributor's pure surface is a timestamp parser, a duration
histogram binner, small record accessors, and human formatting. These tests pin
each on real inputs (Z-suffixed ISO stamps, boundary bin edges, malformed
records) and then drive the full ``fold`` over a SYNTHETIC transcript written to
a tempfile, asserting that a model gap and a tool gap land in the right buckets,
a huge gap is bucketed as IDLE (not model), and a parallel tool batch is counted
as ONE wall segment (no double counting) — the doc's load-bearing guarantees.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "agent_walltime.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("agent_walltime", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load()


class TestParseTs(unittest.TestCase):
    def test_parses_z_suffixed_iso(self):
        a = MOD.parse_ts("2026-01-01T00:00:00Z")
        b = MOD.parse_ts("2026-01-01T00:00:10Z")
        self.assertIsNotNone(a)
        self.assertEqual(b - a, 10.0)

    def test_none_and_junk_return_none(self):
        self.assertIsNone(MOD.parse_ts(None))
        self.assertIsNone(MOD.parse_ts(""))
        self.assertIsNone(MOD.parse_ts("not-a-time"))
        self.assertIsNone(MOD.parse_ts(12345))


class TestBinIndex(unittest.TestCase):
    def test_edges_map_to_expected_buckets(self):
        # BIN_EDGES = [2, 5, 15, 60, 300, inf]
        self.assertEqual(MOD.bin_index(0), 0)      # < 2
        self.assertEqual(MOD.bin_index(2), 1)      # >=2, <5
        self.assertEqual(MOD.bin_index(4.9), 1)
        self.assertEqual(MOD.bin_index(30), 3)     # >=15, <60
        self.assertEqual(MOD.bin_index(10_000), len(MOD.BIN_EDGES) - 1)


class TestRecordAccessors(unittest.TestCase):
    def test_content_blocks_only_from_list(self):
        self.assertEqual(MOD.content_blocks({"message": {"content": [1, 2]}}), [1, 2])
        self.assertEqual(MOD.content_blocks({"message": {"content": "str"}}), [])
        self.assertEqual(MOD.content_blocks({}), [])

    def test_model_of(self):
        self.assertEqual(MOD.model_of({"message": {"model": "opus"}}), "opus")
        self.assertEqual(MOD.model_of({"message": {}}), "")
        self.assertEqual(MOD.model_of({}), "")


class TestHmsAndPct(unittest.TestCase):
    def test_hms_formats(self):
        self.assertEqual(MOD.hms(45), "45s")
        self.assertEqual(MOD.hms(90), "1m30s")
        self.assertEqual(MOD.hms(3661), "1h01m")

    def test_pct_guards_zero_total(self):
        self.assertEqual(MOD.pct(1, 0), 0.0)
        self.assertEqual(MOD.pct(1, 4), 25.0)


def _rec(**kw):
    return json.dumps(kw)


class TestFold(unittest.TestCase):
    def _write(self, lines):
        f = tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8")
        f.write("\n".join(lines) + "\n")
        f.close()
        return f.name

    def test_model_and_tool_gaps_bucketed(self):
        # human prompt @0 -> assistant(w/ tool_use) @3 (model gap 3s)
        #   -> tool_result @8 (tool gap 5s)
        lines = [
            _rec(type="user", timestamp="2026-01-01T00:00:00Z",
                 message={"content": "do it"}),
            _rec(type="assistant", timestamp="2026-01-01T00:00:03Z",
                 message={"model": "opus", "content": [
                     {"type": "tool_use", "id": "t1", "name": "Read"}]}),
            _rec(type="user", timestamp="2026-01-01T00:00:08Z",
                 message={"content": [
                     {"type": "tool_result", "tool_use_id": "t1"}]}),
        ]
        out = MOD.fold(self._write(lines), idle_cutoff=600)
        b = out["buckets"]
        self.assertAlmostEqual(b["model"], 3.0)
        self.assertAlmostEqual(b["tools"], 5.0)
        self.assertEqual(b["idle"], 0.0)

    def test_huge_gap_is_idle_not_model(self):
        lines = [
            _rec(type="user", timestamp="2026-01-01T00:00:00Z",
                 message={"content": "hi"}),
            # 1-hour gap before the assistant answers -> IDLE, not model
            _rec(type="assistant", timestamp="2026-01-01T01:00:00Z",
                 message={"model": "opus", "content": [{"type": "text", "text": "ok"}]}),
        ]
        out = MOD.fold(self._write(lines), idle_cutoff=600)
        b = out["buckets"]
        self.assertEqual(b["model"], 0.0)
        self.assertAlmostEqual(b["idle"], 3600.0)

    def test_parallel_tool_batch_counts_wall_once(self):
        # assistant emits TWO tool_use; one tool_result record answers both.
        # The 4s wall segment must be counted ONCE, not 8s.
        lines = [
            _rec(type="user", timestamp="2026-01-01T00:00:00Z",
                 message={"content": "go"}),
            _rec(type="assistant", timestamp="2026-01-01T00:00:00Z",
                 message={"model": "opus", "content": [
                     {"type": "tool_use", "id": "a", "name": "Read"},
                     {"type": "tool_use", "id": "b", "name": "Grep"}]}),
            _rec(type="user", timestamp="2026-01-01T00:00:04Z",
                 message={"content": [
                     {"type": "tool_result", "tool_use_id": "a"},
                     {"type": "tool_result", "tool_use_id": "b"}]}),
        ]
        out = MOD.fold(self._write(lines), idle_cutoff=600)
        b = out["buckets"]
        self.assertAlmostEqual(b["tools"], 4.0)  # once, not 8


if __name__ == "__main__":
    unittest.main()
