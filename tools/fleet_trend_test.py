#!/usr/bin/env python3
"""Hermetic tests for fleet_trend: the sparkline + delta fold and the bounded ledger.

Pure/deterministic — a temp ledger and an injected `now`, no clock, no network."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_trend  # noqa: E402


class SparkTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(fleet_trend.spark([]), "")

    def test_flat_series_is_lowest_block(self):
        # all-equal (incl. a single point) must not fake a slope.
        self.assertEqual(fleet_trend.spark([3, 3, 3]), "▁▁▁")
        self.assertEqual(fleet_trend.spark([7]), "▁")

    def test_ramp_spans_low_to_high(self):
        s = fleet_trend.spark([0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(s[0], "▁")   # min -> lowest
        self.assertEqual(s[-1], "█")  # max -> highest
        self.assertEqual(len(s), 8)


class MetricsOfTest(unittest.TestCase):
    def test_extracts_load_bearing_scalars(self):
        snap = {
            "sessions": {"total": 5, "by_category": {"LIVE": 2, "AGENT": 3}},
            "accounts": {"usable": 1, "total": 4},
            "system": {"verdict": "NEEDS_YOU", "escalate": 2, "self_healing": 1},
        }
        m = fleet_trend.metrics_of(snap)
        self.assertEqual(m, {"usable": 1.0, "live": 2.0, "sessions": 5.0, "escalate": 2.0})

    def test_partial_snapshot_reads_zero(self):
        self.assertEqual(
            fleet_trend.metrics_of({}),
            {"usable": 0.0, "live": 0.0, "sessions": 0.0, "escalate": 0.0},
        )


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        # a nested path so the dir-creation branch is exercised.
        self.path = str(Path(self.dir.name) / "sub" / "history.jsonl")

    def test_append_creates_dir_and_tail_reads_back(self):
        fleet_trend.append(self.path, {"usable": 3, "live": 1, "sessions": 4, "escalate": 0}, "2026-07-01T00:00:00Z")
        fleet_trend.append(self.path, {"usable": 2, "live": 1, "sessions": 4, "escalate": 1}, "2026-07-01T01:00:00Z")
        rows = fleet_trend.tail(self.path, 24)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["usable"], 3)
        self.assertEqual(rows[-1]["escalate"], 1)
        self.assertEqual(rows[-1]["ts"], "2026-07-01T01:00:00Z")

    def test_append_is_bounded_to_cap(self):
        for i in range(10):
            fleet_trend.append(self.path, {"usable": i}, f"2026-07-01T00:{i:02d}:00Z", cap=3)
        rows = fleet_trend.tail(self.path, 100)
        self.assertEqual(len(rows), 3)                 # ring trimmed to cap
        self.assertEqual([r["usable"] for r in rows], [7, 8, 9])

    def test_tail_returns_last_n(self):
        for i in range(5):
            fleet_trend.append(self.path, {"usable": i}, f"2026-07-01T00:{i:02d}:00Z")
        self.assertEqual([r["usable"] for r in fleet_trend.tail(self.path, 2)], [3, 4])

    def test_missing_ledger_is_empty_not_error(self):
        self.assertEqual(fleet_trend.tail(str(Path(self.dir.name) / "nope.jsonl"), 5), [])

    def test_torn_line_is_tolerated(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.path).write_text(
            '{"ts":"a","usable":3}\n{ this is not json\n{"ts":"b","usable":1}\n',
            encoding="utf-8")
        rows = fleet_trend.tail(self.path, 24)
        self.assertEqual([r["usable"] for r in rows], [3, 1])


class RenderTest(unittest.TestCase):
    def test_no_history_renders_empty(self):
        self.assertEqual(fleet_trend.render_line([]), "")

    def test_single_tick_has_no_arrow_or_delta(self):
        line = fleet_trend.render_line([{"ts": "a", "usable": 2, "escalate": 0}])
        self.assertTrue(line.startswith("trend: "))
        self.assertIn("usable 2 ", line)   # bare value, no "→"
        self.assertNotIn("→", line)
        self.assertNotIn("over", line)

    def test_multi_tick_shows_arrow_spark_and_delta(self):
        rows = [
            {"ts": "a", "usable": 3, "live": 1, "sessions": 4, "escalate": 0},
            {"ts": "b", "usable": 2, "live": 1, "sessions": 4, "escalate": 0},
            {"ts": "c", "usable": 1, "live": 1, "sessions": 4, "escalate": 1},
        ]
        line = fleet_trend.render_line(rows)
        self.assertIn("usable 3→1", line)
        self.assertIn("(-2 over 3)", line)      # draining capacity
        self.assertIn("escalate 0→1", line)
        self.assertIn("(+1 over 3)", line)       # a new escalation appeared
        # a flat metric shows the arrow but no delta clause.
        self.assertIn("live 1→1", line)
        self.assertNotIn("live 1→1 ▁▁▁ (", line)


if __name__ == "__main__":
    unittest.main()
