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


class DefaultPathTest(unittest.TestCase):
    def test_runtime_default_is_gitignored_state(self):
        self.assertEqual(
            Path(fleet_trend.DEFAULT_LEDGER).parts,
            (".fak", "nightrun", "fleet-status-history.jsonl"),
        )


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


def _rows(*live: float) -> list[dict]:
    return [{"ts": f"t{i}", "live": v} for i, v in enumerate(live)]


class NetDeclineAlarmTest(unittest.TestCase):
    """#4591 part 2: `live` falling STRICTLY for NET_DECLINE_ALARM_STREAK
    consecutive ledger appends is the fleet-drain signature and goes red; any
    flat/rising tail, short series, or missing key stays green."""

    def test_three_consecutive_declines_are_red(self):
        out = fleet_trend.net_decline_alarm(_rows(5, 4, 3, 2))
        self.assertTrue(out["red"])
        self.assertEqual(out["declines"], 3)
        self.assertIn("5→4→3→2", out["reason"])
        self.assertIn("net worker decline", out["reason"])

    def test_a_flat_tick_breaks_the_run(self):
        # 3→3 is not a decline: only the trailing 3→2 counts (1 < streak).
        out = fleet_trend.net_decline_alarm(_rows(4, 3, 3, 2))
        self.assertFalse(out["red"])
        self.assertEqual(out["declines"], 1)
        self.assertEqual(out["reason"], "")

    def test_a_recovery_tick_breaks_the_run(self):
        out = fleet_trend.net_decline_alarm(_rows(5, 4, 3, 4, 3, 2))
        self.assertFalse(out["red"])
        self.assertEqual(out["declines"], 2)

    def test_rising_or_flat_series_is_green(self):
        self.assertFalse(fleet_trend.net_decline_alarm(_rows(1, 2, 3))["red"])
        self.assertFalse(fleet_trend.net_decline_alarm(_rows(2, 2, 2, 2))["red"])

    def test_short_or_empty_or_keyless_series_is_green(self):
        self.assertFalse(fleet_trend.net_decline_alarm([])["red"])
        self.assertFalse(fleet_trend.net_decline_alarm(_rows(3, 2))["red"])
        self.assertFalse(fleet_trend.net_decline_alarm(
            [{"ts": "t0", "usable": 3}, {"ts": "t1", "usable": 2}])["red"])

    def test_custom_streak_and_key(self):
        rows = [{"ts": "t0", "usable": 3}, {"ts": "t1", "usable": 2},
                {"ts": "t2", "usable": 1}]
        out = fleet_trend.net_decline_alarm(rows, key="usable", streak=2)
        self.assertTrue(out["red"])
        self.assertEqual(out["declines"], 2)

    def test_rows_missing_the_key_are_skipped_not_zeroed(self):
        # A counter-free append (e.g. an old-schema row) must not fabricate a
        # 0 that manufactures a decline.
        rows = _rows(3, 3) + [{"ts": "tx"}] + _rows(2)
        out = fleet_trend.net_decline_alarm(rows)
        self.assertEqual(out["declines"], 1)
        self.assertFalse(out["red"])


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

    def test_throughput_counters_extracted_and_witness_folded(self):
        snap = {
            "sessions": {"total": 5, "by_category": {"LIVE": 2}},
            "accounts": {"usable": 1},
            "system": {"escalate": 0},
            "throughput": {"lands": 4170, "resumes": 15, "lands_witness": "git"},
        }
        m = fleet_trend.metrics_of(snap)
        self.assertEqual(m["lands"], 4170.0)
        self.assertEqual(m["resumes"], 15.0)
        self.assertEqual(m["lands_witnessed"], 1.0)
        # deaths was not produced → honestly absent, never a fabricated 0.
        self.assertNotIn("deaths", m)

    def test_no_throughput_block_leaves_counters_unset(self):
        # A gauges-only snapshot must not sprout a fabricated zero counter.
        m = fleet_trend.metrics_of({"accounts": {"usable": 1}})
        for k in ("lands", "resumes", "deaths", "lands_witnessed"):
            self.assertNotIn(k, m)


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

    def test_backend_scoped_rows_do_not_fabricate_aggregate_decline(self):
        fleet_trend.append(self.path, {"live": 20}, "2026-08-14T13:38:00Z")
        row = fleet_trend.append(
            self.path,
            {"scope": "backend", "backend": "codex", "backend_live": 0},
            "2026-08-14T13:47:19Z",
        )
        self.assertEqual(row["scope"], "backend")
        self.assertEqual(row["backend"], "codex")
        self.assertEqual(row["backend_live"], 0.0)
        self.assertNotIn("live", row)
        alarm = fleet_trend.net_decline_alarm(fleet_trend.tail(self.path, 10))
        self.assertFalse(alarm["red"])
        self.assertEqual(alarm["n"], 1)

    def test_append_persists_throughput_counters(self):
        # Counters present in metrics land in the row (so the Go goodput/HEAD-stall
        # reader picks them straight off the ledger); counters absent stay absent.
        fleet_trend.append(
            self.path,
            {"usable": 3, "live": 1, "sessions": 4, "escalate": 0,
             "lands": 4170, "resumes": 15, "lands_witnessed": 1},
            "2026-07-01T00:00:00Z",
        )
        fleet_trend.append(
            self.path,
            {"usable": 3, "live": 1, "sessions": 4, "escalate": 0},  # gauges only
            "2026-07-01T01:00:00Z",
        )
        rows = fleet_trend.tail(self.path, 24)
        self.assertEqual(rows[0]["lands"], 4170)
        self.assertEqual(rows[0]["resumes"], 15)
        self.assertEqual(rows[0]["lands_witnessed"], 1)
        self.assertNotIn("lands", rows[1])
        self.assertNotIn("deaths", rows[0])

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

    def test_session_total_series_is_labelled_not_bare(self):
        # #4651 DC#3: the session TOTAL series is a historical count (live + resumable +
        # stuck + terminal-history) and must be labelled explicitly, never rendered as the
        # bare word "sessions" next to the live series where it reads as live fleet scale.
        rows = [
            {"ts": "a", "usable": 3, "live": 1, "sessions": 40, "escalate": 0},
            {"ts": "b", "usable": 2, "live": 1, "sessions": 42, "escalate": 0},
        ]
        line = fleet_trend.render_line(rows)
        self.assertIn("all-sessions 40→42", line)   # explicit historical-total label
        # the bare-word regression would render "· sessions 40→42"; "all-sessions" has a
        # hyphen before "sessions", so a space-delimited " sessions " can only appear if
        # the label reverted to the unqualified word.
        self.assertNotIn(" sessions ", line)
        self.assertIn("live 1→1", line)               # the actionable series stays distinct


if __name__ == "__main__":
    unittest.main()
