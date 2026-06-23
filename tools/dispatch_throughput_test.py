#!/usr/bin/env python3
"""Hermetic tests for tools/dispatch_throughput.py.

The meter's I/O (gh closed-issue fetch, progress.jsonl read) is seamed out; the
windowing math and the ON_TRACK / WARMING_UP / BELOW_TARGET grader are pure and
take an injected ``now_ts`` (epoch seconds), so no subprocess or wall-clock runs.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "dispatch_throughput.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("dispatch_throughput", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# A fixed "now" so the tests never read the wall clock.
NOW = dt.datetime(2026, 6, 23, 12, 0, 0, tzinfo=dt.timezone.utc)
NOW_TS = NOW.timestamp()


def iso_ago(hours: float) -> str:
    return (NOW - dt.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


class WindowedClosesTest(unittest.TestCase):
    def test_buckets_and_rates(self) -> None:
        mod = load()
        closed = [
            {"number": 1, "closedAt": iso_ago(0.5), "stateReason": "COMPLETED"},
            {"number": 2, "closedAt": iso_ago(2.0), "stateReason": "COMPLETED"},
            {"number": 3, "closedAt": iso_ago(2.5), "stateReason": "NOT_PLANNED"},
            {"number": 4, "closedAt": iso_ago(20.0), "stateReason": "COMPLETED"},
        ]
        w = mod.windowed_closes(closed, now_ts=NOW_TS)
        pw = w["per_window"]
        # 1h window: only #1 (0.5h).
        self.assertEqual(pw["1h"]["closed"], 1)
        self.assertEqual(pw["1h"]["completed"], 1)
        self.assertEqual(pw["1h"]["rate_per_hour"], 1.0)
        # 3h window: #1,#2,#3 closed; only #1,#2 completed.
        self.assertEqual(pw["3h"]["closed"], 3)
        self.assertEqual(pw["3h"]["completed"], 2)
        self.assertEqual(pw["3h"]["completed_rate_per_hour"], round(2 / 3, 2))
        # 24h window: all 4 closed; 3 completed -> completed rate 3/24.
        self.assertEqual(pw["24h"]["closed"], 4)
        self.assertEqual(pw["24h"]["completed"], 3)
        self.assertEqual(pw["24h"]["completed_rate_per_hour"], round(3 / 24, 2))

    def test_undated_close_counted_not_dropped(self) -> None:
        mod = load()
        closed = [{"number": 9, "closedAt": None, "stateReason": "COMPLETED"}]
        w = mod.windowed_closes(closed, now_ts=NOW_TS)
        self.assertEqual(w["undated"], 1)
        self.assertEqual(w["per_window"]["24h"]["closed"], 0)


class WindowedLoopClosesTest(unittest.TestCase):
    def test_sums_closed_now_and_last_age(self) -> None:
        mod = load()
        recs = [
            {"utc": iso_ago(0.25), "closed_now": 2},
            {"utc": iso_ago(5.0), "closed_now": 3},
            {"utc": iso_ago(0.75), "closed_now": 0},   # no close -> ignored
        ]
        w = mod.windowed_loop_closes(recs, now_ts=NOW_TS)
        self.assertEqual(w["per_window"]["1h"]["loop_closed"], 2)
        self.assertEqual(w["per_window"]["6h"]["loop_closed"], 5)
        # last attributed close is the 0.25h record -> 15 min.
        self.assertEqual(w["last_loop_close_age_min"], 15.0)

    def test_no_records_yields_none_last(self) -> None:
        mod = load()
        w = mod.windowed_loop_closes([], now_ts=NOW_TS)
        self.assertIsNone(w["last_loop_close_age_min"])
        self.assertEqual(w["per_window"]["6h"]["loop_closed"], 0)


class GradeTest(unittest.TestCase):
    def _payload(self, mod, closed, recs, target=10.0, primary=6):
        return mod.build_payload(
            root=ROOT, closed_issues=closed, progress_records=recs,
            now_ts=NOW_TS, target_per_hour=target, primary_window=primary)

    def test_on_track_when_primary_meets_target(self) -> None:
        mod = load()
        # 60 completed closes spread across the last 6h -> 10/h on the 6h window.
        closed = [{"number": i, "closedAt": iso_ago((i % 6) + 0.1),
                   "stateReason": "COMPLETED"} for i in range(60)]
        recs = [{"utc": iso_ago(0.2), "closed_now": 5}]
        p = self._payload(mod, closed, recs)
        self.assertEqual(p["verdict"], "ON_TRACK")
        self.assertTrue(p["ok"])
        self.assertGreaterEqual(p["completed_rate_per_hour"], 10.0)

    def test_below_target_when_slow(self) -> None:
        mod = load()
        closed = [{"number": 1, "closedAt": iso_ago(1.0), "stateReason": "COMPLETED"}]
        p = self._payload(mod, closed, [])
        self.assertEqual(p["verdict"], "BELOW_TARGET")
        self.assertFalse(p["ok"])
        self.assertIn("short of", p["reason"])
        self.assertIn("NO attributed close", p["reason"])  # empty progress log

    def test_warming_up_when_recent_short_window_recovers(self) -> None:
        mod = load()
        # 6h window is starved (one old completed close) so it's below target, but
        # the last hour has 12 completed closes AND the loop closed 8 min ago.
        closed = [{"number": 0, "closedAt": iso_ago(5.5), "stateReason": "COMPLETED"}]
        closed += [{"number": 100 + i, "closedAt": iso_ago(0.1),
                    "stateReason": "COMPLETED"} for i in range(12)]
        recs = [{"utc": iso_ago(8 / 60.0), "closed_now": 4}]  # 8 min ago
        p = self._payload(mod, closed, recs)
        self.assertEqual(p["verdict"], "WARMING_UP")
        self.assertTrue(p["ok"])

    def test_fetch_error_is_audit_error(self) -> None:
        mod = load()
        p = mod.build_payload(root=ROOT, closed_issues=[{"_error": "gh down"}],
                              progress_records=[], now_ts=NOW_TS,
                              target_per_hour=10.0, primary_window=6)
        self.assertEqual(p["verdict"], "AUDIT_ERROR")
        self.assertFalse(p["ok"])


class RenderTest(unittest.TestCase):
    def test_render_and_md_do_not_raise(self) -> None:
        mod = load()
        closed = [{"number": 1, "closedAt": iso_ago(0.5), "stateReason": "COMPLETED"}]
        p = mod.build_payload(root=ROOT, closed_issues=closed, progress_records=[],
                              now_ts=NOW_TS, target_per_hour=10.0, primary_window=6)
        self.assertIn("dispatch throughput", mod.render(p))
        self.assertIn("Throughput (closed issues per hour)", mod.render_md_block(p))


if __name__ == "__main__":
    unittest.main()
