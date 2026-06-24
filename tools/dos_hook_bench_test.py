#!/usr/bin/env python3
"""Hermetic tests for tools/dos_hook_bench.py.

No real `dos` spawn and no wall-clock: the percentile/summary math is pure, the
event builder is checked for the per-hook stdin shape, the observation fold runs
over a synthetic jsonl, and the spawn seam is injected so timing is deterministic.
The dos-absent SKIP path is exercised by monkeypatching shutil.which.
"""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPT = Path(__file__).resolve().parent / "dos_hook_bench.py"


def load():
    spec = importlib.util.spec_from_file_location("dos_hook_bench", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()


class PercentileTest(unittest.TestCase):
    def setUp(self):
        self.samples = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    def test_nearest_rank_values(self):
        self.assertEqual(m.percentile(self.samples, 10), 10)
        self.assertEqual(m.percentile(self.samples, 50), 50)
        self.assertEqual(m.percentile(self.samples, 90), 90)

    def test_bounds(self):
        self.assertEqual(m.percentile(self.samples, 0), 10)
        self.assertEqual(m.percentile(self.samples, 100), 100)
        self.assertIsNone(m.percentile([], 50))

    def test_summarize_shape(self):
        s = m.summarize(self.samples)
        self.assertEqual(s["n"], 10)
        self.assertEqual(s["p50"], 50.0)
        self.assertEqual(s["min"], 10.0)
        self.assertEqual(s["max"], 100.0)
        self.assertEqual(s["mean"], 55.0)

    def test_summarize_empty(self):
        s = m.summarize([])
        self.assertEqual(s["n"], 0)
        self.assertIsNone(s["p50"])


class EventShapeTest(unittest.TestCase):
    def test_pre_event_has_no_tool_response(self):
        e = m.synthetic_event("Read", 3, kind="pre", workspace="C:/work/fak")
        self.assertEqual(e["hook_event_name"], "PreToolUse")
        self.assertNotIn("tool_response", e)
        self.assertEqual(e["tool_name"], "Read")
        self.assertIn("session_id", e)

    def test_post_event_has_tool_response(self):
        e = m.synthetic_event("Bash", 4, kind="post", workspace="C:/work/fak")
        self.assertEqual(e["hook_event_name"], "PostToolUse")
        self.assertIn("tool_response", e)

    def test_event_varies_by_index(self):
        a = m.synthetic_event("Read", 1, kind="pre", workspace="w")
        b = m.synthetic_event("Read", 2, kind="pre", workspace="w")
        self.assertNotEqual(a["tool_input"], b["tool_input"])


class ObservationStatsTest(unittest.TestCase):
    def test_passthrough_fraction_and_latency(self):
        with TemporaryDirectory() as d:
            obs = Path(d) / "observations.jsonl"
            # Mirrors the live .dos/metrics/observations.jsonl schema (keyed on "verb").
            obs.write_text("\n".join(json.dumps(r) for r in [
                {"verb": "pretool", "outcome": "passthrough", "latency_ms": 2.0},
                {"verb": "posttool", "outcome": "passthrough", "latency_ms": 4.0},
                {"verb": "pretool", "outcome": "warn", "latency_ms": 10.0},
                {"verb": "stop", "outcome": "passthrough"},  # not a per-call hook -> ignored
            ]), encoding="utf-8")
            stats = m.observation_stats(obs)
        self.assertEqual(stats["hook_observations"], 3)   # stop excluded
        self.assertEqual(stats["passthrough"], 2)
        self.assertAlmostEqual(stats["passthrough_fraction"], 2 / 3, places=3)
        self.assertEqual(stats["internal_latency_ms_p50"], 4.0)

    def test_missing_log_returns_empty(self):
        self.assertEqual(m.observation_stats(Path("nope/observations.jsonl")), {})


class TimingWithInjectedSpawnerTest(unittest.TestCase):
    def test_time_spawns_warms_then_times(self):
        calls: list[int] = []

        def fake_spawn(hook, event, ws):
            calls.append(1)
            return 100.0

        out = m.time_spawns("pretool", "Read", 5, Path("w"), spawner=fake_spawn)
        self.assertEqual(len(out), 5)            # 5 timed samples returned
        self.assertEqual(len(calls), 6)          # +1 warm spawn (not counted)
        self.assertTrue(all(v == 100.0 for v in out))

    def test_build_payload_sums_per_call_p50(self):
        p = m.build_payload(workspace="w", pre=[100.0] * 4, post=[200.0] * 4, obs={})
        self.assertEqual(p["pretool_ms"]["p50"], 100.0)
        self.assertEqual(p["posttool_ms"]["p50"], 200.0)
        self.assertEqual(p["per_tool_call_p50_ms"], 300.0)


class DosAbsentSkipTest(unittest.TestCase):
    def test_main_skips_clean_when_dos_absent(self):
        orig_which = m.shutil.which
        orig_collect = m.collect
        spawned: list[int] = []
        m.shutil.which = lambda _name: None
        m.collect = lambda *a, **k: spawned.append(1)  # must NOT be called
        try:
            rc = m.main(["--json"])
        finally:
            m.shutil.which = orig_which
            m.collect = orig_collect
        self.assertEqual(rc, 0)
        self.assertEqual(spawned, [])  # never spawned a process


if __name__ == "__main__":
    unittest.main()
