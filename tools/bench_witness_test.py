#!/usr/bin/env python3
"""Hermetic tests for tools/bench_witness.py.

The keep-bit is a go-benchmark number, so the only thing that must NOT run here
is ``go``. We replace run_bench on the module with synthetic measurements and
assert evaluate()'s KEEP / REVERT / NO_BENCH / NO_BASELINE branches, plus the
pure layers: lane_package mapping, the _BENCH_RE line parser, _primary_metric,
and the baseline load/save round-trip through a tmp file.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "bench_witness.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("bench_witness", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LanePackageTest(unittest.TestCase):
    def test_code_lane_with_test_files_maps_to_package(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            fak = Path(td)
            pkg = fak / "internal" / "adjudicator"
            pkg.mkdir(parents=True)
            (pkg / "decide_test.go").write_text("package adjudicator\n", encoding="utf-8")
            self.assertEqual(mod.lane_package("adjudicator", fak), "./internal/adjudicator")

    def test_doc_lane_without_package_dir_is_none(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(mod.lane_package("docs", Path(td)))

    def test_code_dir_without_test_files_is_none(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            fak = Path(td)
            (fak / "internal" / "empty").mkdir(parents=True)
            self.assertIsNone(mod.lane_package("empty", fak))


class BenchRegexTest(unittest.TestCase):
    def test_parses_a_standard_bench_line(self) -> None:
        mod = load()
        line = "BenchmarkDecide-32   4617806   267.5 ns/op   256 B/op   5 allocs/op"
        m = mod._BENCH_RE.match(line.strip())
        self.assertIsNotNone(m)
        self.assertEqual(m.group("name"), "BenchmarkDecide")
        self.assertEqual(float(m.group("ns")), 267.5)

    def test_parses_line_without_cpu_suffix(self) -> None:
        mod = load()
        m = mod._BENCH_RE.match("BenchmarkRoute   1000   42 ns/op")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("name"), "BenchmarkRoute")
        self.assertEqual(float(m.group("ns")), 42.0)

    def test_non_bench_lines_do_not_match(self) -> None:
        mod = load()
        self.assertIsNone(mod._BENCH_RE.match("PASS"))
        self.assertIsNone(mod._BENCH_RE.match("ok   ./internal/adjudicator   0.123s"))


class PrimaryMetricTest(unittest.TestCase):
    def test_prefers_filter_named_benchmark(self) -> None:
        mod = load()
        metrics = {"BenchmarkOther": 10.0, "BenchmarkDecide": 270.0}
        self.assertEqual(
            mod._primary_metric(metrics, "^BenchmarkDecide$"),
            {"name": "BenchmarkDecide", "ns_per_op": 270.0})

    def test_falls_back_to_fastest_when_no_filter_match(self) -> None:
        mod = load()
        metrics = {"BenchmarkA": 99.0, "BenchmarkB": 5.0}
        self.assertEqual(
            mod._primary_metric(metrics, "BenchmarkZZZ"),
            {"name": "BenchmarkB", "ns_per_op": 5.0})

    def test_empty_metrics_is_none(self) -> None:
        mod = load()
        self.assertIsNone(mod._primary_metric({}, "."))


class BaselineRoundTripTest(unittest.TestCase):
    def test_save_then_load_round_trips(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ".bench-baseline.json"
            key = "./internal/adjudicator::BenchmarkDecide"
            mod.save_baseline(path, key, 267.5, "BenchmarkDecide")
            self.assertEqual(mod.load_baseline(path, key), 267.5)
            doc = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(doc["schema"], "fleet-bench-baseline/1")
            self.assertEqual(doc["baselines"][key]["bench"], "BenchmarkDecide")

    def test_load_missing_file_or_key_is_none(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "absent.json"
            self.assertIsNone(mod.load_baseline(path, "k"))
            mod.save_baseline(path, "present", 1.0, "B")
            self.assertIsNone(mod.load_baseline(path, "missing"))


def fake_bench(primary_name: str, ns: float):
    """Build a run_bench replacement returning one synthetic primary metric."""
    def _run(package, fak_dir, *, count, bench_filter, timeout):
        return {
            "cmd": ["go", "test", package],
            "returncode": 0,
            "metrics": {primary_name: ns},
            "primary": {"name": primary_name, "ns_per_op": ns},
        }
    return _run


class EvaluateTest(unittest.TestCase):
    def test_keep_when_within_tolerance(self) -> None:
        mod = load()
        mod.run_bench = fake_bench("BenchmarkDecide", 280.0)
        # baseline 270, tolerance 25% -> ceiling 337.5; 280 <= ceiling -> KEEP.
        p = mod.evaluate(ROOT, lane="adjudicator", package="./internal/adjudicator",
                         smoke=False, count=1, tolerance_pct=25.0, baseline_ns=270.0,
                         baseline_file="fak/.bench-baseline.json", record=False, timeout=5)
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], mod.KEEP)
        self.assertEqual(p["measured_ns_per_op"], 280.0)
        self.assertEqual(p["baseline_ns_per_op"], 270.0)

    def test_revert_when_regressed_past_tolerance(self) -> None:
        mod = load()
        mod.run_bench = fake_bench("BenchmarkDecide", 400.0)
        # baseline 270, tolerance 25% -> ceiling 337.5; 400 > ceiling -> REVERT.
        p = mod.evaluate(ROOT, lane="adjudicator", package="./internal/adjudicator",
                         smoke=False, count=1, tolerance_pct=25.0, baseline_ns=270.0,
                         baseline_file="fak/.bench-baseline.json", record=False, timeout=5)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], mod.REVERT)
        self.assertGreater(p["regression_pct"], 25.0)

    def test_no_baseline_keeps_and_records_when_asked(self) -> None:
        mod = load()
        mod.run_bench = fake_bench("BenchmarkDecide", 300.0)
        with tempfile.TemporaryDirectory() as td:
            # evaluate joins root / baseline_file, so a basename lands inside td.
            bfile = Path(td) / ".bench-baseline.json"
            p = mod.evaluate(Path(td), lane="adjudicator", package="./internal/adjudicator",
                             smoke=False, count=1, tolerance_pct=25.0, baseline_ns=None,
                             baseline_file=".bench-baseline.json", record=True, timeout=5)
            self.assertTrue(p["ok"])
            self.assertEqual(p["verdict"], mod.KEEP)
            self.assertIsNone(p["baseline_ns_per_op"])
            self.assertTrue(p["baseline_recorded"])
            # The baseline file now carries the just-measured number.
            key = "./internal/adjudicator::."
            self.assertEqual(mod.load_baseline(bfile, key), 300.0)

    def test_no_baseline_without_record_does_not_write(self) -> None:
        mod = load()
        mod.run_bench = fake_bench("BenchmarkDecide", 300.0)
        with tempfile.TemporaryDirectory() as td:
            bfile = Path(td) / ".bench-baseline.json"
            p = mod.evaluate(Path(td), lane="adjudicator", package="./internal/adjudicator",
                             smoke=False, count=1, tolerance_pct=25.0, baseline_ns=None,
                             baseline_file="x.json", record=False, timeout=5)
            self.assertTrue(p["ok"])
            self.assertEqual(p["verdict"], mod.KEEP)
            self.assertFalse(p["baseline_recorded"])
            self.assertFalse(bfile.exists())

    def test_no_bench_for_doc_lane(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            # Empty workspace -> fak/internal/docs absent -> lane_package None -> NO_BENCH.
            p = mod.evaluate(Path(td), lane="docs", package=None, smoke=False, count=1,
                             tolerance_pct=25.0, baseline_ns=None,
                             baseline_file="fak/.bench-baseline.json", record=False, timeout=5)
            self.assertTrue(p["ok"])
            self.assertEqual(p["verdict"], mod.NO_BENCH)

    def test_error_when_bench_cannot_run(self) -> None:
        mod = load()
        mod.run_bench = lambda *a, **k: {"error": "go not found", "cmd": ["go"]}
        p = mod.evaluate(ROOT, lane="adjudicator", package="./internal/adjudicator",
                         smoke=False, count=1, tolerance_pct=25.0, baseline_ns=270.0,
                         baseline_file="fak/.bench-baseline.json", record=False, timeout=5)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], mod.ERROR)
        self.assertIn("go not found", p["reason"])

    def test_error_when_no_package_resolved(self) -> None:
        mod = load()
        p = mod.evaluate(ROOT, lane=None, package=None, smoke=False, count=1,
                         tolerance_pct=25.0, baseline_ns=None,
                         baseline_file="fak/.bench-baseline.json", record=False, timeout=5)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], mod.ERROR)


class RenderTest(unittest.TestCase):
    def test_render_does_not_raise(self) -> None:
        mod = load()
        mod.run_bench = fake_bench("BenchmarkDecide", 280.0)
        p = mod.evaluate(ROOT, lane="adjudicator", package="./internal/adjudicator",
                         smoke=False, count=1, tolerance_pct=25.0, baseline_ns=270.0,
                         baseline_file="fak/.bench-baseline.json", record=False, timeout=5)
        text = mod.render(p)
        self.assertIn("bench-witness", text)
        self.assertIn("KEEP", text)


if __name__ == "__main__":
    unittest.main()
