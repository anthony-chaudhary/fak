#!/usr/bin/env python3
"""Tests for bench_chart.py — the benchmark HTML chart generator.

The charts are a published artefact, so the thing worth pinning is the DATA
REDUCTION that happens before Plotly ever sees a number, not the markup:

* throughput is "the best run per machine, ranked" — picking the wrong run or
  losing the ranking republishes a different claim from the same catalog;
* a run whose batch sweep is missing must be DROPPED from the scaling chart, not
  drawn as a flat zero line, which would read as a real measurement of nothing;
* every chart returns False (and writes nothing) when it has no data, so an empty
  catalog fails loudly instead of shipping a blank page.

These drive the pure generators over synthetic run dicts into a temp directory:
no catalog on disk, no network, no browser.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_chart as M  # noqa: E402


def plot_args(html):
    """Pull the (data, layout) Plotly was called with back out of the page."""
    marker = 'Plotly.newPlot("chart", '
    i = html.index(marker) + len(marker)
    dec = json.JSONDecoder()
    data, end = dec.raw_decode(html, i)
    layout, _ = dec.raw_decode(html, end + 2)
    return data, layout


def run(run_id, machine, peak, baseline=1.0):
    return {"run_id": run_id, "machine_id": machine,
            "peak_tok_per_sec": peak, "baseline_tok_per_sec": baseline}


class Html(unittest.TestCase):
    def test_header_and_footer_form_a_closed_document(self):
        page = M.html_header("Some Title") + M.html_footer()
        self.assertTrue(page.startswith("<!DOCTYPE html>"))
        self.assertIn("<title>Some Title</title>", page)
        self.assertIn("plotly", page)
        self.assertTrue(page.rstrip().endswith("</html>"))


class Throughput(unittest.TestCase):
    def out(self):
        return Path(tempfile.mkdtemp()) / "charts" / "throughput.html"

    def test_no_runs_writes_nothing_and_reports_failure(self):
        out = self.out()
        self.assertFalse(M.chart_throughput([], out))
        self.assertFalse(out.exists())

    def test_one_bar_per_machine_using_that_machine_best_run(self):
        out = self.out()
        runs = [run("r1", "laptop", 10.0), run("r2", "laptop", 30.0),
                run("r3", "m3pro", 20.0)]
        self.assertTrue(M.chart_throughput(runs, out))
        data, layout = plot_args(out.read_text(encoding="utf-8"))
        peak = next(s for s in data if s["name"] == "Peak Throughput")
        # laptop's 10.0 run must be dropped in favour of its 30.0 run
        self.assertEqual(peak["y"], [30.0, 20.0])
        # ...and machines are ranked fastest-first
        self.assertEqual(peak["x"], ["laptop", "m3pro"])
        self.assertEqual(layout["barmode"], "group")

    def test_baseline_series_stays_aligned_with_its_machine(self):
        out = self.out()
        runs = [run("r1", "slow", 5.0, baseline=1.0), run("r2", "fast", 50.0, baseline=9.0)]
        self.assertTrue(M.chart_throughput(runs, out))
        data, _ = plot_args(out.read_text(encoding="utf-8"))
        peak = next(s for s in data if s["name"] == "Peak Throughput")
        base = next(s for s in data if s["name"] == "Baseline (B=1)")
        self.assertEqual(peak["x"], base["x"])
        self.assertEqual(list(zip(base["x"], base["y"])), [("fast", 9.0), ("slow", 1.0)])

    def test_a_run_missing_its_rate_sorts_last_instead_of_crashing(self):
        out = self.out()
        runs = [{"run_id": "r1", "machine_id": "unmeasured"}, run("r2", "good", 7.0)]
        self.assertTrue(M.chart_throughput(runs, out))
        data, _ = plot_args(out.read_text(encoding="utf-8"))
        peak = next(s for s in data if s["name"] == "Peak Throughput")
        self.assertEqual(peak["x"][0], "good")

    def test_output_directory_is_created_on_demand(self):
        out = self.out()
        self.assertFalse(out.parent.exists())
        self.assertTrue(M.chart_throughput([run("r1", "a", 1.0)], out))
        self.assertTrue(out.exists())


class Scaling(unittest.TestCase):
    def out(self):
        return Path(tempfile.mkdtemp()) / "scaling.html"

    def with_batches(self, batches):
        """Swap the catalog/disk lookup for an in-memory {run_id: batch} map."""
        original = M.load_batch_results
        M.load_batch_results = lambda run_id, catalog: batches.get(run_id)
        self.addCleanup(lambda: setattr(M, "load_batch_results", original))

    def test_no_runs_reports_failure(self):
        out = self.out()
        self.assertFalse(M.chart_scaling([], {}, out))
        self.assertFalse(out.exists())

    def test_runs_without_batch_data_are_dropped_not_drawn_as_zero(self):
        out = self.out()
        self.with_batches({"has": {"points": [{"batch": 1, "agg_tok_per_sec": 10.0},
                                              {"batch": 4, "agg_tok_per_sec": 30.0}]},
                           "empty": {"points": []}})
        runs = [{"run_id": "has", "machine_id": "laptop"},
                {"run_id": "empty", "machine_id": "ghost"},
                {"run_id": "absent", "machine_id": "nofile"}]
        self.assertTrue(M.chart_scaling(runs, {}, out))
        traces, layout = plot_args(out.read_text(encoding="utf-8"))
        self.assertEqual([t["name"] for t in traces], ["laptop"])
        self.assertEqual(traces[0]["x"], [1, 4])
        self.assertEqual(traces[0]["y"], [10.0, 30.0])
        # batch size is a doubling sweep, so the axis must stay logarithmic
        self.assertEqual(layout["xaxis"]["type"], "log")

    def test_all_runs_lacking_batch_data_fails_rather_than_drawing_an_empty_chart(self):
        out = self.out()
        self.with_batches({})
        self.assertFalse(M.chart_scaling([{"run_id": "a", "machine_id": "x"}], {}, out))
        self.assertFalse(out.exists())

    def test_each_series_gets_its_own_colour_cycling_when_they_run_out(self):
        out = self.out()
        names = [f"m{i}" for i in range(7)]
        self.with_batches({n: {"points": [{"batch": 1, "agg_tok_per_sec": 1.0}]} for n in names})
        runs = [{"run_id": n, "machine_id": n} for n in names]
        self.assertTrue(M.chart_scaling(runs, {}, out))
        traces, _ = plot_args(out.read_text(encoding="utf-8"))
        colours = [t["line"]["color"] for t in traces]
        self.assertEqual(len(set(colours[:5])), 5)   # first five all distinct
        self.assertEqual(colours[5], colours[0])     # then the palette wraps


class PrefillDecode(unittest.TestCase):
    def test_unimplemented_chart_reports_failure_instead_of_a_silent_success(self):
        out = Path(tempfile.mkdtemp()) / "pd.html"
        self.assertFalse(M.chart_prefill_decode([{"run_id": "a"}], out))
        self.assertFalse(M.chart_prefill_decode([], out))
        self.assertFalse(out.exists())


class Catalog(unittest.TestCase):
    def test_a_missing_catalog_is_none_not_an_exception(self):
        original = M.CATALOG_PATH
        M.CATALOG_PATH = Path(tempfile.mkdtemp()) / "nope.json"
        self.addCleanup(lambda: setattr(M, "CATALOG_PATH", original))
        self.assertIsNone(M.load_catalog())

    def test_a_corrupt_catalog_is_none_not_an_exception(self):
        p = Path(tempfile.mkdtemp()) / "catalog.json"
        p.write_text("{not json", encoding="utf-8")
        original = M.CATALOG_PATH
        M.CATALOG_PATH = p
        self.addCleanup(lambda: setattr(M, "CATALOG_PATH", original))
        self.assertIsNone(M.load_catalog())


if __name__ == "__main__":
    unittest.main()
