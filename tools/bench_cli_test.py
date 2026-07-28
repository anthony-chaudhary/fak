#!/usr/bin/env python3
"""Tests for bench_cli.py — the benchmark catalog query CLI.

Every subcommand here is a lens onto the SAME committed catalog, and the risk is
not a crash but a quietly wrong lens: a filter that admits a run it should
exclude, a "best" that isn't the maximum, a listing that no longer leads with the
most recent run. Those all produce plausible output that misreports the tree, so
each filter and each ranking is pinned against a synthetic in-memory catalog.

``load_run``'s partial-artifact handling is pinned too: a run directory is often
incomplete (no kernel.json yet, a truncated manifest), and the loader must
degrade to "that piece is missing" rather than failing the whole lookup.

No catalog on disk is read; the two loader tests build a throwaway run tree in a
temp directory.
"""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_cli as M  # noqa: E402


def out_of(fn, *args):
    """Run a cmd_* function, returning (rc, captured stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        rc = fn(*args)
    return rc, buf.getvalue()


def entry(run_id, machine, model="smollm2-135m", precision="q8_0",
          timestamp="2026-01-01T00:00:00Z", peak=10.0, **extra):
    row = {"run_id": run_id, "machine_id": machine, "model": model,
           "precision": precision, "timestamp": timestamp,
           "peak_tok_per_sec": peak, "path": f"experiments/benchmark/{run_id}"}
    row.update(extra)
    return row


CATALOG = {"runs": [
    entry("r-laptop-old", "laptop", timestamp="2026-01-01T00:00:00Z", peak=10.0),
    entry("r-laptop-new", "laptop", timestamp="2026-03-01T00:00:00Z", peak=40.0),
    entry("r-m3pro", "m3pro", model="qwen25-7b", precision="q4_k_m",
          timestamp="2026-02-01T00:00:00Z", peak=25.0),
]}


def list_args(**over):
    ns = {"machine": None, "model": None, "precision": None,
          "since": None, "until": None, "format": "json"}
    ns.update(over)
    return argparse.Namespace(**ns)


def listed(**over):
    rc, text = out_of(M.cmd_list, list_args(**over), CATALOG)
    assert rc == 0, rc
    return [r["run_id"] for r in json.loads(text)]


class ListFilters(unittest.TestCase):
    def test_unfiltered_lists_every_run(self):
        self.assertEqual(set(listed()), {"r-laptop-old", "r-laptop-new", "r-m3pro"})

    def test_machine_filter_is_an_exact_match(self):
        self.assertEqual(set(listed(machine="laptop")), {"r-laptop-old", "r-laptop-new"})
        self.assertEqual(listed(machine="lap"), [])

    def test_model_filter_is_a_case_insensitive_substring(self):
        # documented as a substring filter, so a partial, differently-cased name hits
        self.assertEqual(listed(model="QWEN"), ["r-m3pro"])
        self.assertEqual(set(listed(model="smollm2")), {"r-laptop-old", "r-laptop-new"})

    def test_precision_filter_is_an_exact_match(self):
        self.assertEqual(listed(precision="q4_k_m"), ["r-m3pro"])
        self.assertEqual(listed(precision="q4"), [])

    def test_since_and_until_bound_the_window_inclusively(self):
        self.assertEqual(set(listed(since="2026-02-01T00:00:00Z")),
                         {"r-m3pro", "r-laptop-new"})
        self.assertEqual(set(listed(until="2026-02-01T00:00:00Z")),
                         {"r-laptop-old", "r-m3pro"})
        self.assertEqual(listed(since="2026-02-01T00:00:00Z",
                                until="2026-02-01T00:00:00Z"), ["r-m3pro"])

    def test_filters_compose_conjunctively(self):
        self.assertEqual(listed(machine="laptop", since="2026-02-01T00:00:00Z"),
                         ["r-laptop-new"])

    def test_table_output_leads_with_the_most_recent_run(self):
        rc, text = out_of(M.cmd_list, list_args(format="table"), CATALOG)
        self.assertEqual(rc, 0)
        ids = [line.split()[0] for line in text.splitlines()
               if line.startswith("r-")]
        self.assertEqual(ids, ["r-laptop-new", "r-m3pro", "r-laptop-old"])
        self.assertIn("3 run(s)", text)

    def test_empty_result_is_reported_as_success_not_an_error(self):
        # "no runs matched this filter" is an answer, not a failure
        rc, text = out_of(M.cmd_list, list_args(machine="ghost", format="table"), CATALOG)
        self.assertEqual(rc, 0)
        self.assertEqual(text, "")


class Best(unittest.TestCase):
    def args(self, **over):
        ns = {"model": None, "metric": "peak_tok_per_sec"}
        ns.update(over)
        return argparse.Namespace(**ns)

    def test_picks_the_maximum_over_the_whole_catalog(self):
        rc, text = out_of(M.cmd_best, self.args(), CATALOG)
        self.assertEqual(rc, 0)
        self.assertIn("r-laptop-new", text)

    def test_the_model_filter_narrows_which_maximum_wins(self):
        rc, text = out_of(M.cmd_best, self.args(model="qwen25-7b"), CATALOG)
        self.assertEqual(rc, 0)
        self.assertIn("r-m3pro", text)

    def test_a_run_missing_the_metric_never_wins(self):
        cat = {"runs": [entry("no-metric", "x", peak=None), entry("measured", "y", peak=1.0)]}
        rc, text = out_of(M.cmd_best, self.args(), cat)
        self.assertEqual(rc, 0)
        self.assertIn("measured", text)

    def test_an_unknown_metric_is_refused_rather_than_guessed(self):
        rc, _ = out_of(M.cmd_best, self.args(metric="vibes"), CATALOG)
        self.assertEqual(rc, 1)


class Table(unittest.TestCase):
    def args(self, **over):
        ns = {"model": None, "precision": None, "format": "markdown"}
        ns.update(over)
        return argparse.Namespace(**ns)

    def test_markdown_has_a_header_a_rule_and_one_row_per_run(self):
        rc, text = out_of(M.cmd_table, self.args(), CATALOG)
        self.assertEqual(rc, 0)
        rows = [ln for ln in text.splitlines() if ln.startswith("|")]
        self.assertEqual(len(rows), 2 + len(CATALOG["runs"]))  # header + rule + rows

    def test_model_filter_applies_before_rendering(self):
        rc, text = out_of(M.cmd_table, self.args(model="qwen25-7b"), CATALOG)
        self.assertEqual(rc, 0)
        self.assertIn("m3pro", text)
        self.assertNotIn("laptop", text)

    def test_json_format_emits_the_filtered_rows_verbatim(self):
        rc, text = out_of(M.cmd_table, self.args(precision="q4_k_m", format="json"), CATALOG)
        self.assertEqual(rc, 0)
        self.assertEqual([r["run_id"] for r in json.loads(text)], ["r-m3pro"])

    def test_a_run_with_no_rate_renders_as_zero_not_a_crash(self):
        cat = {"runs": [entry("bare", "x", peak=None)]}
        rc, text = out_of(M.cmd_table, self.args(), cat)
        self.assertEqual(rc, 0)
        self.assertIn("0.0", text)


class Summary(unittest.TestCase):
    def test_machine_groups_report_count_best_and_average(self):
        rc, text = out_of(M.cmd_summary, argparse.Namespace(group_by="machine"), CATALOG)
        self.assertEqual(rc, 0)
        self.assertIn("laptop:", text)
        self.assertIn("Runs: 2", text)
        self.assertIn("Best peak: 40.0 tok/s", text)   # max, not the last seen
        self.assertIn("Avg peak: 25.0 tok/s", text)    # (10 + 40) / 2

    def test_model_groups_are_counted_and_sorted(self):
        rc, text = out_of(M.cmd_summary, argparse.Namespace(group_by="model"), CATALOG)
        self.assertEqual(rc, 0)
        self.assertIn("smollm2-135m: 2 run(s)", text)
        self.assertIn("qwen25-7b: 1 run(s)", text)
        self.assertLess(text.index("qwen25-7b"), text.index("smollm2-135m"))

    def test_a_run_with_no_machine_id_is_bucketed_as_unknown(self):
        cat = {"runs": [{"run_id": "x", "peak_tok_per_sec": 1.0}]}
        rc, text = out_of(M.cmd_summary, argparse.Namespace(group_by="machine"), cat)
        self.assertEqual(rc, 0)
        self.assertIn("unknown:", text)


class LoadRun(unittest.TestCase):
    def tree(self, files):
        root = Path(tempfile.mkdtemp())
        run_dir = root / "runs" / "r1"
        run_dir.mkdir(parents=True)
        for name, body in files.items():
            (run_dir / name).write_text(body, encoding="utf-8")
        original = M.ROOT
        M.ROOT = root
        self.addCleanup(lambda: setattr(M, "ROOT", original))
        return {"runs": [{"run_id": "r1", "machine_id": "m", "path": "runs/r1"}]}

    def test_unknown_run_id_is_none(self):
        catalog = self.tree({})
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            self.assertIsNone(M.load_run("nope", catalog))
        self.assertIn("not found", buf.getvalue())

    def test_only_the_artifacts_that_exist_are_loaded(self):
        catalog = self.tree({"manifest.json": '{"config": {"workers": 4}}',
                             "batch.json": '{"peak": {"batch": 8}}'})
        run = M.load_run("r1", catalog)
        self.assertEqual(run["manifest"]["config"]["workers"], 4)
        self.assertEqual(set(run["results"]), {"batch"})   # no kernel/modelbench yet
        self.assertEqual(run["entry"]["run_id"], "r1")

    def test_a_corrupt_manifest_does_not_lose_the_valid_results(self):
        catalog = self.tree({"manifest.json": "{truncated",
                             "kernel.json": '{"ok": true}'})
        run = M.load_run("r1", catalog)
        self.assertIsNone(run["manifest"])
        self.assertEqual(run["results"]["kernel"], {"ok": True})


class CatalogLoader(unittest.TestCase):
    def test_a_missing_catalog_reports_the_repair_command(self):
        original = M.CATALOG_PATH
        M.CATALOG_PATH = Path(tempfile.mkdtemp()) / "catalog.json"
        self.addCleanup(lambda: setattr(M, "CATALOG_PATH", original))
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            self.assertIsNone(M.load_catalog())
        self.assertIn("bench_catalog.py build", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
