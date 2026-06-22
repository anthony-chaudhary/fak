#!/usr/bin/env python3
"""bench_catalog_test.py -- tests for the benchmark-catalog tool.

Witnesses the stale-``fak/``-prefix path fix and the non-destructive merge build:

  * the benchmark tree resolves to ``<repo>/experiments/benchmark`` (NOT a
    ``fak/`` subdir that never existed),
  * a rebuild in a driver/agent-host clone with no local run dirs PRESERVES the
    committed runs instead of wiping them, and normalizes their stale ``fak\\`` path,
  * ``validate`` treats a missing remote run dir as a warning, not a hard error,
  * the committed catalog is structurally valid, fully registered (no TBD stubs),
    and names exactly one agent-host.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import bench_catalog as bc  # noqa: E402


class TestPaths(unittest.TestCase):
    def test_benchmark_dir_is_repo_root_not_fak_prefixed(self):
        self.assertEqual(bc.BENCHMARK_DIR, ROOT / "experiments" / "benchmark")
        self.assertFalse(
            (ROOT / "fak" / "experiments").exists(),
            "the stale fak/experiments tree must not exist",
        )
        self.assertTrue(bc.CATALOG_PATH.exists(), "catalog.json must resolve on disk")

    def test_normalize_run_path_strips_stale_prefix(self):
        self.assertEqual(bc.normalize_run_path("fak/experiments/x"), "experiments/x")
        self.assertEqual(bc.normalize_run_path("fak\\experiments\\x"), "experiments\\x")
        self.assertEqual(bc.normalize_run_path("experiments/x"), "experiments/x")
        self.assertEqual(bc.normalize_run_path(""), "")


class TestCommittedCatalog(unittest.TestCase):
    def setUp(self):
        with open(bc.CATALOG_PATH, encoding="utf-8") as f:
            self.catalog = json.load(f)

    def test_no_stale_fak_paths_remain(self):
        for run in self.catalog["runs"]:
            p = str(run.get("path", "")).replace("\\", "/")
            self.assertFalse(p.startswith("fak/"), f"stale fak/ path: {p}")

    def test_structurally_valid(self):
        self.assertEqual(bc.validate_catalog(self.catalog), [])

    def test_no_tbd_stubs(self):
        for mid, m in self.catalog["machines"].items():
            self.assertNotIn(m.get("arch"), (None, "TBD", "?"), f"{mid} arch unset")
            self.assertNotIn(m.get("gpu"), (None, "TBD"), f"{mid} gpu unset")
            self.assertIn(m.get("role"), ("agent-host", "bench-node"), f"{mid} role unset")

    def test_exactly_one_agent_host(self):
        roles = [m.get("role") for m in self.catalog["machines"].values()]
        self.assertEqual(roles.count("agent-host"), 1,
                         "exactly one node is kept for agent use")


class TestNonDestructiveBuild(unittest.TestCase):
    def test_build_preserves_runs_with_no_local_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            bench = td / "experiments" / "benchmark"
            (bench / "machines").mkdir(parents=True)
            seed = {
                "$schema": "benchmark/catalog.v1", "version": "1.0", "last_updated": None,
                "machines": {"m1": {"id": "m1", "runs": 1, "last_run": "x"}},
                "runs": [{
                    "run_id": "r1", "machine_id": "m1", "timestamp": "20260101T000000Z",
                    "path": "fak\\experiments\\benchmark\\runs\\by-machine\\m1\\r1",
                }],
                "index": {"by_model": {}, "by_precision": {}, "by_date": {}},
            }
            (bench / "catalog.json").write_text(json.dumps(seed), encoding="utf-8")

            saved = (bc.ROOT, bc.BENCHMARK_DIR, bc.MACHINES_DIR, bc.RUNS_DIR, bc.CATALOG_PATH)
            try:
                bc.ROOT = td
                bc.BENCHMARK_DIR = bench
                bc.MACHINES_DIR = bench / "machines"
                bc.RUNS_DIR = bench / "runs" / "by-machine"
                bc.CATALOG_PATH = bench / "catalog.json"
                cat = bc.build_catalog()
                errors = bc.validate_catalog(cat)
                warnings = bc.missing_run_paths(cat)
            finally:
                (bc.ROOT, bc.BENCHMARK_DIR, bc.MACHINES_DIR, bc.RUNS_DIR, bc.CATALOG_PATH) = saved

            self.assertEqual(len(cat["runs"]), 1, "committed run must survive a no-local-dir rebuild")
            self.assertFalse(cat["runs"][0]["path"].replace("\\", "/").startswith("fak/"),
                             "stale fak/ prefix must be normalized on rebuild")
            self.assertEqual(errors, [], "structurally valid")
            self.assertEqual(len(warnings), 1, "the absent run dir is a warning, not an error")

    def test_pathless_runs_do_not_collapse(self):
        # Two distinct committed runs that lack a 'path' must each survive the
        # merge rather than collapsing onto the empty key and overwriting.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            bench = td / "experiments" / "benchmark"
            (bench / "machines").mkdir(parents=True)
            seed = {
                "$schema": "benchmark/catalog.v1", "version": "1.0", "last_updated": None,
                "machines": {"m1": {"id": "m1", "runs": 0, "last_run": None}},
                "runs": [
                    {"run_id": "r1", "machine_id": "m1", "timestamp": "20260101T000000Z"},
                    {"run_id": "r2", "machine_id": "m1", "timestamp": "20260102T000000Z"},
                ],
                "index": {"by_model": {}, "by_precision": {}, "by_date": {}},
            }
            (bench / "catalog.json").write_text(json.dumps(seed), encoding="utf-8")

            saved = (bc.ROOT, bc.BENCHMARK_DIR, bc.MACHINES_DIR, bc.RUNS_DIR, bc.CATALOG_PATH)
            try:
                bc.ROOT, bc.BENCHMARK_DIR = td, bench
                bc.MACHINES_DIR = bench / "machines"
                bc.RUNS_DIR = bench / "runs" / "by-machine"
                bc.CATALOG_PATH = bench / "catalog.json"
                cat = bc.build_catalog()
            finally:
                (bc.ROOT, bc.BENCHMARK_DIR, bc.MACHINES_DIR, bc.RUNS_DIR, bc.CATALOG_PATH) = saved

            self.assertEqual({r["run_id"] for r in cat["runs"]}, {"r1", "r2"},
                             "path-less runs must not collapse onto a shared empty key")


if __name__ == "__main__":
    unittest.main()
