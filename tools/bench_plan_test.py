#!/usr/bin/env python3
"""Hermetic tests for tools/bench_plan.py.

The planner is a pure, deterministic fold over a catalog dict at an injected --now,
so these tests build SYNTHETIC catalogs (robust against the live shared catalog
growing under us) and assert the load-bearing guarantees the design panel pinned:
hardware feasibility (no CUDA on the mac), agent-host exclusion, coverage dominance,
the new-hardware frontier, canonical-emptiness of an uncategorised node, the single
recorded baseline, the four-intent guarantee, --now determinism + clamp, anti-mono
model diversity, the recheck-interval override, the plan-only honesty banner, and a
non-crashing empty catalog. One smoke test runs the REAL catalog through the fold.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "bench_plan.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("bench_plan", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load()
NOW = "20260622T000000Z"


def synth_catalog() -> dict:
    """A small catalog exercising every branch: a never-measured NVIDIA node, an
    Apple node (CUDA-infeasible), an AMD agent-host (excluded), a saturated node with
    ONE recorded baseline + a ran-but-unmeasured cell, and a thin node whose only runs
    carry non-canonical tags (the 'real runs but uncategorised' case)."""
    return {
        "last_updated": "2026-06-20T00:00:00Z",
        "machines": {
            "newnv": {"role": "bench-node", "gpu": "NVIDIA A100", "os": "Linux",
                      "arch": "x86_64", "runs": 0, "last_run": None},
            "mac": {"role": "bench-node", "gpu": "Apple M3 Pro", "os": "macOS",
                    "arch": "arm64", "runs": 5, "last_run": "20260619T000000Z"},
            "host": {"role": "agent-host", "gpu": "AMD Radeon RX 7600", "os": "Windows",
                     "arch": "x86_64", "runs": 0, "last_run": None},
            "busy": {"role": "bench-node", "gpu": "NVIDIA RTX 4070", "os": "Windows",
                     "arch": "x86_64", "runs": 40, "last_run": "20260620T000000Z"},
            "thin": {"role": "bench-node", "gpu": "NVIDIA L4", "os": "Linux",
                     "arch": "x86_64", "runs": 2, "last_run": "20260620T000000Z"},
        },
        "runs": [
            {"run_id": "r1", "machine_id": "busy", "model": "SmolLM2", "precision": "q8",
             "tags": ["model-benchmark"], "timestamp": "20260618T000000Z",
             "peak_tok_per_sec": 31.02, "speedup": None},
            {"run_id": "r2", "machine_id": "busy", "model": "qwen3.6-27b", "precision": "q8",
             "tags": ["qwen36"], "timestamp": "20260619T000000Z",
             "peak_tok_per_sec": None, "speedup": None},
            {"run_id": "r3", "machine_id": "mac", "model": "qwen3.6-27b", "precision": "q8",
             "tags": ["qwen36"], "timestamp": "20260619T000000Z",
             "peak_tok_per_sec": None, "speedup": None},
            # thin's only runs carry non-canonical tags -> occupy no canonical cell.
            {"run_id": "r4", "machine_id": "thin", "model": "qwen2.5-3b", "precision": "Q8_0",
             "tags": ["gcp", "gpu", "ada"], "timestamp": "20260620T000000Z",
             "peak_tok_per_sec": None, "speedup": None},
            {"run_id": "r5", "machine_id": "thin", "model": "qwen2.5-3b", "precision": "Q8_0",
             "tags": ["gcp", "headtohead"], "timestamp": "20260620T000000Z",
             "peak_tok_per_sec": None, "speedup": None},
        ],
    }


def plan(top: int = 1000, **kw):
    args = dict(now=MOD.parse_stamp(NOW), machine_filter=None, intent_filter="all",
                top=top, recheck_days=dict(MOD.RECHECK_DAYS),
                stale_horizon=MOD.DEFAULT_STALE_HORIZON_DAYS)
    args.update(kw)
    return MOD.build_plan(synth_catalog(), **args)


def entry(p, mid, kind):
    return next((e for e in p["ranked"] if e["machine_id"] == mid and e["workload_kind"] == kind), None)


class FeasibilityTest(unittest.TestCase):
    def test_no_cuda_on_the_mac(self):
        p = plan()
        self.assertIsNone(entry(p, "mac", "gpu-benchmark"),
                          "a CUDA gpu-benchmark must never be scheduled on the Apple mac")
        self.assertFalse(p["matrix"]["mac"]["gpu-benchmark"]["feasible"])
        self.assertIn("NVIDIA", p["matrix"]["mac"]["gpu-benchmark"]["note"])

    def test_cuda_feasible_on_nvidia(self):
        p = plan()
        self.assertTrue(p["matrix"]["busy"]["gpu-benchmark"]["feasible"])
        self.assertIsNotNone(entry(p, "newnv", "gpu-benchmark"))


class AgentHostExclusionTest(unittest.TestCase):
    def test_agent_host_never_a_target(self):
        p = plan()
        self.assertFalse(any(e["machine_id"] == "host" for e in p["ranked"]))
        self.assertEqual([x for x in p["excluded"] if x["machine_id"] == "host"],
                         [{"machine_id": "host", "why": "role=agent-host"}])


class CoverageDominanceTest(unittest.TestCase):
    def test_empty_cell_outranks_a_staler_nonempty_cell(self):
        p = plan()
        empty = entry(p, "busy", "session-benchmark")   # never run on busy
        full = entry(p, "busy", "qwen36")               # ran once, no baseline
        self.assertTrue(empty["is_empty"])
        self.assertFalse(full["is_empty"])
        self.assertLess(empty["rank"], full["rank"],
                        "an empty feasible cell must dominate a non-empty one")


class FrontierTest(unittest.TestCase):
    def test_new_machine_is_rank_one_and_learn_collect(self):
        p = plan()
        self.assertEqual(p["ranked"][0]["machine_id"], "newnv")
        nxt = p["per_machine_next"]["newnv"]
        self.assertTrue(nxt["is_empty"])
        self.assertEqual(nxt["intent"], "learn-collect")
        self.assertIsNone(nxt["age_days"])
        self.assertIn("0 runs", nxt["reason"])


class CanonicalEmptinessTest(unittest.TestCase):
    def test_uncategorised_runs_leave_canonical_cells_empty(self):
        p = plan()
        # thin has runs==2 but only non-canonical tags -> every canonical cell empty.
        for kind in ("model-benchmark", "gpu-benchmark", "qwen36"):
            self.assertEqual(entry(p, "thin", kind)["runs_in_cell"], 0,
                             f"thin/{kind} must read empty despite the node having 2 runs")


class SingleBaselineTest(unittest.TestCase):
    def test_exactly_one_cell_has_a_baseline(self):
        p = plan()
        based = [e for e in p["ranked"] if e["has_baseline"]]
        self.assertEqual(len(based), 1)
        self.assertEqual(based[0]["machine_id"], "busy")
        self.assertEqual(based[0]["workload_kind"], "model-benchmark")
        self.assertAlmostEqual(based[0]["baseline_tok_per_sec"], 31.02, places=2)
        self.assertEqual(based[0]["intent"], "regression")

    def test_cells_without_a_number_score_zero_drift(self):
        p = plan()
        self.assertEqual(entry(p, "busy", "qwen36")["dimension_scores"]["baseline_drift"], 0.0)


class FourIntentTest(unittest.TestCase):
    def test_every_intent_is_represented(self):
        p = plan()
        for it in ("benchmark", "learn-collect", "regression", "coverage"):
            self.assertTrue(p["by_intent"][it], f"intent {it} must surface at least one cell")

    def test_rendered_doc_has_all_four_sections(self):
        md = MOD.render_md(plan())
        for header in ("### Benchmark perf", "### Learn / collect new data",
                       "### Prevent regression", "### Fill coverage gaps"):
            self.assertIn(header, md)


class DeterminismTest(unittest.TestCase):
    def test_same_now_byte_identical(self):
        a, b = plan(), plan()
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        self.assertEqual(MOD.render_md(a), MOD.render_md(b))

    def test_later_now_increases_age(self):
        early = plan(now=MOD.parse_stamp("20260621T000000Z"))
        late = plan(now=MOD.parse_stamp("20260625T000000Z"))
        self.assertGreater(entry(late, "busy", "qwen36")["age_days"],
                           entry(early, "busy", "qwen36")["age_days"])

    def test_now_in_past_clamps_to_zero(self):
        p = plan(now=MOD.parse_stamp("20260101T000000Z"))  # before every run
        e = entry(p, "busy", "qwen36")
        self.assertEqual(e["age_days"], 0.0)
        self.assertEqual(e["dimension_scores"]["staleness_overdue"], 0.0)
        self.assertTrue(any("precedes" in n for n in p["notes"]))


class ModelDiversityTest(unittest.TestCase):
    def test_new_model_beats_monoculture_model(self):
        p = plan()
        # newnv is empty everywhere; model-benchmark proposes an unseen model (new pair),
        # qwen36 proposes qwen3.6-27b which is the catalog's most common model.
        new = entry(p, "newnv", "model-benchmark")["dimension_scores"]["model_diversity"]
        mono = entry(p, "newnv", "qwen36")["dimension_scores"]["model_diversity"]
        self.assertGreater(new, mono)


class RecheckOverrideTest(unittest.TestCase):
    def test_parse_override(self):
        d = MOD._parse_recheck_override("model-benchmark=3,qwen36=99")
        self.assertEqual(d["model-benchmark"], 3)
        self.assertEqual(d["qwen36"], 99)

    def test_shorter_interval_raises_staleness(self):
        base = dict(MOD.RECHECK_DAYS)
        short = MOD._parse_recheck_override("model-benchmark=3")
        default_s = entry(plan(recheck_days=base), "busy", "model-benchmark")["dimension_scores"]["staleness_overdue"]
        short_s = entry(plan(recheck_days=short), "busy", "model-benchmark")["dimension_scores"]["staleness_overdue"]
        self.assertLess(default_s, 0.5)   # age 4d / 7d interval -> below interval
        self.assertGreaterEqual(short_s, 0.5)  # age 4d / 3d interval -> overdue


class HonestyTest(unittest.TestCase):
    def test_plan_only_banner(self):
        p = plan()
        self.assertIn("no benchmark", p["honesty"].lower())
        md = MOD.render_md(p)
        self.assertIn("PLAN ONLY", md)
        self.assertIn("no benchmark", md.lower())

    def test_tool_does_not_shell_out(self):
        # the planner is a pure fold: it must not import subprocess (no git/no run).
        self.assertFalse(hasattr(MOD, "subprocess"))


class EmptyCatalogTest(unittest.TestCase):
    def test_missing_catalog_loads_none(self):
        self.assertIsNone(MOD.load_catalog(ROOT / "does" / "not" / "exist.json"))

    def test_empty_catalog_does_not_crash(self):
        p = MOD.build_plan({"machines": {}, "runs": []}, now=MOD.parse_stamp(NOW),
                           machine_filter=None, intent_filter="all", top=10,
                           recheck_days=dict(MOD.RECHECK_DAYS), stale_horizon=14)
        self.assertTrue(p["ok"])
        self.assertEqual(p["totals"]["total_runs"], 0)
        self.assertEqual(p["ranked"], [])
        # the doc still renders.
        self.assertIn("Hardware bench plan", MOD.render_md(p))


class StampTest(unittest.TestCase):
    def test_compact_and_iso(self):
        self.assertEqual(MOD.parse_stamp("20260622T140000Z").year, 2026)
        self.assertEqual(MOD.parse_stamp("2026-06-22T14:00:00Z").hour, 14)
        self.assertIsNone(MOD.parse_stamp("not-a-stamp"))
        self.assertIsNone(MOD.parse_stamp(None))

    def test_age_clamped_and_nullable(self):
        now = MOD.parse_stamp(NOW)
        self.assertIsNone(MOD.age_days(now, None))
        self.assertEqual(MOD.age_days(now, "20260625T000000Z"), 0.0)  # future -> clamp 0
        self.assertAlmostEqual(MOD.age_days(now, "20260620T000000Z"), 2.0, places=3)


class RealCatalogSmokeTest(unittest.TestCase):
    def test_real_catalog_folds_to_a_valid_plan(self):
        path = ROOT / "experiments" / "benchmark" / "catalog.json"
        cat = MOD.load_catalog(path)
        if cat is None:
            self.skipTest("real catalog not present")
        p = MOD.build_plan(cat, now=MOD.parse_stamp(NOW), machine_filter=None,
                           intent_filter="all", top=10, recheck_days=dict(MOD.RECHECK_DAYS),
                           stale_horizon=14)
        self.assertTrue(p["ok"])
        self.assertEqual(set(p["by_intent"]), set(MOD.INTENTS))
        self.assertGreater(p["totals"]["bench_nodes"], 0)
        self.assertIn("PLAN ONLY", MOD.render_md(p))


if __name__ == "__main__":
    unittest.main()
