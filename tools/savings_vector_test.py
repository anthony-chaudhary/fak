#!/usr/bin/env python3
"""Hermetic tests for savings_vector.py.

No model is ever run: every test drives build_vector over a synthetic turnbench
Report dict with the exact field shape internal/turnbench emits. The invariants
under test are the doc's load-bearing honesty claims:
  - the vector DECOMPOSES the flat Net (equal dollars), never INFLATES it;
  - the local-CPU axis Net omits is surfaced and labeled measured;
  - gpu_prefill / wall_clock stay honestly labeled modeled;
  - the happy-path control saves exactly zero on every axis (anti-inflation);
  - the binding account follows the host profile, never hard-coded to dollars.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import savings_vector as sv  # noqa: E402


def report(turns_saved=10, forced=6, elision=4, local_serve_ns=362):
    cm = {
        "prompt_tokens_per_turn": 1200,
        "completion_tokens_per_turn": 120,
        "dollars_per_mtok_in": 3.0,
        "dollars_per_mtok_out": 15.0,
        "model_turn_latency_ms": 1500.0,
    }
    # Mirror netFor exactly: turns x per-unit constant.
    tokens = turns_saved * (cm["prompt_tokens_per_turn"] + cm["completion_tokens_per_turn"])
    dollars = turns_saved * (cm["prompt_tokens_per_turn"] / 1e6 * cm["dollars_per_mtok_in"]
                             + cm["completion_tokens_per_turn"] / 1e6 * cm["dollars_per_mtok_out"])
    return {
        "cost_model": cm,
        "local_serve_ns": local_serve_ns,
        "net": {
            "turns_saved": turns_saved,
            "tokens_saved": tokens,
            "dollars_saved": dollars,
            "latency_saved_ms": turns_saved * cm["model_turn_latency_ms"],
        },
        "turn_kinds": {"forced": forced, "elision": elision},
    }


class DecomposeNotInflate(unittest.TestCase):
    def test_dollar_reprojection_equals_net(self):
        v = sv.build_vector(report(), profile="laptop")
        self.assertAlmostEqual(v.net_dollars_saved, v.vector_dollars_saved, places=9)

    def test_no_axis_is_negative(self):
        v = sv.build_vector(report(), profile="laptop")
        for a in v.axes:
            self.assertGreaterEqual(a["amount"], 0, a["account"])

    def test_forced_elision_preserved(self):
        v = sv.build_vector(report(forced=6, elision=4), profile="laptop")
        self.assertEqual(v.forced, 6)
        self.assertEqual(v.elision, 4)
        self.assertEqual(v.forced + v.elision, v.turns_saved)


class LocalCpuAxisSurfaced(unittest.TestCase):
    def test_local_cpu_present_and_measured(self):
        v = sv.build_vector(report(local_serve_ns=362), profile="laptop")
        cpu = next(a for a in v.axes if a["account"] == "local_cpu")
        self.assertEqual(cpu["provenance"], "measured")
        self.assertGreater(cpu["amount"], 0)

    def test_local_cpu_scales_with_turns_and_serve_ns(self):
        a = next(x for x in sv.build_vector(report(turns_saved=10), "laptop").axes
                 if x["account"] == "local_cpu")["amount"]
        b = next(x for x in sv.build_vector(report(turns_saved=20), "laptop").axes
                 if x["account"] == "local_cpu")["amount"]
        # exact pre-rounding; the amount is rounded to 4 places for display, so
        # compare on the relative scale rather than to display precision.
        self.assertAlmostEqual(b, 2 * a, places=2)

    def test_local_cpu_zero_when_no_serve_ns(self):
        v = sv.build_vector(report(local_serve_ns=0), profile="laptop")
        cpu = next(a for a in v.axes if a["account"] == "local_cpu")
        self.assertEqual(cpu["amount"], 0.0)


class HonestProvenance(unittest.TestCase):
    def test_prefill_and_wallclock_are_modeled(self):
        v = sv.build_vector(report(), profile="laptop")
        for acct in ("gpu_prefill", "wall_clock"):
            ax = next(a for a in v.axes if a["account"] == acct)
            self.assertEqual(ax["provenance"], "modeled", acct)

    def test_context_axis_promotes_to_measured_rate_with_pollution(self):
        modeled = sv.build_vector(report(), "laptop", pollution_rate=None)
        rated = sv.build_vector(report(), "laptop", pollution_rate=0.42)
        m = next(a for a in modeled.axes if a["account"] == "context_window")
        r = next(a for a in rated.axes if a["account"] == "context_window")
        self.assertEqual(m["provenance"], "modeled")
        self.assertEqual(r["provenance"], "measured-rate")


class AntiInflationControl(unittest.TestCase):
    def test_happy_control_saves_zero_everywhere(self):
        v = sv.build_vector(report(turns_saved=0, forced=0, elision=0), profile="laptop")
        for a in v.axes:
            self.assertEqual(a["amount"], 0, a["account"])
        self.assertEqual(v.binding_account, "none")


class BindingFollowsProfile(unittest.TestCase):
    def test_binding_varies_by_profile(self):
        r = report()
        self.assertEqual(sv.build_vector(r, "laptop").binding_account, "context_window")
        self.assertEqual(sv.build_vector(r, "ci-hook").binding_account, "local_cpu")
        self.assertEqual(sv.build_vector(r, "gpu-fleet").binding_account, "gpu_prefill")

    def test_binding_is_not_dollars(self):
        # The whole point: dollars is never the binding account, because it is not
        # a separate budget  -  it is the other axes priced.
        for prof in ("laptop", "ci-hook", "gpu-fleet"):
            self.assertNotIn(sv.build_vector(report(), prof).binding_account,
                             ("dollars", "dollar"))

    def test_unknown_profile_raises(self):
        with self.assertRaises(ValueError):
            sv.build_vector(report(), profile="mainframe")


class Selfcheck(unittest.TestCase):
    def test_selfcheck_passes(self):
        self.assertEqual(sv.cmd_selfcheck(None), 0)


if __name__ == "__main__":
    unittest.main()
