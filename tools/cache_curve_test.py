#!/usr/bin/env python3
"""Tests for cache_curve.py — the frozen-trajectory cache-cliff demonstrator.

These pin the math so the curves can't silently drift: the frozen ceiling formula,
each decay axis's survival factor, the fan-out concurrency wall, and determinism of
the rendered chart (two runs at one commit must be byte-identical)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_curve as cc  # noqa: E402


class FrozenCeiling(unittest.TestCase):
    def test_floor_and_shape(self):
        self.assertEqual(cc.h_frozen(1), 0.0)
        # rises monotonically toward 1
        seq = [cc.h_frozen(t) for t in (2, 10, 50, 200, 1000)]
        self.assertEqual(seq, sorted(seq))
        self.assertLess(seq[-1], 1.0)

    def test_matches_real_anchor(self):
        # the real 205-turn session sits at 99% — the ceiling must reproduce it.
        self.assertAlmostEqual(cc.h_frozen(200), 0.990, places=3)
        self.assertAlmostEqual(cc.h_frozen(205), (205 - 1) / (205 + 1), places=6)
        self.assertGreaterEqual(round(cc.h_frozen(205) * 100), 99)  # rounds to the measured 99%


class Flexibility(unittest.TestCase):
    def test_edit_depth(self):
        self.assertEqual(cc.s_flex(0.0), 1.0)   # append-only = frozen = full survival
        self.assertEqual(cc.s_flex(1.0), 0.0)   # rewrite the head = total loss
        self.assertAlmostEqual(cc.s_flex(0.25), 0.75)
        self.assertEqual(cc.s_flex(2.0), 0.0)   # clamped, never negative


class ToolDensity(unittest.TestCase):
    def test_wall(self):
        # few calls: no wall
        self.assertEqual(cc.s_tooldensity(1, 1), 1.0)
        self.assertEqual(cc.s_tooldensity(5, 1), 1.0)
        # 20-block lookback, 1 breakpoint: budget 20 blocks; 10 calls = 22 blocks -> just over
        self.assertLess(cc.s_tooldensity(10, 1), 1.0)
        self.assertAlmostEqual(cc.s_tooldensity(10, 1), 20 / 22, places=6)
        # dense turn collapses
        self.assertLess(cc.s_tooldensity(40, 1), 0.3)

    def test_more_breakpoints_push_the_wall_out(self):
        for k in (20, 40, 80):
            self.assertGreater(cc.s_tooldensity(k, 4), cc.s_tooldensity(k, 1))
        # 4 breakpoints = 4x the block budget, so the wall sits ~4x further out:
        # a 40-call turn with 4 breakpoints survives as well as a ~10-call turn with 1.
        self.assertGreater(cc.s_tooldensity(40, 4), 0.9)
        self.assertLess(cc.s_tooldensity(40, 1), 0.3)

    def test_breakpoints_clamped(self):
        self.assertEqual(cc.s_tooldensity(40, 99), cc.s_tooldensity(40, cc.MAX_BREAKPOINTS))


class FanOut(unittest.TestCase):
    def test_concurrency_wall(self):
        # concurrent: shared setup paid N times, reused 0 times across the cohort
        for n in (1, 2, 10, 100):
            reuse, pay = cc.fanout_shared_reuse(n, concurrent=True)
            self.assertEqual(reuse, 0.0)
            self.assertEqual(pay, n)

    def test_shared_path_recovers(self):
        # staggered/cloned: paid once, reused N-1 times -> (N-1)/N
        reuse, pay = cc.fanout_shared_reuse(10, concurrent=False)
        self.assertEqual(pay, 1)
        self.assertAlmostEqual(reuse, 9 / 10)

    def test_short_agent_hit_default_below_shared(self):
        # many short agents on a big shared prefix: the shared path strictly wins,
        # and the default does not improve with N (no cross-agent reuse).
        h_def_2 = cc.h_fleet_shortagent(2, 100_000, 2, 2_000, concurrent=True)
        h_def_100 = cc.h_fleet_shortagent(100, 100_000, 2, 2_000, concurrent=True)
        self.assertAlmostEqual(h_def_2, h_def_100, places=6)  # flat in N
        h_sh_100 = cc.h_fleet_shortagent(100, 100_000, 2, 2_000, concurrent=False)
        self.assertGreater(h_sh_100, h_def_100)               # shared wins, widening with N


class CompoundScenario(unittest.TestCase):
    def test_only_two_single_agent_axes_multiply_no_magic_constant(self):
        c = cc.compound_scenario(turns=200)
        ceiling = cc.h_frozen(200)
        # every step's hit is exactly survival*ceiling, and survival is a product of ONLY
        # the two single-agent factors — no hidden constant can hide here.
        for s in c["steps"]:
            self.assertAlmostEqual(s["hit"], s["survival"] * ceiling, places=9)
        self.assertAlmostEqual(c["steps"][0]["survival"], 1.0)
        self.assertAlmostEqual(c["steps"][1]["survival"], cc.s_flex(0.25))
        self.assertAlmostEqual(c["steps"][2]["survival"], cc.s_flex(0.25) * cc.s_tooldensity(20, 1))
        self.assertAlmostEqual(c["steps"][3]["survival"], cc.s_flex(0.75) * cc.s_tooldensity(20, 1))

    def test_fan_out_is_a_separate_fleet_metric_not_a_single_agent_factor(self):
        c = cc.compound_scenario(turns=200, agents=100)
        # fan-out does NOT reduce the single-agent hit further — it is reported separately.
        self.assertAlmostEqual(c["fleet"]["single_agent_hit"], c["steps"][-1]["hit"])
        self.assertEqual(c["fleet"]["cross_agent_reuse_default"], 0.0)   # 0% and flat in N
        self.assertAlmostEqual(c["fleet"]["cross_agent_reuse_shared"], 99 / 100)

    def test_default_fleet_reuse_is_flat_in_N(self):
        # the % does not fall with N (the verifier's point); only the forfeited reuse grows.
        for n in (2, 10, 100, 1000):
            self.assertEqual(cc.compound_scenario(agents=n)["fleet"]["cross_agent_reuse_default"], 0.0)


class SurvivalIdentity(unittest.TestCase):
    def test_h_equals_s_times_ceiling(self):
        # the core claim: a lost reuse becomes recompute, so h(s) = s * h_frozen exactly.
        # rebuild it from first principles and check the closed form.
        T, d = 50, 1000.0
        R0 = d * T * (T - 1) / 2
        P0 = d * T
        ceiling = R0 / (R0 + P0)
        self.assertAlmostEqual(ceiling, cc.h_frozen(T), places=6)
        for s in (0.0, 0.25, 0.5, 1.0):
            read = s * R0
            paid = P0 + (1 - s) * R0
            h = read / (read + paid)
            self.assertAlmostEqual(h, s * ceiling, places=9)


class MeasuredValidation(unittest.TestCase):
    def test_validate_measured_fanout_fixture_reports_residuals(self):
        fixture = {
            "samples": [
                {
                    "name": "default fanout",
                    "axis": "fanout",
                    "agents": 10,
                    "concurrent": True,
                    "measured_reuse": 0.02,
                },
                {
                    "name": "shared fanout",
                    "axis": "fanout",
                    "agents": 10,
                    "concurrent": False,
                    "measured_reuse": 0.88,
                },
                {
                    "name": "quarter edit",
                    "axis": "flex",
                    "edit_depth": 0.25,
                    "measured_survival": 0.71,
                },
            ],
        }
        v = cc.validate_measurements(fixture, tolerance=0.05)
        self.assertEqual(v["schema"], "fak.cache_curve.validation.v1")
        self.assertEqual(v["summary"]["samples"], 3)
        self.assertEqual(v["summary"]["failures"], 0)

        rows = {r["name"]: r for r in v["rows"]}
        self.assertAlmostEqual(rows["default fanout"]["modeled_survival"], 0.0)
        self.assertAlmostEqual(rows["default fanout"]["residual"], 0.02)
        self.assertAlmostEqual(rows["shared fanout"]["modeled_survival"], 0.9)
        self.assertAlmostEqual(rows["shared fanout"]["residual"], -0.02)
        self.assertAlmostEqual(rows["quarter edit"]["modeled_survival"], 0.75)
        self.assertAlmostEqual(rows["quarter edit"]["residual"], -0.04)

        rendered = cc.render_validation(v)
        self.assertIn("measured decay validation", rendered)
        self.assertIn("residual", rendered)
        self.assertIn("default fanout", rendered)

    def test_validate_fails_when_residual_exceeds_tolerance(self):
        v = cc.validate_measurements({
            "samples": [{
                "axis": "fanout",
                "agents": 5,
                "concurrent": True,
                "measured_reuse": 0.50,
            }],
        }, tolerance=0.05)
        self.assertEqual(v["summary"]["status"], "FAIL")
        self.assertEqual(v["summary"]["failures"], 1)


class Determinism(unittest.TestCase):
    def test_chart_is_deterministic(self):
        self.assertEqual(cc.render_chart(200), cc.render_chart(200))
        self.assertEqual(cc.render_curves(cc.curve_table(123)),
                         cc.render_curves(cc.curve_table(123)))


if __name__ == "__main__":
    unittest.main()
