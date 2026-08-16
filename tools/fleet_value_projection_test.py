#!/usr/bin/env python3
"""Tests for fleet_value_projection.py — the executive fleet-scale value sweep.

This sweep is quoted at people, so the property that matters most is not the
arithmetic but the MEASURED/THEORY boundary it advertises. Two rules carry that:

* a ladder row that lands on a cell the 50x50 fleet grid actually measured uses
  that cell verbatim and labels itself ``measured_exact_cell``; every other row
  falls back to the through-origin fit and labels itself
  ``measured_supported_projection``. If the label ever detached from the branch
  that produced the number, an extrapolation would read as a measurement;
* the grid is keyed ``(turns, agents)`` — transposing that key silently swaps a
  measured cell for a projected one, so the key order is pinned explicitly.

Also pinned: turn deletion can never exceed the calls actually made, the coupon
law is a saturating curve bounded by ``(agents-1) * pool``, and the ``--levels``
override rejects a malformed ladder loudly instead of projecting from nonsense.

Every test drives synthetic inputs, so no checked-in experiment artifact needs to
exist and no chart is ever rendered (matplotlib is imported lazily inside plot()).
"""
import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fleet_value_projection as M  # noqa: E402


def inputs(**over):
    """A synthetic `load_inputs()` result with round numbers."""
    base = {
        "fleet_measured": {},
        "shared_saved_rate": 0.5,
        "shared_saved_r2": 0.999,
        "coupon_r2": 0.995,
        "pool": 8.0,
        "p_shared": 0.45,
        "per_turn_cost": 0.01,
        "per_turn_tokens": 1_000,
        "model_turn_latency_s": 2.0,
        "report": {"spawned_hook_baseline": {"p50_ns": 1_000_000},
                   "vdso_on": {"p50_ns": 0}},
    }
    base.update(over)
    return base


def level(agents=10, turns=90, context=100_000, name="L3 pod"):
    return {"level": name, "agents": agents, "turns_per_agent": turns,
            "context_tokens": context}


class Formatters(unittest.TestCase):
    def test_money_switches_unit_at_each_documented_threshold(self):
        self.assertEqual(M.money(2_500_000), "$2.50M")
        self.assertEqual(M.money(2_500), "$2.5k")
        self.assertEqual(M.money(999), "$999")
        self.assertEqual(M.money(9.5), "$9.50")
        self.assertEqual(M.money(0.125), "$0.125")

    def test_money_keeps_the_sign_of_a_negative_value(self):
        self.assertEqual(M.money(-2_500), "$-2.5k")

    def test_a_sub_cent_amount_collapses_to_zero_dollars(self):
        # Known rounding floor: below $0.01 the sweep prints $0.00 rather than
        # inventing precision it does not have.
        self.assertEqual(M.money(0.004), "$0.00")

    def test_human_tokens_switches_unit_at_each_documented_threshold(self):
        self.assertEqual(M.human_tokens(2_500_000_000), "2.50B")
        self.assertEqual(M.human_tokens(2_500_000), "2.5M")
        self.assertEqual(M.human_tokens(2_500), "2k")
        self.assertEqual(M.human_tokens(999), "999")

    def test_duration_switches_unit_at_each_documented_threshold(self):
        self.assertEqual(M.duration(7_200), "2.0h")
        self.assertEqual(M.duration(3_599), "60.0m")
        self.assertEqual(M.duration(90), "1.5m")
        self.assertEqual(M.duration(59.9), "59.9s")

    def test_percent_renders_a_whole_number_of_percent(self):
        self.assertEqual(M.percent(0.456), "46%")

    def test_a_dollar_sign_is_escaped_for_the_chart_text_renderer(self):
        # Unescaped `$` opens matplotlib mathtext and eats the rest of the label.
        self.assertEqual(M.chart_text("$1.5k saved"), r"\$1.5k saved")

    def test_wrap_bullets_marks_the_first_line_and_indents_the_rest(self):
        out = M.wrap_bullets(["alpha beta gamma delta"], width=11).split("\n")
        self.assertEqual(out[0], "- alpha beta")
        self.assertTrue(all(line.startswith("  ") for line in out[1:]))

    def test_an_empty_bullet_still_emits_a_bullet(self):
        self.assertEqual(M.wrap_bullets([""]), "-")


class R2Score(unittest.TestCase):
    def test_a_perfect_prediction_scores_one(self):
        self.assertEqual(M.r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)

    def test_predicting_the_mean_scores_zero(self):
        self.assertAlmostEqual(M.r2_score([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]), 0.0)

    def test_a_worse_than_mean_prediction_scores_negative(self):
        self.assertLess(M.r2_score([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]), 0.0)

    def test_a_constant_series_scores_one_only_if_it_was_predicted_exactly(self):
        # Zero variance: there is nothing to explain, so an exact hit is 1.0 and
        # anything else is 0.0 rather than a division by zero.
        self.assertEqual(M.r2_score([5.0, 5.0], [5.0, 5.0]), 1.0)
        self.assertEqual(M.r2_score([5.0, 5.0], [4.0, 5.0]), 0.0)


class CouponCrossUplift(unittest.TestCase):
    def test_a_solo_agent_has_no_cross_agent_uplift(self):
        self.assertEqual(M.coupon_cross_uplift(100, 1, 8, 0.45), 0.0)

    def test_a_degenerate_pool_or_share_rate_yields_no_uplift(self):
        self.assertEqual(M.coupon_cross_uplift(100, 10, 0, 0.45), 0.0)
        self.assertEqual(M.coupon_cross_uplift(100, 10, 8, 0.0), 0.0)

    def test_uplift_saturates_toward_the_whole_pool_per_extra_agent(self):
        # The coupon-collector law: more turns keep helping, with diminishing
        # returns, and can never exceed (agents-1) * pool.
        ceiling = (10 - 1) * 8
        first10 = M.coupon_cross_uplift(10, 10, 8, 0.45)
        first20 = M.coupon_cross_uplift(20, 10, 8, 0.45)
        long_ = M.coupon_cross_uplift(1_000, 10, 8, 0.45)
        self.assertLess(first10, first20)
        self.assertLess(first20, long_)
        # equal-width windows: the second 10 turns buy less than the first 10
        self.assertLess(first20 - first10, first10)
        self.assertLessEqual(long_, ceiling)
        self.assertAlmostEqual(long_, ceiling, places=3)

    def test_uplift_scales_linearly_in_the_number_of_extra_agents(self):
        one = M.coupon_cross_uplift(50, 2, 8, 0.45)
        many = M.coupon_cross_uplift(50, 11, 8, 0.45)
        self.assertAlmostEqual(many, one * 10)


class Project(unittest.TestCase):
    def setUp(self):
        self.original = M.appversion
        M.appversion = types.SimpleNamespace(app_version=lambda root: "0.0.0-test")
        self.addCleanup(lambda: setattr(M, "appversion", self.original))

    def test_a_row_outside_the_measured_grid_says_so_and_uses_the_fit(self):
        row = M.project(level(), inputs())
        self.assertEqual(row["turn_evidence"], "measured_supported_projection")
        self.assertEqual(row["calls"], 900)
        self.assertEqual(row["shared_saved_turns"], 450.0)   # 0.5 * 900

    def test_a_measured_cell_is_used_verbatim_and_labelled_measured(self):
        cell = {"shared_saved": 123.0, "cross_uplift": 45.0, "source": "measured"}
        row = M.project(level(), inputs(fleet_measured={(90, 10): cell}))
        self.assertEqual(row["turn_evidence"], "measured_exact_cell")
        self.assertEqual(row["shared_saved_turns"], 123.0)
        self.assertEqual(row["cross_agent_uplift_turns"], 45.0)

    def test_the_measured_grid_is_keyed_turns_then_agents_not_the_reverse(self):
        # Transposing the key would quietly demote a measured row to a projection.
        cell = {"shared_saved": 123.0, "cross_uplift": 45.0, "source": "measured"}
        row = M.project(level(agents=10, turns=90), inputs(fleet_measured={(10, 90): cell}))
        self.assertEqual(row["turn_evidence"], "measured_supported_projection")

    def test_turn_deletion_can_never_exceed_the_calls_that_were_made(self):
        row = M.project(level(), inputs(shared_saved_rate=3.0))
        self.assertEqual(row["shared_saved_turns"], row["calls"])
        self.assertEqual(row["shared_saved_pct_of_calls"], 1.0)

    def test_the_measured_and_theory_value_rows_are_reported_separately(self):
        row = M.project(level(), inputs())
        self.assertAlmostEqual(row["tool_dollars_saved"], 4.5)          # measured
        self.assertAlmostEqual(row["tool_tokens_saved"], 450_000)
        self.assertAlmostEqual(row["tool_agent_hours_saved"], 0.25)
        self.assertAlmostEqual(row["control_plane_seconds_saved_measured"], 0.9)
        # theory side: calls * miss_rate * context
        self.assertAlmostEqual(row["cold_context_tokens_avoided_theory"],
                               900 * M.LONG_CONTEXT["effective_kv_miss_rate"] * 100_000)

    def test_the_total_is_the_measured_row_plus_the_matching_theory_row(self):
        row = M.project(level(), inputs())
        self.assertAlmostEqual(row["total_api_equiv_value_per_run"],
                               row["tool_dollars_saved"]
                               + row["context_api_equiv_dollars_theory"])
        self.assertAlmostEqual(row["total_self_host_value_per_run"],
                               row["tool_dollars_saved"]
                               + row["context_gpu_dollars_avoided_theory"])

    def test_the_projected_wall_clock_is_floored_at_zero(self):
        # A huge saving must not project a negative run time.
        row = M.project(level(), inputs(shared_saved_rate=1.0, model_turn_latency_s=1e6))
        self.assertGreaterEqual(row["projected_parallel_wall_s"], 0.0)

    def test_the_projected_wall_clock_never_exceeds_the_baseline(self):
        row = M.project(level(), inputs())
        self.assertLessEqual(row["projected_parallel_wall_s"],
                             row["baseline_parallel_wall_s"])

    def test_an_empty_level_reports_zero_shares_instead_of_dividing_by_zero(self):
        row = M.project(level(agents=0, turns=0), inputs())
        self.assertEqual(row["calls"], 0)
        self.assertEqual(row["shared_saved_pct_of_calls"], 0)
        self.assertEqual(row["cross_uplift_pct_of_calls"], 0)
        self.assertEqual(row["avg_latency_saved_per_agent_s"], 0)

    def test_the_row_carries_the_app_version_it_was_generated_under(self):
        self.assertEqual(M.project(level(), inputs())["version"], "0.0.0-test")


class ReadLevels(unittest.TestCase):
    def write(self, body):
        p = Path(tempfile.mkdtemp()) / "levels.json"
        p.write_text(body, encoding="utf-8")
        return str(p)

    def test_no_path_uses_the_default_ladder(self):
        self.assertIs(M.read_levels(None), M.DEFAULT_LEVELS)
        self.assertIs(M.read_levels(""), M.DEFAULT_LEVELS)

    def test_a_well_formed_ladder_is_returned_as_given(self):
        rows = M.read_levels(self.write(json.dumps([level(name="only")])))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["level"], "only")

    def test_a_json_object_instead_of_an_array_is_refused(self):
        with self.assertRaises(SystemExit):
            M.read_levels(self.write('{"level": "L0"}'))

    def test_a_row_missing_a_required_key_names_the_row_and_the_key(self):
        bad = [level(), {"level": "L1", "agents": 2}]
        with self.assertRaises(SystemExit) as cm:
            M.read_levels(self.write(json.dumps(bad)))
        msg = str(cm.exception)
        self.assertIn("row 1", msg)
        self.assertIn("turns_per_agent", msg)


class DefaultLadder(unittest.TestCase):
    def test_the_ladder_spans_the_documented_endpoints(self):
        first, last = M.DEFAULT_LEVELS[0], M.DEFAULT_LEVELS[-1]
        self.assertEqual((first["agents"], first["turns_per_agent"],
                          first["context_tokens"]), (1, 25, 20_000))
        self.assertEqual((last["agents"], last["turns_per_agent"],
                          last["context_tokens"]), (100, 250, 500_000))

    def test_every_axis_of_the_ladder_ascends(self):
        for key in ("agents", "turns_per_agent", "context_tokens"):
            series = [row[key] for row in M.DEFAULT_LEVELS]
            self.assertEqual(series, sorted(series), key)

    def test_the_theory_assumptions_are_declared_with_their_source(self):
        # These are assumptions, not measurements; the provenance string is what
        # keeps them readable as such downstream.
        self.assertIn("source", M.LONG_CONTEXT)
        self.assertIn("source", M.HARDWARE)
        self.assertLess(0.0, M.LONG_CONTEXT["effective_kv_miss_rate"])
        self.assertLess(M.LONG_CONTEXT["effective_kv_miss_rate"], 1.0)


class Writers(unittest.TestCase):
    def setUp(self):
        self.original = M.appversion
        M.appversion = types.SimpleNamespace(app_version=lambda root: "0.0.0-test")
        self.addCleanup(lambda: setattr(M, "appversion", self.original))
        self.rows = [M.project(lv, inputs()) for lv in M.DEFAULT_LEVELS[:3]]
        self.out = Path(tempfile.mkdtemp())

    def test_the_csv_header_is_the_full_row_schema_and_every_row_lands(self):
        path = self.out / "sweep.csv"
        M.write_csv(self.rows, path)
        with path.open(newline="", encoding="utf-8") as f:
            got = list(csv.DictReader(f))
        self.assertEqual(len(got), 3)
        self.assertEqual(set(got[0]), set(self.rows[0]))
        self.assertEqual(got[0]["level"], self.rows[0]["level"])

    def test_the_json_payload_carries_the_schema_inputs_and_assumptions(self):
        path = self.out / "sweep.json"
        M.write_json_payload(inputs(), self.rows, path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], "fak.value-sweep.v1")
        self.assertEqual(payload["generated_by"], "tools/fleet_value_projection.py")
        self.assertEqual(len(payload["rows"]), 3)
        self.assertEqual(payload["scale_ladder"]["levels"], 3)
        self.assertIn("long_context", payload["theory_assumptions"])
        self.assertIn("hardware", payload["theory_assumptions"])
        # the measured inputs are named by FILE, so a reader can re-derive them
        self.assertIn("fleet_csv", payload["measured_inputs"])

    def test_the_json_payload_is_newline_terminated(self):
        path = self.out / "sweep.json"
        M.write_json_payload(inputs(), self.rows, path)
        self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))


if __name__ == "__main__":
    unittest.main()
