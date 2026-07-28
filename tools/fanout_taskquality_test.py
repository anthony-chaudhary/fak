#!/usr/bin/env python3
"""Tests for fanout_taskquality.py — the fan-out task-quality litmus.

The artifact this builds is explicitly SIMULATED, which makes two things
load-bearing and worth pinning:

* the HONESTY labelling — the artifact must keep announcing that it is a knobbed
  model and must keep pointing at the measured cost source it joins against. If
  the banner or the ``kind`` ever drops out, a simulated quality number starts
  reading as a measured one;
* determinism — the whole gate is "a fixed (profile, grid, trials, seed)
  reproduces byte-for-byte", so the same inputs must fold to the same bytes.

Underneath those, the trial mechanics encode real claims that a refactor could
silently invert, so each is driven with a degenerate profile that makes the
expected value exact rather than statistical: the fak arm CONTAINS an injected
sub-agent while the naive arm loses its work; the fold's finite capacity crowds
correct atoms out; a sub-agent that produces nothing still adds a decoy.

Pure in-memory; no artifact is written.
"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fanout_taskquality as M  # noqa: E402


def profile(**over):
    prof = {"name": "test", "atoms": 1, "p_fail": 0.0, "p_inject": 0.0,
            "verifier_recall": 1.0, "verifier_decoy_fa": 0.0,
            "fold_capacity": 8, "subturns": 1}
    prof.update(over)
    return prof


class AtomWeights(unittest.TestCase):
    def test_weights_are_a_normalised_decreasing_distribution(self):
        w = M._atom_weights(12)
        self.assertEqual(len(w), 12)
        self.assertAlmostEqual(sum(w), 1.0)
        # Zipf-ish: a few easy atoms, a long hard tail. A flat distribution would
        # remove the saturation knee the whole litmus is built to show.
        self.assertEqual(w, sorted(w, reverse=True))
        self.assertGreater(w[0], w[-1])

    def test_a_single_atom_takes_all_the_mass(self):
        self.assertEqual(M._atom_weights(1), [1.0])


class DrawAtom(unittest.TestCase):
    def test_a_degenerate_distribution_always_yields_its_only_atom(self):
        rng = random.Random(1)
        self.assertTrue(all(M._draw_atom(rng, [1.0]) == 0 for _ in range(50)))

    def test_every_draw_is_a_valid_atom_index(self):
        rng = random.Random(2)
        w = M._atom_weights(12)
        for _ in range(500):
            self.assertIn(M._draw_atom(rng, w), range(12))

    def test_the_heavier_atom_is_drawn_more_often(self):
        rng = random.Random(3)
        w = M._atom_weights(12)
        draws = [M._draw_atom(rng, w) for _ in range(2000)]
        self.assertGreater(draws.count(0), draws.count(11))


class FanoutTrial(unittest.TestCase):
    def run_trial(self, n, prof, arm, seed=0, weights=None):
        w = weights if weights is not None else M._atom_weights(prof["atoms"])
        return M._run_fanout_trial(random.Random(seed), n, prof, w, arm)

    def test_a_perfect_pool_covers_the_goal_and_reports_its_duplicate_work(self):
        # 5 sub-agents, one atom, nothing fails: full coverage and 4 of the 5
        # competent outputs re-derive an atom that is already covered.
        got = self.run_trial(5, profile(), "fak")
        self.assertEqual(got["coverage"], 1.0)
        self.assertEqual(got["realized"], 1.0)
        self.assertEqual(got["duplicate_work"], 0.8)
        self.assertEqual(got["failed_rate"], 0.0)

    def test_a_pool_that_always_fails_covers_nothing(self):
        got = self.run_trial(5, profile(p_fail=1.0), "fak")
        self.assertEqual(got["coverage"], 0.0)
        self.assertEqual(got["realized"], 0.0)
        self.assertEqual(got["failed_rate"], 1.0)
        self.assertEqual(got["duplicate_work"], 0.0)   # no competent outputs to divide by

    def test_the_fak_arm_contains_every_injection_and_keeps_the_work(self):
        got = self.run_trial(5, profile(p_inject=1.0), "fak")
        self.assertEqual(got["injected"], 5)
        self.assertEqual(got["injection_contained_rate"], 1.0)
        self.assertEqual(got["coverage"], 1.0)   # quarantined, so the atom survives

    def test_the_naive_arm_contains_nothing_and_loses_the_work(self):
        got = self.run_trial(5, profile(p_inject=1.0), "naive")
        self.assertEqual(got["injected"], 5)
        self.assertEqual(got["injection_contained_rate"], 0.0)
        self.assertEqual(got["coverage"], 0.0)   # poisoned: the atom is lost
        self.assertEqual(got["failed_rate"], 1.0)

    def test_with_no_injections_containment_is_vacuously_perfect_for_both_arms(self):
        for arm in ("fak", "naive"):
            got = self.run_trial(5, profile(p_inject=0.0), arm)
            self.assertEqual(got["injected"], 0)
            self.assertEqual(got["injection_contained_rate"], 1.0, arm)

    def test_the_fold_capacity_crowds_correct_atoms_out(self):
        # 12 atoms, uniform draw, 200 sub-agents: coverage is complete, but a fold
        # that can only surface 3 candidates realizes just 3/12 of the goal. This
        # is the imperfect-verifier inversion the litmus exists to show.
        prof = profile(atoms=12, fold_capacity=3)
        got = self.run_trial(200, prof, "fak", seed=7, weights=[1 / 12] * 12)
        self.assertEqual(got["coverage"], 1.0)
        self.assertEqual(got["realized"], 3 / 12)
        self.assertEqual(got["verifier_success"], 1.0)   # no false accepts at fa=0

    def test_a_false_accepting_verifier_reports_imperfect_precision(self):
        # Every sub-agent fails, so the fold sees only decoys; a verifier that
        # false-accepts them must report precision 0, never a vacuous 1.0.
        prof = profile(p_fail=1.0, verifier_decoy_fa=1.0, fold_capacity=3)
        got = self.run_trial(10, prof, "fak")
        self.assertEqual(got["verifier_success"], 0.0)
        self.assertEqual(got["realized"], 0.0)

    def test_an_empty_fold_reports_precision_one_rather_than_dividing_by_zero(self):
        prof = profile(p_fail=1.0, verifier_decoy_fa=0.0)
        got = self.run_trial(4, prof, "fak")
        self.assertEqual(got["verifier_success"], 1.0)

    def test_the_same_seed_reproduces_the_same_trial(self):
        prof = profile(atoms=12, p_fail=0.4, p_inject=0.2, subturns=4,
                       verifier_recall=0.85, verifier_decoy_fa=0.3)
        a = self.run_trial(16, prof, "fak", seed=99)
        b = self.run_trial(16, prof, "fak", seed=99)
        self.assertEqual(a, b)


class MatchedBudgetControl(unittest.TestCase):
    def test_a_self_context_agent_never_counts_a_rederived_atom_twice(self):
        # The control has its own context, so repeating an atom is free, not
        # duplicate work — that is what makes it a fair matched-budget arm.
        got = M._matched_budget_single_agent(random.Random(0), 20, profile(), [1.0])
        self.assertEqual(got["coverage"], 1.0)
        self.assertEqual(got["failed_rate"], 0.0)
        self.assertEqual(got["calls"], 20)

    def test_a_zero_budget_is_reported_as_zero_not_a_division_error(self):
        got = M._matched_budget_single_agent(random.Random(0), 0, profile(), [1.0])
        self.assertEqual(got["coverage"], 0.0)
        self.assertEqual(got["failed_rate"], 0.0)
        self.assertEqual(got["calls"], 0)

    def test_an_always_failing_agent_covers_nothing(self):
        got = M._matched_budget_single_agent(random.Random(0), 10,
                                             profile(p_fail=1.0), [1.0])
        self.assertEqual(got["coverage"], 0.0)
        self.assertEqual(got["failed_rate"], 1.0)


class Artifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.art = M.build(trials=2, seed=M.DEFAULT_SEED)

    def test_the_simulated_label_and_the_measured_cost_source_both_survive(self):
        self.assertEqual(self.art["kind"], "SIMULATED")
        self.assertIn("NOT a real-model run", self.art["banner"])
        self.assertEqual(self.art["cost_source"], "experiments/fanout/fanbench-research.csv")
        self.assertEqual(self.art["schema"], "fanout-taskquality-litmus/v1")

    def test_one_cell_per_grid_width_carrying_every_reported_metric(self):
        self.assertEqual([c["n"] for c in self.art["cells"]], M.N_GRID)
        required = {"coverage_at_n", "realized_at_n", "verifier_success",
                    "duplicate_work_rate", "failed_subagent_rate",
                    "coverage_naive_arm", "injection_contained_fak",
                    "injection_contained_naive", "matched_budget_single_coverage"}
        for cell in self.art["cells"]:
            self.assertTrue(required <= set(cell), f"n={cell['n']} is missing metrics")

    def test_rates_stay_inside_their_unit_interval(self):
        for cell in self.art["cells"]:
            for key in ("coverage_at_n", "realized_at_n", "verifier_success",
                        "duplicate_work_rate", "failed_subagent_rate",
                        "coverage_naive_arm"):
                self.assertGreaterEqual(cell[key], 0.0, f"{cell['n']}/{key}")
                self.assertLessEqual(cell[key], 1.0, f"{cell['n']}/{key}")

    def test_realized_never_exceeds_coverage(self):
        # You cannot realize an atom nobody produced; if this inverts, the fold
        # accounting has drifted from the production accounting.
        for cell in self.art["cells"]:
            self.assertLessEqual(cell["realized_at_n"], cell["coverage_at_n"],
                                 f"n={cell['n']}")

    def test_the_build_is_byte_reproducible_for_the_same_seed(self):
        again = M.build(trials=2, seed=M.DEFAULT_SEED)
        self.assertEqual(M._dumps(self.art), M._dumps(again))

    def test_a_different_seed_produces_a_different_artifact(self):
        other = M.build(trials=2, seed=M.DEFAULT_SEED + 1)
        self.assertNotEqual(M._dumps(self.art), M._dumps(other))

    def test_csv_header_and_rows_mirror_the_cells(self):
        text = M.to_csv(self.art)
        lines = text.strip().split("\n")
        self.assertEqual(len(lines), 1 + len(self.art["cells"]))
        self.assertTrue(lines[0].startswith("n,coverage_at_n,realized_at_n"))
        self.assertEqual(lines[1].split(",")[0], str(M.N_GRID[0]))
        self.assertTrue(text.endswith("\n"))

    def test_the_json_dump_is_newline_terminated(self):
        self.assertTrue(M._dumps(self.art).endswith("\n"))


class Grid(unittest.TestCase):
    def test_the_n_grid_is_ascending_and_starts_at_one(self):
        self.assertEqual(M.N_GRID, sorted(M.N_GRID))
        self.assertEqual(M.N_GRID[0], 1)   # the single-agent reference point

    def test_the_profile_probabilities_are_probabilities(self):
        for key in ("p_fail", "p_inject", "verifier_recall", "verifier_decoy_fa"):
            self.assertGreaterEqual(M.PROFILE[key], 0.0, key)
            self.assertLessEqual(M.PROFILE[key], 1.0, key)


if __name__ == "__main__":
    unittest.main()
