#!/usr/bin/env python3
"""Hermetic tests for tools/ctxbench.py (pure, no-spawn surface).

ctxbench's spawn/seat path needs a live roster, but its ladder builder and cost
math are pure and deterministic — that determinism is the whole point (a
reproducible needle-in-haystack oracle). These tests pin: ``_nonce`` is a stable
RNG-free token, ``build_ladder`` emits exactly ``turns`` steps with one
plant+checkpoint per needle and an ask turn ``depth`` later (clamped), the plant
nonce is recoverable at grade time, ``estimate_spend_tok`` matches its closed
form (arm-B triangular growth, arm-O1 budget*T), and ``cost_per_correct`` returns
None when nothing was answered (no divide-by-zero headline).
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "ctxbench.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("ctxbench", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load()


class TestNonce(unittest.TestCase):
    def test_deterministic_and_formatted(self):
        a = MOD._nonce(1, 0)
        self.assertEqual(a, MOD._nonce(1, 0))
        self.assertTrue(a.startswith("NONCE-"))
        self.assertEqual(len(a), len("NONCE-") + 5)

    def test_distinct_seeds_or_indices_differ(self):
        self.assertNotEqual(MOD._nonce(1, 0), MOD._nonce(1, 1))
        self.assertNotEqual(MOD._nonce(1, 0), MOD._nonce(2, 0))


class TestBuildLadder(unittest.TestCase):
    def test_step_count_equals_turns(self):
        lad = MOD.build_ladder(turns=30, depth=5, needles=4, seed=3)
        self.assertEqual(len(lad["steps"]), 30)
        self.assertEqual(set(s["turn"] for s in lad["steps"]), set(range(30)))

    def test_one_plant_per_needle_with_recoverable_nonce(self):
        lad = MOD.build_ladder(turns=40, depth=6, needles=3, seed=9)
        plants = [s for s in lad["steps"] if s["kind"] == "plant"]
        self.assertEqual(len(plants), 3)
        # every planted nonce is the map key and appears verbatim in the prompt text
        for s in plants:
            self.assertIn(s["plants"], lad["needle_map"])
            self.assertIn(s["plants"], s["text"])

    def test_ask_turn_is_depth_later_clamped(self):
        lad = MOD.build_ladder(turns=20, depth=100, needles=2, seed=1)
        for nonce, m in lad["needle_map"].items():
            # depth=100 > turns => ask clamped to the last turn
            self.assertEqual(m["ask_turn"], 19)
            self.assertLessEqual(m["plant_turn"], m["ask_turn"])

    def test_checkpoint_turns_are_ask_turns_minus_plant_collisions(self):
        # plant turns win the elif chain, so a checkpoint appears at every ask turn
        # that is not itself a plant turn.
        lad = MOD.build_ladder(turns=30, depth=4, needles=3, seed=2)
        ask_turns = {m["ask_turn"] for m in lad["needle_map"].values()}
        plant_turns = {m["plant_turn"] for m in lad["needle_map"].values()}
        checkpoints = {s["turn"] for s in lad["steps"] if s["kind"] == "checkpoint"}
        self.assertEqual(checkpoints, ask_turns - plant_turns)


class TestEstimateSpendTok(unittest.TestCase):
    def test_matches_closed_form(self):
        lad = MOD.build_ladder(turns=10, depth=3, needles=2, seed=1)
        est = MOD.estimate_spend_tok(lad, budget=8000)
        T = 10
        expect_b = 400 * T * (T + 1) // 2
        expect_o1 = 8000 * T
        self.assertEqual(est["arm_B_upper_tok"], expect_b)
        self.assertEqual(est["arm_O1_tok"], expect_o1)
        self.assertEqual(est["pair_upper_tok"], expect_b + expect_o1)


class TestCostPerCorrect(unittest.TestCase):
    def test_none_when_nothing_correct(self):
        self.assertIsNone(MOD.cost_per_correct({"n_correct": 0}))
        self.assertIsNone(MOD.cost_per_correct({}))


if __name__ == "__main__":
    unittest.main()
