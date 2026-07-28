#!/usr/bin/env python3
"""Tests for fleet_10min_projection.py — the 5x200 fleet wall-clock projector.

This script converts EXACT per-arm token counts into a projected wall-clock, and
publishes a "fak fused is Nx the warm-KV baseline" multiple. Two classes of
regression would quietly corrupt that claim, so both are pinned here:

* the token arithmetic — arm C must prefill the shared prefix exactly ONCE while
  arms A and B pay it per agent. If that ever drifts, every projected minute and
  every published multiple drifts with it, silently;
* the anti-over-claim fences — a rate card that carries a directly MEASURED
  batched-decode aggregate must use it in preference to the optimistic
  single-stream/batch_eff division, and a measured batch efficiency must be
  CAPPED at the agent count so small-model call-overhead amortisation cannot
  inflate the projection above the bandwidth-bound ideal.

Pure arithmetic over in-memory inputs plus one temp-file JSON read: no network,
no model, no GPU.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fleet_10min_projection as M  # noqa: E402


class PrefillTokens(unittest.TestCase):
    def test_headline_regime_matches_the_closed_forms(self):
        # 5 agents x 200 turns, P=2048 D=20 R=40 — the regime the doc headline quotes.
        a, b, c = M.prefill_tokens(2048, 200, 5, 20, 40)
        self.assertEqual(a, 5 * sum(2048 + t * 60 for t in range(200)))
        self.assertEqual(b, 5 * (2048 + 199 * 40))
        self.assertEqual(c, 2048 + 5 * 199 * 40)

    def test_arm_c_pays_the_shared_prefix_exactly_once(self):
        # The whole claim of arm C is prefix-once. Subtracting the incremental
        # per-turn work must leave exactly ONE prefix, not C of them.
        P, T, C, D, R = 1024, 40, 7, 16, 32
        _, b, c = M.prefill_tokens(P, T, C, D, R)
        self.assertEqual(c - C * (T - 1) * R, P)
        self.assertEqual(b - C * (T - 1) * R, C * P)

    def test_arms_are_strictly_ordered_naive_worst_fused_best(self):
        a, b, c = M.prefill_tokens(2048, 200, 5, 20, 40)
        self.assertGreater(a, b)
        self.assertGreater(b, c)

    def test_single_turn_degenerates_to_the_prefix_only(self):
        # With T=1 there is no incremental work, so the only difference between
        # the arms is how many times the prefix is paid.
        a, b, c = M.prefill_tokens(2048, 1, 5, 20, 40)
        self.assertEqual(a, 5 * 2048)
        self.assertEqual(b, 5 * 2048)
        self.assertEqual(c, 2048)


class Project(unittest.TestCase):
    SINGLE = {"prefill_tok_s": 100.0, "decode_tok_s": 10.0}
    MEASURED = {"prefill_tok_s": 100.0, "decode_tok_s": 10.0,
                "decode_batch_agg_tok_s": 25.0}

    def test_totals_are_prefill_plus_decode_for_every_arm(self):
        arms = M.project(1000, 10, 5, 20, 40, self.SINGLE, batch_eff=5.0)
        for name, arm in arms.items():
            self.assertAlmostEqual(arm["total_s"], arm["prefill_s"] + arm["decode_s"],
                                   msg=name)

    def test_serial_arms_share_one_decode_time_fused_arm_does_not(self):
        arms = M.project(1000, 10, 5, 20, 40, self.SINGLE, batch_eff=5.0)
        self.assertEqual(arms["A_naive_stateless"]["decode_s"],
                         arms["B_per_agent_kv"]["decode_s"])
        self.assertLess(arms["C_fak_fused"]["decode_s"],
                        arms["B_per_agent_kv"]["decode_s"])

    def test_only_the_naive_arm_is_flagged_a_lower_bound(self):
        # Arm A's flat-rate prefill ignores O(L^2) attention growth, so its number
        # is a floor and must be labelled as one; A/B are exact.
        arms = M.project(1000, 10, 5, 20, 40, self.SINGLE, batch_eff=5.0)
        self.assertTrue(arms["A_naive_stateless"]["lower_bound"])
        self.assertFalse(arms["B_per_agent_kv"]["lower_bound"])
        self.assertFalse(arms["C_fak_fused"]["lower_bound"])

    def test_a_measured_aggregate_rate_overrides_the_optimistic_division(self):
        # C*T*D = 5*10*20 = 1000 token-decodes. With the MEASURED 25 tok/s
        # aggregate that is 40s regardless of batch_eff; the single-stream
        # fallback at batch_eff=5 would claim 20s. The measured floor must win.
        arms = M.project(1000, 10, 5, 20, 40, self.MEASURED, batch_eff=5.0)
        self.assertAlmostEqual(arms["C_fak_fused"]["decode_s"], 40.0)
        fallback = M.project(1000, 10, 5, 20, 40, self.SINGLE, batch_eff=5.0)
        self.assertAlmostEqual(fallback["C_fak_fused"]["decode_s"], 20.0)

    def test_measured_aggregate_is_insensitive_to_the_batch_eff_argument(self):
        a = M.project(1000, 10, 5, 20, 40, self.MEASURED, batch_eff=5.0)
        b = M.project(1000, 10, 5, 20, 40, self.MEASURED, batch_eff=1.0)
        self.assertEqual(a["C_fak_fused"]["decode_s"], b["C_fak_fused"]["decode_s"])

    def test_prefill_seconds_track_the_token_counts(self):
        arms = M.project(1000, 10, 5, 20, 40, self.SINGLE, batch_eff=5.0)
        a_tok, b_tok, c_tok = M.prefill_tokens(1000, 10, 5, 20, 40)
        self.assertEqual(arms["A_naive_stateless"]["prefill_tokens"], a_tok)
        self.assertAlmostEqual(arms["B_per_agent_kv"]["prefill_s"], b_tok / 100.0)
        self.assertAlmostEqual(arms["C_fak_fused"]["prefill_s"], c_tok / 100.0)


class MeasuredBatchEff(unittest.TestCase):
    def artifact(self, b_ms, c_ms):
        d = tempfile.mkdtemp()
        p = Path(d) / "sessionbench.json"
        p.write_text(json.dumps({"cells": [{"arm_B_per_agent_kv": {"decode_ms": b_ms},
                                            "arm_C_fak_fused": {"decode_ms": c_ms}}]}))
        return str(p)

    def test_raw_ratio_is_reported_and_capped_at_the_agent_count(self):
        # A measured 8x speedup on 5 agents is per-call-overhead amortisation that
        # will NOT transfer to a Metal forward — the projection must use 5.
        raw, capped = M.measured_batch_eff(self.artifact(800.0, 100.0), 5)
        self.assertAlmostEqual(raw, 8.0)
        self.assertAlmostEqual(capped, 5.0)

    def test_a_ratio_below_the_agent_count_passes_through_unchanged(self):
        raw, capped = M.measured_batch_eff(self.artifact(300.0, 100.0), 5)
        self.assertAlmostEqual(raw, 3.0)
        self.assertAlmostEqual(capped, 3.0)


class FmtDur(unittest.TestCase):
    def test_units_switch_at_the_minute_and_hour_boundaries(self):
        self.assertEqual(M.fmt_dur(59.9), "59.9 s")
        self.assertEqual(M.fmt_dur(60), "1.0 min")
        self.assertEqual(M.fmt_dur(3599), "60.0 min")
        self.assertEqual(M.fmt_dur(3600), "1.0 h")


class RateCards(unittest.TestCase):
    def test_every_card_carries_its_provenance_note_and_positive_rates(self):
        # A rate card without a note is an unattributed number; the whole point of
        # this table is that each rate is a citable measurement.
        self.assertTrue(M.RATE_CARDS)
        for name, card in M.RATE_CARDS.items():
            self.assertTrue(card.get("note"), f"{name} has no provenance note")
            self.assertGreater(card["prefill_tok_s"], 0, name)
            self.assertGreater(card["decode_tok_s"], 0, name)


if __name__ == "__main__":
    unittest.main()
