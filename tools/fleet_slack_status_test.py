#!/usr/bin/env python3
"""Hermetic tests for tools/fleet_slack_status.py.

The orchestrator is tested with dispatch_status.collect / .post_to_slack and
fleet_top.snapshot / .post_to_slack stubbed out, so no gh, no subprocess, no
network — only the fold (which posts ran, and the combined ok verdict) is pinned.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "fleet_slack_status.py"
sys.path.insert(0, str(SCRIPT.parent))


def load():
    spec = importlib.util.spec_from_file_location("fleet_slack_status", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RunTest(unittest.TestCase):
    def _patch(self, mod, dispatch_verdict, fleet_verdict):
        mod.dispatch_status.collect = lambda *a, **k: {"verdict": "READY_TO_GROW"}
        mod.dispatch_status.post_to_slack = lambda payload, **k: dict(dispatch_verdict)
        mod.fleet_top.snapshot = lambda *a, **k: {"sessions": {"total": 3}}
        mod.fleet_top.post_to_slack = lambda snap, **k: dict(fleet_verdict)

    def test_both_posted_is_ok(self):
        mod = load()
        self._patch(mod, {"posted": True, "channel": "C0X", "ts": "1"},
                    {"posted": True, "channel": "C0X", "ts": "2"})
        out = mod.run(ROOT)
        self.assertTrue(out["ok"])
        self.assertTrue(out["dispatch"]["posted"])
        self.assertTrue(out["fleet"]["posted"])
        self.assertEqual(out["dispatch"]["card_verdict"], "READY_TO_GROW")
        self.assertEqual(out["fleet"]["sessions"], 3)

    def test_one_failed_is_not_ok_but_other_still_runs(self):
        mod = load()
        self._patch(mod, {"posted": True, "channel": "C0X"},
                    {"posted": False, "error": "channel_not_found"})
        out = mod.run(ROOT)
        self.assertFalse(out["ok"])
        self.assertTrue(out["dispatch"]["posted"])      # the other card still posted
        self.assertFalse(out["fleet"]["posted"])

    def test_dry_run_counts_as_ok(self):
        mod = load()
        self._patch(mod, {"posted": False, "dry_run": True, "channel": "C0X"},
                    {"posted": False, "dry_run": True, "channel": "C0X"})
        out = mod.run(ROOT, dry_run=True)
        self.assertTrue(out["ok"])

    def test_no_fleet_only_posts_dispatch(self):
        mod = load()
        self._patch(mod, {"posted": True, "channel": "C0X"},
                    {"posted": True, "channel": "C0X"})
        out = mod.run(ROOT, do_fleet=False)
        self.assertIsNotNone(out["dispatch"])
        self.assertIsNone(out["fleet"])
        self.assertTrue(out["ok"])

    def test_skipped_post_is_not_ok(self):
        mod = load()
        self._patch(mod, {"posted": False, "skipped": "no channel resolved"},
                    {"posted": False, "skipped": "no channel resolved"})
        out = mod.run(ROOT)
        self.assertFalse(out["ok"])


class ClassifySignalNoiseTest(unittest.TestCase):
    """The list-free signal/noise classifier: every char in exactly one bucket, with
    box-drawing/fence/whitespace and restated tokens as noise, first occurrences as
    signal, and the meta self-score footer excluded from the content measure."""

    def test_buckets_partition_every_char(self):
        mod = load()
        # "alpha beta" -> two new tokens (signal 9), one separating space (noise 1),
        # plus the trailing newline (noise 1).
        m = mod.classify_signal_noise("alpha beta")
        self.assertEqual(m["signal"], 9)            # len("alpha")+len("beta")
        self.assertEqual(m["space"], 2)             # the inter-word space + newline
        self.assertEqual(m["box"], 0)
        self.assertEqual(m["fence"], 0)
        self.assertEqual(m["redundant"], 0)
        self.assertEqual(m["total"], m["signal"] + m["noise"])

    def test_box_drawing_and_fence_are_noise(self):
        mod = load()
        m = mod.classify_signal_noise("```\n║ x\n```")
        self.assertGreater(m["fence"], 0)           # the two ``` delimiter lines
        self.assertGreater(m["box"], 0)             # the ║ rail glyph
        self.assertEqual(m["signal"], 1)            # only "x" is content

    def test_repeated_token_is_redundant_noise(self):
        mod = load()
        m = mod.classify_signal_noise("worker worker worker")
        self.assertEqual(m["signal"], len("worker"))         # first occurrence only
        self.assertEqual(m["redundant"], 2 * len("worker"))  # the two restatements

    def test_meta_self_score_footer_excluded_by_default(self):
        mod = load()
        body = "1/2 live · backlog 47\n_S/N self-score 3.0 (3 signal / 1 noise): basis_"
        kept = mod.classify_signal_noise(body)                       # excludes the footer
        full = mod.classify_signal_noise(body, exclude_meta=False)   # counts it
        self.assertLess(kept["noise"], full["noise"])
        self.assertLess(kept["total"], full["total"])


class SignalScoreTest(unittest.TestCase):
    """The end-to-end scorecard: rendering the canonical fixture compact vs
    boxed-and-fenced cuts Slack noise-debt at least the 3x the goal asks for."""

    def test_hits_3x_noise_reduction(self):
        mod = load()
        score = mod.signal_score(ROOT)
        self.assertEqual(score["schema"], mod.SIGNAL_SCHEMA)
        self.assertTrue(score["ok"], score)
        self.assertEqual(score["verdict"], "SIGNAL_3X")
        self.assertGreaterEqual(score["noise_multiple"], mod.SIGNAL_TARGET_MULTIPLE)
        # the after-card is genuinely smaller, and density rose (densified, not gutted)
        self.assertLess(score["noise_debt_after"], score["noise_debt_before"])
        self.assertGreater(score["combined"]["signal_ratio_after"],
                           score["combined"]["signal_ratio_before"])

    def test_per_card_carries_both_and_dispatch_is_strong(self):
        mod = load()
        score = mod.signal_score(ROOT)
        self.assertEqual(set(score["cards"]), {"dispatch", "fleet"})
        # the dispatcher card drops the whole restated ╚═ footer, so its reduction is large
        self.assertGreaterEqual(score["cards"]["dispatch"]["noise_multiple"], 3.0)
        for cd in score["cards"].values():
            self.assertIn("before", cd)
            self.assertIn("after", cd)

    def test_meta_overhead_reported_not_hidden(self):
        mod = load()
        score = mod.signal_score(ROOT)
        # the with-footer body is measured too, so the meta overhead is surfaced, never hidden
        self.assertGreaterEqual(score["meta_footer_overhead"], 0)
        self.assertGreaterEqual(score["combined"]["noise_after_with_meta"],
                                score["noise_debt_after"])

    def test_deterministic(self):
        mod = load()
        self.assertEqual(mod.signal_score(ROOT), mod.signal_score(ROOT))


if __name__ == "__main__":
    unittest.main()
