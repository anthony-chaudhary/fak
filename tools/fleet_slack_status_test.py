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
    def _patch(self, mod, rollup_verdict=None, dispatch_verdict=None, fleet_verdict=None):
        mod.dispatch_status.collect = lambda *a, **k: {"verdict": "READY_TO_GROW", "ok": True}
        mod.fleet_top.snapshot = lambda *a, **k: {
            "sessions": {"total": 3}, "system": {"verdict": "HEALTHY"}}
        mod.fleet_top.attach_trend = lambda snap, *a, **k: snap
        mod.post_rollup = lambda dispatch_payload, fleet_snap, **k: dict(
            rollup_verdict if rollup_verdict is not None else
            {"posted": True, "channel": "C0X", "ts": "1"})
        mod.dispatch_status.post_to_slack = lambda payload, **k: dict(
            dispatch_verdict if dispatch_verdict is not None else
            {"posted": True, "channel": "C0X", "ts": "1"})
        mod.fleet_top.post_to_slack = lambda snap, **k: dict(
            fleet_verdict if fleet_verdict is not None else
            {"posted": True, "channel": "C0X", "ts": "2"})

    def test_default_posts_one_rollup(self):
        mod = load()
        self._patch(mod, {"posted": True, "channel": "C0X", "ts": "9"})
        out = mod.run(ROOT)
        self.assertTrue(out["ok"])
        self.assertEqual(out["mode"], "rollup")
        self.assertTrue(out["rollup"]["posted"])
        self.assertEqual(out["rollup"]["ts"], "9")
        self.assertEqual(out["dispatch"]["card_verdict"], "READY_TO_GROW")
        self.assertEqual(out["fleet"]["sessions"], 3)

    def test_rollup_failure_is_not_ok(self):
        mod = load()
        self._patch(mod, {"posted": False, "error": "channel_not_found"})
        out = mod.run(ROOT)
        self.assertFalse(out["ok"])
        self.assertFalse(out["rollup"]["posted"])

    def test_dry_run_counts_as_ok(self):
        mod = load()
        self._patch(mod, {"posted": False, "dry_run": True, "channel": "C0X"})
        out = mod.run(ROOT, dry_run=True)
        self.assertTrue(out["ok"])

    def test_no_fleet_keeps_one_rollup_with_dispatch_only(self):
        mod = load()
        self._patch(mod, {"posted": True, "channel": "C0X"})
        out = mod.run(ROOT, do_fleet=False)
        self.assertIsNotNone(out["dispatch"])
        self.assertIsNone(out["fleet"])
        self.assertTrue(out["rollup"]["posted"])
        self.assertTrue(out["ok"])

    def test_empty_history_path_disables_trend_recording(self):
        mod = load()
        self._patch(mod, {"posted": True, "channel": "C0X"})
        calls = []
        mod.fleet_top.attach_trend = lambda snap, *a, **k: calls.append(k) or snap
        mod.run(ROOT, history_path="")
        self.assertEqual(calls, [])

    def test_separate_mode_still_posts_both_messages(self):
        mod = load()
        self._patch(mod,
                    dispatch_verdict={"posted": True, "channel": "C0X", "ts": "1"},
                    fleet_verdict={"posted": True, "channel": "C0X", "ts": "2"})
        out = mod.run(ROOT, separate=True)
        self.assertTrue(out["ok"])
        self.assertEqual(out["mode"], "separate")
        self.assertTrue(out["dispatch"]["posted"])
        self.assertTrue(out["fleet"]["posted"])
        self.assertIsNone(out["rollup"])

    def test_separate_mode_one_failed_is_not_ok_but_other_still_runs(self):
        mod = load()
        self._patch(mod,
                    dispatch_verdict={"posted": True, "channel": "C0X"},
                    fleet_verdict={"posted": False, "error": "channel_not_found"})
        out = mod.run(ROOT, separate=True)
        self.assertFalse(out["ok"])
        self.assertTrue(out["dispatch"]["posted"])
        self.assertFalse(out["fleet"]["posted"])

    def test_skipped_post_is_not_ok(self):
        mod = load()
        self._patch(mod, {"posted": False, "skipped": "no channel resolved"})
        out = mod.run(ROOT)
        self.assertFalse(out["ok"])


class RollupTextTest(unittest.TestCase):
    def test_rollup_uses_operator_terms_and_one_message_shape(self):
        mod = load()
        dispatch_payload = mod.fixture_dispatch_payload(ROOT)
        fleet_snap = mod.fixture_fleet_snapshot()
        fleet_snap["trend"] = "trend: usable 3→1 ▇▁ (-2 over 2)"

        text = mod.rollup_text(dispatch_payload, fleet_snap)

        self.assertIn("*fleet roll-up", text)
        self.assertIn("issue work:", text)
        self.assertIn("open ticket(s)", text)
        self.assertIn("agent sessions:", text)
        self.assertIn("needs you:", text)
        self.assertIn("being handled:", text)
        self.assertIn("ticket trend:", text)
        self.assertIn("capacity trend: usable 3→1", text)
        self.assertNotIn("plane:", text)
        self.assertNotIn("```", text)

    def test_rollup_clean_state_says_no_human_action(self):
        mod = load()
        dispatch_payload = mod.fixture_dispatch_payload(ROOT)
        dispatch_payload["ok"] = True
        dispatch_payload["verdict"] = "READY_TO_GROW"
        dispatch_payload["backend_health"] = {"dead": [], "stub_rate": []}
        dispatch_payload["hook_health"] = {"by_backend": []}
        dispatch_payload["throughput"] = {"na": True}
        fleet_snap = mod.fleet_top.build_snapshot(
            {"rows": [], "accounts": [], "throttle": {}},
            workspace="C:/work/fak", window_h=10.0, now="2026-06-29T18:00:00Z")

        text = mod.rollup_text(dispatch_payload, fleet_snap)

        self.assertIn("needs you: none", text)


class PostRollupTest(unittest.TestCase):
    SLACK_KEYS = ("FAK_DISPATCH_TOKEN", "FAK_DISPATCH_CHANNEL", "FAK_SCOREBOARD_TOKEN")

    def _clear_env(self):
        import os
        saved = {k: os.environ.pop(k, None) for k in self.SLACK_KEYS}
        self.addCleanup(self._restore_env, saved)

    def _restore_env(self, saved):
        import os
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_post_rollup_sends_one_combined_message(self):
        import json as _json
        import os
        mod = load()
        self._clear_env()
        os.environ["FAK_SCOREBOARD_TOKEN"] = "xoxb-test-tok"
        calls = []

        def transport(url, body, headers, timeout):
            calls.append({"body": _json.loads(body.decode("utf-8")),
                          "auth": headers.get("Authorization")})
            return 200, _json.dumps({"ok": True, "ts": "7.7", "channel": "C0FLEET"})

        verdict = mod.post_rollup(
            mod.fixture_dispatch_payload(ROOT),
            mod.fixture_fleet_snapshot(),
            channel="C0FLEET",
            transport=transport,
        )

        self.assertTrue(verdict["posted"])
        self.assertEqual(len(calls), 1)
        text = calls[0]["body"]["text"]
        self.assertIn("*fleet roll-up", text)
        self.assertIn("issue work:", text)
        self.assertIn("agent sessions:", text)
        self.assertNotIn("*dispatch scheduler:*", text)
        self.assertNotIn("*agent session health", text)


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

    def test_after_body_is_one_rollup_message(self):
        mod = load()
        score = mod.signal_score(ROOT)
        self.assertIn("*fleet roll-up", score["after_body"])
        self.assertNotIn("*dispatch scheduler:*", score["after_body"])
        self.assertNotIn("*agent session health", score["after_body"])

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
