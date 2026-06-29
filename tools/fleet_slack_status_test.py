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


if __name__ == "__main__":
    unittest.main()
