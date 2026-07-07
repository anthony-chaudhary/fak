#!/usr/bin/env python3
"""Hermetic tests for tools/meta_superloop_night.py.

The one thing worth testing in isolation is the wave gate: `wave_decision` is a
pure fold from the preflight dict + cron-ownership flag to a typed decision, and it
is what makes a tight (5-minute) cadence safe. We assert every branch — spawn,
skip-if-inflight (no headroom), skip-refused (preflight declined), skip-cron — plus
that each skip carries an observable, non-empty reason. No subprocess runs.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "meta_superloop_night.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("meta_superloop_night", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: the module-level @dataclass (TickReport) resolves
    # cls.__module__ against sys.modules during class creation, so an unregistered
    # module raises AttributeError under spec-based load.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = load()


class WaveDecisionTest(unittest.TestCase):
    def test_spawn_when_headroom_free(self):
        pf = {"ok": True, "verdict": "SPAWN_OK", "live": 3, "headroom": 5}
        d = M.wave_decision(pf, cron_owns_waves=False)
        self.assertEqual(d["code"], M.WAVE_SPAWN)
        self.assertIn("5", d["reason"])

    def test_skip_inflight_when_headroom_zero(self):
        # A prior wave still holds every slot — skip, do NOT double-dispatch.
        pf = {"ok": True, "verdict": "SPAWN_OK", "live": 8, "headroom": 0}
        d = M.wave_decision(pf, cron_owns_waves=False)
        self.assertEqual(d["code"], M.WAVE_SKIP_INFLIGHT)
        self.assertTrue(d["reason"])

    def test_skip_inflight_when_headroom_negative(self):
        pf = {"ok": True, "verdict": "SPAWN_OK", "live": 10, "headroom": -2}
        d = M.wave_decision(pf, cron_owns_waves=False)
        self.assertEqual(d["code"], M.WAVE_SKIP_INFLIGHT)

    def test_skip_refused_takes_priority_over_headroom(self):
        # preflight not-ok wins even if a stale headroom number looks positive.
        pf = {"ok": False, "verdict": "REFUSE_HOST", "reason": "host guard dirty", "headroom": 4}
        d = M.wave_decision(pf, cron_owns_waves=False)
        self.assertEqual(d["code"], M.WAVE_SKIP_REFUSED)
        self.assertIn("REFUSE_HOST", d["reason"])

    def test_skip_cron_takes_priority_over_everything(self):
        # A live cron owns the waves — supervise, never double-dispatch, even with headroom.
        pf = {"ok": True, "verdict": "SPAWN_OK", "live": 0, "headroom": 8}
        d = M.wave_decision(pf, cron_owns_waves=True)
        self.assertEqual(d["code"], M.WAVE_SKIP_CRON)
        self.assertTrue(d["reason"])

    def test_every_skip_has_a_nonempty_reason(self):
        # Observability contract: no silent skip — every non-spawn decision explains itself.
        cases = [
            ({"ok": True, "headroom": 0}, False),
            ({"ok": False, "verdict": "REFUSE_AT_CAP", "reason": ""}, False),
            ({"ok": True, "headroom": 8}, True),
        ]
        for pf, cron in cases:
            d = M.wave_decision(pf, cron_owns_waves=cron)
            if d["code"] != M.WAVE_SPAWN:
                self.assertTrue(d["reason"].strip(), f"empty reason for {d['code']}")

    def test_missing_headroom_does_not_crash_and_spawns(self):
        # A preflight that could not compute headroom (None) must not be treated as
        # inflight — fall through to spawn, letting the refill's own preflight bind.
        pf = {"ok": True, "verdict": "SPAWN_OK", "live": None, "headroom": None}
        d = M.wave_decision(pf, cron_owns_waves=False)
        self.assertEqual(d["code"], M.WAVE_SPAWN)


if __name__ == "__main__":
    unittest.main()
