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
import os
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


class TreeBuildsTest(unittest.TestCase):
    """`tree_builds` must compile to os.devnull, never into the checkout. A plain
    `go build ./cmd/fak` writes `fak.exe` into the cwd; on Windows a LIVE fak
    process holds that file locked, so the build fails with a file-lock error that
    is NOT a compile error — a false BROKEN that mis-routes the refill even on a
    green tree. These tests pin the discard so that regression can't return."""

    def _patch_run(self, rc, err=""):
        seen = {}

        def fake_run(cmd, timeout=240):
            seen["cmd"] = cmd
            seen["timeout"] = timeout
            return rc, "", err

        self._orig_run = M.run
        M.run = fake_run
        self.addCleanup(lambda: setattr(M, "run", self._orig_run))
        return seen

    def test_builds_to_devnull_not_into_checkout(self):
        seen = self._patch_run(0)
        ok, msg = M.tree_builds()
        self.assertTrue(ok)
        self.assertIn("OK", msg)
        cmd = seen["cmd"]
        # The load-bearing regression guard: output is discarded to os.devnull, so
        # the check never contends with a running fak.exe.
        self.assertIn("-o", cmd)
        self.assertIn(os.devnull, cmd)
        self.assertEqual(cmd[cmd.index("-o") + 1], os.devnull)

    def test_broken_tree_surfaces_first_error_line(self):
        seen = self._patch_run(1, err="# some/pkg\nbad.go:1: undefined: X\n")
        ok, msg = M.tree_builds()
        self.assertFalse(ok)
        self.assertIn("BROKEN", msg)
        self.assertIn("# some/pkg", msg)
        # Even the failing path must discard output — a file-lock error must never
        # be mistaken for a compile break.
        self.assertIn(os.devnull, seen["cmd"])


class ClassifyTreeBreakTest(unittest.TestCase):
    """The tree-break classifier is pure (offending file, tracked?, mtime) -> a typed
    recommendation, and it must NEVER escalate to an auto-mutate: a snapshot cannot
    prove a red file is not under a peer's active edit (witnessed flap, 2026-07-09)."""

    def test_unlocalized_when_no_offender(self):
        d = M.classify_tree_break(None, tracked=None, age_min=None)
        self.assertEqual(d["code"], M.TREE_UNLOCALIZED)
        self.assertTrue(d["action"].strip())

    def test_untracked_offender_is_quarantine_safe_and_reversible(self):
        d = M.classify_tree_break("cmd/fak/eve_inspect.go", tracked=False, age_min=388.0)
        self.assertEqual(d["code"], M.TREE_QUARANTINE_SAFE)
        self.assertIn("eve_inspect.go", d["action"])
        self.assertIn("mv", d["action"])          # recommends the reversible move
        self.assertIn("reversible", d["action"])

    def test_tracked_offender_is_coupled_never_edited_beside_owner(self):
        d = M.classify_tree_break("cmd/fak/eve.go", tracked=True, age_min=5.0)
        self.assertEqual(d["code"], M.TREE_COUPLED_TRACKED)
        self.assertIn("owning worker", d["action"])

    def test_parse_offender_first_file_line_windows_slashes(self):
        err = ("# github.com/x/y/cmd/fak\n"
               "cmd\\fak\\eve_inspect.go:56:29: undefined: evebridge.InspectFS\n")
        self.assertEqual(M.parse_build_offender(err), "cmd/fak/eve_inspect.go")

    def test_parse_offender_none_without_file_locus(self):
        self.assertIsNone(M.parse_build_offender("# pkg\nlink error: cannot find symbol\n"))
        self.assertIsNone(M.parse_build_offender(""))

    def test_diagnose_tree_is_noop_on_green(self):
        # A green tree must not spend a second `go build` — diagnose returns OK flat.
        d = M.diagnose_tree(True)
        self.assertEqual(d["code"], M.TREE_OK)
        self.assertIsNone(d["offender"])


class EffectivenessTest(unittest.TestCase):
    """The effectiveness fold surfaces the ships-per-worker leaks (silent exits +
    majority-stub backends) so the operator can act — it never kills unattended."""

    def test_leaking_on_silent_workers(self):
        s = {"workers": {"silent_count": 2, "silent": [{"issue": 11}, {"issue": 22}]},
             "backend_health": {"stub_rate": []}}
        d = M.effectiveness_summary(s)
        self.assertEqual(d["verdict"], M.LEAKING)
        self.assertEqual(d["silent_count"], 2)
        self.assertIn(11, d["silent_issues"])
        self.assertIn("free the slot", d["action"])

    def test_leaking_on_majority_stub_backend(self):
        s = {"workers": {"silent_count": 0, "silent": []},
             "backend_health": {"stub_rate": [
                 {"backend": "codex", "majority_stub": True, "stub_rate": 1.0},
                 {"backend": "claude", "majority_stub": False}]}}
        d = M.effectiveness_summary(s)
        self.assertEqual(d["verdict"], M.LEAKING)
        self.assertEqual(d["majority_stub"], ["codex"])
        self.assertIn("cool", d["action"])

    def test_majority_stub_without_backend_name_is_unattributed(self):
        # Live payloads carry majority-stub rows with a null backend (no .backend
        # sidecar) — still a real leak; surfaced as "unattributed", never dropped.
        s = {"workers": {"silent_count": 0, "silent": []},
             "backend_health": {"stub_rate": [{"backend": None, "majority_stub": True}]}}
        d = M.effectiveness_summary(s)
        self.assertEqual(d["verdict"], M.LEAKING)
        self.assertEqual(d["majority_stub"], ["unattributed"])
        self.assertNotIn("None", d["action"])

    def test_effective_when_no_leak(self):
        s = {"workers": {"silent_count": 0, "silent": []},
             "backend_health": {"stub_rate": [{"backend": "claude", "majority_stub": False}]}}
        d = M.effectiveness_summary(s)
        self.assertEqual(d["verdict"], M.EFFECTIVE)

    def test_none_payload_is_unknown_not_crash(self):
        d = M.effectiveness_summary(None)
        self.assertEqual(d["verdict"], "UNKNOWN")
        self.assertEqual(d["silent_issues"], [])
        self.assertTrue(d["action"].strip())

    def test_silent_count_derived_when_absent(self):
        # A payload with a silent list but no explicit count still reports LEAKING.
        s = {"workers": {"silent": [{"issue": 7}]}, "backend_health": {}}
        d = M.effectiveness_summary(s)
        self.assertEqual(d["verdict"], M.LEAKING)
        self.assertEqual(d["silent_count"], 1)


if __name__ == "__main__":
    unittest.main()
