#!/usr/bin/env python3
"""Hermetic tests for tools/dispatch_status.py.

build_payload() is a pure FOLD over five already-collected sub-tool dicts
(preflight, supervisor, watchdog, backlog, closure). We feed it synthetic dicts
and assert the overall verdict, the watchdog reason line, and the backlog/closure
"na" degradation — no subprocess, no gh, no schtasks. render() is exercised on a
minimal payload to prove it does not raise.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "dispatch_status.py"


def load():
    spec = importlib.util.spec_from_file_location("dispatch_status", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pre(verdict: str = "SPAWN_OK", *, host_safe: bool = True, cap: int = 2,
        live: int = 0) -> dict:
    return {
        "verdict": verdict,
        "reason": f"synthetic {verdict}",
        "cap": cap,
        "live": live,
        "host": {"safe": host_safe},
        "account": {"tag": "worker-a", "tier": 1, "model": "claude", "available": True},
    }


def sup(verdict: str = "READY_TO_CANARY") -> dict:
    return {
        "verdict": verdict,
        "supervise": {"target": 3, "alive": 1},
        "plans": {"total_plans": 2, "total_units": 17},
    }


def backlog_ok() -> dict:
    return {
        "lanes": {"docs": {"issues": [1, 2, 3]}, "agent": {"issues": [4]}},
        "counts": {"open": 4, "routed": 4, "unrouted": 0},
    }


def closure_ok() -> dict:
    return {
        "closure_rate": 0.8,
        "counts": {"TRUE_RESOLVED": 8, "CLAIMED_CLOSED": 10, "OPEN_WITNESSED": 2},
    }


def build(mod, **over):
    kw = dict(
        root=ROOT, pre=pre(), sup=sup(), wd={"installed": True, "status": "Ready"},
        backlog=backlog_ok(), closure=closure_ok(), max_workers=2, fast=False)
    kw.update(over)
    return mod.build_payload(**kw)


class VerdictTest(unittest.TestCase):
    def test_ready_to_grow_when_safe_to_spawn(self) -> None:
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK"))
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "READY_TO_GROW")
        self.assertEqual(p["dispatcher"]["headroom"], 2)

    def test_host_flagged_fails_the_card(self) -> None:
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK", host_safe=False))
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "HOST_FLAGGED")
        self.assertTrue(any("host resource guard flagged" in r for r in p["reasons"]))

    def test_at_cap_is_a_healthy_steady_state(self) -> None:
        mod = load()
        p = build(mod, pre=pre("REFUSE_AT_CAP", cap=2, live=2))
        self.assertTrue(p["ok"])  # at cap is normal, not breakage
        self.assertEqual(p["verdict"], "AT_CAP")
        self.assertEqual(p["dispatcher"]["headroom"], 0)

    def test_blocked_on_account_is_a_healthy_steady_state(self) -> None:
        mod = load()
        p = build(mod, pre=pre("REFUSE_NO_ACCOUNT"))
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "BLOCKED_ON_ACCOUNT")

    def test_inspect_fails_the_card(self) -> None:
        mod = load()
        p = build(mod, pre=pre("REFUSE_INSPECT"))
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "INSPECT")


class BacklogClosureNaTest(unittest.TestCase):
    def test_backlog_na_on_skipped_and_closure_na_on_skipped(self) -> None:
        mod = load()
        p = build(mod, backlog={"_skipped": "fast"}, closure={"_skipped": "fast"}, fast=True)
        self.assertTrue(p["backlog"]["na"])
        self.assertIsNone(p["backlog"]["open_issues"])
        self.assertTrue(p["closure"]["na"])
        self.assertIsNone(p["closure"]["closure_rate"])

    def test_backlog_na_on_error_with_no_lanes(self) -> None:
        mod = load()
        p = build(mod, backlog={"_error": "gh timed out"})
        self.assertTrue(p["backlog"]["na"])

    def test_backlog_present_folds_lane_counts(self) -> None:
        mod = load()
        p = build(mod)
        self.assertFalse(p["backlog"]["na"])
        self.assertEqual(p["backlog"]["open_issues"], 4)
        self.assertEqual(p["backlog"]["by_lane"], {"docs": 3, "agent": 1})
        self.assertEqual(p["backlog"]["routed"], 4)

    def test_closure_present_surfaces_rate_and_open_witnessed(self) -> None:
        mod = load()
        p = build(mod)
        self.assertFalse(p["closure"]["na"])
        self.assertEqual(p["closure"]["closure_rate"], 0.8)
        self.assertEqual(p["closure"]["open_witnessed_closable"], 2)


class WatchdogReasonTest(unittest.TestCase):
    def test_watchdog_installed_reason_line(self) -> None:
        mod = load()
        p = build(mod, wd={"installed": True, "status": "Ready"})
        self.assertTrue(any("watchdog installed (Ready)" in r for r in p["reasons"]))
        self.assertEqual(p["dispatcher"]["watchdog"]["installed"], True)

    def test_watchdog_not_installed_reason_line(self) -> None:
        mod = load()
        p = build(mod, wd={"installed": False, "status": None})
        self.assertTrue(any("watchdog NOT installed" in r for r in p["reasons"]))

    def test_watchdog_unknown_emits_no_install_line(self) -> None:
        mod = load()
        # installed is None (schtasks couldn't run) -> neither install line appears.
        p = build(mod, wd={"installed": None, "error": "schtasks missing"})
        self.assertFalse(any("watchdog" in r for r in p["reasons"]))


class RenderTest(unittest.TestCase):
    def test_render_does_not_raise_on_minimal_payload(self) -> None:
        mod = load()
        p = build(mod)
        text = mod.render(p)
        self.assertIn("DISPATCHER", text)
        self.assertIn("READY_TO_GROW", text)

    def test_render_does_not_raise_on_na_payload(self) -> None:
        mod = load()
        p = build(mod, backlog={"_skipped": "fast"}, closure={"_skipped": "fast"}, fast=True)
        text = mod.render(p)
        self.assertIn("n/a", text)


if __name__ == "__main__":
    unittest.main()
