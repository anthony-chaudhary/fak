#!/usr/bin/env python3
"""Unit tests for tools/dispatch_status_common.py."""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

_TOOLS_DIR = str(Path(__file__).resolve().parent)


def _common():
    if _TOOLS_DIR not in sys.path:
        sys.path.insert(0, _TOOLS_DIR)
    return importlib.import_module("dispatch_status_common")


class DispatchStatusCommonTest(unittest.TestCase):
    def test_json_int_and_time_roundtrips(self) -> None:
        c = _common()
        self.assertEqual(c._last_json('noise\n{"ok": true, "n": 2}\n'), {"ok": True, "n": 2})
        self.assertEqual(c._last_json("not json"), {})
        self.assertEqual(c._int("42", 0), 42)
        self.assertEqual(c._int("bad", 7), 7)

        ep = c._iso_epoch("2026-07-07T03:30:44.123Z")
        self.assertIsNotNone(ep)
        self.assertEqual(c._epoch_iso(ep), "2026-07-07T03:30:44Z")
        self.assertIsNone(c._iso_epoch("not-a-timestamp"))
        self.assertEqual(c._age_text(0.5), "<1m")
        self.assertEqual(c._age_text(90), "1.5h")
        self.assertEqual(c._age_text(None), "?")

    def test_capacity_headroom_and_reconcile(self) -> None:
        c = _common()
        self.assertEqual(c._slots_headroom_note(2, 5), "headroom 3")
        self.assertEqual(c._slots_headroom_note(6, 4), "2 over the 4-slot target")
        self.assertEqual(c._slots_headroom_note(None, 4), "headroom UNKNOWN")
        self.assertEqual(c._workers_live_clause({"live": None, "cap": 4}),
                         "UNKNOWN (dispatcher read incomplete)")

        msg = c._capacity_reconcile({"verdict": "REFUSE_NO_SEAT"}, {"live": 1, "cap": 4})
        self.assertIn("gated by the resolver, not host slots", msg)
        msg2 = c._capacity_reconcile({"verdict": "SPAWN_OK"}, {"live": 4, "cap": 4})
        self.assertIn("gated by host slots", msg2)

    def test_preflight_blocker_limiter_and_next_action(self) -> None:
        c = _common()
        blocker = c._current_preflight_launch_blocker(
            "SPAWN_OK",
            {"verdict": "REFUSE_NO_SEAT", "product": "claude"},
        )
        self.assertEqual(blocker["code"], "NO_FREE_SEAT")
        self.assertEqual(blocker["source"], "claude")

        limiter = c._dispatch_limiter(
            {"cap": 4, "live": 1, "seat": {"total": 3, "free": 1, "leased": 2}},
            {"_error": "API rate limit exceeded"},
            {},
            {"active_count": 0, "blocking_count": 0},
        )
        self.assertEqual(limiter["primary"], "github_rate_limit")
        self.assertIn("github_error=rate_limit", c._dispatch_limiter_terms(limiter))

        self.assertIn(
            "wait for launch admission",
            c._next_action(
                c.VERDICT_LAUNCH_COOLDOWN,
                {"reason": "DEFER", "retry_after": "2026-07-07T04:00Z", "selected": "gem8"},
                {},
                [],
            ),
        )
        self.assertEqual(c._lab_link_label({"phase": "CLEAR", "detail": "ready"}), "CLEAR")
        self.assertEqual(
            c._lab_link_label({"phase": "WAITING", "detail": "private-recovery"}),
            "WAITING/private-recovery",
        )


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DispatchStatusCommonTest)
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
