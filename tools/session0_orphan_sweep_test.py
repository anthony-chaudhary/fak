#!/usr/bin/env python3
"""Hermetic tests for tools/session0_orphan_sweep.py.

The load-bearing invariant of the sweep is a *safety* one: its scoping predicate
must never list a live Session-1 fleet process or a Windows service as a kill
target, even though those share the fleet image names. :func:`sweep_targets` is
pure (rows in, targets out), so that invariant is asserted directly here with no
live host, no PowerShell, and no ``taskkill``.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "session0_orphan_sweep.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("session0_orphan_sweep", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load()


def _zombie(pid: int, name: str, ppid: int = 9999) -> dict:
    """A tombstoned jack-barker Session-0 zombie from the incident day."""
    return {
        "pid": pid,
        "ppid": ppid,  # 9999 is never in live_pids -> dead parent chain
        "name": name,
        "sessionid": 0,
        "owner": "NETRA\\jack-barker",
        "created": "2026-06-29T04:11:07Z",
    }


class SweepTargetsSafety(unittest.TestCase):
    def test_selects_the_three_fleet_zombies(self):
        rows = [
            _zombie(101, "opencode.exe"),
            _zombie(102, "cmd.exe"),
            _zombie(103, "python.exe"),
        ]
        live = frozenset({101, 102, 103})  # the zombies are live; their parent 9999 is not
        targets = S.sweep_targets(
            rows, fleet_owners=("jack-barker",), created_on="2026-06-29", live_pids=live
        )
        self.assertEqual(S.target_pids(targets), [101, 102, 103])

    def test_excludes_live_session1_process_with_same_image(self):
        # The core confusion risk: a LIVE Session-1 opencode.exe must never be a
        # target, even though it shares the image name of the zombies.
        rows = [
            _zombie(101, "opencode.exe"),
            {"pid": 201, "ppid": 200, "name": "opencode.exe", "sessionid": 1,
             "owner": "NETRA\\live-worker", "created": "2026-07-08T09:00:00Z"},
        ]
        live = frozenset({101, 201, 200})
        targets = S.sweep_targets(
            rows, fleet_owners=("jack-barker",), created_on="2026-06-29", live_pids=live
        )
        self.assertEqual(S.target_pids(targets), [101])

    def test_excludes_windows_service_in_session0(self):
        # svchost/services run in Session-0 too, owned by SYSTEM -- excluded by
        # both the image rung and the owner rung.
        rows = [
            _zombie(101, "cmd.exe"),
            {"pid": 4, "ppid": 0, "name": "services.exe", "sessionid": 0,
             "owner": "NT AUTHORITY\\SYSTEM", "created": "2026-06-01T00:00:00Z"},
            {"pid": 88, "ppid": 4, "name": "svchost.exe", "sessionid": 0,
             "owner": "NT AUTHORITY\\SYSTEM", "created": "2026-06-01T00:00:00Z"},
        ]
        live = frozenset({101, 4, 88})
        targets = S.sweep_targets(
            rows, fleet_owners=("jack-barker",), created_on="2026-06-29", live_pids=live
        )
        self.assertEqual(S.target_pids(targets), [101])

    def test_excludes_different_live_account_session0_process(self):
        # A Session-0 fleet-image process owned by a DIFFERENT (live) account is
        # not swept: the owner rung requires a positive tombstoned-owner match.
        rows = [
            _zombie(101, "python.exe"),
            {"pid": 301, "ppid": 300, "name": "python.exe", "sessionid": 0,
             "owner": "NETRA\\other-live-acct", "created": "2026-06-29T05:00:00Z"},
        ]
        live = frozenset({101, 301})  # 300 (parent of 301) is dead, but owner saves it
        targets = S.sweep_targets(
            rows, fleet_owners=("jack-barker",), created_on="2026-06-29", live_pids=live
        )
        self.assertEqual(S.target_pids(targets), [101])

    def test_excludes_wrong_date(self):
        # Date-scoping: a same-owner Session-0 zombie from another day is spared.
        rows = [
            _zombie(101, "opencode.exe"),
            {"pid": 401, "ppid": 9999, "name": "opencode.exe", "sessionid": 0,
             "owner": "NETRA\\jack-barker", "created": "2026-07-01T00:00:00Z"},
        ]
        live = frozenset({101, 401})
        targets = S.sweep_targets(
            rows, fleet_owners=("jack-barker",), created_on="2026-06-29", live_pids=live
        )
        self.assertEqual(S.target_pids(targets), [101])

    def test_excludes_live_parent_chain(self):
        # A Session-0 fleet-image process whose parent is still alive is spared
        # (dead-parent rung) -- it is attended, not orphaned.
        rows = [
            {"pid": 501, "ppid": 500, "name": "cmd.exe", "sessionid": 0,
             "owner": "NETRA\\jack-barker", "created": "2026-06-29T04:00:00Z"},
        ]
        live = frozenset({501, 500})  # parent 500 alive
        targets = S.sweep_targets(
            rows, fleet_owners=("jack-barker",), created_on="2026-06-29", live_pids=live
        )
        self.assertEqual(S.target_pids(targets), [])

    def test_denied_owner_never_matches(self):
        # GetOwner denied (None) is never a positive owner match -- a service that
        # denies GetOwner is not swept.
        rows = [
            {"pid": 601, "ppid": 9999, "name": "python.exe", "sessionid": 0,
             "owner": None, "created": "2026-06-29T04:00:00Z"},
        ]
        live = frozenset({601})
        targets = S.sweep_targets(
            rows, fleet_owners=("jack-barker",), created_on="2026-06-29", live_pids=live
        )
        self.assertEqual(S.target_pids(targets), [])

    def test_targets_are_pid_scoped_never_image(self):
        # The output is a list of ints (PIDs). This is what forbids a `/IM` kill.
        rows = [_zombie(101, "opencode.exe"), _zombie(102, "opencode.exe")]
        live = frozenset({101, 102})
        targets = S.sweep_targets(
            rows, fleet_owners=("jack-barker",), created_on="2026-06-29", live_pids=live
        )
        pids = S.target_pids(targets)
        self.assertTrue(all(isinstance(p, int) for p in pids))
        self.assertEqual(pids, [101, 102])


class MainRefusesOverBroad(unittest.TestCase):
    def test_refuses_without_owner_or_date(self):
        # No owner set and no date == too coarse to enact; main() refuses (rc 2).
        rc = S.main(["--owners", "", "--created-on", ""])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
