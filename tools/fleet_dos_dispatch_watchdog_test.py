#!/usr/bin/env python3
"""Tests for the portable DOS dispatch-supervisor watchdog (issue #566).

Pure/injected surface only — no real process spawn, no real ps/wmic call.

Pure stdlib (``unittest``), no pytest — so CI runs it as a plain script,
exactly like its dispatch-loop tool-test siblings (``python tools/<name>.py``).

Run: ``python tools/fleet_dos_dispatch_watchdog_test.py``
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import fleet_dos_dispatch_watchdog as wd


class _EnvTestCase(unittest.TestCase):
    """Base with a monkeypatch.setenv/delenv stand-in that auto-restores."""

    def set_env(self, key: str, value: str | None) -> None:
        old = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

        def restore() -> None:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

        self.addCleanup(restore)


class BuildRespawnCommandTest(unittest.TestCase):
    def test_build_respawn_command_shape(self) -> None:
        ws = Path("/repo/fleet")
        cmd = wd.build_respawn_command(ws, 4, 120, dos_exe="dos")
        self.assertEqual(
            cmd,
            [
                "dos", "loop", "--enact",
                "--workspace", str(ws),
                "--target", "4", "--interval", "120",
            ],
        )


class SupervisorIsAliveTest(unittest.TestCase):
    def test_supervisor_is_alive_matches_only_enact_loop(self) -> None:
        self.assertTrue(
            wd.supervisor_is_alive(["dos loop --enact --workspace . --target 4"])
        )
        # A readiness probe (no --enact) is NOT the supervisor.
        self.assertFalse(wd.supervisor_is_alive(["dos loop --json --workspace ."]))
        # A worker (no --enact) is NOT the supervisor.
        self.assertFalse(
            wd.supervisor_is_alive(
                ["claude -p /dos-kernel:dos-dispatch-loop --lane tools"]
            )
        )
        self.assertFalse(wd.supervisor_is_alive([]))


class ResolveWorkspaceTest(_EnvTestCase):
    def test_resolve_workspace_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            # explicit arg wins
            self.assertEqual(wd.resolve_workspace(str(tmp)), tmp.resolve())
            # then $DISPATCH_WORKSPACE
            self.set_env("DISPATCH_WORKSPACE", str(tmp))
            self.assertEqual(wd.resolve_workspace(""), tmp.resolve())
            # then repo root (parent of tools/)
            self.set_env("DISPATCH_WORKSPACE", None)
            self.assertEqual(wd.resolve_workspace(""), wd.repo_root())

    def test_repo_root_is_parent_of_tools(self) -> None:
        # this test file lives in tools/, so repo_root() is its grandparent dir.
        self.assertEqual(
            wd.repo_root(Path(__file__)), Path(__file__).resolve().parent.parent
        )


class TickTest(unittest.TestCase):
    def test_tick_alive_is_noop(self) -> None:
        calls: list = []
        out = wd.tick(
            workspace=Path("/repo"), target=4, interval=120, live=True,
            cmdlines=["dos loop --enact --target 4"],
            spawn=lambda c, w: calls.append((c, w)) or 999,
        )
        self.assertEqual(out["action"], "noop_alive")
        self.assertIsNone(out["spawned_pid"])
        self.assertEqual(calls, [])  # never spawns when one is already alive

    def test_tick_dry_run_does_not_spawn(self) -> None:
        calls: list = []
        out = wd.tick(
            workspace=Path("/repo"), target=2, interval=60, live=False,
            cmdlines=[],  # nothing alive
            spawn=lambda c, w: calls.append((c, w)) or 999,
        )
        self.assertEqual(out["action"], "would_respawn")
        self.assertIsNone(out["spawned_pid"])
        self.assertEqual(calls, [])  # dry-run must NOT spawn

    def test_tick_live_respawns_when_dead(self) -> None:
        calls: list = []

        def spawn(cmd, ws):
            calls.append((list(cmd), ws))
            return 4242

        out = wd.tick(
            workspace=Path("/repo"), target=3, interval=90, live=True,
            cmdlines=["some unrelated process"],  # supervisor not alive
            spawn=spawn,
        )
        self.assertEqual(out["action"], "respawn")
        self.assertEqual(out["spawned_pid"], 4242)
        self.assertEqual(len(calls), 1)
        self.assertIn("--enact", calls[0][0])


class IsLiveEnabledTest(_EnvTestCase):
    def test_is_live_enabled_honors_env(self) -> None:
        self.set_env("FAK_DISPATCH_ENABLE", None)
        self.assertIs(wd.is_live_enabled(False), False)
        self.assertIs(wd.is_live_enabled(True), True)
        self.set_env("FAK_DISPATCH_ENABLE", "1")
        self.assertIs(wd.is_live_enabled(False), True)


if __name__ == "__main__":
    unittest.main()
