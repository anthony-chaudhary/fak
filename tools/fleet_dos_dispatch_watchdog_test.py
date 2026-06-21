#!/usr/bin/env python3
"""Tests for the portable DOS dispatch-supervisor watchdog (issue #566).

Pure/​injected surface only — no real process spawn, no real ps/wmic call.

Run: ``python -m pytest tools/fleet_dos_dispatch_watchdog_test.py``
"""
from __future__ import annotations

from pathlib import Path

import fleet_dos_dispatch_watchdog as wd


def test_build_respawn_command_shape() -> None:
    ws = Path("/repo/fleet")
    cmd = wd.build_respawn_command(ws, 4, 120, dos_exe="dos")
    assert cmd == [
        "dos", "loop", "--enact",
        "--workspace", str(ws),
        "--target", "4", "--interval", "120",
    ]


def test_supervisor_is_alive_matches_only_enact_loop() -> None:
    assert wd.supervisor_is_alive(["dos loop --enact --workspace . --target 4"])
    # A readiness probe (no --enact) is NOT the supervisor.
    assert not wd.supervisor_is_alive(["dos loop --json --workspace ."])
    # A worker (no --enact) is NOT the supervisor.
    assert not wd.supervisor_is_alive(["claude -p /dos-kernel:dos-dispatch-loop --lane tools"])
    assert not wd.supervisor_is_alive([])


def test_resolve_workspace_precedence(monkeypatch, tmp_path) -> None:
    # explicit arg wins
    assert wd.resolve_workspace(str(tmp_path)) == tmp_path.resolve()
    # then $DISPATCH_WORKSPACE
    monkeypatch.setenv("DISPATCH_WORKSPACE", str(tmp_path))
    assert wd.resolve_workspace("") == tmp_path.resolve()
    # then repo root (parent of tools/)
    monkeypatch.delenv("DISPATCH_WORKSPACE", raising=False)
    assert wd.resolve_workspace("") == wd.repo_root()


def test_repo_root_is_parent_of_tools() -> None:
    # this test file lives in tools/, so repo_root() is its grandparent dir.
    assert wd.repo_root(Path(__file__)) == Path(__file__).resolve().parent.parent


def test_tick_alive_is_noop() -> None:
    calls: list = []
    out = wd.tick(
        workspace=Path("/repo"), target=4, interval=120, live=True,
        cmdlines=["dos loop --enact --target 4"],
        spawn=lambda c, w: calls.append((c, w)) or 999,
    )
    assert out["action"] == "noop_alive"
    assert out["spawned_pid"] is None
    assert calls == []  # never spawns when one is already alive


def test_tick_dry_run_does_not_spawn() -> None:
    calls: list = []
    out = wd.tick(
        workspace=Path("/repo"), target=2, interval=60, live=False,
        cmdlines=[],  # nothing alive
        spawn=lambda c, w: calls.append((c, w)) or 999,
    )
    assert out["action"] == "would_respawn"
    assert out["spawned_pid"] is None
    assert calls == []  # dry-run must NOT spawn


def test_tick_live_respawns_when_dead() -> None:
    calls: list = []

    def spawn(cmd, ws):
        calls.append((list(cmd), ws))
        return 4242

    out = wd.tick(
        workspace=Path("/repo"), target=3, interval=90, live=True,
        cmdlines=["some unrelated process"],  # supervisor not alive
        spawn=spawn,
    )
    assert out["action"] == "respawn"
    assert out["spawned_pid"] == 4242
    assert len(calls) == 1
    assert "--enact" in calls[0][0]


def test_is_live_enabled_honors_env(monkeypatch) -> None:
    monkeypatch.delenv("FAK_DISPATCH_ENABLE", raising=False)
    assert wd.is_live_enabled(False) is False
    assert wd.is_live_enabled(True) is True
    monkeypatch.setenv("FAK_DISPATCH_ENABLE", "1")
    assert wd.is_live_enabled(False) is True


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
