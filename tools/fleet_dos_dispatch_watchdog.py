#!/usr/bin/env python3
"""Portable (mac/linux/windows) keep-alive for the DOS dispatch supervisor.

The Windows analogue is ``tools/fleet_dos_dispatch_watchdog.ps1``; this is the
cross-platform Python sibling so the SAME keep-alive can be armed from
``register_mac_watchers.sh`` (cron) or a launchd plist on mac/linux nodes, where
the PowerShell + Scheduled-Task path does not exist. Closes the gap in issue
#566: the dispatch keep-alive layer was Windows-only, so a dead dispatch
supervisor never auto-restarted on a mac/linux fleet node.

The dispatch supervisor is the long-lived ``dos loop --enact --target N``
process that spawns one worker per free lane and holds the population at the
target. Nothing keeps IT running on a headless node; this watchdog is that
PID-1: one cheap idempotent tick that respawns the supervisor iff one is not
already alive.

Safe by default: it DRY-RUNS unless ``--live`` (or ``FAK_DISPATCH_ENABLE=1``)
is set, so a cron line installed without the opt-in only REPORTS. The repo root
is resolved from this file's own location (``tools/`` lives at the repo root),
never hardcoded to one operator's machine path, and ``dos`` is resolved on PATH.
The launch shape mirrors the Windows watchdog
(``dos loop --enact --workspace <dir> --target N --interval S``) so the two
nodes spawn byte-identical supervisors.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

SCHEMA = "fleet-dos-dispatch-watchdog/1"
DEFAULT_TARGET = 4
DEFAULT_INTERVAL = 120


def repo_root(start: Path | None = None) -> Path:
    """Repo root = parent of tools/ (this file lives at tools/<name>.py)."""
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def resolve_workspace(arg: str) -> Path:
    """--workspace > $DISPATCH_WORKSPACE > repo root (never a hardcoded path)."""
    if arg:
        return Path(arg).resolve()
    env = os.environ.get("DISPATCH_WORKSPACE")
    if env:
        return Path(env).resolve()
    return repo_root()


def build_respawn_command(
    workspace: Path, target: int, interval: int, dos_exe: str = "dos"
) -> list[str]:
    """Pure: the argv that respawns one dispatch supervisor (mirrors the .ps1)."""
    return [
        dos_exe, "loop", "--enact",
        "--workspace", str(workspace),
        "--target", str(int(target)),
        "--interval", str(int(interval)),
    ]


def supervisor_is_alive(cmdlines: Sequence[str]) -> bool:
    """A dispatch supervisor is alive iff a process cmdline carries BOTH ``loop``
    and ``--enact`` (the .ps1's two ANDed clauses). A ``dos loop --json``
    readiness probe (no ``--enact``) and a worker's ``dos-dispatch-loop`` (no
    ``--enact``) do NOT match, so the watchdog never mistakes them for the
    supervisor."""
    for c in cmdlines:
        if "loop" in c and "--enact" in c:
            return True
    return False


def list_process_cmdlines() -> list[str]:
    """Best-effort cross-platform process cmdline list (impure; injected in tests)."""
    try:
        if os.name == "nt":
            # wmic.exe is removed on Win11 24H2+ (build 26200); use the supported
            # CIM API. Win32_Process.CommandLine carries the full argv, so the
            # supervisor's `loop`/`--enact` markers are visible here.
            out = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "Get-CimInstance Win32_Process | ForEach-Object { $_.CommandLine }"],
                capture_output=True, text=True, timeout=30,
            )
        else:
            out = subprocess.run(
                ["ps", "-eo", "args="],
                capture_output=True, text=True, timeout=20,
            )
    except (OSError, subprocess.SubprocessError):
        return []
    return [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()]


def is_live_enabled(args_live: bool) -> bool:
    """Live iff --live OR the opt-in env the cron installer stamps."""
    return bool(args_live) or os.environ.get("FAK_DISPATCH_ENABLE") == "1"


Spawner = Callable[[Sequence[str], Path], "int | None"]


def tick(
    *,
    workspace: Path,
    target: int,
    interval: int,
    live: bool,
    cmdlines: Sequence[str],
    dos_exe: str = "dos",
    spawn: Spawner | None = None,
) -> dict[str, Any]:
    """One idempotent watchdog tick. Pure except for the injected ``spawn``."""
    alive = supervisor_is_alive(cmdlines)
    command = build_respawn_command(workspace, target, interval, dos_exe=dos_exe)
    if alive:
        action = "noop_alive"
    elif live:
        action = "respawn"
    else:
        action = "would_respawn"
    spawned_pid = None
    if action == "respawn" and spawn is not None:
        spawned_pid = spawn(command, workspace)
    return {
        "schema": SCHEMA,
        "alive": alive,
        "live": live,
        "action": action,
        "workspace": str(workspace),
        "target": int(target),
        "interval": int(interval),
        "command": command,
        "spawned_pid": spawned_pid,
    }


def _detached_spawn(command: Sequence[str], workspace: Path) -> int | None:
    """Launch the supervisor DETACHED so it outlives this watchdog tick."""
    exe = shutil.which(command[0]) or command[0]
    argv = [exe, *command[1:]]
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = 0x00000008  # DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        argv, cwd=str(workspace),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs,
    )
    return proc.pid


def render(payload: dict[str, Any]) -> str:
    return (
        f"dispatch-watchdog: action={payload['action']} alive={payload['alive']} "
        f"live={payload['live']} workspace={payload['workspace']}\n"
        f"command: {' '.join(payload['command'])}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Keep the DOS dispatch supervisor alive (portable mac/linux/windows)."
    )
    ap.add_argument("--workspace", default="", help="workspace root (default: $DISPATCH_WORKSPACE or repo root)")
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET, help="desired worker population")
    ap.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="supervisor tick interval seconds")
    ap.add_argument("--live", action="store_true", help="actually respawn (else dry-run); FAK_DISPATCH_ENABLE=1 also enables")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    ws = resolve_workspace(args.workspace)
    live = is_live_enabled(args.live)
    dos_exe = shutil.which("dos") or "dos"
    payload = tick(
        workspace=ws, target=args.target, interval=args.interval, live=live,
        cmdlines=list_process_cmdlines(), dos_exe=dos_exe, spawn=_detached_spawn,
    )
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
