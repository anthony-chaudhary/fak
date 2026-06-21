#!/usr/bin/env python3
"""Backend-selector launcher for a single DOS dispatch worker.

This is the indirection seam that lets fleet run a MIXED worker fleet -- some
Claude workers, some opencode workers -- behind a single
``dos.toml [supervise].worker_launch_template``. The supervisor (``dos loop
--enact``) spawns this shim; the shim picks the backend and execs the real
worker. Putting the backend choice in the shim (not the template) means:

* the template stays stable: ``python tools/dispatch_worker.py --lane {lane}``
* the backend is switchable per-node via ``FLEET_WORKER_BACKEND`` (no dos.toml
  edit, no second workspace), so one node runs Claude workers and another runs
  opencode workers off the same repo;
* the canary ``--worker-launch-template`` override still works for tests.

Unlike the supervisor/watchdog/canary layer above it (which are dry-run-by-
default), THIS shim launches by default -- it is the leaf launcher, and the
operator has already opted into a live spawn one layer up at the watchdog /
canary. ``--dry-run`` exists for inspection and is the test path.

Backend selection precedence (highest first):

1. ``--backend`` CLI flag.
2. ``FLEET_WORKER_BACKEND`` env var.
3. ``claude`` (the established reference backend).

The selected backend and the lane are stamped into the child env
(``DISPATCH_BACKEND`` / ``DISPATCH_LANE`` / ``DISPATCH_WORKSPACE``) so a worker
can read its assignment from the environment regardless of how its prompt was
rendered -- the same self-describing contract the canary dry-run proved.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

SCHEMA = "fleet-dispatch-worker/1"

BACKENDS = ("claude", "opencode")
DEFAULT_BACKEND = "claude"

# The two launch shapes. Kept here (not read from dos.toml) on purpose: once
# the template becomes ``python tools/dispatch_worker.py --lane {lane}`` this
# module IS the source of truth for how each backend is invoked, so there is no
# second place to drift. Override per-call via the env vars below if a backend
# changes its flags.
CLAUDE_AGENT_PROMPT = "/dos-kernel:dos-dispatch-loop --lane {lane}"
OPENCODE_AGENT = "dos-dispatch"
OPENCODE_MESSAGE = "dispatch lane {lane}"


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def resolve_backend(explicit: str | None, env: dict[str, str] | None) -> str:
    """Pick the backend. Precedence: explicit flag > env > default."""
    if explicit:
        backend = explicit
    else:
        backend = (env or os.environ).get("FLEET_WORKER_BACKEND", DEFAULT_BACKEND)
    backend = backend.strip().lower()
    if backend not in BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {BACKENDS} "
            f"(via --backend or FLEET_WORKER_BACKEND)"
        )
    return backend


def build_command(lane: str, backend: str) -> list[str]:
    """Pure: the logical argv for one worker launch (no path resolution)."""
    if not lane:
        raise ValueError("lane must be a non-empty string")
    if backend == "claude":
        return [
            "claude",
            "-p",
            "--permission-mode",
            "bypassPermissions",
            CLAUDE_AGENT_PROMPT.format(lane=lane),
        ]
    if backend == "opencode":
        return [
            "opencode",
            "run",
            "--dangerously-skip-permissions",
            "--agent",
            OPENCODE_AGENT,
            OPENCODE_MESSAGE.format(lane=lane),
        ]
    raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")


def child_env(
    lane: str,
    backend: str,
    workspace: Path,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """The env the child worker runs under.

    ``DISPATCH_WORKSPACE`` / ``DISPATCH_LANE`` / ``DISPATCH_BACKEND`` are the
    self-describing contract a worker reads to know its assignment independent
    of prompt rendering (the canary dry-run proved ``DISPATCH_WORKSPACE``).
    """
    env = dict(base if base is not None else os.environ)
    env["DISPATCH_WORKSPACE"] = str(workspace)
    env["DISPATCH_LANE"] = lane
    env["DISPATCH_BACKEND"] = backend
    return env


def resolve_exe(name: str) -> str:
    """Resolve a backend shim to a launchable path.

    On Windows the npm shims are ``claude.cmd`` / ``opencode.cmd``;
    ``shutil.which`` finds them via PATHEXT so we can exec without ``shell=True``
    (which would mangle the prompt argument).
    """
    found = shutil.which(name)
    return found or name


Runner = Callable[[Sequence[str], Path, dict[str, str]], dict[str, Any]]


def launch(
    command: Sequence[str],
    cwd: Path,
    env: dict[str, str],
    *,
    runner: Runner | None = None,
    timeout_s: int | None = None,
) -> dict[str, Any]:
    """Exec a worker command. ``runner`` is injectable for hermetic tests.

    The real launcher resolves the backend shim to a full path (so a Windows
    ``.cmd`` shim execs without a shell) and streams stdio to the parent so the
    supervisor sees worker output inline.
    """
    if runner is not None:
        return runner(command, cwd, env)

    resolved = list(command)
    if resolved:
        resolved[0] = resolve_exe(resolved[0])
    try:
        proc = subprocess.run(resolved, cwd=cwd, env=env, timeout=timeout_s)
    except FileNotFoundError as exc:
        return {"returncode": 127, "error": str(exc), "stdout": "", "stderr": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "timeout": True,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or str(exc),
        }
    return {"returncode": proc.returncode, "stdout": "", "stderr": ""}


def build_payload(
    *,
    lane: str,
    backend: str,
    workspace: Path,
    dry_run: bool,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    command = build_command(lane, backend) if not error else []
    ok = error is None and (result is None or result.get("returncode") == 0)
    return {
        "schema": SCHEMA,
        "ok": ok,
        "lane": lane,
        "backend": backend,
        "workspace": str(workspace),
        "dry_run": dry_run,
        "command": command,
        "env": {"DISPATCH_WORKSPACE": str(workspace), "DISPATCH_LANE": lane, "DISPATCH_BACKEND": backend},
        "result": result,
        "error": error,
    }


def render(payload: dict[str, Any]) -> str:
    cmd = " ".join(payload.get("command") or []) or "-"
    lines = [
        f"dispatch-worker: backend={payload.get('backend')} lane={payload.get('lane')} dry_run={payload.get('dry_run')}",
        f"command: {cmd}",
    ]
    if payload.get("error"):
        lines.append(f"error: {payload['error']}")
    result = payload.get("result")
    if isinstance(result, dict):
        lines.append(f"returncode: {result.get('returncode')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Launch one DOS dispatch worker on a selected backend.")
    ap.add_argument("--lane", required=True, help="lane to dispatch on (required)")
    ap.add_argument("--backend", choices=BACKENDS, default=None, help="worker backend (default: env FLEET_WORKER_BACKEND or claude)")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--dry-run", action="store_true", help="print the command instead of launching")
    ap.add_argument("--timeout-s", type=int, default=None, help="child timeout in seconds (default: none)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    error: str | None = None
    backend = DEFAULT_BACKEND
    try:
        backend = resolve_backend(args.backend, None)
    except ValueError as exc:
        error = str(exc)

    if args.dry_run or error:
        payload = build_payload(
            lane=args.lane, backend=backend, workspace=workspace, dry_run=True, error=error
        )
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(render(payload))
        return 0 if not error else 2

    command = build_command(args.lane, backend)
    env = child_env(args.lane, backend, workspace)
    result = launch(command, workspace, env, timeout_s=args.timeout_s)
    payload = build_payload(
        lane=args.lane, backend=backend, workspace=workspace, dry_run=False, result=result
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    return int(result.get("returncode") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
