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
import signal
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

SCHEMA = "fleet-dispatch-worker/1"

BACKENDS = ("claude", "opencode", "codex")
DEFAULT_BACKEND = "claude"

# CREATE_NO_WINDOW — a console child spawned WITHOUT it forces Windows to allocate a
# fresh, VISIBLE console window whenever the parent is windowless, which the scheduled
# dispatch tasks are (Task Scheduler launches them via pythonw.exe, no console of their
# own). Every windowless dispatch tick that then runs a console tool — taskkill,
# tasklist, cmd/mklink, gh, git, fak — pops its OWN window: the "random popup windows"
# the detached-worker path already suppresses (issue_resolve_dispatch.win_creationflags
# / claude_agent_chat.detached_creationflags). This is the SHARED suppressor every
# helper subprocess call in the dispatch family routes through so the suppression is
# total, not just on the worker spawn.
_CREATE_NO_WINDOW = 0x08000000


def no_window_creationflags() -> int:
    """``creationflags`` that keep a synchronously-spawned console child from popping a
    visible window on Windows; ``0`` on POSIX (where ``creationflags`` must be 0). Spread
    into every helper ``subprocess.run``/``Popen`` in the dispatch path."""
    return _CREATE_NO_WINDOW if os.name == "nt" else 0

# Default wall-clock cap on a spawned worker session (seconds). A dispatch worker
# is a full agentic `claude -p` / `opencode run` session that runs UNATTENDED, so
# an unbounded run (the old default=None) let a wedged or runaway session burn
# tokens with nothing to stop it. The supervisor's dos.toml worker_launch_template
# spawns this leaf with no --timeout-s, so this default is the only bound on that
# production path (the watchdog canary wraps its own 120s). 30 min is generous for
# a real lane/ticket yet bounds a runaway; the 120-min issue cooldown retries a
# hard target later. Opt out with `--timeout-s 0` (normalized to None below).
DEFAULT_TIMEOUT_S = 1800

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


def normalize_timeout(value: int | None) -> int | None:
    """Map a CLI ``--timeout-s`` value to the launch timeout.

    A positive value is the wall-clock cap; ``0``/negative/``None`` is the
    explicit unbounded opt-out (``None`` -> ``subprocess.run`` waits forever).
    """
    if value and value > 0:
        return value
    return None


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


def _which_on_exact_path(name: str, path_value: str | None) -> str | None:
    """Resolve ``name`` using exactly ``path_value``.

    Windows' `shutil.which(name, path=...)` still applies current-directory
    search semantics, so a repo-root `fak.exe` can leak into a test or worker env
    that deliberately supplied a PATH excluding it. This helper scans only the
    explicit PATH directories while still honoring PATHEXT for command shims.
    """
    if path_value is None:
        return shutil.which(name)
    suffixes = [""]
    if os.name == "nt" and not Path(name).suffix:
        pathext = os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD")
        suffixes.extend(ext.lower() for ext in pathext.split(os.pathsep) if ext)
        suffixes.extend(ext.upper() for ext in pathext.split(os.pathsep) if ext)
    for raw_dir in path_value.split(os.pathsep):
        if not raw_dir:
            continue
        for suffix in suffixes:
            candidate = Path(raw_dir) / f"{name}{suffix}"
            if candidate.is_file():
                return str(candidate)
    return None


# --- Dogfood: front each worker with the kernel (``fak guard``) ----------------
# A dispatch worker IS the highest-volume dev work on a fleet node, and by default
# it talked STRAIGHT to the provider API -- the kernel adjudicated NONE of it. That
# is the inverse of dogfooding the product. Fronting the worker with ``fak guard``
# puts the SAME kernel ``fak serve`` runs in front of every tool call the worker
# proposes (deny by structure, repair malformed args, quarantine poisoned results),
# and records every verdict in a durable, hash-chained DECISION JOURNAL -- so the
# fleet eats the product on the real workflow, with a witness. Default ON; opt out
# per node with ``FLEET_DOGFOOD_GUARD=0``. resolve_fak_bin already fails OPEN to an
# unwrapped worker on a host that has not built ``fak``, so the default never breaks
# dispatch.
GUARD_OFF_VALUES = frozenset({"0", "off", "false", "no", "", "disable", "disabled"})

# fak guard fronts the real provider API in passthrough, and a Claude Code turn on a
# frontier model (with extended thinking) can run well past ``fak serve``'s default
# 60s planner / 90s write timeouts -- which would TRUNCATE the turn at the gateway.
# Raise both floors for a guarded worker (mirrors scripts/dogfood-claude.*, which
# pre-raise them) without clobbering an operator's explicit value.
GUARD_TIMEOUT_FLOOR_S = 600


def guard_enabled(env: dict[str, str] | None = None) -> bool:
    """Whether to front a worker with ``fak guard``. Dogfood-by-default (ON); a node
    opts out with ``FLEET_DOGFOOD_GUARD`` in {0,off,false,no,disable}."""
    raw = (env if env is not None else os.environ).get("FLEET_DOGFOOD_GUARD")
    if raw is None:
        return True
    return raw.strip().lower() not in GUARD_OFF_VALUES


def resolve_fak_bin(workspace: Path, env: dict[str, str] | None = None) -> str | None:
    """Locate a ``fak`` binary to front the worker with, or ``None``.

    Precedence: ``$FAK_BIN`` (if it exists) -> the in-tree ``tools/.bin/fak[.exe]``
    the dogfood launcher builds -> ``fak`` on PATH. ``None`` means the caller should
    fail OPEN (launch the worker unwrapped) rather than break dispatch on a host that
    has not built fak.
    """
    e = env if env is not None else os.environ
    explicit = (e.get("FAK_BIN") or "").strip()
    if explicit and Path(explicit).exists():
        return explicit
    exe = "fak.exe" if os.name == "nt" else "fak"
    intree = Path(workspace) / "tools" / ".bin" / exe
    if intree.exists():
        return str(intree)
    # Honor the supplied env's PATH for the lookup (so the env param fully governs
    # resolution); an absent PATH key falls back to the process PATH.
    return _which_on_exact_path("fak", e.get("PATH"))


def guard_provider(backend: str) -> str:
    """The upstream wire ``fak guard`` proxies for a worker backend. ``claude`` ->
    the Anthropic API (passthrough/subscription); every other backend is OpenAI-wire."""
    return "anthropic" if backend == "claude" else "openai"


def guard_audit_path(workspace: Path, lane: str, backend: str) -> Path:
    """A PER-SESSION durable decision journal under the gitignored ``.dispatch-runs/``.

    The filename is keyed on the lane+backend (for separability and globbing) PLUS a
    per-process discriminator (pid + a uuid). This is deliberate: ``fak guard``'s
    hash-chained journal has NO inter-process lock, so two concurrent workers sharing
    ONE file would each start an independent sha256 chain and braid them into a forked,
    unverifiable journal (and interleave mid-row). Two close dispatch ticks CAN pick the
    same lane (``pick_lane`` returns the busiest lane before any issue resolves), so a
    per-lane-only path would force exactly that collision. A unique-per-session file lets
    each ``fak guard`` own its own valid chain; ``fak audit verify`` / the coverage
    scorecard glob the lane prefix to aggregate them. The dir is created lazily by the
    journal writer (``journal.Enable`` mkdirs it)."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in f"{lane}-{backend}")
    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    return Path(workspace) / ".dispatch-runs" / "guard-audit" / f"{safe}-{token}.jsonl"


def guard_wrap(
    command: Sequence[str],
    *,
    fak_bin: str | None,
    lane: str,
    backend: str,
    workspace: Path,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Front a raw worker argv with ``fak guard -- <worker>`` so the kernel
    adjudicates every tool call. Pure given ``fak_bin``. Returns the command
    UNCHANGED when:

    * ``fak_bin`` is ``None`` (no binary resolved -> fail open), or
    * the backend fronts a LOCAL upstream we have not been told the base URL of.
      ``claude`` proxies the public Anthropic API (passthrough/subscription) with no
      base-URL override; ``opencode`` (and friends) front a local server (e.g. a GLM
      endpoint), so guard would MISROUTE them to the provider's public API unless
      ``FLEET_DOGFOOD_GUARD_BASEURL`` names that upstream. We refuse to misroute.
    """
    cmd = list(command)
    if not cmd or not fak_bin:
        return cmd
    provider = guard_provider(backend)
    extra: list[str] = []
    if backend != "claude":
        e = env if env is not None else os.environ
        base = (e.get("FLEET_DOGFOOD_GUARD_BASEURL") or "").strip()
        if not base:
            return cmd  # don't misroute a local-upstream worker
        extra = ["--base-url", base]
    audit = guard_audit_path(workspace, lane, backend)
    return [fak_bin, "guard", "--provider", provider, *extra,
            "--audit", str(audit), "--", *cmd]


def guard_env_augment(env: dict[str, str]) -> dict[str, str]:
    """Ensure a guarded worker's gateway won't truncate a long frontier turn: set
    ``FAK_PLANNER_TIMEOUT_S`` / ``FAK_HTTP_WRITE_TIMEOUT_S`` to a generous floor when
    unset (an explicit operator value is left as-is). Mutates and returns ``env``."""
    for key in ("FAK_PLANNER_TIMEOUT_S", "FAK_HTTP_WRITE_TIMEOUT_S"):
        if not env.get(key):
            env[key] = str(GUARD_TIMEOUT_FLOOR_S)
    return env


def guarded_launch_command(
    command: Sequence[str], lane: str, backend: str, workspace: Path,
    env: dict[str, str] | None = None,
) -> tuple[list[str], bool]:
    """Resolve the argv to actually launch: ``command`` fronted by ``fak guard`` when
    dogfood mode is on and a fak binary resolves, else ``command`` unchanged. Returns
    ``(launch_command, guarded)`` so callers can both run it and report what ran."""
    e = env if env is not None else os.environ
    fak_bin = resolve_fak_bin(workspace, e) if guard_enabled(e) else None
    if not fak_bin:
        return list(command), False
    wrapped = guard_wrap(command, fak_bin=fak_bin, lane=lane, backend=backend,
                         workspace=workspace, env=e)
    return wrapped, wrapped != list(command)


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
    # Spawn the worker as its OWN process group (Windows) / session (POSIX) so a
    # timeout can kill the WHOLE tree, not just the immediate child. With guard on,
    # the immediate child is ``fak guard`` and the agent (``claude``) is a GRANDCHILD;
    # subprocess.run's timeout would SIGKILL only fak guard and orphan the agent --
    # leaving it running past the wall-clock cap with the kernel gateway already torn
    # down. New-group + a tree-kill on timeout closes that hole (and is strictly safer
    # for the un-guarded path too).
    popen_kwargs: dict[str, Any] = {"cwd": str(cwd), "env": env}
    if os.name == "nt":
        # CREATE_NEW_PROCESS_GROUP (tree-killable on timeout) OR'd with CREATE_NO_WINDOW
        # so this synchronously-spawned worker — and every git/gh/fak tool it spawns —
        # inherits one HIDDEN console instead of each popping a visible window when we run
        # windowless (pythonw) from a scheduled dispatch tick. Inherited stdio still flows
        # to the parent (CREATE_NO_WINDOW suppresses the window, not the handles).
        popen_kwargs["creationflags"] = 0x00000200 | _CREATE_NO_WINDOW
    else:
        popen_kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(resolved, **popen_kwargs)
    except FileNotFoundError as exc:
        return {"returncode": 127, "error": str(exc), "stdout": "", "stderr": str(exc)}
    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        terminate_tree(proc)
        return {"returncode": 124, "timeout": True, "stdout": "", "stderr": "timeout"}
    return {"returncode": rc, "stdout": "", "stderr": ""}


def terminate_tree(proc: "subprocess.Popen[Any]") -> None:
    """Kill a worker AND its descendants (``fak guard`` + the agent it wraps). On a
    timeout, killing only the immediate child would orphan the grandchild agent with
    the kernel gateway already gone -- the exact runaway the wall-clock cap exists to
    prevent. On Windows ``taskkill /T`` walks the PID tree; on POSIX the worker is a
    session/group leader (``start_new_session``) so killpg reaps the group."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=30,
                           creationflags=no_window_creationflags())
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ValueError, subprocess.SubprocessError):
        # Fall back to a single-process kill if the group/tree kill is unavailable.
        try:
            proc.kill()
        except OSError:
            pass
    finally:
        try:
            proc.wait(timeout=10)
        except (subprocess.TimeoutExpired, OSError):
            pass


def build_payload(
    *,
    lane: str,
    backend: str,
    workspace: Path,
    dry_run: bool,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    command: Sequence[str] | None = None,
    guarded: bool = False,
) -> dict[str, Any]:
    # ``command`` defaults to the raw (unguarded) worker argv for backward compat; a
    # live/dry-run launch passes the actual launched argv (kernel-fronted when guarded)
    # so the record shows exactly what ran.
    if command is None:
        command = build_command(lane, backend) if not error else []
    command = list(command)
    ok = error is None and (result is None or result.get("returncode") == 0)
    return {
        "schema": SCHEMA,
        "ok": ok,
        "lane": lane,
        "backend": backend,
        "guarded": guarded,
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
        f"dispatch-worker: backend={payload.get('backend')} lane={payload.get('lane')} "
        f"guarded={payload.get('guarded')} dry_run={payload.get('dry_run')}",
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
    ap.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S,
                    help=f"child wall-clock timeout in seconds (default: {DEFAULT_TIMEOUT_S}; "
                         "use 0 for unbounded)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    error: str | None = None
    backend = DEFAULT_BACKEND
    try:
        backend = resolve_backend(args.backend, None)
    except ValueError as exc:
        error = str(exc)

    # Resolve the argv to actually launch, fronting it with ``fak guard`` when dogfood
    # mode is on and a fak binary resolves (fail OPEN to an unwrapped worker otherwise).
    # Computed for BOTH paths so ``--dry-run`` reveals the kernel-fronted argv an
    # operator will actually run.
    command: list[str] = []
    guarded = False
    if not error:
        command, guarded = guarded_launch_command(
            build_command(args.lane, backend), args.lane, backend, workspace
        )

    if args.dry_run or error:
        payload = build_payload(
            lane=args.lane, backend=backend, workspace=workspace, dry_run=True,
            error=error, command=command, guarded=guarded,
        )
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(render(payload))
        return 0 if not error else 2

    env = child_env(args.lane, backend, workspace)
    if guarded:
        guard_env_augment(env)
    result = launch(command, workspace, env, timeout_s=normalize_timeout(args.timeout_s))
    payload = build_payload(
        lane=args.lane, backend=backend, workspace=workspace, dry_run=False,
        result=result, command=command, guarded=guarded,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    return int(result.get("returncode") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
