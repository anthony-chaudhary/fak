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
import re
import shutil
import signal
import socket
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


def install_no_window_subprocess_defaults(module: Any = subprocess) -> None:
    """Default subprocess helpers to CREATE_NO_WINDOW on Windows.

    Legacy automation tools have many thin git/gh/powershell probes. Installing
    this once per script keeps those background helpers from flashing consoles
    without rewriting every subprocess.run call site. Explicit creationflags still
    win, so a caller that needs a different process group can opt out locally.
    """
    if os.name != "nt" or getattr(module, "_fak_no_window_defaults", False):
        return

    def with_flags(fn: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("creationflags", no_window_creationflags())
            return fn(*args, **kwargs)

        return wrapped

    module.run = with_flags(module.run)
    module.Popen = with_flags(module.Popen)
    module.call = with_flags(module.call)
    module.check_call = with_flags(module.check_call)
    module.check_output = with_flags(module.check_output)
    module._fak_no_window_defaults = True

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
# Invoke the BARE project-skill form (`/dos-dispatch-loop`), not the namespaced
# plugin form (`/dos-kernel:dos-dispatch-loop`). The skill is git-tracked at
# `.claude/skills/dos-dispatch-loop/SKILL.md`, so a worker launched from the repo
# root sees it under EVERY switched account dir. The plugin form fails closed
# ("Unknown command") whenever a per-account `.claude-<acct>` plugin cache is
# missing/empty — which it is for freshly-enrolled worker accounts — making the
# spawned worker exit 0 with zero work done. The project skill has no such
# per-account dependency.
CLAUDE_AGENT_PROMPT = "/dos-dispatch-loop --lane {lane}"
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
        # --print-logs surfaces opencode's run-level failures (quota wall, unreachable
        # model) — which `run` routes to its logger on STDERR, not stdout — into the
        # worker log instead of leaving a silent banner-only no-op (#1275). Mirrors the
        # detached spawn arm in issue_resolve_dispatch.build_worker_command.
        return [
            "opencode",
            "run",
            "--print-logs",
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
    # Self-describing shared front door (#1501): when the wave routes this worker's fak
    # self-query through one shared gateway, echo it so the worker knows its shared path
    # regardless of prompt rendering (same contract as DISPATCH_LANE).
    if shared := shared_gateway_url(env):
        env["DISPATCH_SHARED_GATEWAY"] = shared
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
OPENCODE_DEFAULT_PROVIDER_ID = "zai-coding-plan"
OPENCODE_GUARD_BASE_URL_ENV = "FLEET_DOGFOOD_GUARD_BASEURL"
OPENCODE_GUARD_UPSTREAM_KEY_ENV = "FAK_OPENCODE_GUARD_UPSTREAM_API_KEY"
# Headless Claude workers should not merely observe the PreCompact hook posture.
# The live fleet failure class was `compact=fired` while issue-resolution workers
# continued tens of thousands of tokens past the compact threshold; interactive
# `fak guard` can keep its shadow default, but unattended dispatch needs the
# actuator enforced at launch.
CLAUDE_GUARD_PRECOMPACT_ARGS = ["--precompact-hook", "enforce"]
# Pair the posture actuator with guard's hard budget restart path. The finite
# restart limit prevents a bad issue prompt from creating an unbounded relaunch loop.
#
# This seeds guard's per-session ContextTokensLeft, which DebitUsage draws down by
# each turn's FULL resident window (prompt+cache_read+cache_creation); at <=0 the
# session drains BUDGET_CONTEXT_EXHAUSTED. The old 48k was picked to mirror
# gateway.DefaultCompactHistoryBudget — but that is a compaction TARGET (what history
# is squeezed toward), not a max resident window. The workers' ~62K irreducible
# baseline prompt can't compact below itself, so a 48k drain ceiling meant every
# worker was born over-budget → 2 restarts burned → 409 → exit 1 → CHILD_CRASH.
#
# DONE(dynamic-budget): no longer a flat constant (a flat value silently falls below
# the baseline the next time the baseline grows). The budget is DERIVED: baseline ×
# birth-headroom, clamped to the model window's effective ceiling (window − output
# reserve). Grows with the baseline (never a birth wall again), shrinks with the
# window (always a runaway backstop under the real model window).
#
# The window/reserve constants MIRROR the Go source of truth in
# internal/ctxplan/envelope.go (GenericTurnEnvelope: HardContextCap 200000,
# OutputReserve 32000; doctrine: docs/long-context-defaults.md — the advertised
# window is a hard CAP, never a raw target); the baseline and headroom factor mirror
# cmd/dispatchworker/guard.go (claudeGuardBaselineTokens /
# claudeGuardBirthHeadroomFactor). Python cannot import the Go package, so keep ALL
# FOUR constants and the arithmetic in sync with Go by hand — the derived integer
# must be identical on both paths.
CLAUDE_GUARD_MODEL_WINDOW_TOKENS = 200000   # ctxplan HardContextCap
CLAUDE_GUARD_OUTPUT_RESERVE_TOKENS = 32000  # ctxplan OutputReserve
# The FLOOR under the measured launch-prompt baseline — the workers' ~62K irreducible
# prompt (issue body + AGENTS/llms orientation + injected fleet memory + the ~40K
# startup.json 'route' blob), hand-measured once. No longer the SOLE baseline:
# measured_claude_guard_baseline sizes the constituents readable at launch and floors
# them here, so a degenerate/partial measurement can never lower the budget below the
# shipped value while an organically grown prompt raises it automatically. Mirrors
# cmd/dispatchworker/guard.go:claudeGuardBaselineTokens.
CLAUDE_GUARD_BASELINE_TOKENS = 62000
# Budget = baseline × this factor, so a worker is born with a whole baseline of
# headroom before the backstop fires. (History: the committed value this derivation
# replaced was the crash-looping flat "48000"; a flat 131072 was only ever an
# uncommitted working-tree interim, never shipped.)
CLAUDE_GUARD_BIRTH_HEADROOM_FACTOR = 2
# Launch-prompt constituents a self-claiming lane worker can SIZE at launch: the
# orientation files every worker loads (AGENTS.md, llms.txt, CLAUDE.md), plus a
# workspace-root MEMORY.md when a repo keeps one. NOTE the real injected fleet memory
# does NOT live at the workspace root — it is in the per-project claude memory dir
# (.../projects/<ws>/memory/MEMORY.md), off the root and not portably derivable here,
# so in the common fleet layout MEMORY.md is absent and the floor covers it; it is
# listed so a repo that DOES keep a root MEMORY.md gets it measured. Likewise the
# per-issue body and the ~40K startup.json 'route' blob are not visible here (the floor
# covers that remainder); the startup blob is measured when the launcher names it via
# DISPATCH_STARTUP_BUNDLE. Mirrors
# cmd/dispatchworker/guard.go:launchConstituentFiles / launchStartupBundleEnv.
LAUNCH_CONSTITUENT_FILES = ["AGENTS.md", "llms.txt", "CLAUDE.md", "MEMORY.md"]
LAUNCH_STARTUP_BUNDLE_ENV = "DISPATCH_STARTUP_BUNDLE"


def approx_tokens_from_bytes(n: int) -> int:
    """Estimate tokens from a byte count via the codebase (n+3)//4 ruler — the SAME
    scale the ctxplan planner sizes context with. Mirrors
    cmd/dispatchworker/guard.go:approxTokensFromBytes."""
    return (n + 3) // 4 if n > 0 else 0


def measure_launch_baseline_tokens(constituent_bytes: dict[str, int]) -> int:
    """Sum the approximate token footprint of the launch constituents from their real
    byte sizes — the measurement seam that replaces the frozen 62000 guess. An empty
    map measures 0 (degenerate); the caller floors it. Mirrors
    cmd/dispatchworker/guard.go:measureLaunchBaselineTokens."""
    return sum(approx_tokens_from_bytes(b) for b in constituent_bytes.values())


def resolve_claude_guard_baseline(measured_tokens: int) -> int:
    """max(measured, floor): a degenerate/partial measurement can never lower the
    budget below the shipped value (the #2972 born-over-budget invariant); a prompt
    grown past the floor raises the baseline automatically. Mirrors
    cmd/dispatchworker/guard.go:resolveClaudeGuardBaseline."""
    return measured_tokens if measured_tokens > CLAUDE_GUARD_BASELINE_TOKENS \
        else CLAUDE_GUARD_BASELINE_TOKENS


def gather_launch_constituent_bytes(
    workspace, env: dict[str, str] | None = None,
) -> dict[str, int]:
    """Read the byte sizes of the launch constituents readable at launch from
    ``workspace`` (an unreadable/absent file contributes nothing — the degenerate
    guard). An empty workspace measures nothing (the hermetic default → the floor).
    Mirrors cmd/dispatchworker/guard.go:gatherLaunchConstituentBytes."""
    out: dict[str, int] = {}
    ws = str(workspace) if workspace else ""
    if not ws:
        return out
    e = env if env is not None else os.environ
    for name in LAUNCH_CONSTITUENT_FILES:
        try:
            p = Path(ws) / name
            if p.is_file():
                out[name] = p.stat().st_size
        except OSError:
            pass
    bundle = (e.get(LAUNCH_STARTUP_BUNDLE_ENV) or "").strip()
    if bundle:
        try:
            bp = Path(bundle)
            if bp.is_file():
                out["startup_bundle"] = bp.stat().st_size
        except OSError:
            pass
    return out


def measured_claude_guard_baseline(
    workspace=None, env: dict[str, str] | None = None,
) -> tuple[int, dict[str, int]]:
    """Size the readable launch constituents, floor the measurement, and return both
    the floored baseline and the raw constituent sizes (for the observable). Mirrors
    cmd/dispatchworker/guard.go:measuredClaudeGuardBaseline."""
    constituents = gather_launch_constituent_bytes(workspace, env)
    return resolve_claude_guard_baseline(
        measure_launch_baseline_tokens(constituents)), constituents


def claude_guard_context_budget_tokens(
    workspace=None, env: dict[str, str] | None = None,
) -> int:
    """Derive the per-session context budget: MEASURED baseline (floored) x headroom,
    window-clamped. An empty workspace measures nothing and falls to the floor, so the
    hermetic default is min(62000*2, 200000-32000) = 124000. Mirrors
    cmd/dispatchworker/guard.go:claudeGuardContextBudgetTokens.
    """
    ceiling = CLAUDE_GUARD_MODEL_WINDOW_TOKENS - CLAUDE_GUARD_OUTPUT_RESERVE_TOKENS
    baseline, _ = measured_claude_guard_baseline(workspace, env)
    return min(baseline * CLAUDE_GUARD_BIRTH_HEADROOM_FACTOR, ceiling)


CLAUDE_GUARD_CONTEXT_BUDGET_TOKENS = str(claude_guard_context_budget_tokens())
# The budget drains in ~2 turns per epoch, so a relaunch happens every ~2 min. The old
# "2" killed a healthy worker after ~4-5 min (reset_limit limit=2 -> 409 -> CHILD_CRASH)
# at ~15% of its runway. 16 lets a healthy-but-slow worker reach its 30-min wall-clock
# backstop (DEFAULT_TIMEOUT_S hard-kill + dispatcher reap) while a degenerate sub-2-min
# reset storm still trips here. Mirrors cmd/dispatchworker/guard.go:claudeGuardRestartLimit.
CLAUDE_GUARD_RESTART_LIMIT = "16"
# In-guard wall-clock budget, backed off DEFAULT_TIMEOUT_S so the guard drains GRACEFULLY
# (TIME_BUDGET_EXHAUSTED + audit flush) a minute before launch()'s hard-kill at exit 124.
# --max-duration is CUMULATIVE across --restart-on-budget relaunches, so it bounds TOTAL
# lifespan regardless of restart count. Mirrors cmd/dispatchworker/guard.go:claudeGuardMaxDuration.
CLAUDE_GUARD_MAX_DURATION = f"{DEFAULT_TIMEOUT_S - 60}s"
# The opencode path takes NO --restart-on-budget (its budget is a single run), so this is
# a plain wall-clock envelope: the guarded opencode worker SELF-terminates at the cap even
# when no external reaper runs. That matters for the detached docs-lane spawn path
# (dispatch_glm_docs.py), which does not sit inside the main dispatcher's reap loop -- an
# opencode worker retry-looping against a DOWN gateway would otherwise burn a slot and its
# host threads for its whole (unbounded) lifespan. Backed off DEFAULT_TIMEOUT_S so the
# in-guard drain still precedes launch()'s hard-kill on the synchronous path.
OPENCODE_GUARD_MAX_DURATION = f"{DEFAULT_TIMEOUT_S - 60}s"


def claude_guard_budget_args(
    workspace=None, env: dict[str, str] | None = None,
) -> list[str]:
    """The claude guard budget argv, carrying the MEASURED context budget for this
    workspace (floored). Order is part of the CLI surface the parity tests pin.
    Mirrors cmd/dispatchworker/guard.go:claudeGuardArgs (the budget portion)."""
    return [
        "--context-budget-tokens",
        str(claude_guard_context_budget_tokens(workspace, env)),
        "--restart-on-budget",
        "--restart-limit", CLAUDE_GUARD_RESTART_LIMIT,
        "--max-duration", CLAUDE_GUARD_MAX_DURATION,
    ]


# Floor-default budget argv (workspace-independent): the shipped 124000 wiring, kept
# as a stable reference. The live path calls claude_guard_budget_args(workspace, env).
CLAUDE_GUARD_BUDGET_ARGS = claude_guard_budget_args()


def _opencode_model_provider(command: Sequence[str]) -> str:
    """Return the provider id from an opencode ``-m provider/model`` argv.

    The issue resolver pins opencode to ``zai-coding-plan/glm-5.2``. The generic
    dispatch worker may rely on the account default instead; in that case keep the
    deployed GLM provider id as the override target.
    """
    for idx, token in enumerate(command):
        if token in ("-m", "--model") and idx + 1 < len(command):
            model = str(command[idx + 1]).strip()
            if "/" in model:
                provider, _ = model.split("/", 1)
                if provider.strip():
                    return provider.strip()
    return OPENCODE_DEFAULT_PROVIDER_ID


def _deep_merge_config(base: Any, overlay: dict[str, Any]) -> dict[str, Any]:
    """Small JSON-object merge for OPENCODE_CONFIG_CONTENT overlays."""
    if not isinstance(base, dict):
        base = {}
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _opencode_guard_addr(env: dict[str, str]) -> str:
    """Loopback addr for the per-session guard gateway.

    Operators/tests can pin it. Otherwise ask the OS for a free port and pass the
    same addr both to ``fak guard --addr`` and OpenCode's inline provider config.
    """
    pinned = (env.get("FLEET_DOGFOOD_GUARD_ADDR") or "").strip()
    if pinned:
        return pinned
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
    return f"{host}:{port}"


def opencode_guard_config_content(command: Sequence[str], gateway_base_url: str,
                                  existing: str = "") -> str:
    """Inline OpenCode config that repoints the selected provider to fak guard.

    OpenCode's named providers do not necessarily honor ``OPENAI_BASE_URL``.
    ``OPENCODE_CONFIG_CONTENT`` is loaded after project config, so this keeps the
    override per-child and avoids writing credentials or transient ports to the repo.
    """
    provider = _opencode_model_provider(command)
    overlay = {
        "provider": {
            provider: {
                "options": {
                    "baseURL": gateway_base_url,
                },
            },
        },
    }
    base: Any = {}
    if existing.strip():
        try:
            base = json.loads(existing)
        except json.JSONDecodeError:
            base = {}
    return json.dumps(_deep_merge_config(base, overlay), separators=(",", ":"))


def _opencode_config_candidates(env: dict[str, str]) -> list[Path]:
    home = Path.home()
    root = Path(env.get("XDG_CONFIG_HOME") or os.environ.get("XDG_CONFIG_HOME")
                or home / ".config")
    candidates: list[Path] = []
    explicit = (env.get("OPENCODE_CONFIG") or os.environ.get("OPENCODE_CONFIG") or "").strip()
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([
        root / "opencode" / "opencode.json",
        root / "opencode" / "opencode.jsonc",
        root / "opencode.json",
        root / "opencode.jsonc",
    ])
    seen: set[str] = set()
    out: list[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def _env_substituted_value(value: Any, env: dict[str, str]) -> str:
    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    if stripped.startswith("{env:") and stripped.endswith("}"):
        return env.get(stripped[5:-1], os.environ.get(stripped[5:-1], "")).strip()
    return stripped


def opencode_upstream_api_key(command: Sequence[str], env: dict[str, str]) -> str:
    """Best-effort read of the opencode provider key for guard's upstream hop."""
    explicit = _env_substituted_value(env.get(OPENCODE_GUARD_UPSTREAM_KEY_ENV), env)
    if explicit:
        return explicit
    provider = _opencode_model_provider(command)
    if content := (env.get("OPENCODE_CONFIG_CONTENT") or "").strip():
        try:
            cfg = json.loads(content)
        except json.JSONDecodeError:
            cfg = {}
        key = (((cfg.get("provider") or {}).get(provider) or {})
               .get("options") or {}).get("apiKey")
        resolved = _env_substituted_value(key, env)
        if resolved:
            return resolved
    for path in _opencode_config_candidates(env):
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        key = (((cfg.get("provider") or {}).get(provider) or {})
               .get("options") or {}).get("apiKey")
        resolved = _env_substituted_value(key, env)
        if resolved:
            return resolved
    return ""


def opencode_upstream_base_url(command: Sequence[str], env: dict[str, str]) -> str:
    """Best-effort read of the selected opencode provider's upstream base URL."""
    provider = _opencode_model_provider(command)
    if content := (env.get("OPENCODE_CONFIG_CONTENT") or "").strip():
        try:
            cfg = json.loads(content)
        except json.JSONDecodeError:
            cfg = {}
        base = (((cfg.get("provider") or {}).get(provider) or {})
                .get("options") or {}).get("baseURL")
        resolved = _env_substituted_value(base, env)
        if resolved:
            return resolved
    for path in _opencode_config_candidates(env):
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        base = (((cfg.get("provider") or {}).get(provider) or {})
                .get("options") or {}).get("baseURL")
        resolved = _env_substituted_value(base, env)
        if resolved:
            return resolved
    return ""


def _env_key_suffix(value: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()
    return suffix or "DEFAULT"


def opencode_guard_base_url(command: Sequence[str], env: dict[str, str]) -> str:
    """Resolve the upstream URL that ``fak guard`` should proxy for opencode.

    Precedence is intentionally provider-scoped:

    1. ``FLEET_DOGFOOD_GUARD_BASEURL_<PROVIDER>`` names an explicit local/DGX
       or lab endpoint for the selected provider, e.g. ``..._DEEPSEEK_AI``.
    2. The selected provider's opencode account config ``options.baseURL``.
    3. Legacy ``FLEET_DOGFOOD_GUARD_BASEURL`` only for the default GLM provider.

    The legacy global variable is not allowed to hijack non-default providers.
    That was the NIM DeepSeek failure mode: a process-wide GLM URL silently
    caught a worker pinned to ``deepseek-ai/deepseek-v4-pro``.
    """
    provider_id = _opencode_model_provider(command)
    provider_env = f"{OPENCODE_GUARD_BASE_URL_ENV}_{_env_key_suffix(provider_id)}"
    provider_base = (env.get(provider_env) or "").strip()
    if provider_base:
        return provider_base

    configured_base = opencode_upstream_base_url(command, env)
    if configured_base:
        return configured_base

    if provider_id == OPENCODE_DEFAULT_PROVIDER_ID:
        return (env.get(OPENCODE_GUARD_BASE_URL_ENV) or "").strip()
    return ""


def guard_enabled(env: dict[str, str] | None = None) -> bool:
    """Whether to front a worker with ``fak guard``. Dogfood-by-default (ON); a node
    opts out with ``FLEET_DOGFOOD_GUARD`` in {0,off,false,no,disable}."""
    raw = (env if env is not None else os.environ).get("FLEET_DOGFOOD_GUARD")
    if raw is None:
        return True
    return raw.strip().lower() not in GUARD_OFF_VALUES


def node_caps(env: dict[str, str] | None = None) -> frozenset[str]:
    """The hardware capabilities THIS node advertises for issue dispatch.

    Read from ``FLEET_NODE_CAPS`` (comma/space separated, case-folded), e.g.
    ``FLEET_NODE_CAPS=gpu``. **Default empty ⇒ GPU-less**: an unconfigured host
    advertises no special capability, so the dispatcher's issue-level capability
    gate (see ``issue_resolve_dispatch.evaluate``) leaves GPU-tagged issues open
    and visible for a node that opts in with ``gpu``. Mirrors the established
    per-node env knob pattern (``FLEET_WORKER_BACKEND``/``FLEET_DOGFOOD_GUARD``).
    """
    raw = (env if env is not None else os.environ).get("FLEET_NODE_CAPS") or ""
    caps = {tok.lower() for tok in re.split(r"[,\s]+", raw) if tok.strip()}
    return frozenset(caps)


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


# --- Shared-path leasing: one gateway front door for a fanned-out wave (#1501) -----
# A fanned-out wave (`fak dispatch wave` / the ultracode Workflow tool) spawns N workers;
# by default each is fronted by its OWN per-session `fak guard` gateway, so they share NO
# L3 cache, NO cross-agent change feed (`fak_changes`, internal/gateway/coherence.go is
# per-Server state) and NO self-index -- each re-discovers the same tree, the opposite of
# fak's "do the shared work once" thesis. When the wave stands up ONE shared `fak serve`
# front door and names it in FLEET_SHARED_GATEWAY_URL, every wave worker's fak self-query
# + effects are repointed at that single gateway, so the shared work is done once and a
# peer's write reaches the others over `fak_changes`. Disjoint lease regions per agent
# are already arbitrated by `fak dispatch wave` (dispatchorder + dos_arbitrate, the
# COLLISION_RISK floor). Opt-in and backward-compatible: unset -> each worker keeps its
# private gateway, argv unchanged.
SHARED_GATEWAY_URL_ENV = "FLEET_SHARED_GATEWAY_URL"


def shared_gateway_url(env: dict[str, str] | None = None) -> str:
    """The wave's one shared `fak serve` HTTP front door, or "" when a worker keeps its
    own per-session gateway. Read from FLEET_SHARED_GATEWAY_URL; trailing slash trimmed."""
    raw = (env if env is not None else os.environ).get(SHARED_GATEWAY_URL_ENV) or ""
    return raw.strip().rstrip("/")


def shared_fak_mcp_url(gateway_url: str) -> str:
    """The gateway's MCP endpoint: the bare origin plus ``/mcp`` (mirrors
    guardMCPURLFromGatewayBase in cmd/fak/guard_mcp.go)."""
    return gateway_url.strip().rstrip("/") + "/mcp"


def shared_fak_mcp_config(gateway_url: str) -> dict[str, Any]:
    """One-server MCP client config naming "fak" as the shared remote HTTP MCP server --
    the SAME ``{type:http,url}`` shape guard writes for its private gateway
    (writeGuardMCPConfig), pointed instead at the wave's shared front door."""
    return {"mcpServers": {"fak": {"type": "http", "url": shared_fak_mcp_url(gateway_url)}}}


def shared_mcp_config_path(workspace: Path, lane: str, backend: str) -> Path:
    """A per-session file under gitignored ``.dispatch-runs/`` for the shared-gateway MCP
    config. Keyed per-process (pid + uuid) so two workers on one lane never clash on the
    same file; mirrors guard_audit_path's discriminator."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in f"{lane}-{backend}")
    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    return Path(workspace) / ".dispatch-runs" / "shared-mcp" / f"{safe}-{token}.json"


def write_shared_fak_mcp_config(path: Path, gateway_url: str) -> Path:
    """Write the shared-gateway "fak" MCP client config to ``path`` and return it. The
    dir is created lazily, matching the guard journal writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(shared_fak_mcp_config(gateway_url), indent=2) + "\n",
                    encoding="utf-8")
    return path


def insert_claude_mcp_config(command: Sequence[str], config_path: str) -> list[str]:
    """Insert ``--mcp-config <path>`` immediately after the claude executable (Claude
    Code's global flags precede the subcommand/user args), mirroring
    appendClaudeMCPConfigArg in cmd/fak/guard_mcp.go."""
    cmd = list(command)
    if not cmd:
        return cmd
    return [cmd[0], "--mcp-config", config_path, *cmd[1:]]


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
      base-URL override; ``opencode`` resolves the selected provider's upstream
      through :func:`opencode_guard_base_url`, so one GLM/DGX override cannot
      silently catch a different provider such as NVIDIA NIM DeepSeek.
    """
    cmd = list(command)
    if not cmd or not fak_bin:
        return cmd
    provider = guard_provider(backend)
    audit = guard_audit_path(workspace, lane, backend)
    extra: list[str] = []
    if backend == "claude":
        extra = [
            *extra,
            *CLAUDE_GUARD_PRECOMPACT_ARGS,
            "--session-id", audit.stem,
            *claude_guard_budget_args(workspace, env),
        ]
        # Shared-path leasing for a fanned-out wave (#1501): when the wave names ONE
        # shared fak gateway (FLEET_SHARED_GATEWAY_URL), route THIS worker's fak
        # self-query + effects through it instead of guard's private per-session
        # gateway. We disable guard's own per-session MCP registration
        # (--mcp-register=false) and register "fak" at the shared serve's /mcp endpoint
        # via Claude Code's --mcp-config -- so every wave agent shares one L3 cache, one
        # cross-agent change feed (fak_changes) and one self-index. Unset -> unchanged.
        shared = shared_gateway_url(env if env is not None else os.environ)
        if shared:
            cfg_path = write_shared_fak_mcp_config(
                shared_mcp_config_path(workspace, lane, backend), shared)
            extra = [*extra, "--mcp-register=false"]
            cmd = insert_claude_mcp_config(cmd, str(cfg_path))
    if backend != "claude":
        e = env if env is not None else os.environ
        base = opencode_guard_base_url(cmd, e) if backend == "opencode" else (
            e.get(OPENCODE_GUARD_BASE_URL_ENV) or "").strip()
        if not base:
            return cmd  # don't misroute a local-upstream worker
        # Every guarded non-claude worker gets a hard wall-clock cap so it self-terminates
        # even off the main dispatcher's reap loop (see OPENCODE_GUARD_MAX_DURATION).
        extra = ["--base-url", base, "--max-duration", OPENCODE_GUARD_MAX_DURATION]
        if backend == "opencode":
            addr = _opencode_guard_addr(e)
            extra = ["--addr", addr, *extra]
            e["OPENCODE_CONFIG_CONTENT"] = opencode_guard_config_content(
                cmd, f"http://{addr}/v1",
                existing=e.get("OPENCODE_CONFIG_CONTENT", ""))
            upstream_key = opencode_upstream_api_key(cmd, e)
            if upstream_key:
                e[OPENCODE_GUARD_UPSTREAM_KEY_ENV] = upstream_key
                extra = ["--api-key-env", OPENCODE_GUARD_UPSTREAM_KEY_ENV, *extra]
            elif not base.startswith(("http://127.0.0.1:", "http://localhost:")):
                return cmd  # remote OpenAI-wire upstreams need a key guard can hold
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
    payload = {
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
    # OBSERVABLE for the measured launch-prompt baseline / seeded context budget (claude
    # only), so fleet drift in the launch prompt is a visible number in the record.
    if backend == "claude":
        baseline, _ = measured_claude_guard_baseline(workspace)
        envelope_ceiling = (CLAUDE_GUARD_MODEL_WINDOW_TOKENS
                            - CLAUDE_GUARD_OUTPUT_RESERVE_TOKENS)
        payload["guard_baseline_tokens"] = baseline
        payload["guard_context_budget_tokens"] = min(
            baseline * CLAUDE_GUARD_BIRTH_HEADROOM_FACTOR, envelope_ceiling)
    return payload


def render(payload: dict[str, Any]) -> str:
    cmd = " ".join(payload.get("command") or []) or "-"
    lines = [
        f"dispatch-worker: backend={payload.get('backend')} lane={payload.get('lane')} "
        f"guarded={payload.get('guarded')} dry_run={payload.get('dry_run')}",
        f"command: {cmd}",
    ]
    if payload.get("error"):
        lines.append(f"error: {payload['error']}")
    budget = payload.get("guard_context_budget_tokens")
    if budget:
        lines.append(
            f"guard: measured_baseline={payload.get('guard_baseline_tokens')} "
            f"context_budget={budget} tokens")
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
    env: dict[str, str] = {}
    if not error:
        env = child_env(args.lane, backend, workspace)
        command, guarded = guarded_launch_command(
            build_command(args.lane, backend), args.lane, backend, workspace, env
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
