#!/usr/bin/env python3
"""proc_resource_guard.py -- catch a runaway process before it pins the host.

Root cause this exists for: a single process can leak OS threads without bound
(the witnessed incident was an external ``llama-cli`` invoked CPU-only with no
``-t`` thread bound that climbed to ~129,427 threads on ONE process, pinning the
machine -- 74% avg CPU, processor-queue 26-41, 73k context-switches/sec). No
existing fleet watchdog watches per-process resource *level*; they keep the
supervisor and sessions ALIVE, but nothing flags a live process whose thread /
handle / working-set count has gone pathological. This is that missing guard.

It is a control-pane loop first (read-only ``--json`` status the pane folds:
``ok:false`` == ACTION, a runaway is live) and an opt-in reaper second (``--enact``
kills the flagged runaways, never a protected OS-critical process or this tool's
own process tree).

Beyond raw resource *level*, the guard also reaps **orphaned sprawl** -- the
quieter way a long-uptime fleet host slows down (a thousand small synchronous
hook/MCP children outliving the sessions that spawned them). Two evidence-based
rules, both opt-in: ``--reap-orphans`` flags an ephemeral helper whose owner is
gone (default pattern ``dos_mcp.server`` -- a per-session MCP server still
resident after the claude/opencode client that launched it died, serving no one),
and ``--reap-idle-shells`` flags a launcher shell (pwsh/powershell/bash) with
zero live children aged past a floor. Both reuse the same protected-names guard,
the same ``--enact`` gate, and the same ledger. The liveness test is direction-
safe under PID reuse: a reused parent PID reads as *alive* and is never reaped
(a missed reap, never a wrong one).

Single-shot by design for the *level* dimensions: thread count is the load-bearing
signal (129k threads is unambiguous and needs no second sample), so the guard never
has to poll a counter twice. The one exception is the opt-in **CPU-pin** dimension
(``--max-cpu-pct``), which catches the quieter runaway the thread ceiling cannot --
a *single-threaded* process pinning one core to 100% forever (a stuck spin loop, a
``while true`` in a terminal, an inference binary wedged on the CPU). That has a
normal thread/handle count, so the only witness is rate-of-CPU: the guard takes two
(or more) cumulative-CPU-seconds samples ``--cpu-window`` apart and flags a process
whose *sustained* per-core CPU (the minimum across consecutive windows, so a brief
legitimate burst that ends mid-measurement is not mistaken for a pin) stays over the
threshold. Because even a multi-second window cannot tell a legitimate minutes-long
CPU job from a wedged loop, *auto-reaping* a CPU-only pin is additionally gated on
``--cpu-reap-confirm`` consecutive runs (a tiny start-time-keyed pid streak ledger): a
standing reaper only kills a core-pin that has persisted across scheduled ticks, while
thread/handle runaways and orphans still reap immediately. Cross-platform via the
platform's own tools (PowerShell on Windows, ``ps`` on POSIX -- in whichever of the
two ``ps`` dialects the host speaks, see ``_ps_census_spec``); no third-party deps.

Exit code: 0 == clean / disabled (no runaway) ; 1 == a runaway is flagged
(ACTION). With ``--enact`` the kills are reported in the JSON ``enacted`` list.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_CREATE_NO_WINDOW = 0x08000000


def _win_creationflags() -> int:
    return _CREATE_NO_WINDOW if os.name == "nt" else 0


SCHEMA = "fleet-proc-resource-guard/1"

# Thread count well above any legitimate process observed on a dev host (the
# busiest legit process in the incident snapshot was the NT kernel "System" at
# ~613 threads; a desktop terminal ~328). 2000 is a wide safety margin under the
# pathological 129k while never tripping on a healthy heavy app.
DEFAULT_MAX_THREADS = 2000
DEFAULT_MAX_HANDLES = 0  # 0 == dimension disabled
DEFAULT_MAX_WS_MB = 0    # 0 == dimension disabled

# CPU-pin dimension (opt-in via --max-cpu-pct; 0 disables). Percent is TOP-style
# per-core: 100 == one full core saturated, 400 == four cores. A single-threaded
# runaway pins exactly one core (100%/core) while showing a normal thread/handle
# count, so this is the only dimension that needs two samples (a rate, not a level).
DEFAULT_MAX_CPU_PCT = 0.0      # 0 == dimension disabled (default)
DEFAULT_CPU_WINDOW_SEC = 3.0   # seconds between consecutive CPU samples
DEFAULT_CPU_SAMPLES = 2        # 2 == one window; >2 requires the pin to hold every window

# Cross-tick reap confirmation for the CPU dimension. A within-one-run sustained
# window (cpu_pct_sustained) tells a burst from a pin over a few SECONDS; it cannot
# tell a 6-second legit compile from a 6-hour wedged process. Duration is the only
# honest separator, and the only honest way to measure MINUTES is across consecutive
# scheduled runs. So a CPU-ONLY pin is reaped (--enact) only after it has been flagged
# in this many consecutive guard runs (1 == reap on first detection, the default for a
# one-shot manual run; a standing reaper should set >=2). Thread/handle runaways and
# orphans are unambiguous and always reap immediately, regardless of this.
DEFAULT_CPU_REAP_CONFIRM = 1
CPU_STREAK_LEDGER = "cpu_pin_streak.json"

# Orphan-sprawl reaping (opt-in via --reap-orphans / --reap-idle-shells). An
# "orphaned helper" is an ephemeral stdio child still resident after its owner
# (the claude/opencode session that spawned it) exited -- it serves no client.
# The default pattern is the DOS MCP server each session launches as
# ``python -m dos_mcp.server``; the match is a substring over "<name> <cmdline>".
DEFAULT_ORPHAN_PATTERNS: tuple[str, ...] = ("dos_mcp.server",)
# Launcher shells that legitimately wrap a session (pwsh -> claude -> mcp). A
# shell with zero live children, aged past the floor, is a stray launcher whose
# session exited. Matched against the bare (extension-stripped) process name.
DEFAULT_IDLE_SHELL_NAMES = frozenset({"pwsh", "powershell", "bash", "sh"})
DEFAULT_IDLE_SHELL_AGE_SEC = 1800  # 30 min: well past any session-launch transient
DEFAULT_ORPHAN_CONSOLE_SHELL_NAMES = frozenset({"cmd"})
DEFAULT_CONSOLE_HOST_CHILD_NAMES = frozenset({"conhost", "openconsole"})
DEFAULT_INTERACTIVE_PARENT_NAMES = frozenset({
    "windowsterminal", "terminal", "conhost", "openconsole",
    "explorer", "cmd", "powershell", "pwsh",
})

# OS-critical processes that must NEVER be killed even with --enact. Matched
# case-insensitively against the bare process name (no path, no extension).
PROTECTED_NAMES = frozenset(
    n.lower()
    for n in (
        # Windows kernel / session / security core
        "system", "idle", "registry", "smss", "csrss", "wininit", "winlogon",
        "services", "lsass", "fontdrvhost", "dwm", "sihost", "memory compression",
        # POSIX init / kernel
        "init", "systemd", "launchd", "kernel_task", "kthreadd",
        # The fleet's own terminal host (#2227): its thread count scales with
        # live panes (~16/pane), not with runaway work, so it legitimately
        # crosses the thread ceiling on a busy interactive host — and killing
        # it kills every session on the box. Protected = still reported, never
        # enact-killed, non-actionable for the dispatch preflight (host_cap
        # remains the governor for pane-scaled thread pressure).
        "windowsterminal",
    )
)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


# --------------------------------------------------------------------------- #
# Pure core (testable without spawning anything)
# --------------------------------------------------------------------------- #
def classify(
    procs: Iterable[dict[str, Any]],
    *,
    max_threads: int = DEFAULT_MAX_THREADS,
    max_handles: int = DEFAULT_MAX_HANDLES,
    max_ws_mb: int = DEFAULT_MAX_WS_MB,
    max_cpu_pct: float = DEFAULT_MAX_CPU_PCT,
    protected_pids: frozenset[int] = frozenset(),
    protected_names: frozenset[str] = PROTECTED_NAMES,
    allow_names: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Return the subset of ``procs`` that breach a configured threshold.

    Each input proc is a dict with at least ``pid`` and ``name``; ``threads``,
    ``handles``, ``ws_mb``, ``cpu_pct`` are optional (a missing / negative value means
    the collector could not read that dimension on this platform and it is skipped,
    never treated as a breach). Output rows carry ``reasons`` and a ``protected``
    bit so the reaper can refuse protected kills.
    """
    allow = {n.lower() for n in allow_names}
    flagged: list[dict[str, Any]] = []
    for proc in procs:
        name = str(proc.get("name") or "")
        if name.lower() in allow:
            continue
        reasons: list[str] = []
        threads = _as_int(proc.get("threads"))
        handles = _as_int(proc.get("handles"))
        ws_mb = _as_int(proc.get("ws_mb"))
        cpu_pct = _as_float(proc.get("cpu_pct"))
        if max_threads > 0 and threads is not None and threads > max_threads:
            reasons.append(f"threads {threads} > {max_threads}")
        if max_handles > 0 and handles is not None and handles > max_handles:
            reasons.append(f"handles {handles} > {max_handles}")
        if max_ws_mb > 0 and ws_mb is not None and ws_mb > max_ws_mb:
            reasons.append(f"ws_mb {ws_mb} > {max_ws_mb}")
        if max_cpu_pct > 0 and cpu_pct is not None and cpu_pct > max_cpu_pct:
            reasons.append(f"cpu {cpu_pct:.0f}%/core > {max_cpu_pct:.0f}% sustained")
        if not reasons:
            continue
        pid = _as_int(proc.get("pid"))
        protected = (pid in protected_pids) or (name.lower() in protected_names)
        flagged.append(
            {
                "pid": pid,
                "name": name,
                "threads": threads,
                "handles": handles,
                "ws_mb": ws_mb,
                "cpu_pct": cpu_pct,
                "start": proc.get("start"),
                "reasons": reasons,
                "protected": protected,
            }
        )
    # Surface the loudest signal first: a CPU pin (a live core-burner) outranks a
    # high static thread count for operator attention, then thread count breaks ties.
    flagged.sort(key=lambda r: (r.get("cpu_pct") or 0.0, r.get("threads") or 0), reverse=True)
    return flagged


def _strip_exe(name: Any) -> str:
    n = str(name or "")
    return n[:-4] if n.lower().endswith(".exe") else n


def _owner_alive(ppid: int | None, live_pids: frozenset[int]) -> bool:
    """A real owner is a live PID > 1. PID 0/1 (idle/init) never *owns* an
    ephemeral stdio helper, so a child reparented there reads as orphaned. Under
    PID reuse a stale ppid that now names a live process reads as alive -- so the
    helper is conservatively spared, never wrongly reaped."""
    return ppid is not None and ppid > 1 and ppid in live_pids


def classify_orphans(
    procs: Iterable[dict[str, Any]],
    *,
    live_pids: frozenset[int],
    child_counts: dict[int, int] | None = None,
    child_names: dict[int, list[str]] | None = None,
    parent_names: dict[int, str] | None = None,
    orphan_patterns: tuple[str, ...] = (),
    idle_shell_names: frozenset[str] = frozenset(),
    orphan_console_shell_names: frozenset[str] = DEFAULT_ORPHAN_CONSOLE_SHELL_NAMES,
    console_host_child_names: frozenset[str] = DEFAULT_CONSOLE_HOST_CHILD_NAMES,
    interactive_parent_names: frozenset[str] = DEFAULT_INTERACTIVE_PARENT_NAMES,
    min_age_sec: int = 0,
    reap_idle_shells: bool = False,
    protected_pids: frozenset[int] = frozenset(),
    protected_names: frozenset[str] = PROTECTED_NAMES,
    allow_names: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Flag orphaned sprawl: ephemeral helpers whose owner is gone, and idle
    launcher shells with no live children. Pure: each ``proc`` dict carries
    ``pid``, ``name`` (extension-stripped), ``ppid``, ``cmdline``, ``age_sec``;
    ``live_pids`` and ``child_counts`` are derived from the same scan. Rows share
    the shape ``classify`` emits (+ a ``kind``) so the reaper treats them alike."""
    patterns = tuple(p for p in orphan_patterns if p)
    shells = {n.lower() for n in idle_shell_names}
    orphan_console_shells = {n.lower() for n in orphan_console_shell_names}
    console_hosts = {n.lower() for n in console_host_child_names}
    counts = child_counts or {}
    kids_by_parent = child_names or {}
    parents = parent_names or {}
    allow = {n.lower() for n in allow_names}
    flagged: list[dict[str, Any]] = []
    for proc in procs:
        name = str(proc.get("name") or "")
        if name.lower() in allow:
            continue
        pid = _as_int(proc.get("pid"))
        ppid = _as_int(proc.get("ppid"))
        cmdline = str(proc.get("cmdline") or "")
        age_sec = _as_int(proc.get("age_sec"))
        reasons: list[str] = []
        kind: str | None = None

        # Orphaned ephemeral helper: matches a known pattern AND its owner is gone.
        if patterns and not _owner_alive(ppid, live_pids):
            hay = f"{name} {cmdline}"
            if any(pat in hay for pat in patterns):
                reasons.append(f"orphaned helper: owner pid {ppid} not alive")
                kind = "orphan-helper"

        # Idle launcher shell: a wrapper shell with no live children, aged out.
        if reap_idle_shells and name.lower() in shells:
            kids = counts.get(pid, 0) if pid is not None else 0
            aged = min_age_sec <= 0 or (age_sec is not None and age_sec >= min_age_sec)
            parent_name = (parents.get(ppid or -1) or "").lower()
            attended_parent = parent_name in interactive_parent_names
            if kids == 0 and aged and not attended_parent:
                age_note = f", age {age_sec}s" if age_sec is not None else ""
                reasons.append(f"idle launcher shell: 0 live children{age_note}")
                kind = kind or "idle-shell"

        # Orphaned console shell: cmd.exe can outlive the parent with only its
        # conhost/openconsole child, so the generic "zero children" idle-shell
        # rule cannot see it. This is safe only when the owner is gone and every
        # remaining child is just the console host.
        if reap_idle_shells and name.lower() in orphan_console_shells:
            aged = min_age_sec <= 0 or (age_sec is not None and age_sec >= min_age_sec)
            child_list = [c.lower() for c in kids_by_parent.get(pid or -1, [])]
            only_console_children = all(child in console_hosts for child in child_list)
            if (not _owner_alive(ppid, live_pids)) and aged and only_console_children:
                age_note = f", age {age_sec}s" if age_sec is not None else ""
                child_note = f", children={','.join(child_list)}" if child_list else ", children=none"
                reasons.append(
                    f"orphaned console shell: owner pid {ppid} not alive{child_note}{age_note}"
                )
                kind = kind or "orphan-console-shell"

        if not reasons:
            continue
        protected = (pid in protected_pids) or (name.lower() in protected_names)
        parent_name = parents.get(ppid or -1)
        flagged.append(
            {
                "pid": pid,
                "name": name,
                "ppid": ppid,
                "parent_name": parent_name,
                "threads": _as_int(proc.get("threads")),
                "handles": None,
                "ws_mb": _as_int(proc.get("ws_mb")),
                "reasons": reasons,
                "protected": protected,
                "kind": kind,
            }
        )
    flagged.sort(key=lambda r: r["pid"] or 0)
    return flagged


def _child_counts(rows: Iterable[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in rows:
        ppid = _as_int(row.get("ppid"))
        if ppid is not None:
            counts[ppid] = counts.get(ppid, 0) + 1
    return counts


def _child_names(rows: Iterable[dict[str, Any]]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for row in rows:
        ppid = _as_int(row.get("ppid"))
        if ppid is None:
            continue
        out.setdefault(ppid, []).append(str(row.get("name") or "").lower())
    return out


def _parent_names(rows: Iterable[dict[str, Any]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for row in rows:
        pid = _as_int(row.get("pid"))
        if pid is not None:
            out[pid] = str(row.get("name") or "")
    return out


def _merge_flagged(
    resource_rows: list[dict[str, Any]], orphan_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Union flagged rows by pid: a process can breach a resource threshold AND
    be orphaned -- concat its reasons, OR its protected bit, keep one row."""
    by_pid: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []
    for row in list(resource_rows) + list(orphan_rows):
        pid = row.get("pid")
        if pid in by_pid:
            tgt = by_pid[pid]
            tgt["reasons"] = list(tgt["reasons"]) + [
                r for r in row["reasons"] if r not in tgt["reasons"]
            ]
            tgt["protected"] = tgt["protected"] or row["protected"]
            tgt["kind"] = tgt.get("kind") or row.get("kind")
        else:
            by_pid[pid] = dict(row)
            order.append(pid)
    merged = [by_pid[p] for p in order]
    merged.sort(key=lambda r: (r.get("cpu_pct") or 0.0, r.get("threads") or 0), reverse=True)
    return merged


def build_payload(
    procs: list[dict[str, Any]],
    *,
    max_threads: int,
    max_handles: int,
    max_ws_mb: int,
    protected_pids: frozenset[int],
    allow_names: frozenset[str],
    enact: bool,
    max_cpu_pct: float = DEFAULT_MAX_CPU_PCT,
    cpu_reap_confirm: int = DEFAULT_CPU_REAP_CONFIRM,
    cpu_streaks_prev: dict[str, int] | None = None,
    killer: Any = None,
    collect_error: str = "",
    orphan_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    flagged = _merge_flagged(
        classify(
            procs,
            max_threads=max_threads,
            max_handles=max_handles,
            max_ws_mb=max_ws_mb,
            max_cpu_pct=max_cpu_pct,
            protected_pids=protected_pids,
            allow_names=allow_names,
        ),
        orphan_rows or [],
    )

    # Cross-tick streak ledger: bump every (pid+start) key CPU-flagged THIS run, drop
    # the rest. Keyed by start time too, so a recycled pid cannot inherit a streak.
    cpu_keys = [
        cpu_streak_key(r["pid"], r.get("start"))
        for r in flagged
        if any(_is_cpu_reason(x) for x in (r.get("reasons") or []))
    ]
    cpu_streaks = _bump_cpu_streaks(cpu_streaks_prev or {}, cpu_keys)

    def _cpu_only(row: dict[str, Any]) -> bool:
        # Flagged ONLY for CPU (no thread/handle/ws level breach, not an orphan/idle
        # shell). Those other classes are unambiguous and reap immediately; a CPU-only
        # pin is the one that must clear the cross-tick confirmation first.
        reasons = row.get("reasons") or []
        return bool(reasons) and all(_is_cpu_reason(x) for x in reasons) and not row.get("kind")

    enacted: list[dict[str, Any]] = []
    confirm = max(1, cpu_reap_confirm)
    for row in flagged:
        is_cpu = any(_is_cpu_reason(x) for x in (row.get("reasons") or []))
        streak = cpu_streaks.get(cpu_streak_key(row["pid"], row.get("start")), 0) if is_cpu else 0
        if is_cpu:
            row["cpu_streak"] = streak
        if not (enact and killer is not None):
            row["action"] = "report"
            continue
        if row["protected"]:
            row["action"] = "protected-skip"
            continue
        if _cpu_only(row) and streak < confirm:
            # A core-pin not yet confirmed across enough runs -- surfaced (still
            # ACTION), but NOT killed: this is the gate that keeps a legitimate
            # minutes-long CPU job from being reaped as if it were a wedged loop.
            row["action"] = "cpu-unconfirmed"
            continue
        ok, detail = killer(row["pid"])
        row["action"] = "killed" if ok else "kill-failed"
        enacted.append({"pid": row["pid"], "name": row["name"], "ok": ok, "detail": detail})

    # ACTION (ok:false) iff a collector failed (we cannot prove the host is
    # clean) OR a NON-PROTECTED process is flagged. A protected breach -- e.g.
    # the NT kernel `System` thread pool transiently crossing the ceiling on a
    # busy many-session host -- is still listed in `flagged` (and logged), but
    # the reaper always skips it (`protected-skip`), so it is non-actionable by
    # construction and must NOT flip the control-pane ok bit into a perpetual
    # false ACTION. (Witnessed: a recurring FLAGGED(1) System(pid 4) alarm.)
    actionable_flagged = [r for r in flagged if not r["protected"]]
    ok = (not collect_error) and (len(actionable_flagged) == 0)
    return {
        "schema": SCHEMA,
        "ok": ok,
        "ts": now_iso(),
        "platform": platform.system(),
        "thresholds": {
            "max_threads": max_threads,
            "max_handles": max_handles,
            "max_ws_mb": max_ws_mb,
            "max_cpu_pct": max_cpu_pct,
        },
        "cpu_reap_confirm": confirm,
        "cpu_streaks": cpu_streaks,
        "scanned": len(procs),
        "flagged_count": len(flagged),
        "actionable_flagged_count": len(actionable_flagged),
        "flagged": flagged,
        "enacted": enacted,
        "enact": enact,
        "collect_error": collect_error or None,
        "next_action": _next_action(flagged, enact, collect_error),
    }


def _next_action(flagged: list[dict[str, Any]], enact: bool, collect_error: str) -> str:
    if collect_error:
        return "process scan failed; rerun the guard and inspect collect_error"
    if not flagged:
        return "no runaway or orphaned process; no action"
    names = ", ".join(sorted({f"{r['name']}(pid {r['pid']})" for r in flagged}))
    if enact:
        killed = sorted({f"{r['name']}(pid {r['pid']})" for r in flagged if r.get("action") == "killed"})
        deferred = sorted({f"{r['name']}(pid {r['pid']})" for r in flagged if r.get("action") == "cpu-unconfirmed"})
        parts: list[str] = []
        if killed:
            parts.append(f"reaped: {', '.join(killed)}")
        if deferred:
            parts.append(f"CPU pin watched (not yet confirmed across runs, NOT reaped): {', '.join(deferred)}")
        if not parts:
            return f"flagged: {names}; nothing reaped (protected or unconfirmed)"
        return "; ".join(parts) + " (protected ones skipped)"
    kinds = {r.get("kind") or "runaway" for r in flagged}
    if kinds <= {"orphan-helper", "idle-shell"}:
        hint = "orphaned sprawl serving no live session; re-run with --enact to reap."
    else:
        hint = (
            "Inspect, then re-run with --enact to kill, "
            "or fix the launcher (bound -t/--threads on inference binaries)."
        )
    return f"flagged: {names}. {hint}"


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def cpu_pct_delta(cpu_s_before: Any, cpu_s_after: Any, dt: float) -> float | None:
    """Per-core (top-style) CPU% over one window: 100 == one full core saturated.

    ``cpu_s_*`` are cumulative CPU-seconds (summed over every core the process used).
    A process that burned one core for the whole ``dt``-second window accrues ``dt``
    CPU-seconds -> 100%; four cores -> 400%. Returns ``None`` (dimension skipped, never
    a breach) when a sample is missing or the counter went backwards -- a backward
    delta means the PID was reused by a newly-started process, so we refuse to
    attribute the old process's time to it (a missed flag, never a wrong kill)."""
    before = _as_float(cpu_s_before)
    after = _as_float(cpu_s_after)
    if before is None or after is None or dt <= 0:
        return None
    delta = after - before
    if delta < 0:
        return None
    return (delta / dt) * 100.0


def cpu_pct_sustained(samples: list[dict[Any, Any]], dt: float) -> dict[Any, float]:
    """Sustained per-core CPU% per pid: the MINIMUM window-% across consecutive
    samples. Each entry of ``samples`` maps pid -> cumulative CPU-seconds at one
    instant, taken ``dt`` apart. Taking the minimum (not the mean) is what makes the
    signal a *pin* and not a *burst*: a process must stay over the threshold in EVERY
    window to score high, so a legitimate compile that saturates a core for one window
    and then finishes scores its quiet window (low) and is not flagged. A pid missing
    from any sample, or whose counter went backwards in any window, is omitted."""
    if len(samples) < 2 or dt <= 0:
        return {}
    pids: set[Any] = set()
    for snap in samples:
        pids.update(snap.keys())
    out: dict[Any, float] = {}
    for pid in pids:
        windows: list[float] = []
        ok = True
        for before, after in zip(samples, samples[1:]):
            pct = cpu_pct_delta(before.get(pid), after.get(pid), dt)
            if pct is None:
                ok = False
                break
            windows.append(pct)
        if ok and windows:
            out[pid] = min(windows)
    return out


def _is_cpu_reason(reason: Any) -> bool:
    return isinstance(reason, str) and reason.startswith("cpu ")


def cpu_streak_key(pid: Any, start: Any) -> str:
    """Stable cross-run identity for a process: pid PLUS its start time. Keying the
    streak on the pair (not the pid alone) is what makes cross-run confirmation
    reuse-safe -- a recycled pid carries a different start time, so it gets a FRESH
    streak instead of inheriting a dead process's confirmation count. When the start
    time is unavailable (POSIX basic scan, or an access-denied Windows process) the
    key degrades to pid-only; the live reaper target is Windows, where start is read."""
    return f"{pid}:{start if start not in (None, '') else ''}"


def _bump_cpu_streaks(prev: dict[str, int], cpu_keys: Iterable[str]) -> dict[str, int]:
    """Increment the consecutive-run streak for each currently CPU-flagged process key
    and DROP every key not flagged this run, so a streak survives only while THAT exact
    process (pid+start) keeps pinning run-to-run. A recycled pid is a different key and
    starts from zero; a within-run counter reset already prevents a false pin score
    (see cpu_pct_delta). Net: a missed reap under reuse, never a wrong one."""
    out: dict[str, int] = {}
    for key in cpu_keys:
        if key is None:
            continue
        out[key] = prev.get(key, 0) + 1
    return out


# --------------------------------------------------------------------------- #
# Platform collectors (I/O)
# --------------------------------------------------------------------------- #
def collect_processes() -> tuple[list[dict[str, Any]], str]:
    system = platform.system()
    try:
        if system == "Windows":
            return _collect_windows(), ""
        return _collect_posix(system)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return [], f"{type(exc).__name__}: {exc}"


def collect_processes_cpu(
    window_sec: float = DEFAULT_CPU_WINDOW_SEC,
    samples: int = DEFAULT_CPU_SAMPLES,
    sleeper: Any = time.sleep,
) -> tuple[list[dict[str, Any]], str]:
    """Like ``collect_processes`` but enriches each row with a ``cpu_pct`` measured
    over ``samples`` snapshots taken ``window_sec`` apart. The LAST (most recent)
    snapshot is returned as the process set, annotated with the sustained per-core
    CPU% (``cpu_pct_sustained`` -> the minimum across windows). Used only when the
    CPU dimension is enabled, so the common path pays no extra scan. ``sleeper`` is
    injectable for hermetic tests."""
    n = max(2, samples)
    snaps: list[dict[Any, Any]] = []
    last: list[dict[str, Any]] = []
    for i in range(n):
        if i:
            sleeper(max(0.1, window_sec))
        procs, err = collect_processes()
        if err:
            return procs, err
        last = procs
        snaps.append({p.get("pid"): p.get("cpu_s") for p in procs if p.get("pid") is not None})
    pct = cpu_pct_sustained(snaps, window_sec)
    for p in last:
        p["cpu_pct"] = pct.get(p.get("pid"))
    return last, ""


def _collect_windows() -> list[dict[str, Any]]:
    script = (
        "Get-Process -ErrorAction SilentlyContinue | ForEach-Object { "
        "try { $st=''; try { $st=$_.StartTime.ToUniversalTime().ToString('o') } catch {}; "
        "[pscustomobject]@{ pid=$_.Id; name=$_.ProcessName; "
        "threads=$_.Threads.Count; handles=$_.HandleCount; ws=[int64]$_.WorkingSet64; "
        "cpu=$_.CPU; start=$st } } catch {} "
        "} | ConvertTo-Json -Compress"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        creationflags=_win_creationflags(),
    )
    return _parse_windows_json(proc.stdout)


def _parse_windows_json(text: str) -> list[dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return []
    obj = json.loads(text)
    rows = obj if isinstance(obj, list) else [obj]
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ws = _as_int(row.get("ws"))
        out.append(
            {
                "pid": _as_int(row.get("pid")),
                "name": str(row.get("name") or ""),
                "threads": _as_int(row.get("threads")),
                "handles": _as_int(row.get("handles")),
                "ws_mb": (ws // (1024 * 1024)) if ws is not None else None,
                "cpu_s": _as_float(row.get("cpu")),
                "start": (str(row.get("start")) if row.get("start") else None),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# `ps` dialects (POSIX)
# --------------------------------------------------------------------------- #
# `ps` is two tools wearing one name, and this is the Python twin of the Go fix in
# internal/procguard (#5385) -- same rules, deliberately, because it is the same bug.
#
# procps-ng (Linux) defines output keywords the BSD dialect has never had -- nlwp,
# etimes, cputimes -- and a `ps` handed a keyword it does not know rejects the
# INVOCATION rather than dropping that one column. A real non-procps `ps` (the MSYS2
# build shipped with Git for Windows, which does not implement -o at all) answers:
#
#     $ ps -eo pid=,nlwp=,rss=,cputimes=,comm=
#     ps: unknown option -- o
#     Try `ps --help' for more information.
#     ; exit 1, stdout EMPTY
#
# so every row is lost, not merely the thread column. Before #5541 that came back
# from _collect_posix as rows=[] paired with an EMPTY error string, and the guard
# printed scanned=0 / ok=true / "no runaway or orphaned process; no action" -- a host
# it could not measure at all, reported as a host measured and found clean. An
# unreadable dimension has to be NAMED, never rendered as a reading of zero.
#
# Three rules follow from that, one per failure the old code had:
#   1. the INVOCATION varies by dialect (_ps_census_spec / _ps_relations_spec), so a
#      BSD host gets an argv its `ps` can actually answer;
#   2. a column the dialect simply does not have is PS_NO_COLUMN and leaves the field
#      None -- "unknown", never a fabricated 0 (classify() skips a None dimension);
#   3. a tool that fails keeps its stdout, and _census_error decides what the failure
#      means from whether anything parsed.
PS_NO_COLUMN = -1

# Keywords procps-ng defines and BSD `ps` does not. Kept in step with
# internal/architest/ps_dialect_test.go's procpsOnlyKeywords, which enforces the same
# list over Go source; the Python half is enforced by proc_resource_guard_test.py.
PS_PROCPS_ONLY_KEYWORDS = ("nlwp", "etimes", "cputimes")


def _ps_bsd(system: str) -> bool:
    """Whether ``platform.system()`` names a host that speaks the BSD `ps` dialect.

    ONLY Darwin, matching internal/procguard.psBSD and for the same reason: Darwin is
    the platform that was actually witnessed rejecting the procps keywords. The other
    BSDs very probably behave the same way, but "very probably" is the assumption that
    shipped this bug -- an unwitnessed platform keeps the invocation known to work on
    the hosts fak is known to run on, and _census_error is what stops such a host
    reading as an empty machine in the meantime. Adding a name here should come with a
    pasted `ps` transcript from it."""
    return system == "Darwin"


def _ps_census_spec(system: str) -> dict[str, Any]:
    """The resource-census `ps` dialect for a host: pid, threads, RSS, CPU seconds.

    Why a branch and not one column set both dialects accept: the vocabularies do
    intersect (pid, rss, comm, time) but the intersection contains NO thread-count
    keyword -- BSD `ps` has none at all. A shared column set would have to drop nlwp,
    silently disabling the thread dimension on Linux, which is the dimension this
    guard exists for (the incident was one process at ~129,427 threads). Fixing BSD by
    blinding Linux is not a fix, so Linux keeps its invocation byte for byte and BSD
    gets the best set its `ps` can answer -- thread count is not among them and stays
    None. ``cpu_time`` sits BEFORE ``comm`` in both so a space-bearing command name
    stays the parser's final field."""
    if _ps_bsd(system):
        # BSD's cumulative-CPU column is `time`, FORMATTED [[dd-]hh:]mm:ss[.ff] rather
        # than the bare seconds procps' `cputimes` emits -- see _parse_ps_duration.
        return {
            "args": ["-eo", "pid=,rss=,time=,comm="],
            "n": 4,
            "pid": 0,
            "threads": PS_NO_COLUMN,  # BSD `ps` has no thread keyword; threads stays None
            "rss_kb": 1,
            "cpu_time": 2,
            "comm": 3,
            "ppid": PS_NO_COLUMN,
            "elapsed": PS_NO_COLUMN,
            "argv": PS_NO_COLUMN,
            "optional": 2,
        }
    return {
        "args": ["-eo", "pid=,nlwp=,rss=,cputimes=,comm="],
        "n": 5,
        "pid": 0,
        "threads": 1,
        "rss_kb": 2,
        "cpu_time": 3,
        "comm": 4,
        "ppid": PS_NO_COLUMN,
        "elapsed": PS_NO_COLUMN,
        "argv": PS_NO_COLUMN,
        "optional": 3,
    }


def _ps_relations_spec(system: str) -> dict[str, Any]:
    """The relations `ps` dialect: pid, ppid, elapsed age, command name, full cmdline.

    BSD has `etime` (formatted) where procps has `etimes` (whole seconds); every other
    keyword in this row is common vocabulary, which is why the two argvs differ by
    exactly one keyword."""
    elapsed = "etime=," if _ps_bsd(system) else "etimes=,"
    return {
        "args": ["-eo", "pid=,ppid=," + elapsed + "comm=,args="],
        "n": 5,
        "pid": 0,
        "threads": PS_NO_COLUMN,
        "rss_kb": PS_NO_COLUMN,
        "cpu_time": PS_NO_COLUMN,
        "comm": 3,
        "ppid": 1,
        "elapsed": 2,
        "argv": 4,
        "optional": 4,  # a zombie has no argv; cmdline then falls back to the comm
    }


def _ps_column(spec: dict[str, Any], parts: list[str], key: str) -> str:
    """One spec column out of an already-split line.

    Answers "" for a column this dialect does not have, and for the ``optional``
    column on a line that came back one field short -- the case where every later
    column shifts one to the left, because `ps` omits the VALUE, not just its
    contents (a zombie has neither accumulated CPU time nor an argv). The pre-#5541
    parser already tolerated exactly that one-field-short shape; this keeps the
    tolerance instead of silently dropping those rows."""
    i = spec[key]
    if i == PS_NO_COLUMN:
        return ""
    optional = spec.get("optional", PS_NO_COLUMN)
    if len(parts) < spec["n"] and optional != PS_NO_COLUMN:
        if i == optional:
            return ""
        if i > optional:
            i -= 1
    if i < 0 or i >= len(parts):
        return ""
    return parts[i]


def _parse_ps_duration(value: Any) -> float | None:
    """One `ps` time column in seconds, in either dialect.

    procps' ``etimes``/``cputimes`` are bare integers; BSD has no such keyword and its
    ``etime``/``time`` columns are FORMATTED [[dd-]hh:]mm:ss[.ff]. A keyword rename
    alone would therefore have produced ages of 0 on a BSD host, so the BSD column set
    is only half a fix without this. One parser reads both: a bare integer is the
    degenerate seconds-only case of the same grammar, so the Linux path still yields
    exactly the integer it yielded before.

    None for anything unreadable (a `-` placeholder, a leaked header, an unexpected
    dialect). The caller then leaves the field None rather than recording a zero: a
    zero age means "started this instant" and a zero CPU time means "never ran", both
    of which are claims, and neither is witnessed by an unparseable string."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    days = 0.0
    if "-" in text:  # dd-hh:mm:ss
        head, _, text = text.partition("-")
        try:
            days = float(head)
        except ValueError:
            return None
        if days < 0:
            return None
    fields = text.split(":")
    if len(fields) > 3:  # nothing in either dialect is deeper than hh:mm:ss
        return None
    total = 0.0
    for field in fields:
        # Digits and one decimal point only. float() alone would accept "nan", "inf"
        # and "1e9"; a census column is never any of those, and a NaN age would poison
        # every comparison downstream instead of being rejected here.
        if not field or any(ch not in "0123456789." for ch in field):
            return None
        try:
            seconds = float(field)
        except ValueError:
            return None
        if seconds < 0:
            return None
        total = total * 60 + seconds
    return days * 86400 + total


def _run_tool(timeout: float, *argv: str) -> tuple[str, str]:
    """Run a census tool and return BOTH its stdout and, if it failed, the failure text.

    The two are not exclusive, which is the whole point: a `ps` that does not
    recognise one requested keyword may still print the rows it does understand and
    only then exit non-zero, and discarding that stdout is what made a POSIX census
    report an empty machine on a host running hundreds of processes. It is the CALLER
    that decides what a non-zero exit means, because only the caller can see whether
    the bytes parsed into anything.

    The tool's stderr rides along in the error text. "exit status 1" alone was true
    and useless -- the sentence that names this bug on sight ("ps: unknown option --
    o") was being thrown away with it."""
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            creationflags=_win_creationflags(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return "", f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        detail = f"exit status {proc.returncode}"
        stderr = (proc.stderr or "").strip()
        if stderr:
            detail += ": " + stderr[:200]
        return (proc.stdout or ""), detail
    return (proc.stdout or ""), ""


def _census_error(rows: int, tool_error: str) -> str:
    """The ONE rule every collector applies to a tool that failed: keep the failure
    only when nothing usable came out of it.

    Both halves matter and they pull in opposite directions.

    Rows AND an error -> report the census, drop the error. A `ps` that printed rows
    and then exited non-zero over one unknown keyword has answered the question.
    Reporting the error anyway would be worse than the original bug in one specific
    way: ``ok`` is computed as ``not collect_error``, so rows-with-an-error would turn
    a host whose census WORKED into a permanent ACTION.

    No rows AND an error -> report the error. Returning zero rows and no error states
    that the host is quiet, which is the one claim this guard must never make on
    evidence it does not have: a guard that saw nothing must say it saw nothing, not
    that there was nothing to see.

    The residual case this cannot separate is a tool that died PART WAY through
    printing; those rows are kept and look complete. That is the direction the rest of
    the guard already fails in (a short census flags fewer runaways: a missed reap,
    never a wrong one)."""
    return "" if rows > 0 else tool_error


def _collect_posix(system: str = "") -> tuple[list[dict[str, Any]], str]:
    spec = _ps_census_spec(system or platform.system())
    out, tool_error = _run_tool(30, "ps", *spec["args"])
    rows = _parse_posix_ps(out, spec)
    error = _census_error(len(rows), tool_error)
    return ([], error) if error else (rows, "")


def _parse_posix_ps(text: str, spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    spec = spec or _ps_census_spec("Linux")
    out: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        parts = line.split(None, spec["n"] - 1)
        if len(parts) < spec["n"] - 1:
            continue
        pid = _as_int(_ps_column(spec, parts, "pid"))
        if pid is None:
            # A line whose first field is not a pid is not a process: a leaked header,
            # or a usage message a failing `ps` printed on stdout. Dropping it is what
            # keeps _census_error honest -- a phantom row would count as a census and
            # suppress the very error that explains the empty result.
            continue
        rss_kb = _as_int(_ps_column(spec, parts, "rss_kb"))
        out.append(
            {
                "pid": pid,
                "name": os.path.basename(_ps_column(spec, parts, "comm").strip()),
                "threads": _as_int(_ps_column(spec, parts, "threads")),
                "handles": None,
                "ws_mb": (rss_kb // 1024) if rss_kb is not None else None,
                "cpu_s": _parse_ps_duration(_ps_column(spec, parts, "cpu_time")),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Relation collectors (ppid / cmdline / age) -- only run when an orphan mode is on
# --------------------------------------------------------------------------- #
def collect_relations() -> tuple[list[dict[str, Any]], str]:
    system = platform.system()
    try:
        if system == "Windows":
            return _collect_windows_relations(), ""
        return _collect_posix_relations(system)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return [], f"{type(exc).__name__}: {exc}"


def _collect_windows_relations() -> list[dict[str, Any]]:
    script = (
        "$now=Get-Date; Get-CimInstance Win32_Process -ErrorAction SilentlyContinue "
        "| ForEach-Object { try { "
        "$a = if ($_.CreationDate) { [int](New-TimeSpan -Start $_.CreationDate -End $now).TotalSeconds } else { -1 }; "
        "[pscustomobject]@{ pid=$_.ProcessId; ppid=$_.ParentProcessId; name=$_.Name; cmd=$_.CommandLine; age=$a } "
        "} catch {} } | ConvertTo-Json -Compress"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
        creationflags=_win_creationflags(),
    )
    return _parse_windows_relations(proc.stdout)


def _parse_windows_relations(text: str) -> list[dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return []
    obj = json.loads(text)
    rows = obj if isinstance(obj, list) else [obj]
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        age = _as_int(row.get("age"))
        out.append(
            {
                "pid": _as_int(row.get("pid")),
                "ppid": _as_int(row.get("ppid")),
                "name": _strip_exe(row.get("name")),
                "cmdline": str(row.get("cmd") or ""),
                "age_sec": age if (age is not None and age >= 0) else None,
            }
        )
    return out


def _collect_posix_relations(system: str = "") -> tuple[list[dict[str, Any]], str]:
    spec = _ps_relations_spec(system or platform.system())
    out, tool_error = _run_tool(30, "ps", *spec["args"])
    rows = _parse_posix_ps_relations(out, spec)
    error = _census_error(len(rows), tool_error)
    return ([], error) if error else (rows, "")


def _parse_posix_ps_relations(text: str,
                              spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    spec = spec or _ps_relations_spec("Linux")
    out: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        parts = line.split(None, spec["n"] - 1)
        if len(parts) < spec["n"] - 1:
            continue
        pid = _as_int(_ps_column(spec, parts, "pid"))
        if pid is None:  # not a process row -- see _parse_posix_ps
            continue
        comm = _ps_column(spec, parts, "comm")
        # An empty argv column (a zombie keeps its accounting name and nothing else)
        # falls back to the command name, exactly as the pre-#5541 parser did.
        args = _ps_column(spec, parts, "argv") or comm
        age = _parse_ps_duration(_ps_column(spec, parts, "elapsed"))
        out.append(
            {
                "pid": pid,
                "ppid": _as_int(_ps_column(spec, parts, "ppid")),
                "name": os.path.basename(comm.strip()),
                "cmdline": args,
                "age_sec": int(age) if age is not None else None,
            }
        )
    return out


def kill_pid(pid: int | None) -> tuple[bool, str]:
    if pid is None or pid <= 0:
        return False, "invalid pid"
    try:
        if platform.system() == "Windows":
            proc = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                creationflags=_win_creationflags(),
            )
            return proc.returncode == 0, (proc.stdout or proc.stderr).strip()[:200]
        os.kill(pid, 9)
        return True, "SIGKILL sent"
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)


# --------------------------------------------------------------------------- #
# Logging + rendering
# --------------------------------------------------------------------------- #
def load_cpu_streaks(log_dir: Path) -> dict[str, int]:
    """Read the cross-tick CPU-pin streak ledger. Any error (absent / corrupt) yields
    an empty ledger -- a lost ledger means a pin must simply re-accumulate its streak,
    which is the safe direction (a missed reap, never a wrong one)."""
    try:
        raw = json.loads((log_dir / CPU_STREAK_LEDGER).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {str(k): int(v) for k, v in raw.items() if _as_int(v) is not None}
    except (OSError, ValueError, TypeError):
        pass
    return {}


def save_cpu_streaks(log_dir: Path, streaks: dict[str, int]) -> None:
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / CPU_STREAK_LEDGER).write_text(json.dumps(streaks), encoding="utf-8")
    except OSError:
        pass


def note(payload: dict[str, Any], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    flagged = payload.get("flagged") or []
    if payload.get("ok"):
        # ok==True can still carry protected (non-actionable) flags -- keep them
        # visible in the log so a System-class breach leaves a trace, not silence.
        protected = sum(1 for r in flagged if r.get("protected"))
        summary = f"CLEAN(protected:{protected})" if protected else "CLEAN"
    else:
        summary = f"FLAGGED({len(flagged)})"
    line = f"{payload.get('ts')}  {summary}  scanned={payload.get('scanned')}  {payload.get('next_action')}"
    with (log_dir / "proc_guard.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def render(payload: dict[str, Any]) -> str:
    lines = [
        f"proc-resource-guard: {'ok' if payload.get('ok') else 'ACTION'} "
        f"(scanned {payload.get('scanned')}, flagged {payload.get('flagged_count')})",
        f"thresholds: {payload.get('thresholds')}",
    ]
    for row in payload.get("flagged") or []:
        tag = "PROTECTED" if row.get("protected") else (row.get("action") or "report")
        kind = f"{row.get('kind')} " if row.get("kind") else ""
        cpu = row.get("cpu_pct")
        if cpu is not None:
            streak = row.get("cpu_streak")
            sfx = f" streak={streak}" if streak is not None else ""
            cpu_str = f"cpu={cpu:.0f}%/core{sfx} "
        else:
            cpu_str = ""
        lines.append(
            f"  [{tag}] {kind}pid={row.get('pid')} {row.get('name')} "
            f"{cpu_str}threads={row.get('threads')} handles={row.get('handles')} "
            f"ws_mb={row.get('ws_mb')} :: {', '.join(row.get('reasons') or [])}"
        )
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Flag (and optionally reap) runaway processes by thread/handle/"
        "working-set count or sustained per-core CPU."
    )
    ap.add_argument("--max-threads", type=int, default=DEFAULT_MAX_THREADS)
    ap.add_argument("--max-handles", type=int, default=DEFAULT_MAX_HANDLES, help="0 disables")
    ap.add_argument("--max-ws-mb", type=int, default=DEFAULT_MAX_WS_MB, help="0 disables")
    ap.add_argument(
        "--max-cpu-pct",
        type=float,
        default=DEFAULT_MAX_CPU_PCT,
        metavar="PCT",
        help="flag a process sustaining > PCT%% of ONE core (top-style: 100 = one full "
        "core) across --cpu-samples windows. Catches a single-threaded runaway pinning a "
        "core that the thread ceiling misses. 0 disables (default).",
    )
    ap.add_argument(
        "--cpu-window",
        type=float,
        default=DEFAULT_CPU_WINDOW_SEC,
        metavar="SEC",
        help=f"seconds between consecutive CPU samples (default {DEFAULT_CPU_WINDOW_SEC}; "
        "use longer on POSIX where cputimes is integer-second)",
    )
    ap.add_argument(
        "--cpu-samples",
        type=int,
        default=DEFAULT_CPU_SAMPLES,
        metavar="N",
        help="CPU snapshots to take (>=2; default 2 = one window). N>2 requires the pin to "
        "hold in EVERY window, so a brief legit burst is not mistaken for a runaway -- the "
        "safe setting before --enact (e.g. --cpu-samples 4 --cpu-window 2 = 6s sustained).",
    )
    ap.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="NAME",
        help="process name to exempt from flagging (repeatable)",
    )
    ap.add_argument(
        "--reap-orphans",
        action="store_true",
        help="also flag orphaned ephemeral helpers (default pattern: dos_mcp.server) "
        "whose owning session has exited",
    )
    ap.add_argument(
        "--orphan-pattern",
        action="append",
        default=[],
        metavar="SUBSTR",
        help="extra name/cmdline substring marking an ephemeral helper "
        "(repeatable; implies --reap-orphans)",
    )
    ap.add_argument(
        "--reap-idle-shells",
        action="store_true",
        help="also flag launcher shells (pwsh/powershell/bash) with zero live "
        "children aged past --idle-shell-age-min",
    )
    ap.add_argument(
        "--idle-shell-age-min",
        type=int,
        default=DEFAULT_IDLE_SHELL_AGE_SEC // 60,
        metavar="MIN",
        help="age floor in minutes for idle-shell reaping (default: 30)",
    )
    ap.add_argument(
        "--enact",
        action="store_true",
        help="DESTRUCTIVE: kill flagged non-protected processes (default: report only)",
    )
    ap.add_argument(
        "--cpu-reap-confirm",
        type=int,
        default=DEFAULT_CPU_REAP_CONFIRM,
        metavar="N",
        help="reap a CPU-ONLY pin (with --enact) only after it is flagged in N consecutive "
        "runs (default 1 = reap on first detection). A standing reaper should set >=2 so a "
        "pin must persist across scheduled ticks (minutes), not just one 6s window, before it "
        "is killed -- this is what keeps a legit minutes-long CPU job from being reaped. "
        "Thread/handle runaways and orphans always reap immediately regardless of this.",
    )
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--log-dir", default="", help="watchdog log dir (default: tools/_watchdog)")
    args = ap.parse_args(argv)

    log_dir = Path(args.log_dir) if args.log_dir else (repo_root() / "tools" / "_watchdog")
    # Cross-tick streak ledger: load before the scan, persist the updated streaks after.
    cpu_streaks_prev = load_cpu_streaks(log_dir) if args.max_cpu_pct > 0 else {}

    # The CPU dimension needs two+ samples; every other dimension is single-shot.
    if args.max_cpu_pct > 0:
        procs, collect_error = collect_processes_cpu(args.cpu_window, args.cpu_samples)
    else:
        procs, collect_error = collect_processes()
    # Never let the guard kill its own process tree.
    protected_pids = frozenset(p for p in (os.getpid(), os.getppid()) if p)

    # Orphan-sprawl pass (opt-in): one extra relation scan only when requested.
    patterns: tuple[str, ...] = ()
    if args.reap_orphans or args.orphan_pattern:
        patterns = DEFAULT_ORPHAN_PATTERNS + tuple(args.orphan_pattern)
    orphan_rows: list[dict[str, Any]] = []
    if patterns or args.reap_idle_shells:
        relations, rel_error = collect_relations()
        if rel_error and not collect_error:
            collect_error = rel_error
        live_pids = frozenset(
            p for p in (_as_int(r.get("pid")) for r in relations) if p
        )
        orphan_rows = classify_orphans(
            relations,
            live_pids=live_pids,
            child_counts=_child_counts(relations),
            child_names=_child_names(relations),
            parent_names=_parent_names(relations),
            orphan_patterns=patterns,
            idle_shell_names=DEFAULT_IDLE_SHELL_NAMES,
            min_age_sec=max(0, args.idle_shell_age_min) * 60,
            reap_idle_shells=args.reap_idle_shells,
            protected_pids=protected_pids,
            allow_names=frozenset(args.allow),
        )

    payload = build_payload(
        procs,
        max_threads=args.max_threads,
        max_handles=args.max_handles,
        max_ws_mb=args.max_ws_mb,
        max_cpu_pct=args.max_cpu_pct,
        cpu_reap_confirm=args.cpu_reap_confirm,
        cpu_streaks_prev=cpu_streaks_prev,
        protected_pids=protected_pids,
        allow_names=frozenset(args.allow),
        enact=args.enact,
        killer=kill_pid,
        collect_error=collect_error,
        orphan_rows=orphan_rows,
    )

    if args.max_cpu_pct > 0:
        save_cpu_streaks(log_dir, payload.get("cpu_streaks") or {})
    try:
        note(payload, log_dir)
    except OSError:
        pass

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
