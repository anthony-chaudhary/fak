#!/usr/bin/env python3
r"""The spawn gate the always-on dispatcher was missing: *is it safe to launch
another dispatch worker right now?*

The supervisor (``dos loop --enact`` → ``dispatch_worker.py --lane {lane}`` →
``claude -p /dos-kernel:dos-dispatch-loop``) historically spawned toward
``dos.toml [supervise].target`` with **no host-load check and no account
check**. On a box that is already a live multi-session fleet that is exactly the
DoS shape we keep re-paying for: each live worker is one ``claude`` session, and
every tool call a session makes spawns a per-tool DOS hook subprocess, so N
unbounded workers × M tool calls = the thread/handle sprawl that makes a 32c box
"feel slow" while every gauge reads idle (see docs/perf-runaway-guard.md and the
129k-thread llama-cli runaway proc_resource_guard.py was written for).

This module is the read-only gate that closes that hole. It answers ONE question
— ``SPAWN_OK`` or a typed ``REFUSE_*`` — by composing four INDEPENDENT checks,
ALL of which must pass:

  1. host_safe       proc_resource_guard.py is CLEAN (no runaway/orphan flagged)
  2. account_free    fleet_accounts.py route returns an available worker account
                     at the requested tier (the switcher — never the ambient
                     default), so a throttled/auth-blocked account can't silently
                     eat the dispatch
  3. under_cap       live worker count < cap, where
                     cap = min(host_cap, dos [supervise].target, --max-workers)
                     when the target is POSITIVE (dos throttles the fleet down),
                     else min(host_cap, --max-workers) when the target is
                     zero/unset (the emit-only `dos loop` manages no standing loop,
                     so the cron-armed self-spawner's own ceiling governs — a zero
                     target is not a spawn kill switch). host_cap is the
                     host-derived ADAPTIVE ceiling (#1337): cores, free RAM, and the
                     live OS-thread total proc_resource_guard polices, turned into
                     the largest worker population the box can sustain right now — so
                     a request for "up to --max-workers" auto-throttles on a loaded
                     box and rises again as load clears, instead of admitting to a
                     magic number. (host_safe above stays the HARD stop on a true
                     runaway; host_cap is the soft gradient that bites first.) and
                     live = max(kernel's `dos loop` alive, an OS process scan for
                     live `dos-dispatch-loop` claudes plus the pid-file witnesses
                     — the issue resolver's `.dispatch-runs` sidecars and the
                     detached /goal launcher's `.goal-runs` breadcrumbs (#2226:
                     a stdin-fed worker carries NO cmdline marker, so only its
                     breadcrumb makes it visible before it leases a lane)) — the
                     MAX so neither a stale lease nor an unleased orphan can hide
                     capacity
  4. commit_headroom on Windows, the same GetPerformanceInfo counters and exact
                     reserve boundary as the running child guard are healthy
                     before a lease or child is created

The bound is the whole DoS proof: with the gate in front of every spawn, the
live worker population is provably ≤ cap, so the per-session hook pressure is
bounded by a constant the operator sets — not by how many lanes have work.

Read-only. Emits JSON (``--json``) or a one-line card. Exit 0 iff ``SPAWN_OK``.
The always-on watchdog and tools/issue_dispatch.py both call this before every
launch; run it by hand any time to ask "could the fleet grow right now, and if
not, exactly which budget is exhausted?".
"""
from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import datetime as dt
import functools
import json
import ntpath
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


def _no_window_creationflags() -> int:
    """``creationflags`` that stop a console child — the gh/git/fak JSON helpers and the
    PowerShell CIM/liveness probes below — from popping a visible window when preflight
    runs windowless (pythonw) from a scheduled dispatch tick; ``0`` on POSIX. Mirrors
    dispatch_worker.no_window_creationflags, kept local so this module imports only
    stdlib."""
    return 0x08000000 if os.name == "nt" else 0

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fleet-dispatch-preflight/1"

OK_VERDICT = "SPAWN_OK"
REFUSE_HOST = "REFUSE_HOST"          # host resource guard flagged a runaway/orphan
REFUSE_NO_ACCOUNT = "REFUSE_NO_ACCOUNT"  # switcher has no available worker account (throttle/auth)
REFUSE_NO_SEAT = "REFUSE_NO_SEAT"    # seat pool depleted: every seat leased to a live worker
REFUSE_AT_CAP = "REFUSE_AT_CAP"      # live workers already at/over the operator/dos cap
REFUSE_INSPECT = "REFUSE_INSPECT"    # a check could not run (fail-safe → refuse)
REFUSE_WEEKLY_CAPPED = "REFUSE_WEEKLY_CAPPED"  # routed account hit a weekly-limit 429; cooling until reset (#2610)
REFUSE_FAK_BIN_STALE = "REFUSE_FAK_BIN_STALE"
REFUSE_FAK_BIN_PROVENANCE = "REFUSE_FAK_BIN_PROVENANCE"
REFUSE_SYSTEM_COMMIT_HEADROOM = "REFUSE_SYSTEM_COMMIT_HEADROOM"

SYSTEM_COMMIT_HEADROOM_REASON = "SYSTEM_COMMIT_HEADROOM"
DEFAULT_SYSTEM_COMMIT_HEADROOM_BYTES = 16 << 30

def _env_pos_int(name: str, default: int) -> int:
    """A positive-int env override, falling back to ``default`` on unset/garbage.

    The knobs read through this are the DYNAMIC half of the cap math: the ceiling
    and the per-worker host budgets assume a box dedicated to the fleet, so an
    operator on a shared or measured box retunes them per host via env instead of
    a code change. The Go mirror (internal/dispatchtick.envPosInt) reads the same
    names with the same tolerant contract."""
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            val = int(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return default


# Operator's *aspirational* ceiling on simultaneous live dispatch workers — NOT the
# safety bound. The real DoS proof is the adaptive cap below: min(this, host_cap,
# seats). host_cap (#1337) auto-throttles to the box's live cores/RAM/thread
# headroom; the seat pool (#1336) hard-bounds at bounded account session slots so a
# spawn can never overbook a rate-limit pool. Darwin defaults to 30; other hosts
# retain 20. The static ceiling's only job is to sit ABOVE the adaptive gates —
# which can only LOWER the effective cap — so concurrency rises to what the box and
# account pool can actually carry and no further. FAK_MAX_WORKERS retunes the
# fleet-wide ceiling per host without a code change.
def _built_in_max_workers(platform_name: str = sys.platform) -> int:
    return 30 if platform_name == "darwin" else 20


DEFAULT_MAX_WORKERS = _env_pos_int("FAK_MAX_WORKERS", _built_in_max_workers())
DEFAULT_CODEX_OAUTH_SESSIONS = _env_pos_int("FAK_CODEX_OAUTH_SESSIONS", 10)


def required_system_commit_headroom_bytes() -> int:
    """Return the running guard's positive-MiB system commit reserve."""
    raw = os.environ.get("FAK_SYSTEM_COMMIT_HEADROOM_MB", "").strip()
    if not raw.isdecimal():
        return DEFAULT_SYSTEM_COMMIT_HEADROOM_BYTES
    mb = int(raw)
    if mb == 0 or mb > ((1 << 64) - 1) >> 20:
        return DEFAULT_SYSTEM_COMMIT_HEADROOM_BYTES
    return mb << 20


def evaluate_system_commit_headroom(*, system_bytes: int, system_limit: int,
                                    required_bytes: int) -> dict[str, Any]:
    """Side-effect-free mirror of procguard.EvaluateSystemCommitHeadroom.

    The JSON shape is deliberately metric-named so launch receipts cannot confuse
    operating-system commit with free physical RAM.
    """
    observed = max(0, system_limit - system_bytes)
    supported = system_limit > 0
    refuse = supported and required_bytes > 0 and observed <= required_bytes
    return {
        "supported": supported,
        "ok": not refuse,
        "reason": SYSTEM_COMMIT_HEADROOM_REASON if refuse else "",
        "observed_bytes": observed,
        "required_bytes": required_bytes,
        "system_commit_bytes": system_bytes,
        "system_commit_limit": system_limit,
    }


def _windows_system_commit_snapshot() -> tuple[int, int]:
    """Read the same GetPerformanceInfo counters as procguard's Windows guard."""
    class PerformanceInformation(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_size_t),
            ("commit_total", ctypes.c_size_t),
            ("commit_limit", ctypes.c_size_t),
            ("commit_peak", ctypes.c_size_t),
            ("physical_total", ctypes.c_size_t),
            ("physical_available", ctypes.c_size_t),
            ("system_cache", ctypes.c_size_t),
            ("kernel_total", ctypes.c_size_t),
            ("kernel_paged", ctypes.c_size_t),
            ("kernel_nonpaged", ctypes.c_size_t),
            ("page_size", ctypes.c_size_t),
            ("handle_count", ctypes.c_uint32),
            ("process_count", ctypes.c_uint32),
            ("thread_count", ctypes.c_uint32),
        ]
    info = PerformanceInformation()
    info.cb = ctypes.sizeof(info)
    get_performance_info = ctypes.WinDLL("psapi.dll", use_last_error=True).GetPerformanceInfo
    get_performance_info.argtypes = [ctypes.POINTER(PerformanceInformation), ctypes.c_uint32]
    get_performance_info.restype = ctypes.c_int
    if not get_performance_info(ctypes.byref(info), info.cb):
        raise OSError(ctypes.get_last_error(), "GetPerformanceInfo failed")
    return int(info.commit_total * info.page_size), int(info.commit_limit * info.page_size)


def system_commit_headroom_check() -> dict[str, Any]:
    """Collect and evaluate launch-time system commit headroom without side effects."""
    required = required_system_commit_headroom_bytes()
    if os.name != "nt":
        return {"supported": False, "ok": True, "reason": "",
                "observed_bytes": 0, "required_bytes": required,
                "system_commit_bytes": 0, "system_commit_limit": 0}
    try:
        used, limit = _windows_system_commit_snapshot()
    except (OSError, AttributeError, ValueError) as exc:
        return {"supported": True, "ok": False, "error": str(exc),
                "reason": "", "observed_bytes": 0, "required_bytes": required,
                "system_commit_bytes": 0, "system_commit_limit": 0}
    return evaluate_system_commit_headroom(system_bytes=used, system_limit=limit,
                                           required_bytes=required)

# A live dispatch worker's command line carries this marker (dispatch_worker.py
# launches `claude -p ... /dos-kernel:dos-dispatch-loop --lane X`). Used to count
# real OS processes, the honest DoS signal the kernel's lease count can miss.
WORKER_CMD_MARKER = "dos-dispatch-loop"
ISSUE_RESOLVE_CMD_MARKER = "resolve GitHub issue #"
WORKER_CMD_MARKERS = (WORKER_CMD_MARKER, ISSUE_RESOLVE_CMD_MARKER)
RUNS_DIRNAME = ".dispatch-runs"
# tools/launch_goal_detached.ps1 drops one pid breadcrumb per detached /goal worker
# (`<pointer-tag>-<yyyyMMdd>-<HHmmss>.pid`, next to the worker's logs) under the
# workspace's goal-run dir. That breadcrumb is the ONLY launch-time witness for a
# stdin-fed worker, whose command line carries no scannable marker (#2226).
GOAL_RUNS_DIRNAME = ".goal-runs"
_GOAL_PID_RE = re.compile(r".+-\d{8}-\d{6}\.pid$")
# Both worker kinds occupy a seat: resolve-<N> (issue resolution) and
# repair-<N> (contract repair, spawned by the same dispatcher when its whole
# scan window fails the issue-contract gate). Counting only resolve-*.pid here
# would let repair workers ride outside the cap.
_RESOLVE_PID_RE = re.compile(r"(?:resolve|repair)-\d+-\d{8}-\d{6}\.pid$")
_SIDECAR_CREATE_BEFORE_WINDOW_SECONDS = 5 * 60
_SIDECAR_CREATE_AFTER_SLOP_SECONDS = 10
# Process names a real dispatch worker actually runs as. The create-time-window
# fallback (used only when the OS hid the cmdline) may ONLY count a sidecar pid
# whose image is one of these — otherwise a recycled pid that merely SHARES a
# claude/opencode backend image, or worse a generic shell (cmd.exe, powershell,
# bash) spawned in the same minute, gets miscounted as a live worker and pins the
# dispatcher at cap against ghosts. A bare shell image NEVER counts; only the
# named agent backends do, and even then only inside the spawn window.
_WORKER_BACKEND_IMAGES = ("claude", "opencode", "codex", "node")
# A guarded issue worker's root process is ``fak guard -- <backend> ...``. The
# backend child may be visible too, but the `.pid` sidecar records the root PID,
# so the sidecar liveness fallback must accept a cmdline-hidden `fak.exe` inside
# the spawn window. Keep this separate from `_WORKER_BACKEND_IMAGES` so a generic
# command-line scan still requires an actual agent backend image.
_SIDECAR_WORKER_IMAGES = (*_WORKER_BACKEND_IMAGES, "fak")


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _py() -> str:
    return sys.executable or "python"


def run_json(cmd: list[str], cwd: Path, timeout: int = 90,
             ok_codes: set[int] | None = None) -> dict[str, Any]:
    """Run a helper and parse its JSON. A failure to run is recorded, never raised
    — a missing check is a FAILING check (fail-safe), so the caller refuses."""
    ok_codes = ok_codes if ok_codes is not None else {0}
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout,
                              creationflags=_no_window_creationflags())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"_error": str(exc), "_cmd": cmd, "_returncode": None}
    payload = _last_json(proc.stdout)
    payload["_returncode"] = proc.returncode
    payload["_cmd"] = cmd
    if proc.returncode not in ok_codes and "_error" not in payload:
        payload["_error"] = (proc.stderr or proc.stdout or "").strip()[-500:] or \
            f"returncode {proc.returncode}"
    return payload


def _last_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def host_check(root: Path, *, max_threads: int | None = None) -> dict[str, Any]:
    """proc_resource_guard.py CLEAN ⇒ safe. Only an ACTIONABLE (non-protected)
    runaway/orphan ⇒ unsafe. A protected process breaching a threshold (e.g. the
    operator's own terminal, whose thread count scales with open panes) is
    surfaced as advisory, never a refusal: the guard's own ``ok`` verdict already
    excludes it as non-actionable, and its reaper refuses the kill — so refusing
    the spawn on it wedges every dispatch behind a recovery step ("reap") that is
    impossible by design (#2227). The carve-out is for FOREIGN protected
    processes only (#2252): a flagged process whose image is a fleet agent
    backend (claude/opencode/node) is fleet-spawned and fleet-reapable, so it
    keeps the hard refusal even if a protected bit reached it — only a process
    the fleet never spawned and can never reap demotes to advisory. Mirrors the
    Go fold's ``ActionableFlaggedCount`` semantics
    (cmd/fak/dispatch_tick_preflight.go)."""
    guard = root / "tools" / "proc_resource_guard.py"
    if not guard.exists():
        return {"safe": False, "error": f"guard not found: {guard}", "flagged": 0}
    cmd = [_py(), str(guard), "--json"]
    if max_threads:
        cmd += ["--max-threads", str(max_threads)]
    doc = run_json(cmd, root, timeout=90, ok_codes={0, 1})
    if doc.get("_error") and not doc.get("schema"):
        return {"safe": False, "error": doc["_error"], "flagged": 0}
    flagged = doc.get("flagged") or []

    def _foreign_protected(row: dict[str, Any]) -> bool:
        # Advisory-eligible = protected AND not a known agent backend image
        # (#2252): a fleet binary is fleet-reapable, so its flag stays actionable.
        return bool(row.get("protected")) and not _is_worker_image(str(row.get("name") or ""))

    actionable = [r for r in flagged if not _foreign_protected(r)]
    protected = [r for r in flagged if _foreign_protected(r)]
    return {"safe": bool(doc.get("ok")) and not actionable,
            "flagged": len(actionable),
            "flagged_names": [str(r.get("name")) for r in actionable][:8],
            "protected_flagged": len(protected),
            "protected_names": [str(r.get("name")) for r in protected][:8]}


def _codex_ambient_account() -> dict[str, Any]:
    """Codex authenticates from a single ambient ``~/.codex`` (ChatGPT) login rather
    than the multi-account switcher, so its 'account check' is just: is codex logged
    in? A present ``~/.codex/auth.json`` ⇒ available, with a synthetic ambient
    account; absent ⇒ not available (the operator must ``codex login``)."""
    home = Path(os.path.expanduser("~")) / ".codex"
    auth = home / "auth.json"
    if auth.exists():
        return {"available": True, "tag": "codex-ambient", "dir": str(home),
                "tier": 1, "model": None, "reason": "ambient ~/.codex login"}
    return {"available": False, "tag": None, "dir": None, "tier": None,
            "model": None, "reason": "no ~/.codex/auth.json — run `codex login`"}


def _fak_command(root: Path, env: dict[str, str] | None = None) -> list[str] | None:
    """Resolve the trusted Go account router without compiling in the live tree."""
    environ = env if env is not None else os.environ
    configured = environ.get("FAK_BIN", "").strip()
    if configured:
        return shlex.split(configured, posix=os.name != "nt")
    # The repository-root binary is a developer artifact and can be older, dirty,
    # or held open by another fleet process. Selecting it here made account policy
    # depend on an uncommitted build while workers were guarded by the installed
    # binary. Prefer the installed PATH artifact; FAK_BIN remains the explicit
    # override for a deliberately pinned build.
    # ``shutil.which`` on Windows may consult the current directory before PATH,
    # even when an explicit PATH is supplied. Walk PATH directly so a root-level
    # ``fak.exe`` cannot masquerade as the installed artifact.
    names = ("fak.exe", "fak") if os.name == "nt" else ("fak",)
    for entry in environ.get("PATH", "").split(os.pathsep):
        directory = Path(entry.strip().strip('"')) if entry.strip() else None
        if directory is None:
            continue
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return [str(candidate)]
    return None


# --- which `fak` build made this decision? (binary provenance) --------------- #
# THREE independent resolvers pick a `fak` binary on ONE dispatch tick, and on a
# dev host they routinely pick THREE DIFFERENT builds:
#
#   preflight_gate  `_fak_command` (above)                  $FAK_BIN -> <root>/fak[.exe] -> PATH
#                   ...runs the account router here, and the SAME repo-root rule
#                   is what `issue_resolve_dispatch._fak_command_prefix` uses to
#                   run the `fak issue contract` admission gate.
#   worker_guard    `dispatch_worker.resolve_fak_bin`       $FAK_BIN -> <ws>/tools/.bin/fak[.exe] -> PATH
#                   ...the binary every dispatched worker is FRONTED by.
#   path            `shutil.which("fak")`                   PATH only
#                   ...what `issue_resolve_dispatch._fak_bin` runs the lease gate with.
#
# Because each rule hits a different candidate BEFORE falling through to PATH,
# they only agree by accident. That is survivable for behaviour drift, but not for
# accountability: when a gate REFUSES, "which build decided that?" was pure
# archaeology. Worse, `<root>/fak.exe` is a hand-build — it is routinely a
# `+uncommitted` working-tree compile, i.e. an admission gate whose verdict comes
# from code no one reviewed and no commit pins.
#
# This block MEASURES that, and nothing else. It deliberately does NOT unify or
# reorder resolution: `resolve_fak_bin` returns None to fail OPEN (launch the
# worker unwrapped) on a host with no fak built, and every probe here is
# stat+`version` only, so a broken/missing/hanging binary degrades to an
# `error` field and never changes a verdict.
FAK_BIN_PROVENANCE_SCHEMA = "fak-bin-provenance/v1"
FAK_BIN_PROVENANCE_FILE = "fak-bin-provenance.json"
# `<path> version` prints `build: <id>[ +uncommitted]` on its second line.
_FAK_BUILD_RE = re.compile(r"^build:\s*(\S+)(?P<dirty>\s+\+uncommitted)?", re.M)
_FAK_VERSION_TIMEOUT_S = 20
# Keyed on (path, size, mtime_ns) so one tick pays at most ONE `version` spawn per
# distinct on-disk build, and a rebuild mid-process invalidates itself.
_FAK_IDENTITY_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


def _fak_build_key(path: str | Path, size: int, mtime_ns: int, *,
                   platform: str | None = None) -> str:
    """Return the guard-inventory key without inventing Windows casing splits."""
    platform = platform or os.name
    name = ntpath.basename(str(path)).casefold() if platform == "nt" else Path(path).name
    return f"{size}-{mtime_ns}-{name}"


def _fak_version_text(path: str) -> tuple[str, str | None]:
    """``(stdout, error)`` from ``<path> version``. Never raises."""
    try:
        proc = subprocess.run([path, "version"], capture_output=True, text=True,
                              timeout=_FAK_VERSION_TIMEOUT_S,
                              creationflags=_no_window_creationflags())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", str(exc)
    if proc.returncode != 0:
        return proc.stdout or "", (proc.stderr or proc.stdout or "").strip()[-200:] \
            or f"returncode {proc.returncode}"
    return proc.stdout or "", None


def fak_build_identity(path: str | Path | None, *,
                       probe: Callable[[str], tuple[str, str | None]] | None = None,
                       ) -> dict[str, Any]:
    """Identity of ONE `fak` binary: absolute path, build id, and clean/dirty.

    ``build_key`` is ``<size>-<mtime_ns>-<basename>``. Windows basenames are
    case-folded because its resolvers may spell one executable as both ``fak.exe``
    and ``fak.EXE``; POSIX remains case-sensitive.

    ``dirty`` is True only when the binary itself reports ``+uncommitted`` — a
    build compiled from an unreviewed working tree. ``None`` means "could not be
    determined" (the binary would not run); it is never coerced to False, because
    silently reporting an unknown build as clean is the exact failure this exists
    to prevent."""
    if not path:
        return {"path": None, "resolved": False, "error": "no fak binary resolved"}
    p = Path(path)
    row: dict[str, Any] = {"path": str(p), "resolved": True}
    try:
        st = p.stat()
        row["size"] = st.st_size
        row["mtime_ns"] = st.st_mtime_ns
        row["build_key"] = _fak_build_key(p, st.st_size, st.st_mtime_ns)
    except OSError as exc:
        row["error"] = str(exc)
        return row
    cache_key = (str(p), int(row["size"]), int(row["mtime_ns"]))
    cached = _FAK_IDENTITY_CACHE.get(cache_key)
    if cached is not None and probe is None:
        return dict(cached)
    text, err = (probe or _fak_version_text)(str(p))
    match = _FAK_BUILD_RE.search(text or "")
    row["build"] = match.group(1) if match else None
    row["dirty"] = bool(match.group("dirty")) if match else None
    if err:
        row["error"] = err
    if probe is None:
        _FAK_IDENTITY_CACHE[cache_key] = dict(row)
    return row


def repository_build_relation(root: Path, build: str) -> dict[str, Any]:
    """Classify one clean build against committed HEAD using Git ancestry."""
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                              timeout=20, check=False,
                              creationflags=_no_window_creationflags())

    head_p = git("rev-parse", "--verify", "HEAD^{commit}")
    head = head_p.stdout.strip() if head_p.returncode == 0 else ""
    observed_p = git("rev-parse", "--verify", f"{build}^{{commit}}") if build else None
    observed = observed_p.stdout.strip() if observed_p and observed_p.returncode == 0 else ""
    row = {"expected_head": head, "observed_build": observed or build,
           "relation": "UNKNOWN"}
    if not head or not observed:
        row["error"] = ((head_p.stderr if not head else observed_p.stderr) or
                        "revision unavailable in repository history").strip()[-200:]
        return row
    row["observed_build"] = observed
    if observed == head:
        row["relation"] = "MATCH"
        return row
    behind = git("merge-base", "--is-ancestor", observed, head)
    if behind.returncode == 0:
        row["relation"] = "BEHIND"
        return row
    if behind.returncode not in (0, 1):
        row["error"] = (behind.stderr or "ancestry unavailable").strip()[-200:]
        return row
    ahead = git("merge-base", "--is-ancestor", head, observed)
    if ahead.returncode == 0:
        row["relation"] = "AHEAD"
    elif ahead.returncode == 1:
        row["relation"] = "DIVERGED"
    else:
        row["error"] = (ahead.stderr or "ancestry unavailable").strip()[-200:]
    return row


def fak_bin_resolutions(root: Path, env: dict[str, str] | None = None,
                        ) -> dict[str, str | None]:
    """resolver name -> the ABSOLUTE `fak` path it picks on this host, right now.

    Each entry CALLS the real resolver rather than restating its rule, so this
    table cannot drift away from what dispatch actually executes. The worker
    resolver is imported lazily and defensively: ``dispatch_preflight`` is
    imported by callers that may not have ``tools/`` on ``sys.path``, and a
    provenance record must never be able to break the spawn gate.

    Caveat worth stating rather than hiding: ``worker_guard`` is evaluated against
    the PREFLIGHT's env, not the child env the worker is finally spawned with. No
    ``worker_env`` builder sets ``FAK_BIN`` today, so the two agree; if one ever
    does, this row becomes a prediction rather than a record."""
    e = env if env is not None else dict(os.environ)
    gate = _fak_command(root, e)
    out: dict[str, str | None] = {
        "preflight_gate": gate[0] if gate else None,
        "worker_guard": None,
        "path": shutil.which("fak"),
    }
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import dispatch_worker  # noqa: PLC0415  (lazy by design; see docstring)
        out["worker_guard"] = dispatch_worker.resolve_fak_bin(root, e)
    except Exception:  # noqa: BLE001  — provenance is advisory; never fail the gate
        out["worker_guard"] = None
    return out


def fak_bin_provenance(root: Path, env: dict[str, str] | None = None, *,
                       identity: Callable[..., dict[str, Any]] | None = None,
                       ) -> dict[str, Any]:
    """The full resolver -> binary table, plus the two facts worth alarming on:

    ``agree``  — do all resolvers that resolved AT ALL point at one build_key?
                 False means a single tick spans multiple disagreeing builds.
    ``dirty``  — the resolver names whose binary self-reports ``+uncommitted``,
                 i.e. a gate about to be decided by unreviewed code.
    """
    ident = identity or fak_build_identity
    resolutions = fak_bin_resolutions(root, env)
    rows = {name: ident(path) for name, path in resolutions.items()}
    # The stat key distinguishes on-disk candidates; the reported revision catches
    # the rarer same-metadata rebuild. Unknown revisions retain the stat-only key.
    keys = {(r.get("build_key"), r.get("build"))
            for r in rows.values() if r.get("build_key")}
    builds = sorted({str(r["build"]) for r in rows.values() if r.get("build")})
    resolved_count = sum(1 for r in rows.values() if r.get("resolved"))
    override = str((env if env is not None else os.environ).get(
        "FAK_PREFLIGHT_ALLOW_BIN_SKEW", "")).strip().lower() in ("1", "true", "yes", "on")
    repository = {"expected_head": "", "observed_build": builds[0] if len(builds) == 1 else "",
                  "relation": "UNRESOLVED" if any(not r.get("resolved") for r in rows.values()) else "UNKNOWN"}
    if len(builds) == 1 and len(keys) <= 1 and not any(r.get("dirty") for r in rows.values()):
        repository = repository_build_relation(root, builds[0])
    return {
        "schema": FAK_BIN_PROVENANCE_SCHEMA,
        "resolvers": rows,
        "distinct_builds": len(keys),
        "builds": builds,
        # One binary (or none resolvable) => nothing to disagree about.
        "agree": len(keys) <= 1,
        "dirty": sorted(n for n, r in rows.items() if r.get("dirty")),
        "unresolved": sorted(n for n, r in rows.items() if not r.get("resolved")),
        "resolved_count": resolved_count,
        "expected_head": repository.get("expected_head"),
        "observed_build": repository.get("observed_build"),
        "repository_relation": repository.get("relation"),
        "repository_error": repository.get("error"),
        "historical_override": override,
    }


def fak_bin_warnings(prov: dict[str, Any]) -> list[str]:
    """Operator-facing lines for a provenance table. EMPTY when there is nothing
    to say — a fleet whose resolvers agree on one clean build stays silent."""
    rows = prov.get("resolvers") or {}
    lines: list[str] = []
    for name in prov.get("dirty") or []:
        row = rows.get(name) or {}
        lines.append(
            f"DIRTY_FAK_BIN: gate '{name}' will execute {row.get('path')} "
            f"(build {row.get('build')} +uncommitted) — this gate's verdict comes "
            "from an unreviewed working-tree build that no commit pins")
    if not prov.get("agree"):
        detail = ", ".join(
            f"{n}={(rows.get(n) or {}).get('build') or '?'}@{(rows.get(n) or {}).get('path')}"
            for n in sorted(rows) if (rows.get(n) or {}).get("resolved"))
        lines.append(
            f"FAK_BIN_DISAGREEMENT: {prov.get('distinct_builds')} different `fak` "
            f"builds are in play on one tick — {detail}")
    return lines


def repository_binary_refusal(prov: dict[str, Any]) -> tuple[str, str] | None:
    """Return the typed repository-freshness refusal, before launch side effects."""
    if int(prov.get("resolved_count") or 0) < 2 or not prov.get("agree") or prov.get("dirty"):
        return None
    relation = str(prov.get("repository_relation") or "").upper()
    if relation in ("", "MATCH", "UNRESOLVED"):
        return None
    observed = str(prov.get("observed_build") or "unknown")[:12]
    expected = str(prov.get("expected_head") or "unknown")[:12]
    recovery = "run `fak self-update --force --root .` and retry"
    if relation == "BEHIND":
        return (REFUSE_FAK_BIN_STALE,
                f"installed fak build {observed} is behind committed HEAD {expected}; {recovery}")
    return (REFUSE_FAK_BIN_PROVENANCE,
            f"installed fak build {observed} has repository relation {relation} to committed "
            f"HEAD {expected}; {recovery}")


def record_fak_bin_provenance(root: Path, prov: dict[str, Any],
                              *, now: str | None = None) -> Path | None:
    """Persist the table to ``.dispatch-runs/fak-bin-provenance.json``, keyed on the
    resolver->build_key FINGERPRINT so a disagreement that happened at 03:00 is still
    answerable at 09:00 (a last-write-wins snapshot only ever answers "right now").
    Each key carries ``first_utc``/``last_utc``/``ticks``; the file therefore stays
    bounded by the number of DISTINCT configurations, not by tick count.

    Best-effort and never raises: a provenance record that could fail a spawn gate
    would be worse than the blindness it replaces."""
    rows = prov.get("resolvers") or {}
    fingerprint = "|".join(
        f"{n}={(rows.get(n) or {}).get('build_key') or '-'}" for n in sorted(rows))
    stamp = now or dt.datetime.now(dt.timezone.utc).isoformat()
    path = Path(root) / ".dispatch-runs" / FAK_BIN_PROVENANCE_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(doc, dict):
                doc = {}
        except (OSError, ValueError):
            doc = {}
        prior = doc.get(fingerprint) if isinstance(doc.get(fingerprint), dict) else {}
        doc[fingerprint] = {
            "first_utc": prior.get("first_utc") or stamp,
            "last_utc": stamp,
            "ticks": int(prior.get("ticks") or 0) + 1,
            "agree": prov.get("agree"),
            "dirty": prov.get("dirty"),
            "builds": prov.get("builds"),
            "expected_head": prov.get("expected_head"),
            "observed_build": prov.get("observed_build"),
            "repository_relation": prov.get("repository_relation"),
            "repository_error": prov.get("repository_error"),
            "historical_override": prov.get("historical_override"),
            "resolvers": {n: {k: r.get(k) for k in ("path", "build", "dirty", "build_key")}
                          for n, r in rows.items()},
        }
        doc["schema"] = FAK_BIN_PROVENANCE_SCHEMA
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
        return path
    except OSError:
        return None


def account_check(root: Path, *, work_kind: str, product: str) -> dict[str, Any]:
    """Native account route returns a credential-ready worker seat, or fails closed.

    The Go router owns refreshed ``login_status``/``can_serve`` truth. Do not use
    the legacy Python roster here: its enabled/available flags can lag an empty or
    expired OAuth token, making preflight disagree with the spawn guard.

    Codex is the exception: it has no switcher roster (single ambient login), so its
    availability is read straight from ``~/.codex`` rather than the switcher."""
    if product == "codex":
        return _codex_ambient_account()
    fak = _fak_command(root)
    if not fak:
        return {"available": False, "error": "fak account router not found"}
    cmd = [*fak, "fleet-accounts", "resolve", "--product", product,
           "--task", work_kind, "--work-kind", work_kind, "--json"]
    doc = run_json(cmd, root, timeout=45, ok_codes={0, 1})
    if doc.get("_error") and "ok" not in doc:
        return {"available": False, "error": doc["_error"]}
    # ``fleet-accounts resolve`` intentionally returns a flat credential-safe row:
    # token material never crosses this boundary, while login/can-serve does.
    login_status = str(doc.get("login_status") or "").strip().lower()
    can_serve = doc.get("can_serve") is True
    ready = bool(doc.get("ok")) and bool(doc.get("tag")) and login_status == "ready" and can_serve
    reason = doc.get("reason")
    if doc.get("tag") and not ready:
        reason = (doc.get("block_reason") or
                  f"account login_status={login_status or 'unknown'} can_serve={can_serve}")
    return {"available": ready,
            "tag": doc.get("tag"), "dir": doc.get("config_dir"),
            "tier": doc.get("selected_tier"), "model": doc.get("model"),
            "login_status": login_status or None, "can_serve": can_serve,
            "reason": reason, "blocked": []}

def seat_check(root: Path, *, product: str) -> dict[str, Any]:
    """fleet_accounts.py seats ⇒ the explicit seat-pool size for this product.

    A SEAT is one distinct routable worker POOL (one Anthropic account / rate limit);
    the seat count M is the real binding constraint on concurrency, so it folds into the
    cap below as another min() term and a depleted pool yields a typed REFUSE_NO_SEAT —
    distinct from REFUSE_NO_ACCOUNT (a throttled/auth tier). Returns
    ``{total, free, leased, depleted}``. ``total`` is None when the pool view could not
    run, in which case the seat fold is SKIPPED and the existing host/account/cap gates
    govern unchanged — fail-OPEN on the seat shaping (the cap still bounds the fleet),
    never fail-closed on a missing view.

    Codex is the one-product exception to the switcher roster: it uses one ambient
    ChatGPT/OAuth login rather than switcher-managed account dirs. The process census is
    telemetry only here: an attended coordinator is not a managed worker lease. Validated
    worker sidecars are folded into ``free`` later by
    :func:`account_unattributed_live_slots`, through the same managed-worker count that
    enforces the process cap."""
    if product == "codex":
        total = DEFAULT_CODEX_OAUTH_SESSIONS
        ambient_live = len(ambient_codex_pids())
        return {"total": total, "free": total, "leased": 0,
                "depleted": False, "ambient_live": ambient_live,
                "reason": (f"codex ambient OAuth telemetry ({ambient_live} attended "
                           f"session(s)); {total} managed session slots")}
    sw = root / "tools" / "fleet_accounts.py"
    if not sw.exists():
        return {"total": None, "error": f"switcher not found: {sw}"}
    cmd = [_py(), str(sw), "seats", "--product", product, "--json"]
    doc = run_json(cmd, root, timeout=45, ok_codes={0, 1})
    if doc.get("_error") and "total_seats" not in doc:
        return {"total": None, "error": doc["_error"]}
    return {"total": _int(doc.get("total_seats")),
            "free": _int(doc.get("free_seats")),
            "leased": _int(doc.get("leased_seats")),
            "depleted": bool(doc.get("depleted"))}


# --- weekly-limit seat cooldown (#2610) ------------------------------------- #
# A weekly-limit 429 (`kind=weekly_limit`) is NOT a stale credential (#2059/#2075):
# the login is valid, but the provider seat is temporarily quota-capped until its
# announced reset window. The always-on resolve dispatcher detects that banner and
# persists a hold in `.dispatch-runs/account-cap-<product>[-<account>].json`
# (issue_resolve_dispatch.check_weekly_cap, which parses the guard's
# `announced_wait=<dur>` window); dispatch_status reads the SAME file to render the
# WEEKLY_CAPPED card. This preflight honors that hold so a routed-but-capped seat is
# NOT re-offered for a fresh spawn — and, because issue_dispatch.py gates on this
# preflight verdict, it stops offering the seat too. Stdlib-only + FAIL-OPEN: any
# read/parse error resolves to "not capped", so the gate can only ever ADD a refusal,
# never wedge spawning on a malformed sidecar. The hold self-expires at its announced
# `until`, so a seat is cooled, never permanently walled.

def _naive_utc_from_iso(text: str) -> dt.datetime | None:
    """Parse a persisted `until` ISO stamp (a trailing 'Z' or an explicit offset) to a
    naive-UTC datetime, or None. Mirrors the write side's `<naive-iso>+'Z'` and the
    tolerant read in dispatch_status.read_active_weekly_cap."""
    s = (text or "").strip()
    if not s:
        return None
    try:
        parsed = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def weekly_cap_check(root: Path, *, product: str, account_tag: str | None,
                     now_ts: float | None = None) -> dict[str, Any]:
    """Is the routed ``account_tag`` under an active weekly-limit cooldown right now?

    Reads the persisted holds the resolve dispatcher writes
    (`.dispatch-runs/account-cap-*.json`) and returns the soonest-expiring UNEXPIRED
    hold matching this product+account as ``{"capped": True, "until", "reset_text",
    "kind", "account", "evidence_log"}`` — else ``{"capped": False}``. A hold whose
    ``account`` is null is a legacy generic hold and matches any tag (mirrors
    dispatch_status.read_active_weekly_cap); a hold for a DIFFERENT product/account is
    ignored, so a capped claude seat never walls an uncapped opencode one. An expired
    hold is ignored — the cooldown self-expires at the announced window, so a seat is
    cooled, not permanently walled. FAIL-OPEN: any error → not capped."""
    try:
        runs_dir = root / RUNS_DIRNAME
        if not runs_dir.is_dir():
            return {"capped": False}
        now = (dt.datetime(1970, 1, 1) + dt.timedelta(seconds=now_ts)
               if now_ts is not None
               else dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))
        best: tuple[dict[str, Any], dt.datetime] | None = None
        for path in runs_dir.glob("account-cap-*.json"):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(state, dict):
                continue
            if state.get("product") not in (None, "", product):
                continue
            if account_tag and state.get("account") not in (None, "", account_tag):
                continue
            until = _naive_utc_from_iso(str(state.get("until") or ""))
            if until is None or now >= until:
                continue
            if best is None or until < best[1]:
                best = (state, until)
        if best is None:
            return {"capped": False}
        state = best[0]
        return {"capped": True, "until": state.get("until"),
                "reset_text": state.get("reset_text") or "",
                "kind": state.get("kind") or "weekly",
                "account": state.get("account"),
                "evidence_log": state.get("evidence_log") or ""}
    except Exception:
        return {"capped": False}


def kernel_alive(root: Path) -> dict[str, Any]:
    """`dos loop` alive = the kernel's lease-backed live worker count + target."""
    doc = run_json(["dos", "loop", "--workspace", str(root), "--json"], root,
                   timeout=60, ok_codes=set(range(0, 16)))
    if doc.get("_error") and "alive" not in doc:
        return {"alive": None, "target": None, "error": doc["_error"]}
    return {"alive": _int(doc.get("alive")), "target": _int(doc.get("target")),
            "verdict": doc.get("verdict")}


def _is_worker_cmdline(cmdline: str) -> bool:
    low = (cmdline or "").lower()
    return any(marker.lower() in low for marker in WORKER_CMD_MARKERS)


def _is_worker_image(image: str) -> bool:
    """True iff ``image`` (a process name or argv[0]) is a real agent backend.

    The marker substrings (``dos-dispatch-loop``, ``resolve GitHub issue #``)
    routinely appear on the command line of a *generic shell* — a bash/powershell
    that merely greps for, logs, or dispatches a worker, or an operator inspecting
    the fleet. Those are NOT workers; counting them inflates the live count and
    pins the dispatcher at cap against ghosts (the exact false-positive the
    create-time-window fallback already guards against via _WORKER_BACKEND_IMAGES,
    but which the primary cmdline scan did not). Require the image be one of the
    named backends so a shell that just mentions the marker never counts.
    """
    base = os.path.basename((image or "").strip().strip('"')).lower()
    if base.endswith(".exe"):
        base = base[:-4]
    return any(base == img or base.startswith(img) for img in _WORKER_BACKEND_IMAGES)


def _collapse_descendant_pids(pids: set[int], parent_by_pid: dict[int, int | None]) -> set[int]:
    """Collapse wrapper/child matches to one worker root.

    Windows .cmd launchers can leave the full prompt on both the wrapper process
    and the real backend child (for example cmd.exe -> opencode.exe). Both match
    the issue prompt marker, but they are one worker tree and must consume one cap
    slot. Keep the highest marked ancestor and drop marked descendants.
    """
    roots: set[int] = set()
    for pid in pids:
        cur = pid
        seen = {pid}
        descendant = False
        while True:
            parent = parent_by_pid.get(cur)
            if parent is None or parent <= 0 or parent in seen:
                break
            if parent in pids:
                descendant = True
                break
            seen.add(parent)
            cur = parent
        if not descendant:
            roots.add(pid)
    return roots


def _cmdline_from_process_row(row: dict[str, Any]) -> str:
    cmdline = row.get("cmdline")
    if cmdline is None:
        cmdline = row.get("CommandLine")
    if isinstance(cmdline, (list, tuple)):
        return " ".join(str(part) for part in cmdline)
    return str(cmdline or "")


def _worker_pids_from_process_rows(
    rows: list[dict[str, Any]],
    *,
    product: str | None = None,
) -> set[int]:
    pids: set[int] = set()
    parent_by_pid: dict[int, int | None] = {}
    for row in rows:
        pid = _int(row.get("pid"), _int(row.get("ProcessId")))
        if pid is None:
            continue
        parent_by_pid[pid] = _int(row.get("ppid"), _int(row.get("ParentProcessId"), 0))
        name = str(row.get("name") or row.get("Name") or "")
        if not _is_worker_image(name):
            continue
        if product is not None and not _image_matches_product(name, product):
            continue
        if _is_worker_cmdline(_cmdline_from_process_row(row)):
            pids.add(pid)
    return _collapse_descendant_pids(pids, parent_by_pid)


def _cmdline_worker_pids_windows(product: str | None = None) -> set[int] | None:
    if os.name != "nt":
        return None
    filter_expr = " OR ".join(
        f"Name = '{img}.exe'" for img in _WORKER_BACKEND_IMAGES)
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             f"Get-CimInstance Win32_Process -Filter \"{filter_expr}\" | "
             "Select-Object ProcessId,ParentProcessId,Name,CommandLine | "
             "ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=5,
            creationflags=_no_window_creationflags(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    text = (proc.stdout or "").strip()
    if not text:
        return set()
    try:
        obj = json.loads(text)
    except ValueError:
        return None
    rows = obj if isinstance(obj, list) else [obj]
    return _worker_pids_from_process_rows(
        [row for row in rows if isinstance(row, dict)], product=product)


def _cmdline_worker_pids(product: str | None = None) -> set[int]:
    """OS-level worker pids by command-line marker.

    This catches the generic DOS-loop worker and the issue-resolution workers
    whose prompt is still on the top-level command line. Best effort: an
    inaccessible process table returns an empty set so the pid-sidecar witness
    below can still govern the issue-dispatch path.

    When ``product`` is given, only workers whose process image is that product's
    backend are kept (``claude`` -> claude*, ``opencode`` -> opencode*), so a
    product-scoped cap counts its OWN generic dos-dispatch-loop workers without a
    sibling product's workers pinning it. These generic workers carry no
    ``.backend`` sidecar, so the process image is the only pool signal available.
    """
    if os.name == "nt":
        cim = _cmdline_worker_pids_windows(product=product)
        if cim is not None:
            return cim
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None
    if psutil is not None:
        rows: list[dict[str, Any]] = []
        for p in psutil.process_iter(["name", "ppid"]):
            try:
                name = p.info.get("name") or ""
            except (psutil.Error, TypeError):
                continue
            # Marker on the cmdline AND a real backend image — a shell that merely
            # mentions the marker (a grep, a launcher, an operator inspecting) is
            # not a live worker and must not consume a cap slot.
            if not _is_worker_image(name):
                continue
            if product is not None and not _image_matches_product(name, product):
                continue
            try:
                cl = " ".join(p.cmdline() or [])
            except psutil.Error:
                continue
            rows.append({
                "pid": int(p.pid),
                "ppid": int(p.info.get("ppid") or 0),
                "name": name,
                "cmdline": cl,
            })
        return _worker_pids_from_process_rows(rows, product=product)
    # Fallback when psutil is absent. wmic.exe is removed on Win11 24H2+ (build
    # 26200), so use the supported CIM API; Win32_Process.CommandLine carries the
    # full argv (incl. the prompt/marker), so this is an honest worker count, not 0.
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "$all = @(Get-CimInstance Win32_Process); "
                 "$all | Where-Object { $_.CommandLine -match 'dos-dispatch-loop|resolve GitHub issue #' } | "
                 "ForEach-Object { \"$($_.ProcessId),$($_.ParentProcessId),$($_.Name)\" }"],
                capture_output=True, text=True, timeout=30,
                creationflags=_no_window_creationflags()).stdout
            pids: set[int] = set()
            parent_by_pid: dict[int, int | None] = {}
            for ln in out.splitlines():
                parts = [p.strip() for p in ln.split(",", 2)]
                if not parts or not parts[0].isdigit():
                    continue
                # require a real backend image — drop a shell (powershell/bash/cmd)
                # that only mentions the marker, the cap-pinning false positive.
                image = parts[2] if len(parts) > 2 else ""
                if not _is_worker_image(image):
                    continue
                if product is not None and not _image_matches_product(image, product):
                    continue
                pid = int(parts[0])
                parent = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                pids.add(pid)
                parent_by_pid[pid] = parent
            return _collapse_descendant_pids(pids, parent_by_pid)
        except (OSError, subprocess.TimeoutExpired):
            return set()
    try:
        out = subprocess.run(["pgrep", "-fa", "|".join(WORKER_CMD_MARKERS)],
                             capture_output=True, text=True, timeout=20).stdout
        pids = set()
        for ln in out.splitlines():
            if not ln.strip() or not _is_worker_cmdline(ln):
                continue
            # `pgrep -fa` => "<pid> <argv0> <argv1...>"; require argv0 be a real
            # backend image so a shell that only mentions the marker never counts.
            toks = ln.split(None, 2)
            if len(toks) < 2 or not toks[0].isdigit():
                continue
            if not _is_worker_image(toks[1]):
                continue
            if product is not None and not _image_matches_product(toks[1], product):
                continue
            pids.add(int(toks[0]))
        return pids
    except (OSError, subprocess.TimeoutExpired):
        return set()


def _process_name_stem(name: Any) -> str:
    base = os.path.basename(str(name or "").strip().strip('"')).lower()
    if base.endswith(".exe") or base.endswith(".cmd") or base.endswith(".bat"):
        base = base.rsplit(".", 1)[0]
    return base


def _is_codex_native_image(name: Any) -> bool:
    """True for the native Codex CLI process image (codex/codex.exe)."""
    return _process_name_stem(name) == "codex"


def _is_codex_node_wrapper(name: Any, cmdline: Any) -> bool:
    """True for the npm node wrapper that launches the native Codex binary."""
    if _process_name_stem(name) != "node":
        return False
    low = str(cmdline or "").replace("\\", "/").lower()
    return "@openai/codex" in low or "codex/bin/codex.js" in low


def _codex_process_pids_from_rows(rows: list[dict[str, Any]]) -> set[int]:
    """Collapse live Codex process rows to one PID per Codex session.

    On Windows the observed shape is ``node.exe .../@openai/codex/bin/codex.js`` →
    ``codex.exe``. Count the native child when present, and count the node wrapper
    only while the native child has not appeared yet. That gives the preflight an
    honest "ambient Codex seat is in use" signal without double-counting one session.
    """
    native: set[int] = set()
    wrappers: set[int] = set()
    parent: dict[int, int] = {}
    for row in rows:
        try:
            pid = int(row.get("pid") or row.get("ProcessId") or 0)
        except (TypeError, ValueError):
            continue
        if pid <= 0:
            continue
        try:
            ppid = int(row.get("ppid") or row.get("ParentProcessId") or 0)
        except (TypeError, ValueError):
            ppid = 0
        name = row.get("name") or row.get("Name") or ""
        cmdline = row.get("cmdline") or row.get("CommandLine") or ""
        parent[pid] = ppid
        if _is_codex_native_image(name):
            native.add(pid)
        elif _is_codex_node_wrapper(name, cmdline):
            wrappers.add(pid)
    wrappers_with_native_child = {parent.get(pid, 0) for pid in native}
    return native | (wrappers - wrappers_with_native_child)


def _pid_has_ancestor(pid: int, parents: dict[int, int], ancestors: set[int]) -> bool:
    """True when pid's known process ancestry reaches any ancestor pid."""
    seen: set[int] = set()
    cur = pid
    while cur > 0 and cur not in seen:
        seen.add(cur)
        parent = parents.get(cur, 0)
        if parent in ancestors:
            return True
        cur = parent
    return False


def _codex_process_rows_psutil() -> list[dict[str, Any]] | None:
    if os.name == "nt":
        return None
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    rows: list[dict[str, Any]] = []
    for p in psutil.process_iter(["pid", "ppid", "name"]):
        try:
            name = str(p.info.get("name") or "")
            if _is_codex_native_image(name):
                rows.append({"pid": int(p.pid), "ppid": int(p.info.get("ppid") or 0),
                             "name": name, "cmdline": ""})
                continue
            if _process_name_stem(name) != "node":
                continue
            cmdline = " ".join(p.cmdline() or [])
            if not _is_codex_node_wrapper(name, cmdline):
                continue
            rows.append({"pid": int(p.pid), "ppid": int(p.info.get("ppid") or 0),
                         "name": name, "cmdline": cmdline})
        except (psutil.Error, TypeError, ValueError):
            continue
    return rows


def _codex_process_rows_windows() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "$rows = @(Get-CimInstance Win32_Process "
             "-Filter \"Name = 'codex.exe' OR Name = 'node.exe'\" | "
             "Select-Object @{n='pid';e={$_.ProcessId}},"
             "@{n='ppid';e={$_.ParentProcessId}},"
             "@{n='name';e={$_.Name}},@{n='cmdline';e={$_.CommandLine}}); "
             "$rows | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=10,
            creationflags=_no_window_creationflags(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    text = (proc.stdout or "").strip()
    if not text:
        return []
    try:
        doc = json.loads(text)
    except ValueError:
        return []
    if isinstance(doc, dict):
        return [doc]
    if isinstance(doc, list):
        return [r for r in doc if isinstance(r, dict)]
    return []


def _codex_process_rows_posix() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(["ps", "-eo", "pid=,ppid=,comm=,args="],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows: list[dict[str, Any]] = []
    for line in (proc.stdout or "").splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        name = parts[2]
        cmdline = parts[3] if len(parts) > 3 else name
        if _is_codex_native_image(name) or _is_codex_node_wrapper(name, cmdline):
            rows.append({"pid": pid, "ppid": ppid, "name": name, "cmdline": cmdline})
    return rows


def _ambient_codex_process_rows() -> list[dict[str, Any]]:
    rows = _codex_process_rows_psutil()
    if rows is None:
        rows = _codex_process_rows_windows() if os.name == "nt" else _codex_process_rows_posix()
    return rows


@functools.lru_cache(maxsize=1)
def ambient_codex_pids() -> set[int]:
    """Live Codex CLI sessions sharing the ambient ChatGPT OAuth bucket.

    This intentionally counts attended Codex sessions too. The background codex
    dispatcher has no separate account pool, so foreground Codex sessions consume
    slots in the same OAuth bucket as background ``codex exec`` workers.
    """
    rows = _ambient_codex_process_rows()
    return _codex_process_pids_from_rows(rows)


def ambient_codex_pids_excluding_sidecar_parents(sidecar_pids: set[int]) -> set[int]:
    """Ambient Codex sessions minus descendants already represented by sidecars.

    On Windows the issue dispatcher records the outer ``cmd.exe /c codex.CMD``
    wrapper PID in its resolve sidecar. That wrapper starts ``node.exe`` which then
    starts the native ``codex.exe``. The ambient Codex scan prefers the native child
    PID. Count that process tree as ONE session: the sidecar already witnesses the
    background worker, so only ambient Codex processes whose ancestry does not reach
    a sidecar get added.
    """
    rows = _ambient_codex_process_rows()
    pids = _codex_process_pids_from_rows(rows)
    parents: dict[int, int] = {}
    for row in rows:
        try:
            pid = int(row.get("pid") or 0)
            ppid = int(row.get("ppid") or 0)
        except (TypeError, ValueError):
            continue
        if pid > 0:
            parents[pid] = ppid
    return {pid for pid in pids if not _pid_has_ancestor(pid, parents, sidecar_pids)}


def _managed_cmdline_pids_excluding_sidecar_trees(sidecar_pids: set[int], *,
                                                   product: str) -> set[int]:
    """Marker-authenticated workers not already represented by a sidecar tree."""
    marker_pids = _cmdline_worker_pids(product=product)
    if not marker_pids or product != "codex" or not sidecar_pids:
        return marker_pids - sidecar_pids
    rows = _ambient_codex_process_rows()
    parents: dict[int, int] = {}
    for row in rows:
        try:
            pid = int(row.get("pid") or 0)
            ppid = int(row.get("ppid") or 0)
        except (TypeError, ValueError):
            continue
        if pid > 0:
            parents[pid] = ppid
    return {
        pid for pid in marker_pids
        if pid not in sidecar_pids and not _pid_has_ancestor(pid, parents, sidecar_pids)
    }


def _parse_process_create_time(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    m = re.match(r"/Date\((\d+)\)/", text)
    if m:
        return int(m.group(1)) / 1000.0
    # WMI DMTF datetime: YYYYMMDDHHMMSS.ffffff+/-UUU, where UUU is offset minutes.
    m = re.match(r"^(\d{14})\.(\d{1,6})([+-])(\d{3})$", text)
    if m:
        base, micros, sign, offset_min = m.groups()
        try:
            naive = dt.datetime.strptime(base, "%Y%m%d%H%M%S")
            micro = int(micros.ljust(6, "0")[:6])
            offset = dt.timedelta(minutes=int(offset_min))
            if sign == "-":
                offset = -offset
            aware = naive.replace(microsecond=micro,
                                  tzinfo=dt.timezone(offset))
            return aware.timestamp()
        except ValueError:
            return None
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.timestamp()


def _probe_cmdline_text(probe: dict[str, Any]) -> str:
    cmdline = probe.get("cmdline")
    if isinstance(cmdline, (list, tuple)):
        return " ".join(str(part) for part in cmdline)
    return str(cmdline or "")


def _psutil_process_probe(pid: int) -> dict[str, Any] | None:
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return {"alive": False}
    except psutil.Error:
        return None
    rec: dict[str, Any] = {"alive": proc.is_running()}
    try:
        rec["create_time"] = float(proc.create_time())
    except psutil.Error:
        pass
    try:
        rec["cmdline"] = proc.cmdline()
    except psutil.Error:
        pass
    try:
        rec["name"] = proc.name()
    except psutil.Error:
        pass
    return rec


def _cim_process_probe(pid: int) -> dict[str, Any] | None:
    if os.name != "nt":
        return None
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             f"$p = Get-CimInstance Win32_Process -Filter \"ProcessId={int(pid)}\"; "
             "if ($null -eq $p) { exit 2 }; "
             "$p | Select-Object ProcessId,Name,CreationDate,CommandLine | "
             "ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=5,
            creationflags=_no_window_creationflags(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode == 2 or not (proc.stdout or "").strip():
        return {"alive": False}
    try:
        obj = json.loads(proc.stdout)
    except ValueError:
        return None
    if isinstance(obj, list):
        obj = obj[0] if obj else {}
    if not isinstance(obj, dict):
        return None
    create_time = _parse_process_create_time(obj.get("CreationDate"))
    return {
        "alive": True,
        "create_time": create_time,
        "cmdline": obj.get("CommandLine") or "",
        "name": obj.get("Name") or "",
    }


def _process_probe(pid: int) -> dict[str, Any]:
    probe = _psutil_process_probe(pid)
    if probe and not probe.get("alive"):
        return probe
    if os.name == "nt" and (
        probe is None
        or probe.get("cmdline") in (None, "")
        or probe.get("create_time") is None
    ):
        cim = _cim_process_probe(pid)
        if cim is not None:
            if probe is None:
                return cim
            merged = dict(probe)
            for key, value in cim.items():
                if merged.get(key) in (None, "", []):
                    merged[key] = value
            return merged
    if probe is not None:
        return probe
    return {"alive": _pid_is_alive(pid)}


def _within_sidecar_spawn_window(create_time: Any, sidecar_mtime: float) -> bool:
    created = _parse_process_create_time(create_time)
    if created is None:
        return False
    if created > sidecar_mtime + _SIDECAR_CREATE_AFTER_SLOP_SECONDS:
        return False
    if sidecar_mtime - created > _SIDECAR_CREATE_BEFORE_WINDOW_SECONDS:
        return False
    return True


def _probe_image_is_worker_backend(probe: dict[str, Any]) -> bool:
    """Does the process image look like a real agent worker backend?

    The honest signal is the cmdline marker; this is the weaker name-only check
    the create-time-window branch leans on when the OS hid the cmdline. A bare
    shell (cmd.exe / powershell / bash) is NEVER a worker, so it is rejected even
    inside the spawn window — that is what stops a recycled shell pid from pinning
    the cap. We match on the image stem so `claude.exe`, `opencode.exe`, `node`
    all qualify. For sidecar-authenticated roots only, `fak` also qualifies
    because guarded workers spawn as `fak guard -- <backend> ...`.
    """
    name = str(probe.get("name") or "").strip().lower()
    if not name:
        return False
    stem = name[:-4] if name.endswith(".exe") else name
    return stem in _SIDECAR_WORKER_IMAGES or stem.startswith("fak-")


def _sidecar_process_matches(pid: int, sidecar_mtime: float, *,
                             probe: Callable[[int], dict[str, Any]] | None = None) -> bool:
    rec = (probe or _process_probe)(pid)
    if not rec.get("alive"):
        return False
    if _is_worker_cmdline(_probe_cmdline_text(rec)):
        return True
    # Fallback for a real worker whose cmdline the OS hid: only an agent-backend
    # image inside the spawn window counts. A recycled shell pid (cmd.exe, etc.)
    # is rejected here even if its create time lands in the window — the
    # create-time coincidence alone is NOT evidence of a live worker.
    if not _probe_image_is_worker_backend(rec):
        return False
    return _within_sidecar_spawn_window(rec.get("create_time"), sidecar_mtime)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 f"Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id"],
                capture_output=True, text=True, timeout=5,
                creationflags=_no_window_creationflags(),
            )
            return proc.returncode == 0 and bool((proc.stdout or "").strip())
        except (OSError, subprocess.TimeoutExpired):
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_resolve_pid_sidecar(pid_file: Path) -> int | None:
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def resolve_sidecar_pid_is_live(
    pid_file: Path,
    *,
    alive: set[int] | None = None,
    probe: Callable[[int], dict[str, Any]] | None = None,
) -> bool:
    pid = _read_resolve_pid_sidecar(pid_file)
    if pid is None:
        return False
    if alive is not None and pid not in alive:
        return False
    try:
        sidecar_mtime = pid_file.stat().st_mtime
    except OSError:
        return False
    return _sidecar_process_matches(pid, sidecar_mtime, probe=probe)


def live_resolve_worker_pids(
    runs_dir: Path,
    *,
    alive: set[int] | None = None,
    probe: Callable[[int], dict[str, Any]] | None = None,
    product: str | None = None,
) -> set[int]:
    """Live dispatch workers from `.dispatch-runs/resolve-*.pid` AND
    `.dispatch-runs/repair-*.pid` — a contract-repair worker burns the same
    account seat a resolution worker does, so both pin the cap.

    When ``product`` is given, only sidecars whose ``.backend`` tag is in that
    product's pool are counted, so a claude worker does not pin the opencode cap.

    `issue_resolve_dispatch.py` is the active always-on issue closer in this
    plan-empty repo, and it writes one pid sidecar per spawned worker. A bare
    live-PID check is not enough on long-running Windows hosts because PIDs get
    reused: the sidecar only counts when the current process either still exposes
    a worker command-line marker or its create time matches the sidecar's spawn
    window.
    """
    if not runs_dir.is_dir():
        return set()
    pids: set[int] = set()
    for pid_file in (*runs_dir.glob("resolve-*.pid"), *runs_dir.glob("repair-*.pid")):
        if not _RESOLVE_PID_RE.match(pid_file.name):
            continue
        pid = _read_resolve_pid_sidecar(pid_file)
        if pid is None:
            continue
        if product is not None and _sidecar_backend(pid_file) not in _product_backends(product):
            continue
        if resolve_sidecar_pid_is_live(pid_file, alive=alive, probe=probe):
            pids.add(pid)
    return pids


# A worker pins one ACCOUNT POOL's cap, not a global one: a claude (opus) worker
# and an opencode (GLM) worker draw on different accounts and rate limits, so they
# must run concurrently up to each pool's own headroom. The pool a sidecar belongs
# to is its `.backend` (claude|opencode), written next to the `.pid`.
_PRODUCT_BACKENDS = {
    "claude": ("claude",),
    "opencode": ("opencode",),
    "codex": ("codex",),
}


def _product_backends(product: str) -> tuple[str, ...]:
    return _PRODUCT_BACKENDS.get(product, (product,))


def _image_matches_product(image: str, product: str) -> bool:
    """True iff process ``image`` is the backend for ``product``.

    Mirrors ``_is_worker_image`` but scopes to ONE product's backend
    (``claude`` -> claude*, ``opencode`` -> opencode*), so a product-scoped
    worker count keeps the generic cmdline-marked dos-dispatch-loop workers that
    belong to this product's pool while a sibling product's workers stay out of
    this pool's cap. A blank/unknown image matches nothing.
    """
    base = _process_name_stem(image)
    if not base:
        return False
    return any(base == img or base.startswith(img) for img in _product_backends(product))


def _sidecar_backend(pid_file: Path) -> str | None:
    """The backend (claude|opencode) a resolve sidecar belongs to, from its
    `.backend` sibling. A missing/unreadable sidecar returns None so a
    product-scoped count treats it conservatively as 'not this product' — a worker
    with no backend tag does not pin a specific pool's cap."""
    try:
        return pid_file.with_suffix(".backend").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def live_goal_worker_pids(
    goal_runs_dir: Path,
    *,
    alive: set[int] | None = None,
    probe: Callable[[int], dict[str, Any]] | None = None,
    product: str | None = None,
) -> set[int]:
    """Live detached /goal workers from the launcher's `.goal-runs/*.pid` breadcrumbs.

    A detached /goal worker (tools/launch_goal_detached.ps1) is fed its goal via
    STDIN — ``claude -p`` with no prompt argument — so its command line carries NO
    scannable marker and the cmdline scan is blind to it until it takes a lane
    lease (#2226). The launcher's pid breadcrumb closes that start-up window: the
    breadcrumb counts while its pid is a live agent-backend process (never a bare
    shell) whose create time sits inside the breadcrumb's spawn window — the same
    pid-reuse guard the resolve sidecars use. A dead pid is simply IGNORED, so a
    stale breadcrumb never inflates the live count and wedges spawning (the
    launcher also sweeps dead breadcrumbs before each spawn, but this scan must
    not depend on that hygiene having run).

    Goal workers write no ``.backend`` sidecar; when ``product`` is given they
    are attributed to a pool by their live process IMAGE
    (``_image_matches_product``), like the generic cmdline-marked workers.
    """
    if not goal_runs_dir.is_dir():
        return set()
    probe_fn = probe or _process_probe
    pids: set[int] = set()
    for pid_file in goal_runs_dir.glob("*.pid"):
        if not _GOAL_PID_RE.match(pid_file.name):
            continue
        pid = _read_resolve_pid_sidecar(pid_file)
        if pid is None:
            continue
        if alive is not None and pid not in alive:
            continue
        try:
            breadcrumb_mtime = pid_file.stat().st_mtime
        except OSError:
            continue
        rec = probe_fn(pid)
        if not rec.get("alive"):
            continue  # worker exited: stale breadcrumb, never counted
        # The goal went in via stdin, so there is usually no cmdline marker to
        # lean on; the backend-image + spawn-window pair is the whole
        # recycled-pid guard. A marker match still counts (belt and braces);
        # anything else must be a real agent image created around the
        # breadcrumb's write time — a recycled shell or later claude never is.
        if not _is_worker_cmdline(_probe_cmdline_text(rec)):
            if not _probe_image_is_worker_backend(rec):
                continue
            if not _within_sidecar_spawn_window(rec.get("create_time"),
                                                breadcrumb_mtime):
                continue
        if product is not None and not _image_matches_product(
                str(rec.get("name") or ""), product):
            continue
        pids.add(pid)
    return pids


def _known_process_product(image: str) -> str | None:
    return next((candidate for candidate in _PRODUCT_BACKENDS
                 if _image_matches_product(image, candidate)), None)


def managed_worker_census(
    root: Path,
    *,
    product: str,
    probe: Callable[[int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the one provenance census used for capacity and refusal."""
    if product != "codex":
        pids = set(live_resolve_worker_pids(root / RUNS_DIRNAME, product=product))
        pids.update(live_goal_worker_pids(root / GOAL_RUNS_DIRNAME, product=product))
        pids.update(_cmdline_worker_pids(product=product))
        return {"pids": sorted(pids), "count": len(pids),
                "status": "CONSISTENT", "ambiguous": []}

    probe_fn = probe or _process_probe
    pids: set[int] = set()
    sidecar_pids: set[int] = set()
    ambiguous: list[dict[str, Any]] = []
    known_backends = {b for values in _PRODUCT_BACKENDS.values() for b in values}

    def inspect(pid_file: Path, *, backend: str | None, goal: bool = False) -> None:
        try:
            raw_pid = pid_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            ambiguous.append({"sidecar": pid_file.name, "reason": "unreadable_pid",
                              "error": type(exc).__name__})
            return
        try:
            pid = int(raw_pid)
        except ValueError:
            ambiguous.append({"sidecar": pid_file.name, "reason": "malformed_pid"})
            return
        if pid <= 0:
            ambiguous.append({"sidecar": pid_file.name, "pid": pid,
                              "reason": "malformed_pid"})
            return
        try:
            rec = probe_fn(pid)
        except Exception as exc:
            ambiguous.append({"sidecar": pid_file.name, "pid": pid,
                              "reason": "probe_inspection_failed",
                              "error": type(exc).__name__})
            return
        if not isinstance(rec, dict) or "alive" not in rec:
            ambiguous.append({"sidecar": pid_file.name, "pid": pid,
                              "reason": "probe_uninspectable"})
            return
        if rec.get("alive") is False:
            return
        if rec.get("alive") is not True:
            ambiguous.append({"sidecar": pid_file.name, "pid": pid,
                              "reason": "probe_uninspectable"})
            return
        try:
            sidecar_mtime = pid_file.stat().st_mtime
        except OSError as exc:
            ambiguous.append({"sidecar": pid_file.name, "pid": pid,
                              "reason": "unreadable_sidecar",
                              "error": type(exc).__name__})
            return
        image_product = _known_process_product(str(rec.get("name") or ""))
        if goal:
            if image_product and image_product != product:
                return
            if image_product != product:
                ambiguous.append({"sidecar": pid_file.name, "pid": pid,
                                  "reason": "recycled_or_unverifiable_pid"})
                return
        elif image_product and image_product != product:
            ambiguous.append({"sidecar": pid_file.name, "pid": pid,
                              "reason": "contradictory_backend_image",
                              "backend": backend, "image": rec.get("name")})
            return
        if not _sidecar_process_matches(
                pid, sidecar_mtime, probe=lambda _pid, rec=rec: rec):
            ambiguous.append({"sidecar": pid_file.name, "pid": pid,
                              "reason": "recycled_or_unverifiable_pid"})
            return
        pids.add(pid)
        sidecar_pids.add(pid)

    runs_dir = root / RUNS_DIRNAME
    if runs_dir.is_dir():
        for pid_file in (*runs_dir.glob("resolve-*.pid"),
                         *runs_dir.glob("repair-*.pid")):
            if not _RESOLVE_PID_RE.match(pid_file.name):
                continue
            backend_file = pid_file.with_suffix(".backend")
            try:
                backend = backend_file.read_text(encoding="utf-8").strip() or None
            except FileNotFoundError:
                ambiguous.append({"sidecar": pid_file.name,
                                  "reason": "missing_backend"})
                continue
            except OSError as exc:
                ambiguous.append({"sidecar": pid_file.name,
                                  "reason": "unreadable_backend",
                                  "error": type(exc).__name__})
                continue
            if backend is None:
                ambiguous.append({"sidecar": pid_file.name,
                                  "reason": "missing_backend"})
                continue
            if backend not in known_backends:
                ambiguous.append({"sidecar": pid_file.name,
                                  "reason": "unknown_backend", "backend": backend})
                continue
            if backend not in _product_backends(product):
                continue
            inspect(pid_file, backend=backend)

    goal_runs_dir = root / GOAL_RUNS_DIRNAME
    if goal_runs_dir.is_dir():
        for pid_file in goal_runs_dir.glob("*.pid"):
            if _GOAL_PID_RE.match(pid_file.name):
                inspect(pid_file, backend=None, goal=True)

    pids.update(_managed_cmdline_pids_excluding_sidecar_trees(
        sidecar_pids, product=product))
    return {"pids": sorted(pids), "count": len(pids),
            "status": "AMBIGUOUS" if ambiguous else "CONSISTENT",
            "ambiguous": ambiguous}


def managed_worker_identity_check(
    root: Path,
    *,
    product: str,
    probe: Callable[[int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    census = managed_worker_census(root, product=product, probe=probe)
    return {"status": census["status"], "ambiguous": census["ambiguous"]}


def proc_worker_count(root: Path | None = None, *, product: str | None = None) -> int:
    """Count live worker processes that consume the dispatch cap.

    This combines the generic DOS-loop command-line marker with the issue
    resolver's pid sidecars and the detached /goal launcher's pid breadcrumbs
    (#2226: a stdin-fed /goal worker has no cmdline marker, so its breadcrumb is
    the only witness between spawn and its first lane lease). Use the union of
    pids so a worker visible through several witnesses is counted once; if psutil
    is absent, each witness is still best effort and may be conservatively low
    rather than fabricating capacity.

    When ``product`` is given, only workers in that account pool count — a claude
    worker no longer pins the opencode lane's cap and vice versa, so the two lanes
    fill to their independent account headrooms instead of starving one another.
    Codex has a single ambient login rather than a roster, but foreground sessions are
    telemetry, not managed worker provenance. Codex capacity therefore uses the same
    positive sidecar/breadcrumb/guarded-marker union as every other product. The generic
    cmdline-marked DOS-loop workers carry no backend tag, so they are attributed to a pool by their process IMAGE
    (``_image_matches_product``): a claude dos-dispatch-loop worker counts against
    the claude pool's cap even though it wrote no `.backend` sidecar — without this,
    a product-scoped count returned 0 while such workers were live and authorized an
    over-subscribing spawn (the no-DoS bound is only sound if live is never
    undercounted).
    """
    root = root or repo_root()
    if product is not None:
        return int(managed_worker_census(root, product=product)["count"])
    pids = set(_cmdline_worker_pids())
    pids.update(live_resolve_worker_pids(root / RUNS_DIRNAME))
    pids.update(live_goal_worker_pids(root / GOAL_RUNS_DIRNAME))
    return len(pids)


def _int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# --- Host-derived adaptive cap (#1337) -------------------------------------- #
# The boolean host_safe gate (REFUSE_HOST) is a HARD stop on a runaway; it never
# derived a NUMBER, so "up to --max-workers" was governed by a static config value,
# not by what the box can actually sustain. host_capacity() closes that: it turns
# the box's CURRENT headroom — cores, free RAM, and the live OS-thread total that
# proc_resource_guard.py polices — into the largest worker population the host can
# carry right now, folded via min into the cap below so a loaded box auto-throttles
# and recovers as load clears. A worker is not free: it is one claude/opencode
# session that fans out a tree of per-tool DOS hook subprocesses, so each is charged
# a slice of every resource. These per-worker budgets are deliberately conservative
# (the safe default is "barely grow"); the operator's --max-workers is still the
# outer ceiling.
# Per-worker host budgets, every one an env knob (FAK_HOST_*) with a conservative
# built-in guess. They assume a box DEDICATED to the fleet: on a shared box the
# live OS-thread total also counts threads the fleet never spawned and cannot reap
# (another user's editor/browser/agent tree), so the thread dimension throttles to
# host_cap=1 even with cores and RAM to spare — the operator raises
# FAK_HOST_THREADS_PER_CORE there to discount that foreign baseline. A measured
# box (headroom-audit lever 5) retunes the per-worker charges the same way. The
# boolean host_safe gate (not this gradient) remains the hard stop on a runaway.
HOST_CORES_PER_WORKER = _env_pos_int("FAK_HOST_CORES_PER_WORKER", 2)       # cores a worker + its hook subprocess tree occupies
HOST_RAM_MB_PER_WORKER = _env_pos_int("FAK_HOST_RAM_MB_PER_WORKER", 1500)  # resident MB across that subprocess tree
HOST_THREADS_PER_CORE = _env_pos_int("FAK_HOST_THREADS_PER_CORE", 400)     # host-wide OS-thread budget, scaled by core count
HOST_THREADS_PER_WORKER = _env_pos_int("FAK_HOST_THREADS_PER_WORKER", 200) # OS threads a worker + its hooks add to the box
HOST_CAP_FLOOR = 1              # never throttle below one worker — the hard stop on
                                # a genuine runaway stays the host_safe gate, not this


def host_capacity(*, cores: Any, free_ram_mb: Any, total_threads: Any,
                  cores_per_worker: int = HOST_CORES_PER_WORKER,
                  ram_mb_per_worker: int = HOST_RAM_MB_PER_WORKER,
                  host_threads_per_core: int = HOST_THREADS_PER_CORE,
                  threads_per_worker: int = HOST_THREADS_PER_WORKER,
                  floor: int = HOST_CAP_FLOOR) -> dict[str, Any]:
    """Largest sustainable worker population from the host's CURRENT headroom (#1337).

    Pure — every input is a measured scalar — so it is hermetically testable. Each
    live worker is charged a slice of three host resources; host_cap is the SCARCEST
    slice, floored at ``floor`` so a momentarily-slammed box still makes progress
    (the hard stop on a true runaway is the boolean host_safe gate, not this
    gradient). A dimension whose reading is unknown (``None``) is skipped — a missing
    probe never fabricates capacity nor falsely throttles; the thread dimension also
    needs the core count (its budget scales with cores). When every dimension is
    unknown the result is ``host_cap=None`` (no host bound available, so the static
    --max-workers/target govern the cap alone)."""
    components: dict[str, int] = {}
    cores_n = _int(cores)
    if cores_n is not None and cores_n > 0 and cores_per_worker > 0:
        components["cores"] = cores_n // cores_per_worker
    ram_n = _int(free_ram_mb)
    if ram_n is not None and ram_n >= 0 and ram_mb_per_worker > 0:
        components["ram"] = ram_n // ram_mb_per_worker
    threads_n = _int(total_threads)
    if (threads_n is not None and threads_n >= 0 and cores_n is not None
            and cores_n > 0 and host_threads_per_core > 0 and threads_per_worker > 0):
        free_threads = max(0, cores_n * host_threads_per_core - threads_n)
        components["threads"] = free_threads // threads_per_worker
    info: dict[str, Any] = {"cores": cores_n, "free_ram_mb": ram_n,
                            "total_threads": threads_n, "components": components}
    if not components:
        info.update(host_cap=None, binding=None)
        return info
    # The binding dimension (the scarcest, reported BEFORE the floor is applied) tells
    # the operator WHICH resource is throttling the fleet right now.
    info.update(host_cap=max(floor, min(components.values())),
                binding=min(components, key=lambda k: components[k]))
    return info


def _host_binding_limiter(binding: Any) -> str:
    return {
        "cores": "cpu",
        "ram": "memory",
        "threads": "threads",
    }.get(str(binding or ""), "host")


def capacity_limiter(*, max_workers: int, target: Any, host_cap_info: dict[str, Any],
                     seat: dict[str, Any], live: int, cap: int) -> dict[str, Any]:
    """Primary worker-count limiter plus the raw terms that selected it.

    The effective cap is a min() across configured ceiling, optional DOS target,
    host headroom, and seat count. When live workers consume that cap, the current
    limiter is the lease/live-worker occupancy rather than the static ceiling.
    """
    host_cap = host_cap_info.get("host_cap")
    host_binding = host_cap_info.get("binding")
    seat_total = seat.get("total")
    raw = {
        "cap": cap,
        "live": live,
        "headroom": cap - live,
        "max_workers": max_workers,
        "dos_target": target,
        "host_cap": host_cap,
        "host_binding": host_binding,
        "host_components": host_cap_info.get("components") or {},
        "seat_total": seat_total,
        "seat_free": seat.get("free"),
        "seat_leased": seat.get("leased"),
        "seat_depleted": bool(seat.get("depleted")),
    }
    if cap > 0 and live >= cap:
        return {"primary": "leases", "term": "live", "raw": raw}

    terms: list[tuple[str, str, int]] = [("configured_max", "max_workers", max_workers)]
    target_n = _int(target)
    if target_n is not None and target_n > 0:
        terms.append(("configured_max", "dos_target", target_n))
    host_cap_n = _int(host_cap)
    if host_cap_n is not None:
        terms.append((_host_binding_limiter(host_binding), "host_cap", host_cap_n))
    seat_n = _int(seat_total)
    if seat_n is not None and seat_n > 0:
        terms.append(("seats", "seat_total", seat_n))

    primary, term, _ = min(terms, key=lambda row: row[2])
    return {"primary": primary, "term": term, "raw": raw}


def account_unattributed_live_slots(seat: dict[str, Any], live: int) -> dict[str, Any]:
    """Conservatively charge live workers that lack account sidecars to free slots.

    Sidecar leases are the precise account binding, but older workers or damaged
    sidecars can leave ``leased`` below the product-scoped live-worker count. In that
    case the safe operator view is not "all slots free"; subtract the unattributed
    live workers from free headroom while leaving the hard cap's total-slot term
    unchanged.
    """
    total = _int(seat.get("total"))
    if total is None or total <= 0 or live <= 0:
        return seat
    leased = _int(seat.get("leased")) or 0
    missing = max(0, live - leased)
    if missing <= 0:
        return seat
    out = dict(seat)
    free = _int(seat.get("free"))
    if free is not None:
        out["free"] = max(0, free - missing)
    else:
        out["free"] = max(0, total - min(live, total))
    out["leased"] = max(leased, min(live, total))
    out["depleted"] = bool(seat.get("depleted")) or out["free"] == 0
    out["unattributed_live"] = missing
    return out


def _capacity_limiter_terms(limiter: dict[str, Any]) -> str:
    raw = limiter.get("raw") or {}
    parts = [
        f"cap={raw.get('cap')}",
        f"live={raw.get('live')}",
        f"headroom={raw.get('headroom')}",
        f"max={raw.get('max_workers')}",
        f"target={raw.get('dos_target')}",
        f"host_cap={raw.get('host_cap')}",
        f"host_binding={raw.get('host_binding')}",
        f"seats={raw.get('seat_total')}",
        f"free={raw.get('seat_free')}",
        f"leased={raw.get('seat_leased')}",
    ]
    return " ".join(parts)


def _ram_and_threads_windows() -> tuple[int | None, int | None]:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command",
         "$os = Get-CimInstance Win32_OperatingSystem; "
         "$t = (Get-Process -ErrorAction SilentlyContinue | "
         "ForEach-Object { $_.Threads.Count } | Measure-Object -Sum).Sum; "
         "[pscustomobject]@{ free_kb = [int64]$os.FreePhysicalMemory; threads = [int]$t } | "
         "ConvertTo-Json -Compress"],
        capture_output=True, text=True, timeout=45,
        creationflags=_no_window_creationflags(), check=False)
    doc = _last_json(out.stdout)
    free_kb = _int(doc.get("free_kb"))
    threads = _int(doc.get("threads"))
    return (free_kb // 1024 if free_kb is not None else None), threads


def _ram_and_threads_posix() -> tuple[int | None, int | None]:
    free_ram_mb: int | None = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):  # kB the kernel can hand out now
                    kb = _int(line.split()[1])
                    free_ram_mb = kb // 1024 if kb is not None else None
                    break
    except OSError:
        free_ram_mb = None
    total_threads: int | None = None
    try:
        out = subprocess.run(["ps", "-eo", "nlwp="], capture_output=True,
                             text=True, timeout=20, check=False)
        total = 0
        seen = False
        for tok in out.stdout.split():
            n = _int(tok)
            if n is not None:
                total += n
                seen = True
        total_threads = total if seen else None
    except (OSError, subprocess.SubprocessError):
        total_threads = None
    return free_ram_mb, total_threads


def host_resources() -> dict[str, Any]:
    """Probe cores + available RAM (MB) + total live OS threads for host_capacity().
    Best effort and read-only: any dimension the platform won't yield comes back
    None so host_capacity() simply skips it. One lightweight platform call (the
    supported CIM API on Windows, ``ps``/``/proc`` on POSIX); no third-party deps.
    Patched out in the hermetic preflight tests, like the other shelling checks."""
    cores = os.cpu_count()
    try:
        if os.name == "nt":
            free_ram_mb, total_threads = _ram_and_threads_windows()
        else:
            free_ram_mb, total_threads = _ram_and_threads_posix()
    except (OSError, subprocess.SubprocessError, ValueError):
        free_ram_mb, total_threads = None, None
    return {"cores": cores, "free_ram_mb": free_ram_mb, "total_threads": total_threads}


def evaluate(root: Path, *, max_workers: int, work_kind: str, product: str,
             max_threads: int | None = None) -> dict[str, Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        host_f = pool.submit(host_check, root, max_threads=max_threads)
        acct_f = pool.submit(account_check, root, work_kind=work_kind, product=product)
        kern_f = pool.submit(kernel_alive, root)
        seat_f = pool.submit(seat_check, root, product=product)
        host_res_f = pool.submit(host_resources)
        worker_census_f = pool.submit(managed_worker_census, root, product=product)
        # Advisory only: which `fak` build each resolver picks, and whether any gate
        # is about to run an unreviewed `+uncommitted` compile. Probed in the same
        # pool because it is pure I/O (stat + one bounded `version` per distinct
        # build) and must not add serial latency to the spawn path.
        fak_bin_f = pool.submit(fak_bin_provenance, root)
        commit_headroom_f = pool.submit(system_commit_headroom_check)
        host = host_f.result()
        acct = acct_f.result()
        kern = kern_f.result()
        seat = seat_f.result()
        host_res = host_res_f.result()
        try:
            worker_identity = worker_census_f.result()
            alive_proc = int(worker_identity["count"])
        except Exception as exc:  # noqa: BLE001 - identity uncertainty fails closed below
            worker_identity = {"pids": [], "count": 0, "status": "AMBIGUOUS",
                               "ambiguous": [
                                   {"reason": "inspection_error", "error": str(exc)}]}
            alive_proc = 0
        try:
            fak_bin = fak_bin_f.result()
        except Exception as exc:  # noqa: BLE001 — provenance NEVER decides a verdict
            fak_bin = {"schema": FAK_BIN_PROVENANCE_SCHEMA, "error": str(exc),
                       "resolvers": {}, "agree": True, "dirty": []}
        try:
            commit_headroom = commit_headroom_f.result()
        except Exception as exc:  # noqa: BLE001 - telemetry uncertainty fails closed on Windows
            commit_headroom = {"supported": os.name == "nt", "ok": False,
                               "error": str(exc), "reason": "",
                               "observed_bytes": 0,
                               "required_bytes": required_system_commit_headroom_bytes(),
                               "system_commit_bytes": 0, "system_commit_limit": 0}
    host_cap_info = host_capacity(**host_res)
    host_cap = host_cap_info.get("host_cap")

    target = kern.get("target")
    # `dos [supervise].target` is the kernel's STANDING-LOOP population — the
    # emit-only `dos loop` path, which never Popens, so it is honestly 0 in this
    # repo (#517) until the keep-alive ARMING (#20) lands. The issue-dispatch cron
    # (FleetIssueDispatch/Glm) is a SEPARATE, already-armed self-spawner that
    # Popens its own workers; its ceiling is the operator's --max-workers, NOT the
    # emit-only standing-loop target. So a POSITIVE target is a throttle-DOWN dial
    # (dos asked for fewer than max_workers); a zero/unset target means "dos is not
    # managing a standing loop" and the dispatcher's own --max-workers governs.
    # target=0 is NOT a kill switch — disable the cron task, or trip the host /
    # weekly-cap gate, to actually freeze spawning. (Before this, target=0 silently
    # pinned cap to 0 and wedged the live issue-closer for ~12h after the #517 fix.)
    cap = min(max_workers, target) if target else max_workers
    # Fold the host-derived recommended cap (#1337): even when dos target is 0/unset
    # (the emit-only standing loop in this repo), the live host headroom still
    # throttles the operator's --max-workers down to what the box can carry right
    # now, so a request for "up to 100" auto-adapts to load. host_cap is None only
    # when no host dimension could be read, in which case the static caps govern.
    if host_cap is not None:
        cap = min(cap, host_cap)
    cap = max(0, cap)
    # The pre-seat cap (operator/dos/host). Fold the EXPLICIT seat pool in next: M
    # distinct routable worker seats back at most M live workers, so the seat count is
    # another hard min() term on concurrency (#1336 "the effective cap becomes
    # number_of_free_seats"). A None/absent total — the seat view could not run, or the
    # product has no roster (codex ambient) — SKIPS the fold so the gate never
    # fail-closes on a missing seat view; the static/host caps still bound the fleet.
    cap_pre_seat = cap
    seats_total = seat.get("total")
    fold_seats = isinstance(seats_total, int) and seats_total > 0
    if fold_seats:
        cap = min(cap, seats_total)
    cap = max(0, cap)
    alive_kernel = kern.get("alive")
    # When dos target is zero/unset, `dos loop` is emit-only in this repo and its
    # `alive` count describes live DOS lanes (operator/peer work such as
    # tools/docs/experiments), not issue-resolution worker processes. Counting it
    # against the issue-dispatcher cap false-pins spawning whenever normal peer
    # lanes are active. A positive target means dos is managing a standing worker
    # population, so kernel alive becomes a real cap consumer again.
    alive_kernel_for_cap = alive_kernel if target else 0
    # Scope the worker count to THIS product's account pool: a claude (opus) worker
    # and an opencode (GLM) worker draw on different accounts/rate limits, so each
    # lane fills to its own headroom instead of the two sharing one global cap and
    # starving each other (claude+GLM ran 3 total instead of 3+3 before this).
    # MAX of the two views: neither a stale lease nor an unleased orphan hides load.
    live = max(alive_kernel_for_cap or 0, alive_proc)
    observed_seat_leases = int(seat.get("leased", 0))
    seat = account_unattributed_live_slots(seat, live)
    process_gap = observed_seat_leases - alive_proc
    seat["process_gap"] = process_gap
    seat["process_consistency"] = (
        "SEATS_EXCEED_PROCESS_TREES" if process_gap > 0 else
        "PROCESS_TREES_EXCEED_SEATS" if process_gap < 0 else
        "CONSISTENT"
    )
    headroom = cap - live
    limiter = capacity_limiter(max_workers=max_workers, target=target,
                               host_cap_info=host_cap_info, seat=seat,
                               live=live, cap=cap)

    # Weekly-limit seat cooldown (#2610): a 429 with kind=weekly_limit is a VALID
    # credential whose seat is temporarily quota-capped (distinct from the stale-cred
    # cases #2059/#2075). The always-on resolve dispatcher persists that hold in
    # .dispatch-runs/account-cap-*.json (with the announced reset window); honor it
    # here so a routed-but-capped seat is not re-offered. Only meaningful once the
    # switcher actually routed an account — a no-account state is REFUSE_NO_ACCOUNT.
    weekly = (weekly_cap_check(root, product=product, account_tag=acct.get("tag"))
              if acct.get("available") and acct.get("tag") else {"capped": False})

    # Fail-safe ordering, evaluated top to bottom:
    #   1. an un-runnable host/kernel safety check  -> REFUSE_INSPECT (never assume safe)
    #   2. a flagged host                            -> REFUSE_HOST
    #   3. a depleted seat pool (seats are binding)  -> REFUSE_NO_SEAT
    #   4. at/over the worker cap                    -> REFUSE_AT_CAP
    #   5. no available account (incl. switcher err) -> REFUSE_NO_ACCOUNT
    #   6. routed account weekly-limit capped        -> REFUSE_WEEKLY_CAPPED (#2610)
    # An account check that merely errored is just "no account available", which
    # branch 5 already reports — it does not need to pre-empt host/cap.
    # A depleted seat pool (every routable seat already leased to a live worker) is its
    # own typed refusal (#1336): there is NO free seat to hand out, so spawning would
    # double-book a busy seat — refuse with REFUSE_NO_SEAT, distinct from the generic
    # operator/dos/host ceiling. It is raised only when the seat count is the BINDING cap
    # term (``seats_total <= cap_pre_seat``); when a tighter operator/host cap bit first,
    # REFUSE_AT_CAP stays the honest reason. The depleted signal is authoritative even if
    # the live-worker scan under-counts, so the N>M-wave remainder always gets a structured
    # "no seat", never a silent double-book.
    seats_deplete = bool(
        fold_seats and seat.get("depleted") and seats_total <= cap_pre_seat
        and (seat.get("leased") or 0) > 0)
    if host.get("error") or kern.get("error"):
        verdict = REFUSE_INSPECT
        reason = (host.get("error") or kern.get("error")
                  or "a preflight safety check could not run")
    elif worker_identity.get("status") == "AMBIGUOUS":
        verdict = REFUSE_INSPECT
        reasons = sorted({str(row.get("reason") or "unknown")
                          for row in worker_identity.get("ambiguous", [])})
        reason = ("managed worker identity is ambiguous: "
                  + ", ".join(reasons)
                  + "; inspect the named sidecar(s) before growing the fleet")
    elif commit_headroom.get("supported") and commit_headroom.get("error"):
        verdict = REFUSE_INSPECT
        reason = ("system commit headroom inspection failed: "
                  + str(commit_headroom.get("error")))
    elif commit_headroom.get("supported") and not commit_headroom.get("ok"):
        verdict = REFUSE_SYSTEM_COMMIT_HEADROOM
        reason = (
            f"{SYSTEM_COMMIT_HEADROOM_REASON}: observed "
            f"{commit_headroom.get('observed_bytes')} bytes is at/below required "
            f"{commit_headroom.get('required_bytes')} bytes; run "
            "`fak recover SYSTEM_COMMIT_HEADROOM` for the bounded recovery route")
    elif not host["safe"]:
        verdict = REFUSE_HOST
        reason = (f"host resource guard flagged {host['flagged']} process(es): "
                  f"{', '.join(host.get('flagged_names') or []) or 'see proc_resource_guard'}"
                  " — reap/inspect before growing the fleet")
    elif seats_deplete:
        verdict = REFUSE_NO_SEAT
        reason = (f"seat pool depleted: 0 of {seats_total} routable seat(s) free "
                  f"({seat.get('leased')} leased to live worker(s), live={live}); a seat "
                  "frees when a worker exits — refusing rather than double-book a busy seat")
    elif live >= cap:
        verdict = REFUSE_AT_CAP
        reason = (f"live workers {live} >= cap {cap} "
                  f"(kernel alive={alive_kernel}, os procs={alive_proc}, "
                  f"dos target={target}, host_cap={host_cap}, "
                  f"max-workers={max_workers})")
    elif not acct["available"]:
        verdict = REFUSE_NO_ACCOUNT
        blocked = ", ".join(b for b in (acct.get("blocked") or []) if b)
        reason = ("switcher has no available worker account at the requested tier"
                  + (f" (blocked: {blocked})" if blocked else "")
                  + f": {acct.get('reason') or acct.get('error') or ''}".rstrip())
    elif weekly.get("capped"):
        verdict = REFUSE_WEEKLY_CAPPED
        reason = (f"account '{acct.get('tag')}' hit a weekly-limit 429 "
                  f"(kind={weekly.get('kind') or 'weekly'}"
                  + (f", resets {weekly.get('reset_text')}" if weekly.get('reset_text') else "")
                  + f"): cooling until {weekly.get('until')} — not re-offering the seat "
                  "until its announced reset window elapses (distinct from a stale "
                  "credential; #2610)")
    else:
        verdict = OK_VERDICT
        reason = (f"safe to spawn: host clean, account '{acct.get('tag')}' "
                  f"(t{acct.get('tier')}) free, {live}/{cap} live (headroom {headroom})")

    bin_refusal = repository_binary_refusal(fak_bin)
    if verdict == OK_VERDICT and bin_refusal:
        if fak_bin.get("historical_override"):
            reason += (f" (bin freshness ADMITTED BY OVERRIDE — {bin_refusal[1]}; "
                       "FAK_PREFLIGHT_ALLOW_BIN_SKEW=1)")
        else:
            verdict, reason = bin_refusal
    ok = verdict == OK_VERDICT
    # Bind the verdict to the build that produced it. Recorded (not just returned)
    # because the tick payload projects preflight through an ALLOWLIST, so a new key
    # alone would never reach the durable run record; the `.dispatch-runs` file is
    # readable after the fact by anyone asking "which build refused?". Warnings go to
    # stderr so they surface in the dispatcher's captured log even under `--json`,
    # where stdout must stay parseable.
    record_fak_bin_provenance(root, fak_bin)
    for line in fak_bin_warnings(fak_bin):
        print(f"dispatch_preflight: {line}", file=sys.stderr)
    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "reason": reason,
        "workspace": str(root),
        "fak_bin": fak_bin,
        "cap": cap,
        "live": live,
        "headroom": headroom,
        "max_workers": max_workers,
        "host_cap": host_cap,
        "host_capacity": host_cap_info,
        "capacity_limiter": limiter,
        "seat": seat,
        "weekly_cap": weekly,
        "host": host,
        "account": {k: acct.get(k) for k in ("available", "tag", "dir", "tier", "model", "login_status", "can_serve")},
        "kernel": kern,
        "os_worker_procs": alive_proc,
        "worker_identity": worker_identity,
        "system_commit_headroom": commit_headroom,
    }


def _host_state(host: dict[str, Any]) -> str:
    """Status-card host state (#2252): ``clean`` | ``ADVISORY(names)`` | ``FLAGGED``.

    ``ADVISORY`` is a SAFE host carrying protected (non-actionable) breaches, so
    the operator can tell "foreign baseline noted" (spawning proceeds) from
    "reap before growing" (``FLAGGED``, which refuses)."""
    if not host.get("safe"):
        return "FLAGGED"
    names = [n for n in (host.get("protected_names") or []) if n]
    return f"ADVISORY({','.join(names)})" if names else "clean"


def render(p: dict[str, Any]) -> str:
    a = p.get("account") or {}
    hc = p.get("host_capacity") or {}
    limiter = p.get("capacity_limiter") or {}
    host_cap = p.get("host_cap")
    host_cap_str = (f"host_cap={host_cap}"
                    + (f" (bound by {hc.get('binding')})" if hc.get("binding") else "")
                    if host_cap is not None else "host_cap=n/a")
    lines = [
        f"dispatch preflight: {p.get('verdict')} ({'ok' if p.get('ok') else 'refuse'})",
        f"  reason: {p.get('reason')}",
        f"  live={p.get('live')}/{p.get('cap')} (headroom {p.get('headroom')})  "
        f"host={_host_state(p.get('host') or {})}  "
        f"account={a.get('tag') or '-'} (t{a.get('tier')})  {host_cap_str}",
        f"  limiter={limiter.get('primary') or '-'} ({_capacity_limiter_terms(limiter)})",
    ]
    weekly = p.get("weekly_cap") or {}
    if weekly.get("capped"):
        lines.append(
            f"  weekly-cap: {weekly.get('account') or a.get('tag') or '-'} cooling "
            f"until {weekly.get('until')} (kind={weekly.get('kind') or 'weekly'}, "
            f"resets {weekly.get('reset_text') or '?'})")
    fak_bin = p.get("fak_bin") or {}
    gate = (fak_bin.get("resolvers") or {}).get("preflight_gate") or {}
    if gate.get("resolved"):
        lines.append(
            f"  fak_bin: gate={gate.get('build') or '?'}"
            f"{' +uncommitted' if gate.get('dirty') else ''} @ {gate.get('path')}  "
            f"({fak_bin.get('distinct_builds')} distinct build(s) across resolvers)")
    lines.extend(f"  !! {w}" for w in fak_bin_warnings(fak_bin))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Read-only spawn gate: is it safe to launch another dispatch worker?")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                    help=f"hard ceiling on live workers (default: {DEFAULT_MAX_WORKERS}); "
                         "effective cap is min(host_cap, dos [supervise].target, this) "
                         "when that target is positive, else min(host_cap, this) (a "
                         "zero/unset target lets the cron-armed self-spawner use its own "
                         "ceiling). host_cap is the host-derived adaptive bound (#1337) "
                         "from cores, free RAM, and the live OS-thread total")
    ap.add_argument("--work-kind", default="engineering",
                    help="work kind for the switcher route (engineering→t1, gardening→t2)")
    ap.add_argument("--product", default="claude", help="worker product (default: claude)")
    ap.add_argument("--max-threads", type=int, default=None,
                    help="override proc_resource_guard --max-threads")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = evaluate(root, max_workers=args.max_workers, work_kind=args.work_kind,
                       product=args.product, max_threads=args.max_threads)
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
