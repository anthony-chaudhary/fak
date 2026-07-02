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
— ``SPAWN_OK`` or a typed ``REFUSE_*`` — by composing three INDEPENDENT checks,
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
import datetime as dt
import functools
import json
import os
import re
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

# Operator's *aspirational* ceiling on simultaneous live dispatch workers — NOT the
# safety bound. The real DoS proof is the adaptive cap below: min(this, host_cap,
# seats). host_cap (#1337) auto-throttles to the box's live cores/RAM/thread
# headroom; the seat pool (#1336) hard-bounds at one worker per routable account so a
# spawn can never double-book a rate limit. With both gates in place this default was
# doubled 2->4: the old 2 sat below host_cap and the seat count alike (it was the
# artificial bottleneck), so raising it simply lets the adaptive gates — which can
# only LOWER the effective cap — govern. The box is still a live multi-session fleet,
# so on a loaded host host_cap pulls the effective cap back down automatically.
DEFAULT_MAX_WORKERS = 4

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
_RESOLVE_PID_RE = re.compile(r"resolve-\d+-\d{8}-\d{6}\.pid$")
_SIDECAR_CREATE_BEFORE_WINDOW_SECONDS = 5 * 60
_SIDECAR_CREATE_AFTER_SLOP_SECONDS = 10
# Process names a real dispatch worker actually runs as. The create-time-window
# fallback (used only when the OS hid the cmdline) may ONLY count a sidecar pid
# whose image is one of these — otherwise a recycled pid that merely SHARES a
# claude/opencode backend image, or worse a generic shell (cmd.exe, powershell,
# bash) spawned in the same minute, gets miscounted as a live worker and pins the
# dispatcher at cap against ghosts. A bare shell image NEVER counts; only the
# named agent backends do, and even then only inside the spawn window.
_WORKER_BACKEND_IMAGES = ("claude", "opencode", "node")


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


def account_check(root: Path, *, work_kind: str, product: str) -> dict[str, Any]:
    """fleet_accounts.py route ⇒ the switcher's pick. ok+account ⇒ a free worker.

    Codex is the exception: it has no switcher roster (single ambient login), so its
    availability is read straight from ``~/.codex`` rather than the switcher."""
    if product == "codex":
        return _codex_ambient_account()
    sw = root / "tools" / "fleet_accounts.py"
    if not sw.exists():
        return {"available": False, "error": f"switcher not found: {sw}"}
    cmd = [_py(), str(sw), "route", "--product", product, "--work-kind", work_kind]
    doc = run_json(cmd, root, timeout=45, ok_codes={0, 1})
    if doc.get("_error") and "ok" not in doc:
        return {"available": False, "error": doc["_error"]}
    acct = doc.get("account") or {}
    return {"available": bool(doc.get("ok")) and bool(acct),
            "tag": acct.get("tag"), "dir": acct.get("dir"),
            "tier": doc.get("selected_tier"), "model": acct.get("model"),
            "reason": doc.get("reason"),
            "blocked": [b.get("tag") for b in (doc.get("blocked_target_accounts") or [])]}


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

    Codex is the one-product exception to the switcher roster: it uses a single ambient
    ChatGPT login, so the honest pool size is ONE seat, not "no seat pool". Any live
    Codex CLI process leases that seat, including an attended operator session, so an
    always-on codex dispatch tick cannot collide with the foreground Codex window/login."""
    if product == "codex":
        live = len(ambient_codex_pids())
        leased = 1 if live else 0
        return {"total": 1, "free": 1 - leased, "leased": leased,
                "depleted": bool(leased), "ambient_live": live,
                "reason": "codex uses one ambient login/seat"}
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
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None
    if psutil is not None:
        pids: set[int] = set()
        parent_by_pid: dict[int, int | None] = {}
        for p in psutil.process_iter(["name", "cmdline", "ppid"]):
            try:
                cl = " ".join(p.info.get("cmdline") or [])
                parent_by_pid[int(p.pid)] = int(p.info.get("ppid") or 0)
            except (psutil.Error, TypeError):
                continue
            # Marker on the cmdline AND a real backend image — a shell that merely
            # mentions the marker (a grep, a launcher, an operator inspecting) is
            # not a live worker and must not consume a cap slot.
            name = p.info.get("name") or ""
            if (_is_worker_cmdline(cl) and _is_worker_image(name)
                    and (product is None or _image_matches_product(name, product))):
                pids.add(int(p.pid))
        return _collapse_descendant_pids(pids, parent_by_pid)
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


@functools.lru_cache(maxsize=1)
def ambient_codex_pids() -> set[int]:
    """Live Codex CLI processes sharing the ambient ChatGPT login/seat.

    This intentionally counts attended Codex sessions too. The background codex
    dispatcher has no separate account pool, so a foreground Codex session consumes
    the only ambient seat and the dispatch tick must hold instead of starting another
    codex worker that can fight the user's terminal/window state.
    """
    rows = _codex_process_rows_psutil()
    if rows is None:
        rows = _codex_process_rows_windows() if os.name == "nt" else _codex_process_rows_posix()
    return _codex_process_pids_from_rows(rows)


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
    all qualify; anything else does not.
    """
    name = str(probe.get("name") or "").strip().lower()
    if not name:
        return False
    stem = name[:-4] if name.endswith(".exe") else name
    return stem in _WORKER_BACKEND_IMAGES


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
    """Live issue-resolution workers from `.dispatch-runs/resolve-*.pid`.

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
    for pid_file in runs_dir.glob("resolve-*.pid"):
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
    Codex has a single ambient seat rather than a roster; for that product, every
    live Codex CLI process also consumes the cap so an attended Codex session blocks
    background codex dispatch. The generic cmdline-marked DOS-loop workers carry no
    backend tag, so they are attributed to a pool by their process IMAGE
    (``_image_matches_product``): a claude dos-dispatch-loop worker counts against
    the claude pool's cap even though it wrote no `.backend` sidecar — without this,
    a product-scoped count returned 0 while such workers were live and authorized an
    over-subscribing spawn (the no-DoS bound is only sound if live is never
    undercounted).
    """
    root = root or repo_root()
    if product is not None:
        pids = set(live_resolve_worker_pids(root / RUNS_DIRNAME, product=product))
        pids.update(live_goal_worker_pids(root / GOAL_RUNS_DIRNAME, product=product))
        pids.update(_cmdline_worker_pids(product=product))
        if product == "codex":
            pids.update(ambient_codex_pids())
        return len(pids)
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
def _env_pos_int(name: str, default: int) -> int:
    """A positive-int env override, falling back to ``default`` on unset/garbage.

    The host budgets below assume a box DEDICATED to the fleet. On a shared box
    the live OS-thread total also counts threads the fleet never spawned and
    cannot reap (another user's editor/browser/agent tree), so the thread
    dimension throttles to host_cap=1 even with cores and RAM to spare. The
    operator raises FAK_HOST_THREADS_PER_CORE on such a box to discount that
    foreign baseline; the boolean host_safe gate (not this gradient) remains the
    hard stop on a genuine runaway."""
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            val = int(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return default


HOST_CORES_PER_WORKER = 2       # cores a worker + its hook subprocess tree occupies
HOST_RAM_MB_PER_WORKER = 1500   # resident MB across that subprocess tree
HOST_THREADS_PER_CORE = _env_pos_int("FAK_HOST_THREADS_PER_CORE", 400)  # host-wide OS-thread budget, scaled by core count
HOST_THREADS_PER_WORKER = 200   # OS threads a worker + its hooks add to the box
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
    host = host_check(root, max_threads=max_threads)
    acct = account_check(root, work_kind=work_kind, product=product)
    kern = kernel_alive(root)
    seat = seat_check(root, product=product)
    host_cap_info = host_capacity(**host_resources())
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
    alive_proc = proc_worker_count(root, product=product)
    # MAX of the two views: neither a stale lease nor an unleased orphan hides load.
    live = max(alive_kernel_for_cap or 0, alive_proc)
    headroom = cap - live
    limiter = capacity_limiter(max_workers=max_workers, target=target,
                               host_cap_info=host_cap_info, seat=seat,
                               live=live, cap=cap)

    # Fail-safe ordering, evaluated top to bottom:
    #   1. an un-runnable host/kernel safety check  -> REFUSE_INSPECT (never assume safe)
    #   2. a flagged host                            -> REFUSE_HOST
    #   3. a depleted seat pool (seats are binding)  -> REFUSE_NO_SEAT
    #   4. at/over the worker cap                    -> REFUSE_AT_CAP
    #   5. no available account (incl. switcher err) -> REFUSE_NO_ACCOUNT
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
    else:
        verdict = OK_VERDICT
        reason = (f"safe to spawn: host clean, account '{acct.get('tag')}' "
                  f"(t{acct.get('tier')}) free, {live}/{cap} live (headroom {headroom})")

    ok = verdict == OK_VERDICT
    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "reason": reason,
        "workspace": str(root),
        "cap": cap,
        "live": live,
        "headroom": headroom,
        "max_workers": max_workers,
        "host_cap": host_cap,
        "host_capacity": host_cap_info,
        "capacity_limiter": limiter,
        "seat": seat,
        "host": host,
        "account": {k: acct.get(k) for k in ("available", "tag", "dir", "tier", "model")},
        "kernel": kern,
        "os_worker_procs": alive_proc,
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
    return "\n".join([
        f"dispatch preflight: {p.get('verdict')} ({'ok' if p.get('ok') else 'refuse'})",
        f"  reason: {p.get('reason')}",
        f"  live={p.get('live')}/{p.get('cap')} (headroom {p.get('headroom')})  "
        f"host={_host_state(p.get('host') or {})}  "
        f"account={a.get('tag') or '-'} (t{a.get('tier')})  {host_cap_str}",
        f"  limiter={limiter.get('primary') or '-'} ({_capacity_limiter_terms(limiter)})",
    ])


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
