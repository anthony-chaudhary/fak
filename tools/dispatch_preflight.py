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
                     cap = min(dos [supervise].target, --max-workers) when the
                     target is POSITIVE (dos throttles the fleet down), else
                     --max-workers when the target is zero/unset (the emit-only
                     `dos loop` manages no standing loop, so the cron-armed
                     self-spawner's own ceiling governs — a zero target is not a
                     spawn kill switch) and
                     live = max(kernel's `dos loop` alive, an OS process scan for
                     live `dos-dispatch-loop` claudes) — the MAX so neither a
                     stale lease nor an unleased orphan can hide capacity

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
REFUSE_NO_ACCOUNT = "REFUSE_NO_ACCOUNT"  # switcher has no available worker account
REFUSE_AT_CAP = "REFUSE_AT_CAP"      # live workers already at/over the cap
REFUSE_INSPECT = "REFUSE_INSPECT"    # a check could not run (fail-safe → refuse)

# Default ceiling on simultaneous live dispatch workers. Intentionally small: the
# box is a live multi-session fleet, so the safe default is "barely grow", not
# "fill to dos target". The operator raises it deliberately with --max-workers.
DEFAULT_MAX_WORKERS = 2

# A live dispatch worker's command line carries this marker (dispatch_worker.py
# launches `claude -p ... /dos-kernel:dos-dispatch-loop --lane X`). Used to count
# real OS processes, the honest DoS signal the kernel's lease count can miss.
WORKER_CMD_MARKER = "dos-dispatch-loop"
ISSUE_RESOLVE_CMD_MARKER = "resolve GitHub issue #"
WORKER_CMD_MARKERS = (WORKER_CMD_MARKER, ISSUE_RESOLVE_CMD_MARKER)
RUNS_DIRNAME = ".dispatch-runs"
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
    """proc_resource_guard.py CLEAN ⇒ safe. Any flagged runaway/orphan ⇒ unsafe."""
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
    return {"safe": bool(doc.get("ok")) and not flagged,
            "flagged": len(flagged),
            "flagged_names": [str(r.get("name")) for r in flagged][:8]}


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


def _cmdline_worker_pids() -> set[int]:
    """OS-level worker pids by command-line marker.

    This catches the generic DOS-loop worker and the issue-resolution workers
    whose prompt is still on the top-level command line. Best effort: an
    inaccessible process table returns an empty set so the pid-sidecar witness
    below can still govern the issue-dispatch path.
    """
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None
    if psutil is not None:
        pids: set[int] = set()
        parent_by_pid: dict[int, int | None] = {}
        for p in psutil.process_iter(["cmdline", "ppid"]):
            try:
                cl = " ".join(p.info.get("cmdline") or [])
                parent_by_pid[int(p.pid)] = int(p.info.get("ppid") or 0)
            except (psutil.Error, TypeError):
                continue
            if _is_worker_cmdline(cl):
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
                 "ForEach-Object { \"$($_.ProcessId),$($_.ParentProcessId)\" }"],
                capture_output=True, text=True, timeout=30,
                creationflags=_no_window_creationflags()).stdout
            pids: set[int] = set()
            parent_by_pid: dict[int, int | None] = {}
            for ln in out.splitlines():
                parts = [p.strip() for p in ln.split(",", 1)]
                if not parts or not parts[0].isdigit():
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
            first = ln.split(None, 1)[0]
            if first.isdigit():
                pids.add(int(first))
        return pids
    except (OSError, subprocess.TimeoutExpired):
        return set()


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


def _sidecar_backend(pid_file: Path) -> str | None:
    """The backend (claude|opencode) a resolve sidecar belongs to, from its
    `.backend` sibling. A missing/unreadable sidecar returns None so a
    product-scoped count treats it conservatively as 'not this product' — a worker
    with no backend tag does not pin a specific pool's cap."""
    try:
        return pid_file.with_suffix(".backend").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def proc_worker_count(root: Path | None = None, *, product: str | None = None) -> int:
    """Count live worker processes that consume the dispatch cap.

    This combines the generic DOS-loop command-line marker with the issue
    resolver's pid sidecars. Use the union of pids so a worker visible through both
    witnesses is counted once; if psutil is absent, each witness is still best
    effort and may be conservatively low rather than fabricating capacity.

    When ``product`` is given, only workers in that account pool count — a claude
    worker no longer pins the opencode lane's cap and vice versa, so the two lanes
    fill to their independent account headrooms instead of starving one another.
    The generic cmdline-marked DOS-loop workers carry no backend tag, so they are
    counted only in the unscoped (global) call; a product-scoped count is the
    issue-resolver sidecars for that product alone.
    """
    root = root or repo_root()
    if product is not None:
        return len(live_resolve_worker_pids(root / RUNS_DIRNAME, product=product))
    pids = set(_cmdline_worker_pids())
    pids.update(live_resolve_worker_pids(root / RUNS_DIRNAME))
    return len(pids)


def _int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def evaluate(root: Path, *, max_workers: int, work_kind: str, product: str,
             max_threads: int | None = None) -> dict[str, Any]:
    host = host_check(root, max_threads=max_threads)
    acct = account_check(root, work_kind=work_kind, product=product)
    kern = kernel_alive(root)

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

    # Fail-safe ordering, evaluated top to bottom:
    #   1. an un-runnable host/kernel safety check  -> REFUSE_INSPECT (never assume safe)
    #   2. a flagged host                            -> REFUSE_HOST
    #   3. at/over the worker cap                    -> REFUSE_AT_CAP
    #   4. no available account (incl. switcher err) -> REFUSE_NO_ACCOUNT
    # An account check that merely errored is just "no account available", which
    # branch 4 already reports — it does not need to pre-empt host/cap.
    if host.get("error") or kern.get("error"):
        verdict = REFUSE_INSPECT
        reason = (host.get("error") or kern.get("error")
                  or "a preflight safety check could not run")
    elif not host["safe"]:
        verdict = REFUSE_HOST
        reason = (f"host resource guard flagged {host['flagged']} process(es): "
                  f"{', '.join(host.get('flagged_names') or []) or 'see proc_resource_guard'}"
                  " — reap/inspect before growing the fleet")
    elif live >= cap:
        verdict = REFUSE_AT_CAP
        reason = (f"live workers {live} >= cap {cap} "
                  f"(kernel alive={alive_kernel}, os procs={alive_proc}, "
                  f"dos target={target}, max-workers={max_workers})")
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
        "host": host,
        "account": {k: acct.get(k) for k in ("available", "tag", "dir", "tier", "model")},
        "kernel": kern,
        "os_worker_procs": alive_proc,
    }


def render(p: dict[str, Any]) -> str:
    a = p.get("account") or {}
    return "\n".join([
        f"dispatch preflight: {p.get('verdict')} ({'ok' if p.get('ok') else 'refuse'})",
        f"  reason: {p.get('reason')}",
        f"  live={p.get('live')}/{p.get('cap')} (headroom {p.get('headroom')})  "
        f"host={'clean' if (p.get('host') or {}).get('safe') else 'FLAGGED'}  "
        f"account={a.get('tag') or '-'} (t{a.get('tier')})",
    ])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Read-only spawn gate: is it safe to launch another dispatch worker?")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                    help=f"hard ceiling on live workers (default: {DEFAULT_MAX_WORKERS}); "
                         "effective cap is min(this, dos [supervise].target) when that "
                         "target is positive, else this (a zero/unset target lets the "
                         "cron-armed self-spawner use its own ceiling)")
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
