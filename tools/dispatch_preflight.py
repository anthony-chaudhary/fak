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
                     cap = min(dos [supervise].target, --max-workers) and
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
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

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
                              timeout=timeout)
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


def account_check(root: Path, *, work_kind: str, product: str) -> dict[str, Any]:
    """fleet_accounts.py route ⇒ the switcher's pick. ok+account ⇒ a free worker."""
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


def proc_worker_count() -> int:
    """OS-level count of live `dos-dispatch-loop` worker processes — the honest
    DoS signal: a real process consuming threads/handles, leased or not. Best
    effort across platforms; 0 if it cannot be determined (the kernel count then
    governs)."""
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None
    if psutil is not None:
        n = 0
        for p in psutil.process_iter(["cmdline"]):
            try:
                cl = " ".join(p.info.get("cmdline") or [])
            except (psutil.Error, TypeError):
                continue
            if WORKER_CMD_MARKER in cl:
                n += 1
        return n
    # Fallback when psutil is absent. wmic.exe is removed on Win11 24H2+ (build
    # 26200), so use the supported CIM API; Win32_Process.CommandLine carries the
    # full argv (incl. the prompt/marker), so this is an honest worker count, not 0.
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name='claude.exe'\" "
                 "| ForEach-Object { $_.CommandLine }"],
                capture_output=True, text=True, timeout=30).stdout
            return sum(1 for ln in out.splitlines() if WORKER_CMD_MARKER in ln)
        except (OSError, subprocess.TimeoutExpired):
            return 0
    try:
        out = subprocess.run(["pgrep", "-fa", WORKER_CMD_MARKER],
                             capture_output=True, text=True, timeout=20).stdout
        return sum(1 for ln in out.splitlines() if ln.strip())
    except (OSError, subprocess.TimeoutExpired):
        return 0


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
    cap = max_workers if target is None else min(max_workers, target)
    cap = max(0, cap)
    alive_kernel = kern.get("alive")
    alive_proc = proc_worker_count()
    # MAX of the two views: neither a stale lease nor an unleased orphan hides load.
    live = max(alive_kernel or 0, alive_proc)
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
                         "effective cap is min(this, dos [supervise].target)")
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
