#!/usr/bin/env python3
r"""THE one-touch status card for the always-on issue dispatcher.

One command, one screen, the whole loop: is the always-on watchdog installed and
ticking, can it grow right now (the DoS-safe spawn gate), how many workers are
live, which account the switcher would pick, how big the open-issue backlog is,
and — the load-bearing honesty metric — what fraction of *closed* issues are
TRULY resolved (a DOS-witnessed resolving commit in git) versus merely
CLAIMED_CLOSED.

It is a pure FOLD over tools that already exist; it launches nothing and writes
nothing:

  * tools/dispatch_preflight.py   host-guard ∧ account-free ∧ under-cap  (fast)
  * tools/dos_supervisor_status.py  the [supervise] readiness card       (fast)
  * schtasks query                 is FleetDOSDispatchWatchdog installed? (fast)
  * tools/issue_lane_router.py     open backlog mapped to lanes          (gh; slow)
  * tools/issue_closure_audit.py   closure_rate = resolved/claimed       (gh; slow)

The two gh-backed folds are bounded and degrade to "n/a" on timeout, so the card
always returns promptly. ``--fast`` skips them entirely (pure-local, sub-5s).
Exit 0 when the dispatcher is healthy (safe to grow OR already at a healthy
target and host clean), 1 when something needs an operator's eye.

    python tools/dispatch_status.py            # the card
    python tools/dispatch_status.py --json     # machine-readable
    python tools/dispatch_status.py --fast      # skip gh-backed folds
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fleet-dispatch-status/1"
# The guarded always-on tick (tools/register_issue_dispatch.ps1). The older
# FleetDOSDispatchWatchdog keeps the un-gated kernel supervisor alive; this card
# tracks the DoS-safe issue dispatcher, so it reports the guarded task.
WATCHDOG_TASK = "FleetIssueDispatch"


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _py() -> str:
    return sys.executable or "python"


def run_json(cmd: list[str], cwd: Path, timeout: int,
             ok_codes: set[int] | None = None) -> dict[str, Any]:
    ok_codes = ok_codes if ok_codes is not None else set(range(0, 16))
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"_error": f"timed out after {timeout}s", "_cmd": cmd}
    except OSError as exc:
        return {"_error": str(exc), "_cmd": cmd}
    doc = _last_json(proc.stdout)
    doc.setdefault("_returncode", proc.returncode)
    if proc.returncode not in ok_codes and "_error" not in doc and not doc.get("schema"):
        doc["_error"] = (proc.stderr or proc.stdout or "").strip()[-300:]
    return doc


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


def watchdog_installed() -> dict[str, Any]:
    """Is the always-on watchdog scheduled task registered, and is it enabled?"""
    try:
        proc = subprocess.run(
            ["schtasks", "/Query", "/TN", WATCHDOG_TASK, "/FO", "LIST"],
            capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"installed": None, "error": str(exc)}
    if proc.returncode != 0:
        return {"installed": False, "status": None}
    status = None
    for line in proc.stdout.splitlines():
        if line.lower().strip().startswith("status:"):
            status = line.split(":", 1)[1].strip()
    return {"installed": True, "status": status}


def _int(v: Any, d: int | None = None) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return d


def collect(root: Path, *, max_workers: int, fast: bool,
            closure_commits: int) -> dict[str, Any]:
    pre = run_json([_py(), str(root / "tools" / "dispatch_preflight.py"),
                    "--json", "--max-workers", str(max_workers)], root, timeout=120)
    sup = run_json([_py(), str(root / "tools" / "dos_supervisor_status.py"),
                    "--json"], root, timeout=90)
    wd = watchdog_installed()

    backlog: dict[str, Any] = {"_skipped": "fast"} if fast else run_json(
        [_py(), str(root / "tools" / "issue_lane_router.py"), "--json"], root,
        timeout=130)
    closure: dict[str, Any] = {"_skipped": "fast"} if fast else run_json(
        [_py(), str(root / "tools" / "issue_closure_audit.py"), "--json",
         "--max-commits", str(closure_commits)], root, timeout=180)

    return build_payload(root=root, pre=pre, sup=sup, wd=wd, backlog=backlog,
                         closure=closure, max_workers=max_workers, fast=fast)


def build_payload(*, root: Path, pre: dict, sup: dict, wd: dict, backlog: dict,
                  closure: dict, max_workers: int, fast: bool) -> dict[str, Any]:
    # --- dispatcher liveness / capacity ---
    cap = _int(pre.get("cap"))
    live = _int(pre.get("live"))
    host_safe = bool((pre.get("host") or {}).get("safe"))
    acct = pre.get("account") or {}
    pre_verdict = pre.get("verdict")

    # --- backlog --- (counts is the router's authoritative routed/unrouted fold)
    lanes = (backlog.get("lanes") or {}) if isinstance(backlog, dict) else {}
    bcounts = (backlog.get("counts") or {}) if isinstance(backlog, dict) else {}
    lane_counts: dict[str, int] = {}
    for ln, info in lanes.items():
        iss = info.get("issues") if isinstance(info, dict) else info
        lane_counts[ln] = len(iss) if hasattr(iss, "__len__") else _int(iss, 0) or 0
    open_issues = _int(bcounts.get("open"), sum(lane_counts.values()))
    routed = _int(bcounts.get("routed"))
    unrouted = _int(bcounts.get("unrouted"))
    backlog_na = "_skipped" in backlog or ("_error" in backlog and not lanes)

    # --- closure honesty ---
    counts = closure.get("counts") or {}
    closure_rate = closure.get("closure_rate")
    closure_na = "_skipped" in closure or ("_error" in closure and closure_rate is None)
    open_witnessed = _int(counts.get("OPEN_WITNESSED"), 0)

    # --- overall verdict ---
    # Healthy = host clean AND (can grow OR already at a healthy target). A flagged
    # host or an un-runnable safety check is the only thing that fails the card —
    # "no account free" / "at cap" are normal steady states, not breakage.
    reasons: list[str] = []
    if not host_safe:
        ok = False
        verdict = "HOST_FLAGGED"
        reasons.append("host resource guard flagged a process — reap/inspect before growing")
    elif pre_verdict == "REFUSE_INSPECT":
        ok = False
        verdict = "INSPECT"
        reasons.append(f"a safety preflight could not run: {pre.get('reason')}")
    elif pre_verdict == "REFUSE_NO_ACCOUNT":
        ok = True
        verdict = "BLOCKED_ON_ACCOUNT"
        reasons.append("no worker account free right now (switcher will resume when one frees)")
    elif pre_verdict == "REFUSE_AT_CAP":
        ok = True
        verdict = "AT_CAP"
        reasons.append(f"{live}/{cap} workers live — at the configured ceiling")
    else:
        ok = True
        verdict = "READY_TO_GROW"
        reasons.append(f"safe to spawn: {live}/{cap} live, account '{acct.get('tag')}' free")

    if wd.get("installed") is False:
        reasons.append("always-on watchdog NOT installed (register_dos_dispatch_watchdog.ps1)")
    elif wd.get("installed"):
        reasons.append(f"always-on watchdog installed ({wd.get('status') or 'scheduled'})")

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "reasons": reasons,
        "workspace": str(root),
        "dispatcher": {
            "cap": cap,
            "live": live,
            "headroom": (cap - live) if (cap is not None and live is not None) else None,
            "host_safe": host_safe,
            "preflight_verdict": pre_verdict,
            "account": {k: acct.get(k) for k in ("tag", "tier", "model", "available")},
            "watchdog": wd,
        },
        "supervisor": {
            "verdict": sup.get("verdict"),
            "target": (sup.get("supervise") or {}).get("target"),
            "alive": (sup.get("supervise") or {}).get("alive"),
            "plans": sup.get("plans"),
        },
        "backlog": {
            "na": backlog_na,
            "open_issues": None if backlog_na else open_issues,
            "routed": None if backlog_na else routed,
            "by_lane": None if backlog_na else lane_counts,
            "unrouted": None if backlog_na else unrouted,
        },
        "closure": {
            "na": closure_na,
            "closure_rate": closure_rate,
            "counts": counts or None,
            "open_witnessed_closable": None if closure_na else open_witnessed,
        },
        "fast": fast,
    }


def render(p: dict[str, Any]) -> str:
    d = p.get("dispatcher") or {}
    s = p.get("supervisor") or {}
    b = p.get("backlog") or {}
    c = p.get("closure") or {}
    a = d.get("account") or {}
    wd = d.get("watchdog") or {}
    lines = [
        f"╔═ DISPATCHER: {p.get('verdict')} ({'ok' if p.get('ok') else 'ACTION'})",
        f"║ workers   : {d.get('live')}/{d.get('cap')} live (headroom {d.get('headroom')})  "
        f"host={'clean' if d.get('host_safe') else 'FLAGGED'}",
        f"║ switcher  : account={a.get('tag') or '-'} (t{a.get('tier')}) "
        f"avail={a.get('available')}  preflight={d.get('preflight_verdict')}",
        "║ always-on : watchdog "
        + ("installed (" + str(wd.get('status') or 'scheduled') + ")"
           if wd.get("installed") else
           ("NOT installed" if wd.get("installed") is False else "unknown")),
        f"║ supervisor: {s.get('verdict')} alive={s.get('alive')}/{s.get('target')}  "
        f"plans={(s.get('plans') or {}).get('total_plans')} "
        f"units={(s.get('plans') or {}).get('total_units')}",
    ]
    if b.get("na"):
        lines.append("║ backlog   : n/a (--fast or gh timeout)")
    else:
        by = b.get("by_lane") or {}
        top = ", ".join(f"{k}={v}" for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:5])
        lines.append(f"║ backlog   : {b.get('open_issues')} open  [{top}]  unrouted={b.get('unrouted')}")
    if c.get("na"):
        lines.append("║ closure   : n/a (--fast or gh timeout)")
    else:
        cnt = c.get("counts") or {}
        lines.append(
            f"║ closure   : rate={c.get('closure_rate')}  "
            f"resolved={cnt.get('TRUE_RESOLVED')} claimed={cnt.get('CLAIMED_CLOSED')} "
            f"closable-now={c.get('open_witnessed_closable')} (OPEN_WITNESSED)")
    lines.append("╚═ " + " | ".join(p.get("reasons") or []))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="One-touch always-on dispatcher status card.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--max-workers", type=int, default=2,
                    help="cap used by the spawn-gate preflight (default: 2)")
    ap.add_argument("--fast", action="store_true",
                    help="skip the two gh-backed folds (backlog + closure); pure-local")
    ap.add_argument("--closure-commits", type=int, default=400,
                    help="git history budget for the closure audit (default: 400)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(root, max_workers=args.max_workers, fast=args.fast,
                      closure_commits=args.closure_commits)
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
