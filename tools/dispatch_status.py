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
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dispatch_preflight  # noqa: E402  (pid-sidecar identity probe)

SCHEMA = "fleet-dispatch-status/1"
# The guarded always-on tick (tools/register_issue_dispatch.ps1). The older
# FleetDOSDispatchWatchdog keeps the un-gated kernel supervisor alive; this card
# tracks the DoS-safe issue dispatcher, so it reports the guarded task.
WATCHDOG_TASK = "FleetIssueDispatch"

RUNS_DIRNAME = ".dispatch-runs"
# resolve-<N>-<stamp>.log written by issue_resolve_dispatch.spawn_issue_worker.
_RESOLVE_LOG_RE = re.compile(r"resolve-(\d+)-(\d{8}-\d{6})\.log$")


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


def silent_workers(
    runs_dir: Path,
    *,
    alive: set[int] | None = None,
    probe: Any | None = None,
) -> list[dict[str, Any]]:
    """Issue-resolution workers that exited having produced NOTHING — a 0-byte
    ``resolve-<N>-<stamp>.log`` whose ``.pid`` process is dead.

    A ``claude -p`` worker writes nothing to stdout until its final message, so a
    0-byte log with a *live* pid is still-running (not silent) and is excluded. A
    dead pid over an empty log is the "spun, produced nothing" residual the cooldown
    self-corrects but leaves operator-invisible — this is the signal. Best effort:
    psutil is optional (its absence means we cannot prove a pid dead, so we report
    nothing rather than a false silent), exactly like
    ``issue_resolve_dispatch.live_resolution_issues``. Newest first.
    """
    if not runs_dir.is_dir():
        return []
    if alive is None and probe is None:
        try:
            import psutil  # type: ignore

            alive = {p.pid for p in psutil.process_iter()}
        except ImportError:
            alive = None  # cannot prove liveness -> report no silents (no false alarms)
    out: list[dict[str, Any]] = []
    for log in runs_dir.glob("resolve-*.log"):
        m = _RESOLVE_LOG_RE.search(log.name)
        if not m:
            continue
        try:
            if log.stat().st_size != 0:
                continue  # produced output -> not silent
        except OSError:
            continue
        pid_file = log.with_suffix(".pid")
        if not pid_file.exists():
            continue
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if alive is None and probe is None:
            continue  # no liveness oracle this run -> do not claim it is silent
        if dispatch_preflight.resolve_sidecar_pid_is_live(
            pid_file, alive=alive, probe=probe):
            continue  # still running -> not (yet) silent
        out.append({"issue": int(m.group(1)), "stamp": m.group(2), "log": log.name, "pid": pid})
    out.sort(key=lambda r: r["stamp"], reverse=True)
    return out


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


def read_active_weekly_cap(runs_dir: Path, account_tag: str | None,
                           now_ts: float | None = None) -> dict[str, Any] | None:
    """The active weekly-cap hold for ``account_tag`` (if any), read from the
    dispatcher's persisted state (``account-cap-*.json``, written by the
    issue_resolve_dispatch weekly-cap gate). None when no unexpired hold matches.
    Read-only / best-effort, so the card can show WHY a logged-in account is held."""
    import datetime as _dt
    import time as _time
    try:
        now_ts = _time.time() if now_ts is None else now_ts
        now = _dt.datetime(1970, 1, 1) + _dt.timedelta(seconds=now_ts)
    except Exception:
        return None
    if not runs_dir.is_dir():
        return None
    best: tuple[dict[str, Any], _dt.datetime] | None = None
    for path in runs_dir.glob("account-cap-*.json"):
        try:
            st = json.loads(path.read_text(encoding="utf-8"))
            until = _dt.datetime.fromisoformat((st.get("until") or "").replace("Z", ""))
        except (OSError, ValueError):
            continue
        if now >= until or (account_tag and st.get("account") not in (None, account_tag)):
            continue
        if best is None or until < best[1]:
            best = (st, until)
    return best[0] if best else None


def read_backend_health(runs_dir: Path) -> list[dict[str, Any]]:
    """The backends currently held DEAD by the dispatcher's backend-health gate, read
    from ``backend-health-*.json`` (written by issue_resolve_dispatch.check_backend_health
    when a backend spins on a banner-only/0-byte streak). Each row carries the product,
    since-when, the lane reallocated to a healthy backend, and the evidence logs — so the
    card shows WHY a backend stopped spawning and where its work went. Read-only /
    best-effort; a corrupt or healthy sidecar is skipped. Newest-dead first."""
    out: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return out
    for path in runs_dir.glob("backend-health-*.json"):
        try:
            st = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if st.get("state") != "dead":
            continue
        out.append({k: st.get(k) for k in
                    ("product", "since", "abandoned_lane", "evidence_logs",
                     "reprobe_min", "last_reprobe")})
    out.sort(key=lambda r: str(r.get("since") or ""), reverse=True)
    return out


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
    # The RATE fold (closed/hour vs target) — the observable the loop's goal is
    # actually stated in. gh-backed, so it degrades to n/a under --fast/timeout
    # exactly like backlog/closure; it never flips the dispatcher-health verdict
    # (a below-target rate is information, not a broken dispatcher).
    throughput: dict[str, Any] = {"_skipped": "fast"} if fast else run_json(
        [_py(), str(root / "tools" / "dispatch_throughput.py"), "--json"],
        root, timeout=140, ok_codes=set(range(0, 16)))

    # Pure-local, always run (no gh/dos): which spawned workers exited producing
    # nothing. Cheap enough that --fast keeps it.
    silent = silent_workers(root / RUNS_DIRNAME)
    weekly_cap = read_active_weekly_cap(root / RUNS_DIRNAME,
                                        (pre.get("account") or {}).get("tag"))
    backend_health = read_backend_health(root / RUNS_DIRNAME)

    return build_payload(root=root, pre=pre, sup=sup, wd=wd, backlog=backlog,
                         closure=closure, max_workers=max_workers, fast=fast,
                         silent=silent, weekly_cap=weekly_cap, throughput=throughput,
                         backend_health=backend_health)


def build_payload(*, root: Path, pre: dict, sup: dict, wd: dict, backlog: dict,
                  closure: dict, max_workers: int, fast: bool,
                  silent: list[dict[str, Any]] | None = None,
                  weekly_cap: dict[str, Any] | None = None,
                  throughput: dict[str, Any] | None = None,
                  backend_health: list[dict[str, Any]] | None = None) -> dict[str, Any]:
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

    # --- throughput (closed/hour vs target) ---
    throughput = throughput or {}
    tp_na = "_skipped" in throughput or "_error" in throughput or not throughput.get("schema")

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

    # A logged-in-but-quota-capped account makes the preflight read SPAWN_OK while
    # the dispatcher's weekly-cap gate is actually HOLDING. Surface that so the card
    # says WEEKLY_CAPPED, not the misleading READY_TO_GROW. Holding is a healthy
    # steady state (the t2 glm pool is unaffected), so ok stays True.
    if weekly_cap and verdict == "READY_TO_GROW":
        verdict = "WEEKLY_CAPPED"
        reasons = [f"account '{acct.get('tag')}' weekly-capped — resets "
                   f"{weekly_cap.get('reset_text') or '?'} (holding spawn until "
                   f"{weekly_cap.get('until')}); the t2 glm/docs pool is unaffected"]

    if wd.get("installed") is False:
        reasons.append("always-on watchdog NOT installed (register_dos_dispatch_watchdog.ps1)")
    elif wd.get("installed"):
        reasons.append(f"always-on watchdog installed ({wd.get('status') or 'scheduled'})")

    silent = silent or []
    if silent:
        nums = ", ".join(f"#{w['issue']}" for w in silent[:6])
        reasons.append(f"{len(silent)} worker(s) exited producing nothing ({nums}) — inspect or re-scope")
    backend_health = backend_health or []
    if backend_health:
        names = ", ".join(f"{b.get('product')}->{b.get('abandoned_lane') or '?'}"
                          for b in backend_health[:4])
        reasons.append(f"{len(backend_health)} backend(s) held dead, lane reallocated "
                       f"({names}) — a healthy backend is covering; auto-restores on recovery")

    if not tp_na:
        tp_verdict = throughput.get("verdict")
        tp_rate = throughput.get("completed_rate_per_hour")
        tp_target = throughput.get("target_per_hour")
        win = throughput.get("primary_window_hours")
        if tp_verdict in ("BELOW_TARGET", "AUDIT_ERROR"):
            reasons.append(f"throughput {tp_rate}/h completed over the {win}h analysis window — below the "
                           f"{tp_target}/h target")
        else:
            reasons.append(f"throughput {tp_verdict} ({tp_rate}/h completed over the {win}h analysis window, "
                           f"target {tp_target}/h)")

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "reasons": reasons,
        "workspace": str(root),
        "weekly_cap": weekly_cap,
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
        "throughput": {
            "na": tp_na,
            "verdict": None if tp_na else throughput.get("verdict"),
            "target_per_hour": None if tp_na else throughput.get("target_per_hour"),
            "primary_window_hours": None if tp_na else throughput.get("primary_window_hours"),
            "completed_rate_per_hour": None if tp_na else throughput.get("completed_rate_per_hour"),
            "raw_rate_per_hour": None if tp_na else throughput.get("raw_rate_per_hour"),
            "per_window": None if tp_na else (throughput.get("gh") or {}).get("per_window"),
            "loop_per_window": None if tp_na else (throughput.get("loop") or {}).get("per_window"),
            "last_loop_close_age_min": None if tp_na else (throughput.get("loop") or {}).get("last_loop_close_age_min"),
        },
        "workers": {
            "silent_count": len(silent),
            "silent": silent,
        },
        "backend_health": {
            "dead_count": len(backend_health),
            "dead": backend_health,
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
    tp = p.get("throughput") or {}
    if tp.get("na"):
        lines.append("║ rate      : n/a (--fast or gh timeout)")
    else:
        lines.append(
            f"║ rate      : {tp.get('verdict')}  {tp.get('completed_rate_per_hour')}/h completed "
            f"over the {tp.get('primary_window_hours')}h analysis window (target {tp.get('target_per_hour')}/h)")
    w = p.get("workers") or {}
    sc = w.get("silent_count") or 0
    if sc:
        nums = ", ".join(f"#{s['issue']}" for s in (w.get("silent") or [])[:6])
        lines.append(f"║ workers   : {sc} silent (0-byte log, exited) [{nums}]")
    lines.append("╚═ " + " | ".join(p.get("reasons") or []))
    return "\n".join(lines)


def render_md(payload: dict[str, Any], *, date: str) -> str:
    """The committed, human-readable status surface: which issues are synced to
    which lanes, how closure is progressing, and any worker that produced nothing.

    This is the plan-doc-equivalent for a plan-empty repo whose backlog is GitHub
    issues — an operator opens ONE file instead of grepping gitignored runtime.
    Date is git-derived by the caller (deterministic; the renderer takes no clock).
    """
    d = payload.get("dispatcher") or {}
    s = payload.get("supervisor") or {}
    b = payload.get("backlog") or {}
    c = payload.get("closure") or {}
    w = payload.get("workers") or {}
    a = d.get("account") or {}
    wd = d.get("watchdog") or {}

    out = [
        # Jekyll front matter so the published status page keeps a stable <title> +
        # meta description (jekyll-seo-tag reads these). Without it, every --md regen
        # would strip the front matter the committed doc needs and the page would
        # read as discoverability debt to the SEO/AEO scorecard. Title/description are
        # deliberately date-independent so the page's identity is stable across regens.
        "---",
        'title: "fak issue dispatch status: fleet worker and backlog tracker"',
        'description: "Auto-generated fak issue dispatch status: dispatcher and worker '
        'state, open-issue backlog by lane, and closure-honesty rate across the fleet."',
        "---",
        "",
        f"# Issue dispatch status — {date}",
        "",
        "_Auto-generated by `tools/dispatch_status.py --md`. Do not hand-edit; "
        "re-run the tool (or the `FleetDispatchStatusDoc` task) to refresh._",
        "",
        f"- **dispatcher**: `{payload.get('verdict')}` "
        f"({'ok' if payload.get('ok') else 'ACTION'})",
        f"- **workers**: {d.get('live')}/{d.get('cap')} live "
        f"(headroom {d.get('headroom')}); host "
        f"{'clean' if d.get('host_safe') else '**FLAGGED**'}",
        f"- **switcher account**: `{a.get('tag') or '-'}` (t{a.get('tier')}, "
        f"{a.get('model') or '?'}), available={a.get('available')}",
        "- **always-on watchdog**: "
        + ("installed (" + str(wd.get('status') or 'scheduled') + ")"
           if wd.get("installed") else
           ("**NOT installed**" if wd.get("installed") is False else "unknown")),
        f"- **supervisor**: `{s.get('verdict')}` "
        f"(alive {s.get('alive')}/{s.get('target')})",
        "",
        "## Backlog by lane (issue → lane sync)",
        "",
    ]
    if b.get("na"):
        out.append("_Backlog n/a this run (gh fold skipped or timed out)._")
    else:
        out += [
            f"Open issues: **{b.get('open_issues')}** — routed {b.get('routed')}, "
            f"unrouted {b.get('unrouted')}.",
            "",
            "| lane | open issues |",
            "|---|---|",
        ]
        by_lane = b.get("by_lane") or {}
        for lane, n in sorted(by_lane.items(), key=lambda kv: (-kv[1], kv[0])):
            out.append(f"| {lane} | {n} |")

    out += ["", "## Closure honesty", ""]
    if c.get("na"):
        out.append("_Closure audit n/a this run (gh/dos fold skipped or timed out)._")
    else:
        cnt = c.get("counts") or {}
        out += [
            f"`closure_rate` = **{c.get('closure_rate')}** "
            f"(TRUE_RESOLVED / (TRUE_RESOLVED + CLAIMED_CLOSED)).",
            "",
            "| bucket | count |",
            "|---|---|",
            f"| TRUE_RESOLVED | {cnt.get('TRUE_RESOLVED', 0)} |",
            f"| CLAIMED_CLOSED | {cnt.get('CLAIMED_CLOSED', 0)} |",
            f"| OPEN_WITNESSED (closable now) | {c.get('open_witnessed_closable')} |",
        ]

    tp = payload.get("throughput") or {}
    out += ["", "## Throughput (closed issues per hour)", ""]
    if tp.get("na"):
        out.append("_Throughput n/a this run (gh fold skipped or timed out)._")
    else:
        out += [
            f"`verdict` = **{tp.get('verdict')}** — **{tp.get('completed_rate_per_hour')}/h** "
            f"completed over the trailing **{tp.get('primary_window_hours')}h** window "
            f"(target **{tp.get('target_per_hour')}/h**). Graded on the *completed* "
            "(resolved, not wontfix/dup) rate.",
            "",
            "| window | closed | completed | completed /h | loop-closed | loop /h |",
            "|---|---|---|---|---|---|",
        ]
        pw = tp.get("per_window") or {}
        lpw = tp.get("loop_per_window") or {}
        for key in ("1h", "3h", "6h", "12h", "24h"):
            g = pw.get(key)
            if not g:
                continue
            lp = lpw.get(key) or {}
            out.append(
                f"| {key} | {g.get('closed')} | {g.get('completed')} | "
                f"{g.get('completed_rate_per_hour')} | {lp.get('loop_closed', '-')} | "
                f"{lp.get('loop_rate_per_hour', '-')} |")
        last = tp.get("last_loop_close_age_min")
        out += ["",
                "Loop's last attributed close: "
                + (f"{last} min ago." if last is not None else "**none on record**.")
                + " A gh-rate far above the loop-rate means humans/peers are draining "
                "the backlog, not the dispatcher."]

    bh = payload.get("backend_health") or {}
    dead = bh.get("dead") or []
    out += ["", "## Backend health / reallocation", ""]
    if not dead:
        out.append("All backends healthy — none held dead, no lane reallocated.")
    else:
        out += [
            f"**{len(dead)}** backend(s) are spinning dead (a streak of banner-only / "
            "0-byte worker logs the weekly-cap gate doesn't catch — e.g. a credit-walled "
            "codex or a glm worker that prints only its startup banner). The dispatcher "
            "holds their spawns and a healthy backend claims the freed lane + budget; "
            "one re-probe worker is admitted per interval, so each auto-restores the "
            "moment it produces a real turn again.",
            "",
            "| backend | dead since | lane reallocated | re-probe (min) |",
            "|---|---|---|---|",
        ]
        for b in dead:
            out.append(f"| {b.get('product')} | {b.get('since')} | "
                       f"{b.get('abandoned_lane') or '—'} | {b.get('reprobe_min') or '—'} |")

    sc = w.get("silent_count") or 0
    out += ["", "## Workers that produced nothing", ""]
    if not sc:
        out.append("None — every spawned worker either produced output or is still running.")
    else:
        out += [
            f"**{sc}** worker(s) exited with a 0-byte log (spawned, committed nothing). "
            "The anti-churn cooldown advances the picker past these, so the loop still "
            "progresses — but each is worth an operator's eye (often an epic-shaped "
            "issue too large to land in one shot).",
            "",
            "| issue | spawned (utc stamp) | log |",
            "|---|---|---|",
        ]
        for sw in (w.get("silent") or []):
            out.append(f"| #{sw.get('issue')} | {sw.get('stamp')} | `{sw.get('log')}` |")

    out += ["", "---", "", "Reasons: " + "; ".join(payload.get("reasons") or [])]
    return "\n".join(out) + "\n"


def git_date(root: Path) -> str:
    """The last-commit date (YYYY-MM-DD) — deterministic, no wall-clock in the tool."""
    try:
        proc = subprocess.run(["git", "log", "-1", "--format=%cs"], cwd=str(root),
                              capture_output=True, text=True, timeout=15)
        date = (proc.stdout or "").strip()
        return date or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="One-touch always-on dispatcher status card.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--max-workers", type=int, default=2,
                    help="cap used by the spawn-gate preflight (default: 2)")
    ap.add_argument("--fast", action="store_true",
                    help="skip the two gh-backed folds (backlog + closure); pure-local")
    ap.add_argument("--closure-commits", type=int, default=400,
                    help="git history budget for the closure audit (default: 400)")
    ap.add_argument("--md", default="",
                    help="write the committed markdown status doc to this path "
                         "(forces the full fold; --fast is ignored when --md is set)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    # The committed doc must carry the real backlog/closure tables, so --md always
    # runs the full fold regardless of --fast.
    fast = args.fast and not args.md
    payload = collect(root, max_workers=args.max_workers, fast=fast,
                      closure_commits=args.closure_commits)

    if args.md:
        md_path = Path(args.md)
        if not md_path.is_absolute():
            md_path = root / md_path
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_md(payload, date=git_date(root)), encoding="utf-8")
        if not args.json:
            print(f"wrote {md_path} ({payload.get('verdict')}, "
                  f"open={ (payload.get('backlog') or {}).get('open_issues') }, "
                  f"silent={ (payload.get('workers') or {}).get('silent_count') })")

    if args.json:
        print(json.dumps(payload, indent=2))
    elif not args.md:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
