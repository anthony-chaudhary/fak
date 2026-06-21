#!/usr/bin/env python3
r"""One guarded, switcher-routed dispatch TICK that spawns an ISSUE-RESOLUTION
worker — the arm that moves the open-issue counter on a plan-empty repo.

``issue_dispatch.py`` spawns the generic ``/dos-kernel:dos-dispatch-loop``
worker, which resolves units from the *plan portfolio*. This public repo ships
no ``PLAN-*.md`` (``PLAN_SURFACE_EMPTY``), so that worker has no work surface and
closes nothing — the backlog lives in GitHub *issues*. This tick is the
issue-driven sibling: it picks ONE concrete open issue on the busiest lane,
renders a scoped resolution prompt (``issue_worker_prompt.py`` — with the
``#N``-in-subject rule that lets the closure auditor witness the resulting
commit), and launches one detached ``claude -p "<prompt>"`` worker to land it.

It shares every safety primitive with ``issue_dispatch.py`` (imported, not
re-implemented): the per-tick registry refresh (route off current account
evidence), the ``dispatch_preflight`` DoS gate (host clean ∧ account free ∧ live
< cap), the switcher-pinned account env, and the detached spawn. DRY-RUN BY
DEFAULT — prints the issue, account, command, and prompt path. ``--live`` spawns.

In-flight de-dup: it skips an issue that already has a live resolution worker (a
dispatch log naming ``#N`` whose process is still alive) so two ticks never storm
the same issue.

    python tools/issue_resolve_dispatch.py                 # plan one tick (dry-run)
    python tools/issue_resolve_dispatch.py --live          # spawn one issue worker
    python tools/issue_resolve_dispatch.py --lane gateway --live
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue_dispatch  # noqa: E402  (refresh_registry/preflight/worker_env/spawn_detached)
import issue_worker_prompt  # noqa: E402  (render the per-issue resolution prompt)

SCHEMA = "fleet-issue-resolve-dispatch/1"
RUNS_DIRNAME = ".dispatch-runs"
_LOG_ISSUE_RE = re.compile(r"resolve-(\d+)-")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _py() -> str:
    return sys.executable or "python"


def lane_issue_numbers(root: Path, explicit_lane: str | None) -> dict[str, Any]:
    """Pick the lane (busiest, or explicit) and return its OPEN issue numbers,
    most-recent first. Reuses the same router fold issue_dispatch.pick_lane uses,
    but keeps the per-issue numbers (which pick_lane discards)."""
    router = issue_dispatch.run_json(
        [_py(), str(root / "tools" / "issue_lane_router.py"), "--json"],
        root, timeout=130)
    lanes = router.get("lanes") or {}
    nums_by_lane: dict[str, list[int]] = {}
    for ln, info in lanes.items():
        iss = info.get("issues") if isinstance(info, dict) else info
        nums: list[int] = []
        for it in (iss or []):
            n = it.get("number") if isinstance(it, dict) else it
            try:
                nums.append(int(n))
            except (TypeError, ValueError):
                continue
        nums_by_lane[ln] = sorted(nums, reverse=True)
    if explicit_lane:
        chosen = explicit_lane
    elif nums_by_lane:
        chosen = max(nums_by_lane, key=lambda k: len(nums_by_lane[k]))
    else:
        chosen = None
    return {"lane": chosen, "numbers": nums_by_lane.get(chosen or "", []),
            "by_lane_count": {k: len(v) for k, v in nums_by_lane.items()},
            "router_error": router.get("_error")}


def live_resolution_issues(runs_dir: Path) -> set[int]:
    """Issue numbers that already have a LIVE resolution worker — read from the
    dispatch logs (``resolve-<N>-<stamp>.log``) whose pid is still alive. Best
    effort: a log without a recoverable pid is treated as not-live."""
    live: set[int] = set()
    if not runs_dir.is_dir():
        return live
    try:
        import psutil  # type: ignore
        alive = {p.pid for p in psutil.process_iter()}
    except ImportError:
        alive = set()
    for log in runs_dir.glob("resolve-*.log"):
        m = _LOG_ISSUE_RE.search(log.name)
        if not m:
            continue
        pid_file = log.with_suffix(".pid")
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue
            if not alive or pid in alive:
                live.add(int(m.group(1)))
    return live


def pick_target_issue(numbers: list[int], live: set[int]) -> int | None:
    """The first lane issue not already being resolved by a live worker."""
    for n in numbers:
        if n not in live:
            return n
    return None


def spawn_issue_worker(prompt: str, env: dict[str, str], cwd: Path,
                       log_dir: Path, issue: int, lane: str) -> dict[str, Any]:
    """Launch a detached ``claude -p "<prompt>"`` worker on one issue; record pid."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_log = log_dir / f"resolve-{issue}-{stamp}.log"
    command = ["claude", "-p", "--permission-mode", "bypassPermissions", prompt]
    exe = shutil.which(command[0]) or command[0]
    argv = [exe, *command[1:]]
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = 0x00000008  # DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    fh = open(out_log, "w", encoding="utf-8")
    proc = subprocess.Popen(argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                            stdout=fh, stderr=subprocess.STDOUT, **kwargs)
    (out_log.with_suffix(".pid")).write_text(str(proc.pid), encoding="utf-8")
    return {"pid": proc.pid, "log": str(out_log), "issue": issue, "lane": lane}


def evaluate(root: Path, *, max_workers: int, work_kind: str, lane: str | None,
             live: bool, refresh: bool = True) -> dict[str, Any]:
    reg = issue_dispatch.refresh_registry(root) if refresh else {"skipped": True}
    pre = issue_dispatch.preflight(root, max_workers=max_workers, work_kind=work_kind)
    pre_ok = pre.get("verdict") == "SPAWN_OK"
    acct = pre.get("account") or {}

    pick = lane_issue_numbers(root, lane)
    chosen_lane = pick.get("lane")
    runs_dir = root / RUNS_DIRNAME
    live_issues = live_resolution_issues(runs_dir)
    target = pick_target_issue(pick.get("numbers") or [], live_issues)

    payload: dict[str, Any] = {
        "schema": SCHEMA, "workspace": str(root), "live": live,
        "max_workers": max_workers, "registry_refresh": reg,
        "preflight": {"verdict": pre.get("verdict"), "reason": pre.get("reason"),
                      "cap": pre.get("cap"), "live": pre.get("live")},
        "account": {k: acct.get(k) for k in ("tag", "tier", "model", "dir")},
        "lane": chosen_lane, "lane_issue_count": len(pick.get("numbers") or []),
        "target_issue": target,
        "already_live": sorted(live_issues),
    }

    if not pre_ok:
        payload.update({"ok": False, "action": "refused",
                        "verdict": pre.get("verdict") or "REFUSE",
                        "reason": f"preflight refused: {pre.get('reason')}"})
        return payload
    if not chosen_lane:
        payload.update({"ok": False, "action": "no_lane", "verdict": "NO_LANE",
                        "reason": "no lane has open issues (router empty/error)"})
        return payload
    if target is None:
        payload.update({"ok": False, "action": "no_issue", "verdict": "NO_ISSUE",
                        "reason": (f"every open issue on lane '{chosen_lane}' "
                                   f"already has a live resolution worker "
                                   f"({sorted(live_issues)})")})
        return payload

    rec = issue_worker_prompt.build(target, chosen_lane, workspace=root)
    payload["prompt_chars"] = rec.get("prompt_chars")
    payload["issue_title"] = rec.get("title")
    command_preview = ["claude", "-p", "--permission-mode", "bypassPermissions",
                       f"<resolve #{target} prompt, {rec.get('prompt_chars')} chars>"]
    payload["command"] = command_preview

    if not live:
        payload.update({"ok": True, "action": "would_spawn", "verdict": "WOULD_SPAWN",
                        "reason": (f"safe to spawn 1 issue-resolution worker on #{target} "
                                   f"(lane '{chosen_lane}') under account '{acct.get('tag')}' "
                                   f"(t{acct.get('tier')})")})
        return payload

    env = issue_dispatch.worker_env(acct.get("dir"), chosen_lane, root)
    env["FLEET_RESOLVE_ISSUE"] = str(target)
    spawned = spawn_issue_worker(rec["prompt"], env, root, runs_dir, target, chosen_lane)
    payload.update({"ok": True, "action": "spawned", "verdict": "SPAWNED",
                    "spawned": spawned,
                    "reason": (f"spawned issue-resolution worker pid {spawned['pid']} on "
                               f"#{target} (lane '{chosen_lane}') under '{acct.get('tag')}'")})
    _record(runs_dir, payload)
    return payload


def _record(runs_dir: Path, payload: dict[str, Any]) -> None:
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "last-resolve-tick.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def render(p: dict[str, Any]) -> str:
    a = p.get("account") or {}
    pf = p.get("preflight") or {}
    lines = [
        f"issue-resolve-dispatch: {p.get('verdict')} ({'ok' if p.get('ok') else 'refuse'})  live={p.get('live')}",
        f"  preflight : {pf.get('verdict')} ({pf.get('live')}/{pf.get('cap')} live)",
        f"  account   : {a.get('tag') or '-'} (t{a.get('tier')})  {a.get('model') or ''}",
        f"  lane      : {p.get('lane') or '-'}  ({p.get('lane_issue_count')} issues)",
        f"  target    : #{p.get('target_issue')}  {(p.get('issue_title') or '')[:54]}",
        f"  -> {p.get('reason')}",
    ]
    if p.get("spawned"):
        s = p["spawned"]
        lines.append(f"  spawned pid={s.get('pid')} issue=#{s.get('issue')} log={s.get('log')}")
    if not p.get("live") and p.get("action") == "would_spawn":
        lines.append("  DRY-RUN — re-run with --live to spawn the issue worker")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="One guarded tick that spawns an issue-RESOLUTION worker (dry-run by default).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--max-workers", type=int, default=2,
                    help="hard cap on live workers, enforced by the preflight (default: 2)")
    ap.add_argument("--work-kind", default="engineering",
                    help="switcher work kind (engineering->t1, gardening->t2)")
    ap.add_argument("--lane", default=None,
                    help="explicit lane (default: the lane with the most open issues)")
    ap.add_argument("--live", action="store_true",
                    help="actually spawn the worker (default: dry-run / plan only)")
    ap.add_argument("--no-refresh", action="store_true",
                    help="skip the per-tick account-registry refresh")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = evaluate(root, max_workers=args.max_workers, work_kind=args.work_kind,
                       lane=args.lane, live=args.live, refresh=not args.no_refresh)
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
