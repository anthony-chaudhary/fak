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
import dispatch_worker  # noqa: E402  (child_env for the opencode backend)

SCHEMA = "fleet-issue-resolve-dispatch/1"
RUNS_DIRNAME = ".dispatch-runs"
_LOG_ISSUE_RE = re.compile(r"resolve-(\d+)-")

# Worker backends this tick can launch:
#   claude   = opus (t1) -- the reference path, the established quota pool.
#   opencode = glm-5.2 via the zai-coding-plan accounts (t2) -- a SEPARATE quota
#              pool that sits idle. Routing a lane (e.g. docs, where glm is proven)
#              to opencode relieves the opus weekly-quota throughput ceiling.
BACKENDS = ("claude", "opencode")
_BACKEND_PRODUCT = {"claude": "claude", "opencode": "opencode"}


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


def recently_attempted_issues(runs_dir: Path, *, cooldown_min: int,
                              now_ts: float | None = None) -> set[int]:
    """Issue numbers attempted within the last ``cooldown_min`` minutes — read from
    the mtime of their ``resolve-<N>-*.log``. This is the anti-churn gate: a hard
    issue (e.g. a mislabeled epic) that a worker could not land must NOT be re-picked
    every tick — re-dispatching it re-storms a known drain while the rest of the
    lane's backlog goes untouched. After the cooldown it becomes eligible again (the
    repo may have changed, or a fresh worker may get further). 0 disables the gate."""
    if cooldown_min <= 0 or not runs_dir.is_dir():
        return set()
    import time
    now = now_ts if now_ts is not None else time.time()
    horizon = now - cooldown_min * 60
    recent: set[int] = set()
    for log in runs_dir.glob("resolve-*.log"):
        m = _LOG_ISSUE_RE.search(log.name)
        if not m:
            continue
        try:
            if log.stat().st_mtime >= horizon:
                recent.add(int(m.group(1)))
        except OSError:
            continue
    return recent


def pick_target_issue(numbers: list[int], skip: set[int]) -> int | None:
    """The first lane issue not in ``skip`` (live workers ∪ recently-attempted)."""
    for n in numbers:
        if n not in skip:
            return n
    return None


def build_worker_command(backend: str, prompt: str, model: str | None) -> list[str]:
    """The argv for one detached issue-resolution worker, per backend. Both forms
    run the SAME issue-resolution prompt (with its ``#N``-in-subject rule), so the
    resulting commit is witnessable by the closure auditor regardless of backend."""
    if backend == "claude":
        return ["claude", "-p", "--permission-mode", "bypassPermissions", prompt]
    if backend == "opencode":
        cmd = ["opencode", "run", "--dangerously-skip-permissions"]
        if model:
            cmd += ["-m", model]  # pin the exact model so the run is reproducible/traced
        cmd.append(prompt)
        return cmd
    raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")


def _opencode_config_home(account_dir: str, runs_dir: Path) -> str:
    """opencode loads its config from ``$XDG_CONFIG_HOME/opencode``. The switcher's
    account dir is either the canonical ``<config_home>/opencode`` (the default
    account) or a sibling ``<config_home>/opencode-<tag>``. For the canonical dir the
    parent IS the XDG home; for a sibling we mint a tiny pin dir whose ``opencode``
    entry is a junction to the account, so opencode loads exactly that account -- we
    never silently mis-attribute a run to the wrong account."""
    p = Path(account_dir)
    if p.name == "opencode":
        return str(p.parent)
    pin = runs_dir / ".opencode-pins" / p.name
    link = pin / "opencode"
    pin.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        try:
            os.symlink(account_dir, link, target_is_directory=True)
        except OSError:
            # Windows without symlink privilege: a directory junction needs none.
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), account_dir],
                           capture_output=True, text=True)
    if not link.exists():
        raise RuntimeError(f"could not pin opencode account dir {account_dir!r}")
    return str(pin)


def opencode_worker_env(account_dir: str | None, lane: str, workspace: Path,
                        runs_dir: Path) -> dict[str, str]:
    """Child env for an opencode/glm worker: the account pinned via XDG_CONFIG_HOME
    (NOT CLAUDE_CONFIG_DIR / oauth token, which are claude-only), plus the same
    self-describing dispatch vars the claude path stamps."""
    env = dispatch_worker.child_env(lane, "opencode", workspace)
    env.pop("CLAUDE_CONFIG_DIR", None)
    env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    if account_dir:
        env["XDG_CONFIG_HOME"] = _opencode_config_home(account_dir, runs_dir)
    return env


def spawn_issue_worker(command: list[str], env: dict[str, str], cwd: Path,
                       log_dir: Path, issue: int, lane: str,
                       backend: str) -> dict[str, Any]:
    """Launch a detached worker (claude or opencode) on one issue; record pid.

    The log keeps the backend-neutral ``resolve-<N>-<stamp>.log`` name so the close
    arm, silent-worker scan, and in-flight de-dup all see it uniformly."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_log = log_dir / f"resolve-{issue}-{stamp}.log"
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
    return {"pid": proc.pid, "log": str(out_log), "issue": issue, "lane": lane,
            "backend": backend}


def evaluate(root: Path, *, max_workers: int, work_kind: str, lane: str | None,
             live: bool, refresh: bool = True, cooldown_min: int = 120,
             backend: str = "claude") -> dict[str, Any]:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    product = _BACKEND_PRODUCT[backend]
    reg = issue_dispatch.refresh_registry(root) if refresh else {"skipped": True}
    pre = issue_dispatch.preflight(root, max_workers=max_workers, work_kind=work_kind,
                                   product=product)
    pre_ok = pre.get("verdict") == "SPAWN_OK"
    acct = pre.get("account") or {}

    pick = lane_issue_numbers(root, lane)
    chosen_lane = pick.get("lane")
    runs_dir = root / RUNS_DIRNAME
    live_issues = live_resolution_issues(runs_dir)
    cooled = recently_attempted_issues(runs_dir, cooldown_min=cooldown_min)
    # Skip both a live worker's issue AND a recently-attempted one (anti-churn), so
    # the picker advances down the lane instead of re-storming one un-landable issue.
    skip = live_issues | cooled
    target = pick_target_issue(pick.get("numbers") or [], skip)

    payload: dict[str, Any] = {
        "schema": SCHEMA, "workspace": str(root), "live": live, "backend": backend,
        "max_workers": max_workers, "registry_refresh": reg,
        "preflight": {"verdict": pre.get("verdict"), "reason": pre.get("reason"),
                      "cap": pre.get("cap"), "live": pre.get("live")},
        "account": {k: acct.get(k) for k in ("tag", "tier", "model", "dir")},
        "lane": chosen_lane, "lane_issue_count": len(pick.get("numbers") or []),
        "cooled_recently": sorted(cooled), "target_issue": target,
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
                        "reason": (f"every open issue on lane '{chosen_lane}' is "
                                   f"either live ({sorted(live_issues)}) or in "
                                   f"cooldown ({sorted(cooled)}) — nothing fresh to "
                                   f"dispatch this tick")})
        return payload

    rec = issue_worker_prompt.build(target, chosen_lane, workspace=root)
    payload["prompt_chars"] = rec.get("prompt_chars")
    payload["issue_title"] = rec.get("title")
    model = acct.get("model") if backend == "opencode" else None
    preview_prompt = f"<resolve #{target} prompt, {rec.get('prompt_chars')} chars>"
    payload["command"] = build_worker_command(backend, preview_prompt, model)

    if not live:
        payload.update({"ok": True, "action": "would_spawn", "verdict": "WOULD_SPAWN",
                        "reason": (f"safe to spawn 1 {backend} issue-resolution worker on "
                                   f"#{target} (lane '{chosen_lane}') under account "
                                   f"'{acct.get('tag')}' (t{acct.get('tier')}, "
                                   f"{acct.get('model') or backend})")})
        return payload

    if backend == "claude":
        env = issue_dispatch.worker_env(acct.get("dir"), chosen_lane, root)
    else:
        env = opencode_worker_env(acct.get("dir"), chosen_lane, root, runs_dir)
    env["FLEET_RESOLVE_ISSUE"] = str(target)
    command = build_worker_command(backend, rec["prompt"], model)
    spawned = spawn_issue_worker(command, env, root, runs_dir, target, chosen_lane, backend)
    payload.update({"ok": True, "action": "spawned", "verdict": "SPAWNED",
                    "spawned": spawned,
                    "reason": (f"spawned {backend} issue-resolution worker pid "
                               f"{spawned['pid']} on #{target} (lane '{chosen_lane}') "
                               f"under '{acct.get('tag')}' ({acct.get('model') or backend})")})
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
        f"issue-resolve-dispatch: {p.get('verdict')} ({'ok' if p.get('ok') else 'refuse'})  "
        f"backend={p.get('backend')}  live={p.get('live')}",
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
    ap.add_argument("--backend", choices=BACKENDS, default="claude",
                    help="worker backend: claude (opus, t1) or opencode (glm-5.2, t2, "
                         "a separate quota pool). Default claude.")
    ap.add_argument("--live", action="store_true",
                    help="actually spawn the worker (default: dry-run / plan only)")
    ap.add_argument("--no-refresh", action="store_true",
                    help="skip the per-tick account-registry refresh")
    ap.add_argument("--cooldown-min", type=int, default=120,
                    help="skip an issue attempted within this many minutes (anti-churn; "
                         "0 disables). Default 120 — stops re-storming one un-landable issue.")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = evaluate(root, max_workers=args.max_workers, work_kind=args.work_kind,
                       lane=args.lane, live=args.live, refresh=not args.no_refresh,
                       cooldown_min=args.cooldown_min, backend=args.backend)
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
