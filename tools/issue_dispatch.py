#!/usr/bin/env python3
r"""One guarded, switcher-routed, bounded dispatch TICK — the keystone that turns
the existing pieces into a safe always-on issue dispatcher.

The historical spawn path (``dos loop --enact`` -> ``dispatch_worker.py --lane
{lane}`` -> ``claude -p /dos-kernel:dos-dispatch-loop``) had two holes this tick
closes, in order, before a single worker is launched:

  1. PREFLIGHT (DoS safety)   tools/dispatch_preflight.py must return SPAWN_OK:
                              host guard clean ∧ an account is free ∧ live workers
                              < cap. If not, this tick REFUSES and exits non-zero
                              — that refusal IS the no-DoS guarantee (the live
                              population can never exceed the cap, so per-session
                              hook pressure stays bounded).
  2. SWITCHER PIN (routing)   the worker is launched with CLAUDE_CONFIG_DIR (and
                              the account's long-lived .oauth-token) pinned to the
                              switcher's chosen account — never the ambient
                              default that historically ate the dispatch when it
                              was throttled/auth-blocked.

It then picks the lane with the most open issues (the issue_lane_router fold), or
an explicit ``--lane``, and launches ONE detached worker on it. DRY-RUN BY
DEFAULT: it prints exactly what it would do (account, lane, command, witness).
``--live`` spawns. The witness the worker should use to keep/revert its own work
is the BENCHMARK (tools/bench_witness.py --lane <lane>), not the test suite — the
operator's rule for this loop — and is named in the tick record for the worker.

    python tools/issue_dispatch.py                 # plan one safe tick (dry-run)
    python tools/issue_dispatch.py --max-workers 2 --live   # spawn one worker
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
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
import dispatch_worker  # noqa: E402  (sibling tool: build_command/child_env)
import fleet_accounts  # noqa: E402  (the switcher: read_oauth_token)

SCHEMA = "fleet-issue-dispatch/1"
RUNS_DIRNAME = ".dispatch-runs"


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
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"_error": str(exc), "_cmd": cmd}
    out = (proc.stdout or "").strip()
    try:
        doc = json.loads(out) if out else {}
    except ValueError:
        doc = {}
        for line in reversed(out.splitlines()):
            try:
                doc = json.loads(line.strip())
                break
            except ValueError:
                continue
    if not isinstance(doc, dict):
        doc = {}
    doc.setdefault("_returncode", proc.returncode)
    return doc


def refresh_registry(root: Path) -> dict[str, Any]:
    """Re-derive the account registry from live sessions BEFORE routing.

    The switcher (``fleet_accounts route``, called inside the preflight) reads the
    cached ``tools/_registry/sessions.json`` snapshot. On an always-on tick that
    snapshot goes stale between launches, so an account that just hit a weekly
    limit — or whose org disabled Claude-Code subscription access — would still be
    handed out, the worker would spawn and instantly die, and the loop would make
    no progress (the exact failure that left ``.dispatch-runs/`` empty: the tick
    routed to a dead default account every time). Regenerating the registry
    each tick folds the current session evidence (throttle/auth/org-disabled) into
    the roster the switcher reads, so a freshly-blocked account is skipped
    automatically. A no-probe scan — cheap, read-only, no model call. Best-effort:
    a refresh failure is recorded, never fatal (the tick proceeds on the prior
    snapshot rather than refusing)."""
    doc = run_json([_py(), str(root / "tools" / "fleet_sessions.py"), "registry"],
                   root, timeout=120, ok_codes=set(range(0, 16)))
    return {"ok": "_error" not in doc, "error": doc.get("_error")}


def preflight(root: Path, *, max_workers: int, work_kind: str,
              product: str = "claude") -> dict[str, Any]:
    return run_json([_py(), str(root / "tools" / "dispatch_preflight.py"), "--json",
                     "--max-workers", str(max_workers), "--work-kind", work_kind,
                     "--product", product],
                    root, timeout=120)


def pick_lane(root: Path, explicit: str | None) -> dict[str, Any]:
    """The lane with the most open issues, or an explicit override."""
    router = run_json([_py(), str(root / "tools" / "issue_lane_router.py"), "--json"],
                      root, timeout=130)
    lanes = router.get("lanes") or {}
    counts = {}
    for ln, info in lanes.items():
        iss = info.get("issues") if isinstance(info, dict) else info
        counts[ln] = len(iss) if hasattr(iss, "__len__") else 0
    if explicit:
        return {"lane": explicit, "issues": counts.get(explicit, 0), "by_lane": counts,
                "explicit": True, "router_error": router.get("_error")}
    if not counts:
        return {"lane": None, "issues": 0, "by_lane": {}, "router_error": router.get("_error")}
    lane = max(counts, key=lambda k: counts[k])
    return {"lane": lane, "issues": counts[lane], "by_lane": counts,
            "router_error": router.get("_error")}


def worker_env(account_dir: str | None, lane: str, workspace: Path) -> dict[str, str]:
    """Child env: the switcher account pinned (CLAUDE_CONFIG_DIR + its oauth token),
    plus the self-describing dispatch vars and the benchmark-witness hint."""
    env = dispatch_worker.child_env(lane, "claude", workspace)
    if account_dir:
        env["CLAUDE_CONFIG_DIR"] = account_dir
        # The switcher's single credential rule: prefer the dir's long-lived
        # .oauth-token over the expiring interactive creds; drop any ambient token when
        # this account has none (never bleed a sibling account's token into the worker).
        tok = fleet_accounts.read_oauth_token(account_dir)
        if tok:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = tok
        else:
            env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    # The witness for this loop is the benchmark, not the unit-test suite.
    env["FLEET_DISPATCH_WITNESS"] = "benchmark"
    env["FLEET_BENCH_WITNESS_CMD"] = f"python tools/bench_witness.py --lane {lane}"
    # Arm the DOS verdict-journal auto-emit (#465) on the *dispatch* surface, NOT the
    # session surface. The kernel's verdict-journal is append-only and NOT auto-rotated
    # (dos verdict_journal.py: "grows unbounded on a long-lived fleet"; the
    # [retention] audits_keep_last cap does not fold over it). So arming it via
    # settings.json `env` — which fires on every idle Claude Code session — would
    # violate this issue's own "retention bounded" acceptance. Arming it here instead
    # bounds growth to actual dispatched issue-resolution runs: a worker's verify/recall
    # adjudications land in .dos/verdict-journal.jsonl while it works, an idle session
    # writes nothing, and the journal rides the existing .dos/ backup story.
    env["DISPATCH_OBSERVE"] = "1"
    return env


def spawn_detached(command: list[str], env: dict[str, str], cwd: Path,
                   log_dir: Path, lane: str) -> dict[str, Any]:
    """Launch the worker DETACHED so it outlives this tick; log to a dated file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_log = log_dir / f"dispatch-{lane}-{stamp}.log"
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
    return {"pid": proc.pid, "log": str(out_log)}


def evaluate(root: Path, *, max_workers: int, work_kind: str, lane: str | None,
             live: bool, refresh: bool = True) -> dict[str, Any]:
    # Refresh the account registry from live sessions FIRST, so the switcher routes
    # off current evidence (a freshly weekly-limited / org-disabled account is
    # skipped, not handed out to a worker that would instantly die). Skippable for
    # tests/inspection via refresh=False.
    reg = refresh_registry(root) if refresh else {"ok": None, "skipped": True}
    pre = preflight(root, max_workers=max_workers, work_kind=work_kind)
    pre_ok = pre.get("verdict") == "SPAWN_OK"
    acct = pre.get("account") or {}
    lane_pick = pick_lane(root, lane)
    chosen = lane_pick.get("lane")
    command = dispatch_worker.build_command(chosen, "claude") if chosen else []

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "workspace": str(root),
        "live": live,
        "max_workers": max_workers,
        "registry_refresh": reg,
        "preflight": {"verdict": pre.get("verdict"), "reason": pre.get("reason"),
                      "cap": pre.get("cap"), "live": pre.get("live")},
        "account": {k: acct.get(k) for k in ("tag", "tier", "model", "dir")},
        "lane": chosen,
        "lane_issue_count": lane_pick.get("issues"),
        "command": command,
        "witness": {"kind": "benchmark",
                    "cmd": f"python tools/bench_witness.py --lane {chosen}" if chosen else None},
    }

    if not pre_ok:
        payload.update({"ok": False, "action": "refused",
                        "verdict": pre.get("verdict") or "REFUSE",
                        "reason": f"preflight refused: {pre.get('reason')}"})
        return payload
    if not chosen:
        payload.update({"ok": False, "action": "no_lane", "verdict": "NO_LANE",
                        "reason": "no lane has open issues (router empty/error)"})
        return payload
    if not live:
        payload.update({"ok": True, "action": "would_spawn", "verdict": "WOULD_SPAWN",
                        "reason": (f"safe to spawn 1 worker on lane '{chosen}' "
                                   f"({lane_pick.get('issues')} issues) under account "
                                   f"'{acct.get('tag')}' (t{acct.get('tier')})")})
        return payload

    env = worker_env(acct.get("dir"), chosen, root)
    spawned = spawn_detached(command, env, root, root / RUNS_DIRNAME, chosen)
    payload.update({"ok": True, "action": "spawned", "verdict": "SPAWNED",
                    "spawned": spawned,
                    "reason": (f"spawned worker pid {spawned['pid']} on lane '{chosen}' "
                               f"under account '{acct.get('tag')}'")})
    _record(root / RUNS_DIRNAME, payload)
    return payload


def _record(runs_dir: Path, payload: dict[str, Any]) -> None:
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "last-tick.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def render(p: dict[str, Any]) -> str:
    a = p.get("account") or {}
    pf = p.get("preflight") or {}
    lines = [
        f"issue-dispatch: {p.get('verdict')} ({'ok' if p.get('ok') else 'refuse'})  live={p.get('live')}",
        f"  preflight : {pf.get('verdict')} ({pf.get('live')}/{pf.get('cap')} live)",
        f"  account   : {a.get('tag') or '-'} (t{a.get('tier')})  {a.get('model') or ''}",
        f"  lane      : {p.get('lane') or '-'}  ({p.get('lane_issue_count')} issues)",
        f"  witness   : {(p.get('witness') or {}).get('cmd') or '-'}",
        f"  command   : {' '.join(p.get('command') or []) or '-'}",
        f"  -> {p.get('reason')}",
    ]
    if p.get("spawned"):
        lines.append(f"  spawned pid={p['spawned'].get('pid')} log={p['spawned'].get('log')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="One guarded, switcher-routed, bounded dispatch tick (dry-run by default).")
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
                    help="skip the per-tick account-registry refresh (route off the "
                         "cached snapshot; for inspection / when a fresh scan just ran)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = evaluate(root, max_workers=args.max_workers, work_kind=args.work_kind,
                       lane=args.lane, live=args.live, refresh=not args.no_refresh)
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
