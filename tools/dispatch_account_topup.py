#!/usr/bin/env python3
"""Top up a SPECIFIC account's live dispatch workers to a target.

The switcher-based ``FleetIssueDispatch`` (``issue_resolve_dispatch.py``)
load-balances each spawn onto the LEAST-busy available account, so a busy-but-
healthy account — e.g. ``day26-netra``, which also runs the orchestrator session —
is effectively never chosen for dispatch even though it has quota. To use BOTH
accounts in parallel toward a throughput goal, this pins ``--account`` and spawns
only the DEFICIT (``--target`` minus the account's currently-live workers) on
distinct fresh issues. Run it on a schedule per account to keep two (or more)
accounts dispatching at once.

  python tools/dispatch_account_topup.py --account day26-netra --target 2

Self-bounded and idempotent: deficit-only (never exceeds ``--target``), dedups
against live + recently-attempted issues, and resolves the account through the
same switcher front door so a freshly-throttled account is skipped. Reuses the
``issue_resolve_dispatch`` spawn machinery (sidecars, early-exit probe, loop
ledger) rather than re-implementing it. Dry-run prints the plan; ``--live`` spawns.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import issue_dispatch                 # noqa: E402  worker_env (CLAUDE_CONFIG_DIR pin)
import issue_resolve_dispatch as ird  # noqa: E402  lane pick, spawn, helpers
import issue_worker_prompt            # noqa: E402

RUNS = REPO / ".dispatch-runs"


def _alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True,
                         creationflags=ird.no_window_creationflags())
    return str(pid) in (out.stdout or "")


def live_by_account(tag: str) -> int:
    """Count live dispatch workers attributed to ``tag`` via the ``.account``
    sidecar + a live pid (a 0-byte log with a live pid is a RUNNING worker, not a
    death — the same silent-worker discipline dispatch_status uses)."""
    n = 0
    for acc in RUNS.glob("resolve-*.account"):
        try:
            rec = json.loads(acc.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (rec.get("tag") or "") != tag:
            continue
        try:
            pid = int(acc.with_suffix(".pid").read_text(encoding="utf-8").split()[0])
        except (OSError, ValueError, IndexError):
            continue
        if _alive(pid):
            n += 1
    return n


def plan(account: str, target: int, lane_arg: str | None) -> dict:
    """Pure-ish planning fold: how many to spawn for ``account`` and on which fresh
    issues. No spawn — the unit a test can exercise (it stubs the helpers)."""
    have = live_by_account(account)
    deficit = max(0, target - have)
    pick = ird.lane_issue_numbers(REPO, lane_arg)
    lane = pick.get("lane")
    nums = [int(n) for n in (pick.get("numbers") or [])]
    skip = {int(x) for x in ird.live_resolution_issues(RUNS)} | \
           {int(x) for x in ird.recently_attempted_issues(RUNS, cooldown_min=120)}
    fresh = [n for n in nums if n not in skip]
    return {"account": account, "live": have, "target": target, "deficit": deficit,
            "lane": lane, "targets": fresh[:deficit]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--account", required=True)
    ap.add_argument("--target", type=int, default=2)
    ap.add_argument("--lane", default=None)
    ap.add_argument("--live", action="store_true", help="actually spawn (default: plan only)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    p = plan(args.account, args.target, args.lane)
    if p["deficit"] == 0 or not p["targets"]:
        p["spawned"] = 0
        p["reason"] = "at target" if p["deficit"] == 0 else "no fresh issue to dispatch"
        print(json.dumps(p) if args.json else f"topup {args.account}: {p['reason']} "
              f"(live={p['live']}/{p['target']})")
        return 0

    if not args.live:
        print(json.dumps(p) if args.json else
              f"topup {args.account}: WOULD spawn {len(p['targets'])} on lane "
              f"{p['lane']} -> {p['targets']} (--live to spawn)")
        return 0

    rec = json.loads(subprocess.run(
        [sys.executable, str(REPO / "tools" / "fleet_accounts.py"), "resolve",
         "--account", args.account], capture_output=True, text=True,
        creationflags=ird.no_window_creationflags()).stdout or "{}")
    if not rec.get("ok"):
        print(json.dumps({"account": args.account, "spawned": 0,
                          "reason": f"unavailable: {rec.get('block_reason') or rec.get('reason')}"}))
        return 0
    acct_dir = rec.get("config_dir")
    acct = {"tag": args.account, "dir": acct_dir, "model": rec.get("model"),
            "tier": rec.get("selected_tier") or rec.get("model_tier")}

    spawned = []
    for issue in p["targets"]:
        rb = issue_worker_prompt.build(issue, p["lane"], workspace=REPO)
        env = issue_dispatch.worker_env(acct_dir, p["lane"], REPO)
        env["FLEET_RESOLVE_ISSUE"] = str(issue)
        cmd = ird.build_worker_command("claude", rb["prompt"], None)
        res = ird.spawn_issue_worker(cmd, env, REPO, RUNS, issue, p["lane"], "claude",
                                     account=acct, spawn_probe_s=5.0)
        spawned.append({"issue": issue, "pid": res.get("pid")})

    out = {"account": args.account, "live_before": p["live"], "target": args.target,
           "lane": p["lane"], "spawned": len(spawned), "issues": spawned}
    print(json.dumps(out) if args.json else
          f"topup {args.account}: spawned {len(spawned)} on lane {p['lane']} -> "
          f"{[s['issue'] for s in spawned]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
