#!/usr/bin/env python3
"""Read-only productivity fold for the generic DOS supervisor canary.

The readiness card answers "may we launch?". The watchdog answers "what exact
bounded launch would happen?". This audit answers the next operator question:
after a canary, did the supervisor produce useful evidence, drain honestly, or
surface a typed blocker?

It never launches workers. It folds only read-back surfaces that the current
process did not author: ``dos loop`` via ``dos_supervisor_status`` and the
dry-run watchdog plan.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import dos_supervisor_status as status
import dos_supervisor_watchdog as watchdog

SCHEMA = "fleet-dos-supervisor-canary-audit/1"

OK_VERDICTS = {
    "PRE_CANARY_READY",
    "CANARY_OBSERVED",
    "PRODUCTIVE",
    "HONEST_DRAIN",
}


def collect(workspace: Path) -> dict[str, Any]:
    root = workspace.resolve()
    readiness = status.collect(root)
    dry_run = watchdog.run_watchdog(
        workspace=root,
        target=None,
        max_ticks=1,
        live=False,
        timeout_s=120,
        readiness=readiness,
    )
    return build_payload(workspace=root, readiness=readiness, dry_run=dry_run)


def build_payload(
    *,
    workspace: Path,
    readiness: dict[str, Any],
    dry_run: dict[str, Any],
) -> dict[str, Any]:
    supervise = readiness.get("supervise") or {}
    plans = readiness.get("plans") or {}
    spawn = [str(lane) for lane in supervise.get("spawn") or []]
    reap = [str(lane) for lane in supervise.get("reap") or []]
    flag = [str(lane) for lane in supervise.get("flag") or []]
    alive = _int(supervise.get("alive"), 0)
    target = _int(supervise.get("target"), 0)
    done_units = _int(plans.get("done_units"), 0)
    drift_count = _int(plans.get("drift_count"), 0)
    blockers = _blockers(readiness=readiness, dry_run=dry_run, reap=reap, flag=flag, drift_count=drift_count)

    if blockers:
        verdict = "BLOCKED"
        ok = False
        finding = "typed_blocker"
        reason = blockers[0]["detail"]
        next_action = blockers[0]["next_action"]
    elif done_units > 0:
        verdict = "PRODUCTIVE"
        ok = True
        finding = "landed_work"
        reason = f"{done_units} plan unit(s) report completed"
        next_action = "run the same audit after the next supervisor tick to confirm continued useful output"
    elif str(readiness.get("verdict") or "") == "READY_TO_CANARY" and dry_run.get("action") == "would_enact":
        verdict = "PRE_CANARY_READY"
        ok = True
        finding = "operator_gate"
        reason = "readiness and watchdog dry-run are green; live canary remains operator-gated"
        next_action = "operator-gated live canary: python tools/dos_supervisor_watchdog.py --live --json"
    elif alive > 0:
        verdict = "CANARY_OBSERVED"
        ok = True
        finding = "live_worker_observed"
        reason = f"{alive} live worker(s) counted by the supervisor"
        next_action = "inspect worker liveness and rerun this audit after the worker stops or ships"
    else:
        verdict = "HONEST_DRAIN"
        ok = True
        finding = "honest_drain"
        reason = "no live worker, spawn, or blocker evidence is currently visible"
        next_action = "no launchable canary work is visible; re-run dos loop --json or add plan work"

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": str(workspace),
        "workers": {
            "alive": alive,
            "target": target,
            "spawn": spawn,
            "reap": reap,
            "flag": flag,
        },
        "plans": {
            "total_plans": _int(plans.get("total_plans"), 0),
            "total_units": _int(plans.get("total_units"), 0),
            "done_units": done_units,
            "drift_count": drift_count,
        },
        "blockers": blockers,
        "evidence": {
            "readiness": {
                "schema": readiness.get("schema"),
                "ok": readiness.get("ok"),
                "verdict": readiness.get("verdict"),
                "why": readiness.get("why"),
                "next_action": readiness.get("next_action"),
            },
            "watchdog": {
                "schema": dry_run.get("schema"),
                "ok": dry_run.get("ok"),
                "action": dry_run.get("action"),
                "reason": dry_run.get("reason"),
                "safety": dry_run.get("safety") or {},
                "live": dry_run.get("live"),
                "target": dry_run.get("target"),
                "max_ticks": dry_run.get("max_ticks"),
                "command": dry_run.get("command") or [],
            },
        },
    }


def _blockers(
    *,
    readiness: dict[str, Any],
    dry_run: dict[str, Any],
    reap: list[str],
    flag: list[str],
    drift_count: int,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if not readiness.get("ok"):
        blockers.append({
            "kind": "readiness",
            "detail": str(readiness.get("why") or readiness.get("verdict") or "readiness failed"),
            "next_action": str(readiness.get("next_action") or "inspect dos supervisor readiness"),
        })
    if not dry_run.get("ok"):
        blockers.append({
            "kind": "watchdog",
            "detail": str(dry_run.get("reason") or dry_run.get("action") or "watchdog dry-run failed"),
            "next_action": "inspect python tools/dos_supervisor_watchdog.py --json",
        })
    safety = dry_run.get("safety") or {}
    if safety and not safety.get("ok"):
        for blocker in safety.get("blockers") or []:
            blockers.append({
                "kind": f"workspace_safety:{blocker.get('kind') or 'unknown'}",
                "detail": str(blocker.get("detail") or "workspace safety preflight failed"),
                "next_action": str(blocker.get("next_action") or "resolve workspace safety blockers before --live"),
            })
    if drift_count > 0:
        blockers.append({
            "kind": "plan_drift",
            "detail": f"plan audit reports {drift_count} drift item(s)",
            "next_action": "resolve plan audit drift before trusting canary productivity",
        })
    if reap:
        blockers.append({
            "kind": "reap",
            "detail": f"kernel asks supervisor to reap stalled lane(s): {', '.join(reap)}",
            "next_action": "let the supervisor scavenge stalled lanes before launching more work",
        })
    if flag:
        blockers.append({
            "kind": "flag",
            "detail": f"kernel flags lane(s) for operator attention: {', '.join(flag)}",
            "next_action": "inspect flagged lanes; the supervisor should not kill SPINNING workers automatically",
        })
    return blockers


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def render(payload: dict[str, Any]) -> str:
    workers = payload.get("workers") or {}
    plans = payload.get("plans") or {}
    return "\n".join(
        [
            f"DOS canary audit: {payload.get('verdict')} ({payload.get('finding')})",
            f"next: {payload.get('next_action')}",
            (
                f"workers: alive={workers.get('alive')} target={workers.get('target')} "
                f"spawn={','.join(workers.get('spawn') or []) or '-'}"
            ),
            (
                f"plans: done={plans.get('done_units')}/{plans.get('total_units')} "
                f"drift={plans.get('drift_count')}"
            ),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only audit of DOS supervisor canary productivity.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else status.repo_root()
    payload = collect(workspace)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
