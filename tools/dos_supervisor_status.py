#!/usr/bin/env python3
"""Read-only readiness card for fleet's generic DOS supervisor.

This is the fleet-native counterpart to the job repo's mature supervisor card:
it folds three existing surfaces without launching workers:

* ``dos doctor --json`` for the standing ``[supervise]`` policy.
* ``dos loop --json`` for the kernel's spawn/reap/flag plan.
* ``tools/plan_audit.py --json`` for the public plan surface.

The output answers one operator question: is fleet ready for the first real
generic-DOS canary, and if not, what exact precondition is missing?
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fleet-dos-supervisor-status/1"

OK_VERDICTS = {
    "READY_TO_CANARY",
    "AT_TARGET",
    "READY",
}
ACTION_VERDICTS = {
    "CONFIG_BLOCKED",
    "TARGET_UNREACHABLE",
    "PLAN_SURFACE_EMPTY",
    "PLAN_DRIFT",
    "INSPECT",
}


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def run_json(cmd: list[str], cwd: Path, *, ok_codes: set[int] | None = None) -> dict[str, Any]:
    ok_codes = ok_codes or {0}
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=45,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"_error": str(exc), "_cmd": cmd}
    payload = read_json_from_text(proc.stdout)
    if proc.returncode not in ok_codes:
        payload.setdefault("_error", (proc.stderr or proc.stdout).strip()[-2000:] or f"returncode {proc.returncode}")
    payload["_returncode"] = proc.returncode
    payload["_cmd"] = cmd
    return payload


def read_json_from_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text:
        try:
            obj = json.loads(text)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            return obj
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


def collect(workspace: Path) -> dict[str, Any]:
    root = workspace.resolve()
    plan_audit = root / "tools" / "plan_audit.py"
    if plan_audit.exists():
        audit = run_json(
            [sys.executable, str(plan_audit), "--glob", "{PLAN,BUILD}-*.md", "--json"],
            root,
        )
    else:
        audit = {"_error": f"plan audit helper not found: {plan_audit}"}
    return build_payload(
        workspace=str(root),
        doctor=run_json(["dos", "doctor", "--workspace", str(root), "--json"], root),
        loop=run_json(["dos", "loop", "--workspace", str(root), "--json"], root),
        plan_audit=audit,
    )


def build_payload(
    *,
    workspace: str,
    doctor: dict[str, Any],
    loop: dict[str, Any],
    plan_audit: dict[str, Any],
) -> dict[str, Any]:
    supervise = doctor.get("supervise") or {}
    paths = doctor.get("paths") or {}
    template = str(supervise.get("worker_launch_template") or "")
    template_has_lane = "{lane}" in template
    spawn = _spawn_lanes(loop)
    loop_verdict = str(loop.get("verdict") or "")
    target = _int(loop.get("target"), _int(supervise.get("target"), 1))
    alive = _int(loop.get("alive"), 0)
    admissible = _int(loop.get("admissible"), 0)
    plan_counts = plan_audit.get("counts") or {}
    task_weighted = (plan_audit.get("work_units") or {}).get("task_weighted") or {}
    total_plans = _int(plan_counts.get("total_plans"), 0)
    total_units = _int(task_weighted.get("total_units"), _sum_plan_units(plan_audit))
    done_units = _int(task_weighted.get("done_units"), 0)
    drift = plan_audit.get("drift") or []

    if doctor.get("_error") or loop.get("_error") or plan_audit.get("_error"):
        verdict = "INSPECT"
        why = "one or more status inputs failed"
        next_action = "inspect the _errors in the JSON payload and rerun the failing command"
    elif not template:
        verdict = "CONFIG_BLOCKED"
        why = "dos.toml [supervise].worker_launch_template is empty"
        next_action = "set [supervise].worker_launch_template with a {lane} placeholder"
    elif not template_has_lane:
        verdict = "CONFIG_BLOCKED"
        why = "worker_launch_template does not contain {lane}"
        next_action = "add the {lane} placeholder so each worker receives its assigned lane"
    elif loop_verdict == "TARGET_UNREACHABLE":
        verdict = "TARGET_UNREACHABLE"
        why = str(loop.get("reason") or "the lane roster cannot satisfy the configured target")
        next_action = "lower [supervise].target or declare additional disjoint lanes"
    elif total_plans <= 0 or total_units <= 0:
        verdict = "PLAN_SURFACE_EMPTY"
        why = "the public plan audit surface has no counted work units"
        next_action = "add or repair a PLAN-*.md surface before launching always-on workers"
    elif drift:
        verdict = "PLAN_DRIFT"
        why = f"plan audit reports {len(drift)} drift item(s)"
        next_action = "resolve the plan audit drift before trusting always-on dispatch"
    elif loop_verdict == "AT_TARGET" or alive >= target:
        verdict = "AT_TARGET"
        why = f"generic DOS supervisor already counts {alive}/{target} live worker(s)"
        next_action = "inspect worker liveness before changing the supervisor population"
    elif spawn:
        canary_target = min(target, alive + 1)
        verdict = "READY_TO_CANARY"
        why = f"kernel would spawn {len(spawn)} worker(s); alive={alive}/{target}"
        next_action = (
            f"operator-gated canary: dos loop --workspace {workspace} "
            f"--enact --target {canary_target} --max-ticks 1"
        )
    else:
        verdict = "READY"
        why = f"dos loop verdict is {loop_verdict or 'unknown'} with no blocking precondition"
        next_action = "rerun dos loop --json, then launch a bounded canary when a spawn appears"

    ok = verdict in OK_VERDICTS
    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "why": why,
        "next_action": next_action,
        "workspace": workspace,
        "supervise": {
            "target": target,
            "alive": alive,
            "admissible": admissible,
            "loop_verdict": loop_verdict or None,
            "spawn": spawn,
            "reap": _lane_list(loop.get("reap") or []),
            "flag": _lane_list(loop.get("flag") or []),
            "worker_launch_template_configured": bool(template),
            "worker_launch_template_has_lane": template_has_lane,
        },
        "plans": {
            "plans_glob": paths.get("plans_glob"),
            "total_plans": total_plans,
            "total_units": total_units,
            "done_units": done_units,
            "drift_count": len(drift),
        },
        "inputs": {
            "doctor_error": doctor.get("_error"),
            "loop_error": loop.get("_error"),
            "plan_audit_error": plan_audit.get("_error"),
        },
    }


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sum_plan_units(plan_audit: dict[str, Any]) -> int:
    return sum(_int(p.get("total_units"), 0) for p in plan_audit.get("plans") or [] if isinstance(p, dict))


def _lane_list(rows: list[Any]) -> list[str]:
    lanes: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            lane = row.get("lane")
        else:
            lane = row
        if lane is not None:
            lanes.append(str(lane))
    return lanes


def _spawn_lanes(loop: dict[str, Any]) -> list[str]:
    return _lane_list(loop.get("spawn") or [])


def render(payload: dict[str, Any]) -> str:
    sup = payload.get("supervise") or {}
    plans = payload.get("plans") or {}
    spawn = ", ".join(sup.get("spawn") or []) or "-"
    return "\n".join(
        [
            f"DOS supervisor: {payload.get('verdict')} ({'ok' if payload.get('ok') else 'action'})",
            f"why: {payload.get('why')}",
            f"next: {payload.get('next_action')}",
            f"workers: alive={sup.get('alive')} target={sup.get('target')} admissible={sup.get('admissible')} spawn={spawn}",
            f"plans: total={plans.get('total_plans')} units={plans.get('total_units')} drift={plans.get('drift_count')}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only fleet DOS supervisor readiness card.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
