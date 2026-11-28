#!/usr/bin/env python3
"""Snapshot summarization, text formatting, and CLI printer helpers for fleet_control_pane."""
from __future__ import annotations

import collections
import json
import os
from pathlib import Path
from typing import Any

from fleet_control_pane_core import (
    BOOTSTRAP_SCHEMA,
    COMMIT_SCHEMA,
    FLEET_SCHEMA,
    LOOP_AUDIT_SCHEMA,
    LOOP_CHECK_SCHEMA,
    LOOP_CONFIG_SCHEMA,
    LOOP_LIST_SCHEMA,
    LOOP_SCAFFOLD_SCHEMA,
    LOOP_SET_SCHEMA,
    PUBLISH_SCHEMA,
    RECOVER_SCHEMA,
    SCHEDULED_TASK_ACTION_STATES,
    SETUP_SCHEMA,
    SUPERVISOR_ACTION_VERDICTS,
    SUPERVISOR_NON_ACTION_VERDICTS,
    SUPERVISOR_SCHEMA,
    SUPERVISOR_WATCHDOG_ACTION_RESULT,
    SYNC_SCHEMA,
    TICK_SCHEMA,
    WINDOWS_TASK_REQUEST_REFUSED_RESULT,
    WINDOWS_TASK_RESULT_LABELS,
    WINDOWS_TASK_RUNNING_RESULT,
    display_command,
    display_path,
    machine_id,
    pane_attr,
    parse_iso_age_min,
    read_json,
    resolve_path,
    write_json,
    write_text_atomic,
)
from fleet_control_pane_loops import loop_check_needs_action, loop_check_snapshot


def summarize_registry(registry: dict[str, Any] | None) -> dict[str, Any]:
    if not registry:
        return {"exists": False, "sessions": 0, "actions": {}, "categories": {}}
    sessions = registry.get("sessions") or []
    accounts = registry.get("accounts") or []
    actions = collections.Counter(str(s.get("action", "")) for s in sessions)
    categories = collections.Counter(str(s.get("category", "")) for s in sessions)
    dispositions = collections.Counter(str(s.get("disp", "")) for s in sessions)
    available = [a for a in accounts if a.get("available")]
    blocked = [a for a in accounts if not a.get("available")]
    blocked_accounts = [blocked_account_snapshot(a) for a in blocked]
    blocked_kinds = collections.Counter(str(a.get("block_kind") or "unknown") for a in blocked_accounts)
    auth_blocked_sessions = dispositions.get("INFRA_AUTH", 0)
    return {
        "exists": True,
        "schema": registry.get("schema"),
        "generated_utc": registry.get("generated_utc"),
        "age_min": pane_attr("parse_iso_age_min", parse_iso_age_min)(registry.get("generated_utc")),
        "sessions": len(sessions),
        "categories": dict(categories),
        "actions": dict(actions),
        "dispositions": dict(dispositions),
        "auto_resume": actions.get("AUTO_RESUME", 0),
        "surface": actions.get("SURFACE", 0),
        "auth_blocked": auth_blocked_sessions,
        "auth_blocked_actions": actions.get("BLOCKED_AUTH", 0),
        "supervised": actions.get("SUPERVISED", 0),
        "accounts": {
            "total": len(accounts),
            "available": len(available),
            "blocked_count": len(blocked),
            "blocked_kinds": dict(blocked_kinds),
            "auth_blocked_count": sum(1 for a in blocked_accounts if account_block_is_auth_like(a)),
            "usage_blocked_count": sum(1 for a in blocked_accounts if account_block_is_usage_like(a)),
            "available_tags": [str(a.get("tag") or a.get("account")) for a in available],
            "blocked": blocked_accounts,
        },
    }


def account_block_is_auth_like(account: dict[str, Any]) -> bool:
    kind = str(account.get("block_kind") or "").lower()
    reason = str(account.get("reason") or "").lower()
    return kind in {"auth", "credit"} or "auth" in reason or "login" in reason or "credit" in reason


def account_block_needs_login(account: dict[str, Any]) -> bool:
    kind = str(account.get("block_kind") or "").lower()
    reason = str(account.get("reason") or "").lower()
    if kind in {"access", "usage"} or "subscription access" in reason or "access disabled" in reason:
        return False
    return kind == "auth" or "auth/login" in reason or "login required" in reason


def account_block_is_usage_like(account: dict[str, Any]) -> bool:
    kind = str(account.get("block_kind") or "").lower()
    reason = str(account.get("reason") or "").lower()
    return bool(account.get("throttled")) or kind == "usage" or "usage limit" in reason


def account_block_is_access_wall(account: dict[str, Any]) -> bool:
    kind = str(account.get("block_kind") or "").lower()
    reason = str(account.get("reason") or "").lower()
    return kind == "access" or "subscription access" in reason or "access disabled" in reason


def account_profile_command(config_dir: str) -> str:
    if not config_dir:
        return ""
    if os.name == "nt":
        escaped = config_dir.replace("'", "''")
        return f"$env:CLAUDE_CONFIG_DIR='{escaped}'; claude"
    import shlex

    return f"CLAUDE_CONFIG_DIR={shlex.quote(config_dir)} claude"


def blocked_account_snapshot(account: dict[str, Any]) -> dict[str, Any]:
    config_dir = str(account.get("config_dir") or "")
    command = account_profile_command(config_dir)
    out = {
        "account": str(account.get("account") or ""),
        "tag": str(account.get("tag") or account.get("account") or "unknown"),
        "config_dir": config_dir,
        "reason": str(account.get("block_reason") or account.get("reason") or ""),
        "block_kind": account.get("block_kind"),
        "reset": account.get("reset"),
        "throttled": bool(account.get("throttled")),
        "active_sessions": int(account.get("active_sessions") or 0),
        "live_sessions": int(account.get("live_sessions") or 0),
        "auth_blocked_sessions": int(account.get("auth_blocked_sessions") or 0),
        "status_source": account.get("status_source"),
        "registry_age_min": account.get("registry_age_min"),
    }
    if command:
        if not account_block_needs_login(out):
            command = ""
    if command:
        out["command"] = command
    return out


def summarize_resume_plan(config: dict[str, Any], *, limit: int = 10) -> dict[str, Any]:
    root = Path(config["root"])
    registry_dir = resolve_path(config.get("registry_dir"), root) or root / "tools" / "_registry"
    path = registry_dir / "resume_plan.json"
    summary: dict[str, Any] = {
        "path": str(path),
        "display_path": display_path(path, root),
        "exists": path.exists(),
        "count": 0,
        "shown": 0,
        "limit": limit,
        "sessions": [],
        "omitted": 0,
    }
    if not path.exists():
        return summary

    doc = read_json(path)
    summary["generated_utc"] = doc.get("generated_utc")
    plan = doc.get("plan")
    if not isinstance(plan, list):
        summary["reason"] = "missing plan list"
        return summary

    sessions: list[dict[str, Any]] = []
    for item in plan[:limit]:
        if not isinstance(item, dict):
            continue
        session = str(item.get("session") or "")
        sessions.append({
            "account": str(item.get("account") or ""),
            "session": session,
            "session_short": session[:8],
            "project": str(item.get("project") or ""),
            "cwd": str(item.get("cwd") or ""),
            "config_dir": str(item.get("config_dir") or ""),
            "resume_cmd": str(item.get("resume_cmd") or ""),
        })

    summary["count"] = len(plan)
    summary["shown"] = len(sessions)
    summary["sessions"] = sessions
    summary["omitted"] = max(0, len(plan) - len(sessions))
    return summary


def scheduler_result_needs_action(task: dict[str, Any]) -> bool:
    result = task.get("last_result")
    if result in (None, 0):
        return False
    try:
        result_int = scheduler_result_int(result)
    except (TypeError, ValueError):
        return True
    state = str(task.get("state") or "").lower()
    if result_int in {WINDOWS_TASK_RUNNING_RESULT, WINDOWS_TASK_REQUEST_REFUSED_RESULT} and state == "running":
        return False
    return True


def scheduler_state_needs_action(task: dict[str, Any]) -> bool:
    if task.get("supported") is not True or task.get("installed") is not True:
        return False
    return str(task.get("state") or "").strip().lower() in SCHEDULED_TASK_ACTION_STATES


def scheduled_task_needs_action(task: dict[str, Any]) -> bool:
    return scheduler_state_needs_action(task) or scheduler_result_needs_action(task)


def scheduler_result_int(result: Any) -> int:
    value = int(result)
    if value < 0:
        value += 2**32
    return value


def scheduler_result_text(task: dict[str, Any]) -> str:
    result = task.get("last_result")
    if result is None:
        return "unknown"
    try:
        result_int = scheduler_result_int(result)
    except (TypeError, ValueError):
        return str(result)
    hex_code = f"0x{result_int & 0xFFFFFFFF:08X}"
    label = WINDOWS_TASK_RESULT_LABELS.get(result_int)
    if label:
        return f"{result_int} ({hex_code}; {label})"
    return f"{result_int} ({hex_code})"


def scheduled_task_snapshot(task: dict[str, Any], *, accepted_results: set[int] | None = None) -> dict[str, Any]:
    accepted_results = accepted_results or set()
    result_accepted = False
    if task.get("last_result") is not None:
        try:
            result_accepted = scheduler_result_int(task.get("last_result")) in accepted_results
        except (TypeError, ValueError):
            result_accepted = False
    state_action = scheduler_state_needs_action(task)
    result_action = scheduler_result_needs_action(task) and not result_accepted
    out = {
        "supported": task.get("supported"),
        "installed": task.get("installed"),
        "state": task.get("state"),
        "last_result": task.get("last_result"),
        "reason": task.get("reason"),
        "task_name": task.get("task_name"),
        "needs_action": state_action or result_action,
    }
    if task.get("last_result") is not None:
        out["last_result_text"] = scheduler_result_text(task)
        out["last_result_accepted"] = result_accepted
    return out


def scheduled_task_summary_needs_action(task: dict[str, Any]) -> bool:
    if not task or task.get("supported") is False:
        return False
    if "needs_action" in task:
        if task.get("needs_action"):
            return True
    elif scheduled_task_needs_action(task):
        return True
    if task.get("supported") is True and task.get("installed") is not True:
        return True
    return task.get("installed") is False


def scheduled_task_label(task: dict[str, Any]) -> str:
    if task.get("supported") is False:
        return "unsupported"
    if task.get("installed") is True:
        state = str(task.get("state") or "installed")
        if scheduler_state_needs_action(task):
            return f"{state}/disabled"
        if task.get("last_result_accepted"):
            result = task.get("last_result_text") or task.get("last_result") or "unknown"
            return f"{state}/result={result}"
        if task.get("needs_action") or scheduler_result_needs_action(task):
            result = task.get("last_result_text") or task.get("last_result") or "unknown"
            return f"{state}/result={result}"
        return state
    if task.get("installed") is False:
        return "missing"
    return "unknown"


def scheduled_task_has_data(task: dict[str, Any]) -> bool:
    return any(task.get(key) is not None for key in ("supported", "installed", "state", "last_result", "reason"))


def task_is_installed(task: dict[str, Any]) -> bool:
    return task.get("supported") and task.get("installed") is True


def supervisor_payload_verdict(supervisor: dict[str, Any]) -> str:
    payload = supervisor.get("payload") or {}
    return str(payload.get("verdict", "")).upper()


def supervisor_payload_diagnosis_health(supervisor: dict[str, Any]) -> str:
    payload = supervisor.get("payload") or {}
    diagnose = payload.get("diagnose") or {}
    return str(diagnose.get("health", "")).upper()


def supervisor_payload_run_health(supervisor: dict[str, Any]) -> str:
    payload = supervisor.get("payload") or {}
    for section in (payload.get("last_decide") or {}, payload.get("ships") or {}):
        run_health = str(section.get("run_health") or "").strip().upper()
        if run_health:
            return run_health
    return ""


def supervisor_run_health_is_actionable(supervisor: dict[str, Any]) -> bool:
    run_health = supervisor_payload_run_health(supervisor)
    return bool(run_health and run_health not in {"OK", "HEALTHY", "RUNNING", "GREEN"})


def supervisor_diagnosis_is_actionable(supervisor: dict[str, Any]) -> bool:
    verdict = supervisor_payload_verdict(supervisor)
    if verdict in SUPERVISOR_ACTION_VERDICTS:
        return True
    if supervisor_run_health_is_actionable(supervisor):
        return True
    if verdict in SUPERVISOR_NON_ACTION_VERDICTS:
        return False
    health = supervisor_payload_diagnosis_health(supervisor)
    return bool(health and health not in {"HEALTHY", "OK"})


def supervisor_log_hint(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    run_id = str(payload.get("run") or "").strip()
    if not run_id:
        return {}
    job_dir_raw = config.get("job_dir") or ""
    if not job_dir_raw:
        return {"run": run_id, "run_dir": f"output/{run_id}"}
    run_dir = Path(str(job_dir_raw)) / "output" / run_id
    hint = {"run": run_id, "run_dir": str(run_dir)}
    try:
        logs = [p for p in run_dir.glob("worker-*/run.log") if p.is_file()]
    except OSError:
        logs = []
    if logs:
        newest = max(logs, key=lambda p: p.stat().st_mtime)
        hint["run_log"] = str(newest)
    return hint


def supervisor_health_summary(
    supervisor: dict[str, Any],
    config: dict[str, Any],
    *,
    watchdog_task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = supervisor.get("payload") or {}
    diagnose = payload.get("diagnose") or {}
    process = payload.get("process") or {}
    last_decide = payload.get("last_decide") or {}
    summary: dict[str, Any] = {
        "available": bool(supervisor.get("available")),
        "verdict": supervisor_payload_verdict(supervisor),
        "alive": process.get("alive"),
        "run_health": supervisor_payload_run_health(supervisor),
        "diagnose": supervisor_payload_diagnosis_health(supervisor),
        "run": payload.get("run"),
        "stop_reason": last_decide.get("stop_reason"),
        "primary_cause": diagnose.get("primary_cause"),
        "primary_action": diagnose.get("primary_action"),
        "next_action": payload.get("next_action"),
        "why": payload.get("why"),
    }
    summary.update(supervisor_log_hint(payload, config))
    if watchdog_task:
        result = watchdog_task.get("last_result")
        if result not in (None, 0):
            summary["watchdog_last_result"] = result
            summary["watchdog_last_result_text"] = scheduler_result_text(watchdog_task)
            try:
                summary["watchdog_last_result_accepted"] = (
                    scheduler_result_int(result) == SUPERVISOR_WATCHDOG_ACTION_RESULT
                )
            except (TypeError, ValueError):
                summary["watchdog_last_result_accepted"] = False
    return summary


def supervisor_health_needs_action(summary: dict[str, Any]) -> bool:
    if not summary.get("available"):
        return True
    if summary.get("alive") is False:
        return True
    verdict = str(summary.get("verdict") or "").upper()
    if verdict in SUPERVISOR_ACTION_VERDICTS:
        return True
    run_health = str(summary.get("run_health") or "").upper()
    if run_health and run_health not in {"OK", "HEALTHY", "RUNNING", "GREEN"}:
        return True
    diagnose = str(summary.get("diagnose") or "").upper()
    if verdict not in SUPERVISOR_NON_ACTION_VERDICTS and diagnose and diagnose not in {"OK", "HEALTHY"}:
        return True
    if summary.get("watchdog_last_result") not in (None, 0):
        return True
    return False


def supervisor_health_action_text(summary: dict[str, Any]) -> str:
    verdict = str(summary.get("verdict") or "UNKNOWN").upper()
    run_health = str(summary.get("run_health") or "").upper()
    diagnose = str(summary.get("diagnose") or "").upper()
    run_id = summary.get("run") or "unknown-run"
    bits = [f"Supervisor run {run_id}"]
    if run_health:
        bits.append(f"health={run_health}")
    bits.append(f"verdict={verdict}")
    if diagnose:
        bits.append(f"diagnosis={diagnose}")
    if summary.get("stop_reason"):
        bits.append(f"stop={summary['stop_reason']}")
    if summary.get("primary_cause"):
        bits.append(f"cause={summary['primary_cause']}")
    if summary.get("watchdog_last_result") not in (None, 0):
        result = summary.get("watchdog_last_result_text") or summary.get("watchdog_last_result")
        accepted = "accepted watchdog status, not setup failure" if summary.get("watchdog_last_result_accepted") else "inspect scheduler result"
        bits.append(f"watchdog={result}; {accepted}")
    inspect = summary.get("run_log") or summary.get("run_dir")
    if inspect:
        bits.append(f"inspect `{inspect}`")
    primary = summary.get("primary_action") or summary.get("next_action") or summary.get("why")
    if primary:
        bits.append(str(primary))
    command = "inspect `python tools/fleet_control_pane.py supervisor --json`"
    if summary.get("alive") is False:
        command += " or restart with `python tools/fleet_control_pane.py supervisor --restart --apply`"
    elif verdict in SUPERVISOR_ACTION_VERDICTS:
        command += " or dry-run restart with `python tools/fleet_control_pane.py supervisor --restart`"
    return "; ".join(bits) + f"; {command}."


def pane_text(status: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"FLEET CONTROL PANE @ {status['generated_utc']}")
    lines.append(f"verdict: {status['verdict']}  host={status['machine']['host']}")

    git_status = status.get("git") or {}
    if git_status.get("available"):
        counts = git_status.get("counts") or {}
        lines.append(
            "git: "
            f"{git_status.get('branch')} {git_status.get('head')}  "
            f"dirty={git_status.get('dirty_total', 0)} "
            f"M={counts.get('modified', 0)} D={counts.get('deleted', 0)} "
            f"?={counts.get('untracked', 0)}"
        )
        if counts.get("unmerged"):
            lines[-1] += f" U={counts.get('unmerged', 0)}"
        if git_status.get("merge_in_progress"):
            lines.append("merge: in-progress")
        safe_ff = git_status.get("safe_ff") or {}
        lines.append(
            "safe-ff: "
            f"state={safe_ff.get('state')} ok={safe_ff.get('ok')} "
            f"divergent={safe_ff.get('divergent_count', 0)}"
        )
        worktrees = git_status.get("worktrees") or {}
        if worktrees.get("available"):
            lines.append(
                "worktrees: "
                f"total={worktrees.get('total', 0)} "
                f"converged={worktrees.get('converged')} "
                f"prune={worktrees.get('prune_count', 0)} "
                f"blocked={worktrees.get('blocked_count', 0)} "
                f"retained={worktrees.get('retained_count', 0)}"
            )
            for item in (worktrees.get("blocked") or [])[:3]:
                reasons = ", ".join(str(reason) for reason in item.get("reasons") or [])
                lines.append(
                    f"  blocked-worktree: {item.get('path')} "
                    f"({item.get('branch') or 'detached'}): {reasons or 'inspect'}"
                )
        elif worktrees:
            lines.append(f"worktrees: unavailable ({worktrees.get('reason')})")
        dirty_plan = git_status.get("dirty_plan") or {}
        dirty_groups = dirty_plan.get("groups") or []
        if dirty_groups:
            lines.append("dirty-plan:")
            for group in dirty_groups[:6]:
                lines.append(f"  {group.get('group')}: {group.get('count', 0)} path(s)")
                if group.get("command"):
                    lines.append(f"    {group['command']}")
                elif group.get("blocked"):
                    lines.append(f"    blocked: {group.get('reason', 'inspect dirty-plan JSON')}")
            omitted = max(0, len(dirty_groups) - 6)
            if omitted:
                lines.append(f"  ... {omitted} more group(s); use `status --json` for all dirty-plan paths")
    else:
        lines.append(f"git: unavailable ({git_status.get('reason')})")

    reg = status.get("registry") or {}
    if reg.get("exists"):
        lines.append(
            "sessions: "
            f"{reg.get('sessions', 0)} age={reg.get('age_min')}m "
            f"categories={compact_counts(reg.get('categories') or {})}"
        )
        lines.append(
            "actions: "
            f"{compact_counts(reg.get('actions') or {})}"
        )
        accts = reg.get("accounts") or {}
        lines.append(
            "accounts: "
            f"available={accts.get('available', 0)}/{accts.get('total', 0)} "
            f"blocked={len(accts.get('blocked', []))} "
            f"tags={','.join(accts.get('available_tags', [])[:8]) or '(none)'}"
        )
        blocked_text = blocked_account_reasons(accts)
        if blocked_text:
            lines.append(f"blocked-accounts: {blocked_text}")
            for account_line in blocked_account_action_lines(accts):
                lines.append(f"account-action: {account_line}")
    else:
        lines.append("sessions: registry missing")

    sup = status.get("supervisor") or {}
    if sup.get("available"):
        payload = sup.get("payload") or {}
        proc = payload.get("process") or {}
        sup_summary = supervisor_health_summary(
            sup,
            status.get("config") or {},
            watchdog_task=(((status.get("watchdogs") or {}).get("supervisor") or {}).get("task") or {}),
        )
        lines.append(
            "supervisor: "
            f"verdict={payload.get('verdict')} alive={proc.get('alive')} "
            f"pid={proc.get('pid')} hb_age={proc.get('heartbeat_age_s')} "
            f"run={sup_summary.get('run') or '(none)'} "
            f"health={sup_summary.get('run_health') or sup_summary.get('diagnose') or 'OK'}"
        )
        if supervisor_health_needs_action(sup_summary):
            lines.append(f"supervisor-action: {supervisor_health_action_text(sup_summary)}")
    else:
        lines.append(f"supervisor: unavailable ({sup.get('reason')})")

    tick = status.get("control_tick") or {}
    tick_task = tick.get("task") or {}
    if tick_task.get("supported"):
        state = "installed" if tick_task.get("installed") else "missing"
        extra = f" state={tick_task.get('state')}" if tick_task.get("state") else ""
    else:
        state = "unsupported"
        extra = f" ({tick_task.get('reason')})"
    lines.append(f"control-tick: {state}{extra}")

    for name, wd in sorted((status.get("watchdogs") or {}).items()):
        task = wd.get("task") or {}
        if task.get("supported"):
            state = "installed" if task.get("installed") else "missing"
            extra = f" state={task.get('state')}" if task.get("state") else ""
        else:
            state = "unsupported"
            extra = f" ({task.get('reason')})"
        lines.append(f"watchdog[{name}]: {state}{extra}")

    loops = status.get("loops") or {}
    if loops.get("count"):
        states = compact_counts(loops.get("states") or {})
        lines.append(f"loops: {states}")
        for check in (loops.get("checks") or [])[:5]:
            if check.get("state") != "OK":
                detail = check.get("detail") or check.get("reason") or ""
                lines.append(f"loop[{check.get('name')}]: {check.get('state')} {detail}".rstrip())
    elif "enabled" in loops:
        lines.append(
            "loops: "
            f"enabled={loops.get('enabled', 0)} "
            f"disabled={loops.get('disabled', 0)} "
            f"configured={loops.get('configured', 0)}"
        )

    if status.get("actions"):
        lines.append("recommended:")
        for action in status["actions"]:
            lines.append(f"  - {action}")
    return "\n".join(lines)


def blocked_account_reasons(accounts: dict[str, Any], *, limit: int = 5) -> str:
    blocked = accounts.get("blocked") or []
    blocked_accounts = []
    for account in blocked[:limit]:
        if not isinstance(account, dict):
            continue
        tag = str(account.get("tag") or "unknown")
        reason = str(account.get("reason") or "").strip()
        kind = str(account.get("block_kind") or "").strip()
        sessions = int(account.get("auth_blocked_sessions") or 0)
        suffix = ""
        if kind:
            suffix += f"/{kind}"
        if sessions:
            suffix += f"/{sessions}s"
        label = f"{tag}[{suffix.lstrip('/')}]" if suffix else tag
        blocked_accounts.append(f"{label}={reason}" if reason else label)
    if not blocked_accounts:
        return ""
    omitted = max(0, len(blocked) - len(blocked_accounts))
    suffix = f" | +{omitted} more" if omitted else ""
    return f"{' | '.join(blocked_accounts)}{suffix}"


def blocked_account_action_lines(accounts: dict[str, Any], *, limit: int = 3) -> list[str]:
    lines = []
    for account in (accounts.get("blocked") or [])[:limit]:
        if not isinstance(account, dict):
            continue
        tag = str(account.get("tag") or "unknown")
        reason = str(account.get("reason") or "blocked").strip()
        config_dir = str(account.get("config_dir") or "").strip()
        command = str(account.get("command") or "").strip()
        detail = f"{tag}: {reason}"
        if config_dir:
            detail += f"; profile={config_dir}"
        if command and account_block_needs_login(account):
            detail += f"; login/check with `{command}`"
        lines.append(detail)
    blocked = accounts.get("blocked") or []
    omitted = max(0, len(blocked) - len(lines))
    if omitted:
        lines.append(f"... {omitted} more blocked account(s); inspect `python tools/fleet_accounts.py list`")
    return lines


def blocked_account_recommended_action(registry_summary: dict[str, Any]) -> str:
    accounts = (registry_summary.get("accounts") or {}).get("blocked") or []
    if not accounts and not int(registry_summary.get("auth_blocked") or 0):
        return ""
    auth_accounts = [a for a in accounts if isinstance(a, dict) and account_block_needs_login(a)]
    credit_accounts = [
        a for a in accounts
        if isinstance(a, dict) and str(a.get("block_kind") or "").lower() == "credit"
    ]
    usage_accounts = [a for a in accounts if isinstance(a, dict) and account_block_is_usage_like(a)]
    access_accounts = [a for a in accounts if isinstance(a, dict) and account_block_is_access_wall(a)]
    other_accounts = [
        a for a in accounts
        if isinstance(a, dict)
        and a not in auth_accounts
        and a not in credit_accounts
        and a not in usage_accounts
        and a not in access_accounts
    ]
    session_count = (
        sum(int(a.get("auth_blocked_sessions") or 0) for a in auth_accounts)
        if auth_accounts else int(registry_summary.get("auth_blocked") or 0)
    )
    parts = []
    if auth_accounts:
        parts.append(
            f"{len(auth_accounts)} login-blocked account(s)"
            f"{f' covering {session_count} auth-stopped session(s)' if session_count else ''}: "
            f"{blocked_account_tags(auth_accounts)}"
        )
    elif session_count and not accounts:
        parts.append(f"{session_count} auth-stopped session(s) but no account-level auth blocker is currently active")
    if credit_accounts:
        parts.append(f"{len(credit_accounts)} credit-blocked account(s): {blocked_account_tags(credit_accounts)}")
    if usage_accounts:
        # Usage limits are expected time walls. They stay visible in the account
        # summary, but should not create an operator remediation prompt.
        pass
    if other_accounts:
        parts.append(f"{len(other_accounts)} other blocked account(s): {blocked_account_tags(other_accounts)}")
    first_command = next((str(a.get("command")) for a in auth_accounts if a.get("command")), "")
    command_text = f"; first remediation command: `{first_command}`" if first_command else ""
    if not parts:
        return ""
    return (
        "Account remediation needed: "
        + "; ".join(parts)
        + command_text
        + "; inspect `python tools/fleet_accounts.py list`."
    )


def blocked_account_tags(accounts: list[dict[str, Any]], *, limit: int = 4) -> str:
    parts = []
    for account in accounts[:limit]:
        tag = str(account.get("tag") or "unknown")
        reason = str(account.get("reason") or "").strip()
        sessions = int(account.get("auth_blocked_sessions") or 0)
        session_text = f", {sessions} session(s)" if sessions else ""
        parts.append(f"{tag}={reason}{session_text}" if reason else f"{tag}{session_text}")
    omitted = max(0, len(accounts) - len(parts))
    if omitted:
        parts.append(f"+{omitted} more")
    return " | ".join(parts)


def dirty_group_summary(groups: list[dict[str, Any]], *, limit: int = 4) -> str:
    parts = [
        f"{group.get('group')}={group.get('count', 0)}"
        for group in groups[:limit]
        if group.get("group")
    ]
    if not parts:
        return "(none)"
    omitted = max(0, len(groups) - len(parts))
    suffix = f" +{omitted} more" if omitted else ""
    return f"{', '.join(parts)}{suffix}"


def machine_setup_action_count(machine: dict[str, Any]) -> int:
    count = 1 if scheduled_task_summary_needs_action(machine.get("control_tick") or {}) else 0
    for task in (machine.get("watchdogs") or {}).values():
        count += 1 if scheduled_task_summary_needs_action(task or {}) else 0
    return count


def machine_setup_summary(machine: dict[str, Any]) -> str:
    parts: list[str] = []
    tick = machine.get("control_tick") or {}
    if scheduled_task_has_data(tick):
        parts.append(f"tick={scheduled_task_label(tick)}")
    watchdogs = machine.get("watchdogs") or {}
    watchdog_parts = [
        f"{name}={scheduled_task_label(task or {})}"
        for name, task in sorted(watchdogs.items())
        if scheduled_task_has_data(task or {})
    ]
    if watchdog_parts:
        parts.append(f"watchdogs={','.join(watchdog_parts)}")
    return " ".join(parts)


def setup_step_snapshot(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": step.get("id"),
        "state": step.get("state"),
        "ok": step.get("ok"),
        "detail": step.get("detail"),
        "display": step.get("display") or display_command(list(step.get("command") or [])),
        "command": list(step.get("command") or []),
    }


def setup_plan_snapshot(plan: dict[str, Any]) -> dict[str, Any]:
    steps = [
        setup_step_snapshot(step)
        for step in plan.get("steps") or []
        if isinstance(step, dict)
    ]
    states = collections.Counter(str(step.get("state") or "unknown") for step in steps)
    actionable = [
        step
        for step in steps
        if step.get("state") in {"todo", "blocked"} or step.get("ok") is False
    ]
    return {
        "states": dict(states),
        "action_count": len(actionable),
        "steps": steps,
    }


def setup_plan_action_steps(machine: dict[str, Any]) -> list[dict[str, Any]]:
    plan = machine.get("setup_plan") or {}
    return [
        step
        for step in (plan.get("steps") or [])
        if step.get("state") in {"todo", "blocked"} or step.get("ok") is False
    ]


def machine_git_action(git_status: dict[str, Any]) -> str:
    state = str(git_status.get("safe_ff_state") or "")
    if state == "ahead":
        if int(git_status.get("dirty_total") or 0) > 0 or git_status.get("merge_in_progress"):
            return ""
        return "python tools/fleet_control_pane.py publish --json"
    if state in {"behind", "diverged"} or git_status.get("safe_ff_ok") is False:
        return "python tools/fleet_control_pane.py sync --fetch --json"
    return ""


def machine_version_drift(machine: dict[str, Any], current_version: str) -> dict[str, Any] | None:
    if not current_version:
        return None
    machine_version = str(machine.get("app_version") or "unknown")
    if machine_version == current_version:
        return None
    git_status = machine.get("git") or {}
    git_state = str(git_status.get("safe_ff_state") or "unknown")
    command = machine_git_action(git_status) or "python tools/fleet_control_pane.py sync --fetch --json"
    detail = "pane version differs from this operator pane"
    if git_state == "in-sync":
        detail += "; publish the newer pane first if it is only local, then sync this host"
    return {
        "machine_version": machine_version,
        "current_version": current_version,
        "git_state": git_state,
        "command": command,
        "detail": detail,
    }


def loop_issue_summary(check: dict[str, Any]) -> str:
    name = check.get("name") or "unknown"
    state = check.get("state") or "UNKNOWN"
    detail = check.get("detail") or check.get("action") or ""
    flags = []
    if check.get("auto_recover"):
        flags.append("auto-recover")
    elif check.get("has_recover_cmd"):
        flags.append("recoverable")
    suffix = f" ({', '.join(flags)})" if flags else ""
    body = f" {detail}" if detail else ""
    return f"loop[{name}]: {state}{suffix}{body}".rstrip()


def compact_counts(counts: dict[str, Any]) -> str:
    parts = [f"{k}={v}" for k, v in sorted(counts.items()) if k and v]
    return " ".join(parts) if parts else "(none)"


def write_status(status: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    reg_dir = Path(config["registry_dir"])
    json_path = reg_dir / "control_pane.json"
    text_path = reg_dir / "CONTROL-PANE.txt"
    machine_path = Path(config["machine_dir"]) / f"{machine_id(config)}.json"
    pane_write_json = pane_attr("write_json", write_json)
    pane_write_text_atomic = pane_attr("write_text_atomic", write_text_atomic)
    pane_pane_text = pane_attr("pane_text", pane_text)
    pane_write_json(json_path, status)
    pane_write_text_atomic(text_path, pane_pane_text(status) + "\n")
    pane_write_json(machine_path, status)
    return {"json": str(json_path), "text": str(text_path), "machine": str(machine_path)}


def summarize_machine_snapshot(path: Path, status: dict[str, Any], stale_min: float) -> dict[str, Any]:
    if not status:
        return {
            "id": path.stem,
            "source": str(path),
            "state": "INVALID",
            "reason": "missing or malformed JSON",
        }
    machine = status.get("machine") or {}
    reg = status.get("registry") or {}
    accts = reg.get("accounts") or {}
    blocked_accounts = []
    for account in accts.get("blocked") or []:
        if not isinstance(account, dict):
            continue
        blocked_accounts.append({
            **account,
            "source_machine": str(machine.get("id") or path.stem),
            "source_host": str(machine.get("host") or path.stem),
            "source_snapshot": str(path),
        })
    supervisor = status.get("supervisor") or {}
    sup_payload = supervisor.get("payload") or {}
    sup_proc = sup_payload.get("process") or {}
    status_config = status.get("config") or {}
    supervisor_summary = supervisor_health_summary(
        supervisor,
        status_config,
        watchdog_task=(((status.get("watchdogs") or {}).get("supervisor") or {}).get("task") or {}),
    )
    tick = status.get("control_tick") or {}
    tick_task = tick.get("task") or {}
    watchdogs = status.get("watchdogs") or {}
    loops = status.get("loops") or {}
    loop_checks = loops.get("checks") or []
    loop_summaries = [loop_check_snapshot(check) for check in loop_checks]
    setup_summary = setup_plan_snapshot(status.get("setup_plan") or {})
    git_status = status.get("git") or {}
    safe_ff = git_status.get("safe_ff") or {}
    worktrees = git_status.get("worktrees") or {}
    dirty_plan = git_status.get("dirty_plan") or {}
    dirty_groups = dirty_plan.get("groups") or []
    age = pane_attr("parse_iso_age_min", parse_iso_age_min)(status.get("generated_utc"))
    verdict = str(status.get("verdict") or "UNKNOWN")
    state = verdict
    if age is None:
        state = "UNKNOWN"
    elif age > stale_min:
        state = "STALE"
    return {
        "id": str(machine.get("id") or path.stem),
        "host": str(machine.get("host") or path.stem),
        "app_version": status.get("app_version"),
        "source": str(path),
        "generated_utc": status.get("generated_utc"),
        "age_min": age,
        "state": state,
        "verdict": verdict,
        "actions_count": len(status.get("actions") or []),
        "actions": list(status.get("actions") or [])[:12],
        "sessions": reg.get("sessions", 0),
        "categories": reg.get("categories") or {},
        "session_actions": reg.get("actions") or {},
        "auth_blocked": reg.get("auth_blocked", 0),
        "auto_resume": reg.get("auto_resume", 0),
        "surface": reg.get("surface", 0),
        "accounts": {
            "available": accts.get("available", 0),
            "total": accts.get("total", 0),
            "available_tags": accts.get("available_tags") or [],
            "blocked": blocked_accounts,
            "blocked_count": accts.get("blocked_count", len(blocked_accounts)),
            "auth_blocked_count": accts.get("auth_blocked_count", sum(1 for a in blocked_accounts if account_block_is_auth_like(a))),
            "usage_blocked_count": accts.get("usage_blocked_count", sum(1 for a in blocked_accounts if account_block_is_usage_like(a))),
            "blocked_kinds": accts.get("blocked_kinds") or {},
        },
        "supervisor": {
            "available": supervisor.get("available", False),
            "verdict": sup_payload.get("verdict"),
            "alive": sup_proc.get("alive"),
            "pid": sup_proc.get("pid"),
            "heartbeat_age_s": sup_proc.get("heartbeat_age_s"),
            "diagnose": (sup_payload.get("diagnose") or {}).get("health"),
            "run": supervisor_summary.get("run"),
            "run_health": supervisor_summary.get("run_health"),
            "primary_cause": supervisor_summary.get("primary_cause"),
            "primary_action": supervisor_summary.get("primary_action"),
            "run_dir": supervisor_summary.get("run_dir"),
            "run_log": supervisor_summary.get("run_log"),
            "watchdog_last_result": supervisor_summary.get("watchdog_last_result"),
            "watchdog_last_result_text": supervisor_summary.get("watchdog_last_result_text"),
            "watchdog_last_result_accepted": supervisor_summary.get("watchdog_last_result_accepted"),
            "needs_action": supervisor_health_needs_action(supervisor_summary),
            "action": supervisor_health_action_text(supervisor_summary)
            if supervisor_health_needs_action(supervisor_summary)
            else "",
        },
        "control_tick": scheduled_task_snapshot(tick_task or {}),
        "watchdogs": {
            name: scheduled_task_snapshot(
                (wd or {}).get("task") or {},
                accepted_results={SUPERVISOR_WATCHDOG_ACTION_RESULT} if name == "supervisor" else None,
            )
            for name, wd in sorted(watchdogs.items())
        },
        "setup_plan": setup_summary,
        "loops": {
            "count": loops.get("count", 0),
            "configured": loops.get("configured", loops.get("count", 0)),
            "enabled": loops.get("enabled", loops.get("count", 0)),
            "disabled": loops.get("disabled", 0),
            "states": loops.get("states") or {},
            "action": sum(1 for check in loop_summaries if loop_check_needs_action(check)),
            "checks": loop_summaries,
            "commands": list(loops.get("commands") or [])[:3],
        },
        "git": {
            "branch": git_status.get("branch"),
            "head": git_status.get("head"),
            "dirty_total": git_status.get("dirty_total", 0),
            "merge_in_progress": git_status.get("merge_in_progress", False),
            "dirty_plan": {
                "count": dirty_plan.get("count", len(dirty_groups)),
                "groups": list(dirty_groups),
            },
            "safe_ff_state": safe_ff.get("state"),
            "safe_ff_ok": safe_ff.get("ok"),
            "safe_ff_reason": safe_ff.get("reason"),
            "worktrees": {
                "available": worktrees.get("available", False),
                "total": worktrees.get("total", 0),
                "converged": worktrees.get("converged"),
                "needs_human": worktrees.get("needs_human", False),
                "primary_offtrack": worktrees.get("primary_offtrack", False),
                "prune_count": worktrees.get("prune_count", 0),
                "blocked_count": worktrees.get("blocked_count", 0),
                "retained_count": worktrees.get("retained_count", 0),
                "prune": list(worktrees.get("prune") or []),
                "blocked": list(worktrees.get("blocked") or []),
                "commands": worktrees.get("commands") or {},
                "reason": worktrees.get("reason"),
            },
        },
    }


def fleet_text(doc: dict[str, Any]) -> str:
    lines = [
        f"FLEET CONTROL PANE AGGREGATE @ {doc['generated_utc']}",
        f"verdict: {doc['verdict']}  machines={len(doc.get('machines') or [])}  states={compact_counts(doc.get('states') or {})}",
    ]
    versions = compact_counts(doc.get("versions") or {})
    if versions != "(none)":
        lines.append(f"versions: {versions}")
    totals = doc.get("totals") or {}
    lines.append(
        "totals: "
        f"sessions={totals.get('sessions', 0)} actions={totals.get('actions', 0)} "
        f"auth_sessions={totals.get('auth_blocked', 0)} "
        f"auth_accounts={totals.get('auth_blocked_accounts', 0)} "
        f"blocked_accounts={totals.get('blocked_accounts', 0)} "
        f"auto_resume={totals.get('auto_resume', 0)} "
        f"setup={totals.get('setup_actions', 0)} setup_plan={totals.get('setup_plan_actions', 0)} "
        f"version={totals.get('version_mismatches', 0)} "
        f"loops={totals.get('loop_actions', 0)} "
        f"dirty={totals.get('dirty_paths', 0)} "
        f"wt_prune={totals.get('worktree_prune', 0)} "
        f"wt_blocked={totals.get('worktree_blocked', 0)}"
    )
    if doc.get("written"):
        written = doc["written"]
        lines.append(f"local-written: {written.get('json')} ; {written.get('machine')}")
    for m in doc.get("machines") or []:
        sup = m.get("supervisor") or {}
        git_status = m.get("git") or {}
        accounts = m.get("accounts") or {}
        loops = m.get("loops") or {}
        source = " live" if m.get("live_local") else ""
        lines.append(
            f"- {m.get('id')}{source}: state={m.get('state')} age={m.get('age_min')}m "
            f"ver={m.get('app_version') or 'unknown'} "
            f"sessions={m.get('sessions', 0)} actions={m.get('actions_count', 0)} "
            f"acct={accounts.get('available', 0)}/{accounts.get('total', 0)} "
            f"sup={sup.get('verdict')}/{sup.get('diagnose')} "
            f"loops={loops.get('action', 0)}/{loops.get('count', 0)} "
            f"git={git_status.get('branch')} sync={git_status.get('safe_ff_state') or 'unknown'} "
            f"dirty={git_status.get('dirty_total', 0)}"
        )
        blocked_text = blocked_account_reasons(accounts, limit=3)
        if blocked_text:
            lines.append(f"    blocked-accounts: {blocked_text}")
            for account_line in blocked_account_action_lines(accounts, limit=2):
                lines.append(f"    account-action: {account_line}")
        setup_text = machine_setup_summary(m)
        if setup_text:
            lines.append(f"    setup: {setup_text}")
        if sup.get("action"):
            lines.append(f"    supervisor-action: {sup['action']}")
        drift = m.get("version_drift") or {}
        if drift:
            lines.append(
                "    version-action: "
                f"pane {drift.get('machine_version')} -> {drift.get('current_version')}; "
                f"{drift.get('command')}"
            )
            if drift.get("detail"):
                lines.append(f"      {drift['detail']}")
        git_action = machine_git_action(git_status)
        if git_action:
            reason = f" ({git_status.get('safe_ff_reason')})" if git_status.get("safe_ff_reason") else ""
            lines.append(f"    git-action: {git_action}{reason}")
        worktrees = git_status.get("worktrees") or {}
        if int(worktrees.get("prune_count") or 0) > 0:
            cmd = (worktrees.get("commands") or {}).get("prune") or "python tools/worktree_doctor.py --prune --fetch"
            lines.append(f"    worktree-action: prune {worktrees.get('prune_count')} safe extra worktree(s); {cmd}")
        if int(worktrees.get("blocked_count") or 0) > 0 or worktrees.get("primary_offtrack"):
            cmd = (worktrees.get("commands") or {}).get("inspect") or "python tools/worktree_doctor.py --fetch"
            lines.append(f"    worktree-action: inspect blocked/off-track worktree state; {cmd}")
        setup_steps = setup_plan_action_steps(m)
        for step in setup_steps[:3]:
            detail = f" {step.get('detail')}" if step.get("detail") else ""
            lines.append(f"    setup-step[{step.get('id')}]: {step.get('state')}{detail}".rstrip())
            if step.get("display"):
                lines.append(f"      {step['display']}")
        omitted_setup = max(0, len(setup_steps) - 3)
        if omitted_setup:
            lines.append(f"    ... {omitted_setup} more setup step(s); use `fleet --json` for all setup-plan commands")
        loop_issues = [
            check
            for check in (loops.get("checks") or [])
            if loop_check_needs_action(check)
        ]
        for check in loop_issues[:3]:
            lines.append(f"    {loop_issue_summary(check)}")
        omitted_loops = max(0, len(loop_issues) - 3)
        if omitted_loops:
            lines.append(f"    ... {omitted_loops} more loop issue(s); use `fleet --json` for all loop checks")
        if int(loops.get("enabled", loops.get("count") or 0) or 0) == 0 and loops.get("commands"):
            lines.append(f"    loop-action: {loops['commands'][0]}")
        dirty_plan = git_status.get("dirty_plan") or {}
        dirty_groups = dirty_plan.get("groups") or []
        if dirty_groups:
            lines.append(f"    dirty-groups: {dirty_group_summary(dirty_groups, limit=4)}")
            shown_commands = 0
            for group in dirty_groups:
                command = group.get("command")
                if not command:
                    continue
                lines.append(f"      {command}")
                shown_commands += 1
                if shown_commands >= 3:
                    break
            omitted = max(0, len(dirty_groups) - shown_commands)
            if omitted:
                lines.append(f"      ... {omitted} more group(s); use `fleet --json` for all dirty-plan paths")
        for action in (m.get("actions") or [])[:3]:
            lines.append(f"    - {action}")
    if not doc.get("machines"):
        lines.append(f"no machine snapshots found in {doc.get('machine_dir')}")
    return "\n".join(lines)


def _print_setup_plan(doc: dict[str, Any]) -> None:
    for step in doc["steps"]:
        state = str(step.get("state") or "step").upper()
        display = step.get("display") or "(no command)"
        print(f"{state} {step['id']}: {display}")
        print(f"  {step['why']}")
        if step.get("detail"):
            print(f"  {step['detail']}")


def _print_doctor(doc: dict[str, Any]) -> None:
    print(f"doctor: {'OK' if doc['ok'] else 'ACTION'}")
    for c in doc["checks"]:
        print(f"  {'OK' if c['ok'] else 'MISS'}  {c['name']}: {c['detail']}")


def _print_tick(doc: dict[str, Any]) -> None:
    print(f"tick: {'OK' if doc['ok'] else 'ACTION'} dry_run={doc['dry_run']} live_resume={doc['live_resume']}")
    for action in doc.get("actions", []):
        state = "SKIP" if action.get("skipped") else ("OK" if action.get("ok") else "FAIL")
        detail = action.get("reason") or f"rc={action.get('returncode')}"
        print(f"  {state}  {action.get('name')}: {detail}")
    print(pane_attr("pane_text", pane_text)(doc["status"]))
    if doc.get("written"):
        print(f"written: {doc['written']['json']} ; {doc['written']['text']} ; {doc['written']['machine']}")


def _print_recover(doc: dict[str, Any]) -> None:
    print(f"recover: {'OK' if doc.get('ok') else 'ACTION'} apply={doc.get('apply')}")
    resume_plan = doc.get("resume_plan") or {}
    plan_path = resume_plan.get("display_path") or resume_plan.get("path") or "resume_plan.json"
    if resume_plan:
        if resume_plan.get("exists"):
            print(f"resume-plan: {resume_plan.get('count', 0)} eligible session(s) from {plan_path}")
            if resume_plan.get("reason"):
                print(f"  {resume_plan['reason']}")
            for session in resume_plan.get("sessions") or []:
                sid = session.get("session_short") or session.get("session") or "unknown"
                account = session.get("account") or "unknown-account"
                project = session.get("project") or session.get("cwd") or "unknown-project"
                print(f"  session={sid} account={account} project={project}")
                if session.get("resume_cmd"):
                    print(f"       {session['resume_cmd']}")
            if resume_plan.get("omitted"):
                print(f"  ... {resume_plan['omitted']} more")
        else:
            print(f"resume-plan: missing ({plan_path})")
    for action in doc.get("actions", []):
        state = "SKIP" if action.get("skipped") else ("OK" if action.get("ok") else "FAIL")
        detail = action.get("reason") or f"rc={action.get('returncode')}"
        print(f"  {state}  {action.get('name')}: {detail}")
        if action.get("command"):
            print(f"       {action.get('command')}")
    if doc.get("status"):
        print(pane_attr("pane_text", pane_text)(doc["status"]))
    if doc.get("written"):
        print(f"written: {doc['written']['json']} ; {doc['written']['text']} ; {doc['written']['machine']}")


def _print_supervisor(doc: dict[str, Any]) -> None:
    before = doc.get("before") or {}
    payload = before.get("payload") or {}
    proc = payload.get("process") or {}
    state = "ACTION" if doc.get("needs_action") else ("OK" if doc.get("ok") else "ERROR")
    print(f"supervisor: {state} verdict={payload.get('verdict')} alive={proc.get('alive')} pid={proc.get('pid')} restart={doc.get('restart')} apply={doc.get('apply')}")
    diagnose = payload.get("diagnose") or {}
    if diagnose:
        print(f"diagnose: {diagnose.get('health')} {diagnose.get('primary_action') or diagnose.get('primary_cause') or ''}".rstrip())
    if before.get("reason"):
        print(f"reason: {before.get('reason')}")
    for action in doc.get("actions", []):
        state = "SKIP" if action.get("skipped") else ("OK" if action.get("ok") else "FAIL")
        detail = action.get("reason") or f"rc={action.get('returncode')}"
        print(f"  {state}  {action.get('name')}: {detail}")
        if action.get("command"):
            print(f"       {action.get('command')}")


def _print_fleet(doc: dict[str, Any]) -> None:
    print(pane_attr("fleet_text", fleet_text)(doc))


def _print_commit(doc: dict[str, Any]) -> None:
    print(f"commit: {'OK' if doc.get('ok') else 'REFUSE'} applied={doc.get('applied')}")
    if doc.get("commit"):
        print(f"commit_sha: {doc['commit']}")
    if doc.get("reason"):
        print(f"reason: {doc['reason']}")
    if doc.get("paths"):
        print("paths:")
        for path in doc["paths"]:
            print(f"  - {path}")
    dirty_groups = doc.get("dirty_groups") or {}
    if dirty_groups.get("requested"):
        print("dirty groups:")
        for group in dirty_groups.get("requested") or []:
            print(f"  - {group}")
        if dirty_groups.get("available") and not dirty_groups.get("ok"):
            print("available dirty groups:")
            for group in dirty_groups["available"]:
                print(f"  - {group}")
    if doc.get("commands"):
        print("commands:")
        for command in doc["commands"]:
            print(f"  {command}")
    if doc.get("foreign_staged"):
        print("foreign staged:")
        for path in doc["foreign_staged"]:
            print(f"  - {path}")


def _print_sync(doc: dict[str, Any]) -> None:
    state = doc.get("state")
    print(f"sync: {'OK' if doc.get('ok') else 'REFUSE'} state={state} apply={doc.get('apply')} fetch={doc.get('fetch')}")
    if doc.get("branch"):
        print(f"branch: {doc.get('branch')} remote={doc.get('remote')}")
    if doc.get("reason"):
        print(f"reason: {doc.get('reason')}")
    info = doc.get("info") or {}
    if info.get("target_ref"):
        print(f"target: {info.get('target_ref')}")
    if info.get("write_count") is not None:
        print(f"write_set: {info.get('write_count')} path(s) divergent={len(info.get('divergent') or [])}")
    if info.get("new_head"):
        print(f"new_head: {info.get('new_head')}")
    if doc.get("commands"):
        print("commands:")
        for command in doc["commands"]:
            print(f"  {command}")


def _print_publish(doc: dict[str, Any]) -> None:
    print(f"publish: {'OK' if doc.get('ok') else 'REFUSE'} state={doc.get('state')} applied={doc.get('applied')} fetch={doc.get('fetch')}")
    if doc.get("branch"):
        print(f"branch: {doc.get('branch')} remote={doc.get('remote')}")
    if doc.get("reason"):
        print(f"reason: {doc.get('reason')}")
    ahead = doc.get("ahead_commits") or {}
    if ahead.get("count"):
        suffix = "" if ahead.get("shown") == ahead.get("count") else f" (showing {ahead.get('shown')})"
        print(f"ahead_commits: {ahead.get('count')} vs {ahead.get('target')}{suffix}")
        for commit in ahead.get("commits", []):
            print(f"  {commit.get('short')} {commit.get('subject')}")
    if doc.get("commands"):
        print("commands:")
        for command in doc["commands"]:
            print(f"  {command}")


def _print_bootstrap(doc: dict[str, Any]) -> None:
    print(f"bootstrap: {'OK' if doc.get('ok') else 'ACTION'} apply={doc.get('apply')} live_resume={doc.get('live_resume')}")
    for action in doc.get("actions", []):
        if action.get("skipped"):
            state = "SKIP"
        else:
            state = "OK" if action.get("ok") else "FAIL"
        print(f"  {state}  {action.get('id')}: {action.get('detail')}")
        if action.get("command"):
            print(f"       {action.get('command')}")


def _print_loop_config(doc: dict[str, Any]) -> None:
    print(f"loop-add: {'OK' if doc.get('ok') else 'REFUSE'} apply={doc.get('apply')} changed={doc.get('changed')}")
    print(f"loop: {doc.get('loop')}")
    print(f"scope: {doc.get('scope') or 'local'}")
    print(f"config: {doc.get('config_path') or doc.get('local_config')}")
    if doc.get("reason"):
        print(f"reason: {doc.get('reason')}")
    spec = doc.get("spec") or {}
    if spec.get("status_cmd"):
        print(f"status_cmd: {display_command(list(spec.get('status_cmd') or []))}")
    if spec.get("recover_cmd"):
        print(f"recover_cmd: {display_command(list(spec.get('recover_cmd') or []))}")
    if doc.get("command") and not doc.get("apply"):
        print("commands:")
        print(f"  {doc['command']}")
    if doc.get("followup_commands"):
        print("next:")
        for command in doc["followup_commands"]:
            print(f"  {command}")


def _print_loop_list(doc: dict[str, Any]) -> None:
    print(
        f"loops: {'OK' if doc.get('ok') else 'ACTION'} "
        f"count={doc.get('count', 0)} enabled={doc.get('enabled', 0)} "
        f"disabled={doc.get('disabled', 0)} blocked={doc.get('blocked', 0)}"
    )
    for loop in doc.get("loops") or []:
        if not loop.get("enabled"):
            state = "DISABLED"
        else:
            state = "OK" if loop.get("ready") else "ACTION"
        suffix = " overridden" if loop.get("overridden") else ""
        print(f"- {loop.get('name')}: {state} source={loop.get('source')}{suffix}")
        if not loop.get("enabled"):
            print("  readiness: skipped because loop is disabled")
            continue
        status_ready = loop.get("status_ready") or {}
        print(f"  status: {'OK' if status_ready.get('ok') else 'MISS'} {status_ready.get('detail') or ''}".rstrip())
        recover_cmd = loop.get("recover_cmd") or []
        if loop.get("auto_recover") or recover_cmd:
            recover_ready = loop.get("recover_ready") or {}
            print(f"  recover: {'OK' if recover_ready.get('ok') else 'MISS'} {recover_ready.get('detail') or ''}".rstrip())
    if doc.get("commands"):
        print("commands:")
        for command in doc["commands"]:
            print(f"  {command}")


def _print_loop_check(doc: dict[str, Any]) -> None:
    print(
        f"loop-check: {doc.get('verdict')} loop={doc.get('loop')} "
        f"recover={doc.get('recover')} apply={doc.get('apply')}"
    )
    if doc.get("reason"):
        print(f"reason: {doc['reason']}")
    check = doc.get("check") or {}
    if check:
        if check.get("cmd"):
            print(f"status_cmd: {display_command(list(check.get('cmd') or []))}")
        detail = check.get("detail") or check.get("reason") or ""
        print(f"status: {check.get('state', 'UNKNOWN')} {detail}".rstrip())
        if check.get("returncode") is not None:
            print(f"returncode: {check.get('returncode')}")
    recovery = doc.get("recovery") or {}
    if recovery:
        if recovery.get("command"):
            print(f"recover_cmd: {recovery.get('command')}")
        state = "SKIP" if recovery.get("skipped") else ("OK" if recovery.get("ok") else "FAIL")
        detail = recovery.get("reason") or f"rc={recovery.get('returncode')}"
        print(f"recover: {state} {detail}")
    if doc.get("available"):
        print("available loops:")
        for loop in doc["available"]:
            print(f"  - {loop}")


def _print_loop_audit(doc: dict[str, Any]) -> None:
    counts = doc.get("counts") or {}
    print(
        f"loop-audit: {'OK' if doc.get('ok') else 'BROKEN'} "
        f"total={counts.get('total', 0)} healthy={counts.get('healthy', 0)} "
        f"action={counts.get('action', 0)} broken={counts.get('broken', 0)}"
    )
    label = {"healthy": "HEALTHY", "action": "ACTION ", "broken": "BROKEN "}
    for loop in doc.get("loops") or []:
        tag = label.get(loop.get("bucket"), loop.get("bucket", "?"))
        detail = loop.get("detail") or loop.get("state") or ""
        print(f"  {tag} {loop.get('name')}: {detail}".rstrip())
    if doc.get("missing"):
        print("requested but not enabled: " + ", ".join(doc["missing"]))


def _print_loop_scaffold(doc: dict[str, Any]) -> None:
    print(
        f"loop-scaffold: {'OK' if doc.get('ok') else 'REFUSE'} "
        f"apply={doc.get('apply')} loop={doc.get('loop') or '(none)'} "
        f"enabled={doc.get('enabled')}"
    )
    if doc.get("reason"):
        print(f"reason: {doc['reason']}")
    if doc.get("files"):
        print("files:")
        for file in doc["files"]:
            marker = "exists" if file.get("exists") else "new"
            print(f"  - {file.get('role')}: {file.get('path')} ({marker})")
    if doc.get("commands"):
        print("commands:")
        for command in doc["commands"]:
            print(f"  {command}")
    if doc.get("followup_commands"):
        print("next:")
        for command in doc["followup_commands"]:
            print(f"  {command}")


def _print_loop_set(doc: dict[str, Any]) -> None:
    print(
        f"loop-set: {'OK' if doc.get('ok') else 'REFUSE'} "
        f"apply={doc.get('apply')} changed={doc.get('changed')} "
        f"loop={doc.get('loop') or '(none)'} scope={doc.get('scope')}"
    )
    print(f"config: {doc.get('config_path') or doc.get('local_config')}")
    if doc.get("reason"):
        print(f"reason: {doc.get('reason')}")
    spec = doc.get("spec") or {}
    if spec:
        print(
            f"settings: enabled={spec.get('enabled', True) is not False} "
            f"auto_recover={bool(spec.get('auto_recover'))}"
        )
    if doc.get("commands"):
        print("commands:")
        for command in doc["commands"]:
            print(f"  {command}")
    if doc.get("followup_commands"):
        print("next:")
        for command in doc["followup_commands"]:
            print(f"  {command}")
    if doc.get("available") and not doc.get("ok"):
        print("available loops:")
        for loop in doc["available"]:
            print(f"  - {loop}")


# One printer per document schema. print_doc is a pure lookup in this table, so
# a new subcommand's human-readable form is a printer plus one row here instead
# of another branch on a 16-deep if/elif chain.
_DOC_PRINTERS = {
    SETUP_SCHEMA: _print_setup_plan,
    "fleet-control-pane.doctor/1": _print_doctor,
    TICK_SCHEMA: _print_tick,
    RECOVER_SCHEMA: _print_recover,
    SUPERVISOR_SCHEMA: _print_supervisor,
    FLEET_SCHEMA: _print_fleet,
    COMMIT_SCHEMA: _print_commit,
    SYNC_SCHEMA: _print_sync,
    PUBLISH_SCHEMA: _print_publish,
    BOOTSTRAP_SCHEMA: _print_bootstrap,
    LOOP_CONFIG_SCHEMA: _print_loop_config,
    LOOP_LIST_SCHEMA: _print_loop_list,
    LOOP_CHECK_SCHEMA: _print_loop_check,
    LOOP_AUDIT_SCHEMA: _print_loop_audit,
    LOOP_SCAFFOLD_SCHEMA: _print_loop_scaffold,
    LOOP_SET_SCHEMA: _print_loop_set,
}


def print_doc(doc: dict[str, Any], as_json: bool) -> None:
    """Print one plan/report document: JSON when asked, else the schema's own
    human-readable form, falling back to JSON for a schema without a printer."""
    printer = None if as_json else _DOC_PRINTERS.get(doc.get("schema"))
    if printer is None:
        print(json.dumps(doc, indent=2))
        return
    printer(doc)
