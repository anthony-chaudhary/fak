#!/usr/bin/env python3
"""Live collectors, watchdog/supervisor planners, control tick, bootstrap, setup plan, doctor, and fleet view."""
from __future__ import annotations

import collections
import os
import platform
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import appversion
from fleet_control_pane_core import (
    BOOTSTRAP_SCHEMA,
    FLEET_SCHEMA,
    RECOVER_SCHEMA,
    SCHEMA,
    SETUP_SCHEMA,
    SUPERVISOR_SCHEMA,
    TICK_SCHEMA,
    clean_config_overrides,
    command_exists,
    command_readiness,
    config_paths,
    display_command,
    display_path,
    ensure_runtime_dirs,
    env_for_tools,
    expand_cmd,
    init_command,
    init_config,
    iso_now,
    local_config_default_drift,
    local_config_mismatches,
    machine_id,
    pane_attr,
    platform_is_windows,
    powershell_exe,
    ps_quote,
    read_json,
    read_json_from_text,
    resolve_path,
    run,
    sh_exe,
)
from fleet_control_pane_git import (
    collect_git,
    git_mutation_guard,
    trunk_guard_status,
    worktree_doctor_command,
)
from fleet_control_pane_loops import (
    collect_loops,
    invoke_loop_recoveries,
    loop_scaffold_template_command,
    loop_setup_steps,
    recovery_invokes_control_bootstrap,
)
from fleet_control_pane_views import (
    blocked_account_recommended_action,
    machine_setup_action_count,
    machine_version_drift,
    scheduled_task_needs_action,
    scheduler_result_needs_action,
    scheduler_result_text,
    scheduler_state_needs_action,
    setup_plan_action_steps,
    summarize_machine_snapshot,
    summarize_registry,
    summarize_resume_plan,
    supervisor_diagnosis_is_actionable,
    supervisor_health_action_text,
    supervisor_health_needs_action,
    supervisor_health_summary,
    task_is_installed,
    write_status,
)


def refresh_registry(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["root"])
    script = root / "tools" / "fleet_sessions.py"
    if not script.exists():
        return {"ok": False, "reason": f"missing {display_path(script, root)}"}
    python = str(config.get("python") or sys.executable)
    if not pane_attr("command_exists", command_exists)(python):
        return {"ok": False, "reason": f"python not found: {python}"}
    args = [
        python,
        str(script),
        "registry",
        "--window",
        str(config.get("session_window_h", 6)),
    ]
    try:
        proc = pane_attr("run", run)(
            args,
            root,
            env=env_for_tools(config),
            timeout=int(config.get("refresh_timeout_s", 90)),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "fleet_sessions.py registry timed out"}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip()[-2000:],
        "stderr": proc.stderr.strip()[-2000:],
    }


def scheduled_task_status(task_name: str) -> dict[str, Any]:
    if not pane_attr("platform_is_windows", platform_is_windows)():
        return {"supported": False, "installed": False, "reason": "not Windows"}
    ps = pane_attr("powershell_exe", powershell_exe)()
    if not ps:
        return {"supported": False, "installed": False, "reason": "PowerShell not found"}
    q = ps_quote(task_name)
    script = (
        "$ErrorActionPreference='SilentlyContinue';"
        f"$t=Get-ScheduledTask -TaskName {q};"
        f"if($null -eq $t){{[pscustomobject]@{{task_name={q};installed=$false}}|ConvertTo-Json -Compress;exit 0}};"
        f"$i=Get-ScheduledTaskInfo -TaskName {q};"
        "[pscustomobject]@{"
        f"task_name={q};installed=$true;"
        "state=[string]$t.State;"
        "last_run=[string]$i.LastRunTime;"
        "last_result=[int64]$i.LastTaskResult;"
        "next_run=[string]$i.NextRunTime;"
        "arguments=[string](($t.Actions|Select-Object -First 1).Arguments)"
        "}|ConvertTo-Json -Compress"
    )
    try:
        proc = pane_attr("run", run)([ps, "-NoProfile", "-Command", script], Path.cwd(), timeout=15)
    except subprocess.TimeoutExpired:
        return {
            "supported": True,
            "installed": None,
            "reason": f"scheduled task query timed out for {task_name}",
        }
    except OSError as exc:
        return {
            "supported": True,
            "installed": None,
            "reason": str(exc),
        }
    if proc.returncode != 0:
        return {
            "supported": True,
            "installed": None,
            "reason": (proc.stderr or proc.stdout).strip(),
        }
    return {"supported": True, **read_json_from_text(proc.stdout)}


def control_tick_register_script_value(config: dict[str, Any]) -> Any:
    spec = config.get("control_tick") or {}
    if pane_attr("platform_is_windows", platform_is_windows)():
        return spec.get("register_script")
    return spec.get("posix_register_script") or spec.get("register_script")


def control_tick_runner_path(config: dict[str, Any]) -> Path:
    suffix = ".cmd" if pane_attr("platform_is_windows", platform_is_windows)() else ".sh"
    return Path(str(config["registry_dir"])) / f"control_pane_tick{suffix}"


def posix_tick_status(register_script: Path | None, config: dict[str, Any]) -> dict[str, Any]:
    if pane_attr("platform_is_windows", platform_is_windows)():
        return {"supported": False, "installed": False, "reason": "Windows host"}
    if not register_script or not register_script.exists():
        return {"supported": True, "installed": False, "reason": "register script missing"}
    cmd = [pane_attr("sh_exe", sh_exe)(), str(register_script), "status", "--json"]
    try:
        proc = pane_attr("run", run)(cmd, Path(config["root"]), env=env_for_tools(config), timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"supported": False, "installed": False, "reason": str(exc), "cmd": cmd}
    doc = read_json_from_text(proc.stdout)
    if doc:
        return {"supported": True, **doc, "returncode": proc.returncode, "cmd": cmd}
    return {
        "supported": True,
        "installed": None,
        "reason": (proc.stderr or proc.stdout).strip()[-1000:],
        "returncode": proc.returncode,
        "cmd": cmd,
    }


def collect_watchdogs(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["root"])
    out: dict[str, Any] = {}
    task_status_fn = pane_attr("scheduled_task_status", scheduled_task_status)
    for name, spec in sorted((config.get("watchdogs") or {}).items()):
        script = resolve_path(spec.get("script"), root)
        register_script = resolve_path(spec.get("register_script"), root)
        task_name = str(spec.get("task_name") or name)
        out[name] = {
            "task_name": task_name,
            "script": display_path(script, root),
            "script_exists": bool(script and script.exists()),
            "register_script": display_path(register_script, root),
            "register_script_exists": bool(register_script and register_script.exists()),
            "interval_min": spec.get("interval_min"),
            "task": task_status_fn(task_name),
        }
    return out


def collect_control_tick(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["root"])
    spec = config.get("control_tick") or {}
    register_script = resolve_path(control_tick_register_script_value(config), root)
    task_name = str(spec.get("task_name") or "FleetControlPaneTick")
    if pane_attr("platform_is_windows", platform_is_windows)():
        task = pane_attr("scheduled_task_status", scheduled_task_status)(task_name)
    else:
        task = pane_attr("posix_tick_status", posix_tick_status)(register_script, config)
    runner = resolve_path(task.get("runner"), root) if task.get("runner") else control_tick_runner_path(config)
    return {
        "task_name": task_name,
        "register_script": display_path(register_script, root),
        "register_script_exists": bool(register_script and register_script.exists()),
        "runner": display_path(runner, root),
        "runner_exists": bool(runner and runner.exists()),
        "runner_required": bool(task.get("runner") or task.get("arguments")),
        "interval_min": spec.get("interval_min"),
        "task": task,
    }


def control_tick_register_script_path(config: dict[str, Any], tick: dict[str, Any] | None = None) -> Path | None:
    root = Path(config["root"])
    if tick and tick.get("register_script"):
        return resolve_path(str(tick.get("register_script")), root)
    return resolve_path(control_tick_register_script_value(config), root)


def control_tick_runner_missing(tick: dict[str, Any], tick_task: dict[str, Any]) -> bool:
    if not task_is_installed(tick_task) or tick.get("runner_exists") is not False:
        return False
    return bool(tick.get("runner_required", True))


def collect_supervisor(config: dict[str, Any]) -> dict[str, Any]:
    raw_cmd = list(config.get("supervisor_status_cmd") or [])
    uses_job_dir = any("{job_dir}" in str(part) for part in raw_cmd)
    cmd = expand_cmd(raw_cmd, config)
    if not cmd:
        return {"available": False, "reason": "no supervisor_status_cmd configured"}
    if not pane_attr("command_exists", command_exists)(cmd[0]):
        return {"available": False, "reason": f"command not found: {cmd[0]}", "cmd": cmd}
    if uses_job_dir and not str(config.get("job_dir", "")):
        return {"available": False, "reason": "job_dir is not configured", "cmd": cmd}
    try:
        proc = pane_attr("run", run)(cmd, Path(config["root"]), timeout=20)
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": "supervisor status timed out", "cmd": cmd}
    doc = read_json_from_text(proc.stdout)
    if doc:
        return {"available": True, "cmd": cmd, "returncode": proc.returncode, "payload": doc}
    if proc.returncode != 0:
        return {
            "available": False,
            "returncode": proc.returncode,
            "reason": (proc.stderr or proc.stdout).strip()[-1000:],
            "cmd": cmd,
        }
    return {"available": False, "reason": "supervisor status was not JSON", "cmd": cmd}


def registry_path(config: dict[str, Any]) -> Path:
    return Path(config["registry_dir"]) / "sessions.json"


def collect_status(
    config: dict[str, Any],
    refresh: bool = False,
    *,
    skip_loop_names: set[str] | None = None,
) -> dict[str, Any]:
    refresh_result = pane_attr("refresh_registry", refresh_registry)(config) if refresh else None
    registry = read_json(registry_path(config))
    control_tick = pane_attr("collect_control_tick", collect_control_tick)(config)
    app_ver_mod = pane_attr("appversion", appversion)
    status: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": pane_attr("iso_now", iso_now)(),
        "app_version": app_ver_mod.app_version(Path(config["root"])),
        "machine": {
            "id": machine_id(config),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "config": {
            "root": config["root"],
            "local_config": config["_paths"]["local"],
            "local_config_exists": config.get("_local_exists", False),
            "registry_dir": config.get("registry_dir"),
            "machine_dir": config.get("machine_dir"),
            "user_home": config.get("user_home"),
            "job_dir": config.get("job_dir"),
            "target": config.get("target"),
            "session_window_h": config.get("session_window_h"),
        },
        "refresh": refresh_result,
        "git": pane_attr("collect_git", collect_git)(config),
        "registry": summarize_registry(registry),
        "control_tick": control_tick,
        "watchdogs": pane_attr("collect_watchdogs", collect_watchdogs)(config),
        "loops": pane_attr("collect_loops", collect_loops)(config, skip_names=skip_loop_names),
        "supervisor": pane_attr("collect_supervisor", collect_supervisor)(config),
        "setup_plan": pane_attr("setup_plan", setup_plan)(config, tick=control_tick),
    }
    status["actions"] = pane_attr("recommended_actions", recommended_actions)(status, config)
    status["verdict"] = "ACTION" if status["actions"] else "OK"
    return status


def recommended_actions(status: dict[str, Any], config: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    root = Path(config["root"])
    if not status["config"].get("local_config_exists"):
        actions.append("Run `python tools/fleet_control_pane.py init` once on this machine.")
    reg = status.get("registry", {})
    if not reg.get("exists"):
        actions.append("Run `python tools/fleet_control_pane.py status --refresh --write` to create the session registry.")
    elif reg.get("age_min") is not None and reg["age_min"] > 15:
        actions.append("Refresh stale session state with `python tools/fleet_control_pane.py status --refresh --write`.")
    account_action = blocked_account_recommended_action(reg)
    if account_action:
        actions.append(account_action)
    if reg.get("auto_resume", 0):
        actions.append(f"{reg['auto_resume']} autonomous session(s) are eligible for auto-resume; inspect `python tools/fleet_control_pane.py recover` or apply with `python tools/fleet_control_pane.py recover --apply`.")

    tick = status.get("control_tick") or {}
    tick_task = tick.get("task") or {}
    pane_tick_installed = task_is_installed(tick_task)
    if tick_task.get("supported") and tick_task.get("installed") is False:
        script = tick.get("register_script")
        if script:
            actions.append(f"Install the control-pane tick with `{display_command(register_command(root / script, 'control_tick', config))}`.")
    elif tick_task.get("supported") and tick_task.get("installed") and scheduler_state_needs_action(tick_task):
        actions.append(
            "Control-pane tick task is "
            f"{tick_task.get('state')}; repair it with `python tools/fleet_control_pane.py bootstrap --apply`."
        )
    elif tick_task.get("supported") and tick_task.get("installed") and scheduler_result_needs_action(tick_task):
        actions.append(
            "Control-pane tick last scheduler result was "
            f"{scheduler_result_text(tick_task)}; check it with "
            "`python tools/fleet_control_pane.py tick --dry-run --no-write` and `python tools/fleet_control_pane.py doctor`."
        )

    for name, wd in (status.get("watchdogs") or {}).items():
        if not wd.get("script_exists"):
            actions.append(f"Missing watchdog script for {name}: {wd.get('script')}")
        task = wd.get("task") or {}
        if task.get("supported") and task.get("installed") is False:
            if not pane_tick_installed:
                reg_script = wd.get("register_script")
                if reg_script:
                    actions.append(f"Install {name} watchdog with `{display_command(register_command(root / reg_script, name, config))}`.")
        elif task.get("supported") and task.get("installed") and scheduled_task_needs_action(task):
            if not pane_tick_installed:
                if scheduler_state_needs_action(task):
                    actions.append(f"{name} watchdog task is {task.get('state')}; inspect its registered command.")
                else:
                    actions.append(f"{name} watchdog last scheduler result was {scheduler_result_text(task)}; inspect its registered command.")

    loops_doc = status.get("loops") or {}
    if loops_doc and "enabled" in loops_doc and int(loops_doc.get("enabled") or 0) == 0:
        commands = loops_doc.get("commands") or [loop_scaffold_template_command(config)]
        actions.append(
            "No enabled loops configured; start a shared loop with "
            f"`{commands[0]}` or inspect `python tools/fleet_control_pane.py loop-list`."
        )

    for check in loops_doc.get("checks") or []:
        state = str(check.get("state") or "UNKNOWN")
        if state in {"OK", "SKIPPED"}:
            continue
        name = check.get("name")
        action = check.get("action") or check.get("detail") or check.get("reason") or "inspect loop status"
        spec = (config.get("loops") or {}).get(name) if name is not None else None
        recover_cmd = expand_cmd(list((spec or {}).get("recover_cmd") or []), config) if isinstance(spec, dict) else []
        if recovery_invokes_control_bootstrap(recover_cmd):
            actions.append(f"Loop {name} is {state}; run `python tools/fleet_control_pane.py bootstrap --apply`. {action}")
        elif check.get("auto_recover") and check.get("has_recover_cmd"):
            actions.append(f"Loop {name} is {state}; it will be recovered by the next live control tick. {action}")
        else:
            actions.append(f"Loop {name} is {state}: {action}.")

    supervisor = status.get("supervisor") or {}
    if not supervisor.get("available"):
        reason = supervisor.get("reason", "unavailable")
        actions.append(f"Supervisor status unavailable: {reason}.")
    else:
        supervisor_task = ((status.get("watchdogs") or {}).get("supervisor") or {}).get("task") or {}
        summary = supervisor_health_summary(supervisor, config, watchdog_task=supervisor_task)
        if supervisor_health_needs_action(summary):
            actions.append(supervisor_health_action_text(summary))

    git_status = status.get("git") or {}
    safe_ff = git_status.get("safe_ff") or {}
    counts = git_status.get("counts") or {}
    unmerged_count = int(counts.get("unmerged") or 0)
    unmerged_paths = [str(path) for path in git_status.get("unmerged_paths") or []]
    merge_in_progress = bool(git_status.get("merge_in_progress"))
    if unmerged_count:
        shown = ", ".join(unmerged_paths[:4])
        omitted = max(0, len(unmerged_paths) - 4)
        suffix = f" (+{omitted} more)" if omitted else ""
        detail = f": {shown}{suffix}" if shown else ""
        actions.append(f"Resolve {unmerged_count} unmerged path(s) before commit/publish{detail}.")
    elif merge_in_progress:
        actions.append("Finish or abort the in-progress merge before commit/publish.")
    if safe_ff.get("state") == "ahead":
        if unmerged_count or merge_in_progress or int(git_status.get("dirty_total") or 0) > 0:
            actions.append("Branch is ahead of remote, but publish is blocked until the worktree is clean.")
        else:
            actions.append("Branch is ahead of remote; inspect `python tools/fleet_control_pane.py publish --json` before pushing.")
    elif safe_ff.get("state") == "diverged" or safe_ff.get("ok") is False:
        actions.append("Do not use plain `git pull`; inspect `python tools/fleet_control_pane.py sync --fetch --json`.")
    worktrees = git_status.get("worktrees") or {}
    if worktrees.get("available"):
        commands = worktrees.get("commands") or {}
        prune_count = int(worktrees.get("prune_count") or 0)
        if prune_count > 0:
            actions.append(
                f"{prune_count} safe extra worktree(s) can be pruned; "
                f"run `{commands.get('prune') or display_command(worktree_doctor_command(config, prune=True, fetch=True))}`."
            )
        if int(worktrees.get("blocked_count") or 0) > 0 or worktrees.get("primary_offtrack"):
            actions.append(
                "Worktree doctor is not converged; inspect blocked/off-track worktree state with "
                f"`{commands.get('inspect') or display_command(worktree_doctor_command(config, prune=False, fetch=True))}`."
            )
    if git_status.get("dirty_total", 0) > 0:
        if unmerged_count:
            actions.append(
                f"Worktree has {git_status['dirty_total']} dirty path(s), including merge conflicts; "
                "resolve conflicts before using dirty-plan commit commands."
            )
        else:
            actions.append(f"Worktree has {git_status['dirty_total']} dirty path(s); use the dirty-plan commit commands before broad sync/release.")
    return actions


def supervisor_needs_action(supervisor: dict[str, Any]) -> bool:
    if not supervisor.get("available"):
        return True
    payload = supervisor.get("payload") or {}
    alive = (payload.get("process") or {}).get("alive")
    if alive is False:
        return True
    return supervisor_diagnosis_is_actionable(supervisor)


def register_command(script: Path, name: str, config: dict[str, Any], *, live_resume: bool = False) -> list[str]:
    if name == "control_tick" and not pane_attr("platform_is_windows", platform_is_windows)():
        tick = config.get("control_tick") or {}
        args = [
            pane_attr("sh_exe", sh_exe)(),
            str(script),
            "install",
            "--task-name",
            str(tick.get("task_name") or "FleetControlPaneTick"),
            "--python",
            str(config.get("python") or sys.executable),
        ]
        if tick.get("interval_min"):
            args.extend(["--interval-min", str(tick.get("interval_min"))])
        if live_resume:
            args.append("--live-resume")
        return args

    ps = "powershell.exe" if pane_attr("platform_is_windows", platform_is_windows)() else "powershell"
    args = [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    if name == "supervisor":
        args.extend(["-Target", str(config.get("target", 4))])
    elif name == "control_tick":
        args.extend(["-Python", str(config.get("python") or sys.executable)])
        tick = config.get("control_tick") or {}
        if tick.get("interval_min"):
            args.extend(["-IntervalMin", str(tick.get("interval_min"))])
        if live_resume:
            args.append("-LiveResume")
    return args


def watchdog_command(name: str, config: dict[str, Any], live_resume: bool = False) -> list[str] | None:
    if not pane_attr("platform_is_windows", platform_is_windows)():
        return None
    ps = pane_attr("powershell_exe", powershell_exe)()
    if not ps:
        return None
    spec = (config.get("watchdogs") or {}).get(name) or {}
    script = resolve_path(spec.get("script"), Path(config["root"]))
    if not script:
        return None
    args = [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    if name == "supervisor":
        if config.get("job_dir"):
            args.extend(["-JobDir", str(config["job_dir"])])
        args.extend(["-Target", str(config.get("target", 4))])
        if config.get("watchdog_log_dir"):
            args.extend(["-LogDir", str(config["watchdog_log_dir"])])
    elif name == "resume":
        args.extend(["-FleetDir", str(config["root"])])
        args.extend(["-WindowH", str(config.get("session_window_h", 6))])
        if config.get("registry_dir"):
            args.extend(["-RegistryDir", str(config["registry_dir"])])
        if config.get("claude_exe"):
            args.extend(["-ClaudeExe", str(config["claude_exe"])])
        if config.get("watchdog_log_dir"):
            args.extend(["-LogDir", str(config["watchdog_log_dir"])])
        if live_resume:
            args.append("-Live")
    return args


def invoke_watchdog(name: str, config: dict[str, Any], *, dry_run: bool, live_resume: bool = False) -> dict[str, Any]:
    cmd = pane_attr("watchdog_command", watchdog_command)(name, config, live_resume=live_resume)
    root = Path(config["root"])
    spec = (config.get("watchdogs") or {}).get(name) or {}
    script = resolve_path(spec.get("script"), root)
    result: dict[str, Any] = {
        "name": name,
        "dry_run": dry_run,
        "script": display_path(script, root),
        "script_exists": bool(script and script.exists()),
        "command": display_command(cmd or []),
    }
    if not pane_attr("platform_is_windows", platform_is_windows)():
        return {**result, "ok": None, "skipped": True, "reason": "PowerShell watchdog scripts are Windows-only"}
    if not cmd:
        return {**result, "ok": False, "skipped": True, "reason": "watchdog command unavailable"}
    if script and not script.exists():
        return {**result, "ok": False, "skipped": True, "reason": "watchdog script missing"}
    if dry_run:
        return {**result, "ok": True, "skipped": True, "reason": "dry run"}
    try:
        # via_files: the watchdog may relaunch a long-lived detached daemon
        # (run_supervise_loop / claude) that inherits our stdout pipe; capturing through
        # a pipe would then deadlock the tick despite `timeout` (see run()).
        proc = pane_attr("run", run)(cmd, root, env=env_for_tools(config), timeout=90, via_files=True)
    except subprocess.TimeoutExpired:
        return {**result, "ok": False, "skipped": False, "reason": "watchdog timed out"}
    return {
        **result,
        "ok": proc.returncode in (0, 10),
        "skipped": False,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip()[-2000:],
        "stderr": proc.stderr.strip()[-2000:],
    }


def terminate_pid(pid: int, root: Path) -> dict[str, Any]:
    if pane_attr("platform_is_windows", platform_is_windows)():
        proc = pane_attr("run", run)(["taskkill", "/PID", str(pid), "/F"], root, timeout=20)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip()[-2000:],
            "stderr": proc.stderr.strip()[-2000:],
        }
    try:
        os.kill(pid, signal.SIGTERM)
        return {"ok": True}
    except OSError as exc:
        return {"ok": False, "reason": str(exc)}


def supervisor_plan(config: dict[str, Any], *, restart: bool = False, apply: bool = False) -> dict[str, Any]:
    root = Path(config["root"])
    sup_fn = pane_attr("collect_supervisor", collect_supervisor)
    before = sup_fn(config)
    actions: list[dict[str, Any]] = []
    payload = before.get("payload") or {}
    proc_info = payload.get("process") or {}
    pid = proc_info.get("pid")
    alive = proc_info.get("alive")
    is_win = pane_attr("platform_is_windows", platform_is_windows)()
    if restart:
        if pid and alive is not False:
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                pid_int = 0
            command = ["taskkill", "/PID", str(pid_int), "/F"] if is_win else ["kill", "-TERM", str(pid_int)]
            action = {
                "name": "terminate-supervisor",
                "pid": pid_int,
                "dry_run": not apply,
                "command": display_command(command),
            }
            if not pid_int:
                actions.append({**action, "ok": False, "skipped": True, "reason": f"invalid supervisor pid: {pid}"})
            elif not apply:
                actions.append({**action, "ok": True, "skipped": True, "reason": "dry run"})
            else:
                result = pane_attr("terminate_pid", terminate_pid)(pid_int, root)
                actions.append({**action, **result, "skipped": False})
        else:
            actions.append({
                "name": "terminate-supervisor",
                "dry_run": not apply,
                "ok": True,
                "skipped": True,
                "reason": "supervisor is not alive; no terminate step needed",
            })
        actions.append(pane_attr("invoke_watchdog", invoke_watchdog)("supervisor", config, dry_run=not apply))
    after = sup_fn(config) if restart and apply else before
    ok = before.get("available", False) if not restart else all(action.get("ok") is not False for action in actions)
    return {
        "schema": SUPERVISOR_SCHEMA,
        "generated_utc": pane_attr("iso_now", iso_now)(),
        "restart": restart,
        "apply": apply,
        "ok": ok,
        "needs_action": supervisor_needs_action(after),
        "before": before,
        "after": after,
        "actions": actions,
    }


def control_tick(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    live_resume: bool = False,
    skip_supervisor: bool = False,
    skip_resume: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    mutation_guard = pane_attr("git_mutation_guard", git_mutation_guard)(Path(config["root"]))
    git_mutation_blocked = bool(mutation_guard.get("blocked"))
    if git_mutation_blocked:
        actions.append({
            "name": "git-merge-guard",
            "ok": True,
            "skipped": True,
            "reason": (
                "merge/unmerged worktree detected; skipping watchdog and loop recovery actions"
            ),
            "blockers": mutation_guard.get("blockers") or [],
            "worktree": mutation_guard.get("worktree") or {},
        })
    invoke_wd = pane_attr("invoke_watchdog", invoke_watchdog)
    if not git_mutation_blocked and not skip_supervisor:
        actions.append(invoke_wd("supervisor", config, dry_run=dry_run))
    if not git_mutation_blocked and not skip_resume:
        actions.append(invoke_wd("resume", config, dry_run=dry_run, live_resume=live_resume))

    tick_skip_loops = {"control-pane-doctor"}
    status_fn = pane_attr("collect_status", collect_status)
    status = status_fn(config, refresh=True, skip_loop_names=tick_skip_loops)
    loop_actions = (
        []
        if git_mutation_blocked
        else pane_attr("invoke_loop_recoveries", invoke_loop_recoveries)(config, status.get("loops") or {}, dry_run=dry_run)
    )
    actions.extend(loop_actions)
    if loop_actions and not dry_run and any(not action.get("skipped") for action in loop_actions):
        status = status_fn(config, refresh=False, skip_loop_names=tick_skip_loops)
    written = pane_attr("write_status", write_status)(status, config) if write else None
    ok = all(a.get("ok") is not False for a in actions)
    return {
        "schema": TICK_SCHEMA,
        "generated_utc": pane_attr("iso_now", iso_now)(),
        "dry_run": dry_run,
        "live_resume": live_resume,
        "ok": ok,
        "mutation_guard": mutation_guard,
        "actions": actions,
        "status": status,
        "written": written,
    }


def recover_plan(
    config: dict[str, Any],
    *,
    apply: bool = False,
    skip_supervisor: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    tick = pane_attr("control_tick", control_tick)(
        config,
        dry_run=not apply,
        live_resume=True,
        skip_supervisor=skip_supervisor,
        skip_resume=False,
        write=write,
    )
    resume_plan = summarize_resume_plan(config)
    return {
        "schema": RECOVER_SCHEMA,
        "generated_utc": pane_attr("iso_now", iso_now)(),
        "apply": apply,
        "ok": tick.get("ok"),
        "resume_plan": resume_plan,
        "tick": tick,
        "actions": tick.get("actions") or [],
        "status": tick.get("status"),
        "written": tick.get("written"),
    }


def _bootstrap_local_config_action(
    config: dict[str, Any],
    root: Path,
    apply: bool,
    init_overrides: dict[str, Any] | None,
    force_local_config: bool,
) -> dict[str, Any]:
    local_path = Path(config["_paths"]["local"])
    mismatches = local_config_mismatches(local_path, init_overrides)
    has_init_overrides = bool(clean_config_overrides(init_overrides))
    init_cfg_fn = pane_attr("init_config", init_config)
    if local_path.exists() and force_local_config and apply and has_init_overrides:
        result = init_cfg_fn(root, force=True, overrides=init_overrides, base_config=config)
        return {
            "id": "local-config",
            "ok": bool(result.get("written")),
            "changed": bool(result.get("written")),
            "skipped": False,
            "detail": result.get("path"),
        }
    if local_path.exists() and mismatches:
        return {
            "id": "local-config",
            "ok": False,
            "changed": False,
            "skipped": True,
            "detail": "local config already exists with different requested value(s); pass --force-local-config to rewrite",
            "mismatches": mismatches,
            "command": display_command(init_command(config, init_overrides, force=True)),
        }
    if local_path.exists():
        return {
            "id": "local-config",
            "ok": True,
            "changed": False,
            "skipped": True,
            "detail": str(local_path),
        }
    if apply:
        result = init_cfg_fn(root, overrides=init_overrides, base_config=config)
        return {
            "id": "local-config",
            "ok": bool(result.get("written")),
            "changed": bool(result.get("written")),
            "skipped": False,
            "detail": result.get("path"),
        }
    return {
        "id": "local-config",
        "ok": True,
        "changed": False,
        "skipped": True,
        "detail": f"would write {local_path}",
        "command": display_command(init_command(config, init_overrides)),
    }


def _bootstrap_runtime_dirs_action(config: dict[str, Any], apply: bool) -> dict[str, Any]:
    dirs = ensure_runtime_dirs(config, apply=apply)
    missing = [check for check in dirs["checks"] if not check.get("exists")]
    failed_existing = [check for check in dirs["checks"] if not check.get("ok") and check.get("exists")]
    if apply:
        return {
            "id": "runtime-dirs",
            "ok": dirs["ok"],
            "changed": dirs["changed"],
            "skipped": False,
            "detail": "runtime directories ready" if dirs["ok"] else "runtime directory check failed",
            "checks": dirs["checks"],
        }
    if failed_existing:
        return {
            "id": "runtime-dirs",
            "ok": False,
            "changed": False,
            "skipped": True,
            "detail": "runtime directories are not writable",
            "checks": dirs["checks"],
        }
    if missing:
        return {
            "id": "runtime-dirs",
            "ok": True,
            "changed": False,
            "skipped": True,
            "detail": "would create runtime directories",
            "checks": dirs["checks"],
        }
    return {
        "id": "runtime-dirs",
        "ok": True,
        "changed": False,
        "skipped": True,
        "detail": "runtime directories ready",
        "checks": [],
    }


def _bootstrap_trunk_guard_action(config: dict[str, Any], root: Path, apply: bool) -> dict[str, Any] | None:
    guard = pane_attr("trunk_guard_status", trunk_guard_status)(config)
    if not (guard.get("supported") or guard.get("installer_exists") or guard.get("hook_exists")):
        return None
    if guard.get("installed"):
        return {
            "id": "trunk-guard",
            "ok": True,
            "changed": False,
            "skipped": True,
            "detail": guard.get("detail"),
        }
    if not guard.get("setup_ok"):
        return {
            "id": "trunk-guard",
            "ok": False,
            "changed": False,
            "skipped": True,
            "detail": guard.get("detail"),
        }
    cmd = list(guard.get("command") or [])
    if not apply:
        return {
            "id": "trunk-guard",
            "ok": True,
            "changed": False,
            "skipped": True,
            "detail": "would install local trunk guard",
            "command": display_command(cmd),
        }
    proc = (
        pane_attr("run", run)(cmd, root, env=env_for_tools(config), timeout=30)
        if cmd
        else subprocess.CompletedProcess(cmd, 1, "", "installer command missing")
    )
    return {
        "id": "trunk-guard",
        "ok": proc.returncode == 0,
        "changed": proc.returncode == 0,
        "skipped": False,
        "detail": proc.stdout.strip()[-2000:] or proc.stderr.strip()[-2000:],
        "returncode": proc.returncode,
        "command": display_command(cmd),
    }


def _bootstrap_control_tick_action(
    config: dict[str, Any],
    root: Path,
    apply: bool,
    live_resume: bool,
) -> dict[str, Any]:
    tick = pane_attr("collect_control_tick", collect_control_tick)(config)
    tick_task = tick.get("task") or {}
    runner_missing = control_tick_runner_missing(tick, tick_task)
    tick_disabled = scheduler_state_needs_action(tick_task)
    if tick_task.get("supported") and tick_task.get("installed") and not runner_missing and not tick_disabled:
        return {
            "id": "control-tick",
            "ok": True,
            "changed": False,
            "skipped": True,
            "detail": f"{tick.get('task_name')} already installed",
        }
    register_script = control_tick_register_script_path(config, tick)
    cmd = register_command(register_script, "control_tick", config, live_resume=live_resume) if register_script else []
    if not register_script or not register_script.exists():
        return {
            "id": "control-tick",
            "ok": False,
            "changed": False,
            "skipped": True,
            "detail": "control-pane tick register script not found",
        }
    if not apply:
        detail = (
            f"would repair missing control-pane tick runner {tick.get('runner')}"
            if runner_missing
            else f"would re-enable control-pane tick task {tick.get('task_name')}"
            if tick_disabled
            else "would install control-pane tick"
        )
        return {
            "id": "control-tick",
            "ok": True,
            "changed": False,
            "skipped": True,
            "detail": detail,
            "command": display_command(cmd),
        }
    proc = pane_attr("run", run)(cmd, root, env=env_for_tools(config), timeout=90)
    return {
        "id": "control-tick",
        "ok": proc.returncode == 0,
        "changed": proc.returncode == 0,
        "skipped": False,
        "detail": proc.stdout.strip()[-2000:] or proc.stderr.strip()[-2000:],
        "returncode": proc.returncode,
        "command": display_command(cmd),
    }


def bootstrap(
    config: dict[str, Any],
    *,
    apply: bool = False,
    live_resume: bool = False,
    init_overrides: dict[str, Any] | None = None,
    force_local_config: bool = False,
) -> dict[str, Any]:
    root = Path(config["root"])
    actions: list[dict[str, Any]] = [
        _bootstrap_local_config_action(config, root, apply, init_overrides, force_local_config),
        _bootstrap_runtime_dirs_action(config, apply),
    ]
    trunk_action = _bootstrap_trunk_guard_action(config, root, apply)
    if trunk_action is not None:
        actions.append(trunk_action)
    actions.append(_bootstrap_control_tick_action(config, root, apply, live_resume))
    ok = all(action.get("ok") for action in actions)
    return {
        "schema": BOOTSTRAP_SCHEMA,
        "generated_utc": pane_attr("iso_now", iso_now)(),
        "apply": apply,
        "live_resume": live_resume,
        "ok": ok,
        "actions": actions,
    }


def setup_plan(config: dict[str, Any], *, tick: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(config["root"])
    python = str(config.get("python") or sys.executable)
    local_path = Path((config.get("_paths") or {}).get("local") or config_paths(root)["local"])
    runtime_dirs = ensure_runtime_dirs(config, apply=False)
    missing_dirs = [check["name"] for check in runtime_dirs["checks"] if not check.get("exists")]
    failed_dirs = [check for check in runtime_dirs["checks"] if not check.get("ok") and check.get("exists")]
    tick = tick or pane_attr("collect_control_tick", collect_control_tick)(config)
    tick_task = tick.get("task") or {}
    bootstrap_cmd = [python, "tools/fleet_control_pane.py", "bootstrap", "--apply"]
    config_drift = local_config_default_drift(config)
    steps = [
        {
            "id": "local-config",
            "state": "done" if local_path.exists() else "todo",
            "ok": True,
            "detail": str(local_path) if local_path.exists() else f"missing {local_path}",
            "why": "Persist host-specific paths under the ignored registry directory.",
            "command": [python, "tools/fleet_control_pane.py", "init"],
        },
    ]
    if config_drift.get("count"):
        steps.append({
            "id": "local-config-tracked-defaults",
            "state": "todo",
            "ok": False,
            "detail": str(config_drift.get("detail") or "local config shadows tracked defaults"),
            "why": "Refresh shared setup defaults in the ignored local config without hand-editing JSON.",
            "command": list(config_drift.get("command") or []),
        })
    guard = pane_attr("trunk_guard_status", trunk_guard_status)(config)
    steps.extend([
        {
            "id": "runtime-dirs",
            "state": "blocked" if failed_dirs else ("todo" if missing_dirs else "done"),
            "ok": not failed_dirs,
            "detail": (
                "not writable: " + ", ".join(f"{c['name']}={c.get('reason')}" for c in failed_dirs)
                if failed_dirs
                else ("missing: " + ", ".join(missing_dirs) if missing_dirs else "registry, machine, and watchdog directories are ready")
            ),
            "why": "Ensure registry, machine snapshot, and watchdog log directories exist and are writable.",
            "command": bootstrap_cmd,
        },
        {
            "id": "control-tick",
            "state": "manual",
            "ok": True,
            "detail": "safe to run any time; scheduled task covers the recurring cadence when installed",
            "why": "Run one keep-alive tick now: refresh sessions, invoke watchdogs, and write the pane.",
            "command": [
                python,
                "tools/fleet_control_pane.py",
                "tick",
            ],
        },
    ])
    if guard.get("supported") or guard.get("installer_exists") or guard.get("hook_exists"):
        steps.append({
            "id": "install-trunk-guard",
            "state": guard.get("state"),
            "ok": bool(guard.get("setup_ok")),
            "detail": guard.get("detail"),
            "why": "Install the repo-local git hook that blocks accidental off-master branch creation below the agent layer.",
            "command": list(guard.get("command") or []),
        })
    script = control_tick_register_script_path(config, tick)
    install_state = "todo"
    install_ok = True
    install_detail = "install the recurring control-pane tick"
    install_command = register_command(script, "control_tick", config) if script else []
    runner_missing = control_tick_runner_missing(tick, tick_task)
    tick_disabled = scheduler_state_needs_action(tick_task)
    if task_is_installed(tick_task) and not runner_missing and not tick_disabled:
        install_state = "done"
        install_detail = f"{tick.get('task_name')} already installed"
    elif runner_missing:
        install_state = "todo"
        install_detail = f"control-pane tick runner missing: {tick.get('runner')}"
    elif tick_disabled:
        install_state = "todo"
        install_detail = f"control-pane tick task is {tick_task.get('state')}"
    elif not script or not script.exists():
        install_state = "blocked"
        install_ok = False
        install_detail = "control-pane tick register script not found"
        install_command = []
    elif tick_task.get("supported") is False:
        install_state = "blocked"
        install_ok = False
        install_detail = f"scheduler unsupported: {tick_task.get('reason') or 'unknown'}"
    if script:
        steps.append({
            "id": "install-control-pane-tick",
            "state": install_state,
            "ok": install_ok,
            "detail": install_detail,
            "why": "Let the OS run one pane tick on a cadence instead of installing per-loop tasks by hand.",
            "command": install_command,
        })
    steps.extend(loop_setup_steps(config))
    return {
        "schema": SETUP_SCHEMA,
        "generated_utc": pane_attr("iso_now", iso_now)(),
        "machine": {"host": socket.gethostname(), "platform": platform.platform()},
        "steps": [
            {**step, "display": display_command(step["command"])}
            for step in steps
        ],
    }


def doctor(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["root"])
    checks: list[dict[str, Any]] = []
    cmd_exists_fn = pane_attr("command_exists", command_exists)

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("repo-root", root.exists(), str(root))
    add("local-config", bool(config.get("_local_exists")), config["_paths"]["local"])
    config_drift = local_config_default_drift(config)
    if config_drift.get("count"):
        add("local-config-tracked-defaults", False, str(config_drift.get("detail") or "local config shadows tracked defaults"))
    add("python", cmd_exists_fn(str(config.get("python"))), str(config.get("python")))
    add("user-home", Path(str(config.get("user_home", ""))).exists(), str(config.get("user_home")))
    job_dir = Path(str(config.get("job_dir", ""))) if config.get("job_dir") else None
    add("job-dir", bool(job_dir and job_dir.exists()), str(job_dir or "not configured"))
    claude = str(config.get("claude_exe", ""))
    add("claude-exe", bool(claude and cmd_exists_fn(claude)), claude or "not configured")
    for check in ensure_runtime_dirs(config, apply=False)["checks"]:
        add(f"{check['name']}-writable", bool(check.get("ok")), str(check.get("path")) if check.get("ok") else f"{check.get('path')}: {check.get('reason')}")
    for rel in (
        "tools/fleet_sessions.py",
        "tools/fleet_accounts.py",
        "tools/safe_ff_sync.py",
        "tools/fleet_control_pane.py",
        "tools/register_control_pane_tick.ps1",
        "tools/register_control_pane_tick.sh",
    ):
        add(rel, (root / rel).exists(), rel)
    guard = pane_attr("trunk_guard_status", trunk_guard_status)(config)
    if guard.get("supported") or guard.get("installer_exists") or guard.get("hook_exists"):
        add("trunk-guard", bool(guard.get("ok")), str(guard.get("detail") or "inspect trunk guard"))
    tick = pane_attr("collect_control_tick", collect_control_tick)(config)
    add("control-tick-register-script", bool(tick.get("register_script_exists")), str(tick.get("register_script")))
    tick_task = tick.get("task") or {}
    pane_tick_installed = task_is_installed(tick_task)
    if tick_task.get("supported"):
        add("control-tick-scheduled-task", bool(tick_task.get("installed")), str(tick_task.get("task_name")))
    if pane_tick_installed and "runner_exists" in tick:
        add("control-tick-runner", bool(tick.get("runner_exists")), str(tick.get("runner")))
    for name, wd in (pane_attr("collect_watchdogs", collect_watchdogs)(config) or {}).items():
        add(f"{name}-watchdog-script", bool(wd.get("script_exists")), str(wd.get("script")))
        task = wd.get("task") or {}
        if task.get("supported"):
            if pane_tick_installed:
                state = "installed" if task.get("installed") else "not installed"
                reason = task.get("reason")
                detail = f"covered by control-pane tick {tick.get('task_name')}; standalone {state}"
                if reason:
                    detail = f"{detail}: {reason}"
                add(f"{name}-scheduled-task", True, detail)
            else:
                add(f"{name}-scheduled-task", bool(task.get("installed")), str(task.get("task_name")))
    for name, spec in sorted((config.get("loops") or {}).items()):
        if not isinstance(spec, dict) or spec.get("enabled") is False:
            continue
        status_cmd = expand_cmd(list(spec.get("status_cmd") or []), config)
        recover_cmd = expand_cmd(list(spec.get("recover_cmd") or []), config)
        status_ready = command_readiness(status_cmd, root)
        add(f"{name}-loop-status-cmd", bool(status_ready["ok"]), str(status_ready["detail"]))
        if recover_cmd:
            recover_ready = command_readiness(recover_cmd, root)
            add(f"{name}-loop-recover-cmd", bool(recover_ready["ok"]), str(recover_ready["detail"]))

    ok = all(c["ok"] for c in checks if c["name"] not in {"local-config", "job-dir", "claude-exe"})
    return {
        "schema": "fleet-control-pane.doctor/1",
        "generated_utc": pane_attr("iso_now", iso_now)(),
        "ok": ok,
        "checks": checks,
        "setup_plan": pane_attr("setup_plan", setup_plan)(config),
    }


def fleet_view(
    config: dict[str, Any],
    *,
    include_live_local: bool = False,
    refresh_live_local: bool = False,
    write_live_local: bool = False,
) -> dict[str, Any]:
    machine_dir = Path(config["machine_dir"])
    stale_min = float(config.get("machine_stale_min", 15))
    summarize_fn = pane_attr("summarize_machine_snapshot", summarize_machine_snapshot)
    machines = [
        summarize_fn(path, read_json(path), stale_min)
        for path in sorted(machine_dir.glob("*.json"))
    ] if machine_dir.exists() else []
    local_written = None
    if include_live_local:
        local_path = machine_dir / f"{machine_id(config)}.json"
        local_status = pane_attr("collect_status", collect_status)(config, refresh=refresh_live_local)
        if write_live_local:
            local_written = pane_attr("write_status", write_status)(local_status, config)
        local_machine = summarize_fn(local_path, local_status, stale_min)
        local_machine["live_local"] = True
        local_machine["refresh_local"] = refresh_live_local
        local_id = local_machine.get("id")
        machines = [m for m in machines if m.get("id") != local_id]
        machines.append(local_machine)
        machines.sort(key=lambda m: str(m.get("id") or ""))
    app_ver_mod = pane_attr("appversion", appversion)
    current_version = app_ver_mod.app_version(Path(config["root"])) if config.get("root") else ""
    for machine in machines:
        drift = machine_version_drift(machine, current_version)
        if drift:
            machine["version_drift"] = drift
    states = collections.Counter(m.get("state", "UNKNOWN") for m in machines)
    versions = collections.Counter(str(m.get("app_version") or "unknown") for m in machines)
    version_mismatches = sum(1 for m in machines if m.get("version_drift"))
    totals = {
        "sessions": sum(int(m.get("sessions") or 0) for m in machines),
        "actions": sum(int(m.get("actions_count") or 0) for m in machines),
        "auth_blocked": sum(int(m.get("auth_blocked") or 0) for m in machines),
        "blocked_accounts": sum(len(((m.get("accounts") or {}).get("blocked") or [])) for m in machines),
        "auth_blocked_accounts": sum(int(((m.get("accounts") or {}).get("auth_blocked_count") or 0)) for m in machines),
        "auto_resume": sum(int(m.get("auto_resume") or 0) for m in machines),
        "surface": sum(int(m.get("surface") or 0) for m in machines),
        "dirty_paths": sum(int((m.get("git") or {}).get("dirty_total") or 0) for m in machines),
        "worktree_prune": sum(int(((m.get("git") or {}).get("worktrees") or {}).get("prune_count") or 0) for m in machines),
        "worktree_blocked": sum(int(((m.get("git") or {}).get("worktrees") or {}).get("blocked_count") or 0) for m in machines),
        "setup_actions": sum(machine_setup_action_count(m) for m in machines),
        "setup_plan_actions": sum(len(setup_plan_action_steps(m)) for m in machines),
        "loop_actions": sum(int((m.get("loops") or {}).get("action") or 0) for m in machines),
        "version_mismatches": version_mismatches,
    }
    verdict = "OK"
    if not machines:
        verdict = "EMPTY"
    elif version_mismatches or any(m.get("state") in {"ACTION", "STALE", "UNKNOWN", "INVALID"} for m in machines):
        verdict = "ACTION"
    return {
        "schema": FLEET_SCHEMA,
        "generated_utc": pane_attr("iso_now", iso_now)(),
        "machine_dir": str(machine_dir),
        "stale_min": stale_min,
        "include_live_local": include_live_local,
        "refresh_live_local": refresh_live_local,
        "current_version": current_version,
        "written": local_written,
        "verdict": verdict,
        "states": dict(states),
        "versions": dict(versions),
        "totals": totals,
        "machines": machines,
    }
