#!/usr/bin/env python3
"""One control pane for the fleet's local agent-loop runtime.

The existing fleet recovery layer already has the hard parts: transcript-derived
session state, account availability, watchdog scripts, and a safe git sync check.
This module composes those pieces behind one host-portable command so each
machine can answer the same questions without hand-editing local paths.

Commands:
  init          write tools/_registry/control_pane.local.json for this machine
  status        print the pane; optionally refresh sessions and write snapshots
  tick          one recurring keep-alive tick; schedule this on each machine
  recover       dry-run/apply one live recovery pass for autonomous loops
  supervisor    inspect or restart the standing supervisor
  fleet         aggregate every published machine snapshot this pane can see
  sync          inspect/apply safe fast-forward sync without plain git pull
  publish       push this branch only when it is safely ahead of the remote
  commit        stage and commit an explicit path set, refusing foreign staged files
  bootstrap     apply the machine setup plan (local config + control tick)
  doctor        check prerequisites and installation state
  setup-plan    print the idempotent commands this host should run
  loop-list     list configured loops, their source, and command readiness
  loop-check    run one configured loop check and optional recovery
  loop-audit    run every enabled loop once and bucket each healthy/action/broken
  loop-scaffold create starter loop status/recovery files and registration
  loop-set      enable/disable or tune an existing loop without retyping commands
  loop-add      add a loop to the tracked catalog or machine-local config
"""
from __future__ import annotations

import argparse
import collections  # noqa: F401
import datetime as dt  # noqa: F401
import json
import os  # noqa: F401
import platform  # noqa: F401
import shutil  # noqa: F401
import signal  # noqa: F401
import socket  # noqa: F401
import subprocess  # noqa: F401
import sys
import tempfile  # noqa: F401
import time  # noqa: F401
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
if __name__ == "__main__":
    sys.modules.setdefault("fleet_control_pane", sys.modules[__name__])

import appversion  # noqa: E402, F401
from fleet_control_pane_core import (  # noqa: E402
    ACTION_LOOP_VALUES as ACTION_LOOP_VALUES,
    ATOMIC_WRITE_REPLACE_ATTEMPTS as ATOMIC_WRITE_REPLACE_ATTEMPTS,
    ATOMIC_WRITE_REPLACE_SLEEP_S as ATOMIC_WRITE_REPLACE_SLEEP_S,
    BOOTSTRAP_SCHEMA as BOOTSTRAP_SCHEMA,
    COMMIT_SCHEMA as COMMIT_SCHEMA,
    CONFIG_OVERRIDE_KEYS as CONFIG_OVERRIDE_KEYS,
    CONFIG_SCHEMA as CONFIG_SCHEMA,
    DEFAULT_WORKTREE_ALLOW_BRANCHES as DEFAULT_WORKTREE_ALLOW_BRANCHES,
    DEFAULT_WORKTREE_MASTER_REF as DEFAULT_WORKTREE_MASTER_REF,
    DIRTY_GROUP_SUBJECTS as DIRTY_GROUP_SUBJECTS,
    FLEET_SCHEMA as FLEET_SCHEMA,
    HEALTHY_LOOP_VALUES as HEALTHY_LOOP_VALUES,
    LOOP_AUDIT_SCHEMA as LOOP_AUDIT_SCHEMA,
    LOOP_AUDIT_SELF_NAME as LOOP_AUDIT_SELF_NAME,
    LOOP_BROKEN_STATES as LOOP_BROKEN_STATES,
    LOOP_CATALOG_SCHEMA as LOOP_CATALOG_SCHEMA,
    LOOP_CHECK_SCHEMA as LOOP_CHECK_SCHEMA,
    LOOP_CONFIG_SCHEMA as LOOP_CONFIG_SCHEMA,
    LOOP_LIST_SCHEMA as LOOP_LIST_SCHEMA,
    LOOP_SCAFFOLD_SCHEMA as LOOP_SCAFFOLD_SCHEMA,
    LOOP_SET_SCHEMA as LOOP_SET_SCHEMA,
    PANE_SOURCE_PATHS as PANE_SOURCE_PATHS,
    PUBLISH_SCHEMA as PUBLISH_SCHEMA,
    RECOVER_SCHEMA as RECOVER_SCHEMA,
    SCHEDULED_TASK_ACTION_STATES as SCHEDULED_TASK_ACTION_STATES,
    SCHEMA as SCHEMA,
    SETUP_SCHEMA as SETUP_SCHEMA,
    SUPERVISOR_ACTION_VERDICTS as SUPERVISOR_ACTION_VERDICTS,
    SUPERVISOR_NON_ACTION_VERDICTS as SUPERVISOR_NON_ACTION_VERDICTS,
    SUPERVISOR_SCHEMA as SUPERVISOR_SCHEMA,
    SUPERVISOR_WATCHDOG_ACTION_RESULT as SUPERVISOR_WATCHDOG_ACTION_RESULT,
    SYNC_SCHEMA as SYNC_SCHEMA,
    TICK_SCHEMA as TICK_SCHEMA,
    TRACKED_DEFAULT_DRIFT_KEYS as TRACKED_DEFAULT_DRIFT_KEYS,
    TRUNK_GUARD_HOOKS_REL as TRUNK_GUARD_HOOKS_REL,
    TRUNK_GUARD_HOOK_REL as TRUNK_GUARD_HOOK_REL,
    TRUNK_GUARD_INSTALLER_REL as TRUNK_GUARD_INSTALLER_REL,
    WINDOWS_TASK_REQUEST_REFUSED_RESULT as WINDOWS_TASK_REQUEST_REFUSED_RESULT,
    WINDOWS_TASK_RESULT_LABELS as WINDOWS_TASK_RESULT_LABELS,
    WINDOWS_TASK_RUNNING_RESULT as WINDOWS_TASK_RUNNING_RESULT,
    apply_env_overrides as apply_env_overrides,
    apply_runtime_overrides as apply_runtime_overrides,
    bind_pane as bind_pane,
    clean_config_overrides as clean_config_overrides,
    command_exists as command_exists,
    command_input_missing as command_input_missing,
    command_readiness as command_readiness,
    config_paths as config_paths,
    config_runtime_dirs as config_runtime_dirs,
    deep_merge as deep_merge,
    default_config as default_config,
    directory_probe as directory_probe,
    display_command as display_command,
    display_path as display_path,
    ensure_runtime_dirs as ensure_runtime_dirs,
    env_for_tools as env_for_tools,
    expand_cmd as expand_cmd,
    expand_value as expand_value,
    git as git,
    git_stdout as git_stdout,
    init_command as init_command,
    init_config as init_config,
    init_config_doc as init_config_doc,
    iso_now as iso_now,
    load_config as load_config,
    local_config_default_drift as local_config_default_drift,
    local_config_mismatches as local_config_mismatches,
    machine_id as machine_id,
    normalize_config as normalize_config,
    now_utc as now_utc,
    pane_attr as pane_attr,
    parse_iso_age_min as parse_iso_age_min,
    platform_is_windows as platform_is_windows,
    powershell_exe as powershell_exe,
    ps_quote as ps_quote,
    read_json as read_json,
    read_json_from_text as read_json_from_text,
    replace_with_retry as replace_with_retry,
    repo_root as repo_root,
    resolve_path as resolve_path,
    run as run,
    sh_exe as sh_exe,
    step_id_slug as step_id_slug,
    tracked_default_config as tracked_default_config,
    write_json as write_json,
    write_text_atomic as write_text_atomic,
)
from fleet_control_pane_git import (  # noqa: E402
    ahead_commits as ahead_commits,
    collect_git as collect_git,
    collect_worktree_doctor as collect_worktree_doctor,
    commit_plan as commit_plan,
    configured_hooks_path as configured_hooks_path,
    dirty_commit_plan as dirty_commit_plan,
    dirty_group_commit_subject as dirty_group_commit_subject,
    dirty_group_key as dirty_group_key,
    dirty_group_selection as dirty_group_selection,
    git_mutation_guard as git_mutation_guard,
    git_worktree_summary as git_worktree_summary,
    normalize_repo_path as normalize_repo_path,
    parse_git_status_porcelain as parse_git_status_porcelain,
    path_compare_key as path_compare_key,
    path_status as path_status,
    publish_plan as publish_plan,
    split_z as split_z,
    staged_paths as staged_paths,
    sync_plan as sync_plan,
    trunk_guard_status as trunk_guard_status,
    worktree_allow_branches as worktree_allow_branches,
    worktree_doctor_command as worktree_doctor_command,
)
from fleet_control_pane_loops import (  # noqa: E402
    classify_loop_status as classify_loop_status,
    collect_loop as collect_loop,
    collect_loops as collect_loops,
    invoke_loop_recoveries as invoke_loop_recoveries,
    loop_add_template_command as loop_add_template_command,
    loop_audit as loop_audit,
    loop_audit_detail as loop_audit_detail,
    loop_check_needs_action as loop_check_needs_action,
    loop_check_plan as loop_check_plan,
    loop_check_snapshot as loop_check_snapshot,
    loop_config_command as loop_config_command,
    loop_config_plan as loop_config_plan,
    loop_cwd as loop_cwd,
    loop_expected_returncodes as loop_expected_returncodes,
    loop_list as loop_list,
    loop_recover_scaffold_text as loop_recover_scaffold_text,
    loop_recovery_action as loop_recovery_action,
    loop_scaffold_command as loop_scaffold_command,
    loop_scaffold_plan as loop_scaffold_plan,
    loop_scaffold_template_command as loop_scaffold_template_command,
    loop_script_slug as loop_script_slug,
    loop_set_command as loop_set_command,
    loop_set_plan as loop_set_plan,
    loop_setup_steps as loop_setup_steps,
    loop_source_entries as loop_source_entries,
    loop_status_scaffold_text as loop_status_scaffold_text,
    parse_command_arg as parse_command_arg,
    recovery_invokes_control_bootstrap as recovery_invokes_control_bootstrap,
)
from fleet_control_pane_runtime import (  # noqa: E402
    _bootstrap_control_tick_action as _bootstrap_control_tick_action,
    _bootstrap_local_config_action as _bootstrap_local_config_action,
    _bootstrap_runtime_dirs_action as _bootstrap_runtime_dirs_action,
    _bootstrap_trunk_guard_action as _bootstrap_trunk_guard_action,
    bootstrap as bootstrap,
    collect_control_tick as collect_control_tick,
    collect_status as collect_status,
    collect_supervisor as collect_supervisor,
    collect_watchdogs as collect_watchdogs,
    control_tick as control_tick,
    control_tick_register_script_path as control_tick_register_script_path,
    control_tick_register_script_value as control_tick_register_script_value,
    control_tick_runner_missing as control_tick_runner_missing,
    control_tick_runner_path as control_tick_runner_path,
    doctor as doctor,
    fleet_view as fleet_view,
    invoke_watchdog as invoke_watchdog,
    posix_tick_status as posix_tick_status,
    recommended_actions as recommended_actions,
    recover_plan as recover_plan,
    refresh_registry as refresh_registry,
    register_command as register_command,
    registry_path as registry_path,
    scheduled_task_status as scheduled_task_status,
    setup_plan as setup_plan,
    supervisor_needs_action as supervisor_needs_action,
    supervisor_plan as supervisor_plan,
    terminate_pid as terminate_pid,
    watchdog_command as watchdog_command,
)
from fleet_control_pane_views import (  # noqa: E402
    _DOC_PRINTERS as _DOC_PRINTERS,
    _print_bootstrap as _print_bootstrap,
    _print_commit as _print_commit,
    _print_doctor as _print_doctor,
    _print_fleet as _print_fleet,
    _print_loop_audit as _print_loop_audit,
    _print_loop_check as _print_loop_check,
    _print_loop_config as _print_loop_config,
    _print_loop_list as _print_loop_list,
    _print_loop_scaffold as _print_loop_scaffold,
    _print_loop_set as _print_loop_set,
    _print_publish as _print_publish,
    _print_recover as _print_recover,
    _print_setup_plan as _print_setup_plan,
    _print_supervisor as _print_supervisor,
    _print_sync as _print_sync,
    _print_tick as _print_tick,
    account_block_is_access_wall as account_block_is_access_wall,
    account_block_is_auth_like as account_block_is_auth_like,
    account_block_is_usage_like as account_block_is_usage_like,
    account_block_needs_login as account_block_needs_login,
    account_profile_command as account_profile_command,
    blocked_account_action_lines as blocked_account_action_lines,
    blocked_account_reasons as blocked_account_reasons,
    blocked_account_recommended_action as blocked_account_recommended_action,
    blocked_account_snapshot as blocked_account_snapshot,
    blocked_account_tags as blocked_account_tags,
    compact_counts as compact_counts,
    dirty_group_summary as dirty_group_summary,
    fleet_text as fleet_text,
    loop_issue_summary as loop_issue_summary,
    machine_git_action as machine_git_action,
    machine_setup_action_count as machine_setup_action_count,
    machine_setup_summary as machine_setup_summary,
    machine_version_drift as machine_version_drift,
    pane_text as pane_text,
    print_doc as print_doc,
    scheduled_task_has_data as scheduled_task_has_data,
    scheduled_task_label as scheduled_task_label,
    scheduled_task_needs_action as scheduled_task_needs_action,
    scheduled_task_snapshot as scheduled_task_snapshot,
    scheduled_task_summary_needs_action as scheduled_task_summary_needs_action,
    scheduler_result_int as scheduler_result_int,
    scheduler_result_needs_action as scheduler_result_needs_action,
    scheduler_result_text as scheduler_result_text,
    scheduler_state_needs_action as scheduler_state_needs_action,
    setup_plan_action_steps as setup_plan_action_steps,
    setup_plan_snapshot as setup_plan_snapshot,
    setup_step_snapshot as setup_step_snapshot,
    summarize_machine_snapshot as summarize_machine_snapshot,
    summarize_registry as summarize_registry,
    summarize_resume_plan as summarize_resume_plan,
    supervisor_diagnosis_is_actionable as supervisor_diagnosis_is_actionable,
    supervisor_health_action_text as supervisor_health_action_text,
    supervisor_health_needs_action as supervisor_health_needs_action,
    supervisor_health_summary as supervisor_health_summary,
    supervisor_log_hint as supervisor_log_hint,
    supervisor_payload_diagnosis_health as supervisor_payload_diagnosis_health,
    supervisor_payload_run_health as supervisor_payload_run_health,
    supervisor_payload_verdict as supervisor_payload_verdict,
    supervisor_run_health_is_actionable as supervisor_run_health_is_actionable,
    task_is_installed as task_is_installed,
    write_status as write_status,
)


def add_config_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", dest="python", help="Python executable stored in local config")
    parser.add_argument("--user-home", help="user home used for account/session discovery")
    parser.add_argument("--job-dir", help="job repo/path used by supervisor status commands")
    parser.add_argument("--claude-exe", help="Claude executable path for resume watchdogs")
    parser.add_argument("--registry-dir", help="local registry directory")
    parser.add_argument("--machine-dir", help="directory where this host publishes machine snapshots")
    parser.add_argument("--machine-id", help="stable id for this machine snapshot")
    parser.add_argument("--watchdog-log-dir", help="watchdog log directory")
    parser.add_argument("--git-remote", help="git remote used for safe-ff checks")
    parser.add_argument("--target", type=int, help="target number of supervised workers")
    parser.add_argument("--session-window-h", type=float, help="session lookback window in hours")


def config_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: getattr(args, key, None)
        for key in CONFIG_OVERRIDE_KEYS
        if hasattr(args, key) and getattr(args, key, None) is not None
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One control pane for fleet agent loops.")
    parser.add_argument("--root", default=None, help="repo root (default: discovered from VERSION)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="write machine-local config")
    p_init.add_argument("--force", action="store_true")
    p_init.add_argument("--json", action="store_true")
    add_config_override_args(p_init)

    p_status = sub.add_parser("status", help="print the pane")
    p_status.add_argument("--refresh", action="store_true", help="refresh tools/_registry/sessions.json first")
    p_status.add_argument("--write", action="store_true", help="write control_pane.json and CONTROL-PANE.txt")
    p_status.add_argument("--fail-on-action", action="store_true", help="exit 1 when the pane verdict is ACTION")
    p_status.add_argument("--json", action="store_true")

    p_tick = sub.add_parser("tick", help="run one keep-alive control tick")
    p_tick.add_argument("--dry-run", action="store_true", help="show watchdog commands without invoking them")
    p_tick.add_argument("--live-resume", action="store_true", help="pass -Live to the resume watchdog")
    p_tick.add_argument("--skip-supervisor", action="store_true")
    p_tick.add_argument("--skip-resume", action="store_true")
    p_tick.add_argument("--no-write", action="store_true", help="do not write control_pane.json / CONTROL-PANE.txt")
    p_tick.add_argument("--json", action="store_true")

    p_recover = sub.add_parser("recover", help="dry-run/apply one live recovery pass")
    p_recover.add_argument("--apply", action="store_true", help="actually resume eligible sessions and configured loop recoveries")
    p_recover.add_argument("--skip-supervisor", action="store_true", help="skip supervisor watchdog during recovery")
    p_recover.add_argument("--no-write", action="store_true", help="do not write control_pane.json / CONTROL-PANE.txt")
    p_recover.add_argument("--json", action="store_true")

    p_supervisor = sub.add_parser("supervisor", help="inspect or restart the standing supervisor")
    p_supervisor.add_argument("--restart", action="store_true", help="plan or apply a supervisor restart")
    p_supervisor.add_argument("--apply", action="store_true", help="actually terminate and respawn; default is dry-run")
    p_supervisor.add_argument("--json", action="store_true")

    p_fleet = sub.add_parser("fleet", help="aggregate published machine snapshots")
    p_fleet.add_argument("--fail-on-action", action="store_true", help="exit 1 when any machine needs action")
    p_fleet.add_argument("--snapshot-only", action="store_true", help="read published snapshots only; do not overlay this host live")
    p_fleet.add_argument("--refresh-local", action="store_true", help="refresh sessions and write this host snapshot before aggregating")
    p_fleet.add_argument("--json", action="store_true")

    p_sync = sub.add_parser("sync", help="check/apply safe fast-forward sync; never plain git pull")
    p_sync.add_argument("--fetch", action="store_true", help="fetch remote branch before checking")
    p_sync.add_argument("--apply", action="store_true", help="apply safe fast-forward when every ff path is safe")
    p_sync.add_argument("--json", action="store_true")

    p_publish = sub.add_parser("publish", help="push current branch only when it is ahead of remote")
    p_publish.add_argument("--apply", action="store_true", help="actually run git push; default is dry-run")
    p_publish.add_argument("--no-fetch", action="store_true", help="skip pre-push fetch/assessment refresh")
    p_publish.add_argument("--allow-dirty", action="store_true", help="allow ordinary dirty paths while still refusing merges/unmerged paths")
    p_publish.add_argument("--json", action="store_true")

    p_commit = sub.add_parser("commit", help="stage and commit only explicitly selected files")
    p_commit.add_argument("--path", action="append", default=[], help="repo path to include (repeatable)")
    p_commit.add_argument("--dirty-group", action="append", default=[], help="include paths from a dirty-plan group")
    p_commit.add_argument("--pane-files", action="store_true", help="include the pane source/doc files")
    p_commit.add_argument("--message", "-m", default="", help="commit subject/body")
    p_commit.add_argument("--apply", action="store_true", help="actually git add and git commit")
    p_commit.add_argument("--allow-dir", action="store_true", help="allow directory pathspecs")
    p_commit.add_argument("--json", action="store_true")

    p_bootstrap = sub.add_parser("bootstrap", help="apply machine setup: local config + control tick")
    p_bootstrap.add_argument("--apply", action="store_true", help="write config/install task; default is dry-run")
    p_bootstrap.add_argument("--live-resume", action="store_true", help="install tick with resume watchdog in live mode")
    p_bootstrap.add_argument("--force-local-config", action="store_true", help="rewrite existing local config when override flags differ")
    p_bootstrap.add_argument("--json", action="store_true")
    add_config_override_args(p_bootstrap)

    p_doctor = sub.add_parser("doctor", help="check host setup")
    p_doctor.add_argument("--json", action="store_true")

    p_setup = sub.add_parser("setup-plan", help="print setup commands for this host")
    p_setup.add_argument("--json", action="store_true")

    p_loop_list = sub.add_parser("loop-list", help="list configured loops, their source, and command readiness")
    p_loop_list.add_argument("--json", action="store_true")

    p_loop_check = sub.add_parser("loop-check", help="run one configured loop check and optional recovery")
    p_loop_check.add_argument("name", help="loop name")
    p_loop_check.add_argument("--recover", action="store_true", help="show the loop's recovery command when action is needed")
    p_loop_check.add_argument("--apply", action="store_true", help="actually run recovery; requires --recover")
    p_loop_check.add_argument("--fail-on-action", action="store_true", help="exit 1 when the loop reports ACTION")
    p_loop_check.add_argument("--json", action="store_true")

    p_loop_audit = sub.add_parser(
        "loop-audit",
        help="run every enabled loop once and bucket each healthy/action/broken (exit 1 only on broken)",
    )
    p_loop_audit.add_argument("--names", help="comma-separated subset of loops to audit; default is all enabled")
    p_loop_audit.add_argument(
        "--fail-on-action",
        action="store_true",
        help="also exit 1 when any loop is in the action bucket (a loop correctly surfacing a condition)",
    )
    p_loop_audit.add_argument("--json", action="store_true")

    p_loop_scaffold = sub.add_parser("loop-scaffold", help="create starter loop files and repo catalog registration")
    p_loop_scaffold.add_argument("name", help="loop name")
    p_loop_scaffold.add_argument("--enabled", action="store_true", help="register the new loop enabled; default is disabled")
    p_loop_scaffold.add_argument("--auto-recover", action="store_true", help="mark the loop for automatic recovery when enabled")
    p_loop_scaffold.add_argument("--force", action="store_true", help="overwrite existing scaffold files or loop entry")
    p_loop_scaffold.add_argument("--apply", action="store_true", help="write scaffold files and catalog entry; default is dry-run")
    p_loop_scaffold.add_argument("--json", action="store_true")

    p_loop_set = sub.add_parser("loop-set", help="enable/disable or tune an existing loop without retyping commands")
    p_loop_set.add_argument("name", help="loop name")
    p_loop_set.add_argument(
        "--scope",
        choices=("local", "repo"),
        default="repo",
        help="write ignored machine-local config or the tracked repo loop catalog",
    )
    set_enabled_group = p_loop_set.add_mutually_exclusive_group()
    set_enabled_group.add_argument("--enable", action="store_true", dest="enabled", default=None)
    set_enabled_group.add_argument("--disable", action="store_false", dest="enabled")
    set_recover_group = p_loop_set.add_mutually_exclusive_group()
    set_recover_group.add_argument("--auto-recover", action="store_true", dest="auto_recover", default=None)
    set_recover_group.add_argument("--no-auto-recover", action="store_false", dest="auto_recover")
    p_loop_set.add_argument("--apply", action="store_true", help="write config; default is dry-run")
    p_loop_set.add_argument("--json", action="store_true")

    p_loop = sub.add_parser("loop-add", help="add or update an enabled loop in local config or the repo catalog")
    p_loop.add_argument("name", help="loop name")
    p_loop.add_argument(
        "--scope",
        choices=("local", "repo"),
        default="local",
        help="write ignored machine-local config or the tracked repo loop catalog",
    )
    p_loop.add_argument("--status-cmd", required=True, help="status command as a shell string or JSON string array")
    p_loop.add_argument("--recover-cmd", default=None, help="recovery command as a shell string or JSON string array")
    recover_group = p_loop.add_mutually_exclusive_group()
    recover_group.add_argument("--auto-recover", action="store_true", dest="auto_recover", default=None)
    recover_group.add_argument("--no-auto-recover", action="store_false", dest="auto_recover")
    p_loop.add_argument("--action", default="", help="operator-facing action text when the loop needs attention")
    p_loop.add_argument("--timeout-s", type=int, default=None)
    p_loop.add_argument("--recover-timeout-s", type=int, default=None)
    p_loop.add_argument("--cwd", default="", help="working directory for loop commands")
    p_loop.add_argument("--ok-returncode", action="append", type=int, default=[])
    p_loop.add_argument("--disabled", action="store_true", help="write the loop entry disabled")
    p_loop.add_argument("--apply", action="store_true", help="write local config; default is dry-run")
    p_loop.add_argument("--json", action="store_true")
    return parser


def _run_loop_command(args: argparse.Namespace, config: dict[str, Any],
                      parser: argparse.ArgumentParser) -> int | None:
    """Run the loop-* family of subcommands and return its exit code, or None
    when args.cmd is not one of them."""
    if args.cmd == "loop-add":
        try:
            status_cmd = parse_command_arg(args.status_cmd)
            recover_cmd = parse_command_arg(args.recover_cmd) if args.recover_cmd is not None else None
        except ValueError as exc:
            doc = {
                "schema": LOOP_CONFIG_SCHEMA,
                "generated_utc": pane_attr("iso_now", iso_now)(),
                "apply": args.apply,
                "ok": False,
                "changed": False,
                "scope": args.scope,
                "loop": args.name,
                "config_path": config["_paths"].get("loop_catalog" if args.scope == "repo" else "local"),
                "local_config": config["_paths"].get("local") if args.scope == "local" else None,
                "reason": str(exc),
                "spec": {},
            }
            pane_attr("print_doc", print_doc)(doc, args.json)
            return 1
        doc = pane_attr("loop_config_plan", loop_config_plan)(
            config,
            loop_name=args.name,
            scope=args.scope,
            status_cmd=status_cmd,
            recover_cmd=recover_cmd,
            auto_recover=args.auto_recover,
            action=args.action,
            timeout_s=args.timeout_s,
            recover_timeout_s=args.recover_timeout_s,
            cwd=args.cwd,
            ok_returncodes=args.ok_returncode,
            enabled=not args.disabled,
            apply=args.apply,
        )
        pane_attr("print_doc", print_doc)(doc, args.json)
        return 0 if doc.get("ok") else 1
    if args.cmd == "loop-list":
        doc = pane_attr("loop_list", loop_list)(config)
        pane_attr("print_doc", print_doc)(doc, args.json)
        return 0 if doc.get("ok") else 1
    if args.cmd == "loop-check":
        if args.apply and not args.recover:
            parser.error("loop-check --apply requires --recover")
        doc = pane_attr("loop_check_plan", loop_check_plan)(config, args.name, recover=args.recover, apply=args.apply)
        pane_attr("print_doc", print_doc)(doc, args.json)
        if not doc.get("ok"):
            return 1
        return 1 if args.fail_on_action and doc.get("needs_action") else 0
    if args.cmd == "loop-audit":
        names = [n.strip() for n in (args.names or "").split(",") if n.strip()] or None
        doc = pane_attr("loop_audit", loop_audit)(config, names=names)
        pane_attr("print_doc", print_doc)(doc, args.json)
        if not doc.get("ok"):
            return 1
        return 1 if args.fail_on_action and doc.get("counts", {}).get("action") else 0
    if args.cmd == "loop-scaffold":
        doc = pane_attr("loop_scaffold_plan", loop_scaffold_plan)(
            config,
            args.name,
            enabled=args.enabled,
            auto_recover=args.auto_recover,
            force=args.force,
            apply=args.apply,
        )
        pane_attr("print_doc", print_doc)(doc, args.json)
        return 0 if doc.get("ok") else 1
    if args.cmd == "loop-set":
        doc = pane_attr("loop_set_plan", loop_set_plan)(
            config,
            args.name,
            scope=args.scope,
            enabled=args.enabled,
            auto_recover=args.auto_recover,
            apply=args.apply,
        )
        pane_attr("print_doc", print_doc)(doc, args.json)
        return 0 if doc.get("ok") else 1
    return None


def _run_commit_command(args: argparse.Namespace, config: dict[str, Any]) -> int:
    paths = list(args.path or [])
    dirty_selection = None
    if args.dirty_group:
        dirty_selection = pane_attr("dirty_group_selection", dirty_group_selection)(config, list(args.dirty_group))
        if not dirty_selection.get("ok"):
            doc = {
                "schema": COMMIT_SCHEMA,
                "generated_utc": pane_attr("iso_now", iso_now)(),
                "ok": False,
                "applied": False,
                "reason": "unknown dirty group(s): " + ", ".join(dirty_selection.get("missing") or []),
                "dirty_groups": dirty_selection,
            }
            pane_attr("print_doc", print_doc)(doc, args.json)
            return 1
        paths.extend(dirty_selection.get("paths") or [])
    if args.pane_files:
        paths.extend(PANE_SOURCE_PATHS)
    try:
        doc = pane_attr("commit_plan", commit_plan)(
            config,
            paths=paths,
            message=args.message,
            apply=args.apply,
            allow_dir=args.allow_dir,
        )
        if dirty_selection is not None:
            doc["dirty_groups"] = dirty_selection
    except ValueError as exc:
        doc = {
            "schema": COMMIT_SCHEMA,
            "generated_utc": pane_attr("iso_now", iso_now)(),
            "ok": False,
            "applied": False,
            "reason": str(exc),
        }
    pane_attr("print_doc", print_doc)(doc, args.json)
    return 0 if doc.get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    with bind_pane(sys.modules[__name__]):
        parser = _build_arg_parser()
        args = parser.parse_args(argv)
        root = pane_attr("repo_root", repo_root)(Path(args.root)) if args.root else pane_attr("repo_root", repo_root)()

        if args.cmd == "init":
            base_config = pane_attr("load_config", load_config)(root)
            overrides = config_overrides_from_args(args)
            base_config = apply_runtime_overrides(base_config, root, overrides)
            result = pane_attr("init_config", init_config)(root, force=args.force, overrides=overrides, base_config=base_config)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(("wrote " if result["written"] else "kept ") + result["path"])
            return 0

        config = pane_attr("load_config", load_config)(root)
        loop_rc = _run_loop_command(args, config, parser)
        if loop_rc is not None:
            return loop_rc
        if args.cmd == "bootstrap":
            overrides = config_overrides_from_args(args)
            config = apply_runtime_overrides(config, root, overrides)
        else:
            overrides = {}
        if args.cmd == "status":
            status = pane_attr("collect_status", collect_status)(config, refresh=args.refresh)
            if args.write:
                status["written"] = pane_attr("write_status", write_status)(status, config)
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                print(pane_attr("pane_text", pane_text)(status))
                if args.write:
                    print(f"written: {status['written']['json']} ; {status['written']['text']}")
            return 1 if args.fail_on_action and status["verdict"] != "OK" else 0
        if args.cmd == "tick":
            doc = pane_attr("control_tick", control_tick)(
                config,
                dry_run=args.dry_run,
                live_resume=args.live_resume,
                skip_supervisor=args.skip_supervisor,
                skip_resume=args.skip_resume,
                write=not args.no_write,
            )
            pane_attr("print_doc", print_doc)(doc, args.json)
            return 0 if doc["ok"] else 1
        if args.cmd == "recover":
            doc = pane_attr("recover_plan", recover_plan)(
                config,
                apply=args.apply,
                skip_supervisor=args.skip_supervisor,
                write=not args.no_write,
            )
            pane_attr("print_doc", print_doc)(doc, args.json)
            return 0 if doc.get("ok") else 1
        if args.cmd == "supervisor":
            doc = pane_attr("supervisor_plan", supervisor_plan)(config, restart=args.restart, apply=args.apply)
            pane_attr("print_doc", print_doc)(doc, args.json)
            return 0 if doc.get("ok") else 1
        if args.cmd == "fleet":
            if args.snapshot_only and args.refresh_local:
                parser.error("fleet --refresh-local cannot be combined with --snapshot-only")
            doc = pane_attr("fleet_view", fleet_view)(
                config,
                include_live_local=not args.snapshot_only,
                refresh_live_local=args.refresh_local,
                write_live_local=args.refresh_local,
            )
            pane_attr("print_doc", print_doc)(doc, args.json)
            return 1 if args.fail_on_action and doc["verdict"] != "OK" else 0
        if args.cmd == "sync":
            doc = pane_attr("sync_plan", sync_plan)(config, fetch=args.fetch, apply=args.apply)
            pane_attr("print_doc", print_doc)(doc, args.json)
            return 0 if doc.get("ok") else 1
        if args.cmd == "publish":
            doc = pane_attr("publish_plan", publish_plan)(
                config,
                fetch=not args.no_fetch,
                apply=args.apply,
                allow_dirty=args.allow_dirty,
            )
            pane_attr("print_doc", print_doc)(doc, args.json)
            return 0 if doc.get("ok") else 1
        if args.cmd == "commit":
            return _run_commit_command(args, config)
        if args.cmd == "bootstrap":
            doc = pane_attr("bootstrap", bootstrap)(
                config,
                apply=args.apply,
                live_resume=args.live_resume,
                init_overrides=overrides,
                force_local_config=args.force_local_config,
            )
            pane_attr("print_doc", print_doc)(doc, args.json)
            return 0 if doc.get("ok") else 1
        if args.cmd == "doctor":
            doc = pane_attr("doctor", doctor)(config)
            pane_attr("print_doc", print_doc)(doc, args.json)
            return 0 if doc["ok"] else 1
        if args.cmd == "setup-plan":
            pane_attr("print_doc", print_doc)(pane_attr("setup_plan", setup_plan)(config), args.json)
            return 0
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
