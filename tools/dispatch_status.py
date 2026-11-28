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
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lane_yield import (  # noqa: E402  (shared low-yield fold, #2062)
    count_lane_ancestry_closes,
    low_yield_lanes,
    _log_turn_count as _log_turn_count,
    _LOW_YIELD_SCHEMA as _LOW_YIELD_SCHEMA,
    _LOW_YIELD_TURNS_FLOOR as _LOW_YIELD_TURNS_FLOOR,
    _LOW_YIELD_LOOKBACK_MIN as _LOW_YIELD_LOOKBACK_MIN,
)
from dispatch_status_common import (  # noqa: E402
    DETAIL_AUTH_BLOCKED as DETAIL_AUTH_BLOCKED,
    DETAIL_GATEWAY_DOWN as DETAIL_GATEWAY_DOWN,
    DETAIL_INDETERMINATE as DETAIL_INDETERMINATE,
    DETAIL_JOB_IN_FLIGHT as DETAIL_JOB_IN_FLIGHT,
    DETAIL_PRIVATE_RECOVERY as DETAIL_PRIVATE_RECOVERY,
    DETAIL_READY as DETAIL_READY,
    LAB_READINESS_KEYS as LAB_READINESS_KEYS,
    LAB_READINESS_SCHEMA as LAB_READINESS_SCHEMA,
    LEGACY_LAB_STATUSES as LEGACY_LAB_STATUSES,
    LINK_CLEAR as LINK_CLEAR,
    LINK_DETAILS as LINK_DETAILS,
    LINK_PHASES as LINK_PHASES,
    LINK_STATE_SCHEMA as LINK_STATE_SCHEMA,
    LINK_WAITING as LINK_WAITING,
    LINK_WORKING as LINK_WORKING,
    RUNS_DIRNAME as RUNS_DIRNAME,
    SCHEMA as SCHEMA,
    VERDICT_ADMISSION_UNKNOWN as VERDICT_ADMISSION_UNKNOWN,
    VERDICT_LAUNCH_COOLDOWN as VERDICT_LAUNCH_COOLDOWN,
    WATCHDOG_TASK as WATCHDOG_TASK,
    _CREATE_NO_WINDOW as _CREATE_NO_WINDOW,
    _GENERIC_TOKEN_RE as _GENERIC_TOKEN_RE,
    _LAB_MARK_CLEAR_COMMAND as _LAB_MARK_CLEAR_COMMAND,
    _LAB_MARK_WAITING_COMMAND as _LAB_MARK_WAITING_COMMAND,
    _LAB_MARK_WORKING_COMMAND as _LAB_MARK_WORKING_COMMAND,
    _LEGACY_LAB_KEYS as _LEGACY_LAB_KEYS,
    _LINK_STATE_KEYS as _LINK_STATE_KEYS,
    _NEXT_ACTION_BY_VERDICT as _NEXT_ACTION_BY_VERDICT,
    _RESOLVE_LOG_RE as _RESOLVE_LOG_RE,
    _STUB_LOG_MAX_BYTES as _STUB_LOG_MAX_BYTES,
    _age_text as _age_text,
    _capacity_reconcile as _capacity_reconcile,
    _current_preflight_launch_blocker as _current_preflight_launch_blocker,
    _default_detail as _default_detail,
    _dispatch_limiter as _dispatch_limiter,
    _dispatch_limiter_terms as _dispatch_limiter_terms,
    _epoch_iso as _epoch_iso,
    _github_rate_limit_error as _github_rate_limit_error,
    _int as _int,
    _iso_epoch as _iso_epoch,
    _lab_link_label as _lab_link_label,
    _lab_readiness_commands as _lab_readiness_commands,
    _last_json as _last_json,
    _next_action as _next_action,
    _or_unknown as _or_unknown,
    _py as _py,
    _resolver_preflight_summary as _resolver_preflight_summary,
    _slots_headroom_note as _slots_headroom_note,
    _win_creationflags as _win_creationflags,
    _workers_live_clause as _workers_live_clause,
    repo_root as repo_root,
)
from dispatch_status_leases import (  # noqa: E402
    LANE_HOLDER_DEAD as LANE_HOLDER_DEAD,
    LANE_HOLDER_LIVE as LANE_HOLDER_LIVE,
    LANE_HOLDER_UNKNOWN as LANE_HOLDER_UNKNOWN,
    _FILETIME_EPOCH_DELTA_S as _FILETIME_EPOCH_DELTA_S,
    _LANE_LEASE_DEFAULT_TTL_MIN as _LANE_LEASE_DEFAULT_TTL_MIN,
    _LANE_LEASE_GRACE_MIN as _LANE_LEASE_GRACE_MIN,
    _LANE_LEASE_NEXT_ACTION as _LANE_LEASE_NEXT_ACTION,
    _LANE_LEASE_NEXT_ACTION_STEPS as _LANE_LEASE_NEXT_ACTION_STEPS,
    _LANE_LEASE_SCHEMA as _LANE_LEASE_SCHEMA,
    _LEASEREF_PREFIX as _LEASEREF_PREFIX,
    _NOOP_BANNER_RE as _NOOP_BANNER_RE,
    _PROC_START_MAX_EPOCH_S as _PROC_START_MAX_EPOCH_S,
    _PROC_START_MIN_EPOCH_S as _PROC_START_MIN_EPOCH_S,
    _PROC_START_TOLERANCE_S as _PROC_START_TOLERANCE_S,
    _RID_RE as _RID_RE,
    _RUN_STATUS_LOOKBACK_MIN as _RUN_STATUS_LOOKBACK_MIN,
    _RUN_STATUS_START_KINDS as _RUN_STATUS_START_KINDS,
    _RUN_STATUS_TERMINAL_KINDS as _RUN_STATUS_TERMINAL_KINDS,
    _RUN_STATUS_TERMINAL_STATUSES as _RUN_STATUS_TERMINAL_STATUSES,
    _active_worker_rows as _active_worker_rows,
    _active_worker_summary as _active_worker_summary,
    _backlog_candidates as _backlog_candidates,
    _banner_noop as _banner_noop,
    _clean_tree as _clean_tree,
    _default_lease_id_for_lane as _default_lease_id_for_lane,
    _dispatch_lease_token as _dispatch_lease_token,
    _fold_lane_leases as _fold_lane_leases,
    _lane_lease_bits as _lane_lease_bits,
    _lane_lease_ttl_min as _lane_lease_ttl_min,
    _lease_acquired_epoch as _lease_acquired_epoch,
    _lease_active_unix as _lease_active_unix,
    _lease_expired as _lease_expired,
    _lease_lane as _lease_lane,
    _lease_liveness as _lease_liveness,
    _normalize_tree as _normalize_tree,
    _proc_starttime_epoch as _proc_starttime_epoch,
    _read_worker_account as _read_worker_account,
    _read_worker_lease_id as _read_worker_lease_id,
    _read_worker_text_sidecar as _read_worker_text_sidecar,
    _read_worker_tree as _read_worker_tree,
    _session_expired as _session_expired,
    _spawn_lane as _spawn_lane,
    _string_list as _string_list,
    _tree_overlap_one as _tree_overlap_one,
    classify_lane_lease_holder as classify_lane_lease_holder,
    cross_check_worker_leases as cross_check_worker_leases,
    dos_status_digest as dos_status_digest,
    has_key_named as has_key_named,
    lane_lease_liveness as lane_lease_liveness,
    lane_lease_verdict_clean as lane_lease_verdict_clean,
    probe_lane_lease_pid as probe_lane_lease_pid,
    process_start_times as process_start_times,
    read_lane_leases as read_lane_leases,
    read_lease_state as read_lease_state,
    read_leaseref_records as read_leaseref_records,
    read_leaseref_records_and_sessions as read_leaseref_records_and_sessions,
    read_run_status_digests as read_run_status_digests,
    run_ids_from_loop_ledger as run_ids_from_loop_ledger,
    scan_live_dispatch_workers as scan_live_dispatch_workers,
    summarize_lane_lease_holders as summarize_lane_lease_holders,
    summarize_leases as summarize_leases,
    trees_overlap as trees_overlap,
    worker_lease_crosscheck as worker_lease_crosscheck,
)
from dispatch_status_telemetry import (  # noqa: E402
    BACKLOG_RATE_K as BACKLOG_RATE_K,
    BACKLOG_RATE_SCHEMA as BACKLOG_RATE_SCHEMA,
    BACKLOG_RATE_WINDOWS as BACKLOG_RATE_WINDOWS,
    BACKLOG_RATE_WINDOW_HOURS as BACKLOG_RATE_WINDOW_HOURS,
    GUARD_AUDIT_DIRNAME as GUARD_AUDIT_DIRNAME,
    PROGRESS_LOG as PROGRESS_LOG,
    ROUTE_HEALTH_LOG as ROUTE_HEALTH_LOG,
    WATCH_DECISION_SCHEMA as WATCH_DECISION_SCHEMA,
    WATCH_DEFAULT_WINDOW_HOURS as WATCH_DEFAULT_WINDOW_HOURS,
    _BACKEND_STUB_LOOKBACK_MIN as _BACKEND_STUB_LOOKBACK_MIN,
    _COMMIT_DROUGHT_HOURS as _COMMIT_DROUGHT_HOURS,
    _FAK_WORKER_TRAILER_RE as _FAK_WORKER_TRAILER_RE,
    _GUARD_DENY_KINDS as _GUARD_DENY_KINDS,
    _GUARD_EMPTY_CLEAN_EXIT_NO_COMMIT as _GUARD_EMPTY_CLEAN_EXIT_NO_COMMIT,
    _GUARD_EMPTY_DIED_BEFORE_EPILOGUE as _GUARD_EMPTY_DIED_BEFORE_EPILOGUE,
    _GUARD_EMPTY_INTERACTIVE as _GUARD_EMPTY_INTERACTIVE,
    _GUARD_EMPTY_LOG_SCAN_CAP as _GUARD_EMPTY_LOG_SCAN_CAP,
    _GUARD_EMPTY_LOG_TAIL_BYTES as _GUARD_EMPTY_LOG_TAIL_BYTES,
    _GUARD_EMPTY_LOG_WINDOW_S as _GUARD_EMPTY_LOG_WINDOW_S,
    _GUARD_EMPTY_MISSING_LOG as _GUARD_EMPTY_MISSING_LOG,
    _GUARD_EMPTY_PROVIDER_QUOTA_WALL as _GUARD_EMPTY_PROVIDER_QUOTA_WALL,
    _GUARD_EMPTY_SPAWN_FAILED as _GUARD_EMPTY_SPAWN_FAILED,
    _GUARD_EMPTY_UNKNOWN as _GUARD_EMPTY_UNKNOWN,
    _GUARD_EMPTY_WITNESS_REASONS as _GUARD_EMPTY_WITNESS_REASONS,
    _GUARD_EPILOGUE_MARKER as _GUARD_EPILOGUE_MARKER,
    _GUARD_INTERACTIVE_PREFIX as _GUARD_INTERACTIVE_PREFIX,
    _GUARD_LIVELOCK_LIMIT as _GUARD_LIVELOCK_LIMIT,
    _GUARD_LIVELOCK_MIN_COUNT as _GUARD_LIVELOCK_MIN_COUNT,
    _GUARD_LIVELOCK_THRESHOLD as _GUARD_LIVELOCK_THRESHOLD,
    _GUARD_PARK_MARKER as _GUARD_PARK_MARKER,
    _GUARD_QUARANTINE_KIND as _GUARD_QUARANTINE_KIND,
    _GUARD_RECENT_LOOKBACK_MIN as _GUARD_RECENT_LOOKBACK_MIN,
    _GUARD_SPAWN_FAILED_MARKER as _GUARD_SPAWN_FAILED_MARKER,
    _GUARD_TERMINAL_KINDS as _GUARD_TERMINAL_KINDS,
    _HOOK_FAIL_LOOKBACK_MIN as _HOOK_FAIL_LOOKBACK_MIN,
    _SEAT_SPAWN_FAIL_RED_STREAK as _SEAT_SPAWN_FAIL_RED_STREAK,
    _SEAT_STREAK_LEDGER_GLOB as _SEAT_STREAK_LEDGER_GLOB,
    _SEAT_STREAK_LEDGER_PREFIX as _SEAT_STREAK_LEDGER_PREFIX,
    _SHIPS_PER_WORKER_GREP as _SHIPS_PER_WORKER_GREP,
    _SHIPS_PER_WORKER_LOOKBACK_MIN as _SHIPS_PER_WORKER_LOOKBACK_MIN,
    _SHIPS_PER_WORKER_SCHEMA as _SHIPS_PER_WORKER_SCHEMA,
    _SPAWN_CAUSE_LOOKBACK_MIN as _SPAWN_CAUSE_LOOKBACK_MIN,
    _SPAWN_FAILED_CAUSE_SCHEMA as _SPAWN_FAILED_CAUSE_SCHEMA,
    _SPAWN_STALE_CRED_RED_MIN_SPAWNS as _SPAWN_STALE_CRED_RED_MIN_SPAWNS,
    _SPAWN_STALE_CRED_RED_RATE as _SPAWN_STALE_CRED_RED_RATE,
    _WAVE_LOG_RE as _WAVE_LOG_RE,
    _WAVE_SPAWN_FAILED_GLOB as _WAVE_SPAWN_FAILED_GLOB,
    _auth_failed_seat_action as _auth_failed_seat_action,
    _backlog_anchor as _backlog_anchor,
    _classify_sched_task as _classify_sched_task,
    _double_booked_seat_action as _double_booked_seat_action,
    _guard_cause_from_tail as _guard_cause_from_tail,
    _guard_empty_cause as _guard_empty_cause,
    _guard_empty_cause_bit as _guard_empty_cause_bit,
    _guard_empty_log_cause as _guard_empty_log_cause,
    _guard_livelock_candidates as _guard_livelock_candidates,
    _guard_livelock_label as _guard_livelock_label,
    _guard_log_tail as _guard_log_tail,
    _guard_row_can_livelock as _guard_row_can_livelock,
    _guard_worker_log_index as _guard_worker_log_index,
    _limited_seat_labels as _limited_seat_labels,
    _parse_watchdog_query as _parse_watchdog_query,
    _recent_backend_products as _recent_backend_products,
    _run_ships_per_worker_git as _run_ships_per_worker_git,
    _sanitize_worker_token as _sanitize_worker_token,
    _sched_recovered as _sched_recovered,
    _seat_inventory_summary_line as _seat_inventory_summary_line,
    _seat_label as _seat_label,
    _total_commits as _total_commits,
    _watch_child_failures as _watch_child_failures,
    _watch_row_witness as _watch_row_witness,
    _wave_spawn_failed_events as _wave_spawn_failed_events,
    annotate_seat_inventory_from_preflight as annotate_seat_inventory_from_preflight,
    backend_hook_failures as backend_hook_failures,
    backend_stub_rates as backend_stub_rates,
    backlog_rates as backlog_rates,
    commit_drought as commit_drought,
    guard_coverage as guard_coverage,
    parse_ships_per_worker as parse_ships_per_worker,
    read_active_weekly_cap as read_active_weekly_cap,
    read_backend_health as read_backend_health,
    read_dispatch_progress as read_dispatch_progress,
    read_fleet_net_decline as read_fleet_net_decline,
    read_seat_inventory as read_seat_inventory,
    read_seat_spawn_failure_streaks as read_seat_spawn_failure_streaks,
    render_spawn_causes as render_spawn_causes,
    route_health_status as route_health_status,
    seat_spawn_fail_alarm as seat_spawn_fail_alarm,
    ships_per_worker as ships_per_worker,
    silent_workers as silent_workers,
    spawn_failed_cause_breakdown as spawn_failed_cause_breakdown,
    spawn_stale_cred_alarm as spawn_stale_cred_alarm,
    watch_decision as watch_decision,
    watchdog_installed as watchdog_installed,
)
from dispatch_status_render import (  # noqa: E402
    _card_capacity_lines as _card_capacity_lines,
    _card_flow_lines as _card_flow_lines,
    _card_gate_lines as _card_gate_lines,
    _card_header_lines as _card_header_lines,
    _card_health_lines as _card_health_lines,
    _card_lane_lease_lines as _card_lane_lease_lines,
    _card_launch_gate_lines as _card_launch_gate_lines,
    _card_lease_lines as _card_lease_lines,
    _card_planner_lines as _card_planner_lines,
    _card_route_lines as _card_route_lines,
    _card_watch_lines as _card_watch_lines,
    _card_yield_lines as _card_yield_lines,
    _dispatch_capacity_line as _dispatch_capacity_line,
    _dispatch_gate_slack_buckets as _dispatch_gate_slack_buckets,
    _dispatch_headline_state as _dispatch_headline_state,
    _dispatch_health_slack_buckets as _dispatch_health_slack_buckets,
    _dispatch_slack_buckets as _dispatch_slack_buckets,
    _dispatch_trend_line as _dispatch_trend_line,
    _join_limited as _join_limited,
    _lease_block_text as _lease_block_text,
    _lease_liveness_text as _lease_liveness_text,
    _lease_summary_bits as _lease_summary_bits,
    _md_admission_clause as _md_admission_clause,
    _md_backend_health_section as _md_backend_health_section,
    _md_backlog_section as _md_backlog_section,
    _md_capacity_bullets as _md_capacity_bullets,
    _md_closure_section as _md_closure_section,
    _md_contract_repair_section as _md_contract_repair_section,
    _md_guard_section as _md_guard_section,
    _md_hook_health_section as _md_hook_health_section,
    _md_lane_lease_section as _md_lane_lease_section,
    _md_lease_bullets as _md_lease_bullets,
    _md_lease_section as _md_lease_section,
    _md_low_yield_section as _md_low_yield_section,
    _md_resolver_bullets as _md_resolver_bullets,
    _md_run_state_bullets as _md_run_state_bullets,
    _md_ships_section as _md_ships_section,
    _md_silent_worker_section as _md_silent_worker_section,
    _md_summary_bullets as _md_summary_bullets,
    _md_throughput_section as _md_throughput_section,
    _md_utilization_bullets as _md_utilization_bullets,
    _md_watch_section as _md_watch_section,
    _md_worker_lease_section as _md_worker_lease_section,
    _rate_str as _rate_str,
    _resolve_tick_state as _resolve_tick_state,
    _worker_lease_bucket_bits as _worker_lease_bucket_bits,
    post_to_slack as post_to_slack,
    render as render,
    render_md as render_md,
    render_slack as render_slack,
    slack_text as slack_text,
)


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass


_configure_stdout()


_RESOLVE_TICK_SCHEMA = "fleet-resolve-ticks/1"
_RESOLVE_TICK_FRESH_MIN = 90.0
_UTILIZATION_SCHEMA = "fleet-utilization/1"

# Per-lane low-yield witness (#2062). `silent_workers` only catches a worker that
# produced NOTHING (a sub-floor log). It is blind to the costlier failure the fleet
# self-audit found: a worker that runs 44-94 turns, writes a full log, and still
# lands ZERO `Fixes #` commits — so `pick_lane` keeps re-seating that lane with no
# feedback. This fold binds turns-spent to ancestry-closes per lane over a recent
# window and flags a lane that burned >= the turn floor yet closed nothing on its
# own tree. The floor sits just below the observed 44-turn low end so a genuinely
# stuck lane surfaces while a short productive session does not. Informational
# only: it adds a reason/row but never flips the card's `ok`. The fold itself
# (low_yield_lanes / count_lane_ancestry_closes) and its constants now live in
# tools/lane_yield.py, re-imported above so the picker shares this exact evidence.


def run_json(cmd: list[str], cwd: Path, timeout: int,
             ok_codes: set[int] | None = None) -> dict[str, Any]:
    ok_codes = ok_codes if ok_codes is not None else set(range(0, 16))
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout, creationflags=_win_creationflags())
    except subprocess.TimeoutExpired:
        return {"_error": f"timed out after {timeout}s", "_cmd": cmd}
    except OSError as exc:
        return {"_error": str(exc), "_cmd": cmd}
    doc = _last_json(proc.stdout)
    doc.setdefault("_returncode", proc.returncode)
    if proc.returncode not in ok_codes and "_error" not in doc and not doc.get("schema"):
        doc["_error"] = (proc.stderr or proc.stdout or "").strip()[-300:]
    return doc


def git_path(root: Path, name: str) -> Path:
    try:
        proc = subprocess.run(["git", "rev-parse", "--git-path", name], cwd=root,
                              capture_output=True, text=True, timeout=5,
                              creationflags=_win_creationflags())
    except (subprocess.TimeoutExpired, OSError):
        return root / ".git" / name
    if proc.returncode != 0:
        return root / ".git" / name
    path = (proc.stdout or "").strip()
    return (root / path).resolve() if path else root / ".git" / name


def merge_state(root: Path) -> dict[str, Any]:
    merge_head = git_path(root, "MERGE_HEAD")
    present = False
    try:
        present = merge_head.exists() and bool(merge_head.read_text(encoding="utf-8").strip())
    except OSError:
        present = False
    out: dict[str, Any] = {
        "merge_in_progress": present,
        "merge_head": str(merge_head),
    }
    if present:
        out["next_action"] = (
            "wait for MERGE_HEAD to clear before starting new worker edits; "
            "partial path commits are unsafe while a peer merge is in progress")
    return out


def _lab_readiness_default_path() -> Path:
    env = os.environ.get("FAK_LAB_READINESS", "").strip()
    if env:
        return Path(env)
    if os.name == "nt":
        root = os.environ.get("APPDATA", "").strip()
        if root:
            return Path(root) / "fak" / "fleet" / "lab-readiness.json"
        return Path.home() / "AppData" / "Roaming" / "fak" / "fleet" / "lab-readiness.json"
    root = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(root) if root else Path.home() / ".config"
    return base / "fak" / "fleet" / "lab-readiness.json"


def _coarsen_legacy_status(status: str) -> tuple[str, str]:
    """Fold a legacy five-state status onto the (phase, detail) it maps to. Any
    status that is not the single ready state — including an unrecognized one —
    coarsens to WAITING, so a stale legacy record can never re-open the gate.
    Mirrors linkstate.Coarsen."""
    return {
        "READY_FOR_DEV_WORK": (LINK_CLEAR, DETAIL_READY),
        "WAIT_PRIVATE_RECOVERY": (LINK_WAITING, DETAIL_PRIVATE_RECOVERY),
        "GATEWAY_UNREACHABLE": (LINK_WAITING, DETAIL_GATEWAY_DOWN),
        "AUTH_OR_CHANNEL_BLOCKED": (LINK_WAITING, DETAIL_AUTH_BLOCKED),
    }.get(status, (LINK_WAITING, DETAIL_INDETERMINATE))


def _phase_for_detail(detail: str) -> str:
    if detail == DETAIL_READY:
        return LINK_CLEAR
    if detail == DETAIL_JOB_IN_FLIGHT:
        return LINK_WORKING
    return LINK_WAITING


def _lab_readiness_indeterminate(*, next_action: str, evidence: str,
                                 error: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": LINK_STATE_SCHEMA,
        "subject": "gpu-server",
        "checked_at": None,
        "phase": LINK_WAITING,
        "detail": DETAIL_INDETERMINATE,
        "next_action": next_action,
        "evidence": evidence,
        "admit_dispatch": False,
        "present": False,
        "valid": error is None,
        "commands": _lab_readiness_commands(),
    }
    if error:
        out["_error"] = error
    return out


def _lab_readiness_problem(doc: dict[str, Any]) -> str | None:
    unknown = sorted(set(doc) - LAB_READINESS_KEYS)
    if unknown:
        return "unknown field(s): " + ", ".join(unknown[:6])
    if doc.get("schema") not in (None, "", LINK_STATE_SCHEMA, LAB_READINESS_SCHEMA):
        return f"unsupported schema {doc.get('schema')!r}"
    # A native record carries `phase`; a legacy record carries `status`. Validate the
    # one that is present against its closed vocabulary.
    phase = str(doc.get("phase") or "")
    if phase:
        if phase not in LINK_PHASES:
            return f"unknown phase {phase!r}"
        detail = str(doc.get("detail") or "")
        if detail:
            if detail not in LINK_DETAILS:
                return f"unknown detail {detail!r}"
            if _phase_for_detail(detail) != phase:
                return f"detail {detail!r} inconsistent with phase {phase!r}"
    else:
        status = str(doc.get("status") or "")
        if status not in LEGACY_LAB_STATUSES:
            return f"unknown status {status!r}"
    subject = str(doc.get("subject") or doc.get("machine_class") or "")
    for key, value in (("subject", subject),
                       ("next_action", str(doc.get("next_action") or "")),
                       ("evidence", str(doc.get("evidence") or ""))):
        if not _GENERIC_TOKEN_RE.match(value):
            return f"{key} must be a generic token-like value"
    return None


def read_lab_readiness(path: Path | None = None) -> dict[str, Any]:
    """Read the public link-state dispatch gate.

    Accepts EITHER the native fak.link_state/v1 record or a legacy fak.lab_readiness/v1
    record (coarsened onto a phase for one rollover cycle), and normalizes both onto the
    phase shape. This intentionally does not run dgxbridge or print the source path. The
    private bridge publishes a scrubbed record; the status card only consumes that public
    yes/no gate (admit_dispatch, re-derived from the phase) and fails closed when it is
    absent or malformed.
    """
    p = path or _lab_readiness_default_path()
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _lab_readiness_indeterminate(
            next_action="publish-lab-readiness",
            evidence="no-readiness-record",
            error="missing readiness record",
        )
    except (OSError, ValueError) as exc:
        return _lab_readiness_indeterminate(
            next_action="fix-lab-readiness-record",
            evidence="invalid-readiness-record",
            error=str(exc),
        )
    if not isinstance(doc, dict):
        return _lab_readiness_indeterminate(
            next_action="fix-lab-readiness-record",
            evidence="invalid-readiness-record",
            error="readiness record is not an object",
        )
    problem = _lab_readiness_problem(doc)
    if problem:
        return _lab_readiness_indeterminate(
            next_action="fix-lab-readiness-record",
            evidence="invalid-readiness-record",
            error=problem,
        )
    # Normalize EITHER schema onto the native phase shape. A record is legacy when it
    # declares the legacy schema or simply carries no `phase` (a pre-migration mirror);
    # fold its old status onto a phase+detail. admit_dispatch is ALWAYS re-derived from
    # the phase, never trusted from the file.
    phase = str(doc.get("phase") or "")
    legacy = doc.get("schema") == LAB_READINESS_SCHEMA or not phase
    if legacy:
        phase, detail = _coarsen_legacy_status(str(doc.get("status") or ""))
    else:
        detail = str(doc.get("detail") or "") or _default_detail(phase)
    return {
        "schema": LINK_STATE_SCHEMA,
        "subject": str(doc.get("subject") or doc.get("machine_class") or "gpu-server"),
        "checked_at": doc.get("checked_at"),
        "phase": phase,
        "detail": detail,
        "next_action": str(doc.get("next_action") or ""),
        "evidence": str(doc.get("evidence") or ""),
        "admit_dispatch": phase == LINK_CLEAR,
        "present": True,
        "valid": True,
        "commands": _lab_readiness_commands(),
    }


def _resolve_tick_next_action(row: dict[str, Any]) -> str:
    gate = row.get("launch_gate") or {}
    blockers = gate.get("blockers") or []
    if blockers:
        action = blockers[0].get("next_action")
        if action:
            return str(action)
    verdict = str(row.get("verdict") or "")
    if verdict == "MULTI_LANE_SCOPE":
        return "split-issue-or-dispatch-under-covering-lane"
    if verdict == "ISSUE_CONTRACT_HOLD":
        return "backfill-issue-contract"
    if verdict == "LANE_BUSY":
        return "wait-for-lane-lease"
    if verdict == "SELF_MODIFY_HOLD":
        return "enable-worktree-isolated-resolver-or-route-non-self-source-lane"
    if verdict.startswith("REFUSE_NO_ACCOUNT"):
        return "wait-for-free-account"
    return ""


def _resolve_tick_rank(row: dict[str, Any]) -> tuple[int, float]:
    """Prefer the most actionable fresh planner row over the newest noisy row."""
    age = float(row.get("age_min") or 0.0)
    gate = row.get("launch_gate") or {}
    action = str(row.get("action") or "")
    verdict = str(row.get("verdict") or "")
    if (row.get("fresh") and gate.get("ready") is True
            and action in ("would_spawn", "would_repair")):
        return (0, age)
    if row.get("fresh") and action in ("spawned", "repair_spawned", "repair_in_flight"):
        return (1, age)
    if row.get("fresh") and gate.get("ready") is False:
        return (2, age)
    if row.get("fresh") and verdict.startswith("REFUSE_"):
        return (4, age)
    if row.get("fresh"):
        return (3, age)
    return (5, age)


def _resolve_tick_live_command(row: dict[str, Any]) -> list[str]:
    if not row or (row.get("launch_gate") or {}).get("ready") is not True:
        return []
    if row.get("action") not in ("would_spawn", "would_repair"):
        return []
    backend = row.get("backend") or "claude"
    max_workers = row.get("max_workers")
    cmd = [
        "python", "tools\\issue_resolve_dispatch.py",
        "--backend", str(backend),
    ]
    if max_workers is not None:
        cmd += ["--max-workers", str(max_workers)]
    work_kind = row.get("work_kind")
    if work_kind:
        cmd += ["--work-kind", str(work_kind)]
    if row.get("action") == "would_spawn":
        lane = row.get("lane")
        issue = row.get("target_issue")
        if not lane or issue is None:
            return []
        cmd += ["--lane", str(lane), "--issue", str(issue)]
    if row.get("action") == "would_repair":
        if row.get("contract_scan") is not None:
            cmd += ["--contract-scan", str(row.get("contract_scan"))]
        if row.get("repair_batch") is not None:
            cmd += ["--repair-batch", str(row.get("repair_batch"))]
    if row.get("force"):
        cmd.append("--force")
    cmd += ["--live", "--json"]
    return cmd


def read_resolve_ticks(root: Path, *, now_ts: float | None = None,
                       fresh_min: float = _RESOLVE_TICK_FRESH_MIN) -> dict[str, Any]:
    """Read last issue-resolver tick artifacts without re-running the planner.

    These files are evidence of the last scheduler/manual tick, not launch
    authorization. Freshness is explicit so dashboards can distinguish a current
    hold from a stale historical one.
    """
    runs = root / RUNS_DIRNAME
    now = time.time() if now_ts is None else now_ts
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in sorted(runs.glob("last-resolve-tick-*.json")):
        try:
            st = path.stat()
            doc = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(doc, dict):
                raise ValueError("tick is not an object")
        except (OSError, ValueError) as exc:
            errors.append({"path": path.name, "error": str(exc)})
            continue
        age_min = max(0.0, (now - st.st_mtime) / 60.0)
        gate = doc.get("launch_gate") if isinstance(doc.get("launch_gate"), dict) else {}
        blockers = gate.get("blockers") if isinstance(gate.get("blockers"), list) else []
        row = {
            "path": path.name,
            "backend": doc.get("backend") or path.stem.replace("last-resolve-tick-", ""),
            "verdict": doc.get("verdict"),
            "action": doc.get("action"),
            "ok": doc.get("ok"),
            "live": doc.get("live"),
            "max_workers": doc.get("max_workers"),
            "work_kind": doc.get("work_kind"),
            "force": bool(doc.get("force")),
            "contract_scan": doc.get("contract_scan"),
            "repair_batch": doc.get("repair_batch"),
            "lane": doc.get("lane"),
            "target_issue": doc.get("target_issue"),
            "reason": doc.get("reason"),
            # #4589: stop dropping the three drain/plateau signals the producer
            # stamps (issue_resolve_dispatch.py) — the seat ramp-cap binding term,
            # the spawn-failure streak, and the spawn-failure cause — so the status
            # card can re-surface WHY fan-out is pinned or spawns are failing.
            "seat_adaptive": doc.get("seat_adaptive") or {},
            "seat_selection": doc.get("seat_selection") or {},
            "spawn_failed_streak": doc.get("spawn_failed_streak"),
            # #4591: the SEAT-keyed run-length stamped alongside (int on a
            # spawn_failed tick; state dict on a seat_cooled tick).
            "seat_streak": doc.get("seat_streak"),
            "cause": doc.get("cause"),
            "contract_repair": doc.get("contract_repair") or {},
            "safe_lanes_busy": doc.get("safe_lanes_busy") or [],
            "self_modify_held": doc.get("self_modify_held") or [],
            "self_modify_held_report": doc.get("self_modify_held_report") or [],
            "self_modify_held_ticks": doc.get("self_modify_held_ticks") or {},
            "held_lanes": doc.get("held_lanes") or [],
            "launch_gate": {
                "ready": gate.get("ready"),
                "blockers": [
                    {
                        "code": b.get("code"),
                        "reason": b.get("reason"),
                        "next_action": b.get("next_action"),
                    }
                    for b in blockers[:4] if isinstance(b, dict)
                ],
            } if gate else {},
            "next_action": _resolve_tick_next_action(doc),
            "age_min": age_min,
            "fresh": age_min <= fresh_min,
        }
        live_command = _resolve_tick_live_command(row)
        if live_command:
            row["live_command"] = live_command
            row["live_command_text"] = " ".join(live_command)
        rows.append(row)
    rows.sort(key=lambda r: float(r.get("age_min") or 0.0))
    selected = sorted(rows, key=_resolve_tick_rank)[0] if rows else None
    return {
        "schema": _RESOLVE_TICK_SCHEMA,
        "fresh_min": fresh_min,
        "count": len(rows),
        "fresh_count": sum(1 for r in rows if r.get("fresh")),
        "latest": rows[0] if rows else None,
        "selected": selected,
        "ticks": rows,
        "errors": errors,
    }


def selected_resolver_preflight(root: Path, resolve_ticks: dict[str, Any],
                                *, max_workers: int) -> dict[str, Any]:
    """Run the spawn gate for the backend named by the selected resolver tick.

    The headline preflight is the generic/default dispatcher view. Non-default
    resolver ticks (notably opencode) can have their own account pool and live
    workers, so surface that product-scoped gate beside the planner evidence.
    """
    tick = (resolve_ticks or {}).get("selected") or (resolve_ticks or {}).get("latest") or {}
    backend = str(tick.get("backend") or "").strip()
    if not backend or not tick.get("fresh"):
        return {}
    tick_max = _int(tick.get("max_workers"), max_workers) or max_workers
    doc = run_json(
        [_py(), str(root / "tools" / "dispatch_preflight.py"),
         "--json", "--product", backend, "--max-workers", str(tick_max)],
        root,
        120,
    )
    if doc:
        doc["_backend"] = backend
    return doc


def _utilization_blocker(code: str, scope: str, next_action: str,
                         detail: str = "") -> dict[str, Any]:
    out = {"code": code, "scope": scope, "next_action": next_action}
    if detail:
        out["detail"] = detail
    return out


def utilization_state(*, live: int | None, cap: int | None, host_safe: bool,
                      pre_verdict: str | None, resolver: dict[str, Any],
                      resolver_preflight: dict[str, Any] | None = None,
                      lab_readiness: dict[str, Any],
                      weekly_cap: dict[str, Any] | None = None,
                      merge: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fold capacity + planner/lab admission into one utilization signal.

    ``dispatcher.verdict`` says whether the local host may grow. This says why
    available slots are, or are not, turning into useful background work.
    """
    headroom: int | None = None
    if live is not None and cap is not None:
        headroom = max(0, int(cap) - int(live))
    blockers: list[dict[str, Any]] = []
    next_actions: list[str] = []
    launch_command: list[str] = []
    launch_command_text = ""
    state = "UNKNOWN"

    def add_blocker(code: str, scope: str, action: str, detail: str = "") -> None:
        blockers.append(_utilization_blocker(code, scope, action, detail))
        if action and action not in next_actions:
            next_actions.append(action)

    preflight_blocker = _current_preflight_launch_blocker(pre_verdict, resolver_preflight)

    if headroom is None:
        state = "CAPACITY_UNKNOWN"
        add_blocker("CAPACITY_UNKNOWN", "host", "inspect-dispatch-preflight")
    elif headroom <= 0:
        state = "SATURATED"
    elif not host_safe:
        state = "HOST_BLOCKED"
        add_blocker("HOST_FLAGGED", "host", "inspect-or-reap-host-process")
    elif preflight_blocker:
        state = preflight_blocker["state"]
        add_blocker(preflight_blocker["code"], preflight_blocker["scope"],
                    preflight_blocker["action"],
                    f"{preflight_blocker['source']} {preflight_blocker['verdict']}")
    elif weekly_cap:
        state = "ACCOUNT_CAPPED"
        add_blocker("WEEKLY_CAPPED", "account", "wait-for-weekly-cap-reset")
    elif (merge or {}).get("merge_in_progress"):
        state = "EDIT_HELD"
        add_blocker("MERGE_IN_PROGRESS", "git",
                    (merge or {}).get("next_action") or "wait-for-merge-head-clear")
    else:
        latest = (resolver or {}).get("selected") or (resolver or {}).get("latest") or {}
        gate = latest.get("launch_gate") or {}
        if not latest:
            state = "HEADROOM_UNASSESSED"
            add_blocker("NO_RECENT_PLANNER_TICK", "planner",
                        "run-issue-resolve-dispatch-dry-run")
        elif not latest.get("fresh"):
            state = "HEADROOM_STALE_PLAN"
            add_blocker("STALE_PLANNER_TICK", "planner",
                        "run-issue-resolve-dispatch-dry-run",
                        f"age={_age_text(latest.get('age_min'))}")
        elif (pre_verdict == "SPAWN_OK"
              and str(latest.get("verdict") or "")
              in ("REFUSE_AT_CAP", "REFUSE_NO_ACCOUNT", "REFUSE_INSPECT")):
            state = "HEADROOM_STALE_PLAN"
            add_blocker(
                "STALE_PLANNER_REFUSAL", "planner",
                "run-issue-resolve-dispatch-dry-run",
                f"last={latest.get('verdict')} current={pre_verdict}")
        elif gate.get("ready") is True and latest.get("action") in ("would_spawn", "would_repair"):
            state = "HEADROOM_LAUNCH_READY"
            action = "approve-live-launch-or-enable-always-on-issue-dispatch"
            if latest.get("action") == "would_repair":
                state = "HEADROOM_REPAIR_READY"
                action = "approve-live-repair-or-enable-always-on-issue-dispatch"
            if action not in next_actions:
                next_actions.append(action)
            launch_command = list(latest.get("live_command") or [])
            launch_command_text = str(latest.get("live_command_text") or "")
        elif latest.get("action") in ("spawned", "repair_spawned"):
            state = "WORKER_STARTING"
        elif latest.get("action") == "repair_in_flight":
            state = "REPAIR_IN_FLIGHT"
        else:
            state = "HEADROOM_HELD"
            blockers_in_gate = gate.get("blockers") or []
            if blockers_in_gate:
                for b in blockers_in_gate[:4]:
                    add_blocker(str(b.get("code") or latest.get("verdict") or "PLANNER_HELD"),
                                "planner",
                                str(b.get("next_action")
                                    or latest.get("next_action")
                                    or "inspect-last-resolve-tick"),
                                str(b.get("reason") or ""))
            else:
                add_blocker(str(latest.get("verdict") or "PLANNER_HELD"),
                            "planner",
                            str(latest.get("next_action") or "inspect-last-resolve-tick"))

    if (lab_readiness or {}).get("schema") and not lab_readiness.get("admit_dispatch"):
        lab_commands = lab_readiness.get("commands") or {}
        add_blocker("LAB_READINESS_HELD", "lab",
                    str(lab_readiness.get("next_action") or "publish-lab-readiness"),
                    str(lab_commands.get("mark_clear")
                        or lab_readiness.get("phase") or LINK_WAITING))

    out = {
        "schema": _UTILIZATION_SCHEMA,
        "state": state,
        "worker_slots": {"live": live, "cap": cap, "headroom": headroom},
        "next_actions": next_actions,
        "blockers": blockers,
    }
    if launch_command:
        out["launch_command"] = launch_command
        out["launch_command_text"] = launch_command_text or " ".join(launch_command)
    return out


# ---------------------------------------------------------------------------
# Lane-lease holder liveness — the cross-check's missing scope (#5859).
# ---------------------------------------------------------------------------
#
# `cross_check_worker_leases` below reconciles LOCAL dispatch-worker sidecars
# against the `refs/fak/locks/*` leaseref store (`read_leaseref_records`). That
# store is NOT the lane-lease substrate. The exclusive lane leases every agent
# actually contends for live in the DOS kernel's WAL (`.dos/lane-journal.jsonl`)
# and are read back with `dos lease-lane live`. The two sets are disjoint by
# construction — on the reference tree `refs/fak/locks/*` held 2 `resolve-*`
# records while `dos lease-lane live` held 23 exclusive lane leases — so the card
# printed `lease chk : clean=N orphan-process=0 unmatched-live-lease=0` and rolled
# "worker/lease cross-check clean" into its summary while EVERY lane lease on the
# box was fenced by a process that no longer exists, `cmd/**` (the largest lane in
# the backlog) and `internal/modver/**` included, the oldest for 22 days. The
# scope was never widened past the resolver leases; this fold widens it.
#
# Liveness is judged the way the KERNEL judges it — TTL/heartbeat staleness as the
# PRIMARY evidence, the holder pid only as corroboration. The first cut of this fold
# had it backwards: it decided deadness from the recorded `(pid, proc_starttime)`
# alone, and that predicate cannot discriminate at all.
#
# A recorded lane-lease `pid` is an EPHEMERAL `dos lease-lane acquire` subprocess
# that journals the ACQUIRE and exits immediately (`dos/lane_lease.py:466-492`); the
# reservation it books is DESIGNED to outlive it (`acquire()` at `:453` says so in
# as many words). So a perfectly healthy, actively-held lease ALWAYS probes
# "dead" by that test — and it did: the card rendered `live=0 dead-holder=25`
# while several of those lanes were held by agents running at that instant, four of
# them acquired MINUTES earlier. In a 9-minute window five of the "dead" holders
# self-released their own leases. A signal that fires for 100% of the population
# separates nothing.
#
# The kernel's own live-set fold (`dos.lane_lease._lease_is_dead` / `_expire_dead`,
# what `pretool_sensor`, `decisions.py`, `dispatch_top` and `dos arbitrate` all read
# through `live_leases(expire_dead=True)`) gets the ordering right, and this fold now
# mirrors it:
#
#   (a) PRIMARY — WAL freshness. The lease's newest stamp (`heartbeat_at`, else
#       `acquired_at`) inside a grace window ⇒ LIVE, no matter what the pid does.
#       Past its own `ttl_minutes` (or the kernel's 50-minute backstop) plus that
#       grace ⇒ DEAD. This is time evidence the WAL carries itself.
#   (b) CORROBORATION — a confidently-dead holder pid on THIS host, which may only
#       speed the reclaim of a lease whose OBSERVED heartbeat has already gone
#       quiet. It may never, on its own, kill a fresh lease.
#   (c) Everything else is `unknown` — never `live`, never `dead`.
#
# One deliberate narrowing vs the kernel, in the fail-safe direction: the kernel
# lets (b) fire off `acquired_at` when no beat exists, but NOTHING on this fleet
# writes one (2871 journal entries: 107 ACQUIRE, 81 RELEASE, 2619 ENFORCE, 64
# REFUSE — and ZERO HEARTBEAT; 0 live records carry `heartbeat_at`). With no beat
# writer, "the heartbeat went quiet" degenerates back into "older than the grace"
# and (b) once again fires for every lease. So the card requires (b) to actually
# corroborate an OBSERVED beat; absent one, TTL expiry is the only positive death
# evidence, and a lease inside its TTL whose ephemeral acquirer has exited reads
# `unknown` — which is the normal, healthy shape, not a reapable orphan.
#
# The pid rung, when it does apply, still checks `(pid, proc_starttime)` and never
# bare pid existence — pids are recycled, and a recycled pid is exactly how a dead
# holder reads "alive". The reference tree carries a live instance: lane `guard`
# recorded pid 44396, and pid 44396 today is a `conhost.exe` that started ~47
# minutes AFTER the recorded start time.


# ---------------------------------------------------------------------------
# Final launch admission (#6495): the gate the LAUNCHER actually obeys.
# ---------------------------------------------------------------------------
#
# `dispatch_preflight` answers "is the host clean, is an account free, are we under
# the worker cap?". It does NOT ask the launch-rate/cooldown gate every launcher
# passes through (`tools/launch_admission.py`, #617). On 2026-08-11 that split let
# this card publish READY_TO_GROW while six nominally free seats produced ZERO
# launches: the one-worker dry-run granted 1, and the matching live launch was
# refused `LAUNCH_RATE_EXCEEDED` with a retry timestamp. A verdict that recommends
# growth the launcher will refuse is not actionable, so the card now folds the SAME
# final admission verdict the launcher folds -- per candidate account, including the
# per-account cooldown and the fleet-wide global cap -- and downgrades READY_TO_GROW
# to the typed wait verdict LAUNCH_COOLDOWN (with the retry time) whenever a
# one-worker launch would be deferred.

_LAUNCH_ADMISSION_SCHEMA = "fak.dispatch-launch-admission.v1"
_SPAWN_GATE_SCHEMA = "fak.dispatch-spawn-gate.v1"
_ACCOUNTS_SCHEMA = "fak.dispatch-accounts.v1"
_ADMIT = "ADMIT"


def _account_key(tag: Any) -> str:
    """Case/whitespace-insensitive account identity for matching and de-duping."""
    return str(tag or "").strip().lower()


def launch_candidate_accounts(seat_inventory: dict[str, Any] | None,
                              selected: str = "") -> list[dict[str, Any]]:
    """The accounts a ONE-worker launch could target right now: every seat holding a
    free slot, plus the account preflight itself picked (which need not appear in the
    pool at all on a single-account host). The switcher's pick leads the list.

    Each row carries the seat's cooldown state so the admission gate is asked the same
    throttle question the launcher asks (``cooling`` -> throttled, with its reset)."""

    def row(seat: dict[str, Any], tag: str) -> dict[str, Any]:
        cooldown = seat.get("cooldown") or {}
        return {
            "account": tag,
            "free_slots": _int(seat.get("free_slots")),
            "throttled": str(seat.get("dispatch_state") or "") == "cooling",
            "throttle_reset": cooldown.get("next_eligible") or cooldown.get("until"),
        }

    out: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for seat in (seat_inventory or {}).get("seats") or []:
        tag = str(seat.get("tag") or seat.get("account") or seat.get("seat") or "").strip()
        key = _account_key(tag)
        if not key or key in by_key:
            continue
        by_key[key] = seat
        if (_int(seat.get("free_slots"), 0) or 0) <= 0:
            continue  # no free slot -> not a launch target this tick
        out.append(row(seat, tag))
    sel = _account_key(selected)
    if sel:
        for i, cand in enumerate(out):
            if _account_key(cand["account"]) == sel:
                out.insert(0, out.pop(i))
                break
        else:
            # The switcher's pick is always asked, even when the pool shows it with no
            # free slot -- carrying ITS cooldown, so a throttled pick cannot read as
            # admitted just because it fell out of the free-seat scan.
            seat = by_key.get(sel)
            out.insert(0, row(seat, str(selected).strip()) if seat else
                       {"account": str(selected).strip(), "free_slots": None,
                        "throttled": False, "throttle_reset": None})
    return out


def _earliest_retry(retries: list[Any]) -> Any:
    """The soonest retry marker among a set of DEFERs. Parseable ISO stamps win (the
    operator can wait on them); an unparseable marker (a throttle reset like "3pm" is
    surfaced verbatim by the gate) is kept only when nothing parseable exists."""
    dated = [(e, r) for r in retries
             if r and (e := _iso_epoch(r)) is not None]
    if dated:
        return min(dated, key=lambda p: p[0])[1]
    return next((r for r in retries if r), None)


def launch_admission_fold(rows: list[dict[str, Any]], *,
                          selected: str = "") -> dict[str, Any]:
    """Pure fold: per-account admission verdicts -> the card's final launch gate.

    ``would_launch`` is True iff at least ONE candidate account is admitted — that is
    exactly the question "could a one-worker launch fire right now?". When nothing is
    admitted the fold carries the blocking reason token (the switcher's own account
    first, else the first deferral) and the soonest retry time, so the card can say
    LAUNCH_COOLDOWN <token> until <ts> instead of READY_TO_GROW."""
    rows = list(rows or [])
    admitted = [r for r in rows if str(r.get("verdict") or "") == _ADMIT]
    deferred = [r for r in rows if str(r.get("verdict") or "") != _ADMIT]
    sel = _account_key(selected)
    lead = next((r for r in deferred if _account_key(r.get("account")) == sel),
                deferred[0] if deferred else {})
    return {
        "schema": _LAUNCH_ADMISSION_SCHEMA,
        "evaluated": bool(rows),
        "would_launch": bool(admitted),
        "selected": str(selected).strip() or None,
        "verdict": _ADMIT if admitted else "DEFER",
        "reason": None if admitted else (lead.get("reason") or None),
        "retry_after": None if admitted else _earliest_retry(
            [r.get("retry_after") for r in deferred]),
        "detail": (lead.get("detail") or "") if not admitted else "",
        "admitted_accounts": [r.get("account") for r in admitted],
        "deferred_accounts": [
            {"account": r.get("account"), "reason": r.get("reason"),
             "retry_after": r.get("retry_after")}
            for r in deferred
        ],
        "accounts": rows,
    }


def read_launch_admission(root: Path, pre: dict[str, Any],
                          seat_inventory: dict[str, Any] | None = None,
                          *, now: Any = None, ledger_path: str | None = None,
                          admitter: Any | None = None) -> dict[str, Any]:
    """Ask the LAUNCHER's own admission gate whether one more worker would launch.

    Delegates to ``launch_admission.admit_or_defer`` (the single #617 seam every
    launcher passes through) once per candidate account, with ``record=False`` so the
    question has no side effect on the ledger. The card therefore grades growth on the
    SAME three gates the next launch gets: the account throttle/cooldown reset, the
    fleet-wide global cap, and the per-account rate ceiling.

    Pure-local ledger read (no gh, no subprocess), so ``--fast`` keeps it. Best effort:
    an import/read failure degrades to ``{"_error": ...}`` like the sibling card folds
    and never raises — but, unlike them, an un-consultable gate is NOT read as a green
    light (see ``_apply_launch_admission``). ``admitter`` is injectable for hermetic
    tests (same signature as the closure built here)."""
    selected = str(((pre or {}).get("account") or {}).get("tag") or "").strip()
    candidates = launch_candidate_accounts(seat_inventory, selected)
    if not candidates:
        return {"schema": _LAUNCH_ADMISSION_SCHEMA, "evaluated": False,
                "would_launch": None, "selected": selected or None,
                "detail": "no candidate account to launch onto"}
    try:
        if admitter is None:
            sys.path.insert(0, str(root / "tools"))
            import launch_admission  # noqa: PLC0415  (lazy: only paid for on the card)
            ledger = ledger_path or launch_admission.DEFAULT_LEDGER

            def admitter(account, *, throttled=False, throttle_reset=None):  # noqa: F811
                return launch_admission.admit_or_defer(
                    account, ledger_path=ledger, now=now, throttled=throttled,
                    throttle_reset=throttle_reset, record=False)

        rows: list[dict[str, Any]] = []
        for cand in candidates:
            v = admitter(cand["account"], throttled=bool(cand.get("throttled")),
                         throttle_reset=cand.get("throttle_reset")) or {}
            rows.append({
                "account": cand["account"],
                "free_slots": cand.get("free_slots"),
                "verdict": v.get("verdict"),
                "reason": v.get("reason"),
                "retry_after": v.get("retry_after"),
                "detail": v.get("detail"),
            })
    except Exception as exc:  # best-effort card fold; never fail the whole card
        return {"schema": _LAUNCH_ADMISSION_SCHEMA, "evaluated": False,
                "would_launch": None, "selected": selected or None,
                "_error": str(exc)}
    return launch_admission_fold(rows, selected=selected)


def _launch_evidence(silent: list[dict[str, Any]],
                     spawn_causes: dict[str, Any]) -> dict[str, Any]:
    """The trailing evidence that a REPORTED launch did not become a worker (#6492):
    workers whose pid is dead over a log that never crossed the real-turn floor —
    the immediate-exit / false-success signature — beside the spawn counters that
    reported those same launches as successful. Carried on the spawn gate so a
    growth recommendation is never read without the last wave's actual outcome."""
    return {
        "immediate_exits": len(silent),
        "issues": [w.get("issue") for w in silent[:6]],
        "spawns": spawn_causes.get("spawns"),
        "spawn_failed": spawn_causes.get("spawn_failed"),
        "note": ("immediate_exits are dead-pid workers whose log never crossed the "
                 "real-turn floor; a nonzero count beside spawn_failed=0 is the "
                 "#6492 false-success signature"),
    }


def _spawn_gate_block(pre_verdict: Any, admission: dict[str, Any] | None,
                      *, verdict: str, next_action: str,
                      evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """The card's spawn-gate block: BOTH halves of the real spawn decision — the
    preflight (host/account/cap) and the launcher's final admission gate — plus the
    single derived answer "would one more worker actually launch right now?"."""
    admission = admission or {}
    would = admission.get("would_launch")
    preflight_ok = str(pre_verdict or "") == "SPAWN_OK"
    return {
        "schema": _SPAWN_GATE_SCHEMA,
        "preflight_verdict": pre_verdict,
        "admission": admission,
        "would_launch": (bool(would) and preflight_ok) if would is not None else None,
        "reason": (None if would else admission.get("reason")) if admission else None,
        "retry_after": admission.get("retry_after") if not would else None,
        "verdict": verdict,
        "next_action": next_action,
        "launch_evidence": evidence or {},
    }


def _accounts_block(seat_inventory: dict[str, Any] | None, acct: dict[str, Any],
                    admission: dict[str, Any] | None) -> dict[str, Any]:
    """The card's accounts block: the seat pool joined to the FINAL admission verdict
    per account, so "N free seats" is never mistaken for "N launchable seats" (the
    #6495 gap — six free seats, zero admitted launches)."""
    seat_inventory = seat_inventory or {}
    admission = admission or {}
    by_account = {_account_key(r.get("account")): r
                  for r in (admission.get("accounts") or [])}
    admitted = [r for r in by_account.values()
                if str(r.get("verdict") or "") == _ADMIT]
    launchable = [r.get("account") for r in admitted]
    # In SLOTS, so it is comparable to free_seats: an admitted account contributes its
    # free slots (the switcher's pick, which need not be a pool seat at all, contributes
    # the one launch we actually asked about).
    launchable_slots = 0
    for row in admitted:
        slots = _int(row.get("free_slots"))
        launchable_slots += 1 if slots is None else max(0, slots)
    return {
        "schema": _ACCOUNTS_SCHEMA,
        "selected": {k: acct.get(k) for k in ("tag", "tier", "model", "available")},
        "total_seats": seat_inventory.get("total_seats"),
        "free_seats": seat_inventory.get("free_seats"),
        "leased_seats": seat_inventory.get("leased_seats"),
        "blocked_seats": seat_inventory.get("blocked_seats"),
        "launchable_seats": None if not admission.get("evaluated") else launchable_slots,
        "launchable_accounts": None if not admission.get("evaluated") else len(launchable),
        "launchable": launchable,
        "deferred": admission.get("deferred_accounts") or [],
        "admission_evaluated": bool(admission.get("evaluated")),
        "admission_error": admission.get("_error"),
    }


def _apply_launch_admission(admission: dict[str, Any] | None, *, ok: bool,
                            verdict: str, reasons: list[str]) -> tuple[bool, str]:
    """Fold the launcher's final admission verdict into the card verdict (#6495).

    Only the GROWTH verdict is rewritten: READY_TO_GROW is the one that tells an
    operator (or an autonomous planner) to launch, so it is the one that must not
    outrun the launcher. A DEFER makes it the typed wait verdict LAUNCH_COOLDOWN; an
    un-consultable gate makes it LAUNCH_ADMISSION_UNKNOWN. Both stay ``ok`` — a
    cooldown is a healthy steady state like AT_CAP, not breakage. Every other verdict
    (already a hold or a failure) keeps its more specific token and only gains the
    reason line."""
    if not admission:
        return ok, verdict
    if admission.get("_error"):
        if verdict == "READY_TO_GROW":
            reasons.insert(0, "final launch admission gate could not be consulted "
                              f"({admission.get('_error')}) — growth UNPROVEN; run "
                              "`python tools/launch_admission.py admit --account <tag>`")
            return ok, VERDICT_ADMISSION_UNKNOWN
        return ok, verdict
    if not admission.get("evaluated") or admission.get("would_launch"):
        return ok, verdict
    token = admission.get("reason") or "LAUNCH_DEFERRED"
    retry = admission.get("retry_after") or "?"
    deferred = admission.get("deferred_accounts") or []
    reasons.insert(0, f"final launch admission would DEFER every candidate account "
                      f"({len(deferred)}): {token} — retry after {retry}")
    if verdict == "READY_TO_GROW":
        return ok, VERDICT_LAUNCH_COOLDOWN
    return ok, verdict


def collect(root: Path, *, max_workers: int, fast: bool,
            closure_commits: int) -> dict[str, Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        pre_f = pool.submit(
            run_json,
            [_py(), str(root / "tools" / "dispatch_preflight.py"),
             "--json", "--max-workers", str(max_workers)],
            root,
            120,
        )
        sup_f = pool.submit(
            run_json,
            [_py(), str(root / "tools" / "dos_supervisor_status.py"), "--json"],
            root,
            90,
        )
        wd_f = pool.submit(watchdog_installed)
        backlog_f = None if fast else pool.submit(
            run_json,
            [_py(), str(root / "tools" / "issue_lane_router.py"), "--json"],
            root,
            130,
        )
        pre = pre_f.result()
        sup = sup_f.result()
        wd = wd_f.result()
        backlog: dict[str, Any] = (
            {"_skipped": "fast"} if fast else backlog_f.result())
    # Cover the WHOLE repo, not a slice. A closure audit whose --max-commits is
    # narrower than the repo's history can't bind a resolving commit older than the
    # window, so a long-since-shipped issue mis-buckets CLAIMED_CLOSED and the
    # closure_rate reads catastrophically low (the 0.20-vs-0.79 artifact). The
    # auditor caches every SHA verdict permanently, so a full-history window is
    # cheap on warm runs; we size it to the repo + headroom (never below the
    # operator's floor) and lift the issue limit above the real backlog so the
    # oldest -- disproportionately closed -- issues all load.
    total_commits = _total_commits(root)
    commit_window = max(closure_commits,
                        (total_commits + 200) if total_commits else closure_commits)
    # Low-yield lanes (#2062): bind turns-spent to ancestry-closes per lane over the
    # recent window. The ancestry-close join is git, so it lives here in the impure
    # collect layer; the fold itself stays pure over this counter. Lane trees come
    # from the router's lane→tree map (backlog), with each worker's lease-tree
    # sidecar as fallback inside the fold.
    low_yield_now = time.time()
    low_yield_since_iso = time.strftime(
        "%Y-%m-%dT%H:%M:%S +0000",
        time.gmtime(low_yield_now - _LOW_YIELD_LOOKBACK_MIN * 60))
    lane_trees = {
        str(ln): _clean_tree((info or {}).get("tree"))
        for ln, info in (
            (backlog.get("lanes") or {}) if isinstance(backlog, dict) else {}).items()
        if isinstance(info, dict) and _clean_tree((info or {}).get("tree"))
    }

    def _lane_closes(_lane: str, tree: list[str]) -> int | None:
        return count_lane_ancestry_closes(root, tree, since_iso=low_yield_since_iso)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        closure_f = None if fast else pool.submit(
            run_json,
            [_py(), str(root / "tools" / "issue_closure_audit.py"), "--json",
             "--max-commits", str(commit_window), "--issue-limit", "4000"],
            root,
            300,
        )
        throughput_f = None if fast else pool.submit(
            run_json,
            [_py(), str(root / "tools" / "dispatch_throughput.py"), "--json"],
            root,
            140,
            set(range(0, 16)),
        )
        silent_f = pool.submit(silent_workers, root / RUNS_DIRNAME)
        weekly_cap_f = pool.submit(
            read_active_weekly_cap,
            root / RUNS_DIRNAME,
            (pre.get("account") or {}).get("tag"),
        )
        backend_health_f = pool.submit(read_backend_health, root / RUNS_DIRNAME)
        backend_stub_rate_f = pool.submit(backend_stub_rates, root / RUNS_DIRNAME)
        hook_failures_f = pool.submit(backend_hook_failures, root / RUNS_DIRNAME)
        guard_f = pool.submit(guard_coverage, root / RUNS_DIRNAME)
        # Spawn-failed cause mix (#4590): the SAME read-only disk fold --spawn-causes
        # emits, now folded into the default card so a rising stale_cred rate reddens
        # instead of hiding behind the sub-flag. Pure-local, so --fast keeps it.
        spawn_causes_f = pool.submit(spawn_failed_cause_breakdown, root / RUNS_DIRNAME)
        # Seat-keyed spawn-failure streaks (#4591): the per-seat consecutive-fail
        # run-lengths the dispatcher persists; joined with seat inventory +
        # stale_cred below so ONE dead seat cycling across issues reddens the card.
        seat_streaks_f = pool.submit(read_seat_spawn_failure_streaks,
                                     root / RUNS_DIRNAME)
        # Net-worker-decline alarm (#4591 part 2): trailing `live` appends from
        # the #4594 fleet-status-history ledger, red on M consecutive declines.
        fleet_decline_f = pool.submit(read_fleet_net_decline, root)
        run_status_f = pool.submit(read_run_status_digests, root)
        merge_f = pool.submit(merge_state, root)
        leases_f = pool.submit(read_lease_state, root, backlog)
        # The KERNEL's exclusive lane leases (#5859) — a different substrate from the
        # `refs/fak/locks/*` set `read_lease_state` folds, and the one the whole fleet
        # actually contends for. Without it the cross-check reported "clean" while 18
        # lanes (`cmd/**` included) were fenced by processes that no longer existed.
        # Local `dos` + a local process probe, no gh, so --fast keeps it.
        lane_leases_f = pool.submit(lane_lease_liveness, root)
        seat_inventory_f = pool.submit(read_seat_inventory, root)
        lab_readiness_f = pool.submit(read_lab_readiness)
        resolve_ticks_f = pool.submit(read_resolve_ticks, root)
        low_yield_f = pool.submit(
            low_yield_lanes, root / RUNS_DIRNAME,
            closes_counter=_lane_closes, lane_trees=lane_trees, now_ts=low_yield_now)
        # Ships-per-worker attribution (#2065): read the best-effort (fak-worker <id>)
        # trailers back out of git. Pure-local git read, cheap, so --fast keeps it.
        ships_f = pool.submit(ships_per_worker, root)

        closure: dict[str, Any] = (
            {"_skipped": "fast"} if fast else closure_f.result())
    # The RATE fold (closed/hour vs target) — the observable the loop's goal is
    # actually stated in. gh-backed, so it degrades to n/a under --fast/timeout
    # exactly like backlog/closure; it never flips the dispatcher-health verdict
    # (a below-target rate is information, not a broken dispatcher).
        throughput: dict[str, Any] = (
            {"_skipped": "fast"} if fast else throughput_f.result())

        # Pure-local, always run (no gh/dos): which spawned workers exited producing
        # nothing. Cheap enough that --fast keeps it.
        silent = silent_f.result()
        weekly_cap = weekly_cap_f.result()
        backend_health = backend_health_f.result()
        backend_stub_rate = backend_stub_rate_f.result()
        hook_failures = hook_failures_f.result()
        guard = guard_f.result()
        spawn_causes = spawn_causes_f.result()
        seat_streaks = seat_streaks_f.result()
        fleet_decline = fleet_decline_f.result()
        run_status = run_status_f.result()
        merge = merge_f.result()
        leases = leases_f.result()
        worker_leases = worker_lease_crosscheck(root / RUNS_DIRNAME, leases,
                                                lane_leases=lane_leases_f.result())
        seat_inventory = annotate_seat_inventory_from_preflight(
            seat_inventory_f.result(), pre)
        lab_readiness = lab_readiness_f.result()
        resolve_ticks = resolve_ticks_f.result()
        resolver_preflight = selected_resolver_preflight(
            root, resolve_ticks, max_workers=max_workers)
        low_yield = low_yield_f.result()
        ships = ships_f.result()

    # The launcher's FINAL admission gate (#6495), asked with no side effect once the
    # seat pool is known. Pure-local ledger read (no gh, no subprocess), so --fast and
    # the full card fold the SAME gate the next launch will pass through.
    launch_admission_state = read_launch_admission(root, pre, seat_inventory)

    # Watch decision (#2642): explain WHY the health-watch (no-)acts, folded from
    # the trailing window of progress rows. Pure-local read, informational only —
    # never launches work or changes caps. The scheduled-task classification
    # consumes the watchdog query's Last Result plus a progress-window recovery
    # witness (self-healing vs unresolved, question 2); follow-ups are listed when
    # a caller supplies them.
    now_ts = time.time()
    progress_records = read_dispatch_progress(root)
    watch = watch_decision(
        progress_records, now_ts=now_ts,
        sched_task={"installed": wd.get("installed"), "status": wd.get("status"),
                    "last_result": wd.get("last_result"),
                    "recovered": _sched_recovered(progress_records, now_ts=now_ts)})
    # Backlog arrival-vs-service rate (#2634): numeric supply/demand meter folded
    # from the same trailing progress window. Pure-local read, informational only.
    backlog_rate = backlog_rates(progress_records, now_ts=now_ts)
    # Selected-route smoke gate (#3035): last probe age, typed failure class,
    # cooldown remaining, and the exact recheck command per route. Pure-local
    # ledger read, so --fast keeps it.
    route_health = route_health_status(root / RUNS_DIRNAME, now_ts=now_ts)

    return build_payload(root=root, pre=pre, sup=sup, wd=wd, backlog=backlog,
                         closure=closure, max_workers=max_workers, fast=fast,
                         silent=silent, weekly_cap=weekly_cap, throughput=throughput,
                         backend_health=backend_health,
                         backend_stub_rate=backend_stub_rate,
                         hook_failures=hook_failures, guard=guard,
                         run_status=run_status, merge=merge, leases=leases,
                         worker_leases=worker_leases, seat_inventory=seat_inventory,
                         lab_readiness=lab_readiness, resolve_ticks=resolve_ticks,
                         resolver_preflight=resolver_preflight,
                         low_yield=low_yield, ships=ships, watch=watch,
                         backlog_rate=backlog_rate, spawn_causes=spawn_causes,
                         seat_streaks=seat_streaks, fleet_decline=fleet_decline,
                         route_health=route_health,
                         launch_admission=launch_admission_state)


# ---------------------------------------------------------------------------
# Watch decision (#2642): emit WHY a long health-watch intentionally no-ops.
# ---------------------------------------------------------------------------
#
# The overnight ef59064f dispatch-health watch correctly stayed hands-off on a
# healthy-but-busy fleet by comparing raw counters against independent witnesses
# instead of treating every scary surface as an operator action: closures kept
# advancing (closed_by_loop_total 772 -> 791) while the backlog stayed high
# because new arrivals outpaced service. That reasoning lived only in an
# operator's head. This pure fold turns a trailing window of progress rows into a
# compact, cited verdict so the "intentionally no-op" case is emitted, not
# re-derived by hand each watch. Informational ONLY: it never launches work or
# changes worker caps (issue Done condition).


def _backlog_block(backlog: dict[str, Any]) -> dict[str, Any]:
    """The card's backlog block: open/routed/unrouted counts and the per-lane
    fold. `counts` is the router's authoritative routed/unrouted fold; a
    skipped or errored gh fold reports `na` instead of a false zero."""
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
    return {
        "na": backlog_na,
        "open_issues": None if backlog_na else open_issues,
        "routed": None if backlog_na else routed,
        "by_lane": None if backlog_na else lane_counts,
        "unrouted": None if backlog_na else unrouted,
    }


def _closure_block(closure: dict[str, Any]) -> dict[str, Any]:
    """The card's closure-honesty block: the strict diff-witnessed closure rate,
    the DATA-crediting honest rate, and the bucket counts behind them."""
    counts = closure.get("counts") or {}
    closure_rate = closure.get("closure_rate")
    honest_close_rate = closure.get("honest_close_rate")
    closure_na = "_skipped" in closure or ("_error" in closure and closure_rate is None)
    open_witnessed = _int(counts.get("OPEN_WITNESSED"), 0)
    return {
        "na": closure_na,
        "closure_rate": closure_rate,
        "honest_close_rate": honest_close_rate,
        "counts": counts or None,
        "open_witnessed_closable": None if closure_na else open_witnessed,
    }


def _throughput_block(throughput: dict[str, Any], *, tp_na: bool) -> dict[str, Any]:
    """The card's throughput block: completed-per-hour against target over the
    primary window, plus the per-window gh and loop-attributed series."""
    return {
        "na": tp_na,
        "verdict": None if tp_na else throughput.get("verdict"),
        "target_per_hour": None if tp_na else throughput.get("target_per_hour"),
        "primary_window_hours": None if tp_na else throughput.get("primary_window_hours"),
        "completed_rate_per_hour": None if tp_na else throughput.get("completed_rate_per_hour"),
        "raw_rate_per_hour": None if tp_na else throughput.get("raw_rate_per_hour"),
        "per_window": None if tp_na else (throughput.get("gh") or {}).get("per_window"),
        "loop_per_window": None if tp_na else (throughput.get("loop") or {}).get("per_window"),
        "last_loop_close_age_min": None if tp_na else (throughput.get("loop") or {}).get("last_loop_close_age_min"),
    }


def _supervisor_block(sup: dict[str, Any]) -> dict[str, Any]:
    """The card's supervisor block: the supervise verdict and how many of the
    targeted always-on plans are actually alive."""
    return {
        "verdict": sup.get("verdict"),
        "target": (sup.get("supervise") or {}).get("target"),
        "alive": (sup.get("supervise") or {}).get("alive"),
        "plans": sup.get("plans"),
    }


def _spawn_causes_block(spawn_causes: dict[str, Any],
                        spawn_alarm: dict[str, Any]) -> dict[str, Any]:
    """The card's spawn-cause block: the trailing SPAWN_FAILED cause mix (#4590)
    and the stale_cred drain alarm derived from it."""
    return {
        "na": not bool(spawn_causes.get("schema")),
        "schema": spawn_causes.get("schema"),
        "spawns": spawn_causes.get("spawns"),
        "spawn_failed": spawn_causes.get("spawn_failed"),
        "rate": spawn_causes.get("rate"),
        "by_cause": spawn_causes.get("by_cause") or {},
        "stale_cred_alarm": spawn_alarm,
    }


def _git_block(merge: dict[str, Any]) -> dict[str, Any]:
    """The card's git block: whether the checkout is parked mid-merge and the
    next action that clears it."""
    return {
        "merge_in_progress": bool(merge.get("merge_in_progress")),
        "merge_head": merge.get("merge_head"),
        "next_action": merge.get("next_action"),
    }


def _dispatcher_block(*, cap: int | None, live: int | None, host_safe: bool,
                      pre_verdict: Any, limiter: dict[str, Any],
                      acct: dict[str, Any], wd: dict[str, Any]) -> dict[str, Any]:
    """The card's dispatcher block: capacity, gate, account, and watchdog state."""
    return {
        "cap": cap,
        "live": live,
        "headroom": (cap - live) if (cap is not None and live is not None) else None,
        "host_safe": host_safe,
        "preflight_verdict": pre_verdict,
        "limiter": limiter,
        "account": {k: acct.get(k) for k in ("tag", "tier", "model", "available")},
        "watchdog": wd,
    }


# Preflight refusals the card has a SPECIFIC name for. Anything else still holds
# (BLOCKED_ON_PREFLIGHT, with the raw token in the reason) — never READY_TO_GROW.
_PREFLIGHT_HOLD_VERDICT = {
    "REFUSE_WEEKLY_CAPPED": "WEEKLY_CAPPED",
    "REFUSE_BIN_SKEW": "BLOCKED_ON_BIN_SKEW",
}


def _dispatch_base_verdict(*, pre: dict, cap: int | None, live: int | None,
                           host_safe: bool, acct: dict[str, Any],
                           weekly_cap: dict[str, Any] | None,
                           merge: dict[str, Any]) -> tuple[bool, str, list[str]]:
    """The card's verdict before any drain alarm: (ok, verdict, reasons).
    Healthy = host clean AND (can grow OR already at a healthy target). A flagged
    host or an un-runnable safety check is the only thing that fails the card —
    "no account free" / "at cap" are normal steady states, not breakage."""
    pre_verdict = pre.get("verdict")
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
    elif pre_verdict == "REFUSE_NO_SEAT":
        ok = True
        verdict = "BLOCKED_ON_SEAT"
        reasons.append("no dispatch seat free right now (seat pool will resume when one frees)")
    elif pre_verdict == "REFUSE_AT_CAP":
        ok = True
        verdict = "AT_CAP"
        reasons.append(f"{live}/{cap} workers live — at the configured ceiling")
    elif pre_verdict == "SPAWN_OK":
        ok = True
        verdict = "READY_TO_GROW"
        reasons.append(f"safe to spawn: {live}/{cap} live, account '{acct.get('tag')}' free")
    else:
        # Every OTHER preflight token is still a REFUSAL to spawn — `REFUSE_BIN_SKEW`
        # (#6508), `REFUSE_HOST`, `REFUSE_WEEKLY_CAPPED`, or a token added tomorrow.
        # Falling through to READY_TO_GROW on an unmodelled refusal is the #6495 failure
        # in its purest form: the card advertised growth on a gate that had already said
        # no. Unknown tokens therefore HOLD (naming the token), and a missing verdict
        # (the preflight itself could not be read) holds too rather than assuming yes.
        ok = True
        verdict = _PREFLIGHT_HOLD_VERDICT.get(str(pre_verdict or ""),
                                              "BLOCKED_ON_PREFLIGHT")
        if pre_verdict:
            reasons.append(f"preflight refuses to spawn: {pre_verdict} — "
                           f"{pre.get('reason') or 'run tools/dispatch_preflight.py --json'}")
        else:
            reasons.append("no preflight verdict — cannot confirm a spawn is safe "
                           "(run `python tools/dispatch_preflight.py --json`)")

    # A logged-in-but-quota-capped account makes the preflight read SPAWN_OK while
    # the dispatcher's weekly-cap gate is actually HOLDING. Surface that so the card
    # says WEEKLY_CAPPED, not the misleading READY_TO_GROW. Holding is a healthy
    # steady state (the t2 glm pool is unaffected), so ok stays True.
    if weekly_cap and verdict == "READY_TO_GROW":
        verdict = "WEEKLY_CAPPED"
        reasons = [f"account '{acct.get('tag')}' weekly-capped — resets "
                   f"{weekly_cap.get('reset_text') or '?'} (holding spawn until "
                   f"{weekly_cap.get('until')}); the t2 glm/docs pool is unaffected"]

    if merge.get("merge_in_progress"):
        ok = False
        verdict = "MERGE_IN_PROGRESS"
        reasons.insert(0, merge.get("next_action") or
                       "wait for MERGE_HEAD to clear before starting worker edits")
    return ok, verdict, reasons


def _apply_red_alarm(alarm: dict[str, Any], *, ok: bool, verdict: str,
                     reasons: list[str], red_verdict: str,
                     default_reason: str = "") -> tuple[bool, str]:
    """Fold one red-alarm record into the card verdict. A red alarm flips a
    healthy verdict to `red_verdict`; an already-failing verdict (host flagged /
    merge in progress) is the more urgent one and is kept, with only the reason
    added. The alarm's reason always leads the reason list."""
    if not alarm.get("red"):
        return ok, verdict
    if ok:
        ok = False
        verdict = red_verdict
    reasons.insert(0, str(alarm.get("reason") or default_reason))
    return ok, verdict


def _watchdog_reasons(wd: dict[str, Any]) -> list[str]:
    """Whether the always-on dispatch watchdog is registered on this host."""
    reasons: list[str] = []
    if wd.get("installed") is False:
        reasons.append("always-on watchdog NOT installed (register_dos_dispatch_watchdog.ps1)")
    elif wd.get("installed"):
        reasons.append(f"always-on watchdog installed ({wd.get('status') or 'scheduled'})")
    return reasons


def _silent_worker_reasons(silent: list[dict[str, Any]]) -> list[str]:
    """The workers that exited producing nothing — inspect or re-scope."""
    reasons: list[str] = []
    if silent:
        nums = ", ".join(f"#{w['issue']}" for w in silent[:6])
        reasons.append(f"{len(silent)} worker(s) exited producing nothing ({nums}) — inspect or re-scope")
    return reasons


def _backend_health_reasons(backend_health: list[dict[str, Any]],
                            backend_stub_rate: list[dict[str, Any]]) -> list[str]:
    """Backends held dead (lane reallocated) and backends whose recent logs are
    majority stub. Informational: a healthy backend covers the freed lane."""
    reasons: list[str] = []
    if backend_health:
        names = ", ".join(f"{b.get('product')}->{b.get('abandoned_lane') or '?'}"
                          for b in backend_health[:4])
        reasons.append(f"{len(backend_health)} backend(s) held dead, lane reallocated "
                       f"({names}) — a healthy backend is covering; auto-restores on recovery")
    majority_stub = [r for r in backend_stub_rate if r.get("majority_stub")]
    if majority_stub:
        names = ", ".join(f"{r.get('product')} {r.get('stub')}/{r.get('total')} stub"
                          for r in majority_stub[:4])
        reasons.append(f"backend stub-rate majority-stub over recent logs ({names}) — inspect backend output")
    return reasons


def _hook_health_reasons(unhooked: list[dict[str, Any]]) -> list[str]:
    """The guard-layer hook binding.
    Hook-layer binding: a backend whose every recent session logs hook failures is
    running UNHOOKED by the guard layer (productive but unguarded by the hook
    backstop). Information, not breakage — like the stub-rate signal it adds a
    reason but never flips ok (the commit-path / OFF_TRUNK guard is the backstop)."""
    reasons: list[str] = []
    if unhooked:
        names = ", ".join(f"{r.get('product')} {r.get('hook_failures')} fail/"
                          f"{r.get('sessions')} sess "
                          f"({int(float(r.get('failure_session_rate') or 0) * 100)}%)"
                          for r in unhooked[:4])
        reasons.append(
            f"guard hook layer UNBOUND on {len(unhooked)} backend(s) ({names}) — "
            f"productive but running unhooked; the commit-path/OFF_TRUNK guard is the "
            f"backstop (#1277)")
    return reasons


def _guard_reasons(guard: dict[str, Any]) -> list[str]:
    """The kernel decisions recorded on the dispatch path.
    Guard coverage: the witnessed proof the dispatch path ran THROUGH `fak guard`
    (per-session decision journals), and the kernel's decision mix. Informational —
    it adds a reason but never flips ok. A present-but-empty trail is its own signal
    (workers booted under guard but proposed no adjudicated tool call)."""
    reasons: list[str] = []
    g_sessions = _int(guard.get("sessions"), 0) or 0
    g_rows = _int(guard.get("rows"), 0) or 0
    g_child_crashes = _int((guard.get("by_kind") or {}).get("CHILD_CRASH"), 0) or 0
    if g_sessions and g_rows:
        crash_text = ""
        if g_child_crashes:
            crash_text = f", {g_child_crashes} child crash"
            if g_child_crashes != 1:
                crash_text += "es"
        loop_text = ""
        if guard.get("livelock_candidates"):
            loop_text = f"; loop candidate: {_guard_livelock_label(guard['livelock_candidates'][0])}"
        reasons.append(
            f"fak guard witnessed {g_rows} kernel decision(s) across {g_sessions} "
            f"dispatch session(s) ({guard.get('denied', 0)} denied, "
            f"{guard.get('quarantined', 0)} quarantined{crash_text}){loop_text}")
    elif g_sessions:
        reasons.append(
            f"fak guard ran {g_sessions} dispatch session(s) but recorded 0 decisions "
            f"({guard.get('empty_sessions', 0)} empty"
            f"{_guard_empty_cause_bit(guard)}) — workers booted under guard "
            f"but proposed no adjudicated tool call")
    return reasons


def _spawn_cause_reasons(spawn_causes: dict[str, Any]) -> list[str]:
    """The trailing SPAWN_FAILED cause mix (#4590), worst cause first. The
    stale_cred drain that REDDENS the card is a separate alarm fold."""
    reasons: list[str] = []
    if spawn_causes.get("schema"):
        sc_by_cause = spawn_causes.get("by_cause") or {}
        sc_failed = _int(spawn_causes.get("spawn_failed"), 0) or 0
        if sc_failed:
            mix = ", ".join(
                f"{c}={_int((sc_by_cause.get(c) or {}).get('count'), 0)}"
                for c in sorted(sc_by_cause,
                                key=lambda c: -(_int((sc_by_cause.get(c) or {}).get("count"), 0) or 0))
                if _int((sc_by_cause.get(c) or {}).get("count"), 0))
            reasons.append(
                f"spawn-failed mix: {sc_failed}/{_int(spawn_causes.get('spawns'), 0)} spawns "
                f"early-exited (rate {spawn_causes.get('rate')}) [{mix}] (#4590)")
    return reasons


def _throughput_reasons(throughput: dict[str, Any], *, tp_na: bool) -> list[str]:
    """Completed-per-hour against target over the primary analysis window."""
    reasons: list[str] = []
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
    return reasons


def _run_status_fold(run_status: list[dict[str, Any]]) -> tuple[dict[str, int], int, list[str]]:
    """Fold the `dos status` digests into (liveness counts, read errors, reasons)."""
    status_counts: dict[str, int] = {}
    status_errors = 0
    reasons: list[str] = []
    for digest in run_status:
        if digest.get("_error"):
            status_errors += 1
            continue
        run_verdict = str(((digest.get("liveness") or {}).get("verdict")) or "UNKNOWN")
        status_counts[run_verdict] = status_counts.get(run_verdict, 0) + 1
    if run_status:
        if status_errors:
            reasons.append(f"dos status digest read had {status_errors} error(s); inspect run_status")
        else:
            reasons.append(f"run truth from dos status digest for {len(run_status)} RID(s)")
    return status_counts, status_errors, reasons


def _lease_reasons(leases: dict[str, Any]) -> list[str]:
    """Active lane leases and which routed candidate issues they block."""
    reasons: list[str] = []
    if leases.get("read_error"):
        reasons.append(f"lease read unavailable: {leases.get('read_error')}")
    elif leases.get("active_count"):
        blocking = _int(leases.get("blocking_count"), 0) or 0
        if leases.get("candidate_source_available") is False:
            reasons.append(
                f"{leases.get('active_count')} active lane lease(s); candidate blocking unknown "
                "(backlog fold unavailable)")
        elif blocking:
            blocked_nums: list[str] = []
            for row in leases.get("active") or []:
                if not row.get("blocks_candidate"):
                    continue
                for cand in row.get("blocking_candidates") or []:
                    issue = cand.get("issue")
                    if issue is not None:
                        blocked_nums.append(f"#{issue}")
            suffix = f" ({', '.join(blocked_nums[:6])})" if blocked_nums else ""
            reasons.append(
                f"{blocking}/{leases.get('active_count')} active lane lease(s) block "
                f"current candidate issue(s){suffix}")
        else:
            reasons.append(f"{leases.get('active_count')} active lane lease(s), none blocking current candidates")
    return reasons


def _lane_lease_reasons(worker_leases: dict[str, Any]) -> list[str]:
    """The KERNEL lane-lease holder verdict (#5859).

    Emitted whether or not the worker-sidecar half of the cross-check is available:
    a dead lane-lease holder fences a whole tree for every agent in the fleet, and it
    must never be silent behind an unrelated probe failure.
    """
    lane = worker_leases.get("lane_leases")
    if not lane:
        return []
    reasons: list[str] = []
    dead = _int(lane.get("dead_count"), 0) or 0
    unknown = _int(lane.get("unknown_count"), 0) or 0
    live = _int(lane.get("live_count"), 0) or 0
    total = _int(lane.get("total"), 0) or 0
    if not lane.get("available"):
        reasons.append(
            "lane-lease holder liveness UNKNOWN (not clean): "
            f"{lane.get('read_error') or 'lane-lease set unreadable'}; "
            "next action: re-run from the workspace root so `dos lease-lane live` resolves")
        return reasons
    if dead:
        bits = _lane_lease_bits(lane.get("dead") or [])
        reasons.append(
            f"lane-lease STALE HOLDERS: dead-holder={dead} of {total} lane lease(s) "
            f"(live={live}, unknown={unknown}) — these leases are past their TTL, so the "
            f"admission fold already elides them; they still sit in the structural WAL "
            f"fold{': ' + bits if bits else ''}")
        reasons.append(f"next action: {lane.get('next_action') or _LANE_LEASE_NEXT_ACTION}")
    elif unknown:
        reasons.append(
            f"lane-lease holders: live={live}, dead-holder=0, unknown={unknown} of {total} "
            "(unknown holders are unproven, not clean)")
    elif total:
        reasons.append(f"lane-lease holders all live ({live} lease(s), dead-holder=0)")
    return reasons


def _worker_lease_reasons(worker_leases: dict[str, Any]) -> list[str]:
    """The worker/lease cross-check: clean pairs, orphan processes, orphan leases,
    and (#5859) the kernel lane-lease holder verdict.

    The word "clean" is now gated on `lane_lease_verdict_clean`: the local
    worker<->resolver-lease pairs can all match while every exclusive lane lease on
    the box is held by a dead process, and reporting THAT as clean is precisely the
    blindness that let 18 fenced lanes go unnoticed for three weeks.
    """
    reasons: list[str] = []
    lane_clean = lane_lease_verdict_clean(worker_leases)
    if worker_leases.get("available") is False:
        reasons.append(f"worker/lease cross-check unavailable: {worker_leases.get('error')}")
        return reasons + _lane_lease_reasons(worker_leases)
    if worker_leases:
        op = _int(worker_leases.get("orphan_process_count"), 0) or 0
        ol = _int(worker_leases.get("orphan_lease_count"), 0) or 0
        clean = _int(worker_leases.get("clean_count"), 0) or 0
        dead = _int(worker_leases.get("dead_holder_count"), 0) or 0
        if op or ol or not lane_clean:
            lease_note = ""
            if ol:
                lease_note = "; unmatched live leases are not necessarily reapable"
            reasons.append(
                f"worker/lease cross-check: clean={clean}, orphan-process={op}, "
                f"unmatched-live-lease={ol}, dead-holder={dead}{lease_note}")
        elif clean:
            reasons.append(f"worker/lease cross-check clean ({clean} matched worker/lease pair(s))")
        active_worker_line = _active_worker_summary(worker_leases)
        if active_worker_line:
            reasons.append(active_worker_line)
    return reasons + _lane_lease_reasons(worker_leases)


def _seat_inventory_reasons(seat_inventory: dict[str, Any]) -> list[str]:
    """The account seat pool.
    Seat inventory (#1799): available/busy/cooling/unavailable counts across the
    explicit account seat pool, so an operator sees WHY a seat is held without
    digging through fleet_accounts directly. Informational only — never flips ok."""
    reasons: list[str] = []
    if seat_inventory.get("_error"):
        reasons.append(f"seat inventory unavailable: {seat_inventory.get('_error')}")
    elif seat_inventory.get("schema"):
        reasons.append(_seat_inventory_summary_line(seat_inventory))
    return reasons


def _resolver_tick_reasons(resolve_ticks: dict[str, Any],
                           preflight_launch_blocker: dict[str, Any] | None,) -> list[str]:
    """The selected resolver tick: stale, launch-ready, held by the current
    preflight, held by its own launch gate, or simply not spawning — plus
    any tick artifact that could not be read."""
    reasons: list[str] = []
    latest_tick = resolve_ticks.get("selected") or resolve_ticks.get("latest") or {}
    if latest_tick:
        tick_age = _age_text(latest_tick.get("age_min"))
        tick_backend = latest_tick.get("backend") or "?"
        tick_verdict = latest_tick.get("verdict") or "UNKNOWN"
        tick_issue = latest_tick.get("target_issue")
        issue_bit = f" #{tick_issue}" if tick_issue is not None else ""
        lane_bit = f" lane {latest_tick.get('lane')}" if latest_tick.get("lane") else ""
        gate = latest_tick.get("launch_gate") or {}
        if not latest_tick.get("fresh"):
            reasons.append(
                f"last resolver tick stale: {tick_backend}{issue_bit}{lane_bit} "
                f"{tick_verdict} age {tick_age}")
        elif gate.get("ready") is True:
            if preflight_launch_blocker:
                reasons.append(
                    f"selected resolver tick held by current preflight: "
                    f"{tick_backend}{issue_bit}{lane_bit} "
                    f"{preflight_launch_blocker['verdict']}; next action: "
                    f"{preflight_launch_blocker['action']}; age {tick_age}")
            else:
                reasons.append(
                    f"selected resolver tick launch-ready: {tick_backend}{issue_bit}{lane_bit} "
                    f"age {tick_age}")
        elif gate.get("ready") is False:
            blockers = gate.get("blockers") or []
            code = (blockers[0].get("code") if blockers else tick_verdict) or tick_verdict
            action = latest_tick.get("next_action") or "inspect-last-resolve-tick"
            reasons.append(
                f"selected resolver tick held: {tick_backend}{issue_bit}{lane_bit} "
                f"{code}; next action: {action}; age {tick_age}")
        elif tick_verdict not in ("WOULD_SPAWN", "SPAWNED"):
            action = latest_tick.get("next_action") or "inspect-last-resolve-tick"
            reasons.append(
                f"selected resolver tick: {tick_backend}{issue_bit}{lane_bit} "
                f"{tick_verdict}; next action: {action}; age {tick_age}")
    if resolve_ticks.get("errors"):
        reasons.append(f"{len(resolve_ticks.get('errors') or [])} resolver tick artifact read error(s)")
    return reasons


def _low_yield_reasons(low_yield: dict[str, Any]) -> list[str]:
    """The lanes that spent turns without closing anything.
    Low-yield lanes (#2062): lanes whose recent sessions spent >= the turn floor
    yet closed nothing on their own tree. Informational — a reason line, never a
    flip of ok. This is the per-lane feedback pick_lane lacked (silent_workers only
    sees the empty-log case)."""
    reasons: list[str] = []
    low_yield_flagged = [r for r in (low_yield.get("lanes") or [])
                         if r.get("verdict") == "LOW_YIELD"]
    if low_yield_flagged:
        names = ", ".join(
            f"{r.get('lane')} ({r.get('turns')}t/{r.get('sessions')} sess, 0 closes)"
            for r in low_yield_flagged[:4])
        reasons.append(
            f"{len(low_yield_flagged)} low-yield lane(s) over the last "
            f"{low_yield.get('lookback_min')}m ({names}) — turns spent with zero "
            f"ancestry-closes; pick_lane should re-scope or exclude (#2062)")
    return reasons


def _ships_reasons(ships: dict[str, Any]) -> list[str]:
    """Ship attribution by worker trailer.
    Ships-per-worker (#2065): best-effort (fak-worker) trailer attribution over the
    recent window. Informational — a reason line, never a flip of ok; the trailer is
    agent-emitted, so this is an attribution aid, not a witness."""
    reasons: list[str] = []
    ships_workers = ships.get("workers") or []
    if ships.get("attributed_ships"):
        top = ", ".join(f"{w.get('worker')}={w.get('ships')}" for w in ships_workers[:4])
        unk = ships.get("unknown") or 0
        unk_bit = f", {unk} unattributed" if unk else ""
        reasons.append(
            f"{ships.get('attributed_ships')} ship(s) attributed to "
            f"{ships.get('worker_count')} worker(s) via (fak-worker) trailer ({top})"
            f"{unk_bit} — best-effort aid, not a witness (#2065)")
    return reasons


def _backlog_rate_reasons(backlog_rate: dict[str, Any]) -> list[str]:
    """Backlog arrival versus service rate.
    Backlog arrival-vs-service rate (#2634): surface a numeric BACKLOG_OUTPACED
    so an operator reads the supply/demand trend without hand-diffing counters.
    Informational — a reason line, never a flip of ok (this is the meter)."""
    reasons: list[str] = []
    if backlog_rate.get("backlog_outpaced"):
        reasons.append(
            f"backlog OUTPACED: arrival {backlog_rate.get('arrival_rate_per_hour')}/h > "
            f"service {backlog_rate.get('service_rate_per_hour')}/h for "
            f"{backlog_rate.get('consecutive_outpaced_windows')} consecutive "
            f"{backlog_rate.get('window_hours')}h window(s) — healthy but supply-bound, "
            f"not stalled; do not hand-launch (#2634)")
    return reasons


def _lab_readiness_reasons(lab_readiness: dict[str, Any]) -> list[str]:
    """Whether the lab-readiness link admits lab-backed dispatch, and if not,
    the next action that would publish it."""
    reasons: list[str] = []
    if lab_readiness.get("schema"):
        link = _lab_link_label(lab_readiness)
        action = lab_readiness.get("next_action") or "publish-lab-readiness"
        if lab_readiness.get("admit_dispatch"):
            reasons.append(f"lab readiness: {link} — lab-backed dispatch may be admitted")
        else:
            reasons.append(f"lab readiness: {link}; next action: {action}; no lab-backed dispatch")
    return reasons


def _utilization_reasons(utilization: dict[str, Any]) -> list[str]:
    """Free worker slots and the next action for every actionable utilization
    state (headroom, account/host blocked, edit held)."""
    reasons: list[str] = []
    if utilization.get("state") in (
            "HEADROOM_LAUNCH_READY", "HEADROOM_REPAIR_READY", "HEADROOM_HELD", "HEADROOM_STALE_PLAN",
            "ACCOUNT_CAPPED", "ACCOUNT_BLOCKED", "HOST_BLOCKED", "EDIT_HELD"):
        actions = ", ".join(utilization.get("next_actions") or [])
        suffix = f"; next action: {actions}" if actions else ""
        reasons.append(
            f"utilization: {utilization.get('state')} "
            f"({(utilization.get('worker_slots') or {}).get('headroom')} free slot(s)){suffix}")
    return reasons


def build_payload(*, root: Path, pre: dict, sup: dict, wd: dict, backlog: dict,
                  closure: dict, max_workers: int, fast: bool,
                  silent: list[dict[str, Any]] | None = None,
                  weekly_cap: dict[str, Any] | None = None,
                  throughput: dict[str, Any] | None = None,
                  backend_health: list[dict[str, Any]] | None = None,
                  backend_stub_rate: list[dict[str, Any]] | None = None,
                  hook_failures: list[dict[str, Any]] | None = None,
                  guard: dict[str, Any] | None = None,
                  run_status: list[dict[str, Any]] | None = None,
                  merge: dict[str, Any] | None = None,
                  leases: dict[str, Any] | None = None,
                  worker_leases: dict[str, Any] | None = None,
                  seat_inventory: dict[str, Any] | None = None,
                  lab_readiness: dict[str, Any] | None = None,
                  resolve_ticks: dict[str, Any] | None = None,
                  resolver_preflight: dict[str, Any] | None = None,
                  low_yield: dict[str, Any] | None = None,
                  ships: dict[str, Any] | None = None,
                  watch: dict[str, Any] | None = None,
                  backlog_rate: dict[str, Any] | None = None,
                  spawn_causes: dict[str, Any] | None = None,
                  seat_streaks: list[dict[str, Any]] | None = None,
                  fleet_decline: dict[str, Any] | None = None,
                  route_health: dict[str, Any] | None = None,
                  launch_admission: dict[str, Any] | None = None) -> dict[str, Any]:
    # --- dispatcher liveness / capacity ---
    cap = _int(pre.get("cap"))
    live = _int(pre.get("live"))
    host_safe = bool((pre.get("host") or {}).get("safe"))
    acct = pre.get("account") or {}
    pre_verdict = pre.get("verdict")

    # --- throughput (closed/hour vs target) ---
    throughput = throughput or {}
    tp_na = "_skipped" in throughput or "_error" in throughput or not throughput.get("schema")

    merge = merge or {}
    ok, verdict, reasons = _dispatch_base_verdict(
        pre=pre, cap=cap, live=live, host_safe=host_safe, acct=acct,
        weekly_cap=weekly_cap, merge=merge)

    # The launcher's FINAL gate (#6495). Folded immediately after the base verdict so
    # no later alarm can be read as the reason growth is held: a READY_TO_GROW that the
    # launch-rate/cooldown gate would refuse becomes LAUNCH_COOLDOWN right here, with
    # the retry time, before any other signal is layered on.
    ok, verdict = _apply_launch_admission(
        launch_admission, ok=ok, verdict=verdict, reasons=reasons)

    reasons += _watchdog_reasons(wd)
    silent = silent or []
    reasons += _silent_worker_reasons(silent)
    backend_health = backend_health or []
    backend_stub_rate = backend_stub_rate or []
    reasons += _backend_health_reasons(backend_health, backend_stub_rate)
    hook_failures = hook_failures or []
    unhooked = [r for r in hook_failures if r.get("all_sessions_unhooked")]
    reasons += _hook_health_reasons(unhooked)
    guard = guard or {}
    reasons += _guard_reasons(guard)

    # Spawn-failed cause mix (#4590): surface the trailing SPAWN_FAILED cause mix on
    # the DEFAULT card (previously reachable only via the --spawn-causes sub-flag),
    # and REDDEN when the stale_cred (permanent-auth) rate crosses the drain
    # threshold — the needs_login-seat drain signature. This is the one spawn signal
    # that FLIPS ok: a dead seat bleeding the fleet is breakage, not a steady state.
    spawn_causes = spawn_causes or {}
    spawn_alarm = spawn_stale_cred_alarm(spawn_causes)
    reasons += _spawn_cause_reasons(spawn_causes)
    ok, verdict = _apply_red_alarm(
        spawn_alarm, ok=ok, verdict=verdict, reasons=reasons,
        red_verdict="SPAWN_STALE_CRED_DRAIN")

    # Seat-keyed spawn-failure streak (#4591): the #4590 rate alarm above needs
    # a high AGGREGATE stale_cred fraction; a single dead seat in a big pool
    # stays under it while still burning one issue per tick. Join the per-SEAT
    # consecutive-fail run-length the dispatcher persists with the seat
    # inventory's auth_failed hold and the stale_cred cause mix, and flip the
    # verdict the moment ANY one seat hits the threshold — the exact drain the
    # target-keyed streak could never see.
    seat_alarm = seat_spawn_fail_alarm(seat_streaks, seat_inventory or {},
                                       spawn_causes)
    ok, verdict = _apply_red_alarm(
        seat_alarm, ok=ok, verdict=verdict, reasons=reasons,
        red_verdict="SEAT_SPAWN_FAIL_STREAK")

    # Net-worker-decline alarm (#4591, part 2): `live` stepping strictly down
    # for M consecutive fleet-status-history appends is a fleet DRAINING —
    # whatever each individual tick verdict said. Degrades silently when the
    # ledger is absent/unreadable (fail-open `_error`).
    fleet_decline = fleet_decline or {}
    ok, verdict = _apply_red_alarm(
        fleet_decline, ok=ok, verdict=verdict, reasons=reasons,
        red_verdict="NET_WORKER_DECLINE", default_reason="net worker decline")

    reasons += _throughput_reasons(throughput, tp_na=tp_na)
    run_status = run_status or []
    status_counts, status_errors, status_reasons = _run_status_fold(run_status)
    reasons += status_reasons
    leases = leases or {}
    reasons += _lease_reasons(leases)
    worker_leases = worker_leases or {}
    reasons += _worker_lease_reasons(worker_leases)
    seat_inventory = seat_inventory or {}
    reasons += _seat_inventory_reasons(seat_inventory)

    resolver_preflight = resolver_preflight or {}
    resolver_preflight_line = _resolver_preflight_summary(resolver_preflight)
    if resolver_preflight_line:
        reasons.append(resolver_preflight_line)
    preflight_launch_blocker = _current_preflight_launch_blocker(pre_verdict, resolver_preflight)
    resolve_ticks = resolve_ticks or {}
    reasons += _resolver_tick_reasons(resolve_ticks, preflight_launch_blocker)

    low_yield = low_yield or {}
    reasons += _low_yield_reasons(low_yield)
    ships = ships or {}
    reasons += _ships_reasons(ships)
    backlog_rate = backlog_rate or {}
    reasons += _backlog_rate_reasons(backlog_rate)
    lab_readiness = lab_readiness or {}
    if lab_readiness.get("schema") and not lab_readiness.get("commands"):
        lab_readiness = {**lab_readiness, "commands": _lab_readiness_commands()}
    reasons += _lab_readiness_reasons(lab_readiness)

    limiter = _dispatch_limiter(pre, backlog, closure, leases)
    utilization = utilization_state(
        live=live, cap=cap, host_safe=host_safe, pre_verdict=pre_verdict,
        resolver=resolve_ticks, resolver_preflight=resolver_preflight,
        lab_readiness=lab_readiness,
        weekly_cap=weekly_cap, merge=merge)
    reasons += _utilization_reasons(utilization)

    next_action = _next_action(verdict, launch_admission, merge, reasons)

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "reasons": reasons,
        "next_action": next_action,
        "spawn_gate": _spawn_gate_block(pre_verdict, launch_admission,
                                        verdict=verdict, next_action=next_action,
                                        evidence=_launch_evidence(silent, spawn_causes)),
        "accounts": _accounts_block(seat_inventory, acct, launch_admission),
        "workspace": str(root),
        "weekly_cap": weekly_cap,
        "dispatcher": _dispatcher_block(
            cap=cap, live=live, host_safe=host_safe, pre_verdict=pre_verdict,
            limiter=limiter, acct=acct, wd=wd),
        "supervisor": _supervisor_block(sup),
        "backlog": _backlog_block(backlog),
        "closure": _closure_block(closure),
        "throughput": _throughput_block(throughput, tp_na=tp_na),
        "watch_decision": watch or {},
        "backlog_rate": backlog_rate or {},
        "spawn_causes": _spawn_causes_block(spawn_causes, spawn_alarm),
        "seat_streaks": {
            "rows": seat_streaks or [],
            "alarm": seat_alarm,
        },
        "fleet_decline": fleet_decline,
        "route_health": route_health or {},
        "workers": {
            "silent_count": len(silent),
            "silent": silent,
        },
        "backend_health": {
            "dead_count": len(backend_health),
            "dead": backend_health,
            "stub_rate": backend_stub_rate,
        },
        "hook_health": {
            "unhooked_count": len(unhooked),
            "by_backend": hook_failures,
        },
        "guard": guard,
        "run_status": {
            "source": "dos status",
            "count": len(run_status),
            "liveness": status_counts,
            "errors": status_errors,
            "digests": run_status,
        },
        "leases": leases,
        "worker_lease_check": worker_leases,
        "seat_inventory": seat_inventory or {},
        "resolver": resolve_ticks or {},
        "resolver_preflight": resolver_preflight or {},
        "low_yield": low_yield or {},
        "ships_per_worker": ships or {},
        "lab_readiness": lab_readiness or {},
        "utilization": utilization,
        "git": _git_block(merge),
        "fast": fast,
    }


def git_date(root: Path) -> str:
    """The last-commit date (YYYY-MM-DD) — deterministic, no wall-clock in the tool."""
    try:
        proc = subprocess.run(["git", "log", "-1", "--format=%cs"], cwd=str(root),
                              capture_output=True, text=True, timeout=15,
                              creationflags=_win_creationflags())
        date = (proc.stdout or "").strip()
        return date or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _default_max_workers() -> int:
    """Mirror of dispatch_preflight.DEFAULT_MAX_WORKERS (built-in 20, FAK_MAX_WORKERS
    env knob applied) so the card's probe matches the gate's own ceiling instead of
    understating the fleet's headroom with a stale local default."""
    raw = os.environ.get("FAK_MAX_WORKERS", "").strip()
    try:
        if raw and int(raw) > 0:
            return int(raw)
    except ValueError:
        pass
    return 20


def main(argv: list[str] | None = None) -> int:
    default_workers = _default_max_workers()
    ap = argparse.ArgumentParser(description="One-touch always-on dispatcher status card.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--max-workers", type=int, default=default_workers,
                    help="cap used by the spawn-gate preflight "
                         f"(default: {default_workers}; FAK_MAX_WORKERS retunes it)")
    ap.add_argument("--fast", action="store_true",
                    help="skip the two gh-backed folds (backlog + closure); pure-local")
    ap.add_argument("--closure-commits", type=int, default=2500,
                    help="MINIMUM git-history budget for the closure audit; the actual "
                         "window auto-grows to the repo size + headroom so the audit "
                         "never scans a stale slice (default floor: 2500)")
    ap.add_argument("--md", default="",
                    help="write the committed markdown status doc to this path "
                         "(forces the full fold; --fast is ignored when --md is set)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--spawn-causes", action="store_true",
                    help="print ONLY the SPAWN_FAILED early-exit rate broken down by "
                         "cause (#2635) over the trailing window and exit — the "
                         "read-only attribution fold, honoring --json")
    ap.add_argument("--slack", nargs="?", const="__env__", default=None,
                    metavar="CHANNEL",
                    help="post the status card to Slack (optional channel id; default: "
                         "$FAK_DISPATCH_CHANNEL via tools/slack_post). Forces the full "
                         "fold so the posted card is never all-n/a.")
    ap.add_argument("--slack-dry-run", action="store_true",
                    help="with --slack: resolve the channel/token and report what WOULD "
                         "be posted without sending")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    if args.spawn_causes:
        # Read-only attribution fold (#2635): no gh/preflight fold, no spawn.
        breakdown = spawn_failed_cause_breakdown(root / RUNS_DIRNAME)
        if args.json:
            print(json.dumps(breakdown, indent=2))
        else:
            print(render_spawn_causes(breakdown))
        return 0
    # The committed doc must carry the real backlog/closure tables, so --md always
    # runs the full fold regardless of --fast. A LIVE Slack post is just as useless when
    # every row reads "n/a (fast)", so --slack forces the full fold too — but a
    # --slack-dry-run is a wiring check where speed matters, so it still honors --fast.
    live_slack = args.slack is not None and not args.slack_dry_run
    fast = args.fast and not args.md and not live_slack
    payload = collect(root, max_workers=args.max_workers, fast=fast,
                      closure_commits=args.closure_commits)
    # Loop-level drought alarm: fold AFTER collect so the witness rides every
    # output (json/slack/md/human). droughty = zero fleet commits in the window
    # AND the loop is armed (a drought while the watchdog is not installed is an
    # intentional idle, not an alarm).
    drought = commit_drought(root)
    _wd = payload.get("watchdog") or {}
    drought["droughty"] = bool(drought.get("dry") and _wd.get("installed"))
    payload["commit_drought"] = drought

    if args.slack is not None:
        channel = "" if args.slack == "__env__" else args.slack
        slack_verdict = post_to_slack(payload, channel=channel,
                                      dry_run=args.slack_dry_run)
        payload["slack"] = slack_verdict
        if not args.json:
            if slack_verdict.get("posted"):
                print(f"slack: posted card to {slack_verdict.get('channel')} "
                      f"(ts={slack_verdict.get('ts')})")
            elif slack_verdict.get("dry_run"):
                print(f"slack (dry-run): would post to "
                      f"{slack_verdict.get('channel') or '(unset)'} "
                      f"[{slack_verdict.get('channel_source')}]")
            elif slack_verdict.get("skipped"):
                print(f"slack: skipped — {slack_verdict.get('skipped')}")
            else:
                print(f"slack: FAILED — {slack_verdict.get('error')}")

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
