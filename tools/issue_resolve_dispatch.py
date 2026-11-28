#!/usr/bin/env python3
r"""One guarded, switcher-routed dispatch TICK that spawns an ISSUE-RESOLUTION
worker — the arm that moves the open-issue counter on a plan-empty repo.

    python tools/issue_resolve_dispatch.py                 # plan one tick (dry-run)
    python tools/issue_resolve_dispatch.py --live          # spawn one issue worker
    python tools/issue_resolve_dispatch.py --lane gateway --live
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import re  # noqa: F401
import shlex  # noqa: F401
import shutil  # noqa: F401
import signal  # noqa: F401
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
if __name__ == "__main__":
    sys.modules.setdefault("issue_resolve_dispatch", sys.modules[__name__])

import account_probe  # noqa: E402, F401
import dispatch_preflight  # noqa: E402, F401
import dispatch_worker  # noqa: E402
import fleet_trend  # noqa: E402, F401
import issue_closure_audit  # noqa: E402, F401
import issue_dispatch  # noqa: E402, F401
import issue_lane_router  # noqa: E402
import issue_worker_prompt  # noqa: E402, F401
import lane_core  # noqa: E402, F401
import lane_yield  # noqa: E402, F401
import tier_launch  # noqa: E402, F401
import worker_worktree  # noqa: E402, F401

import issue_resolve_dispatch_core as _core  # noqa: E402
import issue_resolve_dispatch_eval as _eval  # noqa: E402
import issue_resolve_dispatch_guards as _guards  # noqa: E402
import issue_resolve_dispatch_witness as _witness  # noqa: E402
import issue_resolve_dispatch_workers as _workers  # noqa: E402

from issue_resolve_dispatch_core import (  # noqa: E402
    ACCEPTANCE_GATE_HOLD_REASON as ACCEPTANCE_GATE_HOLD_REASON,
    ACCOUNT_SIDECAR_SUFFIX as ACCOUNT_SIDECAR_SUFFIX,
    ACTIVE_COMPACT_RUNAWAY_MIN_COUNT as ACTIVE_COMPACT_RUNAWAY_MIN_COUNT,
    ACTIVE_COMPACT_RUNAWAY_MIN_PAST_K as ACTIVE_COMPACT_RUNAWAY_MIN_PAST_K,
    ACTIVE_COMPACT_RUNAWAY_SHED_K as ACTIVE_COMPACT_RUNAWAY_SHED_K,
    ACTIVE_GUARD_LIVELOCK_ESCAPE_ROWS as ACTIVE_GUARD_LIVELOCK_ESCAPE_ROWS,
    ACTIVE_GUARD_LIVELOCK_MIN_COUNT as ACTIVE_GUARD_LIVELOCK_MIN_COUNT,
    BACKENDS as BACKENDS,
    BASE_SHA_SIDECAR_SUFFIX as BASE_SHA_SIDECAR_SUFFIX,
    BENIGN_ACTIONS as BENIGN_ACTIONS,
    CLAIM_NO_COMMIT as CLAIM_NO_COMMIT,
    CLAIM_UNWITNESSED as CLAIM_UNWITNESSED,
    CLAIM_WITNESSED as CLAIM_WITNESSED,
    CLAUDE_PROMPT_FILE_SUFFIX as CLAUDE_PROMPT_FILE_SUFFIX,
    CLAUDE_WORKER_EFFORT as CLAUDE_WORKER_EFFORT,
    CLAUDE_WORKER_MODEL as CLAUDE_WORKER_MODEL,
    CONTRACT_OVERLAY_DIRNAME as CONTRACT_OVERLAY_DIRNAME,
    DEFAULT_COLLISION_HOLD_TTL_H as DEFAULT_COLLISION_HOLD_TTL_H,
    DEFAULT_CONTRACT_HOLD_TTL_H as DEFAULT_CONTRACT_HOLD_TTL_H,
    DEFAULT_CONTRACT_SCAN as DEFAULT_CONTRACT_SCAN,
    DEFAULT_FIRST_EDIT_DEADLINE_S as DEFAULT_FIRST_EDIT_DEADLINE_S,
    DEFAULT_GUARD_PROBE_TTL_H as DEFAULT_GUARD_PROBE_TTL_H,
    DEFAULT_ISSUE_CONTRACT_MIN_SCORE as DEFAULT_ISSUE_CONTRACT_MIN_SCORE,
    DEFAULT_MULTI_LANE_HOLD_TTL_H as DEFAULT_MULTI_LANE_HOLD_TTL_H,
    DEFAULT_REALLOC_CEILING as DEFAULT_REALLOC_CEILING,
    DEFAULT_REBLOCK_STREAK_HOLD_N as DEFAULT_REBLOCK_STREAK_HOLD_N,
    DEFAULT_REBLOCK_STREAK_HOLD_TTL_H as DEFAULT_REBLOCK_STREAK_HOLD_TTL_H,
    DEFAULT_REPAIR_BATCH as DEFAULT_REPAIR_BATCH,
    DEFAULT_REPAIR_COOLDOWN_MIN as DEFAULT_REPAIR_COOLDOWN_MIN,
    DEFAULT_SEAT_ADAPTIVE as DEFAULT_SEAT_ADAPTIVE,
    DEFAULT_SEAT_CEILING as DEFAULT_SEAT_CEILING,
    DEFAULT_SEAT_RAMP_DELTA as DEFAULT_SEAT_RAMP_DELTA,
    DEFAULT_SPAWN_BURST as DEFAULT_SPAWN_BURST,
    DEFAULT_SPAWN_BURST_STAGGER_S as DEFAULT_SPAWN_BURST_STAGGER_S,
    DEFAULT_SPAWN_PROBE_S as DEFAULT_SPAWN_PROBE_S,
    DEFAULT_WORKER_TIMEOUT_S as DEFAULT_WORKER_TIMEOUT_S,
    DISPATCH_PRIORITY_WEIGHT as DISPATCH_PRIORITY_WEIGHT,
    EARLY_EXIT_TAIL_CHARS as EARLY_EXIT_TAIL_CHARS,
    LEASE_DRAIN_TIMEOUT_S as LEASE_DRAIN_TIMEOUT_S,
    LEASE_ID_PREFIX as LEASE_ID_PREFIX,
    LEASE_REFUSED_RC as LEASE_REFUSED_RC,
    LEASE_SIDECAR_SUFFIX as LEASE_SIDECAR_SUFFIX,
    LEASE_TIMEOUT_RC as LEASE_TIMEOUT_RC,
    LEASE_TTL_MARGIN_S as LEASE_TTL_MARGIN_S,
    LOOP_ID_PREFIX as LOOP_ID_PREFIX,
    LOW_YIELD_MAX_EXCLUDES as LOW_YIELD_MAX_EXCLUDES,
    LOW_YIELD_MIN_SESSIONS as LOW_YIELD_MIN_SESSIONS,
    NO_COMMIT_AUTH_WALL as NO_COMMIT_AUTH_WALL,
    NO_COMMIT_BANNER_NOOP as NO_COMMIT_BANNER_NOOP,
    NO_COMMIT_CLEAN_EXIT as NO_COMMIT_CLEAN_EXIT,
    NO_COMMIT_DIED_BEFORE_EPILOGUE as NO_COMMIT_DIED_BEFORE_EPILOGUE,
    NO_COMMIT_GUARD_SPAWN_FAILED as NO_COMMIT_GUARD_SPAWN_FAILED,
    NO_COMMIT_MISSING_LOG as NO_COMMIT_MISSING_LOG,
    NO_COMMIT_OFF_TRUNK as NO_COMMIT_OFF_TRUNK,
    NO_COMMIT_POLICY_BLOCK as NO_COMMIT_POLICY_BLOCK,
    NO_COMMIT_PREVIEW_CONFIRM as NO_COMMIT_PREVIEW_CONFIRM,
    NO_COMMIT_RESTART_EXHAUSTED as NO_COMMIT_RESTART_EXHAUSTED,
    NO_COMMIT_SELF_MODIFY as NO_COMMIT_SELF_MODIFY,
    NO_COMMIT_UNKNOWN as NO_COMMIT_UNKNOWN,
    OPENCODE_PROMPT_FILE_SUFFIX as OPENCODE_PROMPT_FILE_SUFFIX,
    OPENCODE_PROMPT_NOTICE as OPENCODE_PROMPT_NOTICE,
    OPEN_WITNESSED as OPEN_WITNESSED,
    REBLOCK_STREAK_FIRST_LOSS_TTL_H as REBLOCK_STREAK_FIRST_LOSS_TTL_H,
    REBLOCK_STREAK_MIN_TTL_MARGIN_H as REBLOCK_STREAK_MIN_TTL_MARGIN_H,
    REFUSAL_CLASS_LANE_LEASE as REFUSAL_CLASS_LANE_LEASE,
    REFUSAL_CLASS_WORKTREE_COTENANCY as REFUSAL_CLASS_WORKTREE_COTENANCY,
    REPAIR_ISSUES_SIDECAR_SUFFIX as REPAIR_ISSUES_SIDECAR_SUFFIX,
    REPAIR_LANE as REPAIR_LANE,
    RUNS_DIRNAME as RUNS_DIRNAME,
    SAME_ISSUE_WIP_LOOKBACK_MIN as SAME_ISSUE_WIP_LOOKBACK_MIN,
    SAME_ISSUE_WIP_ORIENTATION_DOCS as SAME_ISSUE_WIP_ORIENTATION_DOCS,
    SCHEMA as SCHEMA,
    SEAT_STREAK_REPROBE_MIN as SEAT_STREAK_REPROBE_MIN,
    SPAWN_BURST_ENV as SPAWN_BURST_ENV,
    SPAWN_BURST_HARD_CEILING as SPAWN_BURST_HARD_CEILING,
    SPAWN_BURST_STAGGER_ENV as SPAWN_BURST_STAGGER_ENV,
    SPAWN_BURST_STAGGER_MAX_S as SPAWN_BURST_STAGGER_MAX_S,
    SPAWN_CAUSE_CHILD_CRASH as SPAWN_CAUSE_CHILD_CRASH,
    SPAWN_CAUSE_EXEC_RACE as SPAWN_CAUSE_EXEC_RACE,
    SPAWN_CAUSE_STALE_CRED as SPAWN_CAUSE_STALE_CRED,
    SPAWN_CAUSE_UNKNOWN as SPAWN_CAUSE_UNKNOWN,
    SPAWN_CAUSE_WEEKLY_LIMIT as SPAWN_CAUSE_WEEKLY_LIMIT,
    SPAWN_FAILED_CAUSES as SPAWN_FAILED_CAUSES,
    SPAWN_FAILED_RED_STREAK as SPAWN_FAILED_RED_STREAK,
    WAVE_SIDECAR_SUFFIX as WAVE_SIDECAR_SUFFIX,
    WITNESS_SIDECAR_SUFFIX as WITNESS_SIDECAR_SUFFIX,
    _ACCEPTANCE_GATE_SECTION as _ACCEPTANCE_GATE_SECTION,
    _ANNOUNCED_WAIT_DUR_RE as _ANNOUNCED_WAIT_DUR_RE,
    _ANNOUNCED_WAIT_RE as _ANNOUNCED_WAIT_RE,
    _ANY_WORKER_LOG_RE as _ANY_WORKER_LOG_RE,
    _AUDIT_LOG_RE as _AUDIT_LOG_RE,
    _AUTH_GAP_RE as _AUTH_GAP_RE,
    _BACKEND_AUTHGAP_REPROBE_MIN as _BACKEND_AUTHGAP_REPROBE_MIN,
    _BACKEND_DEAD_STREAK as _BACKEND_DEAD_STREAK,
    _BACKEND_PRODUCT as _BACKEND_PRODUCT,
    _BACKEND_REPROBE_MIN as _BACKEND_REPROBE_MIN,
    _CAP_BANNER_RE as _CAP_BANNER_RE,
    _CAP_FIVE_HOUR_RE as _CAP_FIVE_HOUR_RE,
    _CAP_RESET_FALLBACK_RE as _CAP_RESET_FALLBACK_RE,
    _CAP_RESET_RE as _CAP_RESET_RE,
    _CAP_SESSION_RE as _CAP_SESSION_RE,
    _CAP_TAIL_BYTES as _CAP_TAIL_BYTES,
    _CAP_USAGE_RE as _CAP_USAGE_RE,
    _CAP_WEEKLY_RE as _CAP_WEEKLY_RE,
    _CODEX_RESET_RE as _CODEX_RESET_RE,
    _COLLISION_HOLD_LEDGER as _COLLISION_HOLD_LEDGER,
    _COMMIT_EVENT_RE as _COMMIT_EVENT_RE,
    _COMPACT_FIELD_RE as _COMPACT_FIELD_RE,
    _CONTRACT_FORCED_LEDGER as _CONTRACT_FORCED_LEDGER,
    _CONTRACT_HOLD_LEDGER as _CONTRACT_HOLD_LEDGER,
    _CREATE_NO_WINDOW as _CREATE_NO_WINDOW,
    _CTX_BUDGET_RE as _CTX_BUDGET_RE,
    _DETACHED_PROCESS as _DETACHED_PROCESS,
    _DIST_PAST_COMPACT_RE as _DIST_PAST_COMPACT_RE,
    _EDIT_EVENT_RE as _EDIT_EVENT_RE,
    _GATE_OPERATOR_INPUT as _GATE_OPERATOR_INPUT,
    _GATE_UNRESOLVED_TOKEN as _GATE_UNRESOLVED_TOKEN,
    _GLM_WALL_RE as _GLM_WALL_RE,
    _GUARD_CHILD_SPAWN_FAILED_MARKER as _GUARD_CHILD_SPAWN_FAILED_MARKER,
    _GUARD_EPILOGUE_MARKER as _GUARD_EPILOGUE_MARKER,
    _GUARD_FLAG_DEF_RE as _GUARD_FLAG_DEF_RE,
    _GUARD_PROBE_LEDGER as _GUARD_PROBE_LEDGER,
    _GUARD_PROBE_MIN_FLAGS as _GUARD_PROBE_MIN_FLAGS,
    _GUARD_PROBE_PEELED as _GUARD_PROBE_PEELED,
    _GUARD_PROBE_TIMEOUT_S as _GUARD_PROBE_TIMEOUT_S,
    _HEALTH_SKIP_STUB_RATE as _HEALTH_SKIP_STUB_RATE,
    _HOLD_NO_COMMIT_REASONS as _HOLD_NO_COMMIT_REASONS,
    _ISSUE_REF as _ISSUE_REF,
    _ISSUE_TOKEN_RE as _ISSUE_TOKEN_RE,
    _LANE_TREE_CACHE as _LANE_TREE_CACHE,
    _LEASE_SESSION_ID_RE as _LEASE_SESSION_ID_RE,
    _LEASE_SESSION_SAFE_RE as _LEASE_SESSION_SAFE_RE,
    _LOG_ISSUE_RE as _LOG_ISSUE_RE,
    _MISSING_API_KEY_RE as _MISSING_API_KEY_RE,
    _MONTHS as _MONTHS,
    _MULTI_LANE_HOLD_LEDGER as _MULTI_LANE_HOLD_LEDGER,
    _NOOP_BANNER_RE as _NOOP_BANNER_RE,
    _PLANNING_EVENT_RE as _PLANNING_EVENT_RE,
    _PREREQ_MARKER as _PREREQ_MARKER,
    _PREVIEW_CONFIRM_FEEDBACK_RE as _PREVIEW_CONFIRM_FEEDBACK_RE,
    _PT_TO_UTC as _PT_TO_UTC,
    _REFUSAL_CLASSES as _REFUSAL_CLASSES,
    _REPAIR_ATTEMPT_LEDGER as _REPAIR_ATTEMPT_LEDGER,
    _REPAIR_LOG_RE as _REPAIR_LOG_RE,
    _RESET_AT_RE as _RESET_AT_RE,
    _RESTART_EXHAUSTED_RE as _RESTART_EXHAUSTED_RE,
    _RESTART_EXHAUSTED_TAIL_BYTES as _RESTART_EXHAUSTED_TAIL_BYTES,
    _RID_RE as _RID_RE,
    _ROLLING_HOLD_MAX_MIN as _ROLLING_HOLD_MAX_MIN,
    _SAME_ISSUE_WIP_CLAIMS as _SAME_ISSUE_WIP_CLAIMS,
    _SAME_ISSUE_WIP_RE as _SAME_ISSUE_WIP_RE,
    _SESSION_HOLD_MAX_MIN as _SESSION_HOLD_MAX_MIN,
    _SPAWN_CRASH_RE as _SPAWN_CRASH_RE,
    _SPAWN_LANE_RE as _SPAWN_LANE_RE,
    _STUB_LOG_MAX_BYTES as _STUB_LOG_MAX_BYTES,
    _TEST_EVENT_RE as _TEST_EVENT_RE,
    _TOOL_EVENT_RE as _TOOL_EVENT_RE,
    _TYPED_ESCALATION_RE as _TYPED_ESCALATION_RE,
    _WEEKDAYS as _WEEKDAYS,
    _WEEKLY_FALLBACK_MIN as _WEEKLY_FALLBACK_MIN,
    _WILL_RESET_LA_RE as _WILL_RESET_LA_RE,
    _WILL_RESET_RE as _WILL_RESET_RE,
    _WITNESS_OK as _WITNESS_OK,
    _WITNESS_STAMP_RE as _WITNESS_STAMP_RE,
    bind_ird as bind_ird,
    ird_attr as ird_attr,
)

# Re-export exact object expected by dispatch_no_window_test.test_ird_reexport_is_the_same_object.
no_window_creationflags = dispatch_worker.no_window_creationflags


class _GlobalsProxy:
    """Attribute view over a dynamically-loaded facade namespace."""

    def __init__(self, ns: dict[str, Any]) -> None:
        self._ns = ns

    def __getattr__(self, name: str) -> Any:
        try:
            return self._ns[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


_MOD_NS: Any = _GlobalsProxy(globals())


def _bind(fn: Any) -> Any:
    """Bind submodule lookups to this module instance during each test/CLI call."""
    @functools.wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        with _core.bind_ird(_MOD_NS):
            return fn(*args, **kwargs)

    return _wrapped


def _run_lease(root: Path, args: list[str], *, timeout: int = 30) -> dict[str, Any]:
    """Run one `fak leaseref` subcommand, returning {rc, stdout, verdict}. Never
    raises: an exec failure is reported as rc 127 so the caller fails open, and a probe
    that outlives `timeout` is reported as rc LEASE_TIMEOUT_RC.

    This deliberately does NOT use `subprocess.run(timeout=...)`: on Windows that
    timeout is unbounded whenever a descendant git holds the inherited pipe (#4368).
    Popen + a tree reap + a bounded second drain is what actually terminates."""
    bin_argv = _fak_bin(root)
    if not bin_argv:
        return {"rc": 127, "stdout": "", "verdict": None, "skipped": "no fak binary"}
    cmd = [*bin_argv, "leaseref", *args]
    kwargs: dict[str, Any] = {
        "cwd": str(root), "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True,
        "encoding": "utf-8", "errors": "replace",
    }
    if os.name == "nt":
        kwargs["creationflags"] = no_window_creationflags()
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(cmd, **kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"rc": 127, "stdout": "", "verdict": None, "error": str(exc)}
    try:
        stdout, _stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        killed = _terminate_lease_probe(proc)
        drained = True
        try:
            stdout, _stderr = proc.communicate(timeout=LEASE_DRAIN_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError):
            stdout, drained = "", False
        return {"rc": LEASE_TIMEOUT_RC, "stdout": (stdout or "").strip(),
                "verdict": None, "timeout": True, "timeout_s": timeout,
                "killed": killed, "drained": drained,
                "error": (f"LEASE_PROBE_TIMEOUT: `fak leaseref {' '.join(args)}` "
                          f"exceeded {timeout}s; reaped pid {killed.get('pid')} tree "
                          f"(ok={killed.get('ok')}, drained={drained})")}
    except (OSError, subprocess.SubprocessError) as exc:
        _terminate_lease_probe(proc)
        return {"rc": 127, "stdout": "", "verdict": None, "error": str(exc)}
    out = (stdout or "").strip()
    doc: Any = None
    try:
        doc = json.loads(out) if out else None
    except ValueError:
        doc = None
    return {"rc": proc.returncode, "stdout": out, "verdict": doc}


def issue_contract_review(root: Path, issue: dict[str, Any] | None,
                          number: int | None,
                          runner: Any = subprocess.run) -> dict[str, Any]:
    """Review issue readiness through ``fak-dev issue contract`` (issue_contract_gate floor: DEFAULT_ISSUE_CONTRACT_MIN_SCORE)."""
    with _core.bind_ird(_MOD_NS):
        tmp: Path | None = None
        overlay = read_contract_overlay(root / _core.ird_attr("RUNS_DIRNAME", RUNS_DIRNAME), number)
        record = _issue_record_for_contract(issue, number, overlay)
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json",
                                             delete=False) as f:
                tmp = Path(f.name)
                json.dump([record], f, ensure_ascii=False)
                f.write("\n")
            cmd = [
                *_fak_dev_command_prefix(root),
                "issue", "contract",
                "--from-issues", str(tmp),
                "--live", "--dedupe-checked", "--dedupe-cap", "300",
                "--json",
            ]
            proc = runner(
                cmd, cwd=str(root), capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=60, creationflags=no_window_creationflags())
        except (OSError, subprocess.SubprocessError) as exc:
            return {"ok": False, "unavailable": True, "score": 0, "reason": str(exc)}
        finally:
            if tmp is not None:
                try:
                    tmp.unlink()
                except OSError:
                    pass

        try:
            doc = json.loads(proc.stdout or "{}")
        except (TypeError, ValueError):
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            reason = tail[-1] if tail else f"fak issue contract failed ({proc.returncode})"
            return {"ok": False, "unavailable": True, "score": 0, "reason": reason[:300]}
        reviews = doc.get("reviews") if isinstance(doc, dict) else None
        review = reviews[0] if isinstance(reviews, list) and reviews else {}
        score = ((review.get("score") or {}).get("total") if isinstance(review, dict) else 0) or 0
        spine = ((review.get("spine_priority") or {}).get("total")
                 if isinstance(review, dict) else 0) or 0
        gate_hold = unresolved_acceptance_gate(record.get("body"))
        return {
            "ok": bool(proc.returncode == 0 and doc.get("ok") and review.get("ok")
                       and not gate_hold),
            "unavailable": False,
            "acceptance_gate_hold": gate_hold,
            "score": int(score),
            "spine_priority": int(spine),
            "review": review,
            "returncode": proc.returncode,
        }


def spawn_issue_worker(command: list[str], env: dict[str, str], cwd: Path,
                       log_dir: Path, issue: int, lane: str,
                       backend: str,
                       membership: dict[str, Any] | None = None,
                       account: dict[str, Any] | None = None,
                       lease: dict[str, Any] | None = None,
                       base_sha: str | None = None,
                       spawn_probe_s: float = 0.0,
                       log_prefix: str = "resolve",
                       prompt_payload: str | None = None,
                       worktree_git: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Launch a detached worker (claude/opencode/codex) after issue_contract_gate passes."""
    with _core.bind_ird(_MOD_NS):
        return _workers._spawn_issue_worker_impl(
            command, env, cwd, log_dir, issue, lane, backend,
            membership=membership, account=account, lease=lease,
            base_sha=base_sha, spawn_probe_s=spawn_probe_s,
            log_prefix=log_prefix, prompt_payload=prompt_payload,
            worktree_git=worktree_git)


is_orientation_doc = _bind(_core.is_orientation_doc)
worker_model_effort = _bind(_core.worker_model_effort)
launch_gate_for_guard = _bind(_core.launch_gate_for_guard)
refusal_class = _bind(_core.refusal_class)
launch_gate_blocked = _bind(_core.launch_gate_blocked)
win_creationflags = _bind(_core.win_creationflags)
repo_root = _bind(_core.repo_root)
_py = _bind(_core._py)
_fak_command_prefix = _bind(_core._fak_command_prefix)
_fak_dev_command_prefix = _bind(_core._fak_dev_command_prefix)
split_command_env = _bind(_core.split_command_env)
fak_loop_cmd = _bind(_core.fak_loop_cmd)
contract_overlay_path = _bind(_core.contract_overlay_path)
read_contract_overlay = _bind(_core.read_contract_overlay)
contract_overlay_times = _bind(_core.contract_overlay_times)
_issue_record_for_contract = _bind(_core._issue_record_for_contract)
unresolved_acceptance_gate = _bind(_core.unresolved_acceptance_gate)
issue_contract_hold_reason = _bind(_core.issue_contract_hold_reason)
_contract_missing_fields = _bind(_core._contract_missing_fields)
_git_capture = _bind(_core._git_capture)
_spawn_header_lane = _bind(_core._spawn_header_lane)
low_yield_soft_excludes = _bind(_core.low_yield_soft_excludes)
lane_issue_numbers = _bind(_core.lane_issue_numbers)
pick_target_issue = _bind(_core.pick_target_issue)
candidate_priority = _bind(_core.candidate_priority)
candidate_blocked_by = _bind(_core.candidate_blocked_by)
contract_scan_stream = _bind(_core.contract_scan_stream)
contract_held_issues = _bind(_core.contract_held_issues)
_missing_fields_from_hold_reason = _bind(_core._missing_fields_from_hold_reason)
contract_held_records = _bind(_core.contract_held_records)
record_contract_holds = _bind(_core.record_contract_holds)
record_contract_forced_bypass = _bind(_core.record_contract_forced_bypass)
contract_forced_bypass_count = _bind(_core.contract_forced_bypass_count)
multi_lane_held_records = _bind(_core.multi_lane_held_records)
multi_lane_held_issues = _bind(_core.multi_lane_held_issues)
record_multi_lane_holds = _bind(_core.record_multi_lane_holds)
collision_held_records = _bind(_core.collision_held_records)
collision_held_issues = _bind(_core.collision_held_issues)
record_collision_holds = _bind(_core.record_collision_holds)
_parse_gh_timestamp = _bind(_core._parse_gh_timestamp)
open_issue_updated_map = _bind(_core.open_issue_updated_map)
lane_tree = _bind(_core.lane_tree)
multi_lane_scope = _bind(_core.multi_lane_scope)
scan_multi_lane_scope = _bind(_core.scan_multi_lane_scope)
multi_lane_scope_reason = _bind(_core.multi_lane_scope_reason)
parse_git_status_paths = _bind(_core.parse_git_status_paths)
normalize_repo_path = _bind(_core.normalize_repo_path)
dirty_repo_paths = _bind(_core.dirty_repo_paths)
_path_text_variants = _bind(_core._path_text_variants)
text_mentions_repo_path = _bind(_core.text_mentions_repo_path)
dirty_path_collision = _bind(_core.dirty_path_collision)
dirty_path_collision_reason = _bind(_core.dirty_path_collision_reason)
_read_log_tail = _bind(_core._read_log_tail)
same_issue_wip_collision = _bind(_core.same_issue_wip_collision)
same_issue_wip_reason = _bind(_core.same_issue_wip_reason)
_candidate_issue_numbers = _bind(_core._candidate_issue_numbers)
capability_skips = _bind(_core.capability_skips)

_audit_path_from_log = _bind(_guards._audit_path_from_log)
_guard_result_livelock = _bind(_guards._guard_result_livelock)
active_guard_livelock_hold = _bind(_guards.active_guard_livelock_hold)
_compact_runaway_from_log = _bind(_guards._compact_runaway_from_log)
active_compact_runaway_hold = _bind(_guards.active_compact_runaway_hold)
_backend_of_log = _bind(_guards._backend_of_log)
_account_tag_of_log = _bind(_guards._account_tag_of_log)
_log_tail_text = _bind(_guards._log_tail_text)
_log_is_cap_banner = _bind(_guards._log_is_cap_banner)
_log_is_auth_gap = _bind(_guards._log_is_auth_gap)
_cap_hit_from_text = _bind(_guards._cap_hit_from_text)
_write_cap_hold = _bind(_guards._write_cap_hold)
_safe_cap_account_component = _bind(_guards._safe_cap_account_component)
_cap_state_paths = _bind(_guards._cap_state_paths)
_scan_recent_cap_banner = _bind(_guards._scan_recent_cap_banner)
_persisted_hold_matches_account = _bind(_guards._persisted_hold_matches_account)
_parse_reset_to_utc = _bind(_guards._parse_reset_to_utc)
_clamp_reset = _bind(_guards._clamp_reset)
_parse_relative_wait = _bind(_guards._parse_relative_wait)
_iso_to_utc = _bind(_guards._iso_to_utc)
check_weekly_cap = _bind(_guards.check_weekly_cap)
_classify_backend_logs = _bind(_guards._classify_backend_logs)
check_backend_health = _bind(_guards.check_backend_health)
read_dead_backends = _bind(_guards.read_dead_backends)
seat_adaptive_target = _bind(_guards.seat_adaptive_target)
recent_backend_stub_rate = _bind(_guards.recent_backend_stub_rate)
gate_spawn_on_health = _bind(_guards.gate_spawn_on_health)
gate_opencode_gateway = _bind(_guards.gate_opencode_gateway)
guard_build_id = _bind(_guards.guard_build_id)
planned_guard_flags = _bind(_guards.planned_guard_flags)
probe_guard_flags = _bind(_guards.probe_guard_flags)
guard_flag_inventory = _bind(_guards.guard_flag_inventory)
gate_spawn_on_guard_argv = _bind(_guards.gate_spawn_on_guard_argv)
_terminate_lease_probe = _bind(_guards._terminate_lease_probe)
_fak_bin = _bind(_guards._fak_bin)
lease_holder = _bind(_guards.lease_holder)
lease_owner_override = _bind(_guards.lease_owner_override)
lease_session_id = _bind(_guards.lease_session_id)
_safe_lease_session_segment = _bind(_guards._safe_lease_session_segment)
synthetic_lease_session_id = _bind(_guards.synthetic_lease_session_id)
lease_id_for_lane = _bind(_guards.lease_id_for_lane)
publish_lease_session = _bind(_guards.publish_lease_session)
acquire_lane_lease = _bind(_guards.acquire_lane_lease)
release_lane_lease = _bind(_guards.release_lane_lease)
reap_expired_leases = _bind(_guards.reap_expired_leases)
live_lane_lease_lanes = _bind(_guards.live_lane_lease_lanes)

read_lease_sidecar = _bind(_witness.read_lease_sidecar)
classify_restart_exhaustion = _bind(_witness.classify_restart_exhaustion)
refine_unknown_no_commit = _bind(_witness.refine_unknown_no_commit)
classify_no_commit_reason = _bind(_witness.classify_no_commit_reason)
_spawn_header_only = _bind(_witness._spawn_header_only)
classify_spawn_failed_cause = _bind(_witness.classify_spawn_failed_cause)
held_no_commit_issues = _bind(_witness.held_no_commit_issues)
issue_witness_history = _bind(_witness.issue_witness_history)
reblock_streak_held_records = _bind(_witness.reblock_streak_held_records)
reblock_streak_held_issues = _bind(_witness.reblock_streak_held_issues)
_subject_cites_issue = _bind(_witness._subject_cites_issue)
worker_resolving_sha = _bind(_witness.worker_resolving_sha)
_run_commit_audit = _bind(_witness._run_commit_audit)
audit_commit_witness = _bind(_witness.audit_commit_witness)
_run_commit_audit_rows = _bind(_witness._run_commit_audit_rows)
locally_witnessed_issues = _bind(_witness.locally_witnessed_issues)
commit_audit_abstain_holds = _bind(_witness.commit_audit_abstain_holds)
open_witnessed_dispositions = _bind(_witness.open_witnessed_dispositions)
_run_close_arm = _bind(_witness._run_close_arm)
close_open_witnessed = _bind(_witness.close_open_witnessed)
witness_exited_workers = _bind(_witness.witness_exited_workers)
is_dos_run_id = _bind(_witness.is_dos_run_id)
mint_dos_run_id = _bind(_witness.mint_dos_run_id)
loop_id_for_payload = _bind(_witness.loop_id_for_payload)
default_loop_ledger = _bind(_witness.default_loop_ledger)
_loop_metric_args = _bind(_witness._loop_metric_args)
_loop_evidence_args = _bind(_witness._loop_evidence_args)
_dispatch_collision_evidence = _bind(_witness._dispatch_collision_evidence)
append_loop_event = _bind(_witness.append_loop_event)
loop_run_id = _bind(_witness.loop_run_id)
record_loop_tick = _bind(_witness.record_loop_tick)
append_fleet_trend_row = _bind(_witness.append_fleet_trend_row)

live_resolution_issues = _bind(_workers.live_resolution_issues)
live_resolution_lanes = _bind(_workers.live_resolution_lanes)
recently_attempted_issues = _bind(_workers.recently_attempted_issues)
live_repair_workers = _bind(_workers.live_repair_workers)
record_repair_attempt = _bind(_workers.record_repair_attempt)
recently_repaired_issues = _bind(_workers.recently_repaired_issues)
terminate_issue_worker_tree = _bind(_workers.terminate_issue_worker_tree)
bounded_worker_progress = _bind(_workers.bounded_worker_progress)
_compact_investigation_summary = _bind(_workers._compact_investigation_summary)
reap_timed_out_workers = _bind(_workers.reap_timed_out_workers)
prune_dead_sidecars = _bind(_workers.prune_dead_sidecars)
build_worker_command = _bind(_workers.build_worker_command)
_command_basename = _bind(_workers._command_basename)
unwrap_opencode_npm_shim = _bind(_workers.unwrap_opencode_npm_shim)
resolve_worker_executable = _bind(_workers.resolve_worker_executable)
attach_opencode_prompt_file = _bind(_workers.attach_opencode_prompt_file)
resolve_opencode_command = _bind(_workers.resolve_opencode_command)
_opencode_config_home = _bind(_workers._opencode_config_home)
_github_cli_config_dir = _bind(_workers._github_cli_config_dir)
_opencode_temp_dir = _bind(_workers._opencode_temp_dir)
opencode_worker_env = _bind(_workers.opencode_worker_env)
codex_worker_env = _bind(_workers.codex_worker_env)
wave_membership_env = _bind(_workers.wave_membership_env)
write_wave_sidecar = _bind(_workers.write_wave_sidecar)
write_account_sidecar = _bind(_workers.write_account_sidecar)
dispatch_account_id = _bind(_workers.dispatch_account_id)
stamp_dispatch_account = _bind(_workers.stamp_dispatch_account)
write_lease_sidecar = _bind(_workers.write_lease_sidecar)
probe_spawned_worker = _bind(_workers.probe_spawned_worker)
enumerate_wave = _bind(_workers.enumerate_wave)
_spawn_failure_streak_path = _bind(_workers._spawn_failure_streak_path)
_read_spawn_failure_streaks = _bind(_workers._read_spawn_failure_streaks)
_write_spawn_failure_streaks = _bind(_workers._write_spawn_failure_streaks)
bump_spawn_failure_streak = _bind(_workers.bump_spawn_failure_streak)
clear_spawn_failure_streak = _bind(_workers.clear_spawn_failure_streak)
_spawn_failure_streak_seat_path = _bind(_workers._spawn_failure_streak_seat_path)
_read_seat_failure_streaks = _bind(_workers._read_seat_failure_streaks)
_write_seat_failure_streaks = _bind(_workers._write_seat_failure_streaks)
bump_spawn_failure_streak_seat = _bind(_workers.bump_spawn_failure_streak_seat)
clear_spawn_failure_streak_seat = _bind(_workers.clear_spawn_failure_streak_seat)
seat_spawn_failure_state = _bind(_workers.seat_spawn_failure_state)
tick_exit_code = _bind(_workers.tick_exit_code)
_record = _bind(_workers._record)

_maybe_dispatch_contract_repair = _bind(_eval._maybe_dispatch_contract_repair)
evaluate = _bind(_eval.evaluate)
resolve_spawn_burst = _bind(_eval.resolve_spawn_burst)
resolve_spawn_burst_stagger = _bind(_eval.resolve_spawn_burst_stagger)
evaluate_burst = _bind(_eval.evaluate_burst)
_burst_stagger_sleep = _bind(_eval._burst_stagger_sleep)
render = _bind(_eval.render)


def _emit_low_yield_excludes(root: Path) -> int:
    """Emit the #2062 low-yield soft-exclude lane set as JSON and exit."""
    with _core.bind_ird(_MOD_NS):
        runs_dir = root / RUNS_DIRNAME
        try:
            _concurrent, low_yield_trees, _exclusive = issue_lane_router.lane_taxonomy(root)
        except Exception:
            low_yield_trees = {}
        low_yield = low_yield_soft_excludes(root, runs_dir, lane_trees=low_yield_trees)
        payload = {
            "exclude": sorted(str(x) for x in (low_yield.get("exclude") or set())),
            "lanes": low_yield.get("lanes") or [],
            "flagged": low_yield.get("flagged") or [],
        }
        print(json.dumps(payload, indent=2))
        return 0


def main(argv: list[str] | None = None) -> int:
    with _core.bind_ird(_MOD_NS):
        ap = argparse.ArgumentParser(
            description="One guarded tick that spawns an issue-RESOLUTION worker (dry-run by default).")
        ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
        ap.add_argument("--max-workers", type=int, default=20,
                        help="fail-safe cap on live workers, enforced by the preflight "
                             "(default: 20). With seat-adaptive sizing (the default, "
                             "#3246) this binds only when the preflight returns no seat "
                             "signal; with --no-seat-adaptive it is the hard cap.")
        ap.add_argument("--no-seat-adaptive", action="store_true",
                        help="disable seat-adaptive tick sizing (#3246): keep the "
                             "configured --max-workers as the binding cap instead of "
                             "growing to min(live+seat_free, host_cap, --seat-ceiling, "
                             "live+--seat-ramp-delta) when the preflight carries a "
                             "seat signal")
        ap.add_argument("--seat-ceiling", type=int, default=DEFAULT_SEAT_CEILING,
                        help=f"hard fail-safe ceiling on the seat-adaptive cap (default "
                             f"{DEFAULT_SEAT_CEILING}). An explicit --max-workers above "
                             f"it raises it — seat sizing never shrinks an operator's "
                             f"configured cap.")
        ap.add_argument("--seat-ramp-delta", type=int, default=DEFAULT_SEAT_RAMP_DELTA,
                        help=f"max NEW workers one tick may plan above the live count "
                             f"(canary-safe ramp, #3246; default "
                             f"{DEFAULT_SEAT_RAMP_DELTA}; 0 disables the ramp bound so "
                             f"the cap jumps straight to the seat/host minimum)")
        ap.add_argument("--spawn-burst", type=int, default=None,
                        help=f"max workers ONE tick may spawn, filling live headroom "
                             f"(default {DEFAULT_SPAWN_BURST} = one spawn per tick; "
                             f"hard-clamped to {SPAWN_BURST_HARD_CEILING}; env {SPAWN_BURST_ENV}).")
        ap.add_argument("--spawn-burst-stagger-s", type=float, default=None,
                        help=f"seconds between spawns inside one burst (default "
                             f"{DEFAULT_SPAWN_BURST_STAGGER_S}; env "
                             f"{SPAWN_BURST_STAGGER_ENV}; clamped to "
                             f"{SPAWN_BURST_STAGGER_MAX_S}).")
        ap.add_argument("--work-kind", default=None,
                        help="switcher work kind (engineering->t1, gardening->t2).")
        ap.add_argument("--lane", default=None,
                        help="explicit lane (default: the lane with the most open issues)")
        ap.add_argument("--issue", type=int, default=None,
                        help="pin an explicit target issue #N (requires --lane).")
        ap.add_argument("--backend", choices=BACKENDS, default="claude",
                        help="worker backend: claude, opencode, or codex. Default claude.")
        ap.add_argument("--exclude-lane", default="",
                        help="comma-separated lanes to drop from the busiest-pick.")
        ap.add_argument("--live", action="store_true",
                        help="actually spawn the worker (default: dry-run / plan only)")
        ap.add_argument("--force", action="store_true",
                        help="operator best-effort: bypass the issue-contract hold with --force-reason.")
        ap.add_argument("--force-reason", default="",
                        help="structured operator reason recorded when --force bypasses a failed gate.")
        ap.add_argument("--no-refresh", action="store_true",
                        help="skip the per-tick account-registry refresh")
        ap.add_argument("--cooldown-min", type=int, default=120,
                        help="skip an issue attempted within this many minutes (0 disables).")
        ap.add_argument("--worker-timeout-s", type=int, default=DEFAULT_WORKER_TIMEOUT_S,
                        help=f"wall-clock cap for live resolver workers (default: {DEFAULT_WORKER_TIMEOUT_S}; 0 disables)")
        ap.add_argument("--spawn-probe-s", type=float, default=DEFAULT_SPAWN_PROBE_S,
                        help=f"seconds to wait after a live spawn (default: {DEFAULT_SPAWN_PROBE_S}; 0 disables)")
        ap.add_argument("--max-realloc-workers", type=int, default=DEFAULT_REALLOC_CEILING,
                        help=f"ceiling on extra worker slots claimed from dead siblings (default {DEFAULT_REALLOC_CEILING})")
        ap.add_argument("--contract-scan", type=int, default=DEFAULT_CONTRACT_SCAN,
                        help=f"max lane issues one tick may contract-review (default {DEFAULT_CONTRACT_SCAN}).")
        ap.add_argument("--contract-hold-ttl-h", type=int,
                        default=DEFAULT_CONTRACT_HOLD_TTL_H,
                        help=f"skip contract-held issues within this many hours (default {DEFAULT_CONTRACT_HOLD_TTL_H}; 0 disables).")
        ap.add_argument("--repair-batch", type=int, default=DEFAULT_REPAIR_BATCH,
                        help=f"max held issues sent to a contract-repair worker (default {DEFAULT_REPAIR_BATCH}; 0 disables).")
        ap.add_argument("--repair-cooldown-min", type=int,
                        default=DEFAULT_REPAIR_COOLDOWN_MIN,
                        help=f"skip issues attempted by repair within this many minutes (default {DEFAULT_REPAIR_COOLDOWN_MIN}; 0 disables).")
        ap.add_argument("--loop-ledger", default="",
                        help="append this tick to a fak loop ledger (default: FAK_LOOP_LEDGER or .fak/loops.jsonl)")
        ap.add_argument("--no-loop-ledger", action="store_true",
                        help="disable loop-ledger append for this tick")
        ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
        ap.add_argument("--low-yield-excludes", action="store_true",
                        help="emit the #2062 low-yield soft-exclude lane set as JSON and exit")
        args = ap.parse_args(argv)

        root = Path(args.workspace).resolve() if args.workspace else repo_root()
        if args.low_yield_excludes:
            return _emit_low_yield_excludes(root)
        if args.issue is not None and not args.lane:
            ap.error("--issue requires --lane so the lane lease/tree matches the pinned issue")
        exclude_lanes = {s.strip() for s in args.exclude_lane.split(",") if s.strip()}
        work_kind = args.work_kind or ("gardening" if args.backend == "opencode" else "engineering")
        burst = resolve_spawn_burst(args.spawn_burst)
        payload = evaluate_burst(
            root,
            spawn_burst=burst["limit"],
            burst_stagger_s=resolve_spawn_burst_stagger(args.spawn_burst_stagger_s),
            max_workers=args.max_workers, work_kind=work_kind,
            lane=args.lane, live=args.live, refresh=not args.no_refresh,
            cooldown_min=args.cooldown_min, backend=args.backend,
            exclude_lanes=exclude_lanes,
            worker_timeout_s=dispatch_worker.normalize_timeout(args.worker_timeout_s),
            spawn_probe_s=max(0.0, args.spawn_probe_s),
            realloc_ceiling=max(0, args.max_realloc_workers),
            seat_adaptive=not args.no_seat_adaptive,
            seat_ceiling=max(1, args.seat_ceiling),
            seat_ramp_delta=max(0, args.seat_ramp_delta),
            record_loop=not args.no_loop_ledger,
            loop_ledger=(Path(args.loop_ledger).resolve() if args.loop_ledger else None),
            issue_override=args.issue,
            force=args.force,
            force_reason=args.force_reason,
            contract_scan=max(1, args.contract_scan),
            contract_hold_ttl_h=max(0, args.contract_hold_ttl_h),
            repair_batch=max(0, args.repair_batch),
            repair_cooldown_min=max(0, args.repair_cooldown_min))
        if isinstance(payload.get("burst"), dict):
            payload["burst"].update({k: v for k, v in burst.items() if k != "limit"})
        print(json.dumps(payload, indent=2) if args.json else render(payload))
        return tick_exit_code(payload)


if __name__ == "__main__":
    raise SystemExit(main())
