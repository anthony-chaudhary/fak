#!/usr/bin/env python3
"""Tick evaluation orchestration helpers, contract repair dispatch, burst fan-out, and text rendering."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from issue_resolve_dispatch_core import (
    BACKENDS,
    DEFAULT_COLLISION_HOLD_TTL_H,
    DEFAULT_CONTRACT_HOLD_TTL_H,
    DEFAULT_CONTRACT_SCAN,
    DEFAULT_ISSUE_CONTRACT_MIN_SCORE,
    DEFAULT_MULTI_LANE_HOLD_TTL_H,
    DEFAULT_REALLOC_CEILING,
    DEFAULT_REBLOCK_STREAK_HOLD_N,
    DEFAULT_REBLOCK_STREAK_HOLD_TTL_H,
    DEFAULT_REPAIR_BATCH,
    DEFAULT_REPAIR_COOLDOWN_MIN,
    DEFAULT_SEAT_ADAPTIVE,
    DEFAULT_SEAT_CEILING,
    DEFAULT_SEAT_RAMP_DELTA,
    DEFAULT_SPAWN_BURST,
    DEFAULT_SPAWN_BURST_STAGGER_S,
    DEFAULT_SPAWN_PROBE_S,
    DEFAULT_WORKER_TIMEOUT_S,
    LEASE_TTL_MARGIN_S,
    REFUSAL_CLASS_LANE_LEASE,
    REFUSAL_CLASS_WORKTREE_COTENANCY,
    REPAIR_ISSUES_SIDECAR_SUFFIX,
    REPAIR_LANE,
    RUNS_DIRNAME,
    SCHEMA,
    SEAT_STREAK_REPROBE_MIN,
    SPAWN_BURST_ENV,
    SPAWN_BURST_HARD_CEILING,
    SPAWN_BURST_STAGGER_ENV,
    SPAWN_BURST_STAGGER_MAX_S,
    _BACKEND_PRODUCT,
    _BACKEND_REPROBE_MIN,
    _HEALTH_SKIP_STUB_RATE,
    _candidate_issue_numbers,
    _contract_missing_fields,
    _git_capture,
    capability_skips,
    candidate_blocked_by,
    candidate_priority,
    collision_held_issues,
    contract_forced_bypass_count,
    contract_held_issues,
    contract_held_records,
    contract_overlay_times,
    contract_scan_stream,
    dirty_path_collision,
    dirty_path_collision_reason,
    dirty_repo_paths,
    dispatch_worker,
    ird_attr,
    issue_contract_hold_reason,
    issue_dispatch,
    issue_lane_router,
    issue_worker_prompt,
    lane_issue_numbers,
    lane_tree,
    launch_gate_blocked,
    launch_gate_for_guard,
    low_yield_soft_excludes,
    multi_lane_held_issues,
    multi_lane_scope_reason,
    open_issue_updated_map,
    pick_target_issue,
    record_collision_holds,
    record_contract_forced_bypass,
    record_contract_holds,
    record_multi_lane_holds,
    same_issue_wip_collision,
    same_issue_wip_reason,
    scan_multi_lane_scope,
    worker_model_effort,
)
from issue_resolve_dispatch_guards import (
    _cap_hit_from_text,
    _write_cap_hold,
    acquire_lane_lease,
    active_compact_runaway_hold,
    active_guard_livelock_hold,
    check_backend_health,
    check_weekly_cap,
    gate_opencode_gateway,
    gate_spawn_on_guard_argv,
    gate_spawn_on_health,
    live_lane_lease_lanes,
    read_dead_backends,
    reap_expired_leases,
    recent_backend_stub_rate,
    release_lane_lease,
    seat_adaptive_target,
)
from issue_resolve_dispatch_witness import (
    append_fleet_trend_row,
    classify_spawn_failed_cause,
    close_open_witnessed,
    commit_audit_abstain_holds,
    held_no_commit_issues,
    locally_witnessed_issues,
    open_witnessed_dispositions,
    reblock_streak_held_issues,
    record_loop_tick,
    witness_exited_workers,
)
from issue_resolve_dispatch_workers import (
    _record,
    _spawn_issue_worker_impl,
    build_worker_command,
    bump_spawn_failure_streak,
    bump_spawn_failure_streak_seat,
    clear_spawn_failure_streak,
    clear_spawn_failure_streak_seat,
    codex_worker_env,
    live_repair_workers,
    live_resolution_issues,
    live_resolution_lanes,
    opencode_worker_env,
    prune_dead_sidecars,
    reap_timed_out_workers,
    recently_attempted_issues,
    recently_repaired_issues,
    record_repair_attempt,
    seat_spawn_failure_state,
)

_IMPL_FIRST_BUDGET_SUFFIX = (
    "\n\nimplementation-first budget (#8816): investigate for at most 5 minutes "
    "and make the first declared-path edit by then. Status is inferred from "
    "tool events as investigating/editing/testing/committing/stalled; do not "
    "self-report it. Do not launch a planner, fleet, or nested orchestration. "
    "Escalate only by emitting ESCALATION_NEED: followed by one of "
    "missing-path, missing-acceptance, missing-capability, or unsafe-scope, "
    "then wait for coordinator admission. A no-edit timeout preserves a compact "
    "investigation summary for one resume or reassignment.\n"
)


def _preflight_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _preflight_public(doc: dict[str, Any]) -> dict[str, Any]:
    out = {"verdict": doc.get("verdict"), "reason": doc.get("reason"),
           "cap": doc.get("cap"), "live": doc.get("live")}
    for key in ("headroom", "max_workers", "host_cap"):
        if doc.get(key) is not None:
            out[key] = doc.get(key)
    limiter = doc.get("capacity_limiter")
    if isinstance(limiter, dict):
        out["capacity_limiter"] = limiter
    seat = doc.get("seat")
    if isinstance(seat, dict):
        out["seat"] = {k: seat.get(k) for k in (
            "total", "free", "leased", "depleted", "unattributed_live")
            if seat.get(k) is not None}
    return out


def _preflight_refusal_hint(doc: dict[str, Any]) -> dict[str, Any] | None:
    if doc.get("verdict") != "REFUSE_AT_CAP":
        return None
    limiter = doc.get("capacity_limiter") if isinstance(doc.get("capacity_limiter"), dict) else {}
    raw = limiter.get("raw") if isinstance(limiter.get("raw"), dict) else {}
    term = str(limiter.get("term") or "")
    maxw = _preflight_int(raw.get("max_workers"))
    if maxw is None:
        maxw = _preflight_int(doc.get("max_workers"))
    host_cap = _preflight_int(raw.get("host_cap"))
    if host_cap is None:
        host_cap = _preflight_int(doc.get("host_cap"))
    live_count = _preflight_int(doc.get("live"))
    cap_count = _preflight_int(doc.get("cap"))
    needed = live_count + 1 if live_count is not None else None
    configured_cap = term == "max_workers" or (
        maxw is not None and cap_count == maxw
        and host_cap is not None and host_cap > maxw)
    host_headroom_available = (
        host_cap is None or live_count is None or live_count < host_cap)
    needed_within_host = needed is None or host_cap is None or needed <= host_cap
    if configured_cap and host_headroom_available and needed_within_host:
        return {
            "kind": "configured_max_workers",
            "message": (
                f"configured --max-workers={maxw} is the binding cap; "
                + (f"rerun with --max-workers >= {needed} " if needed else "raise --max-workers ")
                + (f"(still bounded by host_cap={host_cap}) "
                   if host_cap is not None else "")
                + "only if the operator intends to use available host headroom"),
            "max_workers": maxw,
            "required_min": needed,
            "host_cap": host_cap,
        }
    if limiter:
        return {
            "kind": term or str(limiter.get("primary") or "capacity"),
            "message": (f"capacity limiter {limiter.get('primary') or '?'}"
                        f"/{term or '?'} is binding; do not route around "
                        "the preflight refusal"),
        }
    return None


def _account_public(account: dict[str, Any]) -> dict[str, Any]:
    return {k: account.get(k) for k in ("tag", "tier", "model", "dir")}


def _same_account(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for key in ("tag", "dir"):
        lv = str(left.get(key) or "")
        rv = str(right.get(key) or "")
        if lv and rv and lv == rv:
            return True
    return False


def _contract_holds(contract: dict[str, Any]) -> bool:
    min_score = ird_attr("DEFAULT_ISSUE_CONTRACT_MIN_SCORE", DEFAULT_ISSUE_CONTRACT_MIN_SCORE)
    return bool(contract.get("unavailable") or not contract.get("ok") or
                int(contract.get("score") or 0) < min_score)


def _candidate_text(rec: dict[str, Any]) -> str:
    return f"{rec.get('title') or ''}\n{(rec.get('issue_record') or {}).get('body') or ''}"


def _finish_tick(
    root: Path,
    runs_dir: Path,
    payload: dict[str, Any],
    *,
    live: bool,
    record_loop: bool,
    loop_ledger: Path | None,
) -> dict[str, Any]:
    if record_loop:
        payload["loop_ledger"] = ird_attr("record_loop_tick", record_loop_tick)(
            root, payload, ledger=loop_ledger)
    if live:
        payload["fleet_trend"] = ird_attr("append_fleet_trend_row", append_fleet_trend_row)(
            root, payload)
    _record(runs_dir, payload)
    return payload


def _maybe_dispatch_contract_repair(
    root: Path, runs_dir: Path, payload: dict[str, Any], *,
    rows: list[dict[str, Any]], live: bool, backend: str,
    acct: dict[str, Any], repair_batch: int, repair_cooldown_min: int,
    spawn_probe_s: float, product: str,
) -> dict[str, Any] | None:
    """Self-serve arm of the readiness gate: dispatch one worker to groom held issues."""
    if repair_batch <= 0:
        payload["contract_repair"] = {"skipped": "disabled (--repair-batch 0)"}
        return None
    in_flight = ird_attr("live_repair_workers", live_repair_workers)(runs_dir)
    if in_flight:
        payload["contract_repair"] = {"in_flight": in_flight}
        payload.update({
            "ok": True, "action": "repair_in_flight", "verdict": "REPAIR_IN_FLIGHT",
            "reason": (f"every scanned candidate fails the contract gate and a "
                       f"contract-repair worker (pid {in_flight[0].get('pid')}) is "
                       f"already grooming the backlog — waiting for it, not "
                       f"stacking a second groomer onto the same issues")})
        return payload
    cooled = ird_attr("recently_repaired_issues", recently_repaired_issues)(
        runs_dir, cooldown_min=repair_cooldown_min)
    batch = [r for r in rows if int(r.get("issue") or 0)
             and int(r.get("issue") or 0) not in cooled][:repair_batch]
    if not batch:
        payload["contract_repair"] = {
            "skipped": (f"all {len(rows)} held candidate(s) inside the "
                        f"{repair_cooldown_min}-min repair cooldown")}
        return None
    prompt_rows = [{
        "number": r.get("issue"),
        "title": r.get("title"),
        "missing_fields": r.get("missing_fields") or [],
        "reasons": [t for t in str(r.get("reason") or "").split(", ")
                    if t.startswith("ISSUE_")][:4],
    } for r in batch]
    iwp_mod = ird_attr("issue_worker_prompt", issue_worker_prompt)
    dw_mod = ird_attr("dispatch_worker", dispatch_worker)
    id_mod = ird_attr("issue_dispatch", issue_dispatch)
    min_score = ird_attr("DEFAULT_ISSUE_CONTRACT_MIN_SCORE", DEFAULT_ISSUE_CONTRACT_MIN_SCORE)
    rec = iwp_mod.build_repair(prompt_rows, workspace=root, min_score=min_score)
    payload["contract_repair"] = {"batch": rec["issues"],
                                  "prompt_chars": rec["prompt_chars"]}
    nums = ",".join(str(n) for n in rec["issues"])
    model, effort = worker_model_effort(backend, acct)
    if not live:
        preview_prompt = f"<contract-repair prompt, {rec.get('prompt_chars')} chars>"
        preview_env = dw_mod.child_env(REPAIR_LANE, backend, root)
        preview_command, preview_guarded = dw_mod.guarded_launch_command(
            build_worker_command(backend, preview_prompt, model, effort),
            REPAIR_LANE, backend, root, preview_env)
        payload["command"] = preview_command
        payload["guarded"] = preview_guarded
        payload["launch_gate"] = launch_gate_for_guard(preview_guarded, backend)
        payload.update({
            "ok": True, "action": "would_repair", "verdict": "WOULD_REPAIR",
            "reason": (f"every scanned candidate fails the contract gate; would "
                       f"spawn 1 {backend} contract-repair worker on issue(s) "
                       f"{nums} to bring them up to contract")})
        return payload
    if backend == "claude":
        env = id_mod.worker_env(acct.get("dir"), REPAIR_LANE, root)
    elif backend == "codex":
        env = codex_worker_env(acct.get("dir"), REPAIR_LANE, root)
    else:
        env = opencode_worker_env(acct.get("dir"), REPAIR_LANE, root, runs_dir)
    env["FLEET_REPAIR_ISSUES"] = nums
    command, guarded = dw_mod.guarded_launch_command(
        build_worker_command(backend, rec["prompt"], model, effort),
        REPAIR_LANE, backend, root, env)
    if guarded:
        dw_mod.guard_env_augment(env)
    payload["command"] = command
    payload["guarded"] = guarded
    spawn_fn = ird_attr("spawn_issue_worker", _spawn_issue_worker_impl)
    spawned = spawn_fn(command, env, root, runs_dir, rec["issues"][0],
                       REPAIR_LANE, backend, account=acct,
                       spawn_probe_s=spawn_probe_s, log_prefix="repair",
                       prompt_payload=rec["prompt"] if backend in ("claude", "opencode", "codex") else None)
    record_repair_attempt(runs_dir, rec["issues"], live=live)
    try:
        Path(str(spawned.get("log") or "")).with_suffix(
            REPAIR_ISSUES_SIDECAR_SUFFIX).write_text(nums, encoding="utf-8")
    except OSError:
        pass
    early = spawned.get("early_exit") or {}
    if early.get("checked") and not early.get("alive"):
        cap_hit = _cap_hit_from_text(
            str(early.get("tail") or ""),
            evidence_log=Path(str(spawned.get("log") or "")).name)
        if cap_hit:
            import time
            payload["quota_cap"] = _write_cap_hold(
                runs_dir, product=product, account_tag=acct.get("tag"),
                hit=cap_hit, now_ts=time.time(), fallback_min=60,
                source="early_exit")
        payload.update({
            "ok": False, "action": "spawn_failed", "verdict": "SPAWN_FAILED",
            "spawned": spawned,
            "cause": classify_spawn_failed_cause(early),
            "seat_streak": bump_spawn_failure_streak_seat(
                runs_dir, acct.get("tag"), backend),
            "reason": (f"{backend} contract-repair worker pid {spawned['pid']} "
                       f"for issue(s) {nums} exited within {early.get('wait_s')}s "
                       f"with code {early.get('returncode')}"
                       + (" and produced an empty log" if early.get("silent") else ""))})
        _record(runs_dir, payload)
        return payload
    clear_spawn_failure_streak_seat(runs_dir, acct.get("tag"), backend)
    payload.update({
        "ok": True, "action": "repair_spawned", "verdict": "REPAIR_SPAWNED",
        "spawned": spawned,
        "reason": (f"spawned {backend} contract-repair worker pid {spawned['pid']} "
                   f"on issue(s) {nums} — it edits the ISSUES up to contract "
                   f"(no lane lease; the repo tree is read-only for it)")})
    _record(runs_dir, payload)
    return payload


def _run_pre_tick_housekeeping(
    root: Path,
    runs_dir: Path,
    *,
    backend: str,
    product: str,
    work_kind: str,
    max_workers: int,
    lane: str | None,
    live: bool,
    refresh: bool,
    worker_timeout_s: int | None,
    realloc_ceiling: int,
    seat_adaptive: bool,
    seat_ceiling: int,
    seat_ramp_delta: int,
    lease_runner: Any | None,
) -> dict[str, Any]:
    id_mod = ird_attr("issue_dispatch", issue_dispatch)
    reg = id_mod.refresh_registry(root) if refresh else {"skipped": True}
    reaped = ird_attr("reap_timed_out_workers", reap_timed_out_workers)(
        runs_dir, timeout_s=worker_timeout_s, live=live)
    witnessed = (
        ird_attr("witness_exited_workers", witness_exited_workers)(
            runs_dir, root, live=live, lease_runner=lease_runner)
        if live else {"skipped": True}
    )
    pruned = ird_attr("prune_dead_sidecars", prune_dead_sidecars)(runs_dir, live=live)
    leases_reaped = (
        ird_attr("reap_expired_leases", reap_expired_leases)(root, runner=lease_runner)
        if live else {"skipped": True}
    )
    own_health = ird_attr("check_backend_health", check_backend_health)(
        runs_dir, product=product, lane=lane)
    realloc: dict[str, Any] = {"claimed_lanes": [], "bonus": 0, "from": []}
    eff_max_workers = max_workers
    if own_health.get("state") != "dead":
        dead_siblings = ird_attr("read_dead_backends", read_dead_backends)(
            runs_dir, exclude=product)
        if dead_siblings:
            bonus = min(len(dead_siblings), max(0, realloc_ceiling))
            eff_max_workers = max_workers + bonus
            realloc = {
                "claimed_lanes": [d.get("abandoned_lane") for d in dead_siblings
                                  if d.get("abandoned_lane")],
                "bonus": bonus,
                "from": [d.get("product") for d in dead_siblings],
            }
    pre = id_mod.preflight(root, max_workers=eff_max_workers, work_kind=work_kind,
                           product=product)
    health_stub_rate = ird_attr("recent_backend_stub_rate", recent_backend_stub_rate)(
        runs_dir, product=product)
    eff_max_workers, health_skip_reason = gate_spawn_on_health(
        eff_max_workers, health_stub_rate)
    seat_sizing: dict[str, Any] | None = None
    if seat_adaptive and health_skip_reason is None:
        seat_target, seat_sizing = seat_adaptive_target(
            pre, fallback=eff_max_workers, ceiling=seat_ceiling,
            ramp_delta=seat_ramp_delta)
        if seat_sizing.get("signal_available") and seat_target != eff_max_workers:
            eff_max_workers = seat_target
            seat_sizing["reprobed"] = True
            pre = id_mod.preflight(root, max_workers=eff_max_workers,
                                   work_kind=work_kind, product=product)
    pre_ok = pre.get("verdict") == "SPAWN_OK"
    acct = pre.get("account") or {}
    gateway_skip_reason = None
    if backend == "opencode" and pre_ok:
        eff_max_workers, gateway_skip_reason = ird_attr(
            "gate_opencode_gateway", gate_opencode_gateway)(eff_max_workers, acct)
    return {
        "reg": reg, "reaped": reaped, "witnessed": witnessed, "pruned": pruned,
        "leases_reaped": leases_reaped, "own_health": own_health, "realloc": realloc,
        "eff_max_workers": eff_max_workers, "pre": pre, "pre_ok": pre_ok, "acct": acct,
        "health_stub_rate": health_stub_rate, "health_skip_reason": health_skip_reason,
        "seat_sizing": seat_sizing, "gateway_skip_reason": gateway_skip_reason,
    }


def _try_weekly_cap_reroute(
    root: Path, runs_dir: Path, ctx: dict[str, Any], *,
    max_workers: int, work_kind: str, live: bool, backend: str, product: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    id_mod = ird_attr("issue_dispatch", issue_dispatch)
    cap_fn = ird_attr("check_weekly_cap", check_weekly_cap)
    pre, pre_ok, acct = ctx["pre"], ctx["pre_ok"], ctx["acct"]
    cap = cap_fn(runs_dir, product=product, account_tag=acct.get("tag")) if pre_ok else {"capped": False}
    if not (pre_ok and cap.get("capped")):
        return None, None
    reroute_pre = id_mod.preflight(
        root, max_workers=ctx["eff_max_workers"], work_kind=work_kind, product=product)
    reroute_ok = reroute_pre.get("verdict") == "SPAWN_OK"
    reroute_acct = reroute_pre.get("account") or {}
    reroute: dict[str, Any] = {
        "attempted": True,
        "from": _account_public(acct),
        "preflight": _preflight_public(reroute_pre),
        "account": _account_public(reroute_acct),
    }
    blocked_cap = cap
    if reroute_ok and reroute_acct and not _same_account(acct, reroute_acct):
        reroute_cap = cap_fn(runs_dir, product=product, account_tag=reroute_acct.get("tag"))
        reroute["weekly_cap"] = reroute_cap
        if not reroute_cap.get("capped"):
            ctx["pre"], ctx["pre_ok"], ctx["acct"] = reroute_pre, True, reroute_acct
            return None, reroute
        reroute["reason"] = "rerouted account is also capped"
    else:
        reroute["reason"] = (
            "reroute did not find a different account"
            if reroute_ok else "reroute preflight refused")
    payload = {
        "schema": SCHEMA, "workspace": str(root), "live": live, "backend": backend,
        "max_workers": max_workers, "registry_refresh": ctx["reg"],
        "timed_out_workers": ctx["reaped"], "pruned_sidecars": ctx["pruned"],
        "preflight": _preflight_public(pre), "account": _account_public(acct),
        "weekly_cap": blocked_cap, "ok": False, "action": "weekly_capped",
        "verdict": "WEEKLY_CAPPED",
        "reason": (f"account '{acct.get('tag')}' is weekly-capped (resets "
                   f"{blocked_cap.get('reset_text') or '?'}); holding spawn until "
                   f"{blocked_cap.get('until')} — re-arm is automatic at the reset"),
    }
    if reroute:
        payload["account_cap_reroute"] = reroute
    return payload, reroute


def _try_seat_cool_reroute(
    root: Path, runs_dir: Path, ctx: dict[str, Any], *,
    max_workers: int, work_kind: str, live: bool, backend: str, product: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    id_mod = ird_attr("issue_dispatch", issue_dispatch)
    pre, pre_ok, acct = ctx["pre"], ctx["pre_ok"], ctx["acct"]
    seat_state = seat_spawn_failure_state(runs_dir, acct.get("tag"), backend) if pre_ok else {"cooled": False}
    if not (pre_ok and seat_state.get("cooled") and not seat_state.get("reprobe_due")):
        return None, None
    reroute_pre = id_mod.preflight(
        root, max_workers=ctx["eff_max_workers"], work_kind=work_kind, product=product)
    reroute_ok = reroute_pre.get("verdict") == "SPAWN_OK"
    reroute_acct = reroute_pre.get("account") or {}
    reroute: dict[str, Any] = {
        "attempted": True,
        "from": _account_public(acct),
        "seat_streak": seat_state,
        "preflight": _preflight_public(reroute_pre),
        "account": _account_public(reroute_acct),
    }
    blocked = False
    if reroute_ok and reroute_acct and not _same_account(acct, reroute_acct):
        reroute_state = seat_spawn_failure_state(runs_dir, reroute_acct.get("tag"), backend)
        reroute["reroute_seat_streak"] = reroute_state
        if reroute_state.get("cooled") and not reroute_state.get("reprobe_due"):
            reroute["reason"] = "rerouted seat is also cooled"
            blocked = True
        else:
            ctx["pre"], ctx["pre_ok"], ctx["acct"] = reroute_pre, True, reroute_acct
            return None, reroute
    else:
        reroute["reason"] = (
            "reroute did not find a different seat"
            if reroute_ok else "reroute preflight refused")
        blocked = True
    if not blocked:
        return None, reroute
    reprobe_min = ird_attr("SEAT_STREAK_REPROBE_MIN", SEAT_STREAK_REPROBE_MIN)
    payload = {
        "schema": SCHEMA, "workspace": str(root), "live": live, "backend": backend,
        "max_workers": max_workers, "registry_refresh": ctx["reg"],
        "timed_out_workers": ctx["reaped"], "pruned_sidecars": ctx["pruned"],
        "preflight": _preflight_public(pre), "account": _account_public(acct),
        "seat_streak": seat_state, "ok": False, "action": "seat_cooled",
        "verdict": "SEAT_COOLED",
        "reason": (f"seat '{seat_state.get('seat')}' has {seat_state.get('streak')} "
                   f"spawn-fails in a row (across targets) — cooling the seat "
                   f"instead of feeding it another issue; run `fak accounts "
                   f"status` and re-login or replace the seat; one re-probe "
                   f"spawn is admitted every {reprobe_min} min (#4591)"),
    }
    if reroute:
        payload["seat_cool_reroute"] = reroute
    return payload, reroute


def _admit_backend_gates(
    root: Path, runs_dir: Path, ctx: dict[str, Any], *,
    max_workers: int, work_kind: str, live: bool, backend: str, product: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    pre, acct = ctx["pre"], ctx["acct"]
    if ctx["gateway_skip_reason"] is not None:
        return {
            "schema": SCHEMA, "workspace": str(root), "live": live, "backend": backend,
            "max_workers": max_workers, "registry_refresh": ctx["reg"],
            "timed_out_workers": ctx["reaped"], "pruned_sidecars": ctx["pruned"],
            "preflight": _preflight_public(pre), "account": _account_public(acct),
            "gateway_health_gate": {"status": "backend_unhealthy",
                                    "reason": "gateway_down", "planned": 0},
            "ok": False, "action": "backend_health_skip", "verdict": "GATEWAY_DOWN",
            "reason": ctx["gateway_skip_reason"],
        }, None, None
    if ctx["health_skip_reason"] is not None:
        return {
            "schema": SCHEMA, "workspace": str(root), "live": live, "backend": backend,
            "max_workers": max_workers, "registry_refresh": ctx["reg"],
            "timed_out_workers": ctx["reaped"], "pruned_sidecars": ctx["pruned"],
            "preflight": _preflight_public(pre), "account": _account_public(acct),
            "backend_health_skip": {
                "backend": backend, "stub_rate": ctx["health_stub_rate"],
                "threshold": _HEALTH_SKIP_STUB_RATE, "planned": 0},
            "ok": False, "action": "backend_health_skip",
            "verdict": "BACKEND_HEALTH_SKIP", "reason": ctx["health_skip_reason"],
        }, None, None
    cap_refusal, account_cap_reroute = _try_weekly_cap_reroute(
        root, runs_dir, ctx, max_workers=max_workers, work_kind=work_kind,
        live=live, backend=backend, product=product)
    if cap_refusal is not None:
        return cap_refusal, account_cap_reroute, None
    seat_refusal, seat_cool_reroute = _try_seat_cool_reroute(
        root, runs_dir, ctx, max_workers=max_workers, work_kind=work_kind,
        live=live, backend=backend, product=product)
    if seat_refusal is not None:
        return seat_refusal, account_cap_reroute, seat_cool_reroute
    own_health = ctx["own_health"]
    pre, pre_ok, acct = ctx["pre"], ctx["pre_ok"], ctx["acct"]
    if pre_ok and own_health.get("state") == "dead" and not own_health.get("reprobe_due"):
        _lane = own_health.get("abandoned_lane") or "?"
        reprobe_default = ird_attr("_BACKEND_REPROBE_MIN", _BACKEND_REPROBE_MIN)
        if own_health.get("auth_gap"):
            _reason = (f"backend '{backend}' is UNAUTHENTICATED — every spawn dies instantly "
                       f"on a missing API key / no resolved login (see its evidence logs) "
                       f"since {own_health.get('since')}; planning 0 spawns. Set the backend's "
                       f"credentials (e.g. export OPENAI_API_KEY / `codex login`) or drop it "
                       f"from the backend rotation; its lane '{_lane}' is reallocated to a "
                       f"healthy backend, and a re-probe is admitted only every "
                       f"{own_health.get('reprobe_min')} min (not {reprobe_default}) "
                       f"because a missing key cannot self-heal")
        else:
            _reason = (f"backend '{backend}' is majority-stub "
                       f"({own_health.get('stub')}/"
                       f"{(own_health.get('stub') or 0) + (own_health.get('productive') or 0)}"
                       f" recent spawns banner-only/0-byte, stub_rate="
                       f"{own_health.get('stub_rate')}) since {own_health.get('since')}; "
                       f"planning 0 spawns — its lane '{_lane}' is reallocated "
                       f"to a healthy backend, and one re-probe worker is admitted every "
                       f"{reprobe_default} min to detect recovery")
        return {
            "schema": SCHEMA, "workspace": str(root), "live": live, "backend": backend,
            "max_workers": max_workers, "registry_refresh": ctx["reg"],
            "timed_out_workers": ctx["reaped"], "pruned_sidecars": ctx["pruned"],
            "preflight": _preflight_public(pre),
            "account": _account_public(acct),
            "backend_health": own_health, "ok": False, "action": "backend_unhealthy",
            "verdict": "BACKEND_UNHEALTHY", "reason": _reason,
        }, account_cap_reroute, seat_cool_reroute
    return None, account_cap_reroute, seat_cool_reroute


def _collect_lane_and_candidate_state(
    root: Path,
    runs_dir: Path,
    ctx: dict[str, Any],
    *,
    lane: str | None,
    exclude_lanes: set[str] | None,
    cooldown_min: int,
    contract_hold_ttl_h: int,
    issue_override: int | None,
) -> dict[str, Any]:
    effective_exclude = set(exclude_lanes or set())
    realloc = ctx["realloc"]
    if realloc["claimed_lanes"]:
        effective_exclude -= {lc for lc in realloc["claimed_lanes"] if lc}
    local_held_lanes = ird_attr("live_resolution_lanes", live_resolution_lanes)(runs_dir)
    live_leases = ird_attr("live_lane_lease_lanes", live_lane_lease_lanes)(root)
    lease_held_lanes = set(live_leases.get("lanes") or [])
    held_lanes = local_held_lanes | lease_held_lanes
    ilr_mod = ird_attr("issue_lane_router", issue_lane_router)
    try:
        _concurrent, low_yield_trees, _exclusive = ilr_mod.lane_taxonomy(root)
    except Exception:
        low_yield_trees = {}
    low_yield = ird_attr("low_yield_soft_excludes", low_yield_soft_excludes)(
        root, runs_dir, lane_trees=low_yield_trees)
    soft_exclude = set(low_yield.get("exclude") or set())
    if lane:
        soft_exclude.discard(lane)
    _soft = {"soft_exclude": soft_exclude} if soft_exclude else {}
    pick = ird_attr("lane_issue_numbers", lane_issue_numbers)(
        root, lane, exclude=effective_exclude | held_lanes, **_soft)
    low_yield_excluded_lanes = pick.get("low_yield_excluded") or []
    low_yield_relief = pick.get("low_yield_relief") or []
    _excluded_set = set(low_yield_excluded_lanes)
    low_yield_evidence = [r for r in (low_yield.get("lanes") or [])
                          if r.get("lane") in _excluded_set]
    chosen_lane = pick.get("lane")
    live_issues = ird_attr("live_resolution_issues", live_resolution_issues)(runs_dir)
    cooled = ird_attr("recently_attempted_issues", recently_attempted_issues)(
        runs_dir, cooldown_min=cooldown_min)
    held_no_commit = held_no_commit_issues(ctx["witnessed"])
    local_witnessed = ird_attr("locally_witnessed_issues", locally_witnessed_issues)(root)
    if (contract_hold_ttl_h > 0 or DEFAULT_MULTI_LANE_HOLD_TTL_H > 0
            or DEFAULT_COLLISION_HOLD_TTL_H > 0
            or DEFAULT_REBLOCK_STREAK_HOLD_TTL_H > 0):
        refreshed_ts = ird_attr("open_issue_updated_map", open_issue_updated_map)(root)
        for n, ts in contract_overlay_times(runs_dir).items():
            refreshed_ts[n] = max(refreshed_ts.get(n) or 0.0, ts)
    else:
        refreshed_ts = None
    contract_held_prior = ird_attr("contract_held_issues", contract_held_issues)(
        runs_dir, ttl_h=contract_hold_ttl_h, updated_ts=refreshed_ts)
    multi_lane_held_prior = ird_attr("multi_lane_held_issues", multi_lane_held_issues)(
        runs_dir, ttl_h=DEFAULT_MULTI_LANE_HOLD_TTL_H, updated_ts=refreshed_ts)
    collision_held_prior = ird_attr("collision_held_issues", collision_held_issues)(
        runs_dir, ttl_h=DEFAULT_COLLISION_HOLD_TTL_H, updated_ts=refreshed_ts)
    reblock_streak_held = reblock_streak_held_issues(
        runs_dir, streak_n=DEFAULT_REBLOCK_STREAK_HOLD_N,
        ttl_h=DEFAULT_REBLOCK_STREAK_HOLD_TTL_H, updated_ts=refreshed_ts)
    eligible_lanes = pick.get("eligible_by_lane")
    if not eligible_lanes and chosen_lane:
        eligible_lanes = [[chosen_lane, pick.get("numbers") or []]]
    candidates = _candidate_issue_numbers(eligible_lanes, pick.get("numbers") or [])
    audit_abstain_holds = ird_attr(
        "commit_audit_abstain_holds", commit_audit_abstain_holds)(root, candidates)
    audit_abstain_held = {int(r["issue"]) for r in audit_abstain_holds
                          if r.get("issue") is not None}
    open_witnessed_rows = ird_attr(
        "open_witnessed_dispositions", open_witnessed_dispositions)(root, candidates)
    open_witnessed_held = {int(r["issue"]) for r in open_witnessed_rows
                           if r.get("issue") is not None}
    dw_mod = ird_attr("dispatch_worker", dispatch_worker)
    node_caps_set = dw_mod.node_caps()
    caps_by_issue = {int(k): v for k, v in (pick.get("caps_by_issue") or {}).items()}
    capability_skipped, capability_skipped_issues = capability_skips(
        candidates, caps_by_issue, node_caps_set)
    skip = (live_issues | cooled | held_no_commit | contract_held_prior
            | multi_lane_held_prior | collision_held_prior
            | reblock_streak_held
            | local_witnessed | audit_abstain_held | open_witnessed_held
            | capability_skipped)
    scan_stream = contract_scan_stream(eligible_lanes, skip)
    if scan_stream:
        chosen_lane, target = scan_stream[0]
    else:
        target = pick_target_issue(pick.get("numbers") or [], skip)
    if issue_override is not None:
        target = int(issue_override)
    return {
        "held_lanes": held_lanes, "lease_held_lanes": lease_held_lanes,
        "live_leases": live_leases, "pick": pick,
        "low_yield_excluded_lanes": low_yield_excluded_lanes,
        "low_yield_evidence": low_yield_evidence, "low_yield_relief": low_yield_relief,
        "chosen_lane": chosen_lane, "live_issues": live_issues, "cooled": cooled,
        "held_no_commit": held_no_commit, "local_witnessed": local_witnessed,
        "refreshed_ts": refreshed_ts, "contract_held_prior": contract_held_prior,
        "multi_lane_held_prior": multi_lane_held_prior,
        "collision_held_prior": collision_held_prior,
        "reblock_streak_held": reblock_streak_held,
        "eligible_lanes": eligible_lanes, "audit_abstain_holds": audit_abstain_holds,
        "audit_abstain_held": audit_abstain_held,
        "open_witnessed_rows": open_witnessed_rows,
        "node_caps_set": node_caps_set, "caps_by_issue": caps_by_issue,
        "capability_skipped_issues": capability_skipped_issues,
        "scan_stream": scan_stream, "target": target,
    }


def _build_base_payload(
    root: Path,
    runs_dir: Path,
    ctx: dict[str, Any],
    state: dict[str, Any],
    *,
    live: bool,
    backend: str,
    max_workers: int,
    work_kind: str,
    force: bool,
    contract_scan: int,
    repair_batch: int,
    account_cap_reroute: dict[str, Any] | None,
    seat_cool_reroute: dict[str, Any] | None,
) -> dict[str, Any]:
    realloc = ctx["realloc"]
    acct = ctx["acct"]
    pick = state["pick"]
    open_witnessed_rows = state["open_witnessed_rows"]
    payload: dict[str, Any] = {
        "schema": SCHEMA, "workspace": str(root), "live": live, "backend": backend,
        "max_workers": max_workers, "work_kind": work_kind, "force": bool(force),
        "contract_scan": int(contract_scan), "repair_batch": int(repair_batch),
        "registry_refresh": ctx["reg"],
        "timed_out_workers": ctx["reaped"],
        **({"reallocation": {"effective_max_workers": ctx["eff_max_workers"], **realloc}}
           if realloc["bonus"] or realloc["claimed_lanes"] else {}),
        **({"leases_reaped": ctx["leases_reaped"]} if live else {}),
        **({"witnessed_slots": ctx["witnessed"]} if live else {}),
        **({"account_cap_reroute": account_cap_reroute} if account_cap_reroute else {}),
        **({"seat_cool_reroute": seat_cool_reroute} if seat_cool_reroute else {}),
        "preflight": _preflight_public(ctx["pre"]),
        **({"seat_adaptive": ctx["seat_sizing"]} if ctx["seat_sizing"] else {}),
        "account": _account_public(acct),
        "lane": state["chosen_lane"],
        "lane_issue_count": len(pick.get("numbers") or []),
        "cooled_recently": sorted(state["cooled"]),
        "target_issue": state["target"],
        **({"held_no_commit": sorted(state["held_no_commit"])} if state["held_no_commit"] else {}),
        **({"multi_lane_held_prior": len(state["multi_lane_held_prior"])}
           if state["multi_lane_held_prior"] else {}),
        **({"locally_witnessed": sorted(state["local_witnessed"])}
           if state["local_witnessed"] else {}),
        **({"commit_audit_abstain_held": state["audit_abstain_holds"]}
           if state["audit_abstain_holds"] else {}),
        **({"open_witnessed": open_witnessed_rows} if open_witnessed_rows else {}),
        **({"self_modify_held": pick.get("self_modify_held")} if pick.get("self_modify_held") else {}),
        **({"safe_lanes_busy": pick.get("safe_lanes_busy")} if pick.get("safe_lanes_busy") else {}),
        **({"capability_skipped_issues": state["capability_skipped_issues"]}
           if state["capability_skipped_issues"] else {}),
        **({"node_caps": sorted(state["node_caps_set"])} if state["node_caps_set"] else {}),
        **({"low_yield_excluded_lanes": sorted(state["low_yield_excluded_lanes"])}
           if state["low_yield_excluded_lanes"] else {}),
        **({"low_yield_lanes": state["low_yield_evidence"]} if state["low_yield_evidence"] else {}),
        **({"low_yield_relief": sorted(state["low_yield_relief"])}
           if state["low_yield_relief"] else {}),
        **({"lease_held_lanes": sorted(state["lease_held_lanes"])}
           if state["lease_held_lanes"] else {}),
        **({"lane_leases": state["live_leases"]} if state["live_leases"].get("fail_open") else {}),
        "already_live": sorted(state["live_issues"]),
        "held_lanes": sorted(state["held_lanes"]),
    }
    if live and open_witnessed_rows:
        payload["open_witnessed_close"] = ird_attr(
            "close_open_witnessed", close_open_witnessed)(
                root, runs_dir, open_witnessed_rows, live=True)
    return payload


def _handle_pre_pick_holds(
    root: Path,
    runs_dir: Path,
    payload: dict[str, Any],
    ctx: dict[str, Any],
    state: dict[str, Any],
    *,
    live: bool,
    backend: str,
    contract_hold_ttl_h: int,
    repair_batch: int,
    repair_cooldown_min: int,
    spawn_probe_s: float,
    product: str,
) -> dict[str, Any] | None:
    pre = ctx["pre"]
    if not ctx["pre_ok"]:
        hint = _preflight_refusal_hint(pre)
        if hint:
            payload["preflight_hint"] = hint
        payload.update({"ok": False, "action": "refused",
                        "verdict": pre.get("verdict") or "REFUSE",
                        "reason": f"preflight refused: {pre.get('reason')}"})
        return payload
    chosen_lane = state["chosen_lane"]
    guard_livelock = ird_attr("active_guard_livelock_hold", active_guard_livelock_hold)(
        root, runs_dir)
    if guard_livelock.get("active"):
        livelock_lanes = set(guard_livelock.get("lanes") or ())
        lane_unknown = bool(guard_livelock.get("lane_unknown", True))
        collides = lane_unknown or not chosen_lane or str(chosen_lane) in livelock_lanes
        guard_livelock = {**guard_livelock, "scoped_lane": chosen_lane, "collides": collides}
        payload["active_guard_livelock"] = guard_livelock
        if collides:
            reason = str(guard_livelock.get("reason") or "active guard livelock")
            payload["launch_gate"] = launch_gate_blocked(
                "ACTIVE_GUARD_LIVELOCK", reason,
                "wait-for-loop-worker-to-exit-or-fix-guard-result-livelock")
            payload.update({"ok": False, "action": "active_guard_livelock",
                            "verdict": "ACTIVE_GUARD_LIVELOCK", "reason": reason})
            return payload
    compact_runaway = ird_attr("active_compact_runaway_hold", active_compact_runaway_hold)(
        root, runs_dir)
    if compact_runaway.get("active"):
        runaway_lanes = set(compact_runaway.get("lanes") or ())
        lane_unknown = bool(compact_runaway.get("lane_unknown", True))
        collides = lane_unknown or not chosen_lane or str(chosen_lane) in runaway_lanes
        compact_runaway = {**compact_runaway, "scoped_lane": chosen_lane, "collides": collides}
        payload["active_compact_runaway"] = compact_runaway
        if collides:
            reason = str(compact_runaway.get("reason") or "active compact runaway")
            payload["launch_gate"] = launch_gate_blocked(
                "ACTIVE_COMPACT_RUNAWAY", reason,
                "wait-for-loop-worker-to-exit-or-fix-compact-control")
            payload.update({"ok": False, "action": "active_compact_runaway",
                            "verdict": "ACTIVE_COMPACT_RUNAWAY", "reason": reason})
            return payload
    pick = state["pick"]
    if not chosen_lane:
        if pick.get("self_modify_held"):
            held = sorted(pick.get("self_modify_held"))
            safe_busy = sorted(pick.get("safe_lanes_busy") or [])
            blockers = []
            if safe_busy:
                blockers.append({
                    "code": "SAFE_LANES_BUSY",
                    "reason": "safe non-self-source lanes already have live workers: "
                    + ", ".join(safe_busy),
                    "next_action": "wait-for-safe-lane-lease",
                })
            safe_lane = held[0] if held else "cmd"
            worktree_command = (
                f"python tools/worker_worktree.py --root . prepare "
                f"--lane {safe_lane} --key <issue>  # edit in .path, then: "
                f"python tools/worker_worktree.py --root . land --worktree <path> "
                f"--msg-file <msg> --path <files>; ... reap --worktree <path>")
            blockers.append({
                "code": "SELF_MODIFY_HOLD",
                "reason": "every remaining open issue lane maps to the shared source tree"
                if safe_busy else "every open issue lane maps to the shared source tree",
                "next_action": "route-non-self-source-lane-or-enable-worktree-isolated-resolver",
                "command": worktree_command,
            })
            if safe_busy:
                reason = (f"safe non-self-source lanes are busy ({safe_busy}) and "
                          f"every remaining open issue lane is self-source ({held}); "
                          f"waiting rather than risking build-poisoning the shared trunk "
                          f"(or land via the worktree-isolated path: {worktree_command})")
            else:
                reason = (f"every lane with open issues is self-source ({held}); "
                          f"refusing to risk build-poisoning the shared trunk — land it "
                          f"via the worktree-isolated path (#1334): {worktree_command}")
            payload.update({"ok": False, "action": "no_lane", "verdict": "SELF_MODIFY_HOLD",
                            "launch_gate": {"ready": False, "blockers": blockers},
                            "reason": reason})
        else:
            payload.update({"ok": False, "action": "no_lane", "verdict": "NO_LANE",
                            "reason": "no lane has open issues (router empty/error)"})
        return payload
    held_lanes = state["held_lanes"]
    if chosen_lane in held_lanes:
        payload.update({"ok": False, "action": "lane_busy", "verdict": "LANE_BUSY",
                        "reason": (f"lane '{chosen_lane}' already holds a live "
                                   f"resolution worker (held lanes "
                                   f"{sorted(held_lanes)}); refusing COLLISION_RISK "
                                   f"— the dispatcher holds the lane lease before "
                                   f"launching, it does not race a second worker "
                                   f"onto the same leaf tree")})
        return payload
    target = state["target"]
    node_caps_set = state["node_caps_set"]
    caps_by_issue = state["caps_by_issue"]
    if target is not None:
        missing_caps = sorted(set(caps_by_issue.get(int(target), [])) - node_caps_set)
        if missing_caps:
            payload.update({"ok": False, "action": "node_incapable",
                            "verdict": "NODE_INCAPABLE", "target_issue": int(target),
                            "required_caps": sorted(caps_by_issue.get(int(target), [])),
                            "node_caps": sorted(node_caps_set),
                            "reason": (f"issue #{int(target)} requires hardware "
                                       f"capabilities {missing_caps} this node does not "
                                       f"advertise (FLEET_NODE_CAPS={sorted(node_caps_set)}); "
                                       f"leaving it OPEN for a capable node rather than "
                                       f"running it here — set FLEET_NODE_CAPS to opt in")})
            return payload
    if target is None:
        held_candidate_numbers: set[int] = set()
        for entry in state["eligible_lanes"] or []:
            try:
                for n in entry[1] or []:
                    if int(n) in state["contract_held_prior"]:
                        held_candidate_numbers.add(int(n))
            except (TypeError, ValueError, IndexError, KeyError):
                continue
        contract_records_fn = ird_attr("contract_held_records", contract_held_records)
        held_rows = contract_records_fn(
            runs_dir, ttl_h=contract_hold_ttl_h, updated_ts=state["refreshed_ts"],
            only=held_candidate_numbers) if held_candidate_numbers else []
        if held_rows:
            payload["contract_held_prior"] = len(state["contract_held_prior"])
            repaired = _maybe_dispatch_contract_repair(
                root, runs_dir, payload, rows=held_rows, live=live,
                backend=backend, acct=ctx["acct"], repair_batch=repair_batch,
                repair_cooldown_min=repair_cooldown_min,
                spawn_probe_s=spawn_probe_s, product=product)
            if repaired is not None:
                return repaired
        lanes_considered = [e[0] for e in (state["eligible_lanes"] or []) if e and e[1]]
        where = (f"lane '{chosen_lane}'" if len(lanes_considered) <= 1
                 else f"all {len(lanes_considered)} eligible lanes")
        repair_note = (payload.get("contract_repair") or {}).get("skipped")
        repair_suffix = f" (contract-repair: {repair_note})" if repair_note else ""
        payload.update({"ok": False, "action": "no_issue", "verdict": "NO_ISSUE",
                        "reason": (f"every open issue on {where} is "
                                   f"either live ({sorted(state['live_issues'])}), locally "
                                   f"witnessed ({sorted(state['local_witnessed'])}), in "
                                   f"cooldown ({sorted(state['cooled'])}), commit-audit-abstain "
                                   f"held ({sorted(state['audit_abstain_held'])}), "
                                   f"multi-lane-held ({sorted(state['multi_lane_held_prior'])}), "
                                   f"or contract-held "
                                   f"within the last {contract_hold_ttl_h}h "
                                   f"({len(state['contract_held_prior'])} issue(s) awaiting "
                                   f"a contract backfill) — nothing fresh to "
                                   f"dispatch this tick{repair_suffix}")})
        return payload
    return None


def _scan_contract_and_collisions(
    root: Path,
    runs_dir: Path,
    state: dict[str, Any],
    *,
    force: bool,
    issue_override: int | None,
    contract_scan: int,
) -> dict[str, Any]:
    iwp_mod = ird_attr("issue_worker_prompt", issue_worker_prompt)
    review_fn = ird_attr("issue_contract_review", None)
    assert review_fn is not None
    same_wip_fn = ird_attr("same_issue_wip_collision", same_issue_wip_collision)
    scope_scan_fn = ird_attr("scan_multi_lane_scope", scan_multi_lane_scope)
    chosen_lane = state["chosen_lane"]
    target = state["target"]
    scan_stream = state["scan_stream"]
    rec = iwp_mod.build(target, chosen_lane, workspace=root)
    rec["prompt"] += _IMPL_FIRST_BUDGET_SUFFIX
    contract = review_fn(root, rec.get("issue_record"), target)
    dirty_paths_doc = ird_attr("dirty_repo_paths", dirty_repo_paths)(root)

    contract_skipped: list[dict[str, Any]] = []
    scan_pos = 0
    while (_contract_holds(contract) and not force and issue_override is None
           and not contract.get("unavailable")
           and len(contract_skipped) + 1 < max(1, contract_scan)):
        if scan_pos + 1 >= len(scan_stream):
            break
        contract_skipped.append({
            "issue": target, "lane": chosen_lane, "title": rec.get("title"),
            "score": int(contract.get("score") or 0),
            "reason": issue_contract_hold_reason(contract)[:200],
            "missing_fields": _contract_missing_fields(contract),
        })
        scan_pos += 1
        chosen_lane, target = scan_stream[scan_pos]
        rec = iwp_mod.build(target, chosen_lane, workspace=root)
        contract = review_fn(root, rec.get("issue_record"), target)

    def _advance_candidate() -> None:
        nonlocal chosen_lane, target, rec, contract, scan_pos
        scan_pos += 1
        chosen_lane, target = scan_stream[scan_pos]
        rec = iwp_mod.build(target, chosen_lane, workspace=root)
        contract = review_fn(root, rec.get("issue_record"), target)

    dirty_path_skipped: list[dict[str, Any]] = []
    same_issue_wip_skipped: list[dict[str, Any]] = []
    multi_lane_skipped: list[dict[str, Any]] = []
    scope_scan: dict[str, Any] | None = None
    dirty_scan: dict[str, Any] | None = None
    wip_scan: dict[str, Any] | None = None
    scan_limit = max(1, contract_scan)
    dirty_list = list(dirty_paths_doc.get("paths") or [])
    while issue_override is None and not _contract_holds(contract):
        total_skipped = (len(contract_skipped) + len(same_issue_wip_skipped)
                         + len(dirty_path_skipped) + len(multi_lane_skipped))
        wip_scan = same_wip_fn(runs_dir, target, dirty_list)
        dirty_scan = dirty_path_collision(_candidate_text(rec), dirty_list)
        if wip_scan.get("collides"):
            if scan_pos + 1 >= len(scan_stream) or total_skipped + 1 >= scan_limit:
                break
            same_issue_wip_skipped.append({
                "issue": target, "lane": chosen_lane, "title": rec.get("title"),
                "reason": same_issue_wip_reason(target, wip_scan)[:240],
                "dirty_paths": wip_scan.get("dirty_paths"),
                "evidence": wip_scan.get("evidence"),
            })
            _advance_candidate()
            scope_scan, dirty_scan = None, None
        elif dirty_scan.get("collides"):
            if scan_pos + 1 >= len(scan_stream) or total_skipped + 1 >= scan_limit:
                break
            dirty_path_skipped.append({
                "issue": target, "lane": chosen_lane, "title": rec.get("title"),
                "reason": dirty_path_collision_reason(target, dirty_scan)[:240],
                "dirty_paths": dirty_scan.get("dirty_paths"),
            })
            _advance_candidate()
            scope_scan = None
        elif not force:
            scope_scan = scope_scan_fn(root, _candidate_text(rec), chosen_lane)
            if not scope_scan.get("multi_lane"):
                break
            if scan_pos + 1 >= len(scan_stream) or total_skipped + 1 >= scan_limit:
                break
            multi_lane_skipped.append({
                "issue": target, "lane": chosen_lane, "title": rec.get("title"),
                "reason": multi_lane_scope_reason(target, scope_scan)[:240],
                "uncovered_lanes": scope_scan.get("uncovered_lanes"),
            })
            _advance_candidate()
            dirty_scan = None
        else:
            break
        while (_contract_holds(contract) and not contract.get("unavailable")
               and issue_override is None
               and scan_pos + 1 < len(scan_stream)
               and (len(contract_skipped) + len(same_issue_wip_skipped)
                    + len(dirty_path_skipped) + len(multi_lane_skipped) + 1 < scan_limit)):
            contract_skipped.append({
                "issue": target, "lane": chosen_lane, "title": rec.get("title"),
                "score": int(contract.get("score") or 0),
                "reason": issue_contract_hold_reason(contract)[:200],
                "missing_fields": _contract_missing_fields(contract),
            })
            _advance_candidate()
    return {
        "chosen_lane": chosen_lane, "target": target, "rec": rec,
        "contract": contract, "dirty_paths_doc": dirty_paths_doc,
        "contract_skipped": contract_skipped,
        "same_issue_wip_skipped": same_issue_wip_skipped,
        "dirty_path_skipped": dirty_path_skipped,
        "multi_lane_skipped": multi_lane_skipped,
        "scope_scan": scope_scan, "dirty_scan": dirty_scan, "wip_scan": wip_scan,
    }


def _admit_candidate_gates(
    root: Path,
    runs_dir: Path,
    payload: dict[str, Any],
    ctx: dict[str, Any],
    state: dict[str, Any],
    scan_res: dict[str, Any],
    *,
    live: bool,
    force: bool,
    force_reason: str,
    issue_override: int | None,
    backend: str,
    repair_batch: int,
    repair_cooldown_min: int,
    spawn_probe_s: float,
    product: str,
) -> dict[str, Any] | None:
    chosen_lane = scan_res["chosen_lane"]
    target = scan_res["target"]
    rec = scan_res["rec"]
    contract = scan_res["contract"]
    contract_skipped = scan_res["contract_skipped"]
    same_issue_wip_skipped = scan_res["same_issue_wip_skipped"]
    dirty_path_skipped = scan_res["dirty_path_skipped"]
    multi_lane_skipped = scan_res["multi_lane_skipped"]
    min_score = ird_attr("DEFAULT_ISSUE_CONTRACT_MIN_SCORE", DEFAULT_ISSUE_CONTRACT_MIN_SCORE)

    payload["target_issue"] = target
    payload["lane"] = chosen_lane
    payload["lane_issue_count"] = (state["pick"].get("by_lane_count") or {}).get(
        chosen_lane, payload.get("lane_issue_count"))
    payload["prompt_chars"] = rec.get("prompt_chars")
    payload["issue_title"] = rec.get("title")
    payload["target_priority"] = candidate_priority(
        ((rec.get("issue_record") or {}).get("labels")) or [])
    target_blocked_by = candidate_blocked_by(
        (rec.get("issue_record") or {}).get("body"))
    if target_blocked_by:
        payload["target_blocked_by"] = target_blocked_by
    if contract_skipped:
        payload["contract_skipped"] = contract_skipped
    if same_issue_wip_skipped:
        payload["same_issue_wip_skipped"] = same_issue_wip_skipped
    if dirty_path_skipped:
        payload["dirty_path_skipped"] = dirty_path_skipped
    if multi_lane_skipped:
        payload["multi_lane_skipped"] = multi_lane_skipped
    if state["contract_held_prior"]:
        payload["contract_held_prior"] = len(state["contract_held_prior"])
    if state["collision_held_prior"]:
        payload["collision_held_prior"] = len(state["collision_held_prior"])
    if state["reblock_streak_held"]:
        payload["reblock_streak_held"] = sorted(state["reblock_streak_held"])
    payload["issue_contract_gate"] = {
        "ok": bool(contract.get("ok")),
        "unavailable": bool(contract.get("unavailable")),
        "score": int(contract.get("score") or 0),
        "spine_priority": int(contract.get("spine_priority") or 0),
        "min_score": min_score,
        "reason": issue_contract_hold_reason(contract),
    }
    contract_hold = _contract_holds(contract)
    ledger_rows = list(contract_skipped)
    if (contract_hold and not force and not contract.get("unavailable")
            and issue_override is None):
        ledger_rows.append({"issue": target, "lane": chosen_lane,
                            "title": rec.get("title"),
                            "score": int(contract.get("score") or 0),
                            "reason": issue_contract_hold_reason(contract)[:200],
                            "missing_fields": _contract_missing_fields(contract)})
    ird_attr("record_contract_holds", record_contract_holds)(runs_dir, ledger_rows, live=live)
    ird_attr("record_multi_lane_holds", record_multi_lane_holds)(
        runs_dir, multi_lane_skipped, live=live)
    rec_collision_fn = ird_attr("record_collision_holds", record_collision_holds)
    rec_collision_fn(
        runs_dir,
        [{**r, "kind": "same_issue_wip"} for r in same_issue_wip_skipped]
        + [{**r, "kind": "dirty_path"} for r in dirty_path_skipped],
        live=live)
    if contract_hold and not force:
        if issue_override is None and not contract.get("unavailable"):
            repaired = _maybe_dispatch_contract_repair(
                root, runs_dir, payload, rows=ledger_rows, live=live,
                backend=backend, acct=ctx["acct"], repair_batch=repair_batch,
                repair_cooldown_min=repair_cooldown_min,
                spawn_probe_s=spawn_probe_s, product=product)
            if repaired is not None:
                return repaired
        reason = issue_contract_hold_reason(contract)
        if contract_skipped:
            lanes_scanned = {r.get("lane") for r in contract_skipped} | {chosen_lane}
            span = (f"issue(s) across {len(lanes_scanned)} lanes"
                    if len(lanes_scanned) > 1 else "lane issue(s)")
            reason = (f"scanned {len(contract_skipped) + 1} {span}, none "
                      f"contract-ready; last (#{target}): {reason}")
        repair_note = (payload.get("contract_repair") or {}).get("skipped")
        if repair_note:
            reason = f"{reason} (contract-repair: {repair_note})"
        payload.update({"ok": False, "action": "issue_contract_hold",
                        "verdict": "ISSUE_CONTRACT_HOLD", "reason": reason})
        return payload
    if contract_hold and force:
        gate_reason = issue_contract_hold_reason(contract)
        reason_text = str(force_reason or "").strip()
        if not reason_text:
            hold_reason = (
                f"--force on a failed issue-contract gate (#{target}: {gate_reason}) "
                f"requires a structured --force-reason so the override is recorded in "
                f"the run ledger; refusing to spawn")
            payload["launch_gate"] = launch_gate_blocked(
                "FORCE_REASON_REQUIRED", hold_reason,
                "re-run with --force --force-reason "
                "\"<why this spawn is worth bypassing the contract gate>\"")
            payload.update({"ok": False, "action": "force_reason_required",
                            "verdict": "FORCE_REASON_REQUIRED", "reason": hold_reason})
            return payload
        record_contract_forced_bypass(
            runs_dir,
            {"issue": target, "lane": chosen_lane,
             "score": int(contract.get("score") or 0),
             "reason": reason_text, "gate_reason": gate_reason,
             "title": rec.get("title")},
            live=live)
        payload["issue_contract_forced"] = {
            "bypassed": True, "gate_reason": gate_reason,
            "reason": reason_text,
            "bypass_count": contract_forced_bypass_count(runs_dir),
        }
    return _admit_collision_and_scope_gates(
        root, runs_dir, payload, scan_res, live=live, force=force,
        issue_override=issue_override, rec_collision_fn=rec_collision_fn)


def _admit_collision_and_scope_gates(
    root: Path,
    runs_dir: Path,
    payload: dict[str, Any],
    scan_res: dict[str, Any],
    *,
    live: bool,
    force: bool,
    issue_override: int | None,
    rec_collision_fn: Any,
) -> dict[str, Any] | None:
    chosen_lane = scan_res["chosen_lane"]
    target = scan_res["target"]
    rec = scan_res["rec"]
    dirty_list = list(scan_res["dirty_paths_doc"].get("paths") or [])
    wip_scan = scan_res["wip_scan"]
    if wip_scan is None:
        wip_scan = ird_attr("same_issue_wip_collision", same_issue_wip_collision)(
            runs_dir, target, dirty_list)
    if wip_scan.get("collides"):
        payload["same_issue_wip"] = wip_scan
        reason = same_issue_wip_reason(target, wip_scan)
        payload["launch_gate"] = launch_gate_blocked(
            "SAME_ISSUE_WIP", reason,
            "continue-or-commit-same-issue-wip-or-pick-disjoint-issue")
        payload.update({
            "ok": False, "action": "same_issue_wip", "verdict": "SAME_ISSUE_WIP",
            "refusal_class": REFUSAL_CLASS_WORKTREE_COTENANCY, "reason": reason})
        if issue_override is None:
            rec_collision_fn(runs_dir, [{
                "kind": "same_issue_wip", "issue": target, "lane": chosen_lane,
                "title": rec.get("title"), "reason": reason[:240],
                "dirty_paths": wip_scan.get("dirty_paths"),
            }], live=live)
        return payload
    dirty_scan = scan_res["dirty_scan"]
    if dirty_scan is None:
        dirty_scan = dirty_path_collision(_candidate_text(rec), dirty_list)
    if dirty_scan.get("collides"):
        payload["dirty_path_collision"] = dirty_scan
        reason = dirty_path_collision_reason(target, dirty_scan)
        payload["launch_gate"] = launch_gate_blocked(
            "DIRTY_PATH_COLLISION", reason,
            "wait-for-or-commit-dirty-paths-or-pick-disjoint-issue")
        payload.update({
            "ok": False, "action": "dirty_path_collision",
            "verdict": "DIRTY_PATH_COLLISION",
            "refusal_class": REFUSAL_CLASS_WORKTREE_COTENANCY, "reason": reason})
        if issue_override is None:
            rec_collision_fn(runs_dir, [{
                "kind": "dirty_path", "issue": target, "lane": chosen_lane,
                "title": rec.get("title"), "reason": reason[:240],
                "dirty_paths": dirty_scan.get("dirty_paths"),
            }], live=live)
        return payload
    scope_scan = scan_res["scope_scan"]
    if scope_scan is None:
        scope_scan = ird_attr("scan_multi_lane_scope", scan_multi_lane_scope)(
            root, _candidate_text(rec), chosen_lane)
    if scope_scan.get("multi_lane") and not force:
        hold_row = {
            "issue": target, "lane": chosen_lane, "title": rec.get("title"),
            "reason": multi_lane_scope_reason(target, scope_scan),
            "uncovered_lanes": scope_scan.get("uncovered_lanes"),
        }
        ird_attr("record_multi_lane_holds", record_multi_lane_holds)(
            runs_dir, [hold_row], live=live)
        payload["multi_lane_scope"] = scope_scan
        payload["multi_lane_hold"] = hold_row
        payload.update({
            "ok": False, "action": "multi_lane_scope",
            "verdict": "MULTI_LANE_SCOPE",
            "reason": multi_lane_scope_reason(target, scope_scan),
        })
        return payload
    if scope_scan.get("multi_lane") and force:
        payload["multi_lane_scope_forced"] = {
            "bypassed": True,
            "uncovered_lanes": scope_scan.get("uncovered_lanes"),
            "uncovered": scope_scan.get("uncovered"),
        }
    return None


def _prepare_command_and_spawn(
    root: Path,
    runs_dir: Path,
    payload: dict[str, Any],
    ctx: dict[str, Any],
    scan_res: dict[str, Any],
    *,
    live: bool,
    backend: str,
    worker_timeout_s: int | None,
    spawn_probe_s: float,
    product: str,
    lease_runner: Any | None,
) -> dict[str, Any]:
    acct = ctx["acct"]
    chosen_lane = scan_res["chosen_lane"]
    target = scan_res["target"]
    rec = scan_res["rec"]
    dw_mod = ird_attr("dispatch_worker", dispatch_worker)
    id_mod = ird_attr("issue_dispatch", issue_dispatch)
    model, effort = worker_model_effort(backend, acct)
    preview_prompt = f"<resolve #{target} prompt, {rec.get('prompt_chars')} chars>"
    preview_env = dw_mod.child_env(chosen_lane, backend, root)
    preview_command, preview_guarded = dw_mod.guarded_launch_command(
        build_worker_command(backend, preview_prompt, model, effort),
        chosen_lane, backend, root, preview_env)
    payload["command"] = preview_command
    payload["guarded"] = preview_guarded
    payload["launch_gate"] = launch_gate_for_guard(preview_guarded, backend)
    if not live:
        payload.update({"ok": True, "action": "would_spawn", "verdict": "WOULD_SPAWN",
                        "reason": (f"safe to spawn 1 {backend} issue-resolution worker on "
                                   f"#{target} (lane '{chosen_lane}') under account "
                                   f"'{acct.get('tag')}' (t{acct.get('tier')}, "
                                   f"{acct.get('model') or backend})")})
        return payload
    ttl_s = int((worker_timeout_s or DEFAULT_WORKER_TIMEOUT_S) + LEASE_TTL_MARGIN_S)
    acq_fn = ird_attr("acquire_lane_lease", acquire_lane_lease)
    rel_fn = ird_attr("release_lane_lease", release_lane_lease)
    lt_fn = ird_attr("lane_tree", lane_tree)
    lease = acq_fn(root, chosen_lane, tree=lt_fn(root, chosen_lane),
                   ttl_s=ttl_s, runner=lease_runner)
    payload["lease"] = lease
    if lease.get("refused"):
        payload.update({"ok": False, "action": "lane_leased", "verdict": "LANE_LEASE_HELD",
                        "refusal_class": REFUSAL_CLASS_LANE_LEASE,
                        "reason": (f"lane '{chosen_lane}' lease is held by a live peer "
                                   f"(fence {lease.get('reason') or lease.get('fence_verdict') or '?'}); "
                                   f"refusing COLLISION_RISK — the fenced "
                                   f"refs/fak/locks/{lease.get('id')} lease serializes "
                                   f"this lane across machines, not just this host")})
        return payload
    if backend == "claude":
        env = id_mod.worker_env(acct.get("dir"), chosen_lane, root)
    elif backend == "codex":
        env = codex_worker_env(acct.get("dir"), chosen_lane, root)
    else:
        env = opencode_worker_env(acct.get("dir"), chosen_lane, root, runs_dir)
    env["FLEET_RESOLVE_ISSUE"] = str(target)
    command, guarded = dw_mod.guarded_launch_command(
        build_worker_command(backend, rec["prompt"], model, effort),
        chosen_lane, backend, root, env)
    if guarded:
        dw_mod.guard_env_augment(env)
    payload["command"] = command
    payload["guarded"] = guarded
    payload["launch_gate"] = launch_gate_for_guard(guarded, backend)
    _argv_refusal = gate_spawn_on_guard_argv(
        command, runs_dir, record=live) if guarded else None
    if _argv_refusal:
        if lease.get("acquired"):
            payload["lease_release"] = rel_fn(root, lease, runner=lease_runner)
        payload.update({
            "ok": False, "action": "guard_argv_unsupported",
            "verdict": _argv_refusal["verdict"],
            "guard_probe": {k: _argv_refusal[k]
                            for k in ("build_id", "missing_flags", "probe")},
            "reason": _argv_refusal["reason"],
        })
        return payload
    rc_head, head_out = ird_attr("_git_capture", _git_capture)(root, ["rev-parse", "HEAD"])
    base_sha = head_out.strip() if rc_head == 0 and head_out.strip() else None
    spawn_fn = ird_attr("spawn_issue_worker", _spawn_issue_worker_impl)
    spawned = spawn_fn(command, env, root, runs_dir, target, chosen_lane, backend,
                       account=acct, lease=lease, base_sha=base_sha,
                       spawn_probe_s=spawn_probe_s,
                       prompt_payload=rec["prompt"] if backend in ("claude", "opencode", "codex") else None)
    early = spawned.get("early_exit") or {}
    if early.get("checked") and not early.get("alive"):
        if lease.get("acquired"):
            release = rel_fn(root, lease, runner=lease_runner)
            payload["lease_release"] = release
            if not release.get("released"):
                payload["lease_held_until_ttl"] = {"id": lease.get("id"),
                                                   "holder": lease.get("holder"),
                                                   "release": release}
        cap_hit = _cap_hit_from_text(
            str(early.get("tail") or ""),
            evidence_log=Path(str(spawned.get("log") or "")).name)
        if cap_hit:
            import time
            payload["quota_cap"] = _write_cap_hold(
                runs_dir, product=product, account_tag=acct.get("tag"),
                hit=cap_hit, now_ts=time.time(), fallback_min=60,
                source="early_exit")
        payload.update({"ok": False, "action": "spawn_failed",
                        "verdict": "SPAWN_FAILED",
                        "spawned": spawned,
                        "cause": classify_spawn_failed_cause(early),
                        "spawn_failed_streak": bump_spawn_failure_streak(
                            runs_dir, target, backend),
                        "seat_streak": bump_spawn_failure_streak_seat(
                            runs_dir, acct.get("tag"), backend),
                        "reason": (f"{backend} worker pid {spawned['pid']} for "
                                   f"#{target} exited within {early.get('wait_s')}s "
                                   f"with code {early.get('returncode')}"
                                   + (" and produced an empty log" if early.get("silent") else ""))})
        return payload
    clear_spawn_failure_streak(runs_dir, target, backend)
    clear_spawn_failure_streak_seat(runs_dir, acct.get("tag"), backend)
    payload.update({"ok": True, "action": "spawned", "verdict": "SPAWNED",
                    "spawned": spawned,
                    "reason": (f"spawned {backend} issue-resolution worker pid "
                               f"{spawned['pid']} on #{target} (lane '{chosen_lane}') "
                               f"under '{acct.get('tag')}' ({acct.get('model') or backend})")})
    return payload


def evaluate(root: Path, *, max_workers: int, work_kind: str, lane: str | None,
             live: bool, refresh: bool = True, cooldown_min: int = 120,
             backend: str = "claude",
             exclude_lanes: set[str] | None = None,
             worker_timeout_s: int | None = DEFAULT_WORKER_TIMEOUT_S,
             spawn_probe_s: float = DEFAULT_SPAWN_PROBE_S,
             realloc_ceiling: int = DEFAULT_REALLOC_CEILING,
             seat_adaptive: bool = DEFAULT_SEAT_ADAPTIVE,
             seat_ceiling: int = DEFAULT_SEAT_CEILING,
             seat_ramp_delta: int = DEFAULT_SEAT_RAMP_DELTA,
             record_loop: bool = False,
             loop_ledger: Path | None = None,
             issue_override: int | None = None,
             force: bool = False,
             force_reason: str = "",
             contract_scan: int = DEFAULT_CONTRACT_SCAN,
             contract_hold_ttl_h: int = DEFAULT_CONTRACT_HOLD_TTL_H,
             repair_batch: int = DEFAULT_REPAIR_BATCH,
             repair_cooldown_min: int = DEFAULT_REPAIR_COOLDOWN_MIN,
             lease_runner: Any | None = None) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    product = _BACKEND_PRODUCT[backend]
    runs_dir = root / ird_attr("RUNS_DIRNAME", RUNS_DIRNAME)
    finish_kw = {"live": live, "record_loop": record_loop, "loop_ledger": loop_ledger}

    ctx = _run_pre_tick_housekeeping(
        root, runs_dir, backend=backend, product=product, work_kind=work_kind,
        max_workers=max_workers, lane=lane, live=live, refresh=refresh,
        worker_timeout_s=worker_timeout_s, realloc_ceiling=realloc_ceiling,
        seat_adaptive=seat_adaptive, seat_ceiling=seat_ceiling,
        seat_ramp_delta=seat_ramp_delta, lease_runner=lease_runner)
    backend_refusal, account_cap_reroute, seat_cool_reroute = _admit_backend_gates(
        root, runs_dir, ctx, max_workers=max_workers, work_kind=work_kind,
        live=live, backend=backend, product=product)
    if backend_refusal is not None:
        return _finish_tick(root, runs_dir, backend_refusal, **finish_kw)

    state = _collect_lane_and_candidate_state(
        root, runs_dir, ctx, lane=lane, exclude_lanes=exclude_lanes,
        cooldown_min=cooldown_min, contract_hold_ttl_h=contract_hold_ttl_h,
        issue_override=issue_override)
    payload = _build_base_payload(
        root, runs_dir, ctx, state, live=live, backend=backend,
        max_workers=max_workers, work_kind=work_kind, force=force,
        contract_scan=contract_scan, repair_batch=repair_batch,
        account_cap_reroute=account_cap_reroute, seat_cool_reroute=seat_cool_reroute)

    early_hold = _handle_pre_pick_holds(
        root, runs_dir, payload, ctx, state, live=live, backend=backend,
        contract_hold_ttl_h=contract_hold_ttl_h, repair_batch=repair_batch,
        repair_cooldown_min=repair_cooldown_min, spawn_probe_s=spawn_probe_s,
        product=product)
    if early_hold is not None:
        return _finish_tick(root, runs_dir, early_hold, **finish_kw)

    scan_res = _scan_contract_and_collisions(
        root, runs_dir, state, force=force, issue_override=issue_override,
        contract_scan=contract_scan)
    candidate_hold = _admit_candidate_gates(
        root, runs_dir, payload, ctx, state, scan_res, live=live, force=force,
        force_reason=force_reason, issue_override=issue_override, backend=backend,
        repair_batch=repair_batch, repair_cooldown_min=repair_cooldown_min,
        spawn_probe_s=spawn_probe_s, product=product)
    if candidate_hold is not None:
        return _finish_tick(root, runs_dir, candidate_hold, **finish_kw)

    out_payload = _prepare_command_and_spawn(
        root, runs_dir, payload, ctx, scan_res, live=live, backend=backend,
        worker_timeout_s=worker_timeout_s, spawn_probe_s=spawn_probe_s,
        product=product, lease_runner=lease_runner)
    return _finish_tick(root, runs_dir, out_payload, **finish_kw)


def resolve_spawn_burst(flag: int | None = None,
                        env: Any | None = None) -> dict[str, Any]:
    """Resolve one tick's spawn-burst limit — ``flag > env > default``, clamped."""
    env = ird_attr("os", os).environ if env is None else env
    source = "default"
    requested: int = DEFAULT_SPAWN_BURST
    env_invalid: str | None = None
    raw_env = env.get(SPAWN_BURST_ENV)
    if raw_env not in (None, ""):
        try:
            requested = int(str(raw_env).strip())
            source = f"env:{SPAWN_BURST_ENV}"
        except (TypeError, ValueError):
            env_invalid = str(raw_env)
    if flag is not None:
        try:
            requested = int(flag)
            source = "flag:--spawn-burst"
        except (TypeError, ValueError):
            pass
    limit = max(1, min(int(requested), SPAWN_BURST_HARD_CEILING))
    out: dict[str, Any] = {
        "limit": limit, "requested": int(requested), "source": source,
        "hard_ceiling": SPAWN_BURST_HARD_CEILING,
        "clamped": limit != int(requested),
    }
    if env_invalid is not None:
        out["env_invalid"] = env_invalid
    return out


def resolve_spawn_burst_stagger(flag: float | None = None,
                                env: Any | None = None) -> float:
    """Seconds to wait between spawns inside one burst — ``flag > env > default``."""
    env = ird_attr("os", os).environ if env is None else env
    value = DEFAULT_SPAWN_BURST_STAGGER_S
    raw_env = env.get(SPAWN_BURST_STAGGER_ENV)
    if raw_env not in (None, ""):
        try:
            value = float(str(raw_env).strip())
        except (TypeError, ValueError):
            value = DEFAULT_SPAWN_BURST_STAGGER_S
    if flag is not None:
        try:
            value = float(flag)
        except (TypeError, ValueError):
            pass
    if value != value or value < 0:
        value = 0.0
    return float(min(value, SPAWN_BURST_STAGGER_MAX_S))


def evaluate_burst(root: Path, *, spawn_burst: int = DEFAULT_SPAWN_BURST,
                   burst_stagger_s: float = DEFAULT_SPAWN_BURST_STAGGER_S,
                   sleeper: Any | None = None,
                   **kw: Any) -> dict[str, Any]:
    """One tick that may spawn up to ``spawn_burst`` workers, filling live headroom."""
    eval_fn = ird_attr("evaluate", evaluate)
    runs_name = ird_attr("RUNS_DIRNAME", RUNS_DIRNAME)
    limit = max(1, min(int(spawn_burst), SPAWN_BURST_HARD_CEILING))
    live = bool(kw.get("live"))
    stagger = max(0.0, float(burst_stagger_s))
    if limit <= 1:
        return eval_fn(root, **kw)
    if not live:
        payload = eval_fn(root, **kw)
        payload["burst"] = {
            "limit": limit, "attempts": 1, "spawned": 0, "stagger_s": stagger,
            "skipped": ("dry_run — a dry-run sub-tick spawns nothing, so headroom "
                        "never moves and every further sub-tick would re-pick the "
                        "same issue; re-run with --live to fan out"),
        }
        return payload

    payloads: list[dict[str, Any]] = []
    spawned = 0
    stopped_on = "burst_limit"
    sub_kw = dict(kw)
    for n in range(limit):
        if n:
            sub_kw["refresh"] = False
            _burst_stagger_sleep(stagger, sleeper)
        payload = eval_fn(root, **sub_kw)
        payloads.append(payload)
        if str(payload.get("action") or "") != "spawned":
            stopped_on = str(payload.get("verdict") or payload.get("action") or "unknown")
            break
        spawned += 1

    out = dict(payloads[0])
    out["burst"] = {
        "limit": limit,
        "attempts": len(payloads),
        "spawned": spawned,
        "stagger_s": stagger,
        "stopped_on": stopped_on,
        "sub_ticks": [{
            "n": i + 1,
            "action": p.get("action"),
            "verdict": p.get("verdict"),
            "issue": p.get("target_issue"),
            "lane": p.get("lane"),
            "pid": (p.get("spawned") or {}).get("pid"),
        } for i, p in enumerate(payloads)],
    }
    if spawned > 1:
        extra = [f"#{r['issue']}" for r in out["burst"]["sub_ticks"][1:]
                 if r.get("action") == "spawned" and r.get("issue")]
        out["reason"] = (f"{out.get('reason')} (+{spawned - 1} more this tick: "
                         f"{' '.join(extra)}; burst {spawned}/{limit})")
    if len(payloads) > 1:
        _record(root / runs_name, out)
    return out


def _burst_stagger_sleep(seconds: float, sleeper: Any | None = None) -> float:
    """Pace one inter-spawn gap. Injectable so tests never really sleep."""
    if seconds <= 0:
        return 0.0
    if sleeper is None:
        import time
        time.sleep(seconds)
    else:
        sleeper(seconds)
    return seconds


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
    sa = p.get("seat_adaptive") or {}
    if sa.get("signal_available"):
        lines.insert(2, f"  seats     : adaptive cap {sa.get('effective_target')} "
                        f"(binding {sa.get('binding')}; live {sa.get('live')} + "
                        f"free {sa.get('seat_free')}, ceiling {sa.get('hard_ceiling')}, "
                        f"ramp +{sa.get('ramp_delta')}/tick)")
    hint = p.get("preflight_hint") or {}
    if hint.get("message"):
        lines.append(f"  hint      : {hint.get('message')}")
    if p.get("spawned"):
        s = p["spawned"]
        lines.append(f"  spawned pid={s.get('pid')} issue=#{s.get('issue')} log={s.get('log')}")
    burst = p.get("burst") or {}
    if burst.get("limit"):
        if burst.get("skipped"):
            lines.append(f"  burst     : limit {burst.get('limit')} — {burst['skipped']}")
        else:
            rows = " ".join(
                f"#{r.get('issue')}:{r.get('verdict')}"
                for r in (burst.get("sub_ticks") or []))
            lines.append(f"  burst     : {burst.get('spawned')}/{burst.get('limit')} "
                         f"spawned in {burst.get('attempts')} sub-tick(s), stopped on "
                         f"{burst.get('stopped_on')} "
                         f"(stagger {burst.get('stagger_s')}s) — {rows}")
    cs = p.get("contract_skipped") or []
    if cs:
        lanes = sorted({str(r.get("lane")) for r in cs if r.get("lane")})
        lines.append(f"  scan      : {len(cs)} contract-held head(s) skipped"
                     + (f" across lanes [{','.join(lanes)}]" if len(lanes) > 1 else ""))
    ws = p.get("same_issue_wip_skipped") or []
    if ws:
        lanes = sorted({str(r.get("lane")) for r in ws if r.get("lane")})
        lines.append(f"  scan      : {len(ws)} same-issue WIP head(s) skipped"
                     + (f" across lanes [{','.join(lanes)}]" if len(lanes) > 1 else ""))
    ds = p.get("dirty_path_skipped") or []
    if ds:
        lanes = sorted({str(r.get("lane")) for r in ds if r.get("lane")})
        lines.append(f"  scan      : {len(ds)} dirty-path head(s) skipped"
                     + (f" across lanes [{','.join(lanes)}]" if len(lanes) > 1 else ""))
    ms = p.get("multi_lane_skipped") or []
    if ms:
        lanes = sorted({str(r.get("lane")) for r in ms if r.get("lane")})
        lines.append(f"  scan      : {len(ms)} multi-lane head(s) skipped"
                     + (f" across lanes [{','.join(lanes)}]" if len(lanes) > 1 else ""))
    icf = p.get("issue_contract_forced") or {}
    if icf.get("bypassed"):
        lines.append(f"  forced    : contract gate bypassed — {icf.get('reason')} "
                     f"(gate: {icf.get('gate_reason')}; total bypasses "
                     f"{icf.get('bypass_count')})")
    cr = p.get("contract_repair") or {}
    if cr.get("batch"):
        lines.append(f"  repair    : issue(s) {cr['batch']} -> {p.get('verdict')}")
    elif cr.get("skipped"):
        lines.append(f"  repair    : skipped — {cr['skipped']}")
    elif cr.get("in_flight"):
        lines.append(f"  repair    : in flight (pid {cr['in_flight'][0].get('pid')})")
    rl = p.get("reallocation") or {}
    if rl.get("bonus") or rl.get("claimed_lanes"):
        frm = ",".join(x for x in (rl.get("from") or []) if x)
        claimed = ",".join(x for x in (rl.get("claimed_lanes") or []) if x)
        lines.append(f"  realloc   : +{rl.get('bonus')} workers, lane(s) [{claimed}] "
                     f"claimed from dead [{frm}] -> eff cap {rl.get('effective_max_workers')}")
    timed_out = p.get("timed_out_workers") or {}
    reaped = timed_out.get("reaped") or []
    would_reap = timed_out.get("would_reap") or []
    if reaped:
        lines.append(f"  reaped timed-out workers: {len(reaped)}")
    elif would_reap:
        lines.append(f"  would reap timed-out workers: {len(would_reap)}")
    ws = p.get("witnessed_slots") or {}
    if ws.get("audited"):
        lines.append(f"  witnessed slots: {len(ws.get('witnessed') or [])} witnessed, "
                     f"{len(ws.get('unwitnessed') or [])} CLAIM_UNWITNESSED, "
                     f"{len(ws.get('no_commit') or [])} no-commit")

    def _gate_suffix(gate: dict[str, Any]) -> str:
        blockers = gate.get("blockers") or []
        bits = []
        for b in blockers[:3]:
            bits.append(f"{b.get('code')}: {b.get('next_action')}")
        return "; ".join(bits) if bits else "launch gate is not ready"

    if not p.get("live") and p.get("action") == "would_spawn":
        gate = p.get("launch_gate") or {}
        if gate.get("ready"):
            lines.append("  DRY-RUN — re-run with --live to spawn the issue worker")
        else:
            lines.append(f"  DRY-RUN — NOT launch-ready ({_gate_suffix(gate)})")
    elif not p.get("live"):
        gate = p.get("launch_gate") or {}
        if gate.get("ready") is False:
            lines.append(f"  launch gate: BLOCKED ({_gate_suffix(gate)})")
    return "\n".join(lines)
