#!/usr/bin/env python3
"""Terminal card, Markdown, and Slack rendering/posting for dispatch_status.

Separated from ``tools/dispatch_status.py`` so operator presentation (terminal card,
Markdown report, Slack roll-up buckets and post hook) stays cleanly isolated from
payload collection and verdict adjudication.
"""
from __future__ import annotations

from typing import Any

from dispatch_status_common import (
    _STUB_LOG_MAX_BYTES,
    _age_text,
    _capacity_reconcile,
    _current_preflight_launch_blocker,
    _dispatch_limiter_terms,
    _int,
    _lab_link_label,
    _or_unknown,
    _resolver_preflight_summary,
    _slots_headroom_note,
    _workers_live_clause,
)
from dispatch_status_leases import (
    _LANE_LEASE_NEXT_ACTION,
    _active_worker_summary,
    _lane_lease_bits,
    lane_lease_verdict_clean,
)
from dispatch_status_telemetry import (
    _SHIPS_PER_WORKER_LOOKBACK_MIN,
    _auth_failed_seat_action,
    _double_booked_seat_action,
    _guard_empty_cause_bit,
    _seat_inventory_summary_line,
)
from lane_yield import _LOW_YIELD_LOOKBACK_MIN


def _lease_block_text(row: dict[str, Any]) -> str:
    blocking = row.get("blocks_candidate")
    if blocking is None:
        return "candidate unknown"
    if not blocking:
        return "no candidate"
    nums = [f"#{c.get('issue')}" for c in (row.get("blocking_candidates") or [])
            if c.get("issue") is not None]
    return "blocks " + (",".join(nums[:4]) if nums else "candidate")


def _lease_liveness_text(row: dict[str, Any]) -> str:
    live = str(row.get("liveness") or "").strip()
    if not live:
        return "unknown"
    if row.get("reclaimable"):
        return f"{live} reclaimable"
    return live


def _lease_summary_bits(leases: dict[str, Any], *, limit: int = 3) -> list[str]:
    rows = leases.get("active") or []
    bits: list[str] = []
    for row in rows[:limit]:
        bits.append(
            f"{row.get('id')} lane={row.get('lane') or '-'} "
            f"age={_age_text(row.get('age_min'))} "
            f"live={_lease_liveness_text(row)} {_lease_block_text(row)}")
    if len(rows) > limit:
        bits.append(f"+{len(rows) - limit} more")
    return bits


def _resolve_tick_state(row: dict[str, Any]) -> str:
    gate = row.get("launch_gate") or {}
    if not row.get("fresh"):
        return "stale"
    if gate.get("ready") is True:
        return "launch-ready"
    if row.get("action") == "repair_in_flight":
        return "repair-in-flight"
    if row.get("action") == "repair_spawned":
        return "repair-spawned"
    if gate.get("ready") is False:
        blockers = gate.get("blockers") or []
        code = blockers[0].get("code") if blockers else row.get("verdict")
        return f"held {code}"
    verdict = str(row.get("verdict") or "UNKNOWN")
    if verdict in ("WOULD_SPAWN", "SPAWNED"):
        return verdict.lower()
    return "held " + verdict


def _worker_lease_bucket_bits(rows: list[dict[str, Any]], *, key: str,
                              limit: int = 3) -> list[str]:
    bits: list[str] = []
    for row in rows[:limit]:
        obj = row.get(key) or {}
        bits.append(str(obj.get("worker") or obj.get("id") or obj.get("lease_id") or "?"))
    if len(rows) > limit:
        bits.append(f"+{len(rows) - limit} more")
    return bits


def _card_header_lines(p: dict[str, Any]) -> list[str]:
    """The card's top block: workers, limiter, switcher, watchdog, supervisor."""
    d = p.get("dispatcher") or {}
    s = p.get("supervisor") or {}
    a = d.get("account") or {}
    wd = d.get("watchdog") or {}
    return [
        f"╔═ DISPATCHER: {p.get('verdict')} ({'ok' if p.get('ok') else 'ACTION'})",
        f"║ workers   : {_workers_live_clause(d)}  "
        f"host={'clean' if d.get('host_safe') else 'FLAGGED'}",
        f"║ limiter   : {(d.get('limiter') or {}).get('primary') or '-'} "
        f"({_dispatch_limiter_terms(d.get('limiter') or {})})",
        f"║ switcher  : account={a.get('tag') or '-'} (t{a.get('tier')}) "
        f"avail={a.get('available')}  preflight={d.get('preflight_verdict')}",
        "║ always-on : watchdog "
        + ("installed (" + str(wd.get('status') or 'scheduled') + ")"
           if wd.get("installed") else
           ("NOT installed" if wd.get("installed") is False else "unknown")),
        f"║ supervisor: {s.get('verdict')} alive={s.get('alive')}/{s.get('target')}  "
        f"plans={(s.get('plans') or {}).get('total_plans')} "
        f"units={(s.get('plans') or {}).get('total_units')}",
    ]


def _card_capacity_lines(p: dict[str, Any]) -> list[str]:
    """Seat inventory and the lab-readiness gate."""
    lines: list[str] = []
    seat_line = _seat_inventory_summary_line(p.get("seat_inventory") or {})
    if seat_line:
        lines.append("║ seats     : " + seat_line.removeprefix("seat inventory: "))
    lab = p.get("lab_readiness") or {}
    if lab.get("schema"):
        gate = "admit" if lab.get("admit_dispatch") else "hold"
        lines.append(
            f"║ lab       : {_lab_link_label(lab)} ({gate}; next={lab.get('next_action') or '-'})")
        if not lab.get("admit_dispatch"):
            cmd = (lab.get("commands") or {}).get("mark_clear")
            if cmd:
                lines.append(f"║ lab cmd   : {cmd}")
    return lines


def _card_planner_lines(p: dict[str, Any]) -> list[str]:
    """The resolver tick: plan, launch command, seat cap/pick, spawn-fail streak."""
    lines: list[str] = []
    d = p.get("dispatcher") or {}
    preflight_blocker = _current_preflight_launch_blocker(
        d.get("preflight_verdict"), p.get("resolver_preflight") or {})
    resolver = p.get("resolver") or {}
    latest_tick = resolver.get("selected") or resolver.get("latest") or {}
    if latest_tick:
        issue = latest_tick.get("target_issue")
        issue_bit = f"#{issue}" if issue is not None else "-"
        next_action = latest_tick.get("next_action") or "-"
        tick_state = _resolve_tick_state(latest_tick)
        if preflight_blocker and tick_state == "launch-ready":
            tick_state = f"held {preflight_blocker['verdict']}"
        lines.append(
            f"║ planner   : {latest_tick.get('backend') or '-'} {issue_bit} "
            f"lane={latest_tick.get('lane') or '-'} {tick_state} "
            f"age={_age_text(latest_tick.get('age_min'))} next={next_action}")
        if latest_tick.get("live_command_text") and not preflight_blocker:
            lines.append(f"║ launch    : {latest_tick.get('live_command_text')}")
        # #4589: re-surface the seat ramp-cap binding term + spawn-fail streak/cause
        # the producer stamps but the card used to drop — so an operator sees WHY
        # fan-out is pinned (a +N/tick ramp plateau) or spawns are failing (stale
        # credential), without a source dive for --seat-ramp-delta.
        sa = latest_tick.get("seat_adaptive") or {}
        if sa.get("signal_available"):
            lines.append(
                f"║ seat cap  : adaptive {sa.get('effective_target')} "
                f"(bound by {sa.get('binding')}, ramp +{sa.get('ramp_delta')}/tick; "
                f"live {sa.get('live')} + free {sa.get('seat_free')}, "
                f"ceiling {sa.get('hard_ceiling')})")
        selection = latest_tick.get("seat_selection") or {}
        if selection.get("summary"):
            lines.append(f"║ seat pick : {selection.get('summary')}")
        streak = _int(latest_tick.get("spawn_failed_streak"), 0) or 0
        seat_streak_n = _int(latest_tick.get("seat_streak"), 0) or 0
        if streak > 0 or seat_streak_n > 0:
            cause_bit = f", cause={latest_tick.get('cause')}" if latest_tick.get("cause") else ""
            seat_bit = f", seat-streak {seat_streak_n} (#4591)" if seat_streak_n else ""
            lines.append(f"║ spawn-fail: streak {streak}{seat_bit}{cause_bit} (#4589)")
    return lines


def _card_gate_lines(p: dict[str, Any]) -> list[str]:
    """The plan-admission gate and the worker-slot utilization row."""
    lines: list[str] = []
    resolver_pre = p.get("resolver_preflight") or {}
    if resolver_pre.get("schema"):
        seat = resolver_pre.get("seat") or {}
        unattributed = _int(seat.get("unattributed_live"), 0) or 0
        unattributed_bit = f" unattributed={unattributed}" if unattributed else ""
        lines.append(
            f"║ plan gate : {resolver_pre.get('_backend') or resolver_pre.get('product') or '-'} "
            f"{resolver_pre.get('verdict') or '-'} "
            f"live={_or_unknown(resolver_pre.get('live'))}/{_or_unknown(resolver_pre.get('cap'))} "
            f"{_slots_headroom_note(resolver_pre.get('live'), resolver_pre.get('cap'))} "
            f"seat={_or_unknown(seat.get('free'))}/{_or_unknown(seat.get('leased'))}{unattributed_bit} "
            f"os={_or_unknown(resolver_pre.get('os_worker_procs'))}")
    util = p.get("utilization") or {}
    if util.get("schema"):
        slots = util.get("worker_slots") or {}
        actions = ",".join(util.get("next_actions") or []) or "-"
        lines.append(
            f"║ use       : {util.get('state')} "
            f"slots={slots.get('live')}/{slots.get('cap')} "
            f"free={slots.get('headroom')} next={actions}")
    lines += _card_launch_gate_lines(p)
    return lines


def _card_launch_gate_lines(p: dict[str, Any]) -> list[str]:
    """The launcher's final admission gate (#6495): would ONE more worker actually
    launch, on which accounts, and — when not — what to wait on."""
    gate = p.get("spawn_gate") or {}
    if not gate.get("schema"):
        return []
    adm = gate.get("admission") or {}
    accounts = p.get("accounts") or {}
    if adm.get("_error"):
        state = f"UNKNOWN ({adm.get('_error')})"
    elif not adm.get("evaluated"):
        state = "n/a (no candidate account)"
    elif gate.get("would_launch"):
        state = f"ADMIT [{', '.join(str(a) for a in adm.get('admitted_accounts') or [])}]"
    else:
        state = (f"{adm.get('reason') or 'DEFER'} retry={adm.get('retry_after') or '?'}"
                 f" (0/{len(adm.get('accounts') or [])} accounts admitted)")
    return [
        f"║ admission : {state} "
        f"launchable={_or_unknown(accounts.get('launchable_seats'))}/"
        f"{_or_unknown(accounts.get('free_seats'))} free seat(s)",
        f"║ next      : {p.get('next_action') or '-'}",
    ]


def _card_flow_lines(p: dict[str, Any]) -> list[str]:
    """Backlog, closure and completion-rate rows."""
    lines: list[str] = []
    b = p.get("backlog") or {}
    c = p.get("closure") or {}
    if b.get("na"):
        lines.append("║ backlog   : n/a (--fast or gh timeout)")
    else:
        by = b.get("by_lane") or {}
        top = ", ".join(f"{k}={v}" for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:5])
        lines.append(f"║ backlog   : {b.get('open_issues')} open  [{top}]  unrouted={b.get('unrouted')}")
    if c.get("na"):
        lines.append("║ closure   : n/a (--fast or gh timeout)")
    else:
        cnt = c.get("counts") or {}
        lines.append(
            f"║ closure   : rate={c.get('closure_rate')} honest={c.get('honest_close_rate')}  "
            f"resolved={cnt.get('TRUE_RESOLVED')} data={cnt.get('DATA_RESOLVED')} "
            f"claimed={cnt.get('CLAIMED_CLOSED')} "
            f"closable-now={c.get('open_witnessed_closable')} (OPEN_WITNESSED)")
    tp = p.get("throughput") or {}
    if tp.get("na"):
        lines.append("║ rate      : n/a (--fast or gh timeout)")
    else:
        lines.append(
            f"║ rate      : {tp.get('verdict')}  {tp.get('completed_rate_per_hour')}/h completed "
            f"over the {tp.get('primary_window_hours')}h analysis window (target {tp.get('target_per_hour')}/h)")
    return lines


def _card_watch_lines(p: dict[str, Any]) -> list[str]:
    """The watch decision and the arrival-vs-service supply rows."""
    lines: list[str] = []
    watch = p.get("watch_decision") or {}
    if watch.get("schema"):
        cite = watch.get("cited") or {}
        wh = watch.get("window_hours") or 0
        fu = watch.get("follow_ups") or []
        fu_bit = (" follow-ups=" + ",".join(f"#{n}" for n in fu)) if fu else ""
        lines.append(
            f"║ watch     : {watch.get('verdict')} "
            f"({'LOOK' if watch.get('action_needed') else 'no action'}; {wh:g}h) "
            f"open {cite.get('open_now_start')}->{cite.get('open_now_end')} "
            f"loop-closed {cite.get('closed_by_loop_total_start')}->"
            f"{cite.get('closed_by_loop_total_end')} audit={watch.get('audit_status')} "
            f"sched={(watch.get('scheduled_task') or {}).get('classification')}{fu_bit}")
    br = p.get("backlog_rate") or {}
    if br.get("schema") and br.get("verdict") != "INSUFFICIENT_DATA":
        flag = " BACKLOG_OUTPACED" if br.get("backlog_outpaced") else ""
        lines.append(
            f"║ supply    : {br.get('verdict')}{flag} "
            f"arrival {br.get('arrival_rate_per_hour')}/h vs service "
            f"{br.get('service_rate_per_hour')}/h over {br.get('span_hours')}h "
            f"({br.get('consecutive_outpaced_windows')}/{br.get('k_consecutive')} "
            f"consecutive @ {br.get('window_hours'):g}h)")
    return lines


def _card_route_lines(p: dict[str, Any]) -> list[str]:
    """Route health: how many routes are probed, suppressed and rechecking."""
    lines: list[str] = []
    rh = p.get("route_health") or {}
    if rh.get("probed"):
        flagged = [r for r in (rh.get("routes") or []) if r.get("suppressed")]
        bits = ", ".join(
            f"{r.get('route')}={r.get('class')} "
            f"age={_age_text((r.get('probe_age_secs') or 0) / 60)} "
            f"cooldown={r.get('cooldown_remaining_secs')}s left"
            for r in flagged[:3])
        lines.append(
            f"║ routes    : {rh.get('probed')} probed, {rh.get('suppressed')} suppressed"
            + (f" [{bits}]" if bits else "") + " (#3035)")
        for r in flagged[:2]:
            if r.get("recheck"):
                lines.append(f"║ recheck   : {r.get('recheck')}")
    return lines


def _card_yield_lines(p: dict[str, Any]) -> list[str]:
    """What the workers produced: silent workers, spawn causes, low yield, ships, drought."""
    lines: list[str] = []
    w = p.get("workers") or {}
    sc = w.get("silent_count") or 0
    if sc:
        nums = ", ".join(f"#{s['issue']}" for s in (w.get("silent") or [])[:6])
        lines.append(f"║ workers   : {sc} silent (<= {_STUB_LOG_MAX_BYTES} B log, exited) [{nums}]")
    scz = p.get("spawn_causes") or {}
    if scz.get("schema") and _int(scz.get("spawn_failed"), 0):
        by = scz.get("by_cause") or {}
        bits = ", ".join(
            f"{c}={_int((by.get(c) or {}).get('count'), 0)}"
            for c in sorted(by, key=lambda c: -(_int((by.get(c) or {}).get("count"), 0) or 0))
            if _int((by.get(c) or {}).get("count"), 0))
        flag = " STALE_CRED_DRAIN" if (scz.get("stale_cred_alarm") or {}).get("red") else ""
        lines.append(
            f"║ spawn-fail: {scz.get('spawn_failed')}/{scz.get('spawns')} early-exit "
            f"(rate {scz.get('rate')}) [{bits}]{flag} (#4590)")
    ly = p.get("low_yield") or {}
    if ly.get("low_yield_count"):
        flagged = [r for r in (ly.get("lanes") or []) if r.get("verdict") == "LOW_YIELD"]
        bits = ", ".join(f"{r.get('lane')}={r.get('turns')}t/{r.get('sessions')}s/0c"
                         for r in flagged[:4])
        lines.append(
            f"║ low-yield : {ly.get('low_yield_count')} lane(s) >= {ly.get('turns_floor')}t, "
            f"0 closes [{bits}] (#2062)")
    spw = p.get("ships_per_worker") or {}
    if spw.get("attributed_ships"):
        bits = ", ".join(f"{w.get('worker')}={w.get('ships')}"
                         for w in (spw.get("workers") or [])[:4])
        unk = spw.get("unknown") or 0
        lines.append(
            f"║ ships/wkr : {spw.get('attributed_ships')} ship(s) / "
            f"{spw.get('worker_count')} worker(s) [{bits}]"
            + (f" +{unk} unattributed" if unk else "")
            + " (aid, #2065)")
    cd = p.get("commit_drought") or {}
    if cd.get("droughty"):
        lines.append(
            f"║ DROUGHT   : 0 fleet commits in {cd.get('hours')}h while ARMED — the "
            f"loop is shipping nothing; check spawn/preflight/backlog")
    return lines


def _card_health_lines(p: dict[str, Any]) -> list[str]:
    """Backend, hook, guard and run-status health rows."""
    lines: list[str] = []
    bh = p.get("backend_health") or {}
    flagged_rates = [r for r in (bh.get("stub_rate") or []) if r.get("majority_stub")]
    if flagged_rates:
        bits = ", ".join(f"{r.get('product')}={r.get('stub')}/{r.get('total')} stub"
                         for r in flagged_rates[:4])
        lines.append(f"║ backend   : majority-stub recent logs [{bits}]")
    hh = p.get("hook_health") or {}
    unhooked_rows = [r for r in (hh.get("by_backend") or []) if r.get("all_sessions_unhooked")]
    if unhooked_rows:
        bits = ", ".join(f"{r.get('product')}={r.get('hook_failures')} fail/{r.get('sessions')} sess"
                         f" ({int(float(r.get('failure_session_rate') or 0) * 100)}%)"
                         for r in unhooked_rows[:4])
        lines.append(f"║ hooks     : guard layer UNBOUND [{bits}] (#1277)")
    gd = p.get("guard") or {}
    if gd.get("sessions"):
        child_crashes = _int((gd.get("by_kind") or {}).get("CHILD_CRASH"), 0) or 0
        crash_bit = f" CRASH={child_crashes}" if child_crashes else ""
        loop_bit = ""
        if gd.get("livelock_candidates"):
            top = gd["livelock_candidates"][0]
            loop_bit = (f"  loop={top.get('tool') or top.get('kind')}/"
                        f"{top.get('reason') or top.get('verdict')} "
                        f"x{top.get('count')} run={top.get('longest_run')}")
        lines.append(
            f"║ guard     : {gd.get('sessions')} session(s) ({gd.get('recent_sessions', 0)} recent), "
            f"{gd.get('rows', 0)} decision(s) [DENY={gd.get('denied', 0)} "
            f"QUAR={gd.get('quarantined', 0)}{crash_bit}]  "
            f"empty={gd.get('empty_sessions', 0)}{_guard_empty_cause_bit(gd)}{loop_bit}")
    rs = p.get("run_status") or {}
    if rs.get("count"):
        bits = ", ".join(f"{k}={v}" for k, v in sorted((rs.get("liveness") or {}).items())) or "none"
        lines.append(f"║ run truth : dos status {rs.get('count')} RID(s), errors={rs.get('errors')} [{bits}]")
    return lines


def _card_lease_lines(p: dict[str, Any]) -> list[str]:
    """Lease inventory, the worker/lease reconciliation and the live-worker row."""
    lines: list[str] = []
    leases = p.get("leases") or {}
    if leases.get("read_error"):
        lines.append(f"║ leases    : unavailable ({leases.get('read_error')})")
    elif leases.get("active_count"):
        bits = "; ".join(_lease_summary_bits(leases))
        phantom = leases.get("blocking_stranded_count", 0)
        phantom_bit = f", {phantom} phantom" if phantom else ""
        lines.append(
            f"║ leases    : {leases.get('active_count')} active, "
            f"{leases.get('blocking_count', 0)} blocking{phantom_bit} [{bits}]")
    wl = p.get("worker_lease_check") or {}
    if wl.get("available") is False:
        lines.append(f"║ lease chk : unknown ({wl.get('error')})")
    elif wl:
        bits = []
        if wl.get("orphan_process_count"):
            bits.append("orphan-process "
                        + ",".join(_worker_lease_bucket_bits(wl.get("orphan_process") or [], key="worker")))
        if wl.get("orphan_lease_count"):
            bits.append("unmatched-live-lease "
                        + ",".join(_worker_lease_bucket_bits(wl.get("orphan_lease") or [], key="lease")))
        detail = f" [{'; '.join(bits)}]" if bits else ""
        lines.append(
            f"║ lease chk : clean={wl.get('clean_count', 0)} "
            f"orphan-process={wl.get('orphan_process_count', 0)} "
            f"unmatched-live-lease={wl.get('orphan_lease_count', 0)} "
            f"dead-holder={wl.get('dead_holder_count', 0)}{detail}")
        active_worker_line = _active_worker_summary(wl)
        if active_worker_line:
            lines.append("║ live work : " + active_worker_line.removeprefix("active resolver worker(s): "))
    lines += _card_lane_lease_lines(wl)
    return lines


def _card_lane_lease_lines(wl: dict[str, Any]) -> list[str]:
    """The kernel lane-lease holder rows (#5859) — dead holders and the reap action.

    Rendered from the SAME `worker_lease_check` dict the card already threads, and
    rendered even when the worker-sidecar half is unavailable, so a fenced `cmd/**`
    can never be hidden behind an unrelated probe failure.
    """
    lane = wl.get("lane_leases")
    if not lane:
        return []
    if not lane.get("available"):
        return [f"║ lane lease: UNKNOWN, not clean ({lane.get('read_error') or 'unreadable'})"]
    total = _int(lane.get("total"), 0) or 0
    if not total:
        return ["║ lane lease: none held"]
    dead = _int(lane.get("dead_count"), 0) or 0
    lines = [
        f"║ lane lease: {total} held — live={_int(lane.get('live_count'), 0) or 0} "
        f"dead-holder={dead} unknown={_int(lane.get('unknown_count'), 0) or 0}"]
    if dead:
        lines.append(f"║   DEAD    : {_lane_lease_bits(lane.get('dead') or [])}")
        # The action is deliberately multi-clause (#5859) — one clause per line so
        # the "do NOT reap" verdict is legible on the card instead of wrapping into
        # the terminal. A payload from an older producer carries a single string.
        action = lane.get("next_action") or _LANE_LEASE_NEXT_ACTION
        steps = [s.strip() for s in str(action).split(" · ") if s.strip()]
        for i, step in enumerate(steps):
            lines.append(f"║   {'next' if i == 0 else '    '}    : {step}")
    return lines


def render(p: dict[str, Any]) -> str:
    lines = _card_header_lines(p)
    lines += _card_capacity_lines(p)
    lines += _card_planner_lines(p)
    lines += _card_gate_lines(p)
    lines += _card_flow_lines(p)
    lines += _card_watch_lines(p)
    lines += _card_route_lines(p)
    lines += _card_yield_lines(p)
    lines += _card_health_lines(p)
    lines += _card_lease_lines(p)
    git = p.get("git") or {}
    if git.get("merge_in_progress"):
        lines.append(f"║ git       : MERGE_HEAD present — {git.get('next_action')}")
    lines.append("╚═ " + " | ".join(p.get("reasons") or []))
    return "\n".join(lines)


def _md_admission_clause(payload: dict[str, Any]) -> str:
    """One line of the launcher's final admission state for the operator doc."""
    gate = payload.get("spawn_gate") or {}
    adm = gate.get("admission") or {}
    accounts = payload.get("accounts") or {}
    seats = (f" ({accounts.get('launchable_seats')}/{accounts.get('free_seats')} "
             f"free seat(s) launchable)") if adm.get("evaluated") else ""
    if adm.get("_error"):
        return f"**UNKNOWN** — gate not consultable ({adm.get('_error')})"
    if not adm.get("evaluated"):
        return "n/a (no candidate account)"
    if gate.get("would_launch"):
        return f"`ADMIT`{seats}"
    return (f"**`{adm.get('reason') or 'DEFER'}`** — retry after "
            f"`{adm.get('retry_after') or '?'}`{seats}")


def _md_summary_bullets(payload: dict[str, Any], *, date: str) -> list[str]:
    """The Jekyll front matter, the dated title, and the dispatcher/worker/
    limiter/switcher/watchdog/supervisor summary bullets that open the doc."""
    d = payload.get("dispatcher") or {}
    s = payload.get("supervisor") or {}
    a = d.get("account") or {}
    wd = d.get("watchdog") or {}
    out = [
        # Jekyll front matter so the published status page keeps a stable <title> +
        # meta description (jekyll-seo-tag reads these). Without it, every --md regen
        # would strip the front matter the committed doc needs and the page would
        # read as discoverability debt to the SEO/AEO scorecard. Title/description are
        # deliberately date-independent so the page's identity is stable across regens.
        "---",
        'title: "fak issue dispatch status: fleet worker and backlog tracker"',
        'description: "Auto-generated fak issue dispatch status: dispatcher and worker '
        'state, open-issue backlog by lane, and closure-honesty rate across the fleet."',
        "---",
        "",
        f"# Issue dispatch status — {date}",
        "",
        "_Auto-generated by `tools/dispatch_status.py --md`. Do not hand-edit; "
        "re-run the tool (or the `FleetDispatchStatusDoc` task) to refresh._",
        "",
        f"- **dispatcher**: `{payload.get('verdict')}` "
        f"({'ok' if payload.get('ok') else 'ACTION'})",
        f"- **workers**: {_workers_live_clause(d)}; host "
        f"{'clean' if d.get('host_safe') else '**FLAGGED**'}",
        f"- **primary limiter**: `{(d.get('limiter') or {}).get('primary') or '-'}` "
        f"({_dispatch_limiter_terms(d.get('limiter') or {})})",
        f"- **switcher account**: `{a.get('tag') or '-'}` (t{a.get('tier')}, "
        f"{a.get('model') or '?'}), available={a.get('available')}",
        # The launcher's FINAL gate (#6495): the doc used to publish a growth verdict
        # the launcher would refuse, with no admission state to check it against.
        f"- **launch admission**: {_md_admission_clause(payload)}",
        f"- **next action**: {payload.get('next_action') or '-'}",
        "- **always-on watchdog**: "
        + ("installed (" + str(wd.get('status') or 'scheduled') + ")"
           if wd.get("installed") else
           ("**NOT installed**" if wd.get("installed") is False else "unknown")),
        f"- **supervisor**: `{s.get('verdict')}` "
        f"(alive {s.get('alive')}/{s.get('target')})",
    ]
    return out


def _md_capacity_bullets(payload: dict[str, Any]) -> list[str]:
    """Seat-inventory and lab-readiness bullets, including the publish command a
    held lab gate needs."""
    out: list[str] = []
    seat_line = _seat_inventory_summary_line(payload.get("seat_inventory") or {})
    if seat_line:
        out.append(f"- **seat inventory**: {seat_line.removeprefix('seat inventory: ')}")
    lab = payload.get("lab_readiness") or {}
    if lab.get("schema"):
        gate = "admit" if lab.get("admit_dispatch") else "hold"
        out.append(
            f"- **lab readiness**: `{_lab_link_label(lab)}` ({gate}; "
            f"next `{lab.get('next_action') or '-'}`)")
        if not lab.get("admit_dispatch"):
            cmd = (lab.get("commands") or {}).get("mark_clear")
            if cmd:
                out.append(f"- **lab publish command**: `{cmd}`")
    return out


def _md_resolver_bullets(payload: dict[str, Any]) -> list[str]:
    """The selected resolver tick: its state (held if a preflight blocker outranks
    launch-ready), the approved live command, the adaptive seat cap and pick,
    the spawn-fail streak, and the product-preflight summary."""
    out: list[str] = []
    d = payload.get("dispatcher") or {}
    preflight_blocker = _current_preflight_launch_blocker(
        d.get("preflight_verdict"), payload.get("resolver_preflight") or {})
    resolver = payload.get("resolver") or {}
    latest_tick = resolver.get("selected") or resolver.get("latest") or {}
    if latest_tick:
        issue = latest_tick.get("target_issue")
        issue_bit = f"#{issue}" if issue is not None else "-"
        tick_state = _resolve_tick_state(latest_tick)
        if preflight_blocker and tick_state == "launch-ready":
            tick_state = f"held {preflight_blocker['verdict']}"
        out.append(
            f"- **last resolver tick**: `{latest_tick.get('backend') or '-'}` "
            f"{issue_bit} lane `{latest_tick.get('lane') or '-'}` — "
            f"{tick_state}, age {_age_text(latest_tick.get('age_min'))}, "
            f"next `{latest_tick.get('next_action') or '-'}`")
        if latest_tick.get("live_command_text") and not preflight_blocker:
            out.append(f"- **approved live command**: `{latest_tick.get('live_command_text')}`")
        # #4589: the seat ramp-cap binding + spawn-fail streak/cause the card used to drop.
        sa = latest_tick.get("seat_adaptive") or {}
        if sa.get("signal_available"):
            out.append(
                f"- **seat cap**: adaptive {sa.get('effective_target')} "
                f"(bound by `{sa.get('binding')}`, ramp +{sa.get('ramp_delta')}/tick; "
                f"live {sa.get('live')} + free {sa.get('seat_free')}, ceiling {sa.get('hard_ceiling')})")
        selection = latest_tick.get("seat_selection") or {}
        if selection.get("summary"):
            out.append(f"- **seat pick**: {selection.get('summary')}")
        streak = _int(latest_tick.get("spawn_failed_streak"), 0) or 0
        seat_streak_n = _int(latest_tick.get("seat_streak"), 0) or 0
        if streak > 0 or seat_streak_n > 0:
            cause_bit = f", cause `{latest_tick.get('cause')}`" if latest_tick.get("cause") else ""
            seat_bit = f", seat-streak {seat_streak_n}" if seat_streak_n else ""
            out.append(f"- **spawn-fail streak**: {streak}{seat_bit}{cause_bit}")
    resolver_pre = payload.get("resolver_preflight") or {}
    resolver_pre_line = _resolver_preflight_summary(resolver_pre)
    if resolver_pre_line:
        out.append(f"- **resolver product preflight**: {resolver_pre_line.removeprefix('selected resolver preflight: ')}")
    return out


def _md_utilization_bullets(payload: dict[str, Any]) -> list[str]:
    """Worker-slot utilization and the capacity reconcile line."""
    out: list[str] = []
    resolver_pre = payload.get("resolver_preflight") or {}
    util = payload.get("utilization") or {}
    if util.get("schema"):
        slots = util.get("worker_slots") or {}
        actions = ", ".join(f"`{a}`" for a in (util.get("next_actions") or [])) or "`-`"
        out.append(
            f"- **utilization**: `{util.get('state')}`; "
            f"worker slots {slots.get('live')}/{slots.get('cap')} "
            f"({_slots_headroom_note(slots.get('live'), slots.get('cap'))}); next {actions}")
    recon = _capacity_reconcile(resolver_pre, payload.get("dispatcher") or {})
    if recon:
        out.append(recon)
    return out


def _md_run_state_bullets(payload: dict[str, Any]) -> list[str]:
    """Where run truth came from and whether git is parked mid-merge."""
    out: list[str] = []
    rs = payload.get("run_status") or {}
    if rs.get("count"):
        out.append(f"- **run status source**: `dos status` digests for {rs.get('count')} RID(s), "
                   f"errors={rs.get('errors')}")
    git = payload.get("git") or {}
    if git.get("merge_in_progress"):
        out.append(f"- **git wait state**: `MERGE_HEAD` present — {git.get('next_action')}")
    return out


def _md_lease_bullets(payload: dict[str, Any]) -> list[str]:
    """Lane-lease counts and the worker/lease cross-check summary bullets."""
    out: list[str] = []
    leases = payload.get("leases") or {}
    if leases.get("read_error"):
        out.append(f"- **lane leases**: unavailable (`{leases.get('read_error')}`)")
    elif leases.get("active_count"):
        out.append(f"- **lane leases**: {leases.get('active_count')} active; "
                   f"{leases.get('blocking_count', 0)} blocking current candidates")
    wl = payload.get("worker_lease_check") or {}
    if wl.get("available") is False:
        out.append(f"- **worker/lease cross-check**: unknown (`{wl.get('error')}`)")
    elif wl:
        out.append(f"- **worker/lease cross-check**: clean={wl.get('clean_count', 0)}, "
                   f"orphan-process={wl.get('orphan_process_count', 0)}, "
                   f"unmatched-live-lease={wl.get('orphan_lease_count', 0)}, "
                   f"dead-holder={wl.get('dead_holder_count', 0)}")
        active_worker_line = _active_worker_summary(wl)
        if active_worker_line:
            out.append(f"- **active resolver workers**: {active_worker_line.removeprefix('active resolver worker(s): ')}")
    lane = wl.get("lane_leases") or {}
    if lane and not lane.get("available"):
        out.append("- **lane-lease holders**: UNKNOWN, not clean "
                   f"(`{lane.get('read_error') or 'unreadable'}`)")
    elif lane:
        out.append(f"- **lane-lease holders**: {_int(lane.get('total'), 0) or 0} held — "
                   f"live={_int(lane.get('live_count'), 0) or 0}, "
                   f"dead-holder={_int(lane.get('dead_count'), 0) or 0}, "
                   f"unknown={_int(lane.get('unknown_count'), 0) or 0}")
    return out


def _md_backlog_section(payload: dict[str, Any]) -> list[str]:
    """`## Backlog by lane` — open issues routed to each lane."""
    out: list[str] = []
    b = payload.get("backlog") or {}
    out += [
        "",
        "## Backlog by lane (issue → lane sync)",
        "",
    ]
    if b.get("na"):
        out.append("_Backlog n/a this run (gh fold skipped or timed out)._")
    else:
        out += [
            f"Open issues: **{b.get('open_issues')}** — routed {b.get('routed')}, "
            f"unrouted {b.get('unrouted')}.",
            "",
            "| lane | open issues |",
            "|---|---|",
        ]
        by_lane = b.get("by_lane") or {}
        for lane, n in sorted(by_lane.items(), key=lambda kv: (-kv[1], kv[0])):
            out.append(f"| {lane} | {n} |")
    return out


def _md_closure_section(payload: dict[str, Any]) -> list[str]:
    """`## Closure honesty` — the strict diff-witnessed closure rate and its buckets."""
    out: list[str] = []
    c = payload.get("closure") or {}
    out += ["", "## Closure honesty", ""]
    if c.get("na"):
        out.append("_Closure audit n/a this run (gh/dos fold skipped or timed out)._")
    else:
        cnt = c.get("counts") or {}
        honest = c.get("honest_close_rate")
        out += [
            f"`closure_rate` = **{c.get('closure_rate')}** "
            f"(TRUE_RESOLVED / (TRUE_RESOLVED + CLAIMED_CLOSED) — strict diff-witness)"
            + (f"; `honest_close_rate` = **{honest}** (also credits the DATA rung)."
               if honest is not None else "."),
            "",
            "| bucket | count |",
            "|---|---|",
            f"| TRUE_RESOLVED | {cnt.get('TRUE_RESOLVED', 0)} |",
            f"| DATA_RESOLVED | {cnt.get('DATA_RESOLVED', 0)} |",
            f"| CLAIMED_CLOSED | {cnt.get('CLAIMED_CLOSED', 0)} |",
            f"| OPEN_WITNESSED (closable now) | {c.get('open_witnessed_closable')} |",
        ]
    return out


def _md_throughput_section(payload: dict[str, Any]) -> list[str]:
    """`## Throughput` — closed/completed issues per hour per trailing window."""
    out: list[str] = []
    tp = payload.get("throughput") or {}
    out += ["", "## Throughput (closed issues per hour)", ""]
    if tp.get("na"):
        out.append("_Throughput n/a this run (gh fold skipped or timed out)._")
    else:
        out += [
            f"`verdict` = **{tp.get('verdict')}** — **{tp.get('completed_rate_per_hour')}/h** "
            f"completed over the trailing **{tp.get('primary_window_hours')}h** window "
            f"(target **{tp.get('target_per_hour')}/h**). Graded on the *completed* "
            "(resolved, not wontfix/dup) rate.",
            "",
            "| window | closed | completed | completed /h | loop-closed | loop /h |",
            "|---|---|---|---|---|---|",
        ]
        pw = tp.get("per_window") or {}
        lpw = tp.get("loop_per_window") or {}
        for key in ("1h", "3h", "6h", "12h", "24h"):
            g = pw.get(key)
            if not g:
                continue
            lp = lpw.get(key) or {}
            out.append(
                f"| {key} | {g.get('closed')} | {g.get('completed')} | "
                f"{g.get('completed_rate_per_hour')} | {lp.get('loop_closed', '-')} | "
                f"{lp.get('loop_rate_per_hour', '-')} |")
        last = tp.get("last_loop_close_age_min")
        out += ["",
                "Loop's last attributed close: "
                + (f"{last} min ago." if last is not None else "**none on record**.")
                + " A gh-rate far above the loop-rate means humans/peers are draining "
                "the backlog, not the dispatcher."]
    return out


def _md_watch_section(payload: dict[str, Any]) -> list[str]:
    """`## Watch decision` — why a trailing window ended in no action."""
    out: list[str] = []
    watch = payload.get("watch_decision") or {}
    if watch.get("schema"):
        cite = watch.get("cited") or {}
        wh = watch.get("window_hours") or 0
        fu = watch.get("follow_ups") or []
        sched = watch.get("scheduled_task") or {}
        out += ["", "## Watch decision (why no action)", "",
                f"`verdict` = **{watch.get('verdict')}** "
                f"({'operator look suggested' if watch.get('action_needed') else 'intentionally no action'}"
                f", trailing **{wh:g}h**) — {watch.get('why')}",
                "",
                f"- `open_now`: {cite.get('open_now_start')} -> {cite.get('open_now_end')} "
                f"(delta {cite.get('open_now_delta')})",
                f"- `closed_by_loop_total`: {cite.get('closed_by_loop_total_start')} -> "
                f"{cite.get('closed_by_loop_total_end')} "
                f"(+{cite.get('closed_now_sum')} witnessed closes in window)",
                f"- closure audit: {watch.get('audit_status')}",
                f"- scheduled task: {sched.get('classification')} "
                f"(status={sched.get('status') or '-'})",
                "- follow-up tickets from this watch: "
                + (", ".join(f"#{n}" for n in fu) if fu else "none listed")]
    return out


def _md_lease_section(payload: dict[str, Any]) -> list[str]:
    """`## Active lane leases` — which leases are held and which block candidates."""
    out: list[str] = []
    leases = payload.get("leases") or {}
    out += ["", "## Active lane leases", ""]
    if leases.get("read_error"):
        out.append(f"_Lease read unavailable: `{leases.get('read_error')}`._")
    elif not leases.get("active_count"):
        extra = ""
        if leases.get("expired_count"):
            extra = f" {leases.get('expired_count')} expired lease record(s) are reapable residue."
        out.append("No active lane leases under `refs/fak/locks/*`." + extra)
    else:
        if leases.get("candidate_source_available") is False:
            out.append(
                "_Backlog routing was unavailable, so candidate-blocking status is unknown._")
        elif leases.get("blocking_count"):
            out.append(
                f"**{leases.get('blocking_count')}** active lease(s) overlap current routed "
                "candidate issues; those candidates should wait or be repartitioned.")
        else:
            out.append("Active leases are present, but none overlap the current routed candidates.")
        if leases.get("expired_count"):
            out.append(
                f"{leases.get('expired_count')} expired lease record(s) are visible but non-blocking.")
        out += [
            "",
            "| lease id | lane | age | ttl | liveness | blocks candidate | holder | tree |",
            "|---|---|---:|---:|---|---|---|---|",
        ]
        for row in leases.get("active") or []:
            tree = ", ".join(f"`{t}`" for t in (row.get("tree") or [])) or "—"
            holder = str(row.get("holder") or "—")
            out.append(
                f"| `{row.get('id')}` | {row.get('lane') or '—'} | "
                f"{_age_text(row.get('age_min'))} | {row.get('ttl_seconds')}s | "
                f"{_lease_liveness_text(row)} | {_lease_block_text(row)} | "
                f"`{holder}` | {tree} |")
    return out


def _md_worker_lease_section(payload: dict[str, Any]) -> list[str]:
    """`## Worker / lease cross-check` — clean matches, orphan processes, orphan leases."""
    out: list[str] = []
    wl = payload.get("worker_lease_check") or {}
    out += ["", "## Worker / lease cross-check", ""]
    if wl.get("available") is False:
        out.append(f"_Worker liveness unavailable: `{wl.get('error')}`._")
    elif not wl:
        out.append("_Worker/lease cross-check did not run._")
    else:
        out.append(
            f"clean={wl.get('clean_count', 0)}, "
            f"orphan-process={wl.get('orphan_process_count', 0)}, "
            f"unmatched-live-lease={wl.get('orphan_lease_count', 0)}, "
            f"dead-holder={wl.get('dead_holder_count', 0)}.")
        if wl.get("orphan_lease_count"):
            out.append("")
            out.append(
                "_Unmatched live leases have no local live worker sidecar in this checkout. "
                "They are not automatically safe to reap; use `fak leaseref audit` / "
                "`fak leaseref liveness` and only `fak leaseref reap` expired records._")
        clean_rows = wl.get("clean") or []
        out += ["", "Clean matches:", ""]
        if not clean_rows:
            out.append("_No live worker has a matching active lease._")
        else:
            out += [
                "| worker | issue | pid | lease |",
                "|---|---:|---:|---|",
            ]
            for row in clean_rows:
                worker = row.get("worker") or {}
                lease = row.get("lease") or {}
                out.append(f"| `{worker.get('worker')}` | #{worker.get('issue')} | "
                           f"{worker.get('pid')} | `{lease.get('id')}` |")
        orphan_process = wl.get("orphan_process") or []
        out += ["", "Orphan processes:", ""]
        if not orphan_process:
            out.append("_No OS-visible dispatch worker is missing an active lease._")
        else:
            out += [
                "| worker | issue | pid | lease id | reason |",
                "|---|---:|---:|---|---|",
            ]
            for row in orphan_process:
                worker = row.get("worker") or {}
                out.append(f"| `{worker.get('worker')}` | #{worker.get('issue')} | "
                           f"{worker.get('pid')} | `{worker.get('lease_id') or '—'}` | "
                           f"{row.get('reason')} |")
        orphan_lease = wl.get("orphan_lease") or []
        out += ["", "Orphan leases:", ""]
        if not orphan_lease:
            out.append("_No active lease is missing a local live worker sidecar._")
        else:
            out += [
                "| lease | lane | holder | reason |",
                "|---|---|---|---|",
            ]
            for row in orphan_lease:
                lease = row.get("lease") or {}
                out.append(f"| `{lease.get('id')}` | {lease.get('lane') or '—'} | "
                           f"`{lease.get('holder') or '—'}` | {row.get('reason')} |")
    out += _md_lane_lease_section(wl)
    return out


def _md_lane_lease_section(wl: dict[str, Any]) -> list[str]:
    """`### Kernel lane leases` — every `dos lease-lane live` record and its holder
    verdict (#5859). Rendered unconditionally so a fenced lane is never silent."""
    lane = wl.get("lane_leases")
    if not lane:
        return []
    out: list[str] = ["", "### Kernel lane leases (`dos lease-lane live`)", ""]
    if not lane.get("available"):
        out.append("_Lane-lease holder liveness UNKNOWN — not clean: "
                   f"`{lane.get('read_error') or 'unreadable'}`._")
        return out
    rows = lane.get("rows") or []
    if not rows:
        out.append("_No lane lease is held._")
        return out
    out.append(f"{len(rows)} held — live={_int(lane.get('live_count'), 0) or 0}, "
               f"dead-holder={_int(lane.get('dead_count'), 0) or 0}, "
               f"unknown={_int(lane.get('unknown_count'), 0) or 0}.")
    if lane.get("dead_count"):
        out += ["", f"**Next action:** {lane.get('next_action') or _LANE_LEASE_NEXT_ACTION}"]
    out += ["", "| lane | holder | pid | held | state | evidence |", "|---|---|---:|---:|---|---|"]
    for row in rows:
        out.append(
            f"| `{row.get('lane') or '—'}` | `{row.get('holder') or '—'}` | "
            f"{row.get('pid') if row.get('pid') is not None else '—'} | "
            f"{_age_text(row.get('age_min'))} | **{row.get('holder_state')}** | "
            f"{row.get('holder_evidence') or ''} |")
    return out


def _md_backend_health_section(payload: dict[str, Any]) -> list[str]:
    """`## Backend health / reallocation` — backends held dead and their stub rate."""
    out: list[str] = []
    bh = payload.get("backend_health") or {}
    dead = bh.get("dead") or []
    stub_rates = bh.get("stub_rate") or []
    majority_stub = [r for r in stub_rates if r.get("majority_stub")]
    out += ["", "## Backend health / reallocation", ""]
    if not dead and not majority_stub:
        out.append("All backends healthy — none held dead, no lane reallocated.")
    elif not dead:
        out.append(
            "No `backend-health-*.json` sidecar is holding a backend dead, but the "
            "recent log sweep flags a majority-stub backend. That means the status "
            "card is no longer treating absence of a sidecar as proof of health.")
    else:
        out += [
            f"**{len(dead)}** backend(s) are spinning dead (a streak of banner-only / "
            "0-byte worker logs the weekly-cap gate doesn't catch — e.g. a credit-walled "
            "codex or a glm worker that prints only its startup banner). The dispatcher "
            "holds their spawns and a healthy backend claims the freed lane + budget; "
            "one re-probe worker is admitted per interval, so each auto-restores the "
            "moment it produces a real turn again.",
            "",
            "| backend | dead since | lane reallocated | re-probe (min) |",
            "|---|---|---|---|",
        ]
        for b in dead:
            out.append(f"| {b.get('product')} | {b.get('since')} | "
                       f"{b.get('abandoned_lane') or '—'} | {b.get('reprobe_min') or '—'} |")
    out += ["", "Backend stub-rate (recent resolve logs):", ""]
    if not stub_rates:
        out.append("_No recent resolve logs in the backend sweep window._")
    else:
        out += [
            "| backend | lookback | recent logs | productive | stub | stub rate | verdict | evidence |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
        for row in stub_rates:
            verdict = "**MAJORITY_STUB**" if row.get("majority_stub") else "ok"
            evidence = ", ".join(f"`{log}`" for log in (row.get("evidence_logs") or [])[:3]) or "—"
            rate = row.get("stub_rate")
            out.append(f"| {row.get('product')} | {row.get('lookback_min')}m | "
                       f"{row.get('total')} | {row.get('productive')} | {row.get('stub')} | "
                       f"{rate} | {verdict} | {evidence} |")
    return out


def _md_hook_health_section(payload: dict[str, Any]) -> list[str]:
    """`## Hook health` — which backends ran unhooked by the guard layer."""
    out: list[str] = []
    hh = payload.get("hook_health") or {}
    hook_rows = hh.get("by_backend") or []
    out += ["", "## Hook health (guard-layer binding)", ""]
    if not hook_rows:
        out.append("_No recent resolve logs in the hook-failure sweep window._")
    else:
        unhooked = [r for r in hook_rows if r.get("all_sessions_unhooked")]
        if unhooked:
            out += [
                f"**{len(unhooked)}** backend(s) ran **UNHOOKED** by the guard layer over "
                "the recent window — every session logged `hook: <name> Failed`, so the fak "
                "guard hooks (PreToolUse / PostToolUse / Stop) never bound. `claude` binds "
                "the guard hooks natively; a non-claude backend (codex/opencode) runs its "
                "own native hook config, and when that config can't reach the dos hook CLI "
                "at runtime every lifecycle hook fails. Such a worker stays productive but "
                "runs unguarded by the hook backstop — the commit-path / `OFF_TRUNK` guard "
                "is what still holds the line (#1277).",
                "",
            ]
        else:
            out += ["All backends bound their guard hooks over the recent window — "
                    "no `hook: <name> Failed` storm.", ""]
        out += [
            "| backend | lookback | sessions | sessions w/ hook fail | fail-session rate | hook failures | verdict | evidence |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
        for row in hook_rows:
            verdict = "**UNHOOKED**" if row.get("all_sessions_unhooked") else "ok"
            evidence = ", ".join(f"`{log}`" for log in (row.get("evidence_logs") or [])[:3]) or "—"
            rate = row.get("failure_session_rate")
            out.append(f"| {row.get('product')} | {row.get('lookback_min')}m | "
                       f"{row.get('sessions')} | {row.get('sessions_with_hook_failures')} | "
                       f"{rate} | {row.get('hook_failures')} | {verdict} | {evidence} |")
    return out


def _md_guard_section(payload: dict[str, Any]) -> list[str]:
    """`## Guard coverage` — the kernel decisions recorded on the dispatch path."""
    out: list[str] = []
    gd = payload.get("guard") or {}
    out += ["", "## Guard coverage (kernel decisions on the dispatch path)", ""]
    if not gd.get("dir_present"):
        out.append(
            "_No `.dispatch-runs/guard-audit/` journal yet — no guarded worker has run "
            "on this host. The dispatch worker fronts every session with `fak guard` by "
            "default (opt out `FLEET_DOGFOOD_GUARD=0`); the trail appears once one runs._")
    elif not gd.get("sessions"):
        out.append(
            "_The guard-audit directory exists but holds no journals — the guard wire is "
            "configured but never exercised by a launched worker._")
    else:
        by_kind = gd.get("by_kind") or {}
        out += [
            f"`fak guard` recorded **{gd.get('rows', 0)}** kernel decision(s) across "
            f"**{gd.get('sessions', 0)}** guarded dispatch session(s) "
            f"(**{gd.get('recent_sessions', 0)}** within {gd.get('lookback_min')}m) — the "
            "WITNESS that the concurrent-dispatch path ran THROUGH the kernel, not just "
            "got configured to. Each session owns a unique hash-chained journal "
            "(`fak audit verify <file>`); the decision mix is the kernel's verdict tally.",
            "",
            f"- **denied** (DENY + RESULT_DENY): {gd.get('denied', 0)}",
            f"- **quarantined**: {gd.get('quarantined', 0)}",
            f"- **empty sessions** (booted under guard, no adjudicated tool call): "
            f"{gd.get('empty_sessions', 0)}"
            f"{_guard_empty_cause_bit(gd)}",
            "",
            "| decision kind | count |",
            "|---|---:|",
        ]
        for kind, n in sorted(by_kind.items(), key=lambda kv: (-kv[1], kv[0])):
            out.append(f"| {kind} | {n} |")
        evidence = gd.get("evidence") or []
        if evidence:
            out += ["", "Recent journals: "
                    + ", ".join(f"`{name}`" for name in evidence) + "."]
    return out


def _md_silent_worker_section(payload: dict[str, Any]) -> list[str]:
    """`## Workers that produced nothing` — spawns at or below the real-turn floor."""
    out: list[str] = []
    w = payload.get("workers") or {}
    sc = w.get("silent_count") or 0
    out += ["", "## Workers that produced nothing", ""]
    if not sc:
        out.append("None — every spawned worker either produced output or is still running.")
    else:
        out += [
            f"**{sc}** worker(s) exited at or below the real-turn floor "
            f"({_STUB_LOG_MAX_BYTES} B) — a 0-byte spawn or a banner-only stub "
            "(spawned, committed nothing). The anti-churn cooldown advances the picker "
            "past these, so the loop still progresses — but each is worth an operator's "
            "eye: an epic-shaped issue too large to land in one shot, or a dead backend "
            "spawning no-ops (a majority-stub backend is the tell — see #1276).",
            "",
            "| issue | spawned (utc stamp) | kind | bytes | log |",
            "|---|---|---|---|---|",
        ]
        for sw in (w.get("silent") or []):
            out.append(f"| #{sw.get('issue')} | {sw.get('stamp')} | {sw.get('kind') or '—'} | "
                       f"{sw.get('size') if sw.get('size') is not None else '—'} | `{sw.get('log')}` |")
    return out


def _md_low_yield_section(payload: dict[str, Any]) -> list[str]:
    """`## Low-yield lanes` — turns spent versus ancestry closes, per lane."""
    out: list[str] = []
    ly = payload.get("low_yield") or {}
    ly_lanes = ly.get("lanes") or []
    out += ["", "## Low-yield lanes (turns spent vs ancestry closes)", ""]
    if not ly_lanes:
        out.append(
            f"No resolve sessions in the last {ly.get('lookback_min', _LOW_YIELD_LOOKBACK_MIN)}m, "
            "so no per-lane turns-vs-closes signal this run.")
    else:
        flagged = [r for r in ly_lanes if r.get("verdict") == "LOW_YIELD"]
        if flagged:
            out += [
                f"**{len(flagged)}** lane(s) spent **>= {ly.get('turns_floor')} turns** over the "
                f"last {ly.get('lookback_min')}m yet landed **zero** ancestry-closes "
                "(a `Fixes #`/`Closes #` commit touching the lane's tree). `silent_workers` "
                "misses this — the log is full, the worker just closed nothing — so this is "
                "the per-lane feedback `pick_lane` needs to re-scope or exclude a stuck lane "
                "instead of re-seating it blind (#2062). Informational only; it never flips "
                "the card's verdict.",
                "",
            ]
        else:
            out += ["Every recent lane either stayed under the turn floor or landed at least "
                    "one ancestry-close — no low-yield lane this window.", ""]
        out += [
            "| lane | sessions | turns | max/session | closes | verdict | tree | evidence |",
            "|---|---:|---:|---:|---:|---|---|---|",
        ]
        for r in ly_lanes:
            verdict = "**LOW_YIELD**" if r.get("verdict") == "LOW_YIELD" else "ok"
            closes = r.get("closes")
            closes_txt = "—" if closes is None else str(closes)
            tree = ", ".join(f"`{t}`" for t in (r.get("tree") or [])) or "—"
            evidence = ", ".join(f"`{log}`" for log in (r.get("evidence_logs") or [])[:3]) or "—"
            out.append(
                f"| {r.get('lane')} | {r.get('sessions')} | {r.get('turns')} | "
                f"{r.get('max_session_turns')} | {closes_txt} | {verdict} | {tree} | {evidence} |")
    return out


def _md_ships_section(payload: dict[str, Any]) -> list[str]:
    """`## Ships per worker` — best-effort `(fak-worker <id>)` trailer attribution."""
    out: list[str] = []
    spw = payload.get("ships_per_worker") or {}
    out += ["", "## Ships per worker (best-effort attribution)", ""]
    spw_workers = spw.get("workers") or []
    if not spw.get("attributed_ships"):
        out.append(
            "No `(fak-worker <id>)` trailers in the recent window — a dispatched worker "
            "carries this trailer best-effort (sourced from the `FLEET_WORKER_ID` env "
            "stamp), so ships-per-worker is attributable once workers start emitting it "
            "(#2065). It is an **attribution aid, not a witness**; nothing is gated on it.")
    else:
        unk = spw.get("unknown") or 0
        out += [
            f"**{spw.get('attributed_ships')}** commit(s) carry a `(fak-worker <id>)` "
            f"trailer over the last {spw.get('lookback_min', _SHIPS_PER_WORKER_LOOKBACK_MIN)}m, "
            f"attributed to **{spw.get('worker_count')}** worker(s)"
            + (f" ({unk} matched but unattributed)" if unk else "")
            + ". The trailer is agent-emitted best-effort — an **attribution aid, not a "
            "witness** (the `(fak <leaf>)` ship stamp + `Fixes #N` remain the ground "
            "truth); this fold never flips the card's verdict (#2065).",
            "",
            "| worker | ships |",
            "|---|---:|",
        ]
        for w in spw_workers:
            out.append(f"| `{w.get('worker')}` | {w.get('ships')} |")
        if unk:
            out.append(f"| _unattributed_ | {unk} |")
    return out


def _md_contract_repair_section(payload: dict[str, Any]) -> list[str]:
    """`## Issue-contract repair flow` — the read-only repair pass a held issue needs."""
    out: list[str] = []
    out += ["", "## Issue-contract repair flow", ""]
    out += [
        "When `fak issue contract` / `fak dispatch route --json` hold an issue "
        "below `DEFAULT_ISSUE_CONTRACT_MIN_SCORE`, that is a scaffold gap, not a "
        "reason to `--force` dispatch. Run the read-only repair-assist pass instead:",
        "",
        "1. `python tools/issue_contract_repair.py --lane <lane> --limit N --json` "
        "— classifies each held issue's contract reasons into a repair kind "
        "(`split`/`scope`/`route`/`noise`/`private`/`template`/`other`) and builds "
        "a manifest row per issue. Never edits, labels, comments on, or closes "
        "an issue.",
        "2. `template`-kind rows carry a dry-run-computed normalized-header fix "
        "(`ready: true`); every other kind lists exactly the missing contract "
        "fields as one-line human questions — content is never invented.",
        "3. An operator or follow-up agent answers the scaffolded questions (or "
        "reviews the `template` fix) and applies it via a manual `gh issue edit`.",
        "4. Re-run `fak issue contract --live` to confirm the score reaches the "
        "floor, then dispatch proceeds through the normal picker — no gate "
        "bypass needed.",
    ]
    return out


def render_md(payload: dict[str, Any], *, date: str) -> str:
    """The committed, human-readable status surface: which issues are synced to
    which lanes, how closure is progressing, and any worker that produced nothing.

    This is the plan-doc-equivalent for a plan-empty repo whose backlog is GitHub
    issues — an operator opens ONE file instead of grepping gitignored runtime.
    Date is git-derived by the caller (deterministic; the renderer takes no clock).
    """
    out = _md_summary_bullets(payload, date=date)
    out += _md_capacity_bullets(payload)
    out += _md_resolver_bullets(payload)
    out += _md_utilization_bullets(payload)
    out += _md_run_state_bullets(payload)
    out += _md_lease_bullets(payload)
    out += _md_backlog_section(payload)
    out += _md_closure_section(payload)
    out += _md_throughput_section(payload)
    out += _md_watch_section(payload)
    out += _md_lease_section(payload)
    out += _md_worker_lease_section(payload)
    out += _md_backend_health_section(payload)
    out += _md_hook_health_section(payload)
    out += _md_guard_section(payload)
    out += _md_silent_worker_section(payload)
    out += _md_low_yield_section(payload)
    out += _md_ships_section(payload)
    out += _md_contract_repair_section(payload)
    out += ["", "---", "", "Reasons: " + "; ".join(payload.get("reasons") or [])]
    return "\n".join(out) + "\n"


def _rate_str(v: Any) -> str:
    """Trim a float rate to a compact form (0.80 -> 0.8, 2.0 -> 2) without trailing
    zero noise, leaving non-numbers untouched."""
    if isinstance(v, (int, float)):
        s = f"{float(v):.2f}".rstrip("0").rstrip(".")
        return s or "0"
    return str(v)


def _join_limited(rows: list[str], *, limit: int = 3) -> str:
    kept = rows[:limit]
    if len(rows) > limit:
        kept.append(f"+{len(rows) - limit} more")
    return "; ".join(kept)


def _dispatch_capacity_line(payload: dict[str, Any]) -> str:
    d = payload.get("dispatcher") or {}
    a = d.get("account") or {}
    b = payload.get("backlog") or {}
    c = payload.get("closure") or {}
    lz = payload.get("leases") or {}
    wl = payload.get("worker_lease_check") or {}

    parts: list[str] = []
    if d.get("live") is not None or d.get("cap") is not None:
        parts.append(f"worker slots {d.get('live')}/{d.get('cap')} active "
                     f"({_slots_headroom_note(d.get('live'), d.get('cap'))})")
    limiter = d.get("limiter") or {}
    if limiter.get("primary"):
        parts.append(f"limiter {limiter.get('primary')} ({_dispatch_limiter_terms(limiter)})")
    if a.get("tag"):
        parts.append(f"next account {a.get('tag')} t{a.get('tier')}"
                     + ("" if a.get("available") else " unavailable"))
    if not d.get("host_safe"):
        parts.append("host flagged")
    if not b.get("na") and b.get("open_issues") is not None:
        ur = b.get("unrouted")
        parts.append(f"backlog {b.get('open_issues')}"
                     + (f" ({ur} unrouted)" if ur else ""))
    if not c.get("na") and c.get("closure_rate") is not None:
        hk = c.get("honest_close_rate")
        parts.append(f"closure {_rate_str(c.get('closure_rate'))}"
                     + (f"/{_rate_str(hk)}" if hk is not None else ""))
    if lz.get("active_count"):
        parts.append(f"leases {lz.get('active_count')} active"
                     + (f" ({lz.get('blocking_count')} blocking)" if lz.get("blocking_count") else ""))
    if wl and wl.get("available") is not False:
        parts.append(f"lease-check clean {wl.get('clean_count', 0)}"
                     f"/orphan-proc {wl.get('orphan_process_count', 0)}"
                     f"/unmatched-live-lease {wl.get('orphan_lease_count', 0)}"
                     f"/dead-holder {wl.get('dead_holder_count', 0)}")
    # A dead lane-lease holder fences a whole tree for the fleet — it rides the
    # capacity line even when the worker-sidecar half is unavailable (#5859).
    lane = wl.get("lane_leases") or {}
    if lane and not lane.get("available"):
        parts.append("lane-lease liveness UNKNOWN")
    elif _int(lane.get("dead_count"), 0):
        parts.append(f"LANE LEASES TTL-EXPIRED: dead-holder {lane.get('dead_count')}"
                     f"/{lane.get('total')} (admission fold already elides; do not reap)")
    return "capacity: " + " · ".join(parts) if parts else ""


def _dispatch_trend_line(tp: dict[str, Any]) -> str:
    if tp.get("na"):
        return ""
    per_window = tp.get("per_window") or {}
    rates: list[str] = []
    for key in ("1h", "3h", "6h", "24h"):
        row = per_window.get(key) or {}
        rate = row.get("completed_rate_per_hour")
        if rate is not None:
            rates.append(f"{key} {_rate_str(rate)}/h")
    if not rates:
        rate = tp.get("completed_rate_per_hour")
        if rate is None:
            return ""
        rates.append(f"{tp.get('primary_window_hours')}h {_rate_str(rate)}/h")

    bits = ["completed " + " · ".join(rates)]
    target = tp.get("target_per_hour")
    if target is not None:
        bits.append(f"target {_rate_str(target)}/h")
    loop_windows = tp.get("loop_per_window") or {}
    primary = f"{tp.get('primary_window_hours')}h"
    loop_primary = loop_windows.get(primary) or {}
    if loop_primary.get("loop_rate_per_hour") is not None:
        bits.append(f"loop {primary} {_rate_str(loop_primary.get('loop_rate_per_hour'))}/h")
    last = tp.get("last_loop_close_age_min")
    if last is not None:
        bits.append(f"last loop close {_rate_str(last)}m ago")
    return "trend: " + "; ".join(bits)


def _dispatch_gate_slack_buckets(payload: dict[str, Any]) -> dict[str, list[str]]:
    d = payload.get("dispatcher") or {}
    a = d.get("account") or {}
    wd = d.get("watchdog") or {}
    sup = payload.get("supervisor") or {}
    buckets = {"expected": [], "auto-solving": [], "action": []}

    verdict = str(payload.get("verdict") or "")
    preflight = str(d.get("preflight_verdict") or "")
    cap = payload.get("weekly_cap") or {}
    if cap:
        buckets["expected"].append(
            f"{a.get('tag') or 'account'} weekly-capped until "
            f"{cap.get('reset_text') or cap.get('until') or '?'}; scheduler waits")
    if verdict == "AT_CAP" or preflight == "REFUSE_AT_CAP":
        buckets["expected"].append("at configured worker-slot cap")
    if verdict == "BLOCKED_ON_ACCOUNT" or preflight == "REFUSE_NO_ACCOUNT":
        buckets["expected"].append("no free worker account; switcher resumes when one frees")
    if verdict == "STALLED" and payload.get("ok"):
        buckets["expected"].append("scheduler liveness says STALLED but gate marks it ok; see auto/action below")

    sup_verdict = str(sup.get("verdict") or "")
    if sup_verdict == "PLAN_SURFACE_EMPTY" and not sup.get("alive") and not sup.get("target"):
        buckets["expected"].append(
            "supervisor PLAN_SURFACE_EMPTY: expected for issue-driven dispatch; not session health")
    elif sup_verdict and sup_verdict not in ("READY", "OK", "READY_TO_CANARY"):
        buckets["action"].append(
            f"supervisor {sup_verdict} (alive {sup.get('alive')}/{sup.get('target')})")

    if wd.get("installed") is False:
        buckets["action"].append("always-on watchdog not installed; register FleetIssueDispatch")
    if not d.get("host_safe"):
        buckets["action"].append("host resource guard flagged a process; inspect before growing")
    if preflight == "REFUSE_INSPECT":
        buckets["action"].append("spawn preflight could not run; inspect the preflight error")
    auth_seat_action = _auth_failed_seat_action(payload.get("seat_inventory") or {})
    if auth_seat_action:
        buckets["action"].append(auth_seat_action)
    double_booked_action = _double_booked_seat_action(payload.get("seat_inventory") or {})
    if double_booked_action:
        buckets["action"].append(double_booked_action)
    lab = payload.get("lab_readiness") or {}
    if lab.get("schema"):
        link = _lab_link_label(lab)
        action = lab.get("next_action") or "publish-lab-readiness"
        if lab.get("admit_dispatch"):
            buckets["expected"].append(f"lab readiness {link}; lab-backed dispatch may be admitted")
        else:
            cmd = (lab.get("commands") or {}).get("mark_clear")
            suffix = f"; publish clear with `{cmd}`" if cmd else ""
            buckets["action"].append(
                f"lab readiness {link}; {action}; lab-backed dispatch held{suffix}")
    resolver_pre = payload.get("resolver_preflight") or {}
    preflight_launch_blocker = _current_preflight_launch_blocker(preflight, resolver_pre)
    resolver = payload.get("resolver") or {}
    latest_tick = resolver.get("selected") or resolver.get("latest") or {}
    if latest_tick:
        tick_desc = (
            f"{latest_tick.get('backend') or '?'} "
            f"#{latest_tick.get('target_issue') or '-'} "
            f"lane {latest_tick.get('lane') or '-'}")
        state = _resolve_tick_state(latest_tick)
        if not latest_tick.get("fresh"):
            buckets["expected"].append(
                f"last resolver tick stale ({tick_desc}, {state}, "
                f"age {_age_text(latest_tick.get('age_min'))})")
        elif state == "launch-ready" and not latest_tick.get("live"):
            if preflight_launch_blocker:
                buckets["expected"].append(
                    f"resolver launch held by current preflight "
                    f"{preflight_launch_blocker['verdict']} ({tick_desc}); "
                    f"{preflight_launch_blocker['action']}")
            else:
                cmd = latest_tick.get("live_command_text")
                suffix = f"; approve `{cmd}`" if cmd else "; live launch requires approval"
                buckets["action"].append(f"resolver dry-run launch-ready ({tick_desc}){suffix}")
        elif state in ("repair-in-flight", "repair-spawned"):
            buckets["auto-solving"].append(f"resolver {state} ({tick_desc})")
        elif state.startswith("held "):
            action = latest_tick.get("next_action") or "inspect-last-resolve-tick"
            buckets["action"].append(
                f"last resolver tick {state} ({tick_desc}); {action}")
    elif latest_tick.get("action") == "spawned":
        buckets["auto-solving"].append(f"resolver spawned worker ({tick_desc})")
    resolver_pre_line = _resolver_preflight_summary(resolver_pre)
    if resolver_pre_line:
        target_bucket = "expected"
        if resolver_pre.get("verdict") in ("REFUSE_INSPECT",):
            target_bucket = "action"
        buckets[target_bucket].append(resolver_pre_line)
    limiter = ((payload.get("dispatcher") or {}).get("limiter") or {})
    if limiter.get("primary") == "github_rate_limit":
        buckets["action"].append("GitHub rate limit is blocking the gh-backed status folds")

    return buckets


def _dispatch_health_slack_buckets(payload: dict[str, Any]) -> dict[str, list[str]]:
    tp = payload.get("throughput") or {}
    buckets = {"expected": [], "auto-solving": [], "action": []}

    workers = payload.get("workers") or {}
    if workers.get("silent_count"):
        nums = " ".join(f"#{s.get('issue')}" for s in (workers.get("silent") or [])[:5])
        buckets["auto-solving"].append(
            f"{workers.get('silent_count')} no-output worker(s) skipped by cooldown"
            + (f" ({nums})" if nums else "")
            + "; inspect only if the same issue repeats")
    active_worker_line = _active_worker_summary(payload.get("worker_lease_check") or {})
    if active_worker_line:
        buckets["auto-solving"].append(active_worker_line)

    low_yield = payload.get("low_yield") or {}
    for r in [x for x in (low_yield.get("lanes") or [])
              if x.get("verdict") == "LOW_YIELD"][:4]:
        buckets["action"].append(
            f"lane {r.get('lane')} low-yield: {r.get('turns')} turns / "
            f"{r.get('sessions')} session(s), 0 ancestry-closes; re-scope or exclude the lane")

    bh = payload.get("backend_health") or {}
    stub_by_product = {
        str(r.get("product") or ""): r
        for r in (bh.get("stub_rate") or [])
        if r.get("majority_stub")
    }
    dead_products: set[str] = set()
    for r in (bh.get("dead") or [])[:4]:
        product = str(r.get("product") or "backend")
        dead_products.add(product)
        stub = stub_by_product.get(product)
        why = ""
        if stub:
            why = f"; evidence {stub.get('stub')}/{stub.get('total')} recent logs are stubs"
        reprobe = r.get("reprobe_min")
        buckets["auto-solving"].append(
            f"{product} held dead; lane {r.get('abandoned_lane') or '?'} reallocated"
            + (f"; re-probe every {reprobe}m" if reprobe else "")
            + why)
    for product, r in stub_by_product.items():
        if product in dead_products:
            continue
        buckets["action"].append(
            f"{product} majority-stub ({r.get('stub')}/{r.get('total')} recent logs); inspect backend output")

    hh = payload.get("hook_health") or {}
    for r in [x for x in (hh.get("by_backend") or []) if x.get("all_sessions_unhooked")][:4]:
        buckets["action"].append(
            f"{r.get('product')} guard hooks unbound "
            f"({r.get('sessions_with_hook_failures')}/{r.get('sessions')} sessions, "
            f"{r.get('hook_failures')} failures); workers ran unhooked")

    if not tp.get("na") and tp.get("verdict") in ("BELOW_TARGET", "AUDIT_ERROR"):
        buckets["action"].append(
            f"throughput {tp.get('verdict')}: {_rate_str(tp.get('completed_rate_per_hour'))}/h "
            f"vs target {_rate_str(tp.get('target_per_hour'))}/h")

    rs = payload.get("run_status") or {}
    if rs.get("errors"):
        buckets["action"].append(f"dos status had {rs.get('errors')} digest error(s)")
    git = payload.get("git") or {}
    if git.get("merge_in_progress"):
        buckets["action"].append(
            "peer merge in progress (MERGE_HEAD present); "
            f"{git.get('next_action') or 'wait before starting worker edits'}")
    leases = payload.get("leases") or {}
    if leases.get("read_error"):
        buckets["action"].append(f"lease read unavailable: {leases.get('read_error')}")
    elif leases.get("active_count"):
        bits = _lease_summary_bits(leases, limit=2)
        buckets["expected"].append(
            f"{leases.get('active_count')} active lane lease(s), "
            f"{leases.get('blocking_count', 0)} blocking current candidates"
            + (f" ({'; '.join(bits)})" if bits else ""))
    wl = payload.get("worker_lease_check") or {}
    if wl.get("available") is False:
        buckets["action"].append(f"worker/lease cross-check unavailable: {wl.get('error')}")
    elif wl:
        op = wl.get("orphan_process_count") or 0
        ol = wl.get("orphan_lease_count") or 0
        dead = wl.get("dead_holder_count") or 0
        lane_clean = lane_lease_verdict_clean(wl)
        if op or ol or not lane_clean:
            buckets["action"].append(
                f"worker/lease mismatch: clean={wl.get('clean_count', 0)}, "
                f"orphan-process={op}, unmatched-live-lease={ol}, dead-holder={dead}")
        elif wl.get("clean_count"):
            buckets["expected"].append(
                f"worker/lease cross-check clean ({wl.get('clean_count')} matched)")
    # Fenced lanes are an ACTION item on their own, independent of the sidecar half.
    lane = wl.get("lane_leases") or {}
    if lane and not lane.get("available"):
        buckets["action"].append(
            "lane-lease holder liveness UNKNOWN (not clean): "
            f"{lane.get('read_error') or 'lane-lease set unreadable'}")
    elif _int(lane.get("dead_count"), 0):
        buckets["action"].append(
            f"{lane.get('dead_count')} of {lane.get('total')} lane lease(s) are TTL-EXPIRED "
            f"in the structural WAL fold ({_lane_lease_bits(lane.get('dead') or [], limit=3)}) — "
            f"{lane.get('next_action') or _LANE_LEASE_NEXT_ACTION}")
    elif lane.get("total"):
        buckets["expected"].append(
            f"lane-lease holders all live ({lane.get('live_count')} lease(s), dead-holder=0)")
    util = payload.get("utilization") or {}
    if util.get("schema"):
        state = str(util.get("state") or "")
        slots = util.get("worker_slots") or {}
        actions = ", ".join(util.get("next_actions") or []) or "inspect-dispatch-status"
        if state in ("HEADROOM_LAUNCH_READY", "HEADROOM_REPAIR_READY"):
            buckets["action"].append(
                f"utilization {state}: {slots.get('headroom')} free worker slot(s); {actions}")
        elif state in ("HEADROOM_HELD", "HEADROOM_STALE_PLAN", "HOST_BLOCKED",
                       "ACCOUNT_BLOCKED", "ACCOUNT_CAPPED", "EDIT_HELD"):
            buckets["action"].append(
                f"utilization {state}: {slots.get('headroom')} free worker slot(s); {actions}")

    return buckets


def _dispatch_slack_buckets(payload: dict[str, Any]) -> dict[str, list[str]]:
    buckets = _dispatch_gate_slack_buckets(payload)
    health = _dispatch_health_slack_buckets(payload)
    for name in buckets:
        buckets[name].extend(health[name])
    return buckets


def _dispatch_headline_state(payload: dict[str, Any]) -> str:
    buckets = _dispatch_slack_buckets(payload)
    if buckets["action"]:
        return "ACTION"
    if buckets["auto-solving"]:
        return "auto-solving"
    if buckets["expected"]:
        return "expected"
    if payload.get("ok"):
        return "healthy"
    return "ACTION"


def render_slack(payload: dict[str, Any]) -> str:
    r"""The COMPACT Slack body for the dispatch card — the signal-dense peer of
    ``render`` (which keeps the box-drawn rails for the terminal + committed doc).

    The boxed ``render`` is built for a monospace wall: ``╔═ ║ ╚═`` rails, a column
    label on every line (``workers   :``), and a ``╚═``-prefixed footer that restates
    every row again in prose. In a Slack channel — read on a phone, in mrkdwn, not a
    code fence — that chrome and that restated footer are pure noise: the reader
    re-reads the same fact twice and scans past box-drawing to reach a number.

    This renderer keeps the SIGNAL (every value an operator acts on) and drops the
    noise: ONE dense summary line carries capacity / account / backlog / closure /
    rate, then one targeted ``⚠``/``🔴`` line PER problem that needs an eye (a silent
    worker, a majority-stub or unhooked backend, a held backend, an uninstalled
    watchdog, a weekly cap). A healthy steady state collapses to the summary line plus
    a single ``✓`` — no restated footer, no rails, no fence. Pure given the payload."""
    tp = payload.get("throughput") or {}
    lines: list[str] = [
        "plane: scheduler/backlog, not session health",
    ]
    cap_line = _dispatch_capacity_line(payload)
    if cap_line:
        lines.append(cap_line)
    trend = _dispatch_trend_line(tp)
    if trend:
        lines.append(trend)

    buckets = _dispatch_slack_buckets(payload)
    for label in ("expected", "auto-solving", "action"):
        rows = buckets[label]
        if rows:
            lines.append(f"{label}: {_join_limited(rows)}")
    cd = payload.get("commit_drought") or {}
    if cd.get("droughty"):
        lines.append(
            f"🔴 drought: 0 fleet commits in {cd.get('hours')}h while armed — "
            f"loop shipping nothing")
    if not any(buckets.values()) and not cd.get("droughty") and payload.get("ok"):
        lines.append("healthy: nothing needs an operator")
    return "\n".join(lines) if lines else "(no dispatcher signal)"


def slack_text(payload: dict[str, Any]) -> str:
    """The Slack message body for a status card: a one-line headline (so the channel
    preview and notification carry the verdict) above the COMPACT, signal-dense card
    (``render_slack``). The boxed ``render`` stays the terminal / committed-doc surface;
    Slack gets mrkdwn, not a monospace box, so the channel/phone reader scans state,
    not chrome (see the fleet-slack signal scorecard in tools/fleet_slack_status.py)."""

    verdict = payload.get("verdict")
    state = _dispatch_headline_state(payload)
    headline = f"*dispatch scheduler:* `{verdict}` ({state})"
    return headline + "\n" + render_slack(payload)


def post_to_slack(payload: dict[str, Any], *, channel: str = "",
                  dry_run: bool = False, transport: Any | None = None) -> dict[str, Any]:
    """Post the rendered status card to Slack via tools/slack_post. Never raises — a
    missing poster or a Slack failure becomes a typed verdict the caller logs, exactly
    like the rest of this read-only fold. Channel/token resolve through slack_post
    (``$FAK_DISPATCH_CHANNEL`` / the shared scoreboard token) unless ``channel`` is set."""
    try:
        import slack_post  # sibling module in tools/
    except Exception as exc:  # noqa: BLE001
        return {"posted": False, "error": f"slack_post unavailable: {exc}", "skipped": None}
    return slack_post.send(slack_text(payload), channel=channel, dry_run=dry_run,
                           transport=transport, include_signal_noise=False)
