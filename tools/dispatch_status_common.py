#!/usr/bin/env python3
"""Shared constants and pure formatting/capacity helpers for dispatch_status modules.

Kept strictly leaf-level (zero imports from sibling ``dispatch_status_*`` modules)
so ``leases``, ``telemetry``, ``render``, and ``dispatch_status`` share one source
of truth for schemas, verdicts, time/JSON parsing, and capacity/readiness formatting
without circular imports.
"""
from __future__ import annotations

import calendar
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "fleet-dispatch-status/1"

# The native readiness schema written today is the general three-phase link-state
# record (internal/linkstate). The older lab-specific schema is still ACCEPTED on
# read for one rollover cycle — the uncommittable private bridge may keep emitting it
# until its local mirror updates — and coarsened onto a phase.
LINK_STATE_SCHEMA = "fak.link_state/v1"
LAB_READINESS_SCHEMA = "fak.lab_readiness/v1"  # LEGACY: accepted on read, never written

# Primary three-phase vocabulary. CLEAR is the ONLY phase that admits dispatch.
LINK_CLEAR = "CLEAR"
LINK_WAITING = "WAITING"
LINK_WORKING = "WORKING"
LINK_PHASES = {LINK_CLEAR, LINK_WAITING, LINK_WORKING}

# The closed `detail` sub-vocabulary (the demoted fine cause). Mirrors internal/linkstate.
DETAIL_READY = "ready"                       # CLEAR
DETAIL_JOB_IN_FLIGHT = "job-in-flight"       # WORKING
DETAIL_GATEWAY_DOWN = "gateway-unreachable"  # WAITING
DETAIL_AUTH_BLOCKED = "auth-or-channel-blocked"  # WAITING
DETAIL_PRIVATE_RECOVERY = "private-recovery"  # WAITING
DETAIL_INDETERMINATE = "indeterminate"       # WAITING
LINK_DETAILS = {
    DETAIL_READY, DETAIL_JOB_IN_FLIGHT, DETAIL_GATEWAY_DOWN,
    DETAIL_AUTH_BLOCKED, DETAIL_PRIVATE_RECOVERY, DETAIL_INDETERMINATE,
}

# Legacy five-state status vocabulary (accepted on read, coarsened to a phase).
LEGACY_LAB_STATUSES = {
    "READY_FOR_DEV_WORK",
    "WAIT_PRIVATE_RECOVERY",
    "GATEWAY_UNREACHABLE",
    "AUTH_OR_CHANNEL_BLOCKED",
    "INDETERMINATE",
}

_LINK_STATE_KEYS = {
    "schema", "subject", "checked_at", "phase", "detail",
    "next_action", "evidence", "admit_dispatch",
}
_LEGACY_LAB_KEYS = {
    "schema", "machine_class", "checked_at", "status",
    "next_action", "evidence", "admit_lab_dispatch",
}
LAB_READINESS_KEYS = _LINK_STATE_KEYS | _LEGACY_LAB_KEYS | {"commands"}
_GENERIC_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_LAB_MARK_CLEAR_COMMAND = "fak lab readiness --phase CLEAR --write-default --json"
_LAB_MARK_WAITING_COMMAND = "fak lab readiness --phase WAITING --write-default --json"
_LAB_MARK_WORKING_COMMAND = "fak lab readiness --phase WORKING --write-default --json"

WATCHDOG_TASK = "FleetIssueDispatch"
RUNS_DIRNAME = ".dispatch-runs"
_RESOLVE_LOG_RE = re.compile(r"resolve-(\d+)-(\d{8}-\d{6})\.log$")
_STUB_LOG_MAX_BYTES = 512

# Typed WAIT verdicts for the launch-admission gate: preflight admits, final gate defers/unknown.
VERDICT_LAUNCH_COOLDOWN = "LAUNCH_COOLDOWN"
VERDICT_ADMISSION_UNKNOWN = "LAUNCH_ADMISSION_UNKNOWN"

_NEXT_ACTION_BY_VERDICT = {
    "READY_TO_GROW": "launch one worker (`python tools/issue_resolve_dispatch.py --live`)",
    "AT_CAP": "hold: at the configured worker ceiling",
    "BLOCKED_ON_ACCOUNT": "hold: wait for an account to free (`fak accounts status`)",
    "BLOCKED_ON_SEAT": "hold: wait for a dispatch seat to free (`fak accounts status`)",
    "BLOCKED_ON_PREFLIGHT": ("clear the preflight refusal named in the reasons "
                             "(`python tools/dispatch_preflight.py --json`)"),
    "BLOCKED_ON_BIN_SKEW": ("rebuild/install a clean `fak` — the adjudicating binary "
                            "is an unreviewed build (or set "
                            "FAK_PREFLIGHT_ALLOW_BIN_SKEW=1 deliberately)"),
    "WEEKLY_CAPPED": "hold: the account is weekly-capped until its reset",
    "INSPECT": "inspect: a safety preflight could not run",
    "HOST_FLAGGED": "reap/inspect the flagged host process before growing",
}

_CREATE_NO_WINDOW = 0x08000000


def _win_creationflags() -> int:
    return _CREATE_NO_WINDOW if os.name == "nt" else 0


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _py() -> str:
    return sys.executable or "python"


def _last_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
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


def _int(v: Any, d: int | None = None) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return d


def _age_text(minutes: Any) -> str:
    if not isinstance(minutes, (int, float)):
        return "?"
    if minutes < 1:
        return "<1m"
    if minutes >= 60:
        hours = f"{minutes / 60:.1f}".rstrip("0").rstrip(".")
        return hours + "h"
    return f"{minutes:.1f}".rstrip("0").rstrip(".") + "m"


def _iso_epoch(ts: Any) -> float | None:
    """Epoch seconds for a UTC ISO stamp like ``2026-07-07T03:30:44Z`` (naive UTC).

    Progress rows are always written in ``...Z`` UTC; we tolerate a trailing
    fractional part or ``+00:00`` offset and return None on anything unparseable
    so a malformed row is skipped, not fatal.
    """
    if not ts:
        return None
    s = str(ts).strip().replace("Z", "")
    s = s.split("+")[0].split(".")[0]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return float(calendar.timegm(time.strptime(s, fmt)))
        except (ValueError, TypeError):
            continue
    return None


def _epoch_iso(ep: float | None) -> str | None:
    """Inverse of ``_iso_epoch`` — a naive-UTC ``...Z`` stamp for a window edge."""
    if ep is None:
        return None
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ep))
    except (ValueError, OSError):
        return None


def _default_detail(phase: str) -> str:
    if phase == LINK_CLEAR:
        return DETAIL_READY
    if phase == LINK_WORKING:
        return DETAIL_JOB_IN_FLIGHT
    return DETAIL_INDETERMINATE


def _lab_readiness_commands() -> dict[str, str]:
    return {
        "mark_clear": _LAB_MARK_CLEAR_COMMAND,
        "mark_waiting": _LAB_MARK_WAITING_COMMAND,
        "mark_working": _LAB_MARK_WORKING_COMMAND,
    }


def _lab_link_label(lab: dict[str, Any]) -> str:
    """Render a readiness record's phase as ``PHASE/detail`` for the card, slack, and
    reasons — e.g. ``CLEAR/ready`` or ``WAITING/private-recovery``. The detail is
    dropped when it is the phase's default (a bare ``WORKING`` reads cleaner)."""
    phase = str(lab.get("phase") or LINK_WAITING)
    detail = str(lab.get("detail") or "")
    if detail and detail != _default_detail(phase):
        return f"{phase}/{detail}"
    return phase


def _or_unknown(v: Any) -> Any:
    """A missing capacity read renders ``UNKNOWN``, never a literal ``None`` that an
    operator reads as a real zero (#4649)."""
    return "UNKNOWN" if v is None else v


def _slots_headroom_note(live: Any, cap: Any) -> str:
    """The honest headroom note for a live/cap worker-slot (or seat) pair (#4649)."""
    if not isinstance(live, int) or not isinstance(cap, int):
        return "headroom UNKNOWN"
    room = int(cap) - int(live)
    if room < 0:
        return f"{-room} over the {cap}-slot target"
    return f"headroom {room}"


def _workers_live_clause(d: dict[str, Any]) -> str:
    """``live/cap live (headroom …)`` for the dispatcher card, honest about a missing
    read and an over-subscribed pool (#4649)."""
    live, cap = d.get("live"), d.get("cap")
    if not isinstance(live, int) or not isinstance(cap, int):
        return "UNKNOWN (dispatcher read incomplete)"
    return f"{live}/{cap} live ({_slots_headroom_note(live, cap)})"


def _capacity_reconcile(resolver_pre: dict[str, Any], dispatcher: dict[str, Any]) -> str:
    """Reconcile the two capacity scopes the operator card shows (#4649)."""
    live, cap = dispatcher.get("live"), dispatcher.get("cap")
    if not isinstance(live, int) or not isinstance(cap, int):
        return ""
    host_room = int(cap) - int(live)
    verdict = str(resolver_pre.get("verdict") or "").upper()
    if not verdict:
        return ""
    if host_room > 0 and verdict.startswith("REFUSE_"):
        return (f"- **capacity reconcile**: {host_room} host worker-slot(s) free but the "
                f"resolver target refuses (`{verdict}`) — a launch is gated by the "
                f"resolver, not host slots")
    if host_room <= 0 and verdict == "SPAWN_OK":
        state = "at" if host_room == 0 else f"{-host_room} over"
        return (f"- **capacity reconcile**: resolver target is ready (`SPAWN_OK`) but host "
                f"slots are {state} the {cap}-slot cap — a launch is gated by host slots, "
                f"not the resolver")
    return ""


def _resolver_preflight_summary(pre: dict[str, Any]) -> str:
    if not pre.get("schema"):
        err = pre.get("_error")
        return f"resolver product preflight unavailable: {err}" if err else ""
    backend = pre.get("_backend") or pre.get("product") or "?"
    seat = pre.get("seat") or {}
    live, cap = pre.get("live"), pre.get("cap")
    bits = [
        f"{backend}",
        str(pre.get("verdict") or "UNKNOWN"),
        f"live={_or_unknown(live)}/{_or_unknown(cap)}",
        _slots_headroom_note(live, cap),
        f"seats free={_or_unknown(seat.get('free'))} leased={_or_unknown(seat.get('leased'))}",
    ]
    unattributed = _int(seat.get("unattributed_live"), 0) or 0
    if unattributed:
        bits.append(f"unattributed_live={unattributed}")
    os_workers = pre.get("os_worker_procs")
    if os_workers is not None:
        bits.append(f"os_workers={os_workers}")
    return "selected resolver preflight: " + " ".join(bits)


def _current_preflight_launch_blocker(pre_verdict: str | None,
                                      resolver_preflight: dict[str, Any] | None
                                      ) -> dict[str, str]:
    """Return the current preflight refusal that makes a launch-ready tick stale."""
    resolver_preflight = resolver_preflight or {}
    checks = [
        (str(resolver_preflight.get("verdict") or ""),
         str(resolver_preflight.get("_backend") or resolver_preflight.get("product") or "resolver")),
        (str(pre_verdict or ""), "dispatcher"),
    ]
    table = {
        "REFUSE_NO_SEAT": ("NO_FREE_SEAT", "seat", "wait-for-free-account",
                           "HEADROOM_HELD"),
        "REFUSE_AT_CAP": ("AT_CAP", "host", "wait-for-worker-exit",
                          "SATURATED"),
        "REFUSE_NO_ACCOUNT": ("NO_FREE_ACCOUNT", "account", "wait-for-free-account",
                              "ACCOUNT_BLOCKED"),
        "REFUSE_INSPECT": ("PREFLIGHT_INSPECT", "host", "inspect-dispatch-preflight",
                           "INSPECT"),
    }
    for verdict, source in checks:
        if verdict in table:
            code, scope, action, state = table[verdict]
            return {
                "verdict": verdict,
                "code": code,
                "scope": scope,
                "action": action,
                "state": state,
                "source": source,
            }
    return {}


def _next_action(verdict: str, admission: dict[str, Any] | None,
                 merge: dict[str, Any], reasons: list[str]) -> str:
    """The one next move, derived from the SAME verdict the card publishes."""
    admission = admission or {}
    if verdict == "MERGE_IN_PROGRESS":
        return str(merge.get("next_action")
                   or "wait for MERGE_HEAD to clear before starting worker edits")
    if verdict == VERDICT_LAUNCH_COOLDOWN:
        return (f"wait for launch admission: {admission.get('reason') or 'DEFER'} "
                f"until {admission.get('retry_after') or '?'} "
                f"(`python tools/launch_admission.py admit --account "
                f"{admission.get('selected') or '<tag>'}`)")
    if verdict == VERDICT_ADMISSION_UNKNOWN:
        return ("re-run the launch admission gate by hand: "
                "`python tools/launch_admission.py admit --account "
                f"{admission.get('selected') or '<tag>'}`")
    known = _NEXT_ACTION_BY_VERDICT.get(verdict)
    if known:
        return known
    return str((reasons or [""])[0])


def _github_rate_limit_error(*docs: dict[str, Any]) -> str:
    for doc in docs:
        err = str((doc or {}).get("_error") or "")
        if "rate limit" in err.lower() or "secondary rate" in err.lower():
            return err
    return ""


def _dispatch_limiter(pre: dict[str, Any], backlog: dict[str, Any],
                      closure: dict[str, Any], leases: dict[str, Any]) -> dict[str, Any]:
    base = dict(pre.get("capacity_limiter") or {})
    raw = dict(base.get("raw") or {})
    raw.setdefault("cap", pre.get("cap"))
    raw.setdefault("live", pre.get("live"))
    raw.setdefault("headroom", (pre.get("cap") - pre.get("live"))
                   if isinstance(pre.get("cap"), int) and isinstance(pre.get("live"), int)
                   else None)
    raw.setdefault("max_workers", pre.get("max_workers"))
    raw.setdefault("host_cap", pre.get("host_cap"))
    seat = pre.get("seat") or {}
    raw.setdefault("seat_total", seat.get("total"))
    raw.setdefault("seat_free", seat.get("free"))
    raw.setdefault("seat_leased", seat.get("leased"))
    raw["lane_leases_active"] = leases.get("active_count")
    raw["lane_leases_blocking"] = leases.get("blocking_count")

    gh_err = _github_rate_limit_error(backlog, closure)
    if gh_err:
        raw["github_error"] = gh_err
        return {"primary": "github_rate_limit", "term": "github_error", "raw": raw}

    blocking = _int(leases.get("blocking_count"), 0) or 0
    if blocking:
        return {"primary": "leases", "term": "lane_leases_blocking", "raw": raw}

    if base:
        base["raw"] = raw
        return base
    return {"primary": "unknown", "term": "unknown", "raw": raw}


def _dispatch_limiter_terms(limiter: dict[str, Any]) -> str:
    raw = limiter.get("raw") or {}
    parts = [
        f"cap={raw.get('cap')}",
        f"live={raw.get('live')}",
        f"headroom={raw.get('headroom')}",
        f"max={raw.get('max_workers')}",
        f"target={raw.get('dos_target')}",
        f"host_cap={raw.get('host_cap')}",
        f"host_binding={raw.get('host_binding')}",
        f"seats={raw.get('seat_total')}",
        f"free={raw.get('seat_free')}",
        f"leased={raw.get('seat_leased')}",
        f"lane_leases={raw.get('lane_leases_blocking')}/{raw.get('lane_leases_active')}",
    ]
    if raw.get("github_error"):
        parts.append("github_error=rate_limit")
    return " ".join(parts)
