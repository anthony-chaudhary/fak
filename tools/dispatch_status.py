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
import calendar
import concurrent.futures
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

_CREATE_NO_WINDOW = 0x08000000


def _win_creationflags() -> int:
    return _CREATE_NO_WINDOW if os.name == "nt" else 0


sys.path.insert(0, str(Path(__file__).resolve().parent))
import dispatch_preflight  # noqa: E402  (pid-sidecar identity probe)
# The low-yield lane fold (#2062) now lives in a neutral shared module so the picker
# (issue_resolve_dispatch) can ACT on the same evidence this card REPORTS, without the
# dispatch_status<->issue_resolve_dispatch import cycle. Re-imported here for back-compat.
from lane_yield import (  # noqa: E402  (shared low-yield fold, #2062)
    count_lane_ancestry_closes,
    low_yield_lanes,
    # Re-exported for back-compat so dispatch_status.<name> keeps resolving
    # (redundant alias marks the intentional re-export; #2062).
    _log_turn_count as _log_turn_count,
    _LOW_YIELD_SCHEMA as _LOW_YIELD_SCHEMA,
    _LOW_YIELD_TURNS_FLOOR as _LOW_YIELD_TURNS_FLOOR,
    _LOW_YIELD_LOOKBACK_MIN,
)

SCHEMA = "fleet-dispatch-status/1"
# The native readiness schema written today is the general three-phase link-state
# record (internal/linkstate). The older lab-specific schema is still ACCEPTED on
# read for one rollover cycle — the uncommittable private bridge may keep emitting it
# until its local mirror updates — and coarsened onto a phase (see _coarsen_legacy_status).
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

# The decode superset: native link-state keys PLUS legacy lab keys, so a record in
# EITHER schema passes the unknown-field scrub while a foreign/private field is refused.
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
# The guarded always-on tick (tools/register_issue_dispatch.ps1). The older
# FleetDOSDispatchWatchdog keeps the un-gated kernel supervisor alive; this card
# tracks the DoS-safe issue dispatcher, so it reports the guarded task.
WATCHDOG_TASK = "FleetIssueDispatch"

RUNS_DIRNAME = ".dispatch-runs"
# Per-session fak-guard decision journals (one file per guarded worker), written by
# the dispatch worker's guard_wrap (dispatch_worker.py / cmd/dispatchworker). Each
# non-empty JSONL line is one kernel decision (internal/journal.Row).
GUARD_AUDIT_DIRNAME = "guard-audit"
# The decision-row kinds the journal records (internal/journal.rowFromEvent). DENY +
# RESULT_DENY are the refusals; QUARANTINE is a poisoned-result hold.
_GUARD_DENY_KINDS = ("DENY", "RESULT_DENY")
_GUARD_QUARANTINE_KIND = "QUARANTINE"
_GUARD_RECENT_LOOKBACK_MIN = 90
_GUARD_LIVELOCK_THRESHOLD = 3
_GUARD_LIVELOCK_MIN_COUNT = 10
_GUARD_LIVELOCK_LIMIT = 10
# resolve-<N>-<stamp>.log written by issue_resolve_dispatch.spawn_issue_worker.
_RESOLVE_LOG_RE = re.compile(r"resolve-(\d+)-(\d{8}-\d{6})\.log$")
_LEASEREF_PREFIX = "refs/fak/locks/"
_NOOP_BANNER_RE = re.compile(r"(?i)>\s*build\s*[·:]")
# The real-turn byte floor: a log at or below this carried no productive turn — it
# is a 0-byte spawn or a banner-only stub (e.g. a detached opencode worker that
# prints `> build · <model>` and exits). Mirrors the canonical
# ``issue_resolve_dispatch._STUB_LOG_MAX_BYTES`` (a drift test pins them equal) so a
# banner-only no-op counts as "produced nothing", not as output. See #1276.
_STUB_LOG_MAX_BYTES = 512
_BACKEND_STUB_LOOKBACK_MIN = 90
_RUN_STATUS_LOOKBACK_MIN = 180
_RID_RE = re.compile(r"^RID-[A-Z0-9]+$")
_RUN_STATUS_START_KINDS = {"start"}
_RUN_STATUS_TERMINAL_KINDS = {"end"}
_RUN_STATUS_TERMINAL_STATUSES = {
    "claimed_done",
    "witnessed_done",
    "failed",
    "refused",
    "cancelled",
}
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

# Ships-per-worker attribution (#2065). A dispatched agent can carry a best-effort
# `(fak-worker <id>)` commit trailer (sourced from the FLEET_WORKER_ID env stamp
# issue_dispatch.worker_env sets). This fold reads those trailers back out of git so a
# worker's witnessed ships become countable on the shared trunk — cumulative-shipped is
# otherwise unrecoverable (the guard-audit journal keeps only an args digest). The
# trailer is agent-emitted BEST-EFFORT, so the count is an attribution AID, not a
# witness: it never flips the card's `ok` and nothing is gated on it.
_SHIPS_PER_WORKER_SCHEMA = "fleet-ships-per-worker/1"
_SHIPS_PER_WORKER_GREP = "(fak-worker "
_SHIPS_PER_WORKER_LOOKBACK_MIN = 24 * 60
# A commit matched by the grep whose trailer id we can still parse; the group is the id.
_FAK_WORKER_TRAILER_RE = re.compile(r"\(fak-worker\s+([^)]+)\)")


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _py() -> str:
    return sys.executable or "python"


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


def _default_detail(phase: str) -> str:
    if phase == LINK_CLEAR:
        return DETAIL_READY
    if phase == LINK_WORKING:
        return DETAIL_JOB_IN_FLIGHT
    return DETAIL_INDETERMINATE


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


def _or_unknown(v: Any) -> Any:
    """A missing capacity read renders ``UNKNOWN``, never a literal ``None`` that an
    operator reads as a real zero (#4649)."""
    return "UNKNOWN" if v is None else v


def _slots_headroom_note(live: Any, cap: Any) -> str:
    """The honest headroom note for a live/cap worker-slot (or seat) pair (#4649).

    Spare capacity keeps the familiar ``headroom N``; an over-subscribed pool
    (live > cap) reads ``N over the M-slot target`` instead of a bare negative headroom
    an operator misreads as spare capacity; an unread pair is ``headroom UNKNOWN``, never
    a literal ``None``. The headroom is recomputed from live/cap here so an upstream
    ``max(0, …)`` clamp cannot mask an overshoot at the card. Mirrors the Slack roll-up's
    ``_worker_slot_phrase`` so the card and the roll-up agree."""
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
    """Reconcile the two capacity scopes the operator card shows (#4649).

    The resolver-target pool (free seats for the selected product) and the host
    worker-slot pool are different capacities, so the card can show host headroom right
    next to a resolver refusal. When the two disagree, name which scope actually gates a
    launch — otherwise two capacity numbers read as a contradiction. Returns '' when the
    scopes agree or either is unread."""
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


def _utilization_blocker(code: str, scope: str, next_action: str,
                         detail: str = "") -> dict[str, Any]:
    out = {"code": code, "scope": scope, "next_action": next_action}
    if detail:
        out["detail"] = detail
    return out


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


def _string_list(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v]
    return []


def _normalize_tree(t: str) -> str:
    t = str(t or "").strip().replace("\\", "/")
    if t.startswith("./"):
        t = t[2:]
    t = t.rstrip("/")
    for suffix in ("/**", "/*"):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
    return t.rstrip("/")


def _clean_tree(tree: Any) -> list[str]:
    out: list[str] = []
    for t in _string_list(tree):
        n = _normalize_tree(t)
        if n:
            out.append(n)
    return out


def _tree_overlap_one(a: str, b: str) -> bool:
    a, b = _normalize_tree(a), _normalize_tree(b)
    if not a or not b:
        return True
    if a in ("**", "**/*") or b in ("**", "**/*"):
        return True
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def trees_overlap(a: Any, b: Any) -> bool:
    ta, tb = _clean_tree(a), _clean_tree(b)
    if not ta or not tb:
        return True
    return any(_tree_overlap_one(x, y) for x in ta for y in tb)


def _lease_active_unix(rec: dict[str, Any]) -> int | None:
    acquired = _int(rec.get("acquired_unix"))
    renewed = _int(rec.get("renewed_unix"))
    if acquired is None and renewed is None:
        return None
    if acquired is None:
        return renewed
    if renewed is None:
        return acquired
    return max(acquired, renewed)


def _lease_expired(rec: dict[str, Any], now_ts: float) -> bool:
    ttl = _int(rec.get("ttl_seconds"), 0) or 0
    if ttl <= 0:
        return False
    active = _lease_active_unix(rec)
    if active is None:
        return False
    return now_ts >= active + ttl


def _session_expired(desc: dict[str, Any], now_ts: float) -> bool:
    ttl = _int(desc.get("ttl_seconds"), 0) or 0
    if ttl <= 0:
        return False
    updated = _int(desc.get("updated_at"))
    if updated is None:
        return False
    return now_ts >= updated + ttl


def _lease_liveness(rec: dict[str, Any], sessions: dict[str, dict[str, Any]],
                    now_ts: float) -> tuple[str, bool, str]:
    session_id = str(rec.get("session_id") or "").strip()
    if not session_id:
        return ("peer-unknown", False,
                "lease carries no session_id; absence is not proof of death")
    desc = sessions.get(session_id)
    if not isinstance(desc, dict):
        return ("peer-unknown", False,
                f"no session descriptor for session-{session_id}; absence is not proof of death")
    state = str(desc.get("pcb_state") or "").strip()
    updated = _int(desc.get("updated_at"))
    ttl = _int(desc.get("ttl_seconds"), 0) or 0
    if state.upper() == "STOPPED":
        return ("peer-dead", True,
                f"session {session_id} published STOPPED at {updated}")
    if _session_expired(desc, now_ts):
        return ("peer-dead", True,
                f"session {session_id} heartbeat lapsed: now >= {updated}+{ttl}")
    return ("peer-live", False,
            f"session {session_id} heartbeating state={state or '?'} updated_at={updated} ttl={ttl}")


def _lease_lane(lease_id: str) -> str:
    lease_id = str(lease_id or "").strip()
    if not lease_id.startswith("resolve-"):
        return lease_id
    lane = lease_id[len("resolve-"):]
    if re.search(r"-\d+$", lane):
        lane = re.sub(r"-\d+$", "", lane)
    return lane or lease_id


def _backlog_candidates(backlog: dict[str, Any]) -> list[dict[str, Any]]:
    lanes = (backlog.get("lanes") or {}) if isinstance(backlog, dict) else {}
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    for row in backlog.get("issues") or []:
        if not isinstance(row, dict):
            continue
        lane = str(row.get("lane") or "")
        issue = _int(row.get("number"))
        if not lane or issue is None:
            continue
        grp = lanes.get(lane) or {}
        cand = {
            "issue": issue,
            "lane": lane,
            "confidence": row.get("confidence"),
            "tree": _string_list(grp.get("tree")),
        }
        out.append(cand)
        seen.add((issue, lane))

    for lane, grp_any in lanes.items():
        if not isinstance(grp_any, dict):
            continue
        lane_s = str(lane)
        for issue_any in grp_any.get("issues") or []:
            issue = _int(issue_any)
            if issue is None or (issue, lane_s) in seen:
                continue
            out.append({
                "issue": issue,
                "lane": lane_s,
                "confidence": None,
                "tree": _string_list(grp_any.get("tree")),
            })
            seen.add((issue, lane_s))
    return out


def summarize_leases(records: list[dict[str, Any]], backlog: dict[str, Any],
                     *, sessions: dict[str, dict[str, Any]] | None = None,
                     now_ts: float | None = None) -> dict[str, Any]:
    """Classify refs/fak/locks records for the status card.

    Active leases block a candidate only when their tree overlaps a currently
    routed issue's lane tree. Expired records stay visible as reapable residue,
    but never block a candidate.
    """
    now_ts = time.time() if now_ts is None else now_ts
    sessions = sessions or {}
    backlog = backlog if isinstance(backlog, dict) else {}
    lanes = backlog.get("lanes") or {}
    candidate_source_available = "_skipped" not in backlog and not ("_error" in backlog and not lanes)
    candidates = _backlog_candidates(backlog)
    active: list[dict[str, Any]] = []
    expired: list[dict[str, Any]] = []

    for rec in records:
        if not isinstance(rec, dict):
            continue
        lease_id = str(rec.get("id") or "").strip()
        if not lease_id:
            continue
        tree = _string_list(rec.get("tree_globs"))
        active_unix = _lease_active_unix(rec)
        age_seconds = max(0, int(now_ts - active_unix)) if active_unix is not None else None
        ttl = _int(rec.get("ttl_seconds"), 0) or 0
        expires_in = None
        if ttl > 0 and active_unix is not None:
            expires_in = int(active_unix + ttl - now_ts)
        session_id = str(rec.get("session_id") or "").strip()
        liveness, reclaimable, liveness_evidence = _lease_liveness(rec, sessions, now_ts)
        row = {
            "id": lease_id,
            "lane": _lease_lane(lease_id),
            "holder": rec.get("holder"),
            "session_id": session_id or None,
            "tree": tree,
            "age_seconds": age_seconds,
            "age_min": round(age_seconds / 60, 1) if age_seconds is not None else None,
            "ttl_seconds": ttl,
            "expires_in_seconds": expires_in,
            "generation": rec.get("generation"),
            "liveness": liveness,
            "reclaimable": reclaimable,
            "liveness_evidence": liveness_evidence,
        }
        if _lease_expired(rec, now_ts):
            row["status"] = "EXPIRED"
            row["blocks_candidate"] = False
            row["blocking_candidates"] = []
            expired.append(row)
            continue
        row["status"] = "LIVE"
        if candidate_source_available:
            blockers = [
                c for c in candidates
                if trees_overlap(tree, c.get("tree"))
            ]
            row["blocks_candidate"] = bool(blockers)
            row["blocking_candidates"] = blockers[:8]
        else:
            row["blocks_candidate"] = None
            row["blocking_candidates"] = []
        active.append(row)

    active.sort(key=lambda r: (not bool(r.get("blocks_candidate")), str(r.get("lane") or ""), str(r.get("id") or "")))
    expired.sort(key=lambda r: str(r.get("id") or ""))
    blocking_count = sum(1 for r in active if r.get("blocks_candidate"))
    # #4324: split blocking leases by holder liveness so the PHANTOM share (a
    # blocking lease whose holder session is provably dead — `reclaimable`) is
    # measurable before/after a release-on-exit fix, not just the raw blocking
    # count. `blocking_live_count` covers both a live holder and an unknown
    # holder (no session_id -> not provably dead; never counted as phantom).
    blocking_stranded_count = sum(
        1 for r in active if r.get("blocks_candidate") and r.get("reclaimable"))
    return {
        "source": "refs/fak/locks",
        "candidate_source_available": candidate_source_available,
        "candidate_count": len(candidates),
        "active_count": len(active),
        "expired_count": len(expired),
        "blocking_count": blocking_count,
        "blocking_stranded_count": blocking_stranded_count,
        "blocking_live_count": blocking_count - blocking_stranded_count,
        "active": active,
        "expired": expired[:8],
    }


def read_leaseref_records(root: Path) -> tuple[list[dict[str, Any]], str | None]:
    records, _sessions, err = read_leaseref_records_and_sessions(root)
    return records, err


def read_leaseref_records_and_sessions(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str | None]:
    try:
        proc = subprocess.run(
            ["git", "for-each-ref", "--format=%(refname)", _LEASEREF_PREFIX],
            cwd=root, capture_output=True, text=True, timeout=10,
            creationflags=_win_creationflags())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], {}, str(exc)
    if proc.returncode != 0:
        return [], {}, (proc.stderr or proc.stdout or "git for-each-ref failed").strip()[-500:]

    refs = [line.strip() for line in (proc.stdout or "").splitlines()
            if line.strip().startswith(_LEASEREF_PREFIX)]

    records: list[dict[str, Any]] = []
    sessions: dict[str, dict[str, Any]] = {}
    skipped = 0

    if refs:
        # Resolve every lease ref's blob in ONE `git cat-file --batch` process
        # rather than one `git cat-file blob <ref>` spawn per ref. A busy host
        # accumulates thousands of refs/fak/locks/* refs (~8k observed); the
        # per-ref-spawn form never returns inside the FleetDispatchStatusDoc
        # 10-min task limit, so --fast broke its "pure-local, sub-5s" contract
        # and the status doc silently went stale. The batch stream emits one
        # record per input ref, in order, so we zip results back positionally.
        try:
            batch = subprocess.run(
                ["git", "cat-file", "--batch"],
                cwd=root, input="\n".join(refs).encode(),
                capture_output=True, timeout=60,
                creationflags=_win_creationflags())
        except (OSError, subprocess.TimeoutExpired) as exc:
            return [], {}, str(exc)
        data = batch.stdout or b""
        i, n = 0, len(data)
        for ref in refs:
            name = ref[len(_LEASEREF_PREFIX):]
            if i >= n:
                skipped += 1
                continue
            nl = data.find(b"\n", i)
            if nl < 0:
                skipped += 1
                break
            header = data[i:nl].decode("utf-8", "replace").split(" ")
            i = nl + 1
            if len(header) != 3:
                # "<ref> missing" / "<ref> ambiguous" — no payload follows.
                skipped += 1
                continue
            try:
                size = int(header[2])
            except ValueError:
                skipped += 1
                continue
            content = data[i:i + size]
            i += size + 1  # always consume payload + trailing newline
            if header[1] != "blob":
                skipped += 1
                continue
            try:
                rec = json.loads(content.decode("utf-8", "replace") or "{}")
            except ValueError:
                skipped += 1
                continue
            if isinstance(rec, dict):
                if name.startswith("session-"):
                    rec.setdefault("id", name[len("session-"):])
                    sessions[str(rec.get("id") or name[len("session-"):])] = rec
                else:
                    rec.setdefault("id", name)
                    records.append(rec)
            else:
                skipped += 1

    if skipped:
        for rec in records:
            rec.setdefault("_skipped_records", skipped)
    return records, sessions, None


def read_lease_state(root: Path, backlog: dict[str, Any],
                     *, now_ts: float | None = None) -> dict[str, Any]:
    records, sessions, err = read_leaseref_records_and_sessions(root)
    if err:
        return {
            "source": "refs/fak/locks",
            "read_error": err,
            "candidate_source_available": False,
            "candidate_count": 0,
            "active_count": 0,
            "expired_count": 0,
            "blocking_count": 0,
            "active": [],
            "expired": [],
        }
    state = summarize_leases(records, backlog, sessions=sessions, now_ts=now_ts)
    skipped = max((_int(r.get("_skipped_records"), 0) or 0) for r in records) if records else 0
    if skipped:
        state["skipped_records"] = skipped
    return state


def _dispatch_lease_token(s: str) -> str:
    s = str(s or "").strip()
    if not s:
        return "unknown"
    out: list[str] = []
    for ch in s:
        if ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("0" <= ch <= "9") or ch in "-_.":
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-.") or "unknown"


def _default_lease_id_for_lane(lane: str) -> str:
    return "resolve-" + _dispatch_lease_token(lane)


def _spawn_lane(log: Path) -> str:
    try:
        first = log.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except (OSError, IndexError):
        return ""
    for field in first.split():
        if field.startswith("lane="):
            return field[len("lane="):]
    return ""


def _banner_noop(log: Path) -> bool:
    try:
        if log.stat().st_size > _STUB_LOG_MAX_BYTES:
            return False
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return bool(_NOOP_BANNER_RE.search(text))


def _read_worker_lease_id(stem: Path, lane: str) -> str:
    try:
        value = (stem.with_suffix(".lease-id")).read_text(encoding="utf-8").strip()
        if value:
            return value
    except OSError:
        pass
    return _default_lease_id_for_lane(lane) if lane else ""


def _read_worker_tree(stem: Path) -> list[str]:
    try:
        obj = json.loads(stem.with_suffix(".lease-tree.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return _string_list(obj)


def _read_worker_text_sidecar(stem: Path, suffix: str) -> str:
    try:
        return stem.with_suffix(suffix).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _read_worker_account(stem: Path) -> dict[str, Any]:
    try:
        obj = json.loads(stem.with_suffix(".account").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return obj if isinstance(obj, dict) else {}


def scan_live_dispatch_workers(
    runs_dir: Path,
    *,
    alive: set[int] | None = None,
    probe: Any | None = None,
) -> dict[str, Any]:
    if not runs_dir.is_dir():
        return {"available": True, "workers": []}
    if alive is None and probe is None:
        try:
            import psutil  # type: ignore

            alive = {p.pid for p in psutil.process_iter()}
        except ImportError:
            return {"available": False, "workers": [], "error": "psutil unavailable"}

    workers: list[dict[str, Any]] = []
    for log in runs_dir.glob("resolve-*.log"):
        m = _RESOLVE_LOG_RE.search(log.name)
        if not m:
            continue
        stem = log.with_suffix("")
        pid_file = stem.with_suffix(".pid")
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if not dispatch_preflight.resolve_sidecar_pid_is_live(
            pid_file, alive=alive, probe=probe):
            continue
        if _banner_noop(log):
            continue
        lane = _spawn_lane(log)
        lease_id = _read_worker_lease_id(stem, lane)
        workers.append({
            "worker": stem.name,
            "issue": int(m.group(1)),
            "stamp": m.group(2),
            "pid": pid,
            "lane": lane,
            "backend": _read_worker_text_sidecar(stem, ".backend"),
            "account": _read_worker_account(stem),
            "lease_id": lease_id,
            "tree": _read_worker_tree(stem),
            "log": log.name,
        })
    workers.sort(key=lambda r: str(r.get("worker") or ""))
    return {"available": True, "workers": workers}


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

_LANE_LEASE_SCHEMA = "fleet-lane-lease-liveness/1"
# Windows FILETIME epoch (1601-01-01) -> Unix epoch (1970-01-01), in seconds.
_FILETIME_EPOCH_DELTA_S = 11644473600.0
# The plausible wall-clock window a decoded process start time must land in. Used
# to pick the right interpretation of `proc_starttime` (Windows FILETIME ticks vs
# unix ns/us/ms/s) without hard-coding the producer's platform.
_PROC_START_MIN_EPOCH_S = 946684800.0    # 2000-01-01
_PROC_START_MAX_EPOCH_S = 4102444800.0   # 2100-01-01
# How far a probed start time may sit from the recorded one and still be the SAME
# process. The Windows probe reads back the same FILETIME the kernel recorded, so
# this only absorbs float rounding; it is deliberately far tighter than any
# realistic pid-reuse interval, because the whole point is to catch reuse.
_PROC_START_TOLERANCE_S = 2.0
# The kernel's own staleness constants, mirrored so the card and the admission fold
# cannot drift apart. `dos.lane_lease._DEFAULT_LIVE_TTL_MINUTES` /
# `_LIVE_TTL_GRACE_MINUTES`.
_LANE_LEASE_DEFAULT_TTL_MIN = 50.0
_LANE_LEASE_GRACE_MIN = 5.0
# The reap action this card USED to print was actively dangerous, and saying so is
# the point of the line (#5859):
#
#   * `dos lease-lane release` performs NO liveness check whatsoever — `cmd_lease_lane`
#     calls `lane_lease.release()` straight through, and that function evicts a live
#     holder exactly as happily as a dead one. Its owner filter
#     (`holder not in (owner, None) and owner != ""`) also short-circuits on an empty
#     string, so `--owner ""` matches ANY lease on the lane regardless of holder.
#     Running it across a 25-lane set would have destroyed live work.
#   * There is no sanctioned per-lane reap verb to point at instead. `lane_lease.adopt()`
#     (the ownership transfer) and the `OP_SCAVENGE` journal op both exist in the kernel
#     but neither has a CLI surface — `dos lease-lane` exposes only
#     acquire/release/heartbeat/spawn/live.
#   * And no reap is needed for the thing the operator actually cares about: the
#     ADMISSION view already self-heals. `live_leases(config, expire_dead=True)` — what
#     `pretool_sensor`, `decisions.py`, `dispatch_top` and `dos arbitrate` read — drops
#     these leases at read time without touching the WAL. A stale STRUCTURAL lease
#     therefore blocks exactly one consumer: `lane_lease.acquire()` at
#     `dos/lane_lease.py:453`, the single caller that deliberately folds without
#     dead-elision (so a serialized acquirer cannot double-book a fresh reservation).
#
# So the honest next action is "observe, do not reap", plus the one narrow symptom
# worth watching for.
_LANE_LEASE_NEXT_ACTION_STEPS = (
    "do NOT reap: `dos lease-lane release` runs NO liveness check (#5859) and its "
    "`--owner \"\"` matches any holder, so releasing across the lane set evicts live work",
    "there is no sanctioned per-lane reap verb to use instead — `lane_lease.adopt()` and "
    "the `OP_SCAVENGE` journal op have no CLI surface (`dos lease-lane` = "
    "acquire|release|heartbeat|spawn|live)",
    "none is needed: the admission fold `live_leases(config, expire_dead=True)` already "
    "elides these at read time for `pretool_sensor`/`decisions.py`/`dispatch_top`/"
    "`dos arbitrate`, without mutating the WAL",
    "a stale STRUCTURAL lease therefore blocks exactly one consumer — `lane_lease.acquire()` "
    "(dos/lane_lease.py:453) — so watch for a repeated acquire REFUSE on one of these lanes "
    "and escalate THAT lane to the operator; releasing blind is the larger risk",
)
_LANE_LEASE_NEXT_ACTION = " · ".join(_LANE_LEASE_NEXT_ACTION_STEPS)

# Holder states. `unknown` is a first-class verdict: anything we cannot PROVE is
# never folded into `live`, because a false "alive" is how this stayed hidden.
LANE_HOLDER_LIVE = "live"
LANE_HOLDER_DEAD = "dead"
LANE_HOLDER_UNKNOWN = "unknown"


def _proc_starttime_epoch(value: Any) -> float | None:
    """Decode a lane-lease record's ``proc_starttime`` to Unix epoch seconds.

    The DOS lane record stamps the holder's process start time in the host's
    native unit — on Windows (the platform that matters here) a FILETIME: 100ns
    ticks since 1601-01-01. Rather than hard-code one platform, each candidate
    scale is tried and the first that decodes into a plausible wall-clock window
    wins; the scales are far enough apart that only one can land in it. An
    undecodable stamp returns ``None``, which the classifier treats as "cannot
    prove either way" — never as proof of life.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    if v <= 0:
        return None
    for candidate in (
        v / 1e7 - _FILETIME_EPOCH_DELTA_S,  # Windows FILETIME (100ns ticks since 1601)
        v / 1e9,                            # unix nanoseconds
        v / 1e6,                            # unix microseconds
        v / 1e3,                            # unix milliseconds
        v,                                  # unix seconds
    ):
        if _PROC_START_MIN_EPOCH_S <= candidate <= _PROC_START_MAX_EPOCH_S:
            return candidate
    return None


def process_start_times() -> dict[int, float] | None:
    """``{pid: create_time_epoch_seconds}`` for every process on this host.

    Returns ``None`` when no liveness oracle is available (psutil absent). ``None``
    is NOT "nothing is running": it means the fold cannot judge, and the caller
    must report the whole set unavailable rather than either `live` or `dead`.
    Mirrors the optional-psutil convention `silent_workers` already uses, except
    that here the *absence* of an oracle must never render as clean.
    """
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    out: dict[int, float] = {}
    try:
        procs = psutil.process_iter(["pid", "create_time"])
    except Exception:  # noqa: BLE001 - probe seam: report no oracle, never crash the card
        return None
    for proc in procs:
        try:
            info = proc.info
            pid, created = info.get("pid"), info.get("create_time")
        except Exception:  # noqa: BLE001 - a process can exit mid-iteration
            continue
        if isinstance(pid, int) and isinstance(created, (int, float)):
            out[pid] = float(created)
    return out


def _lane_lease_ttl_min(rec: dict[str, Any]) -> float:
    """The lease's own declared ``ttl_minutes``, else the kernel's 50-minute backstop.

    Mirrors `dos.lane_lease._lease_is_dead`: a malformed/legacy ACQUIRE that declared
    no TTL still cannot be immortal.
    """
    ttl = rec.get("ttl_minutes")
    if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or ttl <= 0:
        return _LANE_LEASE_DEFAULT_TTL_MIN
    return float(ttl)


def probe_lane_lease_pid(
    rec: dict[str, Any],
    starts: dict[int, float] | None,
    *,
    host_id: str = "",
    tolerance: float = _PROC_START_TOLERANCE_S,
) -> tuple[bool | None, str]:
    """The OS-process rung ALONE: ``(alive, evidence)``, with ``alive`` THREE-valued.

    ``True`` only when the pid is running AND its start time matches the one the
    lease recorded. ``False`` only on positive evidence (the pid is gone, or the pid
    exists but was RECYCLED into a different process — a pid that merely *exists* is
    not a live holder). Everything else — no oracle, a foreign host, no usable pid,
    or a live pid with no recorded ``proc_starttime`` to check it against — is
    ``None``, "cannot tell", which is never proof of either.

    Deliberately SEPARATE from the verdict: on its own this is NOT evidence of
    death, because the pid a lane lease records is the ephemeral `dos lease-lane
    acquire` child that exits by design. `classify_lane_lease_holder` consumes it
    strictly as a corroborator of WAL staleness.
    """
    pid = _int(rec.get("pid"))
    rec_host = str(rec.get("host_id") or "").strip()
    if starts is None:
        return (None, "no process-liveness oracle on this host (psutil unavailable)")
    if rec_host and host_id and rec_host != host_id:
        return (None,
                f"holder recorded on host {rec_host}; this host cannot probe a remote pid")
    if pid is None or pid <= 0:
        return (None, "lease carries no usable pid")
    observed = starts.get(pid)
    if observed is None:
        return (False, f"pid {pid} is not a running process")
    recorded = _proc_starttime_epoch(rec.get("proc_starttime"))
    if recorded is None:
        return (None,
                f"pid {pid} exists but the lease records no readable proc_starttime; "
                "pid existence alone is not proof of life")
    if abs(observed - recorded) <= tolerance:
        return (True,
                f"pid {pid} running, start time matches the lease ({recorded:.0f})")
    return (False,
            f"pid {pid} was RECYCLED: the running process started {observed:.0f}, "
            f"the lease recorded {recorded:.0f}")


def classify_lane_lease_holder(
    rec: dict[str, Any],
    starts: dict[int, float] | None,
    *,
    host_id: str = "",
    tolerance: float = _PROC_START_TOLERANCE_S,
    now_ts: float | None = None,
) -> tuple[str, str]:
    """Judge ONE lane-lease record's holder. Returns ``(state, evidence)``.

    Mirrors the kernel's own live-set predicate (`dos.lane_lease._lease_is_dead`):
    **WAL heartbeat/TTL staleness is the PRIMARY evidence; the holder pid only
    corroborates.** In rung order —

      1. LIVE — the lease's newest WAL stamp (``heartbeat_at``, else ``acquired_at``)
         is inside the freshness grace. This holds *regardless of the pid*, and it is
         precisely the case the first cut got wrong: the recorded pid is the ephemeral
         `dos lease-lane acquire` child, so a fresh, healthy, actively-held lease
         normally has an ABSENT pid.
      2. LIVE — the recorded holder process is confidently up on this host with a
         matching start time. The strongest signal available, when it applies.
      3. UNKNOWN — no readable stamp at all, so staleness cannot be judged and the
         pid says nothing on its own.
      4. DEAD — past the lease's own ``ttl_minutes`` (or the kernel's backstop) plus
         the grace. TTL expiry is positive, WAL-carried death evidence, and it is what
         the admission fold `live_leases(expire_dead=True)` itself acts on.
      5. DEAD — an OBSERVED heartbeat gone quiet past the grace, corroborated by a
         confidently-dead pid: dead *and* silent, so reclaimable before the full TTL.
         Requires a real ``heartbeat_at``; a bare ``acquired_at`` is not a beat, and
         letting it stand in is how the pid rung came to fire for every lease on a
         fleet whose WAL contains zero HEARTBEAT ops.
      6. UNKNOWN — inside its TTL but not fresh. Unproven either way, never `live`.
    """
    now_ts = time.time() if now_ts is None else now_ts
    pid_alive, pid_note = probe_lane_lease_pid(
        rec, starts, host_id=host_id, tolerance=tolerance)

    beat_stamp = str(rec.get("heartbeat_at") or "").strip()
    stamp_epoch = _lease_acquired_epoch(rec)
    age_min = None if stamp_epoch is None else max(0.0, (now_ts - stamp_epoch) / 60.0)
    ttl_min = _lane_lease_ttl_min(rec)
    kind = "heartbeat" if beat_stamp else "acquire stamp"

    # (1) PRIMARY: WAL freshness outranks every process signal.
    if age_min is not None and age_min <= _LANE_LEASE_GRACE_MIN:
        return (LANE_HOLDER_LIVE,
                f"{kind} is {_age_text(age_min)} old — fresh inside the "
                f"{_LANE_LEASE_GRACE_MIN:.0f}-minute grace, so the lease is held "
                f"regardless of its ephemeral acquirer pid [{pid_note}]")

    # (2) The holder process itself is confidently up.
    if pid_alive is True:
        return (LANE_HOLDER_LIVE, pid_note)

    # (3) No credible stamp — the pid alone cannot decide it.
    if age_min is None:
        return (LANE_HOLDER_UNKNOWN,
                "lease carries no readable heartbeat/acquire stamp, so staleness "
                f"cannot be judged [{pid_note}]")

    # (4) TTL expiry — the kernel's primary death evidence.
    if age_min > ttl_min + _LANE_LEASE_GRACE_MIN:
        return (LANE_HOLDER_DEAD,
                f"TTL EXPIRED: no {kind} for {_age_text(age_min)}, past its "
                f"{ttl_min:.0f}-minute TTL + {_LANE_LEASE_GRACE_MIN:.0f}-minute grace "
                f"[{pid_note}]")

    # (5) A real beat that went quiet, corroborated by a confidently-dead pid.
    if beat_stamp and pid_alive is False:
        return (LANE_HOLDER_DEAD,
                f"heartbeat went quiet {_age_text(age_min)} ago and the holder process "
                f"is confirmed gone [{pid_note}]")

    # (6) Inside its TTL, not fresh, ephemeral acquirer gone — the normal shape.
    return (LANE_HOLDER_UNKNOWN,
            f"inside its {ttl_min:.0f}-minute TTL ({_age_text(age_min)} since the "
            f"{kind}) but not beaten within the grace; the recorded pid is the "
            f"ephemeral `dos lease-lane acquire` child, so its absence is not "
            f"evidence of death [{pid_note}]")


def _lease_acquired_epoch(rec: dict[str, Any]) -> float | None:
    """Epoch seconds for a lane record's acquire/heartbeat stamp (ISO-8601 or epoch)."""
    for key in ("heartbeat_at", "loop_ts", "acquired_at"):
        value = rec.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                import datetime as _dt

                return _dt.datetime.fromisoformat(
                    value.strip().replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
    return None


def summarize_lane_lease_holders(
    records: list[dict[str, Any]],
    *,
    starts: dict[int, float] | None = None,
    host_id: str = "",
    now_ts: float | None = None,
    read_error: str | None = None,
) -> dict[str, Any]:
    """Fold `dos lease-lane live` records into a holder-liveness verdict.

    PURE over (records, starts, host_id, now) so it is table-testable: the process
    probe and the `dos` call both live in the callers. ``available`` is False when
    the set could not be read OR no liveness oracle exists — in both cases the
    caller must refuse to render "clean", because an unread lease set is not a
    clean one.
    """
    now_ts = time.time() if now_ts is None else now_ts
    available = read_error is None and starts is not None
    rows: list[dict[str, Any]] = []
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        state, evidence = classify_lane_lease_holder(
            rec, starts, host_id=host_id, now_ts=now_ts)
        acquired = _lease_acquired_epoch(rec)
        age_min = round(max(0.0, (now_ts - acquired) / 60.0), 1) if acquired is not None else None
        rows.append({
            "lane": str(rec.get("lane") or ""),
            "holder": str(rec.get("holder") or ""),
            "host_id": str(rec.get("host_id") or ""),
            "mode": str(rec.get("mode") or ""),
            "pid": _int(rec.get("pid")),
            "acquired_at": rec.get("acquired_at"),
            "age_min": age_min,
            "tree": _string_list(rec.get("tree")),
            "holder_state": state,
            "holder_evidence": evidence,
        })
    # Dead first, then oldest first: the operator's reap order.
    rows.sort(key=lambda r: (
        {LANE_HOLDER_DEAD: 0, LANE_HOLDER_UNKNOWN: 1, LANE_HOLDER_LIVE: 2}.get(
            str(r.get("holder_state")), 3),
        -(r.get("age_min") or 0.0),
        str(r.get("lane") or ""),
    ))
    dead = [r for r in rows if r.get("holder_state") == LANE_HOLDER_DEAD]
    unknown = [r for r in rows if r.get("holder_state") == LANE_HOLDER_UNKNOWN]
    live = [r for r in rows if r.get("holder_state") == LANE_HOLDER_LIVE]
    out: dict[str, Any] = {
        "schema": _LANE_LEASE_SCHEMA,
        "source": "dos lease-lane live",
        "available": available,
        "total": len(rows),
        "dead_count": len(dead),
        "unknown_count": len(unknown),
        "live_count": len(live),
        "rows": rows,
        "dead": dead[:12],
        "next_action": _LANE_LEASE_NEXT_ACTION if dead else "",
    }
    if read_error:
        out["read_error"] = read_error
    elif starts is None:
        out["read_error"] = "no process-liveness oracle (psutil unavailable)"
    return out


def read_lane_leases(root: Path, *,
                     runner: Any | None = None) -> tuple[list[dict[str, Any]], str | None]:
    """Read the DOS kernel's live lane-lease set (`dos lease-lane live`).

    Note the deliberate absence of `--workspace`: the kernel resolves its WAL from
    the cwd, and an explicit path argument makes `dos lease-lane live` re-resolve
    to a different/empty `.dos/` and emit non-JSON — the same trap
    `tools/dos_fleet_lease.kernel_live` documents. `runner` is injectable so tests
    never shell a real `dos`.
    """
    cmd = ["dos", "lease-lane", "live"]
    if runner is not None:
        try:
            proc_out, err = runner(cmd, root)
        except Exception as exc:  # noqa: BLE001 - injected seam: report, never crash
            return [], str(exc)
        if err:
            return [], err
        text = proc_out or ""
    else:
        try:
            proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=30,
                                  creationflags=_win_creationflags())
        except (OSError, subprocess.TimeoutExpired) as exc:
            return [], str(exc)
        if proc.returncode != 0:
            return [], (proc.stderr or proc.stdout or "dos lease-lane live failed").strip()[-300:]
        text = proc.stdout or ""
    text = text.strip()
    if not text:
        return [], None
    try:
        data = json.loads(text)
    except ValueError:
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except ValueError:
                continue
            break
        else:
            return [], "dos lease-lane live emitted no parseable JSON"
    if not isinstance(data, list):
        return [], "dos lease-lane live did not emit a JSON array"
    return [r for r in data if isinstance(r, dict)], None


def lane_lease_liveness(root: Path, *, runner: Any | None = None,
                        starts: dict[int, float] | None = None,
                        host_id: str | None = None,
                        now_ts: float | None = None) -> dict[str, Any]:
    """Impure wrapper: read the lane-lease set + probe this host, then fold."""
    records, err = read_lane_leases(root, runner=runner)
    if starts is None and err is None:
        starts = process_start_times()
    if host_id is None:
        try:
            host_id = socket.gethostname()
        except OSError:
            host_id = ""
    return summarize_lane_lease_holders(
        records, starts=starts, host_id=host_id or "", now_ts=now_ts, read_error=err)


def _lane_lease_bits(rows: list[dict[str, Any]], *, limit: int = 4) -> str:
    """`lane(holder, pid N, 22.1d)` bits for the card/reason renders."""
    bits: list[str] = []
    for row in rows[:limit]:
        lane = row.get("lane") or "?"
        holder = row.get("holder") or "?"
        pid = row.get("pid")
        age = _age_text(row.get("age_min"))
        bits.append(f"{lane}({holder}, pid {pid if pid is not None else '?'}, {age})")
    if len(rows) > limit:
        bits.append(f"+{len(rows) - limit} more")
    return ", ".join(bits)


def _fold_lane_leases(out: dict[str, Any],
                      lane_leases: dict[str, Any] | None) -> dict[str, Any]:
    """Fold the lane-lease holder verdict into the worker/lease cross-check dict.

    Kept as a mutation of the SAME dict the card already threads (`worker_lease_check`)
    so every existing consumer — card, reasons, markdown, slack, JSON — sees the new
    `dead_holder_count` without a payload-schema change.
    """
    lane = lane_leases or {}
    if not lane:
        return out
    out["lane_leases"] = lane
    out["lane_lease_available"] = bool(lane.get("available"))
    out["lane_lease_count"] = _int(lane.get("total"), 0) or 0
    out["dead_holder_count"] = _int(lane.get("dead_count"), 0) or 0
    out["unknown_holder_count"] = _int(lane.get("unknown_count"), 0) or 0
    out["live_holder_count"] = _int(lane.get("live_count"), 0) or 0
    out["dead_holder"] = lane.get("dead") or []
    if lane.get("next_action"):
        out["lane_lease_next_action"] = lane.get("next_action")
    return out


def lane_lease_verdict_clean(worker_leases: dict[str, Any]) -> bool:
    """May this cross-check be described as CLEAN?

    False the moment a lane lease is held by a non-existent process, and false when
    the lane-lease set could not be read or judged at all — an unread lease set is
    not a clean one. Absent the fold entirely (a legacy caller), the historic
    behaviour is preserved.
    """
    lane = worker_leases.get("lane_leases")
    if not lane:
        return True
    if not lane.get("available"):
        return False
    return not (_int(lane.get("dead_count"), 0) or 0)


def cross_check_worker_leases(worker_state: dict[str, Any],
                              leases: dict[str, Any],
                              lane_leases: dict[str, Any] | None = None) -> dict[str, Any]:
    active_leases = {
        str(row.get("id") or ""): row
        for row in (leases.get("active") or [])
        if row.get("id")
    }
    if not worker_state.get("available", True):
        # The lane-lease verdict is folded in even here: an unreadable LOCAL worker
        # sidecar set says nothing about the kernel's lane leases, and a dead holder
        # must never be swallowed by an unrelated probe failure.
        return _fold_lane_leases({
            "available": False,
            "error": worker_state.get("error"),
            "clean_count": 0,
            "orphan_process_count": 0,
            "orphan_lease_count": 0,
            "clean": [],
            "orphan_process": [],
            "orphan_lease": [],
        }, lane_leases)

    clean: list[dict[str, Any]] = []
    orphan_process: list[dict[str, Any]] = []
    matched: set[str] = set()
    for worker in worker_state.get("workers") or []:
        lease_id = str(worker.get("lease_id") or "")
        lease = active_leases.get(lease_id)
        if lease:
            matched.add(lease_id)
            clean.append({"worker": worker, "lease": lease})
        else:
            orphan_process.append({
                "worker": worker,
                "reason": "missing active dispatch lease" if lease_id else "worker has no lease id",
            })

    orphan_lease = [
        {"lease": lease, "reason": "active lease has no local live worker sidecar"}
        for lease_id, lease in sorted(active_leases.items())
        if lease_id not in matched
    ]
    return _fold_lane_leases({
        "available": True,
        "clean_count": len(clean),
        "orphan_process_count": len(orphan_process),
        "orphan_lease_count": len(orphan_lease),
        "clean": clean,
        "orphan_process": orphan_process,
        "orphan_lease": orphan_lease,
    }, lane_leases)


def _active_worker_rows(worker_leases: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in worker_leases.get("clean") or []:
        worker = row.get("worker") or {}
        if worker:
            rows.append(worker)
    for row in worker_leases.get("orphan_process") or []:
        worker = row.get("worker") or {}
        if worker:
            rows.append(worker)
    return rows


def _active_worker_summary(worker_leases: dict[str, Any], *, limit: int = 4) -> str:
    workers = _active_worker_rows(worker_leases)
    if not workers:
        return ""
    bits: list[str] = []
    for worker in workers[:limit]:
        issue = worker.get("issue")
        issue_bit = f"#{issue}" if issue is not None else str(worker.get("worker") or "?")
        lane = worker.get("lane") or "-"
        backend = worker.get("backend") or "-"
        pid = worker.get("pid") or "-"
        lease = worker.get("lease_id") or "-"
        bits.append(f"{issue_bit} {lane}/{backend} pid={pid} lease={lease}")
    if len(workers) > limit:
        bits.append(f"+{len(workers) - limit} more")
    return "active resolver worker(s): " + "; ".join(bits)


def worker_lease_crosscheck(
    runs_dir: Path,
    leases: dict[str, Any],
    *,
    alive: set[int] | None = None,
    probe: Any | None = None,
    lane_leases: dict[str, Any] | None = None,
) -> dict[str, Any]:
    workers = scan_live_dispatch_workers(runs_dir, alive=alive, probe=probe)
    return cross_check_worker_leases(workers, leases, lane_leases)


def has_key_named(obj: Any, key: str) -> bool:
    if isinstance(obj, dict):
        return key in obj or any(has_key_named(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(has_key_named(v, key) for v in obj)
    return False


def run_ids_from_loop_ledger(
    ledger: Path,
    *,
    limit: int = 6,
    lookback_min: int | None = _RUN_STATUS_LOOKBACK_MIN,
    now_ns: int | None = None,
) -> list[str]:
    if limit <= 0 or not ledger.exists():
        return []
    if lookback_min is not None and lookback_min > 0 and now_ns is None:
        now_ns = time.time_ns()
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    runs: dict[str, dict[str, Any]] = {}
    for idx, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        loop_id = str(row.get("loop_id") or "")
        run_id = str(row.get("run_id") or "")
        if not loop_id.startswith("issue-resolve-") or not _RID_RE.fullmatch(run_id):
            continue
        state = runs.setdefault(run_id, {
            "loop_id": loop_id,
            "started": False,
            "last_index": idx,
            "last_kind": "",
            "last_status": "",
            "last_ts_unix_nano": None,
        })
        kind = str(row.get("kind") or "").strip().lower()
        status = str(row.get("status") or "").strip().lower()
        state["loop_id"] = loop_id
        state["last_index"] = idx
        state["last_kind"] = kind
        state["last_status"] = status
        ts = _int(row.get("ts_unix_nano"))
        if ts is not None:
            state["last_ts_unix_nano"] = ts
        if kind in _RUN_STATUS_START_KINDS or status == "running":
            state["started"] = True

    out: list[str] = []
    for run_id, state in sorted(
            runs.items(), key=lambda kv: int(kv[1].get("last_index") or 0),
            reverse=True):
        if not state.get("started"):
            continue
        if state.get("last_kind") in _RUN_STATUS_TERMINAL_KINDS:
            continue
        if state.get("last_status") in _RUN_STATUS_TERMINAL_STATUSES:
            continue
        ts = _int(state.get("last_ts_unix_nano"))
        if ts is not None and now_ns is not None and lookback_min is not None and lookback_min > 0:
            if now_ns - ts > lookback_min * 60 * 1_000_000_000:
                continue
        out.append(run_id)
        if len(out) >= limit:
            break
    return out


def dos_status_digest(root: Path, run_id: str) -> dict[str, Any]:
    if not _RID_RE.fullmatch(run_id):
        return {"run_id": run_id, "_error": "not a DOS RID"}
    try:
        proc = subprocess.run(
            ["dos", "status", "--workspace", str(root), "--json", run_id],
            cwd=root, capture_output=True, text=True, timeout=45,
            creationflags=_win_creationflags())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"run_id": run_id, "_error": str(exc)}
    doc = _last_json(proc.stdout)
    if not doc:
        return {"run_id": run_id, "_error": (proc.stderr or proc.stdout or "no JSON").strip()[-500:]}
    if has_key_named(doc, "claimed"):
        return {"run_id": run_id, "_error": "dos status emitted forbidden claimed field",
                "reason": "RUN_STATUS_CLAIMED_FIELD"}
    doc.setdefault("run_id", run_id)
    doc["_returncode"] = proc.returncode
    return doc


def read_run_status_digests(
    root: Path,
    *,
    ledger: Path | None = None,
    limit: int = 6,
    status_reader: Any | None = None,
) -> list[dict[str, Any]]:
    ledger = ledger or root / ".fak" / "loops.jsonl"
    reader = status_reader or (lambda rid: dos_status_digest(root, rid))
    return [reader(rid) for rid in run_ids_from_loop_ledger(ledger, limit=limit)]


def silent_workers(
    runs_dir: Path,
    *,
    alive: set[int] | None = None,
    probe: Any | None = None,
) -> list[dict[str, Any]]:
    """Issue-resolution workers that exited having produced NOTHING — a
    ``resolve-<N>-<stamp>.log`` at or below the real-turn floor
    (``_STUB_LOG_MAX_BYTES``) whose ``.pid`` process is dead.

    "Produced nothing" is NOT only a 0-byte file: a detached worker that prints its
    spawn header + a TUI banner (e.g. opencode's ``> build · <model>``, ~122 bytes)
    and exits landed zero turns just the same, yet ``size != 0`` hid it from this
    card before #1276. The byte floor catches both; each row carries its ``size`` and
    a ``kind`` (``empty``/``stub``) so the render is honest about which it found.

    A ``claude -p`` worker writes nothing to stdout until its final message, so a
    sub-floor log with a *live* pid is still-running (not silent) and is excluded. A
    dead pid over a sub-floor log is the "spun, produced nothing" residual the cooldown
    self-corrects but leaves operator-invisible — this is the signal. Best effort:
    psutil is optional (its absence means we cannot prove a pid dead, so we report
    nothing rather than a false silent), exactly like
    ``issue_resolve_dispatch.live_resolution_issues``. Newest first.
    """
    if not runs_dir.is_dir():
        return []
    if alive is None and probe is None:
        try:
            import psutil  # type: ignore

            alive = {p.pid for p in psutil.process_iter()}
        except ImportError:
            alive = None  # cannot prove liveness -> report no silents (no false alarms)
    out: list[dict[str, Any]] = []
    for log in runs_dir.glob("resolve-*.log"):
        m = _RESOLVE_LOG_RE.search(log.name)
        if not m:
            continue
        try:
            size = log.stat().st_size
        except OSError:
            continue
        if size > _STUB_LOG_MAX_BYTES:
            continue  # over the real-turn floor -> produced output, not silent
        pid_file = log.with_suffix(".pid")
        if not pid_file.exists():
            continue
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if alive is None and probe is None:
            continue  # no liveness oracle this run -> do not claim it is silent
        if dispatch_preflight.resolve_sidecar_pid_is_live(
            pid_file, alive=alive, probe=probe):
            continue  # still running -> not (yet) silent
        out.append({"issue": int(m.group(1)), "stamp": m.group(2), "log": log.name,
                    "pid": pid, "size": size, "kind": "empty" if size == 0 else "stub"})
    out.sort(key=lambda r: r["stamp"], reverse=True)
    return out


_SPAWN_FAILED_CAUSE_SCHEMA = "fak.spawn-failed-cause-breakdown.v1"
_SPAWN_CAUSE_LOOKBACK_MIN = 24 * 60  # default trailing window (24h)
# A trailing stale_cred (permanent-auth) spawn-failure rate at/above this fraction
# of ALL spawns is the drain signature of a needs_login seat bleeding the fleet
# (#4590). It reddens the DEFAULT card instead of hiding behind --spawn-causes.
# Floored at a minimum spawn count so a 1-of-2 fluke never trips the alarm.
_SPAWN_STALE_CRED_RED_RATE = 0.10
_SPAWN_STALE_CRED_RED_MIN_SPAWNS = 8


def spawn_failed_cause_breakdown(
    runs_dir: Path,
    *,
    lookback_min: int = _SPAWN_CAUSE_LOOKBACK_MIN,
    now_ts: float | None = None,
    alive: set[int] | None = None,
    probe: Any | None = None,
    max_evidence: int = 5,
) -> dict[str, Any]:
    """Read-only fold: the trailing SPAWN_FAILED early-exit rate, BROKEN DOWN by
    cause (#2635).

    The ~1-in-25 baseline early-exit noise is absorbed by backend/account failover
    and dismissed as "known noise" — but it was only ever LABELLED SPAWN_FAILED,
    never attributed, so a regression in one sub-cause could hide inside the ~4%
    aggregate. This fold reads the SAME disk artifacts :func:`silent_workers`
    reads — a dead-pid ``resolve-<N>-<stamp>.log`` at/below the stub floor is one
    empty/stub early-exit event — classifies each by
    ``issue_resolve_dispatch.classify_spawn_failed_cause`` on its log tail, and
    reports the count + rate PER cause with per-event evidence rows, so "≈4%
    baseline" becomes a named, watchable MIX instead of a lumped constant.

    Denominator = every ``resolve-*.log`` inside the window (each is one spawn), so
    ``rate`` is spawn_failed / spawns. Scope note: a crash that dumps a LARGE log
    before exiting clears the stub floor and is out of this disk-fold's population —
    it is attributed instead by the live tick's ``cause`` stamp
    (:func:`issue_resolve_dispatch.classify_spawn_failed_cause` at the spawn_failed
    payload sites). Read-only + FAIL-OPEN: no psutil oracle ⇒ the silent set is
    empty and the mix is reported over 0 events (never a false attribution). Newest
    evidence first."""
    schema = _SPAWN_FAILED_CAUSE_SCHEMA
    empty = {"schema": schema, "lookback_min": lookback_min, "spawns": 0,
             "spawn_failed": 0, "rate": 0.0, "by_cause": {}, "events": []}
    if not runs_dir.is_dir():
        return empty
    try:
        import issue_resolve_dispatch as ird  # type: ignore
    except ImportError:
        return empty
    now_ts = time.time() if now_ts is None else now_ts
    horizon = now_ts - lookback_min * 60
    # Denominator: every spawn (resolve log) whose mtime is inside the window.
    spawns = 0
    for log in runs_dir.glob("resolve-*.log"):
        if not _RESOLVE_LOG_RE.search(log.name):
            continue
        try:
            if log.stat().st_mtime >= horizon:
                spawns += 1
        except OSError:
            continue
    causes = list(getattr(ird, "SPAWN_FAILED_CAUSES",
                          ("weekly_limit", "stale_cred", "child_crash",
                           "exec_race", "unknown")))
    by_cause: dict[str, dict[str, Any]] = {
        c: {"count": 0, "evidence": []} for c in causes}
    events: list[dict[str, Any]] = []
    tail_chars = int(getattr(ird, "EARLY_EXIT_TAIL_CHARS", 8192))
    # Numerator: the silent/stub early-exit population, classified by cause.
    for row in silent_workers(runs_dir, alive=alive, probe=probe):
        log = runs_dir / str(row.get("log") or "")
        try:
            if log.stat().st_mtime < horizon:
                continue
        except OSError:
            continue
        try:
            tail = log.read_text(encoding="utf-8", errors="replace")[-tail_chars:]
        except OSError:
            tail = ""
        size = int(row.get("size") or 0)
        cause = ird.classify_spawn_failed_cause(
            {"tail": tail, "silent": size == 0, "log_bytes": size})
        ev = {"issue": row.get("issue"), "log": row.get("log"),
              "stamp": row.get("stamp"), "size": size, "cause": cause}
        events.append(ev)
        bucket = by_cause.setdefault(cause, {"count": 0, "evidence": []})
        bucket["count"] += 1
        if len(bucket["evidence"]) < max_evidence:
            bucket["evidence"].append(ev)
    spawn_failed = len(events)
    for bucket in by_cause.values():
        n = int(bucket["count"])
        bucket["rate_of_failed"] = round(n / spawn_failed, 3) if spawn_failed else 0.0
        bucket["rate_of_spawns"] = round(n / spawns, 4) if spawns else 0.0
    return {
        "schema": schema, "lookback_min": lookback_min,
        "spawns": spawns, "spawn_failed": spawn_failed,
        "rate": round(spawn_failed / spawns, 4) if spawns else 0.0,
        "by_cause": by_cause, "events": events,
    }


def render_spawn_causes(b: dict[str, Any]) -> str:
    """One-screen render of :func:`spawn_failed_cause_breakdown` (#2635): the
    trailing SPAWN_FAILED rate, then a per-cause mix with a few evidence rows."""
    win_h = round(int(b.get("lookback_min") or 0) / 60, 1)
    spawns = int(b.get("spawns") or 0)
    failed = int(b.get("spawn_failed") or 0)
    lines = [
        f"spawn-failed cause breakdown ({b.get('schema')})",
        f"  window   : trailing {win_h}h — {failed}/{spawns} spawns early-exited "
        f"(rate {b.get('rate')})",
    ]
    by_cause = b.get("by_cause") or {}
    for cause in sorted(by_cause, key=lambda c: (-int(by_cause[c].get('count') or 0), c)):
        row = by_cause[cause]
        n = int(row.get("count") or 0)
        lines.append(f"  {cause:<12}: {n:>3}  (of failed {row.get('rate_of_failed')}, "
                     f"of spawns {row.get('rate_of_spawns')})")
        for ev in (row.get("evidence") or [])[:3]:
            lines.append(f"      · #{ev.get('issue')} {ev.get('log')} ({ev.get('size')}B)")
    if not failed:
        lines.append("  (no early-exit events in window — nothing to attribute)")
    return "\n".join(lines)


def spawn_stale_cred_alarm(breakdown: dict[str, Any] | None) -> dict[str, Any]:
    """Classify a :func:`spawn_failed_cause_breakdown` for the default-card
    stale_cred drain alarm (#4590). A trailing stale_cred rate-of-spawns at/above
    ``_SPAWN_STALE_CRED_RED_RATE`` (with at least ``_..._MIN_SPAWNS`` spawns to be
    real, not a small-sample fluke) is the needs_login-seat drain signature — the
    one spawn signal that FLIPS the card verdict to red. Returns
    ``{red, rate, spawns, count, reason}`` (reason non-empty only when red)."""
    b = breakdown or {}
    spawns = int(b.get("spawns") or 0)
    bucket = (b.get("by_cause") or {}).get("stale_cred") or {}
    count = int(bucket.get("count") or 0)
    rate = float(bucket.get("rate_of_spawns") or 0.0)
    red = (spawns >= _SPAWN_STALE_CRED_RED_MIN_SPAWNS
           and rate >= _SPAWN_STALE_CRED_RED_RATE)
    reason = ""
    if red:
        win_h = round(int(b.get("lookback_min") or 0) / 60, 1)
        reason = (f"stale-cred spawn-failure drain: {count}/{spawns} spawns "
                  f"({int(round(rate * 100))}%) early-exited on stale credentials over "
                  f"the trailing {win_h}h — a needs_login seat is draining the fleet; "
                  f"re-login/replace the seat (#4590)")
    return {"red": red, "rate": rate, "spawns": spawns, "count": count,
            "reason": reason}


# Seat-keyed spawn-failure streak (#4591). The dispatcher persists a per-SEAT
# consecutive-failure run-length (tools/issue_resolve_dispatch.py
# bump_spawn_failure_streak_seat) precisely because the target-keyed streak lets
# a dead needs_login seat cycling across DIFFERENT issues evade every cooldown.
# A seat at/over this threshold flips the card verdict with a named seat +
# operator action. Mirrors issue_resolve_dispatch.SPAWN_FAILED_RED_STREAK (kept
# literal here so the always-on card never imports the heavy dispatcher module).
_SEAT_SPAWN_FAIL_RED_STREAK = 3
_SEAT_STREAK_LEDGER_GLOB = "spawn-failure-streak-seat-*.json"
_SEAT_STREAK_LEDGER_PREFIX = "spawn-failure-streak-seat-"


def read_seat_spawn_failure_streaks(runs_dir: Path) -> list[dict[str, Any]]:
    """Read every per-backend seat-keyed spawn-failure streak ledger (#4591) into
    flat rows ``{seat, backend, streak, last_ts}``, worst first. Pure-local disk
    read, best-effort: an unreadable/malformed ledger contributes nothing."""
    rows: list[dict[str, Any]] = []
    try:
        paths = sorted(runs_dir.glob(_SEAT_STREAK_LEDGER_GLOB))
    except OSError:
        return rows
    for path in paths:
        backend = path.name[len(_SEAT_STREAK_LEDGER_PREFIX):-len(".json")] or "?"
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        for tag, v in doc.items():
            if isinstance(v, dict):
                streak = _int(v.get("count"), 0) or 0
                last_ts = v.get("last_ts")
            else:
                streak = _int(v, 0) or 0
                last_ts = None
            if streak > 0:
                rows.append({"seat": str(tag), "backend": backend,
                             "streak": streak, "last_ts": last_ts})
    rows.sort(key=lambda r: (-r["streak"], r["seat"]))
    return rows


def seat_spawn_fail_alarm(seat_streaks: list[dict[str, Any]] | None,
                          seat_inventory: dict[str, Any] | None,
                          spawn_causes: dict[str, Any] | None,
                          *, threshold: int = _SEAT_SPAWN_FAIL_RED_STREAK) -> dict[str, Any]:
    """The #4591 verdict flip: join the SEAT-keyed spawn-failure streak with the
    seat inventory's ``auth_failed`` hold and the trailing ``stale_cred`` cause
    mix, and go red the moment ANY single seat has ``threshold`` spawn-fails in
    a row — however many distinct issues it burned them on. Pure fold, so it is
    table-testable; returns ``{red, seats, reason}`` with ``seats`` = the
    over-threshold rows annotated ``auth_failed``/``stale_cred``, and a reason
    of the shape ``seat X: N spawn-fails in a row (stale_cred) -> fak accounts
    status`` (cause qualifier present only when the join confirms it)."""
    over = [dict(r) for r in (seat_streaks or [])
            if _int(r.get("streak"), 0) >= threshold > 0]
    if not over:
        return {"red": False, "seats": [], "threshold": threshold, "reason": ""}
    auth_failed_tags = {
        _seat_label(s)
        for s in (seat_inventory or {}).get("seats", [])
        if str(s.get("hold_reason") or "") == "auth_failed"
    }
    stale_cred_seen = bool(_int(
        (((spawn_causes or {}).get("by_cause") or {}).get("stale_cred") or {}).get("count"), 0))
    parts: list[str] = []
    for row in over:
        row["auth_failed"] = row.get("seat") in auth_failed_tags
        row["stale_cred"] = stale_cred_seen
        qualifier = " (stale_cred)" if (row["auth_failed"] or stale_cred_seen) else ""
        parts.append(f"seat {row.get('seat')}: {row.get('streak')} spawn-fails "
                     f"in a row{qualifier} -> fak accounts status")
    return {"red": True, "seats": over, "threshold": threshold,
            "reason": "; ".join(parts) + " (#4591)"}


def parse_ships_per_worker(records: list[str]) -> dict[str, Any]:
    """Fold a list of commit-message records (each ``subject\\n body``) into a
    ships-per-worker attribution (#2065). Pure — the git read lives in the caller.

    Each record is one commit; its first ``(fak-worker <id>)`` trailer names the worker
    that authored it. A record with no parseable trailer is bucketed ``unknown`` (the
    grep can match a commit whose trailer is malformed, and a fixture can feed a
    zero-trailer record) so the fold never silently drops a matched commit. Best-effort:
    the trailer is agent-emitted, so this is an attribution AID, not a witness."""
    counts: dict[str, int] = {}
    for rec in records:
        if not rec.strip():
            continue
        m = _FAK_WORKER_TRAILER_RE.search(rec)
        worker = _sanitize_worker_token(m.group(1)) if m else "unknown"
        counts[worker] = counts.get(worker, 0) + 1
    workers = sorted(({"worker": w, "ships": n} for w, n in counts.items()),
                     key=lambda r: (-r["ships"], r["worker"]))
    total = sum(counts.values())
    return {
        "schema": _SHIPS_PER_WORKER_SCHEMA,
        "attributed_ships": total,
        "worker_count": sum(1 for w in counts if w != "unknown"),
        "unknown": counts.get("unknown", 0),
        "workers": workers,
        "note": "best-effort agent-emitted (fak-worker) trailer — attribution aid, not a witness",
    }


def _sanitize_worker_token(s: str) -> str:
    """Trim a parsed trailer id to a stable token, mirroring
    issue_dispatch._sanitize_worker_id so the two ends agree on the id alphabet."""
    out = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in str(s or "").strip())
    return out.strip("-.") or "unknown"


def ships_per_worker(
    root: Path,
    *,
    lookback_min: int = _SHIPS_PER_WORKER_LOOKBACK_MIN,
    now_ts: float | None = None,
    runner: Any | None = None,
) -> dict[str, Any]:
    """Read-only ships-per-worker fold over recent git history (#2065).

    Runs ``git log --grep='(fak-worker ' -F`` over the lookback window and folds the
    matched commits' ``(fak-worker <id>)`` trailers into per-worker ship counts. The
    grep and the fixed-string flag keep the scan to trailer-carrying commits; the pure
    ``parse_ships_per_worker`` does the bucketing. Fail-open: if git can't answer, the
    fold is empty rather than fatal (like ``count_lane_ancestry_closes``). ``runner`` is
    injectable so the fold is testable without a real repo."""
    now_ts = time.time() if now_ts is None else now_ts
    # Clamp to the epoch floor: a tiny/zero now_ts (e.g. an injected test clock) would
    # otherwise hand time.gmtime a negative timestamp, which raises on Windows.
    since_iso = time.strftime("%Y-%m-%dT%H:%M:%S +0000",
                              time.gmtime(max(0.0, now_ts - lookback_min * 60)))
    run = runner or _run_ships_per_worker_git
    records = run(root, since_iso)
    if records is None:
        out = parse_ships_per_worker([])
        out["unavailable"] = True
        return out
    out = parse_ships_per_worker(records)
    out["lookback_min"] = lookback_min
    return out


def _run_ships_per_worker_git(root: Path, since_iso: str) -> list[str] | None:
    """Shell ``git log`` for trailer-carrying commits; return one ``subject\\n body``
    record per commit (split on the record separator), or ``None`` if git can't answer."""
    cmd = ["git", "log", f"--since={since_iso}", "--no-merges", "-F",
           f"--grep={_SHIPS_PER_WORKER_GREP}", "--pretty=format:%x1e%s%n%b"]
    try:
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=30,
                              creationflags=_win_creationflags())
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return [rec for rec in (proc.stdout or "").split("\x1e") if rec.strip()]


_COMMIT_DROUGHT_HOURS = 3.0


def commit_drought(
    root: Path,
    *,
    hours: float = _COMMIT_DROUGHT_HOURS,
    now_ts: float | None = None,
    runner: Any | None = None,
) -> dict[str, Any]:
    """Loop-level drought witness: ZERO fleet-attributed commits over the last
    ``hours``.

    The per-worker cards (``silent_workers``, ``ships_per_worker``) catch an
    individual dud worker, but they stay green when the WHOLE loop quietly ships
    nothing — the exact blind spot that lets an armed fleet sit at zero commits
    unnoticed. This folds the same ``(fak-worker <id>)`` trailer scan over a
    wall-clock window and reports ``dry`` when the count is zero. The ALARM bit
    (``droughty``) is derived by the caller, which ANDs ``dry`` with the
    armed/watchdog-installed state (a drought while disarmed is idle, not an
    alarm). Fail-open: if git cannot answer (``runner`` -> ``None``) we report
    ``unavailable`` rather than a false drought. ``runner``/``now_ts`` are
    injectable so the witness is testable without a real repo or clock."""
    now_ts = time.time() if now_ts is None else now_ts
    since_iso = time.strftime(
        "%Y-%m-%dT%H:%M:%S +0000",
        time.gmtime(max(0.0, now_ts - hours * 3600.0)))
    run = runner or _run_ships_per_worker_git
    records = run(root, since_iso)
    if records is None:
        return {"hours": hours, "unavailable": True}
    count = len([rec for rec in records if rec.strip()])
    return {"hours": hours, "commit_count": count, "dry": count == 0}


def _parse_watchdog_query(stdout: str) -> dict[str, Any]:
    """Pull ``Status`` and ``Last Result`` from ``schtasks /Query /V /FO LIST``
    output. Returns ``{"status": str|None, "last_result": int|None}``; matches on
    the field label so a red LastTaskResult (#2636) is captured for the #2642 watch
    digest, and tolerates absent fields (``None``) instead of raising."""
    status = None
    last_result = None
    for line in (stdout or "").splitlines():
        label, sep, val = line.partition(":")
        if not sep:
            continue
        label = label.strip().lower()
        val = val.strip()
        if label == "status":
            status = val or None
        elif label == "last result":
            toks = val.split()
            last_result = _int(toks[0]) if toks else None
    return {"status": status, "last_result": last_result}


def watchdog_installed() -> dict[str, Any]:
    """Is the always-on watchdog scheduled task registered, and is it enabled?

    Queries verbose (``/V``) so ``Last Result`` is captured alongside ``Status``:
    the #2642 watch digest consumes that red/green bit to classify a scheduled-task
    result as clean / self-healing / unresolved (issue Done condition, question 2)."""
    try:
        proc = subprocess.run(
            ["schtasks", "/Query", "/TN", WATCHDOG_TASK, "/V", "/FO", "LIST"],
            capture_output=True, text=True, timeout=15,
            creationflags=_win_creationflags())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"installed": None, "error": str(exc)}
    if proc.returncode != 0:
        return {"installed": False, "status": None}
    parsed = _parse_watchdog_query(proc.stdout)
    return {"installed": True, "status": parsed["status"],
            "last_result": parsed["last_result"]}


def _int(v: Any, d: int | None = None) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return d


def read_active_weekly_cap(runs_dir: Path, account_tag: str | None,
                           now_ts: float | None = None) -> dict[str, Any] | None:
    """The active weekly-cap hold for ``account_tag`` (if any), read from the
    dispatcher's persisted state (``account-cap-*.json``, written by the
    issue_resolve_dispatch weekly-cap gate). None when no unexpired hold matches.
    Read-only / best-effort, so the card can show WHY a logged-in account is held."""
    import datetime as _dt
    import time as _time
    try:
        now_ts = _time.time() if now_ts is None else now_ts
        now = _dt.datetime(1970, 1, 1) + _dt.timedelta(seconds=now_ts)
    except Exception:
        return None
    if not runs_dir.is_dir():
        return None
    best: tuple[dict[str, Any], _dt.datetime] | None = None
    for path in runs_dir.glob("account-cap-*.json"):
        try:
            st = json.loads(path.read_text(encoding="utf-8"))
            until = _dt.datetime.fromisoformat((st.get("until") or "").replace("Z", ""))
        except (OSError, ValueError):
            continue
        if now >= until or (account_tag and st.get("account") not in (None, account_tag)):
            continue
        if best is None or until < best[1]:
            best = (st, until)
    return best[0] if best else None


def read_backend_health(runs_dir: Path) -> list[dict[str, Any]]:
    """The backends currently held DEAD by the dispatcher's backend-health gate, read
    from ``backend-health-*.json`` (written by issue_resolve_dispatch.check_backend_health
    when a backend spins on a banner-only/0-byte streak). Each row carries the product,
    since-when, the lane reallocated to a healthy backend, and the evidence logs — so the
    card shows WHY a backend stopped spawning and where its work went. Read-only /
    best-effort; a corrupt or healthy sidecar is skipped. Newest-dead first."""
    out: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return out
    for path in runs_dir.glob("backend-health-*.json"):
        try:
            st = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if st.get("state") != "dead":
            continue
        out.append({k: st.get(k) for k in
                    ("product", "since", "abandoned_lane", "evidence_logs",
                     "reprobe_min", "last_reprobe")})
    out.sort(key=lambda r: str(r.get("since") or ""), reverse=True)
    return out


def _recent_backend_products(runs_dir: Path, *, lookback_min: int,
                             now_ts: float, backend_of_log: Any) -> list[str]:
    if not runs_dir.is_dir():
        return []
    horizon = now_ts - lookback_min * 60
    products: set[str] = set()
    for log in runs_dir.glob("resolve-*.log"):
        if not _RESOLVE_LOG_RE.search(log.name):
            continue
        try:
            if log.stat().st_mtime < horizon:
                continue
        except OSError:
            continue
        product = str(backend_of_log(log) or "claude")
        if product:
            products.add(product)
    return sorted(products)


def backend_stub_rates(
    runs_dir: Path,
    *,
    lookback_min: int = _BACKEND_STUB_LOOKBACK_MIN,
    now_ts: float | None = None,
    alive: set[int] | None = None,
    probe: Any | None = None,
) -> list[dict[str, Any]]:
    """Recent per-backend productive-vs-stub rollup from worker log content.

    This is intentionally independent of ``backend-health-*.json`` sidecars: a
    backend that stopped reaching its own dispatch tick may never persist a DEAD
    hold, but its recent ``resolve-*.log`` files can still prove that it is mostly
    spawning banner-only/0-byte no-ops. The per-log classification is delegated to
    ``issue_resolve_dispatch._classify_backend_logs`` so this status card shares
    the dispatcher's real-turn floor and quota-banner exception.
    """
    if not runs_dir.is_dir():
        return []
    import time
    try:
        import issue_resolve_dispatch as ird  # type: ignore
    except ImportError:
        return []
    now_ts = time.time() if now_ts is None else now_ts
    products = _recent_backend_products(
        runs_dir, lookback_min=lookback_min, now_ts=now_ts,
        backend_of_log=ird._backend_of_log)
    out: list[dict[str, Any]] = []
    for product in products:
        rows = ird._classify_backend_logs(
            runs_dir, product=product, lookback_min=lookback_min,
            now_ts=now_ts, alive=alive, probe=probe)
        if not rows:
            continue
        productive = sum(1 for r in rows if r.get("productive"))
        stub = len(rows) - productive
        evidence = [str(r.get("log")) for r in rows
                    if not r.get("productive") and r.get("log")][:5]
        out.append({
            "product": product,
            "lookback_min": lookback_min,
            "total": len(rows),
            "productive": productive,
            "stub": stub,
            "stub_rate": round(stub / len(rows), 3),
            "majority_stub": stub > productive,
            "evidence_logs": evidence,
        })
    out.sort(key=lambda r: (-float(r.get("stub_rate") or 0),
                            -int(r.get("stub") or 0),
                            str(r.get("product") or "")))
    return out


# A worker logs ``hook: <name> Failed`` when a lifecycle hook (the fak guard layer
# bound via the harness's hook config) fails to execute. ``claude`` binds the
# guard hooks natively; a non-claude backend (codex/opencode) runs its OWN native
# hook config, and when that config can't reach the dos hook CLI at runtime EVERY
# lifecycle hook fails — the worker stays productive while running UNHOOKED by the
# guard layer. Reuse the stub-rate lookback so both backend folds share one window.
_HOOK_FAIL_LOOKBACK_MIN = _BACKEND_STUB_LOOKBACK_MIN


def backend_hook_failures(
    runs_dir: Path,
    *,
    lookback_min: int = _HOOK_FAIL_LOOKBACK_MIN,
    now_ts: float | None = None,
    reader: Any | None = None,
) -> list[dict[str, Any]]:
    """Per-backend hook-failure rollup from recent worker-log content (#1277).

    The fak guard hooks (PreToolUse / PostToolUse / Stop) bind a worker through the
    harness's lifecycle-hook layer. A non-claude backend (codex/opencode) runs its
    OWN native hook config, and when that config can't reach the dos hook CLI at
    runtime every lifecycle hook logs ``hook: <name> Failed`` — the worker stays
    productive while running UNHOOKED by the guard layer. This fold counts those
    lines per backend over the recent ``resolve-*.log`` sessions and flags any
    backend whose EVERY recent session failed its hooks, so a fully-unhooked backend
    is no longer silent in the status card — the explicit "at minimum, surface the
    hook-failure rate" ask of #1277.

    Reuses ``dispatch_log_audit``'s hook detector + backend sidecar reader (one
    source of truth for the ``hook: … Failed`` signature). Best-effort: if that
    module can't import, the fold is empty (no false signal). Worst (unhooked, most
    failures) first.
    """
    if not runs_dir.is_dir():
        return []
    import time
    try:
        import dispatch_log_audit as dla  # type: ignore
    except ImportError:
        return []
    now_ts = time.time() if now_ts is None else now_ts
    horizon = now_ts - lookback_min * 60
    read = reader or dla._read_text
    by_backend: dict[str, dict[str, Any]] = {}
    for log in sorted(runs_dir.glob("resolve-*.log")):
        if not _RESOLVE_LOG_RE.search(log.name):
            continue
        try:
            if log.stat().st_mtime < horizon:
                continue
        except OSError:
            continue
        text = read(log)
        if text is None:
            continue
        backend = str(dla.backend_of_log(log) or "claude")
        row = by_backend.setdefault(backend, {
            "product": backend, "lookback_min": lookback_min, "sessions": 0,
            "sessions_with_hook_failures": 0, "hook_failures": 0, "evidence_logs": []})
        row["sessions"] += 1
        count = sum(int(f["count"]) for f in dla._match_hook_failures(text))
        if count > 0:
            row["sessions_with_hook_failures"] += 1
            row["hook_failures"] += count
            if len(row["evidence_logs"]) < 5:
                row["evidence_logs"].append(log.name)
    out: list[dict[str, Any]] = []
    for row in by_backend.values():
        s = int(row["sessions"])
        failed_sessions = int(row["sessions_with_hook_failures"])
        row["failure_session_rate"] = round(failed_sessions / s, 3) if s else 0.0
        # "fully unhooked" = every recent session of this backend failed its hooks,
        # i.e. the guard hook layer never bound on this backend over the window.
        row["all_sessions_unhooked"] = bool(
            s > 0 and failed_sessions == s and row["hook_failures"] > 0)
        out.append(row)
    out.sort(key=lambda r: (not r["all_sessions_unhooked"],
                            -int(r["hook_failures"]), str(r["product"])))
    return out


def _guard_livelock_candidates(name: str, text: str) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str, str, str], int] = {}
    longest: dict[tuple[str, str, str, str, str], int] = {}
    last_key: tuple[str, str, str, str, str] | None = None
    run_len = 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            last_key = None
            run_len = 0
            continue
        if not _guard_row_can_livelock(row):
            last_key = None
            run_len = 0
            continue
        digest = str(row.get("args_digest") or row.get("result_digest") or "")
        if not digest:
            last_key = None
            run_len = 0
            continue
        key = (
            str(row.get("kind") or "UNKNOWN"),
            str(row.get("verdict") or ""),
            str(row.get("tool") or ""),
            str(row.get("reason") or ""),
            digest,
        )
        counts[key] = counts.get(key, 0) + 1
        if key == last_key:
            run_len += 1
        else:
            last_key = key
            run_len = 1
        longest[key] = max(longest.get(key, 0), run_len)

    out: list[dict[str, Any]] = []
    for key, count in counts.items():
        max_run = longest.get(key, 0)
        if count < _GUARD_LIVELOCK_MIN_COUNT and max_run < _GUARD_LIVELOCK_THRESHOLD:
            continue
        kind, verdict, tool, reason, digest = key
        out.append({
            "file": name,
            "count": count,
            "longest_run": max_run,
            "kind": kind,
            "verdict": verdict,
            "tool": tool,
            "reason": reason,
            "digest": digest,
        })
    return out


def _guard_row_can_livelock(row: dict[str, Any]) -> bool:
    kind = str(row.get("kind") or "").upper()
    verdict = str(row.get("verdict") or "").upper()
    if kind in _GUARD_DENY_KINDS or kind == _GUARD_QUARANTINE_KIND:
        return True
    return verdict in (*_GUARD_DENY_KINDS, _GUARD_QUARANTINE_KIND)


def _guard_livelock_label(row: dict[str, Any]) -> str:
    tool = row.get("tool") or row.get("kind") or "?"
    reason = row.get("reason") or row.get("verdict") or "?"
    digest = str(row.get("digest") or "")
    short = digest[:12] if digest else "-"
    return (f"{row.get('file')} {tool}/{reason} digest={short} "
            f"count={row.get('count')} run={row.get('longest_run')}")


def guard_coverage(
    runs_dir: Path,
    *,
    lookback_min: int = _GUARD_RECENT_LOOKBACK_MIN,
    now_ts: float | None = None,
) -> dict[str, Any]:
    """Roll up the per-session ``fak guard`` decision journals on the dispatch path.

    The concurrent-dispatch fleet fronts every worker with ``fak guard`` by default
    (dispatch_worker.py and cmd/dispatchworker), and each guarded session owns a
    unique hash-chained journal under ``.dispatch-runs/guard-audit/*.jsonl`` whose
    every non-empty line is one kernel decision (``internal/journal.Row``). This fold
    is the WITNESS that the dispatch path actually ran THROUGH the kernel — and what
    the kernel decided (allow / deny / quarantine) — rather than a flag claiming it
    did. It does NOT invent a coverage percent it cannot ground: it reports the
    witnessed session + decision counts, which a self-report cannot fake.

    Read-only / best-effort. Returns a payload with:

      * ``sessions`` — guarded worker sessions on record (= journal files)
      * ``recent_sessions`` — journals touched within ``lookback_min``
      * ``empty_sessions`` — journals with 0 decision rows (booted under guard but
        proposed no adjudicated tool call — the silent empty-turn signature)
      * ``rows`` / ``recent_rows`` — total / recent kernel decisions
      * ``by_kind`` — the decision mix (DECIDE/DENY/RESULT_DENY/QUARANTINE/VDSO_HIT/…)
      * ``denied`` / ``quarantined`` — derived refusal counts
      * ``livelock_candidates`` — bounded repeated digest candidates from recent
        journals (count + longest consecutive run)
      * ``evidence`` — the most-recent journal filenames
    """
    import time

    audit_dir = runs_dir / GUARD_AUDIT_DIRNAME
    payload: dict[str, Any] = {
        "dir_present": audit_dir.is_dir(),
        "sessions": 0,
        "recent_sessions": 0,
        "empty_sessions": 0,
        "rows": 0,
        "recent_rows": 0,
        "by_kind": {},
        "denied": 0,
        "quarantined": 0,
        "lookback_min": lookback_min,
        "evidence": [],
        "livelock_threshold": _GUARD_LIVELOCK_THRESHOLD,
        "livelock_candidates": [],
    }
    if not audit_dir.is_dir():
        return payload

    now_ts = time.time() if now_ts is None else now_ts
    horizon = now_ts - lookback_min * 60
    by_kind: dict[str, int] = {}
    livelock_candidates: list[dict[str, Any]] = []
    files: list[tuple[float, str, int]] = []  # (mtime, name, rows) for evidence/recency
    for jp in audit_dir.glob("*.jsonl"):
        try:
            mtime = jp.stat().st_mtime
            text = jp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rows = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                kind = str(json.loads(line).get("kind") or "UNKNOWN")
            except ValueError:
                kind = "MALFORMED"
            by_kind[kind] = by_kind.get(kind, 0) + 1
        payload["sessions"] += 1
        payload["rows"] += rows
        if rows == 0:
            payload["empty_sessions"] += 1
        if mtime >= horizon:
            payload["recent_sessions"] += 1
            payload["recent_rows"] += rows
            livelock_candidates.extend(_guard_livelock_candidates(jp.name, text))
        files.append((mtime, jp.name, rows))

    payload["by_kind"] = dict(sorted(by_kind.items()))
    payload["denied"] = sum(by_kind.get(k, 0) for k in _GUARD_DENY_KINDS)
    payload["quarantined"] = by_kind.get(_GUARD_QUARANTINE_KIND, 0)
    livelock_candidates.sort(
        key=lambda r: (-int(r.get("longest_run") or 0), -int(r.get("count") or 0),
                       str(r.get("file") or "")))
    payload["livelock_candidates"] = livelock_candidates[:_GUARD_LIVELOCK_LIMIT]
    files.sort(key=lambda r: r[0], reverse=True)
    payload["evidence"] = [name for _, name, _ in files[:5]]
    return payload


def _total_commits(root: Path) -> int | None:
    """Commits reachable from HEAD, or None if git can't answer. Used to size the
    closure-audit window to the repo so it never silently scans a stale slice."""
    try:
        proc = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=str(root),
                              capture_output=True, text=True, timeout=15,
                              creationflags=_win_creationflags())
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (proc.stdout or "").strip()
    return int(out) if proc.returncode == 0 and out.isdigit() else None


def read_seat_inventory(root: Path, *, product: str | None = None) -> dict[str, Any]:
    """The operator-facing seat inventory (#1799): for every known account/seat, its
    dispatch_state (available/busy/cooling/unavailable) and, when not simply
    available, a specific hold_reason (e.g. ``auth_failed``, ``rate_limited``,
    ``cooldown_until=<ts>``, ``no_capacity``) -- never a bare "unavailable".

    Delegates entirely to ``fleet_accounts.seat_pool()`` (the existing explicit
    multi-seat pool) so this reuses the SAME hold-reason vocabulary the roster and
    the dispatch-preflight spawn gate already use, rather than inventing new state
    names. Pure-local (no gh, no subprocess) and best-effort: an import/read failure
    degrades to ``{"_error": ...}`` like the other fast, always-run folds on this
    card, never raises."""
    try:
        sys.path.insert(0, str(root / "tools"))
        import fleet_accounts  # noqa: PLC0415  (lazy: heavy module, only paid for here)
        rows = fleet_accounts.annotate_accounts(fleet_accounts.discover_accounts())
        leases = fleet_accounts.live_seat_leases(root / RUNS_DIRNAME)
        return fleet_accounts.seat_pool(rows, leases, product=product)
    except Exception as exc:  # best-effort card fold; never fail the whole card
        return {"_error": str(exc)}


def _seat_label(seat: dict[str, Any]) -> str:
    return str(seat.get("tag") or seat.get("account") or seat.get("seat") or "?")


def read_fleet_net_decline(root: Path, *, window: int = 24) -> dict[str, Any]:
    """The net-worker-decline alarm (#4591, part 2): fold the trailing
    ``fleet-status-history.jsonl`` appends (the #4594 per-tick producer) through
    ``fleet_trend.net_decline_alarm`` — red when ``live`` fell strictly for
    ``NET_DECLINE_ALARM_STREAK`` consecutive appends. Pure-local ledger read,
    best-effort: an import/read failure degrades to ``{"_error": ...}`` like the
    sibling card folds, never raises."""
    try:
        sys.path.insert(0, str(root / "tools"))
        import fleet_trend  # noqa: PLC0415  (lazy: only paid for on the card)
        rows = fleet_trend.tail(str(root / fleet_trend.DEFAULT_LEDGER), window)
        out = fleet_trend.net_decline_alarm(rows)
        out["window"] = window
        return out
    except Exception as exc:  # best-effort card fold; never fail the whole card
        return {"_error": str(exc)}


def _limited_seat_labels(labels: list[str], *, limit: int = 4) -> str:
    kept = labels[:limit]
    if len(labels) > limit:
        kept.append(f"+{len(labels) - limit} more")
    return ", ".join(kept)


def _auth_failed_seat_action(seat_inventory: dict[str, Any]) -> str:
    tags = [
        _seat_label(s)
        for s in seat_inventory.get("seats", [])
        if str(s.get("hold_reason") or "") == "auth_failed"
    ]
    if not tags:
        return ""
    return (
        f"auth_failed={len(tags)} [{_limited_seat_labels(tags)}]; "
        "next action: run `fak accounts status` and re-login or remove the named seat(s)"
    )


def _double_booked_seat_action(seat_inventory: dict[str, Any]) -> str:
    rows = seat_inventory.get("double_booked") or []
    labels: list[str] = []
    for row in rows:
        tag = str(row.get("tag") or row.get("seat") or "?")
        workers = row.get("workers") or []
        cap = _int(row.get("session_cap"), 1) or 1
        labels.append(f"{tag}:{len(workers)}/{cap}")
    if not labels:
        return ""
    return (
        f"double_booked={len(rows)} [{_limited_seat_labels(labels)}]; "
        "next action: let one worker finish or reap a dead/stale worker before "
        "launching more on that seat"
    )


def _seat_inventory_summary_line(seat_inventory: dict[str, Any]) -> str:
    if not seat_inventory.get("schema"):
        return ""
    by_state = seat_inventory.get("by_dispatch_state") or {}
    line = (
        f"seat inventory: {seat_inventory.get('total_seats', 0)} seat(s) - "
        f"available={by_state.get('available', 0)} busy={by_state.get('busy', 0)} "
        f"cooling={by_state.get('cooling', 0)} unavailable={by_state.get('unavailable', 0)}"
    )
    free = seat_inventory.get("free_seats")
    leased = seat_inventory.get("leased_seats")
    if free is not None or leased is not None:
        line += f"; slots free={free} leased={leased}"
    unattributed = _int(seat_inventory.get("unattributed_live"), 0)
    if unattributed:
        line += f"; unattributed_live={unattributed}"
    double_booked_action = _double_booked_seat_action(seat_inventory)
    if double_booked_action:
        line += f"; {double_booked_action}"
    auth_action = _auth_failed_seat_action(seat_inventory)
    if auth_action:
        line += f"; {auth_action}"
    return line


def annotate_seat_inventory_from_preflight(seat_inventory: dict[str, Any],
                                           pre: dict[str, Any]) -> dict[str, Any]:
    """Surface conservative preflight accounting on the status card.

    ``fleet_accounts.seat_pool`` can only mark a pool busy when a live worker has an
    ``.account`` sidecar. Preflight also sees product-scoped live workers; when that
    count exceeds sidecar leases, it annotates ``seat.unattributed_live``. Carry that
    signal into the operator inventory without fabricating per-account bindings.
    """
    seat = (pre or {}).get("seat") or {}
    missing = _int(seat.get("unattributed_live"), 0) or 0
    if missing <= 0 or not seat_inventory.get("schema"):
        return seat_inventory
    out = dict(seat_inventory)
    out["unattributed_live"] = missing
    free = _int(out.get("free_seats"))
    if free is not None:
        out["free_seats"] = max(0, free - missing)
    leased = _int(out.get("leased_seats"), 0) or 0
    out["leased_seats"] = max(leased, leased + missing)
    if _int(out.get("free_seats"), 0) == 0:
        out["depleted"] = True
    return out


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
                         route_health=route_health)


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

WATCH_DECISION_SCHEMA = "fleet-watch-decision/1"
WATCH_DEFAULT_WINDOW_HOURS = 6.0
PROGRESS_LOG = "progress.jsonl"

# Selected-route smoke-gate ledger (#3035): `fak dispatch route-health probe` appends
# one fak-route-health/1 row per probe here; this card's fold mirrors the Go
# `fak dispatch route-health status` snapshot (latest row per route).
ROUTE_HEALTH_LOG = "route-health.jsonl"

# Backlog arrival-vs-service rate (#2634). The watch verdict above (#2642) names
# the trend qualitatively; this fold quantifies it — a numeric service rate
# (closes/h), arrival rate (closes + net backlog growth ≈ new issues/h), and a
# BACKLOG_OUTPACED flag that fires only when arrivals demonstrably outrun a
# *working* dispatcher for K consecutive windows. Requiring service > 0 is what
# separates 'outpaced' (a supply/demand trend) from 'stalled' (a malfunction —
# already surfaced by the watch verdict). Informational only: it is the meter,
# never the controller — it launches nothing and touches no worker cap.
BACKLOG_RATE_SCHEMA = "fleet-backlog-rate/1"
BACKLOG_RATE_WINDOW_HOURS = 6.0
BACKLOG_RATE_WINDOWS = 4
BACKLOG_RATE_K = 2


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


def read_dispatch_progress(root: Path) -> list[dict[str, Any]]:
    """The close-arm's per-tick rows (utc, open_now, closed_by_loop_total, ...)
    from ``.dispatch-runs/progress.jsonl``. Pure-local read; [] when absent."""
    log = root / RUNS_DIRNAME / PROGRESS_LOG
    if not log.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    except OSError:
        return []
    return out


def route_health_status(runs_dir: Path, *, now_ts: float | None = None) -> dict[str, Any]:
    """Latest-per-route fold of ``.dispatch-runs/route-health.jsonl`` (#3035) — the
    pure mirror of ``fak dispatch route-health status``: last probe age, typed
    failure class, live-cooldown remaining, and the exact recheck command per
    route/model/account key. Suppression stays per-route: a failing sibling never
    marks the whole provider family. A missing ledger is an empty fold (an
    unprobed fleet must never fail the card) and one corrupt row is skipped, not
    fatal. Pure-local, so --fast keeps it."""
    now = time.time() if now_ts is None else now_ts
    log = runs_dir / ROUTE_HEALTH_LOG
    try:
        text = log.read_text(encoding="utf-8")
    except OSError:
        return {"probed": 0, "suppressed": 0, "routes": []}
    latest: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict):
            continue
        route = str(rec.get("route") or "")
        if not route:
            continue
        prev = latest.get(route)
        if prev is None or (_int(rec.get("probed_at_unix"), 0) or 0) >= (
                _int(prev.get("probed_at_unix"), 0) or 0):
            latest[route] = rec
    routes: list[dict[str, Any]] = []
    suppressed = 0
    for route in sorted(latest):
        rec = latest[route]
        cls = str(rec.get("class") or "")
        probed_at = _int(rec.get("probed_at_unix"), 0) or 0
        cooldown_until = _int(rec.get("cooldown_until_unix"), 0) or 0
        row: dict[str, Any] = {
            "route": route,
            "class": cls,
            "probe_age_secs": int(now - probed_at),
            "suppressed": False,
            "cooldown_remaining_secs": 0,
            "recheck": str(rec.get("recheck") or ""),
        }
        if cls != "healthy" and cooldown_until > now:
            row["suppressed"] = True
            row["cooldown_remaining_secs"] = int(cooldown_until - now)
            suppressed += 1
        routes.append(row)
    return {"probed": len(routes), "suppressed": suppressed, "routes": routes}


def _classify_sched_task(sched: dict[str, Any] | None) -> str:
    """clean / self_healing / unresolved_unknown / unknown for a scheduled-task
    result — never asserts ``malfunction`` without a recovery witness (a red
    LastTaskResult that the next tick recovered is self-healing, #2636)."""
    if not sched or sched.get("installed") is False:
        return "unknown"
    lr = _int(sched.get("last_result"))
    status = str(sched.get("status") or "").strip().lower()
    if lr is not None and lr != 0:
        return "self_healing" if sched.get("recovered") else "unresolved_unknown"
    if lr == 0 or status in ("ready", "running"):
        return "clean"
    return "unknown"


def _watch_row_witness(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "utc": rec.get("utc"),
        "open_now": _int(rec.get("open_now")),
        "closed_by_loop_total": _int(rec.get("closed_by_loop_total")),
        "closed_now": _int(rec.get("closed_now")),
        "audit_error": rec.get("audit_error"),
    }


def _watch_child_failures(breakdown: dict[str, Any] | None) -> dict[str, Any] | None:
    """Compact the #2635 SPAWN_FAILED cause breakdown into the cited child-failure
    summary the watch digest carries (issue #2642 Done condition: the section cites
    'any classified child failures').

    This is the witness for the digest's question 2 — a red scheduled-task bit or a
    SPAWN_FAILED event is a self-healing *child* failure, not a dispatcher
    malfunction, exactly when its noise is ATTRIBUTED to named causes. Folds
    ``by_cause`` down to nonzero {cause: count} with a deterministic ``top_cause``
    (highest count, alphabetical tiebreak). Returns None when nothing was folded
    (no attribution oracle, or a clean window) so the digest simply omits it rather
    than asserting a false zero. Pure and informational — consumes #2635's output,
    computes no attribution of its own."""
    if not isinstance(breakdown, dict):
        return None
    spawn_failed = _int(breakdown.get("spawn_failed"), 0) or 0
    spawns = _int(breakdown.get("spawns"), 0) or 0
    counts = {
        str(cause): n
        for cause, info in (breakdown.get("by_cause") or {}).items()
        if (n := _int((info or {}).get("count"), 0) or 0) > 0
    }
    if spawn_failed <= 0 and not counts:
        return None
    top_cause = min(counts, key=lambda c: (-counts[c], c)) if counts else None
    return {
        "spawns": spawns,
        "spawn_failed": spawn_failed,
        "rate": breakdown.get("rate"),
        "by_cause": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "top_cause": top_cause,
    }


def _sched_recovered(progress_records: list[dict[str, Any]], *, now_ts: float,
                     window_hours: float = WATCH_DEFAULT_WINDOW_HOURS) -> bool:
    """Did the loop demonstrably advance closures in the trailing window?

    This is the 'next tick recovered' witness the ef59064f watch applied by hand
    (#2642): a red scheduled-task ``LastTaskResult`` is reclassified self-healing
    only when the loop kept closing issues after it — otherwise it stays
    ``unresolved_unknown``. ``closed_by_loop_total`` is monotonic, so max-min > 0
    over the in-window rows is an order-independent 'advanced' signal."""
    cutoff = now_ts - window_hours * 3600.0
    totals: list[int] = []
    closes = 0
    for rec in progress_records or []:
        if not isinstance(rec, dict):
            continue
        ep = _iso_epoch(rec.get("utc"))
        if ep is None or ep < cutoff or ep > now_ts + 3600.0:
            continue
        lt = _int(rec.get("closed_by_loop_total"))
        if lt is not None:
            totals.append(lt)
        closes += _int(rec.get("closed_now")) or 0
    advanced = bool(totals) and (max(totals) - min(totals) > 0)
    return bool(advanced or closes > 0)


def watch_decision(progress_records: list[dict[str, Any]], *, now_ts: float,
                   window_hours: float = WATCH_DEFAULT_WINDOW_HOURS,
                   sched_task: dict[str, Any] | None = None,
                   follow_ups: list[Any] | None = None,
                   closure_status: str | None = None,
                   idle_floor: int = 0) -> dict[str, Any]:
    """Fold a trailing window of dispatch-progress rows into a cited verdict that
    explains the monitor's decision, especially the intentional no-op:

      OUTPACED     — closures advance but backlog rises (arrivals outpace
                     service); healthy-busy, keep watching, do NOT hand-launch.
      DRAINING     — closures advance and backlog falls; healthy, no action.
      KEEPING_PACE — closures advance and backlog holds; no action.
      STALLED      — no closures while a backlog persists/rises; a candidate for
                     an operator look (surfaced, never auto-actioned).
      HEALTHY_IDLE — no closures because the backlog is at/below ``idle_floor``.
      INSUFFICIENT_DATA — fewer than two rows land in the window.

    Pure and informational only (issue #2642): the verdict is a report, it never
    dispatches work or changes caps. ``action_needed`` merely flags STALLED for a
    human glance.
    """
    cutoff = now_ts - window_hours * 3600.0
    rows: list[tuple[float, dict[str, Any]]] = []
    for rec in progress_records or []:
        if not isinstance(rec, dict):
            continue
        ep = _iso_epoch(rec.get("utc"))
        if ep is None or ep < cutoff or ep > now_ts + 3600.0:
            continue
        rows.append((ep, rec))
    rows.sort(key=lambda kv: kv[0])

    follow = sorted({int(n) for n in (follow_ups or []) if _int(n) is not None})
    sched = dict(sched_task or {})
    sched.setdefault("classification", _classify_sched_task(sched))

    base = {
        "schema": WATCH_DECISION_SCHEMA,
        "window_hours": float(window_hours),
        "rows_in_window": len(rows),
        "scheduled_task": sched,
        "follow_ups": follow,
    }

    if len(rows) < 2:
        base.update({
            "verdict": "INSUFFICIENT_DATA",
            "action_needed": False,
            "why": (f"fewer than two progress rows in the trailing "
                    f"{window_hours:g}h window — cannot judge a trend"),
            "audit_status": closure_status,
            "cited": {},
            "witness_rows": [_watch_row_witness(r) for _e, r in rows],
        })
        return base

    first, last = rows[0][1], rows[-1][1]
    open_start = _int(first.get("open_now"))
    open_end = _int(last.get("open_now"))
    loop_start = _int(first.get("closed_by_loop_total"))
    loop_end = _int(last.get("closed_by_loop_total"))
    open_delta = (open_end - open_start
                  if open_start is not None and open_end is not None else None)
    loop_delta = (loop_end - loop_start
                  if loop_start is not None and loop_end is not None else None)
    closes_in_window = sum(
        c for _e, r in rows if (c := _int(r.get("closed_now")) or 0) > 0)
    closing = bool((loop_delta or 0) > 0 or closes_in_window > 0)

    if closure_status is not None:
        audit_status = closure_status
    else:
        err = last.get("audit_error")
        audit_status = "clean" if err in (None, "", "null") else f"error: {err}"

    if closing:
        if open_delta is not None and open_delta > 0:
            verdict, action = "OUTPACED", False
            why = (f"dispatcher advanced closures (closed_by_loop_total "
                   f"{loop_start}->{loop_end}, +{closes_in_window} witnessed) while "
                   f"the backlog rose (open_now {open_start}->{open_end}, "
                   f"+{open_delta}): new arrivals outpace service, not a stalled "
                   f"dispatcher — keep watching, do not hand-launch work")
        elif open_delta is not None and open_delta < 0:
            verdict, action = "DRAINING", False
            why = (f"dispatcher advanced closures (closed_by_loop_total "
                   f"{loop_start}->{loop_end}) and the backlog fell (open_now "
                   f"{open_start}->{open_end}, {open_delta}): healthy drain, no action")
        else:
            verdict, action = "KEEPING_PACE", False
            why = (f"dispatcher advanced closures (closed_by_loop_total "
                   f"{loop_start}->{loop_end}) and the backlog held near "
                   f"{open_end}: keeping pace, no action")
    else:
        if (open_end or 0) <= idle_floor and (open_delta or 0) <= 0:
            verdict, action = "HEALTHY_IDLE", False
            why = (f"no closures over the window and the backlog sits at "
                   f"{open_end} (<= idle_floor {idle_floor}): nothing to close, "
                   f"healthy-idle, no action")
        else:
            verdict, action = "STALLED", True
            trend = (f"rose +{open_delta}" if (open_delta or 0) > 0
                     else f"held at {open_end}")
            why = (f"no closures over the window (closed_by_loop_total flat at "
                   f"{loop_end}) while the backlog {trend}: dispatcher may be "
                   f"stalled — a candidate for an operator look (informational)")

    base.update({
        "verdict": verdict,
        "action_needed": action,
        "why": why,
        "audit_status": audit_status,
        "cited": {
            "open_now_start": open_start,
            "open_now_end": open_end,
            "open_now_delta": open_delta,
            "closed_by_loop_total_start": loop_start,
            "closed_by_loop_total_end": loop_end,
            "closed_by_loop_total_delta": loop_delta,
            "closed_now_sum": closes_in_window,
            "first_utc": first.get("utc"),
            "last_utc": last.get("utc"),
        },
        "witness_rows": [_watch_row_witness(first), _watch_row_witness(last)],
    })
    return base


def _epoch_iso(ep: float | None) -> str | None:
    """Inverse of ``_iso_epoch`` — a naive-UTC ``...Z`` stamp for a window edge."""
    if ep is None:
        return None
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ep))
    except (ValueError, OSError):
        return None


def _backlog_anchor(parsed: list[tuple[float, int, int]],
                    t: float) -> tuple[float, int, int] | None:
    """The newest ``(epoch, open_now, closed_by_loop_total)`` row at or before ``t``.

    ``parsed`` is epoch-sorted, so a window edge anchors on the last-known counter
    values as of that instant — giving each sub-window an exact ``window_hours``
    span rather than the span between whichever raw rows happened to land inside it.
    """
    best: tuple[float, int, int] | None = None
    for row in parsed:
        if row[0] <= t:
            best = row
        else:
            break
    return best


def backlog_rates(progress_records: list[dict[str, Any]], *, now_ts: float,
                  window_hours: float = BACKLOG_RATE_WINDOW_HOURS,
                  windows: int = BACKLOG_RATE_WINDOWS,
                  k_consecutive: int = BACKLOG_RATE_K) -> dict[str, Any]:
    """Fold trailing progress rows into arrival-rate vs service-rate (#2634).

    Anchoring on the newest progress row at or before each of the trailing
    ``windows`` window edges, each ``window_hours``-long sub-window yields:

      * service rate = Δ``closed_by_loop_total`` / window_hours       (closes/h)
      * arrival rate = (Δ``open_now`` + Δ``closed_by_loop_total``) / window_hours
        (closes + net backlog growth ≈ new arrivals/h)

    ``backlog_outpaced`` fires only when the newest ``k_consecutive`` computable
    windows each show arrival > service AND service > 0 — the dispatcher is
    demonstrably *closing* work yet arrivals still outrun it. The service > 0
    clause is exactly what tells 'outpaced' apart from 'stalled': a rising
    backlog with zero service is STALLED, not OUTPACED.

    Pure and informational only (this is the meter, not the controller): the
    verdict/flag is a report; it never dispatches work or changes worker caps.
    """
    parsed: list[tuple[float, int, int]] = []
    for rec in progress_records or []:
        if not isinstance(rec, dict):
            continue
        ep = _iso_epoch(rec.get("utc"))
        open_now = _int(rec.get("open_now"))
        closed = _int(rec.get("closed_by_loop_total"))
        if ep is None or open_now is None or closed is None:
            continue
        parsed.append((ep, open_now, closed))
    parsed.sort(key=lambda r: r[0])

    w_secs = window_hours * 3600.0
    # boundaries[0]=now (newest edge) .. boundaries[windows]=oldest edge.
    boundaries = [now_ts - i * w_secs for i in range(windows + 1)]
    anchors = [_backlog_anchor(parsed, t) for t in boundaries]

    per_window: list[dict[str, Any]] = []
    for i in range(windows):
        newer, older = anchors[i], anchors[i + 1]
        row: dict[str, Any] = {
            "index": i,
            "start_utc": _epoch_iso(boundaries[i + 1]),
            "end_utc": _epoch_iso(boundaries[i]),
        }
        if newer is None or older is None or newer[0] <= older[0]:
            row["computable"] = False
            per_window.append(row)
            continue
        open_delta = newer[1] - older[1]
        closed_delta = newer[2] - older[2]
        service_rate = closed_delta / window_hours
        arrival_rate = (open_delta + closed_delta) / window_hours
        row.update({
            "computable": True,
            "open_delta": open_delta,
            "closed_delta": closed_delta,
            "arrival_rate_per_hour": round(arrival_rate, 3),
            "service_rate_per_hour": round(service_rate, 3),
            "outpaced": arrival_rate > service_rate and service_rate > 0,
            "serviced": service_rate > 0,
        })
        per_window.append(row)

    # Consecutive run of outpaced windows counted from the newest edge inward.
    run = 0
    for win in per_window:
        if win.get("computable") and win.get("outpaced"):
            run += 1
        else:
            break
    backlog_outpaced = run >= k_consecutive

    base: dict[str, Any] = {
        "schema": BACKLOG_RATE_SCHEMA,
        "window_hours": float(window_hours),
        "windows": int(windows),
        "k_consecutive": int(k_consecutive),
        "consecutive_outpaced_windows": run,
        "backlog_outpaced": backlog_outpaced,
        "per_window": per_window,
    }

    # Headline aggregate rates over the full evaluated span (oldest→newest anchor).
    non_null = [a for a in anchors if a is not None]
    if len(non_null) >= 2 and non_null[0][0] > non_null[-1][0]:
        newest_a, oldest_a = non_null[0], non_null[-1]
        elapsed_h = (newest_a[0] - oldest_a[0]) / 3600.0
        open_delta = newest_a[1] - oldest_a[1]
        closed_delta = newest_a[2] - oldest_a[2]
        service_rate = closed_delta / elapsed_h
        arrival_rate = (open_delta + closed_delta) / elapsed_h
        if service_rate <= 0 and arrival_rate > 0:
            verdict = "STALLED"
        elif arrival_rate > service_rate:
            verdict = "OUTPACED"
        elif arrival_rate < service_rate:
            verdict = "DRAINING"
        else:
            verdict = "KEEPING_PACE"
        flag_bit = (f"; BACKLOG_OUTPACED ({run} consecutive outpaced window(s) "
                    f">= K={k_consecutive})") if backlog_outpaced else ""
        base.update({
            "arrival_rate_per_hour": round(arrival_rate, 3),
            "service_rate_per_hour": round(service_rate, 3),
            "span_hours": round(elapsed_h, 3),
            "verdict": verdict,
            "why": (f"arrival {arrival_rate:.2f}/h vs service {service_rate:.2f}/h "
                    f"over {elapsed_h:.1f}h: {verdict.lower()}{flag_bit}"),
        })
    else:
        base.update({
            "arrival_rate_per_hour": None,
            "service_rate_per_hour": None,
            "span_hours": None,
            "verdict": "INSUFFICIENT_DATA",
            "why": (f"fewer than two anchorable progress rows across the trailing "
                    f"{windows}x{window_hours:g}h windows — cannot derive rates"),
        })
    return base


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
    else:
        ok = True
        verdict = "READY_TO_GROW"
        reasons.append(f"safe to spawn: {live}/{cap} live, account '{acct.get('tag')}' free")

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
            f"({guard.get('empty_sessions', 0)} empty) — workers booted under guard "
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
                  route_health: dict[str, Any] | None = None) -> dict[str, Any]:
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

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "reasons": reasons,
        "workspace": str(root),
        "weekly_cap": weekly_cap,
        "dispatcher": {
            "cap": cap,
            "live": live,
            "headroom": (cap - live) if (cap is not None and live is not None) else None,
            "host_safe": host_safe,
            "preflight_verdict": pre_verdict,
            "limiter": limiter,
            "account": {k: acct.get(k) for k in ("tag", "tier", "model", "available")},
            "watchdog": wd,
        },
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


def _age_text(minutes: Any) -> str:
    if not isinstance(minutes, (int, float)):
        return "?"
    if minutes < 1:
        return "<1m"
    if minutes >= 60:
        hours = f"{minutes / 60:.1f}".rstrip("0").rstrip(".")
        return hours + "h"
    return f"{minutes:.1f}".rstrip("0").rstrip(".") + "m"


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
    return lines


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
            f"QUAR={gd.get('quarantined', 0)}{crash_bit}]  empty={gd.get('empty_sessions', 0)}{loop_bit}")
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
            f"{gd.get('empty_sessions', 0)}",
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


def _dispatch_slack_buckets(payload: dict[str, Any]) -> dict[str, list[str]]:
    d = payload.get("dispatcher") or {}
    a = d.get("account") or {}
    wd = d.get("watchdog") or {}
    sup = payload.get("supervisor") or {}
    tp = payload.get("throughput") or {}
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
