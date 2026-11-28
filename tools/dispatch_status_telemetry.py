#!/usr/bin/env python3
"""Telemetry, guard coverage, seat inventory, yield, and watch-decision folds.

Separated from ``tools/dispatch_status.py`` so spawn-cause, seat-streak,
ships-per-worker, watchdog, backend/hook health, guard-audit, seat inventory,
watch-decision, and backlog-rate folds live cleanly below the file-size ceiling.
"""
from __future__ import annotations

import bisect
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import dispatch_preflight
from dispatch_status_common import (
    RUNS_DIRNAME,
    WATCHDOG_TASK,
    _RESOLVE_LOG_RE,
    _STUB_LOG_MAX_BYTES,
    _epoch_iso,
    _int,
    _iso_epoch,
    _win_creationflags,
)


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


# --- Empty-guard-session cause split (#5862 done-condition 2) --------------------
# `empty=N` on the guard card used to be one opaque number: some sessions booted under
# guard and adjudicated nothing, but not WHY -- so an operator could not tell a fleet
# parked behind a provider wall (benign, the supervisor resumes the same command) from
# one that is silently failing to spawn (not benign at all). These name the causes.
#
# The terminal-row kinds. A guard session's LAST row can be a supervisor-written child
# exit witness (internal/journal.KindChildExit / KindChildCrash) rather than an
# adjudication. It is a genuine chained row but carries NO verdict, so it must not
# count as "this session adjudicated something" -- see _GUARD_EMPTY note below.
_GUARD_TERMINAL_KINDS = ("CHILD_EXIT", "CHILD_CRASH")


# The child-exit reason vocabulary, lowercased from internal/journal/crash.go's Crash*
# constants plus guardCrashRestartExhaustedReason (cmd/fak/guard_crash_restart.go).
# A reason outside this set folds to "unknown" rather than inventing a bucket -- the
# same closed-set discipline internal/dispatchconservation.noCommitReasons enforces on
# the Go side. Kept as a literal COPY, not an import (Python cannot import Go); the
# drift test parses the Go source and fails CI when the two sides diverge.
_GUARD_EMPTY_WITNESS_REASONS = (
    "clean_exit", "signal_crash", "oom", "nonzero_exit", "rate_limit_exit",
    "crash_restart_exhausted",
)


# Causes derived from the worker log tail. These are literal copies of
# internal/dispatchconservation's Reason* constants so the guard card and the
# conservation ledger name the SAME outcome with the SAME string; a cause this repo
# already knows must not be re-spelled here.
_GUARD_EMPTY_PROVIDER_QUOTA_WALL = "provider_quota_wall"


_GUARD_EMPTY_SPAWN_FAILED = "guard_child_spawn_failed"


_GUARD_EMPTY_DIED_BEFORE_EPILOGUE = "died_before_epilogue"


_GUARD_EMPTY_CLEAN_EXIT_NO_COMMIT = "clean_exit_no_commit"


_GUARD_EMPTY_MISSING_LOG = "missing_log_artifact"


# Not a dispatch spawn at all. guardDefaultAuditPath (cmd/fak/guard_support.go) names
# the journal `interactive-<pid>-<repo-root-hash>.jsonl` when the operator named no
# --audit path and set no FAK_AUDIT_JOURNAL -- i.e. an attended/ad-hoc `fak guard`,
# which consumed NO dispatch slot, NO account rotation and NO spawn. Folding these in
# with dispatch sessions is what made `empty=N` read as spawn waste when it was not.
_GUARD_EMPTY_INTERACTIVE = "interactive_not_dispatch"


_GUARD_EMPTY_UNKNOWN = "unknown"


_GUARD_INTERACTIVE_PREFIX = "interactive-"


# Marker literals, copied from internal/dispatchconservation/conservation.go (which
# documents why each was chosen over a looser one -- e.g. LONG_RETRY_AFTER rather than
# a bare "429", which also matches a worker that merely READ a rate-limit source file).
_GUARD_PARK_MARKER = "LONG_RETRY_AFTER"


_GUARD_SPAWN_FAILED_MARKER = "fak guard: could not run"


_GUARD_EPILOGUE_MARKER = "── guard · "


# The guard epilogue prints AFTER the terminal marker, at a median offset of ~7.2 KiB.
# A 4 KiB tail read misses every marker and returns a confident wrong answer, so read a
# tail with real margin over that median.
_GUARD_EMPTY_LOG_TAIL_BYTES = 32768


# Journal mtime vs worker-log mtime for the same session is exact in practice (max
# observed skew 14s over the 26 joinable empties in this repo's corpus), so a 10-minute
# window is generous. Joining on the two FILE mtimes also dodges the trap that bit an
# earlier pass: the resolve-log FILENAME stamp is UTC while file times are local, so a
# name-derived correlation silently finds nothing.
_GUARD_EMPTY_LOG_WINDOW_S = 600.0


# Hard cap on log tails read per fold so the card stays cheap on a large .dispatch-runs.
_GUARD_EMPTY_LOG_SCAN_CAP = 400


# dispatch-<lane>-<stamp>.log written by issue_dispatch.spawn_detached — the WAVE
# spawn path. It was outside this fold's population entirely, which is how a wave that
# lost every child could still read `spawns=8 spawn_failed=0` (#6492): the denominator
# saw only per-issue resolve spawns and the numerator saw only their silent logs.
_WAVE_LOG_RE = re.compile(r"dispatch-(.+)-(\d{8}-\d{6})\.log$")


# The terminal record issue_dispatch.retire_failed_spawn writes beside the log of a
# child that died inside the wave's read-back window. It is a POSITIVE witness (the
# tick watched the child exit), so it is folded whatever the log's pid state says.
_WAVE_SPAWN_FAILED_GLOB = "dispatch-*.spawn-failed.json"


_BACKEND_STUB_LOOKBACK_MIN = 90


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


def _wave_spawn_failed_events(runs_dir: Path, *, horizon: float) -> list[dict[str, Any]]:
    """The wave path's TERMINAL spawn-failure records inside the window (#6492).

    ``issue_dispatch.retire_failed_spawn`` writes one of these beside the log of every
    child that was already dead when the dispatching tick read it back, carrying the
    cause it classified with the SAME #2635 classifier used above. Reading the record
    (rather than re-deriving from a log tail) is what makes a wave's launch failures
    countable at all: a wave child that dies before writing a byte leaves a 0-byte log
    with no pid sidecar, which the ``silent_workers`` fold never enumerated.

    Read-only + FAIL-OPEN: an unreadable or malformed record is skipped, never fatal.
    Newest first, matching the caller's evidence ordering."""
    rows: list[tuple[float, dict[str, Any]]] = []
    try:
        records = sorted(runs_dir.glob(_WAVE_SPAWN_FAILED_GLOB))
    except OSError:
        return []
    for path in records:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime < horizon:
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(rec, dict):
            continue
        rows.append((mtime, {
            "issue": None,
            "lane": rec.get("lane"),
            "log": Path(str(rec.get("log") or path)).name,
            "stamp": rec.get("stamp"),
            "size": int(rec.get("log_bytes") or 0),
            "returncode": rec.get("returncode"),
            "cause": str(rec.get("cause") or "unknown"),
            "source": "wave_spawn_failed_record",
        }))
    rows.sort(key=lambda r: r[0], reverse=True)
    return [row for _, row in rows]


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

    Denominator = every ``resolve-*.log`` AND every wave ``dispatch-*.log`` inside the
    window (each is one spawn), so ``rate`` is spawn_failed / spawns. The wave path
    contributes to BOTH sides (#6492): its launches to the denominator, and its
    terminal ``*.spawn-failed.json`` read-back records to the numerator — before that,
    a wave whose children all died on launch still read ``spawn_failed=0``. Scope
    note: a crash that dumps a LARGE log
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
    # Denominator: every spawn whose mtime is inside the window — the per-issue resolve
    # logs AND (#6492) the wave path's dispatch logs, so a wave's launches are inside
    # the same population as its failures instead of only the failures' denominator.
    spawns = 0
    for pattern, name_re in (("resolve-*.log", _RESOLVE_LOG_RE),
                             ("dispatch-*.log", _WAVE_LOG_RE)):
        for log in runs_dir.glob(pattern):
            if not name_re.search(log.name):
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
    # Numerator, part two (#6492): the wave path's TERMINAL failure records. These need
    # no pid oracle and no stub-floor inference — the dispatching tick watched the child
    # exit inside its read-back window and wrote down the returncode, size and cause.
    # Without them a wave that lost every child reported zero spawn failures.
    for ev in _wave_spawn_failed_events(runs_dir, horizon=horizon):
        events.append(ev)
        bucket = by_cause.setdefault(str(ev.get("cause")), {"count": 0, "evidence": []})
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


def _guard_empty_cause_bit(guard: dict[str, Any]) -> str:
    """``(cause=n, cause=n)`` for the empty-session split, worst cause first, or "".

    One renderer for all three guard-card surfaces (the box line, the reasons list and
    the markdown report) so the three can never disagree about what ``empty=N`` meant.
    Empty string when there is nothing to explain — a fleet with no empty sessions
    should not grow a bucket list that is all zeroes."""
    by_cause = guard.get("empty_by_cause") or {}
    if not by_cause:
        return ""
    bits = ", ".join(f"{k}={v}" for k, v in by_cause.items())
    return f" ({bits})"


def _guard_worker_log_index(runs_dir: Path) -> list[tuple[float, Path]]:
    """``(mtime, path)`` for every worker log under ``.dispatch-runs``, oldest first.

    Only ``stat()`` -- no reads. The reads are paid lazily, per empty session, over the
    tiny mtime window ``_guard_empty_log_cause`` selects."""
    out: list[tuple[float, Path]] = []
    try:
        for lp in runs_dir.glob("*.log"):
            try:
                out.append((lp.stat().st_mtime, lp))
            except OSError:
                continue
    except OSError:
        return []
    out.sort(key=lambda r: r[0])
    return out


def _guard_log_tail(path: Path) -> str:
    """The last ``_GUARD_EMPTY_LOG_TAIL_BYTES`` of a worker log, decoded leniently."""
    try:
        with path.open("rb") as fh:
            try:
                fh.seek(-_GUARD_EMPTY_LOG_TAIL_BYTES, os.SEEK_END)
            except OSError:
                fh.seek(0)
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _guard_cause_from_tail(tail: str) -> str:
    """Name an empty session's cause from its worker log tail, worst-first.

    Order matters and is not arbitrary: a session can carry BOTH a park and an
    epilogue, and the park is the cause while the epilogue is merely evidence the guard
    shut down tidily afterwards."""
    if _GUARD_PARK_MARKER in tail:
        return _GUARD_EMPTY_PROVIDER_QUOTA_WALL
    if _GUARD_SPAWN_FAILED_MARKER in tail:
        return _GUARD_EMPTY_SPAWN_FAILED
    if _GUARD_EPILOGUE_MARKER not in tail:
        return _GUARD_EMPTY_DIED_BEFORE_EPILOGUE
    return _GUARD_EMPTY_CLEAN_EXIT_NO_COMMIT


def _guard_empty_log_cause(
    name: str,
    mtime: float,
    log_index: list[tuple[float, Path]],
    budget: list[int],
) -> str:
    """Join one empty journal to its worker log by mtime proximity, then classify it.

    ``budget`` is a one-element list used as a mutable read counter shared across the
    fold, so a pathological ``.dispatch-runs`` cannot turn the status card into a
    full-corpus scan: once ``_GUARD_EMPTY_LOG_SCAN_CAP`` tails have been read the rest
    degrade to ``missing_log_artifact`` (an honest "we did not look" rather than a
    fabricated cause)."""
    lo = bisect.bisect_left(log_index, (mtime - _GUARD_EMPTY_LOG_WINDOW_S,))
    hi = bisect.bisect_right(log_index, (mtime + _GUARD_EMPTY_LOG_WINDOW_S, Path("￿")))
    for _, lp in log_index[lo:hi]:
        if budget[0] <= 0:
            break
        budget[0] -= 1
        tail = _guard_log_tail(lp)
        if name in tail:
            return _guard_cause_from_tail(tail)
    return _GUARD_EMPTY_MISSING_LOG


def _guard_empty_cause(
    name: str,
    witness_reason: str,
    mtime: float,
    log_index: list[tuple[float, Path]],
    budget: list[int],
) -> str:
    """The cause for ONE decision-empty guard session, cheapest evidence first.

    1. A supervisor-written child-exit witness row IS the answer, and costs nothing --
       it was already parsed out of the journal. This is the path #5862's shipped
       producer fix (ae47d2911d, the exit witness on parked teardowns) feeds, so the
       breakdown gets sharper on its own as the fleet picks up that build.
    2. Otherwise the journal NAME already settles whether this was a dispatch spawn at
       all: `interactive-*` is guardDefaultAuditPath, an attended run.
    3. Only then pay for a log tail.

    A witness row whose Reason is absent or outside the closed set does NOT short to
    "unknown": older CHILD_CRASH rows predate the Reason field entirely, and stamping
    them "unknown" would throw away the name and log evidence that can still explain
    them. "unknown" is the last resort, never a shortcut."""
    low = witness_reason.strip().lower()
    if low in _GUARD_EMPTY_WITNESS_REASONS:
        return low
    if name.startswith(_GUARD_INTERACTIVE_PREFIX):
        return _GUARD_EMPTY_INTERACTIVE
    cause = _guard_empty_log_cause(name, mtime, log_index, budget)
    if cause == _GUARD_EMPTY_MISSING_LOG and low:
        # A terminal row exists but names nothing we recognise and no log survives.
        return _GUARD_EMPTY_UNKNOWN
    return cause


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
      * ``empty_sessions`` — journals with 0 DECISION rows (booted under guard but
        proposed no adjudicated tool call — the silent empty-turn signature). A
        supervisor-written child-exit witness row (CHILD_EXIT/CHILD_CRASH) is NOT a
        decision: it carries no verdict, it is written by the guard at teardown rather
        than by the child, and counting it would let #5862's own producer fix
        (ae47d2911d, which adds that row on parked teardowns) silently DEFLATE this
        number without explaining a single session. ``zero_row_sessions`` keeps the
        older literal "the file is empty" count visible beside it.
      * ``empty_by_cause`` — that count split by a CLOSED cause vocabulary (#5862
        done-condition 2), so an operator can tell a fleet parked behind a provider
        wall from one that is failing to spawn. See ``_guard_empty_cause``.
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
        "zero_row_sessions": 0,
        "empty_by_cause": {},
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
    # (journal stem, witness reason, mtime) for every DECISION-empty session, resolved
    # to causes after the walk so the log index is built at most once and only when
    # there is something to explain.
    empties: list[tuple[str, str, float]] = []
    for jp in audit_dir.glob("*.jsonl"):
        try:
            mtime = jp.stat().st_mtime
            text = jp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rows = 0
        decisions = 0
        witness_reason = ""
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                row = json.loads(line)
                kind = str(row.get("kind") or "UNKNOWN")
            except ValueError:
                row, kind = {}, "MALFORMED"
            if kind in _GUARD_TERMINAL_KINDS:
                # Last one wins: the terminal witness is written at teardown, so a
                # restarted session's FINAL exit is the one that explains the session.
                witness_reason = str(row.get("reason") or "") or _GUARD_EMPTY_UNKNOWN
            else:
                decisions += 1
            by_kind[kind] = by_kind.get(kind, 0) + 1
        payload["sessions"] += 1
        payload["rows"] += rows
        if rows == 0:
            payload["zero_row_sessions"] += 1
        if decisions == 0:
            payload["empty_sessions"] += 1
            empties.append((jp.name[:-6] if jp.name.endswith(".jsonl") else jp.name,
                            witness_reason, mtime))
        if mtime >= horizon:
            payload["recent_sessions"] += 1
            payload["recent_rows"] += rows
            livelock_candidates.extend(_guard_livelock_candidates(jp.name, text))
        files.append((mtime, jp.name, rows))

    if empties:
        log_index = _guard_worker_log_index(runs_dir)
        budget = [_GUARD_EMPTY_LOG_SCAN_CAP]
        by_cause: dict[str, int] = {}
        for name, witness_reason, mtime in empties:
            cause = _guard_empty_cause(name, witness_reason, mtime, log_index, budget)
            by_cause[cause] = by_cause.get(cause, 0) + 1
        payload["empty_by_cause"] = dict(
            sorted(by_cause.items(), key=lambda kv: (-kv[1], kv[0])))

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
