#!/usr/bin/env python3
"""Lease state, worker cross-check, lane-lease holder liveness, and run-status digests.

Separated from ``tools/dispatch_status.py`` so lease-ref scanning, active dispatch
worker scanning, process-start-time holder probing, and run-status digest folding stay
cohesive and under the file-size gate.
"""
from __future__ import annotations

import json
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import dispatch_preflight
from dispatch_status_common import (
    _RESOLVE_LOG_RE,
    _STUB_LOG_MAX_BYTES,
    _age_text,
    _int,
    _last_json,
    _win_creationflags,
)


_LEASEREF_PREFIX = "refs/fak/locks/"


_NOOP_BANNER_RE = re.compile(r"(?i)>\s*build\s*[·:]")


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
