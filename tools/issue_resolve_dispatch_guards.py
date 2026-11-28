#!/usr/bin/env python3
"""Active livelock/compact holds, weekly cap, backend health, seat/gateway/argv gates, and lane leases."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any

from issue_resolve_dispatch_core import (
    ACCOUNT_SIDECAR_SUFFIX,
    ACTIVE_COMPACT_RUNAWAY_MIN_COUNT,
    ACTIVE_COMPACT_RUNAWAY_MIN_PAST_K,
    ACTIVE_COMPACT_RUNAWAY_SHED_K,
    ACTIVE_GUARD_LIVELOCK_ESCAPE_ROWS,
    ACTIVE_GUARD_LIVELOCK_MIN_COUNT,
    DEFAULT_GUARD_PROBE_TTL_H,
    LEASE_DRAIN_TIMEOUT_S,
    LEASE_ID_PREFIX,
    LEASE_REFUSED_RC,
    _ANNOUNCED_WAIT_DUR_RE,
    _ANNOUNCED_WAIT_RE,
    _AUDIT_LOG_RE,
    _AUTH_GAP_RE,
    _BACKEND_AUTHGAP_REPROBE_MIN,
    _BACKEND_DEAD_STREAK,
    _BACKEND_REPROBE_MIN,
    _CAP_BANNER_RE,
    _CAP_FIVE_HOUR_RE,
    _CAP_RESET_FALLBACK_RE,
    _CAP_RESET_RE,
    _CAP_SESSION_RE,
    _CAP_TAIL_BYTES,
    _CAP_USAGE_RE,
    _CAP_WEEKLY_RE,
    _CODEX_RESET_RE,
    _COMPACT_FIELD_RE,
    _CTX_BUDGET_RE,
    _DIST_PAST_COMPACT_RE,
    _GUARD_FLAG_DEF_RE,
    _GUARD_PROBE_LEDGER,
    _GUARD_PROBE_MIN_FLAGS,
    _GUARD_PROBE_PEELED,
    _GUARD_PROBE_TIMEOUT_S,
    _HEALTH_SKIP_STUB_RATE,
    _LEASE_SESSION_ID_RE,
    _LEASE_SESSION_SAFE_RE,
    _LOG_ISSUE_RE,
    _MONTHS,
    _PT_TO_UTC,
    _RESET_AT_RE,
    _ROLLING_HOLD_MAX_MIN,
    _SESSION_HOLD_MAX_MIN,
    _STUB_LOG_MAX_BYTES,
    _WEEKDAYS,
    _WEEKLY_FALLBACK_MIN,
    _WILL_RESET_LA_RE,
    _WILL_RESET_RE,
    _spawn_header_lane,
    account_probe,
    dispatch_preflight,
    ird_attr,
    no_window_creationflags,
    split_command_env,
)


def _audit_path_from_log(log: Path, root: Path) -> Path | None:
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    found: Path | None = None
    for match in _AUDIT_LOG_RE.finditer(text):
        raw = match.group(1).strip().strip("`'\"")
        if not raw:
            continue
        cand = Path(raw)
        if not cand.is_absolute():
            cand = root / cand
        found = cand
    return found


def _guard_result_livelock(
    journal: Path,
    *,
    min_count: int = ACTIVE_GUARD_LIVELOCK_MIN_COUNT,
    escape_rows: int = ACTIVE_GUARD_LIVELOCK_ESCAPE_ROWS,
) -> dict[str, Any]:
    """Return positive evidence for repeated quarantined tool_result rows."""
    counts: dict[tuple[str, str, str], int] = {}
    last_row: dict[tuple[str, str, str], int] = {}
    escapes = 0
    row_idx = 0
    try:
        lines = journal.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {"livelock": False, "unavailable": True, "path": str(journal)}
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        row_idx += 1
        verdict = str(row.get("verdict") or "").upper()
        tool = str(row.get("tool") or row.get("name") or "")
        if verdict != "QUARANTINE" or tool != "tool_result":
            continue
        reason = str(row.get("reason") or "")
        digest = str(row.get("args_digest") or row.get("digest") or "")
        if not digest:
            continue
        key = (tool, reason, digest)
        prev = last_row.get(key)
        if prev is not None and row_idx - prev > escape_rows:
            counts[key] = 0
            escapes += 1
        counts[key] = counts.get(key, 0) + 1
        last_row[key] = row_idx
    for key, idx in last_row.items():
        if row_idx - idx > escape_rows:
            counts[key] = 0
            escapes += 1
    if not counts:
        return {"livelock": False, "path": str(journal), "rows": row_idx}
    (tool, reason, digest), count = max(counts.items(), key=lambda item: item[1])
    if count < min_count:
        return {"livelock": False, "path": str(journal), "count": count,
                "rows": row_idx, "escapes": escapes}
    return {
        "livelock": True,
        "path": str(journal),
        "tool": tool,
        "reason": reason,
        "digest": digest,
        "count": count,
        "rows": row_idx,
        "escapes": escapes,
        "rows_since": row_idx - last_row[(tool, reason, digest)],
    }


def active_guard_livelock_hold(root: Path, runs_dir: Path, *,
                               min_count: int = ACTIVE_GUARD_LIVELOCK_MIN_COUNT) -> dict[str, Any]:
    """Positive live-worker evidence that should pause new resolver spawns."""
    dp_mod = ird_attr("dispatch_preflight", dispatch_preflight)
    livelock_fn = ird_attr("_guard_result_livelock", _guard_result_livelock)
    candidates: list[dict[str, Any]] = []
    for log in sorted(runs_dir.glob("resolve-*.log")):
        pid_file = log.with_suffix(".pid")
        if not pid_file.exists():
            continue
        if not dp_mod.resolve_sidecar_pid_is_live(pid_file):
            continue
        audit = _audit_path_from_log(log, root)
        if not audit:
            continue
        hit = livelock_fn(audit, min_count=min_count)
        if not hit.get("livelock"):
            continue
        issue = None
        m = _LOG_ISSUE_RE.search(log.name)
        if m:
            try:
                issue = int(m.group(1))
            except ValueError:
                issue = None
        hit.update({"log": str(log), "issue": issue,
                    "lane": _spawn_header_lane(log)})
        candidates.append(hit)
    if not candidates:
        return {"active": False}
    candidates.sort(key=lambda h: int(h.get("count") or 0), reverse=True)
    top = candidates[0]
    lane_bit = f" on lane '{top.get('lane')}'" if top.get("lane") else ""
    return {
        "active": True,
        "candidates": candidates[:5],
        "lanes": sorted({str(c.get("lane")) for c in candidates if c.get("lane")}),
        "lane_unknown": any(not c.get("lane") for c in candidates),
        "reason": (
            f"live worker #{top.get('issue') or '?'}{lane_bit} is already in a "
            f"guard result livelock: {top.get('tool')} {top.get('reason')} "
            f"digest={str(top.get('digest') or '')[:12]} count={top.get('count')}"
        ),
    }


def _compact_runaway_from_log(
    log: Path,
    *,
    min_count: int = ACTIVE_COMPACT_RUNAWAY_MIN_COUNT,
    min_past_k: float = ACTIVE_COMPACT_RUNAWAY_MIN_PAST_K,
    shed_k: float = ACTIVE_COMPACT_RUNAWAY_SHED_K,
) -> dict[str, Any]:
    """Return positive evidence that one live worker is past compact and looping."""
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {"runaway": False, "unavailable": True, "log": str(log)}
    hits: list[dict[str, Any]] = []
    prev_ctx_k: float | None = None
    sheds = 0
    for line in lines:
        if "fak-turn" not in line or "finish=tool_use" not in line:
            continue
        ctx_m = _CTX_BUDGET_RE.search(line)
        if not ctx_m:
            continue
        try:
            ctx_k = float(ctx_m.group(1))
            limit_k = float(ctx_m.group(2))
        except ValueError:
            continue
        if prev_ctx_k is not None and ctx_k <= prev_ctx_k - shed_k:
            hits.clear()
            sheds += 1
        prev_ctx_k = ctx_k
        past_k = max(0.0, ctx_k - limit_k)
        dist_m = _DIST_PAST_COMPACT_RE.search(line)
        if dist_m:
            try:
                past_k = max(past_k, float(dist_m.group(1)))
            except ValueError:
                pass
        if past_k < min_past_k:
            continue
        compact_m = _COMPACT_FIELD_RE.search(line)
        hits.append({
            "ctx_k": ctx_k,
            "limit_k": limit_k,
            "past_k": past_k,
            "compact": compact_m.group(1) if compact_m else "",
            "line": line[-320:],
        })
    if len(hits) < min_count:
        return {
            "runaway": False,
            "log": str(log),
            "count": len(hits),
            "sheds": sheds,
            "max_past_k": round(max((h["past_k"] for h in hits), default=0.0), 1),
        }
    last = hits[-1]
    return {
        "runaway": True,
        "log": str(log),
        "count": len(hits),
        "sheds": sheds,
        "max_past_k": round(max(h["past_k"] for h in hits), 1),
        "ctx_k": last["ctx_k"],
        "limit_k": last["limit_k"],
        "past_k": round(last["past_k"], 1),
        "compact": last["compact"],
        "sample": last["line"],
    }


def active_compact_runaway_hold(
    root: Path,
    runs_dir: Path,
    *,
    min_count: int = ACTIVE_COMPACT_RUNAWAY_MIN_COUNT,
    min_past_k: float = ACTIVE_COMPACT_RUNAWAY_MIN_PAST_K,
) -> dict[str, Any]:
    """Positive live-worker evidence that compact control is already failing."""
    del root
    dp_mod = ird_attr("dispatch_preflight", dispatch_preflight)
    runaway_fn = ird_attr("_compact_runaway_from_log", _compact_runaway_from_log)
    candidates: list[dict[str, Any]] = []
    for log in sorted(runs_dir.glob("resolve-*.log")):
        pid_file = log.with_suffix(".pid")
        if not pid_file.exists():
            continue
        if not dp_mod.resolve_sidecar_pid_is_live(pid_file):
            continue
        hit = runaway_fn(
            log,
            min_count=min_count,
            min_past_k=min_past_k,
        )
        if not hit.get("runaway"):
            continue
        issue = None
        m = _LOG_ISSUE_RE.search(log.name)
        if m:
            try:
                issue = int(m.group(1))
            except ValueError:
                issue = None
        hit["issue"] = issue
        hit["lane"] = _spawn_header_lane(log)
        candidates.append(hit)
    if not candidates:
        return {"active": False}
    candidates.sort(
        key=lambda h: (float(h.get("max_past_k") or 0.0), int(h.get("count") or 0)),
        reverse=True,
    )
    top = candidates[0]
    lane_bit = f" on lane '{top.get('lane')}'" if top.get("lane") else ""
    return {
        "active": True,
        "candidates": candidates[:5],
        "lanes": sorted({str(c.get("lane")) for c in candidates if c.get("lane")}),
        "lane_unknown": any(not c.get("lane") for c in candidates),
        "reason": (
            f"live worker #{top.get('issue') or '?'}{lane_bit} is already past "
            f"compact by {top.get('max_past_k')}k tokens across "
            f"{top.get('count')} tool-use turns"
        ),
    }


def _backend_of_log(log: Path) -> str:
    """The backend that produced a worker log, from its ``.backend`` sidecar or header."""
    try:
        sidecar = log.with_suffix(".backend").read_text(encoding="utf-8").strip()
        if sidecar:
            return sidecar
    except OSError:
        pass
    try:
        with log.open("r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline()
    except OSError:
        return "claude"
    m = re.search(r"\bbackend=([A-Za-z0-9_]+)", header)
    return m.group(1) if m else "claude"


def _account_tag_of_log(log: Path) -> str | None:
    """Return the switcher account tag stamped next to ``log``, if any."""
    try:
        rec = json.loads(log.with_suffix(ACCOUNT_SIDECAR_SUFFIX).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(rec, dict):
        tag = rec.get("tag") or rec.get("account")
        return str(tag) if tag else None
    if isinstance(rec, str) and rec:
        return rec
    return None


def _log_tail_text(log: Path, *, nbytes: int = _CAP_TAIL_BYTES) -> str:
    """The last ``nbytes`` of ``log`` decoded as utf-8 (errors replaced), or ''."""
    try:
        size = log.stat().st_size
        with log.open("rb") as fh:
            if size > nbytes:
                fh.seek(-nbytes, os.SEEK_END)
            raw = fh.read()
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace")


def _log_is_cap_banner(log: Path) -> bool:
    """True when ``log``'s tail is the quota-limit banner."""
    return _cap_hit_from_text(_log_tail_text(log)) is not None


def _log_is_auth_gap(log: Path) -> bool:
    """True when a stub log's tail is the permanent auth-gap banner."""
    return bool(_AUTH_GAP_RE.search(_log_tail_text(log)))


def _cap_hit_from_text(text: str, *, evidence_log: str = "") -> dict[str, Any] | None:
    """Parse a quota/rate-limit hit from already-captured worker text."""
    if not _CAP_BANNER_RE.search(text or ""):
        return None
    m = (_CODEX_RESET_RE.search(text) or _RESET_AT_RE.search(text) or _CAP_RESET_RE.search(text)
         or _WILL_RESET_LA_RE.search(text) or _ANNOUNCED_WAIT_RE.search(text)
         or _WILL_RESET_RE.search(text) or _CAP_RESET_FALLBACK_RE.search(text))
    reset_text = m.group(1).strip() if m else ""
    if _CAP_WEEKLY_RE.search(text):
        kind = "weekly"
    elif _CAP_FIVE_HOUR_RE.search(text):
        kind = "rolling_5h"
    elif _CAP_SESSION_RE.search(text):
        kind = "session"
    elif not _CAP_USAGE_RE.search(text):
        kind = "session"
    else:
        kind = "rolling_5h"
    return {"reset_text": reset_text, "evidence_log": evidence_log, "kind": kind}


def _write_cap_hold(runs_dir: Path, *, product: str, account_tag: str | None,
                    hit: dict[str, Any], now_ts: float, fallback_min: int,
                    source: str) -> dict[str, Any]:
    now_utc = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=now_ts)
    kind = hit.get("kind") or "rolling_5h"
    if kind == "weekly":
        fallback_min = max(fallback_min, _WEEKLY_FALLBACK_MIN)
    until = (_parse_relative_wait(str(hit.get("reset_text") or ""), now_utc)
             or _parse_reset_to_utc(str(hit.get("reset_text") or ""), now_utc)
             or now_utc + dt.timedelta(minutes=fallback_min))
    hold_max_min = {"session": _SESSION_HOLD_MAX_MIN,
                    "rolling_5h": _ROLLING_HOLD_MAX_MIN}.get(kind)
    if hold_max_min is not None:
        until = min(until, now_utc + dt.timedelta(minutes=hold_max_min))
    state = {"product": product, "account": account_tag, "kind": kind,
             "reset_text": hit.get("reset_text") or "",
             "evidence_log": hit.get("evidence_log") or "",
             "detected": now_utc.isoformat() + "Z", "until": until.isoformat() + "Z"}
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        for state_path in _cap_state_paths(runs_dir, product=product,
                                           account_tag=account_tag):
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass
    return {"capped": True, "until": state["until"], "kind": kind,
            "reset_text": state["reset_text"], "source": source,
            "evidence_log": state["evidence_log"]}


def _safe_cap_account_component(account_tag: str | None) -> str | None:
    if not account_tag:
        return None
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(account_tag).strip())
    return safe or None


def _cap_state_paths(runs_dir: Path, *, product: str,
                     account_tag: str | None) -> list[Path]:
    paths: list[Path] = []
    safe_account = _safe_cap_account_component(account_tag)
    if safe_account:
        paths.append(runs_dir / f"account-cap-{product}-{safe_account}.json")
    paths.append(runs_dir / f"account-cap-{product}.json")
    return paths


def _scan_recent_cap_banner(runs_dir: Path, *, product: str, lookback_min: int,
                            now_ts: float, account_tag: str | None = None) -> dict[str, Any] | None:
    """The most-recent worker log whose tail is the quota-limit banner."""
    if not runs_dir.is_dir():
        return None
    horizon = now_ts - lookback_min * 60
    best: tuple[float, str, str, str] | None = None
    for log in runs_dir.glob("resolve-*.log"):
        try:
            st = log.stat()
        except OSError:
            continue
        if st.st_mtime < horizon:
            continue
        if _backend_of_log(log) != product:
            continue
        if account_tag and _account_tag_of_log(log) != account_tag:
            continue
        hit = _cap_hit_from_text(_log_tail_text(log), evidence_log=log.name)
        if not hit:
            continue
        if best is None or st.st_mtime > best[0]:
            best = (st.st_mtime, str(hit["reset_text"]), str(hit["evidence_log"]), str(hit["kind"]))
    if best is None:
        return None
    return {"reset_text": best[1], "evidence_log": best[2], "kind": best[3]}


def _persisted_hold_matches_account(runs_dir: Path, state: dict[str, Any],
                                    account_tag: str | None) -> bool:
    """True when a persisted cap hold is still attributable to ``account_tag``."""
    if not account_tag:
        return True
    evidence = str(state.get("evidence_log") or "")
    if not evidence:
        return True
    return _account_tag_of_log(runs_dir / evidence) == account_tag


def _parse_reset_to_utc(reset_text: str, now_utc: dt.datetime) -> dt.datetime | None:
    """Best-effort parse of a banner reset clause to naive UTC."""
    if not reset_text:
        return None
    t = reset_text.lower()
    iso = _iso_to_utc(reset_text.strip())
    if iso:
        if iso <= now_utc:
            return None
        return min(iso, now_utc + dt.timedelta(days=8))
    tm = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", t)
    if not tm:
        return None
    hour = int(tm.group(1)) % 12 + (12 if tm.group(3) == "pm" else 0)
    minute = int(tm.group(2) or 0)
    mo = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})", t)
    if mo:
        month, day = _MONTHS[mo.group(1)], int(mo.group(2))
        try:
            cand = dt.datetime(now_utc.year, month, day, hour, minute) + _PT_TO_UTC
        except ValueError:
            return None
        if cand < now_utc - dt.timedelta(days=1):
            try:
                cand = dt.datetime(now_utc.year + 1, month, day, hour, minute) + _PT_TO_UTC
            except ValueError:
                return None
        return _clamp_reset(cand, now_utc)
    wd = re.search(r"\b(mon(?:day)?|tue(?:s(?:day)?)?|wed(?:nesday)?|thu(?:rs(?:day)?)?"
                   r"|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b", t)
    if wd:
        now_pt = now_utc - _PT_TO_UTC
        ahead = (_WEEKDAYS[wd.group(1)[:3]] - now_pt.weekday()) % 7
        cand_pt = (now_pt + dt.timedelta(days=ahead)).replace(
            hour=hour, minute=minute, second=0, microsecond=0)
        if cand_pt <= now_pt:
            cand_pt += dt.timedelta(days=7)
        return _clamp_reset(cand_pt + _PT_TO_UTC, now_utc)
    now_pt = now_utc - _PT_TO_UTC
    cand_pt = now_pt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if cand_pt <= now_pt:
        cand_pt += dt.timedelta(days=1)
    return _clamp_reset(cand_pt + _PT_TO_UTC, now_utc)


def _clamp_reset(cand: dt.datetime, now_utc: dt.datetime) -> dt.datetime | None:
    if cand <= now_utc:
        return None
    return min(cand, now_utc + dt.timedelta(days=8))


def _parse_relative_wait(reset_text: str, now_utc: dt.datetime) -> dt.datetime | None:
    """A relative cap wait ('1h7m0s', '90m') -> now+dur as naive UTC."""
    m = _ANNOUNCED_WAIT_DUR_RE.match(reset_text or "")
    if not m or not any(m.groups()):
        return None
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    seconds = int(m.group(3) or 0)
    if hours == 0 and minutes == 0 and seconds == 0:
        return None
    cand = now_utc + dt.timedelta(hours=hours, minutes=minutes, seconds=seconds)
    return min(cand, now_utc + dt.timedelta(days=8))


def _iso_to_utc(s: str) -> dt.datetime | None:
    try:
        parsed = dt.datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def check_weekly_cap(runs_dir: Path, *, product: str, account_tag: str | None,
                     now_ts: float | None = None, lookback_min: int = 45,
                     fallback_min: int = 60) -> dict[str, Any]:
    """Is ``account_tag`` quota-capped right now? Fails open."""
    try:
        import time
        now_ts = time.time() if now_ts is None else now_ts
        now_utc = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=now_ts)
        hit = ird_attr("_scan_recent_cap_banner", _scan_recent_cap_banner)(
            runs_dir, product=product, lookback_min=lookback_min,
            now_ts=now_ts, account_tag=account_tag)
        if hit:
            return ird_attr("_write_cap_hold", _write_cap_hold)(
                runs_dir, product=product, account_tag=account_tag,
                hit=hit, now_ts=now_ts, fallback_min=fallback_min,
                source="banner")
        for state_path in _cap_state_paths(runs_dir, product=product,
                                           account_tag=account_tag):
            if not state_path.exists():
                continue
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            until = _iso_to_utc(state.get("until") or "")
            if state.get("account") == account_tag and until and now_utc < until:
                if not _persisted_hold_matches_account(runs_dir, state, account_tag):
                    try:
                        state_path.unlink()
                    except OSError:
                        pass
                    continue
                return {"capped": True, "until": state.get("until"),
                        "reset_text": state.get("reset_text", ""), "source": "state"}
            if until and now_utc >= until:
                try:
                    state_path.unlink()
                except OSError:
                    pass
        return {"capped": False}
    except Exception:
        return {"capped": False}


def _classify_backend_logs(runs_dir: Path, *, product: str, lookback_min: int,
                           now_ts: float, alive: set[int] | None,
                           probe: Any | None) -> list[dict[str, Any]]:
    """The recent resolve logs for ONE backend, newest first."""
    dp_mod = ird_attr("dispatch_preflight", dispatch_preflight)
    out: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return out
    horizon = now_ts - lookback_min * 60
    for log in runs_dir.glob("resolve-*.log"):
        m = _LOG_ISSUE_RE.search(log.name)
        if not m:
            continue
        try:
            st = log.stat()
        except OSError:
            continue
        if st.st_mtime < horizon:
            continue
        if _backend_of_log(log) != product:
            continue
        productive = st.st_size > _STUB_LOG_MAX_BYTES and not _log_is_cap_banner(log)
        if not productive:
            pid_file = log.with_suffix(".pid")
            if pid_file.exists() and dp_mod.resolve_sidecar_pid_is_live(
                    pid_file, alive=alive, probe=probe):
                continue
        out.append({"stamp": st.st_mtime, "productive": productive, "size": st.st_size,
                    "log": log.name})
    out.sort(key=lambda r: r["stamp"], reverse=True)
    return out


def check_backend_health(runs_dir: Path, *, product: str, lane: str | None = None,
                         now_ts: float | None = None, lookback_min: int = 90,
                         alive: set[int] | None = None,
                         probe: Any | None = None) -> dict[str, Any]:
    """Is ``product`` producing real turns right now, or spinning dead? Fails open."""
    try:
        import time
        now_ts = time.time() if now_ts is None else now_ts
        now_utc = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=now_ts)
        state_path = runs_dir / f"backend-health-{product}.json"
        if alive is None and probe is None:
            try:
                import psutil  # type: ignore
                alive = {p.pid for p in psutil.process_iter()}
            except ImportError:
                alive = None
        classify_fn = ird_attr("_classify_backend_logs", _classify_backend_logs)
        logs = classify_fn(runs_dir, product=product, lookback_min=lookback_min,
                           now_ts=now_ts, alive=alive, probe=probe)
        productive = sum(1 for r in logs if r["productive"])
        stub = len(logs) - productive
        prior: dict[str, Any] = {}
        if state_path.exists():
            try:
                prior = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                prior = {}
        prior_auth_gap = prior.get("state") == "dead" and bool(prior.get("auth_gap"))
        stub_logs = [r["log"] for r in logs if not r["productive"]]
        authgap_stub = sum(1 for name in stub_logs if _log_is_auth_gap(runs_dir / name))
        authgap_now = authgap_stub * 2 > len(stub_logs) and authgap_stub > 0
        majority_stub = len(logs) >= _BACKEND_DEAD_STREAK and stub > productive
        sticky_auth_gap = prior_auth_gap and productive == 0
        if not (majority_stub or sticky_auth_gap):
            if productive and state_path.exists():
                try:
                    state_path.unlink()
                except OSError:
                    pass
            return {"state": "healthy",
                    "source": "productive" if productive else "no-streak"}
        auth_gap = authgap_now or prior_auth_gap
        reprobe_min = _BACKEND_AUTHGAP_REPROBE_MIN if auth_gap else _BACKEND_REPROBE_MIN
        since = prior.get("since") if prior.get("state") == "dead" else None
        since = since or (now_utc.isoformat() + "Z")
        abandoned = lane or prior.get("abandoned_lane") or ""
        last_probe = _iso_to_utc(prior.get("last_reprobe") or "") if prior else None
        due = (last_probe is None
               or now_utc - last_probe >= dt.timedelta(minutes=reprobe_min))
        evidence = (stub_logs[:_BACKEND_DEAD_STREAK]
                    or list(prior.get("evidence_logs") or []))
        stub_rate = round(stub / len(logs), 3) if logs else (prior.get("stub_rate") or 1.0)
        state = {
            "product": product, "state": "dead", "since": since,
            "abandoned_lane": abandoned,
            "auth_gap": auth_gap,
            "evidence_logs": evidence,
            "stub": stub, "productive": productive,
            "stub_rate": stub_rate,
            "detected": now_utc.isoformat() + "Z",
            "last_reprobe": (now_utc.isoformat() + "Z") if due else prior.get("last_reprobe"),
            "reprobe_min": reprobe_min,
        }
        try:
            runs_dir.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except OSError:
            pass
        return {"state": "dead", "since": since, "abandoned_lane": abandoned,
                "reprobe_due": bool(due), "evidence_logs": evidence,
                "stub": stub, "productive": productive, "auth_gap": auth_gap,
                "reprobe_min": reprobe_min, "stub_rate": stub_rate,
                "source": "auth-gap" if auth_gap else "majority-stub"}
    except Exception:
        return {"state": "healthy", "source": "error"}


def read_dead_backends(runs_dir: Path, *, exclude: str | None = None,
                       now_ts: float | None = None) -> list[dict[str, Any]]:
    """Sibling backends currently held DEAD, read from backend-health-*.json."""
    del now_ts
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
        if exclude and st.get("product") == exclude:
            continue
        out.append(st)
    return out


def seat_adaptive_target(pre: dict[str, Any], *, fallback: int, ceiling: int,
                         ramp_delta: int) -> tuple[int, dict[str, Any]]:
    """Seat-adaptive effective worker cap for one tick (#3246) — a pure fold."""
    def _num(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    limiter = (pre.get("capacity_limiter")
               if isinstance(pre.get("capacity_limiter"), dict) else {}) or {}
    raw = limiter.get("raw") if isinstance(limiter.get("raw"), dict) else {}
    seat = pre.get("seat") if isinstance(pre.get("seat"), dict) else {}
    seat_free = _num(seat.get("free"))
    if seat_free is None:
        seat_free = _num(raw.get("seat_free"))
    live = _num(pre.get("live"))
    if live is None:
        live = _num(raw.get("live"))
    host_cap = _num(pre.get("host_cap"))
    if host_cap is None:
        host_cap = _num(raw.get("host_cap"))
    hard_ceiling = max(int(ceiling), int(fallback))
    info: dict[str, Any] = {
        "enabled": True, "fallback_max_workers": int(fallback),
        "hard_ceiling": hard_ceiling, "ramp_delta": int(ramp_delta),
    }
    if seat_free is None or live is None:
        info.update({"signal_available": False, "effective_target": int(fallback),
                     "binding": "fallback_max_workers"})
        return int(fallback), info
    terms: list[tuple[str, int]] = [("seat_free", live + seat_free),
                                    ("hard_ceiling", hard_ceiling)]
    if host_cap is not None:
        terms.append(("host_cap", host_cap))
    if ramp_delta > 0:
        terms.append(("ramp_delta", live + ramp_delta))
    binding, target = min(terms, key=lambda t: t[1])
    target = max(target, 0)
    info.update({"signal_available": True, "seat_free": seat_free, "live": live,
                 **({"host_cap": host_cap} if host_cap is not None else {}),
                 "terms": dict(terms), "effective_target": target,
                 "binding": binding})
    return target, info


def recent_backend_stub_rate(runs_dir: Path, *, product: str,
                             lookback_min: int = 90, now_ts: float | None = None,
                             alive: set[int] | None = None,
                             probe: Any | None = None) -> float | None:
    """The recent stub_rate for one backend ``product`` from dispatch_status."""
    try:
        import dispatch_status  # noqa: E402
    except ImportError:
        return None
    try:
        rows = dispatch_status.backend_stub_rates(
            runs_dir, lookback_min=lookback_min, now_ts=now_ts,
            alive=alive, probe=probe)
    except Exception:
        return None
    for row in rows or []:
        if row.get("product") == product and row.get("stub_rate") is not None:
            try:
                return float(row.get("stub_rate"))
            except (TypeError, ValueError):
                return None
    return None


def gate_spawn_on_health(planned: int, stub_rate: float | None,
                         threshold: float = _HEALTH_SKIP_STUB_RATE
                         ) -> tuple[int, str | None]:
    """Gate a tick's planned worker count on the backend's recent health (#3247)."""
    if stub_rate is None:
        return planned, None
    try:
        rate = float(stub_rate)
    except (TypeError, ValueError):
        return planned, None
    if rate < threshold:
        return planned, None
    reason = (
        f"backend is majority-stub (recent stub_rate={round(rate, 3)} >= {threshold}); "
        f"planning 0 spawns instead of {planned} — every seat would burn on a "
        f"banner-only/0-byte no-op producing zero real output. Spawns auto-restore the "
        f"first tick its stub_rate recovers below {threshold} (the live signal is "
        f"re-read each tick; no persistent hold)")
    return 0, reason


def gate_opencode_gateway(planned: int, account: dict, *, probe=None) -> tuple[int, str | None]:
    """Suppress opencode only for a conclusive existing-probe GATEWAY_DOWN verdict."""
    if planned <= 0 or not account:
        return planned, None
    ap_mod = ird_attr("account_probe", account_probe)
    probe = probe or ap_mod.probe_opencode_account
    try:
        verdict = probe(account)
    except Exception:
        return planned, None
    if verdict.get("status") != "GATEWAY_DOWN":
        return planned, None
    detail = verdict.get("block_reason") or "guard gateway unreachable"
    return 0, f"gateway_down: backend_unhealthy: {detail}"


def guard_build_id(fak_bin: str | Path | None) -> str | None:
    """Identity of the guard binary a wave is about to be dispatched against."""
    if not fak_bin:
        return None
    try:
        st = Path(fak_bin).stat()
    except (OSError, ValueError):
        return None
    return f"{st.st_size}-{st.st_mtime_ns}-{Path(fak_bin).name}"


def planned_guard_flags(command: list[str]) -> set[str]:
    """The guard flag names in a guard-wrapped launch argv before ``--``."""
    if len(command) < 2 or command[1] != "guard":
        return set()
    flags: set[str] = set()
    for tok in command[2:]:
        if tok == "--":
            break
        if not tok.startswith("-") or tok == "-":
            continue
        name = tok.lstrip("-").split("=", 1)[0]
        if name and name not in _GUARD_PROBE_PEELED:
            flags.add(name)
    return flags


def probe_guard_flags(fak_bin: str | Path, *, runner=None,
                      timeout_s: float = _GUARD_PROBE_TIMEOUT_S) -> dict[str, Any]:
    """Run the one-shot inventory probe ``fak guard -h -all`` and parse its flags."""
    argv = [str(fak_bin), "guard", "-h", "-all"]
    sub_mod = ird_attr("subprocess", subprocess)
    nowin = ird_attr("no_window_creationflags", no_window_creationflags)
    run = runner or (lambda a: sub_mod.run(
        a, capture_output=True, text=True, errors="replace", timeout=timeout_s,
        creationflags=nowin()))
    try:
        res = run(argv)
    except Exception as exc:
        return {"rc": None, "flags": [], "bytes": 0, "error": f"{type(exc).__name__}: {exc}"}
    text = (getattr(res, "stdout", "") or "") + (getattr(res, "stderr", "") or "")
    return {"rc": getattr(res, "returncode", None),
            "flags": sorted(set(_GUARD_FLAG_DEF_RE.findall(text))),
            "bytes": len(text), "error": None}


def guard_flag_inventory(fak_bin: str | Path | None, runs_dir: Path, *, runner=None,
                         now_ts: float | None = None,
                         ttl_h: int = DEFAULT_GUARD_PROBE_TTL_H,
                         record: bool = True) -> dict[str, Any]:
    """The cached ``{build_id -> flags}`` inventory, probing at most once per build-id."""
    import time
    now = now_ts if now_ts is not None else time.time()
    build_id = guard_build_id(fak_bin)
    if not build_id:
        return {"build_id": None, "flags": [], "cached": False, "usable": False,
                "detail": f"unstattable guard binary {fak_bin!r}"}
    ledger = Path(runs_dir) / _GUARD_PROBE_LEDGER
    cache: dict[str, Any] = {}
    try:
        loaded = json.loads(ledger.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            cache = loaded
    except (OSError, ValueError):
        cache = {}
    hit = cache.get(build_id)
    if (isinstance(hit, dict) and isinstance(hit.get("flags"), list)
            and (now - float(hit.get("ts") or 0.0)) <= ttl_h * 3600):
        return {"build_id": build_id, "flags": list(hit["flags"]), "cached": True,
                "usable": True, "detail": "cached"}
    probed = probe_guard_flags(fak_bin, runner=runner)
    flags = probed["flags"]
    if probed["error"] or probed["rc"] not in (0, None) or len(flags) < _GUARD_PROBE_MIN_FLAGS:
        return {"build_id": build_id, "flags": flags, "cached": False, "usable": False,
                "detail": (f"inconclusive probe (rc={probed['rc']}, "
                           f"{len(flags)} flags, err={probed['error']})")}
    if record:
        try:
            Path(runs_dir).mkdir(parents=True, exist_ok=True)
            ledger.write_text(json.dumps({build_id: {
                "utc": dt.datetime.fromtimestamp(now, dt.timezone.utc).isoformat(),
                "ts": now, "bin": str(fak_bin), "flags": flags}}, indent=1),
                encoding="utf-8")
        except OSError:
            pass
    return {"build_id": build_id, "flags": flags, "cached": False, "usable": True,
            "detail": f"probed {len(flags)} flags"}


def gate_spawn_on_guard_argv(command: list[str], runs_dir: Path, *, runner=None,
                             now_ts: float | None = None,
                             ttl_h: int = DEFAULT_GUARD_PROBE_TTL_H,
                             record: bool = True) -> dict[str, Any] | None:
    """Refuse a wave whose guard binary provably cannot accept the planned argv."""
    if ttl_h <= 0:
        return None
    planned = planned_guard_flags(command)
    if not planned:
        return None
    inv = guard_flag_inventory(command[0], runs_dir, runner=runner, now_ts=now_ts,
                               ttl_h=ttl_h, record=record)
    if not inv["usable"]:
        return None
    missing = sorted(planned - set(inv["flags"]))
    if not missing:
        return None
    return {
        "verdict": "GUARD_ARGV_UNSUPPORTED",
        "build_id": inv["build_id"],
        "missing_flags": missing,
        "probe": {"flags": len(inv["flags"]), "cached": inv["cached"]},
        "reason": (
            f"guard binary {command[0]} (build {inv['build_id']}) does not register "
            f"{len(missing)} flag(s) this dispatcher passes: "
            + ", ".join(f"--{f}" for f in missing)
            + " — every spawn against it would die at flag-parse before the gateway "
              "binds and be miscounted as an idle fleet (#5868: ~350 worker-units "
              "lost that way over six days). Planning 0 spawns. Recovery: rebuild or "
              "reinstall fak (the build id changes and the probe re-runs by itself), "
              "or drop the flag from tools/dispatch_worker.py (the refusal clears "
              "against the same cached inventory, no re-probe needed). Set "
              "DEFAULT_GUARD_PROBE_TTL_H=0 to disable the gate."),
    }


def _terminate_lease_probe(proc: Any) -> dict[str, Any]:
    """Reap a timed-out `fak leaseref` probe AND its git descendants."""
    os_mod = ird_attr("os", os)
    sub_mod = ird_attr("subprocess", subprocess)
    nowin = ird_attr("no_window_creationflags", no_window_creationflags)
    pid = getattr(proc, "pid", None)
    try:
        if os_mod.name == "nt":
            killed = sub_mod.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                                 capture_output=True, text=True,
                                 timeout=LEASE_DRAIN_TIMEOUT_S,
                                 creationflags=nowin())
            out: dict[str, Any] = {"ok": killed.returncode == 0,
                                   "returncode": killed.returncode,
                                   "stderr": (killed.stderr or "").strip()[-300:]}
        else:
            os_mod.killpg(pid, signal.SIGKILL)
            out = {"ok": True, "returncode": 0}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        out = {"ok": False, "returncode": None, "error": str(exc)}
    try:
        proc.kill()
    except (OSError, ValueError):
        pass
    out["pid"] = pid
    return out


def _fak_bin(root: Path) -> list[str] | None:
    """The on-disk `fak` argv for a lease subprocess, or None."""
    del root
    os_mod = ird_attr("os", os)
    shutil_mod = ird_attr("shutil", shutil)
    split_fn = ird_attr("split_command_env", split_command_env)
    configured = os_mod.environ.get("FAK_BIN")
    if configured:
        return split_fn(configured)
    exe = shutil_mod.which("fak")
    return [exe] if exe else None


def lease_holder() -> str:
    """A holder identity stable across this session's separate dispatcher ticks."""
    os_mod = ird_attr("os", os)
    sid = os_mod.environ.get("FAK_LEASE_OWNER") or os_mod.environ.get("CLAUDE_CODE_SESSION_ID")
    if sid:
        return sid.strip()
    host = os_mod.environ.get("COMPUTERNAME") or os_mod.environ.get("HOSTNAME") or "host"
    return f"{host}:{os_mod.getpid()}"


def lease_owner_override() -> str | None:
    owner = (ird_attr("os", os).environ.get("FAK_LEASE_OWNER") or "").strip()
    return owner or None


def lease_session_id() -> str | None:
    """Best-effort session binding for `fak leaseref liveness`."""
    os_mod = ird_attr("os", os)
    sid = (os_mod.environ.get("FAK_LEASE_SESSION_ID")
           or os_mod.environ.get("CLAUDE_CODE_SESSION_ID") or "").strip()
    if (not sid or not _LEASE_SESSION_ID_RE.match(sid)
            or sid.startswith("session-")):
        return None
    return sid


def _safe_lease_session_segment(raw: str) -> str | None:
    segment = _LEASE_SESSION_SAFE_RE.sub("-", raw.strip()).strip("-.")
    if not segment:
        return None
    if segment.startswith("session-"):
        segment = "dispatch-" + segment
    segment = segment[:200].rstrip("-.")
    if not segment or not _LEASE_SESSION_ID_RE.match(segment):
        return None
    return segment


def synthetic_lease_session_id() -> str:
    os_mod = ird_attr("os", os)
    host = os_mod.environ.get("COMPUTERNAME") or os_mod.environ.get("HOSTNAME") or "host"
    return (_safe_lease_session_segment(f"dispatch-{host}-{os_mod.getpid()}")
            or f"dispatch-{os_mod.getpid()}")


def lease_id_for_lane(lane: str) -> str:
    return f"{LEASE_ID_PREFIX}{lane}"


def _default_run_lease(root: Path, args: list[str], *, timeout: int = 30) -> dict[str, Any]:
    import issue_resolve_dispatch as ird

    return ird._run_lease(root, args, timeout=timeout)


def publish_lease_session(root: Path, session_id: str | None, ttl_s: int,
                          *, runner: Any | None = None) -> dict[str, Any]:
    """Best-effort heartbeat for a synthetic lease session."""
    if not session_id:
        return {"published": False, "skipped": "missing session id"}
    run = runner or ird_attr("_run_lease", _default_run_lease)
    res = run(root, ["session-publish", "--session", session_id, "--ttl",
                     str(int(ttl_s)), "--state", "RUNNING"])
    if res.get("rc") == 0:
        return {"published": True, "session_id": session_id}
    return {"published": False, "session_id": session_id, "fail_open": True,
            "rc": res.get("rc"), "detail": res.get("error") or res.get("skipped")}


def acquire_lane_lease(root: Path, lane: str, *, tree: list[str], ttl_s: int,
                       holder: str | None = None,
                       runner: Any | None = None) -> dict[str, Any]:
    """ATOMICALLY take the fenced refs/fak/locks/resolve-<lane> lease before a spawn."""
    run = runner or ird_attr("_run_lease", _default_run_lease)
    session_id = lease_session_id()
    requested_holder = holder or lease_owner_override()
    session_publish: dict[str, Any] | None = None
    if not session_id and not requested_holder:
        session_id = synthetic_lease_session_id()
        session_publish = publish_lease_session(root, session_id, ttl_s,
                                                runner=run)
    lease_id = lease_id_for_lane(lane)
    args = ["acquire", "--id", lease_id, "--ttl", str(int(ttl_s))]
    if requested_holder:
        args += ["--holder", requested_holder]
    elif not session_id:
        requested_holder = lease_holder()
        args += ["--holder", requested_holder]
    if session_id:
        args += ["--session", session_id]
    for t in tree:
        args += ["--tree", t]
    res = run(root, args)
    rc = res.get("rc")
    verdict = res.get("verdict") or {}
    if rc == 0:
        rec = verdict.get("record") if isinstance(verdict, dict) else None
        rec_holder = (rec or {}).get("holder") if isinstance(rec, dict) else None
        actual_holder = rec_holder or requested_holder
        out = {"acquired": True, "refused": False, "id": lease_id, "holder": actual_holder,
               "generation": (rec or {}).get("generation") if isinstance(rec, dict) else None,
               "session_id": session_id,
               "tree": tree}
        if session_publish is not None:
            out["session_publish"] = session_publish
        return out
    if rc == LEASE_REFUSED_RC:
        v = verdict.get("verdict") if isinstance(verdict, dict) else None
        out = {"acquired": False, "refused": True, "id": lease_id, "holder": requested_holder,
               "session_id": session_id,
               "reason": (v or {}).get("reason") if isinstance(v, dict) else None,
               "fence_verdict": v, "tree": tree}
        if session_publish is not None:
            out["session_publish"] = session_publish
        return out
    out = {"acquired": False, "refused": False, "id": lease_id, "holder": requested_holder,
           "session_id": session_id,
           "fail_open": True, "rc": rc, "detail": res.get("error") or res.get("skipped"),
           "tree": tree}
    if session_publish is not None:
        out["session_publish"] = session_publish
    return out


def release_lane_lease(root: Path, lease: dict[str, Any] | None,
                       *, runner: Any | None = None) -> dict[str, Any]:
    """Release a previously-acquired lane lease with holder/generation fencing."""
    src = lease or {}
    lease_id = str(src.get("id") or "").strip()
    holder = str(src.get("holder") or "").strip()
    if not lease_id or not holder:
        return {"released": False, "skipped": "missing lease id/holder"}
    args = ["release", "--id", lease_id, "--holder", holder]
    generation = src.get("generation")
    if generation is not None:
        args += ["--generation", str(generation)]
    run = runner or ird_attr("_run_lease", _default_run_lease)
    res = run(root, args)
    rc = res.get("rc")
    verdict = res.get("verdict") or {}
    if rc == 0:
        return {"released": True, "refused": False, "id": lease_id,
                "holder": holder, "generation": generation}
    if rc == LEASE_REFUSED_RC:
        v = verdict.get("verdict") if isinstance(verdict, dict) else None
        return {"released": False, "refused": True, "id": lease_id,
                "holder": holder, "generation": generation,
                "reason": (v or {}).get("reason") if isinstance(v, dict) else None,
                "fence_verdict": v}
    return {"released": False, "refused": False, "id": lease_id,
            "holder": holder, "generation": generation, "fail_open": True,
            "rc": rc, "detail": res.get("error") or res.get("skipped")}


def reap_expired_leases(root: Path, *, runner: Any | None = None) -> dict[str, Any]:
    """Delete expired (reapable) refs/fak/locks/* leases."""
    run = runner or ird_attr("_run_lease", _default_run_lease)
    res = run(root, ["reap"])
    return {"ok": res.get("rc") == 0, "rc": res.get("rc"), "stdout": res.get("stdout")}


def live_lane_lease_lanes(root: Path, *, runner: Any | None = None) -> dict[str, Any]:
    """Read live refs/fak/locks lane leases for planner-side collision avoidance."""
    run = runner or ird_attr("_run_lease", _default_run_lease)
    res = run(root, ["live"])
    if res.get("rc") != 0:
        out = {"lanes": [], "fail_open": True, "rc": res.get("rc"),
               "detail": res.get("error") or res.get("skipped")}
        if res.get("timeout"):
            out.update({"timeout": True, "timeout_s": res.get("timeout_s"),
                        "killed": res.get("killed"), "drained": res.get("drained")})
        return out
    doc = res.get("verdict")
    if not isinstance(doc, list):
        return {"lanes": [], "fail_open": True, "rc": res.get("rc"),
                "detail": "unparseable leaseref live output"}
    lanes: set[str] = set()
    records: list[dict[str, Any]] = []
    for row in doc:
        if not isinstance(row, dict):
            continue
        records.append(row)
        lane = str(row.get("lane") or row.get("id") or "").strip()
        if lane.startswith(LEASE_ID_PREFIX):
            lane = lane[len(LEASE_ID_PREFIX):]
        if lane:
            lanes.add(lane)
    return {"lanes": sorted(lanes), "records": records}
