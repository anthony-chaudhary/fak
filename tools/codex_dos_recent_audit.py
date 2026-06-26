#!/usr/bin/env python3
"""Roll up recent Codex sessions against DOS hook evidence.

This is the multi-session companion to ``codex_dogfood_witness.py``. It discovers
Codex thread IDs from the local Codex session store, matches those IDs to
``.dos/streams/<thread>.jsonl``, and folds each stream through the same
privacy-preserving DOS audit helper used by the dogfood witness.

It does not copy prompts, tool arguments, tool results, diffs, or model text.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import urllib.request


SCHEMA = "fak-codex-dos-recent-audit/1"
THREAD_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
ACTIVE_STOP_FAILURE_RECENT_HOURS = 6
GIT_WRITE_SUBCOMMANDS = {
    "add",
    "checkout",
    "cherry-pick",
    "clean",
    "commit",
    "merge",
    "mv",
    "push",
    "rebase",
    "reset",
    "restore",
    "revert",
    "rm",
    "stash",
    "switch",
    "tag",
    "worktree",
}


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def codex_home(env: dict[str, str] = os.environ) -> Path:
    configured = env.get("CODEX_HOME")
    if configured:
        return Path(configured)
    return Path.home() / ".codex"


def load_witness_module():
    script = Path(__file__).resolve().with_name("codex_dogfood_witness.py")
    spec = importlib.util.spec_from_file_location("codex_dogfood_witness", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_hook_doctor_module():
    script = Path(__file__).resolve().with_name("codex_dos_hook_doctor.py")
    spec = importlib.util.spec_from_file_location("codex_dos_hook_doctor", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_repo_guard_module():
    script = Path(__file__).resolve().with_name("repo_guard.py")
    spec = importlib.util.spec_from_file_location("repo_guard", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        out = datetime.fromisoformat(text)
    except ValueError:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def nearest_rank(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if pct <= 0:
        return vals[0]
    if pct >= 100:
        return vals[-1]
    rank = max(1, int((pct / 100.0) * len(vals) + 0.999999))
    return vals[min(rank - 1, len(vals) - 1)]


def latency_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "n": len(values),
        "p50_ms": round(float(nearest_rank(values, 50) or 0), 3),
        "p95_ms": round(float(nearest_rank(values, 95) or 0), 3),
        "max_ms": round(max(values), 3),
    }


def discover_codex_threads(home: Path, *, since_days: int = 7) -> dict[str, Path]:
    sessions = home / "sessions"
    if not sessions.exists():
        return {}

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(1, since_days))
    found: dict[str, Path] = {}
    for path in sessions.rglob("*.jsonl"):
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            continue
        match = THREAD_RE.search(path.name)
        if match is None:
            continue
        thread_id = match.group(0)
        prev = found.get(thread_id)
        if prev is None or path.stat().st_mtime > prev.stat().st_mtime:
            found[thread_id] = path
    return found


def stream_bounds(path: Path) -> tuple[datetime | None, datetime | None]:
    first: datetime | None = None
    last: datetime | None = None
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                ts = parse_ts(row.get("ts"))
                if ts is None:
                    continue
                first = ts if first is None or ts < first else first
                last = ts if last is None or ts > last else last
    except OSError:
        return None, None
    return first, last


def discover_streams(repo_root: Path, codex_threads: dict[str, Path], limit: int) -> list[tuple[str, Path]]:
    streams_dir = repo_root / ".dos" / "streams"
    if not streams_dir.exists():
        return []
    candidates: list[tuple[float, str, Path]] = []
    for path in streams_dir.glob("*.jsonl"):
        thread_id = path.stem
        if thread_id not in codex_threads:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, thread_id, path))
    candidates.sort(reverse=True)
    return [(thread_id, path) for _, thread_id, path in candidates[: max(1, limit)]]


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def short_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def find_claude_transcript_path(session_id: str, user_home: Path) -> Path | None:
    for account_dir in sorted(user_home.glob(".claude*")):
        projects = account_dir / "projects"
        if not projects.is_dir():
            continue
        for path in sorted(projects.glob(f"*/{session_id}.jsonl")):
            return path
    return None


def public_claude_transcript(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "MISSING"}
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        mtime = None
    return {
        "status": "FOUND",
        "account": public_claude_account_label(path.parent.parent.parent.name),
        "project": path.parent.name,
        "mtime": mtime.isoformat().replace("+00:00", "Z") if mtime is not None else None,
    }


def find_claude_transcript(session_id: str, user_home: Path) -> dict[str, Any]:
    return public_claude_transcript(find_claude_transcript_path(session_id, user_home))


def limited_counter(counter: Counter[str], limit: int = 12) -> dict[str, int]:
    return {key: int(value) for key, value in counter.most_common(limit) if value}


def public_claude_account_label(name: str) -> str:
    if name == ".claude":
        return name
    if name.startswith(".claude"):
        return ".claude-" + short_digest(name)
    return short_digest(name)


def stop_failure_origin(entry: dict[str, Any]) -> str:
    labels = entry.get("origin_labels")
    if isinstance(labels, list) and labels:
        return "+".join(str(label) for label in labels if str(label))
    return str(entry.get("origin") or "marker_only")


def stop_failure_origin_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for entry in entries:
        counts[stop_failure_origin(entry)] += 1
    return limited_counter(counts)


def stop_failure_settlement_action(entry: dict[str, Any]) -> str:
    total = safe_int(entry.get("total"))
    consecutive = safe_int(entry.get("consecutive"))
    if total <= 0:
        return "ZERO_TOTAL"
    if consecutive <= 0:
        return "HEALED_NONZERO"
    if bool(entry.get("active_recent")):
        return "RECENT_REVIEW"
    if stop_failure_origin(entry) == "marker_only":
        return "STALE_MARKER_ONLY_ARCHIVE_CANDIDATE"
    return "STALE_RESET_CANDIDATE"


def stop_failure_settlement_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for entry in entries:
        counts[stop_failure_settlement_action(entry)] += 1
    return limited_counter(counts)


def compact_stop_failure_plan_entry(entry: dict[str, Any]) -> dict[str, Any]:
    transcript = entry.get("transcript") if isinstance(entry.get("transcript"), dict) else {}
    summary = entry.get("transcript_summary") if isinstance(entry.get("transcript_summary"), dict) else {}
    session_id = str(entry.get("session_id") or "")
    return {
        "session_id": session_id,
        "marker_path": f".dos/stop-failures/{session_id}.json" if session_id else "",
        "total": safe_int(entry.get("total")),
        "consecutive": safe_int(entry.get("consecutive")),
        "age_seconds": safe_int(entry.get("age_seconds")),
        "mtime": entry.get("mtime"),
        "origin": stop_failure_origin(entry),
        "settlement_action": stop_failure_settlement_action(entry),
        "transcript_status": transcript.get("status"),
        "transcript_project": transcript.get("project"),
        "transcript_evidence_tags": [str(tag) for tag in (summary.get("evidence_tags") or [])],
    }


def stop_failure_settlement_plan(entries: list[dict[str, Any]], limit: int = 5) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        action = stop_failure_settlement_action(entry)
        buckets.setdefault(action, []).append(entry)
    ordered: dict[str, list[dict[str, Any]]] = {}
    for action in [
        "RECENT_REVIEW",
        "STALE_RESET_CANDIDATE",
        "STALE_MARKER_ONLY_ARCHIVE_CANDIDATE",
        "HEALED_NONZERO",
        "ZERO_TOTAL",
    ]:
        rows = buckets.get(action) or []
        rows.sort(
            key=lambda item: (
                -safe_int(item.get("consecutive")),
                -safe_int(item.get("total")),
                safe_int(item.get("age_seconds")),
                str(item.get("session_id") or ""),
            )
        )
        if rows:
            ordered[action] = [compact_stop_failure_plan_entry(item) for item in rows[: max(1, limit)]]
    return ordered


def summarize_claude_transcript_shape(path: Path) -> dict[str, Any]:
    line_count = 0
    malformed = 0
    row_types: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    block_types: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    marker_lines: Counter[str] = Counter()
    result_shapes: Counter[str] = Counter()
    first_ts = None
    last_ts = None
    max_tool_result_chars = 0
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {"status": "UNREADABLE", "error_type": type(exc).__name__}

    markers = {
        "stopfailure": ("stopfailure", "stop failure"),
        "api_wall": ("api-wall", "api wall"),
        "hook": ("hook",),
        "blocked": ("blocked",),
        "denied": ("denied", "deny"),
        "permission": ("permission",),
        "error": ("error", "exception", "traceback"),
    }
    for line in lines:
        line_count += 1
        lower = line.lower()
        for marker, needles in markers.items():
            if any(needle in lower for needle in needles):
                marker_lines[marker] += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(row, dict):
            malformed += 1
            continue
        row_types[str(row.get("type") or "unknown")] += 1
        ts = row.get("timestamp") or row.get("created_at")
        if isinstance(ts, str) and ts:
            first_ts = first_ts or ts
            last_ts = ts
        msg = row.get("message") if isinstance(row.get("message"), dict) else {}
        if msg:
            roles[str(msg.get("role") or "unknown")] += 1
            content = msg.get("content") if isinstance(msg.get("content"), list) else []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = str(block.get("type") or "unknown")
                block_types[block_type] += 1
                if block_type == "tool_use":
                    tools[str(block.get("name") or "unknown")] += 1
                if block_type == "tool_result":
                    result = block.get("content")
                    if isinstance(result, str):
                        max_tool_result_chars = max(max_tool_result_chars, len(result))
                        result_lower = result.lower()
                        if "stopfailure" in result_lower or "stop failure" in result_lower:
                            result_shapes["tool_result_stopfailure_text"] += 1
                        elif "permission" in result_lower:
                            result_shapes["tool_result_permission_text"] += 1
                        elif "error" in result_lower or "exception" in result_lower or "traceback" in result_lower:
                            result_shapes["tool_result_error_text"] += 1
                        else:
                            result_shapes["tool_result_other_text"] += 1
                    elif isinstance(result, list):
                        result_shapes["tool_result_list"] += 1
                    elif result is None:
                        result_shapes["tool_result_null"] += 1
                    else:
                        result_shapes[f"tool_result_{type(result).__name__}"] += 1
        tool_result = row.get("toolUseResult")
        if isinstance(tool_result, dict):
            result_shapes["toolUseResult_dict"] += 1
            if tool_result.get("isError"):
                result_shapes["toolUseResult_isError"] += 1
        elif isinstance(tool_result, str):
            result_shapes["toolUseResult_str"] += 1
            max_tool_result_chars = max(max_tool_result_chars, len(tool_result))

    tool_total = sum(tools.values())
    evidence_tags: list[str] = []
    if marker_lines.get("hook") or marker_lines.get("api_wall") or result_shapes.get("tool_result_stopfailure_text"):
        evidence_tags.append("HOOK_OR_API_WALL_FEEDBACK")
    if marker_lines.get("permission") or result_shapes.get("tool_result_permission_text"):
        evidence_tags.append("HOST_PERMISSION_INTERRUPT")
    if marker_lines.get("blocked") or marker_lines.get("denied"):
        evidence_tags.append("DENY_OR_BLOCKED_FEEDBACK")
    if marker_lines.get("error") or result_shapes.get("tool_result_error_text") or result_shapes.get("toolUseResult_isError"):
        evidence_tags.append("TOOL_ERROR_RECOVERY")
    if tool_total and tools.get("Bash", 0) / tool_total >= 0.5:
        evidence_tags.append("SHELL_HEAVY_SESSION")
    if max_tool_result_chars >= 20000:
        evidence_tags.append("LARGE_TOOL_RESULT")

    return {
        "status": "SUMMARIZED",
        "lines": line_count,
        "malformed_lines": malformed,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "row_type_counts": limited_counter(row_types),
        "role_counts": limited_counter(roles),
        "block_type_counts": limited_counter(block_types),
        "tool_counts": limited_counter(tools),
        "marker_line_counts": limited_counter(marker_lines),
        "tool_result_shape_counts": limited_counter(result_shapes),
        "max_tool_result_chars": max_tool_result_chars,
        "evidence_tags": evidence_tags,
        "privacy": {
            "copied_fields": ["counts", "timestamps", "tool names", "derived evidence tags", "maximum tool-result length"],
            "dropped": ["prompts", "tool arguments", "tool result text", "commands", "model text"],
        },
    }
    return {"status": "MISSING"}


def workspace_stop_failure_audit(repo_root: Path, *, since_days: int, limit: int = 20) -> dict[str, Any]:
    stop_dir = repo_root / ".dos" / "stop-failures"
    if not stop_dir.exists():
        return {
            "status": "MISSING",
            "path": ".dos/stop-failures",
            "reason": "no workspace stop-failure directory",
        }

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(1, since_days))
    active_recent_seconds = ACTIVE_STOP_FAILURE_RECENT_HOURS * 3600
    user_home = Path(os.environ.get("FLEET_USER_HOME") or str(Path.home()))
    entries: list[dict[str, Any]] = []
    transcript_paths: dict[str, Path] = {}
    malformed = 0
    for path in stop_dir.glob("*.json"):
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            malformed += 1
            continue
        if not isinstance(raw, dict):
            malformed += 1
            continue
        session_id = path.stem
        transcript_path = find_claude_transcript_path(session_id, user_home)
        if transcript_path is not None:
            transcript_paths[session_id] = transcript_path
        stream_path = repo_root / ".dos" / "streams" / f"{session_id}.jsonl"
        origin_labels: list[str] = []
        if stream_path.exists():
            origin_labels.append("dos_stream")
        if transcript_path is not None:
            origin_labels.append("claude_transcript")
        if not origin_labels:
            origin_labels.append("marker_only")
        entry = {
            "session_id": session_id,
            "mtime": mtime.isoformat().replace("+00:00", "Z"),
            "total": safe_int(raw.get("total")),
            "consecutive": safe_int(raw.get("consecutive")),
            "transcript": public_claude_transcript(transcript_path),
            "origin": "+".join(origin_labels),
            "origin_labels": origin_labels,
        }
        age_seconds = max(0, int((now - mtime).total_seconds()))
        entry["age_seconds"] = age_seconds
        entry["active_recent"] = bool(entry["consecutive"] > 0 and age_seconds <= active_recent_seconds)
        entry["settlement_action"] = stop_failure_settlement_action(entry)
        entries.append(entry)

    entries.sort(key=lambda item: str(item.get("mtime") or ""), reverse=True)
    nonzero = [entry for entry in entries if int(entry.get("total") or 0) > 0]
    active = [entry for entry in entries if int(entry.get("consecutive") or 0) > 0]
    recent_active = [entry for entry in active if bool(entry.get("active_recent"))]
    stale_active = [entry for entry in active if not bool(entry.get("active_recent"))]
    healed_nonzero = [entry for entry in nonzero if int(entry.get("consecutive") or 0) == 0]
    top_nonzero = sorted(
        nonzero,
        key=lambda item: (
            int(item.get("total") or 0),
            int(item.get("consecutive") or 0),
            str(item.get("mtime") or ""),
        ),
        reverse=True,
    )
    top_active = sorted(
        active,
        key=lambda item: (
            int(item.get("consecutive") or 0),
            int(item.get("total") or 0),
            str(item.get("mtime") or ""),
        ),
        reverse=True,
    )
    top_recent_active = sorted(
        recent_active,
        key=lambda item: (
            int(item.get("consecutive") or 0),
            int(item.get("total") or 0),
            str(item.get("mtime") or ""),
        ),
        reverse=True,
    )
    top_stale_active = sorted(
        stale_active,
        key=lambda item: (
            int(item.get("consecutive") or 0),
            int(item.get("total") or 0),
            str(item.get("mtime") or ""),
        ),
        reverse=True,
    )
    summary_targets: dict[str, dict[str, Any]] = {}
    for entry in (
        top_nonzero[: max(1, limit)]
        + top_active[: max(1, limit)]
        + top_recent_active[: max(1, limit)]
        + top_stale_active[: max(1, limit)]
    ):
        session_id = str(entry.get("session_id") or "")
        if session_id:
            summary_targets[session_id] = entry
    transcript_evidence_tags: Counter[str] = Counter()
    summarized_transcripts = 0
    for session_id, entry in summary_targets.items():
        transcript_path = transcript_paths.get(session_id)
        if transcript_path is None:
            continue
        summary = summarize_claude_transcript_shape(transcript_path)
        entry["transcript_summary"] = summary
        if summary.get("status") == "SUMMARIZED":
            summarized_transcripts += 1
            for tag in summary.get("evidence_tags") or []:
                transcript_evidence_tags[str(tag)] += 1
    total_failures = sum(int(entry.get("total") or 0) for entry in entries)
    max_consecutive = max((int(entry.get("consecutive") or 0) for entry in entries), default=0)
    active_consecutive_total = sum(int(entry.get("consecutive") or 0) for entry in active)
    recent_active_consecutive_total = sum(int(entry.get("consecutive") or 0) for entry in recent_active)
    stale_active_consecutive_total = sum(int(entry.get("consecutive") or 0) for entry in stale_active)
    by_day: dict[str, dict[str, int]] = {}
    for entry in entries:
        day = str(entry.get("mtime") or "")[:10] or "unknown"
        bucket = by_day.setdefault(
            day,
            {
                "markers": 0,
                "nonzero_total_markers": 0,
                "active_consecutive_markers": 0,
                "recent_active_consecutive_markers": 0,
                "stale_active_consecutive_markers": 0,
                "healed_nonzero_markers": 0,
                "total_failures": 0,
                "active_consecutive_total": 0,
                "recent_active_consecutive_total": 0,
                "stale_active_consecutive_total": 0,
                "max_consecutive": 0,
            },
        )
        bucket["markers"] += 1
        total = int(entry.get("total") or 0)
        consecutive = int(entry.get("consecutive") or 0)
        if total:
            bucket["nonzero_total_markers"] += 1
        if consecutive:
            bucket["active_consecutive_markers"] += 1
            if entry.get("active_recent"):
                bucket["recent_active_consecutive_markers"] += 1
                bucket["recent_active_consecutive_total"] += consecutive
            else:
                bucket["stale_active_consecutive_markers"] += 1
                bucket["stale_active_consecutive_total"] += consecutive
        if total and not consecutive:
            bucket["healed_nonzero_markers"] += 1
        bucket["total_failures"] += total
        bucket["active_consecutive_total"] += consecutive
        bucket["max_consecutive"] = max(bucket["max_consecutive"], consecutive)
    return {
        "status": "WARN" if total_failures else "PASS",
        "path": ".dos/stop-failures",
        "scope": "workspace StopFailure API-wall breaker markers; includes non-Codex hook sessions",
        "markers": len(entries),
        "zero_total_markers": len(entries) - len(nonzero),
        "nonzero_total_markers": len(nonzero),
        "active_consecutive_markers": len(active),
        "active_consecutive_total": active_consecutive_total,
        "active_recent_threshold_hours": ACTIVE_STOP_FAILURE_RECENT_HOURS,
        "recent_active_consecutive_markers": len(recent_active),
        "recent_active_consecutive_total": recent_active_consecutive_total,
        "stale_active_consecutive_markers": len(stale_active),
        "stale_active_consecutive_total": stale_active_consecutive_total,
        "origin_counts": stop_failure_origin_counts(entries),
        "active_origin_counts": stop_failure_origin_counts(active),
        "recent_active_origin_counts": stop_failure_origin_counts(recent_active),
        "stale_active_origin_counts": stop_failure_origin_counts(stale_active),
        "nonzero_origin_counts": stop_failure_origin_counts(nonzero),
        "settlement_action_counts": stop_failure_settlement_counts(entries),
        "active_settlement_action_counts": stop_failure_settlement_counts(active),
        "recent_active_settlement_action_counts": stop_failure_settlement_counts(recent_active),
        "stale_active_settlement_action_counts": stop_failure_settlement_counts(stale_active),
        "nonzero_settlement_action_counts": stop_failure_settlement_counts(nonzero),
        "healed_nonzero_markers": len(healed_nonzero),
        "total_failures": total_failures,
        "max_consecutive": max_consecutive,
        "malformed_markers": malformed,
        "by_day": {day: by_day[day] for day in sorted(by_day)},
        "recent": entries[: max(1, limit)],
        "top_nonzero": top_nonzero[: max(1, limit)],
        "top_active": top_active[: max(1, limit)],
        "top_recent_active": top_recent_active[: max(1, limit)],
        "top_stale_active": top_stale_active[: max(1, limit)],
        "settlement_plan": stop_failure_settlement_plan(entries, limit=max(5, limit)),
        "summarized_transcripts": summarized_transcripts,
        "transcript_evidence_tag_counts": limited_counter(transcript_evidence_tags),
        "privacy": {
            "copied_fields": ["session id", "marker timestamps", "counts", "Claude account/project names", "sanitized transcript-shape counts"],
            "dropped": ["prompts", "tool arguments", "tool results", "commands", "model text"],
        },
    }


def compact_session(audit: dict[str, Any], codex_path: Path) -> dict[str, Any]:
    obs = audit.get("observations") if isinstance(audit.get("observations"), dict) else {}
    stream = audit.get("stream") if isinstance(audit.get("stream"), dict) else {}
    stop = audit.get("stop_failures") if isinstance(audit.get("stop_failures"), dict) else {}
    return {
        "thread_id": audit.get("thread_id"),
        "status": audit.get("status"),
        "codex_session_file": codex_path.name,
        "stream_path": stream.get("stream_path"),
        "steps": int(stream.get("steps") or 0),
        "tool_counts": stream.get("tool_counts") or {},
        "observations": int(obs.get("observations") or 0),
        "pretool_calls": int(obs.get("pretool_calls") or 0),
        "unknown_tree_admission_warnings": int(obs.get("unknown_tree_admission_warnings") or 0),
        "unknown_tree_warning_rate": obs.get("unknown_tree_warning_rate"),
        "delegate_count": int(obs.get("delegate_count") or 0),
        "stop_blocks": int(obs.get("stop_blocks") or 0),
        "stop_failures_total": int(stop.get("total") or 0),
        "window_start": obs.get("window_start"),
        "window_end": obs.get("window_end"),
        "observation_scope": obs.get("scope"),
        "recommendations": audit.get("recommendations") or [],
    }


def pypi_latest_version() -> dict[str, Any]:
    url = "https://pypi.org/pypi/dos-kernel/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 - fixed HTTPS package index URL.
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        return {"status": "UNKNOWN", "source": url, "error": str(exc)}
    info = data.get("info") if isinstance(data, dict) else None
    version = info.get("version") if isinstance(info, dict) else None
    if not isinstance(version, str) or not version:
        return {"status": "UNKNOWN", "source": url, "error": "PyPI JSON had no info.version"}
    return {"status": "FOUND", "source": url, "version": version}


def local_dos_version(repo_root: Path, *, check_latest: bool = False) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import importlib.metadata as m; print(m.version('dos-kernel'))"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "UNKNOWN", "error": str(exc)}
    version = proc.stdout.strip()
    if proc.returncode != 0 or not version:
        return {"status": "UNKNOWN", "error": (proc.stderr or "version probe returned no output").strip()}
    out = {
        "status": "FOUND",
        "distribution": "dos-kernel",
        "version": version,
        "latest_checked": check_latest,
    }
    if check_latest:
        latest = pypi_latest_version()
        out["latest"] = latest
        if latest.get("status") == "FOUND":
            out["using_latest"] = version == latest.get("version")
    else:
        out["latest_check_note"] = "run with --check-latest to compare against https://pypi.org/pypi/dos-kernel/json"
    return out


def home_relpath(path: Path, home: Path) -> str:
    try:
        return path.resolve().relative_to(home.resolve()).as_posix()
    except (OSError, ValueError):
        return path.name


def hook_commands(manifest: dict[str, Any]) -> list[str]:
    hooks = manifest.get("hooks")
    if not isinstance(hooks, dict):
        return []
    commands: list[str] = []
    for entries in hooks.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            inner = entry.get("hooks")
            if not isinstance(inner, list):
                continue
            for hook in inner:
                if not isinstance(hook, dict):
                    continue
                command = hook.get("command")
                if isinstance(command, str) and command:
                    commands.append(command)
    return commands


def hook_command_mode(command: str) -> str:
    lower = command.lower()
    if "dos-hook.ps1" in lower:
        return "powershell_native_launcher"
    if "dos-hook" in lower:
        return "native_launcher"
    if "dos.cli hook" in lower:
        return "python_cli"
    return "other"


def codex_hook_fast_path(home: Path) -> dict[str, Any]:
    plugin_root = home / "plugins" / "cache" / "dos" / "dos-kernel"
    manifests = sorted(plugin_root.glob("*/hooks/hooks.json"))
    if not manifests:
        cache_root = home / "plugins" / "cache"
        if cache_root.exists():
            manifests = sorted(
                p for p in cache_root.glob("**/hooks/hooks.json")
                if "dos-kernel" in p.as_posix()
            )
    if not manifests:
        return {
            "status": "UNKNOWN",
            "reason": "no cached dos-kernel hook manifest found under Codex home",
        }

    command_modes: Counter[str] = Counter()
    codex_command_modes: Counter[str] = Counter()
    manifest_paths: list[str] = []
    backup_paths: list[str] = []
    repaired_at: datetime | None = None
    malformed = 0
    for manifest_path in manifests:
        manifest_paths.append(home_relpath(manifest_path, home))
        backup_path = manifest_path.with_name(manifest_path.name + ".before-native-dos-hook.bak")
        if backup_path.exists():
            backup_paths.append(home_relpath(backup_path, home))
            try:
                candidate = datetime.fromtimestamp(manifest_path.stat().st_mtime, tz=timezone.utc)
            except OSError:
                candidate = None
            if candidate is not None and (repaired_at is None or candidate > repaired_at):
                repaired_at = candidate
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            malformed += 1
            continue
        for command in hook_commands(manifest):
            mode = hook_command_mode(command)
            command_modes[mode] += 1
            if "--dialect codex" in command.lower():
                codex_command_modes[mode] += 1

    codex_python = int(codex_command_modes.get("python_cli") or 0)
    codex_powershell = int(codex_command_modes.get("powershell_native_launcher") or 0)
    codex_native = int(codex_command_modes.get("native_launcher") or 0)
    if codex_python:
        status = "WARN"
        reason = "Codex hook commands route through the Python CLI hook instead of the bundled native launcher"
    elif codex_powershell:
        status = "WARN"
        reason = "Codex hook commands route through the PowerShell native launcher; use shell:bash and bin/dos-hook to avoid Windows launch-window side effects"
    elif codex_native:
        status = "PASS"
        reason = "Codex hook commands use the native launcher"
    else:
        status = "UNKNOWN"
        reason = "no Codex-dialect hook commands were found in the cached manifest"

    try:
        doctor_report = load_hook_doctor_module().build_report(home, apply=False)
        doctor_summary = doctor_report.get("summary") if isinstance(doctor_report.get("summary"), dict) else {}
        projected_codex = doctor_summary.get("projected_codex_command_modes") or {}
        repair_projection = {
            "status": doctor_report.get("status"),
            "codex_replacements_available": int(doctor_summary.get("codex_replacements_available") or 0),
            "codex_unrepairable_python_cli_hooks": int(doctor_summary.get("codex_unrepairable_python_cli_hooks") or 0),
            "projected_codex_command_modes": projected_codex,
            "would_clear_codex_python_cli": (
                int(projected_codex.get("python_cli") or 0) == 0
                and int(projected_codex.get("native_launcher") or 0) > 0
            ),
        }
    except (OSError, RuntimeError, ValueError, AttributeError) as exc:
        repair_projection = {
            "status": "UNKNOWN",
            "error": type(exc).__name__,
            "would_clear_codex_python_cli": False,
        }

    return {
        "status": status,
        "reason": reason,
        "manifests": manifest_paths,
        "backups": backup_paths,
        "repaired_at": repaired_at.isoformat().replace("+00:00", "Z") if repaired_at is not None else None,
        "manifest_count": len(manifests),
        "malformed_manifests": malformed,
        "command_modes": {k: command_modes[k] for k in sorted(command_modes)},
        "codex_command_modes": {k: codex_command_modes[k] for k in sorted(codex_command_modes)},
        "codex_python_cli_hooks": codex_python,
        "codex_powershell_native_hooks": codex_powershell,
        "codex_native_launcher_hooks": codex_native,
        "doctor": {
            "dry_run": "python tools/codex_dos_hook_doctor.py --codex-home <codex-home>",
            "apply": "python tools/codex_dos_hook_doctor.py --codex-home <codex-home> --apply",
        },
        "repair_projection": repair_projection,
    }


def codex_observations_since(repo_root: Path, since_text: Any) -> dict[str, Any]:
    since = parse_ts(since_text)
    obs_path = repo_root / ".dos" / "metrics" / "observations.jsonl"
    if since is None:
        return {"status": "UNKNOWN", "reason": "no repair timestamp"}
    if not obs_path.exists():
        return {"status": "UNKNOWN", "reason": "missing DOS observation log"}

    rows = 0
    pretool = 0
    delegates = 0
    unknown_tree = 0
    outcomes: Counter[str] = Counter()
    rungs: Counter[str] = Counter()
    verbs: Counter[str] = Counter()
    latencies: dict[str, list[float]] = {}
    try:
        with obs_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or row.get("dialect") != "codex":
                    continue
                ts = parse_ts(row.get("ts"))
                if ts is None or ts < since:
                    continue
                rows += 1
                verb = str(row.get("verb") or "unknown")
                outcome = str(row.get("outcome") or "unknown")
                rung = str(row.get("rung") or "none")
                verbs[verb] += 1
                outcomes[outcome] += 1
                rungs[rung] += 1
                if verb == "pretool":
                    pretool += 1
                if outcome == "delegate":
                    delegates += 1
                if outcome == "warn" and rung == "admission" and row.get("tree_known") is False:
                    unknown_tree += 1
                if isinstance(row.get("latency_ms"), (int, float)):
                    latencies.setdefault(verb, []).append(float(row["latency_ms"]))
    except OSError as exc:
        return {"status": "UNKNOWN", "reason": type(exc).__name__}

    status = "PASS" if rows else "UNKNOWN"
    if delegates or unknown_tree:
        status = "WARN"
    return {
        "status": status,
        "since": since.isoformat().replace("+00:00", "Z"),
        "observations": rows,
        "pretool_calls": pretool,
        "delegate_count": delegates,
        "unknown_tree_admission_warnings": unknown_tree,
        "unknown_tree_warning_rate": round(unknown_tree / pretool, 6) if pretool else None,
        "verb_counts": {k: verbs[k] for k in sorted(verbs)},
        "outcome_counts": {k: outcomes[k] for k in sorted(outcomes)},
        "rung_counts": {k: rungs[k] for k in sorted(rungs)},
        "latency_by_verb": {k: latency_summary(v) for k, v in sorted(latencies.items())},
    }


def parse_arguments(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def classify_shell_shape(command: str, repo_root: Path) -> tuple[str, Counter[str]]:
    try:
        guard = load_repo_guard_module()
        workspace = str(repo_root)
        safe_roots = guard.default_safe_roots() + guard.private_companion_roots(workspace)
        targets = guard.extract_targets(command)
        violations = guard.classify_command(command, workspace_root=workspace, safe_roots=safe_roots)
    except Exception:  # noqa: BLE001 - diagnostic should fail closed to unknown category, not leak.
        return "shell_classifier_error", Counter()

    ops = Counter(str(op) for op, _raw in targets)
    if violations:
        return "shell_out_of_tree_write_target", ops
    if targets:
        return "shell_in_tree_or_safe_write_target", ops
    return "shell_no_write_target_detected", ops


def shell_command_family(command: str) -> str:
    normalized = " ".join(command.strip().split()).lower()
    if not normalized:
        return "empty_shell"
    if normalized.startswith("@'") or normalized.startswith('@"'):
        return "inline_script"
    if ">" in normalized:
        return "shell_redirect"
    if normalized.startswith("rg "):
        return "search_rg"
    if normalized.startswith("get-content ") or normalized.startswith("select-string ") or normalized.startswith("resolve-path "):
        return "powershell_read"
    if normalized.startswith("get-item ") or normalized.startswith("get-childitem ") or normalized.startswith("test-path "):
        return "powershell_inspect"
    if normalized.startswith("git "):
        return "git_write" if git_subcommand(normalized) in GIT_WRITE_SUBCOMMANDS else "git_read"
    if normalized.startswith("python ") or normalized.startswith("py "):
        if "test" in normalized or "_test.py" in normalized or "pytest" in normalized:
            return "python_test"
        return "python_script"
    if normalized.startswith("go test") or normalized.startswith("go vet") or normalized.startswith("go build") or normalized.startswith("make "):
        return "build_test"
    return "other_shell"


def git_subcommand(normalized_git_command: str) -> str | None:
    parts = normalized_git_command.split()
    if not parts or parts[0] != "git":
        return None
    i = 1
    while i < len(parts):
        part = parts[i]
        if part in {"-c", "-C", "--git-dir", "--work-tree", "--namespace"}:
            i += 2
            continue
        if part.startswith("--git-dir=") or part.startswith("--work-tree=") or part.startswith("--namespace="):
            i += 1
            continue
        if part.startswith("-"):
            i += 1
            continue
        return part
    return None


def mutating_shell_family_counts(family_counts: dict[str, Any]) -> dict[str, int]:
    mutating = {"git_write"}
    return {
        family: int(count)
        for family, count in sorted(family_counts.items())
        if family in mutating and int(count or 0) > 0
    }


def git_gate_evidence(paths: list[Path]) -> dict[str, Any]:
    required = {
        "git_add": "DEFAULT_DENY",
        "git_commit": "DEFAULT_DENY",
        "git_push": "POLICY_BLOCK",
    }
    if not paths:
        return {"status": "UNKNOWN", "reason": "no gate reports supplied", "required": required}

    reports: list[dict[str, Any]] = []
    valid: dict[str, dict[str, Any]] = {}
    malformed: list[str] = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            malformed.append(path.name)
            continue
        if not isinstance(data, dict):
            malformed.append(path.name)
            continue
        preflight = data.get("preflight") if isinstance(data.get("preflight"), dict) else {}
        tool = data.get("tool")
        reason = preflight.get("reason")
        created_at = data.get("created_at")
        parsed_ts = parse_ts(created_at)
        compacted = {
            "path": path.name,
            "tool": tool,
            "status": data.get("status"),
            "created_at": created_at if parsed_ts is not None else None,
            "verdict": preflight.get("verdict"),
            "reason": reason,
            "expect_deny": bool(data.get("expect_deny")),
            "expect_reason": data.get("expect_reason"),
            "executed": bool(data.get("executed")),
            "dry_run": bool(data.get("dry_run")),
        }
        reports.append(compacted)
        if (
            isinstance(tool, str)
            and tool in required
            and data.get("status") == "DENIED_EXPECTED"
            and data.get("expect_deny") is True
            and data.get("executed") is False
            and preflight.get("verdict") == "DENY"
            and reason == required[tool]
            and parsed_ts is not None
        ):
            current = valid.get(tool)
            current_ts = parse_ts(current.get("created_at")) if isinstance(current, dict) else None
            if current is None or (current_ts is not None and parsed_ts > current_ts):
                valid[tool] = compacted

    missing = [tool for tool in required if tool not in valid]
    status = "PASS" if not missing and not malformed else "WARN"
    proved_at = None
    if valid:
        times = [parse_ts(row.get("created_at")) for row in valid.values()]
        times = [ts for ts in times if ts is not None]
        if times:
            proved_at = max(times).isoformat().replace("+00:00", "Z")
    return {
        "status": status,
        "required": required,
        "proved_at": proved_at,
        "missing": missing,
        "malformed": malformed,
        "reports": reports,
        "valid": {k: valid[k] for k in sorted(valid)},
        "privacy": {
            "copied_fields": ["report filename", "tool name", "status", "timestamp", "verdict", "reason", "dry-run/executed flags"],
            "dropped": ["command bodies", "command arguments", "command output"],
        },
    }


def codex_command_shape_audit(
    home: Path,
    repo_root: Path,
    codex_threads: dict[str, Path],
    since_text: Any,
    *,
    scope: str = "codex_threads",
) -> dict[str, Any]:
    since = parse_ts(since_text)
    if since is None:
        return {"status": "UNKNOWN", "reason": "no repair timestamp"}

    rows = 0
    malformed = 0
    tools: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    write_ops: Counter[str] = Counter()
    shell_calls = 0
    shell_arg_errors = 0
    inspected_sessions = 0
    mutating_sessions: list[dict[str, Any]] = []
    for thread_id, path in codex_threads.items():
        session_rows = 0
        session_shape_counts: Counter[str] = Counter()
        session_family_counts: Counter[str] = Counter()
        session_write_ops: Counter[str] = Counter()
        try:
            f = path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            malformed += 1
            continue
        with f:
            saw_session_row = False
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(row, dict):
                    malformed += 1
                    continue
                ts = parse_ts(row.get("timestamp"))
                if ts is None or ts < since:
                    continue
                payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
                name = payload.get("name")
                if not isinstance(name, str) or not name:
                    continue
                rows += 1
                session_rows += 1
                saw_session_row = True
                tools[name] += 1
                if name != "shell_command":
                    shape_counts["non_shell_tool"] += 1
                    session_shape_counts["non_shell_tool"] += 1
                    continue
                shell_calls += 1
                args = parse_arguments(payload.get("arguments"))
                command = args.get("command") if isinstance(args, dict) else None
                if not isinstance(command, str):
                    shell_arg_errors += 1
                    shape_counts["shell_missing_command_arg"] += 1
                    session_shape_counts["shell_missing_command_arg"] += 1
                    family_counts["shell_missing_command_arg"] += 1
                    session_family_counts["shell_missing_command_arg"] += 1
                    continue
                family = shell_command_family(command)
                family_counts[family] += 1
                session_family_counts[family] += 1
                shape, ops = classify_shell_shape(command, repo_root)
                shape_counts[shape] += 1
                session_shape_counts[shape] += 1
                write_ops.update(ops)
                session_write_ops.update(ops)
            if saw_session_row:
                inspected_sessions += 1
                mutating = mutating_shell_family_counts(session_family_counts)
                if mutating:
                    mutating_sessions.append(
                        {
                            "thread_id": thread_id,
                            "codex_session_file": path.name,
                            "tool_call_rows": session_rows,
                            "shell_family_counts": {k: session_family_counts[k] for k in sorted(session_family_counts)},
                            "shell_shape_counts": {k: session_shape_counts[k] for k in sorted(session_shape_counts)},
                            "mutating_shell_family_counts": mutating,
                            "write_op_counts": {k: session_write_ops[k] for k in sorted(session_write_ops)},
                        }
                    )

    status = "PASS"
    if not rows:
        status = "UNKNOWN"
    elif shape_counts.get("shell_out_of_tree_write_target"):
        status = "WARN"
    return {
        "status": status,
        "since": since.isoformat().replace("+00:00", "Z"),
        "scope": scope,
        "sessions_inspected": inspected_sessions,
        "threads_supplied": len(codex_threads),
        "tool_call_rows": rows,
        "tool_counts": {k: tools[k] for k in sorted(tools)},
        "shell_command_calls": shell_calls,
        "shell_argument_errors": shell_arg_errors,
        "shell_shape_counts": {k: shape_counts[k] for k in sorted(shape_counts)},
        "shell_family_counts": {k: family_counts[k] for k in sorted(family_counts)},
        "mutating_shell_sessions": mutating_sessions,
        "write_op_counts": {k: write_ops[k] for k in sorted(write_ops)},
        "malformed_rows": malformed,
        "privacy": {
            "copied_fields": ["thread id", "session filename", "tool names", "counts", "shell family categories", "shell shape categories", "write operation kinds"],
            "dropped": ["commands", "tool arguments", "tool results", "prompts", "model text"],
        },
    }


def actionable_gate(
    *,
    hook_fast_path: dict[str, Any],
    post_repair: dict[str, Any],
    command_shapes: dict[str, Any],
    delegate_total: int,
    stop_total: int,
    max_delegates: int | None,
    git_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    residual: list[str] = []
    unknown: list[str] = []

    if hook_fast_path.get("status") != "PASS":
        reasons.append("Codex hook fast path is not PASS")

    if post_repair.get("observations"):
        delegate_count = int(post_repair.get("delegate_count") or 0)
        delegate_source = "post_repair"
    else:
        delegate_count = int(delegate_total)
        delegate_source = "recent_window"
        unknown.append("no post-repair Codex observation window")

    if max_delegates is not None and delegate_count > max_delegates:
        reasons.append(f"{delegate_source} delegate count exceeds budget")

    if stop_total:
        reasons.append("stop blocks or uncleared StopFailure API-wall breaker markers are present")

    shape_counts = command_shapes.get("shell_shape_counts") if isinstance(command_shapes.get("shell_shape_counts"), dict) else {}
    family_counts = command_shapes.get("shell_family_counts") if isinstance(command_shapes.get("shell_family_counts"), dict) else {}
    mutating_families = mutating_shell_family_counts(family_counts)
    post_gate_shapes = (git_gate or {}).get("post_gate_command_shapes") if isinstance(git_gate, dict) else {}
    post_gate_family_counts = post_gate_shapes.get("shell_family_counts") if isinstance(post_gate_shapes, dict) and isinstance(post_gate_shapes.get("shell_family_counts"), dict) else {}
    post_gate_mutating_families = mutating_shell_family_counts(post_gate_family_counts)
    if shape_counts.get("shell_out_of_tree_write_target"):
        reasons.append("post-repair shell command shapes include out-of-tree write targets")
    if mutating_families:
        if isinstance(git_gate, dict) and git_gate.get("status") == "PASS":
            if post_gate_mutating_families:
                reasons.append("post-git-gate shell command families include opaque mutating operations")
            else:
                residual.append("HISTORICAL_GIT_WRITE_BEFORE_STRUCTURED_GATE")
        else:
            reasons.append("post-repair shell command families include opaque mutating operations")
    if command_shapes.get("shell_argument_errors"):
        reasons.append("post-repair shell command arguments could not be parsed")
    if command_shapes.get("status") == "UNKNOWN":
        unknown.append("post-repair command-shape evidence is missing")

    if shape_counts.get("shell_no_write_target_detected") and not reasons:
        residual.append("HOST_SHELL_OPACITY")
    if post_repair.get("unknown_tree_admission_warnings") and not reasons:
        residual.append("UNKNOWN_TREE_WARNINGS")

    status = "PASS"
    if reasons:
        status = "WARN"
    elif unknown:
        status = "UNKNOWN"

    return {
        "status": status,
        "reasons": reasons,
        "unknowns": unknown,
        "residual": list(dict.fromkeys(residual)),
        "delegate_source": delegate_source,
        "delegate_count": delegate_count,
        "max_delegates": max_delegates,
        "stop_total": stop_total,
        "post_repair_unknown_tree_admission_warnings": int(post_repair.get("unknown_tree_admission_warnings") or 0),
        "post_repair_shell_shape_counts": {k: shape_counts[k] for k in sorted(shape_counts)},
        "post_repair_shell_family_counts": {k: family_counts[k] for k in sorted(family_counts)},
        "post_repair_mutating_shell_family_counts": mutating_families,
        "git_gate_status": (git_gate or {}).get("status") if isinstance(git_gate, dict) else None,
        "post_git_gate_mutating_shell_family_counts": post_gate_mutating_families,
    }


def build_report(
    repo_root: Path,
    home: Path,
    *,
    limit: int,
    since_days: int,
    check_latest: bool = False,
    max_unknown_tree_rate: float | None = None,
    max_delegates: int | None = None,
    gate_reports: list[Path] | None = None,
) -> dict[str, Any]:
    witness = load_witness_module()
    codex_threads = discover_codex_threads(home, since_days=since_days)
    streams = discover_streams(repo_root, codex_threads, limit)
    args = argparse.Namespace(repo_root=repo_root)
    hook_fast_path = codex_hook_fast_path(home)
    post_repair = codex_observations_since(repo_root, hook_fast_path.get("repaired_at"))
    workspace_stop = workspace_stop_failure_audit(repo_root, since_days=since_days, limit=limit)
    audited_codex_threads = {thread_id: codex_threads[thread_id] for thread_id, _path in streams if thread_id in codex_threads}
    git_gate = git_gate_evidence(gate_reports or [])
    if hook_fast_path.get("repaired_at"):
        hook_fast_path["post_repair_observations"] = post_repair
        hook_fast_path["post_repair_command_shapes"] = codex_command_shape_audit(
            home,
            repo_root,
            audited_codex_threads,
            hook_fast_path.get("repaired_at"),
            scope="audited_dos_streams",
        )
    if git_gate.get("status") == "PASS" and git_gate.get("proved_at"):
        git_gate["post_gate_command_shapes"] = codex_command_shape_audit(
            home,
            repo_root,
            audited_codex_threads,
            git_gate.get("proved_at"),
            scope="post_git_gate_audited_dos_streams",
        )

    sessions: list[dict[str, Any]] = []
    for thread_id, _path in streams:
        audit = witness.dos_session_audit(args, thread_id)
        sessions.append(compact_session(audit, codex_threads[thread_id]))

    total_steps = sum(s["steps"] for s in sessions)
    total_pretool = sum(s["pretool_calls"] for s in sessions)
    total_unknown = sum(s["unknown_tree_admission_warnings"] for s in sessions)
    tool_counts: Counter[str] = Counter()
    for session in sessions:
        for tool, count in (session.get("tool_counts") or {}).items():
            tool_counts[str(tool)] += int(count)

    unknown_rate = (total_unknown / total_pretool) if total_pretool else None
    warned = [s for s in sessions if s.get("status") == "WARN"]
    bash_steps = tool_counts.get("Bash", 0)
    recommendations: list[str] = []
    unknown_threshold = (
        max_unknown_tree_rate
        if max_unknown_tree_rate is not None
        else witness.DOS_UNKNOWN_TREE_WARN_THRESHOLD
    )
    if sessions and total_steps and bash_steps / total_steps > 0.8:
        recommendations.append("recent Codex DOS streams are Bash-dominated; prefer path-visible tools or narrower shell commands")
    if unknown_rate is not None and unknown_rate > unknown_threshold:
        recommendations.append("unknown-tree admission warning rate exceeds the transfer-playbook threshold")
    delegate_total = sum(s["delegate_count"] for s in sessions)
    if delegate_total and (max_delegates is None or delegate_total > max_delegates):
        if post_repair.get("observations") and not post_repair.get("delegate_count"):
            recommendations.append("recent-window delegate count includes pre-repair history; post-repair Codex delegates are zero")
        else:
            recommendations.append("native DOS hook delegates are present; inspect fallback reasons upstream")
    if hook_fast_path.get("status") == "WARN" and hook_fast_path.get("codex_python_cli_hooks"):
        recommendations.append("Codex hook manifest uses the Python hook path; track native-fast-path wiring separately from package freshness")
    if hook_fast_path.get("status") == "WARN" and hook_fast_path.get("codex_powershell_native_hooks"):
        recommendations.append("Codex hook manifest uses the PowerShell native launcher; rerun the hook doctor so Codex starts through shell:bash and bin/dos-hook")
    if post_repair.get("status") == "WARN":
        if post_repair.get("unknown_tree_admission_warnings") and not post_repair.get("delegate_count"):
            recommendations.append("fast path is repaired; remaining post-repair issue is unknown-tree admission for opaque Codex host calls")
        else:
            recommendations.append("post-repair Codex hook observations still include delegates or unknown-tree warnings")
    command_shapes = hook_fast_path.get("post_repair_command_shapes") if isinstance(hook_fast_path.get("post_repair_command_shapes"), dict) else {}
    shape_counts = command_shapes.get("shell_shape_counts") if isinstance(command_shapes.get("shell_shape_counts"), dict) else {}
    if shape_counts.get("shell_no_write_target_detected"):
        recommendations.append("post-repair shell usage is mostly opaque read/inspect commands; prefer host-visible read/search tools when available")
    if shape_counts.get("shell_out_of_tree_write_target"):
        recommendations.append("post-repair shell usage includes out-of-tree write targets; inspect repo-guard findings before continuing")
    family_counts = command_shapes.get("shell_family_counts") if isinstance(command_shapes.get("shell_family_counts"), dict) else {}
    mutating_families = mutating_shell_family_counts(family_counts)
    if mutating_families:
        recommendations.append("post-repair shell usage includes opaque mutating git operations; route commit/push/add through explicit operator gates")
    post_gate_shapes = git_gate.get("post_gate_command_shapes") if isinstance(git_gate.get("post_gate_command_shapes"), dict) else {}
    post_gate_families = post_gate_shapes.get("shell_family_counts") if isinstance(post_gate_shapes.get("shell_family_counts"), dict) else {}
    if mutating_shell_family_counts(post_gate_families):
        recommendations.append("structured git gates have stale evidence; rerun expected-deny git gate probes after the latest opaque git mutation")
    if sum(s["stop_blocks"] + s["stop_failures_total"] for s in sessions):
        recommendations.append("stop-hook blocks/failures appeared; review before treating affected sessions as closed")
    workspace_stop_total = int(workspace_stop.get("total_failures") or 0)
    workspace_stop_active_total = int(workspace_stop.get("active_consecutive_total") or 0)
    workspace_stop_recent_active_total = int(workspace_stop.get("recent_active_consecutive_total") or 0)
    workspace_stop_stale_active_total = int(workspace_stop.get("stale_active_consecutive_total") or 0)
    if workspace_stop_total:
        recommendations.append("workspace StopFailure API-wall breaker markers appeared outside the audited Codex streams; inspect .dos/stop-failures")
    if workspace_stop_recent_active_total:
        recommendations.append("recent workspace StopFailure markers are still consecutive; clear the underlying stop-hook/API-wall failure before treating the seat as healed")
    if workspace_stop_stale_active_total:
        recommendations.append("stale workspace StopFailure markers still have nonzero consecutive counts; classify them as uncleared breaker state before treating them as live blockage")

    status = "PASS"
    if not sessions:
        status = "UNKNOWN"
        recommendations.append("no recent Codex sessions matched DOS streams")
    if warned or hook_fast_path.get("status") == "WARN" or workspace_stop.get("status") == "WARN":
        status = "WARN"
    stop_total = sum(s["stop_blocks"] + s["stop_failures_total"] for s in sessions) + workspace_stop_active_total
    actionability = actionable_gate(
        hook_fast_path=hook_fast_path,
        post_repair=post_repair,
        command_shapes=command_shapes,
        delegate_total=delegate_total,
        stop_total=stop_total,
        max_delegates=max_delegates,
        git_gate=git_gate,
    )

    return {
        "schema": SCHEMA,
        "status": status,
        "workspace": repo_root.name,
        "codex_home": home.name,
        "since_days": since_days,
        "limit": limit,
        "dos_version": local_dos_version(repo_root, check_latest=check_latest),
        "codex_hook_fast_path": hook_fast_path,
        "git_gate_evidence": git_gate,
        "workspace_stop_failures": workspace_stop,
        "budgets": {
            "max_unknown_tree_rate": unknown_threshold,
            "max_delegates": max_delegates,
        },
        "codex_threads_discovered": len(codex_threads),
        "sessions_audited": len(sessions),
        "summary": {
            "steps": total_steps,
            "tool_counts": {k: tool_counts[k] for k in sorted(tool_counts)},
            "observations": sum(s["observations"] for s in sessions),
            "pretool_calls": total_pretool,
            "unknown_tree_admission_warnings": total_unknown,
            "unknown_tree_warning_rate": round(unknown_rate, 6) if unknown_rate is not None else None,
            "unknown_tree_warning_threshold": unknown_threshold,
            "delegate_count": delegate_total,
            "stop_blocks": sum(s["stop_blocks"] for s in sessions),
            "stop_failures_total": sum(s["stop_failures_total"] for s in sessions),
            "workspace_stop_failures_total": workspace_stop_total,
            "workspace_stop_failure_markers": workspace_stop.get("markers"),
            "workspace_stop_failure_zero_markers": workspace_stop.get("zero_total_markers"),
            "workspace_stop_failure_nonzero_markers": workspace_stop.get("nonzero_total_markers"),
            "workspace_stop_failure_active_markers": workspace_stop.get("active_consecutive_markers"),
            "workspace_stop_failure_active_consecutive_total": workspace_stop_active_total,
            "workspace_stop_failure_active_recent_threshold_hours": workspace_stop.get("active_recent_threshold_hours"),
            "workspace_stop_failure_recent_active_markers": workspace_stop.get("recent_active_consecutive_markers"),
            "workspace_stop_failure_recent_active_consecutive_total": workspace_stop_recent_active_total,
            "workspace_stop_failure_stale_active_markers": workspace_stop.get("stale_active_consecutive_markers"),
            "workspace_stop_failure_stale_active_consecutive_total": workspace_stop_stale_active_total,
            "workspace_stop_failure_healed_nonzero_markers": workspace_stop.get("healed_nonzero_markers"),
            "workspace_stop_failure_transcript_evidence_tags": workspace_stop.get("transcript_evidence_tag_counts"),
            "workspace_stop_failure_origin_counts": workspace_stop.get("origin_counts"),
            "workspace_stop_failure_active_origin_counts": workspace_stop.get("active_origin_counts"),
            "workspace_stop_failure_recent_active_origin_counts": workspace_stop.get("recent_active_origin_counts"),
            "workspace_stop_failure_stale_active_origin_counts": workspace_stop.get("stale_active_origin_counts"),
            "workspace_stop_failure_nonzero_origin_counts": workspace_stop.get("nonzero_origin_counts"),
            "workspace_stop_failure_settlement_action_counts": workspace_stop.get("settlement_action_counts"),
            "workspace_stop_failure_active_settlement_action_counts": workspace_stop.get("active_settlement_action_counts"),
            "workspace_stop_failure_recent_active_settlement_action_counts": workspace_stop.get("recent_active_settlement_action_counts"),
            "workspace_stop_failure_stale_active_settlement_action_counts": workspace_stop.get("stale_active_settlement_action_counts"),
            "warned_sessions": len(warned),
        },
        "actionability": actionability,
        "sessions": sessions,
        "recommendations": recommendations,
        "privacy": {
            "copied_fields": ["thread id", "session filename", "timestamps", "tool names", "counts", "latencies"],
            "dropped": ["prompts", "tool arguments", "tool results", "hook command bodies", "diffs", "model text"],
        },
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact_stop_failure_session(entry: dict[str, Any]) -> dict[str, Any]:
    transcript = entry.get("transcript") if isinstance(entry.get("transcript"), dict) else {}
    summary = entry.get("transcript_summary") if isinstance(entry.get("transcript_summary"), dict) else {}
    return {
        "session_id": entry.get("session_id"),
        "total": safe_int(entry.get("total")),
        "consecutive": safe_int(entry.get("consecutive")),
        "mtime": entry.get("mtime"),
        "origin": entry.get("origin"),
        "settlement_action": entry.get("settlement_action"),
        "transcript_status": transcript.get("status"),
        "transcript_account": transcript.get("account"),
        "transcript_project": transcript.get("project"),
        "transcript_evidence_tags": summary.get("evidence_tags") or [],
    }


def stop_failure_top_sessions(stop: dict[str, Any], limit: int = 5, key: str = "top_nonzero") -> list[dict[str, Any]]:
    entries = stop.get(key) if isinstance(stop.get(key), list) else []
    out: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or safe_int(entry.get("total")) <= 0:
            continue
        out.append(compact_stop_failure_session(entry))
        if len(out) >= max(1, limit):
            break
    return out


def format_stop_failure_session(entry: dict[str, Any]) -> str:
    bits = [
        str(entry.get("session_id")),
        f"total={safe_int(entry.get('total'))}",
        f"consecutive={safe_int(entry.get('consecutive'))}",
    ]
    if entry.get("transcript_status"):
        bits.append(f"transcript={entry.get('transcript_status')}")
    if entry.get("origin"):
        bits.append(f"origin={entry.get('origin')}")
    if entry.get("settlement_action"):
        bits.append(f"settlement={entry.get('settlement_action')}")
    if entry.get("transcript_project"):
        bits.append(f"project={entry.get('transcript_project')}")
    tags = entry.get("transcript_evidence_tags") if isinstance(entry.get("transcript_evidence_tags"), list) else []
    if tags:
        bits.append("evidence=" + ",".join(str(tag) for tag in tags[:3]))
    return " ".join(bits)


def render_debt_packet(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    hook = report.get("codex_hook_fast_path") if isinstance(report.get("codex_hook_fast_path"), dict) else {}
    post = hook.get("post_repair_observations") if isinstance(hook.get("post_repair_observations"), dict) else {}
    shapes = hook.get("post_repair_command_shapes") if isinstance(hook.get("post_repair_command_shapes"), dict) else {}
    actionability = report.get("actionability") if isinstance(report.get("actionability"), dict) else {}
    dos_version = report.get("dos_version") if isinstance(report.get("dos_version"), dict) else {}
    git_gate = report.get("git_gate_evidence") if isinstance(report.get("git_gate_evidence"), dict) else {}
    workspace_stop = report.get("workspace_stop_failures") if isinstance(report.get("workspace_stop_failures"), dict) else {}
    stop_top = stop_failure_top_sessions(workspace_stop)
    stop_active = stop_failure_top_sessions(workspace_stop, key="top_active")
    stop_recent_active = stop_failure_top_sessions(workspace_stop, key="top_recent_active")
    mutating_families = actionability.get("post_repair_mutating_shell_family_counts") or {}
    action_reasons = [str(reason) for reason in actionability.get("reasons") or []]
    if any("StopFailure API-wall" in reason for reason in action_reasons):
        interpretation = (
            "The DOS native fast path is repaired and post-repair delegate count is zero. "
            "The current actionable WARN is workspace StopFailure API-wall breaker state: "
            f"{summary.get('workspace_stop_failure_recent_active_markers')} markers are recent "
            f"(<= {summary.get('workspace_stop_failure_active_recent_threshold_hours')}h) with "
            f"{summary.get('workspace_stop_failure_recent_active_consecutive_total')} consecutive "
            f"StopFailure counts, while {summary.get('workspace_stop_failure_stale_active_markers')} "
            f"older markers still carry {summary.get('workspace_stop_failure_stale_active_consecutive_total')} "
            "uncleared consecutive counts. Recent marker origins are "
            f"{json.dumps(summary.get('workspace_stop_failure_recent_active_origin_counts') or {}, sort_keys=True)}. "
            "Recommended settlement classes are "
            f"{json.dumps(summary.get('workspace_stop_failure_active_settlement_action_counts') or {}, sort_keys=True)}. "
            "The one-day historical friction total is "
            f"{summary.get('workspace_stop_failures_total')} API-wall failures across "
            f"{summary.get('workspace_stop_failure_nonzero_markers')} nonzero markers; "
            f"{summary.get('workspace_stop_failure_healed_nonzero_markers')} of those are healed "
            "to consecutive=0. This is separate from Codex verify-on-stop blocks, which remain "
            "zero in the audited Codex streams."
        )
    elif any("mutating operations" in reason for reason in action_reasons):
        interpretation = (
            "The DOS native fast path is repaired and post-repair delegate count is zero. "
            "The current actionable WARN is opaque mutating shell usage: Codex ran shell "
            "families such as `git_write` without a structured tool boundary, so fak/DOS "
            "could not apply the named operation gate before the mutation-shaped call."
        )
    else:
        interpretation = (
            "The DOS native fast path is repaired and post-repair delegate count is zero. "
            "The remaining strict WARN is host-shell opacity: Codex exposes shell calls as opaque "
            "commands, so the hook cannot derive a precise file-tree footprint for read/inspect calls."
        )

    lines = [
        "# Codex DOS Host-Opacity Debt",
        "",
        "## Summary",
        "",
        f"- audit_status: `{report.get('status')}`",
        f"- actionability_status: `{actionability.get('status')}`",
        f"- residual: `{', '.join(actionability.get('residual') or []) or 'none'}`",
        f"- dos_kernel_version: `{dos_version.get('version')}`",
        f"- dos_kernel_using_latest: `{dos_version.get('using_latest')}`",
        f"- codex_hook_fast_path: `{hook.get('status')}` `{json.dumps(hook.get('codex_command_modes') or {}, sort_keys=True)}`",
        f"- git_gate_status: `{git_gate.get('status')}` proved_at=`{git_gate.get('proved_at')}`",
        f"- sessions_audited: `{report.get('sessions_audited')}`",
        "",
        "## Evidence",
        "",
        f"- recent_window_unknown_tree_rate: `{summary.get('unknown_tree_warning_rate')}`",
        f"- recent_window_delegates: `{summary.get('delegate_count')}`",
        f"- workspace_stop_failure_markers: `{summary.get('workspace_stop_failure_markers')}`",
        f"- workspace_stop_failure_nonzero_markers: `{summary.get('workspace_stop_failure_nonzero_markers')}`",
        f"- workspace_stop_failure_zero_markers: `{summary.get('workspace_stop_failure_zero_markers')}`",
        f"- workspace_stop_failure_active_markers: `{summary.get('workspace_stop_failure_active_markers')}`",
        f"- workspace_stop_failure_active_consecutive_total: `{summary.get('workspace_stop_failure_active_consecutive_total')}`",
        f"- workspace_stop_failure_active_recent_threshold_hours: `{summary.get('workspace_stop_failure_active_recent_threshold_hours')}`",
        f"- workspace_stop_failure_recent_active_markers: `{summary.get('workspace_stop_failure_recent_active_markers')}`",
        f"- workspace_stop_failure_recent_active_consecutive_total: `{summary.get('workspace_stop_failure_recent_active_consecutive_total')}`",
        f"- workspace_stop_failure_stale_active_markers: `{summary.get('workspace_stop_failure_stale_active_markers')}`",
        f"- workspace_stop_failure_stale_active_consecutive_total: `{summary.get('workspace_stop_failure_stale_active_consecutive_total')}`",
        f"- workspace_stop_failure_healed_nonzero_markers: `{summary.get('workspace_stop_failure_healed_nonzero_markers')}`",
        f"- workspace_stop_failure_origin_counts: `{json.dumps(summary.get('workspace_stop_failure_origin_counts') or {}, sort_keys=True)}`",
        f"- workspace_stop_failure_active_origin_counts: `{json.dumps(summary.get('workspace_stop_failure_active_origin_counts') or {}, sort_keys=True)}`",
        f"- workspace_stop_failure_recent_active_origin_counts: `{json.dumps(summary.get('workspace_stop_failure_recent_active_origin_counts') or {}, sort_keys=True)}`",
        f"- workspace_stop_failure_stale_active_origin_counts: `{json.dumps(summary.get('workspace_stop_failure_stale_active_origin_counts') or {}, sort_keys=True)}`",
        f"- workspace_stop_failure_settlement_action_counts: `{json.dumps(summary.get('workspace_stop_failure_settlement_action_counts') or {}, sort_keys=True)}`",
        f"- workspace_stop_failure_active_settlement_action_counts: `{json.dumps(summary.get('workspace_stop_failure_active_settlement_action_counts') or {}, sort_keys=True)}`",
        f"- workspace_stop_failure_recent_active_settlement_action_counts: `{json.dumps(summary.get('workspace_stop_failure_recent_active_settlement_action_counts') or {}, sort_keys=True)}`",
        f"- workspace_stop_failure_stale_active_settlement_action_counts: `{json.dumps(summary.get('workspace_stop_failure_stale_active_settlement_action_counts') or {}, sort_keys=True)}`",
        f"- workspace_stop_failures_total: `{summary.get('workspace_stop_failures_total')}`",
        f"- workspace_stop_failures_by_day: `{json.dumps(workspace_stop.get('by_day') or {}, sort_keys=True)}`",
        f"- workspace_stop_failure_top_sessions: `{json.dumps(stop_top, sort_keys=True)}`",
        f"- workspace_stop_failure_top_recent_active_sessions: `{json.dumps(stop_recent_active, sort_keys=True)}`",
        f"- workspace_stop_failure_top_active_sessions: `{json.dumps(stop_active, sort_keys=True)}`",
        f"- workspace_stop_failure_settlement_plan: `{json.dumps(workspace_stop.get('settlement_plan') or {}, sort_keys=True)}`",
        f"- workspace_stop_failure_transcript_evidence_tags: `{json.dumps(summary.get('workspace_stop_failure_transcript_evidence_tags') or {}, sort_keys=True)}`",
        f"- post_repair_observations: `{post.get('observations')}`",
        f"- post_repair_delegates: `{post.get('delegate_count')}`",
        f"- post_repair_unknown_tree_warnings: `{post.get('unknown_tree_admission_warnings')}`",
        f"- post_repair_shell_shapes: `{json.dumps(shapes.get('shell_shape_counts') or {}, sort_keys=True)}`",
        f"- post_repair_shell_families: `{json.dumps(shapes.get('shell_family_counts') or {}, sort_keys=True)}`",
        f"- post_repair_mutating_shell_families: `{json.dumps(mutating_families, sort_keys=True)}`",
        f"- post_repair_mutating_sessions: `{json.dumps(shapes.get('mutating_shell_sessions') or [], sort_keys=True)}`",
        f"- post_git_gate_shell_families: `{json.dumps(((git_gate.get('post_gate_command_shapes') or {}).get('shell_family_counts') if isinstance(git_gate.get('post_gate_command_shapes'), dict) else {}) or {}, sort_keys=True)}`",
        f"- post_git_gate_mutating_shell_families: `{json.dumps(actionability.get('post_git_gate_mutating_shell_family_counts') or {}, sort_keys=True)}`",
        f"- post_repair_write_ops: `{json.dumps(shapes.get('write_op_counts') or {}, sort_keys=True)}`",
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "## Requested Upstream Improvement",
        "",
        "- Review `.dos/stop-failures` as StopFailure API-wall breaker state before treating actionability as PASS; prioritize recent nonzero `consecutive` markers, then decide whether stale nonzero markers need a success reset or can be archived as old breaker state.",
        "- Start with `workspace_stop_failure_top_recent_active_sessions`, then `workspace_stop_failure_top_active_sessions`, then `workspace_stop_failure_top_sessions`: these are already mapped to sanitized Claude transcript account/project metadata and count-only transcript evidence when available.",
        "- Route mutating Git operations through structured fak-gated tools such as `git_add`, `git_commit`, and `git_push`; do not run them as opaque shell commands.",
        "- Include path/footprint metadata in Codex tool-call hook payloads, or expose host-visible read/search/list tools with path arguments.",
        "- Preserve the current privacy boundary: the audit needs tool names, path metadata, timestamps, and counts, not prompts, command bodies, tool output, diffs, or model text.",
        "- Keep shell command bodies out of durable reports; classify locally and emit only categories such as `shell_no_write_target_detected` or `shell_out_of_tree_write_target`.",
        "",
        "## Privacy Boundary",
        "",
        "- copied: session filenames, thread ids, timestamps, tool names, counts, latency summaries, shell shape categories, shell family categories, write operation kinds",
        "- dropped: prompts, command bodies, tool arguments, tool results, diffs, model text, hook command bodies",
        "",
    ]
    return "\n".join(lines)


def write_debt_packet(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_debt_packet(report), encoding="utf-8")


def render(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    workspace_stop = report.get("workspace_stop_failures") if isinstance(report.get("workspace_stop_failures"), dict) else {}
    lines = [
        f"codex DOS recent audit: {report.get('status')}",
        f"  sessions audited: {report.get('sessions_audited')} of {report.get('codex_threads_discovered')} discovered Codex threads",
        f"  steps: {summary.get('steps')}  tools: {json.dumps(summary.get('tool_counts') or {}, sort_keys=True)}",
        f"  unknown-tree warnings: {summary.get('unknown_tree_admission_warnings')} / {summary.get('pretool_calls')} pretool calls"
        f" ({summary.get('unknown_tree_warning_rate')})",
        f"  delegates: {summary.get('delegate_count')}  stop blocks: {summary.get('stop_blocks')}  stop failures: {summary.get('stop_failures_total')}",
        "  workspace StopFailure API-wall failures: "
        f"{summary.get('workspace_stop_failures_total')} across "
        f"{summary.get('workspace_stop_failure_nonzero_markers')} nonzero markers "
        f"({summary.get('workspace_stop_failure_zero_markers')} zero-total markers)",
        "  active StopFailure blockers: "
        f"{summary.get('workspace_stop_failure_active_markers')} markers, "
        f"{summary.get('workspace_stop_failure_active_consecutive_total')} consecutive failures "
        f"({summary.get('workspace_stop_failure_recent_active_markers')} recent <= "
        f"{summary.get('workspace_stop_failure_active_recent_threshold_hours')}h, "
        f"{summary.get('workspace_stop_failure_stale_active_markers')} stale; "
        f"{summary.get('workspace_stop_failure_healed_nonzero_markers')} healed nonzero markers; "
        f"recent_origins={json.dumps(summary.get('workspace_stop_failure_recent_active_origin_counts') or {}, sort_keys=True)}; "
        f"settlement={json.dumps(summary.get('workspace_stop_failure_active_settlement_action_counts') or {}, sort_keys=True)})",
    ]
    top_recent_active_stop = stop_failure_top_sessions(workspace_stop, limit=3, key="top_recent_active")
    if top_recent_active_stop:
        lines.append(
            "  top recent active StopFailure sessions: "
            + "; ".join(format_stop_failure_session(entry) for entry in top_recent_active_stop)
        )
    top_active_stop = stop_failure_top_sessions(workspace_stop, limit=3, key="top_active")
    if top_active_stop:
        lines.append(
            "  top active StopFailure sessions: "
            + "; ".join(format_stop_failure_session(entry) for entry in top_active_stop)
        )
    top_stop = stop_failure_top_sessions(workspace_stop, limit=3)
    if top_stop:
        lines.append(
            "  top StopFailure sessions: "
            + "; ".join(format_stop_failure_session(entry) for entry in top_stop)
        )
    hook_fast_path = report.get("codex_hook_fast_path") if isinstance(report.get("codex_hook_fast_path"), dict) else {}
    if hook_fast_path:
        lines.append(
            "  codex hook fast path: "
            f"{hook_fast_path.get('status')} "
            f"{json.dumps(hook_fast_path.get('codex_command_modes') or {}, sort_keys=True)}"
        )
        doctor = hook_fast_path.get("doctor") if isinstance(hook_fast_path.get("doctor"), dict) else {}
        if hook_fast_path.get("status") == "WARN" and doctor.get("dry_run"):
            lines.append(f"  hook doctor: {doctor.get('dry_run')}")
        projection = hook_fast_path.get("repair_projection") if isinstance(hook_fast_path.get("repair_projection"), dict) else {}
        if projection:
            lines.append(
                "  hook repair projection: "
                f"{json.dumps(projection.get('projected_codex_command_modes') or {}, sort_keys=True)}"
            )
        post_repair = hook_fast_path.get("post_repair_observations") if isinstance(hook_fast_path.get("post_repair_observations"), dict) else {}
        if post_repair:
            lines.append(
                "  post-repair codex observations: "
                f"{post_repair.get('status')} "
                f"{post_repair.get('observations')} rows, "
                f"{post_repair.get('delegate_count')} delegates, "
                f"{post_repair.get('unknown_tree_admission_warnings')} unknown-tree warnings"
            )
        command_shapes = hook_fast_path.get("post_repair_command_shapes") if isinstance(hook_fast_path.get("post_repair_command_shapes"), dict) else {}
        if command_shapes:
            lines.append(
                "  post-repair command shapes: "
                f"{json.dumps(command_shapes.get('shell_shape_counts') or {}, sort_keys=True)}"
            )
            lines.append(
                "  post-repair shell families: "
                f"{json.dumps(command_shapes.get('shell_family_counts') or {}, sort_keys=True)}"
            )
    for rec in report.get("recommendations") or []:
        lines.append(f"  recommendation: {rec}")
    actionability = report.get("actionability") if isinstance(report.get("actionability"), dict) else {}
    if actionability:
        lines.append(
            "  actionable gate: "
            f"{actionability.get('status')} "
            f"residual={json.dumps(actionability.get('residual') or [])}"
        )
        if actionability.get("reasons"):
            lines.append(f"  actionable reasons: {json.dumps(actionability.get('reasons') or [])}")
    git_gate = report.get("git_gate_evidence") if isinstance(report.get("git_gate_evidence"), dict) else {}
    if git_gate:
        lines.append(
            "  git structured gates: "
            f"{git_gate.get('status')} "
            f"proved_at={git_gate.get('proved_at')}"
        )
        post_gate = git_gate.get("post_gate_command_shapes") if isinstance(git_gate.get("post_gate_command_shapes"), dict) else {}
        if post_gate:
            lines.append(
                "  post-git-gate shell families: "
                f"{json.dumps(post_gate.get('shell_family_counts') or {}, sort_keys=True)}"
            )
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=default_repo_root())
    p.add_argument("--codex-home", type=Path, default=codex_home())
    p.add_argument("--since-days", type=int, default=7)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--out", type=Path)
    p.add_argument("--out-debt", type=Path, help="write a sanitized Markdown host-opacity debt packet")
    p.add_argument("--gate-report", action="append", type=Path, default=[], help="include a JSON report from tools/codex_fak_gate.py; repeat for multiple reports")
    p.add_argument("--check-latest", action="store_true", help="query PyPI JSON and compare the local dos-kernel version")
    p.add_argument("--max-unknown-tree-rate", type=float, help="budget for aggregate unknown-tree warning rate")
    p.add_argument("--max-delegates", type=int, help="budget for aggregate native-hook delegate count")
    p.add_argument("--fail-on-warn", action="store_true", help="exit 1 when the report status is WARN")
    p.add_argument("--fail-on-actionable-warn", action="store_true", help="exit nonzero when the post-repair actionable gate is not PASS")
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    report = build_report(
        args.repo_root.resolve(),
        args.codex_home.resolve(),
        limit=args.limit,
        since_days=args.since_days,
        check_latest=args.check_latest,
        max_unknown_tree_rate=args.max_unknown_tree_rate,
        max_delegates=args.max_delegates,
        gate_reports=args.gate_report,
    )
    if args.out:
        write_report(args.out, report)
    if args.out_debt:
        write_debt_packet(args.out_debt, report)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render(report))
    if args.fail_on_actionable_warn and report.get("actionability", {}).get("status") != "PASS":
        return 1
    if args.fail_on_warn and report["status"] == "WARN":
        return 1
    return 0 if report["status"] in {"PASS", "WARN", "UNKNOWN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
