#!/usr/bin/env python3
"""One-command operator gate for the generic DOS supervisor canary.

Default mode is dry-run: compute readiness, safety, the exact bounded enact
command, and the audit card without launching workers. Passing ``--live`` is the
only path that calls the watchdog's live enact path; the same safety guard still
applies, and the command folds a post-run audit before exiting.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import dos_supervisor_canary_audit as audit
import dos_supervisor_status as status
import dos_supervisor_watchdog as watchdog

SCHEMA = "fleet-dos-supervisor-canary/1"
RECORD_SCHEMA = "fleet-dos-supervisor-canary-record/1"
HISTORY_SCHEMA = "fleet-dos-supervisor-canary-history/1"


def run_canary(
    *,
    workspace: Path,
    live: bool,
    target: int | None,
    max_ticks: int,
    timeout_s: int,
    allow_dirty: bool = False,
    allow_stale: bool = False,
    readiness: dict[str, Any] | None = None,
    safety: dict[str, Any] | None = None,
    runner: watchdog.Runner | None = None,
    post_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = workspace.resolve()
    readiness_payload = readiness if readiness is not None else status.collect(root)
    safety_payload = safety if safety is not None else watchdog.workspace_safety(root)
    dry_run = watchdog.run_watchdog(
        workspace=root,
        target=target,
        max_ticks=max_ticks,
        live=False,
        timeout_s=timeout_s,
        readiness=readiness_payload,
        safety=safety_payload,
    )
    pre_audit = audit.build_payload(workspace=root, readiness=readiness_payload, dry_run=dry_run)

    if not live:
        return _payload(
            action="dry_run",
            ok=bool(pre_audit.get("ok")),
            workspace=root,
            pre_audit=pre_audit,
            launch=dry_run,
            post_audit=None,
            reason=str(pre_audit.get("reason") or dry_run.get("reason") or ""),
            next_action=str(pre_audit.get("next_action") or ""),
        )

    launch = watchdog.run_watchdog(
        workspace=root,
        target=target,
        max_ticks=max_ticks,
        live=True,
        timeout_s=timeout_s,
        readiness=readiness_payload,
        runner=runner,
        safety=safety_payload,
        allow_dirty=allow_dirty,
        allow_stale=allow_stale,
    )
    post = post_audit if post_audit is not None else audit.collect(root)
    ok = bool(launch.get("ok")) and bool(post.get("ok"))
    action = "live_enacted" if launch.get("action") == "enacted" else "live_refused"
    reason = str(
        post.get("reason")
        or launch.get("reason")
        or pre_audit.get("reason")
        or ""
    )
    next_action = str(post.get("next_action") or launch.get("reason") or "")
    return _payload(
        action=action,
        ok=ok,
        workspace=root,
        pre_audit=pre_audit,
        launch=launch,
        post_audit=post,
        reason=reason,
        next_action=next_action,
    )


def _payload(
    *,
    action: str,
    ok: bool,
    workspace: Path,
    pre_audit: dict[str, Any],
    launch: dict[str, Any],
    post_audit: dict[str, Any] | None,
    reason: str,
    next_action: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "ok": ok,
        "action": action,
        "reason": reason,
        "next_action": next_action,
        "workspace": str(workspace),
        "live": action.startswith("live_"),
        "pre_audit": pre_audit,
        "launch": launch,
        "post_audit": post_audit,
    }


def render(payload: dict[str, Any]) -> str:
    if payload.get("schema") == HISTORY_SCHEMA:
        return render_history(payload)
    launch = payload.get("launch") or {}
    post = payload.get("post_audit") or payload.get("pre_audit") or {}
    command = " ".join(launch.get("command") or [])
    lines = [
        f"DOS supervisor canary: {payload.get('action')} ({'ok' if payload.get('ok') else 'blocked'})",
        f"reason: {payload.get('reason')}",
        f"next: {payload.get('next_action')}",
    ]
    if command:
        lines.append(f"command: {command}")
    if post:
        lines.append(f"audit: {post.get('verdict')} ({post.get('finding')})")
    record = payload.get("record") or {}
    if record:
        lines.append(f"record: {record.get('history_path')} events={record.get('history', {}).get('events')}")
    return "\n".join(lines)


def default_record_dir(workspace: Path) -> Path:
    return workspace / "tools" / "_registry" / "dos_supervisor_canary"


def record_payload(
    payload: dict[str, Any],
    *,
    workspace: Path,
    record_dir: Path | None = None,
    now_utc: str | None = None,
) -> dict[str, Any]:
    root = workspace.resolve()
    out_dir = (record_dir or default_record_dir(root)).resolve()
    history_path = out_dir / "history.jsonl"
    latest_path = out_dir / "latest.json"
    event = event_from_payload(payload, now_utc=now_utc or utc_now())

    out_dir.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")
    write_json_atomic(latest_path, {"schema": RECORD_SCHEMA, "event": event, "payload": payload})

    history = summarize_history(root, record_dir=out_dir)
    return {
        "schema": RECORD_SCHEMA,
        "ok": True,
        "history_path": str(history_path),
        "latest_path": str(latest_path),
        "event": event,
        "history": history,
    }


def event_from_payload(payload: dict[str, Any], *, now_utc: str) -> dict[str, Any]:
    audit_payload = payload.get("post_audit") or payload.get("pre_audit") or {}
    launch = payload.get("launch") or {}
    blockers = blocker_keys(audit_payload)
    if not blockers and not payload.get("ok"):
        blockers = [f"canary:{payload.get('action') or 'unknown'}"]
    return {
        "schema": RECORD_SCHEMA,
        "recorded_utc": now_utc,
        "workspace": str(payload.get("workspace") or ""),
        "ok": bool(payload.get("ok")),
        "action": payload.get("action"),
        "live": bool(payload.get("live")),
        "reason": payload.get("reason") or audit_payload.get("reason") or launch.get("reason") or "",
        "next_action": payload.get("next_action") or audit_payload.get("next_action") or "",
        "audit": {
            "verdict": audit_payload.get("verdict"),
            "finding": audit_payload.get("finding"),
            "reason": audit_payload.get("reason"),
            "next_action": audit_payload.get("next_action"),
        },
        "launch": {
            "ok": launch.get("ok"),
            "action": launch.get("action"),
            "reason": launch.get("reason"),
            "live": launch.get("live"),
            "target": launch.get("target"),
            "max_ticks": launch.get("max_ticks"),
            "command": launch.get("command") or [],
            "safety_ok": (launch.get("safety") or {}).get("ok"),
        },
        "blockers": blockers,
    }


def blocker_keys(audit_payload: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for blocker in audit_payload.get("blockers") or []:
        if not isinstance(blocker, dict):
            continue
        kind = str(blocker.get("kind") or "unknown").strip() or "unknown"
        keys.append(kind)
    if not keys and not audit_payload.get("ok") and audit_payload.get("finding"):
        keys.append(str(audit_payload.get("finding")))
    return keys


def summarize_history(
    workspace: Path,
    *,
    record_dir: Path | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    root = workspace.resolve()
    out_dir = (record_dir or default_record_dir(root)).resolve()
    history_path = out_dir / "history.jsonl"
    rows, invalid = read_history(history_path, limit=limit)
    latest = rows[-1] if rows else None
    blocker_counts = collections.Counter(
        blocker
        for row in rows
        for blocker in row.get("blockers") or []
    )
    recurring = [
        {
            "blocker": blocker,
            "count": count,
            "command": "python tools/dos_supervisor_canary.py --history --json",
            "route": "add this row to PLAN-fleet-always-on-dispatch-2026-06-19.md section 5 or the fleet findings queue",
        }
        for blocker, count in sorted(blocker_counts.items())
        if count > 1
    ]
    latest_blocked = bool(latest and not latest.get("ok"))
    if not rows:
        verdict = "NO_HISTORY"
        ok = True
        reason = "no canary records have been written yet"
        next_action = "run python tools/dos_supervisor_canary.py --record --json after an operator-approved canary check"
    elif recurring:
        verdict = "RECURRING_BLOCKER"
        ok = False
        reason = f"{len(recurring)} blocker kind(s) recurred in canary history"
        next_action = "route the recurring blocker row into the plan or fleet findings queue"
    elif latest_blocked:
        verdict = "LATEST_BLOCKED"
        ok = False
        reason = str(latest.get("reason") or "latest recorded canary was blocked")
        next_action = str(latest.get("next_action") or "inspect latest canary record")
    else:
        verdict = "HISTORY_OK"
        ok = True
        reason = "recorded canary history has no current or recurring blocker"
        next_action = "continue recording dry-run and operator-approved live canary outcomes"

    return {
        "schema": HISTORY_SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "reason": reason,
        "next_action": next_action,
        "workspace": str(root),
        "history_path": str(history_path),
        "latest_path": str(out_dir / "latest.json"),
        "events": len(rows),
        "invalid_lines": invalid,
        "blocked_events": sum(1 for row in rows if not row.get("ok")),
        "live_events": sum(1 for row in rows if row.get("live")),
        "dry_run_events": sum(1 for row in rows if not row.get("live")),
        "latest": latest,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "recurring_blockers": recurring,
        "findings": recurring,
    }


def read_history(path: Path, *, limit: int) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    invalid = 0
    if not path.exists():
        return rows, invalid
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            invalid += 1
            continue
        if isinstance(obj, dict):
            rows.append(obj)
        else:
            invalid += 1
    if limit > 0:
        rows = rows[-limit:]
    return rows, invalid


def render_history(payload: dict[str, Any]) -> str:
    lines = [
        f"DOS supervisor canary history: {payload.get('verdict')} ({'ok' if payload.get('ok') else 'action'})",
        f"reason: {payload.get('reason')}",
        f"events: {payload.get('events')} blocked={payload.get('blocked_events')} live={payload.get('live_events')}",
        f"next: {payload.get('next_action')}",
    ]
    for finding in payload.get("findings") or []:
        lines.append(
            f"finding: {finding.get('blocker')} count={finding.get('count')} command={finding.get('command')}"
        )
    return "\n".join(lines)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run or dry-run one guarded DOS supervisor canary.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--target", type=int, default=None, help="bounded canary target")
    ap.add_argument("--max-ticks", type=int, default=1, help="dos loop max ticks (default: 1)")
    ap.add_argument("--timeout-s", type=int, default=120, help="live enact timeout in seconds")
    ap.add_argument("--live", action="store_true", help="actually run the guarded canary")
    ap.add_argument("--allow-dirty", action="store_true", help="permit --live from a dirty worktree")
    ap.add_argument("--allow-stale", action="store_true", help="permit --live when the workspace is behind or diverged")
    ap.add_argument("--record", action="store_true", help="append a compact outcome to tools/_registry/dos_supervisor_canary")
    ap.add_argument("--record-dir", default="", help="override canary record directory")
    ap.add_argument("--history", action="store_true", help="read recorded canary history instead of running a canary check")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else status.repo_root()
    record_dir = Path(args.record_dir).resolve() if args.record_dir else None
    if args.history:
        payload = summarize_history(workspace, record_dir=record_dir)
    else:
        payload = run_canary(
            workspace=workspace,
            live=args.live,
            target=args.target,
            max_ticks=args.max_ticks,
            timeout_s=args.timeout_s,
            allow_dirty=args.allow_dirty,
            allow_stale=args.allow_stale,
        )
        if args.record:
            payload["record"] = record_payload(payload, workspace=workspace, record_dir=record_dir)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
