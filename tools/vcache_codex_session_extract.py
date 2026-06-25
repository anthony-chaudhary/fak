#!/usr/bin/env python3
"""Extract sanitized Codex token-cache telemetry from a session JSONL file.

The output is safe to replay with:

  go run ./cmd/fak vcache prove-telemetry --file OUT.jsonl --json

Only token counter rows are copied. Prompt text, tool calls, tool output, file
diffs, and model responses are intentionally dropped.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


SCHEMA = "fak-vcache-codex-session-extract/1"


def as_nonnegative_int(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return max(n, 0)


def usage_pair(usage: Any) -> tuple[int, int] | None:
    if not isinstance(usage, dict):
        return None

    total = as_nonnegative_int(usage.get("input_tokens") or usage.get("prompt_tokens"))
    cached = as_nonnegative_int(usage.get("cached_input_tokens") or usage.get("cached_tokens"))

    input_details = usage.get("input_tokens_details")
    if cached == 0 and isinstance(input_details, dict):
        cached = as_nonnegative_int(input_details.get("cached_tokens"))
    prompt_details = usage.get("prompt_tokens_details")
    if cached == 0 and isinstance(prompt_details, dict):
        cached = as_nonnegative_int(prompt_details.get("cached_tokens"))

    if total == 0 and cached == 0:
        return None
    return total, min(cached, total) if total else cached


def sanitize_row(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None

    if row.get("type") == "event_msg":
        payload = row.get("payload")
        if isinstance(payload, dict) and payload.get("type") == "token_count":
            info = payload.get("info")
            last_usage = info.get("last_token_usage") if isinstance(info, dict) else None
            pair = usage_pair(last_usage)
            if pair is not None:
                total, cached = pair
                return {
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "last_token_usage": {
                                "input_tokens": total,
                                "cached_input_tokens": cached,
                            }
                        },
                    },
                }

    if row.get("type") == "turn.completed":
        pair = usage_pair(row.get("usage"))
        if pair is not None:
            total, cached = pair
            return {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": total,
                    "cached_input_tokens": cached,
                },
            }

    return None


def codex_home(env: dict[str, str]) -> Path:
    configured = env.get("CODEX_HOME")
    if configured:
        return Path(configured)
    return Path.home() / ".codex"


def find_session(home: Path, thread_id: str) -> Path:
    thread_id = thread_id.strip()
    if not thread_id:
        raise FileNotFoundError("CODEX_THREAD_ID is empty")
    sessions = home / "sessions"
    if not sessions.exists():
        raise FileNotFoundError(f"Codex sessions directory not found: {sessions}")
    candidates = sorted(
        sessions.rglob(f"*{thread_id}*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no Codex session JSONL matched thread id {thread_id!r} under {sessions}")
    return candidates[0]


def extract_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            sanitized = sanitize_row(raw)
            if sanitized is not None:
                rows.append(sanitized)
    return rows


def write_rows(path: str, rows: list[dict[str, Any]], stdout=sys.stdout) -> None:
    if path == "-":
        for row in rows:
            print(json.dumps(row, separators=(",", ":"), sort_keys=True), file=stdout)
        return

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", help="raw Codex session or codex exec --json JSONL path")
    p.add_argument("--thread-id", help="Codex thread id; defaults to CODEX_THREAD_ID")
    p.add_argument("--codex-home", help="Codex home; defaults to CODEX_HOME or ~/.codex")
    p.add_argument("--out", required=True, help="sanitized JSONL output path, or '-' for stdout")
    return p.parse_args(argv)


def run(argv: list[str], *, env=os.environ, stderr=sys.stderr, stdout=sys.stdout) -> int:
    args = parse_args(argv)

    try:
        if args.session:
            session = Path(args.session)
        else:
            thread_id = args.thread_id or env.get("CODEX_THREAD_ID", "")
            home = Path(args.codex_home) if args.codex_home else codex_home(env)
            session = find_session(home, thread_id)

        if not session.exists():
            print(f"vcache_codex_session_extract: session not found: {session}", file=stderr)
            return 2

        rows = extract_rows(session)
        if not rows:
            print(f"vcache_codex_session_extract: no token usage rows found in {session}", file=stderr)
            return 1

        write_rows(args.out, rows, stdout)
    except (OSError, ValueError) as exc:
        print(f"vcache_codex_session_extract: {exc}", file=stderr)
        return 2

    target = "stdout" if args.out == "-" else args.out
    print(f"wrote {len(rows)} sanitized token rows from {session} to {target}", file=stderr)
    return 0


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
