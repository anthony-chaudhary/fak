#!/usr/bin/env python3
"""Replay historical Claude Code tool proposals through fak preflight.

This is the historical companion to the Claude live pilot. It scans local Claude
Code transcript JSONL files for assistant `tool_use` blocks, replays each unique
tool+argument shape through `fak preflight --policy ... --json`, and writes only
aggregate verdict metadata. It never copies prompts, tool arguments, or tool
results into the report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable


SCHEMA = "fak-claude-historical-guard-audit/1"
DEFAULT_POLICY = "examples/dogfood-claude-policy.json"
DEFAULT_NS = "C--work-fak"
# Resolve the config home the way the fleet relocates it: CLAUDE_CONFIG_DIR wins
# (each agent runs under its own ~/.claude-<account>), else the vanilla ~/.claude.
# Hardcoding ~/.claude here would audit the WRONG (stale or foreign) store.
_CONFIG_DIR = os.environ.get("CLAUDE_CONFIG_DIR")
_CLAUDE_HOME = _CONFIG_DIR if _CONFIG_DIR else os.path.join(os.environ.get("USERPROFILE", os.path.expanduser("~")), ".claude")
DEFAULT_ROOT = os.path.join(_CLAUDE_HOME, "projects", DEFAULT_NS)


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def digest_obj(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def discover(root: str, *, since_days: float | None = None, max_sessions: int = 10) -> list[Path]:
    base = Path(root)
    if not base.is_dir():
        return []
    cutoff = None
    if since_days is not None:
        cutoff = dt.datetime.now().timestamp() - since_days * 86400
    paths = []
    for raw in glob.glob(str(base / "*.jsonl")):
        p = Path(raw)
        try:
            st = p.stat()
        except OSError:
            continue
        if cutoff is not None and st.st_mtime < cutoff:
            continue
        paths.append((st.st_mtime, p))
    paths.sort(reverse=True)
    return [p for _, p in paths[:max_sessions]]


def iter_tool_uses(path: Path) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    seen_messages: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return calls
    for line in lines:
        if '"tool_use"' not in line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") != "assistant":
            continue
        msg = row.get("message") if isinstance(row.get("message"), dict) else {}
        msg_id = str(msg.get("id") or "")
        if msg_id:
            if msg_id in seen_messages:
                continue
            seen_messages.add(msg_id)
        content = msg.get("content") if isinstance(msg.get("content"), list) else []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = str(block.get("name") or "")
            if not name:
                continue
            args = block.get("input")
            if not isinstance(args, dict):
                args = {"value": args}
            calls.append({
                "session": path.stem,
                "tool": name,
                "arguments": args,
                "call_digest": digest_obj({"tool": name, "arguments": args}),
            })
    return calls


def find_fak(explicit: str | None = None, root: Path | None = None) -> str | None:
    if explicit:
        return explicit
    repo = root or Path(__file__).resolve().parents[1]
    names = ["fak.exe" if sys.platform == "win32" else "fak", "fak", "fak.exe"]
    for name in names:
        p = repo / name
        if p.is_file():
            return str(p)
    for name in names:
        p = repo / "tools" / ".bin" / name
        if p.is_file():
            return str(p)
    return shutil.which("fak") or shutil.which("fak.exe")


def parse_json_prefix(text: str) -> dict[str, Any]:
    stripped = text.lstrip()
    data, _ = json.JSONDecoder().raw_decode(stripped)
    if not isinstance(data, dict):
        raise ValueError("preflight output was not a JSON object")
    return data


def default_runner(argv: list[str]) -> dict[str, Any]:
    proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or f"exit {proc.returncode}")[:300])
    return parse_json_prefix(proc.stdout)


Runner = Callable[[list[str]], dict[str, Any]]


def replay_call(call: dict[str, Any], *, fak_bin: str, policy: str, runner: Runner = default_runner) -> dict[str, Any]:
    args_json = json.dumps(call["arguments"], sort_keys=True, separators=(",", ":"), default=str)
    raw = runner([fak_bin, "preflight", "--policy", policy, "--tool", call["tool"], "--args", args_json, "--json"])
    return {
        "call_digest": call["call_digest"],
        "tool": call["tool"],
        "verdict": str(raw.get("verdict") or ""),
        "reason": str(raw.get("reason") or ""),
        "by": str(raw.get("by") or ""),
        "claim": str(raw.get("claim") or ""),
        "args_digest": str(raw.get("args_digest") or ""),
        "args_bytes": int(raw.get("args_bytes") or len(args_json.encode("utf-8"))),
    }


def collect(
    *,
    root: str = DEFAULT_ROOT,
    policy: str = DEFAULT_POLICY,
    fak: str | None = None,
    since_days: float | None = 7,
    max_sessions: int = 10,
    max_calls: int = 500,
    runner: Runner | None = None,
) -> dict[str, Any]:
    if runner is None:
        runner = default_runner
    repo = Path(__file__).resolve().parents[1]
    fak_bin = find_fak(fak, repo)
    paths = discover(root, since_days=since_days, max_sessions=max_sessions)
    all_calls: list[dict[str, Any]] = []
    per_session = []
    for p in paths:
        calls = iter_tool_uses(p)
        all_calls.extend(calls)
        per_session.append({
            "session_digest": digest_obj(p.stem),
            "mtime": dt.datetime.fromtimestamp(p.stat().st_mtime, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "tool_calls": len(calls),
        })

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": now_utc(),
        "status": "PASS",
        "root": root,
        "policy": policy,
        "sessions_discovered": len(paths),
        "sessions_audited": sum(1 for s in per_session if s["tool_calls"] > 0),
        "tool_calls_seen": len(all_calls),
        "max_sessions": max_sessions,
        "max_calls": max_calls,
        "since_days": since_days,
        "privacy": {
            "copied_fields": ["tool names", "verdict metadata", "counts", "hash digests"],
            "dropped": ["prompts", "tool arguments", "tool results", "raw transcript text"],
        },
    }
    if not paths:
        payload.update({"status": "NO_CORPUS", "blockers": ["no Claude Code transcript JSONL files found"]})
        return payload
    if not fak_bin:
        payload.update({"status": "BLOCKED_ENV", "blockers": ["fak binary not found"]})
        return payload

    unique: dict[str, dict[str, Any]] = {}
    for call in all_calls:
        unique.setdefault(call["call_digest"], call)
    unique_calls = list(unique.values())
    truncated = len(unique_calls) > max_calls
    selected = unique_calls[:max_calls]

    verdicts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    tools: Counter[str] = Counter(call["tool"] for call in all_calls)
    replay_errors = []
    samples = []
    for call in selected:
        try:
            row = replay_call(call, fak_bin=fak_bin, policy=policy, runner=runner)
        except Exception as exc:  # noqa: BLE001 - report typed replay failures without args
            replay_errors.append({
                "call_digest": call["call_digest"],
                "tool": call["tool"],
                "error_type": type(exc).__name__,
                "error": str(exc)[:200],
            })
            continue
        verdicts[row["verdict"]] += 1
        if row["reason"]:
            reasons[row["reason"]] += 1
        if row["verdict"] != "ALLOW" and len(samples) < 12:
            samples.append(row)

    payload.update({
        "status": "PASS" if not replay_errors else "WARN",
        "replay_engine": {
            "kind": "fak_preflight",
            "binary": fak_bin,
            "policy": policy,
        },
        "sessions": per_session,
        "unique_tool_calls": len(unique_calls),
        "unique_tool_calls_replayed": len(selected) - len(replay_errors),
        "truncated": truncated,
        "verdict_counts": dict(sorted(verdicts.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "tool_counts": dict(tools.most_common()),
        "non_allow_samples": samples,
        "replay_errors": replay_errors,
    })
    if len(all_calls) == 0:
        payload["status"] = "NO_TOOL_CALLS"
    return payload


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Claude Code historical guard replay",
        "",
        f"- generated: `{payload.get('created_at')}`",
        f"- status: **`{payload.get('status')}`**",
        f"- sessions_discovered: `{payload.get('sessions_discovered')}`",
        f"- sessions_audited: `{payload.get('sessions_audited')}`",
        f"- tool_calls_seen: `{payload.get('tool_calls_seen')}`",
        f"- unique_tool_calls_replayed: `{payload.get('unique_tool_calls_replayed')}`",
        f"- truncated: `{payload.get('truncated')}`",
        "",
        "## Verdict Counts",
        "",
    ]
    verdict_counts = payload.get("verdict_counts") if isinstance(payload.get("verdict_counts"), dict) else {}
    if verdict_counts:
        lines.extend(f"- {k}: `{v}`" for k, v in verdict_counts.items())
    else:
        lines.append("- none")
    lines.extend(["", "## Reason Counts", ""])
    reason_counts = payload.get("reason_counts") if isinstance(payload.get("reason_counts"), dict) else {}
    if reason_counts:
        lines.extend(f"- {k}: `{v}`" for k, v in reason_counts.items())
    else:
        lines.append("- none")
    lines.extend(["", "## Non-Allow Samples", ""])
    samples = payload.get("non_allow_samples") if isinstance(payload.get("non_allow_samples"), list) else []
    if samples:
        for row in samples[:12]:
            lines.append(
                f"- `{row.get('call_digest')}` {row.get('tool')} -> "
                f"`{row.get('verdict')}` / `{row.get('reason')}`"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Privacy",
            "",
            "This replay records only tool names, verdict metadata, aggregate counts, and hash digests. It never writes prompts, tool arguments, tool results, or raw transcript text.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--policy", default=DEFAULT_POLICY)
    p.add_argument("--fak", default=None)
    p.add_argument("--since-days", type=float, default=7)
    p.add_argument("--max-sessions", type=int, default=10)
    p.add_argument("--max-calls", type=int, default=500)
    p.add_argument("--out", type=Path)
    p.add_argument("--markdown", type=Path)
    p.add_argument("--json", action="store_true")
    p.add_argument("--fail-on-not-pass", action="store_true")
    args = p.parse_args(argv)
    payload = collect(
        root=args.root,
        policy=args.policy,
        fak=args.fak,
        since_days=args.since_days,
        max_sessions=args.max_sessions,
        max_calls=args.max_calls,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_md(payload), encoding="utf-8")
    if args.json or not (args.out or args.markdown):
        print(json.dumps(payload, indent=2, sort_keys=True))
    if args.fail_on_not_pass and payload.get("status") != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
