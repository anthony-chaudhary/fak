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


def limited_counter(counter: Counter[str], limit: int = 12) -> dict[str, int]:
    return {key: int(value) for key, value in counter.most_common(limit) if value}


TAG_REMEDIATION = {
    "HOOK_OR_API_WALL_FEEDBACK": "clear_hook_or_api_wall_feedback",
    "HOST_PERMISSION_INTERRUPT": "reduce_permission_interruptions_or_scope_policy",
    "DENY_OR_BLOCKED_FEEDBACK": "align_policy_with_real_tool_shapes",
    "TOOL_ERROR_RECOVERY": "fix_tool_contract_or_error_recovery_loop",
    "SHELL_HEAVY_SESSION": "replace_shell_with_path_visible_tools",
    "LARGE_RESULT": "cap_or_summarize_large_outputs",
}


def remediation_for_tags(tags: list[str]) -> list[str]:
    out = []
    for tag in tags:
        bucket = TAG_REMEDIATION.get(str(tag))
        if bucket:
            out.append(bucket)
    return out


def safe_block_type(value: Any) -> str:
    label = str(value or "unknown")
    if label == "tool_result":
        return "result_block"
    return label


def root_label(path: Path) -> str:
    parts = path.parts
    if len(parts) >= 3 and parts[-2] == "projects":
        return f"{account_label(parts[-3])}/{parts[-1]}"
    return path.name


def account_label(name: str) -> str:
    if name == ".claude":
        return name
    if name.startswith(".claude"):
        return ".claude-" + digest_obj(name)[:8]
    return digest_obj(name)[:8]


def default_account_roots(namespace: str = DEFAULT_NS) -> list[Path]:
    configured = os.environ.get("CLAUDE_CONFIG_DIR")
    if configured:
        root = Path(configured) / "projects" / namespace
        return [root] if root.is_dir() else []
    user_home = Path(os.environ.get("USERPROFILE", os.path.expanduser("~")))
    roots = []
    for account_dir in sorted(user_home.glob(".claude*")):
        project = account_dir / "projects" / namespace
        if project.is_dir():
            roots.append(project)
    return roots


def discover(root: str | Path, *, since_days: float | None = None, max_sessions: int = 10) -> list[Path]:
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


def discover_many(roots: list[Path], *, since_days: float | None = None, max_sessions: int = 10) -> list[Path]:
    candidates = []
    for root in roots:
        for path in discover(root, since_days=since_days, max_sessions=max_sessions):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            candidates.append((mtime, path))
    candidates.sort(reverse=True)
    return [path for _, path in candidates[:max_sessions]]


def summarize_transcript_shape(path: Path) -> dict[str, Any]:
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
    max_result_chars = 0
    markers = {
        "hook_or_api_wall": ("hook", "api-wall", "api wall", "stopfailure", "stop failure"),
        "permission": ("permission",),
        "deny_or_blocked": ("denied", "deny", "blocked"),
        "error_recovery": ("error", "exception", "traceback"),
    }
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {"status": "UNREADABLE", "error_type": type(exc).__name__}
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
                block_type = safe_block_type(block.get("type"))
                block_types[block_type] += 1
                if block.get("type") == "tool_use":
                    tools[str(block.get("name") or "unknown")] += 1
                if block.get("type") == "tool_result":
                    result = block.get("content")
                    if isinstance(result, str):
                        max_result_chars = max(max_result_chars, len(result))
                        result_lower = result.lower()
                        if "permission" in result_lower:
                            result_shapes["permission_text"] += 1
                        elif "error" in result_lower or "exception" in result_lower or "traceback" in result_lower:
                            result_shapes["error_text"] += 1
                        else:
                            result_shapes["other_text"] += 1
                    elif isinstance(result, list):
                        result_shapes["list"] += 1
                    elif result is None:
                        result_shapes["null"] += 1
                    else:
                        result_shapes[type(result).__name__] += 1
        tool_result = row.get("toolUseResult")
        if isinstance(tool_result, dict):
            result_shapes["result_record_dict"] += 1
            if tool_result.get("isError"):
                result_shapes["result_record_error"] += 1
        elif isinstance(tool_result, str):
            result_shapes["result_record_str"] += 1
            max_result_chars = max(max_result_chars, len(tool_result))

    tool_total = sum(tools.values())
    evidence_tags = []
    if marker_lines.get("hook_or_api_wall"):
        evidence_tags.append("HOOK_OR_API_WALL_FEEDBACK")
    if marker_lines.get("permission") or result_shapes.get("permission_text"):
        evidence_tags.append("HOST_PERMISSION_INTERRUPT")
    if marker_lines.get("deny_or_blocked"):
        evidence_tags.append("DENY_OR_BLOCKED_FEEDBACK")
    if marker_lines.get("error_recovery") or result_shapes.get("error_text") or result_shapes.get("result_record_error"):
        evidence_tags.append("TOOL_ERROR_RECOVERY")
    if tool_total and tools.get("Bash", 0) / tool_total >= 0.5:
        evidence_tags.append("SHELL_HEAVY_SESSION")
    if max_result_chars >= 20000:
        evidence_tags.append("LARGE_RESULT")

    return {
        "status": "SUMMARIZED",
        "line_count": line_count,
        "malformed_lines": malformed,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "row_type_counts": limited_counter(row_types),
        "role_counts": limited_counter(roles),
        "block_type_counts": limited_counter(block_types),
        "tool_counts": limited_counter(tools),
        "marker_line_counts": limited_counter(marker_lines),
        "result_shape_counts": limited_counter(result_shapes),
        "max_result_chars": max_result_chars,
        "evidence_tags": evidence_tags,
    }


def aggregate_transcript_shapes(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    marker_lines: Counter[str] = Counter()
    evidence_tags: Counter[str] = Counter()
    remediation: Counter[str] = Counter()
    result_shapes: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    max_result_chars = 0
    summarized = 0
    for summary in summaries:
        if summary.get("status") != "SUMMARIZED":
            continue
        summarized += 1
        marker_lines.update({str(k): int(v) for k, v in (summary.get("marker_line_counts") or {}).items()})
        result_shapes.update({str(k): int(v) for k, v in (summary.get("result_shape_counts") or {}).items()})
        tools.update({str(k): int(v) for k, v in (summary.get("tool_counts") or {}).items()})
        for tag in summary.get("evidence_tags") or []:
            evidence_tags[str(tag)] += 1
        for bucket in remediation_for_tags([str(tag) for tag in summary.get("evidence_tags") or []]):
            remediation[bucket] += 1
        max_result_chars = max(max_result_chars, int(summary.get("max_result_chars") or 0))
    return {
        "summarized_sessions": summarized,
        "evidence_tag_counts": limited_counter(evidence_tags),
        "remediation_session_counts": limited_counter(remediation),
        "marker_line_counts": limited_counter(marker_lines),
        "result_shape_counts": limited_counter(result_shapes),
        "tool_counts": limited_counter(tools),
        "max_result_chars": max_result_chars,
    }


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
    claim = str(raw.get("claim") or "")
    return {
        "call_digest": call["call_digest"],
        "tool": call["tool"],
        "verdict": str(raw.get("verdict") or ""),
        "reason": str(raw.get("reason") or ""),
        "by": str(raw.get("by") or ""),
        "claim_digest": digest_obj(claim) if claim else "",
        "claim_bytes": len(claim.encode("utf-8")),
        "args_digest": str(raw.get("args_digest") or ""),
        "args_bytes": int(raw.get("args_bytes") or len(args_json.encode("utf-8"))),
    }


def collect(
    *,
    root: str = DEFAULT_ROOT,
    all_accounts: bool = False,
    namespace: str = DEFAULT_NS,
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
    roots = default_account_roots(namespace) if all_accounts else [Path(root)]
    paths = discover_many(roots, since_days=since_days, max_sessions=max_sessions)
    all_calls: list[dict[str, Any]] = []
    per_session = []
    transcript_summaries: list[dict[str, Any]] = []
    top_friction_sessions: list[dict[str, Any]] = []
    for p in paths:
        calls = iter_tool_uses(p)
        shape = summarize_transcript_shape(p)
        transcript_summaries.append(shape)
        tags = [str(tag) for tag in shape.get("evidence_tags") or []]
        marker_total = sum(int(v) for v in (shape.get("marker_line_counts") or {}).values())
        if tags or marker_total:
            top_friction_sessions.append({
                "session_digest": digest_obj(p.stem),
                "root_label": root_label(p.parent),
                "mtime": dt.datetime.fromtimestamp(p.stat().st_mtime, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                "tool_calls": len(calls),
                "marker_lines": marker_total,
                "max_result_chars": int(shape.get("max_result_chars") or 0),
                "evidence_tags": tags,
                "remediation": remediation_for_tags(tags),
                "tool_counts": shape.get("tool_counts") or {},
                "marker_line_counts": shape.get("marker_line_counts") or {},
                "result_shape_counts": shape.get("result_shape_counts") or {},
            })
        all_calls.extend(calls)
        per_session.append({
            "session_digest": digest_obj(p.stem),
            "root_label": root_label(p.parent),
            "mtime": dt.datetime.fromtimestamp(p.stat().st_mtime, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "tool_calls": len(calls),
            "evidence_tags": tags,
        })
    top_friction_sessions.sort(
        key=lambda row: (
            len(row.get("evidence_tags") or []),
            int(row.get("marker_lines") or 0),
            int(row.get("max_result_chars") or 0),
            str(row.get("mtime") or ""),
        ),
        reverse=True,
    )
    transcript_shape = aggregate_transcript_shapes(transcript_summaries)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": now_utc(),
        "status": "PASS",
        "root": root if not all_accounts else "<all-claude-accounts>",
        "root_labels": [root_label(path) for path in roots],
        "all_accounts": all_accounts,
        "namespace": namespace,
        "policy": policy,
        "sessions_discovered": len(paths),
        "sessions_audited": sum(1 for s in per_session if s["tool_calls"] > 0),
        "tool_calls_seen": len(all_calls),
        "max_sessions": max_sessions,
        "max_calls": max_calls,
        "since_days": since_days,
        "transcript_shape": transcript_shape,
        "top_friction_sessions": top_friction_sessions[:12],
        "privacy": {
            "copied_fields": ["tool names", "verdict metadata", "counts", "hash digests", "root labels", "derived transcript-shape tags"],
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
    lines.extend(["", "## Transcript Friction Signals", ""])
    shape = payload.get("transcript_shape") if isinstance(payload.get("transcript_shape"), dict) else {}
    if shape:
        lines.extend(
            [
                f"- summarized_sessions: `{shape.get('summarized_sessions')}`",
                f"- evidence_tag_counts: `{json.dumps(shape.get('evidence_tag_counts') or {}, sort_keys=True)}`",
                f"- remediation_session_counts: `{json.dumps(shape.get('remediation_session_counts') or {}, sort_keys=True)}`",
                f"- marker_line_counts: `{json.dumps(shape.get('marker_line_counts') or {}, sort_keys=True)}`",
                f"- result_shape_counts: `{json.dumps(shape.get('result_shape_counts') or {}, sort_keys=True)}`",
                f"- max_result_chars: `{shape.get('max_result_chars')}`",
            ]
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Top Friction Sessions", ""])
    friction = payload.get("top_friction_sessions") if isinstance(payload.get("top_friction_sessions"), list) else []
    if friction:
        for row in friction[:8]:
            lines.append(
                f"- `{row.get('session_digest')}` root=`{row.get('root_label')}` "
                f"tool_calls=`{row.get('tool_calls')}` marker_lines=`{row.get('marker_lines')}` "
                f"max_result_chars=`{row.get('max_result_chars')}` tags=`{', '.join(row.get('evidence_tags') or []) or 'none'}` "
                f"remediation=`{', '.join(row.get('remediation') or []) or 'none'}`"
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
    p.add_argument("--all-accounts", action="store_true", help="audit every local .claude* project root for the namespace")
    p.add_argument("--namespace", default=DEFAULT_NS, help="Claude project namespace to use with --all-accounts")
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
        all_accounts=args.all_accounts,
        namespace=args.namespace,
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
