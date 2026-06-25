#!/usr/bin/env python3
"""Create a privacy-preserving fak dogfood witness for the current Codex thread.

The witness is deliberately built from artifacts this script did not author:

* Codex session JSONL token counters, sanitized down to usage rows only.
* `fak preflight` allow/deny verdicts for the coding-agent capability floor.
* `fak vcache prove-telemetry --json` over the sanitized Codex usage rows.

No prompt text, tool calls, tool output, diffs, or model responses are copied.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
from typing import Any, Callable


SCHEMA = "fak-codex-dogfood-witness/1"
PREFLIGHT_RE = re.compile(r"\bverdict=(?P<verdict>[A-Z_]+)\s+reason=(?P<reason>[A-Z_]+)\s+by=(?P<by>[A-Za-z0-9_-]+)")
MCP_EXPECT_TOOLS = {"fak_adjudicate", "fak_syscall", "fak_admit"}
DOS_UNKNOWN_TREE_WARN_THRESHOLD = 0.02
CODEX_MCP_PROBE_PROMPT = (
    "Use the MCP server named fak to call fak_adjudicate for tool git_push with empty "
    "arguments, and fak_adjudicate for tool git_status with empty arguments. Do not "
    "inspect files and do not edit anything. Final answer only: "
    "git_push=<verdict>/<reason>; git_status=<verdict>/<reason>."
)


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


Runner = Callable[[list[str], Path], CommandResult]
MCPProbe = Callable[[argparse.Namespace], dict[str, Any]]


def load_codex_extract():
    script = Path(__file__).resolve().with_name("vcache_codex_session_extract.py")
    spec = importlib.util.spec_from_file_location("vcache_codex_session_extract", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_codex_recent_audit():
    script = Path(__file__).resolve().with_name("codex_dos_recent_audit.py")
    spec = importlib.util.spec_from_file_location("codex_dos_recent_audit", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_subprocess(argv: list[str], cwd: Path) -> CommandResult:
    proc = subprocess.run(
        argv,
        cwd=str(cwd),
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return CommandResult(proc.returncode, proc.stdout, proc.stderr)


def run_subprocess_input(argv: list[str], cwd: Path, input_text: str) -> CommandResult:
    proc = subprocess.run(
        argv,
        cwd=str(cwd),
        text=True,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return CommandResult(proc.returncode, proc.stdout, proc.stderr)


def default_fak_command(root: Path) -> list[str]:
    exe = "fak.exe" if sys.platform == "win32" else "fak"
    local = root / exe
    if local.exists():
        return [str(local)]
    on_path = shutil.which("fak")
    if on_path:
        return [on_path]
    return ["go", "run", "./cmd/fak"]


def fak_argv(args: argparse.Namespace, *tail: str) -> list[str]:
    if args.fak_bin:
        return [args.fak_bin, *tail]
    return [*default_fak_command(args.repo_root), *tail]


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def compact_output(text: str, limit: int = 2000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "...<truncated>"


def parse_timestamp(value: Any) -> datetime | None:
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


def iso_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


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


def sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {k: counter[k] for k in sorted(counter)}


def sorted_int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = int(raw)
        except (TypeError, ValueError):
            continue
    return {k: out[k] for k in sorted(out)}


def stream_audit(args: argparse.Namespace, thread_id: str) -> dict[str, Any]:
    stream_path = args.repo_root / ".dos" / "streams" / f"{thread_id}.jsonl"
    audit: dict[str, Any] = {
        "status": "MISSING",
        "stream_path": relpath(stream_path, args.repo_root),
        "steps": 0,
        "tool_counts": {},
        "first_ts": None,
        "last_ts": None,
        "malformed_rows": 0,
    }
    if not stream_path.exists():
        return audit

    tools: Counter[str] = Counter()
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    malformed = 0
    steps = 0
    try:
        with stream_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(row, dict):
                    malformed += 1
                    continue
                ts = parse_timestamp(row.get("ts"))
                if ts is not None:
                    first_ts = ts if first_ts is None or ts < first_ts else first_ts
                    last_ts = ts if last_ts is None or ts > last_ts else last_ts
                if row.get("op") == "STEP":
                    steps += 1
                    tools[str(row.get("tool_name") or "unknown")] += 1
    except OSError as exc:
        audit["status"] = "ERROR"
        audit["error"] = str(exc)
        return audit

    audit.update(
        {
            "status": "FOUND",
            "steps": steps,
            "tool_counts": sorted_counter(tools),
            "first_ts": iso_timestamp(first_ts),
            "last_ts": iso_timestamp(last_ts),
            "malformed_rows": malformed,
        }
    )
    return audit


def stop_failure_audit(args: argparse.Namespace, thread_id: str) -> dict[str, Any]:
    path = args.repo_root / ".dos" / "stop-failures" / f"{thread_id}.json"
    out: dict[str, Any] = {"status": "MISSING", "path": relpath(path, args.repo_root)}
    if not path.exists():
        return out
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "ERROR", "path": relpath(path, args.repo_root), "error": str(exc)}
    if not isinstance(raw, dict):
        return {"status": "ERROR", "path": relpath(path, args.repo_root), "error": "not a JSON object"}
    return {
        "status": "FOUND",
        "path": relpath(path, args.repo_root),
        "total": int(raw.get("total") or 0),
        "consecutive": int(raw.get("consecutive") or 0),
    }


def observation_window(args: argparse.Namespace, start: datetime | None, end: datetime | None) -> dict[str, Any]:
    obs_path = args.repo_root / ".dos" / "metrics" / "observations.jsonl"
    if not obs_path.exists():
        return {"status": "MISSING", "path": relpath(obs_path, args.repo_root)}
    if start is None or end is None:
        return {
            "status": "UNKNOWN",
            "path": relpath(obs_path, args.repo_root),
            "reason": "session stream has no timestamp window",
        }

    total = 0
    malformed = 0
    verbs: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    rungs: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    dialects: Counter[str] = Counter()
    latencies: dict[str, list[float]] = {}
    pretool_total = 0
    unknown_tree_warnings = 0
    warning_total = 0
    delegate_total = 0
    stop_blocks = 0

    try:
        with obs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(row, dict):
                    malformed += 1
                    continue
                ts = parse_timestamp(row.get("ts"))
                if ts is None or ts < start or ts > end:
                    continue

                total += 1
                verb = str(row.get("verb") or "unknown")
                outcome = str(row.get("outcome") or "unknown")
                rung = str(row.get("rung") or "none")
                reason = str(row.get("reason") or "none")
                dialect = str(row.get("dialect") or "unknown")
                verbs[verb] += 1
                outcomes[outcome] += 1
                rungs[rung] += 1
                reasons[reason] += 1
                dialects[dialect] += 1
                if isinstance(row.get("latency_ms"), (int, float)):
                    latencies.setdefault(verb, []).append(float(row["latency_ms"]))
                if verb == "pretool":
                    pretool_total += 1
                if outcome == "warn":
                    warning_total += 1
                if outcome == "delegate":
                    delegate_total += 1
                if outcome == "warn" and rung == "admission" and row.get("tree_known") is False:
                    unknown_tree_warnings += 1
                if verb == "stop" and outcome in {"block", "deny"}:
                    stop_blocks += 1
    except OSError as exc:
        return {"status": "ERROR", "path": relpath(obs_path, args.repo_root), "error": str(exc)}

    rate = (unknown_tree_warnings / pretool_total) if pretool_total else None
    status = "FOUND" if total else "UNKNOWN"
    if total and (
        (rate is not None and rate > DOS_UNKNOWN_TREE_WARN_THRESHOLD)
        or delegate_total
        or stop_blocks
    ):
        status = "WARN"

    return {
        "status": status,
        "path": relpath(obs_path, args.repo_root),
        "scope": "timestamp-window; concurrent sessions in the same window may contribute observations",
        "window_start": iso_timestamp(start),
        "window_end": iso_timestamp(end),
        "observations": total,
        "malformed_rows": malformed,
        "pretool_calls": pretool_total,
        "verb_counts": sorted_counter(verbs),
        "outcome_counts": sorted_counter(outcomes),
        "rung_counts": sorted_counter(rungs),
        "reason_counts": sorted_counter(reasons),
        "dialect_counts": sorted_counter(dialects),
        "warning_count": warning_total,
        "delegate_count": delegate_total,
        "stop_blocks": stop_blocks,
        "unknown_tree_admission_warnings": unknown_tree_warnings,
        "unknown_tree_warning_rate": round(rate, 6) if rate is not None else None,
        "unknown_tree_warning_threshold": DOS_UNKNOWN_TREE_WARN_THRESHOLD,
        "latency_by_verb": {k: latency_summary(v) for k, v in sorted(latencies.items())},
    }


def dos_session_audit(args: argparse.Namespace, thread_id: str) -> dict[str, Any]:
    stream = stream_audit(args, thread_id)
    start = parse_timestamp(stream.get("first_ts"))
    end = parse_timestamp(stream.get("last_ts"))
    observations = observation_window(args, start, end)
    stop_failures = stop_failure_audit(args, thread_id)

    recommendations: list[str] = []
    tool_counts = stream.get("tool_counts") if isinstance(stream.get("tool_counts"), dict) else {}
    stream_steps = int(stream.get("steps") or 0)
    if stream.get("status") != "FOUND":
        recommendations.append("run under DOS hooks long enough to create a per-session stream")
    if stream_steps and int(tool_counts.get("Bash") or 0) / stream_steps > 0.8:
        recommendations.append("prefer path-visible tool calls or narrower shell commands; this stream is Bash-dominated")
    if observations.get("status") in {"MISSING", "UNKNOWN", "ERROR"}:
        recommendations.append("collect a timestamped DOS observation window for this session")
    if (observations.get("unknown_tree_warning_rate") or 0) > DOS_UNKNOWN_TREE_WARN_THRESHOLD:
        recommendations.append("reduce opaque shell/editor calls or file upstream footprint-derivation debt")
    if observations.get("delegate_count"):
        recommendations.append("inspect native-hook delegate fallbacks; the fast path should own routine calls")
    if observations.get("stop_blocks") or (stop_failures.get("total") or 0):
        recommendations.append("review stop-hook failures before treating the session as closed")

    status = "PASS"
    if stream.get("status") != "FOUND" or observations.get("status") in {"MISSING", "UNKNOWN", "ERROR"}:
        status = "UNKNOWN"
    if observations.get("status") == "WARN" or (stop_failures.get("total") or 0):
        status = "WARN"

    return {
        "status": status,
        "thread_id": thread_id,
        "stream": stream,
        "observations": observations,
        "stop_failures": stop_failures,
        "recommendations": recommendations,
        "privacy": {
            "copied_fields": ["timestamps", "tool names", "counts", "latencies", "hash digests"],
            "dropped": ["tool arguments", "tool results", "prompts", "model text", "diffs"],
        },
    }


def dos_helped_audit(args: argparse.Namespace, runner: Runner, thread_id: str) -> dict[str, Any]:
    if not thread_id:
        return {"status": "UNKNOWN", "reason": "missing thread id"}
    result = runner(
        ["dos", "helped", "--session", thread_id, "--advisory", "--json"],
        args.repo_root,
    )
    if result.exit_code != 0:
        return {
            "status": "ERROR",
            "exit_code": result.exit_code,
            "stderr": compact_output(result.stderr),
        }
    try:
        decoded = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "status": "ERROR",
            "exit_code": result.exit_code,
            "error": str(exc),
            "stderr": compact_output(result.stderr),
        }
    if not isinstance(decoded, dict):
        return {
            "status": "ERROR",
            "exit_code": result.exit_code,
            "error": "dos helped did not return a JSON object",
            "stderr": compact_output(result.stderr),
        }

    return {
        "status": "FOUND",
        "exit_code": result.exit_code,
        "source": "dos helped --session <thread> --advisory --json",
        "total": decoded.get("total"),
        "advisory": decoded.get("advisory"),
        "warned": decoded.get("warned"),
        "blocked": decoded.get("blocked"),
        "withheld": decoded.get("withheld"),
        "deferred": decoded.get("deferred"),
        "since": decoded.get("since"),
        "latest": decoded.get("latest"),
        "by_rung": sorted_int_mapping(decoded.get("by_rung")),
        "by_reason": sorted_int_mapping(decoded.get("by_reason")),
        "by_reason_rung": decoded.get("by_reason_rung") if isinstance(decoded.get("by_reason_rung"), dict) else {},
        "by_tool": sorted_int_mapping(decoded.get("by_tool")),
        "by_advisory_tool": sorted_int_mapping(decoded.get("by_advisory_tool")),
        "by_refused_reason": sorted_int_mapping(decoded.get("by_refused_reason")),
        "privacy": {
            "copied_fields": ["counts", "tool names", "reason classes", "timestamps"],
            "dropped": ["examples", "targets", "tool arguments", "tool results", "prompts", "model text"],
        },
        "stderr": compact_output(result.stderr),
    }


def parse_preflight(result: CommandResult) -> dict[str, Any]:
    m = PREFLIGHT_RE.search(result.stdout)
    parsed = m.groupdict() if m else {}
    parsed["exit_code"] = result.exit_code
    parsed["stdout"] = compact_output(result.stdout)
    parsed["stderr"] = compact_output(result.stderr)
    return parsed


def run_preflight(args: argparse.Namespace, runner: Runner, tool: str) -> dict[str, Any]:
    result = runner(
        fak_argv(
            args,
            "preflight",
            "--policy",
            args.policy,
            "--tool",
            tool,
            "--args",
            "{}",
        ),
        args.repo_root,
    )
    return parse_preflight(result)


def run_vcache_proof(args: argparse.Namespace, runner: Runner, usage_path: Path) -> dict[str, Any]:
    result = runner(
        fak_argv(
            args,
            "vcache",
            "prove-telemetry",
            "--file",
            str(usage_path),
            "--read-mult",
            str(args.read_mult),
            "--json",
        ),
        args.repo_root,
    )
    proof: dict[str, Any] = {}
    if result.stdout.strip():
        try:
            decoded = json.loads(result.stdout)
            if isinstance(decoded, dict):
                proof = decoded
        except json.JSONDecodeError as exc:
            proof = {"parse_error": str(exc)}
    proof["exit_code"] = result.exit_code
    proof["stdout"] = compact_output(result.stdout)
    proof["stderr"] = compact_output(result.stderr)
    return proof


def verdict_from_mcp_tool_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return {}
    first = content[0]
    if not isinstance(first, dict):
        return {}
    text = first.get("text")
    if not isinstance(text, str) or not text.strip():
        return {}
    decoded = json.loads(text)
    verdict = decoded.get("verdict", {}) if isinstance(decoded, dict) else {}
    return verdict if isinstance(verdict, dict) else {}


def fold_codex_exec_mcp_usage(lines: list[str], source: str, deny_tool: str, allow_tool: str) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    thread_id = ""
    usage: dict[str, Any] = {}
    line_count = 0
    for line_no, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        line_count += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            return {"status": "ERROR", "source": source, "error": f"{source}:{line_no}: {exc}"}
        if not isinstance(row, dict):
            continue
        if row.get("type") == "thread.started":
            thread_id = str(row.get("thread_id") or "")
            continue
        if row.get("type") == "turn.completed" and isinstance(row.get("usage"), dict):
            raw_usage = row["usage"]
            usage = {
                "input_tokens": int(raw_usage.get("input_tokens") or 0),
                "cached_input_tokens": int(raw_usage.get("cached_input_tokens") or 0),
            }
            continue
        item = row.get("item")
        if not isinstance(item, dict) or item.get("type") != "mcp_tool_call":
            continue
        if item.get("server") != "fak" or item.get("tool") != "fak_adjudicate":
            continue
        if item.get("status") != "completed" or not isinstance(item.get("result"), dict):
            continue
        arguments = item.get("arguments")
        arguments_tool = arguments.get("tool") if isinstance(arguments, dict) else ""
        verdict = verdict_from_mcp_tool_result(item.get("result"))
        calls.append(
            {
                "server": "fak",
                "tool": "fak_adjudicate",
                "arguments_tool": arguments_tool,
                "status": item.get("status", ""),
                "verdict": verdict,
            }
        )

    deny = next((c for c in calls if c.get("arguments_tool") == deny_tool), {})
    allow = next((c for c in calls if c.get("arguments_tool") == allow_tool), {})
    deny_verdict = deny.get("verdict") if isinstance(deny.get("verdict"), dict) else {}
    allow_verdict = allow.get("verdict") if isinstance(allow.get("verdict"), dict) else {}
    ok = (
        deny_verdict.get("kind") == "DENY"
        and deny_verdict.get("reason") == "POLICY_BLOCK"
        and allow_verdict.get("kind") == "ALLOW"
    )
    return {
        "status": "PASS" if ok else "FAIL",
        "source": source,
        "thread_id": thread_id,
        "jsonl_rows": line_count,
        "mcp_tool_calls": calls,
        "turn_usage": usage,
        "privacy": {
            "copied": ["thread_id", "mcp_tool_call server/tool/arguments_tool/status/verdict", "turn usage counters"],
            "dropped": ["prompt", "agent_message", "tool_output_text", "reasoning_text"],
        },
    }


def read_codex_exec_mcp_usage(path: str, deny_tool: str, allow_tool: str) -> dict[str, Any]:
    if not path:
        return {"status": "SKIPPED", "reason": "no --codex-exec-jsonl"}

    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {"status": "ERROR", "source": str(source), "error": str(exc)}
    return fold_codex_exec_mcp_usage(lines, source.name, deny_tool, allow_tool)


def run_codex_exec_mcp_usage(args: argparse.Namespace, runner: Runner) -> dict[str, Any]:
    codex_name = args.codex_bin or "codex"
    codex = shutil.which(codex_name) or codex_name
    argv = [
        codex,
        "exec",
        "--json",
        "--ephemeral",
        "-c",
        'mcp_servers.fak.default_tools_approval_mode="approve"',
        "--cd",
        str(args.repo_root),
    ]
    if runner is run_subprocess:
        result = run_subprocess_input([*argv, "-"], args.repo_root, CODEX_MCP_PROBE_PROMPT + "\n")
    else:
        result = runner([*argv, CODEX_MCP_PROBE_PROMPT], args.repo_root)
    if result.exit_code != 0:
        return {
            "status": "ERROR",
            "source": "codex exec --json",
            "exit_code": result.exit_code,
            "stderr": compact_output(result.stderr),
        }
    proof = fold_codex_exec_mcp_usage(result.stdout.splitlines(), "codex exec --json", args.deny_tool, args.allow_tool)
    proof["exit_code"] = result.exit_code
    if result.stderr.strip():
        proof["stderr"] = compact_output(result.stderr)
    return proof


class MCPStdioProbe:
    def __init__(self, argv: list[str], cwd: Path, timeout: float) -> None:
        self.proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.timeout = timeout
        self._q: queue.Queue[bytes | None] = queue.Queue()
        self._err: list[bytes] = []
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    def _pump_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._q.put(line)
        self._q.put(None)

    def _pump_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self._err.append(line)

    def stderr_tail(self) -> str:
        return b"".join(self._err).decode("utf-8", "replace")[-1000:].strip()

    def send(self, obj: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def recv(self) -> dict[str, Any]:
        while True:
            line = self._q.get(timeout=self.timeout)
            if line is None:
                raise RuntimeError("fak MCP server closed stdout before replying")
            text = line.decode("utf-8", "replace").strip()
            if text:
                raw = json.loads(text)
                return raw if isinstance(raw, dict) else {}

    def request(self, rid: int, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        return self.recv()

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)

    def close(self) -> None:
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.terminate()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def mcp_tool_verdict(client: MCPStdioProbe, rid: int, tool: str) -> dict[str, Any]:
    response = client.request(
        rid,
        "tools/call",
        {"name": "fak_adjudicate", "arguments": {"tool": tool, "arguments": {}}},
    )
    if "error" in response:
        return {"jsonrpc_error": response["error"]}
    content = ((response.get("result") or {}).get("content") or [])
    text = content[0].get("text", "") if content else ""
    if not text:
        return {}
    decoded = json.loads(text)
    verdict = decoded.get("verdict", {}) if isinstance(decoded, dict) else {}
    return verdict if isinstance(verdict, dict) else {}


def run_mcp_stdio_probe(args: argparse.Namespace) -> dict[str, Any]:
    argv = fak_argv(args, "serve", "--stdio", "--policy", args.policy)
    report: dict[str, Any] = {
        "transport": "stdio",
        "command": " ".join(argv[:3]) + (" ..." if len(argv) > 3 else ""),
    }
    client = MCPStdioProbe(argv, args.repo_root, args.mcp_timeout)
    try:
        init = client.request(
            1,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "codex-dogfood-witness", "version": "1"},
            },
        )
        init_result = init.get("result") or {}
        server_info = init_result.get("serverInfo") or {}
        report["server_info"] = server_info
        report["protocol_version"] = init_result.get("protocolVersion")
        client.notify("notifications/initialized")

        tools_resp = client.request(2, "tools/list")
        tools = sorted(
            str(t.get("name"))
            for t in ((tools_resp.get("result") or {}).get("tools") or [])
            if isinstance(t, dict) and t.get("name")
        )
        report["tools"] = tools
        report["expected_tools_present"] = sorted(MCP_EXPECT_TOOLS & set(tools))
        report["missing_tools"] = sorted(MCP_EXPECT_TOOLS - set(tools))

        deny = mcp_tool_verdict(client, 3, args.deny_tool)
        allow = mcp_tool_verdict(client, 4, args.allow_tool)
        report["denies_publish"] = deny
        report["allows_status"] = allow

        ok = (
            server_info.get("name") == "fak-gateway"
            and not report["missing_tools"]
            and deny.get("kind") == "DENY"
            and deny.get("reason") == "POLICY_BLOCK"
            and allow.get("kind") == "ALLOW"
        )
        report["status"] = "PASS" if ok else "FAIL"
        report["stderr"] = client.stderr_tail()
        return report
    except (OSError, RuntimeError, json.JSONDecodeError, queue.Empty) as exc:
        report["status"] = "ERROR"
        report["error"] = str(exc)
        report["stderr"] = client.stderr_tail()
        return report
    finally:
        client.close()


def resolve_session(args: argparse.Namespace, extract_mod, env: dict[str, str]) -> tuple[Path, str]:
    if args.session:
        session = Path(args.session)
        thread_id = args.thread_id or env.get("CODEX_THREAD_ID", "")
        return session, thread_id

    thread_id = args.thread_id or env.get("CODEX_THREAD_ID", "")
    home = Path(args.codex_home) if args.codex_home else extract_mod.codex_home(env)
    return extract_mod.find_session(home, thread_id), thread_id


def default_paths(args: argparse.Namespace, thread_id: str) -> tuple[Path, Path]:
    safe_thread = re.sub(r"[^A-Za-z0-9_.-]+", "-", thread_id.strip()) or "unknown-thread"
    out = Path(args.out) if args.out else args.repo_root / "experiments" / "agent-live" / f"codex-dogfood-{safe_thread}.json"
    usage = Path(args.usage_out) if args.usage_out else out.with_suffix(".usage.jsonl")
    return out, usage


def sanitize_gate_report(path: Path, repo_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": relpath(path, repo_root)}
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        out.update({"status": "ERROR", "error": str(exc)})
        return out
    if not isinstance(decoded, dict):
        out.update({"status": "ERROR", "error": "gate report is not a JSON object"})
        return out

    preflight = decoded.get("preflight") if isinstance(decoded.get("preflight"), dict) else {}
    command = decoded.get("command") if isinstance(decoded.get("command"), list) else []
    command_redacted = bool(decoded.get("command_redacted"))
    out.update(
        {
            "status": decoded.get("status"),
            "tool": decoded.get("tool"),
            "policy": decoded.get("policy"),
            "executed": bool(decoded.get("executed")),
            "dry_run": bool(decoded.get("dry_run")),
            "expect_deny": bool(decoded.get("expect_deny")),
            "expect_reason": decoded.get("expect_reason"),
            "command_redacted": command_redacted,
            "command_label": decoded.get("command_label"),
            "command_digest": decoded.get("command_digest"),
            "command_executable": decoded.get("command_executable"),
            "command_argc": decoded.get("command_argc"),
            "command_exit_code": decoded.get("command_exit_code"),
            "preflight": {
                "verdict": preflight.get("verdict"),
                "reason": preflight.get("reason"),
                "by": preflight.get("by"),
                "exit_code": preflight.get("exit_code"),
            },
            "privacy": {
                "copied_fields": ["tool", "policy", "command label/digest", "exit code", "preflight verdict"],
                "dropped": ["raw command argv when redacted", "command stdout", "command stderr", "prompts", "tool outputs", "diffs", "model text"],
            },
        }
    )
    if not command_redacted:
        out["command"] = [str(part) for part in command]
    return out


def local_gate_reports(args: argparse.Namespace) -> dict[str, Any]:
    paths = [Path(p) for p in (args.gate_report or [])]
    if not paths:
        return {"status": "SKIPPED", "reason": "no --gate-report supplied", "reports": []}
    reports = [sanitize_gate_report(path, args.repo_root) for path in paths]
    pass_statuses = {"PASS", "ALLOW", "DENIED_EXPECTED"}
    passed = [r for r in reports if r.get("status") in pass_statuses]
    denied = [r for r in reports if r.get("status") in {"DENIED", "DENIED_EXPECTED"}]
    expected_denied = [r for r in reports if r.get("status") == "DENIED_EXPECTED"]
    failed = [r for r in reports if r.get("status") not in pass_statuses]
    return {
        "status": "PASS" if not failed else "FAIL",
        "reports": reports,
        "total": len(reports),
        "passed": len(passed),
        "denied": len(denied),
        "expected_denied": len(expected_denied),
        "failed": len(failed),
        "privacy": {
            "copied_fields": ["report path", "tool", "command label/digest", "exit code", "preflight verdict", "expected-deny flag"],
            "dropped": ["raw command argv when redacted", "command stdout", "command stderr", "prompts", "tool outputs", "diffs", "model text"],
        },
    }


def codex_hook_fast_path(
    args: argparse.Namespace,
    extract_mod,
    env: dict[str, str],
    thread_id: str,
    session: Path,
) -> dict[str, Any]:
    try:
        audit_mod = load_codex_recent_audit()
        home = Path(args.codex_home) if args.codex_home else extract_mod.codex_home(env)
        report = audit_mod.codex_hook_fast_path(home)
        if isinstance(report, dict) and report.get("repaired_at"):
            report["post_repair_observations"] = audit_mod.codex_observations_since(args.repo_root, report.get("repaired_at"))
            if thread_id:
                report["post_repair_command_shapes"] = audit_mod.codex_command_shape_audit(
                    home,
                    args.repo_root,
                    {thread_id: session},
                    report.get("repaired_at"),
                )
    except (OSError, RuntimeError, ValueError, AttributeError) as exc:
        return {"status": "UNKNOWN", "error": type(exc).__name__}
    if not isinstance(report, dict):
        return {"status": "UNKNOWN", "error": "codex hook fast-path report was not an object"}
    return {
        "status": report.get("status"),
        "reason": report.get("reason"),
        "manifest_count": report.get("manifest_count"),
        "malformed_manifests": report.get("malformed_manifests"),
        "manifests": report.get("manifests") if isinstance(report.get("manifests"), list) else [],
        "backups": report.get("backups") if isinstance(report.get("backups"), list) else [],
        "repaired_at": report.get("repaired_at"),
        "command_modes": report.get("command_modes") if isinstance(report.get("command_modes"), dict) else {},
        "codex_command_modes": report.get("codex_command_modes") if isinstance(report.get("codex_command_modes"), dict) else {},
        "codex_python_cli_hooks": report.get("codex_python_cli_hooks"),
        "codex_powershell_native_hooks": report.get("codex_powershell_native_hooks"),
        "codex_native_launcher_hooks": report.get("codex_native_launcher_hooks"),
        "repair_projection": report.get("repair_projection") if isinstance(report.get("repair_projection"), dict) else {},
        "post_repair_observations": report.get("post_repair_observations")
        if isinstance(report.get("post_repair_observations"), dict)
        else {},
        "post_repair_command_shapes": report.get("post_repair_command_shapes")
        if isinstance(report.get("post_repair_command_shapes"), dict)
        else {},
        "privacy": {
            "copied_fields": [
                "manifest-relative paths",
                "hook mode counts",
                "replacement counts",
                "post-repair counts",
                "shell shape categories",
            ],
            "dropped": ["hook command bodies", "commands", "Codex prompts", "tool arguments", "tool results"],
        },
    }


def codex_actionability(
    *,
    hook_fast_path: dict[str, Any],
    dos_audit: dict[str, Any],
    max_delegates: int = 0,
) -> dict[str, Any]:
    """Classify post-repair DOS findings into actionable risk vs residual debt."""
    observations = dos_audit.get("observations") if isinstance(dos_audit.get("observations"), dict) else {}
    stop_failures = dos_audit.get("stop_failures") if isinstance(dos_audit.get("stop_failures"), dict) else {}
    post_repair = (
        hook_fast_path.get("post_repair_observations")
        if isinstance(hook_fast_path.get("post_repair_observations"), dict)
        else {}
    )
    command_shapes = (
        hook_fast_path.get("post_repair_command_shapes")
        if isinstance(hook_fast_path.get("post_repair_command_shapes"), dict)
        else {}
    )
    try:
        audit_mod = load_codex_recent_audit()
        gate = audit_mod.actionable_gate(
            hook_fast_path=hook_fast_path,
            post_repair=post_repair,
            command_shapes=command_shapes,
            delegate_total=int(observations.get("delegate_count") or 0),
            stop_total=int(observations.get("stop_blocks") or 0) + int(stop_failures.get("total") or 0),
            max_delegates=max_delegates,
        )
    except (OSError, RuntimeError, ValueError, AttributeError, TypeError) as exc:
        return {"status": "UNKNOWN", "error": type(exc).__name__}
    if not isinstance(gate, dict):
        return {"status": "UNKNOWN", "error": "actionability gate did not return an object"}
    return {
        "status": gate.get("status"),
        "reasons": [str(item) for item in gate.get("reasons") or []],
        "unknowns": [str(item) for item in gate.get("unknowns") or []],
        "residual": [str(item) for item in gate.get("residual") or []],
        "delegate_source": gate.get("delegate_source"),
        "delegate_count": gate.get("delegate_count"),
        "max_delegates": gate.get("max_delegates"),
        "stop_total": gate.get("stop_total"),
        "post_repair_unknown_tree_admission_warnings": gate.get("post_repair_unknown_tree_admission_warnings"),
        "post_repair_shell_shape_counts": sorted_int_mapping(gate.get("post_repair_shell_shape_counts")),
        "post_repair_shell_family_counts": sorted_int_mapping(gate.get("post_repair_shell_family_counts")),
        "post_repair_mutating_shell_family_counts": sorted_int_mapping(
            gate.get("post_repair_mutating_shell_family_counts")
        ),
        "privacy": {
            "copied_fields": [
                "status",
                "counts",
                "reason classes",
                "residual classes",
                "shell shape categories",
                "shell family categories",
                "mutating shell family categories",
            ],
            "dropped": ["commands", "tool arguments", "tool results", "prompts", "model text", "diffs"],
        },
    }


def status_from_checks(
    preflight_deny: dict[str, Any],
    preflight_allow: dict[str, Any],
    mcp_stdio: dict[str, Any],
    codex_exec: dict[str, Any],
    proof: dict[str, Any],
    gate_reports: dict[str, Any],
) -> str:
    deny_ok = (
        preflight_deny.get("exit_code") == 0
        and preflight_deny.get("verdict") == "DENY"
        and preflight_deny.get("reason") == "POLICY_BLOCK"
    )
    allow_ok = preflight_allow.get("exit_code") == 0 and preflight_allow.get("verdict") == "ALLOW"
    mcp_ok = mcp_stdio.get("status") == "SKIPPED" or mcp_stdio.get("status") == "PASS"
    codex_exec_ok = codex_exec.get("status") == "SKIPPED" or codex_exec.get("status") == "PASS"
    gate_ok = gate_reports.get("status") in {"SKIPPED", "PASS"}
    proof_status = str(proof.get("status", "")).upper()
    proof_exit = proof.get("exit_code")
    if deny_ok and allow_ok and mcp_ok and codex_exec_ok and gate_ok and proof_status == "PROVEN" and proof_exit == 0:
        return "PROVEN"
    if deny_ok and allow_ok and mcp_ok and codex_exec_ok and gate_ok and proof_status == "REFUTED" and proof_exit == 1:
        return "REFUTED"
    return "ERROR"


def dogfood_summary(
    *,
    status: str,
    deny_tool: str,
    allow_tool: str,
    preflight_deny: dict[str, Any],
    preflight_allow: dict[str, Any],
    mcp_stdio: dict[str, Any],
    codex_exec: dict[str, Any],
    proof: dict[str, Any],
    dos_audit: dict[str, Any],
    dos_helped: dict[str, Any],
    gate_reports: dict[str, Any],
    hook_fast_path: dict[str, Any],
    actionability: dict[str, Any],
) -> dict[str, Any]:
    stream = dos_audit.get("stream") if isinstance(dos_audit.get("stream"), dict) else {}
    observations = dos_audit.get("observations") if isinstance(dos_audit.get("observations"), dict) else {}
    helped_by_tool = dos_helped.get("by_advisory_tool") if isinstance(dos_helped.get("by_advisory_tool"), dict) else {}
    post_repair = (
        hook_fast_path.get("post_repair_observations")
        if isinstance(hook_fast_path.get("post_repair_observations"), dict)
        else {}
    )
    command_shapes = (
        hook_fast_path.get("post_repair_command_shapes")
        if isinstance(hook_fast_path.get("post_repair_command_shapes"), dict)
        else {}
    )
    shell_shape_counts = (
        command_shapes.get("shell_shape_counts") if isinstance(command_shapes.get("shell_shape_counts"), dict) else {}
    )
    recommendations = [str(rec) for rec in dos_audit.get("recommendations") or []]

    next_actions: list[str] = []
    if mcp_stdio.get("status") not in {"PASS", "SKIPPED"}:
        next_actions.append("repair fak MCP stdio proof before treating the witness as grounded")
    if codex_exec.get("status") not in {"PASS", "SKIPPED"}:
        next_actions.append("rerun the Codex MCP probe or inspect its sanitized event stream")
    if gate_reports.get("status") not in {"PASS", "SKIPPED"}:
        next_actions.append("repair local fak-gated command reports before treating the operating loop as proven")
    if hook_fast_path.get("status") == "WARN":
        next_actions.append("run codex_dos_hook_doctor dry-run/apply to route Codex hooks through the native launcher")
    if actionability.get("status") == "WARN":
        next_actions.append("resolve post-repair actionable DOS dogfood risks before treating the run as clean")
    if actionability.get("status") == "UNKNOWN":
        next_actions.append("capture post-repair command-shape evidence before classifying residual DOS debt")
    if (
        hook_fast_path.get("status") == "PASS"
        and post_repair.get("status") == "WARN"
        and not post_repair.get("delegate_count")
        and post_repair.get("unknown_tree_admission_warnings")
    ):
        next_actions.append("fast path is repaired; remaining post-repair issue is unknown-tree admission for opaque Codex host calls")
    if shell_shape_counts.get("shell_no_write_target_detected"):
        next_actions.append("post-repair Codex shell calls include commands with no path-visible write target")
    if str(proof.get("status", "")).upper() not in {"PROVEN", "REFUTED"}:
        next_actions.append("repair vcache telemetry proof input or parser")
    next_actions.extend(recommendations)
    if helped_by_tool.get("Bash") and not any("Bash-dominated" in action for action in next_actions):
        next_actions.append("reduce Bash calls that DOS cannot scope to a declared file tree")

    return {
        "status": status,
        "policy_adjudication": {
            "deny": {
                "tool": deny_tool,
                "verdict": preflight_deny.get("verdict"),
                "reason": preflight_deny.get("reason"),
            },
            "allow": {
                "tool": allow_tool,
                "verdict": preflight_allow.get("verdict"),
                "reason": preflight_allow.get("reason"),
            },
            "mcp_stdio_status": mcp_stdio.get("status"),
            "codex_exec_mcp_status": codex_exec.get("status"),
        },
        "vcache": {
            "status": proof.get("status"),
            "requests": proof.get("requests"),
            "saved_pct": proof.get("saved_pct"),
            "saved_token_equiv": proof.get("saved_token_equiv"),
            "correctness_depends_on_hit": proof.get("correctness_depends_on_hit"),
        },
        "local_fak_gate": {
            "status": gate_reports.get("status"),
            "total": gate_reports.get("total", 0),
            "passed": gate_reports.get("passed", 0),
            "failed": gate_reports.get("failed", 0),
            "denied": gate_reports.get("denied", 0),
            "expected_denied": gate_reports.get("expected_denied", 0),
            "tools": sorted(
                {
                    str(report.get("tool"))
                    for report in gate_reports.get("reports", [])
                    if isinstance(report, dict) and report.get("tool")
                }
            ),
        },
        "codex_hook_fast_path": {
            "status": hook_fast_path.get("status"),
            "reason": hook_fast_path.get("reason"),
            "codex_python_cli_hooks": hook_fast_path.get("codex_python_cli_hooks"),
            "codex_powershell_native_hooks": hook_fast_path.get("codex_powershell_native_hooks"),
            "codex_native_launcher_hooks": hook_fast_path.get("codex_native_launcher_hooks"),
            "codex_command_modes": hook_fast_path.get("codex_command_modes") or {},
            "post_repair_status": post_repair.get("status"),
            "post_repair_observations": post_repair.get("observations"),
            "post_repair_delegate_count": post_repair.get("delegate_count"),
            "post_repair_unknown_tree_admission_warnings": post_repair.get("unknown_tree_admission_warnings"),
            "post_repair_unknown_tree_warning_rate": post_repair.get("unknown_tree_warning_rate"),
            "post_repair_command_shape_status": command_shapes.get("status"),
            "post_repair_tool_counts": command_shapes.get("tool_counts") or {},
            "post_repair_shell_shape_counts": shell_shape_counts,
            "would_clear_codex_python_cli": (hook_fast_path.get("repair_projection") or {}).get(
                "would_clear_codex_python_cli"
            )
            if isinstance(hook_fast_path.get("repair_projection"), dict)
            else None,
        },
        "codex_actionability": {
            "status": actionability.get("status"),
            "reasons": actionability.get("reasons") or [],
            "unknowns": actionability.get("unknowns") or [],
            "residual": actionability.get("residual") or [],
            "delegate_source": actionability.get("delegate_source"),
            "delegate_count": actionability.get("delegate_count"),
            "max_delegates": actionability.get("max_delegates"),
            "stop_total": actionability.get("stop_total"),
            "post_repair_unknown_tree_admission_warnings": actionability.get(
                "post_repair_unknown_tree_admission_warnings"
            ),
            "post_repair_shell_shape_counts": actionability.get("post_repair_shell_shape_counts") or {},
            "post_repair_shell_family_counts": actionability.get("post_repair_shell_family_counts") or {},
            "post_repair_mutating_shell_family_counts": actionability.get(
                "post_repair_mutating_shell_family_counts"
            )
            or {},
        },
        "dos": {
            "status": dos_audit.get("status"),
            "steps": stream.get("steps"),
            "tool_counts": stream.get("tool_counts") or {},
            "pretool_calls": observations.get("pretool_calls"),
            "unknown_tree_admission_warnings": observations.get("unknown_tree_admission_warnings"),
            "unknown_tree_warning_rate": observations.get("unknown_tree_warning_rate"),
            "unknown_tree_warning_threshold": observations.get("unknown_tree_warning_threshold"),
            "delegate_count": observations.get("delegate_count"),
            "stop_blocks": observations.get("stop_blocks"),
            "stop_failures_total": (dos_audit.get("stop_failures") or {}).get("total")
            if isinstance(dos_audit.get("stop_failures"), dict)
            else None,
            "session_advisory_cautions": dos_helped.get("advisory"),
            "session_refusals": dos_helped.get("blocked"),
            "session_advisory_by_tool": helped_by_tool,
            "recommendations": recommendations,
        },
        "next_actions": list(dict.fromkeys(next_actions)),
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_report(
    args: argparse.Namespace,
    *,
    env: dict[str, str],
    runner: Runner,
    mcp_probe: MCPProbe = run_mcp_stdio_probe,
    now: Callable[[], datetime],
) -> tuple[int, dict[str, Any]]:
    extract_mod = load_codex_extract()
    session, thread_id = resolve_session(args, extract_mod, env)
    out_path, usage_path = default_paths(args, thread_id)

    if not session.exists():
        raise FileNotFoundError(f"Codex session not found: {session}")

    rows = extract_mod.extract_rows(session)
    if not rows:
        raise RuntimeError(f"no Codex token usage rows found in {session}")
    extract_mod.write_rows(str(usage_path), rows)

    preflight_deny = run_preflight(args, runner, args.deny_tool)
    preflight_allow = run_preflight(args, runner, args.allow_tool)
    mcp_stdio = {"status": "SKIPPED", "reason": "--skip-mcp"} if args.skip_mcp else mcp_probe(args)
    if args.run_codex_exec:
        codex_exec = run_codex_exec_mcp_usage(args, runner)
    else:
        codex_exec = read_codex_exec_mcp_usage(args.codex_exec_jsonl or "", args.deny_tool, args.allow_tool)
    proof = run_vcache_proof(args, runner, usage_path)
    dos_audit = dos_session_audit(args, thread_id)
    dos_helped = dos_helped_audit(args, runner, thread_id)
    gate_reports = local_gate_reports(args)
    hook_fast_path = codex_hook_fast_path(args, extract_mod, env, thread_id, session)
    actionability = codex_actionability(hook_fast_path=hook_fast_path, dos_audit=dos_audit)
    status = status_from_checks(preflight_deny, preflight_allow, mcp_stdio, codex_exec, proof, gate_reports)

    report = {
        "schema": SCHEMA,
        "created_at": now().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "thread_id": thread_id,
        "session_file": session.name,
        "usage_jsonl": relpath(usage_path, args.repo_root),
        "usage_rows": len(rows),
        "summary": dogfood_summary(
            status=status,
            deny_tool=args.deny_tool,
            allow_tool=args.allow_tool,
            preflight_deny=preflight_deny,
            preflight_allow=preflight_allow,
            mcp_stdio=mcp_stdio,
            codex_exec=codex_exec,
            proof=proof,
            dos_audit=dos_audit,
            dos_helped=dos_helped,
            gate_reports=gate_reports,
            hook_fast_path=hook_fast_path,
            actionability=actionability,
        ),
        "checks": {
            "capability_floor_denies_publish": preflight_deny,
            "capability_floor_allows_status": preflight_allow,
            "mcp_stdio_adjudication": mcp_stdio,
            "codex_exec_mcp_usage": codex_exec,
            "local_fak_gate_reports": gate_reports,
            "codex_hook_fast_path": hook_fast_path,
            "codex_actionability": actionability,
            "vcache_telemetry_proof": proof,
            "dos_session_audit": dos_audit,
            "dos_helped_session": dos_helped,
        },
        "privacy": {
            "usage_jsonl_copied_fields": ["input_tokens", "cached_input_tokens"],
            "usage_jsonl_dropped": ["prompts", "tool_calls", "tool_outputs", "diffs", "model_text"],
        },
    }
    write_json(out_path, report)
    report["out"] = relpath(out_path, args.repo_root)

    if status == "PROVEN":
        return 0, report
    if status == "REFUTED":
        return 1, report
    return 2, report


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", help="raw Codex session or codex exec --json JSONL path")
    p.add_argument("--thread-id", help="Codex thread id; defaults to CODEX_THREAD_ID")
    p.add_argument("--codex-home", help="Codex home; defaults to CODEX_HOME or ~/.codex")
    p.add_argument("--out", help="witness JSON output path")
    p.add_argument("--usage-out", help="sanitized usage JSONL output path")
    p.add_argument("--repo-root", type=Path, default=default_repo_root(), help="fak repo root")
    p.add_argument("--fak-bin", help="use an already-built fak binary instead of `go run ./cmd/fak`")
    p.add_argument("--policy", default="examples/dev-agent-policy.json", help="capability floor policy")
    p.add_argument("--deny-tool", default="git_push", help="tool expected to be denied by the policy")
    p.add_argument("--allow-tool", default="git_status", help="tool expected to be allowed by the policy")
    p.add_argument("--read-mult", type=float, default=0.1, help="cached-input read multiplier for vcache proof")
    p.add_argument("--mcp-timeout", type=float, default=30.0, help="seconds to wait for one MCP stdio response")
    p.add_argument("--skip-mcp", action="store_true", help="skip the MCP stdio proof")
    p.add_argument("--codex-exec-jsonl", help="optional codex exec --json JSONL to prove Codex used the fak MCP server")
    p.add_argument("--run-codex-exec", action="store_true", help="run a nested codex exec --json MCP probe and sanitize its event stream")
    p.add_argument("--codex-bin", help="Codex executable for --run-codex-exec (default: codex on PATH)")
    p.add_argument("--gate-report", action="append", help="JSON report from tools/codex_fak_gate.py; repeat for multiple commands")
    return p.parse_args(argv)


def run(argv: list[str], *, env=os.environ, runner: Runner = run_subprocess, stdout=sys.stdout, stderr=sys.stderr) -> int:
    args = parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    try:
        code, report = build_report(args, env=env, runner=runner, now=lambda: datetime.now(timezone.utc))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"codex_dogfood_witness: {exc}", file=stderr)
        return 2

    print(json.dumps({"status": report["status"], "out": report["out"], "usage_rows": report["usage_rows"]}, sort_keys=True), file=stdout)
    return code


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
