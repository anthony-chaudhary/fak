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

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, since_days))
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
    codex_native = int(codex_command_modes.get("native_launcher") or 0)
    if codex_python:
        status = "WARN"
        reason = "Codex hook commands route through the Python CLI hook instead of the bundled native launcher"
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
        reasons.append("stop blocks or stop failures are present")

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
    if hook_fast_path.get("status") == "WARN":
        recommendations.append("Codex hook manifest uses the Python hook path; track native-fast-path wiring separately from package freshness")
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

    status = "PASS"
    if not sessions:
        status = "UNKNOWN"
        recommendations.append("no recent Codex sessions matched DOS streams")
    elif warned or hook_fast_path.get("status") == "WARN":
        status = "WARN"
    stop_total = sum(s["stop_blocks"] + s["stop_failures_total"] for s in sessions)
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


def render_debt_packet(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    hook = report.get("codex_hook_fast_path") if isinstance(report.get("codex_hook_fast_path"), dict) else {}
    post = hook.get("post_repair_observations") if isinstance(hook.get("post_repair_observations"), dict) else {}
    shapes = hook.get("post_repair_command_shapes") if isinstance(hook.get("post_repair_command_shapes"), dict) else {}
    actionability = report.get("actionability") if isinstance(report.get("actionability"), dict) else {}
    dos_version = report.get("dos_version") if isinstance(report.get("dos_version"), dict) else {}
    git_gate = report.get("git_gate_evidence") if isinstance(report.get("git_gate_evidence"), dict) else {}
    mutating_families = actionability.get("post_repair_mutating_shell_family_counts") or {}
    if mutating_families:
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
    lines = [
        f"codex DOS recent audit: {report.get('status')}",
        f"  sessions audited: {report.get('sessions_audited')} of {report.get('codex_threads_discovered')} discovered Codex threads",
        f"  steps: {summary.get('steps')}  tools: {json.dumps(summary.get('tool_counts') or {}, sort_keys=True)}",
        f"  unknown-tree warnings: {summary.get('unknown_tree_admission_warnings')} / {summary.get('pretool_calls')} pretool calls"
        f" ({summary.get('unknown_tree_warning_rate')})",
        f"  delegates: {summary.get('delegate_count')}  stop blocks: {summary.get('stop_blocks')}  stop failures: {summary.get('stop_failures_total')}",
    ]
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
