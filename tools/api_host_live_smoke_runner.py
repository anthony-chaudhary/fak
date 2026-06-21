#!/usr/bin/env python3
"""Evaluate or execute ready API-host live-smoke queue rows."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import fleet_version


SCHEMA = "fak.api-host-live-smoke-runner.v1"
QUEUE_SCHEMA = "fak.api-host-live-smoke-queue.v1"
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PATHS = {
    "queue": "fak/experiments/api-host-bridge/api-host-live-smoke-queue.json",
}

TERMINAL_SKIP_STATES = {
    "COMPLETE": "ALREADY_COMPLETE",
    "BLOCKED_EXTERNAL_STATE": "SKIPPED_EXTERNAL_STATE",
    "WAITING_FOR_CREDENTIAL": "SKIPPED_WAITING_FOR_CREDENTIAL",
    "READY_FOR_PROBE": "SKIPPED_READY_FOR_PROBE",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def tail(text: str, n: int = 4000) -> str:
    return text[-n:]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def load_json(root: Path, rel_path: str) -> tuple[dict[str, Any] | None, str]:
    path = root / rel_path
    if not path.exists():
        return None, f"missing artifact: {rel_path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON in {rel_path}: {exc}"
    except OSError as exc:
        return None, f"cannot read artifact {rel_path}: {exc}"
    if not isinstance(data, dict):
        return None, f"artifact is not a JSON object: {rel_path}"
    return data, ""


def command_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def run_command(command: str, root: Path, timeout_s: int) -> dict[str, Any]:
    started = utc_now()
    t0 = time.perf_counter()
    version = fleet_version.app_version(root)
    try:
        proc = subprocess.run(
            command,
            cwd=str(root),
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        combined = proc.stdout + proc.stderr
        return {
            "version": version,
            "command": command,
            "status": "passed" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
            "started_at": started,
            "elapsed_ms": elapsed_ms,
            "stdout_tail": tail(proc.stdout),
            "stderr_tail": tail(proc.stderr),
            "output_sha256": digest(combined),
        }
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "version": version,
            "command": command,
            "status": "timeout",
            "exit_code": None,
            "started_at": started,
            "elapsed_ms": elapsed_ms,
            "stdout_tail": tail(stdout),
            "stderr_tail": tail(stderr),
            "output_sha256": digest(stdout + stderr),
            "timeout_s": timeout_s,
        }
    except OSError as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        message = str(exc)
        return {
            "version": version,
            "command": command,
            "status": "error",
            "exit_code": None,
            "started_at": started,
            "elapsed_ms": elapsed_ms,
            "stdout_tail": "",
            "stderr_tail": tail(message),
            "output_sha256": digest(message),
            "error_type": type(exc).__name__,
        }


def run_ready_row(row: dict[str, Any], root: Path, timeout_s: int) -> tuple[str, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    for command in command_list(row.get("commands")):
        result = run_command(command, root, timeout_s)
        results.append(result)
        if result["status"] != "passed":
            return "EXECUTED_FAILED", results
    return "EXECUTED_PASSED", results


def queue_rows(queue: dict[str, Any] | None, errors: dict[str, str]) -> list[dict[str, Any]]:
    if queue is None:
        return []
    rows = queue.get("queue", [])
    if not isinstance(rows, list):
        errors["queue_rows"] = "live-smoke queue artifact queue field is not a JSON list"
        return []
    malformed = len([item for item in rows if not isinstance(item, dict)])
    if malformed:
        errors["queue_row_shapes"] = f"{malformed} queue rows are not JSON objects"
    return [item for item in rows if isinstance(item, dict)]


def build_runner_row(row: dict[str, Any], root: Path, execute_ready: bool, timeout_s: int, version: str | None = None) -> dict[str, Any]:
    queue_state = str(row.get("queue_state") or "UNCLASSIFIED")
    commands = command_list(row.get("commands"))
    version = version or fleet_version.app_version(root)
    out: dict[str, Any] = {
        "version": version,
        "target": row.get("target"),
        "queue_state": queue_state,
        "operator_prerequisite": row.get("operator_prerequisite", ""),
        "command_count": len(commands),
        "commands": commands,
        "runner_status": "UNCLASSIFIED",
        "command_results": [],
    }
    if queue_state in TERMINAL_SKIP_STATES:
        out["runner_status"] = TERMINAL_SKIP_STATES[queue_state]
    elif queue_state == "READY_TO_EXECUTE":
        if not commands:
            out["runner_status"] = "READY_COMMAND_GAP"
        elif not execute_ready:
            out["runner_status"] = "READY_NOT_EXECUTED"
        else:
            status, results = run_ready_row(row, root, timeout_s)
            out["runner_status"] = status
            out["command_results"] = results
    return out


def build_report(
    root: Path | None = None,
    paths: dict[str, str] | None = None,
    execute_ready: bool = False,
    timeout_s: int = 900,
) -> dict[str, Any]:
    root = root or ROOT
    paths = paths or DEFAULT_PATHS
    app_ver = fleet_version.app_version(root)
    queue, queue_error = load_json(root, paths["queue"])
    errors: dict[str, str] = {}
    if queue_error:
        errors["queue"] = queue_error

    queue_summary = (queue or {}).get("summary", {})
    if queue is not None and queue.get("schema") != QUEUE_SCHEMA:
        errors["queue_schema"] = f"live-smoke queue artifact schema is not {QUEUE_SCHEMA}"
    if queue is not None and queue_summary.get("live_smoke_queue_gate") is not True:
        errors["queue_gate"] = "live-smoke queue gate is not true"

    rows = [
        build_runner_row(row, root, execute_ready, timeout_s, app_ver)
        for row in queue_rows(queue, errors)
    ]
    statuses = [str(row.get("runner_status") or "UNCLASSIFIED") for row in rows]
    failed = [status for status in statuses if status in {"EXECUTED_FAILED", "READY_COMMAND_GAP", "timeout", "error"}]
    summary = {
        "targets": len(rows),
        "execute_ready": execute_ready,
        "already_complete": statuses.count("ALREADY_COMPLETE"),
        "ready_to_execute": len([row for row in rows if row.get("queue_state") == "READY_TO_EXECUTE"]),
        "ready_not_executed": statuses.count("READY_NOT_EXECUTED"),
        "executed": statuses.count("EXECUTED_PASSED") + statuses.count("EXECUTED_FAILED"),
        "passed": statuses.count("EXECUTED_PASSED"),
        "failed": statuses.count("EXECUTED_FAILED"),
        "skipped_external_state": statuses.count("SKIPPED_EXTERNAL_STATE"),
        "skipped_waiting_for_credential": statuses.count("SKIPPED_WAITING_FOR_CREDENTIAL"),
        "skipped_ready_for_probe": statuses.count("SKIPPED_READY_FOR_PROBE"),
        "command_gaps": statuses.count("READY_COMMAND_GAP"),
        "unclassified": statuses.count("UNCLASSIFIED"),
        "artifact_errors": len(errors),
    }
    summary["ready_execution_gaps"] = summary["ready_not_executed"] + summary["command_gaps"]
    summary["live_smoke_runner_gate"] = (
        summary["targets"] > 0
        and summary["artifact_errors"] == 0
        and summary["unclassified"] == 0
        and summary["ready_execution_gaps"] == 0
        and summary["failed"] == 0
        and not failed
    )

    return {
        "schema": SCHEMA,
        "app_version": app_ver,
        "generated_at": utc_now(),
        "scope_note": (
            "Execution ledger for API-host live-smoke queue rows. By default it does not "
            "spend tokens or call paid/keyed hosts; pass --execute-ready to run rows whose "
            "queue state is READY_TO_EXECUTE. The gate fails closed if ready rows are left "
            "unexecuted."
        ),
        "summary": summary,
        "runs": rows,
        "artifact_errors": errors,
        "artifacts": paths,
        "queue_summary": queue_summary,
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# API-Host Live Smoke Runner",
        "",
        "> Execution ledger for ready API-host live-smoke queue rows.",
        "",
        "## Summary",
        "",
        f"- Execute ready rows: {'yes' if s['execute_ready'] else 'no'}",
        f"- Targets: {s['targets']}",
        f"- Already complete: {s['already_complete']}",
        f"- Ready to execute: {s['ready_to_execute']}",
        f"- Ready not executed: {s['ready_not_executed']}",
        f"- Executed: {s['executed']}",
        f"- Passed: {s['passed']}",
        f"- Failed: {s['failed']}",
        f"- Skipped external state: {s['skipped_external_state']}",
        f"- Waiting for credential: {s['skipped_waiting_for_credential']}",
        f"- Ready for probe: {s['skipped_ready_for_probe']}",
        f"- Runner gate: {'yes' if s['live_smoke_runner_gate'] else 'no'}",
        "",
        "| target | queue state | runner status | command count |",
        "|---|---|---|---:|",
    ]
    for row in report["runs"]:
        lines.append(
            f"| `{row['target']}` | {row['queue_state']} | {row['runner_status']} | {row['command_count']} |"
        )
    if report.get("artifact_errors"):
        lines += ["", "## Artifact Errors", "", "```json", json.dumps(report["artifact_errors"], indent=2), "```", ""]
    lines.append("")
    return "\n".join(lines)


def write_text(path: str, body: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate or execute ready API-host live-smoke queue rows")
    ap.add_argument("--out", default="", help="write JSON report here")
    ap.add_argument("--markdown", default="", help="write Markdown report here")
    ap.add_argument("--root", default=str(ROOT), help="workspace root")
    ap.add_argument("--execute-ready", action="store_true", help="run READY_TO_EXECUTE queue rows")
    ap.add_argument("--timeout-s", type=int, default=900, help="timeout per ready command")
    args = ap.parse_args(argv)

    report = build_report(Path(args.root), execute_ready=args.execute_ready, timeout_s=args.timeout_s)
    body = json.dumps(report, indent=2) + "\n"
    if args.out:
        write_text(args.out, body)
    else:
        print(body, end="")
    if args.markdown:
        write_text(args.markdown, markdown(report))
    return 0 if report["summary"]["live_smoke_runner_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
