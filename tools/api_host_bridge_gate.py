#!/usr/bin/env python3
"""Execute required API-host bridge witness commands."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import api_host_bridge_matrix as matrix
import fleet_version


SCHEMA = "fak.api-host-bridge-gate.v1"
ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def tail(text: str, n: int = 4000) -> str:
    return text[-n:]


def run_command(root: Path, witness: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    argv = witness.get("argv") or []
    cwd_rel = witness.get("cwd") or ""
    base = {
        "version": fleet_version.app_version(root),
        "id": witness["id"],
        "claim": witness["claim"],
        "required": witness["required"],
        "command": witness["command"],
        "cwd": cwd_rel,
        "argv": argv,
        "started_at": utc_now(),
    }
    if not argv:
        return {**base, "status": "skipped", "exit_code": None, "elapsed_ms": 0, "reason": "no argv"}
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(root / cwd_rel if cwd_rel else root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        combined = proc.stdout + proc.stderr
        return {
            **base,
            "status": "passed" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
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
            **base,
            "status": "timeout",
            "exit_code": None,
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
            **base,
            "status": "error",
            "exit_code": None,
            "elapsed_ms": elapsed_ms,
            "stdout_tail": "",
            "stderr_tail": tail(message),
            "output_sha256": digest(message),
            "error_type": type(exc).__name__,
        }


def build_report(root: Path | None = None, timeout_s: int = 180, execute: bool = True) -> dict[str, Any]:
    root = root or ROOT
    app_ver = fleet_version.app_version(root)
    index = matrix.build_report(root)
    runs: list[dict[str, Any]] = []
    for witness in index["witnesses"]:
        if not witness["required"]:
            runs.append({**{k: witness[k] for k in ("version", "id", "claim", "required", "command", "cwd", "argv")}, "status": "skipped", "reason": "supporting witness"})
        elif witness["status"] != "resolved":
            runs.append({**{k: witness[k] for k in ("version", "id", "claim", "required", "command", "cwd", "argv")}, "status": "index_unresolved", "reason": "source witness unresolved"})
        elif execute:
            runs.append(run_command(root, witness, timeout_s))
        else:
            runs.append({**{k: witness[k] for k in ("version", "id", "claim", "required", "command", "cwd", "argv")}, "status": "not_run", "reason": "execute=false"})
    required = [r for r in runs if r["required"]]
    passed = [r for r in required if r["status"] == "passed"]
    return {
        "schema": SCHEMA,
        "app_version": app_ver,
        "generated_at": utc_now(),
        "bridge_claim": index["bridge_claim"],
        "timeout_s": timeout_s,
        "matrix_schema": index["schema"],
        "matrix_summary": index["summary"],
        "summary": {
            "required_witnesses": len(required),
            "passed_required_witnesses": len(passed),
            "failed_required_witnesses": len(required) - len(passed),
            "executable_bridge_covered": index["summary"]["bridge_covered"] and len(required) == len(passed) and len(required) > 0,
        },
        "runs": runs,
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# API-Host Bridge Gate",
        "",
        f"- Required witness commands passed: {s['passed_required_witnesses']}/{s['required_witnesses']}",
        f"- Executable bridge covered: {'yes' if s['executable_bridge_covered'] else 'no'}",
        "",
        "| witness | required | status | elapsed ms | command |",
        "|---|:---:|---|---:|---|",
    ]
    for run in report["runs"]:
        lines.append(f"| `{run['id']}` | {'yes' if run['required'] else 'no'} | {run['status']} | {run.get('elapsed_ms', '')} | `{run['command']}` |")
    lines.append("")
    return "\n".join(lines)


def write_text(path: str, body: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run API-host bridge witness commands")
    ap.add_argument("--out", default="")
    ap.add_argument("--markdown", default="")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--timeout-s", type=int, default=180)
    ap.add_argument("--index-only", action="store_true")
    args = ap.parse_args(argv)
    report = build_report(Path(args.root), timeout_s=args.timeout_s, execute=not args.index_only)
    body = json.dumps(report, indent=2) + "\n"
    if args.out:
        write_text(args.out, body)
    else:
        print(body, end="")
    if args.markdown:
        write_text(args.markdown, markdown(report))
    if args.index_only:
        return 0 if report["matrix_summary"]["bridge_covered"] else 1
    return 0 if report["summary"]["executable_bridge_covered"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
