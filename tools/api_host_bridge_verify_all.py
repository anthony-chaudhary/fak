#!/usr/bin/env python3
"""Regenerate and verify the API-host bridge proof bundle in one command."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import fleet_version


SCHEMA = "fak.api-host-bridge-verify-all.v1"
ROOT = Path(__file__).resolve().parents[1]

PROOF_PATH = "fak/experiments/api-host-bridge/api-host-bridge-proof.json"
GOAL_PATH = "fak/experiments/api-host-bridge/api-host-goal-audit.json"
BENCHMARK_PATH = "fak/experiments/permission-systems/permission-system-benchmark.json"
SOURCE_AUDIT_PATH = "fak/experiments/permission-systems/permission-source-audit.json"
CERTIFICATE_PATH = "fak/experiments/api-host-bridge/api-host-conformance-certificate.json"
EXTERNAL_STATE_PATH = "fak/experiments/api-host-bridge/api-host-external-state-audit.json"
QUALIFICATION_PATH = "fak/experiments/api-host-bridge/api-host-qualification.json"
LIVE_QUEUE_PATH = "fak/experiments/api-host-bridge/api-host-live-smoke-queue.json"
LIVE_RUNNER_PATH = "fak/experiments/api-host-bridge/api-host-live-smoke-runner.json"

EXPECTED_ARTIFACT_SCHEMAS = {
    "proof": "fak.api-host-bridge-proof.v1",
    "goal": "fak.api-host-goal-audit.v1",
    "benchmark": "fak.permission-system-benchmark.v1",
    "source_audit": "fak.permission-source-audit.v1",
    "certificate": "fak.api-host-conformance-certificate.v1",
    "external_state": "fak.api-host-external-state-audit.v1",
    "qualification": "fak.api-host-qualification.v1",
    "live_queue": "fak.api-host-live-smoke-queue.v1",
    "live_runner": "fak.api-host-live-smoke-runner.v1",
}

ARTIFACT_PATHS = {
    "proof": PROOF_PATH,
    "goal": GOAL_PATH,
    "benchmark": BENCHMARK_PATH,
    "source_audit": SOURCE_AUDIT_PATH,
    "certificate": CERTIFICATE_PATH,
    "external_state": EXTERNAL_STATE_PATH,
    "qualification": QUALIFICATION_PATH,
    "live_queue": LIVE_QUEUE_PATH,
    "live_runner": LIVE_RUNNER_PATH,
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def tail(text: str, n: int = 4000) -> str:
    return text[-n:]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def py_cmd(script: str, *args: str) -> list[str]:
    return [sys.executable, script, *args]


def generation_steps() -> list[dict[str, Any]]:
    return [
        {
            "id": "permission_system_benchmark",
            "phase": "generate",
            "argv": py_cmd(
                "tools/permission_system_benchmark.py",
                "--out", "fak/experiments/permission-systems/permission-system-benchmark.json",
                "--markdown", "fak/experiments/permission-systems/permission-system-benchmark.md",
            ),
        },
        {
            "id": "permission_source_audit",
            "phase": "generate",
            "argv": py_cmd(
                "tools/permission_source_audit.py",
                "--out", "fak/experiments/permission-systems/permission-source-audit.json",
                "--markdown", "fak/experiments/permission-systems/permission-source-audit.md",
            ),
        },
        {
            "id": "api_host_bridge_matrix",
            "phase": "generate",
            "argv": py_cmd(
                "tools/api_host_bridge_matrix.py",
                "--out", "fak/experiments/api-host-bridge/api-host-bridge-matrix.json",
                "--markdown", "fak/experiments/api-host-bridge/api-host-bridge-matrix.md",
            ),
        },
        {
            "id": "api_host_bridge_gate",
            "phase": "generate",
            "argv": py_cmd(
                "tools/api_host_bridge_gate.py",
                "--out", "fak/experiments/api-host-bridge/api-host-bridge-gate.json",
                "--markdown", "fak/experiments/api-host-bridge/api-host-bridge-gate.md",
            ),
            "timeout_s": 240,
        },
        {
            "id": "api_host_live_inventory",
            "phase": "generate",
            "argv": py_cmd(
                "tools/api_host_live_inventory.py",
                "--out", "fak/experiments/api-host-bridge/api-host-live-inventory.json",
                "--markdown", "fak/experiments/api-host-bridge/api-host-live-inventory.md",
            ),
        },
        {
            "id": "api_host_roster",
            "phase": "generate",
            "argv": py_cmd(
                "tools/api_host_roster.py",
                "--out", "fak/experiments/api-host-bridge/api-host-roster.json",
                "--markdown", "fak/experiments/api-host-bridge/api-host-roster.md",
            ),
        },
        {
            "id": "api_host_readiness",
            "phase": "generate",
            "argv": py_cmd(
                "tools/api_host_readiness_probe.py",
                "--from-roster", "fak/experiments/api-host-bridge/api-host-roster.json",
                "--out", "fak/experiments/api-host-bridge/api-host-readiness.json",
                "--markdown", "fak/experiments/api-host-bridge/api-host-readiness.md",
            ),
        },
        {
            "id": "api_host_acceptance",
            "phase": "generate",
            "argv": py_cmd(
                "tools/api_host_acceptance_probe.py",
                "--from-roster", "fak/experiments/api-host-bridge/api-host-roster.json",
                "--out", "fak/experiments/api-host-bridge/api-host-acceptance.json",
                "--markdown", "fak/experiments/api-host-bridge/api-host-acceptance.md",
            ),
        },
        {
            "id": "api_host_retry_packet",
            "phase": "generate",
            "argv": py_cmd(
                "tools/api_host_retry_packet.py",
                "--out", "fak/experiments/api-host-bridge/api-host-retry-packet.json",
                "--markdown", "fak/experiments/api-host-bridge/api-host-retry-packet.md",
            ),
        },
        {
            "id": "api_host_external_state_audit",
            "phase": "generate",
            "argv": py_cmd(
                "tools/api_host_external_state_audit.py",
                "--out", EXTERNAL_STATE_PATH,
                "--markdown", "fak/experiments/api-host-bridge/api-host-external-state-audit.md",
            ),
        },
        {
            "id": "api_host_compat_contract",
            "phase": "generate",
            "argv": py_cmd(
                "tools/api_host_compat_contract.py",
                "--out", "fak/experiments/api-host-bridge/api-host-compat-contract.json",
                "--markdown", "fak/experiments/api-host-bridge/api-host-compat-contract.md",
            ),
        },
        {
            "id": "api_host_conformance_certificate",
            "phase": "generate",
            "argv": py_cmd(
                "tools/api_host_conformance_certificate.py",
                "--out", "fak/experiments/api-host-bridge/api-host-conformance-certificate.json",
                "--markdown", "fak/experiments/api-host-bridge/api-host-conformance-certificate.md",
            ),
        },
        {
            "id": "api_host_qualification",
            "phase": "generate",
            "argv": py_cmd(
                "tools/api_host_qualification.py",
                "--out", QUALIFICATION_PATH,
                "--markdown", "fak/experiments/api-host-bridge/api-host-qualification.md",
            ),
        },
        {
            "id": "api_host_live_smoke_queue",
            "phase": "generate",
            "argv": py_cmd(
                "tools/api_host_live_smoke_queue.py",
                "--out", LIVE_QUEUE_PATH,
                "--markdown", "fak/experiments/api-host-bridge/api-host-live-smoke-queue.md",
            ),
        },
        {
            "id": "api_host_live_smoke_runner",
            "phase": "generate",
            "argv": py_cmd(
                "tools/api_host_live_smoke_runner.py",
                "--out", LIVE_RUNNER_PATH,
                "--markdown", "fak/experiments/api-host-bridge/api-host-live-smoke-runner.md",
            ),
        },
        {
            "id": "api_host_bridge_proof",
            "phase": "generate",
            "argv": py_cmd(
                "tools/api_host_bridge_proof.py",
                "--out", "fak/experiments/api-host-bridge/api-host-bridge-proof.json",
                "--markdown", "fak/experiments/api-host-bridge/api-host-bridge-proof.md",
            ),
        },
        {
            "id": "api_host_goal_audit",
            "phase": "generate",
            "argv": py_cmd(
                "tools/api_host_goal_audit.py",
                "--out", "fak/experiments/api-host-bridge/api-host-goal-audit.json",
                "--markdown", "fak/experiments/api-host-bridge/api-host-goal-audit.md",
            ),
        },
    ]


def test_steps() -> list[dict[str, Any]]:
    return [
        {"id": "permission_source_audit_test", "phase": "test", "argv": py_cmd("tools/permission_source_audit_test.py")},
        {"id": "permission_system_benchmark_test", "phase": "test", "argv": py_cmd("tools/permission_system_benchmark_test.py")},
        {"id": "api_host_bridge_matrix_test", "phase": "test", "argv": py_cmd("tools/api_host_bridge_matrix_test.py")},
        {"id": "api_host_bridge_gate_test", "phase": "test", "argv": py_cmd("tools/api_host_bridge_gate_test.py")},
        {"id": "api_host_live_inventory_test", "phase": "test", "argv": py_cmd("tools/api_host_live_inventory_test.py")},
        {"id": "api_host_readiness_probe_test", "phase": "test", "argv": py_cmd("tools/api_host_readiness_probe_test.py")},
        {"id": "api_host_acceptance_probe_test", "phase": "test", "argv": py_cmd("tools/api_host_acceptance_probe_test.py")},
        {"id": "api_host_roster_test", "phase": "test", "argv": py_cmd("tools/api_host_roster_test.py")},
        {"id": "api_host_retry_packet_test", "phase": "test", "argv": py_cmd("tools/api_host_retry_packet_test.py")},
        {"id": "api_host_external_state_audit_test", "phase": "test", "argv": py_cmd("tools/api_host_external_state_audit_test.py")},
        {"id": "api_host_compat_contract_test", "phase": "test", "argv": py_cmd("tools/api_host_compat_contract_test.py")},
        {"id": "api_host_conformance_certificate_test", "phase": "test", "argv": py_cmd("tools/api_host_conformance_certificate_test.py")},
        {"id": "api_host_qualification_test", "phase": "test", "argv": py_cmd("tools/api_host_qualification_test.py")},
        {"id": "api_host_live_smoke_queue_test", "phase": "test", "argv": py_cmd("tools/api_host_live_smoke_queue_test.py")},
        {"id": "api_host_live_smoke_runner_test", "phase": "test", "argv": py_cmd("tools/api_host_live_smoke_runner_test.py")},
        {"id": "api_host_bridge_proof_test", "phase": "test", "argv": py_cmd("tools/api_host_bridge_proof_test.py")},
        {"id": "api_host_goal_audit_test", "phase": "test", "argv": py_cmd("tools/api_host_goal_audit_test.py")},
        {"id": "gateway_go_tests", "phase": "test", "argv": ["go", "test", "./internal/gateway", "-count=1"], "cwd": "fak"},
    ]


def command_text(step: dict[str, Any]) -> str:
    return " ".join(step["argv"])


def run_step(root: Path, step: dict[str, Any], default_timeout_s: int, version: str | None = None) -> dict[str, Any]:
    cwd = root / step.get("cwd", "")
    timeout_s = int(step.get("timeout_s", default_timeout_s))
    started = utc_now()
    t0 = time.perf_counter()
    version = version or fleet_version.app_version(root)
    try:
        proc = subprocess.run(
            step["argv"],
            cwd=str(cwd),
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
            "id": step["id"],
            "phase": step["phase"],
            "status": "passed" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
            "cwd": step.get("cwd", ""),
            "command": command_text(step),
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
            "id": step["id"],
            "phase": step["phase"],
            "status": "timeout",
            "exit_code": None,
            "cwd": step.get("cwd", ""),
            "command": command_text(step),
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
            "id": step["id"],
            "phase": step["phase"],
            "status": "error",
            "exit_code": None,
            "cwd": step.get("cwd", ""),
            "command": command_text(step),
            "started_at": started,
            "elapsed_ms": elapsed_ms,
            "stdout_tail": "",
            "stderr_tail": tail(message),
            "output_sha256": digest(message),
            "error_type": type(exc).__name__,
        }


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


def artifact_summary(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for key, rel_path in ARTIFACT_PATHS.items():
        data, err = load_json(root, rel_path)
        if err:
            errors[key] = err
            out[key] = {}
        else:
            out[key] = data or {}
            expected_schema = EXPECTED_ARTIFACT_SCHEMAS[key]
            if out[key].get("schema") != expected_schema:
                errors[f"{key}_schema"] = f"{key} artifact schema is not {expected_schema}"
    proof_summary = out["proof"].get("summary", {})
    goal_summary = out["goal"].get("summary", {})
    benchmark_metrics = out["benchmark"].get("metrics")
    benchmark_summary = {
        "metrics": len(benchmark_metrics) if isinstance(benchmark_metrics, list) else 0,
        "benchmark_gate": isinstance(benchmark_metrics, list) and bool(benchmark_metrics),
    }
    source_audit_summary = out["source_audit"].get("summary", {})
    certificate_summary = out["certificate"].get("summary", {})
    external_state_summary = out["external_state"].get("summary", {})
    qualification_summary = out["qualification"].get("summary", {})
    live_queue_summary = out["live_queue"].get("summary", {})
    live_runner_summary = out["live_runner"].get("summary", {})
    residuals = [
        row for row in out["goal"].get("requirements", [])
        if isinstance(row, dict) and row.get("status") in {"NOT_PROVEN", "EXTERNAL_STATE"}
    ]
    incomplete = [
        row for row in out["goal"].get("requirements", [])
        if isinstance(row, dict) and row.get("status") not in {"PROVEN", "NOT_PROVEN", "EXTERNAL_STATE"}
    ]
    return {
        "proof_summary": proof_summary,
        "goal_summary": goal_summary,
        "benchmark_summary": benchmark_summary,
        "source_audit_summary": source_audit_summary,
        "certificate_summary": certificate_summary,
        "external_state_summary": external_state_summary,
        "qualification_summary": qualification_summary,
        "live_queue_summary": live_queue_summary,
        "live_runner_summary": live_runner_summary,
        "residual_requirements": residuals,
        "incomplete_requirements": incomplete,
        "artifact_errors": errors,
    }


def evaluate(results: list[dict[str, Any]], artifacts: dict[str, Any], artifacts_only: bool) -> dict[str, Any]:
    executed = [r for r in results if r["status"] != "skipped"]
    failed = [r for r in executed if r["status"] != "passed"]
    proof_summary = artifacts["proof_summary"]
    goal_summary = artifacts["goal_summary"]
    benchmark_summary = artifacts["benchmark_summary"]
    source_audit_summary = artifacts["source_audit_summary"]
    certificate_summary = artifacts["certificate_summary"]
    external_state_summary = artifacts["external_state_summary"]
    qualification_summary = artifacts["qualification_summary"]
    live_queue_summary = artifacts["live_queue_summary"]
    live_runner_summary = artifacts["live_runner_summary"]
    artifact_errors = artifacts["artifact_errors"]
    proof_gate = proof_summary.get("proof_gate") is True
    benchmark_gate = benchmark_summary.get("benchmark_gate") is True
    source_audit_gate = source_audit_summary.get("source_audit_gate") is True
    certificate_gate = certificate_summary.get("certificate_gate") is True
    external_state_gate = external_state_summary.get("external_state_audit_gate") is True
    qualification_gate = qualification_summary.get("qualification_gate") is True
    live_smoke_queue_gate = live_queue_summary.get("live_smoke_queue_gate") is True
    live_smoke_runner_gate = live_runner_summary.get("live_smoke_runner_gate") is True
    goal_no_incomplete = goal_summary.get("incomplete") == 0
    goal_scope_bounded = goal_summary.get("goal_status") == "SCOPE_BOUNDED_PROGRESS_NOT_COMPLETE"
    command_gate = artifacts_only or (bool(executed) and not failed)
    scope_bounded_gate = (
        command_gate
        and not artifact_errors
        and proof_gate
        and benchmark_gate
        and source_audit_gate
        and certificate_gate
        and external_state_gate
        and qualification_gate
        and live_smoke_queue_gate
        and live_smoke_runner_gate
        and goal_no_incomplete
    )
    return {
        "steps": len(results),
        "executed_steps": len(executed),
        "passed_steps": len([r for r in executed if r["status"] == "passed"]),
        "failed_steps": len(failed),
        "artifact_errors": len(artifact_errors),
        "proof_gate": proof_gate,
        "benchmark_gate": benchmark_gate,
        "source_audit_gate": source_audit_gate,
        "certificate_gate": certificate_gate,
        "external_state_gate": external_state_gate,
        "qualification_gate": qualification_gate,
        "live_smoke_queue_gate": live_smoke_queue_gate,
        "live_smoke_runner_gate": live_smoke_runner_gate,
        "goal_complete": goal_summary.get("goal_complete") is True,
        "goal_no_incomplete": goal_no_incomplete,
        "goal_scope_bounded": goal_scope_bounded,
        "residual_requirements": len(artifacts["residual_requirements"]),
        "scope_bounded_verification_gate": scope_bounded_gate,
    }


def build_report(
    root: Path | None = None,
    artifacts_only: bool = False,
    skip_tests: bool = False,
    timeout_s: int = 180,
) -> dict[str, Any]:
    root = root or ROOT
    app_ver = fleet_version.app_version(root)
    steps = generation_steps()
    if not skip_tests:
        steps += test_steps()
    results: list[dict[str, Any]] = []
    if artifacts_only:
        results = [
            {
                "version": app_ver,
                "id": step["id"],
                "phase": step["phase"],
                "status": "skipped",
                "reason": "artifacts_only",
                "cwd": step.get("cwd", ""),
                "command": command_text(step),
            }
            for step in steps
        ]
    else:
        for step in steps:
            result = run_step(root, step, timeout_s, app_ver)
            results.append(result)
            if result["status"] != "passed":
                break
    artifacts = artifact_summary(root)
    return {
        "schema": SCHEMA,
        "app_version": app_ver,
        "generated_at": utc_now(),
        "scope_note": (
            "One-command verifier for the scope-bounded API-host bridge proof. "
            "A green scope_bounded_verification_gate confirms the bounded proof and "
            "reports residuals; it does not prove every API host on the internet."
        ),
        "summary": evaluate(results, artifacts, artifacts_only),
        "steps": results,
        "artifact_summaries": {
            "proof": artifacts["proof_summary"],
            "goal": artifacts["goal_summary"],
            "benchmark": artifacts["benchmark_summary"],
            "source_audit": artifacts["source_audit_summary"],
            "certificate": artifacts["certificate_summary"],
            "external_state": artifacts["external_state_summary"],
            "qualification": artifacts["qualification_summary"],
            "live_queue": artifacts["live_queue_summary"],
            "live_runner": artifacts["live_runner_summary"],
        },
        "residual_requirements": artifacts["residual_requirements"],
        "incomplete_requirements": artifacts["incomplete_requirements"],
        "artifact_errors": artifacts["artifact_errors"],
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# API-Host Bridge Verify All",
        "",
        report["scope_note"],
        "",
        "## Summary",
        "",
        f"- Executed steps passed: {s['passed_steps']}/{s['executed_steps']}",
        f"- Failed steps: {s['failed_steps']}",
        f"- Proof gate: {'yes' if s['proof_gate'] else 'no'}",
        f"- Permission benchmark gate: {'yes' if s['benchmark_gate'] else 'no'}",
        f"- Permission source-audit gate: {'yes' if s['source_audit_gate'] else 'no'}",
        f"- Certificate gate: {'yes' if s['certificate_gate'] else 'no'}",
        f"- External-state audit gate: {'yes' if s['external_state_gate'] else 'no'}",
        f"- Qualification gate: {'yes' if s['qualification_gate'] else 'no'}",
        f"- Live-smoke queue gate: {'yes' if s['live_smoke_queue_gate'] else 'no'}",
        f"- Live-smoke runner gate: {'yes' if s['live_smoke_runner_gate'] else 'no'}",
        f"- Goal complete: {'yes' if s['goal_complete'] else 'no'}",
        f"- Residual requirements: {s['residual_requirements']}",
        f"- Scope-bounded verification gate: {'yes' if s['scope_bounded_verification_gate'] else 'no'}",
        "",
        "| step | phase | status | elapsed ms |",
        "|---|---|---|---:|",
    ]
    for step in report["steps"]:
        lines.append(f"| `{step['id']}` | {step['phase']} | {step['status']} | {step.get('elapsed_ms', '')} |")
    if report["residual_requirements"]:
        lines += ["", "## Residual Requirements", "", "| requirement | status |", "|---|---|"]
        for row in report["residual_requirements"]:
            lines.append(f"| `{row.get('id')}` | {row.get('status')} |")
    lines.append("")
    return "\n".join(lines)


def write_text(path: str, body: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate and verify the API-host bridge proof bundle")
    ap.add_argument("--out", default="", help="write JSON report here")
    ap.add_argument("--markdown", default="", help="write Markdown report here")
    ap.add_argument("--root", default=str(ROOT), help="workspace root")
    ap.add_argument("--artifacts-only", action="store_true", help="verify existing artifacts without running commands")
    ap.add_argument("--skip-tests", action="store_true", help="regenerate artifacts but skip focused test commands")
    ap.add_argument("--timeout-s", type=int, default=180, help="default timeout per command")
    args = ap.parse_args(argv)

    report = build_report(Path(args.root), artifacts_only=args.artifacts_only, skip_tests=args.skip_tests, timeout_s=args.timeout_s)
    body = json.dumps(report, indent=2) + "\n"
    if args.out:
        write_text(args.out, body)
    else:
        print(body, end="")
    if args.markdown:
        write_text(args.markdown, markdown(report))
    return 0 if report["summary"]["scope_bounded_verification_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
