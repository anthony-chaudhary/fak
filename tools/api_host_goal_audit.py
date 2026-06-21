#!/usr/bin/env python3
"""Audit the API-host bridge work against the original operator objective."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import fleet_version


SCHEMA = "fak.api-host-goal-audit.v1"
MATRIX_SCHEMA = "fak.api-host-bridge-matrix.v1"
GATE_SCHEMA = "fak.api-host-bridge-gate.v1"
HOST_AGNOSTIC_ID = "host_agnostic_openai_compatible"
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PATHS = {
    "matrix": "fak/experiments/api-host-bridge/api-host-bridge-matrix.json",
    "gate": "fak/experiments/api-host-bridge/api-host-bridge-gate.json",
    "proof": "fak/experiments/api-host-bridge/api-host-bridge-proof.json",
    "benchmark": "fak/experiments/permission-systems/permission-system-benchmark.json",
    "acceptance": "fak/experiments/api-host-bridge/api-host-acceptance.json",
    "roster": "fak/experiments/api-host-bridge/api-host-roster.json",
    "retry": "fak/experiments/api-host-bridge/api-host-retry-packet.json",
    "external_state": "fak/experiments/api-host-bridge/api-host-external-state-audit.json",
    "certificate": "fak/experiments/api-host-bridge/api-host-conformance-certificate.json",
    "qualification": "fak/experiments/api-host-bridge/api-host-qualification.json",
    "live_queue": "fak/experiments/api-host-bridge/api-host-live-smoke-queue.json",
    "live_runner": "fak/experiments/api-host-bridge/api-host-live-smoke-runner.json",
}

EXPECTED_ARTIFACT_SCHEMAS = {
    "matrix": MATRIX_SCHEMA,
    "gate": GATE_SCHEMA,
    "proof": "fak.api-host-bridge-proof.v1",
    "benchmark": "fak.permission-system-benchmark.v1",
    "acceptance": "fak.api-host-acceptance.v1",
    "roster": "fak.api-host-roster.v1",
    "retry": "fak.api-host-retry-packet.v1",
    "external_state": "fak.api-host-external-state-audit.v1",
    "certificate": "fak.api-host-conformance-certificate.v1",
    "qualification": "fak.api-host-qualification.v1",
    "live_queue": "fak.api-host-live-smoke-queue.v1",
    "live_runner": "fak.api-host-live-smoke-runner.v1",
}

OBJECTIVE = (
    "Hone in on the initial 'works with any API host' bridge, benchmark it against "
    "Claude Code auto permissions and other permission systems, and work toward "
    "DOS-style completion/proof that it is working."
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


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


def proof_req(proof: dict[str, Any] | None, id: str) -> dict[str, Any]:
    if proof is None:
        return {}
    for row in proof.get("requirements", []):
        if isinstance(row, dict) and row.get("id") == id:
            return row
    return {}


def proof_residual(proof: dict[str, Any] | None, id: str) -> dict[str, Any]:
    if proof is None:
        return {}
    for row in proof.get("residual_scope", []):
        if isinstance(row, dict) and row.get("id") == id:
            return row
    return {}


def metric(benchmark: dict[str, Any] | None, system: str) -> dict[str, Any]:
    if benchmark is None:
        return {}
    for row in benchmark.get("metrics", []):
        if isinstance(row, dict) and row.get("system") == system:
            return row
    return {}


def witness(matrix: dict[str, Any] | None, id: str) -> dict[str, Any]:
    if matrix is None:
        return {}
    for row in matrix.get("witnesses", []):
        if isinstance(row, dict) and row.get("id") == id:
            return row
    return {}


def run(gate: dict[str, Any] | None, id: str) -> dict[str, Any]:
    if gate is None:
        return {}
    for row in gate.get("runs", []):
        if isinstance(row, dict) and row.get("id") == id:
            return row
    return {}


def host_agnostic_detail(matrix: dict[str, Any] | None, gate: dict[str, Any] | None) -> tuple[bool, dict[str, Any]]:
    matrix = matrix or {}
    gate = gate or {}
    host_witness = witness(matrix, HOST_AGNOSTIC_ID)
    host_run = run(gate, HOST_AGNOSTIC_ID)
    evidence = host_witness.get("evidence")
    providers = host_witness.get("providers")
    matrix_argv = host_witness.get("argv")
    gate_argv = host_run.get("argv")

    matrix_schema_ok = matrix.get("schema") == MATRIX_SCHEMA
    gate_schema_ok = gate.get("schema") == GATE_SCHEMA and gate.get("matrix_schema") == MATRIX_SCHEMA
    command_matches = isinstance(host_witness.get("command"), str) and host_witness.get("command") == host_run.get("command")
    cwd_matches = isinstance(host_witness.get("cwd"), str) and host_witness.get("cwd") == host_run.get("cwd")
    argv_matches = isinstance(matrix_argv, list) and bool(matrix_argv) and matrix_argv == gate_argv
    required_matches = host_witness.get("required") is True and host_run.get("required") is True
    provider_covered = isinstance(providers, list) and "openai-compatible" in providers
    evidence_resolved = (
        isinstance(evidence, list)
        and bool(evidence)
        and all(isinstance(item, dict) and item.get("status") == "resolved" for item in evidence)
    )
    status_ok = host_witness.get("status") == "resolved" and host_run.get("status") == "passed"
    proven = all([
        matrix_schema_ok,
        gate_schema_ok,
        status_ok,
        required_matches,
        command_matches,
        cwd_matches,
        argv_matches,
        provider_covered,
        evidence_resolved,
    ])
    return proven, {
        "matrix_schema": matrix.get("schema"),
        "gate_schema": gate.get("schema"),
        "gate_matrix_schema": gate.get("matrix_schema"),
        "matrix_witness": host_witness.get("status", "missing"),
        "gate_run": host_run.get("status", "missing"),
        "required_matches": required_matches,
        "command_matches": command_matches,
        "cwd_matches": cwd_matches,
        "argv_matches": argv_matches,
        "provider_covered": provider_covered,
        "evidence_resolved": evidence_resolved,
        "command": host_run.get("command") or host_witness.get("command"),
        "elapsed_ms": host_run.get("elapsed_ms"),
    }


def audit_row(id: str, requirement: str, status: str, evidence: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "version": fleet_version.app_version(),
        "id": id,
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "detail": detail or {},
    }


def build_report(root: Path | None = None, paths: dict[str, str] | None = None) -> dict[str, Any]:
    root = root or ROOT
    paths = paths or DEFAULT_PATHS
    app_ver = fleet_version.app_version(root)
    loaded: dict[str, dict[str, Any] | None] = {}
    errors: dict[str, str] = {}
    for key, rel in paths.items():
        data, err = load_json(root, rel)
        loaded[key] = data
        if err:
            errors[key] = err
        elif data is not None:
            expected_schema = EXPECTED_ARTIFACT_SCHEMAS.get(key)
            if expected_schema and data.get("schema") != expected_schema:
                errors[f"{key}_schema"] = f"{key} artifact schema is not {expected_schema}"

    matrix = loaded["matrix"]
    gate = loaded["gate"]
    proof = loaded["proof"]
    benchmark = loaded["benchmark"]
    acceptance = loaded["acceptance"]
    roster = loaded["roster"]
    retry = loaded["retry"]
    external_state = loaded["external_state"]
    certificate = loaded["certificate"]
    qualification = loaded["qualification"]
    live_queue = loaded["live_queue"]
    live_runner = loaded["live_runner"]

    rows: list[dict[str, Any]] = []
    proof_summary = (proof or {}).get("summary", {})
    proof_gate = proof_summary.get("proof_gate") is True

    bridge_reqs = ["source_witness_matrix", "executed_witness_gate", "host_agnostic_conformance", "compatibility_contract", "api_host_conformance_certificate", "committed_live_inventory"]
    bridge_ok = proof_gate and all(proof_req(proof, req).get("status") == "PROVEN" for req in bridge_reqs)
    rows.append(audit_row(
        "compatible_api_host_bridge",
        "A compatible API host can sit behind the model boundary while FAK/DOS owns tool-call admission.",
        "PROVEN" if bridge_ok else "INCOMPLETE",
        paths["proof"],
        {"required_rows": bridge_reqs},
    ))

    acceptance_req = proof_req(proof, "candidate_host_acceptance")
    acceptance_summary = (acceptance or {}).get("summary", {})
    acceptance_artifact_errors = (acceptance or {}).get("artifact_errors")
    acceptance_ok = (
        acceptance_req.get("status") == "PROVEN"
        and (acceptance or {}).get("schema") == "fak.api-host-acceptance.v1"
        and acceptance_summary.get("acceptance_gate") is True
        and acceptance_summary.get("unclassified") == 0
        and acceptance_summary.get("unsupported_wire") == 0
        and acceptance_summary.get("invalid_targets") == 0
        and acceptance_summary.get("sweep_artifact_errors") == 0
        and isinstance(acceptance_artifact_errors, list)
        and not acceptance_artifact_errors
    )
    rows.append(audit_row(
        "candidate_host_workflow",
        "An operator can point the bridge at candidate hosts and get ready/live-confirmed/typed-blocker/unsupported-wire states.",
        "PROVEN" if acceptance_ok else "INCOMPLETE",
        paths["acceptance"],
        {"summary": acceptance_summary, "artifact_errors": acceptance_artifact_errors},
    ))

    roster_summary = (roster or {}).get("summary", {})
    roster_ok = (
        proof_req(proof, "api_host_roster").get("status") == "PROVEN"
        and (roster or {}).get("schema") == "fak.api-host-roster.v1"
        and roster_summary.get("roster_gate") is True
        and roster_summary.get("targets", 0) >= 10
        and roster_summary.get("invalid_targets") == 0
        and roster_summary.get("unsupported_wire") == 0
    )
    rows.append(audit_row(
        "expanded_candidate_host_roster",
        "A broad no-spend roster of API-host target templates is valid and ready for credentialed readiness/acceptance probing.",
        "PROVEN" if roster_ok else "INCOMPLETE",
        paths["roster"],
        roster_summary,
    ))

    host_agnostic_ok, host_detail = host_agnostic_detail(matrix, gate)
    rows.append(audit_row(
        "host_agnostic_compatible_api_host",
        "Host-agnostic OpenAI-compatible conformance cases cover compatible aliases, arbitrary base paths, opaque model ids, optional auth, ignored vendor extension fields, and stream=true client requests whose chunks are synthesized only after adjudication.",
        "PROVEN" if host_agnostic_ok else "INCOMPLETE",
        paths["gate"],
        host_detail,
    ))

    profile_req = proof_req(proof, "host_profile_conformance")
    rows.append(audit_row(
        "openai_compatible_host_profile_corpus",
        "OpenAI-compatible host profile drift is covered across null arguments, legacy function_call, typed content parts, extra fields, omitted tool_choice without advertised tools, rogue tool calls, multichoice responses, and content-only replies.",
        "PROVEN" if profile_req.get("status") == "PROVEN" else "INCOMPLETE",
        paths["proof"],
        profile_req.get("detail", {}),
    ))

    cert_summary = (certificate or {}).get("summary", {})
    certificate_ok = (
        proof_req(proof, "api_host_conformance_certificate").get("status") == "PROVEN"
        and (certificate or {}).get("schema") == "fak.api-host-conformance-certificate.v1"
        and cert_summary.get("certificate_gate") is True
        and cert_summary.get("proven_capabilities") == cert_summary.get("capabilities")
        and cert_summary.get("failed_capabilities") == 0
    )
    rows.append(audit_row(
        "api_host_conformance_certificate",
        "The compatible-host claim is captured in a machine-readable certificate with proven capabilities and explicit non-claims.",
        "PROVEN" if certificate_ok else "INCOMPLETE",
        paths["certificate"],
        cert_summary,
    ))

    retry_summary = (retry or {}).get("summary", {})
    retry_ok = (
        (retry or {}).get("schema") == "fak.api-host-retry-packet.v1"
        and retry_summary.get("retry_packet_gate") is True
        and retry_summary.get("action_gaps") == 0
        and retry_summary.get("artifact_errors") == 0
        and retry_summary.get("unclassified") == 0
        and retry_summary.get("unsupported_wire") == 0
        and retry_summary.get("invalid_targets") == 0
        and retry_summary.get("shape_mismatch") == 0
        and retry_summary.get("actionable_blockers") == acceptance_summary.get("typed_external_blockers")
    )
    rows.append(audit_row(
        "blocked_host_retry_packet",
        "Typed external host blockers have exact operator prerequisites and rerun commands.",
        "PROVEN" if retry_ok else "INCOMPLETE",
        paths["retry"],
        retry_summary,
    ))

    external_summary = (external_state or {}).get("summary", {})
    external_ok = (
        proof_req(proof, "api_host_external_state_audit").get("status") == "PROVEN"
        and (external_state or {}).get("schema") == "fak.api-host-external-state-audit.v1"
        and external_summary.get("external_state_audit_gate") is True
        and external_summary.get("artifact_errors") == 0
        and external_summary.get("unclassified") == 0
        and external_summary.get("invalid_templates") == 0
        and external_summary.get("unsupported_templates") == 0
        and external_summary.get("roster_targets", 0) >= roster_summary.get("targets", 0) >= 1
    )
    rows.append(audit_row(
        "paid_keyed_external_state_audit",
        "Paid/keyed and no-auth roster targets have typed current external state and next evidence commands without exposing secret values.",
        "PROVEN" if external_ok else "INCOMPLETE",
        paths["external_state"],
        external_summary,
    ))

    qualification_summary = (qualification or {}).get("summary", {})
    qualification_ok = (
        proof_req(proof, "api_host_qualification").get("status") == "PROVEN"
        and (qualification or {}).get("schema") == "fak.api-host-qualification.v1"
        and qualification_summary.get("qualification_gate") is True
        and qualification_summary.get("targets", 0) >= roster_summary.get("targets", 0) >= 1
        and qualification_summary.get("in_contract_targets") == qualification_summary.get("targets")
        and qualification_summary.get("artifact_errors") == 0
        and qualification_summary.get("unclassified") == 0
        and qualification_summary.get("out_of_contract") == 0
        and qualification_summary.get("invalid_targets") == 0
    )
    rows.append(audit_row(
        "api_host_qualification_predicate",
        "Each roster target is reduced to a machine-readable bridge qualification: live, ready for smoke, typed external blocker, credential-needed, or probe-needed.",
        "PROVEN" if qualification_ok else "INCOMPLETE",
        paths["qualification"],
        qualification_summary,
    ))

    live_queue_summary = (live_queue or {}).get("summary", {})
    queue_state_total = sum(int(live_queue_summary.get(key, 0) or 0) for key in [
        "complete",
        "ready_to_execute",
        "blocked_external_state",
        "waiting_for_credential",
        "ready_for_probe",
    ])
    live_queue_ok = (
        proof_req(proof, "api_host_live_smoke_queue").get("status") == "PROVEN"
        and (live_queue or {}).get("schema") == "fak.api-host-live-smoke-queue.v1"
        and live_queue_summary.get("live_smoke_queue_gate") is True
        and live_queue_summary.get("targets", 0) >= roster_summary.get("targets", 0) >= 1
        and queue_state_total == live_queue_summary.get("targets")
        and live_queue_summary.get("artifact_errors") == 0
        and live_queue_summary.get("unqualified") == 0
        and live_queue_summary.get("unclassified") == 0
        and live_queue_summary.get("command_gaps") == 0
    )
    rows.append(audit_row(
        "paid_keyed_live_execution_queue",
        "Remaining paid/keyed live-smoke work is reduced to a credential-conditioned execution queue with exact commands and typed prerequisites.",
        "PROVEN" if live_queue_ok else "INCOMPLETE",
        paths["live_queue"],
        {"summary": live_queue_summary, "state_total": queue_state_total},
    ))

    live_runner_summary = (live_runner or {}).get("summary", {})
    live_runner_ok = (
        proof_req(proof, "api_host_live_smoke_runner").get("status") == "PROVEN"
        and (live_runner or {}).get("schema") == "fak.api-host-live-smoke-runner.v1"
        and live_runner_summary.get("live_smoke_runner_gate") is True
        and live_runner_summary.get("targets", 0) >= roster_summary.get("targets", 0) >= 1
        and live_runner_summary.get("artifact_errors") == 0
        and live_runner_summary.get("unclassified") == 0
        and live_runner_summary.get("ready_execution_gaps") == 0
        and live_runner_summary.get("failed") == 0
    )
    rows.append(audit_row(
        "paid_keyed_live_runner_gate",
        "Ready paid/keyed live-smoke rows are executed by a runner or the proof fails closed; currently blocked rows remain typed skips.",
        "PROVEN" if live_runner_ok else "INCOMPLETE",
        paths["live_runner"],
        live_runner_summary,
    ))

    systems = ["fak_dos_gateway", "claude_code_auto", "codex_workspace_sandbox", "github_copilot_cloud_agent", "manual_prompts", "bypass_permissions"]
    metrics = {system: metric(benchmark, system) for system in systems}
    benchmark_ok = (
        proof_req(proof, "permission_system_benchmark").get("status") == "PROVEN"
        and proof_req(proof, "permission_source_audit").get("status") == "PROVEN"
        and (benchmark or {}).get("schema") == "fak.permission-system-benchmark.v1"
        and all(metrics[system] for system in systems)
        and metrics["fak_dos_gateway"].get("deterministic_controls") == metrics["fak_dos_gateway"].get("risk_scenarios")
        and metrics["fak_dos_gateway"].get("api_host_bridge_controls") == metrics["fak_dos_gateway"].get("api_host_bridge_dimensions")
        and metrics["fak_dos_gateway"].get("api_host_result_quarantine_verdict") == "QUARANTINE"
        and metrics["claude_code_auto"].get("api_host_bridge_controls") == 0
        and metrics["claude_code_auto"].get("known_max_false_negative_pct") == 17.0
    )
    rows.append(audit_row(
        "permission_system_benchmark",
        "FAK/DOS risk controls and API-host bridge controls are benchmarked against Claude auto, Codex, Copilot, prompts, and bypass using source-audited claims.",
        "PROVEN" if benchmark_ok else "INCOMPLETE",
        paths["benchmark"],
        {"systems": list(metrics), "source_audit": proof_req(proof, "permission_source_audit").get("status")},
    ))

    dos_rows = ["source_witness_matrix", "executed_witness_gate", "host_agnostic_conformance", "host_profile_conformance", "api_host_conformance_certificate", "api_host_roster", "api_host_external_state_audit", "api_host_qualification", "api_host_live_smoke_queue", "api_host_live_smoke_runner", "committed_live_inventory", "candidate_host_acceptance", "permission_source_audit"]
    dos_ok = proof_gate and all(proof_req(proof, req).get("status") == "PROVEN" for req in dos_rows)
    rows.append(audit_row(
        "dos_style_proof",
        "The bridge claim is backed by source witnesses, executable gates, live inventory, candidate-host acceptance, and source-audited benchmark evidence.",
        "PROVEN" if dos_ok else "INCOMPLETE",
        paths["proof"],
        {"proof_rows": dos_rows, "proof_summary": proof_summary},
    ))

    universal = proof_residual(proof, "universal_any_api_host")
    rows.append(audit_row(
        "universal_any_api_host",
        "Every API host on the internet works with the bridge.",
        universal.get("status", "NOT_PROVEN"),
        paths["proof"],
        {"reason": universal.get("reason", "No universal-host evidence present.")},
    ))

    external = proof_residual(proof, "blocked_paid_or_keyed_hosts")
    rows.append(audit_row(
        "paid_or_keyed_live_hosts",
        "Paid/keyed hosts have completed live tool-calling runs.",
        external.get("status", "EXTERNAL_STATE"),
        paths["proof"],
        {"reason": external.get("reason", "External host state not resolved.")},
    ))

    if errors:
        rows.append(audit_row(
            "artifact_integrity",
            "All goal-audit input artifacts are readable JSON objects with expected schemas.",
            "INCOMPLETE",
            "",
            errors,
        ))

    proven = [row for row in rows if row["status"] == "PROVEN"]
    residual = [row for row in rows if row["status"] in {"NOT_PROVEN", "EXTERNAL_STATE"}]
    incomplete = [row for row in rows if row["status"] not in {"PROVEN", "NOT_PROVEN", "EXTERNAL_STATE"}]
    for row in rows:
        row["version"] = app_ver
    return {
        "schema": SCHEMA,
        "app_version": app_ver,
        "generated_at": utc_now(),
        "objective": OBJECTIVE,
        "summary": {
            "requirements": len(rows),
            "proven": len(proven),
            "residual": len(residual),
            "incomplete": len(incomplete),
            "goal_complete": len(rows) == len(proven),
            "goal_status": "COMPLETE" if len(rows) == len(proven) else "SCOPE_BOUNDED_PROGRESS_NOT_COMPLETE",
        },
        "requirements": rows,
        "artifacts": paths,
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# API-Host Goal Audit",
        "",
        report["objective"],
        "",
        "## Summary",
        "",
        f"- Requirements proven: {s['proven']}/{s['requirements']}",
        f"- Residual requirements: {s['residual']}",
        f"- Incomplete requirements: {s['incomplete']}",
        f"- Goal complete: {'yes' if s['goal_complete'] else 'no'}",
        f"- Goal status: `{s['goal_status']}`",
        "",
        "| requirement | status | evidence |",
        "|---|---|---|",
    ]
    for row in report["requirements"]:
        lines.append(f"| `{row['id']}` | {row['status']} | `{row['evidence']}` |")
    lines.append("")
    return "\n".join(lines)


def write_text(path: str, body: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit API-host bridge work against the operator objective")
    ap.add_argument("--out", default="", help="write JSON report here")
    ap.add_argument("--markdown", default="", help="write Markdown report here")
    ap.add_argument("--root", default=str(ROOT), help="workspace root")
    args = ap.parse_args(argv)

    report = build_report(Path(args.root))
    body = json.dumps(report, indent=2) + "\n"
    if args.out:
        write_text(args.out, body)
    else:
        print(body, end="")
    if args.markdown:
        write_text(args.markdown, markdown(report))
    return 0 if not report["summary"]["incomplete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
