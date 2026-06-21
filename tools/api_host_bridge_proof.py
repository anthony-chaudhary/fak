#!/usr/bin/env python3
"""Roll up API-host bridge proof artifacts into one auditable gate."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import fleet_version


SCHEMA = "fak.api-host-bridge-proof.v1"
MATRIX_SCHEMA = "fak.api-host-bridge-matrix.v1"
GATE_SCHEMA = "fak.api-host-bridge-gate.v1"
HOST_AGNOSTIC_ID = "host_agnostic_openai_compatible"
HOST_PROFILE_ID = "openai_compatible_host_profiles"
ROOT = Path(__file__).resolve().parents[1]


DEFAULT_PATHS = {
    "matrix": "fak/experiments/api-host-bridge/api-host-bridge-matrix.json",
    "gate": "fak/experiments/api-host-bridge/api-host-bridge-gate.json",
    "live": "fak/experiments/api-host-bridge/api-host-live-inventory.json",
    "readiness": "fak/experiments/api-host-bridge/api-host-readiness.json",
    "acceptance": "fak/experiments/api-host-bridge/api-host-acceptance.json",
    "roster": "fak/experiments/api-host-bridge/api-host-roster.json",
    "external_state": "fak/experiments/api-host-bridge/api-host-external-state-audit.json",
    "contract": "fak/experiments/api-host-bridge/api-host-compat-contract.json",
    "certificate": "fak/experiments/api-host-bridge/api-host-conformance-certificate.json",
    "qualification": "fak/experiments/api-host-bridge/api-host-qualification.json",
    "live_queue": "fak/experiments/api-host-bridge/api-host-live-smoke-queue.json",
    "live_runner": "fak/experiments/api-host-bridge/api-host-live-smoke-runner.json",
    "benchmark": "fak/experiments/permission-systems/permission-system-benchmark.json",
    "source_audit": "fak/experiments/permission-systems/permission-source-audit.json",
}

EXPECTED_ARTIFACT_SCHEMAS = {
    "matrix": MATRIX_SCHEMA,
    "gate": GATE_SCHEMA,
    "live": "fak.api-host-live-inventory.v1",
    "readiness": "fak.api-host-readiness.v1",
    "acceptance": "fak.api-host-acceptance.v1",
    "roster": "fak.api-host-roster.v1",
    "external_state": "fak.api-host-external-state-audit.v1",
    "contract": "fak.api-host-compat-contract.v1",
    "certificate": "fak.api-host-conformance-certificate.v1",
    "qualification": "fak.api-host-qualification.v1",
    "live_queue": "fak.api-host-live-smoke-queue.v1",
    "live_runner": "fak.api-host-live-smoke-runner.v1",
    "benchmark": "fak.permission-system-benchmark.v1",
    "source_audit": "fak.permission-source-audit.v1",
}


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


def row(id: str, claim: str, proven: bool, evidence: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "version": fleet_version.app_version(),
        "id": id,
        "claim": claim,
        "status": "PROVEN" if proven else "FAILED",
        "evidence": evidence,
        "detail": detail or {},
    }


def missing_row(id: str, claim: str, evidence: str) -> dict[str, Any]:
    return {"version": fleet_version.app_version(), "id": id, "claim": claim, "status": "MISSING", "evidence": evidence, "detail": {}}


def metric(report: dict[str, Any], system: str) -> dict[str, Any]:
    for item in report.get("metrics", []):
        if item.get("system") == system:
            return item
    return {}


def witness_status(report: dict[str, Any] | None, id: str) -> str:
    return str(witness_row(report, id).get("status", "missing"))


def run_status(report: dict[str, Any] | None, id: str) -> str:
    return str(run_row(report, id).get("status", "missing"))


def witness_row(report: dict[str, Any] | None, id: str) -> dict[str, Any]:
    if report is None:
        return {}
    for item in report.get("witnesses", []):
        if isinstance(item, dict) and item.get("id") == id:
            return item
    return {}


def run_row(report: dict[str, Any] | None, id: str) -> dict[str, Any]:
    if report is None:
        return {}
    for item in report.get("runs", []):
        if isinstance(item, dict) and item.get("id") == id:
            return item
    return {}


def host_agnostic_detail(matrix: dict[str, Any], gate: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    h_witness = witness_row(matrix, HOST_AGNOSTIC_ID)
    h_run = run_row(gate, HOST_AGNOSTIC_ID)
    evidence = h_witness.get("evidence")
    providers = h_witness.get("providers")
    matrix_argv = h_witness.get("argv")
    gate_argv = h_run.get("argv")

    matrix_schema_ok = matrix.get("schema") == MATRIX_SCHEMA
    gate_schema_ok = gate.get("schema") == GATE_SCHEMA and gate.get("matrix_schema") == MATRIX_SCHEMA
    command_matches = isinstance(h_witness.get("command"), str) and h_witness.get("command") == h_run.get("command")
    cwd_matches = isinstance(h_witness.get("cwd"), str) and h_witness.get("cwd") == h_run.get("cwd")
    argv_matches = isinstance(matrix_argv, list) and bool(matrix_argv) and matrix_argv == gate_argv
    required_matches = h_witness.get("required") is True and h_run.get("required") is True
    provider_covered = isinstance(providers, list) and "openai-compatible" in providers
    evidence_resolved = (
        isinstance(evidence, list)
        and bool(evidence)
        and all(isinstance(item, dict) and item.get("status") == "resolved" for item in evidence)
    )
    status_ok = h_witness.get("status") == "resolved" and h_run.get("status") == "passed"
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
        "matrix_witness": h_witness.get("status", "missing"),
        "gate_run": h_run.get("status", "missing"),
        "required_matches": required_matches,
        "command_matches": command_matches,
        "cwd_matches": cwd_matches,
        "argv_matches": argv_matches,
        "provider_covered": provider_covered,
        "evidence_resolved": evidence_resolved,
    }


def required_witness_detail(matrix: dict[str, Any], gate: dict[str, Any], id: str, provider: str = "") -> tuple[bool, dict[str, Any]]:
    h_witness = witness_row(matrix, id)
    h_run = run_row(gate, id)
    evidence = h_witness.get("evidence")
    providers = h_witness.get("providers")
    matrix_argv = h_witness.get("argv")
    gate_argv = h_run.get("argv")
    provider_covered = not provider or (isinstance(providers, list) and provider in providers)
    evidence_resolved = (
        isinstance(evidence, list)
        and bool(evidence)
        and all(isinstance(item, dict) and item.get("status") == "resolved" for item in evidence)
    )
    command_matches = isinstance(h_witness.get("command"), str) and h_witness.get("command") == h_run.get("command")
    cwd_matches = isinstance(h_witness.get("cwd"), str) and h_witness.get("cwd") == h_run.get("cwd")
    argv_matches = isinstance(matrix_argv, list) and bool(matrix_argv) and matrix_argv == gate_argv
    required_matches = h_witness.get("required") is True and h_run.get("required") is True
    proven = all([
        matrix.get("schema") == MATRIX_SCHEMA,
        gate.get("schema") == GATE_SCHEMA and gate.get("matrix_schema") == MATRIX_SCHEMA,
        h_witness.get("status") == "resolved",
        h_run.get("status") == "passed",
        required_matches,
        command_matches,
        cwd_matches,
        argv_matches,
        provider_covered,
        evidence_resolved,
    ])
    return proven, {
        "matrix_witness": h_witness.get("status", "missing"),
        "gate_run": h_run.get("status", "missing"),
        "required_matches": required_matches,
        "command_matches": command_matches,
        "cwd_matches": cwd_matches,
        "argv_matches": argv_matches,
        "provider_covered": provider_covered,
        "evidence_resolved": evidence_resolved,
    }


def build_report(root: Path | None = None, paths: dict[str, str] | None = None) -> dict[str, Any]:
    root = root or ROOT
    app_ver = fleet_version.app_version(root)
    paths = paths or DEFAULT_PATHS
    loaded: dict[str, dict[str, Any] | None] = {}
    errors: dict[str, str] = {}
    for key, rel_path in paths.items():
        data, err = load_json(root, rel_path)
        loaded[key] = data
        if err:
            errors[key] = err
        elif data is not None:
            expected_schema = EXPECTED_ARTIFACT_SCHEMAS.get(key)
            if expected_schema and data.get("schema") != expected_schema:
                errors[f"{key}_schema"] = f"{key} artifact schema is not {expected_schema}"

    requirements: list[dict[str, Any]] = []

    matrix = loaded["matrix"]
    roster = loaded["roster"]
    acceptance = loaded["acceptance"]
    roster_summary = (roster or {}).get("summary", {})
    if matrix is None:
        requirements.append(missing_row("source_witness_matrix", "Required source witnesses resolve.", errors["matrix"]))
    else:
        s = matrix.get("summary", {})
        requirements.append(row(
            "source_witness_matrix",
            "Required source witnesses resolve and provider shapes are enumerated.",
            matrix.get("schema") == MATRIX_SCHEMA
            and bool(s.get("bridge_covered"))
            and s.get("resolved_required_witnesses") == s.get("required_witnesses"),
            paths["matrix"],
            s,
        ))

    gate = loaded["gate"]
    if gate is None:
        requirements.append(missing_row("executed_witness_gate", "Required witness commands pass now.", errors["gate"]))
    else:
        s = gate.get("summary", {})
        requirements.append(row(
            "executed_witness_gate",
            "Required witness commands pass now.",
            gate.get("schema") == GATE_SCHEMA
            and gate.get("matrix_schema") == MATRIX_SCHEMA
            and bool(s.get("executable_bridge_covered"))
            and s.get("passed_required_witnesses") == s.get("required_witnesses"),
            paths["gate"],
            s,
        ))

    if matrix is None or gate is None:
        requirements.append(missing_row(
            "host_agnostic_conformance",
            "Host-agnostic OpenAI-compatible conformance witness passes.",
            errors.get("matrix") or errors.get("gate", ""),
        ))
    else:
        host_proven, host_detail = host_agnostic_detail(matrix, gate)
        requirements.append(row(
            "host_agnostic_conformance",
            "OpenAI-compatible hosts with compatible aliases, arbitrary base paths, opaque model ids, optional auth, vendor extension fields, and stream=true client requests run through the bridge; streamed chunks are synthesized only after full tool-call adjudication.",
            host_proven,
            paths["gate"],
            host_detail,
        ))

    if matrix is None or gate is None:
        requirements.append(missing_row(
            "host_profile_conformance",
            "OpenAI-compatible host-profile drift corpus passes.",
            errors.get("matrix") or errors.get("gate", ""),
        ))
    else:
        profile_proven, profile_detail = required_witness_detail(matrix, gate, HOST_PROFILE_ID, "openai-compatible")
        requirements.append(row(
            "host_profile_conformance",
            "OpenAI-compatible host profiles covering null arguments, legacy function_call, typed content parts, extra fields, omitted tool_choice without advertised tools, rogue tool calls, multichoice responses, and content-only replies preserve the FAK tool boundary.",
            profile_proven,
            paths["gate"],
            profile_detail,
        ))

    live = loaded["live"]
    if live is None:
        requirements.append(missing_row("committed_live_inventory", "Committed live API-host evidence is classified and complete.", errors["live"]))
    else:
        s = live.get("summary", {})
        requirements.append(row(
            "committed_live_inventory",
            "Committed live API-host evidence includes frontier success, local shim success, and typed external blockers.",
            live.get("schema") == "fak.api-host-live-inventory.v1"
            and bool(s.get("live_inventory_gate"))
            and s.get("live_frontier_successes", 0) >= 2
            and s.get("local_openai_compatible_successes", 0) >= 1
            and s.get("incomplete_or_unclassified") == 0,
            paths["live"],
            s,
        ))

    readiness = loaded["readiness"]
    if readiness is None:
        requirements.append(missing_row("current_host_readiness", "Current /models host states are typed.", errors["readiness"]))
    else:
        s = readiness.get("summary", {})
        acceptance_models_confirmed = len([
            item for item in (acceptance or {}).get("targets", [])
            if isinstance(item, dict)
            and (item.get("readiness_status") == "MODELS_CONFIRMED" or item.get("status") in {"READY_FOR_LIVE_BRIDGE_RUN", "LIVE_BRIDGE_CONFIRMED"})
        ])
        requirements.append(row(
            "current_host_readiness",
            "Current /models host states are typed for the roster-driven OpenAI-compatible target set, with at least one compatible host reachable.",
            readiness.get("schema") == "fak.api-host-readiness.v1"
            and bool(s.get("readiness_gate"))
            and s.get("targets", 0) >= roster_summary.get("openai_compatible_templates", 0) >= 1
            and (s.get("models_confirmed", 0) >= 1 or acceptance_models_confirmed >= 1)
            and s.get("unclassified") == 0
            and s.get("invalid_targets") == 0,
            paths["readiness"],
            {"summary": s, "acceptance_models_confirmed": acceptance_models_confirmed},
        ))

    if acceptance is None:
        requirements.append(missing_row("candidate_host_acceptance", "Candidate API hosts are classified against the compatibility contract.", errors["acceptance"]))
    else:
        s = acceptance.get("summary", {})
        targets = acceptance.get("targets", [])
        acceptance_artifact_errors = acceptance.get("artifact_errors")
        requirements.append(row(
            "candidate_host_acceptance",
            "Roster-driven candidate API hosts are classified against the compatibility contract with ready hosts and typed blockers separated.",
            acceptance.get("schema") == "fak.api-host-acceptance.v1"
            and bool(s.get("acceptance_gate"))
            and s.get("targets", 0) >= roster_summary.get("targets", 0) >= 1
            and s.get("sweep_artifact_errors") == 0
            and s.get("invalid_targets") == 0
            and isinstance(acceptance_artifact_errors, list)
            and not acceptance_artifact_errors
            and s.get("unclassified") == 0
            and s.get("unsupported_wire") == 0
            and all(item.get("contract_class") != "unsupported" for item in targets),
            paths["acceptance"],
            {
                "summary": s,
                "artifact_errors": acceptance_artifact_errors,
                "target_statuses": [
                    {
                        "name": item.get("name"),
                        "provider": item.get("provider"),
                        "contract_class": item.get("contract_class"),
                        "status": item.get("status"),
                    }
                    for item in targets
                ],
            },
        ))

    if roster is None:
        requirements.append(missing_row("api_host_roster", "Expanded API-host target roster is valid and supported.", errors["roster"]))
    else:
        s = roster.get("summary", {})
        targets = roster.get("targets", [])
        requirements.append(row(
            "api_host_roster",
            "Expanded API-host target roster is valid, supported by the bridge contract, and has no duplicate target names.",
            roster.get("schema") == "fak.api-host-roster.v1"
            and bool(s.get("roster_gate"))
            and s.get("targets", 0) >= 10
            and s.get("openai_compatible_templates", 0) >= 10
            and s.get("invalid_targets") == 0
            and s.get("unsupported_wire") == 0
            and s.get("duplicate_names") == 0
            and all(item.get("status") == "SUPPORTED_TEMPLATE" for item in targets),
            paths["roster"],
            {"summary": s},
        ))

    external_state = loaded["external_state"]
    if external_state is None:
        requirements.append(missing_row(
            "api_host_external_state_audit",
            "Credential, billing, readiness, and retry state for roster targets is typed.",
            errors["external_state"],
        ))
    else:
        s = external_state.get("summary", {})
        requirements.append(row(
            "api_host_external_state_audit",
            "Credential, billing, readiness, and retry state for roster targets is typed without treating external state as solved.",
            external_state.get("schema") == "fak.api-host-external-state-audit.v1"
            and bool(s.get("external_state_audit_gate"))
            and s.get("roster_targets", 0) >= (roster or {}).get("summary", {}).get("targets", 0) >= 1
            and s.get("artifact_errors") == 0
            and s.get("unclassified") == 0
            and s.get("invalid_templates") == 0
            and s.get("unsupported_templates") == 0,
            paths["external_state"],
            s,
        ))

    contract = loaded["contract"]
    if contract is None:
        requirements.append(missing_row("compatibility_contract", "Compatible API-host classes are explicitly scoped and proven.", errors["contract"]))
    else:
        s = contract.get("summary", {})
        host_classes = contract.get("host_classes", [])
        contract_artifact_errors = contract.get("artifact_errors")
        requirements.append(row(
            "compatibility_contract",
            "Compatible API-host classes are explicitly scoped, proven, and bounded by non-claims.",
            contract.get("schema") == "fak.api-host-compat-contract.v1"
            and bool(s.get("contract_gate"))
            and s.get("proven_host_classes") == s.get("host_classes")
            and s.get("failed_host_classes") == 0
            and isinstance(contract_artifact_errors, dict)
            and not contract_artifact_errors
            and all(item.get("status") == "PROVEN" for item in host_classes),
            paths["contract"],
            {
                "summary": s,
                "host_class_ids": [item.get("id") for item in host_classes],
                "non_claims": contract.get("non_claims", []),
                "artifact_errors": contract_artifact_errors,
            },
        ))

    certificate = loaded["certificate"]
    if certificate is None:
        requirements.append(missing_row("api_host_conformance_certificate", "A conformance certificate bounds and proves the compatible-host claim.", errors["certificate"]))
    else:
        s = certificate.get("summary", {})
        capabilities = certificate.get("capabilities", [])
        requirements.append(row(
            "api_host_conformance_certificate",
            "A conformance certificate states which API hosts are covered, proves the covered capabilities, and records non-claims.",
            certificate.get("schema") == "fak.api-host-conformance-certificate.v1"
            and bool(s.get("certificate_gate"))
            and s.get("proven_capabilities") == s.get("capabilities")
            and s.get("failed_capabilities") == 0
            and s.get("missing_required_non_claims") == 0
            and s.get("artifact_errors") == 0
            and all(item.get("status") == "PROVEN" for item in capabilities),
            paths["certificate"],
            {
                "summary": s,
                "capability_ids": [item.get("id") for item in capabilities],
                "qualification_rules": certificate.get("qualification_rules", []),
                "missing_required_non_claims": certificate.get("missing_required_non_claims", []),
            },
        ))

    qualification = loaded["qualification"]
    if qualification is None:
        requirements.append(missing_row(
            "api_host_qualification",
            "Roster targets qualify against the proven API-host contract or carry typed next evidence.",
            errors["qualification"],
        ))
    else:
        s = qualification.get("summary", {})
        requirements.append(row(
            "api_host_qualification",
            "Roster targets qualify against the proven API-host contract with live, ready, external-blocked, credential-needed, or probe-needed states.",
            qualification.get("schema") == "fak.api-host-qualification.v1"
            and bool(s.get("qualification_gate"))
            and s.get("targets", 0) >= (roster or {}).get("summary", {}).get("targets", 0) >= 1
            and s.get("in_contract_targets") == s.get("targets")
            and s.get("artifact_errors") == 0
            and s.get("unclassified") == 0
            and s.get("out_of_contract") == 0
            and s.get("invalid_targets") == 0,
            paths["qualification"],
            s,
        ))

    live_queue = loaded["live_queue"]
    if live_queue is None:
        requirements.append(missing_row(
            "api_host_live_smoke_queue",
            "Roster targets have a credential-conditioned live-smoke execution queue.",
            errors["live_queue"],
        ))
    else:
        s = live_queue.get("summary", {})
        state_total = sum(int(s.get(key, 0) or 0) for key in [
            "complete",
            "ready_to_execute",
            "blocked_external_state",
            "waiting_for_credential",
            "ready_for_probe",
        ])
        requirements.append(row(
            "api_host_live_smoke_queue",
            "Roster targets have a credential-conditioned live-smoke queue with exact next commands for executable or externally blocked states.",
            live_queue.get("schema") == "fak.api-host-live-smoke-queue.v1"
            and bool(s.get("live_smoke_queue_gate"))
            and s.get("targets", 0) >= (roster or {}).get("summary", {}).get("targets", 0) >= 1
            and state_total == s.get("targets")
            and s.get("artifact_errors") == 0
            and s.get("unqualified") == 0
            and s.get("unclassified") == 0
            and s.get("command_gaps") == 0,
            paths["live_queue"],
            {"summary": s, "state_total": state_total},
        ))

    live_runner = loaded["live_runner"]
    if live_runner is None:
        requirements.append(missing_row(
            "api_host_live_smoke_runner",
            "Ready live-smoke queue rows are executed or fail closed.",
            errors["live_runner"],
        ))
    else:
        s = live_runner.get("summary", {})
        requirements.append(row(
            "api_host_live_smoke_runner",
            "Ready live-smoke queue rows are executed or the proof fails closed; external blockers remain skipped with typed status.",
            live_runner.get("schema") == "fak.api-host-live-smoke-runner.v1"
            and bool(s.get("live_smoke_runner_gate"))
            and s.get("targets", 0) >= (roster or {}).get("summary", {}).get("targets", 0) >= 1
            and s.get("artifact_errors") == 0
            and s.get("unclassified") == 0
            and s.get("ready_execution_gaps") == 0
            and s.get("failed") == 0,
            paths["live_runner"],
            s,
        ))

    benchmark = loaded["benchmark"]
    if benchmark is None:
        requirements.append(missing_row("permission_system_benchmark", "Permission benchmark pins the FAK-vs-Claude contrast.", errors["benchmark"]))
    else:
        fak = metric(benchmark, "fak_dos_gateway")
        claude = metric(benchmark, "claude_code_auto")
        bypass = metric(benchmark, "bypass_permissions")
        dimensions = benchmark.get("api_host_bridge_dimensions", [])
        requirements.append(row(
            "permission_system_benchmark",
            "Permission benchmark pins FAK deterministic coverage, API-host bridge controls, Claude auto classifier risk, and unguarded bypass.",
            benchmark.get("schema") == "fak.permission-system-benchmark.v1"
            and bool(fak)
            and fak.get("deterministic_controls") == fak.get("risk_scenarios")
            and fak.get("result_admission_verdict") == "QUARANTINE"
            and fak.get("has_api_host_bridge") is True
            and fak.get("api_host_bridge_controls") == fak.get("api_host_bridge_dimensions")
            and fak.get("api_host_bridge_dimensions") == len(dimensions) >= 5
            and fak.get("api_host_result_quarantine_verdict") == "QUARANTINE"
            and claude.get("deterministic_controls") == 0
            and claude.get("api_host_bridge_controls") == 0
            and claude.get("known_max_false_negative_pct") == 17.0
            and claude.get("result_admission_verdict") == "WARNING"
            and claude.get("api_host_result_quarantine_verdict") == "WARNING"
            and bypass.get("unguarded_risk_allows") == bypass.get("risk_scenarios"),
            paths["benchmark"],
            {"fak": fak, "claude": claude, "bypass": bypass, "bridge_dimensions": dimensions},
        ))

    source_audit = loaded["source_audit"]
    if source_audit is None:
        requirements.append(missing_row("permission_source_audit", "External permission-system benchmark sources verify current claims.", errors["source_audit"]))
    else:
        s = source_audit.get("summary", {})
        requirements.append(row(
            "permission_source_audit",
            "External permission-system benchmark sources verify current claims.",
            source_audit.get("schema") == "fak.permission-source-audit.v1"
            and bool(s.get("source_audit_gate"))
            and s.get("verified") == s.get("sources")
            and s.get("failed") == 0,
            paths["source_audit"],
            s,
        ))

    if errors:
        requirements.append(row(
            "artifact_integrity",
            "Proof rollup input artifacts are readable JSON objects with expected schemas.",
            False,
            "",
            errors,
        ))

    failed = [r for r in requirements if r["status"] != "PROVEN"]
    residual_scope = [
        {
            "version": app_ver,
            "id": "universal_any_api_host",
            "status": "NOT_PROVEN",
            "reason": "The proof covers compatible host shapes, committed Gemini/OpenAI-compatible live runs, local OpenAI-compatible shims, current typed readiness, and a reusable candidate-host acceptance gate. It does not prove every API host on the internet.",
        },
        {
            "version": app_ver,
            "id": "blocked_paid_or_keyed_hosts",
            "status": "EXTERNAL_STATE",
            "reason": "Additional live tool-calling runs for paid/keyed roster targets require billing/API-key/access state beyond this no-spend gate; the external-state audit, live-smoke queue, and runner ledger record the current typed blockers and exact retry evidence.",
        },
    ]
    return {
        "schema": SCHEMA,
        "app_version": app_ver,
        "generated_at": utc_now(),
        "scope": "scope-bounded API-host bridge proof rollup",
        "summary": {
            "requirements": len(requirements),
            "proven": len(requirements) - len(failed),
            "failed_or_missing": len(failed),
            "proof_gate": len(failed) == 0,
            "completion_scope": "BRIDGE_PROVEN_SCOPE_BOUNDED" if len(failed) == 0 else "INCOMPLETE",
        },
        "requirements": requirements,
        "residual_scope": residual_scope,
        "artifacts": paths,
        "artifact_errors": errors,
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# API-Host Bridge Proof Rollup",
        "",
        "> Requirement-by-requirement proof gate for the API-host bridge evidence.",
        "",
        "## Summary",
        "",
        f"- Requirements proven: {s['proven']}/{s['requirements']}",
        f"- Proof gate: {'yes' if s['proof_gate'] else 'no'}",
        f"- Completion scope: `{s['completion_scope']}`",
        "",
        "## Requirements",
        "",
        "| requirement | status | evidence |",
        "|---|---|---|",
    ]
    for req in report["requirements"]:
        lines.append(f"| `{req['id']}` | {req['status']} | `{req['evidence']}` |")
    lines += [
        "",
        "## Residual Scope",
        "",
        "| item | status | reason |",
        "|---|---|---|",
    ]
    for item in report["residual_scope"]:
        lines.append(f"| `{item['id']}` | {item['status']} | {item['reason']} |")
    lines.append("")
    return "\n".join(lines)


def write_text(path: str, body: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Roll up API-host bridge proof artifacts")
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
    return 0 if report["summary"]["proof_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
