#!/usr/bin/env python3
"""Emit a conformance certificate for API hosts covered by the bridge contract."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import fleet_version


SCHEMA = "fak.api-host-conformance-certificate.v1"
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PATHS = {
    "matrix": "fak/experiments/api-host-bridge/api-host-bridge-matrix.json",
    "gate": "fak/experiments/api-host-bridge/api-host-bridge-gate.json",
    "contract": "fak/experiments/api-host-bridge/api-host-compat-contract.json",
    "acceptance": "fak/experiments/api-host-bridge/api-host-acceptance.json",
    "roster": "fak/experiments/api-host-bridge/api-host-roster.json",
    "external_state": "fak/experiments/api-host-bridge/api-host-external-state-audit.json",
}

EXPECTED_ARTIFACT_SCHEMAS = {
    "matrix": "fak.api-host-bridge-matrix.v1",
    "gate": "fak.api-host-bridge-gate.v1",
    "contract": "fak.api-host-compat-contract.v1",
    "acceptance": "fak.api-host-acceptance.v1",
    "roster": "fak.api-host-roster.v1",
    "external_state": "fak.api-host-external-state-audit.v1",
}

REQUIRED_PROVIDER_SHAPES = ["anthropic", "gemini", "openai-compatible", "xai"]
REQUIRED_NON_CLAIMS = {
    "arbitrary_api_host_without_compatible_wire",
    "streaming_chat_completions_delta_passthrough",
    "paid_or_keyed_live_execution_without_credentials",
    "provider_semantics_beyond_tool_wire",
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


def witness_status(matrix: dict[str, Any] | None, id: str) -> str:
    if matrix is None:
        return "missing_matrix"
    for item in matrix.get("witnesses", []):
        if isinstance(item, dict) and item.get("id") == id:
            return str(item.get("status"))
    return "missing_witness"


def run_status(gate: dict[str, Any] | None, id: str) -> str:
    if gate is None:
        return "missing_gate"
    for item in gate.get("runs", []):
        if isinstance(item, dict) and item.get("id") == id:
            return str(item.get("status"))
    return "missing_run"


def host_class(contract: dict[str, Any] | None, id: str) -> dict[str, Any]:
    if contract is None:
        return {}
    for item in contract.get("host_classes", []):
        if isinstance(item, dict) and item.get("id") == id:
            return item
    return {}


def non_claim_ids(contract: dict[str, Any] | None) -> set[str]:
    if contract is None:
        return set()
    return {
        str(item.get("id"))
        for item in contract.get("non_claims", [])
        if isinstance(item, dict) and item.get("id")
    }


def capability(id: str, claim: str, checks: dict[str, bool], evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": fleet_version.app_version(),
        "id": id,
        "claim": claim,
        "status": "PROVEN" if all(checks.values()) else "FAILED",
        "checks": checks,
        "evidence": evidence,
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
            expected_schema = EXPECTED_ARTIFACT_SCHEMAS[key]
            if data.get("schema") != expected_schema:
                errors[f"{key}_schema"] = f"{key} artifact schema is not {expected_schema}"

    matrix = loaded["matrix"]
    gate = loaded["gate"]
    contract = loaded["contract"]
    acceptance = loaded["acceptance"]
    roster = loaded["roster"]
    external_state = loaded["external_state"]

    matrix_summary = (matrix or {}).get("summary", {})
    gate_summary = (gate or {}).get("summary", {})
    contract_summary = (contract or {}).get("summary", {})
    acceptance_summary = (acceptance or {}).get("summary", {})
    roster_summary = (roster or {}).get("summary", {})
    external_summary = (external_state or {}).get("summary", {})
    provider_shapes = list(matrix_summary.get("provider_shapes_covered") or [])
    non_claims = non_claim_ids(contract)

    capabilities = [
        capability(
            "openai_compatible_host_conformance",
            "Any host presenting the OpenAI-compatible chat-completions wire can sit behind FAK under compatible aliases, arbitrary base paths, opaque model ids, optional auth, ignored vendor extension fields, and stream=true client responses synthesized after adjudication.",
            {
                "source_witness_resolved": witness_status(matrix, "host_agnostic_openai_compatible") == "resolved",
                "witness_command_passed": run_status(gate, "host_agnostic_openai_compatible") == "passed",
                "host_profile_witness_passed": run_status(gate, "openai_compatible_host_profiles") == "passed",
                "contract_class_proven": host_class(contract, "openai_compatible_upstream").get("status") == "PROVEN",
            },
            {
                "matrix_witness": "host_agnostic_openai_compatible",
                "gate_run": "host_agnostic_openai_compatible",
                "profile_gate_run": "openai_compatible_host_profiles",
                "contract_class": "openai_compatible_upstream",
            },
        ),
        capability(
            "openai_compatible_host_profile_corpus",
            "The OpenAI-compatible bridge is checked against host-profile drift: null arguments, legacy function_call, typed content parts, extra fields, omitted tool_choice without advertised tools, rogue proposed tool calls, multichoice responses, and content-only responses.",
            {
                "source_witness_resolved": witness_status(matrix, "openai_compatible_host_profiles") == "resolved",
                "witness_command_passed": run_status(gate, "openai_compatible_host_profiles") == "passed",
                "contract_class_proven": host_class(contract, "openai_compatible_upstream").get("status") == "PROVEN",
            },
            {
                "matrix_witness": "openai_compatible_host_profiles",
                "gate_run": "openai_compatible_host_profiles",
                "contract_class": "openai_compatible_upstream",
            },
        ),
        capability(
            "openai_client_proxy_boundary",
            "OpenAI-compatible clients see only FAK-admitted proposed tool calls after deny filtering and transform repair.",
            {
                "gateway_witness_passed": run_status(gate, "openai_compatible_gateway") == "passed",
                "provider_proxy_witness_passed": run_status(gate, "provider_proxy_end_to_end") == "passed",
                "contract_class_proven": host_class(contract, "openai_compatible_upstream").get("status") == "PROVEN",
            },
            {
                "gate_runs": ["openai_compatible_gateway", "provider_proxy_end_to_end"],
                "contract_class": "openai_compatible_upstream",
            },
        ),
        capability(
            "native_provider_transcript_wires",
            "Covered native provider transcript wires preserve pre-send quarantine and parse provider-specific tool-call shapes.",
            {
                "provider_shapes_complete": sorted(provider_shapes) == REQUIRED_PROVIDER_SHAPES,
                "native_adapter_witness_passed": run_status(gate, "native_provider_adapters") == "passed",
                "provider_proxy_witness_passed": run_status(gate, "provider_proxy_end_to_end") == "passed",
                "contract_class_proven": host_class(contract, "native_provider_transcript_adapters").get("status") == "PROVEN",
            },
            {
                "required_provider_shapes": REQUIRED_PROVIDER_SHAPES,
                "provider_shapes": provider_shapes,
                "gate_runs": ["native_provider_adapters", "provider_proxy_end_to_end"],
                "contract_class": "native_provider_transcript_adapters",
            },
        ),
        capability(
            "direct_http_syscall_boundary",
            "Any-language clients can bypass provider quirks and call the FAK/DOS kernel boundary over native HTTP.",
            {
                "witness_command_passed": run_status(gate, "direct_http_syscall") == "passed",
                "contract_class_proven": host_class(contract, "direct_kernel_http_syscall").get("status") == "PROVEN",
            },
            {"gate_run": "direct_http_syscall", "contract_class": "direct_kernel_http_syscall"},
        ),
        capability(
            "direct_mcp_syscall_boundary",
            "Any-language clients can call the same FAK/DOS kernel boundary over MCP stdio or MCP-over-HTTP.",
            {
                "witness_command_passed": run_status(gate, "direct_mcp_syscall") == "passed",
                "contract_class_proven": host_class(contract, "direct_kernel_mcp_syscall").get("status") == "PROVEN",
            },
            {"gate_run": "direct_mcp_syscall", "contract_class": "direct_kernel_mcp_syscall"},
        ),
        capability(
            "expanded_candidate_host_roster",
            "A broad no-spend roster of API-host target templates is syntactically valid, maps to supported bridge wire classes, and carries exact readiness/acceptance rerun commands.",
            {
                "schema_ok": (roster or {}).get("schema") == "fak.api-host-roster.v1",
                "roster_gate": roster_summary.get("roster_gate") is True,
                "many_targets": roster_summary.get("targets", 0) >= 10,
                "mostly_openai_compatible": roster_summary.get("openai_compatible_templates", 0) >= 10,
                "no_invalid_targets": roster_summary.get("invalid_targets") == 0,
                "no_unsupported_wire": roster_summary.get("unsupported_wire") == 0,
                "no_duplicate_names": roster_summary.get("duplicate_names") == 0,
            },
            {"roster_summary": roster_summary},
        ),
        capability(
            "candidate_host_acceptance",
            "Candidate API hosts are classified into ready, live-confirmed, typed external blocker, supported-unprobed, or unsupported-wire states without unclassified drift.",
            {
                "schema_ok": (acceptance or {}).get("schema") == "fak.api-host-acceptance.v1",
                "acceptance_gate": acceptance_summary.get("acceptance_gate") is True,
                "no_unsupported_wire": acceptance_summary.get("unsupported_wire") == 0,
                "no_invalid_targets": acceptance_summary.get("invalid_targets") == 0,
                "no_unclassified": acceptance_summary.get("unclassified") == 0,
                "no_sweep_artifact_errors": acceptance_summary.get("sweep_artifact_errors") == 0,
            },
            {"acceptance_summary": acceptance_summary},
        ),
        capability(
            "live_scoped_host_evidence",
            "Committed live artifacts prove at least one real frontier OpenAI-compatible path and local OpenAI-compatible shims, while external auth/billing failures are typed.",
            {
                "contract_class_proven": host_class(contract, "live_scoped_host_evidence").get("status") == "PROVEN",
                "contract_gate": contract_summary.get("contract_gate") is True,
            },
            {"contract_class": "live_scoped_host_evidence", "contract_summary": contract_summary},
        ),
        capability(
            "external_state_residual_audit",
            "Credential, billing, readiness, and retry state for roster targets is machine-audited without treating external state as solved.",
            {
                "schema_ok": (external_state or {}).get("schema") == "fak.api-host-external-state-audit.v1",
                "external_state_audit_gate": external_summary.get("external_state_audit_gate") is True,
                "covers_roster": external_summary.get("roster_targets", 0) >= roster_summary.get("targets", 0) >= 1,
                "no_artifact_errors": external_summary.get("artifact_errors") == 0,
                "no_unclassified": external_summary.get("unclassified") == 0,
                "no_invalid_templates": external_summary.get("invalid_templates") == 0,
                "no_unsupported_templates": external_summary.get("unsupported_templates") == 0,
            },
            {"external_state_summary": external_summary},
        ),
    ]

    failed = [item for item in capabilities if item["status"] != "PROVEN"]
    missing_non_claims = sorted(REQUIRED_NON_CLAIMS - non_claims)
    certificate_gate = (
        not errors
        and not failed
        and matrix_summary.get("bridge_covered") is True
        and gate_summary.get("executable_bridge_covered") is True
        and contract_summary.get("contract_gate") is True
        and not missing_non_claims
    )

    return {
        "schema": SCHEMA,
        "app_version": app_ver,
        "generated_at": utc_now(),
        "certificate": "Any API host is covered when it satisfies one of the proven compatible host classes; hosts outside those wire contracts are explicit non-claims.",
        "summary": {
            "capabilities": len(capabilities),
            "proven_capabilities": len(capabilities) - len(failed),
            "failed_capabilities": len(failed),
            "missing_required_non_claims": len(missing_non_claims),
            "artifact_errors": len(errors),
            "certificate_gate": certificate_gate,
        },
        "qualification_rules": [
            {
                "version": app_ver,
                "id": "openai_compatible_wire",
                "status": "SUPPORTED",
                "rule": "Host exposes an OpenAI-compatible chat-completions wire; model id, base path, auth, and vendor extension fields may vary within that wire contract. Downstream stream=true is supported by emitting synthesized chunks after full adjudication, not by passing through raw upstream deltas.",
            },
            {
                "version": app_ver,
                "id": "covered_native_provider_wire",
                "status": "SUPPORTED",
                "rule": "Host uses one of the covered native provider transcript wires: anthropic, gemini, openai-compatible, or xai.",
            },
            {
                "version": app_ver,
                "id": "direct_kernel_wire",
                "status": "SUPPORTED",
                "rule": "Client calls the FAK/DOS boundary directly over HTTP or MCP.",
            },
            {
                "version": app_ver,
                "id": "unknown_wire",
                "status": "OUT_OF_CONTRACT",
                "rule": "A host with no compatible wire is not covered until a transcript adapter or direct syscall integration is added.",
            },
            {
                "version": app_ver,
                "id": "paid_or_keyed_remote_state",
                "status": "EXTERNAL_STATE",
                "rule": "Billing, credentials, rate limits, and edge-access restrictions must be resolved before live remote smoke runs can prove that host instance.",
            },
        ],
        "capabilities": capabilities,
        "non_claims": [
            fleet_version.versioned(item, app_ver)
            for item in (contract or {}).get("non_claims", [])
            if isinstance(item, dict)
        ],
        "missing_required_non_claims": missing_non_claims,
        "artifact_errors": errors,
        "artifacts": paths,
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# API-Host Conformance Certificate",
        "",
        report["certificate"],
        "",
        "## Summary",
        "",
        f"- Capabilities proven: {s['proven_capabilities']}/{s['capabilities']}",
        f"- Failed capabilities: {s['failed_capabilities']}",
        f"- Missing required non-claims: {s['missing_required_non_claims']}",
        f"- Artifact errors: {s['artifact_errors']}",
        f"- Certificate gate: {'yes' if s['certificate_gate'] else 'no'}",
        "",
        "## Capabilities",
        "",
        "| capability | status | claim |",
        "|---|---|---|",
    ]
    for item in report["capabilities"]:
        lines.append(f"| `{item['id']}` | {item['status']} | {item['claim']} |")
    lines += [
        "",
        "## Qualification Rules",
        "",
        "| rule | status | detail |",
        "|---|---|---|",
    ]
    for item in report["qualification_rules"]:
        lines.append(f"| `{item['id']}` | {item['status']} | {item['rule']} |")
    lines.append("")
    return "\n".join(lines)


def write_text(path: str, body: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Emit an API-host conformance certificate")
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
    return 0 if report["summary"]["certificate_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
