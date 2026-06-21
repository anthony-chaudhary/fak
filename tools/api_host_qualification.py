#!/usr/bin/env python3
"""Qualify API-host targets against the proven bridge contract."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import fleet_version


SCHEMA = "fak.api-host-qualification.v1"
CERTIFICATE_SCHEMA = "fak.api-host-conformance-certificate.v1"
CONTRACT_SCHEMA = "fak.api-host-compat-contract.v1"
EXTERNAL_STATE_SCHEMA = "fak.api-host-external-state-audit.v1"
RETRY_SCHEMA = "fak.api-host-retry-packet.v1"
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PATHS = {
    "certificate": "fak/experiments/api-host-bridge/api-host-conformance-certificate.json",
    "contract": "fak/experiments/api-host-bridge/api-host-compat-contract.json",
    "external_state": "fak/experiments/api-host-bridge/api-host-external-state-audit.json",
    "retry": "fak/experiments/api-host-bridge/api-host-retry-packet.json",
}

CONTRACT_CLASS_RULES = {
    "openai_compatible_upstream": "openai_compatible_wire",
    "native_provider_transcript_adapters": "covered_native_provider_wire",
    "direct_kernel_http_syscall": "direct_kernel_wire",
    "direct_kernel_mcp_syscall": "direct_kernel_wire",
}

EXTERNAL_TO_QUALIFICATION = {
    "LIVE_CONFIRMED": ("IN_CONTRACT_LIVE_CONFIRMED", "LIVE_CONFIRMED"),
    "READY_FOR_LIVE_BRIDGE_RUN": ("IN_CONTRACT_READY_FOR_LIVE_SMOKE", "NEEDS_LIVE_SMOKE"),
    "BLOCKED_BILLING": ("IN_CONTRACT_EXTERNAL_BLOCKER", "NEEDS_OPERATOR_STATE"),
    "BLOCKED_AUTH": ("IN_CONTRACT_EXTERNAL_BLOCKER", "NEEDS_OPERATOR_STATE"),
    "RATE_OR_TRANSPORT_BLOCKED": ("IN_CONTRACT_EXTERNAL_BLOCKER", "NEEDS_OPERATOR_STATE"),
    "NEEDS_CREDENTIAL": ("IN_CONTRACT_NEEDS_CREDENTIAL", "NEEDS_OPERATOR_STATE"),
    "WIRE_SUPPORTED_UNPROBED": ("IN_CONTRACT_NEEDS_PROBE", "NEEDS_READINESS_ACCEPTANCE"),
    "UNPROBED_TEMPLATE": ("IN_CONTRACT_NEEDS_PROBE", "NEEDS_READINESS_ACCEPTANCE"),
    "NO_AUTH_READY_TO_PROBE": ("IN_CONTRACT_NEEDS_PROBE", "NEEDS_READINESS_ACCEPTANCE"),
}

KNOWN_QUALIFICATIONS = {
    "IN_CONTRACT_LIVE_CONFIRMED",
    "IN_CONTRACT_READY_FOR_LIVE_SMOKE",
    "IN_CONTRACT_EXTERNAL_BLOCKER",
    "IN_CONTRACT_NEEDS_CREDENTIAL",
    "IN_CONTRACT_NEEDS_PROBE",
    "OUT_OF_CONTRACT",
    "INVALID_TARGET",
    "UNCLASSIFIED",
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


def capability_status(certificate: dict[str, Any] | None, id: str) -> str:
    if certificate is None:
        return "missing_certificate"
    for item in certificate.get("capabilities", []):
        if isinstance(item, dict) and item.get("id") == id:
            return str(item.get("status"))
    return "missing_capability"


def host_class_status(contract: dict[str, Any] | None, id: str) -> str:
    if contract is None:
        return "missing_contract"
    for item in contract.get("host_classes", []):
        if isinstance(item, dict) and item.get("id") == id:
            return str(item.get("status"))
    return "missing_host_class"


def retry_action(retry: dict[str, Any] | None, name: str, base_url: str) -> dict[str, Any]:
    if retry is None:
        return {}
    actions = retry.get("actions", [])
    if not isinstance(actions, list):
        return {}
    for item in actions:
        if isinstance(item, dict) and str(item.get("target") or "") == name:
            return item
    for item in actions:
        if isinstance(item, dict) and str(item.get("base_url") or "") == base_url:
            return item
    return {}


def command_list(action: dict[str, Any]) -> list[str]:
    commands = action.get("commands")
    if not isinstance(commands, list):
        return []
    return [str(item) for item in commands if isinstance(item, str)]


def contract_capability_id(contract_class: str) -> str:
    if contract_class == "openai_compatible_upstream":
        return "openai_compatible_host_conformance"
    if contract_class == "native_provider_transcript_adapters":
        return "native_provider_transcript_wires"
    if contract_class == "direct_kernel_http_syscall":
        return "direct_http_syscall_boundary"
    if contract_class == "direct_kernel_mcp_syscall":
        return "direct_mcp_syscall_boundary"
    return ""


def qualify_target(
    target: dict[str, Any],
    certificate: dict[str, Any] | None,
    contract: dict[str, Any] | None,
    retry: dict[str, Any] | None,
) -> dict[str, Any]:
    name = str(target.get("name") or "")
    base_url = str(target.get("base_url") or "")
    contract_class = str(target.get("contract_class") or "")
    rule_id = CONTRACT_CLASS_RULES.get(contract_class, "")
    external_state = str(target.get("external_state_status") or "UNCLASSIFIED")
    roster_status = str(target.get("roster_status") or "")
    capability_id = contract_capability_id(contract_class)
    action = retry_action(retry, name, base_url)

    if roster_status == "INVALID_TARGET" or external_state == "INVALID_TEMPLATE":
        qualification, evidence_state = "INVALID_TARGET", "UNQUALIFIED"
    elif roster_status == "UNSUPPORTED_WIRE" or external_state == "UNSUPPORTED_TEMPLATE" or not rule_id:
        qualification, evidence_state = "OUT_OF_CONTRACT", "UNQUALIFIED"
    elif external_state in EXTERNAL_TO_QUALIFICATION:
        qualification, evidence_state = EXTERNAL_TO_QUALIFICATION[external_state]
    else:
        qualification, evidence_state = "UNCLASSIFIED", "UNQUALIFIED"

    checks = {
        "target_has_supported_contract_class": bool(rule_id),
        "host_class_proven": host_class_status(contract, contract_class) == "PROVEN",
        "certificate_capability_proven": bool(capability_id) and capability_status(certificate, capability_id) == "PROVEN",
        "external_state_known": external_state != "UNCLASSIFIED",
    }
    in_contract = qualification.startswith("IN_CONTRACT_") and all(checks.values())
    if qualification.startswith("IN_CONTRACT_") and not all(checks.values()):
        qualification = "UNCLASSIFIED"
        evidence_state = "UNQUALIFIED"

    return {
        "version": fleet_version.app_version(),
        "name": name,
        "provider": target.get("provider", ""),
        "base_url": base_url,
        "model_hint": target.get("model_hint", ""),
        "api_key_env": target.get("api_key_env", ""),
        "contract_class": contract_class,
        "qualification_rule": rule_id or "unknown_wire",
        "contract_capability": capability_id,
        "credential_state": target.get("credential_state", ""),
        "external_state_status": external_state,
        "qualification_status": qualification,
        "evidence_state": evidence_state,
        "in_contract": in_contract,
        "checks": checks,
        "next_evidence_needed": target.get("next_evidence_needed", ""),
        "commands": command_list(action),
    }


def artifact_targets(external_state: dict[str, Any] | None, errors: dict[str, str]) -> list[dict[str, Any]]:
    if external_state is None:
        return []
    targets = external_state.get("targets", [])
    if not isinstance(targets, list):
        errors["external_state_targets"] = "external-state artifact targets field is not a JSON list"
        return []
    malformed = len([item for item in targets if not isinstance(item, dict)])
    if malformed:
        errors["external_state_target_rows"] = f"{malformed} target rows are not JSON objects"
    return [item for item in targets if isinstance(item, dict)]


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

    certificate = loaded["certificate"]
    contract = loaded["contract"]
    external_state = loaded["external_state"]
    retry = loaded["retry"]

    certificate_summary = (certificate or {}).get("summary", {})
    contract_summary = (contract or {}).get("summary", {})
    external_summary = (external_state or {}).get("summary", {})
    retry_summary = (retry or {}).get("summary", {})

    if certificate is not None and certificate.get("schema") != CERTIFICATE_SCHEMA:
        errors["certificate_schema"] = f"certificate artifact schema is not {CERTIFICATE_SCHEMA}"
    if certificate is not None and certificate_summary.get("certificate_gate") is not True:
        errors["certificate_gate"] = "certificate gate is not true"
    if contract is not None and contract.get("schema") != CONTRACT_SCHEMA:
        errors["contract_schema"] = f"compatibility contract artifact schema is not {CONTRACT_SCHEMA}"
    if contract is not None and contract_summary.get("contract_gate") is not True:
        errors["contract_gate"] = "compatibility contract gate is not true"
    if external_state is not None and external_state.get("schema") != EXTERNAL_STATE_SCHEMA:
        errors["external_state_schema"] = f"external-state artifact schema is not {EXTERNAL_STATE_SCHEMA}"
    if external_state is not None and external_summary.get("external_state_audit_gate") is not True:
        errors["external_state_audit_gate"] = "external-state audit gate is not true"
    if retry is not None and retry.get("schema") != RETRY_SCHEMA:
        errors["retry_schema"] = f"retry-packet artifact schema is not {RETRY_SCHEMA}"
    if retry is not None and retry_summary.get("retry_packet_gate") is not True:
        errors["retry_packet_gate"] = "retry-packet gate is not true"

    rows = [
        qualify_target(item, certificate, contract, retry)
        for item in artifact_targets(external_state, errors)
    ]
    statuses = [str(row.get("qualification_status") or "") for row in rows]
    unclassified = [row for row in rows if row.get("qualification_status") not in KNOWN_QUALIFICATIONS or row.get("qualification_status") == "UNCLASSIFIED"]
    out_of_contract = [row for row in rows if row.get("qualification_status") == "OUT_OF_CONTRACT"]
    invalid = [row for row in rows if row.get("qualification_status") == "INVALID_TARGET"]
    in_contract = [row for row in rows if row.get("in_contract") is True]
    summary = {
        "targets": len(rows),
        "in_contract_targets": len(in_contract),
        "live_confirmed": statuses.count("IN_CONTRACT_LIVE_CONFIRMED"),
        "ready_for_live_smoke": statuses.count("IN_CONTRACT_READY_FOR_LIVE_SMOKE"),
        "external_blocked": statuses.count("IN_CONTRACT_EXTERNAL_BLOCKER"),
        "needs_credential": statuses.count("IN_CONTRACT_NEEDS_CREDENTIAL"),
        "needs_probe": statuses.count("IN_CONTRACT_NEEDS_PROBE"),
        "out_of_contract": len(out_of_contract),
        "invalid_targets": len(invalid),
        "unclassified": len(unclassified),
        "artifact_errors": len(errors),
    }
    summary["qualification_gate"] = (
        summary["targets"] > 0
        and summary["in_contract_targets"] == summary["targets"]
        and summary["out_of_contract"] == 0
        and summary["invalid_targets"] == 0
        and summary["unclassified"] == 0
        and summary["artifact_errors"] == 0
    )
    return {
        "schema": SCHEMA,
        "app_version": app_ver,
        "generated_at": utc_now(),
        "scope_note": (
            "Operational predicate for the API-host bridge claim. A target is in contract "
            "when its wire class is proven by the certificate and its current external state "
            "is live-confirmed, ready for smoke, a typed external blocker, credential-needed, "
            "or probe-needed. This does not turn external auth/billing state into live proof."
        ),
        "summary": summary,
        "targets": rows,
        "artifact_errors": errors,
        "artifacts": paths,
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# API-Host Qualification",
        "",
        "> Per-target qualification against the proven API-host bridge contract.",
        "",
        "## Summary",
        "",
        f"- In-contract targets: {s['in_contract_targets']}/{s['targets']}",
        f"- Live confirmed: {s['live_confirmed']}",
        f"- Ready for live smoke: {s['ready_for_live_smoke']}",
        f"- External blocked: {s['external_blocked']}",
        f"- Needs credential: {s['needs_credential']}",
        f"- Needs probe: {s['needs_probe']}",
        f"- Out of contract: {s['out_of_contract']}",
        f"- Invalid targets: {s['invalid_targets']}",
        f"- Unclassified: {s['unclassified']}",
        f"- Qualification gate: {'yes' if s['qualification_gate'] else 'no'}",
        "",
        "| target | rule | qualification | evidence | next evidence |",
        "|---|---|---|---|---|",
    ]
    for row in report["targets"]:
        lines.append(
            f"| `{row['name']}` | `{row['qualification_rule']}` | {row['qualification_status']} | {row['evidence_state']} | {row['next_evidence_needed']} |"
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
    ap = argparse.ArgumentParser(description="Qualify API-host targets against the proven bridge contract")
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
    return 0 if report["summary"]["qualification_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
