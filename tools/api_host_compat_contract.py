#!/usr/bin/env python3
"""Define and verify the scoped API-host compatibility contract."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import fleet_version


SCHEMA = "fak.api-host-compat-contract.v1"
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PATHS = {
    "matrix": "fak/experiments/api-host-bridge/api-host-bridge-matrix.json",
    "gate": "fak/experiments/api-host-bridge/api-host-bridge-gate.json",
    "live": "fak/experiments/api-host-bridge/api-host-live-inventory.json",
    "readiness": "fak/experiments/api-host-bridge/api-host-readiness.json",
    "acceptance": "fak/experiments/api-host-bridge/api-host-acceptance.json",
}

EXPECTED_ARTIFACT_SCHEMAS = {
    "matrix": "fak.api-host-bridge-matrix.v1",
    "gate": "fak.api-host-bridge-gate.v1",
    "live": "fak.api-host-live-inventory.v1",
    "readiness": "fak.api-host-readiness.v1",
    "acceptance": "fak.api-host-acceptance.v1",
}

REQUIRED_PROVIDER_SHAPES = ["anthropic", "gemini", "openai-compatible", "xai"]


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
    for witness in matrix.get("witnesses", []):
        if not isinstance(witness, dict):
            continue
        if witness.get("id") == id:
            return str(witness.get("status"))
    return "missing_witness"


def run_status(gate: dict[str, Any] | None, id: str) -> str:
    if gate is None:
        return "missing_gate"
    for run in gate.get("runs", []):
        if not isinstance(run, dict):
            continue
        if run.get("id") == id:
            return str(run.get("status"))
    return "missing_run"


def contract_row(id: str, claim: str, checks: dict[str, bool], evidence: dict[str, Any]) -> dict[str, Any]:
    proven = all(checks.values())
    return {
        "version": fleet_version.app_version(),
        "id": id,
        "claim": claim,
        "status": "PROVEN" if proven else "FAILED",
        "checks": checks,
        "evidence": evidence,
    }


def acceptance_models_confirmed(acceptance: dict[str, Any] | None) -> int:
    if acceptance is None:
        return 0
    count = 0
    for row in acceptance.get("targets", []):
        if not isinstance(row, dict):
            continue
        if row.get("readiness_status") == "MODELS_CONFIRMED" or row.get("status") in {"READY_FOR_LIVE_BRIDGE_RUN", "LIVE_BRIDGE_CONFIRMED"}:
            count += 1
    return count


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
    live = loaded["live"]
    readiness = loaded["readiness"]
    acceptance = loaded["acceptance"]

    provider_shapes = []
    if matrix is not None:
        provider_shapes = list((matrix.get("summary") or {}).get("provider_shapes_covered") or [])

    live_summary = (live or {}).get("summary", {})
    readiness_summary = (readiness or {}).get("summary", {})
    acceptance_summary = (acceptance or {}).get("summary", {})
    acceptance_confirmed = acceptance_models_confirmed(acceptance)
    current_models_confirmed = int(readiness_summary.get("models_confirmed", 0) or 0) + acceptance_confirmed

    classes = [
        contract_row(
            "openai_compatible_upstream",
            "An OpenAI-compatible upstream can be fronted while FAK/DOS owns tool-call filtering, repair, and downstream stream=true chunk synthesis after adjudication.",
            {
                "source_witness_resolved": witness_status(matrix, "openai_compatible_gateway") == "resolved",
                "witness_command_passed": run_status(gate, "openai_compatible_gateway") == "passed",
                "provider_proxy_end_to_end_passed": run_status(gate, "provider_proxy_end_to_end") == "passed",
                "host_agnostic_alias_passed": run_status(gate, "host_agnostic_openai_compatible") == "passed",
                "host_profile_conformance_passed": run_status(gate, "openai_compatible_host_profiles") == "passed",
                "current_models_surface_confirmed": current_models_confirmed >= 1,
                "no_invalid_readiness_targets": readiness_summary.get("invalid_targets") == 0 and acceptance_summary.get("invalid_targets", 0) == 0,
            },
            {
                "matrix_witness": "openai_compatible_gateway",
                "gate_run": "openai_compatible_gateway",
                "end_to_end_gate_run": "provider_proxy_end_to_end",
                "host_agnostic_gate_run": "host_agnostic_openai_compatible",
                "host_profile_gate_run": "openai_compatible_host_profiles",
                "readiness": readiness_summary,
                "acceptance_models_confirmed": acceptance_confirmed,
                "acceptance": acceptance_summary,
            },
        ),
        contract_row(
            "native_provider_transcript_adapters",
            "Native provider adapters preserve pre-send quarantine and provider-specific tool shapes.",
            {
                "source_witness_resolved": witness_status(matrix, "native_provider_adapters") == "resolved",
                "witness_command_passed": run_status(gate, "native_provider_adapters") == "passed",
                "provider_proxy_end_to_end_passed": run_status(gate, "provider_proxy_end_to_end") == "passed",
                "required_provider_shapes_covered": sorted(provider_shapes) == REQUIRED_PROVIDER_SHAPES,
            },
            {
                "required_provider_shapes": REQUIRED_PROVIDER_SHAPES,
                "provider_shapes_covered": provider_shapes,
                "end_to_end_gate_run": "provider_proxy_end_to_end",
            },
        ),
        contract_row(
            "direct_kernel_http_syscall",
            "Any-language clients can bypass provider quirks and call the kernel over native HTTP.",
            {
                "source_witness_resolved": witness_status(matrix, "direct_http_syscall") == "resolved",
                "witness_command_passed": run_status(gate, "direct_http_syscall") == "passed",
            },
            {"matrix_witness": "direct_http_syscall", "gate_run": "direct_http_syscall"},
        ),
        contract_row(
            "direct_kernel_mcp_syscall",
            "Any-language clients can call the same kernel over MCP stdio or MCP-over-HTTP.",
            {
                "source_witness_resolved": witness_status(matrix, "direct_mcp_syscall") == "resolved",
                "witness_command_passed": run_status(gate, "direct_mcp_syscall") == "passed",
            },
            {"matrix_witness": "direct_mcp_syscall", "gate_run": "direct_mcp_syscall"},
        ),
        contract_row(
            "live_scoped_host_evidence",
            "Committed live artifacts prove the bridge on at least one real frontier OpenAI-compatible host plus local OpenAI-compatible shims.",
            {
                "live_inventory_gate": live_summary.get("live_inventory_gate") is True,
                "frontier_successes": live_summary.get("live_frontier_successes", 0) >= 2,
                "local_shim_success": live_summary.get("local_openai_compatible_successes", 0) >= 1,
                "no_unclassified_live_rows": live_summary.get("incomplete_or_unclassified") == 0,
            },
            {"live": live_summary},
        ),
    ]

    failed = [item for item in classes if item["status"] != "PROVEN"]
    non_claims = [
        {
            "version": app_ver,
            "id": "arbitrary_api_host_without_compatible_wire",
            "status": "OUT_OF_CONTRACT",
            "reason": "A host must expose an OpenAI-compatible wire, one of the covered native provider wires, or use direct HTTP/MCP syscall integration.",
        },
        {
            "version": app_ver,
            "id": "streaming_chat_completions_delta_passthrough",
            "status": "OUT_OF_CONTRACT",
            "reason": "Downstream stream=true is supported by synthesized chunks after full adjudication; raw upstream streaming delta passthrough remains out of contract.",
        },
        {
            "version": app_ver,
            "id": "paid_or_keyed_live_execution_without_credentials",
            "status": "EXTERNAL_STATE",
            "reason": "Billing, API-key, or edge-access failures are typed readiness states, not bridge failures.",
        },
        {
            "version": app_ver,
            "id": "provider_semantics_beyond_tool_wire",
            "status": "OUT_OF_CONTRACT",
            "reason": "The bridge owns tool-call admission and result quarantine; it does not claim vendor model quality or universal task success.",
        },
    ]
    return {
        "schema": SCHEMA,
        "app_version": app_ver,
        "generated_at": utc_now(),
        "contract": "A compatible API host is one that can present an OpenAI-compatible upstream, a covered native provider transcript wire, or direct HTTP/MCP access to the FAK/DOS syscall boundary.",
        "summary": {
            "host_classes": len(classes),
            "proven_host_classes": len(classes) - len(failed),
            "failed_host_classes": len(failed),
            "contract_gate": len(failed) == 0 and not errors,
        },
        "host_classes": classes,
        "non_claims": non_claims,
        "artifact_errors": errors,
        "artifacts": paths,
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# API-Host Compatibility Contract",
        "",
        report["contract"],
        "",
        "## Summary",
        "",
        f"- Host classes proven: {s['proven_host_classes']}/{s['host_classes']}",
        f"- Contract gate: {'yes' if s['contract_gate'] else 'no'}",
        "",
        "## Host Classes",
        "",
        "| host class | status | claim |",
        "|---|---|---|",
    ]
    for item in report["host_classes"]:
        lines.append(f"| `{item['id']}` | {item['status']} | {item['claim']} |")
    lines += [
        "",
        "## Non-Claims",
        "",
        "| item | status | reason |",
        "|---|---|---|",
    ]
    for item in report["non_claims"]:
        lines.append(f"| `{item['id']}` | {item['status']} | {item['reason']} |")
    lines.append("")
    return "\n".join(lines)


def write_text(path: str, body: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the scoped API-host compatibility contract")
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
    return 0 if report["summary"]["contract_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
