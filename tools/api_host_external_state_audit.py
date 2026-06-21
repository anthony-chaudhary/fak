#!/usr/bin/env python3
"""Audit current credential/billing/readiness state for API-host roster targets."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import fleet_version


SCHEMA = "fak.api-host-external-state-audit.v1"
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PATHS = {
    "roster": "fak/experiments/api-host-bridge/api-host-roster.json",
    "readiness": "fak/experiments/api-host-bridge/api-host-readiness.json",
    "acceptance": "fak/experiments/api-host-bridge/api-host-acceptance.json",
    "retry": "fak/experiments/api-host-bridge/api-host-retry-packet.json",
    "live": "fak/experiments/api-host-bridge/api-host-live-inventory.json",
}

EXPECTED_ARTIFACT_SCHEMAS = {
    "roster": "fak.api-host-roster.v1",
    "readiness": "fak.api-host-readiness.v1",
    "acceptance": "fak.api-host-acceptance.v1",
    "retry": "fak.api-host-retry-packet.v1",
    "live": "fak.api-host-live-inventory.v1",
}

AUTH_BLOCKERS = {"AUTH_REQUIRED", "ACCESS_DENIED"}
BILLING_BLOCKERS = {"BILLING_REQUIRED"}
TRANSPORT_BLOCKERS = {"RATE_LIMITED", "TRANSIENT_TRANSPORT"}
KNOWN_EXTERNAL_STATUSES = {
    "LIVE_CONFIRMED",
    "READY_FOR_LIVE_BRIDGE_RUN",
    "BLOCKED_BILLING",
    "BLOCKED_AUTH",
    "NEEDS_CREDENTIAL",
    "RATE_OR_TRANSPORT_BLOCKED",
    "WIRE_SUPPORTED_UNPROBED",
    "UNPROBED_TEMPLATE",
    "NO_AUTH_READY_TO_PROBE",
    "INVALID_TEMPLATE",
    "UNSUPPORTED_TEMPLATE",
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


def env_present(name: str) -> bool:
    return bool(name and os.environ.get(name))


def credential_state(api_key_env: str) -> str:
    if not api_key_env:
        return "NO_AUTH_DECLARED"
    return "ENV_PRESENT" if env_present(api_key_env) else "ENV_MISSING"


def by_name_or_base(rows: list[Any], name: str, base_url: str, name_field: str = "name") -> dict[str, Any]:
    dict_rows = [row for row in rows if isinstance(row, dict)]
    for row in dict_rows:
        if str(row.get(name_field) or "") == name:
            return row
    for row in dict_rows:
        if str(row.get("base_url") or "") == base_url:
            return row
    return {}


def live_proof_for_target(live: dict[str, Any] | None, base_url: str) -> dict[str, Any]:
    if live is None:
        return {}
    for proof in live.get("proofs", []):
        if not isinstance(proof, dict):
            continue
        evidence = proof.get("evidence")
        if isinstance(evidence, dict) and evidence.get("base_url") == base_url:
            return {
                "id": proof.get("id"),
                "status": proof.get("status"),
                "evidence_path": evidence.get("path"),
                "model": evidence.get("model"),
                "base_url": evidence.get("base_url"),
            }
    return {}


def command_count(retry_action: dict[str, Any]) -> int:
    commands = retry_action.get("commands")
    return len(commands) if isinstance(commands, list) else 0


def compact_readiness(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {"status": "NOT_PROBED"}
    return {
        "status": row.get("status", "UNCLASSIFIED"),
        "http_status": row.get("http_status"),
        "models": len(row.get("models") or []) if isinstance(row.get("models"), list) else 0,
        "url": row.get("url", ""),
    }


def compact_acceptance(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {"status": "NOT_ACCEPTED"}
    return {
        "status": row.get("status", "UNCLASSIFIED"),
        "readiness_status": row.get("readiness_status", ""),
        "contract_class": row.get("contract_class", ""),
        "reason": row.get("reason", ""),
        "has_next_live_command": bool(row.get("next_live_command")),
    }


def compact_retry(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {"action_type": "NO_RETRY_ACTION", "status": ""}
    return {
        "status": row.get("status", ""),
        "action_type": row.get("action_type", ""),
        "operator_prerequisite": row.get("operator_prerequisite", ""),
        "commands": command_count(row),
    }


def classify_external_state(
    target: dict[str, Any],
    readiness: dict[str, Any],
    acceptance: dict[str, Any],
    retry_action: dict[str, Any],
    live_proof: dict[str, Any],
    cred_state: str,
) -> tuple[str, str]:
    target_status = str(target.get("status") or "")
    readiness_status = str(readiness.get("status") or "NOT_PROBED")
    acceptance_status = str(acceptance.get("status") or "NOT_ACCEPTED")
    retry_status = str(retry_action.get("status") or "")
    statuses = {readiness_status, acceptance_status, retry_status}

    if target_status == "INVALID_TARGET":
        return "INVALID_TEMPLATE", "Fix the roster target before probing."
    if target_status == "UNSUPPORTED_WIRE":
        return "UNSUPPORTED_TEMPLATE", "Add a compatible transcript adapter or direct syscall path before probing."
    if live_proof.get("status") in {"LIVE_CONFIRMED", "LOCAL_OPENAI_COMPAT_CONFIRMED"} or acceptance_status == "LIVE_BRIDGE_CONFIRMED":
        return "LIVE_CONFIRMED", "Committed evidence already confirms a live bridge run for this host/base URL."
    if statuses & BILLING_BLOCKERS:
        return "BLOCKED_BILLING", compact_retry(retry_action).get("operator_prerequisite") or "Resolve billing/payment state, then rerun the live smoke."
    if statuses & AUTH_BLOCKERS:
        return "BLOCKED_AUTH", compact_retry(retry_action).get("operator_prerequisite") or "Configure a valid bearer token or provider access path."
    if statuses & TRANSPORT_BLOCKERS:
        return "RATE_OR_TRANSPORT_BLOCKED", compact_retry(retry_action).get("operator_prerequisite") or "Retry after quota or transport state recovers."
    if acceptance_status == "READY_FOR_LIVE_BRIDGE_RUN" or readiness_status == "MODELS_CONFIRMED":
        return "READY_FOR_LIVE_BRIDGE_RUN", "Run the transcript adapter live smoke for this target."
    if acceptance_status == "WIRE_SUPPORTED_UNPROBED":
        return "WIRE_SUPPORTED_UNPROBED", "Run the provider-specific live witness for this covered wire."
    if acceptance_status == "NEEDS_AUTH_ENV" or readiness_status == "AUTH_ENV_MISSING" or cred_state == "ENV_MISSING":
        env_name = str(target.get("api_key_env") or "")
        return "NEEDS_CREDENTIAL", f"Set {env_name}, then rerun readiness, acceptance, and live smoke."
    if readiness_status == "NOT_PROBED" and acceptance_status == "NOT_ACCEPTED":
        if cred_state == "NO_AUTH_DECLARED":
            return "NO_AUTH_READY_TO_PROBE", "Run readiness and acceptance probes for this no-auth target."
        return "UNPROBED_TEMPLATE", "Run readiness and acceptance probes for this roster target."
    return "UNPROBED_TEMPLATE", "Run readiness and acceptance probes, then classify any returned blocker."


def audit_target(
    target: dict[str, Any],
    readiness_rows: list[Any],
    acceptance_rows: list[Any],
    retry_rows: list[Any],
    live: dict[str, Any] | None,
) -> dict[str, Any]:
    name = str(target.get("name") or "")
    base_url = str(target.get("base_url") or "")
    readiness = by_name_or_base(readiness_rows, name, base_url)
    acceptance = by_name_or_base(acceptance_rows, name, base_url)
    retry_action = by_name_or_base(retry_rows, name, base_url, name_field="target")
    live_proof = live_proof_for_target(live, base_url)
    api_key_env = str(target.get("api_key_env") or "")
    cred_state = credential_state(api_key_env)
    external_status, next_evidence = classify_external_state(
        target,
        readiness,
        acceptance,
        retry_action,
        live_proof,
        cred_state,
    )
    return {
        "version": fleet_version.app_version(),
        "name": name,
        "provider": target.get("provider", ""),
        "contract_class": target.get("contract_class", ""),
        "base_url": base_url,
        "model_hint": target.get("model_hint", ""),
        "api_key_env": api_key_env,
        "credential_state": cred_state,
        "env_present": cred_state == "ENV_PRESENT",
        "roster_status": target.get("status", ""),
        "readiness": compact_readiness(readiness),
        "acceptance": compact_acceptance(acceptance),
        "retry": compact_retry(retry_action),
        "live_inventory_evidence": live_proof,
        "external_state_status": external_status,
        "next_evidence_needed": next_evidence,
    }


def artifact_rows(data: dict[str, Any] | None, field: str, errors: dict[str, str], key: str) -> list[Any]:
    if data is None:
        return []
    rows = data.get(field, [])
    if not isinstance(rows, list):
        errors[f"{key}_{field}"] = f"{key} artifact field {field!r} is not a JSON list"
        return []
    return rows


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

    roster = loaded["roster"]
    readiness = loaded["readiness"]
    acceptance = loaded["acceptance"]
    retry = loaded["retry"]
    live = loaded["live"]

    roster_rows = artifact_rows(roster, "targets", errors, "roster")
    readiness_rows = artifact_rows(readiness, "probes", errors, "readiness")
    acceptance_rows = artifact_rows(acceptance, "targets", errors, "acceptance")
    retry_rows = artifact_rows(retry, "actions", errors, "retry")

    targets = [
        audit_target(row, readiness_rows, acceptance_rows, retry_rows, live)
        for row in roster_rows
        if isinstance(row, dict)
    ]
    malformed_roster = len([row for row in roster_rows if not isinstance(row, dict)])
    if malformed_roster:
        errors["roster_target_rows"] = f"{malformed_roster} roster target rows are not JSON objects"

    statuses = [str(row.get("external_state_status") or "") for row in targets]
    unclassified = [status for status in statuses if status not in KNOWN_EXTERNAL_STATUSES]
    roster_summary = (roster or {}).get("summary", {})
    if roster is not None and roster_summary.get("roster_gate") is not True:
        errors["roster_gate"] = "roster artifact gate is not true"

    status_counts = {status: statuses.count(status) for status in sorted(set(statuses))}
    summary = {
        "roster_targets": len(targets),
        "env_present": len([row for row in targets if row["credential_state"] == "ENV_PRESENT"]),
        "env_missing": len([row for row in targets if row["credential_state"] == "ENV_MISSING"]),
        "no_auth_declared": len([row for row in targets if row["credential_state"] == "NO_AUTH_DECLARED"]),
        "live_confirmed": status_counts.get("LIVE_CONFIRMED", 0),
        "ready_for_live_run": status_counts.get("READY_FOR_LIVE_BRIDGE_RUN", 0),
        "blocked_auth": status_counts.get("BLOCKED_AUTH", 0),
        "blocked_billing": status_counts.get("BLOCKED_BILLING", 0),
        "needs_credential": status_counts.get("NEEDS_CREDENTIAL", 0),
        "rate_or_transport_blocked": status_counts.get("RATE_OR_TRANSPORT_BLOCKED", 0),
        "unprobed_templates": status_counts.get("UNPROBED_TEMPLATE", 0) + status_counts.get("NO_AUTH_READY_TO_PROBE", 0),
        "invalid_templates": status_counts.get("INVALID_TEMPLATE", 0),
        "unsupported_templates": status_counts.get("UNSUPPORTED_TEMPLATE", 0),
        "unclassified": len(unclassified),
        "artifact_errors": len(errors),
        "status_counts": status_counts,
    }
    summary["external_state_audit_gate"] = (
        summary["roster_targets"] > 0
        and summary["artifact_errors"] == 0
        and summary["unclassified"] == 0
        and summary["invalid_templates"] == 0
        and summary["unsupported_templates"] == 0
    )
    return {
        "schema": SCHEMA,
        "app_version": app_ver,
        "generated_at": utc_now(),
        "scope_note": (
            "Current external-state audit for API-host roster targets. It records "
            "whether credentials are present, and whether host proof is live, ready, "
            "blocked by auth/billing/transport state, or still unprobed. Secret values "
            "are never emitted."
        ),
        "summary": summary,
        "targets": targets,
        "artifact_errors": errors,
        "artifacts": paths,
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# API-Host External State Audit",
        "",
        "> Credential, billing, readiness, and retry state for API-host roster targets.",
        "",
        "## Summary",
        "",
        f"- Roster targets: {s['roster_targets']}",
        f"- Env present: {s['env_present']}",
        f"- Env missing: {s['env_missing']}",
        f"- No auth declared: {s['no_auth_declared']}",
        f"- Live confirmed: {s['live_confirmed']}",
        f"- Ready for live run: {s['ready_for_live_run']}",
        f"- Blocked auth: {s['blocked_auth']}",
        f"- Blocked billing: {s['blocked_billing']}",
        f"- Unprobed templates: {s['unprobed_templates']}",
        f"- Artifact errors: {s['artifact_errors']}",
        f"- External-state audit gate: {'yes' if s['external_state_audit_gate'] else 'no'}",
        "",
        "| target | credential | external state | next evidence |",
        "|---|---|---|---|",
    ]
    for row in report["targets"]:
        lines.append(
            f"| `{row['name']}` | {row['credential_state']} | {row['external_state_status']} | {row['next_evidence_needed']} |"
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
    ap = argparse.ArgumentParser(description="Audit external API-host credential/billing/readiness state")
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
    return 0 if report["summary"]["external_state_audit_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
