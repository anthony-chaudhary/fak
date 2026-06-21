#!/usr/bin/env python3
"""Build a credential-conditioned live-smoke queue for API-host targets."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import fleet_version


SCHEMA = "fak.api-host-live-smoke-queue.v1"
QUALIFICATION_SCHEMA = "fak.api-host-qualification.v1"
RETRY_SCHEMA = "fak.api-host-retry-packet.v1"
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PATHS = {
    "qualification": "fak/experiments/api-host-bridge/api-host-qualification.json",
    "retry": "fak/experiments/api-host-bridge/api-host-retry-packet.json",
}

QUALIFICATION_TO_QUEUE = {
    "IN_CONTRACT_LIVE_CONFIRMED": "COMPLETE",
    "IN_CONTRACT_READY_FOR_LIVE_SMOKE": "READY_TO_EXECUTE",
    "IN_CONTRACT_EXTERNAL_BLOCKER": "BLOCKED_EXTERNAL_STATE",
    "IN_CONTRACT_NEEDS_CREDENTIAL": "WAITING_FOR_CREDENTIAL",
    "IN_CONTRACT_NEEDS_PROBE": "READY_FOR_PROBE",
}

QUEUE_STATES = set(QUALIFICATION_TO_QUEUE.values()) | {"UNQUALIFIED", "UNCLASSIFIED"}
COMMAND_REQUIRED_STATES = {"READY_TO_EXECUTE", "BLOCKED_EXTERNAL_STATE", "WAITING_FOR_CREDENTIAL", "READY_FOR_PROBE"}


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


def retry_actions_by_target(retry: dict[str, Any] | None, errors: dict[str, str]) -> dict[str, dict[str, Any]]:
    if retry is None:
        return {}
    actions = retry.get("actions", [])
    if not isinstance(actions, list):
        errors["retry_actions"] = "retry-packet artifact actions field is not a JSON list"
        return {}
    out: dict[str, dict[str, Any]] = {}
    malformed = 0
    for action in actions:
        if not isinstance(action, dict):
            malformed += 1
            continue
        name = str(action.get("target") or "")
        if name:
            out[name] = action
    if malformed:
        errors["retry_action_rows"] = f"{malformed} retry action rows are not JSON objects"
    return out


def command_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def operator_prerequisite(queue_state: str, qualification: dict[str, Any], action: dict[str, Any]) -> str:
    if queue_state == "COMPLETE":
        return "none"
    if queue_state == "READY_TO_EXECUTE":
        return "none"
    text = str(action.get("operator_prerequisite") or "").strip()
    if text:
        return text
    if queue_state == "WAITING_FOR_CREDENTIAL":
        env = str(qualification.get("api_key_env") or action.get("required_env") or "API key").strip()
        return f"Set {env} for this host."
    if queue_state == "READY_FOR_PROBE":
        return "Run readiness and acceptance probes before live smoke."
    if queue_state == "BLOCKED_EXTERNAL_STATE":
        return "Resolve the typed external host blocker before retrying."
    return "Inspect the target qualification before retrying."


def build_queue_row(qualification: dict[str, Any], action: dict[str, Any], version: str | None = None) -> dict[str, Any]:
    qualification_status = str(qualification.get("qualification_status") or "UNCLASSIFIED")
    queue_state = QUALIFICATION_TO_QUEUE.get(qualification_status, "UNCLASSIFIED")
    version = version or fleet_version.app_version()
    if qualification_status in {"OUT_OF_CONTRACT", "INVALID_TARGET"} or qualification.get("in_contract") is not True:
        queue_state = "UNQUALIFIED"

    commands = command_list(qualification.get("commands"))
    if not commands:
        commands = command_list(action.get("commands"))
    if queue_state == "COMPLETE":
        commands = []

    return {
        "version": version,
        "target": qualification.get("name"),
        "provider": qualification.get("provider"),
        "contract_class": qualification.get("contract_class"),
        "base_url": qualification.get("base_url"),
        "model_hint": qualification.get("model_hint"),
        "api_key_env": qualification.get("api_key_env") or action.get("required_env") or "",
        "qualification_status": qualification_status,
        "evidence_state": qualification.get("evidence_state"),
        "external_state_status": qualification.get("external_state_status"),
        "queue_state": queue_state,
        "operator_prerequisite": operator_prerequisite(queue_state, qualification, action),
        "commands": commands,
        "retry_action_type": action.get("action_type", ""),
        "retry_status": action.get("status", ""),
        "latest_sweep": action.get("latest_sweep") if isinstance(action.get("latest_sweep"), dict) else {},
        "next_evidence_needed": qualification.get("next_evidence_needed", ""),
    }


def qualification_targets(qualification: dict[str, Any] | None, errors: dict[str, str]) -> list[dict[str, Any]]:
    if qualification is None:
        return []
    targets = qualification.get("targets", [])
    if not isinstance(targets, list):
        errors["qualification_targets"] = "qualification artifact targets field is not a JSON list"
        return []
    malformed = len([item for item in targets if not isinstance(item, dict)])
    if malformed:
        errors["qualification_target_rows"] = f"{malformed} qualification target rows are not JSON objects"
    return [item for item in targets if isinstance(item, dict)]


def build_report(root: Path | None = None, paths: dict[str, str] | None = None) -> dict[str, Any]:
    root = root or ROOT
    paths = paths or DEFAULT_PATHS
    app_ver = fleet_version.app_version(root)
    qualification, qualification_error = load_json(root, paths["qualification"])
    retry, retry_error = load_json(root, paths["retry"])
    errors: dict[str, str] = {}
    if qualification_error:
        errors["qualification"] = qualification_error
    if retry_error:
        errors["retry"] = retry_error

    qualification_summary = (qualification or {}).get("summary", {})
    retry_summary = (retry or {}).get("summary", {})
    if qualification is not None and qualification.get("schema") != QUALIFICATION_SCHEMA:
        errors["qualification_schema"] = f"qualification artifact schema is not {QUALIFICATION_SCHEMA}"
    if qualification is not None and qualification_summary.get("qualification_gate") is not True:
        errors["qualification_gate"] = "qualification gate is not true"
    if retry is not None and retry.get("schema") != RETRY_SCHEMA:
        errors["retry_schema"] = f"retry-packet artifact schema is not {RETRY_SCHEMA}"
    if retry is not None and retry_summary.get("retry_packet_gate") is not True:
        errors["retry_packet_gate"] = "retry-packet gate is not true"

    actions = retry_actions_by_target(retry, errors)
    rows = [
        build_queue_row(target, actions.get(str(target.get("name") or ""), {}), app_ver)
        for target in qualification_targets(qualification, errors)
    ]

    states = [str(row.get("queue_state") or "UNCLASSIFIED") for row in rows]
    command_gaps = [
        row for row in rows
        if row.get("queue_state") in COMMAND_REQUIRED_STATES and not row.get("commands")
    ]
    unqualified = [row for row in rows if row.get("queue_state") == "UNQUALIFIED"]
    unclassified = [
        row for row in rows
        if row.get("queue_state") not in QUEUE_STATES or row.get("queue_state") == "UNCLASSIFIED"
    ]

    summary = {
        "targets": len(rows),
        "complete": states.count("COMPLETE"),
        "ready_to_execute": states.count("READY_TO_EXECUTE"),
        "blocked_external_state": states.count("BLOCKED_EXTERNAL_STATE"),
        "waiting_for_credential": states.count("WAITING_FOR_CREDENTIAL"),
        "ready_for_probe": states.count("READY_FOR_PROBE"),
        "unqualified": len(unqualified),
        "unclassified": len(unclassified),
        "command_gaps": len(command_gaps),
        "commands": sum(len(row.get("commands") or []) for row in rows),
        "artifact_errors": len(errors),
    }
    summary["live_smoke_queue_gate"] = (
        summary["targets"] > 0
        and summary["artifact_errors"] == 0
        and summary["unqualified"] == 0
        and summary["unclassified"] == 0
        and summary["command_gaps"] == 0
    )

    return {
        "schema": SCHEMA,
        "app_version": app_ver,
        "generated_at": utc_now(),
        "scope_note": (
            "Credential-conditioned queue for completing live API-host smoke coverage. "
            "It lists executable commands and external prerequisites; it does not run "
            "paid/keyed hosts or mark external billing/auth/access state as solved."
        ),
        "summary": summary,
        "queue": rows,
        "artifact_errors": errors,
        "artifacts": paths,
        "qualification_summary": qualification_summary,
        "retry_summary": retry_summary,
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# API-Host Live Smoke Queue",
        "",
        "> Credential-conditioned execution queue for remaining API-host live-smoke evidence.",
        "",
        "## Summary",
        "",
        f"- Targets: {s['targets']}",
        f"- Complete: {s['complete']}",
        f"- Ready to execute: {s['ready_to_execute']}",
        f"- Blocked external state: {s['blocked_external_state']}",
        f"- Waiting for credential: {s['waiting_for_credential']}",
        f"- Ready for probe: {s['ready_for_probe']}",
        f"- Unqualified: {s['unqualified']}",
        f"- Unclassified: {s['unclassified']}",
        f"- Command gaps: {s['command_gaps']}",
        f"- Queue gate: {'yes' if s['live_smoke_queue_gate'] else 'no'}",
        "",
        "| target | queue state | prerequisite | command count |",
        "|---|---|---|---:|",
    ]
    for row in report["queue"]:
        lines.append(
            f"| `{row['target']}` | {row['queue_state']} | {row['operator_prerequisite']} | {len(row.get('commands') or [])} |"
        )
    for row in report["queue"]:
        commands = row.get("commands") or []
        if not commands:
            continue
        lines += ["", f"## {row['target']}", ""]
        for command in commands:
            lines += ["```powershell", command, "```", ""]
    if report.get("artifact_errors"):
        lines += ["", "## Artifact Errors", "", "```json", json.dumps(report["artifact_errors"], indent=2), "```", ""]
    lines.append("")
    return "\n".join(lines)


def write_text(path: str, body: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a credential-conditioned API-host live-smoke queue")
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
    return 0 if report["summary"]["live_smoke_queue_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
