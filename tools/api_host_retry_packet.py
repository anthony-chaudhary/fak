#!/usr/bin/env python3
"""Build retry instructions for API hosts blocked by external state."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import fleet_version


SCHEMA = "fak.api-host-retry-packet.v1"
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PATHS = {
    "acceptance": "fak/experiments/api-host-bridge/api-host-acceptance.json",
    "live": "fak/experiments/api-host-bridge/api-host-live-inventory.json",
}

EXTERNAL_BLOCKERS = {
    "NEEDS_AUTH_ENV",
    "AUTH_REQUIRED",
    "ACCESS_DENIED",
    "BILLING_REQUIRED",
    "RATE_LIMITED",
    "TRANSIENT_TRANSPORT",
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


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "api-host"


def suggested_api_key_env(target: dict[str, Any]) -> str:
    existing = str(target.get("api_key_env") or "").strip()
    if existing:
        return existing
    name = str(target.get("name") or "api_host").strip().lower()
    for suffix in ("_no_key", "_gateway", "_openai_compatible", "-no-key", "-gateway", "-openai-compatible"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    token = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").upper()
    return f"{token or 'API_HOST'}_API_KEY"


def target_spec(target: dict[str, Any], key_env: str) -> str:
    parts = [
        str(target.get("name") or ""),
        str(target.get("provider") or ""),
        str(target.get("base_url") or ""),
        key_env,
        str(target.get("model_hint") or ""),
    ]
    return "|".join(parts)


def readiness_spec(target: dict[str, Any], key_env: str) -> str:
    parts = [
        str(target.get("name") or ""),
        str(target.get("base_url") or ""),
        key_env,
        str(target.get("model_hint") or ""),
    ]
    return "|".join(parts)


def live_smoke_command(target: dict[str, Any], key_env: str) -> str:
    name = str(target.get("name") or "api_host")
    base_url = str(target.get("base_url") or "<base_url>")
    model = str(target.get("model_hint") or "<model>")
    out_dir = f"fak/experiments/agent-live/transcript-adapter-sweep-{safe_slug(name)}-retry"
    return (
        "pwsh tools/run_transcript_adapter_sweep.ps1 "
        f"-OutDir {out_dir} "
        f"-ApiBaseUrl {ps_quote(base_url)} "
        f"-ApiKeyEnv {key_env} "
        f"-ApiModels {ps_quote(model)} "
        "-SkipOffline -SkipLocalShim -SkipMicrobench "
        "-MaxTurns 12 -Trials 1"
    )


def readiness_command(target: dict[str, Any], key_env: str) -> str:
    spec = readiness_spec(target, key_env)
    return (
        "fak api-host readiness "
        f"--target {ps_quote(spec)} "
        "--out fak/experiments/api-host-bridge/api-host-readiness.json "
        "--markdown fak/experiments/api-host-bridge/api-host-readiness.md"
    )


def acceptance_command(target: dict[str, Any], key_env: str) -> str:
    spec = target_spec(target, key_env)
    return (
        "fak api-host acceptance "
        f"--target {ps_quote(spec)} "
        "--out fak/experiments/api-host-bridge/api-host-acceptance.json "
        "--markdown fak/experiments/api-host-bridge/api-host-acceptance.md"
    )


def matching_live_proof(live: dict[str, Any] | None, target: dict[str, Any]) -> dict[str, Any]:
    if live is None:
        return {}
    base_url = str(target.get("base_url") or "")
    for proof in live.get("proofs", []):
        if not isinstance(proof, dict):
            continue
        evidence = proof.get("evidence")
        if isinstance(evidence, dict) and evidence.get("base_url") == base_url:
            return {
                "id": proof.get("id"),
                "status": proof.get("status"),
                "evidence": evidence,
            }
    return {}


def latest_sweep_evidence(target: dict[str, Any]) -> dict[str, Any]:
    row = target.get("latest_sweep")
    if not isinstance(row, dict):
        return {}
    return {
        "summary_path": row.get("_summary_path"),
        "status": row.get("status"),
        "model": row.get("model"),
        "error_excerpt": str(row.get("error") or "")[:240],
        "transcript_sha": row.get("transcript_sha"),
    }


def classify_action(target: dict[str, Any], live: dict[str, Any] | None, version: str | None = None) -> dict[str, Any]:
    status = str(target.get("status") or "UNCLASSIFIED")
    key_env = suggested_api_key_env(target)
    provider = str(target.get("provider") or "")
    can_run_sweep = provider in {"openai-compatible", "openai", "xai"}
    version = version or fleet_version.app_version()

    row: dict[str, Any] = {
        "version": version,
        "target": target.get("name"),
        "provider": provider,
        "contract_class": target.get("contract_class"),
        "base_url": target.get("base_url"),
        "model_hint": target.get("model_hint"),
        "status": status,
        "readiness_status": target.get("readiness_status"),
        "api_key_env": str(target.get("api_key_env") or ""),
        "required_env": key_env,
        "external_state": status in EXTERNAL_BLOCKERS,
        "action_type": "",
        "operator_prerequisite": "",
        "commands": [],
        "latest_sweep": latest_sweep_evidence(target),
        "live_inventory_evidence": matching_live_proof(live, target),
    }

    if status == "READY_FOR_LIVE_BRIDGE_RUN":
        row["action_type"] = "run_live_smoke"
        row["operator_prerequisite"] = "none"
        row["commands"] = [live_smoke_command(target, key_env)] if can_run_sweep else []
    elif status == "LIVE_BRIDGE_CONFIRMED":
        row["action_type"] = "none_already_confirmed"
        row["operator_prerequisite"] = "none"
    elif status == "NEEDS_AUTH_ENV":
        row["action_type"] = "set_auth_env_then_probe_and_smoke"
        row["operator_prerequisite"] = f"Set {key_env} for this host."
        row["commands"] = [
            readiness_command(target, key_env),
            acceptance_command(target, key_env),
            live_smoke_command(target, key_env),
        ] if can_run_sweep else [readiness_command(target, key_env), acceptance_command(target, key_env)]
    elif status == "BILLING_REQUIRED":
        row["action_type"] = "fix_billing_then_smoke"
        row["operator_prerequisite"] = f"Attach billing/payment method for the account behind {key_env}."
        row["commands"] = [live_smoke_command(target, key_env)] if can_run_sweep else []
    elif status in {"AUTH_REQUIRED", "ACCESS_DENIED"}:
        row["action_type"] = "configure_access_then_probe_and_smoke"
        row["operator_prerequisite"] = f"Configure a valid bearer token or access path via {key_env}."
        row["commands"] = [
            readiness_command(target, key_env),
            acceptance_command(target, key_env),
            live_smoke_command(target, key_env),
        ] if can_run_sweep else [readiness_command(target, key_env), acceptance_command(target, key_env)]
    elif status == "RATE_LIMITED":
        row["action_type"] = "wait_or_raise_quota_then_smoke"
        row["operator_prerequisite"] = "Wait for quota reset or raise the host quota."
        row["commands"] = [live_smoke_command(target, key_env)] if can_run_sweep else []
    elif status == "TRANSIENT_TRANSPORT":
        row["action_type"] = "retry_after_transport_recovers"
        row["operator_prerequisite"] = "Wait until the host transport path is healthy."
        row["commands"] = [live_smoke_command(target, key_env)] if can_run_sweep else []
    elif status == "WIRE_SUPPORTED_UNPROBED":
        row["action_type"] = "add_or_run_provider_specific_live_witness"
        row["operator_prerequisite"] = "Use the covered native/direct wire path; generic OpenAI-compatible smoke does not apply."
    elif status in {"UNSUPPORTED_WIRE", "MODELS_SHAPE_MISMATCH", "INVALID_TARGET", "UNCLASSIFIED"}:
        row["action_type"] = "inspect_or_extend_contract"
        row["operator_prerequisite"] = "Inspect the host shape or fix the target before retrying."
    else:
        row["action_type"] = "inspect_status"
        row["operator_prerequisite"] = "Status is not in the retry-packet vocabulary."
    return row


def build_report(root: Path | None = None, paths: dict[str, str] | None = None) -> dict[str, Any]:
    root = root or ROOT
    paths = paths or DEFAULT_PATHS
    app_ver = fleet_version.app_version(root)
    acceptance, acceptance_error = load_json(root, paths["acceptance"])
    live, live_error = load_json(root, paths["live"])
    artifact_errors = {}
    if acceptance_error:
        artifact_errors["acceptance"] = acceptance_error
    if live_error:
        artifact_errors["live"] = live_error

    targets = acceptance.get("targets", []) if acceptance else []
    if not isinstance(targets, list):
        targets = []
        artifact_errors["acceptance_targets"] = "acceptance artifact targets field is not a JSON list"

    actions = [classify_action(t, live, app_ver) for t in targets if isinstance(t, dict)]
    malformed_targets = len([t for t in targets if not isinstance(t, dict)])
    if malformed_targets:
        artifact_errors["acceptance_target_rows"] = f"{malformed_targets} target rows are not JSON objects"

    external = [a for a in actions if a["status"] in EXTERNAL_BLOCKERS]
    ready = [a for a in actions if a["status"] == "READY_FOR_LIVE_BRIDGE_RUN"]
    live_confirmed = [a for a in actions if a["status"] == "LIVE_BRIDGE_CONFIRMED"]
    unsupported = [a for a in actions if a["status"] == "UNSUPPORTED_WIRE"]
    invalid = [a for a in actions if a["status"] == "INVALID_TARGET"]
    unclassified = [a for a in actions if a["status"] == "UNCLASSIFIED"]
    shape_mismatch = [a for a in actions if a["status"] == "MODELS_SHAPE_MISMATCH"]
    action_gaps = [
        a for a in actions
        if a["status"] in EXTERNAL_BLOCKERS | {"READY_FOR_LIVE_BRIDGE_RUN"}
        and not a.get("commands")
    ]
    acceptance_summary = (acceptance or {}).get("summary", {})
    acceptance_artifact_errors = (acceptance or {}).get("artifact_errors", [])
    if acceptance is not None:
        if not isinstance(acceptance_artifact_errors, list):
            artifact_errors["acceptance_artifact_errors"] = "acceptance artifact artifact_errors field is not a JSON list"
        elif acceptance_artifact_errors:
            artifact_errors["acceptance_artifact_errors"] = f"acceptance artifact reports {len(acceptance_artifact_errors)} artifact errors"
        if "sweep_artifact_errors" not in acceptance_summary:
            artifact_errors["acceptance_sweep_integrity"] = "acceptance artifact summary lacks sweep_artifact_errors"
        elif acceptance_summary.get("sweep_artifact_errors") != 0:
            artifact_errors["acceptance_sweep_integrity"] = f"acceptance artifact reports {acceptance_summary.get('sweep_artifact_errors')} sweep artifact errors"
        if "invalid_targets" not in acceptance_summary:
            artifact_errors["acceptance_target_integrity"] = "acceptance artifact summary lacks invalid_targets"
    gate = (
        bool(actions)
        and not artifact_errors
        and acceptance is not None
        and acceptance.get("schema") == "fak.api-host-acceptance.v1"
        and bool(acceptance_summary.get("acceptance_gate"))
        and not unsupported
        and not invalid
        and not unclassified
        and not shape_mismatch
        and not action_gaps
    )
    return {
        "schema": SCHEMA,
        "app_version": app_ver,
        "generated_at": utc_now(),
        "scope_note": (
            "Operational retry packet for candidate API hosts. It records the exact "
            "operator prerequisite and rerun command for typed external blockers; "
            "it does not mark billing/auth/access as solved."
        ),
        "summary": {
            "targets": len(actions),
            "actionable_blockers": len(external),
            "ready_for_live_bridge_run": len(ready),
            "live_bridge_confirmed": len(live_confirmed),
            "unsupported_wire": len(unsupported),
            "invalid_targets": len(invalid),
            "shape_mismatch": len(shape_mismatch),
            "unclassified": len(unclassified),
            "action_gaps": len(action_gaps),
            "artifact_errors": len(artifact_errors),
            "commands": sum(len(a.get("commands") or []) for a in actions),
            "retry_packet_gate": gate,
        },
        "actions": actions,
        "artifact_errors": artifact_errors,
        "artifacts": paths,
        "acceptance_summary": acceptance_summary,
        "live_summary": (live or {}).get("summary", {}),
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# API-Host Retry Packet",
        "",
        "> Next actions for candidate hosts blocked by auth, billing, access, rate limit, or transport state.",
        "",
        "## Summary",
        "",
        f"- Targets: {s['targets']}",
        f"- Actionable blockers: {s['actionable_blockers']}",
        f"- Ready for live bridge run: {s['ready_for_live_bridge_run']}",
        f"- Live bridge confirmed: {s['live_bridge_confirmed']}",
        f"- Unsupported wire: {s['unsupported_wire']}",
        f"- Invalid targets: {s['invalid_targets']}",
        f"- Unclassified: {s['unclassified']}",
        f"- Action gaps: {s['action_gaps']}",
        f"- Retry packet gate: {'yes' if s['retry_packet_gate'] else 'no'}",
        "",
        "| target | status | required operator state | command count |",
        "|---|---|---|---|",
    ]
    for action in report["actions"]:
        lines.append(
            f"| `{action['target']}` | {action['status']} | {action['operator_prerequisite']} | {len(action.get('commands') or [])} |"
        )
    for action in report["actions"]:
        commands = action.get("commands") or []
        if not commands:
            continue
        lines += ["", f"## {action['target']}", ""]
        for cmd in commands:
            lines += ["```powershell", cmd, "```", ""]
    if report.get("artifact_errors"):
        lines += ["", "## Artifact Errors", "", "```json", json.dumps(report["artifact_errors"], indent=2), "```", ""]
    return "\n".join(lines)


def write_text(path: str, body: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build retry instructions for blocked API-host bridge candidates")
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
    return 0 if report["summary"]["retry_packet_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
