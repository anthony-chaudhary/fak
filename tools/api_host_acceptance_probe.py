#!/usr/bin/env python3
"""Classify candidate API hosts against the scoped FAK/DOS bridge contract."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import api_host_readiness_probe as readiness
import fleet_version


SCHEMA = "fak.api-host-acceptance.v1"
ROOT = Path(__file__).resolve().parents[1]

OPENAI_COMPATIBLE = {"openai-compatible", "openai", "xai"}
NATIVE_PROVIDERS = {"anthropic", "gemini"}
DIRECT_PROVIDERS = {"direct-http", "direct-mcp"}
SUPPORTED_PROVIDERS = OPENAI_COMPATIBLE | NATIVE_PROVIDERS | DIRECT_PROVIDERS
EXTERNAL_BLOCKERS = {"NEEDS_AUTH_ENV", "AUTH_REQUIRED", "ACCESS_DENIED", "BILLING_REQUIRED", "RATE_LIMITED", "TRANSIENT_TRANSPORT"}

DEFAULT_TARGETS = [
    {
        "name": "gemini_openai_compatible",
        "provider": "openai-compatible",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
        "model_hint": "gemini-2.5-flash",
    },
    {
        "name": "glama_gateway",
        "provider": "openai-compatible",
        "base_url": "https://gateway.glama.ai/v1",
        "api_key_env": "GLAMA_API_KEY",
        "model_hint": "openai/gpt-4.1-nano-2025-04-14",
    },
    {
        "name": "pollinations_no_key",
        "provider": "openai-compatible",
        "base_url": "https://gen.pollinations.ai/v1",
        "api_key_env": "",
        "model_hint": "openai-fast",
    },
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_provider(provider: str) -> str:
    p = provider.strip().lower()
    aliases = {
        "": "openai-compatible",
        "gpt": "openai",
        "chat-completions": "openai-compatible",
        "claude": "anthropic",
        "google": "gemini",
        "grok": "xai",
        "http": "direct-http",
        "mcp": "direct-mcp",
    }
    return aliases.get(p, p)


def parse_target(spec: str) -> dict[str, str]:
    parts = [part.strip() for part in spec.split("|")]
    if len(parts) < 3 or len(parts) > 5:
        raise ValueError("target must be name|provider|base_url[|api_key_env[|model_hint]]")
    target = {
        "name": parts[0],
        "provider": normalize_provider(parts[1]),
        "base_url": parts[2],
        "api_key_env": parts[3] if len(parts) >= 4 else "",
        "model_hint": parts[4] if len(parts) >= 5 else "",
    }
    err = target_error(target)
    if err:
        raise ValueError(err)
    return target


def load_roster_targets(path: str) -> list[dict[str, str]]:
    roster_path = Path(path)
    data = json.loads(roster_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"roster artifact is not a JSON object: {path}")
    rows = data.get("targets", [])
    if not isinstance(rows, list):
        raise ValueError(f"roster targets field is not a JSON list: {path}")
    targets: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"roster target {idx} is not a JSON object")
        target = {
            "name": str(row.get("name") or "").strip(),
            "provider": normalize_provider(str(row.get("provider") or "")),
            "base_url": str(row.get("base_url") or "").strip(),
            "api_key_env": str(row.get("api_key_env") or "").strip(),
            "model_hint": str(row.get("model_hint") or "").strip(),
        }
        targets.append(target)
    return targets


def target_error(target: dict[str, str]) -> str:
    if not str(target.get("name", "")).strip():
        return "target name is empty"
    if not str(target.get("base_url", "")).strip():
        return "target base_url is empty"
    return ""


def contract_class(provider: str) -> str:
    if provider in OPENAI_COMPATIBLE:
        return "openai_compatible_upstream"
    if provider in NATIVE_PROVIDERS:
        return "native_provider_transcript_adapters"
    if provider == "direct-http":
        return "direct_kernel_http_syscall"
    if provider == "direct-mcp":
        return "direct_kernel_mcp_syscall"
    return "unsupported"


def acceptance_status(provider: str, probe_status: str) -> tuple[str, str]:
    if provider not in SUPPORTED_PROVIDERS:
        return "UNSUPPORTED_WIRE", "Host does not match a covered API-host wire."
    if provider in DIRECT_PROVIDERS:
        return "WIRE_SUPPORTED_UNPROBED", "Direct HTTP/MCP integration is covered by executable source witnesses; no remote /models probe applies."
    if provider in NATIVE_PROVIDERS:
        return "WIRE_SUPPORTED_UNPROBED", "Native provider wire is covered by adapter and gateway witnesses; no generic no-spend /models endpoint is assumed."
    if probe_status == "MODELS_CONFIRMED":
        return "READY_FOR_LIVE_BRIDGE_RUN", "Compatible /models surface is reachable; candidate is ready for a live bridge smoke run."
    if probe_status == "AUTH_ENV_MISSING":
        return "NEEDS_AUTH_ENV", "Required API key environment variable is not set."
    if probe_status in {"AUTH_REQUIRED", "ACCESS_DENIED", "BILLING_REQUIRED", "RATE_LIMITED", "TRANSIENT_TRANSPORT"}:
        return probe_status, "Candidate host is compatible-shaped but blocked by a typed external state."
    if probe_status == "HTTP_OK_NO_MODELS":
        return "MODELS_SHAPE_MISMATCH", "HTTP 200 response did not expose an OpenAI-compatible model list."
    return "UNCLASSIFIED", "Candidate host returned an unclassified readiness state."


def rel_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_sweep_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for path in sorted((root / "fak/experiments/agent-live").glob("transcript-adapter-sweep*/sweep-summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            errors.append({"path": rel_path(root, path), "error": f"invalid JSON: {exc}"})
            continue
        except OSError as exc:
            errors.append({"path": rel_path(root, path), "error": f"cannot read artifact: {exc}"})
            continue
        if not isinstance(data, list):
            errors.append({"path": rel_path(root, path), "error": "sweep summary is not a JSON list"})
            continue
        for idx, row in enumerate(data):
            if isinstance(row, dict):
                rows.append({**row, "_summary_path": rel_path(root, path)})
            else:
                errors.append({"path": rel_path(root, path), "row_index": idx, "error": "sweep row is not a JSON object"})
    return rows, errors


def classify_sweep_error(error: str) -> str:
    lower = error.lower()
    if "http 402" in lower or "no_payment_method" in lower or "no payment method" in lower:
        return "BILLING_REQUIRED"
    if "http 401" in lower or "authentication required" in lower or "unauthorized" in lower:
        return "AUTH_REQUIRED"
    if "access denied" in lower or "browser_signature_banned" in lower:
        return "ACCESS_DENIED"
    if "http 429" in lower or "rate limit" in lower:
        return "RATE_LIMITED"
    if "http 5" in lower or "timeout" in lower or "connection" in lower:
        return "TRANSIENT_TRANSPORT"
    return "UNCLASSIFIED"


def latest_sweep_row(target: dict[str, str], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    model = target.get("model_hint", "")
    matched = [
        r for r in rows
        if r.get("kind") == "api"
        and r.get("base_url") == target.get("base_url")
        and (not model or r.get("model") == model)
    ]
    if not matched:
        return None
    return sorted(matched, key=lambda r: str(r.get("generated_at") or ""))[-1]


def status_from_live_sweep(row: dict[str, Any] | None) -> tuple[str, str]:
    if row is None:
        return "", ""
    if row.get("status") == "ok" and row.get("live") is True and row.get("transcript_sha"):
        return "LIVE_BRIDGE_CONFIRMED", "A committed live sweep row confirms this host drove the bridge."
    if row.get("status") == "failed":
        status = classify_sweep_error(str(row.get("error") or ""))
        if status != "UNCLASSIFIED":
            return status, "Latest live bridge smoke hit a typed external blocker."
        return "UNCLASSIFIED", "Latest live bridge smoke failed with an unclassified error."
    return "", ""


def classify_target(
    target: dict[str, str],
    timeout_s: float,
    probe_missing_auth: bool,
    sweep_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    provider = normalize_provider(target.get("provider", ""))
    normalized = {**target, "provider": provider}
    cls = contract_class(provider)
    err = target_error(normalized)
    if err:
        return {
            "version": fleet_version.app_version(),
            **normalized,
            "contract_class": cls,
            "status": "INVALID_TARGET",
            "reason": err,
            "readiness_status": "NOT_PROBED",
            "probe": None,
            "latest_sweep": None,
            "next_live_command": "",
        }
    probe: dict[str, Any] | None = None
    probe_status = "NOT_PROBED"
    sweep_row = latest_sweep_row(normalized, sweep_rows or [])
    live_status, live_reason = status_from_live_sweep(sweep_row)
    if provider in OPENAI_COMPATIBLE:
        probe_target = {
            "name": normalized["name"],
            "base_url": normalized["base_url"],
            "api_key_env": normalized.get("api_key_env", ""),
            "model_hint": normalized.get("model_hint", ""),
        }
        probe = readiness.probe_target(probe_target, timeout_s=timeout_s, probe_missing_auth=probe_missing_auth)
        probe_status = str(probe.get("status", "UNCLASSIFIED"))
    status, reason = acceptance_status(provider, probe_status)
    if live_status:
        status, reason = live_status, live_reason
    return {
        "version": fleet_version.app_version(),
        **normalized,
        "contract_class": cls,
        "status": status,
        "reason": reason,
        "readiness_status": probe_status,
        "probe": probe,
        "latest_sweep": sweep_row,
        "next_live_command": next_live_command(normalized) if status == "READY_FOR_LIVE_BRIDGE_RUN" else "",
    }


def next_live_command(target: dict[str, str]) -> str:
    model = target.get("model_hint") or "<model>"
    key_env = target.get("api_key_env") or "<api_key_env>"
    return (
        "pwsh tools/run_transcript_adapter_sweep.ps1 "
        f"-ApiBaseUrl {target['base_url']} -ApiKeyEnv {key_env} -ApiModels {model} "
        "-SkipOffline -SkipLocalShim -SkipMicrobench"
    )


def build_report(
    targets: list[dict[str, str]] | None = None,
    timeout_s: float = 10.0,
    probe_missing_auth: bool = False,
    root: Path | None = None,
) -> dict[str, Any]:
    root = root or ROOT
    app_ver = fleet_version.app_version(root)
    targets = targets or DEFAULT_TARGETS
    sweep_rows, artifact_errors = load_sweep_rows(root)
    rows = [classify_target(t, timeout_s=timeout_s, probe_missing_auth=probe_missing_auth, sweep_rows=sweep_rows) for t in targets]
    known = [r for r in rows if r["status"] != "UNCLASSIFIED"]
    ready = [r for r in rows if r["status"] == "READY_FOR_LIVE_BRIDGE_RUN"]
    live_confirmed = [r for r in rows if r["status"] == "LIVE_BRIDGE_CONFIRMED"]
    external = [
        r
        for r in rows
        if r["status"] in EXTERNAL_BLOCKERS
    ]
    unsupported = [r for r in rows if r["status"] == "UNSUPPORTED_WIRE"]
    unprobed_supported = [r for r in rows if r["status"] == "WIRE_SUPPORTED_UNPROBED"]
    invalid = [r for r in rows if r["status"] == "INVALID_TARGET"]
    return {
        "schema": SCHEMA,
        "app_version": app_ver,
        "generated_at": utc_now(),
        "scope_note": (
            "No-spend candidate-host acceptance gate. READY_FOR_LIVE_BRIDGE_RUN means a host is "
            "compatible-shaped and reachable enough to run a live bridge smoke; typed blockers are "
            "external state, not bridge failures."
        ),
        "summary": {
            "targets": len(rows),
            "known_statuses": len(known),
            "ready_for_live_bridge_run": len(ready),
            "live_bridge_confirmed": len(live_confirmed),
            "wire_supported_unprobed": len(unprobed_supported),
            "typed_external_blockers": len(external),
            "unsupported_wire": len(unsupported),
            "invalid_targets": len(invalid),
            "unclassified": len(rows) - len(known),
            "sweep_artifact_errors": len(artifact_errors),
            "acceptance_gate": len(rows) > 0 and len(rows) == len(known) and not unsupported and not invalid and not artifact_errors,
        },
        "targets": rows,
        "artifact_errors": artifact_errors,
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# API-Host Acceptance Probe",
        "",
        "> Candidate-host gate for the scoped FAK/DOS API-host bridge contract.",
        "",
        "## Summary",
        "",
        f"- Targets: {s['targets']}",
        f"- Ready for live bridge run: {s['ready_for_live_bridge_run']}",
        f"- Live bridge confirmed: {s['live_bridge_confirmed']}",
        f"- Wire supported but unprobed: {s['wire_supported_unprobed']}",
        f"- Typed external blockers: {s['typed_external_blockers']}",
        f"- Unsupported wire: {s['unsupported_wire']}",
        f"- Invalid targets: {s['invalid_targets']}",
        f"- Sweep artifact errors: {s['sweep_artifact_errors']}",
        f"- Acceptance gate: {'yes' if s['acceptance_gate'] else 'no'}",
        "",
        "| target | provider | class | status | readiness |",
        "|---|---|---|---|---|",
    ]
    for row in report["targets"]:
        lines.append(
            f"| `{row['name']}` | `{row['provider']}` | `{row['contract_class']}` | {row['status']} | {row['readiness_status']} |"
        )
    if report.get("artifact_errors"):
        lines += [
            "",
            "## Artifact Errors",
            "",
            "| path | detail |",
            "|---|---|",
        ]
        for err in report["artifact_errors"]:
            lines.append(f"| `{err.get('path', '')}` | {err.get('error', '')} |")
    lines.append("")
    return "\n".join(lines)


def write_text(path: str, body: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Classify candidate API hosts against the FAK/DOS bridge contract")
    ap.add_argument("--target", action="append", default=[], help="name|provider|base_url[|api_key_env[|model_hint]]")
    ap.add_argument("--from-roster", default="", help="read candidate targets from an api_host_roster JSON artifact")
    ap.add_argument("--out", default="", help="write JSON report here")
    ap.add_argument("--markdown", default="", help="write Markdown report here")
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--probe-missing-auth", action="store_true", help="send unauthenticated request even when api_key_env is unset")
    ap.add_argument("--root", default=str(ROOT), help="workspace root used to inspect existing live sweep rows")
    args = ap.parse_args(argv)

    if args.target and args.from_roster:
        raise ValueError("--target and --from-roster are mutually exclusive")
    targets = load_roster_targets(args.from_roster) if args.from_roster else ([parse_target(t) for t in args.target] if args.target else DEFAULT_TARGETS)
    report = build_report(targets, timeout_s=args.timeout_s, probe_missing_auth=args.probe_missing_auth, root=Path(args.root))
    body = json.dumps(report, indent=2) + "\n"
    if args.out:
        write_text(args.out, body)
    else:
        print(body, end="")
    if args.markdown:
        write_text(args.markdown, markdown(report))
    return 0 if report["summary"]["acceptance_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
