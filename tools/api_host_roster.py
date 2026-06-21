#!/usr/bin/env python3
"""Maintain a no-spend roster of API-host bridge target templates."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import api_host_acceptance_probe as acceptance
import api_host_readiness_probe as readiness
import fleet_version


SCHEMA = "fak.api-host-roster.v1"
ROOT = Path(__file__).resolve().parents[1]

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
    {
        "name": "openai_api",
        "provider": "openai-compatible",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model_hint": "gpt-4.1-mini",
    },
    {
        "name": "xai_api",
        "provider": "xai",
        "base_url": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
        "model_hint": "grok-3-mini",
    },
    {
        "name": "openrouter_gateway",
        "provider": "openai-compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_hint": "openai/gpt-4.1-mini",
    },
    {
        "name": "groq_openai",
        "provider": "openai-compatible",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model_hint": "llama-3.3-70b-versatile",
    },
    {
        "name": "together_openai",
        "provider": "openai-compatible",
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "model_hint": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    {
        "name": "mistral_openai",
        "provider": "openai-compatible",
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "model_hint": "mistral-small-latest",
    },
    {
        "name": "deepseek_openai",
        "provider": "openai-compatible",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_hint": "deepseek-chat",
    },
    {
        "name": "fireworks_openai",
        "provider": "openai-compatible",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "model_hint": "accounts/fireworks/models/llama-v3p1-8b-instruct",
    },
    {
        "name": "perplexity_openai",
        "provider": "openai-compatible",
        "base_url": "https://api.perplexity.ai",
        "api_key_env": "PERPLEXITY_API_KEY",
        "model_hint": "sonar",
    },
    {
        "name": "cerebras_openai",
        "provider": "openai-compatible",
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "model_hint": "llama3.1-8b",
    },
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def target_spec(target: dict[str, str]) -> str:
    return "|".join([
        target.get("name", ""),
        target.get("provider", ""),
        target.get("base_url", ""),
        target.get("api_key_env", ""),
        target.get("model_hint", ""),
    ])


def readiness_spec(target: dict[str, str]) -> str:
    return "|".join([
        target.get("name", ""),
        target.get("base_url", ""),
        target.get("api_key_env", ""),
        target.get("model_hint", ""),
    ])


def acceptance_command(target: dict[str, str]) -> str:
    return (
        "python tools/api_host_acceptance_probe.py "
        f"--target {ps_quote(target_spec(target))} "
        "--out fak/experiments/api-host-bridge/api-host-acceptance.json "
        "--markdown fak/experiments/api-host-bridge/api-host-acceptance.md"
    )


def readiness_command(target: dict[str, str]) -> str:
    return (
        "python tools/api_host_readiness_probe.py "
        f"--target {ps_quote(readiness_spec(target))} "
        "--out fak/experiments/api-host-bridge/api-host-readiness.json "
        "--markdown fak/experiments/api-host-bridge/api-host-readiness.md"
    )


def bulk_acceptance_command(targets: list[dict[str, str]]) -> str:
    parts = ["python tools/api_host_acceptance_probe.py"]
    for target in targets:
        parts += ["--target", ps_quote(target_spec(target))]
    parts += [
        "--out", "fak/experiments/api-host-bridge/api-host-acceptance.json",
        "--markdown", "fak/experiments/api-host-bridge/api-host-acceptance.md",
    ]
    return " ".join(parts)


def bulk_readiness_command(targets: list[dict[str, str]]) -> str:
    parts = ["python tools/api_host_readiness_probe.py"]
    for target in targets:
        parts += ["--target", ps_quote(readiness_spec(target))]
    parts += [
        "--out", "fak/experiments/api-host-bridge/api-host-readiness.json",
        "--markdown", "fak/experiments/api-host-bridge/api-host-readiness.md",
    ]
    return " ".join(parts)


def normalize_target(raw: dict[str, Any]) -> dict[str, str]:
    return {
        "name": str(raw.get("name", "")).strip(),
        "provider": acceptance.normalize_provider(str(raw.get("provider", ""))),
        "base_url": str(raw.get("base_url", "")).strip(),
        "api_key_env": str(raw.get("api_key_env", "")).strip(),
        "model_hint": str(raw.get("model_hint", "")).strip(),
    }


def target_row(raw: dict[str, Any]) -> dict[str, Any]:
    target = normalize_target(raw)
    acceptance_err = acceptance.target_error(target)
    readiness_err = ""
    if not acceptance_err:
        readiness_err = readiness.target_error({
            "name": target["name"],
            "base_url": target["base_url"],
            "api_key_env": target["api_key_env"],
            "model_hint": target["model_hint"],
        })
    cls = acceptance.contract_class(target["provider"])
    valid = not acceptance_err and not readiness_err
    supported = cls != "unsupported"
    status = "SUPPORTED_TEMPLATE" if valid and supported else ("UNSUPPORTED_WIRE" if valid else "INVALID_TARGET")
    return {
        "version": fleet_version.app_version(),
        **target,
        "contract_class": cls,
        "status": status,
        "error": acceptance_err or readiness_err,
        "requires_auth": bool(target["api_key_env"]),
        "readiness_command": readiness_command(target) if valid else "",
        "acceptance_command": acceptance_command(target) if valid else "",
    }


def build_report(targets: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    app_ver = fleet_version.app_version()
    raw_targets = targets or DEFAULT_TARGETS
    rows = [target_row(item) for item in raw_targets]
    names = [row["name"] for row in rows]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    invalid = [row for row in rows if row["status"] == "INVALID_TARGET"]
    unsupported = [row for row in rows if row["status"] == "UNSUPPORTED_WIRE"]
    supported = [row for row in rows if row["status"] == "SUPPORTED_TEMPLATE"]
    openai_compatible = [row for row in supported if row["contract_class"] == "openai_compatible_upstream"]
    native = [row for row in supported if row["contract_class"] == "native_provider_transcript_adapters"]
    direct = [row for row in supported if row["contract_class"].startswith("direct_kernel_")]
    gate = len(rows) > 0 and not invalid and not unsupported and not duplicate_names
    return {
        "schema": SCHEMA,
        "app_version": app_ver,
        "generated_at": utc_now(),
        "scope_note": (
            "No-spend target roster. These are operator templates for hosts that match "
            "a supported wire class; live success still requires credentials, billing, "
            "and provider access state."
        ),
        "summary": {
            "targets": len(rows),
            "supported_templates": len(supported),
            "openai_compatible_templates": len(openai_compatible),
            "native_provider_templates": len(native),
            "direct_templates": len(direct),
            "invalid_targets": len(invalid),
            "unsupported_wire": len(unsupported),
            "duplicate_names": len(duplicate_names),
            "auth_required_templates": len([row for row in rows if row["requires_auth"]]),
            "roster_gate": gate,
        },
        "targets": rows,
        "duplicate_names": duplicate_names,
        "bulk_commands": {
            "readiness": bulk_readiness_command([row for row in rows if row["status"] != "INVALID_TARGET"]),
            "acceptance": bulk_acceptance_command([row for row in rows if row["status"] != "INVALID_TARGET"]),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# API-Host Roster",
        "",
        "> No-spend target templates for compatible API-host bridge candidates.",
        "",
        "## Summary",
        "",
        f"- Targets: {s['targets']}",
        f"- Supported templates: {s['supported_templates']}",
        f"- OpenAI-compatible templates: {s['openai_compatible_templates']}",
        f"- Invalid targets: {s['invalid_targets']}",
        f"- Unsupported wire: {s['unsupported_wire']}",
        f"- Duplicate names: {s['duplicate_names']}",
        f"- Roster gate: {'yes' if s['roster_gate'] else 'no'}",
        "",
        "| target | provider | class | env | status |",
        "|---|---|---|---|---|",
    ]
    for row in report["targets"]:
        lines.append(
            f"| `{row['name']}` | `{row['provider']}` | `{row['contract_class']}` | `{row['api_key_env']}` | {row['status']} |"
        )
    lines += [
        "",
        "## Bulk Commands",
        "",
        "```bash",
        report["bulk_commands"]["readiness"],
        "```",
        "",
        "```bash",
        report["bulk_commands"]["acceptance"],
        "```",
        "",
    ]
    return "\n".join(lines)


def write_text(path: str, body: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Emit API-host target roster templates")
    ap.add_argument("--target", action="append", default=[], help="name|provider|base_url[|api_key_env[|model_hint]]")
    ap.add_argument("--out", default="", help="write JSON report here")
    ap.add_argument("--markdown", default="", help="write Markdown report here")
    args = ap.parse_args(argv)

    targets = [acceptance.parse_target(item) for item in args.target] if args.target else DEFAULT_TARGETS
    report = build_report(targets)
    body = json.dumps(report, indent=2) + "\n"
    if args.out:
        write_text(args.out, body)
    else:
        print(body, end="")
    if args.markdown:
        write_text(args.markdown, markdown(report))
    return 0 if report["summary"]["roster_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
