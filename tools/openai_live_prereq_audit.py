#!/usr/bin/env python3
"""Sanitized prerequisite audit for the hosted OpenAI live proof.

The guard/MCP status packet has Codex MCP live evidence and a dependency-free
Agents guardrail adapter. A *hosted* OpenAI API / installed OpenAI Agents SDK
live proof needs external state that may not exist on the host. This tool records
that state without printing secrets:

* whether OPENAI_API_KEY is set;
* whether the openai package is installed;
* whether the openai-agents distribution is installed;
* whether an importable `agents` module is actually from an installed distribution
  or just a local shadow package.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


SCHEMA = "fak-openai-live-prereq-audit/1"


def dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def module_info(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    out: dict[str, Any] = {"installed": spec is not None}
    if spec is None:
        return out
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - audit surface, not app logic
        out.update({"import_error": type(exc).__name__})
        return out
    out["file"] = str(getattr(mod, "__file__", "") or "")
    out["has_custom_span"] = bool(getattr(mod, "custom_span", None))
    return out


def collect() -> dict[str, Any]:
    openai_version = dist_version("openai")
    agents_version = dist_version("openai-agents")
    legacy_agents_dist = dist_version("agents")
    agents_mod = module_info("agents")
    tracing_mod = module_info("agents.tracing")
    key_set = bool(os.environ.get("OPENAI_API_KEY"))
    base_set = bool(os.environ.get("OPENAI_BASE_URL"))

    blockers: list[str] = []
    if not key_set:
        blockers.append("OPENAI_API_KEY is not set")
    if not openai_version:
        blockers.append("openai package is not installed")
    if not agents_version:
        blockers.append("openai-agents distribution is not installed")
    if agents_mod.get("installed") and not agents_version and not legacy_agents_dist:
        blockers.append("importable agents module is not an installed OpenAI Agents SDK distribution")

    hosted_ready = key_set and bool(openai_version)
    agents_ready = bool(agents_version)
    status = "READY" if hosted_ready and agents_ready else "BLOCKED_ENV"
    if hosted_ready and not agents_ready:
        status = "PARTIAL"

    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "hosted_openai_ready": hosted_ready,
        "agents_sdk_ready": agents_ready,
        "blockers": blockers,
        "env": {
            "OPENAI_API_KEY_set": key_set,
            "OPENAI_BASE_URL_set": base_set,
        },
        "packages": {
            "openai": openai_version,
            "openai-agents": agents_version,
            "agents": legacy_agents_dist,
        },
        "modules": {
            "agents": agents_mod,
            "agents.tracing": tracing_mod,
        },
        "privacy": {
            "copied_fields": ["env var presence booleans", "package versions", "module paths"],
            "dropped": ["OPENAI_API_KEY value", "any request/response payloads"],
        },
    }


def render_md(payload: dict[str, Any]) -> str:
    blockers = payload.get("blockers") or []
    lines = [
        "# OpenAI hosted live proof prerequisites",
        "",
        f"- generated: `{payload.get('created_at')}`",
        f"- status: **`{payload.get('status')}`**",
        f"- hosted_openai_ready: `{payload.get('hosted_openai_ready')}`",
        f"- agents_sdk_ready: `{payload.get('agents_sdk_ready')}`",
        "",
        "## Evidence",
        "",
        f"- OPENAI_API_KEY_set: `{payload.get('env', {}).get('OPENAI_API_KEY_set')}`",
        f"- OPENAI_BASE_URL_set: `{payload.get('env', {}).get('OPENAI_BASE_URL_set')}`",
        f"- openai package: `{payload.get('packages', {}).get('openai')}`",
        f"- openai-agents distribution: `{payload.get('packages', {}).get('openai-agents')}`",
        f"- agents distribution: `{payload.get('packages', {}).get('agents')}`",
        f"- agents module file: `{payload.get('modules', {}).get('agents', {}).get('file', '')}`",
        f"- agents.tracing installed: `{payload.get('modules', {}).get('agents.tracing', {}).get('installed')}`",
        "",
        "## Blockers",
        "",
    ]
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Privacy",
            "",
            "This audit records only presence booleans, package versions, and module paths. It never writes API key values or request payloads.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, help="write JSON report")
    p.add_argument("--markdown", type=Path, help="write Markdown report")
    p.add_argument("--json", action="store_true", help="print JSON to stdout")
    args = p.parse_args(argv)
    payload = collect()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_md(payload), encoding="utf-8")
    if args.json or not (args.out or args.markdown):
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
