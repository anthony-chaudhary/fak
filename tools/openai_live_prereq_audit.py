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
import base64
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
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


def codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".codex"


def sha256_short(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        return {"decode_error": "not a JWT"}
    segment = parts[1] + ("=" * (-len(parts[1]) % 4))
    try:
        decoded = base64.urlsafe_b64decode(segment.encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - malformed local auth state is audit data
        return {"decode_error": type(exc).__name__}
    return payload if isinstance(payload, dict) else {"decode_error": "payload is not an object"}


def unix_exp_to_iso(value: Any) -> tuple[int | None, str | None]:
    try:
        exp = int(value)
    except (TypeError, ValueError):
        return None, None
    return exp, datetime.fromtimestamp(exp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def codex_auth_info(*, home: Path | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Return sanitized Codex CLI ChatGPT-login state.

    `~/.codex/auth.json` contains bearer and refresh tokens. This function only
    records booleans, selected non-secret JWT claims, timestamps, and short hashes
    of stable identifiers so proof artifacts can be audited without becoming
    credentials.
    """

    now = now or datetime.now(timezone.utc)
    auth_path = (home or codex_home()) / "auth.json"
    codex_cli = shutil.which("codex")
    info: dict[str, Any] = {
        "auth_json_present": auth_path.is_file(),
        "codex_cli_present": bool(codex_cli),
        "auth_mode": None,
        "openai_api_key_stored": False,
        "access_token_present": False,
        "refresh_token_present": False,
        "id_token_present": False,
        "codex_login_ready": False,
        "blockers": [],
    }
    if not auth_path.is_file():
        info["blockers"].append("Codex auth.json is not present")
        if not codex_cli:
            info["blockers"].append("codex CLI is not on PATH")
        return info
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        info["auth_json_error"] = type(exc).__name__
        info["blockers"].append("Codex auth.json could not be read as JSON")
        if not codex_cli:
            info["blockers"].append("codex CLI is not on PATH")
        return info
    if not isinstance(data, dict):
        info["auth_json_error"] = "not an object"
        info["blockers"].append("Codex auth.json is not a JSON object")
        if not codex_cli:
            info["blockers"].append("codex CLI is not on PATH")
        return info

    tokens = data.get("tokens") if isinstance(data.get("tokens"), dict) else {}
    access_token = str(tokens.get("access_token") or "")
    refresh_token = str(tokens.get("refresh_token") or "")
    id_token = str(tokens.get("id_token") or "")
    account_id = str(tokens.get("account_id") or "")
    info.update(
        {
            "auth_mode": data.get("auth_mode"),
            "openai_api_key_stored": bool(data.get("OPENAI_API_KEY")),
            "last_refresh": data.get("last_refresh"),
            "access_token_present": bool(access_token),
            "refresh_token_present": bool(refresh_token),
            "id_token_present": bool(id_token),
        }
    )
    if account_id:
        info["account_id_sha256_12"] = sha256_short(account_id)

    access_payload = decode_jwt_payload(access_token) if access_token else {}
    access_exp, access_exp_iso = unix_exp_to_iso(access_payload.get("exp"))
    if access_payload:
        auth_claim = access_payload.get("https://api.openai.com/auth")
        if isinstance(auth_claim, dict) and auth_claim.get("chatgpt_plan_type"):
            info["chatgpt_plan_type"] = auth_claim.get("chatgpt_plan_type")
        info["access_token_issuer"] = access_payload.get("iss")
        info["access_token_audience"] = access_payload.get("aud")
        if access_payload.get("decode_error"):
            info["access_token_decode_error"] = access_payload.get("decode_error")
    info["access_token_exp"] = access_exp
    info["access_token_exp_iso"] = access_exp_iso
    info["access_token_expired"] = bool(access_exp is not None and access_exp <= int(now.timestamp()))

    if not codex_cli:
        info["blockers"].append("codex CLI is not on PATH")
    if data.get("auth_mode") != "chatgpt":
        info["blockers"].append("Codex auth_mode is not chatgpt")
    if not (access_token or refresh_token):
        info["blockers"].append("Codex ChatGPT tokens are not present")
    if access_token and info["access_token_expired"] and not refresh_token:
        info["blockers"].append("Codex access token is expired and no refresh token is present")

    refreshable_or_valid = bool(refresh_token) or bool(access_token and not info["access_token_expired"])
    info["codex_login_ready"] = bool(
        codex_cli and data.get("auth_mode") == "chatgpt" and refreshable_or_valid
    )
    if info["codex_login_ready"]:
        info["blockers"] = []
    return info


def collect() -> dict[str, Any]:
    openai_version = dist_version("openai")
    agents_version = dist_version("openai-agents")
    legacy_agents_dist = dist_version("agents")
    agents_mod = module_info("agents")
    tracing_mod = module_info("agents.tracing")
    key_set = bool(os.environ.get("OPENAI_API_KEY"))
    base_set = bool(os.environ.get("OPENAI_BASE_URL"))
    codex_auth = codex_auth_info()
    platform_api_ready = key_set and bool(openai_version)
    codex_login_ready = bool(codex_auth.get("codex_login_ready"))

    hosted_blockers: list[str] = []
    if not (platform_api_ready or codex_login_ready):
        if not key_set:
            hosted_blockers.append("OPENAI_API_KEY is not set")
        if key_set and not openai_version:
            hosted_blockers.append("openai package is not installed")
        hosted_blockers.extend(str(item) for item in codex_auth.get("blockers") or [])

    agents_blockers: list[str] = []
    if not agents_version:
        agents_blockers.append("openai-agents distribution is not installed")
    if agents_mod.get("installed") and not agents_version and not legacy_agents_dist:
        agents_blockers.append("importable agents module is not an installed OpenAI Agents SDK distribution")

    hosted_ready = platform_api_ready or codex_login_ready
    agents_ready = bool(agents_version)
    status = "READY" if hosted_ready and agents_ready else "BLOCKED_ENV"
    if hosted_ready and not agents_ready:
        status = "PARTIAL"
    blockers = hosted_blockers + agents_blockers

    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "hosted_openai_ready": hosted_ready,
        "platform_api_ready": platform_api_ready,
        "codex_login_ready": codex_login_ready,
        "auth_sources": {
            "platform_api_key": platform_api_ready,
            "codex_login": codex_login_ready,
        },
        "agents_sdk_ready": agents_ready,
        "blockers": blockers,
        "hosted_blockers": hosted_blockers,
        "agents_blockers": agents_blockers,
        "env": {
            "OPENAI_API_KEY_set": key_set,
            "OPENAI_BASE_URL_set": base_set,
        },
        "codex_auth": codex_auth,
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
            "copied_fields": [
                "env var presence booleans",
                "package versions",
                "module paths",
                "sanitized Codex auth booleans",
                "selected non-secret token claims",
                "short hashes of stable account identifiers",
            ],
            "dropped": [
                "OPENAI_API_KEY value",
                "Codex access_token value",
                "Codex refresh_token value",
                "Codex id_token value",
                "raw account/user/org identifiers",
                "any request/response payloads",
            ],
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
        f"- platform_api_ready: `{payload.get('platform_api_ready')}`",
        f"- codex_login_ready: `{payload.get('codex_login_ready')}`",
        f"- agents_sdk_ready: `{payload.get('agents_sdk_ready')}`",
        "",
        "## Evidence",
        "",
        f"- OPENAI_API_KEY_set: `{payload.get('env', {}).get('OPENAI_API_KEY_set')}`",
        f"- OPENAI_BASE_URL_set: `{payload.get('env', {}).get('OPENAI_BASE_URL_set')}`",
        f"- Codex auth_mode: `{payload.get('codex_auth', {}).get('auth_mode')}`",
        f"- Codex auth_json_present: `{payload.get('codex_auth', {}).get('auth_json_present')}`",
        f"- Codex CLI present: `{payload.get('codex_auth', {}).get('codex_cli_present')}`",
        f"- Codex access_token_present: `{payload.get('codex_auth', {}).get('access_token_present')}`",
        f"- Codex refresh_token_present: `{payload.get('codex_auth', {}).get('refresh_token_present')}`",
        f"- Codex access_token_exp_iso: `{payload.get('codex_auth', {}).get('access_token_exp_iso')}`",
        f"- Codex access_token_expired: `{payload.get('codex_auth', {}).get('access_token_expired')}`",
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
            "This audit records only presence booleans, package versions, module paths, sanitized Codex auth state, and token-expiry metadata. It never writes API key values, Codex token values, raw account identifiers, or request payloads.",
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
