#!/usr/bin/env python3
"""Cheaply probe current API-host readiness without running chat completions."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import fleet_version


SCHEMA = "fak.api-host-readiness.v1"

DEFAULT_TARGETS = [
    {
        "name": "gemini_openai_compatible",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
        "model_hint": "gemini-2.5-flash",
    },
    {
        "name": "glama_gateway",
        "base_url": "https://gateway.glama.ai/v1",
        "api_key_env": "GLAMA_API_KEY",
        "model_hint": "openai/gpt-4.1-nano-2025-04-14",
    },
    {
        "name": "pollinations_no_key",
        "base_url": "https://gen.pollinations.ai/v1",
        "api_key_env": "",
        "model_hint": "openai-fast",
    },
]

OPENAI_COMPATIBLE_CLASSES = {"openai_compatible_upstream"}

KNOWN_STATUSES = {
    "MODELS_CONFIRMED",
    "HTTP_OK_NO_MODELS",
    "AUTH_ENV_MISSING",
    "AUTH_REQUIRED",
    "ACCESS_DENIED",
    "BILLING_REQUIRED",
    "RATE_LIMITED",
    "TRANSIENT_TRANSPORT",
    "HTTP_STATUS",
    "INVALID_TARGET",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_target(spec: str) -> dict[str, str]:
    parts = [part.strip() for part in spec.split("|")]
    if len(parts) < 2 or len(parts) > 4:
        raise ValueError("target must be name|base_url[|api_key_env[|model_hint]]")
    target = {
        "name": parts[0],
        "base_url": parts[1],
        "api_key_env": parts[2] if len(parts) >= 3 else "",
        "model_hint": parts[3] if len(parts) >= 4 else "",
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
        contract_class = str(row.get("contract_class") or "")
        status = str(row.get("status") or "")
        if contract_class not in OPENAI_COMPATIBLE_CLASSES or status == "INVALID_TARGET":
            continue
        target = {
            "name": str(row.get("name") or "").strip(),
            "base_url": str(row.get("base_url") or "").strip(),
            "api_key_env": str(row.get("api_key_env") or "").strip(),
            "model_hint": str(row.get("model_hint") or "").strip(),
        }
        err = target_error(target)
        if err:
            raise ValueError(f"roster target {target.get('name') or idx}: {err}")
        targets.append(target)
    return targets


def target_error(target: dict[str, str]) -> str:
    name = str(target.get("name", "")).strip()
    base_url = str(target.get("base_url", "")).strip()
    if not name:
        return "target name is empty"
    if not base_url:
        return "target base_url is empty"
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return f"target base_url must be an absolute http(s) URL: {base_url!r}"
    return ""


def models_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/models"


def classify_http(status: int, body: str) -> str:
    lower = body.lower()
    if status == 200:
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return "HTTP_OK_NO_MODELS"
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return "MODELS_CONFIRMED"
        return "HTTP_OK_NO_MODELS"
    if status == 403 and ("access denied" in lower or "error 1010" in lower or "browser_signature_banned" in lower):
        return "ACCESS_DENIED"
    if status in {401, 403}:
        return "AUTH_REQUIRED"
    if status == 402 or "no_payment_method" in lower or "no payment method" in lower:
        return "BILLING_REQUIRED"
    if status == 429:
        return "RATE_LIMITED"
    if 500 <= status <= 599:
        return "TRANSIENT_TRANSPORT"
    return "HTTP_STATUS"


def model_ids(body: str) -> list[str]:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return []
    out = []
    for item in data.get("data", []) if isinstance(data, dict) else []:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            out.append(item["id"])
    return out


def probe_target(target: dict[str, str], timeout_s: float = 10.0, probe_missing_auth: bool = False) -> dict[str, Any]:
    err = target_error(target)
    if err:
        return {
            "version": fleet_version.app_version(),
            **target,
            "url": "",
            "status": "INVALID_TARGET",
            "http_status": None,
            "models": [],
            "body_excerpt": "",
            "error": err,
        }
    key_env = target.get("api_key_env", "")
    key = os.environ.get(key_env) if key_env else ""
    if key_env and not key and not probe_missing_auth:
        return {
            "version": fleet_version.app_version(),
            **target,
            "url": models_url(target["base_url"]),
            "status": "AUTH_ENV_MISSING",
            "http_status": None,
            "models": [],
            "error": f"environment variable {key_env} is not set",
        }

    req = urllib.request.Request(models_url(target["base_url"]))
    req.add_header("Accept", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = int(resp.status)
            return {
                "version": fleet_version.app_version(),
                **target,
                "url": req.full_url,
                "status": classify_http(status, raw),
                "http_status": status,
                "models": model_ids(raw)[:25],
                "body_excerpt": raw[:500],
                "error": "",
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return {
            "version": fleet_version.app_version(),
            **target,
            "url": req.full_url,
            "status": classify_http(int(exc.code), raw),
            "http_status": int(exc.code),
            "models": [],
            "body_excerpt": raw[:500],
            "error": raw[:500],
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "version": fleet_version.app_version(),
            **target,
            "url": req.full_url,
            "status": "TRANSIENT_TRANSPORT",
            "http_status": None,
            "models": [],
            "body_excerpt": "",
            "error": str(exc),
        }


def build_report(
    targets: list[dict[str, str]] | None = None,
    timeout_s: float = 10.0,
    probe_missing_auth: bool = False,
) -> dict[str, Any]:
    app_ver = fleet_version.app_version()
    targets = targets or DEFAULT_TARGETS
    probes = [probe_target(t, timeout_s=timeout_s, probe_missing_auth=probe_missing_auth) for t in targets]
    summary = {
        "targets": len(probes),
        "models_confirmed": len([p for p in probes if p["status"] == "MODELS_CONFIRMED"]),
        "auth_env_missing": len([p for p in probes if p["status"] == "AUTH_ENV_MISSING"]),
        "auth_required": len([p for p in probes if p["status"] == "AUTH_REQUIRED"]),
        "access_denied": len([p for p in probes if p["status"] == "ACCESS_DENIED"]),
        "billing_required": len([p for p in probes if p["status"] == "BILLING_REQUIRED"]),
        "transient_transport": len([p for p in probes if p["status"] == "TRANSIENT_TRANSPORT"]),
        "invalid_targets": len([p for p in probes if p["status"] == "INVALID_TARGET"]),
        "unclassified": len([p for p in probes if p["status"] not in KNOWN_STATUSES]),
    }
    summary["readiness_gate"] = summary["unclassified"] == 0 and summary["invalid_targets"] == 0 and summary["targets"] > 0
    return {
        "schema": SCHEMA,
        "app_version": app_ver,
        "generated_at": utc_now(),
        "scope_note": (
            "Cheap current-state /models probe. It verifies endpoint shape or "
            "records typed auth/billing/transport states; it does not run chat "
            "completions or spend model tokens."
        ),
        "summary": summary,
        "probes": probes,
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# API-Host Readiness Probe",
        "",
        "> Current-state `/models` probe for compatible API hosts.",
        "",
        "## Summary",
        "",
        f"- Targets: {s['targets']}",
        f"- Models confirmed: {s['models_confirmed']}",
        f"- Auth env missing: {s['auth_env_missing']}",
        f"- Auth required: {s['auth_required']}",
        f"- Access denied: {s['access_denied']}",
        f"- Billing required: {s['billing_required']}",
        f"- Transient transport: {s['transient_transport']}",
        f"- Invalid targets: {s['invalid_targets']}",
        f"- Readiness gate: {'yes' if s['readiness_gate'] else 'no'}",
        "",
        "| target | status | HTTP | models |",
        "|---|---|---:|---:|",
    ]
    for probe in report["probes"]:
        lines.append(f"| `{probe['name']}` | {probe['status']} | {probe.get('http_status') or ''} | {len(probe.get('models') or [])} |")
    lines.append("")
    return "\n".join(lines)


def write_text(path: str, body: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Probe API-host /models readiness")
    ap.add_argument("--target", action="append", default=[], help="name|base_url[|api_key_env[|model_hint]]")
    ap.add_argument("--from-roster", default="", help="read probe targets from an api_host_roster JSON artifact")
    ap.add_argument("--out", default="", help="write JSON report here")
    ap.add_argument("--markdown", default="", help="write Markdown report here")
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--probe-missing-auth", action="store_true", help="send unauthenticated request even when api_key_env is unset")
    args = ap.parse_args(argv)

    if args.target and args.from_roster:
        raise ValueError("--target and --from-roster are mutually exclusive")
    targets = load_roster_targets(args.from_roster) if args.from_roster else ([parse_target(t) for t in args.target] if args.target else DEFAULT_TARGETS)
    report = build_report(targets, timeout_s=args.timeout_s, probe_missing_auth=args.probe_missing_auth)
    body = json.dumps(report, indent=2) + "\n"
    if args.out:
        write_text(args.out, body)
    else:
        print(body, end="")
    if args.markdown:
        write_text(args.markdown, markdown(report))
    return 0 if report["summary"]["readiness_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
