#!/usr/bin/env python3
"""Verify external permission-system benchmark source claims."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import fleet_version


SCHEMA = "fak.permission-source-audit.v1"

SOURCES = [
    {
        "id": "anthropic_auto_blog",
        "system": "claude_code_auto",
        "url": "https://www.anthropic.com/engineering/claude-code-auto-mode",
        "claim": "Claude Code auto mode is classifier-mediated and publishes non-zero false-negative rates.",
        "patterns": [
            r"auto mode .* delegates approvals to model-based classifiers",
            r"17% false-negative rate",
            r"5\.7% FNR",
            r"prompt-injection probe .* adds a warning",
        ],
    },
    {
        "id": "claude_permission_modes",
        "system": "claude_code_auto",
        "url": "https://code.claude.com/docs/en/permission-modes",
        "claim": "Claude Code exposes auto and bypass permission modes.",
        "patterns": [
            r"auto.*appears when your account meets the auto mode requirements",
            r"bypassPermissions",
            r"--dangerously-skip-permissions",
        ],
    },
    {
        "id": "claude_permissions",
        "system": "claude_code_auto",
        "url": "https://code.claude.com/docs/en/permissions",
        "claim": "Claude Code permission rules use deny/ask/allow precedence and hooks cannot loosen denies.",
        "patterns": [
            r"deny, then ask, then allow",
            r"Deny and ask rules are evaluated regardless of what a PreToolUse hook returns",
            r"If a tool is denied at any level, no other level can allow it",
        ],
    },
    {
        "id": "codex_sandbox",
        "system": "codex_workspace_sandbox",
        "url": "https://developers.openai.com/codex/concepts/sandboxing",
        "claim": "Codex uses sandbox boundaries together with approval flow.",
        "patterns": [
            r"Sandboxing and approvals are different controls that work together",
            r"filesystem and network boundaries",
            r"danger-full-access",
        ],
    },
    {
        "id": "codex_permissions",
        "system": "codex_workspace_sandbox",
        "url": "https://developers.openai.com/codex/permissions",
        "claim": "Codex permission profiles configure filesystem and network policy with deny precedence.",
        "patterns": [
            r"permissions\.<name>\.filesystem",
            r"permissions\.<name>\.network",
            r"deny.*takes precedence",
        ],
    },
    {
        "id": "copilot_firewall",
        "system": "github_copilot_cloud_agent",
        "url": "https://docs.github.com/en/enterprise-cloud@latest/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/customize-the-agent-firewall",
        "claim": "Copilot cloud agent internet access is limited by a firewall by default.",
        "patterns": [
            r"access to the internet is limited by a firewall",
            r"data exfiltration risks",
            r"recommended allowlist",
        ],
    },
    {
        "id": "copilot_agents",
        "system": "github_copilot_cloud_agent",
        "url": "https://docs.github.com/en/copilot/responsible-use/agents",
        "claim": "Copilot coding agent runs in an ephemeral, firewalled execution environment.",
        "patterns": [
            r"Ephemeral, firewalled execution",
            r"ephemeral development environment",
            r"firewall is enabled by default",
        ],
    },
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_url_once(url: str, timeout_s: float) -> tuple[str, str, int | None]:
    req = urllib.request.Request(url, headers={"User-Agent": "fak-permission-source-audit/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.read().decode("utf-8", errors="replace"), "", int(resp.status)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return body, f"HTTP {int(exc.code)}", int(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return "", str(exc), None


def retryable_fetch_error(error: str, http_status: int | None) -> bool:
    if not error:
        return False
    if http_status in {408, 425, 429}:
        return True
    if http_status is not None:
        return 500 <= http_status <= 599
    return True


def fetch_url(url: str, timeout_s: float, retries: int = 2, retry_sleep_s: float = 0.5) -> tuple[str, str]:
    attempts = max(1, retries + 1)
    errors: list[str] = []
    for attempt in range(attempts):
        body, error, http_status = fetch_url_once(url, timeout_s)
        if not error:
            return body, ""
        errors.append(error)
        if attempt == attempts - 1 or not retryable_fetch_error(error, http_status):
            if len(errors) == 1:
                return body, error
            return body, f"{error} after {len(errors)} attempts; previous errors: {' | '.join(errors[:-1])}"
        if retry_sleep_s > 0:
            time.sleep(retry_sleep_s)
    return "", "unreachable fetch retry state"


def pattern_found(body: str, pattern: str) -> bool:
    without_tags = re.sub(r"<[^>]+>", " ", body)
    compact = re.sub(r"\s+", " ", html.unescape(without_tags))
    return re.search(pattern, compact, flags=re.IGNORECASE) is not None


def audit_source(source: dict[str, Any], timeout_s: float, retries: int = 2, retry_sleep_s: float = 0.5) -> dict[str, Any]:
    body, error = fetch_url(source["url"], timeout_s, retries=retries, retry_sleep_s=retry_sleep_s)
    checks = [
        {"pattern": pattern, "matched": bool(body) and pattern_found(body, pattern)}
        for pattern in source["patterns"]
    ]
    passed = bool(body) and not error and all(c["matched"] for c in checks)
    return {
        "version": fleet_version.app_version(),
        "id": source["id"],
        "system": source["system"],
        "url": source["url"],
        "claim": source["claim"],
        "status": "VERIFIED" if passed else "FAILED",
        "http_error": error,
        "body_sha256": hashlib.sha256(body.encode("utf-8", errors="replace")).hexdigest() if body else "",
        "body_bytes": len(body.encode("utf-8", errors="replace")) if body else 0,
        "checks": checks,
    }


def build_report(
    sources: list[dict[str, Any]] | None = None,
    timeout_s: float = 15.0,
    retries: int = 2,
    retry_sleep_s: float = 0.5,
) -> dict[str, Any]:
    sources = sources or SOURCES
    app_ver = fleet_version.app_version()
    rows = [
        audit_source(source, timeout_s=timeout_s, retries=retries, retry_sleep_s=retry_sleep_s)
        for source in sources
    ]
    failed = [row for row in rows if row["status"] != "VERIFIED"]
    return {
        "schema": SCHEMA,
        "app_version": app_ver,
        "generated_at": utc_now(),
        "scope_note": "Network source audit for external permission-system benchmark claims.",
        "summary": {
            "sources": len(rows),
            "verified": len(rows) - len(failed),
            "failed": len(failed),
            "source_audit_gate": len(rows) > 0 and not failed,
        },
        "sources": rows,
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# Permission-System Source Audit",
        "",
        "> Live source verification for external permission-system benchmark claims.",
        "",
        "## Summary",
        "",
        f"- Sources verified: {s['verified']}/{s['sources']}",
        f"- Source audit gate: {'yes' if s['source_audit_gate'] else 'no'}",
        "",
        "| source | system | status | URL |",
        "|---|---|---|---|",
    ]
    for row in report["sources"]:
        lines.append(f"| `{row['id']}` | `{row['system']}` | {row['status']} | {row['url']} |")
    lines.append("")
    return "\n".join(lines)


def write_text(path: str, body: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify external permission-system benchmark sources")
    ap.add_argument("--out", default="", help="write JSON report here")
    ap.add_argument("--markdown", default="", help="write Markdown report here")
    ap.add_argument("--timeout-s", type=float, default=15.0)
    ap.add_argument("--retries", type=int, default=2, help="retry transient fetch failures this many times")
    ap.add_argument("--retry-sleep-s", type=float, default=0.5, help="seconds to sleep between retries")
    args = ap.parse_args(argv)

    report = build_report(timeout_s=args.timeout_s, retries=args.retries, retry_sleep_s=args.retry_sleep_s)
    body = json.dumps(report, indent=2) + "\n"
    if args.out:
        write_text(args.out, body)
    else:
        print(body, end="")
    if args.markdown:
        write_text(args.markdown, markdown(report))
    return 0 if report["summary"]["source_audit_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
