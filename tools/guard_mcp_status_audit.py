#!/usr/bin/env python3
"""Audit the guard/MCP proof packet against current evidence artifacts.

This is the machine-checkable companion to
experiments/agent-live/GUARD-MCP-STATUS-2026-06-25.md. It does not prove the
whole product; it verifies that the status packet's load-bearing claims still
match repo evidence:

* guard default-floor/default-journal tests are present;
* MCP stdio denies git_push and allows git_status;
* the historical Codex/DOS audit is actionable-PASS after structured git gates;
* the post-git-gate lens has no opaque git_write;
* Claude Code and Codex MCP live pilot artifacts show deny + useful continuation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "fak-guard-mcp-status-audit/1"

STATUS_PACKET = "experiments/agent-live/GUARD-MCP-STATUS-2026-06-25.md"
CODEX_DOGFOOD = "experiments/agent-live/codex-dogfood-019efde3-6794-7401-93a1-e97e6bd72a9c.json"
CODEX_DOS_AUDIT = "experiments/agent-live/codex-dos-recent-audit.json"
CLAUDE_HISTORICAL = "experiments/agent-live/claude-historical-guard-audit-2026-06-25.json"
CLAUDE_HISTORICAL_MD = "experiments/agent-live/CLAUDE-HISTORICAL-GUARD-AUDIT-2026-06-25.md"
CLAUDE_LIVE = "experiments/agent-live/claude-code-fak-guard-live-pilot-2026-06-25.json"
CODEX_MCP_LIVE = "experiments/agent-live/codex-mcp-fak-live-pilot-2026-06-25.json"
OPENAI_AGENTS_OUTPUT = "examples/openai-agents-guardrail/EXAMPLE-OUTPUT.md"
OPENAI_AGENTS_DEMO = "examples/openai-agents-guardrail/demo.py"
OPENAI_PREREQ_JSON = "experiments/agent-live/openai-live-prereq-2026-06-25.json"
OPENAI_PREREQ_MD = "experiments/agent-live/OPENAI-LIVE-PREREQ-2026-06-25.md"
OPENAI_HOSTED_JSON = "experiments/agent-live/openai-hosted-live-pilot-2026-06-25.json"
OPENAI_HOSTED_MD = "experiments/agent-live/OPENAI-HOSTED-LIVE-PILOT-2026-06-25.md"
GUARD_TEST = "cmd/fak/guard_test.go"
GIT_GATES = {
    "git_add": ("experiments/agent-live/codex-fak-gate-git-add.json", "DEFAULT_DENY"),
    "git_commit": ("experiments/agent-live/codex-fak-gate-git-commit.json", "DEFAULT_DENY"),
    "git_push": ("experiments/agent-live/codex-fak-gate-git-push.json", "POLICY_BLOCK"),
}
GUARD_TESTS = [
    "TestGuardDefaultPolicyDeniesDangerAllowsBenign",
    "TestGuardAuditPlan",
    "TestGuardDefaultAuditPath",
    "TestGuardEnableAuditEnablesVerifiableTrail",
]
RESIDUALS = {
    "HISTORICAL_GIT_WRITE_BEFORE_STRUCTURED_GATE",
    "HOST_SHELL_OPACITY",
    "UNKNOWN_TREE_WARNINGS",
}


def read_json(root: Path, rel: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    path = root / rel
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        checks.append({"name": rel, "status": "FAIL", "detail": str(exc)})
        return {}
    if not isinstance(data, dict):
        checks.append({"name": rel, "status": "FAIL", "detail": "not a JSON object"})
        return {}
    return data


def add(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})


def verdict_kind(row: dict[str, Any]) -> str:
    if isinstance(row.get("fak_verdict"), dict):
        return str(row["fak_verdict"].get("kind") or "")
    if isinstance(row.get("fak_audit"), dict):
        return str(row["fak_audit"].get("verdict") or "")
    return str(row.get("verdict") or "")


def verdict_reason(row: dict[str, Any]) -> str:
    if isinstance(row.get("fak_verdict"), dict):
        return str(row["fak_verdict"].get("reason") or "")
    if isinstance(row.get("fak_audit"), dict):
        return str(row["fak_audit"].get("reason") or "")
    return str(row.get("reason") or "")


def check_guard_tests(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / GUARD_TEST
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        add(checks, "guard default tests present", False, str(exc))
        return
    missing = [name for name in GUARD_TESTS if name not in text]
    add(
        checks,
        "guard default tests present",
        not missing,
        "all required guard default tests are present" if not missing else f"missing {missing}",
    )


def check_status_packet(root: Path, checks: list[dict[str, Any]]) -> None:
    path = root / STATUS_PACKET
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        add(checks, "status packet present", False, str(exc))
        return
    required = [
        CODEX_DOS_AUDIT,
        CLAUDE_HISTORICAL,
        CLAUDE_HISTORICAL_MD,
        CLAUDE_LIVE,
        CODEX_MCP_LIVE,
        OPENAI_AGENTS_OUTPUT,
        OPENAI_PREREQ_JSON,
        OPENAI_PREREQ_MD,
        OPENAI_HOSTED_JSON,
        OPENAI_HOSTED_MD,
        "actionability.status=PASS",
        "post-gate lens shows no `git_write`",
        "BLOCKED_ENV",
    ]
    missing = [item for item in required if item not in text]
    add(
        checks,
        "status packet present",
        not missing,
        "status packet names the evidence and residual interpretation" if not missing else f"missing {missing}",
    )


def check_mcp_stdio(root: Path, checks: list[dict[str, Any]]) -> None:
    dogfood = read_json(root, CODEX_DOGFOOD, checks)
    mcp = (((dogfood.get("checks") or {}).get("mcp_stdio_adjudication") or {})
           if isinstance(dogfood.get("checks"), dict) else {})
    deny = mcp.get("denies_publish") if isinstance(mcp.get("denies_publish"), dict) else {}
    allow = mcp.get("allows_status") if isinstance(mcp.get("allows_status"), dict) else {}
    missing = mcp.get("missing_tools") if isinstance(mcp.get("missing_tools"), list) else []
    ok = (
        mcp.get("status") == "PASS"
        and not missing
        and deny.get("kind") == "DENY"
        and deny.get("reason") == "POLICY_BLOCK"
        and allow.get("kind") == "ALLOW"
    )
    add(checks, "mcp stdio adjudication", ok, f"status={mcp.get('status')} deny={deny} allow={allow}")


def check_git_gate_reports(root: Path, checks: list[dict[str, Any]]) -> None:
    for tool, (rel, reason) in GIT_GATES.items():
        report = read_json(root, rel, checks)
        preflight = report.get("preflight") if isinstance(report.get("preflight"), dict) else {}
        ok = (
            report.get("tool") == tool
            and report.get("status") == "DENIED_EXPECTED"
            and report.get("expect_deny") is True
            and report.get("executed") is False
            and preflight.get("verdict") == "DENY"
            and preflight.get("reason") == reason
        )
        add(checks, f"structured git gate {tool}", ok, f"status={report.get('status')} reason={preflight.get('reason')}")


def check_historical_audit(root: Path, checks: list[dict[str, Any]]) -> None:
    report = read_json(root, CODEX_DOS_AUDIT, checks)
    action = report.get("actionability") if isinstance(report.get("actionability"), dict) else {}
    git_gate = report.get("git_gate_evidence") if isinstance(report.get("git_gate_evidence"), dict) else {}
    post_gate = git_gate.get("post_gate_command_shapes") if isinstance(git_gate.get("post_gate_command_shapes"), dict) else {}
    post_gate_families = post_gate.get("shell_family_counts") if isinstance(post_gate.get("shell_family_counts"), dict) else {}
    residual = set(action.get("residual") or [])
    ok = (
        report.get("status") == "WARN"
        and action.get("status") == "PASS"
        and git_gate.get("status") == "PASS"
        and "git_write" not in post_gate_families
        and RESIDUALS.issubset(residual)
    )
    add(
        checks,
        "historical codex/dos actionability",
        ok,
        f"audit={report.get('status')} actionability={action.get('status')} residual={sorted(residual)} post_gate={post_gate_families}",
    )


def check_claude_historical(root: Path, checks: list[dict[str, Any]]) -> None:
    report = read_json(root, CLAUDE_HISTORICAL, checks)
    try:
        md = (root / CLAUDE_HISTORICAL_MD).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        add(checks, "claude code historical replay", False, str(exc))
        return
    verdicts = report.get("verdict_counts") if isinstance(report.get("verdict_counts"), dict) else {}
    reasons = report.get("reason_counts") if isinstance(report.get("reason_counts"), dict) else {}
    serialized = json.dumps(report, sort_keys=True) + md
    leaked_payload = any(token in serialized for token in ["rm -rf", "README.md", "tool_result", "secret result"])
    ok = (
        report.get("schema") == "fak-claude-historical-guard-audit/1"
        and report.get("status") == "PASS"
        and int(report.get("sessions_discovered") or 0) >= 1
        and int(report.get("sessions_audited") or 0) >= 1
        and int(report.get("tool_calls_seen") or 0) >= 1
        and int(report.get("unique_tool_calls_replayed") or 0) >= 1
        and verdicts.get("ALLOW", 0) >= 1
        and verdicts.get("DENY", 0) >= 1
        and reasons.get("POLICY_BLOCK", 0) >= 1
        and report.get("truncated") is False
        and "status: **`PASS`**" in md
        and "It never writes prompts, tool arguments, tool results, or raw transcript text." in md
        and not leaked_payload
    )
    add(
        checks,
        "claude code historical replay",
        ok,
        f"status={report.get('status')} sessions={report.get('sessions_audited')} calls={report.get('tool_calls_seen')} verdicts={verdicts} leaked_payload={leaked_payload}",
    )


def check_claude_live(root: Path, checks: list[dict[str, Any]]) -> None:
    report = read_json(root, CLAUDE_LIVE, checks)
    danger = report.get("dangerous_attempt") if isinstance(report.get("dangerous_attempt"), dict) else {}
    useful = report.get("useful_continuation") if isinstance(report.get("useful_continuation"), dict) else {}
    final = useful.get("final_message") if isinstance(useful.get("final_message"), dict) else {}
    same_session = useful.get("same_claude_session_id") == (report.get("session") or {}).get("claude_session_id")
    ok = (
        report.get("status") == "PASS"
        and verdict_kind(danger) == "DENY"
        and verdict_reason(danger) == "POLICY_BLOCK"
        and verdict_kind(useful) == "ALLOW"
        and same_session
        and final.get("useful_completed") is True
    )
    add(checks, "claude code live pilot", ok, f"status={report.get('status')} same_session={same_session}")


def check_codex_mcp_live(root: Path, checks: list[dict[str, Any]]) -> None:
    report = read_json(root, CODEX_MCP_LIVE, checks)
    danger = report.get("dangerous_attempt") if isinstance(report.get("dangerous_attempt"), dict) else {}
    useful = report.get("useful_continuation") if isinstance(report.get("useful_continuation"), dict) else {}
    final = report.get("final_message") if isinstance(report.get("final_message"), dict) else {}
    ok = (
        report.get("status") == "PASS"
        and verdict_kind(danger) == "DENY"
        and verdict_reason(danger) == "POLICY_BLOCK"
        and verdict_kind(useful) == "ALLOW"
        and final.get("denied_attempt") is True
        and final.get("useful_continued") is True
    )
    add(checks, "codex mcp live pilot", ok, f"status={report.get('status')} denied={final.get('denied_attempt')} continued={final.get('useful_continued')}")


def check_openai_agents_adapter(root: Path, checks: list[dict[str, Any]]) -> None:
    demo_exists = (root / OPENAI_AGENTS_DEMO).is_file()
    try:
        text = (root / OPENAI_AGENTS_OUTPUT).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        add(checks, "openai agents adapter proof", False, str(exc))
        return
    required = [
        "input guardrail blocks git_push",
        "behavior=reject_content verdict=DENY reason=POLICY_BLOCK executed=false",
        "input guardrail allows git_status",
        "output guardrail admits git_status result",
        "output guardrail quarantines web_fetch result",
        "verdict=QUARANTINE reason=SECRET_EXFIL",
        "summary: PASS",
    ]
    missing = [item for item in required if item not in text]
    add(
        checks,
        "openai agents adapter proof",
        demo_exists and not missing,
        "demo and captured output prove deny/run/quarantine mapping"
        if demo_exists and not missing
        else f"demo_exists={demo_exists} missing={missing}",
    )


def check_openai_live_prereq(root: Path, checks: list[dict[str, Any]]) -> None:
    report = read_json(root, OPENAI_PREREQ_JSON, checks)
    try:
        md = (root / OPENAI_PREREQ_MD).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        add(checks, "openai hosted live prereqs", False, str(exc))
        return
    blockers = set(report.get("blockers") or [])
    required_blockers = {
        "OPENAI_API_KEY is not set",
        "openai-agents distribution is not installed",
        "importable agents module is not an installed OpenAI Agents SDK distribution",
    }
    serialized = json.dumps(report, sort_keys=True) + md
    secret_leak = "sk-" in serialized or "OPENAI_API_KEY_value" in serialized
    ok = (
        report.get("schema") == "fak-openai-live-prereq-audit/1"
        and report.get("status") == "BLOCKED_ENV"
        and report.get("hosted_openai_ready") is False
        and report.get("agents_sdk_ready") is False
        and required_blockers.issubset(blockers)
        and "status: **`BLOCKED_ENV`**" in md
        and "It never writes API key values" in md
        and not secret_leak
    )
    add(
        checks,
        "openai hosted live prereqs",
        ok,
        f"status={report.get('status')} blockers={sorted(blockers)} secret_leak={secret_leak}",
    )


def check_openai_hosted_live_pilot(root: Path, checks: list[dict[str, Any]]) -> None:
    report = read_json(root, OPENAI_HOSTED_JSON, checks)
    try:
        md = (root / OPENAI_HOSTED_MD).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        add(checks, "openai hosted live pilot", False, str(exc))
        return
    status = report.get("status")
    serialized = json.dumps(report, sort_keys=True) + md
    secret_leak = "sk-" in serialized or "OPENAI_API_KEY_value" in serialized
    blockers = set(report.get("blockers") or [])
    prereqs = report.get("prereqs") if isinstance(report.get("prereqs"), dict) else {}
    if status == "BLOCKED_ENV":
        required_blockers = {
            "OPENAI_API_KEY is not set",
            "openai-agents distribution is not installed",
            "importable agents module is not an installed OpenAI Agents SDK distribution",
        }
        ok = (
            report.get("schema") == "fak-openai-hosted-live-pilot/1"
            and prereqs.get("hosted_openai_ready") is False
            and required_blockers.issubset(blockers)
            and "status: **`BLOCKED_ENV`**" in md
            and not secret_leak
        )
    elif status == "PASS":
        guard = report.get("guard") if isinstance(report.get("guard"), dict) else {}
        hosted = report.get("hosted_openai") if isinstance(report.get("hosted_openai"), dict) else {}
        danger = guard.get("dangerous_attempt") if isinstance(guard.get("dangerous_attempt"), dict) else {}
        useful = guard.get("useful_continuation") if isinstance(guard.get("useful_continuation"), dict) else {}
        ok = (
            report.get("schema") == "fak-openai-hosted-live-pilot/1"
            and guard.get("status") == "PASS"
            and hosted.get("status") == "PASS"
            and ((danger.get("verdict") or {}).get("kind") == "DENY")
            and ((danger.get("verdict") or {}).get("reason") == "POLICY_BLOCK")
            and danger.get("executed") is False
            and ((useful.get("verdict") or {}).get("kind") == "ALLOW")
            and hosted.get("response_id_present") is True
            and hosted.get("contains_expected_marker") is True
            and "output_text_sha256" in hosted
            and "raw hosted OpenAI response text" not in json.dumps(hosted)
            and not secret_leak
        )
    else:
        ok = False
    add(
        checks,
        "openai hosted live pilot",
        ok,
        f"status={status} blockers={sorted(blockers)} secret_leak={secret_leak}",
    )


def collect(root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    check_status_packet(root, checks)
    check_guard_tests(root, checks)
    check_mcp_stdio(root, checks)
    check_git_gate_reports(root, checks)
    check_historical_audit(root, checks)
    check_claude_historical(root, checks)
    check_claude_live(root, checks)
    check_codex_mcp_live(root, checks)
    check_openai_agents_adapter(root, checks)
    check_openai_live_prereq(root, checks)
    check_openai_hosted_live_pilot(root, checks)
    failed = [c for c in checks if c["status"] != "PASS"]
    return {
        "schema": SCHEMA,
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failures": failed,
        "summary": {
            "passed": len(checks) - len(failed),
            "failed": len(failed),
            "total": len(checks),
        },
    }


def render(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [f"guard-mcp-status: {payload['status']} ({s['passed']}/{s['total']} checks passed)"]
    for check in payload["checks"]:
        mark = "OK" if check["status"] == "PASS" else "XX"
        lines.append(f"  [{mark}] {check['name']}: {check['detail']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    payload = collect(args.root.resolve())
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render(payload))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
