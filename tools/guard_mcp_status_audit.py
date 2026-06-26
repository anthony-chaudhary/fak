#!/usr/bin/env python3
"""Audit the guard/MCP proof packet against current evidence artifacts.

This is the machine-checkable companion to
experiments/agent-live/GUARD-MCP-STATUS-2026-06-25.md. It does not prove the
whole product; it verifies that the status packet's load-bearing claims still
match repo evidence:

* guard default-floor/default-journal tests are present;
* MCP stdio denies git_push and allows git_status;
* the historical Codex/DOS audit states its actionability after structured git gates;
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
REQUIRED_CODEX_RESIDUALS = {
    "HISTORICAL_GIT_WRITE_BEFORE_STRUCTURED_GATE",
}
RESIDUALS = REQUIRED_CODEX_RESIDUALS


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def top_counts(value: Any, limit: int = 8) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    rows = [(str(k), safe_int(v)) for k, v in value.items()]
    rows = [(k, v) for k, v in rows if v]
    rows.sort(key=lambda item: (-item[1], item[0]))
    return {k: v for k, v in rows[:limit]}


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


def summarize_stop_sessions(rows: Any, limit: int = 5) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        transcript = row.get("transcript") if isinstance(row.get("transcript"), dict) else {}
        shape = row.get("transcript_summary") if isinstance(row.get("transcript_summary"), dict) else {}
        out.append({
            "session_id": row.get("session_id"),
            "marker_path": row.get("marker_path"),
            "total": safe_int(row.get("total")),
            "consecutive": safe_int(row.get("consecutive")),
            "age_seconds": safe_int(row.get("age_seconds")),
            "origin": row.get("origin"),
            "settlement_action": row.get("settlement_action"),
            "transcript_status": transcript.get("status") or row.get("transcript_status"),
            "transcript_project": transcript.get("project") or row.get("transcript_project"),
            "evidence_tags": [str(tag) for tag in (shape.get("evidence_tags") or row.get("transcript_evidence_tags") or [])],
        })
        if len(out) >= limit:
            break
    return out


def settlement_plan_rows(plan: Any, actions: list[str], limit: int = 5) -> list[dict[str, Any]]:
    if not isinstance(plan, dict):
        return []
    out: list[dict[str, Any]] = []
    for action in actions:
        rows = plan.get(action)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            out.extend(summarize_stop_sessions([row], limit=1))
            if len(out) >= limit:
                return out
    return out


def summarize_friction_sessions(rows: Any, limit: int = 5) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append({
            "session_digest": row.get("session_digest"),
            "root_label": row.get("root_label"),
            "tool_calls": safe_int(row.get("tool_calls")),
            "marker_lines": safe_int(row.get("marker_lines")),
            "max_result_chars": safe_int(row.get("max_result_chars")),
            "evidence_tags": [str(tag) for tag in (row.get("evidence_tags") or [])],
        })
        if len(out) >= limit:
            break
    return out


def add_blocker(blockers: list[dict[str, Any]], *, rank: int, code: str, surface: str, status: str, evidence: dict[str, Any], next_action: str) -> None:
    blockers.append({
        "rank": rank,
        "code": code,
        "surface": surface,
        "status": status,
        "evidence": evidence,
        "next_action": next_action,
    })


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
        "actionability.status=",
        "post-gate lens shows no `git_write`",
        "Default-On Blocker Queue",
        "WORKSPACE_RECENT_STOPFAILURE_API_WALL",
        "WORKSPACE_STALE_STOPFAILURE_MARKERS",
        "CLAUDE_ALL_ACCOUNT_OPERATIONAL_FRICTION",
        "codex_login",
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
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    residual = set(action.get("residual") or [])
    reasons = set(action.get("reasons") or [])
    post_repair_shapes = action.get("post_repair_shell_shape_counts") if isinstance(action.get("post_repair_shell_shape_counts"), dict) else {}
    active_consecutive = int(summary.get("workspace_stop_failure_active_consecutive_total") or 0)
    stop_failure_active = active_consecutive > 0
    actionability_ok = (
        (
            action.get("status") == "WARN"
            and stop_failure_active
            and any("StopFailure API-wall" in reason for reason in reasons)
        )
        or (
            action.get("status") == "PASS"
            and not stop_failure_active
            and not any("StopFailure API-wall" in reason for reason in reasons)
        )
    )
    ok = (
        report.get("status") == "WARN"
        and actionability_ok
        and git_gate.get("status") == "PASS"
        and "git_write" not in post_gate_families
        and int(summary.get("workspace_stop_failures_total") or 0) > 0
        and int(post_repair_shapes.get("shell_no_write_target_detected") or 0) > 0
        and REQUIRED_CODEX_RESIDUALS.issubset(residual)
    )
    add(
        checks,
        "historical codex/dos actionability",
        ok,
        "audit="
        f"{report.get('status')} actionability={action.get('status')} "
        f"workspace_stop_failures={summary.get('workspace_stop_failures_total')} "
        f"active_consecutive={summary.get('workspace_stop_failure_active_consecutive_total')} "
        f"reasons={sorted(reasons)} residual={sorted(residual)} post_gate={post_gate_families}",
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
    shape = report.get("transcript_shape") if isinstance(report.get("transcript_shape"), dict) else {}
    tags = shape.get("evidence_tag_counts") if isinstance(shape.get("evidence_tag_counts"), dict) else {}
    top_friction = report.get("top_friction_sessions") if isinstance(report.get("top_friction_sessions"), list) else []
    serialized = json.dumps(report, sort_keys=True) + md
    leaked_payload = any(token in serialized for token in ["rm -rf", "README.md", "tool_result", "secret result", "C:\\Users\\", "C:/Users/"])
    ok = (
        report.get("schema") == "fak-claude-historical-guard-audit/1"
        and report.get("status") == "PASS"
        and report.get("all_accounts") is True
        and int(len(report.get("root_labels") or [])) >= 2
        and int(report.get("sessions_discovered") or 0) >= 1
        and int(report.get("sessions_audited") or 0) >= 1
        and int(report.get("tool_calls_seen") or 0) >= 1
        and int(report.get("unique_tool_calls_replayed") or 0) >= 1
        and verdicts.get("ALLOW", 0) >= 1
        and verdicts.get("DENY", 0) >= 1
        and reasons.get("POLICY_BLOCK", 0) >= 1
        and int(shape.get("summarized_sessions") or 0) >= int(report.get("sessions_discovered") or 0)
        and int(tags.get("HOOK_OR_API_WALL_FEEDBACK") or 0) >= 1
        and int(tags.get("HOST_PERMISSION_INTERRUPT") or 0) >= 1
        and len(top_friction) >= 1
        and report.get("truncated") is False
        and "status: **`PASS`**" in md
        and "Transcript Friction Signals" in md
        and "It never writes prompts, tool arguments, tool results, or raw transcript text." in md
        and not leaked_payload
    )
    add(
        checks,
        "claude code historical replay",
        ok,
        f"status={report.get('status')} sessions={report.get('sessions_audited')} calls={report.get('tool_calls_seen')} verdicts={verdicts} tags={tags} leaked_payload={leaked_payload}",
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
    serialized = json.dumps(report, sort_keys=True) + md
    secret_leak = (
        "sk-" in serialized
        or "OPENAI_API_KEY_value" in serialized
        or "access_token_value" in serialized
        or "refresh_token_value" in serialized
        or "id_token_value" in serialized
    )
    if report.get("codex_login_ready") is True:
        auth_sources = report.get("auth_sources") if isinstance(report.get("auth_sources"), dict) else {}
        ok = (
            report.get("schema") == "fak-openai-live-prereq-audit/1"
            and report.get("status") in {"PARTIAL", "READY"}
            and report.get("hosted_openai_ready") is True
            and auth_sources.get("codex_login") is True
            and "OPENAI_API_KEY is not set" not in blockers
            and f"status: **`{report.get('status')}`**" in md
            and "It never writes API key values" in md
            and "Codex token values" in md
            and not secret_leak
        )
    else:
        required_blockers = {
            "OPENAI_API_KEY is not set",
            "openai-agents distribution is not installed",
            "importable agents module is not an installed OpenAI Agents SDK distribution",
        }
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
        auth_source = hosted.get("auth_source") or report.get("auth_source")
        hosted_proof_ok = False
        if auth_source == "codex_login":
            hosted_proof_ok = (
                hosted.get("codex_exec_exit_code") == 0
                and hosted.get("contains_expected_marker") is True
                and "output_text_sha256" in hosted
            )
        elif auth_source == "platform_api_key":
            hosted_proof_ok = (
                hosted.get("response_id_present") is True
                and hosted.get("contains_expected_marker") is True
                and "output_text_sha256" in hosted
            )
        ok = (
            report.get("schema") == "fak-openai-hosted-live-pilot/1"
            and guard.get("status") == "PASS"
            and hosted.get("status") == "PASS"
            and ((danger.get("verdict") or {}).get("kind") == "DENY")
            and ((danger.get("verdict") or {}).get("reason") == "POLICY_BLOCK")
            and danger.get("executed") is False
            and ((useful.get("verdict") or {}).get("kind") == "ALLOW")
            and hosted_proof_ok
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


def synthesize_default_blockers(root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    codex = read_json(root, CODEX_DOS_AUDIT, checks)
    claude = read_json(root, CLAUDE_HISTORICAL, checks)
    prereq = read_json(root, OPENAI_PREREQ_JSON, checks)
    blockers: list[dict[str, Any]] = []

    codex_summary = codex.get("summary") if isinstance(codex.get("summary"), dict) else {}
    workspace_stop = codex.get("workspace_stop_failures") if isinstance(codex.get("workspace_stop_failures"), dict) else {}
    active_consecutive = safe_int(codex_summary.get("workspace_stop_failure_active_consecutive_total"))
    active_markers = safe_int(codex_summary.get("workspace_stop_failure_active_markers"))
    recent_active_consecutive = safe_int(codex_summary.get("workspace_stop_failure_recent_active_consecutive_total"))
    recent_active_markers = safe_int(codex_summary.get("workspace_stop_failure_recent_active_markers"))
    stale_active_consecutive = safe_int(codex_summary.get("workspace_stop_failure_stale_active_consecutive_total"))
    stale_active_markers = safe_int(codex_summary.get("workspace_stop_failure_stale_active_markers"))
    settlement_plan = workspace_stop.get("settlement_plan")
    if recent_active_consecutive:
        add_blocker(
            blockers,
            rank=10,
            code="WORKSPACE_RECENT_STOPFAILURE_API_WALL",
            surface="workspace_dos",
            status="ACTIVE",
            evidence={
                "active_markers": active_markers,
                "active_consecutive_total": active_consecutive,
                "recent_active_markers": recent_active_markers,
                "recent_active_consecutive_total": recent_active_consecutive,
                "active_recent_threshold_hours": safe_int(codex_summary.get("workspace_stop_failure_active_recent_threshold_hours")),
                "stale_active_markers": stale_active_markers,
                "stale_active_consecutive_total": stale_active_consecutive,
                "origin_counts": top_counts(codex_summary.get("workspace_stop_failure_origin_counts")),
                "recent_active_origin_counts": top_counts(codex_summary.get("workspace_stop_failure_recent_active_origin_counts")),
                "stale_active_origin_counts": top_counts(codex_summary.get("workspace_stop_failure_stale_active_origin_counts")),
                "active_settlement_action_counts": top_counts(codex_summary.get("workspace_stop_failure_active_settlement_action_counts")),
                "recent_active_settlement_action_counts": top_counts(codex_summary.get("workspace_stop_failure_recent_active_settlement_action_counts")),
                "stale_active_settlement_action_counts": top_counts(codex_summary.get("workspace_stop_failure_stale_active_settlement_action_counts")),
                "one_day_failures_total": safe_int(codex_summary.get("workspace_stop_failures_total")),
                "healed_nonzero_markers": safe_int(codex_summary.get("workspace_stop_failure_healed_nonzero_markers")),
                "top_recent_active_sessions": summarize_stop_sessions(workspace_stop.get("top_recent_active")),
                "recent_review_plan": settlement_plan_rows(settlement_plan, ["RECENT_REVIEW"]),
                "stale_settlement_plan": settlement_plan_rows(
                    settlement_plan,
                    ["STALE_RESET_CANDIDATE", "STALE_MARKER_ONLY_ARCHIVE_CANDIDATE"],
                ),
                "transcript_evidence_tags": top_counts(codex_summary.get("workspace_stop_failure_transcript_evidence_tags")),
            },
            next_action="Clear or rotate the recent workspace sessions with nonzero consecutive StopFailure markers before treating fak-by-default actionability as healthy.",
        )
    if stale_active_consecutive:
        add_blocker(
            blockers,
            rank=40,
            code="WORKSPACE_STALE_STOPFAILURE_MARKERS",
            surface="workspace_dos",
            status="STALE_DEBT",
            evidence={
                "stale_active_markers": stale_active_markers,
                "stale_active_consecutive_total": stale_active_consecutive,
                "active_recent_threshold_hours": safe_int(codex_summary.get("workspace_stop_failure_active_recent_threshold_hours")),
                "stale_active_origin_counts": top_counts(codex_summary.get("workspace_stop_failure_stale_active_origin_counts")),
                "stale_active_settlement_action_counts": top_counts(codex_summary.get("workspace_stop_failure_stale_active_settlement_action_counts")),
                "stale_settlement_plan": settlement_plan_rows(
                    settlement_plan,
                    ["STALE_RESET_CANDIDATE", "STALE_MARKER_ONLY_ARCHIVE_CANDIDATE"],
                ),
                "top_stale_active_sessions": summarize_stop_sessions(workspace_stop.get("top_stale_active")),
            },
            next_action="Decide whether stale nonzero consecutive markers need a success reset, archival, or a hook fix so old breaker state does not masquerade as live blockage.",
        )

    action = codex.get("actionability") if isinstance(codex.get("actionability"), dict) else {}
    shell_shapes = action.get("post_repair_shell_shape_counts") if isinstance(action.get("post_repair_shell_shape_counts"), dict) else {}
    shell_families = action.get("post_repair_shell_family_counts") if isinstance(action.get("post_repair_shell_family_counts"), dict) else {}
    no_write_target = safe_int(shell_shapes.get("shell_no_write_target_detected"))
    if no_write_target:
        add_blocker(
            blockers,
            rank=20,
            code="CODEX_HOST_SHELL_OPACITY",
            surface="codex_hooks",
            status="ACTIVE_DEBT",
            evidence={
                "shell_no_write_target_detected": no_write_target,
                "shell_shape_counts": top_counts(shell_shapes),
                "shell_family_counts": top_counts(shell_families),
                "unknown_tree_warning_rate": codex_summary.get("unknown_tree_warning_rate"),
            },
            next_action="Prefer path-visible host tools or structured tool payloads so DOS can assign file-tree footprints instead of warning on opaque shell calls.",
        )

    mutating = action.get("post_repair_mutating_shell_family_counts") if isinstance(action.get("post_repair_mutating_shell_family_counts"), dict) else {}
    git_gate = codex.get("git_gate_evidence") if isinstance(codex.get("git_gate_evidence"), dict) else {}
    post_gate = git_gate.get("post_gate_command_shapes") if isinstance(git_gate.get("post_gate_command_shapes"), dict) else {}
    post_gate_families = post_gate.get("shell_family_counts") if isinstance(post_gate.get("shell_family_counts"), dict) else {}
    if safe_int(mutating.get("git_write")) and "git_write" not in post_gate_families:
        add_blocker(
            blockers,
            rank=60,
            code="HISTORICAL_OPAQUE_GIT_WRITE_BEFORE_GATE",
            surface="codex_hooks",
            status="HISTORICAL",
            evidence={
                "post_repair_mutating_shell_family_counts": top_counts(mutating),
                "git_gate_status": git_gate.get("status"),
                "proved_at": git_gate.get("proved_at"),
                "post_gate_shell_family_counts": top_counts(post_gate_families),
            },
            next_action="Keep structured git gates in place; this is not current actionability unless git_write reappears after the gate proof timestamp.",
        )

    claude_shape = claude.get("transcript_shape") if isinstance(claude.get("transcript_shape"), dict) else {}
    claude_tags = claude_shape.get("evidence_tag_counts") if isinstance(claude_shape.get("evidence_tag_counts"), dict) else {}
    marker_lines = claude_shape.get("marker_line_counts") if isinstance(claude_shape.get("marker_line_counts"), dict) else {}
    if claude_tags:
        add_blocker(
            blockers,
            rank=30,
            code="CLAUDE_ALL_ACCOUNT_OPERATIONAL_FRICTION",
            surface="claude_code",
            status="ACTIVE_DEBT",
            evidence={
                "root_count": len(claude.get("root_labels") or []),
                "sessions_discovered": safe_int(claude.get("sessions_discovered")),
                "sessions_audited": safe_int(claude.get("sessions_audited")),
                "tool_calls_seen": safe_int(claude.get("tool_calls_seen")),
                "verdict_counts": top_counts(claude.get("verdict_counts")),
                "reason_counts": top_counts(claude.get("reason_counts")),
                "evidence_tag_counts": top_counts(claude_tags),
                "marker_line_counts": top_counts(marker_lines),
                "max_result_chars": safe_int(claude_shape.get("max_result_chars")),
                "top_friction_sessions": summarize_friction_sessions(claude.get("top_friction_sessions")),
            },
            next_action="Triage the top Claude friction sessions by tag: hook/API-wall and permission interruptions first, then large-result and shell-heavy sessions.",
        )

    prereq_blockers = [str(item) for item in (prereq.get("blockers") or [])]
    if "openai-agents distribution is not installed" in prereq_blockers:
        add_blocker(
            blockers,
            rank=70,
            code="OPENAI_AGENTS_SDK_NOT_INSTALLED",
            surface="openai_hosted",
            status="EXTERNAL_PREREQ",
            evidence={
                "status": prereq.get("status"),
                "codex_login_ready": prereq.get("codex_login_ready"),
                "hosted_openai_ready": prereq.get("hosted_openai_ready"),
                "blockers": prereq_blockers,
            },
            next_action="Install the OpenAI Agents SDK only when the hosted Agents path is the target; Codex-login hosted pilot already passes without it.",
        )

    blockers.sort(key=lambda row: (safe_int(row.get("rank")), str(row.get("code") or "")))
    return blockers


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
    default_blockers = synthesize_default_blockers(root)
    return {
        "schema": SCHEMA,
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failures": failed,
        "default_blockers": default_blockers,
        "summary": {
            "passed": len(checks) - len(failed),
            "failed": len(failed),
            "total": len(checks),
            "default_blockers": len(default_blockers),
            "active_default_blockers": len([b for b in default_blockers if str(b.get("status") or "").startswith("ACTIVE")]),
        },
    }


def render(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [f"guard-mcp-status: {payload['status']} ({s['passed']}/{s['total']} checks passed)"]
    for check in payload["checks"]:
        mark = "OK" if check["status"] == "PASS" else "XX"
        lines.append(f"  [{mark}] {check['name']}: {check['detail']}")
    blockers = payload.get("default_blockers") if isinstance(payload.get("default_blockers"), list) else []
    if blockers:
        lines.append("  default-on blocker queue:")
        for row in blockers:
            evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
            bits = []
            if "recent_active_consecutive_total" in evidence:
                bits.append(f"recent_consecutive={evidence.get('recent_active_consecutive_total')}")
            if "active_consecutive_total" in evidence:
                bits.append(f"active_consecutive={evidence.get('active_consecutive_total')}")
            if "stale_active_consecutive_total" in evidence:
                bits.append(f"stale_consecutive={evidence.get('stale_active_consecutive_total')}")
            if "recent_active_origin_counts" in evidence:
                bits.append(f"recent_origins={evidence.get('recent_active_origin_counts')}")
            if "stale_active_origin_counts" in evidence:
                bits.append(f"stale_origins={evidence.get('stale_active_origin_counts')}")
            if "active_settlement_action_counts" in evidence:
                bits.append(f"settlement={evidence.get('active_settlement_action_counts')}")
            elif "stale_active_settlement_action_counts" in evidence:
                bits.append(f"settlement={evidence.get('stale_active_settlement_action_counts')}")
            if "recent_review_plan" in evidence:
                bits.append(f"recent_plan={len(evidence.get('recent_review_plan') or [])}")
            if "stale_settlement_plan" in evidence:
                bits.append(f"stale_plan={len(evidence.get('stale_settlement_plan') or [])}")
            if "shell_no_write_target_detected" in evidence:
                bits.append(f"opaque_shell={evidence.get('shell_no_write_target_detected')}")
            if "evidence_tag_counts" in evidence:
                bits.append(f"tags={evidence.get('evidence_tag_counts')}")
            if "blockers" in evidence:
                bits.append(f"blockers={evidence.get('blockers')}")
            lines.append(
                f"    [{row.get('status')}] {row.get('code')} "
                f"surface={row.get('surface')} {' '.join(bits)}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", type=Path, help="write the machine-readable status audit JSON to this path")
    args = p.parse_args(argv)
    payload = collect(args.root.resolve())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render(payload))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
