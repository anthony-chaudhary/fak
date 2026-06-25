#!/usr/bin/env python3
"""Hermetic tests for tools/guard_mcp_status_audit.py."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "guard_mcp_status_audit.py"


def load():
    spec = importlib.util.spec_from_file_location("guard_mcp_status_audit", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_json(root: Path, rel: str, data: dict) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def seed_tree(
    root: Path,
    *,
    post_gate_git_write: bool = False,
    codex_continued: bool = True,
    agents_output_ok: bool = True,
    openai_prereq_ok: bool = True,
    openai_hosted_ok: bool = True,
    claude_historical_ok: bool = True,
) -> None:
    mod = load()
    (root / mod.STATUS_PACKET).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.STATUS_PACKET).write_text(
        "\n".join(
            [
                mod.CODEX_DOS_AUDIT,
                mod.CLAUDE_HISTORICAL,
                mod.CLAUDE_HISTORICAL_MD,
                mod.CLAUDE_LIVE,
                mod.CODEX_MCP_LIVE,
                mod.OPENAI_AGENTS_OUTPUT,
                mod.OPENAI_PREREQ_JSON,
                mod.OPENAI_PREREQ_MD,
                mod.OPENAI_HOSTED_JSON,
                mod.OPENAI_HOSTED_MD,
                "actionability.status=PASS",
                "post-gate lens shows no `git_write`",
                "BLOCKED_ENV",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / mod.GUARD_TEST).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.GUARD_TEST).write_text("\n".join(mod.GUARD_TESTS), encoding="utf-8")

    write_json(
        root,
        mod.CODEX_DOGFOOD,
        {
            "checks": {
                "mcp_stdio_adjudication": {
                    "status": "PASS",
                    "missing_tools": [],
                    "denies_publish": {"kind": "DENY", "reason": "POLICY_BLOCK"},
                    "allows_status": {"kind": "ALLOW"},
                }
            }
        },
    )
    for tool, (rel, reason) in mod.GIT_GATES.items():
        write_json(
            root,
            rel,
            {
                "tool": tool,
                "status": "DENIED_EXPECTED",
                "expect_deny": True,
                "executed": False,
                "preflight": {"verdict": "DENY", "reason": reason},
            },
        )
    families = {"python_script": 1}
    if post_gate_git_write:
        families["git_write"] = 1
    write_json(
        root,
        mod.CODEX_DOS_AUDIT,
        {
            "status": "WARN",
            "actionability": {
                "status": "PASS",
                "residual": sorted(mod.RESIDUALS),
            },
            "git_gate_evidence": {
                "status": "PASS",
                "post_gate_command_shapes": {"shell_family_counts": families},
            },
        },
    )
    write_json(
        root,
        mod.CLAUDE_LIVE,
        {
            "status": "PASS",
            "session": {"claude_session_id": "s1"},
            "dangerous_attempt": {"fak_audit": {"verdict": "DENY", "reason": "POLICY_BLOCK"}},
            "useful_continuation": {
                "same_claude_session_id": "s1",
                "fak_audit": {"verdict": "ALLOW"},
                "final_message": {"useful_completed": True},
            },
        },
    )
    historical_verdicts = {"ALLOW": 35, "DENY": 3}
    historical_reasons = {"DEFAULT_DENY": 2, "POLICY_BLOCK": 1}
    if not claude_historical_ok:
        historical_verdicts = {"ALLOW": 38}
        historical_reasons = {}
    write_json(
        root,
        mod.CLAUDE_HISTORICAL,
        {
            "schema": "fak-claude-historical-guard-audit/1",
            "status": "PASS",
            "sessions_discovered": 10,
            "sessions_audited": 6,
            "tool_calls_seen": 39,
            "unique_tool_calls_replayed": 38,
            "truncated": False,
            "verdict_counts": historical_verdicts,
            "reason_counts": historical_reasons,
            "privacy": {
                "dropped": ["prompts", "tool arguments", "tool results", "raw transcript text"],
            },
        },
    )
    (root / mod.CLAUDE_HISTORICAL_MD).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.CLAUDE_HISTORICAL_MD).write_text(
        "# Claude Code historical guard replay\n\n"
        "- status: **`PASS`**\n\n"
        "It never writes prompts, tool arguments, tool results, or raw transcript text.\n",
        encoding="utf-8",
    )
    write_json(
        root,
        mod.CODEX_MCP_LIVE,
        {
            "status": "PASS",
            "dangerous_attempt": {"fak_verdict": {"kind": "DENY", "reason": "POLICY_BLOCK"}},
            "useful_continuation": {"fak_verdict": {"kind": "ALLOW"}},
            "final_message": {"denied_attempt": True, "useful_continued": codex_continued},
        },
    )
    (root / mod.OPENAI_AGENTS_DEMO).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.OPENAI_AGENTS_DEMO).write_text("print('demo')\n", encoding="utf-8")
    agents_text = (
        "input guardrail blocks git_push: behavior=reject_content verdict=DENY reason=POLICY_BLOCK executed=false\n"
        "input guardrail allows git_status: behavior=allow verdict=ALLOW reason= executed=true\n"
        "output guardrail admits git_status result: behavior=allow verdict=DEFER reason=\n"
        "output guardrail quarantines web_fetch result: behavior=reject_content verdict=QUARANTINE reason=SECRET_EXFIL\n"
        "summary: PASS - denied call did not run, clean call ran, poisoned result was quarantined\n"
    )
    if not agents_output_ok:
        agents_text = "summary: FAIL\n"
    (root / mod.OPENAI_AGENTS_OUTPUT).write_text(agents_text, encoding="utf-8")
    prereq_blockers = [
        "OPENAI_API_KEY is not set",
        "openai-agents distribution is not installed",
        "importable agents module is not an installed OpenAI Agents SDK distribution",
    ]
    if not openai_prereq_ok:
        prereq_blockers = ["OPENAI_API_KEY is not set"]
    write_json(
        root,
        mod.OPENAI_PREREQ_JSON,
        {
            "schema": "fak-openai-live-prereq-audit/1",
            "status": "BLOCKED_ENV",
            "hosted_openai_ready": False,
            "agents_sdk_ready": False,
            "blockers": prereq_blockers,
            "env": {"OPENAI_API_KEY_set": False, "OPENAI_BASE_URL_set": False},
            "packages": {"openai": "2.41.0", "openai-agents": None, "agents": None},
            "modules": {"agents": {"installed": True, "file": "C:/work/job/agents/__init__.py"}},
            "privacy": {"dropped": ["OPENAI_API_KEY value", "any request/response payloads"]},
        },
    )
    (root / mod.OPENAI_PREREQ_MD).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.OPENAI_PREREQ_MD).write_text(
        "# OpenAI hosted live proof prerequisites\n\n"
        "- status: **`BLOCKED_ENV`**\n\n"
        "It never writes API key values or request payloads.\n",
        encoding="utf-8",
    )
    hosted_blockers = [
        "OPENAI_API_KEY is not set",
        "openai-agents distribution is not installed",
        "importable agents module is not an installed OpenAI Agents SDK distribution",
    ]
    if not openai_hosted_ok:
        hosted_blockers = ["OPENAI_API_KEY is not set"]
    write_json(
        root,
        mod.OPENAI_HOSTED_JSON,
        {
            "schema": "fak-openai-hosted-live-pilot/1",
            "status": "BLOCKED_ENV",
            "model": "gpt-5.5",
            "blockers": hosted_blockers,
            "prereqs": {
                "hosted_openai_ready": False,
                "agents_sdk_ready": False,
                "blockers": hosted_blockers,
            },
            "privacy": {"dropped": ["OPENAI_API_KEY value", "raw hosted OpenAI response text"]},
        },
    )
    (root / mod.OPENAI_HOSTED_MD).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.OPENAI_HOSTED_MD).write_text(
        "# OpenAI hosted live pilot\n\n"
        "- status: **`BLOCKED_ENV`**\n\n"
        "It never writes API key values or raw hosted OpenAI response text.\n",
        encoding="utf-8",
    )


class GuardMCPStatusAuditTest(unittest.TestCase):
    def test_collect_passes_on_complete_evidence(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seed_tree(root)
            payload = mod.collect(root)
            self.assertEqual(payload["status"], "PASS")
            self.assertEqual(payload["summary"]["failed"], 0)

    def test_collect_fails_on_post_gate_git_write(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seed_tree(root, post_gate_git_write=True)
            payload = mod.collect(root)
            self.assertEqual(payload["status"], "FAIL")
            details = json.dumps(payload)
            self.assertIn("git_write", details)
            self.assertIn("historical codex/dos actionability", details)

    def test_collect_fails_when_live_codex_did_not_continue(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seed_tree(root, codex_continued=False)
            payload = mod.collect(root)
            self.assertEqual(payload["status"], "FAIL")
            self.assertIn("codex mcp live pilot", json.dumps(payload))

    def test_collect_fails_when_claude_historical_has_no_denies(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seed_tree(root, claude_historical_ok=False)
            payload = mod.collect(root)
            self.assertEqual(payload["status"], "FAIL")
            self.assertIn("claude code historical replay", json.dumps(payload))

    def test_collect_fails_when_agents_adapter_output_breaks(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seed_tree(root, agents_output_ok=False)
            payload = mod.collect(root)
            self.assertEqual(payload["status"], "FAIL")
            self.assertIn("openai agents adapter proof", json.dumps(payload))

    def test_collect_fails_when_openai_prereq_blockers_are_incomplete(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seed_tree(root, openai_prereq_ok=False)
            payload = mod.collect(root)
            self.assertEqual(payload["status"], "FAIL")
            self.assertIn("openai hosted live prereqs", json.dumps(payload))

    def test_collect_fails_when_openai_hosted_pilot_blockers_are_incomplete(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seed_tree(root, openai_hosted_ok=False)
            payload = mod.collect(root)
            self.assertEqual(payload["status"], "FAIL")
            self.assertIn("openai hosted live pilot", json.dumps(payload))

    def test_collect_accepts_openai_hosted_pilot_pass_artifact(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seed_tree(root)
            write_json(
                root,
                mod.OPENAI_HOSTED_JSON,
                {
                    "schema": "fak-openai-hosted-live-pilot/1",
                    "status": "PASS",
                    "model": "gpt-5.5",
                    "prereqs": {"hosted_openai_ready": True, "agents_sdk_ready": False},
                    "guard": {
                        "status": "PASS",
                        "dangerous_attempt": {
                            "verdict": {"kind": "DENY", "reason": "POLICY_BLOCK"},
                            "executed": False,
                        },
                        "useful_continuation": {
                            "verdict": {"kind": "ALLOW"},
                            "admit_verdict": {"kind": "DEFER"},
                        },
                    },
                    "hosted_openai": {
                        "status": "PASS",
                        "model": "gpt-5.5",
                        "response_id_present": True,
                        "contains_expected_marker": True,
                        "output_text_sha256": "abc123",
                        "output_text_len": 21,
                    },
                },
            )
            (root / mod.OPENAI_HOSTED_MD).write_text(
                "# OpenAI hosted live pilot\n\n- status: **`PASS`**\n",
                encoding="utf-8",
            )
            payload = mod.collect(root)
            self.assertEqual(payload["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
