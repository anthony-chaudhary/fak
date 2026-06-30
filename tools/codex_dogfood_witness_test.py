#!/usr/bin/env python3
"""Hermetic tests for tools/codex_dogfood_witness.py."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve().parent / "codex_dogfood_witness.py"


def load():
    spec = importlib.util.spec_from_file_location("codex_dogfood_witness", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeRunner:
    def __init__(self, proof_status: str = "PROVEN") -> None:
        self.proof_status = proof_status
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], cwd: Path) -> Any:
        mod = load()
        self.calls.append(argv)
        if "preflight" in argv:
            tool = argv[argv.index("--tool") + 1]
            if tool == "git_push":
                return mod.CommandResult(0, "verdict=DENY reason=POLICY_BLOCK by=monitor\n", "")
            if tool == "git_status":
                return mod.CommandResult(0, "verdict=ALLOW reason=NONE by=monitor\n", "")
            return mod.CommandResult(0, "verdict=DENY reason=DEFAULT_DENY by=monitor\n", "")

        if "prove-telemetry" in argv:
            if self.proof_status == "PROVEN":
                proof = {
                    "status": "PROVEN",
                    "requests": 2,
                    "baseline_token_equiv": 4012,
                    "actual_token_equiv": 2284,
                    "saved_token_equiv": 1728,
                    "saved_pct": 43.07078763708873,
                    "cache_read_tokens": 1920,
                    "correctness_depends_on_hit": False,
                }
                return mod.CommandResult(0, json.dumps(proof), "")
            proof = {
                "status": "REFUTED",
                "requests": 1,
                "saved_token_equiv": 0,
                "reason": "no cache reads",
                "correctness_depends_on_hit": False,
            }
            return mod.CommandResult(1, json.dumps(proof), "")

        if len(argv) >= 3 and argv[1:3] == ["exec", "--json"]:
            return mod.CommandResult(0, codex_exec_jsonl(), "")

        if len(argv) >= 2 and argv[:2] == ["dos", "helped"]:
            helped = {
                "total": 3,
                "advisory": 3,
                "warned": 3,
                "blocked": 0,
                "withheld": 0,
                "deferred": 0,
                "since": "2026-06-25T11:58:00Z",
                "latest": "2026-06-25T11:59:00Z",
                "by_rung": {"WARN": 3},
                "by_reason": {"admission": 3},
                "by_reason_rung": {"admission": {"WARN": 3}},
                "by_tool": {"Bash": 2, "apply_patch": 1},
                "by_advisory_tool": {"Bash": 2, "apply_patch": 1},
                "by_refused_reason": {},
                "examples": {
                    "admission": [
                        {
                            "target": "C:/secret/path.txt",
                            "tool": "Bash",
                            "reason": "drop example reason with target",
                        }
                    ]
                },
            }
            return mod.CommandResult(0, json.dumps(helped), "")

        return mod.CommandResult(2, "", "unexpected command")


def fake_mcp_probe(status: str = "PASS"):
    def probe(args: Any) -> dict[str, Any]:
        if status != "PASS":
            return {"status": status, "error": "simulated MCP failure"}
        return {
            "status": "PASS",
            "transport": "stdio",
            "server_info": {"name": "fak-gateway"},
            "protocol_version": "2024-11-05",
            "tools": ["fak_adjudicate", "fak_admit", "fak_syscall"],
            "expected_tools_present": ["fak_adjudicate", "fak_admit", "fak_syscall"],
            "missing_tools": [],
            "denies_publish": {"kind": "DENY", "reason": "POLICY_BLOCK", "disposition": "TERMINAL"},
            "allows_status": {"kind": "ALLOW"},
            "stderr": "",
        }

    return probe


def fixed_now() -> datetime:
    return datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc)


def codex_exec_jsonl() -> str:
    return "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "nested-thread"}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "drop final text"}}),
            json.dumps(
                {
                    "type": "item.started",
                    "item": {
                        "type": "mcp_tool_call",
                        "server": "fak",
                        "tool": "fak_adjudicate",
                        "arguments": {"tool": "git_push", "arguments": {}},
                        "result": None,
                        "status": "in_progress",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "mcp_tool_call",
                        "server": "fak",
                        "tool": "fak_adjudicate",
                        "arguments": {"tool": "git_push", "arguments": {}, "prompt": "drop prompt"},
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        {
                                            "verdict": {
                                                "kind": "DENY",
                                                "reason": "POLICY_BLOCK",
                                                "by": "monitor",
                                            },
                                            "trace_id": "gw-1",
                                        }
                                    ),
                                }
                            ]
                        },
                        "status": "completed",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "mcp_tool_call",
                        "server": "fak",
                        "tool": "fak_adjudicate",
                        "arguments": {"tool": "git_status", "arguments": {}},
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps({"verdict": {"kind": "ALLOW", "by": "monitor"}}),
                                }
                            ]
                        },
                        "status": "completed",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 51620, "cached_input_tokens": 27392, "output_tokens": 573},
                }
            ),
        ]
    )


def write_dos_fixture(
    root: Path,
    thread_id: str = "abc123",
    observations: list[dict[str, Any]] | None = None,
) -> None:
    stream = root / ".dos" / "streams" / f"{thread_id}.jsonl"
    stream.parent.mkdir(parents=True)
    stream.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "schema": {"family": "tool-stream", "version": 1},
                    "op": "STEP",
                    "step_index": 0,
                    "tool_name": "Bash",
                    "args_digest": "arg0",
                    "result_digest": "res0",
                    "ts": "2026-06-25T11:58:00Z",
                },
                {
                    "schema": {"family": "tool-stream", "version": 1},
                    "op": "STEP",
                    "step_index": 1,
                    "tool_name": "apply_patch",
                    "args_digest": "arg1",
                    "result_digest": "res1",
                    "ts": "2026-06-25T11:59:00Z",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = root / ".dos" / "metrics" / "observations.jsonl"
    metrics.parent.mkdir(parents=True)
    rows = observations or [
        {
            "schema": {"family": "hook-observation", "version": 1},
            "verb": "pretool",
            "outcome": "passthrough",
            "rung": "provenance",
            "dialect": "codex",
            "latency_ms": 2.5,
            "ts": "2026-06-25T11:58:10Z",
        },
        {
            "schema": {"family": "hook-observation", "version": 1},
            "verb": "posttool",
            "outcome": "passthrough",
            "stream_state": "ADVANCING",
            "latency_ms": 3.5,
            "ts": "2026-06-25T11:58:20Z",
        },
        {
            "schema": {"family": "hook-observation", "version": 1},
            "verb": "pretool",
            "outcome": "passthrough",
            "rung": "provenance",
            "dialect": "codex",
            "latency_ms": 4.5,
            "ts": "2026-06-25T11:58:30Z",
        },
    ]
    metrics.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    stops = root / ".dos" / "stop-failures" / f"{thread_id}.json"
    stops.parent.mkdir(parents=True)
    stops.write_text(json.dumps({"total": 0, "consecutive": 0}) + "\n", encoding="utf-8")


def write_native_hook_manifest(home: Path, *, repaired_at: datetime | None = None) -> None:
    hook_dir = home / "plugins" / "cache" / "dos" / "dos-kernel" / "0.28.0" / "hooks"
    hook_dir.mkdir(parents=True)
    launcher = hook_dir.parent / "bin" / "dos-hook"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("# native hook launcher fixture\n", encoding="utf-8")
    manifest = {
        "hooks": {
            "PreToolUse": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "shell": "bash",
                            "command": 'root="${CLAUDE_PLUGIN_ROOT:-${CODEX_PLUGIN_ROOT:-}}"; "$root/bin/dos-hook" pretool --workspace . --dialect codex',
                        }
                    ]
                }
            ]
        }
    }
    manifest_path = hook_dir / "hooks.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    if repaired_at is not None:
        backup = manifest_path.with_name("hooks.json.before-native-dos-hook.bak")
        backup.write_text("backup\n", encoding="utf-8")
        marker = repaired_at.timestamp()
        os.utime(manifest_path, (marker, marker))
        os.utime(backup, (marker, marker))


class DogfoodWitnessTest(unittest.TestCase):
    def test_builds_privacy_preserving_proven_witness(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            home = root / "codex-home"
            sessions = home / "sessions" / "2026" / "06" / "25"
            sessions.mkdir(parents=True)
            session = sessions / "rollout-abc123.jsonl"
            session.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "response_item", "payload": {"content": "drop this prompt"}}),
                        json.dumps(
                            {
                                "type": "event_msg",
                                "payload": {
                                    "type": "token_count",
                                    "info": {
                                        "last_token_usage": {
                                            "input_tokens": 2006,
                                            "cached_input_tokens": 1920,
                                            "output_tokens": 9,
                                        },
                                        "prompt_text": "drop this too",
                                    },
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "turn.completed",
                                "timestamp": "2026-06-25T11:58:10Z",
                                "usage": {"input_tokens": 2006, "cached_input_tokens": 0, "output_tokens": 5},
                                "item": {"text": "drop response"},
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response_item",
                                "timestamp": "2026-06-25T11:58:40Z",
                                "payload": {
                                    "type": "function_call",
                                    "name": "shell_command",
                                    "arguments": json.dumps({"command": "rg needle tools"}),
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response_item",
                                "timestamp": "2026-06-25T11:58:50Z",
                                "payload": {
                                    "type": "function_call",
                                    "name": "shell_command",
                                    "arguments": json.dumps({"command": "echo x > tools/out.txt"}),
                                },
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            write_dos_fixture(root)
            write_native_hook_manifest(home, repaired_at=datetime(2026, 6, 25, 11, 58, 5, tzinfo=timezone.utc))

            out = root / "witness.json"
            usage = root / "usage.jsonl"
            gate = root / "gate.json"
            deny_gate = root / "deny-gate.json"
            gate.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "tool": "run_tests",
                        "policy": "examples/dev-agent-policy.json",
                        "executed": True,
                        "dry_run": False,
                        "command_redacted": True,
                        "command_label": "dogfood-witness-test",
                        "command_digest": "abc123",
                        "command_executable": "python",
                        "command_argc": 2,
                        "command": ["python", "tools\\codex_fak_gate_test.py"],
                        "command_exit_code": 0,
                        "command_stdout": "drop command stdout",
                        "command_stderr": "drop command stderr",
                        "preflight": {
                            "verdict": "ALLOW",
                            "reason": "NONE",
                            "by": "monitor",
                            "exit_code": 0,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            deny_gate.write_text(
                json.dumps(
                    {
                        "status": "DENIED_EXPECTED",
                        "tool": "git_push",
                        "policy": "examples/dev-agent-policy.json",
                        "executed": False,
                        "dry_run": True,
                        "expect_deny": True,
                        "expect_reason": "POLICY_BLOCK",
                        "command_redacted": True,
                        "command_label": "git-push-deny",
                        "command_digest": None,
                        "command_executable": None,
                        "command_argc": 0,
                        "command": ["git", "push"],
                        "preflight": {
                            "verdict": "DENY",
                            "reason": "POLICY_BLOCK",
                            "by": "monitor",
                            "exit_code": 0,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = mod.parse_args(
                [
                    "--thread-id",
                    "abc123",
                    "--codex-home",
                    str(home),
                    "--repo-root",
                    str(root),
                    "--out",
                    str(out),
                    "--usage-out",
                    str(usage),
                    "--gate-report",
                    str(gate),
                    "--gate-report",
                    str(deny_gate),
                    "--target-shell",
                    "bash",
                ]
            )
            args.repo_root = args.repo_root.resolve()
            code, report = mod.build_report(
                args,
                env={},
                runner=FakeRunner(),
                mcp_probe=fake_mcp_probe(),
                now=fixed_now,
            )
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "PROVEN")
            self.assertEqual(report["usage_rows"], 2)

            usage_text = usage.read_text(encoding="utf-8")
            self.assertIn("cached_input_tokens", usage_text)
            self.assertNotIn("drop this prompt", usage_text)
            self.assertNotIn("drop response", usage_text)
            self.assertNotIn("output_tokens", usage_text)

            saved = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema"], mod.SCHEMA)
            self.assertEqual(saved["summary"]["status"], "PROVEN")
            self.assertEqual(saved["summary"]["policy_adjudication"]["deny"]["tool"], "git_push")
            self.assertEqual(saved["summary"]["policy_adjudication"]["deny"]["verdict"], "DENY")
            self.assertEqual(saved["summary"]["policy_adjudication"]["allow"]["tool"], "git_status")
            self.assertEqual(saved["summary"]["policy_adjudication"]["mcp_stdio_status"], "PASS")
            self.assertEqual(saved["summary"]["local_fak_gate"]["status"], "PASS")
            self.assertEqual(saved["summary"]["local_fak_gate"]["passed"], 2)
            self.assertEqual(saved["summary"]["local_fak_gate"]["denied"], 1)
            self.assertEqual(saved["summary"]["local_fak_gate"]["expected_denied"], 1)
            self.assertEqual(saved["summary"]["local_fak_gate"]["tools"], ["git_push", "run_tests"])
            self.assertEqual(saved["summary"]["codex_hook_fast_path"]["status"], "PASS")
            self.assertEqual(saved["summary"]["codex_hook_fast_path"]["target_command_mode"], "native_launcher")
            self.assertEqual(saved["summary"]["codex_hook_fast_path"]["codex_native_launcher_hooks"], 1)
            self.assertEqual(saved["summary"]["codex_hook_fast_path"]["codex_powershell_native_hooks"], 0)
            self.assertEqual(saved["summary"]["codex_hook_fast_path"]["codex_python_cli_hooks"], 0)
            self.assertEqual(saved["summary"]["codex_hook_fast_path"]["post_repair_status"], "PASS")
            self.assertEqual(saved["summary"]["codex_hook_fast_path"]["post_repair_delegate_count"], 0)
            self.assertEqual(saved["summary"]["codex_hook_fast_path"]["post_repair_unknown_tree_admission_warnings"], 0)
            self.assertEqual(saved["summary"]["codex_hook_fast_path"]["post_repair_command_shape_status"], "PASS")
            self.assertEqual(
                saved["summary"]["codex_hook_fast_path"]["post_repair_shell_shape_counts"],
                {
                    "shell_in_tree_or_safe_write_target": 1,
                    "shell_no_write_target_detected": 1,
                },
            )
            self.assertEqual(saved["summary"]["codex_actionability"]["status"], "PASS")
            self.assertEqual(saved["summary"]["codex_actionability"]["residual"], ["HOST_SHELL_OPACITY"])
            self.assertEqual(saved["summary"]["codex_actionability"]["delegate_count"], 0)
            self.assertEqual(saved["summary"]["codex_actionability"]["post_repair_shell_shape_counts"]["shell_no_write_target_detected"], 1)
            self.assertEqual(
                saved["summary"]["codex_actionability"]["post_repair_shell_family_counts"],
                {"search_rg": 1, "shell_redirect": 1},
            )
            self.assertEqual(
                saved["summary"]["codex_actionability"]["post_repair_mutating_shell_family_counts"],
                {},
            )
            self.assertEqual(saved["summary"]["vcache"]["status"], "PROVEN")
            self.assertEqual(saved["summary"]["vcache"]["saved_pct"], 43.07078763708873)
            self.assertEqual(saved["summary"]["dos"]["status"], "PASS")
            self.assertEqual(saved["summary"]["dos"]["unknown_tree_warning_rate"], 0.0)
            self.assertEqual(saved["summary"]["dos"]["session_advisory_cautions"], 3)
            self.assertEqual(saved["summary"]["dos"]["session_advisory_by_tool"], {"Bash": 2, "apply_patch": 1})
            self.assertEqual(
                set(saved["summary"]["next_actions"]),
                {
                    "reduce Bash calls that DOS cannot scope to a declared file tree",
                    "post-repair Codex shell calls include commands with no path-visible write target",
                },
            )
            self.assertEqual(saved["checks"]["capability_floor_denies_publish"]["verdict"], "DENY")
            self.assertEqual(saved["checks"]["capability_floor_allows_status"]["verdict"], "ALLOW")
            self.assertEqual(saved["checks"]["mcp_stdio_adjudication"]["status"], "PASS")
            self.assertEqual(saved["checks"]["mcp_stdio_adjudication"]["denies_publish"]["reason"], "POLICY_BLOCK")
            self.assertEqual(saved["checks"]["codex_exec_mcp_usage"]["status"], "SKIPPED")
            gate_reports = saved["checks"]["local_fak_gate_reports"]
            self.assertEqual(gate_reports["status"], "PASS")
            self.assertEqual(gate_reports["reports"][0]["preflight"]["verdict"], "ALLOW")
            self.assertEqual(gate_reports["reports"][0]["command_label"], "dogfood-witness-test")
            self.assertTrue(gate_reports["reports"][0]["command_redacted"])
            self.assertNotIn("command", gate_reports["reports"][0])
            self.assertEqual(gate_reports["reports"][1]["status"], "DENIED_EXPECTED")
            self.assertEqual(gate_reports["reports"][1]["preflight"]["verdict"], "DENY")
            self.assertTrue(gate_reports["reports"][1]["expect_deny"])
            self.assertEqual(gate_reports["expected_denied"], 1)
            self.assertNotIn("command", gate_reports["reports"][1])
            self.assertNotIn("command_stdout", gate_reports["reports"][0])
            self.assertNotIn("command_stderr", gate_reports["reports"][0])
            self.assertEqual(saved["checks"]["codex_hook_fast_path"]["status"], "PASS")
            self.assertEqual(saved["checks"]["codex_hook_fast_path"]["post_repair_observations"]["status"], "PASS")
            self.assertEqual(saved["checks"]["codex_hook_fast_path"]["post_repair_command_shapes"]["status"], "PASS")
            self.assertNotIn("dos-hook.ps1 pretool", json.dumps(saved["checks"]["codex_hook_fast_path"]))
            self.assertNotIn("rg needle tools", json.dumps(saved["checks"]["codex_hook_fast_path"]))
            self.assertNotIn("echo x", json.dumps(saved["checks"]["codex_hook_fast_path"]))
            self.assertEqual(saved["checks"]["codex_actionability"]["status"], "PASS")
            self.assertEqual(saved["checks"]["codex_actionability"]["residual"], ["HOST_SHELL_OPACITY"])
            self.assertEqual(
                saved["checks"]["codex_actionability"]["post_repair_shell_family_counts"],
                {"search_rg": 1, "shell_redirect": 1},
            )
            self.assertEqual(saved["checks"]["codex_actionability"]["post_repair_mutating_shell_family_counts"], {})
            self.assertNotIn("rg needle tools", json.dumps(saved["checks"]["codex_actionability"]))
            self.assertNotIn("echo x", json.dumps(saved["checks"]["codex_actionability"]))
            self.assertEqual(saved["checks"]["vcache_telemetry_proof"]["status"], "PROVEN")
            self.assertEqual(saved["checks"]["dos_helped_session"]["status"], "FOUND")
            self.assertEqual(saved["checks"]["dos_helped_session"]["by_advisory_tool"], {"Bash": 2, "apply_patch": 1})
            self.assertNotIn("examples", saved["checks"]["dos_helped_session"])
            dos = saved["checks"]["dos_session_audit"]
            self.assertEqual(dos["status"], "PASS")
            self.assertEqual(dos["stream"]["steps"], 2)
            self.assertIn("timestamp-window", dos["observations"]["scope"])
            self.assertEqual(dos["observations"]["pretool_calls"], 2)
            self.assertEqual(dos["observations"]["unknown_tree_admission_warnings"], 0)
            encoded_dos = json.dumps(dos)
            self.assertNotIn("drop this prompt", encoded_dos)
            self.assertNotIn("drop response", encoded_dos)
            encoded_summary = json.dumps(saved["summary"])
            self.assertNotIn("drop this prompt", encoded_summary)
            self.assertNotIn("drop response", encoded_summary)
            self.assertNotIn("drop command stdout", json.dumps(saved))
            self.assertNotIn("drop command stderr", json.dumps(saved))
            self.assertNotIn("C:/secret/path.txt", json.dumps(saved))
            self.assertNotIn("drop example reason", json.dumps(saved))

    def test_dos_session_audit_flags_unknown_tree_warning_rate(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_dos_fixture(
                root,
                observations=[
                    {
                        "schema": {"family": "hook-observation", "version": 1},
                        "verb": "pretool",
                        "outcome": "warn",
                        "rung": "admission",
                        "reason": "admission",
                        "tree_known": False,
                        "dialect": "codex",
                        "latency_ms": 5.0,
                        "ts": "2026-06-25T11:58:10Z",
                    },
                    {
                        "schema": {"family": "hook-observation", "version": 1},
                        "verb": "pretool",
                        "outcome": "passthrough",
                        "rung": "provenance",
                        "dialect": "codex",
                        "latency_ms": 2.0,
                        "ts": "2026-06-25T11:58:20Z",
                    },
                ],
            )
            args = mod.parse_args(["--session", str(root / "session.jsonl"), "--thread-id", "abc123", "--repo-root", str(root)])
            args.repo_root = args.repo_root.resolve()
            audit = mod.dos_session_audit(args, "abc123")

            self.assertEqual(audit["status"], "WARN")
            self.assertEqual(audit["observations"]["unknown_tree_admission_warnings"], 1)
            self.assertEqual(audit["observations"]["unknown_tree_warning_rate"], 0.5)
            self.assertTrue(any("footprint" in rec for rec in audit["recommendations"]))

            summary = mod.dogfood_summary(
                status="PROVEN",
                deny_tool="git_push",
                allow_tool="git_status",
                preflight_deny={"verdict": "DENY", "reason": "POLICY_BLOCK"},
                preflight_allow={"verdict": "ALLOW", "reason": "NONE"},
                mcp_stdio={"status": "PASS"},
                codex_exec={"status": "PASS"},
                proof={"status": "PROVEN", "requests": 2, "saved_pct": 43.0, "saved_token_equiv": 1728},
                dos_audit=audit,
                dos_helped={"status": "FOUND", "advisory": 2, "blocked": 0, "by_advisory_tool": {"Bash": 2}},
                gate_reports={"status": "PASS", "reports": [], "total": 0, "passed": 0, "failed": 0, "denied": 0},
                hook_fast_path={
                    "status": "PASS",
                    "codex_native_launcher_hooks": 1,
                    "codex_python_cli_hooks": 0,
                    "post_repair_observations": {
                        "status": "WARN",
                        "delegate_count": 0,
                        "unknown_tree_admission_warnings": 2,
                        "unknown_tree_warning_rate": 0.5,
                    },
                },
                actionability={
                    "status": "UNKNOWN",
                    "reasons": [],
                    "unknowns": ["post-repair command-shape evidence is missing"],
                    "residual": [],
                    "delegate_source": "post_repair",
                    "delegate_count": 0,
                    "max_delegates": 0,
                    "stop_total": 0,
                    "post_repair_unknown_tree_admission_warnings": 2,
                    "post_repair_shell_shape_counts": {},
                    "post_repair_shell_family_counts": {"git_read": 1, "git_write": 2},
                    "post_repair_mutating_shell_family_counts": {"git_write": 2},
                },
            )
            self.assertEqual(summary["dos"]["status"], "WARN")
            self.assertEqual(summary["dos"]["unknown_tree_warning_rate"], 0.5)
            self.assertEqual(summary["codex_hook_fast_path"]["post_repair_delegate_count"], 0)
            self.assertEqual(summary["codex_hook_fast_path"]["post_repair_unknown_tree_admission_warnings"], 2)
            self.assertEqual(summary["codex_actionability"]["status"], "UNKNOWN")
            self.assertIn("post-repair command-shape evidence is missing", summary["codex_actionability"]["unknowns"])
            self.assertEqual(summary["codex_actionability"]["post_repair_mutating_shell_family_counts"], {"git_write": 2})
            self.assertTrue(any("footprint" in action for action in summary["next_actions"]))
            self.assertTrue(any("remaining post-repair issue" in action for action in summary["next_actions"]))
            self.assertTrue(any("command-shape evidence" in action for action in summary["next_actions"]))

    def test_codex_exec_jsonl_is_sanitized_to_mcp_usage(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "codex-exec.jsonl"
            raw.write_text(codex_exec_jsonl() + "\n", encoding="utf-8")

            proof = mod.read_codex_exec_mcp_usage(str(raw), "git_push", "git_status")
            self.assertEqual(proof["status"], "PASS")
            self.assertEqual(proof["thread_id"], "nested-thread")
            self.assertEqual(proof["jsonl_rows"], 6)
            self.assertEqual(proof["turn_usage"]["cached_input_tokens"], 27392)
            self.assertEqual(proof["mcp_tool_calls"][0]["verdict"]["reason"], "POLICY_BLOCK")
            encoded = json.dumps(proof)
            self.assertNotIn("drop final text", encoded)
            self.assertNotIn("drop prompt", encoded)
            self.assertNotIn("output_tokens", encoded)

    def test_run_codex_exec_probe_is_sanitized_and_used_by_build_report(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            session = root / "session.jsonl"
            session.write_text(
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 2006, "cached_input_tokens": 1920},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            write_dos_fixture(root, thread_id="abc123")
            out = root / "witness.json"
            args = mod.parse_args(
                [
                    "--session",
                    str(session),
                    "--thread-id",
                    "abc123",
                    "--repo-root",
                    str(root),
                    "--out",
                    str(out),
                    "--usage-out",
                    str(root / "usage.jsonl"),
                    "--run-codex-exec",
                ]
            )
            args.repo_root = args.repo_root.resolve()
            runner = FakeRunner()
            code, report = mod.build_report(
                args,
                env={},
                runner=runner,
                mcp_probe=fake_mcp_probe(),
                now=fixed_now,
            )
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "PROVEN")
            nested = report["checks"]["codex_exec_mcp_usage"]
            self.assertEqual(nested["status"], "PASS")
            self.assertEqual(nested["source"], "codex exec --json")
            self.assertEqual(nested["exit_code"], 0)
            self.assertEqual(nested["mcp_tool_calls"][0]["arguments_tool"], "git_push")
            self.assertTrue(any(len(call) >= 3 and call[1:3] == ["exec", "--json"] for call in runner.calls))
            encoded = json.dumps(nested)
            self.assertNotIn("drop final text", encoded)
            self.assertNotIn("drop prompt", encoded)

    def test_refuted_vcache_proof_returns_refuted_not_error(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            session = root / "session.jsonl"
            session.write_text(
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 100, "cached_input_tokens": 0},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = mod.parse_args(
                [
                    "--session",
                    str(session),
                    "--thread-id",
                    "abc123",
                    "--repo-root",
                    str(root),
                    "--out",
                    str(root / "witness.json"),
                    "--usage-out",
                    str(root / "usage.jsonl"),
                ]
            )
            args.repo_root = args.repo_root.resolve()
            code, report = mod.build_report(
                args,
                env={},
                runner=FakeRunner("REFUTED"),
                mcp_probe=fake_mcp_probe(),
                now=fixed_now,
            )
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "REFUTED")

    def test_mcp_failure_blocks_proven_witness(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            session = root / "session.jsonl"
            session.write_text(
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 2006, "cached_input_tokens": 1920},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = mod.parse_args(
                [
                    "--session",
                    str(session),
                    "--thread-id",
                    "abc123",
                    "--repo-root",
                    str(root),
                    "--out",
                    str(root / "witness.json"),
                    "--usage-out",
                    str(root / "usage.jsonl"),
                ]
            )
            args.repo_root = args.repo_root.resolve()
            code, report = mod.build_report(
                args,
                env={},
                runner=FakeRunner(),
                mcp_probe=fake_mcp_probe("ERROR"),
                now=fixed_now,
            )
            self.assertEqual(code, 2)
            self.assertEqual(report["status"], "ERROR")
            self.assertEqual(report["checks"]["mcp_stdio_adjudication"]["status"], "ERROR")

    def test_codex_exec_failure_blocks_proven_witness_when_supplied(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            session = root / "session.jsonl"
            session.write_text(
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 2006, "cached_input_tokens": 1920},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            raw = root / "codex-exec.jsonl"
            raw.write_text(json.dumps({"type": "thread.started", "thread_id": "nested"}) + "\n", encoding="utf-8")
            args = mod.parse_args(
                [
                    "--session",
                    str(session),
                    "--thread-id",
                    "abc123",
                    "--repo-root",
                    str(root),
                    "--out",
                    str(root / "witness.json"),
                    "--usage-out",
                    str(root / "usage.jsonl"),
                    "--codex-exec-jsonl",
                    str(raw),
                ]
            )
            args.repo_root = args.repo_root.resolve()
            code, report = mod.build_report(
                args,
                env={},
                runner=FakeRunner(),
                mcp_probe=fake_mcp_probe(),
                now=fixed_now,
            )
            self.assertEqual(code, 2)
            self.assertEqual(report["status"], "ERROR")
            self.assertEqual(report["checks"]["codex_exec_mcp_usage"]["status"], "FAIL")

    def test_supplied_gate_report_failure_blocks_proven_witness(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            session = root / "session.jsonl"
            session.write_text(
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 2006, "cached_input_tokens": 1920},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            gate = root / "gate.json"
            gate.write_text(
                json.dumps(
                    {
                        "status": "COMMAND_FAILED",
                        "tool": "go_test",
                        "executed": True,
                        "command": ["go", "test", "./cmd/fak"],
                        "command_exit_code": 1,
                        "command_stdout": "drop failing stdout",
                        "command_stderr": "drop failing stderr",
                        "preflight": {"verdict": "ALLOW", "reason": "NONE", "by": "monitor", "exit_code": 0},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = mod.parse_args(
                [
                    "--session",
                    str(session),
                    "--thread-id",
                    "abc123",
                    "--repo-root",
                    str(root),
                    "--out",
                    str(root / "witness.json"),
                    "--usage-out",
                    str(root / "usage.jsonl"),
                    "--gate-report",
                    str(gate),
                ]
            )
            args.repo_root = args.repo_root.resolve()
            code, report = mod.build_report(
                args,
                env={},
                runner=FakeRunner(),
                mcp_probe=fake_mcp_probe(),
                now=fixed_now,
            )
            self.assertEqual(code, 2)
            self.assertEqual(report["status"], "ERROR")
            self.assertEqual(report["checks"]["local_fak_gate_reports"]["status"], "FAIL")
            encoded = json.dumps(report)
            self.assertNotIn("drop failing stdout", encoded)
            self.assertNotIn("drop failing stderr", encoded)

    def test_cli_reports_missing_token_rows_as_error(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            session = root / "session.jsonl"
            session.write_text(json.dumps({"type": "response_item"}) + "\n", encoding="utf-8")
            err = io.StringIO()
            code = mod.run(
                [
                    "--session",
                    str(session),
                    "--thread-id",
                    "abc123",
                    "--repo-root",
                    str(root),
                    "--out",
                    str(root / "witness.json"),
                ],
                env={},
                runner=FakeRunner(),
                stdout=io.StringIO(),
                stderr=err,
            )
            self.assertEqual(code, 2)
            self.assertIn("no Codex token usage rows", err.getvalue())


if __name__ == "__main__":
    unittest.main()
