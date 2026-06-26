#!/usr/bin/env python3
"""Hermetic tests for tools/codex_dos_recent_audit.py."""
from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "codex_dos_recent_audit.py"


def load():
    spec = importlib.util.spec_from_file_location("codex_dos_recent_audit", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def write_hook_manifest(home: Path, command: str) -> Path:
    root = home / "plugins" / "cache" / "dos" / "dos-kernel" / "0.28.0"
    launcher = root / "bin" / "dos-hook"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text("# launcher\n", encoding="utf-8")
    manifest = root / "hooks" / "hooks.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "shell": "bash" if "bin/dos-hook" in command and "dos-hook.ps1" not in command else "powershell",
                                    "command": command,
                                }
                            ]
                        }
                    ]
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def native_bash_hook_command() -> str:
    return 'root="${CLAUDE_PLUGIN_ROOT:-${CODEX_PLUGIN_ROOT:-}}"; "$root/bin/dos-hook" pretool --workspace . --dialect codex'


def write_gate_report(path: Path, *, tool: str, reason: str, created_at: str) -> None:
    path.write_text(
        json.dumps(
            {
                "created_at": created_at,
                "tool": tool,
                "status": "DENIED_EXPECTED",
                "expect_deny": True,
                "expect_reason": reason,
                "executed": False,
                "dry_run": True,
                "preflight": {
                    "verdict": "DENY",
                    "reason": reason,
                    "by": "monitor",
                    "exit_code": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


class RecentCodexDosAuditTest(unittest.TestCase):
    def test_rolls_up_only_codex_threads_with_privacy_boundary(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            home = root / "codex-home"
            thread = "019efde3-6794-7401-93a1-e97e6bd72a9c"
            ignored = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
            session = home / "sessions" / "2026" / "06" / "25" / f"rollout-{thread}.jsonl"
            write_jsonl(session, [{"type": "response_item", "payload": {"content": "drop prompt text"}}])
            write_hook_manifest(
                home,
                "$py = Get-Command python; & $py.Source -m dos.cli hook pretool --workspace . --dialect codex",
            )

            write_jsonl(
                root / ".dos" / "streams" / f"{thread}.jsonl",
                [
                    {"op": "STEP", "tool_name": "Bash", "ts": "2026-06-25T12:00:00Z"},
                    {"op": "STEP", "tool_name": "Bash", "ts": "2026-06-25T12:01:00Z"},
                ],
            )
            write_jsonl(
                root / ".dos" / "streams" / f"{ignored}.jsonl",
                [{"op": "STEP", "tool_name": "Bash", "ts": "2026-06-25T12:00:00Z"}],
            )
            write_jsonl(
                root / ".dos" / "metrics" / "observations.jsonl",
                [
                    {
                        "verb": "pretool",
                        "outcome": "warn",
                        "rung": "admission",
                        "tree_known": False,
                        "dialect": "codex",
                        "latency_ms": 9.0,
                        "ts": "2026-06-25T12:00:10Z",
                    },
                    {
                        "verb": "pretool",
                        "outcome": "passthrough",
                        "rung": "provenance",
                        "dialect": "codex",
                        "latency_ms": 3.0,
                        "ts": "2026-06-25T12:00:20Z",
                    },
                    {
                        "verb": "posttool",
                        "outcome": "passthrough",
                        "latency_ms": 4.0,
                        "ts": "2026-06-25T12:00:30Z",
                    },
                ],
            )
            stops = root / ".dos" / "stop-failures" / f"{thread}.json"
            stops.parent.mkdir(parents=True, exist_ok=True)
            stops.write_text(json.dumps({"total": 0, "consecutive": 0}) + "\n", encoding="utf-8")

            old_home = os.environ.get("CODEX_HOME")
            os.environ["CODEX_HOME"] = str(home)
            try:
                report = mod.build_report(root, home, limit=10, since_days=3650)
            finally:
                if old_home is None:
                    os.environ.pop("CODEX_HOME", None)
                else:
                    os.environ["CODEX_HOME"] = old_home

            self.assertEqual(report["status"], "WARN")
            self.assertEqual(report["budgets"]["max_unknown_tree_rate"], 0.02)
            self.assertIn(report["dos_version"]["status"], {"FOUND", "UNKNOWN"})
            self.assertFalse(report["dos_version"].get("latest_checked", False))
            self.assertEqual(report["codex_hook_fast_path"]["status"], "WARN")
            self.assertEqual(report["codex_hook_fast_path"]["codex_command_modes"], {"python_cli": 1})
            self.assertEqual(
                report["codex_hook_fast_path"]["doctor"]["apply"],
                "python tools/codex_dos_hook_doctor.py --codex-home <codex-home> --apply",
            )
            self.assertEqual(
                report["codex_hook_fast_path"]["repair_projection"]["projected_codex_command_modes"],
                {"native_launcher": 1},
            )
            self.assertTrue(report["codex_hook_fast_path"]["repair_projection"]["would_clear_codex_python_cli"])
            self.assertEqual(report["sessions_audited"], 1)
            self.assertEqual(report["sessions"][0]["thread_id"], thread)
            self.assertEqual(report["summary"]["tool_counts"], {"Bash": 2})
            self.assertEqual(report["summary"]["unknown_tree_admission_warnings"], 1)
            self.assertEqual(report["summary"]["unknown_tree_warning_rate"], 0.5)
            self.assertEqual(report["workspace_stop_failures"]["markers"], 1)
            self.assertEqual(report["workspace_stop_failures"]["total_failures"], 0)
            encoded = json.dumps(report)
            self.assertNotIn("drop prompt text", encoded)
            self.assertNotIn("-m dos.cli hook", encoded)
            self.assertNotIn(str(home), encoded)
            self.assertNotIn(ignored, encoded)

    def test_workspace_stop_failures_surface_non_codex_markers(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            home = root / "codex-home"
            user_home = root / "user-home"
            thread = "019efde3-6794-7401-93a1-e97e6bd72a9c"
            claude_session = "e7f31ce8-185b-4b6b-8e41-9db98bd1f4e6"
            hot_session = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
            zero_session = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
            write_jsonl(home / "sessions" / "2026" / "06" / "25" / f"rollout-{thread}.jsonl", [{"type": "response_item"}])
            write_hook_manifest(home, native_bash_hook_command())
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            write_jsonl(root / ".dos" / "streams" / f"{thread}.jsonl", [{"op": "STEP", "tool_name": "Read", "ts": now}])
            write_jsonl(
                root / ".dos" / "metrics" / "observations.jsonl",
                [{"verb": "pretool", "outcome": "passthrough", "rung": "provenance", "dialect": "codex", "ts": now}],
            )
            transcript = user_home / ".claude" / "projects" / "C--work-fak" / f"{claude_session}.jsonl"
            write_jsonl(
                transcript,
                [
                    {"type": "user", "message": {"content": "prompt should not leak"}},
                    {
                        "type": "assistant",
                        "timestamp": now,
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "tool_use", "name": "Bash", "input": {"command": "secret command should not leak"}},
                                {"type": "tool_result", "content": "permission error: hook blocked secret result should not leak"},
                            ],
                        },
                    },
                ],
            )
            stop_dir = root / ".dos" / "stop-failures"
            stop_dir.mkdir(parents=True, exist_ok=True)
            (stop_dir / f"{claude_session}.json").write_text(json.dumps({"total": 1, "consecutive": 1}) + "\n", encoding="utf-8")
            (stop_dir / f"{hot_session}.json").write_text(json.dumps({"total": 3, "consecutive": 0}) + "\n", encoding="utf-8")
            (stop_dir / f"{zero_session}.json").write_text(json.dumps({"total": 0, "consecutive": 0}) + "\n", encoding="utf-8")

            old_user_home = os.environ.get("FLEET_USER_HOME")
            os.environ["FLEET_USER_HOME"] = str(user_home)
            try:
                report = mod.build_report(root, home, limit=10, since_days=3650, max_delegates=0)
            finally:
                if old_user_home is None:
                    os.environ.pop("FLEET_USER_HOME", None)
                else:
                    os.environ["FLEET_USER_HOME"] = old_user_home

            self.assertEqual(report["status"], "WARN")
            self.assertEqual(report["summary"]["stop_failures_total"], 0)
            self.assertEqual(report["summary"]["workspace_stop_failures_total"], 4)
            self.assertEqual(report["summary"]["workspace_stop_failure_markers"], 3)
            self.assertEqual(report["summary"]["workspace_stop_failure_zero_markers"], 1)
            self.assertEqual(report["summary"]["workspace_stop_failure_nonzero_markers"], 2)
            self.assertEqual(report["summary"]["workspace_stop_failure_active_markers"], 1)
            self.assertEqual(report["summary"]["workspace_stop_failure_active_consecutive_total"], 1)
            self.assertEqual(report["summary"]["workspace_stop_failure_active_recent_threshold_hours"], 6)
            self.assertEqual(report["summary"]["workspace_stop_failure_recent_active_markers"], 1)
            self.assertEqual(report["summary"]["workspace_stop_failure_recent_active_consecutive_total"], 1)
            self.assertEqual(report["summary"]["workspace_stop_failure_stale_active_markers"], 0)
            self.assertEqual(report["summary"]["workspace_stop_failure_stale_active_consecutive_total"], 0)
            self.assertEqual(report["summary"]["workspace_stop_failure_healed_nonzero_markers"], 1)
            self.assertEqual(report["summary"]["workspace_stop_failure_origin_counts"], {"claude_transcript": 1, "marker_only": 2})
            self.assertEqual(report["summary"]["workspace_stop_failure_recent_active_origin_counts"], {"claude_transcript": 1})
            self.assertEqual(
                report["summary"]["workspace_stop_failure_settlement_action_counts"],
                {"HEALED_NONZERO": 1, "RECENT_REVIEW": 1, "ZERO_TOTAL": 1},
            )
            self.assertEqual(report["summary"]["workspace_stop_failure_active_settlement_action_counts"], {"RECENT_REVIEW": 1})
            self.assertEqual(report["actionability"]["status"], "WARN")
            self.assertIn("stop blocks or uncleared StopFailure API-wall breaker markers are present", report["actionability"]["reasons"])
            stop = report["workspace_stop_failures"]
            self.assertEqual(stop["status"], "WARN")
            self.assertEqual(stop["total_failures"], 4)
            self.assertEqual(stop["active_consecutive_markers"], 1)
            self.assertEqual(stop["active_consecutive_total"], 1)
            self.assertEqual(stop["recent_active_consecutive_markers"], 1)
            self.assertEqual(stop["recent_active_consecutive_total"], 1)
            self.assertEqual(stop["stale_active_consecutive_markers"], 0)
            self.assertEqual(stop["stale_active_consecutive_total"], 0)
            self.assertEqual(stop["healed_nonzero_markers"], 1)
            self.assertEqual(stop["origin_counts"], {"claude_transcript": 1, "marker_only": 2})
            self.assertEqual(stop["recent_active_origin_counts"], {"claude_transcript": 1})
            self.assertEqual(stop["settlement_action_counts"], {"HEALED_NONZERO": 1, "RECENT_REVIEW": 1, "ZERO_TOTAL": 1})
            self.assertEqual(stop["recent_active_settlement_action_counts"], {"RECENT_REVIEW": 1})
            self.assertEqual(sum(day["total_failures"] for day in stop["by_day"].values()), 4)
            self.assertEqual(stop["top_nonzero"][0]["session_id"], hot_session)
            self.assertEqual(stop["top_nonzero"][0]["total"], 3)
            self.assertEqual(stop["top_nonzero"][0]["origin"], "marker_only")
            self.assertEqual(stop["top_active"][0]["session_id"], claude_session)
            self.assertEqual(stop["top_active"][0]["consecutive"], 1)
            self.assertEqual(stop["top_active"][0]["origin"], "claude_transcript")
            self.assertEqual(stop["top_active"][0]["settlement_action"], "RECENT_REVIEW")
            self.assertEqual(stop["top_recent_active"][0]["session_id"], claude_session)
            found = [item for item in stop["recent"] if item["session_id"] == claude_session][0]
            self.assertEqual(found["transcript"]["status"], "FOUND")
            self.assertEqual(found["transcript"]["project"], "C--work-fak")
            self.assertIn("HOOK_OR_API_WALL_FEEDBACK", found["transcript_summary"]["evidence_tags"])
            self.assertIn("HOST_PERMISSION_INTERRUPT", found["transcript_summary"]["evidence_tags"])
            self.assertIn("DENY_OR_BLOCKED_FEEDBACK", found["transcript_summary"]["evidence_tags"])
            rendered = mod.render(report)
            self.assertIn("workspace StopFailure API-wall failures: 4", rendered)
            self.assertIn("active StopFailure blockers: 1 markers", rendered)
            self.assertIn('recent_origins={"claude_transcript": 1}', rendered)
            self.assertIn('settlement={"RECENT_REVIEW": 1}', rendered)
            self.assertIn("top recent active StopFailure sessions:", rendered)
            self.assertIn("top active StopFailure sessions:", rendered)
            self.assertIn("top StopFailure sessions:", rendered)
            self.assertIn(hot_session, rendered)
            debt = mod.render_debt_packet(report)
            self.assertIn("current actionable WARN is workspace StopFailure API-wall breaker state", debt)
            self.assertIn("workspace_stop_failures_total: `4`", debt)
            self.assertIn("workspace_stop_failure_active_markers: `1`", debt)
            self.assertIn("workspace_stop_failure_recent_active_markers: `1`", debt)
            self.assertIn("workspace_stop_failure_stale_active_markers: `0`", debt)
            self.assertIn('workspace_stop_failure_recent_active_origin_counts: `{"claude_transcript": 1}`', debt)
            self.assertIn('workspace_stop_failure_active_settlement_action_counts: `{"RECENT_REVIEW": 1}`', debt)
            self.assertIn("workspace_stop_failure_top_recent_active_sessions", debt)
            self.assertIn("workspace_stop_failure_top_active_sessions", debt)
            self.assertIn("workspace_stop_failure_transcript_evidence_tags", debt)
            self.assertIn("workspace_stop_failure_top_sessions", debt)
            encoded = json.dumps(report)
            self.assertNotIn("prompt should not leak", encoded)
            self.assertNotIn("secret command should not leak", encoded)
            self.assertNotIn("secret result should not leak", encoded)
            self.assertNotIn(str(user_home), encoded)

    def test_codex_hook_fast_path_detects_native_vs_python_manifest(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "codex-home"
            self.assertEqual(mod.codex_hook_fast_path(home)["status"], "UNKNOWN")

            write_hook_manifest(
                home,
                "$py = Get-Command python; & $py.Source -m dos.cli hook pretool --workspace . --dialect codex",
            )
            python_report = mod.codex_hook_fast_path(home)
            self.assertEqual(python_report["status"], "WARN")
            self.assertEqual(python_report["codex_python_cli_hooks"], 1)
            self.assertEqual(python_report["codex_native_launcher_hooks"], 0)
            self.assertEqual(
                python_report["repair_projection"]["projected_codex_command_modes"],
                {"native_launcher": 1},
            )
            self.assertEqual(
                python_report["manifests"],
                ["plugins/cache/dos/dos-kernel/0.28.0/hooks/hooks.json"],
            )

        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "codex-home"
            write_hook_manifest(
                home,
                native_bash_hook_command(),
            )
            native_report = mod.codex_hook_fast_path(home)
            self.assertEqual(native_report["status"], "PASS")
            self.assertEqual(native_report["codex_command_modes"], {"native_launcher": 1})
            self.assertEqual(native_report["codex_powershell_native_hooks"], 0)

        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "codex-home"
            write_hook_manifest(
                home,
                "& $env:CLAUDE_PLUGIN_ROOT\\bin\\dos-hook.ps1 pretool --workspace . --dialect codex",
            )
            powershell_report = mod.codex_hook_fast_path(home)
            self.assertEqual(powershell_report["status"], "WARN")
            self.assertEqual(powershell_report["codex_command_modes"], {"powershell_native_launcher": 1})
            self.assertEqual(powershell_report["codex_powershell_native_hooks"], 1)
            self.assertEqual(
                powershell_report["repair_projection"]["projected_codex_command_modes"],
                {"native_launcher": 1},
            )

    def test_shell_family_parses_git_subcommands(self) -> None:
        mod = load()
        self.assertEqual(mod.shell_command_family("git status --short"), "git_read")
        self.assertEqual(mod.shell_command_family("git merge-base HEAD origin/main"), "git_read")
        self.assertEqual(mod.shell_command_family("git -C . diff -- tools"), "git_read")
        self.assertEqual(mod.shell_command_family("git add tools/report.json"), "git_write")
        self.assertEqual(mod.shell_command_family("git commit -s -- tools/report.json"), "git_write")
        self.assertEqual(mod.shell_command_family("git -C . push origin main"), "git_write")

    def test_actionable_gate_flags_mutating_shell_families(self) -> None:
        mod = load()
        gate = mod.actionable_gate(
            hook_fast_path={"status": "PASS"},
            post_repair={"observations": 3, "delegate_count": 0, "unknown_tree_admission_warnings": 2},
            command_shapes={
                "status": "PASS",
                "shell_shape_counts": {"shell_no_write_target_detected": 2},
                "shell_family_counts": {"git_read": 1, "git_write": 2},
            },
            delegate_total=0,
            stop_total=0,
            max_delegates=0,
        )
        self.assertEqual(gate["status"], "WARN")
        self.assertIn("post-repair shell command families include opaque mutating operations", gate["reasons"])
        self.assertEqual(gate["post_repair_mutating_shell_family_counts"], {"git_write": 2})
        self.assertEqual(gate["residual"], [])
        report = {
            "status": "WARN",
            "sessions_audited": 1,
            "summary": {},
            "dos_version": {"version": "0.28.0", "using_latest": True},
            "codex_hook_fast_path": {
                "status": "PASS",
                "codex_command_modes": {"native_launcher": 1},
                "post_repair_observations": {"observations": 3, "delegate_count": 0},
                "post_repair_command_shapes": {
                    "shell_shape_counts": {"shell_no_write_target_detected": 2},
                    "shell_family_counts": {"git_write": 2},
                    "write_op_counts": {},
                },
            },
            "actionability": gate,
            "recommendations": [],
        }
        debt = mod.render_debt_packet(report)
        self.assertIn("current actionable WARN is opaque mutating shell usage", debt)
        self.assertIn("post_repair_mutating_shell_families", debt)
        self.assertIn("git_write", debt)
        self.assertIn("git_add", debt)
        rendered = mod.render(report)
        self.assertIn("actionable reasons:", rendered)
        stale_gate = mod.actionable_gate(
            hook_fast_path={"status": "PASS"},
            post_repair={"observations": 3, "delegate_count": 0, "unknown_tree_admission_warnings": 2},
            command_shapes={
                "status": "PASS",
                "shell_shape_counts": {"shell_no_write_target_detected": 2},
                "shell_family_counts": {"git_write": 2},
            },
            delegate_total=0,
            stop_total=0,
            max_delegates=0,
            git_gate={
                "status": "PASS",
                "post_gate_command_shapes": {"shell_family_counts": {"git_write": 1}},
            },
        )
        self.assertEqual(stale_gate["status"], "WARN")
        self.assertIn("post-git-gate shell command families include opaque mutating operations", stale_gate["reasons"])

    def test_post_repair_command_shapes_use_audited_stream_scope(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            home = root / "codex-home"
            audited = "019efde3-6794-7401-93a1-e97e6bd72a9c"
            unstreamed = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
            write_jsonl(
                home / "sessions" / "2026" / "06" / "25" / f"rollout-{audited}.jsonl",
                [
                    {
                        "type": "response_item",
                        "timestamp": "2026-06-25T12:05:30Z",
                        "payload": {
                            "type": "function_call",
                            "name": "shell_command",
                            "arguments": json.dumps({"command": "rg needle docs"}),
                        },
                    },
                ],
            )
            write_jsonl(
                home / "sessions" / "2026" / "06" / "25" / f"rollout-{unstreamed}.jsonl",
                [
                    {
                        "type": "response_item",
                        "timestamp": "2026-06-25T12:06:30Z",
                        "payload": {
                            "type": "function_call",
                            "name": "shell_command",
                            "arguments": json.dumps({"command": "git commit -s -- tools/report.json"}),
                        },
                    },
                ],
            )
            manifest = write_hook_manifest(
                home,
                native_bash_hook_command(),
            )
            backup = manifest.with_name("hooks.json.before-native-dos-hook.bak")
            backup.write_text("backup\n", encoding="utf-8")
            marker = datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc).timestamp()
            os.utime(manifest, (marker, marker))
            os.utime(backup, (marker, marker))
            write_jsonl(
                root / ".dos" / "streams" / f"{audited}.jsonl",
                [{"op": "STEP", "tool_name": "Bash", "ts": "2026-06-25T12:00:00Z"}],
            )
            write_jsonl(
                root / ".dos" / "metrics" / "observations.jsonl",
                [
                    {
                        "verb": "pretool",
                        "outcome": "passthrough",
                        "rung": "provenance",
                        "dialect": "codex",
                        "latency_ms": 9.0,
                        "ts": "2026-06-25T12:00:30Z",
                    },
                ],
            )

            report = mod.build_report(root, home, limit=10, since_days=3650)

            shapes = report["codex_hook_fast_path"]["post_repair_command_shapes"]
            self.assertEqual(shapes["scope"], "audited_dos_streams")
            self.assertEqual(shapes["threads_supplied"], 1)
            self.assertEqual(shapes["shell_family_counts"], {"search_rg": 1})
            self.assertEqual(shapes["mutating_shell_sessions"], [])
            self.assertNotIn("git_write", json.dumps(report))
            self.assertEqual(report["actionability"]["status"], "PASS")

    def test_mutating_shell_sessions_are_sanitized(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            home = root / "codex-home"
            thread = "019efde3-6794-7401-93a1-e97e6bd72a9c"
            write_jsonl(
                home / "sessions" / "2026" / "06" / "25" / f"rollout-{thread}.jsonl",
                [
                    {
                        "type": "response_item",
                        "timestamp": "2026-06-25T12:05:30Z",
                        "payload": {
                            "type": "function_call",
                            "name": "shell_command",
                            "arguments": json.dumps({"command": "git commit -s -- tools/report.json"}),
                        },
                    },
                ],
            )
            manifest = write_hook_manifest(
                home,
                native_bash_hook_command(),
            )
            backup = manifest.with_name("hooks.json.before-native-dos-hook.bak")
            backup.write_text("backup\n", encoding="utf-8")
            marker = datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc).timestamp()
            os.utime(manifest, (marker, marker))
            os.utime(backup, (marker, marker))
            write_jsonl(
                root / ".dos" / "streams" / f"{thread}.jsonl",
                [{"op": "STEP", "tool_name": "Bash", "ts": "2026-06-25T12:00:00Z"}],
            )
            write_jsonl(
                root / ".dos" / "metrics" / "observations.jsonl",
                [
                    {
                        "verb": "pretool",
                        "outcome": "warn",
                        "rung": "admission",
                        "tree_known": False,
                        "dialect": "codex",
                        "latency_ms": 9.0,
                        "ts": "2026-06-25T12:00:30Z",
                    },
                ],
            )

            report = mod.build_report(root, home, limit=10, since_days=3650)

            shapes = report["codex_hook_fast_path"]["post_repair_command_shapes"]
            self.assertEqual(shapes["shell_family_counts"], {"git_write": 1})
            self.assertEqual(len(shapes["mutating_shell_sessions"]), 1)
            session = shapes["mutating_shell_sessions"][0]
            self.assertEqual(session["thread_id"], thread)
            self.assertEqual(session["codex_session_file"], f"rollout-{thread}.jsonl")
            self.assertEqual(session["mutating_shell_family_counts"], {"git_write": 1})
            self.assertEqual(report["actionability"]["status"], "WARN")
            encoded = json.dumps(report)
            self.assertNotIn("git commit", encoded)
            self.assertNotIn("tools/report.json", encoded)

    def test_gate_reports_make_historical_git_write_non_actionable(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            home = root / "codex-home"
            thread = "019efde3-6794-7401-93a1-e97e6bd72a9c"
            write_jsonl(
                home / "sessions" / "2026" / "06" / "25" / f"rollout-{thread}.jsonl",
                [
                    {
                        "type": "response_item",
                        "timestamp": "2026-06-25T12:05:30Z",
                        "payload": {
                            "type": "function_call",
                            "name": "shell_command",
                            "arguments": json.dumps({"command": "git commit -s -- tools/report.json"}),
                        },
                    },
                ],
            )
            manifest = write_hook_manifest(
                home,
                native_bash_hook_command(),
            )
            backup = manifest.with_name("hooks.json.before-native-dos-hook.bak")
            backup.write_text("backup\n", encoding="utf-8")
            marker = datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc).timestamp()
            os.utime(manifest, (marker, marker))
            os.utime(backup, (marker, marker))
            write_jsonl(
                root / ".dos" / "streams" / f"{thread}.jsonl",
                [{"op": "STEP", "tool_name": "Bash", "ts": "2026-06-25T12:00:00Z"}],
            )
            write_jsonl(
                root / ".dos" / "metrics" / "observations.jsonl",
                [
                    {
                        "verb": "pretool",
                        "outcome": "warn",
                        "rung": "admission",
                        "tree_known": False,
                        "dialect": "codex",
                        "latency_ms": 9.0,
                        "ts": "2026-06-25T12:00:30Z",
                    },
                ],
            )
            gate_dir = root / "gates"
            gate_dir.mkdir()
            gates = [
                gate_dir / "git-add.json",
                gate_dir / "git-commit.json",
                gate_dir / "git-push.json",
            ]
            write_gate_report(gates[0], tool="git_add", reason="DEFAULT_DENY", created_at="2026-06-25T12:10:00Z")
            write_gate_report(gates[1], tool="git_commit", reason="DEFAULT_DENY", created_at="2026-06-25T12:10:01Z")
            write_gate_report(gates[2], tool="git_push", reason="POLICY_BLOCK", created_at="2026-06-25T12:10:02Z")

            report = mod.build_report(root, home, limit=10, since_days=3650, gate_reports=gates)

            self.assertEqual(report["git_gate_evidence"]["status"], "PASS")
            self.assertEqual(report["git_gate_evidence"]["proved_at"], "2026-06-25T12:10:02Z")
            self.assertEqual(
                report["codex_hook_fast_path"]["post_repair_command_shapes"]["shell_family_counts"],
                {"git_write": 1},
            )
            self.assertEqual(report["git_gate_evidence"]["post_gate_command_shapes"]["tool_call_rows"], 0)
            self.assertEqual(report["actionability"]["status"], "PASS")
            self.assertIn("HISTORICAL_GIT_WRITE_BEFORE_STRUCTURED_GATE", report["actionability"]["residual"])
            self.assertEqual(report["actionability"]["post_git_gate_mutating_shell_family_counts"], {})

    def test_fail_on_warn_returns_nonzero_for_fast_path_warn(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            home = root / "codex-home"
            thread = "019efde3-6794-7401-93a1-e97e6bd72a9c"
            session = home / "sessions" / "2026" / "06" / "25" / f"rollout-{thread}.jsonl"
            write_jsonl(
                session,
                [
                    {
                        "type": "response_item",
                        "timestamp": "2026-06-25T12:05:30Z",
                        "payload": {
                            "type": "function_call",
                            "name": "shell_command",
                            "arguments": json.dumps({"command": "rg needle docs"}),
                        },
                    },
                    {
                        "type": "response_item",
                        "timestamp": "2026-06-25T12:05:40Z",
                        "payload": {
                            "type": "function_call",
                            "name": "shell_command",
                            "arguments": json.dumps({"command": "echo x > tools/out.txt"}),
                        },
                    },
                    {
                        "type": "response_item",
                        "timestamp": "2026-06-25T12:05:50Z",
                        "payload": {
                            "type": "function_call",
                            "name": "update_plan",
                            "arguments": "{}",
                        },
                    },
                ],
            )
            write_hook_manifest(
                home,
                "$py = Get-Command python; & $py.Source -m dos.cli hook pretool --workspace . --dialect codex",
            )
            write_jsonl(
                root / ".dos" / "streams" / f"{thread}.jsonl",
                [{"op": "STEP", "tool_name": "Read", "ts": "2026-06-25T12:00:00Z"}],
            )
            write_jsonl(
                root / ".dos" / "metrics" / "observations.jsonl",
                [
                    {
                        "verb": "pretool",
                        "outcome": "passthrough",
                        "rung": "admission",
                        "tree_known": True,
                        "ts": "2026-06-25T12:00:10Z",
                    }
                ],
            )

            with contextlib.redirect_stdout(io.StringIO()):
                rc = mod.main(
                    [
                        "--repo-root",
                        str(root),
                        "--codex-home",
                        str(home),
                        "--since-days",
                        "3650",
                        "--fail-on-warn",
                    ]
                )
            self.assertEqual(rc, 1)

    def test_post_repair_observations_are_split_from_recent_window(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            home = root / "codex-home"
            thread = "019efde3-6794-7401-93a1-e97e6bd72a9c"
            session = home / "sessions" / "2026" / "06" / "25" / f"rollout-{thread}.jsonl"
            write_jsonl(
                session,
                [
                    {
                        "type": "response_item",
                        "timestamp": "2026-06-25T12:05:30Z",
                        "payload": {
                            "type": "function_call",
                            "name": "shell_command",
                            "arguments": json.dumps({"command": "rg needle docs"}),
                        },
                    },
                    {
                        "type": "response_item",
                        "timestamp": "2026-06-25T12:05:40Z",
                        "payload": {
                            "type": "function_call",
                            "name": "shell_command",
                            "arguments": json.dumps({"command": "echo x > tools/out.txt"}),
                        },
                    },
                    {
                        "type": "response_item",
                        "timestamp": "2026-06-25T12:05:50Z",
                        "payload": {
                            "type": "function_call",
                            "name": "update_plan",
                            "arguments": "{}",
                        },
                    },
                ],
            )
            manifest = write_hook_manifest(
                home,
                native_bash_hook_command(),
            )
            backup = manifest.with_name("hooks.json.before-native-dos-hook.bak")
            backup.write_text("backup\n", encoding="utf-8")
            marker = datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc).timestamp()
            os.utime(manifest, (marker, marker))
            os.utime(backup, (marker, marker))
            write_jsonl(
                root / ".dos" / "streams" / f"{thread}.jsonl",
                [{"op": "STEP", "tool_name": "Bash", "ts": "2026-06-25T11:59:00Z"}],
            )
            write_jsonl(
                root / ".dos" / "metrics" / "observations.jsonl",
                [
                    {
                        "verb": "pretool",
                        "outcome": "warn",
                        "rung": "admission",
                        "tree_known": False,
                        "dialect": "codex",
                        "ts": "2026-06-25T11:59:30Z",
                    },
                    {
                        "verb": "pretool",
                        "outcome": "passthrough",
                        "rung": "provenance",
                        "dialect": "codex",
                        "latency_ms": 9.0,
                        "ts": "2026-06-25T12:00:30Z",
                    },
                    {
                        "verb": "posttool",
                        "outcome": "passthrough",
                        "rung": "none",
                        "dialect": "codex",
                        "latency_ms": 4.0,
                        "ts": "2026-06-25T12:01:00Z",
                    },
                ],
            )

            report = mod.build_report(root, home, limit=10, since_days=3650)

            post = report["codex_hook_fast_path"]["post_repair_observations"]
            self.assertEqual(post["status"], "PASS")
            self.assertEqual(post["observations"], 2)
            self.assertEqual(post["pretool_calls"], 1)
            self.assertEqual(post["unknown_tree_admission_warnings"], 0)
            self.assertEqual(post["delegate_count"], 0)
            self.assertEqual(report["codex_hook_fast_path"]["backups"], ["plugins/cache/dos/dos-kernel/0.28.0/hooks/hooks.json.before-native-dos-hook.bak"])
            shapes = report["codex_hook_fast_path"]["post_repair_command_shapes"]
            self.assertEqual(shapes["status"], "PASS")
            self.assertEqual(shapes["tool_counts"], {"shell_command": 2, "update_plan": 1})
            self.assertEqual(
                shapes["shell_shape_counts"],
                {
                    "non_shell_tool": 1,
                    "shell_in_tree_or_safe_write_target": 1,
                    "shell_no_write_target_detected": 1,
                },
            )
            self.assertEqual(
                shapes["shell_family_counts"],
                {
                    "search_rg": 1,
                    "shell_redirect": 1,
                },
            )
            self.assertEqual(report["actionability"]["status"], "PASS")
            self.assertEqual(report["actionability"]["delegate_count"], 0)
            self.assertIn("HOST_SHELL_OPACITY", report["actionability"]["residual"])
            encoded = json.dumps(report)
            self.assertNotIn("rg needle docs", encoded)
            self.assertNotIn("echo x", encoded)

            debt = mod.render_debt_packet(report)
            self.assertIn("Codex DOS Host-Opacity Debt", debt)
            self.assertIn("HOST_SHELL_OPACITY", debt)
            self.assertIn("shell_no_write_target_detected", debt)
            self.assertIn("search_rg", debt)
            self.assertIn("shell_redirect", debt)
            self.assertNotIn("rg needle docs", debt)
            self.assertNotIn("echo x", debt)

            with contextlib.redirect_stdout(io.StringIO()):
                rc = mod.main(
                    [
                        "--repo-root",
                        str(root),
                        "--codex-home",
                        str(home),
                        "--since-days",
                        "3650",
                        "--fail-on-actionable-warn",
                        "--max-delegates",
                        "0",
                    ]
                )
            self.assertEqual(rc, 0)

            debt_path = root / "debt.md"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = mod.main(
                    [
                        "--repo-root",
                        str(root),
                        "--codex-home",
                        str(home),
                        "--since-days",
                        "3650",
                        "--out-debt",
                        str(debt_path),
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertTrue(debt_path.exists())
            written = debt_path.read_text(encoding="utf-8")
            self.assertIn("Requested Upstream Improvement", written)
            self.assertNotIn("rg needle docs", written)

    def test_fail_on_warn_returns_nonzero_for_strict_gate(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            home = root / "codex-home"
            thread = "019efde3-6794-7401-93a1-e97e6bd72a9c"
            session = home / "sessions" / "2026" / "06" / "25" / f"rollout-{thread}.jsonl"
            write_jsonl(session, [{"type": "response_item"}])
            write_jsonl(
                root / ".dos" / "streams" / f"{thread}.jsonl",
                [
                    {"op": "STEP", "tool_name": "Bash", "ts": "2026-06-25T12:00:00Z"},
                    {"op": "STEP", "tool_name": "Bash", "ts": "2026-06-25T12:01:00Z"},
                ],
            )
            write_jsonl(
                root / ".dos" / "metrics" / "observations.jsonl",
                [
                    {
                        "verb": "pretool",
                        "outcome": "warn",
                        "rung": "admission",
                        "tree_known": False,
                        "ts": "2026-06-25T12:00:10Z",
                    }
                ],
            )

            with contextlib.redirect_stdout(io.StringIO()):
                rc = mod.main(
                    [
                        "--repo-root",
                        str(root),
                        "--codex-home",
                        str(home),
                        "--since-days",
                        "3650",
                        "--fail-on-warn",
                        "--max-unknown-tree-rate",
                        "0.01",
                        "--max-delegates",
                        "0",
                    ]
                )
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
