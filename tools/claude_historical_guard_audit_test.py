#!/usr/bin/env python3
"""Hermetic tests for tools/claude_historical_guard_audit.py."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parent / "claude_historical_guard_audit.py"


def load():
    spec = importlib.util.spec_from_file_location("claude_historical_guard_audit", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_transcript(path: Path) -> None:
    rows = [
        {
            "type": "assistant",
            "message": {
                "id": "msg-1",
                "content": [
                    {"type": "tool_use", "id": "u1", "name": "Bash", "input": {"command": "rm -rf ./x"}},
                    {"type": "tool_use", "id": "u2", "name": "Read", "input": {"file_path": "README.md"}},
                ],
            },
        },
        {
            "type": "assistant",
            "message": {
                "id": "msg-1",
                "content": [
                    {"type": "tool_use", "id": "u1", "name": "Bash", "input": {"command": "rm -rf ./x"}},
                ],
            },
        },
        {
            "type": "user",
            "timestamp": "2026-06-25T12:00:00Z",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "content": "permission error hook blocked secret result"}],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def fake_runner(argv: list[str]) -> dict:
    tool = argv[argv.index("--tool") + 1]
    args = json.loads(argv[argv.index("--args") + 1])
    if tool == "Bash" and "rm -rf" in args.get("command", ""):
        return {
            "verdict": "DENY",
            "reason": "POLICY_BLOCK",
            "by": "monitor",
            "claim": "Bash.command deny_regex /rm/",
            "args_digest": "deny123",
            "args_bytes": 24,
        }
    return {"verdict": "ALLOW", "by": "monitor", "args_digest": "allow123", "args_bytes": 12}


class ClaudeHistoricalGuardAuditTest(unittest.TestCase):
    def test_iter_tool_uses_dedupes_duplicate_assistant_lines(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "session.jsonl"
            write_transcript(path)
            calls = mod.iter_tool_uses(path)
        self.assertEqual(len(calls), 2)
        self.assertEqual([c["tool"] for c in calls], ["Bash", "Read"])

    def test_collect_replays_unique_calls_without_copying_args_or_results(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_transcript(root / "session.jsonl")
            with mock.patch.object(mod, "find_fak", return_value="fak"):
                payload = mod.collect(
                    root=str(root),
                    policy="policy.json",
                    since_days=None,
                    max_sessions=10,
                    max_calls=100,
                    runner=fake_runner,
                )
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["sessions_discovered"], 1)
        self.assertEqual(payload["tool_calls_seen"], 2)
        self.assertEqual(payload["unique_tool_calls"], 2)
        self.assertEqual(payload["verdict_counts"], {"ALLOW": 1, "DENY": 1})
        self.assertEqual(payload["reason_counts"], {"POLICY_BLOCK": 1})
        self.assertEqual(payload["transcript_shape"]["summarized_sessions"], 1)
        self.assertIn("HOOK_OR_API_WALL_FEEDBACK", payload["transcript_shape"]["evidence_tag_counts"])
        self.assertIn("HOST_PERMISSION_INTERRUPT", payload["transcript_shape"]["evidence_tag_counts"])
        self.assertEqual(payload["top_friction_sessions"][0]["root_label"], root.name)
        serialized = json.dumps(payload)
        self.assertNotIn("rm -rf", serialized)
        self.assertNotIn("README.md", serialized)
        self.assertNotIn("secret result", serialized)
        self.assertNotIn("tool_result", serialized)

    def test_collect_all_accounts_scans_sibling_claude_roots(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "home"
            first = home / ".claude" / "projects" / "C--work-fak"
            second = home / ".claude-worker" / "projects" / "C--work-fak"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            write_transcript(first / "session-a.jsonl")
            write_transcript(second / "session-b.jsonl")
            with mock.patch.dict(os.environ, {"USERPROFILE": str(home)}, clear=False), mock.patch.object(mod, "find_fak", return_value="fak"):
                payload = mod.collect(
                    root=str(first),
                    all_accounts=True,
                    policy="policy.json",
                    since_days=None,
                    max_sessions=10,
                    max_calls=100,
                    runner=fake_runner,
                )
        self.assertTrue(payload["all_accounts"])
        self.assertEqual(payload["sessions_discovered"], 2)
        self.assertIn(".claude/C--work-fak", payload["root_labels"])
        alt_labels = [label for label in payload["root_labels"] if label != ".claude/C--work-fak"]
        self.assertEqual(len(alt_labels), 1)
        self.assertTrue(alt_labels[0].startswith(".claude-"))
        self.assertTrue(alt_labels[0].endswith("/C--work-fak"))
        self.assertNotIn("worker", alt_labels[0])
        self.assertEqual(payload["transcript_shape"]["summarized_sessions"], 2)

    def test_collect_reports_no_corpus(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            payload = mod.collect(root=str(Path(td) / "missing"), runner=fake_runner)
        self.assertEqual(payload["status"], "NO_CORPUS")
        self.assertIn("no Claude Code transcript", payload["blockers"][0])

    def test_main_writes_json_and_markdown(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "root"
            root.mkdir()
            write_transcript(root / "session.jsonl")
            out = Path(td) / "audit.json"
            md = Path(td) / "audit.md"
            with mock.patch.object(mod, "find_fak", return_value="fak"), mock.patch.object(mod, "default_runner", side_effect=fake_runner):
                rc = mod.main([
                    "--root", str(root),
                    "--out", str(out),
                    "--markdown", str(md),
                    "--since-days", "999",
                ])
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())
            self.assertTrue(md.is_file())
            rendered = md.read_text(encoding="utf-8")
            self.assertIn("Claude Code historical guard replay", rendered)
            self.assertIn("Transcript Friction Signals", rendered)
            self.assertIn("Top Friction Sessions", rendered)
            self.assertNotIn("secret result", rendered)
            self.assertNotIn("tool_result", rendered)


if __name__ == "__main__":
    unittest.main()
