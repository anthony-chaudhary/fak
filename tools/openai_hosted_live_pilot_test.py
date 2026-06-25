#!/usr/bin/env python3
"""Hermetic tests for tools/openai_hosted_live_pilot.py."""
from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parent / "openai_hosted_live_pilot.py"


def load():
    spec = importlib.util.spec_from_file_location("openai_hosted_live_pilot", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def subprocess_completed(*, returncode: int, stdout: str, stderr: str):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def prereqs(*, hosted_ready: bool, source: str = "platform_api_key") -> dict:
    platform_ready = hosted_ready and source == "platform_api_key"
    codex_ready = hosted_ready and source == "codex_login"
    blockers = [] if hosted_ready else ["OPENAI_API_KEY is not set", "Codex auth.json is not present"]
    return {
        "schema": "fak-openai-live-prereq-audit/1",
        "status": "READY" if hosted_ready else "BLOCKED_ENV",
        "hosted_openai_ready": hosted_ready,
        "platform_api_ready": platform_ready,
        "codex_login_ready": codex_ready,
        "auth_sources": {
            "platform_api_key": platform_ready,
            "codex_login": codex_ready,
        },
        "agents_sdk_ready": False,
        "blockers": blockers,
        "env": {"OPENAI_API_KEY_set": platform_ready},
        "codex_auth": {
            "auth_json_present": codex_ready,
            "auth_mode": "chatgpt" if codex_ready else None,
            "codex_login_ready": codex_ready,
            "blockers": [] if codex_ready else ["Codex auth.json is not present"],
        },
        "packages": {"openai": "2.41.0", "openai-agents": None},
    }


class OpenAIHostedLivePilotTest(unittest.TestCase):
    def test_collect_blocks_without_hosted_prereqs(self) -> None:
        mod = load()
        with (
            mock.patch.object(mod, "collect_prereqs", return_value=prereqs(hosted_ready=False)),
            mock.patch.object(mod, "run_guard_probe") as guard,
            mock.patch.object(mod, "run_openai_probe") as openai,
            mock.patch.object(mod, "run_codex_login_probe") as codex,
        ):
            payload = mod.collect("gpt-5.5")
        self.assertEqual(payload["status"], "BLOCKED_ENV")
        self.assertIn("OPENAI_API_KEY is not set", payload["blockers"])
        guard.assert_not_called()
        openai.assert_not_called()
        codex.assert_not_called()

    def test_collect_passes_when_guard_and_api_key_hosted_call_pass(self) -> None:
        mod = load()
        guard_payload = {
            "status": "PASS",
            "dangerous_attempt": {
                "verdict": {"kind": "DENY", "reason": "POLICY_BLOCK"},
                "executed": False,
            },
            "useful_continuation": {
                "verdict": {"kind": "ALLOW"},
                "admit_verdict": {"kind": "DEFER"},
            },
        }
        hosted_payload = {
            "status": "PASS",
            "auth_source": "platform_api_key",
            "model": "gpt-5.5",
            "response_id_present": True,
            "contains_expected_marker": True,
            "output_text_sha256": "abc123",
            "output_text_len": 21,
        }
        with (
            mock.patch.object(mod, "collect_prereqs", return_value=prereqs(hosted_ready=True)),
            mock.patch.object(mod, "kernel_context", return_value=contextlib.nullcontext("http://fak.invalid")),
            mock.patch.object(mod, "run_guard_probe", return_value=guard_payload),
            mock.patch.object(mod, "run_openai_probe", return_value=hosted_payload),
        ):
            payload = mod.collect("gpt-5.5", auth_mode="api-key")
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["auth_source"], "platform_api_key")
        self.assertEqual(payload["guard"], guard_payload)
        self.assertEqual(payload["hosted_openai"], hosted_payload)
        self.assertNotIn("raw hosted OpenAI response text", json.dumps(payload["hosted_openai"]))

    def test_collect_prefers_codex_login_when_available(self) -> None:
        mod = load()
        guard_payload = {
            "status": "PASS",
            "dangerous_attempt": {
                "verdict": {"kind": "DENY", "reason": "POLICY_BLOCK"},
                "executed": False,
            },
            "useful_continuation": {
                "verdict": {"kind": "ALLOW"},
                "admit_verdict": {"kind": "DEFER"},
            },
        }
        hosted_payload = {
            "status": "PASS",
            "auth_source": "codex_login",
            "model": "",
            "codex_exec_exit_code": 0,
            "contains_expected_marker": True,
            "output_text_sha256": "abc123",
            "output_text_len": 21,
            "json_event_count": 3,
        }
        with (
            mock.patch.object(mod, "collect_prereqs", return_value=prereqs(hosted_ready=True, source="codex_login")),
            mock.patch.object(mod, "kernel_context", return_value=contextlib.nullcontext("http://fak.invalid")),
            mock.patch.object(mod, "run_guard_probe", return_value=guard_payload),
            mock.patch.object(mod, "run_openai_probe") as openai,
            mock.patch.object(mod, "run_codex_login_probe", return_value=hosted_payload) as codex,
        ):
            payload = mod.collect("gpt-5.5")
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["auth_source"], "codex_login")
        openai.assert_not_called()
        codex.assert_called_once()

    def test_collect_fails_when_hosted_call_fails(self) -> None:
        mod = load()
        with (
            mock.patch.object(mod, "collect_prereqs", return_value=prereqs(hosted_ready=True)),
            mock.patch.object(mod, "kernel_context", return_value=contextlib.nullcontext("http://fak.invalid")),
            mock.patch.object(mod, "run_guard_probe", return_value={"status": "PASS"}),
            mock.patch.object(
                mod,
                "run_openai_probe",
                return_value={
                    "status": "FAIL",
                    "model": "gpt-5.5",
                    "error_type": "AuthenticationError",
                    "error": "bad key [redacted-openai-key]",
                },
            ),
        ):
            payload = mod.collect("gpt-5.5", auth_mode="api-key")
        self.assertEqual(payload["status"], "FAIL")
        self.assertNotIn("sk-", json.dumps(payload))

    def test_run_codex_login_probe_hashes_output_without_api_key_env(self) -> None:
        mod = load()
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs.get("env") or {}
            output_arg = cmd[cmd.index("--output-last-message") + 1]
            Path(output_arg).write_text(mod.CODEX_LOGIN_MARKER, encoding="utf-8")
            return subprocess_completed(returncode=0, stdout='{"type":"task_started"}\n', stderr="")

        with (
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-secret-value"}, clear=False),
            mock.patch.object(mod.subprocess, "run", side_effect=fake_run),
        ):
            payload = mod.run_codex_login_probe(codex_exe="codex")
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["auth_source"], "codex_login")
        self.assertTrue(payload["contains_expected_marker"])
        self.assertTrue(payload["openai_api_key_env_removed"])
        self.assertNotIn("OPENAI_API_KEY", captured["env"])
        self.assertNotIn("test-secret-value", json.dumps(payload))

    def test_main_writes_artifacts_without_raw_secret(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "pilot.json"
            md = Path(td) / "pilot.md"
            with mock.patch.object(mod, "collect_prereqs", return_value=prereqs(hosted_ready=False)):
                rc = mod.main(["--out", str(out), "--markdown", str(md), "--model", "gpt-5.5"])
            self.assertEqual(rc, 0)
            combined = out.read_text(encoding="utf-8") + md.read_text(encoding="utf-8")
            self.assertIn("BLOCKED_ENV", combined)
            self.assertNotIn("sk-", combined)


if __name__ == "__main__":
    unittest.main()
