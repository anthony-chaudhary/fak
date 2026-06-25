#!/usr/bin/env python3
"""Hermetic tests for tools/openai_hosted_live_pilot.py."""
from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import tempfile
import unittest
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


def prereqs(*, hosted_ready: bool) -> dict:
    blockers = [] if hosted_ready else ["OPENAI_API_KEY is not set"]
    return {
        "schema": "fak-openai-live-prereq-audit/1",
        "status": "READY" if hosted_ready else "BLOCKED_ENV",
        "hosted_openai_ready": hosted_ready,
        "agents_sdk_ready": False,
        "blockers": blockers,
        "env": {"OPENAI_API_KEY_set": hosted_ready},
        "packages": {"openai": "2.41.0", "openai-agents": None},
    }


class OpenAIHostedLivePilotTest(unittest.TestCase):
    def test_collect_blocks_without_hosted_prereqs(self) -> None:
        mod = load()
        with (
            mock.patch.object(mod, "collect_prereqs", return_value=prereqs(hosted_ready=False)),
            mock.patch.object(mod, "run_guard_probe") as guard,
            mock.patch.object(mod, "run_openai_probe") as openai,
        ):
            payload = mod.collect("gpt-5.5")
        self.assertEqual(payload["status"], "BLOCKED_ENV")
        self.assertIn("OPENAI_API_KEY is not set", payload["blockers"])
        guard.assert_not_called()
        openai.assert_not_called()

    def test_collect_passes_when_guard_and_hosted_call_pass(self) -> None:
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
            payload = mod.collect("gpt-5.5")
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["guard"], guard_payload)
        self.assertEqual(payload["hosted_openai"], hosted_payload)
        self.assertNotIn("raw hosted OpenAI response text", json.dumps(payload["hosted_openai"]))

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
            payload = mod.collect("gpt-5.5")
        self.assertEqual(payload["status"], "FAIL")
        self.assertNotIn("sk-", json.dumps(payload))

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
