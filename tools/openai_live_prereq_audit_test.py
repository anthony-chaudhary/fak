#!/usr/bin/env python3
"""Hermetic tests for tools/openai_live_prereq_audit.py."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parent / "openai_live_prereq_audit.py"


def load():
    spec = importlib.util.spec_from_file_location("openai_live_prereq_audit", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def fake_dist(versions: dict[str, str | None]):
    def inner(name: str) -> str | None:
        return versions.get(name)

    return inner


class OpenAILivePrereqAuditTest(unittest.TestCase):
    def test_collect_ready_when_hosted_and_agents_sdk_are_available(self) -> None:
        mod = load()
        with (
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-secret-value", "OPENAI_BASE_URL": "https://example.invalid"}, clear=True),
            mock.patch.object(
                mod,
                "dist_version",
                side_effect=fake_dist({"openai": "2.41.0", "openai-agents": "0.3.0", "agents": None}),
            ),
            mock.patch.object(
                mod,
                "module_info",
                side_effect=lambda name: {"installed": True, "file": f"/site-packages/{name.replace('.', '/')}.py", "has_custom_span": name == "agents"},
            ),
            mock.patch.object(mod, "codex_auth_info", return_value={"codex_login_ready": False, "blockers": ["Codex auth.json is not present"]}),
        ):
            payload = mod.collect()
        self.assertEqual(payload["status"], "READY")
        self.assertTrue(payload["hosted_openai_ready"])
        self.assertTrue(payload["platform_api_ready"])
        self.assertFalse(payload["codex_login_ready"])
        self.assertTrue(payload["agents_sdk_ready"])
        self.assertEqual(payload["blockers"], [])
        self.assertNotIn("test-secret-value", json.dumps(payload))

    def test_collect_partial_when_hosted_ready_but_agents_sdk_is_missing(self) -> None:
        mod = load()
        with (
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-secret-value"}, clear=True),
            mock.patch.object(
                mod,
                "dist_version",
                side_effect=fake_dist({"openai": "2.41.0", "openai-agents": None, "agents": None}),
            ),
            mock.patch.object(mod, "module_info", side_effect=lambda name: {"installed": False}),
            mock.patch.object(mod, "codex_auth_info", return_value={"codex_login_ready": False, "blockers": ["Codex auth.json is not present"]}),
        ):
            payload = mod.collect()
        self.assertEqual(payload["status"], "PARTIAL")
        self.assertTrue(payload["hosted_openai_ready"])
        self.assertTrue(payload["platform_api_ready"])
        self.assertFalse(payload["codex_login_ready"])
        self.assertFalse(payload["agents_sdk_ready"])
        self.assertIn("openai-agents distribution is not installed", payload["blockers"])
        self.assertNotIn("test-secret-value", json.dumps(payload))

    def test_collect_partial_when_codex_login_ready_without_api_key(self) -> None:
        mod = load()
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch.object(
                mod,
                "dist_version",
                side_effect=fake_dist({"openai": "2.41.0", "openai-agents": None, "agents": None}),
            ),
            mock.patch.object(mod, "module_info", side_effect=lambda name: {"installed": False}),
            mock.patch.object(
                mod,
                "codex_auth_info",
                return_value={
                    "auth_json_present": True,
                    "auth_mode": "chatgpt",
                    "codex_cli_present": True,
                    "access_token_present": True,
                    "refresh_token_present": True,
                    "access_token_expired": False,
                    "codex_login_ready": True,
                    "blockers": [],
                },
            ),
        ):
            payload = mod.collect()
        self.assertEqual(payload["status"], "PARTIAL")
        self.assertTrue(payload["hosted_openai_ready"])
        self.assertFalse(payload["platform_api_ready"])
        self.assertTrue(payload["codex_login_ready"])
        self.assertTrue(payload["auth_sources"]["codex_login"])
        self.assertNotIn("OPENAI_API_KEY is not set", payload["blockers"])
        self.assertIn("openai-agents distribution is not installed", payload["blockers"])

    def test_collect_flags_local_shadow_agents_module(self) -> None:
        mod = load()
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch.object(
                mod,
                "dist_version",
                side_effect=fake_dist({"openai": "2.41.0", "openai-agents": None, "agents": None}),
            ),
            mock.patch.object(
                mod,
                "module_info",
                side_effect=lambda name: {"installed": True, "file": f"C:/work/job/{name.replace('.', '/')}/__init__.py"},
            ),
            mock.patch.object(mod, "codex_auth_info", return_value={"codex_login_ready": False, "blockers": ["Codex auth.json is not present"]}),
        ):
            payload = mod.collect()
        self.assertEqual(payload["status"], "BLOCKED_ENV")
        self.assertFalse(payload["hosted_openai_ready"])
        self.assertFalse(payload["agents_sdk_ready"])
        self.assertIn("OPENAI_API_KEY is not set", payload["blockers"])
        self.assertIn("openai-agents distribution is not installed", payload["blockers"])
        self.assertIn("importable agents module is not an installed OpenAI Agents SDK distribution", payload["blockers"])

    def test_main_writes_json_and_markdown_without_secret_values(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "prereq.json"
            md = Path(td) / "prereq.md"
            with (
                mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-secret-value"}, clear=True),
                mock.patch.object(
                    mod,
                    "dist_version",
                    side_effect=fake_dist({"openai": "2.41.0", "openai-agents": None, "agents": None}),
                ),
                mock.patch.object(mod, "module_info", side_effect=lambda name: {"installed": False}),
                mock.patch.object(mod, "codex_auth_info", return_value={"codex_login_ready": False, "blockers": ["Codex auth.json is not present"]}),
            ):
                rc = mod.main(["--out", str(out), "--markdown", str(md)])
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())
            self.assertTrue(md.is_file())
            combined = out.read_text(encoding="utf-8") + md.read_text(encoding="utf-8")
            self.assertIn("OPENAI_API_KEY_set", combined)
            self.assertNotIn("test-secret-value", combined)


if __name__ == "__main__":
    unittest.main()
