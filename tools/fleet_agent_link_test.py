#!/usr/bin/env python3
"""Tests for Fleet Agent Link's in-memory and stdio JSON-RPC surfaces."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_agent_link as link  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


class FleetAgentLinkTest(unittest.TestCase):
    def decode(self, text: str) -> dict:
        return json.loads(text)

    def test_registry_exposes_expected_methods_and_scopes(self) -> None:
        registry = link.build_registry()

        self.assertEqual(registry["agent.info"].scope, "read")
        self.assertEqual(registry["laptop.status"].scope, "read")
        self.assertEqual(registry["laptop.accept"].scope, "act")
        self.assertNotIn("exec", registry)

    def test_agent_info_returns_manifest_repo_and_tools(self) -> None:
        result = link.dispatch("agent.info", {}, ROOT)

        self.assertEqual(result["schema"], link.RESULT_SCHEMA)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["call"]["policy"]["scope"], "read")
        self.assertIn("host", result["result"])
        self.assertIn("repo", result["result"])
        self.assertIn("tools", result["result"])
        self.assertIn("manifest", result["result"])

    def test_agent_ping_is_cheap_liveness_method(self) -> None:
        result = link.dispatch("agent.ping", {}, ROOT)

        self.assertTrue(result["result"]["ok"])
        self.assertIsInstance(result["result"]["monotonic_ms"], int)

    def test_manifest_names_a2a_projection_without_enabling_it(self) -> None:
        result = link.dispatch("protocol.manifest", {}, ROOT)
        manifest = result["result"]

        self.assertEqual(manifest["schema"], link.LINK_SCHEMA)
        self.assertEqual(manifest["adapters"]["a2a"], "agent-card-projection-only")
        self.assertEqual(manifest["adapters"]["a2a_http"], "planned-edge-adapter")
        self.assertIn("generic_exec", manifest["non_goals"])
        self.assertTrue(any(method["name"] == "laptop.accept" for method in manifest["methods"]))

    def test_a2a_agent_card_projects_registry_methods_and_scopes(self) -> None:
        card = link.a2a_agent_card(
            link.build_registry(),
            ROOT,
            url="https://fleet.example.test/a2a",
            tenant="workspace-a",
        )

        self.assertEqual(card["supportedInterfaces"][0]["url"], "https://fleet.example.test/a2a")
        self.assertEqual(card["supportedInterfaces"][0]["protocolVersion"], link.A2A_PROTOCOL_VERSION)
        self.assertEqual(card["supportedInterfaces"][0]["tenant"], "workspace-a")
        methods = {skill["metadata"]["fleet_method"]: skill for skill in card["skills"]}

        self.assertEqual(methods["laptop.status"]["metadata"]["fleet_policy_scope"], "read")
        self.assertFalse(methods["laptop.status"]["metadata"]["fleet_requires_confirmation"])
        self.assertEqual(methods["laptop.accept"]["metadata"]["fleet_policy_scope"], "act")
        self.assertTrue(methods["laptop.accept"]["metadata"]["fleet_requires_confirmation"])
        self.assertEqual(methods["protocol.manifest"]["metadata"]["fleet_manifest_schema"], link.LINK_SCHEMA)
        self.assertNotIn("generic_exec", json.dumps(card, sort_keys=True))

    def test_a2a_lint_accepts_generated_card(self) -> None:
        card = link.a2a_agent_card(link.build_registry(), ROOT, api_key_header="X-Fleet-Key")
        report = link.lint_a2a_agent_card(card, require_auth=True)

        self.assertTrue(report["ok"], report)
        self.assertIn("laptop.accept", report["advertised_methods"])

    def test_a2a_agent_card_can_filter_to_read_scope(self) -> None:
        card = link.a2a_agent_card(link.build_registry(), ROOT, allowed_scopes={"read"})
        report = link.lint_a2a_agent_card(card)

        self.assertTrue(report["ok"], report)
        self.assertIn("laptop.status", report["advertised_methods"])
        self.assertIn("agent.info", report["advertised_methods"])
        self.assertNotIn("laptop.accept", report["advertised_methods"])

    def test_a2a_lint_rejects_unknown_method_and_scope_drift(self) -> None:
        card = link.a2a_agent_card(link.build_registry(), ROOT)
        card["skills"][0]["metadata"]["fleet_method"] = "unknown.method"
        card["skills"][1]["metadata"]["fleet_policy_scope"] = "act"

        report = link.lint_a2a_agent_card(card)
        failed = {check["name"] for check in report["checks"] if check["status"] == "fail"}

        self.assertFalse(report["ok"])
        self.assertIn("skill_0_method_known", failed)
        self.assertIn("skill_1_scope", failed)

    def test_a2a_lint_can_require_signature(self) -> None:
        card = link.a2a_agent_card(link.build_registry(), ROOT)
        report = link.lint_a2a_agent_card(card, require_signature=True)
        failed = {check["name"] for check in report["checks"] if check["status"] == "fail"}

        self.assertFalse(report["ok"])
        self.assertIn("signature_required", failed)

    def test_jsonrpc_request_response(self) -> None:
        raw = json.dumps(link.request_object("agent.ping", {}, "p1"))
        response = self.decode(link.handle_text(raw, ROOT))

        self.assertEqual(response["jsonrpc"], "2.0")
        self.assertEqual(response["id"], "p1")
        self.assertEqual(response["result"]["schema"], link.RESULT_SCHEMA)

    def test_jsonrpc_notification_returns_no_response(self) -> None:
        raw = json.dumps({"jsonrpc": "2.0", "method": "agent.ping", "params": {}})

        self.assertEqual(link.handle_text(raw, ROOT), "")

    def test_unknown_method_returns_method_not_found(self) -> None:
        raw = json.dumps(link.request_object("unknown.method", {}, "bad"))
        response = self.decode(link.handle_text(raw, ROOT))

        self.assertEqual(response["id"], "bad")
        self.assertEqual(response["error"]["code"], link.METHOD_NOT_FOUND)

    def test_invalid_json_returns_parse_error(self) -> None:
        response = self.decode(link.handle_text("{not json", ROOT))

        self.assertEqual(response["id"], None)
        self.assertEqual(response["error"]["code"], link.PARSE_ERROR)

    def test_invalid_params_must_be_object(self) -> None:
        raw = json.dumps({"jsonrpc": "2.0", "id": "x", "method": "agent.ping", "params": []})
        response = self.decode(link.handle_text(raw, ROOT))

        self.assertEqual(response["id"], "x")
        self.assertEqual(response["error"]["code"], link.INVALID_PARAMS)

    def test_laptop_check_builds_reviewed_runner_command(self) -> None:
        params = {
            "require_nvidia": True,
            "require_cuda_toolchain": True,
            "cpu_only": True,
            "fast": True,
            "wsl_distro": "Ubuntu-24.04",
            "out": "fak/experiments/gpu/check.json",
            "timeout_s": 12,
        }
        with mock.patch.object(link, "run_command", return_value={
            "argv": [],
            "cwd": str(ROOT),
            "exit_code": 0,
            "duration_ms": 1,
            "stdout_tail": "",
            "stderr_tail": "",
            "timed_out": False,
        }) as run_command:
            result = link.dispatch("laptop.check", params, ROOT)

        argv, cwd, timeout = run_command.call_args.args
        self.assertEqual(argv[1], str(ROOT / "tools" / "fak_laptop_test.py"))
        self.assertIn("check", argv)
        self.assertIn("--require-nvidia", argv)
        self.assertIn("--require-cuda-toolchain", argv)
        self.assertIn("--cpu-only", argv)
        self.assertIn("--fast", argv)
        self.assertIn("Ubuntu-24.04", argv)
        self.assertIn("fak/experiments/gpu/check.json", argv)
        self.assertEqual(cwd, ROOT)
        self.assertEqual(timeout, 12)
        self.assertEqual(result["call"]["policy"]["scope"], "act")

    def test_laptop_accept_builds_report_flags(self) -> None:
        params = {
            "cpu_only": True,
            "full_cpu": True,
            "wsl_distro": "Ubuntu",
            "precheck_report": "pre.json",
            "check_report": "check.json",
            "run_report": "run.json",
        }
        with mock.patch.object(link, "run_command", return_value={
            "argv": [],
            "cwd": str(ROOT),
            "exit_code": 1,
            "duration_ms": 1,
            "stdout_tail": "",
            "stderr_tail": "failed",
            "timed_out": False,
        }) as run_command:
            result = link.dispatch("laptop.accept", params, ROOT)

        argv = run_command.call_args.args[0]
        self.assertIn("accept", argv)
        self.assertIn("--cpu-only", argv)
        self.assertIn("--full-cpu", argv)
        self.assertIn("--wsl-distro", argv)
        self.assertIn("Ubuntu", argv)
        self.assertIn("--precheck-report", argv)
        self.assertIn("pre.json", argv)
        self.assertIn("--check-report", argv)
        self.assertIn("check.json", argv)
        self.assertIn("--run-report", argv)
        self.assertIn("run.json", argv)
        self.assertEqual(result["status"], "failed")

    def test_laptop_boolean_param_rejects_strings(self) -> None:
        raw = json.dumps(link.request_object("laptop.accept", {"cpu_only": "true"}, "bad-param"))
        response = self.decode(link.handle_text(raw, ROOT))

        self.assertEqual(response["error"]["code"], link.INVALID_PARAMS)

    def test_cli_request_pipes_to_serve_once(self) -> None:
        request = subprocess.run(
            [sys.executable, "tools/fleet_agent_link.py", "request", "agent.ping", "--id", "pipe"],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        response = subprocess.run(
            [sys.executable, "tools/fleet_agent_link.py", "serve-once"],
            cwd=str(ROOT),
            input=request.stdout,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        data = self.decode(response.stdout)
        self.assertEqual(data["id"], "pipe")
        self.assertEqual(data["result"]["schema"], link.RESULT_SCHEMA)

    def test_cli_a2a_card_pipes_to_lint(self) -> None:
        card = subprocess.run(
            [
                sys.executable,
                "tools/fleet_agent_link.py",
                "a2a-card",
                "--url",
                "https://fleet.example.test/a2a",
                "--tenant",
                "workspace-a",
                "--scope",
                "read",
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        lint = subprocess.run(
            [sys.executable, "tools/fleet_agent_link.py", "a2a-lint"],
            cwd=str(ROOT),
            input=card.stdout,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        report = self.decode(lint.stdout)
        self.assertTrue(report["ok"], report)
        self.assertIn("laptop.status", report["advertised_methods"])
        self.assertNotIn("laptop.accept", report["advertised_methods"])

    def test_remote_command_renders_expected_shell(self) -> None:
        ps = link.remote_command(r"C:\work\fleet-laptop-proof", "powershell")
        posix = link.remote_command("/work/fleet-laptop-proof", "posix")

        self.assertEqual(ps, "Set-Location 'C:\\work\\fleet-laptop-proof'; py -3 tools\\fleet_agent_link.py serve-once")
        self.assertEqual(posix, "cd /work/fleet-laptop-proof && python3 tools/fleet_agent_link.py serve-once")


if __name__ == "__main__":
    unittest.main()
