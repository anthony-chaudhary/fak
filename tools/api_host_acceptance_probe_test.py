#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import api_host_acceptance_probe as acceptance


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/ok/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"object":"list","data":[{"id":"m1"}]}')
            return
        if self.path == "/auth/models":
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"Authentication required"}')
            return
        if self.path == "/weird/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"models":["m1"]}')
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


class APIHostAcceptanceProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.thread.join(timeout=5)

    def test_parse_target_and_provider_aliases(self) -> None:
        self.assertEqual(
            acceptance.parse_target("n|grok|http://x|KEY|m"),
            {"name": "n", "provider": "xai", "base_url": "http://x", "api_key_env": "KEY", "model_hint": "m"},
        )
        with self.assertRaises(ValueError):
            acceptance.parse_target("n|openai-compatible")
        with self.assertRaises(ValueError):
            acceptance.parse_target("n|openai-compatible|")

    def test_openai_compatible_ready_for_live_bridge_run(self) -> None:
        row = acceptance.classify_target(
            {"name": "ok", "provider": "openai-compatible", "base_url": self.base + "/ok", "api_key_env": "", "model_hint": "m1"},
            timeout_s=2,
            probe_missing_auth=False,
            sweep_rows=[],
        )
        self.assertEqual(row["status"], "READY_FOR_LIVE_BRIDGE_RUN")
        self.assertEqual(row["contract_class"], "openai_compatible_upstream")
        self.assertIn("run_transcript_adapter_sweep.ps1", row["next_live_command"])
        self.assertIn("-ApiBaseUrl", row["next_live_command"])
        self.assertNotIn("-Provider", row["next_live_command"])

    def test_typed_external_blocker_and_shape_mismatch(self) -> None:
        auth = acceptance.classify_target(
            {"name": "auth", "provider": "openai-compatible", "base_url": self.base + "/auth", "api_key_env": "", "model_hint": ""},
            timeout_s=2,
            probe_missing_auth=False,
            sweep_rows=[],
        )
        weird = acceptance.classify_target(
            {"name": "weird", "provider": "openai-compatible", "base_url": self.base + "/weird", "api_key_env": "", "model_hint": ""},
            timeout_s=2,
            probe_missing_auth=False,
            sweep_rows=[],
        )
        self.assertEqual(auth["status"], "AUTH_REQUIRED")
        self.assertEqual(weird["status"], "MODELS_SHAPE_MISMATCH")

    def test_missing_api_key_env_is_typed_without_network(self) -> None:
        old = os.environ.pop("NO_SUCH_API_KEY_FOR_ACCEPTANCE_TEST", None)
        try:
            row = acceptance.classify_target(
                {
                    "name": "missing",
                    "provider": "openai-compatible",
                    "base_url": self.base + "/ok",
                    "api_key_env": "NO_SUCH_API_KEY_FOR_ACCEPTANCE_TEST",
                    "model_hint": "",
                },
                timeout_s=2,
                probe_missing_auth=False,
                sweep_rows=[],
            )
            self.assertEqual(row["status"], "NEEDS_AUTH_ENV")
            self.assertEqual(row["readiness_status"], "AUTH_ENV_MISSING")
        finally:
            if old is not None:
                os.environ["NO_SUCH_API_KEY_FOR_ACCEPTANCE_TEST"] = old

    def test_native_and_direct_wires_are_supported_but_unprobed(self) -> None:
        for provider, cls in [
            ("anthropic", "native_provider_transcript_adapters"),
            ("gemini", "native_provider_transcript_adapters"),
            ("direct-http", "direct_kernel_http_syscall"),
            ("direct-mcp", "direct_kernel_mcp_syscall"),
        ]:
            row = acceptance.classify_target(
                {"name": provider, "provider": provider, "base_url": "http://example.invalid", "api_key_env": "", "model_hint": ""},
                timeout_s=2,
                probe_missing_auth=False,
                sweep_rows=[],
            )
            self.assertEqual(row["status"], "WIRE_SUPPORTED_UNPROBED")
            self.assertEqual(row["contract_class"], cls)

    def test_unsupported_wire_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = acceptance.build_report([
                {"name": "ok", "provider": "openai-compatible", "base_url": self.base + "/ok", "api_key_env": "", "model_hint": ""},
                {"name": "bad", "provider": "unknown-provider", "base_url": "http://example.invalid", "api_key_env": "", "model_hint": ""},
            ], timeout_s=2, root=Path(td))
            self.assertFalse(report["summary"]["acceptance_gate"])
            self.assertEqual(report["summary"]["unsupported_wire"], 1)

    def test_invalid_target_fails_gate_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = acceptance.build_report([
                {"name": "bad", "provider": "openai-compatible", "base_url": "", "api_key_env": "", "model_hint": ""},
            ], timeout_s=2, root=Path(td))
            self.assertFalse(report["summary"]["acceptance_gate"])
            self.assertEqual(report["summary"]["invalid_targets"], 1)
            self.assertEqual(report["targets"][0]["status"], "INVALID_TARGET")
            self.assertIsNone(report["targets"][0]["probe"])

    def test_malformed_sweep_summary_fails_gate_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bad = root / "fak/experiments/agent-live/transcript-adapter-sweep-bad/sweep-summary.json"
            bad.parent.mkdir(parents=True)
            bad.write_text("{bad json", encoding="utf-8")

            report = acceptance.build_report([
                {"name": "direct", "provider": "direct-http", "base_url": "http://example.invalid", "api_key_env": "", "model_hint": ""},
            ], timeout_s=2, root=root)

            self.assertFalse(report["summary"]["acceptance_gate"])
            self.assertEqual(report["summary"]["sweep_artifact_errors"], 1)
            self.assertEqual(report["artifact_errors"][0]["path"], "fak/experiments/agent-live/transcript-adapter-sweep-bad/sweep-summary.json")
            self.assertIn("invalid JSON", report["artifact_errors"][0]["error"])

    def test_non_object_sweep_rows_fail_gate_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bad = root / "fak/experiments/agent-live/transcript-adapter-sweep-bad/sweep-summary.json"
            bad.parent.mkdir(parents=True)
            bad.write_text(json.dumps([[]]), encoding="utf-8")

            report = acceptance.build_report([
                {"name": "direct", "provider": "direct-http", "base_url": "http://example.invalid", "api_key_env": "", "model_hint": ""},
            ], timeout_s=2, root=root)

            self.assertFalse(report["summary"]["acceptance_gate"])
            self.assertEqual(report["summary"]["sweep_artifact_errors"], 1)
            self.assertEqual(report["artifact_errors"][0]["row_index"], 0)
            self.assertIn("not a JSON object", report["artifact_errors"][0]["error"])

    def test_live_sweep_failure_overrides_models_ready(self) -> None:
        row = acceptance.classify_target(
            {"name": "ok", "provider": "openai-compatible", "base_url": self.base + "/ok", "api_key_env": "", "model_hint": "m1"},
            timeout_s=2,
            probe_missing_auth=False,
            sweep_rows=[{
                "generated_at": "2026-06-18T00:00:00-07:00",
                "kind": "api",
                "base_url": self.base + "/ok",
                "model": "m1",
                "status": "failed",
                "error": 'fak: planner: HTTP 402: {"error":{"code":"no_payment_method"}}',
            }],
        )
        self.assertEqual(row["readiness_status"], "MODELS_CONFIRMED")
        self.assertEqual(row["status"], "BILLING_REQUIRED")

    def test_live_sweep_success_overrides_models_ready(self) -> None:
        row = acceptance.classify_target(
            {"name": "ok", "provider": "openai-compatible", "base_url": self.base + "/ok", "api_key_env": "", "model_hint": "m1"},
            timeout_s=2,
            probe_missing_auth=False,
            sweep_rows=[{
                "generated_at": "2026-06-18T00:00:00-07:00",
                "kind": "api",
                "base_url": self.base + "/ok",
                "model": "m1",
                "status": "ok",
                "live": True,
                "transcript_sha": "abc",
            }],
        )
        self.assertEqual(row["status"], "LIVE_BRIDGE_CONFIRMED")

    def test_cli_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            json_path = root / "acceptance.json"
            md_path = root / "acceptance.md"
            rc = acceptance.main([
                "--target", f"ok|openai-compatible|{self.base}/ok||m1",
                "--root", str(root),
                "--out", str(json_path),
                "--markdown", str(md_path),
            ])
            self.assertEqual(rc, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], acceptance.SCHEMA)
            self.assertIn("API-Host Acceptance Probe", md_path.read_text(encoding="utf-8"))

    def test_from_roster_classifies_all_supported_wire_targets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            roster_path = root / "roster.json"
            roster_path.write_text(json.dumps({
                "targets": [
                    {
                        "name": "ok",
                        "provider": "openai-compatible",
                        "contract_class": "openai_compatible_upstream",
                        "base_url": self.base + "/ok",
                        "api_key_env": "",
                        "model_hint": "m1",
                        "status": "SUPPORTED_TEMPLATE",
                    },
                    {
                        "name": "native",
                        "provider": "anthropic",
                        "contract_class": "native_provider_transcript_adapters",
                        "base_url": "https://example.invalid",
                        "api_key_env": "",
                        "model_hint": "",
                        "status": "SUPPORTED_TEMPLATE",
                    },
                ],
            }), encoding="utf-8")
            json_path = root / "acceptance.json"
            rc = acceptance.main([
                "--from-roster", str(roster_path),
                "--root", str(root),
                "--out", str(json_path),
            ])
            self.assertEqual(rc, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            statuses = {row["name"]: row["status"] for row in data["targets"]}
            self.assertEqual(statuses["ok"], "READY_FOR_LIVE_BRIDGE_RUN")
            self.assertEqual(statuses["native"], "WIRE_SUPPORTED_UNPROBED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
