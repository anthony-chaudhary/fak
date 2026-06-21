#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import api_host_readiness_probe as probe


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/ok/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"object":"list","data":[{"id":"m1"},{"id":"m2"}]}')
            return
        if self.path == "/auth/models":
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"Authentication required"}')
            return
        if self.path == "/billing/models":
            self.send_response(402)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"code":"no_payment_method"}}')
            return
        if self.path == "/edge/models":
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"title":"Error 1010: Access denied","error_name":"browser_signature_banned"}')
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


class APHostReadinessProbeTest(unittest.TestCase):
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

    def test_parse_target(self) -> None:
        self.assertEqual(
            probe.parse_target("n|http://x|KEY|m"),
            {"name": "n", "base_url": "http://x", "api_key_env": "KEY", "model_hint": "m"},
        )
        with self.assertRaises(ValueError):
            probe.parse_target("only-name")
        with self.assertRaises(ValueError):
            probe.parse_target("n|")
        with self.assertRaises(ValueError):
            probe.parse_target("n|/relative")

    def test_probe_models_confirmed(self) -> None:
        row = probe.probe_target({"name": "ok", "base_url": self.base + "/ok", "api_key_env": "", "model_hint": ""})
        self.assertEqual(row["status"], "MODELS_CONFIRMED")
        self.assertEqual(row["models"], ["m1", "m2"])

    def test_probe_typed_http_states(self) -> None:
        auth = probe.probe_target({"name": "auth", "base_url": self.base + "/auth", "api_key_env": "", "model_hint": ""})
        billing = probe.probe_target({"name": "billing", "base_url": self.base + "/billing", "api_key_env": "", "model_hint": ""})
        edge = probe.probe_target({"name": "edge", "base_url": self.base + "/edge", "api_key_env": "", "model_hint": ""})
        self.assertEqual(auth["status"], "AUTH_REQUIRED")
        self.assertEqual(billing["status"], "BILLING_REQUIRED")
        self.assertEqual(edge["status"], "ACCESS_DENIED")

    def test_missing_env_skips_network(self) -> None:
        old = os.environ.pop("NO_SUCH_API_KEY_FOR_TEST", None)
        try:
            row = probe.probe_target({"name": "missing", "base_url": self.base + "/ok", "api_key_env": "NO_SUCH_API_KEY_FOR_TEST", "model_hint": ""})
            self.assertEqual(row["status"], "AUTH_ENV_MISSING")
            self.assertIsNone(row["http_status"])
        finally:
            if old is not None:
                os.environ["NO_SUCH_API_KEY_FOR_TEST"] = old

    def test_invalid_target_is_typed_without_network(self) -> None:
        row = probe.probe_target({"name": "bad", "base_url": "", "api_key_env": "", "model_hint": ""})
        self.assertEqual(row["status"], "INVALID_TARGET")
        self.assertIn("base_url", row["error"])

        report = probe.build_report([{"name": "bad", "base_url": "", "api_key_env": "", "model_hint": ""}])
        self.assertFalse(report["summary"]["readiness_gate"])
        self.assertEqual(report["summary"]["invalid_targets"], 1)

    def test_build_report_and_cli(self) -> None:
        targets = [
            {"name": "ok", "base_url": self.base + "/ok", "api_key_env": "", "model_hint": ""},
            {"name": "auth", "base_url": self.base + "/auth", "api_key_env": "", "model_hint": ""},
        ]
        report = probe.build_report(targets)
        self.assertTrue(report["summary"]["readiness_gate"])
        self.assertEqual(report["summary"]["models_confirmed"], 1)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            json_path = root / "probe.json"
            md_path = root / "probe.md"
            rc = probe.main([
                "--target", f"ok|{self.base}/ok",
                "--out", str(json_path),
                "--markdown", str(md_path),
            ])
            self.assertEqual(rc, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], probe.SCHEMA)
            self.assertIn("API-Host Readiness Probe", md_path.read_text(encoding="utf-8"))

    def test_from_roster_loads_openai_compatible_targets(self) -> None:
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
            json_path = root / "probe.json"
            rc = probe.main([
                "--from-roster", str(roster_path),
                "--out", str(json_path),
            ])
            self.assertEqual(rc, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["summary"]["targets"], 1)
            self.assertEqual(data["probes"][0]["name"], "ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)
