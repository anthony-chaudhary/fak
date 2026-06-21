"""Hermetic tests for the GLM-5.2 serving witness runner."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import glm52_serving_witness as witness  # noqa: E402


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _write(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            self._write(200, {"data": [{"id": "zai-org/GLM-5.2"}]})
            return
        if self.path == "/get_server_info":
            self._write(200, {"engine": "sglang", "version": "test"})
            return
        self._write(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        has_tool_result = any(m.get("role") == "tool" for m in body.get("messages", []))
        payload = {
            "choices": [{"message": {"role": "assistant", "content": "GLM52_OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        if has_tool_result:
            payload["fak"] = {
                "result_admissions": [{
                    "tool_call_id": "call_glm52_witness",
                    "tool": "fetch_url",
                    "verdict": {"kind": "QUARANTINE", "reason": "PROMPT_INJECTION"},
                }]
            }
        self._write(200, payload)


class Glm52ServingWitnessTest(unittest.TestCase):
    def test_dry_run_records_fak_cache_reset_command(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report.json"
            with redirect_stdout(StringIO()):
                code = witness.main([
                    "--base-url", "http://127.0.0.1:9/v1",
                    "--dry-run",
                    "--engine-cache-engine", "sglang",
                    "--engine-cache-base-url", "http://127.0.0.1:9",
                    "--engine-cache-admin-key-env", "GLM52_CACHE_KEY",
                    "--engine-cache-idle-timeout-s", "30",
                    "--engine-cache-require-exact-span",
                    "--context-length", "1000000",
                    "--gpu-memory-total-gb", "640",
                    "--out", str(out),
                ])
            self.assertEqual(code, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["full_size_serving_witness"], "PLANNED")
            command = report["gateway"]["command"]
            self.assertIn("--engine-cache-engine", command)
            self.assertIn("sglang", command)
            self.assertIn("--engine-cache-idle-timeout", command)
            self.assertIn("30s", command)
            self.assertIn("--engine-cache-require-exact-span", command)
            self.assertEqual(report["engine_cache"]["fallback_scope"], "whole_prefix_cache")
            self.assertTrue(report["engine_cache"]["require_exact_span"])

    def test_fake_gateway_report_passes_required_acceptance_fields(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}/v1"
        gateway = f"http://127.0.0.1:{server.server_port}"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report.json"
            md = Path(td) / "report.md"
            with redirect_stdout(StringIO()):
                code = witness.main([
                    "--base-url", base,
                    "--gateway-url", gateway,
                    "--engine-version", "sglang-test",
                    "--context-length", "1000000",
                    "--gpu-memory-total-gb", "640",
                    "--engine-cache-engine", "sglang",
                    "--out", str(out),
                    "--markdown", str(md),
                ])
            self.assertEqual(code, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["full_size_serving_witness"], "PASS")
            self.assertEqual(report["gateway_quarantine"]["status"], "PASS")
            self.assertIn("QUARANTINE", report["gateway_quarantine"]["result_admission_kinds"])
            self.assertEqual(report["acceptance"]["metrics_captured"]["status"], "PASS")
            self.assertIn("external full-size serving report", md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
