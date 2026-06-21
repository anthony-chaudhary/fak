#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import permission_source_audit as audit


class Handler(BaseHTTPRequestHandler):
    flaky_seen = 0

    def do_GET(self) -> None:
        if self.path == "/ok":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"Alpha source claim. Beta second claim.")
            return
        if self.path == "/flaky":
            Handler.flaky_seen += 1
            if Handler.flaky_seen == 1:
                self.send_response(503)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"temporary outage")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"Alpha source claim. Beta second claim.")
            return
        if self.path == "/missing":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"Alpha only.")
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


class PermissionSourceAuditTest(unittest.TestCase):
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

    def test_audit_source_verifies_all_patterns(self) -> None:
        row = audit.audit_source({
            "id": "ok",
            "system": "x",
            "url": self.base + "/ok",
            "claim": "ok",
            "patterns": [r"alpha source", r"beta second"],
        }, timeout_s=2)
        self.assertEqual(row["status"], "VERIFIED")
        self.assertTrue(row["version"])
        self.assertTrue(row["body_sha256"])
        self.assertTrue(all(c["matched"] for c in row["checks"]))

    def test_audit_source_retries_transient_fetch_errors(self) -> None:
        Handler.flaky_seen = 0
        row = audit.audit_source({
            "id": "flaky",
            "system": "x",
            "url": self.base + "/flaky",
            "claim": "flaky",
            "patterns": [r"alpha source", r"beta second"],
        }, timeout_s=2, retries=1, retry_sleep_s=0)
        self.assertEqual(row["status"], "VERIFIED")
        self.assertEqual(Handler.flaky_seen, 2)

    def test_audit_source_fails_missing_pattern(self) -> None:
        row = audit.audit_source({
            "id": "missing",
            "system": "x",
            "url": self.base + "/missing",
            "claim": "missing",
            "patterns": [r"alpha", r"beta"],
        }, timeout_s=2)
        self.assertEqual(row["status"], "FAILED")
        self.assertFalse(row["checks"][1]["matched"])

    def test_build_report_gate(self) -> None:
        report = audit.build_report([
            {
                "id": "ok",
                "system": "x",
                "url": self.base + "/ok",
                "claim": "ok",
                "patterns": [r"alpha", r"beta"],
            },
        ], timeout_s=2)
        self.assertTrue(report["summary"]["source_audit_gate"])
        self.assertTrue(report["app_version"])
        self.assertEqual(report["sources"][0]["version"], report["app_version"])

    def test_cli_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            json_path = root / "audit.json"
            md_path = root / "audit.md"
            old_sources = audit.SOURCES
            try:
                audit.SOURCES = [{
                    "id": "ok",
                    "system": "x",
                    "url": self.base + "/ok",
                    "claim": "ok",
                    "patterns": [r"alpha", r"beta"],
                }]
                rc = audit.main(["--out", str(json_path), "--markdown", str(md_path), "--timeout-s", "2"])
            finally:
                audit.SOURCES = old_sources
            self.assertEqual(rc, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], audit.SCHEMA)
            self.assertTrue(data["app_version"])
            self.assertIn("Permission-System Source Audit", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
