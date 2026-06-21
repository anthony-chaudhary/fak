#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import api_host_bridge_gate as gate


class APIHostBridgeGateTest(unittest.TestCase):
    def test_run_command_records_pass_and_digest(self) -> None:
        witness = {"id": "ok", "claim": "ok", "required": True, "command": "py", "cwd": "", "argv": [sys.executable, "-c", "print('ok')"]}
        result = gate.run_command(Path.cwd(), witness, timeout_s=30)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["version"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(len(result["output_sha256"]), 64)

    def test_run_command_records_failure(self) -> None:
        witness = {"id": "fail", "claim": "fail", "required": True, "command": "py", "cwd": "", "argv": [sys.executable, "-c", "raise SystemExit(7)"]}
        result = gate.run_command(Path.cwd(), witness, timeout_s=30)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_code"], 7)

    def test_run_command_records_start_error(self) -> None:
        witness = {"id": "bad-cwd", "claim": "bad cwd", "required": True, "command": "py", "cwd": "does-not-exist", "argv": [sys.executable, "-c", "print('ok')"]}
        result = gate.run_command(Path.cwd(), witness, timeout_s=30)
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["exit_code"])
        self.assertIn(result["error_type"], {"FileNotFoundError", "NotADirectoryError"})
        self.assertEqual(len(result["output_sha256"]), 64)

    def test_index_only_is_not_executable_success(self) -> None:
        report = gate.build_report(execute=False)
        self.assertTrue(report["app_version"])
        self.assertTrue(all(r["version"] == report["app_version"] for r in report["runs"]))
        self.assertTrue(report["matrix_summary"]["bridge_covered"])
        self.assertFalse(report["summary"]["executable_bridge_covered"])
        self.assertTrue(all(r["status"] in {"not_run", "skipped"} for r in report["runs"]))

    def test_markdown_surfaces_status(self) -> None:
        md = gate.markdown(gate.build_report(execute=False))
        self.assertIn("API-Host Bridge Gate", md)
        self.assertIn("Executable bridge covered: no", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
