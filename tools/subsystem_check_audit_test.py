#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import subsystem_check_audit as audit


class SubsystemCheckAuditTest(unittest.TestCase):
    def test_run_command_records_pass_and_digest(self) -> None:
        check = {
            "id": "ok",
            "subsystem": "Dummy",
            "kind": "unit",
            "cwd": "",
            "argv": [sys.executable, "-c", "print('ok')"],
            "proves": "pass path",
            "does_not_prove": "anything external",
        }
        result = audit.run_command(Path.cwd(), check, timeout_s=30)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(len(result["output_sha256"]), 64)
        self.assertIn("ok", result["stdout_tail"])

    def test_run_command_records_failure(self) -> None:
        check = {
            "id": "fail",
            "subsystem": "Dummy",
            "kind": "unit",
            "cwd": "",
            "argv": [sys.executable, "-c", "raise SystemExit(7)"],
            "proves": "failure path",
            "does_not_prove": "anything external",
        }
        result = audit.run_command(Path.cwd(), check, timeout_s=30)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_code"], 7)

    def test_compare_baseline_flags_pass_to_nonpass(self) -> None:
        report = {
            "profile": "smoke",
            "runs": [
                {
                    "id": "arch-request-path",
                    "command": "go test ./internal/architest",
                    "status": "failed",
                    "exit_code": 1,
                    "elapsed_ms": 10,
                }
            ],
        }
        baseline = {
            "schema": audit.BASELINE_SCHEMA,
            "profile": "smoke",
            "checks": [
                {
                    "id": "arch-request-path",
                    "command": "go test ./internal/architest",
                    "expected_status": "passed",
                    "elapsed_ms": 10,
                }
            ],
        }
        comparison = audit.compare_baseline(report, baseline, 0.5, 0, False)
        self.assertEqual(comparison["status"], "failed")
        self.assertEqual(comparison["failures"][0]["reason"], "pass_to_nonpass")

    def test_compare_baseline_flags_command_drift(self) -> None:
        report = {"profile": "smoke", "runs": [{"id": "x", "command": "new", "status": "passed", "elapsed_ms": 10}]}
        baseline = {"schema": audit.BASELINE_SCHEMA, "profile": "smoke", "checks": [{"id": "x", "command": "old", "expected_status": "passed"}]}
        comparison = audit.compare_baseline(report, baseline, 0.5, 0, False)
        self.assertEqual(comparison["status"], "failed")
        self.assertEqual(comparison["failures"][0]["reason"], "command_drift")

    def test_duration_regression_is_warning_by_default(self) -> None:
        report = {"profile": "smoke", "runs": [{"id": "x", "command": "cmd", "status": "passed", "elapsed_ms": 200}]}
        baseline = {"schema": audit.BASELINE_SCHEMA, "profile": "smoke", "checks": [{"id": "x", "command": "cmd", "expected_status": "passed", "elapsed_ms": 100}]}
        comparison = audit.compare_baseline(report, baseline, 0.5, 0, False)
        self.assertEqual(comparison["status"], "passed_with_warnings")
        self.assertEqual(comparison["warnings"][0]["reason"], "duration_regression")

    def test_duration_slack_absorbs_small_fast_test_noise(self) -> None:
        report = {"profile": "smoke", "runs": [{"id": "x", "command": "cmd", "status": "passed", "elapsed_ms": 200}]}
        baseline = {"schema": audit.BASELINE_SCHEMA, "profile": "smoke", "checks": [{"id": "x", "command": "cmd", "expected_status": "passed", "elapsed_ms": 100}]}
        comparison = audit.compare_baseline(report, baseline, 0.5, 2000, False)
        self.assertEqual(comparison["status"], "passed")
        self.assertEqual(comparison["warnings"], [])

    def test_expand_globs_resolves_pytest_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = root / "tools"
            tools.mkdir()
            (tools / "api_host_a_test.py").write_text("", encoding="utf-8")
            (tools / "api_host_b_test.py").write_text("", encoding="utf-8")
            expanded = audit.expand_globs(root, "", ["python", "-m", "pytest", "tools/api_host_*_test.py"])
        self.assertEqual([Path(arg).name for arg in expanded[3:]], ["api_host_a_test.py", "api_host_b_test.py"])

    def test_markdown_surfaces_summary(self) -> None:
        report = {
            "profile": "smoke",
            "generated_at": "now",
            "app_version": "dev",
            "git": {"branch": "main", "head": "abc123", "dirty": False, "dirty_count": 0},
            "summary": {"passed": 1, "total": 1, "baseline_failures": 0, "baseline_warnings": 0},
            "comparison": {"status": "not_configured", "failures": [], "warnings": []},
            "runs": [{"id": "x", "subsystem": "Dummy", "status": "passed", "elapsed_ms": 1, "command": "cmd"}],
        }
        md = audit.markdown(report)
        self.assertIn("Subsystem Check Audit", md)
        self.assertIn("Checks passed: 1/1", md)
        self.assertIn("`x`", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
