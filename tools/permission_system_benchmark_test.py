#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import permission_system_benchmark as bench


class PermissionSystemBenchmarkTest(unittest.TestCase):
    def test_metrics_pin_core_contrasts(self) -> None:
        report = bench.build_report()
        metrics = {m["system"]: m for m in report["metrics"]}
        fak = metrics["fak_dos_gateway"]
        claude = metrics["claude_code_auto"]
        bypass = metrics["bypass_permissions"]
        self.assertEqual(fak["deterministic_controls"], fak["risk_scenarios"])
        self.assertEqual(fak["result_admission_verdict"], "QUARANTINE")
        self.assertTrue(fak["has_api_host_bridge"])
        self.assertEqual(fak["api_host_bridge_controls"], fak["api_host_bridge_dimensions"])
        self.assertEqual(fak["api_host_result_quarantine_verdict"], "QUARANTINE")
        self.assertEqual(claude["deterministic_controls"], 0)
        self.assertEqual(claude["api_host_bridge_controls"], 0)
        self.assertEqual(claude["known_max_false_negative_pct"], 17.0)
        self.assertEqual(claude["result_admission_verdict"], "WARNING")
        self.assertEqual(bypass["unguarded_risk_allows"], bypass["risk_scenarios"])
        self.assertEqual(bypass["api_host_bridge_controls"], 0)

    def test_bridge_dimensions_are_explicit(self) -> None:
        report = bench.build_report()
        self.assertTrue(report["app_version"])
        dimensions = {row["id"] for row in report["api_host_bridge_dimensions"]}
        self.assertIn("host_agnostic_openai_compatible_proxy", dimensions)
        self.assertIn("synthetic_host_profile_conformance", dimensions)
        self.assertIn("pre_execution_tool_call_admission", dimensions)
        self.assertIn("pre_send_tool_result_quarantine", dimensions)
        self.assertIn("roster_driven_host_qualification", dimensions)
        self.assertIn("dos_style_executable_bridge_proof", dimensions)
        self.assertTrue(all(row["version"] == report["app_version"] for row in report["api_host_bridge_dimensions"]))
        self.assertTrue(all(row["version"] == report["app_version"] for row in report["scenarios"]))
        self.assertTrue(all(row["version"] == report["app_version"] for row in report["systems"]))
        self.assertTrue(all(row["version"] == report["app_version"] for row in report["sources"].values()))
        self.assertTrue(all(row["version"] == report["app_version"] for rows in report["outcomes"].values() for row in rows.values()))
        fak = report["api_host_bridge_outcomes"]["fak_dos_gateway"]
        self.assertEqual(fak["pre_send_tool_result_quarantine"]["verdict"], "QUARANTINE")
        self.assertEqual(fak["pre_send_tool_result_quarantine"]["version"], report["app_version"])

    def test_markdown_contains_bridge_and_sources(self) -> None:
        md = bench.markdown(bench.build_report())
        self.assertIn("Permission-System Benchmark", md)
        self.assertIn("api_host_bridge_proof.py", md)
        self.assertIn("api_host_compat_contract.py", md)
        self.assertIn("api_host_acceptance_probe.py", md)
        self.assertIn("api_host_bridge_gate.py", md)
        self.assertIn("api_host_live_inventory.py", md)
        self.assertIn("api_host_readiness_probe.py", md)
        self.assertIn("api_host_qualification.py", md)
        self.assertIn("TestChatProxyOpenAICompatibleHostProfileConformance", md)
        self.assertIn("TestChatProxyOpenAICompatibleToolResultsAreQuarantinedPreSend", md)
        self.assertIn("permission_source_audit.py", md)
        self.assertIn("https://www.anthropic.com/engineering/claude-code-auto-mode", md)

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            json_path = root / "report.json"
            md_path = root / "report.md"
            self.assertEqual(bench.main(["--out", str(json_path), "--markdown", str(md_path)]), 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], bench.SCHEMA)
            self.assertTrue(data["app_version"])
            self.assertIn("FAK/DOS gateway", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
