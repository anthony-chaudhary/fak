#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import api_host_roster as roster


class APIHostRosterTest(unittest.TestCase):
    def test_default_roster_is_supported_and_broad(self) -> None:
        report = roster.build_report()
        self.assertEqual(report["schema"], roster.SCHEMA)
        self.assertTrue(report["summary"]["roster_gate"])
        self.assertGreaterEqual(report["summary"]["targets"], 10)
        self.assertGreaterEqual(report["summary"]["openai_compatible_templates"], 10)
        self.assertEqual(report["summary"]["invalid_targets"], 0)
        self.assertEqual(report["summary"]["unsupported_wire"], 0)
        self.assertEqual(report["summary"]["duplicate_names"], 0)
        self.assertIn("fak api-host acceptance", report["bulk_commands"]["acceptance"])
        self.assertIn("fak api-host readiness", report["bulk_commands"]["readiness"])

    def test_invalid_target_fails_gate(self) -> None:
        report = roster.build_report([
            {"name": "bad", "provider": "openai-compatible", "base_url": "not-url", "api_key_env": "", "model_hint": ""},
        ])
        self.assertFalse(report["summary"]["roster_gate"])
        self.assertEqual(report["summary"]["invalid_targets"], 1)
        self.assertEqual(report["targets"][0]["status"], "INVALID_TARGET")

    def test_unsupported_wire_fails_gate(self) -> None:
        report = roster.build_report([
            {"name": "bad", "provider": "custom", "base_url": "https://example.invalid/v1", "api_key_env": "", "model_hint": ""},
        ])
        self.assertFalse(report["summary"]["roster_gate"])
        self.assertEqual(report["summary"]["unsupported_wire"], 1)

    def test_duplicate_names_fail_gate(self) -> None:
        target = {"name": "dup", "provider": "openai-compatible", "base_url": "https://example.invalid/v1", "api_key_env": "", "model_hint": ""}
        report = roster.build_report([target, target])
        self.assertFalse(report["summary"]["roster_gate"])
        self.assertEqual(report["summary"]["duplicate_names"], 1)
        self.assertEqual(report["duplicate_names"], ["dup"])

    def test_cli_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            json_path = root / "roster.json"
            md_path = root / "roster.md"
            rc = roster.main([
                "--target", "ok|openai-compatible|https://example.invalid/v1|KEY|model",
                "--out", str(json_path),
                "--markdown", str(md_path),
            ])
            self.assertEqual(rc, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], roster.SCHEMA)
            self.assertIn("API-Host Roster", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
