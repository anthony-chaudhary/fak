#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import api_host_bridge_matrix as matrix


class APIHostBridgeMatrixTest(unittest.TestCase):
    def test_required_witnesses_resolve(self) -> None:
        report = matrix.build_report()
        self.assertEqual(report["schema"], matrix.SCHEMA)
        self.assertTrue(report["app_version"])
        self.assertTrue(all(w["version"] == report["app_version"] for w in report["witnesses"]))
        self.assertTrue(report["summary"]["bridge_covered"])
        self.assertEqual(report["summary"]["resolved_required_witnesses"], report["summary"]["required_witnesses"])

    def test_provider_shapes_are_explicit(self) -> None:
        report = matrix.build_report()
        self.assertEqual(report["summary"]["provider_shapes_covered"], ["anthropic", "gemini", "openai-compatible", "xai"])

    def test_required_witnesses_have_executable_commands(self) -> None:
        for witness in matrix.build_report()["witnesses"]:
            if not witness["required"]:
                continue
            self.assertEqual(witness["cwd"], "fak", witness["id"])
            self.assertTrue(witness["argv"], witness["id"])
            self.assertEqual(witness["argv"][0], "go", witness["id"])
            self.assertIn("-count=1", witness["argv"], witness["id"])

    def test_source_token_checks_are_strings(self) -> None:
        for witness in matrix.WITNESSES:
            for check in witness["checks"]:
                if check["type"] == "source_token":
                    self.assertIsInstance(check["token"], str, witness["id"])

    def test_cli_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            json_path = root / "matrix.json"
            md_path = root / "matrix.md"
            self.assertEqual(matrix.main(["--out", str(json_path), "--markdown", str(md_path)]), 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertTrue(data["app_version"])
            self.assertTrue(data["summary"]["bridge_covered"])
            self.assertIn("API-Host Bridge Matrix", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
