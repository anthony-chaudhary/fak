#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import api_host_compat_contract as contract


def write_json(root: Path, rel_path: str, data: object) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def good_artifacts(root: Path) -> None:
    write_json(root, contract.DEFAULT_PATHS["matrix"], {
        "schema": "fak.api-host-bridge-matrix.v1",
        "summary": {"provider_shapes_covered": ["anthropic", "gemini", "openai-compatible", "xai"]},
        "witnesses": [
            {"id": "openai_compatible_gateway", "status": "resolved"},
            {"id": "native_provider_adapters", "status": "resolved"},
            {"id": "provider_proxy_end_to_end", "status": "resolved"},
            {"id": "host_agnostic_openai_compatible", "status": "resolved"},
            {"id": "openai_compatible_host_profiles", "status": "resolved"},
            {"id": "direct_http_syscall", "status": "resolved"},
            {"id": "direct_mcp_syscall", "status": "resolved"},
        ],
    })
    write_json(root, contract.DEFAULT_PATHS["gate"], {
        "schema": "fak.api-host-bridge-gate.v1",
        "runs": [
            {"id": "openai_compatible_gateway", "status": "passed"},
            {"id": "native_provider_adapters", "status": "passed"},
            {"id": "provider_proxy_end_to_end", "status": "passed"},
            {"id": "host_agnostic_openai_compatible", "status": "passed"},
            {"id": "openai_compatible_host_profiles", "status": "passed"},
            {"id": "direct_http_syscall", "status": "passed"},
            {"id": "direct_mcp_syscall", "status": "passed"},
        ],
    })
    write_json(root, contract.DEFAULT_PATHS["live"], {
        "schema": "fak.api-host-live-inventory.v1",
        "summary": {
            "live_inventory_gate": True,
            "live_frontier_successes": 2,
            "local_openai_compatible_successes": 1,
            "incomplete_or_unclassified": 0,
        },
    })
    write_json(root, contract.DEFAULT_PATHS["readiness"], {
        "schema": "fak.api-host-readiness.v1",
        "summary": {"models_confirmed": 1, "invalid_targets": 0},
    })
    write_json(root, contract.DEFAULT_PATHS["acceptance"], {
        "schema": "fak.api-host-acceptance.v1",
        "summary": {"invalid_targets": 0},
        "targets": [
            {"name": "ok", "status": "READY_FOR_LIVE_BRIDGE_RUN", "readiness_status": "MODELS_CONFIRMED"},
        ],
    })


class APIHostCompatContractTest(unittest.TestCase):
    def test_good_artifacts_pass_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            report = contract.build_report(root)
            self.assertEqual(report["schema"], contract.SCHEMA)
            self.assertTrue(report["summary"]["contract_gate"])
            self.assertEqual(report["summary"]["proven_host_classes"], report["summary"]["host_classes"])
            self.assertTrue(any(item["status"] == "OUT_OF_CONTRACT" for item in report["non_claims"]))
            self.assertTrue(any(item["id"] == "streaming_chat_completions_delta_passthrough" for item in report["non_claims"]))

    def test_missing_artifacts_fail_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = contract.build_report(Path(td))
            self.assertFalse(report["summary"]["contract_gate"])
            self.assertTrue(report["artifact_errors"])

    def test_wrong_artifact_schema_fails_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            data = json.loads((root / contract.DEFAULT_PATHS["readiness"]).read_text(encoding="utf-8"))
            data["schema"] = "fak.api-host-readiness.v0"
            write_json(root, contract.DEFAULT_PATHS["readiness"], data)
            report = contract.build_report(root)
            self.assertFalse(report["summary"]["contract_gate"])
            self.assertIn("readiness_schema", report["artifact_errors"])

    def test_missing_provider_shape_fails_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, contract.DEFAULT_PATHS["matrix"], {
                "schema": "fak.api-host-bridge-matrix.v1",
                "summary": {"provider_shapes_covered": ["openai-compatible"]},
                "witnesses": [
                    {"id": "openai_compatible_gateway", "status": "resolved"},
                    {"id": "native_provider_adapters", "status": "resolved"},
                    {"id": "provider_proxy_end_to_end", "status": "resolved"},
                    {"id": "host_agnostic_openai_compatible", "status": "resolved"},
                    {"id": "openai_compatible_host_profiles", "status": "resolved"},
                    {"id": "direct_http_syscall", "status": "resolved"},
                    {"id": "direct_mcp_syscall", "status": "resolved"},
                ],
            })
            report = contract.build_report(root)
            self.assertFalse(report["summary"]["contract_gate"])
            native = next(item for item in report["host_classes"] if item["id"] == "native_provider_transcript_adapters")
            self.assertEqual(native["status"], "FAILED")

    def test_invalid_readiness_targets_fail_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, contract.DEFAULT_PATHS["readiness"], {
                "schema": "fak.api-host-readiness.v1",
                "summary": {"models_confirmed": 1, "invalid_targets": 1},
            })
            report = contract.build_report(root)
            self.assertFalse(report["summary"]["contract_gate"])
            upstream = next(item for item in report["host_classes"] if item["id"] == "openai_compatible_upstream")
            self.assertFalse(upstream["checks"]["no_invalid_readiness_targets"])

    def test_missing_host_profile_gate_fails_openai_compatible_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            data = json.loads((root / contract.DEFAULT_PATHS["gate"]).read_text(encoding="utf-8"))
            for row in data["runs"]:
                if row["id"] == "openai_compatible_host_profiles":
                    row["status"] = "failed"
            write_json(root, contract.DEFAULT_PATHS["gate"], data)

            report = contract.build_report(root)

            self.assertFalse(report["summary"]["contract_gate"])
            upstream = next(item for item in report["host_classes"] if item["id"] == "openai_compatible_upstream")
            self.assertEqual(upstream["status"], "FAILED")
            self.assertFalse(upstream["checks"]["host_profile_conformance_passed"])

    def test_acceptance_models_can_corroborate_transient_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, contract.DEFAULT_PATHS["readiness"], {
                "schema": "fak.api-host-readiness.v1",
                "summary": {"models_confirmed": 0, "invalid_targets": 0},
            })

            report = contract.build_report(root)

            upstream = next(item for item in report["host_classes"] if item["id"] == "openai_compatible_upstream")
            self.assertEqual(upstream["status"], "PROVEN")
            self.assertTrue(upstream["checks"]["current_models_surface_confirmed"])
            self.assertEqual(upstream["evidence"]["acceptance_models_confirmed"], 1)

    def test_non_object_artifact_fails_contract_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, contract.DEFAULT_PATHS["matrix"], [])
            report = contract.build_report(root)
            self.assertFalse(report["summary"]["contract_gate"])
            self.assertIn("matrix", report["artifact_errors"])
            self.assertIn("not a JSON object", report["artifact_errors"]["matrix"])

    def test_malformed_witness_rows_fail_contract_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, contract.DEFAULT_PATHS["matrix"], {
                "schema": "fak.api-host-bridge-matrix.v1",
                "summary": {"provider_shapes_covered": ["anthropic", "gemini", "openai-compatible", "xai"]},
                "witnesses": [[], {"id": "openai_compatible_gateway", "status": "resolved"}],
            })
            write_json(root, contract.DEFAULT_PATHS["gate"], {
                "schema": "fak.api-host-bridge-gate.v1",
                "runs": [[], {"id": "openai_compatible_gateway", "status": "passed"}],
            })
            report = contract.build_report(root)
            self.assertFalse(report["summary"]["contract_gate"])
            native = next(item for item in report["host_classes"] if item["id"] == "native_provider_transcript_adapters")
            self.assertEqual(native["status"], "FAILED")

    def test_cli_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            json_path = root / "contract.json"
            md_path = root / "contract.md"
            rc = contract.main(["--root", str(root), "--out", str(json_path), "--markdown", str(md_path)])
            self.assertEqual(rc, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertTrue(data["summary"]["contract_gate"])
            self.assertIn("API-Host Compatibility Contract", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
