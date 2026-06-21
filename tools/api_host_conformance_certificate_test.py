#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import api_host_conformance_certificate as cert


def write_json(root: Path, rel_path: str, data: object) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def good_artifacts(root: Path) -> None:
    witness_ids = [
        "host_agnostic_openai_compatible",
        "openai_compatible_host_profiles",
        "openai_compatible_gateway",
        "provider_proxy_end_to_end",
        "native_provider_adapters",
        "direct_http_syscall",
        "direct_mcp_syscall",
    ]
    write_json(root, cert.DEFAULT_PATHS["matrix"], {
        "schema": "fak.api-host-bridge-matrix.v1",
        "summary": {
            "bridge_covered": True,
            "provider_shapes_covered": ["anthropic", "gemini", "openai-compatible", "xai"],
        },
        "witnesses": [{"id": item, "status": "resolved"} for item in witness_ids],
    })
    write_json(root, cert.DEFAULT_PATHS["gate"], {
        "schema": "fak.api-host-bridge-gate.v1",
        "summary": {"executable_bridge_covered": True},
        "runs": [{"id": item, "status": "passed"} for item in witness_ids],
    })
    write_json(root, cert.DEFAULT_PATHS["contract"], {
        "schema": "fak.api-host-compat-contract.v1",
        "summary": {"contract_gate": True},
        "host_classes": [
            {"id": "openai_compatible_upstream", "status": "PROVEN"},
            {"id": "native_provider_transcript_adapters", "status": "PROVEN"},
            {"id": "direct_kernel_http_syscall", "status": "PROVEN"},
            {"id": "direct_kernel_mcp_syscall", "status": "PROVEN"},
            {"id": "live_scoped_host_evidence", "status": "PROVEN"},
        ],
        "non_claims": [
            {"id": "arbitrary_api_host_without_compatible_wire", "status": "OUT_OF_CONTRACT"},
            {"id": "streaming_chat_completions_delta_passthrough", "status": "OUT_OF_CONTRACT"},
            {"id": "paid_or_keyed_live_execution_without_credentials", "status": "EXTERNAL_STATE"},
            {"id": "provider_semantics_beyond_tool_wire", "status": "OUT_OF_CONTRACT"},
        ],
        "artifact_errors": {},
    })
    write_json(root, cert.DEFAULT_PATHS["acceptance"], {
        "schema": "fak.api-host-acceptance.v1",
        "summary": {
            "acceptance_gate": True,
            "unsupported_wire": 0,
            "invalid_targets": 0,
            "unclassified": 0,
            "sweep_artifact_errors": 0,
        },
        "targets": [],
        "artifact_errors": [],
    })
    write_json(root, cert.DEFAULT_PATHS["roster"], {
        "schema": "fak.api-host-roster.v1",
        "summary": {
            "targets": 13,
            "openai_compatible_templates": 13,
            "invalid_targets": 0,
            "unsupported_wire": 0,
            "duplicate_names": 0,
            "roster_gate": True,
        },
        "targets": [],
    })
    write_json(root, cert.DEFAULT_PATHS["external_state"], {
        "schema": "fak.api-host-external-state-audit.v1",
        "summary": {
            "roster_targets": 13,
            "artifact_errors": 0,
            "unclassified": 0,
            "invalid_templates": 0,
            "unsupported_templates": 0,
            "external_state_audit_gate": True,
        },
        "targets": [],
        "artifact_errors": {},
    })


class APIHostConformanceCertificateTest(unittest.TestCase):
    def test_good_artifacts_emit_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            report = cert.build_report(root)
            self.assertEqual(report["schema"], cert.SCHEMA)
            self.assertTrue(report["summary"]["certificate_gate"])
            self.assertEqual(report["summary"]["proven_capabilities"], report["summary"]["capabilities"])
            self.assertEqual(report["summary"]["missing_required_non_claims"], 0)
            self.assertTrue(any(item["id"] == "streaming_chat_completions_delta_passthrough" for item in report["non_claims"]))

    def test_failed_host_conformance_fails_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            data = json.loads((root / cert.DEFAULT_PATHS["gate"]).read_text(encoding="utf-8"))
            for row in data["runs"]:
                if row["id"] == "host_agnostic_openai_compatible":
                    row["status"] = "failed"
            write_json(root, cert.DEFAULT_PATHS["gate"], data)

            report = cert.build_report(root)

            self.assertFalse(report["summary"]["certificate_gate"])
            item = next(row for row in report["capabilities"] if row["id"] == "openai_compatible_host_conformance")
            self.assertEqual(item["status"], "FAILED")

    def test_failed_host_profile_conformance_fails_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            data = json.loads((root / cert.DEFAULT_PATHS["gate"]).read_text(encoding="utf-8"))
            for row in data["runs"]:
                if row["id"] == "openai_compatible_host_profiles":
                    row["status"] = "failed"
            write_json(root, cert.DEFAULT_PATHS["gate"], data)

            report = cert.build_report(root)

            self.assertFalse(report["summary"]["certificate_gate"])
            item = next(row for row in report["capabilities"] if row["id"] == "openai_compatible_host_profile_corpus")
            self.assertEqual(item["status"], "FAILED")

    def test_missing_non_claim_fails_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            data = json.loads((root / cert.DEFAULT_PATHS["contract"]).read_text(encoding="utf-8"))
            data["non_claims"] = data["non_claims"][:1]
            write_json(root, cert.DEFAULT_PATHS["contract"], data)

            report = cert.build_report(root)

            self.assertFalse(report["summary"]["certificate_gate"])
            self.assertGreater(report["summary"]["missing_required_non_claims"], 0)

    def test_wrong_artifact_schema_fails_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            data = json.loads((root / cert.DEFAULT_PATHS["contract"]).read_text(encoding="utf-8"))
            data["schema"] = "fak.api-host-compat-contract.v0"
            write_json(root, cert.DEFAULT_PATHS["contract"], data)

            report = cert.build_report(root)

            self.assertFalse(report["summary"]["certificate_gate"])
            self.assertEqual(report["summary"]["artifact_errors"], 1)
            self.assertIn("contract_schema", report["artifact_errors"])

    def test_acceptance_drift_fails_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            data = json.loads((root / cert.DEFAULT_PATHS["acceptance"]).read_text(encoding="utf-8"))
            data["summary"]["unclassified"] = 1
            write_json(root, cert.DEFAULT_PATHS["acceptance"], data)

            report = cert.build_report(root)

            self.assertFalse(report["summary"]["certificate_gate"])
            item = next(row for row in report["capabilities"] if row["id"] == "candidate_host_acceptance")
            self.assertEqual(item["status"], "FAILED")

    def test_roster_drift_fails_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            data = json.loads((root / cert.DEFAULT_PATHS["roster"]).read_text(encoding="utf-8"))
            data["summary"]["unsupported_wire"] = 1
            data["summary"]["roster_gate"] = False
            write_json(root, cert.DEFAULT_PATHS["roster"], data)

            report = cert.build_report(root)

            self.assertFalse(report["summary"]["certificate_gate"])
            item = next(row for row in report["capabilities"] if row["id"] == "expanded_candidate_host_roster")
            self.assertEqual(item["status"], "FAILED")

    def test_external_state_drift_fails_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            data = json.loads((root / cert.DEFAULT_PATHS["external_state"]).read_text(encoding="utf-8"))
            data["summary"]["unclassified"] = 1
            data["summary"]["external_state_audit_gate"] = False
            write_json(root, cert.DEFAULT_PATHS["external_state"], data)

            report = cert.build_report(root)

            self.assertFalse(report["summary"]["certificate_gate"])
            item = next(row for row in report["capabilities"] if row["id"] == "external_state_residual_audit")
            self.assertEqual(item["status"], "FAILED")

    def test_cli_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            json_path = root / "cert.json"
            md_path = root / "cert.md"
            rc = cert.main(["--root", str(root), "--out", str(json_path), "--markdown", str(md_path)])
            self.assertEqual(rc, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], cert.SCHEMA)
            self.assertIn("API-Host Conformance Certificate", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
