#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import api_host_qualification as qual


def write_json(root: Path, rel_path: str, data: object) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def good_artifacts(root: Path) -> None:
    write_json(root, qual.DEFAULT_PATHS["certificate"], {
        "schema": "fak.api-host-conformance-certificate.v1",
        "summary": {"certificate_gate": True},
        "capabilities": [
            {"id": "openai_compatible_host_conformance", "status": "PROVEN"},
            {"id": "native_provider_transcript_wires", "status": "PROVEN"},
            {"id": "direct_http_syscall_boundary", "status": "PROVEN"},
            {"id": "direct_mcp_syscall_boundary", "status": "PROVEN"},
        ],
    })
    write_json(root, qual.DEFAULT_PATHS["contract"], {
        "schema": "fak.api-host-compat-contract.v1",
        "summary": {"contract_gate": True},
        "host_classes": [
            {"id": "openai_compatible_upstream", "status": "PROVEN"},
            {"id": "native_provider_transcript_adapters", "status": "PROVEN"},
            {"id": "direct_kernel_http_syscall", "status": "PROVEN"},
            {"id": "direct_kernel_mcp_syscall", "status": "PROVEN"},
        ],
    })
    targets = [
        target("live", "LIVE_CONFIRMED"),
        target("ready", "READY_FOR_LIVE_BRIDGE_RUN"),
        target("billing", "BLOCKED_BILLING"),
        target("missing", "NEEDS_CREDENTIAL"),
        target("probe", "NO_AUTH_READY_TO_PROBE"),
    ]
    write_json(root, qual.DEFAULT_PATHS["external_state"], {
        "schema": "fak.api-host-external-state-audit.v1",
        "summary": {"external_state_audit_gate": True},
        "targets": targets,
    })
    write_json(root, qual.DEFAULT_PATHS["retry"], {
        "schema": "fak.api-host-retry-packet.v1",
        "summary": {"retry_packet_gate": True},
        "actions": [
            {
                "target": "billing",
                "base_url": "https://billing.example/v1",
                "status": "BILLING_REQUIRED",
                "commands": ["run billing"],
            },
            {
                "target": "ready",
                "base_url": "https://ready.example/v1",
                "status": "READY_FOR_LIVE_BRIDGE_RUN",
                "commands": ["run ready"],
            },
        ],
    })


def target(name: str, external_state: str, contract_class: str = "openai_compatible_upstream") -> dict[str, object]:
    return {
        "name": name,
        "provider": "openai-compatible",
        "base_url": f"https://{name}.example/v1",
        "model_hint": f"{name}-model",
        "api_key_env": f"{name.upper()}_KEY",
        "credential_state": "ENV_MISSING",
        "roster_status": "SUPPORTED_TEMPLATE",
        "contract_class": contract_class,
        "external_state_status": external_state,
        "next_evidence_needed": f"next {name}",
    }


class APIHostQualificationTest(unittest.TestCase):
    def test_good_artifacts_qualify_known_in_contract_states(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)

            report = qual.build_report(root)

            self.assertEqual(report["schema"], qual.SCHEMA)
            self.assertTrue(report["summary"]["qualification_gate"])
            self.assertEqual(report["summary"]["in_contract_targets"], report["summary"]["targets"])
            statuses = {row["name"]: row["qualification_status"] for row in report["targets"]}
            self.assertEqual(statuses["live"], "IN_CONTRACT_LIVE_CONFIRMED")
            self.assertEqual(statuses["ready"], "IN_CONTRACT_READY_FOR_LIVE_SMOKE")
            self.assertEqual(statuses["billing"], "IN_CONTRACT_EXTERNAL_BLOCKER")
            self.assertEqual(statuses["missing"], "IN_CONTRACT_NEEDS_CREDENTIAL")
            self.assertEqual(statuses["probe"], "IN_CONTRACT_NEEDS_PROBE")
            billing = next(row for row in report["targets"] if row["name"] == "billing")
            self.assertEqual(billing["commands"], ["run billing"])

    def test_missing_artifacts_fail_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = qual.build_report(Path(td))
            self.assertFalse(report["summary"]["qualification_gate"])
            self.assertGreater(report["summary"]["artifact_errors"], 0)

    def test_wrong_artifact_schema_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            data = json.loads((root / qual.DEFAULT_PATHS["external_state"]).read_text(encoding="utf-8"))
            data["schema"] = "fak.api-host-external-state-audit.v0"
            write_json(root, qual.DEFAULT_PATHS["external_state"], data)

            report = qual.build_report(root)

            self.assertFalse(report["summary"]["qualification_gate"])
            self.assertIn("external_state_schema", report["artifact_errors"])

    def test_unproven_certificate_capability_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            data = json.loads((root / qual.DEFAULT_PATHS["certificate"]).read_text(encoding="utf-8"))
            data["capabilities"][0]["status"] = "FAILED"
            write_json(root, qual.DEFAULT_PATHS["certificate"], data)

            report = qual.build_report(root)

            self.assertFalse(report["summary"]["qualification_gate"])
            live = next(row for row in report["targets"] if row["name"] == "live")
            self.assertEqual(live["qualification_status"], "UNCLASSIFIED")
            self.assertFalse(live["in_contract"])

    def test_out_of_contract_target_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            data = json.loads((root / qual.DEFAULT_PATHS["external_state"]).read_text(encoding="utf-8"))
            data["targets"].append(target("unsupported", "UNSUPPORTED_TEMPLATE", contract_class="unsupported"))
            write_json(root, qual.DEFAULT_PATHS["external_state"], data)

            report = qual.build_report(root)

            self.assertFalse(report["summary"]["qualification_gate"])
            unsupported = next(row for row in report["targets"] if row["name"] == "unsupported")
            self.assertEqual(unsupported["qualification_status"], "OUT_OF_CONTRACT")

    def test_cli_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            json_path = root / "qualification.json"
            md_path = root / "qualification.md"
            rc = qual.main(["--root", str(root), "--out", str(json_path), "--markdown", str(md_path)])
            self.assertEqual(rc, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], qual.SCHEMA)
            self.assertIn("API-Host Qualification", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
