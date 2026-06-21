#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import api_host_external_state_audit as audit


def write_json(root: Path, rel_path: str, data: object) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_good_artifacts(root: Path) -> None:
    targets = [
        {
            "name": "live",
            "provider": "openai-compatible",
            "contract_class": "openai_compatible_upstream",
            "base_url": "https://live.example/v1",
            "api_key_env": "LIVE_KEY",
            "model_hint": "live-model",
            "status": "SUPPORTED_TEMPLATE",
        },
        {
            "name": "billing",
            "provider": "openai-compatible",
            "contract_class": "openai_compatible_upstream",
            "base_url": "https://billing.example/v1",
            "api_key_env": "BILLING_KEY",
            "model_hint": "billing-model",
            "status": "SUPPORTED_TEMPLATE",
        },
        {
            "name": "auth",
            "provider": "openai-compatible",
            "contract_class": "openai_compatible_upstream",
            "base_url": "https://auth.example/v1",
            "api_key_env": "AUTH_KEY",
            "model_hint": "auth-model",
            "status": "SUPPORTED_TEMPLATE",
        },
        {
            "name": "ready",
            "provider": "openai-compatible",
            "contract_class": "openai_compatible_upstream",
            "base_url": "https://ready.example/v1",
            "api_key_env": "READY_KEY",
            "model_hint": "ready-model",
            "status": "SUPPORTED_TEMPLATE",
        },
        {
            "name": "missing",
            "provider": "openai-compatible",
            "contract_class": "openai_compatible_upstream",
            "base_url": "https://missing.example/v1",
            "api_key_env": "MISSING_KEY",
            "model_hint": "missing-model",
            "status": "SUPPORTED_TEMPLATE",
        },
        {
            "name": "noauth",
            "provider": "openai-compatible",
            "contract_class": "openai_compatible_upstream",
            "base_url": "https://noauth.example/v1",
            "api_key_env": "",
            "model_hint": "noauth-model",
            "status": "SUPPORTED_TEMPLATE",
        },
    ]
    write_json(root, audit.DEFAULT_PATHS["roster"], {
        "schema": "fak.api-host-roster.v1",
        "summary": {
            "targets": len(targets),
            "roster_gate": True,
            "invalid_targets": 0,
            "unsupported_wire": 0,
            "duplicate_names": 0,
        },
        "targets": targets,
    })
    write_json(root, audit.DEFAULT_PATHS["readiness"], {
        "schema": "fak.api-host-readiness.v1",
        "summary": {"readiness_gate": True},
        "probes": [
            {"name": "ready", "base_url": "https://ready.example/v1", "status": "MODELS_CONFIRMED", "http_status": 200, "models": ["ready-model"]},
        ],
    })
    write_json(root, audit.DEFAULT_PATHS["acceptance"], {
        "schema": "fak.api-host-acceptance.v1",
        "summary": {"acceptance_gate": True},
        "targets": [
            {"name": "billing", "base_url": "https://billing.example/v1", "status": "BILLING_REQUIRED", "readiness_status": "BILLING_REQUIRED", "contract_class": "openai_compatible_upstream"},
            {"name": "auth", "base_url": "https://auth.example/v1", "status": "AUTH_REQUIRED", "readiness_status": "AUTH_REQUIRED", "contract_class": "openai_compatible_upstream"},
            {"name": "ready", "base_url": "https://ready.example/v1", "status": "READY_FOR_LIVE_BRIDGE_RUN", "readiness_status": "MODELS_CONFIRMED", "contract_class": "openai_compatible_upstream"},
        ],
    })
    write_json(root, audit.DEFAULT_PATHS["retry"], {
        "schema": "fak.api-host-retry-packet.v1",
        "summary": {"retry_packet_gate": True},
        "actions": [
            {"target": "billing", "base_url": "https://billing.example/v1", "status": "BILLING_REQUIRED", "action_type": "fix_billing_then_smoke", "operator_prerequisite": "Attach billing/payment method for BILLING_KEY.", "commands": ["run billing"]},
            {"target": "auth", "base_url": "https://auth.example/v1", "status": "AUTH_REQUIRED", "action_type": "configure_access_then_probe_and_smoke", "operator_prerequisite": "Configure a valid token via AUTH_KEY.", "commands": ["run auth"]},
            {"target": "ready", "base_url": "https://ready.example/v1", "status": "READY_FOR_LIVE_BRIDGE_RUN", "action_type": "run_live_smoke", "operator_prerequisite": "none", "commands": ["run ready"]},
        ],
    })
    write_json(root, audit.DEFAULT_PATHS["live"], {
        "schema": "fak.api-host-live-inventory.v1",
        "summary": {"live_inventory_gate": True},
        "proofs": [
            {
                "id": "live_proof",
                "status": "LIVE_CONFIRMED",
                "evidence": {"base_url": "https://live.example/v1", "path": "live.json", "model": "live-model"},
            },
        ],
    })


class APIHostExternalStateAuditTest(unittest.TestCase):
    def test_good_artifacts_classify_state_without_secret_values(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.dict(
            os.environ,
            {"BILLING_KEY": "super-secret-token", "READY_KEY": "ready-secret-token"},
            clear=True,
        ):
            root = Path(td)
            write_good_artifacts(root)

            report = audit.build_report(root)

            self.assertEqual(report["schema"], audit.SCHEMA)
            self.assertTrue(report["summary"]["external_state_audit_gate"])
            statuses = {row["name"]: row["external_state_status"] for row in report["targets"]}
            self.assertEqual(statuses["live"], "LIVE_CONFIRMED")
            self.assertEqual(statuses["billing"], "BLOCKED_BILLING")
            self.assertEqual(statuses["auth"], "BLOCKED_AUTH")
            self.assertEqual(statuses["ready"], "READY_FOR_LIVE_BRIDGE_RUN")
            self.assertEqual(statuses["missing"], "NEEDS_CREDENTIAL")
            self.assertEqual(statuses["noauth"], "NO_AUTH_READY_TO_PROBE")
            body = json.dumps(report)
            self.assertNotIn("super-secret-token", body)
            self.assertNotIn("ready-secret-token", body)

    def test_missing_artifact_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = audit.build_report(Path(td))
            self.assertFalse(report["summary"]["external_state_audit_gate"])
            self.assertGreater(report["summary"]["artifact_errors"], 0)

    def test_wrong_artifact_schema_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_good_artifacts(root)
            data = json.loads((root / audit.DEFAULT_PATHS["acceptance"]).read_text(encoding="utf-8"))
            data["schema"] = "fak.api-host-acceptance.v0"
            write_json(root, audit.DEFAULT_PATHS["acceptance"], data)

            report = audit.build_report(root)

            self.assertFalse(report["summary"]["external_state_audit_gate"])
            self.assertEqual(report["summary"]["artifact_errors"], 1)
            self.assertIn("acceptance_schema", report["artifact_errors"])

    def test_env_present_unprobed_template_is_known(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"TARGET_KEY": "value"}, clear=True):
            root = Path(td)
            target = {
                "name": "unprobed",
                "provider": "openai-compatible",
                "contract_class": "openai_compatible_upstream",
                "base_url": "https://unprobed.example/v1",
                "api_key_env": "TARGET_KEY",
                "model_hint": "m",
                "status": "SUPPORTED_TEMPLATE",
            }
            write_json(root, audit.DEFAULT_PATHS["roster"], {
                "schema": "fak.api-host-roster.v1",
                "summary": {"targets": 1, "roster_gate": True},
                "targets": [target],
            })
            write_json(root, audit.DEFAULT_PATHS["readiness"], {"schema": "fak.api-host-readiness.v1", "probes": []})
            write_json(root, audit.DEFAULT_PATHS["acceptance"], {"schema": "fak.api-host-acceptance.v1", "targets": []})
            write_json(root, audit.DEFAULT_PATHS["retry"], {"schema": "fak.api-host-retry-packet.v1", "actions": []})
            write_json(root, audit.DEFAULT_PATHS["live"], {"schema": "fak.api-host-live-inventory.v1", "proofs": []})

            report = audit.build_report(root)

            self.assertTrue(report["summary"]["external_state_audit_gate"])
            self.assertEqual(report["targets"][0]["credential_state"], "ENV_PRESENT")
            self.assertEqual(report["targets"][0]["external_state_status"], "UNPROBED_TEMPLATE")

    def test_roster_gate_failure_fails_audit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_good_artifacts(root)
            roster_path = root / audit.DEFAULT_PATHS["roster"]
            data = json.loads(roster_path.read_text(encoding="utf-8"))
            data["summary"]["roster_gate"] = False
            roster_path.write_text(json.dumps(data), encoding="utf-8")

            report = audit.build_report(root)

            self.assertFalse(report["summary"]["external_state_audit_gate"])
            self.assertIn("roster_gate", report["artifact_errors"])

    def test_cli_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            root = Path(td)
            write_good_artifacts(root)
            json_path = root / "audit.json"
            md_path = root / "audit.md"
            rc = audit.main(["--root", str(root), "--out", str(json_path), "--markdown", str(md_path)])
            self.assertEqual(rc, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], audit.SCHEMA)
            self.assertIn("API-Host External State Audit", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
