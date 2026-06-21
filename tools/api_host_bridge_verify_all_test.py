#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import api_host_bridge_verify_all as verify


def write_json(root: Path, rel_path: str, data: object) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def good_artifacts(root: Path) -> None:
    write_json(root, verify.PROOF_PATH, {
        "schema": "fak.api-host-bridge-proof.v1",
        "summary": {"proof_gate": True, "requirements": 10, "proven": 10, "failed_or_missing": 0},
    })
    write_json(root, verify.CERTIFICATE_PATH, {
        "schema": "fak.api-host-conformance-certificate.v1",
        "summary": {"certificate_gate": True, "capabilities": 7, "proven_capabilities": 7},
    })
    write_json(root, verify.GOAL_PATH, {
        "schema": "fak.api-host-goal-audit.v1",
        "summary": {
            "requirements": 9,
            "proven": 7,
            "residual": 2,
            "incomplete": 0,
            "goal_complete": False,
            "goal_status": "SCOPE_BOUNDED_PROGRESS_NOT_COMPLETE",
        },
        "requirements": [
            {"id": "compatible_api_host_bridge", "status": "PROVEN"},
            {"id": "universal_any_api_host", "status": "NOT_PROVEN"},
            {"id": "paid_or_keyed_live_hosts", "status": "EXTERNAL_STATE"},
        ],
    })
    write_json(root, verify.BENCHMARK_PATH, {
        "schema": "fak.permission-system-benchmark.v1",
        "metrics": [
            {"system": "fak_dos_gateway"},
            {"system": "claude_code_auto"},
        ],
    })
    write_json(root, verify.SOURCE_AUDIT_PATH, {
        "schema": "fak.permission-source-audit.v1",
        "summary": {"sources": 7, "verified": 7, "failed": 0, "source_audit_gate": True},
    })
    write_json(root, verify.EXTERNAL_STATE_PATH, {
        "schema": "fak.api-host-external-state-audit.v1",
        "summary": {
            "external_state_audit_gate": True,
            "roster_targets": 13,
            "artifact_errors": 0,
            "unclassified": 0,
            "invalid_templates": 0,
            "unsupported_templates": 0,
        },
    })
    write_json(root, verify.QUALIFICATION_PATH, {
        "schema": "fak.api-host-qualification.v1",
        "summary": {
            "qualification_gate": True,
            "targets": 13,
            "in_contract_targets": 13,
            "artifact_errors": 0,
            "unclassified": 0,
            "out_of_contract": 0,
            "invalid_targets": 0,
        },
    })
    write_json(root, verify.LIVE_QUEUE_PATH, {
        "schema": "fak.api-host-live-smoke-queue.v1",
        "summary": {
            "live_smoke_queue_gate": True,
            "targets": 13,
            "complete": 1,
            "blocked_external_state": 2,
            "waiting_for_credential": 10,
            "unqualified": 0,
            "unclassified": 0,
            "command_gaps": 0,
            "artifact_errors": 0,
        },
    })
    write_json(root, verify.LIVE_RUNNER_PATH, {
        "schema": "fak.api-host-live-smoke-runner.v1",
        "summary": {
            "live_smoke_runner_gate": True,
            "targets": 13,
            "already_complete": 1,
            "ready_to_execute": 0,
            "ready_not_executed": 0,
            "skipped_external_state": 2,
            "skipped_waiting_for_credential": 10,
            "failed": 0,
            "ready_execution_gaps": 0,
            "unclassified": 0,
            "artifact_errors": 0,
        },
    })


class APIHostBridgeVerifyAllTest(unittest.TestCase):
    def test_artifacts_only_good_scope_bounded_report_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            report = verify.build_report(root, artifacts_only=True)
            self.assertEqual(report["schema"], verify.SCHEMA)
            self.assertTrue(report["summary"]["scope_bounded_verification_gate"])
            self.assertFalse(report["summary"]["goal_complete"])
            self.assertEqual(report["summary"]["residual_requirements"], 2)

    def test_artifact_failure_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, verify.PROOF_PATH, {"summary": {"proof_gate": False}})
            report = verify.build_report(root, artifacts_only=True)
            self.assertFalse(report["summary"]["scope_bounded_verification_gate"])

    def test_wrong_artifact_schema_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            data = json.loads((root / verify.QUALIFICATION_PATH).read_text(encoding="utf-8"))
            data["schema"] = "fak.api-host-qualification.v0"
            write_json(root, verify.QUALIFICATION_PATH, data)
            report = verify.build_report(root, artifacts_only=True)
            self.assertFalse(report["summary"]["scope_bounded_verification_gate"])
            self.assertEqual(report["summary"]["artifact_errors"], 1)
            self.assertIn("qualification_schema", report["artifact_errors"])

    def test_empty_permission_benchmark_metrics_fail_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, verify.BENCHMARK_PATH, {
                "schema": "fak.permission-system-benchmark.v1",
                "metrics": [],
            })
            report = verify.build_report(root, artifacts_only=True)
            self.assertFalse(report["summary"]["scope_bounded_verification_gate"])
            self.assertFalse(report["summary"]["benchmark_gate"])
            self.assertEqual(report["artifact_summaries"]["benchmark"]["metrics"], 0)

    def test_failed_permission_source_audit_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, verify.SOURCE_AUDIT_PATH, {
                "schema": "fak.permission-source-audit.v1",
                "summary": {"sources": 7, "verified": 6, "failed": 1, "source_audit_gate": False},
            })
            report = verify.build_report(root, artifacts_only=True)
            self.assertFalse(report["summary"]["scope_bounded_verification_gate"])
            self.assertFalse(report["summary"]["source_audit_gate"])
            self.assertEqual(report["artifact_summaries"]["source_audit"]["failed"], 1)

    def test_external_state_failure_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, verify.EXTERNAL_STATE_PATH, {
                "schema": "fak.api-host-external-state-audit.v1",
                "summary": {"external_state_audit_gate": False, "artifact_errors": 1},
            })
            report = verify.build_report(root, artifacts_only=True)
            self.assertFalse(report["summary"]["scope_bounded_verification_gate"])
            self.assertFalse(report["summary"]["external_state_gate"])

    def test_qualification_failure_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, verify.QUALIFICATION_PATH, {
                "schema": "fak.api-host-qualification.v1",
                "summary": {"qualification_gate": False, "out_of_contract": 1},
            })
            report = verify.build_report(root, artifacts_only=True)
            self.assertFalse(report["summary"]["scope_bounded_verification_gate"])
            self.assertFalse(report["summary"]["qualification_gate"])

    def test_live_smoke_queue_failure_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, verify.LIVE_QUEUE_PATH, {
                "schema": "fak.api-host-live-smoke-queue.v1",
                "summary": {"live_smoke_queue_gate": False, "command_gaps": 1},
            })
            report = verify.build_report(root, artifacts_only=True)
            self.assertFalse(report["summary"]["scope_bounded_verification_gate"])
            self.assertFalse(report["summary"]["live_smoke_queue_gate"])

    def test_live_smoke_runner_failure_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, verify.LIVE_RUNNER_PATH, {
                "schema": "fak.api-host-live-smoke-runner.v1",
                "summary": {"live_smoke_runner_gate": False, "ready_execution_gaps": 1},
            })
            report = verify.build_report(root, artifacts_only=True)
            self.assertFalse(report["summary"]["scope_bounded_verification_gate"])
            self.assertFalse(report["summary"]["live_smoke_runner_gate"])

    def test_incomplete_goal_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            data = json.loads((root / verify.GOAL_PATH).read_text(encoding="utf-8"))
            data["summary"]["incomplete"] = 1
            data["requirements"].append({"id": "artifact_integrity", "status": "INCOMPLETE"})
            write_json(root, verify.GOAL_PATH, data)
            report = verify.build_report(root, artifacts_only=True)
            self.assertFalse(report["summary"]["scope_bounded_verification_gate"])
            self.assertEqual(len(report["incomplete_requirements"]), 1)

    def test_failed_command_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            artifacts = verify.artifact_summary(root)
            summary = verify.evaluate([
                {"id": "ok", "phase": "test", "status": "passed"},
                {"id": "bad", "phase": "test", "status": "failed"},
            ], artifacts, artifacts_only=False)
            self.assertFalse(summary["scope_bounded_verification_gate"])
            self.assertEqual(summary["failed_steps"], 1)

    def test_generation_steps_probe_from_roster_after_roster_generation(self) -> None:
        steps = verify.generation_steps()
        ids = [step["id"] for step in steps]
        self.assertLess(ids.index("api_host_roster"), ids.index("api_host_readiness"))
        self.assertLess(ids.index("api_host_roster"), ids.index("api_host_acceptance"))
        self.assertLess(ids.index("api_host_qualification"), ids.index("api_host_live_smoke_queue"))
        self.assertLess(ids.index("api_host_live_smoke_queue"), ids.index("api_host_live_smoke_runner"))
        self.assertLess(ids.index("api_host_live_smoke_runner"), ids.index("api_host_bridge_proof"))
        readiness = next(step for step in steps if step["id"] == "api_host_readiness")
        acceptance = next(step for step in steps if step["id"] == "api_host_acceptance")
        self.assertIn("--from-roster", readiness["argv"])
        self.assertIn("--from-roster", acceptance["argv"])

    def test_cli_artifacts_only_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            json_path = root / "verify.json"
            md_path = root / "verify.md"
            rc = verify.main(["--root", str(root), "--artifacts-only", "--out", str(json_path), "--markdown", str(md_path)])
            self.assertEqual(rc, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertTrue(data["summary"]["scope_bounded_verification_gate"])
            self.assertIn("API-Host Bridge Verify All", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
