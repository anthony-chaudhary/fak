#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import api_host_goal_audit as audit


def write_json(root: Path, rel_path: str, data: object) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def good_artifacts(root: Path) -> None:
    host_command = "go test ./internal/gateway -run 'TestChatProxyOpenAICompatibleAliasIsHostAgnostic$'"
    host_argv = ["go", "test", "./internal/gateway", "-run", "TestChatProxyOpenAICompatibleAliasIsHostAgnostic$", "-count=1"]
    profile_command = "go test ./internal/gateway -run 'TestChatProxyOpenAICompatibleHostProfileConformance$'"
    profile_argv = ["go", "test", "./internal/gateway", "-run", "TestChatProxyOpenAICompatibleHostProfileConformance$", "-count=1"]
    write_json(root, audit.DEFAULT_PATHS["matrix"], {
        "schema": "fak.api-host-bridge-matrix.v1",
        "witnesses": [
            {
                "id": "host_agnostic_openai_compatible",
                "status": "resolved",
                "required": True,
                "command": host_command,
                "cwd": "fak",
                "argv": host_argv,
                "providers": ["openai-compatible"],
                "evidence": [{"status": "resolved"}],
            },
            {
                "id": "openai_compatible_host_profiles",
                "status": "resolved",
                "required": True,
                "command": profile_command,
                "cwd": "fak",
                "argv": profile_argv,
                "providers": ["openai-compatible"],
                "evidence": [{"status": "resolved"}],
            },
        ],
    })
    write_json(root, audit.DEFAULT_PATHS["gate"], {
        "schema": "fak.api-host-bridge-gate.v1",
        "matrix_schema": "fak.api-host-bridge-matrix.v1",
        "runs": [
            {
                "id": "host_agnostic_openai_compatible",
                "status": "passed",
                "required": True,
                "command": host_command,
                "cwd": "fak",
                "argv": host_argv,
                "elapsed_ms": 12,
            },
            {
                "id": "openai_compatible_host_profiles",
                "status": "passed",
                "required": True,
                "command": profile_command,
                "cwd": "fak",
                "argv": profile_argv,
                "elapsed_ms": 12,
            },
        ],
    })
    write_json(root, audit.DEFAULT_PATHS["proof"], {
        "schema": "fak.api-host-bridge-proof.v1",
        "summary": {"proof_gate": True},
        "requirements": [
            {"id": "source_witness_matrix", "status": "PROVEN"},
            {"id": "executed_witness_gate", "status": "PROVEN"},
            {"id": "host_agnostic_conformance", "status": "PROVEN"},
            {"id": "host_profile_conformance", "status": "PROVEN", "detail": {"profile": "ok"}},
            {"id": "committed_live_inventory", "status": "PROVEN"},
            {"id": "current_host_readiness", "status": "PROVEN"},
            {"id": "candidate_host_acceptance", "status": "PROVEN"},
            {"id": "api_host_roster", "status": "PROVEN"},
            {"id": "api_host_external_state_audit", "status": "PROVEN"},
            {"id": "api_host_qualification", "status": "PROVEN"},
            {"id": "api_host_live_smoke_queue", "status": "PROVEN"},
            {"id": "api_host_live_smoke_runner", "status": "PROVEN"},
            {"id": "compatibility_contract", "status": "PROVEN"},
            {"id": "api_host_conformance_certificate", "status": "PROVEN"},
            {"id": "permission_system_benchmark", "status": "PROVEN"},
            {"id": "permission_source_audit", "status": "PROVEN"},
        ],
        "residual_scope": [
            {"id": "universal_any_api_host", "status": "NOT_PROVEN", "reason": "not universal"},
            {"id": "blocked_paid_or_keyed_hosts", "status": "EXTERNAL_STATE", "reason": "billing/key"},
        ],
    })
    write_json(root, audit.DEFAULT_PATHS["benchmark"], {
        "schema": "fak.permission-system-benchmark.v1",
        "metrics": [
            {
                "system": "fak_dos_gateway",
                "risk_scenarios": 6,
                "deterministic_controls": 6,
                "api_host_bridge_dimensions": 6,
                "api_host_bridge_controls": 6,
                "api_host_result_quarantine_verdict": "QUARANTINE",
            },
            {
                "system": "claude_code_auto",
                "known_max_false_negative_pct": 17.0,
                "api_host_bridge_controls": 0,
            },
            {"system": "codex_workspace_sandbox"},
            {"system": "github_copilot_cloud_agent"},
            {"system": "manual_prompts"},
            {"system": "bypass_permissions"},
        ],
    })
    write_json(root, audit.DEFAULT_PATHS["acceptance"], {
        "schema": "fak.api-host-acceptance.v1",
        "summary": {
            "unclassified": 0,
            "unsupported_wire": 0,
            "invalid_targets": 0,
            "sweep_artifact_errors": 0,
            "typed_external_blockers": 3,
            "acceptance_gate": True,
        },
        "artifact_errors": [],
    })
    write_json(root, audit.DEFAULT_PATHS["roster"], {
        "schema": "fak.api-host-roster.v1",
        "summary": {
            "targets": 13,
            "openai_compatible_templates": 13,
            "invalid_targets": 0,
            "unsupported_wire": 0,
            "duplicate_names": 0,
            "roster_gate": True,
        },
    })
    write_json(root, audit.DEFAULT_PATHS["retry"], {
        "schema": "fak.api-host-retry-packet.v1",
        "summary": {
            "targets": 3,
            "actionable_blockers": 3,
            "unsupported_wire": 0,
            "invalid_targets": 0,
            "shape_mismatch": 0,
            "unclassified": 0,
            "action_gaps": 0,
            "artifact_errors": 0,
            "retry_packet_gate": True,
        },
    })
    write_json(root, audit.DEFAULT_PATHS["external_state"], {
        "schema": "fak.api-host-external-state-audit.v1",
        "summary": {
            "roster_targets": 13,
            "env_present": 1,
            "env_missing": 11,
            "no_auth_declared": 1,
            "live_confirmed": 1,
            "blocked_auth": 1,
            "blocked_billing": 1,
            "artifact_errors": 0,
            "unclassified": 0,
            "invalid_templates": 0,
            "unsupported_templates": 0,
            "external_state_audit_gate": True,
        },
        "targets": [],
        "artifact_errors": {},
    })
    write_json(root, audit.DEFAULT_PATHS["certificate"], {
        "schema": "fak.api-host-conformance-certificate.v1",
        "summary": {
            "capabilities": 7,
            "proven_capabilities": 7,
            "failed_capabilities": 0,
            "certificate_gate": True,
        },
    })
    write_json(root, audit.DEFAULT_PATHS["qualification"], {
        "schema": "fak.api-host-qualification.v1",
        "summary": {
            "targets": 13,
            "in_contract_targets": 13,
            "live_confirmed": 1,
            "external_blocked": 2,
            "needs_credential": 10,
            "artifact_errors": 0,
            "unclassified": 0,
            "out_of_contract": 0,
            "invalid_targets": 0,
            "qualification_gate": True,
        },
        "targets": [],
    })
    write_json(root, audit.DEFAULT_PATHS["live_queue"], {
        "schema": "fak.api-host-live-smoke-queue.v1",
        "summary": {
            "targets": 13,
            "complete": 1,
            "ready_to_execute": 0,
            "blocked_external_state": 2,
            "waiting_for_credential": 10,
            "ready_for_probe": 0,
            "unqualified": 0,
            "unclassified": 0,
            "command_gaps": 0,
            "commands": 37,
            "artifact_errors": 0,
            "live_smoke_queue_gate": True,
        },
        "queue": [],
    })
    write_json(root, audit.DEFAULT_PATHS["live_runner"], {
        "schema": "fak.api-host-live-smoke-runner.v1",
        "summary": {
            "targets": 13,
            "execute_ready": False,
            "already_complete": 1,
            "ready_to_execute": 0,
            "ready_not_executed": 0,
            "executed": 0,
            "passed": 0,
            "failed": 0,
            "skipped_external_state": 2,
            "skipped_waiting_for_credential": 10,
            "skipped_ready_for_probe": 0,
            "command_gaps": 0,
            "unclassified": 0,
            "artifact_errors": 0,
            "ready_execution_gaps": 0,
            "live_smoke_runner_gate": True,
        },
        "runs": [],
    })


class APIHostGoalAuditTest(unittest.TestCase):
    def test_good_artifacts_are_scope_bounded_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            report = audit.build_report(root)
            self.assertEqual(report["schema"], audit.SCHEMA)
            self.assertFalse(report["summary"]["goal_complete"])
            self.assertEqual(report["summary"]["goal_status"], "SCOPE_BOUNDED_PROGRESS_NOT_COMPLETE")
            proven = {row["id"] for row in report["requirements"] if row["status"] == "PROVEN"}
            self.assertIn("compatible_api_host_bridge", proven)
            self.assertIn("permission_system_benchmark", proven)
            self.assertIn("host_agnostic_compatible_api_host", proven)
            self.assertIn("openai_compatible_host_profile_corpus", proven)
            self.assertIn("api_host_conformance_certificate", proven)
            self.assertIn("expanded_candidate_host_roster", proven)
            self.assertIn("blocked_host_retry_packet", proven)
            self.assertIn("paid_keyed_external_state_audit", proven)
            self.assertIn("api_host_qualification_predicate", proven)
            self.assertIn("paid_keyed_live_execution_queue", proven)
            self.assertIn("paid_keyed_live_runner_gate", proven)
            residual = {row["id"]: row["status"] for row in report["requirements"]}
            self.assertEqual(residual["universal_any_api_host"], "NOT_PROVEN")
            self.assertEqual(residual["paid_or_keyed_live_hosts"], "EXTERNAL_STATE")

    def test_missing_artifacts_are_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = audit.build_report(Path(td))
            self.assertFalse(report["summary"]["goal_complete"])
            self.assertGreater(report["summary"]["incomplete"], 0)
            integrity = next(row for row in report["requirements"] if row["id"] == "artifact_integrity")
            self.assertEqual(integrity["status"], "INCOMPLETE")

    def test_weak_benchmark_marks_requirement_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, audit.DEFAULT_PATHS["benchmark"], {"metrics": []})
            report = audit.build_report(root)
            bench = next(row for row in report["requirements"] if row["id"] == "permission_system_benchmark")
            self.assertEqual(bench["status"], "INCOMPLETE")

    def test_retry_packet_mismatch_marks_requirement_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, audit.DEFAULT_PATHS["retry"], {
                "schema": "fak.api-host-retry-packet.v1",
                "summary": {
                    "actionable_blockers": 2,
                    "unsupported_wire": 0,
                    "invalid_targets": 0,
                    "shape_mismatch": 0,
                    "unclassified": 0,
                    "action_gaps": 0,
                    "artifact_errors": 0,
                    "retry_packet_gate": True,
                },
            })
            report = audit.build_report(root)
            retry_row = next(row for row in report["requirements"] if row["id"] == "blocked_host_retry_packet")
            self.assertEqual(retry_row["status"], "INCOMPLETE")

    def test_host_agnostic_witness_must_execute(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, audit.DEFAULT_PATHS["gate"], {
                "runs": [
                    {"id": "host_agnostic_openai_compatible", "status": "failed", "elapsed_ms": 12},
                ],
            })
            report = audit.build_report(root)
            host_row = next(row for row in report["requirements"] if row["id"] == "host_agnostic_compatible_api_host")
            self.assertEqual(host_row["status"], "INCOMPLETE")

    def test_host_agnostic_gate_command_must_match_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, audit.DEFAULT_PATHS["gate"], {
                "schema": "fak.api-host-bridge-gate.v1",
                "matrix_schema": "fak.api-host-bridge-matrix.v1",
                "runs": [
                    {
                        "id": "host_agnostic_openai_compatible",
                        "status": "passed",
                        "required": True,
                        "command": "go test ./internal/gateway -run 'TestDifferentWitness$'",
                        "cwd": "fak",
                        "argv": ["go", "test", "./internal/gateway", "-run", "TestDifferentWitness$", "-count=1"],
                        "elapsed_ms": 12,
                    },
                ],
            })
            report = audit.build_report(root)
            host_row = next(row for row in report["requirements"] if row["id"] == "host_agnostic_compatible_api_host")
            self.assertEqual(host_row["status"], "INCOMPLETE")
            self.assertFalse(host_row["detail"]["command_matches"])

    def test_certificate_mismatch_marks_requirement_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, audit.DEFAULT_PATHS["certificate"], {
                "schema": "fak.api-host-conformance-certificate.v1",
                "summary": {
                    "capabilities": 7,
                    "proven_capabilities": 6,
                    "failed_capabilities": 1,
                    "certificate_gate": False,
                },
            })
            report = audit.build_report(root)
            cert_row = next(row for row in report["requirements"] if row["id"] == "api_host_conformance_certificate")
            self.assertEqual(cert_row["status"], "INCOMPLETE")

    def test_roster_mismatch_marks_requirement_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, audit.DEFAULT_PATHS["roster"], {
                "schema": "fak.api-host-roster.v1",
                "summary": {
                    "targets": 13,
                    "openai_compatible_templates": 13,
                    "invalid_targets": 0,
                    "unsupported_wire": 1,
                    "duplicate_names": 0,
                    "roster_gate": False,
                },
            })
            report = audit.build_report(root)
            roster_row = next(row for row in report["requirements"] if row["id"] == "expanded_candidate_host_roster")
            self.assertEqual(roster_row["status"], "INCOMPLETE")

    def test_acceptance_integrity_failure_marks_requirement_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, audit.DEFAULT_PATHS["acceptance"], {
                "summary": {
                    "unclassified": 0,
                    "unsupported_wire": 0,
                    "invalid_targets": 0,
                    "sweep_artifact_errors": 1,
                    "typed_external_blockers": 3,
                    "acceptance_gate": True,
                },
                "artifact_errors": [{"path": "bad.json", "error": "invalid JSON"}],
            })
            report = audit.build_report(root)
            workflow = next(row for row in report["requirements"] if row["id"] == "candidate_host_workflow")
            self.assertEqual(workflow["status"], "INCOMPLETE")

    def test_acceptance_missing_artifact_errors_marks_requirement_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, audit.DEFAULT_PATHS["acceptance"], {
                "schema": "fak.api-host-acceptance.v1",
                "summary": {
                    "unclassified": 0,
                    "unsupported_wire": 0,
                    "invalid_targets": 0,
                    "sweep_artifact_errors": 0,
                    "typed_external_blockers": 3,
                    "acceptance_gate": True,
                },
            })
            report = audit.build_report(root)
            workflow = next(row for row in report["requirements"] if row["id"] == "candidate_host_workflow")
            self.assertEqual(workflow["status"], "INCOMPLETE")

    def test_wrong_artifact_schema_marks_integrity_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, audit.DEFAULT_PATHS["acceptance"], {
                "schema": "fak.api-host-acceptance.v0",
                "summary": {
                    "unclassified": 0,
                    "unsupported_wire": 0,
                    "invalid_targets": 0,
                    "sweep_artifact_errors": 0,
                    "typed_external_blockers": 3,
                    "acceptance_gate": True,
                },
                "artifact_errors": [],
            })
            report = audit.build_report(root)
            integrity = next(row for row in report["requirements"] if row["id"] == "artifact_integrity")
            workflow = next(row for row in report["requirements"] if row["id"] == "candidate_host_workflow")
            self.assertEqual(integrity["status"], "INCOMPLETE")
            self.assertIn("acceptance_schema", integrity["detail"])
            self.assertEqual(workflow["status"], "INCOMPLETE")

    def test_retry_integrity_failure_marks_requirement_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, audit.DEFAULT_PATHS["retry"], {
                "schema": "fak.api-host-retry-packet.v1",
                "summary": {
                    "actionable_blockers": 3,
                    "unsupported_wire": 0,
                    "invalid_targets": 1,
                    "shape_mismatch": 0,
                    "unclassified": 0,
                    "action_gaps": 0,
                    "artifact_errors": 0,
                    "retry_packet_gate": True,
                },
            })
            report = audit.build_report(root)
            retry_row = next(row for row in report["requirements"] if row["id"] == "blocked_host_retry_packet")
            self.assertEqual(retry_row["status"], "INCOMPLETE")

    def test_external_state_drift_marks_requirement_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, audit.DEFAULT_PATHS["external_state"], {
                "schema": "fak.api-host-external-state-audit.v1",
                "summary": {
                    "roster_targets": 13,
                    "artifact_errors": 0,
                    "unclassified": 1,
                    "invalid_templates": 0,
                    "unsupported_templates": 0,
                    "external_state_audit_gate": False,
                },
            })
            report = audit.build_report(root)
            external = next(row for row in report["requirements"] if row["id"] == "paid_keyed_external_state_audit")
            self.assertEqual(external["status"], "INCOMPLETE")

    def test_qualification_drift_marks_requirement_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, audit.DEFAULT_PATHS["qualification"], {
                "schema": "fak.api-host-qualification.v1",
                "summary": {
                    "targets": 13,
                    "in_contract_targets": 12,
                    "artifact_errors": 0,
                    "unclassified": 0,
                    "out_of_contract": 1,
                    "invalid_targets": 0,
                    "qualification_gate": False,
                },
            })
            report = audit.build_report(root)
            qualification = next(row for row in report["requirements"] if row["id"] == "api_host_qualification_predicate")
            self.assertEqual(qualification["status"], "INCOMPLETE")

    def test_live_smoke_queue_drift_marks_requirement_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, audit.DEFAULT_PATHS["live_queue"], {
                "schema": "fak.api-host-live-smoke-queue.v1",
                "summary": {
                    "targets": 13,
                    "complete": 1,
                    "ready_to_execute": 1,
                    "blocked_external_state": 1,
                    "waiting_for_credential": 10,
                    "ready_for_probe": 0,
                    "unqualified": 0,
                    "unclassified": 0,
                    "command_gaps": 1,
                    "artifact_errors": 0,
                    "live_smoke_queue_gate": False,
                },
                "queue": [],
            })
            report = audit.build_report(root)
            live_queue = next(row for row in report["requirements"] if row["id"] == "paid_keyed_live_execution_queue")
            self.assertEqual(live_queue["status"], "INCOMPLETE")

    def test_live_smoke_runner_drift_marks_requirement_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, audit.DEFAULT_PATHS["live_runner"], {
                "schema": "fak.api-host-live-smoke-runner.v1",
                "summary": {
                    "targets": 13,
                    "ready_to_execute": 1,
                    "ready_not_executed": 1,
                    "failed": 0,
                    "ready_execution_gaps": 1,
                    "artifact_errors": 0,
                    "unclassified": 0,
                    "live_smoke_runner_gate": False,
                },
                "runs": [],
            })
            report = audit.build_report(root)
            live_runner = next(row for row in report["requirements"] if row["id"] == "paid_keyed_live_runner_gate")
            self.assertEqual(live_runner["status"], "INCOMPLETE")

    def test_cli_writes_reports_and_exits_zero_for_scope_bounded_progress(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            json_path = root / "goal.json"
            md_path = root / "goal.md"
            rc = audit.main(["--root", str(root), "--out", str(json_path), "--markdown", str(md_path)])
            self.assertEqual(rc, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertFalse(data["summary"]["goal_complete"])
            self.assertIn("API-Host Goal Audit", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
