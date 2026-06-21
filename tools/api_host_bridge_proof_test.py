#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import api_host_bridge_proof as proof


def write_json(root: Path, rel_path: str, data: object) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def good_artifacts(root: Path) -> None:
    host_command = "go test ./internal/gateway -run 'TestChatProxyOpenAICompatibleAliasIsHostAgnostic$'"
    host_argv = ["go", "test", "./internal/gateway", "-run", "TestChatProxyOpenAICompatibleAliasIsHostAgnostic$", "-count=1"]
    profile_command = "go test ./internal/gateway -run 'TestChatProxyOpenAICompatibleHostProfileConformance$'"
    profile_argv = ["go", "test", "./internal/gateway", "-run", "TestChatProxyOpenAICompatibleHostProfileConformance$", "-count=1"]
    write_json(root, proof.DEFAULT_PATHS["matrix"], {
        "schema": "fak.api-host-bridge-matrix.v1",
        "summary": {"bridge_covered": True, "required_witnesses": 9, "resolved_required_witnesses": 9},
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
    write_json(root, proof.DEFAULT_PATHS["gate"], {
        "schema": "fak.api-host-bridge-gate.v1",
        "matrix_schema": "fak.api-host-bridge-matrix.v1",
        "summary": {"executable_bridge_covered": True, "required_witnesses": 9, "passed_required_witnesses": 9},
        "runs": [
            {
                "id": "host_agnostic_openai_compatible",
                "status": "passed",
                "required": True,
                "command": host_command,
                "cwd": "fak",
                "argv": host_argv,
            },
            {
                "id": "openai_compatible_host_profiles",
                "status": "passed",
                "required": True,
                "command": profile_command,
                "cwd": "fak",
                "argv": profile_argv,
            },
        ],
    })
    write_json(root, proof.DEFAULT_PATHS["live"], {
        "schema": "fak.api-host-live-inventory.v1",
        "summary": {
            "live_inventory_gate": True,
            "live_frontier_successes": 2,
            "local_openai_compatible_successes": 1,
            "incomplete_or_unclassified": 0,
        },
    })
    write_json(root, proof.DEFAULT_PATHS["readiness"], {
        "schema": "fak.api-host-readiness.v1",
        "summary": {"targets": 13, "readiness_gate": True, "models_confirmed": 1, "unclassified": 0, "invalid_targets": 0},
    })
    write_json(root, proof.DEFAULT_PATHS["acceptance"], {
        "schema": "fak.api-host-acceptance.v1",
            "summary": {
                "targets": 13,
                "known_statuses": 13,
                "ready_for_live_bridge_run": 1,
                "live_bridge_confirmed": 0,
                "wire_supported_unprobed": 0,
                "typed_external_blockers": 12,
                "unsupported_wire": 0,
                "unclassified": 0,
                "sweep_artifact_errors": 0,
                "invalid_targets": 0,
                "acceptance_gate": True,
            },
        "targets": [
            {"name": "ready", "provider": "openai-compatible", "contract_class": "openai_compatible_upstream", "status": "READY_FOR_LIVE_BRIDGE_RUN"},
            {"name": "missing-key", "provider": "openai-compatible", "contract_class": "openai_compatible_upstream", "status": "NEEDS_AUTH_ENV"},
            {"name": "denied", "provider": "openai-compatible", "contract_class": "openai_compatible_upstream", "status": "ACCESS_DENIED"},
            *[
                {"name": f"missing-key-{i}", "provider": "openai-compatible", "contract_class": "openai_compatible_upstream", "status": "NEEDS_AUTH_ENV"}
                for i in range(10)
            ],
        ],
        "artifact_errors": [],
    })
    write_json(root, proof.DEFAULT_PATHS["roster"], {
        "schema": "fak.api-host-roster.v1",
        "summary": {
            "targets": 13,
            "openai_compatible_templates": 13,
            "invalid_targets": 0,
            "unsupported_wire": 0,
            "duplicate_names": 0,
            "roster_gate": True,
        },
        "targets": [
            {"name": f"target_{i}", "status": "SUPPORTED_TEMPLATE"}
            for i in range(13)
        ],
    })
    write_json(root, proof.DEFAULT_PATHS["external_state"], {
        "schema": "fak.api-host-external-state-audit.v1",
        "summary": {
            "roster_targets": 13,
            "env_present": 2,
            "env_missing": 10,
            "no_auth_declared": 1,
            "live_confirmed": 1,
            "ready_for_live_run": 0,
            "blocked_auth": 1,
            "blocked_billing": 1,
            "needs_credential": 10,
            "unprobed_templates": 0,
            "artifact_errors": 0,
            "unclassified": 0,
            "invalid_templates": 0,
            "unsupported_templates": 0,
            "external_state_audit_gate": True,
        },
        "targets": [],
        "artifact_errors": {},
    })
    write_json(root, proof.DEFAULT_PATHS["contract"], {
        "schema": "fak.api-host-compat-contract.v1",
        "summary": {"contract_gate": True, "host_classes": 5, "proven_host_classes": 5, "failed_host_classes": 0},
        "host_classes": [
            {"id": "openai_compatible_upstream", "status": "PROVEN"},
            {"id": "native_provider_transcript_adapters", "status": "PROVEN"},
            {"id": "direct_kernel_http_syscall", "status": "PROVEN"},
            {"id": "direct_kernel_mcp_syscall", "status": "PROVEN"},
            {"id": "live_scoped_host_evidence", "status": "PROVEN"},
        ],
        "non_claims": [
            {"id": "arbitrary_api_host_without_compatible_wire", "status": "OUT_OF_CONTRACT"},
            {"id": "paid_or_keyed_live_execution_without_credentials", "status": "EXTERNAL_STATE"},
        ],
        "artifact_errors": {},
    })
    write_json(root, proof.DEFAULT_PATHS["certificate"], {
        "schema": "fak.api-host-conformance-certificate.v1",
        "summary": {
            "capabilities": 2,
            "proven_capabilities": 2,
            "failed_capabilities": 0,
            "missing_required_non_claims": 0,
            "artifact_errors": 0,
            "certificate_gate": True,
        },
        "capabilities": [
            {"id": "openai_compatible_host_conformance", "status": "PROVEN"},
            {"id": "candidate_host_acceptance", "status": "PROVEN"},
        ],
        "qualification_rules": [],
        "missing_required_non_claims": [],
    })
    write_json(root, proof.DEFAULT_PATHS["qualification"], {
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
        "artifact_errors": {},
    })
    write_json(root, proof.DEFAULT_PATHS["live_queue"], {
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
        "artifact_errors": {},
    })
    write_json(root, proof.DEFAULT_PATHS["live_runner"], {
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
        "artifact_errors": {},
    })
    write_json(root, proof.DEFAULT_PATHS["benchmark"], {
        "schema": "fak.permission-system-benchmark.v1",
        "api_host_bridge_dimensions": [
            {"id": "host_agnostic_openai_compatible_proxy"},
            {"id": "synthetic_host_profile_conformance"},
            {"id": "pre_execution_tool_call_admission"},
            {"id": "pre_send_tool_result_quarantine"},
            {"id": "roster_driven_host_qualification"},
            {"id": "dos_style_executable_bridge_proof"},
        ],
        "metrics": [
            {
                "system": "fak_dos_gateway",
                "risk_scenarios": 6,
                "deterministic_controls": 6,
                "result_admission_verdict": "QUARANTINE",
                "has_api_host_bridge": True,
                "api_host_bridge_dimensions": 6,
                "api_host_bridge_controls": 6,
                "api_host_result_quarantine_verdict": "QUARANTINE",
            },
            {
                "system": "claude_code_auto",
                "deterministic_controls": 0,
                "known_max_false_negative_pct": 17.0,
                "result_admission_verdict": "WARNING",
                "api_host_bridge_controls": 0,
                "api_host_result_quarantine_verdict": "WARNING",
            },
            {
                "system": "bypass_permissions",
                "risk_scenarios": 6,
                "unguarded_risk_allows": 6,
                "api_host_bridge_controls": 0,
            },
        ],
    })
    write_json(root, proof.DEFAULT_PATHS["source_audit"], {
        "schema": "fak.permission-source-audit.v1",
        "summary": {"sources": 7, "verified": 7, "failed": 0, "source_audit_gate": True},
        "sources": [],
    })


class APIHostBridgeProofTest(unittest.TestCase):
    def test_good_artifacts_pass_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            report = proof.build_report(root)
            self.assertEqual(report["schema"], proof.SCHEMA)
            self.assertTrue(report["summary"]["proof_gate"])
            self.assertEqual(report["summary"]["proven"], report["summary"]["requirements"])
            self.assertEqual(report["summary"]["completion_scope"], "BRIDGE_PROVEN_SCOPE_BOUNDED")
            self.assertTrue(any(item["status"] == "NOT_PROVEN" for item in report["residual_scope"]))

    def test_missing_artifact_fails_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = proof.build_report(Path(td))
            self.assertFalse(report["summary"]["proof_gate"])
            self.assertGreater(report["summary"]["failed_or_missing"], 0)
            self.assertTrue(any(req["status"] == "MISSING" for req in report["requirements"]))

    def test_weak_benchmark_fails_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["benchmark"], {"schema": "fak.permission-system-benchmark.v1", "metrics": []})
            report = proof.build_report(root)
            self.assertFalse(report["summary"]["proof_gate"])
            bench = next(req for req in report["requirements"] if req["id"] == "permission_system_benchmark")
            self.assertEqual(bench["status"], "FAILED")

    def test_wrong_artifact_schema_fails_rollup_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            data = json.loads((root / proof.DEFAULT_PATHS["readiness"]).read_text(encoding="utf-8"))
            data["schema"] = "fak.api-host-readiness.v0"
            write_json(root, proof.DEFAULT_PATHS["readiness"], data)

            report = proof.build_report(root)

            self.assertFalse(report["summary"]["proof_gate"])
            self.assertIn("readiness_schema", report["artifact_errors"])
            integrity = next(req for req in report["requirements"] if req["id"] == "artifact_integrity")
            readiness = next(req for req in report["requirements"] if req["id"] == "current_host_readiness")
            self.assertEqual(integrity["status"], "FAILED")
            self.assertEqual(readiness["status"], "FAILED")

    def test_invalid_readiness_targets_fail_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["readiness"], {
                "schema": "fak.api-host-readiness.v1",
                "summary": {"readiness_gate": False, "models_confirmed": 1, "unclassified": 0, "invalid_targets": 1},
            })
            report = proof.build_report(root)
            self.assertFalse(report["summary"]["proof_gate"])
            readiness = next(req for req in report["requirements"] if req["id"] == "current_host_readiness")
            self.assertEqual(readiness["status"], "FAILED")

    def test_acceptance_probe_can_corroborate_transient_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["readiness"], {
                "schema": "fak.api-host-readiness.v1",
                "summary": {"targets": 13, "readiness_gate": True, "models_confirmed": 0, "unclassified": 0, "invalid_targets": 0},
            })

            report = proof.build_report(root)

            readiness = next(req for req in report["requirements"] if req["id"] == "current_host_readiness")
            self.assertEqual(readiness["status"], "PROVEN")
            self.assertEqual(readiness["detail"]["acceptance_models_confirmed"], 1)

    def test_failed_host_agnostic_conformance_fails_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["gate"], {
                "summary": {"executable_bridge_covered": True, "required_witnesses": 8, "passed_required_witnesses": 8},
                "runs": [
                    {"id": "host_agnostic_openai_compatible", "status": "failed"},
                ],
            })
            report = proof.build_report(root)
            self.assertFalse(report["summary"]["proof_gate"])
            host = next(req for req in report["requirements"] if req["id"] == "host_agnostic_conformance")
            self.assertEqual(host["status"], "FAILED")

    def test_failed_host_profile_conformance_fails_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            data = json.loads((root / proof.DEFAULT_PATHS["gate"]).read_text(encoding="utf-8"))
            for row in data["runs"]:
                if row["id"] == "openai_compatible_host_profiles":
                    row["status"] = "failed"
            write_json(root, proof.DEFAULT_PATHS["gate"], data)

            report = proof.build_report(root)

            self.assertFalse(report["summary"]["proof_gate"])
            profile = next(req for req in report["requirements"] if req["id"] == "host_profile_conformance")
            self.assertEqual(profile["status"], "FAILED")

    def test_host_agnostic_gate_command_must_match_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["gate"], {
                "schema": "fak.api-host-bridge-gate.v1",
                "matrix_schema": "fak.api-host-bridge-matrix.v1",
                "summary": {"executable_bridge_covered": True, "required_witnesses": 8, "passed_required_witnesses": 8},
                "runs": [
                    {
                        "id": "host_agnostic_openai_compatible",
                        "status": "passed",
                        "required": True,
                        "command": "go test ./internal/gateway -run 'TestDifferentWitness$'",
                        "cwd": "fak",
                        "argv": ["go", "test", "./internal/gateway", "-run", "TestDifferentWitness$", "-count=1"],
                    },
                ],
            })
            report = proof.build_report(root)
            self.assertFalse(report["summary"]["proof_gate"])
            host = next(req for req in report["requirements"] if req["id"] == "host_agnostic_conformance")
            self.assertEqual(host["status"], "FAILED")
            self.assertFalse(host["detail"]["command_matches"])

    def test_failed_source_audit_fails_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["source_audit"], {
                "schema": "fak.permission-source-audit.v1",
                "summary": {"sources": 7, "verified": 6, "failed": 1, "source_audit_gate": False},
                "sources": [],
            })
            report = proof.build_report(root)
            self.assertFalse(report["summary"]["proof_gate"])
            source_audit = next(req for req in report["requirements"] if req["id"] == "permission_source_audit")
            self.assertEqual(source_audit["status"], "FAILED")

    def test_failed_acceptance_fails_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["acceptance"], {
                "schema": "fak.api-host-acceptance.v1",
                "summary": {
                    "targets": 1,
                    "known_statuses": 1,
                    "ready_for_live_bridge_run": 0,
                    "unsupported_wire": 1,
                    "unclassified": 0,
                    "sweep_artifact_errors": 0,
                    "invalid_targets": 0,
                    "acceptance_gate": False,
                },
                "targets": [
                    {"name": "bad", "provider": "unknown", "contract_class": "unsupported", "status": "UNSUPPORTED_WIRE"},
                ],
                "artifact_errors": [],
            })
            report = proof.build_report(root)
            self.assertFalse(report["summary"]["proof_gate"])
            acceptance = next(req for req in report["requirements"] if req["id"] == "candidate_host_acceptance")
            self.assertEqual(acceptance["status"], "FAILED")

    def test_failed_roster_fails_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["roster"], {
                "schema": "fak.api-host-roster.v1",
                "summary": {
                    "targets": 1,
                    "openai_compatible_templates": 0,
                    "invalid_targets": 0,
                    "unsupported_wire": 1,
                    "duplicate_names": 0,
                    "roster_gate": False,
                },
                "targets": [{"name": "bad", "status": "UNSUPPORTED_WIRE"}],
            })
            report = proof.build_report(root)
            self.assertFalse(report["summary"]["proof_gate"])
            roster = next(req for req in report["requirements"] if req["id"] == "api_host_roster")
            self.assertEqual(roster["status"], "FAILED")

    def test_failed_external_state_audit_fails_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["external_state"], {
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
            report = proof.build_report(root)
            self.assertFalse(report["summary"]["proof_gate"])
            external = next(req for req in report["requirements"] if req["id"] == "api_host_external_state_audit")
            self.assertEqual(external["status"], "FAILED")

    def test_acceptance_with_only_typed_external_blockers_passes_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["acceptance"], {
                "schema": "fak.api-host-acceptance.v1",
                "summary": {
                    "targets": 13,
                    "known_statuses": 13,
                    "ready_for_live_bridge_run": 0,
                    "live_bridge_confirmed": 0,
                    "unsupported_wire": 0,
                    "unclassified": 0,
                    "sweep_artifact_errors": 0,
                    "invalid_targets": 0,
                    "acceptance_gate": True,
                },
                "targets": [
                    {"name": "missing-key", "provider": "openai-compatible", "contract_class": "openai_compatible_upstream", "status": "NEEDS_AUTH_ENV"},
                    {"name": "billing", "provider": "openai-compatible", "contract_class": "openai_compatible_upstream", "status": "BILLING_REQUIRED"},
                    {"name": "denied", "provider": "openai-compatible", "contract_class": "openai_compatible_upstream", "status": "ACCESS_DENIED"},
                    *[
                        {"name": f"missing-key-{i}", "provider": "openai-compatible", "contract_class": "openai_compatible_upstream", "status": "NEEDS_AUTH_ENV"}
                        for i in range(10)
                    ],
                ],
                "artifact_errors": [],
            })
            report = proof.build_report(root)
            acceptance = next(req for req in report["requirements"] if req["id"] == "candidate_host_acceptance")
            self.assertEqual(acceptance["status"], "PROVEN")

    def test_acceptance_missing_sweep_integrity_fails_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["acceptance"], {
                "schema": "fak.api-host-acceptance.v1",
                "summary": {
                    "targets": 1,
                    "known_statuses": 1,
                    "ready_for_live_bridge_run": 1,
                    "unsupported_wire": 0,
                    "unclassified": 0,
                    "acceptance_gate": True,
                },
                "targets": [
                    {"name": "ready", "provider": "openai-compatible", "contract_class": "openai_compatible_upstream", "status": "READY_FOR_LIVE_BRIDGE_RUN"},
                ],
            })
            report = proof.build_report(root)
            self.assertFalse(report["summary"]["proof_gate"])
            acceptance = next(req for req in report["requirements"] if req["id"] == "candidate_host_acceptance")
            self.assertEqual(acceptance["status"], "FAILED")

    def test_acceptance_missing_artifact_errors_field_fails_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["acceptance"], {
                "schema": "fak.api-host-acceptance.v1",
                "summary": {
                    "targets": 1,
                    "known_statuses": 1,
                    "ready_for_live_bridge_run": 1,
                    "unsupported_wire": 0,
                    "unclassified": 0,
                    "sweep_artifact_errors": 0,
                    "invalid_targets": 0,
                    "acceptance_gate": True,
                },
                "targets": [
                    {"name": "ready", "provider": "openai-compatible", "contract_class": "openai_compatible_upstream", "status": "READY_FOR_LIVE_BRIDGE_RUN"},
                ],
            })
            report = proof.build_report(root)
            self.assertFalse(report["summary"]["proof_gate"])
            acceptance = next(req for req in report["requirements"] if req["id"] == "candidate_host_acceptance")
            self.assertEqual(acceptance["status"], "FAILED")

    def test_acceptance_invalid_targets_fail_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["acceptance"], {
                "schema": "fak.api-host-acceptance.v1",
                "summary": {
                    "targets": 1,
                    "known_statuses": 1,
                    "ready_for_live_bridge_run": 0,
                    "unsupported_wire": 0,
                    "unclassified": 0,
                    "sweep_artifact_errors": 0,
                    "invalid_targets": 1,
                    "acceptance_gate": False,
                },
                "targets": [
                    {"name": "bad", "provider": "openai-compatible", "contract_class": "openai_compatible_upstream", "status": "INVALID_TARGET"},
                ],
                "artifact_errors": [],
            })
            report = proof.build_report(root)
            self.assertFalse(report["summary"]["proof_gate"])
            acceptance = next(req for req in report["requirements"] if req["id"] == "candidate_host_acceptance")
            self.assertEqual(acceptance["status"], "FAILED")

    def test_failed_contract_fails_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["contract"], {
                "schema": "fak.api-host-compat-contract.v1",
                "summary": {"contract_gate": False, "host_classes": 5, "proven_host_classes": 4, "failed_host_classes": 1},
                "host_classes": [{"id": "openai_compatible_upstream", "status": "FAILED"}],
                "non_claims": [],
                "artifact_errors": {},
            })
            report = proof.build_report(root)
            self.assertFalse(report["summary"]["proof_gate"])
            contract = next(req for req in report["requirements"] if req["id"] == "compatibility_contract")
            self.assertEqual(contract["status"], "FAILED")

    def test_failed_live_smoke_queue_fails_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["live_queue"], {
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
            report = proof.build_report(root)
            self.assertFalse(report["summary"]["proof_gate"])
            live_queue = next(req for req in report["requirements"] if req["id"] == "api_host_live_smoke_queue")
            self.assertEqual(live_queue["status"], "FAILED")

    def test_failed_live_smoke_runner_fails_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["live_runner"], {
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
            report = proof.build_report(root)
            self.assertFalse(report["summary"]["proof_gate"])
            live_runner = next(req for req in report["requirements"] if req["id"] == "api_host_live_smoke_runner")
            self.assertEqual(live_runner["status"], "FAILED")

    def test_contract_artifact_errors_fail_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["contract"], {
                "schema": "fak.api-host-compat-contract.v1",
                "summary": {"contract_gate": True, "host_classes": 5, "proven_host_classes": 5, "failed_host_classes": 0},
                "host_classes": [
                    {"id": "openai_compatible_upstream", "status": "PROVEN"},
                    {"id": "native_provider_transcript_adapters", "status": "PROVEN"},
                    {"id": "direct_kernel_http_syscall", "status": "PROVEN"},
                    {"id": "direct_kernel_mcp_syscall", "status": "PROVEN"},
                    {"id": "live_scoped_host_evidence", "status": "PROVEN"},
                ],
                "non_claims": [],
                "artifact_errors": {"readiness": "stale unreadable artifact"},
            })
            report = proof.build_report(root)
            self.assertFalse(report["summary"]["proof_gate"])
            contract = next(req for req in report["requirements"] if req["id"] == "compatibility_contract")
            self.assertEqual(contract["status"], "FAILED")

    def test_failed_certificate_fails_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["certificate"], {
                "schema": "fak.api-host-conformance-certificate.v1",
                "summary": {
                    "capabilities": 1,
                    "proven_capabilities": 0,
                    "failed_capabilities": 1,
                    "missing_required_non_claims": 0,
                    "artifact_errors": 0,
                    "certificate_gate": False,
                },
                "capabilities": [
                    {"id": "openai_compatible_host_conformance", "status": "FAILED"},
                ],
            })
            report = proof.build_report(root)
            self.assertFalse(report["summary"]["proof_gate"])
            certificate = next(req for req in report["requirements"] if req["id"] == "api_host_conformance_certificate")
            self.assertEqual(certificate["status"], "FAILED")

    def test_failed_qualification_fails_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["qualification"], {
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
            report = proof.build_report(root)
            self.assertFalse(report["summary"]["proof_gate"])
            qualification = next(req for req in report["requirements"] if req["id"] == "api_host_qualification")
            self.assertEqual(qualification["status"], "FAILED")

    def test_non_object_artifact_fails_rollup_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            write_json(root, proof.DEFAULT_PATHS["matrix"], [])
            report = proof.build_report(root)
            self.assertFalse(report["summary"]["proof_gate"])
            matrix = next(req for req in report["requirements"] if req["id"] == "source_witness_matrix")
            self.assertEqual(matrix["status"], "MISSING")
            self.assertIn("not a JSON object", matrix["evidence"])

    def test_cli_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            json_path = root / "proof.json"
            md_path = root / "proof.md"
            rc = proof.main(["--root", str(root), "--out", str(json_path), "--markdown", str(md_path)])
            self.assertEqual(rc, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertTrue(data["summary"]["proof_gate"])
            self.assertIn("API-Host Bridge Proof Rollup", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
