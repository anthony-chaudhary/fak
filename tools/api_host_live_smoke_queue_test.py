#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import api_host_live_smoke_queue as queue


def write_json(root: Path, rel_path: str, data: object) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def qual_target(name: str, status: str, commands: list[str] | None = None, in_contract: bool = True) -> dict[str, object]:
    return {
        "name": name,
        "provider": "openai-compatible",
        "contract_class": "openai_compatible_upstream",
        "base_url": f"https://{name}.example/v1",
        "model_hint": f"{name}-model",
        "api_key_env": f"{name.upper()}_KEY",
        "qualification_status": status,
        "evidence_state": "NEEDS_OPERATOR_STATE",
        "external_state_status": "NEEDS_CREDENTIAL",
        "in_contract": in_contract,
        "next_evidence_needed": f"next {name}",
        "commands": commands or [],
    }


def action(name: str, status: str, commands: list[str]) -> dict[str, object]:
    return {
        "target": name,
        "status": status,
        "action_type": "retry",
        "operator_prerequisite": f"operator state for {name}",
        "required_env": f"{name.upper()}_KEY",
        "commands": commands,
        "latest_sweep": {"status": "failed"},
    }


def good_artifacts(root: Path) -> None:
    targets = [
        qual_target("done", "IN_CONTRACT_LIVE_CONFIRMED"),
        qual_target("ready", "IN_CONTRACT_READY_FOR_LIVE_SMOKE", ["run ready"]),
        qual_target("billing", "IN_CONTRACT_EXTERNAL_BLOCKER"),
        qual_target("missing", "IN_CONTRACT_NEEDS_CREDENTIAL"),
        qual_target("probe", "IN_CONTRACT_NEEDS_PROBE"),
    ]
    write_json(root, queue.DEFAULT_PATHS["qualification"], {
        "schema": "fak.api-host-qualification.v1",
        "summary": {"qualification_gate": True, "targets": len(targets)},
        "targets": targets,
    })
    write_json(root, queue.DEFAULT_PATHS["retry"], {
        "schema": "fak.api-host-retry-packet.v1",
        "summary": {"retry_packet_gate": True},
        "actions": [
            action("billing", "BILLING_REQUIRED", ["run billing"]),
            action("missing", "NEEDS_AUTH_ENV", ["probe missing", "run missing"]),
            action("probe", "WIRE_SUPPORTED_UNPROBED", ["probe probe"]),
        ],
    })


class APIHostLiveSmokeQueueTest(unittest.TestCase):
    def test_good_artifacts_build_known_queue_states(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)

            report = queue.build_report(root)

            self.assertEqual(report["schema"], queue.SCHEMA)
            self.assertTrue(report["app_version"])
            self.assertTrue(all(row["version"] == report["app_version"] for row in report["queue"]))
            self.assertTrue(report["summary"]["live_smoke_queue_gate"])
            self.assertEqual(report["summary"]["targets"], 5)
            self.assertEqual(report["summary"]["complete"], 1)
            self.assertEqual(report["summary"]["ready_to_execute"], 1)
            self.assertEqual(report["summary"]["blocked_external_state"], 1)
            self.assertEqual(report["summary"]["waiting_for_credential"], 1)
            self.assertEqual(report["summary"]["ready_for_probe"], 1)
            states = {row["target"]: row["queue_state"] for row in report["queue"]}
            self.assertEqual(states["done"], "COMPLETE")
            self.assertEqual(states["ready"], "READY_TO_EXECUTE")
            self.assertEqual(states["billing"], "BLOCKED_EXTERNAL_STATE")
            self.assertEqual(states["missing"], "WAITING_FOR_CREDENTIAL")
            self.assertEqual(states["probe"], "READY_FOR_PROBE")
            billing = next(row for row in report["queue"] if row["target"] == "billing")
            self.assertEqual(billing["commands"], ["run billing"])
            self.assertEqual(billing["operator_prerequisite"], "operator state for billing")

    def test_missing_artifact_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = queue.build_report(Path(td))

            self.assertFalse(report["summary"]["live_smoke_queue_gate"])
            self.assertEqual(report["summary"]["artifact_errors"], 2)
            self.assertIn("qualification", report["artifact_errors"])
            self.assertIn("retry", report["artifact_errors"])

    def test_unqualified_target_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            data = json.loads((root / queue.DEFAULT_PATHS["qualification"]).read_text(encoding="utf-8"))
            data["targets"].append(qual_target("bad", "OUT_OF_CONTRACT", in_contract=False))
            write_json(root, queue.DEFAULT_PATHS["qualification"], data)

            report = queue.build_report(root)

            self.assertFalse(report["summary"]["live_smoke_queue_gate"])
            self.assertEqual(report["summary"]["unqualified"], 1)

    def test_command_gap_fails_gate_for_next_action_states(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            targets = [qual_target("ready", "IN_CONTRACT_READY_FOR_LIVE_SMOKE")]
            write_json(root, queue.DEFAULT_PATHS["qualification"], {
                "schema": "fak.api-host-qualification.v1",
                "summary": {"qualification_gate": True},
                "targets": targets,
            })
            write_json(root, queue.DEFAULT_PATHS["retry"], {
                "schema": "fak.api-host-retry-packet.v1",
                "summary": {"retry_packet_gate": True},
                "actions": [],
            })

            report = queue.build_report(root)

            self.assertFalse(report["summary"]["live_smoke_queue_gate"])
            self.assertEqual(report["summary"]["command_gaps"], 1)

    def test_does_not_read_secret_values_from_environment(self) -> None:
        old_value = os.environ.get("MISSING_KEY")
        os.environ["MISSING_KEY"] = "literal-secret-value"
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                good_artifacts(root)

                report = queue.build_report(root)
                body = json.dumps(report)

                self.assertTrue(report["summary"]["live_smoke_queue_gate"])
                self.assertNotIn("literal-secret-value", body)
                self.assertIn("MISSING_KEY", body)
        finally:
            if old_value is None:
                os.environ.pop("MISSING_KEY", None)
            else:
                os.environ["MISSING_KEY"] = old_value

    def test_cli_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good_artifacts(root)
            json_path = root / "queue.json"
            md_path = root / "queue.md"

            rc = queue.main(["--root", str(root), "--out", str(json_path), "--markdown", str(md_path)])

            self.assertEqual(rc, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], queue.SCHEMA)
            self.assertTrue(data["app_version"])
            self.assertIn("API-Host Live Smoke Queue", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
