#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import api_host_retry_packet as retry


def write_json(root: Path, rel_path: str, data: object) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def acceptance(targets: list[dict[str, object]], gate: bool = True) -> dict[str, object]:
    return {
        "schema": "fak.api-host-acceptance.v1",
        "summary": {
            "targets": len(targets),
            "known_statuses": len([t for t in targets if t.get("status") != "UNCLASSIFIED"]),
            "ready_for_live_bridge_run": len([t for t in targets if t.get("status") == "READY_FOR_LIVE_BRIDGE_RUN"]),
            "live_bridge_confirmed": len([t for t in targets if t.get("status") == "LIVE_BRIDGE_CONFIRMED"]),
            "typed_external_blockers": len([t for t in targets if t.get("status") in retry.EXTERNAL_BLOCKERS]),
            "unsupported_wire": len([t for t in targets if t.get("status") == "UNSUPPORTED_WIRE"]),
            "invalid_targets": len([t for t in targets if t.get("status") == "INVALID_TARGET"]),
            "unclassified": len([t for t in targets if t.get("status") == "UNCLASSIFIED"]),
            "sweep_artifact_errors": 0,
            "acceptance_gate": gate,
        },
        "targets": targets,
        "artifact_errors": [],
    }


def target(name: str, status: str, env: str = "", model: str = "m1") -> dict[str, object]:
    return {
        "name": name,
        "provider": "openai-compatible",
        "contract_class": "openai_compatible_upstream",
        "base_url": f"https://{name}.example/v1",
        "api_key_env": env,
        "model_hint": model,
        "status": status,
        "readiness_status": "MODELS_CONFIRMED",
        "latest_sweep": None,
    }


class APIHostRetryPacketTest(unittest.TestCase):
    def test_external_blockers_get_exact_retry_actions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [
                target("gemini_openai_compatible", "NEEDS_AUTH_ENV", "GEMINI_API_KEY", "gemini-2.5-flash"),
                {
                    **target("glama_gateway", "BILLING_REQUIRED", "GLAMA_API_KEY", "openai/gpt-4.1-nano-2025-04-14"),
                    "latest_sweep": {
                        "_summary_path": "fak/experiments/agent-live/transcript-adapter-sweep-glama-live-smoke/sweep-summary.json",
                        "status": "failed",
                        "model": "openai/gpt-4.1-nano-2025-04-14",
                        "error": "HTTP 402 no_payment_method",
                    },
                },
                target("pollinations_no_key", "AUTH_REQUIRED", "", "openai-fast"),
            ]
            write_json(root, retry.DEFAULT_PATHS["acceptance"], acceptance(rows))
            write_json(root, retry.DEFAULT_PATHS["live"], {
                "schema": "fak.api-host-live-inventory.v1",
                "summary": {"live_inventory_gate": True},
                "proofs": [
                    {
                        "id": "glama_gateway_billing_state",
                        "status": "BILLING_REQUIRED",
                        "evidence": {"base_url": "https://glama_gateway.example/v1"},
                    },
                ],
            })

            report = retry.build_report(root)

            self.assertTrue(report["app_version"])
            self.assertTrue(report["summary"]["retry_packet_gate"])
            self.assertEqual(report["summary"]["actionable_blockers"], 3)
            self.assertTrue(all(a["version"] == report["app_version"] for a in report["actions"]))
            actions = {a["target"]: a for a in report["actions"]}
            self.assertEqual(actions["gemini_openai_compatible"]["action_type"], "set_auth_env_then_probe_and_smoke")
            self.assertIn("GEMINI_API_KEY", actions["gemini_openai_compatible"]["commands"][0])
            self.assertEqual(actions["glama_gateway"]["action_type"], "fix_billing_then_smoke")
            self.assertEqual(actions["glama_gateway"]["latest_sweep"]["summary_path"], "fak/experiments/agent-live/transcript-adapter-sweep-glama-live-smoke/sweep-summary.json")
            self.assertEqual(actions["pollinations_no_key"]["required_env"], "POLLINATIONS_API_KEY")
            self.assertIn("-OutDir fak/experiments/agent-live/transcript-adapter-sweep-pollinations-no-key-retry", actions["pollinations_no_key"]["commands"][-1])

    def test_ready_and_live_confirmed_targets_pass_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [
                target("ready", "READY_FOR_LIVE_BRIDGE_RUN", "READY_KEY"),
                target("done", "LIVE_BRIDGE_CONFIRMED", "DONE_KEY"),
            ]
            write_json(root, retry.DEFAULT_PATHS["acceptance"], acceptance(rows))
            write_json(root, retry.DEFAULT_PATHS["live"], {"schema": "fak.api-host-live-inventory.v1", "summary": {}, "proofs": []})

            report = retry.build_report(root)

            self.assertTrue(report["summary"]["retry_packet_gate"])
            actions = {a["target"]: a for a in report["actions"]}
            self.assertEqual(actions["ready"]["action_type"], "run_live_smoke")
            self.assertEqual(len(actions["ready"]["commands"]), 1)
            self.assertEqual(actions["done"]["action_type"], "none_already_confirmed")
            self.assertEqual(actions["done"]["commands"], [])

    def test_unclassified_or_unsupported_targets_fail_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [
                target("unknown", "UNCLASSIFIED", "KEY"),
                {**target("unsupported", "UNSUPPORTED_WIRE", "KEY"), "provider": "custom-wire", "contract_class": "unsupported"},
            ]
            write_json(root, retry.DEFAULT_PATHS["acceptance"], acceptance(rows, gate=False))
            write_json(root, retry.DEFAULT_PATHS["live"], {"schema": "fak.api-host-live-inventory.v1", "summary": {}, "proofs": []})

            report = retry.build_report(root)

            self.assertFalse(report["summary"]["retry_packet_gate"])
            self.assertEqual(report["summary"]["unclassified"], 1)
            self.assertEqual(report["summary"]["unsupported_wire"], 1)

    def test_missing_acceptance_artifact_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_json(root, retry.DEFAULT_PATHS["live"], {"schema": "fak.api-host-live-inventory.v1", "summary": {}, "proofs": []})

            report = retry.build_report(root)

            self.assertFalse(report["summary"]["retry_packet_gate"])
            self.assertEqual(report["summary"]["artifact_errors"], 1)
            self.assertIn("acceptance", report["artifact_errors"])

    def test_acceptance_artifact_errors_fail_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            doc = acceptance([target("ready", "READY_FOR_LIVE_BRIDGE_RUN", "READY_KEY")])
            doc["artifact_errors"] = [{"path": "bad.json", "error": "invalid JSON"}]
            write_json(root, retry.DEFAULT_PATHS["acceptance"], doc)
            write_json(root, retry.DEFAULT_PATHS["live"], {"schema": "fak.api-host-live-inventory.v1", "summary": {}, "proofs": []})

            report = retry.build_report(root)

            self.assertFalse(report["summary"]["retry_packet_gate"])
            self.assertIn("acceptance_artifact_errors", report["artifact_errors"])

    def test_stale_acceptance_integrity_fields_fail_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            doc = acceptance([target("ready", "READY_FOR_LIVE_BRIDGE_RUN", "READY_KEY")])
            del doc["summary"]["sweep_artifact_errors"]
            del doc["summary"]["invalid_targets"]
            write_json(root, retry.DEFAULT_PATHS["acceptance"], doc)
            write_json(root, retry.DEFAULT_PATHS["live"], {"schema": "fak.api-host-live-inventory.v1", "summary": {}, "proofs": []})

            report = retry.build_report(root)

            self.assertFalse(report["summary"]["retry_packet_gate"])
            self.assertIn("acceptance_sweep_integrity", report["artifact_errors"])
            self.assertIn("acceptance_target_integrity", report["artifact_errors"])

    def test_cli_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_json(root, retry.DEFAULT_PATHS["acceptance"], acceptance([target("ready", "READY_FOR_LIVE_BRIDGE_RUN", "READY_KEY")]))
            write_json(root, retry.DEFAULT_PATHS["live"], {"schema": "fak.api-host-live-inventory.v1", "summary": {}, "proofs": []})
            json_path = root / "retry.json"
            md_path = root / "retry.md"

            rc = retry.main(["--root", str(root), "--out", str(json_path), "--markdown", str(md_path)])

            self.assertEqual(rc, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], retry.SCHEMA)
            self.assertTrue(data["app_version"])
            self.assertIn("API-Host Retry Packet", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
