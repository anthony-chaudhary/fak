#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import api_host_live_inventory as inv


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class APIHostLiveInventoryTest(unittest.TestCase):
    def test_classify_error(self) -> None:
        self.assertEqual(inv.classify_error("planner: HTTP 402 no_payment_method"), "BILLING_REQUIRED")
        self.assertEqual(inv.classify_error("HTTP 401 Authentication required"), "AUTH_REQUIRED")
        self.assertEqual(inv.classify_error("HTTP 429 rate limit"), "RATE_LIMITED")
        self.assertEqual(inv.classify_error("connection timeout"), "TRANSIENT_TRANSPORT")
        self.assertEqual(inv.classify_error("weird failure"), "UNCLASSIFIED_FAILURE")

    def test_full_inventory_gate_from_minimal_tree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent_live = root / "fak/experiments/agent-live"
            write_json(agent_live / "turntax-injection-live.json", {
                "status": "measured",
                "seam": "LIVE (real model over OpenAI-compatible endpoint)",
                "model": "gemini-2.5-flash",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                "headline": {"both_completed": True, "retry_supported": True},
                "trials": [
                    {"trial": 1, "transcript_sha": "a"},
                    {"trial": 2, "transcript_sha": "b"},
                    {"trial": 3, "transcript_sha": "c"},
                ],
                "raw_trial_1_arms": {
                    "fak": {"injection_in_context": False},
                    "baseline": {"injection_in_context": True},
                },
            })
            for i in range(5):
                write_json(agent_live / f"gemini-2.5-flash-t{i}.json", {
                    "model": "gemini-2.5-flash",
                    "live": True,
                    "transcript_sha": f"sha{i}",
                    "both_completed": True,
                    "fak": {"task_completed": True, "injection_in_context": False, "quarantines": 1},
                    "baseline": {"injection_in_context": True},
                })
            write_json(agent_live / "transcript-adapter-sweep/sweep-summary.json", [
                {
                    "kind": "api",
                    "base_url": "https://gateway.glama.ai/v1",
                    "model": "m",
                    "status": "failed",
                    "error": "HTTP 402: no_payment_method",
                },
                {
                    "kind": "local-shim",
                    "base_url": "http://127.0.0.1:1/v1",
                    "model": "local-a",
                    "status": "ok",
                    "live": True,
                    "both_completed": True,
                    "poison_blocked": False,
                    "transcript_sha": "local-a",
                },
            ])
            write_json(agent_live / "transcript-adapter-sweep-pollinations/sweep-summary.json", [
                {
                    "kind": "api",
                    "base_url": "https://gen.pollinations.ai/v1",
                    "model": "openai-fast",
                    "status": "failed",
                    "error": "HTTP 401 Authentication required",
                },
                {
                    "kind": "local-shim",
                    "base_url": "http://127.0.0.1:2/v1",
                    "model": "local-b",
                    "status": "ok",
                    "live": True,
                    "both_completed": True,
                    "poison_blocked": True,
                    "transcript_sha": "local-b",
                },
            ])

            report = inv.build_report(root)
            self.assertEqual(report["schema"], inv.SCHEMA)
            self.assertTrue(report["app_version"])
            self.assertTrue(all(p["version"] == report["app_version"] for p in report["proofs"]))
            self.assertTrue(report["summary"]["live_inventory_gate"])
            self.assertEqual(report["summary"]["live_frontier_successes"], 2)
            self.assertEqual(report["summary"]["billing_required_hosts"], 1)
            self.assertEqual(report["summary"]["auth_required_hosts"], 1)
            self.assertEqual(report["summary"]["incomplete_or_unclassified"], 0)

    def test_missing_tree_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = inv.build_report(Path(td))
            self.assertFalse(report["summary"]["live_inventory_gate"])
            self.assertGreater(report["summary"]["incomplete_or_unclassified"], 0)

    def test_non_object_turntax_artifact_fails_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent_live = root / "fak/experiments/agent-live"
            write_json(agent_live / "turntax-injection-live.json", [])
            report = inv.build_report(root)
            self.assertFalse(report["summary"]["live_inventory_gate"])
            turntax = next(p for p in report["proofs"] if p["id"] == "gemini_openai_compatible_turntax")
            self.assertEqual(turntax["status"], "INCOMPLETE")
            self.assertIn("artifact_error", turntax["evidence"])

    def test_malformed_sweep_rows_fail_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent_live = root / "fak/experiments/agent-live"
            write_json(agent_live / "transcript-adapter-sweep/sweep-summary.json", [
                [],
                {
                    "kind": "api",
                    "base_url": "https://gateway.glama.ai/v1",
                    "model": "m",
                    "status": "failed",
                    "error": "HTTP 402: no_payment_method",
                },
            ])
            report = inv.build_report(root)
            self.assertFalse(report["summary"]["live_inventory_gate"])
            integrity = next(p for p in report["proofs"] if p["id"] == "sweep_artifact_integrity")
            self.assertEqual(integrity["status"], "INCOMPLETE")
            self.assertEqual(integrity["evidence"]["rows"][0]["error"], "sweep row is not a JSON object")

    def test_single_object_sweep_summary_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            agent_live = root / "fak/experiments/agent-live"
            write_json(agent_live / "transcript-adapter-sweep/sweep-summary.json", {
                "kind": "api",
                "base_url": "https://gateway.glama.ai/v1",
                "model": "m",
                "status": "failed",
                "error": "HTTP 402: no_payment_method",
            })
            rows = inv.load_sweep_rows(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["model"], "m")
            self.assertTrue(rows[0]["version"])

    def test_markdown_surfaces_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = inv.build_report(Path(td))
            md = inv.markdown(report)
            self.assertIn("API-Host Live Inventory", md)
            self.assertIn("Live inventory gate: no", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
