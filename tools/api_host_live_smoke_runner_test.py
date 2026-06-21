#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import api_host_live_smoke_runner as runner


def write_json(root: Path, rel_path: str, data: object) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def queue_row(name: str, state: str, commands: list[str] | None = None) -> dict[str, object]:
    return {
        "target": name,
        "queue_state": state,
        "operator_prerequisite": "none" if state in {"COMPLETE", "READY_TO_EXECUTE"} else "external state",
        "commands": commands or [],
    }


def write_queue(root: Path, rows: list[dict[str, object]], gate: bool = True) -> None:
    write_json(root, runner.DEFAULT_PATHS["queue"], {
        "schema": "fak.api-host-live-smoke-queue.v1",
        "summary": {
            "targets": len(rows),
            "complete": len([row for row in rows if row["queue_state"] == "COMPLETE"]),
            "ready_to_execute": len([row for row in rows if row["queue_state"] == "READY_TO_EXECUTE"]),
            "blocked_external_state": len([row for row in rows if row["queue_state"] == "BLOCKED_EXTERNAL_STATE"]),
            "waiting_for_credential": len([row for row in rows if row["queue_state"] == "WAITING_FOR_CREDENTIAL"]),
            "ready_for_probe": len([row for row in rows if row["queue_state"] == "READY_FOR_PROBE"]),
            "unqualified": 0,
            "unclassified": 0,
            "command_gaps": 0,
            "artifact_errors": 0,
            "live_smoke_queue_gate": gate,
        },
        "queue": rows,
    })


class APIHostLiveSmokeRunnerTest(unittest.TestCase):
    def test_no_ready_rows_passes_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_queue(root, [
                queue_row("done", "COMPLETE"),
                queue_row("billing", "BLOCKED_EXTERNAL_STATE", ["run billing"]),
                queue_row("missing", "WAITING_FOR_CREDENTIAL", ["run missing"]),
                queue_row("probe", "READY_FOR_PROBE", ["probe first"]),
            ])

            report = runner.build_report(root)

            self.assertEqual(report["schema"], runner.SCHEMA)
            self.assertTrue(report["app_version"])
            self.assertTrue(all(row["version"] == report["app_version"] for row in report["runs"]))
            self.assertTrue(report["summary"]["live_smoke_runner_gate"])
            self.assertEqual(report["summary"]["already_complete"], 1)
            self.assertEqual(report["summary"]["ready_to_execute"], 0)
            self.assertEqual(report["summary"]["skipped_external_state"], 1)
            self.assertEqual(report["summary"]["skipped_waiting_for_credential"], 1)
            self.assertEqual(report["summary"]["skipped_ready_for_probe"], 1)

    def test_ready_rows_fail_closed_in_dry_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_queue(root, [
                queue_row("ready", "READY_TO_EXECUTE", ["python -c \"print(123)\""]),
            ])

            report = runner.build_report(root)

            self.assertFalse(report["summary"]["live_smoke_runner_gate"])
            self.assertEqual(report["summary"]["ready_not_executed"], 1)
            self.assertEqual(report["runs"][0]["runner_status"], "READY_NOT_EXECUTED")

    def test_execute_ready_rows_passes_on_zero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            command = f"\"{sys.executable}\" -c \"print(123)\""
            write_queue(root, [queue_row("ready", "READY_TO_EXECUTE", [command])])

            report = runner.build_report(root, execute_ready=True, timeout_s=30)

            self.assertTrue(report["summary"]["live_smoke_runner_gate"])
            self.assertEqual(report["summary"]["executed"], 1)
            self.assertEqual(report["summary"]["passed"], 1)
            self.assertEqual(report["runs"][0]["runner_status"], "EXECUTED_PASSED")
            self.assertEqual(report["runs"][0]["command_results"][0]["exit_code"], 0)
            self.assertEqual(report["runs"][0]["command_results"][0]["version"], report["app_version"])

    def test_execute_ready_rows_fails_on_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            command = f"\"{sys.executable}\" -c \"import sys; sys.exit(3)\""
            write_queue(root, [queue_row("ready", "READY_TO_EXECUTE", [command])])

            report = runner.build_report(root, execute_ready=True, timeout_s=30)

            self.assertFalse(report["summary"]["live_smoke_runner_gate"])
            self.assertEqual(report["summary"]["executed"], 1)
            self.assertEqual(report["summary"]["failed"], 1)
            self.assertEqual(report["runs"][0]["runner_status"], "EXECUTED_FAILED")
            self.assertEqual(report["runs"][0]["command_results"][0]["exit_code"], 3)

    def test_missing_queue_artifact_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = runner.build_report(Path(td))

            self.assertFalse(report["summary"]["live_smoke_runner_gate"])
            self.assertEqual(report["summary"]["artifact_errors"], 1)
            self.assertIn("queue", report["artifact_errors"])

    def test_cli_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_queue(root, [queue_row("done", "COMPLETE")])
            json_path = root / "runner.json"
            md_path = root / "runner.md"

            rc = runner.main(["--root", str(root), "--out", str(json_path), "--markdown", str(md_path)])

            self.assertEqual(rc, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], runner.SCHEMA)
            self.assertTrue(data["app_version"])
            self.assertIn("API-Host Live Smoke Runner", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
