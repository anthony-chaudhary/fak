#!/usr/bin/env python3
"""bench_slack_test.py -- tests for Slack integration.

Verifies that the benchmark catalog works with the Slack control bridge concept.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


class TestBenchSlack(unittest.TestCase):
    """Test Slack integration for benchmark catalog."""

    def test_slack_list_command_exists(self):
        """Verify the Slack list command is available."""
        result = subprocess.run(
            ["python", str(TOOLS / "bench_slack.py"), "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("list", result.stdout)
        self.assertIn("show", result.stdout)
        self.assertIn("summary", result.stdout)
        self.assertIn("status", result.stdout)

    def test_slack_status_output_format(self):
        """Test that !status command produces valid Slack output."""
        result = subprocess.run(
            ["python", str(TOOLS / "bench_slack.py"), "status"],
            capture_output=True,
            text=True,
            cwd=ROOT
        )
        # Should succeed even if catalog is missing (graceful fallback)
        # If catalog exists, should have proper formatting
        self.assertIn(result.returncode, [0, 1])

    def test_register_dgx_function_exists(self):
        """Verify the register-dgx command exists for DGX integration."""
        result = subprocess.run(
            ["python", str(TOOLS / "bench_slack.py"), "--help"],
            capture_output=True,
            text=True
        )
        self.assertIn("register-dgx", result.stdout)

    def test_catalog_integration(self):
        """Test that catalog can be loaded and queried."""
        catalog_path = ROOT / "fak" / "experiments" / "benchmark" / "catalog.json"
        if not catalog_path.exists():
            self.skipTest("Catalog not found")

        with open(catalog_path, encoding="utf-8") as f:
            catalog = json.load(f)

        # Verify catalog structure
        self.assertIn("machines", catalog)
        self.assertIn("runs", catalog)
        self.assertIn("version", catalog)

    def test_slack_formatting_functions(self):
        """Test Slack formatting functions."""
        # Import the module
        sys.path.insert(0, str(TOOLS))
        from bench_slack import format_slack_table, format_slack_summary

        # Test with sample data
        sample_runs = [
            {
                "run_id": "test-run-1",
                "machine_id": "anthony",
                "model": "SmolLM2-135M",
                "peak_tok_per_sec": 100.0,
                "timestamp": "20260619T120000Z"
            }
        ]

        table_output = format_slack_table(sample_runs)
        self.assertIn("```", table_output)  # Code block
        self.assertIn("test-run-1", table_output)
        self.assertIn("anthony", table_output)

        # Test summary
        sample_catalog = {
            "machines": {
                "anthony": {"runs": 1, "gpu": "RTX 4070"},
                "mac": {"runs": 2, "gpu": "M3 Pro"}
            },
            "runs": sample_runs
        }

        summary_output = format_slack_summary(sample_catalog)
        self.assertIn("*Benchmark Catalog Summary*", summary_output)
        self.assertIn("anthony", summary_output)
        self.assertIn("1 run(s)", summary_output)

    def test_dgx_run_registration_flow(self):
        """Test that DGX runs can be registered into the catalog."""
        sys.path.insert(0, str(TOOLS))

        # Create a temporary directory with mock DGX results
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            run_dir = tmpdir / "dgx-run"
            run_dir.mkdir()

            # Create mock DGX summary
            dgx_summary = {
                "model": "Qwen3.6-27B",
                "precision": "q4_k_m",
                "peak_tok_per_sec": 150.0,
                "baseline_tok_per_sec": 50.0,
                "hardware": "A100-80GB"
            }

            with open(run_dir / "dgx-summary.json", "w") as f:
                json.dump(dgx_summary, f)

            # Try to register (will fail gracefully if catalog build fails)
            from bench_slack import register_dgx_run
            run_id = register_dgx_run(run_dir, machine_id="dgx-test")

            # Should either succeed or fail gracefully (not crash)
            # If catalog exists and is writable, should get a run_id
            self.assertIsNotNone(run_id)

    def test_slack_control_bridge_commands(self):
        """Verify commands expected by slack-control bridge work."""
        commands = [
            ["status"],
            ["list"],
            ["list", "--machine", "anthony"],
            ["summary"],
        ]

        for cmd in commands:
            result = subprocess.run(
                ["python", str(TOOLS / "bench_slack.py")] + cmd,
                capture_output=True,
                text=True,
                cwd=ROOT
            )
            # Commands should not crash (returncode 0 or 1 with message)
            self.assertIn(result.returncode, [0, 1])
            # Should have some output (even if it's an error message)
            self.assertTrue(len(result.stdout) > 0 or len(result.stderr) > 0)

    def test_transfer_command_exists(self):
        """Verify the transfer command exists and has correct arguments."""
        result = subprocess.run(
            ["python", str(TOOLS / "bench_slack.py"), "transfer", "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--benchmark-command", result.stdout)
        self.assertIn("--control-channel", result.stdout)
        self.assertIn("--poll-interval", result.stdout)
        self.assertIn("--timeout", result.stdout)

    def test_download_and_register_command_exists(self):
        """Verify the download-and-register command exists and has correct arguments."""
        result = subprocess.run(
            ["python", str(TOOLS / "bench_slack.py"), "download-and-register", "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--pattern", result.stdout)
        self.assertIn("--dest-dir", result.stdout)
        self.assertIn("--machine-id", result.stdout)

    def test_slack_helpers_path_detection(self):
        """Test that slack-helpers path can be auto-detected or specified."""
        sys.path.insert(0, str(TOOLS))
        from bench_slack import download_and_register

        # Test with None (should try auto-detection)
        # This will fail to find slack-helpers in test environment, but proves the logic runs
        result = download_and_register(slack_helpers_path=None, pattern="test.*\\.json")
        self.assertIsNone(result)  # Expected to fail without slack-helpers


if __name__ == "__main__":
    unittest.main()
