#!/usr/bin/env python3
"""Hermetic tests for tools/codex_fak_gate.py."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "codex_fak_gate.py"


def load():
    spec = importlib.util.spec_from_file_location("codex_fak_gate", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_fake_fak(path: Path) -> None:
    if os.name == "nt":
        path.write_text(
            "\n".join(
                [
                    "@echo off",
                    "echo %* | findstr /C:\"--tool git_push\" >NUL",
                    "if %ERRORLEVEL% EQU 0 (",
                    "  echo verdict=DENY reason=POLICY_BLOCK by=monitor",
                    "  exit /B 0",
                    ")",
                    "echo verdict=ALLOW reason=NONE by=monitor",
                    "exit /B 0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        path.write_text(
            "\n".join(
                [
                    "#!/bin/sh",
                    "case \"$*\" in",
                    "  *'--tool git_push'*) echo 'verdict=DENY reason=POLICY_BLOCK by=monitor'; exit 0 ;;",
                    "  *) echo 'verdict=ALLOW reason=NONE by=monitor'; exit 0 ;;",
                    "esac",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        path.chmod(0o755)


class CodexFakGateTest(unittest.TestCase):
    def test_allowed_preflight_runs_command(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fak = root / ("fake-fak.cmd" if os.name == "nt" else "fake-fak")
            marker = root / "ran.txt"
            write_fake_fak(fak)

            code = mod.main(
                [
                    "--repo-root",
                    str(root),
                    "--fak-bin",
                    str(fak),
                    "--policy",
                    "policy.json",
                    "--tool",
                    "run_tests",
                    "--json",
                    "--",
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).write_text('ran', encoding='utf-8')",
                ]
            )

            self.assertEqual(code, 0)
            self.assertEqual(marker.read_text(encoding="utf-8"), "ran")

    def test_redacted_command_report_keeps_digest_not_raw_argv(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fak = root / ("fake-fak.cmd" if os.name == "nt" else "fake-fak")
            marker = root / "ran.txt"
            write_fake_fak(fak)

            code, report = mod.run_gate(
                mod.parse_args(
                    [
                        "--repo-root",
                        str(root),
                        "--fak-bin",
                        str(fak),
                        "--policy",
                        "policy.json",
                        "--tool",
                        "run_tests",
                        "--json",
                        "--redact-command",
                        "--command-label",
                        "secret-safe-test",
                        "--",
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(marker)!r}).write_text('secret-token', encoding='utf-8')",
                    ]
                )
            )

            self.assertEqual(code, 0)
            self.assertEqual(marker.read_text(encoding="utf-8"), "secret-token")
            self.assertNotIn("command", report)
            self.assertTrue(report["command_redacted"])
            self.assertEqual(report["command_label"], "secret-safe-test")
            self.assertEqual(report["command_executable"], Path(sys.executable).name)
            self.assertEqual(report["command_argc"], 3)
            self.assertIsInstance(report["command_digest"], str)
            self.assertNotIn("secret-token", json.dumps(report))

    def test_denied_preflight_does_not_run_command(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fak = root / ("fake-fak.cmd" if os.name == "nt" else "fake-fak")
            marker = root / "ran.txt"
            write_fake_fak(fak)

            code, report = mod.run_gate(
                mod.parse_args(
                    [
                        "--repo-root",
                        str(root),
                        "--fak-bin",
                        str(fak),
                        "--policy",
                        "policy.json",
                        "--tool",
                        "git_push",
                        "--json",
                        "--",
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(marker)!r}).write_text('ran', encoding='utf-8')",
                    ]
                )
            )

            self.assertEqual(code, mod.DENIED_EXIT)
            self.assertEqual(report["status"], "DENIED")
            self.assertEqual(report["preflight"]["verdict"], "DENY")
            self.assertEqual(report["preflight"]["reason"], "POLICY_BLOCK")
            self.assertFalse(marker.exists())

    def test_expected_denial_is_successful_and_does_not_run_command(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fak = root / ("fake-fak.cmd" if os.name == "nt" else "fake-fak")
            marker = root / "ran.txt"
            write_fake_fak(fak)

            code, report = mod.run_gate(
                mod.parse_args(
                    [
                        "--repo-root",
                        str(root),
                        "--fak-bin",
                        str(fak),
                        "--policy",
                        "policy.json",
                        "--tool",
                        "git_push",
                        "--expect-deny",
                        "--expect-reason",
                        "POLICY_BLOCK",
                        "--",
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(marker)!r}).write_text('ran', encoding='utf-8')",
                    ]
                )
            )

            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "DENIED_EXPECTED")
            self.assertTrue(report["expect_deny"])
            self.assertEqual(report["expect_reason"], "POLICY_BLOCK")
            self.assertEqual(report["preflight"]["verdict"], "DENY")
            self.assertEqual(report["preflight"]["reason"], "POLICY_BLOCK")
            self.assertFalse(report["executed"])
            self.assertFalse(marker.exists())

    def test_expected_denial_fails_closed_on_unexpected_allow(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fak = root / ("fake-fak.cmd" if os.name == "nt" else "fake-fak")
            marker = root / "ran.txt"
            write_fake_fak(fak)

            code, report = mod.run_gate(
                mod.parse_args(
                    [
                        "--repo-root",
                        str(root),
                        "--fak-bin",
                        str(fak),
                        "--policy",
                        "policy.json",
                        "--tool",
                        "run_tests",
                        "--expect-deny",
                        "--",
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(marker)!r}).write_text('ran', encoding='utf-8')",
                    ]
                )
            )

            self.assertEqual(code, mod.EXPECTATION_FAILED_EXIT)
            self.assertEqual(report["status"], "UNEXPECTED_ALLOW")
            self.assertEqual(report["preflight"]["verdict"], "ALLOW")
            self.assertFalse(report["executed"])
            self.assertFalse(marker.exists())

    def test_dry_run_allows_without_command_execution(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fak = root / ("fake-fak.cmd" if os.name == "nt" else "fake-fak")
            write_fake_fak(fak)

            code, report = mod.run_gate(
                mod.parse_args(
                    [
                        "--repo-root",
                        str(root),
                        "--fak-bin",
                        str(fak),
                        "--policy",
                        "policy.json",
                        "--tool",
                        "go_test",
                        "--dry-run",
                    ]
                )
            )

            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "ALLOW")
            self.assertFalse(report["executed"])

    def test_main_writes_report(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fak = root / ("fake-fak.cmd" if os.name == "nt" else "fake-fak")
            out = root / "gate.json"
            write_fake_fak(fak)

            code = mod.main(
                [
                    "--repo-root",
                    str(root),
                    "--fak-bin",
                    str(fak),
                    "--policy",
                    "policy.json",
                    "--tool",
                    "go_test",
                    "--dry-run",
                    "--out",
                    str(out),
                ]
            )

            self.assertEqual(code, 0)
            saved = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "ALLOW")
            self.assertEqual(saved["preflight"]["verdict"], "ALLOW")
            self.assertFalse(saved["executed"])

    def test_rejects_invalid_args_json(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fak = root / ("fake-fak.cmd" if os.name == "nt" else "fake-fak")
            write_fake_fak(fak)

            code = mod.main(
                [
                    "--repo-root",
                    str(root),
                    "--fak-bin",
                    str(fak),
                    "--tool",
                    "go_test",
                    "--args-json",
                    "{",
                ]
            )

            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
