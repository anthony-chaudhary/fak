#!/usr/bin/env python3
"""Hermetic tests for install_agent_command.py."""
from __future__ import annotations

import stat
import sys
import tempfile
import unittest
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import install_agent_command as installer  # noqa: E402


class InstallAgentCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bin_dir = self.root / "bin"
        self.launcher = self.root / "tools" / "claude_agent_chat.py"
        self.launcher.parent.mkdir()
        self.launcher.write_text("print('launcher')\n", encoding="utf-8")
        self.addCleanup(self.tmp.cleanup)

    def test_installs_windows_cmd_and_powershell_wrappers(self) -> None:
        result = installer.install(
            name="fleet-agent",
            bin_dir=self.bin_dir,
            python_exe="python",
            launcher=self.launcher,
            platform="win32",
        )

        cmd = self.bin_dir / "fleet-agent.cmd"
        ps1 = self.bin_dir / "fleet-agent.ps1"
        self.assertTrue(result["ok"])
        self.assertTrue(cmd.exists())
        self.assertTrue(ps1.exists())
        self.assertIn(str(self.launcher), cmd.read_text(encoding="utf-8"))
        self.assertIn("%*", cmd.read_text(encoding="utf-8"))
        self.assertIn("@args", ps1.read_text(encoding="utf-8"))

    def test_installs_executable_posix_wrapper(self) -> None:
        result = installer.install(
            name="fleet-agent",
            bin_dir=self.bin_dir,
            python_exe="/usr/bin/python3",
            launcher=self.launcher,
            platform="linux",
        )

        target = self.bin_dir / "fleet-agent"
        self.assertTrue(result["ok"])
        text = target.read_text(encoding="utf-8")
        if os.name != "nt":
            self.assertTrue(target.stat().st_mode & stat.S_IXUSR)
        self.assertIn("#!/usr/bin/env sh", text)
        self.assertIn("exec '/usr/bin/python3'", text)
        self.assertIn(str(self.launcher), text)

    def test_refuses_to_overwrite_foreign_wrapper_without_force(self) -> None:
        self.bin_dir.mkdir()
        target = self.bin_dir / "fleet-agent.cmd"
        target.write_text("@echo off\n", encoding="utf-8")

        result = installer.install(
            name="fleet-agent",
            bin_dir=self.bin_dir,
            python_exe="python",
            launcher=self.launcher,
            platform="win32",
        )

        self.assertFalse(result["ok"])
        self.assertIn("refusing to overwrite", result["reason"])
        self.assertEqual(target.read_text(encoding="utf-8"), "@echo off\n")

    def test_uninstall_removes_only_owned_wrappers(self) -> None:
        installer.install(
            name="fleet-agent",
            bin_dir=self.bin_dir,
            python_exe="python",
            launcher=self.launcher,
            platform="win32",
        )

        result = installer.uninstall(
            name="fleet-agent",
            bin_dir=self.bin_dir,
            platform="win32",
        )

        self.assertTrue(result["ok"])
        self.assertFalse((self.bin_dir / "fleet-agent.cmd").exists())
        self.assertFalse((self.bin_dir / "fleet-agent.ps1").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
