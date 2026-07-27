#!/usr/bin/env python3
"""Tests for tools/fak_loop_task.ps1 -- the shared Scheduled Task installer helpers.

A Scheduled Task action FREEZES the executable path it is handed at install time. The
Microsoft Store build of pwsh lives under a version-stamped directory
(...\\WindowsApps\\Microsoft.PowerShell_<VERSION>_x64__8wekyb3d8bbwe\\pwsh.exe), so a task
holding that absolute path stops working the moment PowerShell updates -- exit 1 on every
fire, no diagnostic. scout-loop/task-scheduler went dark exactly that way: pinned to a
7.6.3.0 that no longer existed while 7.6.4.0 was installed, 10 recorded runs all ending
EXIT_NONZERO.

The source assertions below are the portable half (CI is Linux; these registrars are
Windows-only). The behavioural half runs only where a real pwsh exists.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
HELPERS = TOOLS / "fak_loop_task.ps1"
SCOUT = TOOLS / "register_scout_loop.ps1"

# The version-stamped Store layout that must never be frozen into a task.
VERSIONED_STORE = re.compile(r"WindowsApps\\+Microsoft\.PowerShell_\[?\\?d")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def resolver_code(text: str) -> str:
    """The resolver's body with its `<# ... #>` doc block stripped.

    Ordering assertions must read the CODE, not the prose: the doc block explains why
    `Get-Command pwsh` is the wrong thing to freeze, and that mention would otherwise
    satisfy an index() lookup before the real call site.
    """
    start = text.index("function Resolve-FakLoopPowerShellHost")
    body = text[start:text.index("\nfunction ", start + 1)]
    return re.sub(r"<#.*?#>", "", body, flags=re.DOTALL)


class ResolvePowerShellHostSourceTest(unittest.TestCase):
    """Source-level contract. Runs everywhere, including Linux CI."""

    def test_helper_defines_the_resolver(self) -> None:
        self.assertIn("function Resolve-FakLoopPowerShellHost", read(HELPERS))

    def test_resolver_prefers_upgrade_stable_paths_before_get_command(self) -> None:
        body = resolver_code(read(HELPERS))
        # Both upgrade-stable candidates are probed...
        self.assertIn("PowerShell\\7\\pwsh.exe", body)
        self.assertIn("Microsoft\\WindowsApps\\pwsh.exe", body)
        # ...and they are probed BEFORE falling back to the version-stamped resolution.
        self.assertLess(body.index("LOCALAPPDATA"), body.index("Get-Command pwsh"),
                        "the stable App Execution Alias must be preferred over "
                        "Get-Command pwsh, which returns the version-stamped Store path")

    def test_resolver_warns_when_only_the_version_stamped_path_exists(self) -> None:
        """Installing is better than refusing, but the hazard must be loud: that task
        will stop firing at the next PowerShell update."""
        body = resolver_code(read(HELPERS))
        self.assertIn("Write-Warning", body)
        self.assertRegex(body, r"WindowsApps\\+Microsoft\\?\.PowerShell_")

    def test_scout_registrar_uses_the_resolver_not_get_command(self) -> None:
        """register_scout_loop.ps1 is the one registrar that prefers pwsh, so it is the
        one that froze a Store path. Its siblings prefer Windows PowerShell, whose system
        path is already version-independent."""
        text = read(SCOUT)
        self.assertIn("Resolve-FakLoopPowerShellHost", text)
        self.assertNotIn("(Get-Command pwsh -ErrorAction SilentlyContinue).Source", text)


@unittest.skipUnless(os.name == "nt" and shutil.which("pwsh"),
                     "needs a real pwsh host (Windows only)")
class ResolvePowerShellHostBehaviourTest(unittest.TestCase):
    """Behavioural contract: the resolver returns a host that exists, launches, and is
    not pinned to a version-stamped directory."""

    def _resolve(self) -> str:
        proc = subprocess.run(
            ["pwsh", "-NoProfile", "-Command",
             f". '{HELPERS}'; Resolve-FakLoopPowerShellHost"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout.strip().splitlines()[-1].strip()

    def test_resolved_host_exists_and_is_not_version_pinned(self) -> None:
        host = self._resolve()
        self.assertTrue(Path(host).exists(), f"resolver returned a missing path: {host}")
        self.assertIsNone(re.search(r"Microsoft\.PowerShell_[\d.]+", host),
                          f"resolver froze a version-stamped path that a PowerShell "
                          f"update will invalidate: {host}")

    def test_resolved_host_actually_runs(self) -> None:
        host = self._resolve()
        proc = subprocess.run([host, "-NoProfile", "-Command", "'ok'"],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ok", proc.stdout)


if __name__ == "__main__":
    unittest.main()
