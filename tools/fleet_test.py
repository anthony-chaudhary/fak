#!/usr/bin/env python3
"""Hermetic tests for the `fleet` operator console router.

The router is pure (argv -> intent), so these pin every verb's mapping and the
fixed-arg prefixes without spawning a tool or touching disk."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet  # noqa: E402


class RouteTest(unittest.TestCase):
    def test_bare_is_live_top(self):
        self.assertEqual(fleet.route([]), {"kind": "exec", "tool": "fleet_top.py", "argv": []})

    def test_status_maps_to_once_and_passes_through(self):
        self.assertEqual(
            fleet.route(["status", "--window", "24"]),
            {"kind": "exec", "tool": "fleet_top.py", "argv": ["--once", "--window", "24"]},
        )

    def test_json_and_top_and_pane(self):
        self.assertEqual(fleet.route(["json"])["argv"], ["--json"])
        self.assertEqual(fleet.route(["top", "--interval", "2"])["argv"], ["--interval", "2"])
        self.assertEqual(fleet.route(["pane"]), {"kind": "exec", "tool": "fleet_control_pane.py", "argv": ["status"]})

    def test_sessions_and_resume_and_accounts(self):
        self.assertEqual(fleet.route(["sessions"]), {"kind": "exec", "tool": "fleet_sessions.py", "argv": ["summary"]})
        self.assertEqual(fleet.route(["resume"])["argv"], ["resume"])
        self.assertEqual(fleet.route(["accounts"]), {"kind": "exec", "tool": "fleet_accounts.py", "argv": []})

    def test_bare_flag_is_shorthand_for_top(self):
        self.assertEqual(
            fleet.route(["--once"]),
            {"kind": "exec", "tool": "fleet_top.py", "argv": ["--once"]},
        )

    def test_install_uninstall_help(self):
        self.assertEqual(fleet.route(["install", "--system"]), {"kind": "install", "argv": ["--system"]})
        self.assertEqual(fleet.route(["uninstall"]), {"kind": "uninstall", "argv": []})
        self.assertEqual(fleet.route(["help"]), {"kind": "help"})
        self.assertEqual(fleet.route(["-h"]), {"kind": "help"})

    def test_unknown_verb(self):
        self.assertEqual(fleet.route(["frobnicate"]), {"kind": "unknown", "verb": "frobnicate"})


class ExecCommandTest(unittest.TestCase):
    def test_exec_command_targets_the_repo_tool(self):
        root = Path("C:/work/fak")
        cmd = fleet.exec_command(root, "fleet_top.py", ["--once"])
        self.assertEqual(cmd[0], sys.executable)
        self.assertTrue(cmd[1].replace("\\", "/").endswith("C:/work/fak/tools/fleet_top.py"))
        self.assertEqual(cmd[2:], ["--once"])


class RepoRootTest(unittest.TestCase):
    def test_fleet_root_env_wins(self):
        import os
        prev = os.environ.get("FLEET_ROOT")
        os.environ["FLEET_ROOT"] = str(Path.home())
        try:
            self.assertEqual(fleet.repo_root(), Path.home().resolve())
        finally:
            if prev is None:
                del os.environ["FLEET_ROOT"]
            else:
                os.environ["FLEET_ROOT"] = prev


if __name__ == "__main__":
    unittest.main()
