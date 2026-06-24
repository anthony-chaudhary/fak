#!/usr/bin/env python3
"""Hermetic test for examples/adjudication-demo/demo.py display faithfulness (issue #366).

The demo advertises that it prints the EXACT command the kernel allowed so an
operator can review every effect. The old `command[:56]` slice chopped mid-path and
hid the `&&` chain behind an innocent-looking `mkdir` prefix. These tests pin the
fixed `display_command`: the sandbox path collapses to {sandbox} (noise gone), the
multi-effect chain stays visible, and any cap is an EXPLICIT ellipsis. Pure stdlib;
importing demo.py runs nothing (it is `__main__`-guarded).
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "demo.py"


def load():
    spec = importlib.util.spec_from_file_location("adj_demo", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()
SANDBOX = "/var/folders/z2/qgyzc8gx6kn1mkh6kfhlb6qw0000gp/T/fak-demo-abc123"
CHAIN = (f"mkdir -p {SANDBOX} && printf hello > {SANDBOX}/a.txt"
         f" && printf ' world' > {SANDBOX}/b.txt")


class DisplayCommandTest(unittest.TestCase):
    def test_multi_effect_chain_stays_visible(self):
        shown = m.display_command(CHAIN, SANDBOX)
        # The bug: an operator saw only 'mkdir -p …/T/' and missed the writes.
        self.assertEqual(shown.count("&&"), 2)        # both chain links visible
        self.assertIn("printf hello", shown)
        self.assertIn("printf ' world'", shown)

    def test_sandbox_path_collapsed_to_placeholder(self):
        shown = m.display_command(CHAIN, SANDBOX)
        self.assertNotIn(SANDBOX, shown)              # noisy temp path gone
        self.assertIn("{sandbox}", shown)

    def test_no_silent_chop_short_command(self):
        shown = m.display_command(f"ls {SANDBOX}", SANDBOX)
        self.assertEqual(shown, "ls {sandbox}")
        self.assertNotIn("…", shown)

    def test_cap_is_explicit_ellipsis(self):
        long_cmd = "echo " + "x" * (m.MAX_CMD_DISPLAY + 50)
        shown = m.display_command(long_cmd, SANDBOX)
        self.assertTrue(shown.endswith("…"))          # truncation never silent
        self.assertLessEqual(len(shown), m.MAX_CMD_DISPLAY)

    def test_empty_command(self):
        self.assertEqual(m.display_command(None, SANDBOX), "")
        self.assertEqual(m.display_command("", SANDBOX), "")


if __name__ == "__main__":
    unittest.main()
