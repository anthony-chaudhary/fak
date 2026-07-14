#!/usr/bin/env python3
"""Regression tests for tools/dos_hook.py -- the versioned dos-hook launcher (#2705).

The load-bearing witness (issue #2705 acceptance):
  * ``test_argparse_error_does_not_block`` -- a **simulated dos argparse error**
    (the Python fallback exits code 2, Claude Code's *block* code, with no deny on
    stdout) must NOT block the tool call: ``main`` exits **0** and emits no deny. This
    is the version-drift hard-block the issue exists to kill.

Plus the surrounding contract:
  * a real deny (``permissionDecision: deny`` JSON on stdout) is PRESERVED whether the
    native binary or the Python fallback emits it (exit coerced to 0, deny unaffected);
  * native OWNED (rc == 0) short-circuits -- the Python path is not consulted;
  * native DELEGATE (rc != 0) / **binary absent** falls back to Python, with the SAME
    buffered stdin (a native->python delegation re-decides on the real payload);
  * the launcher never raises -- an internal error still exits 0 (fail-safe).
"""
from __future__ import annotations

import unittest
from pathlib import Path

import dos_hook


class FakeRunner:
    """Records every (argv, stdin) it is handed and replies from a scripted queue.

    Each scripted reply is ``(returncode, stdout_bytes)``. Calls beyond the script
    reuse the last reply (so a test can script just the branch it cares about)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []  # list of (argv, stdin_bytes)

    def __call__(self, argv, stdin_data):
        self.calls.append((list(argv), stdin_data))
        idx = min(len(self.calls) - 1, len(self.replies) - 1)
        return self.replies[idx]


DENY_JSON = b'{"hookSpecificOutput":{"permissionDecision":"deny"}}'
PAYLOAD = b'{"tool_name":"Bash","tool_input":{"command":"rm -rf /"}}'


class RunHookContract(unittest.TestCase):
    def test_argparse_error_does_not_block(self):
        # THE witness: native absent, Python fallback simulates an argparse error
        # (exit code 2, no stdout). The tool call must NOT be blocked.
        runner = FakeRunner([(2, b"")])
        out = dos_hook.run_hook("pretool", ".", PAYLOAD, native=None, run=runner)
        self.assertEqual(out, b"")  # no deny emitted -> call proceeds
        # and end-to-end the launcher process exits 0 despite the code-2 fallback.
        rc = dos_hook.main(
            argv=["pretool", "--workspace", "."],
            stdin_bytes=PAYLOAD,
            run=FakeRunner([(2, b"")]),
            env={"CLAUDE_PROJECT_DIR": "/no/such/root"},  # forces binary-absent
        )
        self.assertEqual(rc, 0)

    def test_real_deny_from_python_is_preserved(self):
        # A genuine deny on stdout survives even though the fallback also exits non-zero.
        runner = FakeRunner([(2, DENY_JSON)])
        out = dos_hook.run_hook("pretool", ".", PAYLOAD, native=None, run=runner)
        self.assertEqual(out, DENY_JSON)

    def test_native_owned_short_circuits(self):
        # rc == 0 = OWNED: forward native stdout, do NOT consult Python.
        native = Path("tools/.bin/dos-hook")
        runner = FakeRunner([(0, DENY_JSON)])
        out = dos_hook.run_hook("pretool", ".", PAYLOAD, native=native, run=runner)
        self.assertEqual(out, DENY_JSON)
        self.assertEqual(len(runner.calls), 1)  # native only
        self.assertIn("dos-hook", runner.calls[0][0][0])

    def test_native_delegate_falls_back_to_python_same_stdin(self):
        # rc != 0 = DELEGATE: fall to Python, re-deciding on the SAME buffered stdin.
        native = Path("tools/.bin/dos-hook")
        runner = FakeRunner([(3, b""), (0, DENY_JSON)])
        out = dos_hook.run_hook("posttool", ".", PAYLOAD, native=native, run=runner)
        self.assertEqual(out, DENY_JSON)
        self.assertEqual(len(runner.calls), 2)  # native then python
        # both backends saw the identical, undrained payload
        self.assertEqual(runner.calls[0][1], PAYLOAD)
        self.assertEqual(runner.calls[1][1], PAYLOAD)
        # second call is the python -m dos.cli fallback
        self.assertIn("dos.cli", " ".join(runner.calls[1][0]))

    def test_binary_absent_uses_python(self):
        runner = FakeRunner([(0, b"")])
        dos_hook.run_hook("pretool", ".", PAYLOAD, native=None, run=runner)
        self.assertEqual(len(runner.calls), 1)
        self.assertIn("dos.cli", " ".join(runner.calls[0][0]))


class LauncherFailSafe(unittest.TestCase):
    def test_main_swallows_runner_exception(self):
        def boom(argv, stdin_data):
            raise RuntimeError("backend blew up")

        rc = dos_hook.main(
            argv=["pretool", "--workspace", "."],
            stdin_bytes=PAYLOAD,
            run=boom,
            env={"CLAUDE_PROJECT_DIR": "/no/such/root"},
        )
        self.assertEqual(rc, 0)  # a launcher/backend error must never wedge the fleet

    def test_native_binary_resolution_prefers_project_dir(self):
        # Absent binary under a bogus root resolves to None (Python fallback).
        self.assertIsNone(dos_hook.native_binary(Path("/no/such/root")))

    def test_repo_root_prefers_claude_project_dir(self):
        root = dos_hook.repo_root(env={"CLAUDE_PROJECT_DIR": "/some/ws"})
        self.assertEqual(str(root).replace("\\", "/"), "/some/ws")


if __name__ == "__main__":
    unittest.main()
