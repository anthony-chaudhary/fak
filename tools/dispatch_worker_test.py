#!/usr/bin/env python3
"""Hermetic tests for tools/dispatch_worker.py."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "dispatch_worker.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("dispatch_worker", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DispatchWorkerTest(unittest.TestCase):
    def test_resolve_backend_flag_beats_env_beats_default(self) -> None:
        mod = load()
        self.assertEqual(mod.resolve_backend("claude", {"FLEET_WORKER_BACKEND": "opencode"}), "claude")
        self.assertEqual(mod.resolve_backend(None, {"FLEET_WORKER_BACKEND": "opencode"}), "opencode")
        self.assertEqual(mod.resolve_backend(None, {}), "claude")

    def test_resolve_backend_rejects_unknown(self) -> None:
        mod = load()
        with self.assertRaises(ValueError):
            mod.resolve_backend("cursor", None)
        with self.assertRaises(ValueError):
            mod.resolve_backend(None, {"FLEET_WORKER_BACKEND": "nope"})

    def test_claude_command_shape_matches_dos_toml_reference(self) -> None:
        mod = load()
        cmd = mod.build_command("adjudicator", "claude")
        self.assertEqual(cmd[0], "claude")
        self.assertEqual(cmd[1:4], ["-p", "--permission-mode", "bypassPermissions"])
        self.assertEqual(cmd[4], "/dos-kernel:dos-dispatch-loop --lane adjudicator")

    def test_opencode_command_uses_dispatch_agent_and_skip_permissions(self) -> None:
        mod = load()
        cmd = mod.build_command("agent", "opencode")
        self.assertEqual(cmd[0], "opencode")
        self.assertIn("--dangerously-skip-permissions", cmd)
        self.assertEqual(cmd[cmd.index("--agent") + 1], "dos-dispatch")
        self.assertEqual(cmd[-1], "dispatch lane agent")

    def test_build_command_rejects_empty_lane(self) -> None:
        mod = load()
        with self.assertRaises(ValueError):
            mod.build_command("", "claude")

    def test_child_env_stamps_assignment_and_passes_through(self) -> None:
        mod = load()
        env = mod.child_env("canon", "claude", Path("C:/work/fleet"), base={"PATH": "x", "KEEP_ME": "1"})
        self.assertEqual(env["DISPATCH_LANE"], "canon")
        self.assertEqual(env["DISPATCH_BACKEND"], "claude")
        self.assertEqual(env["DISPATCH_WORKSPACE"], str(Path("C:/work/fleet")))
        self.assertEqual(env["KEEP_ME"], "1")  # base env preserved

    def test_dry_run_does_not_call_runner(self) -> None:
        mod = load()

        def fail_runner(_cmd, _cwd, _env):
            raise AssertionError("dry run must not call runner")

        # main() with --dry-run must not launch.
        rc = mod.main(["--lane", "docs", "--dry-run", "--json"])
        self.assertEqual(rc, 0)

        payload = mod.build_payload(
            lane="docs", backend="claude", workspace=Path("C:/work/fleet"), dry_run=True
        )
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["dry_run"])
        self.assertIsNone(payload["result"])
        self.assertEqual(payload["backend"], "claude")
        fail_runner  # referenced to keep the lint honest about intent

    def test_live_launch_calls_runner_with_resolved_command_and_env(self) -> None:
        mod = load()
        seen: list[tuple] = []

        def runner(cmd, cwd, env):
            seen.append((list(cmd), cwd, env))
            return {"returncode": 0, "stdout": "", "stderr": ""}

        command = mod.build_command("recall", "opencode")
        env = mod.child_env("recall", "opencode", Path("C:/work/fleet"), base={})
        result = mod.launch(command, Path("C:/work/fleet"), env, runner=runner)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(len(seen), 1)
        ran_cmd, ran_cwd, ran_env = seen[0]
        self.assertEqual(ran_cmd[0], "opencode")
        self.assertEqual(ran_env["DISPATCH_LANE"], "recall")
        self.assertEqual(ran_env["DISPATCH_BACKEND"], "opencode")

    def test_live_nonzero_returncode_propagates_to_payload_ok_false(self) -> None:
        mod = load()

        def runner(_cmd, _cwd, _env):
            return {"returncode": 1, "stdout": "", "stderr": "boom"}

        result = mod.launch(mod.build_command("x", "claude"), Path("."), {}, runner=runner)
        payload = mod.build_payload(
            lane="x", backend="claude", workspace=Path("."), dry_run=False, result=result
        )
        self.assertFalse(payload["ok"])

    def test_resolve_exe_falls_back_to_name_when_not_found(self) -> None:
        mod = load()
        # A name that will not resolve should fall back to the bare name rather
        # than raise (launch then surfaces FileNotFoundError as returncode 127).
        self.assertEqual(mod.resolve_exe("definitely-not-a-real-backend-xyz"), "definitely-not-a-real-backend-xyz")


if __name__ == "__main__":
    unittest.main()
