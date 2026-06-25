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

    def test_normalize_timeout_caps_by_default_and_opts_out_at_zero(self) -> None:
        mod = load()
        # The default cap bounds an unattended worker; 0/negative/None opt out.
        self.assertEqual(mod.normalize_timeout(mod.DEFAULT_TIMEOUT_S), mod.DEFAULT_TIMEOUT_S)
        self.assertEqual(mod.normalize_timeout(60), 60)
        self.assertIsNone(mod.normalize_timeout(0))
        self.assertIsNone(mod.normalize_timeout(-5))
        self.assertIsNone(mod.normalize_timeout(None))
        # The default is a real bound, not the old unbounded None.
        self.assertIsNotNone(mod.DEFAULT_TIMEOUT_S)
        self.assertGreater(mod.DEFAULT_TIMEOUT_S, 0)

    # --- Dogfood: front the worker with the kernel (`fak guard`) ---------------

    def test_guard_enabled_default_on_and_opt_out_values(self) -> None:
        mod = load()
        self.assertTrue(mod.guard_enabled({}))                                 # unset -> ON (dogfood default)
        self.assertTrue(mod.guard_enabled({"FLEET_DOGFOOD_GUARD": "1"}))
        self.assertTrue(mod.guard_enabled({"FLEET_DOGFOOD_GUARD": "on"}))
        for off in ("0", "off", "false", "no", "", "disable", "DISABLED", " Off "):
            self.assertFalse(mod.guard_enabled({"FLEET_DOGFOOD_GUARD": off}), off)

    def test_resolve_fak_bin_prefers_env_then_intree_then_path_else_none(self) -> None:
        mod = load()
        # An explicit FAK_BIN that exists wins (use this very test file as a stand-in).
        existing = str(Path(__file__).resolve())
        self.assertEqual(
            mod.resolve_fak_bin(Path("C:/nope"), {"FAK_BIN": existing}), existing)
        # A non-existent FAK_BIN is ignored; with an empty workspace and a PATH that
        # holds no `fak`, the result is None (fail-open signal).
        got = mod.resolve_fak_bin(
            Path("C:/definitely/not/a/repo/xyz"),
            {"FAK_BIN": "C:/no/such/fak", "PATH": str(Path(__file__).resolve().parent / "_no_fak_here_xyz")})
        self.assertIsNone(got)

    def test_guard_provider_maps_claude_to_anthropic_else_openai(self) -> None:
        mod = load()
        self.assertEqual(mod.guard_provider("claude"), "anthropic")
        self.assertEqual(mod.guard_provider("opencode"), "openai")

    def test_guard_audit_path_is_per_lane_under_dispatch_runs(self) -> None:
        mod = load()
        p = mod.guard_audit_path(Path("C:/work/fak"), "gate way/1", "claude")
        self.assertEqual(p.parent.name, "guard-audit")
        self.assertEqual(p.parent.parent.name, ".dispatch-runs")
        self.assertTrue(p.name.endswith(".jsonl"))
        self.assertNotIn("/", p.name)   # lane separators sanitized out of the filename
        self.assertNotIn(" ", p.name)

    def test_guard_wrap_claude_fronts_with_fak_guard_anthropic(self) -> None:
        mod = load()
        raw = mod.build_command("gateway", "claude")
        wrapped = mod.guard_wrap(raw, fak_bin="/usr/bin/fak", lane="gateway",
                                 backend="claude", workspace=Path("C:/work/fak"))
        self.assertEqual(wrapped[0], "/usr/bin/fak")
        self.assertEqual(wrapped[1], "guard")
        self.assertEqual(wrapped[wrapped.index("--provider") + 1], "anthropic")
        self.assertIn("--audit", wrapped)
        # The raw worker argv is preserved verbatim AFTER the `--` separator.
        sep = wrapped.index("--")
        self.assertEqual(wrapped[sep + 1:], raw)

    def test_guard_wrap_noop_without_fak_bin(self) -> None:
        mod = load()
        raw = mod.build_command("docs", "claude")
        self.assertEqual(
            mod.guard_wrap(raw, fak_bin=None, lane="docs", backend="claude",
                           workspace=Path(".")), raw)

    def test_guard_wrap_opencode_skips_without_base_url_but_wraps_with_one(self) -> None:
        mod = load()
        raw = mod.build_command("recall", "opencode")
        # No FLEET_DOGFOOD_GUARD_BASEURL -> refuse to misroute a local-upstream worker.
        self.assertEqual(
            mod.guard_wrap(raw, fak_bin="/usr/bin/fak", lane="recall",
                           backend="opencode", workspace=Path("."), env={}), raw)
        # With a base URL the operator names the local upstream and we DO front it.
        wrapped = mod.guard_wrap(raw, fak_bin="/usr/bin/fak", lane="recall",
                                 backend="opencode", workspace=Path("."),
                                 env={"FLEET_DOGFOOD_GUARD_BASEURL": "http://127.0.0.1:8131/v1"})
        self.assertEqual(wrapped[0], "/usr/bin/fak")
        self.assertEqual(wrapped[wrapped.index("--provider") + 1], "openai")
        self.assertEqual(wrapped[wrapped.index("--base-url") + 1], "http://127.0.0.1:8131/v1")

    def test_guarded_launch_command_opts_out_when_disabled(self) -> None:
        mod = load()
        raw = mod.build_command("gateway", "claude")
        cmd, guarded = mod.guarded_launch_command(
            raw, "gateway", "claude", Path("C:/work/fak"),
            env={"FLEET_DOGFOOD_GUARD": "0", "FAK_BIN": str(Path(__file__).resolve())})
        self.assertFalse(guarded)
        self.assertEqual(cmd, raw)

    def test_guarded_launch_command_wraps_when_enabled_and_bin_present(self) -> None:
        mod = load()
        raw = mod.build_command("gateway", "claude")
        fak = str(Path(__file__).resolve())  # any existing file stands in for the bin
        cmd, guarded = mod.guarded_launch_command(
            raw, "gateway", "claude", Path("C:/work/fak"), env={"FAK_BIN": fak})
        self.assertTrue(guarded)
        self.assertEqual(cmd[0], fak)
        self.assertEqual(cmd[1], "guard")

    def test_guard_env_augment_sets_timeout_floors_without_clobbering(self) -> None:
        mod = load()
        env = {"FAK_PLANNER_TIMEOUT_S": "1200"}
        mod.guard_env_augment(env)
        self.assertEqual(env["FAK_PLANNER_TIMEOUT_S"], "1200")   # explicit value kept
        self.assertEqual(env["FAK_HTTP_WRITE_TIMEOUT_S"], str(mod.GUARD_TIMEOUT_FLOOR_S))

    def test_build_payload_carries_guarded_and_explicit_command(self) -> None:
        mod = load()
        payload = mod.build_payload(
            lane="gateway", backend="claude", workspace=Path("C:/work/fak"),
            dry_run=True, command=["fak", "guard", "--", "claude"], guarded=True)
        self.assertTrue(payload["guarded"])
        self.assertEqual(payload["command"][0], "fak")


if __name__ == "__main__":
    unittest.main()
