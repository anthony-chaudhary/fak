#!/usr/bin/env python3
"""Hermetic tests for tools/issue_gardener_worker.py — the tier-2 launcher."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "issue_gardener_worker.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("issue_gardener_worker", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ModelResolutionTest(unittest.TestCase):
    def test_default_is_tier2_sonnet(self) -> None:
        mod = load()
        self.assertEqual(mod.DEFAULT_MODEL, "sonnet")
        self.assertEqual(mod.resolve_model(None, None, {}), "sonnet")
        self.assertEqual(mod.model_tier("sonnet"), "2")

    def test_precedence_flag_beats_tier_beats_env_beats_default(self) -> None:
        mod = load()
        env = {"FLEET_GARDENER_MODEL": "haiku"}
        self.assertEqual(mod.resolve_model("opus", "3", env), "opus")   # flag wins
        self.assertEqual(mod.resolve_model(None, "1", env), "opus")     # tier beats env
        self.assertEqual(mod.resolve_model(None, None, env), "haiku")   # env beats default
        self.assertEqual(mod.resolve_model(None, None, {}), "sonnet")   # default

    def test_tier_maps_to_models(self) -> None:
        mod = load()
        self.assertEqual(mod.resolve_model(None, "1", {}), "opus")
        self.assertEqual(mod.resolve_model(None, "2", {}), "sonnet")
        self.assertEqual(mod.resolve_model(None, "3", {}), "haiku")

    def test_unknown_tier_raises(self) -> None:
        mod = load()
        with self.assertRaises(ValueError):
            mod.resolve_model(None, "9", {})


class RouteBridgeTest(unittest.TestCase):
    """The opt-in `fak route` rung: first deployed consumer of the routing spine.

    Every test injects a stub route_runner so nothing shells out — the same
    hermetic pattern the live-launch test uses.
    """

    @staticmethod
    def _runner_returns(primary: str | None, returncode: int = 0):
        import json as _json

        def runner(_cmd):
            stdout = _json.dumps({"primary": primary}) if primary is not None else ""
            return {"returncode": returncode, "stdout": stdout}

        return runner

    def test_route_off_by_default_keeps_sonnet(self) -> None:
        mod = load()
        # No FLEET_GARDENER_ROUTE => the route rung never fires; default holds.
        called = []
        self.assertEqual(
            mod.resolve_model(None, None, {}, route_runner=lambda c: called.append(c) or {}),
            "sonnet",
        )
        self.assertEqual(called, [])  # runner never invoked when routing is off

    def test_route_on_maps_small_to_haiku(self) -> None:
        mod = load()
        env = {"FLEET_GARDENER_ROUTE": "1"}
        self.assertEqual(
            mod.resolve_model(None, None, env, route_runner=self._runner_returns("small")),
            "haiku",
        )

    def test_route_on_maps_large_to_opus(self) -> None:
        mod = load()
        env = {"FLEET_GARDENER_ROUTE": "1"}
        self.assertEqual(
            mod.resolve_model(None, None, env, route_runner=self._runner_returns("large")),
            "opus",
        )

    def test_route_failure_falls_back_to_sonnet(self) -> None:
        mod = load()
        env = {"FLEET_GARDENER_ROUTE": "1"}
        # nonzero exit (e.g. no fak binary) => fail-safe to the default ladder
        self.assertEqual(
            mod.resolve_model(None, None, env, route_runner=self._runner_returns("small", returncode=127)),
            "sonnet",
        )

    def test_route_unknown_id_falls_back_to_sonnet(self) -> None:
        mod = load()
        env = {"FLEET_GARDENER_ROUTE": "1"}
        self.assertEqual(
            mod.resolve_model(None, None, env, route_runner=self._runner_returns("frontier-x")),
            "sonnet",
        )

    def test_explicit_and_tier_still_beat_route(self) -> None:
        mod = load()
        env = {"FLEET_GARDENER_ROUTE": "1"}
        r = self._runner_returns("small")  # would say haiku if consulted
        self.assertEqual(mod.resolve_model("opus", None, env, route_runner=r), "opus")
        self.assertEqual(mod.resolve_model(None, "1", env, route_runner=r), "opus")
        # env model also beats the route rung
        env2 = {"FLEET_GARDENER_ROUTE": "1", "FLEET_GARDENER_MODEL": "sonnet"}
        self.assertEqual(mod.resolve_model(None, None, env2, route_runner=r), "sonnet")

    def test_route_model_returns_none_when_disabled(self) -> None:
        mod = load()
        self.assertIsNone(mod.route_model({}, runner=self._runner_returns("small")))


class CommandShapeTest(unittest.TestCase):
    def test_command_is_claude_print_with_model_and_plan_mode(self) -> None:
        mod = load()
        cmd = mod.build_command("sonnet", as_of="2026-06-20")
        self.assertEqual(cmd[:6], ["claude", "-p", "--model", "sonnet", "--permission-mode", "plan"])
        self.assertEqual(len(cmd), 7)  # + the prompt
        self.assertIn("Garden the open GitHub issue backlog", cmd[6])

    def test_propose_only_prompt_forbids_writes_by_default(self) -> None:
        mod = load()
        prompt = mod.build_prompt(as_of="2026-06-20", scope=None, apply_mechanical=False)
        self.assertIn("Do NOT edit, label, comment on, or close", prompt)
        self.assertNotIn("mark-stale / close-dormant-question", prompt)

    def test_apply_mechanical_prompt_allows_the_batch(self) -> None:
        mod = load()
        prompt = mod.build_prompt(as_of="2026-06-20", scope=None, apply_mechanical=True)
        self.assertIn("mark-stale", prompt)

    def test_prompt_is_ascii(self) -> None:
        # argv passed to claude on Windows; keep it ASCII to avoid mangling.
        mod = load()
        prompt = mod.build_prompt(as_of="2026-06-20", scope="stale", apply_mechanical=False)
        prompt.encode("ascii")  # raises if non-ascii

    def test_build_command_rejects_empty_model(self) -> None:
        mod = load()
        with self.assertRaises(ValueError):
            mod.build_command("", as_of="2026-06-20")

    def test_unknown_backend_rejected(self) -> None:
        mod = load()
        with self.assertRaises(ValueError):
            mod.build_command("sonnet", as_of="2026-06-20", backend="opencode")


class PayloadAndLaunchTest(unittest.TestCase):
    def test_child_env_stamps_model_and_tier(self) -> None:
        mod = load()
        env = mod.child_env("sonnet", Path("C:/work/fleet"), base={"KEEP": "1"})
        self.assertEqual(env["GARDENER_MODEL"], "sonnet")
        self.assertEqual(env["GARDENER_TIER"], "2")
        self.assertEqual(env["DISPATCH_WORKSPACE"], str(Path("C:/work/fleet")))
        self.assertEqual(env["KEEP"], "1")

    def test_dry_run_default_does_not_launch(self) -> None:
        mod = load()
        rc = mod.main(["--json"])  # no --live
        self.assertEqual(rc, 0)
        payload = mod.build_payload(
            model="sonnet", backend="claude", workspace=Path("."), dry_run=True,
            as_of="2026-06-20", scope=None, apply_mechanical=False, permission_mode="plan",
        )
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["dry_run"])
        self.assertTrue(payload["propose_only"])
        self.assertEqual(payload["model"], "sonnet")
        self.assertEqual(payload["tier"], "2")

    def test_live_launch_calls_runner_with_stamped_env(self) -> None:
        mod = load()
        seen: list[tuple] = []

        def runner(cmd, cwd, env):
            seen.append((list(cmd), cwd, env))
            return {"returncode": 0}

        command = mod.build_command("sonnet", as_of="2026-06-20")
        env = mod.child_env("sonnet", Path("C:/work/fleet"), base={})
        result = mod.launch(command, Path("C:/work/fleet"), env, runner=runner)
        self.assertEqual(result["returncode"], 0)
        ran_cmd, _, ran_env = seen[0]
        self.assertEqual(ran_cmd[0], "claude")
        self.assertEqual(ran_env["GARDENER_MODEL"], "sonnet")

    def test_nonzero_returncode_propagates_to_payload_ok_false(self) -> None:
        mod = load()
        result = {"returncode": 1}
        payload = mod.build_payload(
            model="sonnet", backend="claude", workspace=Path("."), dry_run=False,
            as_of="2026-06-20", scope=None, apply_mechanical=False,
            permission_mode="plan", result=result,
        )
        self.assertFalse(payload["ok"])

    def test_normalize_timeout_caps_by_default_and_opts_out_at_zero(self) -> None:
        mod = load()
        # The default cap bounds the unattended cadence gardener; 0/negative/None opt out.
        self.assertEqual(mod.normalize_timeout(mod.DEFAULT_TIMEOUT_S), mod.DEFAULT_TIMEOUT_S)
        self.assertEqual(mod.normalize_timeout(90), 90)
        self.assertIsNone(mod.normalize_timeout(0))
        self.assertIsNone(mod.normalize_timeout(-1))
        self.assertIsNone(mod.normalize_timeout(None))
        self.assertIsNotNone(mod.DEFAULT_TIMEOUT_S)
        self.assertGreater(mod.DEFAULT_TIMEOUT_S, 0)


if __name__ == "__main__":
    unittest.main()
