#!/usr/bin/env python3
"""Hermetic tests for claude_agent_chat.py."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import claude_agent_chat as chat  # noqa: E402


def make_account(root: Path, name: str, projects: bool = True) -> None:
    path = root / name
    path.mkdir()
    if projects:
        (path / "projects").mkdir()


def base_args(**overrides):
    data = {
        "account": "auto",
        "allow_blocked": False,
        "tier": "t1",
        "work_kind": "",
        "allow_tier_fallback": False,
        "cwd": str(Path.cwd()),
        "claude_exe": "claude",
        "name": "",
        "goal": "",
        "prompt": "",
        "print": False,
        "output_format": "",
        "permission_mode": "auto",
        "dangerously_skip_permissions": False,
        "model": "",
        "effort": "",
        "settings": "",
        "mcp_config": [],
        "add_dir": [],
        "no_agent": False,
        "agent_name": chat.DEFAULT_AGENT_NAME,
        "agent_description": chat.DEFAULT_AGENT_DESCRIPTION,
        "agent_prompt": "",
        "agent_prompt_file": "",
        "json": False,
        # GLM / Z.ai mode (all default off/empty so the native path is unchanged).
        "glm": False,
        "glm_base_url": "",
        "glm_model": "",
        "glm_api_key_env": "",
        "glm_provider": "",
        "fak_bin": "",
        "gateway_port": chat.DEFAULT_GATEWAY_PORT,
        "gateway_policy": chat.DEFAULT_GATEWAY_POLICY,
        "require_key_env": "",
    }
    data.update(overrides)
    return Namespace(**data)


class ClaudeAgentChatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        make_account(self.home, ".claude")
        make_account(self.home, ".claude-gem8-netra")
        make_account(self.home, ".claude-backup")

    def rows(self, registry=None):
        reg_path = self.home / "sessions.json"
        if registry is None:
            registry = {"generated_utc": "2026-06-18T12:00:00+00:00", "sessions": []}
        if registry is not None:
            reg_path.write_text(json.dumps(registry), encoding="utf-8")
            return chat.load_roster(str(self.home), str(reg_path))

    def test_auto_selects_available_least_busy_worker(self) -> None:
        registry = {
            "generated_utc": "2026-06-18T12:00:00+00:00",
            "throttle": {".claude-gem8-netra": {"reset": "tomorrow"}},
            "sessions": [
                {"account": ".claude", "disp": "LIVE"},
                {"account": ".claude-gem8-netra", "disp": "STOPPED_LIMIT"},
            ],
        }

        chosen = chat.choose_account(self.rows(registry), "auto", product="claude")

        self.assertEqual(chosen["account"], ".claude")
        self.assertTrue(chosen["available"])

    def test_blocked_explicit_account_refuses_unless_allowed(self) -> None:
        registry = {
            "generated_utc": "2026-06-18T12:00:00+00:00",
            "throttle": {".claude-gem8-netra": {"reset": "tomorrow"}},
            "sessions": [],
        }
        rows = self.rows(registry)

        with self.assertRaises(chat.SelectionError):
            chat.choose_account(rows, "gem8")

        chosen = chat.choose_account(rows, "gem8", allow_blocked=True)
        self.assertEqual(chosen["account"], ".claude-gem8-netra")

    def test_excluded_account_is_not_offerable(self) -> None:
        with self.assertRaises(chat.SelectionError):
            chat.choose_account(self.rows(), "backup")

    def test_build_packet_uses_goal_agent_and_config_dir(self) -> None:
        args = base_args(goal="ship the docs", cwd=str(self.home), print=True, output_format="json")

        packet = chat.build_packet(args, self.rows())

        self.assertEqual(packet["env"]["CLAUDE_CONFIG_DIR"], str(self.home / ".claude"))
        self.assertEqual(packet["prompt"], "/goal ship the docs")
        self.assertEqual(packet["account"]["model_tier"], 1)
        self.assertIn("--model", packet["argv"])
        self.assertIn("opus", packet["argv"])
        self.assertIn("--effort", packet["argv"])
        self.assertIn("xhigh", packet["argv"])
        self.assertIn("--agent", packet["argv"])
        self.assertIn("--agents", packet["argv"])
        self.assertIn("--permission-mode", packet["argv"])
        self.assertIn("auto", packet["argv"])
        self.assertIn("--print", packet["argv"])
        self.assertIn("--output-format", packet["argv"])
        self.assertEqual(packet["agent"]["name"], chat.DEFAULT_AGENT_NAME)

    def test_no_agent_and_skip_permissions_are_reflected_in_argv(self) -> None:
        args = base_args(no_agent=True, dangerously_skip_permissions=True, prompt="hello")

        packet = chat.build_packet(args, self.rows())

        self.assertNotIn("--agent", packet["argv"])
        self.assertNotIn("--agents", packet["argv"])
        self.assertIn("--dangerously-skip-permissions", packet["argv"])
        self.assertEqual(packet["argv"][-1], "hello")

    def test_commands_include_account_env_and_working_directory(self) -> None:
        args = base_args(goal="inspect", cwd=str(self.home))

        packet = chat.build_packet(args, self.rows())

        self.assertIn("CLAUDE_CONFIG_DIR", packet["commands"]["powershell"])
        self.assertIn(str(self.home), packet["commands"]["powershell"])
        self.assertIn("CLAUDE_CONFIG_DIR=", packet["commands"]["posix"])

    def test_explicit_tier1_routes_to_frontier_profile(self) -> None:
        args = base_args(tier="1", prompt="fix a bug")

        packet = chat.build_packet(args, self.rows())

        self.assertEqual(packet["route"]["target_tier"], 1)
        self.assertEqual(packet["route"]["selected_tier"], 1)
        self.assertFalse(packet["route"]["fallback_used"])
        self.assertEqual(packet["account"]["model"], "opus")
        self.assertEqual(packet["account"]["model_effort"], "xhigh")

    def test_work_kind_gardening_overrides_tier1_default(self) -> None:
        # Default tier is t1, but a stated gardening work-kind targets tier 2. With
        # only Claude (tier-1) accounts present it up-shifts (non-strict) rather than
        # stalling -- and the route records the gardening intent (target_tier=2).
        args = base_args(work_kind="gardening", prompt="audit and tidy the docs index")

        packet = chat.build_packet(args, self.rows())

        self.assertEqual(packet["route"]["target_tier"], 2)
        self.assertEqual(packet["route"]["selected_tier"], 1)
        self.assertTrue(packet["route"]["fallback_used"])
        # The account is still pinned (CLAUDE_CONFIG_DIR set) -- dispatch is wired.
        self.assertTrue(packet["env"]["CLAUDE_CONFIG_DIR"])

    def test_work_kind_engineering_pins_tier1(self) -> None:
        # An engineering work-kind on a terse prompt still pins tier 1 (no down-shift).
        args = base_args(work_kind="engineering", prompt="ship it")

        packet = chat.build_packet(args, self.rows())

        self.assertEqual(packet["route"]["target_tier"], 1)
        self.assertEqual(packet["route"]["selected_tier"], 1)
        self.assertFalse(packet["route"]["fallback_used"])
        self.assertEqual(packet["account"]["model"], "opus")

    def test_work_kind_is_parseable_and_defaults_empty(self) -> None:
        parser = chat.build_parser()
        gardening = parser.parse_args(["plan", "--work-kind", "gardening"])
        self.assertEqual(gardening.work_kind, "gardening")
        kind_alias = parser.parse_args(["plan", "--kind", "maintenance"])
        self.assertEqual(kind_alias.work_kind, "maintenance")
        default = parser.parse_args(["plan", "--prompt", "x"])
        self.assertEqual(default.work_kind, "")

    def test_default_tier_is_t1_and_auto_is_opt_in(self) -> None:
        parser = chat.build_parser()

        default_args = parser.parse_args(["plan", "--prompt", "say pong"])
        auto_args = parser.parse_args(["plan", "--tier", "auto", "--prompt", "say pong"])
        packet = chat.build_packet(base_args(prompt="say pong"), self.rows())

        self.assertEqual(default_args.tier, "t1")
        self.assertEqual(auto_args.tier, "auto")
        self.assertEqual(packet["route"]["target_tier"], 1)
        self.assertEqual(packet["route"]["selected_tier"], 1)

    def test_explicit_tier2_is_strict_for_claude_launcher(self) -> None:
        args = base_args(tier="2", prompt="say pong")

        with self.assertRaises(chat.SelectionError):
            chat.build_packet(args, self.rows())

    def test_argparse_accepts_dash_t2_shorthand(self) -> None:
        parser = chat.build_parser()

        args = parser.parse_args(["plan", "-t2", "--prompt", "say pong"])

        self.assertEqual(args.tier, "2")


class GlmBackendResolveTest(unittest.TestCase):
    """resolve_glm_backend: flag > env > built-in Z.ai default, per field."""

    def test_defaults_are_zai_coding_plan(self) -> None:
        backend = chat.resolve_glm_backend(base_args(glm=True), env={})

        self.assertEqual(backend["base_url"], chat.GLM_DEFAULT_BASE_URL)
        self.assertEqual(backend["model"], chat.GLM_DEFAULT_MODEL)
        self.assertEqual(backend["api_key_env"], chat.GLM_DEFAULT_API_KEY_ENV)
        self.assertEqual(backend["provider"], chat.GLM_DEFAULT_PROVIDER)

    def test_env_overrides_default(self) -> None:
        env = {"FAK_GLM_BASE_URL": "http://dgx:8000/v1", "FAK_GLM_MODEL": "glm-5.2"}

        backend = chat.resolve_glm_backend(base_args(glm=True), env=env)

        self.assertEqual(backend["base_url"], "http://dgx:8000/v1")
        self.assertEqual(backend["model"], "glm-5.2")
        # Untouched fields still fall back to the built-in default.
        self.assertEqual(backend["api_key_env"], chat.GLM_DEFAULT_API_KEY_ENV)

    def test_flag_overrides_env_and_default(self) -> None:
        args = base_args(glm=True, glm_base_url="http://local:9000/v1",
                         glm_model="my-glm", glm_api_key_env="MY_KEY", glm_provider="openai")
        env = {"FAK_GLM_BASE_URL": "http://ignored", "FAK_GLM_MODEL": "ignored"}

        backend = chat.resolve_glm_backend(args, env=env)

        self.assertEqual(backend["base_url"], "http://local:9000/v1")
        self.assertEqual(backend["model"], "my-glm")
        self.assertEqual(backend["api_key_env"], "MY_KEY")

    def test_hosted_default_detection(self) -> None:
        self.assertTrue(chat.glm_backend_is_hosted_default(
            chat.resolve_glm_backend(base_args(glm=True), env={})))
        self.assertFalse(chat.glm_backend_is_hosted_default(
            chat.resolve_glm_backend(base_args(glm=True, glm_base_url="http://dgx:8000/v1"), env={})))


class GlmServeArgvTest(unittest.TestCase):
    """build_serve_argv: the fak serve command that fronts the GLM backend."""

    def test_hosted_serve_argv_has_all_flags(self) -> None:
        backend = chat.resolve_glm_backend(base_args(glm=True), env={})

        argv = chat.build_serve_argv(backend, fak_bin="fak", port=8080,
                                     policy="p.json", require_key_env="")

        self.assertEqual(argv[:2], ["fak", "serve"])
        for flag, value in (("--addr", "127.0.0.1:8080"),
                            ("--provider", "openai"),
                            ("--base-url", chat.GLM_DEFAULT_BASE_URL),
                            ("--model", chat.GLM_DEFAULT_MODEL),
                            ("--api-key-env", "ZAI_API_KEY"),
                            ("--policy", "p.json")):
            self.assertIn(flag, argv)
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertNotIn("--require-key-env", argv)

    def test_policy_none_and_require_key_env(self) -> None:
        backend = chat.resolve_glm_backend(base_args(glm=True, glm_base_url="http://dgx:8000/v1"), env={})

        argv = chat.build_serve_argv(backend, fak_bin="/x/fak", port=8090,
                                     policy="none", require_key_env="GATE_KEY")

        self.assertNotIn("--policy", argv)
        self.assertIn("--require-key-env", argv)
        self.assertEqual(argv[argv.index("--require-key-env") + 1], "GATE_KEY")
        self.assertEqual(argv[argv.index("--addr") + 1], "127.0.0.1:8090")


class GlmPacketTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        make_account(self.home, ".claude")
        make_account(self.home, ".claude-gem8-netra")
        # Pin opencode discovery to the temp dir too, so the host's real
        # ~/.config/opencode never leaks into this hermetic roster. fleet_accounts
        # reads CONFIG_HOME at import, so patch the module global (not the env).
        # With no opencode account present here, GLM auto-mode falls back to a
        # .claude dir for the session config dir.
        self._prev_config_home = chat.fleet_accounts.CONFIG_HOME
        chat.fleet_accounts.CONFIG_HOME = str(self.home)
        self.addCleanup(self._restore_config_home)

    def _restore_config_home(self) -> None:
        chat.fleet_accounts.CONFIG_HOME = self._prev_config_home

    def rows(self):
        reg_path = self.home / "sessions.json"
        reg_path.write_text(json.dumps(
            {"generated_utc": "2026-06-18T12:00:00+00:00", "sessions": []}), encoding="utf-8")
        return chat.load_roster(str(self.home), str(reg_path))

    def test_glm_mode_env_routes_through_gateway(self) -> None:
        args = base_args(glm=True, prompt="say pong", cwd=str(self.home))

        packet = chat.build_packet(args, self.rows(), env_source={"ZAI_API_KEY": "k"})

        env = packet["env"]
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8080")
        self.assertTrue(env["ANTHROPIC_API_KEY"])  # non-empty placeholder
        self.assertEqual(env["CLAUDE_CONFIG_DIR"], str(self.home / ".claude"))
        # Every Claude tier (and the small/fast model) maps onto the one GLM model.
        model = chat.GLM_DEFAULT_MODEL
        for key in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
                    "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                    "ANTHROPIC_SMALL_FAST_MODEL"):
            self.assertEqual(env[key], model)

    def test_glm_mode_argv_has_no_anthropic_model_flag(self) -> None:
        args = base_args(glm=True, prompt="say pong", cwd=str(self.home))

        packet = chat.build_packet(args, self.rows(), env_source={"ZAI_API_KEY": "k"})

        # The model is pinned via env; a --model opus would fight the remap.
        self.assertNotIn("--model", packet["argv"])
        self.assertNotIn("opus", packet["argv"])

    def test_glm_packet_carries_serve_argv_and_commands(self) -> None:
        args = base_args(glm=True, prompt="say pong", cwd=str(self.home))

        packet = chat.build_packet(args, self.rows(), env_source={"ZAI_API_KEY": "k"})

        self.assertTrue(packet["glm"]["enabled"])
        self.assertIn("serve", packet["glm"]["serve_argv"])
        self.assertIn("ANTHROPIC_BASE_URL=", packet["commands"]["posix"])

    def test_glm_hosted_default_without_key_is_refused_pre_spawn(self) -> None:
        args = base_args(glm=True, prompt="say pong", cwd=str(self.home))

        with self.assertRaises(chat.SelectionError):
            chat.build_packet(args, self.rows(), env_source={})  # no ZAI_API_KEY

    def test_glm_local_backend_needs_no_key(self) -> None:
        args = base_args(glm=True, glm_base_url="http://dgx:8000/v1", glm_model="glm-5.2",
                         prompt="say pong", cwd=str(self.home))

        packet = chat.build_packet(args, self.rows(), env_source={})  # no key, allowed

        self.assertEqual(packet["glm"]["backend"]["base_url"], "http://dgx:8000/v1")
        self.assertEqual(packet["env"]["ANTHROPIC_MODEL"], "glm-5.2")

    def test_native_path_is_unchanged_without_glm(self) -> None:
        args = base_args(prompt="say pong", cwd=str(self.home))

        packet = chat.build_packet(args, self.rows())

        self.assertNotIn("glm", packet)
        self.assertEqual(list(packet["env"].keys()), ["CLAUDE_CONFIG_DIR"])
        self.assertIn("--model", packet["argv"])
        self.assertIn("opus", packet["argv"])

    def test_selecting_opencode_glm_account_implies_gateway_mode(self) -> None:
        # An opencode GLM account is not Anthropic-native; selecting it by name
        # (without --glm) must still route through the gateway, not send glm-5.2
        # straight to api.anthropic.com.
        oc = self.home / "opencode-glm"
        oc.mkdir()
        (oc / "opencode.json").write_text(
            json.dumps({"model": "zai-coding-plan/glm-5.2"}), encoding="utf-8")

        args = base_args(account="glm", prompt="say pong", cwd=str(self.home))
        packet = chat.build_packet(args, self.rows(), env_source={"ZAI_API_KEY": "k"})

        self.assertTrue(packet.get("glm", {}).get("enabled"))
        self.assertEqual(packet["env"]["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8080")
        self.assertEqual(packet["env"]["CLAUDE_CONFIG_DIR"], str(oc))


if __name__ == "__main__":
    unittest.main(verbosity=2)
