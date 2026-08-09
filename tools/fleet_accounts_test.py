#!/usr/bin/env python3
"""Hermetic tests for fleet account policy and runtime availability."""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "tools"))

# Arrange the DIR, not just the variable (#5390). fleet_accounts binds REG_DIR/REGISTRY_PATH
# at IMPORT time from the host registry ladder, and the ladder's discovery rungs are real
# places: on a maintainer's box FLEET_REG_DIR-unset now resolves to %LOCALAPPDATA%\Fleet\
# registry, which carries a live sessions.json AND a live probe_ledger.jsonl. The routing,
# wave-allocation and runtime-status cases below would then grade against the operator's
# actual fleet instead of their fixtures -- green on CI (no host registry) and false-red on
# a laptop, which is the worst shape a suite can have. So pin an empty registry BEFORE the
# import: the env rung outranks every discovery rung, so nothing on this host can be read.
_HERMETIC_REG = tempfile.mkdtemp(prefix="fleet-accounts-test-reg-")
_SAVED_REG_ENV = {k: os.environ.get(k) for k in ("FLEET_REG_DIR", "FLEET_STATE_DIR")}
os.environ["FLEET_REG_DIR"] = _HERMETIC_REG
os.environ.pop("FLEET_STATE_DIR", None)

import fleet_accounts  # noqa: E402


def tearDownModule() -> None:
    for key, value in _SAVED_REG_ENV.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    shutil.rmtree(_HERMETIC_REG, ignore_errors=True)


def account_dir(root: Path, name: str, projects: bool = True) -> Path:
    path = root / name
    path.mkdir()
    if projects:
        (path / "projects").mkdir()
    return path


def opencode_dir(root: Path, name: str, marker: bool = True,
                 config: dict | None = None) -> Path:
    """An opencode config dir; an account iff it holds an opencode.json marker."""
    path = root / name
    path.mkdir(parents=True)
    if marker:
        (path / "opencode.json").write_text(json.dumps(config or {}), encoding="utf-8")
    return path


def login_dir(root: Path, name: str, *, uuid: str = "", email: str = "",
              org_type: str = "claude_max", touch_transcript: bool = True) -> Path:
    """A Claude worker dir logged into a given Anthropic account (writes .claude.json's
    oauthAccount). uuid="" means a not-logged-in dir (oauthAccount absent)."""
    path = account_dir(root, name)
    doc: dict = {}
    if uuid:
        doc["oauthAccount"] = {
            "accountUuid": uuid, "emailAddress": email,
            "organizationUuid": "org-" + uuid, "organizationType": org_type,
        }
    (path / ".claude.json").write_text(json.dumps(doc), encoding="utf-8")
    if touch_transcript:
        proj = path / "projects" / "C--work-fleet"
        proj.mkdir(parents=True, exist_ok=True)
        (proj / "00000000-0000-0000-0000-000000000000.jsonl").write_text("{}\n",
                                                                          encoding="utf-8")
    return path


def write_accounts_registry(home: Path, homes: list[dict]) -> Path:
    path = home / ".claude-accounts" / "registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": "test", "homes": homes}), encoding="utf-8")
    return path


class FleetAccountsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        # a separate, empty XDG-style config root so claude-only tests stay
        # hermetic and never glob the real ~/.config/opencode on the host
        self.config_home = self.home / "config"
        self.config_home.mkdir()
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop("FAK_ACCOUNTS_REGISTRY", None)
        self.addCleanup(self._env.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_discover_accounts_separates_workers_excluded_and_non_accounts(self) -> None:
        account_dir(self.home, ".claude")
        account_dir(self.home, ".claude-gem8-acct")
        account_dir(self.home, ".claude-backup")
        account_dir(self.home, ".claude-monitor", projects=False)
        (self.home / ".claude.json").write_text("{}", encoding="utf-8")

        rows = fleet_accounts.discover_accounts(str(self.home),
                                                config_home=str(self.config_home))
        by_account = {r["account"]: r for r in rows}

        self.assertEqual(by_account[".claude"]["kind"], "worker")
        self.assertEqual(by_account[".claude"]["product"], "claude")
        self.assertEqual(by_account[".claude"]["tag"], "default")
        self.assertEqual(by_account[".claude-gem8-acct"]["kind"], "worker")
        self.assertEqual(by_account[".claude-gem8-acct"]["product"], "claude")
        self.assertEqual(by_account[".claude-gem8-acct"]["tag"], "gem8")
        self.assertEqual(by_account[".claude-backup"]["kind"], "excluded")
        self.assertEqual(by_account[".claude-monitor"]["kind"], "non-account")
        self.assertEqual(by_account[".claude.json"]["kind"], "non-account")

    def test_discover_accounts_excludes_faklocal_dogfood_homes(self) -> None:
        # The fak-kernel dogfood homes are synthesized on demand by
        # `resolve --faklocal-ok`, never enrolled accounts, so discovery keeps them
        # off the switcher roster even with a projects/ subdir — while a normal seat
        # beside them stays a worker (the exclusion is not over-broad).
        account_dir(self.home, ".claude-faklocal")
        account_dir(self.home, ".claude-faklocal-netra")
        account_dir(self.home, ".claude-gem8-acct")

        rows = fleet_accounts.discover_accounts(str(self.home),
                                                config_home=str(self.config_home))
        by_account = {r["account"]: r for r in rows}

        self.assertEqual(by_account[".claude-faklocal"]["kind"], "excluded")
        self.assertEqual(by_account[".claude-faklocal-netra"]["kind"], "excluded")
        self.assertEqual(by_account[".claude-gem8-acct"]["kind"], "worker")

    def test_policy_exclude_matches_claude_login_email(self) -> None:
        login_dir(self.home, ".claude", uuid="uuid-day28",
                  email="day28@example.com")
        pol = {
            "exclude": ["day28"],
            "include_only": [],
            "notes": {"day28": "retired day28 account"},
        }

        rows = fleet_accounts.discover_accounts(str(self.home), pol,
                                                config_home=str(self.config_home))
        row = {r["account"]: r for r in rows}[".claude"]

        self.assertEqual(row["kind"], "excluded")
        self.assertEqual(row["reason"], "retired day28 account")

    def test_fak_accounts_tombstone_excludes_existing_config_dir(self) -> None:
        acct = login_dir(self.home, ".claude-gem7-netra", uuid="uuid-day30",
                         email="day30@example.com")
        write_accounts_registry(self.home, [
            {
                "name": "gem7-netra",
                "dir": str(acct),
                "status": "tombstoned",
                "rehome_to": "default",
                "identity": {"account_uuid": "uuid-day30", "email": "day30@example.com"},
                "tombstone_reason": "archived by operator",
            }
        ])

        rows = fleet_accounts.discover_accounts(str(self.home),
                                                config_home=str(self.config_home))
        row = {r["account"]: r for r in rows}[".claude-gem7-netra"]

        self.assertEqual(row["kind"], "excluded")
        self.assertIn("archived by operator", row["reason"])
        self.assertFalse(fleet_accounts.is_worker(".claude-gem7-netra", str(self.home)))

    def test_fak_accounts_tombstone_excludes_relogin_alias_by_identity(self) -> None:
        login_dir(self.home, ".claude-gem7NEW-netra", uuid="uuid-retired",
                  email="gem7@example.com")
        write_accounts_registry(self.home, [
            {
                "name": "gem7-netra",
                "dir": str(self.home / ".claude-gem7-netra.DELETED-2026-07-05"),
                "status": "tombstoned",
                "rehome_to": "default",
                "identity": {"account_uuid": "uuid-retired", "email": "gem7@example.com"},
            }
        ])

        rows = fleet_accounts.discover_accounts(str(self.home),
                                                config_home=str(self.config_home))
        row = {r["account"]: r for r in rows}[".claude-gem7NEW-netra"]

        self.assertEqual(row["kind"], "excluded")
        self.assertIn("same account identity", row["reason"])

    def test_policy_file_merges_with_safe_defaults(self) -> None:
        policy_path = self.home / "accounts_policy.json"
        policy_path.write_text(
            json.dumps(
                {
                    "exclude": ["gem7"],
                    "include_only": ["gem"],
                    "notes": {"gem7": "operator hold"},
                }
            ),
            encoding="utf-8",
        )

        policy = fleet_accounts.load_policy(str(policy_path))

        self.assertEqual(policy["exclude"], ["gem7"])
        self.assertEqual(policy["include_only"], ["gem"])
        self.assertEqual(policy["notes"]["backup"], fleet_accounts.DEFAULT_POLICY["notes"]["backup"])
        self.assertEqual(policy["notes"]["gem7"], "operator hold")

    def test_runtime_status_blocks_usage_and_credit_accounts(self) -> None:
        registry = {
            "generated_utc": "2026-06-17T00:00:00+00:00",
            "throttle": {".claude-gem7-acct": {"reset": "tomorrow"}},
            "sessions": [
                {
                    "account": ".claude-gem8-acct",
                    "disp": "INFRA_AUTH",
                    "last": "Credit balance is too low",
                }
            ],
        }

        usage = fleet_accounts.runtime_status(".claude-gem7-acct", registry=registry)
        credit = fleet_accounts.runtime_status(".claude-gem8-acct", registry=registry)

        self.assertFalse(usage["available"])
        self.assertEqual(usage["block_kind"], "usage")
        self.assertEqual(usage["reset"], "tomorrow")
        self.assertFalse(credit["available"])
        self.assertEqual(credit["block_kind"], "credit")

    def test_runtime_status_surfaces_weekly_window_alongside_daily(self) -> None:
        registry = {
            "generated_utc": "2026-06-17T00:00:00+00:00",
            "throttle": {".claude-gem7-acct": {
                "reset": "tomorrow",
                "weekly": "next Monday",
            }},
            "sessions": [],
        }

        status = fleet_accounts.runtime_status(".claude-gem7-acct", registry=registry)

        self.assertFalse(status["available"])
        self.assertEqual(status["reset"], "tomorrow")
        self.assertEqual(status["weekly"], "next Monday")
        self.assertIn("weekly next Monday", status["block_reason"])

    def test_runtime_status_keeps_future_weekly_throttle_despite_fresh_probe_ok(self) -> None:
        registry = {
            "generated_utc": "2026-07-01T00:00:00+00:00",
            "throttle": {".claude-gem7-acct": {
                "reset": "Dec 31, 11pm (America/Los_Angeles)",
                "weekly": "Dec 31, 11pm (America/Los_Angeles)",
            }},
            "sessions": [{
                "account": ".claude-gem7-acct",
                "project": "_probe",
                "probe_status": "OK",
            }],
        }

        status = fleet_accounts.runtime_status(".claude-gem7-acct", registry=registry)

        self.assertFalse(status["available"])
        self.assertEqual(status["block_kind"], "usage")
        self.assertEqual(status["status_source"], "registry")
        self.assertEqual(status["weekly"], "Dec 31, 11pm (America/Los_Angeles)")

    def test_runtime_status_blocks_access_wall_without_login_reason(self) -> None:
        registry = {
            "generated_utc": "2026-06-17T00:00:00+00:00",
            "sessions": [
                {
                    "account": ".claude-gem7-acct",
                    "disp": "INFRA_AUTH",
                    "last": (
                        "Your organization has disabled Claude subscription access "
                        "for Claude Code \u00b7 Use an Anthropic API key instead"
                    ),
                }
            ],
        }

        status = fleet_accounts.runtime_status(".claude-gem7-acct", registry=registry)

        self.assertFalse(status["available"])
        self.assertEqual(status["block_kind"], "access")
        self.assertEqual(status["block_reason"], "Claude subscription access disabled")

    def test_runtime_status_clears_old_auth_after_newer_live_work(self) -> None:
        registry = {
            "generated_utc": "2026-06-17T00:00:00+00:00",
            "throttle": {},
            "sessions": [
                {
                    "account": ".claude-gem8-acct",
                    "disp": "LIVE",
                    "age_min": 1.0,
                    "last": "successful tool result",
                },
                {
                    "account": ".claude-gem8-acct",
                    "disp": "INFRA_AUTH",
                    "age_min": 200.0,
                    "last": "Please run /login",
                },
            ],
        }

        status = fleet_accounts.runtime_status(".claude-gem8-acct", registry=registry)

        self.assertTrue(status["available"])
        self.assertFalse(status["blocked"])

    def test_runtime_status_handles_malformed_age_rows_without_crashing(self) -> None:
        registry = {
            "generated_utc": "2026-06-17T00:00:00+00:00",
            "throttle": {},
            "sessions": [
                {
                    "account": ".claude-gem8-acct",
                    "disp": "INFRA_AUTH",
                    "age_min": "unknown",
                    "last": "Please run /login",
                },
                {
                    "account": ".claude-gem8-acct",
                    "disp": "DONE",
                    "age_min": 5.0,
                    "last": "completed after re-login",
                },
            ],
        }

        status = fleet_accounts.runtime_status(".claude-gem8-acct", registry=registry)

        self.assertFalse(status["available"])
        self.assertTrue(status["blocked"])
        self.assertEqual(status["block_kind"], "auth")

    def test_explicit_empty_registry_does_not_load_live_registry(self) -> None:
        # The claim is about the `registry={}` ARGUMENT short-circuiting, so the registry dir
        # has to be arranged, not merely unnamed: popping FLEET_REG_DIR used to mean "no
        # ledger anywhere", but since #5390 an unnamed dir falls through to the host's real
        # registry, whose live ledger legitimately opens the probe rung. Name the module's
        # empty dir instead -- the same "nothing to read" the pop used to imply by accident.
        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": _HERMETIC_REG}, clear=False):
            with mock.patch.object(
                fleet_accounts, "load_registry",
                side_effect=AssertionError("unexpected load")
            ):
                with mock.patch.object(
                    fleet_accounts, "_fresh_probe_from_ledger",
                    side_effect=AssertionError("unexpected ledger read")
                ):
                    status = fleet_accounts.runtime_status(".claude-gem8-acct", registry={})

        self.assertTrue(status["available"])
        self.assertEqual(status["status_source"], "none")

    def test_runtime_status_blocks_persisted_auth_without_current_session(self) -> None:
        registry = {
            "generated_utc": "2026-06-17T00:00:00+00:00",
            "auth": {
                ".claude-gem8-acct": {
                    "block_kind": "auth",
                    "block_reason": "auth/login required",
                    "seen_utc": "2026-06-16T23:00:00+00:00",
                }
            },
            "sessions": [],
        }

        status = fleet_accounts.runtime_status(".claude-gem8-acct", registry=registry)

        self.assertFalse(status["available"])
        self.assertEqual(status["block_kind"], "auth")

    def test_runtime_status_reclassifies_stale_persisted_access_wall(self) -> None:
        registry = {
            "generated_utc": "2026-06-17T00:00:00+00:00",
            "auth": {
                ".claude-gem7-acct": {
                    "block_kind": "auth",
                    "block_reason": "auth/login required",
                    "last": (
                        "Your organization has disabled Claude subscription access "
                        "for Claude Code \u00b7 Use an Anthropic API key instead"
                    ),
                }
            },
            "sessions": [],
        }

        status = fleet_accounts.runtime_status(".claude-gem7-acct", registry=registry)

        self.assertFalse(status["available"])
        self.assertEqual(status["block_kind"], "access")
        self.assertEqual(status["block_reason"], "Claude subscription access disabled")

    def test_runtime_status_clears_persisted_auth_after_newer_success(self) -> None:
        registry = {
            "generated_utc": "2026-06-17T00:00:00+00:00",
            "auth": {
                ".claude-gem8-acct": {
                    "block_kind": "auth",
                    "block_reason": "auth/login required",
                    "seen_utc": "2026-06-16T23:00:00+00:00",
                }
            },
            "sessions": [
                {
                    "account": ".claude-gem8-acct",
                    "disp": "DONE",
                    "seen_utc": "2026-06-16T23:30:00+00:00",
                    "last": "completed after re-login",
                }
            ],
        }

        status = fleet_accounts.runtime_status(".claude-gem8-acct", registry=registry)

        self.assertTrue(status["available"])
        self.assertFalse(status["blocked"])

    def test_is_worker_is_dir_independent_and_honors_exclude(self) -> None:
        # is_worker is the cheap per-account check the session tools call in their
        # scan loop; it must not require the account dir to exist on disk.
        pol = {"exclude": ["backup", "breakglass"], "include_only": [], "notes": {}}
        self.assertTrue(fleet_accounts.is_worker(".claude-gem99-acct", str(self.home), pol))
        self.assertFalse(fleet_accounts.is_worker(".claude-backup", str(self.home), pol))
        self.assertFalse(
            fleet_accounts.is_worker(".claude-old.DELETED-2026-07-05", str(self.home), pol))

    def test_include_only_excludes_accounts_off_the_allowlist(self) -> None:
        account_dir(self.home, ".claude")
        account_dir(self.home, ".claude-gem8-acct")
        account_dir(self.home, ".claude-c10-acct")
        pol = {"exclude": [], "include_only": ["gem8"], "notes": {}}
        kinds = {r["tag"]: r["kind"]
                 for r in fleet_accounts.discover_accounts(str(self.home), pol,
                                                           config_home=str(self.config_home))}
        self.assertEqual(kinds["gem8"], "worker")
        self.assertEqual(kinds["default"], "excluded")
        self.assertEqual(kinds["c10"], "excluded")

    def test_malformed_policy_file_falls_back_to_defaults(self) -> None:
        bad = self.home / "bad_policy.json"
        bad.write_text("{not valid json", encoding="utf-8")
        pol = fleet_accounts.load_policy(str(bad))
        self.assertIn("backup", pol["exclude"])

    def test_policy_path_does_not_follow_reg_dir(self) -> None:
        # The account POLICY is operator config, not runtime state. The watchdog
        # redirects FLEET_REG_DIR to a host state dir so sessions.json lands off the
        # repo; the policy must NOT move with it (doing so silently sent the watchdog to
        # a non-existent LOCALAPPDATA policy -> example fallback -> CLI/watchdog drift).
        import importlib
        import os
        state_dir = self.home / "state_registry"
        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(state_dir)}, clear=False):
            os.environ.pop("FLEET_POLICY_DIR", None)
            os.environ.pop("FLEET_POLICY_PATH", None)
            reloaded = importlib.reload(fleet_accounts)
            try:
                repo_registry = os.path.join(
                    os.path.dirname(os.path.abspath(reloaded.__file__)), "_registry")
                # registry (runtime state) DOES follow FLEET_REG_DIR
                self.assertEqual(reloaded.REGISTRY_PATH,
                                 os.path.join(str(state_dir), "sessions.json"))
                # policy (operator config) stays pinned to the repo's tools/_registry
                self.assertEqual(reloaded.POLICY_PATH,
                                 os.path.join(repo_registry, "accounts_policy.json"))
                self.assertNotIn(str(state_dir), reloaded.POLICY_PATH)
            finally:
                importlib.reload(fleet_accounts)  # restore module globals for other tests

    def test_policy_path_respects_explicit_override(self) -> None:
        import importlib
        import os
        custom = str(self.home / "my_policy.json")
        with mock.patch.dict(os.environ, {"FLEET_POLICY_PATH": custom}, clear=False):
            reloaded = importlib.reload(fleet_accounts)
            try:
                self.assertEqual(reloaded.POLICY_PATH, custom)
            finally:
                importlib.reload(fleet_accounts)

    # --- opencode: a parallel product family alongside Claude -----------------

    def test_account_product_and_tag_classify_both_families(self) -> None:
        self.assertEqual(fleet_accounts.account_product(".claude"), "claude")
        self.assertEqual(fleet_accounts.account_product(".claude-gem8-acct"), "claude")
        self.assertEqual(fleet_accounts.account_product(".claude.json"), "claude")
        self.assertEqual(fleet_accounts.account_product("opencode"), "opencode")
        self.assertEqual(fleet_accounts.account_product("opencode-glm"), "opencode")

        self.assertEqual(fleet_accounts.account_tag("opencode"), "default")
        self.assertEqual(fleet_accounts.account_tag("opencode-glm"), "glm")
        # claude tag derivation is unchanged
        self.assertEqual(fleet_accounts.account_tag(".claude-gem8-acct"), "gem8")
        self.assertEqual(fleet_accounts.account_tag(".claude"), "default")

    def test_discover_accounts_finds_opencode_alongside_claude(self) -> None:
        # claude side
        account_dir(self.home, ".claude")
        account_dir(self.home, ".claude-gem8-acct")
        # opencode side
        opencode_dir(self.config_home, "opencode")
        opencode_dir(self.config_home, "opencode-glm")

        rows = fleet_accounts.discover_accounts(str(self.home),
                                                config_home=str(self.config_home))
        by_key = {(r["product"], r["account"]): r for r in rows}

        self.assertEqual(by_key[("claude", ".claude")]["kind"], "worker")
        self.assertEqual(by_key[("claude", ".claude-gem8-acct")]["tag"], "gem8")
        self.assertEqual(by_key[("opencode", "opencode")]["kind"], "worker")
        self.assertEqual(by_key[("opencode", "opencode")]["tag"], "default")
        self.assertEqual(by_key[("opencode", "opencode-glm")]["kind"], "worker")
        self.assertEqual(by_key[("opencode", "opencode-glm")]["tag"], "glm")
        # every row carries a product
        self.assertTrue(all(r.get("product") in ("claude", "opencode") for r in rows))

    def test_opencode_dir_without_config_marker_is_non_account(self) -> None:
        opencode_dir(self.config_home, "opencode")              # account
        opencode_dir(self.config_home, "opencode-empty", marker=False)  # no opencode.json
        opencode_dir(self.config_home, "opencode-backup")       # excluded by default policy
        # a plain file named opencode.lock is not an account
        (self.config_home / "opencode.lock").write_text("x", encoding="utf-8")

        rows = fleet_accounts.discover_accounts(str(self.home),
                                                config_home=str(self.config_home))
        oc = {r["account"]: r for r in rows if r["product"] == "opencode"}

        self.assertEqual(oc["opencode"]["kind"], "worker")
        self.assertEqual(oc["opencode-empty"]["kind"], "non-account")
        self.assertIn("opencode.json", oc["opencode-empty"]["reason"])
        self.assertEqual(oc["opencode-backup"]["kind"], "excluded")
        self.assertEqual(oc["opencode.lock"]["kind"], "non-account")

    def test_opencode_account_runtime_status_is_available_without_sessions(self) -> None:
        # runtime_status is account-keyed and product-neutral: an opencode
        # basename with no recorded sessions is simply available/healthy.
        status = fleet_accounts.runtime_status("opencode-glm", registry={})
        self.assertTrue(status["available"])
        self.assertEqual(status["status_source"], "none")

    def test_is_worker_classifies_opencode_basenames(self) -> None:
        pol = {"exclude": ["backup", "breakglass"], "include_only": [], "notes": {}}
        self.assertTrue(fleet_accounts.is_worker("opencode", str(self.home), pol))
        self.assertTrue(fleet_accounts.is_worker("opencode-glm", str(self.home), pol))
        self.assertFalse(fleet_accounts.is_worker("opencode-backup", str(self.home), pol))

    def test_profiles_classify_claude_and_glm52_opencode_tiers(self) -> None:
        account_dir(self.home, ".claude")
        opencode_dir(
            self.config_home,
            "opencode-zai2",
            config={
                "model": "zai-coding-plan/glm-5.2",
                "small_model": "zai-coding-plan/glm-4.5-air",
                "provider": {"zai-coding-plan": {"options": {"apiKey": "secret"}}},
            },
        )

        rows = fleet_accounts.discover_accounts(str(self.home),
                                                config_home=str(self.config_home))
        by_account = {r["account"]: r for r in rows}

        self.assertEqual(by_account[".claude"]["model_tier"], 1)
        self.assertEqual(by_account[".claude"]["model"], "opus")
        self.assertEqual(by_account[".claude"]["model_effort"], "xhigh")
        self.assertEqual(by_account["opencode-zai2"]["model_tier"], 2)
        self.assertEqual(by_account["opencode-zai2"]["model"], "zai-coding-plan/glm-5.2")
        self.assertEqual(by_account["opencode-zai2"]["small_model"], "zai-coding-plan/glm-4.5-air")
        self.assertNotIn("provider", by_account["opencode-zai2"])

    def test_model_tier_from_name_gemini_flash_is_tier2(self) -> None:
        # Mirror of the Go TestModelTierFromNameGeminiFlashIsTier2 (the two ports stay
        # logic-identical): Gemini 3.5 Flash is the new tier-2 lightweight GCP Vertex seat,
        # in every id shape it appears; the existing tiers are unchanged.
        cases = [
            ("gemini-3.5-flash", 2),
            ("google/gemini-3.5-flash", 2),
            ("Gemini 3.5 Flash", 2),
            ("gemini_3.5_flash", 2),
            ("glm-5.2", 2),
            ("zai-coding-plan/glm-5.2", 2),
            ("gpt-5.5", 1),
            ("opus-4.6", 1),
            ("deepseek-v4-pro", 1),
            ("kimi-k2.6", 1),
            ("gemini-3.5-pro", 3),  # only Flash is tier 2
            ("llama3.2", 3),
            ("", 3),
        ]
        for model, tier in cases:
            self.assertEqual(
                fleet_accounts._model_tier_from_name(model), tier,
                f"_model_tier_from_name({model!r})",
            )

    def test_nim_coding_seats_rank_as_tier1_opencode_workers(self) -> None:
        seats = [
            ("opencode-nim-deepseek-v4-pro", "nim-deepseek-v4-pro",
             fleet_accounts.NIM_DEEPSEEK_V4_PRO_MODEL, 30),
            ("opencode-nim-kimi-k26", "nim-kimi-k26",
             fleet_accounts.NIM_KIMI_K26_MODEL, 20),
            ("opencode-nim-glm52", "nim-glm52",
             fleet_accounts.NIM_GLM52_MODEL, 10),
        ]
        for account, _, _, _ in seats:
            opencode_dir(self.config_home, account, config={})

        rows = fleet_accounts.annotate_accounts(
            fleet_accounts.discover_accounts(str(self.home),
                                             config_home=str(self.config_home)),
            registry={},
        )
        by_account = {r["account"]: r for r in rows}

        for account, tag, model, weight in seats:
            row = by_account[account]
            self.assertEqual(row["kind"], "worker")
            self.assertEqual(row["product"], "opencode")
            self.assertEqual(row["tag"], tag)
            self.assertEqual(row["model_tier"], 1)
            self.assertEqual(row["model"], model)
            self.assertEqual(row["profile_source"], f"default:nvidia-nim-coding:{tag}")
            self.assertEqual(row["route_weight"], weight)

        routed = fleet_accounts.route_account(
            rows, "implement the feature", "engineering", product="opencode")
        self.assertTrue(routed["ok"])
        self.assertEqual(routed["account"]["account"], "opencode-nim-deepseek-v4-pro")

    def test_route_account_defaults_hard_to_tier1_and_light_to_tier2(self) -> None:
        account_dir(self.home, ".claude")
        opencode_dir(self.config_home, "opencode-zai2",
                     config={"model": "zai-coding-plan/glm-5.2"})
        rows = fleet_accounts.annotate_accounts(
            fleet_accounts.discover_accounts(str(self.home),
                                             config_home=str(self.config_home)),
            registry={},
        )

        hard = fleet_accounts.route_account(rows, "fix the failing test")
        light = fleet_accounts.route_account(rows, "say pong")

        self.assertTrue(hard["ok"])
        self.assertEqual(hard["selected_tier"], 1)
        self.assertEqual(hard["account"]["product"], "claude")
        self.assertTrue(light["ok"])
        self.assertEqual(light["selected_tier"], 2)
        self.assertEqual(light["account"]["product"], "opencode")

    def test_seat_leases_fold_makes_live_opencode_worker_visible(self) -> None:
        # Defect 3: a dispatched opencode/glm docs worker writes an opencode transcript
        # the watchdog session scan does not fold, so it would show 0 live_sessions. A
        # live dispatch lease (from the .pid/.account sidecars) reflects it into the roster.
        account_dir(self.home, ".claude")
        opencode_dir(self.config_home, "opencode-zai2",
                     config={"model": "zai-coding-plan/glm-5.2"})
        discovered = fleet_accounts.discover_accounts(
            str(self.home), config_home=str(self.config_home))
        oc = next(r for r in discovered if r.get("product") == "opencode")

        rows = fleet_accounts.annotate_accounts(
            discovered, registry={},
            seat_leases=[{"tag": oc["tag"], "dir": oc["dir"]}])
        row = next(r for r in rows if r.get("account") == oc["account"])
        self.assertEqual(row["live_sessions"], 1)
        self.assertGreaterEqual(row["active_sessions"], 1)
        self.assertEqual(row["dispatch_leases"], 1)

    def test_seat_leases_fold_never_touches_claude_rows(self) -> None:
        # The fold is opencode-scoped: a claude row already surfaces via the watchdog
        # registry, so even a same-tag lease must not bump its counts.
        account_dir(self.home, ".claude")
        discovered = fleet_accounts.discover_accounts(
            str(self.home), config_home=str(self.config_home))
        cl = next(r for r in discovered if r.get("product") == "claude"
                  and r.get("kind") == "worker")
        rows = fleet_accounts.annotate_accounts(
            discovered, registry={},
            seat_leases=[{"tag": cl["tag"], "dir": cl["dir"]}])
        row = next(r for r in rows if r.get("account") == cl["account"])
        self.assertNotIn("dispatch_leases", row)
        self.assertEqual(row.get("live_sessions") or 0, 0)

    def test_route_account_strict_tier2_does_not_upshift(self) -> None:
        account_dir(self.home, ".claude")
        rows = fleet_accounts.annotate_accounts(
            fleet_accounts.discover_accounts(str(self.home),
                                             config_home=str(self.config_home)),
            registry={},
        )

        routed = fleet_accounts.route_account(rows, "say pong", "t2", strict_tier=True)

        self.assertFalse(routed["ok"])
        self.assertIn("no matching worker tier", routed["reason"])

    def test_classify_task_work_kind_overrides_prompt_heuristic(self) -> None:
        # "audit the docs" trips HARD_TASK_HINT_RE -> would be tier1 on text alone.
        text_only = fleet_accounts.classify_task("audit the docs and fix the index")
        self.assertEqual(text_only["target_tier"], 1)
        # But a caller that KNOWS it is gardening pins tier2, despite the same words.
        gardening = fleet_accounts.classify_task(
            "audit the docs and fix the index", "gardening")
        self.assertEqual(gardening["target_tier"], 2)
        self.assertEqual(gardening["class"], "gardening")
        self.assertEqual(gardening["confidence"], 1.0)
        # Engineering pins tier1 even for a short prompt that would otherwise be light.
        engineering = fleet_accounts.classify_task("say pong", "engineering")
        self.assertEqual(engineering["target_tier"], 1)
        self.assertEqual(engineering["class"], "engineering")

    def test_classify_task_work_kind_aliases(self) -> None:
        for token in ("maintenance", "garden", "cleanup", "chore", "triage"):
            self.assertEqual(
                fleet_accounts.classify_task("", token)["target_tier"], 2, token)
        for token in ("eng", "dev", "feature", "implementation"):
            self.assertEqual(
                fleet_accounts.classify_task("", token)["target_tier"], 1, token)

    def test_route_account_gardening_picks_tier2_engineering_picks_tier1(self) -> None:
        account_dir(self.home, ".claude")
        opencode_dir(self.config_home, "opencode-zai2",
                     config={"model": "zai-coding-plan/glm-5.2"})
        rows = fleet_accounts.annotate_accounts(
            fleet_accounts.discover_accounts(str(self.home),
                                             config_home=str(self.config_home)),
            registry={},
        )

        # A gardening loop whose prompt would read "hard" still routes to tier2 (GLM).
        gardening = fleet_accounts.route_account(
            rows, "review and audit the cluster index", "gardening")
        self.assertTrue(gardening["ok"])
        self.assertEqual(gardening["selected_tier"], 2)
        self.assertEqual(gardening["account"]["product"], "opencode")

        # Engineering work routes to tier1 (Claude/opus) even for a terse prompt.
        engineering = fleet_accounts.route_account(rows, "ship it", "engineering")
        self.assertTrue(engineering["ok"])
        self.assertEqual(engineering["selected_tier"], 1)
        self.assertEqual(engineering["account"]["product"], "claude")

    def test_route_account_skips_live_leased_pool(self) -> None:
        opencode_dir(self.config_home, "opencode",
                     config={"model": "zai-coding-plan/glm-5.2"})
        opencode_dir(self.config_home, "opencode-zai2",
                     config={"model": "zai-coding-plan/glm-5.2"})
        rows = fleet_accounts.annotate_accounts(
            fleet_accounts.discover_accounts(str(self.home),
                                             config_home=str(self.config_home)),
            registry={},
        )

        routed = fleet_accounts.route_account(
            rows,
            "review and clean up the backlog",
            "gardening",
            product="opencode",
            leases=[{
                "worker": "resolve-3032-20260706-143708",
                "tag": "default",
                "dir": str(self.config_home / "opencode"),
            }],
        )

        self.assertTrue(routed["ok"])
        self.assertEqual(routed["selected_tier"], 2)
        self.assertEqual(routed["account"]["account"], "opencode-zai2")

    def test_route_account_gardening_upshifts_to_tier1_when_no_tier2(self) -> None:
        # No tier-2 account exists. A gardening task must NOT stall: it up-shifts to
        # tier1 (preserving the work) and flags the fallback, rather than failing.
        account_dir(self.home, ".claude")
        rows = fleet_accounts.annotate_accounts(
            fleet_accounts.discover_accounts(str(self.home),
                                             config_home=str(self.config_home)),
            registry={},
        )

        routed = fleet_accounts.route_account(rows, "tidy the docs", "gardening")
        self.assertTrue(routed["ok"])
        self.assertEqual(routed["target_tier"], 2)
        self.assertEqual(routed["selected_tier"], 1)
        self.assertTrue(routed["fallback_used"])

    def test_route_weight_biases_tiebreak_toward_roomy_account(self) -> None:
        # Two equally-available tier-1 Claude accounts with the SAME session load: with no
        # bias the deterministic tiebreak is alphabetical (gem7 < gem8). An operator who
        # KNOWS gem8 has more room (the router can't measure quota) lifts it with a positive
        # route_weight, and the switcher must then prefer gem8 despite the alphabetical order.
        account_dir(self.home, ".claude-gem7-acct")
        account_dir(self.home, ".claude-gem8-acct")

        # baseline: no weights -> alphabetical tiebreak picks gem7
        pol_plain = {"exclude": [], "include_only": [], "notes": {},
                     "account_profiles": {}, "route_weights": {}, "routing": {}}
        rows = fleet_accounts.annotate_accounts(
            fleet_accounts.discover_accounts(str(self.home), pol_plain,
                                             config_home=str(self.config_home)),
            registry={})
        baseline = fleet_accounts.route_account(rows, "ship it", "engineering", policy=pol_plain)
        self.assertTrue(baseline["ok"])
        self.assertEqual(baseline["account"]["tag"], "gem7")

        # weighted: gem8 declared roomier via the dedicated route_weights map -> it wins
        # despite sorting after gem7, and it KEEPS its inferred tier-1 (the whole reason
        # route_weights is separate from account_profiles).
        pol_weighted = {"exclude": [], "include_only": [], "notes": {},
                        "account_profiles": {}, "route_weights": {"gem8": 10}, "routing": {}}
        rows_w = fleet_accounts.annotate_accounts(
            fleet_accounts.discover_accounts(str(self.home), pol_weighted,
                                             config_home=str(self.config_home)),
            registry={})
        weighted = fleet_accounts.route_account(rows_w, "ship it", "engineering",
                                                policy=pol_weighted)
        self.assertTrue(weighted["ok"])
        self.assertEqual(weighted["account"]["tag"], "gem8")
        self.assertEqual(weighted["account"]["model_tier"], 1)  # tier inference intact
        self.assertEqual(weighted["account"]["route_weight"], 10)

    def test_route_weight_defaults_to_zero_and_keeps_session_balancing(self) -> None:
        # With no route_weight, the row still carries the default 0 and routing is unchanged:
        # fewest-live wins regardless of alphabetical order (gem8 idle beats a busy gem7).
        account_dir(self.home, ".claude-gem7-acct")
        account_dir(self.home, ".claude-gem8-acct")
        registry = {
            "generated_utc": "2026-06-17T00:00:00+00:00",
            "sessions": [
                {"account": ".claude-gem7-acct", "disp": "LIVE", "age_min": 1.0},
                {"account": ".claude-gem7-acct", "disp": "LIVE", "age_min": 1.0},
            ],
        }
        rows = fleet_accounts.annotate_accounts(
            fleet_accounts.discover_accounts(str(self.home),
                                             config_home=str(self.config_home)),
            registry=registry)
        routed = fleet_accounts.route_account(rows, "ship it", "engineering")
        self.assertTrue(routed["ok"])
        self.assertEqual(routed["account"]["tag"], "gem8")  # fewest live, not alphabetical
        self.assertEqual(routed["account"]["route_weight"], 0)

    def test_account_cap_sidecar_blocks_selected_tag_and_routes_around_it(self) -> None:
        import datetime as dt
        account_dir(self.home, ".claude-day26NEW-netra-acct")
        account_dir(self.home, ".claude-gem8NEW-netra-acct")
        runs = self.home / "runs"
        runs.mkdir()
        until = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat()
        (runs / "account-cap-claude.json").write_text(json.dumps({
            "product": "claude",
            "account": "day26NEW-netra",
            "kind": "weekly",
            "reset_text": "",
            "evidence_log": "resolve-2272.log",
            "detected": "2026-07-03T22:39:51Z",
            "until": until,
        }), encoding="utf-8")

        rows = fleet_accounts.annotate_accounts(
            fleet_accounts.discover_accounts(str(self.home),
                                             config_home=str(self.config_home)),
            registry={}, cap_runs_dir=str(runs))
        by_tag = {r["tag"]: r for r in rows}

        self.assertFalse(by_tag["day26NEW-netra"]["available"])
        self.assertEqual(by_tag["day26NEW-netra"]["block_kind"], "usage")
        routed = fleet_accounts.route_account(rows, "ship it", "engineering",
                                              product="claude")
        self.assertTrue(routed["ok"])
        self.assertEqual(routed["account"]["tag"], "gem8NEW-netra")

    def test_malformed_account_cap_sidecar_fails_open(self) -> None:
        account_dir(self.home, ".claude-day26NEW-netra-acct")
        runs = self.home / "runs"
        runs.mkdir()
        (runs / "account-cap-claude.json").write_text("{not json", encoding="utf-8")

        rows = fleet_accounts.annotate_accounts(
            fleet_accounts.discover_accounts(str(self.home),
                                             config_home=str(self.config_home)),
            registry={}, cap_runs_dir=str(runs))

        self.assertTrue({r["tag"]: r for r in rows}["day26NEW-netra"]["available"])

    def test_route_account_json_contract_for_detached_launcher(self) -> None:
        # launch_goal_detached.ps1 parses route's account.dir + selected_tier +
        # fallback_used + ok. Lock that shape so a rename can't silently break the
        # PowerShell dispatch path (which has no unit harness here).
        account_dir(self.home, ".claude")
        rows = fleet_accounts.annotate_accounts(
            fleet_accounts.discover_accounts(str(self.home),
                                             config_home=str(self.config_home)),
            registry={},
        )
        routed = fleet_accounts.route_account(
            rows, "", "engineering", product="claude")
        self.assertTrue(routed["ok"])
        self.assertIn("account", routed)
        self.assertIn("dir", routed["account"])
        self.assertTrue(routed["account"]["dir"])
        self.assertIn("selected_tier", routed)
        self.assertIn("fallback_used", routed)
        self.assertEqual(routed["account"]["product"], "claude")

    # ---- resolve_account / read_oauth_token / annotated_roster ----------------

    def test_resolve_account_routes_and_attaches_token(self) -> None:
        # The canonical front-door call: route (no pin) -> a flat record carrying the
        # config_dir, the long-lived oauth token, and the selected tier.
        d = account_dir(self.home, ".claude")
        (d / ".oauth-token").write_text("tok-xyz\n", encoding="utf-8")
        r = fleet_accounts.resolve_account(
            work_kind="engineering", product="claude",
            home=str(self.home), config_home=str(self.config_home), registry={})
        self.assertTrue(r["ok"])
        self.assertEqual(r["config_dir"], str(d))
        self.assertEqual(r["oauth_token"], "tok-xyz")
        self.assertEqual(r["selected_tier"], 1)
        self.assertFalse(r["fallback_used"])
        # The flat contract the shell front doors parse:
        for key in ("ok", "reason", "account", "tag", "product", "config_dir",
                    "oauth_token", "selected_tier", "target_tier", "fallback_used"):
            self.assertIn(key, r)

    def test_resolve_account_pins_a_named_worker(self) -> None:
        account_dir(self.home, ".claude")
        gem = account_dir(self.home, ".claude-gem8-acct")
        r = fleet_accounts.resolve_account(
            "gem8", home=str(self.home), config_home=str(self.config_home), registry={})
        self.assertTrue(r["ok"])
        self.assertEqual(r["config_dir"], str(gem))
        self.assertEqual(r["tag"], "gem8")
        self.assertEqual(r["reason"], "pinned account")
        # No .oauth-token on disk -> oauth_token is None (caller drops the ambient one).
        self.assertIsNone(r["oauth_token"])

    def test_resolve_account_unknown_pin_is_not_ok(self) -> None:
        account_dir(self.home, ".claude")
        r = fleet_accounts.resolve_account(
            "no-such", home=str(self.home), config_home=str(self.config_home), registry={})
        self.assertFalse(r["ok"])
        self.assertIn("not an offered worker", r["reason"])
        self.assertEqual(r["config_dir"], "")

    def test_resolve_account_blocked_pin_refused_without_fallback(self) -> None:
        account_dir(self.home, ".claude-gem8-acct")
        # A throttle on the account makes it unavailable; pin must refuse.
        reg = {"throttle": {".claude-gem8-acct": {"reset": "Dec 31, 11:59pm"}}}
        r = fleet_accounts.resolve_account(
            "gem8", home=str(self.home), config_home=str(self.config_home), registry=reg)
        self.assertFalse(r["ok"])
        self.assertIn("blocked", r["reason"])
        self.assertTrue(r["block_reason"])
        # ...but -AllowTierFallback (allow_tier_fallback) launches it anyway.
        r2 = fleet_accounts.resolve_account(
            "gem8", allow_tier_fallback=True,
            home=str(self.home), config_home=str(self.config_home), registry=reg)
        self.assertTrue(r2["ok"])
        self.assertEqual(r2["tag"], "gem8")

    def test_resolve_account_faklocal_synthesizes_isolated_dir(self) -> None:
        r = fleet_accounts.resolve_account(
            "faklocal", faklocal_ok=True,
            home=str(self.home), config_home=str(self.config_home), registry={})
        self.assertTrue(r["ok"])
        self.assertEqual(r["tag"], "faklocal")
        self.assertEqual(r["config_dir"], str(self.home / ".claude-faklocal"))
        self.assertTrue((self.home / ".claude-faklocal" / "projects").is_dir())

    def test_read_oauth_token_present_absent_and_empty(self) -> None:
        d = account_dir(self.home, ".claude-gem8-acct")
        self.assertIsNone(fleet_accounts.read_oauth_token(str(d)))  # no file
        (d / ".oauth-token").write_text("  abc123  \n", encoding="utf-8")
        self.assertEqual(fleet_accounts.read_oauth_token(str(d)), "abc123")  # stripped
        (d / ".oauth-token").write_text("\n\n", encoding="utf-8")
        self.assertIsNone(fleet_accounts.read_oauth_token(str(d)))  # empty -> None
        self.assertIsNone(fleet_accounts.read_oauth_token(""))      # no dir -> None

    def test_annotated_roster_shape_matches_inlined_call(self) -> None:
        account_dir(self.home, ".claude")
        account_dir(self.home, ".claude-gem8-acct")
        rows = fleet_accounts.annotated_roster(
            str(self.home), config_home=str(self.config_home), registry={})
        # Same rows + the live-availability fields annotate_accounts attaches.
        self.assertTrue(rows)
        for r in rows:
            self.assertIn("available", r)
            self.assertIn("kind", r)
        tags = {r["tag"] for r in rows}
        self.assertIn("gem8", tags)


class WaveAllocationTest(unittest.TestCase):
    """allocate_wave hands a parallel fan-out N bounded account session slots at once.

    The provable-benefit witness: a fan-out that calls single-account resolve() N
    times in a burst gets the SAME account N times (no session has registered yet to
    move the live-load tie-break), so all N lanes share ONE usage pool and the
    fan-out serializes. A wave allocates bounded slots across distinct pools instead,
    so Claude worker accounts can each carry several sessions without unbounded
    overbooking."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.config_home = self.home / "config"
        self.config_home.mkdir()
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop("FAK_ACCOUNTS_REGISTRY", None)
        self.addCleanup(self._env.stop)
        self.addCleanup(self._tmp.cleanup)

    def _three_distinct(self) -> None:
        login_dir(self.home, ".claude-gem5-acct", uuid="uuid-5", email="gem5@x.ai")
        login_dir(self.home, ".claude-gem7-acct", uuid="uuid-7", email="gem7@x.ai")
        login_dir(self.home, ".claude-gem8-acct", uuid="uuid-8", email="gem8@x.ai")

    def test_wave_allocates_session_slots_and_underfills_honestly(self) -> None:
        self._three_distinct()
        w = fleet_accounts.allocate_wave(
            13, task_class="t1", home=str(self.home),
            config_home=str(self.config_home), registry={})
        self.assertTrue(w["ok"])
        self.assertEqual(w["granted"], 12)       # 3 Claude pools x 4 slots
        self.assertEqual(w["distinct_pools"], 3)
        self.assertEqual(w["shortfall"], 1)      # asked 13, got 12 -> honest, never over cap
        pools = [lane["pool"] for lane in w["lanes"]]
        self.assertEqual(max(pools.count(p) for p in set(pools)), 4)
        self.assertTrue(all(lane["session_cap"] == 4 for lane in w["lanes"]))
        self.assertEqual({lane["tag"] for lane in w["lanes"]}, {"gem5", "gem7", "gem8"})

    def test_wave_subtracts_live_lease_slots(self) -> None:
        login_dir(self.home, ".claude-gem8-acct", uuid="uuid-8", email="gem8@x.ai")
        leases = [
            {"worker": f"resolve-{i}", "tag": "gem8",
             "dir": str(self.home / ".claude-gem8-acct")}
            for i in range(3)
        ]
        w = fleet_accounts.allocate_wave(
            4, task_class="t1", home=str(self.home),
            config_home=str(self.config_home), registry={}, leases=leases)
        self.assertTrue(w["ok"])
        self.assertEqual(w["granted"], 1)
        self.assertEqual(w["shortfall"], 3)
        self.assertEqual(w["distinct_pools"], 1)
        self.assertEqual(w["lanes"][0]["session_slot"], 4)
        self.assertEqual(w["lanes"][0]["session_cap"], 4)

    def test_wave_beats_naive_resolve_burst(self) -> None:
        # THE witness: identical roster, identical registry; the only change is wave vs
        # a burst of resolve(). naive collapses to one pool, wave spreads across all three.
        self._three_distinct()
        kw = dict(home=str(self.home), config_home=str(self.config_home), registry={})
        naive = {fleet_accounts.resolve_account(task_class="t1", **kw)["tag"]
                 for _ in range(3)}
        wave = {lane["tag"] for lane in fleet_accounts.allocate_wave(
            3, task_class="t1", **kw)["lanes"]}
        self.assertEqual(len(naive), 1, "naive burst piles all 3 lanes on one pool")
        self.assertGreater(len(wave), 1, "wave spreads slots beyond the first pool")

    def test_wave_excludes_duplicate_identity_dirs(self) -> None:
        # Two dirs logged into ONE Anthropic account are ONE pool; a wave must not hand
        # out both (that would re-collapse two lanes onto a single usage limit).
        login_dir(self.home, ".claude-gem5-acct", uuid="uuid-5", email="gem5@x.ai")
        login_dir(self.home, ".claude", uuid="uuid-5", email="gem5@x.ai")  # same account
        login_dir(self.home, ".claude-gem8-acct", uuid="uuid-8", email="gem8@x.ai")
        w = fleet_accounts.allocate_wave(
            5, task_class="t1", home=str(self.home),
            config_home=str(self.config_home), registry={})
        self.assertEqual(w["granted"], 5, "3 dirs but 2 distinct Claude accounts with 4 slots each")
        pools = [lane["pool"] for lane in w["lanes"]]
        self.assertEqual(set(pools), {"uuid:uuid-5", "uuid:uuid-8"})
        self.assertLessEqual(max(pools.count(p) for p in set(pools)), 4)

    def test_wave_lane_carries_full_resolve_record(self) -> None:
        # Each lane is the flat resolve shape a front door pins (config_dir / oauth_token /
        # tier), plus the pool key for distinctness auditing.
        d = login_dir(self.home, ".claude-gem8-acct", uuid="uuid-8", email="gem8@x.ai")
        (d / ".oauth-token").write_text("tok-8\n", encoding="utf-8")
        w = fleet_accounts.allocate_wave(
            1, task_class="t1", home=str(self.home),
            config_home=str(self.config_home), registry={})
        lane = w["lanes"][0]
        for key in ("ok", "account", "tag", "product", "config_dir", "oauth_token",
                    "selected_tier", "target_tier", "fallback_used", "pool"):
            self.assertIn(key, lane)
        self.assertEqual(lane["config_dir"], str(d))
        self.assertEqual(lane["oauth_token"], "tok-8")
        self.assertEqual(lane["selected_tier"], 1)
        self.assertEqual(lane["pool"], "uuid:uuid-8")
        self.assertEqual(lane["session_slot"], 1)
        self.assertEqual(lane["session_cap"], 4)

    def test_wave_skips_blocked_pool(self) -> None:
        self._three_distinct()
        reg = {"throttle": {".claude-gem7-acct": {"reset": "Dec 31, 11:59pm"}}}
        w = fleet_accounts.allocate_wave(
            9, task_class="t1", home=str(self.home),
            config_home=str(self.config_home), registry=reg)
        self.assertEqual(w["granted"], 9, "healthy pools still satisfy the requested wave")
        lanes_by_tag = {lane["tag"] for lane in w["lanes"]}
        self.assertNotIn("gem7", lanes_by_tag, "the throttled pool contributes zero slots")
        blocked = [b for b in w["blocked_target_accounts"] if b.get("tag") == "gem7"]
        self.assertEqual(len(blocked), 1)
        self.assertTrue(blocked[0]["reason"].startswith("usage limit"))

    def test_wave_respects_product_filter(self) -> None:
        login_dir(self.home, ".claude-gem8-acct", uuid="uuid-8", email="gem8@x.ai")
        opencode_dir(self.config_home, "opencode",
                     config={"model": "zai-coding-plan/glm-5.2"})
        claude_only = fleet_accounts.allocate_wave(
            5, task_class="t1", product="claude", home=str(self.home),
            config_home=str(self.config_home), registry={})
        self.assertEqual({lane["product"] for lane in claude_only["lanes"]}, {"claude"})

    def test_wave_zero_count_is_not_ok(self) -> None:
        self._three_distinct()
        w = fleet_accounts.allocate_wave(
            0, task_class="t1", home=str(self.home),
            config_home=str(self.config_home), registry={})
        self.assertFalse(w["ok"])
        self.assertEqual(w["granted"], 0)

    def test_wave_stamps_rank_waveid_size_membership(self) -> None:
        # The typed-group identity: each granted lane carries rank 0..granted-1, a
        # shared wave_id, and size == granted; the under-fill is recorded once.
        self._three_distinct()
        w = fleet_accounts.allocate_wave(
            5, task_class="t1", home=str(self.home),
            config_home=str(self.config_home), registry={})
        self.assertEqual(w["granted"], 5)
        self.assertEqual(w["size"], 5)
        self.assertEqual(w["shortfall"], 0)
        self.assertTrue(w["wave_id"].startswith("wave-"))
        ranks = [lane["rank"] for lane in w["lanes"]]
        self.assertEqual(ranks, [0, 1, 2, 3, 4])       # ranks 0..granted-1, in order
        self.assertEqual({lane["wave_id"] for lane in w["lanes"]}, {w["wave_id"]})
        self.assertEqual({lane["size"] for lane in w["lanes"]}, {5})

    def test_wave_id_is_deterministic_and_content_addressed(self) -> None:
        # Same roster -> same wave_id (deterministic, no clock/random); an explicit
        # wave_id overrides the derived one.
        self._three_distinct()
        kw = dict(task_class="t1", home=str(self.home),
                  config_home=str(self.config_home), registry={})
        a = fleet_accounts.allocate_wave(3, **kw)
        b = fleet_accounts.allocate_wave(3, **kw)
        self.assertEqual(a["wave_id"], b["wave_id"])
        pinned = fleet_accounts.allocate_wave(3, wave_id="W-42", **kw)
        self.assertEqual(pinned["wave_id"], "W-42")
        self.assertEqual({lane["wave_id"] for lane in pinned["lanes"]}, {"W-42"})


class LaunchWaveDetachedScriptTest(unittest.TestCase):
    """Contract tests for the PowerShell detached-wave planner.

    The script is the high-throughput front door over ``fleet-accounts wave``. These
    tests replace ``fak`` with a tiny PowerShell fixture, so no account roster,
    preflight probe, or worker launch is touched.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.fake_fak = self.tmp / "fake-fak.ps1"
        self.fake_fak.write_text(
            """
$Rest = $args
if ($Rest -contains '-h' -or $Rest -contains '--help') {
  Write-Output 'Usage: fak fleet-accounts wave'
  Write-Output '  --count N'
  exit 0
}
if ($Rest -contains 'definitely-no-such-product') {
  [ordered]@{
    ok = $false
    requested = 1
    granted = 0
    shortfall = 1
    distinct_pools = 0
    size = 0
    wave_id = 'wave-empty'
    target_tier = 1
    reason = 'no available account for a wave'
    lanes = @()
  } | ConvertTo-Json -Depth 12
  exit 0
}
[ordered]@{
  ok = $true
  requested = 2
  granted = 2
  shortfall = 0
  distinct_pools = 2
  size = 2
  wave_id = 'wave-test'
  target_tier = 1
  reason = 'granted 2 session slot(s) across 2 distinct pool(s)'
  lanes = @(
    [ordered]@{
      tag = 'acct-a'
      selected_tier = 1
      session_slot = 1
      session_cap = 4
      pool = 'uuid:a'
      config_dir = 'C:\\fake\\acct-a'
      rank = 0
      wave_id = 'wave-test'
      size = 2
    },
    [ordered]@{
      tag = 'acct-b'
      selected_tier = 1
      session_slot = 1
      session_cap = 4
      pool = 'uuid:b'
      config_dir = 'C:\\fake\\acct-b'
      rank = 1
      wave_id = 'wave-test'
      size = 2
    }
  )
} | ConvertTo-Json -Depth 12
exit 0
""",
            encoding="utf-8",
        )
        self.addCleanup(self._tmp.cleanup)

    def _powershell(self) -> str:
        exe = shutil.which("powershell") or shutil.which("pwsh")
        if not exe:
            self.skipTest("PowerShell unavailable")
        return exe

    def _run_launcher(self, *, product: str = "claude", launch: bool = False) -> subprocess.CompletedProcess[str]:
        cmd = [self._powershell(), "-NoProfile", "-NonInteractive"]
        if os.name == "nt":
            cmd += ["-ExecutionPolicy", "Bypass"]
        cmd += [
            "-File", str(ROOT / "tools" / "launch_wave_detached.ps1"),
            "-Count", "2",
            "-SkipPreflight",
            "-Json",
            "-FakExe", str(self.fake_fak),
            "-Workspace", str(ROOT),
            "-PointerFile", ".claude/goal-prompts/resolve-top-issue-witnessed.md",
            "-Product", product,
        ]
        if launch:
            cmd.append("-Launch")
        return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=30)

    def _json_plan(self, **kwargs: object) -> dict:
        proc = self._run_launcher(**kwargs)
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        return json.loads(proc.stdout)

    def test_json_plan_refuses_launch_flag_before_any_spawn_path(self) -> None:
        proc = self._run_launcher(launch=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("-Json is a dry-run plan format", proc.stderr + proc.stdout)

    def test_json_plan_preserves_wave_membership_and_verdict_fields(self) -> None:
        doc = self._json_plan()
        self.assertTrue(doc["ok"])
        self.assertEqual(doc["verdict"], "WOULD_WAVE")
        self.assertEqual(doc["action"], "would_wave")
        self.assertFalse(doc["live"])
        self.assertFalse(doc["launch"])
        self.assertEqual(doc["size"], 2)
        self.assertEqual(doc["wave_id"], "wave-test")
        self.assertEqual(doc["granted"], 2)
        self.assertEqual(doc["distinct_pools"], 2)
        self.assertEqual([lane["rank"] for lane in doc["lanes"]], [0, 1])
        self.assertEqual({lane["wave_id"] for lane in doc["lanes"]}, {"wave-test"})
        self.assertEqual({lane["size"] for lane in doc["lanes"]}, {2})

    def test_json_plan_returns_structured_account_refusal(self) -> None:
        doc = self._json_plan(product="definitely-no-such-product")
        self.assertFalse(doc["ok"])
        self.assertEqual(doc["verdict"], "WAVE_NO_SEATS")
        self.assertEqual(doc["action"], "no_seats")
        self.assertEqual(doc["allocation_requested"], 2)
        self.assertEqual(doc["granted"], 0)
        self.assertEqual(doc["reason"], "no available account for a wave")


class IdentityReconciliationTest(unittest.TestCase):
    """The roster must see WHO each dir is logged into, not just its name -- so N dirs
    on one Anthropic account collapse to one routable worker."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.config_home = self.home / "config"
        self.config_home.mkdir()
        self.addCleanup(self._tmp.cleanup)

    def _discover(self):
        # hermetic: a fixed policy (default exclude/include) so a host-local
        # _registry/accounts_policy.json (e.g. a tombstoned c10) can't change roles here.
        policy = {"exclude": ["backup", "breakglass"], "include_only": [],
                  "notes": {}, "account_profiles": {}, "routing": {}}
        return fleet_accounts.discover_accounts(str(self.home), policy,
                                                config_home=str(self.config_home))

    def test_reads_login_identity(self) -> None:
        login_dir(self.home, ".claude-gem8-acct", uuid="uuid-a",
                  email="jack@x.ai")
        by = {r["account"]: r for r in self._discover()}
        row = by[".claude-gem8-acct"]
        self.assertEqual(row["account_uuid"], "uuid-a")
        self.assertEqual(row["login_email"], "jack@x.ai")
        self.assertEqual(row["identity_role"], "unique")

    def test_three_dirs_one_account_collapse_to_canonical_plus_duplicates(self) -> None:
        # same uuid across three differently-named dirs (the live gem5/gem7/c10 case)
        for name in (".claude-gem5-acct", ".claude-gem7-acct", ".claude-c10-acct"):
            login_dir(self.home, name, uuid="shared", email="agent@x.ai")
        rows = [r for r in self._discover() if r.get("account_uuid") == "shared"]
        roles = sorted(r["identity_role"] for r in rows)
        self.assertEqual(roles, ["canonical", "duplicate", "duplicate"])
        # exactly one canonical, two duplicates, all listing the other two as peers
        canon = [r for r in rows if r["identity_role"] == "canonical"]
        self.assertEqual(len(canon), 1)
        for r in rows:
            self.assertEqual(len(r["identity_peers"]), 2)

    def test_duplicate_excluded_from_routing(self) -> None:
        for name in (".claude-gem5-acct", ".claude-gem7-acct"):
            login_dir(self.home, name, uuid="shared", email="agent@x.ai")
        rows = self._discover()
        routable = [r for r in rows if fleet_accounts.routable_worker(r)]
        shared = [r for r in routable if r.get("account_uuid") == "shared"]
        self.assertEqual(len(shared), 1, "one account must offer exactly one routable dir")

    def test_no_login_dir_is_not_duplicate(self) -> None:
        login_dir(self.home, ".claude-faklocal", uuid="", email="")  # never logged in
        by = {r["account"]: r for r in self._discover()}
        row = by[".claude-faklocal"]
        self.assertEqual(row["identity_role"], "no-login")
        self.assertTrue(fleet_accounts.routable_worker(row),
                        "a local/no-login worker is still routable, just not Claude-auth'd")

    def test_tag_login_mismatch_flagged(self) -> None:
        # dir named gem5 but logged in as agent@ -> mismatch
        login_dir(self.home, ".claude-gem5-acct", uuid="u", email="agent@x.ai")
        # dir named gem8 logged in as gem8@ -> match
        login_dir(self.home, ".claude-gem8-acct", uuid="v", email="gem8@x.ai")
        by = {r["tag"]: r for r in self._discover()}
        self.assertFalse(by["gem5"]["tag_login_match"])
        self.assertTrue(by["gem8"]["tag_login_match"])

    def test_name_matched_dir_beats_default_for_canonical(self) -> None:
        # gem5 dir holding gem5@ and the default dir ALSO holding gem5@: the purpose-named
        # gem5 dir must be canonical; 'default' (which may legitimately hold any account)
        # is the duplicate, never the other way round.
        login_dir(self.home, ".claude-gem5-acct", uuid="shared", email="gem5@x.ai")
        login_dir(self.home, ".claude", uuid="shared", email="gem5@x.ai")
        by = {r["tag"]: r for r in self._discover()}
        self.assertEqual(by["gem5"]["identity_role"], "canonical")
        self.assertEqual(by["default"]["identity_role"], "duplicate")

    def test_distinct_accounts_stay_unique(self) -> None:
        login_dir(self.home, ".claude-gem8-acct", uuid="a", email="jack@x.ai")
        login_dir(self.home, ".claude", uuid="b", email="gem5@x.ai")
        by = {r["tag"]: r for r in self._discover()}
        self.assertEqual(by["gem8"]["identity_role"], "unique")
        self.assertEqual(by["default"]["identity_role"], "unique")

    def test_huge_config_does_not_crash_discovery(self) -> None:
        # a 40KB+ .claude.json (like a heavily-used account) must still parse identity
        path = login_dir(self.home, ".claude-big-acct", uuid="big", email="b@x.ai")
        cfg = json.loads((path / ".claude.json").read_text(encoding="utf-8"))
        cfg["junk"] = "x" * 60000
        (path / ".claude.json").write_text(json.dumps(cfg), encoding="utf-8")
        by = {r["account"]: r for r in self._discover()}
        self.assertEqual(by[".claude-big-acct"]["account_uuid"], "big")

    def test_opencode_has_no_identity_fields(self) -> None:
        opencode_dir(self.config_home, "opencode-glm", config={"model": "z/glm"})
        by = {r["account"]: r for r in self._discover()}
        row = by["opencode-glm"]
        # opencode workers carry no Claude oauth identity; must not be mislabeled
        self.assertNotEqual(row.get("identity_role"), "duplicate")

    def test_cli_available_and_json_exclude_duplicate_identity(self) -> None:
        # The `u`/switcher surfaces (CLI `available` + `json`) must offer EXACTLY what the
        # router routes to: a duplicate-identity dir is the same Anthropic account as its
        # canonical sibling, so offering it double-counts one account's capacity. Both CLI
        # modes must filter through routable_worker(), like available_accounts()/route do.
        login_dir(self.home, ".claude-gem5-acct", uuid="shared", email="gem5@x.ai")
        login_dir(self.home, ".claude", uuid="shared", email="gem5@x.ai")  # default = dup
        # explicit empty registry so a stale host throttle (gem5 is usage-limited on the
        # real box) can't bleed in and mark the canonical dir unavailable.
        rows = fleet_accounts.annotate_accounts(self._discover(), registry={})
        by = {r["account"]: r for r in rows}
        self.assertEqual(by[".claude"]["identity_role"], "duplicate")
        self.assertTrue(by[".claude"]["available"],
                        "the duplicate is healthy; only its duplicate-ness must hide it")

        with mock.patch.object(fleet_accounts, "discover_accounts",
                               return_value=self._discover()), \
             mock.patch.object(fleet_accounts, "annotate_accounts", return_value=rows):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                fleet_accounts.main(["available"])
            available = out.getvalue().split()

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                fleet_accounts.main(["json"])
            doc = json.loads(out.getvalue())

        offered = {r["account"] for r in doc["available_accounts"]}
        # canonical offered on both surfaces; duplicate suppressed on both
        self.assertIn(".claude-gem5-acct", available)
        self.assertIn(".claude-gem5-acct", offered)
        self.assertNotIn(".claude", available,
                         "CLI `available` must not offer a duplicate-identity dir")
        self.assertNotIn(".claude", offered,
                         "CLI `json` available_accounts must not offer a duplicate dir")


class ResetIsFutureTests(unittest.TestCase):
    """Bare daily resets ("3pm", "12:30am") belong to a ~5h rolling window, so a
    bare time already PAST today has reset -- it does NOT mean "tomorrow". The old
    heuristic rolled any pre-6am time a full day forward when observed after noon,
    so an already-passed "12:30am" read as future and the account stayed falsely
    throttled (the resume-resolver then re-homed off a healthy account)."""

    import datetime as _dt
    try:
        from zoneinfo import ZoneInfo as _ZoneInfo
        LA = _ZoneInfo("America/Los_Angeles")
    except Exception:  # pragma: no cover
        LA = _dt.timezone.utc

    def _at(
        self, h: int, m: int = 0, *, month: int = 6, day: int = 24
    ) -> _dt.datetime:
        return self._dt.datetime(2026, month, day, h, m, tzinfo=self.LA)

    def test_passed_early_morning_reset_is_expired(self) -> None:
        # THE BUG: "12:30am" seen at 1:22pm is ~13h in the past -> reset, not tomorrow.
        self.assertFalse(
            fleet_accounts._reset_is_future("12:30am (America/Los_Angeles)",
                                            self._at(13, 22)))

    def test_future_same_day_reset_is_future(self) -> None:
        self.assertTrue(
            fleet_accounts._reset_is_future("3pm (America/Los_Angeles)",
                                            self._at(13, 22)))

    def test_just_passed_bare_reset_is_expired(self) -> None:
        # "1pm" seen at 1:22pm has passed; tomorrow's 1pm is ~23h away (outside the
        # daily window) -> the limit has reset.
        self.assertFalse(
            fleet_accounts._reset_is_future("1pm (America/Los_Angeles)",
                                            self._at(13, 22)))

    def test_late_night_reset_rolls_into_window(self) -> None:
        # "12:30am" seen at 10pm -> tomorrow 12:30am is 2.5h away, inside the daily
        # window -> still a live future reset.
        self.assertTrue(
            fleet_accounts._reset_is_future("12:30am (America/Los_Angeles)",
                                            self._at(22, 0)))

    def test_dated_weekly_reset_future_and_past(self) -> None:
        self.assertTrue(
            fleet_accounts._reset_is_future("Jun 25, 1pm (America/Los_Angeles)",
                                            self._at(13, 22)))
        self.assertTrue(
            fleet_accounts._reset_is_future("Mon Jun 25 at 1pm (America/Los_Angeles)",
                                            self._at(13, 22)))
        self.assertFalse(
            fleet_accounts._reset_is_future("Jun 23, 8pm (America/Los_Angeles)",
                                            self._at(13, 22)))
        self.assertFalse(
            fleet_accounts._reset_is_future("Jan 1, 1am (America/Los_Angeles)",
                                            self._at(13, 22, month=7, day=1)))

    def test_throttle_is_active_delegates_to_reset(self) -> None:
        # throttle_is_active is "active unless the reset parses as expired": a clearly
        # past DATED reset clears it; an unparseable reset stays active (fail-safe).
        now = self._at(13, 22, month=7, day=1)
        self.assertFalse(fleet_accounts.throttle_is_active(
            {"reset": "Jan 1, 1am (America/Los_Angeles)"}, now))  # long past -> expired
        self.assertTrue(fleet_accounts.throttle_is_active(
            {"reset": "sometime never"}, now))  # unparseable -> stay active
        self.assertTrue(fleet_accounts.throttle_is_active(
            {"reset": "Dec 31, 11pm (America/Los_Angeles)"}, now))  # year-end -> future


class ProbeLedgerConsultTest(unittest.TestCase):
    """runtime_status consults account_probe's probe_ledger.jsonl as a fresh-probe source.

    The bug this guards: account_probe writes its OK/LIMIT verdict ONLY to the probe ledger,
    a file distinct from the watchdog's sessions.json that runtime_status reads. Nothing folded
    the ledger back into the registry, so a fresh probe was invisible to the roster -- the day24
    incident, where a live OK probe left the account still reading "resets 11pm" and the resume
    resolver returned PIN_BLOCKED with a healthy worker available."""

    ACCT = ".claude-day24-acct"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.reg_dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        fleet_accounts._PROBE_LEDGER_CACHE.update(key=None, by_account={}, ages={})
        self.addCleanup(lambda: fleet_accounts._PROBE_LEDGER_CACHE.update(
            key=None, by_account={}, ages={}))

    def _write_ledger(self, *, status: str, age_min: float, **extra) -> None:
        import datetime as _dt
        ts = (_dt.datetime.now(_dt.timezone.utc)
              - _dt.timedelta(minutes=age_min)).isoformat()
        entry = {"ts": ts, "account": self.ACCT, "tag": "day24",
                 "status": status, **extra}
        (self.reg_dir / "probe_ledger.jsonl").write_text(
            json.dumps(entry) + "\n", encoding="utf-8")

    def _carried_throttle_registry(self, info: dict | None = None) -> dict:
        throttle_info = {"reset": "Dec 31, 11pm (America/Los_Angeles)"}
        throttle_info.update(info or {})
        return {"generated_utc": "2026-06-17T00:00:00+00:00",
                "throttle": {self.ACCT: throttle_info}, "sessions": []}

    def test_fresh_ok_probe_overrides_carried_throttle(self) -> None:
        self._write_ledger(status="OK", age_min=5.0)
        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(self.reg_dir)}, clear=False):
            status = fleet_accounts.runtime_status(
                self.ACCT, registry=self._carried_throttle_registry())
        self.assertTrue(status["available"], "a fresh OK probe must clear the carried throttle")
        self.assertEqual(status["status_source"], "probe-ledger")

    def test_stale_ok_probe_does_not_override_throttle(self) -> None:
        self._write_ledger(status="OK", age_min=fleet_accounts.PROBE_LEDGER_FRESH_MIN + 30)
        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(self.reg_dir)}, clear=False):
            status = fleet_accounts.runtime_status(
                self.ACCT, registry=self._carried_throttle_registry())
        self.assertFalse(status["available"], "a stale OK probe must not clear the throttle")
        self.assertEqual(status["block_kind"], "usage")

    def test_fresh_ok_probe_does_not_override_future_weekly_throttle(self) -> None:
        self._write_ledger(status="OK", age_min=5.0)
        registry = self._carried_throttle_registry({
            "weekly": "Dec 31, 11pm (America/Los_Angeles)",
        })
        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(self.reg_dir)}, clear=False):
            status = fleet_accounts.runtime_status(self.ACCT, registry=registry)

        self.assertFalse(status["available"])
        self.assertEqual(status["block_kind"], "usage")
        self.assertEqual(status["weekly"], "Dec 31, 11pm (America/Los_Angeles)")
        self.assertEqual(status["status_source"], "registry")

    def test_fresh_limit_probe_blocks_even_without_carried_throttle(self) -> None:
        self._write_ledger(status="LIMIT", age_min=3.0, reset="9pm (America/Los_Angeles)")
        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(self.reg_dir)}, clear=False):
            status = fleet_accounts.runtime_status(
                self.ACCT, registry={"generated_utc": "2026-06-17T00:00:00+00:00",
                                     "throttle": {}, "sessions": []})
        self.assertFalse(status["available"])
        self.assertEqual(status["block_kind"], "usage")
        self.assertEqual(status["status_source"], "probe-ledger")

    def test_probe_session_row_takes_precedence_over_ledger(self) -> None:
        self._write_ledger(status="LIMIT", age_min=3.0)  # ledger says blocked...
        registry = {"generated_utc": "2026-06-17T00:00:00+00:00", "throttle": {},
                    "sessions": [{"account": self.ACCT, "project": "_probe",
                                  "probe_status": "OK"}]}  # ...but a fresh session-row probe says OK
        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(self.reg_dir)}, clear=False):
            status = fleet_accounts.runtime_status(self.ACCT, registry=registry)
        self.assertTrue(status["available"])
        self.assertEqual(status["status_source"], "probe")

    def test_fresh_ok_ledger_clears_weekly_throttle_after_account_identity_change(self) -> None:
        self._write_ledger(status="OK", age_min=5.0)
        home = self.reg_dir / "home"
        home.mkdir()
        login_dir(home, self.ACCT, uuid="new-account", email="new@example.test")
        registry = self._carried_throttle_registry({
            "reset": "Dec 31, 11pm (America/Los_Angeles)",
            "weekly": "Dec 31, 11pm (America/Los_Angeles)",
            "account_uuid": "old-account",
            "login_email": "old@example.test",
        })
        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(self.reg_dir)}, clear=False), \
             mock.patch.object(fleet_accounts, "USER", str(home)):
            status = fleet_accounts.runtime_status(self.ACCT, registry=registry)

        self.assertTrue(status["available"])
        self.assertFalse(status["blocked"])
        self.assertEqual(status["status_source"], "probe-ledger")

    def test_fresh_ok_probe_surfaces_active_daily_cap_as_usage_soon(self) -> None:
        # Mirror of Go TestFreshProbeSurfacesActiveDailyCapAsUsageSoon: a fresh OK probe reopens
        # a seat whose DAILY cap is still counting down, but carries the still-future reset as an
        # advisory usage_soon_reset instead of silently dropping it -- so the roster can render
        # "serving, cap resets X" rather than a blank near-cap row. Availability is unchanged.
        self._write_ledger(status="OK", age_min=5.0)
        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(self.reg_dir)}, clear=False):
            status = fleet_accounts.runtime_status(
                self.ACCT, registry=self._carried_throttle_registry())
        self.assertTrue(status["available"], "the seat stays offered")
        self.assertFalse(status.get("throttled"))
        self.assertEqual(status["status_source"], "probe-ledger")
        self.assertEqual(status.get("usage_soon_reset"),
                         "Dec 31, 11pm (America/Los_Angeles)")

    def test_serving_seat_without_cap_has_no_usage_soon(self) -> None:
        # The additive contract: a serving seat with no carried cap gains no key, so every
        # existing row's byte-parity JSON is unchanged.
        self._write_ledger(status="OK", age_min=5.0)
        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(self.reg_dir)}, clear=False):
            status = fleet_accounts.runtime_status(
                self.ACCT, registry={"generated_utc": "2026-06-17T00:00:00+00:00",
                                     "throttle": {}, "sessions": []})
        self.assertTrue(status["available"])
        self.assertNotIn("usage_soon_reset", status)

    def test_active_weekly_cap_is_walled_not_usage_soon(self) -> None:
        # Boundary: a still-active WEEKLY cap holds the seat CLOSED rather than surfacing a
        # near-cap advisory (mirror of Go TestActiveWeeklyCapIsWalledNotUsageSoon).
        self._write_ledger(status="OK", age_min=5.0)
        registry = self._carried_throttle_registry({
            "weekly": "Dec 31, 11pm (America/Los_Angeles)",
        })
        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(self.reg_dir)}, clear=False):
            status = fleet_accounts.runtime_status(self.ACCT, registry=registry)
        self.assertFalse(status["available"])
        self.assertNotIn("usage_soon_reset", status)


class LedgerGateGradeTest(unittest.TestCase):
    """#5439's Python half: the probe-ledger rung is gated on what the resolved registry can
    DERIVE, not on FLEET_REG_DIR merely being SET, and a registry that can derive nothing
    publishes unknown-health instead of a proven-free seat.

    Mirror of internal/fleetaccounts/status_ledgergate_test.go. The Go twin has to pin every
    rung accountprobe.ResolveRegDir surveys (LOCALAPPDATA, TMP, ...) so its verdict is not a
    fact about the operator's laptop; account_probe.reg_dir now walks the SAME ladder via
    fleet_regdir, so the env rung -- which outranks every discovery rung -- is what pins it
    outright here, except in the one case that deliberately leaves it unset and patches the
    resolver instead."""

    ACCT = ".claude-gate5439"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        fleet_accounts._PROBE_LEDGER_CACHE.update(key=None, by_account={}, ages={}, obs={})
        self.addCleanup(lambda: fleet_accounts._PROBE_LEDGER_CACHE.update(
            key=None, by_account={}, ages={}, obs={}))

    def _reg_dir(self, name: str, *, ledger: str | None) -> Path:
        """A registry dir carrying sessions.json, and probe_ledger.jsonl iff `ledger` is not
        None (`""` = a wired prober that has recorded nothing)."""
        d = self.root / name
        d.mkdir()
        (d / "sessions.json").write_text("{}", encoding="utf-8")
        if ledger is not None:
            (d / "probe_ledger.jsonl").write_text(ledger, encoding="utf-8")
        return d

    def _live_registry(self) -> dict:
        return {"generated_utc": dt.datetime.now(dt.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
                "throttle": {},
                "sessions": [{"account": self.ACCT, "project": "work", "disp": "LIVE"}]}

    def _capped_registry(self) -> dict:
        return {"generated_utc": "2026-06-17T00:00:00+00:00",
                "throttle": {self.ACCT: {"reset": "Dec 31, 11pm (America/Los_Angeles)"}},
                "sessions": []}

    def test_gate_grades_derivability_not_the_env_var(self) -> None:
        bearing = self._reg_dir("bearing", ledger="")
        ledger_less = self._reg_dir("ledgerless", ledger=None)
        reg = self._live_registry()
        for d, want in ((bearing, True), (ledger_less, False)):
            with self.subTest(dir=d.name):
                with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(d)}, clear=False):
                    self.assertIs(fleet_accounts._registry_blocks_derivable(), want)
                    self.assertIs(
                        fleet_accounts._should_consult_probe_ledger(reg, None), want,
                        "the rung must follow derivability, not FLEET_REG_DIR being set")

    def test_explicit_probe_ledger_argument_still_wins(self) -> None:
        # The caller affordance the Go twin has no equivalent of: an explicit override
        # outranks the filesystem, so every hermetic caller keeps its pinned verdict.
        ledger_less = self._reg_dir("ledgerless", ledger=None)
        reg = self._live_registry()
        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(ledger_less)}, clear=False):
            self.assertTrue(fleet_accounts._should_consult_probe_ledger(reg, True))
            self.assertFalse(fleet_accounts._should_consult_probe_ledger(reg, False))

    def test_ledger_rung_runs_when_derivable_without_reg_dir(self) -> None:
        # The acceptance line: the rung runs whenever blocks are DERIVABLE, whether or not
        # FLEET_REG_DIR is set. Under the old env-var gate the rung never ran here, so a fresh
        # OK probe stayed invisible and the carried throttle won.
        import account_probe
        ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)).isoformat()
        bearing = self._reg_dir("bearing", ledger=json.dumps(
            {"ts": ts, "account": self.ACCT, "tag": "gate", "status": "OK"}) + "\n")
        with mock.patch.dict(os.environ, {}, clear=False), \
                mock.patch.object(account_probe, "reg_dir", lambda: str(bearing)):
            os.environ.pop("FLEET_REG_DIR", None)
            status = fleet_accounts.runtime_status(
                self.ACCT, registry=self._capped_registry())
        self.assertTrue(status["available"],
                        "a derivable registry's fresh ledger OK must clear the carried"
                        " throttle even with FLEET_REG_DIR unset")
        self.assertEqual(status["status_source"], "probe-ledger")

    def test_ledger_rung_skipped_when_reg_dir_cannot_derive_blocks(self) -> None:
        # The one case whose verdict the change FLIPS: FLEET_REG_DIR is set, so the old gate
        # turned the rung on, but it names a dir with no ledger beside its sessions.json.
        # Nothing is lost by refusing -- every read under that dir returned nothing anyway.
        ledger_less = self._reg_dir("ledgerless", ledger=None)
        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(ledger_less)}, clear=False):
            status = fleet_accounts.runtime_status(
                self.ACCT, registry=self._capped_registry())
        self.assertFalse(status["available"])
        self.assertTrue(status["throttled"])
        # A blocked seat keeps the confident source: "blocked" was derived from the registry's
        # own throttle row, so its provenance is not in doubt.
        self.assertEqual(status["status_source"], "registry")

    def test_underivable_registry_publishes_unknown_health(self) -> None:
        # The second acceptance line. Both halves asserted on purpose: the seat stays OFFERED
        # and only the CLAIM is weakened. Converting absence into a block would be
        # self-sealing -- the roster routes the work that runs the prober.
        ledger_less = self._reg_dir("ledgerless", ledger=None)
        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(ledger_less)}, clear=False):
            status = fleet_accounts.runtime_status(
                self.ACCT, registry=self._live_registry())
        self.assertTrue(status["available"], "unknown health must not strand the seat")
        self.assertFalse(status["blocked"])
        self.assertEqual(status["status_source"], "registry-unknown")

    def test_derivable_registry_keeps_registry_status_source(self) -> None:
        # The control that keeps the new state from being a blanket rename.
        bearing = self._reg_dir("bearing", ledger="")
        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(bearing)}, clear=False):
            status = fleet_accounts.runtime_status(
                self.ACCT, registry=self._live_registry())
        self.assertTrue(status["available"])
        self.assertEqual(status["status_source"], "registry")

    def test_empty_registry_keeps_none_status_source(self) -> None:
        # The boundary on the other side: "none" already says nothing was consulted, and is
        # the more precise of the two statements, so unknown-health must not overwrite it.
        ledger_less = self._reg_dir("ledgerless", ledger=None)
        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(ledger_less)}, clear=False):
            status = fleet_accounts.runtime_status(self.ACCT, registry={})
        self.assertEqual(status["status_source"], "none")


class SeatProbeCoverageTest(unittest.TestCase):
    """#5391's Python half: a seat the prober has not spoken about lately is published as
    unknown-health even though the registry it is published out of derives blocks perfectly
    well for OTHER accounts.

    #5439 (LedgerGateGradeTest above) grades per-REGISTRY. The host that filed #5391 shows why
    that is not sufficient: probe_ledger.jsonl there was present, derivable and busy --
    ``opencode-*`` rows current to the minute -- while several claude seats' newest rows were
    8-9 days old. Every registry-level question answered "yes", and none of it was evidence
    about those seats. Mirror of internal/fleetaccounts/status_seatcoverage_test.go; the two
    surfaces must not disagree about one host, which is the #5390 failure in another costume.

    Both halves are asserted on every case: the weakened claim must never become a block.
    Converting absence into blocked is self-sealing -- the roster routes the work that runs the
    prober -- and #5391 files itself as a coverage gap for exactly that reason."""

    ACCT = ".claude-seat5391"
    OTHER = "opencode-fixture"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        fleet_accounts._PROBE_LEDGER_CACHE.update(key=None, by_account={}, ages={}, obs={})
        self.addCleanup(lambda: fleet_accounts._PROBE_LEDGER_CACHE.update(
            key=None, by_account={}, ages={}, obs={}))

    def _row(self, account: str, minutes_ago: float, status: str = "OK") -> str:
        ts = (dt.datetime.now(dt.timezone.utc)
              - dt.timedelta(minutes=minutes_ago)).isoformat()
        return json.dumps({"ts": ts, "account": account, "tag": "seat", "status": status})

    def _reg_dir(self, *rows: str) -> Path:
        """A ledger-bearing registry dir carrying `rows` (none => a wired prober that has
        recorded nothing)."""
        d = self.root / "bearing"
        d.mkdir(exist_ok=True)
        (d / "sessions.json").write_text("{}", encoding="utf-8")
        (d / "probe_ledger.jsonl").write_text(
            "".join(r + "\n" for r in rows), encoding="utf-8")
        return d

    def _live_registry(self, throttle: dict | None = None) -> dict:
        return {"generated_utc": dt.datetime.now(dt.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
                "throttle": throttle or {},
                "sessions": [{"account": self.ACCT, "project": "work", "disp": "LIVE"}]}

    def _status(self, reg_dir: Path, registry: dict) -> dict:
        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(reg_dir)}, clear=False):
            return fleet_accounts.runtime_status(self.ACCT, registry=registry)

    def test_busy_ledger_with_no_row_for_seat_publishes_unknown_health(self) -> None:
        # The acceptance line: "never probed" and "probed OK" must be distinguishable. The
        # prober demonstrably ran two minutes ago and has simply never named this seat.
        d = self._reg_dir(self._row(self.OTHER, 2))
        status = self._status(d, self._live_registry())
        self.assertTrue(status["available"], "unknown health must not strand the seat")
        self.assertFalse(status["blocked"])
        self.assertEqual(status["status_source"], "registry-unknown")

    def test_eight_day_old_seat_row_publishes_unknown_health(self) -> None:
        # The observed shape verbatim: one account current to the minute, this one last heard
        # from eight days ago. A row that old is not evidence about now.
        d = self._reg_dir(self._row(self.ACCT, 8 * 24 * 60), self._row(self.OTHER, 2))
        status = self._status(d, self._live_registry())
        self.assertTrue(status["available"])
        self.assertFalse(status["blocked"])
        self.assertEqual(status["status_source"], "registry-unknown")

    def test_seat_row_inside_coverage_budget_keeps_registry_status_source(self) -> None:
        # The control that keeps the new trigger from being a blanket rename. Two windows, and
        # they are not the same window: a two-hour-old OK is past PROBE_LEDGER_FRESH_MIN (so
        # the fold correctly falls through to the registry rung) and well inside
        # SEAT_COVERAGE_MAX_AGE_MIN, so the claim published there is a confident one.
        d = self._reg_dir(self._row(self.ACCT, 120))
        status = self._status(d, self._live_registry())
        self.assertTrue(status["available"])
        self.assertEqual(status["status_source"], "registry")

    def test_blocked_seat_keeps_registry_status_source(self) -> None:
        # Ordering: "blocked" is a positive derivation from the registry's own throttle row,
        # not a statement about probe evidence, so unknown-health has nothing to add.
        d = self._reg_dir(self._row(self.OTHER, 2))
        status = self._status(d, self._live_registry(
            {self.ACCT: {"reset": "Dec 31, 11pm (America/Los_Angeles)"}}))
        self.assertFalse(status["available"])
        self.assertTrue(status["throttled"])
        self.assertEqual(status["status_source"], "registry")

    def test_empty_ledger_keeps_registry_status_source(self) -> None:
        # The boundary #5439 drew and this change does not move: a ledger present but EMPTY
        # leaves every seat unmeasured, yet the registry-level grade already describes that
        # host and a seat-level downgrade would only restate it. Without the busy-ledger
        # precondition this case would flip, so the assertion is load-bearing for it.
        d = self._reg_dir()
        status = self._status(d, self._live_registry())
        self.assertEqual(status["status_source"], "registry")


def _future_reset_str(minutes: float) -> str:
    """A DATED reset string `minutes` from now, in a format _reset_is_future parses (mirror of
    Go futureResetStr's "Jan 2, 3:04pm"). Negative minutes => a provably-passed daily reset."""
    when = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=minutes)
    return when.strftime("%b %d, %I:%M%p")


class CapDisambiguationCycleTest(unittest.TestCase):
    """The Phase-3 cap-disambiguation cycles (aging valve + probe-override), mirrored from Go
    (internal/fleetaccounts/status_capcycles_test.go + capstate_test.go + capobs_test.go).

    fleet_accounts.py is a compatibility shim held byte-parity with the Go account contract, so
    these prove the SAME wiring end-to-end through runtime_status: a _CapObservation derived from
    the probe ledger reaches both _disambiguate_cap seams and flips the seat's status. The cap
    math is unit-tested with an injected clock; the wiring tests drive a real ledger + registry."""

    ACCT = ".claude-capcycle-acct"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.reg_dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        # The ledger snapshot is memoized on the file's (mtime, size); a stale cache from another
        # test would mask a fresh write, so reset it around each case (as ProbeLedgerConsultTest).
        fleet_accounts._PROBE_LEDGER_CACHE.update(key=None, by_account={}, ages={}, obs={})
        self.addCleanup(lambda: fleet_accounts._PROBE_LEDGER_CACHE.update(
            key=None, by_account={}, ages={}, obs={}))

    def _write_ledger(self, entries: list[dict]) -> None:
        """Append-ordered probe ledger; each entry is {status, age_min, **extra}."""
        lines = []
        for e in entries:
            ts = (dt.datetime.now(dt.timezone.utc)
                  - dt.timedelta(minutes=e["age_min"])).isoformat()
            row = {"ts": ts, "account": self.ACCT, "tag": "capcycle",
                   "status": e["status"]}
            row.update({k: v for k, v in e.items() if k not in ("status", "age_min")})
            lines.append(json.dumps(row))
        (self.reg_dir / "probe_ledger.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")

    def _status(self, throttle: dict) -> dict:
        registry = {"generated_utc": "2026-06-17T00:00:00+00:00",
                    "throttle": {self.ACCT: throttle}, "sessions": []}
        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(self.reg_dir)}, clear=False):
            return fleet_accounts.runtime_status(self.ACCT, registry=registry)

    # --- wiring: aging valve at the carried-throttle seam -------------------------------------
    def test_ledger_aging_reopens_stale_weekly(self) -> None:
        # Mirror of Go TestLedgerAgingReopensStaleWeekly: an 8-day-old LIMIT episode with an
        # unparseable weekly and no daily leg has outlived any real weekly window -> reopen.
        # Deterministic: the episode age is read from the ledger ts, not the wall clock.
        self._write_ledger([{"status": "LIMIT", "age_min": 8 * 24 * 60,
                             "weekly": "sometime never"}])
        st = self._status({"weekly": "sometime never"})
        self.assertTrue(st["available"], f"an 8-day episode must age out and reopen: {st}")
        self.assertFalse(st["blocked"])
        self.assertFalse(st.get("throttled"))

    def test_ledger_aging_holds_young_weekly(self) -> None:
        # Age-threshold control (Go TestLedgerAgingHoldsYoungWeekly): 3 days in is still within
        # _WEEKLY_MAX_AGE, so aging must not fire early and the seat stays walled.
        self._write_ledger([{"status": "LIMIT", "age_min": 3 * 24 * 60,
                             "weekly": "sometime never"}])
        st = self._status({"weekly": "sometime never"})
        self.assertFalse(st["available"], f"a 3-day episode is within the window: {st}")
        self.assertTrue(st["blocked"])
        self.assertTrue(st.get("throttled"))

    # --- wiring: probe-override at the fresh-OK seam ------------------------------------------
    def test_ledger_ok_streak_overrides_stale_weekly(self) -> None:
        # Mirror of Go TestLedgerOKStreakOverridesStaleWeekly: two consecutive OKs past a passed
        # daily reset overturn a stale/unparseable weekly the seat has demonstrably outgrown.
        self._write_ledger([
            {"status": "OK", "age_min": 12.0},
            {"status": "OK", "age_min": 3.0},  # latest is fresh -> led.available
        ])
        st = self._status({"reset": _future_reset_str(minutes=-30),  # provably-passed daily
                           "weekly": "sometime never"})              # fail-closed without streak
        self.assertTrue(st["available"], f"a 2-OK streak must overturn the stale weekly: {st}")
        self.assertFalse(st["blocked"])
        self.assertFalse(st.get("throttled"))
        self.assertEqual(st["status_source"], "probe-ledger")

    def test_ledger_single_ok_holds_unparseable_weekly(self) -> None:
        # Streak-threshold control (Go TestLedgerSingleOKHoldsUnparseableWeekly): a lone fresh OK
        # (streak 1) is not enough to overturn the fail-closed weekly, so the seat stays walled.
        self._write_ledger([{"status": "OK", "age_min": 3.0}])
        st = self._status({"reset": _future_reset_str(minutes=-30),
                           "weekly": "sometime never"})
        self.assertFalse(st["available"], f"a lone fresh OK must not overturn the weekly: {st}")
        self.assertTrue(st["blocked"])
        self.assertTrue(st.get("throttled"))

    # --- unit: the cap math with an injected clock (mirror capstate_test.go) ------------------
    def test_disambiguate_cap_aging_releases_stale_weekly(self) -> None:
        now = dt.datetime(2026, 6, 20, tzinfo=dt.timezone.utc)
        obs = fleet_accounts._CapObservation(
            first_seen=now - dt.timedelta(days=8), ok_streak=0)
        cs = fleet_accounts._disambiguate_cap({"weekly": "sometime never"}, obs, now)
        self.assertTrue(cs.aged_out)
        self.assertFalse(cs.weekly_active)
        self.assertFalse(cs.active, "no live daily leg -> the aged-out seat is fully released")

    def test_disambiguate_cap_young_weekly_holds(self) -> None:
        now = dt.datetime(2026, 6, 20, tzinfo=dt.timezone.utc)
        obs = fleet_accounts._CapObservation(
            first_seen=now - dt.timedelta(days=3), ok_streak=0)
        cs = fleet_accounts._disambiguate_cap({"weekly": "sometime never"}, obs, now)
        self.assertFalse(cs.aged_out)
        self.assertTrue(cs.weekly_active)
        self.assertTrue(cs.active)

    def test_disambiguate_cap_probe_override(self) -> None:
        now = dt.datetime(2026, 6, 20, 12, 0, tzinfo=dt.timezone.utc)
        passed = (now - dt.timedelta(minutes=30)).strftime("%b %d, %I:%M%p")
        obs = fleet_accounts._CapObservation(ok_streak=2)
        cs = fleet_accounts._disambiguate_cap(
            {"reset": passed, "weekly": "sometime never"}, obs, now)
        self.assertEqual(cs.overridden_by, 2)
        self.assertFalse(cs.weekly_active)
        self.assertFalse(cs.active)

    def test_disambiguate_cap_zero_obs_equals_legacy_views(self) -> None:
        # The load-bearing invariant: with the zero observation the two views are unchanged, so
        # every non-ledger caller keeps its legacy single-shot behavior.
        now = dt.datetime(2026, 6, 20, 12, 0, tzinfo=dt.timezone.utc)
        for info in ({"weekly": "sometime never"},
                     {"reset": "Dec 31, 11pm"},
                     {"reset": (now - dt.timedelta(minutes=30)).strftime("%b %d, %I:%M%p")},
                     "Dec 31, 11pm", None):
            cs = fleet_accounts._disambiguate_cap(info, None, now)
            self.assertEqual(cs.active, fleet_accounts.throttle_is_active(info, now), info)
            self.assertEqual(cs.weekly_active,
                             fleet_accounts._weekly_throttle_is_active(info, now), info)

    # --- unit: observation derivation from the ledger (mirror capobs_test.go) -----------------
    def test_derive_cap_observation_ok_streak_and_episode(self) -> None:
        base = dt.datetime(2026, 6, 20, tzinfo=dt.timezone.utc)

        def line(status: str, mins: int) -> dict:
            return {"account": self.ACCT, "status": status,
                    "ts": (base + dt.timedelta(minutes=mins)).isoformat()}

        # Blocked episode then a 2-OK recovery tail: streak counts the tail, and first_seen is
        # cleared because the tail is OK (no live episode).
        obs = fleet_accounts._cap_observation_from(
            [line("LIMIT", 0), line("LIMIT", 5), line("OK", 10), line("OK", 15)])
        self.assertEqual(obs.ok_streak, 2)
        self.assertIsNone(obs.first_seen)

        # A trailing blocked run: streak 0, first_seen is the START of that run (the first LIMIT).
        obs2 = fleet_accounts._cap_observation_from(
            [line("OK", 0), line("LIMIT", 5), line("LIMIT", 10)])
        self.assertEqual(obs2.ok_streak, 0)
        self.assertEqual(obs2.first_seen, base + dt.timedelta(minutes=5))

    def test_derive_cap_observation_empty_is_zero(self) -> None:
        obs = fleet_accounts._cap_observation_from([])
        self.assertEqual(obs.ok_streak, 0)
        self.assertIsNone(obs.first_seen)


def _seat_row(tag: str, *, available: bool = True, block_kind: str | None = None,
              block_reason: str = "", throttled: bool = False,
              reset: str | None = None, weekly: str | None = None,
              throttled_since: str | None = None) -> dict:
    """A minimal routable-worker row shaped like an ``annotate_accounts()`` output --
    just the fields ``seat_pool``/``_seat_hold_reason``/``_seat_cooldown`` read."""
    return {
        "kind": "worker", "product": "claude", "account": f".claude-{tag}", "tag": tag,
        "model_tier": 1, "available": available, "blocked": not available,
        "block_kind": block_kind, "block_reason": block_reason,
        "throttled": throttled, "reset": reset, "weekly": weekly,
        "throttled_since": throttled_since,
    }


class SeatInventoryTest(unittest.TestCase):
    """#1799: seat_pool() classifies every seat into the operator-facing
    available/busy/cooling/unavailable vocabulary with a specific hold_reason,
    reusing runtime_status's own block_kind/throttled/reset fields."""

    def test_available_seat_has_no_hold_reason(self) -> None:
        rows = [_seat_row("free1", available=True)]
        pool = fleet_accounts.seat_pool(rows, [])
        seat = pool["seats"][0]
        self.assertEqual(seat["dispatch_state"], "available")
        self.assertEqual(seat["hold_reason"], "")
        self.assertEqual(pool["by_dispatch_state"], {"available": 1})

    def test_leased_seat_is_busy_with_worker_named(self) -> None:
        rows = [_seat_row("leased1", available=True)]
        leases = [{"worker": "w42", "tag": "leased1", "dir": ".claude-leased1"}]
        pool = fleet_accounts.seat_pool(rows, leases)
        seat = pool["seats"][0]
        self.assertEqual(seat["dispatch_state"], "busy")
        self.assertIn("w42", seat["hold_reason"])
        self.assertEqual(pool["by_dispatch_state"], {"busy": 1})

    def test_usage_throttled_seat_is_cooling_with_cooldown_until(self) -> None:
        rows = [_seat_row("cool1", available=False, block_kind="usage",
                          throttled=True, reset="Dec 31, 11pm (America/Los_Angeles)")]
        pool = fleet_accounts.seat_pool(rows, [])
        seat = pool["seats"][0]
        self.assertEqual(seat["dispatch_state"], "cooling")
        self.assertIn("cooldown_until=", seat["hold_reason"])
        self.assertIn("Dec 31, 11pm", seat["hold_reason"])

    def test_throttled_without_reset_is_cooling_rate_limited(self) -> None:
        rows = [_seat_row("cool2", available=False, block_kind="usage", throttled=True)]
        pool = fleet_accounts.seat_pool(rows, [])
        self.assertEqual(pool["seats"][0]["dispatch_state"], "cooling")
        self.assertEqual(pool["seats"][0]["hold_reason"], "rate_limited")

    def test_throttled_seat_records_cooldown_and_is_excluded_from_capacity(self) -> None:
        """#1801 witness: a throttled seat renders a structured cooldown (start, reason,
        next-eligible) AND is excluded from offerable capacity -- so dispatch status
        explains the capacity loss from named fields, not a parsed string."""
        rows = [_seat_row("cool3", available=False, block_kind="usage", throttled=True,
                          reset="Dec 31, 11pm (America/Los_Angeles)",
                          weekly="Jan 3, 11pm",
                          throttled_since="2026-06-30T18:00:00Z")]
        pool = fleet_accounts.seat_pool(rows, [])
        seat = pool["seats"][0]
        self.assertEqual(seat["dispatch_state"], "cooling")
        self.assertEqual(seat["cooldown"], {
            "reason": "usage",
            "since": "2026-06-30T18:00:00Z",
            "until": "Dec 31, 11pm (America/Los_Angeles)",
            "weekly": "Jan 3, 11pm",
        })
        # excluded from available capacity -- the throttled seat is not offerable headroom
        self.assertEqual(pool["free_seats"], 0)
        self.assertTrue(pool["depleted"])
        self.assertEqual(seat["state"], "blocked")
        self.assertFalse(seat["available"])

    def test_cooling_seat_without_recorded_start_has_since_none(self) -> None:
        """Forward-compatible: a throttle record with no recorded start still renders a
        structured cooldown (since=None), so the field is always present to read."""
        rows = [_seat_row("cool4", available=False, block_kind="usage", throttled=True,
                          reset="9pm")]
        seat = fleet_accounts.seat_pool(rows, [])["seats"][0]
        self.assertEqual(seat["cooldown"],
                         {"reason": "usage", "since": None, "until": "9pm"})

    def test_available_and_busy_seats_have_no_cooldown(self) -> None:
        rows = [_seat_row("free9", available=True), _seat_row("busy9", available=True)]
        leases = [{"worker": "w9", "tag": "busy9", "dir": ".claude-busy9"}]
        by_tag = {s["tag"]: s for s in fleet_accounts.seat_pool(rows, leases)["seats"]}
        self.assertIsNone(by_tag["free9"]["cooldown"])
        self.assertEqual(by_tag["busy9"]["dispatch_state"], "busy")
        self.assertIsNone(by_tag["busy9"]["cooldown"])

    def test_auth_blocked_seat_is_unavailable_auth_failed(self) -> None:
        rows = [_seat_row("auth1", available=False, block_kind="auth",
                          block_reason="auth/login required")]
        pool = fleet_accounts.seat_pool(rows, [])
        seat = pool["seats"][0]
        self.assertEqual(seat["dispatch_state"], "unavailable")
        self.assertEqual(seat["hold_reason"], "auth_failed")

    def test_credit_blocked_seat_is_unavailable_credit_exhausted(self) -> None:
        rows = [_seat_row("credit1", available=False, block_kind="credit")]
        pool = fleet_accounts.seat_pool(rows, [])
        self.assertEqual(pool["seats"][0]["dispatch_state"], "unavailable")
        self.assertEqual(pool["seats"][0]["hold_reason"], "credit_exhausted")

    def test_unclassified_block_falls_back_to_no_capacity(self) -> None:
        rows = [_seat_row("nocap1", available=False, block_kind=None, block_reason="")]
        pool = fleet_accounts.seat_pool(rows, [])
        self.assertEqual(pool["seats"][0]["dispatch_state"], "unavailable")
        self.assertEqual(pool["seats"][0]["hold_reason"], "no_capacity")

    def test_all_four_states_together_roll_up_by_dispatch_state(self) -> None:
        rows = [
            _seat_row("free1", available=True),
            _seat_row("leased1", available=True),
            _seat_row("cool1", available=False, block_kind="usage", throttled=True,
                     reset="9pm"),
            _seat_row("auth1", available=False, block_kind="auth"),
        ]
        leases = [{"worker": "w1", "tag": "leased1", "dir": ".claude-leased1"}]
        pool = fleet_accounts.seat_pool(rows, leases)
        states = {s["tag"]: s["dispatch_state"] for s in pool["seats"]}
        self.assertEqual(states, {
            "free1": "available", "leased1": "busy",
            "cool1": "cooling", "auth1": "unavailable",
        })
        self.assertEqual(pool["by_dispatch_state"],
                         {"available": 1, "busy": 1, "cooling": 1, "unavailable": 1})
        # every non-available seat carries a specific reason, never the bare word
        for s in pool["seats"]:
            if s["dispatch_state"] != "available":
                self.assertNotEqual(s["hold_reason"], "unavailable")
                self.assertTrue(s["hold_reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
