#!/usr/bin/env python3
"""Hermetic tests for fleet account policy and runtime availability."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_accounts  # noqa: E402


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


class FleetAccountsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        # a separate, empty XDG-style config root so claude-only tests stay
        # hermetic and never glob the real ~/.config/opencode on the host
        self.config_home = self.home / "config"
        self.config_home.mkdir()
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
        with mock.patch.object(fleet_accounts, "load_registry", side_effect=AssertionError("unexpected load")):
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
    """allocate_wave hands a parallel fan-out N DISTINCT rate-limit pools at once.

    The provable-benefit witness: a fan-out that calls single-account resolve() N
    times in a burst gets the SAME account N times (no session has registered yet to
    move the live-load tie-break), so all N lanes share ONE usage pool and the
    fan-out serializes. A wave allocates distinct pools instead -> N independent
    per-account limits -> the concurrency multiplies."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.config_home = self.home / "config"
        self.config_home.mkdir()
        self.addCleanup(self._tmp.cleanup)

    def _three_distinct(self) -> None:
        login_dir(self.home, ".claude-gem5-acct", uuid="uuid-5", email="gem5@x.ai")
        login_dir(self.home, ".claude-gem7-acct", uuid="uuid-7", email="gem7@x.ai")
        login_dir(self.home, ".claude-gem8-acct", uuid="uuid-8", email="gem8@x.ai")

    def test_wave_allocates_distinct_pools_and_underfills_honestly(self) -> None:
        self._three_distinct()
        w = fleet_accounts.allocate_wave(
            8, task_class="t1", home=str(self.home),
            config_home=str(self.config_home), registry={})
        self.assertTrue(w["ok"])
        self.assertEqual(w["granted"], 3)        # only 3 distinct pools exist
        self.assertEqual(w["distinct_pools"], 3)
        self.assertEqual(w["shortfall"], 5)      # asked 8, got 3 -> honest, never a dup
        pools = [lane["pool"] for lane in w["lanes"]]
        self.assertEqual(len(pools), len(set(pools)), "every lane must be a distinct pool")
        self.assertEqual({lane["tag"] for lane in w["lanes"]}, {"gem5", "gem7", "gem8"})

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
        self.assertEqual(len(wave), 3, "wave gives 3 distinct pools = 3x the headroom")

    def test_wave_excludes_duplicate_identity_dirs(self) -> None:
        # Two dirs logged into ONE Anthropic account are ONE pool; a wave must not hand
        # out both (that would re-collapse two lanes onto a single usage limit).
        login_dir(self.home, ".claude-gem5-acct", uuid="uuid-5", email="gem5@x.ai")
        login_dir(self.home, ".claude", uuid="uuid-5", email="gem5@x.ai")  # same account
        login_dir(self.home, ".claude-gem8-acct", uuid="uuid-8", email="gem8@x.ai")
        w = fleet_accounts.allocate_wave(
            5, task_class="t1", home=str(self.home),
            config_home=str(self.config_home), registry={})
        self.assertEqual(w["granted"], 2, "3 dirs but only 2 distinct accounts")
        pools = [lane["pool"] for lane in w["lanes"]]
        self.assertEqual(sorted(pools), ["uuid:uuid-5", "uuid:uuid-8"])

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

    def test_wave_skips_blocked_pool(self) -> None:
        self._three_distinct()
        reg = {"throttle": {".claude-gem7-acct": {"reset": "Dec 31, 11:59pm"}}}
        w = fleet_accounts.allocate_wave(
            3, task_class="t1", home=str(self.home),
            config_home=str(self.config_home), registry=reg)
        self.assertEqual(w["granted"], 2, "the throttled pool is not offered")
        self.assertNotIn("gem7", {lane["tag"] for lane in w["lanes"]})
        self.assertTrue(any(b.get("tag") == "gem7" for b in w["blocked_target_accounts"]))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
