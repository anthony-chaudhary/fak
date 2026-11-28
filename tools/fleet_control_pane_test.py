#!/usr/bin/env python3
"""CLI entrypoint, exit-code, and recommended_actions tests for fleet_control_pane.py."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_control_pane as pane  # noqa: E402
import appversion  # noqa: E402


class FleetControlPaneTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "tools").mkdir()
        (self.root / "VERSION").write_text("test\n", encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    def test_app_version_hides_conflicted_version_file(self) -> None:
        (self.root / "VERSION").write_text("<<<<<<< HEAD\n0.8.15\n=======\n0.10.0\n>>>>>>> master\n", encoding="utf-8")

        self.assertEqual(appversion.app_version(self.root), appversion.DEFAULT_VERSION)

    def test_main_init_accepts_machine_override_flags(self) -> None:
        with mock.patch("builtins.print"):
            rc = pane.main([
                "--root",
                str(self.root),
                "init",
                "--machine-id",
                "dgx-2",
                "--machine-dir",
                "../shared/machines",
                "--target",
                "6",
            ])

        self.assertEqual(rc, 0)
        doc = json.loads((self.root / "tools" / "_registry" / "control_pane.local.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["machine_id"], "dgx-2")
        self.assertEqual(doc["machine_dir"], "../shared/machines")
        self.assertEqual(doc["target"], 6)

    def test_main_loop_check_fail_on_action_is_opt_in(self) -> None:
        bad_script = self.root / "tools" / "bad_loop.py"
        bad_script.write_text("print('{\"ok\": false}')\n", encoding="utf-8")
        (self.root / "tools" / "control_pane.loops.json").write_text(json.dumps({
            "schema": pane.LOOP_CATALOG_SCHEMA,
            "loops": {"stalled": {"status_cmd": [sys.executable, str(bad_script)]}},
        }), encoding="utf-8")

        with mock.patch("builtins.print"):
            default_rc = pane.main(["--root", str(self.root), "loop-check", "stalled"])
            failing_rc = pane.main(["--root", str(self.root), "loop-check", "stalled", "--fail-on-action"])

        self.assertEqual(default_rc, 0)
        self.assertEqual(failing_rc, 1)

    def test_main_loop_audit_exit_codes(self) -> None:
        surfacing = self.root / "tools" / "surfacing.py"
        surfacing.write_text("print('{\"ok\": false}')\n", encoding="utf-8")
        broken = {
            "schema": pane.LOOP_CATALOG_SCHEMA,
            "loops": {"missing-cmd": {"status_cmd": ["this-binary-does-not-exist-xyz"]}},
        }
        action_only = {
            "schema": pane.LOOP_CATALOG_SCHEMA,
            "loops": {"surfacer": {"status_cmd": [sys.executable, str(surfacing)]}},
        }
        catalog = self.root / "tools" / "control_pane.loops.json"

        catalog.write_text(json.dumps(action_only), encoding="utf-8")
        with mock.patch("builtins.print"):
            action_rc = pane.main(["--root", str(self.root), "loop-audit"])
            action_fail_rc = pane.main(["--root", str(self.root), "loop-audit", "--fail-on-action"])
        self.assertEqual(action_rc, 0)  # action is a pass by default
        self.assertEqual(action_fail_rc, 1)  # opt-in gate fails on action

        catalog.write_text(json.dumps(broken), encoding="utf-8")
        with mock.patch("builtins.print"):
            broken_rc = pane.main(["--root", str(self.root), "loop-audit"])
        self.assertEqual(broken_rc, 1)  # broken always fails

    def test_main_loop_add_accepts_json_command_arrays(self) -> None:
        with mock.patch("builtins.print"):
            rc = pane.main([
                "--root",
                str(self.root),
                "loop-add",
                "json-loop",
                "--status-cmd",
                json.dumps([sys.executable, "tools/status.py", "--json"]),
                "--recover-cmd",
                json.dumps([sys.executable, "tools/recover.py"]),
                "--auto-recover",
                "--apply",
            ])

        self.assertEqual(rc, 0)
        doc = json.loads((self.root / "tools" / "_registry" / "control_pane.local.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["loops"]["json-loop"]["status_cmd"], [sys.executable, "tools/status.py", "--json"])
        self.assertEqual(doc["loops"]["json-loop"]["recover_cmd"], [sys.executable, "tools/recover.py"])
        self.assertTrue(doc["loops"]["json-loop"]["auto_recover"])

    def test_main_loop_add_accepts_repo_scope(self) -> None:
        with mock.patch("builtins.print"):
            rc = pane.main([
                "--root",
                str(self.root),
                "loop-add",
                "repo-loop",
                "--scope",
                "repo",
                "--status-cmd",
                json.dumps([sys.executable, "tools/repo_status.py", "--json"]),
                "--apply",
            ])

        self.assertEqual(rc, 0)
        catalog = json.loads((self.root / "tools" / "control_pane.loops.json").read_text(encoding="utf-8"))
        self.assertEqual(catalog["loops"]["repo-loop"]["status_cmd"], [sys.executable, "tools/repo_status.py", "--json"])
        self.assertFalse((self.root / "tools" / "_registry" / "control_pane.local.json").exists())

    def test_main_commit_expands_dirty_group(self) -> None:
        config = {"root": str(self.root)}
        selection = {
            "ok": True,
            "requested": ["tools/fleet-control-pane"],
            "missing": [],
            "available": ["tools/fleet-control-pane"],
            "paths": ["tools/fleet_control_pane.py", "tools/fleet_control_pane_test.py"],
        }
        commit_doc = {
            "schema": pane.COMMIT_SCHEMA,
            "ok": True,
            "applied": False,
            "paths": selection["paths"],
        }

        with (
            mock.patch.object(pane, "load_config", return_value=config),
            mock.patch.object(pane, "dirty_group_selection", return_value=selection) as selection_mock,
            mock.patch.object(pane, "commit_plan", return_value=commit_doc) as commit_mock,
            mock.patch("builtins.print"),
        ):
            rc = pane.main([
                "--root",
                str(self.root),
                "commit",
                "--dirty-group",
                "tools/fleet-control-pane",
                "-m",
                "tools: update pane",
            ])

        self.assertEqual(rc, 0)
        selection_mock.assert_called_once_with(config, ["tools/fleet-control-pane"])
        commit_mock.assert_called_once_with(
            config,
            paths=["tools/fleet_control_pane.py", "tools/fleet_control_pane_test.py"],
            message="tools: update pane",
            apply=False,
            allow_dir=False,
        )

    def test_main_commit_refuses_unknown_dirty_group(self) -> None:
        config = {"root": str(self.root)}
        selection = {
            "ok": False,
            "requested": ["missing"],
            "missing": ["missing"],
            "available": ["tools/fleet-control-pane"],
            "paths": [],
        }

        with (
            mock.patch.object(pane, "load_config", return_value=config),
            mock.patch.object(pane, "dirty_group_selection", return_value=selection),
            mock.patch.object(pane, "commit_plan") as commit_mock,
            mock.patch("builtins.print"),
        ):
            rc = pane.main(["--root", str(self.root), "commit", "--dirty-group", "missing", "-m", "msg"])

        self.assertEqual(rc, 1)
        commit_mock.assert_not_called()

    def test_recommended_actions_surface_missing_registry_and_dirty_git(self) -> None:
        status = {
            "config": {"local_config_exists": False},
            "registry": {"exists": False},
            "watchdogs": {},
            "supervisor": {"available": False, "reason": "job_dir is not configured"},
            "git": {
                "dirty_total": 2,
                "safe_ff": {"state": "behind", "ok": False},
            },
        }
        config = {"root": str(self.root)}

        actions = pane.recommended_actions(status, config)

        self.assertTrue(any("init" in a for a in actions))
        self.assertTrue(any("session registry" in a for a in actions))
        self.assertTrue(any("fleet_control_pane.py sync --fetch" in a for a in actions))
        self.assertTrue(any("dirty path" in a for a in actions))

    def test_recommended_actions_prioritize_unmerged_paths(self) -> None:
        status = {
            "config": {"local_config_exists": True},
            "registry": {"exists": True},
            "control_tick": {"task": {}},
            "watchdogs": {},
            "loops": {},
            "supervisor": {"available": True, "payload": {"verdict": "WALL", "process": {"alive": True}}},
            "git": {
                "dirty_total": 3,
                "counts": {"unmerged": 2},
                "unmerged_paths": ["VERSION", "fak/internal/model/hal.go"],
                "safe_ff": {"state": "ahead", "ok": False},
            },
        }

        actions = pane.recommended_actions(status, {"root": str(self.root)})

        self.assertTrue(any("Resolve 2 unmerged path(s)" in action for action in actions))
        self.assertTrue(any("VERSION" in action for action in actions))
        self.assertTrue(any("publish is blocked until the worktree is clean" in action for action in actions))
        self.assertTrue(any("including merge conflicts" in action for action in actions))
        self.assertFalse(any("publish --json` before pushing" in action for action in actions))

    def test_recommended_actions_block_publish_during_merge(self) -> None:
        status = {
            "config": {"local_config_exists": True},
            "registry": {"exists": True},
            "control_tick": {"task": {}},
            "watchdogs": {},
            "loops": {},
            "supervisor": {"available": True, "payload": {"verdict": "WALL", "process": {"alive": True}}},
            "git": {
                "dirty_total": 1,
                "counts": {"modified": 1},
                "merge_in_progress": True,
                "safe_ff": {"state": "ahead", "ok": False},
            },
        }

        actions = pane.recommended_actions(status, {"root": str(self.root)})

        self.assertTrue(any("Finish or abort the in-progress merge" in action for action in actions))
        self.assertTrue(any("publish is blocked until the worktree is clean" in action for action in actions))
        self.assertFalse(any("publish --json` before pushing" in action for action in actions))

    def test_recommended_actions_surface_recover_for_auto_resume(self) -> None:
        status = {
            "config": {"local_config_exists": True},
            "registry": {"exists": True, "auto_resume": 2},
            "control_tick": {"task": {}},
            "watchdogs": {},
            "loops": {},
            "supervisor": {
                "available": True,
                "payload": {"verdict": "OK", "process": {"alive": True}, "diagnose": {"health": "OK"}},
            },
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True, "reason": "up to date"}},
        }

        actions = pane.recommended_actions(status, {"root": str(self.root)})

        self.assertTrue(any("fleet_control_pane.py recover" in action for action in actions))
        self.assertTrue(any("recover --apply" in action for action in actions))

    def test_recommended_actions_ignore_accounts_when_none_blocked(self) -> None:
        status = {
            "config": {"local_config_exists": True},
            "registry": {
                "exists": True,
                "auth_blocked": 0,
                "accounts": {"blocked": [], "auth_blocked_count": 0, "blocked_count": 0},
            },
            "control_tick": {"task": {}},
            "watchdogs": {},
            "loops": {},
            "supervisor": {
                "available": True,
                "payload": {"verdict": "OK", "process": {"alive": True}, "diagnose": {"health": "OK"}},
            },
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
        }

        actions = pane.recommended_actions(status, {"root": str(self.root)})

        self.assertFalse(any("Account remediation needed" in action for action in actions))

    def test_recommended_actions_do_not_prompt_for_usage_or_access_walls(self) -> None:
        blocked = [
            {
                "tag": "default",
                "block_kind": "usage",
                "reason": "usage limit; resets 6pm",
                "throttled": True,
            },
            {
                "tag": "gem7",
                "block_kind": "access",
                "reason": "Claude subscription access disabled",
            },
        ]
        status = {
            "config": {"local_config_exists": True},
            "registry": {
                "exists": True,
                "auth_blocked": 1,
                "accounts": {"blocked": blocked, "auth_blocked_count": 0, "blocked_count": 2},
            },
            "control_tick": {"task": {}},
            "watchdogs": {},
            "loops": {},
            "supervisor": {
                "available": True,
                "payload": {"verdict": "OK", "process": {"alive": True}, "diagnose": {"health": "OK"}},
            },
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
        }

        actions = pane.recommended_actions(status, {"root": str(self.root)})

        self.assertFalse(any("Account remediation needed" in action for action in actions))

    def test_recommended_actions_report_account_and_session_blockers(self) -> None:
        blocked = [
            {
                "tag": "node-agent",
                "account": ".claude-agent",
                "config_dir": "C:/Users/example/.claude-agent",
                "block_kind": "auth",
                "reason": "auth/login required",
                "auth_blocked_sessions": 61,
                "command": "$env:CLAUDE_CONFIG_DIR='C:/Users/example/.claude-agent'; claude",
            },
            {
                "tag": "c10",
                "account": ".claude-c10",
                "config_dir": "C:/Users/example/.claude-c10",
                "block_kind": "auth",
                "reason": "auth/login required",
                "auth_blocked_sessions": 0,
                "command": "$env:CLAUDE_CONFIG_DIR='C:/Users/example/.claude-c10'; claude",
            },
        ]
        status = {
            "config": {"local_config_exists": True},
            "registry": {
                "exists": True,
                "auth_blocked": 61,
                "accounts": {"blocked": blocked, "auth_blocked_count": 2, "blocked_count": 2},
            },
            "control_tick": {"task": {}},
            "watchdogs": {},
            "loops": {},
            "supervisor": {
                "available": True,
                "payload": {"verdict": "OK", "process": {"alive": True}, "diagnose": {"health": "OK"}},
            },
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
        }

        actions = pane.recommended_actions(status, {"root": str(self.root)})

        action = next(a for a in actions if "Account remediation needed" in a)
        self.assertIn("2 login-blocked account(s) covering 61 auth-stopped session(s)", action)
        self.assertIn("node-agent=auth/login required, 61 session(s)", action)
        self.assertIn("c10=auth/login required", action)
        self.assertIn("$env:CLAUDE_CONFIG_DIR", action)
        self.assertIn("fleet_accounts.py list", action)

    def test_recommended_actions_report_session_only_auth_blockers(self) -> None:
        status = {
            "config": {"local_config_exists": True},
            "registry": {
                "exists": True,
                "auth_blocked": 2,
                "accounts": {"blocked": [], "auth_blocked_count": 0, "blocked_count": 0},
            },
            "control_tick": {"task": {}},
            "watchdogs": {},
            "loops": {},
            "supervisor": {
                "available": True,
                "payload": {"verdict": "OK", "process": {"alive": True}, "diagnose": {"health": "OK"}},
            },
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
        }

        actions = pane.recommended_actions(status, {"root": str(self.root)})

        action = next(a for a in actions if "Account remediation needed" in a)
        self.assertIn("2 auth-stopped session(s) but no account-level auth blocker is currently active", action)

    def test_recommended_actions_surface_publish_for_ahead_branch(self) -> None:
        status = {
            "config": {"local_config_exists": True},
            "registry": {"exists": True},
            "control_tick": {"task": {}},
            "watchdogs": {},
            "loops": {},
            "supervisor": {
                "available": True,
                "payload": {"verdict": "OK", "process": {"alive": True}, "diagnose": {"health": "OK"}},
            },
            "git": {"dirty_total": 0, "safe_ff": {"state": "ahead", "ok": False}},
        }

        actions = pane.recommended_actions(status, {"root": str(self.root)})

        self.assertTrue(any("fleet_control_pane.py publish --json" in action for action in actions))

    def test_recommended_actions_surface_worktree_doctor_state(self) -> None:
        status = {
            "config": {"local_config_exists": True},
            "registry": {"exists": True},
            "control_tick": {"task": {}},
            "watchdogs": {},
            "loops": {},
            "supervisor": {
                "available": True,
                "payload": {"verdict": "OK", "process": {"alive": True}, "diagnose": {"health": "OK"}},
            },
            "git": {
                "dirty_total": 0,
                "safe_ff": {"state": "in-sync", "ok": True},
                "worktrees": {
                    "available": True,
                    "prune_count": 2,
                    "blocked_count": 1,
                    "commands": {
                        "prune": "python tools/worktree_doctor.py --prune --fetch",
                        "inspect": "python tools/worktree_doctor.py --fetch",
                    },
                },
            },
        }

        actions = pane.recommended_actions(status, {"root": str(self.root), "python": "python"})

        self.assertTrue(any("2 safe extra worktree(s)" in action for action in actions))
        self.assertTrue(any("worktree_doctor.py --prune --fetch" in action for action in actions))
        self.assertTrue(any("Worktree doctor is not converged" in action for action in actions))

    def test_recommended_actions_surface_no_enabled_loop_starter(self) -> None:
        status = {
            "config": {"local_config_exists": True},
            "registry": {"exists": True},
            "control_tick": {"task": {}},
            "watchdogs": {},
            "loops": {
                "count": 0,
                "configured": 1,
                "enabled": 0,
                "disabled": 1,
                "commands": ["python tools/fleet_control_pane.py loop-scaffold NAME"],
                "checks": [],
            },
            "supervisor": {
                "available": True,
                "payload": {"verdict": "OK", "process": {"alive": True}, "diagnose": {"health": "OK"}},
            },
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
        }

        actions = pane.recommended_actions(status, {"root": str(self.root)})

        self.assertTrue(any("No enabled loops configured" in action for action in actions))
        self.assertTrue(any("loop-scaffold NAME" in action for action in actions))

    def test_recommended_actions_surface_supervisor_command_for_stalled(self) -> None:
        status = {
            "config": {"local_config_exists": True},
            "registry": {"exists": True},
            "control_tick": {"task": {}},
            "watchdogs": {},
            "loops": {},
            "supervisor": {
                "available": True,
                "payload": {
                    "verdict": "STALLED",
                    "process": {"alive": True, "pid": 123},
                    "diagnose": {"health": "STALLED", "primary_action": "inspect log"},
                },
            },
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True, "reason": "up to date"}},
        }

        actions = pane.recommended_actions(status, {"root": str(self.root)})

        self.assertTrue(any("fleet_control_pane.py supervisor --json" in action for action in actions))
        self.assertTrue(any("supervisor --restart" in action for action in actions))

    def test_recommended_actions_ignore_stale_supervisor_diagnosis_when_verdict_is_non_action(self) -> None:
        status = {
            "config": {"local_config_exists": True},
            "registry": {"exists": True},
            "control_tick": {"task": {}},
            "watchdogs": {},
            "loops": {},
            "supervisor": {
                "available": True,
                "payload": {
                    "verdict": "WALL",
                    "process": {"alive": True, "pid": 123},
                    "diagnose": {"health": "STALLED", "primary_action": "inspect old run"},
                },
            },
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True, "reason": "up to date"}},
        }

        actions = pane.recommended_actions(status, {"root": str(self.root)})

        self.assertFalse(any("Supervisor verdict" in action for action in actions))
        self.assertFalse(any("Supervisor diagnosis" in action for action in actions))
        self.assertFalse(pane.supervisor_needs_action(status["supervisor"]))

    def test_recommended_actions_surface_degraded_wall_supervisor_run(self) -> None:
        run_dir = self.root / "job" / "output" / "supervise-20260618T211004Z"
        log_dir = run_dir / "worker-3"
        log_dir.mkdir(parents=True)
        log_path = log_dir / "run.log"
        log_path.write_text('{"type":"result"}\n', encoding="utf-8")
        status = {
            "config": {"local_config_exists": True, "job_dir": str(self.root / "job")},
            "registry": {"exists": True},
            "control_tick": {"task": {}},
            "watchdogs": {
                "supervisor": {
                    "task": {"supported": True, "installed": True, "state": "Ready", "last_result": 10},
                },
            },
            "loops": {},
            "supervisor": {
                "available": True,
                "payload": {
                    "verdict": "WALL",
                    "run": "supervise-20260618T211004Z",
                    "process": {"alive": True, "pid": 123},
                    "last_decide": {"run_health": "DEGRADED", "stop_reason": "degraded-storm"},
                    "diagnose": {
                        "health": "STALLED",
                        "primary_cause": "unclear",
                        "primary_action": "inspect the child run.log",
                    },
                },
            },
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True, "reason": "up to date"}},
        }

        actions = pane.recommended_actions(status, {"root": str(self.root), "job_dir": str(self.root / "job")})

        action = next(a for a in actions if "Supervisor run supervise-20260618T211004Z" in a)
        self.assertIn("health=DEGRADED", action)
        self.assertIn("verdict=WALL", action)
        self.assertIn("cause=unclear", action)
        self.assertIn("watchdog=10 (0x0000000A; supervisor watchdog reported an action-status verdict)", action)
        self.assertIn("accepted watchdog status, not setup failure", action)
        self.assertIn(str(log_path), action)
        self.assertIn("fleet_control_pane.py supervisor --json", action)

    def test_recommended_actions_do_not_require_standalone_watchdog_when_tick_covers_it(self) -> None:
        status = {
            "config": {"local_config_exists": True},
            "registry": {"exists": True},
            "control_tick": {"task": {"supported": True, "installed": True}},
            "watchdogs": {
                "resume": {
                    "script_exists": True,
                    "register_script": "tools/register_resume_watchdog.ps1",
                    "task": {"supported": True, "installed": False},
                },
                "supervisor": {
                    "script_exists": True,
                    "register_script": "tools/register_supervisor_watchdog.ps1",
                    "task": {"supported": True, "installed": True, "state": "Ready", "last_result": 2147946720},
                },
            },
            "loops": {},
            "supervisor": {"available": True, "payload": {"verdict": "WALL", "process": {"alive": True}}},
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
        }

        actions = pane.recommended_actions(status, {"root": str(self.root)})

        self.assertFalse(any("Install resume watchdog" in action for action in actions))
        self.assertFalse(any("supervisor watchdog last scheduler result" in action for action in actions))

    def test_recommended_actions_do_not_claim_tick_recovers_bootstrap_loop(self) -> None:
        status = {
            "config": {"local_config_exists": True},
            "registry": {"exists": True},
            "control_tick": {"task": {"supported": True, "installed": True}},
            "watchdogs": {},
            "loops": {
                "checks": [
                    {
                        "name": "control-pane-doctor",
                        "state": "TIMEOUT",
                        "auto_recover": True,
                        "has_recover_cmd": True,
                        "action": "Run control-pane bootstrap to repair local setup.",
                    }
                ]
            },
            "supervisor": {"available": True, "payload": {"verdict": "WALL", "process": {"alive": True}}},
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
        }
        config = {
            "root": str(self.root),
            "python": sys.executable,
            "loops": {
                "control-pane-doctor": {
                    "recover_cmd": [sys.executable, "tools/fleet_control_pane.py", "bootstrap", "--apply"]
                }
            },
        }

        actions = pane.recommended_actions(status, config)

        self.assertTrue(any("bootstrap --apply" in action for action in actions))
        self.assertFalse(any("next live control tick" in action for action in actions))

    def test_recommended_actions_surface_disabled_control_tick(self) -> None:
        status = {
            "config": {"local_config_exists": True},
            "registry": {"exists": True},
            "control_tick": {"task": {"supported": True, "installed": True, "state": "Disabled", "last_result": 0}},
            "watchdogs": {},
            "loops": {},
            "supervisor": {"available": True, "payload": {"verdict": "WALL", "process": {"alive": True}}},
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
        }

        actions = pane.recommended_actions(status, {"root": str(self.root)})

        self.assertTrue(any("Control-pane tick task is Disabled" in action for action in actions))
        self.assertTrue(any("bootstrap --apply" in action for action in actions))
        self.assertFalse(any("tick --dry-run" in action for action in actions))

    def test_recommended_actions_require_watchdog_task_when_tick_is_missing(self) -> None:
        status = {
            "config": {"local_config_exists": True},
            "registry": {"exists": True},
            "control_tick": {"task": {"supported": True, "installed": False}},
            "watchdogs": {
                "resume": {
                    "script_exists": True,
                    "register_script": "tools/register_resume_watchdog.ps1",
                    "task": {"supported": True, "installed": False},
                },
            },
            "loops": {},
            "supervisor": {"available": True, "payload": {"verdict": "WALL", "process": {"alive": True}}},
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
        }

        actions = pane.recommended_actions(status, {"root": str(self.root)})

        self.assertTrue(any("Install resume watchdog" in action for action in actions))

    def test_recommended_actions_surface_nonzero_scheduler_results(self) -> None:
        status = {
            "config": {"local_config_exists": True},
            "registry": {"exists": True},
            "control_tick": {"task": {"supported": True, "installed": True, "last_result": 1}},
            "watchdogs": {
                "resume": {
                    "script_exists": True,
                    "task": {"supported": True, "installed": True, "last_result": 2},
                },
            },
            "loops": {},
            "supervisor": {
                "available": True,
                "payload": {"verdict": "OK", "process": {"alive": True}, "diagnose": {"health": "OK"}},
            },
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
        }
        config = {"root": str(self.root)}

        actions = pane.recommended_actions(status, config)

        self.assertTrue(any("Control-pane tick last scheduler result was 1 (0x00000001)" in a for a in actions))
        self.assertTrue(any("tick --dry-run --no-write" in a for a in actions))
        self.assertTrue(any("fleet_control_pane.py doctor" in a for a in actions))
        self.assertFalse(any("resume watchdog last scheduler result was 2" in a for a in actions))

        status["control_tick"] = {"task": {"supported": True, "installed": False}}
        actions = pane.recommended_actions(status, config)

        self.assertTrue(any("resume watchdog last scheduler result was 2 (0x00000002)" in a for a in actions))

    def test_recommended_actions_ignore_running_windows_scheduler_code(self) -> None:
        for result in (pane.WINDOWS_TASK_RUNNING_RESULT, pane.WINDOWS_TASK_REQUEST_REFUSED_RESULT):
            with self.subTest(result=result):
                status = {
                    "config": {"local_config_exists": True},
                    "registry": {"exists": True},
                    "control_tick": {
                        "task": {
                            "supported": True,
                            "installed": True,
                            "state": "Running",
                            "last_result": result,
                        },
                    },
                    "watchdogs": {},
                    "loops": {},
                    "supervisor": {
                        "available": True,
                        "payload": {"verdict": "OK", "process": {"alive": True}, "diagnose": {"health": "OK"}},
                    },
                    "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
                }

                actions = pane.recommended_actions(status, {"root": str(self.root)})

                self.assertFalse(any("last scheduler result" in action for action in actions))

    def test_status_exit_code_only_fails_when_requested(self) -> None:
        status = {"verdict": "ACTION"}
        with (
            mock.patch.object(pane, "load_config", return_value={}),
            mock.patch.object(pane, "collect_status", return_value=status),
            mock.patch.object(pane, "pane_text", return_value="pane"),
            mock.patch("builtins.print"),
        ):
            self.assertEqual(pane.main(["--root", str(self.root), "status"]), 0)
            self.assertEqual(
                pane.main(["--root", str(self.root), "status", "--fail-on-action"]),
                1,
            )

    def test_fleet_exit_code_only_fails_when_requested(self) -> None:
        doc = {"schema": pane.FLEET_SCHEMA, "verdict": "ACTION", "generated_utc": "now", "machines": []}
        with (
            mock.patch.object(pane, "load_config", return_value={}),
            mock.patch.object(pane, "fleet_view", return_value=doc),
            mock.patch.object(pane, "fleet_text", return_value="fleet"),
            mock.patch("builtins.print"),
        ):
            self.assertEqual(pane.main(["--root", str(self.root), "fleet"]), 0)
            self.assertEqual(
                pane.main(["--root", str(self.root), "fleet", "--fail-on-action"]),
                1,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
