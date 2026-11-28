#!/usr/bin/env python3
"""Registry summary, pane_text, snapshot summary, and fleet_text tests for fleet_control_pane."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_control_pane as pane  # noqa: E402


class FleetControlPaneViewsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "tools").mkdir()
        (self.root / "VERSION").write_text("test\n", encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    def test_summarize_registry_rolls_up_sessions_and_accounts(self) -> None:
        reg = {
            "schema": "fleet-sessions/3",
            "generated_utc": "2026-06-18T12:00:00+00:00",
            "accounts": [
                {"tag": "gem8", "available": True},
                {
                    "account": ".claude-c10",
                    "tag": "c10",
                    "config_dir": "C:/Users/example/.claude-c10",
                    "available": False,
                    "block_kind": "usage",
                    "block_reason": "usage",
                },
            ],
            "sessions": [
                {"category": "LIVE", "disp": "LIVE", "action": "SKIP_LIVE"},
                {"category": "INFRA", "disp": "INFRA_AUTH", "action": "BLOCKED_AUTH"},
                {"category": "AGENT", "disp": "DEAD_MIDTOOL", "action": "AUTO_RESUME"},
            ],
        }

        got = pane.summarize_registry(reg)

        self.assertTrue(got["exists"])
        self.assertEqual(got["sessions"], 3)
        self.assertEqual(got["categories"]["INFRA"], 1)
        self.assertEqual(got["actions"]["AUTO_RESUME"], 1)
        self.assertEqual(got["auto_resume"], 1)
        self.assertEqual(got["auth_blocked"], 1)
        self.assertEqual(got["accounts"]["available_tags"], ["gem8"])
        self.assertEqual(got["accounts"]["blocked"][0]["reason"], "usage")
        self.assertEqual(got["accounts"]["blocked_count"], 1)
        self.assertEqual(got["accounts"]["usage_blocked_count"], 1)
        self.assertEqual(got["accounts"]["blocked"][0]["config_dir"], "C:/Users/example/.claude-c10")
        self.assertNotIn("command", got["accounts"]["blocked"][0])

    def test_summarize_resume_plan_reads_registry_plan(self) -> None:
        config = pane.normalize_config(pane.default_config(self.root), self.root)
        plan_path = Path(config["registry_dir"]) / "resume_plan.json"
        plan_path.parent.mkdir(parents=True)
        plan_path.write_text(json.dumps({
            "generated_utc": "2026-06-18T12:00:00+00:00",
            "plan": [
                {
                    "account": ".claude-gem8",
                    "config_dir": "C:/Users/example/.claude-gem8",
                    "session": "abcdef1234567890",
                    "cwd": "C:/work/fleet",
                    "project": "fleet",
                    "resume_cmd": "claude --resume abcdef1234567890",
                },
                {
                    "account": ".claude-c10",
                    "config_dir": "C:/Users/example/.claude-c10",
                    "session": "fedcba9876543210",
                    "cwd": "C:/work/job",
                    "project": "job",
                    "resume_cmd": "claude --resume fedcba9876543210",
                },
            ],
        }), encoding="utf-8")

        got = pane.summarize_resume_plan(config, limit=1)

        self.assertTrue(got["exists"])
        self.assertEqual(got["display_path"], str(Path("tools") / "_registry" / "resume_plan.json"))
        self.assertEqual(got["count"], 2)
        self.assertEqual(got["shown"], 1)
        self.assertEqual(got["omitted"], 1)
        self.assertEqual(got["sessions"][0]["account"], ".claude-gem8")
        self.assertEqual(got["sessions"][0]["session_short"], "abcdef12")
        self.assertEqual(got["sessions"][0]["project"], "fleet")
        self.assertEqual(got["sessions"][0]["resume_cmd"], "claude --resume abcdef1234567890")

    def test_pane_text_lists_blocked_account_reasons(self) -> None:
        status = {
            "generated_utc": "2026-06-18T12:00:00Z",
            "verdict": "ACTION",
            "machine": {"host": "host-1"},
            "git": {"available": False, "reason": "not a git repo"},
            "registry": {
                "exists": True,
                "sessions": 2,
                "age_min": 0.0,
                "categories": {"AGENT": 2},
                "actions": {"BLOCKED_AUTH": 1},
                "accounts": {
                    "total": 3,
                    "available": 1,
                    "available_tags": ["gem7"],
                    "blocked": [
                        {"tag": "default", "reason": "usage limit; resets 11:10am"},
                        {"tag": "c10", "reason": "auth/login required"},
                    ],
                },
            },
            "supervisor": {"available": False, "reason": "not configured"},
            "control_tick": {"task": {"supported": False, "reason": "not Windows"}},
            "watchdogs": {},
            "loops": {},
            "actions": [],
        }

        text = pane.pane_text(status)

        self.assertIn("accounts: available=1/3 blocked=2 tags=gem7", text)
        self.assertIn(
            "blocked-accounts: default=usage limit; resets 11:10am | c10=auth/login required",
            text,
        )

    def test_pane_text_lists_dirty_plan_commands(self) -> None:
        status = {
            "generated_utc": "2026-06-18T12:00:00Z",
            "verdict": "ACTION",
            "machine": {"host": "host-1"},
            "git": {
                "available": True,
                "branch": "main",
                "head": "abc123",
                "dirty_total": 2,
                "counts": {"modified": 2},
                "safe_ff": {"state": "ahead", "ok": False, "divergent_count": 0},
                "dirty_plan": {
                    "groups": [
                        {
                            "group": "tools",
                            "count": 2,
                            "command": "python tools/fleet_control_pane.py commit --dirty-group tools -m msg",
                        },
                    ],
                },
            },
            "registry": {"exists": False},
            "supervisor": {"available": False, "reason": "not configured"},
            "control_tick": {"task": {"supported": False, "reason": "not Windows"}},
            "watchdogs": {},
            "loops": {},
            "actions": [],
        }

        text = pane.pane_text(status)

        self.assertIn("dirty-plan:", text)
        self.assertIn("  tools: 2 path(s)", text)
        self.assertIn("python tools/fleet_control_pane.py commit --dirty-group tools", text)

    def test_pane_text_lists_worktree_doctor_summary(self) -> None:
        status = {
            "generated_utc": "2026-06-18T12:00:00Z",
            "verdict": "ACTION",
            "machine": {"host": "host-1"},
            "git": {
                "available": True,
                "branch": "master",
                "head": "abc123",
                "dirty_total": 0,
                "counts": {},
                "safe_ff": {"state": "in-sync", "ok": True, "divergent_count": 0},
                "worktrees": {
                    "available": True,
                    "total": 3,
                    "converged": False,
                    "prune_count": 1,
                    "blocked_count": 1,
                    "retained_count": 0,
                    "blocked": [
                        {
                            "path": "C:/work/fleet-wt/land",
                            "branch": None,
                            "reasons": ["uncommitted changes"],
                        },
                    ],
                },
            },
            "registry": {"exists": False},
            "supervisor": {"available": False, "reason": "not configured"},
            "control_tick": {"task": {"supported": False, "reason": "not Windows"}},
            "watchdogs": {},
            "loops": {},
            "actions": [],
        }

        text = pane.pane_text(status)

        self.assertIn("worktrees: total=3 converged=False prune=1 blocked=1 retained=0", text)
        self.assertIn("blocked-worktree: C:/work/fleet-wt/land (detached): uncommitted changes", text)

    def test_pane_text_lists_supervisor_action_summary(self) -> None:
        job_dir = self.root / "job"
        run_dir = job_dir / "output" / "supervise-20260618T211004Z" / "worker-1"
        run_dir.mkdir(parents=True)
        log_path = run_dir / "run.log"
        log_path.write_text("{}\n", encoding="utf-8")
        status = {
            "generated_utc": "2026-06-18T12:00:00Z",
            "verdict": "ACTION",
            "machine": {"host": "host-1"},
            "config": {"job_dir": str(job_dir)},
            "git": {
                "available": True,
                "branch": "main",
                "head": "abc123",
                "dirty_total": 0,
                "counts": {},
                "safe_ff": {"state": "in-sync", "ok": True, "divergent_count": 0},
            },
            "registry": {"exists": True, "sessions": 1, "age_min": 0.0, "categories": {}, "actions": {}, "accounts": {"available": 1, "total": 1}},
            "supervisor": {
                "available": True,
                "payload": {
                    "verdict": "WALL",
                    "run": "supervise-20260618T211004Z",
                    "process": {"alive": True, "pid": 1, "heartbeat_age_s": 2},
                    "last_decide": {"run_health": "DEGRADED"},
                    "diagnose": {"health": "STALLED", "primary_cause": "unclear", "primary_action": "inspect run"},
                },
            },
            "control_tick": {"task": {"supported": True, "installed": True, "state": "Ready"}},
            "watchdogs": {
                "supervisor": {"task": {"supported": True, "installed": True, "state": "Ready", "last_result": 10}},
            },
            "loops": {},
            "actions": [],
        }

        text = pane.pane_text(status)

        self.assertIn(
            "supervisor: verdict=WALL alive=True pid=1 hb_age=2 run=supervise-20260618T211004Z health=DEGRADED",
            text,
        )
        self.assertIn("supervisor-action: Supervisor run supervise-20260618T211004Z", text)
        self.assertIn(str(log_path), text)

    def test_pane_text_marks_unmerged_dirty_plan_groups(self) -> None:
        status = {
            "generated_utc": "2026-06-18T12:00:00Z",
            "verdict": "ACTION",
            "machine": {"host": "host-1"},
            "git": {
                "available": True,
                "branch": "main",
                "head": "abc123",
                "dirty_total": 1,
                "counts": {"unmerged": 1},
                "safe_ff": {"state": "ahead", "ok": False, "divergent_count": 0},
                "dirty_plan": {
                    "groups": [
                        {
                            "group": "tools/fleet-control-pane",
                            "count": 1,
                            "blocked": True,
                            "reason": "resolve merge conflicts before using pane commit",
                        },
                    ],
                },
            },
            "registry": {"exists": False},
            "supervisor": {"available": False, "reason": "not configured"},
            "control_tick": {"task": {"supported": False, "reason": "not Windows"}},
            "watchdogs": {},
            "loops": {},
            "actions": [],
        }

        text = pane.pane_text(status)

        self.assertIn("U=1", text)
        self.assertIn("blocked: resolve merge conflicts before using pane commit", text)

    def test_pane_text_dirty_plan_omitted_hint_points_to_json(self) -> None:
        groups = [
            {"group": f"group-{idx}", "count": 1, "command": f"cmd-{idx}"}
            for idx in range(7)
        ]
        status = {
            "generated_utc": "2026-06-18T12:00:00Z",
            "verdict": "ACTION",
            "machine": {"host": "host-1"},
            "git": {
                "available": True,
                "branch": "main",
                "head": "abc123",
                "dirty_total": 7,
                "counts": {"modified": 7},
                "safe_ff": {"state": "ahead", "ok": False, "divergent_count": 0},
                "dirty_plan": {"groups": groups},
            },
            "registry": {"exists": False},
            "supervisor": {"available": False, "reason": "not configured"},
            "control_tick": {"task": {"supported": False, "reason": "not Windows"}},
            "watchdogs": {},
            "loops": {},
            "actions": [],
        }

        text = pane.pane_text(status)

        self.assertIn("... 1 more group(s); use `status --json` for all dirty-plan paths", text)

    def test_summarize_machine_snapshot_carries_worktree_doctor_summary(self) -> None:
        status = {
            "schema": pane.SCHEMA,
            "generated_utc": pane.iso_now(),
            "verdict": "ACTION",
            "machine": {"id": "wt-host", "host": "wt-host"},
            "registry": {"sessions": 1, "accounts": {"available": 1, "total": 1}},
            "git": {
                "branch": "master",
                "head": "abc123",
                "dirty_total": 0,
                "safe_ff": {"state": "in-sync", "ok": True},
                "worktrees": {
                    "available": True,
                    "total": 3,
                    "converged": False,
                    "prune_count": 1,
                    "blocked_count": 1,
                    "retained_count": 0,
                    "blocked": [{"path": "C:/work/fleet-wt/land", "branch": None, "reasons": ["dirty"]}],
                    "commands": {"inspect": "python tools/worktree_doctor.py --fetch"},
                },
            },
            "supervisor": {"available": True, "payload": {"verdict": "WALL", "process": {"alive": True}}},
            "control_tick": {"task": {"installed": True}},
            "actions": [],
        }

        machine = pane.summarize_machine_snapshot(self.root / "wt-host.json", status, 15)

        worktrees = machine["git"]["worktrees"]
        self.assertTrue(worktrees["available"])
        self.assertEqual(worktrees["prune_count"], 1)
        self.assertEqual(worktrees["blocked_count"], 1)
        self.assertEqual(worktrees["blocked"][0]["path"], "C:/work/fleet-wt/land")

    def test_fleet_text_lists_blocked_account_reasons(self) -> None:
        doc = {
            "schema": pane.FLEET_SCHEMA,
            "generated_utc": "2026-06-18T12:00:00Z",
            "verdict": "ACTION",
            "states": {"ACTION": 1},
            "totals": {},
            "machines": [
                {
                    "id": "host-1",
                    "state": "ACTION",
                    "age_min": 1.0,
                    "sessions": 2,
                    "actions_count": 1,
                    "accounts": {
                        "available": 1,
                        "total": 3,
                        "blocked": [
                            {"tag": "default", "reason": "usage limit; resets 11:10am"},
                            {"tag": "c10", "reason": "auth/login required"},
                        ],
                    },
                    "supervisor": {"verdict": "WALL", "diagnose": "OK"},
                    "loops": {"action": 0, "count": 0},
                    "git": {"branch": "main", "dirty_total": 0},
                    "actions": ["inspect auth"],
                },
            ],
        }

        text = pane.fleet_text(doc)

        self.assertIn("- host-1: state=ACTION", text)
        self.assertIn(
            "    blocked-accounts: default=usage limit; resets 11:10am | c10=auth/login required",
            text,
        )

    def test_fleet_text_lists_setup_summary(self) -> None:
        doc = {
            "schema": pane.FLEET_SCHEMA,
            "generated_utc": "2026-06-18T12:00:00Z",
            "verdict": "ACTION",
            "states": {"ACTION": 1},
            "totals": {"setup_actions": 1, "loop_actions": 0, "dirty_paths": 0},
            "machines": [
                {
                    "id": "host-1",
                    "state": "ACTION",
                    "age_min": 1.0,
                    "sessions": 2,
                    "actions_count": 1,
                    "accounts": {"available": 1, "total": 3, "blocked": []},
                    "supervisor": {"verdict": "WALL", "diagnose": "OK"},
                    "loops": {"action": 0, "count": 0},
                    "control_tick": {"supported": True, "installed": True, "state": "Ready"},
                    "watchdogs": {
                        "resume": {"supported": True, "installed": True, "state": "Ready"},
                        "supervisor": {"supported": True, "installed": False},
                    },
                    "git": {"branch": "main", "dirty_total": 0},
                    "actions": ["install supervisor watchdog"],
                },
            ],
        }

        text = pane.fleet_text(doc)

        self.assertIn("setup=1 setup_plan=0 version=0 loops=0 dirty=0", text)
        self.assertIn("    setup: tick=Ready watchdogs=resume=Ready,supervisor=missing", text)

    def test_fleet_text_lists_supervisor_action_summary(self) -> None:
        doc = {
            "schema": pane.FLEET_SCHEMA,
            "generated_utc": "2026-06-18T12:00:00Z",
            "verdict": "ACTION",
            "states": {"ACTION": 1},
            "totals": {},
            "machines": [
                {
                    "id": "host-1",
                    "state": "ACTION",
                    "age_min": 1.0,
                    "sessions": 2,
                    "actions_count": 1,
                    "accounts": {"available": 1, "total": 3, "blocked": []},
                    "supervisor": {
                        "verdict": "WALL",
                        "diagnose": "STALLED",
                        "run": "supervise-20260618T211004Z",
                        "run_health": "DEGRADED",
                        "action": "Supervisor run supervise-20260618T211004Z; inspect `C:/work/job/output/supervise-20260618T211004Z/worker-3/run.log`.",
                    },
                    "loops": {"action": 0, "count": 0},
                    "git": {"branch": "main", "dirty_total": 0},
                    "actions": ["inspect supervisor"],
                },
            ],
        }

        text = pane.fleet_text(doc)

        self.assertIn("sup=WALL/STALLED", text)
        self.assertIn(
            "    supervisor-action: Supervisor run supervise-20260618T211004Z; inspect `C:/work/job/output/supervise-20260618T211004Z/worker-3/run.log`.",
            text,
        )

    def test_fleet_text_lists_version_drift_actions(self) -> None:
        doc = {
            "schema": pane.FLEET_SCHEMA,
            "generated_utc": "2026-06-18T12:00:00Z",
            "verdict": "ACTION",
            "states": {"OK": 1},
            "versions": {"0.8.1": 1},
            "totals": {"version_mismatches": 1},
            "machines": [
                {
                    "id": "old-host",
                    "app_version": "0.8.1",
                    "state": "OK",
                    "age_min": 1.0,
                    "sessions": 2,
                    "actions_count": 0,
                    "accounts": {"available": 1, "total": 1, "blocked": []},
                    "supervisor": {"verdict": "WALL", "diagnose": "OK"},
                    "loops": {"action": 0, "count": 0},
                    "git": {"branch": "main", "dirty_total": 0, "safe_ff_state": "behind"},
                    "version_drift": {
                        "machine_version": "0.8.1",
                        "current_version": "0.8.3",
                        "command": "python tools/fleet_control_pane.py sync --fetch --json",
                        "detail": "pane version differs from this operator pane",
                    },
                    "actions": [],
                },
            ],
        }

        text = pane.fleet_text(doc)

        self.assertIn("version=1", text)
        self.assertIn("    version-action: pane 0.8.1 -> 0.8.3; python tools/fleet_control_pane.py sync --fetch --json", text)
        self.assertIn("      pane version differs from this operator pane", text)

    def test_fleet_text_lists_setup_plan_commands(self) -> None:
        doc = {
            "schema": pane.FLEET_SCHEMA,
            "generated_utc": "2026-06-18T12:00:00Z",
            "verdict": "ACTION",
            "states": {"ACTION": 1},
            "totals": {"setup_plan_actions": 2},
            "machines": [
                {
                    "id": "host-1",
                    "state": "ACTION",
                    "age_min": 1.0,
                    "sessions": 2,
                    "actions_count": 1,
                    "accounts": {"available": 1, "total": 3, "blocked": []},
                    "supervisor": {"verdict": "WALL", "diagnose": "OK"},
                    "loops": {"action": 0, "count": 0},
                    "setup_plan": {
                        "steps": [
                            {
                                "id": "local-config",
                                "state": "todo",
                                "ok": True,
                                "detail": "missing local config",
                                "display": "python tools/fleet_control_pane.py init",
                            },
                            {
                                "id": "install-control-pane-tick",
                                "state": "blocked",
                                "ok": False,
                                "detail": "scheduler unsupported",
                                "display": "python tools/fleet_control_pane.py bootstrap --apply",
                            },
                        ],
                    },
                    "git": {"branch": "main", "dirty_total": 0},
                    "actions": ["fix setup"],
                },
            ],
        }

        text = pane.fleet_text(doc)

        self.assertIn("setup_plan=2", text)
        self.assertIn("    setup-step[local-config]: todo missing local config", text)
        self.assertIn("      python tools/fleet_control_pane.py init", text)
        self.assertIn("    setup-step[install-control-pane-tick]: blocked scheduler unsupported", text)
        self.assertIn("      python tools/fleet_control_pane.py bootstrap --apply", text)

    def test_fleet_text_lists_safe_git_actions(self) -> None:
        doc = {
            "schema": pane.FLEET_SCHEMA,
            "generated_utc": "2026-06-18T12:00:00Z",
            "verdict": "ACTION",
            "states": {"ACTION": 2},
            "totals": {},
            "machines": [
                {
                    "id": "ahead-host",
                    "state": "ACTION",
                    "age_min": 1.0,
                    "sessions": 2,
                    "actions_count": 1,
                    "accounts": {"available": 1, "total": 1, "blocked": []},
                    "supervisor": {"verdict": "WALL", "diagnose": "OK"},
                    "loops": {"action": 0, "count": 0},
                    "git": {
                        "branch": "work",
                        "dirty_total": 0,
                        "safe_ff_state": "ahead",
                        "safe_ff_ok": False,
                        "safe_ff_reason": "local branch is ahead",
                    },
                    "actions": [],
                },
                {
                    "id": "behind-host",
                    "state": "ACTION",
                    "age_min": 1.0,
                    "sessions": 2,
                    "actions_count": 1,
                    "accounts": {"available": 1, "total": 1, "blocked": []},
                    "supervisor": {"verdict": "WALL", "diagnose": "OK"},
                    "loops": {"action": 0, "count": 0},
                    "git": {
                        "branch": "work",
                        "dirty_total": 0,
                        "safe_ff_state": "behind",
                        "safe_ff_ok": True,
                        "safe_ff_reason": "fast-forward available",
                    },
                    "actions": [],
                },
            ],
        }

        text = pane.fleet_text(doc)

        self.assertIn("- ahead-host: state=ACTION", text)
        self.assertIn("git=work sync=ahead dirty=0", text)
        self.assertIn("    git-action: python tools/fleet_control_pane.py publish --json (local branch is ahead)", text)
        self.assertIn("git=work sync=behind dirty=0", text)
        self.assertIn("    git-action: python tools/fleet_control_pane.py sync --fetch --json (fast-forward available)", text)

    def test_fleet_text_lists_worktree_actions(self) -> None:
        doc = {
            "schema": pane.FLEET_SCHEMA,
            "generated_utc": "2026-06-18T12:00:00Z",
            "verdict": "ACTION",
            "states": {"ACTION": 1},
            "totals": {"worktree_prune": 2, "worktree_blocked": 1},
            "machines": [
                {
                    "id": "wt-host",
                    "state": "ACTION",
                    "age_min": 1.0,
                    "sessions": 2,
                    "actions_count": 1,
                    "accounts": {"available": 1, "total": 1, "blocked": []},
                    "supervisor": {"verdict": "WALL", "diagnose": "OK"},
                    "loops": {"action": 0, "count": 0},
                    "git": {
                        "branch": "master",
                        "dirty_total": 0,
                        "safe_ff_state": "in-sync",
                        "worktrees": {
                            "prune_count": 2,
                            "blocked_count": 1,
                            "commands": {
                                "prune": "python tools/worktree_doctor.py --prune --fetch",
                                "inspect": "python tools/worktree_doctor.py --fetch",
                            },
                        },
                    },
                    "actions": [],
                },
            ],
        }

        text = pane.fleet_text(doc)

        self.assertIn("wt_prune=2 wt_blocked=1", text)
        self.assertIn("worktree-action: prune 2 safe extra worktree(s)", text)
        self.assertIn("worktree-action: inspect blocked/off-track worktree state", text)

    def test_fleet_text_lists_loop_issues(self) -> None:
        checks = [
            {"name": "healthy", "state": "OK", "detail": "ok"},
            {"name": "alpha", "state": "ACTION", "detail": "heartbeat stale", "auto_recover": True},
            {"name": "beta", "state": "TIMEOUT", "detail": "status command timed out", "has_recover_cmd": True},
            {"name": "gamma", "state": "UNAVAILABLE", "action": "configure loop status"},
            {"name": "delta", "state": "UNKNOWN", "detail": "missing verdict"},
        ]
        doc = {
            "schema": pane.FLEET_SCHEMA,
            "generated_utc": "2026-06-18T12:00:00Z",
            "verdict": "ACTION",
            "states": {"ACTION": 1},
            "totals": {"loop_actions": 4},
            "machines": [
                {
                    "id": "host-1",
                    "state": "ACTION",
                    "age_min": 1.0,
                    "sessions": 2,
                    "actions_count": 1,
                    "accounts": {"available": 1, "total": 3, "blocked": []},
                    "supervisor": {"verdict": "WALL", "diagnose": "OK"},
                    "loops": {"action": 4, "count": 5, "checks": checks},
                    "git": {"branch": "main", "dirty_total": 0},
                    "actions": ["inspect loops"],
                },
            ],
        }

        text = pane.fleet_text(doc)

        self.assertIn("    loop[alpha]: ACTION (auto-recover) heartbeat stale", text)
        self.assertIn("    loop[beta]: TIMEOUT (recoverable) status command timed out", text)
        self.assertIn("    loop[gamma]: UNAVAILABLE configure loop status", text)
        self.assertNotIn("loop[delta]", text)
        self.assertIn("    ... 1 more loop issue(s); use `fleet --json` for all loop checks", text)

    def test_fleet_text_lists_dirty_group_commands(self) -> None:
        groups = [
            {"group": "docs", "count": 3, "command": "python tools/fleet_control_pane.py commit --dirty-group docs -m msg"},
            {"group": "fak", "count": 3, "command": "python tools/fleet_control_pane.py commit --dirty-group fak -m msg"},
            {
                "group": "fak/internal/compute",
                "count": 11,
                "command": "python tools/fleet_control_pane.py commit --dirty-group fak/internal/compute -m msg",
            },
            {"group": "fak/internal/model", "count": 6, "command": "python tools/fleet_control_pane.py commit --dirty-group fak/internal/model -m msg"},
            {"group": "tools/proofs", "count": 3, "command": "python tools/fleet_control_pane.py commit --dirty-group tools/proofs -m msg"},
        ]
        doc = {
            "schema": pane.FLEET_SCHEMA,
            "generated_utc": "2026-06-18T12:00:00Z",
            "verdict": "ACTION",
            "states": {"ACTION": 1},
            "totals": {},
            "machines": [
                {
                    "id": "host-1",
                    "state": "ACTION",
                    "age_min": 1.0,
                    "sessions": 2,
                    "actions_count": 1,
                    "accounts": {"available": 1, "total": 3, "blocked": []},
                    "supervisor": {"verdict": "WALL", "diagnose": "OK"},
                    "loops": {"action": 0, "count": 0},
                    "git": {"branch": "main", "dirty_total": 26, "dirty_plan": {"count": len(groups), "groups": groups}},
                    "actions": ["commit dirty work"],
                },
            ],
        }

        text = pane.fleet_text(doc)

        self.assertIn("    dirty-groups: docs=3, fak=3, fak/internal/compute=11, fak/internal/model=6 +1 more", text)
        self.assertIn("python tools/fleet_control_pane.py commit --dirty-group docs -m msg", text)
        self.assertIn("python tools/fleet_control_pane.py commit --dirty-group fak/internal/compute -m msg", text)
        self.assertNotIn("python tools/fleet_control_pane.py commit --dirty-group fak/internal/model -m msg", text)
        self.assertIn("      ... 2 more group(s); use `fleet --json` for all dirty-plan paths", text)

    def test_fleet_text_marks_live_local_machine(self) -> None:
        doc = {
            "schema": pane.FLEET_SCHEMA,
            "generated_utc": "2026-06-18T12:00:00Z",
            "verdict": "OK",
            "states": {"OK": 1},
            "versions": {"1.2.3": 1},
            "totals": {},
            "written": {"json": "pane.json", "machine": "host.json"},
            "machines": [
                {
                    "id": "host-1",
                    "app_version": "1.2.3",
                    "live_local": True,
                    "state": "OK",
                    "age_min": 0.0,
                    "sessions": 2,
                    "actions_count": 0,
                    "accounts": {"available": 1, "total": 1, "blocked": []},
                    "supervisor": {"verdict": "WALL", "diagnose": "OK"},
                    "loops": {"action": 0, "count": 0},
                    "git": {"branch": "main", "dirty_total": 0},
                    "actions": [],
                },
            ],
        }

        text = pane.fleet_text(doc)

        self.assertIn("versions: 1.2.3=1", text)
        self.assertIn("local-written: pane.json ; host.json", text)
        self.assertIn("- host-1 live: state=OK", text)
        self.assertIn("ver=1.2.3", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
