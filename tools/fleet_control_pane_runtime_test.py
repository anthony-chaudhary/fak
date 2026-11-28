#!/usr/bin/env python3
"""Collector, scheduler, supervisor, tick, recover, bootstrap, doctor, setup_plan, and fleet_view tests for fleet_control_pane."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_control_pane as pane  # noqa: E402


class FleetControlPaneRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "tools").mkdir()
        (self.root / "VERSION").write_text("test\n", encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    def test_scheduled_task_status_accepts_uint32_last_result(self) -> None:
        proc = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"task_name":"FleetSupervisorWatchdog","installed":true,"last_result":2147946720}',
            stderr="",
        )

        with (
            mock.patch.object(platform, "system", return_value="Windows"),
            mock.patch.object(pane, "powershell_exe", return_value="powershell.exe"),
            mock.patch.object(pane, "run", return_value=proc) as run_mock,
        ):
            got = pane.scheduled_task_status("FleetSupervisorWatchdog")

        self.assertTrue(got["installed"])
        self.assertEqual(got["last_result"], 2147946720)
        self.assertIn("[int64]$i.LastTaskResult", run_mock.call_args.args[0][-1])

    def test_scheduled_task_status_reports_timeout_without_crashing(self) -> None:
        with (
            mock.patch.object(platform, "system", return_value="Windows"),
            mock.patch.object(pane, "powershell_exe", return_value="powershell.exe"),
            mock.patch.object(pane, "run", side_effect=subprocess.TimeoutExpired(["powershell.exe"], 15)),
        ):
            got = pane.scheduled_task_status("FleetResumeWatchdog")

        self.assertTrue(got["supported"])
        self.assertIsNone(got["installed"])
        self.assertIn("timed out", got["reason"])

    def test_scheduler_result_text_decodes_windows_codes(self) -> None:
        self.assertEqual(
            pane.scheduler_result_text({"last_result": 2147946720}),
            "2147946720 (0x800710E0; request refused, usually because an earlier run is still active)",
        )
        self.assertEqual(
            pane.scheduler_result_text({"last_result": -2147020576}),
            "2147946720 (0x800710E0; request refused, usually because an earlier run is still active)",
        )
        self.assertEqual(pane.scheduler_result_text({"last_result": 1}), "1 (0x00000001)")

    def test_windows_tick_register_script_starts_task_after_registering(self) -> None:
        script = (Path(__file__).resolve().parent / "register_control_pane_tick.ps1").read_text(encoding="utf-8")

        # The installer moved off `schtasks /Create` to an S4U Register-ScheduledTask
        # (#3322), so the old `/Change /TN $TaskName /ENABLE` repair no longer exists:
        # -Force creates-or-replaces the task already enabled. What still has to hold is
        # the end state that repair existed to guarantee -- the task is registered, then
        # started -- plus a registration failure being surfaced rather than swallowed,
        # because S4U needs an elevated shell and silently skipping that leaves no tick.
        register_idx = script.index("Register-ScheduledTask -TaskName $TaskName")
        start_idx = script.index("Start-ScheduledTask -TaskName $TaskName")

        self.assertLess(register_idx, start_idx)
        self.assertIn("-Force", script[register_idx:start_idx])
        self.assertIn("Register-ScheduledTask failed", script)
        # A re-drift back to the schtasks path would reinstate the disabled-after-create
        # bug; /Delete stays legal for `-Action remove`, the /Change repair does not.
        self.assertNotIn("schtasks /Change", script)

    def test_disabled_scheduled_task_needs_action(self) -> None:
        task = {"supported": True, "installed": True, "state": "Disabled", "last_result": 0}

        self.assertTrue(pane.scheduler_state_needs_action(task))
        self.assertTrue(pane.scheduled_task_needs_action(task))
        self.assertTrue(pane.scheduled_task_snapshot(task)["needs_action"])
        self.assertEqual(pane.scheduled_task_label(task), "Disabled/disabled")

    def test_setup_plan_is_repo_relative_and_has_no_worktree_literal(self) -> None:
        config = pane.normalize_config(
            {
                **pane.default_config(self.root),
                "python": sys.executable,
                "job_dir": "../job",
                "watchdogs": {},
            },
            self.root,
        )

        with mock.patch.object(pane, "collect_control_tick", return_value={
            "task_name": "FleetControlPaneTick",
            "register_script": "tools/register_control_pane_tick.ps1",
            "register_script_exists": False,
            "task": {"supported": True, "installed": False},
        }):
            plan = pane.setup_plan(config)
        blob = json.dumps(plan)

        self.assertEqual(plan["schema"], pane.SETUP_SCHEMA)
        self.assertIn("tools/fleet_control_pane.py", blob)
        self.assertNotIn("C:\\work\\fleet", blob)
        self.assertTrue(any(step["id"] == "control-tick" for step in plan["steps"]))

    def test_setup_plan_marks_done_and_todo_steps(self) -> None:
        local = self.root / "tools" / "_registry" / "control_pane.local.json"
        local.parent.mkdir(parents=True)
        local.write_text(json.dumps({"schema": pane.CONFIG_SCHEMA}), encoding="utf-8")
        (self.root / "tools" / "_registry" / "machines").mkdir(parents=True)
        (self.root / "tools" / "_watchdog").mkdir(parents=True)
        script = self.root / "tools" / "register_control_pane_tick.ps1"
        script.write_text("", encoding="utf-8")
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(local)},
            "_local_exists": True,
        }

        with mock.patch.object(pane, "collect_control_tick", return_value={
            "task_name": "FleetControlPaneTick",
            "register_script": "tools/register_control_pane_tick.ps1",
            "register_script_exists": True,
            "task": {"supported": True, "installed": False},
        }):
            plan = pane.setup_plan(config)

        steps = {step["id"]: step for step in plan["steps"]}
        self.assertEqual(steps["local-config"]["state"], "done")
        self.assertEqual(steps["runtime-dirs"]["state"], "done")
        self.assertEqual(steps["control-tick"]["state"], "manual")
        self.assertEqual(steps["install-control-pane-tick"]["state"], "todo")
        self.assertIn("bootstrap --apply", steps["runtime-dirs"]["display"])

    def test_setup_plan_marks_installed_tick_with_missing_runner_todo(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "watchdogs": {},
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
        }
        tick = {
            "task_name": "FleetControlPaneTick",
            "register_script": str(Path("tools") / "register_control_pane_tick.ps1"),
            "register_script_exists": True,
            "runner": str(Path("tools") / "_registry" / "control_pane_tick.cmd"),
            "runner_exists": False,
            "task": {"supported": True, "installed": True},
        }

        plan = pane.setup_plan(config, tick=tick)

        steps = {step["id"]: step for step in plan["steps"]}
        self.assertEqual(steps["install-control-pane-tick"]["state"], "todo")
        self.assertIn("control-pane tick runner missing", steps["install-control-pane-tick"]["detail"])
        self.assertIn("register_control_pane_tick.ps1", steps["install-control-pane-tick"]["display"])

    def test_setup_plan_marks_disabled_tick_todo(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "watchdogs": {},
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
        }
        tick = {
            "task_name": "FleetControlPaneTick",
            "register_script": str(Path("tools") / "register_control_pane_tick.ps1"),
            "register_script_exists": True,
            "runner": str(Path("tools") / "_registry" / "control_pane_tick.cmd"),
            "runner_exists": True,
            "task": {"supported": True, "installed": True, "state": "Disabled", "last_result": 0},
        }

        plan = pane.setup_plan(config, tick=tick)

        steps = {step["id"]: step for step in plan["steps"]}
        self.assertEqual(steps["install-control-pane-tick"]["state"], "todo")
        self.assertIn("control-pane tick task is Disabled", steps["install-control-pane-tick"]["detail"])
        self.assertIn("register_control_pane_tick.ps1", steps["install-control-pane-tick"]["display"])

    def test_setup_plan_flags_local_config_default_drift(self) -> None:
        (self.root / "tools" / "control_pane.example.json").write_text(
            json.dumps({
                "schema": pane.CONFIG_SCHEMA,
                "machine_dir": "../shared/machines",
            }),
            encoding="utf-8",
        )
        local = self.root / "tools" / "_registry" / "control_pane.local.json"
        local.parent.mkdir(parents=True)
        local.write_text(
            json.dumps({
                "schema": pane.CONFIG_SCHEMA,
                "machine_dir": "tools/_registry/machines",
            }),
            encoding="utf-8",
        )
        config = pane.load_config(self.root)

        with mock.patch.object(pane, "collect_control_tick", return_value={
            "task_name": "FleetControlPaneTick",
            "register_script": "tools/register_control_pane_tick.ps1",
            "register_script_exists": False,
            "task": {"supported": True, "installed": False},
        }):
            plan = pane.setup_plan(config)

        steps = {step["id"]: step for step in plan["steps"]}
        self.assertEqual(steps["local-config-tracked-defaults"]["state"], "todo")
        self.assertFalse(steps["local-config-tracked-defaults"]["ok"])
        self.assertIn("machine_dir local=", steps["local-config-tracked-defaults"]["detail"])
        self.assertIn("init --force", steps["local-config-tracked-defaults"]["display"])
        self.assertIn("--machine-dir", steps["local-config-tracked-defaults"]["display"])

    def test_setup_plan_includes_loop_command_readiness(self) -> None:
        status = self.root / "tools" / "loop_status.py"
        recover = self.root / "tools" / "loop_recover.py"
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text("print('{\"ok\": true}')\n", encoding="utf-8")
        recover.write_text("print('recover')\n", encoding="utf-8")
        config = pane.normalize_config({
            **pane.default_config(self.root),
            "python": sys.executable,
            "loops": {
                "healthy loop": {
                    "status_cmd": [sys.executable, "tools/loop_status.py"],
                    "recover_cmd": [sys.executable, "tools/loop_recover.py"],
                    "auto_recover": True,
                },
                "disabled": {"enabled": False, "status_cmd": [sys.executable, "missing.py"]},
            },
        }, self.root)

        with mock.patch.object(pane, "collect_control_tick", return_value={
            "task_name": "FleetControlPaneTick",
            "register_script": "tools/register_control_pane_tick.ps1",
            "register_script_exists": False,
            "task": {"supported": True, "installed": False},
        }):
            plan = pane.setup_plan(config)

        steps = {step["id"]: step for step in plan["steps"]}
        self.assertEqual(steps["loop-healthy-loop-status"]["state"], "done")
        self.assertEqual(steps["loop-healthy-loop-recover"]["state"], "done")
        self.assertIn("tools/loop_status.py", steps["loop-healthy-loop-status"]["display"])
        self.assertNotIn("loop-disabled-status", steps)

    def test_setup_plan_blocks_missing_loop_inputs(self) -> None:
        config = pane.normalize_config({
            **pane.default_config(self.root),
            "python": sys.executable,
            "loops": {
                "stalled": {
                    "status_cmd": [sys.executable, "tools/missing_status.py"],
                    "auto_recover": True,
                },
            },
        }, self.root)

        with mock.patch.object(pane, "collect_control_tick", return_value={
            "task_name": "FleetControlPaneTick",
            "register_script": "tools/register_control_pane_tick.ps1",
            "register_script_exists": False,
            "task": {"supported": True, "installed": False},
        }):
            plan = pane.setup_plan(config)

        steps = {step["id"]: step for step in plan["steps"]}
        self.assertEqual(steps["loop-stalled-status"]["state"], "blocked")
        self.assertIn("missing input: tools/missing_status.py", steps["loop-stalled-status"]["detail"])
        self.assertEqual(steps["loop-stalled-recover"]["state"], "blocked")
        self.assertIn("auto_recover is true but recover_cmd is not configured", steps["loop-stalled-recover"]["detail"])

    def test_doctor_reports_missing_loop_inputs(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
            "loops": {
                "stalled": {
                    "status_cmd": [sys.executable, "tools/missing_status.py"],
                    "recover_cmd": [sys.executable, "tools/missing_recover.py"],
                },
            },
        }

        with (
            mock.patch.object(pane, "collect_control_tick", return_value={
                "task_name": "FleetControlPaneTick",
                "register_script": "tools/register_control_pane_tick.ps1",
                "register_script_exists": True,
                "task": {"supported": True, "installed": True},
            }),
            mock.patch.object(pane, "collect_watchdogs", return_value={}),
        ):
            doc = pane.doctor(config)

        checks = {check["name"]: check for check in doc["checks"]}
        self.assertFalse(checks["stalled-loop-status-cmd"]["ok"])
        self.assertIn("missing input: tools/missing_status.py", checks["stalled-loop-status-cmd"]["detail"])
        self.assertFalse(checks["stalled-loop-recover-cmd"]["ok"])
        self.assertIn("missing input: tools/missing_recover.py", checks["stalled-loop-recover-cmd"]["detail"])

    def test_doctor_reports_missing_control_tick_runner(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
            "watchdogs": {},
        }
        tick = {
            "task_name": "FleetControlPaneTick",
            "register_script": str(Path("tools") / "register_control_pane_tick.ps1"),
            "register_script_exists": True,
            "runner": str(Path("tools") / "_registry" / "control_pane_tick.cmd"),
            "runner_exists": False,
            "task": {"supported": True, "installed": True},
        }

        with (
            mock.patch.object(pane, "collect_control_tick", return_value=tick),
            mock.patch.object(pane, "collect_watchdogs", return_value={}),
        ):
            doc = pane.doctor(config)

        checks = {check["name"]: check for check in doc["checks"]}
        self.assertFalse(checks["control-tick-runner"]["ok"])
        self.assertIn("control_pane_tick.cmd", checks["control-tick-runner"]["detail"])

    def test_control_tick_dry_run_invokes_no_watchdogs_and_writes_snapshot(self) -> None:
        config = {
            "root": str(self.root),
            "registry_dir": str(self.root / "tools" / "_registry"),
            "machine_dir": str(self.root / "tools" / "_registry" / "machines"),
            "machine_id": "test-host",
            "job_dir": str(self.root / "job"),
            "target": 1,
            "session_window_h": 1,
            "watchdog_log_dir": str(self.root / "logs"),
            "watchdogs": {
                "supervisor": {"script": "tools/fleet_supervisor_watchdog.ps1"},
                "resume": {"script": "tools/fleet_resume_watchdog.ps1"},
            },
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
        }
        (self.root / "tools" / "fleet_supervisor_watchdog.ps1").write_text("", encoding="utf-8")
        (self.root / "tools" / "fleet_resume_watchdog.ps1").write_text("", encoding="utf-8")
        status = {
            "schema": pane.SCHEMA,
            "generated_utc": "2026-06-18T00:00:00Z",
            "machine": {"host": "h"},
            "config": {"local_config_exists": True},
            "git": {"available": False, "reason": "test"},
            "registry": {"exists": False},
            "supervisor": {"available": False, "reason": "test"},
            "watchdogs": {},
            "actions": [],
            "verdict": "OK",
        }
        with (
            mock.patch.object(platform, "system", return_value="Windows"),
            mock.patch.object(pane, "powershell_exe", return_value="powershell.exe"),
            mock.patch.object(pane, "collect_status", return_value=status),
        ):
            doc = pane.control_tick(config, dry_run=True)

        self.assertEqual(doc["schema"], pane.TICK_SCHEMA)
        self.assertTrue(doc["ok"])
        self.assertEqual([a["skipped"] for a in doc["actions"]], [True, True])
        self.assertTrue((self.root / "tools" / "_registry" / "control_pane.json").exists())
        self.assertTrue((self.root / "tools" / "_registry" / "machines" / "test-host.json").exists())

    def test_fleet_view_folds_fresh_and_stale_machine_snapshots(self) -> None:
        machine_dir = self.root / "tools" / "_registry" / "machines"
        machine_dir.mkdir(parents=True)
        fresh = {
            "schema": pane.SCHEMA,
            "generated_utc": pane.iso_now(),
            "app_version": "1.2.3",
            "verdict": "OK",
            "machine": {"id": "fresh", "host": "fresh"},
            "registry": {"sessions": 2, "accounts": {"available": 1, "total": 1}},
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True, "reason": "up to date"}},
            "supervisor": {"available": True, "payload": {"verdict": "WALL", "process": {"alive": True}}},
            "control_tick": {"task": {"installed": True}},
            "actions": [],
        }
        stale = {
            **fresh,
            "generated_utc": "2020-01-01T00:00:00Z",
            "verdict": "OK",
            "machine": {"id": "stale", "host": "stale"},
        }
        (machine_dir / "fresh.json").write_text(json.dumps(fresh), encoding="utf-8")
        (machine_dir / "stale.json").write_text(json.dumps(stale), encoding="utf-8")

        doc = pane.fleet_view({
            "machine_dir": str(machine_dir),
            "machine_stale_min": 15,
        })

        self.assertEqual(doc["schema"], pane.FLEET_SCHEMA)
        self.assertEqual(doc["verdict"], "ACTION")
        self.assertEqual(doc["states"]["OK"], 1)
        self.assertEqual(doc["states"]["STALE"], 1)
        self.assertEqual(doc["versions"], {"1.2.3": 2})
        self.assertEqual(doc["totals"]["sessions"], 4)
        by_id = {machine["id"]: machine for machine in doc["machines"]}
        self.assertEqual(by_id["fresh"]["app_version"], "1.2.3")
        self.assertEqual(by_id["fresh"]["git"]["safe_ff_reason"], "up to date")

    def test_fleet_view_carries_dirty_plan_from_machine_snapshot(self) -> None:
        machine_dir = self.root / "tools" / "_registry" / "machines"
        machine_dir.mkdir(parents=True)
        snapshot = {
            "schema": pane.SCHEMA,
            "generated_utc": pane.iso_now(),
            "verdict": "ACTION",
            "machine": {"id": "dirty-host", "host": "dirty-host"},
            "registry": {"sessions": 1, "accounts": {"available": 1, "total": 1}},
            "git": {
                "dirty_total": 2,
                "dirty_plan": {
                    "count": 1,
                    "groups": [
                        {
                            "group": "tools/fleet-control-pane",
                            "count": 2,
                            "paths": ["tools/fleet_control_pane.py", "tools/fleet_control_pane_test.py"],
                            "command": "python tools/fleet_control_pane.py commit --dirty-group tools/fleet-control-pane -m msg",
                        },
                    ],
                },
                "safe_ff": {"state": "in-sync", "ok": True},
            },
            "supervisor": {"available": True, "payload": {"verdict": "WALL", "process": {"alive": True}}},
            "control_tick": {"task": {"installed": True}},
            "actions": [],
        }
        (machine_dir / "dirty-host.json").write_text(json.dumps(snapshot), encoding="utf-8")

        doc = pane.fleet_view({
            "machine_dir": str(machine_dir),
            "machine_stale_min": 15,
        })

        machine = doc["machines"][0]
        dirty_plan = machine["git"]["dirty_plan"]
        self.assertEqual(dirty_plan["count"], 1)
        self.assertEqual(dirty_plan["groups"][0]["group"], "tools/fleet-control-pane")
        self.assertEqual(dirty_plan["groups"][0]["paths"], ["tools/fleet_control_pane.py", "tools/fleet_control_pane_test.py"])

    def test_fleet_view_flags_pane_version_drift(self) -> None:
        machine_dir = self.root / "tools" / "_registry" / "machines"
        machine_dir.mkdir(parents=True)
        snapshot = {
            "schema": pane.SCHEMA,
            "generated_utc": pane.iso_now(),
            "app_version": "0.8.1",
            "verdict": "OK",
            "machine": {"id": "old-host", "host": "old-host"},
            "registry": {"sessions": 1, "accounts": {"available": 1, "total": 1}},
            "git": {"dirty_total": 0, "safe_ff": {"state": "behind", "ok": True, "reason": "fast-forward available"}},
            "supervisor": {"available": True, "payload": {"verdict": "WALL", "process": {"alive": True}}},
            "control_tick": {"task": {"installed": True}},
            "actions": [],
        }
        (machine_dir / "old-host.json").write_text(json.dumps(snapshot), encoding="utf-8")

        doc = pane.fleet_view({
            "root": str(self.root),
            "machine_dir": str(machine_dir),
            "machine_stale_min": 15,
        })

        self.assertEqual(doc["current_version"], "test")
        self.assertEqual(doc["verdict"], "ACTION")
        self.assertEqual(doc["totals"]["version_mismatches"], 1)
        drift = doc["machines"][0]["version_drift"]
        self.assertEqual(drift["machine_version"], "0.8.1")
        self.assertEqual(drift["current_version"], "test")
        self.assertEqual(drift["command"], "python tools/fleet_control_pane.py sync --fetch --json")

    def test_fleet_view_carries_setup_task_summary(self) -> None:
        machine_dir = self.root / "tools" / "_registry" / "machines"
        machine_dir.mkdir(parents=True)
        snapshot = {
            "schema": pane.SCHEMA,
            "generated_utc": pane.iso_now(),
            "verdict": "ACTION",
            "machine": {"id": "setup-host", "host": "setup-host"},
            "registry": {"sessions": 1, "accounts": {"available": 1, "total": 1}},
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
            "supervisor": {"available": True, "payload": {"verdict": "WALL", "process": {"alive": True}}},
            "control_tick": {"task": {"supported": True, "installed": True, "state": "Ready", "last_result": 1}},
            "watchdogs": {
                "resume": {"task": {"supported": True, "installed": False}},
                "supervisor": {"task": {"supported": False, "reason": "not Windows"}},
            },
            "actions": [],
        }
        (machine_dir / "setup-host.json").write_text(json.dumps(snapshot), encoding="utf-8")

        doc = pane.fleet_view({
            "machine_dir": str(machine_dir),
            "machine_stale_min": 15,
        })

        machine = doc["machines"][0]
        self.assertTrue(machine["control_tick"]["needs_action"])
        self.assertIn("0x00000001", machine["control_tick"]["last_result_text"])
        self.assertFalse(machine["watchdogs"]["resume"]["installed"])
        self.assertEqual(machine["watchdogs"]["supervisor"]["supported"], False)
        self.assertEqual(doc["totals"]["setup_actions"], 2)

    def test_fleet_view_preserves_blocked_account_source_snapshot(self) -> None:
        machine_dir = self.root / "tools" / "_registry" / "machines"
        machine_dir.mkdir(parents=True)
        snapshot = {
            "schema": pane.SCHEMA,
            "generated_utc": pane.iso_now(),
            "verdict": "ACTION",
            "machine": {"id": "acct-host", "host": "acct-hostname"},
            "registry": {
                "sessions": 1,
                "auth_blocked": 1,
                "accounts": {
                    "available": 0,
                    "total": 1,
                    "blocked_count": 1,
                    "auth_blocked_count": 1,
                    "blocked": [
                        {
                            "tag": "c10",
                            "account": ".claude-c10",
                            "config_dir": "C:/Users/example/.claude-c10",
                            "block_kind": "auth",
                            "reason": "auth/login required",
                            "auth_blocked_sessions": 1,
                            "command": "$env:CLAUDE_CONFIG_DIR='C:/Users/example/.claude-c10'; claude",
                        }
                    ],
                },
            },
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
            "supervisor": {"available": True, "payload": {"verdict": "WALL", "process": {"alive": True}}},
            "control_tick": {"task": {"installed": True}},
            "actions": ["inspect account"],
        }
        path = machine_dir / "acct-host.json"
        path.write_text(json.dumps(snapshot), encoding="utf-8")

        doc = pane.fleet_view({
            "machine_dir": str(machine_dir),
            "machine_stale_min": 15,
        })

        machine = doc["machines"][0]
        blocked = machine["accounts"]["blocked"][0]
        self.assertEqual(blocked["tag"], "c10")
        self.assertEqual(blocked["source_machine"], "acct-host")
        self.assertEqual(blocked["source_host"], "acct-hostname")
        self.assertEqual(blocked["source_snapshot"], str(path))
        self.assertEqual(doc["totals"]["blocked_accounts"], 1)
        self.assertEqual(doc["totals"]["auth_blocked_accounts"], 1)

    def test_fleet_view_carries_degraded_supervisor_summary(self) -> None:
        machine_dir = self.root / "tools" / "_registry" / "machines"
        machine_dir.mkdir(parents=True)
        job_dir = self.root / "job"
        run_dir = job_dir / "output" / "supervise-20260618T211004Z"
        log_dir = run_dir / "worker-1"
        log_dir.mkdir(parents=True)
        log_path = log_dir / "run.log"
        log_path.write_text('{"type":"result"}\n', encoding="utf-8")
        snapshot = {
            "schema": pane.SCHEMA,
            "generated_utc": pane.iso_now(),
            "verdict": "ACTION",
            "machine": {"id": "sup-host", "host": "sup-host"},
            "config": {"job_dir": str(job_dir)},
            "registry": {"sessions": 1, "accounts": {"available": 1, "total": 1}},
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
            "supervisor": {
                "available": True,
                "payload": {
                    "verdict": "WALL",
                    "run": "supervise-20260618T211004Z",
                    "process": {"alive": True},
                    "last_decide": {"run_health": "DEGRADED"},
                    "diagnose": {"health": "STALLED", "primary_cause": "unclear", "primary_action": "inspect run"},
                },
            },
            "control_tick": {"task": {"installed": True}},
            "watchdogs": {
                "supervisor": {"task": {"supported": True, "installed": True, "state": "Ready", "last_result": 10}},
            },
            "actions": ["inspect supervisor"],
        }
        (machine_dir / "sup-host.json").write_text(json.dumps(snapshot), encoding="utf-8")

        doc = pane.fleet_view({
            "machine_dir": str(machine_dir),
            "machine_stale_min": 15,
        })

        machine = doc["machines"][0]
        self.assertTrue(machine["supervisor"]["needs_action"])
        self.assertEqual(machine["supervisor"]["run_health"], "DEGRADED")
        self.assertEqual(machine["supervisor"]["run"], "supervise-20260618T211004Z")
        self.assertEqual(machine["supervisor"]["run_log"], str(log_path))
        self.assertTrue(machine["supervisor"]["watchdog_last_result_accepted"])
        self.assertFalse(machine["watchdogs"]["supervisor"]["needs_action"])
        self.assertEqual(doc["totals"]["setup_actions"], 0)

    def test_fleet_view_carries_setup_plan_from_machine_snapshot(self) -> None:
        machine_dir = self.root / "tools" / "_registry" / "machines"
        machine_dir.mkdir(parents=True)
        snapshot = {
            "schema": pane.SCHEMA,
            "generated_utc": pane.iso_now(),
            "verdict": "ACTION",
            "machine": {"id": "setup-plan-host", "host": "setup-plan-host"},
            "registry": {"sessions": 1, "accounts": {"available": 1, "total": 1}},
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
            "supervisor": {"available": True, "payload": {"verdict": "WALL", "process": {"alive": True}}},
            "control_tick": {"task": {"installed": True}},
            "setup_plan": {
                "steps": [
                    {
                        "id": "local-config",
                        "state": "done",
                        "ok": True,
                        "detail": "ready",
                        "command": ["python", "tools/fleet_control_pane.py", "init"],
                        "display": "python tools/fleet_control_pane.py init",
                    },
                    {
                        "id": "install-control-pane-tick",
                        "state": "todo",
                        "ok": True,
                        "detail": "install the recurring control-pane tick",
                        "command": ["python", "tools/fleet_control_pane.py", "bootstrap", "--apply"],
                        "display": "python tools/fleet_control_pane.py bootstrap --apply",
                    },
                ],
            },
            "actions": [],
        }
        (machine_dir / "setup-plan-host.json").write_text(json.dumps(snapshot), encoding="utf-8")

        doc = pane.fleet_view({
            "machine_dir": str(machine_dir),
            "machine_stale_min": 15,
        })

        setup_plan = doc["machines"][0]["setup_plan"]
        self.assertEqual(setup_plan["states"], {"done": 1, "todo": 1})
        self.assertEqual(setup_plan["action_count"], 1)
        self.assertEqual(setup_plan["steps"][1]["command"], ["python", "tools/fleet_control_pane.py", "bootstrap", "--apply"])
        self.assertEqual(doc["totals"]["setup_plan_actions"], 1)

    def test_fleet_view_carries_loop_check_summaries(self) -> None:
        machine_dir = self.root / "tools" / "_registry" / "machines"
        machine_dir.mkdir(parents=True)
        snapshot = {
            "schema": pane.SCHEMA,
            "generated_utc": pane.iso_now(),
            "verdict": "ACTION",
            "machine": {"id": "loop-host", "host": "loop-host"},
            "registry": {"sessions": 1, "accounts": {"available": 1, "total": 1}},
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
            "supervisor": {"available": True, "payload": {"verdict": "WALL", "process": {"alive": True}}},
            "control_tick": {"task": {"installed": True}},
            "loops": {
                "count": 2,
                "states": {"OK": 1, "ACTION": 1},
                "checks": [
                    {"name": "healthy", "state": "OK", "detail": "ok"},
                    {
                        "name": "stalled",
                        "state": "ACTION",
                        "detail": "heartbeat stale",
                        "action": "restart stalled",
                        "auto_recover": True,
                        "has_recover_cmd": True,
                        "returncode": 3,
                        "stdout": "large output omitted from fleet summary",
                    },
                ],
            },
            "actions": [],
        }
        (machine_dir / "loop-host.json").write_text(json.dumps(snapshot), encoding="utf-8")

        doc = pane.fleet_view({
            "machine_dir": str(machine_dir),
            "machine_stale_min": 15,
        })

        loops = doc["machines"][0]["loops"]
        self.assertEqual(loops["action"], 1)
        self.assertEqual(doc["totals"]["loop_actions"], 1)
        self.assertEqual(loops["checks"][1]["name"], "stalled")
        self.assertEqual(loops["checks"][1]["detail"], "heartbeat stale")
        self.assertNotIn("stdout", loops["checks"][1])

    def test_fleet_view_carries_loop_inventory_commands(self) -> None:
        machine_dir = self.root / "tools" / "_registry" / "machines"
        machine_dir.mkdir(parents=True)
        snapshot = {
            "schema": pane.SCHEMA,
            "generated_utc": pane.iso_now(),
            "verdict": "ACTION",
            "machine": {"id": "loopless-host", "host": "loopless-host"},
            "registry": {"sessions": 1, "accounts": {"available": 1, "total": 1}},
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
            "supervisor": {"available": True, "payload": {"verdict": "WALL", "process": {"alive": True}}},
            "control_tick": {"task": {"installed": True}},
            "loops": {
                "count": 0,
                "configured": 1,
                "enabled": 0,
                "disabled": 1,
                "commands": ["python tools/fleet_control_pane.py loop-scaffold NAME"],
                "checks": [],
            },
            "actions": ["No enabled loops configured."],
        }
        (machine_dir / "loopless-host.json").write_text(json.dumps(snapshot), encoding="utf-8")

        doc = pane.fleet_view({
            "machine_dir": str(machine_dir),
            "machine_stale_min": 15,
        })
        text = pane.fleet_text(doc)

        loops = doc["machines"][0]["loops"]
        self.assertEqual(loops["configured"], 1)
        self.assertEqual(loops["enabled"], 0)
        self.assertEqual(loops["commands"], ["python tools/fleet_control_pane.py loop-scaffold NAME"])
        self.assertIn("loop-action: python tools/fleet_control_pane.py loop-scaffold NAME", text)

    def test_fleet_view_overlays_current_machine_with_live_status(self) -> None:
        machine_dir = self.root / "tools" / "_registry" / "machines"
        machine_dir.mkdir(parents=True)
        stale = {
            "schema": pane.SCHEMA,
            "generated_utc": "2020-01-01T00:00:00Z",
            "verdict": "ACTION",
            "machine": {"id": "test-host", "host": "test-host"},
            "registry": {"sessions": 1, "accounts": {"available": 0, "total": 1}},
            "git": {"dirty_total": 9, "safe_ff": {"state": "ahead", "ok": False}},
            "supervisor": {"available": False},
            "control_tick": {"task": {"installed": False}},
            "actions": ["old action"],
        }
        live = {
            **stale,
            "generated_utc": pane.iso_now(),
            "verdict": "OK",
            "registry": {"sessions": 5, "accounts": {"available": 1, "total": 1}},
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
            "actions": [],
        }
        (machine_dir / "test-host.json").write_text(json.dumps(stale), encoding="utf-8")
        config = {
            "root": str(self.root),
            "machine_dir": str(machine_dir),
            "machine_id": "test-host",
            "machine_stale_min": 15,
        }

        with mock.patch.object(pane, "collect_status", return_value=live) as collect_mock:
            doc = pane.fleet_view(config, include_live_local=True)

        self.assertEqual(len(doc["machines"]), 1)
        machine = doc["machines"][0]
        self.assertTrue(machine["live_local"])
        self.assertEqual(machine["state"], "OK")
        self.assertEqual(machine["sessions"], 5)
        self.assertEqual(doc["states"], {"OK": 1})
        collect_mock.assert_called_once_with(config, refresh=False)

    def test_fleet_view_can_refresh_and_write_current_machine(self) -> None:
        machine_dir = self.root / "tools" / "_registry" / "machines"
        machine_dir.mkdir(parents=True)
        live = {
            "schema": pane.SCHEMA,
            "generated_utc": pane.iso_now(),
            "verdict": "OK",
            "machine": {"id": "test-host", "host": "test-host"},
            "registry": {"sessions": 5, "accounts": {"available": 1, "total": 1}},
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
            "supervisor": {"available": True, "payload": {"verdict": "WALL", "process": {"alive": True}}},
            "control_tick": {"task": {"installed": True}},
            "actions": [],
        }
        config = {
            "root": str(self.root),
            "machine_dir": str(machine_dir),
            "registry_dir": str(self.root / "tools" / "_registry"),
            "machine_id": "test-host",
            "machine_stale_min": 15,
        }
        written = {"json": "pane.json", "text": "pane.txt", "machine": "machine.json"}

        with (
            mock.patch.object(pane, "collect_status", return_value=live) as collect_mock,
            mock.patch.object(pane, "write_status", return_value=written) as write_mock,
        ):
            doc = pane.fleet_view(
                config,
                include_live_local=True,
                refresh_live_local=True,
                write_live_local=True,
            )

        self.assertEqual(doc["written"], written)
        self.assertTrue(doc["refresh_live_local"])
        self.assertTrue(doc["machines"][0]["refresh_local"])
        collect_mock.assert_called_once_with(config, refresh=True)
        write_mock.assert_called_once_with(live, config)

    def test_watchdog_command_builds_resume_live_command_on_windows(self) -> None:
        config = {
            "root": str(self.root),
            "session_window_h": 2,
            "registry_dir": str(self.root / "tools" / "_registry"),
            "claude_exe": "claude.exe",
            "watchdog_log_dir": str(self.root / "logs"),
            "watchdogs": {"resume": {"script": "tools/fleet_resume_watchdog.ps1"}},
        }
        (self.root / "tools" / "fleet_resume_watchdog.ps1").write_text("", encoding="utf-8")
        with (
            mock.patch.object(platform, "system", return_value="Windows"),
            mock.patch.object(pane, "powershell_exe", return_value="powershell.exe"),
        ):
            cmd = pane.watchdog_command("resume", config, live_resume=True)

        self.assertIsNotNone(cmd)
        self.assertIn("-Live", cmd)
        self.assertIn("-WindowH", cmd)
        self.assertIn("2", cmd)
        self.assertIn("-RegistryDir", cmd)
        self.assertIn(str(Path(config["root"]) / "tools" / "_registry"), cmd)

    def test_collect_supervisor_accepts_json_even_with_nonzero_returncode(self) -> None:
        proc = subprocess.CompletedProcess(
            args=[],
            returncode=3,
            stdout='warning\n{"verdict":"WALL","process":{"alive":true}}\n',
            stderr="",
        )
        with (
            mock.patch.object(pane, "command_exists", return_value=True),
            mock.patch.object(pane, "run", return_value=proc),
        ):
            got = pane.collect_supervisor({
                "root": str(self.root),
                "job_dir": str(self.root),
                "python": sys.executable,
                "supervisor_status_cmd": ["{python}", "supervise_now.py", "--json"],
            })

        self.assertTrue(got["available"])
        self.assertEqual(got["returncode"], 3)
        self.assertEqual(got["payload"]["verdict"], "WALL")

    def test_default_supervisor_status_is_fleet_native(self) -> None:
        config = pane.default_config(self.root)

        self.assertEqual(config["supervisor_status_cmd"], ["{python}", "tools/dos_supervisor_status.py", "--json"])

    def test_collect_supervisor_without_job_dir_for_fleet_native_command(self) -> None:
        proc = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"verdict":"READY_TO_CANARY","ok":true}\n',
            stderr="",
        )
        with (
            mock.patch.object(pane, "command_exists", return_value=True),
            mock.patch.object(pane, "run", return_value=proc),
        ):
            got = pane.collect_supervisor({
                "root": str(self.root),
                "job_dir": "",
                "python": sys.executable,
                "supervisor_status_cmd": ["{python}", "tools/dos_supervisor_status.py", "--json"],
            })

        self.assertTrue(got["available"])
        self.assertEqual(got["payload"]["verdict"], "READY_TO_CANARY")

    def test_collect_supervisor_requires_job_dir_only_for_job_placeholder(self) -> None:
        with mock.patch.object(pane, "command_exists", return_value=True):
            got = pane.collect_supervisor({
                "root": str(self.root),
                "job_dir": "",
                "python": sys.executable,
                "supervisor_status_cmd": ["{python}", "{job_dir}/scripts/supervise_now.py", "--json"],
            })

        self.assertFalse(got["available"])
        self.assertEqual(got["reason"], "job_dir is not configured")

    def test_control_tick_invokes_auto_recover_loop_and_rewrites_status(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
            "loops": {
                "stalled": {
                    "status_cmd": [sys.executable, "-V"],
                    "recover_cmd": [sys.executable, "-V"],
                    "auto_recover": True,
                },
            },
        }
        before = {
            "schema": pane.SCHEMA,
            "generated_utc": "before",
            "machine": {"host": "h"},
            "loops": {"checks": [{"name": "stalled", "state": "ACTION"}]},
            "actions": ["Loop stalled is ACTION"],
            "verdict": "ACTION",
        }
        after = {
            "schema": pane.SCHEMA,
            "generated_utc": "after",
            "machine": {"host": "h"},
            "loops": {"checks": [{"name": "stalled", "state": "OK"}]},
            "actions": [],
            "verdict": "OK",
        }
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="Python\n", stderr="")

        with (
            mock.patch.object(pane, "git_mutation_guard", return_value={"blocked": False, "blockers": [], "reason": "", "worktree": {}}),
            mock.patch.object(pane, "collect_status", side_effect=[before, after]) as collect_mock,
            mock.patch.object(pane, "run", return_value=proc) as run_mock,
            mock.patch.object(pane, "write_status", return_value={"json": "pane"}) as write_mock,
        ):
            doc = pane.control_tick(config, skip_supervisor=True, skip_resume=True)

        self.assertTrue(doc["ok"])
        self.assertEqual(doc["status"]["generated_utc"], "after")
        self.assertEqual(doc["actions"][0]["name"], "loop:stalled")
        self.assertFalse(doc["actions"][0]["skipped"])
        self.assertEqual(collect_mock.call_count, 2)
        self.assertEqual(collect_mock.call_args_list[0].kwargs["skip_loop_names"], {"control-pane-doctor"})
        self.assertEqual(collect_mock.call_args_list[1].kwargs["skip_loop_names"], {"control-pane-doctor"})
        run_mock.assert_called_once()
        write_mock.assert_called_once_with(after, config)

    def test_control_tick_skips_self_bootstrap_auto_recovery(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
            "loops": {
                "control-pane-doctor": {
                    "status_cmd": [sys.executable, "tools/fleet_control_pane.py", "doctor", "--json"],
                    "recover_cmd": [sys.executable, "tools/fleet_control_pane.py", "bootstrap", "--apply"],
                    "auto_recover": True,
                },
            },
        }
        status = {
            "schema": pane.SCHEMA,
            "generated_utc": "before",
            "machine": {"host": "h"},
            "loops": {"checks": [{"name": "control-pane-doctor", "state": "ACTION"}]},
            "actions": ["Loop control-pane-doctor is ACTION"],
            "verdict": "ACTION",
        }

        with (
            mock.patch.object(pane, "git_mutation_guard", return_value={"blocked": False, "blockers": [], "reason": "", "worktree": {}}),
            mock.patch.object(pane, "collect_status", return_value=status) as collect_mock,
            mock.patch.object(pane, "run") as run_mock,
            mock.patch.object(pane, "write_status", return_value={"json": "pane"}),
        ):
            doc = pane.control_tick(config, skip_supervisor=True, skip_resume=True)

        self.assertTrue(doc["ok"])
        self.assertEqual(doc["actions"][0]["name"], "loop:control-pane-doctor")
        self.assertTrue(doc["actions"][0]["skipped"])
        self.assertIn("bootstrap recovery is skipped", doc["actions"][0]["reason"])
        collect_mock.assert_called_once()
        run_mock.assert_not_called()

    def test_control_tick_skips_live_actions_during_merge(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
            "loops": {
                "stalled": {
                    "status_cmd": [sys.executable, "-V"],
                    "recover_cmd": [sys.executable, "-V"],
                    "auto_recover": True,
                },
            },
        }
        status = {
            "schema": pane.SCHEMA,
            "generated_utc": "during-merge",
            "machine": {"host": "h"},
            "loops": {"checks": [{"name": "stalled", "state": "ACTION"}]},
            "git": {"merge_in_progress": True, "unmerged_paths": ["VERSION"]},
            "actions": ["Finish or abort the in-progress merge before commit/publish."],
            "verdict": "ACTION",
        }
        guard = {
            "blocked": True,
            "blockers": ["merge is in progress", "worktree has 1 unmerged path(s)"],
            "reason": "merge is in progress; worktree has 1 unmerged path(s)",
            "worktree": {
                "status_available": True,
                "dirty": True,
                "dirty_total": 1,
                "counts": {"unmerged": 1},
                "unmerged_paths": ["VERSION"],
                "merge_in_progress": True,
            },
        }

        with (
            mock.patch.object(pane, "git_mutation_guard", return_value=guard),
            mock.patch.object(pane, "collect_status", return_value=status) as collect_mock,
            mock.patch.object(pane, "invoke_watchdog") as watchdog_mock,
            mock.patch.object(pane, "invoke_loop_recoveries") as recoveries_mock,
            mock.patch.object(pane, "write_status", return_value={"json": "pane"}) as write_mock,
        ):
            doc = pane.control_tick(config, dry_run=False, live_resume=True)

        self.assertTrue(doc["ok"])
        self.assertEqual(doc["mutation_guard"], guard)
        self.assertEqual(doc["actions"][0]["name"], "git-merge-guard")
        self.assertTrue(doc["actions"][0]["skipped"])
        self.assertIn("merge/unmerged worktree detected", doc["actions"][0]["reason"])
        watchdog_mock.assert_not_called()
        recoveries_mock.assert_not_called()
        collect_mock.assert_called_once_with(config, refresh=True, skip_loop_names={"control-pane-doctor"})
        write_mock.assert_called_once_with(status, config)

    def test_recover_plan_dry_run_uses_live_resume_without_applying(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
        }
        tick = {
            "schema": pane.TICK_SCHEMA,
            "ok": True,
            "actions": [{"name": "resume", "skipped": True, "ok": True}],
            "status": {"schema": pane.SCHEMA, "verdict": "OK"},
            "written": None,
        }

        with mock.patch.object(pane, "control_tick", return_value=tick) as tick_mock:
            doc = pane.recover_plan(config, apply=False, skip_supervisor=True, write=False)

        self.assertEqual(doc["schema"], pane.RECOVER_SCHEMA)
        self.assertTrue(doc["ok"])
        self.assertFalse(doc["apply"])
        self.assertFalse(doc["resume_plan"]["exists"])
        tick_mock.assert_called_once_with(
            config,
            dry_run=True,
            live_resume=True,
            skip_supervisor=True,
            skip_resume=False,
            write=False,
        )

    def test_recover_plan_includes_resume_plan_summary(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
        }
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
            ],
        }), encoding="utf-8")
        tick = {
            "schema": pane.TICK_SCHEMA,
            "ok": True,
            "actions": [{"name": "resume", "skipped": True, "ok": True}],
            "status": {"schema": pane.SCHEMA, "verdict": "OK"},
            "written": None,
        }

        with mock.patch.object(pane, "control_tick", return_value=tick):
            doc = pane.recover_plan(config, apply=False)

        self.assertEqual(doc["resume_plan"]["count"], 1)
        self.assertEqual(doc["resume_plan"]["sessions"][0]["session_short"], "abcdef12")
        self.assertEqual(doc["resume_plan"]["sessions"][0]["account"], ".claude-gem8")

    def test_recover_plan_apply_runs_live_recovery(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
        }
        tick = {
            "schema": pane.TICK_SCHEMA,
            "ok": True,
            "actions": [{"name": "resume", "skipped": False, "ok": True}],
            "status": {"schema": pane.SCHEMA, "verdict": "OK"},
            "written": {"json": "pane"},
        }

        with mock.patch.object(pane, "control_tick", return_value=tick) as tick_mock:
            doc = pane.recover_plan(config, apply=True)

        self.assertTrue(doc["apply"])
        self.assertEqual(doc["actions"], tick["actions"])
        self.assertEqual(doc["written"], tick["written"])
        tick_mock.assert_called_once_with(
            config,
            dry_run=False,
            live_resume=True,
            skip_supervisor=False,
            skip_resume=False,
            write=True,
        )

    def test_supervisor_plan_inspects_status(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
        }
        status = {
            "available": True,
            "payload": {"verdict": "STALLED", "process": {"alive": True, "pid": 123}},
        }

        with mock.patch.object(pane, "collect_supervisor", return_value=status):
            doc = pane.supervisor_plan(config)

        self.assertEqual(doc["schema"], pane.SUPERVISOR_SCHEMA)
        self.assertTrue(doc["ok"])
        self.assertTrue(doc["needs_action"])
        self.assertFalse(doc["restart"])
        self.assertEqual(doc["before"], status)
        self.assertEqual(doc["actions"], [])

    def test_supervisor_plan_trusts_non_action_verdict_over_stale_diagnosis(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
        }
        status = {
            "available": True,
            "payload": {
                "verdict": "WALL",
                "process": {"alive": True, "pid": 123},
                "diagnose": {"health": "STALLED", "primary_action": "inspect old run"},
            },
        }

        with mock.patch.object(pane, "collect_supervisor", return_value=status):
            doc = pane.supervisor_plan(config)

        self.assertTrue(doc["ok"])
        self.assertFalse(doc["needs_action"])
        self.assertEqual(doc["actions"], [])

    def test_supervisor_plan_dry_run_restart_does_not_terminate(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
        }
        status = {
            "available": True,
            "payload": {"verdict": "STALLED", "process": {"alive": True, "pid": 123}},
        }
        watchdog = {"name": "supervisor", "ok": True, "skipped": True, "reason": "dry run"}

        with (
            mock.patch.object(pane, "collect_supervisor", return_value=status),
            mock.patch.object(pane, "terminate_pid") as terminate_mock,
            mock.patch.object(pane, "invoke_watchdog", return_value=watchdog) as watchdog_mock,
        ):
            doc = pane.supervisor_plan(config, restart=True, apply=False)

        self.assertTrue(doc["ok"])
        self.assertTrue(doc["needs_action"])
        self.assertEqual(doc["actions"][0]["name"], "terminate-supervisor")
        self.assertTrue(doc["actions"][0]["skipped"])
        terminate_mock.assert_not_called()
        watchdog_mock.assert_called_once_with("supervisor", config, dry_run=True)

    def test_supervisor_plan_apply_restart_terminates_and_runs_watchdog(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
        }
        before = {
            "available": True,
            "payload": {"verdict": "STALLED", "process": {"alive": True, "pid": 123}},
        }
        after = {
            "available": True,
            "payload": {"verdict": "WALL", "process": {"alive": True, "pid": 456}},
        }
        watchdog = {"name": "supervisor", "ok": True, "skipped": False, "returncode": 10}

        with (
            mock.patch.object(pane, "collect_supervisor", side_effect=[before, after]),
            mock.patch.object(pane, "terminate_pid", return_value={"ok": True}) as terminate_mock,
            mock.patch.object(pane, "invoke_watchdog", return_value=watchdog) as watchdog_mock,
        ):
            doc = pane.supervisor_plan(config, restart=True, apply=True)

        self.assertTrue(doc["ok"])
        self.assertFalse(doc["needs_action"])
        self.assertEqual(doc["after"], after)
        terminate_mock.assert_called_once_with(123, self.root)
        watchdog_mock.assert_called_once_with("supervisor", config, dry_run=False)

    def test_bootstrap_dry_run_prints_init_and_install_without_writing(self) -> None:
        script = self.root / "tools" / "register_control_pane_tick.ps1"
        script.write_text("", encoding="utf-8")
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": False,
        }
        with (
            mock.patch.object(platform, "system", return_value="Windows"),
            mock.patch.object(pane, "scheduled_task_status", return_value={"supported": True, "installed": False}),
        ):
            doc = pane.bootstrap(config, apply=False)

        self.assertTrue(doc["ok"])
        self.assertFalse((self.root / "tools" / "_registry" / "control_pane.local.json").exists())
        actions = {a["id"]: a for a in doc["actions"]}
        self.assertIn("would write", actions["local-config"]["detail"])
        self.assertIn("would create runtime directories", actions["runtime-dirs"]["detail"])
        self.assertIn("-Python", actions["control-tick"]["command"])
        self.assertIn(str(config["python"]), actions["control-tick"]["command"])

    def test_bootstrap_dry_run_includes_local_config_overrides(self) -> None:
        script = self.root / "tools" / "register_control_pane_tick.ps1"
        script.write_text("", encoding="utf-8")
        overrides = {"machine_dir": "../shared/machines", "machine_id": "dgx-1", "target": 8}
        config = pane.apply_runtime_overrides(
            {
                **pane.normalize_config(pane.default_config(self.root), self.root),
                "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
                "_local_exists": False,
            },
            self.root,
            overrides,
        )
        with (
            mock.patch.object(platform, "system", return_value="Windows"),
            mock.patch.object(pane, "scheduled_task_status", return_value={"supported": True, "installed": False}),
        ):
            doc = pane.bootstrap(config, apply=False, init_overrides=overrides)

        self.assertTrue(doc["ok"])
        actions = {a["id"]: a for a in doc["actions"]}
        command = actions["local-config"]["command"]
        self.assertIn("--machine-dir", command)
        self.assertIn("../shared/machines", command)
        self.assertIn("--machine-id dgx-1", command)
        self.assertIn("--target 8", command)

    def test_bootstrap_apply_writes_config_and_invokes_register(self) -> None:
        script = self.root / "tools" / "register_control_pane_tick.ps1"
        script.write_text("", encoding="utf-8")
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": False,
        }
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="installed\n", stderr="")
        with (
            mock.patch.object(platform, "system", return_value="Windows"),
            mock.patch.object(pane, "scheduled_task_status", return_value={"supported": True, "installed": False}),
            mock.patch.object(pane, "run", return_value=proc) as run_mock,
        ):
            doc = pane.bootstrap(config, apply=True)

        self.assertTrue(doc["ok"])
        self.assertTrue((self.root / "tools" / "_registry" / "control_pane.local.json").exists())
        self.assertTrue((self.root / "tools" / "_registry" / "machines").is_dir())
        self.assertTrue((self.root / "tools" / "_watchdog").is_dir())
        run_mock.assert_called_once()
        called_cmd = run_mock.call_args.args[0]
        self.assertIn("-Python", called_cmd)
        self.assertIn(str(config["python"]), called_cmd)

    def test_bootstrap_apply_seeds_local_config_from_tracked_defaults(self) -> None:
        script = self.root / "tools" / "register_control_pane_tick.ps1"
        script.write_text("", encoding="utf-8")
        (self.root / "tools" / "control_pane.example.json").write_text(
            json.dumps({
                "schema": pane.CONFIG_SCHEMA,
                "machine_dir": "../shared/machines",
                "registry_dir": "../shared/registry",
                "target": 8,
            }),
            encoding="utf-8",
        )
        config = pane.load_config(self.root)
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="installed\n", stderr="")
        with (
            mock.patch.object(platform, "system", return_value="Windows"),
            mock.patch.object(pane, "scheduled_task_status", return_value={"supported": True, "installed": False}),
            mock.patch.object(pane, "run", return_value=proc),
        ):
            doc = pane.bootstrap(config, apply=True)

        self.assertTrue(doc["ok"])
        local = self.root / "tools" / "_registry" / "control_pane.local.json"
        written = json.loads(local.read_text(encoding="utf-8"))
        self.assertEqual(written["machine_dir"], str((self.root / "../shared/machines").resolve()))
        self.assertEqual(written["registry_dir"], str((self.root / "../shared/registry").resolve()))
        self.assertEqual(written["target"], 8)
        reloaded = pane.load_config(self.root)
        self.assertEqual(reloaded["machine_dir"], str((self.root / "../shared/machines").resolve()))
        self.assertEqual(reloaded["target"], 8)

    def test_bootstrap_reports_unusable_runtime_dir(self) -> None:
        script = self.root / "tools" / "register_control_pane_tick.ps1"
        script.write_text("", encoding="utf-8")
        bad_machine_dir = self.root / "not-a-dir"
        bad_machine_dir.write_text("file\n", encoding="utf-8")
        config = {
            **pane.normalize_config({
                **pane.default_config(self.root),
                "machine_dir": str(bad_machine_dir),
            }, self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": False,
        }
        with (
            mock.patch.object(platform, "system", return_value="Windows"),
            mock.patch.object(pane, "scheduled_task_status", return_value={"supported": True, "installed": True}),
        ):
            doc = pane.bootstrap(config, apply=False)

        self.assertFalse(doc["ok"])
        runtime = {a["id"]: a for a in doc["actions"]}["runtime-dirs"]
        self.assertIn("not writable", runtime["detail"])
        self.assertTrue(any(check["name"] == "machine_dir" and check["reason"] == "not a directory" for check in runtime["checks"]))

    def test_doctor_reports_unusable_machine_dir(self) -> None:
        for rel in (
            "tools/fleet_sessions.py",
            "tools/fleet_accounts.py",
            "tools/safe_ff_sync.py",
            "tools/fleet_control_pane.py",
            "tools/register_control_pane_tick.ps1",
            "tools/register_control_pane_tick.sh",
        ):
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        bad_machine_dir = self.root / "not-a-dir"
        bad_machine_dir.write_text("file\n", encoding="utf-8")
        config = {
            **pane.normalize_config({
                **pane.default_config(self.root),
                "machine_dir": str(bad_machine_dir),
                "watchdogs": {},
            }, self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": False,
        }

        with mock.patch.object(pane, "collect_control_tick", return_value={
            "register_script_exists": True,
            "register_script": "tools/register_control_pane_tick.ps1",
            "task": {"supported": False},
        }):
            doc = pane.doctor(config)

        self.assertFalse(doc["ok"])
        checks = {check["name"]: check for check in doc["checks"]}
        self.assertFalse(checks["machine_dir-writable"]["ok"])
        self.assertIn("not a directory", checks["machine_dir-writable"]["detail"])

    def test_doctor_reports_local_config_default_drift(self) -> None:
        (self.root / "tools" / "control_pane.example.json").write_text(
            json.dumps({
                "schema": pane.CONFIG_SCHEMA,
                "machine_dir": "../shared/machines",
            }),
            encoding="utf-8",
        )
        local = self.root / "tools" / "_registry" / "control_pane.local.json"
        local.parent.mkdir(parents=True)
        local.write_text(
            json.dumps({
                "schema": pane.CONFIG_SCHEMA,
                "machine_dir": "tools/_registry/machines",
            }),
            encoding="utf-8",
        )
        config = pane.load_config(self.root)

        with mock.patch.object(pane, "collect_control_tick", return_value={
            "register_script_exists": True,
            "register_script": "tools/register_control_pane_tick.ps1",
            "task": {"supported": False},
        }):
            doc = pane.doctor(config)

        checks = {check["name"]: check for check in doc["checks"]}
        self.assertFalse(doc["ok"])
        self.assertFalse(checks["local-config-tracked-defaults"]["ok"])
        self.assertIn("machine_dir local=", checks["local-config-tracked-defaults"]["detail"])

    def test_doctor_accepts_missing_standalone_watchdog_when_tick_covers_it(self) -> None:
        for rel in (
            "tools/fleet_sessions.py",
            "tools/fleet_accounts.py",
            "tools/safe_ff_sync.py",
            "tools/fleet_control_pane.py",
            "tools/register_control_pane_tick.ps1",
            "tools/register_control_pane_tick.sh",
        ):
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        for rel in ("tools/_registry/machines", "tools/_watchdog"):
            (self.root / rel).mkdir(parents=True)
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
        }

        with (
            mock.patch.object(pane, "collect_control_tick", return_value={
                "task_name": "FleetControlPaneTick",
                "register_script_exists": True,
                "register_script": "tools/register_control_pane_tick.ps1",
                "runner": "tools/_registry/control_pane_tick.cmd",
                "runner_exists": True,
                "task": {"supported": True, "installed": True, "task_name": "FleetControlPaneTick"},
            }),
            mock.patch.object(pane, "collect_watchdogs", return_value={
                "resume": {
                    "script_exists": True,
                    "script": "tools/fleet_resume_watchdog.ps1",
                    "task": {"supported": True, "installed": False, "task_name": "FleetResumeWatchdog"},
                },
            }),
        ):
            doc = pane.doctor(config)

        checks = {check["name"]: check for check in doc["checks"]}
        self.assertTrue(doc["ok"])
        self.assertTrue(checks["resume-scheduled-task"]["ok"])
        self.assertIn("covered by control-pane tick", checks["resume-scheduled-task"]["detail"])
        self.assertIn("standalone not installed", checks["resume-scheduled-task"]["detail"])

    def test_bootstrap_refuses_to_ignore_existing_mismatched_local_config(self) -> None:
        script = self.root / "tools" / "register_control_pane_tick.ps1"
        script.write_text("", encoding="utf-8")
        local = self.root / "tools" / "_registry" / "control_pane.local.json"
        local.parent.mkdir(parents=True)
        local.write_text(json.dumps({"schema": pane.CONFIG_SCHEMA, "machine_dir": "old"}), encoding="utf-8")
        overrides = {"machine_dir": "../shared/machines"}
        config = pane.apply_runtime_overrides(
            {
                **pane.normalize_config(pane.default_config(self.root), self.root),
                "_paths": {"local": str(local)},
                "_local_exists": True,
            },
            self.root,
            overrides,
        )
        with (
            mock.patch.object(platform, "system", return_value="Windows"),
            mock.patch.object(pane, "scheduled_task_status", return_value={"supported": True, "installed": True}),
        ):
            doc = pane.bootstrap(config, apply=True, init_overrides=overrides)

        self.assertFalse(doc["ok"])
        local_action = {a["id"]: a for a in doc["actions"]}["local-config"]
        self.assertIn("machine_dir", local_action["mismatches"])
        self.assertIn("--force", local_action["command"])

    def test_bootstrap_force_rewrites_existing_local_config_with_overrides(self) -> None:
        script = self.root / "tools" / "register_control_pane_tick.ps1"
        script.write_text("", encoding="utf-8")
        local = self.root / "tools" / "_registry" / "control_pane.local.json"
        local.parent.mkdir(parents=True)
        local.write_text(json.dumps({"schema": pane.CONFIG_SCHEMA, "machine_dir": "old"}), encoding="utf-8")
        overrides = {"machine_dir": "../shared/machines"}
        config = pane.apply_runtime_overrides(
            {
                **pane.normalize_config(pane.default_config(self.root), self.root),
                "_paths": {"local": str(local)},
                "_local_exists": True,
            },
            self.root,
            overrides,
        )
        with (
            mock.patch.object(platform, "system", return_value="Windows"),
            mock.patch.object(pane, "scheduled_task_status", return_value={"supported": True, "installed": True}),
        ):
            doc = pane.bootstrap(
                config,
                apply=True,
                init_overrides=overrides,
                force_local_config=True,
            )

        self.assertTrue(doc["ok"])
        self.assertEqual(json.loads(local.read_text(encoding="utf-8"))["machine_dir"], "../shared/machines")

    def test_collect_control_tick_reads_posix_status_json(self) -> None:
        script = self.root / "tools" / "register_control_pane_tick.sh"
        script.write_text("", encoding="utf-8")
        runner = self.root / "tools" / "_registry" / "control_pane_tick.sh"
        runner.parent.mkdir(parents=True)
        runner.write_text("#!/bin/sh\n", encoding="utf-8")
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
        }
        proc = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"installed":true,"mode":"cron","state":"present"}\n',
            stderr="",
        )

        with (
            mock.patch.object(platform, "system", return_value="Linux"),
            mock.patch.object(pane, "sh_exe", return_value="/bin/sh"),
            mock.patch.object(pane, "run", return_value=proc) as run_mock,
        ):
            tick = pane.collect_control_tick(config)

        self.assertEqual(Path(tick["register_script"]), Path("tools/register_control_pane_tick.sh"))
        self.assertTrue(tick["task"]["supported"])
        self.assertTrue(tick["task"]["installed"])
        self.assertEqual(tick["task"]["mode"], "cron")
        self.assertEqual(Path(tick["runner"]), Path("tools") / "_registry" / "control_pane_tick.sh")
        self.assertTrue(tick["runner_exists"])
        called_cmd = run_mock.call_args.args[0]
        self.assertEqual(called_cmd[-2:], ["status", "--json"])

    def test_bootstrap_repairs_missing_installed_tick_runner(self) -> None:
        script = self.root / "tools" / "register_control_pane_tick.ps1"
        script.write_text("", encoding="utf-8")
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
        }
        tick = {
            "task_name": "FleetControlPaneTick",
            "register_script": str(Path("tools") / "register_control_pane_tick.ps1"),
            "register_script_exists": True,
            "runner": str(Path("tools") / "_registry" / "control_pane_tick.cmd"),
            "runner_exists": False,
            "task": {"supported": True, "installed": True},
        }
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="installed\n", stderr="")

        with (
            mock.patch.object(pane, "collect_control_tick", return_value=tick),
            mock.patch.object(pane, "run", return_value=proc) as run_mock,
        ):
            dry = pane.bootstrap(config, apply=False)
            applied = pane.bootstrap(config, apply=True)

        dry_action = {a["id"]: a for a in dry["actions"]}["control-tick"]
        self.assertIn("would repair missing control-pane tick runner", dry_action["detail"])
        applied_action = {a["id"]: a for a in applied["actions"]}["control-tick"]
        self.assertTrue(applied_action["changed"])
        self.assertIn("installed", applied_action["detail"])
        run_mock.assert_called_once()

    def test_bootstrap_repairs_disabled_installed_tick(self) -> None:
        script = self.root / "tools" / "register_control_pane_tick.ps1"
        script.write_text("", encoding="utf-8")
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
        }
        tick = {
            "task_name": "FleetControlPaneTick",
            "register_script": str(Path("tools") / "register_control_pane_tick.ps1"),
            "register_script_exists": True,
            "runner": str(Path("tools") / "_registry" / "control_pane_tick.cmd"),
            "runner_exists": True,
            "task": {"supported": True, "installed": True, "state": "Disabled", "last_result": 0},
        }
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="installed\n", stderr="")

        with (
            mock.patch.object(pane, "collect_control_tick", return_value=tick),
            mock.patch.object(pane, "run", return_value=proc) as run_mock,
        ):
            dry = pane.bootstrap(config, apply=False)
            applied = pane.bootstrap(config, apply=True)

        dry_action = {a["id"]: a for a in dry["actions"]}["control-tick"]
        self.assertIn("would re-enable control-pane tick task", dry_action["detail"])
        applied_action = {a["id"]: a for a in applied["actions"]}["control-tick"]
        self.assertTrue(applied_action["changed"])
        run_mock.assert_called_once()

    def test_bootstrap_posix_dry_run_prints_shell_installer(self) -> None:
        script = self.root / "tools" / "register_control_pane_tick.sh"
        script.write_text("", encoding="utf-8")
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": False,
        }

        with (
            mock.patch.object(platform, "system", return_value="Linux"),
            mock.patch.object(pane, "posix_tick_status", return_value={"supported": True, "installed": False}),
            mock.patch.object(pane, "sh_exe", return_value="/bin/sh"),
        ):
            doc = pane.bootstrap(config, apply=False)

        self.assertTrue(doc["ok"])
        actions = {a["id"]: a for a in doc["actions"]}
        self.assertIn("would install control-pane tick", actions["control-tick"]["detail"])
        command = actions["control-tick"]["command"]
        self.assertIn("register_control_pane_tick.sh", command)
        self.assertIn("install", command)
        self.assertIn("--python", command)

    def test_bootstrap_posix_apply_invokes_shell_installer(self) -> None:
        script = self.root / "tools" / "register_control_pane_tick.sh"
        script.write_text("", encoding="utf-8")
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": False,
        }
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="installed\n", stderr="")

        with (
            mock.patch.object(platform, "system", return_value="Linux"),
            mock.patch.object(pane, "posix_tick_status", return_value={"supported": True, "installed": False}),
            mock.patch.object(pane, "sh_exe", return_value="/bin/sh"),
            mock.patch.object(pane, "run", return_value=proc) as run_mock,
        ):
            doc = pane.bootstrap(config, apply=True)

        self.assertTrue(doc["ok"])
        called_cmd = run_mock.call_args.args[0]
        self.assertEqual(called_cmd[0], "/bin/sh")
        self.assertIn("register_control_pane_tick.sh", called_cmd[1])
        self.assertIn("install", called_cmd)
        self.assertIn("--python", called_cmd)
        self.assertIn(str(config["python"]), called_cmd)

    def test_setup_plan_posix_uses_shell_installer(self) -> None:
        script = self.root / "tools" / "register_control_pane_tick.sh"
        script.write_text("", encoding="utf-8")
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "watchdogs": {},
        }

        with (
            mock.patch.object(platform, "system", return_value="Linux"),
            mock.patch.object(pane, "posix_tick_status", return_value={"supported": True, "installed": False}),
            mock.patch.object(pane, "sh_exe", return_value="/bin/sh"),
        ):
            plan = pane.setup_plan(config)

        install = [step for step in plan["steps"] if step["id"] == "install-control-pane-tick"][0]
        self.assertEqual(install["command"][0], "/bin/sh")
        self.assertIn("register_control_pane_tick.sh", install["command"][1])
        self.assertIn("install", install["command"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
