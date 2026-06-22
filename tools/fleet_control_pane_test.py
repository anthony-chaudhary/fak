#!/usr/bin/env python3
"""Hermetic tests for fleet_control_pane.py."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_control_pane as pane  # noqa: E402
import fleet_version  # noqa: E402


class FleetControlPaneTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "tools").mkdir()
        (self.root / "VERSION").write_text("test\n", encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    def test_run_via_files_does_not_block_on_detached_grandchild(self) -> None:
        # Regression: a watchdog may relaunch a long-lived DETACHED daemon (the job
        # supervisor / a resumed claude) that inherits this process's stdout. With pipe
        # capture, run() blocks on EOF until that daemon also closes the pipe -- which
        # never happens, hanging the tick, and `timeout` cannot rescue it (the post-timeout
        # drain re-blocks on the same pipe). via_files must return as soon as the immediate
        # child exits, regardless of the still-living grandchild.
        import time

        child = (
            "import subprocess,sys;"
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(8)']);"
            "print('child-done')"
        )
        # cwd is the system temp (NOT self.root): the detached grandchild inherits the cwd
        # and holds it for 8s, which would otherwise block this test's tmpdir cleanup.
        cwd = Path(tempfile.gettempdir())
        t0 = time.monotonic()
        proc = pane.run([sys.executable, "-c", child], cwd, timeout=20, via_files=True)
        elapsed = time.monotonic() - t0
        self.assertEqual(proc.returncode, 0)
        self.assertIn("child-done", proc.stdout)
        self.assertLess(elapsed, 5.0, f"run() blocked on the detached grandchild ({elapsed:.1f}s)")

    def test_run_via_files_captures_stdout_and_stderr(self) -> None:
        prog = "import sys; print('out-line'); print('err-line', file=sys.stderr)"
        proc = pane.run([sys.executable, "-c", prog], self.root, timeout=20, via_files=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("out-line", proc.stdout)
        self.assertIn("err-line", proc.stderr)

    def test_write_text_atomic_retries_windows_replace_lock(self) -> None:
        target = self.root / "tools" / "_registry" / "control_pane.json"
        calls: list[Path] = []
        real_replace = Path.replace

        def flaky_replace(src: Path, dst: Path) -> Path:
            calls.append(src)
            if len(calls) == 1:
                raise PermissionError("target locked")
            return real_replace(src, dst)

        with (
            mock.patch.object(Path, "replace", new=flaky_replace),
            mock.patch.object(pane.time, "sleep") as sleep_mock,
        ):
            pane.write_text_atomic(target, "ok\n")

        self.assertEqual(target.read_text(encoding="utf-8"), "ok\n")
        self.assertEqual(sleep_mock.call_count, 1)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0].name, "control_pane.json.tmp")
        self.assertFalse(target.with_name(target.name + ".tmp").exists())
        self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_parse_git_status_porcelain_counts_dirty_shapes(self) -> None:
        raw = " M a.txt\0D  old.txt\0?? new.txt\0R  dst.txt\0src.txt\0"

        got = pane.parse_git_status_porcelain(raw)

        self.assertTrue(got["dirty"])
        self.assertEqual(got["dirty_total"], 4)
        self.assertEqual(got["counts"]["modified"], 1)
        self.assertEqual(got["counts"]["deleted"], 1)
        self.assertEqual(got["counts"]["untracked"], 1)
        self.assertEqual(got["counts"]["renamed"], 1)
        self.assertEqual(got["entries"][3]["old_path"], "src.txt")

    def test_collect_worktree_doctor_summarizes_prune_and_blocked_state(self) -> None:
        config = pane.normalize_config(pane.default_config(self.root), self.root)
        fake_doctor = mock.Mock()
        sigs = [
            {"path": str(self.root), "branch": "master"},
            {"path": str(self.root / "wt-old"), "branch": None},
        ]
        fake_doctor.collect.return_value = sigs
        fake_doctor.make_plan.return_value = {
            "converged": False,
            "needs_human": True,
            "primary_offtrack": False,
            "keeper": str(self.root),
            "prune": [{"path": str(self.root / "wt-old"), "branch": None}],
            "blocked": [{"path": str(self.root / "wt-dirty"), "branch": None, "reasons": ["uncommitted changes"]}],
            "retained": [],
        }

        with mock.patch.dict(sys.modules, {"worktree_doctor": fake_doctor}):
            got = pane.collect_worktree_doctor(config)

        self.assertTrue(got["available"])
        self.assertEqual(got["total"], 2)
        self.assertEqual(got["prune_count"], 1)
        self.assertEqual(got["blocked_count"], 1)
        fake_doctor.collect.assert_called_once_with(str(self.root), "origin/master", fetch=False)
        fake_doctor.make_plan.assert_called_once_with(
            sigs,
            "origin/master",
            allow_branches=["fak-v0.1"],
        )
        self.assertIn("--prune", got["commands"]["prune"])
        self.assertIn("--allow-branch fak-v0.1", got["commands"]["inspect"])

    def test_app_version_hides_conflicted_version_file(self) -> None:
        (self.root / "VERSION").write_text("<<<<<<< HEAD\n0.8.15\n=======\n0.10.0\n>>>>>>> master\n", encoding="utf-8")

        self.assertEqual(fleet_version.app_version(self.root), fleet_version.DEFAULT_VERSION)

    def test_dirty_commit_plan_blocks_unmerged_groups(self) -> None:
        plan = pane.dirty_commit_plan(
            [
                {"code": "UU", "path": "VERSION"},
                {"code": " M", "path": "tools/fleet_version.py"},
                {"code": " M", "path": "docs/readme.md"},
            ],
            {"python": "python"},
        )

        groups = {group["group"]: group for group in plan["groups"]}
        self.assertTrue(groups["tools/fleet-control-pane"]["blocked"])
        self.assertIn("tools/fleet_version.py", groups["tools/fleet-control-pane"]["paths"])
        self.assertNotIn("command", groups["tools/fleet-control-pane"])
        self.assertIn("resolve merge conflicts", groups["tools/fleet-control-pane"]["reason"])
        self.assertIn("command", groups["docs"])

    def test_dirty_commit_plan_blocks_all_groups_during_merge(self) -> None:
        plan = pane.dirty_commit_plan(
            [
                {"code": " M", "path": "VERSION"},
                {"code": " M", "path": "docs/readme.md"},
            ],
            {"python": "python"},
            merge_in_progress=True,
        )

        for group in plan["groups"]:
            self.assertTrue(group["blocked"])
            self.assertNotIn("command", group)
            self.assertIn("finish or abort the merge", group["reason"])

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

    def test_windows_tick_register_script_forces_enable_after_create(self) -> None:
        script = (Path(__file__).resolve().parent / "register_control_pane_tick.ps1").read_text(encoding="utf-8")

        create_idx = script.index("schtasks /Create")
        enable_idx = script.index("schtasks /Change /TN $TaskName /ENABLE")
        run_idx = script.index("schtasks /Run")

        self.assertLess(create_idx, enable_idx)
        self.assertLess(enable_idx, run_idx)
        self.assertIn("schtasks /Change /ENABLE failed", script)

    def test_disabled_scheduled_task_needs_action(self) -> None:
        task = {"supported": True, "installed": True, "state": "Disabled", "last_result": 0}

        self.assertTrue(pane.scheduler_state_needs_action(task))
        self.assertTrue(pane.scheduled_task_needs_action(task))
        self.assertTrue(pane.scheduled_task_snapshot(task)["needs_action"])
        self.assertEqual(pane.scheduled_task_label(task), "Disabled/disabled")

    def test_summarize_registry_rolls_up_sessions_and_accounts(self) -> None:
        reg = {
            "schema": "fleet-sessions/3",
            "generated_utc": "2026-06-18T12:00:00+00:00",
            "accounts": [
                {"tag": "gem8", "available": True},
                {
                    "account": ".claude-c10",
                    "tag": "c10",
                    "config_dir": "C:/Users/USER/.claude-c10",
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
        self.assertEqual(got["accounts"]["blocked"][0]["config_dir"], "C:/Users/USER/.claude-c10")
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
                    "config_dir": "C:/Users/USER/.claude-gem8",
                    "session": "abcdef1234567890",
                    "cwd": "C:/work/fleet",
                    "project": "fleet",
                    "resume_cmd": "claude --resume abcdef1234567890",
                },
                {
                    "account": ".claude-c10",
                    "config_dir": "C:/Users/USER/.claude-c10",
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

    def test_dirty_commit_plan_groups_paths_by_top_level(self) -> None:
        config = {"python": "python.exe"}
        entries = [
            {"path": "tools/fleet_control_pane.py"},
            {"path": "tools/fleet_control_pane_test.py"},
            {"path": "tools/_dashboard_proof.png"},
            {"path": "fak/internal/compute/vulkan.go"},
            {"path": "fak/internal/model/batch.go"},
            {"path": "docs/plan.md"},
            {"path": "tools/loops/nightly_status.py"},
            {"path": "README.md"},
        ]

        got = pane.dirty_commit_plan(entries, config)

        groups = {group["group"]: group for group in got["groups"]}
        self.assertEqual(got["count"], 7)
        self.assertEqual(groups["tools/fleet-control-pane"]["count"], 2)
        self.assertEqual(groups["tools/proofs"]["paths"], ["tools/_dashboard_proof.png"])
        self.assertEqual(groups["tools/loops"]["paths"], ["tools/loops/nightly_status.py"])
        self.assertEqual(groups["fak/internal/compute"]["paths"], ["fak/internal/compute/vulkan.go"])
        self.assertEqual(groups["fak/internal/model"]["paths"], ["fak/internal/model/batch.go"])
        self.assertEqual(groups["docs"]["paths"], ["docs/plan.md"])
        self.assertEqual(groups["root"]["paths"], ["README.md"])
        self.assertIn("--dirty-group tools/fleet-control-pane", groups["tools/fleet-control-pane"]["command"])
        self.assertNotIn("--path tools/fleet_control_pane.py", groups["tools/fleet-control-pane"]["command"])
        self.assertEqual(groups["tools/fleet-control-pane"]["suggested_subject"], "tools: update fleet control pane")
        self.assertIn("tools: update fleet control pane", groups["tools/fleet-control-pane"]["command"])
        self.assertEqual(groups["tools/loops"]["suggested_subject"], "tools: update loop scripts")
        self.assertIn("tools: update loop scripts", groups["tools/loops"]["command"])
        self.assertIn('TODO: describe fak/internal/compute changes', groups["fak/internal/compute"]["command"])

    def test_dirty_group_selection_resolves_current_dirty_plan(self) -> None:
        config = {"root": str(self.root)}
        git_status = {
            "dirty_plan": {
                "groups": [
                    {"group": "tools/fleet-control-pane", "paths": ["tools/fleet_control_pane.py"]},
                    {"group": "docs", "paths": ["docs/plan.md"]},
                ],
            },
        }

        with mock.patch.object(pane, "collect_git", return_value=git_status):
            got = pane.dirty_group_selection(config, ["tools/fleet-control-pane", "missing"])

        self.assertFalse(got["ok"])
        self.assertEqual(got["paths"], ["tools/fleet_control_pane.py"])
        self.assertEqual(got["missing"], ["missing"])
        self.assertEqual(got["available"], ["docs", "tools/fleet-control-pane"])

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

    def test_init_config_writes_ignored_local_config(self) -> None:
        with mock.patch.object(sys, "executable", "/py"):
            result = pane.init_config(self.root)

        self.assertTrue(result["written"])
        path = Path(result["path"])
        self.assertEqual(path, self.root / "tools" / "_registry" / "control_pane.local.json")
        doc = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(doc["schema"], pane.CONFIG_SCHEMA)
        self.assertIn("registry_dir", doc)

    def test_init_config_writes_machine_overrides(self) -> None:
        result = pane.init_config(
            self.root,
            overrides={
                "job_dir": "../job2",
                "machine_dir": "../shared/machines",
                "machine_id": "dgx-1",
                "target": 8,
                "session_window_h": 12,
            },
        )

        self.assertTrue(result["written"])
        doc = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
        self.assertEqual(doc["job_dir"], "../job2")
        self.assertEqual(doc["machine_dir"], "../shared/machines")
        self.assertEqual(doc["machine_id"], "dgx-1")
        self.assertEqual(doc["target"], 8)
        self.assertEqual(doc["session_window_h"], 12.0)

    def test_init_config_uses_effective_base_config_defaults(self) -> None:
        base = pane.normalize_config(
            {
                **pane.default_config(self.root),
                "machine_dir": "../shared/machines",
                "registry_dir": "../shared/registry",
                "machine_id": "dgx-1",
                "target": 8,
            },
            self.root,
        )

        result = pane.init_config(self.root, base_config=base)

        doc = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
        self.assertEqual(doc["machine_dir"], str((self.root / "../shared/machines").resolve()))
        self.assertEqual(doc["registry_dir"], str((self.root / "../shared/registry").resolve()))
        self.assertEqual(doc["machine_id"], "dgx-1")
        self.assertEqual(doc["target"], 8)

    def test_directory_probe_uses_unique_probe_file(self) -> None:
        target = self.root / "tools" / "_registry"
        target.mkdir(parents=True)
        (target / ".control-pane-write-test").mkdir()

        got = pane.directory_probe(target, create=False)

        self.assertTrue(got["ok"])
        self.assertTrue((target / ".control-pane-write-test").is_dir())

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

    def test_loop_config_plan_dry_run_does_not_write_local_config(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": False,
        }

        doc = pane.loop_config_plan(
            config,
            loop_name="build-loop",
            status_cmd=[sys.executable, "tools/build_status.py", "--json"],
            recover_cmd=[sys.executable, "tools/build_watchdog.py"],
            auto_recover=True,
            action="restart build loop",
            timeout_s=15,
            apply=False,
        )

        self.assertTrue(doc["ok"])
        self.assertFalse(doc["apply"])
        self.assertFalse((self.root / "tools" / "_registry" / "control_pane.local.json").exists())
        self.assertEqual(doc["spec"]["status_cmd"], [sys.executable, "tools/build_status.py", "--json"])
        self.assertIn("--apply", doc["command"])

    def test_loop_config_plan_apply_preserves_existing_local_config(self) -> None:
        local = self.root / "tools" / "_registry" / "control_pane.local.json"
        local.parent.mkdir(parents=True)
        local.write_text(json.dumps({"schema": pane.CONFIG_SCHEMA, "target": 8}), encoding="utf-8")
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(local)},
            "_local_exists": True,
        }

        doc = pane.loop_config_plan(
            config,
            loop_name="build-loop",
            status_cmd=[sys.executable, "tools/build_status.py", "--json"],
            recover_cmd=[sys.executable, "tools/build_watchdog.py"],
            auto_recover=True,
            action="restart build loop",
            timeout_s=15,
            recover_timeout_s=60,
            cwd="tools",
            ok_returncodes=[0, 2],
            apply=True,
        )

        written = json.loads(local.read_text(encoding="utf-8"))
        self.assertTrue(doc["ok"])
        self.assertTrue(doc["apply"])
        self.assertEqual(written["target"], 8)
        self.assertEqual(written["loops"]["build-loop"]["status_cmd"], [sys.executable, "tools/build_status.py", "--json"])
        self.assertEqual(written["loops"]["build-loop"]["recover_cmd"], [sys.executable, "tools/build_watchdog.py"])
        self.assertTrue(written["loops"]["build-loop"]["auto_recover"])
        self.assertEqual(written["loops"]["build-loop"]["recover_timeout_s"], 60)
        self.assertEqual(written["loops"]["build-loop"]["ok_returncodes"], [0, 2])

    def test_loop_config_plan_repo_scope_writes_tracked_catalog_not_local_config(self) -> None:
        local = self.root / "tools" / "_registry" / "control_pane.local.json"
        catalog = self.root / "tools" / "control_pane.loops.json"
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(local), "loop_catalog": str(catalog)},
            "_local_exists": False,
        }

        doc = pane.loop_config_plan(
            config,
            loop_name="shared-loop",
            scope="repo",
            status_cmd=[sys.executable, "tools/shared_status.py", "--json"],
            recover_cmd=[sys.executable, "tools/shared_watchdog.py"],
            auto_recover=True,
            apply=True,
        )

        self.assertTrue(doc["ok"])
        self.assertEqual(doc["scope"], "repo")
        self.assertEqual(doc["config_path"], str(catalog))
        self.assertIsNone(doc["local_config"])
        self.assertFalse(local.exists())
        written = json.loads(catalog.read_text(encoding="utf-8"))
        self.assertEqual(written["schema"], pane.LOOP_CATALOG_SCHEMA)
        self.assertEqual(written["loops"]["shared-loop"]["status_cmd"], [sys.executable, "tools/shared_status.py", "--json"])
        self.assertIn("--scope repo", doc["command"])
        self.assertEqual(len(doc["followup_commands"]), 1)
        self.assertIn("commit --dirty-group tools/fleet-control-pane", doc["followup_commands"][0])
        self.assertIn("tools: add shared-loop loop", doc["followup_commands"][0])

    def test_loop_config_plan_matching_repo_loop_has_no_commit_followup(self) -> None:
        catalog = self.root / "tools" / "control_pane.loops.json"
        catalog.write_text(json.dumps({
            "schema": pane.LOOP_CATALOG_SCHEMA,
            "loops": {
                "shared-loop": {
                    "enabled": True,
                    "status_cmd": [sys.executable, "tools/shared_status.py", "--json"],
                    "auto_recover": False,
                },
            },
        }), encoding="utf-8")
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"loop_catalog": str(catalog)},
            "_loop_catalog_exists": True,
        }

        doc = pane.loop_config_plan(
            config,
            loop_name="shared-loop",
            scope="repo",
            status_cmd=[sys.executable, "tools/shared_status.py", "--json"],
            apply=True,
        )

        self.assertTrue(doc["ok"])
        self.assertFalse(doc["changed"])
        self.assertEqual(doc["followup_commands"], [])

    def test_loop_list_reports_sources_and_readiness(self) -> None:
        status_script = self.root / "tools" / "shared_status.py"
        status_script.write_text("print('{}')\n", encoding="utf-8")
        (self.root / "tools" / "control_pane.example.json").write_text(json.dumps({
            "schema": pane.CONFIG_SCHEMA,
            "loops": {
                "example-loop": {
                    "enabled": False,
                    "status_cmd": [sys.executable, "tools/example_status.py"],
                },
            },
        }), encoding="utf-8")
        (self.root / "tools" / "control_pane.loops.json").write_text(json.dumps({
            "schema": pane.LOOP_CATALOG_SCHEMA,
            "loops": {
                "shared-loop": {
                    "enabled": True,
                    "status_cmd": [sys.executable, "tools/shared_status.py"],
                    "auto_recover": True,
                    "recover_cmd": [sys.executable, "tools/missing_recover.py"],
                },
            },
        }), encoding="utf-8")
        local = self.root / "tools" / "_registry" / "control_pane.local.json"
        local.parent.mkdir(parents=True)
        local.write_text(json.dumps({
            "schema": pane.CONFIG_SCHEMA,
            "loops": {
                "shared-loop": {
                    "recover_cmd": [],
                    "auto_recover": False,
                },
            },
        }), encoding="utf-8")
        config = pane.load_config(self.root)

        doc = pane.loop_list(config)

        self.assertTrue(doc["ok"])
        self.assertEqual(doc["count"], 2)
        shared = [loop for loop in doc["loops"] if loop["name"] == "shared-loop"][0]
        self.assertEqual(shared["source"], "local")
        self.assertTrue(shared["overridden"])
        self.assertEqual([entry["source"] for entry in shared["sources"]], ["repo", "local"])
        self.assertTrue(shared["status_ready"]["ok"])
        self.assertTrue(shared["ready"])
        example = [loop for loop in doc["loops"] if loop["name"] == "example-loop"][0]
        self.assertFalse(example["enabled"])
        self.assertEqual(example["source"], "example")

    def test_loop_list_empty_includes_repo_add_command(self) -> None:
        config = pane.load_config(self.root)

        doc = pane.loop_list(config)

        self.assertTrue(doc["ok"])
        self.assertEqual(doc["count"], 0)
        self.assertEqual(len(doc["commands"]), 2)
        self.assertIn("loop-scaffold NAME", doc["commands"][0])
        self.assertIn("loop-add NAME --scope repo", doc["commands"][1])

    def test_loop_check_plan_runs_one_loop_and_plans_recovery(self) -> None:
        bad_script = self.root / "tools" / "bad_loop.py"
        recover_script = self.root / "tools" / "recover_loop.py"
        bad_script.write_text("print('{\"verdict\": \"STALLED\", \"detail\": \"heartbeat stale\"}')\n", encoding="utf-8")
        recover_script.write_text("print('recovering')\n", encoding="utf-8")
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "loops": {
                "stalled": {
                    "status_cmd": [sys.executable, str(bad_script)],
                    "recover_cmd": [sys.executable, str(recover_script)],
                    "auto_recover": False,
                },
            },
        }

        doc = pane.loop_check_plan(config, "stalled", recover=True, apply=False)

        self.assertTrue(doc["ok"])
        self.assertEqual(doc["verdict"], "ACTION")
        self.assertTrue(doc["needs_action"])
        self.assertEqual(doc["check"]["detail"], "verdict=STALLED")
        self.assertEqual(doc["recovery"]["reason"], "dry run")
        self.assertTrue(doc["recovery"]["skipped"])

    def test_loop_check_plan_apply_runs_manual_recovery(self) -> None:
        bad_script = self.root / "tools" / "bad_loop.py"
        recover_script = self.root / "tools" / "recover_loop.py"
        marker = self.root / "recovered.txt"
        bad_script.write_text("print('{\"ok\": false, \"reason\": \"needs restart\"}')\n", encoding="utf-8")
        recover_script.write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('done', encoding='utf-8')\n",
            encoding="utf-8",
        )
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "loops": {
                "stalled": {
                    "status_cmd": [sys.executable, str(bad_script)],
                    "recover_cmd": [sys.executable, str(recover_script)],
                },
            },
        }

        doc = pane.loop_check_plan(config, "stalled", recover=True, apply=True)

        self.assertTrue(doc["ok"])
        self.assertEqual(doc["verdict"], "ACTION")
        self.assertFalse(doc["recovery"]["skipped"])
        self.assertEqual(doc["recovery"]["returncode"], 0)
        self.assertEqual(marker.read_text(encoding="utf-8"), "done")

    def test_loop_check_plan_reports_unknown_loop(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "loops": {"known": {"status_cmd": [sys.executable, "-c", "print('{}')"]}},
        }

        doc = pane.loop_check_plan(config, "missing")

        self.assertFalse(doc["ok"])
        self.assertEqual(doc["reason"], "unknown loop")
        self.assertEqual(doc["available"], ["known"])

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

    def _audit_config(self, loops: dict) -> dict:
        return {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "loops": loops,
        }

    def test_loop_audit_action_is_not_broken(self) -> None:
        # The core invariant: a loop that ran fine but surfaces a condition
        # (ok:false in its payload -> verdict ACTION) is the loop WORKING, so it
        # buckets as `action`, NOT `broken`, and does not fail the audit.
        surfacing = self.root / "tools" / "surfacing.py"
        surfacing.write_text("print('{\"ok\": false, \"reason\": \"closure_rate low\"}')\n", encoding="utf-8")
        healthy = self.root / "tools" / "healthy.py"
        healthy.write_text("print('{\"ok\": true}')\n", encoding="utf-8")
        config = self._audit_config({
            "surfacer": {"status_cmd": [sys.executable, str(surfacing)]},
            "happy": {"status_cmd": [sys.executable, str(healthy)]},
        })

        doc = pane.loop_audit(config)

        self.assertEqual(doc["schema"], pane.LOOP_AUDIT_SCHEMA)
        self.assertTrue(doc["ok"])  # no broken loop
        self.assertEqual(doc["counts"], {"healthy": 1, "action": 1, "broken": 0, "total": 2})
        by_name = {loop["name"]: loop for loop in doc["loops"]}
        self.assertEqual(by_name["surfacer"]["bucket"], "action")
        self.assertEqual(by_name["happy"]["bucket"], "healthy")
        # detail is a single clean line lifted from the payload, not a blob.
        self.assertEqual(by_name["surfacer"]["detail"], "closure_rate low")
        self.assertNotIn("\n", by_name["surfacer"]["detail"])

    def test_loop_audit_broken_loop_fails_the_audit(self) -> None:
        # A loop whose status command cannot run at all (state UNAVAILABLE) is the
        # only real failure: it buckets `broken` and flips top-level ok to false.
        healthy = self.root / "tools" / "healthy.py"
        healthy.write_text("print('{\"ok\": true}')\n", encoding="utf-8")
        config = self._audit_config({
            "happy": {"status_cmd": [sys.executable, str(healthy)]},
            "missing-cmd": {"status_cmd": ["this-binary-does-not-exist-xyz", "--json"]},
        })

        doc = pane.loop_audit(config)

        self.assertFalse(doc["ok"])
        self.assertEqual(doc["counts"]["broken"], 1)
        by_name = {loop["name"]: loop for loop in doc["loops"]}
        self.assertEqual(by_name["missing-cmd"]["bucket"], "broken")
        self.assertEqual(by_name["missing-cmd"]["state"], "UNAVAILABLE")

    def test_loop_audit_names_subsets_and_reports_missing(self) -> None:
        healthy = self.root / "tools" / "healthy.py"
        healthy.write_text("print('{\"ok\": true}')\n", encoding="utf-8")
        config = self._audit_config({
            "a": {"status_cmd": [sys.executable, str(healthy)]},
            "b": {"status_cmd": [sys.executable, str(healthy)]},
        })

        doc = pane.loop_audit(config, names=["a", "ghost"])

        self.assertEqual([loop["name"] for loop in doc["loops"]], ["a"])
        self.assertEqual(doc["counts"]["total"], 1)
        self.assertEqual(doc["missing"], ["ghost"])

    def test_loop_audit_skips_disabled_and_its_own_name(self) -> None:
        healthy = self.root / "tools" / "healthy.py"
        healthy.write_text("print('{\"ok\": true}')\n", encoding="utf-8")
        config = self._audit_config({
            "a": {"status_cmd": [sys.executable, str(healthy)]},
            "off": {"status_cmd": [sys.executable, str(healthy)], "enabled": False},
            pane.LOOP_AUDIT_SELF_NAME: {"status_cmd": [sys.executable, str(healthy)]},
        })

        doc = pane.loop_audit(config)

        names = {loop["name"] for loop in doc["loops"]}
        self.assertEqual(names, {"a"})  # disabled + self both excluded

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

    def test_loop_scaffold_plan_dry_run_prints_apply_command_without_writing(self) -> None:
        config = pane.load_config(self.root)

        doc = pane.loop_scaffold_plan(config, "nightly-build")

        self.assertTrue(doc["ok"])
        self.assertFalse(doc["apply"])
        self.assertFalse((self.root / "tools" / "loops" / "nightly-build_status.py").exists())
        self.assertFalse((self.root / "tools" / "control_pane.loops.json").exists())
        self.assertIn("loop-scaffold nightly-build --apply", doc["commands"][0])
        self.assertFalse(doc["spec"]["enabled"])
        self.assertEqual(doc["spec"]["status_cmd"], ["{python}", "tools/loops/nightly-build_status.py", "--json"])

    def test_loop_scaffold_plan_apply_writes_scripts_and_repo_catalog(self) -> None:
        config = pane.load_config(self.root)

        doc = pane.loop_scaffold_plan(config, "nightly-build", apply=True)

        status_path = self.root / "tools" / "loops" / "nightly-build_status.py"
        recover_path = self.root / "tools" / "loops" / "nightly-build_recover.py"
        catalog_path = self.root / "tools" / "control_pane.loops.json"
        self.assertTrue(doc["ok"])
        self.assertTrue(status_path.exists())
        self.assertTrue(recover_path.exists())
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        loop = catalog["loops"]["nightly-build"]
        self.assertFalse(loop["enabled"])
        self.assertEqual(loop["status_cmd"], ["{python}", "tools/loops/nightly-build_status.py", "--json"])
        self.assertEqual(loop["recover_cmd"], ["{python}", "tools/loops/nightly-build_recover.py"])
        self.assertEqual(len(doc["followup_commands"]), 2)
        self.assertIn("loop-check nightly-build --recover", doc["followup_commands"][0])
        self.assertIn("--path tools/loops/nightly-build_status.py", doc["followup_commands"][1])
        self.assertIn("--path tools/control_pane.loops.json", doc["followup_commands"][1])

    def test_loop_scaffold_plan_refuses_existing_files_without_force(self) -> None:
        status_path = self.root / "tools" / "loops" / "nightly-build_status.py"
        status_path.parent.mkdir(parents=True)
        status_path.write_text("custom\n", encoding="utf-8")
        config = pane.load_config(self.root)

        doc = pane.loop_scaffold_plan(config, "nightly-build", apply=True)

        self.assertFalse(doc["ok"])
        self.assertIn("refusing to overwrite", doc["reason"])
        self.assertEqual(doc["commands"], [])
        self.assertEqual(status_path.read_text(encoding="utf-8"), "custom\n")

    def test_loop_set_plan_enables_repo_loop_without_retyping_commands(self) -> None:
        catalog = self.root / "tools" / "control_pane.loops.json"
        catalog.write_text(json.dumps({
            "schema": pane.LOOP_CATALOG_SCHEMA,
            "loops": {
                "nightly-build": {
                    "enabled": False,
                    "status_cmd": ["{python}", "tools/loops/nightly-build_status.py", "--json"],
                    "recover_cmd": ["{python}", "tools/loops/nightly-build_recover.py"],
                    "auto_recover": False,
                },
            },
        }), encoding="utf-8")
        config = pane.load_config(self.root)

        doc = pane.loop_set_plan(config, "nightly-build", enabled=True, auto_recover=True, apply=True)

        written = json.loads(catalog.read_text(encoding="utf-8"))
        loop = written["loops"]["nightly-build"]
        self.assertTrue(doc["ok"])
        self.assertTrue(doc["changed"])
        self.assertTrue(loop["enabled"])
        self.assertTrue(loop["auto_recover"])
        self.assertEqual(loop["status_cmd"], ["{python}", "tools/loops/nightly-build_status.py", "--json"])
        self.assertEqual(len(doc["followup_commands"]), 2)
        self.assertIn("loop-check nightly-build --recover", doc["followup_commands"][0])
        self.assertIn("--path tools/control_pane.loops.json", doc["followup_commands"][1])

    def test_loop_set_plan_refuses_unknown_loop(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "loops": {"known": {"status_cmd": ["known"]}},
        }

        doc = pane.loop_set_plan(config, "missing", enabled=True)

        self.assertFalse(doc["ok"])
        self.assertEqual(doc["reason"], "loop is not known in this scope")
        self.assertEqual(doc["available"], ["known"])

    def test_loop_set_plan_refuses_without_setting(self) -> None:
        catalog = self.root / "tools" / "control_pane.loops.json"
        catalog.write_text(json.dumps({
            "schema": pane.LOOP_CATALOG_SCHEMA,
            "loops": {"known": {"status_cmd": ["known"]}},
        }), encoding="utf-8")
        config = pane.load_config(self.root)

        doc = pane.loop_set_plan(config, "known")

        self.assertFalse(doc["ok"])
        self.assertEqual(doc["reason"], "select at least one loop setting")
        self.assertEqual(doc["commands"], [])

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

    def test_load_config_merges_example_local_and_env(self) -> None:
        example = {
            "schema": pane.CONFIG_SCHEMA,
            "target": 2,
            "job_dir": "../example-job",
            "registry_dir": "tools/_registry",
            "watchdogs": {"supervisor": {"task_name": "ExampleTask"}},
        }
        local = {
            "target": 7,
            "watchdogs": {"supervisor": {"interval_min": 3}},
        }
        (self.root / "tools" / "control_pane.example.json").write_text(
            json.dumps(example), encoding="utf-8")
        (self.root / "tools" / "_registry").mkdir()
        (self.root / "tools" / "_registry" / "control_pane.local.json").write_text(
            json.dumps(local), encoding="utf-8")

        with mock.patch.dict(os.environ, {"FLEET_JOB_DIR": str(self.root / "env-job")}):
            config = pane.load_config(self.root)

        self.assertEqual(config["target"], 7)
        self.assertEqual(config["job_dir"], str((self.root / "env-job").resolve()))
        self.assertEqual(config["watchdogs"]["supervisor"]["task_name"], "ExampleTask")
        self.assertEqual(config["watchdogs"]["supervisor"]["interval_min"], 3)
        self.assertTrue(config["_local_exists"])

    def test_load_config_merges_repo_loop_catalog_before_local_overrides(self) -> None:
        (self.root / "tools" / "control_pane.example.json").write_text(
            json.dumps({
                "schema": pane.CONFIG_SCHEMA,
                "loops": {
                    "example-loop": {
                        "enabled": False,
                        "status_cmd": ["example"],
                    },
                },
            }),
            encoding="utf-8",
        )
        (self.root / "tools" / "control_pane.loops.json").write_text(
            json.dumps({
                "schema": pane.LOOP_CATALOG_SCHEMA,
                "loops": {
                    "shared-loop": {
                        "enabled": True,
                        "status_cmd": ["catalog"],
                        "auto_recover": True,
                    },
                },
            }),
            encoding="utf-8",
        )
        (self.root / "tools" / "_registry").mkdir()
        (self.root / "tools" / "_registry" / "control_pane.local.json").write_text(
            json.dumps({
                "schema": pane.CONFIG_SCHEMA,
                "loops": {
                    "shared-loop": {
                        "enabled": False,
                        "status_cmd": ["local"],
                    },
                },
            }),
            encoding="utf-8",
        )

        config = pane.load_config(self.root)

        self.assertEqual(config["loops"]["example-loop"]["status_cmd"], ["example"])
        self.assertFalse(config["loops"]["shared-loop"]["enabled"])
        self.assertEqual(config["loops"]["shared-loop"]["status_cmd"], ["local"])
        self.assertTrue(config["loops"]["shared-loop"]["auto_recover"])
        self.assertEqual(config["_paths"]["loop_catalog"], str(self.root / "tools" / "control_pane.loops.json"))

    def test_local_config_default_drift_reports_shadowed_tracked_defaults(self) -> None:
        (self.root / "tools" / "control_pane.example.json").write_text(
            json.dumps({
                "schema": pane.CONFIG_SCHEMA,
                "machine_dir": "../shared/machines",
                "registry_dir": "../shared/registry",
                "target": 8,
            }),
            encoding="utf-8",
        )
        local = self.root / "tools" / "_registry" / "control_pane.local.json"
        local.parent.mkdir(parents=True)
        local.write_text(
            json.dumps({
                "schema": pane.CONFIG_SCHEMA,
                "machine_dir": "tools/_registry/machines",
                "registry_dir": "tools/_registry",
                "target": 4,
            }),
            encoding="utf-8",
        )
        config = pane.load_config(self.root)

        drift = pane.local_config_default_drift(config)

        by_key = {item["key"]: item for item in drift["items"]}
        self.assertEqual(drift["count"], 3)
        self.assertEqual(by_key["target"]["current"], 4)
        self.assertEqual(by_key["target"]["tracked"], 8)
        self.assertIn("--force", drift["command"])
        self.assertIn("--machine-dir", drift["command"])
        self.assertIn(str((self.root / "../shared/machines").resolve()), drift["command"])

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
                "config_dir": "C:/Users/USER/.claude-agent",
                "block_kind": "auth",
                "reason": "auth/login required",
                "auth_blocked_sessions": 61,
                "command": "$env:CLAUDE_CONFIG_DIR='C:/Users/USER/.claude-agent'; claude",
            },
            {
                "tag": "c10",
                "account": ".claude-c10",
                "config_dir": "C:/Users/USER/.claude-c10",
                "block_kind": "auth",
                "reason": "auth/login required",
                "auth_blocked_sessions": 0,
                "command": "$env:CLAUDE_CONFIG_DIR='C:/Users/USER/.claude-c10'; claude",
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
                            "config_dir": "C:/Users/USER/.claude-c10",
                            "block_kind": "auth",
                            "reason": "auth/login required",
                            "auth_blocked_sessions": 1,
                            "command": "$env:CLAUDE_CONFIG_DIR='C:/Users/USER/.claude-c10'; claude",
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

    def test_collect_loops_classifies_status_and_recommendations(self) -> None:
        ok_script = self.root / "tools" / "ok_loop.py"
        bad_script = self.root / "tools" / "bad_loop.py"
        release_script = self.root / "tools" / "release_loop.py"
        ok_script.write_text("print('{\"ok\": true}')\n", encoding="utf-8")
        bad_script.write_text("print('{\"verdict\": \"STALLED\", \"detail\": \"heartbeat stale\"}')\n", encoding="utf-8")
        release_script.write_text(
            "print('{\"ok\": false, \"verdict\": \"ACTION\", \"detail\": \"billing needs attention\"}')\n",
            encoding="utf-8",
        )
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
            "loops": {
                "healthy": {"status_cmd": [sys.executable, str(ok_script)]},
                "stalled": {
                    "status_cmd": [sys.executable, str(bad_script)],
                    "action": "restart stalled loop",
                },
                "release-status": {"status_cmd": [sys.executable, str(release_script)]},
            },
        }

        loops = pane.collect_loops(config)

        self.assertEqual(loops["count"], 3)
        self.assertEqual(loops["configured"], 3)
        self.assertEqual(loops["enabled"], 3)
        self.assertEqual(loops["disabled"], 0)
        self.assertEqual(loops["states"]["OK"], 1)
        self.assertEqual(loops["states"]["ACTION"], 2)
        stalled = [check for check in loops["checks"] if check["name"] == "stalled"][0]
        self.assertEqual(stalled["detail"], "verdict=STALLED")
        release = [check for check in loops["checks"] if check["name"] == "release-status"][0]
        self.assertEqual(release["detail"], "billing needs attention")

        actions = pane.recommended_actions({
            "config": {"local_config_exists": True},
            "registry": {"exists": True},
            "control_tick": {"task": {}},
            "watchdogs": {},
            "loops": loops,
            "supervisor": {
                "available": True,
                "payload": {"verdict": "OK", "process": {"alive": True}, "diagnose": {"health": "OK"}},
            },
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
        }, config)
        self.assertTrue(any("Loop stalled is ACTION" in action for action in actions))
        self.assertTrue(any("restart stalled loop" in action for action in actions))
        self.assertTrue(any("Loop release-status is ACTION" in action for action in actions))
        self.assertTrue(any("billing needs attention" in action for action in actions))

    def test_repo_catalog_tracks_dos_watchdog_as_dry_run_status_loop(self) -> None:
        catalog_path = Path(__file__).resolve().parent / "control_pane.loops.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

        spec = catalog["loops"]["dos-supervisor-watchdog"]

        self.assertTrue(spec["enabled"])
        self.assertEqual(spec["status_cmd"], ["{python}", "tools/dos_supervisor_watchdog.py", "--json"])
        self.assertFalse(spec["auto_recover"])
        self.assertNotIn("recover_cmd", spec)
        self.assertIn("Dry-run only", spec["action"])

    def test_repo_catalog_tracks_dos_canary_audit_as_read_only_status_loop(self) -> None:
        catalog_path = Path(__file__).resolve().parent / "control_pane.loops.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

        spec = catalog["loops"]["dos-supervisor-canary-audit"]

        self.assertTrue(spec["enabled"])
        self.assertEqual(spec["status_cmd"], ["{python}", "tools/dos_supervisor_canary_audit.py", "--json"])
        self.assertFalse(spec["auto_recover"])
        self.assertNotIn("recover_cmd", spec)
        self.assertIn("Read-only productivity fold", spec["action"])

    def test_repo_catalog_tracks_dos_canary_as_dry_run_status_loop(self) -> None:
        catalog_path = Path(__file__).resolve().parent / "control_pane.loops.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

        spec = catalog["loops"]["dos-supervisor-canary"]

        self.assertTrue(spec["enabled"])
        self.assertEqual(spec["status_cmd"], ["{python}", "tools/dos_supervisor_canary.py", "--json"])
        self.assertFalse(spec["auto_recover"])
        self.assertNotIn("recover_cmd", spec)
        self.assertIn("Dry-run only", spec["action"])

    def test_repo_catalog_tracks_dos_canary_history_as_read_only_status_loop(self) -> None:
        catalog_path = Path(__file__).resolve().parent / "control_pane.loops.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

        spec = catalog["loops"]["dos-supervisor-canary-history"]

        self.assertTrue(spec["enabled"])
        self.assertEqual(spec["status_cmd"], ["{python}", "tools/dos_supervisor_canary.py", "--history", "--json"])
        self.assertFalse(spec["auto_recover"])
        self.assertNotIn("recover_cmd", spec)
        self.assertIn("Read-only canary ledger", spec["action"])

    def test_dos_watchdog_would_enact_payload_is_ok_loop_detail(self) -> None:
        proc = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({
                "schema": "fleet-dos-supervisor-watchdog/1",
                "ok": True,
                "action": "would_enact",
                "reason": "bounded canary available for 3 spawn lane(s)",
            }),
            stderr="",
        )

        state, payload, detail = pane.classify_loop_status(proc, {})

        self.assertEqual(state, "OK")
        self.assertEqual(payload["action"], "would_enact")
        self.assertEqual(detail, "bounded canary available for 3 spawn lane(s)")

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
                    "config_dir": "C:/Users/USER/.claude-gem8",
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

    def test_sync_plan_checks_safe_ff_without_applying(self) -> None:
        fake_sync = mock.Mock()
        fake_sync.current_branch.return_value = "work"
        fake_sync.assess.return_value = {
            "ok": True,
            "state": "behind",
            "target_ref": "origin/work",
            "write_count": 1,
            "identical": [{"status": "M", "path": "a.txt"}],
            "divergent": [],
            "reason": "safe",
        }
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "git_remote": "origin",
        }

        with mock.patch.dict(sys.modules, {"safe_ff_sync": fake_sync}):
            doc = pane.sync_plan(config, fetch=True, apply=False)

        self.assertTrue(doc["ok"])
        self.assertEqual(doc["schema"], pane.SYNC_SCHEMA)
        self.assertEqual(doc["state"], "behind")
        self.assertFalse(doc["applied"])
        fake_sync.assess.assert_called_once_with(str(self.root), "origin", "work", True)
        fake_sync.apply_ff.assert_not_called()

    def test_sync_plan_applies_only_when_assessment_is_safe(self) -> None:
        fake_sync = mock.Mock()
        fake_sync.current_branch.return_value = "work"
        fake_sync.assess.return_value = {
            "ok": True,
            "state": "behind",
            "target_ref": "origin/work",
            "identical": [],
            "divergent": [],
            "reason": "safe",
        }
        fake_sync.apply_ff.return_value = "abc123"
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "git_remote": "origin",
        }

        with mock.patch.dict(sys.modules, {"safe_ff_sync": fake_sync}):
            doc = pane.sync_plan(config, fetch=False, apply=True)

        self.assertTrue(doc["ok"])
        self.assertTrue(doc["applied"])
        self.assertEqual(doc["info"]["new_head"], "abc123")
        fake_sync.apply_ff.assert_called_once()

    def test_sync_plan_apply_refuses_in_progress_merge_before_fast_forward(self) -> None:
        fake_sync = mock.Mock()
        fake_sync.current_branch.return_value = "work"
        fake_sync.assess.return_value = {
            "ok": True,
            "state": "behind",
            "target_ref": "origin/work",
            "identical": [{"status": "M", "path": "VERSION"}],
            "divergent": [],
            "reason": "safe",
        }
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "git_remote": "origin",
        }
        guard = {
            "blocked": True,
            "blockers": ["merge is in progress"],
            "reason": "merge is in progress",
            "worktree": {"merge_in_progress": True, "unmerged_paths": []},
        }

        with (
            mock.patch.dict(sys.modules, {"safe_ff_sync": fake_sync}),
            mock.patch.object(pane, "git_mutation_guard", return_value=guard),
        ):
            doc = pane.sync_plan(config, fetch=False, apply=True)

        self.assertFalse(doc["ok"])
        self.assertFalse(doc["applied"])
        self.assertTrue(doc["sync_blocked"])
        self.assertIn("merge is in progress", doc["reason"])
        fake_sync.apply_ff.assert_not_called()

    def test_sync_plan_refuses_unsafe_apply(self) -> None:
        fake_sync = mock.Mock()
        fake_sync.current_branch.return_value = "work"
        fake_sync.assess.return_value = {
            "ok": False,
            "state": "behind",
            "target_ref": "origin/work",
            "identical": [],
            "divergent": [{"status": "M", "path": "a.txt"}],
            "reason": "diverges",
        }
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "git_remote": "origin",
        }

        with mock.patch.dict(sys.modules, {"safe_ff_sync": fake_sync}):
            doc = pane.sync_plan(config, fetch=False, apply=True)

        self.assertFalse(doc["ok"])
        self.assertFalse(doc["applied"])
        self.assertEqual(doc["reason"], "diverges")
        fake_sync.apply_ff.assert_not_called()

    def test_publish_plan_dry_run_for_ahead_branch(self) -> None:
        fake_sync = mock.Mock()
        fake_sync.current_branch.return_value = "work"
        fake_sync.assess.return_value = {
            "ok": False,
            "state": "ahead",
            "reason": "local branch is ahead",
        }
        ahead = {
            "target": "origin/work",
            "count": 2,
            "shown": 2,
            "limit": 20,
            "commits": [
                {"sha": "a" * 40, "short": "aaaaaaa", "subject": "tools: first"},
                {"sha": "b" * 40, "short": "bbbbbbb", "subject": "tools: second"},
            ],
            "ok": True,
            "reason": "",
        }
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "git_remote": "origin",
        }

        with (
            mock.patch.dict(sys.modules, {"safe_ff_sync": fake_sync}),
            mock.patch.object(pane, "ahead_commits", return_value=ahead) as ahead_mock,
        ):
            doc = pane.publish_plan(config, fetch=True, apply=False)

        self.assertTrue(doc["ok"])
        self.assertFalse(doc["applied"])
        self.assertEqual(doc["schema"], pane.PUBLISH_SCHEMA)
        self.assertEqual(doc["ahead_commits"], ahead)
        self.assertIn("git push origin HEAD:work", doc["commands"][0])
        self.assertIn("dry run", doc["reason"])
        ahead_mock.assert_called_once_with(self.root, "origin", "work")

    def test_publish_plan_refuses_non_ahead_branch(self) -> None:
        fake_sync = mock.Mock()
        fake_sync.current_branch.return_value = "work"
        fake_sync.assess.return_value = {
            "ok": True,
            "state": "behind",
            "reason": "sync first",
        }
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "git_remote": "origin",
        }

        with mock.patch.dict(sys.modules, {"safe_ff_sync": fake_sync}):
            doc = pane.publish_plan(config, fetch=True, apply=True)

        self.assertFalse(doc["ok"])
        self.assertFalse(doc["applied"])
        self.assertEqual(doc["reason"], "sync first")

    def test_publish_plan_refuses_dirty_worktree(self) -> None:
        self._init_git_repo()
        subprocess.run(["git", "add", "VERSION"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-m", "base"], cwd=self.root, check=True, capture_output=True)
        (self.root / "dirty.txt").write_text("dirty\n", encoding="utf-8")
        fake_sync = mock.Mock()
        fake_sync.current_branch.return_value = "work"
        fake_sync.assess.return_value = {
            "ok": False,
            "state": "ahead",
            "reason": "local branch is ahead",
        }
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "git_remote": "origin",
        }

        with (
            mock.patch.dict(sys.modules, {"safe_ff_sync": fake_sync}),
            mock.patch.object(pane, "ahead_commits", return_value={"target": "origin/work", "count": 1, "shown": 1, "commits": []}),
        ):
            doc = pane.publish_plan(config, fetch=True, apply=False)

        self.assertFalse(doc["ok"])
        self.assertFalse(doc["applied"])
        self.assertTrue(doc["publish_blocked"])
        self.assertIn("worktree has 1 dirty path(s)", doc["reason"])
        self.assertEqual(doc["worktree"]["dirty_total"], 1)

    def test_publish_plan_allow_dirty_permits_ordinary_dirty_paths(self) -> None:
        self._init_git_repo()
        subprocess.run(["git", "add", "VERSION"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-m", "base"], cwd=self.root, check=True, capture_output=True)
        (self.root / "dirty.txt").write_text("dirty\n", encoding="utf-8")
        fake_sync = mock.Mock()
        fake_sync.current_branch.return_value = "work"
        fake_sync.assess.return_value = {
            "ok": False,
            "state": "ahead",
            "reason": "local branch is ahead",
        }
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "git_remote": "origin",
        }

        with (
            mock.patch.dict(sys.modules, {"safe_ff_sync": fake_sync}),
            mock.patch.object(pane, "ahead_commits", return_value={"target": "origin/work", "count": 1, "shown": 1, "commits": []}),
        ):
            doc = pane.publish_plan(config, fetch=True, apply=False, allow_dirty=True)

        self.assertTrue(doc["ok"])
        self.assertFalse(doc["applied"])
        self.assertTrue(doc["allow_dirty"])
        self.assertEqual(doc["worktree"]["dirty_total"], 1)
        self.assertIn("dry run", doc["reason"])

    def test_publish_plan_allow_dirty_still_refuses_unmerged_paths(self) -> None:
        fake_sync = mock.Mock()
        fake_sync.current_branch.return_value = "work"
        fake_sync.assess.return_value = {
            "ok": False,
            "state": "ahead",
            "reason": "local branch is ahead",
        }
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "git_remote": "origin",
        }
        worktree = {
            "status_available": True,
            "dirty": True,
            "dirty_total": 1,
            "counts": {"unmerged": 1},
            "unmerged_paths": ["conflicted.txt"],
            "merge_in_progress": False,
        }

        with (
            mock.patch.dict(sys.modules, {"safe_ff_sync": fake_sync}),
            mock.patch.object(pane, "ahead_commits", return_value={"target": "origin/work", "count": 1, "shown": 1, "commits": []}),
            mock.patch.object(pane, "git_worktree_summary", return_value=worktree),
        ):
            doc = pane.publish_plan(config, fetch=True, apply=False, allow_dirty=True)

        self.assertFalse(doc["ok"])
        self.assertTrue(doc["publish_blocked"])
        self.assertIn("unmerged path", doc["reason"])

    def test_publish_plan_apply_refuses_in_progress_merge_before_push(self) -> None:
        fake_sync = mock.Mock()
        fake_sync.current_branch.return_value = "work"
        fake_sync.assess.return_value = {
            "ok": False,
            "state": "ahead",
            "reason": "local branch is ahead",
        }
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "git_remote": "origin",
        }
        worktree = {
            "status_available": True,
            "dirty": True,
            "dirty_total": 1,
            "counts": {},
            "unmerged_paths": [],
            "merge_in_progress": True,
        }

        with (
            mock.patch.dict(sys.modules, {"safe_ff_sync": fake_sync}),
            mock.patch.object(pane, "ahead_commits", return_value={"target": "origin/work", "count": 1, "shown": 1, "commits": []}),
            mock.patch.object(pane, "git_worktree_summary", return_value=worktree),
            mock.patch.object(pane, "git") as git_mock,
        ):
            doc = pane.publish_plan(config, fetch=True, apply=True, allow_dirty=True)

        self.assertFalse(doc["ok"])
        self.assertFalse(doc["applied"])
        self.assertTrue(doc["publish_blocked"])
        self.assertIn("merge is in progress", doc["reason"])
        git_mock.assert_not_called()

    def test_publish_plan_applies_push_for_ahead_branch(self) -> None:
        fake_sync = mock.Mock()
        fake_sync.current_branch.return_value = "work"
        fake_sync.assess.return_value = {
            "ok": False,
            "state": "ahead",
            "reason": "local branch is ahead",
        }
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "git_remote": "origin",
        }
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="pushed\n", stderr="")

        with (
            mock.patch.dict(sys.modules, {"safe_ff_sync": fake_sync}),
            mock.patch.object(pane, "ahead_commits", return_value={"target": "origin/work", "count": 1, "shown": 1, "commits": []}),
            mock.patch.object(pane, "git_worktree_summary", return_value={
                "status_available": True,
                "dirty": False,
                "dirty_total": 0,
                "counts": {},
                "entries": [],
                "unmerged_paths": [],
                "merge_in_progress": False,
            }),
            mock.patch.object(pane, "git", return_value=proc) as git_mock,
        ):
            doc = pane.publish_plan(config, fetch=True, apply=True)

        self.assertTrue(doc["ok"])
        self.assertTrue(doc["applied"])
        git_mock.assert_called_once_with(["push", "origin", "HEAD:work"], self.root)

    def test_ahead_commits_lists_remote_delta(self) -> None:
        self._init_git_repo()
        (self.root / "a.txt").write_text("one\n", encoding="utf-8")
        subprocess.run(["git", "add", "a.txt"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-m", "base"], cwd=self.root, check=True, capture_output=True)
        subprocess.run(["git", "update-ref", "refs/remotes/origin/work", "HEAD"], cwd=self.root, check=True)
        (self.root / "a.txt").write_text("two\n", encoding="utf-8")
        subprocess.run(["git", "commit", "-am", "second"], cwd=self.root, check=True, capture_output=True)
        (self.root / "a.txt").write_text("three\n", encoding="utf-8")
        subprocess.run(["git", "commit", "-am", "third"], cwd=self.root, check=True, capture_output=True)

        doc = pane.ahead_commits(self.root, "origin", "work")

        self.assertTrue(doc["ok"])
        self.assertEqual(doc["target"], "origin/work")
        self.assertEqual(doc["count"], 2)
        self.assertEqual(doc["shown"], 2)
        self.assertEqual([c["subject"] for c in doc["commits"]], ["third", "second"])

    def _init_git_repo(self) -> None:
        subprocess.run(["git", "init", "-b", "work"], cwd=self.root, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.root, check=True)

    def _write_trunk_guard_files(self) -> None:
        hook = self.root / "tools" / "githooks" / "reference-transaction"
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        installer = self.root / "tools" / "install_trunk_guard.py"
        installer.write_text(
            "import pathlib, subprocess\n"
            "root = pathlib.Path(__file__).resolve().parents[1]\n"
            "subprocess.run(['git', '-C', str(root), 'config', 'core.hooksPath', 'tools/githooks'], check=True)\n"
            "print('installed trunk guard')\n",
            encoding="utf-8",
        )

    def _write_doctor_tool_files(self) -> None:
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

    def test_setup_plan_marks_missing_trunk_guard_todo(self) -> None:
        self._init_git_repo()
        self._write_trunk_guard_files()
        config = pane.normalize_config(pane.default_config(self.root), self.root)

        plan = pane.setup_plan(config, tick={
            "task_name": "FleetControlPaneTick",
            "register_script": "tools/register_control_pane_tick.ps1",
            "register_script_exists": False,
            "task": {"supported": False},
        })

        steps = {step["id"]: step for step in plan["steps"]}
        self.assertEqual(steps["install-trunk-guard"]["state"], "todo")
        self.assertTrue(steps["install-trunk-guard"]["ok"])
        self.assertIn("expected tools/githooks", steps["install-trunk-guard"]["detail"])
        self.assertIn("tools/install_trunk_guard.py", steps["install-trunk-guard"]["display"])

    def test_setup_plan_marks_installed_trunk_guard_done(self) -> None:
        self._init_git_repo()
        self._write_trunk_guard_files()
        subprocess.run(["git", "config", "core.hooksPath", "tools/githooks"], cwd=self.root, check=True)
        config = pane.normalize_config(pane.default_config(self.root), self.root)

        plan = pane.setup_plan(config, tick={
            "task_name": "FleetControlPaneTick",
            "register_script": "tools/register_control_pane_tick.ps1",
            "register_script_exists": False,
            "task": {"supported": False},
        })

        steps = {step["id"]: step for step in plan["steps"]}
        self.assertEqual(steps["install-trunk-guard"]["state"], "done")
        self.assertIn("core.hooksPath=tools/githooks", steps["install-trunk-guard"]["detail"])

    def test_doctor_reports_missing_trunk_guard(self) -> None:
        self._init_git_repo()
        self._write_trunk_guard_files()
        self._write_doctor_tool_files()
        for rel in ("tools/_registry/machines", "tools/_watchdog"):
            (self.root / rel).mkdir(parents=True)
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
            "watchdogs": {},
        }

        with mock.patch.object(pane, "collect_control_tick", return_value={
            "register_script_exists": True,
            "register_script": "tools/register_control_pane_tick.ps1",
            "task": {"supported": False},
        }):
            doc = pane.doctor(config)

        checks = {check["name"]: check for check in doc["checks"]}
        self.assertFalse(doc["ok"])
        self.assertFalse(checks["trunk-guard"]["ok"])
        self.assertIn("expected tools/githooks", checks["trunk-guard"]["detail"])

    def test_bootstrap_apply_installs_trunk_guard(self) -> None:
        self._init_git_repo()
        self._write_trunk_guard_files()
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": False,
        }

        with mock.patch.object(pane, "collect_control_tick", return_value={
            "task_name": "FleetControlPaneTick",
            "task": {"supported": True, "installed": True},
        }):
            doc = pane.bootstrap(config, apply=True)

        actions = {action["id"]: action for action in doc["actions"]}
        self.assertTrue(doc["ok"])
        self.assertTrue(actions["trunk-guard"]["changed"])
        self.assertIn("installed trunk guard", actions["trunk-guard"]["detail"])
        got = subprocess.check_output(
            ["git", "config", "--get", "core.hooksPath"],
            cwd=self.root,
            text=True,
        ).strip()
        self.assertEqual(got, "tools/githooks")

    def test_commit_plan_refuses_foreign_staged_paths(self) -> None:
        self._init_git_repo()
        (self.root / "a.txt").write_text("a\n", encoding="utf-8")
        (self.root / "b.txt").write_text("b\n", encoding="utf-8")
        subprocess.run(["git", "add", "b.txt"], cwd=self.root, check=True)

        doc = pane.commit_plan(
            {"root": str(self.root)},
            paths=["a.txt"],
            message="test: a",
            apply=False,
        )

        self.assertFalse(doc["ok"])
        self.assertIn("b.txt", doc["foreign_staged"])
        self.assertIn("unrelated paths", doc["reason"])

    def test_commit_plan_refuses_in_progress_merge(self) -> None:
        self._init_git_repo()
        (self.root / "a.txt").write_text("base\n", encoding="utf-8")
        subprocess.run(["git", "add", "a.txt"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-m", "base"], cwd=self.root, check=True, capture_output=True)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root, text=True).strip()
        git_dir = subprocess.check_output(["git", "rev-parse", "--git-dir"], cwd=self.root, text=True).strip()
        (self.root / git_dir / "MERGE_HEAD").write_text(head + "\n", encoding="utf-8")
        (self.root / "a.txt").write_text("changed\n", encoding="utf-8")

        doc = pane.commit_plan(
            {"root": str(self.root)},
            paths=["a.txt"],
            message="test: a",
            apply=False,
        )

        self.assertFalse(doc["ok"])
        self.assertTrue(doc["merge_in_progress"])
        self.assertIn("merge is in progress", doc["reason"])

    def test_commit_plan_apply_commits_only_selected_path(self) -> None:
        self._init_git_repo()
        (self.root / "a.txt").write_text("a\n", encoding="utf-8")
        (self.root / "b.txt").write_text("b\n", encoding="utf-8")

        doc = pane.commit_plan(
            {"root": str(self.root)},
            paths=["a.txt"],
            message="test: commit selected",
            apply=True,
        )

        self.assertTrue(doc["ok"])
        self.assertTrue(doc["applied"])
        self.assertIn("commit", doc)
        tracked = subprocess.check_output(
            ["git", "ls-tree", "--name-only", "HEAD"],
            cwd=self.root,
            text=True,
        ).splitlines()
        self.assertEqual(tracked, ["a.txt"])
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1"],
            cwd=self.root,
            text=True,
        )
        self.assertIn("?? b.txt", status)

    def test_commit_plan_rejects_directory_path_by_default(self) -> None:
        (self.root / "dir").mkdir()
        with self.assertRaises(ValueError):
            pane.commit_plan(
                {"root": str(self.root)},
                paths=["dir"],
                message="test: dir",
                apply=False,
            )

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
