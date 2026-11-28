#!/usr/bin/env python3
"""Git status, trunk guard, commit plan, sync plan, and publish plan tests for fleet_control_pane."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_control_pane as pane  # noqa: E402


class FleetControlPaneGitTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "tools").mkdir()
        (self.root / "VERSION").write_text("test\n", encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

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

    def test_dirty_commit_plan_blocks_unmerged_groups(self) -> None:
        plan = pane.dirty_commit_plan(
            [
                {"code": "UU", "path": "VERSION"},
                {"code": " M", "path": "tools/appversion.py"},
                {"code": " M", "path": "docs/readme.md"},
            ],
            {"python": "python"},
        )

        groups = {group["group"]: group for group in plan["groups"]}
        self.assertTrue(groups["tools/fleet-control-pane"]["blocked"])
        self.assertIn("tools/appversion.py", groups["tools/fleet-control-pane"]["paths"])
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
