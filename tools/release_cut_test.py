#!/usr/bin/env python3
"""Tests for tools/release_cut.py."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "release_cut.py"


def load():
    spec = importlib.util.spec_from_file_location("release_cut", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ReleaseCutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="release_cut_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.old_env = {k: os.environ.get(k) for k in (
            "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_NOSYSTEM",
            "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL",
        )}
        os.environ.update({
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
        })
        self.addCleanup(self._restore_env)

    def _restore_env(self) -> None:
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _git(self, root: Path, *args: str) -> str:
        proc = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
        return proc.stdout.strip()

    def _write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _clone_with_upstream(self) -> tuple[Path, Path]:
        origin = self.tmp / "origin"
        origin.mkdir()
        self._git(origin, "init", "-b", "master")
        self._write(origin / "VERSION", "0.1.0\n")
        self._git(origin, "add", "VERSION")
        self._git(origin, "commit", "-m", "seed")
        clone = self.tmp / "clone"
        self._git(self.tmp, "clone", str(origin), str(clone))
        return origin, clone

    def test_paths_from_bump_report(self) -> None:
        rc = load()
        report = {
            "targets": {
                "version": {"path": "VERSION", "ok": True, "changed": True},
                "skip": {"path": "SKIP", "skipped": True},
            }
        }
        self.assertEqual(rc.paths_from_bump_report(report), ["VERSION"])

    def test_render_notes_groups_commit_subjects(self) -> None:
        rc = load()
        text = rc.render_notes(
            "0.23.0",
            date="2026-06-18",
            level="minor",
            themes=["tools"],
            headline="Minor release for tools",
            commits=[
                {"subject": "feat(tools): add release cut"},
                {"subject": "fix(agent): repair adapter"},
                {"subject": "v0.22.0: prior release"},
            ],
        )
        self.assertIn("version: 0.23.0", text)
        self.assertIn("headline: \"Minor release for tools\"", text)
        self.assertIn("## tools", text)
        self.assertIn("- feat(tools): add release cut", text)
        self.assertNotIn("prior release", text)

    def test_render_notes_normalizes_unicode_subjects(self) -> None:
        rc = load()
        text = rc.render_notes(
            "0.23.0",
            date="2026-06-18",
            level="minor",
            themes=["tools"],
            headline="Minor release",
            commits=[{"subject": "feat(tools): worktree doctor \u2014 safe prune"}],
        )
        self.assertIn("worktree doctor - safe prune", text)
        text.encode("ascii")

    def test_render_notes_preserves_generation_metadata(self) -> None:
        rc = load()
        text = rc.render_notes(
            "0.23.0",
            date="2026-06-18",
            level="minor",
            themes=["tools"],
            headline="Minor release",
            commits=[
                {"subject": "feat(tools): add generation preview", "generation": "gen/now"},
                {"subject": "fix(agent): preserve horizon", "body": "Generation: next"},
                {"subject": "v0.22.0: prior release", "generation": "gen/future"},
            ],
        )

        self.assertIn("generations:\n  gen/now: 1\n  gen/next: 1", text)
        self.assertIn("## Generation", text)
        self.assertIn("- gen/now: 1 commit(s)", text)
        self.assertIn("- feat(tools): add generation preview [gen/now]", text)
        self.assertIn("- fix(agent): preserve horizon [gen/next]", text)
        self.assertNotIn("gen/future: 1", text)

    def test_upstream_state_refuses_behind_branch_for_release_cut(self) -> None:
        rc = load()
        origin, clone = self._clone_with_upstream()
        self._write(origin / "remote.txt", "remote\n")
        self._git(origin, "add", "remote.txt")
        self._git(origin, "commit", "-m", "remote")
        self._git(clone, "fetch", "origin")

        state = rc.upstream_state(clone)
        self.assertEqual(state["state"], "behind")
        self.assertFalse(state["ok_for_release_cut"])
        self.assertEqual(state["behind"], 1)

    def test_upstream_state_allows_ahead_branch(self) -> None:
        rc = load()
        _origin, clone = self._clone_with_upstream()
        self._write(clone / "local.txt", "local\n")
        self._git(clone, "add", "local.txt")
        self._git(clone, "commit", "-m", "local")

        state = rc.upstream_state(clone)
        self.assertEqual(state["state"], "ahead")
        self.assertTrue(state["ok_for_release_cut"])
        self.assertEqual(state["ahead"], 1)

    def test_allow_hold_turns_decider_hold_into_noop_plan(self) -> None:
        rc = load()
        root = self.tmp / "holdrepo"
        root.mkdir()
        self._git(root, "init", "-b", "master")
        self._write(root / "VERSION", "0.1.0\n")
        tools = root / "tools"
        tools.mkdir()
        self._write(tools / "release_context.py", "import json; print(json.dumps({'commits_since_tag': []}))\n")
        self._write(tools / "release_decide.py", (
            "import json, sys\n"
            "print(json.dumps({'decision':'hold','reason':'nothing to ship','blockers':['NOTHING_TO_SHIP']}))\n"
            "sys.exit(2)\n"
        ))
        plan = rc.build_plan(
            root,
            version=None,
            level=None,
            themes=[],
            headline=None,
            date="2026-06-18",
            includes=[],
            force=False,
            allow_hold=True,
            limit_commits=20,
        )
        self.assertTrue(plan["ok"])
        self.assertTrue(plan["held"])
        self.assertEqual(plan["reason"], "nothing to ship")

    def test_require_ci_green_is_passed_to_decider(self) -> None:
        rc = load()
        root = self.tmp / "strictci"
        root.mkdir()
        self._git(root, "init", "-b", "master")
        self._write(root / "VERSION", "0.1.0\n")
        tools = root / "tools"
        tools.mkdir()
        self._write(tools / "release_context.py", (
            "import json\n"
            "print(json.dumps({'last_tag':'v0.1.0','latest_any_tag':'v0.1.0',"
            "'commits_since_tag':[{'subject':'feat(x): y'}]}))\n"
        ))
        self._write(tools / "release_decide.py", (
            "import json, pathlib, sys\n"
            "pathlib.Path('decider-argv.json').write_text(json.dumps(sys.argv), encoding='utf-8')\n"
            "print(json.dumps({'decision':'release','next_version':'0.2.0','level':'minor','themes':['x']}))\n"
        ))
        self._write(tools / "release_bump.py", (
            "import json, sys\n"
            "print(json.dumps({'targets': {'version': {'path': 'VERSION'}}}))\n"
        ))

        plan = rc.build_plan(
            root,
            version=None,
            level=None,
            themes=[],
            headline=None,
            date="2026-06-18",
            includes=[],
            force=False,
            allow_hold=False,
            limit_commits=20,
            require_ci_green=True,
        )

        self.assertTrue(plan["ok"])
        argv = json.loads((root / "decider-argv.json").read_text(encoding="utf-8"))
        self.assertIn("--require-ci-green", argv)

    def test_from_manifest_adds_structured_include_paths(self) -> None:
        rc = load()
        root = self.tmp / "manifestrepo"
        root.mkdir()
        self._git(root, "init", "-b", "master")
        self._write(root / "VERSION", "0.1.0\n")
        tools = root / "tools"
        tools.mkdir()
        self._write(tools / "release_context.py", (
            "import json\n"
            "print(json.dumps({'last_tag':'v0.1.0','latest_any_tag':'v0.1.0',"
            "'commits_since_tag':[{'subject':'feat(x): y'}]}))\n"
        ))
        self._write(tools / "release_decide.py", (
            "import json\n"
            "print(json.dumps({'decision':'release','next_version':'0.2.0','level':'minor','themes':['x']}))\n"
        ))
        self._write(tools / "release_bump.py", (
            "import json\n"
            "print(json.dumps({'targets': {'version': {'path': 'VERSION'}}}))\n"
        ))
        self._write(tools / "release_manifest.py", (
            "import json, sys\n"
            "print(json.dumps({'ok': True, 'staged_paths': ['feature.txt'],"
            "'auto_deferred_paths': ['screenshots/shot.png'],"
            "'producer': {'skill': 'dispatch', 'run_id': 'RID-1', 'packet_path': 'packet.md'},"
            "'commits_section_lines': ['- (P P1) abcdef0 feat: x']}))\n"
        ))
        manifest = root / "release-manifest.json"
        self._write(manifest, "{}\n")

        plan = rc.build_plan(
            root,
            version=None,
            level=None,
            themes=[],
            headline=None,
            date="2026-06-18",
            includes=[],
            from_manifest=str(manifest),
            force=False,
            allow_hold=False,
            limit_commits=20,
        )

        self.assertTrue(plan["ok"])
        self.assertIn("feature.txt", plan["include_paths"])
        self.assertIn("feature.txt", plan["paths"])
        self.assertIn("Release manifest", plan["notes_preview"])
        self.assertIn("RID-1", plan["notes_preview"])

    def test_execute_unwinds_own_release_commit_when_dry_run_fails(self) -> None:
        rc = load()
        root = self.tmp / "faildry"
        root.mkdir()
        self._git(root, "init", "-b", "master")
        self._write(root / "VERSION", "0.1.0\n")
        self._git(root, "add", "VERSION")
        self._git(root, "commit", "-m", "seed")
        tools = root / "tools"
        tools.mkdir()
        self._write(tools / "release_bump.py", (
            "import json, pathlib, sys\n"
            "pathlib.Path('VERSION').write_text(sys.argv[1] + '\\n', encoding='utf-8')\n"
            "print(json.dumps({'targets': {'version': {'path': 'VERSION', 'ok': True}}}))\n"
        ))
        self._write(tools / "release_lock.py", (
            "import json\n"
            "print(json.dumps({'ok': True, 'lock': {'owner': 'test'}}))\n"
        ))
        self._write(tools / "release_dry_run.py", (
            "import json, sys\n"
            "print(json.dumps({'ok': False, 'trailer': 'release-dry-run: FAIL'}))\n"
            "sys.exit(1)\n"
        ))
        self._git(root, "add", "tools")
        self._git(root, "commit", "-m", "test release helpers")
        seed = self._git(root, "rev-parse", "HEAD")
        plan = {
            "ok": True,
            "version": "0.2.0",
            "tag": "v0.2.0",
            "headline": "failed dry run",
            "paths": ["VERSION", "docs/releases/v0.2.0.md"],
            "notes_file": "docs/releases/v0.2.0.md",
            "notes_preview": "release notes\n",
        }

        result = rc.execute_plan(
            root,
            plan,
            includes=[],
            overwrite_notes=False,
            skip_dry_run=False,
            ttl=1800,
            allow_stale_upstream=False,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["aborted"], "release_dry_run failed on release commit")
        self.assertTrue(result["release_commit_unwind"]["ok"])
        self.assertIsNone(result["commit_sha"])
        self.assertEqual(self._git(root, "rev-parse", "HEAD"), seed)
        self.assertNotIn("v0.2.0: failed dry run", self._git(root, "log", "--oneline"))
        self.assertEqual((root / "VERSION").read_text(encoding="utf-8"), "0.2.0\n")

    def test_execute_uses_parent_release_lock_without_releasing_it(self) -> None:
        rc = load()
        root = self.tmp / "parentlock"
        root.mkdir()
        self._git(root, "init", "-b", "master")
        self._write(root / "VERSION", "0.1.0\n")
        self._git(root, "add", "VERSION")
        self._git(root, "commit", "-m", "seed")
        tools = root / "tools"
        tools.mkdir()
        self._write(tools / "release_bump.py", (
            "import json, pathlib, sys\n"
            "pathlib.Path('VERSION').write_text(sys.argv[1] + '\\n', encoding='utf-8')\n"
            "print(json.dumps({'targets': {'version': {'path': 'VERSION', 'ok': True}}}))\n"
        ))
        self._write(tools / "release_lock.py", (
            "import json, pathlib, sys\n"
            "cmd = sys.argv[1]\n"
            "p = pathlib.Path('lock.log')\n"
            "old = p.read_text(encoding='utf-8') if p.exists() else ''\n"
            "p.write_text(old + cmd + '\\n', encoding='utf-8')\n"
            "print(json.dumps({'ok': True, 'cmd': cmd}))\n"
        ))
        self._git(root, "add", "tools")
        self._git(root, "commit", "-m", "test release helpers")
        plan = {
            "ok": True,
            "version": "0.2.0",
            "tag": "v0.2.0",
            "headline": "parent lock release",
            "paths": ["VERSION", "docs/releases/v0.2.0.md"],
            "notes_file": "docs/releases/v0.2.0.md",
            "notes_preview": "release notes\n",
        }

        result = rc.execute_plan(
            root,
            plan,
            includes=[],
            overwrite_notes=False,
            skip_dry_run=True,
            ttl=1800,
            allow_stale_upstream=False,
            lock_already_held=True,
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["release_lock"]["held_by_parent"])
        self.assertEqual(
            (root / "lock.log").read_text(encoding="utf-8").splitlines(),
            ["verify", "guard"],
        )
        self.assertEqual((root / "VERSION").read_text(encoding="utf-8"), "0.2.0\n")

    def test_execute_refuses_when_parent_release_lock_held_by_another_owner(self) -> None:
        # AC #1391: the cadence runs the cut under --lock-already-held (a parent
        # workflow step took the lock). If the lock is actually held by ANOTHER
        # owner — e.g. a human mid-`/release` when the 2h auto-cut tick fires —
        # then `release_lock.py verify` fails (this session is not the holder) and
        # the execute path must REFUSE before any VERSION/tag mutation, deferring
        # to the next tick instead of racing the human.
        rc = load()
        root = self.tmp / "foreignparentlock"
        root.mkdir()
        self._git(root, "init", "-b", "master")
        self._write(root / "VERSION", "0.1.0\n")
        self._git(root, "add", "VERSION")
        self._git(root, "commit", "-m", "seed")
        tools = root / "tools"
        tools.mkdir()
        # A bump stub that WOULD mutate VERSION if the refuse were (wrongly) skipped.
        self._write(tools / "release_bump.py", (
            "import json, pathlib, sys\n"
            "pathlib.Path('VERSION').write_text(sys.argv[1] + '\\n', encoding='utf-8')\n"
            "print(json.dumps({'targets': {'version': {'path': 'VERSION', 'ok': True}}}))\n"
        ))
        # verify exits 3 (release_lock.py's contended/denied code): the live lock
        # is owned by another session, so this bot is NOT the holder.
        self._write(tools / "release_lock.py", (
            "import json, pathlib, sys\n"
            "cmd = sys.argv[1]\n"
            "p = pathlib.Path('lock.log')\n"
            "old = p.read_text(encoding='utf-8') if p.exists() else ''\n"
            "p.write_text(old + cmd + '\\n', encoding='utf-8')\n"
            "print(json.dumps({'ok': False, 'reason': 'held by another session'}))\n"
            "sys.exit(3)\n"
        ))
        self._git(root, "add", "tools")
        self._git(root, "commit", "-m", "test release helpers")
        seed = self._git(root, "rev-parse", "HEAD")
        plan = {
            "ok": True,
            "version": "0.2.0",
            "tag": "v0.2.0",
            "headline": "foreign parent lock",
            "paths": ["VERSION", "docs/releases/v0.2.0.md"],
            "notes_file": "docs/releases/v0.2.0.md",
            "notes_preview": "release notes\n",
        }

        result = rc.execute_plan(
            root,
            plan,
            includes=[],
            overwrite_notes=False,
            skip_dry_run=True,
            ttl=1800,
            allow_stale_upstream=False,
            lock_already_held=True,
        )

        self.assertFalse(result["ok"], result)
        self.assertEqual(result["aborted"], "parent release lock not held")
        # Refused at the verify gate, BEFORE any mutation: no bump, no guard, no commit.
        self.assertEqual(
            (root / "lock.log").read_text(encoding="utf-8").splitlines(),
            ["verify"],
        )
        self.assertNotIn("release_lock", result)
        self.assertEqual((root / "VERSION").read_text(encoding="utf-8"), "0.1.0\n")
        self.assertEqual(self._git(root, "rev-parse", "HEAD"), seed)
        self.assertNotIn("v0.2.0", self._git(root, "log", "--oneline"))

    def test_execute_refuses_when_release_lock_acquire_denied(self) -> None:
        # AC #1391 (symmetric, self-acquired path): when release_cut takes the
        # lock itself (no parent driver), a denial because another owner already
        # holds it must REFUSE before any VERSION/tag mutation — the same defer a
        # concurrent `/release` owes a held cadence lock.
        rc = load()
        root = self.tmp / "acquiredenied"
        root.mkdir()
        self._git(root, "init", "-b", "master")
        self._write(root / "VERSION", "0.1.0\n")
        self._git(root, "add", "VERSION")
        self._git(root, "commit", "-m", "seed")
        tools = root / "tools"
        tools.mkdir()
        self._write(tools / "release_bump.py", (
            "import json, pathlib, sys\n"
            "pathlib.Path('VERSION').write_text(sys.argv[1] + '\\n', encoding='utf-8')\n"
            "print(json.dumps({'targets': {'version': {'path': 'VERSION', 'ok': True}}}))\n"
        ))
        # acquire exits 3 (contended/denied): another owner holds the live lock.
        self._write(tools / "release_lock.py", (
            "import json, pathlib, sys\n"
            "cmd = sys.argv[1]\n"
            "p = pathlib.Path('lock.log')\n"
            "old = p.read_text(encoding='utf-8') if p.exists() else ''\n"
            "p.write_text(old + cmd + '\\n', encoding='utf-8')\n"
            "print(json.dumps({'ok': False, 'reason': 'held by another session'}))\n"
            "sys.exit(3)\n"
        ))
        self._git(root, "add", "tools")
        self._git(root, "commit", "-m", "test release helpers")
        seed = self._git(root, "rev-parse", "HEAD")
        plan = {
            "ok": True,
            "version": "0.2.0",
            "tag": "v0.2.0",
            "headline": "acquire denied",
            "paths": ["VERSION", "docs/releases/v0.2.0.md"],
            "notes_file": "docs/releases/v0.2.0.md",
            "notes_preview": "release notes\n",
        }

        result = rc.execute_plan(
            root,
            plan,
            includes=[],
            overwrite_notes=False,
            skip_dry_run=True,
            ttl=1800,
            allow_stale_upstream=False,
        )

        self.assertFalse(result["ok"], result)
        self.assertEqual(result["aborted"], "could not acquire release lock")
        # Only the acquire ran — no guard, no commit; the lock it never got is not released.
        self.assertEqual(
            (root / "lock.log").read_text(encoding="utf-8").splitlines(),
            ["acquire"],
        )
        self.assertEqual((root / "VERSION").read_text(encoding="utf-8"), "0.1.0\n")
        self.assertEqual(self._git(root, "rev-parse", "HEAD"), seed)
        self.assertNotIn("v0.2.0", self._git(root, "log", "--oneline"))

    def test_unwind_failed_release_commit_refuses_when_head_changed(self) -> None:
        rc = load()
        root = self.tmp / "headmoved"
        root.mkdir()
        self._git(root, "init", "-b", "master")
        self._write(root / "VERSION", "0.1.0\n")
        self._git(root, "add", "VERSION")
        self._git(root, "commit", "-m", "seed")
        self._write(root / "VERSION", "0.2.0\n")
        self._git(root, "commit", "-am", "v0.2.0: release")
        release_sha = self._git(root, "rev-parse", "HEAD")
        self._write(root / "peer.txt", "peer\n")
        self._git(root, "add", "peer.txt")
        self._git(root, "commit", "-m", "peer commit")
        peer_sha = self._git(root, "rev-parse", "HEAD")

        result = rc.unwind_failed_release_commit(root, release_sha)

        self.assertFalse(result["ok"])
        self.assertTrue(result["skipped"])
        self.assertIn("HEAD changed", result["reason"])
        self.assertEqual(self._git(root, "rev-parse", "HEAD"), peer_sha)

    def test_live_cli_dry_run_no_mutation(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--json", "--limit-commits", "20"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        self.assertIn(proc.returncode, (0, 1), proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertIn("ok", payload)
        if payload["ok"]:
            self.assertTrue(payload["dry_run"])
            self.assertIn("VERSION", payload["paths"])
            self.assertRegex(payload["version"], r"^\d+\.\d+\.\d+$")

    def test_live_cli_allow_hold_contract(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--json", "--allow-hold", "--limit-commits", "20"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload.get("dry_run"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
