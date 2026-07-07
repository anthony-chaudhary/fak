#!/usr/bin/env python3
"""Hermetic tests for tools/worker_worktree.py (#1334).

NOTHING live runs: the git-touching create/reap/land path is exercised with an
injected fake ``git`` runner that records the argv it was handed and returns
canned ``(rc, stdout)``, so the whole reconciliation — detach at trunk HEAD,
land onto the trunk by explicit path, reap — is asserted without a real repo.
The pure planners (name/path/env/parse) are tested directly.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "worker_worktree.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("worker_worktree", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = load()


class FakeGit:
    """Records every (root, args) call; replies from a queue or a default by verb."""

    def __init__(self, replies=None, default=(0, "")):
        self.calls: list[list[str]] = []
        self.replies = dict(replies or {})  # keyed by args[0] (the git verb)
        self.default = default

    def __call__(self, root, args):
        self.calls.append(list(args))
        verb = args[0] if args else ""
        rep = self.replies.get(verb)
        if rep is None:
            return self.default
        if isinstance(rep, list):
            return rep.pop(0) if rep else self.default
        return rep


class SafeKeyAndNameTest(unittest.TestCase):
    def test_dir_name_is_one_flat_safe_segment(self) -> None:
        name = mod.worktree_dir_name("tools", "1334")
        self.assertTrue(name.startswith("fak-worker-wt-tools-"))
        self.assertNotIn("/", name)
        self.assertNotIn("\\", name)

    def test_hostile_key_cannot_escape_root(self) -> None:
        # A key with separators / traversal must hash to a flat token, never a path.
        name = mod.worktree_dir_name("tools", "../../etc/passwd")
        self.assertNotIn("/", name)
        self.assertNotIn("..", name)

    def test_odd_lane_is_sanitised(self) -> None:
        name = mod.worktree_dir_name("a/b lane", "k")
        self.assertNotIn("/", name)
        self.assertNotIn(" ", name)

    def test_distinct_keys_distinct_dirs(self) -> None:
        self.assertNotEqual(mod.worktree_dir_name("tools", "1"),
                            mod.worktree_dir_name("tools", "2"))


class WorktreePathTest(unittest.TestCase):
    def test_path_under_explicit_root(self) -> None:
        root = Path("/tmp/wtroot")
        p = mod.worktree_path("gateway", "42", root=root)
        self.assertEqual(p.parent, root)
        self.assertTrue(p.name.startswith("fak-worker-wt-gateway-"))

    def test_default_root_honours_env(self) -> None:
        os.environ[mod.WORKTREE_ROOT_ENV] = str(Path("/custom/wt"))
        try:
            self.assertEqual(mod.default_worktree_root(), Path("/custom/wt"))
        finally:
            del os.environ[mod.WORKTREE_ROOT_ENV]


class WorktreeEnvTest(unittest.TestCase):
    def test_gocache_and_gotmp_isolated_into_worktree(self) -> None:
        wt = Path("/tmp/wt/fak-worker-wt-tools-abc")
        env = mod.worktree_env({"PATH": "/usr/bin"}, wt)
        # The build caches must live INSIDE the worktree — the "broken build in one
        # worktree does not red another" guarantee.
        self.assertTrue(env["GOCACHE"].startswith(str(wt)))
        self.assertTrue(env["GOTMPDIR"].startswith(str(wt)))
        # Workspace is repointed at the isolated tree, not the shared trunk.
        self.assertEqual(env["DISPATCH_WORKSPACE"], str(wt))
        self.assertEqual(env["FLEET_WORKER_WORKTREE"], str(wt))
        # The base env is preserved, not dropped.
        self.assertEqual(env["PATH"], "/usr/bin")

    def test_does_not_mutate_base_env(self) -> None:
        base = {"PATH": "/usr/bin"}
        mod.worktree_env(base, Path("/tmp/wt/fak-worker-wt-x-y"))
        self.assertNotIn("GOCACHE", base)


class ParseAndClassifyTest(unittest.TestCase):
    PORC = (
        "worktree C:/work/fak\nHEAD abc\nbranch refs/heads/main\n\n"
        "worktree C:/tmp/Fleet/worker-worktrees/fak-worker-wt-tools-deadbeef\n"
        "HEAD def\ndetached\n\n"
        "worktree C:/tmp/fak-selfupdate-build-1\nHEAD 123\ndetached\n"
    )

    def test_parse_paths(self) -> None:
        paths = mod.parse_worktree_paths(self.PORC)
        self.assertEqual(len(paths), 3)
        self.assertIn("C:/work/fak", paths)

    def test_is_worker_worktree_only_marker_dirs(self) -> None:
        self.assertTrue(mod.is_worker_worktree(
            "C:/tmp/Fleet/worker-worktrees/fak-worker-wt-tools-deadbeef"))
        self.assertFalse(mod.is_worker_worktree("C:/work/fak"))
        self.assertFalse(mod.is_worker_worktree("C:/tmp/fak-selfupdate-build-1"))

    def test_count_only_counts_our_worktrees(self) -> None:
        git = FakeGit(replies={"worktree": (0, self.PORC)})
        res = mod.count_worker_worktrees(Path("/r"), git=git)
        self.assertEqual(res["count"], 1)
        self.assertTrue(res["paths"][0].endswith("fak-worker-wt-tools-deadbeef"))


class PrepareTest(unittest.TestCase):
    def test_detached_add_at_trunk_head(self) -> None:
        git = FakeGit(replies={"rev-parse": (0, "feedface\n"),
                               "worktree": (0, "")})
        with tempfile.TemporaryDirectory() as d:
            res = mod.prepare_worker_worktree(Path("/r"), "tools", "1334",
                                              wt_root=Path(d), git=git)
        self.assertTrue(res["ok"])
        self.assertEqual(res["base_sha"], "feedface")
        # The add MUST be --detach at the resolved sha (the #1334 reconciliation):
        # not a branch, so git never refuses it and OFF_TRUNK never trips.
        add = [c for c in git.calls if c[:2] == ["worktree", "add"]][0]
        self.assertIn("--detach", add)
        self.assertEqual(add[-1], "feedface")

    def test_fail_open_when_head_unresolvable(self) -> None:
        git = FakeGit(replies={"rev-parse": (127, "")})
        res = mod.prepare_worker_worktree(Path("/r"), "tools", "1", git=git)
        self.assertFalse(res["ok"])
        # Fail-open: no worktree add was even attempted.
        self.assertFalse([c for c in git.calls if c[:2] == ["worktree", "add"]])

    def test_explicit_base_sha_skips_rev_parse(self) -> None:
        git = FakeGit(replies={"worktree": (0, "")})
        with tempfile.TemporaryDirectory() as d:
            res = mod.prepare_worker_worktree(Path("/r"), "tools", "1",
                                              base_sha="cafe", wt_root=Path(d),
                                              git=git)
        self.assertTrue(res["ok"])
        self.assertEqual(res["base_sha"], "cafe")
        self.assertFalse([c for c in git.calls if c and c[0] == "rev-parse"])

    def test_add_failure_is_fail_open_not_raise(self) -> None:
        git = FakeGit(replies={"rev-parse": (0, "abc\n"),
                               "worktree": (1, "fatal: already exists")})
        res = mod.prepare_worker_worktree(Path("/r"), "tools", "1",
                                          wt_root=Path("/wtroot"), git=git)
        self.assertFalse(res["ok"])
        self.assertIn("fail open", res["reason"])


class ReapTest(unittest.TestCase):
    def test_reaps_only_marker_worktree(self) -> None:
        git = FakeGit(replies={"worktree": (0, "")})
        res = mod.reap_worker_worktree(
            Path("/r"), "/wt/fak-worker-wt-tools-abc", git=git)
        self.assertTrue(res["ok"])
        remove = [c for c in git.calls if c[:2] == ["worktree", "remove"]][0]
        self.assertIn("--force", remove)
        # A prune follows so a half-removed admin record never lingers.
        self.assertTrue(any(c == ["worktree", "prune"] for c in git.calls))

    def test_refuses_non_worker_worktree(self) -> None:
        git = FakeGit()
        res = mod.reap_worker_worktree(Path("/r"), "C:/work/fak", git=git)
        self.assertFalse(res["ok"])
        self.assertEqual(git.calls, [])  # never touched git


class LandTest(unittest.TestCase):
    def test_lands_diff_onto_trunk_by_path_signed_off(self) -> None:
        git = FakeGit(replies={
            "diff": (0, "diff --git a/x b/x\n@@\n-old\n+new\n"),
            "apply": (0, ""),
            "commit": (0, "[main abc] msg"),
        })
        res = mod.land_worktree_diff(
            Path("/trunk"), "/wt/fak-worker-wt-tools-abc",
            commit_msg_file="/tmp/msg.txt", paths=["x"], git=git)
        self.assertTrue(res["ok"])
        self.assertTrue(res["committed"])
        commit = [c for c in git.calls if c and c[0] == "commit"][0]
        # Signed-off (DCO) and scoped by explicit path — never add -A of the trunk.
        self.assertIn("-s", commit)
        self.assertIn("--", commit)
        self.assertIn("x", commit)

    def test_empty_diff_is_ok_but_not_committed(self) -> None:
        git = FakeGit(replies={"diff": (0, "   \n")})
        res = mod.land_worktree_diff(
            Path("/trunk"), "/wt/fak-worker-wt-tools-abc",
            commit_msg_file="/tmp/msg.txt", git=git)
        self.assertTrue(res["ok"])
        self.assertFalse(res["committed"])
        # No apply / commit attempted on an empty diff.
        self.assertFalse([c for c in git.calls if c and c[0] in ("apply", "commit")])

    def test_apply_failure_does_not_commit(self) -> None:
        git = FakeGit(replies={
            "diff": (0, "diff --git a/x b/x\n@@\n-o\n+n\n"),
            "apply": (1, "error: patch does not apply"),
        })
        res = mod.land_worktree_diff(
            Path("/trunk"), "/wt/fak-worker-wt-tools-abc",
            commit_msg_file="/tmp/msg.txt", git=git)
        self.assertFalse(res["ok"])
        self.assertFalse(res["committed"])
        self.assertFalse([c for c in git.calls if c and c[0] == "commit"])

    def test_base_sha_is_the_diff_ref_not_head(self) -> None:
        # A ship-when-green worker commits IN its worktree, so diff-vs-HEAD is empty
        # and the work would evaporate; diffing against the pinned base captures it.
        git = FakeGit(replies={
            "diff": (0, "diff --git a/x b/x\n@@\n-old\n+new\n"),
            "apply": (0, ""),
            "commit": (0, "[main abc] msg"),
        })
        res = mod.land_worktree_diff(
            Path("/trunk"), "/wt/fak-worker-wt-tools-abc",
            commit_msg_file="/tmp/msg.txt", base_sha="feedface", paths=["x"], git=git)
        self.assertTrue(res["committed"])
        diff = [c for c in git.calls if c and c[0] == "diff"][0]
        self.assertIn("feedface", diff)  # diff base is the pinned sha, not HEAD
        self.assertNotIn("HEAD", diff)

    def test_verify_failure_refuses_to_land(self) -> None:
        git = FakeGit(replies={
            "diff": (0, "diff --git a/x b/x\n@@\n-o\n+n\n"),
            "apply": (0, ""),
            "commit": (0, "[main abc] msg"),
        })
        called = {}

        def verify(wt):
            called["wt"] = str(wt)
            return {"ok": False, "detail": "go build ./... : x.go:1: broke"}

        res = mod.land_worktree_diff(
            Path("/trunk"), "/wt/fak-worker-wt-tools-abc",
            commit_msg_file="/tmp/msg.txt", base_sha="cafe", verify=verify, git=git)
        self.assertFalse(res["ok"])
        self.assertFalse(res["committed"])
        # A failed witness runs BEFORE any apply/commit — nothing lands on the trunk.
        self.assertFalse([c for c in git.calls if c and c[0] in ("apply", "commit")])
        self.assertIn("wt", called)

    def test_verify_pass_lands(self) -> None:
        git = FakeGit(replies={
            "diff": (0, "diff --git a/x b/x\n@@\n-o\n+n\n"),
            "apply": (0, ""),
            "commit": (0, "[main abc] msg"),
        })
        res = mod.land_worktree_diff(
            Path("/trunk"), "/wt/fak-worker-wt-tools-abc",
            commit_msg_file="/tmp/msg.txt", base_sha="cafe",
            verify=lambda wt: {"ok": True, "detail": "clean"}, git=git)
        self.assertTrue(res["committed"])


class CLITest(unittest.TestCase):
    """The runnable entrypoint: assert `main(argv)` maps each subcommand onto the
    right primitive with the right kwargs, prints ONE JSON object, and returns an
    exit code that mirrors the primitive's ok — all without touching git (the
    primitives are stubbed and record what they were handed)."""

    def _run(self, argv, stubs):
        import contextlib
        import io
        import json
        saved = {name: getattr(mod, name) for name in stubs}
        for name, fn in stubs.items():
            setattr(mod, name, fn)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                code = mod.main(argv)
        finally:
            for name, fn in saved.items():
                setattr(mod, name, fn)
        out = buf.getvalue().strip()
        return code, (json.loads(out) if out else None)

    def test_prepare_dispatches_and_attaches_env(self) -> None:
        seen = {}

        def fake_prepare(root, lane, key, *, base_sha=None, wt_root=None):
            seen.update(root=str(root), lane=lane, key=key, base_sha=base_sha)
            return {"ok": True, "path": "/wt/fak-worker-wt-cmd-abc", "base_sha": "feed"}

        code, res = self._run(
            ["--root", "/r", "prepare", "--lane", "cmd", "--key", "1338"],
            {"prepare_worker_worktree": fake_prepare})
        self.assertEqual(code, 0)
        self.assertEqual(seen["lane"], "cmd")
        self.assertEqual(seen["key"], "1338")
        self.assertIsNone(seen["base_sha"])  # empty --base-sha becomes None
        # prepare hands the caller the build-isolating env additions (compared via
        # str(Path(...)) so the assertion is OS-separator-agnostic).
        wt = str(Path("/wt/fak-worker-wt-cmd-abc"))
        self.assertTrue(res["env"]["GOCACHE"].startswith(wt))
        self.assertEqual(res["env"]["FLEET_WORKER_WORKTREE"], wt)

    def test_prepare_failure_is_nonzero_exit(self) -> None:
        def fake_prepare(root, lane, key, *, base_sha=None, wt_root=None):
            return {"ok": False, "path": None, "reason": "git error — fail open"}

        code, res = self._run(
            ["--root", "/r", "prepare", "--lane", "cmd", "--key", "1"],
            {"prepare_worker_worktree": fake_prepare})
        self.assertEqual(code, 1)
        self.assertNotIn("env", res)  # no env attached when prepare failed

    def test_land_passes_msg_file_and_scoped_paths(self) -> None:
        seen = {}

        def fake_land(root, wt, *, commit_msg_file, base_sha=None, paths=None, verify=None):
            seen.update(root=str(root), wt=wt, msg=str(commit_msg_file), paths=paths,
                        base_sha=base_sha, verify=verify)
            return {"ok": True, "applied": True, "committed": True}

        code, res = self._run(
            ["--root", "/r", "land", "--worktree", "/wt/fak-worker-wt-cmd-abc",
             "--msg-file", "/tmp/msg.txt", "--base-sha", "feedface", "--verify", "go-build",
             "--path", "cmd/fak/x.go", "--path", "cmd/fak/y.go"],
            {"land_worktree_diff": fake_land})
        self.assertEqual(code, 0)
        self.assertEqual(seen["msg"], "/tmp/msg.txt")
        self.assertEqual(seen["paths"], ["cmd/fak/x.go", "cmd/fak/y.go"])
        self.assertEqual(seen["base_sha"], "feedface")  # base sha threaded to the primitive
        self.assertIs(seen["verify"], mod._go_build_verify)  # --verify go-build wires the witness
        self.assertTrue(res["committed"])

    def test_land_without_paths_sends_none(self) -> None:
        seen = {}

        def fake_land(root, wt, *, commit_msg_file, base_sha=None, paths=None, verify=None):
            seen["paths"] = paths
            seen["base_sha"] = base_sha
            seen["verify"] = verify
            return {"ok": True, "applied": True, "committed": True}

        code, _ = self._run(
            ["--root", "/r", "land", "--worktree", "/wt/fak-worker-wt-cmd-abc",
             "--msg-file", "/tmp/msg.txt"],
            {"land_worktree_diff": fake_land})
        self.assertEqual(code, 0)
        self.assertIsNone(seen["paths"])  # no --path -> whole diff (None), not []
        self.assertIsNone(seen["base_sha"])  # no --base-sha -> None (falls back to HEAD)
        self.assertIsNone(seen["verify"])  # default --verify off -> no witness

    def test_reap_refusal_is_nonzero_exit(self) -> None:
        def fake_reap(root, wt):
            return {"ok": False, "removed": False,
                    "reason": "refusing to reap a non-worker worktree"}

        code, res = self._run(
            ["--root", "/r", "reap", "--worktree", "/work/fak"],
            {"reap_worker_worktree": fake_reap})
        self.assertEqual(code, 1)
        self.assertFalse(res["removed"])

    def test_list_ok_when_no_error_key(self) -> None:
        code, res = self._run(
            ["--root", "/r", "list"],
            {"count_worker_worktrees": lambda root: {"count": 2, "paths": ["a", "b"]}})
        self.assertEqual(code, 0)
        self.assertEqual(res["count"], 2)

    def test_list_error_is_nonzero_exit(self) -> None:
        code, _ = self._run(
            ["--root", "/r", "list"],
            {"count_worker_worktrees": lambda root: {"count": 0, "paths": [],
                                                     "error": "not a git repo"}})
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
