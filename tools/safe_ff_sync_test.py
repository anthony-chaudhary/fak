#!/usr/bin/env python3
"""Hermetic tests for safe_ff_sync.py.

Each test builds a throwaway origin repo + a clone, drives the clone into one of
the shared-worktree states the tool must distinguish, and asserts refuse-or-apply
behavior. Pure stdlib + the `git` CLI; no network, no host git config (isolated via
GIT_CONFIG_GLOBAL/SYSTEM = devnull and explicit author/committer identity), so it
runs natively on this host (the WDAC block is on Go test binaries, not python/git).

Run:  python tools/safe_ff_sync_test.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_ff_sync as sfs  # noqa: E402

_SAVED = {}
_ISO = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
}


def setUpModule():
    for k, v in _ISO.items():
        _SAVED[k] = os.environ.get(k)
        os.environ[k] = v


def tearDownModule():
    for k, v in _SAVED.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def git(args, cwd):
    subprocess.run(["git", *args], cwd=str(cwd),
                   capture_output=True, text=True, check=True)


def write(p, text):
    Path(p).write_text(text, encoding="utf-8")


class SafeFfSyncTest(unittest.TestCase):
    def _tmp(self):
        tmp = Path(tempfile.mkdtemp(prefix="ffsync_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return tmp

    def _origin_with_c1(self, tmp):
        origin = tmp / "origin"
        origin.mkdir()
        git(["init", "-b", "work"], origin)
        write(origin / "a.txt", "v1\n")
        write(origin / "keep.txt", "keep\n")
        git(["add", "."], origin)
        git(["commit", "-m", "c1"], origin)
        return origin

    def _behind_clone(self):
        """clone: HEAD at c1, origin/work at c2 (a.txt v1->v2, +new.txt). Behind 1."""
        tmp = self._tmp()
        origin = self._origin_with_c1(tmp)
        clone = tmp / "clone"
        git(["clone", str(origin), str(clone)], tmp)
        write(origin / "a.txt", "v2\n")
        write(origin / "new.txt", "n1\n")
        git(["add", "."], origin)
        git(["commit", "-m", "c2"], origin)
        git(["fetch", "origin"], clone)
        return clone

    # --- the states the tool must distinguish -------------------------------

    def test_behind_dirty_identical_is_safe_and_applies(self):
        clone = self._behind_clone()
        write(clone / "a.txt", "v2\n")      # M, identical to target
        write(clone / "new.txt", "n1\n")    # A, identical to target
        write(clone / "mine.txt", "local\n")  # unrelated dirty work, NOT in ff set

        info = sfs.assess(str(clone), "origin", "work", do_fetch=False)
        self.assertEqual(info["state"], "behind")
        self.assertTrue(info["ok"])
        self.assertEqual(info["divergent"], [])
        self.assertEqual(info["write_count"], 2)

        new_head = sfs.apply_ff(str(clone), "origin", "work", info)
        self.assertEqual(new_head, sfs.rev(str(clone), "origin/work"))
        self.assertEqual((clone / "a.txt").read_text(), "v2\n")
        self.assertEqual((clone / "new.txt").read_text(), "n1\n")
        self.assertTrue((clone / "mine.txt").exists(),
                        "unrelated novel work must survive the ff")

    def test_behind_dirty_divergent_tracked_refuses(self):
        clone = self._behind_clone()
        write(clone / "a.txt", "LOCAL EDIT\n")  # diverges from both HEAD and target

        info = sfs.assess(str(clone), "origin", "work", do_fetch=False)
        self.assertFalse(info["ok"])
        self.assertTrue(any(e["path"] == "a.txt" for e in info["divergent"]))

        head_before = sfs.rev(str(clone), "HEAD")
        rc = sfs.main(["apply", "--repo", str(clone)])
        self.assertEqual(rc, sfs.EXIT_UNSAFE)
        self.assertEqual(sfs.rev(str(clone), "HEAD"), head_before,
                         "a refused apply must not move HEAD")
        self.assertEqual((clone / "a.txt").read_text(), "LOCAL EDIT\n",
                         "a refused apply must not touch the worktree")

    def test_behind_divergent_untracked_addition_refuses(self):
        clone = self._behind_clone()
        write(clone / "a.txt", "v2\n")          # M identical (safe on its own)
        write(clone / "new.txt", "DIFFERENT\n")  # A but content diverges -> unsafe

        info = sfs.assess(str(clone), "origin", "work", do_fetch=False)
        self.assertFalse(info["ok"])
        self.assertEqual([e["path"] for e in info["divergent"]], ["new.txt"])

    def test_in_sync_is_ok_noop(self):
        clone = self._behind_clone()
        git(["merge", "--ff-only", "origin/work"], clone)  # catch up cleanly
        info = sfs.assess(str(clone), "origin", "work", do_fetch=False)
        self.assertEqual(info["state"], "in-sync")
        self.assertEqual(sfs.main(["check", "--repo", str(clone)]), sfs.EXIT_OK)

    def test_diverged_refuses(self):
        clone = self._behind_clone()
        write(clone / "b.txt", "local commit\n")
        git(["add", "b.txt"], clone)
        git(["commit", "-m", "local"], clone)  # now neither is ancestor of the other
        info = sfs.assess(str(clone), "origin", "work", do_fetch=False)
        self.assertEqual(info["state"], "diverged")
        self.assertFalse(info["ok"])
        self.assertEqual(sfs.main(["check", "--repo", str(clone)]), sfs.EXIT_UNSAFE)

    def test_ahead_reports_ahead(self):
        tmp = self._tmp()
        origin = self._origin_with_c1(tmp)
        clone = tmp / "clone"
        git(["clone", str(origin), str(clone)], tmp)
        write(clone / "c.txt", "ahead\n")
        git(["add", "c.txt"], clone)
        git(["commit", "-m", "local-ahead"], clone)  # clone ahead, origin unchanged
        info = sfs.assess(str(clone), "origin", "work", do_fetch=False)
        self.assertEqual(info["state"], "ahead")
        self.assertFalse(info["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
