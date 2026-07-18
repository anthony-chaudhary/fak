#!/usr/bin/env python3
"""Hermetic tests for tools/skill_lifecycle.py.

The lifecycle tracker is a fold over a filesystem of skill dirs plus a JSON
sidecar. These tests build a throwaway ``.claude/skills`` tree in a tempdir and
exercise the real behaviours an operator depends on: record bumps a typed
counter and timestamps it, the sidecar round-trips, staleness is computed from
an injected ``now``, sweep only proposes agent-origin unpinned stale skills, and
archive is a reversible RENAME (never a delete) that a restore brings back.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "skill_lifecycle.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("skill_lifecycle", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load()


def make_root(names):
    d = Path(tempfile.mkdtemp())
    for n in names:
        (d / n).mkdir()
    return d


class TestNowIso(unittest.TestCase):
    def test_passthrough_when_supplied(self):
        self.assertEqual(MOD.now_iso("2026-01-01T00:00:00Z"), "2026-01-01T00:00:00Z")

    def test_generates_utc_z_stamp_when_absent(self):
        s = MOD.now_iso(None)
        self.assertTrue(s.endswith("Z"))
        self.assertEqual(len(s), 20)  # YYYY-MM-DDTHH:MM:SSZ


class TestSkillDirs(unittest.TestCase):
    def test_lists_dirs_excluding_archive_and_dotfiles(self):
        root = make_root(["alpha", "beta"])
        (root / MOD.ARCHIVE_DIR).mkdir()
        (root / ".hidden").mkdir()
        self.assertEqual(MOD.skill_dirs(root), ["alpha", "beta"])


class TestRecordAndSidecar(unittest.TestCase):
    def test_record_bumps_counter_and_persists(self):
        root = make_root(["deploy"])
        r1 = MOD.cmd_record(root, "deploy", "use", "2026-01-01T00:00:00Z")
        self.assertTrue(r1["ok"])
        self.assertEqual(r1["use_count"], 1)
        r2 = MOD.cmd_record(root, "deploy", "use", "2026-01-02T00:00:00Z")
        self.assertEqual(r2["use_count"], 2)
        self.assertEqual(r2["last_activity_at"], "2026-01-02T00:00:00Z")
        # persisted to sidecar on disk
        doc = json.loads((root / MOD.SIDECAR).read_text(encoding="utf-8"))
        self.assertEqual(doc["skills"]["deploy"]["use_count"], 2)

    def test_record_unknown_skill_errors(self):
        root = make_root(["deploy"])
        r = MOD.cmd_record(root, "ghost", "use", None)
        self.assertFalse(r["ok"])
        self.assertIn("unknown skill", r["error"])


class TestStale(unittest.TestCase):
    def test_never_recorded_is_stale(self):
        self.assertTrue(MOD.stale({"last_activity_at": None}, 30, "2026-01-01T00:00:00Z"))

    def test_recent_is_not_stale(self):
        e = {"last_activity_at": "2026-01-20T00:00:00Z"}
        self.assertFalse(MOD.stale(e, 30, "2026-01-25T00:00:00Z"))

    def test_old_is_stale(self):
        e = {"last_activity_at": "2026-01-01T00:00:00Z"}
        self.assertTrue(MOD.stale(e, 30, "2026-03-01T00:00:00Z"))


class TestSweep(unittest.TestCase):
    def test_only_agent_origin_stale_unpinned_proposed(self):
        root = make_root(["fresh", "old_agent", "pinned_old", "bundled_old"])
        doc = {"schema": MOD.SCHEMA, "skills": {
            "fresh": {"origin": "agent", "last_activity_at": "2026-02-25T00:00:00Z",
                      "pinned": False},
            "old_agent": {"origin": "agent", "last_activity_at": "2026-01-01T00:00:00Z",
                          "pinned": False},
            "pinned_old": {"origin": "agent", "last_activity_at": "2026-01-01T00:00:00Z",
                           "pinned": True},
            "bundled_old": {"origin": "bundled", "last_activity_at": "2026-01-01T00:00:00Z",
                            "pinned": False},
        }}
        MOD.save_sidecar(root, doc)
        out = MOD.cmd_sweep(root, 30, live=False, now="2026-03-01T00:00:00Z")
        self.assertEqual(set(out["proposed"]), {"old_agent"})
        exempt = {e["skill"] for e in out["exempt"]}
        self.assertEqual(exempt, {"fresh", "pinned_old", "bundled_old"})


class TestArchiveRestoreIsReversible(unittest.TestCase):
    def test_archive_then_restore_round_trips(self):
        root = make_root(["scratch"])
        (root / "scratch" / "SKILL.md").write_text("hi", encoding="utf-8")
        arch = MOD.transition(root, "scratch", "archive", "idle", "2026-01-01T00:00:00Z")
        self.assertTrue(arch["ok"], arch)
        self.assertFalse((root / "scratch").exists())  # moved, not present as active
        self.assertTrue((root / MOD.ARCHIVE_DIR / "scratch" / "SKILL.md").exists())
        # journal recorded the move
        journal = (root / MOD.JOURNAL).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(json.loads(journal[-1])["action"], "archive")
        # restore brings the content back exactly
        res = MOD.transition(root, "scratch", "restore", "wanted", "2026-01-02T00:00:00Z")
        self.assertTrue(res["ok"], res)
        self.assertEqual((root / "scratch" / "SKILL.md").read_text(encoding="utf-8"), "hi")

    def test_archive_pinned_refused(self):
        root = make_root(["keep"])
        doc = {"schema": MOD.SCHEMA, "skills": {"keep": {"pinned": True}}}
        MOD.save_sidecar(root, doc)
        r = MOD.transition(root, "keep", "archive", "idle", None)
        self.assertFalse(r["ok"])
        self.assertIn("pinned", r["error"])
        self.assertTrue((root / "keep").is_dir())  # untouched


if __name__ == "__main__":
    unittest.main()
