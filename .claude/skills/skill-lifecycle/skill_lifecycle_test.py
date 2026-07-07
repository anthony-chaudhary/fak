#!/usr/bin/env python3
"""Witness tests for skill_lifecycle.py (#2941).

The issue's named witness: a test proving a pinned skill is never auto-archived
while an idle unpinned one is — plus the never-delete, restorable, journaled,
value-guards-idle, and bundled-exempt invariants around it.

Run: python .claude/skills/skill-lifecycle/skill_lifecycle_test.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skill_lifecycle as sl  # noqa: E402

NOW = "2026-07-07T12:00:00Z"
LONG_AGO = "2026-01-01T00:00:00Z"  # ~187 days before NOW
YESTERDAY = "2026-07-06T12:00:00Z"


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="skill-lifecycle-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = sl.Store(self.tmp)

    def make_skill(self, name: str, pin_frontmatter: bool = False) -> Path:
        d = self.tmp / name
        d.mkdir()
        fm = "---\nname: %s\ndescription: test skill\n%s---\n\nbody of %s\n" % (
            name, "pin: true\n" if pin_frontmatter else "", name)
        (d / "SKILL.md").write_text(fm, encoding="utf-8")
        return d

    def seed(self, name: str, *, origin="agent", last=LONG_AGO, use=0, patch=0, pinned=False):
        data = self.store.load()
        rec = self.store.entry(data, name)
        rec.update(origin=origin, last_activity_at=last, use_count=use,
                   patch_count=patch, pinned=pinned)
        self.store.save(data)

    def sweep(self, apply=True, idle_days=30.0, min_value=1):
        now = sl.parse_ts(NOW)
        verdicts = sl.sweep_verdicts(self.store, now, idle_days, min_value)
        archived = []
        if apply:
            for v in verdicts:
                if v["verdict"] == "archive":
                    archived.append(sl.do_archive(self.store, v["skill"],
                                                  "auto-archive: " + v["reason"], now, by="sweep"))
        return verdicts, archived

    def journal_entries(self):
        if not self.store.journal_path.exists():
            return []
        return [json.loads(line) for line in
                self.store.journal_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # -- the issue's named witness ---------------------------------------

    def test_pinned_exempt_while_idle_unpinned_archived(self):
        self.make_skill("pinned-idle", pin_frontmatter=True)
        self.make_skill("unpinned-idle")
        self.seed("pinned-idle", last=LONG_AGO)
        self.seed("unpinned-idle", last=LONG_AGO)

        verdicts, archived = self.sweep()

        by_name = {v["skill"]: v for v in verdicts}
        self.assertEqual(by_name["pinned-idle"]["verdict"], "keep")
        self.assertIn("pinned", by_name["pinned-idle"]["reason"])
        self.assertEqual(by_name["unpinned-idle"]["verdict"], "archive")

        self.assertTrue((self.tmp / "pinned-idle" / "SKILL.md").exists())
        self.assertFalse((self.tmp / "unpinned-idle").exists())
        self.assertTrue((self.store.archived_dir("unpinned-idle") / "SKILL.md").exists())
        self.assertEqual([e["skill"] for e in archived], ["unpinned-idle"])

    def test_sidecar_pin_also_exempts(self):
        self.make_skill("sidecar-pinned")
        self.seed("sidecar-pinned", pinned=True, last=LONG_AGO)
        verdicts, _ = self.sweep()
        self.assertEqual(verdicts[0]["verdict"], "keep")
        self.assertTrue((self.tmp / "sidecar-pinned").exists())

    def test_manual_archive_refuses_pinned(self):
        self.make_skill("precious", pin_frontmatter=True)
        with self.assertRaises(sl.Refusal) as cm:
            sl.do_archive(self.store, "precious", "why not", sl.parse_ts(NOW), by="test")
        self.assertEqual(cm.exception.reason_class, "SKILL_PINNED")
        self.assertTrue((self.tmp / "precious" / "SKILL.md").exists())

    # -- never delete, journaled, reversible ------------------------------

    def test_archive_never_deletes_and_restore_reverses(self):
        self.make_skill("roundtrip")
        original = (self.tmp / "roundtrip" / "SKILL.md").read_bytes()
        self.seed("roundtrip", last=LONG_AGO)

        _, archived = self.sweep()
        self.assertEqual(len(archived), 1)
        arch_md = self.store.archived_dir("roundtrip") / "SKILL.md"
        self.assertEqual(arch_md.read_bytes(), original, "archive must preserve bytes")
        self.assertTrue(Path(archived[0]["snapshot"]).exists(), "pre-archive snapshot must exist")

        event = sl.do_restore(self.store, "roundtrip", sl.parse_ts(NOW), by="test")
        self.assertEqual((self.tmp / "roundtrip" / "SKILL.md").read_bytes(), original)
        self.assertFalse(self.store.archived_dir("roundtrip").exists())

        entries = self.journal_entries()
        self.assertEqual([e["action"] for e in entries], ["archive", "restore"])
        self.assertEqual(entries[0]["skill"], "roundtrip")
        self.assertEqual(entries[1]["to"], event["to"])
        data = self.store.load()
        self.assertEqual(data["skills"]["roundtrip"]["state"], "live")

    def test_archive_refuses_collision_instead_of_overwriting(self):
        self.make_skill("dup")
        self.seed("dup", last=LONG_AGO)
        prior = self.store.archived_dir("dup")
        prior.mkdir(parents=True)
        (prior / "SKILL.md").write_text("prior archived bytes", encoding="utf-8")
        with self.assertRaises(sl.Refusal) as cm:
            sl.do_archive(self.store, "dup", "collide", sl.parse_ts(NOW), by="test")
        self.assertEqual(cm.exception.reason_class, "ARCHIVE_COLLISION")
        self.assertEqual((prior / "SKILL.md").read_text(encoding="utf-8"), "prior archived bytes")
        self.assertTrue((self.tmp / "dup" / "SKILL.md").exists())

    # -- exemptions and the value guard -----------------------------------

    def test_bundled_skill_never_auto_archived(self):
        self.make_skill("core-thing")
        self.seed("core-thing", origin="bundled", last=LONG_AGO)
        verdicts, archived = self.sweep()
        self.assertEqual(verdicts[0]["verdict"], "keep")
        self.assertIn("bundled", verdicts[0]["reason"])
        self.assertEqual(archived, [])

    def test_unknown_origin_fails_safe_to_bundled(self):
        # No sidecar entry and the temp root is not a git work tree: origin
        # cannot be witnessed, so the skill must be kept.
        self.make_skill("mystery")
        verdicts, archived = self.sweep()
        self.assertEqual(verdicts[0]["verdict"], "keep")
        self.assertEqual(archived, [])

    def test_high_value_idle_skill_kept(self):
        # The emergency-only case: long idle but with witnessed value.
        self.make_skill("fire-extinguisher")
        self.seed("fire-extinguisher", last=LONG_AGO, use=3)
        verdicts, archived = self.sweep(min_value=1)
        self.assertEqual(verdicts[0]["verdict"], "keep")
        self.assertIn("value", verdicts[0]["reason"])
        self.assertEqual(archived, [])

    def test_recently_active_skill_kept(self):
        self.make_skill("busy")
        self.seed("busy", last=YESTERDAY)
        verdicts, archived = self.sweep()
        self.assertEqual(verdicts[0]["verdict"], "keep")
        self.assertEqual(archived, [])

    def test_dry_run_moves_nothing(self):
        self.make_skill("stale")
        self.seed("stale", last=LONG_AGO)
        verdicts, archived = self.sweep(apply=False)
        self.assertEqual(verdicts[0]["verdict"], "archive")
        self.assertEqual(archived, [])
        self.assertTrue((self.tmp / "stale" / "SKILL.md").exists())
        self.assertEqual(self.journal_entries(), [])

    # -- telemetry ---------------------------------------------------------

    def test_record_updates_sidecar(self):
        self.make_skill("counted")
        rc = sl.main(["--skills-root", str(self.tmp), "--now", NOW, "record", "counted", "--kind", "use"])
        self.assertEqual(rc, 0)
        rec = self.store.load()["skills"]["counted"]
        self.assertEqual(rec["use_count"], 1)
        self.assertEqual(rec["last_activity_at"], NOW)

    def test_record_unknown_skill_refused(self):
        rc = sl.main(["--skills-root", str(self.tmp), "record", "ghost"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
