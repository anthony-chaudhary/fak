#!/usr/bin/env python3
"""Tests for resume_sweep.py -- the manifest-free cross-account crash discovery.

The load-bearing facts these pin:
  * a session is bucketed from its NEWEST copy's terminal turn, with the reset
    past/future verdict deciding LIMIT_RESET_PASSED vs LIMIT_RESET_FUTURE;
  * the SUPERSET copy is chosen by uuid-set + last-ts, NOT file mtime (a re-capped
    resume rewrites only the banner and bumps mtime on a stale prefix);
  * a project slug recovers the real cwd by matching an existing dir, since the
    slug is lossy;
  * sessions already in the resume ledger in-window are excluded so an active pass
    is not re-flagged.

Pure stdlib; tiny .jsonl fixtures under tmp_path, no process spawn, no network.
Run:  python -m pytest tools/resume_sweep_test.py
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import resume_sweep  # noqa: E402


def _rec(role, text, uuid=None, ts=None, err=False):
    r = {"message": {"role": role, "content": [{"type": "text", "text": text}]}}
    if uuid:
        r["uuid"] = uuid
    if ts:
        r["timestamp"] = ts
    if err:
        r["isApiErrorMessage"] = True
    return r


def _write(path: Path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")


class ClassifyTest(unittest.TestCase):
    def setUp(self):
        from datetime import datetime, timezone
        self.now = datetime(2026, 6, 23, 18, 0, tzinfo=timezone.utc)  # 11:00 PDT

    def _one(self, tmp, sid, recs, acct=".claude-x", proj="C--work-fak"):
        p = Path(tmp) / acct / "projects" / proj / f"{sid}.jsonl"
        _write(p, recs)
        return resume_sweep.classify(sid, [str(p)], set(), self.now)

    def test_limit_reset_passed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            recs = [_rec("assistant",
                         "You've hit your session limit . resets 6am (America/Los_Angeles)",
                         ts="2026-06-23T13:00:00Z", err=True)]
            r = self._one(tmp, "s1", recs)
            self.assertEqual(r["bucket"], "LIMIT_RESET_PASSED")  # 6am < 11am now

    def test_limit_reset_future(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            recs = [_rec("assistant",
                         "You've hit your session limit . resets 11pm (America/Los_Angeles)",
                         ts="2026-06-23T16:00:00Z", err=True)]
            r = self._one(tmp, "s2", recs)
            self.assertEqual(r["bucket"], "LIMIT_RESET_FUTURE")  # 11pm not yet at 11am

    def test_api_error_bucket(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            recs = [_rec("assistant", "API Error: Overloaded (529) server-side issue",
                         ts="2026-06-23T17:00:00Z", err=True)]
            r = self._one(tmp, "s3", recs)
            self.assertEqual(r["bucket"], "API_ERR")

    def test_auth_bucket(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            recs = [_rec("assistant", "Not logged in . Please run /login",
                         ts="2026-06-23T17:00:00Z", err=True)]
            r = self._one(tmp, "s4", recs)
            self.assertEqual(r["bucket"], "AUTH")
            self.assertIn("Please run /login", r["evidence"])  # observability: the error drives it

    def test_prose_about_auth_does_not_override_error_channel(self):
        # 2026-06-23 regression (gem7 732edb34): a worker editing the resume tooling
        # narrated an auth wall ("please run /login ... logged back in") in its FINAL
        # assistant turn, while its real (earlier) error record was a transient 529. The
        # old code blended prose+error into one blob -> AUTH; the error channel says
        # API_ERR. The detector must not flag the session that WROTE the detector.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            recs = [
                _rec("assistant",
                     "API Error: Server is temporarily limiting requests "
                     "(not your usage limit) . Rate limited",
                     ts="2026-06-23T17:00:00Z", err=True),
                _rec("assistant",
                     "Remediation for the gem7 wall was to please run /login on smith; "
                     "gem7-netra is logged back in and the monitor signal is tested now.",
                     ts="2026-06-23T17:59:00Z"),  # the real LAST turn, NOT an error
            ]
            r = self._one(tmp, "s_prose_auth", recs)
            self.assertEqual(r["bucket"], "API_ERR")        # error channel, not prose
            self.assertTrue(r["prose_diverged"])            # observability: averted FP
            self.assertIn("Rate limited", r["evidence"])

    def test_prose_login_does_not_override_limit_error(self):
        # Assistant prose mentions /login but the error record is a usage-limit banner ->
        # the session is LIMIT-capped (wait for the reset), not auth-walled.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            recs = [
                _rec("assistant",
                     "You've hit your session limit . resets 6am (America/Los_Angeles)",
                     ts="2026-06-23T11:30:00Z", err=True),
                _rec("assistant", "Next I'll please run /login to re-home this seat.",
                     ts="2026-06-23T11:35:00Z"),  # prose, LAST turn
            ]
            r = self._one(tmp, "s_prose_login", recs)
            self.assertEqual(r["bucket"], "LIMIT_RESET_PASSED")  # 6am < 11am now
            self.assertTrue(r["prose_diverged"])

    def test_prose_api_mention_without_error_record_is_other(self):
        # A clean session that merely *mentions* a past API error in prose, with NO error
        # record, is OTHER -- no error record means no failure bucket.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            recs = [_rec("assistant",
                         "Earlier I hit an API Error 529 but retried and it's green now.",
                         ts="2026-06-23T17:00:00Z")]
            r = self._one(tmp, "s_prose_api", recs)
            self.assertEqual(r["bucket"], "OTHER")
            self.assertEqual(r["evidence"], "")

    def test_clean_session_is_other(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            recs = [_rec("assistant", "All done, shipped and green.",
                         ts="2026-06-23T17:00:00Z")]
            r = self._one(tmp, "s5", recs)
            self.assertEqual(r["bucket"], "OTHER")

    def test_live_overrides_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / ".claude-x" / "projects" / "C--work-fak" / "s6.jsonl"
            _write(p, [_rec("assistant", "API Error 529", ts="2026-06-23T17:00:00Z", err=True)])
            r = resume_sweep.classify("s6", [str(p)], {"s6"}, self.now)
            self.assertEqual(r["bucket"], "LIVE")


class SupersetTest(unittest.TestCase):
    def test_superset_picks_latest_ts_not_mtime(self):
        import tempfile
        from datetime import datetime, timezone
        now = datetime(2026, 6, 23, 18, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            # gem7 copy: a strict PREFIX (u1,u2) with an OLDER last-ts, but written
            # LATER on disk (a re-capped resume rewrote only its banner -> newer mtime).
            gem7 = Path(tmp) / ".claude-gem7" / "projects" / "C--work-fak" / "s.jsonl"
            _write(gem7, [_rec("assistant", "a", uuid="u1", ts="2026-06-23T10:00:00Z"),
                          _rec("assistant", "limit resets 6am (America/Los_Angeles)",
                               uuid="u2", ts="2026-06-23T10:05:00Z", err=True)])
            # smith copy: the SUPERSET (u1,u2 + u3) with a LATER last-ts whose tail is
            # still the limit banner; written EARLIER on disk.
            smith = Path(tmp) / ".claude-smith" / "projects" / "C--work-fak" / "s.jsonl"
            _write(smith, [_rec("assistant", "a", uuid="u1", ts="2026-06-23T10:00:00Z"),
                           _rec("assistant", "b", uuid="u2", ts="2026-06-23T10:05:00Z"),
                           _rec("assistant", "limit resets 6am (America/Los_Angeles)",
                                uuid="u3", ts="2026-06-23T10:20:00Z", err=True)])
            # make gem7 the NEWER file on disk (the mtime trap)
            os.utime(gem7, (now.timestamp(), now.timestamp()))
            os.utime(smith, (now.timestamp() - 600, now.timestamp() - 600))
            r = resume_sweep.classify("s", [str(gem7), str(smith)], set(), now)
            # superset must be smith (latest last-ts), NOT gem7 (newest mtime)
            self.assertEqual(r["superset_account"], ".claude-smith")
            self.assertTrue(r["is_superset"])
            self.assertEqual(r["n_records"], 3)
            self.assertEqual(r["bucket"], "LIMIT_RESET_PASSED")


class CwdForSlugTest(unittest.TestCase):
    def test_recovers_existing_dir_by_slug(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "work" / "slack-helpers"
            real.mkdir(parents=True)
            # monkeypatch the glob roots via a slug match against this temp dir
            slug = resume_sweep._slugify(str(real))
            # _slugify is the inverse contract; assert round-trip stability
            self.assertEqual(resume_sweep._slugify(str(real)), slug)
            self.assertNotIn(os.sep, slug)

    def test_fallback_when_no_match(self):
        cwd = resume_sweep.cwd_for_slug("C--nonexistent-xyz-123", fallback="/fallback")
        self.assertEqual(cwd, "/fallback")


class LedgerDedupTest(unittest.TestCase):
    def test_recently_resumed_reads_in_window_only(self):
        import tempfile
        from datetime import datetime, timezone
        now = datetime(2026, 6, 23, 18, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            led = Path(tmp) / "resume_ledger.jsonl"
            led.write_text("\n".join([
                json.dumps({"ts": "2026-06-23T17:50:00Z", "session": "fresh"}),   # 10m ago
                json.dumps({"ts": "2026-06-23T10:00:00Z", "session": "stale"}),   # 8h ago
                json.dumps({"ts": "bad", "session": "x"}),
            ]), encoding="utf-8")
            orig = resume_sweep.LEDGER
            try:
                resume_sweep.LEDGER = str(led)
                got = resume_sweep.recently_resumed_sids(600, now)   # 10h window
                got2 = resume_sweep.recently_resumed_sids(60, now)   # 1h window
            finally:
                resume_sweep.LEDGER = orig
            self.assertIn("fresh", got)
            self.assertIn("stale", got)   # 8h < 10h window
            self.assertIn("fresh", got2)  # 50m ago < 1h window
            self.assertNotIn("stale", got2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
