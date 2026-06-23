#!/usr/bin/env python3
"""Tests for resume_relaunch_audit -- the relaunch-OUTCOME verifier."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import resume_relaunch_audit as A  # noqa: E402


def _rec(role, text, ts, err=False):
    r = {"message": {"role": role, "content": [{"type": "text", "text": text}]},
         "timestamp": ts}
    if err:
        r["isApiErrorMessage"] = True
    return r


class RelaunchVerdictTest(unittest.TestCase):
    def test_advanced_past_error_is_ok(self):
        # error, then a real turn after it -> the relaunch took.
        recs = [_rec("assistant", "API Error: 529 Overloaded", "2026-06-23T17:00:00Z", err=True),
                _rec("assistant", "Resumed; wired the lanes and shipped.", "2026-06-23T17:30:00Z")]
        self.assertEqual(A.relaunch_verdict(recs)["verdict"], "RELAUNCHED_OK")

    def test_terminal_limit_banner_is_stranded(self):
        # real work, then a terminal usage-limit cut -> NOT properly relaunched (the
        # 8-cut wave shape: host capped mid-turn).
        recs = [_rec("assistant", "Implementing the witness files now.", "2026-06-23T17:59:00Z"),
                _rec("assistant", "You've hit your session limit . resets 12:10pm (America/Los_Angeles)",
                     "2026-06-23T18:00:00Z", err=True)]
        v = A.relaunch_verdict(recs)
        self.assertEqual(v["verdict"], "STRANDED")
        self.assertEqual(v["kind"], "LIMIT")
        self.assertIn("session limit", v["evidence"])

    def test_terminal_auth_is_stranded_auth(self):
        recs = [_rec("assistant", "Working.", "2026-06-23T17:00:00Z"),
                _rec("assistant", "Not logged in . Please run /login", "2026-06-23T17:01:00Z", err=True)]
        v = A.relaunch_verdict(recs)
        self.assertEqual((v["verdict"], v["kind"]), ("STRANDED", "AUTH"))

    def test_never_worked_when_only_errors(self):
        recs = [_rec("assistant", "API Error: 529", "2026-06-23T17:00:00Z", err=True)]
        self.assertEqual(A.relaunch_verdict(recs)["verdict"], "NEVER_WORKED")

    def test_clean_session_no_error_is_ok(self):
        recs = [_rec("assistant", "All green, nothing outstanding.", "2026-06-23T17:00:00Z")]
        self.assertEqual(A.relaunch_verdict(recs)["verdict"], "RELAUNCHED_OK")

    def test_real_turn_mentioning_resets_is_not_an_error(self):
        # a real turn that merely discusses "resets" (without the limit-banner marker) is
        # NOT counted as the error channel -- only "hit your session limit" / an error
        # record is. So this advances past the earlier 529.
        recs = [_rec("assistant", "API Error: 529", "2026-06-23T17:00:00Z", err=True),
                _rec("assistant", "The reset_passed() helper handles when a window resets.",
                     "2026-06-23T17:05:00Z")]
        self.assertEqual(A.relaunch_verdict(recs)["verdict"], "RELAUNCHED_OK")


class AuditIntegrationTest(unittest.TestCase):
    def test_audit_reads_ledger_and_verifies_transcript(self):
        with tempfile.TemporaryDirectory() as home:
            sid = "11111111-2222-3333-4444-555555555555"
            tpath = Path(home) / ".claude" / "projects" / "C--work-fak" / f"{sid}.jsonl"
            tpath.parent.mkdir(parents=True, exist_ok=True)
            tpath.write_text("\n".join(json.dumps(r) for r in [
                _rec("assistant", "work", "2026-06-23T17:59:00Z"),
                _rec("assistant", "You've hit your session limit . resets 12:10pm (America/Los_Angeles)",
                     "2026-06-23T18:00:00Z", err=True),
            ]), encoding="utf-8")
            ledger = Path(home) / "ledger.jsonl"
            ledger.write_text(json.dumps({"session": sid, "action": "rehome-gem7-11am"}) + "\n",
                              encoding="utf-8")
            rows = A.audit(home=home, ledger=str(ledger))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["verdict"], "STRANDED")
            self.assertEqual(rows[0]["kind"], "LIMIT")
            self.assertEqual(rows[0]["account"], ".claude")


if __name__ == "__main__":
    unittest.main(verbosity=2)
