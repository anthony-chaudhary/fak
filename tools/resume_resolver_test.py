#!/usr/bin/env python3
"""Unit tests for resume_resolver -- the throttle-aware `c --resume` resolver.

The decision is exercised with INJECTED availability / owner-status / copy so it
runs without a live registry or real account dirs: temp ``.claude*`` trees stand
in for the on-disk session stores, and a stub records each re-home copy.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import resume_resolver  # noqa: E402

SID = "caeb3a9c-1efd-4a0c-9c18-8769d924ce35"
PROJECT = "C--work-fleet"


def _write_session(home: str, account: str, sid: str = SID,
                   project: str = PROJECT, *, sidecar: bool = False) -> str:
    """Materialize <home>/<account>/projects/<project>/<sid>.jsonl (+ optional
    <sid>/ sidecar). Returns the jsonl path."""
    proj = os.path.join(home, account, "projects", project)
    os.makedirs(proj, exist_ok=True)
    path = os.path.join(proj, sid + ".jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"type":"user"}\n')
    if sidecar:
        side = os.path.join(proj, sid)
        os.makedirs(side, exist_ok=True)
        with open(os.path.join(side, "agent-x.jsonl"), "w", encoding="utf-8") as f:
            f.write("{}\n")
    return path


def _avail(account: str, available: bool, *, live: int = 0, active: int = 0,
           home: str = "") -> dict:
    return {
        "account": account, "available": available,
        "live_sessions": live, "active_sessions": active,
        "tag": account.replace(".claude-", "").replace(".claude", "default")
                      .replace("-acct", ""),
        "config_dir": os.path.join(home, account) if home else account,
    }


class LocateOwnerTests(unittest.TestCase):
    def test_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            self.assertIsNone(resume_resolver.locate_owner(SID, home))

    def test_single_owner(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct")
            owner = resume_resolver.locate_owner(SID, home)
            self.assertIsNotNone(owner)
            self.assertEqual(owner["account"], ".claude-gem8-acct")
            self.assertEqual(owner["project"], PROJECT)
            self.assertEqual(owner["dup_count"], 1)

    def test_host_is_chosen_last(self) -> None:
        # session present in BOTH ~/.claude (host) and a rotated account -> the
        # non-host account owns it even when the host copy is newer on disk.
        with tempfile.TemporaryDirectory() as home:
            rotated = _write_session(home, ".claude-gem8-acct")
            host = _write_session(home, ".claude")
            # make the host copy strictly newer
            newer = os.path.getmtime(rotated) + 100
            os.utime(host, (newer, newer))
            owner = resume_resolver.locate_owner(SID, home)
            self.assertEqual(owner["account"], ".claude-gem8-acct")
            self.assertEqual(owner["dup_count"], 2)

    def test_host_only_owner(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude")
            owner = resume_resolver.locate_owner(SID, home)
            self.assertEqual(owner["account"], ".claude")

    def test_newest_non_host_wins(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            old = _write_session(home, ".claude-gem7-acct")
            new = _write_session(home, ".claude-gem8-acct")
            os.utime(old, (1_000_000, 1_000_000))
            os.utime(new, (2_000_000, 2_000_000))
            owner = resume_resolver.locate_owner(SID, home)
            self.assertEqual(owner["account"], ".claude-gem8-acct")


class ResolveTests(unittest.TestCase):
    def test_pin_when_owner_available(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct")
            calls: list = []
            rec = resume_resolver.resolve(
                SID, home,
                owner_status={"available": True},
                availability=[],
                rehome_fn=lambda *a: calls.append(a) or True)
            self.assertEqual(rec["action"], "PIN")
            self.assertFalse(rec["rehomed"])
            self.assertEqual(rec["pin_config_dir"],
                             os.path.join(home, ".claude-gem8-acct"))
            self.assertEqual(calls, [])  # no copy on the pin path

    def test_rehome_when_owner_throttled(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct", sidecar=True)
            calls: list = []
            rec = resume_resolver.resolve(
                SID, home,
                owner_status={"available": False,
                              "block_reason": "usage limit; resets 12:50pm"},
                availability=[
                    _avail(".claude-gem8-acct", False, home=home),
                    _avail(".claude-gem5-acct", True, live=1, active=5, home=home),
                    _avail(".claude-jack-barker-claude-acct", True, live=0, active=6, home=home),
                ],
                rehome_fn=lambda *a: calls.append(a) or True)
            self.assertEqual(rec["action"], "REHOME")
            self.assertTrue(rec["rehomed"])
            # least-loaded healthy Claude worker (0 live) wins over gem5 (1 live)
            self.assertEqual(rec["pin_account"], ".claude-jack-barker-claude-acct")
            self.assertEqual(rec["source_config_dir"],
                             os.path.join(home, ".claude-gem8-acct"))
            self.assertEqual(len(calls), 1)
            src, dst, proj, sid = calls[0]
            self.assertEqual(src, os.path.join(home, ".claude-gem8-acct"))
            self.assertEqual(dst, os.path.join(home, ".claude-jack-barker-claude-acct"))
            self.assertEqual(proj, PROJECT)
            self.assertEqual(sid, SID)

    def test_rehome_actually_copies_with_real_primitive(self) -> None:
        # end-to-end through the real rehome_transcript: file + sidecar land at
        # the exact resume-lookup path under the target account, and the re-homed
        # copy is stamped NEWEST so the host-last/newest-mtime owner pick can't
        # re-select the throttled original (which copy2 would otherwise mtime-tie).
        with tempfile.TemporaryDirectory() as home:
            src = _write_session(home, ".claude-gem8-acct", sidecar=True)
            os.utime(src, (1_000_000, 1_000_000))  # old source mtime
            rec = resume_resolver.resolve(
                SID, home,
                owner_status={"available": False, "block_reason": "usage limit"},
                availability=[_avail(".claude-gem5-acct", True, home=home)])
            self.assertEqual(rec["action"], "REHOME")
            dst = os.path.join(home, ".claude-gem5-acct", "projects", PROJECT)
            dst_jsonl = os.path.join(dst, SID + ".jsonl")
            self.assertTrue(os.path.isfile(dst_jsonl))
            self.assertTrue(os.path.isfile(os.path.join(dst, SID, "agent-x.jsonl")))
            # re-homed copy must out-rank the throttled original on mtime
            self.assertGreater(os.path.getmtime(dst_jsonl), os.path.getmtime(src))

    def test_dry_run_does_not_copy(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct")
            rec = resume_resolver.resolve(
                SID, home, dry_run=True,
                owner_status={"available": False, "block_reason": "usage limit"},
                availability=[_avail(".claude-gem5-acct", True, home=home)])
            self.assertEqual(rec["action"], "REHOME")
            self.assertFalse(rec["rehomed"])
            self.assertTrue(rec["would_rehome"])
            dst = os.path.join(home, ".claude-gem5-acct", "projects", PROJECT,
                               SID + ".jsonl")
            self.assertFalse(os.path.exists(dst))  # nothing copied in dry-run

    def test_pin_blocked_when_no_healthy_account(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct")
            rec = resume_resolver.resolve(
                SID, home,
                owner_status={"available": False, "block_reason": "usage limit"},
                availability=[_avail(".claude-gem5-acct", False, home=home)])
            self.assertEqual(rec["action"], "PIN_BLOCKED")
            self.assertFalse(rec["rehomed"])
            self.assertEqual(rec["pin_config_dir"],
                             os.path.join(home, ".claude-gem8-acct"))

    def test_opencode_never_a_rehome_target(self) -> None:
        # a Claude transcript can only resume under another Claude config dir.
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct")
            rec = resume_resolver.resolve(
                SID, home,
                owner_status={"available": False, "block_reason": "usage limit"},
                availability=[
                    {"account": "opencode", "available": True, "live_sessions": 0,
                     "active_sessions": 0, "tag": "default",
                     "config_dir": os.path.join(home, "opencode")},
                ])
            self.assertEqual(rec["action"], "PIN_BLOCKED")

    def test_not_found_record(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            rec = resume_resolver.resolve(SID, home)
            self.assertFalse(rec["ok"])
            self.assertEqual(rec["action"], "NOT_FOUND")
            self.assertIsNone(rec["pin_config_dir"])


class MainCliTests(unittest.TestCase):
    def test_main_prints_bare_dir(self) -> None:
        import io
        import contextlib
        with tempfile.TemporaryDirectory() as home:
            # A name that cannot collide with the live registry, so runtime_status
            # falls back to its available=True default -> PIN, bare dir on stdout.
            acct = ".claude-resolvertest-acct"
            _write_session(home, acct)
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = resume_resolver.main([SID, "--home", home])
            self.assertEqual(rc, 0)
            self.assertEqual(out.getvalue().strip(), os.path.join(home, acct))

    def test_main_not_found_exits_1(self) -> None:
        import io
        import contextlib
        with tempfile.TemporaryDirectory() as home:
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = resume_resolver.main([SID, "--home", home])
            self.assertEqual(rc, 1)
            self.assertEqual(out.getvalue().strip(), "")


if __name__ == "__main__":
    unittest.main()
