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

import fleet_sessions  # noqa: E402  (REHOME_CAP -- the cap-relief tests assert against it)
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


def _probe_all_ok(_acct: dict) -> dict:
    """A target/owner probe stub that reports every account serving -- so the
    target-confirmation probe is a no-op pass-through to the ranked winner. Tests
    that care about the BLOCKED-target path inject their own stub instead."""
    return {"available": True, "block_reason": "", "status_source": "probe"}


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


class DuplicateOwnerReselectTests(unittest.TestCase):
    """A session duplicated across accounts (the signature of a prior re-home): the
    newest-mtime copy is the re-home TARGET, which may be walled. The resolver probes
    it and falls back to a copy whose account a live probe confirms serving -- the
    operator's exact failure (resumed onto a limited day24 while q-netra served)."""

    # Tags chosen so they cannot collide with the live session registry -> the
    # default runtime_status (available=True) applies and the reselect path -- driven
    # entirely by the injected probe_fn -- is what's under test.
    ORIG = ".claude-rrorig-acct"
    TARGET = ".claude-rrtarget-acct"

    def test_reselects_serving_copy_when_newest_is_walled(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            # ORIG is the original copy; TARGET is the newer re-home target.
            orig = _write_session(home, self.ORIG)
            target = _write_session(home, self.TARGET)
            os.utime(orig, (1_000_000, 1_000_000))
            os.utime(target, (2_000_000, 2_000_000))  # target newest -> picked first

            def probe(a: dict) -> dict:
                if a["account"] == self.TARGET:
                    return {"available": False, "block_reason": "usage limit",
                            "status_source": "probe", "block_kind": "usage"}
                return {"available": True, "status_source": "probe"}

            rec = resume_resolver.resolve(SID, home, dry_run=True, probe_fn=probe)
            self.assertEqual(rec["action"], "PIN")
            self.assertEqual(rec["pin_account"], self.ORIG)
            self.assertEqual(rec["owner_reselected"],
                             {"from": self.TARGET, "to": self.ORIG})

    def test_keeps_newest_when_it_serves(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            orig = _write_session(home, self.ORIG)
            target = _write_session(home, self.TARGET)
            os.utime(orig, (1_000_000, 1_000_000))
            os.utime(target, (2_000_000, 2_000_000))
            # newest (TARGET) serves -> no reselect, normal pin to it.
            rec = resume_resolver.resolve(SID, home, dry_run=True,
                                          probe_fn=_probe_all_ok)
            self.assertEqual(rec["action"], "PIN")
            self.assertEqual(rec["pin_account"], self.TARGET)
            self.assertNotIn("owner_reselected", rec)

    def test_rehomes_freshest_when_serving_sibling_is_behind(self) -> None:
        # The reported "resume pins badly": the WALLED newest copy has MORE turns
        # than the serving sibling (the session kept going on the now-throttled
        # account). Pinning the older sibling would silently resume a STALE
        # transcript and drop the latest exchanges. Instead the freshest copy must
        # be re-homed (full content) onto a healthy account and pinned THERE.
        with tempfile.TemporaryDirectory() as home:
            serving = _write_session(home, self.ORIG)        # older, fewer turns
            walled = _write_session(home, self.TARGET, sidecar=True)  # newest+bigger
            with open(walled, "a", encoding="utf-8") as f:
                f.write('{"type":"user"}\n{"type":"assistant"}\n')  # advance it
            os.utime(serving, (1_000_000, 1_000_000))
            os.utime(walled, (2_000_000, 2_000_000))          # walled = newest mtime

            def probe(a: dict) -> dict:
                if a["account"] == self.TARGET:               # walled freshest
                    return {"available": False, "block_reason": "usage limit",
                            "status_source": "probe", "block_kind": "usage"}
                return {"available": True, "status_source": "probe"}

            # The walled freshest must also read blocked from the registry so the
            # downstream re-home fires; inject runtime_status (leaving owner_status
            # unset keeps the reselect path -- which it gates on -- live).
            orig_rs = resume_resolver.fleet_accounts.runtime_status
            resume_resolver.fleet_accounts.runtime_status = (
                lambda acct: {"available": False, "block_kind": "usage",
                              "block_reason": "usage limit", "status_source": "registry"}
                if acct == self.TARGET else {"available": True})
            try:
                rec = resume_resolver.resolve(
                    SID, home, dry_run=True, probe_fn=probe,
                    availability=[_avail(self.ORIG, True, home=home)])
            finally:
                resume_resolver.fleet_accounts.runtime_status = orig_rs

            self.assertEqual(rec["action"], "REHOME")
            self.assertEqual(rec["pin_account"], self.ORIG)
            self.assertEqual(rec["source_config_dir"],
                             os.path.join(home, self.TARGET))

    def test_single_owner_is_not_probed(self) -> None:
        # a non-duplicated session takes the fast path: no owner probe at all.
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, self.ORIG)
            probed: list = []
            rec = resume_resolver.resolve(
                SID, home, dry_run=True,
                probe_fn=lambda a: probed.append(a["account"]) or {"available": True})
            self.assertEqual(rec["action"], "PIN")
            self.assertEqual(rec["pin_account"], self.ORIG)
            self.assertEqual(probed, [])  # single owner -> never probed


class ResolveTests(unittest.TestCase):
    def test_pin_when_owner_available(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct")
            calls: list = []
            # cwd whose slug == the session's PROJECT slug, so the cross-dir mirror
            # (which fires only when the resume cwd differs) does NOT trigger here.
            rec = resume_resolver.resolve(
                SID, home,
                owner_status={"available": True},
                availability=[],
                cwd=r"C:\work\fleet",
                rehome_fn=lambda *a, **k: calls.append((a, k)) or True)
            self.assertEqual(rec["action"], "PIN")
            self.assertFalse(rec["rehomed"])
            self.assertEqual(rec["pin_config_dir"],
                             os.path.join(home, ".claude-gem8-acct"))
            self.assertEqual(calls, [])  # no copy on the pin path (same-slug cwd)

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
                probe_fn=_probe_all_ok,
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
                availability=[_avail(".claude-gem5-acct", True, home=home)],
                probe_fn=_probe_all_ok)
            self.assertEqual(rec["action"], "REHOME")
            dst = os.path.join(home, ".claude-gem5-acct", "projects", PROJECT)
            dst_jsonl = os.path.join(dst, SID + ".jsonl")
            self.assertTrue(os.path.isfile(dst_jsonl))
            self.assertTrue(os.path.isfile(os.path.join(dst, SID, "agent-x.jsonl")))
            # re-homed copy must out-rank the throttled original on mtime
            self.assertGreater(os.path.getmtime(dst_jsonl), os.path.getmtime(src))

    def test_rehome_also_lands_under_cwd_slug(self) -> None:
        # Cross-dir fix: resuming from a DIFFERENT folder than the session was born in,
        # the re-home copy must ALSO land under the launching cwd's project slug -- else
        # `claude --resume` (cwd-scoped) 404s. Here owner is throttled -> REHOME path.
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct", sidecar=True)
            other_slug = resume_resolver.project_slug(r"C:\work\slack-helpers")
            self.assertNotEqual(other_slug, PROJECT)
            rec = resume_resolver.resolve(
                SID, home,
                owner_status={"available": False, "block_reason": "usage limit"},
                availability=[_avail(".claude-gem5-acct", True, home=home)],
                probe_fn=_probe_all_ok,
                cwd=r"C:\work\slack-helpers")
            self.assertEqual(rec["action"], "REHOME")
            self.assertEqual(rec["dest_project_slugs"], [PROJECT, other_slug])
            tgt = os.path.join(home, ".claude-gem5-acct", "projects")
            # the transcript exists under BOTH the owner slug and the cwd slug
            self.assertTrue(os.path.isfile(os.path.join(tgt, PROJECT, SID + ".jsonl")))
            self.assertTrue(os.path.isfile(os.path.join(tgt, other_slug, SID + ".jsonl")))

    def test_pin_mirrors_into_cwd_slug_within_owner(self) -> None:
        # Cross-dir fix on the PIN path: owner is available (no re-home), but the
        # transcript lives under the owner's birth slug; resuming from another folder
        # would 404. The resolver mirrors it WITHIN the owner account into the cwd slug.
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct", sidecar=True)
            other_slug = resume_resolver.project_slug(r"C:\work\slack-helpers")
            rec = resume_resolver.resolve(
                SID, home,
                owner_status={"available": True},
                availability=[],
                cwd=r"C:\work\slack-helpers")
            self.assertEqual(rec["action"], "PIN")
            self.assertEqual(rec["mirrored_to_cwd_slug"], other_slug)
            mirrored = os.path.join(home, ".claude-gem8-acct", "projects",
                                    other_slug, SID + ".jsonl")
            self.assertTrue(os.path.isfile(mirrored))

    def test_dry_run_does_not_copy(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct")
            rec = resume_resolver.resolve(
                SID, home, dry_run=True,
                owner_status={"available": False, "block_reason": "usage limit"},
                availability=[_avail(".claude-gem5-acct", True, home=home)],
                probe_fn=_probe_all_ok)
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

    def test_cap_relief_rehomes_when_only_healthy_account_is_over_cap(self) -> None:
        # The day24 incident: owner walled, and the ONLY healthy Claude account is
        # available but its live load is over REHOME_CAP (so _rehome_targets drops it
        # and the fleet path would PIN_BLOCKED). A single interactive resume relaxes the
        # burst cap -- live-probes the over-cap account, confirms it serves, re-homes there.
        over_cap = fleet_sessions.REHOME_CAP + 3
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct", sidecar=True)
            calls: list = []
            rec = resume_resolver.resolve(
                SID, home,
                owner_status={"available": False, "block_reason": "usage limit"},
                availability=[
                    _avail(".claude-gem8-acct", False, home=home),
                    _avail(".claude-day24-acct", True, live=over_cap, active=41, home=home),
                ],
                probe_fn=_probe_all_ok,
                rehome_fn=lambda *a, **k: calls.append((a, k)) or True)
            self.assertEqual(rec["action"], "REHOME")
            self.assertTrue(rec["rehomed"])
            self.assertEqual(rec["pin_account"], ".claude-day24-acct")
            self.assertEqual(rec["cap_relief"]["rehome_cap"], fleet_sessions.REHOME_CAP)
            # the relief target was live-probe-confirmed before being chosen
            self.assertTrue(any(p["account"] == ".claude-day24-acct" and p["available"]
                                for p in rec.get("target_probes", [])))

    def test_cap_relief_still_pin_blocked_when_over_cap_account_is_walled(self) -> None:
        # Relief relaxes the LOAD cap, not the health check: if the only over-cap
        # account probes BLOCKED, re-homing would just move the resume to another walled
        # seat, so it stays PIN_BLOCKED. Proves relief can't strand on a secretly-walled
        # account.
        over_cap = fleet_sessions.REHOME_CAP + 3
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct")
            rec = resume_resolver.resolve(
                SID, home,
                owner_status={"available": False, "block_reason": "usage limit"},
                availability=[
                    _avail(".claude-gem8-acct", False, home=home),
                    _avail(".claude-day24-acct", True, live=over_cap, active=41, home=home),
                ],
                probe_fn=lambda _a: {"available": False, "block_reason": "usage limit"})
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


class CarriedThrottleProbeTests(unittest.TestCase):
    """The owner is reported throttled by a CARRIED registry block (a stale
    bare-time reset that reads as future after it actually cleared). The resolver
    live-probes the owner before abandoning it; a fresh OK probe means PIN, not
    re-home -- the exact false-throttle that re-homed off a healthy q-netra."""

    def test_carried_throttle_probe_ok_pins_owner(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct")
            calls: list = []
            rec = resume_resolver.resolve(
                SID, home,
                owner_status={"available": False, "block_kind": "usage",
                              "block_reason": "usage limit; resets 3pm",
                              "status_source": "registry"},
                availability=[_avail(".claude-gem5-acct", True, home=home)],
                probe_fn=lambda owner: {"available": True, "block_reason": "",
                                        "status_source": "probe"},
                cwd=r"C:\work\fleet",  # same slug as PROJECT -> no cross-dir mirror
                rehome_fn=lambda *a, **k: calls.append((a, k)) or True)
            self.assertEqual(rec["action"], "PIN")
            self.assertFalse(rec["rehomed"])
            self.assertEqual(rec["pin_account"], ".claude-gem8-acct")
            self.assertTrue(rec["owner_probe"]["available"])
            self.assertEqual(calls, [])  # no copy -- owner is actually serving

    def test_carried_throttle_probe_confirms_block_rehomes(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct", sidecar=True)
            def probe(a: dict) -> dict:
                # owner's carried throttle is REAL (probe confirms block); the target
                # serves, so we re-home onto it.
                if a["account"] == ".claude-gem8-acct":
                    return {"available": False, "block_reason": "usage limit confirmed",
                            "status_source": "probe", "block_kind": "usage"}
                return {"available": True, "status_source": "probe"}
            rec = resume_resolver.resolve(
                SID, home, dry_run=True,
                owner_status={"available": False, "block_kind": "usage",
                              "block_reason": "usage limit; resets 3pm",
                              "status_source": "registry"},
                availability=[_avail(".claude-gem5-acct", True, home=home)],
                probe_fn=probe)
            self.assertEqual(rec["action"], "REHOME")
            self.assertEqual(rec["pin_account"], ".claude-gem5-acct")

    def test_probe_confirmed_block_is_not_reprobed(self) -> None:
        # a block already confirmed by a fresh probe (status_source=="probe") is
        # trusted -- the OWNER is not re-probed (the target still is, to confirm it
        # actually serves before landing the resume there).
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct")
            probed: list = []
            rec = resume_resolver.resolve(
                SID, home, dry_run=True,
                owner_status={"available": False, "block_kind": "usage",
                              "block_reason": "usage limit", "status_source": "probe"},
                availability=[_avail(".claude-gem5-acct", True, home=home)],
                probe_fn=lambda a: probed.append(a["account"]) or {"available": True})
            self.assertEqual(rec["action"], "REHOME")
            self.assertNotIn(".claude-gem8-acct", probed)  # owner never re-probed
            self.assertEqual(probed, [".claude-gem5-acct"])  # only the target

    def test_auth_block_owner_not_reprobed(self) -> None:
        # only a CARRIED usage throttle re-probes the OWNER; an auth/access block is a
        # different, durable kind, so the owner goes straight to re-home (the target is
        # still probed for serving-confirmation).
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct")
            probed: list = []
            rec = resume_resolver.resolve(
                SID, home, dry_run=True,
                owner_status={"available": False, "block_kind": "auth",
                              "block_reason": "auth/login required",
                              "status_source": "registry"},
                availability=[_avail(".claude-gem5-acct", True, home=home)],
                probe_fn=lambda a: probed.append(a["account"]) or {"available": True})
            self.assertEqual(rec["action"], "REHOME")
            self.assertNotIn(".claude-gem8-acct", probed)  # owner not re-probed

    def test_no_probe_flag_trusts_carried_throttle(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct")
            probed: list = []
            rec = resume_resolver.resolve(
                SID, home, dry_run=True, probe_owner=False,
                owner_status={"available": False, "block_kind": "usage",
                              "block_reason": "usage limit", "status_source": "registry"},
                availability=[_avail(".claude-gem5-acct", True, home=home)],
                probe_fn=lambda a: probed.append(a["account"]) or {"available": True})
            self.assertEqual(rec["action"], "REHOME")
            self.assertEqual(probed, [])  # probe suppressed entirely by probe_owner=False

    def test_all_targets_blocked_pins_owner(self) -> None:
        # owner throttled AND every probed target also limited -> re-homing would just
        # move the resume onto another walled account, so pin to the owner instead.
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct")
            rec = resume_resolver.resolve(
                SID, home, dry_run=True,
                owner_status={"available": False, "block_kind": "auth",
                              "block_reason": "auth/login required",
                              "status_source": "registry"},
                availability=[
                    _avail(".claude-gem5-acct", True, home=home),
                    _avail(".claude-gem7-acct", True, home=home),
                ],
                probe_fn=lambda a: {"available": False,
                                    "block_reason": "usage limit confirmed",
                                    "status_source": "probe", "block_kind": "usage"})
            self.assertEqual(rec["action"], "PIN_BLOCKED")
            self.assertEqual(rec["pin_account"], ".claude-gem8-acct")
            self.assertTrue(rec["target_probes"])

    def test_skips_blocked_target_to_next_serving(self) -> None:
        # the top-ranked target probes BLOCKED; the resolver skips it and lands on the
        # next candidate that a live probe confirms serving.
        with tempfile.TemporaryDirectory() as home:
            _write_session(home, ".claude-gem8-acct", sidecar=True)
            def probe(a: dict) -> dict:
                if a["account"] == ".claude-gem5-acct":
                    return {"available": False, "block_reason": "limited",
                            "status_source": "probe", "block_kind": "usage"}
                return {"available": True, "status_source": "probe"}
            rec = resume_resolver.resolve(
                SID, home, dry_run=True,
                owner_status={"available": False, "block_kind": "auth",
                              "block_reason": "auth/login required",
                              "status_source": "registry"},
                # gem5 is least-loaded (ranked first) but limited; gem7 is next.
                availability=[
                    _avail(".claude-gem5-acct", True, live=0, home=home),
                    _avail(".claude-gem7-acct", True, live=0, active=1, home=home),
                ],
                probe_fn=probe)
            self.assertEqual(rec["action"], "REHOME")
            self.assertEqual(rec["pin_account"], ".claude-gem7-acct")


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
