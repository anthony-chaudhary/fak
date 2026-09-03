#!/usr/bin/env python3
"""Hermetic tests for the release lock (`tools/release_lock.py`).

Run natively — `python tools/release_lock_test.py` — NOT through WSL. The WDAC
policy that blocks freshly-built unsigned Go test binaries (see CLAUDE.md) does
not touch the system Python interpreter, so plain `python` is the right runner for
fleet's stdlib helpers. Every test operates in its own `tempfile` root, so it never
reads or writes the repo's real `.release.lock`.
"""
from __future__ import annotations

import sys
import json
import tempfile
import unittest
from unittest import mock
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import the sibling helper
import release_lock as rl  # noqa: E402


class LockTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        # Neutralize an INHERITED lock-root override for the duration of each test, so the
        # tempfile root above actually isolates. lock_path() consults $FAK_RELEASE_LOCK_ROOT
        # ahead of the root it is handed, so a value already in the environment silently
        # redirects every acquire below onto the repo's real .release.lock -- the isolation
        # this module's docstring promises, quietly voided by an ambient variable.
        #
        # That is not a hypothetical: `fak release ship` exports FAK_RELEASE_LOCK_ROOT=<repo>
        # so the release cut it drives shares its lease, and that cut runs THIS suite as its
        # release-substrate dry-run witness. The suite inherited the variable, every acquire
        # landed on the very lock ship was still holding, and six tests failed EXIT_CONTENDED
        # != EXIT_OK -- making a shipped release unable to pass its own witness, and leaving
        # the test's owner token stranded in the repo lock afterwards. The one test that
        # exercises the override sets it explicitly, so it is unaffected.
        self._old_lock_root = os.environ.pop("FAK_RELEASE_LOCK_ROOT", None)

    def tearDown(self) -> None:
        self._tmp.cleanup()
        os.environ.pop("FAK_RELEASE_LOCK_ROOT", None)
        if self._old_lock_root is not None:
            os.environ["FAK_RELEASE_LOCK_ROOT"] = self._old_lock_root

    # --- acquire / mutual exclusion -------------------------------------------------

    def test_acquire_creates_lock(self) -> None:
        out, code = rl.acquire(self.root, ttl=60, owner="A", snapshot=["VERSION"],
                               note=None, steal_stale=True, force=False)
        self.assertEqual(code, rl.EXIT_OK)
        self.assertTrue(out["ok"])
        self.assertTrue(rl.lock_path(self.root).exists())
        self.assertEqual(out["lock"]["owner"], "A")
        self.assertEqual(out["lock"]["snapshot"], ["VERSION"])

    def test_lock_root_env_override_shares_lock_across_worktrees(self) -> None:
        shared = self.root / "shared-lock-root"
        shared.mkdir()
        worktree = self.root / "detached-worktree"
        worktree.mkdir()
        old = os.environ.get("FAK_RELEASE_LOCK_ROOT")
        os.environ["FAK_RELEASE_LOCK_ROOT"] = str(shared)
        try:
            out, code = rl.acquire(worktree, ttl=60, owner="A", snapshot=[],
                                   note=None, steal_stale=True, force=False)
            self.assertEqual(code, rl.EXIT_OK)
            self.assertTrue(out["ok"])
            self.assertTrue((shared / rl.LOCK_NAME).exists())
            self.assertFalse((worktree / rl.LOCK_NAME).exists())
            self.assertEqual(rl.held_by(worktree, "A")[0], True)
        finally:
            if old is None:
                os.environ.pop("FAK_RELEASE_LOCK_ROOT", None)
            else:
                os.environ["FAK_RELEASE_LOCK_ROOT"] = old

    def test_second_live_acquire_is_denied(self) -> None:
        rl.acquire(self.root, ttl=60, owner="A", snapshot=[], note=None, steal_stale=True, force=False)
        out, code = rl.acquire(self.root, ttl=60, owner="B", snapshot=[], note=None, steal_stale=True, force=False)
        self.assertEqual(code, rl.EXIT_DENIED)
        self.assertFalse(out["ok"])
        self.assertEqual(out["reason"], "held")
        self.assertEqual(out["holder"]["owner"], "A")  # A still owns it

    # --- staleness ------------------------------------------------------------------

    def test_stale_lock_is_stolen(self) -> None:
        rl.acquire(self.root, ttl=0, owner="A", snapshot=[], note=None, steal_stale=True, force=False)  # born expired
        out, code = rl.acquire(self.root, ttl=60, owner="B", snapshot=[], note=None, steal_stale=True, force=False)
        self.assertEqual(code, rl.EXIT_OK)
        self.assertTrue(out["ok"])
        self.assertEqual(out["lock"]["owner"], "B")
        self.assertEqual(out["stole"]["owner"], "A")
        self.assertEqual(out["stole"]["_stolen_because"], "expired")

    def test_no_steal_stale_refuses_expired(self) -> None:
        rl.acquire(self.root, ttl=0, owner="A", snapshot=[], note=None, steal_stale=True, force=False)
        out, code = rl.acquire(self.root, ttl=60, owner="B", snapshot=[], note=None, steal_stale=False, force=False)
        self.assertEqual(code, rl.EXIT_DENIED)
        self.assertFalse(out["ok"])

    def test_force_steals_a_live_lock(self) -> None:
        rl.acquire(self.root, ttl=600, owner="A", snapshot=[], note=None, steal_stale=True, force=False)
        out, code = rl.acquire(self.root, ttl=60, owner="B", snapshot=[], note=None, steal_stale=True, force=True, takeover_reason="operator approved")
        self.assertEqual(code, rl.EXIT_OK)
        self.assertEqual(out["lock"]["owner"], "B")
        self.assertEqual(out["stole"]["_stolen_because"], "force: operator approved")

    def test_corrupt_lock_is_stealable(self) -> None:
        rl.lock_path(self.root).write_text("{ not json", encoding="utf-8")
        out, code = rl.acquire(self.root, ttl=60, owner="B", snapshot=[], note=None, steal_stale=True, force=False)
        self.assertEqual(code, rl.EXIT_OK)
        self.assertEqual(out["lock"]["owner"], "B")

    def test_is_stale_variants(self) -> None:
        live = {"acquired_at": 1000.0, "ttl": 100, "expires_at": 1100.0}
        self.assertEqual(rl.is_stale(live, at=1050.0)[0], False)
        self.assertEqual(rl.is_stale(live, at=1200.0)[0], True)
        # missing expiry but reconstructable from acquired+ttl
        self.assertEqual(rl.is_stale({"acquired_at": 1000.0, "ttl": 100}, at=1050.0)[0], False)
        # no expiry info at all → stale by default
        self.assertEqual(rl.is_stale({}, at=1050.0)[0], True)

    # --- held_by / verify -----------------------------------------------------------

    def test_held_by_owner_semantics(self) -> None:
        rl.acquire(self.root, ttl=60, owner="A", snapshot=[], note=None, steal_stale=True, force=False)
        self.assertEqual(rl.held_by(self.root, "A")[0], True)
        self.assertEqual(rl.held_by(self.root, "B")[0], False)   # wrong owner
        self.assertEqual(rl.held_by(self.root, None)[0], True)    # any live lock
        rl.release(self.root, owner="A", force=False)
        self.assertEqual(rl.held_by(self.root, "A")[0], False)    # gone

    def test_held_by_expired_is_false(self) -> None:
        rl.acquire(self.root, ttl=0, owner="A", snapshot=[], note=None, steal_stale=True, force=False)
        ok, why = rl.held_by(self.root, "A")
        self.assertFalse(ok)
        self.assertIn("stale", why)

    # --- release --------------------------------------------------------------------

    def test_release_by_owner(self) -> None:
        rl.acquire(self.root, ttl=60, owner="A", snapshot=[], note=None, steal_stale=True, force=False)
        out, code = rl.release(self.root, owner="A", force=False)
        self.assertEqual(code, rl.EXIT_OK)
        self.assertTrue(out["released"])
        self.assertFalse(rl.lock_path(self.root).exists())

    def test_release_by_wrong_owner_denied(self) -> None:
        rl.acquire(self.root, ttl=60, owner="A", snapshot=[], note=None, steal_stale=True, force=False)
        out, code = rl.release(self.root, owner="B", force=False)
        self.assertEqual(code, rl.EXIT_DENIED)
        self.assertFalse(out["ok"])
        self.assertTrue(rl.lock_path(self.root).exists())  # untouched
        out2, code2 = rl.release(self.root, owner=None, force=True)  # force overrides
        self.assertEqual(code2, rl.EXIT_OK)
        self.assertTrue(out2["released"])

    def test_release_when_absent_is_ok(self) -> None:
        out, code = rl.release(self.root, owner="A", force=False)
        self.assertEqual(code, rl.EXIT_OK)
        self.assertFalse(out["released"])

    # --- guard / no `git add -A` sweep ----------------------------------------------

    def test_classify_staged_allows_version_and_notes(self) -> None:
        staged = ["VERSION", "docs/releases/v0.5.0.md"]
        res = rl.classify_staged(staged, allowed=[], notes_dir="docs/releases")
        self.assertTrue(res["ok"])
        self.assertEqual(res["foreign"], [])

    def test_classify_staged_flags_foreign(self) -> None:
        staged = ["VERSION", "docs/releases/v0.5.0.md", "fak/internal/gateway/gateway.go"]
        res = rl.classify_staged(staged, allowed=["docs/some-doc.md"], notes_dir="docs/releases")
        self.assertFalse(res["ok"])
        self.assertEqual(res["foreign"], ["fak/internal/gateway/gateway.go"])  # the swept-in peer file

    def test_classify_staged_explicit_and_glob_allow(self) -> None:
        staged = ["docs/a.md", "tools/x.py", "tools/y.py"]
        res = rl.classify_staged(staged, allowed=["docs/a.md", "tools/*.py"], notes_dir="docs/releases")
        self.assertTrue(res["ok"])
        self.assertEqual(res["foreign"], [])

    def test_classify_staged_backslashes_normalized(self) -> None:
        res = rl.classify_staged(["docs\\releases\\v1.0.0.md"], allowed=[], notes_dir="docs/releases")
        self.assertTrue(res["ok"])


    def test_live_takeover_requires_reason_and_records_audit(self) -> None:
        root = self.root
        first, code = rl.acquire(root, ttl=60, owner="first", snapshot=[], note=None, steal_stale=True, force=False)
        self.assertEqual(code, rl.EXIT_OK)
        denied, code = rl.acquire(root, ttl=60, owner="second", snapshot=[], note=None, steal_stale=True, force=True)
        self.assertEqual(code, rl.EXIT_USAGE)
        self.assertEqual(denied["reason"], "takeover reason required")
        taken, code = rl.acquire(root, ttl=60, owner="second", snapshot=[], note=None, steal_stale=True, force=True, takeover_reason="first cutter abandoned")
        self.assertEqual(code, rl.EXIT_OK)
        self.assertEqual(taken["lock"]["lease_id"], "second")
        self.assertEqual(taken["lock"]["takeover"]["previous_owner"], "first")
        self.assertEqual(taken["lock"]["takeover"]["reason"], "force: first cutter abandoned")

    def test_renewal_retains_lease_identity(self) -> None:
        with mock.patch.object(rl, "now", side_effect=[100.0, 110.0]):
            acquired, code = rl.acquire(self.root, ttl=20, owner="stable", snapshot=["VERSION"], note="ship", steal_stale=True, force=False)
            self.assertEqual(code, rl.EXIT_OK)
            renewed, code = rl.renew(self.root, owner="stable", ttl=30, force=False)
        self.assertEqual(code, rl.EXIT_OK)
        self.assertEqual(renewed["lock"]["lease_id"], acquired["lock"]["lease_id"])
        self.assertEqual([e["event"] for e in renewed["lock"]["lifecycle"]], ["acquired", "renewed"])
        self.assertEqual(renewed["lock"]["acquired_at"], 100.0)

    def test_release_receipt_survives_lock_and_binds_commit_time(self) -> None:
        with mock.patch.object(rl, "now", side_effect=[100.0, 110.0, 120.0]):
            acquired, code = rl.acquire(self.root, ttl=30, owner="stable", snapshot=[], note="ship", steal_stale=True, force=False)
            self.assertEqual(code, rl.EXIT_OK)
            renewed, code = rl.renew(self.root, owner="stable", ttl=30, force=False)
            self.assertEqual(code, rl.EXIT_OK)
            receipt_path = self.root / "receipts" / "abc123.json"
            released, code = rl.release(self.root, owner="stable", force=False, receipt_commit="abc123", receipt_path=str(receipt_path))
        self.assertEqual(code, rl.EXIT_OK)
        self.assertIsNone(rl.read_lock(self.root))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["owner"], "stable")
        self.assertEqual(receipt["lease_id"], acquired["lock"]["lease_id"])
        self.assertEqual(receipt["commit_sha"], "abc123")
        self.assertEqual(receipt["released_at"], 120.0)
        self.assertEqual([e["event"] for e in receipt["lifecycle"]], ["acquired", "renewed", "released"])
        self.assertEqual(released["receipt"]["commit_sha"], "abc123")


    def test_two_cutters_exactly_one_acquires(self) -> None:
        import concurrent.futures
        mutations: list[str] = []

        def cutter(owner: str) -> int:
            _, code = rl.acquire(self.root, ttl=60, owner=owner, snapshot=[], note=None, steal_stale=False, force=False)
            if code == rl.EXIT_OK:
                mutations.append(owner)
            return code

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            codes = list(pool.map(cutter, ["one", "two"]))
        self.assertEqual(sorted(codes), [rl.EXIT_OK, rl.EXIT_DENIED])
        self.assertEqual(len(mutations), 1, "exactly one cutter may perform the release mutation")
        self.assertEqual(rl.read_lock(self.root)["owner"], mutations[0])

    def test_crash_ttl_recovery_is_identity_safe(self) -> None:
        with mock.patch.object(rl, "now", side_effect=[100.0, 121.0]):
            first, code = rl.acquire(self.root, ttl=20, owner="crashed", snapshot=[], note=None, steal_stale=True, force=False)
            self.assertEqual(code, rl.EXIT_OK)
            recovered, code = rl.acquire(self.root, ttl=20, owner="recovery", snapshot=[], note=None, steal_stale=True, force=False)
        self.assertEqual(code, rl.EXIT_OK)
        self.assertEqual(recovered["lock"]["lease_id"], "recovery")
        self.assertTrue(recovered["lock"]["takeover"]["stale"])
        self.assertEqual(recovered["lock"]["takeover"]["previous_lease_id"], first["lock"]["lease_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
