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
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import the sibling helper
import release_lock as rl  # noqa: E402


class LockTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # --- acquire / mutual exclusion -------------------------------------------------

    def test_acquire_creates_lock(self) -> None:
        out, code = rl.acquire(self.root, ttl=60, owner="A", snapshot=["VERSION"],
                               note=None, steal_stale=True, force=False)
        self.assertEqual(code, rl.EXIT_OK)
        self.assertTrue(out["ok"])
        self.assertTrue(rl.lock_path(self.root).exists())
        self.assertEqual(out["lock"]["owner"], "A")
        self.assertEqual(out["lock"]["snapshot"], ["VERSION"])

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
        out, code = rl.acquire(self.root, ttl=60, owner="B", snapshot=[], note=None, steal_stale=True, force=True)
        self.assertEqual(code, rl.EXIT_OK)
        self.assertEqual(out["lock"]["owner"], "B")
        self.assertEqual(out["stole"]["_stolen_because"], "force")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
