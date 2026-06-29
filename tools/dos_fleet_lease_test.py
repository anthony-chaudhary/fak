#!/usr/bin/env python3
"""Hermetic tests for tools/dos_fleet_lease.py — the multi-node lease transport.

Run natively — `python tools/dos_fleet_lease_test.py` — NOT through WSL. The WDAC
policy that blocks freshly-built unsigned Go test binaries (see CLAUDE.md) does not
touch the system Python interpreter. No real `dos`/git calls: the kernel `live`/
`acquire` seams are injected fakes, the FleetStore is a per-test tempfile LocalDirStore,
and every time-dependent function takes an injected `now`. Mirrors release_lock_test.py.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dos_fleet_lease as fl  # noqa: E402


def rec(lane: str, *, host: str, mode: str = "exclusive", holder: str = "h",
        heartbeat_at: str | float | None = None, acquired_at: str = "2026-06-20T00:00:00Z",
        tree: list[str] | None = None) -> dict:
    r = {"lane": lane, "host_id": host, "mode": mode, "holder": holder,
         "acquired_at": acquired_at, "tree": tree or [f"{lane}/**"]}
    if heartbeat_at is not None:
        r["heartbeat_at"] = heartbeat_at
    return r


T0 = 1_750_000_000.0  # a fixed injected "now" base (epoch seconds)


class ForeignLivenessTest(unittest.TestCase):
    def test_fresh_heartbeat_is_live(self):
        r = rec("gateway", host="B", heartbeat_at=T0 - 10)
        self.assertTrue(fl.foreign_record_live(r, now=T0, ttl_s=900, skew_margin_s=30))

    def test_past_ttl_plus_skew_is_dead(self):
        r = rec("gateway", host="B", heartbeat_at=T0 - 1000)  # 1000 > 900 + 30
        self.assertFalse(fl.foreign_record_live(r, now=T0, ttl_s=900, skew_margin_s=30))

    def test_within_skew_margin_is_still_live(self):
        # 905s old, ttl 900, skew 30 -> 905 < 930 -> live (skew tolerance).
        r = rec("gateway", host="B", heartbeat_at=T0 - 905)
        self.assertTrue(fl.foreign_record_live(r, now=T0, ttl_s=900, skew_margin_s=30))

    def test_iso_stamp_is_parsed(self):
        r = rec("gateway", host="B", heartbeat_at="2026-06-20T00:00:00Z")
        now = fl._epoch_seconds("2026-06-20T00:05:00Z")
        self.assertTrue(fl.foreign_record_live(r, now=now, ttl_s=900, skew_margin_s=30))
        now_far = fl._epoch_seconds("2026-06-20T01:00:00Z")  # 3600s later
        self.assertFalse(fl.foreign_record_live(r, now=now_far, ttl_s=900, skew_margin_s=30))

    def test_falls_back_to_published_then_acquired(self):
        r = {"lane": "x", "host_id": "B", "acquired_at": T0 - 10}  # no heartbeat/published
        self.assertTrue(fl.foreign_record_live(r, now=T0, ttl_s=900, skew_margin_s=30))

    def test_unreadable_record_is_not_live(self):
        self.assertFalse(fl.foreign_record_live({"_unreadable": True}, now=T0, ttl_s=900, skew_margin_s=30))

    def test_no_usable_timestamp_defaults_live(self):
        # Refuse to silently steal a lane whose record we just couldn't time-parse.
        self.assertTrue(fl.foreign_record_live({"lane": "x"}, now=T0, ttl_s=900, skew_margin_s=30))


class UnionTest(unittest.TestCase):
    def test_local_records_pass_through(self):
        local = [rec("docs", host="A")]
        u = fl.union_live_leases(local=local, fleet={}, self_host="A", now=T0)
        self.assertEqual([r["lane"] for r in u["leases"]], ["docs"])

    def test_foreign_live_lease_is_unioned(self):
        fleet = {"B": {"host_id": "B", "leases": [rec("gateway", host="B", heartbeat_at=T0 - 5)]}}
        u = fl.union_live_leases(local=[rec("docs", host="A")], fleet=fleet, self_host="A", now=T0)
        lanes = sorted(r["lane"] for r in u["leases"])
        self.assertEqual(lanes, ["docs", "gateway"])

    def test_foreign_stale_lease_is_dropped(self):
        fleet = {"B": {"host_id": "B", "leases": [rec("gateway", host="B", heartbeat_at=T0 - 5000)]}}
        u = fl.union_live_leases(local=[], fleet=fleet, self_host="A", now=T0)
        self.assertEqual(u["leases"], [])
        self.assertEqual(u["dropped"], [{"host_id": "B", "lane": "gateway"}])

    def test_own_published_copy_is_skipped(self):
        # Our own snapshot in the fleet store is redundant with the local WAL.
        fleet = {"A": {"host_id": "A", "leases": [rec("docs", host="A", heartbeat_at=T0)]}}
        u = fl.union_live_leases(local=[rec("docs", host="A")], fleet=fleet, self_host="A", now=T0)
        self.assertEqual(len([r for r in u["leases"] if r["lane"] == "docs"]), 1)

    def test_unreadable_snapshot_is_surfaced(self):
        fleet = {"B": {"host_id": "B", "_unreadable": True}}
        u = fl.union_live_leases(local=[], fleet=fleet, self_host="A", now=T0)
        self.assertEqual(u["unreadable"], ["B"])


class WouldCollideTest(unittest.TestCase):
    def test_same_lane_name_collides(self):
        out = fl.would_collide("gateway", ["fak/internal/gateway/**"], [rec("gateway", host="B")])
        self.assertTrue(out["collides"])
        self.assertEqual(out["reason"], fl.REASON_HELD_REMOTE)
        self.assertEqual(out["host_id"], "B")

    def test_exclusive_tree_overlap_collides(self):
        live = [rec("gw", host="B", tree=["fak/internal/gateway/**"])]
        out = fl.would_collide("other", ["fak/internal/gateway/sub/**"], live)
        self.assertTrue(out["collides"])

    def test_disjoint_does_not_collide(self):
        live = [rec("gw", host="B", tree=["fak/internal/gateway/**"])]
        out = fl.would_collide("compute", ["fak/internal/compute/**"], live)
        self.assertFalse(out["collides"])

    def test_shared_mode_does_not_block_on_tree(self):
        # A shared-mode holder doesn't trip the tree-overlap rung (only same-name).
        live = [rec("gw", host="B", mode="shared", tree=["fak/internal/gateway/**"])]
        out = fl.would_collide("other", ["fak/internal/gateway/sub/**"], live)
        self.assertFalse(out["collides"])

    def test_pre_filter_only_refuses_never_grants(self):
        # Empty live set -> no collision, but that is "ask the arbiter", not "go".
        out = fl.would_collide("gateway", ["fak/internal/gateway/**"], [])
        self.assertFalse(out["collides"])  # absence of collision != a grant


class SnapshotTest(unittest.TestCase):
    def test_epoch_is_monotone(self):
        s1 = fl.build_snapshot(host="A", leases=[], now=T0, prev_epoch=None)
        s2 = fl.build_snapshot(host="A", leases=[], now=T0, prev_epoch=s1["fleet_epoch"])
        self.assertEqual(s1["fleet_epoch"], 1)
        self.assertEqual(s2["fleet_epoch"], 2)

    def test_records_get_published_and_heartbeat_stamps(self):
        s = fl.build_snapshot(host="A", leases=[rec("docs", host="A")], now=T0, prev_epoch=None)
        r = s["leases"][0]
        self.assertIn("published_at", r)
        self.assertIn("heartbeat_at", r)
        self.assertEqual(r["host_id"], "A")


class LocalDirStoreCASTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = fl.LocalDirStore(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_first_write_with_none_expected_succeeds(self):
        snap = fl.build_snapshot(host="A", leases=[], now=T0, prev_epoch=None)
        ok, observed = self.store.compare_and_swap("A", None, snap)
        self.assertTrue(ok)
        self.assertEqual(observed, 1)

    def test_stale_expected_is_rejected(self):
        s1 = fl.build_snapshot(host="A", leases=[], now=T0, prev_epoch=None)
        self.store.compare_and_swap("A", None, s1)  # store now at epoch 1
        s_bad = fl.build_snapshot(host="A", leases=[], now=T0, prev_epoch=5)  # claims prev was 5
        ok, observed = self.store.compare_and_swap("A", 5, s_bad)  # but store is at 1
        self.assertFalse(ok)
        self.assertEqual(observed, 1)  # the TOCTOU guard the git backend inherits

    def test_correct_expected_succeeds(self):
        s1 = fl.build_snapshot(host="A", leases=[], now=T0, prev_epoch=None)
        self.store.compare_and_swap("A", None, s1)
        s2 = fl.build_snapshot(host="A", leases=[], now=T0, prev_epoch=1)
        ok, observed = self.store.compare_and_swap("A", 1, s2)
        self.assertTrue(ok)
        self.assertEqual(observed, 2)

    def test_corrupt_snapshot_is_surfaced_unreadable(self):
        (Path(self._tmp.name) / "B.json").write_text("{not json", encoding="utf-8")
        fleet = self.store.read_fleet()
        self.assertTrue(fleet["B"].get("_unreadable"))

    def test_read_fleet_keys_by_host_id(self):
        self.store.compare_and_swap("A", None, fl.build_snapshot(host="A", leases=[], now=T0, prev_epoch=None))
        self.store.compare_and_swap("B", None, fl.build_snapshot(host="B", leases=[], now=T0, prev_epoch=None))
        self.assertEqual(sorted(self.store.read_fleet().keys()), ["A", "B"])


class VerbWiringTest(unittest.TestCase):
    """do_* verbs with injected kernel seams + a real LocalDirStore. No subprocess."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = fl.LocalDirStore(Path(self._tmp.name))
        self.ws = Path("C:/work/fleet")

    def tearDown(self):
        self._tmp.cleanup()

    def test_materialize_local_ignores_store(self):
        # scope=local must read ONLY the local WAL — the shard-by-node fast path.
        def live(ws):
            return [rec("docs", host="A")]
        # Even if the store has a foreign lease, local scope (store=None) never sees it.
        payload, code = fl.do_materialize(self.ws, None, scope="local", host="A", now=T0, live=live)
        self.assertEqual(code, fl.EXIT_OK)
        self.assertEqual(payload["scope"], "local")
        self.assertEqual([r["lane"] for r in payload["leases"]], ["docs"])

    def test_materialize_global_unions_foreign(self):
        self.store.compare_and_swap(
            "B", None, fl.build_snapshot(host="B", leases=[rec("gateway", host="B", heartbeat_at=T0)], now=T0, prev_epoch=None))
        def live(ws):
            return [rec("docs", host="A")]
        payload, code = fl.do_materialize(self.ws, self.store, scope="global", host="A", now=T0, live=live)
        self.assertEqual(code, fl.EXIT_OK)
        self.assertEqual(sorted(r["lane"] for r in payload["leases"]), ["docs", "gateway"])

    def test_publish_writes_local_held_set(self):
        def live(ws):
            return [rec("docs", host="A", heartbeat_at=T0)]
        payload, code = fl.do_publish(self.ws, self.store, host="A", now=T0, live=live)
        self.assertEqual(code, fl.EXIT_OK)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["published"], 1)
        self.assertEqual(self.store.read_fleet()["A"]["leases"][0]["lane"], "docs")

    def test_two_node_acquire_is_blocked_by_pre_filter(self):
        # Node B published a live `gateway` lease; node A tries to acquire gateway.
        self.store.compare_and_swap(
            "B", None, fl.build_snapshot(host="B", leases=[rec("gateway", host="B", heartbeat_at=T0)], now=T0, prev_epoch=None))
        called = {"acquire": False}

        def fake_live(ws):
            return []  # A holds nothing locally

        def fake_acquire(ws, **kw):
            called["acquire"] = True
            return {"outcome": "acquire", "lane": kw["lane"], "_returncode": 0}

        payload, code = fl.do_acquire(
            self.ws, self.store, lane="gateway", owner="A", kind="cluster", mode="exclusive",
            run_id="", scope="global", host="A", now=T0, live=fake_live, acquire=fake_acquire)
        self.assertEqual(code, fl.EXIT_DENIED)
        self.assertEqual(payload["reason"], fl.REASON_HELD_REMOTE)
        self.assertEqual(payload["host_id"], "B")
        self.assertFalse(called["acquire"])  # short-circuited before the kernel round-trip

    def test_acquire_succeeds_when_disjoint_and_republishes(self):
        self.store.compare_and_swap(
            "B", None, fl.build_snapshot(host="B", leases=[rec("gateway", host="B", heartbeat_at=T0)], now=T0, prev_epoch=None))

        def fake_live(ws):
            return []

        def fake_acquire(ws, **kw):
            # Kernel admits the disjoint lane; the union it received included B's gateway.
            return {"outcome": "acquire", "lane": kw["lane"], "_returncode": 0,
                    "_leases_seen": [r["lane"] for r in kw["leases"]]}

        payload, code = fl.do_acquire(
            self.ws, self.store, lane="compute", owner="A", kind="cluster", mode="exclusive",
            run_id="", scope="global", host="A", now=T0, live=fake_live, acquire=fake_acquire)
        self.assertEqual(code, fl.EXIT_OK)
        self.assertEqual(payload["lane"], "compute")
        # The kernel saw B's foreign gateway lease in the union it was handed.
        self.assertIn("gateway", payload["kernel"]["_leases_seen"])

    def test_acquire_when_kernel_autopicks_away_is_denied(self):
        def fake_live(ws):
            return []

        def fake_acquire(ws, **kw):
            # Kernel auto-picked a different lane (requested was busy in the union).
            return {"outcome": "acquire", "lane": "adjudicator", "auto_picked": True, "_returncode": 0}

        # Requested gateway, but no pre-filter hit (empty same-lane set) so the kernel
        # is consulted; it auto-picks away -> we report the requested lane as denied.
        payload, code = fl.do_acquire(
            self.ws, self.store, lane="gateway", owner="A", kind="cluster", mode="exclusive",
            run_id="", scope="local", host="A", now=T0, live=fake_live, acquire=fake_acquire)
        # scope=local -> store=None, pre-filter sees only local (empty) -> kernel consulted.
        # Kernel returned a DIFFERENT lane than requested -> denied for the request.
        self.assertEqual(code, fl.EXIT_DENIED)
        self.assertEqual(payload["reason"], fl.REASON_HELD_REMOTE)


@unittest.skipUnless(shutil.which("git"), "git required for GitRefStore transport tests")
class GitRefStoreCASTest(unittest.TestCase):
    """Integration tests for the git-ref CAS transport against a throwaway bare
    'remote' (NOT a fake — GitRefStore IS the git plumbing, so a real per-ref push
    is the unit under test). Each test builds its own bare remote + working clone(s)
    under a tempdir; nothing touches the fleet repo or its origin.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.remote = self.base / "remote.git"
        self._run(["git", "init", "-q", "--bare", str(self.remote)])
        self.storeA = fl.GitRefStore(self._work("A"), remote=str(self.remote))

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, cmd):
        subprocess.run(cmd, cwd=str(self.base), capture_output=True, text=True, check=True)

    def _work(self, name: str) -> Path:
        w = self.base / name
        self._run(["git", "init", "-q", str(w)])
        return w

    def _snap(self, host, prev, leases=None):
        return fl.build_snapshot(host=host, leases=leases or [], now=T0, prev_epoch=prev)

    def test_first_create_succeeds_and_read_returns_content(self):
        ok, observed = self.storeA.compare_and_swap("A", None, self._snap("A", None, [rec("docs", host="A")]))
        self.assertTrue(ok)
        self.assertEqual(observed, 1)
        fleet = self.storeA.read_fleet()
        self.assertIn("A", fleet)
        self.assertEqual(fleet["A"]["leases"][0]["lane"], "docs")

    def test_create_over_existing_is_rejected(self):
        # Node A creates the ref; a second node's create of the SAME host ref with a
        # DIFFERENT held set must lose — its distinct blob is a non-fast-forward over
        # the existing ref, so the non-force push is rejected. (Identical content
        # would be a benign git no-op, which never happens in the real do_publish flow
        # because fleet_epoch is monotone, so each republish differs.)
        self.assertTrue(self.storeA.compare_and_swap("A", None, self._snap("A", None))[0])
        storeB = fl.GitRefStore(self._work("B"), remote=str(self.remote))
        ok, _ = storeB.compare_and_swap("A", None, self._snap("A", None, [rec("docs", host="B")]))
        self.assertFalse(ok)

    def test_correct_expected_updates(self):
        self.storeA.compare_and_swap("A", None, self._snap("A", None))  # epoch 1
        self.storeA.read_fleet()  # the real do_publish flow: read (cache oid), then CAS
        ok, observed = self.storeA.compare_and_swap("A", 1, self._snap("A", 1))
        self.assertTrue(ok)
        self.assertEqual(observed, 2)

    def test_stale_lease_is_rejected(self):
        # Two nodes both observe epoch 1; A advances the remote to epoch 2; B's update
        # carrying its now-stale ref oid must be refused AT THE REMOTE by --force-with-lease.
        self.storeA.compare_and_swap("A", None, self._snap("A", None))  # remote -> oid(epoch1)
        storeB = fl.GitRefStore(self._work("B"), remote=str(self.remote))
        storeB.read_fleet()  # B caches the epoch-1 oid
        self.assertTrue(self.storeA.compare_and_swap("A", 1, self._snap("A", 1))[0])  # remote -> oid(epoch2)
        ok, observed = storeB.compare_and_swap("A", 1, self._snap("A", 1, [rec("docs", host="B")]))
        self.assertFalse(ok)
        self.assertEqual(observed, 2)  # B re-reads the loser-report and sees A's epoch 2

    def test_corrupt_blob_is_surfaced_unreadable(self):
        oid = subprocess.run(
            ["git", "hash-object", "-w", "--stdin"], cwd=str(self.storeA.root),
            input="{not json", capture_output=True, text=True, check=True).stdout.strip()
        self._run(["git", "-C", str(self.storeA.root), "push", str(self.remote),
                   f"{oid}:refs/dos-fleet-leases/A"])
        fleet = self.storeA.read_fleet()
        self.assertTrue(fleet["A"].get("_unreadable"))


class GlobalLaneTest(unittest.TestCase):
    def test_exclusive_lanes_are_global(self):
        for lane in ("abi", "release", "global"):
            self.assertTrue(fl.is_global_lane(lane))

    def test_ordinary_lane_is_not_global(self):
        for lane in ("gateway", "tools", "docs", "compute"):
            self.assertFalse(fl.is_global_lane(lane))


class StoreSelectionTest(unittest.TestCase):
    """The activation seam (#21): global lanes default to the cross-node GitRefStore,
    shard lanes keep the zero-network LocalDirStore, and an explicit store still wins.
    No network — GitRefStore.__init__ only records root/remote (no git call)."""

    def _ns(self, store: str = "") -> argparse.Namespace:
        return argparse.Namespace(store=store)

    def setUp(self):
        # Pin the env override OFF so the default-selection branch is what we exercise.
        self._env = mock.patch.dict("os.environ", {}, clear=False)
        self._env.start()
        import os
        os.environ.pop("FAK_FLEET_LEASE_STORE", None)
        self.root = fl.repo_root()

    def tearDown(self):
        self._env.stop()

    def test_global_lane_defaults_to_git(self):
        for lane in fl.GLOBAL_LANES:
            store = fl._store_for(self._ns(), self.root, lane=lane)
            self.assertIsInstance(store, fl.GitRefStore, f"{lane} must default to git")

    def test_global_scope_defaults_to_git(self):
        # An explicit `materialize --scope global` on no particular lane still goes cross-node.
        store = fl._store_for(self._ns(), self.root, scope="global")
        self.assertIsInstance(store, fl.GitRefStore)

    def test_shard_lane_defaults_to_local(self):
        for lane in ("tools", "docs", "gateway", ""):
            store = fl._store_for(self._ns(), self.root, lane=lane)
            self.assertIsInstance(store, fl.LocalDirStore, f"{lane!r} must stay local")

    def test_explicit_git_flag_forces_git_on_a_shard_lane(self):
        store = fl._store_for(self._ns(store="git"), self.root, lane="tools")
        self.assertIsInstance(store, fl.GitRefStore)

    def test_explicit_path_store_overrides_global_default(self):
        # A test/dev can still force LocalDirStore even on a global lane.
        with tempfile.TemporaryDirectory() as d:
            store = fl._store_for(self._ns(store=d), self.root, lane="release")
            self.assertIsInstance(store, fl.LocalDirStore)

    def test_env_var_git_forces_git(self):
        import os
        os.environ["FAK_FLEET_LEASE_STORE"] = "git"
        store = fl._store_for(self._ns(), self.root, lane="docs")
        self.assertIsInstance(store, fl.GitRefStore)


if __name__ == "__main__":
    unittest.main()
