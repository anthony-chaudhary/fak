#!/usr/bin/env python3
"""Tests for tools/dispatch_status_leases.py (split from dispatch_status_test.py)."""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
ROOT = _TEST_DIR.parent
_SHARED_CACHE: list[object] = []


def _shared():
    if not _SHARED_CACHE:
        spec = importlib.util.spec_from_file_location(
            "dispatch_status_test_helpers", _TEST_DIR / "dispatch_status_test.py"
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SHARED_CACHE.append(mod)
    return _SHARED_CACHE[0]


def load():
    return _shared().load()


def build(*args, **kwargs):
    return _shared().build(*args, **kwargs)


class DispatchStatusLeasesModuleTest(unittest.TestCase):
    def test_direct_module_import_exports(self) -> None:
        tools_dir = str(_TEST_DIR)
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        mod = importlib.import_module("dispatch_status_leases")
        self.assertEqual(mod.LANE_HOLDER_LIVE, "live")
        self.assertTrue(mod.trees_overlap(["tools/a.py"], ["tools/"]))


class RunStatusDigestTest(unittest.TestCase):
    def test_loop_ledger_run_ids_are_recent_active_started_rids_only(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            ledger = Path(d) / "loops.jsonl"
            now_ns = 10_000_000_000_000
            recent_ns = now_ns - 30 * 60 * 1_000_000_000
            stale_ns = now_ns - 3 * 60 * 60 * 1_000_000_000
            ledger.write_text("\n".join([
                '{"loop_id":"issue-resolve-dispatch/claude","run_id":"legacy-1"}',
                '{"loop_id":"other","run_id":"RID-OTHER1"}',
                '{"loop_id":"issue-resolve-progress","run_id":"RID-PROGRESS1","kind":"fire"}',
                '{"loop_id":"issue-resolve-progress","run_id":"RID-PROGRESS1","kind":"end","status":"claimed_done"}',
                '{"loop_id":"issue-resolve-dispatch/codex","run_id":"RID-DRYRUN1","kind":"fire"}',
                '{"loop_id":"issue-resolve-dispatch/codex","run_id":"RID-DRYRUN1","kind":"admit","status":"admitted"}',
                '{"loop_id":"issue-resolve-dispatch/codex","run_id":"RID-DRYRUN1","kind":"end","status":"claimed_done"}',
                '{"loop_id":"issue-resolve-dispatch/claude","run_id":"RID-REFUSED1","kind":"fire"}',
                '{"loop_id":"issue-resolve-dispatch/claude","run_id":"RID-REFUSED1","kind":"admit","status":"refused"}',
                '{"loop_id":"issue-resolve-dispatch/opencode","run_id":"RID-ACTIVE1","kind":"fire"}',
                f'{{"loop_id":"issue-resolve-dispatch/opencode","run_id":"RID-ACTIVE1","kind":"start","status":"running","ts_unix_nano":{recent_ns}}}',
                f'{{"loop_id":"issue-resolve-dispatch/claude","run_id":"RID-STALE1","kind":"start","status":"running","ts_unix_nano":{stale_ns}}}',
                f'{{"loop_id":"issue-resolve-dispatch/claude","run_id":"RID-ACTIVE2","kind":"start","status":"running","ts_unix_nano":{recent_ns + 1}}}',
            ]) + "\n", encoding="utf-8")
            self.assertEqual(
                mod.run_ids_from_loop_ledger(ledger, lookback_min=60, now_ns=now_ns),
                ["RID-ACTIVE2", "RID-ACTIVE1"])

    def test_claimed_key_detector_is_recursive(self) -> None:
        mod = load()
        self.assertTrue(mod.has_key_named({"liveness": [{"claimed": "done"}]}, "claimed"))
        self.assertFalse(mod.has_key_named({"liveness": [{"verdict": "STALLED"}]}, "claimed"))

    def test_build_payload_summarizes_dos_status_digests(self) -> None:
        mod = load()
        p = build(mod, run_status=[
            {"run_id": "RID-DISPATCH1", "liveness": {"verdict": "ADVANCING"}},
            {"run_id": "RID-PROGRESS1", "_error": "dos unavailable"},
        ])
        self.assertEqual(p["run_status"]["source"], "dos status")
        self.assertEqual(p["run_status"]["count"], 2)
        self.assertEqual(p["run_status"]["liveness"], {"ADVANCING": 1})
        self.assertEqual(p["run_status"]["errors"], 1)
        self.assertTrue(any("dos status digest" in r for r in p["reasons"]))
        self.assertIn("run truth", mod.render(p))


class LeaseStateTest(unittest.TestCase):
    def _backlog(self) -> dict:
        return {
            "lanes": {
                "tools": {"tree": ["tools/**"], "issues": [1762]},
            },
            "issues": [
                {"number": 1762, "lane": "tools", "confidence": "path-confirmed"},
            ],
        }

    @staticmethod
    def _swap_run(mod, fake):
        """Temporarily replace mod.subprocess.run; returns a restore thunk."""
        orig = mod.subprocess.run
        mod.subprocess.run = fake
        return lambda: setattr(mod.subprocess, "run", orig)

    def test_read_leaseref_returns_triple_when_git_read_fails(self) -> None:
        # The git-error return paths must carry the SAME 3-tuple arity as the
        # happy path. When `git for-each-ref` returns non-zero, a 2-tuple return
        # makes read_lease_state's `records, sessions, err = ...` unpack raise
        # ValueError, which crashes the whole status-card render (froze the
        # operator card for ~3 days, 2026-07-02..05).
        mod = load()

        class _Boom:
            returncode = 128
            stdout = ""
            stderr = "fatal: not a git repository"

        restore = self._swap_run(mod, lambda *a, **k: _Boom())
        try:
            triple = mod.read_leaseref_records_and_sessions(Path("."))
        finally:
            restore()
        self.assertEqual(len(triple), 3)
        records, sessions, err = triple  # must not raise
        self.assertEqual(records, [])
        self.assertEqual(sessions, {})
        self.assertIn("not a git repository", err)

    def test_read_leaseref_returns_triple_on_git_oserror(self) -> None:
        mod = load()

        def _raise(*a, **k):
            raise OSError("git binary missing")

        restore = self._swap_run(mod, _raise)
        try:
            triple = mod.read_leaseref_records_and_sessions(Path("."))
        finally:
            restore()
        self.assertEqual(len(triple), 3)
        _records, sessions, err = triple
        self.assertEqual(sessions, {})
        self.assertIn("git binary missing", err)

    def test_read_lease_state_survives_git_read_failure(self) -> None:
        # End-to-end: the caller that actually crashed must degrade to a
        # read_error dict, not raise, when the lease-ref read fails.
        mod = load()

        class _Boom:
            returncode = 128
            stdout = ""
            stderr = "fatal: not a git repository"

        restore = self._swap_run(mod, lambda *a, **k: _Boom())
        try:
            state = mod.read_lease_state(Path("."), self._backlog())
        finally:
            restore()
        self.assertIn("read_error", state)
        self.assertEqual(state["active_count"], 0)
        self.assertEqual(state["blocking_count"], 0)

    def test_summarize_leases_marks_blocking_and_non_blocking_active_leases(self) -> None:
        mod = load()
        state = mod.summarize_leases([
            {
                "id": "resolve-tools-1762",
                "tree_globs": ["tools/**"],
                "holder": "peer-a",
                "acquired_unix": 1_000,
                "ttl_seconds": 3_600,
            },
            {
                "id": "resolve-model-1700",
                "tree_globs": ["internal/model/**"],
                "holder": "peer-b",
                "acquired_unix": 1_100,
                "ttl_seconds": 3_600,
            },
            {
                "id": "resolve-docs",
                "tree_globs": ["docs/**"],
                "holder": "old-peer",
                "acquired_unix": 100,
                "ttl_seconds": 10,
            },
        ], self._backlog(), now_ts=1_600)

        self.assertEqual(state["active_count"], 2)
        self.assertEqual(state["expired_count"], 1)
        self.assertEqual(state["blocking_count"], 1)
        by_id = {row["id"]: row for row in state["active"]}
        self.assertTrue(by_id["resolve-tools-1762"]["blocks_candidate"])
        self.assertEqual(by_id["resolve-tools-1762"]["lane"], "tools")
        self.assertEqual(by_id["resolve-tools-1762"]["age_min"], 10.0)
        self.assertEqual(
            by_id["resolve-tools-1762"]["blocking_candidates"][0]["issue"], 1762)
        self.assertFalse(by_id["resolve-model-1700"]["blocks_candidate"])
        self.assertEqual(state["expired"][0]["id"], "resolve-docs")

    def test_candidate_blocking_is_unknown_when_backlog_fold_unavailable(self) -> None:
        mod = load()
        state = mod.summarize_leases([
            {
                "id": "resolve-tools-1762",
                "tree_globs": ["tools/**"],
                "holder": "peer-a",
                "acquired_unix": 1_000,
                "ttl_seconds": 3_600,
            },
        ], {"_skipped": "fast"}, now_ts=1_600)

        self.assertFalse(state["candidate_source_available"])
        self.assertIsNone(state["active"][0]["blocks_candidate"])

    def test_summarize_leases_annotates_session_liveness(self) -> None:
        mod = load()
        state = mod.summarize_leases([
            {
                "id": "resolve-docs",
                "tree_globs": ["docs/**"],
                "holder": "node-a/s-live",
                "session_id": "s-live",
                "acquired_unix": 1_000,
                "ttl_seconds": 3_600,
            },
            {
                "id": "resolve-tools",
                "tree_globs": ["tools/**"],
                "holder": "node-a/s-stopped",
                "session_id": "s-stopped",
                "acquired_unix": 1_000,
                "ttl_seconds": 3_600,
            },
            {
                "id": "resolve-gateway",
                "tree_globs": ["internal/gateway/**"],
                "holder": "legacy",
                "acquired_unix": 1_000,
                "ttl_seconds": 3_600,
            },
        ], self._backlog(), sessions={
            "s-live": {"id": "s-live", "pcb_state": "RUNNING",
                       "updated_at": 1_500, "ttl_seconds": 600},
            "s-stopped": {"id": "s-stopped", "pcb_state": "STOPPED",
                          "updated_at": 1_500, "ttl_seconds": 600},
        }, now_ts=1_600)

        by_id = {row["id"]: row for row in state["active"]}
        self.assertEqual(by_id["resolve-docs"]["liveness"], "peer-live")
        self.assertFalse(by_id["resolve-docs"]["reclaimable"])
        self.assertEqual(by_id["resolve-docs"]["session_id"], "s-live")
        self.assertEqual(by_id["resolve-tools"]["liveness"], "peer-dead")
        self.assertTrue(by_id["resolve-tools"]["reclaimable"])
        self.assertEqual(by_id["resolve-gateway"]["liveness"], "peer-unknown")
        self.assertFalse(by_id["resolve-gateway"]["reclaimable"])

    def test_summarize_leases_counts_stranded_vs_live_blocking(self) -> None:
        # #4324: LANE_LEASE_HELD refusals against a dead holder ("phantom"
        # collisions) must be countable separately from refusals against a
        # genuinely live peer, so a release-on-exit fix's effect is measurable.
        mod = load()
        state = mod.summarize_leases([
            {
                "id": "resolve-tools-1762",
                "tree_globs": ["tools/**"],
                "holder": "node-a/s-stopped",
                "session_id": "s-stopped",
                "acquired_unix": 1_000,
                "ttl_seconds": 3_600,
            },
            {
                "id": "resolve-tools-1763",
                "tree_globs": ["tools/**"],
                "holder": "node-a/s-live",
                "session_id": "s-live",
                "acquired_unix": 1_000,
                "ttl_seconds": 3_600,
            },
            {
                "id": "resolve-model-1700",
                "tree_globs": ["internal/model/**"],
                "holder": "peer-b",
                "acquired_unix": 1_100,
                "ttl_seconds": 3_600,
            },
        ], self._backlog(), sessions={
            "s-stopped": {"id": "s-stopped", "pcb_state": "STOPPED",
                          "updated_at": 1_500, "ttl_seconds": 600},
            "s-live": {"id": "s-live", "pcb_state": "RUNNING",
                       "updated_at": 1_500, "ttl_seconds": 600},
        }, now_ts=1_600)

        self.assertEqual(state["blocking_count"], 2)
        self.assertEqual(state["blocking_stranded_count"], 1)
        self.assertEqual(state["blocking_live_count"], 1)

    def test_age_text_formats_minutes_and_hours(self) -> None:
        mod = load()
        self.assertEqual(mod._age_text(10.0), "10m")
        self.assertEqual(mod._age_text(120.0), "2h")


class WorkerLeaseCrossCheckTest(unittest.TestCase):
    def _leases(self) -> dict:
        return {
            "active": [
                {"id": "resolve-tools-1765", "lane": "tools", "holder": "peer-a"},
                {"id": "resolve-model-1700", "lane": "model", "holder": "peer-b"},
            ],
        }

    def test_cross_check_buckets_clean_orphan_process_and_orphan_lease(self) -> None:
        mod = load()
        workers = {
            "available": True,
            "workers": [
                {
                    "worker": "resolve-1765-20260701-120000",
                    "issue": 1765,
                    "pid": 111,
                    "lane": "tools",
                    "backend": "opencode",
                    "lease_id": "resolve-tools-1765",
                },
                {
                    "worker": "resolve-1770-20260701-120001",
                    "issue": 1770,
                    "pid": 222,
                    "lane": "docs",
                    "backend": "claude",
                    "lease_id": "resolve-docs-1770",
                },
            ],
        }

        out = mod.cross_check_worker_leases(workers, self._leases())

        self.assertEqual(out["clean_count"], 1)
        self.assertEqual(out["orphan_process_count"], 1)
        self.assertEqual(out["orphan_lease_count"], 1)
        self.assertEqual(out["clean"][0]["worker"]["worker"], "resolve-1765-20260701-120000")
        self.assertEqual(out["orphan_process"][0]["worker"]["worker"], "resolve-1770-20260701-120001")
        self.assertEqual(out["orphan_lease"][0]["lease"]["id"], "resolve-model-1700")
        summary = mod._active_worker_summary(out)
        self.assertIn("#1765 tools/opencode pid=111 lease=resolve-tools-1765", summary)
        self.assertIn("#1770 docs/claude pid=222 lease=resolve-docs-1770", summary)

    def test_payload_render_surfaces_active_resolver_workers(self) -> None:
        mod = load()
        worker_leases = {
            "available": True,
            "clean_count": 1,
            "orphan_process_count": 0,
            "orphan_lease_count": 0,
            "clean": [{
                "worker": {
                    "worker": "resolve-2718-20260705-183239",
                    "issue": 2718,
                    "pid": 70232,
                    "lane": "tools",
                    "backend": "opencode",
                    "lease_id": "resolve-tools",
                },
                "lease": {"id": "resolve-tools"},
            }],
            "orphan_process": [],
            "orphan_lease": [],
        }
        p = build(mod, worker_leases=worker_leases)
        self.assertTrue(any("#2718 tools/opencode pid=70232" in r
                            for r in p["reasons"]))
        self.assertIn("live work : #2718 tools/opencode pid=70232", mod.render(p))
        self.assertIn("**active resolver workers**: #2718 tools/opencode pid=70232",
                      mod.render_md(p, date="2026-07-05"))
        self.assertTrue(any("#2718 tools/opencode pid=70232" in r
                            for r in mod._dispatch_slack_buckets(p)["auto-solving"]))

    def test_cross_check_preserves_unavailable_liveness(self) -> None:
        mod = load()
        out = mod.cross_check_worker_leases(
            {"available": False, "error": "psutil unavailable"}, self._leases())
        self.assertFalse(out["available"])
        self.assertEqual(out["orphan_process_count"], 0)
        self.assertEqual(out["orphan_lease_count"], 0)


# Windows FILETIME (100ns ticks since 1601-01-01) for a known Unix epoch second.
def _filetime(epoch_s: float) -> int:
    return int((epoch_s + 11644473600.0) * 1e7)


class LaneLeaseHolderLivenessTest(unittest.TestCase):
    """The cross-check must cover ALL lane leases, and judge holders on
    (pid, proc_starttime) — never on bare pid existence (#5859).

    Regression guard for the real defect: `dos lease-lane live` returned 24 exclusive
    lane leases, 18 held by processes that no longer existed (the oldest 22 days old,
    fencing `cmd/**` and `internal/modver/**`), while the card printed
    `lease chk : clean=3 orphan-process=0 unmatched-live-lease=0` and folded
    "worker/lease cross-check clean" into its summary. The cross-check only ever saw
    the `refs/fak/locks/*` resolver leases, a different substrate entirely.
    """

    NOW = 1786080000.0
    HOST = "fleet-box"

    def _lane_records(self) -> list[dict]:
        """One live holder, one dead (pid gone), one dead by PID RECYCLING."""
        return [
            {"lane": "docs", "holder": "worker-live", "host_id": self.HOST,
             "mode": "exclusive", "pid": 1001, "tree": ["docs/**"],
             "acquired_at": "2026-08-06T12:00:00Z",
             "proc_starttime": _filetime(self.NOW - 600)},
            {"lane": "cmd", "holder": "claude-5031", "host_id": self.HOST,
             "mode": "exclusive", "pid": 51980, "tree": ["cmd/**"],
             "acquired_at": "2026-07-19T09:00:00Z",
             "proc_starttime": _filetime(self.NOW - 1_600_000)},
            {"lane": "guard", "holder": "worker-2461", "host_id": self.HOST,
             "mode": "exclusive", "pid": 44396, "tree": ["internal/modver/**"],
             "acquired_at": "2026-08-06T11:00:00Z",
             "proc_starttime": _filetime(self.NOW - 4000)},
        ]

    def _starts(self) -> dict[int, float]:
        """The live process table. pid 51980 is GONE. pid 44396 EXISTS but was
        recycled — it started long after the lease recorded it."""
        return {1001: self.NOW - 600, 44396: self.NOW - 100, 2: self.NOW}

    def _fold(self, **over):
        mod = load()
        kwargs = {"starts": self._starts(), "host_id": self.HOST, "now_ts": self.NOW}
        kwargs.update(over)
        return mod, mod.summarize_lane_lease_holders(self._lane_records(), **kwargs)

    def test_dead_holder_counted_including_recycled_pid(self) -> None:
        _, lane = self._fold()
        self.assertTrue(lane["available"])
        self.assertEqual(lane["total"], 3)
        self.assertEqual(lane["live_count"], 1)
        # BOTH dead holders — the missing pid AND the recycled one. A bare
        # pid-existence check would score the recycled holder "alive" and report 1.
        self.assertEqual(lane["dead_count"], 2)
        dead_lanes = sorted(r["lane"] for r in lane["dead"])
        self.assertEqual(dead_lanes, ["cmd", "guard"])
        self.assertIn("RECYCLED", next(
            r["holder_evidence"] for r in lane["dead"] if r["lane"] == "guard"))
        # The action must never hand the operator a blind reap (#5859). The verb may
        # be NAMED — as the thing not to run — but never as a command to execute.
        action = lane["next_action"]
        self.assertIn("do NOT reap", action)
        self.assertNotIn("--owner <holder>", action)
        self.assertNotIn("release --lane", action)

    def test_pid_existence_alone_is_not_proof_of_life(self) -> None:
        """A live pid with no readable proc_starttime is UNKNOWN, never live."""
        mod = load()
        state, evidence = mod.classify_lane_lease_holder(
            {"lane": "cmd", "pid": 1001, "host_id": self.HOST}, self._starts(),
            host_id=self.HOST)
        self.assertEqual(state, "unknown")
        self.assertIn("pid existence alone is not proof of life", evidence)

    def test_verdict_is_not_clean_with_a_dead_holder(self) -> None:
        """THE defect, at the verdict seam: a cross-check whose local worker/lease
        pairs all match must still not read "clean" while a lane is fenced."""
        mod, lane = self._fold()
        out = mod.cross_check_worker_leases(
            {"available": True, "workers": []}, {"active": []}, lane)
        self.assertEqual(out["dead_holder_count"], 2)
        self.assertFalse(mod.lane_lease_verdict_clean(out))
        reasons = " | ".join(mod._worker_lease_reasons(out))
        self.assertNotIn("cross-check clean", reasons)
        self.assertIn("dead-holder=2", reasons)
        self.assertIn("next action:", reasons)

    def test_unreadable_lane_lease_set_is_not_clean(self) -> None:
        """An unread lease set is not a clean one — and neither is one this host has
        no liveness oracle for.

        Note the oracle case is now judged from the WAL, not the process table:
        TTL expiry is time evidence the lease record carries itself, so a host with
        no psutil can still prove a 452-hour-old lease stale. `available` stays
        False (the pid corroboration rung is blind), so the card still withholds
        "clean" — the fail-safe direction is preserved without pretending the WAL
        said nothing.
        """
        mod, lane = self._fold(read_error="dos: command not found")
        self.assertFalse(lane["available"])
        out = mod.cross_check_worker_leases({"available": True, "workers": []},
                                            {"active": []}, lane)
        self.assertFalse(mod.lane_lease_verdict_clean(out))
        no_oracle = mod.summarize_lane_lease_holders(
            self._lane_records(), starts=None, host_id=self.HOST, now_ts=self.NOW)
        self.assertFalse(no_oracle["available"])
        self.assertFalse(mod.lane_lease_verdict_clean(
            mod.cross_check_worker_leases({"available": True, "workers": []},
                                          {"active": []}, no_oracle)))
        # All three fixtures are hours-to-weeks past the 50m TTL, and that is WAL
        # evidence — no process probe involved.
        self.assertEqual(no_oracle["dead_count"], 3)
        self.assertEqual(no_oracle["live_count"], 0)
        for row in no_oracle["rows"]:
            self.assertIn("TTL EXPIRED", row["holder_evidence"])
            self.assertIn("psutil unavailable", row["holder_evidence"])

    def test_foreign_host_holder_is_unknown_not_dead(self) -> None:
        """This host cannot probe another host's pid — and must not call it dead."""
        mod = load()
        state, _ = mod.classify_lane_lease_holder(
            {"lane": "cmd", "pid": 51980, "host_id": "other-box"},
            self._starts(), host_id=self.HOST)
        self.assertEqual(state, "unknown")

    def test_proc_starttime_decodes_windows_filetime(self) -> None:
        mod = load()
        # The real record observed on the reference tree.
        self.assertAlmostEqual(
            mod._proc_starttime_epoch(134305505540666317), 1786076954.07, places=1)
        self.assertIsNone(mod._proc_starttime_epoch(None))
        self.assertIsNone(mod._proc_starttime_epoch(0))

    def test_read_lane_leases_parses_kernel_json(self) -> None:
        mod = load()
        seen = {}

        def runner(cmd, cwd):
            seen["cmd"], seen["cwd"] = cmd, cwd
            return (json.dumps(self._lane_records()), None)

        records, err = mod.read_lane_leases(ROOT, runner=runner)
        self.assertIsNone(err)
        self.assertEqual(len(records), 3)
        self.assertEqual(seen["cmd"], ["dos", "lease-lane", "live"])
        # `--workspace` makes the kernel re-resolve to a different .dos/ and emit
        # non-JSON; the cwd carries the workspace instead.
        self.assertNotIn("--workspace", seen["cmd"])

    def test_render_surfaces_dead_holder_on_every_operator_surface(self) -> None:
        mod, lane = self._fold()
        worker_leases = mod.cross_check_worker_leases(
            {"available": True, "workers": []}, {"active": []}, lane)
        p = build(mod, worker_leases=worker_leases)

        text = mod.render(p)
        self.assertIn("dead-holder=2", text)
        self.assertIn("lane lease: 3 held", text)
        self.assertIn("do NOT reap", text)
        self.assertNotIn("release --lane", text)
        self.assertNotIn("cross-check clean", text)

        md = mod.render_md(p, date="2026-08-06")
        self.assertIn("dead-holder=2", md)
        self.assertIn("Kernel lane leases", md)
        self.assertIn("`cmd`", md)
        self.assertIn("do NOT reap", md)

        slack = mod.slack_text(p)
        self.assertIn("dead-holder", slack)
        self.assertIn("TTL-EXPIRED", slack)
        self.assertIn("do NOT reap", slack)
        self.assertNotIn("release --lane", slack)
        self.assertNotIn("worker/lease cross-check clean", slack)

    def test_all_live_holders_still_read_clean(self) -> None:
        """The guard must not cry wolf: with every holder live the card is clean."""
        mod = load()
        lane = mod.summarize_lane_lease_holders(
            self._lane_records()[:1], starts=self._starts(), host_id=self.HOST,
            now_ts=self.NOW)
        self.assertEqual(lane["dead_count"], 0)
        self.assertEqual(lane["live_count"], 1)
        out = mod.cross_check_worker_leases(
            {"available": True,
             "workers": [{"worker": "w", "issue": 1, "pid": 1001, "lane": "tools",
                          "backend": "claude", "lease_id": "resolve-tools-1765"}]},
            {"active": [{"id": "resolve-tools-1765", "lane": "tools", "holder": "peer-a"}]},
            lane)
        self.assertTrue(mod.lane_lease_verdict_clean(out))
        self.assertIn("worker/lease cross-check clean",
                      " | ".join(mod._worker_lease_reasons(out)))

    def test_lane_lease_verdict_absent_fold_preserves_legacy_behaviour(self) -> None:
        mod = load()
        out = mod.cross_check_worker_leases({"available": True, "workers": []},
                                            {"active": []})
        self.assertTrue(mod.lane_lease_verdict_clean(out))
        self.assertNotIn("lane_leases", out)

    def test_card_must_not_read_clean_with_a_dead_holder_lease(self) -> None:
        """THE reported symptom, asserted on the rendered card.

        Deliberately built from a LITERAL payload rather than the fold's own
        helpers, so it exercises only the pre-existing `worker_lease_check` ->
        render seam. That makes it a real fail-first guard: against the unfixed
        `dispatch_status.py` this reaches `render()` and fails on the assertions
        below (the card printed `clean=3 orphan-process=0 unmatched-live-lease=0`
        and "worker/lease cross-check clean" with 18 lanes fenced), rather than
        erroring out on a missing symbol.
        """
        mod = load()
        worker_leases = {
            "available": True,
            # Every LOCAL worker/lease pair matches — this is the state that used to
            # print "clean" while the kernel's lane leases were all fenced.
            "clean_count": 3,
            "orphan_process_count": 0,
            "orphan_lease_count": 0,
            "clean": [], "orphan_process": [], "orphan_lease": [],
            "dead_holder_count": 2,
            "lane_leases": {
                "schema": "fleet-lane-lease-liveness/1",
                "source": "dos lease-lane live",
                "available": True,
                "total": 3, "live_count": 1, "dead_count": 2, "unknown_count": 0,
                "next_action": ("do NOT reap: `dos lease-lane release` runs NO "
                                "liveness check (#5859) · the admission fold already "
                                "elides these"),
                "rows": [], "dead": [
                    {"lane": "cmd", "holder": "claude-5031", "pid": 51980,
                     "age_min": 31680.0, "holder_state": "dead",
                     "holder_evidence": "TTL EXPIRED: no acquire stamp for 528h"},
                    {"lane": "modver", "holder": "worker-2461", "pid": 56980,
                     "age_min": 17280.0, "holder_state": "dead",
                     "holder_evidence": "TTL EXPIRED: no acquire stamp for 288h"},
                ],
            },
        }
        p = build(mod, worker_leases=worker_leases)

        text = mod.render(p)
        self.assertIn("dead-holder=2", text)
        self.assertNotIn("worker/lease cross-check clean", text)
        self.assertIn("cmd(claude-5031, pid 51980", text)
        self.assertIn("do NOT reap", text)
        self.assertNotIn("release --lane", text)

        self.assertTrue(any("dead-holder=2" in r for r in p["reasons"]),
                        f"no dead-holder counter in reasons: {p['reasons']}")
        self.assertFalse(any("cross-check clean" in r for r in p["reasons"]),
                         f"card still reads clean with 2 dead holders: {p['reasons']}")


class LaneLeaseEphemeralAcquirerTest(unittest.TestCase):
    """The holder predicate must judge a lease by the WAL, not by its acquirer pid.

    The first cut of the lane-lease fold decided deadness from `(pid,
    proc_starttime)` alone. That predicate cannot discriminate: a lane lease's
    recorded pid is the EPHEMERAL `dos lease-lane acquire` subprocess, which
    journals the ACQUIRE and exits immediately (`dos/lane_lease.py:466-492`), and
    the reservation is designed to outlive it (`acquire()` at `:453` says so). So
    a healthy, actively-held lease always probes "dead", and the card rendered
    `lane lease: 25 held - live=0 dead-holder=25` while several of those lanes
    were held by agents running at that instant — four of them acquired MINUTES
    earlier, and five holders self-released their own leases inside one 9-minute
    window while probing "dead" throughout.

    The corrected predicate mirrors the kernel's own live-set fold
    (`dos.lane_lease._lease_is_dead`): heartbeat/TTL staleness is PRIMARY, the pid
    only corroborates.

    Run this class against the pre-fix module to see it fail:

        git show <sha>:tools/dispatch_status.py > $SCRATCH/pristine_dispatch_status.py
        DISPATCH_STATUS_SCRIPT=$SCRATCH/pristine_dispatch_status.py \\
            python tools/dispatch_status_test.py LaneLeaseEphemeralAcquirerTest
    """

    NOW = 1786080000.0            # 2026-08-07T05:20:00Z
    HOST = "fleet-box"

    def _classify(self, rec, starts=None):
        """Classify one record, pinning the clock when the module accepts a pin.

        `now_ts` is passed only if the loaded classifier takes it, so that against a
        PRE-FIX module these tests still reach the real predicate and fail on its
        VERDICT rather than erroring out on a missing keyword. The pre-fix classifier
        reads no clock at all, so omitting the pin changes nothing for it; every
        fixture below is dated relative to `NOW` and judged only on the resulting
        state.
        """
        import inspect
        mod = load()
        fn = mod.classify_lane_lease_holder
        kwargs = {"host_id": self.HOST}
        if "now_ts" in inspect.signature(fn).parameters:
            kwargs["now_ts"] = self.NOW
        return fn(rec, {} if starts is None else starts, **kwargs)

    @staticmethod
    def _iso(now: float, minutes_ago: float) -> str:
        import datetime as _dt
        return _dt.datetime.fromtimestamp(
            now - minutes_ago * 60.0, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def test_absent_pid_with_fresh_heartbeat_is_live(self) -> None:
        """THE decisive assertion. A lease whose pid is gone but whose heartbeat is
        FRESH is live — that is the normal, healthy shape of a held lane lease, not
        a reapable orphan.

        Against the pre-fix classifier this returns `dead` on the strength of the
        absent pid alone, with no reference to the heartbeat at all.
        """
        state, evidence = self._classify({
            "lane": "tools", "holder": "guard-livelock-5858", "host_id": self.HOST,
            "pid": 33784, "proc_starttime": _filetime(self.NOW - 4000),
            "acquired_at": self._iso(self.NOW, 240.0),
            "heartbeat_at": self._iso(self.NOW, 0.5),
        }, starts={})       # pid 33784 is NOT in the process table
        self.assertEqual(state, "live", f"fresh heartbeat judged {state}: {evidence}")
        self.assertIn("fresh", evidence)
        self.assertIn("not a running process", evidence)  # the pid IS reported…
        self.assertIn("ephemeral", evidence)              # …and explicitly discounted

    def test_absent_pid_with_fresh_acquire_stamp_is_live(self) -> None:
        """The same, for the shape this fleet actually writes: no heartbeat op has
        ever been journaled here, so `acquired_at` IS the newest stamp. A lease
        acquired seconds ago must never read dead."""
        state, evidence = self._classify({
            "lane": "docs", "holder": "claude-5842", "host_id": self.HOST,
            "pid": 84412, "proc_starttime": _filetime(self.NOW - 900),
            "acquired_at": self._iso(self.NOW, 0.75),
        }, starts={})
        self.assertEqual(state, "live", f"45s-old lease judged {state}: {evidence}")

    def test_absent_pid_inside_ttl_is_unknown_not_dead(self) -> None:
        """Past the freshness grace but inside the TTL, with the ephemeral acquirer
        gone: unproven. `unknown` — never `live`, and never `dead` either, because
        an absent acquirer pid is the expected state of a healthy lease."""
        state, evidence = self._classify({
            "lane": "session", "holder": "claude-5254", "host_id": self.HOST,
            "pid": 61696, "proc_starttime": _filetime(self.NOW - 3000),
            "acquired_at": self._iso(self.NOW, 20.0),
        }, starts={})
        self.assertEqual(state, "unknown", f"20m-old lease judged {state}: {evidence}")
        self.assertIn("ephemeral", evidence)

    def test_ttl_expiry_is_the_death_evidence(self) -> None:
        """A lease past its TTL + grace IS dead — that is the kernel's own primary
        signal, and it is what `live_leases(expire_dead=True)` elides. The pid rung
        rides along as corroboration only."""
        state, evidence = self._classify({
            "lane": "adjudicator", "holder": "fable-superloop-23874",
            "host_id": self.HOST, "pid": 63712,
            "proc_starttime": _filetime(self.NOW - 2_000_000),
            "acquired_at": self._iso(self.NOW, 31680.0),   # 22 days
        }, starts={})
        self.assertEqual(state, "dead")
        self.assertIn("TTL EXPIRED", evidence)

    def test_declared_ttl_minutes_is_honoured_over_the_backstop(self) -> None:
        """A lease that declares a long `ttl_minutes` is not dead at the 50-minute
        backstop — the backstop only catches a record that declared none."""
        rec = {"lane": "bench", "holder": "long-runner", "host_id": self.HOST,
               "pid": 4242, "acquired_at": self._iso(self.NOW, 300.0),
               "ttl_minutes": 600}
        state, _ = self._classify(rec, starts={})
        self.assertEqual(state, "unknown")
        state, evidence = self._classify(dict(rec, ttl_minutes=60), starts={})
        self.assertEqual(state, "dead")
        self.assertIn("60-minute TTL", evidence)

    def test_dead_pid_alone_never_kills_a_lease(self) -> None:
        """The load-bearing inversion, stated directly: across the whole freshness
        range, a confidently-dead pid on its own produces ZERO `dead` verdicts.

        Pre-fix, every one of these is `dead` — the predicate returned the same
        answer for a 30-second-old lease and a three-week-old one, which is why it
        reported live=0 of 25.
        """
        states = []
        for minutes in (0.1, 1.0, 4.9, 5.0, 9.0, 20.0, 45.0):
            rec = {"lane": f"l{minutes}", "holder": "h", "host_id": self.HOST,
                   "pid": 999, "proc_starttime": _filetime(self.NOW - 5000),
                   "acquired_at": self._iso(self.NOW, minutes)}
            states.append(self._classify(rec, starts={})[0])
        self.assertNotIn("dead", states, f"dead pid alone still kills leases: {states}")
        self.assertEqual(states[:4], ["live", "live", "live", "live"])

    def test_whole_set_does_not_collapse_to_all_dead(self) -> None:
        """The reality check: fold a set shaped like the real one and assert the
        verdict actually discriminates. A classifier that answers `dead` for every
        lease is not fixed, whatever its internals say."""
        mod = load()
        records = [
            # freshly acquired, acquirer already exited — the healthy shape
            {"lane": "docs", "holder": "claude-5842", "host_id": self.HOST,
             "pid": 84412, "acquired_at": self._iso(self.NOW, 0.75)},
            {"lane": "hooks", "holder": "claude-5026", "host_id": self.HOST,
             "pid": 50584, "acquired_at": self._iso(self.NOW, 4.0)},
            # inside the TTL, unprovable either way
            {"lane": "tools", "holder": "guard-5858", "host_id": self.HOST,
             "pid": 33784, "acquired_at": self._iso(self.NOW, 9.0)},
            {"lane": "cmd", "holder": "claude-5254", "host_id": self.HOST,
             "pid": 3388, "acquired_at": self._iso(self.NOW, 20.0)},
            # genuinely aged out
            {"lane": "adjudicator", "holder": "fable-superloop", "host_id": self.HOST,
             "pid": 63712, "acquired_at": self._iso(self.NOW, 31680.0)},
        ]
        lane = mod.summarize_lane_lease_holders(
            records, starts={}, host_id=self.HOST, now_ts=self.NOW)
        self.assertEqual(lane["total"], 5)
        self.assertEqual(lane["live_count"], 2)
        self.assertEqual(lane["unknown_count"], 2)
        self.assertEqual(lane["dead_count"], 1)
        self.assertLess(lane["dead_count"], lane["total"],
                        "every lease still reads dead — the predicate is still wrong")

    def test_card_never_calls_dead_what_the_kernel_still_admits(self) -> None:
        """The safety invariant: the card's `dead` set is a SUBSET of what the
        kernel's admission fold drops, never a superset.

        It holds because `dead` now requires the same TTL+grace expiry the kernel
        uses, and the one rung the card narrows (dead-pid corroboration, which the
        card demands an OBSERVED heartbeat for) only ever moves a verdict toward
        `unknown`. Asserted here as monotonicity over age: the verdict may only walk
        live -> unknown -> dead, and it may not reach `dead` before TTL+grace.

        Verified live at authoring time against `dos lease-lane live` on the
        reference tree: the kernel's `live_leases(expire_dead=True)` dropped 25 of
        26 leases, the card called 18 of them dead, and `card_dead - kernel_dropped`
        was empty — the 7 differences were all card-`unknown`, the safe direction.
        """
        order = {"live": 0, "unknown": 1, "dead": 2}
        seen = []
        for minutes in (0.0, 1.0, 5.0, 5.1, 20.0, 54.9, 55.0, 55.1, 600.0, 31680.0):
            state, _ = self._classify({
                "lane": "cmd", "holder": "h", "host_id": self.HOST, "pid": 3388,
                "proc_starttime": _filetime(self.NOW - 9000),
                "acquired_at": self._iso(self.NOW, minutes),
            }, starts={})
            seen.append((minutes, state))
            if state == "dead":
                self.assertGreater(minutes, 50.0 + 5.0,
                                   f"dead at {minutes}m, before TTL+grace: {seen}")
        ranks = [order[s] for _, s in seen]
        self.assertEqual(ranks, sorted(ranks), f"verdict is not monotone in age: {seen}")
        self.assertEqual(seen[-1][1], "dead")
        self.assertEqual(seen[0][1], "live")

    def test_next_action_never_recommends_a_blind_release(self) -> None:
        """`dos lease-lane release` runs no liveness check and its `--owner ""`
        matches any holder (#5859), so the card must never hand an operator a reap
        loop over the lane set. It must instead say what IS true: the admission
        fold self-heals, and a stale structural lease blocks only
        `lane_lease.acquire()`."""
        mod = load()
        action = mod._LANE_LEASE_NEXT_ACTION
        self.assertNotIn("--owner <holder>", action)
        self.assertNotIn("release --lane", action)
        self.assertNotIn("reap each", action)
        self.assertIn("do NOT reap", action)
        self.assertIn("#5859", action)
        self.assertIn("OP_SCAVENGE", action)
        self.assertIn("adopt()", action)
        self.assertIn("expire_dead=True", action)
        self.assertIn("lane_lease.acquire()", action)
        self.assertIn("dos/lane_lease.py:453", action)


if __name__ == "__main__":
    unittest.main()
