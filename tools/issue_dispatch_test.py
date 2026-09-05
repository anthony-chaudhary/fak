#!/usr/bin/env python3
"""Hermetic tests for tools/issue_dispatch.py.

One guarded dispatch tick composes three shelling-out pieces — preflight,
issue_lane_router (via pick_lane's run_json), and the detached spawn. All are
replaced here with synthetic results on the module; NOTHING live (preflight /
gh / dos / claude) is ever invoked, and spawn_detached is never reached in
dry-run. worker_env's account-pinning is exercised against a real tmp dir so the
config-dir pin / token-scrub branches run without any network.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "issue_dispatch.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("issue_dispatch", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PickLaneTest(unittest.TestCase):
    def test_picks_lane_with_most_issues(self) -> None:
        mod = load()
        mod.run_json = lambda cmd, cwd, timeout: {"lanes": {
            "docs": {"issues": [1, 2]},
            "gateway": {"issues": [1, 2, 3, 4]},
            "recall": {"issues": [9]},
        }}
        pick = mod.pick_lane(ROOT, None)
        self.assertEqual(pick["lane"], "gateway")
        self.assertEqual(pick["issues"], 4)
        self.assertEqual(pick["by_lane"],
                         {"docs": 2, "gateway": 4, "recall": 1})
        self.assertIsNone(pick.get("router_error"))

    def test_explicit_lane_override_uses_its_count(self) -> None:
        mod = load()
        mod.run_json = lambda cmd, cwd, timeout: {"lanes": {
            "docs": {"issues": [1, 2]},
            "gateway": {"issues": [1, 2, 3, 4]},
        }}
        pick = mod.pick_lane(ROOT, "docs")
        self.assertEqual(pick["lane"], "docs")     # honored even though gateway has more
        self.assertEqual(pick["issues"], 2)
        self.assertTrue(pick["explicit"])

    def test_explicit_lane_unknown_to_router_counts_zero(self) -> None:
        mod = load()
        mod.run_json = lambda cmd, cwd, timeout: {"lanes": {"docs": {"issues": [1]}}}
        pick = mod.pick_lane(ROOT, "nonesuch")
        self.assertEqual(pick["lane"], "nonesuch")
        self.assertEqual(pick["issues"], 0)

    def test_empty_router_yields_no_lane(self) -> None:
        mod = load()
        mod.run_json = lambda cmd, cwd, timeout: {}
        pick = mod.pick_lane(ROOT, None)
        self.assertIsNone(pick["lane"])
        self.assertEqual(pick["issues"], 0)
        self.assertEqual(pick["by_lane"], {})

    def test_router_error_surfaced(self) -> None:
        mod = load()
        mod.run_json = lambda cmd, cwd, timeout: {"_error": "router crashed"}
        pick = mod.pick_lane(ROOT, None)
        self.assertIsNone(pick["lane"])
        self.assertEqual(pick["router_error"], "router crashed")

    def test_lane_info_as_bare_list(self) -> None:
        mod = load()
        # info may be the issue list directly, not a {"issues": [...]} dict.
        mod.run_json = lambda cmd, cwd, timeout: {"lanes": {
            "docs": [1, 2, 3], "gateway": [1]}}
        pick = mod.pick_lane(ROOT, None)
        self.assertEqual(pick["lane"], "docs")
        self.assertEqual(pick["issues"], 3)


class PickLanePriorityTest(unittest.TestCase):
    """#2064: pick_lane ranks the pool by the highest issue_triage score in a lane
    FIRST and raw open-issue count only as the tiebreak, so a single sequential seat
    resolves the highest-VALUE leaf across lanes instead of always draining the lane
    with the biggest (often low-priority docs) backlog. Fails OPEN to count-only
    ranking when triage data is unavailable — the historical behavior."""

    # docs has the bigger backlog; gateway holds the single high-priority (P0) leaf.
    LANES = {"lanes": {"docs": {"issues": [1, 2, 3, 4]},
                       "gateway": {"issues": [9]}}}

    def _wire(self, mod, triage) -> None:
        """Route the router call to LANES and the issue_triage call to `triage`,
        distinguished by which tool the cmd names — the real two-subprocess shape."""
        def run_json(cmd, cwd, timeout):
            if any("issue_triage.py" in str(part) for part in cmd):
                return json.loads(json.dumps(triage))
            return json.loads(json.dumps(self.LANES))
        mod.run_json = run_json

    def test_high_value_lane_beats_bigger_low_value_backlog(self) -> None:
        mod = load()
        self._wire(mod, {"rows": [
            {"number": 1, "score": 150}, {"number": 2, "score": 150},
            {"number": 3, "score": 150}, {"number": 4, "score": 150},
            {"number": 9, "score": 1300}]})   # gateway #9 = orphan P0
        pick = mod.pick_lane(ROOT, None, guarded=False)
        self.assertEqual(pick["lane"], "gateway")   # P0 leaf beats docs' 4x P2
        self.assertEqual(pick["issues"], 1)
        self.assertEqual(pick["lane_priority"], 1300)
        self.assertEqual(pick["priority_by_lane"], {"docs": 150, "gateway": 1300})

    def test_count_breaks_ties_within_a_priority_tier(self) -> None:
        mod = load()
        # every leaf is the same priority -> the count tiebreak keeps the old order.
        self._wire(mod, {"rows": [
            {"number": 1, "score": 150}, {"number": 2, "score": 150},
            {"number": 3, "score": 150}, {"number": 4, "score": 150},
            {"number": 9, "score": 150}]})
        pick = mod.pick_lane(ROOT, None, guarded=False)
        self.assertEqual(pick["lane"], "docs")      # tie on priority -> bigger backlog
        self.assertEqual(pick["lane_priority"], 150)

    def test_fails_open_to_count_when_triage_unavailable(self) -> None:
        mod = load()
        # a triage read error (no rows) collapses to legacy raw-count ranking.
        self._wire(mod, {"_error": "gh unavailable"})
        pick = mod.pick_lane(ROOT, None, guarded=False)
        self.assertEqual(pick["lane"], "docs")      # richest backlog, unchanged
        self.assertEqual(pick["lane_priority"], 0)
        self.assertEqual(pick["priority_by_lane"], {})

    def test_priority_never_overrides_the_self_source_hold(self) -> None:
        mod = load()
        # gateway is the highest-priority lane but is self-source-held under guard;
        # the hard exclude must win — priority ranking only ever orders the SAFE pool.
        lanes = {"lanes": {"docs": {"issues": [1, 2], "tree": ["docs/**"]},
                           "gateway": {"issues": [9], "tree": ["internal/gateway/**"]}}}

        def run_json(cmd, cwd, timeout):
            if any("issue_triage.py" in str(part) for part in cmd):
                return {"rows": [{"number": 1, "score": 150},
                                 {"number": 2, "score": 150},
                                 {"number": 9, "score": 1300}]}
            return json.loads(json.dumps(lanes))
        mod.run_json = run_json
        pick = mod.pick_lane(ROOT, None, guarded=True)
        self.assertEqual(pick["lane"], "docs")           # gateway held despite its P0
        self.assertEqual(pick["self_modify_held"], ["gateway"])

    def test_busy_high_value_lane_still_rotates_to_free_pool(self) -> None:
        mod = load()
        # the P0 lane is already in flight -> the seat rotates to the free lane even
        # though gateway outranks it, because priority orders the FREE pool.
        self._wire(mod, {"rows": [
            {"number": 1, "score": 150}, {"number": 2, "score": 150},
            {"number": 3, "score": 150}, {"number": 4, "score": 150},
            {"number": 9, "score": 1300}]})
        pick = mod.pick_lane(ROOT, None, busy={"gateway"}, guarded=False)
        self.assertEqual(pick["lane"], "docs")
        self.assertFalse(pick["stacked"])


class WorkerEnvTest(unittest.TestCase):
    def test_pins_config_dir_drops_ambient_token_and_sets_witness(self) -> None:
        mod = load()
        with mock.patch.dict("os.environ", {"CLAUDE_CODE_OAUTH_TOKEN": "ambient"},
                             clear=False):
            with tempfile.TemporaryDirectory() as d:
                (Path(d) / ".oauth-token").write_text("tok-12345\n", encoding="utf-8")
                env = mod.worker_env(d, "docs", ROOT)
        self.assertEqual(env["CLAUDE_CONFIG_DIR"], d)
        self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", env)
        self.assertEqual(env["FLEET_DISPATCH_WITNESS"], "benchmark")
        self.assertIn("--lane docs", env["FLEET_BENCH_WITNESS_CMD"])

    def test_setup_token_is_opt_in_and_stripped(self) -> None:
        mod = load()
        with mock.patch.dict(
            "os.environ",
            {mod.USE_SETUP_TOKEN_ENV: "1", "CLAUDE_CODE_OAUTH_TOKEN": "ambient"},
            clear=False,
        ):
            with tempfile.TemporaryDirectory() as d:
                (Path(d) / ".oauth-token").write_text("tok-12345\n", encoding="utf-8")
                env = mod.worker_env(d, "docs", ROOT)
        self.assertEqual(env["CLAUDE_CONFIG_DIR"], d)
        self.assertEqual(env["CLAUDE_CODE_OAUTH_TOKEN"], "tok-12345")

    def test_missing_token_pops_the_oauth_var(self) -> None:
        mod = load()
        with mock.patch.dict("os.environ", {"CLAUDE_CODE_OAUTH_TOKEN": "ambient"},
                             clear=False):
            with tempfile.TemporaryDirectory() as d:
                # no .oauth-token in this dir
                env = mod.worker_env(d, "gateway", ROOT)
        self.assertEqual(env["CLAUDE_CONFIG_DIR"], d)
        self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", env)
        self.assertEqual(env["FLEET_DISPATCH_WITNESS"], "benchmark")

    def test_no_account_dir_still_sets_witness(self) -> None:
        mod = load()
        env = mod.worker_env(None, "recall", ROOT)
        self.assertEqual(env["FLEET_DISPATCH_WITNESS"], "benchmark")
        self.assertIn("--lane recall", env["FLEET_BENCH_WITNESS_CMD"])

    def test_arms_verdict_journal_observe_on_dispatch_surface(self) -> None:
        # #465: the verdict-journal auto-emit is armed per dispatched run (bounded),
        # NOT per idle session (unbounded — the journal is not auto-rotated). The arm
        # is independent of the account dir, so it holds for every worker env shape.
        mod = load()
        env_no_acct = mod.worker_env(None, "docs", ROOT)
        self.assertEqual(env_no_acct["DISPATCH_OBSERVE"], "1")
        with tempfile.TemporaryDirectory() as d:
            env_acct = mod.worker_env(d, "gateway", ROOT)
        self.assertEqual(env_acct["DISPATCH_OBSERVE"], "1")

    def test_stamps_worker_id_from_account_seat_basename(self) -> None:
        # #2065: the worker id defaults to the pinned seat directory basename so a
        # dispatched worker's (fak-worker <id>) trailer is attributable per seat.
        mod = load()
        with mock.patch.dict("os.environ", {"FLEET_WORKER_ID": ""}, clear=False):
            with tempfile.TemporaryDirectory() as d:
                env = mod.worker_env(d, "docs", ROOT)
        self.assertEqual(env["FLEET_WORKER_ID"],
                         mod._sanitize_worker_id(Path(d).name))
        self.assertNotEqual(env["FLEET_WORKER_ID"], "unknown")

    def test_worker_id_falls_back_to_lane_without_account(self) -> None:
        mod = load()
        with mock.patch.dict("os.environ", {"FLEET_WORKER_ID": ""}, clear=False):
            env = mod.worker_env(None, "recall", ROOT)
        self.assertEqual(env["FLEET_WORKER_ID"], "recall")

    def test_explicit_worker_id_env_overrides_the_seat(self) -> None:
        # An operator/wave override (FLEET_WORKER_ID already set) wins over the seat,
        # sanitized to a trailer-safe token.
        mod = load()
        with mock.patch.dict("os.environ", {"FLEET_WORKER_ID": "wave-3/rank-1"},
                             clear=False):
            with tempfile.TemporaryDirectory() as d:
                env = mod.worker_env(d, "docs", ROOT)
        self.assertEqual(env["FLEET_WORKER_ID"], "wave-3-rank-1")


class EvaluateTest(unittest.TestCase):
    SPAWN_OK = {
        "verdict": "SPAWN_OK", "reason": None, "cap": 2, "live": 0,
        "account": {"tag": "worker-a", "tier": 1, "model": "claude", "dir": "/acct/a"},
    }

    def _no_spawn(self, mod) -> None:
        """Guard: spawn_detached must never be reached in dry-run."""
        def boom(*a, **k):
            raise AssertionError("dry-run must never spawn a worker")
        mod.spawn_detached = boom

    def _patch(self, mod, *, pre, lane_pick) -> None:
        # refresh_registry shells out to fleet_sessions.py; stub it so the tick is
        # hermetic. Its real behavior (route off fresh evidence) is covered below.
        mod.refresh_registry = lambda root: {"ok": True, "stubbed": True}
        mod.preflight = lambda root, **kw: pre
        # busy_lanes reads .dispatch-runs/ from disk; stub it empty so the tick is
        # hermetic. Its real behavior (fold + prune inflight markers) is covered below.
        mod.busy_lanes = lambda runs_dir, **kw: set()
        mod.lease_ref_busy_lanes = lambda root: {"lanes": set()}
        mod.pick_lane = lambda root, explicit, busy=None: lane_pick

    def test_would_spawn_when_preflight_ok_and_lane_chosen(self) -> None:
        mod = load()
        self._no_spawn(mod)
        self._patch(mod, pre=self.SPAWN_OK,
                    lane_pick={"lane": "gateway", "issues": 4, "by_lane": {}})
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertEqual(p["action"], "would_spawn")
        self.assertEqual(p["lane"], "gateway")
        self.assertEqual(p["lane_issue_count"], 4)
        # the build_command for the chosen lane is filled in.
        self.assertEqual(p["command"][0], "claude")
        self.assertIn("gateway", p["command"][-1])
        self.assertIn("worker-a", p["reason"])

    def test_refused_when_preflight_refuses(self) -> None:
        mod = load()
        self._no_spawn(mod)
        self._patch(
            mod,
            pre={"verdict": "REFUSE_AT_CAP", "reason": "2/2 live", "cap": 2,
                 "live": 2, "max_workers": 2, "host_cap": 16,
                 "capacity_limiter": {
                     "primary": "configured_max",
                     "term": "max_workers",
                     "raw": {"max_workers": 2, "host_cap": 16},
                 },
                 "seat": {"total": 20, "free": 18, "leased": 2,
                          "depleted": False},
                 "account": {}},
            lane_pick={"lane": "gateway", "issues": 4, "by_lane": {}})
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertFalse(p["ok"])
        self.assertEqual(p["action"], "refused")
        self.assertEqual(p["verdict"], "REFUSE_AT_CAP")
        self.assertIn("preflight refused", p["reason"])
        self.assertIn("2/2 live", p["reason"])
        self.assertEqual(p["preflight"]["capacity_limiter"]["term"], "max_workers")
        self.assertEqual(p["preflight"]["seat"]["free"], 18)
        self.assertEqual(p["preflight_hint"]["kind"], "configured_max_workers")
        self.assertIn("configured --max-workers=2", mod.render(p))

    def test_preflight_payload_surfaces_host_cap(self) -> None:
        # #1337: the host-derived adaptive ceiling must be observable in the
        # dispatcher's OWN telemetry — its structured preflight payload and the
        # human render — not only buried in the preflight reason string, so an
        # operator can see the live population tracking host headroom.
        mod = load()
        self._no_spawn(mod)
        self._patch(
            mod,
            pre={"verdict": "REFUSE_AT_CAP", "reason": "3/3 live, host_cap=3",
                 "cap": 3, "live": 3, "host_cap": 3, "account": {}},
            lane_pick={"lane": "tools", "issues": 1, "by_lane": {}})
        p = mod.evaluate(ROOT, max_workers=10, work_kind="engineering",
                         lane=None, live=False)
        self.assertEqual(p["preflight"]["host_cap"], 3)
        self.assertIn("host_cap 3", mod.render(p))

    def test_preflight_payload_surfaces_headroom_limiter_and_seat(self) -> None:
        mod = load()
        self._no_spawn(mod)
        self._patch(
            mod,
            pre={"verdict": "SPAWN_OK", "reason": "ok", "cap": 16, "live": 9,
                 "headroom": 7, "max_workers": 20, "host_cap": 16,
                 "capacity_limiter": {
                     "primary": "cpu",
                     "term": "host_cap",
                     "raw": {"seat_free": 11},
                 },
                 "seat": {"total": 20, "free": 11, "leased": 9,
                          "depleted": False},
                 "account": {"tag": "worker-a", "tier": 1, "dir": "/acct/a"}},
            lane_pick={"lane": "tools", "issues": 1, "by_lane": {}})
        p = mod.evaluate(ROOT, max_workers=20, work_kind="engineering",
                         lane=None, live=False)
        self.assertEqual(p["preflight"]["headroom"], 7)
        self.assertEqual(p["preflight"]["capacity_limiter"]["primary"], "cpu")
        self.assertEqual(p["preflight"]["seat"]["leased"], 9)

    def test_preflight_payload_omits_host_cap_when_unbounded(self) -> None:
        # When no host dimension is readable host_cap is None; the render then
        # falls back to the static live/cap form with no host_cap clause.
        mod = load()
        self._no_spawn(mod)
        self._patch(mod, pre=self.SPAWN_OK,
                    lane_pick={"lane": "tools", "issues": 1, "by_lane": {}})
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertNotIn("host_cap", p["preflight"])
        self.assertNotIn("host_cap", mod.render(p))

    def test_no_lane_when_router_empty(self) -> None:
        mod = load()
        self._no_spawn(mod)
        self._patch(mod, pre=self.SPAWN_OK,
                    lane_pick={"lane": None, "issues": 0, "by_lane": {}})
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertFalse(p["ok"])
        self.assertEqual(p["action"], "no_lane")
        self.assertEqual(p["verdict"], "NO_LANE")
        self.assertEqual(p["command"], [])   # no lane -> no command built

    def test_refused_takes_precedence_over_no_lane(self) -> None:
        mod = load()
        self._no_spawn(mod)
        # even with no lane, a refusing preflight short-circuits first.
        self._patch(
            mod,
            pre={"verdict": "REFUSE_HOST", "reason": "guard flagged", "account": {}},
            lane_pick={"lane": None, "issues": 0, "by_lane": {}})
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertEqual(p["verdict"], "REFUSE_HOST")
        self.assertEqual(p["action"], "refused")

    def test_dry_run_with_explicit_lane(self) -> None:
        mod = load()
        self._no_spawn(mod)
        self._patch(mod, pre=self.SPAWN_OK,
                    lane_pick={"lane": "docs", "issues": 1, "by_lane": {},
                               "explicit": True})
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane="docs", live=False)
        self.assertEqual(p["verdict"], "WOULD_SPAWN")
        self.assertEqual(p["lane"], "docs")
        self.assertEqual(p["witness"]["cmd"],
                         "python tools/bench_witness.py --lane docs")


class RefreshRegistryTest(unittest.TestCase):
    """The per-tick registry refresh: route off CURRENT account evidence so a
    freshly-blocked account is never handed to a worker that would instantly die."""

    def test_refresh_runs_before_preflight_and_is_recorded(self) -> None:
        mod = load()
        order: list[str] = []
        mod.refresh_registry = lambda root: (order.append("refresh") or
                                             {"ok": True, "marker": "fresh"})

        def pre(root, **kw):
            order.append("preflight")
            return {"verdict": "REFUSE_AT_CAP", "reason": "x", "cap": 2,
                    "live": 2, "account": {}}
        mod.preflight = pre
        mod.busy_lanes = lambda runs_dir, **kw: set()
        mod.lease_ref_busy_lanes = lambda root: {"lanes": set()}
        mod.pick_lane = lambda root, explicit, busy=None: {"lane": "docs", "issues": 1,
                                                           "by_lane": {}}
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        # refresh happens FIRST (so preflight's switcher reads the fresh roster),
        self.assertEqual(order, ["refresh", "preflight"])
        # and the refresh outcome is surfaced in the tick record.
        self.assertEqual(p["registry_refresh"], {"ok": True, "marker": "fresh"})

    def test_refresh_false_skips_the_scan(self) -> None:
        mod = load()
        def boom(root):
            raise AssertionError("refresh=False must not scan")
        mod.refresh_registry = boom
        mod.preflight = lambda root, **kw: {"verdict": "REFUSE_AT_CAP",
                                            "reason": "x", "account": {}}
        mod.busy_lanes = lambda runs_dir, **kw: set()
        mod.lease_ref_busy_lanes = lambda root: {"lanes": set()}
        mod.pick_lane = lambda root, explicit, busy=None: {"lane": "docs", "issues": 1,
                                                           "by_lane": {}}
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False, refresh=False)
        self.assertTrue(p["registry_refresh"].get("skipped"))


def _seat(i: int) -> dict:
    """A synthetic distinct-pool seat record, shaped like fleet_accounts.allocate_wave
    lanes (config_dir / tag / model_tier / pool)."""
    return {"tag": f"acct-{i}", "model_tier": 1, "pool": f"pool-{i}",
            "config_dir": f"/acct/{i}"}


def _disjoint_arbitrate(root, lane, tree, leases):
    """Stub of dos arbitrate's admission: a lane is admitted iff its tree shares no
    glob with any lease already in ``leases``. Proves both that the accumulator
    threads through every spawn AND that a colliding lane is refused pre-launch."""
    held = {t for L in leases for t in (L.get("tree") or [])}
    collides = any(t in held for t in tree)
    return {"admitted": not collides, "outcome": "acquire",
            "got": lane if not collides else "redirected",
            "auto_picked": collides, "tree": list(tree), "reason": "stub-arbitrate"}


class WaveTest(unittest.TestCase):
    """The #1335 wave tick: K disjoint-lane workers in one tick, priced + arbitrated +
    capped. Every shelling-out piece (registry refresh, seats, router, preflight,
    arbitrate, spawn) is stubbed on the module — nothing live is invoked."""

    SPAWN_OK = {"verdict": "SPAWN_OK", "reason": None, "cap": 10, "live": 0,
                "account": {"tag": "acct-0", "tier": 1, "dir": "/acct/0"}}

    def _wire(self, mod, *, seats, candidates, pre=None, arbitrate=None,
              no_spawn=True) -> None:
        mod.refresh_registry = lambda root: {"ok": True, "stubbed": True}
        mod.busy_lanes = lambda runs_dir, **kw: set()
        mod.lease_ref_busy_lanes = lambda root: {"lanes": set()}
        mod.allocate_seats = lambda root, mw, wk: {
            "granted": len(seats), "requested": 99, "shortfall": 0,
            "wave_id": "wave-test", "lanes": seats}
        mod.lane_candidates = lambda root: {"candidates": candidates,
                                            "router_error": None}
        mod.preflight = (pre if callable(pre)
                         else (lambda root, **kw: pre or self.SPAWN_OK))
        mod.arbitrate_lane = arbitrate or _disjoint_arbitrate
        if no_spawn:
            def boom(*a, **k):
                raise AssertionError("dry-run must never spawn a wave worker")
            mod._spawn_wave_member = boom

    def test_fills_disjoint_lanes_each_on_its_own_seat(self) -> None:
        mod = load()
        cands = [{"lane": "tools", "issues": 9, "tree": ["tools/**"]},
                 {"lane": "docs", "issues": 7, "tree": ["docs/**"]},
                 {"lane": "model", "issues": 5, "tree": ["internal/model/**"]},
                 {"lane": "ci", "issues": 3, "tree": [".github/**"]}]
        self._wire(mod, seats=[_seat(i) for i in range(4)], candidates=cands)
        p = mod.evaluate_wave(ROOT, max_workers=4, work_kind="engineering", live=False)
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "WOULD_WAVE")
        self.assertEqual(p["size"], 4)
        self.assertEqual(p["lanes"], ["tools", "docs", "model", "ci"])
        # each admitted lane drew a DISTINCT seat, ranked in order.
        self.assertEqual(p["seats_used"], ["acct-0", "acct-1", "acct-2", "acct-3"])
        self.assertEqual([m["rank"] for m in p["members"]], [0, 1, 2, 3])

    def test_colliding_lane_is_skipped_before_launch(self) -> None:
        mod = load()
        # 'tools2' shares tools/** with 'tools' -> the priced partition refuses it,
        # and the wave moves on to the next disjoint lane instead of spawning it.
        cands = [{"lane": "tools", "issues": 9, "tree": ["tools/**"]},
                 {"lane": "tools2", "issues": 8, "tree": ["tools/**"]},
                 {"lane": "docs", "issues": 7, "tree": ["docs/**"]}]
        self._wire(mod, seats=[_seat(i) for i in range(4)], candidates=cands)
        p = mod.evaluate_wave(ROOT, max_workers=4, work_kind="engineering", live=False)
        self.assertEqual(p["size"], 2)
        self.assertEqual(p["lanes"], ["tools", "docs"])   # tools2 skipped
        self.assertEqual(p["seats_used"], ["acct-0", "acct-1"])

    def test_cap_recheck_bounds_the_wave_below_seats_and_lanes(self) -> None:
        mod = load()
        # 4 seats, 4 disjoint lanes, K=4 — but the preflight cap is 2, re-checked per
        # spawn. The live population must never exceed the cap, so size == 2.
        cands = [{"lane": L, "issues": 9 - i, "tree": [f"{L}/**"]}
                 for i, L in enumerate(["tools", "docs", "model", "ci"])]
        pre = {"verdict": "SPAWN_OK", "reason": None, "cap": 2, "live": 0,
               "account": {}}
        calls = {"n": 0}

        def counting_pre(root, **kw):
            calls["n"] += 1
            return pre
        self._wire(mod, seats=[_seat(i) for i in range(4)], candidates=cands,
                   pre=counting_pre)
        p = mod.evaluate_wave(ROOT, max_workers=4, work_kind="engineering", live=False)
        self.assertEqual(p["size"], 2)
        self.assertEqual(p["cap"], 2)
        self.assertEqual(p["refusal"], "REFUSE_AT_CAP")
        self.assertEqual(p["last_preflight"]["effective_live"], 2)
        self.assertEqual(p["preflight_hint"]["kind"], "in_tick_cap")
        self.assertIn("wave in-tick accounting", mod.render_wave(p))
        # preflight was re-checked per spawn: 2 admits + 1 that hit the cap = 3 calls.
        self.assertEqual(calls["n"], 3)

    def test_preflight_refusal_stops_the_wave_with_zero_workers(self) -> None:
        mod = load()
        cands = [{"lane": "tools", "issues": 9, "tree": ["tools/**"]}]
        pre = {"verdict": "REFUSE_AT_CAP", "reason": "2/2 live", "cap": 2,
               "live": 2, "max_workers": 2, "host_cap": 16, "account": {},
               "capacity_limiter": {
                   "primary": "configured_max",
                   "term": "max_workers",
                   "raw": {"max_workers": 2, "host_cap": 16},
               },
               "seat": {"total": 20, "free": 18, "leased": 2,
                        "depleted": False}}
        self._wire(mod, seats=[_seat(0), _seat(1)], candidates=cands, pre=pre)
        p = mod.evaluate_wave(ROOT, max_workers=2, work_kind="engineering", live=False)
        self.assertFalse(p["ok"])
        self.assertEqual(p["size"], 0)
        self.assertEqual(p["verdict"], "REFUSE_AT_CAP")
        self.assertEqual(p["last_preflight"]["capacity_limiter"]["term"], "max_workers")
        self.assertEqual(p["preflight_hint"]["kind"], "configured_max_workers")
        self.assertEqual(p["preflight_hint"]["required_min"], 3)
        self.assertIn("configured --max-workers=2", p["preflight_hint"]["message"])
        self.assertIn("configured --max-workers=2", mod.render_wave(p))

    def test_initial_preflight_refusal_requests_zero_seats(self) -> None:
        mod = load()
        cands = [{"lane": "tools", "issues": 9, "tree": ["tools/**"]}]
        calls = {"preflight": 0, "seat_request": None}

        def pre(root, **kw):
            calls["preflight"] += 1
            return {"verdict": "REFUSE_AT_CAP", "reason": "16/16 live",
                    "cap": 16, "live": 16, "headroom": 0,
                    "max_workers": 20, "host_cap": 16,
                    "capacity_limiter": {"primary": "leases", "term": "live"},
                    "seat": {"total": 20, "free": 4, "leased": 16,
                             "depleted": False},
                    "account": {}}

        self._wire(mod, seats=[], candidates=cands, pre=pre)

        def allocate(root, mw, wk):
            calls["seat_request"] = mw
            return {"granted": 0, "requested": mw, "shortfall": mw,
                    "wave_id": "wave-zero", "lanes": []}
        mod.allocate_seats = allocate

        p = mod.evaluate_wave(ROOT, max_workers=20, work_kind="engineering",
                              live=False)
        self.assertEqual(p["verdict"], "REFUSE_AT_CAP")
        self.assertEqual(p["admission_budget"]["requested_seats"], 0)
        self.assertEqual(calls["seat_request"], 0)
        self.assertEqual(calls["preflight"], 1)
        self.assertEqual(p["free_seats"], 0)

    def test_preflight_headroom_bounds_wave_seat_request(self) -> None:
        mod = load()
        cands = [{"lane": L, "issues": 9 - i, "tree": [f"{L}/**"]}
                 for i, L in enumerate(["tools", "docs", "ci"])]
        calls = {"seat_request": None}
        pre = {"verdict": "SPAWN_OK", "reason": None, "cap": 16, "live": 15,
               "headroom": 1, "max_workers": 20,
               "seat": {"total": 20, "free": 4, "leased": 16,
                        "depleted": False},
               "account": {}}
        self._wire(mod, seats=[_seat(0)], candidates=cands, pre=pre)

        def allocate(root, mw, wk):
            calls["seat_request"] = mw
            return {"granted": mw, "requested": mw, "shortfall": 0,
                    "wave_id": "wave-one", "lanes": [_seat(i) for i in range(mw)]}
        mod.allocate_seats = allocate

        p = mod.evaluate_wave(ROOT, max_workers=20, work_kind="engineering",
                              live=False)
        self.assertEqual(calls["seat_request"], 1)
        self.assertEqual(p["admission_budget"]["requested_seats"], 1)
        self.assertEqual(p["size"], 1)
        self.assertEqual(p["free_seats"], 1)

    def test_preflight_refusal_hint_surfaces_live_limiter(self) -> None:
        mod = load()
        hint = mod.preflight_refusal_hint({
            "verdict": "REFUSE_AT_CAP",
            "reason": "16/16 live",
            "cap": 1,
            "live": 16,
            "max_workers": 1,
            "host_cap": 16,
            "capacity_limiter": {"primary": "leases", "term": "live",
                                  "raw": {"max_workers": 1, "host_cap": 16}},
        })
        self.assertIsNotNone(hint)
        self.assertEqual(hint["kind"], "live")
        self.assertIn("capacity limiter leases/live", hint["message"])
        self.assertNotIn("rerun with --max-workers", hint["message"])

    def test_seats_bound_the_wave_below_lanes(self) -> None:
        mod = load()
        cands = [{"lane": L, "issues": 9 - i, "tree": [f"{L}/**"]}
                 for i, L in enumerate(["tools", "docs", "model", "ci"])]
        self._wire(mod, seats=[_seat(0)], candidates=cands)   # only ONE seat
        p = mod.evaluate_wave(ROOT, max_workers=4, work_kind="engineering", live=False)
        self.assertEqual(p["size"], 1)
        self.assertEqual(p["lanes"], ["tools"])
        self.assertEqual(p["refusal"], "SEATS_EXHAUSTED")

    def test_no_candidate_lanes_yields_no_lane(self) -> None:
        mod = load()
        self._wire(mod, seats=[_seat(0)], candidates=[])
        p = mod.evaluate_wave(ROOT, max_workers=4, work_kind="engineering", live=False)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "WAVE_NO_LANE")
        self.assertEqual(p["size"], 0)

    def test_no_candidate_lanes_but_self_modify_held_surfaces_hold(self) -> None:
        mod = load()
        self._wire(mod, seats=[_seat(0)], candidates=[])
        mod.lane_candidates = lambda root: {"candidates": [],
                                            "self_modify_held": ["gateway", "kernel"],
                                            "router_error": None}
        p = mod.evaluate_wave(ROOT, max_workers=4, work_kind="engineering", live=False)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "SELF_MODIFY_HOLD")
        self.assertEqual(p["self_modify_held"], ["gateway", "kernel"])

    def test_no_seats_yields_no_seats(self) -> None:
        mod = load()
        cands = [{"lane": "tools", "issues": 9, "tree": ["tools/**"]}]
        self._wire(mod, seats=[], candidates=cands)
        p = mod.evaluate_wave(ROOT, max_workers=4, work_kind="engineering", live=False)
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "WAVE_NO_SEATS")

    def test_live_wave_writes_the_wave_sidecar(self) -> None:
        mod = load()
        cands = [{"lane": "tools", "issues": 9, "tree": ["tools/**"]},
                 {"lane": "docs", "issues": 7, "tree": ["docs/**"]}]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._wire(mod, seats=[_seat(0), _seat(1)], candidates=cands,
                       no_spawn=False)
            # stub the actual launch so nothing real spawns; record the call.
            spawned: list[str] = []
            mod._spawn_wave_member = lambda root, lane, *a, **k: (
                spawned.append(lane) or {"pid": 1000 + len(spawned), "log": "x.log"})
            p = mod.evaluate_wave(root, max_workers=2, work_kind="engineering",
                                  live=True)
            self.assertEqual(p["verdict"], "WAVED")
            self.assertEqual(p["size"], 2)
            self.assertEqual(spawned, ["tools", "docs"])
            # the wave-level sidecar the done-condition names: {wave_id,size,lanes,seats}
            side = root / mod.RUNS_DIRNAME / "dispatch-wave-wave-test.json"
            self.assertTrue(side.exists(), "wave sidecar must be written on a live wave")
            rec = json.loads(side.read_text(encoding="utf-8"))
            self.assertEqual(rec["wave_id"], "wave-test")
            self.assertEqual(rec["size"], 2)
            self.assertEqual(rec["lanes"], ["tools", "docs"])
            self.assertEqual(rec["seats"], ["acct-0", "acct-1"])


class WaveSpawnReadbackTest(unittest.TestCase):
    """#6492: a wave is WAVED only for children that SURVIVE a bounded read-back.

    Witnessed 2026-08-11: `issue_dispatch.py --wave --live` printed `WAVED (ok)` with
    four pids while all four children were gone within seconds and their logs were 0
    bytes — the tick reported Popen success, not worker liveness. These cases pin the
    demotion end to end: the probe, the typed SPAWN_FAILED record, the terminal on-disk
    evidence, the released in-flight marker, and the wave verdict itself."""

    class _DeadProc:
        """A child that has ALREADY exited when the tick reads it back."""

        def __init__(self, pid: int = 7001, returncode: int = 1) -> None:
            self.pid = pid
            self._rc = returncode

        def wait(self, timeout=None):     # noqa: ARG002  (bounded wait, returns at once)
            return self._rc

    class _LiveProc:
        """A healthy child: still running when the bounded wait elapses."""

        def __init__(self, pid: int = 7002) -> None:
            self.pid = pid

        def wait(self, timeout=None):
            raise __import__("subprocess").TimeoutExpired(cmd="claude", timeout=timeout)

    def _spawn(self, mod, proc, runs: Path, lane: str = "tools", probe_s: float = 0.01):
        with warnings.catch_warnings(), \
                mock.patch.object(mod.subprocess, "Popen", lambda *a, **k: proc), \
                mock.patch.object(mod.shutil, "which", lambda x: x):
            warnings.simplefilter("ignore", ResourceWarning)
            return mod.spawn_detached(["claude", "-p", "x"], {}, runs.parent, runs,
                                      lane, probe_s=probe_s)

    def test_dead_child_is_read_back_and_typed_spawn_failed(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d) / "runs"
            out = self._spawn(mod, self._DeadProc(returncode=2), runs)
            early = out["early_exit"]
            self.assertTrue(early["checked"])
            self.assertFalse(early["alive"])
            self.assertEqual(early["returncode"], 2)
            self.assertTrue(early["silent"])          # 0-byte log: the witnessed shape
            failure = mod.spawn_failure_record(out, "tools", seat="acct-0")
            self.assertEqual(failure["verdict"], "SPAWN_FAILED")
            self.assertEqual(failure["returncode"], 2)
            self.assertEqual(failure["lane"], "tools")
            self.assertEqual(failure["seat"], "acct-0")
            # an empty log is the exec-race signature, not an unattributed failure
            self.assertEqual(failure["cause"], "exec_race")

    def test_live_child_is_never_demoted_by_a_silent_log(self) -> None:
        # The whole point of a BOUNDED probe: a healthy worker is silent for many
        # seconds while the agent starts. Silence from a LIVE pid is not a failure.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d) / "runs"
            out = self._spawn(mod, self._LiveProc(), runs)
            self.assertTrue(out["early_exit"]["alive"])
            self.assertIsNone(mod.spawn_failure_record(out, "tools"))

    def test_zero_probe_reproduces_the_old_popen_success_report(self) -> None:
        # The gate is disable-able, and disabling it restores exactly the behaviour
        # that produced the false WAVED — kept as an explicit operator escape.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d) / "runs"
            out = self._spawn(mod, self._DeadProc(), runs, probe_s=0.0)
            self.assertNotIn("early_exit", out)
            self.assertIsNone(mod.spawn_failure_record(out, "tools"))

    def test_dead_spawn_is_retired_with_a_terminal_record_and_no_marker(self) -> None:
        # Done condition 3: the run does not sit on disk as an apparent run. It carries
        # an explicit terminal failure record, and it stops claiming its lane.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d) / "runs"
            out = self._spawn(mod, self._DeadProc(pid=7010), runs, lane="docs")
            marker = Path(out["inflight"])
            self.assertTrue(marker.exists())
            self.assertEqual(mod.busy_lanes(runs, is_alive=lambda pid: True), {"docs"})
            failure = mod.spawn_failure_record(out, "docs", seat="acct-1")
            record = mod.retire_failed_spawn(out, failure)
            self.assertTrue(record.endswith(mod.SPAWN_FAILED_SIDECAR_SUFFIX))
            rec = json.loads(Path(record).read_text(encoding="utf-8"))
            self.assertEqual(rec["verdict"], "SPAWN_FAILED")
            self.assertEqual(rec["lane"], "docs")
            self.assertEqual(rec["returncode"], 1)
            # the dead child no longer holds its lane against the next tick
            self.assertFalse(marker.exists())
            self.assertEqual(mod.busy_lanes(runs, is_alive=lambda pid: True), set())

    def test_probe_window_is_env_overridable_and_defaults_on(self) -> None:
        mod = load()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(mod.WAVE_SPAWN_PROBE_ENV, None)
            self.assertEqual(mod.wave_spawn_probe_s(),
                             mod.DEFAULT_WAVE_SPAWN_PROBE_S)
        self.assertEqual(mod.wave_spawn_probe_s({mod.WAVE_SPAWN_PROBE_ENV: "0"}), 0.0)
        self.assertEqual(mod.wave_spawn_probe_s({mod.WAVE_SPAWN_PROBE_ENV: "1.5"}), 1.5)
        # a typo must not silently disable a safety gate
        self.assertEqual(mod.wave_spawn_probe_s({mod.WAVE_SPAWN_PROBE_ENV: "soon"}),
                         mod.DEFAULT_WAVE_SPAWN_PROBE_S)


class WaveSpawnFailureVerdictTest(unittest.TestCase):
    """#6492, the tick level: what the wave REPORTS when its children die on launch.

    Borrows WaveTest's stub harness (nothing live is invoked) without inheriting its
    cases. Reproduces the reported failure directly — every spawn returns a pid, every
    child is already dead — and pins that the tick no longer calls that a WAVED wave."""

    SPAWN_OK = WaveTest.SPAWN_OK
    _wire = WaveTest._wire

    @staticmethod
    def _dead_spawn(rank_pid: int = 9000):
        """A `_spawn_wave_member` stub whose child died inside the read-back window."""
        n = {"i": 0}

        def spawn(root, lane, seat, wave_id, rank, size, shortfall, **kw):
            n["i"] += 1
            return {"pid": rank_pid + n["i"], "log": f"{lane}.log",
                    "spawn_failed": {"verdict": "SPAWN_FAILED", "cause": "exec_race",
                                     "lane": lane, "seat": seat.get("tag"),
                                     "returncode": 1, "log_bytes": 0, "silent": True}}
        return spawn

    def test_every_child_dead_is_spawn_failed_not_waved(self) -> None:
        mod = load()
        cands = [{"lane": "tools", "issues": 9, "tree": ["tools/**"]},
                 {"lane": "docs", "issues": 7, "tree": ["docs/**"]}]
        with tempfile.TemporaryDirectory() as d:
            self._wire(mod, seats=[_seat(0), _seat(1)], candidates=cands,
                       no_spawn=False)
            mod._spawn_wave_member = self._dead_spawn()
            p = mod.evaluate_wave(Path(d), max_workers=2, work_kind="engineering",
                                  live=True)
            self.assertFalse(p["ok"])                    # was: True
            self.assertEqual(p["verdict"], "SPAWN_FAILED")   # was: "WAVED"
            self.assertEqual(p["size"], 0)               # was: 2 (two pids)
            self.assertEqual(p["lanes"], [])
            self.assertEqual(p["spawn_failed_count"], 2)
            self.assertEqual([f["lane"] for f in p["spawn_failed"]], ["tools", "docs"])
            self.assertIn("no worker survived launch", p["reason"])
            self.assertIn("SPAWN_FAILED", mod.render_wave(p))
            # a wave that lost every child leaves no wave sidecar claiming a live wave
            self.assertFalse((Path(d) / mod.RUNS_DIRNAME /
                              "dispatch-wave-wave-test.json").exists())

    def test_survivor_is_counted_and_the_dead_lane_is_named_not_hidden(self) -> None:
        mod = load()
        cands = [{"lane": "tools", "issues": 9, "tree": ["tools/**"]},
                 {"lane": "docs", "issues": 7, "tree": ["docs/**"]}]
        with tempfile.TemporaryDirectory() as d:
            self._wire(mod, seats=[_seat(0), _seat(1)], candidates=cands,
                       no_spawn=False)

            def spawn(root, lane, seat, wave_id, rank, size, shortfall, **kw):
                if lane == "tools":
                    return {"pid": 9101, "log": "tools.log",
                            "spawn_failed": {"verdict": "SPAWN_FAILED",
                                             "cause": "stale_cred", "lane": lane,
                                             "seat": seat.get("tag"), "returncode": 1,
                                             "log_bytes": 0, "silent": True}}
                return {"pid": 9102, "log": f"{lane}.log"}
            mod._spawn_wave_member = spawn
            p = mod.evaluate_wave(Path(d), max_workers=2, work_kind="engineering",
                                  live=True)
            self.assertTrue(p["ok"])
            self.assertEqual(p["verdict"], "WAVED")
            self.assertEqual(p["size"], 1)               # only the survivor
            self.assertEqual(p["lanes"], ["docs"])
            self.assertEqual(p["spawn_failed_count"], 1)
            self.assertIn("tools=stale_cred", p["reason"])
            self.assertIn("died on launch", p["reason"])
            # the dead lane took its own seat: the survivor did not inherit rank 0's.
            self.assertEqual(p["seats_used"], ["acct-1"])

    def test_a_dead_spawn_holds_no_lease_so_a_colliding_lane_can_still_run(self) -> None:
        # A child that never lived must not hold the wave's disjointness lease: the
        # next candidate on the same tree is free to take the lane.
        mod = load()
        cands = [{"lane": "tools", "issues": 9, "tree": ["tools/**"]},
                 {"lane": "tools2", "issues": 8, "tree": ["tools/**"]}]
        with tempfile.TemporaryDirectory() as d:
            self._wire(mod, seats=[_seat(0), _seat(1)], candidates=cands,
                       no_spawn=False)

            def spawn(root, lane, seat, wave_id, rank, size, shortfall, **kw):
                if lane == "tools":
                    return {"pid": 9201, "log": "tools.log",
                            "spawn_failed": {"verdict": "SPAWN_FAILED",
                                             "cause": "exec_race", "lane": lane,
                                             "seat": seat.get("tag"), "returncode": 1,
                                             "log_bytes": 0, "silent": True}}
                return {"pid": 9202, "log": f"{lane}.log"}
            mod._spawn_wave_member = spawn
            p = mod.evaluate_wave(Path(d), max_workers=2, work_kind="engineering",
                                  live=True)
            self.assertEqual(p["lanes"], ["tools2"])
            self.assertEqual(p["spawn_failed_count"], 1)


class WaveLaunchCacheTest(unittest.TestCase):
    """#3610 cross-worker floor cache-warm: ONE warm pre-request per wave (not per
    member) primes the byte-identical ~35.8k floor prefix, and a stagger knob spaces
    consecutive launches inside the cache TTL so members 2..N read that prefix instead
    of each paying a full cache-write. Both knobs are off/zero by default, so an
    unconfigured wave must launch exactly as it does today.

    Borrows WaveTest's `_wire` stub harness (by reference, not by subclassing — a
    subclass would re-run every WaveTest case) so nothing live spawns and no token is
    spent; the warm runner and the sleep are both injected."""

    SPAWN_OK = WaveTest.SPAWN_OK
    _wire = WaveTest._wire

    def _live_wave(self, mod, root, *, n=3, **kw):
        """A live wave of `n` disjoint lanes with the spawn stubbed out."""
        cands = [{"lane": f"lane{i}", "issues": 9 - i, "tree": [f"lane{i}/**"]}
                 for i in range(n)]
        self._wire(mod, seats=[_seat(i) for i in range(n)], candidates=cands,
                   no_spawn=False)
        self.spawned: list[str] = []
        mod._spawn_wave_member = lambda root, lane, *a, **k: (
            self.spawned.append(lane) or {"pid": 1, "log": "x.log"})
        return mod.evaluate_wave(root, max_workers=n, work_kind="engineering",
                                 live=True, **kw)

    def test_warm_floor_fires_exactly_once_per_wave_not_once_per_member(self) -> None:
        mod = load()
        calls: list[bool] = []
        mod.warm_floor_prefix = lambda root, **kw: (
            calls.append(kw.get("live")) or {"warmed": True})
        with tempfile.TemporaryDirectory() as d:
            p = self._live_wave(mod, Path(d), n=3, warm_floor=True)
        self.assertEqual(len(self.spawned), 3, "all three members must still spawn")
        # THE assertion the issue turns on: 3 members, ONE warm — a per-member warm
        # would pay the floor N times over and defeat the whole optimization.
        self.assertEqual(calls, [True], "warm must fire once per WAVE, not per member")
        self.assertEqual(p["warm_floor"], {"warmed": True})
        self.assertTrue(p["launch_cache"]["warm_floor"])

    def test_warm_floor_is_off_by_default_and_never_fires(self) -> None:
        mod = load()
        def boom(*a, **k):
            raise AssertionError("an unconfigured wave must not issue a warm request")
        mod.warm_floor_prefix = boom
        with mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop(mod.WARM_FLOOR_ENV, None)
            with tempfile.TemporaryDirectory() as d:
                p = self._live_wave(mod, Path(d), n=2)
        self.assertEqual(len(self.spawned), 2)
        self.assertNotIn("warm_floor", p)
        self.assertFalse(p["launch_cache"]["warm_floor"])

    def test_warm_floor_is_skipped_when_the_wave_spawns_nothing(self) -> None:
        mod = load()
        def boom(*a, **k):
            raise AssertionError("a wave with no members must not pay for a warm")
        mod.warm_floor_prefix = boom
        # No seats -> no member ever reaches the spawn arm, so the lazy latch must
        # never fire: warming a prefix nothing will read is pure waste.
        self._wire(mod, seats=[], candidates=[
            {"lane": "tools", "issues": 9, "tree": ["tools/**"]}])
        with tempfile.TemporaryDirectory() as d:
            p = mod.evaluate_wave(Path(d), max_workers=2, work_kind="engineering",
                                  live=True, warm_floor=True)
        self.assertEqual(p["size"], 0)

    def test_stagger_delays_between_consecutive_spawns_only(self) -> None:
        mod = load()
        slept: list[float] = []
        mod._sleep = slept.append
        with tempfile.TemporaryDirectory() as d:
            p = self._live_wave(mod, Path(d), n=3, stagger_s=7.5)
        self.assertEqual(len(self.spawned), 3)
        # 3 members -> 2 gaps. The delay lands BETWEEN launches: member 1 is never
        # held back (nothing is warm yet for it to wait on).
        self.assertEqual(slept, [7.5, 7.5])
        self.assertEqual(p["launch_cache"]["stagger_s"], 7.5)
        self.assertEqual(p["launch_cache"]["staggered_spawns"], 2)

    def test_zero_stagger_reproduces_todays_back_to_back_launch(self) -> None:
        mod = load()
        slept: list[float] = []
        mod._sleep = slept.append
        with mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop(mod.LAUNCH_STAGGER_ENV, None)
            with tempfile.TemporaryDirectory() as d:
                p = self._live_wave(mod, Path(d), n=3)
        self.assertEqual(len(self.spawned), 3)
        self.assertEqual(slept, [], "an unset stagger must never sleep")
        self.assertEqual(p["launch_cache"]["stagger_s"], 0.0)
        self.assertEqual(p["launch_cache"]["staggered_spawns"], 0)

    def test_env_knobs_supply_the_defaults(self) -> None:
        mod = load()
        calls: list[int] = []
        mod.warm_floor_prefix = lambda root, **kw: calls.append(1) or {"warmed": True}
        slept: list[float] = []
        mod._sleep = slept.append
        with mock.patch.dict("os.environ", {mod.WARM_FLOOR_ENV: "1",
                                            mod.LAUNCH_STAGGER_ENV: "2.5"}):
            with tempfile.TemporaryDirectory() as d:
                p = self._live_wave(mod, Path(d), n=2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(slept, [2.5])
        self.assertEqual(p["launch_cache"]["stagger_s"], 2.5)

    def test_dry_run_warm_never_spends_tokens(self) -> None:
        mod = load()
        ran: list[int] = []
        rec = mod.warm_floor_prefix(ROOT, live=False, run=lambda *a, **k: ran.append(1))
        self.assertTrue(rec["would_warm"])
        self.assertEqual(ran, [], "a dry run must not issue the pre-request")

    def test_warm_failure_is_recorded_and_never_blocks_the_wave(self) -> None:
        mod = load()
        def explode(*a, **k):
            raise OSError("no claude binary here")
        rec = mod.warm_floor_prefix(ROOT, live=True, run=explode)
        # FAIL-OPEN: warming is a cost optimization, so a fault must degrade to
        # today's unwarmed launch, never refuse the dispatch.
        self.assertIn("error", rec)
        self.assertFalse(rec.get("warmed"))

    def test_warm_command_shares_the_worker_prefix_and_differs_only_in_the_turn(self):
        mod = load()
        dw = mod.dispatch_worker
        worker = dw.build_command("tools", "claude")
        warm = dw.build_warm_floor_command("claude")
        # The premise of #3610: everything the provider caches (binary + flags, hence
        # system prompt + tool schemas) is identical; ONLY the trailing user turn —
        # which sits AFTER the cached prefix — differs.
        self.assertEqual(warm[:-1], worker[:-1])
        self.assertNotEqual(warm[-1], worker[-1])
        self.assertEqual(warm[-1], dw.WARM_FLOOR_PROMPT)


class WaveFloorCacheMeterTest(unittest.TestCase):
    """#3610 box 3: the per-wave cache read/write meter and its warm-vs-cold A/B.

    Two on-disk planes are planted and folded — the durable wave sidecars and the
    gateway-usage ledger `fak guard` appends one row per session to. The join key is
    the guard pid (internal/gatewayusageledger.NewRow stamps os.Getpid()). The fold is
    pure, so nothing live is invoked.
    """

    SPAWN_OK = WaveTest.SPAWN_OK
    _wire = WaveTest._wire

    def _usage(self, path: Path, *rows: tuple) -> None:
        """Plant gateway-usage rows: (pid, unix_millis, read, write, uncached)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps({
            "schema": "fak-gateway-usage-ledger/1", "kind": "exit",
            "session_type": "guard", "pid": pid, "unix_millis": ms,
            "counters": {"cached_prompt_tokens": read, "cache_creation_tokens": write,
                         "input_tokens": uncached, "observed_turns": 4},
        }) for pid, ms, read, write, uncached in rows) + "\n", encoding="utf-8")

    def _wave(self, runs: Path, wave_id: str, *, warm: bool, at: int,
              members: list[tuple]) -> None:
        """Plant one durable wave sidecar: members are (rank, pid)."""
        runs.mkdir(parents=True, exist_ok=True)
        (runs / f"dispatch-wave-{wave_id}.json").write_text(json.dumps({
            "wave_id": wave_id, "size": len(members), "launched_at_ms": at,
            "launch_cache": {"warm_floor": warm, "stagger_s": 3.0 if warm else 0.0,
                             "staggered_spawns": len(members) - 1 if warm else 0},
            "members": [{"lane": f"lane{r}", "rank": r, "pid": pid}
                        for r, pid in members],
        }), encoding="utf-8")

    def test_live_wave_sidecar_carries_the_join_key_and_posture(self) -> None:
        """The sidecar is what makes the A/B possible at all: last-wave.json is
        overwritten every tick, so without a durable per-wave record carrying the
        posture AND the pids there is no second arm to compare against."""
        mod = load()
        mod.warm_floor_prefix = lambda root, **kw: {"warmed": True}
        mod._sleep = lambda s: None
        cands = [{"lane": f"lane{i}", "issues": 9 - i, "tree": [f"lane{i}/**"]}
                 for i in range(2)]
        self._wire(mod, seats=[_seat(i) for i in range(2)], candidates=cands,
                   no_spawn=False)
        pids = iter([4101, 4102])
        mod._spawn_wave_member = lambda root, lane, *a, **k: {
            "pid": next(pids), "log": f"{lane}.log"}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = mod.evaluate_wave(root, max_workers=2, work_kind="engineering",
                                  live=True, warm_floor=True, stagger_s=3.0)
            sidecar = json.loads(
                (root / mod.RUNS_DIRNAME /
                 f"dispatch-wave-{p['wave_id']}.json").read_text(encoding="utf-8"))
        self.assertTrue(sidecar["launch_cache"]["warm_floor"])
        self.assertEqual(sidecar["launch_cache"]["stagger_s"], 3.0)
        self.assertEqual([m["pid"] for m in sidecar["members"]], [4101, 4102])
        self.assertEqual([m["rank"] for m in sidecar["members"]], [0, 1])
        self.assertIsInstance(sidecar["launched_at_ms"], int)

    def test_followers_reading_a_warm_prefix_beat_the_cold_arm(self) -> None:
        """The acceptance claim: with a warm prefix the followers READ where the cold
        arm WRITES. Rank 0 writes in BOTH arms — someone always pays the first write."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs, usage = Path(d) / "runs", Path(d) / "usage.jsonl"
            self._wave(runs, "warm1", warm=True, at=1000, members=[(0, 11), (1, 12), (2, 13)])
            self._wave(runs, "cold1", warm=False, at=2000, members=[(0, 21), (1, 22), (2, 23)])
            self._usage(usage,
                        # warm: leader writes the floor, followers re-enter it.
                        (11, 1100, 1_000, 35_800, 500),
                        (12, 1100, 36_000, 1_200, 500),
                        (13, 1100, 36_000, 1_100, 500),
                        # cold: every member pays its own floor write.
                        (21, 2100, 1_000, 35_800, 500),
                        (22, 2100, 1_000, 35_800, 500),
                        (23, 2100, 1_000, 35_800, 500))
            doc = mod.wave_floor_cache(Path(d), runs_dir=runs, usage_path=usage)
        self.assertEqual(doc["verdict"], "WARM_READS_MORE")
        self.assertTrue(doc["ok"])
        self.assertEqual(doc["ab"]["warm"]["reading"], 2)
        self.assertEqual(doc["ab"]["cold"]["reading"], 0)
        # Rank 0 is excluded from BOTH arms: it is never evidence either way.
        self.assertEqual(doc["ab"]["warm"]["followers"], 2)
        self.assertEqual(doc["ab"]["cold"]["followers"], 2)
        self.assertGreater(doc["ab"]["warm"]["mean_read_share"],
                           doc["ab"]["cold"]["mean_read_share"])
        self.assertLess(doc["ab"]["warm"]["mean_cache_write"],
                        doc["ab"]["cold"]["mean_cache_write"])

    def test_a_warm_wave_that_did_not_move_the_needle_reports_no_effect(self) -> None:
        """Demotion evidence must be reportable: if warm followers still write, the
        meter has to say so rather than rationalize the knob it is measuring."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs, usage = Path(d) / "runs", Path(d) / "usage.jsonl"
            self._wave(runs, "warm1", warm=True, at=1000, members=[(0, 11), (1, 12)])
            self._wave(runs, "cold1", warm=False, at=2000, members=[(0, 21), (1, 22)])
            self._usage(usage, (11, 1100, 1_000, 35_800, 500),
                        (12, 1100, 1_000, 35_800, 500),
                        (21, 2100, 1_000, 35_800, 500),
                        (22, 2100, 1_000, 35_800, 500))
            doc = mod.wave_floor_cache(Path(d), runs_dir=runs, usage_path=usage)
        self.assertEqual(doc["verdict"], "NO_MEASURED_EFFECT")
        self.assertFalse(doc["ok"])

    def test_a_missing_arm_refuses_to_grade_instead_of_reporting_a_blank_pass(self):
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs, usage = Path(d) / "runs", Path(d) / "usage.jsonl"
            self._wave(runs, "cold1", warm=False, at=2000, members=[(0, 21), (1, 22)])
            self._usage(usage, (21, 2100, 1_000, 35_800, 500),
                        (22, 2100, 1_000, 35_800, 500))
            doc = mod.wave_floor_cache(Path(d), runs_dir=runs, usage_path=usage)
        self.assertEqual(doc["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertFalse(doc["ok"], "an ungraded A/B must exit non-zero")
        self.assertIn("warm", doc["reason"])

    def test_a_usage_row_older_than_the_launch_never_joins(self) -> None:
        """PID-REUSE GUARD: the OS recycles pids and the ledger spans days, so a row
        written before the wave launched belongs to a different process entirely.
        Without this the meter would happily credit a stranger's cache counters."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs, usage = Path(d) / "runs", Path(d) / "usage.jsonl"
            self._wave(runs, "warm1", warm=True, at=5_000, members=[(0, 11), (1, 12)])
            # Same pid, but stamped LONG before this wave launched.
            self._usage(usage, (12, 40, 36_000, 1_000, 500))
            doc = mod.wave_floor_cache(Path(d), runs_dir=runs, usage_path=usage)
        follower = doc["waves"][0]["members"][1]
        self.assertFalse(follower["matched"])
        self.assertIn("no gateway-usage row", follower["reason"])
        self.assertEqual(doc["ab"]["warm"]["followers"], 0)

    def test_an_unmatched_member_is_named_not_silently_dropped(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs, usage = Path(d) / "runs", Path(d) / "usage.jsonl"
            self._wave(runs, "warm1", warm=True, at=1000, members=[(0, 11), (1, 12)])
            self._usage(usage, (11, 1100, 1_000, 35_800, 500))   # follower still live
            doc = mod.wave_floor_cache(Path(d), runs_dir=runs, usage_path=usage)
        members = doc["waves"][0]["members"]
        self.assertTrue(members[0]["matched"])
        self.assertFalse(members[1]["matched"])
        self.assertEqual(len(members), 2, "the unmatched member stays in the readout")

    def test_legacy_sidecars_are_counted_apart_not_folded_into_the_cold_arm(self):
        """A pre-#3610 sidecar records NO posture and NO pids. Folding it into the
        cold arm would show dozens of 'cold waves' contributing zero followers, which
        reads as evidence when it is silence. It is reported as its own count."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs, usage = Path(d) / "runs", Path(d) / "usage.jsonl"
            runs.mkdir(parents=True)
            (runs / "dispatch-wave-old1.json").write_text(json.dumps(
                {"wave_id": "old1", "size": 2, "lanes": ["a", "b"],
                 "seats": ["s1", "s2"]}), encoding="utf-8")
            self._wave(runs, "cold1", warm=False, at=2000, members=[(0, 21), (1, 22)])
            self._usage(usage, (21, 2100, 1_000, 35_800, 500),
                        (22, 2100, 1_000, 35_800, 500))
            doc = mod.wave_floor_cache(Path(d), runs_dir=runs, usage_path=usage)
        self.assertEqual(doc["legacy_waves"], 1)
        self.assertEqual(doc["ab"]["cold"]["waves"], 1, "only the posture-bearing wave")
        self.assertEqual([w["wave_id"] for w in doc["waves"]], ["cold1"])
        self.assertIn("legacy", mod.render_floor_cache(doc))

    def test_a_missing_ledger_is_a_clean_first_run_not_an_error(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            doc = mod.wave_floor_cache(Path(d), runs_dir=Path(d) / "runs",
                                       usage_path=Path(d) / "absent.jsonl")
        self.assertEqual(doc["usage_rows"], 0)
        self.assertEqual(doc["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(doc["waves"], [])

    def test_corrupt_ledger_lines_are_skipped_not_fatal(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            usage = Path(d) / "usage.jsonl"
            usage.write_text('{"nope\nnot json at all\n'
                             '{"schema":"fak-gateway-usage-ledger/1","pid":7,'
                             '"unix_millis":9,"counters":{}}\n', encoding="utf-8")
            rows = mod.read_usage_rows(usage)
        self.assertEqual([r["pid"] for r in rows], [7])

    def test_the_readout_states_its_aggregate_basis(self) -> None:
        """The counters are whole-session aggregates, so the meter cannot isolate the
        ~35.8k floor from in-session writes. That limit must travel WITH the number."""
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            doc = mod.wave_floor_cache(Path(d), runs_dir=Path(d) / "runs",
                                       usage_path=Path(d) / "absent.jsonl")
        self.assertIn("session-aggregate", doc["basis"])
        self.assertIn("NOT separated", doc["basis"])
        self.assertIn("basis", mod.render_floor_cache(doc))


class BusyLanesTest(unittest.TestCase):
    """busy_lanes folds the inflight markers spawn_detached writes into the set of
    lanes with a LIVE worker, pruning dead / stale / garbage markers in one pass so
    the marker set stays bounded without a separate sweeper."""

    def _marker(self, mod, runs: Path, lane: str, pid: int) -> Path:
        runs.mkdir(parents=True, exist_ok=True)
        p = runs / f"{mod.INFLIGHT_PREFIX}{lane}-{pid}.json"
        p.write_text(json.dumps({"lane": lane, "pid": pid}), encoding="utf-8")
        return p

    def test_live_pid_marks_lane_busy_and_dead_is_pruned(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            live = self._marker(mod, runs, "gateway", 111)
            dead = self._marker(mod, runs, "docs", 222)
            busy = mod.busy_lanes(runs, is_alive=lambda pid: pid == 111)
            self.assertEqual(busy, {"gateway"})
            self.assertTrue(live.exists())     # live marker kept
            self.assertFalse(dead.exists())    # dead marker pruned in the same pass

    def test_stale_marker_pruned_even_if_pid_alive(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            m = self._marker(mod, runs, "model", 333)
            old = os.path.getmtime(m) - (mod.INFLIGHT_TTL_SECONDS + 100)
            os.utime(m, (old, old))
            busy = mod.busy_lanes(runs, is_alive=lambda pid: True)
            self.assertEqual(busy, set())
            self.assertFalse(m.exists())

    def test_garbage_marker_is_pruned(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            runs.mkdir(parents=True, exist_ok=True)
            bad = runs / f"{mod.INFLIGHT_PREFIX}x-1.json"
            bad.write_text("not json{", encoding="utf-8")
            busy = mod.busy_lanes(runs, is_alive=lambda pid: True)
            self.assertEqual(busy, set())
            self.assertFalse(bad.exists())

    def test_missing_runs_dir_yields_empty(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(
                mod.busy_lanes(Path(d) / "nope", is_alive=lambda pid: True), set())


class LeaseRefBusyLanesTest(unittest.TestCase):
    """lease_ref_busy_lanes folds cross-session refs into the planner's busy set."""

    def test_live_non_reclaimable_refs_are_busy(self) -> None:
        mod = load()

        class Proc:
            returncode = 0
            stderr = ""
            stdout = json.dumps([
                {"id": "resolve-tools", "liveness": "peer-live",
                 "reclaimable": False, "holder": "node/sess"},
                {"id": "resolve-docs", "liveness": "peer-dead",
                 "reclaimable": True, "holder": "node/dead"},
            ])

        def fake_run(cmd, **kwargs):
            self.assertEqual(cmd[-2:], ["leaseref", "liveness"])
            return Proc()

        mod._fak_cmd = lambda: ["fak"]
        with mock.patch.object(mod.subprocess, "run", fake_run):
            out = mod.lease_ref_busy_lanes(ROOT)
        self.assertEqual(out["lanes"], {"tools"})
        self.assertEqual(out["records"][0]["lane"], "tools")

    def test_unavailable_fak_fails_open(self) -> None:
        mod = load()
        mod._fak_cmd = lambda: None
        out = mod.lease_ref_busy_lanes(ROOT)
        self.assertEqual(out["lanes"], set())
        self.assertIn("no fak binary", out["error"])

    def test_fak_bin_split_preserves_windows_backslashes(self) -> None:
        mod = load()
        with mock.patch.dict(os.environ, {"FAK_BIN": r"C:\work\fak\fak.exe"}):
            with mock.patch.object(mod.os, "name", "nt"):
                self.assertEqual(mod._fak_cmd(), [r"C:\work\fak\fak.exe"])


class BusyAccountsTest(unittest.TestCase):
    """busy_accounts folds the SAME inflight markers into the set of ACCOUNTS with a
    LIVE worker (#2060 cross-tick account de-confliction), self-healing dead markers in
    one pass exactly like busy_lanes."""

    def _marker(self, mod, runs: Path, lane: str, pid: int, account) -> Path:
        runs.mkdir(parents=True, exist_ok=True)
        p = runs / f"{mod.INFLIGHT_PREFIX}{lane}-{pid}.json"
        rec = {"lane": lane, "pid": pid}
        if account is not None:
            rec["account"] = account
        p.write_text(json.dumps(rec), encoding="utf-8")
        return p

    def test_live_account_busy_and_dead_pruned(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            live = self._marker(mod, runs, "docs", 111, r"C:\U\.claude")
            dead = self._marker(mod, runs, "tools", 222, r"C:\U\.claude-day30-netra")
            busy = mod.busy_accounts(runs, is_alive=lambda pid: pid == 111)
            self.assertEqual(busy, {r"C:\U\.claude"})   # only the live worker's account
            self.assertTrue(live.exists())
            self.assertFalse(dead.exists())             # dead marker pruned

    def test_marker_without_account_contributes_nothing_but_is_kept(self) -> None:
        # An older marker predating the account field: a live pid keeps the marker
        # (busy_lanes still needs it) but it adds no account to the busy set.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            m = self._marker(mod, runs, "docs", 111, None)
            self.assertEqual(mod.busy_accounts(runs, is_alive=lambda pid: True), set())
            self.assertTrue(m.exists())

    def test_write_marker_records_account_and_busy_accounts_reads_it(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            mod._write_inflight_marker(runs, "docs", 4242, account=r"C:\U\.claude")
            self.assertEqual(
                mod.busy_accounts(runs, is_alive=lambda pid: pid == 4242),
                {r"C:\U\.claude"})

    def test_stale_marker_pruned_even_if_pid_alive(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            m = self._marker(mod, runs, "model", 333, r"C:\U\.claude")
            old = os.path.getmtime(m) - (mod.INFLIGHT_TTL_SECONDS + 100)
            os.utime(m, (old, old))
            self.assertEqual(mod.busy_accounts(runs, is_alive=lambda pid: True), set())
            self.assertFalse(m.exists())

    def test_missing_runs_dir_yields_empty(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(
                mod.busy_accounts(Path(d) / "nope", is_alive=lambda pid: True), set())


class PickLaneBusyTest(unittest.TestCase):
    """pick_lane prefers the richest lane NOT already in flight; falls back to the
    richest overall (flagged ``stacked``) only when every lane is busy."""

    LANES = {"lanes": {"docs": {"issues": [1, 2]},
                       "gateway": {"issues": [1, 2, 3, 4]},
                       "recall": {"issues": [9]}}}

    def _router(self, mod) -> None:
        mod.run_json = lambda cmd, cwd, timeout: json.loads(json.dumps(self.LANES))

    def test_busy_richest_lane_is_skipped_for_next_free(self) -> None:
        mod = load()
        self._router(mod)
        pick = mod.pick_lane(ROOT, None, busy={"gateway"})
        self.assertEqual(pick["lane"], "docs")     # gateway (4) busy -> docs (2)
        self.assertEqual(pick["issues"], 2)
        self.assertFalse(pick["stacked"])
        self.assertEqual(pick["busy"], ["gateway"])

    def test_three_lane_pressure_rotates_second_spawn_to_another_lane(self) -> None:
        # #1774 witness: gateway stays the highest-pressure lane, but once it is
        # in flight the next pick moves to docs, then recall, before stacking.
        mod = load()
        self._router(mod)
        first = mod.pick_lane(ROOT, None, busy=set())
        second = mod.pick_lane(ROOT, None, busy={first["lane"]})
        third = mod.pick_lane(ROOT, None, busy={first["lane"], second["lane"]})
        stacked = mod.pick_lane(ROOT, None, busy={"docs", "gateway", "recall"})

        self.assertEqual(first["lane"], "gateway")
        self.assertEqual(second["lane"], "docs")
        self.assertEqual(third["lane"], "recall")
        self.assertFalse(second["stacked"])
        self.assertFalse(third["stacked"])
        self.assertTrue(stacked["stacked"])

    def test_all_busy_falls_back_to_richest_and_flags_stacked(self) -> None:
        mod = load()
        self._router(mod)
        pick = mod.pick_lane(ROOT, None, busy={"docs", "gateway", "recall"})
        self.assertEqual(pick["lane"], "gateway")  # all busy -> richest overall
        self.assertTrue(pick["stacked"])

    def test_no_busy_matches_legacy_richest(self) -> None:
        mod = load()
        self._router(mod)
        pick = mod.pick_lane(ROOT, None, busy=set())
        self.assertEqual(pick["lane"], "gateway")
        self.assertFalse(pick["stacked"])

    def test_explicit_lane_honored_despite_busy(self) -> None:
        mod = load()
        self._router(mod)
        pick = mod.pick_lane(ROOT, "gateway", busy={"gateway"})
        self.assertEqual(pick["lane"], "gateway")
        self.assertTrue(pick["explicit"])


class IsSelfSourceTreeTest(unittest.TestCase):
    def test_cmd_and_internal_prefixes_are_self_source(self) -> None:
        mod = load()
        self.assertTrue(mod.is_self_source_tree(["cmd/**"]))
        self.assertTrue(mod.is_self_source_tree(["internal/kernel/**"]))

    def test_other_trees_are_not_self_source(self) -> None:
        mod = load()
        self.assertFalse(mod.is_self_source_tree(["docs/**", "README.md"]))
        self.assertFalse(mod.is_self_source_tree(["tools/**", "scripts/**"]))
        self.assertFalse(mod.is_self_source_tree(None))
        self.assertFalse(mod.is_self_source_tree([]))


class PickLaneSelfModifyHoldTest(unittest.TestCase):
    """Proactive pre-route hold: a lane whose tree is fak's own source (cmd/**,
    internal/**) is excluded from the automatic pick under guard, mirroring the
    native Go dispatch path's SELF_MODIFY_HOLD (internal/dispatchtick/selfmodify.go).
    Previously this legacy Python path had no proactive check at all."""

    LANES = {"lanes": {
        "docs": {"issues": [1, 2], "tree": ["docs/**"]},
        "gateway": {"issues": [1, 2, 3, 4], "tree": ["internal/gateway/**"]},
        "tools": {"issues": [9], "tree": ["tools/**", "scripts/**"]},
    }}

    def _router(self, mod) -> None:
        mod.run_json = lambda cmd, cwd, timeout: json.loads(json.dumps(self.LANES))

    def test_guarded_skips_self_source_lane_for_richest_safe(self) -> None:
        mod = load()
        self._router(mod)
        pick = mod.pick_lane(ROOT, None, guarded=True)
        self.assertEqual(pick["lane"], "docs")   # gateway (4, self-source) excluded
        self.assertEqual(pick["self_modify_held"], ["gateway"])

    def test_unguarded_does_not_hold_self_source_lane(self) -> None:
        mod = load()
        self._router(mod)
        pick = mod.pick_lane(ROOT, None, guarded=False)
        self.assertEqual(pick["lane"], "gateway")
        self.assertEqual(pick["self_modify_held"], [])

    def test_all_dispatchable_lanes_self_source_yields_no_lane(self) -> None:
        mod = load()
        mod.run_json = lambda cmd, cwd, timeout: {"lanes": {
            "gateway": {"issues": [1, 2], "tree": ["internal/gateway/**"]},
            "kernel": {"issues": [9], "tree": ["internal/kernel/**"]},
        }}
        pick = mod.pick_lane(ROOT, None, guarded=True)
        self.assertIsNone(pick["lane"])
        self.assertEqual(pick["self_modify_held"], ["gateway", "kernel"])

    def test_explicit_lane_honored_despite_self_source(self) -> None:
        mod = load()
        self._router(mod)
        pick = mod.pick_lane(ROOT, "gateway", guarded=True)
        self.assertEqual(pick["lane"], "gateway")
        self.assertTrue(pick["explicit"])

    def test_self_source_lane_is_hard_excluded_even_when_all_others_busy(self) -> None:
        # Busy-lane fallback must NEVER resurrect a self-source lane -- that would
        # spawn exactly the build-poisoning risk the guard exists to prevent.
        mod = load()
        self._router(mod)
        pick = mod.pick_lane(ROOT, None, busy={"docs", "tools"}, guarded=True)
        self.assertEqual(pick["lane"], "docs")   # busy fallback stays within safe pool
        self.assertTrue(pick["stacked"])
        self.assertEqual(pick["self_modify_held"], ["gateway"])

    def test_default_guarded_reads_dispatch_worker_guard_enabled(self) -> None:
        mod = load()
        self._router(mod)
        mod.dispatch_worker.guard_enabled = lambda *a, **k: True
        pick = mod.pick_lane(ROOT, None)
        self.assertEqual(pick["self_modify_held"], ["gateway"])


class LaneCandidatesSelfModifyHoldTest(unittest.TestCase):
    """lane_candidates (the wave path's picker) applies the same proactive
    self-source hold as pick_lane (the single-tick path), so #1335 wave dispatch
    can't route straight at fak's own source either."""

    LANES = {"lanes": {
        "docs": {"issues": [1, 2], "tree": ["docs/**"]},
        "gateway": {"issues": [1, 2, 3, 4], "tree": ["internal/gateway/**"]},
        "cmd": {"issues": [9], "tree": ["cmd/**"]},
    }}

    def _router(self, mod) -> None:
        mod.run_json = lambda cmd, cwd, timeout: json.loads(json.dumps(self.LANES))

    def test_guarded_excludes_self_source_lanes(self) -> None:
        mod = load()
        self._router(mod)
        cand = mod.lane_candidates(ROOT, guarded=True)
        self.assertEqual([c["lane"] for c in cand["candidates"]], ["docs"])
        self.assertEqual(cand["self_modify_held"], ["cmd", "gateway"])

    def test_unguarded_keeps_self_source_lanes(self) -> None:
        mod = load()
        self._router(mod)
        cand = mod.lane_candidates(ROOT, guarded=False)
        self.assertEqual({c["lane"] for c in cand["candidates"]}, {"docs", "gateway", "cmd"})
        self.assertEqual(cand["self_modify_held"], [])


class SpawnInflightMarkerTest(unittest.TestCase):
    """spawn_detached stamps an inflight {lane, pid} marker that busy_lanes reads
    back — the write end of the cross-tick de-confliction signal."""

    def test_spawn_detached_writes_marker_busy_lanes_reads_it(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d) / "runs"

            class FakeProc:
                pid = 4242

            # spawn_detached intentionally leaks the log file handle so the detached
            # child inherits the fd; the fake Popen does not, so ignore the harmless
            # ResourceWarning the unclosed handle raises under the test's GC.
            with warnings.catch_warnings(), \
                 mock.patch.object(mod.subprocess, "Popen", lambda *a, **k: FakeProc()), \
                 mock.patch.object(mod.shutil, "which", lambda x: x):
                warnings.simplefilter("ignore", ResourceWarning)
                out = mod.spawn_detached(["claude", "-p", "x"], {}, Path(d),
                                         runs, "gateway")
            self.assertEqual(out["pid"], 4242)
            marker = runs / f"{mod.INFLIGHT_PREFIX}gateway-4242.json"
            self.assertTrue(marker.exists())
            rec = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(rec["lane"], "gateway")
            self.assertEqual(rec["pid"], 4242)
            self.assertEqual(out["inflight"], str(marker))
            busy = mod.busy_lanes(runs, is_alive=lambda pid: pid == 4242)
            self.assertEqual(busy, {"gateway"})

    def test_spawn_detached_records_pinned_account_for_busy_accounts(self) -> None:
        # The marker also carries the worker's pinned CLAUDE_CONFIG_DIR so a later
        # tick's busy_accounts avoids double-loading that account (#2060).
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d) / "runs"

            class FakeProc:
                pid = 4243

            with warnings.catch_warnings(), \
                 mock.patch.object(mod.subprocess, "Popen", lambda *a, **k: FakeProc()), \
                 mock.patch.object(mod.shutil, "which", lambda x: x):
                warnings.simplefilter("ignore", ResourceWarning)
                out = mod.spawn_detached(
                    ["claude", "-p", "x"],
                    {"CLAUDE_CONFIG_DIR": r"C:\U\.claude-day30-netra"},
                    Path(d), runs, "tools")
            rec = json.loads(Path(out["inflight"]).read_text(encoding="utf-8"))
            self.assertEqual(rec["account"], r"C:\U\.claude-day30-netra")
            self.assertEqual(
                mod.busy_accounts(runs, is_alive=lambda pid: pid == 4243),
                {r"C:\U\.claude-day30-netra"})

    def _spawn(self, mod, runs: Path, lane: str, pid: int, **kw):
        class FakeProc:
            pass
        FakeProc.pid = pid
        with warnings.catch_warnings(), \
                mock.patch.object(mod.subprocess, "Popen", lambda *a, **k: FakeProc()), \
                mock.patch.object(mod.shutil, "which", lambda x: x):
            warnings.simplefilter("ignore", ResourceWarning)
            out = mod.spawn_detached(["claude", "-p", "x"], {}, runs.parent, runs, lane,
                                     **kw)
        return json.loads(Path(out["inflight"]).read_text(encoding="utf-8"))

    def test_spawn_detached_records_guarded_true_by_default(self) -> None:
        # The default worker is guard-fronted, so the marker self-describes guarded=True
        # and guarded_worker_in_flight can read the build-integrity gate off it.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            rec = self._spawn(mod, Path(d) / "runs", "gateway", 5100)
            self.assertIs(rec["guarded"], True)

    def test_spawn_detached_records_guarded_false_for_escape_worker(self) -> None:
        # An explicitly-unguarded escape worker stamps guarded=False so a later tick can
        # tell it apart from a guard-fronted peer.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            rec = self._spawn(mod, Path(d) / "runs", "internal-accounts", 5101,
                              guarded=False)
            self.assertIs(rec["guarded"], False)


class GuardedWorkerInFlightTest(unittest.TestCase):
    """guarded_worker_in_flight folds the same inflight markers as busy_lanes and reports
    whether any LIVE worker is guarded — the build-integrity gate an unguarded escape
    must clear before it lands a self-source commit on top of a possibly-red build."""

    def _marker(self, mod, runs: Path, lane: str, pid: int, guarded) -> Path:
        runs.mkdir(parents=True, exist_ok=True)
        p = runs / f"{mod.INFLIGHT_PREFIX}{lane}-{pid}.json"
        rec = {"lane": lane, "pid": pid}
        if guarded is not None:
            rec["guarded"] = guarded
        p.write_text(json.dumps(rec), encoding="utf-8")
        return p

    def test_live_guarded_marker_closes_the_gate(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._marker(mod, runs, "gateway", 111, True)
            self.assertTrue(
                mod.guarded_worker_in_flight(runs, is_alive=lambda pid: True))

    def test_live_unguarded_marker_leaves_gate_open(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._marker(mod, runs, "internal-accounts", 222, False)
            self.assertFalse(
                mod.guarded_worker_in_flight(runs, is_alive=lambda pid: True))

    def test_missing_guarded_field_is_treated_as_guarded(self) -> None:
        # A marker predating the escape work has no `guarded` field; back then every
        # worker was guarded, so it must NOT open the gate by omission.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._marker(mod, runs, "docs", 333, None)
            self.assertTrue(
                mod.guarded_worker_in_flight(runs, is_alive=lambda pid: True))

    def test_dead_guarded_marker_pruned_and_gate_open(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            m = self._marker(mod, runs, "gateway", 444, True)
            self.assertFalse(
                mod.guarded_worker_in_flight(runs, is_alive=lambda pid: False))
            self.assertFalse(m.exists())     # dead marker pruned in the same pass

    def test_stale_guarded_marker_pruned_and_gate_open(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            m = self._marker(mod, runs, "gateway", 555, True)
            old = os.path.getmtime(m) - (mod.INFLIGHT_TTL_SECONDS + 100)
            os.utime(m, (old, old))
            self.assertFalse(
                mod.guarded_worker_in_flight(runs, is_alive=lambda pid: True))
            self.assertFalse(m.exists())

    def test_missing_runs_dir_leaves_gate_open(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(
                mod.guarded_worker_in_flight(Path(d) / "nope",
                                             is_alive=lambda pid: True))

    def test_guarded_peer_closes_gate_even_beside_an_unguarded_worker(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._marker(mod, runs, "internal-accounts", 666, False)
            self._marker(mod, runs, "gateway", 667, True)
            self.assertTrue(
                mod.guarded_worker_in_flight(runs, is_alive=lambda pid: True))


class EscapeSignalTest(unittest.TestCase):
    """The Arm-1 escape SIGNAL: held_self_source_lanes → rank_held_by_triage →
    escape_candidates surface the held self-source P1s a guarded tick legally cannot
    spawn, marking the top one `preferred` iff it out-scores the best dispatchable lane —
    so the loop stops silently grinding docs while a P1 sits unreachable."""

    # A router with: a self-source P1 lane, a self-source P2 lane, a self-source lane
    # with NO open issues, and a non-self-source docs lane. Only the first two are held.
    ROUTER = {"lanes": {
        "internal-accounts": {"issues": [3091], "tree": ["internal/accounts/**"]},
        "cmd-fak": {"issues": [2036, 2037], "tree": ["cmd/fak/**"]},
        "internal-empty": {"issues": [], "tree": ["internal/empty/**"]},
        "docs": {"issues": [10, 11], "tree": ["docs/**"]},
    }}

    def _stub(self, mod, router=None, scores=None):
        router = self.ROUTER if router is None else router
        mod.run_json = lambda cmd, cwd, timeout: router
        if scores is not None:
            mod.lane_priority_scores = lambda root, mapping: scores

    def test_held_keeps_only_self_source_lanes_with_open_issues(self) -> None:
        mod = load()
        self._stub(mod)
        held = mod.held_self_source_lanes(ROOT)
        lanes = {h["lane"] for h in held["held"]}
        self.assertEqual(lanes, {"internal-accounts", "cmd-fak"})
        acct = next(h for h in held["held"] if h["lane"] == "internal-accounts")
        self.assertEqual(acct["issue_nums"], [3091])
        self.assertEqual(acct["tree"], ["internal/accounts/**"])
        self.assertIsNone(held["router_error"])

    def test_held_fail_open_on_router_error(self) -> None:
        mod = load()
        self._stub(mod, router={"_error": "router boom"})
        held = mod.held_self_source_lanes(ROOT)
        self.assertEqual(held["held"], [])
        self.assertEqual(held["router_error"], "router boom")

    def test_rank_orders_by_score_then_count_then_name(self) -> None:
        mod = load()
        self._stub(mod, scores={"internal-accounts": 740, "cmd-fak": 300})
        ranked = mod.rank_held_by_triage(
            ROOT, mod.held_self_source_lanes(ROOT)["held"])
        self.assertEqual([r["lane"] for r in ranked],
                         ["internal-accounts", "cmd-fak"])
        self.assertEqual(ranked[0]["score"], 740)

    def test_rank_fail_open_collapses_to_count_then_name(self) -> None:
        # A triage read error collapses every score to 0; the deterministic
        # count-then-name order survives (cmd-fak has 2 issues, internal-accounts 1).
        mod = load()
        self._stub(mod, scores={})
        ranked = mod.rank_held_by_triage(
            ROOT, mod.held_self_source_lanes(ROOT)["held"])
        self.assertEqual([r["lane"] for r in ranked],
                         ["cmd-fak", "internal-accounts"])
        self.assertTrue(all(r["score"] == 0 for r in ranked))

    def test_escape_preferred_when_top_held_outscores_best_safe(self) -> None:
        mod = load()
        self._stub(mod, scores={"internal-accounts": 740, "cmd-fak": 300})
        out = mod.escape_candidates(ROOT, {"docs": 100, "recall": 90})
        self.assertEqual(out["top_held_score"], 740)
        self.assertEqual(out["best_safe_score"], 100)
        self.assertTrue(out["escape_candidates"][0]["preferred"])
        self.assertFalse(out["escape_candidates"][1]["preferred"])

    def test_escape_not_preferred_when_best_safe_outscores(self) -> None:
        mod = load()
        self._stub(mod, scores={"internal-accounts": 740, "cmd-fak": 300})
        out = mod.escape_candidates(ROOT, {"gateway": 900})
        self.assertEqual(out["top_held_score"], 740)
        self.assertEqual(out["best_safe_score"], 900)
        self.assertFalse(any(r["preferred"] for r in out["escape_candidates"]))

    def test_escape_empty_when_no_self_source_held(self) -> None:
        mod = load()
        self._stub(mod, router={"lanes": {"docs": {"issues": [1], "tree": ["docs/**"]}}},
                   scores={})
        out = mod.escape_candidates(ROOT, {"docs": 5})
        self.assertEqual(out["escape_candidates"], [])
        self.assertIsNone(out["top_held_score"])


class HeldForeverSurfaceTest(unittest.TestCase):
    """#3125: a self-source issue held by the picker is explicitly reported — per
    issue, with a persisted consecutive-ticks-held counter — never silently dropped
    from the tick surface."""

    def test_fold_increments_across_ticks_and_resets_on_release(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            held = [{"lane": "cmd-fak", "issue_nums": [2514]}]
            self.assertEqual(mod.fold_held_ticks(runs, held), {"cmd-fak": 1})
            self.assertEqual(mod.fold_held_ticks(runs, held), {"cmd-fak": 2})
            # a lane that leaves the held set resets — consecutive, not lifetime.
            self.assertEqual(mod.fold_held_ticks(runs, []), {})
            self.assertEqual(mod.fold_held_ticks(runs, held), {"cmd-fak": 1})

    def test_clear_held_ticks_removes_the_counter(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            mod.fold_held_ticks(runs, [{"lane": "cmd-fak", "issue_nums": [2514]}])
            self.assertTrue((runs / mod.HELD_TICKS_BASENAME).exists())
            mod.clear_held_ticks(runs)
            self.assertFalse((runs / mod.HELD_TICKS_BASENAME).exists())

    def test_report_is_per_issue_and_never_invisible(self) -> None:
        mod = load()
        rows = mod.held_report(
            [{"lane": "cmd-fak", "issue_nums": [2514, 3091]},
             {"lane": "gateway", "issue_nums": []}],
            {"cmd-fak": 3, "gateway": 1})
        self.assertEqual([r["issue"] for r in rows], [2514, 3091, None])
        self.assertTrue(all(r["status"] == "held, needs unguarded/worktree lane"
                            for r in rows))
        self.assertEqual(rows[0]["consecutive_ticks"], 3)
        self.assertEqual(rows[2]["lane"], "gateway")

    def test_evaluate_attaches_report_and_counter_increments_across_ticks(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            mod.refresh_registry = lambda root: {"ok": True}
            mod.preflight = lambda root, **kw: EvaluateTest.SPAWN_OK
            mod.busy_lanes = lambda runs_dir, **kw: set()
            mod.lease_ref_busy_lanes = lambda root: {"lanes": set()}
            mod.pick_lane = lambda root, explicit, busy=None: {
                "lane": "docs", "issues": 2, "by_lane": {},
                "self_modify_held": ["cmd-fak"], "priority_by_lane": {"docs": 10}}
            mod.escape_candidates = lambda root, safe: {
                "escape_candidates": [{"lane": "cmd-fak", "issue_nums": [2514],
                                       "score": 700, "preferred": True}],
                "top_held_score": 700, "best_safe_score": 10}
            p1 = mod.evaluate(root, max_workers=2, work_kind="engineering",
                              lane=None, live=False)
            self.assertEqual([r["issue"] for r in p1["self_modify_held_report"]],
                             [2514])
            self.assertEqual(p1["self_modify_held_ticks"], {"cmd-fak": 1})
            p2 = mod.evaluate(root, max_workers=2, work_kind="engineering",
                              lane=None, live=False)
            self.assertEqual(p2["self_modify_held_ticks"], {"cmd-fak": 2})
            self.assertEqual(p2["self_modify_held_report"][0]["consecutive_ticks"], 2)
            # the human render lists the held issue explicitly — visibly triaged.
            self.assertIn("#2514", mod.render(p2))
            self.assertIn("held, needs unguarded/worktree lane", mod.render(p2))
            # an unheld tick resets the counter — consecutive, not lifetime.
            mod.pick_lane = lambda root, explicit, busy=None: {
                "lane": "docs", "issues": 2, "by_lane": {}}
            mod.evaluate(root, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
            self.assertFalse(
                (root / mod.RUNS_DIRNAME / mod.HELD_TICKS_BASENAME).exists())


class EscapeRenderTest(unittest.TestCase):
    """render()/render_wave() surface the escape SIGNAL so the payload-only signal is
    visible to an operator — but ONLY when a self-source lane is held, so a normal tick's
    render stays byte-for-byte unchanged."""

    def _payload(self, **over):
        p = {"verdict": "WOULD_SPAWN", "ok": True, "live": 0, "lane": "docs",
             "lane_issue_count": 2, "reason": "picked docs",
             "escape_candidates": [
                 {"lane": "internal-accounts", "issue_nums": [3091], "score": 740,
                  "preferred": True},
                 {"lane": "cmd-fak", "issue_nums": [2036, 2037], "score": 300,
                  "preferred": False}],
             "top_held_score": 740, "best_safe_score": 100,
             "guarded_worker_in_flight": False}
        p.update(over)
        return p

    def test_render_surfaces_top_held_prefer_verb_and_more_count(self) -> None:
        mod = load()
        out = mod.render(self._payload())
        self.assertIn("escape    :", out)
        self.assertIn("internal-accounts", out)
        self.assertIn("#3091", out)
        self.assertIn("score=740", out)
        self.assertIn("vs safe 100", out)
        self.assertIn("PREFER --escape-self-source", out)
        self.assertIn("(+1 more held)", out)

    def test_render_gate_clear_when_no_guarded_build(self) -> None:
        mod = load()
        out = mod.render(self._payload(guarded_worker_in_flight=False))
        self.assertIn("gate: clear (no guarded build in flight)", out)

    def test_render_gate_holds_when_guarded_build_in_flight(self) -> None:
        mod = load()
        out = mod.render(self._payload(guarded_worker_in_flight=True))
        self.assertIn("guarded build in flight — hold the unguarded escape", out)

    def test_render_held_but_not_preferred_omits_gate(self) -> None:
        mod = load()
        cands = [{"lane": "internal-accounts", "issue_nums": [3091], "score": 740,
                  "preferred": False}]
        out = mod.render(self._payload(escape_candidates=cands, best_safe_score=900,
                                       guarded_worker_in_flight=True))
        self.assertIn("held (best safe lane wins)", out)
        self.assertNotIn("gate:", out)     # gate only matters once escape is preferred

    def test_normal_tick_render_has_no_escape_line(self) -> None:
        mod = load()
        p = {"verdict": "WOULD_SPAWN", "ok": True, "live": 0, "lane": "gateway",
             "lane_issue_count": 4, "reason": "picked gateway",
             "guarded_worker_in_flight": False}
        self.assertNotIn("escape    :", mod.render(p))

    def test_render_wave_surfaces_escape(self) -> None:
        mod = load()
        p = {"verdict": "WAVE_OK", "ok": True, "live": 0, "reason": "wave of 2",
             "escape_candidates": [
                 {"lane": "internal-accounts", "issue_nums": [3091], "score": 740,
                  "preferred": True}],
             "top_held_score": 740, "best_safe_score": 100,
             "guarded_worker_in_flight": True}
        out = mod.render_wave(p)
        self.assertIn("escape    :", out)
        self.assertIn("PREFER --escape-self-source", out)
        self.assertIn("guarded build in flight", out)


class EscapePlanTest(unittest.TestCase):
    """escape_plan turns the SIGNAL into the read-only operator ACTION: the top held
    self-source P1 must (1) out-score the best safe lane, (2) clear the guarded
    build-integrity gate, and (3) find its lane free — only then does it emit
    recommend=True with the unguarded FLEET_DOGFOOD_GUARD=0 command. Pure plan: no spawn."""

    ROUTER = {"lanes": {
        "swebench": {"issues": [3106, 1012], "tree": ["internal/swebench/**"]},
        "docs": {"issues": [10], "tree": ["docs/**"]},
    }}

    def _stub(self, mod, *, scores, gate=False, busy=None, lease=None, router=None):
        mod.run_json = lambda cmd, cwd, timeout: (router or self.ROUTER)
        mod.lane_priority_scores = lambda root, mapping: scores
        mod.guarded_worker_in_flight = lambda runs_dir, **kw: gate
        mod.busy_lanes = lambda runs_dir, **kw: set(busy or [])
        mod.lease_ref_busy_lanes = lambda root: {"lanes": set(lease or [])}

    def test_recommend_when_preferred_gate_clear_lane_free(self) -> None:
        mod = load()
        self._stub(mod, scores={"swebench": 740})
        plan = mod.escape_plan(ROOT, {"docs": 100})["escape_plan"]
        self.assertTrue(plan["recommend"])
        self.assertEqual(plan["target_lane"], "swebench")
        self.assertEqual(plan["issue_nums"], [3106, 1012])
        self.assertEqual(plan["env_overrides"], {"FLEET_DOGFOOD_GUARD": "0"})
        self.assertEqual(plan["lease"]["mode"], "exclusive")
        self.assertEqual(plan["lease"]["tree"], ["internal/swebench/**"])
        self.assertIn("swebench", " ".join(plan["command"]))
        self.assertTrue(plan["gate"]["clear"])
        self.assertTrue(plan["lane_free"])

    def test_hold_when_guarded_build_in_flight(self) -> None:
        mod = load()
        self._stub(mod, scores={"swebench": 740}, gate=True)
        plan = mod.escape_plan(ROOT, {"docs": 100})["escape_plan"]
        self.assertFalse(plan["recommend"])
        self.assertFalse(plan["gate"]["clear"])
        self.assertIn("guarded", plan["reason"])

    def test_hold_when_lane_already_in_flight(self) -> None:
        mod = load()
        self._stub(mod, scores={"swebench": 740}, busy=["swebench"])
        plan = mod.escape_plan(ROOT, {"docs": 100})["escape_plan"]
        self.assertFalse(plan["recommend"])
        self.assertFalse(plan["lane_free"])
        self.assertIn("in flight", plan["reason"])

    def test_hold_when_lane_busy_via_lease_ref(self) -> None:
        mod = load()
        self._stub(mod, scores={"swebench": 740}, lease=["swebench"])
        plan = mod.escape_plan(ROOT, {"docs": 100})["escape_plan"]
        self.assertFalse(plan["recommend"])
        self.assertFalse(plan["lane_free"])

    def test_no_escape_when_best_safe_outscores(self) -> None:
        mod = load()
        self._stub(mod, scores={"swebench": 740})
        plan = mod.escape_plan(ROOT, {"gateway": 900})["escape_plan"]
        self.assertFalse(plan["recommend"])
        self.assertIsNone(plan["target_lane"])
        self.assertIn("out-score", plan["reason"])

    def test_no_escape_when_no_self_source_held(self) -> None:
        mod = load()
        self._stub(mod, scores={},
                   router={"lanes": {"docs": {"issues": [1], "tree": ["docs/**"]}}})
        plan = mod.escape_plan(ROOT, {"docs": 5})["escape_plan"]
        self.assertFalse(plan["recommend"])
        self.assertIsNone(plan["target_lane"])
        self.assertIn("no held self-source", plan["reason"])

    def test_render_escape_shows_run_command_when_recommended(self) -> None:
        mod = load()
        self._stub(mod, scores={"swebench": 740})
        out = mod.render_escape(mod.escape_plan(ROOT, {"docs": 100}))
        self.assertIn("RECOMMEND", out)
        self.assertIn("swebench", out)
        self.assertIn("FLEET_DOGFOOD_GUARD=0", out)
        self.assertIn("run     :", out)

    def test_render_escape_hold_omits_run_line(self) -> None:
        mod = load()
        self._stub(mod, scores={"swebench": 740}, gate=True)
        out = mod.render_escape(mod.escape_plan(ROOT, {"docs": 100}))
        self.assertIn("HOLD", out)
        self.assertNotIn("run     :", out)


class EvaluateBusyWiringTest(unittest.TestCase):
    """The single tick computes busy_lanes, threads it into pick_lane, and surfaces
    it in the tick payload."""

    def test_busy_threads_into_pick_lane_and_payload(self) -> None:
        mod = load()
        mod.refresh_registry = lambda root: {"ok": True}
        mod.preflight = lambda root, **kw: {
            "verdict": "SPAWN_OK", "cap": 2, "live": 0,
            "account": {"tag": "a", "tier": 1, "dir": "/a"}}
        mod.busy_lanes = lambda runs_dir, **kw: {"gateway"}
        mod.lease_ref_busy_lanes = lambda root: {"lanes": {"docs"}}
        seen: dict = {}

        def pick(root, explicit, busy=None):
            seen["busy"] = busy
            return {"lane": "tools", "issues": 2, "by_lane": {}, "stacked": False}
        mod.pick_lane = pick

        def boom(*a, **k):
            raise AssertionError("dry-run must never spawn a worker")
        mod.spawn_detached = boom

        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertEqual(seen["busy"], {"docs", "gateway"})
        self.assertEqual(p["busy_lanes"], ["docs", "gateway"])
        self.assertEqual(p["busy_lane_sources"]["inflight_markers"], ["gateway"])
        self.assertEqual(p["busy_lane_sources"]["lease_refs"], ["docs"])
        self.assertIn("markers: gateway; lease refs: docs", mod.render(p))
        self.assertFalse(p["lane_stacked"])


class WaveBusySkipTest(unittest.TestCase):
    """A wave skips a lane a prior tick's worker still holds — the cross-tick
    de-confliction the within-tick arbiter does not provide."""

    def test_wave_skips_a_busy_lane(self) -> None:
        mod = load()
        mod.refresh_registry = lambda root: {"ok": True}
        mod.allocate_seats = lambda root, mw, wk: {
            "granted": 2, "requested": 2, "shortfall": 0,
            "wave_id": "w", "lanes": [_seat(0), _seat(1)]}
        cands = [{"lane": "tools", "issues": 9, "tree": ["tools/**"]},
                 {"lane": "docs", "issues": 7, "tree": ["docs/**"]}]
        mod.lane_candidates = lambda root: {"candidates": cands, "router_error": None}
        mod.preflight = lambda root, **kw: {"verdict": "SPAWN_OK", "cap": 10,
                                            "live": 0, "account": {}}
        mod.arbitrate_lane = _disjoint_arbitrate
        mod.busy_lanes = lambda runs_dir, **kw: {"tools"}   # tools already in flight
        mod.lease_ref_busy_lanes = lambda root: {"lanes": set()}

        def boom(*a, **k):
            raise AssertionError("dry-run must never spawn a wave worker")
        mod._spawn_wave_member = boom

        p = mod.evaluate_wave(ROOT, max_workers=2, work_kind="engineering", live=False)
        self.assertEqual(p["lanes"], ["docs"])         # tools skipped as busy
        self.assertEqual(p["skipped_busy"], ["tools"])
        self.assertEqual(p["busy_lanes"], ["tools"])
        self.assertEqual(p["size"], 1)

    def test_wave_skips_a_live_lease_ref_lane(self) -> None:
        mod = load()
        mod.refresh_registry = lambda root: {"ok": True}
        mod.allocate_seats = lambda root, mw, wk: {
            "granted": 2, "requested": 2, "shortfall": 0,
            "wave_id": "w", "lanes": [_seat(0), _seat(1)]}
        cands = [{"lane": "tools", "issues": 9, "tree": ["tools/**"]},
                 {"lane": "docs", "issues": 7, "tree": ["docs/**"]}]
        mod.lane_candidates = lambda root: {"candidates": cands, "router_error": None}
        mod.preflight = lambda root, **kw: {"verdict": "SPAWN_OK", "cap": 10,
                                            "live": 0, "account": {}}
        mod.arbitrate_lane = _disjoint_arbitrate
        mod.busy_lanes = lambda runs_dir, **kw: set()
        mod.lease_ref_busy_lanes = lambda root: {"lanes": {"tools"},
                                                 "records": [{"lane": "tools"}]}

        def boom(*a, **k):
            raise AssertionError("dry-run must never spawn a wave worker")
        mod._spawn_wave_member = boom

        p = mod.evaluate_wave(ROOT, max_workers=2, work_kind="engineering", live=False)
        self.assertEqual(p["lanes"], ["docs"])
        self.assertEqual(p["skipped_busy"], ["tools"])
        self.assertEqual(p["busy_lane_sources"]["lease_refs"], ["tools"])
        self.assertEqual(p["busy_lane_sources"]["inflight_markers"], [])
        self.assertIn("lease refs: tools", mod.render_wave(p))


class SpawnDetachedWorktreeTest(unittest.TestCase):
    """#3181: spawn_detached edits in a per-worker worktree when the flag is on,
    fails open to the shared-trunk cwd when off (byte-identical to today)."""

    def _fake_git(self):
        calls: list[list[str]] = []

        def git(root, args):
            calls.append(list(args))
            if args and args[0] == "rev-parse":
                return 0, "deadbeef\n"
            return 0, ""

        git.calls = calls  # type: ignore[attr-defined]
        return git

    def _spawn(self, env_overrides, td):
        m = load()
        seen: dict[str, object] = {}

        class FakeProc:
            pid = 4321

        def fake_popen(argv, cwd=None, env=None, **kw):
            seen["cwd"] = cwd
            seen["env"] = env
            return FakeProc()

        runs = Path(td) / "runs"
        with mock.patch.object(m.subprocess, "Popen", fake_popen), \
             mock.patch.dict(os.environ, env_overrides, clear=False):
            res = m.spawn_detached(
                ["python", "-c", "pass"], {"PATH": os.environ.get("PATH", "")},
                Path(td), runs, "tools",
                worktree_git=self._fake_git())
        return m, seen, res, runs

    def test_worktree_spawn_off_uses_root_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _, seen, res, runs = self._spawn({"FLEET_WORKER_WORKTREE": "0"}, td)
            self.assertEqual(seen["cwd"], str(Path(td)))
            self.assertNotIn("worktree", res)
            self.assertEqual(list(runs.glob("*.worktree")), [])  # off -> no sidecar

    def test_worktree_spawn_on_uses_worktree_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m, seen, res, runs = self._spawn(
                {"FLEET_WORKER_WORKTREE": "1",
                 "FLEET_WORKER_WORKTREE_ROOT": str(Path(td) / "wt")}, td)
            self.assertNotEqual(seen["cwd"], str(Path(td)))
            self.assertTrue(m.worker_worktree.is_worker_worktree(str(seen["cwd"])))
            self.assertEqual(res.get("worktree"), seen["cwd"])
            # Sidecar is a PLAIN path (the shared witness-sweep contract), not JSON.
            wt_side = list(runs.glob("*.worktree"))
            self.assertEqual(len(wt_side), 1)
            self.assertEqual(wt_side[0].read_text(encoding="utf-8"), seen["cwd"])


class DefaultMaxWorkersTest(unittest.TestCase):
    def test_darwin_defaults_to_30_without_changing_other_platforms(self) -> None:
        mod = load()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(mod._default_max_workers("darwin"), 30)
            self.assertEqual(mod._default_max_workers("linux"), 20)
            self.assertEqual(mod._default_max_workers("win32"), 20)

    def test_env_override_preserves_explicit_lower_ceiling(self) -> None:
        mod = load()
        with mock.patch.dict(os.environ, {"FAK_MAX_WORKERS": "7"}, clear=True):
            self.assertEqual(mod._default_max_workers("darwin"), 7)


class ArbitrateLaneLeasesTest(unittest.TestCase):
    def test_arbitrate_lane_passes_short_leases_inline(self) -> None:
        mod = load()
        seen = {}

        def fake_run_json(cmd, root, timeout=90):
            seen["cmd"] = list(cmd)
            return {"outcome": "acquire", "lane": "docs"}

        mod.run_json = fake_run_json
        short_leases = [{"lane": "docs", "lane_kind": "cluster", "tree": ["docs/**"]}]
        res = mod.arbitrate_lane(ROOT, "docs", ["docs/**"], short_leases)

        self.assertIn("--leases", seen["cmd"])
        idx = seen["cmd"].index("--leases")
        self.assertEqual(seen["cmd"][idx + 1], json.dumps(short_leases))
        self.assertTrue(res["admitted"])

    def test_arbitrate_lane_spills_large_leases_to_temp_file_and_cleans_up(self) -> None:
        mod = load()
        seen = {}
        file_existed_during_run = []
        file_content_during_run = []

        def fake_run_json(cmd, root, timeout=90):
            seen["cmd"] = list(cmd)
            idx = cmd.index("--leases")
            arg = cmd[idx + 1]
            if arg.startswith("@"):
                p = Path(arg[1:])
                file_existed_during_run.append(p.is_file())
                if p.is_file():
                    file_content_during_run.append(p.read_text(encoding="utf-8"))
            return {"outcome": "acquire", "lane": "tools"}

        mod.run_json = fake_run_json
        large_leases = [{"lane": f"lane_{i}", "lane_kind": "cluster", "tree": [f"path/to/resource_{i}/**"]} for i in range(100)]
        payload = json.dumps(large_leases)
        self.assertGreater(len(payload), 2048)

        res = mod.arbitrate_lane(ROOT, "tools", ["tools/**"], large_leases)

        self.assertIn("--leases", seen["cmd"])
        idx = seen["cmd"].index("--leases")
        arg = seen["cmd"][idx + 1]
        self.assertTrue(arg.startswith("@"))
        temp_path = Path(arg[1:])
        self.assertTrue(file_existed_during_run and file_existed_during_run[0])
        self.assertEqual(json.loads(file_content_during_run[0]), large_leases)
        self.assertFalse(temp_path.exists())
        self.assertTrue(res["admitted"])

    def test_arbitrate_lane_cleans_up_temp_file_on_exception(self) -> None:
        mod = load()
        captured_path = []

        def fake_run_json(cmd, root, timeout=90):
            idx = cmd.index("--leases")
            arg = cmd[idx + 1]
            if arg.startswith("@"):
                captured_path.append(Path(arg[1:]))
            raise RuntimeError("simulated error during arbitrate")

        mod.run_json = fake_run_json
        large_leases = [{"lane": f"lane_{i}", "lane_kind": "cluster", "tree": [f"path/to/resource_{i}/**"]} for i in range(100)]
        with self.assertRaises(RuntimeError):
            mod.arbitrate_lane(ROOT, "tools", ["tools/**"], large_leases)

        self.assertTrue(captured_path)
        self.assertFalse(captured_path[0].exists())


if __name__ == "__main__":
    unittest.main()
