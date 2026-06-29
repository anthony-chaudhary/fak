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
import sys
import tempfile
import unittest
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
        mod.pick_lane = lambda root, explicit: lane_pick

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
                 "live": 2, "account": {}},
            lane_pick={"lane": "gateway", "issues": 4, "by_lane": {}})
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertFalse(p["ok"])
        self.assertEqual(p["action"], "refused")
        self.assertEqual(p["verdict"], "REFUSE_AT_CAP")
        self.assertIn("preflight refused", p["reason"])
        self.assertIn("2/2 live", p["reason"])

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

    def test_preflight_payload_omits_host_cap_when_unbounded(self) -> None:
        # When no host dimension is readable host_cap is None; the render then
        # falls back to the static live/cap form with no host_cap clause.
        mod = load()
        self._no_spawn(mod)
        self._patch(mod, pre=self.SPAWN_OK,
                    lane_pick={"lane": "tools", "issues": 1, "by_lane": {}})
        p = mod.evaluate(ROOT, max_workers=2, work_kind="engineering",
                         lane=None, live=False)
        self.assertIsNone(p["preflight"]["host_cap"])
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
        mod.pick_lane = lambda root, explicit: {"lane": "docs", "issues": 1,
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
        mod.pick_lane = lambda root, explicit: {"lane": "docs", "issues": 1,
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
        # preflight was re-checked per spawn: 2 admits + 1 that hit the cap = 3 calls.
        self.assertEqual(calls["n"], 3)

    def test_preflight_refusal_stops_the_wave_with_zero_workers(self) -> None:
        mod = load()
        cands = [{"lane": "tools", "issues": 9, "tree": ["tools/**"]}]
        pre = {"verdict": "REFUSE_AT_CAP", "reason": "2/2 live", "cap": 2,
               "live": 2, "account": {}}
        self._wire(mod, seats=[_seat(0), _seat(1)], candidates=cands, pre=pre)
        p = mod.evaluate_wave(ROOT, max_workers=2, work_kind="engineering", live=False)
        self.assertFalse(p["ok"])
        self.assertEqual(p["size"], 0)
        self.assertEqual(p["verdict"], "REFUSE_AT_CAP")

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


if __name__ == "__main__":
    unittest.main()
