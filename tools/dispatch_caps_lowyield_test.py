#!/usr/bin/env python3
"""Hermetic tests for the Part-B capability gate and Part-C low-yield soft-exclude
seams added to the dispatcher (#2062).

Nothing live runs: ``issue_dispatch.run_json`` (the lane-router shell-out) is stubbed
with a synthetic router payload, and the low-yield git join / pid-liveness probe are
monkeypatched on the module. These tests pin the pure decision seams:

  * ``dispatch_worker.node_caps`` — FLEET_NODE_CAPS parsing, default-empty ⇒ GPU-less;
  * ``lane_issue_numbers`` — the SOFT low-yield demote with its STARVATION FLOOR, the
    explicit-lane bypass, and the ``caps_by_issue`` map fed from the router's flat list;
  * ``low_yield_soft_excludes`` — the finished-sessions-only fold, the min-sessions trust
    floor, the worst-by-turns cap, and fail-open.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "issue_resolve_dispatch.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("issue_resolve_dispatch", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ird = load()


def write_log(runs: Path, issue: int, lane: str, turns: int,
              *, stamp: str = "20260707-120000") -> Path:
    log = runs / f"resolve-{issue}-{stamp}.log"
    lines = [f"spawn lane={lane} account=test backend=claude issue={issue}"]
    lines += [f"fak-turn {i} tool=Edit" for i in range(1, turns + 1)]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log


class NodeCapsTest(unittest.TestCase):
    """Part B: a host declares its hardware via FLEET_NODE_CAPS; default-empty is the
    GPU-less floor that makes THIS host stop grabbing GPU work."""

    def caps(self, val):
        env = {} if val is None else {"FLEET_NODE_CAPS": val}
        return ird.dispatch_worker.node_caps(env)

    def test_default_empty_is_gpu_less(self):
        self.assertEqual(self.caps(None), frozenset())
        self.assertEqual(self.caps(""), frozenset())
        self.assertEqual(self.caps("   "), frozenset())

    def test_parses_and_casefolds(self):
        self.assertEqual(self.caps("gpu"), frozenset({"gpu"}))
        self.assertEqual(self.caps("GPU"), frozenset({"gpu"}))

    def test_comma_and_space_separated(self):
        self.assertEqual(self.caps("gpu, fp8"), frozenset({"gpu", "fp8"}))
        self.assertEqual(self.caps("gpu  cuda"), frozenset({"gpu", "cuda"}))
        self.assertEqual(self.caps("gpu,cuda fp8"), frozenset({"gpu", "cuda", "fp8"}))


class _RouterStub:
    """Patch ird.issue_dispatch.run_json with a fixed router payload for the duration
    of the `with` block (restored on exit)."""

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        self._orig = ird.issue_dispatch.run_json
        ird.issue_dispatch.run_json = lambda *a, **k: self.payload
        return self

    def __exit__(self, *exc):
        ird.issue_dispatch.run_json = self._orig
        return False


ROUTER = {
    "lanes": {
        "tools": {"issues": [{"number": 10}, {"number": 11},
                             {"number": 12}, {"number": 13}], "tree": ["tools/**"]},
        "docs": {"issues": [{"number": 20}, {"number": 21}, {"number": 22}],
                 "tree": ["docs/**"]},
        "compute": {"issues": [{"number": 30}], "tree": ["internal/compute/**"]},
    },
    "issues": [
        {"number": 30, "required_caps": ["gpu"]},
        {"number": 10, "required_caps": []},
        {"number": 20, "required_caps": []},
    ],
}


class LaneIssueNumbersSoftExcludeTest(unittest.TestCase):
    """Part C: the low-yield SOFT demote inside the picker. guarded=False keeps the
    self-source (internal/**) hard-exclude out of the way so the soft path is isolated."""

    def test_soft_exclude_demotes_to_next_busiest(self):
        with _RouterStub(ROUTER):
            pick = ird.lane_issue_numbers(ROOT, None, soft_exclude={"tools"},
                                          guarded=False)
        self.assertEqual(pick["lane"], "docs")            # busiest survivor
        self.assertEqual(pick["low_yield_excluded"], ["tools"])
        self.assertEqual(pick["low_yield_relief"], [])

    def test_starvation_floor_reseats_the_only_lane(self):
        single = {"lanes": {"tools": {"issues": [{"number": 10}, {"number": 11}],
                                      "tree": ["tools/**"]}},
                  "issues": [{"number": 10, "required_caps": []}]}
        with _RouterStub(single):
            pick = ird.lane_issue_numbers(ROOT, None, soft_exclude={"tools"},
                                          guarded=False)
        # excluding the only eligible lane would starve the picker -> seat it back.
        self.assertEqual(pick["lane"], "tools")
        self.assertEqual(pick["low_yield_relief"], ["tools"])
        self.assertEqual(pick["low_yield_excluded"], [])

    def test_explicit_lane_bypasses_soft_exclude(self):
        with _RouterStub(ROUTER):
            pick = ird.lane_issue_numbers(ROOT, "tools", soft_exclude={"tools"},
                                          guarded=False)
        # an operator pin overrides the soft demote (deliberate override).
        self.assertEqual(pick["lane"], "tools")
        self.assertEqual(pick["low_yield_excluded"], [])

    def test_no_soft_exclude_picks_busiest(self):
        with _RouterStub(ROUTER):
            pick = ird.lane_issue_numbers(ROOT, None, guarded=False)
        self.assertEqual(pick["lane"], "tools")
        self.assertEqual(pick["low_yield_excluded"], [])
        self.assertEqual(pick["low_yield_relief"], [])

    def test_caps_by_issue_from_flat_router_list(self):
        with _RouterStub(ROUTER):
            pick = ird.lane_issue_numbers(ROOT, None, guarded=False)
        self.assertEqual(pick["caps_by_issue"][30], ["gpu"])
        self.assertEqual(pick["caps_by_issue"][10], [])


class _LowYieldPatches:
    """Patch the low-yield git join + pid-liveness probe on the module for a block."""

    def __init__(self, *, closes, is_live=lambda _pid: False):
        self.closes = closes
        self.is_live = is_live

    def __enter__(self):
        self._oc = ird.lane_yield.count_lane_ancestry_closes
        self._ol = ird.dispatch_preflight.resolve_sidecar_pid_is_live
        ird.lane_yield.count_lane_ancestry_closes = \
            lambda root, tree, *, since_iso: self.closes
        ird.dispatch_preflight.resolve_sidecar_pid_is_live = self.is_live
        return self

    def __exit__(self, *exc):
        ird.lane_yield.count_lane_ancestry_closes = self._oc
        ird.dispatch_preflight.resolve_sidecar_pid_is_live = self._ol
        return False


class LowYieldSoftExcludesTest(unittest.TestCase):
    """Part C: the dispatcher-side helper that folds the finished resolve corpus into
    the soft-exclude set."""

    def test_flags_finished_high_turn_zero_close_lane(self):
        with TemporaryDirectory() as d:
            runs = Path(d)
            write_log(runs, 100, "tools", 25, stamp="20260707-120000")
            write_log(runs, 101, "tools", 25, stamp="20260707-130000")   # 50t / 2 sessions
            with _LowYieldPatches(closes=0):
                res = ird.low_yield_soft_excludes(
                    ROOT, runs, lane_trees={"tools": ["tools/**"]})
        self.assertEqual(res["exclude"], {"tools"})
        self.assertEqual(res["flagged"], ["tools"])
        row = res["lanes"][0]
        self.assertEqual(row["lane"], "tools")
        self.assertEqual(row["turns"], 50)
        self.assertEqual(row["sessions"], 2)
        self.assertEqual(row["closes"], 0)

    def test_min_sessions_trust_floor_holds_a_single_session(self):
        with TemporaryDirectory() as d:
            runs = Path(d)
            write_log(runs, 100, "tools", 50, stamp="20260707-120000")   # 1 session only
            with _LowYieldPatches(closes=0):
                res = ird.low_yield_soft_excludes(
                    ROOT, runs, lane_trees={"tools": ["tools/**"]})
        # flagged by the raw fold, but < min_sessions(2) -> NOT soft-excluded.
        self.assertEqual(res["exclude"], set())

    def test_live_session_is_not_counted(self):
        with TemporaryDirectory() as d:
            runs = Path(d)
            write_log(runs, 100, "tools", 25, stamp="20260707-120000")
            write_log(runs, 101, "tools", 25, stamp="20260707-130000")

            def is_live(pid_path: Path) -> bool:
                # session 101 reads as still-live -> its 25 turns are dropped, leaving
                # 25t / 1 session -> below floor -> not flagged.
                return "101" in pid_path.name

            with _LowYieldPatches(closes=0, is_live=is_live):
                res = ird.low_yield_soft_excludes(
                    ROOT, runs, lane_trees={"tools": ["tools/**"]})
        self.assertEqual(res["exclude"], set())

    def test_worst_by_turns_capped_at_max_excludes(self):
        with TemporaryDirectory() as d:
            runs = Path(d)
            # 7 flagged low-yield lanes (each 2 finished sessions, 0 closes). The cap
            # keeps only the worst-by-turns LOW_YIELD_MAX_EXCLUDES of them, leaving the
            # lightest still pickable so even a raised cap can never demote every lane.
            specs = [("aa", 50), ("bb", 45), ("cc", 40), ("dd", 35),
                     ("ee", 30), ("ff", 25), ("gg", 20)]   # per-session turns, desc
            trees = {}
            for i, (lane, per) in enumerate(specs):
                write_log(runs, 200 + i * 2, lane, per, stamp="20260707-120000")
                write_log(runs, 201 + i * 2, lane, per, stamp="20260707-130000")
                trees[lane] = [f"internal/{lane}/**"]
            with _LowYieldPatches(closes=0):
                res = ird.low_yield_soft_excludes(ROOT, runs, lane_trees=trees)
        # all seven flagged; only the worst-by-turns cap(5) are soft-excluded.
        self.assertEqual(set(res["flagged"]),
                         {"aa", "bb", "cc", "dd", "ee", "ff", "gg"})
        self.assertEqual(len(res["exclude"]), ird.LOW_YIELD_MAX_EXCLUDES)
        self.assertEqual(res["exclude"], {"aa", "bb", "cc", "dd", "ee"})

    def test_fail_open_on_missing_runs_dir(self):
        res = ird.low_yield_soft_excludes(ROOT, ROOT / "no-such-runs",
                                          lane_trees={"tools": ["tools/**"]})
        self.assertEqual(res["exclude"], set())
        self.assertEqual(res["lanes"], [])
        self.assertEqual(res["flagged"], [])


if __name__ == "__main__":
    unittest.main()
