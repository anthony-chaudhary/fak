#!/usr/bin/env python3
"""Hermetic tests for tools/lane_yield.py — the shared low-yield lane fold (#2062).

The fold is pure given its two injectables (``closes_counter`` and ``turns_of_log``),
so these tests drive it with a synthetic ``.dispatch-runs`` corpus of ``resolve-*.log``
files and an injected closes counter — no git, no gh, no subprocess. They pin the
witness contract: a lane is ``LOW_YIELD`` only when its recent finished sessions burned
``>= turns_floor`` turns AND landed 0 ancestry-closes on a KNOWN tree; everything else
(below floor, tree unknown, or with closes) is ``OK``; the git join is never paid for a
below-floor lane; and the ``include_log`` predicate filters the turn accumulation so a
still-live worker's mid-flight turns cannot prematurely flag its lane.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPT = Path(__file__).resolve().parent / "lane_yield.py"


def load():
    spec = importlib.util.spec_from_file_location("lane_yield", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load()

TREES = {"tools": ["tools/**"], "docs": ["docs/**"], "model": ["internal/model/**"]}


def write_log(runs: Path, issue: int, lane: str, turns: int,
              *, stamp: str = "20260707-120000") -> Path:
    """A resolve-<issue>-<stamp>.log whose spawn header carries lane= and which
    records ``turns`` kernel-adjudicated turns as ``fak-turn`` trace lines (what the
    default ``_log_turn_count`` counts)."""
    log = runs / f"resolve-{issue}-{stamp}.log"
    lines = [f"spawn lane={lane} account=test backend=claude issue={issue}"]
    lines += [f"fak-turn {i} tool=Edit" for i in range(1, turns + 1)]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log


class SpawnLaneAndTurnCountTest(unittest.TestCase):
    """The two primitives the fold rolls up: the spawn-header lane and the fak-turn count."""

    def test_spawn_lane_reads_header_field(self):
        with TemporaryDirectory() as d:
            log = write_log(Path(d), 7, "tools", 3)
            self.assertEqual(m._spawn_lane(log), "tools")

    def test_spawn_lane_blank_when_no_lane_field(self):
        with TemporaryDirectory() as d:
            log = Path(d) / "resolve-7-20260707-120000.log"
            log.write_text("spawn account=test backend=claude\n", encoding="utf-8")
            self.assertEqual(m._spawn_lane(log), "")

    def test_log_turn_count_counts_fak_turn_lines(self):
        with TemporaryDirectory() as d:
            log = write_log(Path(d), 7, "tools", 5)
            self.assertEqual(m._log_turn_count(log), 5)

    def test_log_turn_count_unreadable_is_zero(self):
        self.assertEqual(m._log_turn_count(Path("does-not-exist.log")), 0)


class LowYieldFoldTest(unittest.TestCase):
    def test_flags_high_turn_zero_close_lane(self):
        with TemporaryDirectory() as d:
            runs = Path(d)
            write_log(runs, 100, "tools", 25, stamp="20260707-120000")
            write_log(runs, 101, "tools", 25, stamp="20260707-130000")   # 50 total
            write_log(runs, 200, "docs", 50, stamp="20260707-120000")    # 50, but closes
            write_log(runs, 300, "model", 10, stamp="20260707-120000")   # below floor

            called: list[str] = []

            def closes(lane, _tree):
                called.append(lane)
                return {"tools": 0, "docs": 3}.get(lane)

            fold = m.low_yield_lanes(runs, closes_counter=closes, lane_trees=TREES)
            rows = {r["lane"]: r for r in fold["lanes"]}

            self.assertEqual(fold["low_yield_count"], 1)
            self.assertEqual(rows["tools"]["verdict"], "LOW_YIELD")
            self.assertEqual(rows["tools"]["turns"], 50)
            self.assertEqual(rows["tools"]["sessions"], 2)
            self.assertTrue(rows["tools"]["tree_known"])
            self.assertEqual(rows["tools"]["closes"], 0)
            # a lane WITH closes is OK even at/above the turn floor
            self.assertEqual(rows["docs"]["verdict"], "OK")
            # a below-floor lane is OK and the git join is NEVER paid for it
            self.assertEqual(rows["model"]["verdict"], "OK")
            self.assertNotIn("model", called)
            self.assertIn("tools", called)
            self.assertIn("docs", called)

    def test_tree_unknown_lane_is_never_flagged(self):
        with TemporaryDirectory() as d:
            runs = Path(d)
            write_log(runs, 100, "ghost", 50, stamp="20260707-120000")
            write_log(runs, 101, "ghost", 50, stamp="20260707-130000")

            called: list[str] = []

            def closes(lane, _tree):
                called.append(lane)
                return 0

            fold = m.low_yield_lanes(runs, closes_counter=closes, lane_trees=TREES)
            rows = {r["lane"]: r for r in fold["lanes"]}
            self.assertFalse(rows["ghost"]["tree_known"])
            self.assertEqual(rows["ghost"]["verdict"], "OK")     # fail-open, no tree to join
            self.assertNotIn("ghost", called)                    # join skipped w/o a tree

    def test_include_log_drops_live_session_turns(self):
        # Two 25-turn tools sessions -> 50 total -> LOW_YIELD. Dropping one (as still-live)
        # leaves 25 -> below floor -> OK. Proves include_log filters turn accumulation.
        with TemporaryDirectory() as d:
            runs = Path(d)
            write_log(runs, 100, "tools", 25, stamp="20260707-120000")
            live = write_log(runs, 101, "tools", 25, stamp="20260707-130000")

            keep_all = m.low_yield_lanes(runs, closes_counter=lambda lane, t: 0,
                                         lane_trees=TREES)
            self.assertEqual({r["lane"]: r["verdict"] for r in keep_all["lanes"]},
                             {"tools": "LOW_YIELD"})

            finished_only = m.low_yield_lanes(
                runs, closes_counter=lambda lane, t: 0, lane_trees=TREES,
                include_log=lambda p: p.name != live.name)
            rows = {r["lane"]: r for r in finished_only["lanes"]}
            self.assertEqual(rows["tools"]["verdict"], "OK")
            self.assertEqual(rows["tools"]["sessions"], 1)
            self.assertEqual(rows["tools"]["turns"], 25)

    def test_turns_of_log_is_injectable(self):
        # The fold is pure given turns_of_log; a lane crosses the floor purely by the
        # injected counter, with no fak-turn lines written at all.
        with TemporaryDirectory() as d:
            runs = Path(d)
            log = write_log(runs, 100, "tools", 0, stamp="20260707-120000")
            fold = m.low_yield_lanes(
                runs, closes_counter=lambda lane, t: 0, lane_trees=TREES,
                turns_of_log=lambda p: 99 if p.name == log.name else 0,
                # one session with 99 turns still needs no min-session floor here
            )
            rows = {r["lane"]: r for r in fold["lanes"]}
            self.assertEqual(rows["tools"]["turns"], 99)
            self.assertEqual(rows["tools"]["verdict"], "LOW_YIELD")

    def test_missing_runs_dir_is_empty_fold(self):
        fold = m.low_yield_lanes(Path("no-such-runs-dir"),
                                 closes_counter=lambda lane, t: 0, lane_trees=TREES)
        self.assertEqual(fold["lanes"], [])
        self.assertEqual(fold["low_yield_count"], 0)
        self.assertEqual(fold["schema"], m._LOW_YIELD_SCHEMA)


class CountAncestryClosesTest(unittest.TestCase):
    """The git-join half — its pure guards and its resolving-commit grammar."""

    def test_empty_tree_returns_none(self):
        # No pathspecs -> the fold can't join -> None (fail-open), never 0.
        with TemporaryDirectory() as d:
            self.assertIsNone(
                m.count_lane_ancestry_closes(Path(d), [], since_iso="2026-07-07T00:00:00 +0000"))

    def test_resolving_commit_grammar(self):
        rx = m._LOW_YIELD_RESOLVE_RE
        for msg in ("fixes #123", "fix #7", "closed #3", "resolve #9", "Resolves #42"):
            self.assertTrue(rx.search(msg), msg)
        for msg in ("reference #9", "see #7", "part of #12"):
            self.assertIsNone(rx.search(msg), msg)


if __name__ == "__main__":
    unittest.main()
