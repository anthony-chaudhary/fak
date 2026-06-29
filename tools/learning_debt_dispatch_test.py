#!/usr/bin/env python3
"""Hermetic tests for tools/learning_debt_dispatch.py.

NOTHING live runs: the learning scorecard and `gh` are never invoked. The pure
logic — defect extraction from a scorecard JSON payload, the transparent
priority, the three dedup rungs, issue rendering, and the dedup→sort→CAP planner
— is exercised directly with a fixture payload, plus a real tmp-dir round-trip of
the seen-cache that PROVES the two acceptance witnesses from issue #1283:

    * the CAP — N HARD defects file at most --cap issues;
    * dedup — a SECOND run against the same corpus files nothing new.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "learning_debt_dispatch.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("learning_debt_dispatch", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = load()

# A learning-scorecard JSON payload carrying SIX HARD defects across all three
# buckets: 2 in-doc (a dead link + a stale install pin), 2 coverage (an orphan
# lesson + a missing course), 1 second in-doc defect on the same doc, and 1 stamp
# defect. Mirrors the real `--json` shape (docs[].defects, coverage.defects,
# stamp_freshness.stale_stamp).
PAYLOAD = {
    "schema": "fleet-learning-scorecard/1",
    "ok": False,
    "verdict": "ACTION",
    "docs": [
        {"path": "docs/explainers/loophealth.md", "score": 62.0, "type": "explainer",
         "defects": [
             "links: dead link to docs/loop-index-scorecard.md (target missing)",
             "freshness: stale install pin 0.30.0 (VERSION is 0.34.0)",
         ]},
        {"path": "docs/howto/run-the-gateway.md", "score": 71.5, "type": "howto",
         "defects": ["runnable: how-to has no runnable command block"]},
        {"path": "docs/explainers/clean.md", "score": 99.0, "type": "explainer",
         "defects": []},
    ],
    "coverage": {
        "overall_pct": 80.0,
        "defects": [
            "orphan lesson (unreachable from any front door): docs/explainers/sessionobs-net-true.md",
            "uncovered learning topic: P/D serving",
        ],
    },
    "stamp_freshness": {"stale_stamp": True, "path": "docs/LEARNING-SCORECARD.md",
                        "flag": "scorecard stamp 5 days stale"},
}


class ExtractTest(unittest.TestCase):
    def test_extracts_all_three_buckets(self) -> None:
        defects = M.extract_defects(PAYLOAD)
        # 3 in-doc + 2 coverage + 1 stamp = 6
        self.assertEqual(len(defects), 6)
        classes = sorted(d["defect_class"] for d in defects)
        self.assertEqual(classes,
                         ["freshness", "links", "missing-course", "orphan-lesson",
                          "runnable", "stale-stamp"])

    def test_in_doc_defect_cites_doc_and_verbatim_detail(self) -> None:
        defects = M.extract_defects(PAYLOAD)
        pin = next(d for d in defects if d["defect_class"] == "freshness")
        self.assertEqual(pin["doc"], "docs/explainers/loophealth.md")
        # the verbatim scorecard string is preserved (no prose drift)
        self.assertEqual(pin["detail"],
                         "freshness: stale install pin 0.30.0 (VERSION is 0.34.0)")

    def test_coverage_orphan_and_missing_course_parsed(self) -> None:
        defects = M.extract_defects(PAYLOAD)
        orphan = next(d for d in defects if d["defect_class"] == "orphan-lesson")
        self.assertEqual(orphan["doc"], "docs/explainers/sessionobs-net-true.md")
        course = next(d for d in defects if d["defect_class"] == "missing-course")
        self.assertEqual(course["doc"], "P/D serving")

    def test_clean_doc_contributes_no_defect(self) -> None:
        defects = M.extract_defects(PAYLOAD)
        self.assertFalse(any(d["doc"] == "docs/explainers/clean.md" for d in defects))

    def test_no_defects_when_clean_payload(self) -> None:
        clean = {"docs": [{"path": "x.md", "score": 100.0, "defects": []}],
                 "coverage": {"defects": []}, "stamp_freshness": {"stale_stamp": False}}
        self.assertEqual(M.extract_defects(clean), [])

    def test_source_id_binds_doc_class_and_fingerprint(self) -> None:
        # same doc + class but different detail → distinct source_ids (two pins
        # in one doc are two issues); identical detail → identical id (re-detect).
        a = M.make_defect(bucket="doc", doc="d.md", defect_class="links",
                          detail="links: dead link to A", score=50.0)
        b = M.make_defect(bucket="doc", doc="d.md", defect_class="links",
                          detail="links: dead link to B", score=50.0)
        c = M.make_defect(bucket="doc", doc="d.md", defect_class="links",
                          detail="links: dead link to A", score=50.0)
        self.assertNotEqual(a["source_id"], b["source_id"])
        self.assertEqual(a["source_id"], c["source_id"])


class PriorityTest(unittest.TestCase):
    def test_structural_outranks_in_doc(self) -> None:
        orphan = M.make_defect(bucket="coverage", doc="x", defect_class="orphan-lesson",
                               detail="orphan", score=None)
        runnable = M.make_defect(bucket="doc", doc="y", defect_class="runnable",
                                 detail="runnable: none", score=10.0)
        self.assertGreater(M.priority(orphan), M.priority(runnable))

    def test_more_broken_doc_wins_within_class(self) -> None:
        worse = M.make_defect(bucket="doc", doc="a", defect_class="links",
                              detail="links: x", score=40.0)
        better = M.make_defect(bucket="doc", doc="b", defect_class="links",
                               detail="links: y", score=90.0)
        self.assertGreater(M.priority(worse), M.priority(better))


class DedupTest(unittest.TestCase):
    def _defect(self):
        return M.make_defect(bucket="doc", doc="docs/x.md", defect_class="links",
                             detail="links: dead link to y", score=50.0)

    def test_seen_cache_rung(self) -> None:
        d = self._defect()
        seen = {d["source_id"]: {"filed_at": "2026-01-01"}}
        self.assertEqual(M.is_duplicate(d, seen, set(), [], 0.6), "seen-cache")

    def test_issue_body_stamp_rung(self) -> None:
        d = self._defect()
        issues = [{"number": 1, "title": "old",
                   "body": f"x\n<!-- learning-debt-source: {d['source_id']} -->"}]
        stamped, tsets = M.existing_issue_index(issues)
        self.assertEqual(M.is_duplicate(d, {}, stamped, tsets, 0.6), "issue-body")

    def test_title_near_rung(self) -> None:
        d = self._defect()
        # a human already opened an issue with a near-identical title
        issues = [{"number": 2, "title": M.issue_title(d) + " please fix",
                   "body": "no stamp"}]
        stamped, tsets = M.existing_issue_index(issues)
        self.assertEqual(M.is_duplicate(d, {}, stamped, tsets, 0.6), "title-near")

    def test_genuinely_new_passes(self) -> None:
        d = self._defect()
        self.assertIsNone(M.is_duplicate(d, {}, set(), [], 0.6))


class RenderTest(unittest.TestCase):
    def test_render_cites_doc_class_detail_and_stamps_source(self) -> None:
        d = M.make_defect(bucket="doc", doc="docs/explainers/loophealth.md",
                          defect_class="freshness",
                          detail="freshness: stale install pin 0.30.0 (VERSION is 0.34.0)",
                          score=62.0)
        issue = M.render_issue(d, "2026-06-29")
        self.assertTrue(issue["title"].startswith("learning-debt: "))
        # exact doc cited
        self.assertIn("docs/explainers/loophealth.md", issue["body"])
        # exact defect class cited
        self.assertIn("`freshness`", issue["body"])
        # verbatim scorecard detail (no prose drift)
        self.assertIn("stale install pin 0.30.0 (VERSION is 0.34.0)", issue["body"])
        # the load-bearing dedup anchor
        self.assertIn(f"<!-- learning-debt-source: {d['source_id']} -->", issue["body"])
        self.assertEqual(issue["labels"], ["learning-debt", "rsi"])


class PlanTest(unittest.TestCase):
    def test_cap_and_priority_sort(self) -> None:
        defects = M.extract_defects(PAYLOAD)  # 6 defects
        cfg = dict(M.DEFAULTS, cap=3)
        to_file, stats = M.plan_issues(defects, {}, set(), [], cfg, "2026-06-29")
        self.assertEqual(len(to_file), 3)                  # capped at 3 of 6
        # highest priority first; the orphan-lesson (weight 50) leads
        self.assertEqual(to_file[0]["defect_class"], "orphan-lesson")
        for a, b in zip(to_file, to_file[1:]):
            self.assertGreaterEqual(a["priority"], b["priority"])

    def test_within_run_dedup(self) -> None:
        d = M.make_defect(bucket="doc", doc="x.md", defect_class="links",
                          detail="links: dead link", score=50.0)
        cfg = dict(M.DEFAULTS)
        to_file, stats = M.plan_issues([d, dict(d)], {}, set(), [], cfg, "2026-06-29")
        self.assertEqual(len(to_file), 1)
        self.assertEqual(stats["within-run-dup"], 1)

    def test_seen_cache_skips_in_plan(self) -> None:
        d = M.make_defect(bucket="doc", doc="x.md", defect_class="links",
                          detail="links: dead link", score=50.0)
        cfg = dict(M.DEFAULTS)
        to_file, stats = M.plan_issues([d], {d["source_id"]: {}}, set(), [], cfg,
                                       "2026-06-29")
        self.assertEqual(to_file, [])
        self.assertEqual(stats["seen-cache"], 1)


class CacheTest(unittest.TestCase):
    def test_seen_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            ws = Path(d)
            self.assertEqual(M.load_seen(ws), {})
            M.save_seen(ws, {"learning-debt:x": {"filed_at": "2026-06-29"}})
            self.assertEqual(M.load_seen(ws),
                             {"learning-debt:x": {"filed_at": "2026-06-29"}})
            self.assertTrue(M.cache_path(ws).exists())


class MainHermeticTest(unittest.TestCase):
    """Drive main() end to end with the scorecard + gh boundaries stubbed, to lock
    the load-bearing safety contracts from #1283: dry-run mutates nothing; --live
    files + caches; the cap is never exceeded; and a SECOND run files nothing
    new (the dedup witness)."""

    def setUp(self) -> None:
        self._orig = (M.fetch_existing_issues, M.create_issue, M.ensure_debt_label)
        self._fixture = self._write_fixture()

    def tearDown(self) -> None:
        (M.fetch_existing_issues, M.create_issue, M.ensure_debt_label) = self._orig

    def _write_fixture(self) -> str:
        fd = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8")
        json.dump(PAYLOAD, fd)
        fd.close()
        return fd.name

    def _stub(self, *, existing=None):
        M.fetch_existing_issues = lambda *a, **k: list(existing or [])
        M.ensure_debt_label = lambda: None
        calls: list = []
        counter = {"n": 0}

        def _create(issue):
            calls.append(issue)
            counter["n"] += 1
            return f"https://github.com/x/y/issues/{counter['n']}"
        M.create_issue = _create
        return calls

    def _run(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = M.main(argv)
        return rc, buf.getvalue()

    def test_dry_run_writes_no_cache_and_files_nothing(self) -> None:
        calls = self._stub()
        with tempfile.TemporaryDirectory() as d:
            rc, out = self._run(["--workspace", d, "--scorecard", self._fixture])
            self.assertEqual(rc, 0)
            self.assertEqual(calls, [])                        # nothing filed
            self.assertFalse(M.cache_path(Path(d)).exists())   # cache untouched
            self.assertIn("dry-run", out)
            self.assertIn("would file", out)

    def test_live_respects_cap(self) -> None:
        # 6 defects in the fixture, cap 2 → exactly 2 filed
        calls = self._stub()
        with tempfile.TemporaryDirectory() as d:
            rc, _ = self._run(["--workspace", d, "--scorecard", self._fixture,
                               "--live", "--cap", "2"])
            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 2)                    # cap holds over 6

    def test_live_files_caps_then_second_run_files_nothing(self) -> None:
        """The #1283 acceptance witness: cap + dedup across two runs."""
        calls = self._stub()
        with tempfile.TemporaryDirectory() as d:
            # run 1: 6 defects, cap 4 → exactly 4 filed + recorded
            rc, _ = self._run(["--workspace", d, "--scorecard", self._fixture,
                               "--live", "--cap", "4"])
            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 4)
            seen = M.load_seen(Path(d))
            self.assertEqual(len(seen), 4)                     # cache recorded them

            # run 2: SAME corpus, same cap → the 4 filed are seen-cache dups, and
            # the remaining 2 are now the only new work (still ≤ cap).
            calls.clear()
            rc, out = self._run(["--workspace", d, "--scorecard", self._fixture,
                                 "--live", "--cap", "4"])
            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 2)                    # only the 2 not-yet-filed
            self.assertEqual(len(M.load_seen(Path(d))), 6)

            # run 3: everything is now tracked → files NOTHING new (the dedup proof)
            calls.clear()
            rc, out = self._run(["--workspace", d, "--scorecard", self._fixture,
                                 "--live", "--cap", "4"])
            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 0)
            self.assertIn("nothing new", out)

    def test_refuse_when_issue_fetch_fails_and_no_cache(self) -> None:
        self._stub()

        def _boom(*a, **k):
            raise RuntimeError("gh not authed")
        M.fetch_existing_issues = _boom
        with tempfile.TemporaryDirectory() as d:
            rc, _ = self._run(["--workspace", d, "--scorecard", self._fixture])
            self.assertEqual(rc, 2)        # refuse rather than risk a blind run

    def test_scorecard_load_error_exits_2(self) -> None:
        self._stub()
        with tempfile.TemporaryDirectory() as d:
            rc, _ = self._run(["--workspace", d, "--scorecard",
                               str(Path(d) / "does-not-exist.json")])
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
