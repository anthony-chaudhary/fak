#!/usr/bin/env python3
"""Hermetic tests for tools/issue_resolve_progress.py.

The gh / closure-audit / close-arm shell-outs are stubbed on the module, so no
network or subprocess runs. The pure folds (witnessed_open, fold_closed_history,
the resolved/remaining arithmetic) are asserted directly against a tmp runs dir.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "issue_resolve_progress.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("issue_resolve_progress", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FoldsTest(unittest.TestCase):
    def test_witnessed_open_filters_bucket(self) -> None:
        mod = load()
        audit = {"issues": [
            {"number": 1, "bucket": "OPEN"},
            {"number": 2, "bucket": "OPEN_WITNESSED"},
            {"number": 3, "bucket": "OPEN_WITNESSED"},
            {"number": 4, "bucket": "CLOSED_WITNESSED"},
        ]}
        self.assertEqual(mod.witnessed_open(audit), [2, 3])

    def test_fold_closed_history_sums_closed_now(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = runs / mod.PROGRESS_LOG
            log.write_text("\n".join(json.dumps(r) for r in [
                {"closed_now": 2}, {"closed_now": 0}, {"closed_now": 3},
                {"no_field": 1},
            ]) + "\n", encoding="utf-8")
            self.assertEqual(mod.fold_closed_history(runs), 5)

    def test_fold_closed_history_empty(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(mod.fold_closed_history(Path(d)), 0)

    def test_baseline_recorded_once(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            first = mod.save_baseline(runs, 483)
            self.assertEqual(first["baseline_open"], 483)
            loaded = mod.load_baseline(runs)
            self.assertEqual(loaded["baseline_open"], 483)


class EvaluateTest(unittest.TestCase):
    def _stub(self, mod, *, open_now, witnessed, closed=0, history=0) -> None:
        mod.open_issue_count = lambda root: open_now
        mod.closure_audit = lambda root, *, max_commits: {
            "issues": [{"number": n, "bucket": "OPEN_WITNESSED"} for n in witnessed]}
        mod.fold_closed_history = lambda runs_dir: history
        mod.run_close = lambda root, *, live, audit_path, limit: {
            "verdict": "CLOSED", "closed": closed, "would_close": 0,
            "skipped": 0, "failed": 0}

    def test_snapshot_sets_baseline_and_zero_resolved(self) -> None:
        mod = load()
        self._stub(mod, open_now=483, witnessed=[491, 493])
        with tempfile.TemporaryDirectory() as d:
            p = mod.evaluate(Path(d), target=50, do_close=False, live=False,
                             max_commits=100)
        self.assertTrue(p["ok"])
        self.assertEqual(p["baseline_open"], 483)
        self.assertEqual(p["resolved_toward_target"], 0)   # baseline == open_now
        self.assertEqual(p["target_remaining"], 50)
        self.assertEqual(p["witnessed_open"], 2)

    def test_resolved_counts_drop_from_baseline(self) -> None:
        mod = load()
        self._stub(mod, open_now=479, witnessed=[])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # the baseline lives under root/.dispatch-runs, where evaluate reads it.
            mod.save_baseline(root / mod.RUNS_DIRNAME, 483)
            p = mod.evaluate(root, target=50, do_close=False, live=False,
                             max_commits=100)
        self.assertEqual(p["resolved_toward_target"], 4)  # 483 - 479
        self.assertEqual(p["target_remaining"], 46)

    def test_close_live_counts_this_tick(self) -> None:
        mod = load()
        self._stub(mod, open_now=481, witnessed=[491, 493], closed=2, history=0)
        with tempfile.TemporaryDirectory() as d:
            p = mod.evaluate(Path(d), target=50, do_close=True, live=True,
                             max_commits=100)
        self.assertEqual(p["closed_now"], 2)
        self.assertEqual(p["closed_by_loop_total"], 2)    # history 0 + 2
        self.assertEqual(p["close_result"]["verdict"], "CLOSED")

    def test_close_dry_run_does_not_count(self) -> None:
        mod = load()
        self._stub(mod, open_now=481, witnessed=[491], closed=0, history=1)
        with tempfile.TemporaryDirectory() as d:
            p = mod.evaluate(Path(d), target=50, do_close=True, live=False,
                             max_commits=100)
        self.assertEqual(p["closed_now"], 0)              # dry-run closes nothing
        self.assertEqual(p["closed_by_loop_total"], 1)    # history only

    def test_audit_error_does_not_fail_the_snapshot(self) -> None:
        # A closure-audit hiccup (dos momentarily unreachable) must NOT fail the
        # tick — the open-count is the proof metric; the curve must not gap.
        mod = load()
        mod.open_issue_count = lambda root: 479
        mod.closure_audit = lambda root, *, max_commits: {"_error": "dos not found"}
        mod.fold_closed_history = lambda runs_dir: 0
        with tempfile.TemporaryDirectory() as d:
            p = mod.evaluate(Path(d), target=50, do_close=True, live=True,
                             max_commits=100)
        self.assertTrue(p["ok"])                 # snapshot still OK
        self.assertEqual(p["witnessed_open"], 0)  # but no witnessed work this tick
        self.assertEqual(p["audit_error"], "dos not found")

    def test_no_open_count_fails(self) -> None:
        # Conversely, losing the open-count (gh down) IS a failed tick.
        mod = load()
        mod.open_issue_count = lambda root: None
        mod.closure_audit = lambda root, *, max_commits: {"issues": []}
        mod.fold_closed_history = lambda runs_dir: 0
        with tempfile.TemporaryDirectory() as d:
            p = mod.evaluate(Path(d), target=50, do_close=False, live=False,
                             max_commits=100)
        self.assertFalse(p["ok"])


if __name__ == "__main__":
    unittest.main()
