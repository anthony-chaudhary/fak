#!/usr/bin/env python3
"""Tests for tools/rsi_maturity_scorecard.py — the RSI maturity/usefulness stick.

Pure-stdlib unittest, repo root like the other scorecard tests. Two kinds of
check: (1) the scoring math + control-pane contract are exercised over SYNTHETIC
evidence so the asserts don't drift as the real engine evolves; (2) a small set
of invariants is asserted over the REAL repo (the engine is structurally mature,
the debt equals the count of failing hard criteria, the payload folds).
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import rsi_maturity_scorecard as sc

ROOT = sc.repo_root()


class GradeLetterTest(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(sc.grade_letter(100), "A")
        self.assertEqual(sc.grade_letter(90), "A")
        self.assertEqual(sc.grade_letter(89), "B")
        self.assertEqual(sc.grade_letter(80), "B")
        self.assertEqual(sc.grade_letter(70), "C")
        self.assertEqual(sc.grade_letter(60), "D")
        self.assertEqual(sc.grade_letter(59), "F")
        self.assertEqual(sc.grade_letter(0), "F")


class AxisScoreTest(unittest.TestCase):
    def test_weighted_fraction(self):
        results = [
            {"weight": 3, "passed": True},
            {"weight": 1, "passed": False},
        ]
        # 3 of 4 weight -> 75
        self.assertEqual(sc.axis_score(results), 75)

    def test_all_pass_is_100(self):
        self.assertEqual(sc.axis_score([{"weight": 2, "passed": True}]), 100)

    def test_empty_does_not_divide_by_zero(self):
        self.assertEqual(sc.axis_score([]), 0)


class LevelLadderTest(unittest.TestCase):
    def test_maturity_ladder_needs_lower_rungs(self):
        # multi_harness alone, without the closed loop, must NOT grant level 4.
        self.assertEqual(sc.maturity_level({"multi_harness_pattern": True}), 0)
        full = {
            "closed_loop_engine": True, "nonforgeable_keepbit": True,
            "vs_latest_main": True, "escalation_breaker": True,
            "worktree_isolation": True, "multi_harness_pattern": True,
        }
        self.assertEqual(sc.maturity_level(full), 4)

    def test_usefulness_ladder_monotone(self):
        self.assertEqual(sc.usefulness_level({}), 0)
        self.assertEqual(sc.usefulness_level({"kept_gains_proven": True}), 1)
        top = {
            "kept_gains_proven": True, "real_demo_gain": True,
            "journal_capability": True, "multi_real_subsystem": True,
            "ratchet_enforced_in_ci": True, "regression_gate_in_ci": True,
        }
        self.assertEqual(sc.usefulness_level(top), 5)


class SyntheticEvidenceTest(unittest.TestCase):
    """Drive build_payload over a fake repo so the asserts are stable."""

    def _write(self, root: Path, rel: str, text: str) -> None:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def _mature_repo(self, root: Path, *, second_subsystem=False, ci_ratchet=False):
        self._write(root, "internal/rsiloop/rsiloop.go",
                    "type Harness interface { BaselineMetric() float64; "
                    "Candidates() []int; Measure(c int) M }\nBaselineRefName string\n")
        self._write(root, "internal/shipgate/shipgate.go",
                    "improvedBit bool\nfunc Evaluate(w Witness){}\nfunc (w Witness) "
                    "Kept() bool { return w.improvedBit }\nESCALATE\ntype Gate struct"
                    "{ nonKeeps int }\nfunc ApplyInWorktree(){}\n")
        self._write(root, "internal/rsiloop/kpi.go",
                    "// monotone by construction\nfunc HitRate() {}\n")
        self._write(root, "internal/rsiloop/worktree.go",
                    "func NewWorktreeHarness(){ /* worktree add */ }\nrev-parse\n")
        self._write(root, "internal/rsiloop/rulesynth.go", "func NewRuleSynthHarness(){}\n")
        cmd = "mode := \"track\"\njournal := flag\nbaseline-ref\nNewWorktreeHarness()\n"
        if second_subsystem:
            cmd += "NewRuleSynthHarness()\n"
        self._write(root, "cmd/rsiloop/main.go", cmd)
        self._write(root, "cmd/kpiprobe/main.go", "package main\n")
        self._write(root, "internal/rsiloop/rsiloop_test.go", "Keep\nRevert\n")
        self._write(root, "docs/proofs/shipgate.md", "# proof\n")
        self._write(root, "docs/rsi-loop.md", "# rsi\n")
        wf = "run: go run ./cmd/kpiprobe\n"
        if ci_ratchet:
            wf = ("run: python tools/scorecard_control_pane.py --check\n"
                  "run: go run ./cmd/rsiloop -mode track\n")
        self._write(root, ".github/workflows/ci.yml", "steps:\n  - " + wf)

    def test_debt_equals_failing_hard_criteria(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._mature_repo(root)
            p = sc.build_payload(root)
            hard_fail = [r for r in (p["maturity"] + p["usefulness"])
                         if r["hard"] and not r["passed"]]
            self.assertEqual(p["corpus"]["rsi_debt"], len(hard_fail))
            # demo repo: 1 subsystem, no CI ratchet, no regression gate -> 3 gaps
            self.assertEqual(p["corpus"]["rsi_debt"], 3)
            self.assertEqual(p["corpus"]["maturity_score"], 100)

    def test_closing_gaps_drops_debt_to_zero(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._mature_repo(root, second_subsystem=True, ci_ratchet=True)
            p = sc.build_payload(root)
            self.assertEqual(p["corpus"]["rsi_debt"], 0)
            self.assertTrue(p["ok"])
            self.assertEqual(p["verdict"], "OK")
            self.assertEqual(p["corpus"]["usefulness_score"], 100)


class RealRepoTest(unittest.TestCase):
    def setUp(self):
        self.p = sc.build_payload(ROOT)

    def test_control_pane_contract_keys(self):
        for k in ("schema", "ok", "verdict", "finding", "reason", "next_action",
                  "workspace", "corpus"):
            self.assertIn(k, self.p)
        c = self.p["corpus"]
        self.assertIsInstance(c["rsi_debt"], int)
        self.assertNotIsInstance(c["rsi_debt"], bool)
        for k in ("maturity_score", "usefulness_score"):
            self.assertTrue(0 <= c[k] <= 100)

    def test_folds_into_control_pane(self):
        """scorecard_control_pane.find_int must locate rsi_debt under corpus."""
        try:
            import scorecard_control_pane as cp
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"control pane import failed: {exc}")
        self.assertEqual(cp.find_int(self.p, "rsi_debt"), self.p["corpus"]["rsi_debt"])

    def test_engine_is_structurally_mature(self):
        # The real fak engine: the foundational maturity criteria hold today.
        m = {r["key"]: r["passed"] for r in self.p["maturity"]}
        self.assertTrue(m["nonforgeable_keepbit"])
        self.assertTrue(m["deterministic_metric"])
        self.assertTrue(m["closed_loop_engine"])
        self.assertGreaterEqual(self.p["corpus"]["maturity_score"], 80)

    def test_deterministic_repeat(self):
        # Static build (no probe) reproduces exactly.
        a = sc.build_payload(ROOT)
        b = sc.build_payload(ROOT)
        a.pop("probe", None)
        b.pop("probe", None)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_json_and_markdown_render(self):
        self.assertIn("RSI maturity", sc.render(self.p))
        self.assertTrue(sc.markdown(self.p).startswith("---"))


if __name__ == "__main__":
    unittest.main()
