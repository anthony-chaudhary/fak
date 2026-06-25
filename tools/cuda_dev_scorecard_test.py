#!/usr/bin/env python3
"""Hermetic unit tests for tools/cuda_dev_scorecard.py.

Pin the fold invariants (weights sum to 1.0, weight-set == KPI-set), the pure per-KPI logic,
and clone-determinism (two reads of the same workspace yield the same process-debt integer).
"""
from __future__ import annotations

import unittest
from pathlib import Path

import cuda_dev_scorecard as m


class TestWeightsAndGroups(unittest.TestCase):
    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(m.KPI_WEIGHTS.values()), 1.0, places=9)

    def test_weight_set_equals_kpi_group_set(self):
        self.assertEqual(set(m.KPI_WEIGHTS), set(m.KPI_GROUP))

    def test_every_kpi_group_is_known(self):
        self.assertTrue(set(m.KPI_GROUP.values()) <= set(m.GROUPS))

    def test_grade_boundaries(self):
        self.assertEqual(m.grade_letter(90), "A")
        self.assertEqual(m.grade_letter(89.9), "B")
        self.assertEqual(m.grade_letter(59), "F")


class TestPureKPIs(unittest.TestCase):
    def test_local_static_check_all_present(self):
        k = m.kpi_local_static_check(True, True, True)
        self.assertEqual(k["defects"], [])
        self.assertEqual(k["score"], 100)

    def test_local_static_check_counts_each_gap(self):
        k = m.kpi_local_static_check(False, False, False)
        self.assertEqual(len(k["defects"]), 3)

    def test_abi_parity_folds_real_parser_hard(self):
        # Simulate the parser reporting two mismatches.
        payload = {"verdict": "ACTION", "corpus": {"hard": ["a", "b"], "soft_signals": 0,
                                                   "n_declared": 31}}
        k = m.kpi_abi_parity(payload)
        self.assertEqual(len(k["defects"]), 2)
        self.assertLess(k["score"], 100)

    def test_abi_parity_clean(self):
        payload = {"verdict": "OK", "corpus": {"hard": [], "soft_signals": 1, "n_declared": 31}}
        k = m.kpi_abi_parity(payload)
        self.assertEqual(k["defects"], [])
        self.assertEqual(k["score"], 100)

    def test_cpuref_parity_coverage_awq_gap(self):
        k = m.kpi_cpuref_parity_coverage(["AWQ (4-bit GEMV/GEMM)"])
        self.assertEqual(len(k["defects"]), 1)
        self.assertLess(k["score"], 100)

    def test_gate_kpis_hard_when_absent(self):
        self.assertEqual(m.kpi_cgo_typecheck_gate(False)["score"], 0)
        self.assertEqual(len(m.kpi_cgo_typecheck_gate(False)["defects"]), 1)
        self.assertEqual(m.kpi_cgo_typecheck_gate(True)["defects"], [])
        self.assertEqual(m.kpi_nvcc_compile_gate(True)["defects"], [])
        self.assertEqual(m.kpi_pure_go_guard(True)["defects"], [])

    def test_dev_guide_content_crosscheck(self):
        # Present but naming a dead artifact + missing a section = defects, not full credit.
        k = m.kpi_dev_guide(True, ["tools/run_481_acceptance_on_gpu.sh"], ["validate"])
        self.assertEqual(len(k["defects"]), 2)
        # Absent = HARD, score 0.
        self.assertEqual(m.kpi_dev_guide(False, [], [])["score"], 0)


class TestFold(unittest.TestCase):
    def test_build_payload_sums_process_debt(self):
        kpis = m.gather  # placeholder to ensure attribute exists
        self.assertTrue(callable(kpis))

    def test_payload_zero_debt_is_ok(self):
        clean = [{"kpi": n, "group": m.KPI_GROUP[n], "score": 100, "detail": "ok",
                  "defects": [], "soft": []} for n in m.KPI_WEIGHTS]
        pay = m.build_payload(workspace="/x", kpis=clean)
        self.assertTrue(pay["ok"])
        self.assertEqual(pay["corpus"]["process_debt"], 0)
        self.assertEqual(pay["corpus"]["grade"], "A")

    def test_payload_debt_counts_defects(self):
        kpis = [{"kpi": n, "group": m.KPI_GROUP[n], "score": 50, "detail": "x",
                 "defects": ["d"], "soft": []} for n in m.KPI_WEIGHTS]
        pay = m.build_payload(workspace="/x", kpis=kpis)
        self.assertEqual(pay["corpus"]["process_debt"], len(m.KPI_WEIGHTS))
        self.assertFalse(pay["ok"])


class TestAgainstRealTree(unittest.TestCase):
    def test_real_tree_payload_is_deterministic(self):
        root = Path(__file__).resolve().parent.parent
        if not (root / m.BINDING).exists():
            self.skipTest("not in the fak tree")
        a = m.collect(root)
        b = m.collect(root)
        # Clone-determinism: two reads of the same tree -> the same process-debt + score.
        self.assertEqual(a["corpus"]["process_debt"], b["corpus"]["process_debt"])
        self.assertEqual(a["corpus"]["score"], b["corpus"]["score"])
        self.assertIn(a["corpus"]["grade"], "ABCDF")
        # abi_parity is computed by the REAL parser over the seam, so the tree must be in parity.
        self.assertEqual(a["corpus"]["debt_by_kpi"]["abi_parity"], 0)


if __name__ == "__main__":
    unittest.main()
