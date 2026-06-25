#!/usr/bin/env python3
"""Tests for conflation_scorecard  -  the anti-conflation / provenance-honesty stick.

Each KPI is exercised on a DEFECT fixture (the conflation it must catch) and a CLEAN fixture
(the honest phrasing it must pass), plus a live-tree smoke that pins the disciplined floor at
zero. The fixtures are the rendered fact-strings, since those are exactly what the KPIs read.
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import conflation_scorecard as cs  # noqa: E402


class TestProvenanceLabeled(unittest.TestCase):
    def test_unlabeled_external_value_is_debt(self):
        surfaces = {"x/metrics.go": [
            "Prompt tokens cache_read_input_tokens served by the provider across turns."]}
        k = cs.kpi_provenance_labeled(surfaces)
        self.assertTrue(k["defects"], "an external value with no OBSERVED qualifier must be debt")
        self.assertLess(k["score"], 100.0)

    def test_labeled_external_value_is_clean(self):
        surfaces = {"x/metrics.go": [
            "OBSERVED (provider-reported, relayed verbatim): cache_read_input_tokens on compacted turns."]}
        k = cs.kpi_provenance_labeled(surfaces)
        self.assertEqual(k["defects"], [], "an OBSERVED-labeled external value is honest")
        self.assertEqual(k["score"], 100.0)

    def test_honest_provider_side_phrasing_is_recognized(self):
        # The phrasing already in the tree ("provider-side reuse, distinct from the local caches")
        # is genuinely disambiguated and must NOT be flagged  -  recognizing real honesty.
        surfaces = {"x/metrics.go": [
            "Prompt tokens the provider served from its prompt cache (cache_read). This is "
            "provider-side reuse  -  distinct from the local fak caches."]}
        k = cs.kpi_provenance_labeled(surfaces)
        self.assertEqual(k["defects"], [])


class TestNoFalseAttribution(unittest.TestCase):
    def test_blaming_fak_for_observed_miss_is_debt(self):
        surfaces = {"x/metrics.go": [
            "cache_read on the most recent turn. Cratering to 0 while fires climb means the cache broke."]}
        k = cs.kpi_no_false_attribution(surfaces)
        self.assertTrue(k["defects"], "'the cache broke' attributed to fak with no qualifier is debt")
        self.assertEqual(k["score"], 0.0)

    def test_disambiguated_attribution_is_clean(self):
        # Describing the failure to PREVENT it (not asserting it) is honest.
        surfaces = {"x/metrics.go": [
            "If cache_read craters the prefix was still byte-identical, so the provider missed for a "
            "reason fak does NOT control. Reading the crater as 'the cache broke' is the conflation "
            "this split prevents."]}
        k = cs.kpi_no_false_attribution(surfaces)
        self.assertEqual(k["defects"], [])
        self.assertEqual(k["score"], 100.0)


class TestFaultSignalIsolated(unittest.TestCase):
    def test_mixed_family_without_fault_signal_is_soft(self):
        surfaces = {"x/metrics.go": [
            "WITNESSED fak authored shed tokens.",
            "OBSERVED provider cache_read_input_tokens relayed."]}
        k = cs.kpi_fault_signal_isolated(surfaces, {})
        self.assertTrue(k["soft"], "a mixed family naming no single fak-fault signal is a soft nudge")
        self.assertEqual(k["defects"], [], "fault isolation is SOFT, never HARD debt")

    def test_mixed_family_with_named_fault_is_clean(self):
        surfaces = {"x/metrics.go": [
            "WITNESSED fak authored shed; byte-identical prefix.",
            "OBSERVED provider cache_read_input_tokens; only bail_reason prefix_mismatch>0 is fak's bug."]}
        k = cs.kpi_fault_signal_isolated(surfaces, {})
        self.assertEqual(k["soft"], [])


class TestExtraction(unittest.TestCase):
    def test_extract_help_and_summary_strings(self):
        src = ('writeCounter(b, "n", "OBSERVED provider cache_read_input_tokens relayed", x)\n'
               'fmt.Fprintf(&b, "fak guard: compaction  -  %d fired", n)\n')
        got = cs.extract_help_strings(src)
        self.assertTrue(any("cache_read_input_tokens" in s for s in got))
        self.assertTrue(any("fak guard: compaction" in s for s in got))


class TestPayloadShape(unittest.TestCase):
    def test_control_pane_envelope(self):
        p = cs.run()
        for key in ("schema", "ok", "verdict", "finding", "next_action", "corpus", "kpis"):
            self.assertIn(key, p)
        self.assertIn("conflation_debt", p["corpus"])
        self.assertIn("grade", p["corpus"])


class TestLiveTreeFloor(unittest.TestCase):
    def test_disciplined_tree_emits_zero_conflation_debt(self):
        # The regression sentinel: the real reporting surfaces must stay provenance-honest.
        p = cs.run()
        self.assertEqual(
            p["corpus"]["conflation_debt"], cs.CLEAN_FLOOR,
            f"conflation debt rose above {cs.CLEAN_FLOOR}: {p['reason']}")

    def test_cli_json_exit_zero_on_clean(self):
        r = subprocess.run([sys.executable, str(REPO / "tools" / "conflation_scorecard.py"), "--json"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        payload = json.loads(r.stdout)
        self.assertTrue(payload["ok"])


if __name__ == "__main__":
    unittest.main()
