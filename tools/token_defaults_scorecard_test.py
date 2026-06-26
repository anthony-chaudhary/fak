#!/usr/bin/env python3
"""Tests for the token-saving-defaults scorecard.

Pure-core fixtures (each KPI's defect trigger AND its clean case, the source parsers, the
witness cross-check, the weight/grade fold) plus a live smoke that pins the real tree's
current floor — zero token-defaults-debt — so a regression that turns a saver off, drops a
lock, or breaks parity reds this test. Run::

    python tools/token_defaults_scorecard_test.py        # or: python -m pytest tools/token_defaults_scorecard_test.py
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import token_defaults_scorecard as tds  # noqa: E402


def _lever(**kw):
    """A minimal Lever with sane defaults, overridden by kw — for KPI fixtures."""
    base = tds.Lever(
        key="x", label="x", klass="lossless", sw_feature="", flag="", default_const="",
        entrypoints=("serve",), guard_symbol="", guard_file="", note_tokens=(),
        sentinel_tokens=(),
    )
    return replace(base, **kw)


class WeightsAndGrade(unittest.TestCase):
    def test_weights_sum_to_one_and_match_kpis(self):
        self.assertAlmostEqual(sum(tds.KPI_WEIGHTS.values()), 1.0, places=9)
        self.assertEqual(set(tds.KPI_WEIGHTS), set(tds.KPI_GROUP))

    def test_grade_letter_boundaries(self):
        self.assertEqual(tds.grade_letter(90), "A")
        self.assertEqual(tds.grade_letter(89.9), "B")
        self.assertEqual(tds.grade_letter(59), "F")


class SourceParsers(unittest.TestCase):
    def test_parse_const_ints(self):
        src = "const DefaultCompactHistoryBudget = 48000\nconst DefaultElideResultBytes = 0\n"
        got = tds.parse_const_ints(src)
        self.assertEqual(got["DefaultCompactHistoryBudget"], 48000)
        self.assertEqual(got["DefaultElideResultBytes"], 0)

    def test_parse_flag_defaults_and_resolve(self):
        src = ('x := fs.Int("ctx-view-budget", 0, "h")\n'
               'y := fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget, "h")\n'
               'z := fs.Bool("vdso", true, "h")\n')
        flags = tds.parse_flag_defaults(src)
        consts = {"DefaultCompactHistoryBudget": 48000}
        self.assertEqual(tds.resolve_default(flags["ctx-view-budget"][1], consts)[0], False)
        self.assertEqual(tds.resolve_default(flags["compact-history-budget"][1], consts)[0], True)
        self.assertEqual(tds.resolve_default(flags["vdso"][1], consts)[0], True)

    def test_resolve_unknown_symbol_abstains(self):
        on, disp = tds.resolve_default("gateway.DefaultMystery", {})
        self.assertIsNone(on)

    def test_servewiring_verdicts(self):
        src = '{"compacthistory", "--x", "F", verdictWired, "cs", "n"},\n{"ctxview", "--y", "G", verdictOffByDefault, "cs", "n"},'
        v = tds.parse_servewiring_verdicts(src)
        self.assertEqual(v["compacthistory"], "WIRED")
        self.assertEqual(v["ctxview"], "OFF_BY_DEFAULT_BUT_WIRED")
        self.assertTrue(tds.sw_verdict_is_on("WIRED"))
        self.assertFalse(tds.sw_verdict_is_on("OFF_BY_DEFAULT_BUT_WIRED"))


class WitnessCrossCheck(unittest.TestCase):
    def test_magnitude_is_a_witness(self):
        docs = {"CLAIMS.md": "the ctxplanbench replay: 13.3x fewer resident, 715/715 turns exact recall"}
        self.assertTrue(tds.find_witness(docs, ("ctxplan", "resident")))

    def test_mechanism_word_is_not_a_witness(self):
        # "elision sheds tokens" describes the mechanism — no number → not a witness.
        docs = {"docs/x.md": "the elide-result lever sheds an oversized result to head+tail"}
        self.assertFalse(tds.find_witness(docs, ("elide-result",)))

    def test_cross_feature_number_does_not_leak(self):
        # A magnitude on a DIFFERENT line must not witness this lever (the same-line rule).
        docs = {"CLAIMS.md": "prefill closed ~16x\nPlanned-elision -> KV eviction bridge"}
        self.assertFalse(tds.find_witness(docs, ("elide-result",)))


class KPIDefectTriggers(unittest.TestCase):
    def test_lossless_off_is_debt(self):
        on = tds.kpi_lossless_stack([_lever(klass="lossless", on=True)])
        off = tds.kpi_lossless_stack([_lever(klass="lossless", on=False, default_repr="off")])
        self.assertEqual(len(on["defects"]), 0)
        self.assertEqual(len(off["defects"]), 1)

    def test_high_value_unwitnessed_off_is_soft_not_hard(self):
        # The honesty rule: an OFF bounded saver with no witness is a soft "needs a witness",
        # never hard debt (flipping it on would ship an unproven claim).
        lv = _lever(klass="bounded", on=False, guard_present=True, witnessed=False, default_repr="off")
        out = tds.kpi_high_value_defaults([lv])
        self.assertEqual(len(out["defects"]), 0)
        self.assertEqual(len(out["soft"]), 1)

    def test_high_value_witnessed_ungated_off_is_hard_debt(self):
        # A PROVEN-safe saver with no remaining gate that is still off IS a real defect.
        lv = _lever(klass="bounded", on=False, guard_present=True, witnessed=True, gated=False,
                    default_repr="off")
        out = tds.kpi_high_value_defaults([lv])
        self.assertEqual(len(out["defects"]), 1)

    def test_high_value_witnessed_gated_off_is_soft(self):
        lv = _lever(klass="bounded", on=False, guard_present=True, witnessed=True, gated=True,
                    witness_note="13.3x")
        out = tds.kpi_high_value_defaults([lv])
        self.assertEqual(len(out["defects"]), 0)
        self.assertTrue(out["soft"])

    def test_dark_lever_ungated_is_debt(self):
        gated = tds.kpi_dark_lever_gated([_lever(on=False, gated=True)])
        bare = tds.kpi_dark_lever_gated([_lever(on=False, gated=False)])
        self.assertEqual(len(gated["defects"]), 0)
        self.assertEqual(len(bare["defects"]), 1)

    def test_default_notes_missing_is_debt(self):
        ok = tds.kpi_default_notes([_lever(klass="bounded", on=True, noted=True)])
        bad = tds.kpi_default_notes([_lever(klass="bounded", on=True, noted=False,
                                            note_tokens=("a", "b"))])
        self.assertEqual(len(ok["defects"]), 0)
        self.assertEqual(len(bad["defects"]), 1)

    def test_default_on_locked_missing_is_debt(self):
        ok = tds.kpi_default_on_locked([_lever(on=True, locked=True)])
        bad = tds.kpi_default_on_locked([_lever(on=True, locked=False, sentinel_tokens=("T",))])
        self.assertEqual(len(ok["defects"]), 0)
        self.assertEqual(len(bad["defects"]), 1)

    def test_entrypoint_parity_divergence_is_debt(self):
        lv = _lever(key="k", flag="f", on=False, sw_feature="")
        guard = {"f": ("Int", "0")}
        serve = {"f": ("Int", "100")}  # off in guard, on in serve → divergence
        out = tds.kpi_entrypoint_parity([lv], guard, serve, {})
        self.assertEqual(len(out["defects"]), 1)

    def test_servewiring_verdict_mismatch_is_debt(self):
        # Real default ON but the audited row says off-by-default → a stale wiring row.
        lv = _lever(key="k", flag="", on=True, sw_feature="k", sw_verdict="OFF_BY_DEFAULT_BUT_WIRED")
        out = tds.kpi_entrypoint_parity([lv], {}, {}, {})
        self.assertEqual(len(out["defects"]), 1)


class LiveSmoke(unittest.TestCase):
    def test_real_tree_holds_zero_debt(self):
        root = tds.repo_root()
        payload = tds.collect(root)
        self.assertNotEqual(payload.get("verdict"), "AUDIT_ERROR", payload.get("reason"))
        corpus = payload["corpus"]
        self.assertEqual(corpus["token_defaults_debt"], 0,
                         f"token-defaults-debt regressed: {payload['reason']}")
        self.assertGreaterEqual(corpus["score"], 90)
        # The roster shape the loop depends on: 6 levers, the 4 lossless/bounded-on stacked.
        self.assertEqual(corpus["levers_total"], 6)
        self.assertGreaterEqual(corpus["stacked_on"], 4)
        # elideresult is honestly unwitnessed; ctxview is witnessed — the roadmap invariant.
        by = {lv["key"]: lv for lv in corpus["lever_status"]}
        self.assertFalse(by["elideresult"]["witnessed"])
        self.assertTrue(by["ctxview"]["witnessed"])


if __name__ == "__main__":
    unittest.main()
