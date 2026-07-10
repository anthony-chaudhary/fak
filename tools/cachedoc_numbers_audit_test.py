#!/usr/bin/env python3
"""Hermetic contract tests for the doc-numbers gate.

Each case builds a throwaway guarded doc + manifest + trimmed snapshot in a
tempdir and asserts the gate's pass/fail contract:

  * a doc whose numbers trace to their snapshot and close arithmetically PASSES;
  * a rendered string missing from the doc FAILS (binding);
  * a manifest expected that disagrees with its snapshot field FAILS (snapshot);
  * a doc render that is not a correct rounding of the value FAILS (rounding);
  * a decomposition that does not sum FAILS (invariants) — the 60≠15+36 guard
    this gate was born from;
  * a table row missing its provenance label WARNs but does not FAIL;
  * a snapshot older than stale_after_days WARNs but does not FAIL.

No real repo is touched. The live/refresh paths (which need `fak`) are not
exercised here — the hermetic audit is the CI contract.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cachedoc_numbers_audit as cda  # noqa: E402

STEM = "widget"
DOC_REL = "docs/widget.md"


def _good():
    """A canonical passing (doc_text, manifest, snapshot) triple."""
    doc = (
        "# Widget\n\n"
        "| metric | value | provenance | field |\n"
        "|---|---|---|---|\n"
        "| Fires | 100 fires over 40 sessions | WITNESSED | `m.fires` |\n"
        "| Cache read | 1.23M | OBSERVED | `m.big` |\n"
        "| Avoided | **$60.00 (50.0% reduction)** | OBSERVED | `m.avoided` |\n"
        "| Sessions | 10 discovered — 3 priced, 7 held out (5 synthetic, 2 other) | WITNESSED | `m.disc` |\n"
    )
    snapshot = {"m": {"fires": 100, "sessions": 40, "big": 1234567,
                      "counter": 120.0, "actual": 60.0, "avoided": 60.0,
                      "reduction": 50.0, "disc": 10, "priced": 3, "syn": 5}}
    manifest = {
        "schema": cda.SCHEMA,
        "doc": DOC_REL,
        "snapshot_dir": f"tools/docnumbers/snapshots/{STEM}",
        "snapshot_date": dt.date.today().isoformat(),
        "stale_after_days": 30,
        "sources": {"main.json": {"cmd": None, "window": "live_snapshot"}},
        "claims": [
            {"id": "fires", "appears_as": "100 fires over 40 sessions",
             "source": "main.json", "provenance": "WITNESSED",
             "numbers": [{"display": "100", "field": "m.fires", "expected": 100},
                         {"display": "40", "field": "m.sessions", "expected": 40}]},
            {"id": "big", "appears_as": "1.23M", "source": "main.json",
             "provenance": "OBSERVED",
             "numbers": [{"display": "1.23M", "field": "m.big", "expected": 1234567}]},
            {"id": "avoided", "appears_as": "**$60.00 (50.0% reduction)**",
             "source": "main.json", "provenance": "OBSERVED",
             "numbers": [{"display": "$60.00", "field": "m.avoided", "expected": 60.0},
                         {"display": "50.0%", "field": "m.reduction", "expected": 50.0}]},
            {"id": "sessions",
             "appears_as": "10 discovered — 3 priced, 7 held out (5 synthetic, 2 other)",
             "source": "main.json", "provenance": "WITNESSED",
             "numbers": [{"display": "10", "field": "m.disc", "expected": 10},
                         {"display": "3", "field": "m.priced", "expected": 3},
                         {"display": "5", "field": "m.syn", "expected": 5}]},
        ],
        "invariants": [
            {"kind": "sum", "label": "sessions", "total": 10, "parts": [3, 5, 2]},
            {"kind": "formula", "label": "avoided",
             "expr": "120.0 - 60.0", "expect": 60.0, "tol": 0.001},
        ],
    }
    return doc, manifest, snapshot


def _run(doc, manifest, snapshot):
    """Write the triple into a tempdir and return (exit_code, fails, warns)."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "docs").mkdir(parents=True)
        (root / DOC_REL).write_text(doc, encoding="utf-8")
        snap_dir = root / manifest["snapshot_dir"]
        snap_dir.mkdir(parents=True)
        (snap_dir / "main.json").write_text(json.dumps(snapshot), encoding="utf-8")
        man_dir = root / "tools" / "docnumbers"
        man_dir.mkdir(parents=True, exist_ok=True)
        (man_dir / f"{STEM}.json").write_text(json.dumps(manifest), encoding="utf-8")
        # the tool prints its report; the tests assert on the return code and the
        # structured audit result, so swallow its output to keep CI logs clean.
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = cda.main(["--root", str(root)])
            loaded = cda.load_manifests(str(root), None)
            fails, warns = cda.audit_manifest(str(root), loaded[0], dt.date.today())
        return code, fails, warns


def _checks(items):
    return {i["check"] for i in items}


class DocNumbersGate(unittest.TestCase):
    def test_good_passes(self):
        code, fails, warns = _run(*_good())
        self.assertEqual(code, 0, f"clean triple must PASS; fails={fails}")
        self.assertEqual(fails, [])

    def test_binding_fail(self):
        doc, man, snap = _good()
        doc = doc.replace("100 fires over 40 sessions", "lots of fires")
        code, fails, _ = _run(doc, man, snap)
        self.assertEqual(code, 1)
        self.assertIn("binding", _checks(fails))

    def test_snapshot_fail(self):
        doc, man, snap = _good()
        # manifest claims 100 but the snapshot field says 999
        snap["m"]["fires"] = 999
        code, fails, _ = _run(doc, man, snap)
        self.assertEqual(code, 1)
        self.assertIn("snapshot", _checks(fails))

    def test_rounding_fail(self):
        doc, man, snap = _good()
        # doc renders $61.00 but the true value is $60.00
        doc = doc.replace("$60.00", "$61.00")
        for c in man["claims"]:
            if c["id"] == "avoided":
                c["appears_as"] = c["appears_as"].replace("$60.00", "$61.00")
                c["numbers"][0]["display"] = "$61.00"
        code, fails, _ = _run(doc, man, snap)
        self.assertEqual(code, 1)
        self.assertIn("rounding", _checks(fails))

    def test_sum_invariant_fail(self):
        # the real bug: 10 discovered read as 3 + 5 (=8), not 3 + 5 + 2.
        doc, man, snap = _good()
        for inv in man["invariants"]:
            if inv["kind"] == "sum":
                inv["parts"] = [3, 5]  # drops the "2 other" — 8 != 10
        code, fails, _ = _run(doc, man, snap)
        self.assertEqual(code, 1)
        self.assertIn("invariants", _checks(fails))

    def test_formula_invariant_fail(self):
        doc, man, snap = _good()
        for inv in man["invariants"]:
            if inv["kind"] == "formula":
                inv["expect"] = 99.0  # 120 - 60 = 60, not 99
        code, fails, _ = _run(doc, man, snap)
        self.assertEqual(code, 1)
        self.assertIn("invariants", _checks(fails))

    def test_provenance_missing_warns_not_fails(self):
        doc, man, snap = _good()
        # strip the WITNESSED label from the fires row
        doc = doc.replace("| 100 fires over 40 sessions | WITNESSED |",
                          "| 100 fires over 40 sessions | |")
        code, fails, warns = _run(doc, man, snap)
        self.assertEqual(code, 0, "a missing provenance label is a WARN, not a FAIL")
        self.assertIn("provenance", _checks(warns))

    def test_staleness_warns_not_fails(self):
        doc, man, snap = _good()
        man["snapshot_date"] = "2000-01-01"
        man["stale_after_days"] = 7
        code, fails, warns = _run(doc, man, snap)
        self.assertEqual(code, 0, "a stale snapshot is a WARN, not a FAIL")
        self.assertIn("staleness", _checks(warns))

    def test_approx_render_ok(self):
        doc, man, snap = _good()
        # "≈$60" is an honest approx of 60.0 and must PASS
        doc = doc.replace("| Fires |", "| About | ≈$60 saved | WITNESSED | `m.avoided` |\n| Fires |")
        man["claims"].append({
            "id": "approx", "appears_as": "≈$60 saved", "source": "main.json",
            "provenance": "WITNESSED",
            "numbers": [{"display": "≈$60", "field": "m.avoided", "expected": 60.0}]})
        code, fails, _ = _run(doc, man, snap)
        self.assertEqual(code, 0, f"approx render must PASS; fails={fails}")


class ParseDisplay(unittest.TestCase):
    def test_parses_variants(self):
        self.assertAlmostEqual(cda.parse_display("62.6M")["value"], 62.6e6)
        self.assertAlmostEqual(cda.parse_display("$1,181.37")["value"], 1181.37)
        self.assertAlmostEqual(cda.parse_display("89.3%")["value"], 89.3)
        self.assertTrue(cda.parse_display("≈34,000")["approx"])
        self.assertFalse(cda.parse_display("2,347")["approx"])

    def test_rounding_boundaries(self):
        # a correct rounding passes; a wrong one fails
        self.assertTrue(cda.rounding_ok("$18.64", 18.635071500000002)[0])
        self.assertTrue(cda.rounding_ok("89.3%", 89.34669014110735)[0])
        self.assertFalse(cda.rounding_ok("$18.70", 18.635071500000002)[0])
        self.assertTrue(cda.rounding_ok("≈20%", 20.307425929069925)[0])

    def test_safe_eval_rejects_non_arithmetic(self):
        with self.assertRaises(ValueError):
            cda.safe_eval("__import__('os').system('echo hi')")


if __name__ == "__main__":
    unittest.main()
