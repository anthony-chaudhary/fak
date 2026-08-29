#!/usr/bin/env python3
"""Tests for benchmark_authority.py — the registry→view generator.

Pure-stdlib (unittest). Covers the three properties the redesign leans on:
  * the RECORD is validated — a malformed registry fails loudly (exit 2), never
    renders a lossy view;
  * the VIEW is a deterministic pure function of the record — so `--check` is a
    real freshness gate;
  * the SHIPPED registry + committed sample view are in sync — the same
    discipline as TestSupportMaturityMatrixDocFresh, but for this doc.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent))
import benchmark_authority as ba
import bench_provenance

ROOT = Path(__file__).resolve().parents[1]

_GOOD_ROW = {
    "id": "sample-claim", "claim": "Sample claim", "headline": "2.0×",
    "status": "canonical", "tier": 1, "provenance": "measured",
    "model": "m", "baseline": "b", "commit": "abc1234",
    "artifact": "a.json", "reproduce": "read it", "fences": [],
}


def _write_registry(rows: list[dict]) -> str:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


class LoadRegistryTest(unittest.TestCase):
    def test_good_row_loads(self):
        path = _write_registry([_GOOD_ROW])
        try:
            rows = ba.load_registry(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], "sample-claim")
        finally:
            os.unlink(path)

    def test_blank_lines_skipped(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(_GOOD_ROW) + "\n\n   \n")
        try:
            self.assertEqual(len(ba.load_registry(path)), 1)
        finally:
            os.unlink(path)

    def test_missing_field_is_error(self):
        for field in ba.REQUIRED_FIELDS:
            bad = dict(_GOOD_ROW)
            del bad[field]
            path = _write_registry([bad])
            try:
                with self.assertRaises(ba.RegistryError, msg=f"missing {field} should raise"):
                    ba.load_registry(path)
            finally:
                os.unlink(path)

    def test_unknown_status_is_error(self):
        bad = dict(_GOOD_ROW, status="shipped")  # not in STATUS_ORDER
        path = _write_registry([bad])
        try:
            with self.assertRaises(ba.RegistryError):
                ba.load_registry(path)
        finally:
            os.unlink(path)

    def test_unknown_provenance_is_error(self):
        bad = dict(_GOOD_ROW, provenance="empirical")  # not in bench_provenance.TAGS
        path = _write_registry([bad])
        try:
            with self.assertRaises(ba.RegistryError):
                ba.load_registry(path)
        finally:
            os.unlink(path)

    def test_fences_must_be_list(self):
        bad = dict(_GOOD_ROW, fences="a single string")
        path = _write_registry([bad])
        try:
            with self.assertRaises(ba.RegistryError):
                ba.load_registry(path)
        finally:
            os.unlink(path)

    def test_duplicate_id_is_error(self):
        path = _write_registry([_GOOD_ROW, dict(_GOOD_ROW)])
        try:
            with self.assertRaises(ba.RegistryError):
                ba.load_registry(path)
        finally:
            os.unlink(path)

    def test_bad_json_is_error(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("{not json}\n")
        try:
            with self.assertRaises(ba.RegistryError):
                ba.load_registry(path)
        finally:
            os.unlink(path)


class RenderTest(unittest.TestCase):
    def test_render_is_deterministic(self):
        rows = ba.load_registry(_write_registry([_GOOD_ROW]))
        self.assertEqual(ba.render(rows), ba.render(rows))

    def test_render_injects_no_wallclock(self):
        # The view must be a pure function of the record so --check is a real
        # freshness gate. The generator itself must inject no wall-clock: render
        # a registry whose text carries NO dates and assert none appear, and
        # assert no "Last updated:"-style stamp is emitted.
        rows = ba.load_registry(_write_registry([_GOOD_ROW]))  # _GOOD_ROW has no dates
        out = ba.render(rows)
        self.assertNotIn("2026", out)
        self.assertNotIn("Last updated", out)
        self.assertNotIn("generated_at", out)

    def test_headline_table_only_tier1_canonical_or_live(self):
        rows = [
            dict(_GOOD_ROW, id="a", status="canonical", tier=1),
            dict(_GOOD_ROW, id="b", status="live", tier=1),
            dict(_GOOD_ROW, id="c", status="live", tier=2),      # excluded (tier 2)
            dict(_GOOD_ROW, id="d", status="stale", tier=1),     # excluded (stale)
        ]
        table = "\n".join(ba._headline_table(rows))
        self.assertIn("#claim-a", table)
        self.assertIn("#claim-b", table)
        self.assertNotIn("#claim-c", table)
        self.assertNotIn("#claim-d", table)

    def test_every_claim_gets_a_card_anchor(self):
        rows = ba.load_registry(str(ROOT / ba.REGISTRY_REL))
        out = ba.render(rows)
        for r in rows:
            self.assertIn(f'<a id="{ba._anchor(r["id"])}">', out,
                          f"{r['id']} has no card anchor")
            # The generator pipe-escapes cell text, so compare the rendered form.
            self.assertIn(ba._md(r["headline"]), out, f"{r['id']} headline missing from view")

    def test_measured_and_modeled_are_distinct_in_headline_table(self):
        # measured vs modeled is the whole point of the provenance discipline;
        # the headline-table code must not collapse them (both start with "m").
        rows = [
            dict(_GOOD_ROW, id="a", provenance="measured", tier=1, status="live"),
            dict(_GOOD_ROW, id="b", provenance="modeled", tier=1, status="live"),
        ]
        self.assertNotEqual(ba.PROVENANCE_CODE["measured"], ba.PROVENANCE_CODE["modeled"])
        table = "\n".join(ba._headline_table(rows))
        self.assertIn(ba.PROVENANCE_CODE["measured"], table)
        self.assertIn(ba.PROVENANCE_CODE["modeled"], table)

    def test_pipe_in_text_is_escaped(self):
        rows = [dict(_GOOD_ROW, claim="A | B", headline="1|2")]
        table = "\n".join(ba._headline_table(rows))
        self.assertIn("A \\| B", table)
        self.assertIn("1\\|2", table)


class ShippedRegistryTest(unittest.TestCase):
    """The committed registry and sample view must be real and in sync."""

    def test_registry_loads_and_tags_are_known(self):
        rows = ba.load_registry(str(ROOT / ba.REGISTRY_REL))
        self.assertGreaterEqual(len(rows), 1)
        for r in rows:
            self.assertIn(r["provenance"], bench_provenance.TAGS)
            self.assertIn(r["status"], ba.STATUS_ORDER)

    def test_superseded_by_targets_exist(self):
        rows = ba.load_registry(str(ROOT / ba.REGISTRY_REL))
        ids = {r["id"] for r in rows}
        for r in rows:
            if r.get("superseded_by"):
                self.assertIn(r["superseded_by"], ids,
                              f"{r['id']} superseded_by dangling {r['superseded_by']!r}")

    def test_canonical_support_maturity_row_rejects_stale_and_accepts_registry(self):
        rows = ba.load_registry(str(ROOT / ba.REGISTRY_REL))
        support = next(r for r in rows if r["id"] == "support-maturity-matrix")
        stale = ba.canonical_row({
            **support,
            "headline": (
                "Grade A · score 100 · support_maturity_debt 0 · "
                "32/56 cells SUPPORTED (0 PROOF-PATH-ONLY, 24 FENCED, "
                "0 UNDEFINED across 14 families × 4 backends) — "
                "a coverage instrument, not a vs-baseline win"
            ),
        })
        fresh = ba.canonical_row(support)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / ba.CANONICAL_AUTHORITY_REL
            target.write_text(stale + "\n", encoding="utf-8")
            self.assertEqual(
                ba.check_canonical_rows(str(root), rows),
                ["support-maturity-matrix"],
            )
            target.write_text(fresh + "\n", encoding="utf-8")
            self.assertEqual(ba.check_canonical_rows(str(root), rows), [])

    def test_committed_sample_view_is_fresh(self):
        # Mirrors --check: the shipped AUTHORITY-GENERATED-SAMPLE.md must equal
        # a fresh render of the shipped registry, or the doc has drifted.
        rc = ba.main(["--root", str(ROOT), "--check", ba.DEFAULT_VIEW_REL])
        self.assertEqual(rc, 0, "sample view is stale — run "
                                "`python tools/benchmark_authority.py --write "
                                f"{ba.DEFAULT_VIEW_REL}`")


if __name__ == "__main__":
    unittest.main()
