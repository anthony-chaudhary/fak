#!/usr/bin/env python3
"""Hermetic tests for tools/context_tape.py.

Three things must hold for the tape to be trustworthy:

  1. PARITY — the synthetic `scenario fleet-5x50` reproduces the live demo's headline
     panel EXACTLY. The per-agent token totals are the Go catalog's, recomputed by the
     Python LCG mirror; if the mirror drifts from cmd/ctxdemo/scenario.go the totals move
     and this test fails. (9,569 / 9,343 / 8,120 / 9,568 / 8,991 are the demo's numbers.)

  2. EXACT BANDS — a real trajectory's bands are provider token counts, not estimates,
     and billed turns are de-duped on message.id the same way session_audit.py folds them
     (Claude Code writes multiple transcript lines per billed turn). A regression here
     would silently double a turn's apparent context.

  3. WELL-FORMED, ESCAPED SVG — every adapter renders a parseable SVG, and a hostile tool
     name (`a<b&"c`) is escaped rather than breaking the XML. The Tape JSON is the modular
     seam, so it must round-trip to an identical render.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import xml.dom.minidom as minidom
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "context_tape.py"


def load():
    spec = importlib.util.spec_from_file_location("context_tape", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["context_tape"] = mod   # dataclasses look the module up in sys.modules
    spec.loader.exec_module(mod)
    return mod


def _assistant(msg_id, *, inp=0, out, cread, ccreate, tool=None, model="claude-opus-4-8"):
    """One assistant transcript record (the schema session_audit.py reads)."""
    content = [{"type": "tool_use", "name": tool, "input": {}}] if tool else [{"type": "text"}]
    return {"type": "assistant", "timestamp": "2026-06-24T00:00:00.000Z",
            "message": {"id": msg_id, "model": model,
                        "usage": {"input_tokens": inp, "output_tokens": out,
                                  "cache_read_input_tokens": cread,
                                  "cache_creation_input_tokens": ccreate},
                        "content": content}}


def _write_jsonl(records):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in records:
        tmp.write(json.dumps(r) + "\n")
    tmp.close()
    return tmp.name


class ScenarioParityTest(unittest.TestCase):
    def test_fleet_5x50_totals_match_the_demo(self):
        ct = load()
        tape = ct.scenario_tape("fleet-5x50")
        totals = [r.total_label for r in tape.rows]
        # These are the live demo's exact per-agent context totals (cmd/ctxdemo).
        self.assertEqual(totals, ["9,569 tok", "9,343 tok", "8,120 tok",
                                  "9,568 tok", "8,991 tok"])

    def test_lcg_is_deterministic(self):
        ct = load()
        a = ct.scenario_workload("mixed-fleet")
        b = ct.scenario_workload("mixed-fleet")
        self.assertEqual(a, b, "the seeded LCG must be a pure function of the scenario")

    def test_every_catalog_scenario_builds_a_tape(self):
        ct = load()
        for sid in ct.CATALOG:
            tape = ct.scenario_tape(sid)
            self.assertEqual(len(tape.rows), ct.CATALOG[sid]["agents"])
            # legend = prefix + decode + one entry per tool
            self.assertEqual(len(tape.legend), 2 + len(ct.CATALOG[sid]["tools"]))
            minidom.parseString(ct.render_svg(tape))  # raises if malformed


class TrajectoryTest(unittest.TestCase):
    def test_billed_turn_deduped_on_message_id(self):
        ct = load()
        recs = ([_assistant("m1", inp=200, out=120, cread=0, ccreate=1800, tool="Read")] * 3
                + [_assistant("m2", inp=40, out=260, cread=2000, ccreate=900, tool="Grep")])
        turns = ct.trajectory_turns(_write_jsonl(recs))
        self.assertEqual(len(turns), 2, "three lines of m1 are ONE billed turn")
        self.assertEqual(turns[0]["cache_read"], 0)
        self.assertEqual(turns[0]["cache_create"], 1800)
        self.assertEqual(turns[0]["output"], 120)
        self.assertEqual(turns[0]["tools"], ["Read"])

    def test_bands_are_exact_usage(self):
        ct = load()
        recs = [_assistant("m1", inp=40, out=260, cread=2000, ccreate=900, tool="Grep")]
        tape = ct.trajectory_tape(_write_jsonl(recs))
        segs = {s.key: s.value for s in tape.rows[0].segments}
        self.assertEqual(segs["reused"], 2000)        # cache_read
        self.assertEqual(segs["Grep"], 40 + 900)      # input + cache_creation, coloured by tool
        self.assertEqual(segs["decode"], 260)         # output
        self.assertEqual(tape.rows[0].total_label, "2,940 in")  # reused + fresh

    def test_no_tool_turn_uses_none_key(self):
        ct = load()
        recs = [_assistant("m1", inp=10, out=90, cread=5000, ccreate=300)]
        tape = ct.trajectory_tape(_write_jsonl(recs))
        keys = [s.key for s in tape.rows[0].segments]
        self.assertIn("none", keys)

    def test_sampling_is_noted_not_silent(self):
        ct = load()
        recs = [_assistant(f"m{i}", inp=10, out=10, cread=100 * i, ccreate=50, tool="Read")
                for i in range(120)]
        tape = ct.trajectory_tape(_write_jsonl(recs), max_rows=20)
        self.assertEqual(len(tape.rows), 20)
        self.assertIn("of 120 turns", tape.caption)  # truncation is disclosed in the caption

    def test_empty_trajectory_raises(self):
        ct = load()
        with self.assertRaises(SystemExit):
            ct.trajectory_tape(_write_jsonl([{"type": "user", "message": {"content": "hi"}}]))


class RsiTest(unittest.TestCase):
    def _journal(self):
        return [
            {"cycle": 1, "candidate": "cache=8", "metric_name": "vdso_hit_rate",
             "baseline": 0.50, "candidate_metric": 0.62, "measured": True,
             "decision": "KEEP", "kept": True, "lower_better": False},
            {"cycle": 2, "candidate": "cache=16", "metric_name": "vdso_hit_rate",
             "baseline": 0.62, "candidate_metric": 0.60, "measured": True,
             "decision": "REVERT", "kept": False, "lower_better": False},
        ]

    def test_verdict_colours_the_contested_delta(self):
        ct = load()
        tape = ct.rsi_tape(_write_jsonl(self._journal()))
        self.assertEqual(len(tape.rows), 2)
        # row 1 keep: floor 0.50 baseline + delta 0.12 tinted "keep"
        keys1 = [s.key for s in tape.rows[0].segments]
        self.assertEqual(keys1, ["baseline", "keep"])
        self.assertAlmostEqual(tape.rows[0].segments[1].value, 0.12, places=6)
        # row 2 revert: floor 0.60 candidate + delta 0.02 tinted "revert"
        keys2 = [s.key for s in tape.rows[1].segments]
        self.assertEqual(keys2, ["baseline", "revert"])
        self.assertIn("reverted", tape.rows[1].total_label)

    def test_unmeasured_candidate_labelled_na(self):
        ct = load()
        j = [{"cycle": 1, "candidate": "broken", "metric_name": "m", "baseline": 0.5,
              "candidate_metric": 0.5, "measured": False, "decision": "REVERT",
              "kept": False, "lower_better": False}]
        tape = ct.rsi_tape(_write_jsonl(j))
        self.assertIn("n/a", tape.rows[0].total_label)


class RenderTest(unittest.TestCase):
    def test_hostile_tool_name_is_escaped(self):
        ct = load()
        tape = ct.Tape(title='t & <x>', rows=[
            ct.Row(label="agent 0", segments=[ct.Seg('a<b&"c', 100, 'a<b')], total_label="100")])
        svg = ct.render_svg(tape)
        minidom.parseString(svg)  # must remain well-formed despite the hostile strings
        self.assertIn("&lt;", svg)
        self.assertIn("&amp;", svg)

    def test_tape_json_round_trips_to_identical_svg(self):
        ct = load()
        tape = ct.scenario_tape("coding-agent")
        back = ct.Tape.from_json(json.loads(json.dumps(tape.to_json())))
        self.assertEqual(ct.render_svg(back), ct.render_svg(tape))

    def test_ascii_has_a_box_glyph_and_labels(self):
        ct = load()
        tape = ct.scenario_tape("support-bot")
        art = ct.render_ascii(tape)
        self.assertIn("│", art)
        self.assertIn("agent 0", art)

    def test_html_embeds_the_svg(self):
        ct = load()
        html = ct.render_html(ct.scenario_tape("support-bot"))
        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertIn("<svg", html)


if __name__ == "__main__":
    unittest.main()
