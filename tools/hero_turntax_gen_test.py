#!/usr/bin/env python3
"""Tests for the fak turn-tax efficiency-curves generator.

Doubles as a CI drift + honesty guard: the three multipliers, the measured panel,
and the "we don't lead with naive" caption on the SOTA panel are asserted hard.

Run:  python -m pytest tools/hero_turntax_gen_test.py
"""
from __future__ import annotations

import json
import os
import sys
import xml.dom.minidom as minidom

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import hero_turntax_gen as gen  # noqa: E402

DATA = json.load(open(os.path.join(HERE, "hero_turntax.data.json"), encoding="utf-8"))


def _svg():
    return gen.svg_turntax(DATA)


def test_well_formed_xml():
    minidom.parseString(_svg())


def test_three_panels_each_with_a_multiplier():
    assert len(DATA["panels"]) == 3
    svg = _svg()
    for p in DATA["panels"]:
        assert p["title"] in svg
        assert p["mult"] in svg, f"missing multiplier {p['mult']!r}"


def test_each_panel_draws_both_a_baseline_and_a_fak_curve():
    # two polylines per panel (baseline + fak) -> >= 6 total
    svg = _svg()
    assert svg.count("<polyline") >= 2 * len(DATA["panels"])
    # both house colours present
    assert gen.BASE in svg and gen.FAK in svg


def test_measured_panel_is_labelled_measured():
    # panel 2 is a real WebVoyager elimination, not a model — say so.
    svg = _svg()
    assert "MEASURED" in svg.upper()
    assert "643" in svg
    assert "9.7×" in svg


def test_sota_panel_does_not_lead_with_naive():
    # The 50-turn fleet panel headlines the conservative 4.1x vs a TUNED SOTA cache;
    # the ~60x naive number is mentioned only as the thing we DON'T lead with.
    p3 = DATA["panels"][2]
    assert p3["mult"] == "4.1×"
    assert "tuned warm-cache SOTA" in p3["subtitle"]
    foot = p3["mult_foot"].lower()
    assert "60×" in p3["mult_foot"] and "don't lead with" in foot


def test_legend_and_footer_present():
    svg = _svg()
    for it in DATA["legend"]:
        assert it["label"] in svg
    assert DATA["footer"][:24] in svg


def test_check_is_idempotent():
    assert gen.main(["--check"]) == 0
