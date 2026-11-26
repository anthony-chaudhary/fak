#!/usr/bin/env python3
"""Tests for the fak benchmark-sweep grid generator.

Doubles as a CI drift + honesty guard. Invariants: eight panels, Inkling-2-Mini
inference first, then model-independent kernel axes, then a fence; every figure present and
commit-traced; every inference panel names a non-fak engine to beat (except the
opt-in spec panel, which is fak against its own default); and the fence panel
inverts the accent to the engine that leads.

Run:  python -m pytest tools/hero_sweep_gen_test.py
"""
from __future__ import annotations

import json
import os
import sys
import xml.dom.minidom as minidom

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import hero_sweep_gen as gen  # noqa: E402

DATA = json.load(open(os.path.join(HERE, "hero_sweep.data.json"), encoding="utf-8"))


def _svg():
    return gen.svg_sweep(DATA)


def test_well_formed_xml():
    minidom.parseString(_svg())


def test_eight_panels_inference_first_then_kernel_then_a_fence():
    assert len(DATA["panels"]) == 8
    pillars = [p["pillar"] for p in DATA["panels"]]
    assert pillars[0] == "INFERENCE" and pillars[-1] == "FENCE"
    assert {"INFERENCE", "CORRECTNESS", "SECURITY", "FENCE"} <= set(pillars)
    assert all(p in gen.PILLAR for p in pillars)


def test_every_panel_and_headline_figure_renders():
    svg = _svg()
    for p in DATA["panels"]:
        assert p["name"] in svg
        for b in p["bars"]:
            assert b["label"] in svg, f"missing bar label {b['label']!r}"
    for fig in ["118.1 tok/s", "81.0", "119.9 tok/s", "119.5", "171.5 tok/s", "119.7",
                "40.3", "37.5", "23.4", "5861 tok/s", "3012", "max|Δ|=0", "2.4 µs", "6.9 ms", "2,849×",
                "89.0 tok/s", "113.8 tok/s"]:
        assert fig in svg, f"missing headline figure {fig!r}"


def test_desc_names_the_figures_shown():
    svg = _svg()
    assert DATA["desc"].split(".")[0] in svg
    for fig in ["118.1", "81.0", "171.5", "5861", "3012", "40.3", "2,849x", "89.0", "113.8"]:
        assert fig in DATA["desc"]


def test_inference_panels_compare_against_another_engine():
    for p in DATA["panels"]:
        if p["pillar"] != "INFERENCE":
            continue
        refs = [b for b in p["bars"] if not b.get("subject")]
        assert refs, p["name"]
        if p["name"] == "Speculative decode":
            assert all(b["who"].startswith("fak") for b in p["bars"])  # opt-in vs default
            continue
        assert all(not b["who"].startswith("fak") for b in refs), p["name"]
        subj = [b for b in p["bars"] if b.get("subject")][0]
        assert subj["who"] == "fak"


def test_fence_panel_inverts_accent_to_the_leader():
    fence = [p for p in DATA["panels"] if p.get("fence")]
    assert len(fence) == 1
    bars = fence[0]["bars"]
    fak = [b for b in bars if b["who"] == "fak"][0]
    leader = [b for b in bars if b.get("subject")][0]
    assert leader["who"] != "fak" and not fak.get("subject")
    assert leader["v"] > fak["v"]


def test_every_panel_carries_a_commit():
    svg = _svg()
    for p in DATA["panels"]:
        assert p.get("commit") and p["commit"] in svg


def test_log_multiplier_decode_panel_present():
    svg = _svg()
    assert "2,849×" in svg  # in-process decide vs spawned hook


def test_check_is_idempotent():
    assert gen.main(["--check"]) == 0
