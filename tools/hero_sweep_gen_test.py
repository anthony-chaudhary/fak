#!/usr/bin/env python3
"""Tests for the fak benchmark-sweep grid generator.

Doubles as a CI drift + honesty guard. Invariants: eight panels across the three
pillars + a fence; every fak figure present and commit-traced; the fence panel
inverts the accent to llama.cpp (the SOTA leader owns the tall bar); and the two
capability panels compare against a field that ships no such gate.

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


def test_eight_panels_across_three_pillars_plus_a_fence():
    assert len(DATA["panels"]) == 8
    pillars = {p["pillar"] for p in DATA["panels"]}
    assert {"SERVING", "CORRECTNESS", "SECURITY", "FENCE"} <= pillars


def test_every_panel_and_fak_figure_renders():
    svg = _svg()
    for p in DATA["panels"]:
        assert p["name"] in svg
        for b in p["bars"]:
            assert b["label"] in svg, f"missing bar label {b['label']!r}"
    for fig in ["4.1×", "86.7%", "6.95×", "max|Δ|=0", "7.50×", "362 ns", "QUARANTINE", "8.7 tok/s", "17.3 tok/s"]:
        assert fig in svg, f"missing headline figure {fig!r}"


def test_fence_panel_inverts_accent_to_the_sota_leader():
    fence = [p for p in DATA["panels"] if p.get("fence")]
    assert len(fence) == 1
    bars = {b["who"]: b for b in fence[0]["bars"]}
    # llama.cpp is the subject (tall, accented) and the taller value; fak is not
    assert bars["llama.cpp"].get("subject") is True
    assert bars["llama.cpp"]["v"] > bars["fak"]["v"]
    assert not bars["fak"].get("subject")


def test_capability_panels_compare_against_an_unshipped_field():
    caps = [p for p in DATA["panels"] if p.get("capability")]
    assert len(caps) == 2  # mid-run eviction + injection containment
    for p in caps:
        field = [b for b in p["bars"] if not b.get("subject")][0]
        assert field["v"] == 0.0 and field["label"] == "—"


def test_every_panel_carries_a_commit():
    svg = _svg()
    for p in DATA["panels"]:
        assert p.get("commit") and p["commit"] in svg


def test_log_multiplier_decode_panel_present():
    svg = _svg()
    assert "2,849×" in svg  # in-process decide vs spawned hook


def test_check_is_idempotent():
    assert gen.main(["--check"]) == 0
