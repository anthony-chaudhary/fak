#!/usr/bin/env python3
"""Tests for the fak benchmark-breadth card generator.

Doubles as a CI drift + honesty guard. Expectations are derived from the shared
`hero_statcard.data.json`, so the card stays correct as the data evolves — but
the structural honesty invariants (breadth not a hero number; the NAIVE row is
fenced from the SOTA rows; the single-stream losses are inverted and visible;
every number carries a commit) are asserted hard.

Run:  python -m pytest tools/hero_statcard_gen_test.py
"""
from __future__ import annotations

import json
import os
import sys
import xml.dom.minidom as minidom

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import hero_statcard_gen as gen  # noqa: E402

DATA = json.load(open(os.path.join(HERE, "hero_statcard.data.json"), encoding="utf-8"))


def _svg():
    return gen.svg_statcard(DATA)


def _all_rows():
    for p in DATA["pillars"]:
        for r in p["rows"]:
            yield r


def test_card_is_well_formed_xml():
    minidom.parseString(_svg())


def test_breadth_not_a_single_number():
    # The whole point: a capabilities sweep, not one hero stat. >= 3 pillars,
    # each carrying multiple benchmarks, and every pillar's lead metric shares the
    # same type class (no single number enlarged above the others).
    assert len(DATA["pillars"]) >= 3
    assert all(len(p["rows"]) >= 3 for p in DATA["pillars"])
    assert sum(len(p["rows"]) for p in DATA["pillars"]) >= 9
    svg = _svg()
    # the metric type sizes are fixed by class (.big / .bigV) — no per-tile font bump
    assert 'class="big"' in svg and 'class="bigV"' in svg


def test_every_pillar_and_row_renders():
    svg = _svg()
    for p in DATA["pillars"]:
        assert p["name"] in svg
        for r in p["rows"]:
            assert r["metric"] in svg, f"missing metric: {r['metric']!r}"
            assert r["label"][:24] in svg, f"missing label: {r['label']!r}"
    for r in DATA["fence"]["rows"]:
        assert r["label"] in svg and r["owned"] in svg and r["fak"] in svg


def test_naive_row_is_fenced_from_the_sota_rows():
    # The biggest number on the card (139.3x) is vs a NAIVE baseline. It must be
    # rendered in NEUTRAL GRAY (not a pillar accent), carry the NAIVE chip, and be
    # explicitly captioned so it can never read as a SOTA win.
    svg = _svg()
    naive = [r for r in _all_rows() if r.get("naive")]
    assert naive, "expected at least one NAIVE-baselined row"
    for r in naive:
        assert f'style="fill:{gen.GRAY};">{r["metric"]}' in svg     # gray, not accent
        assert r.get("chip") == "NAIVE"
    assert "NAIVE baseline" in svg and "not a SOTA win" in svg


def test_4_1x_is_sota_style_not_a_green_win():
    # The 4.1x baseline is fak's OWN kernel held constant (isolates reuse, not
    # kernel speed) — it must be [SOTA-style] (teal), never green [SOTA], and carry
    # the "reuse isolated" qualifier.
    svg = _svg()
    assert "SOTA-style" in svg
    assert "reuse isolated" in svg
    # the green WIN fill must not be used as a metric colour anywhere on this card
    assert f'style="fill:{gen.GREEN};">' not in svg


def test_single_stream_fence_is_inverted_and_visible():
    # llama.cpp OWNS the bold baseline number; fak sits in a loss cell with a
    # [LOSS] tag and an em-dash verdict — never dressed as a win.
    svg = _svg()
    n = len(DATA["fence"]["rows"])
    assert svg.count(">LOSS<") == n
    assert svg.count(">—<") == n
    assert "llama.cpp" in svg


def test_verdict_tiles_use_a_check_not_a_bar():
    svg = _svg()
    verdicts = [r for r in _all_rows() if r.get("verdict")]
    assert verdicts
    assert svg.count("✓") >= len(verdicts)


def test_every_metric_carries_a_commit():
    svg = _svg()
    for r in _all_rows():
        assert r.get("commit"), f"row {r['label']!r} has no commit"
        assert r["commit"] in svg
    for r in DATA["fence"]["rows"]:
        assert r["commit"] in svg


def test_loss_number_is_verifiable_against_its_real_artifact():
    # Don't only verify the wins: the single-stream decode loss (8.7 tok/s) must
    # match the committed modelbench artifact.
    art = os.path.join(ROOT, "fak", "experiments", "model-ladder", "modelbench-qwen25-7b-q8.json")
    obj = json.load(open(art, encoding="utf-8"))
    assert abs(round(obj["decode"]["tok_per_sec"], 1) - 8.7) < 1e-9
    assert "8.7 tok/s" in _svg()


def test_provenance_and_synthesis_present():
    svg = _svg()
    assert "BENCHMARK-AUTHORITY.md" in svg
    assert "one chat → llama.cpp" in svg
    assert "an agent fleet → fak" in svg


def test_check_is_idempotent():
    assert gen.main(["--check"]) == 0
