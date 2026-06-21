#!/usr/bin/env python3
"""Tests for the data-driven hero-benchmark generator.

These double as a CI drift guard: `test_check_is_idempotent` fails if the
committed doc/SVGs/html are stale vs `hero_benchmark.data.json` (someone edited
the data and forgot to rerun the generator), and `test_verify_sources` fails if
a data-file number no longer matches its committed benchmark artifact.

Run:  python -m pytest tools/hero_benchmark_gen_test.py
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import xml.dom.minidom as minidom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import hero_benchmark_gen as gen  # noqa: E402

DATA = json.load(open(os.path.join(HERE, "hero_benchmark.data.json"), encoding="utf-8"))


def test_data_has_required_sections():
    for k in ["meta", "headline", "cost_collapse", "top3_sota", "palette",
              "top10", "leaderboard", "lineage", "reproduce", "see_also"]:
        assert k in DATA, f"missing section: {k}"
    # leaderboard row count is data-driven (SOTA-only curation may add/drop rows);
    # require a meaningful comparison, not a fixed 10.
    assert len(DATA["top10"]) >= 4
    assert len(DATA["top3_sota"]["items"]) == 3
    assert len(DATA["cost_collapse"]["arms"]) >= 2


def test_sota_only_no_naive_baselines():
    # v2 framing rule: every leaderboard baseline is a real SOTA system, never a
    # naive/serial re-prefill strawman. Guards the SOTA-only curation from regressing.
    for r in DATA["top10"]:
        base = r["baseline"].lower()
        assert "naive" not in base and "serial" not in base, \
            f"row {r['rank']} baseline is a strawman, not SOTA: {r['baseline']!r}"


def test_win_loss_invariant():
    wins = [r for r in DATA["top10"] if r["win"]]
    losses = [r for r in DATA["top10"] if not r["win"]]
    assert len(wins) >= 1 and len(losses) >= 1
    assert len(wins) + len(losses) == len(DATA["top10"])
    # the only non-wins are the single-stream raw-throughput rows (honest fence)
    assert all(r["regime"] == "SINGLE STREAM" for r in losses)


def test_every_regime_has_a_palette_color():
    for r in DATA["top10"]:
        assert r["regime"] in DATA["palette"], f"no palette color for regime {r['regime']!r}"


def test_hero_svg_is_well_formed_and_has_anchors():
    svg = gen.svg_hero(DATA)
    minidom.parseString(svg)  # raises on malformed XML
    assert svg.lstrip().startswith("<svg")
    assert DATA["headline"]["primary"]["value"] in svg          # 4.1×
    assert DATA["headline"]["secondary"]["value"] in svg        # 60.3×
    for it in DATA["top3_sota"]["items"]:
        assert it["tag"] in svg


def test_leaderboard_bolds_wins_only():
    svg = gen.svg_leaderboard(DATA)
    minidom.parseString(svg)
    nwins = sum(1 for r in DATA["top10"] if r["win"])
    nloss = len(DATA["top10"]) - nwins
    assert svg.count('class="fakWin"') == nwins   # winners get the bold-green class
    assert svg.count('class="fakLose"') == nloss  # losers get the plain class
    assert svg.count(">✅<") == nwins         # ✅ verdict on wins
    for r in DATA["top10"]:
        assert r["benchmark"] in svg              # every benchmark row rendered


def test_doc_bolds_winning_fak_values_not_losing_ones():
    doc = gen.render_doc(DATA)
    win = next(r for r in DATA["top10"] if r["win"])
    loss = next(r for r in DATA["top10"] if not r["win"])
    assert f"**{win['fak_md']}**" in doc                       # win bolded
    assert f"| {loss['fak_md']} |" in doc                      # loss present, plain
    assert f"**{loss['fak_md']}**" not in doc                  # loss NOT bolded


def test_leaderboard_reflows_when_rows_change():
    base = gen.svg_leaderboard(DATA)
    d2 = copy.deepcopy(DATA)
    d2["top10"].append({**DATA["top10"][0], "rank": len(DATA["top10"]) + 1})
    bigger = gen.svg_leaderboard(d2)
    h1 = int(re.search(r'height="(\d+)"', base).group(1))
    h2 = int(re.search(r'height="(\d+)"', bigger).group(1))
    assert h2 == h1 + 66, "adding a benchmark row must grow the canvas by one row pitch"


def test_resolve_field_handles_dotted_indexed_paths():
    obj = {"cells": [{"x": 1.5}, {"x": 2.5}], "workloads": [{"y": 7.5}]}
    assert gen.resolve_field(obj, "cells[0].x") == 1.5
    assert gen.resolve_field(obj, "cells[1].x") == 2.5
    assert gen.resolve_field(obj, "workloads[0].y") == 7.5


def test_data_validates_against_json_schema():
    import pytest
    jsonschema = pytest.importorskip("jsonschema")  # skip on nodes without it
    schema = json.load(open(os.path.join(HERE, "schemas", "hero-benchmark.v1.json"), encoding="utf-8"))
    jsonschema.validate(DATA, schema)


def test_validate_data_catches_missing_palette_colour():
    import pytest
    bad = copy.deepcopy(DATA)
    bad["top10"][0]["regime"] = "NONEXISTENT REGIME"
    with pytest.raises(ValueError):
        gen.validate_data(bad)


def test_check_is_idempotent():
    # The committed doc/SVGs/html must already match the data file (drift guard).
    assert gen.main(["--check"]) == 0


def test_verify_sources_matches_committed_artifacts():
    # Every data-file `verify` block must still match its benchmark JSON.
    assert gen.main(["--verify-sources"]) == 0
