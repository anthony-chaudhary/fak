#!/usr/bin/env python3
"""Tests for the fak capability-matrix generator.

Doubles as a CI drift + honesty guard. The structural invariants are asserted
hard: the subject (fak) column is the only one carrying measured numbers; the
serving stacks carry an em-dash (not a fabricated zero) on the correctness and
security rows; the single-stream fence inverts the highlight to the SOTA leader;
every fak figure carries a commit; and the methodology spells out what "—" means.

Run:  python -m pytest tools/hero_matrix_gen_test.py
"""
from __future__ import annotations

import json
import os
import sys
import xml.dom.minidom as minidom

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import hero_matrix_gen as gen  # noqa: E402

DATA = json.load(open(os.path.join(HERE, "hero_matrix.data.json"), encoding="utf-8"))


def _svg():
    return gen.svg_matrix(DATA)


def _rows():
    for c in DATA["categories"]:
        for r in c["rows"]:
            yield c, r


def test_well_formed_xml():
    minidom.parseString(_svg())


def test_subject_column_is_highlighted():
    subj = [c for c in DATA["columns"] if c.get("subject")]
    assert len(subj) == 1 and subj[0]["key"] == "fak"
    svg = _svg()
    assert gen.GREEN_WASH in svg  # the fak-column wash


def test_fak_type_is_addressable():
    # The differentiator surfaced as a column type: fak == "addressable", the
    # serving stacks == "front-prefix" / "front-only".
    fak = next(c for c in DATA["columns"] if c.get("subject"))
    assert fak["type"] == "addressable"
    assert all(c.get("type") for c in DATA["columns"])
    svg = _svg()
    assert ">addressable<" in svg
    assert ">front-prefix<" in svg
    # rendered in the htypeF (white-on-green pill) class for fak
    assert 'class="htypeF"' in svg


def test_every_row_has_an_inline_coverage_pip_per_stack():
    # one pip per stack per row (filled if shipped, hollow ring if not); the
    # security/correctness rows must show fak alone (a single filled pip).
    svg = _svg()
    n_cols = len(DATA["columns"])
    n_rows = sum(len(c["rows"]) for c in DATA["categories"])
    assert svg.count("<circle") >= n_cols * n_rows  # pips (+ any methodology bullets)
    for c, r in _rows():
        covered = [k for k in r["cells"] if r["cells"][k][0] not in ("no", "na")]
        if c["name"].startswith("SECURITY"):
            assert covered == ["fak"], f"{r['cap']!r} should be fak-only, got {covered}"


def test_every_category_and_row_renders():
    svg = _svg()
    for c in DATA["categories"]:
        assert c["name"] in svg
        for r in c["rows"]:
            assert r["cap"] in svg


def test_only_fak_carries_measured_numbers_others_are_coverage_marks():
    # The fabrication guard: a non-subject column may carry a COVERAGE mark
    # (yes/no/na/weak/base/lead) but never a measured number ('num') — we only
    # benchmarked fak. And on the CORRECTNESS/SECURITY rows the serving stacks
    # must be an explicit em-dash ('no'), not even a soft ✓.
    for c, r in _rows():
        for col in DATA["columns"]:
            if col.get("subject"):
                continue
            kind, _ = r["cells"][col["key"]]
            assert kind != "num", f"{col['key']} on {r['cap']!r} carries a fabricated number"
            if c["name"].startswith("SECURITY"):
                assert kind == "no", f"{col['key']} on {r['cap']!r} should be an em-dash, got {kind}"


def test_single_stream_fence_inverts_to_the_sota_leader():
    fence = [c for c in DATA["categories"] if c.get("fence")]
    assert fence, "expected a single-stream fence category"
    for r in fence[0]["rows"]:
        # fak's own cell is a 'lose' (honest lower number); a competitor 'leads'
        assert r["cells"]["fak"][0] == "lose"
        assert any(r["cells"][k][0] == "lead" for k in r["cells"]), r["cap"]
    svg = _svg()
    assert "17.3 tok/s" in svg and "0.50×" in svg


def test_every_fak_figure_carries_a_commit():
    svg = _svg()
    for _c, r in _rows():
        assert r.get("commit"), f"row {r['cap']!r} has no commit"
        assert r["commit"] in svg


def test_methodology_explains_the_em_dash_and_omits_the_strawman():
    svg = _svg()
    assert "Evaluation methodology" in svg
    joined = " ".join(DATA["methodology"]["notes"])
    assert "not a measured zero" in joined
    assert "omitted as a strawman" in joined  # the naive ~60x is not led with


def test_check_is_idempotent():
    assert gen.main(["--check"]) == 0
