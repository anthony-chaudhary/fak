#!/usr/bin/env python3
"""Tests for the implicit-explicit scorecard.

Three things are exercised. (1) The pure helpers: grade, token normalization, the
heading -> concept-phrase reduction (structure headings drop out), and the
explicitness verdict the evidence IMPLIES (explicit needs a resolving symbol AND an
anchored definition). (2) Each KPI's defect trigger - the malformed-row catch, the
UNEVIDENCED catch (a fictional signal the strict cross-check refuses), the dangling
name-claim catch, the missing/dangling anchor catch, the no-naming-plan catch, and
the verdict-overclaim catch - plus each of the four detectors (hedges, magic
literals with the trivial/idiom floor, repo-declared code-only identifiers, and
recurring doc-only headings) and the exact-key coverage fold. (3) The disk shell +
the fold to the composite.

Closes with the load-bearing live smoke: the REAL committed catalog must fold to
ZERO naming-debt (every positioned concept is clean), carry nonzero coverage-debt
(the implicit-signal space is only partly mapped - the honest birth state), and
score in the intended band.

Run: `python tools/implicit_explicit_scorecard_test.py`  (exit 0 = all pass),
or `python -m pytest tools/implicit_explicit_scorecard_test.py -q`.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import implicit_explicit_scorecard as ie  # noqa: E402


def row(**over) -> dict:
    """A minimal well-formed, evidenced, fully explicit row."""
    r = {
        "id": "r1", "canonical": "warm window", "signal": "hinted",
        "evidence": "warm window", "evidence_kind": "phrase",
        "current_name": "", "proposed_name": "WarmWindow",
        "named_symbol": "warmwindow", "doc_anchor": "docs/g.md",
        "definition": "the interval a cache entry stays hot", "aliases": [],
        "verdict": "explicit", "gaps": [],
    }
    r.update(over)
    return r


def tree(**over) -> dict:
    """Synthetic tree-facts covering every detector input + verification predicate."""
    code_tokens = {"warmwindow", "alpha", "budgettokens"}
    t = {
        "sym_files": {"warmwindow": {"a.go", "b.go"},
                      "alpha": {"a.go"},
                      "budgettokens": {f"f{i}.go" for i in range(12)}},
        "structural": {"gateway"},
        "decl_tokens": {"budgettokens", "warmwindow"},
        "literal_files": {"300": {f"f{i}.go" for i in range(6)},
                          "0.8": {"a.go", "b.go", "c.go"}},
        "hedges": [{"phrase": "warm window", "file": "docs/x.md"},
                   {"phrase": "grace lap", "file": "a.go"}],
        "headings": [{"phrase": "Honest fences", "file": "docs/x.md"},
                     {"phrase": "The model", "file": "docs/x.md"},
                     {"phrase": "What it does", "file": "docs/y.md"}],
        "heading_keys": {"honestfences", "themodel", "whatitdoes"},
        "doc_tokens": {"warmwindow", "gateway", "honest", "fences"},
        "in_code": lambda tok: tok in code_tokens or tok == "gateway",
        "in_doc_text": lambda p: p.strip().lower() in {"warm window", "grace lap"},
        "count_doc_files": lambda p: {"honest fences": 4}.get(p.strip().lower(), 0),
        "exists": lambda p: p in {"docs/g.md"},
    }
    t.update(over)
    return t


# --- pure helpers ----------------------------------------------------------

def test_grade_letter() -> None:
    assert ie.grade_letter(95) == "A" and ie.grade_letter(82) == "B"
    assert ie.grade_letter(42) == "F"


def test_norm_token_collapses_variants() -> None:
    assert ie.norm_token("warm window") == "warmwindow"
    assert ie.norm_token("Warm_Window") == "warmwindow"


def test_concept_phrase_strips_articles_and_drops_structure() -> None:
    assert ie._concept_phrase("The honest fence") == "honest fence"
    assert ie._concept_phrase("The model") == ""            # one word after article
    assert ie._concept_phrase("What it does") == ""         # structural stopwords
    assert ie._concept_phrase("Run it") == ""
    assert ie._concept_phrase("Trust boundary") == "Trust boundary"
    assert ie._concept_phrase("Section 2.3") == ""          # non-alphabetic word


def test_row_keys_claims_names_and_raw_evidence() -> None:
    keys = ie.row_keys(row(evidence="0.8", evidence_kind="literal",
                           aliases=["hot lap"]))
    assert "0.8" in keys            # raw literal claimed verbatim
    assert "warmwindow" in keys     # canonical + named_symbol normalize
    assert "hotlap" in keys


# --- expected verdict ladder -------------------------------------------------

def test_expected_verdict_ladder() -> None:
    t = tree()
    assert ie.expected_verdict(row(), t)[0] == "explicit"
    assert ie.expected_verdict(row(doc_anchor=""), t)[0] == "named-code"
    assert ie.expected_verdict(row(named_symbol=""), t)[0] == "named-doc"
    assert ie.expected_verdict(row(named_symbol="", doc_anchor=""), t)[0] == "hinted"
    assert ie.expected_verdict(row(named_symbol="", doc_anchor="", proposed_name=""), t)[0] == "latent"
    # a definition-less anchor is not a written definition
    assert ie.expected_verdict(row(named_symbol="", definition=""), t)[0] == "hinted"
    # a dangling named_symbol does not count as named
    assert ie.expected_verdict(row(named_symbol="fictional", doc_anchor=""), t)[0] == "hinted"


# --- per-KPI defect triggers -------------------------------------------------

def test_well_formed_catches_missing_and_bad_enum_and_dups() -> None:
    k = ie.kpi_well_formed([row(), row()])  # duplicate id r1
    assert any("duplicate id" in d for d in k["defects"])
    k2 = ie.kpi_well_formed([row(signal="nonsense")])
    assert any("signal" in d for d in k2["defects"])
    k3 = ie.kpi_well_formed([row(verdict="amazing")])
    assert any("verdict" in d for d in k3["defects"])
    k4 = ie.kpi_well_formed([{"id": "x"}])  # missing fields
    assert len(k4["defects"]) > 5


def test_evidenced_refuses_fictional_signals() -> None:
    t = tree()
    assert ie.kpi_evidenced([row(named_symbol="", evidence="no such phrase")], t)["defects"]
    assert ie.kpi_evidenced([row()], t)["defects"] == []
    # literal evidence must be a real repeated literal
    bad = row(named_symbol="", evidence="9.99", evidence_kind="literal")
    assert ie.kpi_evidenced([bad], t)["defects"]
    ok = row(named_symbol="", evidence="0.8", evidence_kind="literal")
    assert ie.kpi_evidenced([ok], t)["defects"] == []
    # a resolving named_symbol re-grounds a row whose raw signal was retired
    renamed = row(evidence="9.99", evidence_kind="literal")  # named_symbol resolves
    assert ie.kpi_evidenced([renamed], t)["defects"] == []
    # heading evidence checks the heading key set
    head = row(named_symbol="", evidence="Honest fences", evidence_kind="heading")
    assert ie.kpi_evidenced([head], t)["defects"] == []


def test_named_resolves_catches_dangling_claim() -> None:
    t = tree()
    k = ie.kpi_named_resolves([row(named_symbol="fictionalsymbol")], t["in_code"])
    assert k["defects"] and "does not resolve" in k["defects"][0]
    assert ie.kpi_named_resolves([row()], t["in_code"])["defects"] == []
    assert ie.kpi_named_resolves([row(named_symbol="")], t["in_code"])["defects"] == []


def test_anchored_catches_missing_and_dangling() -> None:
    t = tree()
    k = ie.kpi_anchored([row(doc_anchor="")], t["exists"])  # explicit needs an anchor
    assert k["defects"] and "no doc_anchor" in k["defects"][0]
    k2 = ie.kpi_anchored([row(doc_anchor="docs/missing.md")], t["exists"])
    assert k2["defects"] and "does not exist" in k2["defects"][0]
    assert ie.kpi_anchored([row()], t["exists"])["defects"] == []


def test_naming_planned_demands_a_plan() -> None:
    t = tree()
    unplanned = row(named_symbol="", doc_anchor="", proposed_name="", verdict="latent")
    k = ie.kpi_naming_planned([unplanned], t)
    assert k["defects"] and "no proposed_name" in k["defects"][0]
    planned = row(named_symbol="", doc_anchor="", verdict="hinted")
    assert ie.kpi_naming_planned([planned], t)["defects"] == []
    assert ie.kpi_naming_planned([row()], t)["defects"] == []  # explicit needs no plan


def test_explicitness_consistent_catches_overclaim() -> None:
    t = tree()
    over = row(named_symbol="", doc_anchor="", verdict="explicit")
    k = ie.kpi_explicitness_consistent([over], t)
    assert k["defects"] and "claims 'explicit'" in k["defects"][0]
    assert ie.kpi_explicitness_consistent([row()], t)["defects"] == []


def test_name_quality_soft_nudges() -> None:
    k = ie.kpi_name_quality_soft([row(proposed_name="the thing that keeps entries warm")])
    assert k["soft"] and k["defects"] == []
    k2 = ie.kpi_name_quality_soft([row(proposed_name="warm window",
                                       evidence="warm window", signal="hinted")])
    assert any("restates" in s for s in k2["soft"])


# --- detectors ----------------------------------------------------------------

def test_detector_hinted_dedupes_by_phrase() -> None:
    sigs = ie.discover_signals(tree(), None)
    hinted = [s for s in sigs if s["kind"] == "hinted"]
    assert {s["key"] for s in hinted} == {"warmwindow", "gracelap"}


def test_detector_latent_literal_applies_floor() -> None:
    sigs = ie.discover_signals(tree(), None)
    lits = {s["key"] for s in sigs if s["kind"] == "latent-literal"}
    assert "300" in lits and "0.8" in lits
    # trivial idiom floor: a mode/status literal never enters the universe
    t2 = tree(literal_files={"0644": {"a.go", "b.go", "c.go", "d.go"}})
    assert not [s for s in ie.discover_signals(t2, None) if s["kind"] == "latent-literal"]
    # min_files threshold from meta overrides
    sigs3 = ie.discover_signals(tree(), {"latent_literal": {"min_files": 7}})
    assert "300" not in {s["key"] for s in sigs3 if s["kind"] == "latent-literal"}


def test_detector_code_only_needs_declaration_and_no_doc_mention() -> None:
    sigs = ie.discover_signals(tree(), {"code_only": {"min_files": 10, "min_len": 8}})
    code = {s["key"] for s in sigs if s["kind"] == "code-only"}
    assert "budgettokens" in code       # declared, 12 files, zero doc mentions
    # mentioned in docs -> not code-only
    t2 = tree(doc_tokens={"budgettokens"})
    sigs2 = ie.discover_signals(t2, {"code_only": {"min_files": 10, "min_len": 8}})
    assert "budgettokens" not in {s["key"] for s in sigs2 if s["kind"] == "code-only"}
    # used widely but never DECLARED in the repo (stdlib idiom) -> not a concept
    t3 = tree(decl_tokens=set())
    sigs3 = ie.discover_signals(t3, {"code_only": {"min_files": 10, "min_len": 8}})
    assert not [s for s in sigs3 if s["kind"] == "code-only"]


def test_detector_doc_only_keeps_concepts_drops_structure() -> None:
    sigs = ie.discover_signals(tree(), {"doc_only": {"min_doc_files": 3}})
    doc = {s["key"] for s in sigs if s["kind"] == "doc-only"}
    assert doc == {"honestfences"}      # "The model" / "What it does" are structure
    # a heading with a code identifier behind it is already explicit
    t2 = tree(sym_files={**tree()["sym_files"], "honestfences": {"a.go"}})
    sigs2 = ie.discover_signals(t2, {"doc_only": {"min_doc_files": 3}})
    assert "honestfences" not in {s["key"] for s in sigs2 if s["kind"] == "doc-only"}


def test_detector_ignore_lists_prune_noise() -> None:
    sigs = ie.discover_signals(tree(), {"hinted": {"ignore": ["grace lap"]},
                                        "latent_literal": {"ignore": ["300"]}})
    keys = {s["key"] for s in sigs}
    assert "gracelap" not in keys and "300" not in keys


# --- coverage fold -------------------------------------------------------------

def test_coverage_exact_key_only() -> None:
    sigs = ie.discover_signals(tree(), None)
    # one row claiming the warm-window phrase covers exactly that signal
    cov = ie.coverage_report(sigs, [row()])
    assert cov["covered"] == 1
    assert cov["coverage_debt"] == cov["discovered"] - 1
    assert all(u["key"] != "warmwindow" for u in cov["uncovered"])
    # a literal row must claim the literal verbatim
    lit_row = row(id="r2", canonical="grace period", evidence="300",
                  evidence_kind="literal", named_symbol="", doc_anchor="",
                  verdict="hinted")
    cov2 = ie.coverage_report(sigs, [row(), lit_row])
    assert cov2["covered"] == 2


# --- disk shell + composite -----------------------------------------------------

def test_load_data_dir_merges_rows() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "_meta.json").write_text(json.dumps(
            {"meta": {"as_of": "2026-07-03"}, "detectors": {"hinted": {"ignore": []}}}),
            encoding="utf-8")
        (d / "rows-a.json").write_text(json.dumps({"rows": [row()]}), encoding="utf-8")
        (d / "rows-b.json").write_text(json.dumps({"rows": [row(id="r2")]}), encoding="utf-8")
        data, err = ie.load_data(d)
        assert err == "" and len(data["rows"]) == 2
        assert data["rows"][0]["_source_file"] == "rows-a.json"


def test_build_payload_folds_and_flags_error() -> None:
    p = ie.build_payload(workspace="w", data=None, tree=tree(), error="boom")
    assert p["verdict"] == "AUDIT_ERROR" and not p["ok"]
    data = {"meta": {}, "detectors": {}, "rows": [row()]}
    p2 = ie.build_payload(workspace="w", data=data, tree=tree())
    c = p2["corpus"]
    assert c["implicitness_debt"] == c["naming_defects"] + c["coverage_debt"]
    assert c["naming_defects"] == 0            # the explicit row is clean
    assert c["coverage_debt"] > 0              # discovered signals unpositioned
    assert p2["finding"] == "coverage_debt"
    # renderers do not crash
    for fn in (ie.render, ie.render_chart, ie.render_critical, ie.render_gaps):
        assert isinstance(fn(p2), str)
    assert isinstance(ie.render_compare(p2, p2), str)
    assert "README.md" in ie.render_doc_folder(p2)


# --- live smoke over the real committed catalog ---------------------------------

def test_live_real_catalog_is_clean_with_honest_coverage_debt() -> None:
    root = ie.repo_root()
    payload = ie.collect(root)
    assert payload.get("verdict") != "AUDIT_ERROR", payload.get("reason")
    c = payload["corpus"]
    assert c["rows"] >= 10, "the seed catalog positions a real starting set"
    assert c["naming_defects"] == 0, \
        f"positioned rows must be clean: {[k['defects'] for k in payload['kpis'] if k['defects']]}"
    assert c["coverage_debt"] > 0, "birth state: the implicit space is only partly mapped"
    assert 15 <= c["score"] <= 60, f"score {c['score']} outside the honest birth band"
    assert c["coverage"]["discovered"] >= 50, "detectors must find a real universe"


def _main() -> int:
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print(f"ERROR {name}: {exc!r}")
    print(f"{'PASS' if fails == 0 else 'FAIL'} implicit_explicit_scorecard_test")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_main())
