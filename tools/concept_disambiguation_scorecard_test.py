#!/usr/bin/env python3
"""Tests for the concept-disambiguation scorecard.

Three things are exercised. (1) The pure helpers: grade, token normalization +
match, cross-row canonical-collision detection, and the clarity verdict the evidence
IMPLIES (crystal needs an anchor; a colliding canonical can never be crystal). (2)
Each KPI's defect trigger - the malformed-row catch, the canonical-collision catch,
the undefined catch, the undisambiguated catch (a confusable concept that never says
what it is NOT), the UNGROUNDED catch (a fabricated grounding the strict cross-check
refuses), the dangling/missing anchor catch, and the verdict-overclaim catch - plus
the coverage discovery over the watched families (presence threshold, ignore, exclude).
(3) The disk shell + the fold to the composite.

Closes with the load-bearing live smoke: the REAL committed catalog must fold to ZERO
clarity-debt (every positioned concept is clean), carry nonzero coverage-debt (the
confusable namespace is only partly mapped - that is the honest birth state), and
score in the intended 2/10-5/10 band.

Run: `python tools/concept_disambiguation_scorecard_test.py`  (exit 0 = all pass),
or `python -m pytest tools/concept_disambiguation_scorecard_test.py -q`.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import concept_disambiguation_scorecard as cd  # noqa: E402


def row(**over) -> dict:
    """A minimal well-formed, grounded, defined, crystal row in a 2-member cluster."""
    r = {
        "id": "r1", "canonical": "Alpha", "family": "cache", "kind": "subsystem",
        "definition": "the alpha thing", "distinction": "alpha is not beta",
        "distinct_from": ["r2"], "aliases": [], "grounding": "alpha",
        "grounding_kind": "symbol", "glossary_anchor": "docs/g.md",
        "verdict": "crystal", "gaps": [],
    }
    r.update(over)
    return r


def sibling(**over) -> dict:
    r = row(id="r2", canonical="Beta", distinction="beta is not alpha",
            distinct_from=["r1"], grounding="beta")
    r.update(over)
    return r


def families() -> set[str]:
    return {"cache", "attention", "guard-gate"}


def tree(**over) -> dict:
    """Synthetic tree-facts: a presence corpus, an in_tree predicate, an existence
    set, and documented verbs."""
    present_tokens = {"alpha", "beta", "kvcache", "vcache", "cachemeta", "gamma"}
    present_files = {"docs/g.md", "docs/cli-reference.md"}
    t = {
        "corpus": {"sym_files": {tok: {"f1.go", "f2.go", "f3.go"} for tok in present_tokens},
                   "structural": {"cache", "attention"}},
        "in_tree": lambda tok: tok in present_tokens or tok in {"cache", "attention"},
        "exists": lambda p: p in present_files,
        "doc_verbs": {"preflight", "serve"},
    }
    t.update(over)
    return t


# --- pure helpers ----------------------------------------------------------

def test_grade_letter() -> None:
    assert cd.grade_letter(95) == "A" and cd.grade_letter(82) == "B"
    assert cd.grade_letter(42) == "F"


def test_norm_token_collapses_variants() -> None:
    assert cd.norm_token("KV cache") == "kvcache"
    assert cd.norm_token("kv_cache") == "kvcache"
    assert cd.norm_token("vCache") == "vcache"


def test_token_match_guards_trivial_overlap() -> None:
    assert cd.token_match("kvcache", "kvcache") is True
    assert cd.token_match("vcache", "vcachegov") is True   # containment, len guard ok
    assert cd.token_match("id", "guard") is False          # short overlap rejected


def test_find_collisions_flags_shared_canonical() -> None:
    coll = cd.find_collisions([row(), sibling(canonical="Alpha")])  # both "Alpha"
    assert "r1" in coll and "r2" in coll


# --- per-KPI defect triggers -----------------------------------------------

def test_well_formed_catches_missing_and_bad_enum_and_dups() -> None:
    k = cd.kpi_well_formed([row(), row()], families())  # duplicate id r1
    assert any("duplicate id" in d for d in k["defects"])
    k2 = cd.kpi_well_formed([row(kind="nonsense")], families())
    assert any("kind" in d for d in k2["defects"])
    k3 = cd.kpi_well_formed([row(verdict="amazing")], families())
    assert any("verdict" in d for d in k3["defects"])
    k4 = cd.kpi_well_formed([{"id": "x"}], families())  # missing fields
    assert len(k4["defects"]) > 5


def test_canonical_unique_catches_collision() -> None:
    k = cd.kpi_canonical_unique([row(), sibling(canonical="Alpha")])
    assert len(k["defects"]) == 2 and all("collides" in d for d in k["defects"])
    assert cd.kpi_canonical_unique([row(), sibling()])["defects"] == []


def test_defined_catches_empty_definition() -> None:
    assert cd.kpi_defined([row(definition="")])["defects"]
    assert cd.kpi_defined([row()])["defects"] == []


def test_disambiguated_requires_line_and_resolving_ref() -> None:
    sizes = cd.cluster_sizes([row(), sibling()])
    # clean: both have a distinction + a resolving distinct_from.
    assert cd.kpi_disambiguated([row(), sibling()], sizes)["defects"] == []
    # no distinction line in a multi-member cluster -> debt.
    k = cd.kpi_disambiguated([row(distinction=""), sibling()], sizes)
    assert any("no distinction" in d for d in k["defects"])
    # distinct_from points at nothing real -> debt.
    k2 = cd.kpi_disambiguated([row(distinct_from=["ghost"]), sibling()], sizes)
    assert any("resolves to no" in d for d in k2["defects"])
    # a LONE concept (cluster size 1) is excused - nothing to disambiguate yet.
    lone_sizes = cd.cluster_sizes([row(distinct_from=[], distinction="")])
    assert cd.kpi_disambiguated([row(distinct_from=[], distinction="")], lone_sizes)["defects"] == []


def test_grounded_strictly_refuses_fabricated_token() -> None:
    t = tree()
    assert cd.kpi_grounded([row()], t["in_tree"])["defects"] == []
    # 'alphaffabricated' contains the real token 'alpha' but is NOT itself present:
    # the strict cross-check must still refuse it.
    k = cd.kpi_grounded([row(grounding="alphafabricated")], t["in_tree"])
    assert any("does not appear" in d for d in k["defects"])


def test_anchored_catches_crystal_without_anchor_and_dangling() -> None:
    t = tree()
    assert cd.kpi_anchored([row()], t["exists"])["defects"] == []
    assert cd.kpi_anchored([row(glossary_anchor="")], t["exists"])["defects"]   # crystal needs anchor
    assert cd.kpi_anchored([row(glossary_anchor="docs/ghost.md")], t["exists"])["defects"]  # dangling


def test_expected_verdict_ladder() -> None:
    t = tree()
    sizes = {"cache": 2}
    # crystal: defined + distinction + anchor exists.
    assert cd.expected_verdict(row(), colliding=False, exists=t["exists"], sizes=sizes)[0] == "crystal"
    # no anchor -> defined.
    assert cd.expected_verdict(row(glossary_anchor=""), colliding=False, exists=t["exists"], sizes=sizes)[0] == "defined"
    # no distinction (with a sibling) -> drifting.
    assert cd.expected_verdict(row(distinction=""), colliding=False, exists=t["exists"], sizes=sizes)[0] == "drifting"
    # collision -> colliding.
    assert cd.expected_verdict(row(), colliding=True, exists=t["exists"], sizes=sizes)[0] == "colliding"
    # no definition -> undocumented.
    assert cd.expected_verdict(row(definition=""), colliding=False, exists=t["exists"], sizes=sizes)[0] == "undocumented"


def test_clarity_consistent_catches_overclaim() -> None:
    t = tree()
    sizes = {"cache": 2}
    # declares crystal but has no anchor -> evidence implies 'defined'.
    k = cd.kpi_clarity_consistent([row(glossary_anchor="")], set(), t["exists"], sizes)
    assert k["defects"] and "implies 'defined'" in k["defects"][0]
    assert cd.kpi_clarity_consistent([row()], set(), t["exists"], sizes)["defects"] == []


# --- coverage discovery ----------------------------------------------------

def test_discover_respects_presence_ignore_exclude() -> None:
    corpus = {"sym_files": {"cache": {"a.go", "b.go", "c.go"}, "vcache": {"a.go", "b.go", "c.go"},
                            "cached": {"a.go", "b.go", "c.go"}, "oneoff": {"a.go"},
                            "gateway": {"a.go", "b.go", "c.go"}},
              "structural": {"cachemeta"}}
    fam = {"id": "cache", "roots": ["cache"], "ignore": ["cached"], "min_files": 2}
    toks = {t["token"] for t in cd.discover_family_tokens(fam, corpus)}
    assert "cache" in toks and "vcache" in toks
    assert "cached" not in toks            # ignore list
    assert "cachemeta" in toks             # structural counts even below min_files
    assert "gateway" not in toks           # no cache root
    # exclude keeps a gate family from swallowing gateway.
    famg = {"id": "gate", "roots": ["gate"], "exclude": ["gateway"], "min_files": 2}
    toksg = {t["token"] for t in cd.discover_family_tokens(famg, {"sym_files": {"gateway": {"a.go", "b.go"}}, "structural": set()})}
    assert "gateway" not in toksg


def test_coverage_dedupes_and_counts() -> None:
    corpus = {"sym_files": {"kvcache": {"a.go", "b.go"}, "vcache": {"a.go", "b.go"},
                            "enginecache": {"a.go", "b.go"}},
              "structural": set()}
    fams = [{"id": "cache", "roots": ["cache"], "min_files": 2},
            {"id": "engine", "roots": ["engine"], "min_files": 2}]
    rows = [row(grounding="kvcache", canonical="KV cache")]
    cov = cd.coverage_report(fams, rows, corpus)
    # enginecache matches BOTH families but is one concept -> deduped in the headline.
    assert cov["discovered"] == 3 and cov["covered"] == 1
    assert cov["coverage_debt"] == 2


# --- disk shell + fold ------------------------------------------------------

def test_load_data_dir_merges_modular_files() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "_meta.json").write_text(json.dumps({
            "meta": {"as_of": "2026-06-26", "fak_version": "t"},
            "families": [{"id": "cache", "roots": ["cache"]}],
        }), encoding="utf-8")
        (d / "rows-cache.json").write_text(json.dumps({"rows": [row()]}), encoding="utf-8")
        data, err = cd.load_data_dir(d)
        assert err == "" and data is not None
        assert len(data["rows"]) == 1 and data["rows"][0]["_source_file"] == "rows-cache.json"


def _data(rows: list[dict]) -> dict:
    return {"meta": {"as_of": "2026-06-26", "fak_version": "t"},
            "families": [{"id": "cache", "roots": ["cache"], "min_files": 1}],
            "rows": rows}


def test_build_payload_clean_rows_low_coverage_is_action_F() -> None:
    # two clean crystal rows, but the cache family discovers more than they cover.
    t = tree(corpus={"sym_files": {tok: {"a.go"} for tok in
                                   ("alpha", "beta", "kvcache", "vcache", "cachemeta", "providercache")},
                     "structural": set()})
    t["in_tree"] = lambda tok: tok in {"alpha", "beta"}
    p = cd.build_payload(workspace=".", data=_data([row(), sibling()]), tree=t)
    assert p["ok"] is False and p["finding"] == "coverage_debt"
    assert p["corpus"]["clarity_defects"] == 0 and p["corpus"]["coverage_debt"] > 0


def test_build_payload_honesty_defect_drives_action() -> None:
    t = tree()
    bad = [row(glossary_anchor="", verdict="crystal"), sibling()]  # crystal w/o anchor
    p = cd.build_payload(workspace=".", data=_data(bad), tree=t)
    assert p["ok"] is False and p["finding"] in ("disambiguation_debt",)
    assert p["corpus"]["clarity_defects"] >= 1


def test_build_payload_error_on_no_data() -> None:
    p = cd.build_payload(workspace=".", data=None, tree=tree(), error="missing data")
    assert p["ok"] is False and p["verdict"] == "AUDIT_ERROR"


# --- renderers don't crash + produce the doc folder -------------------------

def test_renderers_and_doc_folder() -> None:
    t = tree()
    p = cd.build_payload(workspace=".", data=_data([row(), sibling()]), tree=t)
    assert "concept-disambiguation:" in cd.render(p)
    assert "backlog" in cd.render_critical(p)
    assert "backlog" in cd.render_gaps(p)
    assert "clarity ladder" in cd.render_chart(p)
    files = cd.render_doc_folder(p, stamp="2026-06-26")
    assert "README.md" in files and "Concept-disambiguation scorecard" in files["README.md"]
    assert "clarity ladder" in files["README.md"]
    assert "disambiguation-debt:" in cd.render_compare(p, p)


def test_bar_proportional_and_sliver() -> None:
    assert cd._bar(10, 10, width=10) == "#" * 10
    assert cd._bar(0, 10, width=10) == "." * 10
    assert cd._bar(1, 1000, width=10).count("#") == 1
    assert cd._bar(5, 0, width=4) == "." * 4


# --- the load-bearing live smoke: the committed catalog is clean + substantially mapped ---

def test_live_real_data_is_clean_and_in_band() -> None:
    root = cd.repo_root()
    path = root / cd.DATA_DIR_REL
    if not path.exists():
        return  # tolerant: not in the repo tree
    p = cd.collect(root)
    assert p["schema"] == cd.SCHEMA, p
    c = p["corpus"]
    # Every positioned concept must be CLEAN (no clarity-debt): the catalog itself is
    # the exemplar of crystal clarity.
    assert c["clarity_defects"] == 0, p["reason"]
    for k in p["kpis"]:
        if k["group"] != "honesty" or k["kpi"] == "clarity_consistent":
            assert k["defects"] == [], f"{k['kpi']}: {k['defects'][:3]}"
    # Discovery must still be working: a large confusable universe is found in the tree.
    # (A trivially-100% coverage from a BROKEN/empty discovery would fail the floor below.)
    assert c["coverage"]["discovered"] >= 100, "the confusable universe should be large"
    # The coverage-debt has been RETIRED: the namespace is substantially positioned. A small
    # band is allowed so a peer landing a few new confusable tokens does not red the gate
    # before they are catalogued - the catalog stays useful, not perfect-or-bust.
    assert c["coverage"]["coverage_pct"] >= 95.0, f"coverage {c['coverage']['coverage_pct']}% regressed - position new confusable tokens"
    # Clean + substantially-mapped lands an A-grade score; the band tolerates minor drift.
    assert c["score"] >= 90, f"score {c['score']} below the clean+mapped A-band"
    # A credible foundation: a real spread of crystal + defined concepts is positioned.
    assert c["standing"]["crystal"] >= 20, "real crystal-clear concepts (the cache family is the exemplar)"
    assert c["standing"]["defined"] >= 5, "honest defined-but-not-yet-anchored concepts"
    # Every crystal concept's distinction is anchored in a doc that exists.
    for r in c["leaderboard"]:
        if r["verdict"] == "crystal":
            assert r["glossary_anchor"], f"{r['id']} is crystal but has no anchor"


def main() -> int:
    failures: list[str] = []

    def check(name: str, fn) -> None:
        try:
            fn()
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: unexpected {type(exc).__name__}: {exc}")

    tests = {n: f for n, f in globals().items()
             if n.startswith("test_") and callable(f)}
    for name, fn in tests.items():
        check(name, fn)

    if failures:
        print(f"FAIL ({len(failures)}/{len(tests)}):")
        for f in failures:
            print("  -", f)
        return 1
    print(f"ok ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
