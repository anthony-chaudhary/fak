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
import random
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


# --- hierarchical roll-up (the namespace at its abstraction heads) -----------

def _head_and_child(child_verdict: str = "defined"):
    """A 2-concept abstraction: a crystal head `p` over a child `c`. The child's own
    verdict is the knob - `defined` makes the head OVERCLAIM under weakest-link."""
    head = row(id="p", canonical="Parent", distinct_from=["c"], distinction="p not c",
               verdict="crystal")
    kid = row(id="c", canonical="Child", distinct_from=["p"], distinction="c not p",
              grounding="beta", glossary_anchor="", verdict=child_verdict, parent="p")
    return head, kid


def test_resolve_parents_keeps_resolvable_and_flags_bad() -> None:
    rows = [row(id="p"), sibling(id="c", parent="p"),
            row(id="orphan", canonical="Orphan", parent="ghost"),
            row(id="loop", canonical="Loop", parent="loop")]
    parent, children, soft = cd.resolve_parents(rows)
    assert parent.get("c") == "p" and children.get("p") == ["c"]
    assert "orphan" not in parent and "loop" not in parent  # bad edges are not kept
    assert any("resolves to no catalog id" in s for s in soft)   # orphan -> ghost
    assert any("points at itself" in s for s in soft)            # loop -> loop


def test_roll_up_weakest_link_flags_overclaim() -> None:
    head, kid = _head_and_child("defined")
    ru = cd.roll_up([head, kid], {"c": 3, "p": 0})
    assert ru["heads"] == 1 and ru["roots"] == 1 and ru["max_depth"] == 1
    a = ru["abstractions"][0]
    assert a["id"] == "p"
    # the head DECLARES crystal but the subtree only rolls up to its foggiest member.
    assert a["declared_verdict"] == "crystal" and a["rolled_verdict"] == "defined"
    assert a["overclaim"] is True
    assert a["subtree_size"] == 2 and a["subtree_debt"] == 3
    assert a["weakest"] == {"id": "c", "verdict": "defined"}
    assert ru["overclaims"] and "rolls up to 'defined'" in ru["overclaims"][0]


def test_roll_up_clean_subtree_does_not_overclaim() -> None:
    head, kid = _head_and_child("crystal")
    ru = cd.roll_up([head, kid], {})
    a = ru["abstractions"][0]
    assert a["rolled_verdict"] == "crystal" and a["overclaim"] is False
    assert ru["overclaims"] == []


def test_roll_up_never_rolls_up_clearer_than_declared() -> None:
    # Fold invariant: a subtree can only ever DRAG a head down (weakest-link), never lift
    # it above what it declares. rolled_rank >= declared_rank for every head.
    head, kid = _head_and_child("colliding")
    ru = cd.roll_up([head, kid], {})
    a = ru["abstractions"][0]
    assert cd.VERDICT_RANK[a["rolled_verdict"]] >= cd.VERDICT_RANK[a["declared_verdict"]]
    assert a["rolled_verdict"] == "colliding"  # the foggiest descendant wins


def test_roll_up_is_cycle_safe() -> None:
    # A malformed a<->b cycle must not hang the fold; the subtree terminates via a seen-set.
    a = row(id="a", canonical="A", parent="b", verdict="crystal")
    b = sibling(id="b", canonical="B", parent="a", verdict="defined")
    ru = cd.roll_up([a, b], {})
    assert ru["heads"] >= 1
    assert any(x["rolled_verdict"] == "defined" for x in ru["abstractions"])


def test_kpi_hierarchy_soft_is_advisory_but_flags_overclaim() -> None:
    head, kid = _head_and_child("defined")
    rows = [head, kid, row(id="orphan", canonical="Orphan", parent="ghost")]
    k = cd.kpi_hierarchy_soft(rows)
    assert k["group"] == "honesty" and k["defects"] == []   # never hard debt - hierarchy is optional
    assert any("resolves to no catalog id" in s for s in k["soft"])
    assert any("head declares 'crystal'" in s for s in k["soft"])
    assert k["score"] < 100                                 # soft signals dent the advisory score


def test_render_rollup_and_doc_section() -> None:
    head, kid = _head_and_child("defined")
    p = cd.build_payload(workspace=".", data=_data([head, kid]), tree=tree())
    assert p["corpus"]["clarity_defects"] == 0            # the fixture is clean per-row...
    assert p["corpus"]["rollup"]["overclaims"]            # ...yet the ABSTRACTION overclaims
    txt = cd.render_rollup(p)
    assert "roll-up" in txt and "WEAKEST-LINK" in txt and "overclaim" in txt
    assert "weakest-link" in cd.render(p)                 # headline carries a roll-up line
    files = cd.render_doc_folder(p, stamp="2026-06-26")
    assert "Concept roll-up" in files["README.md"]


# --- separation: is each concept disambiguated FROM THE OTHERS? -------------

def test_bare_name_strips_the_catalog_gloss() -> None:
    # The gloss is the catalog explaining itself; the tree only ever says `Decision`.
    assert cd.bare_name("Decision (kernel)") == "Decision"
    assert cd.bare_name("Decision") == "Decision"
    assert cd.bare_name("fak_gateway_cache_ttl_upgrade_total") == "fak_gateway_cache_ttl_upgrade_total"
    assert cd.bare_name("Plan (memq) (nested)") == "Plan (memq)"   # only the trailing gloss


def test_split_words_handles_camel_snake_and_acronyms() -> None:
    assert cd.split_words("SessionRef") == ["session", "ref"]
    assert cd.split_words("ffn_gate") == ["ffn", "gate"]
    assert cd.split_words("KVLayout") == ["kv", "layout"]          # acronym run stays whole
    assert cd.split_words("q4Kernel") == ["q4", "kernel"]
    assert cd.split_words("") == []


def test_edit_distance_within_bands_at_the_cap() -> None:
    assert cd.edit_distance_within("sessionref", "sessionrow", 2) == 2
    assert cd.edit_distance_within("kernel", "kernel", 2) == 0
    # Over the cap the banded walk short-circuits to cap+1 rather than the true distance.
    assert cd.edit_distance_within("alpha", "omegaomega", 2) == 3


def test_shared_affix_takes_the_longer_run() -> None:
    assert cd.shared_affix("sessionref", "sessionrow") == 8        # head run
    assert cd.shared_affix("kvlayout", "layout") == 6              # tail run
    assert cd.shared_affix("alpha", "omega") == 1                  # 'a' tail only


def _pair_rows() -> list[dict]:
    """Two of every confusable kind, plus a control pair that must NOT be flagged."""
    return [
        # homonym: identical once the gloss is stripped.
        row(id="d1", canonical="Decision (kernel)", grounding="alpha", distinct_from=[]),
        row(id="d2", canonical="Decision (scheduler)", grounding="beta", distinct_from=[]),
        # permuted: same words, different order.
        row(id="p1", canonical="witnessPath", grounding="alpha", distinct_from=[]),
        row(id="p2", canonical="PathWitness", grounding="beta", distinct_from=[]),
        # near: 2 edits apart sharing an 8-character head.
        row(id="n1", canonical="SessionRef", grounding="alpha", distinct_from=[]),
        row(id="n2", canonical="SessionRow", grounding="beta", distinct_from=[]),
        # control: same trailing word, but only a 4-character shared run - a bare family
        # root is not evidence that a reader confuses them.
        row(id="c1", canonical="AlphaGate", grounding="alpha", distinct_from=[]),
        row(id="c2", canonical="OmegaGate", grounding="beta", distinct_from=[]),
    ]


def test_confusable_pairs_finds_all_three_kinds_and_spares_the_control() -> None:
    pairs = cd.confusable_pairs(_pair_rows())
    got = {(p["a"], p["b"]): p["kind"] for p in pairs}
    assert got.get(("d1", "d2")) == "homonym"
    assert got.get(("p1", "p2")) == "permuted"
    assert got.get(("n1", "n2")) == "near"
    assert ("c1", "c2") not in got, got            # shared affix below the floor
    # Discovered from the names themselves: no row DECLARED any of these.
    assert all(not r["distinct_from"] for r in _pair_rows())
    # Deterministic ordering, strongest kind first.
    assert [p["kind"] for p in pairs] == sorted(
        (p["kind"] for p in pairs), key=lambda k: cd.PAIR_KINDS.index(k))


def test_confusable_pairs_takes_the_strongest_kind_when_several_apply() -> None:
    # `Cache Key` / `Key Cache` are BOTH permuted and (normalized) near; homonym-first
    # ranking means the strongest applicable kind labels the pair.
    rows = [row(id="a", canonical="Cache Key", grounding="alpha", distinct_from=[]),
            row(id="b", canonical="Key Cache", grounding="beta", distinct_from=[])]
    pairs = cd.confusable_pairs(rows)
    assert len(pairs) == 1 and pairs[0]["kind"] == "permuted"


def test_separation_edges_resolve_by_id_canonical_and_index() -> None:
    rows = [row(id="r1", canonical="Alpha", aliases=["alpha-thing"], distinct_from=["Beta"]),
            sibling(id="r2", canonical="Beta", distinct_from=["alpha-thing"])]
    index = cd.build_index(rows)
    edges, unresolved = cd.separation_edges(rows, index)
    # r1 resolved by canonical name; r2 resolved through the index, by ALIAS.
    assert ("r1", "r2") in edges and ("r2", "r1") in edges
    assert unresolved == []


def test_separation_edges_report_dangling_and_ambiguous_references() -> None:
    rows = [row(id="r1", canonical="Alpha", distinct_from=["ghost"]),
            sibling(id="r2", canonical="Beta", distinct_from=["Decision"]),
            row(id="d1", canonical="Decision (kernel)", grounding="alpha", distinct_from=[]),
            row(id="d2", canonical="Decision (scheduler)", grounding="beta", distinct_from=[])]
    _, unresolved = cd.separation_edges(rows, cd.build_index(rows))
    assert any("dangling boundary" in u and "ghost" in u for u in unresolved)
    # A reference that names TWO concepts is its own defect: it does not say which.
    assert any("names 2 concepts at once" in u for u in unresolved)
    assert cd.kpi_reference_resolves(unresolved)["defects"]
    assert cd.kpi_reference_resolves([])["defects"] == []


def test_separation_report_grades_mutual_one_sided_and_undrawn() -> None:
    rows = _pair_rows()
    rows[0]["distinct_from"] = ["d2"]           # homonym drawn one way only
    rows[2]["distinct_from"] = ["p2"]
    rows[3]["distinct_from"] = ["p1"]           # permuted drawn both ways
    sep = cd.separation_report(rows, cd.separation_edges(rows, cd.build_index(rows))[0],
                               cd.confusable_pairs(rows))
    state = {(p["a"], p["b"]): p["state"] for p in sep["pairs"]}
    assert state[("d1", "d2")] == "one_sided"
    assert state[("p1", "p2")] == "mutual"
    assert state[("n1", "n2")] == "undrawn"
    assert sep["counts"] == {"mutual": 1, "one_sided": 1, "undrawn": 1}
    # Being clean per-row cannot clear these: only the boundary itself does.
    assert cd.kpi_pair_separated(sep)["defects"]        # the undrawn pair
    assert cd.kpi_pair_mutual(sep)["defects"]           # the one-sided pair


def test_entangled_rung_catches_the_receiving_end_of_a_one_sided_line() -> None:
    rows = _pair_rows()
    rows[0]["distinct_from"] = ["d2"]           # d1 warns about d2; d2 says nothing back
    edges = cd.separation_edges(rows, cd.build_index(rows))[0]
    sep = cd.separation_report(rows, edges, cd.confusable_pairs(rows))
    ent = cd.entangled_rows(sep, edges)
    assert "d1" not in ent                      # it drew its own boundary
    assert "d2" in ent                          # a reader arriving HERE is still unwarned
    assert "n1" in ent and "n2" in ent          # neither side of the undrawn pair
    # The verdict ladder puts `entangled` between drifting and defined, and the row
    # cannot clear it by re-labelling itself - only by drawing the boundary.
    t = tree()
    v, why = cd.expected_verdict(row(), colliding=False, exists=t["exists"],
                                 sizes={"cache": 2}, entangled="names no boundary against d1")
    assert v == "entangled" and "d1" in why
    assert (cd.VERDICT_RANK["defined"] < cd.VERDICT_RANK["drifting"]
            < cd.VERDICT_RANK["entangled"] < cd.VERDICT_RANK["colliding"])


# --- indexing: can a reader who meets a NAME find the concept? --------------

def test_build_index_covers_canonical_alias_and_grounding_surfaces() -> None:
    rows = [row(id="r1", canonical="Alpha (the first)", aliases=["A1"], grounding="alpha")]
    ix = cd.build_index(rows)
    keys = {e["key"] for e in ix["entries"]}
    assert {"alpha", "a1"} <= keys              # bare canonical + alias + grounding
    assert ix["by_key"]["alpha"] == ["r1"]
    assert ix["ambiguous_keys"] == 0


def test_index_finds_the_lookup_ambiguity_canonical_names_hide() -> None:
    # Both canonicals are unique, yet one row claims the OTHER's name as an alias:
    # the name is genuinely ambiguous and `canonical_unique` cannot see it.
    rows = [row(id="r1", canonical="Alpha", grounding="alpha"),
            sibling(id="r2", canonical="Beta", aliases=["Alpha"], grounding="beta")]
    assert cd.kpi_canonical_unique(rows)["defects"] == []
    ix = cd.build_index(rows)
    assert ix["ambiguous_keys"] == 1
    assert sorted(ix["ambiguous"][0]["targets"]) == ["r1", "r2"]


def test_index_pairs_leave_spelling_pairs_to_the_separation_checks() -> None:
    # `Decision (kernel)` / `Decision (scheduler)` share a lookup key AND are homonyms;
    # counting one missing boundary as two defects would inflate the debt.
    rows = _pair_rows()
    ix = cd.build_index(rows)
    spelling = cd.confusable_pairs(rows)
    assert ("d1", "d2") in {(p["a"], p["b"]) for p in spelling}
    assert ("d1", "d2") not in {(p["a"], p["b"]) for p in cd.index_pairs(ix, spelling)}


def test_kpi_index_resolves_tolerates_ambiguity_that_is_separated() -> None:
    # Two concepts really are both called `Alpha` in the tree; the catalog does not get
    # to rename a Go type. The defect is not the shared name - it is the reader landing
    # on both with nothing to tell them apart.
    def rows_with(df1, df2):
        return [row(id="r1", canonical="One", grounding="alpha", distinct_from=df1),
                sibling(id="r2", canonical="Two", aliases=["One"], grounding="beta",
                        distinct_from=df2)]
    rows = rows_with([], [])
    ix = cd.build_index(rows)
    ipairs = cd.index_pairs(ix, cd.confusable_pairs(rows))
    edges = cd.separation_edges(rows, ix)[0]
    assert cd.kpi_index_resolves(ipairs, rows, edges, ix)["defects"]
    # ...and one-sided is still a fog for the reader arriving from the other side.
    rows = rows_with(["r2"], [])
    ix = cd.build_index(rows)
    edges = cd.separation_edges(rows, ix)[0]
    assert cd.kpi_index_resolves(cd.index_pairs(ix, cd.confusable_pairs(rows)),
                                 rows, edges, ix)["defects"]
    # Drawn BOTH ways, the index can honestly answer "both, and here is the difference".
    rows = rows_with(["r2"], ["r1"])
    ix = cd.build_index(rows)
    edges = cd.separation_edges(rows, ix)[0]
    assert cd.kpi_index_resolves(cd.index_pairs(ix, cd.confusable_pairs(rows)),
                                 rows, edges, ix)["defects"] == []


def test_render_pairs_index_and_the_generated_name_index() -> None:
    rows = _pair_rows()
    p = cd.build_payload(workspace=".", data=_data(rows), tree=tree())
    sp, ix = p["corpus"]["separation"], p["corpus"]["index"]
    assert sp["confusable_pairs"] == 3 and sp["undrawn"] == 3
    # All EIGHT rows are entangled, not six: the control pair's spellings stay far
    # apart, but both rows ground on the same tree token, so LOOKUP makes them twins
    # even where spelling does not.
    assert sp["entangled_concepts"] == 8
    assert ix["ambiguous_keys"] == 3 and ix["unresolved_shared_names"] > 0

    pairs_txt = cd.render_pairs(p)
    assert "pairwise separation" in pairs_txt
    assert "homonym" in pairs_txt and "permuted" in pairs_txt and "near" in pairs_txt
    idx_txt = cd.render_index(p, limit=2)
    assert "name index" in idx_txt and "more name(s)" in idx_txt

    files = cd.render_doc_folder(p, stamp="2026-07-26")
    assert set(files) == {"README.md", "INDEX.md"}
    assert "Separation - is each concept disambiguated FROM THE OTHERS?" in files["README.md"]
    assert "not to be confused with" in files["INDEX.md"].lower()
    assert "Decision (kernel)" in files["INDEX.md"]
    # The index POINTS at the scorecard rather than copying it.
    assert "README.md" in files["INDEX.md"]


def test_lookup_answers_one_name_the_way_a_reader_meets_it() -> None:
    """A reader does not read the index; they arrive holding ONE spelling."""
    p = cd.build_payload(workspace=".", data=_data(_pair_rows()), tree=tree())
    rows = cd._index_rows(p)
    # The spelling a reader met is normalized the same way the index was built, so
    # 'Session Ref', 'session_ref' and 'sessionRef' are the same question.
    for spelling in ["SessionRef", "session_ref", "session ref"]:
        hits, exact = cd.index_lookup(rows, spelling)
        assert exact and len(hits) == 1 and hits[0]["key"] == "sessionref", spelling
    # A near miss answers with the nearest keys rather than a bare "no".
    hits, exact = cd.index_lookup(rows, "session")
    assert not exact and {h["key"] for h in hits} >= {"sessionref", "sessionrow"}
    assert cd.index_lookup(rows, "nothing-like-this-at-all") == ([], False)
    # The rendered answer carries the contrast set, not just the definition.
    out = cd.render_index(p, lookup="SessionRef")
    assert "'SessionRef' -> 1 lookup name(s)" in out
    assert "not to be confused with" in out
    assert "no lookup name matches" in cd.render_index(p, lookup="nothing-like-this-at-all")


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
    # The hierarchical roll-up folds the parent forest into an honest higher-level view.
    ru = c["rollup"]
    assert ru["roots"] >= 1 and ru["forest_nodes"] >= ru["heads"], ru
    assert isinstance(ru["overclaims"], list)
    # WEAKEST-LINK invariant on real data: a roll-up can only DRAG a head down to its
    # foggiest descendant, never lift it above what its head declares.
    for a in ru["abstractions"]:
        assert cd.VERDICT_RANK[a["rolled_verdict"]] >= cd.VERDICT_RANK[a["declared_verdict"]], a
    # The coverage-debt has been RETIRED: the namespace is substantially positioned. A small
    # band is allowed so a peer landing a few new confusable tokens does not red the gate
    # before they are catalogued - the catalog stays useful, not perfect-or-bust.
    assert c["coverage"]["coverage_pct"] >= 95.0, f"coverage {c['coverage']['coverage_pct']}% regressed - position new confusable tokens"
    # Clean + substantially-mapped lands an A-grade score; the band tolerates minor drift.
    assert c["score"] >= 90, f"score {c['score']} below the clean+mapped A-band"
    # A credible foundation: a real spread of crystal + defined concepts is positioned.
    assert c["standing"]["crystal"] >= 20, "real crystal-clear concepts (the cache family is the exemplar)"
    assert c["standing"]["defined"] >= 5, "honest defined-but-not-yet-anchored concepts"
    # SEPARATION on real data: discovery is still finding confusable twins, and every one
    # of them is drawn from BOTH sides, so a reader arriving from either name is warned.
    sp = c["separation"]
    assert sp["confusable_pairs"] >= 50, "pair discovery should find a real twin population"
    assert sp["undrawn"] == 0, f"{sp['undrawn']} confusable pair(s) unseparated - see --pairs"
    assert sp["one_sided"] == 0, f"{sp['one_sided']} pair(s) drawn one way only - see --pairs"
    assert sp["mutual"] == sp["confusable_pairs"]
    assert sp["dangling_references"] == 0, "every distinct_from must name exactly one concept"
    assert c["standing"]["entangled"] == 0
    # INDEXING on real data: names really are shared (Decision is four Go types), and
    # every shared name lands on concepts that separate from each other.
    ix = c["index"]
    assert ix["keys"] > c["rows"], "the lookup surface is wider than the concept list"
    assert ix["ambiguous_keys"] > 0, "shared names are a fact of the tree, not a bug"
    assert ix["unresolved_shared_names"] == 0, "a shared name must land on a CHOICE - see --index"
    # Every crystal concept's distinction is anchored in a doc that exists.
    for r in c["leaderboard"]:
        if r["verdict"] == "crystal":
            assert r["glossary_anchor"], f"{r['id']} is crystal but has no anchor"


def test_indexed_pairs_match_serial_on_boundary_and_seeded_rows() -> None:
    rng = random.Random(9884)
    rows = [
        {"id": f"random-{i:03d}",
         "canonical": "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(12))}
        for i in range(400)
    ]
    rows.extend([
        {"id": "prefix-a", "canonical": "abcde-tail-one"},
        {"id": "prefix-b", "canonical": "abcde-tail-two"},
        {"id": "suffix-a", "canonical": "first-abcde"},
        {"id": "suffix-b", "canonical": "frost-abcde"},
        {"id": "edit-two-a", "canonical": "boundarysameaa"},
        {"id": "edit-two-b", "canonical": "boundarysamebb"},
        {"id": "duplicate-a", "canonical": "Duplicate (first gloss)"},
        {"id": "duplicate-b", "canonical": "Duplicate (second gloss)"},
        {"id": "same-token-a", "canonical": "Same token"},
        {"id": "same-token-b", "canonical": "same_token"},
    ])
    assert cd.confusable_pairs(rows) == cd.confusable_pairs(rows, indexed=False)


def test_indexed_coverage_matches_serial_on_closed_edge_cases() -> None:
    families = [
        {"id": "short", "roots": ["ab"], "min_files": 2,
         "ignore": ["abignored"], "exclude": ["abexclude"]},
        {"id": "cache", "roots": ["cache"], "min_files": 2},
        {"id": "gate", "roots": ["gate"], "min_files": 2,
         "exclude": ["gateway"]},
    ]
    rows = [
        {"canonical": "AB covered", "aliases": ["cache shared"],
         "grounding": "gatecovered"},
    ]
    corpus = {
        "sym_files": {
            "abcovered": {"a.go", "b.go"},
            "abignored": {"a.go", "b.go"},
            "abexcludeitem": {"a.go", "b.go"},
            "cacheshared": {"a.go", "b.go"},
            "gatecovered": {"a.go", "b.go"},
            "gatewayclient": {"a.go", "b.go"},
            "cachegate": {"a.go", "b.go"},
        },
        "structural": {"abstructural", "cachestructural"},
    }
    indexed = cd.coverage_report(families, rows, corpus)
    serial = cd.coverage_report(families, rows, corpus, indexed=False)
    assert indexed == serial
    assert indexed["discovered"] < sum(f["discovered"] for f in indexed["per_family"])


def test_indexed_fold_matches_serial_payload_and_documents() -> None:
    rows = [
        row(canonical="Cache key", grounding="cachekey"),
        sibling(canonical="Key cache", grounding="keycache"),
    ]
    corpus = {
        "sym_files": {
            "cachekey": {"a.go", "b.go"},
            "keycache": {"a.go", "b.go"},
            "cacheworker": {"a.go", "b.go"},
        },
        "structural": {"cachequeue"},
    }
    grounded = set(corpus["sym_files"]) | corpus["structural"]
    tree_facts = {
        "corpus": corpus,
        "in_tree": lambda token: token in grounded,
        "exists": lambda _path: True,
        "doc_verbs": set(),
    }
    data = _data(rows)
    indexed = cd.build_payload(workspace="/same", data=data, tree=tree_facts)
    serial = cd.build_payload(workspace="/same", data=data, tree=tree_facts, indexed=False)
    assert json.dumps(indexed, indent=2) == json.dumps(serial, indent=2)
    assert cd.render_doc_folder(indexed, stamp="2026-08-28") == cd.render_doc_folder(
        serial, stamp="2026-08-28")


def test_current_catalog_indexes_bound_candidate_work() -> None:
    root = Path(".").resolve()
    data, err = cd.load_data(root / cd.DATA_DIR_REL)
    assert data is not None and not err
    roots = [cd.norm_token(raw) for family in data["families"]
             for raw in family.get("roots", []) if cd.norm_token(raw)]
    counts = {f"x{i:05d}{roots[i % len(roots)]}tail": 2 for i in range(18000)}
    counts.update({cd.norm_token(r.get("grounding", "")): 2 for r in data["rows"]
                   if cd.norm_token(r.get("grounding", ""))})
    structural = set(list(counts)[::200])
    pair_stats: dict[str, int] = {}
    cd.confusable_pairs(data["rows"], _stats=pair_stats)
    assert pair_stats["near_candidate_pairs"] * 20 < pair_stats["near_legacy_pairs"]
    assert pair_stats["near_distance_checks"] <= pair_stats["near_candidate_pairs"]

    coverage_stats: dict[str, int] = {}
    cd.coverage_report(
        data["families"], data["rows"],
        {"sym_files": counts, "structural": structural}, _stats=coverage_stats)
    assert (coverage_stats["coverage_candidate_tokens"] * 4 <
            coverage_stats["coverage_legacy_family_token_probes"])
    assert (coverage_stats["coverage_root_candidate_probes"] <
            coverage_stats["coverage_legacy_family_token_probes"])


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


# --- provenance (#5609) -------------------------------------------------------------
# The `As of` row dates every other number on the page. It rendered as `| As of |  (fak ) |`
# for as long as the data directory carried no `meta` block: the card builder coerces a missing
# key to "", so the renderer's `'?'` fallback could never fire — `.get(key, '?')` returns the
# present empty string, not the default. The table still looked well-formed, which is exactly
# why it survived. Empty is now treated as missing, and missing refuses.

def test_provenance_row_renders_both_values() -> None:
    got = cd.provenance_row({"as_of": "2026-08-05", "fak_version": "0.43.0"})
    assert got == "| As of | 2026-08-05 (fak 0.43.0) |", got


def test_provenance_row_refuses_empty_or_missing() -> None:
    for card in (
        {},                                               # neither key present
        {"as_of": "", "fak_version": ""},                 # present and empty — the real bug
        {"as_of": "2026-08-05"},                          # version missing
        {"as_of": "2026-08-05", "fak_version": ""},       # version present and empty
        {"fak_version": "0.43.0"},                        # date missing
        {"as_of": "   ", "fak_version": "0.43.0"},        # whitespace is not a date
        {"as_of": None, "fak_version": None},             # explicit null
    ):
        try:
            out = cd.provenance_row(card)
        except ValueError as exc:
            assert "undated" in str(exc), str(exc)
            continue
        raise AssertionError(f"rendered {out!r} for {card!r}; an empty provenance must refuse")


def test_committed_data_carries_a_populated_provenance() -> None:
    """The shipped data directory must be able to date its own scorecard.

    Without this the refusal above would simply move the failure to generation time for
    everyone; the point is that the committed corpus satisfies it.
    """
    meta_path = Path(__file__).resolve().parent / "concept_disambiguation_scorecard.data" / "_meta.json"
    meta = (json.loads(meta_path.read_text(encoding="utf-8")).get("meta") or {})
    row_out = cd.provenance_row({"as_of": meta.get("as_of", ""),
                                 "fak_version": meta.get("fak_version", "")})
    assert row_out.startswith("| As of | ") and "(fak " in row_out, row_out
