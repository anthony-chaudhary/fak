#!/usr/bin/env python3
"""Tests for the persona-fit scorecard — across each feature space, how much would each
persona like fak?

Drives the PURE core with fixtures (no disk needed): the value-dimension weight
normalization, the grounding checks, the depth-graded delivery, the cell fit + argmax,
the three KPIs (well-formed / evidence-grounded / fit-honest), and the coverage fold.
Then one LIVE assertion that the shipped data dir scores debt-free and deterministically.

The properties under test are the ones that make the matrix trustworthy:
  * a cell is COMPUTED (weights · delivery) — you cannot type a fit number;
  * delivery DEPTH comes only from DISTINCT grounding checks that RESOLVE in the tree —
    a phantom check (a path/command/section that does not exist) is debt, and a duplicate
    witness cannot pad depth;
  * a declared favourite (top_feature / top_persona) that disagrees with the computed
    argmax is an overclaim — the matrix cannot lie about who loves what.
"""
from __future__ import annotations

import persona_fit_scorecard as M


# ---------------------------------------------------------------------------
# Fixture tree — a tiny fake of the real-tree facts the checks cross-check.
# ---------------------------------------------------------------------------

def _tree(present=(), docs=None, cmd_dirs=(), doc_verbs=(), section_tags=None):
    present = set(present)
    docs = docs or {}
    return {
        "exists": lambda p: p in present,
        "doc_text": lambda d: docs.get(d, ""),
        "cmd_dirs": set(cmd_dirs),
        "doc_verbs": set(doc_verbs),
        "section_tags": section_tags or {},
    }


# ---------------------------------------------------------------------------
# normalize_weights
# ---------------------------------------------------------------------------

def test_normalize_weights_sums_to_one():
    w = M.normalize_weights({"runs_today": 3, "documented": 1})
    assert abs(sum(w.values()) - 1.0) < 1e-9, w
    assert abs(w["runs_today"] - 0.75) < 1e-9
    assert abs(w["documented"] - 0.25) < 1e-9
    # every dimension is present (zero for the unweighted ones)
    assert set(w) == set(M.DIM_IDS)


def test_normalize_weights_drops_unknown_and_negative():
    w = M.normalize_weights({"runs_today": 2, "bogus_dim": 5, "proven": -3})
    assert w["runs_today"] == 1.0  # the only positive, known weight
    assert "bogus_dim" not in [k for k, v in w.items() if v > 0]
    assert w["proven"] == 0.0


def test_normalize_weights_zero_vector_is_all_zero():
    w = M.normalize_weights({})
    assert sum(w.values()) == 0.0
    assert set(w) == set(M.DIM_IDS)


# ---------------------------------------------------------------------------
# eval_check — each grounding kind over the fake tree
# ---------------------------------------------------------------------------

def test_eval_check_path_exists():
    tree = _tree(present={"internal/policy"})
    ok, _ = M.eval_check({"kind": "path_exists", "target": "internal/policy"}, tree)
    assert ok
    bad, _ = M.eval_check({"kind": "path_exists", "target": "internal/nope"}, tree)
    assert not bad


def test_eval_check_command_resolves_needs_dir_and_documented_verb():
    tree = _tree(cmd_dirs={"fak"}, doc_verbs={"policy"})
    ok, _ = M.eval_check({"kind": "command_resolves", "command": "go run ./cmd/fak policy --dump"}, tree)
    assert ok
    # verb not documented -> fails
    bad, _ = M.eval_check({"kind": "command_resolves", "command": "go run ./cmd/fak ghostverb"}, tree)
    assert not bad
    # cmd dir missing -> fails
    nodir, _ = M.eval_check({"kind": "command_resolves", "command": "go run ./cmd/missing -x"}, tree)
    assert not nodir
    # a non-fak cmd dir only needs to exist (no verb-doc gate)
    tree2 = _tree(cmd_dirs={"fanbench"})
    okf, _ = M.eval_check({"kind": "command_resolves", "command": "go run ./cmd/fanbench -profile research"}, tree2)
    assert okf


def test_eval_check_claim_section_tag():
    tree = _tree(section_tags={"security substrate": {"SHIPPED"}})
    ok, _ = M.eval_check({"kind": "claim_section", "section": "Security substrate", "tags": ["SHIPPED"]}, tree)
    assert ok
    bad, _ = M.eval_check({"kind": "claim_section", "section": "Security substrate", "tags": ["STUB"]}, tree)
    assert not bad


def test_eval_check_doc_mentions():
    tree = _tree(docs={"d.md": "run fak preflight to gate a tool"})
    ok, _ = M.eval_check({"kind": "doc_mentions", "doc": "d.md", "tokens": ["preflight"]}, tree)
    assert ok
    bad, _ = M.eval_check({"kind": "doc_mentions", "doc": "d.md", "tokens": ["nonexistent"]}, tree)
    assert not bad


# ---------------------------------------------------------------------------
# feature_delivery — depth grading + phantom-check debt
# ---------------------------------------------------------------------------

def test_delivery_depth_graded_by_resolving_checks():
    # targets must be RELEVANT to their dimension (proven accepts internal/ packages).
    tree = _tree(present={"internal/a", "internal/b", "internal/c", "internal/d"})
    # 1 resolving check -> ~1/CAP ; 3+ -> capped at 1.0
    row = {"id": "f", "delivery": [
        {"dim": "proven", "checks": [{"kind": "path_exists", "target": "internal/a"}]},
        {"dim": "trustworthy", "checks": [
            {"kind": "path_exists", "target": "internal/a"},   # 'internal' has no trust token...
        ]},
    ]}
    # trustworthy needs a trust-relevant subject, so give it real ones:
    row["delivery"][1]["checks"] = [
        {"kind": "path_exists", "target": "internal/policy"},
        {"kind": "path_exists", "target": "internal/adjudicator"},
        {"kind": "path_exists", "target": "internal/normgate"},
        {"kind": "path_exists", "target": "internal/ifc"},
    ]
    tree = _tree(present={"internal/a", "internal/policy", "internal/adjudicator",
                          "internal/normgate", "internal/ifc"})
    d = M.feature_delivery(row, tree)
    assert abs(d["vec"]["proven"] - round(1 / M.DEPTH_CAP, 3)) < 1e-9, d["vec"]
    assert d["vec"]["trustworthy"] == 1.0  # 4 relevant+resolving, capped at 1.0
    assert not d["ground_defects"]
    assert not d["relevance_defects"]


def test_delivery_phantom_check_is_debt_and_loses_depth():
    tree = _tree(present={"internal/a"})
    row = {"id": "f", "delivery": [
        {"dim": "proven", "checks": [
            {"kind": "path_exists", "target": "internal/a"},
            {"kind": "path_exists", "target": "internal/ghost"},  # relevant but does not resolve
        ]},
    ]}
    d = M.feature_delivery(row, tree)
    # one of two relevant checks resolved -> depth from 1, not 2
    assert abs(d["vec"]["proven"] - round(1 / M.DEPTH_CAP, 3)) < 1e-9
    assert any("ghost" in g for g in d["ground_defects"]), d["ground_defects"]


def test_relevance_blocks_cross_dimension_gaming():
    """The ungameability blocker the adversarial review found: a resolving-but-unrelated
    check cannot inflate a dimension. Pointing `benchmarked` at security packages earns
    NO depth and is hard relevance-debt."""
    tree = _tree(present={"internal/policy", "internal/adjudicator", "internal/ifc"})
    gamed = {"id": "memory", "delivery": [
        {"dim": "benchmarked", "checks": [
            {"kind": "path_exists", "target": "internal/policy"},
            {"kind": "path_exists", "target": "internal/adjudicator"},
            {"kind": "path_exists", "target": "internal/ifc"},
        ]},
    ]}
    d = M.feature_delivery(gamed, tree)
    assert d["vec"]["benchmarked"] == 0.0, "irrelevant checks must earn no delivery"
    assert len(d["relevance_defects"]) == 3, d["relevance_defects"]
    # and the KPI turns those into hard debt + a sub-100 score
    k = M.kpi_dimension_relevant([gamed], {"memory": d})
    assert len(k["defects"]) == 3 and k["score"] < 100, k
    # a legit benchmark witness IS relevant and earns depth
    tree2 = _tree(cmd_dirs={"fanbench"})
    legit = {"id": "perf", "delivery": [
        {"dim": "benchmarked", "checks": [{"kind": "command_resolves", "command": "go run ./cmd/fanbench -x"}]}]}
    d2 = M.feature_delivery(legit, tree2)
    assert d2["vec"]["benchmarked"] > 0 and not d2["relevance_defects"]


def test_reuse_soft_flags_dual_purpose_witness():
    tree = _tree(present={"ARCHITECTURE.md"})
    row = {"id": "f", "delivery": [
        {"dim": "documented", "checks": [{"kind": "path_exists", "target": "ARCHITECTURE.md"}]},
        {"dim": "extensible", "checks": [{"kind": "path_exists", "target": "ARCHITECTURE.md"}]},
    ]}
    d = M.feature_delivery(row, tree)
    assert any("grounds both" in s for s in d["reuse_soft"]), d["reuse_soft"]
    # reuse is soft, NOT debt — both dimensions still count it
    assert d["vec"]["documented"] > 0 and d["vec"]["extensible"] > 0
    assert not d["relevance_defects"] and not d["ground_defects"]


# ---------------------------------------------------------------------------
# cell_fit + argmax_id + build_matrix
# ---------------------------------------------------------------------------

def test_cell_fit_is_weighted_dot_product():
    w = M.normalize_weights({"runs_today": 1})  # all weight on one dim
    deliv = {d: 0.0 for d in M.DIM_IDS}
    deliv["runs_today"] = 0.5
    assert M.cell_fit(w, deliv) == 50
    deliv["runs_today"] = 1.0
    assert M.cell_fit(w, deliv) == 100


def test_argmax_id_ties_break_by_id_and_zero_is_none():
    assert M.argmax_id({"b": 5, "a": 5}) == "a"   # tie -> lowest id
    assert M.argmax_id({"a": 0, "b": 0}) is None   # all zero -> no honest best
    assert M.argmax_id({}) is None


def test_build_matrix_discriminates_by_persona_values():
    personas = [
        {"id": "runner", "weights": {"runs_today": 1}},
        {"id": "prover", "weights": {"proven": 1}},
    ]
    features = [{"id": "fast"}, {"id": "solid"}]
    deliveries = {
        "fast": {**{d: 0.0 for d in M.DIM_IDS}, "runs_today": 1.0, "proven": 0.0},
        "solid": {**{d: 0.0 for d in M.DIM_IDS}, "runs_today": 0.0, "proven": 1.0},
    }
    mx = M.build_matrix(personas, features, deliveries)
    assert mx["cells"]["runner"]["fast"] == 100
    assert mx["cells"]["runner"]["solid"] == 0
    assert mx["persona_read"]["runner"]["top_feature"] == "fast"
    assert mx["persona_read"]["prover"]["top_feature"] == "solid"
    assert mx["feature_read"]["fast"]["top_persona"] == "runner"


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------

def test_kpi_well_formed_catches_bad_weight_dim_and_dup_check():
    personas = [{"id": "p", "persona": "P", "top_feature": "", "weights": {"ghost_dim": 2}}]
    features = [{"id": "f", "feature": "F", "group": "g", "what": "x", "top_persona": "",
                 "delivery": [{"dim": "proven", "checks": [
                     {"kind": "path_exists", "target": "a"},
                     {"kind": "path_exists", "target": "a"},  # duplicate witness
                 ]}]}]
    k = M.kpi_rows_well_formed(personas, features, {"g"})
    blob = " ".join(k["defects"])
    assert "ghost_dim" in blob, k["defects"]
    assert "duplicate grounding check" in blob, k["defects"]


def test_kpi_evidence_grounded_counts_phantom_as_debt():
    tree = _tree(present={"internal/a"})
    features = [{"id": "f", "delivery": [
        {"dim": "proven", "checks": [
            {"kind": "path_exists", "target": "internal/a"},
            {"kind": "path_exists", "target": "internal/ghost"},
        ]},
    ]}]
    deliv = {"f": M.feature_delivery(features[0], tree)}
    k = M.kpi_evidence_grounded(features, deliv)
    assert len(k["defects"]) == 1, k
    assert k["score"] < 100


def test_kpi_fit_honest_accepts_any_tied_favourite():
    # two features tie for a persona's top fit -> declaring EITHER is honest, not just argmax.
    personas = [{"id": "p", "weights": {"runs_today": 1}, "top_feature": "b"}]  # 'b' is the non-argmax tie member
    features = [{"id": "a", "top_persona": ""}, {"id": "b", "top_persona": ""}]
    deliveries = {
        "a": {**{d: 0.0 for d in M.DIM_IDS}, "runs_today": 1.0},
        "b": {**{d: 0.0 for d in M.DIM_IDS}, "runs_today": 1.0},  # tie with a
    }
    mx = M.build_matrix(personas, features, deliveries)
    k = M.kpi_fit_honest(personas, features, mx)
    # 'a' is the alphabetical argmax, but 'b' ties it -> no overclaim
    assert not any("p:" in d for d in k["defects"]), k["defects"]


def test_kpi_fit_honest_flags_overclaimed_favourite():
    personas = [{"id": "prover", "weights": {"proven": 1}, "top_feature": "fast"}]  # lies: really loves 'solid'
    features = [{"id": "fast", "top_persona": ""}, {"id": "solid", "top_persona": ""}]
    deliveries = {
        "fast": {**{d: 0.0 for d in M.DIM_IDS}, "proven": 0.0},
        "solid": {**{d: 0.0 for d in M.DIM_IDS}, "proven": 1.0},
    }
    mx = M.build_matrix(personas, features, deliveries)
    k = M.kpi_fit_honest(personas, features, mx)
    assert any("overclaim" in d for d in k["defects"]), k["defects"]


def test_coverage_report_counts_unpositioned():
    req_p = [{"id": "a"}, {"id": "b"}]
    req_f = [{"id": "x"}, {"id": "y"}]
    personas = [{"id": "a", "weights": {"runs_today": 1}}]   # b unpositioned
    features = [{"id": "x", "delivery": [{"dim": "runs_today", "checks": [{}]}]}]  # y unpositioned
    cov = M.coverage_report(req_p, req_f, personas, features)
    assert cov["coverage_debt"] == 2  # missing b + missing y
    assert cov["positioned_personas"] == 1
    assert cov["positioned_features"] == 1


# ---------------------------------------------------------------------------
# Live: the shipped data dir is complete, grounded, and deterministic.
# ---------------------------------------------------------------------------

def test_live_shipped_matrix_is_debt_free_and_deterministic():
    root = M.repo_root()
    p1 = M.collect(root)
    p2 = M.collect(root)
    assert p1["corpus"] == p2["corpus"], "scorecard must be deterministic over the tree"
    c = p1["corpus"]
    assert c["persona_fit_debt"] == 0, (
        f"shipped persona-fit matrix carries debt {c['persona_fit_debt']}: "
        f"{p1['reason']} — a grounding check stopped resolving (a renamed package / "
        f"command / CLAIMS section) or a declared favourite drifted from the matrix")
    assert c["coverage"]["coverage_pct"] == 100.0
    # every persona and feature space is positioned and has an honest favourite
    assert len(c["grid"]) == c["coverage"]["required_personas"]
    assert len(c["feature_summary"]) == c["coverage"]["required_features"]


def test_live_every_grounding_check_resolves():
    root = M.repo_root()
    payload = M.collect(root)
    grounded = next(k for k in payload["kpis"] if k["kpi"] == "evidence_grounded")
    assert not grounded["defects"], (
        "phantom grounding check(s) in the shipped data — each names a tree fact that "
        f"does not exist: {grounded['defects'][:5]}")


def test_live_every_grounding_check_is_dimension_relevant():
    root = M.repo_root()
    payload = M.collect(root)
    rel = next(k for k in payload["kpis"] if k["kpi"] == "dimension_relevant")
    assert not rel["defects"], (
        "the shipped data grounds a dimension with an off-topic witness (gaming surface): "
        f"{rel['defects'][:5]}")


def test_live_has_efficient_dimension_and_importance_weighted_appeal():
    # the cost/efficiency dimension the adversarial review asked for, and the
    # importance-weighted appeal that ranks a mass-market win over a niche one.
    assert "efficient" in M.DIM_IDS and len(M.DIM_IDS) == 9
    root = M.repo_root()
    c = M.collect(root)["corpus"]
    perf = next(f for f in c["feature_summary"] if f["feature"] == "performance")
    assert perf["delivery"].get("efficient", 0) > 0, "performance must deliver the efficient dimension"
    # every feature carries a weighted_mean alongside the flat mean
    assert all("weighted_mean" in f for f in c["feature_summary"])


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERR  {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
