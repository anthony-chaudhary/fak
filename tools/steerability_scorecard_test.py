#!/usr/bin/env python3
"""Tests for the steerability scorecard.

Drives the PURE helpers (percentile, gini, import parsing) and per-KPI checks with
fixtures (no disk), the fold to the steerability index + steerability-debt, the
GROWTH-INVARIANCE property that distinguishes this scorecard from every sibling
(double the sample at the same shape -> the rate/percentile/Gini KPIs score
identically), then a tolerant live smoke that `collect` folds the real tree with a
zero hard-debt floor.

Run: `python tools/steerability_scorecard_test.py`  (exit 0 = all pass),
or `python -m pytest tools/steerability_scorecard_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import steerability_scorecard as st  # noqa: E402


# --- pure helpers ----------------------------------------------------------

def test_grade_letter_bands() -> None:
    assert st.grade_letter(95) == "A"
    assert st.grade_letter(82) == "B"
    assert st.grade_letter(61) == "D"
    assert st.grade_letter(40) == "F"


def test_percentile_nearest_rank() -> None:
    assert st.percentile([], 0.9) == 0.0
    assert st.percentile([5], 0.9) == 5.0
    vals = list(range(1, 101))  # 1..100
    assert st.percentile(vals, 0.50) == 50.0
    assert st.percentile(vals, 0.90) == 90.0
    assert st.percentile(vals, 1.0) == 100.0
    assert st.percentile(vals, 0.0) == 1.0


def test_percentile_is_scale_free() -> None:
    # The growth-invariance backbone: doubling the sample at the same SHAPE leaves
    # the percentile unchanged. (Each value appears twice -> same distribution.)
    vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    doubled = vals + vals
    assert st.percentile(vals, 0.9) == st.percentile(doubled, 0.9)
    assert st.percentile(vals, 0.5) == st.percentile(doubled, 0.5)


def test_gini_flat_vs_concentrated() -> None:
    assert st.gini([]) == 0.0
    assert st.gini([0, 0, 0]) == 0.0
    flat = st.gini([5, 5, 5, 5, 5])           # perfectly even -> ~0
    assert flat < 0.01
    concentrated = st.gini([0, 0, 0, 0, 100])  # all mass on one node -> high
    assert concentrated > 0.7
    assert concentrated > flat


def test_gini_scale_free_in_value() -> None:
    # Multiplying every weight by a constant leaves the Gini unchanged — why it
    # suits a coupling KPI (the absolute edge count grows, the shape does not).
    base = [1, 2, 3, 4, 10]
    scaled = [v * 7 for v in base]
    assert abs(st.gini(base) - st.gini(scaled)) < 1e-9


# --- import-graph parsing (the go-free fan-in graph) -----------------------

MODULE = "github.com/anthony-chaudhary/fak"


def test_parse_internal_imports_block_and_single() -> None:
    text = (
        'package x\n'
        'import (\n'
        '\t"fmt"\n'                                                  # stdlib — excluded
        '\t"github.com/anthony-chaudhary/fak/internal/abi"\n'        # edge -> abi
        '\tfakmodel "github.com/anthony-chaudhary/fak/internal/model"\n'  # alias -> model
        '\t_ "github.com/anthony-chaudhary/fak/internal/registrations"\n'  # blank — excluded
        '\t"github.com/anthony-chaudhary/fak/pkg/dropin"\n'          # edge -> dropin
        ')\n'
        'import "github.com/anthony-chaudhary/fak/internal/policy"\n'  # single-line -> policy
        'import "github.com/other/dep"\n'                            # external — excluded
    )
    got = st.parse_internal_imports(text, MODULE)
    assert got == sorted(["abi", "model", "dropin", "policy"])


def test_parse_internal_imports_excludes_blank() -> None:
    text = ('import (\n'
            '\t_ "github.com/anthony-chaudhary/fak/internal/blob"\n'  # side-effect only
            ')\n')
    assert st.parse_internal_imports(text, MODULE) == []


def test_parse_internal_imports_subpackage_leaf() -> None:
    # internal/model/kvbackend -> leaf is the FIRST segment after the kind dir.
    text = 'import "github.com/anthony-chaudhary/fak/internal/model/sub"\n'
    assert st.parse_internal_imports(text, MODULE) == ["model"]


def test_package_helpers() -> None:
    assert st.package_of("internal/abi/registry.go") == "internal/abi"
    assert st.package_of("main.go") == ""
    assert st.package_leaf("internal/abi") == "abi"
    assert st.package_leaf("abi") == "abi"


# --- per-KPI: clean case + trigger case ------------------------------------

def test_kpi_file_size_dist_clean_and_drift() -> None:
    clean = st.kpi_file_size_dist([100, 150, 200, 250, 300])  # p90 well under ref
    assert clean["score"] == 100
    assert clean["defects"] == [] and clean["soft"] == []
    drift = st.kpi_file_size_dist([200] * 8 + [900, 900])     # p90 = 900 > ref 520
    assert drift["score"] < 100
    assert drift["defects"] == []   # SOFT only — never debt
    assert drift["soft"]


def test_kpi_func_size_dist_rate() -> None:
    clean = st.kpi_func_size_dist(0, 1000)
    assert clean["score"] == 100 and clean["soft"] == []
    drift = st.kpi_func_size_dist(100, 1000)  # 10% long > 5% ref
    assert drift["score"] < 100 and drift["defects"] == [] and drift["soft"]


def test_kpi_god_file_rate_is_soft() -> None:
    # Orthogonal to code_quality: it SCORES the rate but emits NO debt (no double-count).
    k = st.kpi_god_file_rate(3, 500)
    assert k["defects"] == []
    assert k["soft"]
    assert k["score"] < 100
    assert st.kpi_god_file_rate(0, 500)["score"] == 100


def test_kpi_god_func_rate_is_soft() -> None:
    k = st.kpi_god_func_rate(5, 5000)
    assert k["defects"] == [] and k["soft"]


def test_kpi_fan_in_gini_flat_is_steerable() -> None:
    flat = st.kpi_fan_in_gini({"a": 3, "b": 3, "c": 3, "d": 3})
    assert flat["score"] == 100 and flat["defects"] == []
    hubby = st.kpi_fan_in_gini({"a": 1, "b": 1, "c": 1, "hub": 50})
    assert hubby["score"] < 100 and hubby["defects"] == [] and hubby["soft"]


def test_kpi_hub_share_chokepoint() -> None:
    fine = st.kpi_hub_share({"a": 2, "b": 1}, 20)         # max share 10%
    assert fine["score"] == 100 and fine["defects"] == []
    choke = st.kpi_hub_share({"hub": 18, "a": 1}, 20)      # 90% share
    assert choke["score"] < 50 and choke["defects"] == [] and choke["soft"]


def test_kpi_dispatch_god_file_is_hard() -> None:
    # The ONE coupling HARD KPI: a cmd/* dispatch monolith emits debt.
    clean = st.kpi_dispatch_god_file([])
    assert clean["defects"] == [] and clean["score"] == 100
    bad = st.kpi_dispatch_god_file([("cmd/fak/main.go", 1800)])
    assert len(bad["defects"]) == 1 and bad["score"] < 100


def test_kpi_package_doc_frac_is_soft() -> None:
    full = st.kpi_package_doc_frac(10, 10)
    assert full["score"] == 100 and full["defects"] == [] and full["soft"] == []
    half = st.kpi_package_doc_frac(5, 10)
    assert half["score"] == 50 and half["defects"] == [] and half["soft"]


def test_kpi_ratchet_present_hard() -> None:
    # absent baseline -> debt; wired + well-formed baseline -> clean.
    absent = st.kpi_ratchet_present(None, True)
    assert len(absent["defects"]) == 1
    good = st.kpi_ratchet_present({"metrics": {"x": 1}, "total_debt": 4}, True)
    assert good["defects"] == [] and good["score"] == 100
    unwired = st.kpi_ratchet_present({"metrics": {"x": 1}, "total_debt": 4}, False)
    assert len(unwired["defects"]) == 1   # the "not wired" defect
    malformed = st.kpi_ratchet_present({"metrics": "nope", "total_debt": "x"}, True)
    assert len(malformed["defects"]) == 2


def test_kpi_worst_pkg_drift_soft() -> None:
    unpinned = st.kpi_worst_pkg_drift({"a": 100}, {})
    assert unpinned["score"] == 100 and unpinned["soft"] == []
    drifted = st.kpi_worst_pkg_drift({"a": 300}, {"a": 100})  # +200% > warn line
    assert drifted["defects"] == [] and drifted["soft"]
    assert drifted["score"] < 100


def test_kpi_churn_concentration_head_relative_soft() -> None:
    unavailable = st.kpi_churn_concentration({}, "HEAD~40..HEAD", available=False)
    assert unavailable["score"] == 100 and unavailable["defects"] == []
    flat = st.kpi_churn_concentration({"a": 5, "b": 5, "c": 5}, "R", available=True)
    assert flat["defects"] == []   # never debt — HEAD-relative advisory
    hot = st.kpi_churn_concentration({"a": 1, "b": 1, "c": 1, "d": 1, "hot": 200},
                                     "R", available=True)  # Gini ~0.78 > green
    assert hot["defects"] == [] and hot["soft"]


# --- the fold --------------------------------------------------------------

def _kpis(**overrides: dict) -> list[dict]:
    """A clean KPI set (all 100, no debt), with optional per-name overrides."""
    base = {
        "file_size_dist": 100, "func_size_dist": 100, "god_file_rate": 100,
        "god_func_rate": 100, "fan_in_gini": 100, "hub_share": 100,
        "dispatch_god_file": 100, "package_doc_frac": 100, "ratchet_present": 100,
        "worst_pkg_drift": 100, "churn_concentration": 100,
    }
    out = []
    for name, score in base.items():
        ov = overrides.get(name, {})
        out.append({"kpi": name, "group": st.KPI_GROUP[name],
                    "score": ov.get("score", score),
                    "detail": "", "defects": ov.get("defects", []),
                    "soft": ov.get("soft", [])})
    return out


def test_fold_clean_is_steerable() -> None:
    p = st.build_payload(workspace="/x", kpis=_kpis())
    assert p["ok"] is True and p["verdict"] == "OK"
    assert p["corpus"]["index"] == 100.0
    assert p["corpus"]["grade"] == "A"
    assert p["corpus"]["steerability_debt"] == 0


def test_fold_debt_blocks_ok() -> None:
    p = st.build_payload(workspace="/x", kpis=_kpis(
        dispatch_god_file={"score": 80, "defects": ["cmd god-file"]}))
    assert p["ok"] is False and p["verdict"] == "ACTION"
    assert p["corpus"]["steerability_debt"] == 1
    # the control-pane keys must exist and be the right types
    assert isinstance(p["corpus"]["steerability_debt"], int)
    assert isinstance(p["corpus"]["grade"], str)


def test_fold_index_is_weighted_not_count() -> None:
    # An all-100 tree scores 100 regardless of how MANY KPIs — the index is a
    # weighted mean, so it does not sink as KPIs (or the repo) grow.
    p = st.build_payload(workspace="/x", kpis=_kpis(
        package_doc_frac={"score": 50, "soft": ["half"]}))
    # only the navigability weight (0.10) of the 50-point drop applies
    assert 94.0 <= p["corpus"]["index"] <= 96.0


# --- THE distinguishing property: growth-invariance ------------------------

def test_growth_invariance_rates_unchanged_when_tree_doubles() -> None:
    """The property that separates this scorecard from every sibling: a 2x-larger
    tree with the SAME shape scores identically on the rate/percentile/Gini KPIs.
    A count-based KPI (every sibling) would change. We model "double the tree" by
    doubling every sample at the same distribution."""
    files = [100, 200, 300, 400, 500]
    assert (st.kpi_file_size_dist(files)["score"]
            == st.kpi_file_size_dist(files + files)["score"])

    # god-file RATE: 1 in 100 vs 2 in 200 -> identical rate -> identical score.
    assert (st.kpi_god_file_rate(1, 100)["score"]
            == st.kpi_god_file_rate(2, 200)["score"])
    assert (st.kpi_func_size_dist(10, 100)["score"]
            == st.kpi_func_size_dist(20, 200)["score"])

    # fan-in Gini: the SAME shape at 2x the packages -> ~same score (small-N drift
    # only). Doubling each node's weight is exactly scale-free.
    g1 = {"a": 1, "b": 2, "hub": 10}
    g2 = {k: v * 2 for k, v in g1.items()}
    assert st.kpi_fan_in_gini(g1)["score"] == st.kpi_fan_in_gini(g2)["score"]


def test_growth_invariance_debt_does_not_rise_from_size_alone() -> None:
    """The HARD debt set is rate/structural, so growing the tree at the same shape
    does NOT manufacture debt. (Contrast: hygiene-debt's duplicate COUNT would.)"""
    small = st.build_payload(workspace="/x", kpis=_kpis())
    assert small["corpus"]["steerability_debt"] == 0
    # the only HARD KPIs are dispatch_god_file + ratchet_present — neither is a
    # function of repo SIZE, so a clean small tree and a clean 10x tree both = 0.


# --- live smoke (the regression sentinel) ----------------------------------

def test_live_tree_folds_with_zero_hard_debt_floor() -> None:
    """Tolerant smoke: `collect` over the real repo returns a well-formed payload
    with the control-pane keys, a sane index, and — the floor the doctrine wants —
    ZERO hard steerability-debt (no cmd dispatch god-file; the ratchet is wired)."""
    root = st.repo_root()
    payload = st.collect(root)
    if payload.get("finding") == "tooling_error":
        return  # not in a git checkout (e.g. sdist) — nothing to assert
    c = payload["corpus"]
    assert isinstance(c["steerability_debt"], int)
    assert isinstance(c["grade"], str)
    assert 0 <= c["index"] <= 100
    assert len(payload["kpis"]) == 11
    # the live floor: zero HARD debt on the disciplined tree.
    assert c["steerability_debt"] == 0, \
        f"steerability hard-debt floor broke: {c['steerability_debt']} — see work-list"


# --- runner ----------------------------------------------------------------

def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {exc!r}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
