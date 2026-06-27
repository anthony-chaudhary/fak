#!/usr/bin/env python3
"""Tests for the industry scorecard — industry-first + modular (schema /2).

Two axes are exercised. (1) PARITY-DEBT — the honesty of the rows that exist:
the favorability ratio and the verdict it implies (lead / parity / trails, plus
oracle-only-parity, ceiling-never-leads, band-membership, no-claim-needs-unbuilt),
then each KPI's defect trigger. (2) COVERAGE — the industry-first half: how much
of the taxonomy is positioned, the measured-vs-gap split, and industry-drift
freshness. Then the modular disk shell (merge a data DIRECTORY) and the fold to
the composite. Closes with the load-bearing live smoke: the REAL committed data
directory must fold to ZERO parity-debt AND ZERO coverage-debt — the proof the
shipped competitive map is complete, honest, and sourced.

Run: `python tools/industry_scorecard_test.py`  (exit 0 = all pass),
or `python -m pytest tools/industry_scorecard_test.py -q`.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import industry_scorecard as isc  # noqa: E402


def row(**over) -> dict:
    """A minimal well-formed, honest, traced single-stream comparison row."""
    r = {
        "id": "r1", "dim_id": "d1", "axis": "Axis", "category": "single-stream",
        "competitor": "llama.cpp", "competitor_class": "sota",
        "competitor_value": 10.0, "fak_value": 5.0, "unit": "tok/s",
        "higher_is_better": True, "verdict": "trails", "status": "shipped",
        "fak_commit": "abc1234", "competitor_source": "llama-bench, same box",
        "apples_to_apples": True, "measured_on": "2026-06-20",
    }
    r.update(over)
    return r


def dim(**over) -> dict:
    d = {"id": "d1", "category": "single-stream", "dimension": "Decode tok/s",
         "why_it_matters": "raw speed", "sota_systems": ["llama.cpp"],
         "sota_bar": "X tok/s", "metric_unit": "tok/s", "best_value": None,
         "source_url": "https://example.com", "source_date": "2026-05", "in_scope": True}
    d.update(over)
    return d


# --- favorability + the verdict it implies ---------------------------------

def test_favorability_higher_and_lower() -> None:
    assert abs(isc.favorability(5, 10, True) - 0.5) < 1e-9
    assert abs(isc.favorability(19, 78, False) - 78 / 19) < 1e-9
    assert isc.favorability(None, 10, True) is None
    assert isc.favorability(5, 0, True) is None


def test_expected_verdict_lead_parity_trails() -> None:
    assert isc.expected_verdict(row(fak_value=20, competitor_value=10))[0] == "lead"
    assert isc.expected_verdict(row(fak_value=10, competitor_value=10))[0] == "parity"
    assert isc.expected_verdict(row(fak_value=5, competitor_value=10))[0] == "trails"


def test_expected_verdict_oracle_is_always_parity() -> None:
    r = row(category="correctness", competitor_class="reference-oracle",
            fak_value=None, competitor_value=None, verdict="lead")
    assert isc.expected_verdict(r)[0] == "parity"


def test_expected_verdict_ceiling_never_leads() -> None:
    r = row(competitor_class="theoretical-ceiling", fak_value=8, competitor_value=7)
    assert isc.expected_verdict(r)[0] == "parity"
    r2 = row(competitor_class="theoretical-ceiling", fak_value=5, competitor_value=10)
    assert isc.expected_verdict(r2)[0] == "trails"


def test_expected_verdict_band_inside_is_parity() -> None:
    r = row(competitor_value=None, competitor_range=[50, 99], fak_value=86.7)
    assert isc.expected_verdict(r)[0] == "parity"
    r_above = row(competitor_value=None, competitor_range=[50, 99], fak_value=120)
    assert isc.expected_verdict(r_above)[0] == "lead"


def test_expected_verdict_no_claim_needs_unshipped() -> None:
    ok = row(category="capability", verdict="no-claim", status="stub",
             fak_value=None, competitor_value=None)
    assert isc.expected_verdict(ok)[0] == "no-claim"
    bad = row(category="capability", verdict="no-claim", status="shipped",
              fak_value=None, competitor_value=None)
    assert isc.expected_verdict(bad)[0] != "no-claim"


def test_parse_date_accepts_month_only() -> None:
    assert isc.parse_date("2026-05") is not None
    assert isc.parse_date("2026-05-12") is not None
    assert isc.parse_date("nonsense") is None


# --- per-KPI defect triggers (parity-debt) ---------------------------------

def test_verdict_consistency_catches_overclaim() -> None:
    k = isc.kpi_verdict_consistency([row(verdict="lead", fak_value=5, competitor_value=10)])
    assert len(k["defects"]) == 1 and "lead" in k["defects"][0] and "trails" in k["defects"][0]


def test_verdict_consistency_clean_when_aligned() -> None:
    k = isc.kpi_verdict_consistency([row(), row(verdict="lead", fak_value=20, competitor_value=10)])
    assert k["defects"] == []


def test_baseline_sota_flags_naive() -> None:
    k = isc.kpi_baseline_sota([row(competitor_class="naive")])
    assert len(k["defects"]) == 1 and "NAIVE" in k["defects"][0]
    assert isc.kpi_baseline_sota([row()])["defects"] == []


def test_competitor_named_flags_blank() -> None:
    assert len(isc.kpi_competitor_named([row(competitor="  ")])["defects"]) == 1
    assert isc.kpi_competitor_named([row()])["defects"] == []


def test_apples_disclosed_requires_note() -> None:
    bad = isc.kpi_apples_disclosed([row(apples_to_apples=False, comparison_note="")])
    assert len(bad["defects"]) == 1
    ok = isc.kpi_apples_disclosed([row(apples_to_apples=False, comparison_note="f32 vs Q8, same GPU")])
    assert ok["defects"] == []


def test_fak_traced_shipped_untraced_is_hard_but_inflight_is_soft() -> None:
    hard = isc.kpi_fak_traced([row(status="shipped", fak_commit="", fak_artifact="", fak_doc="")])
    assert len(hard["defects"]) == 1
    soft = isc.kpi_fak_traced([row(status="in-flight", fak_commit="", fak_artifact="", fak_doc="")])
    assert soft["defects"] == [] and len(soft["soft"]) == 1


def test_competitor_sourced_measured_needs_source_gap_exempt() -> None:
    # a measured row with no source is debt...
    assert len(isc.kpi_competitor_sourced([row(competitor_source="")])["defects"]) == 1
    # ...but a pure no-claim gap (no competitor number) is exempt.
    gap = row(verdict="no-claim", status="stub", fak_value=None,
              competitor_value=None, competitor_source="")
    assert isc.kpi_competitor_sourced([gap])["defects"] == []


def test_axis_coverage_missing_category_and_missing_loss() -> None:
    req = [{"category": "single-stream", "min_rows": 1, "must_include_verdict": "trails"},
           {"category": "agent-fleet", "min_rows": 1}]
    k = isc.kpi_axis_coverage([row(verdict="lead", fak_value=20, competitor_value=10)], req)
    joined = " || ".join(k["defects"])
    assert "agent-fleet" in joined and "trails" in joined
    rows = [row(), row(category="agent-fleet", verdict="lead", fak_value=20,
                       competitor_value=10, unit="min", higher_is_better=False)]
    assert isc.kpi_axis_coverage(rows, req)["defects"] == []


def test_well_formed_flags_enum_category_dim_and_missing_numeric() -> None:
    cats = {"single-stream", "agent-fleet"}
    dids = {"d1"}
    assert any("competitor_class" in d for d in
               isc.kpi_well_formed([row(competitor_class="x")], cats, dids)["defects"])
    # undeclared category
    assert any("category" in d for d in
               isc.kpi_well_formed([row(category="bogus")], cats, dids)["defects"])
    # dim_id that doesn't resolve
    assert any("dim_id" in d for d in
               isc.kpi_well_formed([row(dim_id="nope")], cats, dids)["defects"])
    # a measured verdict (trails) with neither pair nor band is malformed
    bad = row(fak_value=None, competitor_value=None)
    assert any("measured" in d for d in isc.kpi_well_formed([bad], cats, dids)["defects"])
    # clean row, no debt
    assert isc.kpi_well_formed([row()], cats, dids)["defects"] == []


# --- coverage + industry freshness (the industry-first half) ----------------

def test_coverage_report_counts_and_gaps() -> None:
    dims = [dim(id="d1", category="single-stream"),
            dim(id="d2", category="throughput"),
            dim(id="d3", category="quantization")]
    cat_group = {"single-stream": "serving", "throughput": "serving", "quantization": "numerics"}
    rows = [row(id="r1", dim_id="d1", category="single-stream")]  # only d1 positioned
    cov = isc.coverage_report(dims, rows, cat_group)
    assert cov["in_scope"] == 3 and cov["covered"] == 1
    assert cov["coverage_debt"] == 2
    assert set(cov["uncovered"]) == {"d2", "d3"}
    assert cov["by_group"]["serving"] == {"total": 2, "covered": 1}
    assert cov["by_group"]["numerics"] == {"total": 1, "covered": 0}


def test_coverage_report_respects_in_scope_false() -> None:
    dims = [dim(id="d1"), dim(id="d2", in_scope=False)]
    cov = isc.coverage_report(dims, [row(dim_id="d1")], {"single-stream": "serving"})
    assert cov["in_scope"] == 1 and cov["coverage_debt"] == 0


def test_measured_report_splits_measured_from_gap() -> None:
    rows = [row(verdict="lead", fak_value=20, competitor_value=10, status="shipped"),
            row(verdict="no-claim", status="stub", fak_value=None, competitor_value=None),
            row(verdict="trails", status="shipped")]
    m = isc.measured_report(rows)
    assert m["measured"] == 2 and m["gap"] == 1


def test_industry_freshness_flags_old_sources() -> None:
    as_of = isc.parse_date("2026-06-23")
    dims = [dim(id="d1", source_date="2026-06"), dim(id="d2", source_date="2024-01")]
    systems = [{"name": "vLLM", "last_reviewed": "2024-02"},
               {"name": "SGLang", "last_reviewed": "2026-06"}]
    f = isc.industry_freshness(dims, systems, as_of, 180)
    assert len(f["stale_dimensions"]) == 1 and "d2" in f["stale_dimensions"][0]
    assert len(f["stale_systems"]) == 1 and "vLLM" in f["stale_systems"][0]


def test_industry_freshness_last_reviewed_preferred_over_source_date() -> None:
    as_of = isc.parse_date("2026-06-23")
    # d1: old source_date, but recent last_reviewed → NOT stale
    # d2: old source_date, no last_reviewed → stale (falls back to source_date)
    # d3: old source_date, old last_reviewed → stale
    dims = [
        dim(id="d1", source_date="2024-01", last_reviewed="2026-05"),
        dim(id="d2", source_date="2024-01"),
        dim(id="d3", source_date="2024-01", last_reviewed="2024-02"),
    ]
    f = isc.industry_freshness(dims, [], as_of, 180)
    assert len(f["stale_dimensions"]) == 2
    stale_ids = [s.split(":")[0] for s in f["stale_dimensions"]]
    assert "d2" in stale_ids and "d3" in stale_ids
    assert "d1" not in stale_ids


# --- modular disk shell (merge a data directory) ---------------------------

def test_load_data_dir_merges_modular_files() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "_meta.json").write_text(json.dumps({
            "meta": {"as_of": "2026-06-23", "fak_version": "t"},
            "categories": [{"id": "single-stream", "group": "serving", "name": "Single stream"}],
            "required_axes": [],
        }), encoding="utf-8")
        (d / "_taxonomy.json").write_text(json.dumps({"dimensions": [dim()]}), encoding="utf-8")
        (d / "_competitors.json").write_text(json.dumps({"systems": [{"name": "vLLM"}]}), encoding="utf-8")
        (d / "rows-serving.json").write_text(json.dumps({"rows": [row()]}), encoding="utf-8")
        data, err = isc.load_data_dir(d)
        assert err == "" and data is not None
        assert len(data["rows"]) == 1 and data["rows"][0]["_source_file"] == "rows-serving.json"
        assert len(data["taxonomy"]) == 1 and len(data["competitors"]) == 1
        assert data["categories"][0]["group"] == "serving"


# --- fold to the composite --------------------------------------------------

def _data(rows: list[dict], dims: list[dict] | None = None,
          required: list[dict] | None = None) -> dict:
    return {
        "meta": {"as_of": "2026-06-23", "fak_version": "t", "fresh_window_days": 120,
                 "industry_review_window_days": 180},
        "categories": [{"id": "single-stream", "group": "serving", "name": "Single stream"},
                       {"id": "agent-fleet", "group": "agent", "name": "Agent fleet"},
                       {"id": "capability", "group": "capability", "name": "Capability"}],
        "required_axes": required or [],
        "taxonomy": dims if dims is not None else [dim()],
        "competitors": [{"name": "llama.cpp", "last_reviewed": "2026-06"}],
        "rows": rows,
    }


def test_build_payload_zero_debt_full_coverage_is_ok() -> None:
    p = isc.build_payload(workspace=".", data=_data([row()]))
    assert p["ok"] is True and p["verdict"] == "OK"
    assert p["corpus"]["parity_debt"] == 0 and p["corpus"]["coverage_debt"] == 0
    assert p["corpus"]["grade"] == "A"
    assert p["corpus"]["coverage"]["coverage_pct"] == 100.0


def test_build_payload_coverage_gap_drives_action() -> None:
    # two dims, only one positioned → coverage_debt but rows honest.
    dims = [dim(id="d1"), dim(id="d2", category="agent-fleet")]
    p = isc.build_payload(workspace=".", data=_data([row(dim_id="d1")], dims=dims))
    assert p["ok"] is False and p["finding"] == "coverage_debt"
    assert p["corpus"]["parity_debt"] == 0 and p["corpus"]["coverage_debt"] == 1


def test_build_payload_parity_debt_drives_action() -> None:
    dishonest = [row(verdict="lead", fak_value=5, competitor_value=10),
                 row(id="r2", competitor_class="naive")]
    p = isc.build_payload(workspace=".", data=_data(dishonest))
    assert p["ok"] is False and p["finding"] == "parity_debt"
    assert p["corpus"]["parity_debt"] >= 2
    assert p["corpus"]["debt_by_group"]["honesty"] >= 2


def test_build_payload_error_on_no_data() -> None:
    p = isc.build_payload(workspace=".", data=None, error="missing data")
    assert p["ok"] is False and p["verdict"] == "AUDIT_ERROR"


def test_positions_counts_verdicts() -> None:
    pos = isc.positions([row(verdict="lead", fak_value=20, competitor_value=10),
                         row(verdict="trails")])
    assert pos["overall"]["lead"] == 1 and pos["overall"]["trails"] == 1


# --- renderers don't crash + produce the modular folder ---------------------

def test_render_doc_folder_emits_index_and_group_pages() -> None:
    data = _data([row()])
    p = isc.build_payload(workspace=".", data=data)
    files = isc.render_doc_folder(p, data, stamp="2026-06-23")
    assert "README.md" in files and "taxonomy.md" in files and "UPDATE-PROCESS.md" in files
    assert "serving.md" in files  # the group of the single-stream category
    assert "Industry scorecard" in files["README.md"]
    assert "coverage" in files["README.md"].lower()
    # the terminal + gaps + stale renderers run clean
    assert "industry-scorecard:" in isc.render(p)
    assert "backlog" in isc.render_gaps(p)
    assert "backlog" in isc.render_stale(p)


# --- the load-bearing live smoke: the shipped map is complete + honest ------

def test_live_real_data_is_complete_and_honest() -> None:
    root = isc.repo_root()
    path = isc.default_data_path(root)
    if not path.exists():
        return  # tolerant: not in the repo tree
    p = isc.collect(root, data_path=path)
    assert p["schema"] == isc.SCHEMA, p
    # The committed competitive scorecard must carry ZERO parity-debt...
    assert p["corpus"]["parity_debt"] == 0, p["reason"]
    # ...and have positioned every in-scope industry dimension (zero coverage-debt).
    assert p["corpus"]["coverage_debt"] == 0, p["reason"]
    assert p["ok"] is True and p["corpus"]["grade"] == "A"
    # the losses must be shown, not buried.
    assert p["corpus"]["positions"]["overall"]["trails"] >= 1, "the losses must be shown"
    # and the map must be genuinely industry-first, not just the handful fak measured.
    assert p["corpus"]["dimensions"] >= 30, "the taxonomy must cover the field, not a highlight reel"


def test_live_verify_sources_present_match() -> None:
    root = isc.repo_root()
    path = isc.default_data_path(root)
    data, _ = isc.load_data(path)
    if not isinstance(data, dict):
        return
    results = isc.verify_sources(root, data.get("rows") or [])
    bad = [r for r in results if r["status"] in ("MISMATCH", "error")]
    assert not bad, f"source verification failed: {bad}"


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
