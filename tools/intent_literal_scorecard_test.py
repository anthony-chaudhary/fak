#!/usr/bin/env python3
"""Tests for intent_literal_scorecard: the cross-checks must BITE (a fabricated metric, an
undisclosed divergence, or a verdict overclaim is debt) and the REAL tree must hold at the
pinned floor (intent_literal_debt 0) so a future undisclosed metric reds the ratchet."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intent_literal_scorecard as ils  # noqa: E402


# ---- fixtures ---------------------------------------------------------------------------

def _scaffold(tmp: Path, rows: list[dict], surface_text: str = "metric_x is served over kernel submissions") -> Path:
    """Build a tmp repo with one Go surface + a data dir holding `rows`."""
    data = tmp / ils.DATA_DIR_REL
    data.mkdir(parents=True, exist_ok=True)
    surface_path = "internal/gateway/metrics.go"
    (tmp / "internal" / "gateway").mkdir(parents=True, exist_ok=True)
    (tmp / surface_path).write_text(surface_text, encoding="utf-8")
    (data / "_meta.json").write_text(json.dumps({
        "schema": "fak-intent-literal-scorecard.data/1",
        "meta": {"as_of": "2026-06-27", "fak_version": "vtest", "title": "t"},
        "surfaces": [{"id": "s", "name": "S", "path": surface_path, "note": ""}],
    }), encoding="utf-8")
    (data / "rows-test.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return tmp


def _row(**kw) -> dict:
    base = {
        "id": "r1", "canonical": "metric_x", "surface": "s", "kind": "ratio",
        "invited_intent": "a cache hit rate", "literal": "an adjudication bypass rate",
        "denominator": "self-referential", "grounding": "metric_x",
        "disclosure": "over kernel submissions", "verdict": "disclosed", "gaps": [],
    }
    base.update(kw)
    return base


def _debt(payload) -> int:
    return payload["corpus"]["intent_literal_debt"]


# ---- the cross-checks must BITE ---------------------------------------------------------

def test_clean_divergent_disclosed_is_zero_debt():
    with tempfile.TemporaryDirectory() as d:
        p = ils.run(_scaffold(Path(d), [_row()]))
    assert p["ok"] is True, p["reason"]
    assert _debt(p) == 0
    assert p["corpus"]["grade"] == "A"


def test_undisclosed_divergence_is_debt():
    # disclosure phrase the row claims is NOT present in the surface -> intent_disclosed bites
    with tempfile.TemporaryDirectory() as d:
        p = ils.run(_scaffold(Path(d), [_row(disclosure="this phrase is absent", verdict="misleading")],
                              surface_text="metric_x with no fence"))
    assert _debt(p) >= 1, p
    kpi = next(k for k in p["kpis"] if k["kpi"] == "intent_disclosed")
    assert kpi["defects"], "an undisclosed divergent metric must be a defect"


def test_missing_disclosure_phrase_is_debt():
    with tempfile.TemporaryDirectory() as d:
        p = ils.run(_scaffold(Path(d), [_row(disclosure="", verdict="misleading")]))
    kpi = next(k for k in p["kpis"] if k["kpi"] == "intent_disclosed")
    assert kpi["defects"], "a divergent row naming no disclosure phrase must be a defect"


def test_ungrounded_metric_is_debt():
    # grounding token not present in the surface -> grounded bites (cannot position a phantom)
    with tempfile.TemporaryDirectory() as d:
        p = ils.run(_scaffold(Path(d), [_row(grounding="metric_does_not_exist")],
                              surface_text="metric_x served over kernel submissions"))
    kpi = next(k for k in p["kpis"] if k["kpi"] == "grounded")
    assert kpi["defects"], "a grounding token absent from the surface must be a defect"


def test_verdict_overclaim_is_debt():
    # divergent + disclosure absent, but the row declares 'disclosed' -> verdict_consistency bites
    with tempfile.TemporaryDirectory() as d:
        p = ils.run(_scaffold(Path(d), [_row(disclosure="absent fence", verdict="disclosed")],
                              surface_text="metric_x with no fence"))
    kpi = next(k for k in p["kpis"] if k["kpi"] == "verdict_consistency")
    assert kpi["defects"], "declaring 'disclosed' when the fence is absent must be a defect"


def test_malformed_row_is_debt():
    with tempfile.TemporaryDirectory() as d:
        p = ils.run(_scaffold(Path(d), [_row(kind="not-a-kind", denominator="bogus")]))
    kpi = next(k for k in p["kpis"] if k["kpi"] == "well_formed")
    assert len(kpi["defects"]) >= 2, "bad kind + bad denominator must both be defects"


def test_non_divergent_denominator_is_clear_not_disclosed():
    # an absolute/external-universe denominator whose name matches needs no fence
    with tempfile.TemporaryDirectory() as d:
        p = ils.run(_scaffold(Path(d), [_row(denominator="external-universe", verdict="clear",
                                             disclosure="")]))
    assert _debt(p) == 0, p["reason"]


# ---- pure-function units ----------------------------------------------------------------

def test_expected_verdict_logic():
    assert ils.expected_verdict({"denominator": "absolute"}, False) == "clear"
    assert ils.expected_verdict({"denominator": "self-referential"}, True) == "disclosed"
    assert ils.expected_verdict({"denominator": "subset"}, False) == "misleading"


def test_grade_letter_boundaries():
    assert ils.grade_letter(100) == "A"
    assert ils.grade_letter(95) == "A"
    assert ils.grade_letter(94.9) == "B"
    assert ils.grade_letter(59) == "F"


def test_present_is_whitespace_insensitive_and_case_insensitive():
    assert ils._present("HELP: served  OVER   Kernel Submissions here", "over kernel submissions")
    assert not ils._present("nothing here", "over kernel submissions")


# ---- the REAL tree holds at the pinned floor (regression lock) --------------------------

def test_real_repo_holds_at_zero_debt():
    p = ils.run(ils.repo_root())
    assert _debt(p) == 0, f"intent_literal_debt regressed: {p['reason']}"
    assert p["ok"] is True
    assert p["corpus"]["grade"] == "A"
    # control-pane contract: the debt integer is discoverable and is a real int (not a bool)
    debt = p["corpus"]["intent_literal_debt"]
    assert isinstance(debt, int) and not isinstance(debt, bool)
    # every positioned row is grounded + (if divergent) disclosed in the real tree
    assert p["corpus"]["rows"] >= 10
    assert p["corpus"]["divergent"] >= 8


def test_real_repo_payload_schema():
    p = ils.run(ils.repo_root())
    for key in ("schema", "ok", "verdict", "finding", "reason", "next_action", "corpus", "kpis"):
        assert key in p, f"payload missing {key}"
    assert p["schema"] == ils.SCHEMA


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERR  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
