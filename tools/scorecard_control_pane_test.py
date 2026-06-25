#!/usr/bin/env python3
"""Tests for the unified scorecard debt control-pane.

Drives the PURE fold: debt-integer extraction across the family's two nestings
(corpus.* and doc.*), the portfolio sum, the per-metric trend vs a pinned
baseline (improved / regressed / flat / unpinned), the verdict ladder
(all_clear / scorecard_debt / scorecard_regressed / scorecard_unmeasured), and
the baseline round-trip — then a tolerant live smoke that the real tree folds.

Run: `python tools/scorecard_control_pane_test.py`  (exit 0 = all pass),
or `python -m pytest tools/scorecard_control_pane_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scorecard_control_pane as scp  # noqa: E402


# --- fixtures: minimal scorecard payloads in each family nesting -----------

def corpus_card(debt_key: str, debt: int, grade: str | None = None, ok: bool | None = None) -> dict:
    corpus: dict = {debt_key: debt}
    if grade is not None:
        corpus["grade"] = grade
    return {
        "schema": f"fake/{debt_key}", "ok": ok if ok is not None else debt == 0,
        "verdict": "OK" if debt == 0 else "ACTION", "finding": debt_key,
        "corpus": corpus,
    }


def doc_card(debt: int, grade: str = "A") -> dict:
    # doc-appeal nests its debt under `doc`, not `corpus`.
    return {
        "schema": "fake/appeal_debt", "ok": debt == 0, "verdict": "ACTION",
        "finding": "appeal_debt", "doc": {"appeal_debt": debt, "grade": grade},
    }


def full_metrics(**debts: int) -> list[dict]:
    """Build the folded metric dict per SCORECARDS entry from a debt-by-key map (default 0)."""
    metrics = []
    for card in scp.SCORECARDS:
        if card["key"] == "appeal":
            payload = doc_card(debts.get("appeal", 0))
        else:
            payload = corpus_card(card["debt"], debts.get(card["key"], 0), grade="B")
        metrics.append(scp.metric_from_payload(card, payload))
    return metrics


# --- debt-integer extraction -----------------------------------------------

def test_find_int_corpus_nesting() -> None:
    assert scp.find_int(corpus_card("hygiene_debt", 5), "hygiene_debt") == 5


def test_find_int_doc_nesting() -> None:
    assert scp.find_int(doc_card(4), "appeal_debt") == 4


def test_find_int_missing_returns_none() -> None:
    assert scp.find_int(corpus_card("doc_debt", 3), "code_debt") is None


def test_find_int_ignores_bool() -> None:
    # `ok: true` must never be mistaken for a debt integer of 1.
    assert scp.find_int({"corpus": {"x_debt": True}}, "x_debt") is None


def test_find_grade_prefers_corpus() -> None:
    assert scp.find_grade(corpus_card("code_debt", 9, grade="B")) == "B"
    assert scp.find_grade(doc_card(4, grade="A")) == "A"
    assert scp.find_grade(corpus_card("seo_debt", 6)) is None


def test_metric_from_payload_marks_errors() -> None:
    m = scp.metric_from_payload(scp.SCORECARDS[0], None, error="timed out")
    assert m["debt"] is None and m["verdict"] == "ERROR" and "timed out" in m["error"]
    # payload present but missing the debt key -> measured error, not a crash.
    m2 = scp.metric_from_payload({"key": "x", "debt": "x_debt", "label": "x"},
                                 {"corpus": {"other": 1}})
    assert m2["debt"] is None and "missing x_debt" in m2["error"]


# --- fold: portfolio sum + verdict ladder ----------------------------------

def test_fold_sums_portfolio_debt() -> None:
    metrics = full_metrics(doc=16, code=9, seo=6, hygiene=5, appeal=4, demo=3, robustness=0, parity=0)
    out = scp.fold(metrics, None, workspace=".", commit="abc1234")
    assert out["total_debt"] == 43
    assert out["measured"] == len(scp.SCORECARDS) and out["errored"] == 0
    assert out["schema"] == scp.SCHEMA


def test_fold_all_clear_when_zero_debt() -> None:
    out = scp.fold(full_metrics(), None, workspace=".", commit="c0")
    assert out["ok"] is True and out["verdict"] == "OK" and out["finding"] == "all_clear"


def test_fold_action_when_debt_present_unpinned() -> None:
    out = scp.fold(full_metrics(code=2), None, workspace=".", commit="c0")
    assert out["ok"] is False and out["finding"] == "scorecard_debt"
    assert out["trend"]["direction"] == "unpinned"


def test_fold_flags_unmeasured_scorecard() -> None:
    metrics = full_metrics(code=2)
    metrics[0] = scp.metric_from_payload(scp.SCORECARDS[0], None, error="boom")
    out = scp.fold(metrics, None, workspace=".", commit="c0")
    assert out["finding"] == "scorecard_unmeasured" and out["errored"] == 1


# --- trend vs a pinned baseline --------------------------------------------

def baseline_from(**debts: int) -> dict:
    metrics = full_metrics(**debts)
    payload = scp.fold(metrics, None, workspace=".", commit="base01")
    return scp.baseline_doc(payload)


def test_trend_flat() -> None:
    base = baseline_from(code=5)
    out = scp.fold(full_metrics(code=5), base, workspace=".", commit="now01")
    assert out["trend"]["direction"] == "flat" and out["trend"]["total_delta"] == 0


def test_trend_improved() -> None:
    base = baseline_from(code=5, doc=10)
    out = scp.fold(full_metrics(code=2, doc=10), base, workspace=".", commit="now01")
    t = out["trend"]
    assert t["direction"] == "improved" and t["total_delta"] == -3
    assert "code" in t["deltas"] and t["deltas"]["code"] == -3
    assert "code" in t["improved"]


def test_trend_regressed_sets_action() -> None:
    base = baseline_from(seo=1)
    out = scp.fold(full_metrics(seo=4), base, workspace=".", commit="now01")
    assert out["finding"] == "scorecard_regressed" and out["ok"] is False
    assert out["trend"]["direction"] == "regressed" and out["trend"]["total_delta"] == 3
    assert "seo" in out["trend"]["worsened"]


def test_early_warning_fires_under_a_green_portfolio() -> None:
    """The #712 case: a per-metric rise hidden under a net-improved portfolio. seo
    rises 6->8 while doc falls 16->10, so the portfolio TOTAL drops (22->18, green) —
    the ratchet would stay silent, but the early-warning lens must still name seo."""
    base = baseline_from(seo=6, doc=16)
    out = scp.fold(full_metrics(seo=8, doc=10), base, workspace=".", commit="now01")
    t = out["trend"]
    # the portfolio is GREEN (improved), so the ratchet does not trip...
    assert t["direction"] == "improved" and t["total_delta"] == -4
    # ...but the per-metric lens flags the hidden seo rise (and ONLY seo).
    ew = {e["key"]: e for e in t["early_warning"]}
    assert set(ew) == {"seo"}
    assert ew["seo"]["delta"] == 2 and ew["seo"]["from"] == 6 and ew["seo"]["to"] == 8
    # it is surfaced on the payload + woven into the advisory, verdict unchanged.
    assert out["early_warning"] == t["early_warning"]
    assert "EARLY-WARNING" in out["reason"] and "seo" in out["reason"]


def test_early_warning_empty_when_no_metric_rose() -> None:
    base = baseline_from(seo=6, doc=16)
    out = scp.fold(full_metrics(seo=6, doc=10), base, workspace=".", commit="now01")
    assert out["trend"]["early_warning"] == [] and out["early_warning"] == []
    assert "EARLY-WARNING" not in out["reason"]


def test_check_gate_green_but_surfaces_early_warning_advisory() -> None:
    """--check keeps portfolio ratchet semantics (exit 0 when the total held) yet
    appends the per-metric early-warning as an ADVISORY so a re-pin isn't blind."""
    base = baseline_from(seo=6, doc=16)
    out = scp.fold(full_metrics(seo=8, doc=10), base, workspace=".", commit="now01")
    code, msg = scp.check_gate(out)
    assert code == 0 and "RATCHET OK" in msg           # gate stays green
    assert "EARLY-WARNING" in msg and "seo" in msg     # but the rise is surfaced


def test_check_gate_green_no_warning_when_flat() -> None:
    base = baseline_from(code=5)
    out = scp.fold(full_metrics(code=5), base, workspace=".", commit="now01")
    code, msg = scp.check_gate(out)
    assert code == 0 and "EARLY-WARNING" not in msg


def test_render_shows_early_warning_line() -> None:
    base = baseline_from(seo=6, doc=16)
    out = scp.fold(full_metrics(seo=8, doc=10), base, workspace=".", commit="now01")
    text = scp.render(out)
    assert "early-warning" in text.lower() and "seo" in text


def test_baseline_round_trip() -> None:
    payload = scp.fold(full_metrics(code=9, hygiene=5), None, workspace=".", commit="pin01")
    doc = scp.baseline_doc(payload)
    assert doc["schema"] == scp.BASELINE_SCHEMA
    assert doc["total_debt"] == 14 and doc["metrics"]["code"] == 9
    # folding the same numbers against its own baseline reads flat.
    again = scp.fold(full_metrics(code=9, hygiene=5), doc, workspace=".", commit="pin02")
    assert again["trend"]["direction"] == "flat"


# --- the CI ratchet gate (--check) -----------------------------------------

def test_check_gate_green_when_flat() -> None:
    base = baseline_from(code=5)
    out = scp.fold(full_metrics(code=5), base, workspace=".", commit="now01")
    code, msg = scp.check_gate(out)
    assert code == 0 and "RATCHET OK" in msg


def test_check_gate_green_when_improved_with_residual_debt() -> None:
    # The point of the ratchet: green even with nonzero debt, as long as it fell.
    base = baseline_from(code=9, doc=10)
    out = scp.fold(full_metrics(code=2, doc=10), base, workspace=".", commit="now01")
    code, msg = scp.check_gate(out)
    assert code == 0 and out["total_debt"] > 0 and "RATCHET OK" in msg


def test_check_gate_red_when_regressed() -> None:
    base = baseline_from(seo=1)
    out = scp.fold(full_metrics(seo=4), base, workspace=".", commit="now01")
    code, msg = scp.check_gate(out)
    assert code == 1 and "RATCHET FAIL" in msg and "seo" in msg


def test_check_gate_unpinned_is_distinct_exit() -> None:
    out = scp.fold(full_metrics(code=2), None, workspace=".", commit="now01")
    code, msg = scp.check_gate(out)
    assert code == 2 and "UNPINNED" in msg


def test_check_gate_red_when_unmeasured() -> None:
    base = baseline_from(code=2)
    metrics = full_metrics(code=2)
    metrics[0] = scp.metric_from_payload(scp.SCORECARDS[0], None, error="boom")
    out = scp.fold(metrics, base, workspace=".", commit="now01")
    code, msg = scp.check_gate(out)
    assert code == 1 and "unmeasured" in msg


def test_main_check_wires_gate_exit_code() -> None:
    """`main(['--check'])` must return check_gate's exit code, not the default
    zero-debt verdict — the contract a CI gate (#506/#511) depends on. Drive it
    with stubbed collect/baseline so the wiring is tested fast and deterministic,
    not the live scorecards."""
    orig_collect, orig_load = scp.collect, scp.load_baseline
    try:
        # regressed: current seo=4 above a baseline of seo=1 -> non-zero exit.
        scp.collect = lambda root, timeout=120: full_metrics(seo=4)
        scp.load_baseline = lambda p: baseline_from(seo=1)
        assert scp.main(["--check"]) == 1
        # improved: current debt below baseline -> green even though nonzero before.
        scp.collect = lambda root, timeout=120: full_metrics(seo=0)
        scp.load_baseline = lambda p: baseline_from(seo=5)
        assert scp.main(["--check"]) == 0
        # unpinned: no baseline -> the distinct exit 2.
        scp.load_baseline = lambda p: None
        assert scp.main(["--check"]) == 2
    finally:
        scp.collect, scp.load_baseline = orig_collect, orig_load


def test_main_check_json_ok_reflects_ratchet_not_raw_fold() -> None:
    """`--check --json` must emit a top-level ok/verdict that reflect the RATCHET
    (held vs regressed), not the raw fold (debt==0). A loop runner folds the pane
    off this `ok` bool, so a green-but-nonzero portfolio must read ok:true. #509."""
    import contextlib
    import io
    import json as _json

    orig_collect, orig_load = scp.collect, scp.load_baseline
    try:
        # improved with residual debt: raw fold is ok:false (debt>0) but the
        # ratchet HELD -> the emitted payload must be ok:true / verdict OK.
        scp.collect = lambda root, timeout=120: full_metrics(seo=0)
        scp.load_baseline = lambda p: baseline_from(seo=5)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = scp.main(["--check", "--json"])
        out = _json.loads(buf.getvalue())
        assert code == 0
        assert out["ok"] is True and out["verdict"] == "OK"
        assert out["gate_exit"] == 0 and "RATCHET OK" in out["gate_message"]

        # regressed: ratchet trips -> ok:false / verdict ACTION, exit 1.
        scp.collect = lambda root, timeout=120: full_metrics(seo=4)
        scp.load_baseline = lambda p: baseline_from(seo=1)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = scp.main(["--check", "--json"])
        out = _json.loads(buf.getvalue())
        assert code == 1
        assert out["ok"] is False and out["verdict"] == "ACTION"
        assert out["gate_exit"] == 1
    finally:
        scp.collect, scp.load_baseline = orig_collect, orig_load


# --- tolerant live smoke ----------------------------------------------------

def test_live_collect_and_fold() -> None:
    root = scp.repo_root()
    metrics = scp.collect(root, timeout=120)
    assert len(metrics) == len(scp.SCORECARDS)
    out = scp.fold(metrics, scp.load_baseline(root / scp.BASELINE_REL),
                   workspace=str(root), commit=scp.head_commit(root))
    # The real tree must fold to a well-formed control-pane payload.
    for field in ("schema", "ok", "verdict", "finding", "reason", "next_action", "total_debt", "trend"):
        assert field in out, f"missing {field} in folded payload"
    assert isinstance(out["total_debt"], int)
    # Every scorecard on disk should report a debt integer (no silent ERROR).
    errored = [m["label"] for m in metrics if m["debt"] is None]
    assert not errored, f"scorecards failed to report debt: {errored}"


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
