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

import os
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


def test_readme_scorecard_registered() -> None:
    card = next(c for c in scp.SCORECARDS if c["key"] == "readme")
    assert card == {
        "key": "readme",
        "debt": "readme_debt",
        "script": "readme_freshness_audit.py",
        "label": "readme-freshness",
    }


def test_support_maturity_scorecard_registered() -> None:
    card = next(c for c in scp.SCORECARDS if c["key"] == "support_maturity")
    assert card == {
        "key": "support_maturity",
        "debt": "support_maturity_debt",
        "script": "",
        "cmd": "go run ./cmd/fak support-maturity-scorecard --json",
        "label": "support-maturity",
    }


def test_maturity_scorecard_registered() -> None:
    card = next(c for c in scp.SCORECARDS if c["key"] == "maturity")
    assert card == {
        "key": "maturity",
        "debt": "maturity_debt",
        "script": "",
        "cmd": "go run ./cmd/fak maturity --json",
        "label": "maturity",
    }


def test_operator_heaviness_scorecard_registered() -> None:
    card = next(c for c in scp.SCORECARDS if c["key"] == "heaviness")
    assert card == {
        "key": "heaviness",
        "debt": "heaviness_debt",
        "script": "",
        "cmd": "go run ./cmd/fak operator heaviness --json",
        "label": "operator-heaviness",
    }


# Every tools/*_scorecard.py on disk must be registered in the fold OR named in a
# documented exclusion list — the breadth invariant (#1270). Without this an
# unfolded surface can regress freely and never trip the ratchet, the gap that
# left 7 scorecards (incl. tooling_quality F/debt 66) out of the "universal" rank.
# A *_scorecard.py whose metric is folded under a DIFFERENT script name (the go
# native scorecards) or is intentionally standalone goes here with a reason.
EXCLUDED_SCORECARDS: dict[str, str] = {
    # The conflation/token-defaults/guard-rsi/dogfood/maturity/growth/support-maturity
    # metrics ARE folded, but via `go run ./cmd/fak ...` (no python script); their
    # python test/helper files (if any) are not the fold entry point.
    "vcache_scorecard_gate.py": "a CI gate wrapper, not a portfolio debt scorecard",
}


def _registered_scripts() -> set[str]:
    return {c["script"] for c in scp.SCORECARDS if c.get("script")}


def test_every_scorecard_is_registered_or_excluded() -> None:
    """Breadth invariant: no tools/*_scorecard.py is silently left out of the fold."""
    import glob
    import os
    on_disk = {
        os.path.basename(p)
        for p in glob.glob(str(scp.repo_root() / "tools" / "*_scorecard.py"))
        if not p.endswith("_test.py")
    }
    registered = _registered_scripts()
    unaccounted = sorted(on_disk - registered - set(EXCLUDED_SCORECARDS))
    assert not unaccounted, (
        "scorecard(s) neither registered in SCORECARDS nor in EXCLUDED_SCORECARDS — "
        f"an unfolded surface escapes the universal ranking: {unaccounted}. "
        "Wire it into SCORECARDS or add it to EXCLUDED_SCORECARDS with a reason."
    )


def test_newly_wired_scorecards_registered() -> None:
    # the 7 folded under #1270 — each present with its real debt key.
    want = {
        "observability": "observability_debt", "learning": "learning_debt",
        "rsi_maturity": "rsi_debt", "tooling_quality": "py_debt",
        "bench_dx": "bench_dx_debt", "cuda_dev": "process_debt",
        "persona_fit": "persona_fit_debt",
    }
    by_key = {c["key"]: c for c in scp.SCORECARDS}
    for key, debt in want.items():
        assert key in by_key, f"{key} not registered"
        assert by_key[key]["debt"] == debt


def test_propagation_scorecard_registered() -> None:
    by_key = {c["key"]: c for c in scp.SCORECARDS}
    card = by_key["propagation"]
    assert card["debt"] == "propagation_debt"
    assert card["cmd"] == "go run ./cmd/fak propagation-scorecard --json"
    assert card["label"] == "propagation"


# --- fold: portfolio sum + verdict ladder ----------------------------------

def test_fold_sums_portfolio_debt() -> None:
    metrics = full_metrics(doc=16, code=9, seo=6, hygiene=5, appeal=4, demo=3, robustness=0, parity=0)
    out = scp.fold(metrics, None, workspace=".", commit="abc1234")
    assert out["total_debt"] == 43
    assert out["measured"] == len(scp.SCORECARDS) and out["errored"] == 0
    assert out["schema"] == scp.SCHEMA


def test_readme_debt_folds_and_renders() -> None:
    out = scp.fold(full_metrics(readme=7), None, workspace=".", commit="abc1234")
    assert out["total_debt"] == 7
    text = scp.render(out)
    assert "readme-freshness" in text and "readme_debt" in text and "7" in text


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


# --- grade-weighted severity lens (scale-invariant companion) --------------

def graded_metric(key: str, debt: int, grade: str | None) -> dict:
    """A folded metric for `key` carrying an explicit grade (or none)."""
    card = next(c for c in scp.SCORECARDS if c["key"] == key)
    if grade is None:
        payload = {"corpus": {card["debt"]: debt}}
    else:
        payload = corpus_card(card["debt"], debt, grade=grade)
    return scp.metric_from_payload(card, payload)


def test_grade_weight_maps_letters_to_severity() -> None:
    # the scale-invariant ladder: A free, F worst.
    assert scp.GRADE_DEBT == {"A": 0, "B": 1, "C": 2, "D": 4, "F": 8}
    assert scp.grade_weight({"grade": "A", "debt": 999}) == 0   # grade wins over raw debt
    assert scp.grade_weight({"grade": "F", "debt": 0}) == 8
    assert scp.grade_weight({"grade": "c", "debt": 1}) == 2      # case-insensitive


def test_grade_weight_falls_back_to_debt_when_no_grade() -> None:
    # a scorecard that emits debt but no letter AND no score (readme-freshness)
    # still contributes via the last-resort debt-magnitude tier.
    assert scp.grade_weight({"grade": None, "debt": 0}) == 0     # derive A
    assert scp.grade_weight({"grade": None, "debt": 2}) == 1     # derive B
    assert scp.grade_weight({"grade": None, "debt": 4}) == 2     # derive C
    assert scp.grade_weight({"grade": "??", "debt": 20}) == 8    # bad grade -> derive F


def test_grade_from_score_shared_ladder() -> None:
    assert (scp.grade_from_score(90), scp.grade_from_score(89.9)) == ("A", "B")
    assert (scp.grade_from_score(80), scp.grade_from_score(70)) == ("B", "C")
    assert (scp.grade_from_score(60), scp.grade_from_score(59.9)) == ("D", "F")


def test_score_tier_beats_debt_for_scoreless_scorecard() -> None:
    """The fix: a scoreless-but-scored surface (seo/demo/docs/robustness) with a
    high score but a pile of OCCURRENCE-count debt is graded A from its score, not
    F from debt magnitude — the units-not-severity error the lens exists to kill."""
    # seo on the real tree: score 92.5, seo_debt 25. Debt-magnitude would say F(8);
    # the score says A(0).
    m = {"grade": None, "score": 92.5, "debt": 25}
    assert scp.display_grade(m) == "A" and scp.grade_weight(m) == 0
    # a genuinely-low score still grades F via the SAME score tier.
    assert scp.grade_weight({"grade": None, "score": 41.0, "debt": 25}) == 8


def test_grade_precedence_emitted_letter_beats_score_beats_debt() -> None:
    # emitted letter wins even over a contradicting score...
    assert scp.display_grade({"grade": "C", "score": 99.0, "debt": 0}) == "C"
    # ...score wins over debt when no letter...
    assert scp.display_grade({"grade": None, "score": 75.0, "debt": 999}) == "C"
    # ...debt is the last resort when neither letter nor score is present.
    assert scp.display_grade({"grade": None, "score": None, "debt": 3}) == "C"


def test_metric_from_payload_stamps_corpus_score_for_scoreless_cards() -> None:
    # the seo card emits overall_score at corpus level but no corpus grade; the
    # fold must carry that score so grade_weight can derive the true grade.
    seo = next(c for c in scp.SCORECARDS if c["key"] == "seo")
    payload = {"corpus": {"seo_debt": 25, "overall_score": 92.5}}
    m = scp.metric_from_payload(seo, payload)
    assert m["score"] == 92.5 and m["grade"] is None
    assert scp.grade_weight(m) == 0          # A, from the score — not F from debt 25
    # a card with no SCORE_KEYS entry carries score=None (debt/letter tiers only).
    code = next(c for c in scp.SCORECARDS if c["key"] == "code")
    m2 = scp.metric_from_payload(code, {"corpus": {"code_debt": 5, "grade": "C", "score": 72}})
    assert m2["score"] is None and m2["grade"] == "C"


def test_find_score_prefers_corpus_over_nested_entries() -> None:
    # a per-page mean_score must NOT be picked over the corpus aggregate.
    payload = {"corpus": {"mean_score": 96.0}, "demos": [{"mean_score": 10.0}]}
    assert scp.find_score(payload, "mean_score") == 96.0
    assert scp.find_score({"corpus": {}}, "mean_score") is None
    assert scp.find_score({"corpus": {"mean_score": True}}, "mean_score") is None  # bool != score


def test_grade_debt_is_scale_invariant_vs_raw_sum() -> None:
    """The core fix: a huge raw-unit debt at a GOOD grade weighs less than a tiny
    raw debt at a BAD grade — the opposite of what the raw sum reports. This is
    the discrimination the raw `total_debt` lost to two occurrence-counters."""
    # metric H: 500 raw units but grade B (an occurrence-counter, mostly clean).
    # metric L: 3 raw units but grade F (a bounded metric in real trouble).
    metrics = [graded_metric("slop", 500, "B"), graded_metric("stability", 3, "F")]
    out = scp.fold(metrics, None, workspace=".", commit="c0")
    assert out["total_debt"] == 503           # raw sum: H dominates 500:3
    assert out["grade_debt"] == 1 + 8          # severity: L dominates 8:1
    # the severity breakdown ranks the F metric first, not the 500-unit one.
    assert out["grade_breakdown"].startswith("stability F(8)")


def test_grade_debt_folds_onto_payload_and_baseline() -> None:
    out = scp.fold(full_metrics(code=9, hygiene=5), None, workspace=".", commit="pin01")
    # full_metrics grades every corpus card B (weight 1); appeal nests under doc
    # with no grade and debt 0 -> derive A (weight 0).
    measured = out["measured"]
    assert out["grade_debt"] == measured - 1   # every B except the A-graded appeal
    doc = scp.baseline_doc(out)
    assert doc["grade_debt"] == out["grade_debt"]


def test_grade_regression_reds_the_gate_under_green_raw_ratchet(monkeypatch=None) -> None:
    """#1423 — the "stays Excellent" gate. A severity regression the raw ratchet's
    units mask: slop's occurrence count FALLS (raw total improves, the raw gate
    would stay green) while a bounded metric drops B->F. The grade ratchet must RED
    the gate (HARD by default), not merely warn."""
    os.environ.pop(scp.GRADE_RATCHET_ENV, None)
    base_metrics = [graded_metric("slop", 500, "B"), graded_metric("stability", 1, "B")]
    base = scp.baseline_doc(scp.fold(base_metrics, None, workspace=".", commit="base01"))
    assert base["grade_weights"]["stability"] == 1     # B pinned per-metric
    # now: slop fell 500->480 (raw total drops), but stability B->F (severity rises).
    now_metrics = [graded_metric("slop", 480, "B"), graded_metric("stability", 1, "F")]
    out = scp.fold(now_metrics, base, workspace=".", commit="now01")
    assert out["trend"]["total_delta"] == -20          # raw improved
    assert out["trend"]["grade_delta"] == 7            # 1+1 -> 1+8 severity rose
    assert [g["key"] for g in out["trend"]["grade_regressed"]] == ["stability"]
    code, msg = scp.check_gate(out)
    assert code == 1                                    # HARD: the grade ratchet reds
    assert "GRADE-RATCHET FAIL" in msg and "stability" in msg and "B->F" in msg


def test_grade_regression_demoted_to_advisory_by_env_knob() -> None:
    """FAK_SCORECARD_GRADE_RATCHET=0 restores the pre-#1423 advisory read: the same
    B->F slip stays green, surfaced only as a non-blocking GRADE-DEBT WARN."""
    base_metrics = [graded_metric("slop", 500, "B"), graded_metric("stability", 1, "B")]
    base = scp.baseline_doc(scp.fold(base_metrics, None, workspace=".", commit="base01"))
    now_metrics = [graded_metric("slop", 480, "B"), graded_metric("stability", 1, "F")]
    out = scp.fold(now_metrics, base, workspace=".", commit="now01")
    os.environ[scp.GRADE_RATCHET_ENV] = "0"
    try:
        code, msg = scp.check_gate(out)
    finally:
        os.environ.pop(scp.GRADE_RATCHET_ENV, None)
    assert code == 0                                    # demoted: gate stays green
    assert "GRADE-DEBT WARN" in msg and "+7" in msg     # surfaced advisory
    assert "GRADE-RATCHET FAIL" not in msg


def test_grade_ratchet_green_when_no_letter_slips() -> None:
    """The clean read: raw debt fell and every grade held — the gate passes. Proves
    the grade ratchet does not red on a pure improvement (no false positive)."""
    os.environ.pop(scp.GRADE_RATCHET_ENV, None)
    base_metrics = [graded_metric("slop", 500, "B"), graded_metric("stability", 1, "B")]
    base = scp.baseline_doc(scp.fold(base_metrics, None, workspace=".", commit="base01"))
    # slop fell, stability held its B — no grade slipped.
    now_metrics = [graded_metric("slop", 480, "B"), graded_metric("stability", 1, "B")]
    out = scp.fold(now_metrics, base, workspace=".", commit="now01")
    assert out["trend"]["grade_regressed"] == []
    code, msg = scp.check_gate(out)
    assert code == 0 and "RATCHET OK" in msg


def test_baseline_carries_per_metric_grade_weights() -> None:
    """#1423 DoD: the re-pinned baseline records each scorecard's grade severity, so
    an A->B slip reds even at flat raw debt. Round-trips through baseline_doc."""
    payload = scp.fold([graded_metric("slop", 5, "A"), graded_metric("stability", 3, "C")],
                       None, workspace=".", commit="pin01")
    doc = scp.baseline_doc(payload)
    assert doc["grade_weights"]["slop"] == 0           # A
    assert doc["grade_weights"]["stability"] == 2      # C
    assert doc["grade_debt"] == 2                       # 0 + 2
    # re-fold against its own pin: flat, no grade regression.
    again = scp.fold([graded_metric("slop", 5, "A"), graded_metric("stability", 3, "C")],
                     doc, workspace=".", commit="pin02")
    assert again["trend"]["grade_regressed"] == []
    code, _ = scp.check_gate(again)
    assert code == 0


def test_tracked_baseline_carries_grade_debt_and_weights() -> None:
    """The committed baseline file (#1423 re-pin) carries the grade axis: a
    grade_debt integer AND per-metric grade_weights for the gate to compare."""
    base = scp.load_baseline(scp.repo_root() / scp.BASELINE_REL)
    assert base is not None
    assert isinstance(base.get("grade_debt"), int)
    assert isinstance(base.get("grade_weights"), dict) and base["grade_weights"]


def test_render_shows_grade_debt_line() -> None:
    out = scp.fold([graded_metric("slop", 500, "B"), graded_metric("stability", 3, "F")],
                   None, workspace=".", commit="c0")
    text = scp.render(out)
    assert "grade-debt" in text.lower() and "severity" in text.lower()
    assert "stability F" in text


def test_baseline_round_trip() -> None:
    payload = scp.fold(full_metrics(code=9, hygiene=5), None, workspace=".", commit="pin01")
    doc = scp.baseline_doc(payload)
    assert doc["schema"] == scp.BASELINE_SCHEMA
    assert doc["total_debt"] == 14 and doc["metrics"]["code"] == 9
    assert doc["metrics"]["readme"] == 0
    # folding the same numbers against its own baseline reads flat.
    again = scp.fold(full_metrics(code=9, hygiene=5), doc, workspace=".", commit="pin02")
    assert again["trend"]["direction"] == "flat"


def test_tracked_baseline_pins_readme_floor() -> None:
    base = scp.load_baseline(scp.repo_root() / scp.BASELINE_REL)
    assert base is not None
    assert base["metrics"]["readme"] == 0


def test_make_ci_runs_portfolio_ratchet() -> None:
    makefile = (scp.repo_root() / "Makefile").read_text(encoding="utf-8")
    assert "ci:" in makefile and "scorecard-ratchet" in makefile.split("ci:", 1)[1].splitlines()[0]
    assert "scorecard-ratchet:" in makefile
    assert "tools/readme_freshness_audit_test.py" in makefile
    assert "tools/scorecard_control_pane.py --check" in makefile


def test_ci_posts_scorecards_to_scoreboard_without_lab_token() -> None:
    """#997: CI must produce #scoreboard cards from the control-pane scorecard family.

    This is a workflow contract test, not a Slack integration test: it pins the
    dedicated scoreboard secrets, the always-on no-network dry-run path, the
    control-pane registry source, and the `--from <payload> --debt-key <key>` post
    route. It also guards against accidentally authenticating with the lab
    `SLACK_BOT_TOKEN` instead of the scoreboard workspace token.
    """
    ci = (scp.repo_root() / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    marker = "- name: scorecard scoreboard producer (dry-run + optional post)"
    assert marker in ci
    block = ci.split(marker, 1)[1].split("# Learning-docs scorecard", 1)[0]

    assert "FAK_SCOREBOARD_TOKEN: ${{ secrets.FAK_SCOREBOARD_TOKEN }}" in block
    assert "FAK_SCOREBOARD_CHANNEL: ${{ secrets.FAK_SCOREBOARD_CHANNEL }}" in block
    assert "FAK_SCOREBOARD_SOURCE: ci:${{ github.run_id }}" in block
    assert "SLACK_BOT_TOKEN" not in block

    assert "import scorecard_control_pane as scp" in block
    assert "for card in scp.SCORECARDS" in block
    assert "--from \"$payload\"" in block
    assert "--debt-key \"$debt_key\"" in block
    assert "--source \"$FAK_SCOREBOARD_SOURCE\"" in block
    assert "--dry-run | tee \"$card_text\"" in block
    assert "GITHUB_EVENT_NAME" in block and "GITHUB_REF_NAME" in block
    assert "Live posting requires a push to main/master plus FAK_SCOREBOARD_TOKEN and FAK_SCOREBOARD_CHANNEL CI secrets." in block


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


def test_check_gate_red_when_readme_debt_regresses() -> None:
    base = baseline_from(readme=0)
    out = scp.fold(full_metrics(readme=2), base, workspace=".", commit="now01")
    code, msg = scp.check_gate(out)
    assert code == 1 and "RATCHET FAIL" in msg and "readme-freshness" in msg


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


# --- build-break vs card-bug triage (B0 #1416) -----------------------------

def _error_card(metrics: list[dict], key: str, error: str = "boom") -> list[dict]:
    """Replace the folded metric for `key` with an ERRORED one (no payload)."""
    for i, card in enumerate(scp.SCORECARDS):
        if card["key"] == key:
            metrics[i] = scp.metric_from_payload(card, None, error=error)
            return metrics
    raise KeyError(key)


def test_go_backed_keys_cover_the_go_run_cards() -> None:
    # Every card folded via `go run ./cmd/fak …` is a Go-backed key; the python
    # cards are not. This set drives the build-vs-bug triage.
    assert "maturity" in scp.GO_BACKED_KEYS and "conflation" in scp.GO_BACKED_KEYS
    assert "loopindex" in scp.GO_BACKED_KEYS and "support_maturity" in scp.GO_BACKED_KEYS
    assert "doc" not in scp.GO_BACKED_KEYS and "slop" not in scp.GO_BACKED_KEYS
    # and it is derived from SCORECARDS, never hand-listed.
    derived = {c["key"] for c in scp.SCORECARDS if "go run" in (c.get("cmd") or "")}
    assert scp.GO_BACKED_KEYS == derived


def test_build_break_hint_empty_for_python_card_error() -> None:
    # a python card erroring is a genuine card/measurement fault — no build triage.
    metrics = _error_card(full_metrics(), "doc")
    errored = [m for m in metrics if not isinstance(m.get("debt"), int)]
    assert scp.build_break_hint(errored) == ""


def test_build_break_hint_fires_for_go_backed_card_error() -> None:
    metrics = _error_card(full_metrics(), "maturity")
    errored = [m for m in metrics if not isinstance(m.get("debt"), int)]
    hint = scp.build_break_hint(errored)
    assert "go build ./..." in hint and "Go-backed" in hint
    assert "maturity" in hint and "clean-read recipe" in hint


def test_fold_unmeasured_go_card_carries_build_break_hint() -> None:
    metrics = _error_card(full_metrics(code=2), "conflation")
    out = scp.fold(metrics, None, workspace=".", commit="c0")
    assert out["finding"] == "scorecard_unmeasured" and out["errored"] == 1
    assert "go build ./..." in out["next_action"]


def test_check_gate_build_break_hint_in_message() -> None:
    base = baseline_from(code=2)
    metrics = _error_card(full_metrics(code=2), "growth")
    out = scp.fold(metrics, base, workspace=".", commit="now01")
    code, msg = scp.check_gate(out)
    assert code == 1 and "unmeasured" in msg and "go build ./..." in msg


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
