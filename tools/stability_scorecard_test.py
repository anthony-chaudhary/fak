#!/usr/bin/env python3
"""Tests for the stability scorecard — the stability-debt measuring stick.

Drives the PURE checks with fixtures (no disk needed): each KPI's defect trigger
(a regression gate not wired, a missing ratchet baseline, an untagged claim, an
unencoded invariant, no fail-closed witness, zero determinism witnesses, no
keep/revert ladder, no version marker, no CI-gated tagging, no rollback runbook),
the clean case for each, and the fold to stability-debt + the verdict ladder.
Closes with the load-bearing live smoke: the REAL tracked tree must fold to ZERO
stability-debt — the proof that a regression fails loudly and there is a written
path back to a stable version, and a regression sentinel for the day someone
removes a gate, an invariant, a baseline, or the rollback runbook.

Run: `python tools/stability_scorecard_test.py`  (exit 0 = all pass),
or `python -m pytest tools/stability_scorecard_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stability_scorecard as sc  # noqa: E402


# --- the small helpers ------------------------------------------------------

def test_grade_letter_bands() -> None:
    assert sc.grade_letter(100) == "A" and sc.grade_letter(90) == "A"
    assert sc.grade_letter(85) == "B" and sc.grade_letter(72) == "C"
    assert sc.grade_letter(61) == "D" and sc.grade_letter(40) == "F"


def test_untagged_claims_counts_tags() -> None:
    text = ("- [SHIPPED] real thing\n"
            "- [SIMULATED] [STUB] two tags is malformed\n"
            "- [TODO] a bracketed claim with no status tag\n"
            "- a plain bullet (not a `- [` claim line) is not graded\n")
    bad = sc.untagged_claims(text)
    assert len(bad) == 2
    assert any("2 status tag" in b for b in bad)
    assert any("0 status tag" in b for b in bad)
    assert sc.untagged_claims("- [SHIPPED] all good\n- [STUB] also good") == []


def test_count_fences() -> None:
    assert sc._count_fences("```\ncmd\n```\nprose\n```\ncmd2\n```") == 2
    assert sc._count_fences("no fences here") == 0
    assert sc._count_fences("") == 0


# --- sentinel KPIs ----------------------------------------------------------

def test_regression_gates_wired_defect_and_clean() -> None:
    bad = sc.kpi_regression_gates_wired(["race detector", "main-KPI regression gate"])
    assert len(bad["defects"]) == 2 and bad["score"] < 100
    assert all("regression gate" in d for d in bad["defects"])
    clean = sc.kpi_regression_gates_wired([])
    assert clean["defects"] == [] and clean["score"] == 100


def test_ratchet_baselines_committed_defect_and_clean() -> None:
    bad = sc.kpi_ratchet_baselines_committed(["tools/scorecard_baseline.json"])
    assert len(bad["defects"]) == 1 and "no floor" in bad["defects"][0]
    assert sc.kpi_ratchet_baselines_committed([])["defects"] == []


def test_honesty_ledger_clean_states() -> None:
    assert sc.kpi_honesty_ledger_clean(False, [])["defects"]  # missing ledger
    assert sc.kpi_honesty_ledger_clean(True, [])["defects"] == []  # clean
    dirty = sc.kpi_honesty_ledger_clean(True, ["CLAIMS.md:3: 0 status tag(s) (need exactly 1): x"])
    assert len(dirty["defects"]) == 1
    # untagged is capped at 8 hard defects; the rest spill to soft.
    many = sc.kpi_honesty_ledger_clean(True, [f"CLAIMS.md:{i}: 0 status tag(s)" for i in range(12)])
    assert len(many["defects"]) == 8 and len(many["soft"]) == 1


# --- invariant KPIs ---------------------------------------------------------

def test_invariant_tests_present_defect_and_clean() -> None:
    bad = sc.kpi_invariant_tests_present(["TestNoUpwardImports"])
    assert len(bad["defects"]) == 1 and "TestNoUpwardImports" in bad["defects"][0]
    assert sc.kpi_invariant_tests_present([])["defects"] == []
    assert sc.kpi_invariant_tests_present([])["score"] == 100


def test_fail_closed_witnessed_defect_and_clean() -> None:
    bad = sc.kpi_fail_closed_witnessed(["TestRung0MalformedJSONDenied"])
    assert len(bad["defects"]) == 1 and "REFUSED" in bad["defects"][0]
    assert sc.kpi_fail_closed_witnessed([])["defects"] == []


def test_determinism_witness_zero_thin_and_healthy() -> None:
    zero = sc.kpi_determinism_witness(0)
    assert len(zero["defects"]) == 1 and zero["score"] == 0
    thin = sc.kpi_determinism_witness(2)
    assert thin["defects"] == [] and thin["soft"] and thin["score"] < 100
    healthy = sc.kpi_determinism_witness(18)
    assert healthy["defects"] == [] and healthy["soft"] == [] and healthy["score"] == 100


# --- revert KPIs ------------------------------------------------------------

def test_keep_revert_ladder_states() -> None:
    missing = sc.kpi_keep_revert_ladder(ladder_ok=False, ladder_tested=False, state_seam_present=True)
    assert len(missing["defects"]) == 2
    # committed ladder + tested + the state seam present -> clean, no soft note.
    clean = sc.kpi_keep_revert_ladder(ladder_ok=True, ladder_tested=True, state_seam_present=True)
    assert clean["defects"] == [] and clean["soft"] == []
    # ladder fine but the portable state seam absent (a fork) -> clean + one soft note.
    no_seam = sc.kpi_keep_revert_ladder(ladder_ok=True, ladder_tested=True, state_seam_present=False)
    assert no_seam["defects"] == [] and len(no_seam["soft"]) == 1
    assert "no portable state/session" in no_seam["soft"][0]


def test_version_pin_states() -> None:
    assert sc.kpi_version_pin(True, True)["defects"] == []
    assert len(sc.kpi_version_pin(False, True)["defects"]) == 1
    assert len(sc.kpi_version_pin(False, False)["defects"]) == 2


def test_release_tagging_gated_states() -> None:
    assert sc.kpi_release_tagging_gated(True)["defects"] == []
    assert sc.kpi_release_tagging_gated(False)["defects"]


def test_rollback_runbook_absent_and_complete() -> None:
    sections = [lbl for lbl, _ in sc.ROLLBACK_RUNBOOK_SECTIONS]
    # absent runbook: the doc-missing defect plus one per uncovered section.
    absent = sc.kpi_rollback_runbook(
        present=False, where="", missing_sections=sections, linked=False, has_commands=False)
    assert len(absent["defects"]) == 1 + len(sections)
    # present + every section + linked + has commands -> zero debt.
    complete = sc.kpi_rollback_runbook(
        present=True, where="docs/ROLLBACK.md", missing_sections=[], linked=True, has_commands=True)
    assert complete["defects"] == [] and complete["score"] == 100
    # present but no runnable commands is one unit (documentation theater).
    no_cmds = sc.kpi_rollback_runbook(
        present=True, where="docs/ROLLBACK.md", missing_sections=[], linked=True, has_commands=False)
    assert len(no_cmds["defects"]) == 1 and "no runnable commands" in no_cmds["defects"][0]
    # present but not linked is one unit (an operator can't find it).
    unlinked = sc.kpi_rollback_runbook(
        present=True, where="docs/ROLLBACK.md", missing_sections=[], linked=False, has_commands=True)
    assert len(unlinked["defects"]) == 1 and "linked from no entry point" in unlinked["defects"][0]


def test_rollback_runbook_tokens_are_real_anchors_not_prose_theater() -> None:
    """The anti-gaming guard: an empty file prose-stuffed with English phrases must
    NOT satisfy the section checks — each token is a REAL anchor (a git command, a
    `fak` verb / package, a tool path, an env var)."""
    junk = "downgrade restore a snapshot --pin pin to v rollback revert state"
    missing = [lbl for lbl, toks in sc.ROLLBACK_RUNBOOK_SECTIONS if not sc._has(junk, *toks)]
    assert missing == [lbl for lbl, _ in sc.ROLLBACK_RUNBOOK_SECTIONS], (
        f"junk prose satisfied sections {set(l for l,_ in sc.ROLLBACK_RUNBOOK_SECTIONS) - set(missing)} "
        "— the tokens are gameable, not real anchors")
    # a real runbook naming the real mechanisms covers every section.
    real = ("git checkout v0.31.0 ... git revert ... fak snapshot restore-fleet ... "
            "internal/snapshot ... scorecard_control_pane.py --pin ... FAK_APP_VERSION")
    assert [lbl for lbl, toks in sc.ROLLBACK_RUNBOOK_SECTIONS if not sc._has(real, *toks)] == []


# --- drift KPIs -------------------------------------------------------------

def test_drift_detectors_wired_defect_and_soft() -> None:
    bad = sc.kpi_drift_detectors_wired(
        ["tools/check_index_sync.py"], no_tail_wag_tool=True, no_early_warning=True)
    assert len(bad["defects"]) == 1
    assert len(bad["soft"]) == 2  # tail-wag-no-tool + no-early-warning
    clean = sc.kpi_drift_detectors_wired([], no_tail_wag_tool=False, no_early_warning=False)
    assert clean["defects"] == [] and clean["soft"] == [] and clean["score"] == 100


def test_confusion_escalation_signal_is_soft_only() -> None:
    absent = sc.kpi_confusion_escalation_signal(has_signal=False, doctrine_present=True)
    assert absent["defects"] == [] and len(absent["soft"]) == 1  # never HARD debt
    assert "verification-ladder" in absent["soft"][0]
    present = sc.kpi_confusion_escalation_signal(has_signal=True, doctrine_present=True)
    assert present["defects"] == [] and present["soft"] == [] and present["score"] == 100


# --- the fold ---------------------------------------------------------------

def test_weights_sum_to_one_and_cover_every_kpi() -> None:
    assert abs(sum(sc.KPI_WEIGHTS.values()) - 1.0) < 1e-9
    assert set(sc.KPI_WEIGHTS) == set(sc.KPI_GROUP)


def _clean_kpis() -> list[dict]:
    """An all-zero-defect fixture: every KPI at its clean case."""
    return [
        sc.kpi_regression_gates_wired([]),
        sc.kpi_ratchet_baselines_committed([]),
        sc.kpi_honesty_ledger_clean(True, []),
        sc.kpi_invariant_tests_present([]),
        sc.kpi_frozen_pins_present([]),
        sc.kpi_fail_closed_witnessed([]),
        sc.kpi_determinism_witness(18),
        sc.kpi_keep_revert_ladder(True, True, state_seam_present=True),
        sc.kpi_version_pin(True, True),
        sc.kpi_release_tagging_gated(True),
        sc.kpi_rollback_runbook(True, "docs/ROLLBACK.md", [], True, True),
        sc.kpi_drift_detectors_wired([], False, False),
        sc.kpi_confusion_escalation_signal(True, True),
    ]


def test_fold_zero_debt_is_grade_a_and_ok() -> None:
    payload = sc.build_payload(workspace="/x", kpis=_clean_kpis())
    c = payload["corpus"]
    assert c["stability_debt"] == 0 and c["grade"] == "A" and c["score"] == 100
    assert payload["ok"] is True and payload["verdict"] == "OK"


def test_fold_debt_counts_defects_and_flips_action() -> None:
    kpis = _clean_kpis()
    kpis[0] = sc.kpi_regression_gates_wired(["race detector", "main-KPI regression gate"])
    payload = sc.build_payload(workspace="/x", kpis=kpis)
    c = payload["corpus"]
    assert c["stability_debt"] == 2 == sum(len(k["defects"]) for k in kpis)
    assert payload["ok"] is False and payload["verdict"] == "ACTION"
    assert c["debt_by_group"]["sentinel"] == 2


def test_fold_audit_error_shape() -> None:
    err = sc.build_payload(workspace="/x", kpis=[], error="not a git repo")
    assert err["ok"] is False and err["verdict"] == "AUDIT_ERROR"
    assert err["corpus"] == {} and err["kpis"] == []


# --- the load-bearing live smoke -------------------------------------------

def test_live_real_tree_is_stable() -> None:
    """The real tracked tree must fold to ZERO stability-debt (grade A): every
    regression gate wired, every core invariant encoded, every ratchet baseline
    committed, a committed keep/revert ladder, and a linked rollback runbook. This
    is the regression sentinel — removing any of those turns this red. (Requires the
    scorecard lane to be committed/on-disk; runs green on a fresh checkout once the
    lane is committed, since the rollback runbook is then tracked.)"""
    payload = sc.collect(sc.repo_root())
    c = payload["corpus"]
    assert c, f"audit error: {payload.get('reason')}"
    assert c["stability_debt"] == 0, (
        f"stability-debt {c['stability_debt']} (expected 0); "
        f"work-list: {[d for k in payload['kpis'] for d in k['defects']]}")
    assert c["grade"] == "A" and payload["ok"] is True


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
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
