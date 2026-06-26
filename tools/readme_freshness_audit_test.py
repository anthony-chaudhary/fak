#!/usr/bin/env python3
"""Tests for the README front-page freshness auditor.

Drives the PURE check functions + grader with fixture strings (no disk needed),
then a tolerant live smoke that `collect` folds the real committed README.

Run: `python tools/readme_freshness_audit_test.py`  (exit 0 = all pass),
or `python -m pytest tools/readme_freshness_audit_test.py -q`.
"""
from __future__ import annotations

import datetime as _dt
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import readme_freshness_audit as rfa  # noqa: E402

TODAY = _dt.date(2026, 6, 20)


# --- pure-check unit tests (each returns a {check,status,...} dict) ---------

def test_links_pass_on_existing(tmp_path: Path) -> None:
    (tmp_path / "real.md").write_text("x", encoding="utf-8")
    c = rfa.check_links("see [it](real.md) and [web](https://x.io) and [a](#sec)", tmp_path)
    assert c["status"] == "OK", c


def test_links_fail_on_dead(tmp_path: Path) -> None:
    c = rfa.check_links("[gone](docs/nope.md)", tmp_path)
    assert c["status"] == "FAIL" and "docs/nope.md" in c["items"], c


def test_version_pins_ok_when_current() -> None:
    c = rfa.check_version_pins("we are at v0.25.0 today", "0.25.0\n")
    assert c["status"] == "OK", c


def test_version_pins_ok_on_forward_range() -> None:
    # A deliberate vX.Y.x range on the CURRENT minor must pass.
    c = rfa.check_version_pins("pinned to v0.25.x", "0.25.0\n")
    assert c["status"] == "OK", c


def test_version_pins_fail_on_stale() -> None:
    c = rfa.check_version_pins("still says v0.3.2 here", "0.25.0\n")
    assert c["status"] == "FAIL" and any("0.3" in s for s in c["items"]), c


def test_naive_baseline_fail_when_bold_leads_naive() -> None:
    c = rfa.check_naive_baseline("that's **~60× vs a naive re-send loop** wow")
    assert c["status"] == "FAIL", c


def test_naive_baseline_ok_when_sota_leads() -> None:
    # SOTA-led bold headline; 'naive' only appears in plain prose, not the bold.
    txt = ("**~4× vs a tuned warm-cache stack**.\n"
           "The naive pattern re-sends everything, which is the cost model.")
    c = rfa.check_naive_baseline(txt)
    assert c["status"] == "OK", c


def test_headline_authority_warn_when_number_absent() -> None:
    c = rfa.check_headline_authority("**99× faster**", "authority lists 4× and 5.3×")
    assert c["status"] == "WARN" and "99×" in c["items"], c


def test_headline_authority_ok_when_mirrored() -> None:
    c = rfa.check_headline_authority("**~4× vs SOTA**", "row: 4× session value")
    assert c["status"] == "OK", c


def test_headline_authority_traces_rate_latency_numbers() -> None:
    c = rfa.check_headline_authority("**~362 ns per call and 120 tok/s**",
                                     "authority: 362 ns overhead; 120 tok/s decode")
    assert c["status"] == "OK", c


def test_headline_authority_warns_on_untraced_rate() -> None:
    c = rfa.check_headline_authority("**999 tok/s**", "authority: 120 tok/s decode")
    assert c["status"] == "WARN" and "999 tok/s" in c["items"], c


def test_freshness_stamp_ok_when_recent() -> None:
    txt = "<!-- readme-verified: 2026-06-18 vs VERSION 0.25.0 -->"
    c = rfa.check_freshness_stamp(txt, today=TODAY, max_age_days=14)
    assert c["status"] == "OK", c


def test_freshness_stamp_warn_when_absent() -> None:
    c = rfa.check_freshness_stamp("no stamp here", today=TODAY, max_age_days=14)
    assert c["status"] == "WARN", c


def test_freshness_stamp_warn_when_stale() -> None:
    txt = "<!-- readme-verified: 2026-01-01 vs VERSION 0.25.0 -->"
    c = rfa.check_freshness_stamp(txt, today=TODAY, max_age_days=14)
    assert c["status"] == "WARN", c


def test_jargon_density_is_advisory_never_fail() -> None:
    txt = "the KV cache and vDSO and context-MMU run inside the address space"
    c = rfa.check_jargon_density(txt, first_screen_lines=110)
    assert c["status"] == "ADVISORY", c
    assert c["items"], "expected naked jargon terms flagged"


def test_jargon_glossed_term_not_flagged() -> None:
    txt = "the KV cache (the model's short-term memory) holds tokens"
    c = rfa.check_jargon_density(txt, first_screen_lines=110)
    assert "KV cache" not in c["items"], c


# --- substance checks (graded affordances → 0..1 score) --------------------

def test_guard_prominence_high_when_leads() -> None:
    txt = ("```\nfak guard -- claude\n```\n"
           "A drop-in secure floor; no api key needed, it forwards your credential.\n")
    c = rfa.check_guard_prominence(txt, first_screen_lines=110)
    assert c["status"] == "OK" and c["score"] >= 0.75, c


def test_guard_prominence_fails_when_absent() -> None:
    c = rfa.check_guard_prominence("no guard here, just serve", first_screen_lines=110)
    assert c["status"] == "FAIL" and c["score"] == 0.0, c


def test_lcd_onramp_high_when_complete() -> None:
    txt = ("# fak\n> **fak in one line:** put it in front of your agent.\n\n"
           "No key, no model, no GPU. Install: `curl … install.sh | sh`.\n"
           "```\nfak preflight --tool refund_payment   # -> DENY\n```\n")
    c = rfa.check_lcd_onramp(txt, first_screen_lines=110, one_glance_lines=8)
    assert c["status"] == "OK" and c["score"] >= 0.75, c


def test_lcd_onramp_scores_real_fak_verb_when_dispatch_known() -> None:
    txt = ("# fak\n> **fak in one line:** put it in front of your agent.\n\n"
           "No key, no model, no GPU. Install: `curl … install.sh | sh`.\n"
           "```\nfak preflight --tool refund_payment   # -> DENY\n```\n")
    c = rfa.check_lcd_onramp(txt, first_screen_lines=110, one_glance_lines=8,
                             dispatch={"preflight"})
    assert "bare_binary_cmd" not in c["items"], c


def test_lcd_onramp_rejects_fake_fak_verb_when_dispatch_known() -> None:
    txt = ("# fak\n> **fak in one line:** put it in front of your agent.\n\n"
           "No key, no model, no GPU. Install: `curl … install.sh | sh`.\n"
           "```\nfak totally-made-up --tool refund_payment   # -> DENY\n```\n")
    c = rfa.check_lcd_onramp(txt, first_screen_lines=110, one_glance_lines=8,
                             dispatch={"preflight", "guard"})
    assert "bare_binary_cmd" in c["items"], c
    assert c["score"] < 1.0, c


def test_lcd_onramp_fails_with_no_command() -> None:
    c = rfa.check_lcd_onramp("just prose, no code, no install", first_screen_lines=110,
                             one_glance_lines=8)
    assert c["status"] == "FAIL" and c["score"] == 0.0, c


def test_speed_claim_zero_without_a_speed_token() -> None:
    # No speed token above the fold ⇒ the framing affordances do NOT score.
    txt = "we link to benchmarks and say vs tuned but quote no speed number"
    c = rfa.check_speed_claim(txt, "authority", first_screen_lines=110)
    assert c["score"] == 0.0 and "speed_token" in c["items"], c


def test_speed_claim_high_when_traced_and_bounded() -> None:
    txt = ("The kernel decision adds ~362 ns per call (in-process), and GPU decode "
           "hits ~120 tok/s — see the benchmarks. Numbers vs a tuned warm-cache stack.")
    authority = "row: ~362 ns decide … 120 tok/s parity"
    c = rfa.check_speed_claim(txt, authority, first_screen_lines=110)
    assert c["score"] >= 0.75, c


def test_speed_claim_untraced_rate_not_saved_by_stray_measured() -> None:
    txt = ("Measured on the lab rig last week.\n"
           "The hero now claims 999 tok/s vs tuned, see BENCHMARK-AUTHORITY.")
    c = rfa.check_speed_claim(txt, "authority says 120 tok/s", first_screen_lines=110)
    assert "traced_or_marked" in c["items"], c
    assert c["score"] < 1.0, c


def test_speed_claim_same_sentence_measured_marks_rate() -> None:
    txt = "The hero reports 999 tok/s measured on a replay run vs tuned; see benchmarks."
    c = rfa.check_speed_claim(txt, "", first_screen_lines=110)
    assert "traced_or_marked" not in c["items"], c


def test_speed_claim_paired_honesty_drops_unsourced_tok_per_second() -> None:
    sourced = ("The hero reports 120 tok/s vs tuned in-process; see benchmarks.")
    fabricated = ("The hero reports 999 tok/s vs tuned in-process; see benchmarks.")
    good = rfa.check_speed_claim(sourced, "authority: 120 tok/s", first_screen_lines=110)
    bad = rfa.check_speed_claim(fabricated, "authority: 120 tok/s", first_screen_lines=110)
    assert "traced_or_marked" not in good["items"], good
    assert "traced_or_marked" in bad["items"], bad
    assert bad["score"] < good["score"], (bad, good)


def test_hero_above_fold_zero_without_a_number() -> None:
    c = rfa.check_hero_above_fold("a page with prose but no headline result",
                                  "authority", first_screen_lines=110)
    assert c["score"] == 0.0 and "has_number" in c["items"], c


def test_hero_above_fold_high_when_traced_sota() -> None:
    txt = "**~4.1× vs a tuned warm-cache stack** on a 50-turn × 5-agent run."
    authority = "headline: 4.1× vs tuned warm-cache"
    c = rfa.check_hero_above_fold(txt, authority, first_screen_lines=110)
    assert c["score"] >= 0.75, c


def test_audience_footholds_all_personas() -> None:
    txt = ("Start here. No key, no GPU — run `fak agent --offline` for an ALLOW/DENY proof. "
           "A default-deny capability floor. Cache prefix reuse keeps the discount. "
           "See CLAIMS.md for what's real.")
    c = rfa.check_audience_footholds(txt, first_screen_lines=110)
    assert c["score"] == 1.0, c


# --- grader / payload tests ------------------------------------------------

def test_payload_ok_all_green() -> None:
    checks = [{"check": "links", "status": "OK", "detail": ""}]
    p = rfa.build_payload(workspace=".", checks=checks)
    assert p["ok"] is True and p["verdict"] == "OK", p


def test_payload_not_ok_on_fail() -> None:
    checks = [{"check": "naive_baseline", "status": "FAIL", "detail": "x"}]
    p = rfa.build_payload(workspace=".", checks=checks)
    assert p["ok"] is False and p["verdict"] == "ACTION", p
    assert "naive_baseline" in p["reason"], p


def test_payload_ok_with_warns() -> None:
    checks = [{"check": "freshness_stamp", "status": "WARN", "detail": "x"}]
    p = rfa.build_payload(workspace=".", checks=checks)
    assert p["ok"] is True and p["finding"] == "readme_fresh_with_notes", p


def test_run_checks_bad_readme_fails_overall() -> None:
    # A README that trips two FAILs: dead link + naive-led headline.
    bad = "[x](nope/missing.md)\n**~60× vs naive loop**"
    checks = rfa.run_checks(bad, "0.25.0", "", Path("."), today=TODAY, max_age_days=14)
    p = rfa.build_payload(workspace=".", checks=checks)
    assert p["ok"] is False, p
    failed = {c["check"] for c in checks if c["status"] == "FAIL"}
    assert {"links", "naive_baseline"} <= failed, failed


def test_synthetic_thin_page_has_positive_readme_debt() -> None:
    checks = rfa.run_checks("# fak\nplain readme\n", "0.25.0", "", Path("."),
                            today=TODAY, max_age_days=14, dispatch=set())
    p = rfa.build_payload(workspace=".", checks=checks)
    assert p["corpus"]["readme_debt"] > 0, p


def test_fak_dispatch_verbs_parses_main_switch(tmp_path: Path) -> None:
    main_go = tmp_path / "cmd" / "fak" / "main.go"
    main_go.parent.mkdir(parents=True)
    main_go.write_text(
        'package main\nfunc main() { switch os.Args[1] {\n'
        'case "preflight":\ncase "version", "-v", "--version":\ncase "help":\n'
        '}}\n',
        encoding="utf-8",
    )
    assert rfa.fak_dispatch_verbs(tmp_path) == {"preflight", "version"}


def test_run_checks_cross_checks_lcd_command_against_dispatch() -> None:
    readme = ("# fak\n> **fak in one line:** put it in front of your agent.\n\n"
              "No key, no model, no GPU. Install: `curl … install.sh | sh`.\n"
              "```\nfak madeup --tool refund_payment   # -> DENY\n```\n")
    checks = rfa.run_checks(readme, "0.25.0", "", Path("."), today=TODAY,
                            max_age_days=14, dispatch={"preflight"})
    lcd = next(c for c in checks if c["check"] == "lcd_onramp")
    assert "bare_binary_cmd" in lcd["items"], lcd


# --- composite score tests -------------------------------------------------

def test_score_is_substance_only_not_padded_by_hygiene() -> None:
    # All hygiene green but every substance check at 0 ⇒ a low score, NOT ~100.
    checks = [
        {"check": "links", "status": "OK"},
        {"check": "version_pins", "status": "OK"},
        {"check": "speed_claim", "status": "WARN", "score": 0.0},
        {"check": "hero_above_fold", "status": "WARN", "score": 0.0},
        {"check": "guard_prominence", "status": "OK", "score": 0.0},
        {"check": "lcd_onramp", "status": "OK", "score": 0.0},
        {"check": "audience_footholds", "status": "OK", "score": 0.0},
    ]
    p = rfa.build_payload(workspace=".", checks=checks)
    assert p["score"] == 0, p  # substance-only: hygiene OK rows do not pad it
    assert p["finding"] == "readme_fresh_thin", p


def test_score_full_when_substance_maxed() -> None:
    checks = [{"check": c, "status": "OK", "score": 1.0} for c in rfa.SUBSTANCE_CHECKS]
    p = rfa.build_payload(workspace=".", checks=checks)
    assert p["score"] == 100 and p["grade"] == "A", p
    assert p["finding"] == "readme_fresh", p
    assert p["readme_debt"] == 0 and p["corpus"] == {
        "score": 100, "grade": "A", "readme_debt": 0,
    }, p


def test_payload_debt_counts_hygiene_fails_and_missing_affordances() -> None:
    checks = [
        {"check": "links", "status": "FAIL", "detail": "dead link"},
        {"check": "freshness_stamp", "status": "WARN", "detail": "old"},
        {"check": "speed_claim", "status": "WARN", "score": 0.5,
         "items": ["bounded_vs_sota", "traced_or_marked"]},
        {"check": "lcd_onramp", "status": "FAIL", "score": 0.0,
         "items": ["bare_binary_cmd"]},
    ]
    p = rfa.build_payload(workspace=".", checks=checks)
    assert p["readme_debt"] == 4, p
    assert p["corpus"]["readme_debt"] == 4, p


def test_hygiene_fail_caps_the_grade() -> None:
    checks = [{"check": c, "status": "OK", "score": 1.0} for c in rfa.SUBSTANCE_CHECKS]
    checks.append({"check": "links", "status": "FAIL", "detail": "dead link"})
    p = rfa.build_payload(workspace=".", checks=checks)
    assert p["ok"] is False, p
    assert p["score"] <= rfa.FAIL_SCORE_CAP, p  # a broken page is not a passing grade


def test_grade_letter_boundaries() -> None:
    assert rfa._grade_letter(90) == "A" and rfa._grade_letter(89) == "B"
    assert rfa._grade_letter(60) == "D" and rfa._grade_letter(59) == "F"


# --- compare (before/after delta) tests ------------------------------------

def test_readme_debt_prefers_payload_contract_over_score() -> None:
    # debt is a lower-is-better integer, not the good-is-high score field.
    assert rfa.readme_debt({"score": 100, "readme_debt": 3}) == 3
    assert rfa.readme_debt({"score": 100, "corpus": {"readme_debt": 2}}) == 2
    assert rfa.readme_debt({"score": 47}) == 53  # legacy pre-/2 baseline fallback


def test_compare_improved_reports_multiplier_verdict() -> None:
    out = rfa.compare(
        {"corpus": {"score": 100, "grade": "A", "readme_debt": 0}},
        {"corpus": {"score": 47, "grade": "F", "readme_debt": 9}},
    )
    assert "readme_debt: 9 -> 0" in out, out
    assert "score:       47/100 -> 100/100" in out, out
    assert ">=3x improvement" in out, out


def test_compare_two_x_when_debt_halved() -> None:
    # baseline score 40 (debt 60) -> current 70 (debt 30): exactly halved.
    out = rfa.compare({"score": 70}, {"score": 40})
    assert ">=2x improvement (debt 60 -> 30)" in out, out


def test_compare_flat_reports_no_change() -> None:
    out = rfa.compare({"score": 80, "grade": "B"}, {"score": 80, "grade": "B"})
    assert "readme_debt: 20 -> 20" in out, out
    assert "no change" in out and "REGRESSED" not in out, out


def test_compare_regressed_reports_regression() -> None:
    # current worse than baseline: debt rose 10 -> 40.
    out = rfa.compare({"score": 60, "grade": "D"}, {"score": 90, "grade": "A"})
    assert "readme_debt: 10 -> 40" in out, out
    assert "REGRESSED" in out, out


def test_compare_surfaces_hygiene_fail_delta() -> None:
    cur = {"score": 90, "grade": "A", "counts": {"FAIL": 1}}
    base = {"score": 90, "grade": "A", "counts": {"FAIL": 0}}
    out = rfa.compare(cur, base)
    assert "hygiene FAILs: 0 -> 1" in out, out


def test_compare_is_deterministic() -> None:
    cur, base = {"score": 88, "grade": "B"}, {"score": 50, "grade": "F"}
    assert rfa.compare(cur, base) == rfa.compare(cur, base)


def test_compare_cli_reads_baseline_payload(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# fak\nplain readme\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"score": 0, "grade": "F"}', encoding="utf-8")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert rfa.main(["--workspace", str(tmp_path), "--compare", str(baseline)]) == 0
    assert "readme-freshness compare:" in out.getvalue()
    baseline.write_text('{"score": 100, "grade": "A"}', encoding="utf-8")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert rfa.main(["--workspace", str(tmp_path), "--compare", str(baseline)]) == 1
    assert "REGRESSED" in out.getvalue()


# --- live smoke: the real committed README folds without error -------------

def test_live_collect_real_readme() -> None:
    root = rfa.repo_root()
    if not (root / rfa.README_REL).exists():
        return  # tolerant: not in the repo tree
    p = rfa.collect(root, today=TODAY)
    assert p["schema"] == rfa.SCHEMA
    assert "ok" in p and isinstance(p["checks"], list) and p["checks"]
    assert p["corpus"]["readme_debt"] == 0, p["corpus"]


# --- self-contained runner (mirrors memory_recall_audit_test.py) -----------

def main() -> int:
    failures: list[str] = []
    import tempfile

    def check(name: str, fn) -> None:
        try:
            # Inject a tmp dir for the two tests that need real files on disk.
            import inspect
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
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
