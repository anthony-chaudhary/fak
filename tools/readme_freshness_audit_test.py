#!/usr/bin/env python3
"""Tests for the README front-page freshness auditor.

Drives the PURE check functions + grader with fixture strings (no disk needed),
then a tolerant live smoke that `collect` folds the real committed README.

Run: `python tools/readme_freshness_audit_test.py`  (exit 0 = all pass),
or `python -m pytest tools/readme_freshness_audit_test.py -q`.
"""
from __future__ import annotations

import datetime as _dt
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


def test_hygiene_fail_caps_the_grade() -> None:
    checks = [{"check": c, "status": "OK", "score": 1.0} for c in rfa.SUBSTANCE_CHECKS]
    checks.append({"check": "links", "status": "FAIL", "detail": "dead link"})
    p = rfa.build_payload(workspace=".", checks=checks)
    assert p["ok"] is False, p
    assert p["score"] <= rfa.FAIL_SCORE_CAP, p  # a broken page is not a passing grade


def test_grade_letter_boundaries() -> None:
    assert rfa._grade_letter(90) == "A" and rfa._grade_letter(89) == "B"
    assert rfa._grade_letter(60) == "D" and rfa._grade_letter(59) == "F"


# --- live smoke: the real committed README folds without error -------------

def test_live_collect_real_readme() -> None:
    root = rfa.repo_root()
    if not (root / rfa.README_REL).exists():
        return  # tolerant: not in the repo tree
    p = rfa.collect(root, today=TODAY)
    assert p["schema"] == rfa.SCHEMA
    assert "ok" in p and isinstance(p["checks"], list) and p["checks"]


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
