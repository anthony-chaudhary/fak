#!/usr/bin/env python3
"""Tests for the repo-hygiene scorecard.

Drives the PURE per-KPI checks + helpers + grader with fixtures (no disk needed),
covers the calibration that keeps the number honest (size-prefiltered duplicate
detection, the dated-doc rule, reader-facing scoping, reachability/orphans, the
SOFT/HARD split), then a tolerant live smoke that `collect` folds the real tree.

Run: `python tools/repo_hygiene_scorecard_test.py`  (exit 0 = all pass),
or `python -m pytest tools/repo_hygiene_scorecard_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repo_hygiene_scorecard as rh  # noqa: E402


# --- helpers ---------------------------------------------------------------

def test_dated_doc_detection() -> None:
    assert rh.is_dated_doc("PLAN-2026-06-19.md")
    assert rh.is_dated_doc("trust-floor-decomposition-492.md")
    assert not rh.is_dated_doc("tutorial.md")
    assert not rh.is_dated_doc("README.md")


def test_reader_facing_scoping() -> None:
    assert rh.is_reader_facing("README.md")          # root front-door doc
    assert rh.is_reader_facing("docs/FAQ.md")        # docs/ surface
    assert not rh.is_reader_facing("docs/releases/v0.3.0.md")   # archival
    assert not rh.is_reader_facing("docs/proofs/policy.md")     # own ledger
    assert not rh.is_reader_facing("docs/notes/X-2026-06-19.md")  # journal
    assert not rh.is_reader_facing("tools/x.py")     # not a doc
    assert not rh.is_reader_facing("WHATEVER.md")    # root .md not on allowlist


def test_grade_letter_bands() -> None:
    assert rh.grade_letter(95) == "A"
    assert rh.grade_letter(61) == "D"
    assert rh.grade_letter(40) == "F"


def test_jaccard_and_shingles() -> None:
    a = "the quick brown fox jumps over the lazy dog and runs away fast today now"
    b = a  # identical text -> identical shingles -> jaccard 1.0
    assert rh.jaccard(rh.shingles(a), rh.shingles(b)) == 1.0
    c = "completely different words here nothing matches at all zero overlap whatsoever ok"
    assert rh.jaccard(rh.shingles(a), rh.shingles(c)) < 0.1


def test_shingles_ignore_code_fences() -> None:
    # fenced code is not prose, so it must not inflate similarity
    s = rh.shingles("# T\n\n```\nfunc main() { x := 1 }\n```\nplain words here only")
    assert all(isinstance(h, int) for h in s)


# --- verbosity -------------------------------------------------------------

def test_redundancy_flags_near_duplicate() -> None:
    text = " ".join(f"word{i}" for i in range(60))
    docs = [{"path": "a.md", "shingles": rh.shingles(text), "words": 60},
            {"path": "b.md", "shingles": rh.shingles(text), "words": 60}]
    c = rh.kpi_redundancy(docs)
    assert any("near-duplicate" in d for d in c["defects"]), c


def test_redundancy_clean_when_distinct() -> None:
    docs = [{"path": "a.md", "shingles": rh.shingles("alpha beta gamma delta epsilon zeta eta theta iota"),
             "words": 9},
            {"path": "b.md", "shingles": rh.shingles("one two three four five six seven eight nine ten"),
             "words": 10}]
    c = rh.kpi_redundancy(docs)
    assert c["defects"] == [], c


def test_bloat_flags_oversized_only() -> None:
    c = rh.kpi_bloat([{"path": "big.md", "n_lines": rh.DOC_HARD_LINES + 5},
                      {"path": "ok.md", "n_lines": 100},
                      {"path": "longish.md", "n_lines": rh.DOC_SOFT_LINES + 10}])
    assert any("big.md" in d for d in c["defects"]), c
    assert all("ok.md" not in d for d in c["defects"])
    assert any("longish.md" in s for s in c["soft"]), c


# --- organization ----------------------------------------------------------

def test_root_hygiene_flags_stray() -> None:
    c = rh.kpi_root_hygiene(["README.md", "RANDO.md"], ["go.mod", "err.txt"])
    assert any("RANDO.md" in d for d in c["defects"]), c
    assert any("err.txt" in d for d in c["defects"]), c
    assert all("README.md" not in d and "go.mod" not in d for d in c["defects"])


def test_placement_flags_misplaced_dated() -> None:
    c = rh.kpi_placement(["docs/gpu-parity-tracking-480.md"])
    assert c["defects"] and "480" in c["defects"][0], c


def test_dir_discipline_flags_near_dup_siblings() -> None:
    c = rh.kpi_dir_discipline(["docs/benchmark", "docs/benchmarks", "docs/benchmarking",
                               "docs/fak", "internal/model"])
    assert any("benchmark" in d for d in c["defects"]), c
    # distinct dirs are not flagged
    assert all("internal/model" not in d for d in c["defects"])


# --- indexing --------------------------------------------------------------

def test_index_presence_flags_missing() -> None:
    c = rh.kpi_index_presence({"INDEX.md": False, "llms.txt": True, "docs/index.md": True})
    assert any("INDEX.md" in d for d in c["defects"]), c
    assert c["score"] < 100


def test_index_integrity_flags_dead_entry() -> None:
    c = rh.kpi_index_integrity({"llms.txt": ["docs/gone.md"]})
    assert any("gone.md" in d for d in c["defects"]), c


def test_orphans_pct_and_defects() -> None:
    c = rh.kpi_orphans(["docs/lonely.md"], n_reader=4)
    assert any("lonely.md" in d for d in c["defects"]), c
    assert c["score"] == 75, c  # 3/4 indexed


def test_reachable_md_bfs() -> None:
    links = {"README.md": ["docs/a.md"], "docs/a.md": ["docs/b.md"], "docs/b.md": [],
             "docs/orphan.md": []}
    reached = rh.reachable_md(["README.md"], links)
    assert "docs/a.md" in reached and "docs/b.md" in reached
    assert "docs/orphan.md" not in reached


# --- accessibility ---------------------------------------------------------

def test_image_alt_defects_flags_empty_alt() -> None:
    # markdown: empty bracket alt is a defect; populated alt is clean
    md = "![](visuals/chart.png)\n\n![a real description](visuals/ok.png)"
    miss = rh.image_alt_defects(md)
    assert miss == ["visuals/chart.png"], miss


def test_image_alt_defects_handles_html_img_multiline() -> None:
    # an HTML <img> spanning lines with a real alt is clean; one without alt fails
    good = '<img\n  src="a.png"\n  alt="a chart of throughput"/>'
    bad = '<img src="b.png" width="100%"/>'
    assert rh.image_alt_defects(good) == []
    assert rh.image_alt_defects(bad) == ["b.png"]


def test_image_alt_defects_clean_when_all_described() -> None:
    txt = "![first](a.png) and [![badge alt](badge.svg)](https://x) and ![second](b.png)"
    assert rh.image_alt_defects(txt) == []


def test_alt_text_is_hard() -> None:
    c = rh.kpi_alt_text([{"path": "docs/x.md", "missing": ["a.png", "b.svg"]}])
    assert len(c["defects"]) == 2, c
    assert any("a.png" in d for d in c["defects"])
    assert c["score"] < 100


def test_alt_text_clean_when_no_missing() -> None:
    c = rh.kpi_alt_text([])
    assert c["defects"] == [] and c["score"] == 100, c


def test_ai_tells_are_hard() -> None:
    c = rh.kpi_ai_tells([{"path": "x.md", "hits": ["leverage", "in a nutshell"],
                          "emdash_over": 0}])
    assert len(c["defects"]) == 2, c
    assert any("leverage" in d for d in c["defects"])


def test_ai_tells_per_doc_cap() -> None:
    hits = ["leverage"] * (rh.AITELL_PER_DOC_CAP + 5)
    c = rh.kpi_ai_tells([{"path": "x.md", "hits": hits, "emdash_over": 3}])
    assert len(c["defects"]) == rh.AITELL_PER_DOC_CAP, c
    assert any("more AI-tells" in s for s in c["soft"])
    assert any("em-dash" in s for s in c["soft"])


def test_jargon_emits_no_hard_debt() -> None:
    c = rh.kpi_jargon(["docs/x.md: vDSO", "docs/x.md: RadixAttention"], n_reader=3)
    assert c["defects"] == [], c
    assert c["score"] < 100  # but it does score lower


def test_jargon_is_growth_invariant() -> None:
    # same per-doc rate over a bigger corpus -> same score (not mechanically lower)
    small = rh.kpi_jargon([f"d{i}.md: vDSO" for i in range(5)], n_reader=10)
    big = rh.kpi_jargon([f"d{i}.md: vDSO" for i in range(50)], n_reader=100)
    assert small["score"] == big["score"], (small, big)


def test_plain_language_is_soft() -> None:
    c = rh.kpi_plain_language(["dense reading-ease 12 (< 30): x.md"], n_dense=1,
                              n_acro_docs=0, n_idiom=0, n_reader=5)
    assert c["defects"] == [], c
    assert c["score"] < 100


def test_flesch_dense_text_scores_low() -> None:
    dense = ("Notwithstanding the aforementioned heterogeneous instrumentation, the "
             "reconfiguration necessitates comprehensive recalibration of the "
             "multidimensional optimization subsystem accordingly.")
    plain = "The cat sat on the mat. It was a good day. We ran fast. You can too."
    assert rh.flesch(dense) < rh.flesch(plain)


# --- fold + grader ---------------------------------------------------------

def _clean_kpi(name: str) -> dict:
    return {"kpi": name, "group": rh.KPI_GROUP[name], "score": 100,
            "detail": "clean", "defects": [], "soft": []}


def test_payload_clean_is_ok() -> None:
    kpis = [_clean_kpi(n) for n in rh.KPI_WEIGHTS]
    p = rh.build_payload(workspace=".", kpis=kpis)
    assert p["ok"] is True and p["verdict"] == "OK", p
    assert p["corpus"]["hygiene_debt"] == 0


def test_payload_counts_debt_by_group() -> None:
    kpis = [_clean_kpi(n) for n in rh.KPI_WEIGHTS]
    kpis[0]["defects"] = ["x", "y"]  # redundancy (verbosity)
    p = rh.build_payload(workspace=".", kpis=kpis)
    assert p["ok"] is False and p["corpus"]["hygiene_debt"] == 2, p
    assert p["corpus"]["debt_by_group"]["verbosity"] == 2, p


def test_a11y_debt_rolls_up_accessibility_hard() -> None:
    # a11y-debt is the accessibility group's HARD defects, broken out as a
    # first-class integer (issue #510) — a slice of hygiene_debt, never extra.
    kpis = [_clean_kpi(n) for n in rh.KPI_WEIGHTS]
    by = {k["kpi"]: k for k in kpis}
    by["alt_text"]["defects"] = ["img1", "img2"]
    by["ai_tells"]["defects"] = ["tell1"]
    by["root_hygiene"]["defects"] = ["stray"]  # a NON-accessibility defect
    p = rh.build_payload(workspace=".", kpis=kpis)
    c = p["corpus"]
    assert c["a11y_debt"] == 3, c          # alt_text(2) + ai_tells(1), not root_hygiene
    assert c["a11y_debt"] <= c["hygiene_debt"], c
    assert c["hygiene_debt"] == 4, c


def test_compare_reports_a11y_debt() -> None:
    base = {"corpus": {"hygiene_debt": 30, "score": 60, "a11y_debt": 8,
                       "debt_by_group": {g: 0 for g in rh.GROUPS}}}
    cur = {"corpus": {"hygiene_debt": 9, "score": 88, "a11y_debt": 2,
                      "debt_by_group": {g: 0 for g in rh.GROUPS}}}
    out = rh.render_compare(base, cur)
    assert "a11y-debt:    8 -> 2" in out, out


def test_compare_reports_3x() -> None:
    base = {"corpus": {"hygiene_debt": 30, "score": 60,
                       "debt_by_group": {g: 0 for g in rh.GROUPS}}}
    cur = {"corpus": {"hygiene_debt": 9, "score": 88,
                      "debt_by_group": {g: 0 for g in rh.GROUPS}}}
    out = rh.render_compare(base, cur)
    assert ">=3x" in out, out
    cur2 = {"corpus": {"hygiene_debt": 20, "score": 70,
                       "debt_by_group": {g: 0 for g in rh.GROUPS}}}
    assert "not yet 3x" in rh.render_compare(base, cur2)


# --- live smoke ------------------------------------------------------------

def test_live_collect() -> None:
    root = rh.repo_root()
    if not (root / "README.md").exists():
        return  # tolerant: not in the repo tree
    p = rh.collect(root)
    assert p["schema"] == rh.SCHEMA
    assert "hygiene_debt" in p["corpus"]
    assert set(p["corpus"]["debt_by_group"]) == set(rh.GROUPS)
    assert isinstance(p["kpis"], list) and len(p["kpis"]) == len(rh.KPI_WEIGHTS)


# --- self-contained runner (mirrors docs_scorecard_test.py) ----------------

def main() -> int:
    import inspect
    failures: list[str] = []

    def check(name: str, fn) -> None:
        try:
            fn()
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: unexpected {type(exc).__name__}: {exc}")

    tests = {n: f for n, f in globals().items()
             if n.startswith("test_") and callable(f)
             and not inspect.signature(f).parameters}
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
