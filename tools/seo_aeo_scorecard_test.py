#!/usr/bin/env python3
"""Tests for the SEO/AEO scorecard.

Drives the PURE per-KPI checks + front-matter parser + grader + site-level fold
with fixture strings (mostly no disk), then a tolerant live smoke that `collect`
folds the real published surfaces. Mirrors tools/docs_scorecard_test.py.

Run: `python tools/seo_aeo_scorecard_test.py`  (exit 0 = all pass),
or `python -m pytest tools/seo_aeo_scorecard_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seo_aeo_scorecard as sc  # noqa: E402


# --- front-matter parser ---------------------------------------------------

def test_fm_quoted_oneliner() -> None:
    fm = sc.parse_front_matter('---\ntitle: "Hello World"\n---\n# body\n')
    assert fm.get("title") == "Hello World", fm


def test_fm_bare_oneliner() -> None:
    fm = sc.parse_front_matter("---\ntitle: Hello World\n---\nbody")
    assert fm.get("title") == "Hello World", fm


def test_fm_folded_block_scalar() -> None:
    text = ("---\n"
            "title: T\n"
            "description: >-\n"
            "  one line\n"
            "  two line\n"
            "slug: x\n"
            "---\nbody")
    fm = sc.parse_front_matter(text)
    assert fm.get("description") == "one line two line", fm
    assert fm.get("title") == "T", fm


def test_fm_none_when_absent() -> None:
    assert sc.parse_front_matter("# no front matter\n\ntext") == {}


# --- title / description KPIs ----------------------------------------------

def test_title_missing_is_hard_defect() -> None:
    k = sc.kpi_title({})
    assert k["score"] == 0 and any("no front-matter title" in d for d in k["defects"]), k


def test_title_present_in_band_is_100() -> None:
    k = sc.kpi_title({"title": "A clear page title for fak"})
    assert k["score"] == 100 and not k["defects"], k


def test_title_too_long_is_soft_not_hard() -> None:
    long_title = ("fak the agent kernel: a default-deny permission gate fused with an "
                  "addressable bit-exact KV cache for self-hosted AI agent fleets")
    assert len(long_title) > sc.TITLE_MAX
    k = sc.kpi_title({"title": long_title})
    assert k["defects"] == [] and k["score"] < 100 and k["soft"], k


def test_description_missing_is_hard_defect() -> None:
    k = sc.kpi_description({})
    assert k["score"] == 0 and any("no front-matter description" in d for d in k["defects"]), k


def test_description_in_band_is_100() -> None:
    desc = ("fak is an agent kernel that gates every tool call a model makes and "
            "reuses its KV cache across turns for cheaper self-hosted agent fleets.")
    assert 70 <= len(desc) <= 160
    k = sc.kpi_description({"description": desc})
    assert k["score"] == 100 and not k["defects"], k


def test_description_thin_is_soft() -> None:
    k = sc.kpi_description({"description": "too short"})
    assert k["defects"] == [] and k["soft"], k


# --- headings --------------------------------------------------------------

def test_headings_missing_h1_is_hard() -> None:
    k = sc.kpi_headings("no title\n\n## section\n")
    assert any("no H1" in d for d in k["defects"]), k


def test_headings_clean_is_high() -> None:
    k = sc.kpi_headings("# Title\n\nintro.\n\n## A\n\ntext\n\n## B\n\ntext\n")
    assert not k["defects"] and k["score"] >= 90, k


def test_headings_skip_level_is_soft() -> None:
    k = sc.kpi_headings("# Title\n\n### deep\n\ntext\n")
    assert k["defects"] == [] and any("skips" in s for s in k["soft"]), k


def test_headings_ignores_front_matter() -> None:
    # A '---' front-matter fence must not be read as content; the real H1 counts.
    k = sc.kpi_headings('---\ntitle: "T"\n---\n# Real H1\n\ntext\n')
    assert not k["defects"], k


# --- links -----------------------------------------------------------------

def test_links_dead_is_hard(tmp_path: Path) -> None:
    k = sc.kpi_links("[x](nope/missing.md)", tmp_path, "docs/index.md")
    assert k["score"] < 100 and any("missing.md" in d for d in k["defects"]), k


def test_links_ignore_network(tmp_path: Path) -> None:
    k = sc.kpi_links("[w](https://x.io) [a](#sec)", tmp_path, "docs/index.md")
    assert not k["defects"], k


# --- answerability (never hard-fails) --------------------------------------

def test_answerability_no_hard_defects() -> None:
    k = sc.kpi_answerability("# T\n\n## only scaffolding\n\n| a | b |\n")
    assert k["defects"] == [], k


def test_answerability_prose_opener_scores_well() -> None:
    good = sc.kpi_answerability("# T\n\nfak is an agent kernel that gates tool calls.\n\n## A\n")
    bad = sc.kpi_answerability("# T\n\n## A\n\n```code```\n")
    assert good["score"] > bad["score"], (good, bad)


# --- per-page fold + grader ------------------------------------------------

def test_score_page_perfect(tmp_path: Path) -> None:
    # The page lives at docs/a.md, so a link to "b.md" resolves to docs/b.md.
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "b.md").write_text("x", encoding="utf-8")
    desc = ("A clear plain-language description of what this fak page covers, with "
            "the primary keywords a searcher would use, kept inside the length band.")
    body = ('---\ntitle: "A solid page title here for fak"\n'
            'description: "' + desc + '"\n---\n'
            "# Title\n\nfak is a thing that does a thing, explained in plain words.\n\n"
            "## A\n\n[x](b.md)\n")
    d = sc.score_page(body, "docs/a.md", tmp_path)
    assert d["n_defects"] == 0 and d["grade"] == "A", d


def test_missing_page_is_worst() -> None:
    d = sc.missing_page_entry("docs/gone.md")
    assert d["score"] == 0.0 and d["grade"] == "F" and d["n_defects"] == 1, d


def test_grade_bands() -> None:
    assert sc.grade_letter(95) == "A" and sc.grade_letter(40) == "F"


# --- published-set enumeration ---------------------------------------------

def test_published_excludes_releases_and_nonmd() -> None:
    assert sc._published(Path(), "docs/releases/v0.1.0.md") is False
    assert sc._published(Path(), "docs/index.md") is True
    assert sc._published(Path(), "docs/explainers/x.md") is True
    assert sc._published(Path(), "docs/showcase.html") is False
    assert sc._published(Path(), "README.md") is False


# --- site-level checks -----------------------------------------------------

def test_site_flags_missing_jsonld_and_llms_full(tmp_path: Path) -> None:
    # A bare repo: no config, no JSON-LD, no llms-full -> several HARD site defects.
    (tmp_path / "docs").mkdir()
    site = sc.site_checks(tmp_path)
    names = {c["name"] for c in site["checks"] if not c["ok"] and c["hard"]}
    assert "jsonld_SoftwareApplication" in names, site
    assert "jsonld_FAQPage" in names, site
    assert "llms_full" in names, site


def test_site_detects_jsonld_when_present(tmp_path: Path) -> None:
    inc = tmp_path / "docs" / "_includes"
    inc.mkdir(parents=True)
    (inc / "head-custom.html").write_text(
        '<script type="application/ld+json">{"@type":"SoftwareApplication"}</script>\n'
        '<script type="application/ld+json">{"@type":"WebSite"}</script>\n',
        encoding="utf-8")
    site = sc.site_checks(tmp_path)
    assert "SoftwareApplication" in site["present_jsonld"], site
    assert "WebSite" in site["present_jsonld"], site
    soft_or_ok = {c["name"]: c["ok"] for c in site["checks"]}
    assert soft_or_ok["jsonld_SoftwareApplication"] is True, site


def test_site_llms_full_freshness(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "llms.txt").write_text("Key facts: the substrate", encoding="utf-8")
    # Fresh = llms-full CONTAINS the current llms.txt body (gen inlines it verbatim).
    (tmp_path / "llms-full.txt").write_text(
        "# corpus\n\n## Index\n\nKey facts: the substrate\n\n--- more docs ---",
        encoding="utf-8")
    site = sc.site_checks(tmp_path)
    by = {c["name"]: c for c in site["checks"]}
    # llms.txt has a Key-facts block; llms-full inlines it -> both should be ok.
    assert by["llms_txt"]["ok"] is True, by["llms_txt"]
    assert by["llms_full"]["ok"] is True, by["llms_full"]


def test_site_llms_full_stale_by_content(tmp_path: Path) -> None:
    # llms-full exists but does NOT contain the current llms.txt body -> STALE.
    (tmp_path / "docs").mkdir()
    (tmp_path / "llms.txt").write_text("Key facts: the NEW substrate line", encoding="utf-8")
    (tmp_path / "llms-full.txt").write_text("old corpus without the new line", encoding="utf-8")
    by = {c["name"]: c for c in sc.site_checks(tmp_path)["checks"]}
    assert by["llms_full"]["ok"] is False, by["llms_full"]


# --- payload + compare -----------------------------------------------------

def test_payload_clean_is_ok() -> None:
    pages = [{"path": "docs/a.md", "score": 100.0, "grade": "A", "n_defects": 0,
              "defects": [], "soft": [], "kpis": {"title": 100, "description": 100}}]
    site = {"checks": [], "score": 100.0, "n_ok": 13, "n_total": 13,
            "defects": [], "soft": [], "present_jsonld": ["SoftwareApplication"]}
    p = sc.build_payload(workspace=".", pages=pages, site=site, scope="core")
    assert p["ok"] is True and p["verdict"] == "OK", p


def test_payload_counts_seo_debt() -> None:
    pages = [{"path": "docs/a.md", "score": 40.0, "grade": "F", "n_defects": 2,
              "defects": ["x", "y"], "soft": [], "kpis": {"title": 0, "description": 0}}]
    site = {"checks": [], "score": 60.0, "n_ok": 8, "n_total": 13,
            "defects": ["no JSON-LD FAQPage"], "soft": [], "present_jsonld": []}
    p = sc.build_payload(workspace=".", pages=pages, site=site, scope="core")
    assert p["ok"] is False and p["corpus"]["seo_debt"] == 3, p
    assert p["corpus"]["seo_debt_in_pages"] == 2 and p["corpus"]["seo_debt_in_site"] == 1, p
    assert p["corpus"]["meta_coverage_pct"] == 0.0, p


def test_compare_reports_10x() -> None:
    base = {"corpus": {"seo_debt": 27, "overall_score": 53.0, "meta_coverage_pct": 25.0,
                       "site_checks_ok": "7/13", "present_jsonld": []}}
    cur = {"corpus": {"seo_debt": 2, "overall_score": 95.0, "meta_coverage_pct": 100.0,
                      "site_checks_ok": "13/13", "present_jsonld": ["SoftwareApplication", "FAQPage"]}}
    out = sc.render_compare(base, cur)
    assert ">=10x" in out, out


def test_compare_not_yet_10x() -> None:
    base = {"corpus": {"seo_debt": 27, "overall_score": 53.0}}
    cur = {"corpus": {"seo_debt": 10, "overall_score": 70.0}}
    out = sc.render_compare(base, cur)
    assert "not yet 10x" in out, out


# --- live smoke ------------------------------------------------------------

def test_live_collect_core() -> None:
    root = sc.repo_root()
    if not (root / "docs" / "index.md").exists():
        return  # tolerant: not in the repo tree
    p = sc.collect(root, scope="core")
    assert p["schema"] == sc.SCHEMA
    # The core set is DERIVED (reachable discovery), so n_pages == the derivation.
    assert p["corpus"]["n_pages"] == len(sc.enumerate_pages(root, "core"))
    assert isinstance(p["pages"], list) and p["pages"]
    assert "discovery_orphans" in p["corpus"]


def test_live_collect_published_is_superset() -> None:
    root = sc.repo_root()
    if not (root / "docs" / "index.md").exists():
        return
    p = sc.collect(root, scope="published")
    assert p["scope"] == "published"
    # the full indexable tree is a superset of the reachable discovery subset
    assert p["corpus"]["n_pages"] >= len(sc.enumerate_pages(root, "core"))


# --- adversarial / anti-gaming tests (each guards a flagged vector) --------

def test_degenerate_title_is_hard_defect() -> None:
    # 'x'*100 is "present" but useless — must NOT score a clean 100.
    k = sc.kpi_title({"title": "x" * 100})
    assert k["score"] == 0 and any("degenerate" in d for d in k["defects"]), k


def test_degenerate_description_is_hard_defect() -> None:
    k = sc.kpi_description({"description": "." * 120})
    assert k["score"] == 0 and any("degenerate" in d for d in k["defects"]), k


def test_real_short_title_not_degenerate() -> None:
    # a real two-word title is not degenerate (only filler/repeats are)
    k = sc.kpi_title({"title": "fak FAQ"})
    assert not k["defects"], k


def test_malformed_jsonld_not_counted(tmp_path: Path) -> None:
    inc = tmp_path / "docs" / "_includes"
    inc.mkdir(parents=True)
    (inc / "head-custom.html").write_text(
        '<script type="application/ld+json">{ "@type": "SoftwareApplication" BROKEN }</script>\n',
        encoding="utf-8")
    site = sc.site_checks(tmp_path)
    by = {c["name"]: c for c in site["checks"]}
    # the broken block must NOT contribute a phantom type, and must trip jsonld_valid
    assert "SoftwareApplication" not in site["present_jsonld"], site
    assert by["jsonld_valid"]["ok"] is False, by["jsonld_valid"]


def test_valid_jsonld_array_and_graph(tmp_path: Path) -> None:
    inc = tmp_path / "docs" / "_includes"
    inc.mkdir(parents=True)
    (inc / "head-custom.html").write_text(
        '<script type="application/ld+json">{"@graph":[{"@type":"WebSite"},{"@type":"Organization"}]}</script>',
        encoding="utf-8")
    site = sc.site_checks(tmp_path)
    assert "WebSite" in site["present_jsonld"] and "Organization" in site["present_jsonld"], site


def test_faq_nonquestion_h2_not_counted(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "FAQ.md").write_text(
        "# FAQ\n\n" + "\n".join(f"## Section {i}\n\ntext\n" for i in range(8)),
        encoding="utf-8")
    by = {c["name"]: c for c in sc.site_checks(tmp_path)["checks"]}
    assert by["faq_structured"]["ok"] is False, by["faq_structured"]


def test_discovery_excludes_evidence_subtrees() -> None:
    assert sc._discovery("docs/fak/server-quickstart.md") is True
    assert sc._discovery("docs/proofs/policy.md") is False
    assert sc._discovery("docs/benchmarks/TURN-TAX-RESULTS.md") is False
    assert sc._discovery("docs/notes/x.md") is False
    assert sc._discovery("docs/releases/v0.1.0.md") is False  # also non-published


def test_reachable_and_orphan_detection(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("see [a](docs/a.md)", encoding="utf-8")
    (tmp_path / "docs" / "a.md").write_text("# A\n\n[b](b.md)", encoding="utf-8")
    (tmp_path / "docs" / "b.md").write_text("# B", encoding="utf-8")
    (tmp_path / "docs" / "orphan.md").write_text("# Orphan, linked from nowhere", encoding="utf-8")
    reach = sc.reachable_published(tmp_path)
    assert "docs/a.md" in reach and "docs/b.md" in reach, reach
    orphans = sc.discovery_orphans(tmp_path)
    assert "docs/orphan.md" in orphans, orphans
    assert "docs/a.md" not in orphans, orphans


def test_links_skip_fenced_code(tmp_path: Path) -> None:
    # an illustrative path inside a code fence is not a real dead link
    body = "see real prose\n\n```\nrm docs/does-not-exist.md\n```\n"
    k = sc.kpi_links(body, tmp_path, "docs/index.md")
    assert k["defects"] == [], k


# --- self-contained runner -------------------------------------------------

def main() -> int:
    import inspect
    import tempfile
    failures: list[str] = []

    def check(name: str, fn) -> None:
        try:
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
