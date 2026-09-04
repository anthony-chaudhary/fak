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
import json
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


def _hardware_front_page(date: str = TODAY.isoformat(), *, extra_row: str = "") -> str:
    return f"""## Latest hardware results — {date}

**Latest** means the newest committed performance receipt for that platform.

| Platform | Latest result | Qualification and detail |
| :--- | :--- | :--- |
| Mac | result observed 2026-06-18. | Historical/expired review. [Detail](docs/benchmarks/mac.md) |
| AMD | result observed 2026-06-19. | Narrow microbench only. [Detail](docs/benchmarks/amd.md) |
| NVIDIA | result captured 2026-06-20. | Held after failed cache quality. [Detail](docs/benchmarks/nvidia.md) |
{extra_row}
History and specific envelopes live in the [benchmark index](docs/benchmarks/README.md).
"""


def _hardware_manifest(date: str = TODAY.isoformat(), *, platform: str | None = None,
                       row_suffix: str = "") -> str:
    readme = _hardware_front_page(date)
    rows = {line.split("|", 2)[1].strip(): line for line in readme.splitlines()
            if line.startswith(("| Mac |", "| AMD |", "| NVIDIA |"))}
    if platform:
        rows[platform] += row_suffix
    return json.dumps({
        "schema": rfa.HARDWARE_LATEST_SCHEMA,
        "as_of": date,
        "platforms": {
            "Mac": {"observed": "2026-06-18", "detail": "docs/benchmarks/mac.md", "row": rows["Mac"]},
            "AMD": {"observed": "2026-06-19", "detail": "docs/benchmarks/amd.md", "row": rows["AMD"]},
            "NVIDIA": {"observed": "2026-06-20", "detail": "docs/benchmarks/nvidia.md", "row": rows["NVIDIA"]},
        },
    })


def _hardware_root(tmp_path: Path) -> Path:
    for name in ("mac.md", "amd.md", "nvidia.md", "README.md"):
        path = tmp_path / "docs" / "benchmarks" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("receipt\n", encoding="utf-8")
    return tmp_path


def _check_hardware(tmp_path: Path, readme: str | None = None,
                    manifest: str | None = None) -> dict[str, object]:
    return rfa.check_hardware_front_page(
        readme or _hardware_front_page(),
        _hardware_manifest() if manifest is None else manifest,
        _hardware_root(tmp_path), today=TODAY, max_age_days=14)


def test_hardware_front_page_accepts_exact_three_rows(tmp_path: Path) -> None:
    assert _check_hardware(tmp_path)["status"] == "OK"


def test_hardware_front_page_rejects_stale_date(tmp_path: Path) -> None:
    date = "2026-06-01"
    c = _check_hardware(tmp_path, _hardware_front_page(date), _hardware_manifest(date))
    assert c["status"] == "FAIL" and "fresh_date" in c["items"]


def test_hardware_front_page_rejects_extra_result_row(tmp_path: Path) -> None:
    c = _check_hardware(tmp_path, _hardware_front_page(
        extra_row="| Ultracode | result | hold |"))
    assert c["status"] == "FAIL"
    assert "README must contain no extra result rows" in c["items"]


def test_hardware_front_page_rejects_missing_platform(tmp_path: Path) -> None:
    readme = _hardware_front_page().replace(
        "| AMD | result observed 2026-06-19. | Narrow microbench only. [Detail](docs/benchmarks/amd.md) |\n", "")
    c = _check_hardware(tmp_path, readme)
    assert c["status"] == "FAIL"
    assert "README must contain exactly Mac, AMD, NVIDIA rows" in c["items"]
    assert any(item.startswith("AMD:") for item in c["items"])


def test_hardware_front_page_rejects_generated_qwen_marker(tmp_path: Path) -> None:
    c = _check_hardware(tmp_path, _hardware_front_page() + "\n<!-- qwen38-frontdoor:begin -->\n")
    assert c["status"] == "FAIL"
    assert "README must not contain generated qwen38-frontdoor markers" in c["items"]


def test_hardware_front_page_rejects_each_platform_manifest_drift(tmp_path: Path) -> None:
    for platform in ("Mac", "AMD", "NVIDIA"):
        c = _check_hardware(tmp_path, manifest=_hardware_manifest(platform=platform, row_suffix=" updated"))
        assert c["status"] == "FAIL", platform
        assert platform in c["detail"]
        assert rfa.HARDWARE_LATEST_REL in c["detail"] and rfa.README_REL in c["detail"]


def test_hardware_front_page_rejects_heading_manifest_date_drift(tmp_path: Path) -> None:
    c = _check_hardware(tmp_path, manifest=_hardware_manifest("2026-06-19"))
    assert c["status"] == "FAIL"
    assert "heading date differs" in c["detail"]


def test_hardware_front_page_rejects_missing_or_malformed_manifest(tmp_path: Path) -> None:
    for manifest in ("", "{not json"):
        c = _check_hardware(tmp_path, manifest=manifest)
        assert c["status"] == "FAIL"
        assert rfa.HARDWARE_LATEST_REL in c["detail"] and rfa.README_REL in c["detail"]


def test_hardware_front_page_rejects_extra_manifest_platform(tmp_path: Path) -> None:
    manifest = json.loads(_hardware_manifest())
    manifest["platforms"]["Intel"] = manifest["platforms"]["AMD"]
    c = _check_hardware(tmp_path, manifest=json.dumps(manifest))
    assert c["status"] == "FAIL"
    assert "manifest must contain exactly Mac, AMD, NVIDIA" in c["items"]


def test_hardware_front_page_rejects_missing_detail_receipt(tmp_path: Path) -> None:
    manifest = json.loads(_hardware_manifest())
    manifest["platforms"]["Mac"]["detail"] = "docs/benchmarks/missing.md"
    c = _check_hardware(tmp_path, manifest=json.dumps(manifest))
    assert c["status"] == "FAIL"
    assert "Mac" in c["detail"] and "does not exist" in c["detail"]


def test_standard_gates_run_actual_readme_audit() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = "python3 tools/readme_freshness_audit.py --json"
    assert expected in (root / "Makefile").read_text(encoding="utf-8")
    assert "python tools/readme_freshness_audit.py --json" in (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")


def _qwen_result_doc(peer: str) -> str:
    return f"""# result
<!-- qwen38-frontdoor:begin -->
generated
<!-- qwen38-frontdoor:end -->
[{peer}]({peer})
"""


def test_qwen_result_docs_accept_reciprocal_generated_pages() -> None:
    c = rfa.check_qwen_result_docs(
        _qwen_result_doc("QWEN38-27B-LATEST.md"),
        _qwen_result_doc("QWEN-PERFORMANCE-INDEX.md"),
    )
    assert c["status"] == "OK"


def test_qwen_result_docs_reject_missing_block_or_route() -> None:
    c = rfa.check_qwen_result_docs(
        "# index without generated block",
        _qwen_result_doc("elsewhere.md"),
    )
    assert c["status"] == "FAIL"
    assert "index_generated_block" in c["items"]
    assert "latest_to_index_link" in c["items"]


_SHOWCASE_OK = """<!doctype html>
<!--
  Hand-authored homepage. Re-sync the <script> block below when the benchmark
  data changes.  (This mention of a script tag INSIDE a comment is the ordering
  trap html_text exists to survive - see the regression test at the bottom.)
-->
<html><head><style>body { color: #000 }</style></head><body>
<h1>fak</h1>
<pre><code>curl -fsSL https://example.invalid/install.sh | sh
fak agent --offline</code></pre>
<p>Every number traces to <a href="BENCHMARK-AUTHORITY.md">the benchmark
authority</a>; the hero run was measured at fak v0.30.0.</p>
<script>renderChart({series: [1, 2, 3]});</script>
</body></html>
"""

_README_OK = """# fak

Install the one binary and run one deterministic check:

    curl -fsSL https://example.invalid/install.sh | sh
    fak agent --offline

Show it off: [the published showcase](docs/showcase.html) is the configured
homepage. Numbers live in BENCHMARK-AUTHORITY.md.
"""

_SHOWCASE_KW = dict(version="0.43.0", dataset_versions={"0.30.0"},
                    dispatch={"agent", "guard", "serve"})



def test_showcase_sync_ok_when_front_doors_agree() -> None:
    c = rfa.check_showcase_sync(_README_OK, _SHOWCASE_OK, **_SHOWCASE_KW)
    assert c["status"] == "OK" and c["score"] == 1.0, c
    assert not c["items"], c


def test_showcase_sync_fails_when_readme_does_not_link_it() -> None:
    # The #5476 defect shape: the page is fine, nothing points at it.
    readme = _README_OK.replace("[the published showcase](docs/showcase.html)",
                                "the demos")
    c = rfa.check_showcase_sync(readme, _SHOWCASE_OK, **_SHOWCASE_KW)
    assert c["status"] == "FAIL" and "linked_from_readme" in c["items"], c


def test_showcase_sync_fails_when_install_command_drifts() -> None:
    # Rename the installer on one door only: the homepage now sends a visitor at
    # a script the README does not stand behind.
    showcase = _SHOWCASE_OK.replace("install.sh", "get-fak-v2.sh")
    c = rfa.check_showcase_sync(_README_OK, showcase, **_SHOWCASE_KW)
    assert c["status"] == "FAIL" and "install_matches" in c["items"], c
    assert "get-fak-v2.sh" in c["detail"], c


def test_showcase_sync_fails_when_first_run_verb_is_fabricated() -> None:
    # Anti-gaming, applied to the homepage: a first command the binary does not
    # dispatch earns nothing, exactly as in lcd_onramp.
    showcase = _SHOWCASE_OK.replace("fak agent --offline", "fak totally-made-up")
    c = rfa.check_showcase_sync(_README_OK, showcase, **_SHOWCASE_KW)
    assert c["status"] == "FAIL" and "first_run_matches" in c["items"], c


def test_showcase_sync_fails_when_a_real_verb_is_not_taught_by_the_readme() -> None:
    # A real verb that only ONE door teaches is still a divergence.
    showcase = _SHOWCASE_OK.replace("fak agent --offline", "fak serve --addr :8080")
    c = rfa.check_showcase_sync(_README_OK, showcase, **_SHOWCASE_KW)
    assert c["status"] == "FAIL" and "first_run_matches" in c["items"], c


def test_showcase_sync_first_run_abstains_without_a_dispatch_set() -> None:
    # No cmd/fak/main.go to parse (run outside the repo): fall back to "both
    # doors name the same command" rather than inventing a defect.
    c = rfa.check_showcase_sync(_README_OK, _SHOWCASE_OK,
                                version="0.43.0", dataset_versions={"0.30.0"})
    assert "first_run_matches" not in c["items"], c


def test_showcase_sync_fails_when_authority_link_is_dropped() -> None:
    showcase = _SHOWCASE_OK.replace("BENCHMARK-AUTHORITY.md", "our internal notes")
    c = rfa.check_showcase_sync(_README_OK, showcase, **_SHOWCASE_KW)
    assert c["status"] == "FAIL" and "authority_linked" in c["items"], c


def test_showcase_sync_fails_on_a_version_nothing_backs() -> None:
    # The drift the check is really for: the dataset gets regenerated (or the
    # caption hand-edited) and the homepage quotes a version no artifact stamps.
    showcase = _SHOWCASE_OK.replace("fak v0.30.0", "fak v0.9.9")
    c = rfa.check_showcase_sync(_README_OK, showcase, **_SHOWCASE_KW)
    assert c["status"] == "FAIL" and "version_traced" in c["items"], c
    assert "0.9.9" in c["detail"], c


def test_showcase_sync_accepts_a_dataset_stamped_as_of_version() -> None:
    # The mirror image: v0.30.0 != VERSION 0.43.0, but a dataset stands behind it
    # as the version the run was MEASURED at, so quoting it is honest.
    c = rfa.check_showcase_sync(_README_OK, _SHOWCASE_OK, **_SHOWCASE_KW)
    assert "version_traced" not in c["items"], c
    bare = rfa.check_showcase_sync(_README_OK, _SHOWCASE_OK, version="0.43.0",
                                   dispatch={"agent"})
    assert "version_traced" in bare["items"], bare


def test_showcase_sync_abstains_when_there_is_no_page() -> None:
    c = rfa.check_showcase_sync(_README_OK, None, **_SHOWCASE_KW)
    assert c["status"] == "WARN" and "showcase" in c["detail"], c


def test_showcase_sync_fails_on_an_empty_page() -> None:
    # An empty page is a real defect, NOT the same as "no page to read".
    c = rfa.check_showcase_sync(_README_OK, "", **_SHOWCASE_KW)
    assert c["status"] == "FAIL", c


def test_html_text_survives_a_script_mention_inside_a_comment() -> None:
    # REGRESSION GUARD, and the reason html_text strips comments first: the real
    # docs/showcase.html header comment contains the literal string "<script>",
    # so reducing script/style FIRST matches from inside that comment to the
    # page's real </script> and swallows the whole page. An empty extraction
    # makes every cross-check above vacuously true - a silent false GREEN.
    text = rfa.html_text(_SHOWCASE_OK)
    assert "fak agent --offline" in text, text[:200]
    assert "benchmark" in text, "reader-visible prose after the comment was lost"
    assert "Hand-authored homepage" not in text, "comment text leaked into reader text"
    assert "renderChart" not in text, "script body leaked into reader text"
    assert "color: #000" not in text, "style body leaked into reader text"


def test_bench_dataset_versions_reads_the_stamped_as_of(tmp_path: Path) -> None:
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "hero_benchmark.data.json").write_text(
        '{"meta": {"fak_version": "0.30.0"}}', encoding="utf-8")
    (tmp_path / "tools" / "other.data.json").write_text(
        '{"meta": {"fak_version": "v0.31.2"}}', encoding="utf-8")
    assert rfa.bench_dataset_versions(tmp_path) == {"0.30.0", "0.31.2"}


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


# --- front_page_focus (the size-law counterweight) -------------------------

def test_front_page_focus_ok_when_lean() -> None:
    txt = ("# fak\n"
           "> **one static Go binary in front of the agent you already run.**\n\n"
           "It routes, checks, and reuses. See the [overflow page](docs/README-legacy.md).\n"
           "## Install\n## Docs\n")
    c = rfa.check_front_page_focus(txt, line_budget=250, section_budget=12, max_lead=2)
    assert c["score"] == 1.0 and c["status"] == "OK", c


def test_front_page_focus_flags_triple_lead() -> None:
    # The pitch restated 3x in the preamble (above the first `## `) is the
    # regrowth pattern. A restatement inside a section body must NOT count.
    txt = ("# fak\n"
           "a single Go binary you drop in front of the agent you already run.\n"
           "one static Go binary sits in front of an agent's calls.\n"
           "use one binary with the agent you already run.\n"
           "## Get started\nwrap the agent you already run in one command.\n")
    c = rfa.check_front_page_focus(txt, line_budget=250, section_budget=12, max_lead=2)
    assert "single_lead" in c["items"], c
    assert c["score"] < 1.0, c


def test_front_page_focus_flags_over_line_budget() -> None:
    txt = "# fak\n" + "\n".join(f"line {i}" for i in range(60))
    c = rfa.check_front_page_focus(txt, line_budget=40, section_budget=12, max_lead=2)
    assert "within_line_budget" in c["items"], c


def test_front_page_focus_flags_section_sprawl() -> None:
    txt = "# fak\n" + "\n".join(f"## Section {i}" for i in range(20))
    c = rfa.check_front_page_focus(txt, line_budget=250, section_budget=12, max_lead=2)
    assert "sections_bounded" in c["items"], c


def test_front_page_focus_only_counts_top_level_sections() -> None:
    # `### ` subsections are not section sprawl; only `## ` counts.
    txt = "# fak\n## One\n### a\n### b\n### c\n## Two\n"
    c = rfa.check_front_page_focus(txt, line_budget=250, section_budget=12, max_lead=2)
    assert "sections_bounded" not in c["items"], c


def test_front_page_focus_exposes_overage_magnitude() -> None:
    # The check surfaces HOW FAR over (not just a boolean) so the score can weight
    # it: a preamble with a thrice-restated lead, over a tiny line budget.
    readme = ("# fak\none binary in front of the agent you already run.\n"
              "one static Go binary in front of the agent you already run.\n"
              "drop-in one binary in front of the agent you already run.\n"
              + "\n".join(f"x{i}" for i in range(20)))
    c = rfa.check_front_page_focus(readme, line_budget=10, section_budget=12, max_lead=2)
    assert c["lines_over"] == len(readme.splitlines()) - 10, c
    assert c["lead_over"] >= 1, c
    assert "lines_over" in c and "overflow_linked" in c, c


# --- the UNBOUNDED, magnitude-aware composite score ------------------------

def _focus_check(*, lines_over=0, lines_under=0, sections_over=0, lead_over=0,
                 overflow_linked=True) -> dict:
    """A front_page_focus check dict carrying explicit overage, for score tests.

    ``lines_under`` is the UNDER-budget headroom that feeds the leanness credit
    (mutually exclusive with lines_over: a page is either over or under budget).
    """
    missing = []
    if lines_over:
        missing.append("within_line_budget")
    if sections_over:
        missing.append("sections_bounded")
    if lead_over:
        missing.append("single_lead")
    if not overflow_linked:
        missing.append("overflow_linked")
    met = 4 - len(missing)
    return {"check": "front_page_focus", "status": "OK" if met >= 3 else "WARN",
            "score": round(met / 4, 3), "items": missing,
            "lines_over": lines_over, "lines_under": lines_under,
            "sections_over": sections_over,
            "lead_over": lead_over, "overflow_linked": overflow_linked}


def _all_substance_ok() -> list[dict]:
    """The five non-focus substance checks maxed (1.0) + a lean focus (0 penalty)."""
    return [{"check": c, "status": "OK", "score": 1.0}
            for c in rfa.SUBSTANCE_CHECKS if c != "front_page_focus"] + [_focus_check()]


def test_clean_lean_page_scores_100() -> None:
    p = rfa.build_payload(workspace=".", checks=_all_substance_ok())
    assert p["score"] == 100 and p["grade"] == "A", p


def test_score_is_unbounded_below_on_gross_bloat() -> None:
    # A page 250 lines over budget scores far BELOW zero — the old score floored
    # at 0, hiding the difference between a little and a lot of bloat.
    checks = _all_substance_ok()[:-1] + [_focus_check(lines_over=250)]
    p = rfa.build_payload(workspace=".", checks=checks)
    assert p["score"] < 0, p
    assert p["score"] == round(100 - 250 * rfa.LINE_OVER_PENALTY), p


def test_score_is_magnitude_aware_lines_over() -> None:
    # 5 lines over vs 200 lines over differ by exactly the overage delta — the
    # boolean within_line_budget used to make them identical (both = one defect).
    small = rfa.build_payload(
        workspace=".", checks=_all_substance_ok()[:-1] + [_focus_check(lines_over=5)])
    big = rfa.build_payload(
        workspace=".", checks=_all_substance_ok()[:-1] + [_focus_check(lines_over=200)])
    assert big["score"] < small["score"], (big["score"], small["score"])
    assert small["score"] - big["score"] == round(195 * rfa.LINE_OVER_PENALTY)


def test_second_hygiene_fail_drops_below_the_cap() -> None:
    # One FAIL lands exactly on the old cap; a second takes the score below it,
    # unbounded — a broken page can no longer hide near a passing grade.
    one = rfa.build_payload(
        workspace=".", checks=_all_substance_ok() + [{"check": "links", "status": "FAIL"}])
    two = rfa.build_payload(
        workspace=".", checks=_all_substance_ok()
        + [{"check": "links", "status": "FAIL"}, {"check": "version_pins", "status": "FAIL"}])
    assert one["score"] == rfa.FAIL_SCORE_CAP, one
    assert two["score"] < rfa.FAIL_SCORE_CAP, two
    assert two["score"] == round(100 - 2 * rfa.HYGIENE_FAIL_PENALTY), two


def test_excess_section_and_lead_outweigh_a_single_line() -> None:
    # Magnitude weights: an extra section (8) and an extra lead (10) each cost
    # more than one line over budget (1) — sprawl and confusion bite harder.
    line = rfa.build_payload(
        workspace=".", checks=_all_substance_ok()[:-1] + [_focus_check(lines_over=1)])
    section = rfa.build_payload(
        workspace=".", checks=_all_substance_ok()[:-1] + [_focus_check(sections_over=1)])
    lead = rfa.build_payload(
        workspace=".", checks=_all_substance_ok()[:-1] + [_focus_check(lead_over=1)])
    assert section["score"] < line["score"], (section["score"], line["score"])
    assert lead["score"] < section["score"], (lead["score"], section["score"])


# --- the UNBOUNDED-ABOVE leanness credit (no fake-perfect 100 ceiling) ------

def test_score_is_unbounded_above_on_lean_complete_page() -> None:
    # A COMPLETE page 180 lines under budget scores far ABOVE 100 — 100 is the
    # at-budget zero-point, not a ceiling, so a leaner page is not "fake perfect".
    checks = _all_substance_ok()[:-1] + [_focus_check(lines_under=180)]
    p = rfa.build_payload(workspace=".", checks=checks)
    assert p["score"] > 100, p
    assert p["score"] == round(100 + 180 * rfa.LINE_UNDER_CREDIT), p
    assert p["finding"] == "readme_fresh", p  # complete + lean = the top verdict


def test_leaner_complete_page_scores_strictly_higher() -> None:
    # The score keeps MOVING as a complete page gets tighter — no plateau. Every
    # trimmed line is worth exactly LINE_UNDER_CREDIT.
    lean = rfa.build_payload(
        workspace=".", checks=_all_substance_ok()[:-1] + [_focus_check(lines_under=50)])
    leaner = rfa.build_payload(
        workspace=".", checks=_all_substance_ok()[:-1] + [_focus_check(lines_under=90)])
    assert leaner["score"] > lean["score"], (leaner["score"], lean["score"])
    assert leaner["score"] - lean["score"] == round(40 * rfa.LINE_UNDER_CREDIT)


def test_leanness_credit_scales_with_completeness() -> None:
    # Leanness only counts in proportion to completeness: the same headroom earns
    # LESS on a page missing an affordance, so you cannot score high by deleting
    # content — only by saying the same complete thing in fewer lines.
    full = _all_substance_ok()[:-1] + [_focus_check(lines_under=100)]
    partial = (
        [{"check": c, "status": "OK", "score": 1.0}
         for c in rfa.SUBSTANCE_CHECKS if c not in ("front_page_focus", "hero_above_fold")]
        + [{"check": "hero_above_fold", "status": "WARN", "score": 0.5,
            "items": ["sota_framed"]}]
        + [_focus_check(lines_under=100)])
    p_full = rfa.build_payload(workspace=".", checks=full)
    p_partial = rfa.build_payload(workspace=".", checks=partial)
    assert p_partial["score"] < p_full["score"], (p_partial["score"], p_full["score"])
    assert p_full["score"] > 100 and p_partial["score"] > 100, (p_full, p_partial)


def test_hygiene_fail_voids_the_leanness_credit() -> None:
    # A lean but BROKEN page earns no concision reward — a dead link voids the
    # credit, so it lands on the FAIL cap exactly like a bloated broken page.
    checks = (_all_substance_ok()[:-1] + [_focus_check(lines_under=180)]
              + [{"check": "links", "status": "FAIL", "detail": "dead link"}])
    p = rfa.build_payload(workspace=".", checks=checks)
    assert p["score"] == rfa.FAIL_SCORE_CAP, p  # 100 - one FAIL, credit voided
    assert p["ok"] is False, p


def test_lean_but_incomplete_page_is_still_flagged_thin() -> None:
    # The credit can lift an incomplete page's score above 100, but the verdict is
    # keyed to real completeness, not the score: a missing affordance is THIN even
    # when the number looks great. (This is what the old score<90 test missed.)
    checks = (
        [{"check": c, "status": "OK", "score": 1.0}
         for c in rfa.SUBSTANCE_CHECKS if c not in ("front_page_focus", "hero_above_fold")]
        + [{"check": "hero_above_fold", "status": "WARN", "score": 0.5,
            "items": ["sota_framed"]}]
        + [_focus_check(lines_under=180)])
    p = rfa.build_payload(workspace=".", checks=checks)
    assert p["score"] > 100, p
    assert p["finding"] == "readme_fresh_thin", p


def test_complete_but_overbudget_is_notes_not_thin() -> None:
    # A COMPLETE page that is merely over budget is not thin (nothing to add) — it
    # is "with notes": trim, don't add. Bloat and incompleteness are different.
    checks = _all_substance_ok()[:-1] + [_focus_check(lines_over=30)]
    p = rfa.build_payload(workspace=".", checks=checks)
    assert p["score"] == 70, p  # 100 - 30 lines over, no credit (over budget)
    assert p["finding"] == "readme_fresh_with_notes", p


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
    # substance-only: hygiene OK rows add zero penalty, so the score is driven
    # purely by the missing substance affordances (weight-scaled: 2+2+1.5+1.5+1.5
    # = 8.5 x SUBSTANCE_SHORTFALL_SCALE = 85 penalty => 100-85 = 15).
    assert p["score"] == 15, p
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
    assert "score:       47 -> 100" in out, out  # unbounded: no "/100" ceiling
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
    p = rfa.collect(root, today=_dt.date(2026, 9, 3))
    assert p["schema"] == rfa.SCHEMA
    assert "ok" in p and isinstance(p["checks"], list) and p["checks"]
    assert p["ok"] is True, p
    qwen = next(c for c in p["checks"] if c["check"] == "qwen_result_docs")
    assert qwen["status"] == "OK", qwen


def test_live_showcase_sync_reads_the_real_homepage() -> None:
    # The committed pair must actually satisfy the check - and the extraction
    # must not be vacuous: a 40 KB page reduced to nothing would pass every
    # cross-check for the wrong reason, so assert the reader text is substantial.
    root = rfa.repo_root()
    page = root / rfa.SHOWCASE_REL
    if not page.exists() or not (root / rfa.README_REL).exists():
        return  # tolerant: not in the repo tree
    text = rfa.html_text(page.read_text(encoding="utf-8"))
    assert len(text) > 5000, f"showcase reduced to {len(text)} chars - extraction broke"
    c = rfa.check_showcase_sync(
        (root / rfa.README_REL).read_text(encoding="utf-8"),
        page.read_text(encoding="utf-8"),
        version=rfa._safe_read(root / rfa.VERSION_REL),
        dataset_versions=rfa.bench_dataset_versions(root),
        dispatch=rfa.fak_dispatch_verbs(root))
    assert c["status"] == "OK", c


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
