#!/usr/bin/env python3
"""Tests for the core-docs scorecard.

Drives the PURE per-KPI checks + grader with fixture strings (no disk needed),
covers the calibration that killed the early false positives (install-context
gating, IP-guard, naive two-number exemption, dedup), then a tolerant live smoke
that `collect` folds the real committed core docs.

Run: `python tools/docs_scorecard_test.py`  (exit 0 = all pass),
or `python -m pytest tools/docs_scorecard_test.py -q`.
"""
from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docs_scorecard as dsc  # noqa: E402


# --- freshness -------------------------------------------------------------

def test_freshness_clean_is_100() -> None:
    c = dsc.kpi_freshness("# Title\n\nplain prose, no pins or placeholders.", "0.30.0")
    assert c["score"] == 100 and not c["defects"], c


def test_freshness_flags_install_pin_behind_version() -> None:
    c = dsc.kpi_freshness("Install: `FAK_VERSION=0.29.0 sh install.sh`", "0.30.0\n")
    assert c["score"] < 100 and any("0.29.0" in d for d in c["defects"]), c


def test_freshness_flags_explicit_current_version_declaration() -> None:
    text = "- **Version:** **0.29.0** ([`VERSION`](VERSION) is authoritative)."
    c = dsc.kpi_freshness(text, "0.30.0\n")
    assert c["score"] < 100, c
    assert any("current-version declaration 0.29.0" in d for d in c["defects"]), c


def test_freshness_accepts_matching_current_version_declaration() -> None:
    c = dsc.kpi_freshness("The current version is v0.30.0.", "0.30.0")
    assert not c["defects"], c


def test_freshness_ignores_explicit_historical_version_declaration() -> None:
    c = dsc.kpi_freshness("Historical snapshot — Version: 0.29.0.", "0.30.0")
    assert not c["defects"], c


def test_freshness_ignores_historical_version_mention() -> None:
    # A changelog/status mention is HISTORY, not an install pin → no defect.
    c = dsc.kpi_freshness("In v0.2.1 we shipped the gateway; v0.1.0 had the gate.",
                          "0.30.0")
    assert not c["defects"], c


def test_freshness_ignores_ip_address() -> None:
    # 0.0.0.0:8080 is a bind address, not a stale version.
    c = dsc.kpi_freshness("Run `fak serve --addr 0.0.0.0:8080` to install.", "0.30.0")
    assert not c["defects"], c


def test_freshness_ignores_benchmark_lineage_version() -> None:
    # A lineage field records version-at-run-time (no '='); it is correct history,
    # not stale install advice. (regression: MODEL-LADDER-VS-SOTA app_version row)
    c = dsc.kpi_freshness("| **fak** | app_version 0.22.0, commit `381c330` |", "0.30.0")
    assert not c["defects"], c


def test_freshness_ignores_version_in_filename() -> None:
    # A version inside a filename/path is a reference, not a pin.
    # (regression: explain-executive-summary releases/v0.9.0-candidate link)
    c = dsc.kpi_freshness("- `docs/releases/v0.9.0-candidate-strategy.md` - the plan.",
                          "0.30.0")
    assert not c["defects"], c


def test_freshness_dedupes_repeated_pin() -> None:
    txt = ("FAK_VERSION=0.29.0\nVERSION=0.29.0\n$Version = \"0.29.0\"\n"
           "APP_VERSION=0.29.0")
    c = dsc.kpi_freshness(txt, "0.30.0")
    stale = [d for d in c["defects"] if "stale install pin" in d]
    assert len(stale) == 1, stale  # one DISTINCT stale version, not four


def test_freshness_flags_placeholder() -> None:
    c = dsc.kpi_freshness("# Doc\n\nThis section is TODO.", "0.30.0")
    assert any("todo" in d.lower() for d in c["defects"]), c


def test_freshness_does_not_flag_noisy_markers() -> None:
    # XXX/WIP/"placeholder" were dropped — they fire on templates & meta-prose.
    c = dsc.kpi_freshness("Row: | XXX ms | YYY ms |. The {placeholder} is WIP.",
                          "0.30.0")
    assert not c["defects"], c


# --- links -----------------------------------------------------------------

def test_links_resolve_relative_to_doc_dir(tmp_path: Path) -> None:
    (tmp_path / "fak").mkdir()
    (tmp_path / "fak" / "EXTENDING.md").write_text("x", encoding="utf-8")
    (tmp_path / "fak" / "GETTING-STARTED.md").write_text("x", encoding="utf-8")
    # A link in fak/GETTING-STARTED.md to "EXTENDING.md" is fak/EXTENDING.md.
    c = dsc.kpi_links("see [ext](EXTENDING.md)", tmp_path, "fak/GETTING-STARTED.md")
    assert c["score"] == 100 and not c["defects"], c


def test_links_flag_dead(tmp_path: Path) -> None:
    c = dsc.kpi_links("[gone](nope/missing.md)", tmp_path, "README.md")
    assert c["score"] < 100 and any("nope/missing.md" in d for d in c["defects"]), c


def test_links_ignore_network_and_anchor(tmp_path: Path) -> None:
    c = dsc.kpi_links("[web](https://x.io) [a](#sec) [m](mailto:x@y.z)",
                      tmp_path, "README.md")
    assert not c["defects"], c


# --- structure -------------------------------------------------------------

def test_structure_flags_missing_h1() -> None:
    c = dsc.kpi_structure("no title here\n\n## but a section\n", "x.md")
    assert any("no H1" in d for d in c["defects"]), c


def test_structure_ok_with_h1_and_sections() -> None:
    body = "# Title\n\nintro.\n\n## A\n\n[link](other.md)\n\n## B\n\ntext\n"
    c = dsc.kpi_structure(body, "x.md")
    assert not c["defects"] and c["score"] >= 90, c


def test_structure_long_doc_without_sections_is_defect() -> None:
    body = "# Title\n\n" + "\n".join(f"line {i}" for i in range(60))
    c = dsc.kpi_structure(body, "x.md")
    assert any("no section headings" in d for d in c["defects"]), c


# --- readability (never hard-fails) ----------------------------------------

def test_readability_emits_no_hard_defects() -> None:
    txt = "# T\n\nthe KV cache and vDSO and context-MMU run in the address space"
    c = dsc.kpi_readability(txt)
    assert c["defects"] == [], c
    assert c["score"] < 100, c  # but it does score lower


def test_readability_glossed_term_scores_better() -> None:
    naked = dsc.kpi_readability("# T\n\nthe KV cache holds tokens across turns here")
    glossed = dsc.kpi_readability("# T\n\nthe KV cache (the model's short-term memory) holds tokens")
    assert glossed["score"] >= naked["score"], (naked, glossed)


# --- evidence --------------------------------------------------------------

def test_evidence_flags_naive_led_headline() -> None:
    c = dsc.kpi_evidence("we get **~60× vs a naive re-send loop** here", "")
    assert any("naive baseline" in d for d in c["defects"]), c


def test_evidence_exempts_honest_two_number_span() -> None:
    # naive AND tuned both named → the honest ledger framing, not a strawman lead.
    c = dsc.kpi_evidence("**60.3× vs naive · 4.1× vs tuned**", "")
    assert c["defects"] == [], c


def test_evidence_unmirrored_number_is_soft_not_hard() -> None:
    c = dsc.kpi_evidence("**99× faster**", "authority lists 4× and 5.3×")
    assert c["defects"] == [], c
    assert any("99" in s for s in c["soft"]), c


# --- per-doc fold + grader -------------------------------------------------

def test_score_doc_perfect(tmp_path: Path) -> None:
    body = "# Title\n\nA clear one-line intro sentence about the thing.\n\n## A\n\n[x](b.md)\n"
    (tmp_path / "b.md").write_text("x", encoding="utf-8")
    d = dsc.score_doc(body, "a.md", tmp_path, version="0.30.0", authority="")
    assert d["n_defects"] == 0 and d["grade"] == "A", d


def test_grade_letter_bands() -> None:
    assert dsc.grade_letter(95) == "A"
    assert dsc.grade_letter(85) == "B"
    assert dsc.grade_letter(72) == "C"
    assert dsc.grade_letter(61) == "D"
    assert dsc.grade_letter(40) == "F"


def test_missing_doc_is_worst() -> None:
    d = dsc.missing_doc_entry("docs/gone.md")
    assert d["score"] == 0.0 and d["grade"] == "F" and d["n_defects"] == 1, d


def test_payload_clean_is_ok() -> None:
    docs = [{"path": "a.md", "score": 100.0, "grade": "A", "n_defects": 0,
             "defects": [], "soft": [], "kpis": {}}]
    cov = {"overall_pct": 100.0, "reachability_pct": 100.0, "topic_pct": 100.0,
           "defects": [], "orphans": [], "missing_topics": []}
    p = dsc.build_payload(workspace=".", docs=docs, cov=cov)
    assert p["ok"] is True and p["verdict"] == "OK", p


def test_payload_carries_continuous_value(tmp_path: Path) -> None:
    # Corpus value is the continuous quality ratio; the 0-100 keys stay as the
    # legacy bridge — mirrors pkg/scorecard's Fold contract.
    body = "# Title\n\nA clear one-line intro sentence about the thing.\n\n## A\n\n[x](b.md)\n"
    (tmp_path / "b.md").write_text("x", encoding="utf-8")
    d = dsc.score_doc(body, "a.md", tmp_path, version="0.30.0", authority="")
    assert d["value"] == round(d["score"] / 100, 3), d
    cov = {"overall_pct": 100.0, "reachability_pct": 100.0, "topic_pct": 100.0,
           "defects": [], "orphans": [], "missing_topics": []}
    p = dsc.build_payload(workspace=".", docs=[d], cov=cov)
    c = p["corpus"]
    assert c["value"] == round(c["mean_score"] / 100, 3), c
    assert c["value_unit"] == "quality_ratio", c
    assert c["legacy_score"] == c["mean_score"], c
    assert c["legacy_score_scale"] == 100, c
    assert c["median_value"] == round(c["median_score"] / 100, 3), c
    assert c["worst"][0]["value"] == round(c["worst"][0]["score"] / 100, 3), c


def test_payload_counts_doc_debt() -> None:
    docs = [{"path": "a.md", "score": 80.0, "grade": "B", "n_defects": 2,
             "defects": ["x", "y"], "soft": [], "kpis": {}}]
    cov = {"overall_pct": 90.0, "reachability_pct": 80.0, "topic_pct": 100.0,
           "defects": ["orphan z"], "orphans": ["z"], "missing_topics": []}
    p = dsc.build_payload(workspace=".", docs=docs, cov=cov)
    assert p["ok"] is False and p["corpus"]["doc_debt"] == 3, p
    assert p["corpus"]["doc_debt_in_docs"] == 2 and p["corpus"]["doc_debt_in_coverage"] == 1, p


# --- coverage --------------------------------------------------------------

def test_coverage_orphan_detection(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("see [a](a.md)", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A\n[b](b.md)", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B", encoding="utf-8")
    (tmp_path / "orphan.md").write_text("# Orphan, linked from nowhere", encoding="utf-8")
    cov = dsc.coverage(tmp_path, ["a.md", "b.md", "orphan.md"])
    assert "orphan.md" in cov["orphans"], cov
    assert "a.md" not in cov["orphans"] and "b.md" not in cov["orphans"], cov


def test_generated_report_is_not_scored() -> None:
    # The scorecard's own output is excluded (circular / self-oscillating).
    assert dsc._excluded("docs/DOCS-SCORECARD.md") is True
    assert dsc._excluded("docs/FAQ.md") is False
    assert dsc._excluded("node_modules/x/README.md") is True


def test_coverage_front_doors_exempt(tmp_path: Path) -> None:
    # INDEX.md is a front door; not linking TO it from README is not an orphan.
    (tmp_path / "README.md").write_text("hi", encoding="utf-8")
    (tmp_path / "INDEX.md").write_text("# Index", encoding="utf-8")
    cov = dsc.coverage(tmp_path, ["INDEX.md"])
    assert cov["orphans"] == [], cov


# --- live smoke ------------------------------------------------------------

def test_live_collect_core() -> None:
    root = dsc.repo_root()
    if not (root / "README.md").exists():
        return  # tolerant: not in the repo tree
    p = dsc.collect(root, scope="core")
    assert p["schema"] == dsc.SCHEMA
    assert p["corpus"]["n_docs"] == len(dsc.CORE_DOCS)
    assert isinstance(p["docs"], list) and p["docs"]


def test_live_collect_reachable() -> None:
    root = dsc.repo_root()
    if not (root / "README.md").exists():
        return
    p = dsc.collect(root, scope="reachable")
    assert p["scope"] == "reachable"
    assert p["corpus"]["n_docs"] > len(dsc.CORE_DOCS)  # closure is larger


# --- --since shift-left skip-gate ------------------------------------------

def _have_git() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True)
        return True
    except OSError:
        return False


def _git(dir_: str, *args: str) -> None:
    env = dict(os.environ,
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    proc = subprocess.run(["git", "-C", dir_, *args],
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"git {args}: {proc.stderr}")


def _run_main(argv: list[str]) -> tuple[int, str]:
    """Run the card's main() capturing stdout; returns (rc, stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = dsc.main(argv)
    return rc, buf.getvalue()


def test_since_skips_when_no_corpus_changed(tmp_path: Path) -> None:
    # A repo whose only change since HEAD is a NON-markdown file: the docs corpus
    # (**/*.md) is untouched, so the gate must skip the whole scan and exit 0.
    if not _have_git():
        return
    d = str(tmp_path)
    (tmp_path / "keep.go").write_text("package a\n", encoding="utf-8")
    _git(d, "init", "-q")
    _git(d, "add", "-A")
    _git(d, "commit", "-qm", "base")
    (tmp_path / "keep.go").write_text("package a\n// edit\n", encoding="utf-8")

    rc, out = _run_main(["--workspace", d, "--since", "HEAD"])
    assert rc == 0, (rc, out)
    assert "unchanged since HEAD" in out, out
    # Proof it did NOT do a full scan: the render header never printed.
    assert "per-doc (worst first)" not in out, out


def test_since_fullscans_when_corpus_changed(tmp_path: Path) -> None:
    # A markdown file changed since HEAD → the corpus WAS touched → the gate must
    # fall through to a full scan (no "unchanged" short-circuit).
    if not _have_git():
        return
    d = str(tmp_path)
    (tmp_path / "README.md").write_text("# Home\n\nhello\n", encoding="utf-8")
    _git(d, "init", "-q")
    _git(d, "add", "-A")
    _git(d, "commit", "-qm", "base")
    (tmp_path / "README.md").write_text("# Home\n\nhello, edited\n", encoding="utf-8")

    rc, out = _run_main(["--workspace", d, "--since", "HEAD"])
    assert "unchanged since HEAD" not in out, out
    assert "per-doc (worst first)" in out, out  # the full render did run


def test_since_absent_is_byte_identical() -> None:
    # Contract preserved: with --since absent the gate is fully inert, so default
    # and --json output are byte-identical to an explicit empty --since.
    root = dsc.repo_root()
    if not (root / "README.md").exists():
        return
    rc_a, out_a = _run_main([])
    rc_b, out_b = _run_main(["--since", ""])
    assert rc_a == rc_b and out_a == out_b, "default output drifted with empty --since"
    jrc_a, jout_a = _run_main(["--json"])
    jrc_b, jout_b = _run_main(["--json", "--since", ""])
    assert jrc_a == jrc_b and jout_a == jout_b, "--json output drifted with empty --since"


def test_since_inert_on_markdown_snapshot(tmp_path: Path) -> None:
    # The gate must NOT fire in --markdown snapshot mode even with --since set:
    # a snapshot is a doc render, not a scan, so it always produces its body.
    if not _have_git():
        return
    d = str(tmp_path)
    (tmp_path / "keep.go").write_text("package a\n", encoding="utf-8")
    _git(d, "init", "-q")
    _git(d, "add", "-A")
    _git(d, "commit", "-qm", "base")
    (tmp_path / "keep.go").write_text("package a\n// edit\n", encoding="utf-8")

    rc, out = _run_main(["--workspace", d, "--since", "HEAD", "--markdown"])
    assert "unchanged since HEAD" not in out, out
    assert "# Core-docs scorecard" in out, out  # the markdown body still rendered


# --- self-contained runner (mirrors readme_freshness_audit_test.py) --------

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
