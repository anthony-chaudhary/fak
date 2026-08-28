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

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repo_hygiene_scorecard as rh  # noqa: E402


# --- helpers ---------------------------------------------------------------

def test_dated_doc_detection() -> None:
    assert rh.is_dated_doc("PLAN-2026-06-19.md")
    assert rh.is_dated_doc("trust-floor-decomposition-492.md")
    assert not rh.is_dated_doc("tutorial.md")
    assert not rh.is_dated_doc("README.md")


def test_dated_doc_ignores_standards_citation() -> None:
    # A trailing -NNN is an ISSUE number — unless it is the number of a STANDARD.
    # `safety-case-iec-61508-iso-26262.md` argues a functional-safety case against
    # ISO 26262; there is no issue #26262. Same for an RFC / IEEE / NIST citation.
    assert not rh.is_dated_doc("safety-case-iec-61508-iso-26262.md")
    assert not rh.is_dated_doc("http-semantics-rfc-9110.md")
    assert not rh.is_dated_doc("floating-point-ieee-754.md")
    assert not rh.is_dated_doc("crypto-baseline-nist-800.md")
    # a date stamp still wins over a standards citation in the same basename
    assert rh.is_dated_doc("iso-26262-review-2026-07-01.md")


def test_dated_doc_still_catches_real_issue_numbers() -> None:
    """NEGATIVE CONTROL for the standards narrowing. Deliberately passes BOTH before
    and after the change — that is exactly its job: it proves the detector got
    narrower, not blind. A detector that went blind is a worse bug than the FP."""
    assert rh.is_dated_doc("fix-the-thing-3855.md")
    assert rh.is_dated_doc("trust-floor-decomposition-492.md")
    # `-440` here IS issue #440 (the doc's own title says "(#440)" and it links
    # github.com/.../issues/440), so the real repo doc must STAY flagged.
    assert rh.is_dated_doc("QWEN36-LOAD-PROFILE-440.md")
    # a plain word that merely ENDS in a standards-body token ("gold-EN") must not
    # buy an exemption — the token has to be its own delimited segment.
    assert rh.is_dated_doc("golden-440.md")
    assert rh.is_dated_doc("useful-440.md")


def test_is_fixture_path_judges_directory_segments_only() -> None:
    assert rh.is_fixture_path("internal/session/testdata/compactaudit/w-2026-07-16.md")
    assert rh.is_fixture_path("internal/codexlifecycle/testdata/w-2026-07-18.md")
    # a FILE merely named testdata is not inside the reserved directory
    assert not rh.is_fixture_path("docs/testdata.md")
    assert not rh.is_fixture_path("docs/notes/testdata-plan-2026-07-01.md")
    assert not rh.is_fixture_path("docs/benchmarks/X-2026-07-01.md")


def test_misplaced_dated_docs_exempts_testdata_fixtures() -> None:
    # `testdata/` is toolchain-RESERVED (go/build ignores it), which is why a fixture
    # and the witness that documents it must live there — "move it to docs/notes/"
    # would divorce the witness from the corpus it annotates.
    md = [
        "docs/benchmarks/RESULT-2026-07-01.md",                       # real defect
        "internal/session/testdata/compactaudit/w-2026-07-16.md",     # fixture
        "internal/codexlifecycle/testdata/w-2026-07-18.md",           # fixture
        "docs/notes/AT-HOME-2026-07-01.md",                           # at home
        "PLAN-2026-07-01.md",                                         # root: not ours
    ]
    out = rh.misplaced_dated_docs(md)
    assert out == ["docs/benchmarks/RESULT-2026-07-01.md"], out


def test_misplaced_dated_docs_negative_control() -> None:
    """NEGATIVE CONTROL for the whole placement input: a genuinely misplaced dated doc
    and a genuinely issue-numbered doc must BOTH still be caught. Passes before and
    after by design."""
    md = ["docs/foo/BAR-2026-07-01.md", "docs/foo/thing-3855.md"]
    assert rh.misplaced_dated_docs(md) == sorted(md)


def test_source_paths_drops_tracked_but_ignored() -> None:
    # A path can be BOTH tracked and ignored (committed before the rule, or force-added
    # into a scratch tree). Note `git ls-files --cached --others --exclude-standard`
    # does NOT drop it: --exclude-standard filters only the --others half.
    tracked = ["README.md", ".dispatch-runs/contract-overlays/issue-2206.md",
               "docs/keep/PLAN-2026-07-01.md"]
    ignored = [".dispatch-runs/contract-overlays/issue-2206.md"]
    assert rh.source_paths(tracked, ignored) == ["README.md", "docs/keep/PLAN-2026-07-01.md"]
    # nothing ignored -> the corpus is untouched, in order (purely subtractive)
    assert rh.source_paths(tracked, []) == tracked


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


# --- hermetic temp-repo: enumeration immunity, end to end -------------------
# Follows the harness in code_quality_scorecard_test.py (_have_git/_git/_write).
# The scratch dir is deliberately named `.overlay-scratch/` — a name that appears in
# NO hand-maintained exclusion list in this card — so these tests can only pass via
# the git-ignore read, never vacuously via a name filter.

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


def _write(dir_: str, rel: str, body: str) -> None:
    p = Path(dir_) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


# The five shapes the placement KPI has to tell apart, in one hermetic repo.
_REAL_DATED = "docs/foo/BAR-2026-07-01.md"          # must STAY flagged
_REAL_ISSUE = "docs/foo/thing-3855.md"              # must STAY flagged
_SCRATCH_DOC = ".overlay-scratch/issue-2206.md"     # tracked AND ignored -> dropped
_FIXTURE_DOC = "internal/pkg/testdata/witness-2026-07-18.md"   # reserved dir -> dropped
_STANDARD_DOC = "docs/foo/safety-case-iso-26262.md"  # standards citation -> dropped


def _seed_placement_repo(d: str) -> None:
    _write(d, ".gitignore", ".overlay-scratch/\n")
    _write(d, "README.md", "# x\n")
    _write(d, "INDEX.md", "# index\n")
    _write(d, "docs/notes/AT-HOME-2026-07-01.md", "# at home\n")
    for rel in (_REAL_DATED, _REAL_ISSUE, _SCRATCH_DOC, _FIXTURE_DOC, _STANDARD_DOC):
        _write(d, rel, f"# {rel}\n")
    _git(d, "init", "-q")
    _git(d, "add", ".gitignore", "README.md", "INDEX.md",
         "docs/notes/AT-HOME-2026-07-01.md",
         _REAL_DATED, _REAL_ISSUE, _FIXTURE_DOC, _STANDARD_DOC)
    # -f: model the historical defect: a doc committed inside a gitignored
    # runtime tree must not enter the reader-facing corpus.
    _git(d, "add", "-f", _SCRATCH_DOC)
    _git(d, "commit", "-qm", "base")


def test_enumeration_drops_tracked_but_ignored_doc() -> None:
    if not _have_git():
        print("repo_hygiene_scorecard_test: git unavailable, skipping enumeration case")
        return
    with tempfile.TemporaryDirectory() as d:
        _seed_placement_repo(d)
        root = Path(d)
        # the polluted read the old enumeration made: plain `ls-files` DOES list it,
        # so this is not a straw man — the scratch doc really is in the index.
        assert _SCRATCH_DOC in rh._git_lines(["ls-files"], root)
        # ...and so does the doctrine's `--cached --others --exclude-standard` form:
        # --exclude-standard filters only the --others half.
        both = rh._git_lines(["ls-files", "--cached", "--others", "--exclude-standard"], root)
        assert _SCRATCH_DOC in both, both
        # the clean read drops it and keeps every legitimate tracked path.
        src = rh._source_paths(root)
        assert _SCRATCH_DOC not in src, src
        for rel in ("README.md", "INDEX.md", _REAL_DATED, _REAL_ISSUE,
                    _FIXTURE_DOC, _STANDARD_DOC):
            assert rel in src, (rel, src)


def test_placement_kpi_immune_end_to_end() -> None:
    """The whole point, measured the way the operator measures it: `collect()` on a
    repo carrying all five shapes reports EXACTLY the two real defects."""
    if not _have_git():
        print("repo_hygiene_scorecard_test: git unavailable, skipping end-to-end case")
        return
    with tempfile.TemporaryDirectory() as d:
        _seed_placement_repo(d)
        payload = rh.collect(Path(d))
        placement = next(k for k in payload["kpis"] if k["kpi"] == "placement")
        flagged = sorted(dfx.split(": ", 1)[1].split(" →")[0] for dfx in placement["defects"])
        # NEGATIVE CONTROL: the detector is narrower, not blind.
        assert flagged == sorted([_REAL_DATED, _REAL_ISSUE]), flagged
        assert payload["corpus"]["debt_by_kpi"]["placement"] == 2, payload["corpus"]


def test_untracked_dir_bloat_flags_huge_dir() -> None:
    # `git ls-files --others --directory` collapses a wholly-untracked tree to one
    # `dir/` entry; the un-collapsed list sizes it. A 150-file dir trips the canary.
    others = [f".st_full/internal/pkg{i}/x.go" for i in range(150)]
    others_dirs = [".st_full/"]
    hits = rh.untracked_dir_bloat(others, others_dirs, threshold=100)
    assert len(hits) == 1, hits
    assert ".st_full/" in hits[0] and "150 untracked files" in hits[0]
    # names the remedy (gitignore rule or removal) so the reader knows what to do
    assert "gitignore it" in hits[0] and "/.st_full/" in hits[0]


def test_untracked_dir_bloat_ignores_small_dir() -> None:
    # a normal in-progress feature (tens of files) is not bloat — no false positive.
    others = [f"internal/newpkg/f{i}.go" for i in range(12)]
    assert rh.untracked_dir_bloat(others, ["internal/newpkg/"], threshold=100) == []


def test_untracked_dir_bloat_orders_heaviest_first() -> None:
    others = ([f".st_full/a{i}" for i in range(300)]
              + [f".st_gen/b{i}" for i in range(120)])
    hits = rh.untracked_dir_bloat(others, [".st_gen/", ".st_full/"], threshold=100)
    assert len(hits) == 2 and hits[0].startswith(".st_full/"), hits


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

def test_witness_claim_and_research_roots_are_placement_homes():
    files = [
        "docs/_witnesses/run-2026-08-27.md",
        "docs/claims/claim-2026-08-27.md",
        "docs/research/study-2026-08-27.md",
        "docs/random/study-2026-08-27.md",
    ]
    assert rh.misplaced_dated_docs(files) == ["docs/random/study-2026-08-27.md"]


def test_disambiguation_ledgers_are_not_redundancy_candidates(tmp_path):
    # The integration path filters these schema-shaped identity ledgers before
    # kpi_redundancy; ordinary near-identical docs remain detectable below.
    reader = [
        "docs/concepts/disambiguation-cache.md",
        "docs/concepts/disambiguation-loop.md",
        "docs/a.md",
        "docs/b.md",
    ]
    dup_reader = [f for f in reader if not f.startswith("docs/concepts/disambiguation-")]
    assert dup_reader == ["docs/a.md", "docs/b.md"]
    prose = " ".join(f"word{i}" for i in range(100))
    docs = [{"path": p, "shingles": rh.shingles(prose), "words": 100} for p in dup_reader]
    assert len(rh.kpi_redundancy(docs)["defects"]) == 1
