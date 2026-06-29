#!/usr/bin/env python3
"""Tests for the learning-docs scorecard.

Drives the PURE pedagogy KPIs + doc-type classifier + grader with fixture strings
(no disk), then a tolerant live smoke that `collect` folds the real committed
learning corpus.

Run: `python tools/learning_scorecard_test.py`  (exit 0 = all pass),
or `python -m pytest tools/learning_scorecard_test.py -q`.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))
import learning_scorecard as lsc  # noqa: E402


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --- doc-type classifier ---------------------------------------------------

def test_doc_type_curriculum() -> None:
    assert lsc.doc_type("LEARNING-PATH.md") == "curriculum"


def test_doc_type_tutorial() -> None:
    assert lsc.doc_type("docs/fak/tutorial.md") == "tutorial"
    assert lsc.doc_type("GETTING-STARTED.md") == "tutorial"
    assert lsc.doc_type("START-HERE.md") == "tutorial"
    assert lsc.doc_type("docs/fak/server-quickstart.md") == "tutorial"


def test_doc_type_howto_vs_reference() -> None:
    assert lsc.doc_type("docs/fak/policy-guide.md") == "howto"
    assert lsc.doc_type("docs/fak/observability.md") == "howto"
    # index/meta/reference under the learning dirs are typed reference (relaxed)
    assert lsc.doc_type("docs/fak/README.md") == "reference"
    assert lsc.doc_type("docs/fak/api-reference.md") == "reference"
    assert lsc.doc_type("docs/fak/documentation-roadmap.md") == "reference"
    assert lsc.doc_type("docs/fak/faq.md") == "reference"


def test_doc_type_explainer() -> None:
    assert lsc.doc_type("docs/explainers/sota-optimizations.md") == "explainer"
    assert lsc.doc_type("docs/concepts-and-story.md") == "explainer"


# --- orientation -----------------------------------------------------------

def test_orientation_clean_tutorial_is_high() -> None:
    text = ("# Tutorial\n\n**Audience:** newcomers. **Prereqs:** Go.\n\n"
            "By the end you'll be able to run it. See [next](docs/fak/faq.md).\n"
            + "\n".join(f"line {i}" for i in range(70))
            + "\n## Next steps\nWhere to go from here: read the FAQ.\n")
    c = lsc.kpi_orientation(text, "tutorial")
    assert c["score"] >= 90 and not c["defects"], c


def test_orientation_missing_signpost_is_hard_for_tutorial() -> None:
    text = "# Thing\n\n" + "\n".join(f"prose line {i}" for i in range(60))
    c = lsc.kpi_orientation(text, "tutorial")
    assert c["score"] < 100 and any("signpost" in d for d in c["defects"]), c


def test_orientation_missing_signpost_is_soft_for_reference() -> None:
    text = "# Ref\n\n" + "\n".join(f"row {i}" for i in range(20))
    c = lsc.kpi_orientation(text, "reference")
    assert not c["defects"], c  # reference is exempt → soft only


def test_orientation_dead_end_long_howto_is_hard() -> None:
    # long, has a signpost, but links to no other doc → curriculum dead-end
    text = ("# Guide\n\n**Audience:** ops.\n\n"
            + "\n".join(f"step {i}" for i in range(60)))
    c = lsc.kpi_orientation(text, "howto")
    assert any("dead-end" in d for d in c["defects"]), c


# --- runnable --------------------------------------------------------------

def test_runnable_prose_only_howto_is_hard() -> None:
    c = lsc.kpi_runnable("# Guide\n\njust prose, no commands at all.", "howto")
    assert c["score"] < 100 and any("runnable" in d for d in c["defects"]), c


def test_runnable_explainer_prose_only_is_soft() -> None:
    c = lsc.kpi_runnable("# Idea\n\nconceptual prose, no commands.", "explainer")
    assert not c["defects"], c


def test_runnable_two_blocks_is_clean() -> None:
    text = "# T\n\n```sh\nfak version\n```\n\nOutput:\n\n```\nv0.30.0\n```\n"
    c = lsc.kpi_runnable(text, "tutorial")
    assert c["score"] == 100 and not c["defects"] and not c["soft"], c


def test_runnable_single_block_tutorial_soft_nudge() -> None:
    text = "# T\n\n**Audience** x.\n\n```sh\nfak version\n```\n"
    c = lsc.kpi_runnable(text, "tutorial")
    assert not c["defects"] and any("one code block" in s for s in c["soft"]), c


# --- worked ----------------------------------------------------------------

def test_worked_tutorial_without_example_is_hard() -> None:
    c = lsc.kpi_worked("# T\n\nrun the command and trust me it works.", "tutorial")
    assert c["score"] < 100 and any("worked example" in d for d in c["defects"]), c


def test_worked_with_lab_and_checkpoint_clean() -> None:
    text = "# Course\n\n**Lab:**\n```\ngo test ./...\n```\n**Checkpoint:** explain it.\n"
    c = lsc.kpi_worked(text, "curriculum")
    assert c["score"] == 100 and not c["defects"], c


def test_worked_explainer_missing_is_soft() -> None:
    c = lsc.kpi_worked("# Idea\n\npure concept, no lab.", "explainer")
    assert not c["defects"] and c["soft"], c


# --- honesty ---------------------------------------------------------------

def test_honesty_naked_multiplier_is_soft() -> None:
    text = "# Perf\n\nfak is 60× faster than the baseline, full stop.\n```\ncode\n```"
    c = lsc.kpi_honesty(text, "")
    assert any("no honesty marker" in s for s in c["soft"]), c


def test_honesty_marked_multiplier_is_clean() -> None:
    text = "# Perf\n\nfak shows 60× vs naive (SIMULATED) — see CLAIMS.md.\n```\nx\n```"
    c = lsc.kpi_honesty(text, "")
    assert not any("no honesty marker" in s for s in c["soft"]), c


def test_honesty_naive_led_bold_headline_is_hard() -> None:
    # reuses dsc.kpi_evidence: a bold naive multiplier with no tuned counter
    text = "# X\n\n**60× faster than naive**\n"
    c = lsc.kpi_honesty(text, "")
    assert any("naive" in d for d in c["defects"]), c


# --- freshness (tighter install-context gate) ------------------------------

def test_freshness_flags_real_uppercase_install_pin() -> None:
    c = lsc.kpi_freshness("Install: `FAK_VERSION=0.30.0 sh install.sh`", "0.32.0")
    assert c["score"] < 100 and any("0.30.0" in d for d in c["defects"]), c


def test_freshness_ignores_prometheus_exposition_version() -> None:
    # `text/plain; version=0.0.4` is the Prometheus exposition-format constant,
    # not a fak install pin — a lowercase `version=` must NOT be flagged.
    c = lsc.kpi_freshness("// resp.Body is text/plain; version=0.0.4 — counters.", "0.32.0")
    assert not c["defects"], c


def test_freshness_ignores_captured_metric_version() -> None:
    # Captured Prometheus metric output is the honest record of a build.
    c = lsc.kpi_freshness('fak_gateway_build_info{version="0.30.0",engine="x"} 1', "0.32.0")
    assert not c["defects"], c


def test_freshness_flags_app_version_build_arg() -> None:
    c = lsc.kpi_freshness("docker build --build-arg APP_VERSION=0.30.0 -t fak:x .", "0.32.0")
    assert any("0.30.0" in d for d in c["defects"]), c


# --- generated scorecard stamp freshness ----------------------------------

def test_snapshot_stamp_parses_generated_comment() -> None:
    text = "<!-- learning-scorecard: 2026-06-29 · process: tools/learning_scorecard.py -->"
    assert lsc.snapshot_stamp(text) == "2026-06-29"


def test_stamp_freshness_flags_stale_stamp_after_corpus_drift() -> None:
    c = lsc.stamp_freshness_from_dates(
        stamp="2026-06-01",
        reference_date=date(2026, 6, 4),
        last_corpus_change_date=date(2026, 6, 3),
        cadence_days=1,
    )
    assert c["stamp_age_days"] == 3, c
    assert c["corpus_changed_since_stamp"], c
    assert c["stale_stamp"] and c["flag"] == "stale-stamp", c


def test_stamp_freshness_requires_corpus_drift() -> None:
    c = lsc.stamp_freshness_from_dates(
        stamp="2026-06-01",
        reference_date=date(2026, 6, 4),
        last_corpus_change_date=date(2026, 6, 1),
        cadence_days=1,
    )
    assert c["stamp_age_days"] == 3, c
    assert not c["corpus_changed_since_stamp"], c
    assert not c["stale_stamp"], c


def test_build_payload_counts_stale_stamp_debt() -> None:
    payload = lsc.build_payload(
        workspace=".",
        docs=[],
        cov={"overall_pct": 100.0, "defects": []},
        importance={},
        stamp={"stale_stamp": True, "reason": "stale-stamp"},
    )
    assert payload["corpus"]["learning_debt"] == 1, payload
    assert payload["corpus"]["learning_debt_in_stamp"] == 1, payload
    assert not payload["ok"], payload


def test_detect_latency_bench_is_net_true_after_cost() -> None:
    payload = lsc.detect_latency_bench_from_cost(0.25)
    assert payload["schema"] == lsc.DETECT_LATENCY_SCHEMA
    assert payload["stamp_witness"]["stamp_age_days"] == 3, payload
    assert payload["stamp_witness"]["flag"] == "stale-stamp", payload
    assert payload["loop"]["detect_latency_days"] == 1.0, payload
    assert payload["manual"]["detect_latency_days"] == 5.0, payload
    assert payload["net"]["days_saved"] > 3.99, payload
    assert payload["ok"] and payload["verdict"] == "net-true", payload


# --- shipped-surface coverage ---------------------------------------------

def test_added_on_or_after_stamp_excludes_same_day_stamp() -> None:
    original = lsc._git_first_added_date
    try:
        lsc._git_first_added_date = lambda root, rel: date(2026, 6, 29)
        same_day = lsc._added_on_or_after_stamp(
            Path("."), "cmd/fak/example.go", date(2026, 6, 29),
            require_git_date=True,
        )
        lsc._git_first_added_date = lambda root, rel: date(2026, 6, 30)
        next_day = lsc._added_on_or_after_stamp(
            Path("."), "cmd/fak/example.go", date(2026, 6, 29),
            require_git_date=True,
        )
    finally:
        lsc._git_first_added_date = original

    assert not same_day
    assert next_day


def _seed_surface_fixture(root: Path) -> list[str]:
    _write(root, "docs/LEARNING-SCORECARD.md",
           "<!-- learning-scorecard: 2026-06-29 - process: tools/learning_scorecard.py -->\n")
    _write(root, "dos.toml", 'faklesson = ["internal/faklesson/**"]\n')
    _write(root, "internal/faklesson/doc.go",
           "// Package faklesson is a teachable fixture leaf.\npackage faklesson\n")
    _write(root, "cmd/fak/main.go", """package main

func main() {
    switch "x" {
    case "faklesson":
        cmdFakLesson(nil)
    }
}
""")
    _write(root, "cmd/fak/faklesson.go",
           "package main\n\nfunc cmdFakLesson(argv []string) {}\n")
    return ["internal/faklesson/doc.go", "cmd/fak/faklesson.go"]


def test_shipped_surface_without_course_reference_counts_coverage_debt() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        added = _seed_surface_fixture(root)
        _write(root, "LEARNING-PATH.md", "# Learning path\n\nNo new course here.\n")

        cov = lsc.coverage(root, ["LEARNING-PATH.md"], added_paths=added,
                           require_git_dates=False)
        assert "internal/faklesson" in cov["missing_shipped_surfaces"], cov
        assert "fak faklesson" in cov["missing_shipped_surfaces"], cov
        surface_defects = [d for d in cov["defects"] if "shipped-but-untaught" in d]
        assert len(surface_defects) == 2, cov

        payload = lsc.build_payload(
            workspace=str(root), docs=[],
            cov={"overall_pct": 100.0, "defects": surface_defects},
            importance={},
        )
        assert payload["corpus"]["learning_debt_in_coverage"] == 2, payload


def test_shipped_surface_reference_in_learning_path_greens() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        added = _seed_surface_fixture(root)
        course = ("# Learning path\n\n"
                  "**Read:** the `faklesson` course.\n\n"
                  "Lab: `go run ./cmd/fak faklesson` and "
                  "`go test ./internal/faklesson`.\n")
        _write(root, "LEARNING-PATH.md", course)

        cov = lsc.coverage(root, ["LEARNING-PATH.md"], added_paths=added,
                           require_git_dates=False)
        assert "internal/faklesson" in cov["covered_shipped_surfaces"], cov
        assert "fak faklesson" in cov["covered_shipped_surfaces"], cov
        assert not cov["missing_shipped_surfaces"], cov
        assert not [d for d in cov["defects"] if "shipped-but-untaught" in d], cov


def test_new_fak_verb_counts_even_when_handler_file_is_not_new() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        _seed_surface_fixture(root)
        _write(root, "LEARNING-PATH.md", "# Learning path\n\nNo new course here.\n")

        cov = lsc.coverage(
            root, ["LEARNING-PATH.md"],
            added_paths=[],
            added_fak_verbs=["faklesson"],
            require_git_dates=False,
        )
        assert "fak faklesson" in cov["missing_shipped_surfaces"], cov
        assert any("shipped-but-untaught surface: fak faklesson" in d
                   for d in cov["defects"]), cov


def test_internal_plumbing_surface_is_not_teachable_coverage_spam() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        _write(root, "dos.toml", 'cmdutil = ["internal/cmdutil/**"]\n')
        _write(root, "internal/cmdutil/doc.go",
               "// Package cmdutil is fixture plumbing.\npackage cmdutil\n")
        surfaces = lsc.derive_teachable_surfaces(
            root, ["internal/cmdutil/doc.go"],
            stamp_date=date(2026, 6, 29),
            require_git_dates=False,
        )
        assert surfaces == [], surfaces


# --- doc-type-aware fold + importance --------------------------------------

def test_score_doc_shape() -> None:
    text = ("# Tutorial\n\n**Audience:** all. By the end you'll run it.\n\n"
            "See [faq](docs/fak/faq.md).\n\n```sh\nfak version\n```\n\nOutput:\n```\nv1\n```\n"
            "**Lab:** do it. **Checkpoint:** explain. Next: read more.\n")
    d = lsc.score_doc(text, "docs/fak/tutorial.md", Path("."), version="0.30.0", authority="")
    assert d["type"] == "tutorial"
    assert set(d["kpis"]) == set(lsc.KPI_WEIGHTS)
    assert 0 <= d["score"] <= 100 and d["grade"] in "ABCDF"


def test_weights_sum_to_one() -> None:
    assert abs(sum(lsc.KPI_WEIGHTS.values()) - 1.0) < 1e-9


# --- live smoke ------------------------------------------------------------

def test_collect_live_smoke() -> None:
    root = lsc.dsc.repo_root()
    if not (root / "LEARNING-PATH.md").exists():
        return  # not in a full checkout; skip
    payload = lsc.collect(root)
    assert payload["schema"] == lsc.SCHEMA
    c = payload["corpus"]
    assert c["n_docs"] > 10, c
    assert "priorities" in c and isinstance(c["priorities"], list)
    # LEARNING-PATH should rank as a high-importance doc (front door, hop 0).
    imps = {d["path"]: d.get("importance", 0) for d in payload["docs"]}
    assert imps.get("LEARNING-PATH.md", 0) > 0, imps
    # every per-doc record carries the seven KPIs and a type
    for d in payload["docs"]:
        assert d.get("type") in {"curriculum", "tutorial", "howto", "explainer", "reference"}


# --- the --check learning-debt ratchet (#1280) -----------------------------
# The keep-bit witness every #1278 child reads: a synthetic debt regression must
# RED, a clean corpus must GREEN, and --baseline N pins the allowed ceiling.

def _clean_payload() -> dict:
    """A corpus with zero learning-debt."""
    return lsc.build_payload(workspace=".", docs=[],
                             cov={"overall_pct": 100.0, "defects": []},
                             importance={}, stamp=None)


def _debt_payload(n: int) -> dict:
    """A corpus carrying exactly ``n`` units of learning-debt (uncovered topics)."""
    return lsc.build_payload(workspace=".", docs=[],
                             cov={"overall_pct": 0.0, "defects": [f"d{i}" for i in range(n)]},
                             importance={}, stamp=None)


def test_check_gate_greens_clean_corpus() -> None:
    code, msg = lsc.check_gate(_clean_payload(), baseline=0)
    assert code == 0, msg
    assert "OK" in msg, msg


def test_check_gate_reds_debt_over_baseline() -> None:
    payload = _debt_payload(3)
    assert payload["corpus"]["learning_debt"] == 3, payload
    code, msg = lsc.check_gate(payload, baseline=0)
    assert code != 0, msg
    assert "3" in msg, msg


def test_check_gate_baseline_pins_ceiling() -> None:
    # debt at-or-below the ceiling holds green; above it reds.
    assert lsc.check_gate(_debt_payload(5), baseline=5)[0] == 0
    assert lsc.check_gate(_debt_payload(6), baseline=5)[0] != 0


def test_check_cli_exit_codes() -> None:
    saved = lsc.collect
    try:
        lsc.collect = lambda ws: _debt_payload(2)
        assert lsc.main(["--check"]) != 0                      # debt 2 > default 0 -> red
        assert lsc.main(["--check", "--baseline", "2"]) == 0   # ceiling pinned -> green
        lsc.collect = lambda ws: _clean_payload()
        assert lsc.main(["--check"]) == 0                      # clean corpus -> green
    finally:
        lsc.collect = saved


def _run_all() -> int:
    mod = sys.modules[__name__]
    tests = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
