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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import learning_scorecard as lsc  # noqa: E402


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
