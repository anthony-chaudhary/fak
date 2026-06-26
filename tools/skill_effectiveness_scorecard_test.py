#!/usr/bin/env python3
"""Tests for the skill-effectiveness scorecard.

Each KPI gets a defect-trigger fixture AND a clean fixture; the fold is checked for
the skill-debt count + grade; pure helpers (frontmatter split, path extraction,
template-slot + flaggable rules) get their own cases; and a live smoke asserts the
real skill tree's payload is well-formed and the weight set is exactly the KPI set.
Run directly: ``python3 tools/skill_effectiveness_scorecard_test.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skill_effectiveness_scorecard as se  # noqa: E402


def _fact(**over) -> dict:
    """A clean (all-affordances-present) skill fact, overridable per test."""
    base = {
        "name": "demo",
        "fm_name": "demo",
        "description": "Drive a thing. Use when the operator asks to drive the thing.",
        "has_allowed_tools": True,
        "commits": False,
        "metric_driving": False,
        "n_lines": 120,
        "dead_refs": [],
        "body_has_pathdisc": True,
        "body_has_proof": True,
        "body_has_antigame": True,
    }
    base.update(over)
    return base


# --- pure helpers -----------------------------------------------------------

def test_grade_letter_bands() -> None:
    assert se.grade_letter(100) == "A" and se.grade_letter(90) == "A"
    assert se.grade_letter(85) == "B" and se.grade_letter(72) == "C"
    assert se.grade_letter(61) == "D" and se.grade_letter(40) == "F"


def test_weights_sum_to_one_and_match_kpi_set() -> None:
    assert abs(sum(se.KPI_WEIGHTS.values()) - 1.0) < 1e-9
    assert set(se.KPI_WEIGHTS) == set(se.KPI_GROUP)
    assert all(g in se.GROUPS for g in se.KPI_GROUP.values())


def test_split_frontmatter() -> None:
    text = '---\nname: foo\ndescription: "Bar baz"\nmetadata:\n  opencode: claude-only\n---\nBODY here\n'
    fm, body = se.split_frontmatter(text)
    assert fm["name"] == "foo" and fm["description"] == "Bar baz"
    assert "opencode" not in fm  # nested keys are not flattened
    assert body.strip() == "BODY here"
    # no frontmatter at all
    fm2, body2 = se.split_frontmatter("just body, no fence")
    assert fm2 == {} and body2 == "just body, no fence"


def test_template_slots_and_flaggable() -> None:
    assert se.is_template_slot("docs/notes/X-YYYY-MM-DD.md")
    assert se.is_template_slot("foo/<name>/SKILL.md")
    assert se.is_template_slot("tools/*.py")
    assert not se.is_template_slot("tools/real_tool.py")
    # only prefixed or explicitly-relative refs are flaggable; bare names never are.
    assert se.is_flaggable("tools/x.py") and se.is_flaggable("../scorecard/SKILL.md")
    assert se.is_flaggable("./local.py")
    assert not se.is_flaggable("MEMORY_archive.md")  # bare external artifact
    assert not se.is_flaggable("README.md")          # bare, ambiguous


def test_cited_paths_extracts_links_and_code_paths() -> None:
    body = ("See [the doctrine](../scorecard/SKILL.md) and run `python tools/x_scorecard.py`.\n"
            "External [site](https://example.com) and anchor [a](#sec) are skipped.\n"
            "Prose mentioning the tools/ directory or a slot `<path>` is not a ref.\n"
            "An output named docs/notes/OUT-YYYY-MM-DD.md is a template, not a ref.")
    paths = se.cited_paths(body)
    assert "../scorecard/SKILL.md" in paths
    assert "tools/x_scorecard.py" in paths
    assert not any(p.startswith("http") for p in paths)
    assert not any("YYYY" in p for p in paths)
    assert "#sec" not in paths


# --- per-KPI: defect trigger + clean case -----------------------------------

def test_kpi_description_present() -> None:
    bad = se.kpi_description_present([_fact(description="too short")])
    assert bad["score"] == 0 and len(bad["defects"]) == 1
    good = se.kpi_description_present([_fact()])
    assert good["score"] == 100 and good["defects"] == []


def test_kpi_trigger_clause() -> None:
    bad = se.kpi_trigger_clause([_fact(description="Drives a thing and does stuff at length.")])
    assert len(bad["defects"]) == 1 and "WHEN" in bad["defects"][0]
    good = se.kpi_trigger_clause([_fact()])  # has "Use when"
    assert good["defects"] == []


def test_kpi_name_resolves() -> None:
    bad = se.kpi_name_resolves([_fact(name="alpha", fm_name="beta")])
    assert len(bad["defects"]) == 1 and "won't resolve" in bad["defects"][0]
    missing = se.kpi_name_resolves([_fact(fm_name="")])
    assert len(missing["defects"]) == 1 and "no `name:`" in missing["defects"][0]
    good = se.kpi_name_resolves([_fact(name="x", fm_name="x")])
    assert good["defects"] == []


def test_kpi_refs_resolve() -> None:
    bad = se.kpi_refs_resolve([_fact(dead_refs=["tools/gone.py", "docs/dead.md"])])
    assert len(bad["defects"]) == 2
    good = se.kpi_refs_resolve([_fact(dead_refs=[])])
    assert good["defects"] == []


def test_kpi_tools_scoped_only_gates_committers() -> None:
    # a committing skill with no allowed-tools is a defect
    bad = se.kpi_tools_scoped([_fact(commits=True, has_allowed_tools=False)])
    assert len(bad["defects"]) == 1
    # a NON-committing skill without allowed-tools is NOT applicable (no defect)
    na = se.kpi_tools_scoped([_fact(commits=False, has_allowed_tools=False)])
    assert na["defects"] == [] and na["score"] == 100
    good = se.kpi_tools_scoped([_fact(commits=True, has_allowed_tools=True)])
    assert good["defects"] == []


def test_kpi_commit_discipline_only_gates_committers() -> None:
    bad = se.kpi_commit_discipline([_fact(commits=True, body_has_pathdisc=False)])
    assert len(bad["defects"]) == 1 and "sweep" in bad["defects"][0]
    na = se.kpi_commit_discipline([_fact(commits=False, body_has_pathdisc=False)])
    assert na["defects"] == []
    good = se.kpi_commit_discipline([_fact(commits=True, body_has_pathdisc=True)])
    assert good["defects"] == []


def test_kpi_proof_step_only_gates_committers() -> None:
    bad = se.kpi_proof_step([_fact(commits=True, body_has_proof=False)])
    assert len(bad["defects"]) == 1
    na = se.kpi_proof_step([_fact(commits=False, body_has_proof=False)])
    assert na["defects"] == []


def test_kpi_anti_gaming_is_soft() -> None:
    k = se.kpi_anti_gaming([_fact(metric_driving=True, body_has_antigame=False)])
    assert k["defects"] == [] and len(k["soft"]) == 1  # SOFT: never debt
    na = se.kpi_anti_gaming([_fact(metric_driving=False, body_has_antigame=False)])
    assert na["soft"] == []


def test_kpi_context_budget_is_soft() -> None:
    k = se.kpi_context_budget([_fact(n_lines=400)])
    assert k["defects"] == [] and len(k["soft"]) == 1
    ok = se.kpi_context_budget([_fact(n_lines=120)])
    assert ok["soft"] == []


# --- the fold ---------------------------------------------------------------

def test_build_payload_counts_hard_debt_only() -> None:
    facts = [_fact(commits=True, has_allowed_tools=False,   # 1 tools_scoped
                   body_has_pathdisc=False,                  # 1 commit_discipline
                   dead_refs=["tools/x.py"],                 # 1 refs_resolve
                   metric_driving=True, body_has_antigame=False,  # SOFT (not debt)
                   n_lines=999)]                              # SOFT (not debt)
    kpis = [f(facts) for f in se.KPI_FUNCS]
    p = se.build_payload(workspace="/x", kpis=kpis, n_skills=1)
    assert p["corpus"]["skill_debt"] == 3        # only the HARD defects
    assert p["corpus"]["soft_signals"] == 2      # anti_gaming + context_budget
    assert p["ok"] is False and p["finding"] == "skill_debt"


def test_build_payload_clean_is_grade_a() -> None:
    kpis = [f([_fact()]) for f in se.KPI_FUNCS]
    p = se.build_payload(workspace="/x", kpis=kpis, n_skills=1)
    assert p["corpus"]["skill_debt"] == 0 and p["ok"] is True
    assert p["corpus"]["grade"] == "A" and p["corpus"]["score"] == 100.0


def test_error_payload() -> None:
    p = se.build_payload(workspace="/x", kpis=[], n_skills=0, error="boom")
    assert p["ok"] is False and p["verdict"] == "AUDIT_ERROR" and p["reason"] == "boom"


# --- live smoke on the real skill tree --------------------------------------

def test_live_payload_is_well_formed() -> None:
    p = se.collect(se.repo_root())
    for field in ("schema", "ok", "verdict", "finding", "reason", "next_action", "corpus", "kpis"):
        assert field in p, f"missing {field}"
    assert len(p["kpis"]) == len(se.KPI_WEIGHTS)
    for k in p["kpis"]:
        assert {"kpi", "group", "score", "detail", "defects", "soft"} <= set(k)
    # the real pack has skills, and skill_debt is the sum of HARD defects.
    assert p["corpus"]["skills"] >= 20
    assert p["corpus"]["skill_debt"] == sum(len(k["defects"]) for k in p["kpis"])


def main() -> int:
    failures: list[str] = []

    def check(name: str, fn) -> None:
        try:
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
