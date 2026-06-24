#!/usr/bin/env python3
"""Tests for the persona-readiness scorecard — does fak serve its top-10 personas?

Drives the PURE core with fixtures (no disk needed): every affordance check kind
(path_exists / any_path_exists / doc_mentions / fenced_command / command_resolves /
claim_section) in its met AND unmet form, the per-persona verdict fold, the overclaim
catch (declared 'served' on a missing affordance), well-formed defects, coverage, and
the fold to persona-debt. Closes with two load-bearing live smokes over the REAL
tracked tree: (1) every persona row's data is well-formed and (2) the real tree folds
to ZERO persona-debt — the proof that fak mechanically serves the personas it claims
to, and a regression sentinel for the day someone removes a release binary, a deploy
guide, or a determinism witness.

Run: `python tools/persona_readiness_scorecard_test.py`  (exit 0 = all pass),
or `python -m pytest tools/persona_readiness_scorecard_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import persona_readiness_scorecard as pr  # noqa: E402


# A tree double: the check kinds only call exists/doc_text/cmd_dirs/doc_verbs/section_tags.
def fake_tree(*, paths=(), docs=None, cmd_dirs=(), doc_verbs=(), section_tags=None) -> dict:
    pset = set(paths)
    dmap = dict(docs or {})
    return {
        "exists": lambda p: p in pset,
        "doc_text": lambda d: dmap.get(d, ""),
        "cmd_dirs": set(cmd_dirs),
        "doc_verbs": set(doc_verbs),
        "section_tags": dict(section_tags or {}),
    }


def good_row(**over) -> dict:
    """A minimal well-formed persona row whose one hard affordance is present."""
    row = {
        "id": "p", "persona": "P", "who": "who", "job": "job",
        "tier": "consume", "effort": "minimal", "entry_doc": "README.md",
        "verdict": "served", "gaps": [],
        "affordances": [
            {"id": "a", "kind": "path_exists", "target": "FILE", "need": "n", "severity": "hard"},
        ],
    }
    row.update(over)
    return row


# --- the small helpers ------------------------------------------------------

def test_grade_letter_bands() -> None:
    assert pr.grade_letter(100) == "A" and pr.grade_letter(90) == "A"
    assert pr.grade_letter(85) == "B" and pr.grade_letter(72) == "C"
    assert pr.grade_letter(61) == "D" and pr.grade_letter(40) == "F"


def test_parse_command() -> None:
    assert pr.parse_command("go run ./cmd/fak serve --addr x") == ("fak", "serve")
    assert pr.parse_command("go run ./cmd/fanbench") == ("fanbench", None)
    assert pr.parse_command("make ci") == (None, None)


def test_fenced_blocks_only_inside_fence() -> None:
    text = "prose run fak preflight here\n```\nfak preflight --policy p.json\n```\ntail"
    blocks = pr.fenced_blocks(text)
    assert len(blocks) == 1 and "fak preflight" in blocks[0]


def test_norm_and_section_match() -> None:
    assert pr.norm_section("## The product (whole surface)") == "the product"
    assert pr.section_match("The product", "the product") is True
    assert pr.section_match("Gateway", "the product") is False


# --- the check kinds: met AND unmet ----------------------------------------

def test_path_exists() -> None:
    t = fake_tree(paths=["FILE"])
    assert pr.eval_affordance({"kind": "path_exists", "target": "FILE"}, t)[0] is True
    assert pr.eval_affordance({"kind": "path_exists", "target": "NOPE"}, t)[0] is False
    # malformed: no target
    assert pr.eval_affordance({"kind": "path_exists"}, t)[0] is False


def test_any_path_exists() -> None:
    t = fake_tree(paths=["b"])
    assert pr.eval_affordance({"kind": "any_path_exists", "targets": ["a", "b"]}, t)[0] is True
    assert pr.eval_affordance({"kind": "any_path_exists", "targets": ["a", "c"]}, t)[0] is False


def test_doc_mentions_any_and_all() -> None:
    t = fake_tree(docs={"D": "has go install and @latest"})
    assert pr.eval_affordance({"kind": "doc_mentions", "doc": "D", "tokens": ["go install", "@latest"], "match": "all"}, t)[0] is True
    assert pr.eval_affordance({"kind": "doc_mentions", "doc": "D", "tokens": ["go install", "missing"], "match": "all"}, t)[0] is False
    assert pr.eval_affordance({"kind": "doc_mentions", "doc": "D", "tokens": ["nope", "@latest"], "match": "any"}, t)[0] is True
    # empty/missing doc never matches
    assert pr.eval_affordance({"kind": "doc_mentions", "doc": "X", "tokens": ["go install"]}, t)[0] is False
    # an empty token list never satisfies (no vacuous match=all pass over `[]`)
    assert pr.eval_affordance({"kind": "doc_mentions", "doc": "D", "tokens": [], "match": "all"}, t)[0] is False
    assert pr.eval_affordance({"kind": "doc_mentions", "doc": "D", "tokens": [], "match": "any"}, t)[0] is False


def test_fenced_command_requires_fence() -> None:
    prose = fake_tree(docs={"R": "just run fak preflight in prose"})
    fenced = fake_tree(docs={"R": "```\nfak preflight --policy p.json\n```"})
    aff = {"kind": "fenced_command", "docs": ["R"], "tokens": ["fak preflight"]}
    assert pr.eval_affordance(aff, prose)[0] is False
    assert pr.eval_affordance(aff, fenced)[0] is True


def test_command_resolves() -> None:
    t = fake_tree(cmd_dirs=["fak", "fanbench"], doc_verbs=["serve"])
    assert pr.eval_affordance({"kind": "command_resolves", "command": "go run ./cmd/fanbench"}, t)[0] is True
    assert pr.eval_affordance({"kind": "command_resolves", "command": "go run ./cmd/fak serve"}, t)[0] is True
    # fak verb not in the cli-reference word set -> unmet
    assert pr.eval_affordance({"kind": "command_resolves", "command": "go run ./cmd/fak bogusverb"}, t)[0] is False
    # cmd dir absent -> unmet
    assert pr.eval_affordance({"kind": "command_resolves", "command": "go run ./cmd/ghost"}, t)[0] is False


def test_claim_section() -> None:
    t = fake_tree(section_tags={"the product": {"SHIPPED"}})
    assert pr.eval_affordance({"kind": "claim_section", "section": "The product", "tags": ["SHIPPED"]}, t)[0] is True
    assert pr.eval_affordance({"kind": "claim_section", "section": "The product", "tags": ["STUB"]}, t)[0] is False
    assert pr.eval_affordance({"kind": "claim_section", "section": "Nonexistent"}, t)[0] is False


def test_unknown_kind_is_unmet() -> None:
    assert pr.eval_affordance({"kind": "telepathy"}, fake_tree())[0] is False


# --- per-persona verdict fold ----------------------------------------------

def test_expected_verdict_ladder() -> None:
    assert pr.expected_verdict(1.0, 4) == "served"
    assert pr.expected_verdict(0.8, 5) == "mostly-served"
    assert pr.expected_verdict(0.5, 4) == "partially-served"
    assert pr.expected_verdict(0.1, 5) == "unserved"
    assert pr.expected_verdict(1.0, 0) == "served"  # no hard affordances -> vacuously served


def test_score_persona_counts_hard_only() -> None:
    row = good_row(affordances=[
        {"id": "a", "kind": "path_exists", "target": "FILE", "need": "n", "severity": "hard"},
        {"id": "b", "kind": "path_exists", "target": "NOPE", "need": "n", "severity": "hard"},
        {"id": "c", "kind": "path_exists", "target": "NOPE2", "need": "n", "severity": "soft"},
    ])
    s = pr.score_persona(row, fake_tree(paths=["FILE"]))
    assert s["hard_total"] == 2 and s["hard_met"] == 1
    assert s["expected_verdict"] == "partially-served"  # 1/2 = 0.5
    assert len(s["soft"]) == 1 and s["soft"][0]["met"] is False


# --- the KPIs ---------------------------------------------------------------

def test_affordances_present_debt_per_missing_hard() -> None:
    row = good_row(affordances=[
        {"id": "a", "kind": "path_exists", "target": "NOPE", "need": "the thing", "severity": "hard"},
    ])
    s = pr.score_persona(row, fake_tree())
    k = pr.kpi_affordances_present([row], {"p": s})
    assert len(k["defects"]) == 1 and k["defects"][0].startswith("p: missing affordance 'a'")
    assert k["score"] == 0  # 0/1 hard present


def test_verdict_honest_catches_overclaim_only() -> None:
    # declared 'served' but only 0/1 hard present -> overclaim (hard defect).
    over = good_row(verdict="served", affordances=[
        {"id": "a", "kind": "path_exists", "target": "NOPE", "need": "n", "severity": "hard"}])
    s = pr.score_persona(over, fake_tree())
    k = pr.kpi_verdict_honest([over], {"p": s})
    assert len(k["defects"]) == 1 and "overclaim" in k["defects"][0]
    # declared 'unserved' but evidence 'served' -> underclaim (soft, not debt).
    under = good_row(verdict="unserved", affordances=[
        {"id": "a", "kind": "path_exists", "target": "FILE", "need": "n", "severity": "hard"}])
    s2 = pr.score_persona(under, fake_tree(paths=["FILE"]))
    k2 = pr.kpi_verdict_honest([under], {"p": s2})
    assert k2["defects"] == [] and len(k2["soft"]) == 1


def test_rows_well_formed_flags_bad_fields_and_affordances() -> None:
    bad = {
        "id": "p", "persona": "P", "who": "w", "job": "j",
        "tier": "ghost-tier", "effort": "minimal", "entry_doc": "x",
        "verdict": "served", "gaps": [],
        "affordances": [{"id": "a", "kind": "path_exists", "severity": "hard"}],  # no target, no need
    }
    k = pr.kpi_rows_well_formed([bad], tiers={"consume"})
    joined = " ".join(k["defects"])
    assert "tier 'ghost-tier' not declared" in joined
    assert "path_exists needs a 'target' string" in joined
    assert "missing field 'need'" in joined


def test_per_row_debt_attributes_empty_id() -> None:
    # a present-but-empty id is labeled row[i] by the well-formed KPI; per_row_debt
    # must bucket it the same way, not under the empty string.
    bad = {"id": "", "persona": "P", "who": "w", "job": "j", "tier": "consume",
           "effort": "minimal", "entry_doc": "x", "verdict": "served", "gaps": [],
           "affordances": [{"id": "a", "kind": "path_exists", "target": "NOPE",
                            "need": "n", "severity": "hard"}]}
    kpis = pr.run_kpis([bad], {"consume"}, {"row[0]": pr.score_persona(bad, fake_tree())})
    debt = pr.per_row_debt([bad], kpis)
    assert debt.get("row[0]", 0) >= 1 and "" not in debt


def test_coverage_counts_unpositioned_required() -> None:
    required = [{"id": "a", "name": "A", "tier": "consume"},
                {"id": "b", "name": "B", "tier": "operate"}]
    rows = [good_row(id="a")]
    cov = pr.coverage_report(required, rows)
    assert cov["coverage_debt"] == 1 and cov["uncovered"][0]["id"] == "b"
    assert cov["coverage_pct"] == 50.0


# --- the fold ---------------------------------------------------------------

def test_build_payload_clean_and_dirty() -> None:
    tree = fake_tree(paths=["FILE"])
    data = {
        "meta": {"as_of": "x"},
        "tiers": [{"id": "consume"}],
        "required_personas": [{"id": "p", "name": "P", "tier": "consume"}],
        "rows": [good_row()],
    }
    clean = pr.build_payload(workspace="w", data=data, tree=tree)
    assert clean["ok"] is True and clean["corpus"]["persona_debt"] == 0
    assert clean["corpus"]["grade"] == "A"
    assert clean["finding"] == "every_persona_served"

    # break the affordance + overclaim -> persona-debt rises, ok False.
    dirty_data = dict(data, rows=[good_row(affordances=[
        {"id": "a", "kind": "path_exists", "target": "NOPE", "need": "n", "severity": "hard"}])])
    dirty = pr.build_payload(workspace="w", data=dirty_data, tree=tree)
    assert dirty["ok"] is False
    # one missing affordance (reality) + one overclaim (honesty) = 2 persona-debt.
    assert dirty["corpus"]["persona_debt"] == 2
    assert dirty["corpus"]["debt_by_group"]["reality"] == 1
    assert dirty["corpus"]["debt_by_group"]["honesty"] == 1


def test_build_payload_coverage_only() -> None:
    tree = fake_tree(paths=["FILE"])
    data = {
        "tiers": [{"id": "consume"}],
        "required_personas": [{"id": "p", "name": "P", "tier": "consume"},
                              {"id": "q", "name": "Q", "tier": "consume"}],
        "rows": [good_row()],  # q is unpositioned
    }
    payload = pr.build_payload(workspace="w", data=data, tree=tree)
    assert payload["finding"] == "coverage_debt"
    assert payload["corpus"]["coverage_debt"] == 1
    assert payload["corpus"]["honesty_defects"] == 0


def test_audit_error_on_missing_data() -> None:
    payload = pr.build_payload(workspace="w", data=None, tree=fake_tree(), error="missing data directory")
    assert payload["ok"] is False and payload["verdict"] == "AUDIT_ERROR"


# --- live smokes over the REAL tree (the load-bearing regression sentinels) --

def _live_payload() -> dict:
    return pr.collect(pr.repo_root())


def test_live_rows_are_well_formed() -> None:
    payload = _live_payload()
    wf = next(k for k in payload["kpis"] if k["kpi"] == "rows_well_formed")
    assert wf["defects"] == [], f"malformed persona rows: {wf['defects']}"


def test_live_tree_serves_every_persona_zero_debt() -> None:
    """THE sentinel: the real tree must fold to zero persona-debt with all top-10
    personas positioned. If this fails, a persona affordance regressed (a removed
    release binary, a dead deploy guide, a missing witness) — add it back."""
    payload = _live_payload()
    c = payload["corpus"]
    assert c["coverage"]["required_total"] == 10, "the roster must hold the top-10 personas"
    assert c["coverage_debt"] == 0, f"unpositioned personas: {c['coverage']['uncovered']}"
    assert c["persona_debt"] == 0, (
        f"persona-debt {c['persona_debt']} — a persona affordance regressed; "
        f"run `python tools/persona_readiness_scorecard.py --critical`")
    assert payload["ok"] is True and c["grade"] == "A"


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERR  {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
