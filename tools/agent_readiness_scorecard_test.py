#!/usr/bin/env python3
"""Tests for the agent-readiness scorecard — the friction-debt measuring stick.

Drives the PURE checks with fixtures (no disk needed): each KPI's defect trigger
(no AGENTS.md / a thin one missing build-test-run, a missing harness config, a
dead orientation link, a first command that lives only in prose, a missing
install one-liner, an untagged claim, a missing integration recipe / leaf
scaffold / surfaced guardrail / contributor contract), the clean case for each,
and the fold to friction-debt + the verdict ladder. Closes with the load-bearing
live smoke: the REAL tracked tree must fold to ZERO friction-debt — the proof that
fak is, mechanically, a repo an agent can discover, adopt, and build on; and a
regression sentinel for the day someone removes an affordance.

Run: `python tools/agent_readiness_scorecard_test.py`  (exit 0 = all pass),
or `python -m pytest tools/agent_readiness_scorecard_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_readiness_scorecard as ar  # noqa: E402


# A full AGENTS.md fixture: identity + build + test + run + every surfaced rule.
GOOD_AGENTS = """# AGENTS.md
## What this project is
**fak** is an agent kernel.
```bash
go build ./cmd/fak
make test
go run ./cmd/fak preflight --policy p.json --tool t --args "{}"
```
Work on the trunk; the trunk guard refuses OFF_TRUNK commits.
Commit by explicit path (`git commit -- <paths>`), never `git add -A`.
Sign off with `git commit -s` (DCO).
Each claim in CLAIMS.md carries a tag. Add a feature as a leaf via new_leaf.py.
Writes outside the repo are refused by the repo-guard (OUT_OF_TREE_WRITE).
See CONTRIBUTING.md. Green = `make ci`.
"""


# --- the small helpers ------------------------------------------------------

def test_grade_letter_bands() -> None:
    assert ar.grade_letter(100) == "A" and ar.grade_letter(90) == "A"
    assert ar.grade_letter(85) == "B" and ar.grade_letter(72) == "C"
    assert ar.grade_letter(61) == "D" and ar.grade_letter(40) == "F"


def test_untagged_claims_counts_tags() -> None:
    text = ("- [SHIPPED] real thing\n"
            "- [SIMULATED] [STUB] two tags is malformed\n"
            "- [TODO] a bracketed claim with no status tag\n"
            "- a plain bullet (not a `- [` claim line) is not graded\n"
            "  - [STUB] indented claim is fine\n"
            "not a claim line at all\n")
    bad = ar.untagged_claims(text)
    # the two-tag line and the bracketed-but-untagged line are bad; the
    # single-tag lines and the plain bullet are fine.
    assert len(bad) == 2
    assert any("2 status tag" in b for b in bad)
    assert any("0 status tag" in b for b in bad)
    assert ar.untagged_claims("- [SHIPPED] all good\n- [STUB] also good") == []


def test_find_first_command_requires_a_fence() -> None:
    # The same token in PROSE does not count — an agent pastes a fenced line.
    prose = {"README.md": "Run fak preflight to see a denial."}
    assert ar.find_first_command(prose)[0] is False
    fenced = {"README.md": "Try it:\n```\nfak preflight --policy p.json\n```\n"}
    assert ar.find_first_command(fenced) == (True, "README.md")


def test_find_install_oneliner_needs_both_tokens() -> None:
    assert ar.find_install_oneliner({"README.md": "go install foo/cmd/fak@latest"})[0] is True
    # `go install` without @latest is not the resolvable one-liner.
    assert ar.find_install_oneliner({"README.md": "go install ./cmd/fak"})[0] is False


def test_find_identity_near_top_only() -> None:
    assert ar.find_identity({"AGENTS.md": "**fak** is an agent kernel."})[0] is True
    # a match buried past the head window does not count.
    buried = {"AGENTS.md": ("\n" * 60) + "fak is a kernel"}
    assert ar.find_identity(buried)[0] is False


def test_missing_guardrails_detects_gaps() -> None:
    assert ar.missing_guardrails(GOOD_AGENTS) == []
    thin = "# AGENTS.md\njust build and test, nothing about the rules"
    miss = ar.missing_guardrails(thin)
    assert len(miss) == len(ar.GUARDRAIL_CLUSTERS)


# --- per-KPI defect triggers + clean cases ----------------------------------

def test_agents_entrypoint_missing_file() -> None:
    k = ar.kpi_agents_entrypoint(None)
    assert k["score"] == 0 and len(k["defects"]) == 1 and "missing" in k["defects"][0]


def test_agents_entrypoint_missing_elements() -> None:
    # has identity but no build/test/run commands → 3 defects.
    k = ar.kpi_agents_entrypoint("**fak** is an agent kernel. No commands here.")
    assert len(k["defects"]) == 3
    assert ar.kpi_agents_entrypoint(GOOD_AGENTS)["defects"] == []


def test_agent_config_missing_and_clean() -> None:
    k = ar.kpi_agent_config(["Cursor (.cursorrules)"])
    assert len(k["defects"]) == 1 and "Cursor" in k["defects"][0]
    assert ar.kpi_agent_config([])["defects"] == [] and ar.kpi_agent_config([])["score"] == 100


def test_llms_map_hard_and_soft() -> None:
    missing = ar.kpi_llms_map({ar.LLMS_FILE: False, ar.LLMS_FULL_FILE: False})
    assert len(missing["defects"]) == 1 and len(missing["soft"]) == 1
    clean = ar.kpi_llms_map({ar.LLMS_FILE: True, ar.LLMS_FULL_FILE: True})
    assert clean["defects"] == [] and clean["soft"] == []


def test_identity_statement_kpi() -> None:
    assert ar.kpi_identity_statement(False, "")["defects"]
    assert ar.kpi_identity_statement(True, "AGENTS.md")["defects"] == []


def test_entry_links_resolve_kpi() -> None:
    k = ar.kpi_entry_links_resolve(["AGENTS.md -> docs/gone.md", "AGENTS.md -> x.md"])
    assert len(k["defects"]) == 2
    assert ar.kpi_entry_links_resolve([])["defects"] == []


def test_first_command_kpi() -> None:
    assert ar.kpi_first_command(False, "")["score"] == 20
    assert ar.kpi_first_command(True, "AGENTS.md")["defects"] == []


def test_install_oneliner_kpi() -> None:
    assert ar.kpi_install_oneliner(False, "")["defects"]
    assert ar.kpi_install_oneliner(True, "README.md")["defects"] == []


def test_honesty_ledger_missing_untagged_clean() -> None:
    assert ar.kpi_honesty_ledger(False, [])["score"] == 0
    untagged = ar.kpi_honesty_ledger(True, ["CLAIMS.md:5: 0 status tag(s): - foo"])
    assert len(untagged["defects"]) == 1 and untagged["score"] < 100
    assert ar.kpi_honesty_ledger(True, [])["defects"] == []


def test_integration_recipes_kpi() -> None:
    k = ar.kpi_integration_recipes(["Cursor", "MCP client"])
    assert len(k["defects"]) == 2
    assert ar.kpi_integration_recipes([])["score"] == 100


def test_extension_scaffold_kpi() -> None:
    assert len(ar.kpi_extension_scaffold(False, False)["defects"]) == 2
    assert ar.kpi_extension_scaffold(True, True)["defects"] == []


def test_guardrails_surfaced_kpi() -> None:
    k = ar.kpi_guardrails_surfaced(["DCO sign-off"])
    assert len(k["defects"]) == 1 and k["score"] < 100
    assert ar.kpi_guardrails_surfaced([])["score"] == 100


def test_contributor_contract_kpi() -> None:
    # present but unlinked + no green gate → 2 defects.
    k = ar.kpi_contributor_contract(True, False, False)
    assert len(k["defects"]) == 2
    assert ar.kpi_contributor_contract(True, True, True)["defects"] == []


def test_machine_consumable_is_soft() -> None:
    k = ar.kpi_machine_consumable(6, 8, ["tools/x_scorecard.py", "tools/y_scorecard.py"])
    assert k["defects"] == []          # SOFT: never hard debt
    assert k["score"] == 75 and len(k["soft"]) == 2


# --- fold to friction-debt --------------------------------------------------

def _clean_kpis() -> list[dict]:
    """Every KPI in its zero-defect (clean) state — the all-green tree."""
    return [
        ar.kpi_agents_entrypoint(GOOD_AGENTS),
        ar.kpi_agent_config([]),
        ar.kpi_llms_map({ar.LLMS_FILE: True, ar.LLMS_FULL_FILE: True}),
        ar.kpi_identity_statement(True, "AGENTS.md"),
        ar.kpi_entry_links_resolve([]),
        ar.kpi_first_command(True, "AGENTS.md"),
        ar.kpi_install_oneliner(True, "AGENTS.md"),
        ar.kpi_honesty_ledger(True, []),
        ar.kpi_integration_recipes([]),
        ar.kpi_extension_scaffold(True, True),
        ar.kpi_guardrails_surfaced([]),
        ar.kpi_contributor_contract(True, True, True),
        ar.kpi_machine_consumable(8, 8, []),
    ]


def test_build_payload_zero_debt_is_ok() -> None:
    p = ar.build_payload(workspace=".", kpis=_clean_kpis())
    assert p["ok"] is True and p["verdict"] == "OK" and p["finding"] == "agent_ready"
    assert p["corpus"]["friction_debt"] == 0 and p["corpus"]["grade"] == "A"
    assert p["corpus"]["score"] == 100.0
    # weights cover exactly the 13 KPIs and sum to 1.0 (the score can reach 100).
    assert abs(sum(ar.KPI_WEIGHTS.values()) - 1.0) < 1e-9
    assert set(ar.KPI_WEIGHTS) == {k["kpi"] for k in _clean_kpis()}


def test_build_payload_debt_drives_action_with_group_attribution() -> None:
    kpis = _clean_kpis()
    # break one affordance in each step: a missing harness config (discover), a
    # missing recipe (adopt), a missing scaffold piece (build) = 3 friction-debt.
    kpis[1] = ar.kpi_agent_config(["Cursor (.cursorrules)"])
    kpis[8] = ar.kpi_integration_recipes(["MCP client"])
    kpis[9] = ar.kpi_extension_scaffold(True, False)
    p = ar.build_payload(workspace=".", kpis=kpis)
    assert p["ok"] is False and p["finding"] == "friction_debt"
    assert p["corpus"]["friction_debt"] == 3
    assert p["corpus"]["debt_by_group"] == {"discover": 1, "adopt": 1, "build": 1}
    assert p["corpus"]["score"] < 100


def test_build_payload_error() -> None:
    p = ar.build_payload(workspace=".", kpis=[], error="not a git repo")
    assert p["ok"] is False and p["verdict"] == "AUDIT_ERROR"


# --- the load-bearing live smoke: the real tree is agent-ready --------------

def test_live_real_tree_is_agent_ready() -> None:
    root = ar.repo_root()
    if not (root / ar.AGENTS_FILE).exists():
        return  # tolerant: not in the repo tree
    p = ar.collect(root)
    assert p["schema"] == ar.SCHEMA, p
    # The shipped tree must carry ZERO friction-debt — every agent affordance
    # present, every orientation link alive, every claim tagged. A regression
    # sentinel: removing an affordance turns this red.
    assert p["corpus"]["friction_debt"] == 0, p["reason"]
    assert p["ok"] is True and p["corpus"]["grade"] == "A"
    # all three steps of the agent journey must score full.
    for g in ar.GROUPS:
        assert p["corpus"]["group_scores"][g] == 100, (g, p["corpus"]["group_scores"])


def test_live_payload_is_well_formed() -> None:
    root = ar.repo_root()
    p = ar.collect(root)
    for field in ("schema", "ok", "verdict", "finding", "reason", "next_action", "corpus", "kpis"):
        assert field in p, f"missing {field}"
    # exactly the 13 KPIs, each with the control-pane shape.
    assert len(p["kpis"]) == len(ar.KPI_WEIGHTS)
    for k in p["kpis"]:
        assert {"kpi", "group", "score", "detail", "defects", "soft"} <= set(k)


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
