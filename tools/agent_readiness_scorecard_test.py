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
Each claim in CLAIMS.md carries a tag. Add a feature as a leaf via fak new-leaf.
Writes outside the repo are refused by the repo-guard (OUT_OF_TREE_WRITE).
See CONTRIBUTING.md. Green = `make ci`.
"""

GOOD_CODEX = """# fak + OpenAI Codex
Codex is OpenAI's coding agent. Current Codex surfaces include the CLI, IDE
extension, Codex app, and cloud tasks.
Codex reads AGENTS.md before it works in this repo.
```bash
codex mcp add fak -- ./fak serve --stdio --policy examples/dev-agent-policy.json
codex exec --json "Summarize AGENTS.md"
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
```
Responses clients use /v1/responses; fak's current client-facing OpenAI-compatible
surface is Chat Completions, so current Codex users should use MCP first.
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


def test_find_identity_required_in_every_doc() -> None:
    # an identity near the top of ALL THREE orientation docs => none missing.
    every = {"AGENTS.md": "**fak** is an agent kernel.",
             ar.LLMS_FILE: "`fak` is an agent kernel.",
             "README.md": "`fak` is one Go binary you put in front of the agent."}
    present_in, missing = ar.find_identity(every)
    assert missing == [] and set(present_in) == set(ar.IDENTITY_DOCS)
    # README's "one Go binary" framing is a legitimate identity, not only kernel/gate.
    assert "README.md" in present_in
    # a match buried past the head window does not count — that doc is missing.
    buried = dict(every, **{"README.md": ("\n" * 60) + "fak is one Go binary"})
    _, missing2 = ar.find_identity(buried)
    assert missing2 == ["README.md"]
    # no identity anywhere => all three orientation docs are missing it.
    _, missing3 = ar.find_identity({})
    assert set(missing3) == set(ar.IDENTITY_DOCS)


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
    # missing from one orientation doc => one defect, score below 100, doc named.
    one = ar.kpi_identity_statement([ar.AGENTS_FILE, ar.LLMS_FILE], ["README.md"])
    assert len(one["defects"]) == 1 and one["score"] < 100
    assert "README.md" in one["defects"][0]
    # present near the top of all three orientation docs => clean, full score.
    allp = ar.kpi_identity_statement(list(ar.IDENTITY_DOCS), [])
    assert allp["defects"] == [] and allp["score"] == 100


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


def test_codex_recipe_currentness_detects_stale_or_missing_surface() -> None:
    assert ar.codex_recipe_gaps(GOOD_CODEX) == []
    clean = ar.kpi_codex_recipe_current([])
    assert clean["score"] == 100 and clean["defects"] == [] and clean["group"] == "adopt"

    stale = ("OpenAI has deprecated the standalone Codex API. Use gpt-4-turbo "
             "through a generic SDK.")
    gaps = ar.codex_recipe_gaps(stale)
    assert any("MCP server path" in g for g in gaps)
    assert any("stale Codex-era copy" in g for g in gaps)
    k = ar.kpi_codex_recipe_current(gaps)
    assert len(k["defects"]) == len(gaps) and k["score"] < 100


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

# --- the paste-and-run success KPIs (do the docs WORK, not just exist) ------

def test_path_operands_extracts_paths_skips_noise() -> None:
    block = ('cd fleet/fak\n'
             'go build ./cmd/fak\n'
             'go run ./cmd/fak preflight --policy examples/p.json\n'
             'curl https://example.com/x\n'        # a URL, not a path
             '  "method": "tools/call",\n'          # a JSON value, not a path
             '"self_modify_globs": ["internal/"]\n' # a JSON glob, not a path
             'make ci   # Windows: scripts/ci.ps1)\n')  # prose in a comment
    ops = ar._path_operands(block)
    assert "fleet/fak" in ops                 # the cd operand
    assert "./cmd/fak" in ops                 # ./ repo-relative
    assert "examples/p.json" in ops           # --policy operand
    assert "https://example.com/x" not in ops
    assert "tools/call" not in ops            # JSON value rejected (was quote-wrapped)
    assert not any("internal" in o for o in ops)   # JSON glob rejected
    assert not any("ci.ps1" in o for o in ops)     # inline-comment prose rejected


def test_template_slots_are_not_paths() -> None:
    assert ar._is_template_slot("<your-policy>")
    assert ar._is_template_slot("YOUR_ENV_VAR")
    assert not ar._is_template_slot("examples/p.json")
    # a bracketed slot inside a cd is an adapt-me marker, never a broken path.
    assert ar._path_operands("cd <your-clone>") == []


def test_fenced_paths_resolve_kpi() -> None:
    clean = ar.kpi_fenced_paths_resolve([])
    assert clean["defects"] == [] and clean["score"] == 100 and clean["group"] == "adopt"
    bad = ar.kpi_fenced_paths_resolve([
        "docs/integrations/cursor.md: `fleet/fak` — stale private-monorepo path",
        "docs/integrations/cursor.md: `/path/to/…` placeholder in a runnable command",
    ])
    assert len(bad["defects"]) == 2 and bad["score"] == 80


def test_first_command_runs_kpi() -> None:
    # present + policy resolves + no key => clean.
    ok = ar.kpi_first_command_runs(True, True, "examples/p.json", False)
    assert ok["defects"] == [] and ok["score"] == 100 and ok["group"] == "adopt"
    # a named policy that doesn't exist => one defect.
    miss = ar.kpi_first_command_runs(True, False, "examples/gone.json", False)
    assert len(miss["defects"]) == 1 and "doesn't exist" in miss["defects"][0]
    # a first command that secretly needs a key => one defect.
    keyed = ar.kpi_first_command_runs(True, True, "examples/p.json", True)
    assert len(keyed["defects"]) == 1 and "key" in keyed["defects"][0]
    # no first command present => this KPI abstains (first_command books the absence).
    absent = ar.kpi_first_command_runs(False, True, "", False)
    assert absent["defects"] == [] and absent["score"] == 100


def test_platform_guidance_consistent_kpi() -> None:
    # sells make ci AND names the Windows bridge => clean.
    assert ar.kpi_platform_guidance_consistent(True, True)["defects"] == []
    # sells make ci but no bridge => one defect (a Windows agent is stranded).
    bad = ar.kpi_platform_guidance_consistent(True, False)
    assert len(bad["defects"]) == 1 and bad["score"] == 40 and bad["group"] == "build"
    # doesn't sell make ci => nothing to reconcile, clean.
    assert ar.kpi_platform_guidance_consistent(False, False)["defects"] == []


# --- executable-truth KPIs (the pasted command actually runs / resolves) ----

# A main()-shaped switch: top-level verbs in the func main() body; the sub-command
# switch in another function (cmdPolicy) must NOT leak in.
MAIN_GO_FIXTURE = '''package main

func main() {
	switch os.Args[1] {
	case "serve":
		cmdServe(os.Args[2:])
	case "hook":
		cmdHook()
	case "hooks":
		cmdHooks(os.Args[2:])
	case "version", "-v", "--version":
		fmt.Println(v)
	case guard.TrampolineVerb:
		guard.LandlockTrampoline(os.Args[2:])
	default:
		usage()
	}
}

func cmdPolicy(argv []string) {
	switch argv[0] {
	case "budget":
		runBudget(argv[1:])
	}
}
'''


def test_dispatch_verbs_parses_only_main_switch() -> None:
    verbs = ar.dispatch_verbs(MAIN_GO_FIXTURE)
    assert {"serve", "hook", "hooks", "version", "-v", "--version"} <= verbs
    # the sub-command switch in cmdPolicy must NOT leak top-level verbs.
    assert "budget" not in verbs
    # a non-string case (guard.TrampolineVerb) contributes nothing.
    assert "TrampolineVerb" not in verbs
    # an unreadable/absent main.go => empty set (the KPI then abstains).
    assert ar.dispatch_verbs(None) == set()
    assert ar.dispatch_verbs("package main\n// no func main here\n") == set()


def test_command_verbs_extracts_command_context_only() -> None:
    # fenced commands (incl. `go run ./cmd/fak` and a `&&` chain) are captured.
    fenced = ("```bash\n"
              "go run ./cmd/fak preflight --policy p.json\n"
              "cd repo && fak serve --stdio\n"
              "# fak governs the call   <- a comment, not a command\n"
              "```\n")
    got = ar.command_verbs(fenced)
    assert "preflight" in got and "serve" in got
    assert "governs" not in got  # a fenced comment line is not a command

    # an inline `code` span that begins with the binary IS a command (the real-world
    # `fak hooks` case); prose and the Cursor `@fak` tool handle are NOT.
    prose = ("Run it via `fak hooks pre-commit` to gate the commit. "
             "fak governs the call (prose, no backticks). "
             "In Cursor: `@fak please adjudicate` is a tool mention, not a CLI call.")
    got = ar.command_verbs(prose)
    assert "hooks" in got            # inline command span captured
    assert "governs" not in got      # bare prose ignored
    assert "please" not in got       # @fak handle ignored

    # REGRESSION: a `fak <verb>` inline span that FOLLOWS a fenced block must still be
    # found — fence triple-backticks must not desync the inline-span pairing.
    desync = ("```bash\nfak serve\n```\n\nThen `fak hooks pre-commit` runs the gates.\n")
    assert "hooks" in ar.command_verbs(desync)


def test_command_verbs_hardening_against_audit_vectors() -> None:
    # An output BANNER (`fak guard: 131 decisions …`) is sample output, not a command —
    # the `:` right after the verb disqualifies it (would otherwise flag `fak summary:`).
    banner = "```\nfak guard: 131 kernel decision(s) — 121 allowed\nfak summary: done\n```\n"
    got = ar.command_verbs(banner)
    assert "guard" not in got and "summary" not in got

    # A leading env-var assignment must not hide the verb (`FOO=bar fak <verb>`).
    env = "```bash\nFAK_AUDIT_JOURNAL=x.jsonl fak badverb --flag\n```\n"
    assert "badverb" in ar.command_verbs(env)

    # A command after the ` -- ` wrapper boundary is still captured; the wrapped agent
    # name after `fak guard --` is NOT a fak verb and must not be captured.
    wrapped = "```bash\ncodex mcp add fak -- ./fak serve --stdio\nfak guard -- claude\n```\n"
    got = ar.command_verbs(wrapped)
    assert "serve" in got and "claude" not in got

    # Windows invocation forms resolve to the same verb set.
    win = "```powershell\n.\\fak serve --policy p.json\nfak.exe preflight --tool t\n```\n"
    got = ar.command_verbs(win)
    assert "serve" in got and "preflight" in got

    # A trailing inline comment naming a non-verb must not leak (`fak serve  # fak foo`).
    assert ar.command_verbs("```\nfak serve   # then fak foo later\n```\n") == ["serve"]


def test_command_verbs_resolve_kpi() -> None:
    clean = ar.kpi_command_verbs_resolve([])
    assert clean["defects"] == [] and clean["score"] == 100 and clean["group"] == "adopt"
    bad = ar.kpi_command_verbs_resolve(["AGENTS.md: fak hooks", "README.md: fak frobnicate"])
    assert len(bad["defects"]) == 2 and bad["score"] == 60


def test_unknown_command_verbs_abstains_without_dispatch() -> None:
    docs = {"AGENTS.md": "Run `fak hooks pre-commit`."}
    # no dispatch set parsed => abstain (don't blame the docs for a missing source).
    assert ar._unknown_command_verbs(docs, set()) == []
    # with a dispatch set lacking `hooks`, the verb is flagged once, deduped per doc.
    out = ar._unknown_command_verbs(docs, {"hook", "serve"})
    assert out == ["AGENTS.md: fak hooks"]
    # a verb that IS dispatched is not flagged.
    assert ar._unknown_command_verbs({"AGENTS.md": "`fak serve`"}, {"serve"}) == []


def test_recipe_links_resolve_kpi() -> None:
    clean = ar.kpi_recipe_links_resolve([])
    assert clean["defects"] == [] and clean["score"] == 100 and clean["group"] == "discover"
    bad = ar.kpi_recipe_links_resolve([
        "docs/integrations/cursor.md -> ../gone.md",
        "docs/integrations/claude.md -> missing.md",
    ])
    assert len(bad["defects"]) == 2 and bad["score"] == 76


def test_agent_config_valid_kpi() -> None:
    clean = ar.kpi_agent_config_valid([])
    assert clean["defects"] == [] and clean["score"] == 100 and clean["group"] == "discover"
    bad = ar.kpi_agent_config_valid([".mcp.json server 'x' names no launch command"])
    assert len(bad["defects"]) == 1 and bad["score"] == 66


def test_agent_config_integrity_reads_mcp_json() -> None:
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # absent .mcp.json => no integrity defect (presence is agent_config's job).
        assert ar._agent_config_integrity(root) == []
        # malformed JSON => one defect.
        (root / ".mcp.json").write_text("{ not json", encoding="utf-8")
        assert any("does not parse" in b for b in ar._agent_config_integrity(root))
        # a server with neither command nor url => one defect.
        (root / ".mcp.json").write_text('{"mcpServers": {"x": {"args": []}}}', encoding="utf-8")
        assert any("neither a launch command nor a url" in b for b in ar._agent_config_integrity(root))
        # a well-formed command server => clean.
        (root / ".mcp.json").write_text('{"mcpServers": {"dos": {"command": "dos-mcp"}}}', encoding="utf-8")
        assert ar._agent_config_integrity(root) == []
        # a remote url-based server (SSE/HTTP, no command) => clean.
        (root / ".mcp.json").write_text('{"mcpServers": {"r": {"url": "https://x/sse", "type": "sse"}}}', encoding="utf-8")
        assert ar._agent_config_integrity(root) == []
        # a `//` comment key inside mcpServers is idiomatic JSON, not a server => clean.
        (root / ".mcp.json").write_text('{"mcpServers": {"//": "a note", "dos": {"command": "dos-mcp"}}}', encoding="utf-8")
        assert ar._agent_config_integrity(root) == []


# --- hardened bar: refusal-recovery / quickstart signal / toolchain pin ------

def test_dos_reason_tokens_parses_reason_blocks() -> None:
    toml = ('[reasons.OFF_TRUNK]\nrefusal = true\n\n'
            '[reasons.ARCH_LAYER_VIOLATION]\nsummary = "x"\n\n'
            '[other.NOT_A_REASON]\nx = 1\n'
            '  [reasons.INDENTED]\n')  # an indented header is not a top-level block
    toks = ar.dos_reason_tokens(toml)
    assert toks == ["ARCH_LAYER_VIOLATION", "OFF_TRUNK"]   # sorted + deduped
    assert "NOT_A_REASON" not in toks and "INDENTED" not in toks
    # unreadable / empty source => abstain (empty set), never invented debt.
    assert ar.dos_reason_tokens(None) == []
    assert ar.dos_reason_tokens("no reason blocks here") == []


def test_unmapped_refusal_tokens_requires_nearby_recovery_cue() -> None:
    tokens = ["OFF_TRUNK", "ARCH_LAYER_VIOLATION", "OUT_OF_DIRECTION"]
    # a real recovery section (a cue word — "recover"/"fix" — near each token); the third
    # token is absent entirely => the only unmapped one.
    recovery = ("## How to recover from a refusal\n"
                "OFF_TRUNK: commit to main instead.\n"
                "ARCH_LAYER_VIOLATION: fix the import.\n")
    assert ar.unmapped_refusal_tokens(tokens, recovery) == ["OUT_OF_DIRECTION"]
    # ANTI-GAMING: a bare glossary that names the tokens with NO recovery cue nearby does
    # not count — both present tokens stay unmapped.
    glossary = "OFF_TRUNK is a thing.\nARCH_LAYER_VIOLATION is another thing.\n"
    assert set(ar.unmapped_refusal_tokens(tokens, glossary)) == set(tokens)
    assert ar.unmapped_refusal_tokens(tokens, None) == tokens   # no surface => all unmapped
    assert ar.unmapped_refusal_tokens([], recovery) == []


def test_refusal_recovery_mapped_kpi() -> None:
    clean = ar.kpi_refusal_recovery_mapped([], 6)
    assert clean["defects"] == [] and clean["score"] == 100 and clean["group"] == "build"
    bad = ar.kpi_refusal_recovery_mapped(["ARCH_LAYER_VIOLATION", "OUT_OF_DIRECTION"], 6)
    assert len(bad["defects"]) == 2 and bad["score"] == 67   # 4/6 mapped
    assert all("recovery" in d for d in bad["defects"])
    # ABSTAIN: dos.toml couldn't be parsed (total 0) => clean, never blame the docs.
    assert ar.kpi_refusal_recovery_mapped([], 0)["score"] == 100


def test_quickstart_signal_finds_signal_in_proof_block() -> None:
    with_signal = {"AGENTS.md": "Proof:\n```\nfak preflight --policy p.json   # -> DENY\n```\n"}
    assert ar.quickstart_signal(with_signal) == (True, True)
    no_signal = {"AGENTS.md": "Proof:\n```\nfak preflight --policy p.json\n```\n"}
    assert ar.quickstart_signal(no_signal) == (True, False)
    # ANTI-GAMING: an incidental `--allow-foo` flag (or a glued `->`) is NOT a success
    # marker — the anchored `-> ` / `exit code` form is required.
    flag_only = {"AGENTS.md": "Proof:\n```\nfak preflight --allow-foo --policy p.json\n```\n"}
    assert ar.quickstart_signal(flag_only) == (True, False)
    assert ar.quickstart_signal({"README.md": "no fenced first command at all"}) == (False, False)


def test_quickstart_success_signal_kpi() -> None:
    ok = ar.kpi_quickstart_success_signal(True, True)
    assert ok["defects"] == [] and ok["score"] == 100 and ok["group"] == "adopt"
    bad = ar.kpi_quickstart_success_signal(True, False)
    assert len(bad["defects"]) == 1 and bad["score"] == 40 and "expected" in bad["defects"][0]
    # no first command => abstain (first_command books the absence).
    absent = ar.kpi_quickstart_success_signal(False, False)
    assert absent["defects"] == [] and absent["score"] == 100


def test_toolchain_pinned_kpi() -> None:
    ok = ar.kpi_toolchain_pinned(True, True)
    assert ok["defects"] == [] and ok["score"] == 100 and ok["group"] == "adopt"
    # go.mod has no directive AND no doc names the version => 2 defects.
    both = ar.kpi_toolchain_pinned(False, False)
    assert len(both["defects"]) == 2 and both["score"] == 0
    # directive present but undocumented => one defect.
    one = ar.kpi_toolchain_pinned(True, False)
    assert len(one["defects"]) == 1 and one["score"] == 50


def _clean_kpis() -> list[dict]:
    """Every KPI in its zero-defect (clean) state — the all-green tree."""
    return [
        ar.kpi_agents_entrypoint(GOOD_AGENTS),
        ar.kpi_agent_config([]),
        ar.kpi_agent_config_valid([]),
        ar.kpi_llms_map({ar.LLMS_FILE: True, ar.LLMS_FULL_FILE: True}),
        ar.kpi_identity_statement(list(ar.IDENTITY_DOCS), []),
        ar.kpi_entry_links_resolve([]),
        ar.kpi_recipe_links_resolve([]),
        ar.kpi_first_command(True, "AGENTS.md"),
        ar.kpi_command_verbs_resolve([]),
        ar.kpi_first_command_runs(True, True, "examples/p.json", False),
        ar.kpi_install_oneliner(True, "AGENTS.md"),
        ar.kpi_honesty_ledger(True, []),
        ar.kpi_integration_recipes([]),
        ar.kpi_codex_recipe_current([]),
        ar.kpi_fenced_paths_resolve([]),
        ar.kpi_extension_scaffold(True, True),
        ar.kpi_guardrails_surfaced([]),
        ar.kpi_contributor_contract(True, True, True),
        ar.kpi_platform_guidance_consistent(True, True),
        ar.kpi_machine_consumable(8, 8, []),
        ar.kpi_refusal_recovery_mapped([], 6),
        ar.kpi_quickstart_success_signal(True, True),
        ar.kpi_toolchain_pinned(True, True),
    ]


def test_build_payload_zero_debt_is_ok() -> None:
    p = ar.build_payload(workspace=".", kpis=_clean_kpis())
    assert p["ok"] is True and p["verdict"] == "OK" and p["finding"] == "agent_ready"
    assert p["corpus"]["friction_debt"] == 0 and p["corpus"]["grade"] == "A"
    assert p["corpus"]["score"] == 100.0
    # weights cover exactly the KPI set and sum to 1.0 (the score can reach 100).
    assert abs(sum(ar.KPI_WEIGHTS.values()) - 1.0) < 1e-9
    assert set(ar.KPI_WEIGHTS) == {k["kpi"] for k in _clean_kpis()}


def test_build_payload_debt_drives_action_with_group_attribution() -> None:
    # break one affordance in each step (by name, not index, so a reorder can't
    # silently un-test this): a missing harness config (discover), a missing recipe
    # (adopt), a missing scaffold piece (build) = 3 friction-debt.
    swap = {
        "agent_config": ar.kpi_agent_config(["Cursor (.cursorrules)"]),
        "integration_recipes": ar.kpi_integration_recipes(["MCP client"]),
        "extension_scaffold": ar.kpi_extension_scaffold(True, False),
    }
    kpis = [swap.get(k["kpi"], k) for k in _clean_kpis()]
    p = ar.build_payload(workspace=".", kpis=kpis)
    assert p["ok"] is False and p["finding"] == "friction_debt"
    assert p["corpus"]["friction_debt"] == 3
    assert p["corpus"]["debt_by_group"] == {"discover": 1, "adopt": 1, "build": 1}
    assert p["corpus"]["score"] < 100


def test_build_payload_error() -> None:
    p = ar.build_payload(workspace=".", kpis=[], error="not a git repo")
    assert p["ok"] is False and p["verdict"] == "AUDIT_ERROR"


# --- the unbounded experience-frontier (the headline that is NOT a 0-100 grade) ----

def test_frontier_units_are_pinned() -> None:
    # The per-affordance weights are load-bearing (they set the frontier's units) and
    # must not drift silently — change one only on purpose. Pin the exact set + values.
    assert ar.FRONTIER_UNITS == {
        "integration_recipes": 8,
        "harness_configs": 10,
        "refusal_recoveries": 3,
        "machine_consumable": 2,
    }
    # every weight a positive int — a zero/negative weight would make a real affordance
    # contribute nothing (or subtract), defeating "add the affordance to climb".
    assert all(isinstance(w, int) and w > 0 for w in ar.FRONTIER_UNITS.values())


def test_frontier_harness_configs_supersets_the_required_core() -> None:
    # the frontier rewards BREADTH (optional, climb it); the gate requires only the core.
    # The breadth list must CONTAIN the required core so the core is never double-counted
    # and a core regression still shows up as friction-debt, not just a smaller frontier.
    core = {label for label, _ in ar.AGENT_CONFIGS}
    breadth = {label for label, _ in ar.FRONTIER_HARNESS_CONFIGS}
    assert core <= breadth
    assert len(breadth) > len(core)  # there is real breadth beyond the core


def test_experience_frontier_is_weighted_sum() -> None:
    facts = {"integration_recipes": 20, "harness_configs": 3,
             "refusal_recoveries": 16, "machine_consumable": 27}
    total, by_term = ar.experience_frontier(facts)
    assert by_term == {"integration_recipes": 160, "harness_configs": 30,
                       "refusal_recoveries": 48, "machine_consumable": 54}
    assert total == 292 == sum(by_term.values())


def test_experience_frontier_is_unbounded_above_100() -> None:
    # The whole point: it is NOT a 0-100 grade. A realistic tree blows past 100, and a
    # bigger tree blows past that — there is no ceiling to saturate against.
    small, _ = ar.experience_frontier({"integration_recipes": 20, "harness_configs": 3,
                                        "refusal_recoveries": 16, "machine_consumable": 27})
    big, _ = ar.experience_frontier({"integration_recipes": 200, "harness_configs": 30,
                                     "refusal_recoveries": 160, "machine_consumable": 270})
    assert small > 100
    assert big > small * 9  # 10x the affordances -> ~10x the frontier, no clamp


def test_experience_frontier_is_monotonic() -> None:
    # Adding ONE real affordance raises the frontier by exactly that dimension's weight —
    # the property that makes "climb the frontier by adding the affordance" honest.
    base = {"integration_recipes": 4, "harness_configs": 3,
            "refusal_recoveries": 16, "machine_consumable": 27}
    base_total, _ = ar.experience_frontier(base)
    for dim, w in ar.FRONTIER_UNITS.items():
        bumped = dict(base)
        bumped[dim] += 1
        bumped_total, _ = ar.experience_frontier(bumped)
        assert bumped_total == base_total + w, dim


def test_experience_frontier_missing_fact_is_zero() -> None:
    # A fact the shell couldn't resolve counts as zero, so the frontier fails LOW,
    # never high — it can't be inflated by an absent measurement.
    total, by_term = ar.experience_frontier({})
    assert total == 0 and all(v == 0 for v in by_term.values())
    assert set(by_term) == set(ar.FRONTIER_UNITS)  # every dim still reported


def test_build_payload_carries_frontier() -> None:
    facts = {"integration_recipes": 20, "harness_configs": 3,
             "refusal_recoveries": 16, "machine_consumable": 27}
    p = ar.build_payload(workspace=".", kpis=_clean_kpis(), facts=facts)
    c = p["corpus"]
    assert c["experience_frontier"] == 292
    assert c["frontier_by_term"]["integration_recipes"] == 160
    assert c["frontier_units"] == ar.FRONTIER_UNITS
    # the unbounded headline rides ALONGSIDE the bounded gate (back-compat preserved).
    assert c["score"] == 100.0 and c["friction_debt"] == 0


def test_build_payload_without_facts_is_back_compatible() -> None:
    # A caller that folds KPIs without the gather shell (the existing call shape) still
    # gets a well-formed payload: frontier 0, score/grade/friction-debt unchanged.
    p = ar.build_payload(workspace=".", kpis=_clean_kpis())
    c = p["corpus"]
    assert c["experience_frontier"] == 0
    assert c["score"] == 100.0 and c["grade"] == "A" and c["friction_debt"] == 0


def test_render_compare_reports_35pct_frontier_goal() -> None:
    base_facts = {"integration_recipes": 10, "harness_configs": 2,
                  "refusal_recoveries": 4, "machine_consumable": 5}
    base = ar.build_payload(workspace=".", kpis=_clean_kpis(), facts=base_facts)
    base_total = base["corpus"]["experience_frontier"]
    # current climbs >= 35% (add recipes until past 1.35x) -> "achieved".
    up = dict(base_facts, integration_recipes=base_facts["integration_recipes"] + 8)
    cur_up = ar.build_payload(workspace=".", kpis=_clean_kpis(), facts=up)
    assert cur_up["corpus"]["experience_frontier"] >= base_total * 1.35
    out_up = ar.render_compare(base, cur_up)
    assert "+35% achieved" in out_up, out_up
    # a smaller climb -> "not yet +35%".
    small = dict(base_facts, integration_recipes=base_facts["integration_recipes"] + 1)
    cur_small = ar.build_payload(workspace=".", kpis=_clean_kpis(), facts=small)
    out_small = ar.render_compare(base, cur_small)
    assert "not yet +35%" in out_small, out_small


def test_is_substantive_recipe_requires_real_content() -> None:
    # the frontier's top term must be a REAL affordance, never a stub — "add the real
    # affordance, never game the check", enforced for the unbounded headline too.
    assert ar._is_substantive_recipe("") is False
    assert ar._is_substantive_recipe("# Title only\n") is False     # too short
    assert ar._is_substantive_recipe("x" * 300) is False            # long but no fence/link
    long_link = "# fak + Foo\n" + ("Point your agent at fak. " * 20) + "\n[setup](./s.md)\n"
    assert ar._is_substantive_recipe(long_link) is True             # length + a link
    long_fence = "# fak + Bar\n" + ("Run the proof. " * 20) + "\n```\nfak preflight\n```\n"
    assert ar._is_substantive_recipe(long_fence) is True            # length + a fence


def test_render_compare_zero_baseline_and_gate_line() -> None:
    cur = ar.build_payload(workspace=".", kpis=_clean_kpis(),
                           facts={"integration_recipes": 5, "harness_configs": 1,
                                  "refusal_recoveries": 2, "machine_consumable": 3})
    base0 = ar.build_payload(workspace=".", kpis=_clean_kpis())  # frontier 0 (no facts)
    out = ar.render_compare(base0, cur)
    assert "no prior frontier" in out, out          # the bf == 0 verdict branch
    assert "(gate)" in out, out                      # the friction-debt 2x gate still reports


def test_render_compare_signed_score_delta_on_regression() -> None:
    # a score regression must render a single-signed delta, never "(+-5.0)".
    hi = ar.build_payload(workspace=".", kpis=_clean_kpis())  # score 100
    lo_kpis = [ar.kpi_first_command(False, "") if k["kpi"] == "first_command" else k
               for k in _clean_kpis()]
    lo = ar.build_payload(workspace=".", kpis=lo_kpis)        # score < 100
    out = ar.render_compare(hi, lo)
    assert "(+-" not in out, out


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
    # the unbounded headline rides on the live tree too: a real, >100 frontier whose
    # per-term breakdown sums to it (the surface an agent gains, with headroom to climb).
    c = p["corpus"]
    assert isinstance(c["experience_frontier"], int) and c["experience_frontier"] > 100, c
    assert sum(c["frontier_by_term"].values()) == c["experience_frontier"]
    assert set(c["frontier_by_term"]) == set(ar.FRONTIER_UNITS)


def test_live_payload_is_well_formed() -> None:
    root = ar.repo_root()
    p = ar.collect(root)
    for field in ("schema", "ok", "verdict", "finding", "reason", "next_action", "corpus", "kpis"):
        assert field in p, f"missing {field}"
    # exactly the weighted KPI set, each with the control-pane shape.
    assert len(p["kpis"]) == len(ar.KPI_WEIGHTS)
    for k in p["kpis"]:
        assert {"kpi", "group", "score", "detail", "defects", "soft"} <= set(k)


def test_live_identity_resolves_in_every_orientation_doc() -> None:
    """The instant-orientation DoD (#1119): the one-sentence 'fak is a/an …' identity
    must resolve near the top of EVERY orientation doc, so a cold agent quotes it
    wherever it enters the repo — AGENTS.md, llms.txt, OR README, not just whichever
    doc happens to be read first. A regression sentinel for the day a doc rewrite
    drops the identity line from one of the three."""
    root = ar.repo_root()
    if not (root / ar.AGENTS_FILE).exists():
        return  # tolerant: not in the repo tree
    texts = {d: ar._safe_read(root / d) for d in ar.IDENTITY_DOCS}
    present_in, missing = ar.find_identity(texts)
    assert missing == [], f"identity missing from {missing} (present in {present_in})"
    assert set(present_in) == set(ar.IDENTITY_DOCS), present_in


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
