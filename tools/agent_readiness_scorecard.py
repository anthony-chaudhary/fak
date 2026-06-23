#!/usr/bin/env python3
"""Agent-readiness scorecard — the measuring stick for how attractive fak is to an AI agent.

The sibling scorecards each point at a surface a *human* reviewer cares about:
``repo_hygiene_scorecard`` grades the tree's shape, ``code_quality_scorecard``
grades the Go module, ``doc_appeal_scorecard`` grades a doc's prose,
``industry_scorecard`` grades the competitive claims. None of them grade the thing
that decides whether the project even gets *built on* in an agent-first world: can
an autonomous coding agent — Claude Code, OpenAI Codex, Cursor, an MCP client —
**(1) discover** fak, **(2) want** to adopt and build on it, and **(3) do so
effectively and easily**? That used to be a vibe ("we have an AGENTS.md, we're
fine"). This is the number.

It scores the git-tracked tree on thirteen mechanical KPIs in three groups — the
exact three steps an agent walks — folds them into a weighted score and an A-F
grade, and counts **friction-debt**: the total of concrete, re-derivable defects
that make fak harder for an agent to find, trust, and build on. Each is a defect
you fix by *adding the missing affordance an agent reaches for*, not by writing
more prose.

  DISCOVER      — an agent can find fak and orient in seconds
    agents_entrypoint  AGENTS.md exists and carries the agents.md contract: what
                       this is, plus build / test / run commands an agent can run
    agent_config       the zero-setup configs an agent's harness auto-loads —
                       .mcp.json (MCP), .cursorrules (Cursor), copilot-instructions
                       (Copilot) — so "point your agent here" needs no hand-wiring
    llms_map           llms.txt (the answer-engine / agent doc-map) is present
    identity_statement a one-sentence "what fak is" an agent can quote verbatim
    entry_links_resolve every local link on the orientation path resolves (no 404
                       when an agent follows AGENTS.md / the integration index)

  ADOPT         — an agent has a low-friction path to first run, and a reason to trust it
    first_command      a copy-pasteable first command that needs no key/model/GPU
    install_oneliner   the one-line install that resolves (module at the repo root)
    honesty_ledger     CLAIMS.md exists and every claim carries one status tag — an
                       agent trusts a tagged ledger, not an un-caveated promise
    integration_recipes a per-agent recipe for each family an agent identifies with
                       (Claude, Codex/OpenAI, Cursor, MCP) — "point your agent here"

  BUILD         — an agent can contribute effectively and the guards won't ambush it
    extension_scaffold the additive extension path exists (new_leaf.py + EXTENDING)
    guardrails_surfaced the enforced rules that WILL bite an agent are documented up
                       front (trunk-only, commit-by-path, DCO sign-off, claim tag,
                       leaf/ABI discipline, the out-of-tree write guard) — an agent
                       that knows them doesn't fight the guard
    contributor_contract CONTRIBUTING.md is present + linked, and a one-command green
                       gate (make ci / make test) is documented (the feedback loop)
    machine_consumable an agent acts on structure, not prose — how much of the
                       measurement family speaks machine-readable JSON (SOFT)

The headline metric is **friction-debt**: the count of concrete HARD defects above.
Driving friction-debt to zero means an agent that lands in this repo cold can
discover what fak is, trust it enough to adopt it, and build on it without tripping
over an undocumented guard or a dead link. The companion process — the
``/agent-readiness`` skill — runs this, retires the worst-first defect by adding the
missing agent affordance, and re-runs to prove the drop. It folds into the unified
``scorecard_control_pane`` alongside the other inward sticks.

Deterministic + read-only by construction: it reads the git-tracked tree (so two
clones of the same commit score identically) and edits nothing. Run from the repo
ROOT::

    python tools/agent_readiness_scorecard.py                 # human scorecard
    python tools/agent_readiness_scorecard.py --json          # machine payload
    python tools/agent_readiness_scorecard.py --markdown      # the committed snapshot body
    python tools/agent_readiness_scorecard.py --compare base.json   # prove the debt moved
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fak-agent-readiness-scorecard/1"
GENERATED_SNAPSHOT = "docs/AGENT-READINESS-SCORECARD.md"

# ---------------------------------------------------------------------------
# The contract an agent expects. Each constant is a deliberate, named affordance
# an autonomous agent reaches for — never a hand-picked file list where a rule
# would do. A fork that drops one of these scores lower, by construction.
# ---------------------------------------------------------------------------

# The agents.md machine-read entry point (https://agents.md). Its presence AND
# its content (identity + the three command classes) are what an agent reads first.
AGENTS_FILE = "AGENTS.md"
# The answer-engine / agent doc-map convention. llms-full.txt is the inlined body.
LLMS_FILE = "llms.txt"
LLMS_FULL_FILE = "llms-full.txt"

# The honesty ledger and its claim-line rule (mirrors `make claims-lint`): every
# line beginning with `- [` carries exactly one of these tags.
CLAIMS_FILE = "CLAIMS.md"
CLAIM_TAGS = ("[SHIPPED]", "[SIMULATED]", "[STUB]")
CLAIM_LINE = re.compile(r"^\s*- \[")

# The additive-extension path: a stamper for a conforming skeleton + the doc that
# teaches "add a leaf, don't edit core".
LEAF_SCAFFOLD = "tools/new_leaf.py"
EXTENDING_FILE = "EXTENDING.md"
CONTRIBUTING_FILE = "CONTRIBUTING.md"

# The zero-setup configs an agent's harness auto-discovers on entry. Each entry is
# (harness label, [candidate paths]); the config is present if ANY path exists.
# These are what make "point your agent at the repo" need no hand-wiring — the
# product thesis (fak fronts the agent) realized in the repo's own dotfiles.
AGENT_CONFIGS: list[tuple[str, list[str]]] = [
    ("MCP clients (.mcp.json)", [".mcp.json", "examples/mcp/.mcp.json"]),
    ("Cursor (.cursorrules)", [".cursorrules", ".cursor/rules"]),
    ("GitHub Copilot (copilot-instructions)", [".github/copilot-instructions.md"]),
]

# Per-agent integration recipes — one for each family an agent identifies with.
# Each entry is (label, [candidate paths]); the recipe is present if ANY exists.
REQUIRED_RECIPES: list[tuple[str, list[str]]] = [
    ("Claude Code / Anthropic", ["docs/integrations/claude.md"]),
    ("OpenAI Codex / OpenAI", ["docs/integrations/openai-codex.md"]),
    ("Cursor", ["docs/integrations/cursor.md"]),
    ("MCP client", ["examples/mcp/README.md", "docs/integrations/mcp.md"]),
]

# The enforced rules an agent WILL hit — each must be surfaced in AGENTS.md so an
# agent learns it before the guard refuses it. Each cluster matches if ANY synonym
# is present (case-insensitive). These are the closed-vocabulary refusals the
# pre-commit guards raise one commit at a time; an undocumented one is an ambush.
GUARDRAIL_CLUSTERS: list[tuple[str, list[str]]] = [
    ("trunk-only (no feature branch)", ["off_trunk", "on the trunk", "feature branch", "trunk guard"]),
    ("commit by explicit path (no add -A)", ["git add -a", "explicit path", "commit -- <", "commit by explicit"]),
    ("DCO sign-off", ["git commit -s", "sign off", "sign-off", "dco"]),
    ("tagged claims ledger", ["claims.md", "claims-lint", "[shipped]"]),
    ("leaf / frozen-ABI discipline", ["new_leaf", "as a leaf", "frozen abi", "additive-only"]),
    # The repo-guard PreToolUse hook denies writes that resolve outside the repo
    # (a ../sibling path, an absolute sibling, even `> /dev/null`). It is on by
    # default and silently bites an agent's Bash/Write/Edit — so the entry point
    # must teach it. (docs/repo-guard.md; surfaced 2026-06-23 after it bit a session.)
    ("out-of-tree write guard", ["out_of_tree", "out-of-tree", "fak_repo_guard", "repo-guard", "outside the repo", "outside this repo"]),
]

# The first-command signal: a canonical command an agent can paste with NO key,
# model, or GPU. fak's is the preflight / offline proof.
FIRST_COMMAND_TOKENS = ["fak preflight", "fak agent --offline", "preflight --policy"]
FIRST_COMMAND_DOCS = [AGENTS_FILE, "README.md", "START-HERE.md", "GETTING-STARTED.md"]

# The install one-liner: `go install <module>/cmd/fak@latest` (module at the root).
INSTALL_TOKENS = ["go install", "@latest"]
INSTALL_DOCS = [AGENTS_FILE, "README.md", "GETTING-STARTED.md", "INSTALL.md"]

# The one-command green gate (the feedback loop an agent runs to know it's safe).
GREEN_GATE_TOKENS = ["make ci", "make test", "make test-fast", "scripts/ci.ps1", "./test.ps1"]

# The orientation path whose every local link must resolve — the docs an agent
# clicks through while adopting. A dead link here is friction at the worst moment.
ORIENTATION_DOCS = [AGENTS_FILE, "docs/integrations/README.md"]

# The measurement family an agent drives — used by `machine_consumable` to ask
# "can an agent parse what these tools emit". A glob over the tracked tree.
TOOL_FAMILY_GLOB = "tools/*scorecard*.py"

# Identity: a one-line "<name> is a/an <kind>" an agent can lift as the answer to
# "what is fak". Matched near the top of the orientation docs.
IDENTITY_RE = re.compile(r"\bfak\b[^.\n]{0,60}?\bis\b[^.\n]{0,80}?"
                         r"(kernel|firewall|gate|gateway|proxy)", re.IGNORECASE)
IDENTITY_DOCS = [AGENTS_FILE, LLMS_FILE, "README.md"]
IDENTITY_HEAD_LINES = 40

GROUPS = ("discover", "adopt", "build")
KPI_GROUP: dict[str, str] = {
    "agents_entrypoint": "discover",
    "agent_config": "discover",
    "llms_map": "discover",
    "identity_statement": "discover",
    "entry_links_resolve": "discover",
    "first_command": "adopt",
    "install_oneliner": "adopt",
    "honesty_ledger": "adopt",
    "integration_recipes": "adopt",
    "extension_scaffold": "build",
    "guardrails_surfaced": "build",
    "contributor_contract": "build",
    "machine_consumable": "build",
}
KPI_WEIGHTS: dict[str, float] = {
    # discover (0.34)
    "agents_entrypoint": 0.11,
    "agent_config": 0.07,
    "llms_map": 0.06,
    "entry_links_resolve": 0.05,
    "identity_statement": 0.05,
    # adopt (0.33)
    "first_command": 0.10,
    "honesty_ledger": 0.08,
    "integration_recipes": 0.08,
    "install_oneliner": 0.07,
    # build (0.33)
    "guardrails_surfaced": 0.10,
    "contributor_contract": 0.08,
    "extension_scaffold": 0.08,
    "machine_consumable": 0.07,
}

_LINK_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<target>[^)]+)\)")
_FENCE_RE = re.compile(r"^(```|~~~)")


# ---------------------------------------------------------------------------
# Small pure helpers (the testable core).
# ---------------------------------------------------------------------------

def _clamp(score: float) -> int:
    return int(max(0, min(100, round(score))))


def grade_letter(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _has(text: str | None, *tokens: str) -> bool:
    """True if the text (case-insensitive) contains any of the tokens."""
    if not text:
        return False
    low = text.lower()
    return any(t.lower() in low for t in tokens)


def _fenced_blocks(text: str) -> list[str]:
    """The contents of every fenced code block — where a copy-pasteable command
    lives. A token found only in prose does not count as a runnable command."""
    blocks: list[str] = []
    cur: list[str] = []
    in_fence = False
    for raw in text.split("\n"):
        if _FENCE_RE.match(raw.strip()):
            if in_fence:
                blocks.append("\n".join(cur))
                cur = []
            in_fence = not in_fence
            continue
        if in_fence:
            cur.append(raw)
    return blocks


def find_first_command(texts: dict[str, str]) -> tuple[bool, str]:
    """Is a no-key/no-model/no-GPU first command present inside a fenced block of
    an adoption doc? Returns (found, where). A command must be inside a fence — an
    agent pastes a fenced line, not a sentence."""
    for doc in FIRST_COMMAND_DOCS:
        for block in _fenced_blocks(texts.get(doc, "")):
            if _has(block, *FIRST_COMMAND_TOKENS):
                return True, doc
    return False, ""


def find_install_oneliner(texts: dict[str, str]) -> tuple[bool, str]:
    """Is the `go install …@latest` one-liner present (in any block or prose) in an
    install doc? Both tokens must co-occur in the same doc."""
    for doc in INSTALL_DOCS:
        t = texts.get(doc, "")
        if all(_has(t, tok) for tok in INSTALL_TOKENS):
            return True, doc
    return False, ""


def find_identity(texts: dict[str, str]) -> tuple[bool, str]:
    """Is a one-sentence '<fak> is a/an <kind>' identity near the top of an
    orientation doc — the answer an agent quotes for 'what is fak'?"""
    for doc in IDENTITY_DOCS:
        head = "\n".join(texts.get(doc, "").splitlines()[:IDENTITY_HEAD_LINES])
        if IDENTITY_RE.search(head):
            return True, doc
    return False, ""


def untagged_claims(claims_text: str | None) -> list[str]:
    """Claim lines (`- [ …`) in the ledger that do NOT carry exactly one status
    tag — the claims-lint rule, as the measure of a trustworthy ledger."""
    if not claims_text:
        return []
    bad: list[str] = []
    for i, line in enumerate(claims_text.splitlines(), 1):
        if not CLAIM_LINE.match(line):
            continue
        n = sum(line.count(tag) for tag in CLAIM_TAGS)
        if n != 1:
            snippet = line.strip()[:80]
            bad.append(f"CLAIMS.md:{i}: {n} status tag(s) (need exactly 1): {snippet}")
    return bad


def missing_recipes(present: dict[str, bool]) -> list[str]:
    """Agent families with no integration recipe on disk. ``present`` maps each
    REQUIRED_RECIPES label to whether any candidate path exists."""
    return [label for label, _ in REQUIRED_RECIPES if not present.get(label)]


def missing_agent_configs(present: dict[str, bool]) -> list[str]:
    """Agent harnesses whose auto-discovered config file is absent. ``present``
    maps each AGENT_CONFIGS label to whether any candidate path exists."""
    return [label for label, _ in AGENT_CONFIGS if not present.get(label)]


def missing_guardrails(agents_text: str | None) -> list[str]:
    """Enforced rules NOT surfaced in AGENTS.md (an agent learns each before the
    guard refuses it). One miss per undocumented rule cluster."""
    return [label for label, syns in GUARDRAIL_CLUSTERS if not _has(agents_text, *syns)]


# ---------------------------------------------------------------------------
# Per-KPI pure checks. Each returns
#   {kpi, group, score (0-100 int), detail, defects: [str], soft: [str]}
# defects = HARD units of friction-debt; soft = score-only judgment nudges.
# ---------------------------------------------------------------------------

def kpi_agents_entrypoint(agents_text: str | None) -> dict[str, Any]:
    """AGENTS.md is the agents.md machine-read entry point. It must exist, name
    what the project is, and carry the three command classes an agent runs:
    build, test, and a runnable first verb. Each missing element is one defect —
    an agent that can't build/test/run from the entry point is stuck on step one."""
    defects: list[str] = []
    if not agents_text or not agents_text.strip():
        defects.append(f"missing {AGENTS_FILE} — the agents.md machine-read entry point (agents.md)")
        return {"kpi": "agents_entrypoint", "group": "discover", "score": 0,
                "detail": f"no {AGENTS_FILE} at the repo root", "defects": defects, "soft": []}
    if not _has(agents_text, "what this project is", "what this is", "is an agent", "**fak**"):
        defects.append(f"{AGENTS_FILE} does not state what the project is (a 'what this is' line)")
    if not _has(agents_text, "go build", "make build", "make test-fast"):
        defects.append(f"{AGENTS_FILE} has no build command (go build / make …)")
    if not _has(agents_text, "make test", "go test", "test.ps1", "make ci"):
        defects.append(f"{AGENTS_FILE} has no test command (make test / go test / ./test.ps1)")
    if not _has(agents_text, "go run ./cmd/fak", "fak preflight", "fak serve", "fak agent"):
        defects.append(f"{AGENTS_FILE} has no runnable first verb (go run ./cmd/fak … / fak …)")
    return {"kpi": "agents_entrypoint", "group": "discover",
            "score": _clamp(100 - 22 * len(defects)),
            "detail": (f"{len(defects)} missing entry-point element(s)" if defects
                       else "AGENTS.md states identity + build/test/run"),
            "defects": defects, "soft": []}


def kpi_agent_config(missing: list[str]) -> dict[str, Any]:
    """The zero-setup config files an agent's harness auto-loads on entry. Their
    presence is what turns 'put fak in front of your agent' from a wiring chore
    into a drop-in — and is fak's own product thesis applied to its own repo. Each
    missing harness config is one unit: that harness's users can't drop in cold."""
    defects = [f"no auto-discovered config for {label} — add it so that harness drops in with no setup"
               for label in missing]
    covered = len(AGENT_CONFIGS) - len(missing)
    return {"kpi": "agent_config", "group": "discover",
            "score": _clamp(100 * covered / max(1, len(AGENT_CONFIGS))),
            "detail": f"{covered}/{len(AGENT_CONFIGS)} agent harnesses have a zero-setup config",
            "defects": defects, "soft": []}


def kpi_llms_map(present: dict[str, bool]) -> dict[str, Any]:
    """The llms.txt doc-map is how an answer engine and an agent discover the doc
    set. Its absence is hard friction; a missing llms-full.txt (the inlined body)
    is a soft nudge."""
    defects: list[str] = []
    soft: list[str] = []
    if not present.get(LLMS_FILE):
        defects.append(f"missing {LLMS_FILE} — the agent / answer-engine doc-map")
    if not present.get(LLMS_FULL_FILE):
        soft.append(f"no {LLMS_FULL_FILE} (the inlined full doc-map an answer engine ingests)")
    return {"kpi": "llms_map", "group": "discover",
            "score": _clamp(100 - 60 * len(defects) - 8 * len(soft)),
            "detail": (f"{LLMS_FILE} present" if not defects else f"no {LLMS_FILE}"),
            "defects": defects, "soft": soft}


def kpi_identity_statement(found: bool, where: str) -> dict[str, Any]:
    """A one-sentence '<fak> is a/an <kind>' an agent can quote as the answer to
    'what is this'. Without it an agent has to infer the pitch, and infers wrong."""
    defects: list[str] = []
    if not found:
        defects.append("no one-sentence 'fak is a/an <kernel/gate/…>' identity near the "
                       f"top of {', '.join(IDENTITY_DOCS)} — add a quotable one-liner")
    return {"kpi": "identity_statement", "group": "discover",
            "score": 100 if found else 30,
            "detail": (f"identity statement found in {where}" if found
                       else "no quotable one-sentence identity"),
            "defects": defects, "soft": []}


def kpi_entry_links_resolve(dead: list[str]) -> dict[str, Any]:
    """Every local link on the orientation path (AGENTS.md + the integration index)
    must resolve on disk. A 404 when an agent follows the map is friction exactly
    when it is trying to adopt. ``dead`` is a list of '<doc> -> <target>'."""
    defects = [f"dead orientation link: {d}" for d in sorted(dead)]
    return {"kpi": "entry_links_resolve", "group": "discover",
            "score": _clamp(100 - 14 * len(defects)),
            "detail": (f"{len(defects)} dead orientation link(s)" if defects
                       else "every orientation link resolves"),
            "defects": defects, "soft": []}


def kpi_first_command(found: bool, where: str) -> dict[str, Any]:
    """A copy-pasteable first command that needs no key, model, or GPU — the
    30-second proof an agent runs to see fak work before committing to it."""
    defects: list[str] = []
    if not found:
        defects.append("no copy-pasteable no-key/no-model/no-GPU first command in a fenced "
                       f"block of {', '.join(FIRST_COMMAND_DOCS)} (e.g. `fak preflight …`)")
    return {"kpi": "first_command", "group": "adopt",
            "score": 100 if found else 20,
            "detail": (f"first command present in {where}" if found
                       else "no runnable no-setup first command"),
            "defects": defects, "soft": []}


def kpi_install_oneliner(found: bool, where: str) -> dict[str, Any]:
    """The one-line install that resolves because the module is at the repo root
    (`go install …/cmd/fak@latest`). The shortest path from 'interested' to 'have
    the binary'."""
    defects: list[str] = []
    if not found:
        defects.append("no one-line install (`go install …@latest`) in "
                       f"{', '.join(INSTALL_DOCS)} — give an agent the one-command install")
    return {"kpi": "install_oneliner", "group": "adopt",
            "score": 100 if found else 40,
            "detail": (f"install one-liner present in {where}" if found
                       else "no one-line install"),
            "defects": defects, "soft": []}


def kpi_honesty_ledger(present: bool, untagged: list[str]) -> dict[str, Any]:
    """CLAIMS.md, with every claim carrying exactly one status tag, is what lets an
    agent trust a capability statement instead of discounting it. A missing ledger
    is hard friction; each untagged claim is one unit (capped so it can't dominate)."""
    defects: list[str] = []
    if not present:
        defects.append(f"missing {CLAIMS_FILE} — the honesty ledger an agent trusts "
                       "(every claim tagged shipped/simulated/stub)")
    else:
        for d in untagged[:8]:
            defects.append(d)
    soft = ([f"... and {len(untagged) - 8} more untagged claim line(s)"]
            if present and len(untagged) > 8 else [])
    return {"kpi": "honesty_ledger", "group": "adopt",
            "score": _clamp((0 if not present else 100) - 12 * len([d for d in defects if present])),
            "detail": (f"{CLAIMS_FILE} present, {len(untagged)} untagged claim(s)" if present
                       else f"no {CLAIMS_FILE}"),
            "defects": defects, "soft": soft}


def kpi_integration_recipes(missing: list[str]) -> dict[str, Any]:
    """A per-agent recipe ('point your agent here') for each family an agent
    identifies with. A missing one means that agent's operator has to invent the
    wiring — friction that sends them elsewhere."""
    defects = [f"no integration recipe for {label} — add one under docs/integrations/"
               for label in missing]
    covered = len(REQUIRED_RECIPES) - len(missing)
    return {"kpi": "integration_recipes", "group": "adopt",
            "score": _clamp(100 * covered / max(1, len(REQUIRED_RECIPES))),
            "detail": f"{covered}/{len(REQUIRED_RECIPES)} agent families have an integration recipe",
            "defects": defects, "soft": []}


def kpi_extension_scaffold(scaffold: bool, extending: bool) -> dict[str, Any]:
    """The additive-extension path: a stamper that emits a conforming skeleton
    (new_leaf.py) and the doc that teaches 'add a leaf, don't edit core'
    (EXTENDING.md). Both are how an agent contributes without breaking the ABI."""
    defects: list[str] = []
    if not scaffold:
        defects.append(f"no {LEAF_SCAFFOLD} — the leaf scaffolder an agent runs to add a feature additively")
    if not extending:
        defects.append(f"no {EXTENDING_FILE} — the doc that teaches the plug-in/prove-it path")
    return {"kpi": "extension_scaffold", "group": "build",
            "score": _clamp(100 - 50 * len(defects)),
            "detail": (f"{len(defects)} missing extension affordance(s)" if defects
                       else "leaf scaffolder + EXTENDING.md present"),
            "defects": defects, "soft": []}


def kpi_guardrails_surfaced(missing: list[str]) -> dict[str, Any]:
    """The enforced rules an agent WILL hit must be documented in AGENTS.md so the
    guard teaches, not ambushes. Each undocumented rule is one unit — an agent that
    doesn't know the trunk-only law wastes a turn fighting OFF_TRUNK."""
    defects = [f"enforced rule not surfaced in {AGENTS_FILE}: {label}" for label in missing]
    covered = len(GUARDRAIL_CLUSTERS) - len(missing)
    return {"kpi": "guardrails_surfaced", "group": "build",
            "score": _clamp(100 * covered / max(1, len(GUARDRAIL_CLUSTERS))),
            "detail": f"{covered}/{len(GUARDRAIL_CLUSTERS)} enforced rules surfaced up front",
            "defects": defects, "soft": []}


def kpi_contributor_contract(contributing: bool, linked: bool,
                             green_gate: bool) -> dict[str, Any]:
    """CONTRIBUTING.md present and linked from the entry point, plus a one-command
    green gate (make ci / make test) documented — the contract and the feedback
    loop an agent needs to ship a change confidently."""
    defects: list[str] = []
    if not contributing:
        defects.append(f"no {CONTRIBUTING_FILE} — the contributor contract")
    elif not linked:
        defects.append(f"{CONTRIBUTING_FILE} exists but is not linked from {AGENTS_FILE}/README — an agent can't find it")
    if not green_gate:
        defects.append("no one-command green gate documented (make ci / make test / ./test.ps1) — "
                       "the feedback loop an agent runs before shipping")
    return {"kpi": "contributor_contract", "group": "build",
            "score": _clamp(100 - 30 * len(defects)),
            "detail": (f"{len(defects)} missing contract/feedback affordance(s)" if defects
                       else "CONTRIBUTING linked + green gate documented"),
            "defects": defects, "soft": []}


def kpi_machine_consumable(json_tools: int, total_tools: int,
                           missing: list[str]) -> dict[str, Any]:
    """SOFT: an agent acts on structure, not prose. How much of the measurement
    family an agent drives speaks machine-readable JSON. It scores (a tree whose
    tools only print prose grades lower) but emits no hard debt — the 'right' count
    is a judgment, and a token is cheap to game."""
    soft: list[str] = []
    for m in missing[:8]:
        soft.append(f"tool without a --json surface: {m}")
    rate = (json_tools / total_tools) if total_tools else 1.0
    return {"kpi": "machine_consumable", "group": "build",
            "score": _clamp(round(100 * rate)),
            "detail": (f"{json_tools}/{total_tools} measurement tools expose --json "
                       f"({rate:.0%})" if total_tools else "no measurement tools found"),
            "defects": [], "soft": soft}


# ---------------------------------------------------------------------------
# Fold: KPIs -> composite score, grade, friction-debt, control-pane payload.
# ---------------------------------------------------------------------------

def build_payload(*, workspace: str, kpis: list[dict[str, Any]],
                  error: str | None = None) -> dict[str, Any]:
    if error:
        return {
            "schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
            "finding": "tooling_error", "reason": error,
            "next_action": "fix the read (run from repo ROOT, with git), then re-run",
            "workspace": workspace, "corpus": {}, "kpis": [],
        }
    by_name = {k["kpi"]: k for k in kpis}
    score = round(sum(KPI_WEIGHTS[n] * by_name[n]["score"]
                      for n in KPI_WEIGHTS if n in by_name), 1)
    friction_debt = sum(len(k["defects"]) for k in kpis)
    n_soft = sum(len(k["soft"]) for k in kpis)
    grade = grade_letter(score)

    debt_by_group = {g: 0 for g in GROUPS}
    for k in kpis:
        debt_by_group[k["group"]] += len(k["defects"])
    score_by_group = {g: 0.0 for g in GROUPS}
    wsum_by_group = {g: 0.0 for g in GROUPS}
    for k in kpis:
        w = KPI_WEIGHTS.get(k["kpi"], 0.0)
        score_by_group[k["group"]] += w * k["score"]
        wsum_by_group[k["group"]] += w
    group_scores = {g: (round(score_by_group[g] / wsum_by_group[g], 1)
                        if wsum_by_group[g] else 0.0) for g in GROUPS}

    breakdown = sorted(
        ({"kpi": k["kpi"], "group": k["group"], "score": k["score"],
          "debt": len(k["defects"]), "detail": k["detail"]} for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))

    corpus = {
        "score": score, "grade": grade, "friction_debt": friction_debt,
        "soft_signals": n_soft,
        "group_scores": group_scores,
        "debt_by_group": debt_by_group,
        "kpi_scores": {k["kpi"]: k["score"] for k in kpis},
        "debt_by_kpi": {k["kpi"]: len(k["defects"]) for k in kpis},
        "breakdown": breakdown,
    }

    gs = group_scores
    standing = (f"discover {gs['discover']:.0f} · adopt {gs['adopt']:.0f} "
                f"· build {gs['build']:.0f}")
    if friction_debt == 0:
        ok, verdict, finding = True, "OK", "agent_ready"
        reason = (f"agent-ready: score {score}/100 (grade {grade}), zero friction-debt "
                  f"across {len(kpis)} KPIs ({standing}; {n_soft} advisory). An agent can "
                  f"discover, adopt, and build on fak with no missing affordance")
        next_action = ("hold the line; re-run after a change to an agent surface "
                       "(AGENTS.md, llms.txt, CLAIMS.md, integration recipes, the guards)")
    else:
        ok, verdict, finding = False, "ACTION", "friction_debt"
        worst = breakdown[0]
        reason = (f"{friction_debt} unit(s) of friction-debt; score {score}/100 (grade {grade}); "
                  f"heaviest: {worst['kpi']} ({worst['debt']} defect(s)); standing {standing}")
        next_action = ("retire friction-debt worst-first (see corpus.breakdown + per-KPI defects): "
                       "fix the agents.md entry point, the doc-map, the quotable identity, dead "
                       "orientation links, the first command, the install one-liner, the tagged "
                       "ledger, the per-agent recipes, the leaf scaffold, the surfaced guardrails, "
                       "the contributor contract; re-run to prove the drop")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus, "kpis": kpis,
    }


# ---------------------------------------------------------------------------
# Disk + git gathering (the impure shell around the pure KPIs).
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _git_lines(args: list[str], root: Path) -> list[str]:
    try:
        p = subprocess.run(["git", *args], cwd=str(root), capture_output=True,
                           text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    if p.returncode != 0:
        return []
    return [ln for ln in p.stdout.splitlines() if ln.strip()]


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _local_links(text: str, doc_rel: str, root: Path) -> list[tuple[str, bool]]:
    """Every local link in a doc as (target_display, exists). Skips external /
    anchor-only links; resolves both root-absolute (/foo) and relative targets."""
    base = (root / doc_rel).parent
    out: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for m in _LINK_RE.finditer(text):
        t = m.group("target").strip()
        if t.startswith(("http://", "https://", "mailto:", "#", "tel:")):
            continue
        path_part = t.split("#", 1)[0].split("?", 1)[0].strip()
        if not path_part or path_part in seen:
            continue
        seen.add(path_part)
        resolved = (root / path_part.lstrip("/")) if path_part.startswith("/") \
            else (base / path_part)
        out.append((path_part, resolved.exists()))
    return out


def _tool_has_json(text: str) -> bool:
    """A measurement tool 'speaks JSON' if it wires a --json argparse flag."""
    return '"--json"' in text or "'--json'" in text


def gather(root: Path) -> list[dict[str, Any]]:
    """Read the git-tracked tree and run every pure KPI."""
    tracked = set(_git_lines(["ls-files"], root))

    def present(rel: str) -> bool:
        # Tracked OR on disk — a freshly-added affordance counts even pre-commit.
        return rel in tracked or (root / rel).exists()

    # Read the orientation / adoption docs once.
    read_docs = set(IDENTITY_DOCS) | set(FIRST_COMMAND_DOCS) | set(INSTALL_DOCS) \
        | set(ORIENTATION_DOCS) | {AGENTS_FILE, "README.md"}
    texts = {d: _safe_read(root / d) for d in read_docs}
    agents_text = texts.get(AGENTS_FILE, "")

    # discover
    config_present = {label: any(present(p) for p in paths) for label, paths in AGENT_CONFIGS}
    llms_present = {LLMS_FILE: present(LLMS_FILE), LLMS_FULL_FILE: present(LLMS_FULL_FILE)}
    id_found, id_where = find_identity(texts)
    dead_links: list[str] = []
    for doc in ORIENTATION_DOCS:
        if not present(doc):
            continue
        for target, ok in _local_links(_safe_read(root / doc), doc, root):
            if not ok:
                dead_links.append(f"{doc} -> {target}")

    # adopt
    fc_found, fc_where = find_first_command(texts)
    inst_found, inst_where = find_install_oneliner(texts)
    claims_present = present(CLAIMS_FILE)
    claims_text = _safe_read(root / CLAIMS_FILE) if claims_present else None
    untagged = untagged_claims(claims_text)
    recipe_present = {label: any(present(p) for p in paths) for label, paths in REQUIRED_RECIPES}

    # build
    scaffold = present(LEAF_SCAFFOLD)
    extending = present(EXTENDING_FILE)
    contributing = present(CONTRIBUTING_FILE)
    contributing_linked = _has(agents_text, CONTRIBUTING_FILE) or _has(texts.get("README.md"), CONTRIBUTING_FILE)
    green_gate = _has(agents_text, *GREEN_GATE_TOKENS)
    guard_missing = missing_guardrails(agents_text)
    # machine_consumable: the measurement family an agent drives.
    tool_files = sorted(f for f in tracked if re.search(r"tools/[^/]*scorecard[^/]*\.py$", f)
                        and not f.endswith("_test.py"))
    json_tools = 0
    missing_json: list[str] = []
    for f in tool_files:
        if _tool_has_json(_safe_read(root / f)):
            json_tools += 1
        else:
            missing_json.append(f)

    return [
        kpi_agents_entrypoint(agents_text if present(AGENTS_FILE) else None),
        kpi_agent_config(missing_agent_configs(config_present)),
        kpi_llms_map(llms_present),
        kpi_identity_statement(id_found, id_where),
        kpi_entry_links_resolve(dead_links),
        kpi_first_command(fc_found, fc_where),
        kpi_install_oneliner(inst_found, inst_where),
        kpi_honesty_ledger(claims_present, untagged),
        kpi_integration_recipes(missing_recipes(recipe_present)),
        kpi_extension_scaffold(scaffold, extending),
        kpi_guardrails_surfaced(guard_missing),
        kpi_contributor_contract(contributing, contributing_linked, green_gate),
        kpi_machine_consumable(json_tools, len(tool_files), missing_json),
    ]


def collect(workspace: Path) -> dict[str, Any]:
    root = workspace.resolve()
    if not (root / ".git").exists() and not _git_lines(["rev-parse", "--git-dir"], root):
        return build_payload(workspace=str(root), kpis=[],
                             error=f"not a git repo at {root} — run from the repo ROOT")
    return build_payload(workspace=str(root), kpis=gather(root))


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    gs = c.get("group_scores") or {}
    lines = [
        f"agent-readiness-scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) "
         f"· FRICTION-DEBT {c.get('friction_debt', 0)} · {c.get('soft_signals', 0)} advisory"),
        (f"agent journey:  discover {gs.get('discover', 0):.0f}  ·  "
         f"adopt {gs.get('adopt', 0):.0f}  ·  build {gs.get('build', 0):.0f}"),
        ("debt by step: " + "  ".join(
            f"{g}:{(c.get('debt_by_group') or {}).get(g, 0)}" for g in GROUPS)),
        "",
        "per-KPI (worst first):",
        f"  {'score':>5} {'debt':>4}  {'step':<10} {'kpi':<22} detail",
    ]
    for b in c.get("breakdown", []):
        lines.append(f"  {b['score']:>5} {b['debt']:>4}  {b['group']:<10} "
                     f"{b['kpi']:<22} {b['detail']}")
    lines.append("")
    lines.append("friction-debt work-list:")
    any_defect = False
    for k in sorted(payload.get("kpis", []), key=lambda x: -len(x["defects"])):
        if not k["defects"]:
            continue
        any_defect = True
        lines.append(f"  {k['kpi']} ({len(k['defects'])}):")
        for it in k["defects"][:12]:
            lines.append(f"      - {it}")
        if len(k["defects"]) > 12:
            lines.append(f"      ... and {len(k['defects']) - 12} more")
    if not any_defect:
        lines.append("  (none — zero friction-debt)")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    gs = c.get("group_scores") or {}
    out: list[str] = []
    out.append("---")
    out.append('title: "fak agent-readiness scorecard — the friction-debt measuring stick"')
    out.append('description: "fak\'s deterministic agent-readiness scorecard: thirteen KPIs across '
               'the three steps an AI agent walks — discover, adopt, build — folded into a '
               'composite score and the headline friction-debt metric, re-derived from the '
               'git-tracked tree."')
    out.append("---")
    out.append("")
    out.append("# Agent-readiness scorecard — can an agent discover, adopt, and build on fak")
    out.append("")
    if stamp:
        out.append(f"<!-- agent-readiness-scorecard: {stamp} · process: tools/agent_readiness_scorecard.py -->")
        out.append("")
    out.append("This is the measuring stick for fak's **agent attractiveness** — the question an "
               "agent-first project lives or dies on: can an autonomous coding agent (Claude Code, "
               "OpenAI Codex, Cursor, an MCP client) **discover** fak, **want** to adopt it, and "
               "**build** on it effectively? Every number below is re-derived from the git-tracked "
               "tree by `tools/agent_readiness_scorecard.py` — no hand-entry. The headline metric "
               "is **friction-debt**: the count of concrete, mechanical defects that make fak harder "
               "for an agent to find, trust, and build on — a missing entry point, a dead "
               "orientation link, no copy-pasteable first command, an un-tagged claim, a guard "
               "that ambushes instead of teaches. Driving friction-debt to zero is what makes fak "
               "the path of least resistance for the agent that lands in it cold.")
    out.append("")
    out.append("> Regenerate: `python tools/agent_readiness_scorecard.py --markdown --stamp DATE > docs/AGENT-READINESS-SCORECARD.md`")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| **Friction-debt (total HARD defects)** | **{c.get('friction_debt', 0)}** |")
    out.append(f"| Composite score | {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) |")
    out.append(f"| Agent journey | discover {gs.get('discover', 0):.0f} · adopt {gs.get('adopt', 0):.0f} "
               f"· build {gs.get('build', 0):.0f} |")
    out.append(f"| Advisory (soft) signals | {c.get('soft_signals', 0)} |")
    g = c.get("debt_by_group", {})
    out.append(f"| Debt by step | discover:{g.get('discover',0)} · adopt:{g.get('adopt',0)} "
               f"· build:{g.get('build',0)} |")
    out.append("")
    out.append("## The three steps an agent walks")
    out.append("")
    out.append("Thirteen KPIs, each 0–100, grouped by the step they gate. `debt` = units of HARD "
               "friction-debt. `machine_consumable` is advisory (it scores but emits no hard debt — "
               "a token is cheap to game).")
    out.append("")
    out.append("| Step | KPI | Score | Debt | Detail |")
    out.append("|---|---|---:|:--:|---|")
    for b in c.get("breakdown", []):
        out.append(f"| {b['group']} | `{b['kpi']}` | {b['score']} | {b['debt']} | {b['detail']} |")
    out.append("")
    out.append("## Friction-debt work-list")
    out.append("")
    any_defect = False
    for k in sorted(payload.get("kpis", []), key=lambda x: -len(x["defects"])):
        if not k["defects"]:
            continue
        any_defect = True
        out.append(f"### `{k['kpi']}` ({k['group']}) — {len(k['defects'])} defect(s), score {k['score']}")
        for it in k["defects"]:
            out.append(f"- {it}")
        out.append("")
    if not any_defect:
        out.append("No friction-debt: an agent can discover, adopt, and build on fak with no "
                   "missing affordance. 🎉")
        out.append("")
    return "\n".join(out)


def render_compare(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    b = baseline.get("corpus") or {}
    cur = current.get("corpus") or {}
    bd, cd = b.get("friction_debt", 0), cur.get("friction_debt", 0)
    bo, co = b.get("score", 0), cur.get("score", 0)
    ratio = "∞ (zero)" if cd == 0 else f"{bd / cd:.1f}×"
    lines = [
        f"friction-debt: {bd} -> {cd}   ({ratio} fewer defects)",
        f"score:         {bo}/100 -> {co}/100   (+{round(co - bo, 1)})",
    ]
    for gp in GROUPS:
        gb = (b.get("debt_by_group") or {}).get(gp, 0)
        gc = (cur.get("debt_by_group") or {}).get(gp, 0)
        lines.append(f"  {gp:<10} {gb} -> {gc}")
    target = max(0, bd // 2)
    if cd <= target:
        lines.append(f"VERDICT: >=2x friction-debt reduction achieved ({bd} -> {cd}).")
    else:
        lines.append(f"VERDICT: not yet 2x — need friction-debt <= {target} (now {cd}).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Agent-readiness scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true", help="emit the snapshot markdown body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    ap.add_argument("--compare", default="", metavar="BASELINE.json",
                    help="print the friction-debt delta vs a prior baseline JSON")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace)

    if args.compare:
        try:
            baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"error: cannot read baseline {args.compare}: {exc}", file=sys.stderr)
            return 2
        print(render_compare(baseline, payload))
    elif args.json:
        print(json.dumps(payload, indent=2))
    elif args.markdown:
        print(render_markdown(payload, stamp=args.stamp or None))
    else:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
