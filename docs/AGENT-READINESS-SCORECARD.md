---
title: "fak agent-readiness scorecard — the experience-frontier + friction-debt measuring stick"
description: "fak's deterministic agent-readiness scorecard: KPIs across the three steps an AI agent walks — discover, adopt, build — folded into an unbounded experience-frontier (higher = better, the surface to grow) and a baseline friction-debt gate, re-derived from the git-tracked tree. Presence KPIs ask does-the-affordance-exist; the paste-and-run and Codex-currentness KPIs ask whether the docs work for current agents."
---

# Agent-readiness scorecard — can an agent discover, adopt, and build on fak

<!-- agent-readiness-scorecard: 2026-06-30 · process: tools/agent_readiness_scorecard.py -->

This is the measuring stick for fak's **agent attractiveness** — the question an agent-first project lives or dies on: can an autonomous coding agent (Claude Code, OpenAI Codex, Cursor, an MCP client) **discover** fak, **want** to adopt it, and **build** on it effectively? Every number below is re-derived from the git-tracked tree by `tools/agent_readiness_scorecard.py` — no hand-entry. There are two headline numbers. **Experience-frontier** (unbounded, higher = better) is the one to grow: the weighted count of real, working agent affordances the tree provides — a never-done program with no ceiling, the mirror of the operator-heaviness gauge. **Friction-debt** (lower = better, floor 0) is the baseline gate: the count of concrete, mechanical defects that make fak harder for an agent to find, trust, and build on — a missing entry point, a dead orientation link, no copy-pasteable first command, an un-tagged claim, a guard that ambushes instead of teaches. Driving friction-debt to zero makes fak the path of least resistance for the agent that lands in it cold; climbing the frontier keeps widening the set of agents it serves.

> Regenerate: `python tools/agent_readiness_scorecard.py --markdown --stamp DATE > docs/AGENT-READINESS-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Experience-frontier (unbounded · higher = better)** | **292** |
| Frontier by affordance (weight×count) | integration_recipes 160 · harness_configs 30 · refusal_recoveries 48 · machine_consumable 54 |
| Friction-debt (total HARD defects) | 0 |
| Baseline coverage score | 100.0/100 (grade A) |
| Agent journey | discover 100 · adopt 100 · build 100 |
| Advisory (soft) signals | 0 |
| Debt by step | discover:0 · adopt:0 · build:0 |

### The two questions (why two headline numbers)

**Friction-debt** (lower = better, floor 0) is the BASELINE gate — are the expected affordances present and working? It saturates: drive it to zero and there is nothing left to fix. **Experience-frontier** (higher = better, *unbounded*) is the never-done program — the weighted count of real, working agent affordances the tree actually provides (an integration recipe an agent follows, a zero-setup harness config, a kernel refusal an agent can recover from, a tool an agent drives via `--json`). It has **no ceiling**: there is always one more agent harness to make fak first-class for, one more refusal to make recoverable — so a "100% / done" line would be a category error. It is the deliberate mirror of `internal/heavinessscore`'s unbounded `heaviness_pressure` (the load an operator carries); this is the surface an agent gains. You climb it by ADDING a real affordance, never by gaming a substring. Per-unit weights: `integration_recipes`×8, `harness_configs`×10, `refusal_recoveries`×3, `machine_consumable`×2.

## The three steps an agent walks

23 KPIs, each 0–100, grouped by the step they gate. `debt` = units of HARD friction-debt. The presence KPIs ask does-the-affordance-exist; the paste-and-run / executable-truth KPIs ask the question presence can't reach — does an agent who pastes the docs actually succeed: `fenced_paths_resolve` (the path resolves), `command_verbs_resolve` (the `fak <verb>` is a real dispatched verb, parsed live from cmd/fak/main.go), `first_command_runs` (the proof runs cold), `recipe_links_resolve` (the link inside the recipe is alive), `agent_config_valid` (the auto-loaded config parses), `platform_guidance_consistent` (the gate names its Windows bridge). `codex_recipe_current` asks whether the Codex guide still matches the current Codex MCP / AGENTS.md / exec JSON / Responses-vs-Chat-Completions shape. `machine_consumable` is advisory (it scores but emits no hard debt — a token is cheap to game).

| Step | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| discover | `agents_entrypoint` | 100 | 0 | AGENTS.md states identity + build/test/run |
| discover | `agent_config` | 100 | 0 | 3/3 agent harnesses have a zero-setup config |
| discover | `llms_map` | 100 | 0 | llms.txt present |
| discover | `identity_statement` | 100 | 0 | identity resolves near the top of all 3 orientation docs (AGENTS.md, llms.txt, README.md) |
| discover | `entry_links_resolve` | 100 | 0 | every orientation link resolves |
| adopt | `first_command` | 100 | 0 | first command present in AGENTS.md |
| adopt | `first_command_runs` | 100 | 0 | the first command runs cold (policy examples/customer-support-readonly-policy.json resolves, no key) |
| adopt | `install_oneliner` | 100 | 0 | install one-liner present in AGENTS.md |
| adopt | `honesty_ledger` | 100 | 0 | CLAIMS.md present, 0 untagged claim(s) |
| adopt | `integration_recipes` | 100 | 0 | 4/4 agent families have an integration recipe |
| adopt | `codex_recipe_current` | 100 | 0 | Codex recipe covers MCP, AGENTS.md, exec JSON, proxy URL, and Responses fence |
| adopt | `fenced_paths_resolve` | 100 | 0 | every fenced command path resolves from a clean clone |
| adopt | `command_verbs_resolve` | 100 | 0 | every pasted `fak <verb>` resolves to a real dispatched verb |
| discover | `recipe_links_resolve` | 100 | 0 | every link inside every integration recipe resolves |
| discover | `agent_config_valid` | 100 | 0 | .mcp.json parses and every server names a launch command |
| build | `extension_scaffold` | 100 | 0 | leaf scaffolder + EXTENDING.md present |
| build | `guardrails_surfaced` | 100 | 0 | 6/6 enforced rules surfaced up front |
| build | `contributor_contract` | 100 | 0 | CONTRIBUTING linked + green gate documented |
| build | `platform_guidance_consistent` | 100 | 0 | the green gate names its native-Windows bridge |
| build | `machine_consumable` | 100 | 0 | 27/27 measurement tools expose --json (100%) |
| build | `refusal_recovery_mapped` | 100 | 0 | 16/16 kernel refusal tokens have an agent-facing recovery |
| adopt | `quickstart_success_signal` | 100 | 0 | the proof block shows an observable success signal |
| adopt | `toolchain_pinned` | 100 | 0 | go.mod pins the Go version and an entry doc names it |

## Friction-debt work-list

No friction-debt: an agent can discover, adopt, and build on fak with no missing affordance. 🎉

