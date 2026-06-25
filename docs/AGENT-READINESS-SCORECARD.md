---
title: "fak agent-readiness scorecard — the friction-debt measuring stick"
description: "fak's deterministic agent-readiness scorecard: KPIs across the three steps an AI agent walks — discover, adopt, build — folded into a composite score and the headline friction-debt metric, re-derived from the git-tracked tree. Presence KPIs ask does-the-affordance-exist; the paste-and-run and Codex-currentness KPIs ask whether the docs work for current agents."
---

# Agent-readiness scorecard — can an agent discover, adopt, and build on fak

<!-- agent-readiness-scorecard: 2026-06-25 · process: tools/agent_readiness_scorecard.py -->

This is the measuring stick for fak's **agent attractiveness** — the question an agent-first project lives or dies on: can an autonomous coding agent (Claude Code, OpenAI Codex, Cursor, an MCP client) **discover** fak, **want** to adopt it, and **build** on it effectively? Every number below is re-derived from the git-tracked tree by `tools/agent_readiness_scorecard.py` — no hand-entry. The headline metric is **friction-debt**: the count of concrete, mechanical defects that make fak harder for an agent to find, trust, and build on — a missing entry point, a dead orientation link, no copy-pasteable first command, an un-tagged claim, a guard that ambushes instead of teaches. Driving friction-debt to zero is what makes fak the path of least resistance for the agent that lands in it cold.

> Regenerate: `python tools/agent_readiness_scorecard.py --markdown --stamp DATE > docs/AGENT-READINESS-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Friction-debt (total HARD defects)** | **0** |
| Composite score | 100.0/100 (grade A) |
| Agent journey | discover 100 · adopt 100 · build 100 |
| Advisory (soft) signals | 0 |
| Debt by step | discover:0 · adopt:0 · build:0 |

## The three steps an agent walks

17 KPIs, each 0–100, grouped by the step they gate. `debt` = units of HARD friction-debt. Five presence KPIs per step ask does-the-affordance-exist; the paste-and-run success KPIs (`fenced_paths_resolve`, `first_command_runs`, `platform_guidance_consistent`) ask does-an-agent-who-pastes-the-docs-actually-succeed; `codex_recipe_current` asks whether the Codex guide still matches the current Codex MCP / AGENTS.md / exec JSON / Responses-vs-Chat-Completions shape. `machine_consumable` is advisory (it scores but emits no hard debt — a token is cheap to game).

| Step | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| discover | `agents_entrypoint` | 100 | 0 | AGENTS.md states identity + build/test/run |
| discover | `agent_config` | 100 | 0 | 3/3 agent harnesses have a zero-setup config |
| discover | `llms_map` | 100 | 0 | llms.txt present |
| discover | `identity_statement` | 100 | 0 | identity statement found in AGENTS.md |
| discover | `entry_links_resolve` | 100 | 0 | every orientation link resolves |
| adopt | `first_command` | 100 | 0 | first command present in AGENTS.md |
| adopt | `first_command_runs` | 100 | 0 | the first command runs cold (policy examples/customer-support-readonly-policy.json resolves, no key) |
| adopt | `install_oneliner` | 100 | 0 | install one-liner present in AGENTS.md |
| adopt | `honesty_ledger` | 100 | 0 | CLAIMS.md present, 0 untagged claim(s) |
| adopt | `integration_recipes` | 100 | 0 | 4/4 agent families have an integration recipe |
| adopt | `codex_recipe_current` | 100 | 0 | Codex recipe covers MCP, AGENTS.md, exec JSON, proxy URL, and Responses fence |
| adopt | `fenced_paths_resolve` | 100 | 0 | every fenced command path resolves from a clean clone |
| build | `extension_scaffold` | 100 | 0 | leaf scaffolder + EXTENDING.md present |
| build | `guardrails_surfaced` | 100 | 0 | 6/6 enforced rules surfaced up front |
| build | `contributor_contract` | 100 | 0 | CONTRIBUTING linked + green gate documented |
| build | `platform_guidance_consistent` | 100 | 0 | the green gate names its native-Windows bridge |
| build | `machine_consumable` | 100 | 0 | 18/18 measurement tools expose --json (100%) |

## Friction-debt work-list

No friction-debt: an agent can discover, adopt, and build on fak with no missing affordance. 🎉

