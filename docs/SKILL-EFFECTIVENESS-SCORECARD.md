---
title: "fak skill-effectiveness scorecard — is each skill built to be effective"
description: "fak's deterministic skill-effectiveness scorecard: nine mechanical KPIs across discover, operate, trust, and economy, folded into a composite score and the headline skill-debt metric, re-derived from the git-tracked .claude/skills tree. It grades whether each Claude Code skill carries the affordances that make it discoverable, safe to operate, and trustworthy — a sharp trigger, references that resolve, the commit-by-path discipline."
---

# Skill-effectiveness scorecard — is each skill built to be effective

<!-- skill-effectiveness-scorecard: 2026-06-26 · process: tools/skill_effectiveness_scorecard.py -->

This is the measuring stick for the **skill pack itself**. The other scorecards grade product surfaces; this one asks whether each `.claude/skills/<name>/SKILL.md` is *built to be effective* — when the model reaches for it, can it **discover** the skill (a sharp "Use when …" trigger), **operate** it safely (a scoped tool surface, references that resolve on disk), and **trust** what it ships (a witness step, the commit-by-path discipline this shared trunk demands)? Every number below is re-derived from the git-tracked skill tree by `tools/skill_effectiveness_scorecard.py` — no hand-entry. The headline metric is **skill-debt**: the count of concrete affordances a skill is missing. Driving it to zero makes every skill in the pack pull its weight.

> Regenerate: `python tools/skill_effectiveness_scorecard.py --markdown --stamp DATE > docs/SKILL-EFFECTIVENESS-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Skill-debt (total HARD defects)** | **14** |
| Composite score | 91.5/100 (grade A) |
| Skills graded | 33 |
| By group | discover 99 · operate 75 · trust 100 · economy 93 |
| Advisory (soft) signals | 3 |

## The four things that make a skill effective

9 KPIs, each 0–100, grouped by the job they gate. `debt` = units of HARD skill-debt. `refs_resolve` is the ungameable anchor — a cited helper either exists on disk or it doesn't. The commit-discipline cluster (`commit_discipline`, `proof_step`, `tools_scoped`) gates only the skills that commit to the trunk. The two ECONOMY KPIs are SOFT — they lower the score but emit no debt, because the cheap way to fix either is a keyword, which is gaming.

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| operate | `tools_scoped` | 44 | 10 | 8/18 skill(s) pass |
| operate | `refs_resolve` | 91 | 3 | 30/33 skill(s) pass |
| discover | `trigger_clause` | 97 | 1 | 32/33 skill(s) pass |
| economy | `anti_gaming` | 86 | 0 | 19/22 skill(s) pass |
| discover | `description_present` | 100 | 0 | 33/33 skill(s) pass |
| discover | `name_resolves` | 100 | 0 | 33/33 skill(s) pass |
| trust | `commit_discipline` | 100 | 0 | 18/18 skill(s) pass |
| trust | `proof_step` | 100 | 0 | 18/18 skill(s) pass |
| economy | `context_budget` | 100 | 0 | 33/33 skill(s) pass |

## Skill-debt work-list

### `tools_scoped` (operate) — 10 defect(s), score 44
- agent-readiness: commits to the trunk but declares no `allowed-tools` — a high-privilege skill should state its tool surface (least privilege)
- appeal-score: commits to the trunk but declares no `allowed-tools` — a high-privilege skill should state its tool surface (least privilege)
- curate-cluster: commits to the trunk but declares no `allowed-tools` — a high-privilege skill should state its tool surface (least privilege)
- dos-dispatch-loop: commits to the trunk but declares no `allowed-tools` — a high-privilege skill should state its tool surface (least privilege)
- industry-score: commits to the trunk but declares no `allowed-tools` — a high-privilege skill should state its tool surface (least privilege)
- persona-score: commits to the trunk but declares no `allowed-tools` — a high-privilege skill should state its tool surface (least privilege)
- refresh-readme: commits to the trunk but declares no `allowed-tools` — a high-privilege skill should state its tool surface (least privilege)
- scorecard: commits to the trunk but declares no `allowed-tools` — a high-privilege skill should state its tool surface (least privilege)
- stability-score: commits to the trunk but declares no `allowed-tools` — a high-privilege skill should state its tool surface (least privilege)
- token-defaults-score: commits to the trunk but declares no `allowed-tools` — a high-privilege skill should state its tool surface (least privilege)

### `refs_resolve` (operate) — 3 defect(s), score 91
- dos-plan-price: cites `examples/plan_price/plan_price.py` which does not exist on disk — a dead reference (deleted/renamed/typo)
- token-defaults-score: cites `tools/token_defaults_scorecard.py` which does not exist on disk — a dead reference (deleted/renamed/typo)
- token-defaults-score: cites `tools/token_defaults_scorecard_test.py` which does not exist on disk — a dead reference (deleted/renamed/typo)

### `trigger_clause` (discover) — 1 defect(s), score 97
- phased-plan: description states WHAT it does but not WHEN — add a "Use when / Use to / Use after …" trigger clause

