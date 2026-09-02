---
title: "fak popularization-readiness scorecard — does the front door convert?"
description: "Inward popularization scorecard (concept-popularization epic, dimension K): fak's front door modeled as a conversion funnel (land → orient → trust → install → act), each stage positioned on whether the affordances a first-time visitor reaches for actually exist in the tree, with one honest conversion verdict per stage. Two driven numbers: coverage (of the funnel) and popularization-debt."
---

# Popularization-readiness scorecard — does the front door convert?

The sibling scorecards grade fak through one lens each: [`agent-readiness`](../AGENT-READINESS-SCORECARD.md) asks whether an AI *agent* can adopt fak, [`persona-readiness`](../persona-scorecard/README.md) whether each of the top-10 humans who land on fak is served. This one asks the **popularization** question the concept-popularization epic exists to answer (dimension K, adoption measurement): when a first-time visitor lands on fak's front door — the README, the five-concepts framing, the install one-liner, the social-proof surfaces, the first runnable command — **does that door actually convert them, stage by stage, or does it leak them?** The door is modeled as an ordered conversion funnel (land → orient → trust → install → act); every number below is re-derived from `tools/popularization_readiness_scorecard.data/` by `tools/popularization_readiness_scorecard.py` and cross-checked against the real tree, so no verdict is hand-typed: to stop a stage leaking you ADD the real affordance.

> Regenerate: `python tools/popularization_readiness_scorecard.py --markdown-dir docs/popularization-scorecard`.

## Headline

| Metric | Value |
|---|---|
| **Coverage** | **100.0%** (5/5 funnel stages positioned) |
| **Popularization-debt** | **0** (affordance/honesty 0 + coverage 0) |
| Composite score | 100.0/100 (grade A) |
| Stages converting | 5 of 5 |
| Hard affordances present | 25 / 25 |
| As of | 2026-09-01 (fak v0.45.0) |

> **Read this right.** The score grades how *complete and honest the funnel map is* — every declared stage positioned, every conversion verdict matching the affordances actually on disk. A missing affordance is a real leak you ADD; an *overclaimed* verdict is the defect this catches.

## The funnel at a glance

> Regenerate this chart in the terminal with `python tools/popularization_readiness_scorecard.py --chart`.

```text
popularization-readiness chart — 5 funnel stages · score 100.0/100 (grade A) · popularization-debt 0

verdict ladder (count of stages, best -> worst):
  ★ converts            ████████████████████████████ 5
  ● mostly-converts     ···························· 0
  ◐ partially-converts  ···························· 0
  ○ leaks               ···························· 0

the funnel (in order — where a visitor lands, and where they leak):
  ★ act       ████████████████████ 5/5  Act — the first win
  ★ install   ████████████████████ 4/4  Install — get the binary
  ★ land      ████████████████████ 4/4  Land — first glance at the README
  ★ orient    ████████████████████ 5/5  Orient — the five concepts, named and reachable
  ★ trust     ████████████████████ 7/7  Trust — the proof a skeptic needs

coverage  [████████████████████████████████] 100.0%  (5/5 funnel stages positioned)

legend: ★ converts   ● mostly-converts   ◐ partially-converts   ○ leaks
```

## The conversion ladder

| Verdict | Means |
|---|---|
| ★ converts | every hard affordance this stage needs is present — the visitor advances |
| ● mostly-converts | ≥ 75% of the hard affordances present — a small leak |
| ◐ partially-converts | ≥ 40% present — the stage is half-built |
| ○ leaks | < 40% present — visitors drop here |

## The funnel stages (best-converting first)

| | Verdict | Phase | Effort | Affordances | Stage — the job that must happen to advance |
|---|---|---|---|---|---|
| ★ | converts | act | glance | 5/5 | **Act — the first win** — Run one offline command and see a labelled result (a cost/quality delta, a DENY verdict), backed by a quickstart from clone and a demo doc that shows what else to run. |
| ★ | converts | install | skim | 4/4 | **Install — get the binary** — Get fak with one command that resolves — a go install one-liner (module at the repo root) or a prebuilt archive — with a fuller install doc for the other platforms. |
| ★ | converts | land | glance | 4/4 | **Land — first glance at the README** — In one glance, understand what fak is and see that it does something real — an identity line, a headline number, a runnable first command visible above the fold. |
| ★ | converts | orient | skim | 5/5 | **Orient — the five concepts, named and reachable** — Find the five core concepts named in plain language, each reachable from a front-door surface — so the visitor can re-explain fak to a colleague a week later. |
| ★ | converts | trust | read | 7/7 | **Trust — the proof a skeptic needs** — See that the claims are backed (a benchmark authority, an honesty ledger where every claim carries a status tag), understand how fak differs from what they already run, and confirm it is real open source. |

## Per-KPI (popularization-debt = affordance/honesty of the stages that exist)

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| well-formed | `stages_well_formed` | 100 | 0 | all 5 funnel stages well-formed |
| reality | `affordances_present` | 100 | 0 | 25/25 hard affordances present (100%) |
| honesty | `verdict_honest` | 100 | 0 | every verdict matches its affordance evidence |

