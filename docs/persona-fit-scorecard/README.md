---
title: "fak persona-fit scorecard — across each feature space, how much would each persona like fak?"
description: "Inward persona×feature-space scorecard: a fit matrix of fak's top-10 personas (free-tier dev → infra engineer → researcher → decision-maker) against its key feature spaces (security, performance, memory, model, tooling, platform). Each cell is a tree-grounded fit score — the weighted match of what a persona values and what a feature delivers. Two driven numbers: coverage of the grid and persona-fit-debt (the integrity of the matrix)."
---

# Persona-fit scorecard — how much would each persona like each feature space?

<!-- persona-fit-scorecard: 513f3319 · process: tools/persona_fit_scorecard.py · data: tools/persona_fit_scorecard.data/ -->

The sibling [`persona-readiness`](../persona-scorecard/README.md) scorecard asks whether each top persona's *entry path* is served. This one asks the matrix question a go-to-market person draws: **across the key feature spaces fak ships, how much would each persona LIKE each one?** How an *engineer* feels about the model internals, how a *decision-maker* (≈ a product manager) feels about the benchmarks, how a *researcher* feels about reproducibility. Every cell is computed, not typed: a persona's WEIGHTS over the value dimensions × a feature's tree-grounded DELIVERY. To raise a cell you make the feature actually deliver what that persona values — never by editing the score.

> Regenerate: `python tools/persona_fit_scorecard.py --markdown-dir docs/persona-fit-scorecard`.

## Headline

| Metric | Value |
|---|---|
| **Coverage** | **100.0%** (10/10 personas × 6/6 feature spaces) |
| **Persona-fit-debt** | **0** (grounding/honesty 0 + coverage 0) |
| Composite score | 100.0/100 (grade A) |
| Value dimensions | 9 |
| As of | 2026-06-26 (fak v0.34.0) |

> **Read this right.** A LOW cell is not a defect — it is honest (a free-tier dev simply does not value the model internals the way a researcher does). The score grades the *integrity of the matrix*: every grounding check relevant to its dimension AND resolving in the tree, every declared favourite matching the computed matrix, the whole grid positioned. An *off-topic* or *ungrounded* delivery claim, or an *overclaimed* favourite, is the defect this catches. (A sub-100 cell can mean a dimension is genuinely under-served for that persona OR simply under-declared — delivery depth caps at 3 grounded checks per dimension, so a 33% delivery means one witness was named, not that two failed.)

## The fit matrix

Each cell is the 0-100 fit of a feature space for a persona (the weighted match of what the persona values and what the feature delivers).

| Persona | platform | model | tooling | performan… | security | memory | mean | loves |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **free-tier-dev** | 75 | 67 | 69 | 64 | 58 | 42 | 62.5 | platform |
| **ml-researcher** | 69 | 64 | 58 | 86 | 39 | 47 | 60.5 | performance |
| **app-developer** | 76 | 58 | 61 | 30 | 64 | 42 | 55.2 | platform |
| **security-engineer** | 52 | 45 | 64 | 27 | 76 | 64 | 54.7 | security |
| **ai-agent** | 80 | 60 | 53 | 30 | 60 | 43 | 54.3 | platform |
| **oss-contributor** | 81 | 72 | 42 | 36 | 39 | 39 | 51.5 | platform |
| **benchmark-engineer** | 73 | 48 | 42 | 91 | 24 | 27 | 50.8 | performance |
| **decision-maker** | 59 | 46 | 41 | 74 | 28 | 38 | 47.7 | performance |
| **infra-engineer** | 57 | 33 | 38 | 40 | 33 | 26 | 37.8 | platform |
| **backend-integrator** | 69 | 53 | 25 | 11 | 42 | 25 | 37.5 | platform |
| _feature mean_ | 69.1 | 54.6 | 49.3 | 48.9 | 46.3 | 39.3 | | |

## Standing at a glance

> Regenerate this chart with `python tools/persona_fit_scorecard.py --chart`.

```text
persona-fit chart — 10 personas × 6 feature spaces · score 100.0/100 (grade A) · persona-fit-debt 0

fit heat-grid (each cell = one persona×feature; denser = better fit):
  pla… mod… too… per… sec… mem…   persona
     █    ▓    ▓    ▓    ▓    ▒   free-tier-dev
     ▓    ▓    ▓    █    ▒    ▒   ml-researcher
     █    ▓    ▓    ▒    ▓    ▒   app-developer
     ▓    ▒    ▓    ░    █    ▓   security-engineer
     █    ▓    ▓    ▒    ▓    ▒   ai-agent
     █    █    ▒    ▒    ▒    ▒   oss-contributor
     █    ▒    ▒    █    ░    ░   benchmark-engineer
     ▓    ▒    ▒    █    ░    ▒   decision-maker
     ▓    ▒    ▒    ▒    ▒    ░   infra-engineer
     ▓    ▓    ░    ·    ▒    ░   backend-integrator

feature-space appeal (mean fit across all personas):
  platform       ██████████████████████████ 69.1  wins oss-contributor
  model          ████████████████████······ 54.6  wins oss-contributor
  tooling        ██████████████████········ 49.3  wins free-tier-dev
  performance    ██████████████████········ 48.9  wins benchmark-engineer
  security       █████████████████········· 46.3  wins security-engineer
  memory         ███████████████··········· 39.3  wins security-engineer

persona satisfaction (mean fit across all feature spaces):
  free-tier-dev      ██████████████████████████ 62.5  loves platform
  ml-researcher      █████████████████████████· 60.5  loves performance
  app-developer      ███████████████████████··· 55.2  loves platform
  security-engineer  ███████████████████████··· 54.7  loves security
  ai-agent           ███████████████████████··· 54.3  loves platform
  oss-contributor    █████████████████████····· 51.5  loves platform
  benchmark-engineer █████████████████████····· 50.8  loves performance
  decision-maker     ████████████████████······ 47.7  loves performance
  infra-engineer     ████████████████·········· 37.8  loves platform
  backend-integrator ████████████████·········· 37.5  loves platform

legend: █ ≥70   ▓ ≥50   ▒ ≥30   ░ ≥15   · <15  (fit 0-100)
```

## The value dimensions

Both a persona's weights and a feature's delivery are vectors over this fixed set.

| Dimension | A lander asks |
|---|---|
| `runs_today` | can I run it now, ideally offline on a laptop? |
| `proven` | is it backed by a witness / results / test that exists? |
| `measured` | does it expose metrics / observability I can watch? |
| `operable` | is there a deploy / config / ops surface to run it? |
| `trustworthy` | does it have a security / refusal / determinism floor? |
| `extensible` | can I build on it (frozen ABI, seams, a leaf)? |
| `documented` | is there an entry doc where a person learns it? |
| `benchmarked` | is it measured against the field / SOTA? |
| `efficient` | is it cheap to run — does it save tokens / compute / cache? |

## Feature spaces — delivery + who they win

| Feature space | Group | Mean appeal | Weighted appeal | Wins persona |
|---|---|---:|---:|---|
| **Platform / adoption** (`platform`) | adoption | 69.1 | 70.4 | oss-contributor |
| **In-kernel model** (`model`) | throughput | 54.6 | 55.2 | oss-contributor |
| **Agent tooling** (`tooling`) | trust | 49.3 | 50.7 | free-tier-dev |
| **Performance / reuse** (`performance`) | throughput | 48.9 | 45.4 | benchmark-engineer |
| **Security floor** (`security`) | trust | 46.3 | 48.9 | security-engineer |
| **Agent memory** (`memory`) | trust | 39.3 | 38.7 | security-engineer |

> _Weighted appeal_ weights each persona by its market `importance` (a mass-market free-tier dev counts more than a niche role), so a broad win outranks a flat-mean tie.

## Personas — who the feature set serves best

| Persona | Mean fit | Loves most |
|---|---:|---|
| **Free-tier solo dev** (`free-tier-dev`) | 62.5 | platform (75) |
| **ML researcher** (`ml-researcher`) | 60.5 | performance (86) |
| **App developer / vibe-coder** (`app-developer`) | 55.2 | platform (76) |
| **Security engineer** (`security-engineer`) | 54.7 | security (76) |
| **AI coding agent** (`ai-agent`) | 54.3 | platform (80) |
| **Open-source contributor** (`oss-contributor`) | 51.5 | platform (81) |
| **Benchmark / eval engineer** (`benchmark-engineer`) | 50.8 | performance (91) |
| **Evaluator / decision-maker (PM)** (`decision-maker`) | 47.7 | performance (74) |
| **Infra / platform engineer (SRE)** (`infra-engineer`) | 37.8 | platform (57) |
| **Backend integrator** (`backend-integrator`) | 37.5 | platform (69) |

## Per-KPI (persona-fit-debt = grounding/honesty of the rows that exist)

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| well-formed | `rows_well_formed` | 100 | 0 | all 16 persona+feature rows well-formed |
| reality | `dimension_relevant` | 100 | 0 | 75/75 grounding checks are relevant to their dimension (100%) |
| reality | `evidence_grounded` | 100 | 0 | 75/75 relevant grounding checks resolve in the tree (100%) |
| honesty | `fit_honest` | 100 | 0 | every declared favourite matches the computed matrix |

