---
title: "fak persona-readiness scorecard — is each top persona served?"
description: "Inward persona scorecard: each of fak's top-10 personas (free-tier dev → infra engineer → researcher) positioned on whether the affordances that persona reaches for actually exist in the tree, with one honest readiness verdict per persona. Two driven numbers: coverage (of the persona roster) and persona-debt."
---

# Persona-readiness scorecard — is each top persona served?

<!-- persona-readiness-scorecard: 2026-06-24 · process: tools/persona_readiness_scorecard.py · data: tools/persona_readiness_scorecard.data/ -->

The sibling scorecards grade fak through one lens each: [`agent-readiness`](../AGENT-READINESS-SCORECARD.md) asks whether an AI *agent* can adopt fak, [`product`](../product-scorecard/README.md) whether a person can use each *concept*. This one asks the **go-to-market** question: of the kinds of human (and one machine) who actually land on fak — from the free-tier dev who will download a binary and not read a word, through the infra engineer who has to operate it, to the researcher who wants to reproduce it — **is each one served?** Does the first affordance that persona reaches for exist in the tree? Every number below is re-derived from `tools/persona_readiness_scorecard.data/` by `tools/persona_readiness_scorecard.py` and cross-checked against the real tree, so no verdict is hand-typed: to lift a persona you ADD the real affordance.

> Regenerate: `python tools/persona_readiness_scorecard.py --markdown-dir docs/persona-scorecard`.

## Headline

| Metric | Value |
|---|---|
| **Coverage** | **100.0%** (10/10 top personas positioned) |
| **Persona-debt** | **0** (affordance/honesty 0 + coverage 0) |
| Composite score | 100.0/100 (grade A) |
| Personas served | 10 of 10 |
| Hard affordances present | 61 / 61 |
| As of | 2026-06-24 (fak v0.32.0) |

> **Read this right.** The score grades how *complete and honest the persona map is* — every required persona positioned, every readiness verdict matching the affordances actually on disk. A missing affordance is a real gap you ADD; an *overclaimed* verdict is the defect this catches.

## Standing at a glance

> Regenerate this chart in the terminal with `python tools/persona_readiness_scorecard.py --chart`.

```text
persona-readiness chart — 10 personas · score 100.0/100 (grade A) · persona-debt 0

verdict ladder (count of personas, best -> worst):
  ★ served            ████████████████████████████ 10
  ● mostly-served     ···························· 0
  ◐ partially-served  ···························· 0
  ○ unserved          ···························· 0

readiness by tier (each cell = one persona, best-served first):
  build      ★★           (2 persona(s); 2 served)
  consume    ★★★          (3 persona(s); 3 served)
  evaluate   ★★★          (3 persona(s); 3 served)
  operate    ★★           (2 persona(s); 2 served)

affordance fill per persona (hard affordances present):
  ★ oss-contributor        ████████████████████ 7/7
  ★ ai-agent               ████████████████████ 5/5
  ★ free-tier-dev          ████████████████████ 5/5
  ★ app-developer          ████████████████████ 6/6
  ★ backend-integrator     ████████████████████ 6/6
  ★ ml-researcher          ████████████████████ 6/6
  ★ benchmark-engineer     ████████████████████ 6/6
  ★ decision-maker         ████████████████████ 6/6
  ★ infra-engineer         ████████████████████ 7/7
  ★ security-engineer      ████████████████████ 7/7

coverage  [████████████████████████████████] 100.0%  (10/10 top personas positioned)

legend: ★ served   ● mostly-served   ◐ partially-served   ○ unserved
```

## The readiness ladder

| Verdict | Means |
|---|---|
| ★ served | every hard affordance this persona needs is present — they can do their first job today |
| ● mostly-served | ≥ 75% of the hard affordances present — a small, named gap |
| ◐ partially-served | ≥ 40% present — the path is half-built |
| ○ unserved | < 40% present — this persona's path is mostly missing |

## The personas (best-served first)

| | Verdict | Tier | Effort | Affordances | Persona — the job they came to do |
|---|---|---|---|---|---|
| ★ | served | build | deep | 7/7 | **Open-source contributor** — Add a feature as a leaf and ship it green, knowing the enforced rules up front so the guard teaches instead of ambushes. |
| ★ | served | build | minimal | 5/5 | **AI coding agent** — Discover what fak is, adopt it, and build on it cold — the deep audit of this is the agent-readiness scorecard. _(delegates to tools/agent_readiness_scorecard.py)_ |
| ★ | served | consume | minimal | 5/5 | **Free-tier solo dev** — Download a binary and watch it deny a tool call in under a minute — no toolchain, no key, no GPU. |
| ★ | served | consume | moderate | 6/6 | **App developer / vibe-coder** — Drop fak in front of my agent so every tool call is governed, using a config my harness already auto-loads. |
| ★ | served | consume | moderate | 6/6 | **Backend integrator** — Install fak into a service, front my model server, and depend on a frozen interface I can build a leaf onto. |
| ★ | served | evaluate | deep | 6/6 | **ML researcher** — Reproduce the determinism witness and the benchmarks offline, read the research notes, and cite the work. |
| ★ | served | evaluate | moderate | 6/6 | **Benchmark / eval engineer** — Run a benchmark command, read the measured numbers, and see fak's honest position vs the field. |
| ★ | served | evaluate | minimal | 6/6 | **Evaluator / decision-maker** — See what fak is, what's proven, how it compares, and its license — fast, with no overclaim. |
| ★ | served | operate | moderate | 7/7 | **Infra / platform engineer (SRE)** — Deploy fak serve with /metrics, /healthz, rate limits, and a container image — and a guide for Docker, Compose, and k8s. |
| ★ | served | operate | deep | 7/7 | **Security engineer** — Audit what fak denies/contains, confirm the refusals are a closed vocabulary, the audit trail is tamper-evident, and the CI scans for CVEs. |

## Per-KPI (persona-debt = affordance/honesty of the rows that exist)

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| well-formed | `rows_well_formed` | 100 | 0 | all 10 persona rows well-formed |
| reality | `affordances_present` | 100 | 0 | 61/61 hard affordances present (100%) |
| honesty | `verdict_honest` | 100 | 0 | every verdict matches its affordance evidence |

