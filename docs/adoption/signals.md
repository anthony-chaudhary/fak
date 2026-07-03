---
title: "Adoption-signals dashboard: the honest signals worth watching"
description: "An honest spec for tracking whether fak/DOS is actually being adopted — stars, forks, watchers, directory listings, mentions, integration recipes, distinct harnesses, docs reachability. Every signal is labeled OBSERVED (a number a third party controls) vs WITNESSED (an artifact fak authored), each is a proxy and not proof, and each row names what it does NOT prove. No vanity metric stands unlabeled; no number is invented."
slug: adoption-signals
keywords:
  - adoption metrics
  - vanity metrics
  - GitHub stars
  - proxy vs proof
  - witnessed vs observed
  - conflation discipline
  - go-to-market measurement
  - honest dashboard
date: 2026-07-02
---

# Adoption-signals dashboard: the honest signals worth watching

> **TL;DR:** stars, forks, and mentions are **proxies**, not proof of adoption.
> This spec lists the external signals worth watching, where each is collected,
> and — the part most dashboards skip — **what each one does NOT prove**. Every
> row is labeled `OBSERVED` (a number a third party controls and can inflate or
> zero) or `WITNESSED` (an artifact fak authored and can verify on disk), per the
> [conflation discipline](../CONFLATION-SCORECARD.md). A rising number here means
> *look closer*, never *we have won*.

This is dimension **K — Adoption measurement** of the
[concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md).
It serves one concept: **verify, don't trust** — the same rule the kernel applies
to a tool call, turned on our own go-to-market numbers. A good-looking number must
not masquerade as adoption.

## The one rule this dashboard exists to enforce

**Popularity signals are observed, not witnessed.** A star count is a number GitHub
computes and reports; fak neither authored it nor controls it. It can be bought,
brigaded, or bot-inflated, and it says nothing about whether a single person ran
the binary. So every signal below carries two honesty tags:

- **Provenance** — `OBSERVED` (relayed from an external party; fak cannot vouch for
  it) or `WITNESSED` (fak authored the artifact and can re-derive it from the tree).
  This is exactly the split the [conflation scorecard](../CONFLATION-SCORECARD.md)
  and `tools/conflation_scorecard.py` enforce on every number fak reports.
- **What it does NOT prove** — the load-bearing column. A signal with no honest
  disclaimer is a vanity metric.

Nothing here is proof of *usage*. The closest we get to witnessed usage is the
count of artifacts fak itself ships (recipes, harnesses covered, reachable docs) —
and even those prove only that the on-ramp *exists*, not that anyone walked it.

## The signals

Grouped by provenance. `OBSERVED` signals come first because they are the ones most
easily mistaken for proof.

### OBSERVED — third-party numbers (a proxy at best)

| Signal | Source | Collection method | Honestly measures | Does NOT prove |
|---|---|---|---|---|
| **Stars** | GitHub repo | `gh api repos/<owner>/<repo> --jq .stargazers_count` | Bookmark-level interest; a person saw the repo and tapped the button | That anyone cloned, ran, or kept using it. Buyable and brigade-able. |
| **Forks** | GitHub repo | `gh api repos/<owner>/<repo> --jq .forks_count` | Slightly stronger intent — someone took a copy | Active use or contribution. A fork can be abandoned or an archive snapshot. |
| **Watchers / subscribers** | GitHub repo | `gh api repos/<owner>/<repo> --jq .subscribers_count` | Intent to follow changes over time | That the watcher runs the binary; watching is not using. |
| **Mentions** | Web / social / forums (HN, Reddit, X, blogs) | Manual scan + dated links captured in a ledger; no scraping claim beyond what is actually run | That the concept surfaced in a place fak did not control | Sentiment, reach, or that the mention was positive. Count says nothing about quality. |
| **Traffic (views / uniques / clones)** | GitHub Insights (`Traffic` API, 14-day window) | `gh api repos/<owner>/<repo>/traffic/{views,clones}` (needs push scope) | Short-window curiosity | Retention — the window is only 14 days and cannot be back-filled. |
| **Directory listings merged** | External lists (awesome-*, registries) | Link to the merged PR/entry; the *acceptance* is external | A third-party maintainer judged fak list-worthy | Downstream installs from that listing. |

Every `OBSERVED` row is a number a party other than fak controls. Treat a jump as a
prompt to ask *why*, not as an outcome.

### WITNESSED — artifacts fak authored (on-ramp exists, not that it was walked)

| Signal | Source | Collection method | Honestly measures | Does NOT prove |
|---|---|---|---|---|
| **Integration recipes shipped** | This repo | Count `docs/integrations/*` / recipe files on the git-tracked tree | The number of stacks fak has a documented drop-in path for | That any reader followed a recipe. Supply, not demand. |
| **Distinct harnesses covered** | This repo | Enumerate the harnesses in [`docs/supported/agent-harnesses.md`](../supported/agent-harnesses.md) | Breadth of the documented integration surface | Real-world use with each harness; coverage is a claim of *fit*, verified separately. |
| **Docs reachable** | This repo | `python tools/check_index_sync.py --audit-tree` (every doc linked from `INDEX.md` / `llms.txt`) | That the human- and machine-facing front doors have no orphans | That the docs are read or understood. |
| **SEO/AEO surface intact** | This repo | `python tools/seo_aeo_scorecard.py` (front-matter present, no regression) | That new docs are machine-discoverable | Discovery itself — findable is not found. |

`WITNESSED` rows are re-derivable from the tree, so they cannot be inflated by an
outside party. Their honest ceiling is that they measure *what fak built*, not
*what anyone did with it*.

## Vanity-metric fence (what stays OFF this dashboard)

Per the epic's honest-scope fence, the dashboard MUST NOT carry:

- **Any market-adoption or market-share claim.** We have not run that measurement.
- **A composite "adoption score"** that blends observed and witnessed signals into
  one headline number — it would launder a buyable star count into an authored fact.
- **Vanity totals with no disclaimer** — a star count shown without its "does not
  prove" line. If the honesty column can't be filled, the row does not ship.
- **Invented or projected numbers.** Cells are populated by the collection command
  or left empty. Simulated values, if ever added, are labeled `SIMULATED`.

## Proposed dashboard shape

A single flat table (or two stacked tables, OBSERVED then WITNESSED), never a leaderboard:

```
ADOPTION SIGNALS — <date>            provenance | value | Δ since last | does-not-prove
────────────────────────────────────────────────────────────────────────────────────
OBSERVED  stars                       OBSERVED  |   —   |     —        | not usage; buyable
OBSERVED  forks                       OBSERVED  |   —   |     —        | not active use
OBSERVED  watchers                    OBSERVED  |   —   |     —        | not usage
OBSERVED  mentions (dated ledger)     OBSERVED  |   —   |     —        | not sentiment/reach
OBSERVED  traffic (14d views/clones)  OBSERVED  |   —   |     —        | not retention
OBSERVED  directory listings merged   OBSERVED  |   —   |     —        | not installs
WITNESSED integration recipes         WITNESSED |   —   |     —        | supply, not demand
WITNESSED distinct harnesses covered  WITNESSED |   —   |     —        | fit-claim, not use
WITNESSED docs reachable (orphans=0)  WITNESSED |   —   |     —        | reachable, not read
WITNESSED seo/aeo scorecard           WITNESSED |   —   |     —        | findable, not found
```

Design rules:

- **Provenance column is mandatory and first-class**, not a footnote.
- **The `does-not-prove` column ships with every row.** It is the point of the doc.
- **Δ is descriptive, not a target.** No goal-seeking on an observed number, or the
  team optimizes a proxy.
- Values shown as `—` until a real collection run fills them. This spec ships the
  *shape*; it invents no numbers.

## Refresh cadence

| Signal group | Cadence | Why |
|---|---|---|
| Stars / forks / watchers | Weekly | Slow-moving; daily noise adds nothing. |
| Mentions ledger | On sighting + a weekly sweep | Mentions are discovered, not polled. |
| Traffic (views/clones) | Weekly (14-day window is non-back-fillable) | Miss the window and the data is gone. |
| WITNESSED artifact counts | On every docs ship (already gated in CI) | They change only when the tree changes. |

Each refresh stamps a date and appends, rather than overwriting, so a trend is
visible and a suspicious spike stays on the record.

## Verify

```
gh api repos/<owner>/<repo> --jq '{stars: .stargazers_count, forks: .forks_count, watchers: .subscribers_count}'
python tools/seo_aeo_scorecard.py         # this doc has front-matter; no SEO regression
python tools/check_index_sync.py --audit-tree   # this doc is reachable from INDEX.md
```

## What this doc does and does not claim

- **Does:** name the signals worth watching, where each comes from, how to collect
  it, and — for every one — what it honestly measures and what it does not prove.
- **Does not:** assert fak is adopted, report any count, or blend proxies into a
  single "we're winning" number. Rising signals here mean *look closer*.
