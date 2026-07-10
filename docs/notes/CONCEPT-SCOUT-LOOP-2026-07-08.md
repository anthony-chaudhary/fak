---
title: "Concept: scout-loop — the research→backlog super-loop (2026-07-08)"
description: "scout-loop is a cadenced super-loop that orders idea-scout, study-repo, and field-borrow to turn crawled repo leads into witnessed backlog, one lead per pass."
---

# Concept: `scout-loop` — the research→backlog super-loop (2026-07-08)

**What shipped:** a meta-skill, `.claude/skills/scout-loop/SKILL.md`, plus its
unattended fuel (`.claude/goal-prompts/scout-and-study-witnessed.md`) and cadence
installer (`tools/register_scout_loop.ps1`). Epic **#3357**; leaves **#3358**
(SKILL.md), **#3359** (fuel), **#3360** (installer), **#3361** (this indexing).

## The gap it closes

fak had the two halves of a research→backlog loop but not the seam between them:

- **The crawler** — `tools/idea_scout.py` (installed as the `FleetIdeaScout`
  Scheduled Task) scans arXiv + GitHub daily and files deduped `idea-scout`
  triage issues. It *feeds*; it stops at a needs-triage queue.
- **The study pipeline** — [`/study-repo`](../../.claude/skills/study-repo/SKILL.md)
  clones a specific repo, reads the CODE at a pinned `@sha`, and decomposes it into
  small tickets; [`/field-borrow`](../../.claude/skills/field-borrow/SKILL.md)
  witnesses one named capability against fak (`fak_feature_query`/`fak index` →
  PRESENT/PARTIAL/ABSENT) and files it grounded.

Nothing connected them. A repo-shaped lead surfaced by the crawler sat in the queue
until a human *remembered* to run `/study-repo` on it. The deep study→witness→file
pass — the expensive, high-value part — was never on a cadence.

## The move: a super-loop that orders the tools we already have

`scout-loop` re-implements none of the crawlers or study skills. It **orders** them,
one lead per pass:

**CRAWL** (read the `idea-scout` queue + `industry_scorecard.py` /
`industry_freshness_cadence.py` + the `RESEARCH-*`/`CONCEPT-*` corpus) → **SELECT**
the single highest-value repo-shaped lead → **STUDY** it via `/study-repo` →
**WITNESS + FILE** each borrow via `/field-borrow` (PRESENT dropped; PARTIAL/ABSENT
filed as small leaves under the right epic) → **REGISTER** a dated
`CONCEPT-STUDY-*` note → **CADENCE** (a Scheduled Task fires one detached `/goal`
pass, plan-by-default).

It is to `idea-scout`+`study-repo` what `/super-loop` is to a single
`/dos-dispatch`: the durable, cadenced front door that runs the proven unit
repeatedly and safely. It *consumes* the crawler's feed and *produces* the studied
backlog the dispatch loop (`/super-loop`) then resolves.

## Where it sits (do not duplicate a sibling)

| | Starts from | Mechanism | Terminal action |
|---|---|---|---|
| `idea-scout` (tool) | an outward feed | automated arXiv/GitHub scan + dedup | files raw `idea-scout` triage issues |
| `/study-repo` | one repo URL | clone → read code → decompose | files small witnessed leaves |
| `/field-borrow` | one named capability | dogfood witness | files one grounded issue |
| `/industry-score` | the field taxonomy | coverage + parity-debt scorecard | a scorecard (files nothing) |
| `/super-loop` | the ready backlog | launch N detached workers | resolves leaves (witnessed ancestry) |
| **`/scout-loop`** | **the crawler output** | **crawl → select → study → witness → file**, cadenced | **runs the study pass unattended, one lead/pass** |

## The honesty boundary (inherited, load-bearing)

- **A crawl is not a borrow; a study is not a ship.** The loop files *triaged,
  witnessed* backlog and NEVER reports an issue resolved — ancestry (`Fixes #N`)
  does that, later, on the trunk.
- **Every borrow cites a real `path:line@sha`** in an actually-cloned source — the
  `dos_citation_resolve` fabrication class `/study-repo` exists to catch.
- **Witness before file.** A PRESENT ("fak already has this") is dropped, not
  filed; that is a complete, honest pass, not a miss.
- **Decompose, never a monolith** — the ship-alone test; no "adopt everything from
  repo X" leaf.
- **One lead per pass** — the anti-storm bound; throughput is a cadence, not a batch.
- **Clone into scratch, never the tree**, and leak-check every issue body (no
  absolute path / host / secret / PII from the foreign clone).

## "Set it to run" — the cadence, safe by default

`tools/register_scout_loop.ps1` installs the `FleetScoutLoop` Scheduled Task (daily,
default `10:30`, after the 09:00 `FleetIdeaScout` so the triage queue is fresh). It
mirrors `register_idea_scout.ps1` exactly: `fak loop run` wrapper (not a
`powershell -Command` wrapper — the space-in-path silent-no-op trap), current-user
S4U principal (windowless), `MultipleInstances IgnoreNew`. **Plan by default** —
installed without `-Launch`, each fire runs `launch_goal_detached.ps1 -PlanOnly`,
resolving the account + preflight but **spawning nothing**, logging only the plan to
the loop ledger. `-Launch` is the explicit opt-in that detaches one study worker per
fire; the launch path keeps the `dispatch_preflight.py` no-DoS cap intact
(`SPAWN_OK` re-checked per spawn — `-SkipPreflight` is deliberately not exposed).

Registering the S4U task itself requires an **elevated** shell (same as its
siblings `FleetIdeaScout` / `FleetIssueDispatch`); from a non-elevated session the
script builds and validates but `Register-ScheduledTask` returns `Access is denied`.

## Companions

- Feeder: `tools/idea_scout.py` / `register_idea_scout.ps1` (the `FleetIdeaScout` task).
- Front-half it drives: [`study-repo`](../../.claude/skills/study-repo/SKILL.md).
- Back-half it drives: [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md).
- The competitive map it reads: [`industry-score`](../../.claude/skills/industry-score/SKILL.md).
- The resolve side that consumes its output: [`super-loop`](../../.claude/skills/super-loop/SKILL.md) + [`wave-harvest`](../../.claude/skills/wave-harvest/SKILL.md).
- Worked study passes in the same family: [`CONCEPT-STUDY-DYNAMO-2026-07-08`](CONCEPT-STUDY-DYNAMO-2026-07-08.md), [`CONCEPT-STUDY-MINIO-MEMKV-2026-07-08`](CONCEPT-STUDY-MINIO-MEMKV-2026-07-08.md).
- Epic: **#3357**.
