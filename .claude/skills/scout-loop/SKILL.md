---
name: scout-loop
description: "The super-loop that closes the research→backlog loop — it chains the outward CRAWLERS (the daily `idea-scout` arXiv/GitHub feed, the industry scans, the RESEARCH/CONCEPT corpus) into the STUDY pipeline (`/study-repo` → `/field-borrow`) and runs the whole thing on a cadence. The crawler surfaces repo-shaped leads into a needs-triage queue and stops; turning any one into scoped, witnessed, license-clean backlog is still a manual pass someone has to remember to run. This skill is that seam, automated: once per pass it CRAWLS. Use when this named workflow matches the task."
allowed-tools: Read, Bash, Write
metadata:
  opencode: claude-only   # the commit-by-explicit-path, honesty-boundary, no-monolith, and scratch-only-clone discipline are load-bearing and not portable per-skill
---

# /scout-loop — crawl the field, study the best lead, file witnessed backlog

> **The research→backlog super-loop.** fak already has an outward crawler
> (`fak idea-scout` plans deduped `idea-scout` triage issues from arXiv, GitHub,
> Hacker News, and Reddit; `--live` is the separate mutation gate) and a study pipeline (`/study-repo` reads a repo at a pinned `@sha` and
> decomposes it; `/field-borrow` witnesses one capability against fak and files it
> grounded). Nothing connects them: leads pile up in a triage queue and the deep
> study→witness→file pass waits for a human to remember it. `scout-loop` is that
> seam — it drives the tools we already have, in order, one lead per pass, and can
> run unattended on a cadence.

This is a **meta-skill**: it re-implements none of the crawlers or study skills.
It orders them and holds the discipline the raw pieces assume — **PLAN/witness
first, one lead per pass, no monolith, and a crawl is never a ship.**

## Its place in the family (do not duplicate a sibling)

| Skill / tool | Starts from | Mechanism | Terminal action |
|---|---|---|---|
| `fak idea-scout` | an outward **feed** | dry-run arXiv/GitHub/HN/Reddit scan + filed-stamp dedup | plans raw triage issues; `--live` files |
| [`/study-repo`](../study-repo/SKILL.md) | one **repo URL** | clone → read the code → decompose | files many small witnessed leaves |
| [`/field-borrow`](../field-borrow/SKILL.md) | one **named capability** | dogfood `fak_feature_query`/`fak index` | files one grounded issue |
| [`/industry-score`](../industry-score/SKILL.md) | the **field taxonomy** | coverage + parity-debt scorecard | a scorecard (files nothing) |
| [`/super-loop`](../super-loop/SKILL.md) | the **ready backlog** | launch N detached `/goal` workers | resolves leaves (witnessed ancestry) |
| **`/scout-loop`** (this) | **the crawler output** | **crawl → select → study → witness → file**, cadenced | **runs the study pass unattended, one lead per pass** |

`scout-loop` is to `idea-scout`+`study-repo` what `/super-loop` is to a single
`/dos-dispatch`: the durable, cadenced front door that runs the proven unit
repeatedly and safely. If you were handed **one specific repo**, skip straight to
`/study-repo`. If you already have a **named capability** to check, use
`/field-borrow`. Reach for `scout-loop` when the job is *"keep finding new leads
and converting them"* — the loop, not a single pass.

## Durable monitored-source queue

Start each pass from [`docs/research/monitored-repositories.json`](../../../docs/research/monitored-repositories.json), rendered with:

```powershell
fak study-monitor --due-days 14
```

Prefer the highest-priority due `candidate` or `watch` row before opening a fresh outward search. Exact `owner/name` is the deduplication key. When outward discovery finds a stronger untracked lead, register its checked revision and `last_checked` before deep study; after study, update the same row and link its `study_note`. This makes “most recently checked” queryable rather than recoverable only from filenames or session memory.
## The pass — one lead, end to end

### 1 — CRAWL: capture the first-class scout plan

Run the Go front door from the repo root. Dry-run is the mandatory default:

```bash
fak idea-scout --json > idea-scout-plan.json
```

Inspect candidates, scores, source revisions, dedupe evidence, and planned issue contracts. For a
pinned replay or unavailable network, use `--candidates FILE`. Direct
`python tools/idea_scout.py` use is a legacy/debug comparison only.

Issue creation is a separate explicit gate and requires operator mutation intent:

```bash
fak idea-scout --json --live
```

Do not infer permission to cross that gate from this skill matching. Validate selected contracts
through `fak-dev issue contract` and price any worker launch through `fak dispatch` before execution.

### 2 — SELECT: one highest-value repo-shaped lead (never a batch)

From the crawl, pick **one** lead that is **repo-shaped** — a GitHub URL, a
paper-with-code link, a named codebase — and where the field signal says the
parity-debt or novelty is highest. **Prefer the fresh-lane leads** (a `**Why
surfaced**` line marked `trending` / `very fresh` / `actively updated`): a repo that
just appeared or is climbing relative to ours is the highest-novelty, most-perishable
lead — study it before it's an incumbent everyone already read. One lead per pass is
the anti-storm bound (the same discipline as idea-scout's per-run cap). If nothing
fresh is repo-shaped, **stop clean** — an empty pass is a valid result; never invent
a lead to have something to do.

Prefer a lead that is *not already studied*: grep `docs/notes/CONCEPT-STUDY-*` and
`gh issue list --search` for its name first, so the loop doesn't re-study a repo a
prior pass (or a human) already decomposed.

### 3 — STUDY: hand the lead to `/study-repo` (do not restate it)

Require the returned study to expose both the default frontier and the **bounded-superset
coverage frontier**. Do not select leads only because they might replace fak's current default;
a lead is also high-value when it serves a credible user/job/constraint cohort through an
`OPTIONAL-MODULE` or `RECIPE` seam. Deprioritize feature-count novelty, genuinely superseded
approaches, and moment-in-time claims without a dated `WATCH` review trigger. This keeps
scouting broad in user coverage but bounded by fak's scope and support economics.

Invoke [`/study-repo`](../study-repo/SKILL.md) on the selected lead. That skill
owns the whole acquisition front-half — shallow-clone **into scratch, never the
tree**, pin the commit `@sha`, read the CODE (load-bearing modules + tests + recent
commits) not the README pitch, extract candidate borrows each grounded at a real
`path:line@sha`, and decide borrow-vs-integrate on the license. Do not re-implement
any of that here; drive it.

### 4 — WITNESS + FILE: `/field-borrow`'s witness step, then file small

For **each** surviving candidate, run [`/field-borrow`](../field-borrow/SKILL.md)'s
witness step — dogfood `fak_feature_query` / `fak index docs|leaves|verbs|claims`
(plus a raw `Grep` to guard the lexical ranker's false-ABSENT) → classify
**PRESENT / PARTIAL / ABSENT**:

- **PRESENT** → fak already has it: drop the candidate, record the card. A
  witnessed "we already had this" is a real, good result — not a miss.
- **PARTIAL / ABSENT** → ground it in the **fak seam** (`path:line` in *fak*), then
  file a **small independently-shippable leaf** under the right epic, carrying
  **both** anchors (source `path:line@sha` + fak seam), the dogfood witness, and a
  first checkable step. Dedup first (`gh issue list --search`). **Never a
  monolith** — the ship-alone test from `/study-repo` step 5. **Leak-check every
  body**: no absolute path, host, secret, or PII from the foreign clone.

### 5 — REGISTER: the trail, or it lived only in a transcript

Record the pass in a dated `docs/notes/CONCEPT-STUDY-<repo>-<date>.md` (URL +
pinned `@sha`, what you read, the borrow · source `path:line@sha` · witness ·
inspire|integrate · filed # table), add its **`INDEX.md`** line, and commit the
docs lane by explicit path with a `(fak docs)` trailer. This registration — not
the issues alone — is what makes the research durable and keeps `fak index
freshness` clean.

### 6 — CADENCE: set it to run (unattended)

One *pass* is above; the *loop* is a cadence of passes. Two shapes:

- **In-session (attended).** Run steps 1–5 now for one lead, then stop. Re-run for
  the next lead when you choose.
- **Scheduled (unattended).** `tools/register_scout_loop.ps1` installs a daily
  Scheduled Task (`FleetScoutLoop`) that fires one pass via a detached `/goal`
  worker reading `.claude/goal-prompts/scout-and-study-witnessed.md`. **Plan by
  default** — installed without `-Launch`, the fire runs the launcher in
  `-PlanOnly` mode and spawns nothing; `-Launch` is the explicit opt-in. Same
  safe-by-default contract as `register_idea_scout.ps1` and `/super-loop`.

```powershell
.\tools\register_scout_loop.ps1                 # install, PLAN-mode daily (spawns nothing)
.\tools\register_scout_loop.ps1 -Action status
.\tools\register_scout_loop.ps1 -Launch -At 10:30   # go live: one detached study pass/day, after the 09:00 idea-scout
```

Because each fire spawns a Claude session, the launch path keeps the
`dispatch_preflight.py` no-DoS cap intact (`launch_goal_detached.ps1` re-checks
`SPAWN_OK` per spawn) — never route around it with `-SkipPreflight`.

## The honesty boundary (do not cross)

- **A crawl is not a borrow; a study is not a ship.** This loop files *triaged,
  witnessed* backlog. It NEVER reports an issue resolved — ancestry (`Fixes #N` on
  the trunk) does that. It hands its filed leaves to the dispatch loop /
  `/super-loop`, which resolve them.
- **Every borrow cites a real `path:line@sha`** in an actually-cloned source, never
  a README claim — the `dos_citation_resolve` fabrication class `/study-repo`
  exists to catch.
- **Witness before file.** A PARTIAL/ABSENT is grounded in a dogfood witness; a
  PRESENT is dropped, not filed. "fak already has this" is a complete pass.
- **Decompose, never a monolith.** No "adopt everything from repo X" leaf — the
  ship-alone test decides.
- **One lead per pass.** The anti-storm bound; the loop's throughput is a cadence,
  not a batch.
- **Clone into scratch, never `C:\work\fak`.** The foreign tree must never be
  committable (the `*scratchpad*` gitignore is a backstop, not the plan).

## What "done" proves (per pass)

- One repo-shaped lead was **cloned and read at a pinned `@sha`**; every filed
  borrow cites real code.
- Each filed leaf carries a **source `path:line@sha`**, a **fak seam**, a **dogfood
  witness (PRESENT/PARTIAL/ABSENT)**, and a **first checkable step**; PRESENT cases
  were dropped, not re-filed.
- **No monolith** was filed; the work is small independently-shippable leaves under
  the right epic.
- The pass is **registered**: a dated `docs/notes/CONCEPT-STUDY-*` note + its
  `INDEX.md` line; `fak index freshness` reads clean.
- If unattended: the `FleetScoutLoop` fire is witnessed in the loop ledger
  (`fak loop status`), and no issue was reported "resolved" by the loop.

## Anti-patterns

- ❌ Re-implementing the crawl (re-scanning arXiv) or the study (re-reading the
  clone logic) — drive `idea-scout` / `/study-repo` / `/field-borrow`, don't
  duplicate them.
- ❌ Studying a batch of leads in one pass — one lead, then stop; the cadence is the
  throughput.
- ❌ Filing a borrow you did not locate at a real `path:line@sha`, or the
  "adopt repo X" monolith.
- ❌ Filing a PRESENT capability fak already ships — the witness step exists to drop
  it.
- ❌ Cloning the foreign repo into the fak tree, or leaking an absolute path / host /
  secret from it into a fak ticket.
- ❌ Installing the scheduled task with `-Launch` before a plan-mode fire is
  witnessed, or passing `-SkipPreflight` to route around the no-DoS cap.
- ❌ Reporting a filed leaf as "resolved" — a crawl/study is not a ship; ancestry is.
- ❌ Inventing a lead when the crawl is empty — a clean empty pass is the correct
  result.

## When NOT to use

- **You were handed one specific repo.** Go straight to `/study-repo`; the loop is
  overhead for a single known lead.
- **You have a named capability to check.** Use `/field-borrow` directly.
- **You want to resolve the backlog, not grow it.** That's `/super-loop` /
  `/dos-dispatch` — the resolve side of the loop, which consumes what `scout-loop`
  files.
- **The idea-scout queue and the field scans are empty/stale with nothing new.**
  Nothing to convert — an empty pass, not a forced study.

## Honest limits

- **The witness is lexical + a snapshot.** `fak_feature_query`'s ranker is
  substring matching (false-ABSENT risk) and its verdict is true only as of today
  — re-witness before acting on an old pass (same caveat as `/field-borrow` /
  `/study-repo`).
- **The crawl is only as fresh as its feeders.** If `FleetIdeaScout` hasn't run,
  the triage queue is stale; the loop studies what's there, it does not force a
  re-scan.
- **Composes with, does not replace:** `idea-scout` (the feed), `/study-repo` (the
  acquisition front-half it drives), `/field-borrow` (the witness+file back-half),
  `/industry-score` (a witnessed PRESENT/PARTIAL/ABSENT is a coverage row it can
  absorb), `/super-loop` (resolves the leaves this loop files).
