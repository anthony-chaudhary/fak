---
title: "fak idea-scout: daily arXiv + GitHub research-to-issue feeder"
description: "The fak idea-scout searches arXiv and GitHub once a day for ideas related to agent-kernel work, dedups them against the backlog and a persistent cache, and files the best few as triage-ready GitHub issues — dry-run by default."
---

# The idea-scout (`idea-scout`)

The fak idea-scout is a research-to-issue feeder that, once a day, searches arXiv and GitHub for work adjacent to agent-kernel development. It scores each hit with a transparent, auditable relevance number, dedups candidates four ways — against a node-local seen-cache, the source-id stamp on every issue it has ever filed (the durable rung), existing issue bodies, and near-duplicate titles — and files at most a few of the best as triage-ready GitHub issues. It runs dry-run by default, planning the issues without creating any; `--live` is the explicit opt-in to actually file them. Paired with the issue-dispatch loop, it closes the backlog cycle: the scout fills the backlog, the dispatcher drains it.

> The fleet's **inbound** idea feeder. The [issue-dispatch loop](dispatch-loop.md)
> *resolves* the open backlog; nothing *fills* it. The idea-scout is that missing
> half: once a day it searches the outside world — arXiv papers and GitHub repos —
> for work adjacent to what `fak` is (an agent kernel that adjudicates tool calls
> and reuses cross-turn setup work), then files the genuinely-new, genuinely-relevant
> hits as triage-ready GitHub issues. Deduped three ways and hard-capped, so an
> unattended daily run can never storm the tracker. **Dry-run by default**; `--live`
> is the explicit opt-in to actually creating issues.

## The gap this closes

A self-hosted agent project lives or dies on staying current with two fast-moving
fields at once: agent **security** (prompt injection, tool-description poisoning,
MCP supply-chain) and inference **performance** (KV/prefix-cache reuse, paged
attention, speculative decoding). Keeping up by hand is a daily reading tax that
quietly slips. The idea-scout pays that tax automatically and lands the result
where work actually happens — the issue backlog — instead of a reading list nobody
revisits.

## The parts → the pipeline

| Stage | What it does |
|---|---|
| 0. **Topics** | A baked-in `DEFAULT_TOPICS` table maps fak's domain onto concrete queries: each topic carries an arXiv API query, a GitHub repo query, the relevance terms that earn score, and the GitHub **area label** to file under. Override the whole set with `--config` (see [`tools/idea_scout_topics.example.json`](https://github.com/anthony-chaudhary/fak/blob/main/tools/idea_scout_topics.example.json)). |
| 1. **Gather** | For every topic, fetch arXiv (the keyless Atom export API) and GitHub (`gh search repos` on the same authed CLI the dispatch loop uses). A failing source or topic is logged and skipped — one dead query never sinks the run. |
| 2. **Score** | A **transparent integer** relevance score: term hits in the title weigh more than the abstract, fresh arXiv papers and well-starred / recently-pushed repos earn bonuses. The reasons are surfaced on every candidate, so the ranking is auditable — never a black box (the same discipline as [`issue_triage.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/issue_triage.py)). |
| 3. **Dedup** | Four rungs gate every candidate (below). The durable one is a label-targeted scan of every issue the scout has ever filed; if it cannot be built completely the run refuses rather than file blind. |
| 4. **Cap** | Top-scored first, keep at most `--max-issues` (default **3**). Even a pathological day cannot storm the tracker. |
| 5. **File** | `--live` only: ensure the `idea-scout` label exists, `gh issue create` each kept candidate (labels `idea-scout`, `research`, + the topic's area), and record it in the seen-cache. Dry-run prints the plan and writes nothing. |

## The four dedup rungs (the anti-spam guarantee)

Because the tool files issues unattended, *not re-filing* is the load-bearing
property, not fetching. Every candidate must clear all four:

- **seen-cache** — `.idea-scout/seen.json`, a node-local `{source_id: record}` of
  what *this machine* filed. A **fast path only**. It is git-ignored, so it is not
  replicated and can be lost; the guarantee below therefore does not rest on it.
- **filed-stamp** — the candidate's `source_id`, read back out of the
  `<!-- idea-scout-source: … -->` stamp that every filed issue carries. **This is
  the durable rung: a source filed once is never filed again, even years later.**
  The index is built by a query *targeted* at the `idea-scout` label
  (`gh issue list --state all --label idea-scout`), so it covers the scout's entire
  filing history — open and closed, from issue #1 onward — and its completeness
  depends on how many issues the *scout* has filed (≤ `--max-issues` per day), never
  on how fast the tracker as a whole grows. GitHub is the replicated store.
- **issue-body** — the candidate's source *URL* appears verbatim in some existing
  issue body ⇒ a human already wrote it up. Best-effort: scanned over the
  `issue_scan_limit` most recent issues.
- **title-near** — token-overlap (Jaccard ≥ `dup_jaccard`) with any existing
  issue title ⇒ a near-duplicate a human already opened by hand. Same window.

A candidate is filed only if it is new on all four rungs **and** scores ≥
`--min-score`.

### Why rung 2 is targeted and not just a bigger window

The first two rungs used to be "a local cache" plus "whatever turns up in the
`issue_scan_limit` (800) most recent issues". Both failed at once (#5543): the cache
is unreplicated and was lost on a node, and the 800-issue window had shrunk to under
three weeks of coverage as issue volume grew, hiding every idea-scout issue below
`#4737`. Three already-triaged, already-closed sources were re-filed.

Raising the window is not the fix — it is the same race with a later start. The fix
is that rung 2's query is **scoped to the population being deduped** instead of to
recency, so its cost and coverage track the scout's own output rather than the
tracker's. `scout_scan_limit` (default **5000**) is a saturation *tripwire*, not a
window: at ≤3 issues/day it is years of headroom, and if the scan ever comes back
saturated the run **refuses (exit 2)** rather than filing against a possibly-truncated
index. Same refusal if `gh` fails outright. Silent degradation is what caused the
bug, so growth has to surface as a loud stop.

The run report makes this auditable — `--json` carries a `dedup_index` block with
`filed_issues_scanned`, `filed_stamps`, and `scout_index_complete`.

## Run it

```bash
# dry-run: plan the issues, file nothing, write nothing (the default)
python tools/idea_scout.py

# machine-readable plan (what a scheduled run logs)
python tools/idea_scout.py --json

# file at most 3 issues for real, and record them in the seen-cache
python tools/idea_scout.py --max-issues 3 --live

# narrow/replace the topic set and tune the knobs
python tools/idea_scout.py --config tools/idea_scout_topics.example.json
```

Exit codes: `0` ran clean · `2` infra error (gh missing / not authed / not a repo,
or every source failed with no cache to fall back on — it **refuses** rather than
risk a blind spam run).

## The daily task (the "keep current" loop)

One Windows Scheduled Task fires the scout once a day. It installs **dry-run by
default**; `-Live` opts into issue creation. Unlike the dispatch loop's 10-minute
spawn tick, this task spawns no worker — its only side effect is `gh issue create`,
so there is no worker-cap DoS surface to bound, just the per-run issue cap.

| Task | Installer | Cadence | Side effect |
|---|---|---|---|
| `FleetIdeaScout` | [`register_idea_scout.ps1`](https://github.com/anthony-chaudhary/fak/blob/main/tools/register_idea_scout.ps1) | daily (`-At`, default 09:00) | FILE — up to `-MaxIssues` triage-ready issues (`-Live` only). |

The task is installed through `fak loop run` (not python directly), so each daily fire
records `fire`/`start`/`end` wrapper rows in `.fak/loops.jsonl` under
`idea-scout/task-scheduler` — `fak loop status` then shows whether the scout actually ran,
not just that Task Scheduler logged `LastResult=0`.

```powershell
# install dry-run (logs the plan daily, files nothing)
.\tools\register_idea_scout.ps1 -Workspace C:\work\fak

# go live: file at most 3 issues each morning
.\tools\register_idea_scout.ps1 -Workspace C:\work\fak -Live -MaxIssues 3 -At 09:00

# status / remove
.\tools\register_idea_scout.ps1 -Action status
.\tools\register_idea_scout.ps1 -Action remove
```

Together with the dispatch loop, the backlog becomes a closed cycle: **the scout
feeds it, the dispatcher drains it** — `search → file → route → ship #N → witness →
close`, unattended.

## A note on what it does *not* do

The scout does not judge whether an idea is *correct* or *worth building* — it
judges whether it is *new and on-topic*, and hands a human the link. Every filed
issue says so in its body and carries a triage hint; close it `wontfix` /
`duplicate` if it is not worth pursuing. The labels (`idea-scout` + `research`)
make the whole inbound stream filterable, so a triage pass over "what did the scout
bring in this week" is one `gh issue list --label idea-scout` away.
