---
title: "Unsticking Fak Ops — what the naive headless path gets right"
description: "Why fak's dispatch machinery launches 4% of what it plans and closes nothing, read against the job repo's measured record where a hand-given list of ten issues ships at 92.6%. Five transferable lessons, the counterweight, and a ranked do-this-first list."
---

# Unsticking Fak Ops

> **In one breath:** fak dispatches workers, but most of them never start. A dead
> worker looks like an idle fleet. Count commits, not processes. The simple path
> ships more because it has fewer layers to fail.

**One line:** the admission and observation layers around Fak Ops cost more than
the work they admit and cannot see whether that work landed, so the layers must be
paid per shipped commit instead of per iteration and scored with the same ruler as
a naive headless dispatch.

**Date:** 2026-09-28 · **Scope:** read-only analysis. Nothing here is implemented.
**Method:** evidence-over-narrative. Every number below was re-derived from the
working tree on 2026-09-28, not read off a summary doc.

## The question

An operator observed that plain copy-paste dispatch — hand the model the next ten
issues, let it fan out headlessly — seemed to outperform Fak Ops itself. The job
repo had already answered the same question in a file triggered by the same
complaint:

> "before, raw copy-paste of next-up + fanout worked pretty well; now it's hard to
> get it to do anything despite 100s of open pickable items."
> — `docs/_audits/dispatch-throughput-audit-2026-06-11.md:5`, job repo

Its verdict: *"Not broken at the core — strangled by three compounding feedback
loops in the layers AROUND the work... The work layer they wrap is the part that
still ships."* (`:13`, `:87`). This page asks the same question of fak's own
ledger, and reaches a harsher answer.

## The comparison

| | Fak Ops | Naive headless dispatch |
|---|---|---|
| Prompt | ~54,000 tokens of control-plane `SKILL.md` (19 skills, 217 KB) | `fak issue-orchestrator --top 10 --max-waves 1` — 45 characters |
| Gate code | 55,232 lines of Go (`internal/dispatch*`, `ops`, `orchestration`, `issueorchestrator`) | none |
| What it launches | **1 of 25** goal runs (4%) | a list, every time |
| What it closes | **0** across 56 logged issue-resolve ticks | 312 of 337 picks (92.6%) |
| Operator headline number | live worker count | shipped commits |

The last row is where the two systems actually differ, and the next section is why.

## Where Fak Ops is stuck — four measured symptoms

### 1. It plans and does not launch

`.goal-runs/*.receipt.json`, n=25: **24 `plan_only`, 1 `launched`.** A 4% launch
rate. The same shape shows in the docs: `docs/agentic-issue-dispatch.md:18` still
describes itself as "the manual packet shape **that a native command can later
mechanize**" — a 539-line runbook, as of its last write, still run by hand. Fifty-five
thousand lines of Go wrap a runbook the pipeline is not reliably executing.

### 2. When it does launch, most workers die on arrival

`internal/dispatchdoa/doa.go:11` records the incident from the repo's own retained
corpus. Between 2026-07-28 and 2026-08-03 the dispatcher spawned workers with an
argv flag the `fak.exe` on `PATH` did not define; Go's `flag` package rejected it
and exited before the guard launched the agent. **350 dead worker-units — 91.6% of
the 382 spawned — and nothing surfaced it.** Every one wrote a `.witness` carrying
`claim: CLAIM_NO_COMMIT, reason: unknown`, and `unknown` was already the fleet's
largest no-commit bucket, so a total outage was indistinguishable from background
noise.

The present corpus shows the same ratio: of 59 `.witness` files,
**26 `CLAIM_WITNESSED`, 25 `CLAIM_NO_COMMIT`, 8 `CLAIM_UNWITNESSED`** — 42% no-commit.

### 3. Its headline number cannot register failure

`cmd/fak/dispatch_status.go:243` sets `LiveWorkerCount: len(workers)`, and `:280`
and `:313` print it as the surface an operator reads. The DOA docstring is explicit
about the consequence: *"`fak dispatch status`, the surface an operator actually
reads, showed '0 live worker(s)' throughout — which reads exactly like an idle
fleet."*

A process that died at argv parse is not live. So a total outage and a quiet night
produce the same screen. The gauge is measuring liveness, and liveness is not the
thing anyone is trying to buy.

The backlog loop shows the same blindness in the aggregate. Across 56 logged
`issue-resolve-progress` ticks: `baseline_open: 2516` → `open_now: 2594`,
`closed_by_loop_total: 0`, `closures_toward_target: 0`. The backlog **grew 78** and
the loop closed nothing, while reporting metrics every tick.

### 4. Admission costs more than the work

From `.fak/loops.jsonl`, the most recent tick: `preflight_ms: 31625` of
`tick_total_ms: 52378` — **60.3% of wall time is admission checking**. Loop-wide
the preflight mean is 10,425 ms against a 21,828 ms tick mean (47.8%). The same
tick carries `preflight_cap: 6` against `max_workers: 30`.

And the provisioning is oversized against its own measurement:
`cmd/dispatchworker/guard.go:115` hardcodes `claudeGuardBaselineTokens = 62000`,
where the measured launch constituent set (`AGENTS.md` + `llms.txt` + `CLAUDE.md`)
is ~20,028 tokens — a ~42,000-token dead reservation per worker, per tick.

## Five transferable lessons

### Lesson 1 — Measure launches, not live workers

The single most expensive defect above is not that workers died. It is that the
operator-facing surface could not tell. `LiveWorkerCount` answers "is a process
running," which is a different question from "did work land."

A naive headless fanout has no such ambiguity. A human watching a run sees zero
commits and knows immediately. The elaborate system's advantage — rich telemetry —
became its liability, because the one number that was on top was the one number
that could not fail.

**Apply:** make commits-landed-in-window the headline; demote live workers to a
secondary line. A surface whose bad news and good news look the same is not an
observability surface.

### Lesson 2 — The checker an agent can call must fail closed

`docs/agentic-issue-dispatch.md:296`: the **MCP** `dos_arbitrate` reads no live
lease set, so it arbitrates against nothing and every lane reads free. Observed
live: it returned `"cluster lane 'tools' free — admitted"` for a lane
`dos lease-lane acquire` refused the same second. The runbook's own summary:
*"A FREE from a checker that cannot see the store is not evidence of a free lane."*

Fifty-two lines exist to explain that a GO is not an acquirability promise, and the
diagnosis that was being pursued to fix it (forcing all three readers to agree) was
**retracted by its author** — the dead pid was the expected steady state, and the
fix would have reintroduced double-booking.

An agent calling the convenient tool picks held lanes every time. Correctness of
the admission stack is irrelevant if the tool handed to the agent lies about it.

**Apply:** one authority. Either the MCP tool reads the WAL, or the runbook stops
pointing agents at it. Do not ship a second opinion that disagrees with the first
and is the one with the nicer name.

### Lesson 3 — Ship the launcher before the arbiter

At a 4% launch rate, every improvement to lane selection, dedup, arbitration,
capacity planning, and conservation is an improvement to a queue that mostly never
drains. `docs/ISSUE-PICKUP-10X-PROGRAM.md` makes the same argument from the
product side — a perfect backlog cannot be picked up if nothing picks it up.

The repo's own tail-wag audit reached the identical conclusion on the job side:
context-window economics, not planning, was driving the orchestration topology, and
the recommendation was to collapse the prose layer into a wrapper around the
mechanism that already worked.

**Apply:** the launch rate is the KPI. It is currently on no scorecard. Put
`launched / planned` next to every loop's commit count, and treat a number under
~80% as a launch bug, not a selection bug.

### Lesson 4 — Audit the payload-to-gate ratio per tick

Three independent measurements agree that the gate is the product: 60% of tick
wall time is preflight; 62,000 tokens are provisioned against 20,028 measured; and
54,000 tokens of `SKILL.md` must be read before the first pick is made. Add the
job-side control-plane figure and the pattern is complete — a dispatch tick there
costs 1.66x a naive tick in mandatory read context before any work happens.

**Apply:** emit `preflight_ms / tick_total_ms` and `baseline_tokens_reserved /
baseline_tokens_measured` as standing metrics. When the first exceeds ~0.3 the
ladder is inverted, and the correct response is to delete a gate, not to tune it.

### Lesson 4b — A guard that cannot tell "peer is working" from "snapshot is incomplete" blocks everything

Found while landing the Lesson-5 work, and the purest instance of this page's
thesis found so far.

`fak commit` refused with `PEER_WIP_COLLISION` on
`internal/heavinessscore/*` — files no live session was editing. The tree holds
**46 `refs/fak/wip/*` checkpoints** and the attribution scan reads every one. Two
defects compound:

1. **Partial-tree snapshots are read as complete trees.** The offending refs hold
   8,069 and 333 files against **23,708** on HEAD, so the ~19,000 files absent from
   each are attributed to that peer. Fifteen `reap-*` refs from 2026-09-25
   collectively claim essentially the whole repository.
2. **Historical presence is not classified as historical presence.** The other six
   offending refs date from 2026-09-10 and 2026-09-25, and their diff against HEAD
   is `616 insertions(+)` with **zero deletions** — `internal/heavinessscore/` did
   not exist yet in those snapshots. `internal/safecommit/peerwip.go` already
   defines `OwnershipHistoricalPresence` for exactly this case, and the guard
   fires the collision anyway.

**Verified before overriding:** no live process owned any of the six session ids;
the thirteen refs from 2026-09-26 onward are full trees carrying only the
clean-base version; and no ref held a *conflicting* hunk. Committed with
`FAK_PEER_WIP_GUARD=warn` — a first-class mode of the guard
(`internal/safecommit/peerwip.go`), not an override: the attribution scan still
runs and still records, it just stops refusing. `off` was not used, and no ref was
deleted.

**Why this is Lesson 1 again.** The DOA outage was invisible because a liveness
count cannot register failure. This is the same failure in the safety path: a
collision check that fires on "absent from an old snapshot" cannot register the
case it exists for, and a guard that blocks every commit is indistinguishable from
a guard protecting you. **A gate whose false-positive rate approaches 100% is not a
gate** — it is an outage shaped like a safety mechanism.

**Apply:** expire `refs/fak/wip/*` by age (a three-day-old checkpoint from a dead
session is not a peer), and skip any ref in the attribution scan whose tree is not
a superset of the paths it is asked about. Until both land, a `PEER_WIP_COLLISION`
on a file with no live owner should be re-derived by hand before it is believed.

### Lesson 5 — The simple path needs a scorecard or it can never win

This is the lesson with the longest tail, because it protects the thing that is
working.

The job repo relabelled its naive fanout era as `"human-driven"` in
`docs/_dispatch_loops/pick_success_series.json`, annotated: *"Volume in this era
reflects HUMAN effort, so it is not a fair apples-to-apples baseline for the
autonomous era."* The simple path's output was reclassified as human labor and
excluded from the comparison, while the loop's output was scored against a target.
A path with no number cannot be credited when it wins, and cannot be defended when
it loses.

fak's case is sharper, and it cuts the opposite way from the one I first assumed.
`docs/OPERATOR-HEAVINESS.md` was last regenerated 2026-08-20 and still reads
**Grade D**, 79 front-door flags, `refusal_vocab_size` **0.000**. Re-running
`fak operator heaviness --markdown` on 2026-09-28 returns:

| | stale doc (2026-08-20) | live meter (2026-09-28) |
|---|---|---|
| Grade | D | **B** |
| pressure | 233 | **139** |
| front-door flags | 79 | **10** |
| `front_door_flag_burden` | 0.017 (score 2) | **1.000 (score 100)** |
| refusal reasons | 86 | 149 |
| `refusal_vocab_size` | 0.000 (score 0) | 0.000 (score 0) |

A real consolidation landed — the front door was deliberately tiered from 79 flags
to 10 — and it moved the grade from D to B. **Nobody re-ran the meter, so the
repo's own record of its own improvement is wrong.** In the same window the refusal
vocabulary grew 86 → 149 (+73%) and that one KPI read 0.000 before and after.

That is the sharper form of the lesson. A KPI that reads 0.000 carries no
information, so re-running it cannot help: it needs re-basing or replacing, not
refreshing. And the refresh that would have cost one command was not run for 39
days — in a repo that has 73 skills and 133 verbs generating meters for exactly
this purpose.

**Apply:** score the naive path, even though it is a 45-character prompt. A 45-char
arm with a number is the cheapest control group in the system, and it is the only
thing standing between "the machinery is working" and "the machinery is running."

## The counterweight — what not to copy

The naive path is faster partly because it is less safe, and that part should not
be ported.

- **No structural double-ship guard.** The job repo's verified field lie rate on
  `SHIPPED` claims is 2/28 ≈ 7%, and its design ledger records *"a lying dispatch
  driver whose SHIPPED/BLOCKED tokens were overridden by git-ancestry 5/5 times."*
- **fak's clean `lease_churn` record is partly an artifact of the same problem.**
  All windows read `scavenges: 0, double_spawns: 0, tag_collisions: 0` — but with a
  4% launch rate, few workers get far enough to collide. Zero double-spawns on a
  fleet that mostly does not launch is not a safety record.
- **Human judgment does not scale past one operator.** 4.8% of the job repo's 187
  recorded fanout runs shipped zero picks. That is a real failure rate, and a model
  handed a fixed list of ten inherits it.

So the conclusion is not "delete the machinery." It is narrower and more useful:
**the machinery should be paid per shipped commit rather than per iteration, and
it should be measured with the same ruler as the path that already works.**

## Do this first

| # | Action | State |
|---|---|---|
| 1 | **Repoint the headline.** `fak dispatch status` led with live workers. It now leads with commits landed in window, and separates "nothing ran" from "everything ran and nothing landed". | **SHIPPED** — `cmd/fak/dispatch_status_throughput.go` |
| 2 | **Publish `launched / planned`.** It was already the field distinguishing the 24 from the 1; it was just never aggregated. | **SHIPPED** — `cmd/fak/dispatch_status_goallaunch.go` |
| 3 | **Emit `preflight_ms / tick_total_ms` per tick** and alarm above 300‰. | **SHIPPED** — `dispatchTickLoopMetrics` |
| 4 | **Fix or un-recommend the MCP `dos_arbitrate` path.** It fails open; the runbook says so; the earlier fix was retracted. | **OPEN** — cross-repo, in dos-kernel |
| 5 | **Re-base or replace `refusal_vocab_size`.** | **SHIPPED** — `9c2ca42c7`; 86 → 22.2, 149 → 7.8 |
| 6 | **Stand up the naive arm as a measured control.** | **SHIPPED** — `f2e9caded`; `fak naive-control compare` |
| 7 | **Expire `refs/fak/wip/*` by age and skip non-superset refs in the collision scan.** | **OPEN** — blocking every `fak commit` (Lesson 4b) |
| 8 | **Ratchet `AGENTS.md` bytes.** It was the largest context slice and the only one of the three context-budget floors with no gate. | **SHIPPED** — `b81fa0d7a`, floor pinned at 52,301 |

### What shipped looks like

```
$ fak dispatch status
LANDING [landing] 78 commit(s) in 24h0m0s | live workers 0 (secondary) | 78 commit(s) landed in the last 24h0m0s
dispatch status — 0 live worker(s)
runs-dir: .dispatch-runs
lanes held: (none)
goal launches: 1/25 launched (4.0%) [low]; 24 of 25 planned runs never launched — suspect the LAUNCHER (preflight, argv-vs-binary skew, or lease refusal), not the selector
```

During the 2026-07-28 outage the same card would now have read
`LANDING [outage_suspect] 0 commit(s) in 24h0m0s ... 382 spawn(s) finished,
nothing landed, 0 still live — this is an outage, not an idle fleet` instead of
`0 live worker(s)`.

## See also

- `internal/dispatchdoa` — the dead-on-arrival detector built from the 2026-07-28
  incident; the package doc is the primary source for symptom 2.
- `internal/naivecontrol` + `fak naive-control compare` — the two arms on one ruler
  (item 6). Today it prints `UNKNOWN` for every field and `INCONCLUSIVE`; that is the
  correct answer until the arms have run.
- `internal/agentsindex` byte floor — the `AGENTS.md` ratchet (item 8).
- `docs/agentic-issue-dispatch.md` — the runbook; §3 is the GO-is-not-a-promise
  clarification behind lesson 2.
- `docs/OPERATOR-HEAVINESS.md` — the scorecard behind lesson 5, regenerated
  2026-09-28 after 39 days of staleness.
- `docs/ISSUE-PICKUP-10X-PROGRAM.md` — the product-side argument for lesson 3.
- `docs/STEERABILITY-SCORECARD.md` — the sibling growth-invariant index; graded A
  where operator-heaviness grades D, which is itself worth noticing.
