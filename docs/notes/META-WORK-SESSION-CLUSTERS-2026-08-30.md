---
title: "Meta-work from 2,000 recent operator sessions"
description: "A privacy-safe demand map and ranked portfolio of recurring loops and event-triggered automations."
date: 2026-08-30
status: research
---

# Meta-work from 2,000 recent operator sessions

## Verdict

The largest opportunity is not another general-purpose super-loop. It is a small
**meta-work control plane** that turns repeated operator intent into four typed
actions:

1. **dispatch** a witnessed unit of issue work;
2. **observe** live work without prompting it to continue blindly;
3. **react** to a typed event such as a stall, refusal, merge, drift, or release;
4. **garden** recurring evidence into a better harness, skill, policy, or backlog.

The corpus strongly supports this direction. In the latest 2,000 session files,
61.5% of human-candidate sessions mentioned issues or tickets, 50.3% mentioned
workers/fleet/delegation, and 37.9% mentioned audit/review/verification. Normalized
first-request templates were highly repetitive: 74.6% belonged to a template
seen at least twice, 63.8% to one seen at least three times, and 45.1% to one
seen at least five times.

The first implementation priority should be **event-triggered run supervision**,
not semantic auto-dispatch. Repeated `continue`, retry, status, and stop prompts
show that the operator is still acting as the liveness interrupt controller.
That work is mechanical when grounded in authoritative run state; selecting or
changing product intent is not.

## Scope and method

This is a local, privacy-safe study of recent Claude and Codex activity. It does
not commit raw prompts, transcript paths, arguments, outputs, or exemplar text.
The working artifacts were allocated under `_scratch/meta-session-research/` and
are intentionally not source artifacts.

### Corpus boundary

At 2026-08-30 10:31 PT, the local transcript roots contained 5,507 JSONL files.
The study selected the 2,000 most recently modified files:

| Source | Files |
|---|---:|
| Codex | 1,480 |
| Claude | 520 |
| **Total** | **2,000** |

The selected window ran from 2026-08-26 09:29 PT through 2026-08-30 10:31 PT,
roughly 98 hours. A time-window audit scanned 2,072 sessions because files were
still arriving and the shipped audit command selects by time rather than exact
count.

```powershell
fak tree-doctor --scratch-dir meta-session-research

go run ./cmd/fak trajectory audit --since 98h `
  --jsonl _scratch/meta-session-research/audit.jsonl `
  --md _scratch/meta-session-research/audit.md
```

The exact-2,000 projection parsed user-role messages from the selected files,
removed known harness-injected envelopes (repository instructions, goal context,
plugin manifests, approval-assessor histories, subagent notifications, abort
envelopes, and environment records), and retained only aggregate counts and
normalized signatures for this report. It found 1,895 sessions with a
human-candidate request and 2,268 such prompts.

### Limits

- “Human-candidate” means “not a known injected envelope.” A few generated worker
  prompts remain because the operator intentionally launched them. This is useful
  demand evidence, but not a count of individually typed sentences.
- Keyword clusters are multi-label demand signals, not a semantic ground truth.
- `fak traj concepts` clusters **tool-set and transition workflows** with weighted
  Jaccard. It is an excellent drill-down layer, but it explicitly is not semantic
  prompt/task clustering (`docs/workflow-concepts.md`).
- Health probes and harness validation sessions are real operational demand but
  should be separated from product-work demand.
- Counts characterize this four-day operating envelope, not every future week.

## Evidence

### Shipped trajectory audit

`fak trajectory audit --since 98h` reported:

| Measure | Result |
|---|---:|
| Sessions | 2,072 |
| Records | 694,032 |
| Exact usage records | 110,469 |
| Tool calls | 140,503 |
| Tool errors | 12,227 (8.7%) |
| Expected wait timeouts | 648 |
| Mutation-churn events | 75 |
| Schema-drift rows | 31 |
| Input tokens | 927,552,278 |
| Output tokens | 31,826,334 |
| Cache-read tokens | 8,257,444,224 |
| Input:output ratio | 288.6:1 |

Tool results occupied 78.3% of deterministic model-visible bytes. Shell-family
results dominated tool-result volume: `exec_command` represented 75.5% and
`exec` 18.9%. This makes output bounding, result admission, and verified waits
high-leverage meta-work surfaces in addition to prompt automation.

### Demand signals by session

These signals are deliberately multi-label; one implementation request can also
be a worker, audit, performance, and release request.

| Signal | Sessions | Share of 1,895 |
|---|---:|---:|
| Issue/ticket work | 1,165 | 61.5% |
| Worker, worktree, fleet, or dispatch | 953 | 50.3% |
| Audit, review, or verification | 718 | 37.9% |
| Performance or hardware | 328 | 17.3% |
| Research, study, or scouting | 273 | 14.4% |
| Release, deploy, or self-update | 248 | 13.1% |
| Continue, resume, or recovery | 208 | 11.0% |
| Memory, trajectory, harness, or meta-work | 200 | 10.6% |
| Quality/refactor/scorecard | 91 | 4.8% |

The first-request classifier also exposed a large health/probe cohort (399
sessions, 21.1%) and a fleet/dispatch cohort (573, 30.2%). Those are not product
categories; they are evidence that the operator spends substantial attention on
starting and checking machinery.

### Repetition

After replacing issue numbers, SHAs, and paths with placeholders:

| Minimum occurrences | Sessions covered | Share | Template families |
|---|---:|---:|---:|
| 2 | 1,413 | 74.6% | 249 |
| 3 | 1,209 | 63.8% | 147 |
| 5 | 855 | 45.1% | 37 |
| 10 | 708 | 37.4% | 13 |

This is the strongest argument for meta-work: the repeated part is usually the
**protocol envelope** (claim issue, isolate files, dispatch worker, validate,
witness, land), not the issue-specific implementation.

## The meta-work taxonomy

A useful taxonomy needs two axes.

### Axis 1: what judgment is being automated?

| Class | Judgment | Safe default |
|---|---|---|
| Protocol | How to execute known work safely | Automate aggressively |
| Liveness | Whether a known run is alive, progressing, stalled, or done | Automate from witnessed state |
| Selection | Which bounded item should run next | Automate behind declared priorities and collision gates |
| Diagnosis | Why evidence is bad or incomplete | Automate detection; propose fixes |
| Product intent | What should exist, what tradeoff to choose | Keep operator/judge in loop |

### Axis 2: what wakes it up?

| Trigger form | Use when | Example |
|---|---|---|
| Recurring loop | The world changes continuously and batching is efficient | nightly trajectory garden |
| Event trigger | One authoritative transition demands a bounded reaction | run becomes STALLED |
| Threshold trigger | Accumulated debt crosses a meaningful limit | unknown tool errors exceed 5% |
| Completion trigger | A delivered artifact should start the next lifecycle stage | spine commit starts fanout |
| Manual macro | Intent is human but the protocol is repetitive | “work issue 123” packet renderer |

This avoids treating every automation as a forever loop. Most operator attention
in this corpus belongs in event triggers and macros; portfolio gardening belongs
in loops.

## Ranked portfolio

Scores use a 1–5 ordinal scale. **Demand** reflects corpus frequency, **attention**
is likely operator interruption saved, **safety** is how strongly the action can
be grounded in non-self-authored evidence, and **reuse** reflects existing fak
primitives. Total is the sum; ties prefer lower product-intent risk.

| Rank | Meta-work unit | Form | Demand | Attention | Safety | Reuse | Total | Existing spine / missing piece |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | Run-state supervisor | Event | 5 | 5 | 5 | 5 | 20 | Reuse `dos status`, trajectory-control, resume watchdog; add one typed event router |
| 2 | Issue-to-worker protocol compiler | Macro + completion events | 5 | 5 | 4 | 5 | 19 | Reuse issue-owner lifecycle, DOS arbitration, managed worktrees, witness/land |
| 3 | Post-run harvest and reconciliation | Completion event | 5 | 4 | 5 | 5 | 19 | Reuse `wave-harvest`, `dos verify`, commit audit/review |
| 4 | Failure/refusal recovery router | Event + threshold | 4 | 5 | 5 | 4 | 18 | Reuse closed reason vocabulary, `fak recover`, error-family audit |
| 5 | Prompt-template miner | Recurring garden | 4 | 4 | 4 | 4 | 16 | Reuse trajectory audit; missing privacy-safe semantic intent projection |
| 6 | Trajectory-to-harness gardener | Recurring + threshold | 4 | 4 | 4 | 4 | 16 | Reuse trajectory/harness garden; connect measured cluster to one guarded change |
| 7 | Backlog and bottleneck mapper | Recurring + repo events | 4 | 4 | 4 | 4 | 16 | Reuse bottleneck-map, issue-triage, stale-work loop |
| 8 | Spine fanout generator | Completion event | 3 | 4 | 5 | 4 | 16 | Reuse `fak-dev issue fanout`; trigger only after witnessed spine |
| 9 | Evidence freshness and schema-drift watcher | Threshold | 3 | 4 | 5 | 4 | 16 | Reuse trajectory audit baseline/schema rows and claim gates |
| 10 | Research-to-backlog scout | Recurring | 3 | 4 | 3 | 5 | 15 | Reuse scout-loop/study-repo; human chooses adoption |
| 11 | Quality-debt scheduler | Recurring + changed-path event | 3 | 3 | 4 | 5 | 15 | Reuse scorecard family; route worst witnessed debt by lane |
| 12 | Release-readiness assembler | Completion + tag/release event | 3 | 4 | 4 | 4 | 15 | Reuse release skill, plan audit, CI evidence |
| 13 | Tool-output budget governor | Threshold | 4 | 4 | 4 | 2 | 14 | High byte leverage; needs typed output-volume telemetry/action policy |
| 14 | Hardware witness dispatcher | Capability event | 3 | 4 | 4 | 3 | 14 | Reuse hwgate and fleet-node routing; avoid local-hardware dead ends |
| 15 | Documentation drift updater | Changed-source event | 2 | 3 | 4 | 4 | 13 | Reuse freshness/claim/doc scorecards; only mutate derived facts |
| 16 | Product-intent question loop | Recurring, judge-gated | 2 | 3 | 2 | 5 | 12 | Reuse question-loop; never auto-implement an unapproved answer |

## Designs

### 1. Run-state supervisor — first missing spine

**For:** an operator supervising many sessions.  
**Problem:** 11.0% of sessions contain resume/recovery language, while the audit
contains 648 expected wait timeouts. The operator repeatedly decides whether to
wait, continue, stop, or recover.  
**Today:** the operator types a control prompt, often without a current verified
run digest.  
**Better because:** a watcher reacts to `LIVE`, `STALLED`, `DONE`, `REFUSED`, or
`MISSING_HANDLE`, not elapsed conversation time.  
**Witness:** every reaction records the run ID, prior/current witnessed status,
action, reason token, and resulting progress delta.

Trigger table:

| State transition | Action |
|---|---|
| `LIVE` with progress | Do nothing |
| Observation timeout, handle still live | Re-poll same handle |
| `STALLED` with recoverable reason | Run the named recovery plan or propose it |
| `REFUSED` | Route by the closed reason token; never blind retry |
| Terminal success | Start harvest/reconciliation |
| Missing handle | Reconcile ledger and artifacts; never restart solely from silence |

This is mostly composition of existing surfaces, not a new agent framework.

### 2. Issue-to-worker protocol compiler

The recurring worker prompt should become structured data:

```text
issue + objective + lane/tree + allowed effects + proof class + tests + stop gate
```

The compiler renders the harness-specific prompt, acquires the lease, prepares
the sanctioned worktree, launches the worker, and registers the witness gate.
The operator still chooses or approves the issue; the protocol no longer needs
to be rewritten in prose hundreds of times.

Completion events chain the protocol rather than creating one giant loop:

```text
issue claimed -> worker launched -> terminal run -> harvest -> validate -> land
             -> witnessed spine -> deduplicated fanout -> release readiness
```

### 3. Failure/refusal recovery router

The audit found 12,227 tool errors: 1,482 timeout-family, 572 not-found, and 331
permission-family events, with most remaining errors not yet typed. The router
should consume typed errors and DOS reason tokens, then choose among:

- wait/re-poll;
- transform to the canonical safe call;
- execute a declared recovery plan;
- create one instrumentation ticket for unknown error families;
- escalate to the operator.

It must not retry an unknown or terminal refusal merely because a prompt says
“continue.” Unknown classification itself is a threshold-triggered meta-work
item.

### 4. Prompt-template and intent miner

Run nightly or after every 500 new sessions:

1. select a bounded corpus by exact count and time;
2. project only user-authored intent, stripping injected envelopes;
3. separate health/probe, worker-generated, and product-intent cohorts;
4. normalize volatile identifiers;
5. cluster lexical templates deterministically;
6. optionally use a local semantic scorer only to propose neighboring families;
7. rank by frequency × operator attention × evidence strength;
8. emit aggregate signatures and exemplar IDs, never prompt text;
9. reconcile each cluster with existing skills/verbs/issues;
10. propose at most one new automation spine per pass.

The output should connect to `fak traj concepts`: semantic intent says **why the
session started**, while workflow concepts say **how tools were used**. Their
cross-tab identifies mismatches such as one intent producing many workflows or
many intents being forced through one brittle protocol.

### 5. Trajectory-to-harness gardener

Trigger when one of these is true:

- a prompt template repeats at least five times in seven days;
- one intent cluster consumes at least 5% of operator sessions;
- a control/recovery prompt follows the same workflow family at least three
  times;
- tool-error fraction for a workflow concept exceeds its baseline;
- operator intervention rate rises week over week.

The gardener chooses one change class: skill wording, capability discovery,
router policy, tool schema, or new macro. It runs a bounded before/after replay
and keeps only a witnessed improvement. This is the bridge from observation to
self-improvement; raw frequency alone must never authorize a behavior change.

## Existing primitives to compose first

| Need | Existing primitive |
|---|---|
| Cross-harness accounting | `fak trajectory audit` / `trajectory-audit` |
| Similarity and redundant query proposals | `fak traj similar`, `cluster`, `gc` / `trajectory-garden` |
| Tool-workflow concepts | `fak traj concepts` and `docs/workflow-concepts.md` |
| Fleet execution | `super-loop`, `fleet-wave`, DOS dispatch/goal-fleet |
| Run control | `trajectory-control`, `dos status`, resume watchdog |
| Result harvest | `wave-harvest`, `dos-witness-claim`, commit audit/review |
| Research conversion | `scout-loop`, `study-repo`, `field-borrow` |
| Harness improvement | `harness-garden`, guard/dojo RSI scorecards |
| Backlog health | `issue-triage`, `bottleneck-map`, `stale-work-loop` |
| Debt reduction | scorecard family and `score-2x` |
| Spine productization | `spine-fanout`, `fak-dev issue fanout` |

The gap is therefore **routing and joining**, plus a privacy-safe semantic intent
projection. Do not build a second fleet launcher, scheduler, trajectory store,
or issue tracker.

## What should remain human or judge-gated

Do not move these directly into autonomous meta-work:

- choosing product direction from frequency alone;
- accepting a research proposal as a roadmap commitment;
- changing security, release, or destructive-action policy;
- automatically deleting trajectories or memories;
- merging semantic clusters into a single “intent” without exemplar review;
- treating worker self-report as completion evidence;
- turning health probes into product-demand counts;
- using a language model’s cluster label as a policy decision.

Automation should remove protocol repetition and observation toil, not erase
operator intent.

## Suggested implementation sequence

1. **Compose the run-state supervisor** from `dos status`, resume history,
   trajectory-control, and harvest. Prove that live progress causes no action,
   observation timeout causes re-poll, and terminal/refused states route once.
2. **Define a privacy-safe `operator_intent` projection** with source, stable
   session ID, normalized signature hash, coarse cohort, length, and timestamps;
   raw text remains private and optional for local exemplar review.
3. **Add intent clustering/reporting** beside, not inside, workflow concepts.
   Cross-tab intent cluster × workflow concept × error/intervention rate.
4. **Wire the threshold gardener** to create one deduplicated proposal when a
   cluster crosses its declared threshold.
5. **Measure operator-attention outcomes:** manual control prompts per 100 runs,
   interventions per completed issue, unknown-error fraction, false wakeups, and
   median time from terminal worker to witnessed harvest.

## Issue-tracking status

A deduplicated GitHub issue was attempted before this note was written, but the
GitHub API rate limit was exhausted on 2026-08-30. The ready title is:

> `research(meta): turn recent session clusters into a meta-work trigger portfolio`

Reconcile or file it when the rate limit resets. This report remains a research
artifact, not proof that any of the missing spines shipped.

## Bottom line

The corpus says to automate the **boring edges between real decisions**:
launching a known protocol, observing authoritative state, routing typed failure,
harvesting proof, and gardening repeated demand. Keep product selection and
tradeoffs judge-gated. The smallest useful new spine is an event-driven run-state
supervisor; the highest-value recurring loop is a privacy-safe intent-to-harness
garden that runs after enough new evidence accumulates.
