---
title: "Why loops matter to end users"
description: "Product-facing explanation of normal loops, harness loops, fleet loops, and super loops: how they reduce repeated token spend, raise fleet efficiency, and make autonomous work measurable instead of self-reported."
---

# Why loops matter to end users

A loop is a way to spend agent budget only where the next step is still worth it.
It is not just a developer workflow. The same pattern applies to support queues,
data QA, compliance review, evaluations, content operations, and any workflow where
an agent can keep working after each result is checked.

The useful shape has four parts:

| Part | Plain meaning | User-facing value |
|---|---|---|
| Selector | Pick the next ready unit from evidence | Work advances on the highest-value item, not on whichever task a model remembers |
| Executor | Run one bounded unit | A failed or expensive task cannot consume the whole queue |
| Witness | Check the result outside the worker's self-report | The system can close, retry, or stop from facts |
| Budget / stop policy | Decide when another turn, worker, or retry is still allowed | Cost, latency, and risk stay bounded |

That shape is why loops are product value rather than internal machinery.

## Token and Cost Value

Long agent runs waste tokens in three common ways: they resend stable context, they
repeat failed tool attempts, and they keep exploring after a result is already
known to be blocked. A loop can reduce that waste when each tick records what was
read, what was reused, what was denied, and what was witnessed.

For `fak`, keep the savings accounts separate:

| Account | What it means | Claim status |
|---|---|---|
| Provider-observed | A provider prompt-cache or billing surface reported a cache benefit | Observed, not owned by `fak` |
| fak-authored | `fak` removed, reused, or served token-equivalent work itself | Witnessed only when the repo has a proof for that mechanism |
| Modeled | A formula projects savings from measured geometry | Useful for planning, not a billing claim |
| Not yet | The loop has no strong witness for that saving | Must stay unlabeled as a win |

The user benefit is not a generic "loops save tokens" slogan. The honest claim is:
loops make repeated work measurable, then let the runtime stop or reuse only where
the witness says reuse is valid.

Examples:

- A support loop can avoid re-reading the same policy pack for every ticket when
  the policy input is stable and cacheable.
- A data-cleanup loop can stop after a deterministic validator shows the remaining
  rows need human input.
- An eval loop can reuse a benchmark harness and record per-case cost instead of
  re-running the whole suite after every small change.

## Fleet Efficiency Value

A fleet loop is the queue version of the same idea. It does not simply launch more
agents. It asks what can run now without colliding, which seats and rate limits are
free, which workers are already live, and whether previous attempts produced
witnessed progress.

That matters to end users because the useful unit is not worker count. The useful
unit is witnessed throughput per dollar, per minute, and per risk envelope.

Use this mental model:

```text
effective_workers = min(requested_workers, host_cap, seat_cap, lease_cap, ready_work_cap)
net_throughput = effective_workers * success_rate / (1 + retry_rate)
```

Raising `requested_workers` helps only when it is the limiting term. If seats,
leases, or ready work are the limiter, another worker mostly adds contention and
retry spend. A loop makes that visible before launch.

For a customer workflow, fleet efficiency shows up as:

- fewer duplicate attempts on the same case;
- fewer workers blocked on the same file, account, ticket, or API quota;
- fewer human interruptions because the close/retry decision is witnessed;
- clearer cost allocation because every tick has a bounded unit and outcome.

## Harness Loop Value

A harness loop wraps one workflow so an agent can repeat it safely:

1. Select one ready item.
2. Load only the context needed for that item.
3. Execute one bounded action.
4. Validate the result outside the model's self-report.
5. Record the outcome, budget, and next admissible action.

That is the general version of the dev loop. The payload can be anything with a
checkable outcome.

| Workflow | Selector | Executor | Witness | Stop policy |
|---|---|---|---|---|
| Support operations | Next high-priority ticket with complete account context | Draft response, search KB, or propose an allowed account action | Ticket diff, policy check, account-system receipt | Stop on missing authority, max retries, or human-required class |
| Data ingestion QA | Next failed row batch by severity | Normalize, enrich, or quarantine one batch | Schema validator, row counts, checksum, sampled review | Stop on validator pass or unresolved source ambiguity |
| Evaluation runs | Next stale benchmark or failed case | Run one benchmark slice | Stored result row plus parity/quality gate | Stop on budget breach or score regression |

The important point for adoption is that a loop is not "let the agent keep going."
It is "let the agent keep going only through a selector, a witness, and a budget."

## Super Loop Value

A super loop sits above ordinary loops. It does not do the work itself. It reads the
status of several loops, folds their debt, and points to the worst-first member to
enter next.

That is valuable for users who have many automated workflows. Without a super loop,
each loop looks locally reasonable while the overall system drifts: support gets
fast but QA rots, evals stay green but costs climb, or a retry loop quietly burns
budget on a blocked class. A super loop gives the operator one read-first control
surface over the whole portfolio.

In `fak` terms, `fak superloop walk <intent>` is the safe orientation move. It is an
interior node: it reads, ranks, and recommends. Effects happen only when a member
loop is entered through its own admission gates.

That is different from a bulk worker launcher. A launcher is an executor control
surface that may start processes after dispatch admission passes; a super loop is a
selector control surface that should be safe to run first because it mutates nothing
at its own altitude.

For an end user, that means:

- one place to ask "what should the automation do next?";
- one ranking that can put dark, stale, or unmeasured loops above noisy but healthy
  loops;
- one aggregate exit condition for an intent such as "improve support quality" or
  "reduce ingestion cost";
- fewer accidental launches because the read-first step is separate from the
  executor step.

## Adoption Rungs

Adopt loops in this order:

1. Observe one workflow with a ledger: selected item, action, witness, cost, and
   outcome.
2. Add the refusal vocabulary: the closed set of reasons the loop must stop or ask.
3. Automate one bounded tick, still dry-run by default.
4. Allow live execution only after the selector, witness, and budget pass.
5. Add a fleet loop when there are many independent ready units.
6. Add a super loop when there are many loops and the hard question becomes "which
   loop should receive attention next?"

Each rung adds evidence. None requires trusting a worker's final message as proof.

## Where to Look in This Repo

- [`docs/dispatch-loop.md`](dispatch-loop.md) shows the GitHub-issue fleet loop:
  spawn, ship a commit, witness, close, and refresh status.
- [`docs/super-loops.md`](super-loops.md) defines the read-first super loop and its
  five properties.
- [`docs/fak/dojo.md`](fak/dojo.md) shows the token-saving prediction, run,
  measure, score, and recalibrate loop.
- [`docs/cache-value-rollup.md`](cache-value-rollup.md) keeps witnessed kernel reuse
  and observed provider savings in separate accounts.
- [`docs/standards/net-true-value.md`](standards/net-true-value.md) is the standard
  for any efficiency claim this page would let a user make.

Useful readouts:

```powershell
python tools/dispatch_status.py
go run ./cmd/fak dispatch progress --target 50
go run ./cmd/fak loop economics --ledger .fak/loops.jsonl
go run ./cmd/fak superloop walk drain-issues
go run ./cmd/fak superloop walk improve-quality
go run ./cmd/fak vcache status
```

Read these as control surfaces, not as automatic proof of savings. The proof for a
specific claim is the witness behind the row: a provider cache metric, a fak-authored
token-equivalent witness, a closure audit, or an explicit `not yet`.

## The Loop-Economics Readout

`fak loop economics` folds the loop ledger (`.fak/loops.jsonl`) into one readout that
answers "did the loop save tokens, wall time, worker attention, or retries versus the
real baseline?" — and keeps the four claim classes separate so a row is never read as
a stronger claim than its witness supports. It reads only the hash-chained ledger and
writes nothing.

The **witnessed** block is derived purely from the recorded events and the dispatch
progress / worker / cooldown metrics they carry:

| Field | How to read it | Do not read it as |
|---|---|---|
| `baseline_open` / `observed_open` | The open-issue count the loop started from vs. the most recent snapshot | A billing figure |
| `issues_closed_by_loop` | Peak cumulative issues the loop witnessed-closed | Issues closed *because of* one mechanism |
| `close_rate` | `closed / (closed + observed_open)` — the share of the loop's lifetime worklist witnessed-closed | A guarantee about future issues |
| `retry_rate` | `refused / fires` — the share of ticks the governor braked (cooldown, cap, collision) instead of re-spawning | A per-issue failure rate |
| `duplicate_attempts_avoided` | Refused admissions — spawns the governor declined, i.e. worker+token spend a naive always-spawn loop would have burned | A measured token count |
| `effective_workers` / `worker_cap` | Peak concurrent workers observed vs. the configured cap | Sustained parallelism |
| `wall_time_seconds` | Summed measured run durations (`run_durations`), else the event-window span (`window_span`) | Human attention time |

The **token-equivalent saved** accounts are kept strictly separate and each defaults to
`not_yet` — the readout never invents a saving the ledger cannot prove:

- **provider cache** — a provider prompt-cache/billing benefit. *Observed, not owned by
  fak.* Stays `not_yet` until you fold a provider figure with `--provider-cache-tokens`.
- **fak-authored** — token-equivalent work fak removed/reused/served itself. Witnessed
  only with a proof for that mechanism (`--fak-authored-tokens`); `not_yet` otherwise.
- **modeled** — a projection, `duplicate_attempts_avoided * tokens_per_avoided`, for
  planning only. Never a billing claim; `not_yet` until you supply
  `--modeled-tokens-per-avoided` with your own per-attempt estimate.

The `not_yet` array names every field with no witness in the given inputs, so a real `0`
(or an un-owned account) is never mistaken for a proven win. A rate whose denominator was
zero (`close_rate`, `retry_rate`) is reported as `not_yet` rather than a misleading `0%`.

What this readout does **not** yet auto-fold — the honest next step — is the external
token witnesses: the gateway cache-saved token metrics and the closure-audit/benchmark
scorecards are folded only when an operator passes them as explicit figures. Auto-folding
those sources (so provider/fak-authored populate without an operator input) is the
follow-on to this leaf.

## The Product Claim

Loops make agent work cheaper and safer only when they are evidence-bound. The
customer-facing value is the combination:

- token and cost controls from measured reuse and bounded retries;
- fleet efficiency from admission, leases, caps, and witnessed close rates;
- harness reliability from selector/executor/witness/stop separation;
- super-loop steering from a read-first view over many loops.

If any of those witnesses are missing, the honest product state is `not yet`.
