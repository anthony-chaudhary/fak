---
title: "The managed-context glossary and product contract"
description: "The public glossary for fak's managed-context promise: assumption, resident view, pinned objective, budget envelope, reset transaction, context query, memory promotion, cache state, and relay vocabulary — grounded in shipped mechanisms where available and explicitly marked when planned."
---

# The managed-context glossary and product contract

*Program: [managed-context](https://github.com/anthony-chaudhary/fak/issues/1570). Issue:
[#1571](https://github.com/anthony-chaudhary/fak/issues/1571).*

fak already runs strong context mechanisms — a planner, a durability gate, a wall-clock
budget, a reset ledger. What was missing was a plain-language contract that lets a user
stop managing those assumptions in their head. This page is that contract: the core terms,
each defined by what fak automatically manages, what it will ask the user about instead of
silently assuming, and what remains fully user-controlled. Entries grounded in shipped code
name the real Go type or function behind them. The relay vocabulary is called out as a
planned, data-only extension until the #1860 relay rungs consume it.

**How to read each entry:**

- **What fak manages** — the mechanism runs without you; you do not have to track it.
- **What fak asks about** — the cases where fak will not guess; it surfaces a decision or a
  query instead of silently assuming.
- **What stays user-controlled** — the knobs and overrides that are yours, always.

---

## 1. Assumption

**Definition.** A fact-like item a context plan may rely on, tagged with *where it came
from* and *how confident fak is in it* — not a bare string, a scored, provenanced claim.

**Shipped mechanism:** `ctxplan.Assumption` (`internal/ctxplan/assumption.go`). Every
assumption carries a closed provenance class (`AssumptionSource`: `user_stated`,
`witnessed`, `inferred`, `stale`, `unknown`) and a confidence in `[0,1]`. `AssessAssumptions`
scores each one against a policy (`DefaultAssumptionPolicy`: 0.65 minimum confidence, 0.80
for inferred claims) and resolves it to one of three closed actions — `use`, `query`, or
`refresh` — never a fourth, silent "assume it's fine."

- **What fak manages:** scoring every assumption's confidence against its source class and
  deciding, deterministically, whether it's safe to act on. A `user_stated` fact with
  reasonable confidence is used without friction.
- **What fak asks about:** anything that scores below the bar — an `inferred` guess under
  0.80 confidence, or a `stale`/`unknown`-sourced fact — comes back as `query` or `refresh`,
  not a silent pass. `AssumptionReport.EffectSafe` is `false` until every assumption clears,
  so an effectful action cannot proceed on an unresolved guess.
- **What stays user-controlled:** the confidence thresholds (`AssumptionPolicy`) and every
  `user_stated` fact's content — fak never overrides something you told it directly.

---

## 2. Resident view

**Definition.** The bounded, O(1) set of context spans that are actually materialized into
the current turn — as opposed to the full, unbounded history sitting behind it.

**Shipped mechanism:** `ctxplan` (`internal/ctxplan/doc.go`, `plan.go`, `layout.go`). fak
treats the current turn's context as a re-planned *view* over a lossless store, not a
linearly growing transcript and not a lossy summary. A `Layout` controls four independently
tunable areas (`Base`, `Current`, `Recent`, `Deep`), each with its own span count and
precision (`exact`, `planned`, or `pointer`); `Optimize` chooses which spans populate the
view under a hard token `Budget`, and every span it elides is `Faithful` — kept recoverable
by a content-address handle, never destroyed (`faithful.go`).

- **What fak manages:** which spans are resident right now, replanned every turn against the
  budget, and how the old-vs-new tradeoff is struck — recency, durability, and predicted
  relevance all factor into `Optimize`'s selection, so you don't hand-curate the window.
- **What fak asks about:** nothing silently disappears. An elided span is cold, not gone; a
  mid-turn reference to something not currently resident produces one closed-vocabulary page
  fault outcome (`ctxplan.PageFaultOutcome`) — served back in or explicitly refused — never a
  silent omission.
- **What stays user-controlled:** the `Layout` shape itself (how much of each area to keep,
  and at what precision) and the token `Budget` cap. You can widen recent history or force a
  span resident via a pin; fak never makes the view unbounded on your behalf.

---

## 3. Pinned objective

**Definition.** The user's standing, active goal, represented as a stable, addressable span
that must survive a hidden context reset, replan, or session migration unchanged.

**Shipped mechanism:** `ctxplan.ObjectivePin` (`internal/ctxplan/objective.go`), wired into
session-reset carryover by `sessionreset.PinObjective` / `RepinObjective` /
`CarryObjective` (`internal/sessionreset/objective.go`). A pin's `PinID` is assigned once and
never regenerated; its `Digest` is a content address over the pin's identity fields, so
"the objective was preserved" is a checkable equality (`ReconcileObjective`), not a narrative
claim. Reconciliation returns a closed `ObjectiveOutcome` a host must branch on.

- **What fak manages:** carrying your stated objective forward across every hidden restart —
  a budget-exhaustion reset, a replan, a session migration — and detecting, structurally,
  whether the carried objective drifted from what you actually asked for.
- **What fak asks about:** any reconcile outcome other than `Preserved`/`Established`
  surfaces as a typed decision the host must act on, not a silently-accepted rewrite. A pin
  is never inferred from thin air — it derives from your own stated text (the first durable
  user line), never a model's paraphrase standing in for your intent.
- **What stays user-controlled:** the objective's text. fak reconciles identity and content
  drift; it does not decide what your goal *should* be, and it never substitutes its own
  guess at your objective for what you stated.

---

## 4. Budget envelope

**Definition.** The set of orthogonal limits — turns, output tokens, context tokens,
clarification queries, and wall-clock time — that together bound how much a session may
spend before it must pause, reset, or stop.

**Shipped mechanism:** `session.Budget` (`internal/session/session.go`) covers the token
axes (`TurnsLeft`, `TokensLeft`, `ContextTokensLeft`/`Cap`,
`ClarificationQueriesLeft`/`Cap`); `session.TimeBudget` (`internal/session/timebudget.go`,
issue #1584) adds the orthogonal wall-clock axis, because real elapsed time keeps ticking
whether or not the model is spending tokens. Both are reset-aware: a hidden context reset
carries the remaining budget and elapsed time forward onto the fresh trace
(`session.ResetBudgetRearm`, `Table.RecontinueAt`) instead of quietly resetting to zero.

- **What fak manages:** debiting every axis as the session runs, warning before
  exhaustion (the pre-exhaustion warning measures consumed share against the stamped cap),
  and re-arming the *same* remaining budget onto a fresh trace after a hidden reset — so a
  reset never silently grants (or loses) unearned runway.
- **What fak asks about:** exhaustion on any axis is a distinct, named reason
  (`ReasonBudgetTokens`/`Turns`/`Context`, `ReasonTimeBudgetExhausted`) surfaced to the
  caller — never a bare timeout with no explanation of which envelope ran out.
- **What stays user-controlled:** every limit is a value you set (or leave unconfigured,
  meaning unbounded — `session.Unbounded`/`TimeUnbounded`). fak never lowers a configured
  budget on its own; it only spends against the number you gave it.

---

## 5. Reset transaction

**Definition.** The replayable audit row for one context-budget reset — what carried over
from the old trace to the new one, what did not, and why.

**Shipped mechanism:** `session.ResetTransaction` (`internal/session/reset_transaction.go`).
It records the old/new trace ids, the fresh `ResetBudgetRearm`, a `SeedDigest` over the
carryover seed, the `Contributors` that built it, and — critically — every
`ResetOmittedSpan`: a payload-free pointer (role + content digest + reason) to transcript
bytes that did **not** land verbatim in the fresh session. Nothing is dropped without a
named reason attached to a checkable digest.

- **What fak manages:** performing the reset — building the carryover seed, re-arming the
  budget envelope, and writing the full transaction record — without your intervention when
  a budget axis exhausts.
- **What fak asks about:** nothing about what got omitted is left implicit. Every span that
  didn't survive the reset verbatim is named in `OmittedSpans` with a reason and a digest, so
  "what did the reset drop" is answerable from the record itself, not from re-reading the old
  transcript.
- **What stays user-controlled:** you can inspect the transaction after the fact (it's a
  plain JSON-serializable record — `ResetTransactionSchema`), and the objective pin plus any
  explicit pins in your `Layout` are the mechanism you use to keep specific facts from ever
  being reset-eligible in the first place.

---

## 6. Context query

**Definition.** A model-authored (or user-authored) request that states *what the next
turns will need* and gets back a typed, inspectable view — the agent-callable front door
onto the same planner that runs the resident view.

**Shipped mechanism:** `ctxplan.PlanQuery` / `Plan` / `PlanView`
(`internal/ctxplan/query.go`). A caller states `Intents` (predicted reference strings for
the upcoming horizon) plus optional `Budget`, `Horizon`, `Pins`, and `Weights`; the query
lowers into the same `Forecast` the host path uses and runs the identical `PlanCells` —
the facade adds no second planner and no divergence. An under-specified query (empty
intents) still plans against sane defaults rather than erroring.

- **What fak manages:** resolving an unset budget from the query's own shape
  (`RecommendBudgetForForecast`) so a caller that only states intent still gets a sensibly
  sized working set, and enforcing every invariant the host path already enforces — a query
  can never pin a sealed or tombstoned span into view, and it can never exceed budget.
- **What fak asks about:** the returned `PlanView` is fully inspectable — selected spans,
  elided spans, cost used, and an `Explain` trace — so "why did I get this view" is always
  answerable, not a black box.
- **What stays user-controlled:** what you (or the agent acting on your behalf) declare as
  intent, and any explicit budget/horizon/pin override. fak plans against your stated intent;
  it does not substitute its own prediction when you've supplied one.

---

## 7. Memory promotion

**Definition.** The write-time decision that moves a fact from live, ephemeral context into
durable, cross-session memory — gated so that "it's 3pm" never becomes a standing belief
and "I prefer afternoons" does.

**Shipped mechanism:** the durability classifier in `internal/ctxmmu` stamps a closed
durability class (`turn`, `session`, `bounded`, `durable`) on every admitted value, defaulting
to the shortest-lived class for anything unclassified (see
[`CONTEXT-IS-NOT-MEMORY.md`](CONTEXT-IS-NOT-MEMORY.md)). `memq.PromotionRecord`
(`internal/memq/promotion.go`) is the audit trail for any write that actually crosses into
durable storage: it names the source span, the durability class earned, and a closed
consent class — `ConsentExplicit` (you asked for it), `ConsentInferred` (the system judged
it durable-worthy), or `ConsentUnknown` (fails closed to the weakest claim). A `turn`-class
fact never mints a promotion record at all — it was never a candidate.

- **What fak manages:** refusing promotion by default. An observation has to *earn* the
  `durable` class before it's written to long-term memory; the default for anything
  unclassified is to expire, not persist.
- **What fak asks about:** every promotion that actually happens is tagged with *why* —
  `ConsentExplicit` vs `ConsentInferred` — so you can distinguish "I told it to remember
  this" from "it inferred this was worth keeping," rather than one undifferentiated memory
  store.
- **What stays user-controlled:** anything you state explicitly earns `ConsentExplicit` and
  the `durable` class outright. Inferred promotions are the ones fak is conservative about —
  the false-negative (forgetting something durable) is cheap and recoverable by re-asking;
  the false-positive (promoting an ephemeral remark) is the expensive direction fak biases
  against.

---

## 8. Cache state

**Definition.** The warmth belief fak holds about a provider's prompt-prefix cache — whether
a chained request is likely to land on a shard that already has your prefix warm — and the
guards that keep fak from trusting that belief past what it's worth.

**Shipped mechanism:** `internal/vcachegov` and `internal/vcachecal`. `vcachegov.AffinityKey`
biases chained requests onto one shard so a warm prefix is read by the request that warmed
it; the affinity router also watches for a correlated warmth collapse (an autoscale reshard
invalidating a whole warm set at once) and throttles its own warming bursts so it never
triggers that reshard itself. `vcachecal.Concentration` measures whether a workload is
concentrated enough (a Zipf exponent `s > 1`) for warming to pay off at all, and flags a flat
workload as structurally defeated rather than warming a tail that will never pay back.

- **What fak manages:** routing chained requests for affinity, detecting a correlated
  cache-warmth collapse, capping its own warming bursts, and measuring up front whether your
  workload's shape makes cache warming worth doing.
- **What fak asks about:** nothing here is ever load-bearing for correctness — cache
  warmth is a cost/latency *belief*, confirmed by provider telemetry, never an authority to
  omit context. A miss costs latency; it never silently drops a fact from your session.
- **What stays user-controlled:** correctness never depends on cache state (a provider's
  load balancer can always route elsewhere), so nothing about your session's behavior changes
  based on whether the cache is warm or cold — only its cost and latency do.

---

## 9. Relay vocabulary

**Definition.** The planned perpetual-session vocabulary for running one long goal as a
sequence of bounded context windows without asking the model to summarize itself.

**Status:** planned/data-only contract for epic #1860. The concept spine is
[`docs/notes/CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md`](notes/CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md);
the closed reason rows live in
[`docs/notes/RELAY-REASON-VOCABULARY-2026-07-01.md`](notes/RELAY-REASON-VOCABULARY-2026-07-01.md);
the baton wire shape lives in
[`docs/notes/RELAY-BATON-SCHEMA-2026-07-01.md`](notes/RELAY-BATON-SCHEMA-2026-07-01.md).
No shipped driver consumes these terms yet.

- **Relay.** A single goal executed as an ordered sequence of bounded legs; the operator
  starts the relay, and the `done_when` check decides when the goal is over.
- **Leg.** One bounded session window in a relay, corresponding to one `Generation`, that
  ends only at a safe point and never mid-tool-call or mid-write.
- **Baton.** The pointer-only handoff a closing leg writes for its successor: objective pin,
  re-verifiable progress cursor, next action, open questions, durable artifact pointers, and
  tombstone, with no transcript bytes and no `claimed` field.
- **Tombstone.** The closing leg's typed exit record: a closed relay reason token plus the
  commit anchor and short display note explaining why the leg ended.
- **Externalize gate.** The fail-closed pre-rotation check that every load-bearing fact from
  the leg is already durable as a commit, ledger row, memory slug, issue, or other witnessed
  pointer before a baton may be trusted for rotation.
- **Safe point.** A quiescent boundary where ending the leg cannot corrupt work: no in-flight
  tool call, no half-written commit, a green or explicitly parked tree, and a next action that
  fits in one baton line.
- **Flat-context invariant.** The relay property that peak resident context is bounded by
  the per-leg ceiling, independent of the goal's total duration or number of rotations.

- **What fak manages:** not yet shipped. The planned driver will detect arm/fire conditions,
  write and read batons, re-verify cursors, preserve lease continuity, and rotate only at a
  safe point.
- **What fak asks about:** anything the externalize gate cannot prove durable, any stale
  baton cursor, and any objective mismatch. Those become typed relay outcomes, not prose.
- **What stays user-controlled:** the objective text, `done_when` predicate, relay budget
  envelope, and operator choice to launch, park, resume, or stop the relay.

---

## What this adds up to

A reader should now be able to state, plainly: fak automatically manages *what's resident in
context right now* (assumption scoring, the resident view, budget accounting, reset
bookkeeping) and *what's structurally safe to skip* (elided-but-recoverable spans, cache
warmth as a non-authoritative bias). It asks the user (or defers to an explicit decision)
whenever a fact's confidence, durability, or promotion status is ambiguous — never silently
guessing on the expensive-to-get-wrong side. And it leaves fully user-controlled: every
explicit statement of fact, every budget number, every layout/pin override, and the
objective itself. The relay section extends the vocabulary for the planned perpetual-session
mode while keeping its unshipped status explicit.

## See also

- [Managed-context continuous usage semantics](managed-context-continuous-usage.md) — the
  user-facing contract for preserving continuity across hidden context resets without
  making the user manage the context window manually.
- [Perpetual sessions](notes/CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md) — the relay concept
  spine that defines the long-running, flat-context mode.
- [Relay baton schema](notes/RELAY-BATON-SCHEMA-2026-07-01.md) — field-level contract for
  `fak.relay.baton.v1`.
- [Relay reason vocabulary](notes/RELAY-REASON-VOCABULARY-2026-07-01.md) — closed
  relay/tombstone reason rows.
- [Context is not memory](CONTEXT-IS-NOT-MEMORY.md) — the durability axis behind memory
  promotion, in full.
- [`internal/ctxplan` package doc](https://github.com/anthony-chaudhary/fak/blob/main/internal/ctxplan/doc.go) — the resident-view / context-query
  planner in full, including the Postgres-planner correspondence this glossary summarizes.
- [The four layers of agent memory](MEMORY-LAYERS-EXPLAINER.md) — the spatial/trust axis this
  glossary's memory-promotion entry complements.
- [Glossary: core vocabulary, shared memory, and before/during words](glossary.md) — the
  disambiguation glossary for overloaded internal terms; this page is the product-facing
  contract, that page is the internal-vocabulary house style.
