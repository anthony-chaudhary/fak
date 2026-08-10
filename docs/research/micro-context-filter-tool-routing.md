---
title: "Micro-window routing across filters and tool calls"
description: "A typed control-loop contract for deciding whether to stop, filter, open semantic context, call a tool, widen, or escalate."
status: proposed
last_reviewed: 2026-08-09
---

# Micro-window routing across filters and tool calls

## Verdict

The micro-context pattern is more general than splitting a large payload into small prompts. A
bounded window can be a **decision point over the next computation**:

```text
partition + typed facts + receipts + remaining budget
  -> bounded selector window
  -> propose one allowlisted next stage
  -> kernel admission
  -> deterministic filter | semantic window | read tool | widen | stop | escalate
  -> typed fact/receipt
  -> repeat or fold
```

This applies equally to filters and tool calls. The model may propose which stage has the highest
expected decision value; it does not mint the authority to execute that stage. The kernel owns the
catalog, capabilities, dependencies, budgets, idempotency, scheduling, cancellation, and terminal
contract.

This is a design hypothesis, not yet a net-true performance claim. #6105 tracks the executable
comparison; #6033 tracks the broader tuned-baseline falsification.

## Three different micro-window seams

"A micro-window for filters and tools" can mean three different executions. Keeping them separate
prevents a useful optimization from becoming an authority leak:

| Seam | Bounded input | Output | Authority | Typical cache key |
|---|---|---|---|---|
| **Control window** | record metadata, prior typed facts, candidate descriptors, remaining budget | propose `skip`, `run filter`, `call read tool`, `widen`, `stop`, or `escalate` | proposal only; the kernel admits a catalogued stage | policy + candidate-set + fact-summary digest |
| **Filter window** | one record, field, or value plus one semantic predicate | typed match / class / abstention with evidence span | no tools and no new capabilities | filter/rubric version + normalized value digest |
| **Tool-result window** | one bounded read receipt or result slice plus the unresolved question | typed fact / relevance / abstention with citation | cannot mint another call or convert a read into an effect | tool/version + canonical args + result digest + rubric |

The windows may use the same model endpoint, but they are not interchangeable calls. A control-window
cache hit cannot stand in for a filter result; a tool-result interpretation cannot authorize another
tool; and a filter's semantic answer cannot certify that a global task is complete. Separate receipts
also make it possible to batch filter work, cache stable read results, and invalidate routing policy
without discarding underlying facts.

A concrete path is:

```text
issue record
  -> exact metadata filters
  -> control window: is a semantic residual worth opening?
  -> filter window: classify one title/body/value
  -> control window: does freshness uncertainty justify a read tool?
  -> kernel admits a canonical read call
  -> tool-result window: extract one relevant bounded fact
  -> typed fold
```

At every arrow the controller may avoid unopened work. That is the economic promise. It is not a
claim that model decisions should precede cheap deterministic filters.

## Why route stages instead of running everything?

A conventional pipeline often runs a fixed cascade:

```text
parse -> exact filter -> retrieval -> rerank -> model -> tools -> reducer
```

That is excellent when stage utility and ordering are stable. It becomes wasteful when work is
heterogeneous: an exact field may settle one record, another may need relation context, a third may
need a read-back from an external system, and most may already be irrelevant. A tiny selector can
avoid expensive downstream work, but only if the work avoided exceeds selector overhead and the
quality loss from wrong skips stays inside the answer contract.

The useful abstraction is therefore not "one model call per item." It is a typed, bounded
computation graph whose nodes may be deterministic code, model windows, or tools.

## Stage descriptor

Each candidate stage should be declared data, not free-form model output:

```text
stage_id/version
kind: deterministic_filter | semantic_window | read_tool | widen | stop | escalate
input/output schemas
applicability predicate
dependency fact/receipt types
required capability and resource scope
estimated token/tool/latency cost
cache and freshness policy
retry, timeout, and cancellation policy
maximum fan-out and depth
```

The selector sees only the subset of descriptors admissible for the current state. It returns a
stage ID plus typed arguments constrained by schema and, optionally, a confidence or expected
utility estimate. The executor independently revalidates everything.

## Decision rule

A useful conceptual objective is value of information (VOI):

```text
run stage s when
  E[downstream decision loss before s] - E[loss after s]
  > token cost + tool cost + latency penalty + scheduler cost + risk penalty
```

A production controller need not calculate calibrated dollars for every record. It can use tuned
thresholds, a deterministic decision table, a learned router, or a hybrid. But the ledger should
retain enough information to estimate **oracle regret** afterward:

- wrong skip: a skipped stage would have changed the correct answer;
- unnecessary run: the stage did not affect any admissible downstream decision;
- bad ordering: the same quality was available with less work in another order;
- budget regret: early cheap work consumed budget needed by a later decisive stage;
- escalation regret: the controller widened or escalated when a cheaper route sufficed.

A selector that saves calls but increases false negatives is not a win.

### Sufficiency is task-shaped, not "N records arrived"

Cancellation is sound only when a predicate over independently witnessed facts proves unfinished work
cannot change the answer contract:

- **Existence** (`is any open issue a release blocker?`) may stop after one witnessed positive.
- **Top-k** may stop only when every unopened partition has a structural upper bound below the current
  witnessed kth score; merely having k candidates is insufficient.
- **Exhaustive count/list** normally must inspect every eligible partition, unless an exact index or
  partition-completeness receipt proves that the remainder is empty.
- **Contradiction/duplicate clustering** generally needs group-aware coverage; per-record confidence
  alone cannot prove that no cross-record relation remains.

The routing model must not author and approve its own stop condition. The kernel evaluates a declared
predicate over receipts, or an independent grader/read-back does so. Otherwise "sufficient" is just
self-certified early exit. S8l's fixed-count stop proves that cancellation changes the latency/quality
point, not that the stopping rule is sound.

## Authority and safety invariants

1. **Choice is not authority.** Selector output names a catalog entry; capability checks occur at
   dispatch.
2. **Typed dependencies precede execution.** A tool requiring `repo_id` cannot infer it from prose
   or bypass a missing receipt.
3. **Read and effect stages differ.** Read-only candidates may be raced under a bounded policy;
   effects are never speculative by default and remain approval/idempotency/read-back bound.
4. **Cancellation is phase typed.** Before dispatch means no call; during a read means an incomplete
   read; after an effect dispatch means unknown until independent read-back, never rollback.
5. **Budgets are global and local.** Enforce per-partition and batch limits for depth, calls, model
   tokens, tool spend, wall time, and concurrency.
6. **Stopping is answer typed.** `stop` is valid only when required fields are proven, uncertainty is
   within policy, and unresolved dissent/unknown effects remain visible.
7. **Cache identity includes semantics.** Keys bind stage version, policy version, input digest,
   dependency receipts, model/prompt identity where relevant, and freshness class.
8. **Receipts, not narration, feed the fold.** Tool/effect output is admitted only through typed
   validation and, for effects, independent read-back.

## Scheduling options

### Sequential adaptive routing

Run one selected stage, update state, then decide again. This minimizes wasted work and makes budgets
simple, but adds control-loop latency.

### Queue-aware deadlines and selective read races

A deadline is useful only if timeout releases the scheduler slot; a timed-out stream that still occupies
admission capacity can make the whole task slower. Record cancellation request, transport completion,
slot release, returned usage, and unknown post-cancel billing as separate facts.

For safe read-only candidates, selective hedging may launch a duplicate only when measured tail risk
multiplied by decision value exceeds the duplicate's expected work. Keep the first result satisfying a
fixed quality predicate and cancel the loser. Compare this with both no hedge and universal hedge.
S8l's universal arm opened 18 duplicates for two wins and no wall-time gain, so micro-windows make
hedging possible, but current evidence does not show that hedging helps. Effects remain forbidden.

### Deterministic fast path plus semantic residual

Apply cheap exact predicates to every partition, then route only ambiguous residuals. This will often
be the strongest practical default because deterministic work is cheap, reproducible, and easy to
cache. The micro-window should not replace SQL, search, schemas, or policy engines where those settle
the question.

### Batch compatible stages

Group equal stage/model/prompt/tool descriptors and dispatch them together. Routing granularity and
provider batch granularity are independent: decisions may be per record while execution is batched.

## Steelman alternatives

- **Run all declared filters/tools:** simplest to reason about, easiest to observe, and often best
  when stages are cheap or false-negative cost is very high. Adaptive routing only wins after its
  own inference, telemetry, and misroute costs.
- **Fixed tuned cascade:** deterministic, low variance, straightforward to test, and likely superior
  under a stationary workload. A learned selector earns its complexity only under meaningful input
  heterogeneity or drift.
- **Query planner/rules engine:** for structured data, a cost-based planner can choose indexes,
  joins, and predicates without an LLM. Micro-windows are for semantic ambiguity and open-world
  stage choice, not a replacement for database optimization.
- **Retrieval plus reranking:** mature, batchable, and often sufficient. It may dominate when the
  key uncertainty is relevance rather than multi-stage action choice.
- **One long-context call:** preserves global relationships and avoids controller round trips. It can
  win on dense cross-record reasoning, especially with native prefix caching and predictable tool
  use.
- **Coarse chunk agents:** amortize per-call overhead and preserve local discourse. They trade exact
  invalidation and fine-grained cancellation for fewer scheduler events.
- **Adaptive micro-window routing:** strongest when decisive work is sparse, stage choice varies by
  partition, typed intermediates are reusable, and expensive reads can be avoided or cancelled.

There is no universal winner; workload density, relation depth, provider pricing/cache behavior,
tool latency, and quality constraints determine the boundary.

## Failure modes

- selector cost exceeds avoided work;
- correlated wrong skips silently remove evidence;
- recursively spawned stages cause budget or fan-out explosion;
- stale receipts make a cached decision look current;
- tool latency pins model slots or defeats batching;
- races duplicate billed reads despite cancellation;
- a selector smuggles arguments outside the declared schema;
- stage order changes semantics, not just cost;
- summaries erase dissent, abstention, or provenance;
- effect completion is inferred from worker output rather than read-back;
- optimization overfits a benchmark's stage distribution.

## What to measure

At one explicit quality target, compare adaptive routing with tuned run-all, fixed cascade,
retrieval/reranking, long context, and coarse chunks. Record:

- false positives, false negatives, abstentions, citation validity, and relation-heavy slices;
- stage proposals, admissions/denials, wrong skips, unnecessary runs, and oracle regret;
- model input/output/cached tokens, tool calls, retries, billed cancelled work, and dollars;
- TTFT, critical-path and tail latency, scheduler CPU/allocation, queue delay, and peak concurrency;
- cache hits, invalidations, stale-read detections, and replay behavior;
- budget exhaustion, widen/escalate rates, circuit-breaker trips, and unresolved unknowns.

Report decision regions rather than one average. A routing policy can win on sparse mixed workloads
and lose on homogeneous or dense ones.

## Current evidence boundary

- S8h (`#6105`) observed a controlled crossover: a static filter+tool policy won the exact-heavy
  mixture, while adaptive routing won several residual-heavy mixtures under the fixture's common
  utility. Its costs are calibrated units, not provider billing.
- S8l (`#6160`) showed that bounded records permit typed cancellation and partial folds. Count-based
  early stop reduced wall time and returned prompt tokens but also reduced exact quality; fixed
  deadlines retained slots; universal hedging was wasteful. Cancelled-but-billed usage remains unknown.
- The live alternatives still miss the strict quality floor, and tool-need labels await stabilization
  in `#6140`. There is no quality-qualified net-true winner yet.
- `#6167` is the executable next falsification: task-specific witnessed sufficiency, queue-aware slot
  release, and selective filter/tool/window admission against tuned static alternatives.

## Falsification sequence

1. Controlled 1,000-partition workload with an oracle that knows which stages can change each answer.
2. Tune run-all and fixed-cascade alternatives rather than preserving naive ordering.
3. Verify authority, dependency, budget, replay, and cancellation invariants under adversarial output.
4. Sweep ambiguity, relation density, tool latency, cache reuse, and update locality.
5. Repeat on a leakage-controlled non-fixture corpus with an independent grader.
6. Run live endpoints and real tools; separate API evidence from controlled-kernel evidence.
7. Apply the net-true standard, including selector and scheduler overhead and quality misses.

Until those steps are witnessed, the defensible claim is architectural: micro-windows can provide a
bounded decision seam over heterogeneous filter/tool stages. Whether that seam is cheaper or faster
than a tuned static alternative is workload dependent and remains to be measured.
