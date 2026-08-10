---
title: "Micro-context operators for general large-input LLM work"
description: "Research note and falsifiable design for bounded, cached, cancellable micro-context maps, filters, tools, and folds."
status: proposed
last_reviewed: 2026-08-09
---

# Micro-context operators: a general large-input pattern

## Verdict

The strongest version of the micro-context idea is larger than "many cheap agents." It is a
**general execution pattern for large inputs whose units can be judged or transformed with
bounded local context**:

```text
large input
  -> deterministic structural partition
  -> cheap deterministic prefilter (when available)
  -> bounded micro-context map / select / tool step
  -> typed, cacheable intermediate records
  -> hierarchical witnessed fold
  -> stop when the answer contract is satisfied
```

A GitHub response containing 1,000 issues is cheap to fetch and expensive to place in one
model window. The proposed operator gives one issue—or, when justified, one field or one
small related group—to each logical context. Those contexts can classify relevance, choose
which deterministic or semantic filter should run next, invoke a read-only tool, or emit a
small typed fact. The kernel batches compatible model turns, caches pure stages, enforces
budgets and capabilities, cancels work that can no longer affect the answer, and folds only
validated records. The final model sees the compact evidence set, not the raw thousand-item
payload.

This is **not** a claim that every large prompt should become 1,000 agents. It is a proposal
for a reusable operator algebra whose planner may choose zero model calls, a few chunk calls,
one long-context call, retrieval/SQL, or a micro-context fabric according to measured cost,
quality, dependence, and risk.

## Why this extends the existing program

The current [micro-context fabric](micro-context-fabrics.md) proves that many bounded logical
contexts can share an immutable base and execute over bounded physical slots. That is the
runtime substrate. This note adds the missing **large-input semantics**:

- how raw input becomes independently addressable records;
- how a micro-window chooses or executes filters;
- how tool calls become typed stages rather than arbitrary side effects;
- how partial results roll up without recreating a giant context at the reducer;
- how caching, cancellation, provenance, and failure affect correctness;
- when *not* to use the pattern.

```text
execution fabric: how thousands of logical contexts run cheaply
operator contract: what each context receives, emits, may do, and how results compose
```

Neither layer alone solves the general large-input problem.

## Concrete example: triage 1,000 GitHub issues

Question: "Which open issues are release blockers caused by the new authentication path,
and what are the three recurring causes?"

1. **Fetch once.** `gh issue list --limit 1000 --json ...` produces an immutable source
   artifact with a content hash.
2. **Partition structurally.** One record per issue, preserving number, selected fields, and
   source ordinal. Do not ask a model to rediscover JSON boundaries.
3. **Run free filters first.** State, milestone, labels, timestamps, exact path mentions, and
   duplicate IDs are deterministic predicates.
4. **Route uncertain records.** A tiny selector context returns a typed decision such as
   `exclude`, `run(auth-relevance)`, `run(duplicate-linker)`, `fetch-comments`, or `escalate`.
   It does not receive authority to execute arbitrary tools.
5. **Execute bounded stages.** A relevance context sees one issue plus the stable rubric.
   A separate tool stage may fetch comments for only the uncertain survivors. A field-level
   context is allowed only when the question is field-local; relationship questions receive
   an explicit neighborhood or group.
6. **Emit facts, not prose.** Each stage writes a schema-versioned record carrying source ID,
   source hash, operator version, decision, confidence/reason code, evidence spans, cost, and
   error status.
7. **Fold hierarchically.** Deterministic reducers deduplicate and count first; small model
   reducers cluster only the surviving evidence; the final reducer receives bounded summaries
   plus citations to source issue IDs.
8. **Stop safely.** Cancel unopened records only when a declared stopping rule proves they
   cannot change the requested top-k, threshold, exhaustive count, or confidence interval.
9. **Read back.** Sample negatives, inspect all escalations, and independently verify final
   issue IDs against the fetched artifact.

The same shape applies to logs, traces, rows, documents, code symbols, API catalogs, support
cases, alerts, experiment outputs, and tool inventories. The partition key and reducer laws
change; the execution fabric does not.

## Operator algebra

| Operator | Input | Output | Default implementation |
|---|---|---|---|
| `partition` | immutable source artifact | ordered records with stable IDs | deterministic parser |
| `prefilter` | one record | keep/exclude/route | deterministic code/query |
| `select` | one record + allowed stage catalog | typed next-stage decision | tiny model only for semantic uncertainty |
| `map` | one record/group + rubric | typed fact(s) | micro-context model turn |
| `tool` | typed request + capability | typed observation/effect receipt | kernel-mediated tool call |
| `reduce` | bounded typed facts | typed aggregate | deterministic when laws permit; otherwise model-assisted |
| `verify` | aggregate + sampled/source records | pass/fail/findings | independent read-back |
| `control` | live metrics + answer contract | continue/cancel/retry/escalate | deterministic policy |

Two compositions matter:

- **Control plane:** micro-windows decide *which* filter, tool, or specialist stage should run.
  Decisions are constrained to a declared catalog and are cacheable when pure.
- **Data plane:** micro-windows actually run semantic filters, transformations, or tool calls.
  Tool execution remains capability-checked, idempotent where possible, and journaled.

A context may play both roles only when the combined contract remains bounded and auditable.
Separating them is safer: a selector cannot smuggle arguments into an unrestricted shell, and
an executor cannot silently rewrite its own routing policy.

## Granularity: record, field, group, or adaptive split

"One issue per context" is a useful default, not a law.

- **Field-level** windows are cheapest for local extraction, but destroy interactions between
  title, body, labels, and comments.
- **Record-level** windows preserve local coherence and are the default for issue triage.
- **Group-level** windows are required for duplicates, chronology, contradictions, graph edges,
  and comparative ranking.
- **Adaptive split** starts with a cheap record pass, then expands only uncertain records or
  joins candidate neighbors. This is often the strongest design.

The descriptor needs `unit_id`, `unit_kind`, `source_hash`, `neighborhood`, and
`required_relations`, not merely a text slice. If the answer depends on a relation absent from
the unit, the scheduler must widen the unit or abstain; more parallelism cannot recover erased
information.

## Folding without rebuilding the original context

Naive map-reduce moves the context problem to the reducer. A valid fold must be bounded at
every level and preserve evidence needed by the final answer.

- Counts, sets, min/max, histograms, and keyed joins can be associative and deterministic.
- Top-k is safe only with stable scores/tie-breaking and proof that pruned candidates cannot
  re-enter.
- Semantic clusters and summaries are lossy and order-sensitive, so retain exemplars, source
  IDs, dissent/outliers, and reduction-tree provenance.
- Exhaustive questions cannot use a "good enough" early stop unless the contract changes.

Every fold record should carry:

```text
source_ids + source_hashes + operator/version + decision/facts + evidence_refs
+ cost/latency + error/abstain + reducer_path
```

Intermediate outputs become content-addressed materialized views. A changed issue invalidates
that issue's descendants, not the entire thousand-record run. A changed rubric invalidates the
semantic stage keyed by that rubric version. Cache hits are evidence only when the key includes
all meaning-bearing inputs.

## Filters and tool calls are first-class stages

The same bounded decision seam can choose the next computation across both categories: run a deterministic filter, open a semantic window, call an allowlisted read tool, widen, stop, or escalate. The selector proposes a typed stage; the kernel retains authority, dependencies, budgets, and cancellation semantics. See [Micro-window routing across filters and tool calls](micro-context-filter-tool-routing.md) and executable follow-up [#6105](https://github.com/anthony-chaudhary/fak/issues/6105).

### Filter selection

A model should not replace a cheap predicate. The ladder is:

1. schema projection and exact predicates;
2. indexes, search, SQL, or domain parsers;
3. embeddings/rerankers/classifiers when independently adequate;
4. micro-context semantic judgment for residual ambiguity;
5. larger/group context for relation-dependent cases.

A selector can choose among these stages, but the measured baseline is the best tuned pipeline,
not "send all JSON to one model." The selector must expose confusion: chosen stage,
alternatives, and why deterministic routing was insufficient.

### Tool execution

- **Read-only enrichment** (fetch issue comments, inspect a file, query a row) can fan out with
  per-resource quotas, dedupe, caching, and cancellation.
- **Effects** (edit, close, refund, send, deploy) require the existing capability/resource lease,
  idempotency, journal, and independent read-back seams. Speculative effect execution is off by
  default; cancellation after dispatch does not undo an effect.

The reducer consumes observations or effect receipts—not the worker's claim that a tool
succeeded. Tool outputs may themselves be large and recursively enter the same operator,
subject to depth, spend, and amplification limits.

## Early stopping and adaptive compute

Cheap cancellation is a major upside, but only when tied to answer semantics.

Safe examples include stopping a branch after a deterministic exclusion, deduplicating identical
read calls, proving unopened records cannot enter a top-k, meeting a declared confidence bound,
or tripping a circuit breaker when canaries fail. Unsafe examples include finding three plausible
examples when the user asked for all matches, treating timeout as a negative, or suppressing
low-confidence units without measuring false negatives.

The controller must classify `negative`, `abstain`, `error`, `cancelled`, and `not-run`
separately. Collapsing them makes cheapness look like quality.

## Steelman: why this could be a general-purpose solution

1. Much enterprise input is naturally record-shaped and many judgments are locally bounded.
2. Thousands of units share instructions, schemas, tools, and output contracts, matching prefix
   reuse and batching.
3. Bounded parallelism can turn slow individual streams into useful aggregate throughput.
4. Easy negatives remain in code; only ambiguity buys model reasoning or wider context.
5. Content-addressed pure stages make incremental reruns natural.
6. Failures localize to units, while canaries can halt a correlated rubric failure.
7. Typed facts and source IDs can be more auditable than one opaque long prompt.
8. The same scheduler composes semantic filtering, read enrichment, and mediated effects.

## Steelman: why this might fail or be narrower than claimed

1. Partitioning can erase global dependencies, duplicates, chronology, and distribution facts.
2. A reducer may recreate the original context or silently lose evidence; semantic folds are not
   generally associative.
3. Per-call overhead, output tokens, retries, and rate limits may lose to tuned long context,
   retrieval, SQL, or classifiers.
4. Closed providers may not expose or deliver prefix reuse, and heterogeneous work fragments
   batches.
5. A 1% per-record false-negative rate is about ten misses over 1,000 records; voting does not
   fix systematic rubric bias.
6. Read fan-out can overload dependencies; effect fan-out can race or duplicate.
7. A learned selector is a second fallible model and may add more work than it removes.
8. Small windows can be adversarially brittle, while one shared base is a correlated-failure
   domain.
9. Provenance, invalidation, fairness, and cancellation introduce distributed-systems costs.
10. Exact questions belong in SQL/search/code, and a tuned long-context model may simply win.

The defensible claim is a **broad, selectable operator for decomposable large-input work**, not
a universal replacement for retrieval, long context, or databases.

## Competing baselines and decision rule

Every benchmark compares tuned deterministic query/filter, tuned retrieval/reranking, one tuned
long-context call when the payload fits, provider-native batching, coarse chunk map-reduce, and
the micro-context operator at an identical quality target. Select micro-context only when it is
Pareto-competitive on quality (including false negatives), wall time, spend, critical-path
latency, and auditability. A cheaper run that silently skips hard records is not a win.

## Safety and correctness invariants

1. The source artifact is immutable and hashed; every result names its source unit.
2. Partition coverage is checkable: no silent omission or duplicate before an explicit filter.
3. Selector output is a typed choice from an allowlisted stage catalog.
4. Pure stages are cacheable; effects are never replayed from an ambiguous cache key.
5. Capability, resource lease, idempotency, and read-back bind every effect.
6. Reduction is bounded and records its tree; lossy reducers preserve citations and dissent.
7. Error, abstain, cancel, and negative remain distinct.
8. Stopping rules are declared against the answer contract before execution.
9. Random negatives and boundary cases receive independent read-back.
10. Depth, fan-out, token, tool, time, and dollar budgets cap recursive expansion.

## Minimal working spine

Build a `fak` leaf/operator around an immutable 1,000-issue JSON fixture and a question/rubric:

```text
partition -> deterministic prefilter -> per-issue semantic map
-> typed JSONL facts -> deterministic fold -> cited final artifact
```

The selfcheck proves exact source accounting, bounded physical concurrency, reuse on unchanged
input, precise invalidation after one issue changes, cited final output against an independent
fixture oracle, comparison with tuned long-window/coarse-chunk baselines, and cancellation only
under a declared stopping contract. A fixture-backed worker is `[SIMULATED]`; the first promoted
claim requires a real endpoint and observed usage.

## Experiments that can falsify the thesis

1. Issue-triage quality/cost A/B across exact filters, retrieval, long context, chunks, and
   adaptive micro-contexts.
2. Field vs record vs adaptive-neighborhood granularity sweep.
3. Reducer reorder/fan-in stress; unjustified drift fails the reducer contract.
4. Incremental rerun after 1%, 10%, and 100% source mutation.
5. Deterministic-only vs model-selector vs always-run filter ablation.
6. Read-only tool enrichment under quotas, timeout, dedupe, and cancellation.
7. Duplicate/restart/cancel effect-safety drill with one independently read-back effect.
8. Correlated rubric defect proving canary and negative-sample circuit breaking.
9. Relation-heavy countercorpus identifying where grouping or a non-micro method wins.
10. Controlled-kernel vs API-only run, with cache claims scoped to observed telemetry.

## Related work and limits

- Dean and Ghemawat, [MapReduce](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/)
  motivates explicit map/reduce boundaries, but LLM reducers may be lossy/non-associative.
- Jiang et al., [LLMLingua](https://arxiv.org/abs/2310.05736) makes compression a competing and
  composable baseline; it does not provide scheduling, cancellation, or tool mediation.
- Chen et al., [FrugalGPT](https://arxiv.org/abs/2305.05176) motivates quality/cost cascades; a
  micro-selector is a finer cascade whose orchestration/tool costs must be included.
- Li et al., [Can Long-Context LLMs Subsume Retrieval, RAG, SQL, and More?](https://arxiv.org/abs/2406.13121)
  is a warning against assuming decomposition wins; long context is a required baseline.
- Li et al., [NeedleBench](https://arxiv.org/abs/2407.11963) highlights information-density and
  reasoning-type effects; benchmarks must include multi-record dependencies.

These references motivate comparisons, not a net-true fak claim.

## Proposed issue cohort

Track this under existing epic [#5785](https://github.com/anthony-chaudhary/fak/issues/5785):

1. [#6029](https://github.com/anthony-chaudhary/fak/issues/6029) — runnable 1,000-issue large-input operator spine;
2. [#6030](https://github.com/anthony-chaudhary/fak/issues/6030) — adaptive deterministic/semantic filter selector with confusion telemetry;
3. [#6031](https://github.com/anthony-chaudhary/fak/issues/6031) — read-only tool-enrichment fan-out with quotas and cancellation;
4. [#6032](https://github.com/anthony-chaudhary/fak/issues/6032) — typed hierarchical fold with provenance and incremental invalidation;
5. [#6033](https://github.com/anthony-chaudhary/fak/issues/6033) — falsification benchmark against tuned filters, retrieval, long context, and chunks;
6. [#6034](https://github.com/anthony-chaudhary/fak/issues/6034) — effectful-tool extension only after read-only and fold contracts are witnessed.
