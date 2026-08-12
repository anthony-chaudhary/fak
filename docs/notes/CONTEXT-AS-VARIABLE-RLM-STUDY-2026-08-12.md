---
title: "Context as variable vs addressable context: Prime RLM study (2026-08-12)"
date: 2026-08-12
status: research-note
source: https://github.com/PrimeIntellect-ai/verifiers
source_revision: a506236ec5a853225a72fc262d82ca1c5e741f6d
license: MIT
---

# Context as variable vs addressable context

## Verdict

**Related, but not the same concept.** Prime Intellect's RLM work makes a large
input *computable*: the root model gets a REPL in which the input is bound as a
variable, then iteratively searches, slices, aggregates, and asks sub-models
about selected observations. fak's existing addressable-context work makes
context *identifiable and governable*: bytes have stable references, can be
paged, screened, pinned, and materialized into bounded resident views while
preserving cache economics.

The clean composition is:

> **addressable source pages -> governed query operators -> derived context
> views -> model-visible resident set**

A memory address answers **where/which bytes?** A variable answers **what can the
program do with them?** A view answers **which bounded derivation is visible
now?** The new learning is the middle query/derivation plane, not a rename of
fak's current page identity.

For public fak vocabulary, prefer **queryable context** for the capability and
**derived context view** for each result. “Context as variable” is a useful RLM
teaching metaphor, but `variable` suggests mutable aliasing and arbitrary code;
those are the wrong default contracts for a content-addressed, default-deny
kernel.

## Source pin and study scope

- Repository: `PrimeIntellect-ai/verifiers`
- Pinned revision: [`a506236ec5a853225a72fc262d82ca1c5e741f6d`](https://github.com/PrimeIntellect-ai/verifiers/tree/a506236ec5a853225a72fc262d82ca1c5e741f6d)
- License at inspection: MIT
- Primary implementation seam:
  [`verifiers/v1/harnesses/rlm/harness.py`](https://github.com/PrimeIntellect-ai/verifiers/blob/a506236ec5a853225a72fc262d82ca1c5e741f6d/verifiers/v1/harnesses/rlm/harness.py)
- Historical implementation seam:
  [`verifiers/legacy/envs/experimental/composable/harnesses/rlm.py`](https://github.com/PrimeIntellect-ai/verifiers/blob/a506236ec5a853225a72fc262d82ca1c5e741f6d/verifiers/legacy/envs/experimental/composable/harnesses/rlm.py)
- Author explanation: [Prime Intellect, “RLMs: Recursive Language Models”](https://www.primeintellect.ai/blog/rlm)
- Benchmark dependency: [Oolong at `0bb7eabe839218fee7fe8d007f41cfc2fd3ae24c`](https://github.com/abertsch72/oolong/tree/0bb7eabe839218fee7fe8d007f41cfc2fd3ae24c), MIT at inspection.

This is a concept-and-seam study, not a benchmark reproduction. No Prime code
is copied. The implementation changed rapidly around the study date, so the
revision above, not `main`, is the authority for source claims.

## What the source actually does

At the studied seam, the RLM harness supplies an interactive execution
workspace around the root-model rollout. The accompanying system prompt tells
the model that the large input is available through an environment variable
rather than requiring the whole input to be reasoned over in one forward pass.
The model can inspect pieces, compute intermediate state, and call an LLM over
selected material before returning a final answer.

The important mechanism is not Python specifically. It is this loop:

1. Keep the large source outside the root model's immediate visible prompt.
2. Give the model operations over a stable source binding.
3. Let it form small observations and intermediate products.
4. Spend model tokens on those products, not indiscriminately on all source
   bytes.
5. Repeat until an answer condition is met.

Oolong supplies tasks where long-input *aggregation* matters, which is why
simple semantic retrieval is an incomplete baseline: a query may need counts,
grouping, joins, ordering, or exhaustive scans rather than the one most similar
chunk.

## Relation to fak's current concepts

| Plane | RLM “context as variable” | Existing fak seam | Assessment |
|---|---|---|---|
| Identity | one environment binding points at the task input | `abi.Ref`, CAS-backed page-out, content hashes, context restore IDs | fak is stronger; identity is durable and governable |
| Access | model-directed REPL inspection | page fault / page-in, pins, resident-set planning | partial overlap; fak controls access but exposes less agent-directed derivation |
| Transformation | arbitrary code can slice, filter, aggregate, and retain intermediates | context-plan/view documents specify bounded selection and materialization | conceptual overlap, missing one small public query algebra |
| Visibility | only printed/tool-returned observations enter the root rollout | bounded resident views and prompt transforms | same direction; fak additionally screens admission |
| Recursion | model may call another LLM over a selected observation | fak routes model/tool calls but does not yet make recursive context analysis the primitive | adjacent, not required for the first spine |
| Governance | container/tool policy bounds execution | default-deny adjudication, quarantine, provenance, budgets | fak should preserve its stronger structural floor |
| Cache economics | source stays out of repeated root prompts, but observations still accrue | stable prefixes, generation cache, planned resident views | fak can make the composition explicit and measurable |

Current local authorities:

- [`internal/ctxmmu/mmu.go`](../../internal/ctxmmu/mmu.go) already turns large or
  quarantined bodies into governed references and refuses unwitnessed byte loss.
- [`CONTEXT-VIEWS-AT-MARGINAL-COST-2026-07-04.md`](CONTEXT-VIEWS-AT-MARGINAL-COST-2026-07-04.md)
  already defines bounded resident views over KV/attention economics.
- [`GENERATION-CACHE-CONTEXT-PROGRAM-MAP-2026-06-30.md`](GENERATION-CACHE-CONTEXT-PROGRAM-MAP-2026-06-30.md)
  already names the context-program control plane.
- [`CONCEPT-AUTOMATIC-CONTEXT-2026-07-01.md`](CONCEPT-AUTOMATIC-CONTEXT-2026-07-01.md)
  requires automatic placement rather than user-managed paging knobs.
- Managed-context epic [#1570](https://github.com/anthony-chaudhary/fak/issues/1570)
  owns the product/runtime contract.

Therefore “context as variable” is **not** a replacement name for addressable
context. It is a useful behavioral layer that addressable context can safely
support.

## Vocabulary map

| Term | Keep? | Precise meaning in fak |
|---|---:|---|
| addressable context | yes | immutable or versioned source bytes have a stable identity/ref |
| context as variable | source term only | model can compute over a named large input without seeing it all at once |
| queryable context | recommended capability name | bounded operators derive observations from governed source refs |
| derived context view | recommended result name | immutable, provenance-stamped output of one query over source refs |
| context program | yes | policy/plan selecting queries, budgets, and materialization over turns |
| context workspace | optional UX term | task-scoped namespace containing source refs and derived views |
| agentic search | broader adjacent term | model chooses iterative searches; often retrieval-oriented and not necessarily exhaustive |
| recursive language model (RLM) | source architecture | root LM delegates computation/inspection and may invoke sub-LMs recursively |
| virtual context / context MMU | yes, lower plane | paging, residency, protection, and restoration mechanics |

“Queryable” is preferable to “programmable” for the first fak spine: it
commits to deterministic, inspectable operators, not general-purpose execution.
“Derived view” is preferable to “variable value”: it carries source lineage and
immutability naturally.

## Candidate borrows, with disconfirming checks

### 1. A bounded query algebra over context refs — survive

- **Source fact:** RLM quality comes from model-chosen inspection and
  transformation of a stable large-input binding.
- **Inferred principle:** addressability becomes more useful when an agent can
  derive small, task-specific observations without materializing the whole
  source into its prompt.
- **fak opportunity:** expose deterministic operations such as byte/record
  slice, literal/regex match, count, group, sort/top-k, and provenance-preserving
  projection over one or more `abi.Ref` sources; return a new immutable derived
  view with source hashes, operator plan, output hash, truncation flag, and
  budget debit.
- **Disconfirming check:** if existing public `ctxview`/`ctxplan` APIs already let
  the agent request these transformations over arbitrary refs with lineage,
  this is PRESENT and no issue should be filed. Repository and issue searches
  on 2026-08-12 found planning/materialization and specialized projections, but
  not this generic bounded query seam. **ABSENT.**

### 2. Arbitrary Python REPL inside fak — reject

- **Source fact:** the studied harness uses a code-execution environment as its
  flexible transformation surface.
- **Inferred principle:** general computation is an expedient way to discover
  useful operators.
- **fak opportunity:** none at the kernel boundary by default.
- **Disconfirming check:** fak's default-deny floor, deterministic replay,
  cross-platform Go binary, and provenance requirements would all be weakened
  by embedding unrestricted Python. Existing governed tools can remain an
  escape hatch outside the first-class context plane. **INAPPLICABLE.**

### 3. Recursive sub-model calls as the first spine — defer

- **Source fact:** RLM can ask an LM to analyze selected observations and feed
  results back to the root loop.
- **Inferred principle:** cheap specialist calls may compress or interpret
  derived evidence better than the root model alone.
- **fak opportunity:** route a derived view to a selected model under an
  explicit budget and preserve source/model/prompt provenance.
- **Disconfirming check:** this adds model-quality and cost variables before the
  deterministic query path is witnessed. A no-model query spine can prove the
  core distinction first. **PARTIAL / follow only after measured need.**

### 4. Aggregation-first evaluation — bind to candidate 1

- **Source fact:** Oolong tests long-context aggregation, where nearest-chunk
  retrieval is insufficient.
- **Inferred principle:** a context-query witness must include an exhaustive
  operation whose answer cannot be obtained from one semantically similar
  chunk.
- **fak opportunity:** the candidate-1 spine should include a synthetic
  many-record count/group task, compare whole-context materialization against
  a derived view, and report source bytes, visible bytes, exact answer, query
  work, and elapsed time without claiming a model-quality win.
- **Disconfirming check:** a grep-only demo would prove access but not the new
  aggregation capability. **ABSENT as a generic witness; bind to the same
  issue, not a separate matrix-first issue.**

## Minimal working spine

**Primary problem:** P1 (context limits), with P2 measurement as a witness.

- **For:** an agent operating over a large, already-addressed tool/result corpus.
- **Problem:** today it can page or materialize context, but cannot ask the
  kernel for a small computed observation such as “count records by status and
  show the three failing IDs.”
- **Today:** materialize broad bytes, rely on a specialized one-off projector,
  or execute an external general-purpose tool.
- **Better because:** one governed query returns an exact, bounded, replayable
  view while source bytes remain addressable and out of the prompt.
- **Witness:** a real `abi.Ref` source containing many records is queried by a
  deterministic group/count+filter plan; the result has exact expected values,
  source/operator/output hashes and truncation status; a captured command shows
  fewer model-visible bytes than whole-source materialization. No model and no
  GPU are needed for this spine.

Safety needed in the spine: operator allowlist, source/output byte and record
limits, deterministic ordering, timeout/work budget, provenance, quarantine
inheritance, and fail-closed handling of malformed plans. Arbitrary eval,
network access, mutation, and recursive model calls are explicitly out.

## Net-true hypothesis (not yet a gain claim)

The candidate should be kept only if, on an aggregation task, it preserves the
exact answer while reducing model-visible bytes versus the real alternative of
materializing the whole already-addressed source. Query CPU time, index/build
cost, stored derived bytes, and any extra model calls belong in the denominator.
Until that paired witness exists, this note claims a conceptual gap, **not** a
performance or quality win.

## Registration

This study is registered by its dated note and the surviving candidate filed as [#6518](https://github.com/anthony-chaudhary/fak/issues/6518). `INDEX.md` was intentionally not edited in this
shared checkout because it already contained peer-owned uncommitted changes;
the issue links this note as the durable reverse route.



## Concrete meaning: what is “context as a variable”? (clarification 2026-08-12)

It means the long input is **data in an execution environment, referenced by a
name, rather than text that must all be visible inside every model forward
pass**.

A simplified interaction looks like this:

```text
execution environment:
    context -> <the complete 500,000-token source>   # outside the LM prompt

model-visible turn:
    "The source is available as `context`. Use bounded operations to answer."

model action:
    count(group(filter(context, status == "failed"), owner))

next model-visible observation:
    {"alice": 19, "bob": 7}                         # only this result enters
```

The name is not magical and it is not a neural-network variable. It is closer
to a **read-only database relation, file handle, array, or lazy collection** in
a tool runtime. The LM writes a small program/query against it. The runtime
executes that operation over the complete source and returns a bounded
observation. The LM can repeat this loop, keep intermediate derived values, or
ask another model to interpret one selected value.

That changes the scaling shape: the model no longer needs all source tokens in
its attention window at once, although the external runtime still must read or
index the source and the returned observations still consume context. It also
does not guarantee correctness: the model can write a bad query, omit needed
records, or accumulate too many observations. This is why fak should expose a
small typed query algebra, provenance, limits, and exact aggregation witnesses
rather than unrestricted eval.

In fak terms:

```text
"context"                  = optional workspace binding (#6524)
abi.Ref / content hash      = addressable source identity (existing)
filter/group/count plan     = queryable-context operation (#6518)
derived immutable result   = derived context view (#6518)
reused result              = materialized/memoized view (#6525)
why/how/replay              = derivation explanation (#6528)
quality/cost counterfactual = aggregation evaluation (#6526)
helper-model interpretation = governed later recursion (#6527)
```

So **“context as a variable” is the programming-interface metaphor**;
**“queryable context over addressable sources” is the more precise fak system
contract**.

## Filed follow-on graph

The surviving opportunities are now tracked rather than deferred in prose:

- [#6518](https://github.com/anthony-chaudhary/fak/issues/6518) — minimal query
  algebra and derived-view spine.
- [#6524](https://github.com/anthony-chaudhary/fak/issues/6524) — immutable
  task-scoped names/context workspace.
- [#6525](https://github.com/anthony-chaudhary/fak/issues/6525) — derived-view
  memoization by full semantic identity.
- [#6526](https://github.com/anthony-chaudhary/fak/issues/6526) — exact
  aggregation comparison with whole-source and tuned retrieval.
- [#6528](https://github.com/anthony-chaudhary/fak/issues/6528) — derivation
  explain and replay.
- [#6527](https://github.com/anthony-chaudhary/fak/issues/6527) — governed
  helper-model interpretation after deterministic evaluation.

Dependency order:
`#6518 -> {#6524, #6525, #6526, #6528} -> #6527`.

## Is this “just lazy load”? Page/cache/filter/call clarification (2026-08-12)

**Partly, but “lazy load” names only one transition.** “Context as a variable”
combines an ergonomic **binding** with a demand-driven dataflow. A correct
implementation must not collapse the following stages:

| Stage | Input -> output | Does it create new semantic bytes? | Typical cache/identity |
|---|---|---:|---|
| bind | name -> immutable source/view identity | no | workspace manifest / resolver |
| page/fetch | nonresident ref -> the same source bytes resident | no | CAS/blob/page cache |
| filter/query | source ref + canonical plan -> derived view | yes | derived/materialized-view cache |
| admit | resident source/view -> model-visible prompt view | no semantic derivation, but serialization may change | resident/prompt plan; provider KV cache is downstream |
| call | recipe -> call-result snapshot | yes, and may have effects | idempotency/call-outcome cache, then result blob/page cache |
| refresh | old call snapshot -> new explicitly requested snapshot | yes | never an implicit page fault |

A binding should therefore be **lazy/inert by default**: creating or listing it
performs no source read, query, call, or prompt admission. A later demand names
which operation is required. Metadata demand may need no bytes. Source demand
may page in existing bytes. Query demand may page source bytes and compute a
new immutable derived view. Prompt demand separately admits selected bytes.

### Relation to fak's existing MMU and context-plan machinery

This is intentionally a layer over existing mechanisms, not a parallel paging
system:

- [`internal/ctxmmu/mmu.go`](../../internal/ctxmmu/mmu.go) already replaces
  large/held tool-result bodies with CAS-backed refs and can page the same bytes
  back under policy. That is source storage/residency.
- [`internal/ctxplan/pagefault.go`](../../internal/ctxplan/pagefault.go) already
  models demand page-fault requests and bounded resolution.
- [`internal/ctxplan/materialize.go`](../../internal/ctxplan/materialize.go)
  already materializes bounded resident views.
- [`internal/ctxplan/query.go`](../../internal/ctxplan/query.go) already has a
  bounded demand-query selection seam.
- [`internal/ctxplan/plancache.go`](../../internal/ctxplan/plancache.go) caches
  planning decisions, which is different from caching derived result bytes.

The missing first-class contract is to connect a human/task-scoped name to
those fetch/materialize/admit stages with exact identities, independent
budgets, and observable reasons. [#6531](https://github.com/anthony-chaudhary/fak/issues/6531)
tracks that integration.

### Filtering and caching apply to variables—but at the right identity

A filter does not mutate `tickets`. It creates a new immutable view, for
example:

```text
tickets                       -> source snapshot hash S1
failed = filter(tickets, ...) -> view hash V1, lineage (S1, plan P1)
```

`failed` may itself be named, paged out, queried again, admitted, shared, or
evicted. Its cache key must include the complete source snapshot(s), canonical
plan/operator version, policy/taint identity, and output bounds. Caching only by
the alias `failed` is incorrect because aliases can be rebound. This is the
materialized-view contract in [#6525](https://github.com/anthony-chaudhary/fak/issues/6525).

Several caches remain deliberately distinct:

| Cache | Reuses | Must not be confused with |
|---|---|---|
| page/blob cache | exact existing source or view bytes | query-result correctness |
| plan cache | selection/planning decision | materialized result bytes |
| derived-view cache | source snapshot + query semantics -> immutable view | alias name or provider KV |
| call/idempotency cache | witnessed execution outcome for a canonical call recipe/scope | paging an existing result |
| provider KV/prefix cache | model computation for stable serialized prefix | source truth or tool-result cache |

### Relation to calls

A tool/model call can produce a large result that becomes addressable context,
but dereferencing a variable must **not silently reissue the call**. The safe
model is:

```text
call recipe R1 --explicit execution--> call snapshot C1 -> result ref S1
tickets@rev7 -------------------------------------------> S1
```

Reading, paging, filtering, or admitting `tickets@rev7` uses S1 and executes
zero calls. An explicit refresh adjudicates R1 again, possibly reuses a
witnessed idempotent outcome, and creates C2/S2 plus a new binding revision.
Only a structurally proven read-only recipe may be deferred until first demand;
effectful calls can never hide behind lazy dereference. This is tracked in
[#6532](https://github.com/anthony-chaudhary/fak/issues/6532).

### First-class naming model

Human aliases and machine identities serve different purposes:

```text
human alias:       tickets
qualified binding task-42@7:tickets
kind:              call_snapshot
immutable target:  sha256:S1
resolved record:   workspace rev + alias + kind + S1 + policy/taint
```

Every operation resolves the alias to an exact target before fetch/query/call,
and provenance/cache keys use that target—not `tickets` or an unresolved
`latest`. [#6533](https://github.com/anthony-chaudhary/fak/issues/6533) tracks
the shared naming and conformance contract.

The resulting architecture is:

```text
#6533 names/identity
       |
#6524 workspace bindings
       |
#6531 demand lifecycle: bind -> fetch -> materialize -> admit
       |                 |          |
       |              ctxmmu     #6518 query
       |                            |
       |                         #6525 view cache
       |                            |
       +------------------------- #6528 explain/replay
       |
#6532 explicit call snapshot/refresh (never implicit on read)
       |
#6526 exact aggregation counterfactual -> #6527 optional helper model
```
