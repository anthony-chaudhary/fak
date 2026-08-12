# Context as a variable: named, lazy, queryable context

**Audience:** people designing or evaluating fak's managed-context runtime who
need one precise model for names, paging, filtering, caching, and call results.

**Status:** this page defines the intended architecture and vocabulary. The
lower paging and context-plan seams exist today; the first-class binding/query
API is tracked by the linked issues below and is not claimed shipped yet.

## The short answer

“Context as a variable” means a large input lives outside the model's immediate
prompt under a stable name. The model operates on that name through bounded
queries, and only selected results enter its context window.

```text
outside the model:
    tickets -> the complete 500,000-token ticket snapshot

model-visible instruction:
    "The source is available as `tickets`."

model action:
    count(group(filter(tickets, status == "failed"), owner))

model-visible result:
    {"alice": 19, "bob": 7}
```

The variable is not neural-network hidden state. It is closer to a read-only
database relation, lazy collection, array, or file handle in a governed tool
runtime.

The precise fak phrase is:

> **queryable context over addressable sources**

“Context as a variable” is the programming-interface metaphor.

## Is this just lazy loading?

Lazy loading is one important part, but not the whole contract. A useful
variable has a name and an immutable target; demand may then fetch existing
bytes, compute a new view, or admit selected bytes into the prompt. Those are
separate transitions:

```text
bind -> fetch/page-in -> materialize/query -> admit
```

| Stage | What happens | New semantic bytes? |
|---|---|---:|
| **Bind** | A task-scoped name resolves to an immutable source or view identity. | No |
| **Fetch / page-in** | Existing bytes behind a ref become resident. | No |
| **Materialize / query** | A filter, projection, or aggregation computes a derived view. | Yes |
| **Admit** | Selected resident bytes enter the model-visible prompt view. | No new source fact |
| **Call** | A tool/model recipe executes and creates a result snapshot. | Yes; it may also have effects |
| **Refresh** | An explicit new execution creates another snapshot and binding revision. | Yes |

Creating or listing a binding must be inert. It performs no source-byte read,
query, call, or prompt admission. A later demand names the work that is needed.
A single `loaded` boolean cannot represent this lifecycle safely.

## The objects and their names

Human names are for ergonomics. Machine identities are for correctness.

```text
human alias:       tickets
qualified binding: task-42@7:tickets
target kind:       call_snapshot
immutable target:  sha256:S1
```

A resolver turns the qualified binding into an exact target kind, immutable
identity, policy, and taint record before any fetch, query, cache lookup, call,
or admission occurs.

The vocabulary is:

| Term | Meaning |
|---|---|
| **Addressable context** | Source bytes or a view have a stable immutable/versioned identity. |
| **Binding name / alias** | Human-facing task-scoped name such as `tickets`. |
| **Qualified binding** | Workspace revision plus alias, such as `task-42@7:tickets`. |
| **Context workspace** | Task-scoped namespace and versioned binding manifest. |
| **Queryable context** | Bounded operators compute observations over addressed sources. |
| **Derived context view** | Immutable, provenance-stamped result of a query. |
| **Context program** | Policy and plan choosing queries, budgets, residency, and admission. |
| **Resident view** | Bytes currently available in a serving tier. |
| **Admitted view** | Bytes actually serialized into the model-visible context. |

An unresolved name such as `latest` is UI sugar only. It must resolve once to
an exact revision before an operation starts. Cache keys, provenance, and
replay records never use a bare alias as the source identity.

## Filtering creates another value

A filter does not mutate its input:

```text
tickets                        -> source snapshot hash S1
failed = filter(tickets, P1)   -> derived view hash V1
```

`failed` can itself be named, paged out, restored, filtered again, aggregated,
shared, admitted, or evicted. Its lineage records at least the source snapshot,
canonical query plan, operator version, policy/taint identity, output bounds,
and output hash.

This functional/immutable rule makes filters replayable and makes caching
correct. A cache keyed only by the alias `failed` would be wrong because an
alias can be rebound in a later workspace revision.

## There is more than one cache

“Cached” is not a sufficient operational explanation. Each layer reuses a
different thing:

| Cache | Reuses | Does not prove |
|---|---|---|
| **Page/blob cache** | Exact source or view bytes already stored. | That a query result is semantically current. |
| **Plan cache** | A selection/planning decision. | That derived result bytes exist. |
| **Derived-view cache** | Exact `source snapshot + query semantics -> view` result. | That an alias still resolves to the same source. |
| **Call/idempotency cache** | A witnessed execution outcome for a canonical recipe and scope. | That paging a result should execute a call. |
| **Provider KV/prefix cache** | Model computation for a stable serialized prefix. | Source truth, tool-result freshness, or query correctness. |

A derived-view key therefore includes every semantic input that could change
the answer: all source snapshot hashes, canonical query plan, operator/runtime
schema, policy and taint identity, and output-limit contract.

The explain surface must report `page_hit`, `plan_hit`, `derived_view_hit`,
`call_outcome_reuse`, or `provider_kv_hit`, not a generic `cache_hit`.

## Page-in/page-out is the lower layer

This architecture reuses fak's existing MMU and plan mechanisms:

- [`internal/ctxmmu/mmu.go`](../../internal/ctxmmu/mmu.go) replaces large or
  held tool-result bodies with governed CAS-backed refs and can restore the
  same bytes under policy.
- [`internal/ctxplan/pagefault.go`](../../internal/ctxplan/pagefault.go) models
  demand page-fault requests and bounded resolution.
- [`internal/ctxplan/materialize.go`](../../internal/ctxplan/materialize.go)
  materializes bounded resident views.
- [`internal/ctxplan/query.go`](../../internal/ctxplan/query.go) has a bounded
  demand-query selection seam.
- [`internal/ctxplan/plancache.go`](../../internal/ctxplan/plancache.go) caches
  planning decisions, not derived result bytes.

A page-in retrieves bytes that already exist behind an identity. A query
creates a new immutable identity. Prompt admission is yet another decision.
The named-binding layer joins these seams; it does not introduce a competing
paging system.

This is similar to the agent virtual filesystem described in
[Agent virtual filesystem](agent-virtual-filesystem.md): both give stable names
to content-addressed objects and fault bytes on demand. Queryable context adds
typed relational/dataflow operations and model-visible admission semantics.
It is also complementary to [Addressable KV cache](addressable-kv-cache.md):
that page addresses model-computation spans, while this page addresses source
and derived data objects.

## Calls produce snapshots; reads do not execute calls

A tool or model call can produce a large result that becomes addressable
context. But a binding must pin a result snapshot, not hide a live call:

```text
call recipe R1 --explicit execution--> call snapshot C1 -> result ref S1

task-42@7:tickets -------------------------------------> S1
```

Reading, paging, filtering, querying, or admitting `tickets` uses S1 and
executes zero calls. An explicit refresh adjudicates R1 again and creates a new
snapshot and binding revision:

```text
refresh R1 -> call snapshot C2 -> result ref S2 -> task-42@8:tickets
```

Only a recipe structurally proven read-only may be deferred until first demand.
Effectful calls must never hide behind lazy dereference. A page fault can fetch
an existing result blob; it cannot rerun the call that produced it.

Keep three identities separate:

1. **Call recipe:** tool/model, canonical arguments, capability/policy identity,
   caller scope, and freshness contract.
2. **Call snapshot:** one witnessed execution and immutable result ref/hash,
   timestamp, taint, and outcome.
3. **Binding revision:** the task-scoped name pinned to that snapshot.

The call/idempotency layer may reuse a witnessed outcome during an explicit
refresh. That still differs from the page cache and derived-view cache.

## The full architecture

```text
canonical names and immutable resolution (#6533)
                    |
task-scoped workspace bindings (#6524)
                    |
demand lifecycle: bind -> fetch -> materialize -> admit (#6531)
                    |         |            |
                 ctxmmu    ctxplan      query/view (#6518)
                                              |
                                  derived-view cache (#6525)
                                              |
                                  explain and replay (#6528)
                    |
explicit call recipe -> snapshot -> refresh (#6532)
                    |
exact aggregation counterfactual (#6526)
                    |
optional governed helper-model interpretation (#6527)
```

The implementation order should be:

1. Freeze canonical target kinds and resolution rules in
   [#6533](https://github.com/anthony-chaudhary/fak/issues/6533).
2. Ship the smallest real query/derived-view spine in
   [#6518](https://github.com/anthony-chaudhary/fak/issues/6518).
3. Add workspaces and inert bindings in
   [#6524](https://github.com/anthony-chaudhary/fak/issues/6524), then connect
   them to fetch/materialize/admit in
   [#6531](https://github.com/anthony-chaudhary/fak/issues/6531).
4. Add semantic view caching in
   [#6525](https://github.com/anthony-chaudhary/fak/issues/6525) and derivation
   explain/replay in [#6528](https://github.com/anthony-chaudhary/fak/issues/6528).
5. Add immutable call snapshots and explicit refresh in
   [#6532](https://github.com/anthony-chaudhary/fak/issues/6532).
6. Run the exact aggregation counterfactual in
   [#6526](https://github.com/anthony-chaudhary/fak/issues/6526).
7. Only then evaluate helper-model interpretation in
   [#6527](https://github.com/anthony-chaudhary/fak/issues/6527).

All issues are children of managed-context epic
[#1570](https://github.com/anthony-chaudhary/fak/issues/1570).

## What this architecture guarantees—and does not

It should guarantee:

- binding is inert;
- identities are immutable and replayable;
- page faults never execute calls;
- filters create new views rather than mutate sources;
- each cache has a typed identity and outcome;
- refresh is explicit and versioned;
- source, query, view, call, and admission provenance are inspectable;
- byte, work, storage, call, and prompt-token budgets remain separate.

It does not promise unlimited context or free computation. The runtime must
still read or index source bytes; queries consume CPU and storage; admitted
observations consume prompt tokens; helper calls consume model tokens. A model
can also issue a bad query. The benchmark in #6526 therefore requires exact
aggregation answers and the complete cost denominator before any net-true gain
is claimed.

## Related reading

- [Context management](context.md) — the current managed-context operator route.
- [You never manage the context window](you-never-manage-the-context-window.md)
  — the product promise this architecture should eventually fulfill.
- [Context shedding](context-shedding.md) — removing low-value resident turns
  without invalidating stable cache prefixes.
- [Agent virtual filesystem](agent-virtual-filesystem.md) — named,
  content-addressed objects and demand faults.
- [Addressable KV cache](addressable-kv-cache.md) — addressability of model
  computation spans rather than source objects.
- [Context as variable vs addressable context study](../notes/CONTEXT-AS-VARIABLE-RLM-STUDY-2026-08-12.md)
  — pinned Prime RLM source study, candidate analysis, and research trail.
