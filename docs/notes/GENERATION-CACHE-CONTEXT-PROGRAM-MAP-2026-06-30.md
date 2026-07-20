---
title: "Generation Map For Cache And Context Programs"
description: "Maps open cache and context GitHub issues to generation labels (now/next/second-next/future) and defines the rules for promoting or demoting those bets."
---

# Generation Map For Cache And Context Programs

This note closes #1647. It maps active cache/context work to generation labels
and defines the promotion rules for context-system bets.

Snapshot source on 2026-06-30:

- `gh issue list --state open --label managed-context`
- `gh issue list --state open --label prompt-caching`
- `gh issue list --state open --search 'cache-default repo:anthony-chaudhary/fak'`

Generation remains orthogonal to priority, shared trunk, and runtime feature
gates. Priority labels still rank urgency. Every generation still lands on
`main` by explicit path. Runtime feature gates still decide whether a cache or
context mechanism is exposed. A `gen/*` label only says which horizon owns the
assumption and what evidence can move it.

## Active Classification

This is a grooming map, not an automatic label mutation. Apply labels during
issue grooming only when the issue body, milestone, and witness can be updated
to match.

| Stream | Active cache/context pool | Why it belongs there | First evidence to collect |
|---|---|---|---|
| `gen/now` | Immediate managed-context product/runtime work: #1571-#1588, #1590-#1600, #1621-#1624. Immediate cache observability/scoring foundations: #1519-#1528, #1564-#1568. | These items improve current reliability, operator visibility, context reset honesty, stale-fact safety, or cache reporting without needing a new serving architecture. | Focused tests, captured CLI/report output, or operator readouts showing the current path is safer or clearer now. |
| `gen/next` | Near-term cache/context integration: #1490-#1498, #1529-#1548, #1559-#1563, #1602-#1607, #1609, #1611, #1613-#1614, #1615-#1620, and this issue #1647. Cache-value accounting (2026-07-19 backfill): epic #2828 and children #2829-#2831. | These are runnable soon after gates, handoffs, and visibility exist: default-on vCache gates, O(1) context, live provider/cache metrics, breakpoint planning, context page faults, and clarification workflows. | Gate-off/gate-on witnesses, dogfood runs on real guard/serve sessions, cache economics with provenance, and bounded-context miss/fault rates. |
| `gen/second-next` | Cross-engine and architecture options: #39, #40, #53, #805, #985, #1258, #1463, #1467, #1469, #1549-#1558. | These need compatibility policy, engine capability contracts, disaggregated cache evidence, or adapter conformance before agents can safely run them as normal work. | Capability inventory rows, adapter conformance tests, cold-path correctness witnesses, and a compatibility rule that can split into gen/next implementation issues. |
| `gen/future` | Long-horizon benchmark, hardware, and market-shaping cache/context bets: #1010, #1476-#1477, #1678, and similar research or neo-hardware cache/context options. | These preserve option value or narrative direction, but their next action is usually research, a hardware witness, or a roadmap decision rather than default product behavior. | A research memo, benchmark decision, standards/market analogue, or hardware-run witness that changes the second-next option set. |

## Cache-Value Accounting Cluster (#2828), 2026-07-19 Backfill

The 2026-06-30 snapshot above predates the cache-value-accounting epic entirely,
so #2828 and its children carried no `generation` label and no milestone. This
subsection records the classification and the evidence behind it, so the label
move is not a hidden demotion.

**Classified `gen/next`, milestone `Generation G1 - Next Gen`.** The anchor case
is child A, #2829 (`fak savings counterfactual` — what fak's own kernel would
save on the sessions it proxies today).

Per this map's own rule that classification is grooming rather than automatic
label mutation, labels were applied only where the evidence was actually read:
#2829 (the anchor, examined here) and #2828 (the parent epic, so the child's
stream does not disagree with its epic). #2830 stays unlabeled pending its own
evidence pass, and #2831 closed before this backfill.

Why not `gen/now`: the promotion rule "cache value is net-true: measured against
the real alternative, net of miss, invalidation, freshness, and safety cost"
fails. Every own-serve dollar #2829 emits is MODELED, and its decisive negative
term — `own_serve_prefill_cost` — has no priced witness in the tree. Shipping it
as `gen/now` would be current-work laundering: a headline number whose cost side
is an undocumented guess does not clear the
[net-true-value standard](../standards/net-true-value.md).

Why not `gen/second-next`: it needs no compatibility policy, engine capability
contract, or cross-engine simulation. It is a pure fold over ledgers fak already
writes (`docs/nightrun/gateway-usage.jsonl`, `cache-savings.jsonl`) plus pricing
constants that already exist (`internal/gateway/cache_pricing.go`), and it
decomposes into agent-runnable work today.

Promotion evidence that would move the cluster toward `gen/now`:

- A priced prefill cost model. The analytic half already exists —
  `internal/compute.PrefillCostModel` returns exact per-stage prefill FLOPs and
  bytes with no timer, so the issue's "no per-model prefill cost model in the
  tree" gap is narrower than #2829 states. What is missing is only the $
  conversion: achieved FLOP/s (or tok/s) on a named device × an amortized
  $/GPU-hour constant. That measurement is CUDA-node work, not host work.
- The producer filling the consumer seam that already landed. Child C (#2831)
  closed having shipped `internal/ablate.ChildAOwnServeEstimate
  func() *float64`, a hook nothing currently supplies. A live caller is one of
  the six `gen/now` promotion conditions and it is already wired, waiting.
- The provenance fence holding under test: OBSERVED provider rebate and MODELED
  own-serve estimate rendered separately and never summed.

Demotion, parking, or retirement evidence for this cluster:

- Provider cache price or `CacheReadMultiplier` semantics change such that the
  0.1× read floor is no longer the dominant residual tax — the counterfactual's
  first term, and most of its headline.
- The own-kernel serve path (#1072's keystone) is itself demoted or retired. The
  estimator's number is only decision-relevant if own-serving stays on the
  table; if that bet dies, #2829 retires with it rather than standing alone.
- The modeled uplift lands inside the error bar of its own prefill-cost
  assumption, which would make the estimate unfalsifiable rather than merely
  imprecise.

Invalidating assumption: this classification treats `own_serve_prefill_cost` as
the binding blocker. If an operator instead accepts the conservative bound #2829
offers as an alternative — "prefill is free on a hit, full-recompute on a miss",
which needs no hardware constant at all — then the net-true objection weakens to
a documented worst-case bound and the cluster could be promoted to `gen/now`
without any CUDA-node measurement. That is an operator modeling decision, not an
evidence gap, and it is the cheapest path to promotion.

## Promotion Rules

Promote `gen/next` cache/context work toward `gen/now` only when all of these
are true:

- The mechanism has a live caller, command, report, or operator workflow.
- Gate-off or no-op behavior is witnessed when code exists.
- Gate-on behavior is bounded to the intended context/cache path.
- Cache value is net-true: measured against the real alternative, net of miss,
  invalidation, freshness, and safety cost.
- Context quality has a witness: reset idempotence, plan ID stability,
  miss/fault rate, task-success proxy, or before/after operator readout.
- The issue names the runtime gate or states why the artifact is safe by
  default.

Promote `gen/second-next` to `gen/next` only after the option has a compatibility
contract or simulation that can be decomposed into agent-runnable work. For
external engines, the minimum evidence is a cache capability row plus a cold-path
correctness witness; an adapter stub alone is not enough.

Promote `gen/future` to `gen/second-next` only after the research or benchmark
changes a decision: engine target, hardware target, product narrative, roadmap
priority, or compatibility rule. A future note that only says "this may matter"
stays future.

## Demotion, Parking, And Retirement

Move cache/context work farther from `now`, park it, or retire it when evidence
shows:

- Provider cache TTL, price, vary axes, or warming semantics invalidated the
  planned economics.
- The bounded resident view drops needed context or has no task-success witness.
- A cache path cannot prove provenance, secret handling, invalidation, or
  cold-path correctness.
- A context feature keeps asking the user or refreshing stale recall without
  improving action safety.
- External-engine integration cannot distinguish "fronted" from
  "cache-integrated" behavior.
- The issue has no owner, no witness, or no updated label/milestone after its
  review date.

Priority labels do not change just because a generation changes. A P1 future
research issue can stay P1, and a P2 now issue can stay P2.

## Follow-On Wiring

The durable version of this map should become machine-readable in one of these
places:

- `fak index generation` gains a cache/context filter;
- `fak program report` emits generation rows for cache-optimization and
  managed-context work;
- the milestone report folds cache/context program labels into the generation
  readout;
- issue grooming applies `generation` plus exactly one `gen/*` label to the
  open cache/context issues above.

## Invalidating Assumption

This note assumes the live GitHub label search remains a good proxy for active
cache/context work. Several `cache-default[...]` issues currently have no labels
or milestone, so the classification is a snapshot, not a complete control plane.
If those issues are not backfilled or surfaced through a report, future agents
must refresh the live issue query before dispatching from this map.

