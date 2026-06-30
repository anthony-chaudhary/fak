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
| `gen/next` | Near-term cache/context integration: #1490-#1498, #1529-#1548, #1559-#1563, #1602-#1607, #1609, #1611, #1613-#1614, #1615-#1620, and this issue #1647. | These are runnable soon after gates, handoffs, and visibility exist: default-on vCache gates, O(1) context, live provider/cache metrics, breakpoint planning, context page faults, and clarification workflows. | Gate-off/gate-on witnesses, dogfood runs on real guard/serve sessions, cache economics with provenance, and bounded-context miss/fault rates. |
| `gen/second-next` | Cross-engine and architecture options: #39, #40, #53, #805, #985, #1258, #1463, #1467, #1469, #1549-#1558. | These need compatibility policy, engine capability contracts, disaggregated cache evidence, or adapter conformance before agents can safely run them as normal work. | Capability inventory rows, adapter conformance tests, cold-path correctness witnesses, and a compatibility rule that can split into gen/next implementation issues. |
| `gen/future` | Long-horizon benchmark, hardware, and market-shaping cache/context bets: #1010, #1476-#1477, #1678, and similar research or neo-hardware cache/context options. | These preserve option value or narrative direction, but their next action is usually research, a hardware witness, or a roadmap decision rather than default product behavior. | A research memo, benchmark decision, standards/market analogue, or hardware-run witness that changes the second-next option set. |

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

