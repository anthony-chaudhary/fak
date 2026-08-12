# The problems fak exists to solve

> **Direction:** make operating useful agent systems feel like “don’t make me think.”
> One integrated kernel should absorb recurring context, measurement, market-change, and
> operations decisions so a person can state the outcome instead of assembling a new stack.

This is fak’s canonical problem map. It is a product-direction document, **not** a claim
that every part is already automatic or shipped. Capability claims remain in
[`CLAIMS.md`](../CLAIMS.md), and measured gains remain subject to the
[net-true-value standard](standards/net-true-value.md).

## The promise: less operator cognition, not less operator control

“Don’t make me think” does not mean hiding consequential choices or removing review. It
means making the routine correct path the easy default: retain the right context, choose
and measure efficiency techniques honestly, adapt integrations as the field changes, and
operate the fleet through one coherent boundary. fak should ask a human only for a real
decision, expose why it acted, and fail closed when it cannot act safely.

The next-best alternative is not “do nothing.” It is a person repeatedly wiring prompt and
context policy, benchmark scripts, provider changes, serving, dispatch, leases, retries,
security gates, and evidence ledgers. As agent count and market velocity rise, that bespoke
integration tax grows faster than human attention. fak’s value is to internalize that tax
behind one binary and one tool-call checkpoint.

## P1 — Context should manage itself

**Problem.** Humans are poor context schedulers. They cannot reliably decide what every
agent should retain, retrieve, compact, share, or forget on every turn; at fleet scale, and
as tasks, models, prices, and context windows change, manual context management is not a
viable control loop. More tokens alone do not solve relevance, continuity, cache survival,
or isolation.

**Direction.** fak should make context an automatically managed resource: reuse shared
setup, preserve provider-cache value, retrieve or compact at the boundary, keep continuity
across long sessions, and expose intervention only when policy or uncertainty requires it.
The desired operator experience is to specify the task—not curate every prompt.

**Current seams and evidence maps:** [`managed-context-continuous-usage.md`](managed-context-continuous-usage.md),
[`CONTEXT-IS-NOT-MEMORY.md`](CONTEXT-IS-NOT-MEMORY.md),
[`long-session-value.md`](long-session-value.md), and [`cache-value-rollup.md`](cache-value-rollup.md).

## P2 — Efficiency must be net-true

**Problem.** The agent market has too many “efficiency” techniques supported by weak or
misleading comparisons. A cache, router, compactor, smaller model, or orchestration layer
can move cost to another component, degrade task success, add latency or operational work,
or beat only a deliberately naive baseline. An optimization that hurts the complete system
is not a gain.

**Direction.** fak should organize optimization around reproducible, end-to-end evidence:
compare with the real tuned alternative, count added costs, preserve outcome quality and
safety, label provenance, and say `not yet` when no witness exists. Selection and reuse
should become automatic only after the gain is shown to be net-true.

**Current seams and evidence maps:** [`standards/net-true-value.md`](standards/net-true-value.md),
[`BENCHMARK-METHODOLOGY.md`](BENCHMARK-METHODOLOGY.md),
[`../BENCHMARK-AUTHORITY.md`](../BENCHMARK-AUTHORITY.md), and
[`DEFAULT-VALUE-SCORECARD.md`](DEFAULT-VALUE-SCORECARD.md).

## P3 — Fast adaptation without another standard

**Problem.** Models, providers, protocols, memory methods, serving engines, agent
frameworks, and claimed best practices change too quickly for every team to continuously
re-evaluate and rewire. A new universal standard would become one more moving dependency;
fak must not pretend to freeze a market that is still discovering its shape.

**Direction.** fak should be an adaptable kernel at the existing tool-call boundary: thin
leaves and adapters can absorb the latest useful technique, test it against the same
contracts and evidence floor, and replace it when the field moves. The stable thing is the
checkpoint and its verification discipline—not a mandated provider, framework, protocol,
or technique.

**Current seams and evidence maps:** [`integrations/README.md`](integrations/README.md),
[`a2a-value-opportunities.md`](a2a-value-opportunities.md),
[`notes/CONCEPT-EMERGING-MARKET-ADOPTION-2026-06-30.md`](notes/CONCEPT-EMERGING-MARKET-ADOPTION-2026-06-30.md),
and the additive leaf model in [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

## P4 — Agent operations should be one product, not a kit

**Problem.** Rolling an agent stack still means combining serving, model routing, context,
tool policy, credentials, dispatch, concurrency, leases, retries, observability, proof, and
recovery. Each new agent, provider, and accelerator multiplies interactions. A pile of
individually good components still leaves the operator as the integration layer.

**Direction.** fak should converge those recurring decisions behind one binary and a small
set of coherent operating shapes—from one local agent to a witnessed fleet. Safe defaults,
self-checks, routing, serving, supervision, and evidence should work together; advanced
controls remain available without being prerequisites for the first useful run.

**Current seams and evidence maps:** [`fleet.md`](fleet.md), [`fleet-rollup.md`](fleet-rollup.md),
[`loops-user-value.md`](loops-user-value.md), [`integrations/README.md`](integrations/README.md),
and the runnable routes in [`../START-HERE.md`](../START-HERE.md).

## How work maps to the problems

Every new issue or plan should name one **primary problem ID** (`P1`–`P4`) and state the
observable operator burden it removes. Supporting more than one problem is welcome, but
“improves the architecture” is not sufficient by itself. Use this compact frame:

- **For:** who gets the benefit?
- **Problem:** which `P1`–`P4` burden appears in their real workflow?
- **Today:** what is the real next-best alternative, including human integration work?
- **Better because:** what observable step, decision, failure, cost, or delay disappears?
- **Witness:** what artifact would prove that movement without relying on the author’s claim?

A completed issue should be explainable as movement on at least one row below. If it cannot,
rescope it, connect it to an enabling issue that can, or do not prioritize it.

| ID | We are moving when… | We are **not** done merely because… |
|---|---|---|
| P1 | fewer context-curation decisions are required while continuity, relevance, isolation, and outcome quality hold | more tokens were accepted or a memory component was added |
| P2 | an end-to-end, reproducible comparison shows a gain against the tuned alternative, net of added costs | one component benchmark got faster or cheaper |
| P3 | a new field technique/provider/protocol can be evaluated and integrated through an existing seam without making it permanent doctrine | a trend was documented or a new abstraction was invented |
| P4 | a real operator completes a fleet/serve/agent outcome with fewer separately managed systems and recoverable defaults | more knobs or another standalone tool were added |

## Product decision rule

When priorities compete, prefer the smallest working spine that removes the most repeated
human integration or judgment while preserving evidence and control. Then fan out the
operating envelope. This is the “all in one, easy by default” direction in practical form:
**one boundary, fewer decisions, measured outcomes, replaceable internals.**
