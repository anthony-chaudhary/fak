---
title: "Generation Second-Next Architectural Option Contracts"
description: "The option-contract shape a gen/second-next architectural bet must carry before it can be evaluated — assumption set, reversible seam, cost to keep alive, promotion trigger, and kill trigger — so architecture options get enough shape to compare without prematurely committing the trunk."
---

# Generation Second-Next Architectural Option Contracts

**Issue:** #1664.
**Parent:** #1625.
**Stream:** `gen/second-next`.
**Milestone:** Generation G2 - Second Next Gen.
**Status:** research memo / option-contract template for second-next architecture bets.

This page gives a future agent the contract an architectural option must carry to
be a real `gen/second-next` bet rather than an undated aspiration — without
rereading the generation epic
([#1625](https://github.com/anthony-chaudhary/fak/issues/1625)). It is the
`gen/second-next` companion to the `gen/future`
[real-options model](generation-future-real-options-model.md): that model
*values* a research option; this contract *shapes* an architecture option so a
later stream can evaluate it against the trunk it will eventually touch.

## Core Rule

A `gen/second-next` item is an **architectural option under contract**, not an
approved design. The fleet holds the right to build a seam into current APIs
*later*, on evidence, but must not spend that right early by shipping the
architecture as a default or forking the trunk to explore it. An architecture
bet without a stated contract is not "early" — it is unpriced, and an unpriced
option is the thing this contract refuses.

Second-next is the horizon that "needs simulation, compatibility policy, or
cross-generation dependency management before it can become active product
work" ([`docs/generation.md`](generation.md) §Streams). The contract below is
what makes those three prerequisites checkable rather than aspirational: each
field names the evidence a later agent can read to decide *promote*, *hold*,
*demote*, or *retire*.

## The Option Contract

Every `gen/second-next` architectural option states these five fields. The issue
that seeds the option (or the memo that carries it) is incomplete until all five
are named. They map onto the promotion verbs in
[`docs/generation.md`](generation.md) §Promotion Verbs — the contract adds the
*shape* that justifies the verb, it does not invent a new verb set.

### 1. Assumption set

The named beliefs the architecture bet rests on, each stated so a later agent can
check it cheaply against live repo, benchmark, issue, or market evidence. At
least one must be an **invalidating assumption**: a belief whose failure retires
the option outright (it feeds the kill trigger below). The assumption set is the
uncertainty the second-next horizon exists to burn down; an option whose
assumptions are already retired belongs in `gen/next`, and one whose assumptions
cannot be stated belongs in `gen/future` research until they can.

### 2. Reversible seam

The single, additive interface boundary through which the architecture would
touch current code — and the proof that touching it is *reversible* without a
trunk-wide edit. The seam is what lets second-next influence current APIs
without a per-generation branch: it must be expressible under the additive-only
promise in
[`docs/generation-abi-compatibility-policy.md`](generation-abi-compatibility-policy.md)
(a new enum value, a new versioned schema, a default-off gate, a new interface
method with a fail-closed default), never a renumber, a removed field, or a
breaking signature change. If the smallest honest seam requires a breaking
change, the option is not yet reversible and stays parked until it is. "Seam"
is deliberately singular: an option that can only be evaluated by editing many
call sites at once has no reversible seam and is not contractible as a
second-next bet.

### 3. Cost to keep alive

The carry cost of holding the option open per recheck cycle, denominated in the
scarce resources the
[capacity model](generation-agent-capacity-model.md) already names — attention,
recheck overhead, the simulation/compat fixture the bet must keep green, and the
context tax on every backlog scan. Unlike a `gen/future` research option, a
second-next option's carry usually includes **keeping a simulation or
compatibility witness green** as the trunk moves underneath it; a bet whose
fixture has rotted is paying no carry and is a retirement candidate. An option
with no named carry has not become free — it has externalized its carry onto the
lane, which this contract refuses exactly as the real-options model does.

### 4. Promotion trigger

The concrete witness whose arrival moves the option `second-next → next` — the
strike condition for architecture rather than research. For second-next this is
almost always one of: a passing **simulation** that shows the architecture beats
the incumbent on a named metric; a **compatibility fixture** proving the seam is
additive across a `/1`↔`/2` boundary (the promotion witness named in
[`docs/generation-abi-compatibility-policy.md`](generation-abi-compatibility-policy.md));
or a **dependency edge** resolving so the seam's prerequisites are all shipped.
The trigger names the artifact (test, fixture, benchmark, or shipped
prerequisite) — not a date and not a vibe. When it fires, promotion follows the
witness-ladder rung for the next horizon
([`docs/generation-witness-ladders.md`](generation-witness-ladders.md)); it does
not skip straight to `now`.

### 5. Kill trigger

The named condition that retires the option immediately rather than at the next
recheck. It reuses the closed sunset-trigger vocabulary in
[`docs/generation-future-sunset-criteria.md`](generation-future-sunset-criteria.md)
(`CARRY_EXHAUSTED`, `STRIKE_UNREACHABLE`, `ASSUMPTION_FIRED`, `SUPERSEDED`,
`ORPHANED`, `STALE_RECHECK`, `HORIZON_LAUNDERED`) so second-next and future
retirements speak one language. The load-bearing second-next kill triggers are
`ASSUMPTION_FIRED` (an invalidating assumption from field 1 failed),
`STRIKE_UNREACHABLE` (the reversible seam in field 2 provably cannot stay
additive), and `SUPERSEDED` (a nearer stream shipped a design that covers the
bet). A retirement cites exactly one token and carries the four-piece
retirement-evidence contract from that doc (trigger token, witness, disposition,
orthogonality note); a retirement that cannot name a token is a hidden demotion
and is re-opened.

## Orthogonality

This contract is planning shape, not a branch, a priority, or a runtime switch.

- **Orthogonal to priority.** A high-priority architecture bet still needs all
  five contract fields before it is dispatchable; a low-priority one that names
  them cleanly is a legitimate carried option. Priority answers "how urgent";
  the contract answers "is this a real, reversible, retirable option or an
  unpriced aspiration."
- **Orthogonal to shared trunk.** Every second-next option lands on `main` by
  explicit path under the normal DCO and ship-stamp rules. The reversible-seam
  field exists precisely so the architecture can be explored *without* a
  per-generation branch or worktree escape — the contract forbids the branch it
  might otherwise tempt.
- **Orthogonal to runtime feature gates.** An option's seam may land inert behind
  a default-off gate; the gate decides runtime exposure, the contract decides
  whether the architecture bet was worth shaping and carrying at all. Neither
  substitutes for the other: a gated no-op still owes its five fields, and a
  fully-specified option is still invisible at runtime until a gate exposes it.

Do not resolve option pressure with a generation branch, a hidden worktree, or
ungated default exposure.

## Promotion Evidence

The option promotes `second-next → next` when its **promotion trigger** fires
with a readable witness:

- A simulation or compatibility fixture named in field 4 is committed and green,
  and the recheck note records the metric it beat or the `/1`↔`/2` boundary it
  proved additive.
- The reversible seam landed as an additive change (new enum value, new schema
  version, new interface method with a fail-closed default, or a default-off
  gate) with the compatibility witness attached — never as a breaking edit.
- Any dependency edges the option declared are all resolved (the prerequisite
  issues are shipped and witnessed), so the seam has nothing left blocking it.
- Promotion updates labels, milestone, and the evidence comment on the existing
  issue; it does not open a duplicate in the `gen/next` lane.

## Demotion / Retirement Evidence

The option demotes or retires when its **kill trigger** fires, with the same
four-piece evidence contract the sunset criteria require:

- `ASSUMPTION_FIRED` — a named invalidating assumption from the assumption set
  failed against live repo, benchmark, issue, or market evidence; cite the
  assumption text and the artifact that fired it.
- `STRIKE_UNREACHABLE` — the reversible seam provably cannot stay additive (a
  breaking change is unavoidable), so the compatibility policy closed the path;
  cite the policy, gate, or proof.
- `SUPERSEDED` — a nearer stream shipped a design that covers the bet; cite the
  superseding commit/issue/doc.
- `CARRY_EXHAUSTED` / `STALE_RECHECK` — the cost to keep alive was paid with no
  progress toward the trigger, or the option's simulation/compat witness rotted
  and was not re-witnessed within the cadence; cite the recheck history.

Demote (rather than retire) when uncertainty fell but so did value: keep the
option as context at a cheaper carry tier with a fresh recheck date. Do not
retire or demote by moving labels alone — name the field that moved and the
witness that moved it.

## Worked Example

A hypothetical `gen/second-next` option: *"Replace the contiguous KV context
store with a paged-KV allocator so long sessions reuse cache pages across
turns."*

- **Assumption set** — (a) paged allocation beats contiguous on reuse-heavy
  sessions by a measurable margin; (b) *invalidating:* the paging index and
  per-page state can be expressed without a fak-side state-cache the current
  engine seam cannot carry (this assumption is exactly where a pure-fak paged-KV
  bet has historically failed for GLM-style DSA index/state caches, so it is the
  kill criterion, not a footnote).
- **Reversible seam** — a new default-off `PagedKV` allocator behind the existing
  context-store interface, selected by a gate, emitting a new versioned
  `paged-kv/1` telemetry schema; the contiguous path is untouched and stays the
  default, so the seam is additive and reversible.
- **Cost to keep alive** — one recheck cycle of keeping a paged-vs-contiguous
  A/B simulation fixture green as the context store evolves, plus the attention
  tax of one open architecture issue.
- **Promotion trigger** — the A/B simulation shows a positive reuse-token delta
  on a named workload AND the `paged-kv/1` schema passes an additive
  compatibility fixture; then promote toward `gen/next` behind the gate.
- **Kill trigger** — `ASSUMPTION_FIRED` if the paging index provably needs
  engine-side state the fak seam cannot carry (the paged-KV / DSA gap), or
  `SUPERSEDED` if an engine adapter ships the capability first.

Disposition today: **hold.** The reversible seam is stated and additive, but the
invalidating assumption has not been burned down by a simulation, so the
promotion trigger has not fired. Recheck on the standard cadence; retire on the
kill trigger.

This is the shape a second-next memo should have: five contract fields named, a
disposition, a promotion trigger, and a kill trigger drawn from the closed
vocabulary.

## Relationship To Existing Surfaces

This contract composes with, and does not duplicate, the existing generation
surfaces:

- **Real-options model
  ([`docs/generation-future-real-options-model.md`](generation-future-real-options-model.md))**
  values a `gen/future` *research* option (`V`/`X`/`C`/`σ`/`T`/`S`). This
  contract shapes a `gen/second-next` *architecture* option. Carry cost here is
  the same `C`; the promotion trigger is the same `S` specialized to
  simulation/compatibility witnesses; the kill trigger is the same retire
  disposition. The difference is the reversible-seam field, which a research
  option does not yet need and an architecture option cannot omit.
- **ABI/schema compatibility policy
  ([`docs/generation-abi-compatibility-policy.md`](generation-abi-compatibility-policy.md))**
  is what makes the reversible-seam field *enforceable*: the additive-only wire
  ABI and versioned schemas are the promise that a seam stays reversible. That
  memo's `future → second-next → next → now` promotion ladder is the same ladder
  a bet under this contract climbs.
- **Sunset and kill criteria
  ([`docs/generation-future-sunset-criteria.md`](generation-future-sunset-criteria.md))**
  supplies the closed trigger vocabulary and four-piece retirement-evidence
  contract the kill-trigger field reuses, so second-next and future retirements
  are audited identically.
- **Witness ladders
  ([`docs/generation-witness-ladders.md`](generation-witness-ladders.md))**
  set the minimum evidence rung a promotion trigger must clear for the target
  horizon; this contract names *which* witness, the ladder sets *how strong* it
  must be.
- **Debt metric ([`docs/generation.md`](generation.md) §Debt Metric)** is the
  lane-aggregate signal; an option missing contract fields shows up there as an
  `unpromoted_bet` or a `missing_witness`. The per-option contract explains
  *which* field is missing.

## Invalidating Assumptions

This contract itself depends on these assumptions, stated so a later agent can
check them:

1. **Every real second-next bet has a single reversible seam.** The contract
   assumes an architecture option worth carrying can name one additive interface
   boundary. If a class of genuinely valuable bets can only be evaluated through
   an irreducibly breaking, multi-site change, the reversible-seam field is too
   strict and the contract must gain an explicit, witnessed breaking-change
   protocol (the same gap the compatibility policy names as *its* most-likely
   failure). **This is the assumption most likely to fail.**
2. **The five fields are cheap to state at intake.** If most second-next issues
   cannot name their assumption set, seam, carry, promotion, and kill triggers,
   the contract is vacuous and collapses into "carry every architecture idea
   forever" — at which point the lane-aggregate `debt_score` is the honest
   surface and this contract should be retired in favor of it, not defended with
   fabricated fields.
3. **Simulation/compatibility witnesses are reachable on this host.** The
   promotion trigger assumes a second-next bet can produce a runnable simulation
   or compat fixture. A bet whose only honest witness needs hardware or an
   external engine the fleet cannot reach stays parked with that gap named — it
   is not promoted on a modeled-as-measured witness.

If these assumptions fail, replace the contract with the stronger measured
surface (the debt metric, or a `fak generation audit` view that reads contract
completeness from issue bodies). Do not keep an option-contract template as an
operator-facing fact once its fields have stopped being honest.

## Handoff (continue from here without the epic)

A future agent picking this up should: (a) apply the five-field contract to the
open `gen/second-next` issues under [#1625](https://github.com/anthony-chaudhary/fak/issues/1625),
flagging any that cannot name a reversible seam as demotion candidates; (b) file
the smallest next increment — a `gen/second-next` issue-template section that
captures the five fields at intake — as its own issue, not a note; (c) leave the
compatibility policy's additive proof and the sunset vocabulary untouched (this
contract reuses them, it does not fork them). The contract here is planning data
until an intake template or a `fak generation audit` makes field-completeness
checkable at commit time.
</content>
</invoke>
