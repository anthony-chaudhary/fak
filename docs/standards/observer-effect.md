---
title: "The observer effect — how fak reports its own overhead honestly"
description: "The provenance-honesty standard for every overhead number fak reports about itself. It states the duality fak is built on — a security floor a bad call can't get through, and a perf floor a good call can't silently slip below — requires WITNESSED / OBSERVED / MODELED / SIMULATED on every cost number, and pins the cost of the meter itself: measuring is not free, so the meter's own overhead must be bounded by a declared cap and that cap must be a green test, not a hope. The self-tax counterpart of net-true-value (which grades the gain); this grades the cost number's honesty."
---

# The observer effect

fak sits in the hot path of every tool call, every result, every turn. Each insertion
costs something — latency, tokens, wall-clock, sometimes a changed answer. So every time
fak reports one of its own overhead numbers, two failure modes are in play that the
[net-true-value](net-true-value.md) rubric (which grades *the gain*) doesn't fully catch:
the number can be **measured dishonestly** (a modeled floor quoted as a wall-clock; a
provider-side miss blamed on a fak action), and **the act of measuring it costs something
the number forgets to count**. This page is the standard for both. It is the cost-side
companion to net-true-value's gain-side rubric, used on fak's claims about **itself**.

## The duality: a security floor and a perf floor

fak describes itself as both a security gate and a performance gate. The security half is
real and mechanized: a default-deny **security floor** the model can't talk past — a bad
call can't get through, and a test reds the tree if one does. The performance half owes the
same shape:

> The security floor proves a bad call can't get through.
> **The perf floor proves a good call doesn't get slower than its declared budget — and,
> when fak makes it faster, says so with the same rigor it reports a safety win.**

These are two readings of one invariant: *a decision no participant can move by narrating a
number.* On the security side the decision is admit/deny; on the perf side it is
within-budget / over-budget. The full build-out of the perf floor — the per-turn meter, the
budget envelope, the CI regression gate — is the
[self-tax plane epic (#1147)](../notes/self-tax-performance-assurance-tracking-1147.md). This
page is its honesty contract: the rule every number that plane emits must obey.

## Every overhead number carries a provenance label

This is the same closed vocabulary [net-true-value](net-true-value.md) uses for a gain;
here it is mandatory on every **cost** fak reports about itself. A number with no label is
`not yet`, not a result.

| Label | Means | Example overhead use |
|---|---|---|
| **WITNESSED** | a fact fak authored and controls, re-derivable by a third party | "the acceptance meter allocates 0 bytes per sample" — a `go test` reads it back |
| **OBSERVED** | a value relayed from an external party fak does not control | a wall-clock ns on a shared box; a provider's cache-hit latency |
| **MODELED** | a deterministic projection, never run on the target | the turn-tax meter's live sampled-ns cap, until the live meter ships |
| **SIMULATED** | labeled stand-in data, not a real workload | the cost model's per-turn latency in the turn-tax demo |

The rules that follow from the labels: a **MODELED** floor is never quoted as a measured
wall-clock; an **OBSERVED** provider-side cost is never reported as a fak action's cost
(the [conflation scorecard](../CONFLATION-SCORECARD.md) enforces this separately); a
**SIMULATED** number never stands in for a witnessed one without the word. A wall-clock
overhead is OBSERVED, not WITNESSED, because the box's load — not fak — moves it; that is
why the witnessed bound below is an *allocation* count, which is deterministic across runs
and hosts, rather than a nanosecond figure that would flake.

## The meter's own cost is bounded — and the bound is a green test

The observer effect is the literal version: instrumentation that measures a hot path slows
it, and the slowdown is variable (10–53% for full instrumentation in the profiling
literature; 1–2% and stable for sampling). The honesty fence fak holds itself to is plain:
**a meter fak puts on a hot path must cost less than a declared cap, and that cap must be a
green test — because you cannot honestly report an overhead you never bounded.**

fak's shipped hot-path meter is the speculative-decode `AcceptanceMeter`
(`internal/spec`, #284). It is designed to the fence two ways:

- **The un-metered path pays nothing.** `SpeculativeGreedy` passes a nil meter and is the
  fast path; the metered run is byte-identical to it. *Witness (WITNESSED):*
  `internal/spec/metrics_test.go` — the nil-meter wrapper reproduces the same output tokens.
- **The metered path's own cost is capped, and the cap is measured.** Each `Observe` is a
  pure accumulator (a few integer adds, no I/O, no allocation), so its declared cap is the
  tightest a meter can have: **zero heap allocations per sample**. *Witness (WITNESSED):*
  `internal/spec/metrics_cost_test.go::TestAcceptanceMeterObserveUnderCostCap`, which pins
  `testing.AllocsPerRun(…, m.Observe) == 0` — deterministic, so it is a witnessed bound, not
  a noisy wall-clock one.

So the pinned cost is honest in both directions: the *resource* cost of metering is
**0 allocations/sample (WITNESSED)** and the *behavioral* cost is **byte-identical output
(WITNESSED)**. The per-turn self-tax meter the #1147 plane promotes from `cmd/turntaxdemo`
is the next meter this fence governs; its **live sampled-ns cap is MODELED** until that
meter ships against a real workload — labeled, not quoted as measured. That is the fence
working: the number we have is witnessed; the number we don't have is named MODELED rather
than dressed up.

## How this is encountered by default

- **Agents** meet it in [`AGENTS.md`](https://github.com/anthony-chaudhary/fak/blob/main/AGENTS.md) beside the "every claim carries a
  tag" rule and the [net-true-value](net-true-value.md) lens — the cost-side check an agent
  runs before reporting any "fak adds X%" or "fak saved Y" number.
- **Humans** meet it in the doc map ([`llms.txt`](https://github.com/anthony-chaudhary/fak/blob/main/llms.txt), [`INDEX.md`](https://github.com/anthony-chaudhary/fak/blob/main/INDEX.md))
  beside the net-true-value standard and the [self-tax plane note](../notes/self-tax-performance-assurance-tracking-1147.md).

## Honest fences

- This page is a **standard plus an honesty contract**, not the perf floor itself. The
  always-on per-turn meter, the budget envelope, and the CI regression gate are the
  [#1147 epic](../notes/self-tax-performance-assurance-tracking-1147.md) build-out; this
  states the rule they must satisfy, and pins the one shipped meter that already does.
- A budget is an envelope with a stated scope, not a promise of zero cost. A gate that costs
  8% and saves 40% is a net win; the perf floor must say that rather than red on the 8%
  alone — the same net-of-cost reading [net-true-value](net-true-value.md) Question 2 asks.
- Pinning the meter's cost at zero allocations bounds its *per-sample* cost; it does not
  bound the cost of a meter that samples too often. Rate-bounded sampling — so the meter
  reads a fraction of events, never full-instruments the hot path — is the companion fence,
  tracked as the observer-effect ticket (T4) in the [#1147 plane](../notes/self-tax-performance-assurance-tracking-1147.md).
