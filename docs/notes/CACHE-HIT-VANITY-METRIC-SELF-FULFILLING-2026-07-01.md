---
title: "The cache-hit vanity metric: a self-fulfilling prophecy toward over-long sessions"
description: "Prompt-cache hit rate rises with trajectory length by construction, so any loop that reads it as an efficiency signal is pushed to run longer — the direction that also makes each turn cost more and the reasoning worse. This is the incentive layer under the frozen-trajectory cache cliff: a metric that improves precisely because the pathology it appears to reward got bigger. Names the self-fulfilling prophecy on both sides (long is rewarded, cutting is punished), gives the counter-metrics to steer on instead, and draws the many-agent-workflow corollary."
date: 2026-07-01
keywords:
  - prompt caching
  - cache hit rate
  - vanity metric
  - context rot
  - session length
  - self-fulfilling prophecy
  - perpetual sessions
  - multi-agent
---

# The cache-hit vanity metric

**Thesis.** Prompt-cache hit rate is a *workload-shape* number that rises toward
100% as a single agent's trajectory gets longer — and it rises for the *same
reason* each turn gets more expensive and the model's reasoning degrades: the
cached prefix keeps growing. So the moment any loop (an operator watching a
dashboard, a scoreboard, a reward signal, or the model's own sense that "the
cache is hot, this is cheap") treats a high hit rate as an efficiency win, it is
steered toward exactly the failure mode — the over-long session — that the number
is silently *measuring*, not curing. Cache-hit rate is a fine **diagnostic of
trajectory shape**. It is a **vanity metric as an objective**, and optimizing it
is a self-fulfilling prophecy.

This is the *incentive* companion to the descriptive
[frozen-trajectory cache cliff](../explainers/frozen-trajectory-cache-cliff.md)
(which proves hit rises with length) and the
[session cache-savings ablation](SESSION-CACHE-SAVINGS-ABLATION-2026-06-29.md)
(which shows the "saved-token" headline is provider-driven and grows with length).
It is the *why-it's-a-trap* between those two and the fix in
[perpetual sessions / the relay](CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md).

## 1. The number is high because the trajectory is frozen

A single append-only agent that only ever *appends* to its transcript has a
cache-read fraction that rises with turns:

```text
h_frozen(T) = (T-1)/(T+1)   →  82% at 10 turns, 96% at 50, 99% at 200
```

(Derivation and the real 30-day machine anchor — 96.6% machine-wide, a 205-turn
session at 99% — are in the cliff explainer.) The number is high *because the
harness refuses to touch history* so the byte-identical prefix keeps matching.
That is a property of the *shape* (single, linear, append-only), not of caching
being free. Quote the number without the shape and it misleads the instant the
shape changes.

## 2. The inversion: the hit rate rises *as* the bill rises

Here is the part the headline hides. Price that same frozen trajectory in
base-input-price units (`cache read = 0.10×`, fresh delta `= 1×`, cache write
`= 1.25×`). Turn *t* re-reads a prefix of `(t-1)·d` cached tokens and pays for a
fresh delta `d`:

```text
per-turn bill   c(t) = d · (1 + 0.10·(t-1))     →  rises linearly, without bound
reported hit    h(t) = (t-1)/(t+1)               →  rises toward 1, saturating
```

They move **together**, and that is the whole point: the hit rate is high
*because* the cached prefix grew, and that same grown prefix is what makes each
turn cost more. Run `python tools/cache_curve.py inversion` (deterministic,
stdlib-only) and read the two columns side by side:

```text
 turn  hit(cum)   turn bill       cum bill   "saved" hdln  cut pays back in
    2     33.3%       2,200          4,200          1,800               n/a
   10     81.8%       3,800         29,000         81,000               n/a
   50     96.1%      11,800        345,000      2,205,000        3.21 turns
  100     98.0%      21,800      1,190,000      8,910,000        1.40 turns
  200     99.0%      41,800      4,380,000     35,820,000        0.66 turns
```

At turn 200 the dashboard shows a **99% hit** and a **35.8M "saved-token"
headline** — both at their most flattering — while the *per-turn* bill is
**19× larger** than it was at turn 2 and the cumulative bill is quadratic in
length. A metric that looks best exactly when the run is most bloated is not
measuring efficiency; it is measuring length wearing efficiency's clothes.

Two derived numbers make the same point:

- **Cumulative cost is `~0.05·d·T²`** — superlinear. Doubling the turns *more*
  than doubles the bill.
- **The "saved-token-equiv" headline is `~0.9·d·T²/2`** — also quadratic. The
  celebrated "we saved 25.5M tokens" number is arithmetically *"we ran a very
  long session."* (See the ablation note: the provider cache did ~99.9% of the
  visible saving and would have done it without fak; fak's job is to price it
  honestly, not to be flattered by it.)

## 3. The self-fulfilling prophecy, on both sides

The trap is that the *cheap-to-read* signals all point one way and the *true*
signals are counterfactual and invisible.

**Long is self-reinforcing (the metric rewards continuing).** As a session runs
on: the hit rate climbs, the "saved" headline climbs, and every marginal token is
billed at `0.10×`, so each *individual* turn *looks* cheap in isolation. Nothing
in the visible telemetry says "stop." The model's own local view agrees — a warm,
familiar, append-only context feels productive right up until context rot sets in.
So the loop keeps going, the number gets prettier, and the prettier number is read
as vindication. The prophecy fulfills itself: *"the cache is hot, so running long
is efficient"* → run longer → *"look, the hit rate went up."*

**Cutting is self-suppressing (the metric punishes stopping).** Ending a leg and
starting fresh eats one cold prefix write and shows up as a **visible, localized
hit-rate dip** — a legible "regression" on exactly the dashboard everyone watches.
Its benefits — a flat per-turn bill from a small prefix, and the shed context rot
that makes the *next* decisions better — are **diffuse and counterfactual**: you
never see the expensive long turns you didn't take, or the mistakes you didn't
make. Visibility asymmetry plus loss aversion means the cut is avoided even when it
is plainly net-positive. And it *is* net-positive fast: the `cut pays back in`
column shows a fresh leg (system + an O(1) baton, ~20k tokens) repays its cold
write in **under one turn** once the monolith prefix has grown past ~100 turns —
`n* = 1.25·B / (0.10·(P−B))`. The visible cost is recouped almost immediately; the
metric just refuses to show you that.

> The two prophecies are the same coin. The hit-rate metric makes the pathology
> (a growing frozen prefix) look like the cure, and makes the cure (a cut to a
> flat prefix) look like the pathology.

## 4. What to steer on instead

Cache-hit rate keeps its job as a **shape diagnostic** — a low or decaying hit
tells you the trajectory went flexible or tool-dense (the cliff's axes 1–2), which
is real information. It just must never be the *objective*. Steer on quantities
that fall when the session should end:

- **Verified progress per token.** Not turns, not tokens, not "saved" tokens —
  *witnessed* forward motion per unit spend. fak already has the witness half:
  `dos_status` folds ledger-**verified** progress (it structurally has *no*
  `claimed` field), and `dos_commit_audit` confirms a commit's diff did the kind
  of thing its message claims. Progress-per-token is the numerator the vanity
  metric lacks; when it flattens, the leg is done regardless of how warm the cache
  is.
- **Marginal cost per turn (the trend, not the level).** `tools/ctxcost.py
  marginal` already emits the per-turn `dC/dn` ledger and the *cumulative-rebill
  fraction* — how much of the running bill is just re-billed prefix. A rising
  rebill fraction is the honest early-warning the hit rate suppresses.
- **Peak / flat context.** The relay's flat-context invariant: peak resident
  tokens bounded by a per-leg ceiling, *independent of goal duration*. A relay
  that runs for a week peaks no higher than one that runs an hour. This is the
  objective the perpetual-sessions epic (#1860) is built to hold.

The rule of thumb: **arm a rotation on a soft context mark (50–70% of the window),
reach the next safe point, externalize, and cut** — do not wait for the ~95%
auto-compaction cliff, and do not let a climbing hit rate talk you out of it. The
one-time reset is the cheapest turn you'll buy.

## 5. The many-agent-workflow corollary

Fan-out is where the vanity metric is most actively *backwards*, and where the
antidote is already structural.

- **A short, fresh sub-agent is the opposite of the trap.** Born with a clean
  window, does one scoped thing, returns *structured output*, dies. Its trajectory
  is bounded and its prefix small by construction — no context rot, no quadratic
  per-turn bill, no "keep going because the cache is warm." A workflow that
  pipelines many short agents is the flat-context invariant applied across a fleet
  instead of across a relay's legs.
- **A hit-rate maximizer would prefer the monolith — which is wrong.** A single
  "do it all in one session" agent hoards a growing prefix and scores as a
  *hit-rate hero* (that 99%). Fan the same work across N short agents launched
  concurrently and cross-agent reuse of the shared setup is **0% and flat in N**
  (the concurrency wall, cliff axis 3): the blended fleet hit *looks worse*. So a
  loop optimizing fleet hit rate would kill the fan-out and keep the monolith —
  penalizing the very structure (short, disjoint, parallel) that is faster,
  cheaper per unit of verified progress, and more correct. Same inversion, fleet
  scale.
- **Recover the fan-out cost the right way, not by avoiding fan-out.** The 0%
  cross-agent reuse is a *forfeited* win, not a reason to serialize into one long
  session. Recover it by **paying the shared prefix once and cloning it
  bit-identically** into every agent (the shipped
  [pay-the-prefix-once](../../visuals/65-pay-the-prefix-once.svg) path), or by
  staggering launches within the cache TTL. Then judge the workflow by
  **confirmed findings / commits per dollar**, not by fleet hit rate.

Concretely, for orchestrated work (the `Workflow` tool, `super-loop`, wave
dispatch): prefer `pipeline()` of short, single-purpose agents over one deep
agent; give each a tight scope and a structured return so its context stays O(1);
recover shared-prefix cost by cloning, not by collapsing the fan-out; and score
the run on adjudicated progress, never on how much the fleet "saved."

## 6. Where this lives in fak

| Layer | Surface | Status |
|---|---|---|
| Descriptive: hit rises with length | `docs/explainers/frozen-trajectory-cache-cliff.md`, `tools/cache_curve.py` | shipped |
| The cost inversion (this note's core) | `tools/cache_curve.py inversion` (`turn_cost` / `cum_cost` / `saved_headline` / `cut_break_even_turns`) | shipped (this note) |
| Honest attribution of the "saved" headline | `SESSION-CACHE-SAVINGS-ABLATION-2026-06-29.md`, `internal/gateway/cache_pricing.go` | shipped |
| Marginal per-turn ledger + rebill fraction | `tools/ctxcost.py marginal`, `tools/ctxbench.py` | shipped |
| Sessions-to-break-even for warm cache | `fak turntax --breakeven`, `internal/turnbench/stochastic.go` | shipped |
| The fix: cut, don't compact (flat context) | `CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md`, epic #1860 | concept + epic |

## Honest fences

- The frozen ceiling `(T-1)/(T+1)`, the per-turn bill `d·(1+0.10·(t-1))`, and the
  cut break-even `n* = 1.25·B / (0.10·(P−B))` are a **first-order model** with a
  constant per-turn delta `d` — deliberately simple so every constant is a flag in
  `cache_curve.py`. They reproduce the measured ceiling; they are a model of the
  dynamics, not a fit to a benchmark. Real deltas vary per turn (tool results
  spike them); the direction of the inversion does not.
- The simplest break-even omits the fresh leg's **on-demand retrieval** cost (a
  relay re-reads the durable store as needed). That is bounded and small relative
  to re-reading a large grown prefix every turn, but it is not zero — the honest
  claim is "a couple of turns," not "instantly free."
- Cache-hit rate is genuinely useful as a **shape diagnostic** and for the
  provider-cost accounting fak already does. The claim here is narrow: it is the
  wrong thing to *maximize*, and reading it as an efficiency objective biases both
  operator and model toward over-long sessions.
- The multipliers (`0.10×` read, `1.25×` 5-minute write) are Anthropic's
  documented hosted-cache economics; other providers share the prefix-match
  premise but differ in the exact knobs. The shape of the inversion is
  provider-independent; the exact numbers are not.

## Reproduce

```sh
python tools/cache_curve.py inversion                 # the hit-vs-bill table + cut break-even
python tools/cache_curve.py inversion --turns 400 --baton 30000
python tools/cache_curve_test.py                      # pins the inversion math (CostInversion)
python tools/ctxcost.py marginal                      # the per-turn dC/dn ledger + rebill fraction
```

---

**Related:**
[frozen-trajectory-cache-cliff](../explainers/frozen-trajectory-cache-cliff.md)
(hit rises with length) ·
[session cache-savings ablation](SESSION-CACHE-SAVINGS-ABLATION-2026-06-29.md)
(the "saved" headline is length) ·
[perpetual sessions / the relay](CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md)
(cut, don't compact) ·
[o1-context-window-economics](../explainers/o1-context-window-economics.md)
(the cost model this sits inside).
