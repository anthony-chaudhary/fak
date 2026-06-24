---
title: "The frozen-trajectory cache cliff: why the prompt-cache hit rate is high, and the scaling laws that take it to 0%"
description: "The high prompt-cache hit rate everyone quotes is purchased with a frozen, append-only trajectory. It is a prefix match, so it stays high only while the harness refuses to touch history. The moment the trajectory becomes flexible — or the workload gains per-turn tool density or cross-agent fan-out — the default cache decays toward 0%. With a runnable demonstrator and the measured ceiling from this machine."
slug: frozen-trajectory-cache-cliff
keywords:
  - prompt caching
  - prefix cache
  - KV cache
  - cache hit rate
  - agentic context
  - multi-agent
  - tool use
  - scaling laws
  - flexible trajectory
date: 2026-06-24
---

# The frozen-trajectory cache cliff

**Short answer.** The high prompt-cache hit rate vendors quote — 90%+, "we cache
almost everything" — is real, but it is *bought with rigidity*. Prompt caching is a
**prefix match**: any byte change in the prefix invalidates everything after it. A hit
rate near 100% is only achievable when the harness **never touches history** — a single,
linear, append-only trajectory. That number is high *because the trajectory is frozen*,
not because caching is free. Two axes bend a single agent's hit toward 0%: making the
trajectory flexible (edit, compact, re-summarize, reorder — exactly what an agent OS does to
manage context), and dense per-turn tool use. A third — cross-agent fan-out — doesn't lower
the percentage but forfeits the shared-setup reuse entirely (0% across the fleet, waste
linear in N). The agent world is moving along all three at once, and the **default** prefix
cache has no answer to any of them.

*For people who operate agent loops or reason about agent-serving economics. By the end
you will know why the headline number is an artifact of one workload shape, the three
scaling laws that bend it to zero, and why this is the case for an addressable, coherence-
checked cache rather than a frozen prefix. Every number here comes from
[`tools/cache_curve.py`](../../tools/cache_curve.py) (deterministic, stdlib-only) and the
real transcripts on this machine via [`tools/session_audit.py`](../../tools/session_audit.py).*

This is the demand-side companion to two existing notes. The mechanics of prefix reuse are
in [`kv-cache-agentic-context.md`](kv-cache-agentic-context.md); the supply-side answer —
how fak *deletes* the reread work the cliff exposes — is
[`SCALING-LAWS-OF-AGENTS-2026-06-19.md`](../notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md). This
note is the part in between: a demonstration that the default cache is heading to zero, so
the deletion problem is not optional.

## The one fact everything follows from

A transformer caches each token's attention Key and Value so it never re-reads earlier
tokens. Attention is causal, so token *i*'s state depends only on tokens *0..i*. Two
requests that share a token-identical prefix produce identical state for that prefix, and
the cache can be reused **up to the first token that differs**. From that token on,
everything is recomputed.

> Reuse is always a contiguous run from token 0. A change at position *N* costs you
> everything at or after *N*. This is the whole game.

The hosted prompt caches build on exactly this — Anthropic's `cache_control` breakpoints,
OpenAI's automatic prefix caching, vLLM's Automatic Prefix Caching, SGLang's
RadixAttention. The render order is `tools` → `system` → `messages`, so the most stable
content has to come first or nothing downstream caches.

## Why the public number is high: it's the frozen ceiling

Walk a single agent that only ever **appends** (the well-behaved loop): system + tools,
then user, then assistant + tool call, then tool result, then assistant + tool call, and
so on. Each model call re-sends the growing transcript, and the previous call's prompt is
an exact prefix of this one. A correct cache reuses the whole prior context and prefills
only the new delta.

Count the tokens. Over *T* turns, each appending a fresh delta *d*, turn *t* re-sends a
prefix of `(t−1)·d` tokens (all cache-read) and pays for its own *d* (cache-create):

```text
read(T)  = Σ (t−1)·d = d·T(T−1)/2
paid(T)  = d·T
hit(T)   = read / (read + paid) = (T−1)/(T+1)   →  rises toward 1
```

So a frozen append-only agent's cache-hit **rises** with length: 82% at 10 turns, 96% at
50, **99% at 200**. That is the ceiling, and it is the number that gets quoted.

The real transcripts on this machine sit exactly there. A fresh 30-day window
(`session_audit.py audit --since-days 30`, 199 sessions) shows **96.6% of all ingested
context served from cache**, an input:output ratio of 126.6:1, and per-session hit of
median 0.894 / p90 0.968. The single biggest session — 205 turns, 32 tool calls — runs at
**99%**. None of that is a caching triumph to celebrate; it is the signature of a harness
(Claude Code) that is *deliberately* a single, linear, append-only trajectory. It freezes
history on purpose, precisely so the prefix stays byte-identical.

A detail that matters for honesty: in that real data, cache-hit **goes up** as tool-call
count goes up — roughly 81% mean for sessions with no tool calls, ~98% for sessions with 16+,
on this 30-day window (reproduce with the two commands at the end) — because in a linear agent
more tool calls just means a longer append-only transcript with more reusable prefix. So
"more tool calls lowers the cache" is **false** for a frozen agent. The decay needs you to
*leave* the frozen single-linear regime. That is the rest of this note.

## Axis 1 — flexibility: the moment you stop freezing history (the product)

The frozen ceiling assumes the harness never edits history. But context management *is*
editing history: compaction summarizes old turns, RSI re-summarizes and re-orders,
context-editing clears stale tool results, a memory layer injects recalled pages near the
top. Every one of these is a change *ahead* of the stable prefix, and by the one fact
above it invalidates everything after the edit point.

Model it as an edit that reaches a fraction `e` back into the cached prefix (append-only is
`e = 0`; rewriting the system prompt is `e = 1`). The surviving reuse is `1 − e`, and since
a lost reuse becomes recompute, the hit is just `(1 − e) · ceiling`:

```text
edit-depth into prefix     cache-hit (from a 99% ceiling)
        0%  (append-only)        99.0%
        5%                       94.1%
       25%  (compact ¼)          74.3%
       50%                       49.5%
      100%  (rewrite the head)    0.0%
```

This is the crux of the product thesis. A flexible trajectory — the thing that makes an
agent OS more than a chat loop — is *fundamentally antagonistic* to a prefix cache. You do
not lose a few points; you lose everything downstream of wherever you touched. A system
that re-plans, compacts, or re-summarizes its own history every few turns cannot keep the
frozen ceiling, full stop. It needs a cache that is keyed on *content and identity*, not on
*position in a frozen prefix* — which is the addressable, coherence-checked cache fak is
built around (see the scaling-laws note).

## Axis 2 — per-turn tool density: the 20-block, 4-breakpoint wall

This is the one that is easy to state wrong, so be precise: it is tool calls **in a single
turn** (parallel or batched tool use), not tool calls across a session.

Anthropic's cache has two hard structural limits. A cache breakpoint walks backward **at
most 20 content blocks** to find a prior entry, and you get **4 breakpoints** per request.
A turn that emits many tool_use/tool_result pairs adds ~2 blocks each. Once a turn's new
content outruns the block budget, the next request's breakpoint can't reach the previous
cache and silently misses on that span.

```text
tool calls in one turn   hit (naive 1 breakpoint)   hit (careful 4 breakpoints)
         5                       99.0%                       99.0%
        10                       90.0%                       99.0%
        20                       47.1%                       99.0%
        40                       24.1%                       96.6%
        80                       12.2%                       48.9%
```

A naive harness that caches only at the end of the message list hits the wall at ~10
parallel calls; a careful one that staircases 4 breakpoints through the new content pushes
it to ~40. Either way it is a real ceiling, and parallel tool use — the direction every
major API is pushing — drives straight at it. The mitigation (intermediate breakpoints) is
a finite budget of 4; it buys a 4× headroom, not immunity.

## Axis 3 — cross-agent fan-out: the concurrency wall

Multi-agent is where the shared-prefix dream breaks hardest. A cache entry is readable only
**after the response that wrote it begins streaming**. Fire *N* agents at once on a cold
shared prefix (the system prompt + fat tool schemas they all share) and none of them can
read what the simultaneous cohort is still writing. The shared prefix is cold-**written** N
times and cross-agent **read** zero times.

```text
agents   cross-agent reuse (default concurrent)   reuse (staggered / cloned)   shared setup re-paid
    2                     0%                              50%                    2× (1 wasted copy)
   10                     0%                              90%                   10× (9 wasted)
  100                     0%                              99%                  100× (99 wasted)
```

Be precise about what this does and does not do — it is the one place the thesis is easy to
overstate. Cross-agent reuse under the default concurrent fan-out is **0% and stays 0%**
regardless of N, and the forfeited re-prefill of the shared setup grows **linearly with N**.
But the per-agent *percentage* does **not** fall with N: each agent re-pays the shared prefix
identically, so the blended hit is flat in N — far below the near-100% the shared/cloned path
reaches, yet not "toward zero." For short agents dominated by a big shared context — a swarm
of small tool-running sub-agents, increasingly the common shape — that flat number is already
low (a 2-turn agent that is half cold prefill sits at ~50%), and the fleet stays pinned there
instead of climbing as a shared prefix would. So fan-out's honest claim is **a reuse win
forfeited, growing with N — not a percentage that craters with N.** The "toward 0%" framing
belongs to axes 1 and 2 (which genuinely bend one agent's hit to zero); fan-out's number to
watch is the 0% cross-agent reuse rate and the linearly-growing waste. Recovering it requires
leaving the default: stagger launches within the TTL, or prefill the prefix once and clone it
bit-identically into every agent — exactly
[pay-the-prefix-once](../../visuals/65-pay-the-prefix-once.svg).

## The compound collapse

The two single-agent axes are independent cache-read fractions, so they **multiply** into one
agent's hit. (Fan-out is deliberately left out of this product — it is a fleet-aggregate
effect, not a single agent's percentage, as the section above explains; folding it in would
be a category error.) A single agent that is *moderately-to-aggressively flexible* **and**
*tool-dense* — the direction the field is actually moving — does not lose a few points; it
falls through the floor:

```text
 99.0%   frozen single linear agent (append-only)            ← the quoted number
 74.3%   + moderate flexibility (compact 25% of prefix)
 35.4%   + tool-dense turns (20 calls/turn, 1 breakpoint)
 11.8%   + aggressive flexibility (compact 75%) + tool-dense
```

Now fan that 11.8%-hit agent out to 100 workers: the default concurrent cache recovers **0%**
of the shared setup across them (a shared/cloned prefix would recover 99%). The fleet pays
this collapsed-cache work 100× over, with no cross-agent amortization.

> The scaling law: a single agent's default cache-hit is `s_flex × s_tools × (T−1)/(T+1)` —
> the frozen ceiling scaled by one survival factor per single-agent axis, each driven toward
> 0 as history gets flexible or turns get tool-dense. Fan-out does not enter that product; it
> multiplies the **cost** of the collapsed state by N while recovering 0% across the fleet.
> So the headline number is not a property of caching; it is a property of the *one workload
> shape* (single, linear, append-only, sparse) that doesn't move — and every direction the
> agent world is moving bends either the percentage (axes 1–2) or the amortization (fan-out)
> toward zero.

## What this means

The frozen-trajectory cache hit is a measurement of how *little* a harness is allowed to
do, dressed up as an efficiency win. It holds for today's single-linear coding agents (the
99% on this machine is real). It does not survive contact with the three things every
serious agent system is adding: flexible context management, dense parallel tool use, and
multi-agent fan-out.

So the choice is not "tune the prefix cache harder." A prefix cache asks one question — *are
these bytes the same?* — and a flexible, fanned-out agent fleet violates that premise by
design. The durable answer is a cache keyed on **content + identity + world-version +
taint** that can reuse a span wherever it legally lives, not only when it sits byte-
identical at the front of a frozen prompt. That is the
[agent coherence kernel](../notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md) thesis, and this
cliff is why it is load-bearing rather than nice-to-have.

## How fak works toward this

The fix is not a better prefix cache; it is the same substrate the
[regenerable-KV plan](../serving/regenerable-kv-plan.md) already names from a different angle.
That plan treats the cliff as a *model rollout* — the nine-axis binding tuple invalidates
every span at once. This note's cliff is the same fragility hit by *trajectory edits* and
*fan-out* instead. One root underneath both: a prefix cache binds reuse to **byte-position in
a frozen prompt**. The durable answer binds reuse to **content + identity** — the text is the
source, the KV is a regenerable artifact — so an edit re-derives only the changed span and a
fan-out clones the shared prefix once instead of paying it N times.

Map each cliff axis to what is already shipped versus the open build:

| Cliff axis | The frozen cache's failure | fak's answer | Status |
|---|---|---|---|
| Flexibility (edit / compact / RSI) | head-mutation invalidates the suffix | suffix-only regen on the live per-turn path (re-prefill only the divergent suffix), plus addressable, bit-exact span eviction (the KV-MMU, `max|Δ|=0`) so an edit removes exactly the touched span — [FAK 404/406](../../LEARNING-PATH.md), [addressable-kv-cache](addressable-kv-cache.md) | **shipped** (per-session, CPU path) |
| Per-turn tool density | the 20-block / 4-breakpoint budget overruns | RadixAttention prefix tree keyed on token-ids (model-agnostic), on by default — [FAK 405](../../LEARNING-PATH.md) | **shipped** |
| Cross-agent fan-out | concurrency wall → 0% cross-agent reuse; prefix paid N× | prefill the shared prefix once and clone it bit-identically into every agent (`max|Δ|=0`) — [pay-the-prefix-once](../../visuals/65-pay-the-prefix-once.svg) | **shipped** (this is the fan-out demo's "shared" path) |
| Durable across rollout / fleet | every binding-axis bump cold-starts the whole fleet | text-as-source regenerable cache; backfill replaces the synchronized cold start | **plan** ([regenerable-KV R1–R8](../serving/regenerable-kv-plan.md)) |

Three of the four axes already have a shipped supply-side answer; the unbuilt part is the
durable, fleet-shared, regenerable tier — sequenced as R1–R8 in the regenerable-KV plan (give
`SourceDigest` a consumer → durable text tier → regen-from-text → eager backfill → two-class
scheduler → cross-regime integrity oracle → fleet quarantine). The honesty fence there
transfers intact: never serve one regime's KV bytes to another — re-derive.

The near-term step this demonstrator points at is its own: **turn the model into a meter.**
`cache_curve.py` *predicts* the survival factors; the offline prefix-divergence analysis from
[FAK 401](../../LEARNING-PATH.md) / [kv-cache-agentic-context](kv-cache-agentic-context.md)
*measures* the flexibility factor on a real transcript (longest-common-prefix reuse per
turn), and `session_audit.py` already reads the provider `cache_read` / `cache_creation`
split. Wiring those into a measured survival-per-axis report makes the cliff falsifiable on a
live workload and supplies the meters the scaling-laws note asks for (reread rate, legal
cache-hit rate, residency pressure). That is the concrete next move — and the cache substrate
it would measure is already the program above.

## Learning points

Three lessons worth carrying past this one doc:

1. **A headline cache number is a workload-shape claim, not a caching claim.** "90%+ cache
   hit" silently asserts *single, linear, append-only, sparse*. Quote the number with the
   shape, or it misleads the instant the shape changes — which is the direction every agent
   system is moving.
2. **Flexibility and prefix caching are antagonistic by construction.** So the answer to the
   cliff is not "tune the cache harder" but a different binding (content + identity +
   world-version) — the addressable / regenerable-KV program, which is the *same* substrate a
   model rollout needs. Two cliffs, one fix.
3. **Keep fleet-aggregate and single-agent quantities apart, and verify a flattering number
   adversarially.** An earlier draft of this demonstrator folded the fan-out reuse rate (a
   fleet metric) into a single agent's cache-hit percentage via an undisclosed constant,
   fabricating the headline collapse. A four-lens adversarial pass (math · mechanics ·
   prior-art consistency · red-team) caught it; the fix reports fan-out as its own metric and
   pins the math with tests so no constant can creep back. The general rule: a demo number
   that is *more* impressive than the honest model is a defect, not a feature — run the
   skeptic before you ship it.

## Reproduce it

```sh
python tools/cache_curve.py curves      # frozen ceiling + the 2 single-agent decay axes
python tools/cache_curve.py fanout      # cross-agent reuse: default vs shared
python tools/cache_curve.py compound    # single-agent collapse, then the fleet fan-out
python tools/cache_curve.py chart       # the decay, at a glance

# the real measured ceiling on this machine:
python tools/session_audit.py audit --since-days 30 --json /tmp/a.json
python tools/cache_curve.py anchor /tmp/a.json
```

## Honest fences

- The frozen ceiling `(T−1)/(T+1)` and each axis's survival factor are a **first-order
  model**, deliberately simple so every constant is a flag in `cache_curve.py`. They
  reproduce the measured ceiling (99% at ~200 turns; 96.6% machine-wide) but they are a
  model of the dynamics, not a fit to a benchmark.
- The 20-block lookback and 4-breakpoint limits are Anthropic's documented hosted-cache
  behavior; other providers' prefix caches share the prefix-match premise but differ in the
  exact knobs. The *shape* of the decay is provider-independent; the exact wall positions
  are not.
- Fan-out is a **fleet-aggregate** metric, not a single agent's hit. Its number is the
  cross-agent reuse rate of the shared setup — 0% under a simultaneous launch on a cold
  prefix, **flat in N** — and the linearly-growing forfeited reuse. The blended fleet hit %
  does *not* fall with N; do not read fan-out as a per-agent percentage that craters. The
  compound collapse therefore multiplies only the two single-agent axes (flexibility, tool
  density); fan-out is reported as the cost-multiplier-with-0%-recovery, never folded into the
  percentage. Staggered launches within the TTL, or a shared/cloned prefix, recover it.
- The per-bucket and per-session figures are EXACT token counts from this machine's 30-day
  transcript window (a different window shifts them); the dollar figures in
  `session_audit.py` use an assumed price table and are not used here.

---

**Related:** [`kv-cache-agentic-context.md`](kv-cache-agentic-context.md) (the prefix
mechanics and the input:output ratio that makes the cache matter) ·
[`SCALING-LAWS-OF-AGENTS-2026-06-19.md`](../notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md) (the
supply-side: deleting the reread under legality checks) ·
[`AGENTIC-CACHING-SOTA-2026-06-19.md`](../notes/AGENTIC-CACHING-SOTA-2026-06-19.md) (the
SOTA cache-layer parity map) · [`pay-the-prefix-once`](../../visuals/65-pay-the-prefix-once.svg)
(the multi-agent clone-once picture).
