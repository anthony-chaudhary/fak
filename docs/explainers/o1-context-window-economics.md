---
title: "The O(1) Context Window: When Sending Less Beats Caching More"
description: "An append-only agent transcript leans on the provider's prefix cache to stay cheap as it grows — but a warm cache demands a byte-immutable prefix, which forbids the per-turn injection, reordering, and replay that observability needs. The O(1)+history alternative keeps history in a lossless store and reconstructs a bounded context each turn, so every step is deterministically replayable and fully observable. Replaying real billed usage shows that observable design is also the cheaper one: the cost crossover is exactly the cache's effective discount, ~12% of the billed prompt."
slug: o1-context-window-economics
keywords:
  - O(1) context window
  - prefix caching
  - prompt caching economics
  - agentic context
  - context reconstruction
  - cache read cost
  - KV cache
  - input output ratio
  - time to first token
date: 2026-06-23
---

# The O(1) Context Window: When Sending Less Beats Caching More

*Who this is for: engineers building long agent loops who are deciding between append-only-plus-prefix-cache and reconstructing a bounded context each turn. Prerequisite: a working grasp of prefix caching (see the companion [KV Cache](kv-cache-agentic-context.md) explainer). You'll come away able to say when the O(1) window is cheaper than a warm cache (it wins below the cache's own discount fraction, ~12%) and why it buys observability and deterministic replay — and how to check both against real billed usage with `tools/ctxcost.py`.*

**Short answer:** An append-only agent loop keeps the whole transcript and re-sends it every turn, staying cheap only because the provider's prefix cache serves the repeated prefix at about one-tenth the price of fresh input. The contrarian design keeps the full history in a lossless store and sends a *bounded, freshly-reconstructed* context each turn. Reconstruction mutates the prefix, so you lose the cache hits — but you send dramatically less in the first place. Whether that nets out cheaper has a clean answer, and you can measure it without spending a dollar: a Claude Code transcript already records the *exact billed token accounting* of the append-only-with-cache regime per turn, so you only have to model the reconstruction against ground truth. The crossover turns out to be a near-identity: **the O(1) window wins exactly while it is smaller than the cache's effective input discount.** On real heavy agent sessions that discount is about 12%, so a reconstructed window under roughly 12% of the full context beats even a perfectly warm cache — and at a realistic 4K-token window it is about 4× cheaper than the warm cache and 28× cheaper than no cache, with a bounded prefill tail instead of an unbounded one. But cost is only the affordability proof. The real reason to keep history in a store and reconstruct each turn is **observability and deterministic replay** — every turn's context becomes a reconstructable function of a lossless store, so you can replay any step and see exactly what the model saw, while a cache, by demanding a byte-immutable prefix, structurally forbids the per-turn injection, reordering, and annotation that observability needs. And these proxy-path numbers are a *floor* on the benefit, not a cap: when fak runs the engine itself the KV cache becomes an addressable kernel object, so the bounded view is reconstructed by reusing and evicting cached spans rather than re-sending a prompt — you get the bounded-context win and keep cache reuse at once, instead of trading one for the other.

## Two ways to carry context

A long agent loop has to get its growing history in front of the model somehow. There are two structural choices.

**Append-only + prefix cache (the status quo).** Keep one ever-growing message list. Every turn appends the new tool result and re-sends the whole thing. This is cheap *only* because of prefix caching: the unchanged prefix is served as `cache_read` at about 0.1× the base input price, and just the new delta is written fresh. The companion explainer [How the KV Cache Changes as Agentic Context Grows](kv-cache-agentic-context.md) walks the mechanism. The catch is structural: the context still grows without bound, you still re-send and re-read all of it every turn, and a single cache miss (a tool slower than the ~5-minute TTL, a head mutation, a cold start) re-prefills the entire prefix at full price.

**O(1) context + lossless history (the alternative).** Keep the full history in a store off to the side (the repo's `internal/ctxplan` is exactly this: a lossless span store plus a bounded planner that selects a working set under a token budget, with anything pruned still demand-pageable). Each turn, reconstruct a *bounded* context — system prompt, the current task, the few spans the turn actually needs — and send that. The resident context is O(1) in the turn count, not O(n). The price you pay is that the reconstructed prefix differs every turn, so the provider's prefix cache almost never hits.

The question the cost sections answer: does the second design's "send much less at full price" beat the first design's "send everything at a cache discount"? But that is the affordability question. The deeper one comes first.

## Why give up the cache at all: observability and replay

A prefix cache only stays warm if the head of the context is byte-immutable. That is not a soft preference, it is the mechanism: any change before a token re-prefills everything after it. So an append-only-plus-cache loop is under standing orders to *not touch* the context — no per-turn injected observation, no reordering, no annotation, no summarizing of old turns, deterministic serialization only. Every one of those is a thing you would do to make a run more observable or to learn from it, and every one of them busts the cache. Cache-friendliness and observability pull in opposite directions, and the append-only design has already chosen cache-friendliness.

The O(1)+history design makes the other choice. Because the context sent each turn is a *deterministic function* of (the lossless history store, the turn's forecast, the budget), three things follow that an opaque growing transcript cannot offer:

- **Every step replays exactly.** Re-run the reconstruction over the same store and forecast and you get the same context, byte for byte. You can replay a whole session, step into any turn, see precisely what the model saw, and re-run it under a different budget or policy to learn what *would* have happened. This is the same deterministic-replay discipline the repo's trajectory-replay work uses to score many policies against one recorded run.
- **Every prune is a recorded decision, not a silent loss.** The planner's audit partitions the *probed* candidate set into selected and elided; a span that is pruned is one demand-page fault away, never destroyed, because the store is lossless (`internal/ctxplan`). So "the agent sees everything that can be observed" is literally true: anything outside this turn's window is a page-in away, still behind the trust gate. Nothing observed is thrown away to fit a window.
- **The failure mode is visible.** The one real risk of a bounded view — a turn whose genuinely-new content does not fit the budget — stops being a silent assumption and becomes a flagged event. The harness `trace` verb emits a per-turn ledger of what was billed, what was new, what was pruned, and a `holds_new_context` flag that goes false on exactly the turns where the window would truncate essential content. On one real 302-turn session at an 8K budget, 12 turns are flagged — you can see them, name them, and decide, rather than discovering a degraded answer after the fact.

```sh
python tools/ctxcost.py trace --budget 8000                     # replay one session, step by step
python tools/ctxcost.py trace --budget 8000 --jsonl trace.jsonl # the full per-turn ledger, for offline learning
```

The inversion the whole approach rests on: append-only + cache optimizes the context to be *held still* so the cache survives, which makes it opaque on purpose; O(1) + history optimizes it to be *replayable and fully seen*, and pays for that by reconstructing each turn — which, as the rest of this doc shows, is not only affordable but usually cheaper.

## Agent-navigable context: dynamic resolution

Observability is passive — you can see every step. The active half is that the agent, or the system, can *navigate* the store: every node in the reconstructed view is a tombstone at some resolution, and one operation moves it up or down.

```
memory(ref, "expand",  budget)   # zoom in: return ~budget more tokens of this node,
                                 #   leaving the still-elided middle as a child tombstone
memory(ref, "contract")          # zoom out: drop the node back to a one-line tombstone
```

Say the planner left a large file read as a one-line tombstone because the forecast did not need it. Mid-reasoning the agent decides it does, calls `memory(ref, expand, 1000)`, and gets a 1,000-token head-and-tail window with the middle elided to a fresh child tombstone. If that is not enough it expands the child, then the child's child, drilling to any depth; when it is done it contracts the branch back to a tombstone and frees the budget. Two drivers share the one operation: the **system** sets each node's initial resolution from the turn's budget and forecast, and the **agent** overrides it from its own reasoning. For any node, up or down, on demand.

Three properties make this more than a convenience:

- **Resident stays O(1); the full history stays reachable.** A wall of tombstones costs almost nothing. On a real session of 17 tool-result nodes holding 8,992 tokens of content, the all-tombstone view is 133 tokens — 1.5% of the full. Expanding four nodes deep brings the resident view to about 4,100 tokens, and contracting drops it straight back to 133. You pay for resolution only where you spend it. This is the same "send dramatically less" the cost sections measure, taken to its conclusion: send tombstones, expand on demand.
- **Nothing is lost; expansion is exact.** The store is lossless, so an expand returns the real bytes, never a lossy summary. A tombstone is the *smallest* rendering of a node, not a deletion — which is why "the agent sees everything that can be observed" is literally true: anything is one budgeted expand away, recursively.
- **The exploration replays.** Every expand and contract is a recorded, budgeted journal event, so an agent's path through the store is deterministic and reproducible. You can replay exactly how it navigated and learn from it; the harness verifies that re-applying the journal reproduces the resident view byte-for-byte.

```sh
python tools/ctxnav.py demo --budget 1000 --steps 3   # watch an agent drill in and zoom back out
python tools/ctxnav.py selfcheck                       # O(1) tombstones, expand/contract, recursion, replay
```

The live path for this is the lossless span store and demand-page in `internal/ctxplan` (a pruned span pages back in through the trust gate), surfaced to the model as a memory tool at the gateway. `tools/ctxnav.py` is the proof harness for the operation itself — the bytes `ctxplan` deliberately does not hold.

## How to validate it honestly

The honest lever is that you do not have to guess the incumbent's bill. A Claude Code transcript records, for every assistant turn, the provider's own usage accounting:

```
usage = { input_tokens,                  # fresh, full price (1.0x)
          cache_creation_input_tokens,   # cache write (1.25x for 5-min TTL)
          cache_read_input_tokens,       # cache hit  (0.1x)
          output_tokens }                # generation (5.0x base input)
```

The full context the model saw that turn is the sum of the three input fields. The append-only-with-cache bill is therefore *measured*, not modelled. The harness `tools/ctxcost.py` replays each turn under four regimes, in base-input units (fresh input = 1.0×, which cancels in any ratio and converts to dollars by the model's input price):

- **A — naive / no cache.** Re-send the full prompt at full price every turn. The "random API with no usable prefix cache" world. Cost grows with the square of the turn count.
- **B — append-only + cache.** The measured real bill: `fresh·1.0 + write·1.25 + read·0.1`.
- **C — O(1) reconstruct, no cache.** Send a bounded window `min(prompt, budget)` at full price every turn. Linear in the turn count.
- **D — O(1) reconstruct + stable cached head.** Keep a byte-stable head (system + tools) cached at 0.1× after a one-time write, reconstruct only the tail fresh.

Output tokens are held identical across all four regimes. This prices the *bytes you send*, not the quality of what comes back — see the limits section. Token counts for A and B are the provider's exact billed usage; only the reconstruction budget for C and D is a model of the bounded planner.

## What the replay shows

Driven over the 20 heaviest real sessions on one machine (3,501 turns after de-duplicating streaming snapshots). The average **prompt billed per turn is 283,680 tokens** — but about 98% of that is `cache_read` of the same growing prefix, re-counted every turn. The *genuinely new* context per turn (the uncached fresh + cache-write delta) is only about **4,451 tokens**. That gap is the whole opportunity: the append-only loop bills a quarter-million-token prompt each turn to carry a few thousand tokens of new information.

| regime | $ at Opus input rate | mean TTFT (prefill tok) | max TTFT (prefill tok) |
|---|--:|--:|--:|
| A · naive / no cache | $5,071 | 283,680 | 671,928 |
| B · append-only + cache (measured) | $691 | 32,374 | 397,588 |

The warm cache is doing real work: it cuts the bill 7.3× versus naive, because 98.4% of all input tokens are served as `cache_read`. (Counting a cache read at the same 0.1× for prefill time as for dollars, the cache's mean TTFT is ~32K prefill-tokens, not the near-zero a "cache-read is free" model would print.) Now the O(1) reconstruct, swept by per-turn budget:

| budget | C cost | C vs B | C vs A | C max TTFT | D vs B |
|--:|--:|--:|--:|--:|--:|
| 4,000 | $175 | **0.25×** | 0.035× | 4,000 | 0.21× |
| 8,000 | $245 | 0.35× | 0.048× | 8,000 | 0.31× |
| 16,000 | $384 | 0.56× | 0.076× | 16,000 | 0.51× |
| 32,000 | $663 | 0.96× | 0.131× | 32,000 | 0.91× |

A 4K reconstructed window costs a quarter of the warm cache and one-twenty-eighth of no cache. The win shrinks as the window grows, and the break-even is sharp:

**The crossover is the cache's own discount.** C beats B exactly when the per-turn window drops below **33,616 tokens — 11.8% of the average billed prompt.** That fraction is not a fitted curve; it is a near-identity. The warm cache's *effective input multiplier* on this corpus (its input bill divided by the full-price bill on the same tokens) is **0.118**, and the crossover fraction is **0.1185**. They match because for a long session C saturates at `budget` per turn while B costs `prompt × effective_multiplier` per turn, so they cross at `budget / prompt = effective_multiplier`. In words: **a freshly-reconstructed window beats a prefix cache whenever the window is smaller than the fraction the cache actually discounts you to.** Anthropic's warm cache discounts heavy sessions to about 12%, so the O(1) window has to fit in about 12% of the billed prompt — roughly 34K tokens against a 284K-token prompt. That is about **8× the genuinely-new context per turn** (≈4,451 tokens), so the bounded planner has real room to add relevant history on top of the new tool result, not just barely fit the latest turn.

This is corpus-robust, not a quirk of the heaviest sessions. Re-run over 100 sessions (15,003 turns, lighter ones included) and the warm crossover is **12.2%**, with C@4K still at 0.28× of B.

### Sensitivity: the crossover widens as the cache degrades (a B-only dial)

The 12% figure is the *best case for the cache* — a perfectly warm one. Real caches are not perfectly warm: policy events like TTL expiration (e.g., a tool call slower than the ~5-minute cache TTL) evict the prefix, and the next turn re-prefills it all at full price. Model that as a fraction of turns forced cold. Note this degrades regime **B only** — C and D hold no provider cache, so eviction never touches them; their cost is byte-identical across every scenario. So the table below is a *sensitivity* dial, not a second empirical measurement: it shows the crossover rising because B's bill rises against an invariant C, which is definitional, not a discovery. The eviction fraction is illustrative, not a measured tool-latency-versus-TTL distribution on this corpus.

| cache scenario (B degraded) | crossover (window beats B) | as % of billed prompt |
|---|--:|--:|
| warm (no eviction; B fully measured) | 33,616 tok | 11.8% |
| 1 turn in 4 cold | 98,294 tok | 34.6% |
| 1 turn in 2 cold | 168,930 tok | 59.6% |
| no usable cache ("random API") | every budget | wins outright |

The reading is directional and robust even if the exact fractions are illustrative: the worse the cache performs — slow tools, multi-tenant eviction, head mutation, a provider with weak or no prefix caching — the larger the reconstructed window can be and still win. On a provider with no usable prefix cache at all, the O(1) window wins at every budget, because there is no discount left to beat: it is just "send 4K" versus "send 284K," and the only question is how much you save (here, 28×). That last case needs no eviction model at all — it is regime A versus C, both measured.

### Latency: a bounded prefill tail instead of an unbounded one

Time-to-first-token tracks the prefill tokens. A cache read is cheaper than recompute but not free — reading KV from memory costs bandwidth — so the proxy charges it 0.1× the time of a fresh token, the same factor it costs in dollars (charging it zero, the "cache-read is free" view, is what makes a warm cache look near-instant; it is not). On that consistent basis the append-only cache's mean TTFT is about **32,374 prefill-tokens**, and its tail reaches **397,588** on the worst turn. That worst turn is *not* an eviction miss — in the warm scenario there are none by construction. It is a single large cache *write*: an oversized ~393K-token tool-result delta prefilled once and recovered as `cache_read` the next turn. The point is that the append-only delta is *unbounded* — one fat tool result can prefill hundreds of thousands of tokens in a turn.

The O(1) window caps that: at a 4K budget no turn ever prefills more than 4,000 tokens. The honest framing is bounded-versus-unbounded, and it carries the same caveat as the cost result. C's 4,000-token worst turn is bounded only because it holds about 0.9% of that turn's 440K-token context; calling that "faster" is a latency win only if the bounded window is a faithful substitute (the same faithfulness assumption the cost numbers rest on). What is unconditional is the shape: the O(1) regime's prefill, and therefore its TTFT, has a hard ceiling at the budget, and it never approaches the context-window limit.

## When fak owns the cache: both wins at once

Everything above is the **proxy** story — fak in front of a black-box API, where the wire prompt *is* the cache key. That fusion is what forces the choice: to bound the context you re-send a smaller prompt and the provider re-prefills it (regime C, bounded but re-prefilled), or you keep appending and ride the cache (regime B, cached but unbounded). You cannot have both, because the only handle you have on a black-box cache is the prefix you send.

When fak runs the engine itself, the KV cache stops being the provider's opaque prefix and becomes an addressable kernel object — and the choice dissolves. The three things a black-box API fuses into one — what is **on the wire**, what is **cached**, and what is **attended** — become separate axes:

- **Reconstruction is a cache operation, not a re-prefill.** To shrink the resident context you do not re-send a smaller prompt; you keep the cached run and *evict* the pruned spans. fak's eviction is bit-exact and cheap: it drops the span's rows from every layer and re-derives each shifted survivor's key from the kept pre-RoPE `Kraw` in a single rotation — no forward pass, no re-prefill (`internal/model/kvcache.go` `Evict`). The survivors' KV is reused as it sits. So the bounded view costs the new tokens you add, not the whole window you keep.
- **Prefill drops to the floor.** The kernel prefills only the genuinely-new content each turn; everything else is reused from cache. On the same 20 sessions the irreducible new-information floor is about **15.3M tokens — 1.5%** of what the naive regime re-prefills. Regime C, which cannot address the cache, re-prefills its window every turn (about **1.8× that floor at an 8K budget**) because it re-sends history it cannot reuse. The kernel deletes that penalty. (A warm provider cache also prefills roughly the floor — so the kernel matches B on prefill rather than beating it; the kernel's edge over B is the next two points.)
- **Decode stays bounded.** Each generated token attends over the resident KV. B's resident set is the unbounded growing prefix; the kernel's is the bounded working set, which cuts the decode-attention work to a few percent of B's at a small budget. A black-box cache cannot bound this, because it cannot evict — only append.
- **Eviction is governance, not just economy.** The same bit-exact removal is the durable point of the whole design: a poisoned tool result or an expired secret can be removed from the middle of a kept run and *proven* gone (`max|Δ| = 0` against a run that never saw it). That is the half of "addressable" that does not commoditize — see [addressable-kv-cache.md](addressable-kv-cache.md).

The honest bounds on this regime are real and the harness labels them as such. It is a **compute** axis (prefill plus attention FLOPs), not the API-dollar bill of the proxy regimes, so the harness reports it separately and never prints a fused "E is N× cheaper than B in dollars." And it is **projected, not measured live**: the KV kernel is dormant on the live proxy loop, the bit-exact eviction is proven on a synthetic model (`internal/kvmmu`) rather than driven by a bounded-reconstruct serve loop that does not yet exist, the cheap exact case today is write-time eviction or append-after-evict (the general mid-stream bit-exact reselect is the audited non-prefix-reuse research item), and linear-attention layers cannot evict a span at all and fail closed. Treat regime E as the *ceiling* that owning the cache buys — the reason the proxy-path crossover is a floor on the benefit, not a cap. The harness emits it under a "When fak owns the cache (projected)" block so it can never be read as a shipped number.

## What this does and does not prove

This is a cost-and-latency result, held to a strict honesty line:

- **Output is held identical across regimes, so this prices bytes sent, not answer quality.** The load-bearing assumption is that a faithful bounded window lets the model produce the same turn it would have with the full transcript. That is the separate *faithfulness* axis, and it is exactly what `internal/ctxplan` exists to establish (a pruned span is a demand-page fault, not a lost fact) plus a task-success eval. A cost win on a window that breaks the agent's reasoning is not a win. Do not read this doc as a quality claim.
- **A and B are exact billed usage; C and D are a model.** The reconstruction budget `min(prompt, budget)` is the *total* context sent, which is the right quantity for cost — but it assumes the planner can actually fit the system prompt, tool definitions, the latest tool result, and enough relevant history into that budget. Oversized single results have to be windowed to a recoverable pointer (the `tools/ctxwin.py` lever) for the budget to be achievable.
- **"Billed prompt" is not "distinct context."** The 283,680-token average is the prompt *billed* per turn, about 98% of which is `cache_read` of the same growing prefix re-counted each turn. The cross-turn sum of billed prompt (993M tokens) is a token×turns area under that prefix, not a distinct-context size — the largest single context ever held across all 20 sessions sums to about 8M tokens. The crossover fraction is measured against the per-turn billed prompt, which is the right regime-versus-regime comparison; do not read the 993M or the 284K as "context the model had to understand."
- **The eviction sweep is a B-only sensitivity dial, not a measurement.** Forcing turns cold degrades regime B only; C and D have no cache to lose. The widening crossover is therefore definitional, and the eviction fractions are illustrative — the one eviction-free, fully-measured comparison is regime A (no cache) versus C, which the O(1) window wins outright.
- **The store is not free.** The O(1) design needs a lossless history store and a planner, and a demand-page fault on a wrongly-pruned span costs a round trip. Those are real costs; they are small next to deleting 88% of a 284K-token re-send, but they are not zero.
- **TTFT is a prefill-token proxy, not milliseconds.** Decode time (identical across regimes) is excluded, and a cache read is charged 0.1× the prefill time of a fresh token — the same factor it costs in dollars, not zero. A large warm-cache TTFT spike is usually a big cache *write* delta, not an eviction miss. The robust latency claim is the bounded-versus-unbounded *shape*, not the exact mean.

## Reproduce it

```sh
python tools/ctxcost.py selfcheck                 # anti-overclaim: C==A at full budget, B<=A, crossover in range
python tools/ctxcost.py replay   --scenario warm  # full per-regime cost + latency table
python tools/ctxcost.py crossover                 # the crossover budget across cache scenarios
```

`selfcheck` is the honesty gate: on a synthetic perfectly-warm session it asserts the model cannot fabricate a saving (at a budget at or above the largest prompt, the reconstruct is a no-op and C equals A exactly), that a cache never costs more than no cache, and that the thesis is a *crossover* — a small window beats the warm cache and a large one loses to it — rather than an unconditional win.

## Frequently asked questions

**Doesn't reconstructing the context every turn just throw away the cache savings?**
Yes, and that is the point of measuring it. You lose the 0.1× cache-read discount, but you send a window roughly 70× smaller than the billed prompt (4K versus 284K). On input alone that is about 8× cheaper; including the output tokens both regimes pay equally, the total bill lands about 4× cheaper at a 4K window. The break-even is the cache's own discount fraction: if your window is smaller than the fraction the cache discounts you to (~12% here), fresh-and-small wins.

**Why is the crossover the same as the cache's effective discount?**
For a long session the reconstructed window saturates at the budget, so C costs about `budget` per turn at full price, while the cache costs about `prompt × effective_multiplier` per turn. They cross at `budget/prompt = effective_multiplier`. The effective multiplier is just how cheap the cache made your input — about 0.12 on warm heavy sessions, because ~98% of input is cache-read at 0.1×.

**Does this depend on which model I use?**
No, for the crossover. Output is 5× input for Opus, Sonnet, and Haiku alike, and the crossover is set by the cache multipliers (read 0.1×, write 1.25×) and the workload shape, not the per-token price. The dollar figures scale with the model's input price; the ratios do not.

**When is the append-only cache still the right call?**
When the cache stays warm and the window cannot be made small. If your reconstructed context would have to be a large fraction of the full transcript anyway — because the task genuinely needs most of the history every turn — you are above the crossover and the cache wins. The O(1) design pays off precisely when most of the transcript is *not* needed on most turns, which is the common shape of long tool-use loops (92% of context is tool results, most of them stale).

**What about a provider with no prefix caching?**
Then there is no discount to beat. The O(1) window wins at every budget, by the full size ratio — 28× cheaper at a 4K window on this corpus. This is the case the contrarian design is most obviously right for: a "random API" where you cannot rely on the cache at all.

---

*Token counts for the naive and cached regimes are the provider's own billed `usage` accounting from real transcripts, not estimates. The reconstruction budget is a model of the bounded planner (`internal/ctxplan`), and output tokens are held constant — this is a cost-and-latency result, not a quality claim. The crossover-equals-effective-discount identity is verified numerically (0.118 vs 0.118 on 20 sessions, 0.121 vs 0.122 on 100). Harness and self-check: `tools/ctxcost.py`.*

**Related:** [How the KV Cache Changes as Agentic Context Grows](kv-cache-agentic-context.md) — why append-only + prefix cache is cheap, and what erodes the hit rate. The prefix-stable cousin of this work is `tools/ctxwin.py`, which *halves* the window while keeping the prefix byte-stable so the cache survives; this doc explores the opposite bet, where you give up cache stability to send far less.
