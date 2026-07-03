---
title: "Long-session economics: why a growing agent transcript costs more"
description: "Why a long agent session re-sends its transcript each turn, why the cache discount only holds while the prefix is byte-identical, and how fak keeps it alive."
slug: long-session-economics
keywords:
  - long agent session cost
  - prompt cache economics
  - byte-identical prefix
  - KV cache reuse
  - agent token cost
  - context re-send
  - cache discount
date: 2026-07-03
---

# Long-session economics

*Who this is for: anyone paying for a long agent run — a Claude Code session, a
coding agent, any loop that keeps talking to a model — who has watched the per-turn
cost climb and wants to know why, and what actually stops it. No `fak` internals
required: only the one fact that an agent re-sends its whole conversation every turn.*

**Short version.** A long session gets expensive because every turn re-sends the
entire transcript so far. Providers soften that with a **prompt cache** — but the
discount only survives while the front of your prompt is *byte-for-byte* what they saw
last time. Anything that rewrites the prompt to shrink it (the obvious fix) breaks that
match and re-charges the whole thing. `fak` keeps the discount alive by copying the
cacheable prefix through untouched — a `memcpy`, never a re-serialize — and dropping
old turns only where doing so leaves the prefix intact. What it *guarantees* is the
byte-identical prefix; whether the provider then reuses the cache is the provider's
call, and `fak` reports the provider's own number rather than claiming the win.

## Why the bill grows with the work

An agent has no memory between turns. So each turn the client re-sends everything: the
system prompt, every prior question, every tool call, every tool result. A short chat
barely notices. A session that has grown to 100k tokens re-sends 100k tokens just to
ask its next question — and it does that again next turn, and the turn after. The input
you pay for on turn *N* is roughly the sum of everything that happened in turns 1
through *N−1*. Across a whole session that is quadratic in the length of the work: the
longer the session runs, the more each additional turn costs.

## The discount, and the word that makes or breaks it

Providers soften the re-send with a **prompt cache**. If the front of your prompt is
exactly what they served last time, they charge a fraction for that cached span instead
of full price. On an Anthropic-style cache the cached prefix reads back at roughly a
tenth of the input rate.

The load-bearing word is *exactly*. A prompt cache is **prefix-addressed**: it reuses a
contiguous run starting at the first token, and only while that run stays byte-for-byte
identical to what you sent before. Change one byte at position *N* and everything from
*N* onward falls out of the cache and is recomputed at full price. (The mechanism, and
why every production cache — vLLM, SGLang, OpenAI, Anthropic — works this way, is the
subject of the [addressable KV cache](addressable-kv-cache.md) explainer.)

## Why the obvious fix backfires

The natural way to shrink a long transcript is to **summarize** the old turns and send
the summary in their place. It feels thrifty and it costs more.

Summarizing rewrites the body of the prompt. A rewrite reorders bytes. Reordered bytes
break the byte-for-byte prefix match the cache depends on — so the provider stops
discounting and re-charges the whole prompt at full price. You did work to save money
and the bill went *up*.

## What keeps the discount: splice on the original bytes

`fak` takes the other route. Instead of rewriting the prompt to make it smaller, it
**drops** old middle turns and splices the remaining bytes back together. The cacheable
front of the prompt is copied through untouched — a `memcpy` of the original bytes,
never a re-marshal — so the provider's cache prefix still matches and the discount
holds. It only drops a span when doing so leaves the prefix byte-identical; on any
ambiguity it does nothing and forwards the original prompt, so it never breaks a turn.
There is a deliberately conservative case: if the span it would drop is itself marked
`cache_control` (already provider-warm), `fak` refuses to drop it, because a smaller
prompt is not automatically cheaper than one already served from cache.

The practical side of this — the one `fak guard` flag, when it fires, what it does and
does not promise — is in the sibling explainer
[Long sessions: shed history, keep the cache hit](long-sessions-keep-the-cache-hit.md).

## The honest boundary: fak relays the provider's number

Here is the fence that matters, stated plainly. **`fak` guarantees exactly one thing:
the prefix it ships is byte-identical to what you sent.** That makes a cache hit
*possible*. It does not *force* one. Whether the provider actually reuses the cache —
versus a TTL expiry, an eviction, or your client moving its own cache breakpoint — is
the provider's decision, made on the provider's side.

So `fak` does not claim the saving. It **relays the provider's own number**: `/metrics`
exposes `fak_gateway_compaction_*`, putting the tokens `fak` shed (what it sent) next to
the provider's reported `cache_read` (what actually came back discounted), and the
`fak guard` exit line summarizes both. If `cache_read` is low while the prefix was
byte-identical, the miss is provider-side and you see it either way, rather than
overpaying silently. The guarantee is `fak`'s; the reuse number is the provider's, and
the two are reported separately.

## A worked cost example

Take a session that has grown to a **100,000-token** prefix and consider the cost of one
more turn. Use round rates for the arithmetic: **1×** per input token at full price, and
an Anthropic-style **0.1×** for a token read back from the prompt cache — a 10× discount
on the cached span. (This is illustrative arithmetic to show the shape, not a price
quote; substitute your model's real per-token rates.)

| One more turn on a 100k-token prefix | Prefix billed at | Relative input cost |
|---|---|---|
| No cache (re-send at full price) | 100,000 × 1.0 | **100,000** |
| Summarize to 20k, sent uncached (prefix match broken) | 20,000 × 1.0 | **20,000** |
| Cache hit on the full prefix (`fak` keeps it byte-identical) | 100,000 × 0.1 | **10,000** |

The summarize row looks like the cheap one until you remember it *broke* the prefix
match: it pays full price for its 20k **and** forfeits the discount on everything it
kept, so every later turn re-establishes the cache from scratch. The cache-hit row keeps
the whole 100k prefix but pays a tenth for it — cheaper than the rewrite, and it stays
cheap next turn because the prefix is still byte-identical. Keeping the discount beats
shrinking the prompt.

The witnessed version of this at fleet scale: on a 50-turn × 5-agent run
(Qwen2.5-1.5B, Apple M3 Pro), reuse did **~60.3× less work than the naive re-send loop**,
and **~4.1× less than a *tuned* warm-cache stack** — the honest, few-fold headline, not
the naive multiplier alone. The naive arm's ~19 hours is modeled from the prefill cost
curve (validated within ~0.4%), not run live; the artifact and its fences are in
[`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md).

## The one line to keep

A long session is expensive because it re-sends everything, and the prompt cache is the
only real relief — but the cache pays out *only* while the prefix is byte-identical.
`fak` protects that byte-identity by splicing on original bytes instead of rewriting the
prompt, guarantees the byte-identical prefix, and relays the provider's own reuse number
rather than claiming a saving it cannot force.

## See also

- [Addressable KV cache](addressable-kv-cache.md) — what "prefix-addressed" means, and the one thing `fak` does that shipped engines don't.
- [Long sessions: shed history, keep the cache hit](long-sessions-keep-the-cache-hit.md) — the practical `fak guard` flag and its break-even rule.
- [The frozen-trajectory cache cliff](frozen-trajectory-cache-cliff.md) — why a long trajectory's cache-hit rate rises by construction, and why that is not the same as getting cheaper.
- [O(1) context-window economics](o1-context-window-economics.md) — the accounting model underneath the per-turn cost.

*Last updated: 2026-07-03*
