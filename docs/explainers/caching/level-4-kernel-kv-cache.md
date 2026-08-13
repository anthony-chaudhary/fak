---
title: "The kernel-owned KV cache"
description: "One level under the provider abstraction: the transformer KV cache. Prefix reuse, bit-exact span eviction (max|Δ|=0), and how a vLLM sleep silently drops it."
slug: level-4-kernel-kv-cache
keywords:
  - KV cache
  - prefix caching
  - addressable KV cache
  - bit-exact eviction
  - RadixAttention
  - vLLM sleep mode
  - prompt injection quarantine
  - transformer prefill
date: 2026-07-10
---

# The kernel-owned KV cache

*You are on **Level 4 of 5** of the [fak caching ladder](README.md).*

> **Audience.** Engineers who know the provider-level view from Levels 1–3 and want the
> layer underneath. By the end you'll be able to explain why cache reuse is always a
> prefix in a transformer KV cache, and what kernel ownership adds: bit-exact interior
> span eviction (`max|Δ| = 0`) and honest evidence when an upstream drops the cache.

> **Short answer.** The provider "prompt cache" from Levels 1–3 is, one layer down, a
> **transformer KV cache**: the key/value vectors each token produces during prefill.
> Reuse is always a **prefix** — a contiguous run from token 0 — because attention is
> causal. When fak *owns* that cache as a kernel object it can do something no shipped
> serving engine does: name an interior span, evict it, and leave the cache
> **bit-identical to one that never saw it** (`max|Δ| = 0`). And it knows when an
> engine underneath has silently thrown the whole cache away.

## What the cache actually is

During **prefill** the model processes the whole prompt at once and, for every token,
computes two vectors — a **key** and a **value**. Later tokens attend back over those
saved K/V vectors so the model never re-reads earlier tokens while decoding. Saving them
is the KV cache; **decode** then emits one token at a time, each attending over all
cached K/V. Prefill is the expensive part for long contexts, and it is exactly the part
a cache deletes. (`docs/explainers/kv-cache-agentic-context.md`.)

The load-bearing fact is that attention is **causal**: token *i*'s K/V depends only on
tokens *0..i*. So two requests that share a token-identical prefix produce *bit-for-bit
identical* K/V for that prefix — you splice in the cached copy and prefill only the
suffix. This is precisely what "prefix caching" is (vLLM Automatic Prefix Caching,
SGLang RadixAttention, and the hosted prompt caches). Reuse is only ever a run from
token 0; a change at position *N* costs everything at or after *N*. fak proves this
directly: `TestKVPrefixReuseMatchesRecompute` checks prefix reuse against a full
recompute to `max|Δ| = 0` with identical argmax — holding a fixed model, tokenizer,
precision, serializer, and position scheme. (`internal/model/kvreuse_test.go`;
`docs/explainers/addressable-kv-cache.md`.)

## "Addressable" is four different words

The provider caches are prefix-addressed and nothing more. "Addressable" gets used for
four distinct capabilities, and keeping them apart is the whole game
(`docs/explainers/addressable-kv-cache.md`):

1. **Prefix-addressed** — reuse the longest cached run from token 0. Append-only. Every
   production engine ships this; it is saturated.
2. **Span-addressed** — name an interior span `[i, j)`, operate on it (evict, isolate),
   and keep the rest correct. Production does *not* expose this as a clean exact
   operation.
3. **Content-addressed** — a piece of state is named by the hash of its bytes, so a tool
   result is a `Ref` into a CAS blob store; this is the cross-model/cross-session layer.
4. **Queryable-context** — ask for a working set under a budget and a policy, with a
   verdict per piece (HIT / FAULT / RECOMPUTE / REFUSE / ABSTAIN).

fak's contribution is #2 (exact, and as a security primitive), #3, and an early,
honestly-bounded #4. For agentic workloads the payoff of #2 is **governance**: eviction
by *policy*, not just by memory pressure. The cache-pressure LRU that SGLang and vLLM
run evicts on recency when memory is tight; `radixkv.EvictNode` adds span-exact,
provable eviction of a named prefix because a verdict said so
(`docs/explainers/addressable-kv-cache.md`).

## Eviction and the exactness condition

Why does a middle span not lift out cleanly? Every token's K/V also encodes its
*position* (via RoPE) and, at deeper layers, what it attended to. Drop the bytes and the
easy half is done — the hard half is the **survivors**: every token after the cut had
its key rotated at its **old** absolute position and now sits at a new one. fak keeps the
pre-RoPE key (`Kraw`) and re-rotates each survivor **once** at its new position; one
clean rotation is exact. llama.cpp's K-shift *composes* a second rotation and drifts
~1e-6 (enough to flip a greedy token); vLLM and SGLang store post-RoPE keys only, so an
exact middle removal means recomputing the tail. (`internal/model/kv.go`, `applyRopeRow`;
`docs/explainers/addressable-kv-cache.md`.)

The precise, tested claim is about **bit-exactness**, and it is two distinct witnesses —
do not conflate them:

| Witness | What it compares | Result |
|---|---|---|
| evict-vs-never (`max|Δ|`) | fak's post-evict logits vs a fak run never shown the span | **`max|Δ| = 0`** — bit-identical, every logit, not just argmax |
| poison-vs-never (control) | keeping the span vs never-saw | `max|Δ| > 0` (≈ 0.326) — non-vacuous, so the zero is a *real* erasure |
| eviction reposition | evict-then-continue vs full recompute | `max|Δ| = 0` (`TestKVQuarantineEqualsNeverSaw`) |
| vs HuggingFace | fak's forward pass vs HF never-saw run | argmax-exact; logits track HF at `max|Δ| ≈ 4.4e-5` |

So: the *within-fak* eviction guarantee is bit-identical (`max|Δ| = 0`); the numerics of
the synthetic model itself are oracle-checked against HF at `≈ 4.4e-5` (argmax-exact).
This is what turns quarantine from "don't show the model the poison" into "the model is
mechanically incapable of attending to it" — one verdict, two enforcement media.

> **Honest scope.** This is witnessed on a synthetic model in `internal/kvmmu`, whose
> numerics are separately checked against HuggingFace. The primitive (`Evict`, re-RoPE,
> ledger renumber) is done and tested; the live `fak agent` HTTP loop does **not** drive
> the in-kernel engine yet, so today's live path quarantines at the byte layer.
> Attention-state eviction is the proven next rung, not a shipped default. It is a
> guarantee bought on a **different axis** — memory (each radix node holds a full-prefix
> KV copy), not throughput. fak is not faster than a tuned GPU engine and does not claim
> to be; the comparable axis is hit rate (77–88% across few-shot/chat/ToT/agents).

## When a serving engine LOSES the cache

Owning the cache also means knowing when someone else dropped it. An external vLLM worker
can go dormant in ways that silently discard the KV, per vLLM's own `sleep_mode.md`:
**sleep level 1** offloads weights but *forgets the KV cache*; **sleep level 2** forgets
both; a **prefix-cache reset** forgets the KV while the engine stays up; a **pause**
keeps the KV resident. If fak keeps reporting a prefix "warm" while the engine slept, a
resume routes as a **cache hit onto memory the engine no longer holds** — a false warm
hit — and a fleet view reads "healthy" for an asleep worker
(`docs/explainers/vllm-lifecycle-cache-loss-bridge.md`).

fak lowers every dormancy action to one **cache-loss witness** with a closed three-value
KV vocabulary — `preserved` / `forgotten` / `unknown` — where `unknown` is first-class
and fails closed (treated as potentially lost, never collapsed to `preserved`)
(`internal/cachemeta/sleep_witness.go`, `WitnessDormancy` / `KVDisposition`):

| Action | Phase | KV | Warm demoted | Serving |
|---|---|---|---|---|
| `pause` | `paused` | `preserved` | no | no |
| `sleep` (level 1/2) | `sleeping` | `forgotten` | yes | no |
| `sleep` (no level) | `error` | `unknown` | yes | no |
| `reset` | `serving` | `forgotten` | yes | yes |
| `wake` | `serving` | `unknown` | yes | yes |

`wake` is deliberately not a re-warm: a resume may only claim a warm prefix hit against a
possibly-slept worker after a fresh `BlockStored`/cache signal re-proves residency. Note
the honest status: the witness kernel and its decision tests are **shipped**
(`internal/cachemeta`); the engine/CLI wiring (the `/sleep`·`/wake_up` adapter, the
`session ls --fleet` surface) is **not yet wired**.

## The frozen-trajectory cache cliff

This is why kernel ownership matters, not just for correctness but for the whole agent
economy. The 90%+ hit rate vendors quote is real but **bought with rigidity**: a prefix
match rises toward 1 (`(T−1)/(T+1)` → 82% at 10 turns, 96% at 50, 99% at 200) *only*
while the harness never touches history. Three axes bend it back down
(`docs/explainers/frozen-trajectory-cache-cliff.md`):

- **Flexibility** — compaction, RSI, context-editing, injected memories all edit *ahead*
  of the stable prefix and invalidate everything after the edit point. An edit reaching a
  fraction `e` back survives `1 − e`: rewrite the head and you hit 0%.
- **Per-turn tool density** — Anthropic's cache walks back at most **20 content blocks**
  and gives **4 breakpoints**; dense parallel tool use in one turn overruns the budget.
- **Cross-agent fan-out** — a cache entry is readable only after the response that wrote
  it begins streaming, so *N* agents on a cold shared prefix recover **0%** cross-agent
  and re-pay the shared setup *N* times (waste linear in N; the per-agent percentage does
  not itself crater).

A prefix cache asks one question — *are these bytes the same?* — and a flexible,
fanned-out fleet violates that premise by construction. The durable answer keys reuse on
**content + identity**, so an edit re-derives only the changed span and a fan-out clones
the shared prefix once. That is the addressable / regenerable-KV program above; three of
its four axes already have a shipped supply-side answer (per-session CPU path), the
durable fleet-shared tier is a plan.

## Try it

```bash
# The offline, bit-exact span-eviction demo (no key, model, GPU, or network):
./examples/addressable-evict/run.sh
# prints:  max|Δ| evict-vs-never = 0.000e+00   (poison-vs-never ≈ 3.257e-01 control)

# Prove prefix reuse == full recompute (max|Δ| = 0):
go test ./internal/model -run TestKVPrefixReuseMatchesRecompute

# The vLLM cache-loss witness decision tests:
go test ./internal/cachemeta

# Your live managed-cache posture (Level 2–3 wire lever), for contrast:
fak manage --managed-cache on -- claude  # witness metric: fak_gateway_cache_ttl_upgrade_total
```

## See also

- [Level 3 — Cache economics & the wire](level-3-cache-economics-and-the-wire.md) — the
  provider-side prefix cache, the read/write multipliers, and the money.
- [Level 5 — The caching frontier](level-5-the-caching-frontier.md) — the SOTA, the
  non-prefix-splice research, and the open edges.
- [The fak caching ladder](README.md) — the ladder index.
- [Addressable KV cache](../addressable-kv-cache.md) · [The addressable KV cache in 5 minutes](../addressable-kv-cache-in-5-min.md) — the deep internals this rung compresses.
- [How the KV cache changes as agentic context grows](../kv-cache-agentic-context.md) — prefix mechanics and the input:output ratio.
- [The vLLM lifecycle cache-loss bridge](../vllm-lifecycle-cache-loss-bridge.md) · [The frozen-trajectory cache cliff](../frozen-trajectory-cache-cliff.md) — the two failure modes above, in full.

---

*[← Level 3](level-3-cache-economics-and-the-wire.md) · [full ladder](README.md) · next → [Level 5](level-5-the-caching-frontier.md)*
