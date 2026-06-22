---
title: "Addressable KV Cache: What Production Actually Offers, and What It Doesn't"
description: "Every production prefix cache — vLLM, SGLang, OpenAI, Anthropic — is append-only and prefix-addressed: reuse is a run from token 0, and a change at position N costs you everything after N. fak owns its KV cache as a kernel object, which lets it do one thing no shipped engine does: remove a tool result from the MIDDLE of a kept sequence, bit-identically to never having seen it. That is the underdiscussed half of 'addressable'."
slug: addressable-kv-cache
keywords:
  - KV cache
  - prefix caching
  - RadixAttention
  - addressable cache
  - prompt caching
  - KV eviction
  - prompt injection
  - provable forgetting
  - cache coherence
date: 2026-06-19
---

# Addressable KV Cache: What Production Offers, and What It Doesn't

**Short answer:** the KV cache reuse that ships in production today is, in every
case, **prefix reuse**. It is a contiguous run starting at token 0. vLLM's Automatic
Prefix Caching, SGLang's RadixAttention, and the OpenAI / Anthropic / Gemini prompt
caches all reuse a prefix and only a prefix. The moment your context changes at
position *N*, everything from *N* onward is invalidated and recomputed. That is an
enormous, real win, and it is the part of "addressable" that is already saturated.

The part nobody ships is the other direction: reaching *into* a kept sequence,
removing one span (a poisoned tool result, an expired secret), and leaving the
cache **bit-for-bit identical to a run that never saw it**. `fak` does that. It owns
the KV cache as a plain kernel data structure rather than renting it from a
serving engine. This page is careful about which claims are which. The loose
version, "no one can address a KV span," is simply false, and the precise version
is the interesting one.

## First, the word "addressable" is doing four jobs

People use "addressable cache" to mean four different things. Keeping them apart is
the whole game:

1. **Prefix-addressed.** You can reuse the longest cached run starting at token 0.
   This is what every production engine ships. The address is "how many leading
   tokens match." It is append-only: you can extend a prefix and reuse more of it,
   but you cannot point at the middle.

2. **Span-addressed.** You can name an interior span `[i, j)` and operate on it
   (evict it, isolate it) and have the rest of the cache stay correct. This is the
   one production engines do *not* expose as a clean, exact operation.

3. **Content-addressed.** A piece of state is named by the hash of its bytes, so its
   identity *is* its content (a tool result is a `Ref` into a CAS blob store). This
   is the semantic layer — it works across models and sessions, because a hash
   doesn't care which transformer produced the bytes.

4. **Queryable-context.** A user or agent asks for a working set ("the API inventory
   plus the Qwen pages, exclude stale release notes"). The system materializes it
   under a budget and a policy, with a verdict per piece (HIT / FAULT / RECOMPUTE /
   REFUSE / ABSTAIN). The prompt becomes one *render* of a queryable memory image,
   rather than the memory itself.

Production has #1 solved and commoditized. `fak`'s contribution is #2 (exactly, and
as a security primitive), #3 (as the cross-model unit of reuse), and an early,
honestly-bounded version of #4.

## Why production reuse is always a prefix (the mechanism)

This is not a limitation anyone chose; it falls out of how a decoder transformer
works. Attention is **causal**: token *i*'s key and value vectors depend only on
tokens *0..i*. Once token 5 is processed, its K/V is fixed. It cannot depend on
anything that comes later.

So if two requests share an identical token prefix, the
K/V for that prefix is *bit-for-bit identical* between them, and you can splice in
the cached copy and prefill only the suffix. (`fak` proves exactly this.
`TestKVPrefixReuseMatchesRecompute` checks that prefix reuse matches a full
recompute to `max|Δ| = 0` with identical argmax. That holds given a fixed model and
tokenizer at the same precision, serializer, and position scheme.)

The flip side is the trap. Every token's K/V also encodes its *position*
(via RoPE or absolute embeddings) and, at deeper layers, what it *attended to*. So
you cannot just lift a span out of the middle of one sequence and drop it into another.
At layer 1 a token's K/V is mostly its embedding and position; by deeper layers it
has already mixed in everything before it. Change the preceding context and the same
surface tokens get *different* K/V.

That is why arbitrary mid-sequence KV reuse is
**not exact**. It is also why "addressable as in mix-and-match KV lego bricks" stays
fragile. The research community (CacheBlend, MiniPIC, SparseX, CacheSlide) is
still chipping at it with corrective tricks. Those include position repair and
selective recompute, plus quality probes and a fallback to exact recompute. It is
real work, but it buys a fault
budget rather than a clean primitive, and **none of it has shipped in a production serving
stack.** `fak` does not claim to have solved it either. Non-prefix splice is an
audited research item with explicit kill criteria; it is not yet a feature.

So the honest frame for #1 and the speculative #2-by-splice is: **prefix reuse is
exact and shipped everywhere; non-prefix splice is approximate and shipped nowhere.**

## The thing fak does that no shipped engine does: exact span removal

Here is where the precise claim lives, and it is narrower and sharper than the
slogan. Production engines are not *incapable* of touching a span — that is the
false version to avoid. vLLM's PagedAttention can copy-on-write a block; SGLang's
RadixAttention can drop a trie leaf; llama.cpp exposes `seq_rm` / `seq_cp` and a
K-shift. They have branch isolation and even forms of middle removal. So do not say
"no one can remove a span."

The defensible, shipped-and-tested claim is about **bit-exactness**:

> `fak` is the only KV cache that can remove a tool-result span from the *middle* of
> a kept sequence and leave the cache **byte-identical to one that never saw the
> span** — witnessed token-for-token against HuggingFace at `max|Δ| = 0`
> (`TestKVQuarantineEqualsNeverSaw`).

Why can it, when the others can't quite? Removing a middle span is only the easy
half (drop the bytes). The hard half is the *survivors*: every token after the cut
had its key rotated by RoPE at its **old** absolute position, and now sits at a new
one. To be exact you must re-derive those keys at their new positions from the
*unrotated* key, and you only have the unrotated key if you kept it.

- `fak` keeps the pre-RoPE key (`Kraw`) and re-rotates each survivor **once** at its
  new position. One clean rotation → exact.
- llama.cpp's K-shift *composes* rotations on the already-rotated key, which drifts
  ~1e-6: small, but enough to flip a greedy token.
- vLLM and SGLang store post-RoPE keys only, so an exact middle removal means
  recomputing the tail.

This is not a throughput claim. `fak` pays for the guarantee in memory (each radix
node holds a full-prefix KV copy, where SGLang shares one-token paged slabs). The
win is a *guarantee on a different axis*, bought with bytes. And the operation is the
same `Clone()` + `Evict()` the radix tree uses for its edge splits, proven bit-exact
in `TestReuseThroughSplitMatchesRecompute`.

## Why exact span removal is the feature, not a curiosity

Span-addressed, bit-exact removal is what turns the cache from a speed structure into
a **governance** structure, and that is the part a serving engine structurally does
not own. Two concrete payoffs:

- **Quarantine that reaches attention state.** When the byte-gate flags a tool result
  as poisoned, the *same verdict* evicts that result's K/V span from the attention
  cache. The model is not merely not-shown the poison. It is mechanically incapable
  of attending to it, and the cache is left bit-identical to never having seen it.
  (`max|Δ|` on logits for evict-vs-never = 0. The negative control, poison-vs-never, is a
  non-vacuous `max|Δ|` ≈ 0.326, so poison genuinely perturbs the distribution.) One
  decision, two enforcement media. (Proven on a synthetic model in
  `internal/kvmmu` today; not yet wired into the live `fak agent` HTTP loop.)

- **Eviction by policy, not just by pressure.** The cache-pressure LRU that SGLang
  and vLLM run evicts on a recency heuristic when memory is tight.
  `radixkv.EvictNode` adds policy-driven, span-exact, *provable* eviction of a named
  prefix on the same radix tree. It evicts because a verdict said so, rather than
  because the cache filled up. That is the one governance mode a pressure-only LRU
  cannot offer. And `fak` adds it *while* still reproducing SGLang's reuse efficiency:
  a 77–88% hit rate across few-shot/chat/ToT/agents, inside SGLang's verified 50–99% band.

This is also the durable leg. Prefix-cache cost wins erode as hardware loosens or
providers ship the feature server-side. "Provably remove this span and prove it's
gone" does not erode, because no hardware generation makes a forgetting requirement
disappear. It is the part of "addressable" that is both unshipped elsewhere and not
going to commoditize.

## The honest bounds (read these before citing)

- **KV reuse is intra-model only.** A KV cache is not portable across model
  architectures or tokenizers, which have different head dims, RoPE bases, and
  vocabularies. "Share one KV pool across Claude and Gemini" is a non-starter at the
  tensor layer. The *cross-model* sharing story is the content-addressed semantic
  layer (CAS-addressed tool results with provenance), running over per-model KV
  materialization rather than shared K/V bytes.

- **Non-prefix splice is not exact and not built.** Everything past exact prefix /
  radix reuse (arbitrary mid-sequence KV reuse) is a corrective, audited path with a
  fault-to-exact fallback. It is a design target with kill criteria and zero
  implementation today. Do not read "addressable KV" as "mix and match KV at will."

- **The queryable-context layer is early and partly in-flight.** The five-verdict
  materialization (HIT/FAULT/RECOMPUTE/REFUSE/ABSTAIN) is proven reachable in one
  test. A warm pass over cached views pages 0 raw bytes versus a cold build's
  6699. But that is on a synthetic demo image, and the context-layout compiler and
  non-prefix KV reuse are explicitly unbuilt. Answer-*quality* on queryable context
  is an open, unmeasured axis. Treat #4 as a real V1 surface that is still unfinished.

- **The comparison to SGLang is on hit rate, not throughput.** `fak` is not faster
  than a tuned GPU serving engine and does not claim to be. Cache hit rate is
  hardware-independent (it's a token count), and that is the one axis on which a Go
  cache on a laptop and a datacenter engine can be compared honestly.

## The one-line version

Production gives you an exact, append-only, **prefix**-addressed cache, and that's
genuinely most of the speed. What it does not give you is the ability to point at a
span in the middle, remove it, and *prove* it's gone: to make the cache a thing
policy can address, where today only pressure can. That is the underdiscussed half of
"addressable," it is the half that doesn't commoditize, and owning the cache as a
kernel object is what makes it possible.

## Where to go deeper

- The full vLLM / SGLang / llama.cpp / HF / fak span-surgery comparison: `TOOL-RESULT-TREE-KV-RESULTS.md` (private companion)
- The SOTA parity map (what every production cache exposes, with arxiv/doc URLs): [`AGENTIC-CACHING-SOTA-2026-06-19.md`](../notes/AGENTIC-CACHING-SOTA-2026-06-19.md)
- The Feynman walk-through of why prefix reuse is bit-exact + the radix tree: `RADIXATTENTION-EXPLAINER.md` (private companion)
- The measured hit-rate head-to-head with SGLang: `RADIXATTENTION-RESULTS.md` (private companion)
- The quarantine-verdict-drives-KV-eviction bridge: `KV-QUARANTINE-BRIDGE-RESULTS.md` (private companion)
- The queryable on-demand context proof + kill criteria: [`ON-DEMAND-CONTEXT-KV-REUSE-2026-06-19.md`](../notes/ON-DEMAND-CONTEXT-KV-REUSE-2026-06-19.md)
- Why this is the lead cross-tenant feature (provable forgetting): `DISAGGREGATED-AGENT-MEMORY.md` (private companion)
- How the KV cache erodes in agent loops (the input:output lever): [`kv-cache-agentic-context.md`](kv-cache-agentic-context.md)
