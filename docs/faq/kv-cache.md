---
title: "fak FAQ — The addressable KV cache, in detail"
description: "How fak reuses provider and local KV caches, measures cache hits, handles prefixes and eviction, and separates cache claims from evidence."
---

# The addressable KV cache, in detail

Part of the [fak FAQ](../FAQ.md) — the essentials and every other theme are
indexed there.

Quarantine makes the gate's decision durable and enforceable, but it does not improve the decision — a crafted injection that never trips the screen's marker set is never flagged and will resolve into context. The honest scope is that the structural floor (an unlisted irreversible tool stays refused; a flagged result stays sealed across the process boundary and re-screenable) is what holds, while the *detection* layer is explicitly evadable and the durable-seal guarantee is conditional on the gate having flagged the page in the first place. The lever to re-catch a missed injection is the re-screen on reload: once you tighten the markers, a reloaded session is re-judged by the stricter chain. Keep exfil-shaped and irreversible tools off the allow-list rather than relying on the detector.

## The addressable KV cache, in detail

How `fak` reaches into the middle of a kept model run and evicts a single span — a poisoned result, an expired secret — and leaves the cache bit-for-bit identical to a run that never saw it.

## What is the difference between front-of-prompt prefix reuse and mid-run causal eviction?

Prefix reuse extends a cached run forward from the front; mid-run causal eviction removes a span from the middle of a kept run and leaves the rest bit-identical to never having seen it. Every shipped engine does the first: vLLM's APC, SGLang's RadixAttention, and the OpenAI/Anthropic/Gemini prompt caches all reuse a contiguous run that starts at token 0, so changing context at position N invalidates everything after N. `fak` adds the second. Its `KVCache.Evict(from, n)` slices a span out of every layer's K/V tensors, compacts the absolute-position array, and re-derives each survivor's key from the stored pre-RoPE values in one clean rotation at its new position. RoPE is linear in position, so that single rotation is exact rather than a drift-accumulating shift.

## How does fak remove a single tool-result span from the middle of a kept run?

`fak` keeps a ledger of named segments over the cache, and evicting one calls `KVCache.Evict(seg.From, seg.Len)` then shifts every later segment's offset down so the ledger tracks the compaction. The cache stores the pre-RoPE keys (`Kraw`) alongside the rotated keys, so after slicing the span out it re-rotates each survivor whose absolute position changed in a single clean RoPE step at its new index; values are unrotated and need no fix. The kvmmu gate evicts at write-time, before any later segment is prefilled, so the removed span is causally upstream of nothing and the result equals a run that never saw it. Removing a span after later tokens have attended to it can only be un-seen if nothing downstream attended yet, which the code states honestly.

## What does max|Δ| = 0 mean, and how is it actually verified?

`max|Δ| = 0` means the largest absolute difference between two logit vectors is exactly zero: the post-eviction cache produces bit-identical next-token logits to a cache that never saw the evicted span. It is verified by witness tests that compare full logit vectors, not just the greedy argmax, because an untrained transformer's argmax can collapse while the vector stays context-sensitive. `TestWriteTimeEvictEqualsNeverSaw` reads real poison bytes through the real gate, quarantines and evicts the span, then asserts `max|Δ| evict-vs-never = 0.000e+00` with a non-vacuity control showing `poison-vs-never = 3.257e-01` (greater than zero). `TestLedgerRenumberAfterMiddleEvict` evicts a middle span then a tail span and asserts the survivors equal a fresh prefill at `max|Δ| = 0`.

## Why can fak evict a span bit-exactly when llama.cpp's K-shift cannot?

`fak` keeps the pre-RoPE keys and re-derives a moved survivor with one fresh rotation, so the result is exact; llama.cpp's K-shift composes rotations and drifts about 1e-6, which is enough to flip a greedy token. vLLM and SGLang store only post-RoPE keys, so for them an exact span removal means recomputing the tail rather than rotating in place. `fak`'s `applyRopeRow` casts through float32 to pin the rotation against FMA fusion, so the single rotation is bit-identical across architectures and call sites. That is the structural reason the addressable cache exists: it is the one degree of freedom no shipped serving engine kept.

## Why does owning the cache as a kernel object enable mid-run eviction?

Production engines rent the KV cache from a serving process behind an HTTP boundary, so policy can at best ask not to show a span; `fak`'s `KVCache` lives in the kernel's own Go address space, so the gate can physically delete the span and the model becomes mechanically incapable of attending to it. One detector verdict drives two enforcement media: the context-MMU bars the bytes from the text context, and the kvmmu bars the K/V from the attention state. Holding the cache as a plain Go data structure (per-layer K/Kraw/V slices plus an absolute-position array) is what makes span eviction and cross-session splice real operations rather than API requests. This is the durable leg of the design: prefix-cost wins erode as hardware loosens, but "provably remove this span and prove it is gone" does not.

## What is a deletion certificate and what does it actually prove?

A deletion certificate is a single portable, re-checkable receipt that binds a bit-exact KV-cache eviction to a tamper-evident audit journal. It proves three things under one ed25519 signature: that a named-span eviction ran (carrying the evicted count and span), that the equivalence was byte-identical (`MaxAbsDelta == 0`), and that it is anchored to a journal row whose `Subject` pins exactly which result was deleted. `Verify` fails closed on any tampered field: a signature mismatch, a non-zero delta ("equivalence not bit-exact"), an absent or broken journal chain, or a subject relabel each yields an invalid verdict. It is honest about its bounds: v1 is self-signed (integrity, not third-party independence), and it proves deletion only from the inference working set and agent memory, never from weights, embeddings, backups, or replicas.

## Is the deletion certificate's third-party verifiability shipped?

No. The v1 deletion certificate is self-attesting: its ed25519 signature proves integrity, not issuer independence, and third-party validation through an RFC-3161 timestamp or a CT-log is a named but empty seam (`ExternalAnchor`). The certificate's other honesty caveat is that `EvictedCount` is a self-report from the `Evict` call, not an independent re-count of the cache. The tamper-evident journal it anchors to is real and proven, but the external anchor that would let an outside party verify the receipt without trusting the issuer is design-target plumbing, not built.

## What is content-addressed storage and how does it back the cache?

Content-addressed storage (CAS) is a blob store where the sha256 digest is the identity, so a byte-identical payload is stored exactly once. `fak`'s `blob.Store` backs the resolver, region backend, and page-out backend, so the vDSO tier-2 cache and the context-MMU page-out share one store; small payloads (256 bytes or under) stay inline. It is pin-aware: a digest a live holder will resolve later is pinned and never evicted, while transient call arguments and results are LRU-evictable once the footprint passes the byte bound (default 1 GiB), and eviction never breaks the "cache hit equals a fresh call" invariant. This is the cross-model reuse layer, since a KV cache is intra-model only; cross-model sharing happens at this semantic byte layer, not as shared K/V tensors.

## Can two different models share the same KV cache?

No. KV reuse is intra-model only at the tensor layer, because head dimensions, RoPE, and vocabulary differ between models, so K/V bytes from one model are meaningless to another. What is shared across models is the content-addressed storage layer: tool results and their provenance are CAS blobs keyed by digest, a semantic byte-level reuse rather than shared attention state. Within a single model instance, cross-session prefix reuse comes from `Clone`/`SessionFromPrefix` and the radix tree; cross-worker residency moves are modeled by the `cachemeta.KVTransfer` metadata contract, whose live external engine is out of tree.

## How does radix prefix sharing relate to fak's addressable cache?

`fak`'s `radixkv` rebuilds SGLang's RadixAttention over the addressable cache, adding automatic longest-prefix discovery so callers don't have to declare the shared prefix. The tree is a compressed radix trie keyed on token-id runs; a `Lookup` walks to the longest cached prefix and splits an edge when divergence lands mid-run, so a real node boundary with a reusable cache exists there. The split is the interesting move: it truncates the child's cache via `Clone` plus `Evict` of the tail, which leaves no survivor to re-rotate, so the prefix is exact. `TestReuseThroughSplitMatchesRecompute` diverges two requests inside a compressed edge, splits, serves the second from the truncated clone plus a suffix prefill, and asserts the logits match a fresh full prefill at `max|Δ| = 0`.

## What can radixkv evict that an ordinary LRU prefix cache cannot?

`radixkv` can evict a named subtree as policy, regardless of recency, which an opportunistic LRU cache structurally cannot offer. `EvictToBudget` is ordinary LRU leaf eviction with upward collapse (RadixAttention's policy verbatim, where leased nodes survive pressure), but `EvictNode` removes a specific subtree because a quarantine verdict said so, not because of memory pressure. `TestPolicyEvictNode` witnesses that capability. The honest cost: each node stores the full-prefix cache rather than SGLang's per-segment paged slabs, so it uses more memory, and `Stats` exposes both `Tokens` (the LRU metric) and `PrefixTokens` (the true resident footprint) so the gap is measurable rather than silent.

## How does fak prove that prefix reuse equals a full recompute?

`fak` proves prefix reuse is exact with witness tests that compare a reused-prefix session against a full recompute at `max|Δ| = 0` with identical argmax. `Clone` deep-copies a computed prefix and `SessionFromPrefix` starts a session on that clone so only the suffix is prefilled, and because the copy is exact the reusing session is bit-identical to one that prefilled the whole prefix. `TestKVPrefixReuseMatchesRecompute` pins reuse-equals-recompute, and `TestCachedDecodeMatchesPrefill` asserts cached decode equals a full forward pass to the last bit, failing if any difference appears. These exact-equality gates are the honesty check that the speedup comes from reuse, not from a numerics shortcut.

## What happens to the segment ledger when a middle span is evicted?

When a middle span is evicted, the kvmmu ledger calls `Cache.Evict(seg.From, seg.Len)` and then renumbers: every later segment's `From` offset shifts down by the evicted length, so the ledger keeps tracking the physical compaction. Segments are addressed by name, not by position or token content, so a by-id eviction removes exactly that segment's range and the proof's bijection theorem guarantees no survivor is lost and no slot aliases another. `TestLedgerRenumberAfterMiddleEvict` evicts a middle segment of one length then a tail segment of a different length and asserts the surviving segments equal a fresh prefill at `max|Δ| = 0`; a stale offset would misfire precisely because the lengths differ.

## Is the quarantine-drives-KV-eviction bridge wired into the live fak agent loop yet?

No: the kvmmu bridge that turns a quarantine verdict into a bit-exact KV-span eviction is proven on a synthetic model but is not yet wired into the live `fak agent` HTTP loop. The mechanism is real and witnessed (`TestWriteTimeEvictEqualsNeverSaw` runs the real ctxmmu gate over real poison bytes), but the witness uses a small synthetic Llama (hidden 32, two layers) to prove the wiring, while the HF numerics are proven separately by the `internal/model` oracle. No `radixkv` or `kvmmu` import appears under the kernel package today. The context-MMU side that bars poisoned bytes from the text context is shipped on the gateway path; the K/V-eviction half is the part still to be connected.

## Is arbitrary mid-sequence KV splicing (not just prefix or span removal) supported?

No. Non-prefix, arbitrary mid-sequence KV splice (inserting or rearranging spans anywhere) is approximate and has zero implementation; it is a documented design target, audited with kill criteria, not built. What is shipped and bit-exact is the pair that matters in practice: front-of-prompt prefix reuse and removal of a span from the middle of a kept run. The queryable-context materialization with its five verdicts (HIT, FAULT, RECOMPUTE, REFUSE, ABSTAIN) is early and partly in flight, proven reachable on a synthetic demo image, with answer quality still unmeasured. Treat arbitrary splice as a roadmap item rather than a capability.

## What numbers can fak honestly claim for KV cache reuse, and against which baseline?

On agent workloads the WITNESSED kernel-owned `radixkv` benchmark matches SGLang's regime at an 86.7% cache hit rate and a 7.50× token speedup versus naive re-prefill, and it adds about 1.22× cross-worker reuse where SGLang is 0%. The cited bottom line is a 20-24× infrastructure cost reduction versus naive re-prefill and 1.13-1.22× cross-worker; the radixkv explainer cites a 77-88% hit rate across few-shot, chat, tree-of-thought, and agent workloads, inside SGLang's verified 50-99% band. Hit rate is a token count, so it is hardware-independent, which is the one axis where a Go cache on a laptop and a datacenter GPU engine compare honestly. The honest fence: the 1.22× cross-worker figure is a measured/projected fleet number, not a live multi-node deployment.

## Does a quarantined span ever physically leave the model's attention, or is it just hidden from view?

When the kvmmu bridge evicts a quarantined span, the span physically leaves the model's attention state: its K/V columns are sliced out of every layer, so the model is mechanically incapable of attending to it, not merely "not shown" it. This is distinct from the context-MMU's text-side quarantine, which holds poisoned bytes out of the conversation by paging them to a stub pointer. The two are one decision enforced in two media: the context-MMU keeps the bytes out of the prompt, kvmmu keeps the K/V out of attention. The write-time path is the clean case, because evicting before any later token attended makes the result identical to never having seen the span; the after-the-write path carries the honest caveat that it can only un-see a span nothing downstream attended to yet.

## What is the cachemeta contract and why is its KV-residency layer not fully live?

