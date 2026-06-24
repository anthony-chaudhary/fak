---
title: "fak serving plans: scale-out KV cache, poly-model, dual-track"
description: "Index of fak's large-scale serving design docs: dual-track serving, poly-model prefill sharing, hardware-aware and regenerable KV cache, and CXL memory pooling."
---

# fak serving plans — the scale-out roadmap

These are fak's design and decision docs for serving at fleet scale. They are
**plans and architecture briefs**, not shipped throughput claims — each one says
explicitly what is built, what is a seam, and what is still a gap. The reuse kernel
(addressable, bit-exact KV cache + the default-deny capability floor) is the shared
substrate every track below builds on.

| Plan | What it covers |
|---|---|
| [Dual-track serving](dual-track-serving-plan.md) | The authoritative sequencing contract: **ride** best-in-class engines (vLLM, SGLang) *and* grow a **native** in-kernel engine, over one shared, track-neutral spine. |
| [Poly-model prefill sharing](polymodel-prefill-share-plan.md) | Host tens of models in one kernel, share prefill across them, decode one at a time, and put idle models to work on speculative decoding. |
| [Hardware-aware KV cache](hardware-aware-cache.md) | Plan where a KV span lives across HBM, DRAM, NUMA-far, CXL, disk, and remote tiers — per-tier TTL and demote-not-evict placement. |
| [Regenerable KV cache](regenerable-kv-plan.md) | Treat the KV cache as a build artifact rebuilt from durable transcript text, so a model rollout becomes a backfill instead of a cold start. |
| [Multi-tenant CXL memory pool](cxl-memory-pool.md) | Price CXL.mem pooled KV reuse across a fleet and gate cross-tenant cell reuse, failing closed on poisoned or wrong-model cells. |

## How these relate

`fak` is **not** a faster model server — vLLM, SGLang, and llama.cpp win raw
throughput, and you can run `fak serve` in front of any of them. These plans cover the
orthogonal questions a fleet hits at scale: where reused KV lives, how it is shared and
rebuilt, and which reuse is still *legal* across tenants. See the
[addressable KV cache](../explainers/addressable-kv-cache.md) explainer for the core idea.
