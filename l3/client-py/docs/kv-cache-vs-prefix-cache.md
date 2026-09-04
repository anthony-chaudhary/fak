# KV Cache (Attention) vs SGLang Prefix Caching

## KV Cache in Attention

During transformer inference, each layer computes Key and Value projections for every token. Without caching, generating token N requires recomputing K and V for all N-1 previous tokens — O(N²) work.

The KV cache stores K and V tensors from prior tokens so each decoding step only computes K/V for the new token and appends it. This is a per-request, per-layer optimization that avoids redundant matrix multiplications during autoregressive generation.

```
Step 1: "The"         → compute K₁,V₁, store in cache
Step 2: "The cat"     → reuse K₁,V₁, compute K₂,V₂, append
Step 3: "The cat sat" → reuse K₁₂,V₁₂, compute K₃,V₃, append
```

Every LLM inference engine does this (vLLM, TensorRT-LLM, SGLang, etc.).

## SGLang's RadixAttention / Prefix Caching

Operates **across requests**. Reuses the KV cache from one request for a different request that shares the same prefix.

Example: if 100 requests share the same system prompt, the KV cache for that prompt is computed once and shared. SGLang uses a radix tree (trie) keyed by token sequences to efficiently find the longest matching prefix.

```
Request A: [system prompt] + "What is 2+2?"
Request B: [system prompt] + "Explain gravity"
                 ↑
        KV cache computed once, reused for both
```

## Key Differences

| | Attention KV Cache | SGLang Prefix Cache |
|---|---|---|
| **Scope** | Single request | Across requests |
| **What it avoids** | Recomputing K/V for prior tokens within a sequence | Recomputing K/V for shared prefixes across sequences |
| **Granularity** | Token-level, per layer | Prefix-level, stored in a radix tree |
| **Where it lives** | GPU memory, tied to one generation | GPU memory, managed by a cache pool with eviction |
| **Analogy** | Not re-reading a book page you already read | Sharing highlighted notes between people reading the same book |

## How They Compose

SGLang uses both. When a new request arrives:

1. **Radix tree lookup** — find the longest cached prefix from any prior request
2. **Reuse those KV tensors** — skip forward computation for those layers/tokens
3. **Compute remaining tokens** — standard forward pass for the new suffix
4. **Append to per-request KV cache** — normal autoregressive decoding from there

The attention KV cache is the *mechanism*; SGLang's prefix caching is a *memory management strategy* that shares those cached tensors across requests to avoid redundant prefill computation.
