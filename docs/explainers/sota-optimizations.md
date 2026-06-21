# SOTA Serving Optimizations — What "Tuned" Actually Means

**Context:** When we say "tuned SOTA stack" or "vs tuned baseline" in benchmark results, we're
referring to a serving stack with multiple optimizations already applied. This page explains
what those optimizations are, which are common in production stacks, and fak's status on each.

---

## What "tuned SOTA" means

A **tuned SOTA stack** is a production serving setup with these characteristics:

1. **KV cache / prefix caching** — Reuse computation across requests with shared prefixes
2. **Batched inference** — Process multiple requests simultaneously on the same GPU
3. **Quantization** — Use lower-precision weights (Q8, Q4, Q2) to reduce memory and increase speed
4. **Serving engine optimizations** — SIMD, fused kernels, optimized attention implementations
5. **Multi-GPU / tensor parallelism** — Distribute large models across multiple GPUs
6. **Paged attention** — KV cache management that handles varying context lengths efficiently
7. **Speculative decoding** — Use a smaller draft model to accelerate larger model decoding
8. **Continuous batching** — Dynamic scheduling that adds/removes requests as they complete
9. **Request routing** — Route requests to appropriate model tiers or endpoints
10. **Tool batching** — Process multiple tool calls in a single model call

**The key point:** Most of these are **implemented in the serving engine** (llama.cpp, vLLM,
SGLang, Ollama, etc.) and apply regardless of whether fak is in front. fak's contribution is
the **governance layer** on top of these optimizations.

---

## Top 10 Optimizations: fak Status

### 1. KV Cache / Prefix Caching ✅ IMPLEMENTED

**What it is:** Cache the Key-Value attention vectors computed during prefill and reuse them
for subsequent requests that share a prefix. Eliminates redundant computation.

**SOTA implementations:**
- vLLM: Automatic Prefix Caching
- SGLang: RadixAttention (radix tree of token sequences)
- OpenAI: Prompt caching API
- LMCache: Distributed KV cache

**fak status:** ✅ **Implemented** — `internal/radixkv` implements RadixAttention algorithm with
86.7% hit rate on agents workload (inside SGLang's 50–99% band). See [`fak/RADIXATTENTION-RESULTS.md`](../../fak/RADIXATTENTION-RESULTS.md).

**Differentiator:** fak adds **policy-driven eviction** — can evict by quarantine verdict, not
just LRU memory pressure.

---

### 2. Batched Inference ✅ IMPLEMENTED

**What it is:** Process multiple independent requests simultaneously on the same GPU.
Increases throughput by keeping all compute units busy.

**SOTA implementations:** vLLM, SGLang, llama.cpp, TensorRT-LLM

**fak status:** ✅ **Implemented** — `internal/model.BatchFromPrefix` processes C agents
concurrently with shared prefix. See [`fak/MODEL-BATCHING-RESULTS.md`](../../fak/MODEL-BATCHING-RESULTS.md).

---

### 3. Quantization ✅ IMPLEMENTED

**What it is:** Store model weights in lower precision (8-bit, 4-bit, etc.) to reduce memory
requirements and increase compute speed. Modern quantization preserves most accuracy.

**SOTA implementations:** llama.cpp (Q8_0, Q4_K_M, Q2_K, etc.), vLLM, AWQ, GPTQ

**fak status:** ✅ **Implemented** — Q8_0 quantization with proven bit-exact forward pass
against HF reference. See [`fak/IN-KERNEL-MODEL-DESIGN.md`](../../fak/IN-KERNEL-MODEL-DESIGN.md).

---

### 4. SIMD / Fused Kernels 🔄 PARTIAL

**What it is:** Use CPU SIMD instructions (AVX-512, NEON, etc.) and GPU fused kernels to
accelerate matrix operations and reduce memory bandwidth.

**SOTA implementations:** llama.cpp (heavily optimized SIMD), vLLM (CUDA kernels), FlashAttention

**fak status:** 🔄 **Partial** — Uses Go's native SIMD where available. For maximal SIMD
performance, `fak` can front `llama-server` which has extensive hand-tuned SIMD.

---

### 5. PagedAttention / KV Management ✅ IMPLEMENTED

**What it is:** Manage KV cache in pages rather than contiguous blocks, allowing efficient
handling of variable-length sequences and cache eviction.

**SOTA implementations:** vLLM (PagedAttention), SGLang

**fak status:** ✅ **Implemented** — `internal/kvmmu` provides context-MMU with span-level
management. Differentiator: **policy-aware invalidation** (not just memory pressure).

---

### 6. Multi-GPU / Tensor Parallelism ❌ NOT FOCUSED

**What it is:** Distribute a large model across multiple GPUs using tensor parallelism or
pipeline parallelism.

**SOTA implementations:** vLLM (tensor parallelism), DeepSpeed, TensorRT-LLM

**fak status:** ❌ **Not focused** — fak runs on single GPU or CPU. Multi-GPU is handled
by the serving engine `fak` fronts (e.g., vLLM cluster).

---

### 7. Speculative Decoding ❌ NOT IMPLEMENTED

**What it is:** Use a small draft model to predict tokens, verify in parallel with the larger
target model. Can accelerate decoding by 2-3×.

**SOTA implementations:** vLLM, SGLang (experimental), llama.cpp (draft models)

**fak status:** ❌ **Not implemented** — Could be added as an optimization; currently relies
on serving engine for this.

---

### 8. Continuous Batching ❌ ENGINE-LEVEL

**What it is:** Dynamically add and remove requests from batches as they complete, rather
than fixed batch sizes. Improves throughput for variable-length workloads.

**SOTA implementations:** vLLM (continuous batching), SGLang, TGI

**fak status:** ❌ **Engine-level** — Implemented by serving engines. `fak` works with
whatever batching strategy the engine uses.

---

### 9. Request Routing / Tiered Serving ✅ PARTIALLY

**What it is:** Route requests to different model tiers or specialized endpoints based on
request characteristics (complexity, cost, etc.).

**SOTA implementations:** Custom routers, API gateways, provider routing

**fak status:** ✅ **Partial** — `fak` can route to different backends via `--base-url`,
but doesn't automatically classify requests. This is typically done upstream.

---

### 10. Tool Batching ✅ SUPPORTED

**What it is:** Emit multiple tool calls in a single model response and process them in
parallel. Reduces turn count and latency.

**SOTA implementations:** Anthropic Claude, OpenAI, many agent frameworks

**fak status:** ✅ **Supported** — The kernel doesn't interfere with tool batching. Tool
calls are validated individually regardless of batch size.

---

## Vision / Multimodal ❌ NOT FOCUSED

**What it is:** Process images, audio, or video alongside text in the same model or pipeline.

**SOTA implementations:** GPT-4V, Claude 3.5 Sonnet (Vision), Gemini Pro Vision, LLaVA

**fak status:** ❌ **Not focused** — fak works with text-only models. Vision models can be
used via gateway, but vision-specific governance (e.g., image quarantine) is not implemented.

---

## What This Means for Benchmarks

When we report "1.5–4× vs tuned SOTA", we're comparing against a stack that has:

- ✅ KV cache / prefix caching
- ✅ Batched inference
- ✅ Quantization
- ✅ Optimized kernels (SIMD, fused)
- ✅ Efficient KV management

The **1.5–4× gain comes from**:
1. **Fused serving** — Avoid process spawn per request
2. **Cross-agent prefix sharing** — Multiple agents share one KV copy
3. **Batch scheduling** — Cache-aware request ordering

**Not from:**
- Raw model speed (we're ~parity or slightly behind)
- Basic KV reuse (SOTA already has this)
- Quantization (SOTA already has this)

---

## SOTA Engines We Compare Against

| Engine | Strengths | Notes |
|---|---|---|
| **llama.cpp** | CPU optimization, quantization, broad model support | SOTA for local serving |
| **vLLM** | GPU throughput, PagedAttention, continuous batching | SOTA for GPU serving |
| **SGLang** | RadixAttention, structured generation | SOTA for cache hit rates |
| **Ollama** | Ease of use, local serving | User-friendly local stack |
| **OpenAI API** | Frontier models, prompt caching | Cloud SOTA baseline |

---

## Honest Baseline Disclosure

All benchmark results explicitly state:

1. **What the baseline is** (e.g., "vLLM with automatic prefix caching")
2. **What optimizations are enabled** (e.g., "Q8_0 quantization, batch size 4")
3. **What hardware is used** (e.g., "Apple M3 Pro, 32GB RAM")
4. **What the gain is attributed to** (e.g., "fused serving + cross-agent sharing")

See [`fak/BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) for the single source of truth on all benchmark numbers.

---

## FAQ

**Q: Is fak trying to replace llama.cpp or vLLM?**
A: No. `fak` fronts these engines, adding a governance layer. For raw throughput, use
`llama-server` or vLLM directly. `fak` is for safety, coherence, and legal reuse — not
raw tok/s.

**Q: Why compare against tuned SOTA instead of naive?**
A: Because tuned SOTA is what people actually use in production. Comparing against a
stateless loop that re-sends everything would be misleading — nobody runs that way at
scale.

**Q: Does fak implement all these optimizations?**
A: No, and it doesn't need to. The serving engine implements the throughput optimizations.
`fak` implements the **governance layer** (permissions, quarantine, policy-driven invalidation)
that serving engines don't have.

---

*Last updated: 2026-06-19*
