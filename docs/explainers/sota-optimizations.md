---
title: "What Tuned SOTA Serving Optimizations Mean in fak Benchmarks"
description: "What tuned SOTA means in fak benchmarks: KV cache, batching, quantization, paged attention — and which mechanisms fak owns natively versus uses as explicit external references."
---

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
 4. **SIMD / Fused Kernels** — CPU SIMD instructions and GPU fused kernels for faster matrix ops
 5. **Paged attention** — KV cache management that handles varying context lengths efficiently
 6. **Multi-GPU / tensor parallelism** — Distribute large models across multiple GPUs
7. **Speculative decoding** — Use a smaller draft model to accelerate larger model decoding
8. **Continuous batching** — Dynamic scheduling that adds/removes requests as they complete
9. **Request routing** — Route requests to appropriate model tiers or endpoints
10. **Tool batching** — Process multiple tool calls in a single model call

**The key point:** [fak-native inference](../native-inference-goal.md) is the local product and
performance path: fak aims to own the model path, kernels, memory, scheduling, cache,
adaptation, and operations. External engines remain supported when explicitly selected for a
gateway, benchmark, parity diagnosis, interoperability, or prior-art study; they are never a
silent fallback or evidence of fak-native performance.

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
86.7% hit rate on agents workload (inside SGLang's 50–99% band). See `RADIXATTENTION-RESULTS.md` (private companion).

**Differentiator:** fak adds **policy-driven eviction** — can evict by quarantine verdict, not
just LRU memory pressure.

---

### 2. Batched Inference ✅ IMPLEMENTED

**What it is:** Process multiple independent requests simultaneously on the same GPU.
Increases throughput by keeping all compute units busy.

**SOTA implementations:** vLLM, SGLang, llama.cpp, TensorRT-LLM

**fak status:** ✅ **Implemented** — `internal/model.BatchFromPrefix` processes C agents
concurrently with shared prefix. See `MODEL-BATCHING-RESULTS.md` (private companion).

---

### 3. Quantization ✅ IMPLEMENTED

**What it is:** Store model weights in lower precision (8-bit, 4-bit, etc.) to reduce memory
requirements and increase compute speed. Modern quantization preserves most accuracy.

**SOTA implementations:** llama.cpp (Q8_0, Q4_K_M, Q2_K, etc.), vLLM, AWQ, GPTQ

**fak status:** ✅ **Implemented** — Q8_0 quantization with proven bit-exact forward pass
against HF reference. See `IN-KERNEL-MODEL-DESIGN.md` (private companion).

---

### 4. SIMD / Fused Kernels 🔄 PARTIAL

**What it is:** Use CPU SIMD instructions (AVX-512, NEON, etc.) and GPU fused kernels to
accelerate matrix operations and reduce memory bandwidth.

**SOTA implementations:** llama.cpp (heavily optimized SIMD), vLLM (CUDA kernels), FlashAttention

**fak status:** 🔄 **Partial** — Native CPU and device kernels exist, but current matched
benchmarks still show gaps in some envelopes. llama.cpp may be selected explicitly as a
benchmark or parity reference; fak does not silently replace a requested native path with it.

---

### 5. Host-paged KV Management ✅ IMPLEMENTED; Device PagedAttention Kernel ❌ NOT CLAIMED

**What it is:** Manage KV cache in pages rather than contiguous blocks, allowing efficient
handling of variable-length sequences and cache eviction.

**SOTA implementations:** vLLM (PagedAttention), SGLang

**fak status:** ✅ **Host-side paging implemented** — `internal/model.PagedKVPool` provides
fixed-size physical blocks, page tables, copy-on-write prefix sharing, and exact gathers;
`internal/kvmmu` adds policy-aware span invalidation. The opt-in CPU-reference HAL gathers
paged K/V into contiguous host tensors before attention. That is not a device PagedAttention
kernel that consumes page tables directly, and no such device kernel is claimed here.

---

### 6. Multi-GPU / Tensor Parallelism 🟡 PRIMITIVE SHIPPED, DEVICE RUN HARDWARE-GATED

**What it is:** Distribute a large model across multiple GPUs using tensor parallelism or
pipeline parallelism.

**SOTA implementations:** vLLM (tensor parallelism), DeepSpeed, TensorRT-LLM

**fak status:** 🟡 The native kernel now has a tensor-parallel decomposition (Megatron
column/row sharding), a four-collective HAL seam, and both an in-process and a **real
cross-process (TCP)** collective — all **bit-exact** vs a single-device reference, proven
on a CPU with no GPU. The live multi-GPU **device** run still needs an NCCL/RCCL
`CollectiveBackend` plus a 2-/4-GPU host (hardware-gated). fak can also still front a
serving engine's own multi-GPU cluster (e.g. vLLM). See
[multi-gpu-tensor-parallelism.md](multi-gpu-tensor-parallelism.md).

---

### 7. Speculative Decoding 🟡 IMPLEMENTED, FEATURE-GATED

**What it is:** Use a small draft model to predict tokens, verify in parallel with the larger
target model. Can accelerate decoding by 2-3×.

**SOTA implementations:** vLLM, SGLang (experimental), llama.cpp (draft models)

**fak status:** 🟡 **Implemented, off by default** — `internal/model` binds live target and
drafter sessions to one-pass verification plus bit-exact `KVCache.Evict` rollback;
`internal/polymodel` supplies the draft/accept loop and `AcceptGreedy`/`AcceptTree`.
`FAK_POLYMODEL` gates the request path. The current authority proves token identity and
effective tokens per verify pass, not a wall-clock 2–3× hardware speedup.

---

### 8. Continuous Batching ✅ NATIVE LIFECYCLE SCHEDULER SHIPPED

**What it is:** Dynamically add and remove requests from batches as they complete, rather
than fixed batch sizes. Improves throughput for variable-length workloads.

**SOTA implementations:** vLLM (continuous batching), SGLang, TGI

**fak status:** ✅ **Native scheduler shipped** — `internal/modelengine` registers the
`inkernel` continuous-batching lifecycle scheduler. Its current benchmark is a synthetic CPU
modelengine witness, not a vLLM/SGLang production SLA; multi-tenant p99 policy remains a
separate leaf. Explicit gateway deployments still use the upstream engine's scheduler.

---

### 9. Request Routing / Tiered Serving ✅ SHIPPED; LEARNED ROUTING OPEN

**What it is:** Route requests to different model tiers or specialized endpoints based on
request characteristics (complexity, cost, etc.).

**SOTA implementations:** Custom routers, API gateways, provider routing

**fak status:** ✅ **Shipped** — `internal/modelroute` and `fak route` provide deterministic
per-aspect picks and ensembles. `--route-manifest` executes single-model picks in the served
gateway and standalone agent path, and ensembles in the gateway. Learned routing remains a
follow-on; see [model routing](../model-routing.md).

---

### 10. Tool Batching ✅ SUPPORTED

**What it is:** Emit multiple tool calls in a single model response and process them in
parallel. Reduces turn count and latency.

**SOTA implementations:** Anthropic Claude, OpenAI, many agent frameworks

**fak status:** ✅ **Supported** — The kernel doesn't interfere with tool batching. Tool
calls are validated individually regardless of batch size.

---

## Vision / Multimodal ✅ GOVERNANCE SEAM SHIPPED; NATIVE ENCODER OPEN

**What it is:** Process images, audio, or video alongside text in the same model or pipeline.

**SOTA implementations:** GPT-4V, Claude 3.5 Sonnet (Vision), Gemini Pro Vision, LLaVA

**fak status:** ✅ **Governed input seam shipped** — `internal/model.ForwardMultimodal`
accepts ordered text plus externally produced vision embeddings behind a fail-closed
`MultimodalPolicy`. Image count, bytes, pixels, embedding width/count, media type, and active
content are bounded; admitted image metadata, raw bytes, and embedding fingerprints bind a
`vision-sha256:` quarantine pointer. This is not a built-in CLIP/OCR/VLM encoder or classifier.

---

## What This Means for Benchmarks

For any result, its named baseline and operating envelope—not this generic checklist—are
authoritative. The decision-grade headline reports **4.1× vs tuned per-agent warm KV** in one
declared 50-turn × 5-agent Qwen2.5-1.5B Q8 envelope. The same run's **60.3×** is versus naive
stateless re-send and is context, not the tuned adoption baseline. Common tuned stacks include:

- ✅ KV cache / prefix caching
- ✅ Batched inference
- ✅ Quantization
- ✅ Optimized kernels (SIMD, fused)
- ✅ Efficient KV management

The **4.1× gain in that envelope comes from**:
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

See [`fak/BENCHMARK-AUTHORITY.md`](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md) for the single source of truth on all benchmark numbers.

---

## FAQ

**Q: Is fak trying to replace llama.cpp or vLLM?**
A: For local inference, fak-native is the product path and aims to beat explicit reference
engines in matched, quality-constrained envelopes. fak can also front external engines for
gateway and interoperability use, but never silently substitutes one for native execution.

**Q: Why compare against tuned SOTA instead of naive?**
A: Because tuned SOTA is what people actually use in production. Comparing against a
stateless loop that re-sends everything would be misleading — nobody runs that way at
scale.

**Q: Does fak implement all these optimizations?**
A: Not yet. The native path owns the implementation boundary and labels remaining gaps; an
explicit external-engine run remains gateway, comparison, or interoperability evidence rather
than proof that fak-native implements that optimization.

---

*Last updated: 2026-08-24*
