---
title: "The memory wall is a design space, not a verdict"
description: "An exhaustive fak-native research map for reducing, reusing, hiding, predicting, relocating, or eliminating inference memory traffic—with falsifiable experiments and a GitHub execution portfolio."
---

# The memory wall is a design space, not a verdict

> **Thesis:** “Bandwidth-bound” describes one measured envelope. It does not mean the bottleneck is immutable.

Attack every term in accepted-token time rather than accepting the measured envelope:

```text
T_accepted_token = (T_weight + T_KV + T_state + T_activation + T_transfer
                    + T_launch + T_sync + T_compute + T_queue + T_recovery
                    + T_rejected_speculation) / accepted_tokens

T_tier >= demanded_bytes_at_tier / sustainable_bandwidth_at_tier
```

Every optimization must change at least one numerator term, increase useful accepted tokens,
or move work to a tier with a better effective cost. Terms can overlap, so their sum is an
accounting decomposition, not a claim that all phases serialize.

Start by making the baseline falsifiable:

```bash
go run ./cmd/modelprof -dir <model-export-dir> -prompt 16 -steps 24 -prefill 64 -out before.json
```

Then change one byte/reuse/overlap term, rerun the identical envelope, and require a
quality-qualified end-to-end receipt before calling it a gain.
## Assumptions to push on

Treat each statement below as a hypothesis, never a premise:

| Assumption | Why it may be false | Decisive intervention |
|---|---|---|
| Every active weight must be read for every token | Cache persistence, batching, sparsity, recurrent state, architecture changes | Hold one layer/expert persistent or skip known-zero blocks; measure bytes and accepted-token time |
| Model bytes equal DRAM bytes | L2/SLC/SRAM hits, compression metadata, rereads, write amplification | Same-run tier counters plus address/transaction efficiency |
| Peak bandwidth is the roof | Access pattern, clocks, ECC, contention, partition camping, thermals | Matched streaming and production-pattern roofs |
| Decode is always bandwidth-bound | Launch gaps, unpack compute, occupancy, dependencies, sync can dominate | Reduce bytes 2× with semantics fixed; inspect time response |
| Prefill is always compute-bound | Skinny/short prompts and unfused operators may have low reuse | Prompt-shape roofline sweep |
| Quantization speedup follows bit width | Dequantization, scales, alignment, and quality recovery add cost | Fused native quant kernel, same quality, full receipt |
| More batching is always better | Tails, padding, KV divergence, queueing, and SLOs can dominate | SLO-constrained batch frontier |
| Prefetch always hides latency | Bad prediction, cache pollution, dependency distance, and bandwidth contention | Prefetch on/off with useful-prefetch precision and timeliness |
| Speculation saves bandwidth | Draft and rejected verification can increase bytes per accepted token | Accepted tokens / total bytes, including draft and rejects |
| KV is secondary to weights | Long context can make KV traffic dominant | Context-length crossover sweep |
| Paging solves KV memory | Indirection, fragmentation, TLB/cache misses, and copies may dominate | Contiguous versus paged matched trace |
| Offload only hurts | Capacity admission or overlapped pipelines may beat a smaller/slower local model | Full-tier pipeline versus resident alternative |
| MoE inherently saves traffic | Router/shared experts, imbalance, misses, and all-to-all can erase sparsity | Active expert bytes and communication per accepted token |
| Unified memory is transparent | Faults, migration granularity, pinning, and eviction can create cliffs | Fault/migration trace under controlled residency |
| Hardware counters tell the truth directly | Replay, multiplexing, unsupported events, and coarse sampling distort | Cross-check counters with controlled byte perturbation |
| The fastest kernel wins | Queueing, copies, launch orchestration, and quality can erase kernel gains | Net-true request receipt |
| Memory optimization must preserve the exact architecture | GQA/MLA/SSM/recurrent hybrids can change KV and weight traffic fundamentally | Quality-matched architecture-envelope comparison |
| Static optimization is enough | Request mix, context, batch, and cache state change the best policy | Online controller with bounded adaptation and rollback |

The wall has at least seven escape directions:

1. **Demand fewer bytes:** quantize, prune, compress, share, skip, or change the architecture.
2. **Move fewer physical bytes:** improve locality, coalescing, tiling, packing, and cache hits.
3. **Reuse each byte more:** batch, fuse, persist, cache experts, and amortize weights.
4. **Hide movement:** prefetch, pipeline, double-buffer, overlap, and schedule ahead.
5. **Predict demand:** prefetch likely experts/KV/weights and cancel cheaply when wrong.
6. **Do less rejected work:** optimize accepted tokens per byte, especially under speculation.
7. **Relocate work:** choose HBM, SRAM/cache, unified memory, host DRAM, peer memory, CXL,
   storage, or another node from measured cost—not habit.

The bottleneck is therefore usually *solvable in degree*. Physics prevents zero traffic, but
product performance depends on useful tokens per joule, dollar, and second, not on defeating a
marketing peak. A lever can move the frontier even when it does not abolish memory traffic.

## Lever atlas

### 1. Measurement and causal attribution

Before optimization, make bytes falsifiable:

- Attribute requested, transferred, and useful bytes separately by weight, scale/metadata, KV,
  recurrent state, activation/workspace, allocator, copy, peer, host, and storage tiers.
- Record transaction efficiency, cache-line waste, replay, bank/partition imbalance, page faults,
  TLB pressure, cache hit/miss bytes, and read/write amplification where hardware exposes them.
- Measure cold, warm, and steady-state roofs for the exact allocation size, alignment, stride,
  vector width, concurrency, NUMA placement, clocks, and power state.
- Pair sampled fleet telemetry with triggered deep captures. Utilization is a trigger, not proof.
- Use controlled perturbations: halve represented bytes, pad stride, disable fusion, pin clocks,
  change batch, or force residency. A label without a predicted response is not causal.
- Build a byte conservation check: estimated logical traffic should reconcile with observed tier
  traffic within declared uncertainty; unexplained amplification becomes its own bug.

### 2. Weight representation: fewer stored and fetched bits

- Native INT8/INT6/INT5/INT4/INT3/INT2 and binary/ternary formats, with quality floors.
- Mixed precision by layer, channel, group, token phase, or outlier sensitivity.
- AWQ/GPTQ/SmoothQuant-style calibration; activation-aware scaling and outlier side paths.
- Vector-quantized/codebook, additive, lattice, entropy, and low-rank residual representations.
- Compress once at load versus on-disk compressed plus fused streaming decompression.
- Separate hot shared/router/norm weights from colder expert or optional blocks.
- Reorder scale/zero-point metadata to coalesce with the values it governs.
- Fuse unpack, dequantize, dot product, bias, activation, and requantization so expanded weights
  never reach DRAM.
- Measure total bytes, metadata, unpack operations, register pressure, occupancy, quality, load
  time, and recovery—not nominal bits/weight.

### 3. Sparsity and conditional computation: do not fetch zeros or unused blocks

- Structured N:M sparsity that maps to native hardware support.
- Block/channel/head/neuron/layer pruning with contiguous skip units.
- Unstructured sparsity only when index and irregular-access cost is demonstrably recovered.
- Dynamic activation sparsity, zero skipping, and token-dependent block gating.
- Early-exit layers, adaptive depth, and confidence-gated expensive paths.
- Head pruning and query-dependent sparse attention.
- Train-free versus retrained sparsity must carry separate quality and deployment costs.
- Store sparse metadata near use, batch tokens sharing masks, and compare sparse bytes against a
  dense compressed baseline rather than dense FP16.

### 4. Layout, transactions, and locality: make each physical transfer useful

- Align, pad, swizzle, interleave, transpose, and pack around actual warp/SIMD access patterns.
- Coalesce Q/K/V and gate/up projections where one input fan-outs to multiple matrices.
- Choose row/column/block-major layouts per GEMV/GEMM shape rather than one universal layout.
- Eliminate redundant materialization, transposes, copies, and format conversions.
- Match cache-line and memory-transaction widths; quantify overfetch and partial transactions.
- Partition work to avoid HBM partition camping, bank conflicts, false sharing, and NUMA remote
  reads.
- Use huge pages/pinned allocations only when measured TLB/fault savings exceed downsides.
- Prepack at model load when repeated inference amortizes conversion cost; include setup in
  short-lived session accounting.

### 5. Fusion, persistence, and on-chip reuse

- Fuse normalization, projection, rotary position, residual, activation, sampling, and state
  update where dependencies permit.
- Persistent decode kernels keep weights, scales, recurrent state, or descriptors resident in
  registers/shared memory/cache across tokens.
- Cooperative/fused multi-layer kernels trade launch and DRAM traffic against occupancy and
  synchronization.
- Tile for SRAM/shared memory and register reuse; use warp specialization or producer/consumer
  groups to separate movement from arithmetic.
- Recompute cheap intermediates instead of storing and rereading them.
- Keep the smallest hot set—norms, router, state, top experts—resident before trying to pin the
  entire model.
- Graph/command-buffer capture can remove host gaps but must not be called a bandwidth gain.

### 6. Prefetch, asynchronous movement, and “speculative bandwidth”

Prefetch is useful only when four conditions hold: prediction is accurate, data arrives before
use, it does not evict more valuable data, and it does not steal bandwidth from demand traffic.

- Software prefetch the next tile/layer while computing the current tile/layer.
- Use CUDA async copies/TMA-style pipelines or Metal command-buffer/encoder overlap where the
  backend supports them; double- or triple-buffer with bounded memory.
- Prefetch weight scales and sparse indices with values, not as late dependent streams.
- Build separate copy and compute queues with explicit dependency and cancellation semantics.
- Predict MoE experts from router history, hidden-state approximations, token class, or a cheap
  early router; prefetch top-k plus calibrated reserve.
- Predict KV blocks/attention pages from recency, position, retrieval, or attention history.
- Prefetch from storage→host→device as a staged pipeline; measure each tier and overlap window.
- Cooperative multi-request scheduling can manufacture prefetch distance by running another
  ready request while one waits.
- Speculative prefetch may intentionally fetch alternatives. Score **useful bytes / fetched
  bytes**, timeliness, pollution, cancellation waste, and tail regressions.
- Adaptive prefetch depth should back off under low precision, contention, or cache pressure.

### 7. Speculative generation and accepted-token bandwidth

Speculation changes the denominator. It wins only if verified accepted tokens rise faster than
all target, draft, KV, and rejection traffic.

- Small draft models, self-speculation, layer skipping, early-exit drafts, Medusa-style heads,
  EAGLE-style feature drafts, and prompt/lookahead methods.
- Verify multiple candidate tokens in one target pass to reuse target weights across positions.
- Tree verification can improve acceptance but may expand KV/workspace traffic.
- Batch verification across requests when it preserves latency and acceptance.
- Cache or share target-side prefixes and draft states without copying more than recomputation.
- Adapt draft length and branching to observed acceptance, context, entropy, and load.
- Co-locate draft and target only if capacity contention does not lower target bandwidth; split
  devices only if link traffic is lower than saved target passes.
- Required metric: total target + draft + verification bytes and joules per **accepted** token,
  with quality/distribution equivalence and rollback behavior.

### 8. KV cache and recurrent state

- MQA/GQA/MLA reduce KV heads or latent dimensions; architecture conversion needs quality proof.
- KV INT8/INT4/INT2, mixed precision by layer/head/age, residual windows, and outlier handling.
- Low-rank, product-quantized, delta, entropy, and learned KV compression.
- Sliding windows, streaming sinks, heavy-hitter retention, attention-based eviction, retrieval,
  clustering, and token merging.
- Sparse/block attention avoids reading irrelevant history; prove selection overhead and quality.
- Paged versus contiguous/vAttention-style virtual allocation; tune page size from measured
  fragmentation, copy, cache, TLB, and scheduler behavior.
- Prefix block deduplication, copy-on-write, immutable shared blocks, and device-side clone
  avoidance.
- Tier KV by hotness across SRAM/cache, HBM, peer memory, host, CXL, storage, or remote nodes.
- For SSM/GDN/recurrent models, keep compact state device-resident, batch state updates, and
  avoid host round trips; compare state traffic against transformer KV honestly.

### 9. Batching, scheduling, and reuse manufacturing

- Continuous batching and iteration-level scheduling reuse loaded weights across active requests.
- SLO-aware dynamic batching finds the throughput/latency frontier rather than maximizing batch.
- Group by model, adapter, quant format, context/KV length, expert set, prefix, and memory tier.
- Split prefill and decode when disaggregation improves shapes or resource interference; include
  transfer and queue cost.
- Chunk prefill to prevent head-of-line blocking and create overlap opportunities.
- Schedule locality as a first-class objective: cache/expert/KV affinity plus fairness and aging.
- Use admission control from capacity *and bandwidth demand*, not just free bytes.
- Coordinate batch shape with kernel tile/layout choices; a scheduler can move arithmetic
  intensity across the roofline ridge.
- Exploit slack: while one stream waits on a dependency or transfer, run another whose working
  set is already resident.

### 10. MoE-specific traffic

- Count active expert, shared expert, router, state, dispatch, gather, and all-to-all bytes.
- Cache experts by measured popularity; reserve space for shared/hot experts and adapt under
  distribution shift.
- Predict and prefetch experts before final routing, with calibrated misprediction bounds.
- Batch and sort tokens by expert to turn tiny irregular reads into reusable GEMMs.
- Replicate hot experts to reduce communication; shard cold experts when capacity dominates.
- Co-locate experts that co-activate; rebalance from observed routing, not static assumptions.
- Compress experts independently, including mixed precision by popularity/sensitivity.
- Spill rare experts through staged host/peer pipelines only when overlap and tails remain green.
- Consider expert pruning/merging/distillation where quality permits.

### 11. Hierarchy, offload, and placement

- Model every tier by capacity, sustainable bandwidth, latency, granularity, concurrency, energy,
  and contention: registers/SRAM/L1/L2/SLC, HBM/VRAM/unified memory, peer GPU, host DRAM, CXL,
  NVMe, network, object storage.
- Place by hotness and next-use distance; migration should be explicit, observable, cancellable,
  and hysteretic.
- Pipeline layer/expert streaming so compute overlaps storage/host/device movement.
- Pin only proven hot pages; broad pinning can starve the OS and destroy other workloads.
- Compare offload against smaller quantization, sparse active sets, and remote execution.
- Treat unified memory faults and OS page cache as mechanisms to measure, not invisible magic.
- On Apple unified memory, distinguish shared physical capacity from effective GPU transaction,
  cache, residency, and command scheduling limits.
- On multi-socket CPU, bind threads and pages, interleave only when aggregate bandwidth beats
  remote-latency cost, and test channel saturation.

### 12. Multi-device and distributed memory

- Tensor, pipeline, expert, sequence, context, and data parallelism trade local bytes for link
  traffic differently.
- Replicate to avoid collectives only when capacity and update cost allow it.
- Overlap collectives with independent compute and use topology-aware rings/trees.
- Quantize/compress activations, KV, and collectives; include encode/decode and error impact.
- Route requests to where model/KV/prefix/expert state already resides instead of moving state.
- Move compute to data when remote execution beats remote memory reads.
- Use peer memory as a cache with explicit ownership and invalidation, not an accidental fallback.
- Measure incast, head-of-line blocking, congestion, retries, and tail amplification—not only link
  bandwidth.

### 13. Architecture and training co-design

- GQA/MQA/MLA, recurrent/SSM hybrids, local/global attention, sparse attention, and memory tokens
  change required state traffic.
- Distillation, smaller active models, cascades, routers, and early exits reduce demanded weights.
- Mixture-of-depths and conditional layers reduce active bytes per token.
- Train quantization/sparsity/low-rank/KV compression in rather than forcing post-training repair.
- Optimize quality per active byte and accepted-token joule, not parameter count.
- Separate changes compatible with existing GGUF artifacts from new-model/retraining programs.

### 14. Runtime adaptation and autotuning

- Choose kernel, layout, tile, quant path, batch, prefetch depth, residency, and speculation policy
  from the observed envelope.
- Maintain safe static fallbacks and confidence bounds; adaptation must be bounded and reversible.
- Search multi-objective frontiers: TTFT, inter-token tail, throughput, quality, energy, and cost.
- Detect regime changes (context crossover, batch ridge, expert shift, thermal throttling) and
  switch policies without oscillation.
- Cache tuning results by exact module/device/model/envelope identity and invalidate honestly.

### 15. Hardware and system levers

- Populate memory channels, use correct NUMA placement, enable supported clocks/power modes, and
  verify ECC/thermal effects rather than assuming nominal specifications.
- Compare devices by sustainable production-pattern bandwidth, cache, capacity, link topology,
  and software maturity—not GB/s alone.
- Use tensor/matrix cores only when dequant/layout shapes feed them efficiently.
- Exploit hardware sparse paths only with their exact structural constraints.
- Investigate larger-cache CPUs/GPUs, HBM generations, on-package memory, CXL expanders, DPUs,
  near-memory processing, and model-specific accelerators as measured placement options.
- Hardware purchase is a lever, but software that halves demanded bytes may beat twice the peak
  bandwidth at lower cost and power.

## Experiment contract for every lever

Every GitHub child under #9133 should answer:

1. **Term changed:** logical bytes, physical transactions, reuse, overlap, accepted tokens,
   placement, launch/sync, or quality-adjusted work.
2. **Envelope:** model/artifact digest, native engine, device, phase, prompt/context, batch/load,
   cache/residency, clocks/power, and module revision.
3. **Prediction:** a numerical or ordinal response before implementation.
4. **Counter-hypothesis:** what else could produce the same observation?
5. **Micro witness:** matched mechanism-level evidence, with raw samples.
6. **End-to-end witness:** TTFT, inter-token tails, accepted tokens/s, request throughput, total
   wall time, bytes/joules/cost per accepted token, and quality.
7. **Waste:** metadata, rejects, pollution, migration, setup, recovery, and verification.
8. **Stop rule:** retire, narrow, or escalate based on a predeclared threshold.
9. **Rollback:** safe fallback and state compatibility.
10. **Provenance:** measured versus estimated versus vendor-stated values remain distinct.

## Prioritization model

Rank work by expected net value, not theoretical multiplier:

```text
expected_value = probability_bottleneck_is_addressed
               * plausible_end_to_end_gain
               * envelope_frequency
               * confidence_in_quality
               / (implementation_cost + operational_risk + complexity_tax)
```

Start with cheap causal probes. A 2× byte reduction that does not move time is extraordinarily
valuable evidence: it retires an entire class of mistaken work. Promote only interventions that
move the quality-constrained end-to-end frontier.

Suggested first wave:

1. Complete #9128 so “bandwidth-bound” is measured rather than heuristic.
2. Build byte conservation and context/batch crossover sweeps.
3. Test one fused quantized projection that reduces actual DRAM transactions.
4. Test async next-layer/tile prefetch with precision, timeliness, and pollution counters.
5. Measure speculation in total bytes per accepted token, not target passes alone.
6. Test KV precision/retention at the context length where KV overtakes weights.
7. Add locality-aware batching/expert grouping when concurrency exists.
8. Re-rank all later work from those receipts.

## Research trail

Authoritative starting points, accessed 2026-08-26:

- Roofline model: <https://doi.org/10.1145/1498765.1498785>.
- FlashAttention and IO-aware exact attention: <https://arxiv.org/abs/2205.14135> and
  `Dao-AILab/flash-attention@0251105a2fb19d2957484b7f023cd8c115286ced`.
- CUTLASS pipelining/layout/collective implementation reference:
  `NVIDIA/cutlass@ffa119a1255d78998536107466cc7097ecefa393`.
- Speculative decoding: <https://arxiv.org/abs/2211.17192> and
  <https://arxiv.org/abs/2302.01318>.
- Medusa multi-head decoding: <https://arxiv.org/abs/2401.10774> and
  `FasterDecoding/Medusa@e2a5d20c048a9b0a4092e6933c34313687422518`.
- EAGLE feature-level drafting: <https://arxiv.org/abs/2401.15077> and
  `SafeAILab/EAGLE@cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b`.
- PagedAttention/vLLM: <https://arxiv.org/abs/2309.06180> and
  `vllm-project/vllm@c71f6f8a81d3d3c49a045c8b88eed36366cc7d92`.
- SGLang prefix reuse/scheduling: <https://arxiv.org/abs/2312.07104> and
  `sgl-project/sglang@f8cc1f9525c3a0bf3b14480cc76eccb79db1b4ea`.
- GQA: <https://arxiv.org/abs/2305.13245>.
- FlexGen offload/pipeline search: <https://arxiv.org/abs/2303.06865> and
  `FMInference/FlexGen@004ffef82b46e8dc8685c55d0cdda650bdaf1269`.
- AWQ: <https://arxiv.org/abs/2306.00978> and
  `mit-han-lab/llm-awq@d6e797a42b9ef7778de8ee2352116e0f48a78d61`.
- GPTQ: <https://arxiv.org/abs/2210.17323> and
  `IST-DASLab/gptq@2d65066eeb06a5c9ff5184d8cebdf33662c67faf`.
- SmoothQuant: <https://arxiv.org/abs/2211.10438>.
- SparseGPT: <https://arxiv.org/abs/2301.00774>.
- StreamingLLM: <https://arxiv.org/abs/2309.17453>.
- H2O heavy-hitter KV eviction: <https://arxiv.org/abs/2306.14048>.
- KIVI KV quantization: <https://arxiv.org/abs/2402.02750>.
- TensorRT-LLM serving/kernels:
  `NVIDIA/TensorRT-LLM@ca939b7baea13a8f3a7ecfd2fbb71807e772d0e5`.
- DeepSpeed inference/offload: `microsoft/DeepSpeed@715965e027894a2e72ac2e27f2daed2c599e99f0`.
- PyTorch quantization primitives: `pytorch/ao@b000ae4f5a1f4de8453e496a696135be1e2abf3e`.

External implementations are candidates to borrow from, never evidence that fak gained. Every
adoption still needs native execution identity, a matched envelope, quality, and net-true
receipts.
