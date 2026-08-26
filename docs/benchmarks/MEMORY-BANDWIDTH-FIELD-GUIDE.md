---
title: "Memory bandwidth in fak-native inference"
description: "A measurement-first field guide to when memory bandwidth limits inference, how fak distinguishes it from adjacent bottlenecks, and how to benchmark and monitor it without turning a theoretical ceiling into a performance claim."
---

# Memory bandwidth in fak-native inference

> **TL;DR:** Memory bandwidth limits performance when data arrives slower than useful arithmetic can consume it.

For low-batch transformer decode, active weights often move once per token with little reuse.
That makes bandwidth a strong first hypothesis, not a universal answer. In fak, require phase
timing, arithmetic intensity, same-run counters, and a byte-reducing intervention to agree.

Try the existing small-model profiler with a local export:

```bash
go run ./cmd/modelprof -dir <model-export-dir> -prompt 16 -steps 24 -prefill 64 -out modelprof.json
```

Keep four questions separate: **capacity** (does it fit?), **ceiling** (what could matched
bandwidth sustain?), **diagnosis** (what stalled this run?), and **operations** (what pressure
is the live service experiencing?). High allocation, low compute utilization, or a vendor
bandwidth specification alone answers none of them. Only an end-to-end, quality-qualified
receipt supports a performance gain.
## The short model

The roofline model bounds attainable arithmetic rate:

```text
arithmetic intensity I = useful operations / bytes moved          [op/byte]
compute roof          = peak useful operations / second           [op/s]
memory roof           = measured bandwidth * I                    [op/s]
attainable rate        <= min(compute roof, memory roof)
ridge point            = compute roof / measured bandwidth        [op/byte]
```

Equivalently, for token generation:

```text
bandwidth-only tokens/s <= measured sustainable bytes/s / bytes moved per token
```

This is why quantization can help twice: it reduces resident capacity and bytes fetched per
token. It does **not** guarantee proportional speedup; dequantization, metadata, alignment,
cache behavior, kernel occupancy, launch overhead, synchronization, and quality constraints
still count. Use measured sustainable bandwidth for the relevant access pattern, not vendor
peak, and count traffic at the tier being diagnosed.

The model is most predictive when one resource dominates and traffic is known. It becomes a
weak approximation when kernels overlap compute and transfer, data is served by cache,
multiple memory tiers participate, sparse routing is irregular, or launch/host gaps leave the
device idle.

## Why decode and prefill differ

| Inference work | Typical reuse | First hypothesis | What changes the answer |
|---|---:|---|---|
| Single/few-stream decode GEMV | Low weight reuse per launch | Weight-bandwidth or launch limited | Fusion, cache residency, quant format, recurrent state, CPU/device waits |
| Batched decode | Reuses weights across requests | Moves toward compute-bound | Batch shape, scheduler efficiency, padding, KV lengths |
| Prompt prefill GEMM | Reuses weights and activations across many tokens | Often compute/occupancy limited | Short prompts, skinny matrices, unfused operators, host gaps |
| Long-context attention decode | Reads growing KV history | KV-bandwidth/capacity pressure | GQA/MQA, KV quantization, paging, prefix sharing, attention sparsity |
| MoE decode | Reads only routed experts, often irregularly | Active-weight bandwidth plus routing/communication | Expert residency, imbalance, all-to-all, batching by expert |
| Prefix restore/offload | Explicit copies | Interconnect/storage bandwidth or latency | Tier, block size, overlap, hit rate, copied bytes actually avoided |

A dense model with `P_active` bytes of weights that are read once per token has a first-order
`P_active / bandwidth` floor. An MoE model should use **active routed bytes**, not total model
size, and must add router/shared-expert/state traffic. KV bytes grow with sequence length and
can overtake weight traffic. For multi-device inference, the slow roof may instead be PCIe,
NVLink, host memory, storage, or network; label the tier.

## What this means for fak

fak owns the native execution path, so the useful unit is the complete, quality-constrained
request—not an isolated impressive kernel. Preserve the engine identity required by
[`native-inference-goal.md`](../native-inference-goal.md): a receipt must name a fak-native
backend. Reference backends are valid only when explicitly selected for benchmark, parity diagnosis,
migration/interoperability, or borrowing; they never substitute for native-performance evidence.

The current surfaces form a measurement ladder:

- `cmd/modelprof` profiles the small proven in-kernel model's decode and prefill, reporting
  per-op MACs, streamed weight bytes, arithmetic intensity, achieved GB/s, and a verdict
  against **measured** host bandwidth. Its uninstrumented `Session.Step` timing prevents the
  profiler's overhead from masquerading as clean throughput.
- `cmd/wfmembench` measures whole-file access patterns, separating cold/warm and sequential/
  shuffled behavior. It characterizes storage/page-cache feeding, not GPU HBM throughput.
- `fak nativeperf profile` consumes strict phase/counter bundles, preserves backend-specific
  evidence, requires native engine identity, and classifies a bottleneck without inventing
  equivalent Metal and CUDA counters. See
  [`NATIVE-PERFORMANCE-HILLCLIMB.md`](NATIVE-PERFORMANCE-HILLCLIMB.md).
- End-to-end experiment receipts carry output quality, envelope identity, TTFT, steady decode,
  and net accounting. A profile selects the next lever; a comparable receipt proves or rejects
  its value.

For fak's goals, bandwidth-reducing levers include lower-bit native weights/KV, fused kernels,
keeping recurrent/KV state device-resident, prefix reuse that avoids real work, active-expert
residency, and batching enough requests to reuse fetched weights. Each has a counter-hypothesis:
quantization may add expensive unpacking; fusion may reduce occupancy; batching may hurt latency;
prefix restore may copy nearly as much as recomputation; offload may trade HBM bytes for a
slower link. Measure the full envelope.

## Detection: evidence ladder

Do not infer a memory bottleneck from high memory allocation, low compute utilization, model
size, or a vendor bandwidth specification alone. Use this order:

1. **Lock the envelope.** Record model and artifact digest, quantization, backend and native
   engine, device, clocks/power policy, prompt/decode lengths, batch/concurrency, cache state,
   quality floor, and software revision (`module@rev`).
2. **Split phases.** At minimum: load/setup, prefill, first token, steady decode, verification,
   teardown. Record host gaps, dispatches, copies, synchronization, and per-layer attribution
   when available.
3. **Estimate bytes.** Separate weight, KV, activation/workspace, explicit copy, metadata, and
   inter-device traffic. Mark estimated traffic as estimated.
4. **Measure the local roof.** Run a bandwidth microbenchmark that matches device, memory tier,
   data type, size, direction, access pattern, and concurrency. A host STREAM-like result is not
   an HBM result; H2D bandwidth is not DRAM-kernel bandwidth.
5. **Collect counters.** For CUDA, preserve Nsight Compute metrics such as DRAM bytes/throughput,
   cache hit/traffic, achieved occupancy, SM throughput, and stall reasons, plus Nsight Systems
   launch/copy/wait timelines. For Metal, preserve GPU capture/Instruments dispatch duration,
   occupancy/limiter and bandwidth/cache statistics where exposed; unsupported counters remain
   absent rather than synthesized.
6. **Triangulate.** A credible bandwidth diagnosis combines (a) high achieved bandwidth relative
   to a matched measured roof, (b) low arithmetic intensity below the ridge, (c) phase time
   dominated by kernels moving those bytes, and ideally (d) a controlled intervention that
   reduces bytes and improves end-to-end time.
7. **Falsify neighbors.** Check launch count and idle gaps, compute saturation, occupancy,
   synchronization, CPU orchestration, page faults, thermal/power throttling, and interconnect
   copies. “Low SM utilization” can mean launch-bound or starved, not necessarily DRAM-bound.

Useful classifications are tiered and confidence-bearing, for example
`device_memory_bandwidth`, `cache_capacity/traffic`, `interconnect_transfer`, `compute`,
`launch_or_host`, `synchronization`, or `mixed/unknown`. Never collapse all of these into
“memory.”

## Benchmark protocol

### A. Capacity and traffic inventory

Record resident weight bytes, active expert bytes/token, KV bytes/token/layer and sequence
length, workspace peak, cache allocation/fragmentation, and bytes copied between every tier.
Capacity explains whether an envelope can run; it does not prove throughput.

### B. Matched microbenchmarks

Measure each plausible roof independently:

- host DRAM: read/write/copy with working sets above last-level cache and NUMA placement pinned;
- GPU/accelerator memory: device-resident streaming kernels at matched allocation size and
  access width;
- H2D/D2H and peer links: both directions, transfer sizes, pinned/pageable state, and overlap;
- storage/page cache: cold and warm, sequential and randomized (`wfmembench`);
- KV/prefix copy: actual production block sizes and destination tier.

Report median plus tails and run count; retain raw samples. Do not compare measured achieved
bandwidth with a marketing peak without labeling both.

### C. Kernel profile

Capture warm steady-state kernels and a cold request separately. Attribute bytes, operations,
time, and counters by phase/op/layer. Calculate arithmetic intensity from the same definitions
used for bytes. Instrumentation changes timing, so pair the profile with an uninstrumented run.

### D. End-to-end matrix

Sweep one dimension at a time across prompt length, decode length, batch/concurrency, context
length, quantization/KV format, cache cold/warm/hit, and residency/offload tier. Always include:

- TTFT, inter-token latency distribution, tokens/s, request throughput, and total wall time;
- quality/correctness result and native engine identity;
- setup, transfer, recovery, and verification overhead;
- energy/power when the claim is efficiency;
- an incumbent only under the matched-envelope rules in
  [`production-benchmark-methodology.md`](../production-benchmark-methodology.md).

A strong causal test changes predicted bytes while holding semantics and the rest of the
envelope fixed. If halving weight bytes leaves decode unchanged and achieved bandwidth was low,
look at launches, unpacking/compute, synchronization, or measurement error instead.

## Live monitoring

Live telemetry should answer **when**, **which tier**, and **which request shape**—not pronounce a
roofline verdict from one gauge.

### Always-on, low overhead

Export per backend/device and request class:

- TTFT, inter-token latency, tokens/s, queue time, active batch/concurrency, prompt/decode/KV
  lengths, prefix/cache hit state, and native engine identity;
- device memory used/free, allocation failures, evictions, KV occupancy/fragmentation, bytes
  copied/offloaded/restored, page faults, and OOM/recovery counts;
- GPU/accelerator busy, memory-controller utilization where available, power, clocks,
  temperature/throttling, and host CPU/RSS/I/O;
- launch/dispatch count and synchronization/idle time if inexpensive.

Sample device telemetry at a bounded cadence and correlate it with request/phase IDs. DCGM or
vendor exporters are appropriate for fleet health; their utilization percentages are coarse
signals, not byte-accurate kernel proof. Alert on sustained windows and service symptoms, for
example rising inter-token latency plus high memory-controller activity at stable clocks—not on
one instantaneous “memory util” sample.

### Triggered deep capture

When a regression, saturation window, or canary fires, preserve a short scrubbed Nsight
Systems/Compute or Metal capture plus the matching fak receipt. Deep profilers are too invasive
for continuous use. Compare against an envelope-local baseline and retain unsupported counters
as missing. The profile bundle should drive exactly one next experiment through `fak nativeperf
profile`; only the subsequent end-to-end receipt can claim a gain.

### Minimal dashboard

Put these panels on one aligned time axis: request latency/throughput, queue and batch shape,
phase durations, device compute and memory activity, HBM/VRAM occupancy, KV/cache state,
copy/interconnect rate, dispatch/idle time, clocks/power/temperature, and deploy/model/engine
revision markers. This makes bandwidth pressure distinguishable from queueing, throttling,
launch storms, and cache churn.

## Interpretation traps

- **“The model is large, therefore bandwidth-bound.”** Size predicts traffic only after active
  bytes, cache residency, sparsity, and batching are known.
- **“Memory utilization is 95%.”** Capacity utilization and controller activity are different;
  neither alone measures achieved GB/s.
- **“The kernel reached peak bandwidth.”** Confirm the denominator, memory tier, read/write mix,
  cache contribution, clocks, and profiler replay behavior.
- **“Decode is bandwidth-bound, therefore prefill is too.”** Their matrix shapes and reuse differ.
- **“A faster microkernel means faster serving.”** Net-true setup, scheduling, copies, quality,
  and tails can erase it.
- **“More bandwidth always wins.”** Work may be compute-, launch-, synchronization-, latency-,
  capacity-, or communication-bound, and cost/power can dominate the product objective.

## Source trail

These are starting points, not imported claims. Accessed 2026-08-26.

- Williams, Waterman, Patterson, *Roofline: An Insightful Visual Performance Model for
  Multicore Architectures* (CACM 2009), DOI: <https://doi.org/10.1145/1498765.1498785>.
- NVIDIA, *GPU Performance Background User's Guide* (memory hierarchy, latency hiding, and
  throughput): <https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html>.
- NVIDIA, *Matrix Multiplication Background User's Guide* (arithmetic intensity and tile/reuse
  effects): <https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html>.
- NVIDIA, *Nsight Compute Profiling Guide* (roofline and hardware-counter methodology):
  <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>.
- NVIDIA, *Nsight Systems User Guide* (timeline, launch, synchronization, and transfer capture):
  <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>.
- NVIDIA DCGM documentation and source (`NVIDIA/DCGM@64df9f894541e426e416131a9820cae97aa4dd81`),
  for fleet telemetry semantics: <https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html>.
- Pope et al., *Efficiently Scaling Transformer Inference*, arXiv:2211.05102 (2022), for
  inference scaling, partitioning, and model/attention memory costs:
  <https://arxiv.org/abs/2211.05102>.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with
  PagedAttention*, arXiv:2309.06180 (2023), for KV allocation/fragmentation and serving
  throughput: <https://arxiv.org/abs/2309.06180>.
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*,
  arXiv:2312.07104 (2023), for prefix/KV reuse at serving level:
  <https://arxiv.org/abs/2312.07104>.

Repository implementations are inspiration only, not proof for fak: vLLM
`080a66a69c6fd1fe464756f88ab958baad66ce69`, SGLang
`f8cc1f9525c3a0bf3b14480cc76eccb79db1b4ea`, and TensorRT-LLM
`ca939b7baea13a8f3a7ecfd2fbb71807e772d0e5` were inspected as current serving reference points.
Their metrics and scheduling choices require a matched fak-native experiment before adoption.
