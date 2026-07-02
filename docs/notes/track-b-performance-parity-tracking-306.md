---
title: "fak Track B performance-parity tracker: the 8 children, honest on-disk status, and the gate that keeps the epic open"
description: "Umbrella tracker for epic #306 (Track B - Performance Parity). Records the CORRECT child issue map (the epic body's links are stale), each child's honest shipped/scaffold/pending status with deciding commits, and the named gate — hardware-gated wall-clock measurement + genuinely-unimplemented children — that keeps the roll-up open. No number is invented."
---

# Track B — Performance Parity (epic #306) — status tracker

> **Umbrella tracker for [#306](https://github.com/anthony-chaudhary/fak/issues/306)** —
> *"close prefill gap (1.76×), Vulkan optimization (100×), speculative decoding, continuous
> batching, fused kernels, PagedAttention, INT4/INT2, dynamic batching."*
>
> **#306 is a pure roll-up: it closes only when all 8 children close.** This doc does not
> claim the epic is done — it records the **honest on-disk state** of each child so an
> operator (or the next agent) can see exactly what shipped, what is hardware-gated, and
> what is not implemented at all. **House rule: every number comes from a real run.** The
> wall-clock parity cells are deferred to a CUDA/GPU bench node and say so plainly; no
> tok/s figure is invented here. Written **2026-06-25** on a win32 dev box (no CUDA, no
> discrete GPU reachable from this host).

---

## 0. The child map — the epic's own links are stale (read this first)

The migrated epic body (#306) lists its children as **#280, #282, #285, #287, #290, #292,
#294, #297**. Those numbers are **wrong** — they are pre-migration ids that the tracker
renumbered, and several now point at unrelated issues (e.g. #280 is a *closed docs* issue,
#290 is *"Vision/Multimodal Support [A-012]"* on **Track A**). The epic's note *"cross-
references are auto-maintained"* is therefore not holding for this epic. The **real**
children — the issues actually tagged `track/B-performance` with a `B-00x` slug — are:

| Slug | Live issue | Priority | Title | Epic body shows (WRONG) |
|---|---|---|---|---|
| **B-001** | [#289](https://github.com/anthony-chaudhary/fak/issues/289) | P0 | Close Prefill Throughput Gap | #280 |
| **B-002** | [#287](https://github.com/anthony-chaudhary/fak/issues/287) | P0 | Vulkan Backend Optimization | #282 |
| **B-003** | [#284](https://github.com/anthony-chaudhary/fak/issues/284) | P1 | Speculative Decoding | #285 |
| **B-004** | [#282](https://github.com/anthony-chaudhary/fak/issues/282) | P1 | Continuous Batching Integration | #287 |
| **B-005** | [#279](https://github.com/anthony-chaudhary/fak/issues/279) | P1 | Fused Kernel Optimization | #290 |
| **B-006** | [#277](https://github.com/anthony-chaudhary/fak/issues/277) | P2 | PagedAttention Implementation | #292 |
| **B-007** | [#275](https://github.com/anthony-chaudhary/fak/issues/275) | P2 | INT4/INT2 Quantization | #294 |
| **B-008** | [#272](https://github.com/anthony-chaudhary/fak/issues/272) | P2 | Dynamic Batching Optimization | #297 |

> The children's own migrated bodies still cite the **internal** epic id *"Epic #262"*; the
> live GitHub umbrella is **#306**. Both refer to the same Track B.

This doc keys everything off the **live** numbers above.

> **Operator action (the one GitHub-side fix):** for #306's *"closes when all children
> close"* mechanism to actually fire, repoint the epic body's checkbox list from the stale
> ids to the live ones — `#280→#289, #282→#287, #285→#284, #287→#282, #290→#279,
> #292→#277, #294→#275, #297→#272`. Until then the epic tracks the wrong (and partly
> closed/Track-A) issues and will never auto-close from its children. This in-repo tracker
> is the authoritative map in the meantime.

---

## 1. Honest status — each child, verified against the working tree (2026-06-25)

Status is read from the tree + `git log`, not from any worker's say-so. "Scaffold" means a
CPU-correct, bit-exact, test-witnessed host implementation whose *throughput acceptance* is
a device measurement deferred to a GPU node (the repo's standard honest form — see
[`internal/compute/PREFILL-B001-NOTES.md`](https://github.com/anthony-chaudhary/fak/blob/main/internal/compute/PREFILL-B001-NOTES.md)).

| Slug · issue | On-disk state | What is real today | Deciding file / commit |
|---|---|---|---|
| **B-001** #289 prefill | 🟡 **Scaffold shipped; perf CUDA-gated** | `PrefillCostModel` (exact FLOP/byte roofline, surfaces the O(P²) attention term), `PrefillGEMM` **bit-exact** to `Backend.BatchedMatMul`, `PrefillGraphCapturer` HAL seam (CUDA-satisfied under `-tags cuda`). CPU suite green. | [`internal/compute/prefill.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/compute/prefill.go) · notes [`PREFILL-B001-NOTES.md`](https://github.com/anthony-chaudhary/fak/blob/main/internal/compute/PREFILL-B001-NOTES.md) |
| **B-002** #287 Vulkan | 🟡 **Real backend; numerical parity; throughput NOT parity** | Real Windows Vulkan backend (`-tags vulkan`) reaching **argmax-exact greedy decode, prefill cosine 1.0** on a real AMD **RX 7600** (SmolLM2-135M). Still **~60× slower** than llama.cpp CPU; named, fixable cause = **per-dispatch CPU/driver overhead** (not compute efficiency). | [`internal/compute/vulkan.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/compute/vulkan.go) · witness [`docs/benchmarks/VULKAN-AMD-RESULTS.md`](../benchmarks/VULKAN-AMD-RESULTS.md) |
| **B-003** #284 spec decode | 🔴 **Not implemented** | No draft-model API, no parallel verify-accept loop, no acceptance-rate metric. (The only nearby code is a *dynamic-precision* Q8→f32 rollback comment in `internal/model/kv.go` — that is precision policy, **not** speculative decoding.) Design-only. | — (none) |
| **B-004** #282 continuous batching | 🟡 **Native lifecycle scheduler shipped; production serving bars open** | Multi-user batched decode `StepBatch` + ragged `StepBatchActive` (idle-lane skip), **bit-identical to serial** (`TestBatchedDecodeMatchesSerial`); `modelengine.NativeScheduler` now backs the registered `inkernel` lifecycle path (#401) and shows B8 1.54× req/s vs the legacy per-request lifecycle on the synthetic CPU witness. Still not vLLM-class serving: no paged attention and no multi-tenant SLA p99 claim. | [`internal/model/batch.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/model/batch.go) · [`internal/modelengine/nativesched.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/modelengine/nativesched.go) · `experiments/modelengine/native-continuous-batching-20260629.json` |
| **B-005** #279 fused kernels | 🟡 **Built + GPU-gated** | Fused **flash / online-softmax attention** CUDA kernel (MHA/GQA/MQA), correctness-gated cosine ≥ 0.999; Vulkan FFN-tail + Q8 fused-decode kernels. Correctness witnessed; **throughput gain is a GPU measurement** (`tools/run_486_acceptance_on_gpu.sh`). | [`internal/compute/cuda_flash_test.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/compute/cuda_flash_test.go) · `cuda_kernels.cu` · commit `49d445b` |
| **B-006** #277 PagedAttention | 🔴 **Not implemented** | No page-based KV allocator, no page table, no copy-on-write. `nativesched.go` lists paged KV as an explicit **non-goal**; the paged/block KV allocator is the sibling design issues #33/#34, unshipped. Design-only. | — (non-goal note `nativesched.go:23`) |
| **B-007** #275 INT4/INT2 | 🟡 **INT4 shipped; INT2 absent** | **Q4_K** full load+run path (CPU ref `q4kRowDot`, CUDA `k_q4k_gemm`, Metal fused dequant) + **Q4_0** load/stream; Q4_K device GEMM stays int4-resident (`#485`). **INT2 / Q2 is not present** — not in the `Dtype` enum, no forward kernel (Q2_K can *load* but has no compute path). | [`internal/compute/quant_q4k.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/compute/quant_q4k.go) · [`internal/ggufload/gguf_dequant.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/ggufload/gguf_dequant.go) · commits `e1513da`/`a8cb3fc` |
| **B-008** #272 dynamic batching | 🟡 **Partial native scheduler shipped; B2/p99 serving gate open** | `StepBatchActive` compacts active lanes into a sub-panel (padding/idle work removed, measured by `LastStepMACs()`); `NativeScheduler` dynamically admits/retires lanes between decode steps and `FAK_NATIVE_MAX_RUNNING` caps the running set. The #401 witness improves throughput but measures B2 1.34×, so the stricter B2≥1.8× and p99≤1.5× serving bars remain open. | [`internal/model/batch.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/model/batch.go) · [`internal/modelengine/nativesched.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/modelengine/nativesched.go) · `experiments/modelengine/native-continuous-batching-20260629.json` |

Legend: 🟢 done · 🟡 partial (scaffold / built-but-gated) · 🔴 not implemented.

**Roll-up: 0 / 8 children closeable today.** None of the 8 has met its *acceptance* gate, so
#306 stays **OPEN** — correctly.

---

## 2. The gate — why #306 cannot honestly close here

The eight acceptance gates fall into three classes, and **none is reachable on this host**:

1. **Hardware-gated wall-clock measurement (B-001, B-002, B-005).** The acceptances are
   tok/s ratios on a CUDA / AMD-Vulkan device ("within 1.2× llama.cpp Q8_0 at P=256/512/1024";
   "decode within 2× llama.cpp on RX 7600, ≥ 50 tok/s"; "15%+ throughput gain on long
   sequences"). This box has no CUDA toolkit and no discrete NVIDIA GPU; the Vulkan number
   is already measured on the RX 7600 and is **honestly NOT at parity** (~60× off). Per
   BENCHMARK-AUTHORITY rules, no parity tok/s is asserted — the cells stay deferred. The
   nearest live perf loop and its honesty gate are
   [`docs/perf-parity-rsi-loop.md`](../perf-parity-rsi-loop.md) and
   [`docs/benchmarks/GUARD-HOP-OVERHEAD-PENDING.md`](../benchmarks/GUARD-HOP-OVERHEAD-PENDING.md).

2. **Genuinely unimplemented children (B-003 speculative decoding, B-006 PagedAttention,
   and the INT2 half of B-007).** These are not "scaffold + gated" — there is no
   implementation to measure. Closing them is days-of-engineering features (a draft-model
   verify-accept loop; a paged/block KV allocator with COW; an INT2 dtype + kernel), each a
   separate leaf, not a single small change.

3. **Production-scheduler children (B-004, B-008).** The batching *kernel* and the native
   in-kernel lifecycle scheduler now ship and are bit-exact, but the acceptances
   ("compatible with vLLM continuous batching, throughput within 10% of direct vLLM";
   "near-linear scaling 1.8× on 2× requests, p99 within 1.5×") still require the production
   serving layer: paged KV, preemption/fairness/admission on the live serve loop, and p99
   measurement on serving hardware.

**The single honest gate, stated plainly:** epic #306 closes only when all 8 children meet
their acceptance, and **3 of the 8 gates are wall-clock measurements on GPU hardware not
attached to this host**, while **3 more children are unimplemented features**. No code or
doc change on this host can flip those bits without fabricating a number — which the repo's
witness ledger and `make claims-lint` would reject. So the correct deliverable is this
tracker, not a closed epic.

---

## 3. Smallest next step per child (for the agent that picks one up)

| Child | Smallest honest next step | Where it runs |
|---|---|---|
| B-001 #289 | Lower `PrefillGEMM` tiling into the device GEMM, wrap `CapturePrefillGraph` around the prefill op-stream, run the timed P=256/512/1024 bench vs llama.cpp Q8_0 | a CUDA bench node |
| B-002 #287 | Land pre-baked descriptor bindings + Q8 device GEMM + async compute to cut per-dispatch overhead; re-measure on the RX 7600 | the AMD/Vulkan box |
| B-003 #284 | Implement the draft-model API + parallel verify-accept (SmolLM2 as draft), bit-exact vs non-spec; add acceptance-rate metric | host-tractable (CPU correctness first) |
| B-004 #282 | Define the request-scheduling + KV-sharing interface against the `StepBatch` kernel; bench adjudication latency (≤ 100µs) | host-tractable + a vLLM peer |
| B-005 #279 | Run `tools/run_486_acceptance_on_gpu.sh` to record the fused-vs-unfused throughput/memory-traffic delta | a CUDA node |
| B-006 #277 | Build the paged/block KV allocator (issues #33/#34) preserving the bit-exact mid-span Evict witness, then COW for cache sharing | host-tractable (correctness) |
| B-007 #275 | Add an INT2/Q2 `Dtype` + a CPU forward kernel + accuracy bench (≥ 90% of INT8); INT4 is already shipped | host-tractable |
| B-008 #272 | Add dynamic batch-size selection + batch-aware scheduling over `StepBatchActive`; padding-overhead + p99 sweep | host-tractable + bench |

---

## 4. Provenance

- **Child set:** `gh issue list --label track/B-performance --state all` (live, 2026-06-25);
  acceptance criteria from each child body (#289/#287/#284/#282/#279/#277/#275/#272).
- **On-disk evidence:** `internal/compute/prefill.go`, `internal/compute/PREFILL-B001-NOTES.md`,
  `internal/compute/vulkan.go`, `internal/model/batch.go`, `internal/modelengine/nativesched.go`,
  `internal/compute/cuda_flash_test.go`, `internal/compute/quant_q4k.go`,
  `internal/ggufload/gguf_dequant.go`; commits `76b931a` (#520 ragged batch),
  `49d445b` (#486 fused flash attn), `e1513da`/`a8cb3fc` (#485 Q4_K device GEMM).
- **Measured baselines (cited, not restated):** [`docs/benchmarks/VULKAN-AMD-RESULTS.md`](../benchmarks/VULKAN-AMD-RESULTS.md)
  (RX 7600 numerical parity + the ~60× throughput gap), [`docs/notes/gpu-parity-tracking-480.md`](gpu-parity-tracking-480.md)
  (the GPU tok/s protocol + pending cells).
- **Honesty rails:** `make claims-lint`, BENCHMARK-AUTHORITY, and the perf-loop `--check`
  gate ([`docs/perf-parity-rsi-loop.md`](../perf-parity-rsi-loop.md)) — every parity tok/s
  cell here is deferred, never asserted.

## 5. See also

- [`docs/serving/dual-track-serving-plan.md`](../serving/dual-track-serving-plan.md) — the RIDE/NATIVE serving spine the batching/scheduler children build on.
- [`docs/notes/gpu-parity-tracking-480.md`](gpu-parity-tracking-480.md) · [`docs/notes/simd-cpu-parity-tracking-400.md`](simd-cpu-parity-tracking-400.md) — sibling parity trackers in the same house format.
- [`docs/perf-parity-rsi-loop.md`](../perf-parity-rsi-loop.md) — the gateway/forward RSI loop whose fitness signal is the live guard-hop number.
