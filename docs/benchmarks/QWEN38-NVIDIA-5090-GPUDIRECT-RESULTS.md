---
title: "Qwen3.8-27B & Flash Next NVIDIA RTX 5090 GPU Direct NVMe Model [SIMULATED] (2026-09-04)"
description: "BaM-style queue and hierarchical-memory modeling, historical illustrative values, and the missing physical witness for the NVIDIA RTX 5090 target."
---

# Qwen3.8-27B & Flash Next NVIDIA RTX 5090 GPU Direct NVMe Model [SIMULATED] (2026-09-04)

> ### PROVENANCE: IN-MEMORY MODEL, NOT A HARDWARE BENCHMARK
>
> **No performance figure in this document is a physical measurement of the RTX 5090, NVMe P2PDMA, or Qwen inference.** The historical figures, including 318.8 decode tok/s, 3,150.0 prefill tok/s, 24.50 ms TTFT, 7.1 GB/s, and 0.42 s context restoration, were illustrative model inputs/outputs, not measured baselines or calibrated hardware predictions. They are retained below only to identify and correct earlier claims.
>
> The current CLI requires explicit simulation for `bench`, `qwen38`, and `qwen38-bench`; `qwen38-sim` is the shorthand. Simulation remains in memory on any host. Hardware `inspect` reports observed inventory and `audit` leaves unsupported checks UNKNOWN/UNVERIFIED. Neither implements physical DMA or token generation.
>
> Pure Go tests can exercise descriptor serialization, queue arithmetic, memory-tier policy, and synthetic state round-trips. `StagingCopyCount() == 0` is an interface/model assertion, not a physical copy counter. Test success does not establish GPU queue residency, pinned host memory, MMIO, bandwidth, or model quality.
>
> **Status:** `[SIMULATED]`; physical backend integration and qualification remain separate work.
> **Representative target:** RTX 5090 FE / `sm_120`, Ryzen 9 5950X, Gigabyte X570 Aorus Elite WiFi, 128 GB DDR4, 2 TB NVMe SSD. This is a model profile, not observed inventory.
> **Tracking issue:** [#11326](https://github.com/anthony-chaudhary/fak/issues/11326).
> **Promotion requirements:** See [Section 8](#8-promotion-gate--the-missing-physical-witness).

Simulation schema: `fak.modelengine.qwen38-cudadirect-swap/1`
Explicit simulation attempt: `go run ./cmd/fak-dev cuda-gpudirect qwen38 --simulate --json`
Inventory/audit: `go run ./cmd/fak-dev cuda-gpudirect inspect --json` / `audit --json`
Subsystems: `internal/compute/cuda_gpudirect_storage.go`, `internal/compute/cuda_gpudirect_topology.go`, `internal/compute/hierarchical_memory.go`, `internal/model/qwen38_cudadirect_swap.go`.

---

## 1. System Platform and Hardware Topology Assumptions

The representative model uses:
- An RTX 5090 FE with a nominal 32 GiB VRAM capacity and an assumed full BAR1 aperture.
- A 2 TB NVMe SSD in a candidate CPU-attached `M2A_CPU` slot.
- A Ryzen 9 5950X / Gigabyte X570 platform with 128 GB host DRAM.
- Illustrative PCIe, DRAM, and GPU bandwidth inputs, not measured sustained rates.

The model's address constants, including BAR1 `0x8000000000` and an NVMe doorbell base, are placeholders. They do not map physical GPU or NVMe memory. The documented slot, ReBAR, ACS, IOMMU, driver, and root-complex properties require independent observation; they cannot be derived from a product name or PCI address prefix.

Hardware discovery now preserves available identity/resource observations without substituting this profile. Hardware route evaluation remains `UNKNOWN`, with no validated P2P path, optimality claim, latency, or bandwidth. Zero-valued performance fields are unmeasured placeholders, not measured zero-speed links.

## 2. The Memory Wall and 3-Tier Model

The architecture explores capacity beyond a GPU's local memory:
1. A dense hybrid Gated-DeltaNet model can accumulate full-attention KV pages plus convolutional and recurrent state as context grows.
2. A mixture-of-experts model can use an expert-residency policy across faster and colder tiers rather than keeping all weights resident.

The modeled hierarchy is:

| Tier | Representative capacity | Modeled role | Not established by the simulation |
| :--- | :--- | :--- | :--- |
| Tier 0 GPU VRAM | 32 GiB | Resident weights, active KV/GDN state, expert slots, queue descriptors | Actual GPU allocations or VRAM-resident NVMe queues |
| Tier 1 host DRAM | 128 GiB | Embedding/PLE placeholders, demotion targets | Page pinning, UVA mapping, or physical copy behavior |
| Tier 2 NVMe | 2 TB target | Serialized KV/GDN blocks and inactive expert/context records | NVMe controller ownership, DMA-BUF mapping, or direct DMA |

`HierarchicalMemoryManager` exercises policy and bookkeeping. The Qwen coordinator uses synthetic state and in-memory storage. The historical 2.37 GB embedding, 51 GB PLE, and 32 expert-slot values describe the representative configuration; they are not a validated loaded model footprint.

## 3. Core Architectural Mechanisms and Evidence Boundaries

### 1. BaM-style submission and completion queues

`CUDABaMVRAMQueue` models 64-byte NVMe submission entries and 16-byte completion entries, pointers, and doorbell bookkeeping. The intended architecture draws on BaM-style GPU-initiated storage, but the current simulation does not establish that GPU warps submit commands, that a controller consumes those queues, or that CPU/OS I/O overhead is eliminated.

### 2. Zero-staging descriptor contract

`StagingCopyCount() == 0` expresses the direct-path interface contract. It does not count actual CPU copies. Synthetic serialization and in-memory storage may copy host bytes; therefore a passing descriptor assertion cannot certify zero host DRAM bounce buffers in a physical transfer.

### 3. Candidate root-complex routing

The target layout prefers exploring a CPU-attached NVMe path. The earlier ~680 ns / <700 ns route value was a modeled assumption, not a measured round-trip latency. Hardware topology alone does not establish driver-supported P2PDMA or absence of ACS/IOMMU restrictions.

### 4. Expert-slot and embedding coordination

`BlackwellModelCoordinator`, `StreamMoEExperts`, and embedding bookkeeping exercise the intended hierarchy and synchronization contract. References to `cuStreamWaitValue64` describe the intended GPU synchronization mechanism, not proof that a CUDA stream wait ran. The model does not establish physical pinned memory, expert DMA, overlap, or absence of kernel stalls.

### 5. Context-restoration target

`Qwen38Max32KRestorationDuration` names a 450 ms target. The historical 420 ms value was not an achieved 32K-token hardware restoration. Synthetic state round-tripping checks bytes and structure, not restoration latency at the target model size.

---

## 4. Historical Illustrative Values, Not Benchmark Results

The following values appeared in earlier documentation/receipts. **Every arm, including the CPU-staged and external-reference columns, was illustrative rather than a measured matched run.** The ratios are arithmetic over those values, not evidence of an achieved speedup, model-quality equivalence, or a calibrated roofline.

The current CLI no longer emits these constants as performance measurements: its legacy numeric performance fields remain zero/unmeasured and are explained by `unknown_fields`.

| Metric | Historical CPU-staged assumption | Historical reference assumption | Historical fak model assumption |
| :--- | :---: | :---: | :---: |
| Host staging copies | 3 | 2 | 0, descriptor contract only |
| TTFT | 125.40 ms | 98.20 ms | 24.50 ms |
| Prefill | 920.5 tok/s | 1,140.0 tok/s | 3,150.0 tok/s |
| Decode | 52.4 tok/s | 64.8 tok/s | 318.8 tok/s |
| Decode ITL p50 | 19.08 ms | 15.43 ms | 3.14 ms |
| Decode ITL p95 | 35.10 ms | 28.50 ms | 5.20 ms |
| Storage bandwidth | 2.1 GB/s | 2.8 GB/s | 7.1 GB/s |
| 32K restoration | 2.85 s | 1.95 s | 0.42 s |

The former 6.08× decode, 5.12× TTFT, and 6.79× restoration ratios were computed from this table. They must not be cited as wins over CPU staging, llama.cpp, vLLM, or any other runtime. The earlier “theoretical hardware ceiling” values were not a validated bound for a complete inference workload either.

### Operating envelope and omitted effects

A useful future model needs a named workload, exact model and quantization, source/configuration, batch/concurrency, quality constraints, and reproducible input assumptions. Single-stream and batched workloads expose different bottlenecks; a model that assumes perfect overlap does not predict both.

Physical qualification must account for at least:
1. PCIe packetization, link negotiation, routing, and bus arbitration.
2. Host/GPU memory contention and refresh behavior.
3. Power limits, thermal state, clock changes, and sustained-run duration.
4. OS, driver, CGO, and Go-runtime scheduling overhead.
5. Workload-dependent expert routing, cache residency, and prefetch misses.
6. NVMe controller/flash behavior, queue depth, background work, and data-integrity costs.

These effects were not measured by the simulation. Their impact also cannot be assigned a reliable percentage without evidence. The [simulated-results discipline](../standards/simulated-results-discipline.md) and [net-true value standard](../standards/net-true-value.md) govern any later comparison.

## 5. External Architecture Comparisons

BaM, llama.cpp, vLLM, DeepSpeed, FlexGen, and TensorRT-LLM are useful architectural/reference candidates, not measured competitors in this artifact. The previous feature-comparison table asserted runtime capabilities, copy counts, consumer compatibility, and latency advantages without version-bound witnesses; those assertions are not supported here.

Any future comparison must pin each runtime/version, exact model and quality, hardware, workload, configuration, and complete setup/transfer/verification costs. Distinguish a reusable Go interface from an integrated serving path. No claim of superiority, parity, or consumer-hardware readiness follows from this simulation.

---

## 6. Machine-Readable Receipt Contract

There is no captured physical receipt in this document. The former hand-written `PASS` JSON example was not a hardware witness and is not a current successful-run artifact.

For the Qwen simulation schema `fak.modelengine.qwen38-cudadirect-swap/1`:
- A successful in-memory execution is labeled `provenance: MODELED`, `verdict: MODELED`, and `is_physical_silicon: false`.
- Legacy bandwidth, token-rate, timing, and speedup fields are zero placeholders for unmeasured performance, not measured zeros or validated comparisons. Read `unknown_fields` with them.
- Descriptor counts, byte accounting, or a synthetic state round-trip describe only the modeled operations completed by that run.
- A constructor, queue, or restore failure must return nonzero without a successful receipt. An available schema or example command is not evidence that a run completed.

Do not change the provenance string to manufacture a hardware receipt. Physical receipts will need an implemented physical runner and independently reviewable data, not a different label on this simulation.

## 7. Verification and Reproducibility Runbook

Keep hardware observation separate from explicit fixture execution. With `--sysfs` omitted, Linux uses `/sys` and Windows uses CIM display identity/driver inventory only, with no inferred VRAM/BAR1/ReBAR or fallback NVMe. Explicit nonempty `--sysfs PATH` selects the Linux sysfs parser on any platform. Simulation selects fixtures before probing; see the [diagnostics guide](../hardware/NVIDIA-RTX5090-BAM-P2PDMA-GUIDE.md) for the platform and exit-code contract.

Exit codes below describe the **`fak-dev` executable**. The `go run` examples return Go's wrapper status and print `exit status N` for a child failure, rather than forwarding the exact child code to the shell.

1. **Inspect available inventory:**
   ```bash
   go run ./cmd/fak-dev cuda-gpudirect inspect --json
   ```
   Successful enumeration exits 0, but the verdict remains UNVERIFIED. It does not certify BAR1/ReBAR, a named motherboard slot, ACS, or P2PDMA performance. Probe errors return 1 with diagnostics and any retained partial inventory.

2. **Read unsupported hardware checks:**
   ```bash
   go run ./cmd/fak-dev cuda-gpudirect audit --json
   ```
   The current audit reports UNKNOWN checks, `healthy: false`, and verdict UNVERIFIED with exit 1. `audit --simulate` likewise exits 1 with MODELED provenance/verdict, UNKNOWN checks, and `healthy: false`. This is missing capability evidence, not a fabricated all-PASS result or proof that the hardware is defective.

3. **Exercise the in-memory queue path:**
   ```bash
   go run ./cmd/fak-dev cuda-gpudirect bench --simulate --json
   ```
   The result is MODELED with `zero_copy_verified: false`; throughput and IOPS remain unmeasured. Without `--simulate`, `bench`, `qwen38`, and `qwen38-bench` refuse the unimplemented hardware path with usage exit 2. Invalid or surplus arguments also return 2 before probing.

4. **Attempt the Qwen state-coordination model:**
   ```bash
   go run ./cmd/fak-dev cuda-gpudirect qwen38 --simulate --json
   # Equivalent explicit simulation entry points:
   go run ./cmd/fak-dev cuda-gpudirect qwen38-bench --simulate --json
   go run ./cmd/fak-dev cuda-gpudirect qwen38-sim --json
   ```
   These are simulation entry points on every host, not physical inference commands. Preserve any failure; a constructor/restore error cannot be reported as successful model validation.

5. **Capture fresh evidence:** allocate a new, exclusively owned per-run evidence directory before redirecting output. Retain source identity, invocation, stdout, stderr, exit status, and receipt hashes. Never overwrite `docs/_witnesses/` history or imply a receipt exists merely because a command names an output path.

## 8. Promotion Gate & The Missing Physical Witness

The claim remains `[SIMULATED]` under [`CLAIMS.md`](../../CLAIMS.md), [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md), and the [native inference goal](../native-inference-goal.md). **The existing commands do not become physical when moved onto a Blackwell workstation.** Installing a driver, adding hardware, or removing `--simulate` cannot supply the missing backend.

### 1. Implement and admit the physical path

Integrate actual GPU allocations, supported peer-memory mapping, NVMe ownership/submission/completion, synchronization, and data-integrity checks. Establish exact device/driver/kernel compatibility and safe resource ownership before device writes. Keep inventory checks, synthetic tests, physical transfer tests, and real inference as separate evidence classes.

### 2. Capture a hardware-bound execution witness

The future runner must identify source/configuration, GPU UUID/PCI address, storage device, driver/kernel, model artifact and quantization, workload/concurrency, and timing method. Record measured transfers and data checks, not merely a `PHYSICAL_WITNESSED` label. Hardware traces and instrumented copy accounting must support the precise zero-copy claim; a utilization sample or `nvidia-smi` output alone is insufficient.

### 3. Verify model correctness and net-true performance

Run the model inside fak-native with matched quality and workload constraints. Validate restored state and resulting tokens against the specified reference, including a sufficiently long context and representative tasks. Include setup, transfer, recovery, and verification overhead in end-to-end latency/throughput; report distributions and failures, not only a favorable scalar.

### 4. Review calibration and promotion separately

The [hardware-simulation methodology](speculative-hardware-simulation-methodology.md) describes an 8% error-budget gate when a simulation is claimed to predict hardware. These historical constants are not a calibrated predictor and confer no performance entitlement. A future model must declare inputs and a calibration/holdout protocol; disagreement requires revising the model or claim, not altering a physical receipt to match it.

Only independent review of the implemented path and its committed physical evidence can support a scoped hardware claim. A green unit test, successful simulation, inventory enumeration, or source publication alone cannot promote this claim to `[SHIPPED]` performance.

---

## Related Documents

- [Benchmark Evidence Authority](../../BENCHMARK-AUTHORITY.md) — Canonical authority ledger for scoped benchmark rows and reproduction commands.
- [Benchmark Directory Index](README.md) — Repository benchmark sheets and runbooks.
- [Speculative Decoding and Hardware Simulation Methodology](speculative-hardware-simulation-methodology.md) — Evaluation funnel and simulation calibration gate.
- [Native Inference Goal](../native-inference-goal.md) — Performance invariant governing native model execution.
- [Net-True Value Standard](../standards/net-true-value.md) — Requirements for non-inflationary performance gains.
- [NVIDIA RTX 5090 BaM P2PDMA Planning and Diagnostics](../hardware/NVIDIA-RTX5090-BAM-P2PDMA-GUIDE.md) — Target planning, inventory limits, and explicit simulation commands.
