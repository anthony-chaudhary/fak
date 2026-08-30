---
title: "Local inference platforms: dated 64/128 GB inventory"
description: "A source-addressable 2026-08-27 inventory of AMD Strix Halo, Apple unified-memory, NVIDIA compact-appliance, and CPU/high-memory platforms for long-context estimation."
---

# Local inference platforms — 2026-08-27

**Verdict:** issue #9573 now has a compact, dated hardware-input inventory for the
[`internal/modelperfobs` long-context estimator](../../internal/modelperfobs/long_context_estimator.go).
AMD Strix Halo leads because it is the commercially meaningful 64/128 GB local shared-memory
option the issue asked to resolve first. The inventory does not rank products or predict token
throughput.

Machine-readable authority:
[`inventory.json`](../_witnesses/issue-9573-local-inference-platforms/inventory.json).
Method and validation:
[`README.md`](../_witnesses/issue-9573-local-inference-platforms/README.md).

## Comparison

| Platform (US configuration observed 2026-08-27) | Memory boundary | Conservative inference working set | Bandwidth evidence | Compute/software boundary | Power boundary | Price boundary |
|---|---|---:|---|---|---|---:|
| **Framework Desktop / Ryzen AI Max+ 395 / 64 GB** | 256-bit LPDDR5x-8000 shared by CPU and Radeon 8060S; **not VRAM** | **48–52 GiB** after a 12–16 GiB planning reserve | 256 GB/s official peak; ~215 GB/s maximum GPU MBW measured on related pre-production 128 GB Framework hardware | 40-CU RDNA 3.5 + Zen 5 AVX-512; ROCm/Vulkan are usable but version-sensitive | 120 W sustained, 140 W processor boost; 400 W PSU | **$1,959** DIY system selection; storage/OS/fan/cable omitted |
| **Framework Desktop / Ryzen AI Max+ 395 / 128 GB** | same shared architecture; Windows cites up to 96 GB dedicated, AMD reference cites up to 112 GB allocatable; neither is extra memory | **92–104 GiB** after a 24–36 GiB planning reserve | 256 GB/s official peak; ~215 GB/s maximum GPU MBW on pre-production Framework hardware | same 40-CU path; large capacity is the differentiator, software remains less settled than CUDA/Metal | 120 W sustained, 140 W processor boost; 400 W PSU | **$3,449** DIY system selection; storage/OS/fan/cable omitted |
| **Apple Mac Studio / M5 Max 40-core GPU / 128 GB** | Apple unified CPU/GPU memory; **not discrete VRAM** | **96–108 GiB** after a 20–32 GiB planning reserve | 614 GB/s official peak; sustainable result **null** because the system was a preorder | cohesive Metal/MLX path; M5-specific shipping evidence not yet available | 480 W system maximum continuous, not measured inference draw | **$5,399** configured estimate; preorder, availability 2026-09-22 |
| **`NVIDIA DGX Spark / GB10 / 128 GB`** | coherent LPDDR5x shared by Grace CPU and Blackwell GPU; **not GDDR/HBM VRAM** | **104–112 GiB** after a 16–24 GiB planning reserve | 273 GB/s official peak; GPU inference sustainable result **null**; CPU STREAM retained only as context | mature `CUDA/DGX` stack with newer GB10/SM121 kernel/container caveats; sparse FP4 peak is not a dense rate | 140 W GB10 TDP; 240 W supply | **$4,699** reported list; **$5,299.99** one observed retail listing |
| **Dell Precision 7875 / Threadripper PRO 7975WX / 128 GB** | eight-channel DDR5-5200 ECC CPU memory; the included 4 GB RTX A400 does not convert it to VRAM | **108–116 GiB** after a 12–20 GiB planning reserve | 332.8 GB/s theoretical pin rate; same-Dell sustainable result **null**; 206.1 GB/s only on a comparable 8-DIMM OEM system | broad x86-64 CPU-runtime compatibility, but much lower low-precision acceleration than current GPUs | 350 W CPU TDP; 1,350 W configured PSU | **$19,897.33**, Ubuntu + A400 + memory, explicitly no storage |

The usable-memory ranges are assumptions for initial fit exploration, not measurements. Their
explicit reserves prevent the common mistake of feeding installed shared memory directly into
`UsableMemoryBytes`.

## What can enter the estimator

The estimator takes bytes, bandwidth bounds, compute bounds, and an explicit efficiency range.
This inventory supplies:

1. **Capacity normalization.** Vendor memory labels remain verbatim, while each row includes an
   estimator byte count and a conservative usable-byte range.
2. **Bandwidth evidence without false precision.** Official peaks, GPU bandwidth probes, host
   STREAM, and comparable-topology measurements remain separate. Where no workload-matched
   sustainable range exists, the estimator-ready range is `null`.
3. **Format fences.** AMD's FP16 reference peak, NVIDIA's sparse FP4 PFLOP, Apple's Neural
   Accelerators, and NPU TOPS are not interchangeable `ComputeFLOPS` values.
4. **Operating boundaries.** Power, network, commercial status, missing components, and price
   date travel with the hardware row.

These rows are **hardware inputs, not measured fak-native Qwen3.8 or GLM-5.3 performance**.
The next valid performance step is a receipt naming the fak-native engine, exact model artifact,
quantization, context/concurrency envelope, resident memory, measured bandwidth range, wall power,
quality gate, and completed-job time.

## Method

### Outward evidence

The machine-readable ledger contains **12 authoritative** and **7 third-party** sources. Official
sources establish product configuration, architecture, peak specifications, availability, and
list boundaries. Third-party evidence is restricted to:

- the ~215 GB/s Strix Halo maximum GPU-memory-bandwidth observation;
- host STREAM context for AMD Halo and `NVIDIA DGX Spark`;
- a comparable Threadripper PRO eight-DIMM memory measurement;
- current Apple configuration deltas where the dynamic 128 GB page did not expose a stable total;
- `NVIDIA DGX Spark` list/street observations when NVIDIA Marketplace itself timed out.

Every source carries observation date, source event date where available, state, context, refresh
trigger, and `INSPIRE-ONLY` disposition. No external expressive implementation was copied.

### Inward witness

The capability seam is **PRESENT**:
[`LongContextEstimatorInput`](../../internal/modelperfobs/long_context_estimator.go) already owns
`UsableMemoryBytes`, `BandwidthBytesPerSec`, `ComputeFLOPS`, and `Efficiency`. The dated commercial
inventory was **ABSENT** from the existing
[`Hardware Catalog`](HARDWARE-CATALOG.md),
[`Hardware Matrix`](../HARDWARE-MATRIX.md), and per-machine
[`specs.json`](../../experiments/benchmark/machines/) records. `fak study search` returned no
matching durable receipt. `fak-dev` was unavailable, so the dev self-index could not run; direct
repository inspection was the declared fallback.

### Reserve accounting

The conservative working set subtracts an explicit planning reserve:

- **64 GB shared systems:** 12–16 GiB for OS, display, drivers, runtime, workspaces, and
  fragmentation.
- **128 GB Strix Halo:** 24–36 GiB because GPU-addressable policy and OS choice materially change
  the pool visible to the accelerated path.
- **128 GB Apple:** 20–32 GiB because the configuration was not shipping and no M5 pressure trace
  existed.
- **128 GB `NVIDIA DGX Spark`:** 16–24 GiB for `DGX OS`, CUDA/runtime state, services, and workspaces.
- **128 GB CPU workstation:** 12–20 GiB for Ubuntu, page cache, runtime workspaces, and services.

These reserves are deliberately `assumption_speculative`. Replace them with live pressure and
resident-set evidence before a fit claim.

## Interpretation fences

- **Shared/unified memory is not VRAM.** It removes a hard host/device partition, but the OS,
  display, drivers, runtime, and other processes still consume the same pool.
- **Advertised bandwidth is not sustainable bandwidth.** It is a ceiling. CPU STREAM, GPU
  microbenchmarks, and end-to-end inference expose different paths.
- **TOPS is not a throughput prediction.** Precision, sparsity, accumulation, model architecture,
  batch, memory traffic, and kernel availability determine how much of a peak is relevant.
- **Purchase price is not completed-job cost.** A valid API/local comparison must use
  quality-qualified completed-job cost and completion time, including queueing, setup, retries,
  rejected work, failures, power, migration, and operator burden.
- **No silent engine substitution.** A future local measurement must name the fak-native engine;
  a llama.cpp/MLX/vLLM result can be a reference or borrowing witness, not a fak-native result.

## Evidence gaps

1. No production Framework 64 GB GPU-bandwidth capture; the ~215 GB/s observation is from a
   pre-production 128 GB system.
2. No workload-matched sustainable lower/upper bandwidth range for any row.
3. No shipping M5 Max 128 GB memory-pressure, bandwidth, power, or street-price evidence as of
   2026-08-27.
4. No GPU-counter sustainable bandwidth result for `NVIDIA DGX Spark`; host STREAM is not a substitute.
5. No same-configuration Dell Precision 7875 STREAM/counter result.
6. No row has a quality-qualified fak-native Qwen3.8 or GLM-5.3 completed-job receipt.
