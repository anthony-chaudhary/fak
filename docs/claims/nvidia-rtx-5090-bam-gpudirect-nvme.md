---
title: "NVIDIA RTX 5090 BaM-style Storage & Hierarchical Memory [SIMULATED]"
description: "In-memory queue and state-coordination model for a representative NVIDIA RTX 5090 target; no physical NVMe P2PDMA or inference performance qualification."
---

# NVIDIA RTX 5090 BaM-style Storage & Hierarchical Memory

[← Claims index](../../CLAIMS.md)

> **PROVENANCE: IN-MEMORY MODEL ([SIMULATED]), NOT A HARDWARE WITNESS**
>
> Historical figures such as 318.8 tok/s decode, 0.42s 32K restoration, and 7.1 GB/s P2PDMA were illustrative constants, not measured results or calibrated hardware predictions. This includes the earlier CPU-staged/reference values and their speedup ratios.
>
> Pure Go tests exercise queue/descriptor arithmetic, memory-tier policy, and synthetic state serialization. `StagingCopyCount() == 0` is a model/interface assertion, not a measurement of physical host copies. The CLI has no physical transfer or Qwen inference backend; installing Blackwell hardware or a driver does not change that.
>
> Discipline reference: [simulated-results discipline](../standards/simulated-results-discipline.md).

- [SIMULATED] **BaM-style queue and hierarchical-memory model (Issue #11326).** `CUDABaMVRAMQueue` and the Qwen coordinator model submission/completion descriptors, tier bookkeeping, expert slots, and KV/GDN state coordination using a representative RTX 5090 FE / Ryzen 9 5950X / NVMe SSD profile. The simulation does not establish VRAM queue residency, pinned/UVA host memory, MMIO, `cuStreamWaitValue64` execution, physical zero-copy DMA, or inference performance.

- [SIMULATED] **Explicit execution boundary.** `bench`, `qwen38`, and `qwen38-bench` require `--simulate`; `qwen38-sim` is the shorthand. Attempt the model with `go run ./cmd/fak-dev cuda-gpudirect qwen38 --simulate --json`. Successful output uses `fak.modelengine.qwen38-cudadirect-swap/1`, MODELED provenance/verdict, and `is_physical_silicon: false`. Legacy numeric performance fields are zero/unmeasured, not measured zeros. Errors return nonzero without a success receipt. Hardware `inspect` and `audit` do not silently inject fixtures and leave unprobed capabilities UNKNOWN/UNVERIFIED.

Interface-test sources: `internal/compute/cuda_gpudirect_storage_test.go`, `internal/model/qwen38_cudadirect_swap_test.go`. Evidence limits and physical promotion requirements: [NVIDIA GPUDirect model record](../benchmarks/QWEN38-NVIDIA-5090-GPUDIRECT-RESULTS.md). An implemented physical backend plus independently reviewed on-device transfers, model correctness, and end-to-end measurements are required before a scoped hardware-performance claim.
