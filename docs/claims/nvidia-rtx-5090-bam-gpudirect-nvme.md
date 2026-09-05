---
title: "NVIDIA RTX 5090 BaM GPU Direct NVMe & Hierarchical Memory [SIMULATED]"
description: "Zero-copy NVMe P2PDMA and BaM-architecture storage overflow engine for Qwen3.8-27B and Flash Next on NVIDIA RTX 5090 FE."
---

# NVIDIA RTX 5090 BaM GPU Direct NVMe & Hierarchical Memory

[← Claims index](../../CLAIMS.md)

> ⚠️ **CRITICAL PROVENANCE NOTICE: THEORETICAL ROOFLINE PROJECTION ([SIMULATED])**
>
> All performance figures below (318.8 tok/s decode, 0.42s 32K context restoration, 7.1 GB/s P2PDMA) are **modeled analytical roofline projections ([SIMULATED])**, NOT physical empirical measurements on Blackwell hardware.
>
> **Why we over-explain this simulation:**
> - **Zero physical execution on Blackwell silicon:** Execution on hardware is pending active Linux driver DMA-BUF P2PDMA support (`nvidia-open`).
> - **What is verified:** Pure Go unit tests (`internal/compute/cuda_gpudirect_storage_test.go`, `internal/model/qwen38_cudadirect_swap_test.go`) verify interface invariants (`StagingCopyCount() == 0`, queue descriptor math, bit-exact state serialization), not empirical throughput or bus latency.
> - **Unmodeled physical effects:** PCIe TLP packetization headers, DRAM bank conflicts / auto-refresh, thermal throttling (DVFS), OS/CGO scheduling jitter, and MoE expert routing skew.
> - **Discipline reference:** [`docs/standards/simulated-results-discipline.md`](../standards/simulated-results-discipline.md).

- [SIMULATED] **NVIDIA RTX 5090 BaM GPU Direct NVMe & Hierarchical Memory (Issue #11326).** Zero-copy NVMe P2PDMA and BaM-architecture storage overflow engine for Qwen3.8-27B and Flash Next on NVIDIA Blackwell sm_120 (RTX 5090 FE 32GB GDDR7 + Samsung 990 Pro 2TB NVMe). Mapped direct NVMe queues in BAR1 VRAM (`CUDABaMVRAMQueue`), zero host DRAM bounce copies (`StagingCopyCount() == 0`), Direct CPU Root Complex P2PDMA (<700ns latency), and 3-Tier Hierarchical Memory Manager (Tier 0 VRAM 32GB, Tier 1 Host Pinned DRAM 128GB, Tier 2 NVMe P2PDMA). Modeled projection: 318.8 tok/s decode, 0.42s 32K context restoration, 7.1 GB/s sustained P2PDMA bandwidth, 0 host staging copies, and 51B PLE offloaded to 128GB Host DRAM with dynamic VRAM expert slot-streaming via `cuStreamWaitValue64`. Algorithmic zero-copy invariant and bit-exact hybrid state verified in pure Go tests (`internal/compute/cuda_gpudirect_storage_test.go`, `internal/model/qwen38_cudadirect_swap_test.go`). Witness: `docs/benchmarks/QWEN38-NVIDIA-5090-GPUDIRECT-RESULTS.md` (`fak.modelengine.qwen38-cudadirect-swap/1`; reproduce: `go run ./cmd/fak-dev cuda-gpudirect qwen38 --json`). Physical Blackwell on-device execution required to promote to the SHIPPED tier.
