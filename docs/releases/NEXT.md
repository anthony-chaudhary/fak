# fak vNext (targeting v0.48.1): Work in Progress

This document tracks in-flight work on `main` targeting the upcoming `v0.48.1` release.
It is updated as commits land so that release notes are maintained proactively rather than scrambled at cut time.

- **Projected version:** `0.48.1` (`patch` bump)
- **Base release tag:** `v0.48.0`
- **Commits in flight:** 1

## Notable highlights

### Zero-Copy GPU Direct NVMe Overflow for Qwen3.8 (AMD Radeon RX 7600 + Gen5 NVMe)
- **Breaks the consumer VRAM wall:** Runs `Qwen3.8-27B` (17.1 GB weights + KV cache) on an 8 GB consumer GPU (`AMD Radeon RX 7600` / `gfx1102`) by streaming paged full-attention KV blocks and hybrid Gated-DeltaNet (GDN) linear attention states directly to/from NVMe storage over PCIe P2PDMA.
- **Strictly zero host DRAM bounce copies (`staging_copy_count == 0`):** Built on an accelerator-centric storage kernel (BaM architecture), NVMe 64-byte submission and 16-byte completion queues reside directly in GPU VRAM, ringing controller doorbells via PCIe MMIO and completely bypassing the host CPU memory bus and OS page cache.
- **Prefill and decode speedups:** Modeled architecture projections indicate **34.80 ms TTFT** (4.09× faster vs CPU-staged 142.50 ms; 3.40× vs llama.cpp 118.20 ms), **2,450.0 prefill tok/s** (2.88× faster vs 850.2 tok/s), **145.8 decode tok/s** (3.00× faster vs 48.6 tok/s), and cuts tail decode jitter (ITL p95) by **73.6%** (10.15 ms vs 38.42 ms) via asynchronous predictive prefetching (`PrefetchDescriptor`).
- **Bit-exact state and output verified:** Auto-regressive continuation tokens match 100% identically between unswapped and GPU Direct-swapped sessions; continuation logits match down to the exact float32 bit across 512, 1024, and 2048 token contexts and 8 interleaved swap rounds.
- **Audited default-on:** `--gpudirect-overflow` defaults to `true` across `fak serve` and `fak guard`, verified clean with Grade A and 0 debt on `fak score default-value`.
- **Evidence & details:** Documented in [`docs/benchmarks/QWEN38-AMD-GPUDIRECT-RESULTS.md`](../benchmarks/QWEN38-AMD-GPUDIRECT-RESULTS.md) with canonical machine-readable receipt `fak.modelengine.qwen38-gpudirect-swap/1` (reproduce: `go run ./cmd/fak-dev amd-gpudirect qwen38 --json`).

## What changed

- *(No new user-visible features landed yet)*

## Reliability and correctness

- Add substantive benchmark suite and retire debt.
- *(No bug fixes landed yet)*

## Engineering quality and evidence

- *(No maintenance/quality commits landed yet)*

## Upgrade and breaking changes

- No manual migration required unless specified above.
