---
title: "Qwen3.6-27B GGUF->Q8 load profile + page-churn before/after (#440)"
description: "Phase-attributed load profile of the Qwen3.6-27B q4_k_m GGUF->Q8 quant-on-load path, with a same-host arena-reuse before/after on peak RSS and page faults. Turns the issue's single aggregate load=75180ms number into a per-phase breakdown, and witnesses the #440 page-churn fix at -4.0% peak RSS / -4.0% page faults."
date: 2026-06-26
---

# Qwen3.6-27B GGUF->Q8 load profile + page-churn before/after (#440)

Issue [#440](https://github.com/anthony-chaudhary/fak/issues/440) asked for the
GGUF->Q8 *load* path to stop being an opaque tax: a per-phase timing breakdown
instead of a single aggregate `load=75180ms`, the avoidable full-tensor copies
removed, a regression-visible report, and a re-run of the 27B smoke with
before/after numbers. This is the published before/after.

The load path here is `cmd/modelbench -lean -load-only` (the GGUF->Q8 quant-on-load
path, `ggufload.LoadModelQuantProfile`). `-load-only` loads the model, emits the
report, and exits **without running a forward pass** — so the load tax is isolated
from prefill/decode kernel speed (tracked separately, per the issue's own note) and
needs no GPU.

## Reproduce

```sh
go build -o modelbench ./cmd/modelbench
# AFTER (default: the #440 reused dequant arena is on)
./modelbench -gguf ~/.cache/fak-models/gguf/Qwen3.6-27B-Q4_K_M.gguf \
  -lean -load-only -load-profile -out after.json
# BEFORE (escape hatch disables the arena -> the pre-#440 fresh-alloc-per-tensor path)
FAK_GGUF_NO_ARENA_REUSE=1 ./modelbench -gguf ~/.cache/fak-models/gguf/Qwen3.6-27B-Q4_K_M.gguf \
  -lean -load-only -load-profile -out before.json
```

## 1. Phase attribution — where the 27B load time actually goes

The headline ask: replace `load=75180ms` with an attributable breakdown. Measured
AFTER profile (arena on), 851 GGUF tensors, q4_k_m -> f32 -> Q8:

| phase | meaning | ms | % of load |
|---|---|---:|---:|
| `gguf_dequant` | Q4_K/Q6_K/F32 payload -> f32 | 21492 | **47.7%** |
| `quant_builder_add` | f32 -> Q8 quant + non-matmul f32 retention | 13939 | 30.9% |
| `gguf_normalize` | canonical (HF) normalization/reorder | 4482 | 10.0% |
| `gguf_read` | tensor payload read from the shard | 3708 | 8.2% |
| `gguf_open_index` | open + index the GGUF tensor table | 1410 | 3.1% |
| `gguf_map_shape` + `gguf_config` | metadata/shape mapping | ~3 | 0.0% |

**Bottleneck: `gguf_dequant` (47.7%)** — decoding the packed Q4_K/Q6_K blocks to
f32 is the dominant cost, with the f32->Q8 re-quant second. The two heaviest single
tensors are the wide vocab projections: `lm_head.weight` (Q6_K, 2410 ms) and
`model.embed_tokens.weight` (Q4_K, 1928 ms).

This is the actionable finding: the load is **dequant-CPU-bound, not I/O- or
allocation-bound**. The next load-time win is therefore not more allocation
trimming but skipping the f32 round-trip entirely — a resident-Q4_K load path for
`qwen35` like the one already shipped for GLM-5.2 (`ggufload.LoadModelQ4K`,
commits 54d4f0d / 7d049e2), which streams the q4_k_m bytes without dequantizing.
That is out of scope for #440 (a separate path) and is the recommended follow-up.

## 2. Page-churn before/after — the #440 arena reuse

Commit 138774c collapsed the 800+ throwaway `elems*4` f32 dequant buffers (one per
tensor, each faulting in fresh zeroed pages the GC then unmaps) into a single reused
arena grown to the largest tensor. Same host, same binary, same model — arena
**off** (pre-#440) vs **on** (current), mean of repeated runs:

| metric | BEFORE (arena off) | AFTER (arena on) | delta |
|---|---:|---:|---:|
| peak RSS | 58.05 GB | 55.71 GB | **-4.0%** |
| page faults | 14,201,498 | 13,627,603 | **-4.0%** |
| load wall time | ~48-51 s | ~45-52 s | within run-to-run noise |

`page_faults` here is `windows:PageFaultCount` (total hard+soft faults over the
process lifetime — labeled in the report because the metric is **not** the same
across OSes). The peak-RSS and page-fault reductions are tight and repeatable (the
BEFORE peak RSS was byte-identical across runs); the load *wall time* is dominated
by the dequant CPU work above, so the arena does not move it beyond noise. The
correctness of the reuse is gated by
`TestDequantF32IntoReusesArenaAndNeverLeaksStaleData` and the new
`TestQuantLoadArenaToggleProducesIdenticalModel` (arena off vs on yields
bit-identical prefill logits — the A/B is a memory knob, never a numerics knob).

## 3. Cross-host reference — the issue's M3 baseline

The issue's original numbers (Apple M3 Pro / 36 GB, 2026-06-19, pre-#440):
`load=75180ms` (aggregate, no phase breakdown); `-n 1` smoke `208.58 real /
354.21 user / 538.40 sys`, peak RSS `29,059,743,744` B.

**These are not directly comparable to the win32 numbers above** and are included
only as the documented "before this work" reference. Two reasons: (a) different
hardware (M3 Pro vs Ryzen 9 9950X); (b) **different RSS metric** — macOS
`ru_maxrss` reports unified-memory resident set, while Windows `PeakWorkingSetSize`
counts the peak f32 dequant arena + the Q8 resident store + mapped input pages, so
the absolute GB figures measure different things. The honest, apples-to-apples
before/after is the **same-host arena A/B in section 2**; the M3 line establishes
that the per-phase attribution (section 1) did not exist before #440.

## Acceptance status

- [x] Phase timing around `LoadModelQuant`/`QuantModel` — `ggufload.LoadProfiler`
  (phases `gguf_open_index`/`gguf_config`/`gguf_read`/`gguf_dequant`/
  `gguf_normalize`/`quant_builder_add`/`quant_builder_finalize`), gated by
  `TestLoadModelQuantProfileReportsLoadPhases`. Section 1.
- [x] Remove avoidable full-tensor temp allocations/copies — the reused dequant
  arena (138774c) + `dequantF32Into`. Witnessed at -4.0% peak RSS / -4.0% page
  faults. Section 2.
- [x] Report load timing + peak/resident allocation in a modelbench mode —
  `modelbench -load-only` (`load_ms`, `peak_rss_bytes`, `page_faults`,
  `page_fault_metric`, `load_profile`).
- [x] Re-run the Qwen3.6-27B `-n 1` smoke and publish before/after — this doc
  (load wall time, page faults, peak RSS; the `-n 1` forward itself is the
  separately-tracked prefill/decode kernel path, not the load tax #440 scopes).

## Provenance

Host: AMD Ryzen 9 9950X (16C/32T), 256 GiB RAM (4×64 GiB DDR5; ≈253.6 GiB /
272 GB usable after hardware-reserved), Windows 11, `windows/amd64`,
Go 1.26.3, fak v0.34.0. Model: `Qwen3.6-27B-Q4_K_M.gguf` (16.5 GB, 851 tensors).
Measured 2026-06-26 via `modelbench -lean -load-only -load-profile`. Raw reports
from the commands in [Reproduce](#reproduce).
