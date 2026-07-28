---
title: "GLM-5.2 status by machine size — the one organized rollup (2026-07-18)"
description: "A single organized view of where GLM-5.2 UD-Q4_K_M actually runs and how fast, laid out by machine size instead of by ticket: small boxes (does not fit), the 40 GiB/card node (expert-offload only, ~2.2 tok/s), the 80 GiB/card node (resident, ~23 tok/s), and the CPU-only large-RAM host (~0.24 tok/s). Reconciles the scattered per-box notes and folds the freshest 2026-07-18 lab-bridge numbers, keeping the engine-honesty fence between the llama.cpp baseline and fak's own pre-performance kernel."
---

# GLM-5.2 status by machine size

> **What this is.** The single "where does GLM-5.2 run, and how fast" rollup, organized by
> **machine size** rather than by ticket. It reconciles the per-box baseline notes
> (2026-07-01/06) and folds the freshest lab-bridge measurements (2026-07-18) into one table
> so the size→throughput story is legible at a glance.
>
> **What this is NOT.** No new claim. Every measured cell traces to an owning note or a
> committed artifact and carries its label + date. The **engine-honesty fence** is kept
> throughout: the fast numbers are the **llama.cpp baseline engine**, not fak's own kernel —
> fak's native GLM-5.2 path is functionally witnessed but **pre-performance** (§4).

## TL;DR — the number the question usually means

**"~2.2 tok/s with expert offloading on the big datacenter node" is real and current: it is the
40 GiB/card 8-GPU node running GLM-5.2 with its experts offloaded to host RAM** — **2.24 tok/s decode**
(llama.cpp `-ngl 99 -ot exps=CPU`, measured 2026-07-18). That is the *offload* regime, and it
is the honest ceiling on a node too small to hold the model in VRAM.

The thing the reporting most needs to make legible: **there are two very different datacenter-node
numbers, and card size — not GPU count — decides which one you get.**

| Node | Fit | Best current GLM-5.2 decode | Engine | Regime |
|---|---|--:|---|---|
| **40 GiB/card** (320 GiB VRAM) | model does **not** fit → experts on host | **2.24 tok/s** | llama.cpp | host-bandwidth bound (offload) |
| **80 GiB/card** (640 GiB VRAM) | model **fits resident** (~206 GiB free) | **23.2 tok/s** | llama.cpp | GPU memory-bandwidth bound |

Same 8-GPU chassis, **~10× apart**, because on 40 GiB cards the 433.82 GiB model cannot be
held resident and the ~13 GiB/token routed-expert stream is read from host RAM instead of HBM.

## The model (fixes the fit law for every row)

| Input | Value | Provenance |
|---|--:|---|
| Total params | **753.86 B** | WITNESSED (llama-bench, 2026-06-28) |
| On-disk / resident (UD-Q4_K_M) | **433.82 GiB** (11 shards, ~466 GB) | WITNESSED |
| Architecture | `glm_moe_dsa` — MoE + DeepSeek-Sparse-Attention; 79 layers (3 dense + 76 MoE), 256 experts, K=8/tok, 1 shared | WITNESSED GGUF header (#3074, 2026-07-13) |
| Active params / token | **~30–33 B** | DERIVED from measured K=8 |
| Serving-arch floor | DSA kernels in stock SGLang/vLLM hard-depend on **sm_90+**; the lab's sm_80 nodes get `BLOCKED_ARCH` → **llama.cpp MLA is the resident path** | vLLM #35021; `tools/glm52_serve_preflight.py` |

## The size ladder — one row per machine class

| # | Machine class | GLM-5.2 fit | Path | Decode tok/s | Prefill | Engine | Label · date |
|--:|---|---|---|--:|--:|---|---|
| 1 | **Small box** — Mac M3 Pro 36 GB · desktops ≤256 GB RAM / ≤24 GB GPU | **Does not fit** (434 GiB ≫ box) | — | — | — | — | not a serving target; carries the small-model ladder + fak tiny-reference `glm_moe_dsa` correctness witness (f32, small scale) |
| 2 | **CPU-only large-RAM host** — 8× no-GPU-used, model fully in ~973 GB RAM | Fits in **RAM**, no GPU | CPU-only (`-ngl 0`, 256 threads) | **0.243** | 0.985 | llama.cpp | OBSERVED · lab bridge 2026-07-18 |
| 3a | **40 GiB/card node** — 8-GPU, 320 GiB VRAM, large host RAM | **Not resident** (320 < 434) → experts on host | llama.cpp GPU + expert-offload (`-ngl 99 -ot exps=CPU`) | **2.24** | 3.72 | llama.cpp | OBSERVED · lab bridge 2026-07-18 |
| 3b | 40 GiB/card node (same box, fak kernel) | same | fak kernel cpu-offload | **0.2324** | — | **fak** | WITNESSED · 2026-06-28 |
| 4a | **80 GiB/card node** — 8-GPU, 640 GiB VRAM | **Resident** (~206 GiB free) | llama.cpp resident MLA (sm_80) | **23.2** | — | llama.cpp | WITNESSED · 2026-07-01 (two runs 23.23/23.22) |
| 4b | 80 GiB/card node (same box, fak kernel) | same | pure-fak CUDA+NCCL EP-8 resident | **0.09–0.21** | — | **fak** | WITNESSED functional, **pre-performance** · 2026-07-15 (#4777) |
| 5 | **CPU-only large-RAM host, contended** | Q4/Q3 exceed *available* RAM under other users | NVMe thrash | ~0.3 (stalled) | — | llama.cpp | NOT viable right now · 2026-07-18 |

Notes on the rows:

- **Row 2 vs 3a — the GPU is worth ~9×.** On the same class of large-RAM host, adding 8 GPUs to
  carry the dense/attention layers takes decode from **0.243 → 2.24 tok/s** (9.2×) even though
  ~400 GB of experts stay on the CPU. Model load also drops ~250 s → 40 s. The experts are the
  remaining wall.
- **Row 3a is host-bound, not GPU-count-bound.** Under expert-offload the ~13 GiB/token routed
  stream (K=8 × 1.619 GiB) is read from host RAM and the expert GEMM runs on the CPU, so **adding
  concurrency makes it *worse* (0.27× at 2 streams)** and the 7 idle GPUs cannot be batched in as
  configured. Pushing more experts onto the ~290 GiB of free VRAM was tried (`--n-cpu-moe 20/32/44`)
  and **all OOMed** — the default tensor-split is uneven for these large experts. Clean best config
  stays all-experts-on-host at 2.24 tok/s.
- **Row 4a is the fast lane** and ~9–10× the offload path — the payoff for 80 GiB cards. Its
  roofline ceiling is **~150–200 tok/s** single-stream (80% target 120–160); the dominant unpulled
  lever is killing llama.cpp's `-sm layer` (one GPU per token, 7 idle) for a row/tensor split
  (L1, #3075). Cold first turn carries a one-time ~500 s backend warmup (not a KV miss).
- **Row 4b is the strategic goal, honestly pre-performance.** fak's own CUDA+NCCL expert-parallel
  path loads the exact 11-shard checkpoint across 8×80 GiB, all ranks join, and generates coherent
  content — but the uncached decode baseline is **0.091662 tok/s** (#4777). The resident CUDA
  expert path (#4843) that must beat this had not landed at measurement time.

## The engine-honesty fence

The fast, quotable numbers (**23.2** resident, **2.24** offload) are the **llama.cpp baseline
engine** — fak *fronts / adjudicates* it but does not compute those tokens. fak's **own** kernel
numbers are Row 3b (0.2324, offload) and Row 4b (0.09–0.21, resident functional-only). The
strategic goal (epic #1010: GLM-5.2 in fak's pure-Go kernel, performant) is **on-track and
box-gated, not yet a result** — see [`docs/EXECUTIVE-ROLLUP.md`](../EXECUTIVE-ROLLUP.md) §"Live
strategic goal". Do not lead with a fak GLM-5.2 throughput number until the resident CUDA expert
path (#4843) lands and is re-measured against the #4777 baseline under the same checkpoint + topology.

## What moves each class next

- **40 GiB/card (Row 3a):** move the expert GEMM off the host — a PCIe-resident *partial* hot-expert
  set and/or an int8 host expert GEMM (VNNI reducers already show 8–18× on the host path); or fit a
  smaller quant (≤~2.9 bits/weight) resident in 320 GiB to earn a real per-token GPU roofline. Lanes
  G-offload / G-fit; ties #2336.
- **80 GiB/card (Row 4a, llama.cpp):** L1 tensor/row split (#3075, ~3–6×) → continuous batching
  (#3079/#3080, 10–40× aggregate) → FA + CUDA graphs (#3076) → spec-decode (#3078). Drives 23 → ~120–160.
- **80 GiB/card (Row 4b, fak):** land the resident CUDA expert path (#4843) and re-measure; this is
  the pure-fak witness that closes the strategic goal, separate from the llama.cpp baseline.
- **CPU-only host (Row 2 / 5):** the offload regime is the honest home for the SSD/host-memory tier;
  a clean run needs an uncontended large-RAM host (Row 5 stalls under other users).

## Provenance — the owning notes this rollup folds

- Per-box baseline + next steps (40 vs 80 GiB/card): [`GLM52-DGX-PER-BOX-BASELINE-AND-NEXT-STEPS-2026-07-06.md`](GLM52-DGX-PER-BOX-BASELINE-AND-NEXT-STEPS-2026-07-06.md)
- Roofline ceiling + lever map: [`GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md`](GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md)
- 80 GiB/card resident serve (WITNESSED 23.2): [`GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md`](GLM52-8GPU-FULL-RESIDENT-SERVE-2026-07-01.md)
- Pure-fak 8-GPU resident functional witness (#4777): [`GLM52-PURE-FAK-GPU-SERVER-RESULTS-2026-07-15.md`](GLM52-PURE-FAK-GPU-SERVER-RESULTS-2026-07-15.md)
- Roofline dashboard (current-vs-ceiling, generated): [`GLM52-DGX-ROOFLINE-DASHBOARD.md`](GLM52-DGX-ROOFLINE-DASHBOARD.md)
- Hardware coverage matrix (all platforms): [`../HARDWARE-MATRIX.md`](../HARDWARE-MATRIX.md)
- Full-size serving witness runbook (#413): [`../serving/glm52-full-size-serving-witness.md`](../serving/glm52-full-size-serving-witness.md)

> **Freshness fence.** The 2026-07-18 lab-bridge rows (2, 3a, 5) are OBSERVED measurements from a
> live `llama-server` `/completion` run (`n_predict:128, temperature:0`, reading the `timings`
> block), **not yet committed as roofline artifacts** under `experiments/benchmark/runs/` — so the
> generated roofline dashboard still shows the 40 GiB/card offload lane (G-offload) as PENDING until
> a real artifact lands. They are recorded here as the current honest reading, labeled as such.
</content>
</invoke>
