---
title: CPU inference scaling on box-873af63b — model-size, core-count, context-length, decode-depth
description: >
  Overnight fak-native pure-Go CPU forward characterization on a GPU-less 32-core / 253 GiB box.
  Five scaling axes measured with modelbench: decode/prefill tok/s vs model size (1.5B→27B),
  vs core count (2→32 workers), vs context length (256→2048), decode stability vs KV depth, and
  the decode sweet-spot vs model size. Plus one witnessed bug: gemma4 Q4_K resident forward panics,
  and one methodology finding: sequential multi-model benchmarking contends and under-measures.
  All numbers OBSERVED from single-experiment artifacts.
date: 2026-07-11
---

# CPU inference scaling — box-873af63b (32c / 253 GiB, gpu=none)

**Box:** box-873af63b, 32 logical cores, ~253 GiB RAM (~206 GiB free), no GPU.
**Engine:** fak-native pure-Go CPU forward via `modelbench` (no server, backend `legacy cpu-ref`).
**Models:** local GGUFs — Qwen2.5 {1.5B,3B,7B} Q8_0, Qwen3.6-27B Q4_K_M, gemma4-coding Q4_K_M.
**Honesty:** every tok/s below is read straight from a modelbench artifact field
(`decode.tok_per_sec`, `prefill[].tok_per_sec`). A model that did not decode has no number.

## TL;DR
- **Decode is memory-bandwidth-bound, not compute-bound.** Across model size (clean, 24 workers)
  it falls 16.4 → 8.9 → 4.0 → 1.05 tok/s for 1.5B → 3B → 7B → 27B — inversely with active
  bytes/token. Across core count on the 7B it moves only 2.63 → 3.96 tok/s from 2 → 24 workers
  (12× the cores, 1.5× the throughput) and *regresses* to 3.54 at 32c: once a single stream
  saturates the memory controller, more cores only add contention.
- **Prefill is compute-bound and scales with cores.** 7B prefill(256) climbs 17.6 (2c) →
  115 (24c) tok/s. [context-length + full core curve below]
- **A measurement caveat that became a finding:** the first pass benchmarked all five models
  *sequentially in one session*, which contended for cores and bandwidth and depressed absolute
  throughput — 2.04× for the 1.5B decode, 8.3× for the 27B prefill. Every number in this study is
  from a **single-experiment** re-run on an otherwise-idle box; the contention factors are in §A.
- **One bug found:** gemma4-coding Q4_K panics in the resident CPU forward — the generic
  tensor-parallel qk-norm band uses the scalar `HeadDim`, but gemma4 has per-layer head_dim and
  the `-q4k` path never dispatches to its dedicated `gemma4.go` forward. Filed as an issue.

## A. Decode & prefill vs model size (24 workers, single-experiment)
Authoritative numbers: each model measured **alone on an idle box** at 24 workers (the clean
run from the §B/§E/§G core-scaling sweeps). The original concurrent grid under-measured these —
see the contention correction below.

| Model | Precision | decode tok/s @24c | prefill(256) tok/s @24c |
|---|---|---:|---:|
| Qwen2.5-1.5B | Q8_0 | **16.43** | 421 |
| Qwen2.5-3B | Q8_0 | **8.89** | 196 |
| Qwen2.5-7B | Q8_0 | **3.96** | 115 |
| Qwen3.6-27B | Q4_K_M | **1.05** | 23 |
| gemma4-coding | Q4_K_M | **PANIC** | — (see §F) |

Decode scales inversely with active bytes/token, as expected for a bandwidth-bound single stream.
(The 27B is Q4_K, ~½ the bytes/token of a Q8 of equal params, which is why it is ~3.8× slower than
the 7B-Q8 rather than the ~8× a naive param-count would predict.)

**Contention correction — a methodology finding.** The first pass benchmarked all five models
*sequentially in one session*; that stole cores and memory bandwidth and depressed every number,
**unevenly**. Same `workers=24`, same 16-tok prompt — only the box's idleness differed:

| Model | decode: concurrent → clean | prefill(256): concurrent → clean |
|---|---:|---:|
| 1.5B | 8.05 → 16.43 (**2.04×**) | 317 → 421 (1.3×) |
| 3B | 6.31 → 8.89 (1.41×) | 186 → 196 (1.1×) |
| 7B | 3.65 → 3.96 (1.09×) | 103 → 115 (1.1×) |
| 27B | 0.82 → 1.05 (1.28×) | 2.8 → 23.1 (**8.3×**) |

The gradient is itself informative: **decode** contention is worst for the *smallest* model (most
bandwidth-hungry per unit time), while **prefill** contention is catastrophic for the *largest*
model (most core-hungry, and near the memory-pressure edge). Lesson baked into the rest of this
study: measure one model at a time. (This is why the box was kept single-experiment for §B–§G.)

## B. Decode & prefill vs core count — Qwen2.5-7B-Q8 (FAK_WORKERS pinned)
Prefill(256)×3, decode 64×3. Worker count pinned exactly via `FAK_WORKERS`.

| workers | prefill(256) tok/s | prefill speedup vs 2c | decode tok/s | decode speedup vs 2c |
|---:|---:|---:|---:|---:|
| 2 | 17.6 | 1.00× | 2.63 | 1.00× |
| 4 | 28.6 | 1.62× | 3.51 | 1.33× |
| 8 | 52.0 | 2.95× | 3.56 | 1.35× |
| 16 | 88.9 | 5.04× | 3.94 | 1.50× |
| 24 | **115.3** | 6.53× | **3.96** | 1.50× |
| 32 | 116.3 | 6.60× | 3.54 | 1.34× |

Two distinct regimes, both bottlenecked by memory bandwidth before they run out of cores:
- **Prefill (compute-heavy GEMM) scales but saturates at ~24 workers.** 6.53× at 24c, and 24→32
  buys essentially nothing (115.3 → 116.3, +0.9%). Parallel efficiency at 24c is only 6.53/12 ≈
  54% — the shared memory controller, not core count, caps it past ~16 cores.
- **Decode (GEMV, one row of weights per token) plateaus almost immediately** and then *regresses*:
  it peaks at 24c (3.96 tok/s) and **drops to 3.54 at 32c** — oversubscribing all logical cores
  starves the memory controller and leaves no headroom, so full-core decode is *slower* than 24c.

**Actionable:** on this box, 24 workers is the sweet spot for the 7B; `FAK_WORKERS=32` (all cores)
is a pessimization for decode and a wash for prefill. Cross-check with §A (budget-0.75 ≈ 24c:
prefill 102.8, decode 3.64) reproduces the 24c point within run-to-run variance (~8%).

## C. Prefill vs context length — Qwen2.5-7B-Q8 @32 workers
Prefill P∈{256,512,1024,2048}×3 reps.

| context | prefill tok/s | median ms | vs P=256 |
|---:|---:|---:|---:|
| 256 | 119.4 | 2,143 | 1.00× |
| 512 | 108.3 | 4,727 | 0.91× |
| 1024 | 91.0 | 11,248 | 0.76× |
| 2048 | 74.7 | 27,431 | 0.63× |

Prefill throughput degrades **sub-linearly** with context (0.63× at 8× the context), the signature
of attention's O(n²) term growing against a still-dominant O(n) FFN/GEMM cost — at 2048 tokens on
a 7B the quadratic attention term is measurable but not yet dominant. Wall-clock per prefill still
grows super-linearly (2.1 s → 27.4 s for 8× tokens ≈ 12.8× time) as expected.

## D. Decode stability vs KV depth — Qwen2.5-7B-Q8 @32 workers
Decode 256 steps×3 (vs the 64-step §A/§B).

| decode steps | tok/s | per-token median ms |
|---:|---:|---:|
| 64 (§B, 32c) | 3.54 | 282.5 |
| 256 (§D, 32c) | 3.49 | 286.8 |

**Decode is depth-stable.** Generating 4× more tokens changes per-token throughput by −1.4%
(3.54 → 3.49). The KV cache after 256 tokens is negligible against the ~8 GiB of Q8 weights
streamed per token, so decode stays weight-bandwidth-bound and flat — no degradation cliff in
this range. (Confirms decode cost is dominated by the per-token weight sweep, not attention.)

## E. Does the decode sweet-spot shift with model size? — 1.5B/3B/7B/27B core-scaling
The §B finding (7B decode peaks at 24 workers, regresses at 32) raised the question: is 24 a
universal sweet-spot, or does it move with per-token bandwidth pressure? Same sweep (workers
2/8/16/24/32, prefill 256, decode 48–64 steps) on the 1.5B, 3B and the 27B, folded against §B's 7B.
All single-experiment (see the §A contention correction for why that matters).

**Decode tok/s vs workers (peak in bold):**

| model | 2c | 8c | 16c | 24c | 32c | peak | 32c vs peak |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1.5B-Q8 | 12.79 | 16.13 | **16.50** | 16.43 | 9.61 | 16.50 @16c | −42% |
| 3B-Q8 | 6.29 | 7.75 | **8.89** | **8.89** | 5.91 | 8.89 @16–24c | −34% |
| 7B-Q8 | 2.63 | 3.56 | 3.94 | **3.96** | 3.54 | 3.96 @24c | −11% |
| 27B-Q4K | – | – | **1.17** | 1.05 | 0.97 | 1.17 @16c | −17% |

The sweet-spot is **U-shaped in model size, not monotonic** — both the *lightest* (1.5B) and the
*heaviest* (27B) peak at **16** workers, while the *middle* pair (3B, 7B) reach **24**. Two
different mechanisms meet at 16 cores:
- The **1.5B** does so little work per token that thread-coordination overhead dominates past 16c.
- The **27B** streams the most bytes/token (Q4_K, ~13 GiB/token) and saturates the memory
  controller with the fewest cores — extra cores find no bandwidth left to claim.
- The **3B/7B** sit between: enough per-token compute to use 24 cores, not enough bandwidth
  pressure to saturate before then.

And **all four regress at 32c** — full-core oversubscription is a universal decode pessimization,
worst for the 1.5B (−42%), mildest for the 7B (−11%).

**Prefill tok/s vs workers** (compute-bound, for contrast):

| model | 2c | 8c | 16c | 24c | 32c |
|---:|---:|---:|---:|---:|---:|
| 1.5B-Q8 | 69.2 | 191.0 | 332.3 | **421.5** | 336.7 |
| 3B-Q8 | 33.8 | 97.3 | 156.3 | **195.6** | 176.5 |
| 7B-Q8 | 17.6 | 52.0 | 88.9 | 115.3 | **116.3** |
| 27B-Q4K | – | – | 21.9 | 23.1 | **31.7** |

Prefill runs the *opposite* way to decode: the small models (1.5B/3B) peak at 24c and regress at
32, while the **larger** models (7B, 27B) — with GEMMs big enough to amortize thread overhead —
keep rising to 32c (the 27B gains +37% from 16→32c). **Takeaway: `FAK_WORKERS≈24` is the robust
default. For decode, both extremes (≤2B and the 27B) do best at 16 and all-32 regresses every
model; for prefill, larger models keep scaling and still gain at 32c. Match workers to the
workload — decode and prefill want opposite ends for the same big model.**

## F. Bug: gemma4-coding Q4_K resident CPU forward panics
Preflight READY (arch `gemma4`, 667 tensors, ~6.86 GiB); panics at first Prefill:
`panic: model: qk-norm weight length does not match head_dim or projection width`
(`arch.go:226` ← `tensor_parallel_forward.go:237 tpApplyQKNormBand`). Root cause: the generic TP
qk-norm band uses the scalar `cfg.HeadDim`, but gemma4 has per-layer head_dim (`headDimForLayer`),
and the `-q4k` resident path (`tokenHiddenQ`→`blockStep`) never dispatches to the dedicated
`gemma4.go` forward. Runtime-confirmed **not** TP-specific: `FAK_WORKERS=1` panics identically
(single band = full NumHeads=16/NumKVHeads=8), so gemma4 has no working resident-quant CPU path at
any worker count. Filed as **#4274** (bug/model-arch) with the full trace and two fix options
(fail-closed typed refusal vs. real per-layer-head_dim dispatch through `gemma4.go`).

## Reproduce
```
# box-873af63b, GPU-less. modelbench = go build ./cmd/modelbench (run in BACKGROUND on Windows —
# a foreground timeout leaves an unkillable session-0 orphan holding ~22 GiB).
GGUF=~/.cache/fak-models/gguf
# A: model-size grid (24 workers)
for g in Qwen2.5-1.5B-Instruct.Q8_0 Qwen2.5-3B-Instruct-Q8_0 Qwen2.5-7B-Instruct-Q8_0; do
  modelbench -gguf $GGUF/$g.gguf -lean -budget 0.75 \
    -prefill-sizes 16,64,256 -prefill-reps 3 -decode-prompt 16 -decode-steps 64 -decode-reps 3 \
    -out experiments/nightrun/box-873af63b/modelbench-$g-cpu.json
done
# 27B uses -q4k instead of -lean. B: sweep FAK_WORKERS∈{2,4,8,16,24,32} on the 7B.
```
Ledger rows: `docs/nightrun/collected.jsonl` (`task_id: cpu-decode-scaling-*`, 2026-07-11).
Artifacts are box-local (`experiments/nightrun/*/` is gitignored) — the ledger + this note are the
durable trunk record.
