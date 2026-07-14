# Qwen3.6-27B CPU decode — NUMA placement/worker sweep + bandwidth roofline (2026-07-14)

Campaign: epic #4623. Real-weights witness on the target CPU server for **C2 #4625**
(NUMA-aware decode parallelism) and **C3 #4626** (bandwidth roofline). Correctness
(first-token 248068) held on every cell — see the companion first-token witness
(`int8-q4k-realweights-firsttoken-witness-20260714.md`).

**Host:** CPU server — 2× AMD EPYC 7742 (256 threads), **8 NUMA nodes** (NPS-style,
~128 GiB/node), ~1 TB RAM, AVX2, no GPU. Go 1.26, resident-Q4_K (`FAK_Q4K=1`).
**Model:** Qwen3.6-27B-Q4_K_M.gguf (16.5 GB). **Tool:** `q4kdiag -decode 40 -warmup 2`
(in-process greedy decode; tok/s = 40 / timed wall).

## Decode tok/s sweep (int8 Q4_K unless noted)

| cell | placement | FAK_WORKERS | decode tok/s | vs default |
|---|---|---|---|---|
| base | default (first-touch) | 256 (all) | **0.565** | 1.00× |
| il-def | `--interleave=all` | 256 (all) | 0.629 | 1.11× |
| il-w16 | `--interleave=all` | 16 | 0.563 | 1.00× |
| il-w32 | `--interleave=all` | 32 | 0.995 | 1.76× |
| **il-w64** | **`--interleave=all`** | **64** | **1.397** | **2.47×** |
| il-w128 | `--interleave=all` | 128 | 1.369 | 2.42× |
| node0-w32 | `--cpunodebind=0 --membind=0` | 32 | 0.829 | 1.47× |
| f32-il32 | `--interleave=all` (int8=0) | 32 | 1.203 | 2.13× |

### What the sweep proves
- **The 256-worker default collapses (0.565 tok/s).** This is the `parFor` shared-cursor
  busy-wait barrier ping-ponging across 8 NUMA nodes — the exact pathology the
  `internal/model/budget.go` manycore-cap comment warns about, on the **Q4_K decode path
  which is *not* covered by that cap** (it runs uncapped at GOMAXPROCS=256). The Q8 cap
  (≤16) is present; the Q4_K path's absence of it is the asymmetry C2 fixes.
- **Zero-code win: `--interleave=all` + `FAK_WORKERS=64` → 1.397 tok/s = 2.47×**, no code
  change. The worker knee is ~64 (w128 regresses slightly; w16 is too few to cover
  bandwidth).
- **Placement dominates.** node0-local (0.829) < interleave-w32 (0.995) < interleave-w64
  (1.397): striping weights across nodes beats confining to one node, and beats the
  default first-touch-on-node-0 confinement badly.
- **int8 is *not* faster than f32 in the full decode** at these worker counts (f32-il32
  1.203 > il-w32 int8 0.995). The "~8× kernel" is a single-GEMV microbench number; in the
  barrier-bound full-decode regime the activation-quant overhead shows and the memory
  barrier dominates. int8's win is bytes/token (roofline headroom), realized only once the
  barrier is removed (C2 code work). Honest correction to the "int8 is strictly faster"
  framing for the end-to-end decode.

## Memory-bandwidth roofline (pure-Go STREAM-Triad, 1 GiB arrays, 8 iters)

| placement | aggregate GB/s | single-thread GB/s |
|---|---|---|
| default (first-touch → node 0) | **18.1** | 14.1 |
| `--interleave=all` | **87.6** | 10.7 |
| `--cpunodebind=0 --membind=0` | 26.9 | 16.4 |

Interleave gives **4.8× the aggregate** of default first-touch — the same placement story
the decode sweep shows. (This Triad is a pure-Go floor, no non-temporal stores; a tuned
STREAM would read higher, but the decode kernel is likewise pure-Go weight-streaming, so
~88 GB/s is a fair proxy for the *achievable decode* bandwidth on this access pattern.)

### decode_bw_util % and the honest ceiling
- Best decode cell: 1.397 tok/s × 15 GB/tok (0.5625 B/param × 27e9) ≈ **21 GB/s** effective
  → **~24 %** of the 87.6 GB/s interleave aggregate. Headroom remains (the barrier + reads
  that are node-far under interleave), which is what per-node **weight replication** (C2)
  buys: 8/8 local reads instead of 1/8.
- **10 tok/s at 15 GB/tok needs 150 GB/s sustained** — above the ~88 GB/s this access
  pattern extracts (⇒ **~5.8 tok/s practical single-stream ceiling** at Q4_K on this box).
  So **10 tok/s is not reachable by NUMA placement alone**; it requires **C5 #4628**
  (sub-4-bit MLP → ~10 GB/tok lifts the ceiling toward ~9 tok/s) and/or **C6 / #3197**
  (MTP self-speculative amortization, which multiplies tok/s without more bandwidth).
  C2 (per-node replication + barrier-free schedule) is the multiplier that turns today's
  ~24 % utilization into the mid-single-digits; C5+C6 carry it to 10.

## Immediate, shippable takeaway
For anyone serving Qwen3.6-27B Q4_K_M on a many-NUMA EPYC box **today**, launch with
`numactl --interleave=all` and `FAK_WORKERS≈64` (not the 256-thread default) for **~2.5×**
decode — pending the C2 code redesign that generalizes this without the manual knob.

## Reproduce
`experiments/qwen36/witness-cpu-decode.sh <gguf>` runs the placement×worker sweep and the
Triad roofline and writes a dated manifest. (These numbers were taken via the same
`q4kdiag -decode` path the runner drives.)
