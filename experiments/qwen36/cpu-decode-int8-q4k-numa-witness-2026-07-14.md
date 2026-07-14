# Qwen3.6-27B Q4_K_M — int8 decode correctness + NUMA worker-collapse witness (2026-07-14)

Real-weights witness on the AVX2 **CPU server** (EPYC-7742, dual-socket, **8 NUMA nodes**,
256 threads, ~1 TB RAM, no GPU) for epic #4623 (Qwen3.6-27B CPU decode → 10 tok/s).
Run via `cmd/q4kdiag -decode` (added in this campaign, `(fak cmd)`), the real 16.5 GB GGUF
resident on local NVMe, built with go1.26.5 on the box. Model checkout HEAD `df8499eb`.

All numbers are **OBSERVED** (single-stream, greedy, 22-token "Say OK." oracle prefill,
40 timed decode steps after 2 warmup). Bytes/token for int8 Q4_K = 0.5625 B/param × 27e9 ≈
**15.2 GB/tok**.

## 1. int8 Q4_K decode is correct on real weights (C1 #4624)

First-token top-8 logits, f32 Q4_K reference (`FAK_KQ_INT8=0`) vs int8 Q4_K
(`FAK_KQ_INT8=1`), both `FAK_Q4K=1`:

| rank | id | f32 logit | int8 logit |
|---|---|---|---|
| 1 | **248068** (`<think>`, the llama.cpp oracle) | 28.2392 | 28.2415 |
| 2 | 3793 | 23.3626 | 23.2394 |
| 3 | 31248 | 17.9527 | 17.8851 |
| 4 | 11245 | 16.7981 | 16.7408 |
| 5 | 547 | 14.7255 | 14.7525 |
| 6 | 248069 | 14.4601 | 14.4229 |
| 7 | 10092 | 14.2613 | 14.2311 |
| 8 | 220 | 13.9766 | 13.9140 |

The int8 Q4_K reducer reproduces the f32 argmax **and the full top-8 rank order**; logits
agree to within ~0.01–0.12. **The #4624 real-weights blocker (first-token id 248068 agreement)
is witnessed.** Every decode cell below also kept `first_token_id=248068` at every worker
count and placement — so correctness is invariant to the parallelization knobs.

## 2. The Q4_K decode path collapses at high worker counts (C2 #4625)

`q4kMatRowsInto` runs the decode GEMV at `numWorkers = GOMAXPROCS` (256 here) — **uncapped**,
unlike the Q8 path (`q8DecodeWorkersFor` caps amd64 to workers/8 ≤ 16). Sweep of decode tok/s
vs `FAK_WORKERS` under `numactl --interleave=all`, int8 Q4_K:

| placement | FAK_WORKERS | decode tok/s | wall (40 tok) |
|---|---|---|---|
| default (weights on node 0) | 256 (default) | 0.593 | 67.4 s |
| interleave | 256 (default) | 0.629 | 63.6 s |
| interleave | 16 | 0.566 | 70.7 s |
| interleave | 32 | 0.971 | 41.2 s |
| interleave | **64** | **1.395** | 28.7 s |
| interleave | 128 | 1.392 | 28.7 s |
| node0 local (`--cpunodebind=0 --membind=0`) | 32 (of node-0's 32 threads) | 0.829 | 48.3 s |

### Findings
- **The uncapped 256-worker default (0.593) is the worst config.** Capping to **64 workers is
  2.35× faster (1.395)**. The `parFor` shared-cursor busy-wait barrier collapses across 8 NUMA
  nodes at high worker counts — witnessed, exactly as the #4625 hypothesis predicted.
- **The knee is ~64 workers** (= 8 workers/node × 8 nodes); 128 plateaus, 256 regresses. Note
  the Q8 cap of 16 is *too aggressive* for int8 Q4_K (16 → 0.566): the int8 path streams half
  the bytes, so it tolerates ~4× the workers before the barrier dominates. A Q4_K decode cap
  wants ≈ workers/4 (≤ 64), not the Q8 workers/8 (≤ 16).
- **Placement is secondary to the barrier at these worker counts:** interleave vs default at
  256 workers was only +6% (0.629 vs 0.593). But single-node-local (node0, 0.829 at 32
  threads) shows each node delivers ~12.6 GB/s effective at 15.2 GB/tok — well under a node's
  ~40 GB/s channel peak, i.e. **not yet bandwidth-saturated per node**. Spreading 64 workers
  across all 8 nodes (1.395) beats cramming 32 into one node (0.829) because aggregate
  cross-node compute helps even with interleaved (mostly-remote) memory.

## 3. Immediate lever + the path to 10 tok/s

- **Immediate (zero code):** `FAK_WORKERS=64` on this box → **1.395 tok/s**, 2.35× over the
  256-worker default and 1.6× over the 0.868 lean-Q8 baseline. This is the operator lever today.
- **Next code lever (#4625):** interleave leaves 7/8 of weight reads remote. **Per-NUMA-node
  weight replication** (8 × 15.2 GB = 122 GB, trivial in ~945 GiB free) makes every read
  node-local, and a **barrier-free per-node schedule** removes the cross-socket cursor. The
  node0 result (0.829 at one node, not bandwidth-bound) implies ~8 node-local streams could
  aggregate toward the ~10 tok/s target once the barrier is gone. That is the unbuilt multiplier.
- **Also justified now:** the Q4_K decode path should cap workers on manycore amd64 (it is
  currently uncapped at 256 → the witnessed 0.593 collapse), at a Q4_K-appropriate ~workers/4
  (≤ 64), overridable by `FAK_WORKERS`.

## Provenance
- Host: AVX2 EPYC-7742 CPU server, dual-socket, 8 NUMA nodes (128 GB/node), 256 threads, ~1 TB RAM.
- Model: `Qwen3.6-27B-Q4_K_M.gguf` (16.5 GB, resident on local NVMe), fak checkout HEAD `df8499eb`, go1.26.5.
- Tool: `cmd/q4kdiag -decode 40 -warmup 2`; `FAK_Q4K=1`, `FAK_KQ_INT8∈{0,1}`, `FAK_WORKERS` swept, `numactl` placement.
- Method: greedy continuation of the 22-token oracle; `decode_tok_s = 40 / timed_wall`; first-token argmax = the correctness check.
