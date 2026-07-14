# C2 witness — Qwen3.6-27B CPU decode: worker/placement sweep on real weights (2026-07-14)

**Issues:** #4625 (C2), #4624 (C1), epic #4623. **Box:** datacenter CPU server
(dual EPYC-7742, 256 threads, **8 NUMA nodes**, ~1 TB RAM, AVX2, no GPU).
**Model:** resident `Qwen3.6-27B-Q4_K_M.gguf` (16.5 GB). **Tool:** `cmd/q4kdiag -decode 40 -warmup 2`
(greedy decode from the 22-token oracle prefill; `decode_tok_s = 40 / timed_wall`).
Weights placed with `numactl --interleave=all` unless noted; worker count set by `FAK_WORKERS`.

## Results (single-stream decode tok/s, real 27B)

| placement | workers | int8 Q4_K (FAK_KQ_INT8=1) | f32 Q4_K (FAK_KQ_INT8=0) |
|---|---|---|---|
| default (node-0 first-touch) | 256 (uncapped) | **0.597** | — |
| interleave=all | 256 | 0.633 | — |
| interleave=all | 16 | 0.560 | — |
| interleave=all | 32 | 0.992 | 1.184 |
| interleave=all | 64 | 1.404 | **1.622** |
| interleave=all | 96 | 1.423 | — |
| interleave=all | 128 | 1.334 | 1.512 |
| interleave=all | 192 | 1.188 | — |
| cpunodebind=0 membind=0 | 32 | 0.832 | — |

`first_token_id=248068` (the oracle) on **every** cell — argmax stable across all
placements, worker counts, and both kernels.

## Findings

1. **Worker count is the dominant knob, and the uncapped default is the worst case.**
   The resident Q4_K decode path runs at `FAK_WORKERS`=GOMAXPROCS=256 by default (it does
   **not** consult the Q8 `amd64-manycore-cap`). At 256 workers across 8 NUMA nodes the
   `parFor` shared-cursor + busy-wait barrier collapses decode to **0.597 tok/s**. Dialing
   workers down to ~64 lifts it to **1.62 tok/s — a 2.7× win from an env var, no code.**
   The knee is ~64 (int8 peaks w64–96, f32 peaks w64); beyond it, more workers regress.

2. **The exact f32 Q4_K path beats the int8 Q4_K path for decode here.** f32 wins at every
   matched worker count (w64: 1.62 vs 1.40; w32: 1.18 vs 0.99; w128: 1.51 vs 1.33). Decode
   on this box is **barrier/synchronization-bound, not kernel-compute-bound**, so the int8
   reducer's ~8× kernel advantage never materializes — instead its per-step activation-Q8
   quantization is pure added overhead. **This redirects C1 #4624:** flipping `FAK_KQ_INT8`
   on is *not* justified by CPU server decode speed. (int8 remains argmax-faithful — see the C1
   first-token witness — and its bytes-halving still matters for a genuinely bandwidth-bound
   regime; it just isn't the decode win here.)

3. **Interleave beats single-node** (w32: 0.992 interleaved vs 0.832 node-0-bound), i.e.
   aggregate cross-node bandwidth helps even paying remote-access latency — direct support
   for the C2 per-node **replication** thesis (make 8/8 accesses local rather than 1/8).

4. **We are ~6% of the memory roofline.** Peak 1.62 tok/s × ~15 GB/token (f32 Q4_K is
   0.5625 B/param) ≈ **~24 GB/s effective**, against a dual-EPYC-7742 roofline of
   ~380 GB/s (16 DDR4-3200 channels). ~16× of bandwidth is stranded behind the `parFor`
   barrier and node-0 placement — exactly the headroom C2's barrier-free per-node schedule
   targets.

## Implication for the 10 tok/s goal

Env-only tuning (workers≈64 + interleave, f32 kernel) tops out near **1.6 tok/s** — a real
2.7× over the shipped default, but ~6× short of 10. The remaining gap is **not** a kernel or
a quant problem; it is the cross-socket `parFor` barrier + node-0 weight placement. Closing
it needs the **C2 #4625 code**: per-NUMA-node weight replication (~15 GiB × 8 = 120 GiB, fits
the ~945 GiB free) + a static/per-node barrier-free decode schedule with `LockOSThread` +
`sched_setaffinity` pinned workers reading node-local replicas. This witness quantifies why
that redesign — not the int8 flip — is the primary lever, and gives the worker-knee (~64,
≈8/node) the schedule should start from.

Raw: `q4kdiag -decode` RESULT lines captured on-box; tool committed at e7b407214.
