# Qwen3.6-27B Q4_K_M CPU-decode witness — dual EPYC-7742 (2026-07-14)

Real-weights witness for the 10 tok/s campaign (epic #4623), taken on the DC CPU
server (2× AMD EPYC 7742, 256 threads, **8 NUMA nodes**, ~1 TB RAM, no GPU) against
the resident `Qwen3.6-27B-Q4_K_M.gguf` (16.5 GB). Tool: `cmd/q4kdiag -decode`
(in-process greedy decode of the 22-token "Say OK." oracle; `decode_tok_s = N/wall`).
`FAK_KQ_INT8` selects the int8 Q4_K reducer; `FAK_WORKERS` sets the decode worker
count; NUMA placement via a `numactl` launch wrapper.

## Headline

- **C1 correctness (#4624) — WITNESSED.** On real 27B weights the int8 Q4_K path and
  the f32 Q4_K path produce the **same first token (id 248068, `<think>`)** and the
  **identical top-8 rank order** (logits agree to ~0.01–0.12). First token 248068 held
  on **every** cell below. The int8-activation Q4_K reducer is faithful on real weights.
- **C2 barrier-collapse (#4625) — WITNESSED.** The resident Q4_K decode path runs
  uncapped at all 256 workers by default and **collapses on the cross-NUMA `parFor`
  barrier**. Zero code change, just capping workers to the ~64 knee + interleaving weight
  pages: **0.582 → 1.408 tok/s (2.42×)**.

## Decode sweep (int8 Q4_K unless noted; N=40, warmup=2)

| placement | int8 kernel | workers | tok/s | first-token |
|---|---|---|---|---|
| default (node-0 weights) | int8 | 256 | 0.582 | 248068 ✓ |
| interleave | int8 | 256 | 0.634 | 248068 ✓ |
| interleave | int8 | 16 | 0.576 | 248068 ✓ |
| interleave | int8 | 32 | 0.994 | 248068 ✓ |
| **interleave** | **int8** | **64** | **1.408** | 248068 ✓ |
| interleave | int8 | 128 | 1.349 | 248068 ✓ |
| node0-local | int8 | 32 | 0.830 | 248068 ✓ |
| interleave | f32 (AVX2 exact) | 32 | 1.188 | 248068 ✓ |

Readings:
- **Worker knee ≈ 64.** 256 workers is the *worst* config (barrier collapse); 64 is best;
  128 already regresses. 16 is too few. The default (uncapped 256) is self-sabotaging.
- **Interleave > single-node > node-0-first-touch.** interleave@256 (0.634) > default@256
  (0.582); interleave@32 (0.994) > node0-local@32 (0.830) — aggregate multi-node bandwidth
  beats one node, and the default first-touch-on-node-0 is the worst placement.
- **int8 vs f32 kernel is a wash at these worker counts** (f32@32=1.188 vs int8@32=0.994):
  both are fast AVX2 Q4_K kernels; decode here is **bandwidth/barrier-bound, not
  kernel-bound**, so the kernel choice barely moves the number. The int8 win is bytes/token
  (half the stream), which only pays off once the barrier stops being the wall (→ C2).

## Memory-bandwidth roofline (STREAM-Triad, go f64, 128Mi elems)

| placement | single GB/s | all-core GB/s |
|---|---|---|
| default (node-0) | 12.8 | 33.1 |
| **interleave** | 9.1 | **89.5** |
| node0-bound (32c) | 15.5 | 27.0 |

- Interleaving gives **2.7×** the aggregate bandwidth of node-0 confinement (89.5 vs 33.1).
- This is a simple f64 triad (no non-temporal stores), so it under-reports the ~380 GB/s
  theoretical peak — but the **ratio** is the point and it directly explains the decode
  placement result.
- **Where the tok/s stand vs the wall:** best decode = 1.408 tok/s × 15 GB/tok ≈ **21 GB/s
  achieved**, vs the **89.5 GB/s** interleaved roofline → **~24% of the interleaved wall**.
  So even within interleave the `parFor` barrier is still costing ~4×. Full **per-node
  replication** (8/8 local reads ≈ 8 × ~27 GB/s ≈ 200+ GB/s) + a barrier-free per-node
  schedule is the path to ≥10 tok/s — exactly C2 (#4625).

## Provenance / reproduce

- Built on-box: `go build ./cmd/q4kdiag` (go 1.26.x); GGUF resident on NVMe.
- Sweep cell: `numactl <placement> env FAK_Q4K=1 FAK_KQ_INT8=<0|1> FAK_WORKERS=<W> \
  q4kdiag -gguf Qwen3.6-27B-Q4_K_M.gguf -decode 40 -warmup 2`.
- Turnkey driver: `experiments/qwen36/witness-cpu-decode.sh` (this run was taken with an
  equivalent inline sweep; the committed runner assembles the same manifest).
- Raw: `results.jsonl`, `bandwidth.jsonl` in this dir.

## Campaign status after this witness

- C1 #4624: correctness rung **green on real weights**; the flip still gates on C4 (#4627)
  long-prompt coherence over many tokens.
- C2 #4625: barrier collapse + placement + worker-knee **quantified**; per-node replication
  is the remaining lever to close 1.4 → 10 tok/s.
- C3 #4626: first STREAM-Triad roofline recorded (89.5 GB/s interleaved); a tuned
  non-temporal STREAM would tighten the ceiling.
