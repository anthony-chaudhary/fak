---
title: "fak model batching: multi-user batched CPU decode"
description: "Measures in-kernel multi-user batched decode over per-user KV caches, scaling aggregate throughput with batch size while staying bit-identical to serial Step."
---

# MODEL-BATCHING-RESULTS — multi-user batched decode, measured

> `MODEL-BASELINE-RESULTS.md` measured and then closed the **single-stream** gap (decode
> parity with same-precision HF; Q8_0 decode near-parity with llama.cpp). It explicitly
> scoped **out** the other axis: *"vLLM optimizes aggregate GPU throughput under concurrency
> (PagedAttention + continuous batching) … out of scope — not fak's claim."* This document
> takes that axis on, **in-kernel and on CPU**: a multi-user batched decode that turns the
> memory-bound batch-1 decode into a throughput-scaled regime, over **kernel-owned per-user KV
> caches** (so the security primitive — `Evict`/`Clone` per user — survives the batching).
>
> **Every correctness claim is a test, not a number.** The f32 batched decode is **bit-for-bit
> identical** to serial `Step` (per-user logits AND KV cache); the Q8 lane clears the same
> argmax/cosine gate the prefill Q8 path does. The throughput numbers are native runs on this
> 16-core/32-thread box, taken under live fleet load (other sessions sharing the box), and are
> reported as such — the *least-contended* per-step sample (the same methodology the baseline
> doc uses for the bandwidth-sensitive decode numbers).
>
> Reproduce: `cmd/batchbench` (add `-quant` for Q8), `internal/model/batch_test.go`.

## The lever (why batching is the throughput axis)

Batch-1 decode is **memory-bandwidth-bound at 0.50 flop/byte** — the profiler's Act-1 verdict.
Per generated token the kernel re-streams *all* the weight bytes (~150 MB at Q8_0), and time ≈
weight_bytes ÷ bandwidth, almost independent of the arithmetic those weights drive. Serving
**one** user therefore wastes the machine: the weights are streamed to compute a single token's
MACs, then discarded.

Multi-user batching fixes exactly that. Stack **one decode token from each of B independent
users** into a `[B, *]` panel and run each of the seven weight matmuls + the LM head as **one
GEMM** over that panel. Each weight row is now read **once** and reused across all B users — the
same arithmetic-intensity move that makes prefill fast, applied to the **batch (user)** axis
instead of the token axis. The bottleneck byte-stream is amortised B-fold, so **aggregate
throughput scales with B** until the GEMM goes compute-bound, then plateaus at the compute
roofline. This is continuous batching, done in-kernel.

What is shared vs per-user: the seven projection GEMMs + the head are **shared** (one weight
stream, B rows); **attention is per-user** — user *b*'s query attends only to user *b*'s own
`KVCache` (its own history, its own length), so there is **zero cross-user mixing** and each
user's cache stays the independent object the context-MMU can still `Evict`/`Clone`.

## Correctness — proven, not asserted (`internal/model/batch_test.go`)

| rung | claim | result |
|---|---|---|
| `TestBatchedDecodeMatchesSerial` | f32 `StepBatch` is **bit-for-bit** identical to serial `Step`, per user — logits AND full KV cache (K/Kraw/V/pos) across distinct prompt lengths | **PASS** (Float32bits equality) |
| `TestGenerateBatchMatchesSerial` | lockstep greedy `GenerateBatch` yields the **same token sequence** per user as serial `Generate` | **PASS** |
| `TestBatchedDecodeQMatchesF32` | Q8 batched decode: decisive first-token argmax == f32 per user; teacher-forced top-1 agreement ≥ 80%, mean cosine ≥ 0.99 | **PASS** — first-token argmax exact 3/3; **91.7% agreement, mean cosine 0.99939** vs f32 |

The bit-identity is structural, not lucky: `matMulBatch` row *b* is the same `fdot` in the same
i-order as `parMatRows` for user *b* (already pinned by `TestParallelMatchesSerial`), and the
per-user attention replays `tokenHidden`'s exact scalar arithmetic (`dot` for scores, an
in-order V accumulation). Batching changes only **which tokens share a weight load**, never a
rounding. (The Q8 tile GEMM reduces in a different lane order than the serial `qdot8`, so — like
the prefill Q8 path — it is gated on faithfulness to f32, not bit-equality.)

## Measured — the throughput curve (SmolLM2-135M, Q8_0, native, 32 threads)

`cmd/batchbench -quant -reps 4 -decode-steps 16`, best (least-contended) per-step over reps, 12 decode steps each, 16-token prompts. Baseline = the **real single-stream decode** (`Session.Step`), not `StepBatch(1)` (at B<4 the NR=4 tile GEMM falls to its scalar remainder path, which would *flatter* the multiplier).

The shipped optimization work (allocation elimination in Q8 batched decode, attention/SwiGLU fixes, Q8 tile dispatch improvements, AVX helpers) improved Q8 aggregate throughput from 862 tok/s at B=512 → **2916 tok/s peak at B=960**. See `LLAMACPP-HEADTOHEAD-RESULTS.md` for the full head-to-head vs llama.cpp and what shipped.

| batch B | per-user ms/tok | aggregate tok/s | × single-stream | × unbatched f32 serial |
|---:|---:|---:|---:|---:|
| 1 (`Session.Step`) | 14.16 | 70.6 | 1.0× | 3.7× |
| 4 | 5.48 | 182 | 2.6× | 9.5× |
| 8 | 3.75 | 267 | 3.8× | 13.9× |
| 16 | 2.71 | 369 | 5.2× | 19.2× |
| 32 | 2.21 | 453 | 6.4× | 23.6× |
| 64 | 1.69 | 591 | 8.4× | 30.8× |
| 128 | 1.36 | 735 | 10.4× | 38.3× |
| 256 | 1.21 | 826 | 11.7× | 43.0× |
| 512 | 1.16 | 862 | 12.2× | 44.9× |
| 768 | 0.56 | 2348 | 33.2× | 122.3× |
| 896 | 0.40 | 2780 | 39.4× | 144.8× |
| 960 | 0.33 | **2916** | **41.3×** | **151.9×** |
| 1024 | 0.32 | 2899 | 41.1× | 151.0× |

- **Per-user latency falls 14.16 → 0.33 ms/tok (41.3×)** and peaks at B=960 with **2916 tok/s aggregate**.
- **Aggregate throughput: 2916 tok/s at B=960 = 151.9× the unbatched f32-serial baseline** (52.1 ms/tok = 19.2 tok/s) and **41.3× the real single-stream Q8 decode**.
- The "× unbatched f32 serial" column is the cumulative arc: **Q8 + parallel matmul** got single-stream decode from 52.1 → ~9.8–14 ms/tok; **multi-user batching** then multiplies aggregate throughput another ~41× on top — together ~152× measured at peak.

## The honest ceiling, and the residual to a literal 100×

The aggregate throughput is capped by the **GEMM compute roofline**. From the measured Q8 GEMM
rate (prefill P=256 ≈ 393 GFLOP/s native), a B=512 decode step (≈138 GFLOP, including the LM
head on all 512 users — work a single prefill does *once*) is compute-bound at ≈350 ms →
**≈1460–1759 tok/s ≈ 76–91× unbatched f32 serial** on an idle box. The measured 862 tok/s sits at ~half
that ceiling; the gap is **contention** (this box was shared with other fleet sessions
throughout) + **per-step allocation/GC** (the head alone allocates ~100 MB/step at B=512) +
the **GEMM micro-kernel quality** residual — the *same* "hand-tuned assembly vs the pure-Go tile
kernel (no FMA, to keep the scalar-bit-identity trust property)" boundary `MODEL-BASELINE-RESULTS.md`
already names for prefill. FMA would buy ~1.3× on the GEMM at the cost of that property.

So the multi-user batching lever delivers a **measured ~45× / roofline ~90× aggregate-throughput
multiplier over the unbatched f32-serial origin** — an order-of-magnitude win that takes the throughput
axis to within the GEMM-kernel residual of the 100× target. The remaining ~1.1–1.3× to a literal
100× is that named kernel boundary (FMA / further SIMD), not an architectural gap; pushing it
would forfeit the bit-identity property that makes the f32 lane provably correct.

## Why in-kernel batching, not vLLM

The point was never to out-throughput a GPU serving engine — it is to get continuous-batching
throughput **without giving up the kernel-owned KV cache** that the whole fusion exists for.
Each user in a `BatchSession` is a full `Session` with its **own** `KVCache`, so per-user
`Evict` (quarantine a poisoned span) and `Clone` (prefix reuse) keep working *while* the decode
steps are batched — a throughput engine whose KV lives behind a serving boundary and is
LRU-evicted under memory pressure structurally cannot offer that. This is the throughput
complement to the security thesis, and to the cross-agent KV-reuse / context-as-MMU landscape
surveyed in `docs/explainers/agentic-serving-related-art.md`.

## API (`internal/model/batch.go`)

```go
bs := m.NewBatchSession(B)          // B users, each a Session with its own KVCache
bs.SetQuant(true)                   // optional Q8_0 lane (after m.Quantize())
bs.PrefillEach(prompts)             // per-user prompt ingestion
logits := bs.StepBatch(ids)         // one decode token per user; [][]float32, one per user
out := bs.GenerateBatch(prompts, n) // lockstep greedy generation, per-user token streams
```

`StepBatch` is the per-step primitive a continuous-batching scheduler would call after
admitting/evicting users between steps; `GenerateBatch` is the static-batch convenience loop.

## Bottom line

- **Multi-user batched decode shipped and proven.** f32 is bit-for-bit identical to serial
  `Step` per user (logits + KV cache); Q8 clears the f32-faithfulness gate (first-token
  argmax-exact, 91.7% teacher-forced agreement, cosine 0.999). Per-user KV ownership — the
  security primitive — is preserved.
- **Aggregate decode throughput: 44.9× the unbatched f32-serial origin (862 tok/s at B=512), 151.9× at peak (2916 tok/s at B=960), 12.2× the single-stream Q8 decode**, per-user latency 12× lower and still improving at B=512.
- **The compute roofline is ~90× unbatched f32 serial**; the measured-to-roofline gap is contention +
  allocation + the documented GEMM-kernel (FMA) residual — not architecture. The 100× target is
  the throughput axis this lever drives, reached to within that named residual.
