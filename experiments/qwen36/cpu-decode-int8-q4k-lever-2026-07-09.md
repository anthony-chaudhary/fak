# Qwen3.6-27B q4_k_m — CPU decode: the int8 Q4_K lever (2026-07-09)

Single-stream CPU decode of Qwen3.6-27B q4_k_m on an AVX2 CPU server (box-873af63b,
253 GB RAM, no GPU) currently runs the **default lean-Q8 path** and decodes at
**0.868 tok/s** (`experiments/benchmark/runs/by-machine/box-873af63b/20260709T003308Z-cpu-throughput-curve`).
That is the top optimization target for this model on a CPU server.

## Why it is slow, and the ladder out

The decode weight-matmul dominates. Three resident regimes, cheapest bytes/param last:

| Path | bytes/param | gate | note |
|---|---|---|---|
| lean Q8 (**default**) | 1.125 | — | Q4→f32→Q8 round-trip at load; Q8 GEMV |
| resident Q4_K **scalar** | 0.5625 | `FAK_Q4K=1` | fewer bytes but **compute-bound** on scalar f32 dequant → *slower* (~0.3 tok/s historically) |
| resident Q4_K **int8 AVX2** | 0.5625 | `FAK_Q4K=1` + `FAK_KQ_INT8=1` | raw nibbles, int reductions → the real win |

The scalar Q4_K path streams fewer bytes but pays a 256-wide f32 dequant + f32 dot per
super-block per token, so it is compute-bound and loses to lean-Q8. The int8 AVX2 path
keeps the exact Q4_K nibbles and does the dot as int8×int8→int32 (VPMADDWD), spending
compute proportional to the compact byte stream — the same shape llama.cpp's
`vec_dot_q4_K_q8_K` takes.

## On-box kernel A/B (this session, box-873af63b, tier=AVX2)

`go test ./internal/model -bench 'BenchmarkQ4K(F32|Int8)GEMV'` (out=2048, in=6144):

| kernel | ns/op |
|---|---|
| `Q4KF32GEMV` (scalar default decode GEMV) | ~14,400,000 |
| `Q4KInt8GEMV` (AVX2 int8 reducer) | ~1,700,000 |

**≈ 8× on the weight-matmul kernel.** At 0.5625 B/param the 27B streams ~15 GB/token, so
the int8 path is bandwidth-bound rather than dequant-bound — the regime where the box's
memory bandwidth (not the CPU) sets the ceiling, and where a multi-tok/s decode is reachable.

## Faithfulness of the int8 path

The int8 path's only approximation is quantizing the **activation** to Q8_0 — exactly what
the shipping lean-Q8 path already does for every matmul — while keeping the **weights** as
exact Q4_K nibbles (no Q8 re-quant). So it is *strictly more faithful* to the q4_k_m
artifact than the current default, not less.

Witness rungs:

- `TestQ4KReduceAsmMatchesScalar` (amd64) — AVX2 integer reducer bit-identical to scalar. ✅
- `TestQ4KNativeMatchesGGUFToQ8Reference` — f32 native Q4_K ≈ Q8 reference (cosine 0.999, argmax parity). ✅
- `TestQ4KInt8DecodeFaithfulAMD64` (**new, ebc41f9e7**) — AVX2 int8 end-to-end decode dot
  within 0.05 activation-Q8 tolerance of the f32 dequant reference at the 27B hidden dim
  (in=5120); measured **2.25%** on representative bounded weights. ✅
  Closes the gap that `TestQ4KInt8DotMatchesF32` left (it skips on amd64 because the gate is
  off there).

## Actionable now

Serve the 27B on a CPU server with the int8 Q4_K decode lever on:

```
FAK_Q4K=1 FAK_KQ_INT8=1 fak serve --gguf Qwen3.6-27B-Q4_K_M.gguf --context-budget-tokens 8192
```

(`--context-budget-tokens` caps KV so the auto-sized 262144 window does not overflow the fit
budget — see the throughput-curve manifest finding.)

## Remaining blocker to flipping the default

`FAK_KQ_INT8` stays default-off pending a **real-weights 27B witness**: the q4_k_m greedy
continuation + first-token id (248068) agreement that `cmd/q4kdiag` runs on the actual GGUF.
The kernel-level and synthetic-forward faithfulness rungs above are now green on amd64; the
open item is the real-model greedy check on a box that has the GGUF resident.
