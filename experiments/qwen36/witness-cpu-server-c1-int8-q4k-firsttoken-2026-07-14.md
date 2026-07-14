# C1 witness — int8 Q4_K decode is argmax-faithful on real Qwen3.6-27B weights (2026-07-14)

**Issue:** #4624 (C1, epic #4623). **Status:** the first-token / argmax-agreement
blocker is **WITNESSED** on real weights. The decode-tok/s half is separate (C2 #4625).

## What was run

On the datacenter CPU server (dual EPYC-7742, 256 threads, 8 NUMA nodes, ~1 TB RAM,
AVX2, no GPU), against the resident real GGUF
`Qwen3.6-27B-Q4_K_M.gguf` (16.5 GB), built from the on-box fak checkout
(`HEAD df8499eb`, clean) with the box Go 1.26.

`cmd/q4kdiag` prefills the fixed 22-token "Say OK." ChatML oracle and prints the
top-8 first-token logits. It was run twice, toggling only the int8 Q4_K reducer:

```
FAK_Q4K=1 FAK_KQ_INT8=0 q4kdiag -gguf Qwen3.6-27B-Q4_K_M.gguf   # exact f32 Q4_K decode
FAK_Q4K=1 FAK_KQ_INT8=1 q4kdiag -gguf Qwen3.6-27B-Q4_K_M.gguf   # int8 (VPMADDWD) Q4_K decode
```

## Result — top-8 first-token logits

| rank | id | f32 Q4_K logit (int8=0) | int8 Q4_K logit (int8=1) | Δ |
|---|---|---|---|---|
| 1 | **248068** (`<think>`, the oracle) | 28.2392 | 28.2415 | +0.0023 |
| 2 | 3793  | 23.3626 | 23.2394 | -0.1232 |
| 3 | 31248 | 17.9527 | 17.8851 | -0.0676 |
| 4 | 11245 | 16.7981 | 16.7408 | -0.0573 |
| 5 | 547   | 14.7255 | 14.7525 | +0.0270 |
| 6 | 248069| 14.4601 | 14.4229 | -0.0372 |
| 7 | 10092 | 14.2613 | 14.2311 | -0.0302 |
| 8 | 220   | 13.9766 | 13.9140 | -0.0626 |

- **Argmax agreement:** both paths put **248068** first at logit ~28.24 — matches the
  llama.cpp q4_k_m oracle and fak's Q8 reference.
- **Full top-8 rank order is identical** between the int8 and f32 Q4_K paths; every
  logit agrees to within ~0.12 (max |Δ| on rank 2), well under the gap to the next token.
- Resident memory: int8 path Go `sys=44.6 GB` (heap 34.5 GB) — trivial on ~1 TB.

## Interpretation

The int8 Q4_K reducer's only approximation is the **activation** Q8 quantization —
which the shipping lean-Q8 default already does for every matmul — while keeping the
**weights** as exact Q4_K nibbles. This witness confirms that on real 27B weights that
approximation does **not** perturb the argmax or the top-8 rank order at the first
decoded token. This is the "first-token id 248068 agreement" rung named as the C1
blocker in `experiments/qwen36/cpu-decode-int8-q4k-lever-2026-07-09.md`.

## What remains for C1 to ship the default flip

- **Decode tok/s** on this box (int8 vs f32 Q4_K), the performance half — tracked with
  the placement/worker sweep under C2 #4625 via `q4kdiag -decode`.
- **Coherence over many tokens** (not just the first): the #4273 long-prompt gate on
  the int8 path — C4 #4627. The first-token argmax match is necessary but not
  sufficient for the full-generation coherence guarantee.

The kernel-level rungs (`TestQ4KReduceAsmMatchesScalar`,
`TestQ4KNativeMatchesGGUFToQ8Reference`, `TestQ4KInt8DecodeFaithfulAMD64`) were already
green; this is the first **real-weights 27B** confirmation.
