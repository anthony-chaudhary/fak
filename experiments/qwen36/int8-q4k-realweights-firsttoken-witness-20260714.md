# Qwen3.6-27B int8 Q4_K decode — real-weights first-token witness (2026-07-14)

Campaign: epic #4623 (Qwen3.6-27B CPU decode → 10 tok/s). Closes the correctness
blocker on **C1 #4624** (flip `FAK_KQ_INT8` decode default).

## What was witnessed

The int8 Q4_K decode reducer (`FAK_KQ_INT8=1`) is **numerically faithful to the f32
Q4_K reference on real Qwen3.6-27B-Q4_K_M weights** — not just a synthetic/kernel rung.
Both paths were run through `cmd/q4kdiag` on the real 16.5 GB GGUF (the 22-token
"Say OK." ChatML oracle prefill), and the first-token logit distribution is compared.

**Host:** CPU server — 2× AMD EPYC 7742 (256 threads), 8 NUMA nodes, ~1 TB RAM,
AVX2 (no AVX-512), no GPU. Go 1.26, resident-Q4_K path (`FAK_Q4K=1`).
**Model:** Qwen3.6-27B-Q4_K_M.gguf (16,547,398,784 bytes).

## Result — first-token top-8 (oracle wants id 248068 `<think>` at ~28.3)

| rank | `FAK_KQ_INT8=0` (f32 Q4_K reference) | `FAK_KQ_INT8=1` (int8 Q4_K) |
|---|---|---|
| 1 | **248068**  logit 28.2392 | **248068**  logit 28.2415 |
| 2 | 3793   23.3626 | 3793   23.2394 |
| 3 | 31248  17.9527 | 31248  17.8851 |
| 4 | 11245  16.7981 | 11245  16.7408 |
| 5 | 547    14.7255 | 547    14.7525 |
| 6 | 248069 14.4601 | 248069 14.4229 |
| 7 | 10092  14.2613 | 10092  14.2311 |
| 8 | 220    13.9766 | 220    13.9140 |

- **Argmax agreement:** both paths put **248068** first — the oracle's `<think>` token
  (llama.cpp q4_k_m oracle + fak's Q8 path also put 248068 first at ~28.3).
- **Full top-8 rank order is identical**; logits agree to **~0.01–0.12** (≤0.5%). The
  int8 path's only approximation is the activation Q8 quant (which the shipping lean-Q8
  default already does); the Q4_K weight nibbles stay exact.

## Why this matters

This is the one open rung the lever write-up
(`experiments/qwen36/cpu-decode-int8-q4k-lever-2026-07-09.md`) named as the blocker to
flipping the default: *"the q4_k_m greedy continuation + first-token id (248068)
agreement that `cmd/q4kdiag` runs on the actual GGUF."* The synthetic/kernel rungs
(`TestQ4KReduceAsmMatchesScalar`, `TestQ4KNativeMatchesGGUFToQ8Reference`,
`TestQ4KInt8DecodeFaithfulAMD64`) were already green; this is the **real-weights 27B**
confirmation.

## Still gated (not claimed here)

- **Long-context coherence (C4 #4627 / #4273):** first-token parity ≠ N-token coherence.
  The default flip stays gated on the coherence A/B over a long prompt.
- **Decode tok/s (C1/C2):** the throughput lever + NUMA placement sweep are witnessed
  separately (see the `witness-cpu-decode.sh` sweep artifact).

## Reproduce

```sh
# on the CPU server, resident-Q4_K path:
FAK_Q4K=1 FAK_KQ_INT8=0 q4kdiag -gguf Qwen3.6-27B-Q4_K_M.gguf   # f32 reference
FAK_Q4K=1 FAK_KQ_INT8=1 q4kdiag -gguf Qwen3.6-27B-Q4_K_M.gguf   # int8 path
# assert both print id=248068 first with an identical top-8 ordering.
```
