---
title: "Mac metal node overnight run plan — 2026-06-28"
description: "Per-box overnight data-collection plan for the Apple-Silicon Metal verify node: the fak-kernel Qwen3.6-27B decode witness, what was collected tonight, and the resumable runbook."
---

# Mac metal node overnight run plan — 2026-06-28

The Apple-Silicon companion to [`DGX-OVERNIGHT-PLAN-2026-06-28.md`](DGX-OVERNIGHT-PLAN-2026-06-28.md).
That doc covers the GPU-server/CPU boxes reached over the Slack control-bridge; this one
covers the **Mac Metal verify node** reached over the tailnet gateway — the box where fak's
own Apple-Silicon Metal kernel actually runs. Tonight it collected its first witness in a
while: the GLM-server boxes were saturated last night, but the Mac had collected *nothing*,
so it was the highest-novelty surface.

## What changed tonight (the new condition)

The Mac gateway was found serving **`coder14b`** — a `fak serve --provider openai
--base-url …` **proxy** in front of a `llama-server` (llama.cpp) running **Qwen2.5-Coder-14B**,
held up by a `KeepAlive` launchd agent (mislabelled `com.fak.qwen36-model`). That is *not*
fak's kernel and *not* the model we want a witness for. Per operator direction, the coder14b
stack was **replaced** with fak's own Metal kernel serving **Qwen3.6-27B Q4_K_M**:

```sh
# the -tags fakmetal fak build (a metal=1 binary), on the Mac:
FAK_Q4K=1 FAK_GATEWAY_KEY=$(cat ~/.fak-gateway-key) \
  fak serve --metal --gguf <mac>/.cache/fak-models/gguf/Qwen3.6-27B.q4_k_m.gguf \
            --model qwen3.6-27b --addr $FAK_MAC_GATEWAY \
            --require-key-env FAK_GATEWAY_KEY \
            --context-budget-tokens 8192
```

`--context-budget-tokens 8192` is **load-bearing on the 36 GB box**: the default plan sizes
the KV cache for the model's full native window (**192 GiB**), so the boot path's capacity
pre-check refuses it with a typed `FitTooBig` (15.40 GiB weights + 192 GiB KV ≫ 30.6 GiB
available) rather than OOMing. Bounding the context to 8 K tokens drops KV to a few GiB and
the plan fits. The real Mac host/user resolve from the gitignored `fak-mac.local.ps1` (see
[`../fak/scrubbing-real-values.md`](../fak/scrubbing-real-values.md)).

## What it collected (WITNESSED, fak's own Metal kernel)

A single-stream decode-length sweep against the live `--metal` serve (`engine=inkernel`),
`prompt_tokens=25`, `finish=length`:

| max_tokens | completion | wall (s) | tok/s | note |
|---|---|---|---|---|
| 16 | 16 | 85.7 | **0.187** | **cold** first-request warm-up outlier — discard |
| 32 | 32 | 19.6 | **1.63** | warm |
| 64 | 64 | 38.2 | **1.67** | warm |
| 128 | 128 | 69.6 | **1.84** | warm — approaching steady-state |

**The finding:** warm Qwen3.6-27B Q4_K decode in fak's Metal kernel runs at **~1.6–1.9 tok/s**
and *climbs with generation length* (1.63 → 1.67 → 1.84) as the one-time cold cost amortizes —
the same prefill-amortization shape as the GLM-server GLM-5.2 curve, but ~8× the throughput on
this far smaller model/box. The cold 16-tok point (0.187) is the first-request warm-up and is
flagged separately, never folded into the steady-state.

## Honesty boundary

- The number is **WITNESSED** on fak's own Apple-Silicon Metal kernel (`--metal`,
  `engine=inkernel`), not a third-party engine and not the displaced `coder14b`/llama.cpp.
- The cold first-request point is labelled cold and excluded from the warm steady-state.
- It is a timed live-serve completion (`completion_tokens` over wall, prefill included), the
  same non-forgeable shape as the GLM-server witness — comparable across boxes.

## Resume / next conditions

The decode-length curve is now characterized. Genuinely-new Mac conditions for a later night:
warm steady-state at longer generations (256/512 tok), a prefill-length sweep (the upload +
GPU round-trip cost the prefill-witness memory describes), `FAK_METAL_RESIDENT` resident-forward
if available, and a 2-stream concurrency point. Re-witness cadence: 14 days (the backlog task
`witness-qwen36-27b-metal-decode`).
