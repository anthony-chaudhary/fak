# P5-01 Qwen2.5-7B Q8 on the 8GB RTX 4070: a VRAM-fit OOM, not a missing run (#376)

> **Resolution of [#376](https://github.com/anthony-chaudhary/fak/issues/376)** —
> *"P5-01 (Qwen2.5-7B Q8) benchmark was not run due to API rate limit … may OOM on 8GB GPU."*
> The named blocker — an **API rate limit during the agent workflow** — is transient and
> orthogonal to the result; it tells us nothing about whether the model runs. The substantive
> question the issue raises is its own **Risk Assessment #1: OOM during load** ("7GB > 6.5GB
> available"). That question is **answerable now from artifacts already in this tree**, and the
> answer is: **Qwen2.5-7B Q8_0 does not fit in the ~6.5 GiB free on this 8GB board — it OOMs at
> load.** This is the issue's own **Next Step #3** — *"If OOM: Document VRAM requirement as upper
> bound for 8GB GPU."* Grounded in the committed same-box 3B witness and the committed 7B-Q8 CPU
> witness; **no new GPU measurement is claimed** (same discipline as the sibling resolutions
> [#377](F32-BASELINE-GRAPHCOMPILE-377.md) / [#378](Q8-WSL-CUDA-RUNTIME-378.md) /
> [#379](Q8-CLOCK-RESOLUTION-379.md)).

## The arithmetic that closes it (Q8 weight bytes vs the board)

Q8_0 stores each block of 32 weights as 32×int8 + one fp16 scale = 34 bytes → **1.0625
bytes/param**. GPU-resident weight footprint is therefore `params × 1.0625`, and decode never
needs the bf16 source on the device (it is quantized host-side at load, `quantized_at_load`).

**Anchor — the committed same-box 3B Q8 run.** [`qwen2.5-3b-q8-cuda-4070.json`](qwen2.5-3b-q8-cuda-4070.json)
was captured on *this exact box* (RTX 4070 Laptop, 8188 MiB, ~6.5 GiB free, WSL2 — see
[`README.md` §Conditions](README.md#conditions-this-run)) and is **~3.5 GiB resident** for
Qwen2.5-3B (≈3.09B params, tied embeddings):

```
3.09e9 params × 1.0625 B = 3.28e9 B = 3.06 GiB   (Q8 weights)
3.5 GiB resident − 3.06 GiB weights ≈ 0.44 GiB   (fixed runtime: CUDA context + activation/KV buffers)
```

**Projection — Qwen2.5-7B.** Qwen2.5-7B-Instruct is ≈**7.6B params** (HF count; unlike the 3B it
does *not* tie embeddings — `embed_tokens` and `lm_head` are two separate 152064×3584 matrices).
Even taking the nominal "7.0B" as a hard lower bound, the **Q8 weights alone already exceed the
free budget**:

| param count | Q8 weights | + ~0.44 GiB runtime | fits in ~6.5 GiB free? |
|---|---:|---:|:--:|
| 7.0B (nominal lower bound) | 6.93 GiB | 7.37 GiB | **no** (weights alone > 6.5) |
| **7.6B (actual HF count)** | **7.52 GiB** | **~7.96 GiB** | **no** (OOM by ~1.5 GiB) |
| 3.09B (committed witness, for scale) | 3.06 GiB | 3.5 GiB | yes (25.1 tok/s, fits) |

The conclusion is **robust across the entire plausible param range**: the weight residency
(6.9–7.5 GiB) is above the ~6.5 GiB free regardless of the exact count, so the result does not
hinge on a precise figure. And the **full board is 8188 MiB = 7.99 GiB** — so even a *headless*
card with nearly the whole board free only barely clears the ~7.5 GiB weights, leaving
sub-0.5 GiB for the ~0.44 GiB fixed runtime + the KV cache that grows with context: marginal at
best, OOM in practice. The `-lean` flag the issue reaches for changes **host-side** load staging,
not the **GPU-resident** weight bytes, so it cannot rescue the fit.

## Why this is an OOM, not a slow run

This is the issue's **Risk #1 (OOM during load)**, not #2 (fits with minimal headroom) or #3
(works with `-lean`). The Q8 weights are uploaded to the device *before* the first token; with
6.5 GiB free and ≥6.9 GiB of weights, the allocation fails during load — the run never reaches
prefill/decode. The only configuration that fits a 7B in this budget is a **lighter quant**
(Q4_K ≈ ~4.3 GiB resident), which is a *different precision* than the Q8 #376 specifies — i.e. a
different benchmark, not this one.

## The model and the Q8 path are independently witnessed (just not on this GPU)

The 7B Q8 *model* is not hypothetical — it is committed and runs at Q8_0, off the GPU:
[`../model-ladder/modelbench-qwen25-7b-q8.json`](../model-ladder/modelbench-qwen25-7b-q8.json)
is `qwen2.5-7b-instruct (qwen2) [lean]`, `precision: Q8_0`, `quantized_at_load`, decoding at
**8.68 tok/s** on the CPU path. So the load+quantize pipeline and the model footprint are real;
what does not exist is GPU headroom for them on an 8GB laptop board. The CUDA Q8 path itself is
likewise witnessed at 3B on this board ([#378](Q8-WSL-CUDA-RUNTIME-378.md), 25.14 tok/s,
argmax-exact). Nothing is broken — the model is simply ~1.5 GiB too large for the device.

## VRAM requirement documented (the upper bound the issue asked for)

> **Qwen2.5-7B Q8_0 needs ≈7.5 GiB of GPU-resident weights + ≈0.5 GiB runtime ≈ 8.0 GiB total.**
> On an **8 GB** laptop board where WSL2 + the NVIDIA driver + the desktop already consume
> ~1.5 GiB (leaving ~6.5 GiB free), **it does not fit and OOMs during load.** 8 GB is therefore
> the practical **floor** for 7B-Q8 only on a *dedicated/headless* card with nearly the whole
> board free — not on this shared 8 GB laptop GPU.

For completeness against the issue's **Next Step #4** ("if success, document tok/s"): decode is
bandwidth-bound (throughput ∝ 1/weight-bytes), so *were* it to fit on a larger card, scaling the
committed 3B witness gives `25.1 × (3.06 / 7.52) ≈ 10.2 tok/s` — squarely inside the issue's
"10-20 if it fits" target. That is a **contingent projection**, not a measured number, and is moot
on *this* 8 GB box where the model OOMs first.

## What is genuinely still open (and where)

The **fit question** (#376's actual risk) is resolved by arithmetic robust across the param range.
The single residual is the **live on-box OOM observation** — running the issue's command when the
API quota resets and watching the device allocation fail. That needs the RTX 4070 / WSL2 box; it
cannot be produced on a host with no NVIDIA GPU, and it would only *confirm* (not change) the
documented upper bound. If a future run targets a card that *does* fit 7B Q8 (≥ ~10 GiB free), the
~10 tok/s projection above becomes the bar to verify.

## Reproduce (the witnesses this rests on)

```bash
# the same-box 3B Q8 anchor: ~3.5 GiB resident, 25.14 tok/s, on the RTX 4070
cat experiments/gpu/qwen2.5-3b-q8-cuda-4070.json          # decode.tok_per_sec 25.14, precision Q8_0

# the 7B Q8 model + path, witnessed off-GPU (CPU): proves the model & quantize-at-load are real
cat experiments/model-ladder/modelbench-qwen25-7b-q8.json # model qwen2.5-7b-instruct, precision Q8_0, 8.68 tok/s

# the upper-bound check (any box): 7B-Q8 weights alone exceed an 8GB board's free budget
python3 -c "p=7.6e9; b=p*1.0625; print(f'{b/2**30:.2f} GiB Q8 weights vs ~6.5 GiB free')"

# the live on-box confirmation (needs the RTX 4070; expected: OOM at load):
FAK_CUDA_Q8=1 go run -tags cuda ./cmd/gpucheck -hf <qwen2.5-7b-dir> -lean -backend cuda -n 8
```
