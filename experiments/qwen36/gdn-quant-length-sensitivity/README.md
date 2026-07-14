<!-- RESULT NUMBERS BELOW ARE FILLED FROM result.json (H=1024, 48-layer, alog=-2 run); do not edit by hand -->
# GDN weight-quant / decode-length sensitivity — does quant error compound over long context? (device-independent)

Host-runnable witness for the **length** half of the Qwen3.6-27B GGUF quality collapse —
[#4273](https://github.com/) (the 27B degenerates into a verbatim repetition loop on ~1.3k-token
prompts while short prompts stay coherent). It is the **decode-length** sibling of
[`gdn-divergence-sensitivity`](../gdn-divergence-sensitivity), which answers the *token-3
correctness drift* on the **depth** axis. Same GDN math, orthogonal axis.

## What the on-box witnesses already established (#4273)

The on-device real-artifact runs (`q36rawrepro`, `q36safe18`, `q36tokenloop7`) narrowed the collapse to
**long-context real quantized inference**, and ruled OUT: the tokenizer/chat-template (HF
tokenization byte-identical), generic f32 Gated-DeltaNet recurrence (the Ornith **f32** oracle
matches HF greedy at short *and* 300-token horizons), batched-vs-token-loop prefill (both collapse
identically), recurrent-state carry (the long-horizon self-consistency guard is green), and
universal corruption (`2+2=4` decodes cleanly at short context). The surviving signature is:
**short coherent · long collapse · f32 fine · quant fails**, and the collapse is **quant-kind
independent** (Q8_0 and Q4_K both fail).

## The variable this isolates

In the real loader ([`safetensors_quant.go:isQuantWeight`](../../../internal/model/safetensors_quant.go))
all five `linear_attn` matmul weights — `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`,
`out_proj` — are quantized to **Q8_0** at compute time in **both** failing paths: the default
lean-Q8 path quantizes everything, and the resident-Q4_K path *refuses* `linear_attn` from raw-Q4_K
residency ([`quant_q4k.go:ResidentQ4KEligible`](../../../internal/model/quant_q4k.go) returns
`false` for `.linear_attn.`) then re-quantizes the dequant'd f32 to Q8 via the builder. That is
exactly why the collapse is quant-kind independent — the GDN path runs Q8 either way — and it means
the **one** difference between the coherent f32 oracle and the collapsing quantized runs, on the GDN
projections, is **weight quantization**. This experiment applies precisely that perturbation.

```text
seeded-random weights + inputs, P carried positions
   |
   +--> run A: 48-GDN-layer stack, f32 projection weights ----+
   |                                                           +--> rho = ‖Δhidden(lastток)‖/‖hidden‖
   +--> run B: same stack, projections round-tripped Q8_0/Q4_K +     as a function of P
```

Both runs share identical inputs and identical (pre-quantization) weights, so the **only** source of
A↔B divergence is projection weight quantization. The GDN recurrence body is copied verbatim from
[`internal/model/metal_prefill_hybrid_core.go:202-246`](../../../internal/model/metal_prefill_hybrid_core.go)
via the sibling experiment; only the projection weights are round-tripped through Q8_0 (block=32,
symmetric, `d=amax/127`, bit-faithful to [`quant.go:quantizeQ8`](../../../internal/model/quant.go))
or a per-32 affine 4-bit sub-block (a **lower bound** on Q4_K, which additionally quantizes the
sub-scales).

## The question it answers (falsifiable)

> #4273's defining symptom is **length-onset**: short prompts are fine, long prompts collapse. Does
> GDN-projection weight-quant error, fed through the delta-rule recurrent scan, **compound with
> decode length** — i.e. does rho GROW with the number of carried positions P and reach a
> decode-destabilizing magnitude by the ~1757-token failure horizon?

A near-tie of margin `m` logits at logit scale `|logit| ≈ 20` flips when the hidden state moves by
rho\* ≈ `m/|logit|` ≈ `1.75/20` ≈ **0.0875** in relative L2 (|Δlogit| ≈ |logit|·rho). The
experiment reports rho at the **last** decode token for each P.

## Run

```
go run ./experiments/qwen36/gdn-quant-length-sensitivity                      # human table
go run ./experiments/qwen36/gdn-quant-length-sensitivity -json                # machine result (result.json)
go run ./experiments/qwen36/gdn-quant-length-sensitivity -positions 16,128    # quick smoke
go run ./experiments/qwen36/gdn-quant-length-sensitivity -alog -5             # near-1-decay stress (long memory)
go run ./experiments/qwen36/gdn-quant-length-sensitivity -hidden 5120         # real-H confirm (slow)
```

No build tags, no GPU, no model file, no `internal/model` dependency — pure stdlib Go (builds even
while `internal/model` is mid-refactor). Verified green on this `windows/amd64`, `CGO_ENABLED=0` host.

## Result (H=1024, 48-GDN-layer, representative decay `alog=-2` → g≈0.88)

Relative hidden divergence rho = ‖Δhidden‖/‖hidden‖ at the **last** decode token, the **only**
difference between the two runs being projection weight quantization:

| mode (GDN projections) | rho @ P=16 | rho @ P=128 | rho @ P=512 | rho @ P=1757 | growth (1757/16) | implied \|Δlogit\| @1757 | reaches rho\*=0.0875? |
|---|---:|---:|---:|---:|:---:|---:|:---:|
| **Q8_0** — the actual GDN compute dtype in *both* failing paths | 8.22e-04 | 8.22e-04 | 8.23e-04 | **8.52e-04** | **1.04×** | 0.017 | **no** |
| **Q4_K** — per-32 affine 4-bit (lower bound) | 1.23e-02 | 1.19e-02 | 1.19e-02 | **1.26e-02** | **1.02×** | 0.25 | n/a (flat) |

The full per-P curve is in [`result.json`](result.json). **rho is essentially flat across a 110×
increase in decode length** for both modes.

### Robustness: it holds under near-1 decay too

The decay strength (`A_log`) is the load-bearing knob — if the recurrent state barely forgot, per-step
error *could* accumulate. It does not. Stress run at `alog=-5` (g≈0.995, ~200-position effective
memory, long-memory heads):

| mode | rho @ P=16 | rho @ P=512 | rho @ P=1757 | growth |
|---|---:|---:|---:|:---:|
| Q8_0 | 8.25e-04 | 9.31e-04 | 9.31e-04 | **1.13×** (saturates by P=512) |
| Q4_K | 1.26e-02 | 1.38e-02 | 1.36e-02 | **1.08×** |

From short-memory (g≈0.88) to long-memory (g≈0.995) heads, quant-induced rho **saturates at a
length-invariant steady state** and does not compound. This is a property of the delta-rule scan
itself: `st = st·g + k⊗((v − k·st)·b)` is a *stable* fixed-point iteration (the `(v − k·st)`
correction is contractive), not an unbounded accumulator, so a fixed per-step weight perturbation
reaches a bounded relative floor rather than growing with the number of steps.

## What the number means

**Verdict: weight-quantization MAGNITUDE is INSUFFICIENT to explain #4273.** The collapse is defined
by **length-onset** (short fine, long broken), but the operative perturbation — Q8 weight-quant of
the GDN projections, present identically in both failing paths — produces a hidden divergence that is
**length-invariant** (flat from P=16 to the P≈1757 failure horizon) and, for the Q8 compute dtype,
~5× *below* the near-tie flip order. A length-invariant cause cannot produce a length-onset symptom.

So #4273 is **not** accumulated quant rounding in the GDN recurrence — it is an **algorithmic,
length-triggered** defect in the quantized long-context path: something that changes past a length
threshold (a chunked-prefill / sliding-window boundary, a KV- or conv-state index, position/rotary
handling, or an integer/window boundary), which a per-step precision model cannot exhibit. This
**redirects** the investigation away from "make the GDN state higher precision" and toward the
on-artifact **early token/logit comparison** that localizes the diverging `(layer, op)` at the length
where coherence breaks — see the decisive next witness below.

## The decisive next witness (on-device-gated, turnkey)

The one experiment that resolves this consumes the existing tooling. Capture fak's per-layer /
per-op hidden dump at a position **inside** the collapse region and diff it against llama.cpp/HF at
the same position, then let the probe find the first *anomalously* diverging op:

```bash
# fak side (on the box holding the artifact), tap the decode forward ~40 tokens into the repetition loop:
FAK_HIDDEN_TAP=/tmp/fak_dump FAK_HIDDEN_TAP_POS=<collapse_pos> FAK_HIDDEN_TAP_OPS=1 \
  fak run --model qwen3.6-27b-q4_k_m.gguf --prompt-file <the_1.3k_prompt> --max-tokens <past_collapse>
# llama.cpp side: eval-callback graph dump at the same absolute position -> /tmp/llama_dump
go run ./experiments/qwen36/token3-divergence-probe -fak /tmp/fak_dump -llama /tmp/llama_dump -auto
```

The `-auto` finder flags the first layer whose cosine drops **anomalously** below the agreeing-layer
noise floor — a real op mismatch, not rounding (this experiment is exactly what licenses the
anomaly-threshold rather than a 1-ULP floor). That names the `(layer, op)` #4273 lives in.

## What this does NOT do

It does not run llama.cpp, does not load the 27B artifact, and uses seeded-random (not trained)
weights — so it **cannot** reproduce the trained repetition attractor itself, only bound the
magnitude and length-slope of quant-induced recurrent divergence. Its claim is about **order of
magnitude and P-slope** — that weight-quant divergence is length-invariant and (for Q8) sub-threshold
— which the flatness across the full decay range and a ~2-order margin (Q8) make robust to the
modeling choices. The on-artifact `(layer, op)` localization remains the honest `not yet`.
