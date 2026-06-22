---
title: "fak in-kernel model: forward pass proven against HF"
description: "A kernel-owned in-process CPU forward pass for SmolLM2-135M, proven rung-by-rung bit-for-bit against HuggingFace transformers including KV cache and quarantine."
---

# IN-KERNEL-MODEL-RESULTS — the model fused INTO the kernel, proven against HF

> The 72h build plan (`../BUILD-72h-fused-agent-kernel.md`) deliberately punted on
> the three deepest fusions — a local in-process engine, zero-copy co-residence of
> tool state with the KV cache, and a syscall-tuned model — calling them out of
> reach because "vLLM owns the KV cache in Python/CUDA" and "no GPU on the box".
> That scoping was about **production GPU serving**. This lane crosses the line a
> different way: a **correct, single-model, CPU, in-process forward pass that the
> kernel OWNS**, so the KV cache is a kernel data structure — not a production
> engine, a **reference** one ("like an nvidia reference cloud"), whose purpose is
> to understand the real challenges at every level of the stack and to make the
> context-MMU / vDSO claims **real operations on real attention state** instead of
> metaphors over an HTTP boundary.
>
> Nothing here is asserted; every rung is **proven against HuggingFace transformers**
> — the witness we did not author. The oracle is dumped layer-by-layer, so a bug in
> any rung is localized, not hidden.

## The target model

`HuggingFaceTB/SmolLM2-135M-Instruct` — a clean Llama-architecture model small
enough that a pure-Go CPU forward pass is tractable and its numerics are checkable
end-to-end:

| | |
|---|---|
| params (exported f32) | **134,515,008** across **272 tensors** (538.1 MB) |
| arch | Llama: 30 layers, hidden 576, 9 query / 3 KV heads (GQA), head_dim 64 |
| MLP | SwiGLU, intermediate 1536 | 
| norm / pos | RMSNorm (eps 1e-5) · RoPE θ=100000, rotate-half (non-interleaved) |
| head | tied to the input embedding · vocab 49152 |

Swapping the target is a **re-export, not a code edit** — the Go core reads the arch
from the exported `config.json`; Qwen2.5-0.5B (which adds QKV bias) is the next
target and the code already branches on `attention_bias`.

## The provable rung ladder (each rung's witness is HF, not us)

| Rung | What runs in the kernel | Witness (HF-authored) | Measured result |
|---|---|---|---|
| **R0** | eager+version-pinned export → flat f32 + manifest; per-layer oracle | param count + per-tensor shapes match HF | 272 tensors / 134.5M params ✓ |
| **R1a** | **pure-Go safetensors reader + bf16→f32 decode** | bitwise == torch's decode (lossless 16-bit widening) | **0 / 134,515,008 param bit-mismatches** — weights are Go-authored, not torch-decoded |
| **R1** | pure-Go forward pass (RMSNorm·RoPE·GQA·SwiGLU·tied head) | HF `output_hidden_states` + logits, per layer | embedding **exact (0.0)**; every checked layer **cos = 1.000000**; final logits **max\|Δ\| ≈ 4.4e-5**; **argmax exact at every position, 3/3 prompts** |
| **R2a** | kernel-owned KV cache, incremental decode | must equal the R1-verified full prefill | last-token logits **max\|Δ\| = 0.000e+00** (bit-identical), 3/3 |
| **R2b** | greedy generation off the in-kernel forward + KV cache | HF greedy continuation | **token-for-token identical, 3/3 prompts × 12 tokens**; also reproduced from the **Go-decoded** weights |
| **R3** | **KV-level write-time quarantine**: evict the poisoned span's K/V | HF greedy of the **never-saw-poison** sequence | **token-for-token identical**; reposition invariant `K==RoPE(Kraw,pos)` **bit-exact (0.0)** |
| **R14** | **KV-prefix reuse** (vDSO payoff): clone a computed prefix, prefill only the suffix | the (R2-verified) full recompute | last-logit **max\|Δ\| = 0.000e+00**, greedy identical, prefix prefill **skipped** |

R3 carries three more witnesses beyond the headline:
- a **non-vacuous negative control** — a Go run that keeps the poison reproduces HF's
  *poisoned* continuation token-for-token **and differs from never** (the poison
  genuinely perturbs generation, so the guarantee is non-trivial);
- a **bit-exact reposition invariant** — every survivor's post-RoPE K equals a single
  rotation of its pre-RoPE `Kraw` at its new position (composing two rotations would
  drift ~1e-6 and flip a greedy token; the pre-RoPE store makes it exact);
- the **boundary** (the deepest finding) — a span evicted *after* downstream tokens
  already attended to it is **not** un-seen: those tokens' hidden states / cached V
  already absorbed it, so middle-span evict ≠ never. This is *why* quarantine must be
  **write-time** (before the model's next turn attends), exactly what `ctxmmu`'s Admit
  gate does.

This is exactly the witness build-plan landmine #7 demands: not "mark X poison, assert
X absent" (which proves nothing), but "evict X, assert output == the run that never
saw X, assert the un-evicted run differs, **and** prove the one thing eviction
cannot do."

## The fusion payoff: quarantine becomes a KV-level guarantee, not a string filter

The shipped `internal/ctxmmu` quarantine rewrites a poisoned tool **result's bytes**
to a stub. That is a *content* defense: it depends on the bytes still being
filterable, and once any poisoned text has entered the model's context window the
attention state already encodes it.

With the model **inside the kernel**, quarantine becomes a *mechanical* defense:

```
prefill[prefix] → prefill[poison] → cache.Evict(prefix_len, poison_len) → prefill[query] → decode
        positions 0..P-1   P..P+Q-1        (removed, survivors renumbered)     P..             P+..
```

`KVCache.Evict` drops the span's K/V from **every layer** and renumbers the
survivors so the next token lands at the same absolute RoPE position it would have
had if the poison were never seen. Because the prefix's K/V are causally upstream of
the poison (unchanged by it) and the query lands at identical positions, the evicted
run is **provably identical** to the never-saw-it run — and R3 confirms it equals
HF's reference output, byte for byte. The model **physically cannot attend to the
evicted span**; there is no residual to filter.

**This only works at write-time.** Eviction is clean precisely because `ctxmmu`'s
`Admit()` fires the *moment a poisoned result is produced* — before the model's next
turn attends to it. A span evicted *after* downstream tokens already attended to it
cannot be un-seen: those tokens' hidden states and cached V already absorbed it (the
test proves the post-evict cache is bit-identical to a clean prefill at layer 0 yet
diverges at deep layers, because the survivors' `Kraw` already encodes the poison).
That boundary is the mechanical justification for the write-time Admit gate.

This is the seam to `internal/ctxmmu`: today `Admit()` returns `Quarantine` and
pages bytes out; the bridge is to have that same verdict call `Session.Cache.Evict`
on the span the poisoned result occupied. The primitive is proven here; wiring it as
the **live agent engine** needs a Go tokenizer + chat template (see scope below).

## What this honestly is NOT (labeled, so it isn't vaporware)

- **NOT a production serving engine.** Naive triple-loop CPU matmul: correct, not
  fast. No continuous batching, no paged-attention, no GPU kernels, no throughput
  claim. Those stay **modular** — production points fak at vLLM/SGLang. The value
  here is the **reference**: owning every level so the kernel's operations on KV are
  real and measurable.
- **NOT yet the live agent loop.** `fak agent` still drives a model over the
  OpenAI-compatible HTTP seam. Replacing that engine with this in-process core needs
  a Go byte-level BPE tokenizer + the chat template (the next rung). The forward
  pass, KV cache, decode, and KV-quarantine are done and proven; the
  text↔token boundary is not yet ported.
- **NOT a tuned model.** This is the stock SmolLM2-135M weights, unmodified. The
  "syscall-tuned small model" remains future work; what is proven is the
  *substrate* that such tuning would target.

Weights are now **Go-authored** (R1a: a pure-Go safetensors reader + bf16→f32 decode,
witnessed bitwise vs torch), so the only thing torch does is author the *oracle* — the
independent witness the Go core is checked against, which is the point. The full
every-level-of-the-stack map (built vs designed, the landmine list, the payoff map)
is in `IN-KERNEL-MODEL-DESIGN.md`.

## Reproduce

```
# 1. export weights + dump the HF reference oracle (needs torch+transformers, CPU)
python internal/model/export_oracle.py --model HuggingFaceTB/SmolLM2-135M-Instruct \
       --out internal/model/.cache/smollm2-135m

# 2. prove every rung against HF (skips cleanly if weights absent)
go test ./internal/model/ -v
```

The 538 MB f32 export + oracle live under `internal/model/.cache/` (gitignored,
regenerable). The Go tests `t.Skip` when it is absent, so CI without weights stays
green; locally the oracle is the real, non-forgeable witness.

## Bottom line

A real open-source model now runs **inside the fak kernel's address space** on CPU,
with the **KV cache as a kernel-owned Go data structure**, and every step — the
forward pass, the cache, greedy decode, and KV-level quarantine — is proven equal to
HuggingFace transformers to f32 tolerance or token-for-token. The deepest fusion the
build plan called out of reach is reachable as a **reference**; what stays out of
scope (GPU serving throughput, the live tokenizer wire-up, a tuned model) is labeled,
not hidden.
