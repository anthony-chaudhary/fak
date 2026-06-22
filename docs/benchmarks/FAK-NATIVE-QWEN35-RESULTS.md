---
title: "fak in-kernel Qwen3.5/3.6 Gated-DeltaNet on M3"
description: "fak's own in-kernel forward pass runs the Qwen3.5/3.6 hybrid Gated-DeltaNet architecture end-to-end, witnessed from 0.8B safetensors to the 27B q4_k_m GGUF on Apple M3."
---

# fak's own engine runs Qwen3.6-27B (`qwen35`) — witnessed on M3

> The headline for the goal lane: fak's **own in-kernel forward pass** — not
> llama.cpp — now runs the **Qwen3.5/Qwen3.6 hybrid Gated-DeltaNet architecture**
> end-to-end in chat on Apple M3 Pro. The 0.8B safetensors run is the coherent
> f32 architecture witness; the **27B q4_k_m GGUF now also loads and generates
> through fak's pure in-kernel path** via the GGUF->Q8 cached GDN runtime.

## Witness (reproducible)

```
$ /tmp/fakchat -hf ~/.cache/fak-models/qwen3.5-0.8b \
    -p "What is the capital of France? Answer in one short sentence." -n 40
model=qwen3_5  load=1203ms  prompt_tokens=32  backend=fak in-kernel Gated-DeltaNet (f32, cacheless)
<think>

</think>

Paris is the capital of France.
```

Correct, coherent, and it even emits the Qwen `<think>` block. The full path is
fak's own: `internal/tokenizer` (Encode/Decode, oracle-validated) → ChatML template
→ `model.Forward` running the **Gated-DeltaNet linear-attention scan + gated
full-attention** (`internal/model/qwen35.go`, ported from transformers
`Qwen3_5GatedDeltaNet`) → sampling → detokenize → stream.

## 2026-06-19 cached-decode refresh

The original M3 chat witness above was the cacheless path. Current fak now routes the
Qwen3.5/Qwen3.6 f32 safetensors path through `Session.Prefill` / `Session.Step`:
full-attention KV and the Gated-DeltaNet recurrent conv/state both live in the session.
That makes `cmd/fakchat` and `cmd/qwen35check` use cached decode instead of rerunning
whole-sequence `Forward` for each generated token. Current unit witnesses are
`TestQwen35HybridSessionMatchesForwardAndPersistsState` and
`TestQwen35HybridQuantTokenLoopPersistsState`.

## 2026-06-19 Qwen3.6-27B pure-fak GGUF witness

This is the real 27B artifact on this M3 Pro, with no `llama-server`, no external
OpenAI-compatible proxy, and no llama.cpp in the execution path:

```sh
/usr/bin/time -l go -C fak run ./cmd/fakchat \
  -gguf /Users/USER/.cache/fak-models/gguf/Qwen3.6-27B.q4_k_m.gguf \
  -tok /Users/USER/.cache/fak-models/tokenizers/qwen3.6 \
  -p "Say OK." \
  -n 1
```

Observed output:

```text
model=qwen35  load=75505ms  prompt_tokens=22  backend=fak in-kernel Gated-DeltaNet (GGUF->Q8, cached)
<think>
---
prefill: 22 tok in 40.62s (0.5 tok/s)  |  cached qwen3_5 decode: 1 tok in 16.25s (0.1 tok/s)
      135.67 real       339.68 user        97.23 sys
         25785204736  maximum resident set size
```

What this proves:

- The real 15 GB `Qwen3.6-27B.q4_k_m.gguf` parses as `qwen35`, maps all 851 tensors,
  loads through `ggufload.LoadModelQuant`, and builds a runnable `*model.Model`.
- `cmd/fakchat` runs tokenizer -> ChatML ids -> `Session.Prefill` -> sample ->
  detokenize -> `Session.Step` entirely inside fak.
- The 27B model fits this 36 GB M3 Pro in the current GGUF->Q8 runtime, peaking at
  about 25.8 GB RSS.
- The first generated token matches llama.cpp b9707 greedy raw-ChatML output for the
  same 22-token prompt: token id `248068`, `<think>`.
- The #93 oracle has now been extended beyond one token. llama.cpp b9707 returns
  `[248068, 198, 90700]` (`<think>\nThinking`) for the same prompt; fak's current
  GGUF->Q8 path returns `[248068, 198, 8160]`, so the first two tokens match and the
  third token is the current real-artifact divergence. Artifacts:
  `experiments/qwen36/llamacpp-qwen36-multitoken-oracle-20260619.json` and
  `experiments/qwen36/native-gguf-q8-multitoken-parity-20260619.json`.

What this does **not** claim yet:

- It is not a llama.cpp speed parity result. The current pure-fak path is an
  unoptimized CPU/Q8 reference for this architecture and is about 0.1-0.5 tok/s here.
- It is not a full-logit or multi-token parity claim yet. The same 22-token ChatML
  prompt tokenizes byte-exactly against llama.cpp and now matches the first two
  llama.cpp greedy tokens, but the three-token #93 oracle fails at step 2 (`8160`
  instead of llama.cpp's `90700`).

## How it works today

- **Arch dispatch**: `cfg.IsQwen35Hybrid()` (from `layer_types`) selects the hybrid
  forward. 24 layers for 0.8B; the 27B is the same op set at 64 layers (48 GDN + 16
  full-attn).
- **Linear-attention (GDN)**: causal depthwise conv1d + SiLU, q/k L2-norm + query
  scale, recurrent gated delta rule (decay `exp(-exp(A)·softplus(a+dt_bias))`,
  sigmoid β, rank-1 state update), per-head gated RMSNorm, out-proj.
- **Full-attention**: split per-head `[query|gate]`, sigmoid output gate, partial
  RoPE (rotary_dim < head_dim), qk-norm.
- **Decode is cached for the f32 safetensors path**: `Session.Prefill` builds the
  recurrent GDN state and full-attention KV once; `Session.Step` advances both. Cache
  eviction for GDN state is deliberately fail-loud until the quarantine semantics are
  implemented for recurrent state.

## The 27B-size status on a 36 GB box (precise)

The f32 path is still too large for 36 GB, but the GGUF->Q8 path now runs:

| Path | Footprint | Fits 36 GB? | Missing |
|---|---|---|---|
| f32 (`LoadSafetensorsDir`, the validated GDN path) | 27B×4 ≈ **108 GB** | ✗ | — |
| GGUF->Q8 cached runtime | observed **25.8 GB RSS** | ✓ | speed/broader logit-parity work |
| native q4 GGUF runtime | ≈ 16 GB | ✓ in principle | direct q4 kernels / no Q8 expansion |

The size gate is closed for an end-to-end command-line smoke. The remaining gap is
performance/correctness evidence at the real-artifact level: load-time reduction (#95),
direct q4 residency (#96), GDN/full-attention phase profiling and acceleration (#97),
device prefill for the GDN/full-attention projections (#92), and a short llama.cpp/HF
oracle to prove logits rather than just execution (#93).

## Status summary

- ✅ Qwen3.6-27B runs end-to-end in chat **on this setup** in llama.cpp b9707 Metal —
  the speed/quality parity bar.
- ✅ Qwen3.6-27B now also runs end-to-end in chat through **fak's own in-kernel engine**
  using the local GGUF->Q8 cached GDN path.
- ✅ The real 27B GGUF path now matches llama.cpp's greedy first token on the exact
  ChatML smoke prompt (`<think>`, id `248068`).
- ✅ The Qwen3.6 **architecture** runs end-to-end in chat in **fak's own engine**
  (0.8B witness above).
- ✅ The f32 hybrid session path now has cached prefill/step decode witnesses, so
  standalone test benches no longer exercise only the cacheless path.
- ⏳ Recurrent-state eviction/quarantine, direct q4/chunked/device GDN performance work,
  and broader real-artifact llama.cpp/HF logit witnesses remain open.

For DGX and standalone endpoint-backed test benches, use a multi-GPU A100 serving host.

_Witnessed 2026-06-18 and refreshed 2026-06-19 on Apple M3 Pro (36 GB). fak rows are fak's own forward pass._
