---
title: "Qwen3.6-27B Parity Bar on Apple M3 Pro"
description: "The witnessed llama.cpp Metal reference for Qwen3.6-27B on M3 Pro, the speed bar fak's own engine targets plus first-token greedy parity."
---

# Qwen3.6-27B on Apple M3 Pro — the llama.cpp parity bar (witnessed)

> Goal lane: *"complete and prove out working end to end in chat Qwen3.6-27B on
> this gpu/cpu setup; reach performance parity with llama.cpp (if it can even run
> it); hybrid cpu/gpu."* This doc records the **llama.cpp reference** — the answer
> to "can it even run it" and the speed bar fak's own engine targets. As of
> 2026-06-19, fak also has a pure in-kernel Qwen3.6-27B GGUF->Q8 smoke with
> first-token greedy parity; that row is still not a speed-parity claim.

## Hardware

Apple **M3 Pro**, 12 CPU cores (6P+6E), **18-core GPU**, **36 GB** unified memory,
Metal 4. Native `go1.26` (GOTOOLCHAIN=auto). darwin/arm64.

## The model is real, new, and architecturally radical

`Qwen/Qwen3.6-27B` (HF `config.json`: `Qwen3_5ForConditionalGeneration`,
`model_type: qwen3_5`; GGUF arch string **`qwen35`**). It is **not** a standard
transformer:

- **64 layers**, `full_attention_interval=4` → **48 `linear_attention` (Gated
  DeltaNet) + 16 `full_attention`** layers.
- hidden 5120, intermediate 17408 (SwiGLU), rms_norm_eps 1e-6, vocab **248320**,
  `tie_word_embeddings=false`.
- **Full attention**: 24 Q / 4 KV heads, head_dim 256, `attn_output_gate=true`
  (output-gated), **partial RoPE 0.25**, **mRoPE** (`mrope_section [11,11,10]`,
  interleaved), θ=1e7.
- **Linear attention (Gated DeltaNet)**: `linear_conv_kernel_dim=4`, 16 key heads
  / 48 value heads × 128 dim, `output_gate_type="swish"`, `mamba_ssm_dtype=float32`
  (recurrent state).
- MTP head (`mtp_num_hidden_layers=1`); vision tower (depth 27) — skipped for text.

## Can it even run it? — YES, on llama.cpp b9707 (Metal)

The GDN operators are brand-new; the stock Homebrew llama.cpp (**b8200**) has **no**
`gated_delta_net` / `ssm_conv` kernels and cannot load `qwen35`. The current release
**b9707** ships them (`ggml_metal_kargs_gated_delta_net`, `ssm_conv`,
`src/models/qwen3next.cpp`, `llama-memory-recurrent.cpp` in `libggml`/`libllama`).
Installed locally (no sudo) at `~/.local/llamacpp-b9707/`.

Quant: `Qwen3.6-27B.q4_k_m.gguf` (15.4 GB) from `AaryanK/Qwen3.6-27B-GGUF`.

**It generates coherent chat.** Prompt (ChatML): *"In one short paragraph, explain
what makes the Apple M3 Pro good for local LLM inference."* → model opened a
`<think>` block and correctly began reasoning about unified memory architecture.

## Measured results (q4_k_m, 36 GB, this M3 Pro)

| Backend | Prefill (tok/s) | Decode (tok/s) | Peak RSS |
|---|---|---|---|
| **Metal**, full offload (`-ngl 99`) | **51.55** | **7.29** | ~24.5 GB |
| **CPU** only (`-ngl 0`, `-t 6`)      | 20.12       | 6.48       | 24.5 GB |
| **fak in-kernel GGUF->Q8**, cached GDN | 0.5 | 0.1 | 25.8 GB |
| Metal speedup                        | **2.56×**   | **1.13×**  | — |

(llama.cpp `common_perf_print`; Metal: load 0.98 s. CPU peak RSS from
`/usr/bin/time -l`. fak row: `cmd/fakchat --gguf ... --tokenizer ... --prompt "Say OK." --max-new 1`,
load 75.51 s, prefill 22 tokens in 40.62 s, one cached decode token in 16.25 s,
peak RSS 25,785,204,736 bytes.)

## Which fak Qwen3.6-27B decode number is which (reconciliation)

Several fak Qwen3.6-27B decode numbers circulate, and a reader landing on any one
doc cannot tell which is "the" number. This is the single table that reconciles
them: they are **different paths on different hardware**, not three rival claims
for the same configuration. Every row is **single-stream, batch=1, on this M3
Pro**, and each value is labelled with where it is cited.

| fak decode path | Decode (tok/s) | What it measures | Cited at (path) |
|---|---|---|---|
| **GGUF→Q8 cached** (CPU) | **0.1** | one cached decode token through the GGUF→Q8 round-trip path | the **Measured results** table above; `BENCHMARK-AUTHORITY.md` ("Qwen3.6-27B Q8 decode"); `docs/benchmarks/FAK-NATIVE-QWEN35-RESULTS.md` |
| **resident-q4k microbench** (CPU) | **0.9** | raw q4_k blocks stay resident (no Q8 round-trip), scalar-f32 GEMV — the model-ladder rung-4b refresh | the *fak-native status* bullet below ("decode **0.9 tok/s** … ~9× the Q8 path's 0.1"); artifact `experiments/model-ladder/qwen36-resident-q4k-parity-20260619.json` (`"decode_tok_s": 0.9`); `docs/notes/MACBOOK-SERVE-AND-AGENTIC-BENCH-2026-06-24.md` |
| **resident-Q4_K Metal** (GPU, `-tags fakmetal FAK_Q4K=1 FAK_METAL=1`) | **1.2** | the int8-SDOT Metal decode GEMV path; bit-correct (GEMV cosine 1.0) but launch-bound (~336 command-buffer GEMVs/token) | `BENCHMARK-AUTHORITY.md` ("Qwen3.6-27B fak **Metal Q4_K**"); `docs/notes/MAC-QWEN36-27B-Q4K-METAL-PERF-DIAGNOSIS-2026-06-26.md` |

So **"fak Qwen3.6-27B decode" is 0.1 → 0.9 → 1.2 tok/s** along the GGUF→Q8-cached →
resident-q4k-CPU → resident-Q4_K-Metal progression — one M3 Pro, all single-stream.
It is a measured progression toward the 7.29 tok/s llama.cpp-Metal bar, not a
self-contradiction.

**Different axis — served throughput (NOT the kernel rows above).** fak's 8-GPU
*served* Qwen3.6-27B figures (single-stream ≈59–93 tok/s, batched peak ≈820–1085
completion tok/s) come from **SGLang-serves + fak-adjudicates on 8×A100/DGX**, not
fak's own M3 Pro engine; they live in `BENCHMARK-AUTHORITY.md` (the two "8-GPU
SGLang serving" rows) and `docs/benchmarks/QWEN36-27B-GPU-SERVER-RESULTS.md`. Do
not quote them on the same line as the single-stream kernel rows above.

**On the "1.6–1.8 tok/s served-warm" figure.** An earlier migrated note cited a
"Warm decode ≈ 1.6–1.8 tok/s" from a `FAK-NATIVE-CHAT-RESULTS.md` that **predates
the v0.30.0 public squash and is not tracked in this tree**; no committed artifact
reproduces 1.6–1.8 tok/s for this model. It is therefore **not asserted here** —
the witnessed single-stream figures are the three rows above.

## What the numbers say — and why "hybrid CPU/GPU" is the right design

- **Decode is bandwidth-bound**: Metal beats CPU by only 1.13×. A 27B-q4 model
  (~15 GB of weights streamed per token) saturates memory bandwidth; the GPU can't
  help much. The 48 Gated-DeltaNet layers keep the per-token compute low (linear
  attention has a small recurrent state, not a growing KV scan), so decode stays
  cheap *and* bandwidth-pinned.
- **Prefill loves the GPU**: 2.56× — prefill is compute-bound (parallel GEMMs over
  the whole prompt), exactly where the 18-core GPU pays off.
- This is the empirical basis for fak's existing hybrid split (**Metal prefill +
  NEON/CPU decode**): put the compute-bound phase on the GPU, leave the
  bandwidth-bound phase on the CPU. The 27B reference confirms the split is correct
  at scale, not just on small models.

## fak-native status against this bar

- **Tokenizer**: landed + cross-validated byte-exact vs this same llama.cpp build
  on the Qwen vocab (`internal/tokenizer`, oracle gate, #90).
- **Pure-fak runnability**: landed. `internal/model/qwen35.go` implements the
  Gated-DeltaNet linear-attention layers plus recurrent session state, full-attention
  output gating, qk-norm, partial RoPE, and per-layer dispatch. `internal/ggufload`
  maps the real qwen35 GGUF tensors, and `cmd/fakchat` now loads
  `Qwen3.6-27B.q4_k_m.gguf` through fak's GGUF->Q8 path and emits a token with no
  external model server. See `FAK-NATIVE-QWEN35-RESULTS.md`.
- **CPU reference, bit-exact vs HF**: landed (#447). A tiny text-only `qwen3_5`
  fixture (`internal/model/make_qwen35_tiny.py`: 3 Gated-DeltaNet + 1 gated
  full-attention layer) is the f32 witness the pure-Go forward is proven against —
  `TestOptionalQwen35HybridOracleForwardMatchesHF` reproduces HF transformers'
  per-layer hidden states (cosine 1.000000, max|Δ| ~4e-9) and argmax at every
  position. This confirms the hybrid arch math — Gated DeltaNet recurrence, the
  sigmoid output gate, per-head qk-norm, partial RoPE, and the (1+w) RMSNorm — is
  numerically correct, so the 27B token-3 drift below is a **scale / kernel-numerics
  / quant** divergence, not a bug in the reference path. Runs on a plain CPU box
  (transformers>=5.10), no GPU/27B artifact node needed.
- **Remaining bar**: fak is not yet speed-parity with llama.cpp on the 27B artifact.
  The broader #93 real-artifact oracle now exists: llama.cpp b9707 returns
  `[248068, 198, 90700]` (`<think>\nThinking`) for the exact 22-token ChatML prompt,
  while fak's current GGUF->Q8 path returns `[248068, 198, 8160]`. That preserves the
  first-token claim, adds a second-token match, and gives the current third-token
  correctness failure. Direct-q4/chunked/device GDN kernels are the next speed gates
  after this correctness gap is understood.
- **Token-3 drift RE-DIAGNOSED 2026-06-19 (model-ladder rung 4b):** the resident-q4k
  path (which skips the GGUF→Q8 round-trip) was tested against the same oracle and
  **also drifts at token 3** — `<think>\nHere's a thinking` (`fak`) vs `<think>\nThinking`
  (`[248068,198,90700]` llama.cpp), 2-token match then divergence. This **disproves**
  the earlier "Q8-round-trip quant artifact" hypothesis: the drift survives the move
  to native q4_k weights on BOTH engines. Combined with rung 3 (Qwen3.5-0.8B f32 is
  semantically correct through the same arch path), the drift is a **kernel numerics
  divergence at 27B scale on the hybrid GDN path** — most likely small float
  differences compounding through the GDN recurrent state / mRoPE / partial-RoPE
  handling. fak and llama.cpp agree for 2 tokens then diverge, the signature of
  compounding float error in a recurrent path. Artifact:
  `experiments/model-ladder/qwen36-resident-q4k-parity-20260619.json`. Resident-q4k
  perf at this rung: load 36 s, decode **0.9 tok/s** (~9× the Q8 path's 0.1, ~2×
  faster load) — real progress, still ~8× under the 7.29 tok/s bar.
- **Optimization tickets**: #95 tracks load time/page churn, #96 tracks native q4/q6
  residency, and #97 tracks first-token phase profiling before the #92 kernel work.
- **Reproduce**:
  ```sh
  LL=~/.local/llamacpp-b9707/llama-b9707
  GG=~/.cache/fak-models/gguf/Qwen3.6-27B.q4_k_m.gguf
  DYLD_LIBRARY_PATH=$LL $LL/llama-completion -m $GG -ngl 99 -t 6 -c 4096 -n 96 \
    -p $'<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n'

  go run ./cmd/fakchat --gguf $GG --tokenizer ~/.cache/fak-models/tokenizers/qwen3.6 \
    -p "Say OK." -n 1
  ```

_Witnessed 2026-06-18 on the M3 Pro; fak row refreshed 2026-06-19. llama.cpp numbers are llama.cpp's own perf counters._
