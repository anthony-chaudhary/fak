---
title: "GLM-5.2 on the pure fak engine: a real glm_moe_dsa checkpoint generates tokens that match HuggingFace, witnessed on an 8-GPU datacenter server node (2026-06-23)"
description: "Fresh on-hardware capture on an idle 8-GPU datacenter server-80GB (sm_80) datacenter node: fak loads a REAL glm_moe_dsa checkpoint (yujiepan/glm-5-tiny-random) and its forward is argmax-exact vs HuggingFace while Session.Generate matches HF greedy decode token-for-token; the full GLM-5.2 DSA forward runs bit-exact (cosine=1.0) on fak's own CUDA kernels; and SmolLM2-135M decodes at 131 tok/s through the pure k_q8_gemm path (zero cuBLAS). Also fixes an over-strict DSA top-k trace assertion (compare the selected SET, not the implementation-defined tie ORDER). The honest boundary — fak serving the full 753B natively — is unchanged: still the labeled multi-month gap."
---

# GLM-5.2 pure fak, end to end: real-checkpoint generation matches HuggingFace (2026-06-23)

> **Goal:** show GLM-5.2 working *pure fak, end to end* on the datacenter GPU server — not
> just that the kernels are bit-exact, but that fak's **own** engine loads a **real**
> glm_moe_dsa checkpoint and **generates** tokens that match the HuggingFace reference. This
> note records the on-device witnesses (grounded in `go test` exit codes and run logs, not
> self-report) and is explicit about the one boundary that is still open.

> **Superseded by progress (#917; see the [staged plan](native-753b-track-staged-plan.md)).**
> The §5 "honest boundary" below — "not fak serving the full 753B natively … the full 753B
> still serves only via the llama.cpp CPU-offload baseline" — was the 2026-06-23 snapshot.
> The quantized device GEMM + `--cpu-offload-experts` + `glm_moe_dsa` loader rungs have since
> landed and fak's **own** engine loads the full 466 GB model natively and binds `/v1/*`
> ([2026-06-25 native-serve note](GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25.md)). The
> real-oracle, bit-exact DSA-forward, and SmolLM2 decode witnesses (§1–3) stand; only the
> serving boundary moved.

All runs are on a fresh, idle **8× NVIDIA datacenter GPU (sm_80, compute 8.0), CUDA 12.8,
~2 TB host RAM** datacenter node, at the then-current `origin/main`, Go 1.26.4.

## 1. A real glm_moe_dsa checkpoint generates tokens that match HuggingFace

`export_oracle.py` exported the real **`yujiepan/glm-5-tiny-random`** glm_moe_dsa checkpoint
(40 tensors, 3.4M params, 13.6 MB f32) via HF transformers 5.6.0, and fak loaded it through
its own `Load()` path. The witnesses (`go test -run GLMMoeDsaOracle ./internal/model/`, all
PASS):

- **Forward matches HF, argmax-exact** (`...OracleForwardMatchesHFCacheless`): every prompt's
  final-token logits agree with HF (`max|Δ|≈1–8e-5`), **argmax identical** (`111258`, `13438`,
  `13438`); per-layer hidden states `cosine=1.000000`.
- **Incremental generation matches HF greedy** (`...OracleSessionCacheMatchesHF`):
  Prefill/Step, PrefillNoLogits/Step, and SessionFromPrefix all reproduce the cacheless
  forward, **`Session.Generate` reproduces HF's greedy token ids**, and the tail-eviction
  path reproduces HF's "never saw the poison" continuation.
- **DSA attention trace matches HF** (`...OracleReproducesDSAAttentionTrace`,
  `...ReproducesDensePrefixLayer`, `...DocumentsDSABoundary`): the selected key set and the
  attention output reproduce HF's forward hooks (`cosine=1.000000`, `max|Δ|<3.1e-4`).

This is the "real architecture, real weights, real generation, matches the reference" rung —
**pure fak, no llama.cpp, no upstream** — on a real datacenter GPU node. It is a *tiny random*
checkpoint (the only public glm_moe_dsa artifact small enough to export), so it proves the
**pipeline and the math**, not flagship answer quality.

## 2. The full GLM-5.2 DSA forward is bit-exact on fak's own CUDA kernels

Re-witnessed on this idle node (`private GPU witness runner`, three isolated `-tags cuda`
tests, all rc=0):

- full GLM-MoE-DSA forward — MoE/FFN + router + head on `k_q8_gemm`, the DSA attention dense
  projections on `k_q8_gemm`, and the DSA sparse-attention compute on `k_dsa_sparse_attend` —
  **cosine=1.000000, argmax-exact** vs the all-host CPU Q8 reference, `tier=sm_80`;
- the `--n-cpu-moe` CPU-offload hybrid (experts host-resident, dense + DSA on the GPU) —
  **cosine=1.000000, argmax-exact**;
- the indexer score + top-k **selection** on the device (`k_dsa_index_score` +
  `k_dsa_index_topk`) — **cosine=1.000000, argmax-exact**.

## 3. Real tokens through the pure kernel: SmolLM2-135M at 131 tok/s

To show fak's own GPU kernel *generating* real tokens end-to-end (the closest real-model
decode that loads today), `private pure-kernel benchmark runner` decoded SmolLM2-135M (a Llama
family model) on the datacenter GPU through the pure `k_q8_gemm` path, **zero cuBLAS**: **decode 7.6
ms/tok = 131.0 tok/s** (prefill 107–166 tok/s), `engine: fak-in-kernel via compute HAL
backend "cuda"`, `tier=sm_80`.

## 4. Fix: compare the DSA top-k *selection set*, not the tie *order*

`TestOptionalGLMMoeDsaOracleReproducesDSAAttentionTrace` had asserted fak's indexer top-k
in the exact element order HF returned. fak breaks equal-score ties by ascending key position
(`dsaTopKIndices`); HF's `torch.topk` uses an implementation-defined tie order, so on a score
tie the two return the **same keys in a different order**. DSA attention over the selected
keys is order-invariant (softmax + ΣwV), and the numeric attention-output trace
(`cosine=1.000000`) is the real correctness guard — so the assertion now compares the selected
**set** (`sortedInts`). A genuine divergence that selected a *different* key still fails.

## 5. The honest boundary (unchanged)

This is fak's **own** engine doing GLM-5.2 end to end at the achievable scale today. It is
**not** fak serving the full **753B** weights natively — that remains the labeled multi-month
gap: the native device GEMM is f32-output, there is no multi-GPU NCCL/tensor-parallel, and the
GGUF loader has no `glm_moe_dsa` config path (the real export's per-layer `indexer_types` issue,
#1418). The full 753B still serves only via the llama.cpp CPU-offload comparison baseline
(~2.66 tok/s), explicitly **not** fak. See
[`GLM52-DSA-FULL-FORWARD-ON-PURE-KERNEL-GPU-SERVER-2026-06-23.md`](GLM52-DSA-FULL-FORWARD-ON-PURE-KERNEL-GPU-SERVER-2026-06-23.md)
and [`GLM52-PURE-KERNEL-ON-GPU-SERVER-2026-06-21.md`](GLM52-PURE-KERNEL-ON-GPU-SERVER-2026-06-21.md).
