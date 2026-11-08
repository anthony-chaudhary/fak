---
title: "Kimi K3.1 text stack: SGLang direct-port study and subsystem order"
description: "Deep study of SGLang's landed K3/K3.1 stack vs fak GLM5Next geometry; port order, reuse map, risks."
---

# Kimi K3.1 text stack: SGLang direct-port study and subsystem order

- **Observed:** 2026-11-06 (UTC)
- **Objective:** direct-port the largest existing OSS subsystems of Kimi K3.1 (moonshotai/Kimi-K3.1-Pro, HF `model_type kimi_k31`, `KimiLinearForCausalLM` text stack) into fak, one subsystem at a time largest-first, validated on the external 8xH200 node.
- **Port source (field prior art):** SGLang `python/sglang/srt/models/kimi_k3.py` (4010 lines) + `python/sglang/srt/configs/kimi_linear.py` + HF `config.json` — Apache-2.0, DIRECT-PORT class. Basic K3.1 support merged in SGLang #48415; umbrella #48414 open.
- **Reuse target:** fak's GLM5Next subsystem (43 files: KDA conv/decay/step/state/forward, DSA indexer, MoE router, spine, cache, cadence) — closest in-repo geometry, MBL-060 layout-predicate pattern.
- **Method:** 7-reader parallel study (SGLang model file, config, HF config, fak GLM5Next KDA/DSA/MoE/spine, compute kernels, loader) with file:line citations; 16-axis geometry diff; synthesis ranked the port order and risks.

## K3.1 pinned geometry (HF config, witnessed 2026-11-06)

93 layers = 69 KDA (linear attention) + 24 full-attn (MLA+DSA); hidden 7168, 96 heads;
q_lora 1536, kv_lora 512, qk_nope 128, qk_rope 64, v_head 128; KDA head_dim 128, conv 4;
896 routed experts (top-16) + 2 shared, routed experts in 3584-dim latent space;
situ activation (beta 4.0, linear_beta 25.0); attn_res_block_size 12; DSA indexer 32 heads x 128 dim,
topk 2048 on every MLA layer; gate_lower_bound -5.0; quantization: bf16 + mxfp4-pack-quantized
routed experts only (compressed-tensors, group 32, uint8 e8m0 scales).

Layer indices: `kda_layers`/`full_attn_layers`/`dsa_layers` config values are ONE-BASED (convert
idx+1). 0-based full-attn = {3,7,...,91,92} — the final layer 92 breaks the period-4 pattern, so a
`%4==3` predicate alone is WRONG for K3.1.

## The 16-axis geometry diff (K3.1 vs fak GLM5Next)

REUSE (identical or config-parameterized in fak today):
1. KDA head_dim 128 + conv window 4 — match GLM pins exactly; heads (96) is a runtime push-constant.
2. DSA indexer dims (32 heads x 128, topk 2048) — identical; `glm_dsa.go` already reads
   q_lora/kv_lora/qk_nope/qk_rope/v_head from cfg; K3.1's 1536/512/128/64/128 thread straight through.
3. KDA recurrence family — fak's `StepGLM5NextKDAHead` is the same delta-rule recurrence; the
   difference is the decay/gate parameterization (axis 4), not the recurrence.

EXTEND (fak primitive exists, K3.1 parameterization differs):
4. KDA decay: K3.1 uses per-(head,dim) ELEMENTWISE decay along Dk vs GLM's per-head scalar alpha;
   gate = full-rank output gate in fused_qkvg + low-rank forget gate f_a->f_b + SEPARATE b_proj beta
   + A_log/dt_bias — this is exactly fak's existing-but-unwired `BoundedAsymmetricForget`
   (internal/compute/bounded_forget.go:9-25, exp(-5*sigmoid(exp(a_log)*(f_proj+dt_bias))),
   from ds4_rocm_glm5_kda.cuh:145-160) with gate_lower_bound -5.0.
5. Router: `RouteGLM5NextMoE` implements sigmoid + top-k + renorm only; K3.1 requires noaux_tc
   (sigmoid scoring with bias-ranked ungrouped top-k via e_score_correction_bias) — parsed in fak's
   GLM config but NOT implemented. routed_scaling_factor 1.0 makes scaling a no-op; bias ranking is
   load-bearing for expert selection.
6. MoE slotstream plan: re-parameterize to 896 experts / 92 MoE layers / new expert chunk bytes.

NEW (no fak primitive; port from SGLang):
7. Latent MoE: routed experts compute in the 3584-dim latent; the TP all-reduce completes in latent
   space BEFORE the latent RMSNorm (sum(norm(x_i)) != norm(sum(x_i)) — ordering is load-bearing);
   latent-MoE layers own the delayed residual add.
8. Situ activation: SituAndMul (beta 4.0, linear_beta 25.0) — formula lives in SGLang
   `ops/activation/kimi_k3_situ_and_mul.py`, not in the studied model file; must be fetched and
   cross-checked before any expert executes. Used by dense MLP layer 0, shared experts, routed experts.
9. Attention-residual bank: replaces the standard residual stream with a per-forward bank of block
   residuals mixed by learned per-token scalar weights (get_cw); a port keeping add-and-norm will
   NOT match checkpoints. 4 small params per layer (2x RMSNorm + 2x hidden->1 unquantized).
10. MLA output gate: sigmoid gate before o_proj (kimi_k3.py:2325-2393 semantics).
11. Config spine: separate exact-equality gate `isKimiK31Config` (never relax isExactGLM5NextConfig);
    IsQwen35Hybrid must explicitly exclude KimiK31 (both share the 'linear_attention' layer_types
    string — qwen35.go:38-48 precedent).
12. Manifest/loader: fused qkvg projection, fp32 conv1d + A_log (legacy 4-D [1,1,H,1] and 1-D
    formats) + dt_bias, expert w1/w2/w3 with gate_up_interleaved=False, single shared beta scale
    block copied whole to every rank, mxfp4 on routed experts only with the ignore-regex set honored
    exactly (attention, shared experts, dense mlp, router, lm_head stay bf16).
13. Assembled forward: no production forward exists even for GLM5Next (ForwardGLM5Next is only a
    ForwardPathKind constant; the only forward is a 4-layer test oracle in glm5next_cadence.go).
    K3.1 needs the full 93-layer assembly + generation loop + Kimi chat template (do NOT reuse GLM).
14. CUDA KDA decode kernel: zero CUDA KDA code exists in fak (grep 'kda' across
    internal/compute/cuda/* returns nothing). Greenfield; donors: cuda/gdn_decode_vllm.cuh (same
    recurrence family, kDim=128 matches), wave32_kda.go + glm_kda_recurrent_wave32.comp (reference
    math), ds4_rocm_glm5_kda.cuh (HIP->CUDA per copy-direct-from-OSS).
15. State/cache: K3.1 KDA state 96x128x128 x 69 layers ~434MB/seq (~3x GLM's ~146MB) + 24 DSA layers
    of O(N) K/V/IndexK buffers; SGLang uses a mamba-style recurrent state cache (KimiLinearStateShape,
    TP-sharded by attn_tp_size) — layout in sglang mamba_utils, not studied.
16. Chat template: Kimi tokens differ from GLM; new template file required.

## Subsystem port order (largest first, per synthesis)

1. **kimi_k31 config spine + arch registration** — blocks everything; separate exact-equality gate
   (near-miss tests pin rejection set), 1-based->0-based layer-list conversion, IsQwen35Hybrid
   exclusion. Files: internal/model/kimi_k31_spine.go (new, mirroring glm5next_spine.go:85-248),
   kimi_k31_spine_test.go, arch_support.go (ForwardKimiK31), config.go seam, qwen35.go exclusion.
   Witness: go test ./internal/model -run 'KimiK31' -count=1
2. **KDA recurrence core + hybrid state (per-element decay variant)** — largest novel-math
   subsystem; clone GLM5Next KDA host files (compact F32 reference), add per-row-decay variant
   alongside scalar StepGLM5NextKDAHead; A_log/dt_bias + separate-beta gate =
   BoundedAsymmetricForget wiring; parameterize 96 heads x 128 x 128, 69 layers; conv window 4
   reused as-is. Must keep prefill==decode bit-exact parity tests.
   Witness: go test ./internal/model -run 'KDA' -count=1
3. **MLA + DSA consumption path (glm_dsa.go reuse) + MLA output gate** — cheapest large subsystem;
   dims thread straight through; DSA on all 24 full-attn layers (index_topk_freq=1); new math is
   only the sigmoid output gate. Files: glm_dsa.go (reuse), glm_dsa_session.go, dsa_index.go,
   kimi_mla_output_gate.go (new).
   Witness: go test ./internal/model -run 'GLMDsa' -count=1
4. **Situ activation** — small but blocks all expert execution (dense MLP layer 0, shared experts,
   routed experts). Files: internal/model/situ.go (new: SituAndMul), situ_test.go (monotonicity/
   clamp invariants + SGLang reference vectors), ffn/gated.go seam.
   Witness: go test ./internal/model/ffn ./internal/model -run 'Situ' -count=1
5. **Latent MoE + router extension + slotstream re-parameterization** — structurally NEW; routed
   experts in 3584-dim latent, reduce-before-norm ordering load-bearing, layers own delayed residual
   add; extend router with e_score_correction_bias bias-ranked top-k; re-parameterize slotstream
   (896 experts, 92 MoE layers). Files: kimi_latent_moe.go (new), glm5next_moe_router.go (extend),
   glm5next_moe.go, internal/compute/slotstream/slotstream.go.
   Witness: go test ./internal/model -run 'MoE|SlotStream|Latent' -count=1
6. **Attention-residual bank** — structurally NEW; per-forward bank of block residuals mixed by
   learned per-token scalar weights; REPLACE_AT_EXIT semantics. Files: kimi_attn_residual.go (new),
   kimi_attn_residual_test.go, glm5next_cache.go (Snapshot/Restore bank-carriage reference).
   Witness: go test ./internal/model -run 'AttnRes' -count=1
7. **Manifest + weight loader + mxfp4 quantization** — fused qkvg, fp32 conv1d + A_log (dual
   formats) + dt_bias, w1/w2/w3 non-interleaved, shared beta scale block, mxfp4 group-32 uint8
   e8m0 dequant (new codec; slotstream codecs are a different format), ignore-regex set honored
   exactly. Files: kimi_k31_manifest.go (new), glm5next_manifest.go (extract shared helper),
   mxfp4.go (new), ggufload kimi_k31 row only if GGUF route needed (safetensors-first).
   Witness: go test ./internal/model ./internal/ggufload -run 'Manifest|Mxfp4|Kimi' -count=1
8. **CUDA KDA decode kernel (greenfield — top risk)** — per-row-decay delta-rule decode; template
   donor cuda/gdn_decode_vllm.cuh (kDim=128 matches, kThreads=256/cp.async/warp-shuffle); vendor
   licensed upstream first per copy-direct-from-OSS (third_party + NOTICE + thin adapter); Go
   bridge cuda_kimi_kda.go with fail-closed contract mirroring vulkan_glm_kda.go:55-114 (exact
   shapes, F32 row-major, backend-owned live allocations, pairwise-disjoint alias check, typed
   contract error, NO host fallback).
   Witness: go test ./internal/compute -run 'KDA' -count=1 (host contract) + NVIDIA node
   go test ./internal/compute -tags cuda -run 'KimiKDA|GLMKDA' -count=1
9. **Assembled 93-layer forward + generation loop + chat template** — the multi-layer forward does
   not exist even for GLM5Next; build embed -> per-layer KDA/MLA-DSA dispatch -> latent-MoE ->
   residual-bank plumbing -> lm_head; two cache regimes; Kimi chat template.
   Witness: go test ./internal/model -run 'KimiK31Forward' -count=1 (tiny deterministic weights,
   end-to-end logits vs layer-by-layer oracle)

## Top risks (from synthesis)

- **No CUDA KDA kernel in fak** (top risk): greenfield; triple-implemented recurrence math in-tree
  (Go StepGLM5NextKDAHead, wave32 CPU simulator, Vulkan SPIR-V) gives three parity donors; vendor
  upstream first (ds4 HIP->CUDA or vllm gdn already vendored at cuda/gdn_decode_vllm.cuh with
  third_party/vllm-gdn-decode/NOTICE).
- **Numerical parity**: SiTU formula not in studied files (lives in SGLang SituAndMul; only
  beta=4.0/linear_beta=25.0 pinned via MegaMoE assert) — fetch and cross-check before any expert
  executes; latent MoE reduce-before-norm ordering load-bearing; exact KDA gate composition
  (A_log/dt_bias relative to low-rank forget gate) needs the SGLang kernel source read.
- **State layout mismatch**: K3.1 mamba-style recurrent state cache (KimiLinearStateShape,
  TP-sharded by attn_tp_size; exact layout in sglang mamba_utils — not studied) vs fak's bespoke
  GLM5NextHybridState; ~434MB/seq KDA state (~3x GLM) + 24 DSA layers of O(N) buffers.
- **Config mapping traps**: one-based layer lists (idx+1); final layer 92 breaks period-4 (a %4==3
  predicate alone is WRONG); derived head_dim = 7168//96 = 74 is a non-divisible floor artifact and
  vestigial for MLA — must never drive head geometry.
- **Quantization**: compressed-tensors mxfp4-pack-quantized (4-bit float, group 32, symmetric,
  uint8 e8m0 scales) has no dequant path in fak (slotstream codecs are Q4_K/Q6_K/NVFP4-style, a
  different format); ignore-regex set (attention, shared experts, dense mlp gate/up/down, router,
  lm_head, vision stay bf16 — effectively only mlp.experts.* quantized) must be honored exactly.
- **GLM5Next gate fragility**: isExactGLM5NextConfig is an exact-equality pin over ~25 fields with
  near-miss rejection tests (glm5next_spine_test.go:37-72) — a kimi_k31 sibling must add its OWN
  gate, never relax the GLM one; IsQwen35Hybrid must explicitly exclude KimiK31.
- **No production forward for GLM5Next either**: the port includes building the assembled
  multi-layer forward + weight loading + generation loop machinery itself.
- **Router gaps**: e_score_correction_bias (bias-ranked top-k, the noaux_tc contract) and
  routed_scaling_factor are parsed into Config but unimplemented; bias ranking is load-bearing.
- **Prefill performance**: fak's GLM KDA prefill is a serial per-token loop (bit-exact with decode
  by test contract); K3.1's 1M-token context on 69 KDA layers makes serial prefill impractical at
  scale; the in-repo chunked donor is qwen35_gdn_chunked_host.go (chunk-64 triangular solve,
  vLLM/SGLang-derived, deliberately NOT bitwise-equal to serial); whether a chunkwise K3.1
  formulation exists is an open question for the KDA port step.

## Disposition

- Study note: this file (committed to docs/notes/ per CONCEPT-STUDY convention).
- Port execution: goal spec `_scratch/goals/GOAL-kimi-k31-port.md` (fak-private, uncommitted
  scratch) tracks the 6-step plan; subsystem order above is the authoritative sequence.
- Lane 1611-kimi-k31: wire-contract WIP (recognition + effort tiers + tests) is correct and lands
  first; the ggufload kimi_k31->deepseek2 collapse is corrected to a documented GGUF-only interim
  with the real route named (GLM5Next-seam safetensors port).
- H200 validation: HITL request kimi-k31-h200-qual created (stage requested, revision-bound to
  20bd1d1b98e7d620fa562577dfb8e9526655e706); qualification plan validated (plan_valid true,
  PLANNING class); arms run per PLAN.md after subsystems land.
