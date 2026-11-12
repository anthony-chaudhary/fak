# Kimi K3.1 Pro NVFP4 — checkpoint contract, fak gap map, and support plan (2026-11-12)

Deep-research note for supporting [nvidia/Kimi-K3.1-Pro-NVFP4](https://huggingface.co/nvidia/Kimi-K3.1-Pro-NVFP4)
(released 2026-11-10) in fak. Every number below is read from the checkpoint's own
files (config.json, hf_quant_config.json, conversion_report.json,
model.safetensors.index.json) or from fak source at the cited path — none is inferred
from the model card prose alone. Companion issues: #14363, #14364, #14365, #14366,
#14367. Prior art: #5147 (Kimi K3 GGUF admission, closed), #11085 (zero-day onboarding).

## 1. What the checkpoint is

Kimi K3.1 Pro is a 2.8T-total / 104B-active multimodal MoE with a 1,048,576-token
context, converted by NVIDIA Model Optimizer (producer string
`modelopt 0.49.0.dev0+gbb13106`) into a **mixed-precision** checkpoint: NVFP4 routed
experts, MXFP8 KDA attention + shared experts, BF16 everything else. The upstream
conversion report states the method plainly: *"raw byte-copy surgery; no quantization,
no calibration (NVFP4 experts = lossless cast of the final MXFP4 experts by the
converter, input_scale 1.0; MXFP8 = converter MXFP8QTensor weight casts)"* — variant
`final-R3-nvfp4_experts-mxfp8_shared-mxfp8_kda-bf16_rest`. Upstream serving targets are
vLLM/SGLang on Blackwell (GB300).

Headline axes (from `text_config` of config.json):

| Axis | Value |
|---|---|
| hidden_size / intermediate_size | 7168 / 33792 |
| Layers | 93 total: layer 0 dense, layers 1–92 MoE |
| Routed experts | 896, 16 active, latent width 3584 (`routed_expert_hidden_size`), moe_intermediate 3072 |
| Shared experts | 2 |
| Vocab / context | 163840 / 1,048,576 |
| Embeddings | untied (lm_head separate) |
| Attention hybrid | KDA layers {0,1,2,4,…,90} (0-based), DSA layers {3,7,…,91,92}; MLA dims: q_lora 1536, kv_lora 512, qk_nope 128, qk_rope 64, v_head 128, 96 heads, `mla_use_nope` + `mla_use_output_gate` |
| KDA | 96 heads × 128 head_dim, conv kernel 4, gate lower bound −5.0, `use_full_rank_gate` |
| DSA indexer | 32 heads × 128 dim, topk 2048, skip offset 1, freq 1, indexer layernorm |
| Activation | `situ` (β 4.0, linear β 25.0) |
| Residual blocks | `attn_res_block_size: 12`, `latent_moe_use_norm: true` |
| Router | sigmoid / noaux_tc / grouped topk, `routed_scaling_factor 1.0`, `e_score_correction_bias` |
| Vision | MoonViT-V2 tower (27 layers, hidden 1024, 12 heads, patch 14) + `patchmergerv2` projector → 7168; image placeholder token 163605 |

## 2. Quantization contract

| Surface | Key | Value |
|---|---|---|
| config.json | `quantization_config.quant_method` | `"modelopt_mixed"` |
| hf_quant_config.json | `quant_algo` | `"MIXED_PRECISION"` |
| hf_quant_config.json | `kv_cache_quant_algo` | `null` (KV stays unquantized) |
| group_0 (routed experts) | weights / input activations | NVFP4: 4-bit, group 16, **static** input scales (`input_scale 1.0`) |
| group_1 (attention + shared experts) | weights / input activations | MXFP8: 8-bit, group 32 weights static; input activations 8-bit group 32 **dynamic** (runtime-quantized) |
| ignore list (stays BF16) | — | `*block_sparse_moe.gate*`, `*routed_expert_up/down_proj*`, `*routed_expert_norm*`, `*self_attn.{q,k,v}_conv1d*`, `*self_attn.f_b_proj*`, `*o_norm*`, `*q_a_layernorm*`, `*kv_a_layernorm*`, `*_res_proj*`, `*_res_norm*`, `*layernorm*`, `layers.0.mlp.*`, `lm_head`, `vision_tower*`, `mm_projector*` |

The latent-MoE shape matters as much as the dtypes: routed experts consume a 3584-wide
latent produced by BF16 `routed_expert_up_proj` (7168→3584) and return through BF16
`routed_expert_down_proj` (3584→7168) + `routed_expert_norm`. This mirrors fak's
`v4_expert_*` latent machinery rather than a classic 7168-wide expert MLP.

## 3. Tensor census (ground truth from `model.safetensors.index.json`)

992,691 tensors, 1,604,984,317,952 bytes, 96 shards. Key prefixes: `language_model.*`
(992,523 tensors), `vision_tower.*` (165), `mm_projector.*` (3). dtype buckets (HF API):
BF16 14,704,796,672 · F32 11,122,432 · F8_E4M3 42,653,614,080 · U8 1,361,370,415,104.

| Role | Tensor pattern (indices stripped) | Quant |
|---|---|---|
| Routed expert (per expert) | `language_model.model.layers.N.block_sparse_moe.experts.E.{w1,w2,w3}.{weight,weight_scale,weight_scale_2,input_scale}` — 12 tensors × 896 experts × 92 layers | NVFP4 |
| Router | `…block_sparse_moe.gate.{weight,e_score_correction_bias}` | BF16 |
| Shared experts | `…block_sparse_moe.shared_experts.{gate,up,down}_proj.{weight,weight_scale}` | MXFP8 |
| Attention | `…self_attn.{q,k,v,o,b,f_a,g}_proj.{weight,weight_scale}` | MXFP8 |
| Attention (unquantized) | `…self_attn.{q,k,v}_conv1d.weight`, `self_attn.f_b_proj.weight`, `self_attn.o_norm.weight`, `self_attn.A_log`, `self_attn.dt_bias` | BF16 |
| Residual stream | `…layers.N.{mlp_res_proj,self_attention_res_proj}.weight` + `{mlp_res_norm,self_attention_res_norm}.weight`; top-level `language_model.model.output_attn_res_{proj,norm}.weight` | BF16 |
| Norms / embed | `input_layernorm`, `post_attention_layernorm`, `language_model.model.norm.weight`, `language_model.model.embed_tokens.weight` | BF16 |
| Vision | `vision_tower.*` (165 tensors), `mm_projector.*` (3) | BF16 |

Note: config.json's group targets list both `block_sparse_moe.experts` and `mlp.experts`
as aliases; the index shows the real tensors only under `block_sparse_moe.experts`.

**Scale layout: pinned from real shard headers (2026-11-12).** The safetensors headers
of shards 1–2 (first 8 bytes = little-endian header length, then JSON; HTTP Range works
with retries — some clients saw connection resets) give the exact per-tensor contract:

| Tensor | dtype | shape | meaning |
|---|---|---|---|
| `experts.E.w1.weight` / `w3.weight` | U8 | [3072, 1792] | NVFP4 packed 2/byte → [3072, 3584] (gate / up: latent 3584 → moe_intermediate 3072) |
| `experts.E.w2.weight` | U8 | [3584, 1536] | NVFP4 packed → [3584, 3072] (down) |
| `experts.E.w{1,3}.weight_scale` | F8_E4M3 | [3072, 224] | per-16-group E4M3 scales (3584/16) |
| `experts.E.w2.weight_scale` | F8_E4M3 | [3584, 192] | per-16-group E4M3 scales (3072/16) |
| `experts.E.w{1,2,3}.weight_scale_2` | F32 | scalar | global scale |
| `experts.E.w{1,2,3}.input_scale` | F32 | scalar | static input scale (1.0 — lossless MXFP4 cast) |
| `self_attn.{q,k,v,g}_proj.weight` | F8_E4M3 | [12288, 7168] | MXFP8 data (12288 = 96×128) |
| `self_attn.o_proj.weight` | F8_E4M3 | [7168, 12288] | MXFP8 data |
| `self_attn.b_proj.weight` | F8_E4M3 | [96, 7168] | per-head beta projection |
| `self_attn.f_a_proj.weight` | F8_E4M3 | [128, 7168] | low-rank f_a |
| MXFP8 `weight_scale` (all MXFP8 tensors) | U8 | [out, in/32] | E8M0 exponent: scale = 2^(byte − 127) |
| `shared_experts.{gate,up}_proj.weight` | F8_E4M3 | [6144, 7168] | fused 2×3072 shared-expert intermediate |
| `shared_experts.down_proj.weight` | F8_E4M3 | [7168, 6144] | MXFP8 data |
| `self_attn.{q,k,v}_conv1d.weight` | F32 | [12288, 1, 4] | KDA conv, kernel 4 |
| `self_attn.A_log` / `self_attn.o_norm.weight` | F32 | [128] | KDA internals / output RMS |
| `self_attn.dt_bias` | F32 | [12288] | KDA time-step bias |
| `{mlp,self_attention}_res_proj.weight` | BF16 | [1, 7168] | per-token scalar residual gates (`attn_res_block_size 12`) |
| `routed_expert_up_proj` / `routed_expert_down_proj` | BF16 | [7168, 3584] / [3584, 7168] | latent projections |
| `routed_expert_norm` | BF16 | [3584] | latent RMS |
| `mlp.{gate,up,down}_proj` (layer 0 only) | BF16 | [33792, 7168] | dense layer |
| `gate.weight` / `gate.e_score_correction_bias` | BF16 / F32 | [896, 7168] / [896] | sigmoid router + noaux_tc bias |

Arithmetic cross-check: 92 × 896 × (3 × 3072×1792 packed bytes) = 1,361,370,415,104 —
exactly the U8 census bucket, so the U8 bucket is precisely the packed NVFP4 expert
weights. (An earlier HF-API dtype-bucket estimate for F8_E4M3 did not include the
~170B per-group expert scales; the per-tensor header reads above supersede it.)

## 4. fak today (what exists, what refuses)

- **K3.1 GGUF admission stack: landed** (L1–L6). `internal/ggufload/kimik31.go`,
  `kimik31_tensors.go`; KDA oracle `internal/model/kimik31_kda.go`; DSA oracle
  `kimik31_dsa.go` / `kimik31_dsa_forward.go`; trio `kimik31_trio.go`. Canonical arch
  `kimik31`.
- **L8 dispatch: open** (#14363). `ClassifyForwardPath`
  (`internal/model/arch_support.go:130`) has no `kimik31` branch, so even the landed
  GGUF-side stack never executes.
- **HF/safetensors K3.1 path: not wired yet.** No config parsing, no safetensors
  admission, no vision tower loader (`VisionEncoder`,
  `internal/model/multimodal.go:108`, is an external-embedding seam only, #14366).
- **ModelOpt NVFP4 (plain) loader: landed.** `internal/model/modelopt_nvfp4_checkpoint.go`
  + `modelopt_nvfp4_load.go` detect `quant_method: "modelopt"` + `quant_algo: "NVFP4"`,
  index bases by `.weight_scale_2` / `.weights_scaling_factor_2`, and refuse anything
  else with `ErrModelOptNVFP4Config`. Execution rides `compute.NVFP4MatMulBackend`
  (CUDA sm_100/103/120, `internal/compute/cuda.go:1006-1036`) plus a CPU oracle; no
  Metal/Vulkan/ROCm NVFP4 GEMM.
- **Mixed-precision admission, kimi_k31 config parsing, MXFP8 decode: landing
  2026-11-12** in companion commits (detection + typed group summary + structured
  load refusal; text_config unwrap + `archFamilyKey`; E8M0/E4M3 block-32 decode
  oracle). These are enabling leaves — they make the checkpoint *recognized*, not yet
  *runnable*.

## 5. Gap map → issues

| # | Gap | Witness gate |
|---|---|---|
| #14363 | L8 `ClassifyForwardPath` branch + `ForwardKimiK31` wiring over the landed KDA/DSA/trio oracles | classify test + small-tensor forward smoke (finite logits) |
| #14364 | NVFP4 routed-expert MoE execution (latent 3584, w1/w2/w3, batched 896-expert GEMM, 16-active routing) | CPU oracle parity on synthetic NVFP4 experts; hardware claims need a Blackwell/L4 HIL receipt |
| #14365 | MXFP8 wiring: `weight_scale` branch in `quantizeNamedTensorsInto` (`internal/model/safetensors_quant.go:461-542`) + runtime dynamic-activation quantizer | synthetic mixed-checkpoint load test decoding MXFP8 → BF16 within tolerance |
| #14366 | MoonViT-V2 tower + patchmergerv2 projector loader through the `VisionEncoder` seam | synthetic vision-tensor load + image→embeddings→logits smoke |
| #14367 | 1M-context KDA/DSA envelope witness | fleet-node staged prefill (32k→1M) with retrieval correctness + net-true tok/s; bench workstream, own receipts |

Order of attack: #14363 (small, keystone) → #14365 (decode exists, wiring is bounded) →
#14364 (biggest lift; scale shapes are pinned in §3 — the remaining unknown is execution,
not the contract) → #14366 → #14367.

## 6. Honest status

Kimi K3.1 Pro NVFP4 is **recognized, not runnable yet** in fak: the checkpoint contract
is fully characterized (this note), admission/config/decode leaves are landing, and the
five execution/witness issues carry typed witnesses. The next checkable step is the L8
dispatch branch (#14363) — after it, GGUF-side K3.1 checkpoints become executable ahead
of the NVFP4 execution work.
