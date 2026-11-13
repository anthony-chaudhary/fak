# Kimi-K3.1-Pro-NVFP4 on H200 — sizing verdict + serving recipes (2026-11-12)

Question: can `nvidia/Kimi-K3.1-Pro-NVFP4` fit and perform on 4×H200 with offloading, or does it need all 8? What vLLM recipe, and what performance/concurrency to expect?

All numbers below are tied to a fetched source (evidence index at the bottom). Anything derived is labeled ESTIMATE.

---

## Verdict

1. **No on 4×H200 — offloading does not change the answer.** The checkpoint is 1,622 GB (1,476 GiB) of safetensors; TP4 needs ~406 GB/GPU against 141 GB VRAM (2.9× over). Offloading cannot rescue it: no shipping system supports expert-granular offload for this architecture (vLLM expert offload is RFC-only [#33869](https://github.com/vllm-project/vllm/issues/33869)/[#63666](https://github.com/vllm-project/vllm/issues/63666); ktransformers/llama.cpp/MoE-Infinity/Fiddler all lack KDA/DSA/NVFP4 support), naive weight streaming caps at ~4 tok/s, and smart hot-expert caching still caps at ~40–85 tok/s (six-lane section below).
2. **No on 8×H200 either.** TP8 needs ~203 GB/GPU (1.44× over) → OOM during weight load, before any KV is allocated. ~70% of the checkpoint fits; the remaining ~494 GB has nowhere to go.
3. **The only capacity-viable Hopper shape is 16×H200 (2 nodes, TP16)** — ~101–110 GB weights/GPU — and K3.1-Pro on it is **plausible today but not witnessed yet**: every ingredient is individually SM90-proven (KDA via FlashKDA "SM90 and above"; DSA via FlashMLA's Hopper FP8 sparse kernel; MoE FP4 via Marlin W4A16 auto-fallback; predecessor K3 ran 16×H200 end-to-end, measured 16.8 tok/s single-stream), but no Hopper preset or published K3.1 run exists — the packed-cache wiring is the untested combination. See the six-lane research section below.
4. **~~Practical H200 recipe: K2-Thinking-NVFP4~~ — SUPERSEDED by operator decision (2026-11-12): K2-Thinking and older models are dropped as serving targets (not worth it).** The target is K3.1-Pro itself: 16×H200 (plausible, unwitnessed — build it fak-native; ticket set in the [value note](2026-11-12-fak-native-value-kimi-k31-pro-upstream-gaps.md)) or Blackwell (official recipe below). The K2-Thinking math stays below as historical evidence only.
5. **If K3.1-Pro specifically is the requirement: buy Blackwell** (B300/GB300). The official recipe is TP8 on 8×B300/GB300, and day-0 perf is sourced (375.9 tok/s/user decode on GB300).

---

## The checkpoint (measured from HF API, `?blobs=true`)

| Property | Value | Source |
|---|---|---|
| Safetensors | 96 shards, **1,622 GB (1,476 GiB)** | HF API blobs |
| usedStorage | 1,605,246,062,422 B | HF API |
| Param dtypes | U8 1,361.37 B params (FP4-packed routed experts) · F8_E4M3 42.65 B · BF16 14.70 B · F32 11.1 M — total 1.419 T stored | HF API blobs |
| Total params | 2.8 T total / 104 B active (MoE) | model card |
| Architecture | 93 layers = 69 KDA (linear attention) + 24 full-attn MLA/DSA (layers 4..93, indexer top-k 2048); hidden 7168; 896 routed experts / 16 active + 2 shared; vocab 163,840; multimodal (MoonViT-V2) | `moonshotai/Kimi-K3.1-Pro` config.json |
| Context | 1,048,576 | model card |
| Released | 2026-11-10; "Supported Hardware Microarchitecture Compatibility: **NVIDIA Blackwell**"; test hardware GB300 | model card |
| Runtimes | vLLM (≥ 0.34.0 nightly, image `vllm/vllm-openai:kimi-k31-cu134`) + SGLang | model card + vLLM recipe |

## Capacity math on H200 (141 GB HBM3e each, ~131 GiB usable)

| Shape | Weights/GPU | vs 141 GB | Verdict |
|---|---|---|---|
| TP4 | 406 GB (369 GiB) | 2.9× over | Impossible — weights alone |
| TP8 | 203 GB (184.5 GiB) | 1.44× over | **OOM during weight load** (even at 0.95 util, ~50 GB short) |
| TP16 | 101 GB (92 GiB) | fits, ~72% of VRAM | Capacity-viable; ~26 GB/GPU left for KV + SSM at 0.9 util |

4×H200 holds only ~35% of the checkpoint; 8×H200 holds ~70%.

## Why offloading does not save 4× or 8×

1. **The tuned day-0 paths are Blackwell-only; the SM90 ingredient kernels are not.** Model card: Blackwell only. The official vLLM recipe's hardware keys are gb300, b300, vr200, mi355x, ironwood, 810e, m890 — no Hopper entry. The K3.1-specific *tuned* kernels merged into vLLM (#66030, #66037, #66532) are FlashInfer trtllm-gen / CuTe DSL (Blackwell) plus ROCm. But the six-lane research (below) shows the underlying SM90 kernels exist per component: FlashKDA prefill is "SM90 and above", FlashMLA ships a Hopper FP8 sparse-DSA decode kernel, and MoE FP4 falls back to Marlin W4A16 automatically. What is missing on SM90 is the K3.1-Pro wiring/preset and any witnessed run — not the kernels themselves.
2. **CPU weight streaming is a non-starter (ESTIMATE).** vLLM's `--cpu-offload-gb` streams weights from host RAM each forward. Active params ≈ 104 B × 0.5 B/param ≈ 52 GB/token ÷ 4 GPUs ≈ 13 GB/GPU/step over PCIe (~50 GB/s) ≈ 0.26 s → **~4 tok/s ceiling** single-stream before attention/framework overhead — on a 1M-context agent workload.
3. **KV offload connectors move KV, not weights.** `SimpleCPUOffloadConnector` (used in the Blackwell preset) relieves KV pressure; it does nothing about a 2.9×/1.44× weight overage.

## The 16×H200 shape — capacity-viable, not supported yet

- 92 GiB weights/GPU → ~26 GB/GPU for KV + SSM at 0.9 util.
- Full-attn KV (24 layers, replicated under TP): **16,992 B/token/GPU** (FP8 latent + FP8 index) per the official recipe → ~1.6M tokens/GPU → ~26M tokens fleet (ESTIMATE). NVFP4-latent variant: 10,080 B/token/GPU.
- KDA SSM state (ESTIMATE, derived from config): 69 layers × 96 heads × 128×128 × 4 B ≈ **434 MB/request** aggregate; head-sharded across TP16 → ~27 MB/request/GPU. Replication behavior under TP needs a witness before trusting this for sizing.
- Precedent ([recipes#698](https://github.com/vllm-project/recipes/issues/698)): Kimi-K3 (1,453.74 GiB MXFP4) OOMs at TP8 on H200; community runs TP16 (2×8 H200) Marlin W4A16 with `--max-model-len 32768`, no DSpark with pipeline parallelism. K3.1-Pro adds KDA/DSA/multimodal on top — no Hopper validation exists anywhere.
- Secondary confirmation (Qiniu deployment guide, 2026-11-05): "H100, H200, H20 not in validated list; 8×H200 ≈ 1.1 TB can't hold 1.56 TB weights; Hopper has no native FP4 compute units."

---

## Recipe A — DROPPED as a serving target (operator decision 2026-11-12): Kimi-K2-Thinking-NVFP4, TP8 on 8×H200

> Operator decision 2026-11-12: K2-Thinking and older models are **not worth it** as serving targets — dropped. The serving target is K3.1-Pro (16×H200 fak-native build, or Blackwell). Math kept below as historical evidence.

- Checkpoint 594 GB / 553 GiB (519 B params: U8 507.34 B FP4-packed + BF16 11.72 B) — HF API.
- `DeepseekV3ForCausalLM` (MLA — no linear attention), 61 layers, `kv_lora_rank` 512 + rope 64, 256k ctx, 384 experts / 8 active + 1 shared — config.json (fetched).
- NVFP4 quantizes Linear/MoE only; attention + lm_head stay BF16 (quantization_config `ignore` list); **FP8 KV scheme is baked into the checkpoint** (`kv_cache_scheme` 8-bit float); `num_nextn_predict_layers: 0` → **no MTP draft in this checkpoint → no speculative decoding** (unlike K3.1-Pro's DSpark).
- Card example is TP4 (B200-class, 180 GB). On H200 TP4 = 148 GB/GPU → over capacity → **use TP8** = 74 GB/GPU weights → ~52 GB/GPU KV at 0.9 util.
- Honesty note: this card also lists "Supported Hardware: NVIDIA Blackwell" (test hardware B200). The H200 path runs NVFP4 weights through **Marlin W4A16 kernels** (NVIDIA NIM documents this explicitly: "H200 profiles run the NVFP4 weights with Marlin kernels", `NIM_ALLOW_NVFP4_EMULATION=1`; open vLLM falls back to Marlin for NVFP4 on SM90). **Not witnessed on our fleet yet — the first run is the witness.**

```sh
docker run --gpus all --ipc=host -p 8000:8000 vllm/vllm-openai:v0.11.2 \
  python3 -m vllm.entrypoints.openai.api_server \
  --model nvidia/Kimi-K2-Thinking-NVFP4 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

- KV/concurrency (ESTIMATE, FP8 MLA latent 576 B/token/layer × 61 layers = 35,136 B/token/GPU, replicated under TP): ~1.5M tokens/GPU → ~12M fleet → **~46 concurrent @ 256k ctx** or **~370 @ 32k avg** (capped by `--max-num-seqs 256`).
- Perf expectation (ESTIMATE, not witnessed): memory-bound decode; ~32 B active × 0.5 B ≈ 16 GB/token ÷ 8 GPUs = 2 GB/GPU/token ÷ 4.8 TB/s ≈ 0.42 ms → roofline ~2.4k tok/s single-stream; DeepSeek-class MoE+Marlin real-world efficiency historically ~5–10% of roofline → **~120–250 tok/s single-stream**, aggregate several thousand tok/s at high concurrency. A witnessed bench replaces this range.

## Recipe B (official, Blackwell — for B300/GB300 acquisition)

Official serve command (model card):

```sh
vllm serve nvidia/Kimi-K3.1-Pro-NVFP4 --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 1048576 \
  --max-num-seqs 256 \
  --load-format fastsafetensors \
  --trust-remote-code \
  --reasoning-parser kimi_k3 \
  --enable-auto-tool-choice --tool-call-parser kimi_k3
```

Tuned `b300_agentx` preset (8×B300, concurrency-1 latency point, from the vLLM recipe):

```sh
# image: vllm/vllm-openai:kimi-k31-mc@sha256:f81998764e8594d71b40788c5285ae112acf9230129d5d0d238b5df83da5acf6
# env: VLLM_ENGINE_READY_TIMEOUT_S=7200 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_DIRECT_DCP_A2A=1 \
#      VLLM_USE_DIRECT_DCP_Q_GATHER=1 VLLM_USE_DIRECT_DCP_KV_GATHER=1 VLLM_ALLREDUCE_USE_FLASHINFER=1
vllm serve nvidia/Kimi-K3.1-Pro-NVFP4 \
  --tensor-parallel-size 8 \
  --load-format fastsafetensors \
  --decode-context-parallel-size 8 --dcp-comm-backend a2a --cp-kv-cache-interleave-size 1 \
  --enable-prefix-caching --kv-cache-dtype fp8 --mamba-ssm-cache-dtype float32 --use-replayssm \
  --no-enable-flashinfer-autotune --max-num-batched-tokens 16384 --gpu-memory-utilization 0.9 \
  --max-num-seqs 2 \
  --attention-config '{"indexer_kv_dtype":"mxfp4","tokenspeed_mla_min_split_kv":32}' \
  --speculative-config '{"model":"Inferact/Kimi-K3.1-Pro-DSpark-preview","num_speculative_tokens":4,"method":"dspark","draft_sample_method":"probabilistic","revision":"6a2ef20a05efe88aabbf9ecaa12d020c3b0edf94","attention_backend":"TOKENSPEED_MLA","rejection_sample_method":"block"}' \
  --language-model-only \
  --additional-config '{"decode_backend":"latency_specialized"}' \
  --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use_per_rank":281125000000,"lazy_offload":false}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,10]}' \
  --reasoning-parser kimi_k3 --enable-auto-tool-choice --tool-call-parser kimi_k3
```

Key Blackwell mechanics: DCP8 ≈ 7× KV tokens (~18.5M vs 2.6M FP8 tokens on 8×B300); NVFP4 KV ≈ 1.5× FP8 tokens; KV bytes/token/GPU (24 full-attn layers, replicated): 16,992 B (FP8) / 10,080 B (NVFP4+MXFP4).

## Performance & concurrency expectations

| Metric | Value | Status |
|---|---|---|
| GB300 TP8 decode (AgentX) | 375.9 tok/s/user (end) | SOURCED (vLLM day-0 blog) |
| Rubin TP8 decode (AgentX) | 815.6 tok/s/user | SOURCED (same) |
| DSpark spec decode | 3.0× decode at batch 1, acceptance len 4.597 (7 drafts) | SOURCED (same) |
| Prefill, 340K-token prompt | ~10 s (TP8+EP8) vs ~15 s (TP8) on 8×B300 | SOURCED (same) |
| ~~K2-Thinking-NVFP4 on 8×H200, single-stream~~ | ~120–250 tok/s | DROPPED with the model (operator decision 2026-11-12) — ESTIMATE, never witnessed |
| ~~K2-Thinking-NVFP4 on 8×H200, aggregate @ high concurrency~~ | several thousand tok/s | DROPPED with the model — ESTIMATE, never witnessed |
| ~~K2-Thinking concurrency~~ | ~46 @ 256k ctx / ~370 @ 32k (cap 256 seqs) | DROPPED with the model — ESTIMATE (KV math above) |
| K3.1-Pro on H200 (as-shipped NVFP4) | Does not run today — no Hopper preset/witnessed run; 16×H200 plausible via MXFP4 + Marlin + FlashKDA/FlashMLA at ~15–20 tok/s single-stream | SOURCED (card + recipe) + ESTIMATE (six-lane section) |

## Six-lane H200 runnability research (2026-11-12)

Six parallel research lanes (SM90 kernel matrix, conversion paths, 4×H200 offload feasibility, 16×H200 multi-node assembly, spec-decode, perf levers) upgraded the early picture: the "no SM90 kernels" framing was too pessimistic — per-component Hopper kernels largely exist; what is missing is the K3.1-Pro wiring, a preset, and any witnessed run.

### Corrected SM90 kernel picture (per component)

| Component | SM90 status | Evidence |
|---|---|---|
| KDA linear attention | SHIPPED (functional, untuned) — FlashKDA CUTLASS prefill is "SM90 and above" (H20-benchmarked 1.72–2.22×) + FLA Triton decode fallback; auto-dispatch via flash-linear-attention ≥ 0.5.0 | https://vllm-project.github.io/2026/07/27/k3.html · https://github.com/MoonshotAI/FlashKDA · https://github.com/fla-org/flash-linear-attention/pull/852 |
| DSA indexer + sparse MLA | SHIPPED for the same DSA (V3.2) — FlashMLA ships a Hopper FP8 sparse decode kernel (pinned pre-2026-09-30, `ba89a34`; current FlashMLA needs SM100+); SGLang DSA verified on Hopper (`flashmla_sparse`, fa3 default decode on Hopper, DeepGEMM indexer, `triton_sparse_mla` SM90+); vLLM `FLASHMLA_SPARSE` = "SM90/SM100 only" | https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md · https://docs.sglang.io/docs/advanced_features/attention_backend |
| MLA dense | SHIPPED — K3 ran on H200 (`--decode-attention-backend flashmla`, 16.8 tok/s single-stream) | https://huggingface.co/moonshotai/Kimi-K3/discussions/136 |
| MoE NVFP4 | SHIPPED via Marlin W4A16 auto-fallback on SM90 (machete is int4-only → FP4 always routes to Marlin); watch the NVFP4-Marlin MoE swiglu-bridge bug class (#34694) and NVFP4-MoE Hopper garbage/IMA history (#49070) | https://docs.vllm.ai/en/stable/features/quantization/modelopt/ · https://github.com/vllm-project/vllm/issues/49070 · https://github.com/vllm-project/vllm/pull/54427 |
| MoonViT-V2 | No blocker found / unwitnessed (standard NaViT-style ViT, no arch-specific kernels) | https://vllm.ai/blog/2026-07-22-kimi-k3-preview |

Real SM90 gaps: (1) K3.1-Pro's packed 3-state cache (MLA latent + indexer keys + KDA state) wired to the SM90 kernels — day-0 only exercised Blackwell/ROCm/TPU; (2) NVFP4-KV-latent/MXFP4-index KV is Blackwell-only — H200 must run FP8 latent + FP8 index (1.5× worse KV capacity than the Blackwell feature); (3) no Hopper preset anywhere (recipe, Dynamo, SGLang H200 row = "placeholder, not verified"); (4) DSpark fused verify/commit is SM10x-only (spec-decode section below).

### Weight format: use the original MXFP4 checkpoint, not a requant

- `moonshotai/Kimi-K3.1-Pro` (MXFP4 experts + BF16 rest, 1,561 GB) is the Hopper-proven format (K3 precedent). Serving the NVFP4 checkpoint as-is on SM90 also auto-falls-back to Marlin, but buys zero capacity (same bits) and carries the #49070 bug history. Requant NVFP4→MXFP4 is pointless (source already MXFP4; SGLang `--quantization quark_mxfp4` exists if ever needed). No FP8/BF16/Air variants exist (HF enumeration) — FP8 would need ~32 GPUs at TP16 (ESTIMATE).
- 8×H200 stays dead for any 4-bit format: ~206 GB/GPU weights vs ~140 usable.
- 16×H200: TP16 = 109.5 GB/GPU plain; Marlin padding (expert dim 3072/16=192→256, +33%) → K3 measured **130.2 GiB/GPU, 262k ctx served**; alternative **TP8×PP2 = 96.79 GiB weights + 27.15 GiB KV (1.66M tokens, 256k ctx)**, no padding (DSpark+PP was fixed and closed 2026-09-16, #50098). Sources: recipes#698 thread.
- Gimmick fallback: `vllm-project/Kimi-K3.1-Pro-pruned75` (896→224 experts, ~443 GB) fits 8×H200 at ~55 GB/GPU — but GSM8K 43.7% vs ~96% full: smoke-test only.

### 16×H200 recipes (K3-proven shape; K3.1 = same shape, unwitnessed)

vLLM official K3 H200 profile (2 nodes, TP16, Marlin):

```sh
vllm serve moonshotai/Kimi-K3 --trust-remote-code --tensor-parallel-size 16 \
  --nnodes 2 --node-rank 0 --master-addr $HEAD_IP \
  --max-num-seqs 5 --max-model-len 32768 --moe-backend marlin \
  --disable-custom-all-reduce --disable-flashinfer-autotune \
  --max-num-batched-tokens 4096 --attention-backend FLASHMLA \
  --enable-auto-tool-choice --tool-call-parser kimi_k3 --reasoning-parser kimi_k3
# node 1 adds --headless; requires InfiniBand; NCCL_DMABUF_ENABLE=0 on mlx5 errno 524
```
Source: https://recipes.vllm.ai/moonshotai/Kimi-K3.md?hardware=h200&strategy=multi_node_tp&nodes=2

SGLang measured end-to-end (2026-07-28): `--tp-size 16 --ep-size 16 --nnodes 2 --moe-runner-backend marlin --decode-attention-backend flashmla --mem-fraction-static 0.85` → 102.75 GB/GPU weights, 7.95 GB KV (bf16) = 308,608 tokens, **16.8 tok/s single-stream, 146.9 tok/s aggregate @ 54 conc** (not saturated), TTFT 0.37–0.54 s, 13-min cold start. Production tuning (Periodic Labs): TP32+EP32 → ~4M KV tokens/replica vs ~800K at TP16; FP8 KV doubled capacity, no quality/throughput regression; DP+EP loses to TP+EP (only 24/93 layers are MLA); symm-mem unstable multi-node. Sources: https://huggingface.co/moonshotai/Kimi-K3/discussions/136 · https://www.linkedin.com/posts/byronhsu1230_scaling-kimi-k3-on-h200s-withsglang-while-activity-7489912256059572224-fWxi

K3.1-Pro concurrency at TP16 (ESTIMATE): ~25 GB/GPU KV budget ÷ 16,992 B/token/GPU (MLA KV is TP-replicated) → ~1.47M tokens engine-wide → ~45 seqs @32k, ~11 @128k. KDA SSM = 27.1 MB/req/GPU at TP16 (96 heads ÷ 16) → not the binding cap; KV is.

### 4×H200 with offloading — honest numbers (ESTIMATE; no shipping system today)

- Resident: non-expert weights ≈ 72 GB + KV → ~400 GB of 564 GB free ≈ **60% of the ~680 GB expert payload cacheable** in CPU RAM.
- Decode ceiling vs hot-expert hit rate h (aggregate PCIe 5.0 ×4 ≈ 256 GB/s; routed reads ≈ 12.2 GB/token): h=0.5 → 42 tok/s; h=0.75 → 84; h=0.85 → 140. Measured hit-rate anchors (MoE-Infinity: 17% cache → 46% hit; MoE-Beyond: 10% → >70%) → expect **~40–85 tok/s** decode. Fiddler-style CPU-computes-misses: ~37 tok/s at h=0 (~450 GB/s DDR5). Prefill amortizes: ~1,500 tok/s (a 4k chunk touches ~all 896 experts/layer).
- Build gaps: vLLM expert-granular offload is RFC-only (#33869, #63666); FlashKDA + pinned FlashMLA need wiring into an offload path; popularity-profiled placement (Fiddler-style) must be tuned to K3.1 routing. Verdict: not today; possible as a build — and 16×H200 beats it on every axis anyway.

### Speculative decoding on H200 (the "what else is needed" answer)

- **DSpark mechanics**: 5-layer parallel drafter (2.7B BF16, taps 5 target layers' hidden states) + rank-512 Markov-head correction + confidence-scheduled per-request verification length (arXiv 2607.05147). Draft acceptance pooled 3.154 (GSM8K 4.597) — consistent with the B300 3.0× claim. Draft card: https://huggingface.co/Inferact/Kimi-K3.1-Pro-DSpark-preview
- **The draft is not Blackwell-bound; the fast path is.** `TOKENSPEED_MLA` attention backend is compute-cap 10.x-only (vLLM attention-backend docs); FlashInfer fused speculative KDA is SM100 (#54255); CuTe DSL KDA verify/commit is SM10x. But AMD serves the same DSpark draft on MI355X, and DeepSeek-V4 DSpark was validated on 8×H200.
- **Recommended H200 config (ESTIMATE 1.3–2.0× at conc ≤8; ~1.0× at conc ≥32–64)**:

```sh
vllm serve moonshotai/Kimi-K3.1-Pro --tensor-parallel-size 16 --trust-remote-code \
  --kv-cache-dtype fp8 --load-format fastsafetensors \
  --speculative-config '{"model":"Inferact/Kimi-K3.1-Pro-DSpark-preview","num_speculative_tokens":4,"method":"dspark","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
# NO --attention-backend TOKENSPEED_MLA (SM10x-only); SM90 defaults to FlashMLA
```
  Shorter blocks than the B300 preset's 7: pooled acceptance decays 0.72→0.50→0.34→0.23 by position 4, and W4A16 verify is relatively costlier on Hopper.
- **MTP: dead end** — `num_nextn_predict_layers: 0`, zero `mtp_*` shards (config.json + blobs API). EAGLE3 substitute exists (`lightseekorg/kimi-k3-eagle3-mla` collection; `RedHatAI/Kimi-K3-speculator.dspark` — B300-validated only); SM90 plumbing unverified.
- **Crossover**: above ~64 concurrent requests, draft-and-verify overhead outweighs savings (Periodic Labs, K3 on H200) — one draft/cost point, not a constant.
- Gaps to DSpark-class 3× on Hopper: no SM90 fused verify/commit for the KDA+DSA hybrid; no published SM90 K3-DSpark integration test (the sibling V4 family needed per-model SM90 fixes: KV shape #47648, DeepGEMM illegal address #52065, Marlin `map_draft_to_target` assert #56443); W4A16 verify inflates target-forward cost; multi-node TP+EP+DSpark untested.

### Perf levers beyond spec-decode (ranked)

| # | Lever | Expected gain | SM90-portable? | Evidence |
|---|---|---|---|---|
| 1 | NVFP4→FP8-compute at load (`VLLM_NVFP4_GEMM_BACKEND=fp8-compute`, open PR #38728) | +62% tok/s, −46% p99 ITL (2×H200, dense 119B; MoE unproven) | Yes (Hopper CUTLASS FP8) | https://github.com/vllm-project/vllm/pull/38728 |
| 2 | K3 engine-fusion pass (fused KDA decode kernel, LatentMoE tail fusion ~20%, custom reduce-scatter/all-gather 1.7–4.5× NCCL, skinnyGEMM) | 2.8× aggregate (measured B300) | Mostly | https://vllm-project.github.io/2026/09/13/kimi-k3-performance-optimization.html |
| 3 | SGLang FlashInfer cutlass MXFP4 MoE (FlashInfer ≥ 0.6.11) | prefill +24–36% at m≥1024; decode stays Marlin (+12–15% at m≤64) | Yes | https://github.com/sgl-project/sglang/pull/24816 |
| 4 | KV dtype: SGLang H200 FP8 KV = 2× tokens; vLLM H200 bf16-only for K3 (#51313); NVFP4-KV latent is Blackwell | capacity, not speed | SGLang yes / vLLM no | https://github.com/vllm-project/vllm/issues/51313 |
| 5 | Shape discipline: TP+EP; no MNNVL on Hopper → FlashInfer multi-node allreduce dead; symm-mem unstable multi-node | avoids regressions | shape-fixed | https://github.com/vllm-project/vllm/issues/44079 |
| 6 | RecoverSSM `--use-replayssm` +10.97% KV (requires spec decode); CPU KV-offload connector | capacity | Yes | https://github.com/vllm-project/vllm/pull/51855 · https://github.com/vllm-project/vllm/pull/37160 |

Parity gaps vs GB300: no native FP4 GEMM on SM90 (NVFP4×FP8 W4A8 MegaMoE in progress, #62603); NVFP4-KV latent + `latency_specialized` CuTe decode (1.2–1.6×) Blackwell-only; no MNNVL; DCP sparse-indexer decode (1.3–1.5×) GB300-only.

**Realistic 16×H200 expectations (ESTIMATE, anchored to the K3 measurements)**: Marlin as-is ~15–20 tok/s single-stream, ~130–170 aggregate @50–60 conc; with lever #1 ~22–30 / ~200–260. Roofline basis: ~104B active × 0.55 B/param ≈ 57 GB ÷ 16 GPUs ÷ 4.8 TB/s ≈ 0.8 ms/token HBM floor vs ~59.5 ms measured → ~98% of the single-stream step is all-to-all/launch/attention overhead — which is exactly why speculative decoding (3.14× acceptance) is the dominant lever and why GB300's NVLink-fused TP8 wins single-stream latency.

## Evidence index

- Model card (Blackwell-only, GB300 test hw, official cmd): https://huggingface.co/nvidia/Kimi-K3.1-Pro-NVFP4
- Blobs/dtype breakdown: https://huggingface.co/api/models/nvidia/Kimi-K3.1-Pro-NVFP4?blobs=true
- Architecture: https://huggingface.co/moonshotai/Kimi-K3.1-Pro/raw/main/config.json
- Official recipe + hardware keys + KV bytes: https://recipes.vllm.ai/moonshotai/Kimi-K3.1-Pro.md
- Day-0 perf blog: https://vllm.ai/blog/2026-11-04-kimi-k31-pro
- H200 OOM precedent + TP16 community path: https://github.com/vllm-project/recipes/issues/698
- K2-Thinking-NVFP4 card + config: https://huggingface.co/nvidia/Kimi-K2-Thinking-NVFP4 (+ `/resolve/main/config.json`)
- K2-Thinking FP8 H200 minimum = 16 GPUs: https://github.com/MoonshotAI/Kimi-K2/blob/main/docs/deploy_guidance.md
- FlashKDA (SM90+): https://github.com/MoonshotAI/FlashKDA · fla#852: https://github.com/fla-org/flash-linear-attention/pull/852
- K3 on 16×H200 measured (SGLang TP16+EP16): https://huggingface.co/moonshotai/Kimi-K3/discussions/136
- Official K3 H200 multi-node TP profile: https://recipes.vllm.ai/moonshotai/Kimi-K3.md?hardware=h200&strategy=multi_node_tp&nodes=2
- K3 Hopper engine-fusion perf pass: https://vllm-project.github.io/2026/09/13/kimi-k3-performance-optimization.html
- DSpark framework: https://arxiv.org/abs/2607.05147 · draft: https://huggingface.co/Inferact/Kimi-K3.1-Pro-DSpark-preview
- NVFP4→FP8-compute PR: https://github.com/vllm-project/vllm/pull/38728
- vLLM expert-offload RFCs: https://github.com/vllm-project/vllm/issues/33869 · https://github.com/vllm-project/vllm/issues/63666
- Formalization ticket (this trajectory → deterministic leaf): https://github.com/anthony-chaudhary/fak/issues/14362
- Why-fak-native comparison (this trajectory as the value example): [2026-11-12-fak-native-value-kimi-k31-pro-upstream-gaps.md](2026-11-12-fak-native-value-kimi-k31-pro-upstream-gaps.md)

## Next checkable step

1. Boot K3.1-Pro (MXFP4 base) on 16×H200 with the K3 vLLM H200 profile / SGLang TP16+EP16 Marlin and read the warmup log line for real weights/GPU + KV tokens — that single line settles whether the SM90 combination runs (nobody has published a K3.1 Hopper run).
2. On the same boot, run the DSpark-on-SM90 flag sketch at c=1/8/32/64 vs a no-spec baseline to pin acceptance length and the spec-decode crossover.
3. ~~K2-Thinking smoke~~ — dropped with the model (operator decision 2026-11-12). Instead: work the fak-native ticket set for K3.1-Pro (spine + bulk kernel ports + hot-expert offload + bench; see the [value note](2026-11-12-fak-native-value-kimi-k31-pro-upstream-gaps.md)).

---

## Addendum (2026-11-12): leaf-confirmed arithmetic — `fak serve-receipt` owns the numbers now

The capacity ESTIMATEs above are superseded on the arithmetic axis by the deterministic leaf: builtin profile `h200-141gb` (VRAM 140.4 GiB — NVML reports 143,771 MiB, so the "141 GB" label is binary-ish; SM 9.0; FP4Native false; NVLink true), weights reconciled to the exact shard sum **1,605,125,645,632 B (1,494.89 GiB)** — the 1,622 GB figure above was the provisional pre-reconciliation total. All rows pinned by `TestH200KimiNVFP4Receipt` in `internal/servereceipt/h200_test.go` (`go test ./internal/servereceipt/ -run TestH200KimiNVFP4Receipt -v`):

| Rig | Weights/GPU | Verdict | Physical fp8-KV tokens (util 0.90, 32 seqs) |
|---|---|---|---|
| 8× TP8 | 200,640,705,704 B (186.86 GiB) vs 126.36 GiB budget | **INFEASIBLE** — the expert-offload escape hatch is rejected by the printed curve: 874,857,142,784 B/forward = **17.50 s @ 50 GB/s** | — |
| 16× TP16 | 100,320,352,852 B (93.43 GiB) | **OK** — headroom 859,878 tokens over max_model_len 1,048,576 | **1,908,454** (usable pool 32.93 GiB → KV pool ≈30.20 GiB after SSM state + 2 GB activation) |
| 24× TP8×PP3 | 66,880,235,235 B (62.29 GiB) | **OK** (overkill row) | **3,821,761** |

Corrections vs the hand math above: TP16 weights/GPU is 93.43 GiB (not 92), the 0.90-util pool is 32.93 GiB usable (not ~26 GB), yielding **1.9M fp8-KV tokens** (not ~1.6M), and the non-native-FP4 warning fires on every H200 plan — Hopper rides the Marlin W4A16 fallback (vLLM modelopt docs; SGLang NVFP4-Marlin default on SM80–90), while `modelopt_mixed` on H200 stays UNVERIFIED (no public datapoint). The H200 constraint is the kernel gate, not memory: 16× fits with room to spare, but nobody has witnessed the SM90 K3.1-Pro kernel combination (step 1 above still stands).
