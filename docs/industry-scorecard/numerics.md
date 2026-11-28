---
title: "fak industry scorecard — numerics"
description: "The numerics dimensions that matter in LLM serving, the current SOTA bar on each, and fak's honest position. Generated from tools/industry_scorecard.data/."
---

# numerics — the dimensions that matter, and where fak stands

[← back to the scorecard index](README.md) · part of the industry-first scorecard. Each dimension is a thing the field competes on; the fak column is honest — mostly `no-claim` gaps for a focused reuse kernel.

## Quantization & precision (`quantization`)

### ≈ KV-cache quantization (FP8 / INT8 / INT4 KV) — fak: **parity**

*Why it matters:* Quantizing KV from BF16 to FP8 halves bytes-per-cached-token, directly doubling concurrency or context length at fixed HBM. INT8/INT4 push further. The buyer cares about the precision/accuracy/throughput trade: which formats are supported, whether attention runs in low precision (not just storage), and the measured accuracy delta. It is one of the cheapest capacity multipliers available.

- **SOTA bar:** FP8 (e4m3) KV-cache quantization is the cross-GPU default (~46% real memory reduction / 54% of BF16 per-token cost in vLLM Apr 2026, +14.9% output throughput, recovering 97-98% long-context AUC at 128k). On Blackwell, vLLM, SGLang, and FlashInfer also ship 4-bit FP4 KV (4x reduction vs FP16), while local runtimes (llama.cpp #26989, Ollama) use integer q4_0/iq4_nl KV.
- **Leading systems:** FP8 e4m3 KV (vLLM/SGLang production, Apr 2026), FP4 KV (vLLM/SGLang/FlashInfer on Blackwell, 4x vs FP16), q4_0 / iq4_nl KV (llama.cpp #26989 / Ollama), INT4 / KIVI KV (research: ~75% reduction)
- **Source:** [https://vllm-project.github.io/2026/04/22/fp8-kvcache.html](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html) (2026-04)
- **fak:** parity — 4 x KV memory reduction (KV dtype) (shipped)
- **fak note:** HW-WITNESSED on RTX 5080 16 GB (sm_120): fak ships auto-FP8 global KV (>16K, 2x vs FP16, 32K mean KL 0.004225 nats), opt-in NVFP4 block-16 global KV (4x vs FP16, 128k all-resident with 0 GiB CPU spill, 4.21x prefill 1,190 vs 283 tok/s, +21.7% decode 16.9 vs 13.9 tok/s, exact 128k needle retrieval, 4/4 greedy oracle pass), and opt-in asymmetric fp8k_nvfp4v (0eb366df29). Parity on 4x KV capacity reduction with Blackwell FP4 KV stacks; both low-bit modes stay opt-in.
- **Trace:** eb68a8cf8a · docs/benchmarks/published/rtx5080-inkling2-v060/perf-receipt-v2.json · README.md (128k NVFP4 KV on RTX 5080); docs/benchmarks/published/rtx5080-inkling2-v060/RESULTS.md (32K auto-FP8 KV); commits 7b462f9b5e & 0eb366df29 (asymmetric fp8k_nvfp4v KV)

### ○ FP8 (W8A8, E4M3) accuracy retention vs bf16 reference — fak: **no-claim**

*Why it matters:* FP8 W8A8 is the production default at vLLM/SGLang/TRT-LLM on Hopper+ and is the baseline every buyer assumes is 'free.' Operators need proof it is effectively lossless before turning it on fleet-wide, because it roughly halves memory and adds 1.4-1.8x serving throughput only if quality holds.

- **SOTA bar:** FP8 (W8A8, E4M3) weight+activation quantization is effectively lossless: ~99.9% accuracy recovery on HumanEval and <=1% degradation vs BF16 across all model scales, including Llama 3.1 405B fully recovering accuracy.
- **Leading systems:** FP8 W8A8 via LLM Compressor (vLLM), NVIDIA TensorRT Model Optimizer FP8, AMD Quark FP8
- **Source:** [https://developers.redhat.com/articles/2024/10/17/we-ran-over-half-million-evaluations-quantized-llms](https://developers.redhat.com/articles/2024/10/17/we-ran-over-half-million-evaluations-quantized-llms) (2024-10)
- **fak:** no-claim — no number (in-flight)
- **fak note:** PARTIAL PRIMITIVES, NO W8A8 EVAL: fak ships FP8 global KV cache (>16K auto-FP8, 32K mean KL 0.004225 nats) and fp8_block32_e8m0 CUDA weight kernels plus SM120 NVFP4 W4A4 prefill (not f32-only), but still lacks a full FP8 W8A8 weight+activation serving path and HumanEval/MMLU accuracy-retention comparison vs BF16. Honest no-claim.
- **Trace:** docs/notes/PACKING-COVERAGE-MATRIX-2026-11-03.md (fp8_block32_e8m0 CUDA kernel); docs/benchmarks/published/rtx5080-inkling2-v060/RESULTS.md (32K auto-FP8 KV); no W8A8 HumanEval eval

### ○ INT8 W8A8 (SmoothQuant) accuracy under activation outliers — fak: **no-claim**

*Why it matters:* INT8 W8A8 runs on far more hardware than FP8 (no FP8 tensor cores required) and dominates throughput in async continuous-batching serving. Its hard problem is activation outliers; how well a system tames them (SmoothQuant migration, per-channel/per-token dynamic scales) determines whether INT8 is viable for a given model.

- **SOTA bar:** INT8 W8A8 (per-channel static weight + per-token dynamic activation, optionally SmoothQuant) recovers ~99% of BF16 accuracy, i.e. ~1% average loss; vLLM/LLM Compressor report negligible Open-LLM-leaderboard drop.
- **Leading systems:** INT8 W8A8 SmoothQuant via LLM Compressor (vLLM), SmoothQuant (Xiao et al.)
- **Source:** [https://developers.redhat.com/articles/2025/03/03/deployment-ready-reasoning-quantized-deepseek-r1-models](https://developers.redhat.com/articles/2025/03/03/deployment-ready-reasoning-quantized-deepseek-r1-models) (2025-03)
- **fak:** no-claim — no number (stub)
- **fak note:** Weight+activation W8A8 SmoothQuant eval unmeasured: fak now quantizes activations to Q8_1/int8 on-the-fly in CUDA MMQ/MMVQ and Inkling-2 prefill/decode paths alongside native Q8_0 weights across backends, but has no SmoothQuant migration or BF16 per-task accuracy degradation benchmark. Honest no-claim.
- **Trace:** docs/notes/PACKING-COVERAGE-MATRIX-2026-11-03.md (native Q8_0 CUDA/Vulkan/CPU/Metal + Q8_1 activations for CUDA decode/prefill); no SmoothQuant W8A8 Open-LLM benchmark

### ○ INT4 weight-only (W4A16: AWQ, GPTQ) accuracy and calibration sensitivity — fak: **no-claim**

*Why it matters:* W4A16 is the most cost-efficient choice for latency/memory-bound and synchronous single-stream serving, and is the dominant format for consumer/edge deployment. Buyers must know the accuracy floor and how sensitive it is to calibration data and algorithm choice (AWQ activation-aware vs GPTQ second-order), since results flip with implementation and calibration set size.

- **SOTA bar:** INT4 weight-only (W4A16) via AWQ or GPTQ keeps accuracy degradation within negligible-to-moderate range (<4% typical, often <1-2% on HumanEval); AWQ is the de-facto best-practice INT4 format for vLLM in 2026 and generally edges out GPTQ on perplexity.
- **Leading systems:** AWQ (AutoAWQ) via vLLM, GPTQ / GPTQModel + Marlin (vLLM), W4A16 LLM Compressor
- **Source:** [https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) (2026-01)
- **fak:** no-claim — no number (in-flight)
- **fak note:** PARTIAL PRIMITIVES, NO W4A16 CALIBRATION BENCH: fak ships native CUDA/Vulkan/Metal/CPU 4-bit GGUF kernels (Q4_K MMQ default on sm_80+, IQ1_M..IQ4_NL, NVFP4) and host-side AWQ/GPTQ packing, serving Inkling-2-Mini (26B) and Qwen3.8-27B Q4_K_M on CUDA; however, GPU device-resident AWQ/GPTQ kernels and W4A16 HumanEval/perplexity calibration-sensitivity benchmarks remain unbuilt/unmeasured.
- **Trace:** docs/notes/PACKING-COVERAGE-MATRIX-2026-11-03.md (native Q4_K/Q3_K/IQ* CUDA/Vulkan/Metal/CPU, host AWQ/GPTQ); docs/benchmarks/receipts/cuda-prefill-sm120-20261106/RESULTS.md (Qwen3.8-27B Q4_K_M MMQ)

### ○ 4-bit microscaling float (NVFP4 / MXFP4 / FP4) accuracy vs FP8 on Blackwell — fak: **no-claim**

*Why it matters:* FP4 on Blackwell doubles peak throughput vs FP8 and is the new frontier format MLPerf submissions now use for Llama-3.1-405B. The differentiator is NVFP4 (E4M3 per-16 block scale, FP32 second-level) vs MXFP4 (E8M0 per-32 power-of-two scale): block size and scale precision drive whether 4-bit holds MMLU/perplexity near FP8.

- **SOTA bar:** NVFP4 (4-bit, 16-value blocks, FP8 E4M3 fractional block scales) now achieves <=1% accuracy drop vs FP8 on large reasoning models (DeepSeek-R1-0528: MMLU-PRO 85->84, GPQA 81->80) and ~99% recovery vs BF16 for 70B-235B models, at ~2.3x higher throughput than weight-only 4-bit; clearly the SOTA 4-bit format on Blackwell, well ahead of MXFP4 (32-value, power-of-two scales).
- **Leading systems:** NVFP4 via NVIDIA TensorRT Model Optimizer + LLM Compressor (vLLM, Blackwell B200/GB200), NVFP4 (vLLM Marlin)
- **Source:** [https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) (2025-09)
- **fak:** no-claim — no number (in-flight)
- **fak note:** HONEST NO-CLAIM on downstream MMLU-PRO/GPQA accuracy vs FP8: fak ships native SM120 NVFP4 W4A4 grouped prefill on Inkling-2-Mini-UD-Q3_K_XL (default on sm_120f/a at 27355b23dd) and passed its RTX 5080 KL gate vs per-token step (mean KL 0.003728 nats, max 0.105, 84/85 same-top, 4/4 llama.cpp CPU greedy oracle pass), but has NOT measured MMLU-PRO or GPQA accuracy retention against an FP8 baseline. Witnessed NVFP4 W4A4 prefill throughput (5861.0 vs realugbun/llama.cpp 3012.4 tok/s on UD-Q3_K_XL) lives in single-stream-prefill-gpu.
- **Trace:** 27355b23dd · docs/benchmarks/published/aegis-inkling2-prefill-nvfp4-w4a4/RESULTS.md · docs/benchmarks/published/aegis-inkling2-prefill-nvfp4-w4a4/RESULTS.md

### ○ Full 4-bit (W4A4 + KV4) via rotation/outlier removal — fak: **no-claim**

*Why it matters:* The aggressive end of the design space: quantizing weights, activations, AND KV to 4 bits unlocks the largest memory/throughput wins but is gated entirely by activation-outlier handling. Hadamard/learned-rotation methods (QuaRot, SpinQuant) define how close W4A4 can get to full precision, which bounds what an operator can push without quality collapse.

- **SOTA bar:** Learned-rotation methods now lead 4-bit weight+activation: SpinQuant (ICLR 2025) closes the gap to full precision by up to 45.1% relative to QuaRot on hard-to-quantize Llama-3 8B, and 2025-2026 successors (Qronos, ResQ PCA-rotations, TurboQuant for KV cache, WUSH adaptive transforms) advance further; Hadamard/learned rotations are the standard outlier-removal primitive feeding NVFP4/W4A4.
- **Leading systems:** SpinQuant (learned rotations), Qronos (2025), ResQ PCA-rotation (2025), TurboQuant (2026, KV cache), QuaRot (Hadamard baseline)
- **Source:** [https://arxiv.org/abs/2405.16406](https://arxiv.org/abs/2405.16406) (2025-04)
- **fak:** no-claim — no number (in-flight)
- **fak note:** ROTATION PRIMITIVE SHIPPED, END-TO-END PPL GAP REMAINS: fak now ships SM120 NVFP4 W4A4 prefill, 4-bit NVFP4 KV cache, and FWHT16 Hadamard-rotated NVFP4-V values (0eb366df29, internal/compute/block_diagonal_wht.go), but has no QuaRot/SpinQuant-style full-model WikiText-2 perplexity or zero-shot retention benchmark vs FP16. Honest no-claim.
- **Trace:** 0eb366df29 · commit 0eb366df29 & internal/compute/block_diagonal_wht.go (FWHT16-rotated NVFP4-V KV); docs/benchmarks/published/aegis-inkling2-prefill-nvfp4-w4a4/RESULTS.md (SM120 NVFP4 W4A4 prefill)

### ○ Quantization granularity (per-tensor / per-channel / per-group / microscaling block size) — fak: **no-claim**

*Why it matters:* Granularity is the single biggest accuracy lever at a fixed bit-width and the hardest cost/quality knob: finer scales (per-channel, group-128, MX block-32, two-level) cut quantization error but add scale storage and GEMM overhead. It also determines hardware compatibility (MX needs block-32 tensor-core support). A scorecard that ignores granularity can't compare two 'INT4' or 'FP4' systems honestly.

- **SOTA bar:** Per-block (block-32) microscaling beats per-tensor/per-channel; measured SNR gains of 3.0-3.4 dB over group-128 and 9.2-9.4 dB over per-tensor for two-level microscaling. Common group sizes 32/64/128; group-128 is the de-facto 4-bit weight-only default. NVFP4's block-16 + FP32 scale is finer than MXFP4's block-32 power-of-two.
- **Leading systems:** MX formats (OCP MXFP4/MXFP8, block-32), NVFP4 (block-16, FP32 second-level), group-128 INT4 (AWQ/GPTQ), MOSS two-level microscaling
- **Source:** [https://arxiv.org/html/2511.05811v2](https://arxiv.org/html/2511.05811v2) (2025-11)
- **fak:** no-claim — no number (in-flight)
- **fak note:** BLOCK GRANULARITIES SHIPPED, SNR ABLATION UNMEASURED: fak executes two-level block-16 NVFP4 (E4M3 + FP32), block-32 E8M0 MXFP4/FP8, and GGUF Q/IQ super-blocks plus INT8 blk16 KV, but has authored no controlled SNR-vs-granularity dB study against MOSS. Honest no-claim.
- **Trace:** docs/notes/PACKING-COVERAGE-MATRIX-2026-11-03.md (block-16 NVFP4, block-32 E8M0 MXFP4/FP8, GGUF super-blocks); commit 0eb366df29 (INT8-K blk16 + FWHT16 NVFP4-V)

### ○ Accuracy-constrained throughput on a neutral benchmark (MLPerf Inference) — fak: **no-claim**

*Why it matters:* MLPerf Inference is the only vendor-neutral, audited benchmark that ties quantized throughput to a fixed accuracy bar (99% or 99.9% of FP reference), preventing quality-for-speed cheating. It is the buyer's apples-to-apples ground truth: a number only counts if it clears the accuracy gate, which is exactly how operators should read all quantization claims.

- **SOTA bar:** Under MLPerf's accuracy- and latency-constrained server scenario (TTFT and TPS/user SLOs enforced at p99), MLPerf Inference v6.0 (2026-03-30) DeepSeek-R1 reaches 8,064 tokens/second/GPU on GB300 NVL72 (vs 9,821 tok/s/GPU unconstrained offline); DeepSeek-R1's server SLO is 2s TTFT with a 12.5 tok/s/user p99 target while holding the required accuracy bar, and the GB300 server result is ~25% higher per-GPU than GB200 and up to 2.7x the GB300 debut via software.
- **Leading systems:** NVIDIA GB300 NVL72 (Blackwell Ultra, TensorRT-LLM/Dynamo), NVIDIA GB200 NVL72 (Blackwell), CoreWeave / Nebius GB300 NVL72 submissions
- **Source:** [https://www.storagereview.com/news/nvidia-sets-mlperf-inference-v6-0-records-with-blackwell-ultra-platform](https://www.storagereview.com/news/nvidia-sets-mlperf-inference-v6-0-records-with-blackwell-ultra-platform) (2026-04-01)
- **fak:** no-claim — no number (stub)
- **fak note:** EXCLUDE / out-of-scope for fak-native's mid-market 1-8 GPU target. OUT OF SCOPE BY DESIGN for a reuse kernel. The accuracy-constrained number is now MLPerf v6.0's (2026-03-30) server scenario, carried by DeepSeek-R1 671B MoE (8,064 tok/s/GPU on GB300 NVL72 under a 2 s TTFT / 12.5 tok/s/user p99 SLO while holding the accuracy bar), superseding v5.x's Llama-3.1-405B framing. fak has never submitted to MLPerf and runs no accuracy-constrained throughput benchmark. Its closest real number — 1085.6 vs 1451.6 tok/s vs raw SGLang on 8-GPU datacenter server (it TRAILS, the gateway tax) — is an ungated throughput race, not an MLPerf accuracy-gated result, so no MLPerf claim is made. fak does NOT intend an own-engine MLPerf-shaped DeepSeek-R1 number; on a 671B MoE it fronts SGLang/llama.cpp, and its in-kernel DeepSeek-class-MoE path is the GLM-5.2 work (#1010, #1026; single device, commit 498a4ab). See max-model-and-moe-coverage.
- **Trace:** none — fak has no MLPerf Inference submission and no accuracy-gated throughput run; its only live concurrent head-to-head (served-throughput-vs-sglang) is a raw tok/s race fak TRAILS, with no 99%/99.9%-accuracy gate applied

### ○ Quantization format & low-precision datatype coverage — fak: **no-claim**

*Why it matters:* Low precision is how a frontier model fits and runs fast: FP8/FP4 on tensor cores, plus weight-only INT4 (AWQ/GPTQ) and KV-cache quantization. The relevant axes are which formats the engine supports, which the target hardware accelerates natively, and how much memory/throughput each buys at what accuracy. Quantization coverage tied to hardware generation (e.g. NVFP4 on Blackwell, FP8 on Hopper/MI300X) directly sets the cost-per-token frontier.

- **SOTA bar:** Reference serving stacks now cover the full ladder: vLLM (2026) supports FP8 (W8A8), INT8 (W8A8 + W4A8), INT4 (W4A16), NVFP4 and MXFP4 (via Marlin), AWQ, GPTQ, BitsAndBytes (4/8-bit), AMD Quark, and online runtime quantization, across Turing->Hopper->Blackwell plus x86/Arm CPU for INT8.
- **Leading systems:** vLLM quantization (LLM Compressor, Marlin, AutoAWQ, GPTQModel, bitsandbytes, Quark), TensorRT Model Optimizer (FP8/INT4/NVFP4)
- **Source:** [https://docs.vllm.ai/en/latest/features/quantization/](https://docs.vllm.ai/en/latest/features/quantization/) (2026-01)
- **fak:** no-claim — no number (in-flight)
- **fak note:** PARTIAL GAP: fak now ships native SM120 NVFP4 W4A4 prefill, FP8/NVFP4 KV cache, and resident Q3_K/Q4_K/Q8_0/IQ* CUDA & Metal kernels, but lacks FP8 W8A8/AWQ/GPTQ weight paths and Hopper FP8 GEMM parity. No single coverage-breadth scalar claimed.
- **Trace:** README.md; docs/benchmarks/published/aegis-inkling2-prefill-nvfp4-w4a4/RESULTS.md; docs/benchmarks/published/rtx5080-inkling2-v060/RESULTS.md: native NVFP4 W4A4 grouped prefill (sm_120f), FP8 & NVFP4 global KV cache, plus Q3_K/Q4_K/Q8_0/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_NL resident CUDA & Metal kernels.

### ▲ int8 / Q8_0 SIMD decode throughput vs the same-rung int8 peer — fak: **lead**

*Why it matters:* The execution speed of an int8-quantized model on CPU SIMD (AVX2/AVX-512) decides local-inference viability. fak's hand-written int8 GEMM beats the standard HuggingFace dynamic-int8 reference.

- **SOTA bar:** HuggingFace dynamic-int8 is the standard same-rung CPU int8 reference (1×); llama.cpp Q8_0 CPU is the harder peer (~parity).
- **Leading systems:** HuggingFace dynamic-int8, llama.cpp Q8_0
- **Source:** [https://huggingface.co/docs/transformers/main/quantization/overview](https://huggingface.co/docs/transformers/main/quantization/overview) (2026-06)
- **fak:** lead — 2.97 × vs HF int8 (in-flight)
- **fak note:** The int8/Q8_0 SIMD lane (AVX2/AVX-512 Go assembly) is the ACTIVE IN-FLIGHT increment — deliberately NOT a [SHIPPED] CLAIMS row until the lane commits. vs the same-rung HF dynamic-int8 peer fak is ~2.97× faster (argmax-exact); vs llama.cpp Q8_0 CPU it is near-parity (~0.90×, 7.7 vs 6.9 ms/tok), a separate harder peer.
- **Trace:** MODEL-BASELINE-RESULTS.md (Act 3) · experiments/model-baseline/INT8-RUNG-VERIFICATION.md

## Numerical correctness (`numerical-correctness`)

### ≈ Numerical-correctness error metric vs reference (perplexity, KL-divergence, recovery %) — fak: **parity**

*Why it matters:* Aggregate task scores (MMLU) hide quantization distortion because rare tokens barely move perplexity. Serious evaluation reports the deviation of the quantized model's token distribution from the bf16/fp16 reference (KL-divergence) plus generation-similarity, not just downstream accuracy. The choice of error metric is itself a differentiator buyers must scrutinize.

- **SOTA bar:** Best practice pairs perplexity with KL-divergence against the FP16 baseline (llama.cpp emits both); KLD captures per-token distribution shift perplexity hides. Empirically: 5-bit GGUF is a strong near-lossless trade-off, 3-bit shows clear degradation; imatrix is essential below 3-bit, marginal at Q4_K_M+. Generation text-similarity is reported as a supplementary correctness check.
- **Leading systems:** llama.cpp (perplexity + KL-divergence tooling), Neural Magic / vLLM recovery-% methodology, ModelOpt accuracy harness
- **Source:** [https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) (2026-01)
- **fak:** parity — no number (shipped)
- **fak note:** Embedding exact, per-layer cos=1.000000, final-logits max|Δ|≈4.4e-5, KV-decode + KV-quarantine-evict token-for-token identical (max|Δ|=0) vs the HF oracle. This correctness parity is the floor under every fak speed claim — a win on speed is meaningless if the tokens differ; they do not.
- **Trace:** IN-KERNEL-MODEL-RESULTS.md · CLAIMS.md (In-kernel model)

### ○ Deterministic / bitwise-reproducible inference (batch invariance) — fak: **no-claim**

*Why it matters:* Quantized serving is system-nondeterministic: identical prompts give different outputs because reduction order changes with batch size (floating-point non-associativity in matmul/attention/norm). For eval reproducibility, RL training-inference consistency, debugging, and audit/compliance, operators increasingly require bitwise-identical outputs - a hard correctness property distinct from accuracy.

- **SOTA bar:** Batch-invariant kernels (invariant normalization, matmul, attention reductions) yield 1000/1000 bitwise-identical completions under dynamic batching, vs dozens of distinct outputs without; cost ~60% slower than default kernels. Now exposed as a vLLM batch-invariance feature.
- **Leading systems:** Thinking Machines batch-invariant-ops (on vLLM FlexAttention), vLLM batch-invariance mode, LLM-42 (deterministic speculation)
- **Source:** [https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (2025-09)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP, partially adjacent. fak DOES ship strong determinism witnesses: greedy decode is deterministic, and its batched decode is bit-identical (math.Float32bits equality) to the serial reference — the property that the cross-agent fusion preserves tokens. BUT the SOTA bar is BATCH-INVARIANCE: identical output regardless of dynamic batch SIZE/co-tenant composition (Thinking Machines / vLLM batch-invariant kernels, 1000/1000 identical). fak proves batched==serial for a FIXED shape, never tested across varying dynamic batch sizes, so it makes no batch-invariance claim. Honest verdict no-claim on the SOTA-defined dimension; the related fixed-shape bit-identity is real but a different property.
- **Trace:** CLAIMS.md: greedy decode is deterministic+input-driven and bit-identical serial-vs-batched (TestBatchedDecodeMatchesSerial, TestBatchFromPrefixMatchesIndependentPrefill, parallel matmul math.Float32bits-equal) — but these prove SAME-batch-shape determinism, not the cross-batch-SIZE invariance the SOTA bar measures

### ≈ Dense GPU compute correctness with ZERO vendor GEMM (cuBLAS-free) — fak: **parity**

*Why it matters:* Every mainstream stack depends on NVIDIA cuBLAS for its GEMMs. Running a real model's dense path on an owned, dependency-free kernel — and proving it bit-matches the vendor oracle — is a portability/auditability differentiator.

- **SOTA bar:** cuBLAS is the tuned vendor GEMM every other stack carries; matching it bit-for-bit (cosine=1.0) is parity by construction.
- **Leading systems:** cuBLAS (NVIDIA tuned vendor GEMM)
- **Source:** [https://docs.nvidia.com/cuda/cublas/](https://docs.nvidia.com/cuda/cublas/) (2026-06)
- **fak:** parity — no number (shipped)
- **fak note:** GLM-5.2's MoE/FFN experts + router + vocab head (the bulk of its params) run on fak's OWN k_q8_gemm kernel — cosine=1.000000, argmax-exact vs the CPU Q8 forward on a real datacenter GPU, with ZERO cuBLAS dependency. Honest fence: the DSA sparse-attention + DSA-KV stay host-side (the next #86/#413 slice); the dense path is pure-GPU, the sparse-attention is not yet.
- **Trace:** 498a4ab · docs/notes/GLM52-PURE-KERNEL-ON-GPU-SERVER-2026-06-21.md

