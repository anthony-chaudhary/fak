---
title: "fak industry scorecard — numerics"
description: "The numerics dimensions that matter in LLM serving, the current SOTA bar on each, and fak's honest position. Generated from tools/industry_scorecard.data/."
---

# numerics — the dimensions that matter, and where fak stands

[← back to the scorecard index](README.md) · part of the industry-first scorecard. Each dimension is a thing the field competes on; the fak column is honest — mostly `no-claim` gaps for a focused reuse kernel.

## Quantization & precision (`quantization`)

### ○ KV-cache quantization (FP8 / INT8 / INT4 KV) — fak: **no-claim**

*Why it matters:* Quantizing KV from BF16 to FP8 halves bytes-per-cached-token, directly doubling concurrency or context length at fixed HBM. INT8/INT4 push further. The buyer cares about the precision/accuracy/throughput trade: which formats are supported, whether attention runs in low precision (not just storage), and the measured accuracy delta. It is one of the cheapest capacity multipliers available.

- **SOTA bar:** FP8 KV halves per-token KV memory (BF16->FP8), enabling ~2-3x larger batch on H100 with FP8 KV recommended over INT8 on Hopper/Ada for accuracy; TensorRT-LLM supports FP8(E4M3)+INT8 KV and FP8 attention (use_fp8_context_fmha), vLLM supports FP8 E4M3/E5M2 KV
- **Leading systems:** TensorRT-LLM, vLLM
- **Source:** [https://blog.squeezebits.com/vllm-vs-tensorrtllm-8-kv-cache-quantization-35079](https://blog.squeezebits.com/vllm-vs-tensorrtllm-8-kv-cache-quantization-35079) (2025-01)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for the current kernel. fak's quantization work is on weight bytes (Q8/Q4_K), not on the KV cache dtype; it has no FP8/INT8/INT4 KV path and so claims none of TensorRT-LLM/vLLM's ~2-3x batch gain from FP8 KV. fak's KV story is exact eviction/reuse on an f32 cache, a different axis from KV-precision compression.
- **Trace:** fak quantizes WEIGHTS (Q8_0 SIMD lane, CLAIMS line 95; Q4_K/Q8 device GEMV witnesses line 146) but there is no FP8/INT8/INT4 KV-CACHE quantization claim. The kernel-owned KVCache and KV-decode parity (line 85) are at f32; no per-token KV-byte reduction from a quantized KV dtype is measured.

### ○ FP8 (W8A8, E4M3) accuracy retention vs bf16 reference — fak: **no-claim**

*Why it matters:* FP8 W8A8 is the production default at vLLM/SGLang/TRT-LLM on Hopper+ and is the baseline every buyer assumes is 'free.' Operators need proof it is effectively lossless before turning it on fleet-wide, because it roughly halves memory and adds 1.4-1.8x serving throughput only if quality holds.

- **SOTA bar:** FP8 W8A8 (E4M3) is effectively lossless across the full Llama-3.1 family (8B/70B/405B) over 500k+ evaluations; ~99% median accuracy recovery vs bf16. E4M3 consistently beats E5M2 for weights/activations (more mantissa, narrower range). Dynamic FP8 needs no calibration set.
- **Leading systems:** vLLM (FP8 W8A8 dynamic/static), TensorRT-LLM (ModelOpt), SGLang, Red Hat / Neural Magic llm-compressor
- **Source:** [https://arxiv.org/abs/2411.02355](https://arxiv.org/abs/2411.02355) (2025-08)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a reuse kernel. fak has no FP8 (W8A8/E4M3) path at all — its compute HAL is f32-only and its sole quant rung is Q8_0 weight-narrowing on CPU SIMD/GPU. No accuracy-retention-vs-bf16 study exists, so fak makes no claim. Not a measurable gap fak should chase: FP8 lossless retention is a model-numerics property of vLLM/TensorRT-LLM, orthogonal to fak's prefix-reuse/adjudication differentiator.
- **Trace:** none — CLAIMS.md: compute HAL is f32-only; the only narrowing fak ships is Q8_0 (8-bit), and BENCHMARK-AUTHORITY.md carries no FP8 or accuracy-recovery number

### ○ INT8 W8A8 (SmoothQuant) accuracy under activation outliers — fak: **no-claim**

*Why it matters:* INT8 W8A8 runs on far more hardware than FP8 (no FP8 tensor cores required) and dominates throughput in async continuous-batching serving. Its hard problem is activation outliers; how well a system tames them (SmoothQuant migration, per-channel/per-token dynamic scales) determines whether INT8 is viable for a given model.

- **SOTA bar:** Well-tuned INT8 W8A8 holds 1-3% per-task accuracy degradation vs bf16 across the Llama-3.1 family using SmoothQuant + dynamic per-token activation scales; ~1.56x speedup, ~2x memory cut with negligible loss on OPT/BLOOM/LLaMA/Mixtral.
- **Leading systems:** SmoothQuant (MIT HAN Lab), vLLM INT8 W8A8, TensorRT-LLM, llm-compressor
- **Source:** [https://arxiv.org/abs/2411.02355](https://arxiv.org/abs/2411.02355) (2025-08)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a reuse kernel. fak's int8/Q8_0 SIMD lane (in-flight) is weight-only and proven argmax-exact against its own f32 oracle, NOT an activation-quantized W8A8 path with SmoothQuant. fak has no per-token activation scaling, no outlier-suppression study, and no per-task degradation measurement — it makes no SmoothQuant accuracy claim.
- **Trace:** none — fak's int8 lane is Q8_0 WEIGHT-only (W8A16-style), argmax-exact vs the f32 oracle; there is no W8A8 activation-quant path and no SmoothQuant outlier-handling, so no accuracy-under-outliers number exists

### ○ INT4 weight-only (W4A16: AWQ, GPTQ) accuracy and calibration sensitivity — fak: **no-claim**

*Why it matters:* W4A16 is the most cost-efficient choice for latency/memory-bound and synchronous single-stream serving, and is the dominant format for consumer/edge deployment. Buyers must know the accuracy floor and how sensitive it is to calibration data and algorithm choice (AWQ activation-aware vs GPTQ second-order), since results flip with implementation and calibration set size.

- **SOTA bar:** W4A16 INT4 keeps accuracy degradation in the negligible-to-moderate range (<4% on commonsense QA / math) and rivals 8-bit quantization; AWQ generally beats GPTQ on perplexity/throughput but a well-tuned GPTQ with a large calibration set can be more accurate on real-world tasks. Calibration set size materially shifts retention.
- **Leading systems:** AWQ (LMDeploy, vLLM), GPTQ (vLLM, AutoGPTQ), TensorRT-LLM W4A16
- **Source:** [https://arxiv.org/abs/2411.02355](https://arxiv.org/abs/2411.02355) (2025-08)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a reuse kernel. fak has no INT4 weight-only path (no AWQ, no GPTQ, no calibration-set tooling); its lowest precision is 8-bit Q8_0. Note BENCHMARK-AUTHORITY does report serving a q4_k_m GGUF (Qwen3.6-27B) but at 0.9 tok/s with only 2-token parity before drift — that is a GGUF-load capability, not an INT4-accuracy-retention claim, so no W4A16 accuracy number is asserted.
- **Trace:** none — fak ships only Q8_0 (8-bit weight) and f32; no W4A16 / INT4 / AWQ / GPTQ path and no calibration-sensitivity study in CLAIMS.md or BENCHMARK-AUTHORITY.md

### ○ 4-bit microscaling float (NVFP4 / MXFP4 / FP4) accuracy vs FP8 on Blackwell — fak: **no-claim**

*Why it matters:* FP4 on Blackwell doubles peak throughput vs FP8 and is the new frontier format MLPerf submissions now use for Llama-3.1-405B. The differentiator is NVFP4 (E4M3 per-16 block scale, FP32 second-level) vs MXFP4 (E8M0 per-32 power-of-two scale): block size and scale precision drive whether 4-bit holds MMLU/perplexity near FP8.

- **SOTA bar:** NVFP4 quantizing DeepSeek-R1 from FP8 to FP4 drops MMLU only 0.1 point (90.8 -> 90.7); NVFP4 reports ~88% lower quantization error than MXFP4 power-of-two scaling and MXFP4 needs up to 36% more tokens to reach comparable loss. FP4 gives 2x peak throughput vs FP8 while meeting MLPerf accuracy targets.
- **Leading systems:** NVIDIA Blackwell + TensorRT-LLM (ModelOpt NVFP4), vLLM NVFP4, llama.cpp (NVFP4/MXFP4)
- **Source:** [https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/) (2025-04)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a reuse kernel. fak has no microscaling 4-bit float format, no NVFP4/MXFP4 kernels, and no Blackwell tensor-core path. This is squarely vendor-stack (TensorRT-LLM/ModelOpt) territory; fak asserts nothing and should not — it is orthogonal to its prefix-reuse differentiator.
- **Trace:** none — no 4-bit microscaling-float path (NVFP4/MXFP4/FP4) anywhere in fak; compute HAL is f32-only, quant rung is Q8_0, no Blackwell-class hardware path

### ○ Full 4-bit (W4A4 + KV4) via rotation/outlier removal — fak: **no-claim**

*Why it matters:* The aggressive end of the design space: quantizing weights, activations, AND KV to 4 bits unlocks the largest memory/throughput wins but is gated entirely by activation-outlier handling. Hadamard/learned-rotation methods (QuaRot, SpinQuant) define how close W4A4 can get to full precision, which bounds what an operator can push without quality collapse.

- **SOTA bar:** QuaRot W4A4KV4 on Llama-2-70B loses at most 0.47 WikiText-2 perplexity and retains 99% of zero-shot accuracy (~3.5pt avg gap); SpinQuant narrows the W4A4KV4 gap to ~1.9 points (avg 64.0 on Llama-2-7B), with up to 3.33x prefill speedup / 3.89x decode memory saving.
- **Leading systems:** QuaRot (Hadamard rotation), SpinQuant (learned rotation), QServe W4A8KV4
- **Source:** [https://arxiv.org/html/2404.00456v1](https://arxiv.org/html/2404.00456v1) (2024-04)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a reuse kernel. Full 4-bit (W4A4+KV4) via rotation/outlier removal is an aggressive model-numerics research line; fak has none of it (no activation quant, no KV quant, no rotation). It makes no perplexity-delta or zero-shot-retention claim here.
- **Trace:** none — no W4A4, no KV4, no Hadamard/learned rotation (QuaRot/SpinQuant) path; fak is f32 compute with an 8-bit weight rung only

### ○ Quantization granularity (per-tensor / per-channel / per-group / microscaling block size) — fak: **no-claim**

*Why it matters:* Granularity is the single biggest accuracy lever at a fixed bit-width and the hardest cost/quality knob: finer scales (per-channel, group-128, MX block-32, two-level) cut quantization error but add scale storage and GEMM overhead. It also determines hardware compatibility (MX needs block-32 tensor-core support). A scorecard that ignores granularity can't compare two 'INT4' or 'FP4' systems honestly.

- **SOTA bar:** Per-block (block-32) microscaling beats per-tensor/per-channel; measured SNR gains of 3.0-3.4 dB over group-128 and 9.2-9.4 dB over per-tensor for two-level microscaling. Common group sizes 32/64/128; group-128 is the de-facto 4-bit weight-only default. NVFP4's block-16 + FP32 scale is finer than MXFP4's block-32 power-of-two.
- **Leading systems:** MX formats (OCP MXFP4/MXFP8, block-32), NVFP4 (block-16, FP32 second-level), group-128 INT4 (AWQ/GPTQ), MOSS two-level microscaling
- **Source:** [https://arxiv.org/html/2511.05811v2](https://arxiv.org/html/2511.05811v2) (2025-11)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a reuse kernel. fak's only granularity is whatever Q8_0/GGUF gives it; it has authored no microscaling block-size study and no SNR-vs-granularity numbers. The MX/NVFP4 granularity question is a format-design axis owned by the quant-format authors, not fak — no claim.
- **Trace:** none — fak consumes Q8_0 (the GGUF 8-bit block scheme) but has shipped no per-tensor/per-channel/per-group/microscaling granularity comparison or SNR measurement

### ○ Accuracy-constrained throughput on a neutral benchmark (MLPerf Inference) — fak: **no-claim**

*Why it matters:* MLPerf Inference is the only vendor-neutral, audited benchmark that ties quantized throughput to a fixed accuracy bar (99% or 99.9% of FP reference), preventing quality-for-speed cheating. It is the buyer's apples-to-apples ground truth: a number only counts if it clears the accuracy gate, which is exactly how operators should read all quantization claims.

- **SOTA bar:** MLPerf enforces 99%/99.9%-of-FP-reference accuracy gates per benchmark; under that gate Blackwell GB200 NVL72 delivered up to 3.4x higher per-GPU Llama-3.1-405B throughput vs an 8-GPU H200 system using FP4, with FP4 meeting the same accuracy target as FP8.
- **Leading systems:** NVIDIA GB200 NVL72 / B200 (TensorRT-LLM, NVFP4), AMD MI300X/MI355X (ROCm, FP8), MLPerf Inference v5.0/v5.1 submitters
- **Source:** [https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/) (2025-04)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a reuse kernel. fak has never submitted to MLPerf and runs no accuracy-constrained throughput benchmark. Its closest real number — 1085.6 vs 1451.6 tok/s vs raw SGLang on 8-GPU datacenter server (it TRAILS, the gateway tax) — is an ungated throughput race, not an MLPerf accuracy-gated result, so no MLPerf claim is made. Neutral accuracy-gated throughput is a vendor/submitter axis far from fak's reuse/adjudication value.
- **Trace:** none — fak has no MLPerf Inference submission and no accuracy-gated throughput run; its only live concurrent head-to-head (served-throughput-vs-sglang) is a raw tok/s race fak TRAILS, with no 99%/99.9%-accuracy gate applied

### ○ Quantization format & low-precision datatype coverage — fak: **no-claim**

*Why it matters:* Low precision is how a frontier model fits and runs fast: FP8/FP4 on tensor cores, plus weight-only INT4 (AWQ/GPTQ) and KV-cache quantization. The relevant axes are which formats the engine supports, which the target hardware accelerates natively, and how much memory/throughput each buys at what accuracy. Quantization coverage tied to hardware generation (e.g. NVFP4 on Blackwell, FP8 on Hopper/MI300X) directly sets the cost-per-token frontier.

- **SOTA bar:** NVFP4 (4-bit FP, E2M1 + dual-level scaling) has native Blackwell tensor-core acceleration, reportedly matching FP8 accuracy at ~2.3x higher throughput than weight-only 4-bit (AWQ); vLLM/TensorRT-LLM support FP8 W8A8 (Hopper/Ada/MI300X), NVFP4/MXFP4 W4A4 on Blackwell, plus INT4 AWQ/GPTQ and KV-cache quantization. llama.cpp GGUF spans 1.58-bit to 8-bit integer plus fp16/bf16.
- **Leading systems:** TensorRT-LLM, vLLM, NVIDIA Blackwell
- **Source:** [https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/) (2025-01-01)
- **fak:** no-claim — no number (in-flight)
- **fak note:** REAL but partial gap. fak LOADS GGUF Q8_0/Q4_K and witnesses Q8 device GEMM (Vulkan 24.6 tok/s 1.49x vs f32; datacenter GPU k_q8_gemm cosine~1.0), and has an in-flight AVX2/AVX-512 int8 SIMD decode lane (~2.97x vs HF dynamic-int8, int8-simd-vs-hf row) — but its own engine still largely DEQUANTS to f32, and it has NO FP8 W8A8, NVFP4/MXFP4 W4A4, AWQ/GPTQ, or KV-cache quantization. The TensorRT-LLM/vLLM Blackwell-tensor-core NVFP4 path is entirely out of fak's reach. No shipped fak coverage-breadth number; the int8 lane is deliberately not yet [SHIPPED]. Verdict no-claim on coverage; the speed sub-claim lives in int8-simd-vs-hf.
- **Trace:** CLAIMS: GGUF Q8_0/Q4_K loaded but DEQUANT-to-f32 on own engine; int8/Q8_0 SIMD lane is in-flight (not [SHIPPED]); no FP8/NVFP4/MXFP4/AWQ/GPTQ

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
- **Trace:** 498a4ab · docs/notes/GLM52-PURE-KERNEL-ON-GPU-DGX-A100-2026-06-21.md

