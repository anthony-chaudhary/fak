---
title: "fak industry scorecard — models"
description: "The models dimensions that matter in LLM serving, the current SOTA bar on each, and fak's honest position. Generated from tools/industry_scorecard.data/."
---

# models — the dimensions that matter, and where fak stands

[← back to the scorecard index](README.md) · part of the industry-first scorecard. Each dimension is a thing the field competes on; the fak column is honest — mostly `no-claim` gaps for a focused reuse kernel.

## Hardware coverage (`hardware-coverage`)

### ▼ Hardware / format coverage and portability of quantized kernels — fak: **trails**

*Why it matters:* A quantization format is only useful where the serving stack has fast kernels for it. FP8 needs Hopper+/Ada or MI300X; NVFP4 is Blackwell-exclusive; MXFP4 is the cross-vendor OCP standard but kernel support lags. Buyers running heterogeneous fleets (NVIDIA + AMD + edge) must know which precision actually accelerates on their silicon vs silently falls back.

- **SOTA bar:** FP8 W8A8 is hardware-accelerated on NVIDIA Hopper/Ada/Blackwell and AMD MI300X (hipBLASLt FP8 GEMM + aiter attention); NVFP4 is Blackwell-only; MXFP4 acceleration on MI300X/MI355X is on the roadmap, not yet broadly shipped. GGUF k-quants give the widest CPU/consumer-GPU portability.
- **Leading systems:** vLLM (FP8 on Hopper/Ada/Blackwell + MI300X via hipBLASLt/aiter), SGLang (FP8 MI300X), TensorRT-LLM (NVFP4 Blackwell), llama.cpp (broad CPU/GPU GGUF + FP4)
- **Source:** [https://docs.vllm.ai/en/stable/features/quantization/fp8/](https://docs.vllm.ai/en/stable/features/quantization/fp8/) (2026-01)
- **fak:** trails — no number (shipped)
- **fak note:** Disclosed coverage gap where fak still TRAILS SOTA engines: fak now ships native SM120 NVFP4 W4A4 prefill, FP8/NVFP4/fp8k_nvfp4v KV cache, Q4_K MMQ (sm_80+), and native CUDA/Vulkan/Metal/CPU kernels for Q2_K..Q8_0 and IQ1_M..IQ4_NL (plus host AWQ/GPTQ). However, it lacks Hopper/Ada/MI300X FP8 W8A8 GEMMs, GPU AWQ/GPTQ device kernels, and production MXFP4 MoE coverage that vLLM, SGLang, TensorRT-LLM, and llama.cpp accelerate across targets.
- **Trace:** 27355b23dd · docs/benchmarks/published/rtx5080-inkling2-v060/perf-receipt-v2.json · docs/notes/PACKING-COVERAGE-MATRIX-2026-11-03.md; docs/benchmarks/published/rtx5080-inkling2-v060/RESULTS.md; docs/benchmarks/published/aegis-inkling2-prefill-nvfp4-w4a4/RESULTS.md; docs/benchmarks/receipts/cuda-prefill-sm120-20261106/RESULTS.md

### ▼ Hardware backend breadth (NVIDIA, AMD, Intel, TPU, AWS, Apple, CPU) — fak: **trails**

*Why it matters:* Vendor lock-in to a single accelerator is a top procurement risk. The buyer axis is how many silicon families a stack runs on with maintained, performant paths: NVIDIA Hopper/Blackwell, AMD ROCm (MI300X/MI355X), Intel Gaudi/XPU, Google TPU, AWS Inferentia/Trainium, Apple Metal, and CPU. Breadth determines portability and negotiating leverage; depth (kernel maturity per backend) determines whether the non-NVIDIA path is actually usable in production.

- **SOTA bar:** vLLM runs on NVIDIA, AMD ROCm (MI300X/MI325X/MI350X/MI355X), x86/ARM/PowerPC CPU, plus plugin backends for Google TPU, Intel Gaudi 2/3, AWS Inferentia/Trainium (Neuron), Apple Silicon, Huawei Ascend, IBM Spyre/Z, and more; llama.cpp targets the widest portability surface (CUDA, HIP/ROCm, Metal, Vulkan, SYCL, CANN, OpenCL, CPU SIMD on x86 AVX-512/AMX and ARM Neon/SVE/SME). MLC-LLM uniquely adds WebGPU/browser via TVM compilation.
- **Leading systems:** vLLM, llama.cpp, SGLang, MLC-LLM
- **Source:** [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) (2026-01-01)
- **fak:** trails — 4 distinct compute backends shipped (CPU + NVIDIA CUDA + AMD Vulkan + Apple Metal), all single-device (shipped)
- **fak note:** REAL but narrower than SOTA breadth. fak ships four numerically-verified single-device backends: pure-Go CPU (x86 AVX2/AVX-512 + arm64), NVIDIA CUDA (sm_80..sm_120, reaching decode parity & W4A4 prefill lead on 26B Inkling-2-Mini on RTX 5080), AMD Vulkan (RX 7600, Q8/Q4_K/GDN wired; full Qwen3.8 receipt awaiting remeasure), and Apple Metal (internal/metalgemm; retired Splash Inkling-2 port retracted, full M3 Pro resident receipts pending). Still trails vLLM (TPU/Gaudi/Trainium/Ascend/Spyre) and llama.cpp (HIP/SYCL/CANN/OpenCL) on accelerator breadth and non-CUDA GPU maturity.
- **Trace:** be228d79ef · docs/benchmarks/published/rtx5080-inkling2-v060/perf-receipt-v2.json · docs/benchmarks/published/rtx5080-inkling2-v060/RESULTS.md (CUDA sm_120f); docs/claims/engine.md (CPU/CUDA/Vulkan/Metal); docs/benchmarks/QWEN-PERFORMANCE-INDEX.md & docs/benchmarks/published/m3pro-inkling2-mini-metal/RESULTS.md

## Hardware shape neutrality (`hardware-shape-neutrality`)

### ≈ Hardware-shape neutrality: explicit fences for host-CPU assumptions — fak: **parity**

*Why it matters:* Hardware breadth can be GPU-biased if the score only counts brands or devices. A serving stack also needs to say whether its core execution contract assumes f32-only tensors, host-addressable buffers, x86 dispatch, synchronous execution, goroutine row splitting, row-major layout, or eager host-resident weights. Operators comparing new silicon need these assumptions named as FENCED or UNDEFINED, otherwise a broad backend list can hide a hard host-shape dependency.

- **SOTA bar:** A hardware-neutral stack exposes backend contracts, capability discovery, layout/dtype/residency boundaries, and explicit fallback behavior so new accelerators do not require a fork of the model loop. Mature serving stacks cover more device families, but most public scorecards still report breadth without a seven-assumption shape-neutrality ledger.
- **Leading systems:** vLLM plugin backends, llama.cpp backend matrix, MLC-LLM / TVM compilation
- **Source:** [https://github.com/anthony-chaudhary/fak/blob/main/docs/explainers/hardware-portability.md](https://github.com/anthony-chaudhary/fak/blob/main/docs/explainers/hardware-portability.md) (2026-06-30)
- **fak:** parity — 7 of 7 host-shape assumptions explicitly FENCED in the compute HAL contract (shipped)
- **fak note:** This corrects the GPU-bias in a pure backend-breadth row. fak still TRAILS the mature engines on how many accelerator families it serves, but it has a FENCED contract for the seven host-shape assumptions the HAL explainer names: (1) float32 monoculture -> Dtype/QuantSpec, (2) host-pointer aliasing -> Tensor/Host/Read boundary, (3) x86 build-tag dispatch -> runtime registry and Tier, (4) synchronous return-by-value -> Caps.Async with explicit Read/Argmax fences, (5) goroutine-only parallelism -> whole-op Backend methods, (6) row-major only -> Layout descriptor, and (7) eager full-RAM plus little-endian host -> WeightSource and Upload. Provenance labels: contract FENCED in code and docs; CUDA/Vulkan/Metal/CPU backends WITNESSED where their support rows say so; broad vendor conformance remains not-yet until the BCK/fak-certified work lands. apples_to_apples=false because this dimension measures contract shape, not device count or throughput.
- **Trace:** docs/explainers/hardware-portability.md plus internal/compute/compute.go: Dtype/QuantSpec, Tensor/Host, Register/Pick/Tier, Caps.Async/Read/Argmax, whole-op Backend methods, Layout, WeightSource/Upload

## Model coverage (`model-coverage`)

### ▼ Maximum parameter count & frontier-MoE architecture coverage — fak: **trails**

*Why it matters:* The first capability question for any serving stack is which models it can actually run. The hard end is the 671B-class sparse MoE (DeepSeek-V3/R1) plus Qwen MoE and Mixtral; supporting these requires MoE kernels, EP, FP8 weights, and MLA/MTP support, not just more memory. A stack that tops out at dense 70B is in a different market than one that serves 671B MoE at scale.

- **SOTA bar:** Kimi K2 (Moonshot AI) -- 1 trillion total parameters / 32B active, 384 experts (8+1 active) -- is the largest open-weight MoE routinely served by vLLM and SGLang, exceeding DeepSeek-V3/R1's 671B.
- **Leading systems:** Kimi K2 / K2.x (Moonshot AI, 1T MoE; vLLM + SGLang + KTransformers), DeepSeek-V3/R1 671B (prior bar), Qwen3 MoE / Llama-4 (other frontier MoE)
- **Source:** [https://huggingface.co/moonshotai/Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct) (2025-07)
- **fak:** trails — 26 B params (max on own engine) (shipped)
- **fak note:** HW-WITNESSED on RTX 5080 16 GB (rtx5080-inkling2-v060/perf-receipt-v2.json): fak-native serves Inkling-2-Mini (26B hybrid MoE, UD-Q3_K_XL) on a single consumer NVIDIA GPU, and Qwen3.8-27B BF16 TP2 / Q4_K_M on datacenter/laptop CUDA (issue-8623, cuda-prefill-sm120), fronting SGLang/vLLM for frontier 100B-1T models. Historical M3 Pro Inkling-2-Mini vendored Splash receipts were retired and retracted as fak-native Metal claims (pure internal/metalgemm resident M3 Pro receipts pending). Trails llama.cpp (72B CPU) and SGLang multi-GPU TP.
- **Trace:** be228d79ef · docs/benchmarks/published/rtx5080-inkling2-v060/perf-receipt-v2.json · docs/benchmarks/published/rtx5080-inkling2-v060/RESULTS.md; docs/_witnesses/issue-8623-qwen38-27b/README.md (TP2 BF16); docs/benchmarks/published/m3pro-inkling2-mini-metal/RESULTS.md (retired Splash port retracted)

### ○ Multimodal / vision-language (VLM) model coverage — fak: **no-claim**

*Why it matters:* A growing share of production traffic is image/video/audio + text (VLMs like Qwen-VL, Llama Vision, InternVL, Pixtral). Serving these adds a vision encoder, modality-aware batching, and image-token KV/prefix handling on top of the LLM path. Whether a stack treats VLMs as first-class (and reuses vision embeddings across requests) is a real differentiator versus text-only engines.

- **SOTA bar:** SGLang is explicitly a serving framework 'for large language models and multimodal models' and extends RadixAttention prefix caching to multimodal inputs with vision-embedding reuse; vLLM also provides broad multimodal model support. Coverage spans the major open VLM families (Qwen-VL, Llama Vision, InternVL, Pixtral, etc.).
- **Leading systems:** SGLang, vLLM
- **Source:** [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) (2026-01-01)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a text-tool-loop reuse kernel. fak's in-kernel model executes text-only forward passes (up to 26B Inkling-2-Mini and 27B Qwen3.8 on 1-2 GPUs); there is NO vision encoder, no image-token patch path, and no multimodal RadixAttention. Larger or VLM workloads ride fronted SGLang/vLLM endpoints. Verdict stays no-claim.
- **Trace:** none — in-kernel forward serves text-only architectures (SmolLM2, Qwen2.5/3.8, Inkling-2, GLM-5.2)

### ▼ Long-context serving (128K-1M tokens) & KV/prefix-cache reuse — fak: **trails**

*Why it matters:* Context windows have moved from 4K to 128K-1M. Serving long context is gated by KV-cache memory (linear in tokens), prefill cost (quadratic attention), and cache reuse. The operator-relevant axes are the max context the engine can hold per request, KV quantization / paging to fit it, and prefix-cache hit rates that recover the prefill cost on RAG / multi-turn / agent workloads.

- **SOTA bar:** Million-token windows are production-ready (Qwen2.5-1M open weights; 128K-1M is mainstream). SGLang's RadixAttention prefix caching delivers up to ~6.4x gains on prefix-heavy RAG/multi-turn workloads and a ~29% throughput edge over vLLM on H100 (16.2K vs 12.5K tok/s) on shared-prefix traffic.
- **Leading systems:** SGLang (RadixAttention), vLLM (PagedAttention/prefix caching)
- **Source:** [https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison) (2026-01-01)
- **fak:** trails — 131072 tokens max context length (shipped)
- **fak note:** HW-WITNESSED on 16 GB RTX 5080: fak serves 131,072 (128K) all-resident context with opt-in NVFP4 KV (0 GiB CPU spill, 1,190 tok/s prefill, 16.9 tok/s decode, exact 128k needle retrieval, rev eb68a8cf8a) and 32K auto-FP8 KV by default (rtx5080-inkling2-v060). Trails datacenter multi-GPU SGLang/vLLM 1,000,000-token context deployments (apples_to_apples=false: single 16 GB consumer GPU all-resident vs datacenter multi-GPU / server deployment).
- **Trace:** eb68a8cf8a · README.md · README.md (128k all-resident NVFP4 KV on 16 GB RTX 5080); docs/benchmarks/published/rtx5080-inkling2-v060/RESULTS.md (32K auto-FP8 KV)

### ○ LoRA / multi-LoRA hot-swap serving — fak: **no-claim**

*Why it matters:* Multi-tenant and per-customer customization is delivered cheaply by serving many LoRA adapters over one shared base model instead of one full model per tenant. The differentiating axes are how many adapters can be resident on GPU concurrently, how many can be cached in CPU (LRU), and whether adapters can be hot-swapped/updated at runtime without restarting (critical for async-RL weight updates). This is the economics of fine-tune-once-serve-many.

- **SOTA bar:** The peer-reviewed SOTA is S-LoRA (MLSys'24, arXiv:2311.03285): ~2000 LoRA adapters served over one base model on a single A100-80GB at up to 4x vLLM throughput, via unified paging of adapter weights + custom MBGMM/MBGMV kernels. vLLM Multi-LoRA integrated S-LoRA-style techniques as a follow-on (max_loras GPU-resident + max_cpu_loras LRU CPU cache + runtime hot-swap); SGLang/TensorRT-LLM offer comparable multi-adapter batching. Practical resident count is bounded by GPU memory and adapter rank.
- **Leading systems:** S-LoRA (MLSys'24, ~2000 adapters/A100-80GB), vLLM Multi-LoRA, SGLang, TensorRT-LLM
- **Source:** [https://www.lmsys.org/blog/2023-11-15-slora/](https://www.lmsys.org/blog/2023-11-15-slora/) (2025-08)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for the reuse/adjudication kernel. fak has no LoRA adapter loading, no multi-adapter batching, no hot-swap path. Its multi-model story is the polymodel residency Pool (host many full models, share prefill, decode one) — a DIFFERENT primitive (full models, not low-rank adapters) and itself off-mainline/policy-only (CLAIMS polymodel rows, FAK_POLYMODEL default off). No fak LoRA number; verdict no-claim.
- **Trace:** none — no LoRA in CLAIMS

### ○ Embedding & reranker (non-generative) model coverage — fak: **no-claim**

*Why it matters:* RAG and search stacks need an engine that also serves embedding and cross-encoder reranker models, ideally on the same serving fabric as the generative LLM (shared pooling, batching, an OpenAI-compatible /v1/embeddings + rerank API). A stack that only does autoregressive decode forces operators to run a second serving system for retrieval, so embedding/reranker coverage is a real platform-completeness axis.

- **SOTA bar:** vLLM and SGLang both serve embedding and reranker/cross-encoder models alongside generative models with pooling support and OpenAI-compatible embeddings endpoints, covering the common open embedding/reranker families (BGE, E5, GTE, Qwen-embedding, etc.) on the same engine.
- **Leading systems:** vLLM, SGLang, TensorRT-LLM
- **Source:** [https://docs.vllm.ai/en/latest/](https://docs.vllm.ai/en/latest/) (2026-01-01)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a generative-tool-loop reuse kernel. fak serves only causal-LM next-token decode; there is no pooling head, no embeddings endpoint, no cross-encoder reranker path. fak's OpenAI-compatible surface is /v1/chat/completions (CLAIMS Gateway), not /v1/embeddings. No fak number; verdict no-claim.
- **Trace:** none — no pooling/embedding endpoint in CLAIMS

### ○ Capability/quality of the served model (SWE-bench Verified, GPQA, AIME, LiveCodeBench) — fak: **no-claim**

*Why it matters:* A serving stack only matters if it serves a model that can do the job; buyers pick the model on SWE-bench/GPQA/AIME first, then the engine to serve it. Engines must be framed as transport for these scores, and constrained decoding/scaffolding can move them, so the dimension belongs on the scorecard explicitly.

- **SOTA bar:** Frontier models served by these engines define the quality bar buyers cite: SWE-bench Verified ~80.6% (Gemini 3.1 Pro) / 79.2% (Claude Opus 4.5 + Live-SWE-agent, Nov 2025); GPQA Diamond effectively saturated ~93-94% (Gemini 3 Deep Think 93.8%, GPT-5.4 92.8%); AIME 2025 up to 100% (GPT-5.4). These are model-served-by-engine results, not engine claims.
- **Leading systems:** Gemini 3.x (SWE-bench/GPQA), Claude Opus 4.5 + agent scaffold, GPT-5.x (AIME)
- **Source:** [https://epoch.ai/benchmarks/swe-bench-verified](https://epoch.ai/benchmarks/swe-bench-verified) (2025-11-24)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE by construction. fak is a serving KERNEL, not a model: SWE-bench Verified / GPQA / AIME / LiveCodeBench are set entirely by whatever model it serves, so fak claims NO resolve rate of its own. The frontier bars cited (SWE-bench Verified ~79-80.6%, GPQA Diamond ~93-94%, AIME up to 100%) are model-served-by-engine results. fak's claim is orthogonal: ~20-24x prefill-work reduction ON TOP of any model at the same quality, lower infra cost. Carried so the scorecard never quietly omits the headline a frontier-model comparison leads with. Honest no-claim.
- **Trace:** industry_scorecard.data.json swebench-resolve-rate row (no-claim/stub); SOTA-COMPARISON.md; CLAIMS: fak is a serving kernel, resolve rate set by the served model

