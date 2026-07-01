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
- **fak note:** A real, disclosed head-to-head where fak TRAILS on coverage breadth. fak's hardware portability is genuine (CPU SIMD + Vulkan + CUDA, with bit-exact cross-platform reproduction), but its FORMAT coverage is narrow: f32 plus 8-bit Q8_0, with GGUF k-quant load-only. The SOTA stacks accelerate FP8 across Hopper/Ada/Blackwell + MI300X, NVFP4 on Blackwell, and broad GGUF k-quants (llama.cpp). fak has no hardware-accelerated FP8/INT4/FP4 GEMM on ANY device, so on the format×hardware coverage axis it clearly trails — named, not buried.
- **Trace:** BENCHMARK-AUTHORITY.md + CLAIMS.md: fak runs f32 + Q8_0 on CPU (arm64/x86 SIMD), Vulkan (RX 7600), and CUDA (RTX 4070 sm_89, datacenter GPU); GGUF q4_k_m load proven (Qwen3.6-27B) but slow. NO FP8/INT8-W8A8/INT4-accelerated/NVFP4/MXFP4 kernel on any target

### ▼ Hardware backend breadth (NVIDIA, AMD, Intel, TPU, AWS, Apple, CPU) — fak: **trails**

*Why it matters:* Vendor lock-in to a single accelerator is a top procurement risk. The buyer axis is how many silicon families a stack runs on with maintained, performant paths: NVIDIA Hopper/Blackwell, AMD ROCm (MI300X/MI355X), Intel Gaudi/XPU, Google TPU, AWS Inferentia/Trainium, Apple Metal, and CPU. Breadth determines portability and negotiating leverage; depth (kernel maturity per backend) determines whether the non-NVIDIA path is actually usable in production.

- **SOTA bar:** vLLM runs on NVIDIA, AMD ROCm (MI300X/MI325X/MI350X/MI355X), x86/ARM/PowerPC CPU, plus plugin backends for Google TPU, Intel Gaudi 2/3, AWS Inferentia/Trainium (Neuron), Apple Silicon, Huawei Ascend, IBM Spyre/Z, and more; llama.cpp targets the widest portability surface (CUDA, HIP/ROCm, Metal, Vulkan, SYCL, CANN, OpenCL, CPU SIMD on x86 AVX-512/AMX and ARM Neon/SVE/SME). MLC-LLM uniquely adds WebGPU/browser via TVM compilation.
- **Leading systems:** vLLM, llama.cpp, SGLang, MLC-LLM
- **Source:** [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) (2026-01-01)
- **fak:** trails — 4 distinct compute backends shipped (CPU + NVIDIA CUDA + AMD Vulkan + Apple Metal), all single-device (shipped)
- **fak note:** REAL but narrow vs the SOTA breadth bar. fak ships four numerically-correct single-device backends: pure-Go CPU (x86 AVX/arm64) plus NVIDIA CUDA (RTX 4070 + 8-GPU datacenter server sm_80), AMD Vulkan (RX 7600), and Apple Metal (M3 Pro), each argmax-exact / cosine=1.0 vs cpu-ref. But vLLM (TPU, Gaudi 2/3, Inferentia/Trainium, Ascend, Spyre) and llama.cpp (HIP, SYCL, CANN, OpenCL, WebGPU via MLC) span far more silicon, and fak's GPU paths are mostly f32 and throughput-immature (Vulkan ~58x slower than llama.cpp CPU; CUDA reaches parity only on a small model that fits VRAM). fak trails on breadth and on per-backend maturity; disclosed, not hidden.
- **Trace:** CLAIMS Engine: pure-Go CPU; '-tags vulkan' RX 7600; '-tags cuda' RTX 4070 / datacenter GPU; 'darwin && metal' M3 Pro — each numerically witnessed (cosine=1.0 / argmax-exact)

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
- **fak:** trails — 7 B params (max on own engine) (shipped)
- **fak note:** fak's OWN generic GGUF-dequant engine ceilings at ~7B on 36 GB (dequant-to-f32 OOMs above; two arch families fail at load; f32 27B = 108 GB > 80 GB VRAM). fak's actual DeepSeek-class-MoE serving path is the in-kernel GLM-5.2 work (#1010, #1026): it runs the MoE/FFN experts + router on its own pure k_q8_gemm GPU kernel (cosine=1.0, commit 498a4ab) — but on a SINGLE device, with the DSA sparse-attention still host-side and no cross-node expert parallelism (see expert-parallelism-large-moe and attention-arch-kv-footprint). So for a frontier 671B MoE — the new MLPerf v6.0 / DeepSeek-R1 reference workload — fak is NOT in the race on its own engine and must FRONT llama.cpp/SGLang, exactly what the 27B GPU server run does (SGLang-serves + fak-adjudicates). A hard, disclosed capability ceiling that bounds where any fak engine 'win' can apply. apples_to_apples=false: fak's dequant/single-device engine vs llama.cpp quantized/CPU and SGLang multi-GPU TP are different configs by construction.
- **Trace:** HERO-BENCHMARK-2026-06-21.md (honest fence) · CLAIMS.md

### ○ Multimodal / vision-language (VLM) model coverage — fak: **no-claim**

*Why it matters:* A growing share of production traffic is image/video/audio + text (VLMs like Qwen-VL, Llama Vision, InternVL, Pixtral). Serving these adds a vision encoder, modality-aware batching, and image-token KV/prefix handling on top of the LLM path. Whether a stack treats VLMs as first-class (and reuses vision embeddings across requests) is a real differentiator versus text-only engines.

- **SOTA bar:** SGLang is explicitly a serving framework 'for large language models and multimodal models' and extends RadixAttention prefix caching to multimodal inputs with vision-embedding reuse; vLLM also provides broad multimodal model support. Coverage spans the major open VLM families (Qwen-VL, Llama Vision, InternVL, Pixtral, etc.).
- **Leading systems:** SGLang, vLLM
- **Source:** [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) (2026-01-01)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a text-tool-loop reuse kernel. fak's in-kernel model is a text-only forward pass (SmolLM2-135M / Qwen2.5 / GLM-5.2); there is NO vision encoder, no image-token path, no multimodal RadixAttention. fak's prefix-reuse mechanism is modality-agnostic in principle but no VLM is wired or measured. No fak number; verdict no-claim. (A REAL gap only if fak intended to serve VLMs on its own engine — it does not; it fronts an external engine for anything beyond ~7B text.)
- **Trace:** none — no VLM in CLAIMS; in-kernel model is text-only SmolLM2/Qwen forward

### ○ Long-context serving (128K-1M tokens) & KV/prefix-cache reuse — fak: **no-claim**

*Why it matters:* Context windows have moved from 4K to 128K-1M. Serving long context is gated by KV-cache memory (linear in tokens), prefill cost (quadratic attention), and cache reuse. The operator-relevant axes are the max context the engine can hold per request, KV quantization / paging to fit it, and prefix-cache hit rates that recover the prefill cost on RAG / multi-turn / agent workloads.

- **SOTA bar:** Million-token windows are production-ready (Qwen2.5-1M open weights; 128K-1M is mainstream). SGLang's RadixAttention prefix caching delivers up to ~6.4x gains on prefix-heavy RAG/multi-turn workloads and a ~29% throughput edge over vLLM on H100 (16.2K vs 12.5K tok/s) on shared-prefix traffic.
- **Leading systems:** SGLang (RadixAttention), vLLM (PagedAttention/prefix caching)
- **Source:** [https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison) (2026-01-01)
- **fak:** no-claim — no number (stub)
- **fak note:** fak's 86.7% prefix-cache hit rate does NOT measure long-context capacity. fak's real long-context posture is its model-size ceiling (~7B on 36 GB) plus an arithmetic ultra-long-context WORK floor — not a served-capacity-vs-SOTA number. Honest gap.
- **Trace:** experiments/radixattention/radixbench-qwen2.5-1.5b-q8-agents-fresh-20260619.json (workloads[0].cache_hit_rate=0.8667), commit 92896a4; BENCHMARK-AUTHORITY RadixAttention rows

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

