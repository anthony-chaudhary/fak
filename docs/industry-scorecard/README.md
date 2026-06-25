---
title: "fak industry scorecard — where fak stands in the LLM-serving field"
description: "Industry-first competitive scorecard: a researched taxonomy of the dimensions that matter in LLM inference-serving + agent infrastructure (vLLM, SGLang, TensorRT-LLM, llama.cpp, …), the current SOTA bar on each, and fak's honest position — most as named gaps. Two driven numbers: coverage (of the field) and parity-debt (honesty of the rows)."
---

# Industry scorecard — fak vs the LLM-serving field

<!-- industry-scorecard: 2026-06-24 · process: tools/industry_scorecard.py · data: tools/industry_scorecard.data/ -->

This is the **outward** measuring stick — the counterpart of the inward scorecards (hygiene, code, docs). It does not start from what fak happened to measure; it starts from the **industry**. The source of truth is a researched taxonomy of the dimensions a serious operator, buyer, or analyst uses to evaluate an LLM-serving system (`tools/industry_scorecard.data/_taxonomy.json`), each with the current SOTA bar and a dated source. fak is then positioned honestly on every dimension — and for most of them the honest answer is a **named gap**, not a win. Everything below is re-derived from the data by `tools/industry_scorecard.py`; no number is hand-typed.

Two numbers are driven:

- **coverage** — of the industry dimensions that matter, how many the scorecard has considered and positioned fak against (toward 100%).
- **parity-debt** — of the comparisons that exist, how many are dishonest, incomplete, or unsourced (kept at 0).

> Regenerate: `python tools/industry_scorecard.py --markdown-dir docs/industry-scorecard`. Update process: [UPDATE-PROCESS.md](UPDATE-PROCESS.md). Full dimension catalog: [taxonomy.md](taxonomy.md).

## Headline

| Metric | Value |
|---|---|
| **Coverage** | **100.0%** (88/88 industry dimensions positioned) |
| **Parity-debt (honesty defects)** | **0** |
| Coverage-debt (unpositioned dimensions) | 0 |
| Composite score | 98.8/100 (grade A) — honesty 98.0 × 60% + coverage 100.0% × 40% |
| Standing | 4 lead · 11 parity · 8 trails · 65 honest gap |
| Measured vs gap | 22 measured · 66 honest gaps |
| Tracked | 88 dimensions · 103 competitors · 88 positions |
| As of | 2026-06-23 (fak 0.31.0) |
| Advisory signals | 7 fak-freshness · 47 industry-drift |

> **Read this right.** The score grades how *complete and honest fak's competitive map is* — not how much fak wins. fak is a focused reuse + trust kernel, so most dimensions are honest `no-claim` gaps (out-of-scope or not-yet-measured), shown plainly below.

## Standing at a glance

```text
industry standing chart — 88 dimensions · 103 competitors · score 98.8/100 (grade A) · parity-debt 0

coverage of the field (positioned / in-scope dimensions):
  positioned  [████████████████████████████████]  88/88  (100.0%)

standing on the positioned axes (shown, not hidden):
  ▲ lead      █······················· 4
  ≈ parity    ████···················· 11
  ▼ trails    ███····················· 8
  ○ no-claim  ████████████████████████ 65

coverage by group:
  agent          ████████················ 5/5
  cost           ████████················ 5/5
  decoding       █████████████████████··· 13/13
  distributed    ████████················ 5/5
  memory         █████████████████████··· 13/13
  models         █████████████··········· 8/8
  numerics       █████████████████████··· 13/13
  operability    ██████████████·········· 9/9
  security       ███····················· 2/2
  serving        ████████████████████████ 15/15
```

## Coverage by group

| Group | Positioned / in-scope | Pages |
|---|---|---|
| agent | 5/5 | [agent.md](agent.md) |
| cost | 5/5 | [cost.md](cost.md) |
| decoding | 13/13 | [decoding.md](decoding.md) |
| distributed | 5/5 | [distributed.md](distributed.md) |
| memory | 13/13 | [memory.md](memory.md) |
| models | 8/8 | [models.md](models.md) |
| numerics | 13/13 | [numerics.md](numerics.md) |
| operability | 9/9 | [operability.md](operability.md) |
| security | 2/2 | [security.md](security.md) |
| serving | 15/15 | [serving.md](serving.md) |

## Standing across the field (data-derived)

▲ lead · ≈ parity · ▼ trails (shown, not hidden) · ○ honest gap (no claim yet).

| | Verdict | Category | Axis | fak | vs competitor | Ratio | Competitor |
|---|---|---|---|---|---|---|---|
| ▲ | lead | agent-fleet | Cross-agent fleet serving time (N agents × T turns): work eliminated by shared-prefix fusion | 19 min | 78 min | 4.11× | Tuned warm per-agent KV cache (the SGLang / vLLM / OpenAI-prompt-caching floor) |
| ▲ | lead | agent-fleet | Marginal value of cross-agent prefix fusion ON TOP of an already-warm per-agent cache | 2.4 × | 1 × | 2.40× | Tuned warm per-agent KV cache, already hot (B) |
| ▲ | lead | model-routing | Model routing granularity: per-aspect + first-class ensemble routing vs whole-request single-model selection | — | — | — | RouteLLM (LMSYS); Martian; NotDiamond; Unify.ai; OpenRouter (+ Fusion); Portkey; LiteLLM Router |
| ▲ | lead | quantization | int8 / Q8_0 SIMD decode throughput vs the same-rung int8 peer | 2.97 × vs HF int8 | 1 × vs HF int8 | 2.97× | HuggingFace dynamic-int8 (the standard same-rung int8 reference) |
| ≈ | parity | kv-cache | Prefix caching / automatic KV reuse across requests (shared-context workloads) | 6.95 × | 7.5 × | 0.93× | fak's own deterministic token-reuse ceiling (the exact upper bound) |
| ≈ | parity | kv-cache | Prefix/KV-cache reuse impact on TTFT (cache-hit latency) | 6.95 × | 7.5 × | 0.93× | fak's own deterministic token-reuse ceiling (the exact upper bound) |
| ≈ | parity | kv-cache | Automatic prefix caching / RadixAttention prefix reuse and cache hit rate | 86.7 % | 50–99 % | — | SGLang RadixAttention published hit-rate band |
| ≈ | parity | kv-cache | Automatic cross-request prefix / KV-cache reuse (shared system prompts, agent scaffolds, few-shot) | 86.7 % | 50–99 % | — | SGLang RadixAttention published hit-rate band |
| ≈ | parity | numerical-correctness | Numerical-correctness error metric vs reference (perplexity, KL-divergence, recovery %) | — | — | — | HuggingFace transformers f32 reference (the oracle) |
| ≈ | parity | numerical-correctness | Dense GPU compute correctness with ZERO vendor GEMM (cuBLAS-free) | — | — | — | cuBLAS (NVIDIA's tuned vendor GEMM — the standard dependency every other stack carries) |
| ≈ | parity | operability | OpenAI-compatible API surface + reliable structured output / tool-calling | — | — | — | vLLM (XGrammar/guided decoding); SGLang; LiteLLM (gateway) |
| ≈ | parity | scheduling | Request scheduling and cache-aware routing (FCFS vs priority vs prefix/KV-cache-aware) | 86.7 % | 50–99 % | — | SGLang RadixAttention published hit-rate band |
| ≈ | parity | security | Tool/agent sandboxing, structural containment, and PII/exfil prevention | — | — | — | Container/microVM sandboxes; MELON (provable IPI defense); IFC / capability allow-lists |
| ≈ | parity | single-stream | Single-stream decode throughput on a consumer GPU | 120 tok/s | 105–135 tok/s | 1.00× | llama.cpp Q8_0 on the same RTX 4070 (-ngl 99) |
| ≈ | parity | speculative-decoding | Output-distribution losslessness of speculation (greedy and stochastic) | — | — | — | vLLM; speculative sampling (Leviathan et al.) |
| ▼ | trails | hardware-coverage | Hardware / format coverage and portability of quantized kernels | — | — | — | vLLM (FP8 on Hopper/Ada/Blackwell + MI300X via hipBLASLt/aiter); SGLang (FP8 MI300X); TensorRT-LLM (NVFP4 Blackwell); llama.cpp (broad CPU/GPU GGUF + FP4) |
| ▼ | trails | hardware-coverage | Hardware backend breadth (NVIDIA, AMD, Intel, TPU, AWS, Apple, CPU) | 4 distinct compute backends shipped (CPU + NVIDIA CUDA + AMD Vulkan + Apple Metal), all single-device | — | — | vLLM; llama.cpp; SGLang; MLC-LLM |
| ▼ | trails | model-coverage | Maximum parameter count & frontier-MoE architecture coverage | 7 B params (max on own engine) | 72 B params (max on own engine) | 0.10× | llama.cpp (72B CPU / ~32B Metal; Qwen3-30B-A3B MoE @ 50 tok/s) · SGLang (multi-GPU TP, 27B+ quantized) |
| ▼ | trails | operability | Production observability: Prometheus metrics (TTFT/TPOT/queue) + OpenTelemetry tracing | — | — | — | vLLM; SGLang; NVIDIA Dynamo |
| ▼ | trails | security | Prompt-injection defense and agent security (attack-success-rate vs utility-under-attack) | — | — | — | Tuned guardrail / defense stacks on a SOTA evasion battery (AgentDojo-style) |
| ▼ | trails | single-stream | Single-stream (one chat) decode throughput on CPU | 8.7 tok/s | 17.3 tok/s | 0.50× | llama.cpp Metal (M3 Pro) |
| ▼ | trails | single-stream | Single-stream prefill throughput, apples-to-apples CPU-vs-CPU | 0.12 × of llama.cpp CPU | 1 × of llama.cpp CPU | 0.12× | llama.cpp CPU, same box + Q8 weights (CPU-vs-CPU) |
| ▼ | trails | throughput | Peak served throughput per accelerator (model-served-by-engine, MLPerf-grounded) | 1085.6 tok/s @ conc 64 (peak) | 1451.6 tok/s @ conc 64 (peak) | 0.75× | Raw SGLang 0.5.10 (TP=8, no fak proxy), Qwen3.6-27B on an 8-GPU datacenter server |
| ○ | no-claim | agent-fleet | Cross-agent reuse marginal value vs a LIVE tuned shared-prefix engine (head-to-head) | — | — | — | SGLang RadixAttention / vLLM Automatic Prefix Caching (actual competing processes) |
| ○ | no-claim | cost-efficiency | Goodput under an SLA (requests/s meeting TTFT and TPOT SLOs simultaneously) | — | — | — | DistServe (introduced goodput, P/D disaggregation); SGLang-PD; MuxWise (P/D multiplexing) |
| ○ | no-claim | cost-efficiency | SLO-constrained goodput (per-GPU) | — | 4.48 SLO-attained req/s per GPU | — | DistServe; DynaServe; NVIDIA Dynamo; GenAI-Perf goodput mode |
| ○ | no-claim | cost-efficiency | End-to-end effective request capacity / cost gain from the KV system under SLO | — | — | — | Mooncake; LMCache + vLLM; NVIDIA Dynamo |
| ○ | no-claim | cost-efficiency | Inference unit economics ($ per 1M tokens) at realistic utilization | — | — | — | Hosted frontier APIs; vLLM/SGLang self-host on H100/H200; FP8/FP4 quantization |
| ○ | no-claim | cost-efficiency | Energy efficiency (Wh per token / tokens-per-watt) and power-capped throughput | — | — | — | NVIDIA Blackwell (GB200/GB300, FP4); AMD MI355X (FP4) |
| ○ | no-claim | kv-cache | KV-cache memory efficiency and max concurrent sequences (paging / fragmentation) | — | — | — | vLLM (PagedAttention); vAttention (paging without PagedAttention); SGLang (token-level KV pool) |
| ○ | no-claim | kv-cache | Paged KV block management and memory utilization (anti-fragmentation) | — | 96.3 % KV memory utilization | — | vLLM (PagedAttention); SGLang; TensorRT-LLM |
| ○ | no-claim | kv-cache | Non-prefix / cross-document KV reuse (RAG chunk caching) | — | — | — | LMCache (CacheBlend) |
| ○ | no-claim | kv-cache | KV-cache offloading to CPU/host/NVMe/remote (multi-tier hierarchy depth) | — | — | — | LMCache; Mooncake (Mooncake Store) |
| ○ | no-claim | kv-cache | KV compression / token eviction under a fixed cache budget | — | — | — | H2O; StreamingLLM; SnapKV; Quest |
| ○ | no-claim | kv-cache | KV-cache transfer for prefill/decode disaggregation (NIXL / NCCL / UCX) | — | — | — | NVIDIA NIXL; NVIDIA Dynamo; Mooncake |
| ○ | no-claim | kv-cache | KV-cache-aware / prefix-aware request routing across a fleet | — | 2 x speedup from cache-aware routing | — | NVIDIA Dynamo (KV-aware router); SGLang router; Baseten |
| ○ | no-claim | kv-cache | Attention architecture impact on KV footprint (MQA/GQA/MLA bytes-per-token) | — | 70 KB/token KV footprint | — | DeepSeek-V2/V3 (MLA); GQA-based models (Llama-3, Qwen) |
| ○ | no-claim | kv-cache | KV-cache quantization (FP8 / INT8 / KIVI sub-4-bit) accuracy at long context | — | — | — | vLLM (--kv-cache-dtype fp8, E4M3); KIVI (2-bit asymmetric); KVTuner (mixed-precision); KVQuant |
| ○ | no-claim | latency | Interactive latency SLO attainment: TTFT and TPOT/ITL tail under load | — | — | — | MLPerf Inference v5.1 interactive scenario; TensorRT-LLM; Sarathi-Serve / vLLM chunked prefill |
| ○ | no-claim | latency | Time-to-first-token (TTFT) under realistic prefill | — | 4500 ms TTFT (P50/P99) | — | MLPerf Inference v5.1 (Llama3.1-405B interactive); vLLM benchmark_serving; NVIDIA GenAI-Perf |
| ○ | no-claim | latency | Inter-token latency / time-per-output-token (ITL/TPOT) | — | 80 ms TPOT (P50/P99) | — | MLPerf Inference v5.1; vLLM benchmark_serving; NVIDIA GenAI-Perf |
| ○ | no-claim | latency | End-to-end request latency (E2EL) | — | — | — | NVIDIA GenAI-Perf; vLLM benchmark_serving (e2el percentile) |
| ○ | no-claim | latency | Tail latency distribution (P50/P95/P99/P99.9) | — | — | — | MLPerf Inference (P99-based constraints); vLLM benchmark_serving; production SLO practice |
| ○ | no-claim | latency | Prefill-decode interference & chunked-prefill TTFT/TPOT tension | — | 68 % P95 ITL improvement from chunked prefill | — | Sarathi-Serve; vLLM chunked prefill; SGLang |
| ○ | no-claim | model-coverage | Multimodal / vision-language (VLM) model coverage | — | — | — | SGLang; vLLM |
| ○ | no-claim | model-coverage | Long-context serving (128K-1M tokens) & KV/prefix-cache reuse | — | — | — | SGLang (RadixAttention); vLLM (PagedAttention/prefix caching) |
| ○ | no-claim | model-coverage | LoRA / multi-LoRA hot-swap serving | — | — | — | vLLM; SGLang; TensorRT-LLM |
| ○ | no-claim | model-coverage | Embedding & reranker (non-generative) model coverage | — | — | — | vLLM; SGLang; TensorRT-LLM |
| ○ | no-claim | model-coverage | Capability/quality of the served model (SWE-bench Verified, GPQA, AIME, LiveCodeBench) | — | — | — | Gemini 3.x (SWE-bench/GPQA); Claude Opus 4.5 + agent scaffold; GPT-5.x (AIME) |
| ○ | no-claim | numerical-correctness | Deterministic / bitwise-reproducible inference (batch invariance) | — | 1000 bitwise-identical completions under dynamic batching | — | Thinking Machines batch-invariant-ops (on vLLM FlexAttention); vLLM batch-invariance mode; LLM-42 (deterministic speculation) |
| ○ | no-claim | operability | Fairness, priority, and tenant isolation in batch formation / scheduling | — | — | — | FairBatching (fairness-aware batch formation); SGLang (FCFS + page-eviction preemption); hybrid real-time/best-effort schedulers (e.g. arXiv 2504.09590) |
| ○ | no-claim | operability | Latency degradation under concurrency / load (saturation behavior) | — | — | — | vLLM benchmark_serving (request-rate sweep); NVIDIA GenAI-Perf (concurrency sweep) |
| ○ | no-claim | operability | Cold-start / autoscaling latency (scale-to-zero & scale-out) | — | 250 ms/s cold-start to first token | — | RunPod FlashBoot; ServerlessLLM; HydraServe; PipeBoost |
| ○ | no-claim | operability | Cross-instance / persistent KV cache sharing and coherence | — | — | — | LMCache; Mooncake Store |
| ○ | no-claim | operability | Multi-node fault tolerance & failure recovery (resilient serving) | — | — | — | FailSafe (research); KevlarFlow (research); ReviveMoE (research); NVIDIA Dynamo |
| ○ | no-claim | operability | Kubernetes-native deployment + LLM-aware autoscaling (metric-driven, scale-to-zero) | — | — | — | KServe + KEDA; Ray Serve; llm-d / vLLM on Kubernetes |
| ○ | no-claim | operability | Multi-tenant fairness, per-tenant quotas, and SLO attainment under contention | — | — | — | VTC (Sheng et al.); LiteLLM (virtual keys/budgets); Equinox / FairBatching |
| ○ | no-claim | parallelism | Prefill-decode disaggregation: independent scaling of compute-bound prefill and memory-bound decode | — | — | — | DistServe; Splitwise; SGLang-PD / vLLM disaggregated; llm-d (vLLM + NIXL KV connector) |
| ○ | no-claim | parallelism | Prefill-decode disaggregation & per-phase SLO isolation | — | 7.4 x request rate / SLO-attainment at >90% | — | NVIDIA Dynamo; DistServe; llm-d; DeepSeek (production PD-disagg) |
| ○ | no-claim | parallelism | Large-scale expert parallelism (EP) for giant MoE models | — | — | — | SGLang; DeepSeek (DeepEP); vLLM |
| ○ | no-claim | parallelism | Data-parallel (DP) attention / hybrid attention-FFN parallelism | — | — | — | SGLang; vLLM |
| ○ | no-claim | parallelism | Multi-node scale-out (combined TP x PP x EP x DP across hosts) | — | — | — | SGLang; NVIDIA Dynamo; TensorRT-LLM; NVIDIA GB200/GB300 NVL72 |
| ○ | no-claim | quantization | KV-cache quantization (FP8 / INT8 / INT4 KV) | — | 2 x KV memory reduction (KV dtype) | — | TensorRT-LLM; vLLM |
| ○ | no-claim | quantization | FP8 (W8A8, E4M3) accuracy retention vs bf16 reference | — | 99 % accuracy recovery vs bf16 | — | vLLM (FP8 W8A8 dynamic/static); TensorRT-LLM (ModelOpt); SGLang; Red Hat / Neural Magic llm-compressor |
| ○ | no-claim | quantization | INT8 W8A8 (SmoothQuant) accuracy under activation outliers | — | — | — | SmoothQuant (MIT HAN Lab); vLLM INT8 W8A8; TensorRT-LLM; llm-compressor |
| ○ | no-claim | quantization | INT4 weight-only (W4A16: AWQ, GPTQ) accuracy and calibration sensitivity | — | — | — | AWQ (LMDeploy, vLLM); GPTQ (vLLM, AutoGPTQ); TensorRT-LLM W4A16 |
| ○ | no-claim | quantization | 4-bit microscaling float (NVFP4 / MXFP4 / FP4) accuracy vs FP8 on Blackwell | — | 0.1 MMLU-point drop vs FP8 | — | NVIDIA Blackwell + TensorRT-LLM (ModelOpt NVFP4); vLLM NVFP4; llama.cpp (NVFP4/MXFP4) |
| ○ | no-claim | quantization | Full 4-bit (W4A4 + KV4) via rotation/outlier removal | — | 99 WikiText-2 perplexity delta / zero-shot retention | — | QuaRot (Hadamard rotation); SpinQuant (learned rotation); QServe W4A8KV4 |
| ○ | no-claim | quantization | Quantization granularity (per-tensor / per-channel / per-group / microscaling block size) | — | 3 SNR dB by granularity | — | MX formats (OCP MXFP4/MXFP8, block-32); NVFP4 (block-16, FP32 second-level); group-128 INT4 (AWQ/GPTQ); MOSS two-level microscaling |
| ○ | no-claim | quantization | Accuracy-constrained throughput on a neutral benchmark (MLPerf Inference) | — | — | — | NVIDIA GB200 NVL72 / B200 (TensorRT-LLM, NVFP4); AMD MI300X/MI355X (ROCm, FP8); MLPerf Inference v5.0/v5.1 submitters |
| ○ | no-claim | quantization | Quantization format & low-precision datatype coverage | — | — | — | TensorRT-LLM; vLLM; NVIDIA Blackwell |
| ○ | no-claim | request-routing | Cache-aware request routing across replicas (route to the worker that already holds the prefix) | — | — | — | NVIDIA Dynamo (Smart Router); SGLang router; vLLM production router |
| ○ | no-claim | scheduling | Chunked / piggyback prefill and prefill-decode interference (stall-free batching) | — | — | — | Sarathi-Serve (chunked-prefill + stall-free scheduling); vLLM (chunked prefill); SGLang (chunked prefill) |
| ○ | no-claim | speculative-decoding | Speculative decoding throughput effect across batch sizes (does it survive at high concurrency?) | — | — | — | EAGLE-3; EAGLE-2 (regresses at batch); vLLM / SGLang / TensorRT-LLM speculative backends |
| ○ | no-claim | speculative-decoding | Speculative decoding effect on ITL (acceptance-rate-bound) | — | 3.5 x decode token-rate / acceptance rate | — | EAGLE-3; Medusa-2; TensorRT-LLM / vLLM speculative decoding |
| ○ | no-claim | speculative-decoding | Speculative-decoding wall-clock speedup (single-stream / low-concurrency) | — | 4.79 x wall-clock speedup | — | EAGLE-3 (SafeAILab); vLLM; SGLang; P-EAGLE (vLLM) |
| ○ | no-claim | speculative-decoding | Draft acceptance length / acceptance rate (mean accepted tokens per verify step) | — | 0.85 mean accepted tokens / verify step | — | EAGLE-3; DeepSeek-V3 MTP; Medusa |
| ○ | no-claim | speculative-decoding | Breadth of speculative methods supported (draft-model, EAGLE/-2/-3, Medusa, MTP, lookahead, n-gram/prompt-lookup) | — | — | — | TensorRT-LLM; vLLM; SGLang |
| ○ | no-claim | speculative-decoding | Retrieval / prompt-lookup (n-gram) drafting for input-grounded tasks | — | 2.8 x speedup on input-grounded tasks | — | vLLM (ngram / prompt-lookup); TensorRT-LLM (Lookahead) |
| ○ | no-claim | speculative-decoding | Speculative-decoding behavior under high concurrency / large batch (break-even) | — | — | — | MagicDec; vLLM continuous batching; SGLang |
| ○ | no-claim | structured-output | Constrained/guided decoding per-token overhead (JSON-schema, regex, grammar/CFG) | — | 40 microseconds / token mask overhead | — | XGrammar; LLGuidance; SGLang; vLLM |
| ○ | no-claim | structured-output | Grammar/schema compilation latency and cache (cold-start for dynamic agentic schemas) | — | 10 ms grammar/schema compile | — | XGrammar-2; XGrammar; LLGuidance/Guidance |
| ○ | no-claim | structured-output | Structured-output validity vs reasoning-quality tax (constraint-induced degradation) | — | — | — | JSONSchemaBench (Outlines/XGrammar/Guidance/llama.cpp/OpenAI/Gemini); OpenAI Structured Outputs |
| ○ | no-claim | structured-output | Tool/function-call parsing coverage and robustness (per-model formats) | — | — | — | vLLM tool-call parsers; SGLang; XGrammar-2 (grammar-backed tool calling) |
| ○ | no-claim | structured-output | Sampler coverage and correctness (temperature, top-p, top-k, min-p) | — | — | — | vLLM; llama.cpp; SGLang; TensorRT-LLM |
| ○ | no-claim | throughput | Continuous / in-flight (iteration-level) batching: max aggregate token throughput at high concurrency | — | — | — | vLLM (PagedAttention + continuous batching); Orca (iteration-level scheduling, originator); TensorRT-LLM (in-flight batching) |
| ○ | no-claim | throughput | Absolute aggregate tokens/s at scale on standardized hardware (MLPerf Inference) | — | 13886 tokens/s (standardized MLPerf hardware) | — | NVIDIA GB200 NVL72 (Blackwell); NVIDIA HGX B200 / Blackwell Ultra; MLPerf Inference v5.0 / v5.1 submitters |
| ○ | no-claim | throughput | Latency-vs-throughput Pareto frontier | — | — | — | Sarathi-Serve (chunked prefill); DynaServe; vLLM/SGLang sweeps |

## Per-KPI (parity-debt = honesty of the rows that exist)

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| honesty | `verdict_consistency` | 90 | 0 | every verdict matches its evidence (7 unverifiable) |
| structure | `well_formed` | 100 | 0 | all 88 rows well-formed |
| completeness | `competitor_named` | 100 | 0 | every row names a concrete competitor |
| completeness | `axis_coverage` | 100 | 0 | all 5 contracted regimes covered |
| honesty | `baseline_sota` | 100 | 0 | every comparison is vs a tuned / SOTA / next-best baseline |
| honesty | `apples_disclosed` | 100 | 0 | every non-comparable row discloses what differs |
| traceability | `fak_traced` | 100 | 0 | every shipped claim traces to evidence |
| traceability | `competitor_sourced` | 100 | 0 | every competitor number is sourced |
| traceability | `freshness` | 100 | 0 | every fak measurement within 150d of 2026-06-23 |

