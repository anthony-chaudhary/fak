---
title: "fak industry scorecard — serving"
description: "The serving dimensions that matter in LLM serving, the current SOTA bar on each, and fak's honest position. Generated from tools/industry_scorecard.data/."
---

# serving — the dimensions that matter, and where fak stands

[← back to the scorecard index](README.md) · part of the industry-first scorecard. Each dimension is a thing the field competes on; the fak column is honest — mostly `no-claim` gaps for a focused reuse kernel.

## Throughput & batching (`throughput`)

### ○ Continuous / in-flight (iteration-level) batching: max aggregate token throughput at high concurrency — fak: **no-claim**

*Why it matters:* Iteration-level batching admits and retires requests mid-flight so a long sequence never blocks short ones, and a fixed GPU serves far more simultaneous sequences. It is the single largest lever on tokens/s per GPU and on serving cost, so it is the first thing any operator benchmarks.

- **SOTA bar:** Continuous (in-flight) batching is now table stakes across vLLM/SGLang/TensorRT-LLM; the live competitive number is no longer 'vLLM 24x vs Orca/TGI' but cross-engine: SGLang's RadixAttention holds roughly a 29% aggregate-throughput edge over vLLM on H100 (~16,200 vs ~12,500 tok/s, Llama 3.1 8B), widening to up to ~6.4x on prefix-heavy RAG/multi-turn and shrinking to near-zero on unique-prompt batches; continuous batching itself still delivers ~3-4x over naive/static batching.
- **Leading systems:** SGLang (RadixAttention), vLLM v1 (continuous batching + automatic prefix caching), TensorRT-LLM (in-flight batching)
- **Source:** [https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison) (2026)
- **fak:** no-claim — no number (in-flight)
- **fak note:** LIVE BATCHING SHIPPED ON CONVERGED MAIN, NO H100 vLLM/SGLANG HEAD-TO-HEAD YET: NativeScheduler (internal/modelengine) and CUDA cohorts ship live concurrent batching — RTX 5080 C16 ragged cohort prefill reaches 372.076 agg E2E tok/s (2.70x vs lockstep 137.702, commit 507693d73b), lockstep C24 f16 KV hits 783.7 agg decode tok/s (2e7682ba2a), and Wave-2 C4_p2048 HTTP reaches 130.44 agg E2E tok/s vs realugbun/llama.cpp 122.47 (C2_p512 still trails 121.16 vs 139.58). Because no comparable req/s or tok/s run exists against H100 SGLang/vLLM, verdict stays no-claim.
- **Trace:** 507693d73b · docs/benchmarks/published/HEADLINES.json · internal/modelengine/nativesched.go (NativeScheduler continuous batching); docs/benchmarks/published/HEADLINES.json (507693d73b C16 cohort 372.076 agg tok/s; 2e7682ba2a C24 783.7 tok/s); docs/benchmarks/published/rtx5080-inkling2-v060/perf-receipt-v2.json (C4_p2048 130.44 agg E2E)

### ○ Absolute aggregate tokens/s at scale on standardized hardware (MLPerf Inference) — fak: **no-claim**

*Why it matters:* Cross-vendor, audited absolute throughput on a fixed model and SLO is what a buyer uses to compare full systems and compute cost-per-token. MLPerf Inference is the only neutral, reproducible apples-to-apples leaderboard with both offline (max throughput) and server (latency-constrained) scenarios, so it anchors the absolute SOTA bar that engine micro-benchmarks cannot.

- **SOTA bar:** MLPerf Inference v6.0 (results 2026-03-30 / published 2026-04-01) supersedes v5.0 and v5.1, and the headline model is now DeepSeek-R1 (671B MoE reasoning), not Llama 3.1 405B: NVIDIA's largest-ever submission of 288 Blackwell Ultra GPUs (4x GB300 NVL72 over Quantum-X800 InfiniBand) hit ~2.494 million tokens/second aggregate in the offline scenario, at 9,821 tok/s/GPU offline (8,064 tok/s/GPU server) — up to 2.7x higher than the GB300 debut six months prior via TensorRT-LLM/Dynamo software.
- **Leading systems:** NVIDIA GB300 NVL72 x4 (288 Blackwell Ultra GPUs, TensorRT-LLM/Dynamo), NVIDIA GB300 NVL72 single rack, CoreWeave / Nebius GB300 NVL72
- **Source:** [https://www.storagereview.com/news/nvidia-sets-mlperf-inference-v6-0-records-with-blackwell-ultra-platform](https://www.storagereview.com/news/nvidia-sets-mlperf-inference-v6-0-records-with-blackwell-ultra-platform) (2026-04-01)
- **fak:** no-claim — no number (stub)
- **fak note:** EXCLUDE / out-of-scope for fak-native's mid-market 1-8 GPU target. MLPerf Inference v6.0 (2026-03-30) ranks aggregate tok/s on 288-GPU (4x GB300 NVL72) DeepSeek-R1 671B submissions (~2.494M tok/s). fak targets 1-8 GPU mid-market (26B-27B native on consumer/datacenter GPUs, plus GLM-5.2 MoE on 1 datacenter GPU) and has no MLPerf submission.
- **Trace:** No MLPerf submission or standardized-hardware aggregate tok/s in BENCHMARK-AUTHORITY.md. The only concurrent server number is the GPU server run (data.json 'served-throughput-vs-sglang', 1085.6 tok/s @ conc 64) which is fak-gateway-fronting-SGLang on non-MLPerf hardware/harness, not a standardized MLPerf result.

### ○ Latency-vs-throughput Pareto frontier — fak: **no-claim**

*Why it matters:* There is no single operating point - every system trades batch size against per-request latency. The buyer's real question is the shape of the whole frontier: how much throughput can I buy before TTFT/TPOT breaches my SLO, and how gracefully does the curve bend. A system that advances the frontier serves more users per GPU at the SAME latency guarantee.

- **SOTA bar:** The latency-throughput Pareto frontier is now framed around goodput (max request rate meeting both TTFT and TBT/TPOT SLOs), and the dominant lever has moved from Sarathi-Serve chunked prefill (2024) to prefill-decode (PD) disaggregation: DistServe reports up to 7.4x more goodput or 12.6x stricter SLOs than prior systems while keeping >90% of requests in budget, and 2025-2026 work (DynaServe, MuxWise, DuetServe, RAPID-Serve) pushes intra-GPU PD multiplexing to recover disaggregation's isolation without dedicating GPU pools.
- **Leading systems:** DistServe (PD disaggregation), Mooncake / Splitwise (PD disaggregation), Sarathi-Serve (chunked prefill, now a baseline point), DuetServe / MuxWise / DynaServe (intra-GPU PD multiplexing, 2025-2026)
- **Source:** [https://haoailab.com/blogs/distserve/](https://haoailab.com/blogs/distserve/) (2024-2026)
- **fak:** no-claim — no number (stub)
- **fak note:** fak has not recorded a full P99-TTFT/TPOT vs RPS Pareto frontier. Adjacent committed measurements show C1-C4 RTX 5080 median TTFT+agg E2E (rtx5080-inkling2-v060) and 8-GPU datacenter server fak-gateway vs raw SGLang throughput where fak trails at 0.75x peak (peak-serving-throughput-per-gpu). Named as a gap.
- **Trace:** experiments/qwen36/gpu-server-r4-20260622/compare.json + docs/benchmarks/QWEN36-27B-GPU-SERVER-RESULTS.md: fak-gateway vs raw SGLang on 8-GPU datacenter server / Qwen3.6-27B, concurrency sweep 1->128; fak/raw = 0.60x (conc 8) -> 0.75x (conc 64 peak, 1085.6 vs 1451.6 tok/s) -> 0.97x (conc 128).

### ▼ Peak served throughput per accelerator (model-served-by-engine, MLPerf-grounded) — fak: **trails**

*Why it matters:* Throughput per GPU is the denominator of cost-per-token and the headline capacity number every operator sizes a fleet against. MLPerf is the only neutral, audited cross-vendor source, so it anchors the SOTA bar buyers actually trust over vendor blogs.

- **SOTA bar:** MLPerf Inference v6.0 (2026-03-30) is the current round and the audited per-accelerator peak: DeepSeek-R1 671B on GB300 NVL72 reaches 9,821 tok/s/GPU offline (8,064 tok/s/GPU accuracy+latency-constrained server); the full 288-GPU (4x GB300 NVL72) submission aggregates ~2.494M tok/s (see absolute-tokens-per-second-mlperf). The audited single-H200 Llama-2-70B peak of ~33,000 tok/s (40% over H100) still stands for that older/smaller workload. The prior v5.1/v5.0 ~1.1M tok/s/rack Azure GB300 figure was an unverified submission and is superseded.
- **Leading systems:** NVIDIA GB300 NVL72 (DeepSeek-R1 671B, MLPerf v6.0, TensorRT-LLM/Dynamo), NVIDIA H200 (Llama-2-70B audited), MangoBoost/AMD MI300X
- **Source:** [https://www.storagereview.com/news/nvidia-sets-mlperf-inference-v6-0-records-with-blackwell-ultra-platform](https://www.storagereview.com/news/nvidia-sets-mlperf-inference-v6-0-records-with-blackwell-ultra-platform) (2026-04-01)
- **fak:** trails — 1085.6 tok/s @ conc 64 (peak) (shipped)
- **fak note:** THE honest counter to the fleet 'lead' rows: this is the ONLY real concurrent-serving head-to-head against a LIVE SGLang process (same 8-GPU datacenter server / 27B / load harness, fak-in-front vs raw). fak TRAILS — it is the gateway/adjudication TAX: 0.60× worst (conc 8) → 0.75× at peak (conc 64) → ~0.97× converged at saturation (conc 128). fak's value here is the adjudication/coherence/measurement plane, NOT raw tok/s; the 4.1× fleet 'lead' is reuse-WORK eliminated on fak's own kernel held constant, a different axis from a throughput race against a tuned engine.
- **Trace:** experiments/qwen36/gpu-server-r4-20260622/compare.json · docs/benchmarks/QWEN36-27B-GPU-SERVER-RESULTS.md

## Scheduling & routing (`scheduling`)

### ○ Chunked / piggyback prefill and prefill-decode interference (stall-free batching) — fak: **no-claim**

*Why it matters:* When a long prefill shares an iteration with ongoing decodes it stalls every decode, spiking inter-token latency (TPOT/TBT). Chunked prefill splits a prefill into token-bounded chunks and piggybacks them onto decode batches so decodes never stall, decoupling the throughput/latency tradeoff. It is the core mechanism that lets a co-located engine hit a tight TPOT SLO at high load.

- **SOTA bar:** Proactive intra-GPU prefill/decode disaggregation (Nexus) now beats chunked-prefill, delivering up to 2.2x higher throughput, 20x lower TTFT, and 2.5x lower TBT vs vLLM (and up to 2x vs SGLang), superseding Sarathi-Serve's 5.6x chunked-prefill capacity claim.
- **Leading systems:** Nexus (proactive intra-GPU PD disaggregation, arXiv 2507.06608), DuetServe (adaptive GPU multiplexing, arXiv 2511.04791), Sarathi-Serve (chunked prefill, original bar)
- **Source:** [https://arxiv.org/abs/2507.06608](https://arxiv.org/abs/2507.06608) (2025-08)
- **fak:** no-claim — no number (in-flight)
- **fak note:** 512-token prefill chunking and NativeScheduler exist in code, but fak has no measured mixed-long/short TBT/ITL stall-free capacity benchmark against Nexus, DuetServe, or Sarathi-Serve. Honest no-claim.
- **Trace:** internal/ggufload/inkling2loop_prefill.go (512-token chunked prefill) & internal/modelengine/nativesched.go; no TBT-SLO interference benchmark vs Nexus/Sarathi-Serve

### ≈ Request scheduling and cache-aware routing (FCFS vs priority vs prefix/KV-cache-aware) — fak: **parity**

*Why it matters:* The scheduling discipline decides which requests batch together and where they route. A naive FCFS engine ignores that requests sharing a prefix (system prompts, RAG context, multi-turn chat, agents) can reuse KV cache; a cache-aware scheduler co-locates and orders them to maximize reuse. This directly multiplies effective throughput on shared-context workloads and trades off against load balance.

- **SOTA bar:** SGLang's cache-aware load balancer still holds the cited bar: up to 1.9x throughput and 3.8x higher prefix-cache hit rate in multi-node deployments, via an approximate radix tree mirroring each worker's cache.
- **Leading systems:** SGLang router / cache-aware load balancer (v0.4+, current v0.5.x)
- **Source:** [https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/) (2024-12)
- **fak:** parity — 86.7 % (shipped)
- **fak note:** fak measured on-box: cache-aware (≡ DFS) scheduling recovers FCFS 62.1% → 86.7% = 100% of optimal, inside SGLang's published band. Split-reuse proven bit-identical to recompute (max|Δ|=0), so the hit is a real reuse, not a numerics shortcut. FENCE: 86.7% is on radixbench's synthetic 'agents' workload (deliberately cache-favorable); on a REAL tau2-airline trace fak's measured addressable purity is ~0.7% (CLAIMS vDSO unit 33/83) — so this is a same-workload-as-SGLang reuse-mechanism parity, not a promise of 86.7% on every production corpus.
- **Trace:** 92896a4 · experiments/radixattention/radixbench-qwen2.5-1.5b-q8-agents-fresh-20260619.json · data.json row 'kv-hit-rate'; BENCHMARK-AUTHORITY.md L23 + 'RadixAttention Model Ladder' (FCFS 62.1% -> cache-aware 86.7% = 100% of optimal); artifact radixbench-qwen2.5-1.5b-q8-agents-fresh-20260619.json (verify workloads[0].cache_hit_rate=0.8666666666666667). Commit 92896a4.

## Latency & SLO (`latency`)

### ○ Interactive latency SLO attainment: TTFT and TPOT/ITL tail under load — fak: **no-claim**

*Why it matters:* For chatbots, coding assistants, and agents the contract is felt latency: time-to-first-token and steady per-token cadence at the tail (p99), not the mean. A serving system is differentiated by how high it can push concurrency before TTFT or TPOT breach the SLO. This is the latency face of the same tradeoff goodput measures on the throughput side.

- **SOTA bar:** MLPerf Inference v6.0 (Apr 2026) tightened the interactive SLO floor: the most stringent latency-aware bounds are now TPOT <= 15 ms with TTFT <= 1.5-2.0 s P99 (GPT-OSS 120B: TTFT <= 2.0 s / TPOT <= 15 ms; DeepSeek-R1: TTFT <= 1.5 s / TPOT <= 15 ms), superseding the v5.1 30 ms-TPOT / 0.5 s-TTFT interactive bar.
- **Leading systems:** MLPerf Inference v6.0 (NVIDIA Blackwell Ultra GB300, AMD MI355X), MLPerf Inference v5.1
- **Source:** [https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/](https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/) (2026-03-24)
- **fak:** no-claim — no number (stub)
- **fak note:** fak ships median C1-C4 HTTP TTFT/decode sweeps on RTX 5080 (rtx5080-inkling2-v060) and 8-GPU gateway-fronting-SGLang P50/P99 E2EL (trailing raw SGLang 0.75x throughput at conc 64), but no P99 TTFT+TPOT joint interactive SLO pass-rate measurement matching MLPerf v6.0 interactive bounds.
- **Trace:** data.json row 'served-throughput-vs-sglang'; artifact experiments/qwen36/gpu-server-r4-20260622/compare.json; docs/benchmarks/QWEN36-27B-GPU-SERVER-RESULTS.md s2. fak 1085.6 vs raw SGLang 1451.6 tok/s @ conc 64 on 8-GPU datacenter server, Qwen3.6-27B.

### ○ Time-to-first-token (TTFT) under realistic prefill — fak: **no-claim**

*Why it matters:* TTFT is the perceived responsiveness of an interactive system - the wait before any output appears. It is dominated by queueing delay plus prefill compute, so it is the headline latency metric for chat, voice, and copilot UX and the gating constraint in every serving SLO. Buyers compare TTFT at a fixed input length and concurrency because a single-user TTFT (just network + one forward pass) tells you almost nothing about production behavior.

- **SOTA bar:** MLPerf Inference v6.0 defines per-model P99 TTFT SLOs: GPT-OSS 120B interactive TTFT <= 2.0 s (server <= 3.0 s), DeepSeek-R1 interactive TTFT <= 1.5 s (server <= 2.0 s), Llama-3.1-405B interactive TTFT <= 4.5 s (server <= 6.0 s). The 4.5 s P99 405B bar from v5.1 still stands for the largest model.
- **Leading systems:** MLPerf Inference v6.0 (DeepSeek-R1 interactive), MLPerf Inference v5.1 (Llama-3.1-405B interactive)
- **Source:** [https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/](https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/) (2026-03-24)
- **fak:** no-claim — no number (in-flight)
- **fak note:** MEDIAN HTTP TTFT MEASURED ON RTX 5080, P99 SLO DISTRIBUTION UNMEASURED: Wave-2 rtx5080-inkling2-v060 (<=16K f16 KV) witnesses median TTFT vs realugbun/llama.cpp: C1_p512 274.35 vs 267.81 ms (trails), C1_p2048 638.20 vs 723.77 ms, C2_p512 351.33 vs 530.63 ms, C4_p512 598.89 vs 981.30 ms, and C4_p2048 2017.61 vs 3718.34 ms (1.84x faster TTFT; 2.32x vs stock 4683.83 ms). Earlier C1 NVFP4 W4A4 pp512 run saw median first visible token trail (1165.6 vs 195.5 ms). Because this dimension's SOTA bar scores P99 TTFT (1.5 s DeepSeek-R1), verdict stays no-claim without percentile TTFT.
- **Trace:** be228d79ef · docs/benchmarks/published/rtx5080-inkling2-v060/perf-receipt-v2.json · docs/benchmarks/published/rtx5080-inkling2-v060/perf-receipt-v2.json (C1-C4 <= 16K median HTTP TTFT vs llama.cpp); docs/benchmarks/published/aegis-inkling2-prefill-nvfp4-w4a4/RESULTS.md (visible TTFT caveat); experiments/qwen36/gpu-server-r4-20260622/compare.json

### ○ Inter-token latency / time-per-output-token (ITL/TPOT) — fak: **no-claim**

*Why it matters:* After the first token, ITL/TPOT governs how fast text streams - the felt 'typing speed.' It must clear a per-user threshold (e.g. faster than human reading) and stay smooth; stalls produce visible jank even at good averages. TPOT excludes the first token by construction (denominator subtracts 1) so it isolates the decode phase from prefill.

- **SOTA bar:** MLPerf Inference v6.0 reasoning-model interactive scenarios now require TPOT <= 15 ms P99 (GPT-OSS 120B and DeepSeek-R1), down from the prior 80 ms TPOT bar; the legacy Llama-2-70B interactive bound is 40 ms TPOT (~25 tok/s/user) and server is 80-200 ms.
- **Leading systems:** MLPerf Inference v6.0 (GPT-OSS 120B, DeepSeek-R1 interactive), MLPerf Inference v5.1
- **Source:** [https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/](https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/) (2026-03-24)
- **fak:** no-claim — no number (in-flight)
- **fak note:** MEAN STREAM DECODE MEASURED, P99 TPOT DISTRIBUTION UNMEASURED: Wave-2 RTX 5080 HTTP sweep records C1_p512 102.45 vs realugbun 118.63 tok/s (~9.8 vs 8.4 ms/tok mean) and C4_p2048 43.72 vs 55.30 tok/s/stream (~22.9 vs 18.1 ms/tok), while in-process C1 reaches 122.34 tok/s. Because MLPerf v6.0 scores P99 TPOT (<=15 ms), verdict stays no-claim pending SSE P99 TPOT percentiles.
- **Trace:** be228d79ef · docs/benchmarks/published/rtx5080-inkling2-v060/perf-receipt-v2.json · docs/benchmarks/published/rtx5080-inkling2-v060/perf-receipt-v2.json (C1-C4 stream decode rates); experiments/qwen36/gpu-server-r4-20260622/compare.json

### ▼ End-to-end request latency (E2EL) — fak: **trails**

*Why it matters:* E2EL is the full submit-to-final-token time and is what a downstream agent or pipeline actually budgets against. It bundles queueing, prefill, all decode steps, batching effects, and network, so it is the only metric that captures the compound cost of long generations and is the right unit for multi-step agent SLA math.

- **SOTA bar:** End-to-end request latency (TTFT + total decode) is reported per request by the standard tools (NVIDIA GenAI-Perf, vLLM bench serve, LLMPerf) with P99 as the default percentile; there is no single SOTA number - it is a tunable composite governed by the per-model MLPerf v6.0 TTFT+TPOT SLOs (e.g. a 1000-token DeepSeek-R1 interactive response budgets ~1.5 s TTFT + ~15 s decode).
- **Leading systems:** NVIDIA GenAI-Perf, vLLM bench serve (--metric-percentiles, default p99), LLMPerf
- **Source:** [https://docs.vllm.ai/en/latest/benchmarking/cli/](https://docs.vllm.ai/en/latest/benchmarking/cli/) (2026-01)
- **fak:** trails — 9026.9 ms E2EL P50 @ conc 64 (shipped)
- **fak note:** WITNESSED gateway tax, the honest framing. Same 8-GPU datacenter server / Qwen3.6-27B / load harness, fak-gateway-in-front vs raw SGLang (the gateway is the only delta). At peak conc 64 fak's E2EL P50 is 9026.9 ms vs raw 7222.0 ms -> a +1804.9 ms (0.80x) end-to-end gateway tax. This is the FULL fak-in-front delta (proxy relay overhead PLUS the queueing induced by fak's lower peak throughput, 0.75x; see peak-serving-throughput-per-gpu), NOT just the in-process adjudication primitive (Decide 362 ns / AdmitChain <=87 us idle-box) which is only the small floor under it. fak fronts the backend and does NOT claim to win the raw-latency race; its value is the adjudication/coherence/measurement plane.
- **Trace:** 044220f8 · experiments/qwen36/gpu-server-r4-20260622/compare.json · docs/benchmarks/QWEN36-27B-GPU-SERVER-RESULTS.md

### ▼ Tail latency distribution (P50/P95/P99/P99.9) — fak: **trails**

*Why it matters:* Averages hide the requests that ruin trust. A degrading P99 while P50 holds steady is the early-warning signature of queue buildup or KV-memory pressure. At scale a 1-in-100 bad request is a continuous problem (100/hr at 10k req/hr), so operators size capacity and write SLOs against tail percentiles, not means.

- **SOTA bar:** P99 (99th percentile) is the canonical tail SLO across MLPerf Inference v6.0 (all TTFT/TPOT constraints are '99th percentile <= X'), NVIDIA GenAI-Perf/NIM, and vLLM bench (default --metric-percentiles p99). P99 remains the industry tail metric; some leaderboards also surface P95/P50.
- **Leading systems:** MLPerf Inference v6.0 (99th-percentile TTFT/TPOT constraints), vLLM bench (p99 default), NVIDIA GenAI-Perf / NIM
- **Source:** [https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/](https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/) (2026-03-24)
- **fak:** trails — 9997.6 ms request-latency P99 @ conc 64 (shipped)
- **fak note:** WITNESSED full request-latency tail on the same harness, fak-gateway-in-front vs raw SGLang. At peak conc 64 the committed distribution is fak P50/P95/P99 = 9026.9 / 9979.9 / 9997.6 ms vs raw 7222.0 / 8134.7 / 8157.9 ms; the P99 gateway tax is +1839.7 ms (0.82x). The tail is NOT widened by the gateway: fak's P99/P50 spread is 1.11x vs raw 1.13x, so fak shifts the WHOLE distribution by the ~+1.8 s tax rather than fattening the tail. Honest framing: fak fronts the backend and adds a measured tax, it does not claim to win the raw-latency race. (Sub-ms in-process primitive percentiles -- Decide 362 ns / AdmitChain <=87 us idle-box, BENCHMARK-AUTHORITY.md mac-m3pro-kernel-20260620 -- are the floor, not this under-load serving distribution.)
- **Trace:** 044220f8 · experiments/qwen36/gpu-server-r4-20260622/compare.json · docs/benchmarks/QWEN36-27B-GPU-SERVER-RESULTS.md

### ○ Prefill-decode interference & chunked-prefill TTFT/TPOT tension — fak: **no-claim**

*Why it matters:* When a long-context prefill shares a GPU batch with short interactive requests, it starves decode steps and spikes ITL/TPOT for everyone in the batch - and conversely, protecting decode can balloon TTFT. How a scheduler resolves this tension (chunked prefill, stall-free batching) is a primary differentiator of mixed-workload latency stability, which naive single-shape benchmarks never reveal.

- **SOTA bar:** Chunked prefill cuts P95 inter-token latency by ~68% under mixed long/short workloads (e.g. 2,800 ms -> 890 ms P95 ITL at 32K inputs) by prioritizing decode and chunking prefill into the token budget.
- **Leading systems:** vLLM chunked prefill (decode-prioritized scheduling), SGLang chunked prefill
- **Source:** [https://www.spheron.network/blog/llm-serving-optimization-continuous-batching-paged-attention/](https://www.spheron.network/blog/llm-serving-optimization-continuous-batching-paged-attention/) (2026-06)
- **fak:** no-claim — no number (in-flight)
- **fak note:** OUT OF SCOPE / GAP: 512-token prefill chunking and NativeScheduler exist for native cohorts, and fak fronts SGLang/llama.cpp on external GPUs, but no 32K mixed-load P95 ITL interference ablation vs Sarathi-Serve/vLLM is measured.
- **Trace:** internal/ggufload/inkling2loop_prefill.go & internal/modelengine/nativesched.go; no 32K P95 ITL interference bench

## Single-stream (one chat) (`single-stream`)

### ▼ Single-stream (one chat) decode throughput on CPU — fak: **trails**

*Why it matters:* The regime local/edge stacks (llama.cpp, Ollama, MLC) compete hardest on: one user, one chat, raw tokens/s on commodity hardware. fak explicitly does NOT target this — showing the loss keeps the scorecard honest.

- **SOTA bar:** llama.cpp Metal/CPU leads single-stream decode on the same box (e.g. Qwen2.5-7B Q8 ~17.3 tok/s Metal, M3 Pro).
- **Leading systems:** llama.cpp, Ollama, MLC-LLM
- **Source:** [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (2026-06)
- **fak:** trails — 8.7 tok/s (shipped)
- **fak note:** fak explicitly does NOT target single-stream raw throughput — one chat → use llama.cpp. The gap narrows with model size (0.39× → 0.53×, 1.5B → 7B) toward the shared ~150 GB/s unified-memory bandwidth floor; arm64 still lacks the register-blocked int8 GEMM tile that already exists on x86.
- **Trace:** 34c74f4 · model-ladder/modelbench-qwen25-7b-q8.json · QWEN25-7B-RESULTS.md

### ▼ Single-stream prefill throughput, apples-to-apples CPU-vs-CPU — fak: **trails**

*Why it matters:* Prefill (prompt ingestion) speed on CPU is a core local-inference axis; fak's x86 Zen5 per-token compute slope reaches parity while its arm64 integer prefill path historically trailed llama.cpp CPU.

- **SOTA bar:** llama.cpp CPU sets the prefill bar on the same box + Q8 weights (1× reference); fak arm64 is ~0.12× until the int8 tile lands.
- **Leading systems:** llama.cpp
- **Source:** [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (2026-06)
- **fak:** trails — 0.12 × of llama.cpp CPU (shipped)
- **fak note:** Apples-to-apples CPU-vs-CPU is ~0.12x on arm64 (historical MODEL-LADDER-VS-SOTA 2026-06-21; internal/compute/arm64tile landed since without a newer Qwen2.5-7B remeasure); x86 Zen5 per-token compute slope reached parity (1.03x). GPU single-stream prefill lives on single-stream-prefill-gpu.
- **Trace:** 34c74f4 · MODEL-LADDER-VS-SOTA-2026-06-21.md · QWEN25-7B-RESULTS.md

### ≈ Single-stream decode throughput on a consumer GPU — fak: **parity**

*Why it matters:* Local GPU single-stream decode (the RTX-class laptop/desktop regime) is where llama.cpp's CUDA backend sets the bar; fak reaches parity with tuned forks on RTX 5080 sm_120 short context while trailing at depth 1946.

- **SOTA bar:** On RTX 5080 16 GB (-ngl 99, Inkling-2-Mini UD-Q3_K_XL), realugbun/llama.cpp (2eb2530e) decodes at 119.47 tok/s (upstream llama.cpp ad50351f5 at 80.99 tok/s); prior RTX 4070 Q8_0 reference is ~120 ± 15 tok/s.
- **Leading systems:** realugbun/llama.cpp fork (CUDA, RTX 5080), upstream llama.cpp (CUDA)
- **Source:** [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (2026-11)
- **fak:** parity — 122.34 tok/s (Inkling-2-Mini UD-Q3_K_XL, RTX 5080) (shipped)
- **fak note:** HW-WITNESSED on RTX 5080 16 GB (sm_120f, Inkling-2-Mini-UD-Q3_K_XL): v0.60.0 in-process decode reaches 122.34 tok/s (TestInkling2CUDADecodeBench, kv=f16), and interleaved ABBA fork parity reaches 119.89 vs realugbun/llama.cpp 119.47 tok/s (1.51x upstream llama.cpp 80.99). Different harnesses (Go bench vs llama-bench), so read as parity, not a lead; at depth 1946 plain decode still trails the fork (88.97 vs 113.81 tok/s).
- **Trace:** be228d79ef · docs/benchmarks/published/rtx5080-inkling2-v060/perf-receipt-v2.json · docs/benchmarks/published/rtx5080-inkling2-v060/RESULTS.md; docs/benchmarks/published/aegis-inkling2-fork-parity/RESULTS.md; BENCHMARK-AUTHORITY.md

### ▲ Single-stream prefill throughput on a consumer GPU (prompt ingestion tok/s) — fak: **lead**

*Why it matters:* Prompt ingestion speed on a single consumer GPU governs TTFT on long repo context and agent tool turns. On Blackwell sm_120, grouped W4A4 tensor-core prefill (NVFP4) vs standard MMQ/cuBLAS GGUF kernels decides cold prompt cost.

- **SOTA bar:** On RTX 5080 16 GB (Inkling-2-Mini UD-Q3_K_XL HTTP pp512), realugbun/llama.cpp (2eb2530e) ingests prompts at 3012.4 tok/s (upstream llama.cpp 33aeb17 at 2144.6 tok/s); on dense Qwen3.8-27B Q4_K_M (RTX 5090 Laptop), llama.cpp b11870 hits 904.1 tok/s at P=512 and 1053.2 tok/s at P=2048.
- **Leading systems:** realugbun/llama.cpp fork (CUDA RTX 5080 UD-Q3_K_XL), upstream llama.cpp (CUDA RTX 5080 / RTX 5090 Laptop)
- **Source:** [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (2026-11)
- **fak:** lead — 5861 tok/s HTTP pp512 server prompt rate (Inkling-2-Mini-UD-Q3_K_XL, RTX 5080) (shipped)
- **fak note:** HW-WITNESSED on RTX 5080 16 GB (sm_120f, Inkling-2-Mini-UD-Q3_K_XL): fak executes the checkpoint's NVFP4-quantized experts via native SM120 NVFP4 W4A4 grouped prefill (default at 27355b23dd), reaching 5861.0 tok/s HTTP pp512 server prompt rate (1.95x realugbun/llama.cpp fork 3012.4 tok/s, 2.73x upstream llama.cpp 2144.6 tok/s on the exact same UD-Q3_K_XL GGUF — llama.cpp runs its standard CUDA UD-Q3_K_XL kernels, NOT NVFP4 W4A4 activations), and 6037.6/6038.2/6033.9 vs 3337.4/3537.8/3648.6 tok/s at C2/C4/C8 pooled median (rev e7ad0c8dee5, 1 output token). HONEST FENCES: (1) server prompt rate only — on that C1 pp512 run median first visible token still trailed (1165.6 vs 195.5 ms), though Wave-2 C4_p2048 HTTP TTFT leads at 2017.61 vs 3718.34 ms (rtx5080-inkling2-v060); (2) on dense Qwen3.8-27B-Q4_K_M (RTX 5090 Laptop, cuda-prefill-sm120-20261106), fak default Q4_K MMQ prefill still trails llama.cpp b11870 at 119.2 vs 904.1 tok/s (pp512, 0.13x) and 153.8 vs 1053.2 tok/s (pp2048, 0.15x).
- **Trace:** 27355b23dd · docs/benchmarks/published/aegis-inkling2-prefill-nvfp4-w4a4/RESULTS.md · docs/benchmarks/published/aegis-inkling2-prefill-nvfp4-w4a4/RESULTS.md; README.md (C2/C4/C8 pooled median rev e7ad0c8dee5); docs/benchmarks/receipts/cuda-prefill-sm120-20261106/RESULTS.md (Q4_K_M MMQ)

