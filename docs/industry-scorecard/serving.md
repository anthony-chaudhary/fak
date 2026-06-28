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
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP fak should measure but currently OUT OF SCOPE for the kernel as shipped: CLAIMS labels continuous/in-flight batching [SIMULATED] read-only telemetry, not on the live serving path, and the polymodel decode lane is explicitly SERIAL (at-most-one-model-decodes-per-step, off-mainline). Iteration-level batching is the defining vLLM/SGLang throughput lever; fak has NO shipped number. fak's fleet win is prefix-reuse work-ELIMINATION, a different axis from concurrent-decode batching, so no parity is implied.
- **Trace:** CLAIMS.md L163 ([SIMULATED] continuous-batching read-only telemetry, not on live serving path); CLAIMS.md L6 (SIMULATED tag definition); existing data.json row 'continuous-batching'.

### ○ Absolute aggregate tokens/s at scale on standardized hardware (MLPerf Inference) — fak: **no-claim**

*Why it matters:* Cross-vendor, audited absolute throughput on a fixed model and SLO is what a buyer uses to compare full systems and compute cost-per-token. MLPerf Inference is the only neutral, reproducible apples-to-apples leaderboard with both offline (max throughput) and server (latency-constrained) scenarios, so it anchors the absolute SOTA bar that engine micro-benchmarks cannot.

- **SOTA bar:** MLPerf Inference v6.0 (results 2026-03-30 / published 2026-04-01) supersedes v5.0 and v5.1, and the headline model is now DeepSeek-R1 (671B MoE reasoning), not Llama 3.1 405B: NVIDIA's largest-ever submission of 288 Blackwell Ultra GPUs (4x GB300 NVL72 over Quantum-X800 InfiniBand) hit ~2.494 million tokens/second aggregate in the offline scenario, at 9,821 tok/s/GPU offline (8,064 tok/s/GPU server) â€” up to 2.7x higher than the GB300 debut six months prior via TensorRT-LLM/Dynamo software.
- **Leading systems:** NVIDIA GB300 NVL72 x4 (288 Blackwell Ultra GPUs, TensorRT-LLM/Dynamo), NVIDIA GB300 NVL72 single rack, CoreWeave / Nebius GB300 NVL72
- **Source:** [https://www.storagereview.com/news/nvidia-sets-mlperf-inference-v6-0-records-with-blackwell-ultra-platform](https://www.storagereview.com/news/nvidia-sets-mlperf-inference-v6-0-records-with-blackwell-ultra-platform) (2026-04-01)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE BY DESIGN for a reuse kernel: MLPerf Inference v6.0 (2026-03-30) ranks absolute aggregate tok/s on standardized Blackwell-Ultra hardware, and its headline model is now DeepSeek-R1 671B MoE, not Llama-3.1-405B — NVIDIA's 288-GPU (4x GB300 NVL72) submission hit ~2.494M tok/s aggregate / 9,821 tok/s/GPU offline, superseding the old v5.x GB200 / 405B bar. fak has no MLPerf submission and its own generic engine ceilings at ~7B (data.json model-size-ceiling), so it is not in this race and does NOT intend an own-engine MLPerf-shaped 671B number; on a frontier MoE it FRONTS SGLang/llama.cpp rather than serving them itself. fak's actual DeepSeek-class-MoE serving path is the in-kernel GLM-5.2 work (#1010, #1026): MoE/FFN experts + router on its own pure GPU kernel (cosine=1.0, commit 498a4ab) but on a SINGLE device, so no standardized fak tok/s exists. See max-model-and-moe-coverage.
- **Trace:** No MLPerf submission or standardized-hardware aggregate tok/s in BENCHMARK-AUTHORITY.md. The only concurrent server number is the GPU server run (data.json 'served-throughput-vs-sglang', 1085.6 tok/s @ conc 64) which is fak-gateway-fronting-SGLang on non-MLPerf hardware/harness, not a standardized MLPerf result.

### ○ Latency-vs-throughput Pareto frontier — fak: **no-claim**

*Why it matters:* There is no single operating point - every system trades batch size against per-request latency. The buyer's real question is the shape of the whole frontier: how much throughput can I buy before TTFT/TPOT breaches my SLO, and how gracefully does the curve bend. A system that advances the frontier serves more users per GPU at the SAME latency guarantee.

- **SOTA bar:** The latency-throughput Pareto frontier is now framed around goodput (max request rate meeting both TTFT and TBT/TPOT SLOs), and the dominant lever has moved from Sarathi-Serve chunked prefill (2024) to prefill-decode (PD) disaggregation: DistServe reports up to 7.4x more goodput or 12.6x stricter SLOs than prior systems while keeping >90% of requests in budget, and 2025-2026 work (DynaServe, MuxWise, DuetServe, RAPID-Serve) pushes intra-GPU PD multiplexing to recover disaggregation's isolation without dedicating GPU pools.
- **Leading systems:** DistServe (PD disaggregation), Mooncake / Splitwise (PD disaggregation), Sarathi-Serve (chunked prefill, now a baseline point), DuetServe / MuxWise / DynaServe (intra-GPU PD multiplexing, 2025-2026)
- **Source:** [https://haoailab.com/blogs/distserve/](https://haoailab.com/blogs/distserve/) (2024-2026)
- **fak:** no-claim — no number (stub)
- **fak note:** fak has not recorded a latency-vs-throughput Pareto frontier (no P99 TTFT/TPOT vs request-rate). The adjacent committed evidence is a throughput-vs-concurrency sweep where fak trails raw SGLang (see peak-serving-throughput-per-gpu). Named as a gap, not a Pareto measurement.
- **Trace:** experiments/qwen36/dgx-r4-20260622/compare.json + docs/benchmarks/QWEN36-27B-GPU-SERVER-RESULTS.md: fak-gateway vs raw SGLang on 8-GPU datacenter server / Qwen3.6-27B, concurrency sweep 1->128; fak/raw = 0.60x (conc 8) -> 0.75x (conc 64 peak, 1085.6 vs 1451.6 tok/s) -> 0.97x (conc 128).

### ▼ Peak served throughput per accelerator (model-served-by-engine, MLPerf-grounded) — fak: **trails**

*Why it matters:* Throughput per GPU is the denominator of cost-per-token and the headline capacity number every operator sizes a fleet against. MLPerf is the only neutral, audited cross-vendor source, so it anchors the SOTA bar buyers actually trust over vendor blogs.

- **SOTA bar:** MLPerf Inference v6.0 (2026-03-30) is the current round and the audited per-accelerator peak: DeepSeek-R1 671B on GB300 NVL72 reaches 9,821 tok/s/GPU offline (8,064 tok/s/GPU accuracy+latency-constrained server); the full 288-GPU (4x GB300 NVL72) submission aggregates ~2.494M tok/s (see absolute-tokens-per-second-mlperf). The audited single-H200 Llama-2-70B peak of ~33,000 tok/s (40% over H100) still stands for that older/smaller workload. The prior v5.1/v5.0 ~1.1M tok/s/rack Azure GB300 figure was an unverified submission and is superseded.
- **Leading systems:** NVIDIA GB300 NVL72 (DeepSeek-R1 671B, MLPerf v6.0, TensorRT-LLM/Dynamo), NVIDIA H200 (Llama-2-70B audited), MangoBoost/AMD MI300X
- **Source:** [https://www.storagereview.com/news/nvidia-sets-mlperf-inference-v6-0-records-with-blackwell-ultra-platform](https://www.storagereview.com/news/nvidia-sets-mlperf-inference-v6-0-records-with-blackwell-ultra-platform) (2026-04-01)
- **fak:** trails — 1085.6 tok/s @ conc 64 (peak) (shipped)
- **fak note:** THE honest counter to the fleet 'lead' rows: this is the ONLY real concurrent-serving head-to-head against a LIVE SGLang process (same 8-GPU datacenter server / 27B / load harness, fak-in-front vs raw). fak TRAILS — it is the gateway/adjudication TAX: 0.60× worst (conc 8) → 0.75× at peak (conc 64) → ~0.97× converged at saturation (conc 128). fak's value here is the adjudication/coherence/measurement plane, NOT raw tok/s; the 4.1× fleet 'lead' is reuse-WORK eliminated on fak's own kernel held constant, a different axis from a throughput race against a tuned engine.
- **Trace:** experiments/qwen36/dgx-r4-20260622/compare.json · docs/benchmarks/QWEN36-27B-GPU-SERVER-RESULTS.md

## Scheduling & routing (`scheduling`)

### ○ Chunked / piggyback prefill and prefill-decode interference (stall-free batching) — fak: **no-claim**

*Why it matters:* When a long prefill shares an iteration with ongoing decodes it stalls every decode, spiking inter-token latency (TPOT/TBT). Chunked prefill splits a prefill into token-bounded chunks and piggybacks them onto decode batches so decodes never stall, decoupling the throughput/latency tradeoff. It is the core mechanism that lets a co-located engine hit a tight TPOT SLO at high load.

- **SOTA bar:** Proactive intra-GPU prefill/decode disaggregation (Nexus) now beats chunked-prefill, delivering up to 2.2x higher throughput, 20x lower TTFT, and 2.5x lower TBT vs vLLM (and up to 2x vs SGLang), superseding Sarathi-Serve's 5.6x chunked-prefill capacity claim.
- **Leading systems:** Nexus (proactive intra-GPU PD disaggregation, arXiv 2507.06608), DuetServe (adaptive GPU multiplexing, arXiv 2511.04791), Sarathi-Serve (chunked prefill, original bar)
- **Source:** [https://arxiv.org/abs/2507.06608](https://arxiv.org/abs/2507.06608) (2025-08)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for fak as a reuse kernel today, and an unmeasured GAP if pursued: Sarathi-Serve-style chunked/piggyback prefill caps per-iteration prefill work to bound TBT under decode interference. fak owns its own forward but has not built or measured chunked-prefill scheduling; no number exists, so no claim.
- **Trace:** No chunked-prefill / TBT-SLO / prefill-decode-interference number in BENCHMARK-AUTHORITY.md or CLAIMS.md. fak's prefill rows (single-stream-prefill, radix-live-vs-ceiling) are single-stream or reuse, not stall-free chunked scheduling under load.

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
- **fak note:** fak ships no ms-level P99 TTFT/TPOT tail-latency number; the only live head-to-head is a THROUGHPUT sweep (it trails raw SGLang 0.75× at peak — see peak-serving-throughput-per-gpu). An honest gap on the latency-tail axis, not a measured trail.
- **Trace:** data.json row 'served-throughput-vs-sglang'; artifact experiments/qwen36/dgx-r4-20260622/compare.json; docs/benchmarks/QWEN36-27B-GPU-SERVER-RESULTS.md s2. fak 1085.6 vs raw SGLang 1451.6 tok/s @ conc 64 on 8-GPU datacenter server, Qwen3.6-27B.

### ○ Time-to-first-token (TTFT) under realistic prefill — fak: **no-claim**

*Why it matters:* TTFT is the perceived responsiveness of an interactive system - the wait before any output appears. It is dominated by queueing delay plus prefill compute, so it is the headline latency metric for chat, voice, and copilot UX and the gating constraint in every serving SLO. Buyers compare TTFT at a fixed input length and concurrency because a single-user TTFT (just network + one forward pass) tells you almost nothing about production behavior.

- **SOTA bar:** MLPerf Inference v6.0 defines per-model P99 TTFT SLOs: GPT-OSS 120B interactive TTFT <= 2.0 s (server <= 3.0 s), DeepSeek-R1 interactive TTFT <= 1.5 s (server <= 2.0 s), Llama-3.1-405B interactive TTFT <= 4.5 s (server <= 6.0 s). The 4.5 s P99 405B bar from v5.1 still stands for the largest model.
- **Leading systems:** MLPerf Inference v6.0 (DeepSeek-R1 interactive), MLPerf Inference v5.1 (Llama-3.1-405B interactive)
- **Source:** [https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/](https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/) (2026-03-24)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP fak should measure. TTFT is adjacent to fak's actual lever (prefix-reuse cuts prefill tokens, which IS the dominant TTFT term on a cache hit), yet fak has never reported TTFT as a percentile under realistic prefill/concurrency the way MLPerf v6.0 (DeepSeek-R1 1.5 s P99 TTFT interactive; 405B 4.5 s still defined) or vLLM/GenAI-Perf do. The adjacent committed evidence is the +1804.9 ms E2EL gateway tax at conc 64 (end-to-end-request-latency), which is the UPPER BOUND on the TTFT tax since the gateway's admission decision (Decide 362 ns / AdmitChain <=87 us idle-box) runs before the first upstream byte. No-claim stands until a STREAMING run isolates the per-request TTFT distribution -- host-gated on the GPU box. The streaming-capable harness already exists: `fak webbench serving --dataset FILE --tracks ours,sglang --endpoints ours=<fak-gateway/v1>,sglang=<raw-sglang/v1>` captures per-request ttft_ms/itl_ms/tpot_ms over SSE (internal/webbench/serving.go MeasureSSERequest, ServingTrackStats.TTFTMillis). The committed compare.json was the OLDER non-streaming runner, so the only gap is a live fak-fronts-SGLang streaming run on the GPU box (off-box here), not new code.
- **Trace:** No TTFT row exists in BENCHMARK-AUTHORITY.md or CLAIMS.md. fak commits prefill-WORK ratios (single-stream prefill 0.58x vs llama.cpp CPU; RadixAttention live prefill 4.58x-6.95x vs full re-prefill) but never a TTFT measured as a latency percentile at fixed ISL/concurrency. The committed fak-fronts-SGLang harness (experiments/qwen36/dgx-r4-20260622/compare.json) is NON-STREAMING -- it captures total request latency (E2EL, now committed; see end-to-end-request-latency) but cannot isolate the first-token wait, so it bounds but does not measure the TTFT tax.

### ○ Inter-token latency / time-per-output-token (ITL/TPOT) — fak: **no-claim**

*Why it matters:* After the first token, ITL/TPOT governs how fast text streams - the felt 'typing speed.' It must clear a per-user threshold (e.g. faster than human reading) and stay smooth; stalls produce visible jank even at good averages. TPOT excludes the first token by construction (denominator subtracts 1) so it isolates the decode phase from prefill.

- **SOTA bar:** MLPerf Inference v6.0 reasoning-model interactive scenarios now require TPOT <= 15 ms P99 (GPT-OSS 120B and DeepSeek-R1), down from the prior 80 ms TPOT bar; the legacy Llama-2-70B interactive bound is 40 ms TPOT (~25 tok/s/user) and server is 80-200 ms.
- **Leading systems:** MLPerf Inference v6.0 (GPT-OSS 120B, DeepSeek-R1 interactive), MLPerf Inference v5.1
- **Source:** [https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/](https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/) (2026-03-24)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP fak should measure. fak has decode-rate numbers (which trail llama.cpp single-stream, as it openly states it does not target single-stream) but never the per-output-token latency distribution MLPerf v6.0 encodes (<= 15 ms P99 TPOT interactive for DeepSeek-R1 / GPT-OSS 120B; the prior 80 ms is now the server-scenario bound). The adjacent committed evidence is the +1804.9 ms E2EL gateway tax at conc 64 (end-to-end-request-latency); since the gateway is a streaming pass-through that adjudicates ONCE at admission and does not re-decide per output token, most of that tax is front-loaded into TTFT, leaving a small per-token relay component -- but the non-streaming artifact cannot separate it. No-claim stands until a STREAMING run isolates the per-request TPOT/ITL distribution -- host-gated on the GPU box. The streaming-capable harness already exists: `fak webbench serving --dataset FILE --tracks ours,sglang --endpoints ours=<fak-gateway/v1>,sglang=<raw-sglang/v1>` captures per-request tpot_ms/itl_ms over SSE (internal/webbench/serving.go MeasureSSERequest, ServingTrackStats.TPOTMillis). The committed compare.json was the OLDER non-streaming runner, so the only gap is a live fak-fronts-SGLang streaming run on the GPU box (off-box here), not new code.
- **Trace:** fak commits single-stream DECODE throughput (8.7 tok/s 7B Q8 CPU, ~0.55-0.73x of llama.cpp at HEAD 374776a; ~120 tok/s GPU parity), but throughput is not inter-token LATENCY: no TPOT/ITL percentile is committed in BENCHMARK-AUTHORITY.md or CLAIMS.md. The committed fak-fronts-SGLang harness (experiments/qwen36/dgx-r4-20260622/compare.json) is NON-STREAMING -- it captures total request latency (E2EL, now committed; see end-to-end-request-latency) but does not timestamp individual output tokens, so it cannot isolate the per-token (TPOT) component of the gateway tax.

### ▼ End-to-end request latency (E2EL) — fak: **trails**

*Why it matters:* E2EL is the full submit-to-final-token time and is what a downstream agent or pipeline actually budgets against. It bundles queueing, prefill, all decode steps, batching effects, and network, so it is the only metric that captures the compound cost of long generations and is the right unit for multi-step agent SLA math.

- **SOTA bar:** End-to-end request latency (TTFT + total decode) is reported per request by the standard tools (NVIDIA GenAI-Perf, vLLM bench serve, LLMPerf) with P99 as the default percentile; there is no single SOTA number - it is a tunable composite governed by the per-model MLPerf v6.0 TTFT+TPOT SLOs (e.g. a 1000-token DeepSeek-R1 interactive response budgets ~1.5 s TTFT + ~15 s decode).
- **Leading systems:** NVIDIA GenAI-Perf, vLLM bench serve (--metric-percentiles, default p99), LLMPerf
- **Source:** [https://docs.vllm.ai/en/latest/benchmarking/cli/](https://docs.vllm.ai/en/latest/benchmarking/cli/) (2026-01)
- **fak:** trails — 9026.9 ms E2EL P50 @ conc 64 (shipped)
- **fak note:** WITNESSED gateway tax, the honest framing. Same 8-GPU datacenter server / Qwen3.6-27B / load harness, fak-gateway-in-front vs raw SGLang (the gateway is the only delta). At peak conc 64 fak's E2EL P50 is 9026.9 ms vs raw 7222.0 ms -> a +1804.9 ms (0.80x) end-to-end gateway tax. This is the FULL fak-in-front delta (proxy relay overhead PLUS the queueing induced by fak's lower peak throughput, 0.75x; see peak-serving-throughput-per-gpu), NOT just the in-process adjudication primitive (Decide 362 ns / AdmitChain <=87 us idle-box) which is only the small floor under it. fak fronts the backend and does NOT claim to win the raw-latency race; its value is the adjudication/coherence/measurement plane.
- **Trace:** 044220f8 · experiments/qwen36/dgx-r4-20260622/compare.json · docs/benchmarks/QWEN36-27B-GPU-SERVER-RESULTS.md

### ▼ Tail latency distribution (P50/P95/P99/P99.9) — fak: **trails**

*Why it matters:* Averages hide the requests that ruin trust. A degrading P99 while P50 holds steady is the early-warning signature of queue buildup or KV-memory pressure. At scale a 1-in-100 bad request is a continuous problem (100/hr at 10k req/hr), so operators size capacity and write SLOs against tail percentiles, not means.

- **SOTA bar:** P99 (99th percentile) is the canonical tail SLO across MLPerf Inference v6.0 (all TTFT/TPOT constraints are '99th percentile <= X'), NVIDIA GenAI-Perf/NIM, and vLLM bench (default --metric-percentiles p99). P99 remains the industry tail metric; some leaderboards also surface P95/P50.
- **Leading systems:** MLPerf Inference v6.0 (99th-percentile TTFT/TPOT constraints), vLLM bench (p99 default), NVIDIA GenAI-Perf / NIM
- **Source:** [https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/](https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/) (2026-03-24)
- **fak:** trails — 9997.6 ms request-latency P99 @ conc 64 (shipped)
- **fak note:** WITNESSED full request-latency tail on the same harness, fak-gateway-in-front vs raw SGLang. At peak conc 64 the committed distribution is fak P50/P95/P99 = 9026.9 / 9979.9 / 9997.6 ms vs raw 7222.0 / 8134.7 / 8157.9 ms; the P99 gateway tax is +1839.7 ms (0.82x). The tail is NOT widened by the gateway: fak's P99/P50 spread is 1.11x vs raw 1.13x, so fak shifts the WHOLE distribution by the ~+1.8 s tax rather than fattening the tail. Honest framing: fak fronts the backend and adds a measured tax, it does not claim to win the raw-latency race. (Sub-ms in-process primitive percentiles -- Decide 362 ns / AdmitChain <=87 us idle-box, BENCHMARK-AUTHORITY.md mac-m3pro-kernel-20260620 -- are the floor, not this under-load serving distribution.)
- **Trace:** 044220f8 · experiments/qwen36/dgx-r4-20260622/compare.json · docs/benchmarks/QWEN36-27B-GPU-SERVER-RESULTS.md

### ○ Prefill-decode interference & chunked-prefill TTFT/TPOT tension — fak: **no-claim**

*Why it matters:* When a long-context prefill shares a GPU batch with short interactive requests, it starves decode steps and spikes ITL/TPOT for everyone in the batch - and conversely, protecting decode can balloon TTFT. How a scheduler resolves this tension (chunked prefill, stall-free batching) is a primary differentiator of mixed-workload latency stability, which naive single-shape benchmarks never reveal.

- **SOTA bar:** Chunked prefill cuts P95 inter-token latency by ~68% under mixed long/short workloads (e.g. 2,800 ms -> 890 ms P95 ITL at 32K inputs) by prioritizing decode and chunking prefill into the token budget.
- **Leading systems:** vLLM chunked prefill (decode-prioritized scheduling), SGLang chunked prefill
- **Source:** [https://www.spheron.network/blog/llm-serving-optimization-continuous-batching-paged-attention/](https://www.spheron.network/blog/llm-serving-optimization-continuous-batching-paged-attention/) (2026-06)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a reuse kernel. Chunked prefill (Sarathi-Serve: ~68% P95 ITL improvement at 32K inputs) is a SCHEDULER technique inside a serving engine to remove decode-side ITL spikes; fak's own decode path is serial and small-model, and on the GPU server it fronts SGLang/llama.cpp which own this scheduling. fak makes no claim and shows no number on prefill-decode interference.
- **Trace:** No chunked-prefill mechanism is shipped or claimed in CLAIMS.md / BENCHMARK-AUTHORITY.md. fak's own engine prefills a prefix once and clones it (NewBatchFromPrefix) but does not interleave long prefills with decode steps to smooth ITL.

## Single-stream (one chat) (`single-stream`)

### ▼ Single-stream (one chat) decode throughput on CPU — fak: **trails**

*Why it matters:* The regime local/edge stacks (llama.cpp, Ollama, MLC) compete hardest on: one user, one chat, raw tokens/s on commodity hardware. fak explicitly does NOT target this â€” showing the loss keeps the scorecard honest.

- **SOTA bar:** llama.cpp Metal/CPU leads single-stream decode on the same box (e.g. Qwen2.5-7B Q8 ~17.3 tok/s Metal, M3 Pro).
- **Leading systems:** llama.cpp, Ollama, MLC-LLM
- **Source:** [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (2026-06)
- **fak:** trails — 8.7 tok/s (shipped)
- **fak note:** fak explicitly does NOT target single-stream raw throughput — one chat → use llama.cpp. The gap narrows with model size (0.39× → 0.53×, 1.5B → 7B) toward the shared ~150 GB/s unified-memory bandwidth floor; arm64 still lacks the register-blocked int8 GEMM tile that already exists on x86.
- **Trace:** 34c74f4 · model-ladder/modelbench-qwen25-7b-q8.json · QWEN25-7B-RESULTS.md

### ▼ Single-stream prefill throughput, apples-to-apples CPU-vs-CPU — fak: **trails**

*Why it matters:* Prefill (prompt ingestion) speed on CPU is a core local-inference axis. fak's arm64 build lacks the register-blocked int8 GEMM tile that x86 already has â€” a disclosed, narrowing loss.

- **SOTA bar:** llama.cpp CPU sets the prefill bar on the same box + Q8 weights (1Ã— reference); fak arm64 is ~0.12Ã— until the int8 tile lands.
- **Leading systems:** llama.cpp
- **Source:** [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (2026-06)
- **fak:** trails — 0.12 × of llama.cpp CPU (shipped)
- **fak note:** Apples-to-apples CPU-vs-CPU is ~0.12× on arm64 (which lacks the register-blocked int8 GEMM tile); the x86 Zen5 per-token compute SLOPE already reaches parity (1.03×), the ceiling arm64 inherits once the tile lands. The often-cited 0.083× vs a Metal GPU is NOT apples-to-apples — ~12× of that gap is the GPU device, so it collapses toward ~0.12× once the device is removed.
- **Trace:** 34c74f4 · MODEL-LADDER-VS-SOTA-2026-06-21.md · QWEN25-7B-RESULTS.md

### ≈ Single-stream decode throughput on a consumer GPU — fak: **parity**

*Why it matters:* Local GPU single-stream decode (the RTX-class laptop/desktop regime) is where llama.cpp's CUDA backend sets the bar; fak reaches parity here at higher precision via a reusable CUDA graph.

- **SOTA bar:** llama.cpp Q8_0 ~120 Â± 15 tok/s on an RTX 4070 (-ngl 99).
- **Leading systems:** llama.cpp (CUDA)
- **Source:** [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (2026-06)
- **fak:** parity — 120 tok/s (shipped)
- **fak note:** Same GPU (RTX 4070, WSL2), but fak runs f32 — 4× the weight bytes of llama.cpp's 8-bit Q8_0 — so parity is reached at HIGHER precision and ~46% GPU utilization (a disclosed fak advantage, not a hidden tilt). Gated FAK_CUDA_GRAPH=1; small model that fits VRAM (a 7B f32 set won't fit WSL's ~15 GB, so large-model parity is not claimed).
- **Trace:** 1029e37 · GPU.md §3b · LLAMACPP-HEADTOHEAD-RESULTS.md

