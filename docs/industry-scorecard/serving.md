---
title: "fak industry scorecard — serving"
description: "The serving dimensions that matter in LLM serving, the current SOTA bar on each, and fak's honest position. Generated from tools/industry_scorecard.data/."
---

# serving — the dimensions that matter, and where fak stands

[← back to the scorecard index](README.md) · part of the industry-first scorecard. Each dimension is a thing the field competes on; the fak column is honest — mostly `no-claim` gaps for a focused reuse kernel.

## Throughput & batching (`throughput`)

### ○ Continuous / in-flight (iteration-level) batching: max aggregate token throughput at high concurrency — fak: **no-claim**

*Why it matters:* Iteration-level batching admits and retires requests mid-flight so a long sequence never blocks short ones, and a fixed GPU serves far more simultaneous sequences. It is the single largest lever on tokens/s per GPU and on serving cost, so it is the first thing any operator benchmarks.

- **SOTA bar:** vLLM reports up to 24x higher throughput than HuggingFace TGI and Orca at the same latency by combining PagedAttention with continuous batching; field-typical gain over static batching is 2-4x at high concurrency with mixed output lengths.
- **Leading systems:** vLLM (PagedAttention + continuous batching), Orca (iteration-level scheduling, originator), TensorRT-LLM (in-flight batching)
- **Source:** [https://arxiv.org/pdf/2309.06180](https://arxiv.org/pdf/2309.06180) (2023-09)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP fak should measure but currently OUT OF SCOPE for the kernel as shipped: CLAIMS labels continuous/in-flight batching [SIMULATED] read-only telemetry, not on the live serving path, and the polymodel decode lane is explicitly SERIAL (at-most-one-model-decodes-per-step, off-mainline). Iteration-level batching is the defining vLLM/SGLang throughput lever; fak has NO shipped number. fak's fleet win is prefix-reuse work-ELIMINATION, a different axis from concurrent-decode batching, so no parity is implied.
- **Trace:** CLAIMS.md L163 ([SIMULATED] continuous-batching read-only telemetry, not on live serving path); CLAIMS.md L6 (SIMULATED tag definition); existing data.json row 'continuous-batching'.

### ○ Absolute aggregate tokens/s at scale on standardized hardware (MLPerf Inference) — fak: **no-claim**

*Why it matters:* Cross-vendor, audited absolute throughput on a fixed model and SLO is what a buyer uses to compare full systems and compute cost-per-token. MLPerf Inference is the only neutral, reproducible apples-to-apples leaderboard with both offline (max throughput) and server (latency-constrained) scenarios, so it anchors the absolute SOTA bar that engine micro-benchmarks cannot.

- **SOTA bar:** MLPerf Inference v5.0: GB200 NVL72 hit ~13,886 tokens/s offline on Llama 3.1 405B (and ~30x the 8-Hopper server throughput); v5.1: Blackwell Ultra hit 5,842 tokens/s/GPU offline and 2,907 tokens/s/GPU server on the relevant LLM benchmark. Server-scenario SLOs: 405B at 6s TTFT / 175ms TPOT; new interactive scenario at 0.5s TTFT / 30ms TPOT.
- **Leading systems:** NVIDIA GB200 NVL72 (Blackwell), NVIDIA HGX B200 / Blackwell Ultra, MLPerf Inference v5.0 / v5.1 submitters
- **Source:** [https://mlcommons.org/2025/04/llm-inference-v5/](https://mlcommons.org/2025/04/llm-inference-v5/) (2025-04)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a reuse kernel: MLPerf Inference ranks absolute aggregate tok/s on standardized Blackwell/Hopper hardware (GB200 NVL72 ~13,886 tok/s offline on Llama 3.1 405B). fak has no MLPerf submission and its own engine ceilings at ~7B (data.json model-size-ceiling), so it is not in this race; on big models it FRONTS SGLang/llama.cpp rather than serving them itself. No standardized fak tok/s exists.
- **Trace:** No MLPerf submission or standardized-hardware aggregate tok/s in BENCHMARK-AUTHORITY.md. The only concurrent server number is the DGX A100 run (data.json 'served-throughput-vs-sglang', 1085.6 tok/s @ conc 64) which is fak-gateway-fronting-SGLang on non-MLPerf hardware/harness, not a standardized MLPerf result.

### ○ Latency-vs-throughput Pareto frontier — fak: **no-claim**

*Why it matters:* There is no single operating point - every system trades batch size against per-request latency. The buyer's real question is the shape of the whole frontier: how much throughput can I buy before TTFT/TPOT breaches my SLO, and how gracefully does the curve bend. A system that advances the frontier serves more users per GPU at the SAME latency guarantee.

- **SOTA bar:** Qualitative/relative: frontier-advancing techniques (Sarathi-Serve chunked prefill, PD-disaggregation) push higher throughput at fixed latency SLO; the frontier is reported as a sweep of request rate vs P99 TTFT/TPOT rather than a single point. PD-colocation maximizes throughput but breaches latency SLO; disaggregation holds SLO but under-utilizes GPUs.
- **Leading systems:** Sarathi-Serve (chunked prefill), DynaServe, vLLM/SGLang sweeps
- **Source:** [https://arxiv.org/pdf/2403.02310](https://arxiv.org/pdf/2403.02310) (2024-03)
- **fak:** no-claim — no number (stub)
- **fak note:** fak has not recorded a latency-vs-throughput Pareto frontier (no P99 TTFT/TPOT vs request-rate). The adjacent committed evidence is a throughput-vs-concurrency sweep where fak trails raw SGLang (see peak-serving-throughput-per-gpu). Named as a gap, not a Pareto measurement.
- **Trace:** experiments/qwen36/dgx-r4-20260622/compare.json + docs/benchmarks/QWEN36-27B-DGX-RESULTS.md: fak-gateway vs raw SGLang on 8xA100 / Qwen3.6-27B, concurrency sweep 1->128; fak/raw = 0.60x (conc 8) -> 0.75x (conc 64 peak, 1085.6 vs 1451.6 tok/s) -> 0.97x (conc 128).

### ▼ Peak served throughput per accelerator (model-served-by-engine, MLPerf-grounded) — fak: **trails**

*Why it matters:* Throughput per GPU is the denominator of cost-per-token and the headline capacity number every operator sizes a fleet against. MLPerf is the only neutral, audited cross-vendor source, so it anchors the SOTA bar buyers actually trust over vendor blogs.

- **SOTA bar:** MLPerf Inference v5.1 / v5.0: ~1.1M tokens/s aggregate on one Azure ND GB300 v6 (GB300 NVL72) rack (unverified submission, beating 865K tok/s on GB200 NVL72); ~33,000 tok/s on a single H200 for Llama-2-70B (40% over H100). These are the buyer-citable upper bounds for engine+hardware throughput.
- **Leading systems:** NVIDIA GB300/GB200 NVL72 (TensorRT-LLM), NVIDIA H200, MangoBoost/AMD MI300X
- **Source:** [https://www.hpcwire.com/2025/09/10/mlperf-inference-v5-1-results-land-with-new-benchmarks-and-record-participation/](https://www.hpcwire.com/2025/09/10/mlperf-inference-v5-1-results-land-with-new-benchmarks-and-record-participation/) (2025-09-10)
- **fak:** trails — 1085.6 tok/s @ conc 64 (peak) (shipped)
- **fak note:** THE honest counter to the fleet 'lead' rows: this is the ONLY real concurrent-serving head-to-head against a LIVE SGLang process (same 8×A100 / 27B / load harness, fak-in-front vs raw). fak TRAILS — it is the gateway/adjudication TAX: 0.60× worst (conc 8) → 0.75× at peak (conc 64) → ~0.97× converged at saturation (conc 128). fak's value here is the adjudication/coherence/measurement plane, NOT raw tok/s; the 4.1× fleet 'lead' is reuse-WORK eliminated on fak's own kernel held constant, a different axis from a throughput race against a tuned engine.
- **Trace:** experiments/qwen36/dgx-r4-20260622/compare.json · docs/benchmarks/QWEN36-27B-DGX-RESULTS.md

## Scheduling & routing (`scheduling`)

### ○ Chunked / piggyback prefill and prefill-decode interference (stall-free batching) — fak: **no-claim**

*Why it matters:* When a long prefill shares an iteration with ongoing decodes it stalls every decode, spiking inter-token latency (TPOT/TBT). Chunked prefill splits a prefill into token-bounded chunks and piggybacks them onto decode batches so decodes never stall, decoupling the throughput/latency tradeoff. It is the core mechanism that lets a co-located engine hit a tight TPOT SLO at high load.

- **SOTA bar:** Sarathi-Serve delivers up to 2.6x higher serving capacity (Mistral-7B, 1xA100), 3.7x (Yi-34B, 2xA100), and up to 5.6x (Falcon-180B with pipeline parallelism) under a TBT SLO vs vLLM, by capping per-iteration prefill work to a chunk size (typically 256-512 tokens).
- **Leading systems:** Sarathi-Serve (chunked-prefill + stall-free scheduling), vLLM (chunked prefill), SGLang (chunked prefill)
- **Source:** [https://arxiv.org/abs/2403.02310](https://arxiv.org/abs/2403.02310) (2024-03)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for fak as a reuse kernel today, and an unmeasured GAP if pursued: Sarathi-Serve-style chunked/piggyback prefill caps per-iteration prefill work to bound TBT under decode interference. fak owns its own forward but has not built or measured chunked-prefill scheduling; no number exists, so no claim.
- **Trace:** No chunked-prefill / TBT-SLO / prefill-decode-interference number in BENCHMARK-AUTHORITY.md or CLAIMS.md. fak's prefill rows (single-stream-prefill, radix-live-vs-ceiling) are single-stream or reuse, not stall-free chunked scheduling under load.

### ≈ Request scheduling and cache-aware routing (FCFS vs priority vs prefix/KV-cache-aware) — fak: **parity**

*Why it matters:* The scheduling discipline decides which requests batch together and where they route. A naive FCFS engine ignores that requests sharing a prefix (system prompts, RAG context, multi-turn chat, agents) can reuse KV cache; a cache-aware scheduler co-locates and orders them to maximize reuse. This directly multiplies effective throughput on shared-context workloads and trades off against load balance.

- **SOTA bar:** SGLang v0.4's cache-aware load balancer delivers up to 1.9x throughput and 3.8x prefix-cache hit-rate improvement; SGLang's prefix-sharing batch heuristic gives ~29% higher throughput than vLLM when requests share context. The core tension is cache affinity vs load balance.
- **Leading systems:** SGLang (RadixAttention + cache-aware load balancer), sglang-router, llm-d Endpoint Picker (prefix-cache-aware GAIE routing)
- **Source:** [https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/) (2024-12)
- **fak:** parity — 86.7 % (shipped)
- **fak note:** fak measured on-box: cache-aware (≡ DFS) scheduling recovers FCFS 62.1% → 86.7% = 100% of optimal, inside SGLang's published band. Split-reuse proven bit-identical to recompute (max|Δ|=0), so the hit is a real reuse, not a numerics shortcut. FENCE: 86.7% is on radixbench's synthetic 'agents' workload (deliberately cache-favorable); on a REAL tau2-airline trace fak's measured addressable purity is ~0.7% (CLAIMS vDSO unit 33/83) — so this is a same-workload-as-SGLang reuse-mechanism parity, not a promise of 86.7% on every production corpus.
- **Trace:** 92896a4 · experiments/radixattention/radixbench-qwen2.5-1.5b-q8-agents-fresh-20260619.json · data.json row 'kv-hit-rate'; BENCHMARK-AUTHORITY.md L23 + 'RadixAttention Model Ladder' (FCFS 62.1% -> cache-aware 86.7% = 100% of optimal); artifact radixbench-qwen2.5-1.5b-q8-agents-fresh-20260619.json (verify workloads[0].cache_hit_rate=0.8666666666666667). Commit 92896a4.

## Latency & SLO (`latency`)

### ○ Interactive latency SLO attainment: TTFT and TPOT/ITL tail under load — fak: **no-claim**

*Why it matters:* For chatbots, coding assistants, and agents the contract is felt latency: time-to-first-token and steady per-token cadence at the tail (p99), not the mean. A serving system is differentiated by how high it can push concurrency before TTFT or TPOT breach the SLO. This is the latency face of the same tradeoff goodput measures on the throughput side.

- **SOTA bar:** MLPerf v5.1 interactive scenario defines the current strict bar at 0.5s p99 TTFT and 30ms p99 TPOT (~1,600 words/min) for chatbot/coding/creative uses; the v5.0 405B server scenario uses 6s TTFT / 175ms TPOT. Systems are ranked by max throughput sustainable under these constraints.
- **Leading systems:** MLPerf Inference v5.1 interactive scenario, TensorRT-LLM, Sarathi-Serve / vLLM chunked prefill
- **Source:** [https://www.marktechpost.com/2025/10/01/mlperf-inference-v5-1-2025-results-explained-for-gpus-cpus-and-ai-accelerators/](https://www.marktechpost.com/2025/10/01/mlperf-inference-v5-1-2025-results-explained-for-gpus-cpus-and-ai-accelerators/) (2025-10)
- **fak:** no-claim — no number (stub)
- **fak note:** fak ships no ms-level P99 TTFT/TPOT tail-latency number; the only live head-to-head is a THROUGHPUT sweep (it trails raw SGLang 0.75× at peak — see peak-serving-throughput-per-gpu). An honest gap on the latency-tail axis, not a measured trail.
- **Trace:** data.json row 'served-throughput-vs-sglang'; artifact experiments/qwen36/dgx-r4-20260622/compare.json; docs/benchmarks/QWEN36-27B-DGX-RESULTS.md s2. fak 1085.6 vs raw SGLang 1451.6 tok/s @ conc 64 on 8xA100, Qwen3.6-27B.

### ○ Time-to-first-token (TTFT) under realistic prefill — fak: **no-claim**

*Why it matters:* TTFT is the perceived responsiveness of an interactive system - the wait before any output appears. It is dominated by queueing delay plus prefill compute, so it is the headline latency metric for chat, voice, and copilot UX and the gating constraint in every serving SLO. Buyers compare TTFT at a fixed input length and concurrency because a single-user TTFT (just network + one forward pass) tells you almost nothing about production behavior.

- **SOTA bar:** MLPerf v5.1 interactive scenario fixes a 4.5s P99 TTFT bound for Llama3.1-405B (vs 6s P99 in the server scenario); production interactive serving targets are commonly ~200-450ms TTFT. TTFT is measured at fixed ISL and concurrency, excluding empty first responses.
- **Leading systems:** MLPerf Inference v5.1 (Llama3.1-405B interactive), vLLM benchmark_serving, NVIDIA GenAI-Perf
- **Source:** [https://mlcommons.org/2025/09/small-llm-inference-5-1/](https://mlcommons.org/2025/09/small-llm-inference-5-1/) (2025-09)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP fak should measure. TTFT is adjacent to fak's actual lever (prefix-reuse cuts prefill tokens, which IS the dominant TTFT term on a cache hit), yet fak has never reported TTFT as a percentile under realistic prefill/concurrency the way MLPerf v5.1 (4.5s P99 TTFT, 405B interactive) or vLLM/GenAI-Perf do. The honest answer is no-claim until fak instruments the gateway proxy for a per-request TTFT distribution.
- **Trace:** No TTFT row exists in BENCHMARK-AUTHORITY.md or CLAIMS.md. fak commits prefill-WORK ratios (single-stream prefill 0.58x vs llama.cpp CPU; RadixAttention live prefill 4.58x-6.95x vs full re-prefill) but never a TTFT measured as a latency percentile at fixed ISL/concurrency.

### ○ Inter-token latency / time-per-output-token (ITL/TPOT) — fak: **no-claim**

*Why it matters:* After the first token, ITL/TPOT governs how fast text streams - the felt 'typing speed.' It must clear a per-user threshold (e.g. faster than human reading) and stay smooth; stalls produce visible jank even at good averages. TPOT excludes the first token by construction (denominator subtracts 1) so it isolates the decode phase from prefill.

- **SOTA bar:** MLPerf v5.1 interactive requires 12.5 tokens/s/user (≈80ms TPOT) for Llama3.1-405B at 4.5s TTFT; server scenario allows 175ms P99 TPOT. Small-model interactive targets 40ms TPOT. ~200ms TPOT maps to ~240 wpm human reading speed.
- **Leading systems:** MLPerf Inference v5.1, vLLM benchmark_serving, NVIDIA GenAI-Perf
- **Source:** [https://mlcommons.org/2025/04/llm-inference-v5/](https://mlcommons.org/2025/04/llm-inference-v5/) (2025-04)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP fak should measure. fak has decode-rate numbers (which trail llama.cpp single-stream, as it openly states it does not target single-stream) but never the per-output-token latency distribution MLPerf v5.1 encodes (12.5 tok/s/user ~= 80ms TPOT for 405B interactive). Reciprocal of a tok/s is not an ITL percentile under load; no-claim until measured.
- **Trace:** fak commits single-stream DECODE throughput (8.7 tok/s 7B Q8 CPU, ~0.55-0.73x of llama.cpp at HEAD 374776a; ~120 tok/s GPU parity), but throughput is not inter-token LATENCY: no TPOT/ITL percentile is committed in BENCHMARK-AUTHORITY.md or CLAIMS.md.

### ○ End-to-end request latency (E2EL) — fak: **no-claim**

*Why it matters:* E2EL is the full submit-to-final-token time and is what a downstream agent or pipeline actually budgets against. It bundles queueing, prefill, all decode steps, batching effects, and network, so it is the only metric that captures the compound cost of long generations and is the right unit for multi-step agent SLA math.

- **SOTA bar:** Qualitative: E2EL ≈ TTFT + (output_len-1)×TPOT plus scheduling/network overhead; serious harnesses report it as a percentile (P50/P99) at a fixed output length, using a sliding window that discards warm-up/cool-down requests for a stable measurement.
- **Leading systems:** NVIDIA GenAI-Perf, vLLM benchmark_serving (e2el percentile)
- **Source:** [https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) (2025-01)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP fak should measure. E2EL = TTFT + (out_len-1)xTPOT + overhead, reported as a sliding-window percentile by GenAI-Perf / vLLM. fak's gateway sits exactly on the request path (and adds a measured tax) so this is directly instrumentable, but no E2EL percentile has been committed; no-claim.
- **Trace:** No E2EL percentile is committed anywhere. fak reports whole-run wall-clocks for fleet/session benchmarks (e.g. 19.0 min for a 50-turn x 5-agent run) and a concurrency throughput sweep on the DGX, but never a per-request end-to-end latency reported as a P50/P99 over a fixed output length with a warm-up/cool-down window.

### ○ Tail latency distribution (P50/P95/P99/P99.9) — fak: **no-claim**

*Why it matters:* Averages hide the requests that ruin trust. A degrading P99 while P50 holds steady is the early-warning signature of queue buildup or KV-memory pressure. At scale a 1-in-100 bad request is a continuous problem (100/hr at 10k req/hr), so operators size capacity and write SLOs against tail percentiles, not means.

- **SOTA bar:** MLPerf encodes its latency SLOs at the 99th percentile (e.g. 6s TTFT / 175ms TPOT P99 for 405B server). Industry guidance treats a P99 that is many multiples of P50 (e.g. P50 400ms vs P99 4s) as a queueing/memory-pressure red flag requiring capacity or scheduler intervention.
- **Leading systems:** MLPerf Inference (P99-based constraints), vLLM benchmark_serving, production SLO practice
- **Source:** [https://www.spheron.network/blog/llm-inference-slo-ttft-itl-latency-budget-guide-2026/](https://www.spheron.network/blog/llm-inference-slo-ttft-itl-latency-budget-guide-2026/) (2026-01)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP fak should measure. The only percentile fak commits is a p50 of the in-process adjudication primitive on an idle box (worst-payload AdmitChain 87 us) - useful for bounding the adjudication tax but NOT a request tail-latency distribution. MLPerf encodes its SLOs at P99 and production practice flags P99>>P50 as a saturation red flag; fak has no P99 serving number, so no-claim.
- **Trace:** fak reports median (p50) pure-kernel decision latencies in isolation (Decide 362 ns, Admit 3.3-15.8 us, AdmitChain 29-87 us; BENCHMARK-AUTHORITY.md mac-m3pro-kernel-20260620) but NO serving-path latency DISTRIBUTION: no P95/P99/P99.9 of request latency under load is committed.

### ○ Prefill-decode interference & chunked-prefill TTFT/TPOT tension — fak: **no-claim**

*Why it matters:* When a long-context prefill shares a GPU batch with short interactive requests, it starves decode steps and spikes ITL/TPOT for everyone in the batch - and conversely, protecting decode can balloon TTFT. How a scheduler resolves this tension (chunked prefill, stall-free batching) is a primary differentiator of mixed-workload latency stability, which naive single-shape benchmarks never reveal.

- **SOTA bar:** Chunked prefill interleaves long prefills with decode to remove ITL spikes; reported P95 improvement of ~68% at 32K-token inputs (2,800ms -> 890ms) while costing only a slight TTFT increase and ~5-20% throughput change depending on workload.
- **Leading systems:** Sarathi-Serve, vLLM chunked prefill, SGLang
- **Source:** [https://www.spheron.network/blog/llm-serving-optimization-continuous-batching-paged-attention/](https://www.spheron.network/blog/llm-serving-optimization-continuous-batching-paged-attention/) (2026-01)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a reuse kernel. Chunked prefill (Sarathi-Serve: ~68% P95 ITL improvement at 32K inputs) is a SCHEDULER technique inside a serving engine to remove decode-side ITL spikes; fak's own decode path is serial and small-model, and on the GPU server it fronts SGLang/llama.cpp which own this scheduling. fak makes no claim and shows no number on prefill-decode interference.
- **Trace:** No chunked-prefill mechanism is shipped or claimed in CLAIMS.md / BENCHMARK-AUTHORITY.md. fak's own engine prefills a prefix once and clones it (NewBatchFromPrefix) but does not interleave long prefills with decode steps to smooth ITL.

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

*Why it matters:* Prefill (prompt ingestion) speed on CPU is a core local-inference axis. fak's arm64 build lacks the register-blocked int8 GEMM tile that x86 already has — a disclosed, narrowing loss.

- **SOTA bar:** llama.cpp CPU sets the prefill bar on the same box + Q8 weights (1× reference); fak arm64 is ~0.12× until the int8 tile lands.
- **Leading systems:** llama.cpp
- **Source:** [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (2026-06)
- **fak:** trails — 0.12 × of llama.cpp CPU (shipped)
- **fak note:** Apples-to-apples CPU-vs-CPU is ~0.12× on arm64 (which lacks the register-blocked int8 GEMM tile); the x86 Zen5 per-token compute SLOPE already reaches parity (1.03×), the ceiling arm64 inherits once the tile lands. The often-cited 0.083× vs a Metal GPU is NOT apples-to-apples — ~12× of that gap is the GPU device, so it collapses toward ~0.12× once the device is removed.
- **Trace:** 34c74f4 · MODEL-LADDER-VS-SOTA-2026-06-21.md · QWEN25-7B-RESULTS.md

### ≈ Single-stream decode throughput on a consumer GPU — fak: **parity**

*Why it matters:* Local GPU single-stream decode (the RTX-class laptop/desktop regime) is where llama.cpp's CUDA backend sets the bar; fak reaches parity here at higher precision via a reusable CUDA graph.

- **SOTA bar:** llama.cpp Q8_0 ~120 ± 15 tok/s on an RTX 4070 (-ngl 99).
- **Leading systems:** llama.cpp (CUDA)
- **Source:** [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (2026-06)
- **fak:** parity — 120 tok/s (shipped)
- **fak note:** Same GPU (RTX 4070, WSL2), but fak runs f32 — 4× the weight bytes of llama.cpp's 8-bit Q8_0 — so parity is reached at HIGHER precision and ~46% GPU utilization (a disclosed fak advantage, not a hidden tilt). Gated FAK_CUDA_GRAPH=1; small model that fits VRAM (a 7B f32 set won't fit WSL's ~15 GB, so large-model parity is not claimed).
- **Trace:** 1029e37 · GPU.md §3b · LLAMACPP-HEADTOHEAD-RESULTS.md

