---
title: "fak industry scorecard — operability"
description: "The operability dimensions that matter in LLM serving, the current SOTA bar on each, and fak's honest position. Generated from tools/industry_scorecard.data/."
---

# operability — the dimensions that matter, and where fak stands

[← back to the scorecard index](README.md) · part of the industry-first scorecard. Each dimension is a thing the field competes on; the fak column is honest — mostly `no-claim` gaps for a focused reuse kernel.

## Operability (`operability`)

### ○ Fairness, priority, and tenant isolation in batch formation / scheduling — fak: **no-claim**

*Why it matters:* A multi-tenant or mixed-SLA deployment must prevent one heavy or long-prefill tenant from starving others, and must honor priority classes (real-time vs best-effort). FCFS plus cache-affinity heuristics can create head-of-line blocking and load hot-spots; fairness-aware batch formation and priority preemption are the controls an operator needs to keep per-tenant SLOs while sharing a fleet.

- **SOTA bar:** No single audited cross-system number; SOTA is qualitative: FairBatching reformulates batch formation to bound per-request unfairness, and serving stacks add priority classes plus preemption (KV-page eviction with recompute-on-resume in SGLang) to isolate tenants. Buyers evaluate head-of-line-blocking behavior and priority-preemption support directly.
- **Leading systems:** FairBatching (fairness-aware batch formation), SGLang (FCFS + page-eviction preemption), hybrid real-time/best-effort schedulers (e.g. arXiv 2504.09590)
- **Source:** [https://arxiv.org/pdf/2510.14392](https://arxiv.org/pdf/2510.14392) (2025-10)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE as a measured serving-fairness number, though adjacent to fak's trust plane: fak provides per-agent KV isolation and targeted causal eviction (witnessed bit-exact), but it has no FairBatching-style per-request unfairness bound, no priority classes with preemption, and no measured head-of-line-blocking behavior in batch formation. The SOTA bar here is itself qualitative; fak still has no number, so no claim.
- **Trace:** No fairness / priority-class / tenant-isolation batch-formation number in BENCHMARK-AUTHORITY.md or CLAIMS.md. fak ships per-agent KV ownership and causal cache invalidation (BENCHMARK-AUTHORITY.md causal-invalidation witness, max|delta|=0) but no measured head-of-line-blocking or priority-preemption behavior.

### ○ Latency degradation under concurrency / load (saturation behavior) — fak: **no-claim**

*Why it matters:* TTFT and tail latency are flat at low load and explode past a knee point as the request queue grows. The location of that knee, and how steeply latency climbs past it, determine real capacity. Testing at concurrency=1 is meaningless; operators must characterize TTFT/P99 across a concurrency sweep up to and beyond their P95 concurrent-request count.

- **SOTA bar:** Under concurrency, single-GPU TTFT P99 explodes (e.g. ~8.6 s at C=180, ~34 s at C=420); pipeline parallelism (PP=2) holds it to ~3 s at C=180 and ~13 s at C=420 (~2.5-3x improvement). At 100 concurrent requests TensorRT-LLM P95 TTFT ~1,280 ms vs vLLM ~1,450 ms; vLLM sustains QoS to C=32 and stays usable to C=64.
- **Leading systems:** TensorRT-LLM (lowest P95 TTFT under load), SGLang, vLLM (most robust queueing/scheduling)
- **Source:** [https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/) (2026-01)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP fak should measure. fak has the harness to vary concurrency (it ran 1->128 on the GPU server) but reports the throughput envelope, not the saturation diagnostic (P99 TTFT climbing while P50 stays flat as the queue builds). Since the gateway proxy adds queueing of its own, the TTFT-under-load curve is exactly what an honest operability claim needs; not yet measured, so no-claim.
- **Trace:** The GPU server concurrency sweep (conc 1->128, QWEN36-27B-GPU-SERVER-RESULTS.md) varies LOAD but records completion-tokens/sec, not a TTFT/queueing curve: there is no committed P99-TTFT-vs-RPS saturation sweep showing the rate at which the SLO is crossed.

### ○ Cold-start / autoscaling latency (scale-to-zero & scale-out) — fak: **no-claim**

*Why it matters:* The first request after a scale-out or a scale-to-zero wake pays a model-load tax - weights fetched and loaded into GPU memory before any token streams - inflating TTFT by tens of seconds. For elastic/serverless deployments this dominates worst-case latency and decides whether scale-to-zero is even viable for interactive use, so cold-start TTFT and warm-up time are core operability dimensions buyers must score.

- **SOTA bar:** GPU-snapshotting cold start is now sub-200 ms: RunPod FlashBoot achieves sub-200 ms for ~48% of requests (<250 ms generally) and Modal GPU memory snapshots cut Ministral-3 3B median cold start ~10x (118 s -> 12 s) - the corroborable bar. InferX additionally claims an industry-leading 177 ms on H100 via GPU snapshotting, but that single number traces to one vendor source and is not independently confirmable, so the defensible figure is sub-200 ms.
- **Leading systems:** RunPod FlashBoot (sub-200 ms / <250 ms), Modal GPU memory snapshots (10x: 118 s -> 12 s), InferX (177 ms vendor claim, GPU snapshot, uncorroborated)
- **Source:** [https://www.runpod.io/articles/guides/flashboot-instant-cold-starts](https://www.runpod.io/articles/guides/flashboot-instant-cold-starts) (2026)
- **fak:** no-claim — no number (in-flight)
- **fak note:** PARTIAL SURFACE SHIPPED; no comparable scale-to-zero claim. fak now measures boot-to-ready and named startup phases, and its optional warmup gate keeps /healthz unready until a representative first token so the operator's first real request does not absorb a known cold backend warmup. The committed GLM-5.2 observation (~500 s cold warmup versus ~1 s warm turns) is a real cold-start diagnostic, but it is not an apples-to-apples serverless snapshot / scale-out benchmark and fak still ships no autoscaler or scale-to-zero lifecycle. The full axis therefore remains no-claim rather than converting the non-comparable observation into a manufactured trail.
- **Trace:** internal/gateway/startup.go emits fak_gateway_time_to_ready_seconds plus per-phase startup duration; internal/gateway/readiness_warmup.go can hold readiness until a representative warmup turn completes; docs/notes/GLM52-COLD-START-VS-CACHING-ABLATION-2026-07-06.md records the observed ~500 s backend warmup separately from KV reuse. No comparable scale-to-zero / scale-out latency result is committed.

### ○ Cross-instance / persistent KV cache sharing and coherence — fak: **no-claim**

*Why it matters:* In production, the same KV should be reusable across engine instances, restarts, and prefill/decode roles, not trapped in one process's HBM. A shared, persistent, deduplicated KV store with peer-to-peer access raises fleet hit rate and survives autoscaling churn. Operators evaluate whether caching is per-process-ephemeral or a first-class shared substrate with consistent addressing.

- **SOTA bar:** LMCache enables KV reuse across different vLLM engine instances plus multi-GPU peer-to-peer KV sharing and disaggregated prefill/decode; Mooncake Store exposes a global, deduplicated KVCache pool addressable across the cluster
- **Leading systems:** LMCache, Mooncake Store
- **Source:** [https://arxiv.org/html/2510.09665v2](https://arxiv.org/html/2510.09665v2) (2025-10)
- **fak:** no-claim — no number (in-flight)
- **fak note:** REAL GAP, with adjacent mechanism shipped. fak has durable cross-process session core images and a coherence bus that broadcasts cache refutations with byte-exact causal eviction, but NOT a global deduplicated KV pool shared across live engine instances (LMCache cross-instance reuse / Mooncake Store). The pieces that point toward it (recall, the coherence bus, the not-yet-witness-keyed tier-2) are real; the cross-instance live-sharing claim is unbuilt, so no parity is asserted.
- **Trace:** CLAIMS lines 66-71: recall persists a finished session as a durable, integrity-checked CORE IMAGE reloadable in a FRESH process; BENCHMARK-AUTHORITY causal-invalidation row proves a coherence-bus broadcast + targeted causal eviction (max|Δ|=0). But this is cross-PROCESS session reload + invalidation coherence, not a live cross-INSTANCE shared KV POOL like LMCache/Mooncake Store.

### ○ Multi-node fault tolerance & failure recovery (resilient serving) — fak: **no-claim**

*Why it matters:* At 96+ GPU scale a single GPU/node failure is a when, not an if, and tightly-coupled TP/EP means one failed rank can halt the whole instance. Operators evaluating large disaggregated/EP deployments need to know mean-time-to-recovery, whether the system degrades gracefully (reroute traffic, replicate KV, take only a failed replica offline) versus full restart. Resilience is a fast-emerging differentiator as deployments cross the single-node boundary.

- **SOTA bar:** Research systems define the bar: KevlarFlow reports ~20x lower mean-time-to-recovery via decoupled parallel init, dynamic traffic rerouting, and background KV-cache replication; FailSafe sustains high-performance tensor-parallel serving under irregular GPU availability; ReviveMoE targets fast recovery from hardware failures specifically in large-scale MoE inference. Production stacks (Dynamo) add dynamic GPU scheduling and rerouting but full resilient serving is still maturing.
- **Leading systems:** FailSafe (research), KevlarFlow (research), ReviveMoE (research), NVIDIA Dynamo
- **Source:** [https://arxiv.org/html/2601.22438v1](https://arxiv.org/html/2601.22438v1) (2026-01-01)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a single-box reuse kernel. fak has no multi-node serving topology, hence no node-failure rerouting, no background KV replication, no decoupled parallel re-init. Its nearest adjacent witness — causal invalidation-on-external-write (vdso.Revoke, commit 0fc39aa, max|Delta|=0) — is an INTEGRITY/coherence property (evict cache reads a write made stale), explicitly 'a containment/coherence witness, not a throughput number', NOT failure recovery. The research bar (KevlarFlow/FailSafe/ReviveMoE) is multi-GPU resilient serving fak does not attempt. No fak number; verdict no-claim.
- **Trace:** none — no multi-node failure-recovery number in CLAIMS; causal-invalidation (commit 0fc39aa) is an integrity/coherence witness, not failure recovery

### ≈ OpenAI-compatible API surface + reliable structured output / tool-calling — fak: **parity**

*Why it matters:* Agents depend on machine-parseable tool calls; a malformed JSON object breaks the loop. OpenAI-compatibility plus near-guaranteed schema conformance is what lets an operator swap engines/models without rewriting clients, a primary procurement criterion.

- **SOTA bar:** Constrained/guided decoding now guarantees 100% schema-valid structured output by construction (XGrammar-2 is the default engine across vLLM/SGLang/TensorRT-LLM, <40 us/token), with the OpenAI-compatible /v1/chat/completions + response_format json_schema surface as the de facto contract.
- **Leading systems:** vLLM + XGrammar-2 (default), SGLang, TensorRT-LLM, OpenAI Structured Outputs (response_format)
- **Source:** [https://arxiv.org/pdf/2601.04426](https://arxiv.org/pdf/2601.04426) (2026-01)
- **fak:** parity — no number (shipped)
- **fak note:** Parity on CAPABILITY PRESENCE, not a JSON-schema conformance or guided-decoding performance number. fak ships the de facto OpenAI-compatible Chat Completions and Responses surfaces, forwards structured-output carriers to ridden OpenAI-compatible engines, and now owns an opt-in in-kernel decode-time constraint hook: logit_bias and an injected per-step LogitMask apply before argmax, with a deterministic load-bearing schema-valid JSON witness. The remaining gap versus XGrammar-2 is breadth and measurement: no committed arbitrary-schema conformance percentage, grammar-compile coverage, or per-token overhead comparison. apples_to_apples=false keeps this a bounded API + mechanism parity claim, not an unqualified guided-decoding parity claim.
- **Trace:** CLAIMS.md Gateway [SHIPPED] (/v1/chat/completions, /v1/responses, /v1/models, /healthz, MCP); Engine [SHIPPED] OpenAI-compatible client base_url-swappable; internal/agent forwards response_format / logit_bias carriers; internal/model constraint_test.go witnesses the opt-in FAK_NATIVE_GUIDED_DECODE per-step LogitMask before argmax

### ▼ Production observability: Prometheus metrics (TTFT/TPOT/queue) + OpenTelemetry tracing — fak: **trails**

*Why it matters:* You cannot meet an SLO you cannot see. Per-request TTFT/TPOT histograms and KV/cache-hit telemetry are what drive autoscaling, capacity planning, and incident response; their absence is disqualifying for a production buyer.

- **SOTA bar:** vLLM v1 exposes an extensive Prometheus /metrics surface (vllm:time_to_first_token_seconds, TPOT/inter-token, vllm:num_requests_running, vllm:kv_cache_usage_perc, prefix-cache hit) plus native OpenTelemetry distributed tracing via --otlp-traces-endpoint -- the de facto serving-observability contract.
- **Leading systems:** vLLM v1 (Prometheus + OpenTelemetry), SGLang, NVIDIA Dynamo / GenAI-Perf
- **Source:** [https://docs.vllm.ai/en/latest/design/metrics/](https://docs.vllm.ai/en/latest/design/metrics/) (2026-01)
- **fak:** trails — no number (shipped)
- **fak note:** METRIC-SET PARITY; DISTRIBUTED-TRACING GAP. The native surface emits fak_gateway_inference_ttft_seconds, fak_gateway_inference_tpot_seconds, and fak_gateway_inference_e2e_seconds; fak_gateway_inflight_requests, fak_gateway_inflight_requests_by_route, and fak_gateway_inflight_max_age_seconds; fak_gateway_kv_memory_resident_bytes, fak_gateway_kv_memory_capacity_bytes, and fak_gateway_kv_memory_headroom_ratio; fak_gateway_kv_prefix_reuse_ratio; and fak_gateway_inference_cached_prompt_hit_ratio. The normalized serving schema emits fak_serving_time_to_first_token_seconds, fak_serving_time_per_output_token_seconds, fak_serving_inter_token_latency_seconds, fak_serving_num_requests_running, fak_serving_num_requests_waiting, fak_serving_kv_cache_usage_perc, and fak_serving_prefix_cache_hit_rate for native or ridden workers. The separate [SIMULATED] metrics-service scrape adapter is not needed as evidence for this native surface and is intentionally omitted from the grounds. What is still absent is W3C trace-context propagation (traceparent/tracestate) and OTLP/OpenTelemetry span export: X-Trace-Id supplies fak-local correlation, not standards-based distributed tracing (#5629). Because this axis combines the serving metric set with OpenTelemetry, the composite honestly trails even though the named Prometheus metric set is at parity.
- **Trace:** CLAIMS.md [SHIPPED] Serving-latency observability; internal/gateway/metrics_render.go emits fak_gateway_inference_ttft_seconds, fak_gateway_inference_tpot_seconds, fak_gateway_inference_e2e_seconds, inflight / KV-memory / prefix-reuse families, and provider cache-hit ratio; internal/gateway/serving_metrics.go emits the normalized fak_serving_* serving families. W3C trace-context propagation and OTLP span export are not shipped (#5629).

### ○ Kubernetes-native deployment + LLM-aware autoscaling (metric-driven, scale-to-zero) — fak: **no-claim**

*Why it matters:* Fleet economics live or die on elasticity: scaling on queue depth/running-requests (not CPU) and scaling idle models to zero is how operators avoid paying for idle GPUs. K8s-native lifecycle (canary, rollout, multi-model) is the deployment substrate buyers require.

- **SOTA bar:** KServe + KEDA autoscale vLLM on LLM-specific signals (e.g. vllm:num_requests_running) rather than CPU, supporting scale-from/to-zero with a default 5-min cooldown; the 2025-2026 production stack is vLLM + KServe/llm-d + Ray + Kueue + KEDA on K8s. Ray Serve and KServe provide canary, traffic-split, and multi-model endpoints. The SOTA bar is custom-LLM-metric autoscaling with scale-to-zero, not generic HPA.
- **Leading systems:** KServe + KEDA, Ray Serve, llm-d / vLLM on Kubernetes
- **Source:** [https://developers.redhat.com/articles/2025/09/23/how-set-kserve-autoscaling-vllm-keda](https://developers.redhat.com/articles/2025/09/23/how-set-kserve-autoscaling-vllm-keda) (2025-09-23)
- **fak:** no-claim — no number (in-flight)
- **fak note:** KUBERNETES DEPLOYMENT SHIPPED; LLM-AWARE AUTOSCALING UNBUILT. fak now has copy-paste base and GPU-overlay manifests with liveness/readiness probes, hardened pod settings, kustomize entry points, and a documented rollout/smoke path. That removes the old 'no K8s deployment' claim. The scored contract is broader, however: fak still has no KServe/KEDA/Ray operator, no custom-metric HPA driven by running/waiting requests, no traffic-split/canary controller, and no scale-to-zero lifecycle. apples_to_apples=false and no-claim preserve that residual rather than treating deployability alone as autoscaling parity.
- **Trace:** deploy/k8s/fak.yaml + kustomization.yaml ship an apply-ready hardened proxy Deployment/Service/Secret/ConfigMap; deploy/k8s/overlays/gpu ships the in-kernel one-GPU/PVC sibling; docs/fak/deployment-guide.md is the operator contract. No fak-owned KEDA/HPA/operator or scale-to-zero controller is shipped.

### ○ Multi-tenant fairness, per-tenant quotas, and SLO attainment under contention — fak: **no-claim**

*Why it matters:* A shared cluster serving many agents/tenants must stop a noisy neighbor from starving everyone else and must hold each tenant's latency SLO under load. Fairness, quotas, and SLO-attainment-under-contention are exactly the properties a multi-tenant platform buyer evaluates.

- **SOTA bar:** VTC (the 2024 first fair scheduler) is now superseded by a 2025 family: Equinox (end-to-end fair scheduling), FairBatching (fairness-aware batch formation), PROSERVE (multi-priority SLO-aware), and DLPM (locality-aware fair scheduling) -- all addressing VTC's gaps on prefix-locality, SLO, and diverse workloads.
- **Leading systems:** Equinox (2025), FairBatching (2025), PROSERVE (2025), DLPM locality-aware fair scheduling (2025), VTC (2024, prior bar), LiteLLM virtual-keys/budgets
- **Source:** [https://arxiv.org/pdf/2508.16646](https://arxiv.org/pdf/2508.16646) (2025-08)
- **fak:** no-claim — no number (in-flight)
- **fak note:** TENANT QUOTA/BUDGET HALF SHIPPED; FAIR-SCHEDULING/SLO HALF UNBUILT. Authenticated org/project principals now receive isolated token-rate allotments, so one noisy tenant cannot consume another's allotment, and control-plane spend caps can bound the aggregate token/USD spend of every session under a tenant. That removes the old 'no per-tenant quotas or spend caps' claim. fak still has no work-conserving VTC/FairBatching-style scheduler, no priority-preemption policy keyed by tenant, and no measured per-tenant SLO attainment under contention. The full fairness axis therefore remains no-claim, with apples_to_apples=false to separate shipped governance quotas from serving-scheduler fairness.
- **Trace:** internal/gateway/token_admission_principal.go gives each authenticated principal an isolated rolling-window token allotment; internal/gateway/spend_governor.go + fak serve --spend-cap enforce tenant/team/agent/session token or USD caps across the served boundary. No VTC-style fair scheduler or per-tenant SLO-under-contention benchmark is committed.
