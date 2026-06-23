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

- **SOTA bar:** Qualitative: TTFT decomposes into queueing delay + prefill compute; at P50 queueing dominates and request latency grows sharply once RPS pushes the queue to build. The diagnostic is a load sweep showing the rate at which P99 TTFT crosses the SLO; P99 climbing while P50 is flat is the saturation signature.
- **Leading systems:** vLLM benchmark_serving (request-rate sweep), NVIDIA GenAI-Perf (concurrency sweep)
- **Source:** [https://tianpan.co/blog/2026-03-10-llm-latency-decomposition-ttft-vs-throughput](https://tianpan.co/blog/2026-03-10-llm-latency-decomposition-ttft-vs-throughput) (2026-03)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP fak should measure. fak has the harness to vary concurrency (it ran 1->128 on the GPU server) but reports the throughput envelope, not the saturation diagnostic (P99 TTFT climbing while P50 stays flat as the queue builds). Since the gateway proxy adds queueing of its own, the TTFT-under-load curve is exactly what an honest operability claim needs; not yet measured, so no-claim.
- **Trace:** The GPU server concurrency sweep (conc 1->128, QWEN36-27B-DGX-RESULTS.md) varies LOAD but records completion-tokens/sec, not a TTFT/queueing curve: there is no committed P99-TTFT-vs-RPS saturation sweep showing the rate at which the SLO is crossed.

### ○ Cold-start / autoscaling latency (scale-to-zero & scale-out) — fak: **no-claim**

*Why it matters:* The first request after a scale-out or a scale-to-zero wake pays a model-load tax - weights fetched and loaded into GPU memory before any token streams - inflating TTFT by tens of seconds. For elastic/serverless deployments this dominates worst-case latency and decides whether scale-to-zero is even viable for interactive use, so cold-start TTFT and warm-up time are core operability dimensions buyers must score.

- **SOTA bar:** Naive serverless GPU cold starts are 30-120s (often >40s to first token). ServerlessLLM's multi-tier checkpoint loading cuts startup 6-8x; RunPod FlashBoot reports sub-250ms cold starts for most requests (≈half under 200ms), making scale-to-zero viable for real-time serving.
- **Leading systems:** RunPod FlashBoot, ServerlessLLM, HydraServe, PipeBoost
- **Source:** [https://regolo.ai/scale-to-zero-cold-start-latency-why-serverless-gpu-breaks-real-time-ai-and-how-to-fix-it/](https://regolo.ai/scale-to-zero-cold-start-latency-why-serverless-gpu-breaks-real-time-ai-and-how-to-fix-it/) (2025-01)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP, partially out of scope. Cold-start/scale-to-zero latency (RunPod FlashBoot sub-250ms; ServerlessLLM 6-8x faster load) is a serverless-platform metric for the weight-loading path fak's own engine does not yet exercise at scale (its largest faithfully-served model is ~7B, it fronts external engines above that). fak has a residency/page-out POLICY but no measured load-latency number; no-claim.
- **Trace:** No cold-start, scale-to-zero, or autoscaling latency is measured in CLAIMS.md / BENCHMARK-AUTHORITY.md. fak has a model-residency/budget policy plane (internal/residency, internal/cachemeta) but it is off-mainline (FAK_POLYMODEL-gated) and moves no weight bytes - no checkpoint-load or startup-to-first-token timing is committed.

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

- **SOTA bar:** Guided/constrained decoding (XGrammar, LLGuidance) in vLLM and SGLang raises JSON-schema conformance from ~90-94% (free generation) to >96%, up to 98.2%, behind a drop-in OpenAI-compatible /v1/chat/completions surface with guided_json/regex/grammar/choice. OpenAI-compat is the de facto integration contract; LiteLLM unifies 100+ providers behind the same format.
- **Leading systems:** vLLM (XGrammar/guided decoding), SGLang, LiteLLM (gateway)
- **Source:** [https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang](https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang) (2025-01-01)
- **fak:** parity — no number (shipped)
- **fak note:** Parity on the INTEGRATION CONTRACT, not on guided-decoding conformance. fak ships the de facto OpenAI-compatible surface (/v1/chat/completions adjudication proxy, /v1/models, MCP over stdio/HTTP, base_url-swappable client) so a non-Go agent routes through the same boundary, and a grammar rung does in-syscall positional->named tool-call repair. But the STRONGEST structured-output form, decode-time logit-mask / grammar-constrained generation (XGrammar-style never-emit-malformed, the >96-98.2% JSON-conformance lever vLLM/SGLang cite) is explicitly [STUB] because it requires owning the decode loop. apples_to_apples=false: no committed JSON-schema conformance % to compare, so this is a capability-presence parity on the API surface, a gap on guaranteed schema decoding.
- **Trace:** CLAIMS.md Gateway [SHIPPED] (/v1/chat/completions, /v1/models, /healthz, MCP); Engine [SHIPPED] OpenAI-compatible client base_url-swappable; Pre-flight grammar rung [SHIPPED] positional->named repair; decode-time logit-mask [STUB]

### ▼ Production observability: Prometheus metrics (TTFT/TPOT/queue) + OpenTelemetry tracing — fak: **trails**

*Why it matters:* You cannot meet an SLO you cannot see. Per-request TTFT/TPOT histograms and KV/cache-hit telemetry are what drive autoscaling, capacity planning, and incident response; their absence is disqualifying for a production buyer.

- **SOTA bar:** vLLM v1 exposes a Prometheus /metrics endpoint with per-request TTFT (vllm:time_to_first_token_seconds), TPOT/inter-token latency, queue depth, running/waiting requests, KV-cache usage and prefix-cache hit-rate, plus OpenTelemetry distributed traces for per-request spans. This metric set (TTFT, TPOT, goodput, cache-hit) is the de facto observability contract serving systems are judged against.
- **Leading systems:** vLLM, SGLang, NVIDIA Dynamo
- **Source:** [https://docs.vllm.ai/en/latest/design/metrics/](https://docs.vllm.ai/en/latest/design/metrics/) (2025-01-01)
- **fak:** trails — no number (shipped)
- **fak note:** fak exposes ADJUDICATION observability (a /metrics endpoint backing the exit AdjudicationSummary operation counters, an end-to-end TraceID the IFC ledger / plan-CFI key on) but NOT the serving-systems observability contract it is judged against: there is no committed per-request TTFT (vllm:time_to_first_token_seconds), TPOT/inter-token, queue depth, KV-cache usage or prefix-cache-hit-rate Prometheus series, and the metrics-service scrape adapter is [SIMULATED] telemetry. OpenTelemetry distributed traces per the vLLM v1 spec are not shipped. fak's tracing is adjudication-plane, the de facto serving metric-set is the gap. Honest trails.
- **Trace:** CLAIMS.md Gateway [SHIPPED] /metrics + Server.AdjudicationSummary + TraceID threaded end-to-end; Engine [SIMULATED] metrics-service scrape adapter (unit 43); no committed TTFT/TPOT/queue-depth/cache-hit Prometheus series

### ○ Kubernetes-native deployment + LLM-aware autoscaling (metric-driven, scale-to-zero) — fak: **no-claim**

*Why it matters:* Fleet economics live or die on elasticity: scaling on queue depth/running-requests (not CPU) and scaling idle models to zero is how operators avoid paying for idle GPUs. K8s-native lifecycle (canary, rollout, multi-model) is the deployment substrate buyers require.

- **SOTA bar:** KServe + KEDA autoscale vLLM on LLM-specific signals (e.g. vllm:num_requests_running) rather than CPU, supporting scale-from/to-zero with a default 5-min cooldown; the 2025-2026 production stack is vLLM + KServe/llm-d + Ray + Kueue + KEDA on K8s. Ray Serve and KServe provide canary, traffic-split, and multi-model endpoints. The SOTA bar is custom-LLM-metric autoscaling with scale-to-zero, not generic HPA.
- **Leading systems:** KServe + KEDA, Ray Serve, llm-d / vLLM on Kubernetes
- **Source:** [https://developers.redhat.com/articles/2025/09/23/how-set-kserve-autoscaling-vllm-keda](https://developers.redhat.com/articles/2025/09/23/how-set-kserve-autoscaling-vllm-keda) (2025-09-23)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a single-process reuse kernel. fak is one statically-linked Go binary (fak serve / fak guard) with no Kubernetes operator, no custom-LLM-metric HPA (vllm:num_requests_running), no KServe/KEDA/Ray integration and no scale-to-zero. The 2025-2026 vLLM+KServe/llm-d+Ray+Kueue+KEDA stack has no fak counterpart and fak does not claim one. Honest no-claim; would only become a gap if fak grew into a multi-replica serving platform.
- **Trace:** No KServe/KEDA/Ray/K8s deployment claim in CLAIMS.md or BENCHMARK-AUTHORITY.md; fak ships one statically-linked Go binary + fak guard / fak serve

### ○ Multi-tenant fairness, per-tenant quotas, and SLO attainment under contention — fak: **no-claim**

*Why it matters:* A shared cluster serving many agents/tenants must stop a noisy neighbor from starving everyone else and must hold each tenant's latency SLO under load. Fairness, quotas, and SLO-attainment-under-contention are exactly the properties a multi-tenant platform buyer evaluates.

- **SOTA bar:** Virtual Token Counter (VTC) is the first work-conserving fair scheduler for LLM serving, tracking cumulative per-client input/output tokens to prevent any tenant monopolizing the batch; follow-ups (QLM, Prism, Equinox, FairBatching, PROSERVE) optimize multi-priority SLO attainment. At the gateway layer LiteLLM enforces per-key/project token budgets and spend caps. The SOTA bar is provable fairness + per-tenant SLO under contention, not raw throughput.
- **Leading systems:** VTC (Sheng et al.), LiteLLM (virtual keys/budgets), Equinox / FairBatching
- **Source:** [https://arxiv.org/abs/2401.00588](https://arxiv.org/abs/2401.00588) (2024-01-01)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP for the multi-agent regime fak targets. fak has per-agent KV ownership and a lease-disjointness steward (isolation primitives) and an agent-scoped tainted Ref per wire client, but NO work-conserving fair scheduler (VTC-style cumulative per-client token tracking), no per-tenant token budgets / spend caps (the LiteLLM virtual-key axis), and no measured SLO-attainment-under-contention number. Since fak's product is fleets of agents, provable fairness is a gap it arguably SHOULD measure rather than out-of-scope. No fak number exists; honest no-claim.
- **Trace:** No VTC-style fair-scheduler or per-tenant quota/budget claim in CLAIMS.md; lease-disjointness steward + per-agent KV ownership are isolation, not provable batch fairness

