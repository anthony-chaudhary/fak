# Production workload parameter ledger

**As of:** 2026-08-26. **Tracker:** #9301. This table records measured populations and
parameters instead of collapsing every serving workload into “Poisson arrivals, Zipf
popularity, fixed token lengths.” The source-level claims and limitations remain in
[`index.json`](index.json).

## Trace populations

| Trace | Population and window | Scale | Directly observed fields | Boundary |
|---|---|---:|---|---|
| ServeGen / Alibaba Cloud Model Studio | Worldwide production cloud service, four months | 3.54B requests, 12 model groups, O(10K) GPUs across dozens of regions/zones | arrival time, client, model group, input/output lengths, conversation/multimodal/reasoning structure | Provider-wide managed-service mix; serving internals and exact customer identity are hidden. |
| GitHub Copilot coding agent | Anonymized Visual Studio/VS Code telemetry, first week of June 2026 | 3.2M users, 13.5M sessions, 95.1M turns, 760.5M LLM calls, 774.7M tool calls, 44.9T prompt tokens, 39.3B completion tokens | timestamps, durations, tokens excluding hidden reasoning, model/tool, tool success/failure, turns and sessions | One product and sampled week; workload evolved January–June and will continue to drift. |
| TraceLab | Claude Code and Codex use by 43 developers over about eight months | ~4.3K sessions, ~350K LLM steps, ~430K tool calls, >20 model versions | normalized conversation/tool logs, context growth/reduction, latency, cacheable prefixes | Cross-provider but self-selected developer population, not provider-wide random traffic. |
| FineServe | Global commercial model marketplace | Multi-model production requests; 100K-request task sample, 20K silver labels | architecture/scale, arrivals, input/output tokens, task intent, longitudinal model availability | Marketplace traffic differs from first-party apps; some raw counts/parameters are withheld. |
| Aliyun KVCache “in the wild” | Consumer and developer-API production services | Two real workload families | prefix reuse, reuse probability/time, request category, single-turn and multi-turn reuse | One cloud provider; business-sensitive population and all raw distributions are not public. |
| Chutes one-year trace | Open multi-model production platform over one year | Full longitudinal model/user trace promised with paper | model popularity, long tail, user-model affinity, cache/load-balancing implications | Open marketplace/platform, not closed frontier first-party products. |
| BurstGPT v1.1 | Production trace used for capacity work | 5.29M raw requests over 121 days; 5.19M completed | timestamps, input/output tokens, concurrency, failures | Earlier language-model trace; later multimodal/reasoning/agent workloads differ materially. |

## Measured parameters that change system design

### GitHub Copilot coding agents

| Parameter | Measured value | System consequence |
|---|---:|---|
| Agent-initiated LLM calls | 87% | Schedule and account at session/workflow scope, not only independent requests. |
| LLM-to-tool relationship | Approximately 1:1 overall | GPU execution alternates with CPU/I/O tool phases; orchestrator latency and reliability affect compute. |
| Prefix-cached token share within sessions | 90% average | Cache reads and residency dominate more than fresh-prefill-only models imply. |
| Cache share at turn boundaries | 55% | A user turn is a major cache-lifecycle boundary. |
| Cache share after a model switch | 8% | Model routing can destroy locality and must price cold restart. |
| Sessions with context compaction | 7.8% | Compaction is uncommon by session count but operationally important. |
| Total tokens in compacted sessions | 44% | Rare heavy sessions dominate token work; count-weighted and token-weighted views disagree. |
| Prompt tokens dropped by compaction | >70% | Compaction resets state and changes the effective request geometry. |
| Turns with tool failure | 9% | Tool reliability is a serving-efficiency variable. |
| Retry-loop amplification | Up to 4× compute | Failures create growing contexts and repeated inference rather than a fixed retry cost. |
| User archetypes | 5 | Uniform resource policy is not population-representative. |
| Per-turn token range across archetypes | 23K–1.1M (~50×) | User/tenant heterogeneity must be preserved in admission, quotas, and benchmarks. |
| Mean reclaimable idle at turn boundary | 4.1 min container; 2.9 min KV cache | Turn boundaries support asymmetric resource reclamation/offload. |

### TraceLab coding agents

| Parameter | Measured value | System consequence |
|---|---:|---|
| LLM calls per user request | 8.8 average | One task is a multi-step chain, not one inference request. |
| Tool calls per user request | 10.8 average | Tool infrastructure is at least as frequent as model invocation. |
| Task completion time | 4.3 min average; P90 >6.4 min | Cache and container policies operate over minutes and long tails. |
| Context shape | Usually append-growing, with compaction and Codex micro-reduction exceptions | Append-only is useful but not a correctness-complete assumption. |
| Tool-call latency | Diverse and heavily tailed | Idle-window predictions and offload need tool semantics, not one timeout. |
| Prefix cache | High but imperfect; cached-token reads dominate API cost in the paper’s price snapshot | Prefix-cache hit ratio alone misses the cost of repeatedly reading long history. |

### General cloud serving (ServeGen and FineServe)

| Dimension | Measured behavior | Benchmark implication |
|---|---|---|
| Arrival process | Bursts and distribution shifts are often driven by a few top clients; most individual clients are more stable | Compose per-client processes; do not fit only the aggregate. |
| Dense vs MoE arrivals | Dense models show higher-frequency jitter; MoE shows lower-frequency, high-amplitude bursts | Provision and scale by architecture/workload class. |
| Input lengths | FineServe sees extended heavy tails and fits log-normal inputs rather than BurstGPT’s Zipf assumption | Use fitted log-normal/mixture inputs when that population applies; Zipf is not universal. |
| Dense input→output relation | Often inverted-bowl: output peaks around 1.5K–2K input tokens, then falls >60% for long inputs | Do not assume output grows linearly with input. |
| MoE input→output relation | Monotonic growth then saturation at long context | Condition output geometry on architecture and task. |
| Reasoning lengths | Long and bimodal in ServeGen | Decode capacity and batching need explicit reasoning-mode mixture. |
| Multimodal demand | Load varies substantially by modality and request content | Include fetch, normalize, encode, media count/size, and prefill interference. |
| Task classification | FineServe uses 10 intent classes, up to two labels/request | Workload generators need task mixture, not only model identity. |

### Output-length uncertainty

A 2026 study generated 100 responses for each of 1,000 LMSYS prompts under fixed model and
decoding settings. It reported average skewness **3.10**, mean coefficient of variation
**1.09**, CV >1 for **78.6%** of prompts, top-decile share **35.7%** of generated length,
P90/P50 **4.62**, and P99/P50 **10.77**. Its fitted log-t result is experimental rather
than a production arrival trace, but it falsifies deterministic per-prompt output length.

## Distribution-selection rules

1. **Poisson:** use only as a stationary control or after demonstrating inter-arrival fit for
   the exact client/window. Aggregate production arrivals are often bursty and nonstationary.
2. **Zipf:** use only with a named object (model, tenant, prefix, tool, geography), fitted
   exponent, goodness-of-fit, window, and drift. Heavy tail alone does not establish Zipf.
3. **Log-normal:** supported for FineServe input lengths within its marketplace population;
   do not transfer its parameters blindly to first-party frontier APIs.
4. **Log-t/heavy-tail output:** useful for uncertainty-aware scheduling sensitivity tests;
   preserve model, prompt, decoding, and sampling conditions.
5. **Bimodal/mixture:** appropriate when modes correspond to real classes such as reasoning
   length, cache-affinity state, modalities, or turn boundaries. Name the latent class.
6. **Per-client composition:** preferred when a few clients create aggregate bursts or drift.
7. **Longitudinal drift:** model releases, availability, pricing, routing, and user learning can
   move distributions; a one-week trace is not a permanent parameter file.

## Minimum benchmark record

Every generated or replayed workload should report:

- source trace and date, sampling/anonymization, population, and excluded fields;
- clients/tenants/users, models, modalities, tasks, regions, and channels;
- arrival model, parameters, time unit, burst/seasonal/regime-change treatment;
- input/output/reasoning distributions and their dependence, truncation, and quantiles;
- session/turn/tool structure, idle gaps, failures/retries, model switches, and compaction;
- prefix/cache reuse definition, hit distribution, residency, transfer/recompute, and eviction;
- batching/admission/routing/autoscaling policy, hardware/topology, SLO, and quality;
- both request-weighted and token/work-weighted results.

## Remaining parameter debt

The public record still lacks stable provider-wide parameters for tenant concentration,
geography/diurnal seasonality, first-party frontier prompt popularity, exact cache reuse by
product, reasoning-mode selection, speculative acceptance, retries outside coding agents,
and installed-to-goodput conversion. Those must remain benchmark sweeps or explicitly
synthetic priors rather than asserted production facts.
