# Production workload parameter ledger

**As of:** 2026-08-27. **Tracker:** #9301. This table records measured populations and
parameters instead of collapsing every serving workload into “Poisson arrivals, Zipf
popularity, fixed token lengths.” The source-level claims and limitations remain in
[`index.json`](index.json).

## Chinese platform and recommendation envelopes (#9362)

| Source | Typed population / envelope | Bounded implication |
|---|---|---|
| iFLYTEK filing | >8.7M AI developer teams; >3.42M production apps; 1.52M large-model developers; API daily-call growth 4.3x YTD; agents +85% YTD | Five different ecosystem denominators, not users, requests, tokens, or one traffic distribution. |
| LongCat-Flash | 560B total / 18.6B–31.3B activated parameters; >20T pretraining tokens; 4,096-H800 pipeline; batch-32 H800 throughput about 100k tok/s | Bind capacity and throughput to activated parameters, accelerator, batch, and source-estimated cost. |
| DORA | Thousands of production accelerators; up to 6.2x rollout speedup for an approximately 500B MoE | Trajectory tails, policy versions, and KV migration invalidate one synchronous-batch model. |
| MTServe | Unique user/item state; request- and prefix-level caches; up to 4.93x throughput and 4.19x lower latency | Stateful recommendation locality is not ordinary shared-prefix reuse and does not prove Zipf. |
| MTGenRec | 200M real user sequences over one week; 128 A100s | Sequences are training examples, not unique users, sessions, requests, or live QPS. |

## Speculative decoding and acceptance parameters (#9366)

| Mechanism | Bounded result | Planning boundary |
|---|---|---|
| Foundational speculative decoding | T5-XXL roughly 2x-3x with output distribution preserved | Drafter/target agreement and verification cost determine waste. |
| SpecInfer | 1.5x-2.8x distributed; 2.6x-3.5x offloading; tree 1.2x-1.5x vs sequence | Different bottlenecks; ranges are not stackable. |
| Medusa | >2.2x frozen-backbone Medusa-1; 2.3x-3.6x jointly trained Medusa-2 | Different training and quality contracts. |
| EAGLE | Roughly 1.5x-2.8x wall-clock speedup across named LLaMA/Vicuna evaluations | Acceptance is conditional on task, model, temperature, draft position, and load. |
| MagicDec | Batch 32-256, eight A100s; up to 2x/1.84x for named long-context models | Large batches help only in the measured KV-dominated regime. |

## Non-coding agent workload envelopes (#9367)

| Workload | Bounded population | Systems implication |
|---|---|---|
| WebArena | 812 tasks, 5 sites, 4 application domains | Stateful browser actions and end-state evaluators; not production sessions. |
| WorkArena | 33 task types, 19,912 instances; cleaned HTML 40k-500k tokens | Observation representation can dominate context and varies independently of dialogue. |
| tau-bench | Retail and airline multi-turn API interaction; pass^8 reliability | Track user turns, policy checks, reads/writes, and repeated-task consistency. |
| GAIA | 466 questions; 355 web, 154 coding, 138 multimodal, 129 diverse-file tags | Tags overlap; route by capability composition rather than summed counts. |
| OSWorld | 369 real-computer tasks across 3 operating systems | Budget screenshots/state, GUI/CLI actions, files, apps, evaluators, and resets. |

## Public conversation populations (#9370)

| Dataset | Bounded population | Denominator boundary |
|---|---|---|
| LMSYS-Chat-1M | 1M conversations, 25 models, April-August 2023 | Public Vicuna/Arena opt-in traffic, not provider-wide production. |
| WildChat | 1M conversations, ~2.5M turns, >68k anonymized users | IP-derived user/country fields are not verified people or universal regional weights. |
| Chatbot Arena | >240k votes, >90k users, >50 models | Votes/users/model appearances are different and self-selected. |
| OpenAssistant | 161,443 messages, 35 languages, 66,497 trees, 461,292 ratings, >13,500 volunteers | Branching crowdsourced alignment data, not an arrival stream. |

## Trace populations

| Trace | Population and window | Scale | Directly observed fields | Boundary |
|---|---|---:|---|---|
| ServeGen / Alibaba Cloud Model Studio | Worldwide production cloud service, four months | 3.54B requests, 12 model groups, O(10K) GPUs across dozens of regions/zones | arrival time, client, model group, input/output lengths, conversation/multimodal/reasoning structure | Provider-wide managed-service mix; serving internals and exact customer identity are hidden. |
| GitHub Copilot coding agent | Anonymized Visual Studio/VS Code telemetry, first week of June 2026 | 3.2M users, 13.5M sessions, 95.1M turns, 760.5M LLM calls, 774.7M tool calls, 44.9T prompt tokens, 39.3B completion tokens | timestamps, durations, tokens excluding hidden reasoning, model/tool, tool success/failure, turns and sessions | One product and sampled week; workload evolved January–June and will continue to drift. |
| TraceLab | Claude Code and Codex use by 43 developers over about eight months | ~4.3K sessions, ~350K LLM steps, ~430K tool calls, >20 model versions | normalized conversation/tool logs, context growth/reduction, latency, cacheable prefixes | Cross-provider but self-selected developer population, not provider-wide random traffic. |
| FineServe | Global commercial model marketplace | Multi-model production requests; 100K-request task sample, 20K silver labels | architecture/scale, arrivals, input/output tokens, task intent, longitudinal model availability | Marketplace traffic differs from first-party apps; some raw counts/parameters are withheld. |
| Aliyun KVCache “in the wild” | Consumer and developer-API production services | Two real workload families | prefix reuse, reuse probability/time, request category, single-turn and multi-turn reuse | One cloud provider; business-sensitive population and all raw distributions are not public. |
| Chutes one-year trace | Open multi-model production platform over one year | Full longitudinal model/user trace promised with paper | model popularity, long tail, user-model affinity, cache/load-balancing implications | Open marketplace/platform, not closed frontier first-party products. |
| Azure OpenAI / BurstGPT | Four Azure OpenAI-powered services over 213 days | 10.31M requests: GPT-4 8.69M; GPT-3.5 0.95M; ChatGPT 0.30M; GPT-4o 0.16M | service, arrivals, users for selected days, input/output tokens, periodicity, bursts, separated failures | Four traces are not all Azure traffic; day-4 users are not total users; burst examples are not quantiles. |
| Microsoft Azure / Splitwise | Two production services over one day: Conversation and Coding | A few thousand requests per service | empirical input/output token distributions and request rates | One day and small samples; no universal tenant/geography/session distribution or disclosed model weights. |
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

## Audited distribution parameters (#9381)

| Source | Production variables retained | Directly reported family/test/parameter | Benchmark treatment |
|---|---|---|---|
| BurstGPT / Azure OpenAI v1 | Request timestamps, aggregate arrivals, input tokens/request, output tokens/request | No family parameter or goodness-of-fit test in arXiv v1 (2024-01-31) | Replay each released 121-day and 36-day population separately; exclude later v1.1 synthetic Zipf material from this record |
| Chutes Year-in-Serving | Model request-count rank/share, selected-model interarrivals, input/output tokens per request | No Zipf exponent or fitted family/test | Preserve rank/share, lengths, and interarrivals as different axes; segment time because model rankings and load evolve |
| ServeGen v3 | One-minute interarrival time by workload tier; category input/output lengths; five-minute aggregate request rate | IAT candidates Exponential/Gamma/Weibull are KS-tested; best families are Gamma (M-large), Weibull (M-mid), and Exponential (M-small). Category inputs use a Pareto/lognormal mixture and outputs Exponential. Trend model periods are 24h and 12h. No token-mixture parameters or token-fit statistic published | Preserve variable and tier/category labels; replay the week or label generated traffic synthetic; do not turn the bounded fits into a universal law |
| Alibaba KV-cache trace | Prefix-sharing ratio/length, tree geometry, session size, prefix reuse frequency/lifetime, request lengths | No Zipf/Pareto/lognormal parameter or fit/test | Use empirical per-workload distributions and keep request/workload/prefix/session/block denominators explicit |

ServeGen v3 retains named fitted families and KS test evidence, but it publishes no reusable
shape/scale values for its IAT families and no weights or Pareto/lognormal parameters for its
input-length mixture. Its 24-hour and 12-hour values are aggregate seasonality periods, not
family parameters. No source in this audit reports a defensible universal Zipf exponent,
Pareto tail index, lognormal parameters, Poisson rate, Hawkes kernel, or MMPP transition matrix.

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

## Production-trace parameter addendum

This addendum records exact population and shape facts extracted from the primary
production papers. It does not turn reported observations into universal fitted laws.

| Trace | Exact population/window | Concentration, burst, drift, or cache parameter | Safe replay rule | Missing parameter |
|---|---|---|---|---|
| ServeGen | Four months; 3.54B requests; 12 model groups; dozens of regions; O(10K) GPUs; 2,412 profiled clients | A small set of 29 top clients drives much of the dynamic aggregate behavior. One multimodal workload's input length rose 13% on average while output length fell 18%, shifting prefill and decode load differently. | Draw client first, then model/workload class; vary request rate, input, and output independently over time. | Per-client weights, geography/timezone distribution, exact arrival fit, and public tenant identifiers. |
| FineServe | Four months; 1.48B requests; 57 models; 10 model families | Even the least burst-dominant architecture group has its top 5% of seconds carry about 9.5% of hourly traffic. Long-term shifts vary by architecture/scale, and new releases or availability changes can abruptly change arrival/token distributions. | Combine second-scale extreme bursts with slower release-driven drift; condition token shapes on task and model class. | Raw quantiles/fitted family per model/region, tenant concentration, and geographic/diurnal coefficients. |
| Chutes year trace | One year; 6.12B unsampled requests; 9,174 models; user and serving-instance IDs; cache logging for final two months | Dominant models turn over; user-model affinity changes; most models remain bursty; request-level cached fractions are strongly bimodal and differ by model/user. | Re-rank model popularity and user affinity over time; use model/user-conditioned bimodal cacheability rather than one hit rate. | Public rank-frequency exponent, region, customer tier, request-length raw quantiles, and policy-independent reuse opportunity. |
| Aliyun KVCache | Two production workloads | Single-turn requests account for 97% of reuses in one trace. Reuse probability is well modeled by exponential distributions after conditioning on request category; global behavior is heterogeneous. | Include repeated single-turn prefixes and fit category-specific reuse/lifetime parameters. | Published universal parameter table, raw tenant weights, and transfer/admission overhead by category. |
| GitHub Copilot coding agent | First week of June 2026; 3.2M users; 13.5M sessions; 95.1M turns; 760.5M LLM calls; 774.7M tool calls | Average session prefix-cached share 90%; only 7.8% of sessions compacted but those held 44% of tokens; 9% of turns had a tool failure; retries can amplify compute up to 4×; five archetypes span about 23K–1.1M tokens/turn. | Replay session structure, model switches, compaction, tool failure, and retries; sample archetype before token volume. | Tenant/company concentration, geography, weekday/hour curve, reasoning-mode choice, and speculative acceptance. |
| TraceLab coding agents | About eight months; 43 developers; ~4,300 sessions; ~350K LLM steps; ~430K tool calls | Mean 8.8 LLM calls and 10.8 tool calls per request; mean completion 4.3 minutes and P90 above 6.4 minutes; tool-call counts are heavy-tailed. | Preserve long-lived alternating tool/model loops and idle KV/container windows. | Population representativeness, per-developer shares, geography, exact tail fit, and production failure/retry rates. |
| Output-length uncertainty experiment | 1,000 LMSYS prompts × 100 generations each | Average skewness 3.10; mean CV 1.09; CV >1 for 78.6%; top decile 35.7% of generated length; P90/P50 4.62; P99/P50 10.77. | Use prompt-conditioned stochastic output length and reserve for tail token-time, not only request count. | Production sampling policy, user/tenant population, arrival process, and cross-model drift. |
| Azure OpenAI / BurstGPT | 213 days; 10.31M requests; four services | Daily and weekly periodicity; dominant period about one day; long-tailed tokens; case-study bursts of 400 req/s for 10 s, 100 req/s for 60 s, and 15 req/s for 600 s; GPT-4 failure fractions about 0.068 instance, 0.034 trigger/context exceed, and 0.005 content policy. | Preserve service mix, daily/weekly cycles, token tails, multiple burst durations, and separate failure classes. | All-Azure denominator, total users, burst quantiles, geography, tenant weights, and fitted family parameters. |
| Microsoft Azure / Splitwise | One day; two production services; a few thousand requests each | Conversation and Coding expose distinct empirical input/output token distributions and rates. | Replay the two service classes separately when testing prefill/decode balance or phase splitting. | Longer-window drift, tenants, geography, sessions, disclosed model weights, and deployed-cluster proof. |

### What the corpus can and cannot say about Zipf

- **Supported:** popularity, load, token lengths, tool counts, and reuse can be skewed,
  category-dependent, heavy-tailed, bursty, bimodal, multimodal, and time-varying.
- **Unsupported:** one global Zipf exponent for users, tenants, models, prefixes,
  sessions, or token volume.
- **Required before using Zipf:** name the random variable and population; fit the
  exponent and cutoff; report estimator, goodness of fit, confidence interval, time
  window, and drift; compare against lognormal, Pareto, and alternatives.
- **Benchmark fallback:** if those fields are absent, label Zipf/Pareto/lognormal/Poisson/
  Hawkes/MMPP inputs as synthetic sensitivity scenarios rather than production measurements.

### Directly observed versus still missing

| Variable | Current evidence | Status |
|---|---|---|
| Request arrival and burst | Billions of requests across ServeGen, FineServe, and Chutes; second-scale extremes and release-driven drift are observed. | Partial: raw regional/tenant fits are not public. |
| Periodicity and multi-duration bursts | Azure OpenAI / BurstGPT reports daily and weekly periodicity plus 10-second, 60-second, and 600-second burst case studies. | Partial: examples are not quantiles and four services are not all Azure traffic. |
| Model popularity | One-year Chutes evolution and broad long-tail model coverage. | Partial: no stable universal rank exponent. |
| Tenant/client concentration | ServeGen identifies 29 dynamic top clients among 2,412 profiled clients. | Partial: exact traffic shares and identities are unavailable. |
| Prefix popularity/reuse | Aliyun category-conditioned reuse, Chutes bimodal realized cache fractions, Copilot session reuse. | Partial: policy-independent opportunity and cross-tenant fits are missing. |
| Prompt/output length | FineServe task/model modes, Copilot archetypes, TraceLab agent sessions, output-length uncertainty quantiles. | Partial: no cross-provider production family/parameter table. |
| Geography and seasonality | ServeGen has dozens of regions; global traces expose temporal change. | Missing parameters: timezone, country, diurnal amplitude, weekday/weekend, holiday/event coefficients. |
| Failure and retry | Copilot tool failure and retry amplification; Anthropic multicloud postmortem elsewhere in the corpus. | Partial and coding-agent-heavy. |
| Reasoning-mode selection | Model releases expose configurable reasoning, but production selection shares are not public. | Missing. |
| Speculative acceptance | Sailor2 names a 1B speculative tier, but no acceptance distribution is present. | Missing. |
| Installed-to-goodput conversion | Physical/capacity evidence exists in other slices. | Missing a joined production distribution from installed → healthy → schedulable → active → useful goodput. |

## Source-bounded production parameter matrix

The matrix below records only what each trace can support. `Approx.` marks a
value read from a released figure rather than reported in prose or a table. None
of the three sources reports population confidence intervals.

| Source | Variable | Population | Window | Estimator / metric | Family or empirical form | Parameters | Fit quality / selection | Confidence interval | Drift | Missing fields |
|---|---|---|---|---|---|---|---|---|---|---|
| FineServe | arrivals | Global marketplace requests to 29 open-source models and 9 task intents | 23 days; 300-second bins for short-horizon signals | CV and mean squared successive difference; per-workload candidate comparison | Poisson-family, negative-binomial-family, self-exciting / Hawkes, and Markov-modulated candidates | No universal parameter vector reported | Selected family varies by model, scale, and task; no universal winner | Not reported | Architecture, scale, task, and time dependent | Request denominator, geography, tenant shares, retries, and CIs |
| Azure OpenAI / BurstGPT | service arrivals, tokens, users, and failures | Four Azure OpenAI-powered services | 213 days | Empirical request counts, periodicity, token distributions, burst case studies, and failure fractions | Daily/weekly periodic empirical trace with long-tailed tokens | 10.31M requests; dominant period about 1 day; bursts 400 req/s × 10 s, 100 req/s × 60 s, 15 req/s × 600 s; GPT-4 failure fractions about 0.068/0.034/0.005 by separated class | No universal parametric family or burst quantiles reported | Not reported | Daily and weekly periodicity reported | All-Azure denominator, total users, geography, tenant shares, burst quantiles, and CIs |
| Microsoft Azure / Splitwise | input/output tokens and rates | Conversation and Coding production services | One day | Empirical distributions and rates | Service-conditioned empirical distributions | A few thousand requests per service | No universal fit reported | Not reported | Not measurable from one day | Tenants, geography, sessions, longer drift, model weights, and deployed-cluster receipt |
| FineServe | dense-model token geometry | Same trace, dense-model requests | 23 days | Conditional median / trend read from figure | Piecewise empirical envelope | Approx. output peak 620 tokens at 1,500 input; stable near 165 at input >=5,000 | No parametric fit reported | Not reported | Not reported | Per-bin counts and uncertainty |
| FineServe | MoE token geometry | Same trace, MoE-model requests | 23 days | Conditional median / trend read from figure | Piecewise empirical envelope | Approx. 100 output at 500 input; 290 at 5,000; maximum near 400 at input >=8,000 | No parametric fit reported | Not reported | Not reported | Per-bin counts and uncertainty |
| FineServe | task-conditioned output | Programme, Science, Law, Social, and Writing intents | 23 days | Conditional trend read from figure | Peaked or approximately flat empirical curves | Approx. Programme `(input peak 1,100, output peak 950, tail 520)`; Science `(1,200, 860, 540)`; Law `(750, 620, 280)`; Social flat near 140; Writing flat near 155 | No parametric fit reported | Not reported | Task dependence reported; temporal drift not reported | Per-task denominators and uncertainty |
| ServeGen | text client concentration and heterogeneity | 2,412 production text clients | One day | Ranked request share; per-client CV | Empirical ranked share and client distributions | Top 29 clients account for 90% of requests; cross-client CV range exceeds 2x | Generator validates reconstructed marginals; no universal popularity family selected | Not reported | One-day temporal structure modeled; longer drift unresolved | Geography, tenant definition, retry/failure, and CIs |
| ServeGen | multimodal prompt size and client rate | Production image / audio / video / omni requests | One day | Empirical CDF and client maxima, read from figures | Modality-conditioned empirical distributions | Approx. image count P90 = 5; prompts with >1 image about 20%; maximum client request-rate variation about 5x | No parametric family selected | Not reported | One-day trace only | Population denominators by modality and CIs |
| ServeGen | reasoning arrival and output geometry | Production reasoning-model requests and clients | One day | Maximum request rate; correlation with configured budget; output / budget ratio; tail extent | Budget-conditioned empirical distribution | Approx. model max 2.17 RPS; client max 0.047 RPS; output-budget correlation 0.7; minimum output / budget ratio 1.05; heavy tail starts near 4K and reaches 40K output tokens; multi-turn share nearly 10% | No universal arrival or length family selected | Not reported | One-day trace only | Budget mix, stop causes, retries, and CIs |
| A Year in LLM Serving | corpus scale | Chutes production trace | 365 days | Direct counts | Census-like trace summary | 6.122B requests; 314,970 users; 9,174 models; 35,795,761 million input tokens; 2,522,394 million output tokens | Not applicable | Not reported | Monthly evolution analyzed | Geographic and retry/failure fields |
| A Year in LLM Serving | model and user concentration | Same trace | 365 days | Empirical shares / CDFs, partly read from figures | Empirical heavy concentration; no universal Zipf fit | Approx. 75% of models have one user; 67% have <100 requests; >70% of users target one or two models | No universal popularity exponent reported | Not reported | Model popularity and user behavior evolve over the year | Tenant / organization boundaries and CIs |
| A Year in LLM Serving | cadence and periodicity | Same users and `(user, model)` pairs | 365 days | Periodicity detection and empirical IAT CDFs | Empirical mixture | Approx. 22% of users show daily periodicity; user median IAT spans 0.02-105,700 s; about 50% of consecutive same-pair requests arrive within 0.1 s and 80% within 10 s | No universal stationary arrival fit | Not reported | Monthly distributions drift | Timezone and geography |
| A Year in LLM Serving | prefix reuse | Same trace, exact prompts and token prefixes | 365 days | Exact-request hit ratio, token hit ratio, and reuse-distance CDF | Empirical cache-locality curves | Approx. exact-request hit about 5%; token hit varies below 20% to above 80% by model; 50% of prefix reuse within 90 s and 95% within 24 h | Cache result depends on model and metric | Not reported | Monthly drift present | Cache policy sensitivity and uncertainty |
| A Year in LLM Serving | selected periodic models and load balance | Selected periodic workloads and replica counts | Case-study windows within the year | Cacheable-token share, token-hit ratio, max / mean and max / min load | Empirical case studies | Approx. 85% cacheable input tokens and 78% token hit for selected periodic models; max / mean load 5.5-8.8x; max / min 4.8x | Case studies, not population fits | Not reported | Configuration dependent | Broader-model denominator and CIs |

Use these values as separate benchmark classes, not as one merged synthetic law.
Request share is not token share; exact prompt hits are not prefix-token hits;
aggregate arrival curves are not per-client processes; and reconstruction error is
not a population confidence interval.

## Geography and session-locality parameter addendum

| Source | Metric | Population / window | Parameters | Do not infer |
|---|---|---|---|---|
| OpenRouter | weekly regional spend share | >100T production tokens; source does not publish exact observation dates for this geography figure | North America <50% for most weeks; Europe about 15-22%; Asia about 13% initially and 31% most recently | Spend share is not request, token, user, tenant, or serving-region share |
| SkyLB / SkyWalker | country-local demand and cross-region system result | WildChat-derived demand mapped across six countries; evaluation window and country denominators are source-specific | 1.12-2.06x throughput, 1.74-6.30x lower latency, 25% lower serving cost; round-robin peak KV imbalance 2.64x | System improvement is not a fitted geography distribution |
| SageServe | multiregion placement envelope | >8M trace requests, four models, three regions | up to 25% GPU-hour savings and 80% lower scaling overhead | No tenant, timezone, or session distribution is disclosed |
| Chutes | periodicity and reuse without geography | 365 days, 314,970 users | about 22% daily-periodic users; roughly half of repeated `(user, model)` requests within 0.1 s and 80% within 10 s | Daily periodicity cannot be assigned to a timezone or country |
| ServeGen | client-composed daily demand | one day, 2,412 text clients | top 29 clients produce 90% of requests; client rates and CVs vary materially | Aggregate daily shape is not a regional or tenant distribution |

The benchmark schema must therefore carry request origin, local time, billing geography,
serving region, tenant, session, prefix, WAN latency, sovereignty eligibility, retries,
and metric units separately. The fuller field contract is in
[`geography-session-locality.md`](geography-session-locality.md).

## Serving mechanism parameter addendum

| Source | Model / hardware envelope | Mechanism | Reported result | Required receipt fields |
|---|---|---|---|---|
| Orca | Transformer models through 175B; 2022 distributed GPU stack | iteration-level scheduling and selective batching | up to 36.9x throughput at the same latency versus FasterTransformer | scheduler version, model, parallelism, active sequences, arrival/length trace, latency target |
| vLLM | A100 40GB and A10G 24GB evaluations; ShareGPT and sampling traces | PagedAttention / block-based KV allocation | 2-4x throughput at comparable latency versus FasterTransformer and Orca | block size, useful/reserved KV bytes, fragmentation, sharing, eviction, concurrency |
| Sarathi-Serve | Mistral-7B on one A100; Falcon-180B on 64 A100s | chunked prefill and stall-free batching | 2.6x capacity versus vLLM in the Mistral case; 6.3x versus Orca and 4.3x versus vLLM in the Falcon case | chunk size, TTFT/TPOT SLO, prefill/decode mix, scheduler overhead, fairness |
| DistServe | evaluated models, GPU groups, traces, and TTFT/TPOT SLOs | prefill/decode disaggregation | up to 7.4x request rate or 12.6x tighter SLO | phase allocation, topology, KV-transfer bytes/time, utilization, stranded capacity, SLO attainment |

Do not multiply these maxima. They use different years, baselines, models, hardware, traces,
and SLOs. The benchmark question is the current break-even surface, not whether a mechanism
once beat an older stack.

## Agentic system-envelope addendum

| Source | Population | Measured envelope | System implication | Missing distribution |
|---|---|---|---|---|
| AgentSysBench | 10 agentic applications plus a production-trace sidecar | Non-LLM work dominates latency in 5/10; sandbox state reaches 28 GB/session; intra-application task latency differs up to 32x; production state idles minutes-to-hours | Model the workflow DAG, component affinity, sandbox state, transfers, and idle residency | Universal session length, tool-chain length, retry/failure, tenant, and geography |
| AgentSysBench optimization probes | Evaluated task graphs and load levels | Task disaggregation cuts latency 29-40%; communication-aware placement reaches 4.5x; state offload cuts memory 4.6x | Separate mechanism gain from workload distribution and count transfers/overhead | Cross-provider replication and CIs |
| AgentSysBench tool trace | Production query/fetch traces with a 10-minute query TTL | 35.2% redundant search calls removed; aggregate search latency falls 19.3% | Cache exact tool queries/results with freshness and authorization boundaries | Query population, user concentration, staleness cost, and failure/retry rates |

These parameters complement, rather than replace, the existing Copilot and TraceLab rows.
Copilot contributes production model/tool prevalence; TraceLab contributes trajectory and
state reconstruction; AgentSysBench contributes heterogeneous component, memory, idle, and
tool-cache envelopes.

## Agentic accounting additions

| Variable | Production evidence | Modeling rule | Missing |
|---|---|---|---|
| Calls per user task | Copilot: 760.5M LLM calls and 774.7M tool calls over 95.1M turns; TraceLab: mean 8.8 LLM and 10.8 tool calls per request. | Draw workflow fan-out and alternating model/tool phases before token lengths. | Product/generalization beyond coding, tenant/geography, and per-tool breakdown. |
| Workflow wall time | TraceLab mean 4.3 minutes; P90 above 6.4 minutes. | Model multi-minute state occupancy and tool/runtime queues. | Cross-product wall-time and critical-path distributions. |
| Idle retained state | Copilot average container idle 4.1 minutes and KV idle 2.9 minutes. | Charge resource-time across idle gaps and compare keepalive versus rehydrate. | Tail quantiles, memory/container sizes, and user think time. |
| Tool failure and retry | Copilot: 9% of turns with tool failure; retry amplification up to 4× compute. | Replay failure type, retry budget, idempotency, partial side effects, and abandonment. | Production distributions outside coding agents. |
| Tool-result reuse | Systems work motivates semantic tool caching. | Treat tool result as a separate object with tenant, freshness, and side-effect policy. | Direct production redundancy/hit distributions. |
| Workflow critical path | Systems studies expose DAG-aware scheduling. | Optimize accepted task completion and critical-path delay, not isolated request latency. | Public production DAG shapes and stage-time shares. |

## Geographic and session-locality additions

| Variable | Evidence | Modeling rule | Missing |
|---|---|---|---|
| Regional diurnal demand | SkyWalker uses six-country WildChat demand with distinct local-time peaks. | Sample region/timezone first; preserve local peak phase and WAN/residency constraints. | Provider production country weights, enterprise/API segmentation, holidays, launches, and exact amplitude. |
| Multi-region mixed SLOs | SageServe evaluates >8M production-trace requests, 4 models, and 3 regions with latency-sensitive/insensitive classes. | Jointly sample region, model, SLO class, and provisioning type; include placement/scale overhead. | Public raw trace, tenant weights, region matrix, and real spot interruptions. |
| Agent session hotspot | SMetric studies 2 real-world agent traces and finds cache-affinity routing can overload a few instances. | Track per-session cached state and projected remaining work, not only queue or hit. | Production session share, cache size, hardware, tool-gap, and fairness distributions across products. |
| Tool-gap KV lifetime | Continuum assigns cache TTL from reload cost and eviction-induced queueing during tool calls. | Draw tool-gap and return probability; charge retained-memory time versus offload/reload. | Cross-product tool-gap quantiles, KV sizes, abandonment, and storage bandwidth. |
