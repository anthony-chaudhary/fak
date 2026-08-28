---
title: "Workload and cluster assumption registry"
description: "As of: 2026-08-27. This registry turns the source ledger into assumptions that can be."
---

# Workload and cluster assumption registry

**As of:** 2026-08-27. This registry turns the source ledger into assumptions that can be
accepted, rejected, or parameterized in fak benchmarks. It is not a substitute for the
per-source provenance in [`index.json`](index.json).

## Google AI Hypercomputer topology and admission assumptions (#9384)

The five-source Google slice supports a typed control-plane benchmark, not one flat
accelerator count:

| Boundary | Source-backed envelope | Benchmark rule |
|---|---|---|
| Cluster Director GPU hierarchy | Physical host → single-rack sub-block → block → cluster; one GPU-network hop inside a sub-block, at most two inside a block; clusters can scale to thousands of GPUs | Treat “thousands” as a supported scale class, not a configured maximum, prevalent layout, available fleet, or active job |
| Current GKE TPU Multislice | Two or more homogeneous multi-host slices; same TPU type, size, and topology; ICI within each slice, DCN between slices; synchronous multicontroller training; atomic node-pool scale-up | Preserve job/slice/VM-or-host/chip levels and admit the slice atomically; no current maximum slices-per-JobSet is public |
| Legacy Cloud TPU API | v4+; one Multislice node equals one slice; maximum 256 slices per queued resource; gang scheduling is all-or-none | Version the 256-slice ceiling to the maintenance-only API; do not copy it into GKE or future TPU control planes |
| Dynamic Workload Scheduler | Flex-start: standalone wait up to two hours, MIG request persists until available/canceled, 10-minute to seven-day run, best-effort capacity and dense placement. Reservation-bound/calendar: up to 90-day workloads, very high assurance after approval, exclusive reserved capacity | Record requested → pending → approved/admitted → provisioned → running → expired separately; a wait ceiling is not a queue-wait distribution |
| XPK/GKE achieved run | 50,944 TPU v5e chips across 199 pods; 256 chips/pod; ICI within pod and Jupiter DCN across pods; XPK creates/resizes clusters and submits Kueue JobSets | Treat as provider-reported achieved active work, not announced capacity, a schedulable customer maximum, or a prevalent topology |

Use separate physical and workload hierarchies in receipts:

- physical: `fleet/region → cluster → block or TPU pod → slice → host/VM → chip`;
- workload allocation: `job/JobSet → one or more pods/slices → hosts/VMs → chips`.

The order is not identical across GPU and TPU products: Cluster Director exposes
host/rack/block/cluster placement, while Multislice exposes job/slice/host/chip
communication domains. Do not manufacture a one-to-one mapping between a Cluster
Director block, a TPU pod, and a TPU slice.

Missing values remain first-class parameters. None of these official sources reports a
production queue-wait distribution, utilization, provider-wide healthy/schedulable/active
scale, failure or retry distribution, power, energy, or total workload cost. Sweep those
values synthetically or leave them unknown; do not derive them from supported maxima or
the 2023 achieved job.

## Provider-scale population denominators (#9379)

Official first-party disclosures now bound several unlike populations: 950 million Gemini
app MAU; more than 2.4 million Antigravity WAU; more than 9 million monthly model
developers; approximately 22 billion model-API tokens per minute; almost 1 billion Meta AI
MAU; 1 billion
people using ChatGPT weekly; more than 30 million paid Microsoft 365 Copilot seats; 50
million GitHub Copilot users; 100,000 Microsoft Foundry customers; nearly 40 million
registered Agent 365 agents; more than 50 billion Purview-audited Copilot interactions to
date; and more than 100,000
customers running Claude on Amazon Bedrock. Google also reports customer cohorts above
explicit annual token thresholds, which establish high-volume enterprise populations but not
total traffic. These quantities are reach, entitlement, customer, or threshold-cohort
denominators, not directly comparable traffic measurements.

Preserve the denominator and product boundary in every benchmark or capacity claim:

- registered users, WAU, MAU, and DAU are distinct; a DAU growth multiple is not a DAU count;
- consumer users and subscribers are distinct from paid enterprise seats, organizations,
  business customers, API developers, and API customers;
- Gemini app, Meta AI, ChatGPT, Microsoft 365 Copilot, and Anthropic business customers do not
  represent provider-wide or model-specific traffic;
- requests or queries are distinct from sessions, messages, and tokens; and
- a population count supplies no concurrency, request rate, interarrival law, geography,
  session length, tenant concentration, market share, or Zipf parameter.

Therefore fak may use these disclosures to test population-scale metadata and denominator
hygiene, but must not synthesize arrivals, concurrency, cache popularity, or capacity from
them. Any such workload shape still requires a direct production trace or an explicitly
synthetic sensitivity sweep.

## Denominator and locality additions (#9362)

The new Chinese-platform evidence reinforces typed denominators: developer teams,
applications, developers, API growth, agents, user sequences, unique users, sessions,
requests, and tokens are not interchangeable. MTServe adds per-request user/item state
whose reuse geometry differs from exact prompt or token-prefix reuse; MTGenRec adds a
one-week sequence-training population, not a serving arrival law. None supports a
universal Zipf, lognormal, Pareto, Poisson, Hawkes, or MMPP model.

## Speculation, failures, and retries (#9366)

Speculative acceptance is not one global probability. It varies by draft/target pair, task,
temperature, token position, context length, batch/load, tree geometry, and implementation.
Provider failure reasons are not client retry counts, and rejected draft tokens are compute
work even when they never become output. The current papers provide bounded mechanism
benchmarks, not a production acceptance or retry distribution; that gap remains explicit.

## Non-coding agent workload boundary (#9367)

Browser pages, enterprise DOMs, API/database mutations, open-web retrieval, files, screenshots,
GUI actions, and environment resets create workload components absent from coding-only traces.
The benchmark task/instance counts are designed evaluation populations, not user, session,
arrival, turn-count, action-count, or tool-frequency distributions. No Zipf or other universal
popularity law follows from these benchmarks; production trajectory telemetry remains required.

## Public conversation denominator boundary (#9370)

The new datasets expose real or crowdsourced conversation geometry but do not identify one
universal user or popularity distribution. Anonymized/IP-derived users are not verified people;
votes are not conversations; messages are not sessions; branches are not independent arrivals;
language and inferred country are not tenant or spend shares. Any Zipf, geographic, session,
turn-count, or model-demand claim still requires an explicit fitted population and collection bias.

## Distribution assumptions

### Production distribution evidence audited for #9381

The public production traces support **bounded empirical distributions, not one universal
family**. Keep the random variable, denominator, product, window, and aggregation level in
every benchmark record. In particular, heavy tail is not a power law; a power law is not
Zipf unless the variable is rank-frequency and the fit is supported; an empirical CDF is
not a fitted family; and a synthetic generator is not production evidence.

| Primary trace | Exact variable / rank variable | Population, window, granularity | Evidence method | Reported family / parameter | Stationarity and denominator limit |
|---|---|---|---|---|---|
| BurstGPT / Azure OpenAI v1 (2024-01-31) | Aggregate request arrivals; interarrival time derivable from released timestamps; per-request input and output token counts | Azure OpenAI GPT services, 121 days; separate research API, 36 days; request-level traces and aggregate time-series/histogram views | Released trace plus qualitative plots/empirical distributions | None; no fit statistic in v1 | Aggregate arrivals are not per-user/client arrivals; the two products/windows are separate; later v1.1 synthetic Zipf sampling is outside this record and is not a production fit |
| Chutes Year-in-Serving | Per-model request-count rank/share; selected-model interarrival time; per-request input/output length | Chutes.ai, 212B requests and 46T tokens across 60K models, calendar 2025; product aggregate or selected-model figures | Rank/share summaries and empirical distributions | None; no fit statistic | Model request share is not token, spend, prefix, or user share; rankings turn over during the year; selected models are not the whole product |
| ServeGen / Alibaba Cloud Model Studio v3 | Per-request input/output length; one-minute interarrival time; aggregate request rate | 300+ APIs, 2024-11-25 through 2024-12-01; one-minute IAT samples and five-minute rate buckets | KS tests for Exponential/Gamma/Weibull IAT candidates; fitted length curves; linear trend plus 24h/12h Fourier seasonality | Best IAT family by tier: Gamma (M-large), Weibull (M-mid), Exponential (M-small); category inputs: Pareto/lognormal mixture; outputs: Exponential. No mixture parameters or length-fit statistic published | Fits are variable- and tier/category-specific; KS p-values vary and may be insignificant. One aggregate week does not prove stationarity, per-client laws, or a universal family |
| Alibaba Cloud Model Studio KV-cache trace | Request input/output length; shared-prefix length/ratio; prefix-tree height/width; session request count; cacheable-prefix reuse frequency/lifetime | 58 workloads, 2024-06-07 through 2024-07-25; request-, workload-, prefix-, session-, and block-level denominators vary by figure | Empirical CDFs and workload summaries | None; no rank exponent or fit statistic | Prefix frequency is not model popularity or user/token share; 49 days and cross-workload heterogeneity do not prove stationarity |

No retained primary trace supports a universal Zipf, lognormal, Pareto, Poisson, Hawkes, or
MMPP law. ServeGen does support bounded, source-specific Gamma/Weibull/Exponential IAT fits
and Pareto/lognormal-mixture and Exponential token-length fits; retain them only with their
exact variable and tier/category. Other named families remain labeled sensitivity or synthetic
scenarios unless a benchmark cites a source-specific fit or computes one reproducibly from an
official released trace.

| Dimension | Convenient but weak default | Evidence-backed expectation | Benchmark requirement |
|---|---|---|---|
| Request arrivals | Stationary Poisson | Bursty, time-varying, client-composed, and shifted by releases/availability | Include stationary baseline **and** regime-switching/burst scenarios; label synthetic generators. |
| Model popularity | Uniform or one permanent Zipf | Heavy-tailed/long-tail and longitudinally changing; user-model affinity matters | Sweep head/tail mass and churn; do not cite a Zipf exponent without a measured population. |
| Tenant/user demand | IID users | Highly unequal clients/tenants are likely, but public concentration parameters are scarce | Carry tenant IDs; sweep concentration; report missing production calibration honestly. |
| Prefix popularity | Global LFU/Zipf | Skewed and category-dependent; single-turn reuse can rival multi-turn reuse; per-replica affinity may be bimodal | Measure by channel/request class/time and compare LRU/LFU/learned policies. |
| Input tokens | One fixed length | Model-, modality-, task-, and client-specific, with long-context tails | Use empirical or mixture distributions; report P50/P90/P99 and truncation. |
| Output tokens | Fixed or accurately predicted | Strongly right-skewed and uncertain even for identical prompts; top requests dominate work | Model conditional uncertainty and tail quantiles; include preemption/fairness effects. |
| Reasoning compute | Fixed decode length | Seconds-to-minutes, selectable budgets, multi-sample/high-compute modes | Sweep reasoning budgets, samples, and early-exit policies; measure quality and accepted work. |
| Modality | Text only | Text, image, audio/video, and computer-use/tool phases create different work | Preserve modality and preprocessing/tool phases in traces. |
| Agent sessions | Independent chat requests | Long append-only loops, tool calls, idle/human gaps, and partial prefix reuse | Benchmark session affinity, cache residency across idle gaps, and tool latency. |
| Geography/time | One region, constant load | Multi-region userbases, diurnal/weekly effects, sovereign placement, and site constraints | Parameterize region, jurisdiction, time, and failover; do not average away peaks. |
| Failures/retries | Zero | Node, rack, network, software, quota, and tool failures affect useful goodput | Inject failures and count retries/recovery/verification in net-true output. |

### Zipfian claims

The corpus supports **heavy tails and skew**, but not a universal Zipf law for frontier AI
traffic. “Zipfian” is acceptable only as a transparent synthetic sensitivity test unless a
source provides the fitted population, interval, exponent, goodness-of-fit, and drift. A
model-popularity Zipf does not justify the same exponent for tenants, prefixes, prompts,
output lengths, tools, or geography.

### Production-trace distinctions

- **No universal arrival or popularity law is supported.** FineServe selects among Poisson-family, negative-binomial-family, self-exciting / Hawkes, and Markov-modulated candidates per workload; Chutes reports evolving empirical concentration without a universal Zipf exponent.
- **Keep request share separate from token share.** A client or model can dominate request count without dominating prefill or decode work.
- **Keep exact prompt hits separate from prefix-token hits.** Chutes shows a small exact-request hit rate alongside much higher token-prefix reuse for some models.
- **Keep aggregate arrivals separate from client arrivals.** ServeGen's aggregate daily curve coexists with large cross-client rate and burstiness differences.
- **Partition text, multimodal, reasoning, and agentic traffic.** Modality counts, reasoning budgets, multi-turn sessions, and tool-mediated pauses create different service-time and locality processes.
- **Treat stationary fits as window-scoped.** A one-day reconstruction, a 23-day fit, and a 365-day evolving trace answer different questions; monthly drift invalidates a single frozen default.
- **Preserve source-bounded periodicity and burst scales.** Azure OpenAI / BurstGPT adds daily and weekly periodicity plus 10-second, 60-second, and 600-second burst examples, but those examples are not quantiles and four services are not all Azure traffic.
- **Separate service and failure classes.** Splitwise's Conversation and Coding traces have distinct empirical token/rate shapes, while BurstGPT separates instance, trigger/context-exceed, and content-policy failures; neither supports one universal service or retry process.

### Production benchmark class matrix

Run at least these workload classes independently before mixing them:

| Class | Required conditioning | Primary stress |
|---|---|---|
| Text chat | client, input/output lengths, turn depth, time of day | prefill/decode mix and client burstiness |
| Multimodal | modality and media-count distribution | encoder cost and prompt-size clusters |
| Reasoning | model, reasoning budget, output tail, multi-turn flag | long decode occupancy and budget-conditioned tails |
| Agentic | session, tool-call chain, think/tool/wait timing | synchronized waves, idle gaps, retries, and long-lived state |
| Marketplace / multi-model | model architecture, scale, task intent, popularity epoch | routing churn, heterogeneous arrivals, and model residency |
| Cache-locality | exact-request hit and token-prefix hit as separate metrics | TTL, reuse distance, prefix placement, and migration |

Do not collapse this matrix into one Zipf-plus-Poisson default. Record the
population, observation window, estimator, selected family, parameters, fit
quality, confidence interval, and drift for every generated trace; write `not
reported` rather than manufacturing missing evidence.

The preserved conclusion is category-specific: observed workloads are
**category-dependent, heavy-tailed, bursty, multimodal, and nonstationary**. The corpus
does not support a universal Zipf, lognormal, Pareto, Poisson, Hawkes, or MMPP law.

## Serving assumptions

| Surface | Evidence-backed expectation | Required receipt fields |
|---|---|---|
| Batching | Interactive continuous batching and deadline-tolerant offline batch are different products; batch composition is disrupted by variable decode lengths and SLOs | batch type, active sequences, admitted/rejected work, TTFT, inter-token latency, completion deadline, padding/waste |
| Prefill/decode | Bottlenecks differ enough to motivate disaggregation and specialized chips, but transfers can erase gains | placement, topology, KV bytes moved, transfer time, queueing, retries, quality, accepted-token goodput |
| KV cache | Multi-tier memory/storage, locality-aware routing, eviction, prefetch, and session affinity matter | key scope, hit type, reuse distance/time, bytes, tier, eviction cause, recompute/transfer cost |
| Speculation | Draft/verify changes work and quality; benefits depend on acceptance and hardware balance | draft model, verifier, accepted/rejected tokens, quality check, extra compute, net latency/cost |
| Routing | Hardware, region, cache locality, tenant SLO, modality, and reasoning budget all influence placement | model revision, engine, accelerator, region, route reason, fallback, cache state, SLO |
| Autoscaling | Requests are an incomplete load signal; token-phase work, bursts, cache warmth, and startup time matter | queued/running prefill and decode tokens, warm capacity, startup delay, saturation, shed work |
| Fairness | Long outputs and large prefills can monopolize batches/queues | tenant/user, waiting time, slowdown, preemptions, deadline misses, starvation metrics |

## Cluster and datacenter assumptions

| Boundary | Expectation |
|---|---|
| Accelerator | Multiple generations and vendors coexist; H100-equivalent counts are not physical inventory. |
| Host | CPU, RAM, NICs, storage, NUMA, and software versions can gate accelerator goodput. |
| Rack | Rack-scale fabrics and liquid cooling are first-class for MoE, reasoning, and dense inference. |
| Pod/cluster | Thousands to hundreds of thousands of accelerators use mixed parallelism and need topology-aware scheduling/RAS. |
| Site | Power, cooling, water, grid interconnect, optics/electrical supply, construction, permitting, and community consent gate delivery. |
| Multi-site fleet | Training and serving can cross regions/sites and clouds; WAN, sovereignty, failover, and data location matter. |

A capacity number must be typed as one of: **announced, intended, contracted, under
construction, powered, installed, accepted, healthy, schedulable, allocated, active,
or useful-goodput-producing**. Conflating these states is a hard evidence defect.

## Metrics hierarchy

Use the strongest available outcome and retain the weaker denominators:

1. accepted quality-constrained task completions;
2. accepted output tokens within SLO;
3. cluster/service goodput within SLO;
4. raw tokens, requests, or jobs completed;
5. active/healthy accelerator time;
6. allocated accelerator time;
7. installed accelerators or nameplate power;
8. announced/contracted capacity.

Cost and resource accounting should include accelerator/CPU/network/storage work, cache
transfer/recompute, routing/control-plane work, retries, verification, idle reservation,
energy, cooling/water boundary, and long-term commitments where relevant.

## Missing calibration data

Public sources still do not adequately reveal:

- tenant concentration and active-user/request/token conversion;
- fitted prompt/prefix popularity distributions and drift by product;
- geography and diurnal/weekly seasonality by model/channel;
- session length, idle gaps, retry/failure, and tool-call distributions at frontier-provider scale;
- real prefill/decode mix, cache-hit opportunity, speculative acceptance, and routing decisions;
- installed-to-healthy-to-goodput conversion for announced million-chip/gigawatt fleets.

Until these are measured, benchmark sweeps must expose the assumption ranges rather than
present one guessed distribution as “the frontier workload.”

## Operator-level autoscaling envelope (#9387)

OpScale supports a bounded operator-level benchmark fixture, not a production-default
assumption. Keep these evidence modes separate:

| Evidence mode | Exact disclosed envelope | Safe benchmark use |
|---|---|---|
| Offline analytical opportunity model | Qwen2-7B and Qwen2-57B-A14B; 10-100 QPS; 128-64K sequence lengths; production-trace-derived length input; each operator approximated as M/M/R with Poisson arrivals, exponential service, and Erlang-C waiting time; exhaustive SLO-compliant search | Sensitivity/oracle only. Do not label the Poisson queue law as the production replay's arrival distribution. |
| Dynamic prototype trace replay | 929K requests and 1.5B prompt tokens per model; Qwen2-7B and Qwen2-57B-A14B; a displayed one-hour window; 40 A100-80GB GPUs in five Azure VMs; NVLink within each eight-GPU VM and InfiniBand between VMs | Replay evidence, not a production deployment. Preserve the source trace, model, prompt-token denominator, hardware topology, and one-hour displayed-window boundary. |
| Static/minimum-capacity prototype sweeps | Dense QPS points 80/120/160/200/240; MoE 25/50/75/100/125; 1K/4K/8K sequence lengths; fixed clusters up to 40 GPUs; capacity increased until the latency-SLO violation threshold is met | Compare total GPUs and total power against model-level provisioning, or maximum sustainable **input** TPS against the model-level static deployment on the same GPU budget, only under the matching model, length, hardware, and TTFT threshold. |
| Hardware/granularity sensitivity | A100 and 24-GB200 same-domain NVLink/NVL clusters; monolithic, Attn-FFN, and operator-level placement | A100 reports up to 33% lower GPU demand and 1.7x maximum **request throughput (RPS)** versus Attn-FFN. At the readable 40-QPS point, A100-to-GB200 GPU-count reduction is 52% for OpScale and 38% for the monolith. The figure omits model, length, trace/arrival law, and SLO, so treat it as incompletely specified topology sensitivity. |
| Profiling/control microbenchmarks | Typical profile space B=1-256, L=1-65,536, SM=1-100%; 57B profile under one hour on one GB200 node; Qwen2-7B plan 2.6 ms median/3.2 ms P99; Qwen2-57B-A14B plan 4.4 ms | The batch range is an offline profiling grid, not configured or achieved active batch. Charge profiling, plan error, placement, dispatch, transfer, and actuation separately. |

The dynamic controller uses a one-second interval; model-level baselines use 20 seconds.
For Qwen2-7B, the paper reports a one-second P99 TTFT SLO, 7.1 average GPUs, and
98.4% SLO attainment, versus 11.2/14.3/13.0 GPUs for
DynamoLLM/AIBrix/Production Stack and 88-95% baseline attainment. For
Qwen2-57B-A14B, the corresponding two-second P99 TTFT envelope reports 10.8
average GPUs and 98.1% attainment, versus 17.3/23.1/18.0 GPUs and 84.2-97%.
The MoE figure/prose gives an approximate OpScale high-load peak near 25 GPUs
and says AIBrix/DynamoLLM frequently exceed roughly 30-35; retain the
approximation label.

Qwen2-7B on-demand scale-up latency is:

| Scaling action | Average | P90 | P99 |
|---|---:|---:|---:|
| Full model replica | 10.68 s | 11.04 s | 11.55 s |
| One operator | 0.03 s | 0.05 s | 0.10 s |
| 50% of operators | 0.18 s | 0.26 s | 0.42 s |
| All operators | 0.33 s | 0.38 s | 0.45 s |

Warm standby is counted as provisioned capacity even while idle, so these scale-up
comparisons are on-demand rather than pre-warmed. Horizontal operator replication is
the favored actuation in this prototype: dynamic operator resharding reports **11x**
higher overhead than replica scaling and **1.15 s P99**.

Use the following mechanism assumptions only as explicit experimental choices:

- retain a monolithic base-instance floor, then add elastic operator replicas;
- choose `(batch, replicas, tensor-parallel shards)` from traffic and the SLO model;
- place elastic replicas by interference-aware best fit, opening a new GPU only when
  no existing SLO-feasible placement exists;
- break placement ties by same device, then intra-server NVLink/NVL, then
  cross-server InfiniBand locality; and
- dispatch through capacity-weighted shortest operator queues.

The paper does **not** disclose achieved active batch, queue depth, pending prompt
tokens, numeric queue wait, achieved GPU-utilization percentage, realized
operator-replica totals or placement series, SM-allocation distribution, migration
count, configured per-model tensor/pipeline-parallel degrees, numeric TBT target,
failure/retry/cancellation behavior, output-token total, explicit goodput, or
currency/GPU-hour cost. Store each as `unknown`; do not derive them from the
profiling grid, average GPU count, qualitative plot lines, or configured baseline
signals. No OpScale code/data repository is linked in arXiv v1; referenced nano-vLLM
and other repositories are dependencies or baselines.

## Foundational serving mechanism matrix

| Mechanism | Helps when | Costs / break-even variables | Bounded evidence |
|---|---|---|---|
| Iteration-level scheduling and selective batching | decode lengths differ and completed sequences should leave immediately | scheduler cadence, kernel shape changes, active-sequence churn, fairness | Orca reports up to 36.9x over its 2022 FasterTransformer baseline on the largest evaluated model |
| Paged KV memory | variable sequence lengths create fragmentation and cap batch concurrency | block-table overhead, block size, useful/reserved bytes, eviction and sharing policy | vLLM reports 2-4x over its 2023 FasterTransformer/Orca baselines at comparable latency |
| Chunked prefill and stall-free batching | long prefills stall decode and TTFT/TPOT must be balanced | chunk size, launch overhead, TTFT, TPOT, throughput, preemption, fairness | Sarathi-Serve reports up to 2.6x capacity for Mistral-7B on one A100 and 6.3x vs Orca for Falcon-180B on 64 A100s |
| Prefill/decode disaggregation | phase interference and independent scaling exceed KV-transfer and stranded-capacity cost | input/output mix, TTFT/TPOT targets, transfer bytes/time, topology, allocation granularity | DistServe reports up to 7.4x request rate or 12.6x tighter SLO in its evaluated envelope |
| Hybrid aggregation/disaggregation | workload and SLO regimes change over time | reconfiguration, routing, placement, cache state, estimator error | TaiChi reports aggregation, disaggregation, or hybrid can each win different envelopes |

These are historical mechanism witnesses, not stackable universal multipliers. Re-run each
comparison against current fak-native kernels and the same model, quality, hardware, trace,
SLO, topology, and accounting boundary.

## Serving and cluster mechanism envelope

| Mechanism | Evidence now indexed | When it may help | Required counter-evidence before defaulting |
|---|---|---|---|
| Monolithic autoscaling | Simple model replicas remain the common baseline. | Stable model/workload mix; low control complexity; fast replica start. | Phase imbalance, long model load, heterogeneous hardware, or SLOs that require separate prefill/decode capacity. |
| Phase-specific autoscaling | NVIDIA Dynamo Planner scales prefill/decode replicas; HeteroScale reports production coordination at tens-of-thousands-GPU scale. | Prefill and decode demand diverge and topology/forecast signals are accurate. | Forecast error, cold-start and model-load time, network bottlenecks, failure recovery, and pool fragmentation. |
| Operator-level scaling | OpScale trace replay reports 7.1/10.8 average GPUs with 98.4%/98.1% SLO attainment for Qwen2-7B/Qwen2-57B-A14B on 40 A100s; versus model-level provisioning, separate sweeps report 20.1-36.3% GPU reduction, 14-28% lower cluster power, and up to 44% higher input TPS on the same GPU budget in specific envelopes. | Fine-grained operators have separable bottlenecks; one-second control and 0.03-0.33 s average on-demand operator actuation beat a 10.68 s full-model scale-out; topology permits safe colocation. | Research prototype/trace replay only; active batch, queue wait/depth, exact achieved utilization, configured TP/PP, realized operator counts/placement, failures/retries, and goodput are undisclosed; resharding is 11x slower than replica scaling at P99 1.15 s. |
| Aggregated prefill/decode | TaiChi reports an advantage under tight TTFT and relaxed TPOT regimes. | Interference is tolerable; first-token latency dominates; transfer overhead would be high. | Decode jitter, long outputs, strict TPOT, and queue interference. |
| Disaggregated prefill/decode | Dynamo/llm-d/TaiChi/TokenScale expose separate pools and KV transfer. | Strict decode SLO, phase-specific hardware, reusable prefill, or scaling asymmetry beats transfer/control cost. | Short prompts/outputs, weak fabric, small batches, KV transfer, extra failure domains, and underfilled pools. |
| Hybrid aggregation/disaggregation | TaiChi reports up to 77% benchmark goodput gain under balanced SLOs. | Traffic mixes contain both TTFT- and TPOT-sensitive requests and the scheduler can shift latency safely. | Maximum result is not universal; baseline, SLO mix, hardware, and scheduler overhead must match. |
| Token-work autoscaling | TokenScale reports higher SLO attainment and 4–14% lower cost in production-trace experiments. | Request counts/GPU utilization lag token work and burst backpressure. | Metric robustness under model churn, multimodality, failures, speculative decoding, and heterogeneous accelerators. |
| Prefix-aware routing and KV offload | llm-d, Dynamo, CacheRoute, Aliyun, Chutes, and Copilot evidence. | Prefix reuse is predictable enough to beat load imbalance and transfer/index overhead. | Cache staleness, privacy, fragmentation, routing skew, multi-tier latency, and policy-dependent realized hits. |
| Topology-aware heterogeneous placement | Dynamo DSX, AWS topology scheduling, HeteroScale, and training reliability evidence. | Communication-heavy phases and mixed accelerators/network tiers dominate. | Placement delay, fragmentation, gang-size constraints, failure domain coupling, and cross-generation quality/performance differences. |
| Tenant fairness/admission | Multi-tenant admission studies and token-pool research expose responsibility boundaries. | Shared fleets have priorities, quotas, budgets, or noisy-neighbor risk. | Per-tenant objectives, starvation, burst credits, cached-work ownership, cancellation, and auditability are still under-measured in production. |

### Default decision record

For every serving experiment, record:

```text
model + precision + quality
hardware SKU/count + topology + fabric
aggregated / disaggregated / hybrid phase layout
replica/operator scaling unit
arrival, prompt, output, prefix, tenant, and failure distributions
batch/admission/fairness policy
TTFT / TPOT(ITL) / E2E SLO and attainment
KV transfer, cache indexing, control, startup, recovery, and verification overhead
accepted goodput / cost / energy
```

The mechanism with the highest peak throughput is not necessarily the mechanism with the
highest SLO-satisfied, quality-constrained goodput.

## Agentic workflow boundary

AgentSysBench strengthens the boundary: in five of ten applications non-LLM components
dominate latency, per-session sandbox state reaches 28 GB, task latency differs by 32x,
and production state can sit idle for minutes to hours. A request-only arrival/service
model therefore misses the workflow DAG, component affinity, live OS state, transfers,
external-tool latency, and idle residency.

A model request is often the wrong scheduling/accounting unit for an agent. The current
production traces and systems work support a wider envelope:

```text
user task
  -> session / turn
  -> workflow DAG
  -> model call(s) + tool call(s) + sandbox/runtime work
  -> retries / compaction / cache lookup / policy checks
  -> accepted task outcome
```

| Agent assumption | Evidence | Benchmark consequence |
|---|---|---|
| Workflow DAGs matter | Parrot and workflow-aware scheduling treat dependent model/tool stages and critical paths explicitly. | Replay fan-out/fan-in, sequential dependencies, shared prefixes, and tool queues; report task completion, not only model latency. |
| Runtime and OS state matter | Agentic-OS research makes context, memory, tools, storage, policy, and concurrent agents first-class. | Include sandbox/container start, idle retention, filesystem/process state, authorization, and recovery overhead. |
| Tool reuse differs from KV reuse | Semantic tool-result caching targets repeated or near-duplicate tool calls with freshness and side-effect constraints. | Record tool/args/result/freshness/tenant/side effects; never reuse mutating results by semantic similarity alone. |
| Model speed can be non-critical-path | Copilot and TraceLab expose long alternating model/tool loops, idle state, failures, and retries. | Measure critical-path share and optimize the dominant stage; a faster decoder may not reduce task time. |
| Failures amplify whole workflows | Copilot reports 9% of turns with tool failure and retry loops up to 4× compute. | Replay partial failure, compensation, idempotency, retry budgets, and abandoned work. |
| Session state consumes capacity while idle | Copilot reports multi-minute average container/KV idle windows. | Account for retained KV, containers, files, and leases across idle gaps; request-only utilization is incomplete. |

### Required agent receipt

```text
session / turn / workflow identifiers
DAG stages and critical path
model + tool + sandbox + storage + network time
input/output/cached/compacted tokens
container/KV/filesystem state and idle lifetime
tool success, side effect, freshness, retry, and idempotency
policy/admission/fairness outcome
accepted task outcome, wall time, cost, and resource-time
```

## Geography and session locality

The current evidence separates three different facts: OpenRouter measures time-varying
regional **spend**, SkyLB/SkyWalker evaluates country-local diurnal demand and WAN-aware
routing, and Chutes/ServeGen measure user/client temporal locality without geography.
None licenses converting spend into request load, daily periodicity into timezone, or a
regional system gain into a universal demand law. See
[`geography-session-locality.md`](geography-session-locality.md).

- **Regional arrivals are phase-shifted, not one global stationary process.** SkyWalker
  observes distinct diurnal peaks across six country groups. Use region/timezone-specific
  curves and sensitivity ranges; WildChat is not a provider production distribution.
- **Cross-region pooling is constrained.** WAN latency, residency/export policy, carbon,
  failure domains, data movement, and session/KV locality can dominate spare capacity.
- **Agent sessions create hotspots.** Cache-affinity routing can strand idle replicas while
  a few session-owning replicas queue. Balance projected session work against reuse.
- **Idle KV needs a lifetime policy.** Tool gaps make uniform pinning waste memory and
  uniform eviction waste prefill. Compare TTL, offload, migration, and eviction with
  reload, queue, fairness, and abandonment included.
- **SLO class and geography interact.** Latency-sensitive and elastic/batch work can use
  different regions and provisioning types only when data, quality, and deadlines allow.
## Remote-browser operational envelopes (#9376)

Treat hosted browser capacity as four distinct controls. Never collapse them into one
"browser scale" number.

| Provider | Directly documented operational quantity | Correct use | Boundary that remains |
|---|---|---|---|
| Steel | Launch: 10 concurrent sessions, 60 requests/minute, 15-minute max session; Scale: 100, 600 requests/minute, 1-hour max; Enterprise: 1,000+ concurrent sessions, custom request rate, up to 24-hour max | account-plan admission and lifecycle ceilings | allowance is not observed concurrency or throughput; no public queue behavior found |
| Browserbase | active concurrency: Free 3, Developer 25, Startup 100, Scale 250+; session creations/min: Free 5, Developer 25, Startup 50, Scale 150+; either excess returns HTTP 429 and over-limit creation is dropped; project-default session duration is configurable, maximum duration is 6 hours; CDP connections close after 10 minutes without commands | model active concurrency and creation rate as separate admission controls; reject/back off on 429; configure session duration separately from CDP heartbeat/inactivity handling | 429 does not mean queued; the 10-minute CDP inactivity timeout is a connection bound, not session duration |
| Kernel | standby deletion defaults to 60 seconds and permits up to 72 hours; pool acquire long-polls and returns HTTP 204 on poll timeout | distinguish idle reclamation from waiting-for-capacity; retry a timed-out poll without counting it as running | pool wait is not running work; no public numeric concurrency allowance found |
| Hyperbrowser | session timeout is configurable per request; official code example sets `timeoutMinutes: 60` | record 60 minutes only when reproducing that configuration example; otherwise load an explicitly chosen timeout | 60 minutes is not a default, maximum, plan allowance, or observed duration; no numeric concurrency, queue, or request-rate boundary found in the reviewed guide |
| Anchor Browser | idle timeout defaults to 5 minutes after disconnect and can be disabled with `-1`; hard `max_duration` defaults to 180 minutes and has no documented upper bound | operate independent idle and hard-lifetime timers; end at the first timer reached | neither timer is task duration; no robust public concurrency/rate quantity found |

Source pages were accessed 2026-08-27. Steel's pricing page identifies a 2026-06-30 last
edit. The other reviewed pages expose mutable documentation rather than a stable release or
commit identifier, so the access date is the evidence pin.

### Admission rules for benchmark workloads

- **Allowance is configuration, not a sample.** A plan's concurrent-session allowance bounds
  how many starts fak may admit; it does not prove that many sessions were observed running.
- **Rate and concurrency use separate buckets.** Requests/minute constrains request starts;
  concurrent sessions constrains active browsers. One cannot substitute for the other.
- **Waiting is not running.** A Kernel pool long-poll remains queued/waiting until it returns a
  browser. Browserbase HTTP 429 is refusal/backpressure for either active concurrency or per-minute creation rate, not a queued session; over-limit creation is described as dropped.
- **Timeouts are lifecycle controls.** Standby/idle timeouts and maximum session lifetimes do
  not define browser-task duration. Report task duration from the task witness only.
- **Provider specifications are not populations.** These records describe account or API
  envelopes, not production traffic distributions, fleet occupancy, or achieved throughput.

## Topology and batching slice (issue #9382)

These are benchmark configurations, not production-deployment disclosures. A configured batch or concurrency value is a ceiling/control, not achieved active batch; accelerator count and TP/PP/EP do not establish replica count. QPS, tokens/s, and queries/s remain different units.

| Record | Model / precision | Accelerator topology | TP / PP / EP | Replicas | Batch / concurrency controls | Length envelope | Scenario / latency gate | Result | Software / result date |
|---|---|---|---|---|---|---|---|---|---|
| `mlperf-v51-nvidia-b200-deepseek-r1-server-2025` | DeepSeek-R1; FP4 weights, FP8 KV | 1 node; 8x B200 180GB | 8 / 1 / 8 | unstated | configured GPU batch 512; max concurrency 5,120; target 5 QPS | max input 3,140; max sequence 23,140 tokens | MLPerf Server; TTFT and TPOT early-stopping passed, numeric thresholds not exposed in selected files | 18,592.2 tokens/s | TensorRT-LLM v5.1 submission; 2025-09-04 |
| `mlperf-v51-nvidia-b200-llama31-405b-interactive-2025` | Llama 3.1 405B; FP4 weights, FP8 KV | 1 node; 8x B200 180GB | 4 / 1 / unstated | unstated | configured GPU batch 256; inflight batching; max concurrency unstated; target 1.15 QPS | max input 20,000; max sequence 22,000 tokens | MLPerf Interactive; TTFT and TPOT early-stopping passed, numeric thresholds not exposed | 751.366 tokens/s | TensorRT-LLM v5.1 submission; 2025-09-04 |
| `mlperf-v60-nvidia-b300-deepseek-r1-server-2026` | DeepSeek-R1; FP4 weights, FP8 KV | 1 node; 8x B300 270GB | 8 / 1 / 8 | unstated | configured GPU batch 640; max concurrency 10,240; target 9.5 QPS | max input 3,140; max sequence 35,908 tokens | MLPerf Server; TTFT and TPOT early-stopping passed, numeric thresholds not exposed | 42,721.4 tokens/s | TensorRT 10.14, CUDA 13.1, cuDNN 9.17, TensorRT-LLM `feat/1.2-mlpinf`, Dynamo 0.8.0 submission; 2026-03-26 |
| `mlperf-v60-amd-mi355x-llama31-405b-interactive-2026` | Llama 3.1 405B; FP4 weights | 1 node; 8x MI355X 288GB | unstated / unstated / unstated | unstated | runtime batch and max concurrency unstated; target 1.04 QPS | unstated | MLPerf Interactive; TTFT and TPOT early-stopping passed, numeric thresholds not exposed | 793.994 tokens/s | PyTorch 2.9.x, ROCm 7.0/7.1; 2026-03-26 |
| `mlperf-v60-redhat-b200-qwen3vl-235b-server-2026` | Qwen3-VL-235B-A22B; FP4 weights | 1 node; 8x B200 180GB | unstated / unstated / unstated | unstated | runtime batch and max concurrency unstated; target 5 QPS | unstated | MLPerf Server; early-stopping passed, numeric threshold not exposed | 67.8642 queries/s | vLLM + NVIDIA Dynamo + Qwen3-VL harness, RHEL 10.1; 2026-03-26 |

The two v6.0 rows with omitted topology demonstrate the intended discipline: the submission exposes accelerator count and a result, but that does not authorize reconstructing TP, replica count, or achieved batch. The Qwen3-VL result also remains in queries/s; it is not converted to tokens/s.
