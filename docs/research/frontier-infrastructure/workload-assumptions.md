# Workload and cluster assumption registry

**As of:** 2026-08-26. This registry turns the source ledger into assumptions that can be
accepted, rejected, or parameterized in fak benchmarks. It is not a substitute for the
per-source provenance in [`index.json`](index.json).

## Distribution assumptions

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
