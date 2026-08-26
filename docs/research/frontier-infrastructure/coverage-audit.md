# Frontier infrastructure coverage audit

**Snapshot:** 2026-08-26  
**Issue:** [#9306](https://github.com/anthony-chaudhary/fak/issues/9306)  
**Authority:** [`index.json`](index.json)

## Verdict

The corpus has a strong initial spine but is **incomplete**. It covers the major
architectural seams and several high-value production traces; it does not establish
entity-complete global coverage, complete production demand distributions, normalized
financial or site lifecycles, or a resolved rumor history. “Exhaustive” remains an
operating method—explicit taxonomy, dated evidence, and visible gaps—not a claim that
the open web has a finite or fully enumerated boundary.

## Status vocabulary

| Status | Meaning |
|---|---|
| **Complete** | The declared bounded requirement is enumerated, source-linked, normalized, and internally checked for this snapshot. It does not mean no future source can exist. |
| **Partial** | Useful evidence exists, but named entities, regions, variables, dates, lifecycle states, or source classes remain absent. |
| **Missing** | No entry currently answers the requirement directly. |
| **Unverified** | A claim is present, but its denominator, original dataset, independence, or later outcome is not strong enough to rely on. |

No broad topical slice below qualifies as complete.

## Machine-derived inventory

The following counts are derived from `index.json`, not hand-maintained estimates.

| Measure | Current value | Audit note |
|---|---:|---|
| Entries | **123** | Every entry has an ID, entity, category, evidence class, confidence, `published_at`, `event_at`, source title, and source URL. |
| Unique source URLs | **118** | Repeated URLs represent distinct claims/events extracted from the same source; they are not independent corroboration. |
| Distinct entity labels | **95** | Joint labels such as “OpenAI / Oracle / SoftBank” are one ledger label, not three independently audited entities. |
| Categories | **11** | `accelerator_platform` 3; `ai_cloud` 5; `datacenter_physical` 17; `frontier_lab` 41; `hyperscaler` 16; `market_signal` 9; `serving_system` 11; `standard` 1; `supply_chain` 9; `workload_model` 3; `workload_trace` 8. |
| Evidence classes used | **10** | `official_statement` 61; `vendor_claim` 17; `reported_observation` 9; `production_measurement` 8; `production_observation` 8; `benchmark_measurement` 7; `analyst_estimate` 5; `synthetic_experiment` 4; `rumor` 3; `reported_estimate` 1. The allowed `inference` class currently has zero entries. |
| Confidence labels | **4** | `high` 86; `medium_high` 28; `medium` 6; `low` 3. Confidence describes evidentiary strength, not business likelihood. |
| Date fields | **123/123 published; 123/123 event** | Presence is complete. Date precision and continuing-event semantics are not separately encoded. |
| Explicit rumors | **3** | All three are low-confidence and carry rumor metadata; complete resolution histories are absent. |

### Structural checks

| Check | Status | Evidence |
|---|---|---|
| Required fields present | **Complete** | All 123 entries contain the schema's required fields. |
| JSON parseability | **Complete** | `python3 -m json.tool` is the local validation command. |
| Unique-entry semantics | **Partial** | IDs are intended to be unique and URLs are counted, but no committed schema/link checker enforces the contract yet. |
| Source-class separation | **Complete for current entries** | The ledger keeps production, benchmark/synthetic, official, vendor, analyst/reported, and rumor classes distinct. |
| Claim contradiction handling | **Partial** | [`contradiction-matrix.md`](contradiction-matrix.md) defines normalization; project-level reproductions and automated clustering remain open. |
| Refreshability | **Partial** | [`refresh-protocol.md`](refresh-protocol.md) defines manual refresh; scheduled validation and link checking are absent. |

## Requirement-by-requirement coverage

### 1. Frontier labs and regions — **Partial**

**Present:** OpenAI, Anthropic, xAI, Google DeepMind, Meta, Amazon Nova, Apple,
Cohere, Ai2, DeepSeek, Alibaba/Qwen, Moonshot, MiniMax, ByteDance Seed, Baidu,
Tencent, Z.ai/Zhipu, Mistral, AI21, TII, Sarvam, Sakana, NAVER Cloud, and Sea AI Lab.
The census now spans the U.S./Canada, China, Europe, Israel, the Middle East, India,
Japan, Korea, and Southeast Asia and records model/serving statements without treating
plans as delivered capacity.

**Missing or shallow:** Microsoft AI/Phi, NVIDIA Nemotron, Huawei/Pangu/Ascend,
01.AI, StepFun, Baichuan, Shanghai AI Lab, SenseTime, G42/Inception/MBZUAI, Saudi
programs, LG/Samsung/SKT/Kakao, NTT/Preferred Networks/Fujitsu/SoftBank, AI Singapore,
Grab/GoTo/SCB10X/VinAI, Europe beyond Mistral/DeepMind, and many private labs. Most
checked labs still lack physical serving fleets, traffic distributions, and lifecycle
resolution. Sovereign programs and export-control interactions are not complete.

**Proof needed for complete:** a declared entity/region universe, at least one current
primary source per entity, model and infrastructure lifecycle fields, and dated checks
for launches, cancellations, partnerships, and regional constraints.

### 2. Hyperscalers, clouds, and AI clouds — **Partial**

**Present:** AWS, Google/Alphabet, Microsoft/Azure, Meta, Alibaba Cloud, CoreWeave,
Nebius, and selected partner/cloud deployments. Official filings and earnings evidence for Alphabet, Microsoft, Meta, Amazon, Oracle,
CoreWeave, Alibaba, Nebius, and Baidu are extracted in [`filings-ledger.md`](filings-ledger.md).

**Missing or shallow:** IBM, Tencent, and sovereign-cloud financial normalization;
full annual-report normalization remains incomplete for Oracle, CoreWeave, Nebius, Alibaba, and Baidu; customer-supplied hardware; partner
prepayments; finance and operating leases; purchase obligations; depreciation;
backlog quality; customer concentration; AI versus non-AI capex.

**Proof needed for complete:** issuer-by-issuer filings with identical field definitions,
fiscal-calendar normalization, lifecycle linkage from obligation to installed capacity,
and coverage of non-U.S. and sovereign clouds.

### 3. Datacenter power, grid, cooling, water, construction, and permitting — **Partial**

**Present:** IEA demand/grid evidence, U.S. project-pipeline and delay reporting,
selected utilities/site announcements, NVIDIA DSX power architecture, GE Vernova/Crusoe ordered generation, a phased Siemens/Start Campus site record, and a
supply-chain lifecycle framework. Power is treated as a gating resource rather than an
afterthought.

**Missing or shallow:** named operator/site census; utility interconnection queues;
deliverable versus requested MW; turbines, transformers, switchgear, substations;
PUE and curtailment; water source/consumption; cooling-vendor and heat-rejection data;
construction labor; permits; community opposition; cancellations; commissioning and
acceptance dates. The disputed “half delayed” claim remains unverified at project level.

**Proof needed for complete:** site-level records from announcement through operation,
with power-boundary labels, vendors, dates, MW, water/cooling method, permit status, and
subsequent delay/cancellation resolution.

### 4. HBM, advanced packaging, optics, network, and storage — **Partial**

**Present:** selected NVIDIA/HBM, SK hynix HBM4/HBM4E, TSMC CoWoS, optical-interconnect, networking, and platform
announcements; architectural recognition that accelerator availability alone does not
set cluster capacity.

**Missing or shallow:** supplier-by-supplier HBM output and allocation, CoWoS/advanced
packaging capacity, substrates, optics/transceivers, switch silicon, cables, NICs,
object/block storage, checkpoint paths, lead times, yields, shipment versus installed
states, and China/regional supply chains. Electrical equipment is similarly sparse.

**Proof needed for complete:** component/vendor ledgers with physical units, time
windows, order/shipment/install states, dependencies, and independent production or
shipment evidence rather than only vendor roadmaps.

### 5. Batching and scheduling — **Partial**

**Present:** serving-system papers and releases cover scheduling, continuous/dynamic
batching assumptions, queueing, and several workload traces. The corpus recognizes that
batch opportunities depend on arrival, lengths, SLOs, hardware, and tenant policy.

**Missing or shallow:** comparable production distributions for batch size, queue wait,
SLO class, cancellation, priority, fairness, admission, and cross-tenant interference;
policy prevalence by lab/cloud; scheduler behavior during failures and regional bursts.

**Proof needed for complete:** production traces or operator measurements with batch and
queue fields, stratified by model, tenant, hardware, region, and SLO.

### 6. Prefill/decode disaggregation — **Partial**

**Present:** llm-d/Google, NVIDIA Dynamo, FineServe, and related system evidence make
prefill/decode separation, topology, and KV transfer explicit product surfaces.

**Missing or shallow:** installed production share, transfer sizes, topology-specific
break-even curves, failure domains, decode imbalance, WAN/regional use, and matched
end-to-end comparisons that count orchestration and transfer overhead.

**Proof needed for complete:** neutral production evidence over representative prompt and
output distributions, with quality, SLO, utilization, transfer, and failure accounting.

### 7. KV cache, prefix reuse, routing, and offload — **Partial**

**Present:** Mnemosyne, CacheRoute, Tair KVCache/HiSim, multi-tenant admission studies,
robust cache work, and the Copilot trace's approximately **90% average prefix-cached token
share within sessions**. The corpus distinguishes benchmark/synthetic results from
production traces.

**Missing or shallow:** cross-tenant production prefix-popularity fits, reuse-distance
and object-lifetime distributions, cache-hit opportunity by application, privacy
boundaries, invalidation, fragmentation, offload bandwidth, routing overhead, and drift.

**Proof needed for complete:** raw production distributions with tenant weighting and
cache policy, plus end-to-end hit/goodput measurements under realistic churn.

### 8. Autoscaling, placement, resilience, and fairness — **Partial**

**Present:** cluster-scale reliability research, routing/scheduling papers, and system
releases expose topology-aware placement and failure/retry costs. The Copilot trace
reports **9% of turns with tool failure** and retry loops reaching **4× compute** in the
captured population.

**Missing or shallow:** production cold-start and scale-up times, capacity headroom,
regional failover, heterogeneous accelerator placement, maintenance/health attrition,
preemption, tenant fairness, quota enforcement, noisy-neighbor distributions, and
installed-to-schedulable-to-goodput conversion.

**Proof needed for complete:** operator traces linking demand, scaling decisions, health,
placement, retries, SLOs, and per-tenant outcomes over failures and seasonal peaks.

### 9. User, tenant, geography, and seasonality distributions — **Partial to missing**

**Present:** the GitHub Copilot production corpus provides **3.2M users**, **13.5M
sessions**, **95.1M turns**, **760.5M LLM calls**, **774.7M tool calls**, **44.9T prompt
tokens**, **39.3B completion tokens**, five user archetypes, and compaction/failure
statistics. Other trace papers provide request/length/burst observations.

**Present but incomplete:** ServeGen identifies 29 dynamic top clients among 2,412
profiled clients; FineServe quantifies second-scale burst concentration; Chutes tracks
one-year model/user evolution; Aliyun, Chutes, and Copilot expose distinct cache-reuse
shapes. These are provider-specific observations, not universal tenant laws.

**Missing:** exact tenant traffic shares; geography; timezone; diurnal and weekly
coefficients; launch/event burst parameters; paid/API/consumer segmentation;
reasoning-mode selection; retry distributions beyond coding agents; speculative-token
acceptance; cancellation; and cache opportunity by user cohort. User/adoption totals
from labs do not fill these gaps.

**Proof needed for complete:** anonymized multi-product production traces with explicit
population, interval, geography, tenant weighting, concurrency, and token accounting.

### 10. Distribution-family and parameter claims — **Partial**

**Present:** output-length evidence reports skewness **3.10**, mean coefficient of
variation **1.09**, CV above 1 for **78.6%** of examined workloads, top-decile share
**35.7%**, P90/P50 **4.62**, and P99/P50 **10.77**. BurstGPT and other studies cover
burstiness/arrival modeling. [`workload-parameters.md`](workload-parameters.md) records
exact parameters available from selected sources.

**Unverified:** any universal Zipf law. Current evidence supports category-dependent,
heavy-tailed, bursty, and nonstationary behavior. It does not prove one Zipf exponent,
lognormal fit, Pareto fit, or MMPP for users, tenants, prompts, prefixes, outputs, and
arrivals collectively.

**Missing:** fitted parameters, estimator, goodness-of-fit, confidence interval, sample
population, and drift window for every cited workload paper and every random variable.

**Proof needed for complete:** variable-specific parameter extraction and reproducible
fit checks; “unknown” must remain a valid result.

### 11. Startups, launches, releases, and partnerships — **Partial**

**Present:** inference/serving startups, AI clouds, accelerator partnerships, financing,
product launches, and selected acquisitions/partnerships appear in
[`startups-landscape.md`](startups-landscape.md) and
[`market-chronology.md`](market-chronology.md).

**Missing or biased:** the sample is launch- and funding-heavy. Failures, shutdowns,
acquihires, distressed financing, deployment cancellations, missed roadmaps, customer
concentration, revenue/profitability, and category-complete coverage across routers,
KV systems, power, cooling, networking, and modular datacenters are sparse.

**Proof needed for complete:** a declared startup universe with founded/launch/funding/
deployment/acquisition/failure states and periodic resolution checks.

### 12. Rumors and resolution history — **Unverified**

**Present:** **3** rumor entries are explicitly labeled with origin/provenance fields and
kept out of factual capacity totals.

**Missing:** complete original-source lineage, circular-republication detection,
independent corroboration graph, claim-fragment matching, expiry dates, and later
confirmed/refuted/partially-confirmed outcomes.

**Proof needed for complete:** a rumor state machine and dated resolution history.
Until then, rumors are watch items only.

### 13. News and market chronology — **Partial**

**Present:** completed events, future announcements, partnerships, funding, and rumors
are separated in [`market-chronology.md`](market-chronology.md). Publication and event
dates are present on every current entry.

**Missing:** systematic global media/source watch, change detection, corrections,
archived copies, cancellation/failure follow-up, and resolution of future plans after
their target date.

**Proof needed for complete:** a refresh cadence with source inventories, link/archive
health, plan-expiry checks, and additive snapshots rather than silent rewrites.

### 14. Standards, regulation, export controls, and sovereign AI — **Missing to partial**

**Present:** one standards entry and selected policy/market context.

**Missing:** comprehensive export-control chronology, sanctions, accelerator and
interconnect restrictions, energy/water/permitting regulation, sovereign compute
programs, procurement, data-residency constraints, and their effects on regional supply
and workload placement.

**Proof needed for complete:** jurisdiction-by-jurisdiction primary sources with effective
dates, affected hardware/services, implementation status, and later amendments.

## Coverage by source strength

| Evidence tier | Current condition | Consequence |
|---|---|---|
| Production measurement/observation | **16 entries; valuable but narrow** | Strongest demand evidence is concentrated in coding-agent and selected serving workloads. Do not universalize it. |
| Benchmark/synthetic | **11 entries** | Useful for mechanism and break-even hypotheses; not proof of installed production prevalence. |
| Official statements | **61 entries** | Strong for what an entity said or filed, not for future delivery or neutral performance. |
| Vendor claims | **17 entries** | Retain exact envelope and reproduce before using as a fak gain claim. |
| Analyst/reported evidence | **15 entries** | Useful for market/site visibility; denominators and original datasets require checking. |
| Rumor | **3 entries** | Watch-only until independently corroborated or resolved. |

## Preserved explicit gaps

The machine-readable `coverage.explicit_gaps` remains authoritative. In plain language,
the open work is:

1. entity-complete global frontier-lab and sovereign-program coverage;
2. direct production demand distributions, especially tenants, geography, seasonality,
   retries, sessions, tools, prefix popularity, and cache opportunity;
3. exact distribution/quantile/CV/time-window extraction from every workload source;
4. normalized hyperscaler/cloud filings and contractual obligations;
5. named physical-infrastructure and component lifecycle ledgers;
6. startup failures, acquisitions, cancellations, and delivered economics—not launches
   alone;
7. China/export-control/regional supply-chain coverage;
8. rumor provenance and resolution graphs; and
9. committed schema/link/contradiction validation and scheduled refresh.

## Next audit actions

1. Complete the frontier-lab census by region, beginning with the unchecked high-scale
   China and hyperscaler labs.
2. Expand filings to Oracle, CoreWeave, Nebius, IBM, Alibaba, Tencent, and Baidu with
   normalized obligations and leases.
3. Build a named site/vendor lifecycle ledger for grid, power equipment, cooling/water,
   HBM/packaging, optics/network, storage, construction, and permitting.
4. Extract production-distribution parameters and explicitly test—not assume—Zipf,
   lognormal, Pareto, and MMPP fits per variable.
5. Add failure/acquisition/cancellation and rumor-resolution histories.
6. Only after the schema stabilizes, consider a committed Go validation leaf for field,
   uniqueness, link, lifecycle, and coverage checks.
