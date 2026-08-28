---
title: "Frontier infrastructure contradiction matrix"
description: "Most apparent contradictions in frontier-infrastructure reporting disappear when the."
---

# Frontier infrastructure contradiction matrix

**As of:** 2026-08-26  
**Issue:** [#9306](https://github.com/anthony-chaudhary/fak/issues/9306)  
**Source of truth:** [`index.json`](index.json)

## Result first

Most apparent contradictions in frontier-infrastructure reporting disappear when the
**denominator, lifecycle state, boundary, time window, and evidence class** are made
explicit. The remainder are real disagreements or unverified claims; they must stay
open rather than being averaged into false precision.

This matrix does not choose the largest number. It records what each number actually
measures and the comparison that would be required to reconcile it. Source labels
retain the corpus evidence classes: production measurement/observation, benchmark or
synthetic result, official statement, vendor claim, analyst/reported estimate,
reported observation, and rumor.

## Reading rule

Normalize every comparison to this tuple before using it:

```text
(entity, geography, time window, lifecycle state, physical boundary,
 hardware/model mix, workload/quality constraint, accounting basis,
 evidence class, source independence)
```

A blank or incompatible tuple is not evidence that one side is wrong. It is evidence
that the claims cannot yet be compared.

## Matrix

| Cluster | Claims that can look contradictory | Normalization that resolves or isolates the conflict | Current verdict | fak consequence |
|---|---|---|---|---|
| **Announced, contracted, built, online, healthy, schedulable, and useful capacity** | OpenAI's Stargate launch announced an intended **$500B over four years**, beginning with **$100B**; later OpenAI releases described sites and partner commitments. Datacenter pipeline sources separately count projects as planned, under construction, delayed, or operational. | Track dollars and infrastructure through distinct lifecycle states: announced → financed/contracted → permitted/interconnected → built → hardware installed → accepted/healthy → schedulable → active → quality-constrained goodput. Never convert a spend plan or site announcement directly into tokens/s. | **Not contradictory after normalization.** The corpus still lacks a complete site-by-site conversion ledger from announcement to useful serving capacity. | Capacity admission and benchmarks must use witnessed active resources, not press-release totals. Preserve lifecycle state in every receipt. |
| **Physical accelerators versus “H100-equivalent” capacity** | xAI descriptions use physical H100 counts for Colossus/Grok infrastructure. NVIDIA and cloud materials also use mixed generations, platform systems, or performance-equivalent language. | Record physical SKU count, precision, sparsity, clock/power envelope, interconnect, memory capacity/bandwidth, model, quality target, and equivalence formula. “Equivalent” is a computed performance unit, not a device census. | **Potentially incomparable, not additive.** The corpus does not contain a neutral conversion table across H100, H200, Blackwell, TPU, Trainium, Groq, and other accelerators. | Keep native receipts hardware-specific. Do not normalize fleet size to H100 equivalents unless the matched workload and formula are published. |
| **IT load, facility load, grid connection, and generation capacity** | IEA electricity forecasts, utility/interconnection reports, campus announcements, and vendor power architectures cite MW or GW at different boundaries. Some numbers describe IT load; others describe total facility demand, utility supply, interconnection requests, or generation nameplate. | Label MW/GW as IT critical load, total facility load, contracted utility service, requested interconnection, deliverable grid capacity, or generation nameplate. Add PUE, utilization, redundancy, curtailment, commissioning date, and geography. | **Usually denominator mismatch.** Grid-request and generation numbers do not prove continuous IT power. | Hardware capacity is not admitted until power at the relevant boundary is deliverable. Include PUE and derating when translating facility power into compute. |
| **Peak FLOPS or bandwidth versus quality-constrained goodput** | Accelerator and system releases advertise peak compute, memory bandwidth, interconnect, or benchmark maxima. Serving papers report latency/throughput improvements under specific traces, models, and baselines. | Compare end-to-end accepted tokens or completed requests per wall-clock and dollar under matched model, quantization, quality, context mix, SLO, failure/retry accounting, warmup, and utilization. Count transfer, orchestration, verification, and recovery overhead. | **No direct contradiction is established.** Peak component metrics and serving goodput answer different questions; vendor maxima remain claims until reproduced in a matched envelope. | fak performance claims must stay net-true and quality-constrained. A faster kernel or link is not a faster service by itself. |
| **Users or adoption versus requests, tokens, concurrency, and tenant concentration** | OpenAI and Anthropic publish adoption/user-scale statements. The GitHub Copilot production trace reports **3.2M users**, **13.5M sessions**, **95.1M turns**, **760.5M LLM calls**, **774.7M tool calls**, and **44.9T prompt tokens** for a specific coding-agent population. | Separate registered, weekly active, daily active, paid-seat, API-account, and measured-trace users. Preserve application, geography, interval, tenant weights, session/turn/call expansion, prompt/completion tokens, and concurrency. | **Not contradictory.** Adoption counts do not identify workload shape. One coding-agent trace cannot be projected to all frontier-lab traffic. | Workload models need tenant-weighted request/token/session distributions, not a single user total. Benchmark both typical and dominant-tenant bursts. |
| **Capex, cash PP&E, finance leases, obligations, and backlog** | Alphabet, Microsoft, Meta, and Amazon disclosures use different fiscal calendars and accounting lines: capital expenditures, cash purchases of property/equipment, finance leases, and future commitments are not interchangeable. AI and non-AI infrastructure are often combined. | Preserve issuer definition, fiscal period, cash versus accrual basis, PP&E versus lease treatment, gross versus net presentation, AI share, and whether a number is recognized spend, guidance, purchase obligation, minimum lease payment, or remaining contract value. | **Cross-company totals are not yet normalized.** The filings ledger is useful within issuer but incomplete for an industry sum. | Do not infer accelerator purchases or serving capacity from headline capex. Keep source accounting fields in economic models. |
| **“30–50%” or “half of 2026 U.S. capacity delayed”** | Data Center Watch reporting characterized a large share of the U.S. pipeline as delayed. SemiAnalysis published a rebuttal arguing that the “half delayed” headline depended on project definitions and denominator choices. | Reproduce both project universes; deduplicate sites; distinguish announced, proposed, under construction, powered, and operational MW; fix the observation date; record delay threshold and whether slips are capacity-weighted or project-count-weighted. | **Genuine unresolved methodological disagreement.** The corpus records the claim and rebuttal but does not yet reproduce either underlying dataset end to end. | Model schedule risk as a range by lifecycle state. Do not use one disputed percentage as a universal haircut. |
| **Maximum context window versus production prevalence** | Model releases advertise very large maximum contexts, including MiniMax's **1M-token** M3 context. Production traces show broad length distributions and compaction; the Copilot trace found only **7.8%** of sessions compacted but those sessions held **44%** of tokens. | Separate supported maximum, tested maximum, billed maximum, observed prompt distribution, active KV footprint, prefix reuse, compaction policy, and SLO. Weight by token-time and memory occupancy rather than request count alone. | **Not contradictory.** Capability ceilings do not establish that production traffic commonly reaches them. Prevalence outside the captured workloads remains unverified. | Provision long-context paths explicitly, but do not make maximum context the fleet-wide median. Benchmark memory tails and compaction separately. |
| **Vendor benchmark maximum versus neutral production evidence** | Vendor system releases and product pages report favorable throughput/latency or scale results. Research papers and production traces use different hardware, software, traces, queueing regimes, and quality constraints. | Require source class, code/artifact availability, baseline tuning parity, hardware availability, workload trace, prompt/output distributions, batching policy, cache state, SLO, and independent reproduction. | **Evidence-strength conflict, not necessarily numeric contradiction.** Current corpus mixes claims and measurements deliberately; neutral production coverage is much thinner. | Route architectural decisions toward production measurements first, then neutral benchmarks, then vendor claims with explicit discount and reproduction work. |
| **Heavy tail versus Zipf, lognormal, Pareto, or MMPP labels** | BurstGPT and other workload studies model arrival processes or report burstiness; output-length work reports skewness **3.10**, mean CV **1.09**, top-decile share **35.7%**, P90/P50 **4.62**, and P99/P50 **10.77**. Cache studies use workload-specific popularity assumptions. These observations are sometimes summarized as “Zipfian.” | Identify the random variable: tenant volume, request arrival, prefix popularity, prompt length, output length, session size, or reuse distance. Preserve fitted family, parameter, estimator, goodness of fit, interval, and drift. A heavy tail in one variable does not identify the family of another. | **Universal Zipf claim rejected.** The corpus supports heterogeneous, heavy-tailed, bursty, and nonstationary behavior, but does not establish one global distribution family. | Parameterize per workload and re-fit over time. Never hard-code one Zipf exponent as the production prior. |
| **Funding or backlog versus revenue, profitability, and deployed capacity** | Groq, xAI, CoreWeave, Nebius, and other infrastructure/startup sources cite funding rounds, valuations, customer commitments, backlog, or planned deployments. | Distinguish signed financing, cash received, debt, valuation, non-cancellable contract value, remaining performance obligations, recognized revenue, gross margin, profitable operation, capex paid, and accepted online capacity. Check customer concentration and termination rights. | **Not interchangeable.** Funding and backlog establish access to capital or contracted demand, not profitable delivery or healthy compute. Failure/cancellation evidence remains sparse. | Treat financing and backlog as market signals only. Capacity planning requires delivery and operations evidence. |
| **Rumor repetition versus independent corroboration** | Three current entries are explicitly labeled rumors. Reposts can make one anonymous-source report appear multiply sourced; later official announcements may confirm only part of the original claim. | Track the original source, named versus anonymous evidence, publication lineage, truly independent corroborators, exact claim fragments, expiry, and later confirmation/refutation. Do not count syndication or circular citation as corroboration. | **Unverified by default.** The present chronology records provenance but does not yet provide a complete resolution graph for all rumors. | Rumors may create watch items, never default capacity, routing, roadmap, or benchmark assumptions. Expire unresolved claims. |

## Comparison templates

### Capacity receipt

```text
entity / site / region
announcement date / target date / observation date
announced / contracted / permitted / grid-ready / built / installed /
accepted / healthy / schedulable / active
physical accelerator SKUs and count
IT MW / facility MW / deliverable grid MW / generation MW
model + precision + quality + SLO
measured goodput and accounting boundary
source class + confidence
```

### Workload-distribution receipt

```text
population / application / tenant weighting / geography / time window
random variable (arrival, prompt, output, prefix, session, tenant volume, ...)
raw quantiles / fitted family and parameters / goodness of fit
diurnal + weekly seasonality / burst duration / drift interval
retries, failures, speculative acceptance, tool-call expansion
cache and compaction policy
source class + confidence
```

### Financial receipt

```text
issuer / fiscal period / currency
reported accounting line and exact definition
cash / accrual / finance lease / operating lease / obligation / backlog
AI share stated? yes/no
recognized / guided / contracted / contingent
customer and supplier concentration
source class + confidence
```

## Open reconciliation work

1. Reproduce the Data Center Watch and SemiAnalysis delay denominators from
   project-level data.
2. Build a physical-SKU ledger and refuse unsupported “H100-equivalent” joins.
3. Extend filings normalization beyond four U.S. hyperscalers and include leases,
   obligations, depreciation, AI share, and customer concentration.
4. Extract fitted distributions and goodness-of-fit evidence from every workload
   paper; retain “unknown” where papers provide only qualitative heavy-tail claims.
5. Add site lifecycle and rumor-resolution histories so claims can move from
   announced or rumored to confirmed, refuted, expired, or delivered.

Until those checks exist, the matrix is a guard against conflation, not a claim that
all disagreements have been resolved.
