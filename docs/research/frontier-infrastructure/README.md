# Frontier infrastructure and workload expectations index

**Status:** initial end-to-end spine, incomplete by design. **As of:** 2026-08-27. **Tracker:** #9269.

This index records what frontier labs, hyperscalers, AI clouds, datacenter operators,
accelerator vendors, serving-system builders, researchers, reporters, and market actors
say or measure about the infrastructure and workloads behind frontier AI. It exists to
stop fak from optimizing against an invented cluster, a stationary synthetic workload,
or an unlabeled market rumor.

The machine-readable source of truth is [`index.json`](index.json). The derived
views are:

- [`coverage-audit.md`](coverage-audit.md) — requirement-by-requirement completeness audit, exact corpus counts, and proof still needed;
- [`policy-standards-ledger.md`](policy-standards-ledger.md) — export controls, sovereign compute, datacenter policy, regulation, and standards lifecycle;
- [`contradiction-matrix.md`](contradiction-matrix.md) — denominator, lifecycle, accounting, distribution, and rumor conflicts normalized before comparison;
- [`slices/frontier-lab-census.md`](slices/frontier-lab-census.md) — compact lab-by-lab census of the current source set;
- [`workload-assumptions.md`](workload-assumptions.md) — architecture and benchmark priors;
- [`workload-parameters.md`](workload-parameters.md) — exact production-trace distributions and benchmark synthesis;
- [`filings-ledger.md`](filings-ledger.md) — normalized hyperscaler capex and accounting-boundary evidence;
- [`supply-chain-ledger.md`](supply-chain-ledger.md) — power, grid, cooling, water, construction, HBM, packaging, optics, networking, and electrical bottlenecks;
- [`market-chronology.md`](market-chronology.md) — completed events, future announcements, partnerships, funding, and rumors;
- [`startups-landscape.md`](startups-landscape.md) — startup and alternative-infrastructure map;
- [`refresh-protocol.md`](refresh-protocol.md) — how to extend the ledger without laundering claims.

Empty or missing slices are coverage debt, not evidence that a category has no activity.

## Latest slice: Chinese platform envelopes (#9362)

Six bounded records add Baichuan 2 training, iFLYTEK ecosystem denominators, Meituan
LongCat training/inference, production-scale asynchronous RL, stateful generative-
recommendation caching, and one-week recommendation-training sequences. The corpus now
contains **213 entries**, **208 unique source URLs**, and **176 entity labels**. Developer
teams, applications, developers, API growth, agents, sequences, users, requests, and
tokens remain separate denominators; internal/vendor maxima are not universal.

## Latest slice: realized failures and cancellations (#9363)

This slice reconciles the existing Untether AI bankruptcy record and adds Builder.ai insolvency, a bare-metal cloud wind-down,
semiconductor-project cancellation and delay, one named data-center withdrawal, and a
25-project lower-bound U.S. cancellation cohort. The corpus now contains **218 entries**,
**213 unique source URLs**, and **181 entity labels**. Bankruptcy, insolvency, wind-down,
impairment, withdrawal, cancellation, delay, operating shutdown, and lost live MW remain
distinct states.

## Latest slice: named grid and utility outcomes (#9364)

Four records connect large-load policy to service reality: FERC’s rejection of the
Susquehanna 300-to-480 MW co-location amendment, AEP Ohio’s collateralized data-center
tariff, Dominion’s GS-5 large-load class, and LPSC approval of 2,262 MW plus 500-kV
transmission for Meta’s named Louisiana project. The corpus now contains **222 entries**,
**217 unique source URLs**, and **185 entity labels**. Tariff, contract, collateral, queue,
regulatory approval, construction, energization, actual load, and live IT MW remain distinct.

## Latest slice: component shipment receipts (#9365)

Five primary-source records add Micron HBM3E volume production, Micron HBM4 high-volume
shipments, Broadcom Tomahawk 6 production-volume shipments, TSMC CoWoS-L production and
customer qualification, and TSMC Arizona N4 high-volume wafer production. The corpus now
contains **227 entries**, **222 unique source URLs**, and **190 entity labels**. Samples,
qualification, volume production, shipped components, assembled systems, deployed clusters,
and useful goodput remain separate lifecycle states.

## Latest slice: speculative acceptance envelopes (#9366)

Five dedicated records add foundational draft/verify sampling, SpecInfer token trees, Medusa
decoding heads, EAGLE feature speculation, and MagicDec long-context batching. The corpus
now contains **232 entries**, **227 unique source URLs**, and **195 entity labels**. Drafted,
verified, accepted, rejected, fallback, and emitted tokens remain separate work; benchmark
speedups are task/model/hardware/load envelopes, not stackable production multipliers.

## Latest slice: non-coding agent workloads (#9367)

Five benchmark families add browser, enterprise, customer-service API, general-assistant, and
desktop-computer workloads. The corpus now contains **237 entries**, **232 unique source
URLs**, and **200 entity labels**. Task counts, generated instances, websites, tools, capability
tags, observations, actions, turns, users, sessions, and production arrivals remain separate
denominators; benchmark success is not a production workload distribution.

## Latest slice: public conversation populations (#9370)

Four direct population records add LMSYS-Chat-1M, WildChat, Chatbot Arena preferences, and
OpenAssistant conversation trees. The corpus now contains **241 entries**, **236 unique source
URLs**, and **204 entity labels**. Conversations, messages/turns, trees, paths, votes, users or
anonymized IDs, languages, countries, timestamps, model appearances, and production requests
remain separate denominators; public and crowdsourced datasets are not provider-wide traffic.

## Result first

The initial evidence already rejects several convenient defaults:

1. **The datacenter is becoming the system boundary.** Public plans and product designs
   are expressed in racks, pods, sites, multi-site fleets, hundreds of megawatts, and
   gigawatts—not one eight-GPU server.
2. **Power and delivery time are first-class constraints.** Chip supply is not enough;
   grid connection, firm generation, cooling, permitting, water, construction, and local
   consent can determine when nominal capacity becomes usable.
3. **Serving fleets are heterogeneous.** Frontier providers publicly describe mixes of
   NVIDIA GPUs, TPUs, Trainium, custom silicon, multiple clouds, regions, and product
   channels. A receipt that names only the model is operationally incomplete.
4. **Inference is splitting into phases and state tiers.** Prefill/decode disaggregation,
   KV-aware routing, cache offload, multi-tier memory, long-context specialization, and
   traffic-aware scaling are moving from papers into vendor and open-source platforms.
5. **One stationary request distribution is not credible.** Production-trace work finds
   client-specific, modality-specific, reasoning-specific, bursty, nonstationary arrival
   and token behavior. “Poisson + fixed input/output length” is a test fixture, not a
   production claim.
6. **Nameplate capacity is not useful capacity.** Plans, contracted gigawatts, installed
   chips, healthy schedulable accelerators, utilization, cluster goodput, SLO attainment,
   and accepted-token output are different quantities.
7. **The public record is rich on supply and poor on demand shape.** Providers disclose
   capital, chip counts, regions, and power more often than tenant skew, prefix popularity,
   token distributions, cache-hit opportunity, concurrency, geography, or retries. Those
   missing distributions are high-priority unknowns for fak.

## Evidence contract

Every entry must include:

- publication date and event/as-of date separately;
- source kind and evidence class;
- confidence;
- an original short summary instead of a copied article;
- quantified fields when the source supports them;
- assumptions exposed by the source;
- contradictions or limits;
- concrete relevance to fak;
- a rumor object, even when `is_rumor` is false.

Evidence classes are intentionally not interchangeable:

| Class | What it can prove |
|---|---|
| `production_measurement` | Behavior measured on a production population, within its disclosed sample limits. |
| `production_observation` | Operational behavior disclosed without a complete measurement dataset. |
| `benchmark_measurement` | Performance inside a stated experimental envelope. |
| `synthetic_experiment` | Behavior under a generated model; never proof of production prevalence. |
| `official_statement` | What an organization says it did, plans, or contracted. |
| `vendor_claim` | A product or performance assertion needing matched independent validation. |
| `analyst_estimate` / `reported_estimate` | A third party's bounded estimate, with methodology risk. |
| `reported_observation` | Credible reporting of events or constraints, not a controlled measurement. |
| `inference` | A conclusion drawn from cited evidence and labeled as such. |
| `rumor` | Unconfirmed information with origin, corroboration, and confidence recorded. |

A plan is not delivered capacity. Peak FLOPS are not goodput. Registered developers are
not active users. A benchmark histogram is not a production distribution. A repeated
rumor is not independently corroborated merely because many sites copied one origin.

## Taxonomy

### Actors

- frontier labs and model/API providers;
- hyperscalers and enterprise clouds;
- neoclouds, inference clouds, sovereign clouds, and GPU marketplaces;
- accelerator, CPU, memory, networking, optics, storage, and cooling vendors;
- datacenter developers, colocation operators, utilities, generators, and financiers;
- schedulers, serving engines, orchestration stacks, and standards bodies;
- regulators, local governments, communities, analysts, and reporters.

### Expectations and assumptions

- training versus inference share and their hardware overlap;
- cluster and failure-domain shape: chip, host, rack, pod, site, region, fleet;
- tensor/pipeline/expert/data/sequence/KV parallelism;
- continuous batching, batch limits, admission, fairness, preemption, and SLOs;
- prefill/decode disaggregation and KV transfer cost;
- prompt/prefix popularity, cacheability, and eviction distributions;
- request arrivals, bursts, seasonality, geography, tenant concentration, and Zipf-like skew;
- input/output tokens, reasoning length, context length, modality, sessions, tools, and retries;
- accelerator mix, networking, storage, memory hierarchy, utilization, and goodput;
- capex, opex, price, contracts, idle risk, procurement, lead time, and stranded capacity;
- power, cooling, water, land, permitting, labor, supply chain, and social license;
- roadmap, release, partnership, acquisition, cancellation, failure, leak, and rumor signals.

## What fak should assume now

Until stronger evidence exists, use these as conservative design defaults—not universal
facts:

- model the fleet as heterogeneous and topology-aware;
- model capacity as a lifecycle, not one number;
- report goodput, SLO attainment, quality, retries, and energy/cost per accepted token;
- compose workload generators per tenant/client and include bursts/regime shifts;
- keep measured traces separate from synthetic assumptions;
- treat prefix reuse and long context as distributions whose prevalence must be measured;
- price cache transfer, routing, verification, recovery, and control-plane overhead;
- retain jurisdiction, region, power, and failure domain in execution receipts;
- require rumor provenance and corroboration before it influences a roadmap.

## Coverage ledger

The current spine contains **207 dated entries**, **202 unique source URLs**, and
**170 distinct entity labels** across **13 categories** spanning frontier labs,
hyperscalers, AI clouds,
datacenter supply, accelerators, serving systems, workload traces, market signals,
and **2 explicit rumors**. FineServe, ServeGen, the one-year Chutes trace, OpenRouter geography, and
SkyLB/SkyWalker now provide source-bounded production parameters. Azure OpenAI /
BurstGPT adds 10.31M requests over 213 days with daily/weekly periodicity, token tails,
separated failures, and multi-duration burst examples; Splitwise adds one-day Conversation
and Coding traces; Huawei Atlas 900 A3 adds a vendor reference topology rather than a
deployment receipt. Confidence intervals, geography,
retries, speculative acceptance, session/tool-call distributions, and comparable
denominators remain explicit gaps. It is broad, but it is not entity-complete. The
requirement-level verdict is in [`coverage-audit.md`](coverage-audit.md); the detailed
missing slices remain machine-readable under `coverage.explicit_gaps` in `index.json`.

Immediate next slices:

1. finish the checked/unchecked [`slices/frontier-lab-census.md`](slices/frontier-lab-census.md), including Chinese and regional labs;
2. extend [`workload-parameters.md`](workload-parameters.md) and [`geography-session-locality.md`](geography-session-locality.md) with direct provider tenant/timezone/session distributions;
3. extend the [`filings-ledger.md`](filings-ledger.md) beyond the four largest U.S. hyperscalers;
4. extend the [`supply-chain-ledger.md`](supply-chain-ledger.md) into a named site/vendor delivery census;
5. expand the [`market-chronology.md`](market-chronology.md) with startup failures, cancellations, and resolved rumors;
6. add committed schema/link validation and scheduled refresh after the ledger schema stabilizes.

The clearest workload-distribution evidence so far is not a universal law. It is a
set of **category-dependent, heavy-tailed, bursty, multimodal, nonstationary** behaviors: model popularity
and user-model affinity evolve over months; KV-prefix reuse is skewed but differs between
consumer and API traffic; coding-agent tool calls are heavily tailed; output lengths are
strongly right-skewed even for identical prompts; request-pod cache affinity can be
roughly bimodal; daily and weekly periodicity coexists with short and sustained bursts;
and traffic changes abruptly after model releases. When a benchmark uses a Zipf, Poisson,
lognormal, Pareto, Hawkes, or MMPP assumption, it must name the fitted population
or label the distribution as synthetic.

## Validation

```bash
python3 -m json.tool docs/research/frontier-infrastructure/index.json >/dev/null
python3 - <<'PY'
import json
from pathlib import Path
p = Path('docs/research/frontier-infrastructure/index.json')
d = json.loads(p.read_text())
required = set(d['required_entry_fields'])
ids = set()
for i, row in enumerate(d['entries']):
    missing = required - row.keys()
    assert not missing, (i, sorted(missing))
    assert row['id'] not in ids, row['id']
    ids.add(row['id'])
    assert row['source_url'].startswith('https://')
    assert row['evidence_class'] in d['evidence_classes']
    assert row['confidence'] in d['confidence_levels']
assert d['coverage']['entry_count'] == len(d['entries'])
print(len(d['entries']), 'valid entries')
PY
```
