# Frontier infrastructure and workload expectations index

**Status:** initial end-to-end spine, incomplete by design. **As of:** 2026-08-26. **Tracker:** #9269.

This index records what frontier labs, hyperscalers, AI clouds, datacenter operators,
accelerator vendors, serving-system builders, researchers, reporters, and market actors
say or measure about the infrastructure and workloads behind frontier AI. It exists to
stop fak from optimizing against an invented cluster, a stationary synthetic workload,
or an unlabeled market rumor.

The machine-readable source of truth is [`index.json`](index.json). The derived
[`workload-assumptions.md`](workload-assumptions.md) registry converts the ledger into
benchmark assumptions and explicitly bounds Zipfian, batching, cluster, and userbase
claims. [`startups-landscape.md`](startups-landscape.md) is the adjacent-provider taxonomy
and watchlist. [`refresh-protocol.md`](refresh-protocol.md) defines the online-search,
contradiction, and rumor lifecycle. The `slices/` directory is reserved for independently
researched coverage ledgers.
Empty or missing slices are coverage debt, not evidence that a category has no activity.

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

The current spine contains **89 dated entries** across frontier labs, hyperscalers,
AI clouds, accelerators, serving systems, measured workload traces, workload models,
physical capacity, supply chains, standards, market signals, and one explicitly labeled
rumor. It is **not exhaustive yet**. The authoritative missing-work list is
`coverage.explicit_gaps` in `index.json`.

Immediate next slices:

1. frontier-lab entity census, including Chinese and regional labs;
2. hyperscaler/neocloud product and filing ledger;
3. datacenter/power/cooling/network/supply-chain ledger;
4. production workload-distribution table with exact parameters and sample limits;
5. startup, release, acquisition, failure, and rumor chronology;
6. contradiction matrix and a schema/link validator.

The clearest workload-distribution evidence so far is not a universal Zipf law. It is a
set of **heavy-tailed, category-dependent, nonstationary** behaviors: model popularity
and user-model affinity evolve over months; KV-prefix reuse is skewed but differs between
consumer and API traffic; coding-agent tool calls are heavily tailed; output lengths are
strongly right-skewed even for identical prompts; request-pod cache affinity can be
roughly bimodal; and traffic changes abruptly after model releases. When a benchmark uses
a Zipf, Poisson, lognormal, Pareto, or MMPP assumption, it must name the fitted population
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
