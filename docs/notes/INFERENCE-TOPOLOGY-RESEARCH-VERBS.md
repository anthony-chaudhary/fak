# Inference topology resource research verbs

`fak-dev inference-topology` is a reproducible, offline decode-step resource model and
evidence ledger reader. It does not run a model or measure hardware. Its output always
carries `ANALYTICAL_NOT_MEASURED`. Source review dates support later re-inventory;
inventory never silently fetches or revises evidence.

## Commands and ownership

```text
fak-dev inference-topology validate --manifest manifest.json [--json]
fak-dev inference-topology analyze --manifest manifest.json --out analysis.json
fak-dev inference-topology inventory --manifest manifest.json --at 2026-10-22 --out inventory.json
fak-dev inference-topology render --manifest manifest.json --prose argument.md --out report.md
```

All commands accept `--out`; without it they write stdout. Analyze and inventory emit
JSON. `--json` selects JSON validation output; render refuses that flag. Render includes the supplied prose, a generated resource table, scenario assumptions,
claim/source ledgers and limitations. Inventory defaults to the manifest's `as_of`, so
reproduction has no implicit wall-clock input. Evidence is due after its `review_by` date.
A stale source propagates to dependent claims and scenarios. No command makes network calls.

The pure model is `internal/inferencetopology`; the CLI adapter is
`internal/devcmd/inference_topology.go`. Selection is the `fak-dev` dispatch case.
Removing the feature removes that case, the leaf, adapter, catalog/ownership registration
and this document. Manifests are caller-owned JSON and require no state migration.
There are no serving, scheduler or engine dependencies.

## Reuse and scope

This extends the repository's developer tooling pattern: strict typed manifests,
explicit units, bounded reads, stable evidence classes and atomic output files.
`internal/modelperfobs/topology.go` already handles topology economics and completed-job
comparisons for 1/2/4 nodes; its independent-replica, prefix-routing and sharded modes
remain unchanged. See [the economics model](LONG-CONTEXT-TOPOLOGY-MODEL-2026-08-28.md).
This leaf instead exposes per-rank memory, HBM, FLOP and collective constraints.
It does not allocate clusters, choose commercial policy, collect live measurements,
or duplicate operational bandwidth audit stores.

For users trying to choose a GPU partition, the problem is that resident weight capacity,
per-step traffic, collective latency and service throughput are different quantities.
Today a GPU count or aggregate GB/s is often used as a speed proxy. The result is better
because it makes partitioning and transferred bytes explicit and rejects incompatible
inputs. Witness: the real CLI commands above plus
`go test ./internal/inferencetopology`.
Centrality: Enabling, serving priority 2. P1–P4: exposes a concrete serving bottleneck,
uses the existing developer entry point, preserves native execution, and captures
reproducible evidence without inventing a measured speedup.

## Manifest contract

Schema: `fak-inference-topology/1`. JSON is strict (unknown fields, trailing values,
non-finite/negative numbers, invalid references/dates and inputs over 16 MiB are refused).
All floats below use SI bytes, seconds and FLOPs; integers are counts.

Top-level fields:

| Field | Meaning |
|---|---|
| `id`, `as_of` | Nonempty identity and ISO `YYYY-MM-DD` evidence cutoff |
| `sources` | Nonempty source array, unique IDs |
| `claims` | Claim array with explicit dispositions and references |
| `scenarios` | Nonempty scenario array, unique IDs |

Each source has `id`, `url` (HTTP/S), `title`, `accessed`, `review_by`, `kind`, and optional
`revision`. Kinds: `primary`, `vendor`, `oss`, `secondary`, `social`, `assumption`.
`accessed <= as_of` and `review_by >= accessed` are required. An assumption URL identifies
where its rationale lives; it does not turn an assumed rate into a measured rate.

Each claim has `id`, `text`, `status`, `source_ids`, `review_by`, and optional
`limitations`. Status is `supported`, `disputed`, `open`, or `derived`.
References must resolve to unique source IDs. These labels are supplied by the author,
not an automatic truth judgement.

Each scenario has `id`, `label`, `mode`, `source_ids`, nonempty `assumptions`, `model`,
`hardware`, and optional `per_rank`. `model` fields:

| Field | Units / interpretation |
|---|---|
| `dense_parameters` | Always-active parameters, including shared weights in MoE |
| `moe_layers`, `experts_per_layer`, `active_experts_per_token` | Counts; MoE layers must not exceed total layers |
| `parameters_per_expert` | Parameters per expert **per MoE layer** |
| `bytes_per_weight` | Effective storage bytes/parameter including quantization metadata |
| `kv_bytes_per_token` | Logical KV bytes per context token across all layers for one sequence |
| `batch_size`, `context_tokens` | Decode tokens/step and existing context length/sequence |
| `hidden_size`, `layers`, `bytes_per_activation` | Hidden-vector shape and effective activation bytes/element |
| `additional_flops_per_step` | Whole-group attention and other FLOPs beyond parameter matmuls |
| `activation_traffic_bytes_per_step` | Whole-group HBM activation traffic beyond weights/KV |

`hardware` fields:

| Field | Units / interpretation |
|---|---|
| `gpus`, `tp_ranks`, `kv_shards` | Participating GPUs, one TP subgroup, KV partition count; counts must divide consistently |
| `memory_bytes_per_gpu`, `reserved_bytes_per_gpu` | Physical capacity and caller budget for workspaces, activation buffers, fragmentation, other users |
| `hbm_bytes_per_second_per_gpu`, `flops_per_second_per_gpu` | Supplied resource rates; identify peak versus sustained in scenario assumptions |
| `link_bytes_per_second_per_rank_one_way` | Per-rank bottleneck one-way rate, never aggregate full-duplex link marketing bandwidth |
| `ring_alpha_seconds`, `allreduces_per_step` | Per-ring-phase latency and internally dependent all-reduce count |
| `collective_payload_bytes` | Optional payload/all-reduce; zero defaults to batch × hidden × activation bytes |
| `extra_network_bytes_per_rank_per_step` | Additional bottleneck-link bytes, e.g. explicit EP dispatch/combine or amortized KV migration |
| `extra_network_events_per_step`, `extra_network_latency_seconds_per_event` | Internally serialized extra-network events and per-event latency |

`tensor_parallel` requires `gpus == tp_ranks` and hidden size divisible by TP ranks.
It models one balanced group. Extra GPUs do not create a fictitious shared capacity pool.
KV sharding is independently explicit, because GQA/MLA layouts may replicate KV.
Hidden divisibility is a minimal condition; attention-head, tile and expert divisibility
still need a real implementation witness.

`explicit_per_rank` requires `per_rank` with all numeric fields below. It replaces all
derived partitioning; it supports a caller-derived EP/DP or irregular layout without
assuming its speedup. Inputs must describe jointly realizable work on the physical
critical rank, including peak imbalance. Average rank load is insufficient.

```text
resident_weight_bytes
weight_stream_bytes_per_step
kv_resident_bytes
kv_read_bytes_per_step
kv_write_bytes_per_step
activation_traffic_bytes_per_step
flops_per_step
```

## Resource equations and saturation interpretation

For batch `B`, experts/layer `E`, selected experts/token `k`, MoE layers `L`,
parameters/expert `Pe`, shared/dense parameters `Pd`, weight bytes/parameter `q`:

```text
expected unique experts/layer U = E × (1 − (1 − k/E)^B)
logical resident weight bytes = q × (Pd + L × E × Pe)
whole-group weight stream bytes = q × (Pd + L × U × Pe)
whole-group matmul FLOPs/step = 2 × B × (Pd + L × k × Pe)
```

The union assumes independent uniform top-k routing and ideal reuse of selected expert
weights across the batch. Repeated experts reduce streamed weight bytes, not logical
resident capacity. Real skew can reduce the union while worsening peak expert load;
padding and tile shapes can increase both traffic and compute. No expert parallel
scaling is inferred from this union. `uniform_format_logical_weight_bytes` is metadata
computed with the scalar `bytes_per_weight`; it is not authoritative mixed-format
residency in explicit mode. The explicit per-rank ledger is authoritative there.

In TP mode weight/compute/activation traffic divides by TP ranks. Existing KV read is
`B × context × kv_bytes_per_token / kv_shards`; append writes are
`B × kv_bytes_per_token / kv_shards`. Peak KV capacity includes the appended token.

```text
HBM time = (weight reads + KV reads + KV writes + activation traffic) / per-GPU HBM rate
compute time = per-rank FLOPs / per-GPU FLOP rate
ring time = C × [2(n−1)α + 2(n−1)/n × payload / one-way per-rank link rate]
extra network time = events × latency + bytes / one-way bottleneck link rate
ideal-overlap lower bound = max(HBM, compute, ring, extra network)
serial resource sum = HBM + compute + ring + extra network
```

The ring is illustrative, not a universal NCCL/NVSwitch algorithm. Its collectives and
extra-network events are internally dependent; overlap between those two paths is
allowed in the lower bound. Shared HBM/network resources can make that bound loose.
The serial sum is **not an upper bound** on actual execution; layer dependencies, launches, synchronization, imbalance and omitted work can increase runtime.

The analysis reports arithmetic intensity (FLOPs/HBM byte; omitted with a warning when
HBM traffic is zero), machine balance
(FLOPs/s divided by HBM bytes/s), and TP-mode KV-read/weight-read equality context
length at the supplied batch. Dense-only weight-compute crossover batch is
`q × machine_balance / 2`, excluding KV, activation traffic, attention, communication
and efficiency effects. It is a search-space marker, not the serving saturation batch.

No request-rate saturation is predicted: SLOs, arrival distributions, scheduler policy,
prefill interference, variable contexts, queueing and useful output quality require
external measurements. Capacity-infeasible rows retain arithmetic diagnostics and an
early warning, never an executable latency prediction.

## Acceptance and reconnaissance record

Native spine: `inference-topology-spine-20261022`. Before build: queried public and
companion feature catalogs, existing topology economics and study notes, placement,
lessons and blast radius. Reuse decision: extend developer tooling with a pure
resource leaf; preserve economic and operational bandwidth readers.

Atomic steps: strict schema/resource model (leaf tests); developer CLI and bounded
read/atomic write adapter (real CLI smoke); generated ownership/catalog registration
(ownership tests); external manifests and report prose consume the verified verbs.
No physical accelerator result is claimed by this acceptance witness.
