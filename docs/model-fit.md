# Declared multi-device model fit

`fak model fit` evaluates a fixed model's declared memory placement. It performs no model loading,
device probing, layer movement, token generation, or hardware qualification. Evaluation artifacts
(`plan`, `check`, `compare`) carry
`modeled_only: true` and `qualification: "unverified"`. `eligible` means that declared capacity fits and
the caller supplied a supported-runtime declaration with evidence; it does not prove execution.

```text
fak model fit compose --input unassigned.json
fak model fit plan --input placement.json
fak model fit check --input placement.json
fak model fit compare --input alternatives.json
fak model fit bands --input placement.json --manifest shards.json --manifest-sha256 SHA256
```

`bands` authenticates a GGUF shard manifest using the trusted SHA-256 of the canonical
`json.Marshal` of its ordered `ShardIdentity` array (absolute path, size, and raw-file
SHA-256 for each shard), then checks the loader's logical storage census against the
existing composed declaration. Each accounting shard represents one layer; rank order and pool pins
are preserved. Every band must fit its declared sharded charge, and first/last rank extra
bytes must cover embedding/head. Replicated weight charges, zero-layer ranks, repeated
pools, illegal hybrid cuts and unsupported census routes are refused. The result remains
modeled and unverified; it does not validate KV, workspace, physical allocations, loader
placement, or execution. Keep authenticated files immutable during the operation.
The result includes the complete declaration and loader profile. Profile
`q4k-dense-kquant-no-native-rows-no-mtp-no-packed-embedding/1` passes the explicit
Q4K loader options: dense k-quant on; MTP, native linear rows, native SSM output,
Q2/Q4 packed embedding and tied Q6 packed embedding off. Streaming and expert
sharding are not selected. The estimate rejects the W3 selection; it does not
establish that a future loader uses this profile.

All five verbs write JSON; `--json` is accepted. `--input -` reads stdin. Input is limited to 4 MiB;
unknown fields, extra JSON values, and omitted accounting fields are rejected. Specify every pool's
`usable_bytes`, all three weight fields, and every rank's fields below. Use explicit zero for a cost
that does not apply. Capacity zero means unknown. Usable capacity must already exclude headroom,
other applications, and reservations. Supply bytes as integers.

`plan` exits 0 for a valid evaluated declaration, including a refusal. `check` exits 1 unless its one
report is eligible. Malformed declarations exit 2; output failures exit 1. A refused or unknown plan
remains inspectable as JSON with reasons and per-rank/per-pool totals.

`compose` assigns uniform weight shards in input rank order within declared pool budgets, after
charging every rank's fixed replicated/KV/workspace/extra cost. Give every rank `shards: 0`; nonzero
pins are rejected. A rank may receive zero shards; its fixed costs remain charged. It writes a raw
placement declaration, without an artifact wrapper, directly
consumable by `check`. Unknown or insufficient capacity exits 1 with a closed reason and no plan.
Support remains unchanged: a memory assignment with unknown support still fails `check`.

```text
fak model fit compose --input unassigned.json | fak model fit check --input -
```

The pipeline's final exit code describes `check`; inspect both processes' exit codes when diagnosing
a composition failure. Save and review the composed JSON before using it elsewhere.

This example uses two unequal pools: 16 GiB and 32 GiB. A fixed model has 8 GiB replicated on every
rank and 24 GiB split into eight uniform accounting shards. Assigning two shards to the smaller GPU
and six to the larger charges 16 GiB and 28 GiB after 1 GiB each of KV and workspace per rank.

```json
{
  "pools": [
    {"id": "gpu-small", "usable_bytes": 17179869184},
    {"id": "gpu-large", "usable_bytes": 34359738368}
  ],
  "placement": {
    "name": "asymmetric",
    "support": "unknown",
    "support_evidence": "",
    "weights": {"replicated_bytes": 8589934592, "sharded_bytes": 25769803776, "shard_count": 8},
    "ranks": [
      {"rank": 0, "pool_id": "gpu-small", "shards": 2, "kv_bytes": 1073741824, "workspace_bytes": 1073741824, "extra_bytes": 0},
      {"rank": 1, "pool_id": "gpu-large", "shards": 6, "kv_bytes": 1073741824, "workspace_bytes": 1073741824, "extra_bytes": 0}
    ]
  }
}
```

This declaration has fitting capacity but unknown execution support, so `check` exits 1. Change
both rank `shards` values to zero and save as `unassigned.json` to let `compose` derive the two/six
assignment automatically. Change
support to `supported` only with caller evidence covering the exact model encoding, backend,
partition scheme and required transfers. `unsupported` produces an unavailable report. These fields
are declarations, not inspected or independently verified evidence.

For comparison, replace `placement` with `placements`, an array containing the asymmetric placement
and an alternative named `equal` whose rank shard counts are four and four. The equal alternative
charges 22 GiB per rank and exceeds the smaller pool even though aggregate capacity is sufficient.
Every alternative must use identical `weights`; changing encoding or model sizes is rejected.
Results preserve input order and select no optimum. Keep context, batch, KV precision and workload
assumptions equivalent yourself; per-rank KV/workspace may differ with replication strategy.

## Next tiers on refusal

A refusal over declared pools is tiered, not absolute. When `capacity_status` is `refused`, the
report carries `next_tiers`, a best-first list of the concrete lower tiers that would admit the
placement; each step names the tier token, the `shortfall_bytes` that tier must absorb, and a
fixed-wording note. `host_offload` appears only when declared-pool arithmetic proves it viable:
enough sharded/KV/workspace bytes are movable (replicated bytes stay on-rank) and a declared host
pool — a pool no rank draws from — has at least the shortfall in spare capacity. `multi_device`
appears whenever the known pools fall short, naming the additional capacity to declare or the
larger pools to re-shard onto; over fully known pools it is always arithmetically admissible, so a
known-pool refusal always names at least one tier. When a used pool's capacity is unknown
(`usable_bytes` zero), no tier can be proven: the report carries no `next_tiers` and the closed
reason `pool_capacity_unknown` says why. Reports that are not capacity refusals omit the field
entirely (`omitempty`). `streamed_dense` is a reserved tier token pointing at the unimplemented
streamed-dense route (fak#13175); it is never emitted as an implemented placement. All of this
stays `modeled_only: true` with `qualification: "unverified"`: the tiers are declared arithmetic,
not hardware qualification.

Pool IDs are physical accounting identities. Multiple ranks drawing from the same physical UMA
memory must reference the same pool so their demands are summed. A host pool can model explicit
host residency; a refusal may enumerate `host_offload` as a next tier, but this command enumerates
tiers only and supplies no host-offload implementation. Shards are uniform byte
accounting units, rounded upward per rank; they do not prove tensor or layer boundaries can split.
Peer access, links, synchronization, heterogeneous backend compatibility, compute support and speed
are not simulated. Do not infer a usable multi-GPU configuration from memory arithmetic alone.

## Split and offload pricing (`fitplan.PlanSplit`)

The commands above check whether declared placements fit in memory; they do not price speed.
`fitplan.PlanSplit(Input, Interconnect)` is the matching pure pricer. It takes the same header facts,
calibration, tier rates and stream/active-set gates that `Plan` and `--fit auto` use (an
unknown per-token stream or MoE active set is omitted, not priced), and ranks one decode step for every
(split mode, offload mode) pair across `Interconnect.Devices` identical discrete devices.
`Hardware.PoolUsableBytes` is per device.

| Mode | Placement | Decode step |
| --- | --- | --- |
| `single` | dense core first on one device | device, then spilled tiers |
| `layer` | 1/N of dense, KV and experts per device | sum of stages + N-1 activation hops |
| `tensor` | 1/N of every layer; KV over `min(N, kv_heads)` ranks (a compressed KV is replicated) | slowest rank + two ring all-reduces per layer, each `2(N-1)` latency steps |
| `expert` | dense core and KV on device 0, routed experts across all devices | dense + slowest expert rank + two all-to-alls per layer |

Weights that spill below the devices are priced twice:

- **`cpu`** computes them in place: host-RAM bandwidth, NVMe rate for paged weights, the stated
  host FLOP rate when one is given. Host experts spread over every MoE layer, so each layer pays
  two host↔device crossings (out and back); a contiguous dense spill pays two in total. Attention
  never runs on the host, because the KV is device-resident.
- **`stream`** copies them over the host link at `min(link, host RAM)`. The copy is serial with
  compute, because experts are selected after the router. Each rank pulls its own share over its
own link: the dense spill is sharded only under `tensor`, the routed spill under `tensor` and
`expert`.

A batch reads `max(active, routed·(1-(1-k/E)^b))` routed bytes, the same rule as
`Model.StreamBytes`, so every batch agrees with `Plan`. Without expert counts, the declared
per-token active share stands in for `k/E`. These bytes are split by where the experts live. With
serial decode (`batched_decode` off), the per-sequence rate is the aggregate divided by the
stream count, as in `Plan`.

`Chosen` is the fastest unrefused row the engine can actually run (`Interconnect.Executable`). The
default, `ServeExecutableSplits()`, is what `fak serve` runs in one process today: `single`, and
`expert` (`--expert-parallel N` on a `-tags cuda,nccl` build). `layer` (multi-process transport
only) and `tensor` (a collective gate with no sharded serve forward) are priced but never chosen.
When one of them would be faster, it appears as `Best`, so the gap that engine work would close is
visible. If some row fits but no mode in the list does, the status is `unknown` with
`no_executable_split`, never `refused`. `refused` means no row fits at all. A rate the caller did not state (host link, peer latency, host sync) is filled with a
labelled generic default. Only rates a row's phases actually use are listed in its `generic_rates`,
and only a chosen (or best) row that used one demotes the report to the `empirical` tier.
`Devices` above `MaxSplitDevices` (64) is not priced: the status is `unknown` with
`split_devices_out_of_range`.

Worked case, all MODELED: a DeepSeek-V4.1-Flash-class MoE on one 16 GB card with 32 GB of host RAM
is NVMe-bound. A second card over PCIe (`expert`) holds about 15 GB more experts, a modest gain,
because NVMe still binds. Host RAM that holds every expert removes the NVMe term, a gain several
times larger (`TestSplitSecondGPUVersusRAM`). Tensor split over PCIe-class latency loses to one
device on a model that already fits, and wins over an NVLink-class link
(`TestSplitTensorLatencyCrossover`). `PlanSplit` is not yet wired into `--fit auto`, because serve
probes only device 0.
