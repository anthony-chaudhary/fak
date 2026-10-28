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
```

All four verbs write JSON; `--json` is accepted. `--input -` reads stdin. Input is limited to 4 MiB;
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

Pool IDs are physical accounting identities. Multiple ranks drawing from the same physical UMA
memory must reference the same pool so their demands are summed. A host pool can model explicit
host residency, but this command supplies no host-offload implementation. Shards are uniform byte
accounting units, rounded upward per rank; they do not prove tensor or layer boundaries can split.
Peer access, links, synchronization, heterogeneous backend compatibility, compute support and speed
are not simulated. Do not infer a usable multi-GPU configuration from memory arithmetic alone.
