# Long-context topology and economics model — 2026-08-28

This note documents the deterministic topology spine added for #9576. It composes supplied bounds; it is not a benchmark report and contains no inferred measurements.

## Scope and modes

`internal/modelperfobs/topology.go` distinguishes:

- **independent-agent replication**: one weight copy per node; jobs can run independently;
- **shared-prefix routing**: one weight copy per node, with caller-supplied KV/state movement for routed or migrated state; and
- **model-parallel/sharded execution**: one logical weight copy partitioned across nodes, one sharded job at a time in this deliberately conservative model.

Supported topology inputs are exactly 1, 2, or 4 nodes and 10, 25, or 100 GbE. Model execution remains fak-native. llama.cpp may be used only as a separately identified external parity/reference baseline, never as fallback execution or evidence that fak executed a job.

## Assumptions and formulas

All numeric uncertainty uses inclusive `[min,max]` bounds. Range arithmetic preserves conservative endpoints.

For replicas and shared-prefix routing:

```text
weight copies = nodes
per-node residency = weights + resident KV / nodes
aggregate physical residency = weights * nodes + resident KV
job waves = jobs / min(nodes, jobs)
```

For model parallelism:

```text
weight copies = 1
per-node residency = (weights + resident KV) / nodes
job waves = jobs
productive time = jobs * single-host job time / supplied shard speedup
```

The equal KV/shard partition is an explicit idealized lower-level assumption. Callers must widen bounds or reject the result where placement is uneven. Crucially, fit is checked against **per-node** memory. Aggregate physical memory is reported for capacity/cost visibility but is never treated as a single allocatable pool.

For every mode:

```text
productive time /= (1 - idle-capacity fraction)
productive time *= (1 + scheduler-imbalance fraction)
network bytes = KV/state movement + weight transfer
wire bytes/s = Ethernet Gb/s * 1e9 / 8 * network efficiency
serialization time = network bytes / wire bytes/s
total time = productive + serialization + synchronization
           + setup/recovery + replay/compaction
total cost = total time * cluster cost/s
```

The link rate is not multiplied by node count. This is a bottleneck-link serialization bound, preventing the common but invalid assumption that all traffic consumes the sum of every NIC's bandwidth. Callers represent overlap, collective behavior, protocol loss, and fabric contention through evidence-backed efficiency and time bounds; absent such evidence, they should use broad bounds rather than point estimates.

Duplicated weights are represented by `WeightCopies` and aggregate residency. Weight loading/recovery traffic is explicit in `WeightTransferBytes`; KV migration, shared-prefix state, or shard exchange is explicit in `KVStateMovementBytes`. Synchronization, scheduler imbalance, setup/recovery, replay/compaction, and idle capacity are separate inputs so none can disappear into an unexplained throughput multiplier.

## Completed-job economics and quality

Comparisons are against two caller-described alternatives: one larger host and API bursting. Each alternative supplies completed jobs, total elapsed time, total cost, and quality qualification. The topology supplies the same quantities. Cost and time are normalized only as:

```text
cost per qualified completed job = total cost / qualified completed jobs
time per qualified completed job = total time / qualified completed jobs
```

A `qualified` result means an external, task-specific quality gate passed. `failed` and `unknown` quality produce explicit unknown qualified-job metrics and no cheaper/faster verdict. This prevents failed output volume from being valued as useful throughput. A strict break-even Boolean is emitted only when the topology's worst bound is below the alternative's best bound; overlapping bounds remain a non-proof, not a benchmark claim.

## Limitations

- No hardware, model, protocol, topology, kernel, or workload measurements are embedded.
- The estimator does not predict shard speedup. That bound must come from identified evidence; otherwise it must remain broad.
- Equal partitioning does not model layer placement, hot spots, replication for fault tolerance, switch oversubscription, collectives, NUMA, or communication/compute overlap.
- Shared-prefix savings are not presumed. Callers must provide actual reduced work to the underlying long-context envelope and actual movement bytes here.
- Queue distributions, failures, retry probability, tail latency, energy, depreciation, API minimum charges, and egress are not synthesized. Include them in supplied bounds or report them unknown.
- A larger host or API comparison with unknown cost, completion count, time, or quality must not be converted into a point break-even claim.

## Consumption by #9578

#9578 should treat this model as the topology/economics composition layer above `EstimateLongContextEnvelope`:

1. Build the per-job fak-native memory and service-time envelope from identified model and host inputs.
2. Select one topology mode explicitly; do not silently switch engines or modes.
3. Translate that envelope into bounded weights, resident KV, and single-node job time.
4. Supply evidence-backed movement, efficiency, shard speedup, overhead, idle-capacity, and cost intervals. Preserve unknowns instead of substituting benchmark-looking constants.
5. Run the same task-specific quality gate for fak-native topology, larger-host, and API jobs.
6. Compare qualified completed-job cost and time, retaining interval overlap and unknown qualification in receipts.
7. Record engine identity and evidence provenance. llama.cpp results, if collected, are external parity/reference rows only.

The consuming benchmark must retain raw receipts and must not present this analytical estimate as observed performance.
