---
title: "Compute-placement tax"
description: "Compute-placement tax is the preferred name for the incremental cost caused by placing a fixed compute graph and its state across coherence, device, host, or network boundaries. “Placement."
---

# Compute-placement tax

**Compute-placement tax** is the preferred name for the incremental cost caused by placing a fixed compute graph and its state across coherence, device, host, or network boundaries. “Placement loss,” “distribution overhead,” and “parallelism overhead” remain search aliases, but *tax* is more accurate: distribution can still win when extra useful compute or capacity exceeds the tax.

Every comparison fixes one workload envelope: model/graph, state bytes, useful work, quality, precision, batch/concurrency, and SLO. Feasibility is reported before speed. A machine that cannot hold the required state is not a slow reference; it is an infeasible counterfactual, so no speedup or penalty ratio is emitted.

## Ledger and equations

For each transfer, the alpha-beta estimate is:

`Traw = messages * link_latency + bytes / effective_bandwidth`

Only critical-path communication counts as tax:

`Texposed = max(0, Traw - min(Traw, overlap_with_useful_compute))`

The minimal spine then reports an inspectable critical-path ledger:

`Ttotal = Tuseful + Texposed + Tsync + Tinput_imbalance + Treplication_or_migration + Tremote_memory_or_paging + Torchestration_or_recovery`

It preserves latency, useful-work throughput, monetary cost, energy, capacity headroom, and SLO independently. For a feasible candidate `c` and explicit feasible reference `r`, it reports candidate-minus-reference deltas, dimension-wise ratios, and latency placement efficiency `T_r / T_c`. It does not collapse unlike dimensions into a magic scalar.

This first spine adds component times. A calibration may replace that with a witnessed dependency DAG/critical path, but must not double-count hidden communication. Bandwidth means achieved bandwidth for the traffic pattern, not a link's marketing line rate. Latency includes software launch and protocol latency when measured end to end.

## Parallelism mappings

| Placement strategy | State/work split | Typical tax entries |
|---|---|---|
| Tensor parallel | shard tensor dimensions within each layer | per-layer all-reduce/reduce-scatter/all-gather, launch latency, stragglers |
| Pipeline parallel | shard consecutive layers/stages | activation transfers, bubbles, stage imbalance, microbatch synchronization |
| Expert parallel | place MoE experts on domains | all-to-all dispatch/combine, skew, expert replication and overflow |
| Data parallel | replicate parameters; split independent requests/batches | parameter/gradient synchronization, replication, fan-in; inference may need no per-token collective |
| Sequence/context parallel | shard tokens or attention context | KV/activation exchange, collectives, synchronization and remote-memory traffic |

Hybrid plans simply contribute multiple components to the same ledger. Topology labels are insufficient: a two-host tensor-parallel plan over Ethernet and a two-device plan over a coherent fabric have different links, overlap, failure, and capacity receipts.

## Example: one larger coherent host versus two hosts

Compare a host with the same processor but twice the unified-memory capacity against two smaller identical hosts only under the same model, precision, quality, batch and useful work. The larger host pays no cross-host collective but may expose less aggregate compute or memory bandwidth. The two-host placement may fit through sharding and offer more compute, while paying finite-link latency/bandwidth, synchronization, imbalance, replication, orchestration and recovery. The model intentionally permits either result and identifies the break-even envelope instead of assuming aggregate bytes or FLOPS imply equivalence.

## Provenance and calibration

Each placement and link is `estimated` or `measured`. Analytical estimates are decision aids, never hardware speedup claims. A measured receipt should name hardware/firmware, topology and direction, message-size distribution, achieved bandwidth/latency, model and precision, batch/concurrency, warmup, repetitions, quality check, engine, failures/retries, and raw artifact. Native-performance evidence must name the fak-native engine; another runtime is reference/benchmark evidence only.

Useful prior art to borrow, not silently substitute at runtime:

- Hockney/alpha-beta collective cost model and [LogP](https://doi.org/10.1145/173284.155333) latency/overhead/gap model.
- [Roofline](https://doi.org/10.1145/1498765.1498785) and communication-roofline reasoning for compute versus bandwidth limits.
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) tensor/pipeline/expert parallel implementations.
- [Alpa](https://arxiv.org/abs/2201.12023), [FlexFlow](https://arxiv.org/abs/1802.04799), [Unity](https://arxiv.org/abs/2205.11938), [DistIR](https://arxiv.org/abs/2111.05426), and [GSPMD](https://arxiv.org/abs/2105.04663) for automated placement/cost-model ideas.
- [vLLM distributed serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html) and [SGLang distributed inference](https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html) as field references, never implicit fallbacks.

The next integration step is to populate this model from default performance-analysis receipts and require tuning proposals that cross a memory/device/host boundary to include the resulting feasibility and tax ledger.
