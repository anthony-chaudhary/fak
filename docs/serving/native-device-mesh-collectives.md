---
title: "Native device mesh and collective seam for TP and EP"
description: "Design-only R3+ scope gate for fak's native DP x TP x PP x EP device mesh: process groups, rank/world-size, CollectiveBackend primitives, CPU-ref single-rank behavior, and the dependency chain from native TP to MoE expert parallelism."
---

# Native device mesh and collective seam

This is the design contract for issue #25: a shared device-mesh and collective-comms
substrate under `compute.Backend`, so native tensor parallelism (TP) and MoE expert
parallelism (EP) do not grow two incompatible communication layers.

**Scope:** design only. No NCCL/RCCL binding, no multi-device CUDA change, and no MoE
expert sharding lands here. This is R3+ scope-gated. Until the native communicator exists,
fak rides external TP/EP through Track A (vLLM/SGLang/Dynamo workers) and does not chase raw
single-GPU throughput parity with vLLM.

## Ground truth

| Claim | Status | Pointer |
|---|---|---|
| Base `compute.Backend` whole-op surface is still the forward-loop target (`MatMul`, `BatchedMatMul`, `RMSNorm`, `RoPE`, `SwiGLU`, `Attention`, `Argmax`). | [SHIPPED] | `internal/compute/compute.go:318-344` |
| The tensor collective seam exists as the optional `compute.CollectiveBackend` interface with `AllReduceSum`, `AllGather`, `ReduceScatter`, and `AllToAll`. | [SEAM-ONLY] | `internal/compute/compute.go:346-408` |
| CPU reference collectives are single-box, rank-order exact, fail closed on malformed parts, and the single-rank case is identity. | [SHIPPED] | `internal/compute/collective.go:97-179` |
| Cross-process host-f32 collectives exist through `model.DistComm`, but they are not a device/NCCL communicator. | [PARTIAL] | `internal/model/dist_collective.go`, `docs/serving/multi-node-compute.md` |
| CUDA is single-device by construction today. | [GAP] | `internal/compute/cuda_kernels.cu:52-57` (`cudaSetDevice(0)`) |
| MoE routing and selected experts run in-process: router top-k, then per-expert SwiGLU, then weighted accumulation. | [PARTIAL] | `internal/model/moe.go:258-289`, `internal/model/moe.go:342-362` |
| Native TP explicitly refuses MoE/GLM-MoE-DSA instead of mis-serving those decompositions. | [GAP] | `internal/model/tensor_parallel_forward.go:79-82` |

Line numbers drift; re-anchor with:

```bash
rg -n 'type Backend interface|type CollectiveBackend interface|func \(c \*cpuBackend\) AllToAll' internal/compute
rg -n 'cudaSetDevice|func route\(|type moeFFN|ForwardTP does not yet shard MoE' internal
```

## Collective seam

`compute.CollectiveBackend` is the additive method surface under `compute.Backend`.
Backends that do not implement it remain valid single-device backends. Backends that do
implement it are bound to one communicator/process group, so tensors reduced by that backend
are rank-local shards in the same group.

The process-group layer owns:

- `rank`: the current process's rank within the group.
- `world_size`: the number of ranks in the group.
- `local_rank` / `device_id`: the rank's device within a node.
- `group_name`: the semantic group (`tp`, `ep`, `pp`, `dp`, or a composed subgroup).
- `mesh_coord`: the rank coordinate in the DP x TP x PP x EP mesh.
- `backend`: the `compute.CollectiveBackend` implementation for the rank's resident tensors.

A future concrete shape can be a runtime object rather than fields on base `Backend`:

```go
type CollectiveGroup struct {
	Name      string
	Rank      int
	WorldSize int
	LocalRank int
	DeviceID  int
	Coord     MeshCoord
	Backend   compute.CollectiveBackend
}
```

The four primitives stay exactly the HAL primitives already named:

- `AllReduceSum`: row-parallel partial sum; equal-length parts; rank-order reference.
- `AllGather`: column-parallel output concatenation; rank-ordered shards.
- `ReduceScatter`: sequence-parallel activation reduction while keeping only each rank's
  shard; must satisfy `AllReduceSum == AllGather(ReduceScatter(parts))`.
- `AllToAll`: layout transpose; the primitive EP needs for expert dispatch/combine and TP
  needs for sequence/head layout changes; must be an involution.

The CPU reference group is the degenerate group: `rank=0`, `world_size=1`, `local_rank=0`,
`device_id=0`. All four collectives are identity for one rank, preserving the bit-exact
single-device path. A real NCCL/RCCL backend is correct only if its single-rank behavior and
rank-order conformance match the CPU reference before any throughput claim is made.

## Device mesh

The native mesh coordinate is:

```text
rank := (dp, pp, tp, ep)
world_size := DP * PP * TP * EP
```

The planner derives process groups from fixed coordinates:

| Group | Varies | Holds fixed | Communication |
|---|---|---|---|
| DP | `dp` | `pp,tp,ep` | Inference replicas and placement/routing; no per-token collective by default. |
| PP | `pp` | `dp,tp,ep` | Point-to-point stage handoff through `StageTransport`; deliberately not a collective. |
| TP | `tp` | `dp,pp,ep` | `AllGather`, `AllReduceSum`, and later `ReduceScatter` inside dense/attention blocks. |
| EP | `ep` | `dp,pp,tp` | `AllToAll` dispatch to expert owners and `AllToAll`/local reduce combine back to token order. |

Topology is part of placement, not an afterthought:

- TP and EP are innermost axes and should stay inside an NVLink/NVSwitch island where possible,
  because they run per layer and can require all-reduce or all-to-all on the critical path.
- PP is outermost across nodes/fabric. It moves activations between layer bands and can tolerate
  the higher latency of TCP/RDMA/UCX/NCCL point-to-point better than TP/EP collectives can.
- DP spans replicas and failure domains. It is a gateway/router concern first; any weight-sync or
  metric sync is out of the decode hot path.
- Cross-node EP is a last resort. If the expert set does not fit inside one island, the planner
  must price the inter-node all-to-all explicitly rather than hiding it behind the same label.

## Dependency chain

1. **Collective seam / device mesh** - multi-engineer-month. Land communicator binding,
   rank/world-size, mesh placement, and a device `CollectiveBackend` over NCCL/RCCL, witnessed
   against CPU-ref. This issue (#25) is the design gate.
2. **Native TP** - multi-engineer-month. Consume the seam for Megatron-style column/row sharding
   over real device tensors. The host and CPU-ref decomposition exists; the missing rung is the
   real communicator, per-rank device binding, and measured multi-GPU run.
3. **EP-for-MoE** - multi-engineer-month after TP. Add expert ownership and load-aware routing on
   top of the same mesh. The EP delta over TP is `AllToAll` token dispatch to expert owners,
   per-owner expert execution, and dispatch-combine back to token order. It does not mint a
   second collective substrate.

## Coordination notes

The migrated #25 body cites old internal tracker IDs `#274` and `#492`. On live GitHub those
numbers resolve to unrelated issues. The corrected live coordination targets are:

- Native TP: live GitHub #295 documents multi-GPU tensor parallelism; #25 is its lower
  communicator/device-mesh design gate, not a duplicate TP issue.
- GLM-DSA / MoE backend consumer: live GitHub #86 is the GLM-DSA `compute.Backend` consumer.
  EP-for-MoE remains downstream of #25 because it needs the shared `AllToAll` substrate.

Do not post coordination comments to live GitHub #274 or #492 for this topic; doing so would
attach TP/EP design notes to unrelated tickets. If those internal tracker IDs are ever imported
with their original meanings, this doc should be cited there.

## Non-goals

- No NCCL/RCCL implementation in this issue.
- No CUDA multi-device selection change in this issue.
- No MoE expert placement or all-to-all dispatch implementation in this issue.
- No new duplicate TP or EP issue minted from this design.
- No raw single-GPU throughput parity claim.
